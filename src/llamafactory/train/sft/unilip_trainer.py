"""Mixed text/image loss for Qwen-Omni + UniLIP; preserves StatefulDataLoader."""
from __future__ import annotations

import json
import os
import torch
from .trainer import CustomSeq2SeqTrainer


class UniLIPSeq2SeqTrainer(CustomSeq2SeqTrainer):
    """Saves supplemental composite weights when root model is a PeftModel."""

    EXTRA_FILENAME = "unilip_extra.safetensors"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Detached sums stay on device until the regular logging interval.
        self._unilip_loss_totals = torch.zeros(3, device=self.accelerator.device)

    def _record_losses(self, model, text_loss, image_loss=None):
        """Accumulate raw component losses once per training microbatch."""
        if not model.training:
            return
        totals = self._unilip_loss_totals
        for index, value in enumerate((text_loss, image_loss)):
            if value is not None:
                totals[index].add_(value.detach().to(totals))
        totals[2].add_(1)

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            totals = self.accelerator.reduce(self._unilip_loss_totals, reduction="sum")
            text_loss, image_loss = (totals[:2] / totals[2].clamp_min(1)).tolist()
            logs = dict(logs, text_loss=text_loss, image_loss=image_loss)
            self._unilip_loss_totals.zero_()
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        # Root PEFT saves its own Qwen adapter. Decoder/source-head weights
        # are saved separately, even if currently frozen.
        from peft import PeftModel
        if not isinstance(self.model, PeftModel):
            return
        from .unilip_training import unwrap_composite
        from safetensors.torch import save_file

        base = unwrap_composite(self.model)
        destination = output_dir or self.args.output_dir
        os.makedirs(destination, exist_ok=True)
        with open(os.path.join(destination, "unilip_config.json"), "w") as handle:
            json.dump(base.vision_decoder.spec, handle, indent=2)
        weights = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in base.named_parameters()
            if (
                (name.startswith("vision_decoder.")
                 and not name.startswith("vision_decoder.image_tokenizer.")
                 and (not name.startswith("vision_decoder.dit.")
                      or "lora_" in name
                      or base.vision_decoder.spec.get("full_dit_weights", False)))
                or "trainable_tokens_delta" in name
                or name.startswith("projectors.") or ".fusion_gate." in name
            )
        }
        # Explicitly protect the new feature projections against accidental
        # omission from an adapter checkpoint.
        head_names = {
            name for name, _ in base.named_parameters()
            if name.startswith(("vision_decoder.source_heads.", "vision_decoder.query_projector.")) or (
                base.vision_decoder.spec.get("conditioning_projection_mode","separate") == "shared"
                and name.startswith("vision_decoder.output_head."))
        }
        if missing := (head_names - set(weights)):
            raise RuntimeError(
                f"UniLIP source projections absent from checkpoint: {sorted(missing)[:8]}"
            )
        if weights:
            save_file(weights, os.path.join(destination, self.EXTRA_FILENAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, *args, **kwargs):
        result = super()._load_from_checkpoint(resume_from_checkpoint, *args, **kwargs)
        extra = os.path.join(resume_from_checkpoint, self.EXTRA_FILENAME)
        if os.path.isfile(extra):
            from safetensors.torch import load_file
            from .unilip_training import unwrap_composite
            base = unwrap_composite(self.model)
            state = load_file(extra, device="cpu")
            required = {
                name for name, _ in base.named_parameters()
                if name.startswith(("vision_decoder.source_heads.", "vision_decoder.query_projector.")) or (
                base.vision_decoder.spec.get("conditioning_projection_mode","separate") == "shared"
                and name.startswith("vision_decoder.output_head."))
            }
            if missing := (required - set(state)):
                raise RuntimeError(
                    f"UniLIP resume missing source-head tensors: {sorted(missing)[:8]}"
                )
            incompatible = base.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"UniLIP resume unexpected keys: {incompatible.unexpected_keys[:8]}"
                )
        elif os.path.isfile(os.path.join(resume_from_checkpoint, "unilip_config.json")):
            raise FileNotFoundError(
                f"Missing {self.EXTRA_FILENAME} in {resume_from_checkpoint}. "
                "A PEFT-only checkpoint cannot restore supplemental UniLIP weights."
            )
        return result

    def _image_forward_kwargs(self, model, group):
        if getattr(self.finetuning_args, 'unilip_timestep_sampling', 'uniform') == 'uniform':
            return {"target_images": group["target_images"]}
        from .unilip_training import unwrap_composite

        decoder = unwrap_composite(model).vision_decoder
        latents = decoder.encode_images(group["target_images"]).detach().float()
        schedule = (1000, 1000, 1000, 900, 750, 500, 250, 100)
        total_steps = int(decoder.spec['scheduler_config']['num_train_timesteps'])
        timestep = max(1, round(schedule[self.state.global_step % len(schedule)] * total_steps / 1000))
        # Fresh every microbatch; equal across prompts within this resolution bucket.
        noise = torch.randn_like(latents[:1]).expand_as(latents).contiguous()
        return dict(latents=latents, noise=noise,
                    timesteps=torch.full((len(latents),), timestep, device=latents.device, dtype=torch.long))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Data collator emits groups of equally-shaped image targets.
        groups = inputs.pop("unilip_image_groups", [])
        image_only = (getattr(self.finetuning_args, 'unilip_projector_only', False)
                      or getattr(self.finetuning_args, 'unilip_bridge_only', False))
        if not groups:
            if image_only:
                raise ValueError('Image-only UniLIP training requires image targets in each batch')
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            self._record_losses(model, outputs.text_loss)
            return (loss, outputs) if return_outputs else loss
        if (self.finetuning_args.use_asft_loss or self.finetuning_args.use_dft_loss
                or self.finetuning_args.use_eaft_loss):
            raise ValueError("UniLIP does not support ASFT/DFT/EAFT custom language losses yet")
        if len(groups) == 1:
            group = groups[0]
            outputs = model(
                **inputs,
                **self._image_forward_kwargs(model, group),
                image_batch_indices=group["image_batch_indices"],
            )
            if outputs.loss is None or outputs.image_loss is None:
                raise RuntimeError("UniLIP forward did not produce both language and image losses")
            if image_only:
                from .unilip_training import unwrap_composite
                outputs.loss = outputs.image_loss * float(unwrap_composite(model).vision_decoder.spec['image_loss_weight'])
            self._record_losses(model, None if image_only else outputs.text_loss, outputs.image_loss)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Different HxW buckets cannot be stacked. Compute LM loss once, then
        # one diffusion loss per image group. The text loss is not repeated.
        text_outputs = None if image_only else model(**inputs)
        if not image_only and text_outputs.loss is None:
            raise RuntimeError("Composite model returned no text loss")
        loss = torch.zeros((), device=inputs['input_ids'].device) if image_only else text_outputs.loss
        image_loss = loss.detach().new_zeros(())
        n_images = sum(int(group["target_images"].shape[0]) for group in groups)
        for group in groups:
            image_inputs = dict(inputs)
            image_inputs.pop("labels", None)
            result = model(
                **image_inputs,
                **self._image_forward_kwargs(model, group),
                image_batch_indices=group["image_batch_indices"],
            )
            if result.image_loss is None:
                raise RuntimeError("UniLIP image group returned no diffusion loss")
            from .unilip_training import unwrap_composite
            decoder = unwrap_composite(model).vision_decoder
            fraction = group["target_images"].shape[0] / n_images
            loss = loss + float(decoder.spec["image_loss_weight"]) * fraction * result.image_loss
            image_loss = image_loss + fraction * result.image_loss.detach()
        if image_only:
            text_outputs = result
            text_outputs.loss = loss
            text_outputs.image_loss = image_loss
        self._record_losses(model, None if image_only else text_outputs.text_loss, image_loss)
        return (loss, text_outputs) if return_outputs else loss
