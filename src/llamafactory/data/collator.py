# Copyright 2025 OpenAccess AI Collective and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy, inspect, logging, os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, MROPE_MODELS
from ..extras.nvtx import nvtx_range
from ..extras.packages import is_pillow_available
from .extra_encoders import (
    OMNI_MODEL_TYPES,
    cast_encoder_inputs,
    get_encoder_specs,
    pop_encoder_samples,
    samples_by_encoder_name,
)

import io

if is_pillow_available():
    from PIL import Image

if TYPE_CHECKING:
    from transformers import ProcessorMixin

from .template import Template


# =====================================================================
# Multimodal input slicing
# =====================================================================

def _slice_mm_inputs_for_sample(
    mm_inputs: dict[str, Any], batch_imglens: list[int], batch_vidlens: list[int],
    batch_idx: int, images_per_subseq: Optional[list[int]] = None,
    videos_per_subseq: Optional[list[int]] = None, subseq_idx: Optional[int] = None,
) -> dict[str, Any]:
    image_start_idx = sum(batch_imglens[:batch_idx])
    image_end_idx = sum(batch_imglens[: batch_idx + 1])
    video_start_idx = sum(batch_vidlens[:batch_idx])
    video_end_idx = sum(batch_vidlens[: batch_idx + 1])

    if subseq_idx is not None and images_per_subseq is not None:
        image_start_idx += sum(images_per_subseq[:subseq_idx])
        image_end_idx = image_start_idx + images_per_subseq[subseq_idx]

    if subseq_idx is not None and videos_per_subseq is not None:
        video_start_idx += sum(videos_per_subseq[:subseq_idx])
        video_end_idx = video_start_idx + videos_per_subseq[subseq_idx]

    sliced_mm_inputs: dict[str, Any] = {}
    key_to_slice_meta = {
        "image_grid_thw": (image_start_idx, image_end_idx, True),
        "video_grid_thw": (video_start_idx, video_end_idx, True),
        "second_per_grid_ts": (video_start_idx, video_end_idx, False),
        "video_second_per_grid": (video_start_idx, video_end_idx, False),
    }

    for key, (start_idx, end_idx, assign_none_when_empty) in key_to_slice_meta.items():
        if key not in mm_inputs: continue
        mm_value = mm_inputs[key]
        if mm_value is not None and end_idx > start_idx:
            sliced_mm_inputs[key] = mm_value[start_idx:end_idx]
        elif assign_none_when_empty:
            sliced_mm_inputs[key] = None
    return sliced_mm_inputs


# =====================================================================
# 4D attention mask
# =====================================================================

def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


# =====================================================================
# Main multimodal collator
# =====================================================================

@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")
        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model
        if getattr(getattr(self.model, "config", None), "model_type", None) == "moss_vl":
            self.get_rope_func = None
        elif self.model is not None and hasattr(self.model, "get_rope_index"):
            self.get_rope_func = self.model.get_rope_index
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index
        else:
            self.get_rope_func = None

    # =================================================================
    # RoPE
    # =================================================================

    _rope_logger = logging.getLogger(__name__ + ".rope_fallback")

    @staticmethod
    def _fallback_rope_position_ids(features: dict[str, "torch.Tensor"]) -> None:
        """Flat 3-axis positional encoding fallback when get_rope_index fails."""
        bsz, seq_len = features["input_ids"].shape
        pos = torch.arange(seq_len, device=features["input_ids"].device)
        features["position_ids"] = pos.unsqueeze(0).unsqueeze(0).expand(3, bsz, seq_len).contiguous()
        features["rope_deltas"] = torch.zeros(bsz, 1, device=features["input_ids"].device, dtype=torch.long)

    def _compute_rope_position_ids(self, features: dict[str, "torch.Tensor"], mm_inputs: dict[str, Any]) -> None:
        rope_index_kwargs = {
            "input_ids": features["input_ids"],
            "image_grid_thw": mm_inputs.get("image_grid_thw"),
            "video_grid_thw": mm_inputs.get("video_grid_thw"),
            "attention_mask": (features["attention_mask"] >= 1).float(),
        }
        if features["attention_mask"].sum() == 0:
            seq_len = features["input_ids"].shape[-1]
            features["position_ids"] = (
                torch.arange(seq_len).view(1, 1, seq_len).expand(3, *features["input_ids"].shape).contiguous()
            )
            features["rope_deltas"] = torch.zeros(features["input_ids"].shape[0])
            return

        if "mm_token_type_ids" in inspect.signature(self.get_rope_func).parameters:
            image_token_id = getattr(self.model.config, "image_token_id", None)
            video_token_id = getattr(self.model.config, "video_token_id", None)
            if image_token_id is not None or video_token_id is not None:
                mm_token_type_ids = torch.zeros_like(features["input_ids"])
                if image_token_id is not None:
                    mm_token_type_ids[features["input_ids"] == image_token_id] = 1
                if video_token_id is not None:
                    mm_token_type_ids[features["input_ids"] == video_token_id] = 2
                rope_index_kwargs["mm_token_type_ids"] = mm_token_type_ids

        if "second_per_grid_ts" in mm_inputs:
            rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
        elif "video_second_per_grid" in mm_inputs:
            rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

        model_type = getattr(self.model.config, "model_type", None)
        if model_type in OMNI_MODEL_TYPES:
            rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
            feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                rope_index_kwargs["audio_seqlens"] = audio_feature_lengths
            try:
                features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
            except (IndexError, RuntimeError) as e:
                self._rope_logger.warning("get_rope_index mismatch (omni), fallback: %s", e)
                self._fallback_rope_position_ids(features)
                return
            features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(dim=-1).unsqueeze(-1)
        else:
            try:
                features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)
            except (IndexError, RuntimeError) as e:
                self._rope_logger.warning("get_rope_index mismatch, fallback: %s", e)
                self._fallback_rope_position_ids(features)

    # =================================================================
    # Packed RoPE
    # =================================================================

    def _compute_rope_position_ids_with_packing(
        self, features: dict[str, "torch.Tensor"], mm_inputs: dict[str, Any],
        packing_params_list: list[dict[str, Any] | None], batch_imglens: list[int],
        batch_vidlens: list[int], batch_audlens: list[int], has_dummy_image: bool,
    ) -> None:
        bsz = features["input_ids"].size(0)
        seq_len = features["input_ids"].size(1)
        all_position_ids, all_rope_deltas = [], []

        if has_dummy_image:
            unpadded_length = int(features["attention_mask"][0].bool().sum().item())
            right_padding_length = int((packing_params_list[0] or {}).get("right_padding_length") or 0)
            fake_input_padding_length = max(0, seq_len - unpadded_length - right_padding_length)
            dummy_image_right_padding_mrope = (
                torch.arange(fake_input_padding_length).view(1, 1, fake_input_padding_length).expand(3, bsz, fake_input_padding_length)
            )
            dummy_image_right_padding_attention_mask = torch.zeros((bsz, fake_input_padding_length))
            assert self.tokenizer.padding_side == "right"
            dummy_mm_inputs = copy.deepcopy(mm_inputs)

        for sample_idx in range(bsz):
            sample_packing = (packing_params_list[sample_idx] or {}) if sample_idx < len(packing_params_list) else {}
            sequence_boundaries = sample_packing.get("sequence_boundaries")
            num_sub_seqs = (len(sequence_boundaries) - 1) if sequence_boundaries and len(sequence_boundaries) > 1 else 1
            image_subseq_ids = sample_packing.get("image_subseq_ids") or []
            video_subseq_ids = sample_packing.get("video_subseq_ids") or []
            images_per_subseq = [image_subseq_ids.count(i) for i in range(num_sub_seqs)] if image_subseq_ids and num_sub_seqs > 1 else None
            videos_per_subseq = [video_subseq_ids.count(i) for i in range(num_sub_seqs)] if video_subseq_ids and num_sub_seqs > 1 else None

            if has_dummy_image: mm_inputs = {}

            if num_sub_seqs <= 1:
                sample_features = {
                    "input_ids": features["input_ids"],
                    "attention_mask": features["attention_mask"][sample_idx:sample_idx + 1],
                }
                mm_inputs_for_sample = _slice_mm_inputs_for_sample(mm_inputs, batch_imglens, batch_vidlens, sample_idx=sample_idx)
                self._compute_rope_position_ids(sample_features, mm_inputs_for_sample)
                all_position_ids.append(sample_features["position_ids"])
                all_rope_deltas.append(sample_features["rope_deltas"])
            else:
                sample_position_ids = []
                for subseq_idx in range(num_sub_seqs):
                    subseq_start = sequence_boundaries[subseq_idx]
                    subseq_end = sequence_boundaries[subseq_idx + 1]
                    subseq_features = {
                        "input_ids": features["input_ids"][sample_idx:sample_idx + 1, subseq_start:subseq_end],
                        "attention_mask": features["attention_mask"][sample_idx:sample_idx + 1, subseq_start:subseq_end],
                    }
                    mm_inputs_for_subseq = _slice_mm_inputs_for_sample(
                        mm_inputs, batch_imglens, batch_vidlens, sample_idx, images_per_subseq, videos_per_subseq, subseq_idx
                    )
                    self._compute_rope_position_ids(subseq_features, mm_inputs_for_subseq)
                    sample_position_ids.append(subseq_features["position_ids"])
                all_position_ids.append(torch.cat(sample_position_ids, dim=-1))

        batch_dim_for_position_ids = 1 if all_position_ids[0].dim() == 3 else 0
        features["position_ids"] = torch.cat(all_position_ids, dim=batch_dim_for_position_ids)

        if has_dummy_image:
            mm_inputs = dummy_mm_inputs

        expected_position_ids_shape = (
            (bsz, seq_len) if all_position_ids[0].dim() == 2 else (all_position_ids[0].size(0), bsz, seq_len)
        )

        if has_dummy_image:
            features["position_ids"] = torch.cat([features["position_ids"], dummy_image_right_padding_mrope], dim=-1)
            features["attention_mask"] = torch.cat([features["attention_mask"], dummy_image_right_padding_attention_mask], dim=-1)

        # Mirror the non-FA2 packing fix (#10737) for the packed-mrope path. The merged position_ids
        # is built from per-subseq sequence_boundaries, which end at cutoff_len, while
        # `DataCollatorForSeq2Seq(pad_to_multiple_of=...)` right-pads input_ids/attention_mask past
        # cutoff_len. Right-pad the trailing (masked) positions with 0 so the merged position_ids
        # matches seq_len before validating. Works for both 2D and 3D (mrope) position_ids since the
        # sequence axis is last, and is idempotent with the has_dummy_image cat above.
        pad_len = seq_len - features["position_ids"].shape[-1]
        if pad_len > 0:
            features["position_ids"] = F.pad(features["position_ids"], (0, pad_len), value=0)

        if features["position_ids"].shape != expected_position_ids_shape:
            raise ValueError(
                f"Merged position_ids shape mismatch: got {features['position_ids'].shape}, expected {expected_position_ids_shape}."
            )

    # =================================================================
    # TerraMind batching helpers
    # =================================================================

    @staticmethod
    def _normalize_terramind_value(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, bytes):
            with np.load(io.BytesIO(value)) as archive:
                if "data" not in archive:
                    raise ValueError("TerraMind NPZ payload must contain a 'data' array.")
                tensor = torch.from_numpy(np.asarray(archive["data"], dtype=np.float32).copy())
        elif isinstance(value, str):
            if not os.path.isfile(value):
                raise FileNotFoundError(f"TerraMind input file not found: {value}")
            if value.endswith(".npy"):
                tensor = torch.from_numpy(np.load(value))
            elif value.endswith(".npz"):
                with np.load(value) as archive:
                    if "data" in archive:
                        array = archive["data"]
                    elif len(archive.files) == 1:
                        array = archive[archive.files[0]]
                    else:
                        raise ValueError(f"TerraMind .npz file must contain a 'data' array or exactly one array: {value}")
                    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
            elif value.endswith((".pt", ".pth")):
                tensor = torch.load(value, map_location="cpu", weights_only=True)
                if not torch.is_tensor(tensor):
                    raise TypeError(f"TerraMind torch file must contain a tensor: {value}")
            else:
                raise ValueError(f"Unsupported TerraMind file type: {value}")
        else:
            tensor = torch.as_tensor(value)

        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"TerraMind modality tensor must have shape [C,H,W], got {tuple(tensor.shape)}.")
        return tensor.float()

    @classmethod
    def _extract_terramind_sample(cls, feature):
        terramind = feature.pop("terramind", None)
        if terramind is None:
            return None
        if not isinstance(terramind, dict):
            raise TypeError("'terramind' must be a dict mapping modality names to values.")
        result = {
            modality: cls._normalize_terramind_value(value)
            for modality, value in terramind.items()
            if value is not None
        }
        return result or None

    @staticmethod
    def _build_terramind_batch(samples):
        present = [sample for sample in samples if sample]
        if not present:
            return None

        modalities = set(present[0])
        for sample in present[1:]:
            if set(sample) != modalities:
                raise ValueError(
                    "All TerraMind examples in the same batch must contain the same modality set."
                )

        output = {}
        for modality in sorted(modalities):
            values = [sample[modality] for sample in present]
            try:
                output[modality] = torch.stack(values, dim=0)
            except RuntimeError as exc:
                shapes = [tuple(value.shape) for value in values]
                raise ValueError(
                    f"TerraMind modality {modality!r} must have one consistent shape per batch; got {shapes}."
                ) from exc
        return output

    def _collate_extra_encoders(self, batch_samples):
        r"""Build ``encoder_inputs`` from processor specs, with a TerraMind fallback."""
        specs = get_encoder_specs(self.processor, self.model)
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        use_generic = bool(specs) and hasattr(self.processor, "_collate_encoder_inputs")

        encoder_inputs = None
        terramind_pixel_values = None
        if use_generic:
            grouped = samples_by_encoder_name(batch_samples, specs)
            encoder_inputs = self.processor._collate_encoder_inputs(grouped) or None
            if model_type == "qwen2_5_omni_terramind" and encoder_inputs:
                terramind = encoder_inputs.get("terramind")
                if terramind is not None:
                    terramind_pixel_values = (
                        terramind["inputs"]
                        if isinstance(terramind, dict) and "inputs" in terramind
                        else terramind
                    )
        else:
            extra_names = {
                name
                for sample in batch_samples
                for name, value in (sample or {}).items()
                if value is not None and name != "terramind"
            }
            if extra_names:
                raise ValueError(
                    "Batch has extra encoder columns "
                    f"{sorted(extra_names)} but the processor has no "
                    "encoder_specs/_collate_encoder_inputs. Load the composite "
                    "checkpoint with trust_remote_code."
                )
            terramind_samples = []
            for sample in batch_samples:
                terramind = (sample or {}).get("terramind")
                if terramind is None:
                    terramind_samples.append(None)
                else:
                    terramind_samples.append(self._extract_terramind_sample({"terramind": terramind}))
            terramind_pixel_values = self._build_terramind_batch(terramind_samples)

        if model_type == "qwen2_5_omni_terramind":
            return encoder_inputs, terramind_pixel_values
        if model_type == "qwen2_5_omni_composite" or use_generic:
            return encoder_inputs, None
        return None, terramind_pixel_values

    # =================================================================
    # Main collator
    # =================================================================

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        is_moss_vl = model_type == "moss_vl"

        # Accelerate's IterableDatasetShard pads the final batch of a streaming epoch by recycling
        # sample dicts it already yielded. Popping keys below would strip "images"/"videos"/"audios"
        # from those shared dicts, so a recycled sample would keep its mm placeholder tokens while
        # losing its media. Work on copies to keep the caller's dicts intact.
        features = [dict(feature) for feature in features]

        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        packing_params_list = []
        batch_encoder_samples = []

        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            batch_encoder_samples.append(pop_encoder_samples(feature))
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])
            packing_params_list.append(feature.pop("packing_params", None))

        encoder_inputs, terramind_pixel_values = self._collate_extra_encoders(
            batch_encoder_samples
        )

        # Fake image for text-only Qwen examples.
        fake_input_ids = []
        has_dummy_image = False
        if (self.template.mm_plugin.image_token is not None and sum(batch_imglens) == 0
                and sum(batch_vidlens) == 0 and not is_moss_vl):
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(fake_messages, fake_images, [], [], self.processor)
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_images = fake_images
            batch_imglens[0] = 1
            has_dummy_image = True

        # Fake audio
        if self.template.mm_plugin.audio_token is not None and sum(batch_audlens) == 0:
            fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
            fake_audios = [np.zeros(1600)]
            fake_messages = self.template.mm_plugin.process_messages(fake_messages, [], [], fake_audios, self.processor)
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_audios = fake_audios
            batch_audlens[0] = 1

        # Inject fake multimodal tokens.
        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]
            batch_input_ids[0] = features[0]["input_ids"]

        with nvtx_range(f"data/collate/mm_inputs(imgs={len(batch_images)})"):
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                batch_images,
                batch_videos,
                batch_audios,
                batch_imglens,
                batch_vidlens,
                batch_audlens,
                batch_input_ids,
                self.processor,
            )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        if "mm_token_type_ids" in mm_inputs:
            mm_token_type_ids = mm_inputs.pop("mm_token_type_ids")
            max_len = max(len(ids) for ids in mm_token_type_ids)
            padded = []
            for ids in mm_token_type_ids:
                pad_len = max_len - len(ids)
                if self.tokenizer.padding_side == "right":
                    padded.append(ids + [0] * pad_len)
                else:
                    padded.append([0] * pad_len + ids)
            mm_inputs["mm_token_type_ids"] = torch.tensor(padded, dtype=torch.long)

        with nvtx_range("data/collate/pad"):
            features: dict[str, torch.Tensor] = super().__call__(features)

        # Extra encoders: generic ``encoder_inputs`` for the composite; legacy
        # ``terramind_pixel_values`` only for the old TerraMind wrapper.
        if encoder_inputs:
            features["encoder_inputs"] = encoder_inputs
        if terramind_pixel_values is not None:
            features["terramind_pixel_values"] = terramind_pixel_values

        bsz, seq_len = features["input_ids"].shape[:2]
        is_omni = model_type in OMNI_MODEL_TYPES

        # RoPE
        if self.get_rope_func is not None:
            boundaries_list = [(p.get("sequence_boundaries") if p is not None else None) for p in packing_params_list]
            has_packing = any(b is not None and len(b) > 2 for b in boundaries_list)
            if has_dummy_image and has_packing:
                features["has_dummy_image"] = True
            # When fake image/audio was injected, sequence_boundaries no longer match the tensor; use non-packing path.
            with nvtx_range("data/collate/rope"):
                if not has_packing:
                    self._compute_rope_position_ids(features, mm_inputs)
                else:
                    if is_omni:  # TODO: support omni models for packed sequences @kuangdd
                        raise RuntimeError("Omni models are not supported for packed sequences for now.")

                    self._compute_rope_position_ids_with_packing(
                        features,
                        mm_inputs,
                        packing_params_list,
                        batch_imglens,
                        batch_vidlens,
                        batch_audlens,
                        has_dummy_image,
                    )

            # For transformers compatibility, after https://github.com/huggingface/transformers/issues/39400
            if features["position_ids"].dim() == 3:
                features["position_ids"] = torch.cat([features["position_ids"][0].unsqueeze(0), features["position_ids"]], dim=0)

        # MRoPE validation.
        if (self.model is not None and getattr(self.model.config, "model_type", None) in MROPE_MODELS
                and ("position_ids" not in features or features["position_ids"].dim() != 3)):
            raise ValueError(f"{self.model.config.model_type} requires 3D position ids for mrope.")

        # Cross attention mask.
        if "cross_attention_mask" in mm_inputs and mm_inputs["cross_attention_mask"].dtype != torch.bool:
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        # MOSS-VL
        if is_moss_vl:
            mm_inputs = self.template.mm_plugin.post_process_mossvl_inputs(features, mm_inputs, self.processor)

        # Merge Qwen multimodal inputs.
        features.update(mm_inputs)

        # MiniCPM-V
        if "image_bound" in features:
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


# =====================================================================
# 4D attention collator
# =====================================================================

@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32
    neat_packing: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.neat_packing and self.attn_implementation == "flash_attention_2":
            if self.model is not None and getattr(self.model.config, "model_type", None) in ["gemma4", "gpt_oss"]:
                raise ValueError("Neat packing is not supported for gemma4, gpt_oss models for now.")

    @staticmethod
    def _unpad_packed_features(features: dict[str, Any]) -> None:
        attention_mask = features.get("attention_mask")
        if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2 or attention_mask.size(0) != 1:
            return
        seq_len = attention_mask.size(1)
        non_padding_indices = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
        if non_padding_indices.numel() == seq_len: return

        keys_on_seq_dim_1 = {"input_ids", "labels", "attention_mask", "token_type_ids"}
        for key, value in list(features.items()):
            if not torch.is_tensor(value): continue
            if key == "position_ids" and value.size(-1) == seq_len:
                features[key] = value.index_select(-1, non_padding_indices)
            elif key == "cross_attention_mask" and value.dim() >= 2 and value.size(0) == 1 and value.size(1) == seq_len:
                features[key] = value.index_select(1, non_padding_indices)
            elif key in keys_on_seq_dim_1 and value.dim() == 2 and value.size(0) == 1 and value.size(1) == seq_len:
                features[key] = value.index_select(1, non_padding_indices)
                
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        has_dummy_image = features.pop("has_dummy_image", False)

        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        if self.neat_packing and self.attn_implementation == "flash_attention_2":
            assert features["input_ids"].shape[0] == 1, "bsz should be 1 for neat packing"
            if not has_dummy_image:
                self._unpad_packed_features(features)
            features["attention_mask"] = None  # let transformers handle causal packed mask.
        else:
            # `DataCollatorForSeq2Seq(pad_to_multiple_of=...)` pads `input_ids`/`attention_mask`
            # but leaves `position_ids` untouched (it is not in `model_input_names`). On the
            # non-FA2 packing path we do not unpad, so `position_ids` stays shorter than
            # `input_ids`, which makes cos/sin shorter than query and crashes
            # `apply_rotary_pos_emb`. Right-pad `position_ids` to the padded length to match.
            position_ids = features.get("position_ids")
            if torch.is_tensor(position_ids):
                pad_len = features["input_ids"].shape[-1] - position_ids.shape[-1]
                if pad_len > 0:
                    features["position_ids"] = F.pad(position_ids, (0, pad_len), value=0)

        for key, value in features.items():
            if key in {"encoder_inputs", "kl_encoder_inputs"}:
                features[key] = cast_encoder_inputs(value, self.compute_dtype)
                continue
            if key == "terramind_pixel_values" and isinstance(value, dict):
                for modality, tensor in value.items():
                    if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
                        value[modality] = tensor.to(self.compute_dtype)
                continue
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)
        return features


# =====================================================================
# Pairwise collator
# =====================================================================

@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature.get("images", []),
                    "videos": feature.get("videos", []),
                    "audios": feature.get("audios", []),
                    "encoders": feature.get("encoders")
                    or ({"terramind": feature["terramind"]} if feature.get("terramind") is not None else {}),
                }
                concatenated_features.append(target_feature)
        return super().__call__(concatenated_features)


# =====================================================================
# KTO collator
# =====================================================================

@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        target_features, kl_features, kto_tags = [], [], []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature.get("images", []),
                "videos": feature.get("videos", []),
                "audios": feature.get("audios", []),
                "encoders": feature.get("encoders")
                or ({"terramind": feature["terramind"]} if feature.get("terramind") is not None else {}),
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature.get("images", []),
                "videos": feature.get("videos", []),
                "audios": feature.get("audios", []),
                "encoders": feature.get("encoders")
                or ({"terramind": feature["terramind"]} if feature.get("terramind") is not None else {}),
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)

        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "cross_attention_mask" in kl_batch:
            batch["kl_cross_attention_mask"] = kl_batch["cross_attention_mask"]
        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]
        if "terramind_pixel_values" in kl_batch:
            batch["kl_terramind_pixel_values"] = kl_batch["terramind_pixel_values"]
        if "encoder_inputs" in kl_batch:
            batch["kl_encoder_inputs"] = kl_batch["encoder_inputs"]
        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch