"""Configure UniLIP trainability in LlamaFactory after root Qwen PEFT setup.

Supports the UniLIP conditioning-source selector implemented in the companion
omni_composite patch: queries, prompt, image, and encoder:<registered_name>.
"""
from __future__ import annotations

import json
import inspect
import logging
from pathlib import Path

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

LOGGER = logging.getLogger(__name__)


def _upcast_unilip_trainable_parameters(model, finetuning_args):
    """Apply LlamaFactory's mixed-precision policy after enabling UniLIP modules.

    Root PEFT setup runs before configure_unilip_training and therefore only
    upcasts parameters trainable at that point. Newly enabled connector/query/
    projector weights and DiT adapters otherwise remain BF16, with BF16 Adam
    states and rounded-away small updates. Keep their stored weights in FP32;
    Trainer autocast still performs BF16 matrix operations.
    """
    if (
        getattr(finetuning_args, "pure_bf16", False)
        or getattr(finetuning_args, "use_badam", False)
        or is_deepspeed_zero3_enabled()
    ):
        return
    upcast = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32:
            upcast += parameter.numel()
            parameter.data = parameter.data.float()
    if upcast:
        LOGGER.info("Upcast %s newly trainable UniLIP parameters to FP32 for optimizer updates.", f"{upcast:,}")


def unwrap_composite(model):
    """Reach the composite underneath optional PEFT/Distributed wrappers."""
    for _ in range(8):
        # PeftModel proxies attributes; unwrap it before inspecting modules.
        if hasattr(model, "get_base_model"):
            other = model.get_base_model()
            if other is not model:
                model = other
                continue
        if getattr(model, "vision_decoder", None) is not None:
            return model
        if hasattr(model, "module"):
            model = model.module
        else:
            break
    raise ValueError("UniLIP training requires QwenOmniComposite with vision_decoder")


def _check_checkpoint_spec(decoder, saved_spec):
    """Reject architectural mismatches before loading supplemental weights.

    In particular, source_heads are named by the POSITION of the corresponding
    source in conditioning_sources, so simply comparing the sets is unsafe.
    """
    current = decoder.spec
    current_sources = list(current.get("conditioning_sources", ["queries"]))
    saved_sources = list(saved_spec.get("conditioning_sources", ["queries"]))
    if current_sources != saved_sources:
        raise ValueError(
            "UniLIP conditioning_sources differ from checkpoint: "
            f"model={current_sources}, checkpoint={saved_sources}. "
            "Build/load a matching base model; ordering matters."
        )

    # Older query-only checkpoints might not have conditioning_mode recorded.
    default_mode = "parallel_queries" if current_sources == ["queries"] else "token_positions"
    for name, default in (
        ("conditioning_mode", default_mode),
        ("query_attention_mode", "bidirectional"),
        ("query_projector_mode", "none"),
        ("connector_mode", "pretrained"),
        ("tokenizer_hidden_size", None),
        ("conditioning_projection_mode", "separate"),
        ("prompt_max_tokens", 256),
        ("prompt_text_selection", None),
        ("num_conditioning_tokens", None),
        ("query_capacity", None),
        ("adapter_dim", None),
        ("max_conditioning_tokens", 1024),
    ):
        if (name in current or name in saved_spec) and current.get(name, default) != saved_spec.get(name, default):
            raise ValueError(
                f"UniLIP {name} differs from checkpoint: "
                f"model={current.get(name, default)!r}, checkpoint={saved_spec.get(name, default)!r}."
            )
    if saved_spec.get("query_capacity") != decoder.latent_queries.num_embeddings:
        raise ValueError("Saved UniLIP query_capacity differs from model latent_queries")
    # Shape-invariant but semantically essential: avoid silent position routing changes.
    current_names = list(decoder._source_head_keys) if hasattr(decoder, "_source_head_keys") else []
    expected_names = [source for source in saved_sources if source != "queries"]
    if current_names != expected_names:
        raise ValueError("UniLIP source-head module registration does not match conditioning sources")


def _load_unilip_extra(composite, checkpoint_dir):
    """Restore supplemental weights (including projection heads) for PEFT runs."""
    from safetensors.torch import load_file

    folder = Path(checkpoint_dir)
    extra = folder / "unilip_extra.safetensors"
    manifest = folder / "unilip_config.json"
    if manifest.is_file() and not extra.is_file():
        raise FileNotFoundError(
            f"{manifest} exists but {extra.name} is missing: "
            "restoring only the root Qwen PEFT adapter would lose UniLIP weights."
        )
    if not extra.is_file():
        return
    state = load_file(str(extra), device="cpu")
    # New source projection weights should never silently remain random on resume.
    source_head_keys = {
        name for name, _ in composite.named_parameters()
        if name.startswith(("vision_decoder.source_heads.", "vision_decoder.query_projector.")) or (
            composite.vision_decoder.spec.get("conditioning_projection_mode","separate") == "shared"
            and name.startswith("vision_decoder.output_head."))
    }
    if composite.vision_decoder.spec.get('connector_mode') == 'omni':
        source_head_keys.update(n for n,_ in composite.named_parameters()
            if n.startswith(('vision_decoder.llm_connector.', 'vision_decoder.projector.', 'vision_decoder.latent_queries.')))
    missing_heads = source_head_keys - set(state)
    if missing_heads:
        raise RuntimeError(
            "UniLIP supplemental checkpoint is missing source-head parameters: "
            f"{sorted(missing_heads)[:8]}"
        )
    incompatible = composite.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"UniLIP checkpoint has unexpected keys: {incompatible.unexpected_keys[:8]}"
        )
    LOGGER.info("Restored UniLIP supplemental weights from %s", extra)


def _configure_projector_only(model, decoder):
    if (decoder.spec.get('conditioning_projection_mode') != 'shared'
            or getattr(decoder, 'query_projector', None) is not None):
        raise ValueError('unilip_projector_only requires the shared Omni-to-UniLIP projector model')
    model.requires_grad_(False)
    decoder.output_head.requires_grad_(True)


def _freeze_pretrained_bridge(decoder):
    for module in (decoder.llm_connector, decoder.projector):
        if module is not None:
            module.requires_grad_(False)


def _configure_prompt_only(decoder):
    if (decoder.spec.get('conditioning_projection_mode') != 'shared'
            or getattr(decoder, 'query_projector', None) is not None):
        raise ValueError('unilip_prompt_only requires the shared Omni-to-UniLIP projector model')
    if 'prompt' not in decoder.spec.get('conditioning_sources', []):
        raise ValueError('unilip_prompt_only requires a model with prompt conditioning configured')
    if not decoder.spec.get('prompt_text_selection') or decoder.spec.get('prompt_max_tokens') != 256:
        raise ValueError('unilip_prompt_only requires the saved non-system last-256-text-token selection')
    decoder.spec['conditioning_sources'] = ['prompt']
    # Shared mode has no source-specific head weights. Match the registration
    # constructed from this saved prompt-only spec when loading the checkpoint.
    decoder._source_head_keys = {'prompt': 'source_0'}
    decoder.latent_queries.requires_grad_(False)


def _configure_bridge_only(model, decoder):
    if decoder.spec.get('connector_mode', 'pretrained') in ('none', 'omni'):
        connector_mode = decoder.spec['connector_mode']
        omni_lora = [p for n,p in model.named_parameters()
                     if p.requires_grad and 'qwen.thinker.model.' in n and 'lora_' in n]
        if connector_mode == 'omni':
            if (decoder.llm_connector is None or decoder.projector is None
                    or decoder.query_projector is not None
                    or decoder.spec.get('conditioning_sources') != ['queries']):
                raise ValueError('Omni connector training requires query -> connector -> linear projector.')
            model.requires_grad_(False)
            for parameter in omni_lora:
                parameter.requires_grad_(True)
            decoder.llm_connector.requires_grad_(True)
            decoder.projector.requires_grad_(True)
            decoder.latent_queries.requires_grad_(True)
            return
        if decoder.llm_connector is not None or decoder.projector is not None or decoder.output_head is not None:
            raise ValueError('connector_mode=none must not allocate the UniLIP bridge')
        if decoder.query_projector is None or decoder.spec.get('conditioning_sources') != ['queries']:
            raise ValueError('Connector-free training requires direct query projection')
        # Root PEFT owns Omni LoRA. Preserve only the explicitly enabled text
        # decoder adapters; all Omni base, vision/audio, and DiT weights freeze.
        omni_lora = [p for n,p in model.named_parameters()
                     if p.requires_grad and 'qwen.thinker.model.' in n and 'lora_' in n]
        model.requires_grad_(False)
        for parameter in omni_lora:
            parameter.requires_grad_(True)
        decoder.query_projector.requires_grad_(True)
        decoder.latent_queries.requires_grad_(True)
        return
    if (decoder.spec.get('conditioning_projection_mode') != 'shared'
            or getattr(decoder, 'query_projector', None) is not None):
        raise ValueError('unilip_bridge_only requires the shared Omni-to-UniLIP projector model')
    model.requires_grad_(False)
    for module in (decoder.output_head, decoder.llm_connector, decoder.projector):
        module.requires_grad_(True)
    decoder.latent_queries.requires_grad_('queries' in decoder.spec.get('conditioning_sources', ['queries']))


def configure_unilip_training(model, finetuning_args, checkpoint_dir=None):
    from peft import LoraConfig, TrainableTokensConfig, inject_adapter_in_model

    mode = finetuning_args.unilip_train_mode
    projector_only = getattr(finetuning_args, 'unilip_projector_only', False)
    if projector_only and mode != 'frozen':
        raise ValueError('unilip_projector_only requires unilip_train_mode: frozen')
    if mode == "disabled":
        return model
    if mode not in {"frozen", "lora", "full"}:
        raise ValueError(f"Unsupported unilip_train_mode={mode!r}")
    composite = unwrap_composite(model)
    decoder = composite.vision_decoder
    if getattr(finetuning_args, 'unilip_prompt_only', False):
        _configure_prompt_only(decoder)
    sources = list(decoder.spec.get("conditioning_sources", ["queries"]))
    # Old queries-only decoders have no source_heads; remain compatible.
    source_heads = getattr(decoder, "source_heads", None)
    if source_heads is None and any(source != "queries" for source in sources):
        raise RuntimeError(
            "Selected image/encoder conditioning requires the updated "
            "omni_composite/decoders_unilip.py with source_heads. Rebuild the model."
        )

    saved_spec = None
    if checkpoint_dir:
        manifest = Path(checkpoint_dir) / "unilip_config.json"
        if manifest.is_file():
            saved_spec = json.loads(manifest.read_text())
            _check_checkpoint_spec(decoder, saved_spec)

    # Parallel queries share one full Qwen pass; image training does not use cache.
    composite.gradient_checkpointing_disable()
    for module in (composite, composite.qwen.thinker, composite.qwen.thinker.model):
        module.config.use_cache = False
    decoder._activation_checkpointing = getattr(
        finetuning_args, "unilip_activation_checkpointing", True
    )
    if decoder._activation_checkpointing:
        composite.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if decoder.llm_connector is not None:
            decoder.llm_connector.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        decoder.dit.enable_gradient_checkpointing()
    else:
        if decoder.llm_connector is not None:
            decoder.llm_connector.gradient_checkpointing_disable()
        decoder.dit.disable_gradient_checkpointing()

    def validated_counts(counts):
        candidate = dict(decoder.spec, conditioning_tokens_by_resolution=counts)
        validator = composite.config._validate_vision_decoder
        if "encoder_names" in inspect.signature(validator).parameters:
            validated = validator(
                candidate, encoder_names={spec["name"] for spec in composite.config.encoders}
            )
        else:
            # Older queries-only configurations do not accept encoder_names.
            validated = validator(candidate)
        return validated["conditioning_tokens_by_resolution"]

    overrides = getattr(finetuning_args, "unilip_conditioning_tokens_by_resolution", None)
    if saved_spec is not None:
        checkpoint_counts = validated_counts(saved_spec.get("conditioning_tokens_by_resolution", {}))
        if overrides is not None and validated_counts(overrides) != checkpoint_counts:
            raise ValueError(
                "unilip_conditioning_tokens_by_resolution conflicts with resumed checkpoint. "
                "Use the checkpoint's counts, or start a new training run."
            )
        decoder.spec["conditioning_tokens_by_resolution"] = checkpoint_counts
    elif overrides is not None:
        decoder.spec["conditioning_tokens_by_resolution"] = validated_counts(overrides)

    adapters = decoder.spec.setdefault("adapters", {})
    decoder.spec["full_dit_weights"] = mode == "full" or bool(
        (saved_spec or {}).get("full_dit_weights")
    )
    existing_lora = any("lora_A" in name for name, _ in decoder.dit.named_parameters())
    saved_dit = (saved_spec or {}).get("adapters", {}).get("dit")
    if (existing_lora or saved_dit) and mode == "full":
        raise ValueError("Merge the saved DiT LoRA before switching to full DiT training")
    if saved_dit and mode == "frozen" and not existing_lora:
        options = dict(saved_dit)
        adapter_name = options.pop("adapter_name", "unilip")
        inject_adapter_in_model(LoraConfig(**options), decoder.dit, adapter_name=adapter_name)
        adapters["dit"] = dict(saved_dit)

    def setting(field, fallback):
        value = getattr(finetuning_args, field, None)
        return fallback if value is None else value

    train_connector = setting("unilip_train_connector", True)
    weight = float(setting("unilip_image_loss_weight", 1.0))
    if weight < 0:
        raise ValueError("unilip_image_loss_weight must be nonnegative")
    decoder.spec["image_loss_weight"] = weight
    if mode == "lora":
        rank = int(setting("unilip_lora_rank", 16))
        alpha = float(setting("unilip_lora_alpha", rank * 2))
        dropout = float(setting("unilip_lora_dropout", 0.0))
        target = (setting("unilip_lora_target", None)
                  or decoder.spec.get("lora_targets") or ["to_q", "to_k", "to_v"])
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("UniLIP LoRA requires positive rank/alpha and dropout in [0, 1)")
        if isinstance(target, str):
            target = [x.strip() for x in target.split(",") if x.strip()]
        options = dict(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                       target_modules=list(target), bias="none")
        adapter_name = "unilip"
        if saved_dit:
            options = dict(saved_dit)
            adapter_name = options.pop("adapter_name", "unilip")
        if not existing_lora:
            inject_adapter_in_model(LoraConfig(**options), decoder.dit, adapter_name=adapter_name)
            adapters["dit"] = dict(options, adapter_name=adapter_name)
        elif "dit" not in adapters:
            adapters["dit"] = dict(
                r=decoder.spec["lora_rank"], lora_alpha=decoder.spec["lora_alpha"],
                target_modules=decoder.spec["lora_targets"], adapter_name="default"
            )
    for name, parameter in decoder.dit.named_parameters():
        parameter.requires_grad_(mode == "full" or (mode == "lora" and "lora_" in name))
    decoder._freeze_dit = mode == "frozen"
    if decoder._freeze_dit:
        decoder.dit.eval()

    # IMPORTANT: root get_peft_model() froze every base-model parameter, including
    # the newly added source_heads. Turn them back on exactly like output_head.
    for module in (
        decoder.output_head,
        decoder.latent_queries,
        decoder.trigger_embedding,
        decoder.trigger_head,
        decoder.llm_connector,
        decoder.projector,
    ):
        if module is not None:
            module.requires_grad_(train_connector)
    if getattr(decoder, 'query_projector', None) is not None:
        decoder.query_projector.requires_grad_(train_connector)
        if decoder.output_head is not None:
            decoder.output_head.requires_grad_(False)  # Legacy query route is unused.
        if sources == ['queries']:
            # A direct query-only model never executes the imported bridge.
            _freeze_pretrained_bridge(decoder)
    if source_heads is not None:
        source_heads.requires_grad_(train_connector)

    # Source heads are randomly initialized in new models. If the connector
    # itself is frozen, source heads must nevertheless be trainable initially.
    if source_heads is not None and len(source_heads) > 0 and not train_connector:
        LOGGER.warning(
            "UniLIP source_heads are frozen together with unilip_train_connector=false. "
            "For a newly built multi-source checkpoint, set unilip_train_connector=true."
        )
    decoder._freeze_image_tokenizer = True
    if decoder.image_tokenizer is not None:
        decoder.image_tokenizer.requires_grad_(False).eval()

    # The root fine-tuning method controls Qwen (full, freeze, or LoRA).
    # Restore full-FT projectors after PEFT's global freeze.
    train_projectors = setting("unilip_train_projectors", True)
    composite.projectors.requires_grad_(train_projectors)
    for encoder in composite.encoders.values():
        encoder.freeze_backbone()
        gate = getattr(encoder, "fusion_gate", None)
        if gate is not None:
            gate.requires_grad_(train_projectors)

    token_ids = sorted(set(decoder.spec.get("resolution_token_ids", {}).values()))
    train_tokens = setting("unilip_train_resolution_tokens", True)
    thinker = composite.qwen.thinker
    saved_resolution = (saved_spec or {}).get("adapters", {}).get("resolution")
    if (train_tokens and token_ids) or saved_resolution:
        if not any("trainable_tokens_delta" in name for name, _ in thinker.named_parameters()):
            # Token injection freezes its host. Preserve existing Qwen LoRA
            # or full tuning policy instead of silently disabling language SFT.
            previous = {id(p): p.requires_grad for p in thinker.parameters()}
            token_options = saved_resolution or dict(token_indices=token_ids, adapter_name="resolution")
            inject_adapter_in_model(TrainableTokensConfig(
                token_indices=token_options["token_indices"],
                target_modules=["model.embed_tokens", "lm_head"]
            ), thinker, adapter_name=token_options.get("adapter_name", "resolution"))
            for p in thinker.parameters():
                if id(p) in previous:
                    p.requires_grad_(previous[id(p)])
        adapters["resolution"] = saved_resolution or dict(
            token_indices=token_ids, adapter_name="resolution"
        )
    for name, param in thinker.named_parameters():
        if "trainable_tokens_delta" in name:
            param.requires_grad_(train_tokens)

    composite.config.vision_decoder = decoder.spec
    if getattr(finetuning_args, 'unilip_freeze_pretrained_bridge', False):
        _freeze_pretrained_bridge(decoder)
    if projector_only:
        # Apply last, after PEFT/token adapter setup, so no
        # earlier policy can leave any other parameter trainable.
        _configure_projector_only(model, decoder)
        train_tokens = train_projectors = False
    if getattr(finetuning_args, 'unilip_bridge_only', False):
        _configure_bridge_only(model, decoder)
        train_tokens = train_projectors = False
    # All trainability choices and adapter injections must precede this cast.
    # Loading older BF16 supplemental weights preserves the FP32 destination.
    _upcast_unilip_trainable_parameters(model, finetuning_args)
    if checkpoint_dir:
        # Restore into the final training dtype so FP32 connector updates are
        # not rounded through a temporary BF16 destination when resuming.
        _load_unilip_extra(composite, checkpoint_dir)
    trainable_dit = [name for name, p in decoder.dit.named_parameters() if p.requires_grad]
    if mode == "lora" and (
        not trainable_dit or any("lora_" not in name for name in trainable_dit)
    ):
        raise RuntimeError("UniLIP LoRA mode must train only DiT LoRA matrices")
    trainable_heads = sum(
        p.numel() for p in (source_heads.parameters() if source_heads is not None else ())
        if p.requires_grad
    )
    if source_heads is not None and len(source_heads) > 0 and train_connector and trainable_heads == 0:
        raise RuntimeError("UniLIP source-head parameters are not trainable after PEFT setup")
    LOGGER.warning(
        "UniLIP training: DiT=%s connector=%s source_heads=%s "
        "(%d trainable params), query_projector=%d trainable params, sources=%s, resolution_tokens=%s, "
        "projectors=%s, activation_checkpointing=%s",
        mode, decoder.llm_connector is not None and any(p.requires_grad for p in decoder.llm_connector.parameters()),
        source_heads is not None and len(source_heads) > 0, trainable_heads,
        sum(p.numel() for p in decoder.query_projector.parameters() if p.requires_grad)
        if getattr(decoder, 'query_projector', None) is not None else 0,
        sources, train_tokens, train_projectors, decoder._activation_checkpointing,
    )
    if getattr(finetuning_args, 'unilip_bridge_only', False):
        LOGGER.warning(
            'UniLIP bridge-only: queries_trainable=%s; connector_mode=%s; conditioning projection modules trainable; '
            'Omni base, control tokens, DiT and extra-encoder projectors frozen; '
            'Omni text LoRA trainable parameters=%d; diffusion loss only; timesteps=%s',
            any(p.requires_grad for p in decoder.latent_queries.parameters()),
            decoder.spec.get('connector_mode', 'pretrained'),
            sum(p.numel() for n,p in model.named_parameters()
                if p.requires_grad and 'qwen.thinker.model.' in n and 'lora_' in n),
            getattr(finetuning_args, 'unilip_timestep_sampling', 'uniform'),
        )
    elif projector_only:
        LOGGER.warning(
            'UniLIP projector-only: output_head=%d trainable parameters; '
            'Qwen/LoRA, queries, connector, caption projector, and DiT frozen; diffusion loss only',
            sum(p.numel() for p in decoder.output_head.parameters() if p.requires_grad)
            if decoder.output_head is not None else 0,
        )
    elif getattr(finetuning_args, 'unilip_freeze_pretrained_bridge', False):
        LOGGER.warning(
            'UniLIP pretrained connector and caption projection frozen; '
            'Qwen/LoRA policy preserved; output_head=%d and queries=%d trainable parameters',
            sum(p.numel() for p in decoder.output_head.parameters() if p.requires_grad)
            if decoder.output_head is not None else 0,
            sum(p.numel() for p in decoder.latent_queries.parameters() if p.requires_grad),
        )
    return model
