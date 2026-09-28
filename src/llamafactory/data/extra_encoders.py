# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generic extra-encoder columns, matching native images/audios handling.

Dataset ``columns`` keys that are not ShareGPT/Alpaca/media fields are treated
as extra encoder names. Marker expansion and collation read encoder specs from
the composite processor (``processor.encoder_specs``), not from TerraMind-only
CLI flags.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import torch


STANDARD_COLUMN_KEYS = {
    "prompt",
    "query",
    "response",
    "history",
    "messages",
    "system",
    "tools",
    "images",
    "videos",
    "audios",
    "chosen",
    "rejected",
    "kto_tag",
}

COMPOSITE_ENCODER_MODEL_TYPES = {
    "qwen2_5_omni_composite",
    "qwen2_5_omni_terramind",
}

OMNI_MODEL_TYPES = {
    "qwen2_5_omni_thinker",
    "qwen3_omni_moe_thinker",
    *COMPOSITE_ENCODER_MODEL_TYPES,
}

# Keep in sync with omni_composite.encoders_omniavsr. LipCrops prompts/columns
# use <lipread>; the composite encoder is still named omniavsr_video.
LIPREAD_MARKER = "<lipread>"
LIPREAD_COLUMN = "lipread"


def _omniavsr_stream(spec: dict[str, Any]) -> str:
    extra = spec.get("extra") if isinstance(spec.get("extra"), dict) else {}
    return str(extra.get("stream") or "")


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def spec_prompt_markers(spec: dict[str, Any]) -> list[str]:
    name = str(spec.get("name") or "")
    marker = str(spec.get("marker") or (f"<{name}>" if name else ""))
    aliases: list[str] = []
    aliases.extend(str(item) for item in (spec.get("marker_aliases") or []))
    extra = spec.get("extra") if isinstance(spec.get("extra"), dict) else {}
    aliases.extend(str(item) for item in (extra.get("marker_aliases") or []))
    if spec.get("type") == "omniavsr" and _omniavsr_stream(spec) == "video":
        aliases.append(LIPREAD_MARKER)
    return _unique([marker, *aliases])


def spec_column_names(spec: dict[str, Any]) -> list[str]:
    name = str(spec.get("name") or "")
    aliases: list[str] = []
    aliases.extend(str(item) for item in (spec.get("column_aliases") or []))
    extra = spec.get("extra") if isinstance(spec.get("extra"), dict) else {}
    aliases.extend(str(item) for item in (extra.get("column_aliases") or []))
    if spec.get("type") == "omniavsr" and _omniavsr_stream(spec) == "video":
        aliases.append(LIPREAD_COLUMN)
    return _unique([name, *aliases])


def remap_encoder_samples(
    encoder_samples: dict[str, Any] | None,
    specs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    remapped = dict(encoder_samples or {})
    for spec in specs or []:
        name = spec.get("name")
        if not name or remapped.get(name) is not None:
            continue
        for alias in spec_column_names(spec):
            if alias == name:
                continue
            if remapped.get(alias) is not None:
                remapped[name] = remapped[alias]
                break
    return remapped


def encoder_columns_from_config(columns: dict[str, Any] | None) -> dict[str, str]:
    """Return extra encoder name -> dataset field, ignoring standard columns."""
    extra: dict[str, str] = {}
    for key, value in (columns or {}).items():
        if key in STANDARD_COLUMN_KEYS or not value:
            continue
        extra[str(key)] = str(value)
    return extra


def get_encoder_specs(processor: Any = None, model: Any = None) -> list[dict[str, Any]]:
    specs = getattr(processor, "encoder_specs", None)
    if specs:
        return list(specs)
    config = getattr(model, "config", None)
    specs = getattr(config, "encoders", None)
    if specs:
        return list(specs)
    return []


def row_encoder_samples(examples: dict[str, Any], index: int) -> dict[str, Any]:
    """Read one aligned row's extra-encoder payloads."""
    encoders = examples.get("_encoders")
    if encoders is not None:
        return dict(encoders[index] or {})
    terramind_column = examples.get("_terramind")
    if terramind_column is None:
        return {}
    terramind = terramind_column[index]
    return {"terramind": terramind} if terramind is not None else {}


def spec_required_modalities(spec: dict[str, Any]) -> list[str]:
    """Modalities understood by this encoder, in canonical order."""
    extra = spec.get("extra") if isinstance(spec.get("extra"), dict) else {}
    modalities = extra.get("modalities") or spec.get("modalities") or []
    return [str(name) for name in modalities]


def spec_min_modalities(spec: dict[str, Any]) -> int:
    """Minimum number of configured modalities required in one sample.

    Encoders are strict by default. Mask-capable encoders can advertise
    ``extra.min_modalities`` (for example, TerraMind accepts either S1GRD or
    S2L2A while preserving fixed token slots for both).
    """
    modalities = spec_required_modalities(spec)
    if not modalities:
        return 0
    extra = spec.get("extra") if isinstance(spec.get("extra"), dict) else {}
    minimum = int(extra.get("min_modalities", len(modalities)))
    if minimum < 1 or minimum > len(modalities):
        raise ValueError(
            f"Encoder {spec.get('name')!r} min_modalities must be between 1 "
            f"and {len(modalities)}, got {minimum}."
        )
    return minimum


def missing_required_modalities(sample: Any, spec: dict[str, Any]) -> list[str]:
    """Return missing keys only when too few modalities are present."""
    required = spec_required_modalities(spec)
    if not required:
        return []
    if not isinstance(sample, dict):
        return list(required)
    missing = [name for name in required if sample.get(name) is None]
    present_count = len(required) - len(missing)
    return [] if present_count >= spec_min_modalities(spec) else missing


def incomplete_encoder_payloads(
    encoder_samples: dict[str, Any] | None,
    specs: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    """Encoder name -> missing keys for payloads that cannot be collated.

    A missing encoder column is fine in mixed batches. A present multimodal
    payload must meet its encoder's ``min_modalities`` contract; strict
    encoders default to requiring every configured modality.
    """
    incomplete: dict[str, list[str]] = {}
    encoder_samples = remap_encoder_samples(encoder_samples, specs)
    for spec in specs or []:
        name = spec.get("name")
        if not name:
            continue
        sample = encoder_samples.get(name)
        if sample is None:
            continue
        missing = missing_required_modalities(sample, spec)
        if missing:
            incomplete[str(name)] = missing
    return incomplete


def _media_paths(value: Any) -> Iterator[str]:
    """Yield the filesystem paths one encoder payload refers to."""
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        if os.sep in text:
            yield text
    elif isinstance(value, dict):
        for item in value.values():
            yield from _media_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _media_paths(item)


def unreadable_encoder_media(
    encoder_samples: dict[str, Any] | None,
    specs: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    """Encoder name -> payload paths that are missing or zero-byte.

    LipCrops ships a few empty .flac / .mp4 files. Extra-encoder media is only
    decoded in the collator, where the row already carries its placeholder
    tokens, so a failure there kills the whole job. Drop the row here instead.
    """
    unreadable: dict[str, list[str]] = {}
    encoder_samples = remap_encoder_samples(encoder_samples, specs)
    for spec in specs or []:
        name = spec.get("name")
        if not name:
            continue
        sample = encoder_samples.get(name)
        if sample is None:
            continue
        bad = [
            path
            for path in _media_paths(sample)
            if not os.path.isfile(path) or os.path.getsize(path) == 0
        ]
        if bad:
            unreadable[str(name)] = bad
    return unreadable


def prepare_encoder_messages(
    messages: list[dict[str, str]],
    encoder_samples: dict[str, Any] | None,
    specs: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, str]]:
    """Expand each encoder's marker from its spec.

    Every dataset states the marker of the encoder that will run, so a row
    using an extra encoder says ``<medgemma>`` / ``<terramind>``. ``<image>``
    always stays with native Omni vision.
    """
    if specs:
        return expand_encoder_markers(messages, encoder_samples, specs, tokenizer)
    return messages


def expand_encoder_markers(
    messages: list[dict[str, str]],
    encoder_samples: dict[str, Any] | None,
    specs: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, str]]:
    """Replace each encoder marker with that encoder's placeholder token * N."""
    messages = [dict(message) for message in messages]
    encoder_samples = remap_encoder_samples(encoder_samples, specs)
    if not specs:
        return messages

    vocab = tokenizer.get_vocab()
    for spec in specs:
        name = spec["name"]
        sample = encoder_samples.get(name)
        markers = spec_prompt_markers(spec)
        placeholder = spec["placeholder_token"]
        num_tokens = int(spec["num_tokens"])
        if placeholder not in vocab:
            raise ValueError(
                f"Encoder token {placeholder!r} is missing from the tokenizer. "
                "Load the tokenizer saved with the composite checkpoint."
            )

        marker_count = sum(
            message["content"].count(marker)
            for message in messages
            for marker in markers
        )
        if sample is None:
            if marker_count:
                raise ValueError(
                    f"Messages contain marker {markers[0]!r} but no {name!r} sample."
                )
            replacement = ""
        else:
            missing = missing_required_modalities(sample, spec)
            if missing:
                raise ValueError(
                    f"{name} sample is missing required modalities {missing}."
                )
            if marker_count != 1:
                raise ValueError(
                    f"A {name!r} example must contain exactly one of {markers} "
                    f"markers, found {marker_count}."
                )
            replacement = placeholder * num_tokens

        for message in messages:
            for marker in markers:
                if marker in message["content"]:
                    message["content"] = message["content"].replace(marker, replacement)

    return messages


def pop_encoder_samples(feature: dict[str, Any]) -> dict[str, Any]:
    """Pull extra-encoder payloads off a collator feature."""
    encoders = feature.pop("encoders", None)
    if encoders is None:
        encoders = {}
    else:
        encoders = dict(encoders)
    if "terramind" in feature:
        encoders.setdefault("terramind", feature.pop("terramind"))
    else:
        feature.pop("terramind", None)
    return encoders


def samples_by_encoder_name(
    batch_samples: list[dict[str, Any]],
    specs: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    names = [spec["name"] for spec in specs]
    grouped: dict[str, list[Any]] = {name: [] for name in names}
    for sample in batch_samples:
        remapped = remap_encoder_samples(sample or {}, specs)
        for name in names:
            grouped[name].append(remapped.get(name))
    return grouped


def has_encoder_payload(encoder_samples: dict[str, Any] | None) -> bool:
    return any(value is not None for value in (encoder_samples or {}).values())


def cast_encoder_inputs(value: Any, dtype: torch.dtype) -> Any:
    """Cast floating tensors in ``encoder_inputs`` to the training dtype."""
    if torch.is_tensor(value):
        if torch.is_floating_point(value):
            return value.to(dtype)
        return value
    if isinstance(value, dict):
        return {key: cast_encoder_inputs(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_encoder_inputs(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_encoder_inputs(item, dtype) for item in value)
    return value
