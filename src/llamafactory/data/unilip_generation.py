"""ShareGPT <unilip> image targets and shape-aware SFT collation.

HEIGHTxWIDTH throughout. Target media is independent of input `images`.
The converter stores Arrow-safe encoded image bytes, never PIL objects or GPU tensors.
"""
from __future__ import annotations

import io
import logging
import math
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from .collator import SFTDataCollatorWith4DAttentionMask

LOG = logging.getLogger(__name__)
RES_PATTERN = re.compile(r"<res_(\d+)x(\d+)>")
TRIGGER = "<unilip>"
DEFAULT_BUCKETS = "448x448,448x336,336x448,560x336,336x560,504x392,392x504"


def parse_buckets(value: str | list | None) -> list[tuple[int, int]]:
    if not value:
        value = DEFAULT_BUCKETS
    if isinstance(value, str):
        items = [item.strip().removeprefix("<res_").removesuffix(">")
                 for item in value.split(",") if item.strip()]
    else:
        items = list(value)
    buckets = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            h, w = map(int, item)
        else:
            h, w = map(int, str(item).lower().split("x"))
        # Buckets describe OUTPUT pixels. Codec grids may use 28 pixels per
        # latent cell (legacy) or the DC-AE decoder's native 32-pixel output.
        # The model validates its own pixel/latent geometry.
        if h <= 0 or w <= 0:
            raise ValueError(f"UniLIP bucket {(h,w)} must contain positive pixel dimensions")
        if (h, w) not in buckets:
            buckets.append((h, w))
    if not buckets:
        raise ValueError("UniLIP requires at least one resolution bucket")
    return buckets


def closest_bucket(height: int, width: int, buckets: list[tuple[int,int]]) -> tuple[int,int]:
    aspect = width / height
    area = width * height
    return min(buckets, key=lambda hw: (
        abs(math.log(aspect / (hw[1] / hw[0]))) +
        0.12 * abs(math.log(area / (hw[0] * hw[1]))),
        hw[0] * hw[1], hw[0], hw[1]
    ))


def _open_image(value: Any, media_dir: str | None):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("UniLIP currently supports one target image per assistant response")
        value = value[0]
    if isinstance(value, Image.Image):
        return value.copy(), "<PIL>"
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])), value.get("path") or "<bytes>"
        value = value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)), "<bytes>"
    if not isinstance(value, str) or not value:
        raise ValueError("UniLIP target_image needs a readable path or encoded image")
    path = os.path.join(media_dir, value) if media_dir and not os.path.isabs(value) else value
    if not os.path.isfile(path):
        raise FileNotFoundError(f"UniLIP target_image not found: {path}")
    return Image.open(path), path


def convert_sharegpt_target(
    response: list[dict[str, str]], target_image: Any,
    *, media_dir: str | None, buckets: str | list | None,
    resize_mode: str = "center_crop", warn_on_resize: bool = True,
    preserve_matching_bytes: bool = False,
) -> dict[str, Any] | None:
    """Mutate final assistant message to <res_HxW><unilip> and return encoded bytes.

    Non-image ShareGPT examples are unchanged. Input `images` remain untouched.
    """
    if not response or response[-1].get("role") != "assistant":
        if target_image is not None:
            raise ValueError("target_image requires a final assistant response")
        return None
    content = response[-1]["content"]
    marked = TRIGGER in content
    if not marked:
        if target_image is not None:
            raise ValueError("target_image provided but assistant field has no <unilip>")
        return None
    if content.count(TRIGGER) != 1 or not content.rstrip().endswith(TRIGGER):
        raise ValueError("Image generation requires exactly one final <unilip> marker")
    if target_image is None:
        raise ValueError("Assistant <unilip> response requires target_image")
    source, path = _open_image(target_image, media_dir)
    with source:
        raw = target_image.get("bytes") if isinstance(target_image, dict) else target_image
        h, w = source.height, source.width
        exact_bucket = closest_bucket(h, w, parse_buckets(buckets)) == (h, w)
        # Native-resolution WebDataset targets can keep their JPEG bytes.
        # Avoid decoding/re-encoding millions of images merely for shuffling.
        if (preserve_matching_bytes and exact_bucket and source.mode == "RGB"
                and source.getexif().get(274, 1) == 1 and isinstance(raw, (bytes, bytearray))):
            prefix = RES_PATTERN.sub("", content[:content.rfind(TRIGGER)])
            response[-1]["content"] = prefix.rstrip() + f"<res_{h}x{w}>" + TRIGGER
            return {"bytes": bytes(raw), "height": h, "width": w}
        source = ImageOps.exif_transpose(source).convert("RGB")
        original_width, original_height = source.size
        target_height, target_width = closest_bucket(
            original_height, original_width, parse_buckets(buckets)
        )
        if resize_mode == "center_crop":
            result = ImageOps.fit(source, (target_width, target_height), Image.Resampling.LANCZOS)
        elif resize_mode == "pad":
            result = ImageOps.pad(source, (target_width, target_height),
                                  Image.Resampling.LANCZOS, color=(0,0,0))
        else:
            raise ValueError(f"Unknown UniLIP resizing mode: {resize_mode!r}")
        if (original_height, original_width) != (target_height, target_width) and warn_on_resize:
            LOG.warning("UniLIP target image resized %s: HxW %dx%d -> %dx%d; mode=%s; token=<res_%dx%d>",
                        path, original_height, original_width, target_height, target_width,
                        resize_mode, target_height, target_width)
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
    prefix = content[:content.rfind(TRIGGER)]
    prefix = RES_PATTERN.sub("", prefix)
    token = f"<res_{target_height}x{target_width}>"
    response[-1]["content"] = prefix.rstrip() + token + TRIGGER
    return {"bytes": buffer.getvalue(), "height": target_height, "width": target_width}


def validate_image_labels(input_ids: list[int], labels: list[int], trigger_id: int):
    """Drop template suffix tokens so <unilip> is the last non-pad input.

    Raise if truncation cut away marker or assistant token is not supervised.
    """
    locations = [i for i, token in enumerate(input_ids) if token == trigger_id]
    if len(locations) != 1:
        raise ValueError("One image example must have exactly one tokenized <unilip> marker")
    end = locations[0]
    if labels[end] == -100:
        raise ValueError("Image trigger has masked language-model label")
    return input_ids[:end + 1], labels[:end + 1]


def _to_tensor(value: dict[str, Any] | bytes) -> torch.Tensor:
    image_bytes = value["bytes"] if isinstance(value, dict) else value
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("UniLIP expects Arrow-safe encoded image bytes")
    with Image.open(io.BytesIO(image_bytes)) as img:
        image = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


class UniLIPDataCollatorWith4DAttentionMask(SFTDataCollatorWith4DAttentionMask):
    """Preserve existing multimodal collator and group image targets by HxW."""

    @staticmethod
    def _without_target_fields(feature: dict[str, Any]) -> dict[str, Any]:
        """Remove assistant output media BEFORE native/extra-encoder collation.

        Keep original user input media and all selected encoder columns unchanged.
        Copy nested `encoders` so collating a batch never mutates shared dataset
        examples (important for streaming and StatefulDataLoader recycling).
        """
        clean = {k: v for k, v in feature.items() if k not in {"unilip_target", "target_image"}}
        payload = clean.get("encoders")
        if isinstance(payload, dict) and "target_image" in payload:
            payload = {k: v for k, v in payload.items() if k != "target_image"}
            if payload:
                clean["encoders"] = payload
            else:
                clean.pop("encoders")
        return clean

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        image_targets = [feature.get("unilip_target") for feature in features]
        text_features = [self._without_target_fields(feature) for feature in features]

        # `target_image` is the assistant output, never an input visual encoder.
        batch = super().__call__(text_features)

        grouped: dict[tuple[int,int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
        for row, target in enumerate(image_targets):
            if target is None:
                continue
            tensor = _to_tensor(target)
            h, w = tensor.shape[-2:]
            expected = (int(target["height"]), int(target["width"]))
            if (h, w) != expected:
                raise ValueError(f"UniLIP target shape {(h,w)} != saved metadata {expected}")
            grouped[(h,w)].append((row, tensor))
        batch["unilip_image_groups"] = [
            {"target_images": torch.stack([item[1] for item in rows]),
             "image_batch_indices": torch.tensor([item[0] for item in rows], dtype=torch.long)}
            for _, rows in sorted(grouped.items())
        ]
        return batch
