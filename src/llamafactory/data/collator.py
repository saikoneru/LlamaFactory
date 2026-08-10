# Copyright 2025 OpenAccess AI Collective and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy, inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, MROPE_MODELS
from ..extras.packages import is_pillow_available

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
        if model_type in ["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]:
            rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
            feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                rope_index_kwargs["audio_seqlens"] = audio_feature_lengths
            features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
            features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(dim=-1).unsqueeze(-1)
        else:
            features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

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

        if features["position_ids"].shape != expected_position_ids_shape:
            raise ValueError(
                f"Merged position_ids shape mismatch: got {features['position_ids'].shape}, expected {expected_position_ids_shape}."
            )

    # =================================================================
    # TerraMind batching helpers
    # =================================================================

    @staticmethod
    def _normalize_terramind_value(value):
        """Normalize one TerraMind sample into a tensor. None means this sample has no TerraMind input."""
        if value is None: return None
        if torch.is_tensor(value): return value.detach().cpu().float()
        if isinstance(value, np.ndarray): return torch.from_numpy(value).float()
        if isinstance(value, bytes):
            shape = (12, 120, 120)
            array = np.frombuffer(value, dtype=np.float32)
            expected_numel = int(np.prod(shape))
            if array.size != expected_numel:
                raise ValueError(f"Invalid TerraMind S2L2A byte payload: got {array.size} float32 values, expected {expected_numel}.")
            return torch.from_numpy(array.reshape(shape).copy()).float()
        return torch.as_tensor(value, dtype=torch.float32)

    @classmethod
    def _extract_terramind_sample(cls, feature):
        """Return a per-example TerraMind dictionary.
        Supported: {"terramind": {"S2L2A": tensor}} and legacy {"terramind_s2l2a": tensor}
        """
        terramind = feature.pop("terramind", None)
        legacy_s2l2a = feature.pop("terramind_s2l2a", None)
        if terramind is not None:
            if not isinstance(terramind, dict):
                raise TypeError("'terramind' must be a dict mapping modality names to values.")
            result = {modality: cls._normalize_terramind_value(value) for modality, value in terramind.items() if value is not None}
            return result if result else None
        if legacy_s2l2a is not None:
            return {"S2L2A": cls._normalize_terramind_value(legacy_s2l2a)}
        return None

    @staticmethod
    def _stack_optional_terramind_values(values, valid_indices):
        """Stack values belonging only to examples that actually contain a given TerraMind modality. All selected examples must have the same shape."""
        selected = []
        for idx in valid_indices:
            value = values[idx]
            if value is None:
                raise ValueError("Inconsistent TerraMind modality presence inside a batch of valid TerraMind examples.")
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32)
            selected.append(value.float())
        if not selected: return None
        try:
            return torch.stack(selected, dim=0)
        except RuntimeError as exc:
            shapes = [tuple(x.shape) for x in selected]
            raise ValueError(f"TerraMind samples for the same modality must have the same shape before batching. Got shapes: {shapes}") from exc

    def _build_terramind_batch(self, terramind_samples):
        """Convert list of per-sample dicts/None into terramind_mask + terramind_inputs (only valid samples)."""
        batch_size = len(terramind_samples)
        terramind_mask = torch.tensor([sample is not None and len(sample) > 0 for sample in terramind_samples], dtype=torch.bool)
        valid_indices = [i for i, sample in enumerate(terramind_samples) if sample is not None and len(sample) > 0]
        if not valid_indices: return terramind_mask, {}

        modality_names = set()
        for idx in valid_indices: modality_names.update(terramind_samples[idx].keys())

        terramind_inputs = {}
        for modality in sorted(modality_names):
            values = [terramind_samples[idx].get(modality) for idx in valid_indices]
            # If a modality is present for any TerraMind example, it must be present for all of them.
            if any(value is None for value in values):
                raise ValueError(
                    f"TerraMind modality {modality!r} is present for only some TerraMind examples in the batch. "
                    "Either provide the modality for every example that has TerraMind, or split the batch."
                )
            terramind_inputs[modality] = torch.stack([value.float() for value in values], dim=0)
        return terramind_mask, terramind_inputs

    # =================================================================
    # Main collator
    # =================================================================

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        is_moss_vl = model_type == "moss_vl"

        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        packing_params_list = []
        terramind_samples = []

        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            terramind_samples.append(self._extract_terramind_sample(feature))
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])
            packing_params_list.append(feature.pop("packing_params", None))

        terramind_mask, terramind_inputs = self._build_terramind_batch(terramind_samples)

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

        # Qwen multimodal inputs.
        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_audios, batch_imglens, batch_vidlens, batch_audlens, batch_input_ids, self.processor
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

        # Standard text padding.
        features = super().__call__(features)

        # TerraMind (attach after HF padding)
        features["terramind_mask"] = terramind_mask
        if terramind_inputs:
            features["terramind_inputs"] = terramind_inputs
        if "S2L2A" in terramind_inputs:
            features["terramind_s2l2a"] = terramind_inputs["S2L2A"]

        bsz, seq_len = features["input_ids"].shape[:2]
        is_omni = model_type in ["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]

        # RoPE
        if self.get_rope_func is not None:
            boundaries_list = [(p.get("sequence_boundaries") if p is not None else None) for p in packing_params_list]
            has_packing = any(b is not None and len(b) > 2 for b in boundaries_list)
            if has_dummy_image and has_packing:
                features["has_dummy_image"] = True
            if not has_packing:
                self._compute_rope_position_ids(features, mm_inputs)
            else:
                if is_omni:
                    raise RuntimeError("Omni models are not supported for packed sequences for now.")
                self._compute_rope_position_ids_with_packing(
                    features, mm_inputs, packing_params_list, batch_imglens, batch_vidlens, batch_audlens, has_dummy_image
                )
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
            features["attention_mask"] = None

        # TerraMind tensors are intentionally kept as model inputs. terramind_mask is boolean and is not cast.
        for key, value in features.items():
            if key == "terramind_mask": continue
            if key == "terramind_inputs":
                if isinstance(value, dict):
                    for modality, tensor in value.items():
                        if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
                            value[modality] = tensor.to(self.compute_dtype)
                continue
            if key == "terramind_s2l2a":
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    features[key] = value.to(self.compute_dtype)
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
                    "terramind": feature.get("terramind", None),
                    "terramind_s2l2a": feature.get("terramind_s2l2a", None),
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
                "terramind": feature.get("terramind", None),
                "terramind_s2l2a": feature.get("terramind_s2l2a", None),
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature.get("images", []),
                "videos": feature.get("videos", []),
                "audios": feature.get("audios", []),
                "terramind": feature.get("terramind", None),
                "terramind_s2l2a": feature.get("terramind_s2l2a", None),
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
        if "terramind_mask" in kl_batch:
            batch["kl_terramind_mask"] = kl_batch["terramind_mask"]
        if "terramind_inputs" in kl_batch:
            batch["kl_terramind_inputs"] = kl_batch["terramind_inputs"]
        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch