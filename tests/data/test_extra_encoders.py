# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch

from llamafactory.data.extra_encoders import (
    encoder_columns_from_config,
    expand_encoder_markers,
    incomplete_encoder_payloads,
    pop_encoder_samples,
    prepare_encoder_messages,
    samples_by_encoder_name,
)
from llamafactory.data.parser import DatasetAttr


class _VocabTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


SPECS = [
    {
        "name": "terramind",
        "marker": "<terramind>",
        "placeholder_token": "<|terramind|>",
        "num_tokens": 128,
        "type": "terramind",
        "extra": {"modalities": ["S1GRD", "S2L2A"]},
    },
    {
        "name": "medgemma",
        "marker": "<medgemma>",
        "placeholder_token": "<|medgemma|>",
        "num_tokens": 256,
        "type": "medgemma",
    },
]

MASK_CAPABLE_TERRAMIND_SPEC = {
    **SPECS[0],
    "extra": {
        **SPECS[0]["extra"],
        "min_modalities": 1,
    },
}


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_encoder_columns_skip_standard_media_fields():
    extra = encoder_columns_from_config(
        {
            "messages": "messages",
            "images": "images",
            "terramind": "terramind",
            "medgemma": "cxr",
        }
    )
    assert extra == {"terramind": "terramind", "medgemma": "cxr"}


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_dataset_attr_join_collects_encoder_columns():
    attr = DatasetAttr("file", "smoke")
    attr.join(
        {
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
                "terramind": "terramind",
                "medgemma": "medgemma",
            },
        }
    )
    assert attr.images == "images"
    assert attr.encoder_columns == {"terramind": "terramind", "medgemma": "medgemma"}
    assert attr.terramind == "terramind"


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_prepare_encoder_messages_expands_medgemma_marker():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [{"role": "user", "content": "<medgemma>\nGenerate a radiology report."}]
    out = prepare_encoder_messages(messages, {"medgemma": "cxr.png"}, SPECS, tokenizer)
    assert out[0]["content"].count("<|medgemma|>") == 256
    assert "<medgemma>" not in out[0]["content"]
    assert messages[0]["content"].startswith("<medgemma>")


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_image_placeholder_is_left_to_native_omni_vision():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [
        {"role": "user", "content": "<image>\n<medgemma>\nCompare the photo and the X-ray."}
    ]
    out = prepare_encoder_messages(messages, {"medgemma": "cxr.png"}, SPECS, tokenizer)
    assert out[0]["content"].startswith("<image>")
    assert out[0]["content"].count("<|medgemma|>") == 256


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_two_medgemma_payloads_expand_one_block_per_marker():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [
        {
            "role": "user",
            "content": "<medgemma><medgemma>\nHas the AV block resolved?",
        }
    ]
    out = prepare_encoder_messages(
        messages,
        {"medgemma": ["ecg_prior.png", "ecg_recent.png"]},
        SPECS,
        tokenizer,
    )
    assert out[0]["content"].count("<|medgemma|>") == 512
    assert "<medgemma>" not in out[0]["content"]


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_marker_count_must_match_payload_count():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [{"role": "user", "content": "<medgemma>\nCompare the ECGs."}]
    with pytest.raises(ValueError, match="per payload"):
        prepare_encoder_messages(
            messages,
            {"medgemma": ["ecg_prior.png", "ecg_recent.png"]},
            SPECS,
            tokenizer,
        )


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_two_terramind_payloads_expand_one_block_per_marker():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [
        {
            "role": "user",
            "content": "<terramind> earlier and <terramind> now. What changed?",
        }
    ]
    out = expand_encoder_markers(
        messages,
        {
            "terramind": [
                {"S1GRD": "a_s1.npy", "S2L2A": "a_s2.npy"},
                {"S1GRD": "b_s1.npy", "S2L2A": "b_s2.npy"},
            ]
        },
        SPECS,
        tokenizer,
    )
    assert out[0]["content"].count("<|terramind|>") == 256
    assert "<terramind>" not in out[0]["content"]


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_incomplete_payload_inside_list_is_detected():
    incomplete = incomplete_encoder_payloads(
        {
            "terramind": [
                {"S1GRD": "a_s1.npy", "S2L2A": "a_s2.npy"},
                {"S2L2A": "b_s2.npy"},
            ]
        },
        SPECS,
    )
    assert incomplete == {"terramind": ["S1GRD"]}


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_encoder_payload_without_its_marker_is_rejected():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [{"role": "user", "content": "<image>\nGenerate a radiology report."}]
    with pytest.raises(ValueError, match="exactly one"):
        prepare_encoder_messages(messages, {"medgemma": "cxr.png"}, SPECS, tokenizer)


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_incomplete_terramind_payload_is_detected():
    incomplete = incomplete_encoder_payloads(
        {"terramind": {"S2L2A": "s2.npy"}},
        SPECS,
    )
    assert incomplete == {"terramind": ["S1GRD"]}
    null_s1 = incomplete_encoder_payloads(
        {"terramind": {"S1GRD": None, "S2L2A": "s2.npy"}},
        SPECS,
    )
    assert null_s1 == {"terramind": ["S1GRD"]}
    mixed = incomplete_encoder_payloads(
        {
            "terramind": {"S1GRD": "s1.npy", "S2L2A": "s2.npy"},
            "medgemma": "cxr.png",
        },
        SPECS,
    )
    assert mixed == {}
    medgemma_only = incomplete_encoder_payloads({"medgemma": "cxr.png"}, SPECS)
    assert medgemma_only == {}


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_mask_capable_terramind_accepts_either_modality():
    s1_only = incomplete_encoder_payloads(
        {"terramind": {"S1GRD": "s1.npy"}},
        [MASK_CAPABLE_TERRAMIND_SPEC],
    )
    s2_only = incomplete_encoder_payloads(
        {"terramind": {"S2L2A": "s2.npy"}},
        [MASK_CAPABLE_TERRAMIND_SPEC],
    )
    empty = incomplete_encoder_payloads(
        {"terramind": {"S1GRD": None, "S2L2A": None}},
        [MASK_CAPABLE_TERRAMIND_SPEC],
    )

    assert s1_only == {}
    assert s2_only == {}
    assert empty == {"terramind": ["S1GRD", "S2L2A"]}


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_mask_capable_terramind_expands_s2_only_marker():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1})
    messages = [{"role": "user", "content": "<terramind>\nDescribe this scene."}]

    out = expand_encoder_markers(
        messages,
        {"terramind": {"S2L2A": "s2.npy"}},
        [MASK_CAPABLE_TERRAMIND_SPEC],
        tokenizer,
    )

    assert out[0]["content"].count("<|terramind|>") == 128
    assert "<terramind>" not in out[0]["content"]


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_expand_encoder_markers_rejects_incomplete_terramind():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    with pytest.raises(ValueError, match="missing required modalities"):
        expand_encoder_markers(
            [{"role": "user", "content": "<terramind>\nDescribe this pair."}],
            {"terramind": {"S2L2A": "s2.npy"}},
            SPECS,
            tokenizer,
        )


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_expand_encoder_markers_uses_spec_num_tokens():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [{"role": "user", "content": "<terramind>\nDescribe this pair."}]
    out = expand_encoder_markers(
        messages,
        {"terramind": {"S1GRD": "s1.npy", "S2L2A": "s2.npy"}},
        SPECS,
        tokenizer,
    )
    assert out[0]["content"].startswith("<|terramind|>" * 128)
    assert "<terramind>" not in out[0]["content"]
    assert "<|medgemma|>" not in out[0]["content"]


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_expand_encoder_markers_mixed_samples_only_replace_present():
    tokenizer = _VocabTokenizer({"<|terramind|>": 1, "<|medgemma|>": 2})
    messages = [{"role": "user", "content": "<medgemma>\nReport."}]
    out = expand_encoder_markers(
        messages,
        {"medgemma": "cxr.png"},
        SPECS,
        tokenizer,
    )
    assert out[0]["content"].count("<|medgemma|>") == 256
    assert "<medgemma>" not in out[0]["content"]


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_pop_encoder_samples_and_group_by_name():
    features = [
        {"encoders": {"terramind": {"S2L2A": "a.npy"}}, "input_ids": [1]},
        {"encoders": {"medgemma": "cxr.png"}, "input_ids": [2]},
        {"terramind": {"S2L2A": "b.npy"}, "input_ids": [3]},
    ]
    batch = [pop_encoder_samples(feature) for feature in features]
    grouped = samples_by_encoder_name(batch, SPECS)
    assert grouped["terramind"][0] == {"S2L2A": "a.npy"}
    assert grouped["terramind"][1] is None
    assert grouped["terramind"][2] == {"S2L2A": "b.npy"}
    assert grouped["medgemma"][1] == "cxr.png"
    assert "terramind" not in features[2]


VIDEO_SPEC = {
    "name": "omniavsr_video",
    "type": "omniavsr",
    "marker": "<omniavsr_video>",
    "placeholder_token": "<|omniavsr_video|>",
    "num_tokens": 80,
    "extra": {"stream": "video"},
}


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_lipread_marker_expands_to_omniavsr_video_placeholders():
    tokenizer = _VocabTokenizer({"<|omniavsr_video|>": 3})
    messages = [
        {"role": "user", "content": "<audio><lipread>Transcribe the speech from the video."}
    ]
    out = expand_encoder_markers(
        messages,
        {"lipread": "clip_lips.mp4"},
        [VIDEO_SPEC],
        tokenizer,
    )
    assert out[0]["content"].startswith("<audio>")
    assert out[0]["content"].count("<|omniavsr_video|>") == 80
    assert "<lipread>" not in out[0]["content"]


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_lipread_column_groups_under_omniavsr_video():
    grouped = samples_by_encoder_name(
        [{"lipread": "clip_lips.mp4"}, {"omniavsr_video": "other.npy"}, {}],
        [VIDEO_SPEC],
    )
    assert grouped["omniavsr_video"] == ["clip_lips.mp4", "other.npy", None]


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_cast_encoder_inputs_keeps_batch_indices_long():
    from llamafactory.data.extra_encoders import cast_encoder_inputs

    payload = {
        "terramind": {
            "batch_indices": torch.tensor([0, 2], dtype=torch.long),
            "inputs": {"S2L2A": torch.ones(2, 12, 4, 4, dtype=torch.float32)},
        }
    }
    out = cast_encoder_inputs(payload, torch.bfloat16)
    assert out["terramind"]["batch_indices"].dtype == torch.long
    assert out["terramind"]["inputs"]["S2L2A"].dtype == torch.bfloat16
