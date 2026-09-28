# Copyright 2025 the LlamaFactory team.

from types import SimpleNamespace

import pytest
from datasets import Dataset

from llamafactory.data.data_utils import merge_dataset


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_interleave_mixed_encoder_schemas():
    mimic = Dataset.from_list(
        [
            {
                "_prompt": [{"role": "user", "content": "<medgemma> report"}],
                "_encoders": {"medgemma": {"bytes": None, "path": "cxr.jpg"}},
                "_terramind": None,
            }
        ]
    )
    benx = Dataset.from_list(
        [
            {
                "_prompt": [{"role": "user", "content": "<terramind> labels"}],
                "_encoders": {
                    "terramind": {"S1GRD": {"bytes": b"s1", "path": None}, "S2L2A": {"bytes": b"s2", "path": None}}
                },
                "_terramind": {"S1GRD": {"bytes": b"s1", "path": None}},
            }
        ]
    )
    data_args = SimpleNamespace(
        mix_strategy="interleave_over",
        interleave_probs=[0.5, 0.5],
        streaming=False,
    )
    merged = merge_dataset([mimic, benx], data_args, seed=0)
    rows = list(merged)
    assert len(rows) >= 2
    names = {tuple((row["_encoders"] or {}).keys()) for row in rows}
    assert ("medgemma",) in names
    assert ("terramind",) in names


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_interleave_streaming_allows_pil_and_nested_encoders():
    pil = pytest.importorskip("PIL.Image")
    image = pil.Image.new("RGB", (8, 8), (1, 2, 3))
    mimic = Dataset.from_list(
        [
            {
                "_prompt": [{"role": "user", "content": "<medgemma> report"}],
                "_encoders": {"medgemma": image},
                "_terramind": None,
            }
        ]
    ).to_iterable_dataset()
    benx = Dataset.from_list(
        [
            {
                "_prompt": [{"role": "user", "content": "<terramind> labels"}],
                "_encoders": {"terramind": {"S1GRD": b"s1", "S2L2A": b"s2"}},
                "_terramind": {"S1GRD": b"s1"},
            }
        ]
    ).to_iterable_dataset()
    data_args = SimpleNamespace(
        mix_strategy="interleave_over",
        interleave_probs=[0.5, 0.5],
        streaming=True,
    )
    merged = merge_dataset([mimic, benx], data_args, seed=0)
    rows = []
    for row, _ in zip(merged, range(4)):
        rows.append(row)
    assert len(rows) == 4
    kinds = {tuple((row["_encoders"] or {}).keys()) for row in rows}
    assert ("medgemma",) in kinds
    assert ("terramind",) in kinds
