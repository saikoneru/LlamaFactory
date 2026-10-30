# Copyright 2025 the LlamaFactory team.

import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset, IterableDataset

from llamafactory.data.converter import align_dataset
from llamafactory.data.data_utils import _PythonMix, merge_dataset, requires_python_mix


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
    # interleave_over restarts exhausted sources, so the Python mix never ends.
    rows = [row for row, _ in zip(merged, range(8))]
    assert len(rows) == 8
    names = {tuple((row["_encoders"] or {}).keys()) for row in rows}
    assert ("medgemma",) in names
    assert ("terramind",) in names


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_interleave_streaming_allows_pil_and_nested_encoders():
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (8, 8), (1, 2, 3))
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


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_requires_python_mix_only_for_differing_encoders():
    assert requires_python_mix(None)
    assert requires_python_mix([frozenset({"medgemma"}), frozenset({"terramind"})])
    assert requires_python_mix([frozenset(), frozenset({"medgemma"})])
    assert not requires_python_mix([frozenset({"medgemma"}), frozenset({"medgemma"})])
    assert not requires_python_mix([frozenset(), frozenset()])


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_same_encoder_schema_keeps_hf_iterable_dataset():
    def make(name: str) -> "IterableDataset":
        return Dataset.from_list(
            [
                {
                    "_prompt": [{"role": "user", "content": f"<medgemma> {name}"}],
                    "_encoders": {"medgemma": {"bytes": None, "path": f"{name}.jpg"}},
                    "_terramind": None,
                }
            ]
        ).to_iterable_dataset()

    data_args = SimpleNamespace(
        mix_strategy="interleave_over",
        interleave_probs=[0.5, 0.5],
        eval_interleave_probs=None,
        streaming=True,
    )
    merged = merge_dataset(
        [make("a"), make("b")],
        data_args,
        seed=0,
        encoder_schemas=[frozenset({"medgemma"}), frozenset({"medgemma"})],
    )
    assert isinstance(merged, IterableDataset)
    # HuggingFace all_exhausted stops once both one-row sources have been read.
    rows = [row for row, _ in zip(merged, range(4))]
    assert sorted(row["_prompt"][0]["content"] for row in rows) == ["<medgemma> a", "<medgemma> b"]


def _mixed_encoder_pipeline(streaming: bool = True):
    pil = pytest.importorskip("PIL.Image")

    def build(rows):
        dataset = Dataset.from_list(rows)
        if not streaming:
            return dataset

        # A per-source map over one multi-row Arrow table is where HF's own resume state drops rows.
        return dataset.to_iterable_dataset().map(lambda example: {"id": example["id"]})

    medgemma = build(
        [
            {"id": f"medgemma-{index}", "_encoders": {"medgemma": pil.new("RGB", (4, 4), (index, 0, 0))}}
            for index in range(7)
        ]
    )
    terramind = build(
        [
            {"id": f"terramind-{index}", "_encoders": {"terramind": {"S2L2A": [[float(index)] * 3] * 2}}}
            for index in range(5)
        ]
    )
    autoavsr = build(
        [{"id": f"autoavsr-{index}", "_encoders": {"autoavsr_video": bytes([index]) * 4}} for index in range(3)]
    )
    data_args = SimpleNamespace(
        mix_strategy="interleave_over",
        interleave_probs=[0.4, 0.3, 0.3],
        eval_interleave_probs=None,
        streaming=streaming,
    )
    merged = merge_dataset(
        [medgemma, terramind, autoavsr],
        data_args,
        seed=7,
        encoder_schemas=[frozenset({"medgemma"}), frozenset({"terramind"}), frozenset({"autoavsr_video"})],
    )
    assert isinstance(merged, _PythonMix)

    def preprocess(batch):
        kept = [index for index, row_id in enumerate(batch["id"]) if not row_id.endswith("-2")]
        return {"row_id": [batch["id"][index] for index in kept]}

    return (
        merged.shuffle(buffer_size=4, seed=11)
        .map(preprocess, batched=True, batch_size=5, remove_columns=["id", "_encoders"])
        .take(40)
    )


def _roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
@pytest.mark.parametrize("streaming", [True, False])
def test_python_mix_resumes_exactly(streaming):
    expected = [row["row_id"] for row in _mixed_encoder_pipeline(streaming)]
    assert len(expected) == 40
    assert {row_id.split("-")[0] for row_id in expected} == {"medgemma", "terramind", "autoavsr"}

    for cut in (0, 1, 3, 7, 12, 23, 39):
        pipeline = _mixed_encoder_pipeline(streaming)
        iterator = iter(pipeline)
        before = [next(iterator)["row_id"] for _ in range(cut)]
        state = _roundtrip(pipeline.state_dict())

        resumed = _mixed_encoder_pipeline(streaming)
        resumed.load_state_dict(state)
        after = [row["row_id"] for row in resumed]
        assert before + after == expected, f"cut={cut}"


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("cut", [3, 4])
def test_python_mix_resumes_through_stateful_dataloader(num_workers, cut):
    from torchdata.stateful_dataloader import StatefulDataLoader

    def make_loader():
        pipeline = _mixed_encoder_pipeline()
        next(iter(pipeline))  # the loader previews one example before training
        return StatefulDataLoader(
            pipeline,
            batch_size=2,
            num_workers=num_workers,
            collate_fn=lambda rows: [row["row_id"] for row in rows],
            snapshot_every_n_steps=3,
        )

    expected = list(make_loader())
    assert len(expected) == 20

    loader = make_loader()
    iterator = iter(loader)
    before = [next(iterator) for _ in range(cut)]
    state = _roundtrip(loader.state_dict())
    del iterator, loader

    resumed = make_loader()
    resumed.load_state_dict(state)
    assert before + list(resumed) == expected


def _aligned_mix(converted):
    pil = pytest.importorskip("PIL.Image")

    def convert(example):
        converted.append(example["id"])
        return {"row_id": example["id"]}

    def source(name, count):
        rows = [{"id": f"{name}-{index}", "image": pil.new("RGB", (4, 4), (index, 0, 0))} for index in range(count)]
        dataset = Dataset.from_list(rows).to_iterable_dataset()
        with patch("llamafactory.data.converter.get_dataset_converter", return_value=convert):
            return align_dataset(dataset, SimpleNamespace(formatting="sharegpt"), SimpleNamespace(streaming=True), None)

    data_args = SimpleNamespace(
        mix_strategy="interleave_over",
        interleave_probs=[0.6, 0.4],
        eval_interleave_probs=None,
        streaming=True,
    )
    merged = merge_dataset(
        [source("medgemma", 9), source("terramind", 6)],
        data_args,
        seed=3,
        encoder_schemas=[frozenset({"medgemma"}), frozenset({"terramind"})],
    )
    assert isinstance(merged, _PythonMix)
    return merged.take(30)


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
@pytest.mark.parametrize("cut", [1, 5, 11, 20])
def test_python_mix_resume_skips_raw_rows_without_converting(cut):
    converted = []
    expected = [row["row_id"] for row in _aligned_mix(converted)]
    assert len(expected) == 30

    converted.clear()
    pipeline = _aligned_mix(converted)
    iterator = iter(pipeline)
    before = [next(iterator)["row_id"] for _ in range(cut)]
    state = _roundtrip(pipeline.state_dict())

    converted.clear()
    resumed = _aligned_mix(converted)
    resumed.load_state_dict(state)
    after = [row["row_id"] for row in resumed]
    assert before + after == expected
    assert converted == after, "rows skipped on resume must not reach the converter"


@pytest.mark.runs_on(["cpu", "mps", "cuda"])
def test_python_mix_shards_rows_across_ranks():
    expected = [row["row_id"] for row in _mixed_encoder_pipeline()]
    shards = [
        [row["row_id"] for row in _mixed_encoder_pipeline().shard_by_node(rank=rank, world_size=2)]
        for rank in range(2)
    ]
    assert shards[0] == expected[0::2]
    assert shards[1] == expected[1::2]
