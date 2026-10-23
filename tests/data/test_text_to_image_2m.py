"""Exercise real tar loading, target isolation, streaming and SFT collation offline."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import IterableDataset
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.dataset_adapters.text_to_image_2m import (
    NUM_SHARDS,
    REPO_ID,
    TextToImage2MDatasetConverter,
    resolve_data_files,
)
from llamafactory.data.loader import _get_preprocessed_dataset, _load_single_dataset
from llamafactory.data.parser import DatasetAttr, get_dataset_list
from llamafactory.data.unilip_generation import UniLIPDataCollatorWith4DAttentionMask
from llamafactory.hparams import DataArguments


@pytest.fixture
def shards(tmp_path):
    paths = []
    for shard in range(2):
        path = tmp_path / f"data_{shard:06d}.tar"
        with tarfile.open(path, "w") as archive:
            for size in (512, 1024):
                key = f"sample_{shard}_{size}"
                buffer = io.BytesIO()
                Image.new("RGB", (size, size), (80 + shard * 40, 40, 20)).save(buffer, "JPEG")
                payloads = {"jpg": buffer.getvalue(), "json": json.dumps({"prompt": key}).encode()}
                for extension, value in payloads.items():
                    member = tarfile.TarInfo(f"{key}.{extension}")
                    member.size = len(value)
                    archive.addfile(member, io.BytesIO(value))
        paths.append(str(path))
    return paths


def load_targets(tmp_path, shards, streaming):
    attr = get_dataset_list(
        ["test"],
        {
            "test": {
                "webdataset_url": REPO_ID,
                "formatting": "text_to_image_2m",
                "data_files": shards,
            }
        },
    )[0]
    data = DataArguments(
        dataset="test",
        dataset_dir=str(tmp_path),
        streaming=streaming,
        unilip_resolution_buckets="512x512,1024x1024",
        unilip_warn_on_resize=False,
        preprocessing_num_workers=1,
        preprocessing_batch_size=2,
    )
    training = SimpleNamespace(
        local_process_index=0,
        should_log=False,
        predict_with_generate=False,
        world_size=1,
        dataloader_num_workers=0,
        use_stateful_dataloader=False,
    )
    model = SimpleNamespace(cache_dir=str(tmp_path / "cache"), hf_hub_token=None)
    return _load_single_dataset(attr, model, data, training), data, training


@pytest.mark.parametrize("streaming", [True, False])
def test_tar_to_target_preserves_resolution_and_never_exposes_input_image(tmp_path, shards, streaming):
    dataset, _, _ = load_targets(tmp_path, shards, streaming)
    assert isinstance(dataset, IterableDataset) == streaming
    assert dataset.features is not None
    rows = list(dataset)
    assert len(rows) == 4
    for row in rows:
        target = row["_unilip_target"]
        assert target["bytes"].startswith(b"\xff\xd8")  # Preserve the original JPEG, avoiding PNG re-encoding.
        size = target["height"]
        assert size in (512, 1024) and target["width"] == size
        assert Image.open(io.BytesIO(target["bytes"])).size == (size, size)
        assert row["_response"] == [{"role": "assistant", "content": f"<res_{size}x{size}><unilip>"}]
        assert row["_images"] is None and row["_encoders"] == {}
        assert "<image>" not in row["_prompt"][0]["content"]
        caption = row["_prompt"][0]["content"]
        assert caption.startswith("sample_") and caption.endswith(f"_{size}")


@pytest.mark.parametrize("known_features", [True, False])
def test_streaming_tokenization_and_mixed_resolution_collation(tmp_path, shards, known_features):
    dataset, data, training = load_targets(tmp_path, shards, True)
    if not known_features:
        dataset = dataset.map(lambda row: row)
        assert dataset.features is None
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<unilip>", "<res_512x512>", "<res_1024x1024>"]})
    data.template = "qwen"
    template = get_template_and_fix_tokenizer(tokenizer, data)
    dataset = dataset.shuffle(seed=42, buffer_size=2)
    processed = _get_preprocessed_dataset(dataset, data, training, "sft", template, tokenizer)
    rows = list(processed)
    trigger = tokenizer.convert_tokens_to_ids("<unilip>")
    for row in rows:
        assert row["input_ids"][-1] == trigger and row["labels"][-1] == trigger
        size = row["unilip_target"]["height"]
        assert row["input_ids"][-2] == tokenizer.convert_tokens_to_ids(f"<res_{size}x{size}>")
        assert row["images"] == []
    batch = UniLIPDataCollatorWith4DAttentionMask(tokenizer=tokenizer, template=template)(rows)
    groups = batch["unilip_image_groups"]
    assert [tuple(g["target_images"].shape) for g in groups] == [(2, 3, 512, 512), (2, 3, 1024, 1024)]
    assert sorted(torch.cat([g["image_batch_indices"] for g in groups]).tolist()) == list(range(4))
    assert "pixel_values" not in batch


def test_streaming_shards_partition_without_duplicate_targets(tmp_path, shards):
    from datasets.distributed import split_dataset_by_node

    dataset, _, _ = load_targets(tmp_path, shards, True)
    parts = [split_dataset_by_node(dataset, rank=rank, world_size=2) for rank in range(2)]
    keys = [{row["_prompt"][0]["content"] for row in part} for part in parts]
    assert len(keys[0]) == len(keys[1]) == 2
    assert keys[0].isdisjoint(keys[1])


def test_default_urls_and_staged_globs(monkeypatch, tmp_path):
    attr = DatasetAttr("webdataset", REPO_ID)
    urls = resolve_data_files(attr, str(tmp_path))
    assert len(urls) == NUM_SHARDS and urls[0].endswith("data_000000.tar")
    assert urls[-2].endswith("data_000045.tar")
    assert urls[-1].endswith("data_1024_10K/data_000000.tar")
    monkeypatch.setenv("T2I_2M_ROOT", str(tmp_path))
    attr.data_files = {"train": "${T2I_2M_ROOT}/data_512_2M/*.tar"}
    assert resolve_data_files(attr, "ignored") == {"train": f"{tmp_path}/data_512_2M/*.tar"}
    attr.data_files = "shards/*.tar"
    assert resolve_data_files(attr, str(tmp_path)) == f"{tmp_path}/shards/*.tar"


@pytest.mark.parametrize("metadata", [{}, {"prompt": ""}, {"prompt": None}])
def test_invalid_caption_reports_sample_key(metadata):
    converter = TextToImage2MDatasetConverter(DatasetAttr("webdataset", REPO_ID), DataArguments())
    with pytest.raises(ValueError, match="broken.*json.prompt"):
        converter({"__key__": "broken", "json": metadata, "jpg": b"invalid"})


def test_repacking_preserves_every_pair_and_supports_shard_partitioning(tmp_path, shards):
    path = Path(__file__).resolve().parents[2] / "examples/datasets/text_to_image_2m/prepare_shards.py"
    module_spec = importlib.util.spec_from_file_location("prepare_t2i_shards", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    output = tmp_path / "repacked"
    report = module.prepare(shards, output, num_shards=2)
    assert report["samples"] == 4 and report["samples_per_shard"] == [2, 2]
    original = []
    for path in shards:
        with tarfile.open(path) as archive:
            original.extend(archive.extractfile(member).read() for member in archive if member.isfile())
    copied = []
    for path in sorted(output.glob("*.tar")):
        with tarfile.open(path) as archive:
            copied.extend(archive.extractfile(member).read() for member in archive if member.isfile())
    assert sorted(copied) == sorted(original)
    attr = get_dataset_list(["text_to_image_2m"], str(output))[0]
    assert len(resolve_data_files(attr, str(output))["train"]) == 2
    with pytest.raises(FileExistsError):
        module.prepare(shards, output, num_shards=2)
