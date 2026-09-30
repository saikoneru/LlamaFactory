# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from collections import defaultdict
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, TypedDict, Union

import fsspec
import numpy as np
import torch.utils.data
from datasets import DatasetDict, Image, Value, concatenate_datasets, interleave_datasets
from datasets.features import List

from ..extras import logging


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


class _PythonMix(torch.utils.data.IterableDataset):
    """Iterable mix that never asks Arrow to type mixed extra-encoder rows.

    Hugging Face ``interleave_datasets`` calls ``_resolve_features()``, which
    tries to build a PyArrow table from a sample. MIMIC MedGemma values are
    PIL JPEGs; BEN TerraMind values are nested arrays. Arrow cannot type that
    mix, so we iterate Python dicts and implement the ``map`` used by the
    LLaMA-Factory loader. Subclassing torch IterableDataset stops the Trainer
    from using a length-based sampler.
    """

    def __init__(self, factory: Callable[[], Iterator[dict[str, Any]]]) -> None:
        super().__init__()
        self._factory = factory

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        stream = self._factory()
        if worker is None or worker.num_workers <= 1:
            yield from stream
            return
        for index, example in enumerate(stream):
            if index % worker.num_workers == worker.id:
                yield example

    def shuffle(self, buffer_size: int = 16384, seed: int | None = None):
        def shuffled_factory() -> Iterator[dict[str, Any]]:
            rng = np.random.default_rng(seed)
            buffer: list[dict[str, Any]] = []
            for example in self._factory():
                buffer.append(example)
                if len(buffer) >= buffer_size:
                    index = int(rng.integers(0, len(buffer)))
                    yield buffer.pop(index)
            rng.shuffle(buffer)
            yield from buffer

        return _PythonMix(shuffled_factory)

    def map(self, function, batched=True, batch_size=1000, remove_columns=None, **_kwargs):
        remove = set(remove_columns or [])

        def out_factory() -> Iterator[dict[str, Any]]:
            if batched:
                batch: dict[str, list[Any]] = defaultdict(list)
                size = 0
                for example in self._factory():
                    for key, value in example.items():
                        batch[key].append(value)
                    size += 1
                    if size >= batch_size:
                        yield from _emit_batch(function(dict(batch)), remove)
                        batch = defaultdict(list)
                        size = 0
                if size:
                    yield from _emit_batch(function(dict(batch)), remove)
                return
            for example in self._factory():
                mapped = function(example)
                yield {key: value for key, value in mapped.items() if key not in remove}

        return _PythonMix(out_factory)

    def take(self, n: int):
        def taken_factory() -> Iterator[dict[str, Any]]:
            for index, example in enumerate(self._factory()):
                if index >= n:
                    return
                yield example

        return _PythonMix(taken_factory)

    def skip(self, n: int):
        def skipped_factory() -> Iterator[dict[str, Any]]:
            for index, example in enumerate(self._factory()):
                if index >= n:
                    yield example

        return _PythonMix(skipped_factory)


def _emit_batch(output: dict[str, list[Any]], remove: set[str]) -> Iterator[dict[str, Any]]:
    keys = [key for key in output if key not in remove]
    if not keys:
        return
    length = len(output[keys[0]])
    for index in range(length):
        yield {key: output[key][index] for key in keys}


def _iter_interleave(
    datasets: list[Any],
    probabilities: list[float] | None,
    seed: int,
    stopping_strategy: str,
) -> Iterator[dict[str, Any]]:
    count = len(datasets)
    weights = np.array(probabilities if probabilities is not None else [1.0 / count] * count, dtype=float)
    rng = np.random.default_rng(seed)
    iterators = [iter(dataset) for dataset in datasets]
    alive = [True] * count
    restart = stopping_strategy != "first_exhausted"
    while True:
        candidates = [index for index, is_alive in enumerate(alive) if is_alive]
        if not candidates:
            return
        if not restart and len(candidates) < count:
            return
        local = weights[candidates]
        local = local / local.sum()
        picked = int(candidates[int(rng.choice(len(candidates), p=local))])
        try:
            yield next(iterators[picked])
        except StopIteration:
            if not restart:
                alive[picked] = False
                continue
            iterators[picked] = iter(datasets[picked])
            try:
                yield next(iterators[picked])
            except StopIteration:
                alive[picked] = False


def _is_null_feature(feature: Any) -> bool:
    return isinstance(feature, Value) and feature.dtype == "null"


def _unify_mm_features_for_mix(
    all_datasets: list[Union["Dataset", "IterableDataset"]],
) -> list[Union["Dataset", "IterableDataset"]]:
    r"""Make image columns mixable across path-based and bytes-based sources.

    HuggingFace infers `_images` from the first example, so a path-only corpus becomes
    `List({bytes: null, path: string})` while a bytes corpus becomes
    `List({bytes: binary, path: null})`. `interleave_datasets` then refuses to align them.
    Recasting to `List(Image(decode=False))` keeps both representations and leaves decoding
    to LlamaFactory's collator.
    """
    image_feature = List(Image(decode=False))
    unified = []
    for dataset in all_datasets:
        resolve_features = getattr(dataset, "_resolve_features", None)
        if callable(resolve_features):
            dataset = resolve_features()

        features = getattr(dataset, "features", None)
        if features is not None and "_images" in features and not _is_null_feature(features["_images"]):
            dataset = dataset.cast_column("_images", image_feature)

        unified.append(dataset)

    return unified


def requires_python_mix(encoder_schemas: list[Optional[frozenset[str]]] | None) -> bool:
    r"""Report whether the extra-encoder columns disagree across the datasets.

    Arrow types `_encoders` from one sample, so a mix of PIL images and nested
    arrays cannot share a column and must go through ``_PythonMix``. When every
    dataset declares the same encoders (including none at all), HuggingFace can
    interleave them and the result stays a `datasets` object, which the stateful
    dataloader needs for exact checkpoint/resume.
    """
    if encoder_schemas is None:  # caller did not report the columns
        return True

    return len({frozenset(schema or ()) for schema in encoder_schemas}) > 1


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    seed: int,
    is_eval: bool = False,
    encoder_schemas: list[Optional[frozenset[str]]] | None = None,
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    all_datasets = _unify_mm_features_for_mix(all_datasets)

    if data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)

    if data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        strategy_map: str = {
            "interleave_under": "first_exhausted",
            "interleave_over": "all_exhausted",
            "interleave_once": "all_exhausted_without_replacement",
        }[data_args.mix_strategy]
        probabilities = data_args.eval_interleave_probs if is_eval else data_args.interleave_probs
        if not requires_python_mix(encoder_schemas):
            return interleave_datasets(
                datasets=all_datasets,
                probabilities=probabilities,
                seed=seed,
                stopping_strategy=strategy_map,  # type: ignore
            )

        logger.info_rank0(
            "Interleaving datasets in Python so mixed extra-encoder rows "
            "are not forced through PyArrow feature inference. "
            "`use_stateful_dataloader` cannot resume this mix."
        )
        return _PythonMix(
            lambda: _iter_interleave(
                all_datasets,
                probabilities,
                seed,
                strategy_map,
            )
        )

    raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
) -> tuple[dict, dict]:
    r"""Split the dataset and returns two dicts containing train set and validation set.

    Support both map dataset and iterable dataset.

    Returns:
        train_dict: Dictionary containing training data with key "train"
        eval_dict: Dictionary containing evaluation data with keys "validation" or "validation_{name}"
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    # the train and eval better to in dict dtype and separately return for cpode clearly and good handle outside
    train_dict, eval_dict = {}, {}

    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if data_args.streaming:
                eval_dict["validation"] = dataset.take(int(data_args.val_size))
                train_dict["train"] = dataset.skip(int(data_args.val_size))
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                split_result = dataset.train_test_split(test_size=val_size, seed=seed)
                train_dict["train"] = split_result["train"]
                eval_dict["validation"] = split_result["test"]
        else:
            train_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, data in eval_dataset.items():
                eval_dict[f"validation_{name}"] = data
        else:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            eval_dict["validation"] = eval_dataset

    return train_dict, eval_dict


def get_dataset_module(dataset: Union["Dataset", "DatasetDict", dict]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict) or (
        isinstance(dataset, dict) and any(key in dataset for key in ("train", "validation"))
    ):
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def setup_fs(path: str, anon: bool = False) -> "fsspec.AbstractFileSystem":
    r"""Set up a filesystem object based on the path protocol."""
    storage_options = {"anon": anon} if anon else {}
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
    elif path.startswith(("gs://", "gcs://")):
        fs = fsspec.filesystem("gcs", **storage_options)
    else:
        raise ValueError(f"Unsupported protocol in path: {path}. Use 's3://' or 'gs://'.")

    if not fs.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    return fs


def _read_json_with_fs(fs: "fsspec.AbstractFileSystem", path: str) -> list[Any]:
    r"""Helper function to read JSON/JSONL files using fsspec."""
    with fs.open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def read_cloud_json(cloud_path: str) -> list[Any]:
    r"""Read a JSON/JSONL file from cloud storage (S3 or GCS).

    Args:
        cloud_path: str
            Cloud path in the format:
            - 's3://bucket-name/file.json' for AWS S3
            - 'gs://bucket-name/file.jsonl' or 'gcs://bucket-name/file.jsonl' for Google Cloud Storage
    """
    try:
        fs = setup_fs(cloud_path, anon=True)  # try with anonymous access first
    except Exception:
        fs = setup_fs(cloud_path)  # try again with credentials

    # filter out non-JSON files
    files = [x["Key"] for x in fs.listdir(cloud_path)] if fs.isdir(cloud_path) else [cloud_path]
    files = list(filter(lambda file: file.endswith(".json") or file.endswith(".jsonl"), files))
    if not files:
        raise ValueError(f"No JSON/JSONL files found in the specified path: {cloud_path}.")

    return sum([_read_json_with_fs(fs, file) for file in files], [])
