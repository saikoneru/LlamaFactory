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

import itertools
import json
import os
import pickle
from collections import defaultdict, deque
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


class _MixStage:
    r"""One iterator stage of a ``_PythonMix`` pipeline.

    ``load_state_dict`` is only valid before the first ``__next__``. States are
    built when requested, which StatefulDataLoader does on snapshot steps only.
    """

    def __iter__(self) -> "_MixStage":
        return self

    def __next__(self) -> dict[str, Any]:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


def _dump_examples(examples: list[dict[str, Any]]) -> bytes:
    # A single blob keeps torchdata's per-key state diff cheap for image and array payloads.
    return pickle.dumps(examples, protocol=pickle.HIGHEST_PROTOCOL)


class _InterleaveStage(_MixStage):
    def __init__(
        self,
        datasets: list[Any],
        probabilities: list[float] | None,
        seed: int,
        stopping_strategy: str,
    ) -> None:
        count = len(datasets)
        self._datasets = datasets
        self._weights = np.array(
            probabilities if probabilities is not None else [1.0 / count] * count, dtype=float
        )
        self._rng = np.random.default_rng(seed)
        self._restart = stopping_strategy != "first_exhausted"
        self._alive = [True] * count
        self._iterators: list[Optional[Iterator[dict[str, Any]]]] = [None] * count
        self._positions = [0] * count
        self._source_states: list[Optional[dict[str, Any]]] = [None] * count

    def _open(self, index: int, state: Optional[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        # Sources resume by example count. HF's own IterableDataset state is not used: through a
        # map over Arrow tables it records the end of the current table and drops its unread rows.
        if state is not None and "position" not in state:
            raise ValueError("Mixed dataset state was saved by an older version and cannot be resumed exactly.")

        dataset = self._datasets[index]
        start = (state or {}).get("position", 0)
        self._positions[index] = start
        if start and hasattr(dataset, "__len__") and start >= len(dataset):
            iterator = iter(())
        elif start and hasattr(dataset, "skip"):
            iterator = iter(dataset.skip(start))
        else:
            iterator = itertools.islice(iter(dataset), start, None)

        self._iterators[index] = iterator
        return iterator

    def _next_from(self, index: int) -> dict[str, Any]:
        iterator = self._iterators[index]
        if iterator is None:
            iterator = self._open(index, self._source_states[index])
            self._source_states[index] = None

        example = next(iterator)
        self._positions[index] += 1
        return example

    def __next__(self) -> dict[str, Any]:
        count = len(self._datasets)
        while True:
            candidates = [index for index, is_alive in enumerate(self._alive) if is_alive]
            if not candidates:
                raise StopIteration
            if not self._restart and len(candidates) < count:
                raise StopIteration

            local = self._weights[candidates]
            local = local / local.sum()
            picked = int(candidates[int(self._rng.choice(len(candidates), p=local))])
            try:
                return self._next_from(picked)
            except StopIteration:
                if not self._restart:
                    self._alive[picked] = False
                    continue

                self._positions[picked] = 0
                self._open(picked, None)
                try:
                    return self._next_from(picked)
                except StopIteration:
                    self._alive[picked] = False

    def _source_state(self, index: int) -> Optional[dict[str, Any]]:
        if self._iterators[index] is None:
            return self._source_states[index]

        return {"position": self._positions[index]}

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self._rng.bit_generator.state,
            "alive": list(self._alive),
            "sources": [self._source_state(index) for index in range(len(self._datasets))],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if len(state["sources"]) != len(self._datasets):
            raise ValueError(
                f"Mixed dataset state has {len(state['sources'])} sources, current mix has {len(self._datasets)}."
            )

        self._rng.bit_generator.state = state["rng"]
        self._alive = list(state["alive"])
        self._iterators = [None] * len(self._datasets)
        self._source_states = list(state["sources"])


class _ShuffleStage(_MixStage):
    def __init__(self, child: _MixStage, buffer_size: int, seed: int | None) -> None:
        self._child = child
        self._buffer_size = buffer_size
        self._rng = np.random.default_rng(seed)
        self._buffer: list[dict[str, Any]] = []
        self._draining = False

    def __next__(self) -> dict[str, Any]:
        while not self._draining:
            try:
                example = next(self._child)
            except StopIteration:
                self._draining = True
                self._rng.shuffle(self._buffer)
                break

            self._buffer.append(example)
            if len(self._buffer) >= self._buffer_size:
                index = int(self._rng.integers(0, len(self._buffer)))
                return self._buffer.pop(index)

        if not self._buffer:
            raise StopIteration

        return self._buffer.pop(0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self._rng.bit_generator.state,
            "draining": self._draining,
            "buffer": _dump_examples(self._buffer),
            "child": self._child.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = state["rng"]
        self._draining = state["draining"]
        self._buffer = pickle.loads(state["buffer"])
        self._child.load_state_dict(state["child"])


class _MapStage(_MixStage):
    def __init__(
        self,
        child: _MixStage,
        function: Callable[..., dict[str, Any]],
        batched: bool,
        batch_size: int,
        remove: set[str],
    ) -> None:
        self._child = child
        self._function = function
        self._batched = batched
        self._batch_size = batch_size
        self._remove = remove
        self._pending: deque[dict[str, Any]] = deque()
        self._exhausted = False

    def _fill(self) -> None:
        if not self._batched:
            try:
                example = next(self._child)
            except StopIteration:
                self._exhausted = True
                return

            mapped = self._function(example)
            self._pending.append({key: value for key, value in mapped.items() if key not in self._remove})
            return

        batch: dict[str, list[Any]] = defaultdict(list)
        size = 0
        while size < self._batch_size:
            try:
                example = next(self._child)
            except StopIteration:
                self._exhausted = True
                break

            for key, value in example.items():
                batch[key].append(value)
            size += 1

        if size:
            self._pending.extend(_emit_batch(self._function(dict(batch)), self._remove))

    def __next__(self) -> dict[str, Any]:
        while not self._pending:
            if self._exhausted:
                raise StopIteration
            self._fill()

        return self._pending.popleft()

    def state_dict(self) -> dict[str, Any]:
        # Outputs of the current batch are stored because their inputs are already consumed.
        return {
            "pending": _dump_examples(list(self._pending)),
            "exhausted": self._exhausted,
            "child": self._child.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._pending = deque(pickle.loads(state["pending"]))
        self._exhausted = state["exhausted"]
        self._child.load_state_dict(state["child"])


class _TakeStage(_MixStage):
    def __init__(self, child: _MixStage, limit: int) -> None:
        self._child = child
        self._limit = limit
        self._count = 0

    def __next__(self) -> dict[str, Any]:
        if self._count >= self._limit:
            raise StopIteration

        example = next(self._child)
        self._count += 1
        return example

    def state_dict(self) -> dict[str, Any]:
        return {"count": self._count, "child": self._child.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._count = state["count"]
        self._child.load_state_dict(state["child"])


class _SkipStage(_MixStage):
    def __init__(self, child: _MixStage, skip: int) -> None:
        self._child = child
        self._skip = skip
        self._count = 0

    def __next__(self) -> dict[str, Any]:
        while self._count < self._skip:
            next(self._child)
            self._count += 1

        example = next(self._child)
        self._count += 1
        return example

    def state_dict(self) -> dict[str, Any]:
        return {"count": self._count, "child": self._child.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._count = state["count"]
        self._child.load_state_dict(state["child"])


class _MixIterator:
    r"""One pass over a ``_PythonMix``, sharded across ranks and dataloader workers."""

    def __init__(
        self,
        stage: _MixStage,
        state: Optional[dict[str, Any]],
        rank: int,
        world_size: int,
    ) -> None:
        worker = torch.utils.data.get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        self.pid = os.getpid()
        self.started = False
        self._stage = stage
        self._stride = world_size * num_workers
        self._offset = rank * num_workers + worker_id
        self._index = 0
        if state is not None:
            self.restore(state)

    def restore(self, state: dict[str, Any]) -> None:
        if self.started:
            raise RuntimeError("Cannot restore a mixed dataset iterator that has already yielded examples.")

        self._index = state["index"]
        if state["stage"] is not None:
            self._stage.load_state_dict(state["stage"])

    def __iter__(self) -> "_MixIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        self.started = True
        while True:
            example = next(self._stage)
            index = self._index
            self._index += 1
            if self._stride == 1 or index % self._stride == self._offset:
                return example

    def state_dict(self) -> dict[str, Any]:
        return {"index": self._index, "stage": self._stage.state_dict()}


class _PythonMix(torch.utils.data.IterableDataset):
    """Iterable mix that never asks Arrow to type mixed extra-encoder rows.

    Hugging Face ``interleave_datasets`` calls ``_resolve_features()``, which
    tries to build a PyArrow table from a sample. MIMIC MedGemma values are
    PIL JPEGs; BEN TerraMind values are nested arrays. Arrow cannot type that
    mix, so we iterate Python dicts and implement the ``map`` used by the
    LLaMA-Factory loader. Subclassing torch IterableDataset stops the Trainer
    from using a length-based sampler.

    ``state_dict`` and ``load_state_dict`` follow the protocol StatefulDataLoader
    uses for Hugging Face iterables, so a mix of different extra encoders resumes
    from the exact example where its checkpoint was taken.
    """

    def __init__(self, build: Callable[[], _MixStage], rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self._build = build
        self._rank = rank
        self._world_size = world_size
        self._pending_state: Optional[dict[str, Any]] = None
        self._live: Optional[_MixIterator] = None

    def __iter__(self) -> _MixIterator:
        live = _MixIterator(self._build(), self._pending_state, self._rank, self._world_size)
        self._pending_state = None
        self._live = live
        return live

    def _owns_live(self) -> bool:
        # A forked worker inherits the parent's preview iterator, which is not its position.
        return self._live is not None and self._live.pid == os.getpid()

    def state_dict(self) -> dict[str, Any]:
        if self._owns_live():
            return self._live.state_dict()

        if self._pending_state is not None:
            return self._pending_state

        return {"index": 0, "stage": None}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._owns_live() and not self._live.started:
            self._live.restore(state)
            return

        self._pending_state = state
        self._live = None

    def shard_by_node(self, rank: int, world_size: int) -> "_PythonMix":
        r"""Return a copy that yields one rank's share of the stream."""
        return _PythonMix(self._build, rank=rank, world_size=world_size)

    def _derive(self, build: Callable[[], _MixStage]) -> "_PythonMix":
        return _PythonMix(build, rank=self._rank, world_size=self._world_size)

    def shuffle(self, buffer_size: int = 16384, seed: int | None = None):
        parent = self._build
        return self._derive(lambda: _ShuffleStage(parent(), buffer_size, seed))

    def map(self, function, batched=True, batch_size=1000, remove_columns=None, **_kwargs):
        parent = self._build
        remove = set(remove_columns or [])
        return self._derive(lambda: _MapStage(parent(), function, batched, batch_size, remove))

    def take(self, n: int):
        parent = self._build
        return self._derive(lambda: _TakeStage(parent(), n))

    def skip(self, n: int):
        parent = self._build
        return self._derive(lambda: _SkipStage(parent(), n))


def _emit_batch(output: dict[str, list[Any]], remove: set[str]) -> Iterator[dict[str, Any]]:
    keys = [key for key in output if key not in remove]
    if not keys:
        return
    length = len(output[keys[0]])
    for index in range(length):
        yield {key: output[key][index] for key in keys}


def _is_null_feature(feature: Any) -> bool:
    return isinstance(feature, Value) and feature.dtype == "null"


def _unify_mm_features_for_mix(
    all_datasets: list[Union["Dataset", "IterableDataset"]],
    resolve: bool = True,
) -> list[Union["Dataset", "IterableDataset"]]:
    r"""Make image columns mixable across path-based and bytes-based sources.

    HuggingFace infers `_images` from the first example, so a path-only corpus becomes
    `List({bytes: null, path: string})` while a bytes corpus becomes
    `List({bytes: binary, path: null})`. `interleave_datasets` then refuses to align them.
    Recasting to `List(Image(decode=False))` keeps both representations and leaves decoding
    to LlamaFactory's collator.

    With ``resolve=False`` only datasets whose features are already known are recast.
    Inference builds a PyArrow table from a sample and cannot type decoded PIL values.
    """
    image_feature = List(Image(decode=False))
    unified = []
    for dataset in all_datasets:
        resolve_features = getattr(dataset, "_resolve_features", None)
        if resolve and callable(resolve_features):
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

    use_python_mix = data_args.mix_strategy.startswith("interleave") and requires_python_mix(encoder_schemas)
    all_datasets = _unify_mm_features_for_mix(all_datasets, resolve=not use_python_mix)

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
        if not use_python_mix:
            return interleave_datasets(
                datasets=all_datasets,
                probabilities=probabilities,
                seed=seed,
                stopping_strategy=strategy_map,  # type: ignore
            )

        logger.info_rank0(
            "Interleaving datasets in Python so mixed extra-encoder rows "
            "are not forced through PyArrow feature inference."
        )
        return _PythonMix(lambda: _InterleaveStage(all_datasets, probabilities, seed, strategy_map))

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
