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

r"""Optional NVTX instrumentation for Nsight Systems profiling.

Everything here is a no-op unless `LLAMAFACTORY_NVTX=1` is set, so the markers can
live in the tree without affecting normal runs.

Environment variables:
    LLAMAFACTORY_NVTX:            `1` to emit NVTX ranges.
    LLAMAFACTORY_NVTX_START_STEP: global step at which to call `cudaProfilerStart`.
    LLAMAFACTORY_NVTX_STOP_STEP:  global step at which to call `cudaProfilerStop`.

The two step bounds pair with `nsys profile --capture-range=cudaProfilerApi` to keep
the capture window to a handful of steps instead of the whole run.
"""

import os
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Iterator, Optional


if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


NVTX_ENABLED = _env_flag("LLAMAFACTORY_NVTX")
NVTX_START_STEP = int(os.getenv("LLAMAFACTORY_NVTX_START_STEP", "-1"))
NVTX_STOP_STEP = int(os.getenv("LLAMAFACTORY_NVTX_STOP_STEP", "-1"))

_profiler_running = False
_profiler_finished = False


if NVTX_ENABLED:
    import torch

    @contextmanager
    def nvtx_range(name: str) -> Iterator[None]:
        r"""Emit an NVTX range around the wrapped block.

        Works in dataloader worker processes too: `range_push`/`range_pop` are
        thread-local and do not require an initialized CUDA context.
        """
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def nvtx_mark(name: str) -> None:
        r"""Emit an instantaneous NVTX marker."""
        torch.cuda.nvtx.mark(name)

    def nvtx_profiler_window(step: int) -> None:
        r"""Start/stop the CUDA profiler so nsys only captures the steps of interest."""
        global _profiler_running, _profiler_finished
        if NVTX_START_STEP < 0 or _profiler_finished:
            return

        if not _profiler_running and step >= NVTX_START_STEP:
            torch.cuda.profiler.start()
            _profiler_running = True
            nvtx_mark(f"profile/start@{step}")
        elif _profiler_running and 0 <= NVTX_STOP_STEP <= step:
            nvtx_mark(f"profile/stop@{step}")
            torch.cuda.profiler.stop()
            _profiler_running = False
            _profiler_finished = True  # the window fires once, never reopens

else:

    def nvtx_range(name: str) -> Any:
        return nullcontext()

    def nvtx_mark(name: str) -> None:
        pass

    def nvtx_profiler_window(step: int) -> None:
        pass


class NVTXDataLoaderProxy:
    r"""Wrap a dataloader so every blocking `next()` in the main process is an NVTX range.

    This is the signal for "is the input pipeline keeping up": a wide `dataloader/wait`
    band means the training loop is starved waiting on the workers, a thin sliver means
    prefetch is covering the step.

    Every other attribute is delegated, so `set_epoch`, `state_dict`, `load_state_dict`
    and `__len__` keep working on the underlying `StatefulDataLoader`.
    """

    def __init__(self, dataloader: "DataLoader", name: str = "train") -> None:
        self._dataloader = dataloader
        self._name = name

    def __getattr__(self, item: str) -> Any:
        # `_dataloader` lives in __dict__, so this never recurses.
        return getattr(self.__dict__["_dataloader"], item)

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Iterator[Any]:
        import torch

        iterator = iter(self._dataloader)
        step = 0
        while True:
            torch.cuda.nvtx.range_push(f"dataloader/wait/{self._name}#{step}")
            try:
                batch = next(iterator)
            except StopIteration:
                return
            finally:
                torch.cuda.nvtx.range_pop()

            step += 1
            yield batch


def maybe_wrap_dataloader(dataloader: "DataLoader", name: str = "train") -> Any:
    r"""Return an NVTX-instrumented dataloader, or the original one when profiling is off."""
    if not NVTX_ENABLED:
        return dataloader

    return NVTXDataLoaderProxy(dataloader, name=name)


def maybe_wrap_callable(obj: Any, attr: str, name: str) -> None:
    r"""Wrap `obj.attr` in an NVTX range in place. No-op when profiling is off."""
    if not NVTX_ENABLED:
        return

    original = getattr(obj, attr, None)
    if original is None or getattr(original, "_nvtx_wrapped", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with nvtx_range(name):
            return original(*args, **kwargs)

    wrapped._nvtx_wrapped = True
    setattr(obj, attr, wrapped)


def nvtx_optional_range(name: str, enabled: bool = True) -> Any:
    r"""Like `nvtx_range`, but skipped entirely when `enabled` is False."""
    if not enabled:
        return nullcontext()

    return nvtx_range(name)


class NVTXCallableWrapper:
    r"""Picklable callable that wraps another callable in an NVTX range.

    Used for the functions that cross into dataloader worker processes (the `datasets`
    map function and the collate function), where a closure would not survive a `spawn`
    start method.
    """

    def __init__(self, fn: Any, name: str) -> None:
        self.fn = fn
        self.name = name

    @staticmethod
    def _size(arg: Any) -> Optional[int]:
        try:
            if isinstance(arg, dict):  # a `datasets` batch: {column: [values]}
                return len(next(iter(arg.values())))

            return len(arg)  # a list of features
        except Exception:
            return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        size = self._size(args[0]) if args else None
        with nvtx_range(self.name if size is None else f"{self.name}(n={size})"):
            return self.fn(*args, **kwargs)


def maybe_wrap_fn(fn: Any, name: str) -> Any:
    r"""Wrap a worker-side callable in an NVTX range. Returns `fn` unchanged when profiling is off."""
    if not NVTX_ENABLED or fn is None:
        return fn

    return NVTXCallableWrapper(fn, name)


__all__ = [
    "NVTXCallableWrapper",
    "NVTXDataLoaderProxy",
    "NVTX_ENABLED",
    "maybe_wrap_callable",
    "maybe_wrap_dataloader",
    "maybe_wrap_fn",
    "nvtx_mark",
    "nvtx_optional_range",
    "nvtx_profiler_window",
    "nvtx_range",
]
