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

r"""Every `torch.compile` decision for training lives in this file.

`torch_compile: true` hands the whole `nn.Module` to `torch.compile` in one shot
(`accelerate/accelerator.py:1665`). For a multimodal model that fails: the glue between the
towers - `masked_scatter` on a boolean mask, `.tolist()` on `cu_seqlens`, the Python loop over
window splits - forces graph breaks and data-dependent guards, and Dynamo spends minutes
retracing before the first step.

Regional compilation inverts it. The repeated transformer blocks are compiled one by one, so
Inductor traces a single block and reuses that artifact for all of them, while everything
between the blocks stays eager and needs no `torch.compiler.disable` anywhere.

Shapes:
    Without `packing` the sequence length changes every step. `regional_compile_dynamic`
    controls what Dynamo does about that:

      auto  (default) compile static first, recompile dynamic once a second shape shows up.
                      Two compilations, then stable, and the best kernels for a fixed shape.
      true            compile with symbolic shapes immediately. One compilation, slightly
                      slower kernels. This is the setting for a run without `packing`.
      false           specialise on every shape. Only with `packing`, otherwise Dynamo
                      recompiles until it hits the cache limit and falls back to eager.
"""

import atexit
from typing import TYPE_CHECKING, Optional, Union

import torch.nn as nn

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)

# One compilation amortised over N blocks. Below this a stack is not worth the compile time.
MIN_BLOCKS = 4

# Dynamo holds one compiled entry per distinct guard set and, once the limit is reached, stops
# compiling that frame and silently runs it eager. Varying sequence lengths burn through the
# default of 8 within a few steps, so give it room even when `dynamic` should prevent it.
CACHE_SIZE_LIMIT = 64
ACCUMULATED_CACHE_SIZE_LIMIT = 512

# The towers are excluded by default: their input shape is a function of the images and audio
# in the batch, not of the sequence length, so they retrace far more often than the LM, and
# the vision tower's symbolic backward has been seen to fail outright in Inductor.
DEFAULT_EXCLUDE = "visual,vision_tower,vision_model,audio_tower,image_encoder"

_DYNAMIC: dict[str, Optional[bool]] = {"auto": None, "true": True, "false": False, "none": None, "null": None}


def _resolve_dynamic(value: Union[bool, str, None]) -> Optional[bool]:
    r"""Map the config value onto `torch.compile(dynamic=...)`.

    A yaml `true` is parsed as a real boolean long before the dataclass sees it, so accept
    booleans as well as the spelled-out strings. `null` and `auto` both mean "let Dynamo
    promote to symbolic shapes once a second shape shows up".
    """
    if value is None or isinstance(value, bool):
        return value

    key = str(value).strip().lower()
    if key not in _DYNAMIC:
        raise ValueError(f"`regional_compile_dynamic` must be one of {sorted(_DYNAMIC)} or a bool, got {value!r}.")

    return _DYNAMIC[key]


def _split(value: Union[str, list[str], None]) -> list[str]:
    r"""Accept a yaml list, a comma-separated string, or an OmegaConf `key=a,b` override."""
    if value is None:
        return []

    if isinstance(value, str):
        value = value.split(",")

    return [str(part).strip() for part in value if str(part).strip()]


def _find_block_stacks(model: "PreTrainedModel", exclude: list[str]) -> list[tuple[str, "nn.ModuleList"]]:
    r"""Find the repeated-block stacks: a `ModuleList` of enough blocks, all the same class."""
    stacks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < MIN_BLOCKS:
            continue

        if any(part in exclude for part in name.split(".")):
            continue

        if len({type(block) for block in module}) != 1:  # a mixed stack retraces per class
            continue

        stacks.append((name, module))

    return stacks


def _relax_dynamo_caches() -> None:
    from torch._dynamo import config as dynamo_config

    dynamo_config.cache_size_limit = max(dynamo_config.cache_size_limit, CACHE_SIZE_LIMIT)
    dynamo_config.accumulated_cache_size_limit = max(
        dynamo_config.accumulated_cache_size_limit, ACCUMULATED_CACHE_SIZE_LIMIT
    )


def _log_compile_stats() -> None:
    r"""Report at exit whether Dynamo actually compiled anything.

    `TORCH_LOGS="recompiles,graph_breaks"` only prints when something goes wrong, so a clean
    regional compile is silent and looks identical to a compile that never ran. These counters
    are the positive signal.
    """
    try:
        from torch._dynamo.utils import compile_times, counters

        graphs = counters["stats"].get("unique_graphs", 0)
        breaks = sum(counters["graph_break"].values())
        headers, values = compile_times(repr="csv", aggregate=True)
        seconds = dict(zip(headers, values)).get("_compile.compile_inner", "?")
    except Exception as err:  # never let a stats dump break the end of a run
        logger.debug(f"could not collect compile stats: {err}")
        return

    if graphs:
        logger.info_rank0(f"torch.compile: {graphs} graphs captured, {breaks} graph breaks, {seconds}s compiling.")
    else:
        logger.warning_rank0("torch.compile: nothing was captured - the compiled modules never ran.")


def configure_compile(model: "PreTrainedModel", model_args: "ModelArguments", is_trainable: bool) -> None:
    r"""Compile the repeated block stacks. Call once, after the adapter is attached."""
    if not is_trainable or not model_args.regional_compile:
        return

    dynamic = _resolve_dynamic(model_args.regional_compile_dynamic)
    exclude = _split(model_args.regional_compile_exclude)
    stacks = _find_block_stacks(model, exclude)
    if not stacks:
        logger.warning_rank0(f"`regional_compile` found no block stack outside {exclude}, compiled nothing.")
        return

    _relax_dynamo_caches()
    for name, stack in stacks:
        for block in stack:
            block.compile(
                backend=model_args.regional_compile_backend,
                mode=model_args.regional_compile_mode,
                dynamic=dynamic,
            )

        logger.info_rank0(
            f"Compiled {len(stack)} x {type(stack[0]).__name__} at `{name}` "
            f"(backend={model_args.regional_compile_backend}, mode={model_args.regional_compile_mode}, "
            f"dynamic={dynamic})."
        )

    atexit.register(_log_compile_stats)
