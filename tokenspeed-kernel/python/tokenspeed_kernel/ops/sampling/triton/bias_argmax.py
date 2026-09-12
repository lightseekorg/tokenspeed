# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Two-stage bias argmax, preserving both native dtype roundings."""

from __future__ import annotations

import importlib.metadata
import re

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling.bias_argmax import (
    BIAS_ARGMAX_TILE,
    BiasArgmaxWorkspace,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.sampling_metadata import NativeLaunchPair


@triton.jit
def _bias_argmax_partials(
    logits,
    bias,
    values,
    indices,
    VOCAB: tl.constexpr,
    LOGIT_ROW_STRIDE: tl.constexpr,
    LOGIT_COL_STRIDE: tl.constexpr,
    BIAS_ROW_STRIDE: tl.constexpr,
    BIAS_COL_STRIDE: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * TILE + tl.arange(0, TILE)
    valid = col < VOCAB
    dtype: tl.constexpr = logits.dtype.element_ty
    x = tl.load(
        logits
        + row.to(tl.int64) * LOGIT_ROW_STRIDE
        + col.to(tl.int64) * LOGIT_COL_STRIDE,
        valid,
        other=-float("inf"),
    ).to(tl.float32)
    b = (
        tl.load(
            bias
            + row.to(tl.int64) * BIAS_ROW_STRIDE
            + col.to(tl.int64) * BIAS_COL_STRIDE,
            valid,
            other=0,
        )
        .to(dtype)
        .to(tl.float32)
    )
    # Native torch first casts bias to logits.dtype, then rounds the addition
    # into a tensor of that dtype. Both boundaries can change the winning ID.
    x = (x + b).to(dtype).to(tl.float32)
    nan = (x != x) & valid
    first_nan = tl.min(tl.where(nan, col, 2147483647), 0)
    maximum = tl.max(tl.where(nan, -float("inf"), x), 0)
    first_max = tl.min(tl.where(valid & (x == maximum), col, 2147483647), 0)
    has_nan = first_nan != 2147483647
    offset = row.to(tl.int64) * PARTIAL_STRIDE + tile
    tl.store(values + offset, tl.where(has_nan, float("nan"), maximum))
    tl.store(indices + offset, tl.where(has_nan, first_nan, first_max))


@triton.jit
def _bias_argmax_finalize(
    values,
    indices,
    out,
    TILES: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    GLOBAL_OFFSET: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK_TILES)
    valid = tile < TILES
    offset = row.to(tl.int64) * PARTIAL_STRIDE + tile
    x = tl.load(values + offset, valid, other=-float("inf"))
    idx = tl.load(indices + offset, valid, other=2147483647)
    nan = (x != x) & valid
    first_nan = tl.min(tl.where(nan, idx, 2147483647), 0)
    maximum = tl.max(tl.where(nan, -float("inf"), x), 0)
    first_max = tl.min(tl.where(valid & (x == maximum), idx, 2147483647), 0)
    target = tl.where(first_nan != 2147483647, first_nan, first_max)
    global_id = target.to(tl.int64) + GLOBAL_OFFSET
    tl.store(out + row.to(tl.int64) * OUT_STRIDE, global_id)


# Official CompiledKernel[grid] runners retain code and grid, never tensors or
# streams. All current arguments and constexpr slots are passed on each call.
_LAUNCHERS = {}
_MAX_LAUNCHERS = 128


def _make_native_pair(partial, final):
    # This direct driver ABI is bounded to the inspected pinned compiler. All
    # instrumentation, nonstandard launches and scratch allocations retain the
    # official CompiledKernel runner. No GPU algorithm differs between modes.
    if (
        NativeLaunchPair is None
        or triton.__version__ != "3.8.10"
        or importlib.metadata.version("tokenspeed-triton") != "3.8.10.post20260906"
    ):
        return None
    for kernel, pointers in ((partial, 6), (final, 5)):
        runner = kernel.run
        if (
            runner.global_scratch_size
            or runner.profile_scratch_size
            or runner.gsan_enabled
            or runner.launch_cooperative_grid
            or runner.launch_pdl
            or kernel.metadata.num_ctas != 1
            or kernel.metadata.num_warps != 4
        ):
            return None
        declaration = kernel.asm["ptx"].split(".visible .entry", 1)[1].split(")", 1)[0]
        parameters = re.findall(r"\.param\s+([^,\n]+)", declaration)
        if len(parameters) != pointers or any(
            not p.startswith(".u64 .ptr .global") for p in parameters
        ):
            return None
    return NativeLaunchPair(
        partial.function,
        final.function,
        partial.metadata.num_warps,
        final.metadata.num_warps,
        partial.metadata.shared,
        final.metadata.shared,
    )


def _inactive_launch_hook(hook):
    # The pinned vendor runtime uses an empty HookChain, not None, by default.
    # Inspect its current calls on every launch; arbitrary callable replacements
    # and HookChain subclasses always keep the official runner path.
    return hook is None or (type(hook) is triton.knobs.HookChain and not hook.calls)


def _can_reuse_launchers():
    runtime = triton.knobs.runtime
    compilation = triton.knobs.compilation
    return not (
        _bias_argmax_partials.pre_run_hooks
        or _bias_argmax_finalize.pre_run_hooks
        or _bias_argmax_partials.debug
        or _bias_argmax_finalize.debug
        or runtime.debug
        or runtime.add_stages_inspection_hook is not None
        or compilation.instrumentation_mode
        or compilation.fpsan_homomorphic_casts
    )


@register_kernel(
    "sampling",
    "bias_argmax",
    name="triton_bias_argmax_sm120",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(
        "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.SPECIALIZED,
    tags={"latency", "determinism"},
)
def bias_argmax_sm120(
    logits: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    workspace: BiasArgmaxWorkspace,
    global_offset: int,
) -> None:
    """Run exact dtype-rounded partials followed by a strided-output merge."""
    metadata = (
        logits.shape[0],
        logits.shape[1],
        logits.device.index,
        logits.dtype,
        bias.dtype,
        out.dtype,
        logits.stride(),
        bias.stride(),
        out.stride(0),
        workspace.values.stride(0),
        tuple(
            tensor.data_ptr() % 16
            for tensor in (logits, bias, out, workspace.values, workspace.indices)
        ),
    )
    _launch_from_metadata(logits, bias, out, workspace, global_offset, metadata)


def _launch_from_metadata(logits, bias, out, workspace, global_offset, metadata):
    """Use this invocation's checked scalars; retain no tensors or streams."""
    (
        rows,
        vocab,
        device_index,
        logits_dtype,
        bias_dtype,
        out_dtype,
        logit_strides,
        bias_strides,
        out_stride,
        partial_stride,
        alignments,
    ) = metadata
    values, indices = workspace.values, workspace.indices
    tiles = triton.cdiv(vocab, BIAS_ARGMAX_TILE)
    partial_constants = (
        vocab,
        logit_strides[0],
        logit_strides[1],
        bias_strides[0],
        bias_strides[1],
        partial_stride,
        BIAS_ARGMAX_TILE,
    )
    final_constants = (
        tiles,
        partial_stride,
        out_stride,
        global_offset,
        triton.next_power_of_2(tiles),
    )
    partial_args = (logits, bias, values, indices) + partial_constants
    final_args = (values, indices, out) + final_constants
    partial_grid, final_grid = (rows, tiles, 1), (rows, 1, 1)
    cacheable = _can_reuse_launchers()
    if cacheable:
        key = (
            device_index,
            device_index,
            logits_dtype,
            bias_dtype,
            out_dtype,
            # JIT tensor specialization distinguishes 16-byte divisibility.
            # Exact residues also separate every strided output column that
            # can have weaker alignment; each class first uses normal JIT.
            alignments,
            partial_grid,
            final_grid,
            partial_constants,
            final_constants,
            _bias_argmax_partials.src,
            _bias_argmax_finalize.src,
        )
        cached = _LAUNCHERS.get(key)
        if cached is not None:
            if (
                cached[2] is not None
                and _inactive_launch_hook(triton.knobs.runtime.launch_enter_hook)
                and _inactive_launch_hook(triton.knobs.runtime.launch_exit_hook)
            ):
                stream = triton.runtime.driver.active.get_current_stream(device_index)
                cached[2].run(logits, bias, out, values, indices, rows, tiles, stream)
            else:
                cached[0](*partial_args)
                cached[1](*final_args)
            return
    partial = _bias_argmax_partials[partial_grid](*partial_args, num_warps=4)
    final = _bias_argmax_finalize[final_grid](*final_args, num_warps=4)
    if cacheable and partial is not None and final is not None:
        if len(_LAUNCHERS) >= _MAX_LAUNCHERS:
            _LAUNCHERS.clear()
        _LAUNCHERS[key] = (
            partial[partial_grid],
            final[final_grid],
            _make_native_pair(partial, final),
        )


# Only this selected implementation opts into the checked-descriptor handoff.
bias_argmax_sm120._from_checked_metadata = _launch_from_metadata
