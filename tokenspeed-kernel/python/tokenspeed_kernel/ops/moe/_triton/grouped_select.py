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

"""Grouped-selection pieces the lane-parallel router kernel imports.

It keeps, by identity for
:mod:`tokenspeed_kernel.ops.moe._triton.router_select`, the parts of the replay of
the Torch dispatch the torch kernel runs (``torch 2.13.0+cu130``, recorded in
:data:`PINNED_TORCH`): the Beam routing constants and launch options, the
radix-select key order of ``TopKTypeConfig<float>::convert`` (``_ordered_key_tl``),
one compare-exchange stage of the 32-item ``bitonicSortKVInPlace<..., 16, 16, float,
long, GTOp<float, true>>`` network (``_swap_desc``; the 15 stages are
:data:`NETWORK`) and the admission rule :func:`grouped_select_rejection`.

Tie contract: the tie permutation is the pinned Torch dispatch's (the fill order
plus the unstable network) -- do not add a stable tie-break. A Torch dispatch
change is detected by the tie-row byte comparisons of the router kernel's GPU test
(``test/nvidia/ops/moe/test_router_select_cuda.py``); :data:`PINNED_TORCH` is
recorded, not enforced.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = [
    "NETWORK",
    "PINNED_TORCH",
    "grouped_select_rejection",
]

# Torch whose dispatch (gatherTopK + bitonicSortKVInPlace<..., 16, 16, ...>) is
# replayed; recorded, not enforced (the tie-row byte equality is the detector).
PINNED_TORCH = "2.13.0+cu130"
# Compile-time constants: the Beam routing shape.
# ``rows`` is the grid size only (no ``rows`` constant or trait: module doc).
_NUM_EXPERTS = 256
_NUM_GROUPS = 8
_GROUP_SIZE = 32
_GROUP_TOPK = 2
_TOPK_GROUPS = 4
_TOPK = 8
# Power2SortSize of the pinned bitonic launch (block_dim_x 16, two items per thread).
_SORT_LANES = 32
# Key-space sentinels: ``_ordered_key_tl`` maps every non-NaN float to
# ``int32(convert(x) ^ 0x80000000)``; no float maps to INT32_MIN, NaN is
# rewritten to KEY_NAN.
_INT32_MIN = -(2**31)
_KEY_NEG_INF = -2139095041  # ordered_key(-inf) == int32(0x807FFFFF)
_KEY_NAN = 0x7FFFFFFF
# Rank counting is chunked: the group stage compares its 8 keys in one [8, 8]
# block, the expert stage its 256 keys in eight [256, 32] blocks.
_GROUP_CHUNK = 8
_EXPERT_CHUNK = 32
# Launch options: fp fusion and FTZ off so the single
# add is one IEEE add.rn on subnormals as in Torch.
_LAUNCH_OPTIONS = dict(
    num_warps=4,
    enable_fp_fusion=False,
    enable_reflect_ftz=False,
    launch_pdl=False,
)


def _network() -> tuple[tuple[int, int], ...]:
    """The 15 ``(STRIDE, SIZE)`` stages of ATen's 32-item bitonic network
    (``SortUtils.cuh`` bitonicSort): merge phases ``size = 2, 4, 8, 16`` with ``stride =
    size / 2 .. 1`` and ``dir = (thread & (size / 2)) != 0``, then the final phase ``stride =
    16 .. 1`` with ``dir = false`` (encoded as ``SIZE = 0``)."""
    stages: list[tuple[int, int]] = []
    for size in (2, 4, 8, 16):
        stride = size // 2
        while stride:
            stages.append((stride, size))
            stride //= 2
    stages.extend((stride, 0) for stride in (16, 8, 4, 2, 1))
    return tuple(stages)


NETWORK = _network()


# ---------------------------------------------------------------------------------------
# Triton helpers the lane-parallel router kernel imports.
# ---------------------------------------------------------------------------------------


@triton.jit
def _ordered_key_tl(x, KEY_NAN: tl.constexpr):
    # ordered_key: bits ^ ((bits >> 31) & 0x7FFFFFFF), NaN -> KEY_NAN (arithmetic shift).
    bits = x.to(tl.int32, bitcast=True)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return tl.where(x != x, KEY_NAN, key)


@triton.jit
def _swap_desc(keys, order, valid, STRIDE: tl.constexpr, SIZE: tl.constexpr):
    # _swap: ATen bitonicSwap (SortUtils.cuh), 32 lanes, FP32 keys, GTOp<float, true>.
    lane = tl.arange(0, 32)
    partner = lane ^ STRIDE
    other_keys = tl.gather(keys, partner, axis=0)
    other_order = tl.gather(order, partner, axis=0)
    other_valid = tl.gather(valid, partner, axis=0)
    low = (lane & STRIDE) == 0
    a = tl.where(low, keys, other_keys)
    b = tl.where(low, other_keys, keys)
    valid_a = tl.where(low, valid, other_valid)
    valid_b = tl.where(low, other_valid, valid)
    # GTOp<float, true>: (isnan(a) && !isnan(b)) || a > b -- a float compare (+0 == -0).
    gt_ab = ((a != a) & ~(b != b)) | (a > b)
    comparison = (gt_ab & valid_a) | ~valid_b
    lower = lane & ~STRIDE
    thread = (lower // (2 * STRIDE)) * STRIDE + lower % STRIDE
    if SIZE == 0:
        direction = tl.full(keys.shape, False, tl.int1)
    else:
        direction = (thread & (SIZE // 2)) != 0
    exchange = comparison == direction
    return (
        tl.where(exchange, other_keys, keys),
        tl.where(exchange, other_order, order),
        tl.where(exchange, other_valid, valid),
    )


# ---------------------------------------------------------------------------------------
# Host: admission
# ---------------------------------------------------------------------------------------


def grouped_select_rejection(
    choices: torch.Tensor,
    num_groups: int,
    group_topk: int,
    topk_groups: int,
    topk: int,
) -> str:
    """Why the grouped-selection admission rejects this call; empty when admitted.

    ``triton_router_route_lanepar`` applies this rule to its logits
    (``router_select_rejection``).

    Metadata only (no device read): ``choices`` 2-D with a positive row count
    (an empty grid is not launched), FP32, 256 experts; the four routing
    parameters equal to the kernel's constexprs (8 groups, group score over the
    top 2, 4 kept groups, top-8); ``choices`` contiguous (the kernel reads
    ``row * 256 + lane`` with no stride argument); a CUDA tensor. The reason
    names the first rejecting attribute; the device clause comes last so that
    shape and layout findings are reported on any device.

    Args:
        choices: Biased FP32 scores the kernel selects from.
        num_groups: Requested group count.
        group_topk: Requested group-score term count.
        topk_groups: Requested kept-group count.
        topk: Requested expert count.

    Returns:
        The rejection reason, or ``""`` when the kernel admits the call.
    """
    if choices.ndim != 2:
        return f"choices must be 2-D, got {choices.ndim}-D"
    rows = int(choices.shape[0])
    if rows == 0:
        return "rows=0: an empty grid is not launched (rows must be positive)"
    if choices.dtype is not torch.float32:
        return f"choices must be {torch.float32}, got {choices.dtype}"
    if int(choices.shape[1]) != _NUM_EXPERTS:
        return (
            f"choices must have shape (rows, {_NUM_EXPERTS}), got "
            f"{tuple(choices.shape)}"
        )
    for name, value, expected in (
        ("num_groups", num_groups, _NUM_GROUPS),
        ("group_topk", group_topk, _GROUP_TOPK),
        ("topk_groups", topk_groups, _TOPK_GROUPS),
        ("topk", topk, _TOPK),
    ):
        if value != expected:
            return f"{name} must be {expected} (the kernel's constexpr), got {value}"
    if not choices.is_contiguous():
        return (
            f"choices must be contiguous (the kernel reads row * {_NUM_EXPERTS} + "
            f"lane), got strides {tuple(choices.stride())}"
        )
    if not choices.is_cuda:
        return f"choices must be a CUDA tensor, got device {choices.device}"
    return ""
