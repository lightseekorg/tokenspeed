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

"""Softmax top-k MoE routing in one GFX950 Gluon launch."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = ["launch_gluon_softmax_topk_gfx950"]

cdna4 = gl.amd.cdna4

_MAX_EXPERTS = 1024
_MAX_TOPK = 32
# One wave owns one token row; a workgroup holds this many rows.
_ROWS_PER_PROGRAM = 4


def _softmax_topk_launch_metadata(grid, kernel, args):
    """Report the logits read, the routing writes, and the selection work."""
    tokens = args["num_tokens"]
    experts = args["E"]
    topk = args["TOPK"]
    return {
        "name": kernel.name,
        # Each selection round scans every expert of the row once.
        "flops32": tokens * experts * (topk + 2),
        "bytes": tokens * experts * args["logits_ptr"].element_size()
        + tokens
        * topk
        * (args["weights_ptr"].element_size() + args["ids_ptr"].element_size()),
    }


@gluon.jit(
    launch_metadata=_softmax_topk_launch_metadata,
    do_not_specialize=("num_tokens",),
)
def gluon_softmax_topk_gfx950(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    num_tokens,
    stride_lm,
    stride_wm,
    stride_im,
    routed_scaling_factor,
    E: gl.constexpr,
    EP: gl.constexpr,
    TOPK: gl.constexpr,
    TKP: gl.constexpr,
    RENORMALIZE: gl.constexpr,
    ROWS: gl.constexpr,
):
    # Softmax is monotonic, so experts are selected on the logits. Each value
    # is packed with its inverted expert id into one integer key; the key
    # maximum is the largest logit with the lowest expert id on ties, and
    # keys are unique, so a selected key is removed by equality.
    gl.static_assert(EP % 64 == 0, "EP must cover whole wave64 lanes")
    VEC: gl.constexpr = EP // 64
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [ROWS, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, layout)
    DTYPE: gl.constexpr = logits_ptr.dtype.element_ty
    WIDE_KEY: gl.constexpr = DTYPE == gl.float32

    rows = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=row_layout)
    cols = gl.arange(0, EP, layout=col_layout)
    row_ok = rows < num_tokens
    mask = gl.expand_dims(row_ok, 1) & gl.expand_dims(cols < E, 0)
    offsets = gl.expand_dims(rows * stride_lm, 1) + gl.expand_dims(cols, 0)
    logits = cdna4.buffer_load(logits_ptr, offsets, mask=mask, other=0.0)
    # Fold -0.0 into +0.0 so both order as one value.
    logits = gl.where(logits == 0.0, gl.zeros_like(logits), logits)

    inverted_col = gl.expand_dims((EP - 1 - cols).to(gl.uint32), 0)
    if WIDE_KEY:
        bits = logits.to(gl.uint32, bitcast=True)
        ordered = bits ^ gl.where(
            (bits & 0x80000000) != 0,
            gl.full(bits.shape, 0xFFFFFFFF, gl.uint32, layout),
            gl.full(bits.shape, 0x80000000, gl.uint32, layout),
        )
        key = (ordered.to(gl.uint64) << 32) | inverted_col.to(gl.uint64)
        key = gl.where(mask, key, gl.zeros_like(key))
    else:
        bits = logits.to(gl.uint16, bitcast=True).to(gl.uint32)
        ordered = bits ^ gl.where(
            (bits & 0x8000) != 0,
            gl.full(bits.shape, 0xFFFF, gl.uint32, layout),
            gl.full(bits.shape, 0x8000, gl.uint32, layout),
        )
        key = (ordered << 16) | inverted_col
        key = gl.where(mask, key, gl.zeros_like(key))

    slot = gl.expand_dims(gl.arange(0, TKP, layout=col_layout), 0)
    selected = gl.full([ROWS, TKP], float("-inf"), gl.float32, layout)
    chosen = gl.full([ROWS, TKP], 0, gl.int32, layout)
    for rank in gl.static_range(TOPK):
        best = gl.max(key, axis=1)
        key = gl.where(key == gl.expand_dims(best, 1), gl.zeros_like(key), key)
        if WIDE_KEY:
            expert = (EP - 1) - (best & 0xFFFFFFFF).to(gl.int32)
            ordered_best = (best >> 32).to(gl.uint32)
            value_bits = ordered_best ^ gl.where(
                (ordered_best & 0x80000000) != 0,
                gl.full(ordered_best.shape, 0x80000000, gl.uint32, row_layout),
                gl.full(ordered_best.shape, 0xFFFFFFFF, gl.uint32, row_layout),
            )
            value = value_bits.to(gl.float32, bitcast=True)
        else:
            expert = (EP - 1) - (best & 0xFFFF).to(gl.int32)
            ordered_best = best >> 16
            value_bits = ordered_best ^ gl.where(
                (ordered_best & 0x8000) != 0,
                gl.full(ordered_best.shape, 0x8000, gl.uint32, row_layout),
                gl.full(ordered_best.shape, 0xFFFF, gl.uint32, row_layout),
            )
            value = value_bits.to(gl.uint16).to(DTYPE, bitcast=True).to(gl.float32)
        chosen = gl.where(slot == rank, gl.expand_dims(expert, 1), chosen)
        selected = gl.where(slot == rank, gl.expand_dims(value, 1), selected)

    # The first selection is the row maximum.
    row_max = gl.max(selected, axis=1, keep_dims=True)
    numerators = gl.exp(selected - row_max)
    if RENORMALIZE:
        # softmax(all)[topk] / sum(softmax(all)[topk]) is a softmax over the
        # selected logits; the full-row denominator cancels.
        denominator = gl.sum(numerators, axis=1, keep_dims=True)
    else:
        full = gl.where(mask, gl.exp(logits.to(gl.float32) - row_max), 0.0)
        denominator = gl.sum(full, axis=1, keep_dims=True)
    weights = numerators / denominator * routed_scaling_factor
    # The ids collect in registers like the weights, so each output leaves
    # with one buffer store instead of one store per selection round.
    slot_mask = gl.expand_dims(row_ok, 1) & (slot < TOPK)
    cdna4.buffer_store(
        chosen.to(ids_ptr.dtype.element_ty),
        ids_ptr,
        gl.expand_dims(rows * stride_im, 1) + slot,
        mask=slot_mask,
    )
    cdna4.buffer_store(
        weights,
        weights_ptr,
        gl.expand_dims(rows * stride_wm, 1) + slot,
        mask=slot_mask,
    )


def launch_gluon_softmax_topk_gfx950(
    *,
    router_logits: torch.Tensor,
    topk: int,
    topk_indices_dtype: torch.dtype,
    renormalize: bool,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse softmax, top-k selection, normalization and route scaling.

    Ties select the lowest expert id first, matching the Triton kernel.

    Args:
        router_logits: FP16, BF16 or FP32 logits ``[tokens, experts]`` with a
            contiguous expert dimension and at most 1024 experts.
        topk: Experts selected per token, at most 32.
        topk_indices_dtype: ``torch.int32`` or ``torch.int64`` for the ids.
        renormalize: Normalize the selected weights to sum to one. When false,
            the weights are the full-row softmax probabilities.
        routed_scaling_factor: Scale applied to every selected weight.
        enable_pdl: Ignored; GFX950 has no programmatic dependent launch.

    Returns:
        FP32 weights and expert ids, both ``[tokens, topk]``, ordered by
        descending weight.
    """
    del enable_pdl
    if (
        router_logits.ndim != 2
        or not router_logits.is_cuda
        or router_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or router_logits.stride(1) != 1
    ):
        raise ValueError(
            "GFX950 softmax top-k requires 2D FP16/BF16/FP32 logits with a "
            "contiguous expert dimension"
        )
    tokens, experts = router_logits.shape
    if experts > _MAX_EXPERTS:
        raise ValueError(
            f"GFX950 softmax top-k supports at most {_MAX_EXPERTS} experts"
        )
    if not 0 < topk <= min(experts, _MAX_TOPK):
        raise ValueError(
            f"topk must be in [1, min({experts}, {_MAX_TOPK})], got {topk}"
        )
    if topk_indices_dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_indices_dtype must be torch.int32 or torch.int64")

    device = router_logits.device
    weights = torch.empty((tokens, topk), dtype=torch.float32, device=device)
    ids = torch.empty((tokens, topk), dtype=topk_indices_dtype, device=device)
    if tokens == 0:
        return weights, ids
    padded_experts = max(64, 1 << (experts - 1).bit_length())
    grid = ((tokens + _ROWS_PER_PROGRAM - 1) // _ROWS_PER_PROGRAM,)
    gluon_softmax_topk_gfx950[grid](
        router_logits,
        weights,
        ids,
        tokens,
        router_logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        float(routed_scaling_factor),
        E=experts,
        EP=padded_experts,
        TOPK=topk,
        TKP=1 << (topk - 1).bit_length(),
        RENORMALIZE=bool(renormalize),
        ROWS=_ROWS_PER_PROGRAM,
        num_warps=_ROWS_PER_PROGRAM,
    )
    return weights, ids
