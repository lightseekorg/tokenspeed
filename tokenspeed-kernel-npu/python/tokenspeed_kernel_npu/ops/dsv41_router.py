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

"""Portable DeepSeek V4 sqrt-softplus expert routing."""

# NaN comparisons preserve the selection order on this Triton-Ascend version.
# ruff: noqa: PLR0124

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton


@lru_cache(maxsize=1)
def _range_check_library():
    path = Path(__file__).resolve().parents[1] / "lib" / "libassert_indices.so"
    if not path.is_file():
        raise RuntimeError("Build tokenspeed-kernel-npu/csrc/assert_indices/build.sh")
    library = ctypes.CDLL(str(path))
    library.launch_assert_indices.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    library.launch_assert_indices.restype = None
    return library


def assert_indices_in_range(indices, upper_bound):
    """Device-side range check, including graph replay; never copies to the CPU."""
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("MoE indices must be INT32 or INT64")
    flat = indices.contiguous()
    if flat.numel():
        # torch._assert_async on the supported torch_npu performs a host copy.
        _range_check_library().launch_assert_indices(
            flat.data_ptr(),
            flat.numel(),
            upper_bound,
            flat.element_size(),
            torch.npu.current_stream(flat.device).npu_stream,
        )


@triton.jit(do_not_specialize=["routed_scaling_factor"])
def _sqrt_softplus_topk_kernel(
    logits_ptr,
    bias_ptr,
    hash_ptr,
    ids_ptr,
    weights_ptr,
    experts_ptr,
    scores_ptr,
    logits_stride_m,
    logits_stride_e,
    bias_stride,
    hash_stride_m,
    hash_stride_k,
    routed_scaling_factor,
    EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HASH: tl.constexpr,
    BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    NEED_SCORES: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
):
    """Write expert ids and sqrt-softplus weights using top-k or hash routing."""
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK_E)
    logits = tl.load(
        logits_ptr + token * logits_stride_m + expert * logits_stride_e,
        mask=expert < EXPERTS,
        other=0,
    ).to(tl.float32)
    if PRECOMPUTED:
        scores = logits
    else:
        softplus = tl.maximum(logits, 0.0) + libdevice.log1p(tl.exp(-tl.abs(logits)))
        scores = tl.sqrt(tl.where(logits > 20.0, logits, softplus))
    if NEED_SCORES:
        tl.store(scores_ptr + token * EXPERTS + expert, scores, mask=expert < EXPERTS)
    lanes = tl.arange(0, BLOCK_K)
    chosen_ids = tl.full((BLOCK_K,), -1, tl.int32)
    chosen_weights = tl.zeros((BLOCK_K,), tl.float32)
    if HASH:
        input_id = tl.load(ids_ptr + token).to(tl.int64)
    else:
        choice = scores
        if BIAS:
            choice += tl.load(
                bias_ptr + expert * bias_stride, mask=expert < EXPERTS, other=0
            )
        # Monotonic FP32 bit ordering, including equal +/-0 and NaNs first.
        # INT32_MIN is below -inf, so removed/padded lanes cannot win a tie.
        bits = choice.to(tl.int32, bitcast=True)
        ordered = tl.where(bits < 0, -2147483648 - bits, bits)
        ordered = tl.where(choice != choice, 2147483647, ordered)
        ordered = tl.where(expert < EXPERTS, ordered, -2147483648)
    for rank in range(TOPK):
        if HASH:
            selected = tl.load(
                hash_ptr + input_id * hash_stride_m + rank * hash_stride_k
            ).to(tl.int32)
        else:
            maximum = tl.max(ordered, 0)
            selected = tl.min(tl.where(ordered == maximum, expert, EXPERTS), 0)
            ordered = tl.where(expert == selected, -2147483648, ordered)
        weight = tl.sum(tl.where(expert == selected, scores, 0.0), 0)
        chosen_ids = tl.where(lanes == rank, selected, chosen_ids)
        chosen_weights = tl.where(lanes == rank, weight, chosen_weights)
    if RENORMALIZE:
        denominator = tl.sum(chosen_weights, 0)
        denominator = tl.where(
            denominator < 1.1754943508222875e-38, 1.1754943508222875e-38, denominator
        )
        chosen_weights /= denominator
    chosen_weights *= routed_scaling_factor
    tl.store(weights_ptr + token * TOPK + lanes, chosen_weights, mask=lanes < TOPK)
    tl.store(experts_ptr + token * TOPK + lanes, chosen_ids, mask=lanes < TOPK)


@triton.jit(do_not_specialize=["routed_scaling_factor"])
def _sqrt_softplus_top6_batch(
    S,
    B,
    W,
    I,
    N,
    BIAS_STRIDE,
    routed_scaling_factor,
    ROWS: tl.constexpr,
    BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    e = tl.arange(0, 512)
    k = tl.arange(0, 8)
    score = tl.load(
        S + r[:, None] * 384 + e[None, :],
        (r[:, None] < N) & (e[None, :] < 384),
        other=0,
    )
    bias = (
        tl.load(B + e * BIAS_STRIDE, e < 384, other=0)
        if BIAS
        else tl.full((512,), 0.0, tl.float32)
    )
    choice = tl.where(e[None, :] < 384, score + bias[None, :], -float("inf"))
    bits = choice.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, -2147483648 - bits, bits)
    ordered = tl.where(choice != choice, 2147483647, ordered)
    ordered = tl.where(e[None, :] < 384, ordered, -2147483648)
    ids = tl.full((ROWS, 8), -1, tl.int32)
    weights = tl.full((ROWS, 8), 0.0, tl.float32)
    for j in range(6):
        maximum = tl.max(ordered, 1)
        ix = tl.min(tl.where(ordered == maximum[:, None], e[None, :], 384), 1)
        ordered = tl.where(e[None, :] == ix[:, None], -2147483648, ordered)
        value = tl.sum(tl.where(e[None, :] == ix[:, None], score, 0.0), 1)
        ids = tl.where(k[None, :] == j, ix[:, None], ids)
        weights = tl.where(k[None, :] == j, value[:, None], weights)
    if RENORMALIZE:
        denominator = tl.sum(weights, 1)
        denominator = tl.where(
            denominator < 1.1754943508222875e-38, 1.1754943508222875e-38, denominator
        )
        weights = weights / denominator[:, None]
    weights *= routed_scaling_factor
    tl.store(
        W + r[:, None] * 6 + k[None, :], weights, (r[:, None] < N) & (k[None, :] < 6)
    )
    tl.store(I + r[:, None] * 6 + k[None, :], ids, (r[:, None] < N) & (k[None, :] < 6))


def ascend_sqrt_softplus_topk(
    router_logits,
    top_k,
    renormalize,
    correction_bias,
    hash_indices_table,
    input_ids,
    need_scores,
    routed_scaling_factor,
    weights_dtype,
):
    """Keep CANN score generation; fuse the V4.1 384-expert top6 selection.

    This preserves score rounding and CANN's lower-ID tie / NaN-first order.
    Integer ordering keys preserve every FP32 comparison without rounding;
    selected lanes receive a sentinel below all valid keys.
    """
    if router_logits.device.type != "npu" or router_logits.ndim != 2:
        raise ValueError("Expected NPU [tokens, experts] logits")
    tokens, experts = router_logits.shape
    if experts != 384 or top_k != 6:
        raise ValueError("Ascend V4.1 router supports exactly 384 experts/top6")
    scores = torch.sqrt(torch.nn.functional.softplus(router_logits.float()))
    bias = None
    if correction_bias is not None and hash_indices_table is None:
        if correction_bias.ndim != 1:
            raise ValueError("Per-token bias keeps the native torch route")
        bias = correction_bias.to(device=scores.device, dtype=scores.dtype)
    if hash_indices_table is not None and input_ids is None:
        raise ValueError("Hash routing requires input_ids")
    token_ids = None if input_ids is None else input_ids.reshape(-1).contiguous()
    weights = torch.empty((tokens, top_k), device=scores.device, dtype=weights_dtype)
    ids = torch.empty((tokens, top_k), device=scores.device, dtype=torch.int32)
    if (
        tokens >= 32
        and hash_indices_table is None
        and bias is not None
        and bias.stride(0) == 1
    ):
        # Four rows share a vector program. Keep CANN score generation and
        # the exact lower-ID / NaN-first selection and normalization order.
        _sqrt_softplus_top6_batch[(triton.cdiv(tokens, 4),)](
            scores,
            bias,
            weights,
            ids,
            tokens,
            0 if bias is None else bias.stride(0),
            routed_scaling_factor,
            ROWS=4,
            BIAS=bias is not None,
            RENORMALIZE=renormalize,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    elif tokens:
        _sqrt_softplus_topk_kernel[(tokens,)](
            scores,
            bias,
            hash_indices_table,
            token_ids,
            weights,
            ids,
            scores,
            *scores.stride(),
            0 if bias is None else bias.stride(0),
            0 if hash_indices_table is None else hash_indices_table.stride(0),
            0 if hash_indices_table is None else hash_indices_table.stride(1),
            routed_scaling_factor,
            EXPERTS=384,
            TOPK=6,
            BLOCK_E=512,
            BLOCK_K=8,
            HASH=hash_indices_table is not None,
            BIAS=bias is not None,
            RENORMALIZE=renormalize,
            NEED_SCORES=False,
            PRECOMPUTED=True,
            num_warps=4,
        )
    return weights, ids, scores if need_scores else router_logits
