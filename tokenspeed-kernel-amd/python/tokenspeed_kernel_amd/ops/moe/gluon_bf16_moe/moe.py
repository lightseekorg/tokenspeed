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

"""End-to-end bf16 Gluon MoE (gfx950): routing-sort -> stage1 -> stage2.

Public entry point ``gluon_bf16_moe`` computes an unquantized bf16 fused MoE
FFN: bf16 activations, bf16 weights, SwiGLU (g1u1), silu activation,
routed-weight fold in stage 2.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.moe_align_device import (
    moe_align_block_size_device,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.moe_align_fused import (
    moe_align_block_size_fused,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage1_kernel import invoke_stage1
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage1_splitk_kernel import (
    invoke_stage1_splitk,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.stage2_kernel import invoke_stage2

# Pure-Gluon warp-reduce GEMV decode path (equivalent to the Triton reference).
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.warp_decode_gluon_kernel import (
    invoke_stage1_warp_decode_gluon as invoke_stage1_warp_decode,
)
from tokenspeed_kernel_amd.ops.moe.gluon_bf16_moe.warp_decode_gluon_kernel import (
    invoke_stage2_warp_decode_gluon as invoke_stage2_warp_decode,
)

# Sort granularity == kernel BLOCK_M for both stages (each BLOCK_M tile maps
# to exactly one entry of sorted_expert_ids).
BLOCK_M = 64

# ---- Decode-specialised stage-1 config -------------------------------------
# Decode schedule: small M tile, large K tile, single LDS buffer -> minimal
# resource -> max occupancy, plus split-K for extra CTAs. Tuned on MI355X DSv3.
DECODE_MAX_M = 16  # auto-enable the decode path at/below this M
WARP_DECODE_MAX_M = 8  # warp-GEMV decode wins at/below this M; above it the
# split-K + reduce decode path is (marginally) faster
DECODE_BLOCK_M = 32  # sort + stage tile M
DECODE_S1_BLOCK_N = 64  # stage-1 N tile
DECODE_S1_BLOCK_K = 128  # stage-1 K tile
DECODE_S1_NBUF = 1  # single LDS buffer


def _decode_split_k(num_tokens: int) -> int:
    """Decode split-K factor for the stage-1 tile (K=128 -> 56 K-tiles).
    Tuned: more splits at tiny M for occupancy, fewer as M grows."""
    if num_tokens <= 4:
        return 4
    if num_tokens <= 8:
        return 2
    return 1


def gluon_bf16_moe(
    hidden_states: torch.Tensor,  # (num_tokens, D)  bf16
    w1: torch.Tensor,  # (E, 2*I, D)      bf16  gate rows [0:I], up [I:2I]
    w2: torch.Tensor,  # (E, D, I)        bf16
    topk_ids: torch.Tensor,  # (num_tokens, topk) int
    topk_weights: torch.Tensor,  # (num_tokens, topk) float
    block_m: int = BLOCK_M,
    split_k: int | None = None,
    decode: bool | None = None,
    warp_decode: bool | None = None,
) -> torch.Tensor:
    """Compute the fused bf16 MoE FFN and return ``(num_tokens, D)`` bf16.

        y[t] = sum_{s in topk} w[t,s] * down_e( silu(gate_e(h)) * up_e(h) )
        where e = topk_ids[t, s], w = topk_weights[t, s].

    ``decode`` selects the decode-specialised path (``None`` = auto: on at
    ``num_tokens <= DECODE_MAX_M``). The decode path uses a smaller sort/tile
    ``BLOCK_M``, the fused single-workgroup align, and a decode-specialised stage 1
    (small M, K=128, single LDS buffer) with split-K; the prefill path uses the
    device align + the pipelined XCD-remap stage 1. ``split_k`` overrides the
    stage-1 split factor (``None`` = auto by M).
    """
    assert hidden_states.dtype == torch.bfloat16
    assert w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    num_tokens, D = hidden_states.shape
    E, two_I, Dw = w1.shape
    assert Dw == D
    I_r = two_I // 2
    E2, D2, I2 = w2.shape
    assert (E2, D2, I2) == (
        E,
        D,
        I_r,
    ), f"w2 {tuple(w2.shape)} inconsistent with w1 (E={E}, D={D}, I={I_r})"
    topk = topk_ids.shape[1]

    if decode is None:
        decode = num_tokens <= DECODE_MAX_M
    if warp_decode is None:
        warp_decode = decode and num_tokens <= WARP_DECODE_MAX_M
    if decode:
        block_m = DECODE_BLOCK_M

    if decode and warp_decode:
        # Pure-Gluon warp-reduce GEMV decode: no sort, expert read directly from
        # topk_ids, topk combine fused into stage 2.
        block_n1 = 4 if num_tokens < 8 else 8
        inter = torch.empty(
            (num_tokens * topk, I_r), dtype=torch.bfloat16, device=hidden_states.device
        )
        invoke_stage1_warp_decode(
            hidden_states, w1, topk_ids, inter, topk, BLOCK_N=block_n1
        )
        out = torch.empty(
            (num_tokens, D), dtype=torch.bfloat16, device=hidden_states.device
        )
        invoke_stage2_warp_decode(inter, w2, topk_ids, topk_weights, out, topk)
        return out

    if decode:
        # sync-free, atomic-free single-workgroup align (outputs sized at the
        # fixed decode upper bound; stages early-out on the padded tail).
        sorted_token_ids, sorted_expert_ids, sorted_weights, num_valid = (
            moe_align_block_size_fused(topk_ids, topk_weights, E, block_m)
        )
    else:
        sorted_token_ids, sorted_expert_ids, sorted_weights, num_valid = (
            moe_align_block_size_device(topk_ids, topk_weights, E, block_m)
        )

    # empty() is safe: stage 1 writes every inter row (one per (token, slot))
    # before stage 2 reads it (avoids a per-call fill kernel).
    inter = torch.empty(
        (num_tokens * topk, I_r), dtype=torch.bfloat16, device=hidden_states.device
    )
    if decode:
        # decode stage 1 (small M/N, K=128, single LDS buffer) + split-K.
        sk = split_k if split_k is not None else _decode_split_k(num_tokens)
        invoke_stage1_splitk(
            hidden_states,
            w1,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid,
            inter,
            topk,
            sk,
            BLOCK_M=block_m,
            BLOCK_N=DECODE_S1_BLOCK_N,
            BLOCK_K=DECODE_S1_BLOCK_K,
            nbuf=DECODE_S1_NBUF,
        )
    else:
        invoke_stage1(
            hidden_states,
            w1,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid,
            inter,
            topk,
            BLOCK_M=block_m,
            split_k=split_k,
        )

    out = torch.empty(
        (num_tokens, D), dtype=torch.bfloat16, device=hidden_states.device
    )
    invoke_stage2(
        inter,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid,
        out,
        topk,
        BLOCK_M=block_m,
    )
    return out
