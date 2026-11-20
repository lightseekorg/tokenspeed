# SPDX-License-Identifier: Apache-2.0
#
# Copyright SGLang Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang's small-M FP8 dense GEMM with per-row and per-channel scales.

Adapted from SGLang's ``fp8w_small_m_gemm.py`` (the split-K GEMM, its
reduction and its MI355X tuning entries) and ``fused_norm_residual.py`` (the
per-row FP8 activation recipe); see ``README.md`` for the local changes.
``tokenspeed_kernel.ops.gemm.triton_fp8_channel_splitk`` registers and
launches them.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_FP8 = torch.float8_e4m3fn
# The largest finite FP8 E4M3 value: a row's or channel's amax maps to it.
_FP8_MAX = 448.0
# Floors on the amax before it becomes a scale, so an all-zero row or channel
# gets a positive scale and zero codes instead of dividing by zero. Both are
# SGLang's values, kept for its bits. Each acts only on an amax below it: no
# BF16 activation row of a served model is that small, and a weight channel
# below 1e-12 is zero in BF16 products, so the two floors never meet a value
# that matters and need not agree.
_ROW_AMAX_FLOOR = 1e-20
_CHANNEL_AMAX_FLOOR = 1e-12
# Triton reads module globals only as constexprs.
_FP8_MAX_C = tl.constexpr(_FP8_MAX)
_ROW_AMAX_FLOOR_C = tl.constexpr(_ROW_AMAX_FLOOR)

# (N, K, BM) -> (BN, BK, SK, num_warps, num_stages), SGLang's MI355X tuning.
FP8W_SMALL_M_CONFIGS = {
    (4608, 8192, 32): (128, 256, 4, 8, 3),
    (4608, 8192, 64): (128, 256, 4, 8, 3),
    (4608, 8192, 128): (64, 256, 2, 8, 3),
    (8192, 2048, 32): (32, 512, 1, 4, 3),
    (8192, 2048, 64): (32, 512, 1, 4, 3),
    (8192, 2048, 128): (32, 256, 1, 8, 3),
    (4096, 8192, 32): (64, 256, 4, 8, 3),
    (4096, 8192, 64): (64, 256, 4, 8, 3),
    (4096, 8192, 128): (64, 256, 4, 8, 3),
    (8192, 4096, 32): (64, 256, 2, 8, 3),
    (8192, 4096, 64): (64, 256, 2, 8, 3),
    (8192, 4096, 128): (64, 256, 2, 8, 3),
    (2304, 8192, 32): (64, 256, 4, 8, 3),
    (2304, 8192, 64): (64, 256, 4, 8, 3),
    (2304, 8192, 128): (64, 256, 4, 4, 3),
}


@triton.jit
def fp8_row_codes(x, mask):
    """FP8 E4M3 codes and FP32 scale of one row ``x`` (FP32, masked lanes
    ignored): scale = max(row amax, _ROW_AMAX_FLOOR) / _FP8_MAX,
    SGLang's per-row recipe (fused_norm_residual fp8_out). Every producer of
    the FP8 split-K GEMM's activation rows calls this, so their bits agree."""
    amax = tl.max(tl.where(mask, tl.abs(x), 0.0), axis=0)
    scale = tl.maximum(amax, _ROW_AMAX_FLOOR_C) / _FP8_MAX_C
    return (x / scale).to(tl.float8e4nv), scale


@triton.jit
def _quantize_rows_kernel(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(X + row.to(tl.int64) * K + offs, mask=mask, other=0.0).to(tl.float32)
    codes, scale = fp8_row_codes(x, mask)
    tl.store(Q + row.to(tl.int64) * K + offs, codes, mask=mask)
    tl.store(S + row, scale)


@triton.jit(do_not_specialize=["M"])
def _fp8_rowmajor_partial(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    pn, pk = tl.program_id(0), tl.program_id(1)
    KT_PER: tl.constexpr = K // BK // SK
    m = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    n = pn * BN + tl.arange(0, BN)
    bp = W + n[None, :].to(tl.int64) * K + (pk * KT_PER * BK + kk)[:, None]
    ap = X + m[:, None] * K + (pk * KT_PER * BK + kk)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(KT_PER):
        a = tl.load(ap, mask=m[:, None] < M, other=0.0)
        b = tl.load(bp, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    # int32 offsets: SK * M * N < 2**31 for every tuned entry at M <= 256,
    # checked where the configs are registered (triton_fp8_channel_splitk).
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, mask=m[:, None] < M)


@triton.jit(do_not_specialize=["M"])
def _sum_scaled(P, XS, WS, Y, M, N: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ok = i < M * N
    s = tl.arange(0, SK)
    # int32 offsets, bounded as in _fp8_rowmajor_partial (SK * M * N < 2**31).
    p = tl.load(P + s[:, None] * M * N + i[None, :], mask=ok[None, :], other=0.0)
    y = (
        tl.sum(p, axis=0)
        * tl.load(XS + i // N, mask=ok, other=0.0)
        * tl.load(WS + i % N, mask=ok, other=0.0)
    )
    tl.store(Y + i, y.to(Y.dtype.element_ty), mask=ok)


@torch.no_grad()
def quantize_weight_fp8_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 ``[N, K]`` copy and FP32 per-output-channel scales ``[N]`` of a
    BF16 weight (scale = channel amax x FP32 1 / 448), as SGLang converts at load."""
    values = weight.float()
    # Multiplied by FP32 1 / 448: how a GPU tensor divided by a Python scalar
    # evaluates in PyTorch, spelled out so every device gives these bits.
    scale = values.abs().amax(1).clamp_min(_CHANNEL_AMAX_FLOOR) * (1.0 / _FP8_MAX)
    return (values / scale[:, None]).to(_FP8).contiguous(), scale.contiguous()
