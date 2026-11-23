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

"""Torch references of a DFlash2 draft layer's math (the regular path's
rounding points, ``runtime/models/dflash2.py``) for the monokernel tests, one
TP8 rank at a time.

The draft is Muse Spark's (RadixArk/Muse-Spark-Open-DFlash2): hidden 8192,
intermediate 28672, a grouped two-tap convolution of 512 groups of 16 channels
around each sublayer, blocks of 8 rows a request.
"""

from __future__ import annotations

import muse_spark_reference
import torch
from tokenspeed_kernel.numerics.torch.conv import torch_block_grouped_conv
from tokenspeed_kernel.numerics.torch.layernorm import torch_rmsnorm

HIDDEN, INTER, WORLD = 8192, 28672, 8
INTER_RANK = INTER // WORLD  # 3584 gate rows and 3584 up rows a rank
BLOCK, TAPS, GROUP_SIZE = 8, 2, 16
GROUPS = HIDDEN // GROUP_SIZE  # 512
COEFFICIENTS = 2 * TAPS * GROUPS  # side x tap x group: 2048
EPS = 1e-5


def assert_bf16_close(got: torch.Tensor, want: torch.Tensor) -> None:
    """Assert ``got`` is within one BF16 step of ``want``, or a step of
    ``want``'s largest value where a sum cancels to near zero."""
    torch.testing.assert_close(
        got.float(),
        want.float(),
        rtol=2**-7,
        atol=want.float().abs().max().item() * 2**-9,
    )


def grouped_conv(
    x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    """``dflash2_grouped_conv``: FP32 sum over the taps in tap order of
    ``(base + delta) * x[t - tap]`` within each block of ``BLOCK`` rows,
    rounded once to BF16 (the shared reference ``torch_block_grouped_conv``).

    Args:
        x: BF16 ``[R, HIDDEN]``, ``R`` a multiple of ``BLOCK``.
        delta: BF16 ``[R, TAPS, GROUPS]`` (one side of the coefficients).
        base: BF16 ``[TAPS, HIDDEN]`` (the same side of ``base_kernel``).
    """
    return torch_block_grouped_conv(x, delta, base, BLOCK, GROUP_SIZE)


def draft_attention(q, keys, values):
    """The draft layer's attention of one row (``q [heads, 128]``, one KV
    head ``keys`` / ``values`` ``[n, 128]``, every row seeing the same keys)
    in FP64 with logit scale ``128 ** -0.5``, and a bound on the kernel's
    BF16 output's difference from it: the error of its BF16 softmax weights
    and FP32 sums (``verify_attention``), then one BF16 rounding (unit
    roundoff 2^-8).

    Returns ``(exact, bound)``, both FP64 ``[heads, 128]``.
    """
    empty = keys[:0]
    exact, error = muse_spark_reference.verify_attention(
        q, q, empty, empty, keys, values, q.shape[-1] ** -0.5
    )
    return exact, error + 2**-8 * (exact.abs() + error)


def tp_sum_norm(
    parts: list[torch.Tensor],
    residual: torch.Tensor,
    post_norm: torch.Tensor,
    *,
    round_tp_sum: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ranks' sum of the attention partials, the residual add and the
    post-attention RMSNorm.

    Args:
        parts: The ``WORLD`` ranks' BF16 attention partials ``[R, HIDDEN]``
            (after the attention convolution), summed in rank order.
        residual: BF16 ``[R, HIDDEN]``, the same on every rank.
        post_norm: BF16 ``[HIDDEN]`` gain of ``post_attention_layernorm``.
        round_tp_sum: Round the ranks' sum to BF16 before the residual add,
            as the separate all-reduce does; the fused all-reduce + residual
            + RMSNorm adds it in FP32.

    Returns:
        ``(hidden, residual)``: the BF16 MLP input and the BF16 residual
        after the attention sublayer.
    """
    total = torch.zeros_like(parts[0], dtype=torch.float32)
    for part in parts:  # rank order
        total += part.float()
    if round_tp_sum:
        total = total.to(torch.bfloat16).float()
    # The norm reads the FP32 sum with the residual; only the stored residual
    # is rounded.
    hidden, stream = torch_rmsnorm(total, post_norm, EPS, residual)
    return hidden.to(torch.bfloat16), stream.to(torch.bfloat16)


def mlp(
    hidden: torch.Tensor,
    coef_weight: torch.Tensor,
    base: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """The MLP with its convolution, from the normed hidden to this rank's
    BF16 partial (after the MLP convolution; the next layer's input norm sums
    the ranks').

    Args:
        hidden: BF16 ``[R, HIDDEN]`` MLP input.
        coef_weight: BF16 ``[COEFFICIENTS, HIDDEN]`` ``mlp_conv``'s
            ``kernel_projection`` (replicated).
        base: BF16 ``[2, TAPS, HIDDEN]`` ``mlp_conv``'s ``base_kernel``.
        gate_up: BF16 ``[2 * INTER_RANK, HIDDEN]``: this rank's gate rows,
            then its up rows (the merged column-parallel shard).
        down: BF16 ``[HIDDEN, INTER_RANK]``: this rank's columns of
            ``down_proj``.
    """
    delta = (hidden.float() @ coef_weight.float().T).to(torch.bfloat16)
    delta = delta.view(-1, 2, TAPS, GROUPS)
    hidden = grouped_conv(hidden, delta[:, 0], base[0])
    projected = (hidden.float() @ gate_up.float().T).to(torch.bfloat16)
    gate, up = projected.float().split(INTER_RANK, dim=-1)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    partial = (act.float() @ down.float().T).to(torch.bfloat16)
    return grouped_conv(partial, delta[:, 1], base[1])


def mlp_half(
    parts: list[torch.Tensor],
    residual: torch.Tensor,
    post_norm: torch.Tensor,
    coef_weight: torch.Tensor,
    base: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    round_tp_sum: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The draft layer's MLP half on one rank, from the attention sublayer's
    partials to this rank's MLP partial (:func:`tp_sum_norm`, then
    :func:`mlp`).

    Returns:
        ``(partial, residual)``: this rank's BF16 MLP partial and the BF16
        residual after the attention sublayer.
    """
    hidden, residual_out = tp_sum_norm(
        parts, residual, post_norm, round_tp_sum=round_tp_sum
    )
    return mlp(hidden, coef_weight, base, gate_up, down), residual_out
