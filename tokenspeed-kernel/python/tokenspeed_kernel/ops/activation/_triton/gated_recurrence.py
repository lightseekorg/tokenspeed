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

"""Capped logit sums and gated tanh recurrences matching Torch's BF16 statements."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.ops.activation.triton import (
    _TANH_SOFTCAP_BLOCK,
    _TANH_SOFTCAP_LAUNCH_OPTIONS,
    _mul_rn_f32,
    tanh_softcap_cap_normal,
    tanh_softcap_scalars,
)

__all__ = [
    "capped_logit_sum",
    "gated_tanh_recurrence",
    "projected_gated_tanh_recurrence",
]


@triton.jit(do_not_specialize=["valid_cols"])
def _capped_logit_sum_kernel(
    base_ptr,
    correction_ptr,
    out_ptr,
    stride_base,
    stride_correction,
    stride_out,
    valid_cols,
    out_cols,
    inv_cap,
    cap,
    HAS_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = cols < valid_cols
    base = tl.load(base_ptr + row * stride_base + cols, mask=valid, other=0.0)
    correction = tl.load(
        correction_ptr + row * stride_correction + cols, mask=valid, other=0.0
    )
    # Torch's BF16 add, then tanh_softcap's FP32 statement, rounded to BF16.
    y = (base.to(tl.float32) + correction.to(tl.float32)).to(tl.bfloat16)
    if HAS_CAP:
        y = _mul_rn_f32(y.to(tl.float32), inv_cap)
        y = libdevice.tanh(y)
        y = _mul_rn_f32(y, cap).to(tl.bfloat16)
    y = tl.where(valid, y, -float("inf"))
    tl.store(out_ptr + row * stride_out + cols, y, mask=cols < out_cols)


def capped_logit_sum(
    base: torch.Tensor,
    correction: torch.Tensor,
    cap: float | None,
    valid_cols: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out = mask(softcap(base + correction))`` in one pass, matching the
    Torch statement ``(cap * torch.tanh((base + correction).float() / cap))
    .to(torch.bfloat16)`` bit for bit, with every column from ``valid_cols``
    on set to ``-inf``.

    Args:
        base: BF16 ``[rows, >= valid_cols]`` logits with dense columns.
        correction: BF16 logits shaped like ``base``.
        cap: The soft cap (see :func:`tanh_softcap_rejection`), or None for
            no cap.
        valid_cols: Leading columns that keep their logits.
        out: BF16 ``[rows, width]`` destination with dense columns; ``width``
            may exceed the inputs' columns, which reads as ``-inf`` padding.

    Returns:
        ``out``.
    """
    rows, width = out.shape
    if (
        base.dtype != torch.bfloat16
        or correction.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or base.shape[0] != rows
        or correction.shape != base.shape
        or not 0 <= valid_cols <= min(base.shape[1], width)
        or any(t.stride(-1) != 1 for t in (base, correction, out))
    ):
        raise ValueError("capped_logit_sum needs matching BF16 rows with dense columns")
    if cap is not None and not tanh_softcap_cap_normal(cap):
        raise ValueError(f"capped_logit_sum cap must be a normal FP32, got {cap!r}")
    inv_cap, cap32 = tanh_softcap_scalars(cap) if cap is not None else (1.0, 1.0)
    if rows and width:
        _capped_logit_sum_kernel[(rows, triton.cdiv(width, _TANH_SOFTCAP_BLOCK))](
            base,
            correction,
            out,
            base.stride(0),
            correction.stride(0),
            out.stride(0),
            valid_cols,
            width,
            inv_cap,
            cap32,
            HAS_CAP=cap is not None,
            BLOCK=_TANH_SOFTCAP_BLOCK,
            **_TANH_SOFTCAP_LAUNCH_OPTIONS,
        )
    return out


@triton.jit
def _gated_tanh_update(gate, candidate, output, state):
    """The BF16 recurrence from BF16 projection values (held in FP32): each
    Torch BF16 op computes in FP32 and rounds to BF16: sigmoid as
    1 / (1 + exp(-x)), then gate * state, 1 - gate, tanh(candidate), their
    product and the sum. Returns the BF16 ``(new_state, output)``."""
    gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    kept = _mul_rn_f32(gate, state.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    complement = (1.0 - gate).to(tl.bfloat16).to(tl.float32)
    candidate = libdevice.tanh(candidate).to(tl.bfloat16).to(tl.float32)
    update = _mul_rn_f32(complement, candidate).to(tl.bfloat16).to(tl.float32)
    return (kept + update).to(tl.bfloat16), libdevice.tanh(output).to(tl.bfloat16)


@triton.jit
def _gated_tanh_recurrence_kernel(
    projected_ptr,
    state_ptr,
    new_state_ptr,
    output_ptr,
    stride_projected,
    stride_state,
    width,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    base = projected_ptr + row * stride_projected + cols
    gate = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    candidate = tl.load(base + width, mask=mask, other=0.0).to(tl.float32)
    output = tl.load(base + 2 * width, mask=mask, other=0.0).to(tl.float32)
    state = tl.load(state_ptr + row * stride_state + cols, mask=mask, other=0.0)
    new_state, output = _gated_tanh_update(gate, candidate, output, state)
    tl.store(new_state_ptr + row * width + cols, new_state, mask=mask)
    tl.store(output_ptr + row * width + cols, output, mask=mask)


@triton.jit
def _projected_gated_tanh_recurrence_kernel(
    static_ptr,
    state_ptr,
    tokens_ptr,
    embedding_ptr,
    weight_ptr,
    new_state_ptr,
    output_ptr,
    stride_static,
    stride_state,
    stride_embedding,
    stride_weight,
    token_width,
    token_shard,
    WIDTH: tl.constexpr,
    EMBED: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One row's ``BLOCK_C`` state columns: their gate, candidate and output
    projections add the row's ``[state | embedding[token]]`` products to the
    FP32 static terms, round once to BF16, then update. A token indexes
    shards of ``token_shard`` rows padded to ``token_width``."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = cols < WIDTH
    token = tl.load(tokens_ptr + row).to(tl.int64)
    token -= token // token_width * (token_width - token_shard)
    static = static_ptr + row * stride_static + cols
    gate = tl.load(static, mask=mask_c, other=0.0)
    candidate = tl.load(static + WIDTH, mask=mask_c, other=0.0)
    output = tl.load(static + 2 * WIDTH, mask=mask_c, other=0.0)
    weights = weight_ptr + cols[:, None].to(tl.int64) * stride_weight
    for k0 in tl.static_range(0, WIDTH + EMBED, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        from_state = k < WIDTH
        x = tl.load(state_ptr + row * stride_state + k, mask=from_state, other=0.0)
        x += tl.load(
            embedding_ptr + token * stride_embedding + (k - WIDTH),
            mask=(k >= WIDTH) & (k < WIDTH + EMBED),
            other=0.0,
        )
        x = x.to(tl.float32)[None, :]
        tile_mask = mask_c[:, None] & (k < WIDTH + EMBED)[None, :]
        tile = weights + k[None, :]
        gate += tl.sum(tl.load(tile, tile_mask, other=0.0).to(tl.float32) * x, 1)
        candidate += tl.sum(
            tl.load(tile + WIDTH * stride_weight, tile_mask, other=0.0).to(tl.float32)
            * x,
            1,
        )
        output += tl.sum(
            tl.load(tile + 2 * WIDTH * stride_weight, tile_mask, other=0.0).to(
                tl.float32
            )
            * x,
            1,
        )
    state = tl.load(state_ptr + row * stride_state + cols, mask=mask_c, other=0.0)
    new_state, output = _gated_tanh_update(
        gate.to(tl.bfloat16).to(tl.float32),
        candidate.to(tl.bfloat16).to(tl.float32),
        output.to(tl.bfloat16).to(tl.float32),
        state,
    )
    tl.store(new_state_ptr + row * WIDTH + cols, new_state, mask=mask_c)
    tl.store(output_ptr + row * WIDTH + cols, output, mask=mask_c)


def gated_tanh_recurrence(
    projected: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One BF16 gated recurrence step from its ``[gate | candidate | output]``
    projection, bit for bit the Torch statements::

        gate = torch.sigmoid(gate)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate)
        output = torch.tanh(output)

    Args:
        projected: BF16 ``[rows, 3 * width]`` projection with dense columns.
        state: BF16 ``[rows, width]`` state with dense columns.

    Returns:
        Contiguous BF16 ``(new_state, output)``, each ``[rows, width]``.
    """
    rows, width = state.shape
    if (
        projected.dtype != torch.bfloat16
        or state.dtype != torch.bfloat16
        or projected.shape != (rows, 3 * width)
        or projected.stride(-1) != 1
        or state.stride(-1) != 1
    ):
        raise ValueError("gated_tanh_recurrence needs BF16 [rows, 3w] and [rows, w]")
    new_state = torch.empty((rows, width), dtype=state.dtype, device=state.device)
    output = torch.empty_like(new_state)
    if rows:
        _gated_tanh_recurrence_kernel[(rows,)](
            projected,
            state,
            new_state,
            output,
            projected.stride(0),
            state.stride(0),
            width,
            BLOCK=triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return new_state, output


def projected_gated_tanh_recurrence(
    static: torch.Tensor,
    state: torch.Tensor,
    tokens: torch.Tensor,
    embedding: torch.Tensor,
    weight: torch.Tensor,
    *,
    shard_padding: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``gated_tanh_recurrence`` of a projection whose step-invariant inputs
    were projected beforehand::

        projected = bf16(static + [state | embedding[tokens]] @ weight.T)

    with one FP32 sum per element and one rounding: a step reads only the
    recurrent columns of the projection weight.

    Args:
        static: FP32 ``[rows, 3 * width]`` projection of the step-invariant
            inputs, bias included; dense columns.
        state: BF16 ``[rows, width]`` state with dense columns.
        tokens: Integer ``[rows]`` rows of ``embedding`` to project.
        embedding: BF16 ``[vocab, embed]`` with dense columns.
        weight: BF16 ``[3 * width, width + embed]`` projection columns of
            ``[state | embedding[tokens]]``, dense columns.
        shard_padding: ``(padded, shard)`` when ``tokens`` are
            ``rank * padded + column`` indices over vocabulary shards of
            ``shard`` rows each padded to ``padded`` columns (a distributed
            argmax over padded rows): the row read is ``rank * shard +
            column``. None when ``tokens`` are rows of ``embedding``.

    Returns:
        Contiguous BF16 ``(new_state, output)``, each ``[rows, width]``.
    """
    rows, width = state.shape
    embed = embedding.shape[1]
    if (
        static.dtype != torch.float32
        or state.dtype != torch.bfloat16
        or embedding.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or static.shape != (rows, 3 * width)
        or tokens.shape != (rows,)
        or not tokens.is_contiguous()
        or weight.shape != (3 * width, width + embed)
        or any(t.stride(-1) != 1 for t in (static, state, embedding, weight))
    ):
        raise ValueError(
            "projected_gated_tanh_recurrence needs FP32 [rows, 3w] static terms, "
            "BF16 [rows, w] state and BF16 [3w, w + e] weights"
        )
    new_state = torch.empty((rows, width), dtype=state.dtype, device=state.device)
    output = torch.empty_like(new_state)
    if rows:
        # 4-column tiles over all of K: a 256-wide step takes 2.3 us on B300, 5.3 at 16.
        block_c = 4
        _projected_gated_tanh_recurrence_kernel[(rows, triton.cdiv(width, block_c))](
            static,
            state,
            tokens,
            embedding,
            weight,
            new_state,
            output,
            static.stride(0),
            state.stride(0),
            embedding.stride(0),
            weight.stride(0),
            # Rows are their own unpadded shard.
            *(shard_padding or (embedding.shape[0], embedding.shape[0])),
            WIDTH=width,
            EMBED=embed,
            BLOCK_C=block_c,
            BLOCK_K=512,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return new_state, output
