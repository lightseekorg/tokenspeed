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

"""Sparse MLA attention over MXFP4 latent cache rows.

The rows are ``ops.kvcache.mla_mxfp4``'s: ``kv_lora_rank / 2`` bytes of
nibble-packed E2M1 codes, one UE8M0 exponent per 32 latent values, then the
BF16 RoPE key. The kernel reads the selected rows in place -- no BF16 copy
of the cache -- and attends them with the absorbed query, as the BF16 DSA
cores do (``dsa.triton``: same key ranges, split merge and output layout).

The latent ``q . k`` takes one of two explicit forms (``qk_format``):

- ``"mxfp8"``: the query's latent part is quantized in-kernel to MXFP8 (E4M3
  with a UE8M0 scale per 32 channels) and multiplied with the packed codes
  and their exponents by ``tl.dot_scaled`` -- on gfx950 the block-scaled
  ``v_mfma_scale_f32_*_f8f6f4`` instructions, which consume the cache bytes
  without dequantizing them. The query rounding (E4M3, 3 mantissa bits) is
  this form's only error beyond the cache's.
- ``"bf16"``: the codes are decoded to BF16 (exactly: an E2M1 value times a
  power of two) and multiplied with the BF16 query by a BF16 MFMA.

The value product ``p . v`` always runs in BF16 on the decoded tile: the
latent's scales run along the key axis there, which a block-scaled MFMA
(scales along the reduction axis) cannot apply. The RoPE ``q . k`` is BF16.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsa.triton import (
    _dsa_key_range,
    _dsa_merge_splits_kernel,
    _dsa_softmax_update,
    _dsa_store_output,
    _num_kv_splits,
)
from tokenspeed_kernel.ops.kvcache.mla_mxfp4 import (
    mla_latent_mxfp4_geometry,
    mla_mxfp4_row_bytes,
)
from tokenspeed_kernel.ops.quantization._triton_mxfp4 import (
    _e2m1_decode,
    _ue8m0_exponent,
)
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "MXFP4_QK_FORMATS",
    "mxfp4_sparse_mla_attention",
    "mxfp4_sparse_mla_prefill",
]

# The latent q . k forms; see the module docstring.
MXFP4_QK_FORMATS = ("mxfp8", "bf16")

_BLOCK_H = 16
_BLOCK_N = 32


@triton.jit
def _mxfp4_sparse_mla_kernel(
    q,
    kv,  # u8 view of the cache rows
    kv_rope,  # bf16 view of the same bytes
    topk_indices,
    topk_lens,
    out,
    partial_out,
    partial_lse,
    q_stride_t,
    q_stride_h,
    topk_stride,
    page_size,
    page_stride,  # bytes
    row_stride,  # bytes
    softmax_scale,
    num_heads: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    topk: tl.constexpr,
    QK_MXFP8: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    split = tl.program_id(2)
    head_mask = heads < num_heads
    rows = token * num_heads + heads
    groups: tl.constexpr = kv_lora_rank // 32
    code_cols = tl.arange(0, kv_lora_rank // 2)
    group_cols = tl.arange(0, groups)
    q_rows = q + token * q_stride_t + heads.to(tl.int64) * q_stride_h

    q_nope = tl.load(
        q_rows[:, None] + tl.arange(0, kv_lora_rank)[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    if QK_MXFP8:
        # MXFP8 query: per 32 channels the smallest power of two that brings
        # the amax within E4M3's 448 (floored so the scale stays normal).
        q_groups = tl.reshape(q_nope.to(tl.float32), (BLOCK_H, groups, 32))
        amax = tl.max(tl.abs(q_groups), 2)
        scaled_max = tl.div_rn(tl.maximum(amax, 448.0 * 1.1754943508222875e-38), 448.0)
        q_exponent = tl.minimum(_ue8m0_exponent(scaled_max), 254)
        q_scale = (q_exponent << 23).to(tl.float32, bitcast=True)
        q_fp8 = tl.reshape(
            tl.div_rn(q_groups, q_scale[:, :, None]), (BLOCK_H, kv_lora_rank)
        ).to(tl.float8e4nv)
        q_exponent = q_exponent.to(tl.uint8)
    else:
        q_nope = q_nope.to(tl.bfloat16)
    if qk_rope_head_dim > 0:
        rope_dims = tl.arange(0, qk_rope_head_dim)
        q_rope = tl.load(
            q_rows[:, None] + kv_lora_rank + rope_dims[None, :],
            mask=head_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        # The RoPE key's BF16 element offset within a row.
        rope_start: tl.constexpr = (kv_lora_rank // 2 + groups) // 2

    start, end = _dsa_key_range(topk_lens, token, split, topk, NUM_SPLITS, BLOCK_N)
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, kv_lora_rank], tl.float32)
    for block_start in range(start, end, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        slots = tl.load(
            topk_indices + token * topk_stride + cols, mask=cols < end, other=-1
        )
        valid = slots >= 0
        slots = tl.maximum(slots, 0).to(tl.int64)
        kv_rows = (slots // page_size) * page_stride + (slots % page_size) * row_stride
        codes = tl.load(
            kv + kv_rows[:, None] + code_cols[None, :], mask=valid[:, None], other=0
        )
        exponents = tl.load(
            kv + kv_rows[:, None] + kv_lora_rank // 2 + group_cols[None, :],
            mask=valid[:, None],
            other=0,
        )
        # The values: E2M1 codes (even element in the low nibble) times the
        # group's power of two, exact in BF16.
        codes32 = codes.to(tl.int32)
        values = _e2m1_decode(tl.join(codes32 & 0xF, codes32 >> 4))
        values = tl.reshape(values, (BLOCK_N, groups, 32))
        scale = (exponents.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        latent = tl.reshape(values * scale[:, :, None], (BLOCK_N, kv_lora_rank)).to(
            tl.bfloat16
        )
        if QK_MXFP8:
            scores = tl.dot_scaled(
                q_fp8, q_exponent, "e4m3", tl.trans(codes), exponents, "e2m1"
            )
        else:
            scores = tl.dot(q_nope, tl.trans(latent))
        if qk_rope_head_dim > 0:
            k_rope = tl.load(
                kv_rope + (kv_rows[:, None] // 2 + rope_start) + rope_dims[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            scores = tl.dot(q_rope, tl.trans(k_rope), scores)
        m_i, l_i, acc = _dsa_softmax_update(
            scores, valid, latent, m_i, l_i, acc, softmax_scale
        )

    _dsa_store_output(
        out,
        partial_out,
        partial_lse,
        partial_lse,
        False,
        acc,
        m_i,
        l_i,
        rows,
        head_mask,
        split,
        kv_lora_rank,
        NUM_SPLITS,
    )


def mxfp4_sparse_mla_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    kv_lora_rank: int,
    softmax_scale: float,
    qk_format: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse MLA over MXFP4 latent rows; prefill and decode rows alike.

    Args:
        q: ``[tokens, heads, kv_lora_rank + rope_dim]`` absorbed query, BF16
            or FP16, unit channel stride.
        kv_cache: the uint8 ``[slots, 1, row_bytes]`` or ``[pages, page_size,
            1, row_bytes]`` view ``quantize_store_mla_latent_mxfp4`` writes.
        topk_slots: ``[tokens, topk]`` int32 cache slots; -1 is no key.
        topk_lens: ``[tokens]`` live leading entries per row, or None for
            every entry.
        kv_lora_rank: latent width; the rest of ``q`` is the RoPE part.
        softmax_scale: scale of ``q . k`` before the softmax.
        qk_format: ``"mxfp8"`` (block-scaled MFMA, gfx950 only) or ``"bf16"``.
        out: optional ``[tokens, heads, kv_lora_rank]`` (or any view of that
            size) output in ``q``'s dtype.

    Returns:
        ``[tokens, heads, kv_lora_rank]`` attention over the latent values.
    """
    if qk_format not in MXFP4_QK_FORMATS:
        raise ValueError(
            f"qk_format must be one of {list(MXFP4_QK_FORMATS)}, got {qk_format!r}"
        )
    if qk_format == "mxfp8" and not current_platform().is_cdna4:
        raise RuntimeError(
            "the MXFP8 x MXFP4 latent product runs on gfx950's block-scaled MFMA"
        )
    if q.dim() != 3 or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"q must be [tokens, heads, rank + rope] BF16/FP16, got "
            f"{tuple(q.shape)} {q.dtype}"
        )
    tokens, num_heads, head_dim = q.shape
    rope = head_dim - kv_lora_rank
    if rope < 0 or (rope & (rope - 1)) or q.stride(-1) != 1:
        raise ValueError(
            f"q {tuple(q.shape)} is not rank {kv_lora_rank} + a power-of-two "
            "RoPE part with dense channels"
        )
    page_size, page_stride, row_stride = mla_latent_mxfp4_geometry(
        kv_cache, mla_mxfp4_row_bytes(kv_lora_rank, rope)
    )
    if topk_slots.dim() != 2 or topk_slots.shape[0] != tokens:
        raise ValueError(
            f"topk_slots {tuple(topk_slots.shape)} is not [{tokens}, topk]"
        )
    topk_slots = topk_slots.to(torch.int32).contiguous()
    topk = topk_slots.shape[1]
    topk_lens = (
        torch.full((tokens,), topk, dtype=torch.int32, device=q.device)
        if topk_lens is None
        else topk_lens.to(torch.int32).contiguous()
    )
    shape = (tokens, num_heads, kv_lora_rank)
    if out is not None and (
        out.numel() != tokens * num_heads * kv_lora_rank or out.dtype != q.dtype
    ):
        raise ValueError(
            f"out {tuple(out.shape)} {out.dtype} does not hold {shape} {q.dtype}"
        )
    # Written in place where out is dense, else copied into it.
    direct = out is not None and out.is_contiguous()
    result = (
        out.view(shape)
        if direct
        else torch.empty(shape, dtype=q.dtype, device=q.device)
    )
    if tokens == 0:
        return result if out is None else out
    head_blocks = triton.cdiv(num_heads, _BLOCK_H)
    num_splits = _num_kv_splits(tokens * head_blocks, topk)
    partial_out = partial_lse = result
    if num_splits > 1:
        partial_out = torch.empty(
            (tokens, num_heads, num_splits, kv_lora_rank),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (tokens, num_heads, num_splits), dtype=torch.float32, device=q.device
        )
    _mxfp4_sparse_mla_kernel[(tokens, head_blocks, num_splits)](
        q,
        kv_cache,
        kv_cache.view(torch.bfloat16),
        topk_slots,
        topk_lens,
        result,
        partial_out,
        partial_lse,
        q.stride(0),
        q.stride(1),
        topk_slots.stride(0),
        page_size,
        page_stride,
        row_stride,
        float(softmax_scale),
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope,
        topk=topk,
        QK_MXFP8=qk_format == "mxfp8",
        NUM_SPLITS=num_splits,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        # Two stages pipeline the gathered rows behind the dots: 1.20x at 4096 query
        # tokens and 1.03x at one on MI355X (Kimi TP8 shapes), same output as one stage.
        num_stages=2,
    )
    if num_splits > 1:
        block_d = min(kv_lora_rank, 128)
        _dsa_merge_splits_kernel[(tokens * num_heads, kv_lora_rank // block_d)](
            partial_out,
            partial_lse,
            result,
            partial_lse,
            False,
            kv_lora_rank,
            num_splits,
            BLOCK_D=block_d,
            num_warps=4,
        )
    if out is None:
        return result
    if not direct:
        out.copy_(result.view(out.shape))
    return out


def mxfp4_sparse_mla_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    kv_lora_rank: int,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse MLA prefill over MXFP4 latent rows on gfx950's Gluon kernel.

    The arithmetic of ``mxfp4_sparse_mla_attention(qk_format="mxfp8")`` --
    MXFP8 query times the MXFP4 codes on the block-scaled MFMA, BF16 RoPE,
    BF16 ``p . v`` on the exactly decoded values -- scheduled for prefill
    rows (``launch_gluon_mxfp4_sparse_mla_prefill_gfx950``). It serves the
    DeepSeek/Kimi MLA geometry only: a BF16 query of 512 latent and 64 RoPE
    channels and dense cache rows; anything else raises.

    Args and return value as ``mxfp4_sparse_mla_attention``'s.
    """
    if not current_platform().is_cdna4:
        raise RuntimeError("the MXFP4 sparse MLA prefill kernel runs on gfx950")
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.mxfp4_prefill import (
        launch_gluon_mxfp4_sparse_mla_prefill_gfx950,
    )

    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, rank + rope], got {q.shape}")
    tokens, num_heads, head_dim = q.shape
    rope = head_dim - kv_lora_rank
    page_size, page_stride, row_stride = mla_latent_mxfp4_geometry(
        kv_cache, mla_mxfp4_row_bytes(kv_lora_rank, rope)
    )
    if topk_slots.dim() != 2 or topk_slots.shape[0] != tokens:
        raise ValueError(
            f"topk_slots {tuple(topk_slots.shape)} is not [{tokens}, topk]"
        )
    topk_slots = topk_slots.to(torch.int32).contiguous()
    topk_lens = (
        torch.full((tokens,), topk_slots.shape[1], dtype=torch.int32, device=q.device)
        if topk_lens is None
        else topk_lens.to(torch.int32).contiguous()
    )
    shape = (tokens, num_heads, kv_lora_rank)
    if out is not None and (
        out.numel() != tokens * num_heads * kv_lora_rank or out.dtype != q.dtype
    ):
        raise ValueError(
            f"out {tuple(out.shape)} {out.dtype} does not hold {shape} {q.dtype}"
        )
    direct = out is not None and out.is_contiguous()
    result = (
        out.view(shape)
        if direct
        else torch.empty(shape, dtype=q.dtype, device=q.device)
    )
    launch_gluon_mxfp4_sparse_mla_prefill_gfx950(
        q,
        kv_cache,
        topk_slots,
        topk_lens,
        page_size=page_size,
        page_stride=page_stride,
        row_stride=row_stride,
        softmax_scale=softmax_scale,
        out=result,
    )
    if out is None:
        return result
    if not direct:
        out.copy_(result.view(out.shape))
    return out
