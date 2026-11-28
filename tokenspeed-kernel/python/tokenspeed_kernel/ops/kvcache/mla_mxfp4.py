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

"""MXFP4 MLA latent cache rows: Triton quantize-store and dequantize-gather.

One token's row of an MLA layer is ``kv_lora_rank`` latent values (keys and
values share them) and ``rope_dim`` RoPE key values. An MXFP4 row stores, in
bytes::

    [ kv_lora_rank / 2 packed E2M1 | kv_lora_rank / 32 UE8M0 | rope_dim BF16 ]

400 bytes for Kimi-K3.1's 512 + 64, against 576 for an FP8 row and 1152 for
BF16. Each group of 32 latent values shares one power-of-two scale, the
smallest that brings the group's absolute maximum within E2M1's 6.0 (the
OCP MX ``ceil`` rule); the codes round to nearest with ties to the even code
and the even element sits in the low nibble, the layout ``tl.dot_scaled``'s
``"e2m1"`` operand reads. The RoPE part is copied as BF16.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.quantization._triton_mxfp4 import (
    _e2m1_decode,
    _e2m1_encode,
    _ue8m0_exponent,
)

__all__ = [
    "MXFP4_LATENT_GROUP",
    "dequant_gather_mla_latent_mxfp4",
    "mla_latent_mxfp4_geometry",
    "mla_mxfp4_row_bytes",
    "quantize_store_mla_latent_mxfp4",
]

MXFP4_LATENT_GROUP = 32
# torch.finfo(torch.bfloat16).max: a sanitizing write stores +-inf as this,
# as set_mla_kv_buffer_triton does for a BF16 cache.
_BF16_MAX = tl.constexpr(3.3895313892515355e38)


def mla_mxfp4_row_bytes(kv_lora_rank: int, rope_dim: int) -> int:
    """Bytes of one MXFP4 MLA latent row."""
    if kv_lora_rank <= 0 or kv_lora_rank % (2 * MXFP4_LATENT_GROUP):
        raise ValueError(
            f"MXFP4 latent rows need kv_lora_rank % 64 == 0, got {kv_lora_rank}"
        )
    if rope_dim < 0:
        raise ValueError(f"rope_dim must be non-negative, got {rope_dim}")
    return kv_lora_rank // 2 + kv_lora_rank // MXFP4_LATENT_GROUP + 2 * rope_dim


def mla_latent_mxfp4_geometry(
    cache: torch.Tensor, row_bytes: int
) -> tuple[int, int, int]:
    """``(page_size, page_stride, row_stride)`` in bytes of a uint8
    ``[slots, 1, row_bytes]`` or ``[pages, page_size, 1, row_bytes]`` cache
    view; slot ``s`` starts at ``(s // page_size) * page_stride + (s %
    page_size) * row_stride``."""
    if cache.dtype != torch.uint8 or cache.shape[-1] != row_bytes:
        raise ValueError(
            f"an MXFP4 latent cache holds uint8 rows of {row_bytes} bytes, got "
            f"{tuple(cache.shape)} {cache.dtype}"
        )
    if cache.stride(-1) != 1:
        raise ValueError("MXFP4 latent cache rows must be contiguous")
    if cache.dim() == 3 and cache.shape[1] == 1:
        geometry = (1, cache.stride(0), cache.stride(0))
    elif cache.dim() == 4 and cache.shape[2] == 1:
        geometry = (cache.shape[1], cache.stride(0), cache.stride(1))
    else:
        raise ValueError(
            "an MXFP4 latent cache is a [slots, 1, row_bytes] or [pages, "
            f"page_size, 1, row_bytes] view, got {tuple(cache.shape)}"
        )
    # The BF16 RoPE part is read and written through a BF16 view of the rows.
    if any(stride % 2 for stride in geometry[1:]) or cache.data_ptr() % 2:
        raise ValueError("MXFP4 latent cache rows must start on 2-byte boundaries")
    return geometry


@triton.jit
def _quantize_store_mla_latent_mxfp4_kernel(
    kv_ptr,  # [T, RANK] latent rows
    pe_ptr,  # [T, ROPE] RoPE key rows
    cache_ptr,  # u8 cache rows
    slots_ptr,  # [num_slots] destination slots
    mask_ptr,  # [num_slots] bool write mask (read only when HAS_MASK)
    q_pe_ptr,  # [T, H, ROPE] query RoPE rows (read only when COPY_Q_PE)
    query_ptr,  # [T, H, RANK + ROPE] query (written only when COPY_Q_PE)
    num_slots,
    kv_stride,
    pe_stride,
    page_size,
    page_stride,  # u8 elements
    row_stride,  # u8 elements
    q_pe_stride_t,
    q_pe_stride_h,
    query_stride_t,
    query_stride_h,
    num_heads,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SANITIZE: tl.constexpr,
    COPY_Q_PE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    if COPY_Q_PE:
        # The query's RoPE channels: every row the prologue returns, written
        # or not.
        heads = tl.arange(0, BLOCK_HEADS)
        rope_cols = tl.arange(0, ROPE)
        head_mask = heads[:, None] < num_heads
        q_rope = tl.load(
            q_pe_ptr
            + tok * q_pe_stride_t
            + heads[:, None] * q_pe_stride_h
            + rope_cols[None, :],
            mask=head_mask,
        )
        tl.store(
            query_ptr
            + tok * query_stride_t
            + heads[:, None] * query_stride_h
            + RANK
            + rope_cols[None, :],
            q_rope,
            mask=head_mask,
        )
    if tok >= num_slots:
        return
    if HAS_MASK:
        if tl.load(mask_ptr + tok).to(tl.int1) == 0:
            return
    slot = tl.load(slots_ptr + tok).to(tl.int64)
    row = (
        cache_ptr + (slot // page_size) * page_stride + (slot % page_size) * row_stride
    )
    groups: tl.constexpr = RANK // 32

    x = tl.load(kv_ptr + tok * kv_stride + tl.arange(0, RANK)).to(tl.float32)
    if SANITIZE:
        x = tl.where(x != x, 0.0, x)
        x = tl.minimum(tl.maximum(x, -_BF16_MAX), _BF16_MAX)
    x = tl.reshape(x, (groups, 32))
    amax = tl.max(tl.abs(x), 1)
    # Floor amax at 6 * 2**-126 so the scale stays a normal FP32 number.
    scaled_max = tl.div_rn(tl.maximum(amax, 6.0 * 1.1754943508222875e-38), 6.0)
    exponent = tl.minimum(_ue8m0_exponent(scaled_max), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    codes = _e2m1_encode(tl.div_rn(x, scale[:, None])).to(tl.int32)
    pairs = tl.reshape(codes, (RANK // 2, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
    tl.store(row + tl.arange(0, RANK // 2), packed)
    tl.store(row + RANK // 2 + tl.arange(0, groups), exponent.to(tl.uint8))

    if ROPE > 0:
        pe = tl.load(pe_ptr + tok * pe_stride + tl.arange(0, ROPE)).to(tl.float32)
        if SANITIZE:
            pe = tl.where(pe != pe, 0.0, pe)
            pe = tl.minimum(tl.maximum(pe, -_BF16_MAX), _BF16_MAX)
        pe_row = (row + RANK // 2 + groups).to(tl.pointer_type(tl.bfloat16))
        tl.store(pe_row + tl.arange(0, ROPE), pe.to(tl.bfloat16))


def quantize_store_mla_latent_mxfp4(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    write_mask: torch.Tensor | None,
    sanitize: bool,
    q_pe: torch.Tensor | None = None,
    query: torch.Tensor | None = None,
) -> None:
    """Quantize the leading ``slots.numel()`` latent rows to MXFP4 and store
    them with their RoPE rows (as BF16) at ``slots``, in one launch that can
    also assemble the attention query's RoPE channels.

    Args:
        kv_c: ``[T, kv_lora_rank]`` latent rows (trailing stride 1).
        k_pe: ``[T, rope_dim]`` RoPE key rows (trailing stride 1).
        cache: uint8 ``[slots, 1, row_bytes]`` or ``[pages, page_size, 1,
            row_bytes]`` view, ``row_bytes = mla_mxfp4_row_bytes(...)``.
        slots: ``[S]`` destination slots, ``S <= T``; rows past ``S`` are not
            stored.
        write_mask: optional ``[S]`` bool; rows where it is False are not
            written.
        sanitize: store NaN as 0 and +-inf as +-BF16 max before quantizing.
        q_pe: optional ``[T, heads, rope_dim]`` query RoPE rows, copied into
            ``query[..., kv_lora_rank:]`` for all ``T`` rows.
        query: ``[T, heads, kv_lora_rank + rope_dim]`` with ``q_pe``.
    """
    kv_c = kv_c.reshape(kv_c.shape[0], -1)
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    t, rank = kv_c.shape
    rope = k_pe.shape[1]
    num_slots = slots.numel()
    if k_pe.shape[0] != t or slots.dim() != 1 or num_slots > t:
        raise ValueError(
            f"kv_c {tuple(kv_c.shape)}, k_pe {tuple(k_pe.shape)} and slots "
            f"{tuple(slots.shape)} must give one row per token, slots leading"
        )
    if write_mask is not None and write_mask.shape != slots.shape:
        raise ValueError("write_mask must match slots")
    if rope & (rope - 1):
        raise ValueError(f"rope_dim must be a power of two, got {rope}")
    if kv_c.stride(-1) != 1 or k_pe.stride(-1) != 1:
        raise ValueError("kv_c and k_pe must have dense channels")
    page_size, page_stride, row_stride = mla_latent_mxfp4_geometry(
        cache, mla_mxfp4_row_bytes(rank, rope)
    )
    copy_q_pe = q_pe is not None
    if copy_q_pe:
        if query is None or query.dim() != 3 or q_pe.dim() != 3:
            raise ValueError("q_pe needs the [T, heads, rank + rope] query")
        if (
            query.shape != (t, q_pe.shape[1], rank + rope)
            or q_pe.shape != (t, query.shape[1], rope)
            or query.stride(-1) != 1
            or q_pe.stride(-1) != 1
        ):
            raise ValueError(
                f"query {tuple(query.shape)} and q_pe {tuple(q_pe.shape)} do not "
                f"hold {t} rows of rank {rank} + rope {rope} channels"
            )
        if q_pe.data_ptr() == query[..., rank:].data_ptr():
            copy_q_pe = False
    if t == 0 or (num_slots == 0 and not copy_q_pe):
        return
    grid_rows = t if copy_q_pe else num_slots
    _quantize_store_mla_latent_mxfp4_kernel[(grid_rows,)](
        kv_c,
        k_pe,
        cache,
        slots,
        slots if write_mask is None else write_mask,
        q_pe if copy_q_pe else kv_c,
        query if copy_q_pe else kv_c,
        num_slots,
        kv_c.stride(0),
        k_pe.stride(0),
        page_size,
        page_stride,
        row_stride,
        q_pe.stride(0) if copy_q_pe else 0,
        q_pe.stride(1) if copy_q_pe else 0,
        query.stride(0) if copy_q_pe else 0,
        query.stride(1) if copy_q_pe else 0,
        query.shape[1] if copy_q_pe else 1,
        RANK=rank,
        ROPE=rope,
        HAS_MASK=write_mask is not None,
        SANITIZE=sanitize,
        COPY_Q_PE=copy_q_pe,
        BLOCK_HEADS=triton.next_power_of_2(query.shape[1]) if copy_q_pe else 1,
    )


@triton.jit
def _dequant_gather_mla_latent_mxfp4_kernel(
    cache_ptr,
    slots_ptr,  # [N] source slots, < 0 reads as a zero row
    out_ptr,  # [N, RANK + ROPE]
    out_stride,
    page_size,
    page_stride,
    row_stride,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + i).to(tl.int64)
    valid = slot >= 0
    slot = tl.maximum(slot, 0)
    row = (
        cache_ptr + (slot // page_size) * page_stride + (slot % page_size) * row_stride
    )
    groups: tl.constexpr = RANK // 32

    packed = tl.load(row + tl.arange(0, RANK // 2), mask=valid, other=0).to(tl.int32)
    codes = tl.join(packed & 0xF, packed >> 4)  # [RANK/2, 2]: low nibble first
    values = tl.reshape(_e2m1_decode(codes), (groups, 32))
    exponent = tl.load(row + RANK // 2 + tl.arange(0, groups), mask=valid, other=0)
    scale = (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    latent = tl.reshape(values * scale[:, None], (RANK,))
    out = out_ptr + i * out_stride
    tl.store(out + tl.arange(0, RANK), latent.to(out_ptr.dtype.element_ty))

    if ROPE > 0:
        pe_row = (row + RANK // 2 + groups).to(tl.pointer_type(tl.bfloat16))
        pe = tl.load(pe_row + tl.arange(0, ROPE), mask=valid, other=0.0)
        tl.store(out + RANK + tl.arange(0, ROPE), pe.to(out_ptr.dtype.element_ty))


def dequant_gather_mla_latent_mxfp4(
    cache: torch.Tensor,
    slots: torch.Tensor,
    out: torch.Tensor,
    *,
    kv_lora_rank: int,
) -> None:
    """Dequantize the MXFP4 latent rows at ``slots`` into ``out``.

    Args:
        cache: the uint8 view ``quantize_store_mla_latent_mxfp4`` writes.
        slots: ``[N]`` source slots; negative slots give zero rows.
        out: ``[N, kv_lora_rank + rope_dim]`` BF16, FP16 or FP32 rows
            (trailing stride 1), latent first.
        kv_lora_rank: latent width.
    """
    n = slots.shape[0]
    out2 = out.reshape(n, -1)
    rope = out2.shape[1] - kv_lora_rank
    page_size, page_stride, row_stride = mla_latent_mxfp4_geometry(
        cache, mla_mxfp4_row_bytes(kv_lora_rank, rope)
    )
    if out2.stride(-1) != 1:
        raise ValueError("out rows must have dense channels")
    if n == 0:
        return
    _dequant_gather_mla_latent_mxfp4_kernel[(n,)](
        cache,
        slots,
        out2,
        out2.stride(0),
        page_size,
        page_stride,
        row_stride,
        RANK=kv_lora_rank,
        ROPE=rope,
    )
