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

"""Paged OCP MXFP4 KV cache for GQA/MHA attention (w4a16, gfx950).

A layer's K (or V) field is a sequence of kernel pages of ``P`` token rows.
Each page holds its packed E2M1 rows ``[P, heads, head_dim // 2]`` (even
channel in the low nibble) followed by their UE8M0 exponents ``[P, heads,
head_dim // 32]``, one per 32 channels. :func:`mxfp4_cache_planes` views a
field as the two planes; slot ``s`` is row ``s % P`` of page ``s // P``.

The attention prologue's ``triton_mxfp4_gqa_prologue`` quantizes rows into
the planes as it writes them; :func:`mxfp4_store_kv` is the standalone store.
Attention keeps the query in BF16: QK is ``tl.dot_scaled`` of BF16 against
E2M1 keys with their scales, and V is dequantized in registers for a BF16 PV.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.quantization._triton_mxfp4 import _mxfp4_encode_rows

__all__ = [
    "mxfp4_cache_planes",
    "mxfp4_decode",
    "mxfp4_dequant_pages",
    "mxfp4_extend",
    "mxfp4_row_bytes",
    "mxfp4_store_kv",
]

MXFP4_BLOCK = 32


def mxfp4_row_bytes(head_dim: int) -> int:
    """Bytes of one MXFP4 head row: packed E2M1 pairs plus one exponent per block."""
    return head_dim // 2 + head_dim // MXFP4_BLOCK


def mxfp4_cache_planes(
    field: torch.Tensor, page_size: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """View an MXFP4 cache field as its data and scale planes.

    Args:
        field: uint8 ``[..., heads, mxfp4_row_bytes(head_dim)]`` cache field
            whose bytes from its base are consecutive kernel pages.
        page_size: Token rows of one kernel page.
        head_dim: Width of a dequantized head.

    Returns:
        Packed E2M1 rows ``[pages, page_size, heads, head_dim // 2]`` and
        UE8M0 exponents ``[pages, page_size, heads, head_dim // 32]``, both
        uint8 views of ``field`` with one page stride.
    """
    heads, row = field.shape[-2:]
    if (
        field.dtype != torch.uint8
        or row != mxfp4_row_bytes(head_dim)
        or head_dim % MXFP4_BLOCK
        or field.stride(-1) != 1
        or field.stride(-2) != row
    ):
        raise ValueError(
            f"{tuple(field.shape)} {field.dtype} is not a field of dense MXFP4 "
            f"rows of {head_dim}-wide heads"
        )
    page_bytes = page_size * heads * row
    span = 1 + sum((n - 1) * s for n, s in zip(field.shape, field.stride()))
    pages = span // page_bytes
    if pages == 0:
        raise ValueError(f"{tuple(field.shape)} holds no {page_size}-row page")
    data_row, scale_row = head_dim // 2, head_dim // MXFP4_BLOCK
    data = field.as_strided(
        (pages, page_size, heads, data_row),
        (page_bytes, heads * data_row, data_row, 1),
    )
    scales = field.as_strided(
        (pages, page_size, heads, scale_row),
        (page_bytes, heads * scale_row, scale_row, 1),
        field.storage_offset() + page_size * heads * data_row,
    )
    return data, scales


def _check_planes(
    data: torch.Tensor, scales: torch.Tensor, heads: int, head_dim: int
) -> None:
    pages, page_size = data.shape[:2]
    if (
        data.dtype != torch.uint8
        or scales.dtype != torch.uint8
        or data.shape != (pages, page_size, heads, head_dim // 2)
        or scales.shape != (pages, page_size, heads, head_dim // MXFP4_BLOCK)
        or data.stride()[1:] != (heads * head_dim // 2, head_dim // 2, 1)
        or scales.stride()[1:]
        != (heads * head_dim // MXFP4_BLOCK, head_dim // MXFP4_BLOCK, 1)
        or scales.stride(0) != data.stride(0)
    ):
        raise ValueError(
            "MXFP4 planes are dense uint8 pages of one stride (mxfp4_cache_planes)"
        )


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


@triton.jit
def _mxfp4_store_kernel(
    k_ptr,
    v_ptr,
    slots_ptr,
    kd_ptr,
    ks_ptr,
    vd_ptr,
    vs_ptr,
    k_stride,
    v_stride,
    page_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + t).to(tl.int64)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    row = (slot % PAGE) * H + offs_h[:, None]
    page = (slot // PAGE) * page_stride
    od = page + row * (D // 2) + tl.arange(0, D // 2)[None, :]
    os_ = page + row * (D // 32) + tl.arange(0, D // 32)[None, :]
    k = tl.load(k_ptr + t * k_stride + offs_h[:, None] * D + offs_d[None, :])
    packed, exponent = _mxfp4_encode_rows(k.to(tl.float32), H, D)
    tl.store(kd_ptr + od, packed)
    tl.store(ks_ptr + os_, exponent.to(tl.uint8))
    v = tl.load(v_ptr + t * v_stride + offs_h[:, None] * D + offs_d[None, :])
    packed, exponent = _mxfp4_encode_rows(v.to(tl.float32), H, D)
    tl.store(vd_ptr + od, packed)
    tl.store(vs_ptr + os_, exponent.to(tl.uint8))


def mxfp4_store_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    k_planes: tuple[torch.Tensor, torch.Tensor],
    v_planes: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Quantize K/V rows to MXFP4 and store them at their slots.

    Args:
        k: Key rows ``[tokens, heads, head_dim]`` with dense heads.
        v: Value rows of the same geometry.
        slots: Destination slot of each row ``[tokens]``.
        k_planes: Key data and scale planes (:func:`mxfp4_cache_planes`).
        v_planes: Value planes of the same geometry.
    """
    tokens, heads, head_dim = k.shape
    if tokens == 0:
        return
    for planes in (k_planes, v_planes):
        _check_planes(*planes, heads, head_dim)
    if (
        v.shape != k.shape
        or slots.shape != (tokens,)
        or any(x.stride(-1) != 1 or x.stride(-2) != head_dim for x in (k, v))
    ):
        raise ValueError("k, v and slots must be dense rows of one geometry")
    _mxfp4_store_kernel[(tokens,)](
        k,
        v,
        slots,
        k_planes[0],
        k_planes[1],
        v_planes[0],
        v_planes[1],
        k.stride(0),
        v.stride(0),
        k_planes[0].stride(0),
        H=heads,
        D=head_dim,
        PAGE=k_planes[0].shape[1],
    )


# --------------------------------------------------------------------------- #
# tile helpers
# --------------------------------------------------------------------------- #


@triton.jit
def _mxfp4_dequant(packed, exponent, N: tl.constexpr, D: tl.constexpr):
    """uint8 [N, D // 2] E2M1 pairs and [N, D // 32] exponents -> bf16 [N, D],
    by v_cvt_scalef32_pk_bf16_fp4 (gfx950)."""
    scale = (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], (N, D // 32, 16)), (N, D // 2)
    )
    x = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2",
        "=v,v,v",
        [packed.to(tl.int32), scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    lo = (x & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = ((x >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return tl.reshape(tl.join(lo, hi), (N, D))


@triton.jit
def _attend_tile(
    q,
    m_i,
    l_i,
    acc,
    kd_ptr,
    ks_ptr,
    vd_ptr,
    vs_ptr,
    phys,
    keys,
    row_lo,
    row_end,
    sm_scale_log2,
    page_stride,
    kvh,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One online-softmax step over the BLOCK_N keys of pages ``phys``: QK by
    BF16 x E2M1 ``dot_scaled``, PV in BF16 on dequantized values."""
    offs_n = tl.arange(0, BLOCK_N)
    row = ((offs_n % PAGE) * H + kvh)[:, None]
    page = phys[:, None] * page_stride
    data = page + row * (D // 2) + tl.arange(0, D // 2)[None, :]
    scale = page + row * (D // 32) + tl.arange(0, D // 32)[None, :]
    k = tl.load(kd_ptr + data)
    k_scale = tl.load(ks_ptr + scale)
    qk = tl.dot_scaled(q, None, "bf16", tl.trans(k), k_scale, "e2m1")
    qk = qk * sm_scale_log2
    visible = (keys[None, :] < row_end[:, None]) & (keys[None, :] >= row_lo[:, None])
    qk = tl.where(visible, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_safe)
    p = tl.exp2(qk - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    v = _mxfp4_dequant(tl.load(vd_ptr + data), tl.load(vs_ptr + scale), BLOCK_N, D)
    acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
    return m_new, l_i, acc


# --------------------------------------------------------------------------- #
# decode: split-K over pages, merged by the last split to arrive
# --------------------------------------------------------------------------- #


@triton.jit
def _decode_kernel(
    q_ptr,
    kd_ptr,
    ks_ptr,
    vd_ptr,
    vs_ptr,
    page_table,
    seqlens,
    mid_o,
    mid_lse,
    counters,
    out_ptr,
    gate_ptr,
    gate_stride,
    sm_scale_log2,
    pt_stride,
    num_splits,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    QLEN: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    PPI: tl.constexpr,
    HAS_GATE: tl.constexpr,
    GATE_EPS: tl.constexpr,
):
    """Rows are (query position, head of the KV group) of one sequence and
    KV head; each split attends a run of pages and stores its normalized
    partial and log-sum-exp. The split whose ticket arrives last merges them,
    applies the output gate, stores the output and re-arms the counter."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    GROUP: tl.constexpr = HQ // H
    ROWS: tl.constexpr = QLEN * GROUP
    BLOCK_N: tl.constexpr = PAGE * PPI
    rows = tl.arange(0, BLOCK_M)
    row_ok = rows < ROWS
    qpos = rows // GROUP
    head = kvh * GROUP + rows % GROUP
    token = b.to(tl.int64) * QLEN + qpos
    offs_d = tl.arange(0, D)

    seq_len = tl.load(seqlens + b)
    row_end = seq_len - (QLEN - 1 - qpos)
    first_end = tl.maximum(seq_len - (QLEN - 1), 0)
    if WINDOW_LEFT >= 0:
        kv_lo = tl.maximum(first_end - (WINDOW_LEFT + 1), 0)
        row_lo = row_end - (WINDOW_LEFT + 1)
    else:
        kv_lo = 0
        row_lo = row_end * 0
    first_page = kv_lo // PAGE
    end_page = tl.cdiv(seq_len, PAGE)
    per_split = tl.cdiv(end_page - first_page, num_splits)
    p0 = first_page + split * per_split
    p1 = tl.minimum(p0 + per_split, end_page)

    q = tl.load(
        q_ptr + token[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    row_end_split = tl.minimum(row_end, p1 * PAGE)
    for pg in range(p0, p1, PPI):
        pages = pg + offs_n // PAGE
        phys = tl.load(page_table + b * pt_stride + pages, mask=pages < p1, other=0).to(
            tl.int64
        )
        m_i, l_i, acc = _attend_tile(
            q,
            m_i,
            l_i,
            acc,
            kd_ptr,
            ks_ptr,
            vd_ptr,
            vs_ptr,
            phys,
            pg * PAGE + offs_n,
            row_lo,
            row_end_split,
            sm_scale_log2,
            PAGE_STRIDE,
            kvh,
            H,
            D,
            PAGE,
            BLOCK_N,
        )
    has = l_i > 0
    o = tl.where(has[:, None], acc / tl.where(has, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), float("-inf"))
    base = (token * HQ + head) * MAX_SPLITS
    tl.store(
        mid_o + (base + split)[:, None] * D + offs_d[None, :], o, mask=row_ok[:, None]
    )
    tl.store(mid_lse + base + split, lse, mask=row_ok)
    counter = counters + b * H + kvh
    tl.debug_barrier()
    ticket = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    # The barrier orders partial reads after the acquire, however the atomic lowers.
    tl.debug_barrier()
    if ticket == num_splits - 1:
        m_run = lse
        weight = tl.where(has, 1.0, 0.0)
        merged = o
        for s in range(num_splits):
            other = row_ok & (s != split)
            lse_s = tl.load(
                mid_lse + base + s,
                mask=other,
                other=float("-inf"),
                cache_modifier=".cg",
            )
            o_s = tl.load(
                mid_o + (base + s)[:, None] * D + offs_d[None, :],
                mask=other[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            m_new = tl.maximum(m_run, lse_s)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_run - m_safe)
            beta = tl.exp2(lse_s - m_safe)
            merged = merged * alpha[:, None] + o_s * beta[:, None]
            weight = weight * alpha + beta
            m_run = m_new
        merged = merged / tl.where(weight > 0, weight, 1.0)[:, None]
        dtype = out_ptr.dtype.element_ty
        if HAS_GATE:
            # store_rmsnorm_gated: weightless RMSNorm of the rounded output
            # head, times the sigmoid of the gate.
            merged = merged.to(dtype).to(tl.float32)
            gate = tl.load(
                gate_ptr
                + token[:, None] * gate_stride
                + head[:, None] * D
                + offs_d[None, :],
                mask=row_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            variance = tl.sum(merged * merged, axis=1) / D
            merged = (
                merged
                * tl.rsqrt(variance[:, None] + GATE_EPS)
                * (1 / (1 + tl.exp(-gate)))
            )
        tl.store(
            out_ptr + token[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :],
            merged.to(dtype),
            mask=row_ok[:, None],
        )
        tl.atomic_xchg(counter, 0, sem="release", scope="gpu")


# Pages per inner iteration, software pipeline stages, target programs per
# CU and the smallest row block; tuned on MI355X at 16/2 heads, head_dim 64.
_DECODE_PAGES_PER_ITER = 4
_DECODE_STAGES = 2
_DECODE_SPLIT_MULT = 4
_DECODE_MIN_BLOCK_M = 16
_DECODE_MAX_SPLITS = 64

# Split partials and the arrival counters, per (device, geometry); the
# counters start at zero and the last split of each (sequence, KV head)
# re-arms its own, so a graph replays with them as captured.
_workspaces: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _decode_workspace(
    tokens: int, q_heads: int, head_dim: int, max_splits: int, counters: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, tokens, q_heads, head_dim, max_splits, counters)
    if key not in _workspaces:
        rows = tokens * q_heads * max_splits
        _workspaces[key] = (
            torch.empty(rows * head_dim, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.zeros(counters, dtype=torch.int32, device=device),
        )
    return _workspaces[key]


@lru_cache(maxsize=None)
def _compute_units(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _decode_splits(programs: int, page_iters: int, device: torch.device) -> int:
    target = _DECODE_SPLIT_MULT * _compute_units(device)
    return max(1, min(_DECODE_MAX_SPLITS, triton.cdiv(target, programs), page_iters))


def mxfp4_decode(
    q: torch.Tensor,
    k_planes: tuple[torch.Tensor, torch.Tensor],
    v_planes: tuple[torch.Tensor, torch.Tensor],
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    window_left: int,
    qlen: int,
    softmax_scale: float,
    gate: torch.Tensor | None,
    gate_eps: float,
) -> torch.Tensor:
    """Decode attention over an MXFP4 paged cache, in one launch.

    Args:
        q: BF16 queries ``[batch * qlen, q_heads, head_dim]``; each sequence's
            ``qlen`` rows are its last positions.
        k_planes: Key data and scale planes (:func:`mxfp4_cache_planes`).
        v_planes: Value planes of the same geometry.
        page_table: Kernel page ids ``[batch, max_pages]``.
        seq_lens: Cached length of each sequence, its new rows included.
        window_left: Keys a row attends before itself; -1 for all.
        qlen: Query rows per sequence.
        softmax_scale: Logit scale.
        gate: Sigmoid output gate logits, rows of ``q_heads * head_dim`` with
            unit channel stride, or ``None``.
        gate_eps: Epsilon of the gate's weightless RMSNorm.

    Returns:
        The output ``[batch * qlen, q_heads, head_dim]`` in ``q``'s dtype,
        gated when ``gate`` is given.
    """
    tokens, q_heads, head_dim = q.shape
    k_data = k_planes[0]
    page_size, kv_heads = k_data.shape[1], k_data.shape[2]
    for planes in (k_planes, v_planes):
        _check_planes(*planes, kv_heads, head_dim)
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("mxfp4_decode takes dense bf16 queries")
    if tokens % qlen or q_heads % kv_heads:
        raise ValueError(f"{tokens} rows of {q_heads} heads do not split by sequence")
    if gate is not None and (
        gate.shape[0] != tokens
        or gate.shape[1:].numel() != q_heads * head_dim
        or gate.stride(-1) != 1
    ):
        raise ValueError(f"gate {tuple(gate.shape)} is not a row per query token")
    batch = tokens // qlen
    out = torch.empty_like(q)
    if tokens == 0:
        return out
    max_pages = page_table.shape[1]
    if window_left >= 0:
        max_pages = min(max_pages, triton.cdiv(window_left + 1, page_size) + 1)
    num_splits = _decode_splits(
        batch * kv_heads, triton.cdiv(max_pages, _DECODE_PAGES_PER_ITER), q.device
    )
    max_splits = triton.next_power_of_2(num_splits)
    mid_o, mid_lse, counters = _decode_workspace(
        tokens, q_heads, head_dim, max_splits, batch * kv_heads, q.device
    )
    rows = qlen * (q_heads // kv_heads)
    _decode_kernel[(batch, kv_heads, num_splits)](
        q,
        k_planes[0],
        k_planes[1],
        v_planes[0],
        v_planes[1],
        page_table,
        seq_lens,
        mid_o,
        mid_lse,
        counters,
        out,
        q if gate is None else gate,
        0 if gate is None else gate.stride(0),
        softmax_scale * 1.4426950408889634,
        page_table.stride(0),
        num_splits,
        HQ=q_heads,
        H=kv_heads,
        D=head_dim,
        PAGE=page_size,
        PAGE_STRIDE=k_data.stride(0),
        QLEN=qlen,
        WINDOW_LEFT=window_left,
        BLOCK_M=max(_DECODE_MIN_BLOCK_M, triton.next_power_of_2(rows)),
        MAX_SPLITS=max_splits,
        PPI=_DECODE_PAGES_PER_ITER,
        HAS_GATE=gate is not None,
        GATE_EPS=gate_eps,
        num_warps=4,
        num_stages=_DECODE_STAGES,
        waves_per_eu=2,
    )
    return out


# --------------------------------------------------------------------------- #
# extend: each sequence's new rows attend the cached prefix and themselves
# --------------------------------------------------------------------------- #


@triton.jit
def _extend_kernel(
    q_ptr,
    kd_ptr,
    ks_ptr,
    vd_ptr,
    vs_ptr,
    page_table,
    seqlens,
    cu_q,
    out,
    sm_scale_log2,
    pt_stride,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    PPI: tl.constexpr,
):
    """Rows are (token, head of the KV group)."""
    qblk = tl.program_id(0)
    kvh = tl.program_id(1)
    b = tl.program_id(2)
    GROUP: tl.constexpr = HQ // H
    BLOCK_M: tl.constexpr = BLOCK_Q * GROUP
    BLOCK_N: tl.constexpr = PAGE * PPI
    q_start = tl.load(cu_q + b)
    q_len = tl.load(cu_q + b + 1) - q_start
    if qblk * BLOCK_Q >= q_len:
        return
    prefix = tl.load(seqlens + b) - q_len
    rows = tl.arange(0, BLOCK_M)
    qi = qblk * BLOCK_Q + rows // GROUP
    row_ok = qi < q_len
    head = kvh * GROUP + rows % GROUP
    token = (q_start + qi).to(tl.int64)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + token[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    row_end = prefix + qi + 1
    last_end = prefix + tl.minimum(qblk * BLOCK_Q + BLOCK_Q, q_len)
    if WINDOW_LEFT >= 0:
        row_lo = row_end - (WINDOW_LEFT + 1)
        first = tl.maximum(prefix + qblk * BLOCK_Q + 1 - (WINDOW_LEFT + 1), 0)
    else:
        row_lo = row_end * 0
        first = 0
    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    end_page = tl.cdiv(last_end, PAGE)
    for pg in range(first // PAGE, end_page, PPI):
        pages = pg + offs_n // PAGE
        phys = tl.load(
            page_table + b * pt_stride + pages, mask=pages < end_page, other=0
        ).to(tl.int64)
        m_i, l_i, acc = _attend_tile(
            q,
            m_i,
            l_i,
            acc,
            kd_ptr,
            ks_ptr,
            vd_ptr,
            vs_ptr,
            phys,
            pg * PAGE + offs_n,
            row_lo,
            row_end,
            sm_scale_log2,
            PAGE_STRIDE,
            kvh,
            H,
            D,
            PAGE,
            BLOCK_N,
        )
    o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(
        out + token[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :],
        o.to(out.dtype.element_ty),
        mask=row_ok[:, None],
    )


def mxfp4_extend(
    q: torch.Tensor,
    k_planes: tuple[torch.Tensor, torch.Tensor],
    v_planes: tuple[torch.Tensor, torch.Tensor],
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    *,
    window_left: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Causal attention of each sequence's new rows over an MXFP4 paged
    cache that already holds them.

    Args:
        q: BF16 queries ``[tokens, q_heads, head_dim]``.
        k_planes: Key data and scale planes (:func:`mxfp4_cache_planes`).
        v_planes: Value planes of the same geometry.
        page_table: Kernel page ids ``[batch, max_pages]``.
        seq_lens: Cached length of each sequence, its new rows included.
        cu_seqlens_q: Cumulative new rows ``[batch + 1]``.
        max_seqlen_q: Most new rows of one sequence.
        window_left: Keys a row attends before itself; -1 for all.
        softmax_scale: Logit scale.

    Returns:
        The output ``[tokens, q_heads, head_dim]`` in ``q``'s dtype.
    """
    tokens, q_heads, head_dim = q.shape
    k_data = k_planes[0]
    page_size, kv_heads = k_data.shape[1], k_data.shape[2]
    for planes in (k_planes, v_planes):
        _check_planes(*planes, kv_heads, head_dim)
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("mxfp4_extend takes dense bf16 queries")
    out = torch.empty_like(q)
    batch = seq_lens.shape[0]
    if tokens == 0 or batch == 0:
        return out
    block_q = 16
    _extend_kernel[(triton.cdiv(max_seqlen_q, block_q), kv_heads, batch)](
        q,
        k_planes[0],
        k_planes[1],
        v_planes[0],
        v_planes[1],
        page_table,
        seq_lens,
        cu_seqlens_q,
        out,
        softmax_scale * 1.4426950408889634,
        page_table.stride(0),
        HQ=q_heads,
        H=kv_heads,
        D=head_dim,
        PAGE=page_size,
        PAGE_STRIDE=k_data.stride(0),
        WINDOW_LEFT=window_left,
        BLOCK_Q=block_q,
        PPI=2,
        num_warps=4,
        num_stages=1,
        waves_per_eu=2,
    )
    return out


# --------------------------------------------------------------------------- #
# page dequantization, for BF16 kernels over a long cached prefix
# --------------------------------------------------------------------------- #


@triton.jit
def _dequant_pages_kernel(
    kd_ptr,
    ks_ptr,
    vd_ptr,
    vs_ptr,
    ids_ptr,
    k_out,
    v_out,
    page_stride,
    ROWS: tl.constexpr,
    D: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    page = tl.load(ids_ptr + i).to(tl.int64) * page_stride
    offs_r = tl.arange(0, ROWS)
    data = page + offs_r[:, None] * (D // 2) + tl.arange(0, D // 2)[None, :]
    scale = page + offs_r[:, None] * (D // 32) + tl.arange(0, D // 32)[None, :]
    dst = i * ROWS * D + offs_r[:, None] * D + tl.arange(0, D)[None, :]
    k = _mxfp4_dequant(tl.load(kd_ptr + data), tl.load(ks_ptr + scale), ROWS, D)
    tl.store(k_out + dst, k)
    v = _mxfp4_dequant(tl.load(vd_ptr + data), tl.load(vs_ptr + scale), ROWS, D)
    tl.store(v_out + dst, v)


def mxfp4_dequant_pages(
    k_planes: tuple[torch.Tensor, torch.Tensor],
    v_planes: tuple[torch.Tensor, torch.Tensor],
    page_ids: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    """Dequantize MXFP4 pages to BF16.

    Args:
        k_planes: Key data and scale planes (:func:`mxfp4_cache_planes`).
        v_planes: Value planes of the same geometry.
        page_ids: Pages to dequantize ``[n]``.
        k_out: Dense BF16 ``[n, page_size, heads, head_dim]``; receives page
            ``page_ids[i]`` at ``i``.
        v_out: Same, for the values.
    """
    n, page_size, heads, head_dim = k_out.shape
    for planes in (k_planes, v_planes):
        _check_planes(*planes, heads, head_dim)
    if (
        k_planes[0].shape[1] != page_size
        or v_out.shape != k_out.shape
        or not (k_out.is_contiguous() and v_out.is_contiguous())
        or k_out.dtype != torch.bfloat16
        or v_out.dtype != torch.bfloat16
        or page_ids.shape != (n,)
    ):
        raise ValueError("outputs must be dense bf16 pages, one per page id")
    if n == 0:
        return
    _dequant_pages_kernel[(n,)](
        k_planes[0],
        k_planes[1],
        v_planes[0],
        v_planes[1],
        page_ids,
        k_out,
        v_out,
        k_planes[0].stride(0),
        ROWS=page_size * heads,
        D=head_dim,
        num_warps=4,
    )
