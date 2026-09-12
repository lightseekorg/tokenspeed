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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# (decoded dimensions, quantization group, value bytes, total row bytes)
_LAYOUTS = {
    "swa": (512, 32, 512, 528),
    "global": (512, 16, 256, 288),
    "index": (128, 32, 64, 68),
}
_CAPABILITY = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))
_FLOAT_SIGNATURES = frozenset(
    format_signature(x=dense_tensor_format(dtype))
    for dtype in (torch.bfloat16, torch.float32)
)
_BYTE_SIGNATURES = frozenset({format_signature(x=dense_tensor_format(torch.uint8))})
_QUERY_SIGNATURES = frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))})


@triton.jit
def _e2m1_encode(x):
    a = tl.minimum(tl.abs(x), 6.0)
    # Nearest, ties to even *code*, including signed zero. V4's codec is not RNE.
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return (code | ((x.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)


@triton.jit
def _e2m1_decode(code):
    a = code & 7
    value = tl.where(
        a < 4,
        a.to(tl.float32) * 0.5,
        tl.where(a == 4, 2.0, tl.where(a == 5, 3.0, tl.where(a == 6, 4.0, 6.0))),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _planar_offset(byte, CR: tl.constexpr, CB: tl.constexpr, ROW_BYTES: tl.constexpr):
    # Byte offsets are page-planar, while the external field retains its LCM
    # [pages, rows, row_bytes] shape. Honor even noncontiguous portable views.
    if CR == ROW_BYTES * CB:
        return byte * CB
    return (byte // ROW_BYTES) * CR + (byte % ROW_BYTES) * CB


@triton.jit
def _pack_kernel(
    X,
    C,
    Slots,
    XS0,
    XS1,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    SS,
    PAGE_ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    FORMAT: tl.constexpr,
    SCATTER: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row * SS).to(tl.int64) if SCATTER else row.to(tl.int64)
    if slot >= 0 and slot < CAPACITY:
        d = tl.arange(0, D)
        x = tl.load(X + row * XS0 + d * XS1).to(tl.float32).reshape(D // GROUP, GROUP)
        amax = tl.max(tl.abs(x), axis=1)
        if FORMAT == "global":
            scale = (
                tl.div_rn(tl.maximum(amax, 6 * 2.0**-9), 6.0)
                .to(tl.float8e4nv)
                .to(tl.float32)
            )
            scale_byte = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            if FORMAT == "swa":
                scaled_max = tl.maximum(amax, 1.0e-4) * (1.0 / 448.0)
            else:
                scaled_max = tl.maximum(amax, 6 * 2.0**-126) * (1.0 / 6.0)
            # Match reference IEEE-754 ceil(log2), not an approximate log2.
            bits = scaled_max.to(tl.int32, bitcast=True)
            exponent = (
                ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            )
            scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
            scale_byte = (exponent + 127).to(tl.uint8)
        y = tl.div_rn(x, scale[:, None]).reshape(D)
        base = C + (slot // PAGE_ROWS) * CP
        data_byte = (slot % PAGE_ROWS) * VALUES
        scale_byte_offset = PAGE_ROWS * VALUES + (slot % PAGE_ROWS) * (D // GROUP)
        ROW_BYTES: tl.constexpr = VALUES + D // GROUP
        if FORMAT == "swa":
            encoded = (
                tl.clamp(y, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            )
            tl.store(base + _planar_offset(data_byte + d, CR, CB, ROW_BYTES), encoded)
        else:
            codes = _e2m1_encode(y).reshape(D // 2, 2)
            lo, hi = tl.split(codes)
            tl.store(
                base
                + _planar_offset(data_byte + tl.arange(0, D // 2), CR, CB, ROW_BYTES),
                lo | (hi << 4),
            )
        tl.store(
            base
            + _planar_offset(
                scale_byte_offset + tl.arange(0, D // GROUP), CR, CB, ROW_BYTES
            ),
            scale_byte,
        )


@triton.jit
def _gather_kernel(
    C,
    Slots,
    O,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    OS0,
    OS1,
    PAGE_ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    FORMAT: tl.constexpr,
    GATHER: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row).to(tl.int64) if GATHER else row.to(tl.int64)
    valid = (slot >= 0) & (slot < CAPACITY)
    slot = tl.where(valid, slot, 0)
    d = tl.arange(0, D)
    base = C + (slot // PAGE_ROWS) * CP
    data_byte = (slot % PAGE_ROWS) * VALUES + tl.where(VALUES == D, d, d // 2)
    scale_byte_offset = (
        PAGE_ROWS * VALUES + (slot % PAGE_ROWS) * (D // GROUP) + d // GROUP
    )
    ROW_BYTES: tl.constexpr = VALUES + D // GROUP
    byte = tl.load(base + _planar_offset(data_byte, CR, CB, ROW_BYTES), valid, other=0)
    if FORMAT == "swa":
        value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        value = _e2m1_decode((byte >> ((d % 2) * 4)) & 15)
    scale_byte = tl.load(
        base + _planar_offset(scale_byte_offset, CR, CB, ROW_BYTES), valid, other=0
    )
    if FORMAT == "global":
        scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        # E8M0 byte 0 represents 2**-127 (a float32 subnormal).
        scale = tl.where(
            scale_byte == 0,
            2.0**-127,
            (scale_byte.to(tl.int32) << 23).to(tl.float32, bitcast=True),
        )
    tl.store(O + row * OS0 + d * OS1, tl.where(valid, value * scale, 0.0))


def _layout(cache_format):
    if cache_format not in _LAYOUTS:
        raise ValueError("cache_format must be 'swa', 'global', or 'index'")
    return _LAYOUTS[cache_format]


def _same_device(x, tensors):
    if not x.is_cuda or any(t.device != x.device for t in tensors if t is not None):
        raise ValueError("all tensors must share a CUDA/ROCm device")


def _integers(x, shape, name):
    if x.shape != shape or x.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32/int64 with shape {shape}")


def _cache(cache, cache_format):
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1:] != (64, _layout(cache_format)[3])
    ):
        raise ValueError(
            f"{cache_format} cache must be uint8 [pages, 64, {_layout(cache_format)[3]}]"
        )


def _output(out, shape, dtype, device):
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must have the specified shape, dtype, device and be contiguous"
        )
    return out


def _pack(rows, cache, slots, cache_format):
    dim, group, values, _ = _layout(cache_format)
    if (
        rows.ndim != 2
        or rows.shape[1] != dim
        or rows.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError(f"rows must be BF16/FP32 [rows, {dim}]")
    _same_device(rows, (cache, slots))
    if slots is not None:
        _integers(slots, (rows.shape[0],), "slots")
    if rows.shape[0]:
        _pack_kernel[(rows.shape[0],)](
            rows,
            cache,
            slots,
            *rows.stride(),
            *cache.stride(),
            slots.stride(0) if slots is not None else 0,
            PAGE_ROWS=cache.shape[1],
            CAPACITY=cache.shape[0] * cache.shape[1],
            D=dim,
            GROUP=group,
            VALUES=values,
            FORMAT=cache_format,
            SCATTER=slots is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )


def _gather(cache, slots, cache_format, out):
    dim, group, values, _ = _layout(cache_format)
    _same_device(cache, (slots, out))
    shape = (*slots.shape, dim) if slots is not None else (cache.shape[0], dim)
    dtype = torch.bfloat16 if out is None else out.dtype
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("dequantized output must be BF16 or FP32")
    out = _output(out, shape, dtype, cache.device)
    if slots is not None:
        _integers(slots, slots.shape, "slots")
        slots = slots.contiguous()
    rows = out.numel() // dim
    if rows:
        _gather_kernel[(rows,)](
            cache,
            slots,
            out,
            *cache.stride(),
            dim,
            1,
            PAGE_ROWS=cache.shape[1],
            CAPACITY=cache.shape[0] * cache.shape[1],
            D=dim,
            GROUP=group,
            VALUES=values,
            FORMAT=cache_format,
            GATHER=slots is not None,
            num_warps=4,
        )
    return out


@register_kernel(
    "attention",
    "dsv41_cache_pack",
    name="triton_dsv41_cache_pack",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_FLOAT_SIGNATURES,
    priority=Priority.PORTABLE,
)
def cache_pack(rows, cache_format, out):
    width = _layout(cache_format)[3]
    out = _output(out, (rows.shape[0], width), torch.uint8, rows.device)
    _pack(rows, out.unsqueeze(1), None, cache_format)
    return out


@register_kernel(
    "attention",
    "dsv41_cache_unpack",
    name="triton_dsv41_cache_unpack",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_BYTE_SIGNATURES,
    priority=Priority.PORTABLE,
)
def cache_unpack(packed, cache_format, out):
    if (
        packed.ndim != 2
        or packed.dtype != torch.uint8
        or packed.shape[1] != _layout(cache_format)[3]
    ):
        raise ValueError("packed must be a uint8 row matrix of the specified format")
    return _gather(packed.unsqueeze(1), None, cache_format, out)


@register_kernel(
    "attention",
    "dsv41_cache_scatter",
    name="triton_dsv41_cache_scatter",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_FLOAT_SIGNATURES,
    priority=Priority.PORTABLE,
)
def cache_scatter(rows, cache, slots, cache_format):
    _cache(cache, cache_format)
    _pack(rows, cache, slots, cache_format)


@register_kernel(
    "attention",
    "dsv41_cache_gather",
    name="triton_dsv41_cache_gather",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_BYTE_SIGNATURES,
    priority=Priority.PORTABLE,
)
def cache_gather(cache, slots, cache_format, out):
    _cache(cache, cache_format)
    return _gather(cache, slots, cache_format, out)


@triton.jit
def _compressor_tail_scatter_kernel(
    Content,
    Scores,
    Tail,
    Slots,
    CS0,
    CS1,
    SS0,
    SS1,
    TP,
    TR,
    TK,
    TD,
    LS,
    CAPACITY: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row * LS).to(tl.int64)
    if slot >= 0 and slot < CAPACITY:
        d = tl.arange(0, 512)
        base = Tail + slot // 2 * TP + slot % 2 * TR + d * TD
        tl.store(base, tl.load(Content + row * CS0 + d * CS1))
        tl.store(base + TK, tl.load(Scores + row * SS0 + d * SS1))


@register_kernel(
    "attention",
    "dsv41_compressor_tail_scatter",
    name="triton_dsv41_compressor_tail_scatter",
    solution="triton",
    capability=_CAPABILITY,
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.float32))}),
    priority=Priority.PORTABLE,
)
def compressor_tail_scatter(content, scores, tail, slots):
    if (
        content.ndim != 2
        or content.shape[1] != 512
        or scores.shape != content.shape
        or content.dtype != torch.float32
        or scores.dtype != torch.float32
        or tail.ndim != 4
        or tail.shape[1:] != (2, 2, 512)
        or tail.dtype != torch.float32
    ):
        raise ValueError(
            "require FP32 content/scores [T, 512] and tail [pages, 2, 2, 512]"
        )
    _same_device(content, (scores, tail, slots))
    _integers(slots, (content.shape[0],), "slots")
    if content.shape[0]:
        _compressor_tail_scatter_kernel[(content.shape[0],)](
            content,
            scores,
            tail,
            slots,
            *content.stride(),
            *scores.stride(),
            *tail.stride(),
            slots.stride(0),
            CAPACITY=tail.shape[0] * 2,
            num_warps=4,
        )


@register_kernel(
    "attention",
    "dsv41_index_q_quantize",
    name="triton_dsv41_index_q_quantize",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_QUERY_SIGNATURES,
    priority=Priority.PORTABLE,
)
def index_q_quantize(q, out):
    if q.ndim != 3 or q.shape[-1] != 128 or q.dtype != torch.bfloat16:
        raise ValueError("index q must be BF16 [tokens, heads, 128]")
    out = _output(out, q.shape, q.dtype, q.device)
    packed = cache_pack(q.reshape(-1, 128), "index", None)
    cache_unpack(packed, "index", out.view(-1, 128))
    return out


@register_kernel(
    "attention",
    "dsv41_selected_attention",
    name="triton_dsv41_selected_attention",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_QUERY_SIGNATURES,
    priority=Priority.PORTABLE,
)
def selected_attention(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    query_chunk_size,
    schedule,
    prefill_kv,
    prefill_indices,
):
    attn_sink = attn_sink[: q.shape[1]]
    from tokenspeed_kernel.ops.attention import dsv4_prefill

    if prefill_kv is not None:
        if prefill_indices is None:
            raise ValueError("prefill_kv requires prefill_indices")
        out = _output(out, q.shape, q.dtype, q.device)
        lengths = torch.full(
            (q.shape[0],), prefill_indices.shape[-1], dtype=torch.int32, device=q.device
        )
        dsv4_prefill(
            q=q.contiguous(),
            kv=prefill_kv,
            indices=prefill_indices,
            lens=lengths,
            attn_sink=attn_sink.contiguous(),
            softmax_scale=softmax_scale,
            out=out,
            override=None,
            solution="triton",
        )
        return out
    if prefill_indices is not None:
        raise ValueError("prefill_indices requires prefill_kv")
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16 or q.shape[1] < 1:
        raise ValueError("q must be BF16 [tokens, heads, 512]")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    _same_device(
        q,
        (
            swa_cache,
            swa_slots,
            swa_lens,
            global_cache,
            global_slots,
            global_lens,
            attn_sink,
            out,
        ),
    )
    if attn_sink.shape != (q.shape[1],) or attn_sink.dtype != torch.float32:
        raise ValueError("attn_sink must be FP32 [heads]")
    segments = [(swa_cache, swa_slots, swa_lens, "swa")]
    if global_cache is None:
        if global_slots is not None or global_lens is not None:
            raise ValueError(
                "absent global cache requires None global slots and lengths"
            )
    else:
        if global_slots is None or global_lens is None:
            raise ValueError("global cache requires global slots and lengths")
        segments.append((global_cache, global_slots, global_lens, "global"))
    for cache, slots, lens, fmt in segments:
        _cache(cache, fmt)
        if slots.ndim != 2 or slots.shape[0] != q.shape[0]:
            raise ValueError("slots must have one row per query")
        _integers(slots, slots.shape, "slots")
        _integers(lens, (q.shape[0],), "lens")
    out = _output(out, q.shape, q.dtype, q.device)
    width = sum(slots.shape[1] for _, slots, _, _ in segments)
    if width == 0:
        return out.zero_()
    # ponytail: bounded gather costs 640 KiB/query at width 640; replace with a
    # direct two-reader attention kernel when memory bandwidth/launches dominate.
    for start in range(0, q.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, q.shape[0])
        kv_parts, valid_parts = [], []
        for cache, slots, lens, fmt in segments:
            slots = slots[start:end]
            valid = (
                (torch.arange(slots.shape[1], device=q.device) < lens[start:end, None])
                & (slots >= 0)
                & (slots < cache.shape[0] * 64)
            )
            masked_slots = slots.masked_fill(~valid, -1)
            kv_parts.append(cache_gather(cache, masked_slots, fmt, None))
            valid_parts.append(valid)
        kv = torch.cat(kv_parts, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        indices = torch.arange(
            (end - start) * width, dtype=torch.int32, device=q.device
        ).view(end - start, width)
        indices = indices.masked_fill(~valid, -1)
        lens = torch.full((end - start,), width, dtype=torch.int32, device=q.device)
        dsv4_prefill(
            q=q[start:end].contiguous(),
            kv=kv,
            indices=indices,
            lens=lens,
            attn_sink=attn_sink.contiguous(),
            softmax_scale=softmax_scale,
            out=out[start:end],
            override=None,
            solution="triton",
        )
    return out


def _score(q, weights, cache, slots, process_group):
    q, weights, shards = _index_gather_heads(q, weights, process_group)
    keys = cache_gather(cache, slots, "index", None)
    # Reference einsum materializes BF16 dots; weighting and each shard's head
    # sum round to weights' dtype, as in the fused scan below.
    dots = torch.bmm(q, keys.transpose(1, 2))
    products = dots.relu_() * weights.unsqueeze(-1)
    scores = torch.zeros(slots.shape, dtype=torch.float32, device=q.device)
    for partial in products.chunk(shards, dim=1):
        scores.add_(partial.sum(dim=1).float())
    scores = scores.to(weights.dtype)
    return scores.masked_fill((slots < 0) | (slots >= cache.shape[0] * 64), -torch.inf)


def _index_inputs(q, weights, cache):
    if q.ndim != 3 or q.shape[-1] != 128 or q.dtype != torch.bfloat16 or q.shape[1] < 1:
        raise ValueError("index_q must be BF16 [tokens, heads, 128]")
    if weights.shape != q.shape[:2] or weights.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("weights must be BF16/FP32 [tokens, heads], already scaled")
    _cache(cache, "index")
    _same_device(q, (weights, cache))


@register_kernel(
    "attention",
    "dsv41_index_score",
    name="triton_dsv41_index_score",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_QUERY_SIGNATURES,
    priority=Priority.PORTABLE,
)
def index_score(index_q, weights, index_cache, physical_slots, process_group, out):
    _index_inputs(index_q, weights, index_cache)
    if physical_slots.ndim != 2 or physical_slots.shape[0] != index_q.shape[0]:
        raise ValueError("physical_slots must have one row per query")
    _integers(physical_slots, physical_slots.shape, "physical_slots")
    _same_device(index_q, (physical_slots, out))
    out = _output(out, physical_slots.shape, weights.dtype, index_q.device)
    out.copy_(
        _score(
            index_q_quantize(index_q, None),
            weights,
            index_cache,
            physical_slots,
            process_group,
        )
    )
    return out


@triton.jit
def _index_pack_scores(scores, ids):
    bits = scores.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    packed = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids.to(tl.uint32)).to(
        tl.uint64
    )
    return tl.where(scores > -float("inf"), packed, 0)


@triton.jit
def _index_merge(acc, scores, ids, N: tl.constexpr, K: tl.constexpr):
    # Two opposite-order lists form a bitonic sequence. The pairwise maxima
    # contain its exact upper half, not an elementwise approximation to TopK.
    offsets = tl.arange(0, K)
    packed = _index_pack_scores(scores, ids)
    padded = tl.where(offsets < N, tl.gather(packed, offsets % N, 0), 0)
    new = tl.sort(padded, descending=True)
    return tl.sort(tl.maximum(tl.flip(acc, 0), new), descending=True)


@triton.jit
def _index_unpack(packed):
    ordered = (packed >> 32).to(tl.uint32)
    bits = ordered ^ tl.where((ordered & 0x80000000) != 0, 0x80000000, 0xFFFFFFFF)
    scores = tl.where(packed != 0, bits.to(tl.float32, bitcast=True), -float("inf"))
    ids = tl.where(packed != 0, (0xFFFFFFFF - (packed & 0xFFFFFFFF)).to(tl.int64), -1)
    return scores, ids


@triton.jit
def _index_scan_kernel(
    Q,
    W,
    Cache,
    Table,
    Visible,
    Candidates,
    RowScores,
    RowIds,
    BlockScores,
    BlockIds,
    QS0,
    QS1,
    QS2,
    WS0,
    WS1,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    TS0,
    TS1,
    VS,
    CS0,
    CS1,
    PAGES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    CANDIDATES: tl.constexpr,
    HEADS: tl.constexpr,
    SHARD_HEADS: tl.constexpr,
    SHARDS: tl.constexpr,
    H: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
    STEP: tl.constexpr,
    ROW_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAKE_BLOCKS: tl.constexpr,
):
    query, part = tl.program_id(0), tl.program_id(1)
    visible = tl.minimum(tl.maximum(tl.load(Visible + query * VS), 0), TABLE_WIDTH * 64)
    if CANDIDATES >= 0:
        width = tl.where(visible > 0, CANDIDATES * 8, 0)
    else:
        width = visible
    # All partitions have stable storage and launches; their bounds are device
    # values and are recomputed on every graph replay. Never scan unused pages.
    span = tl.cdiv(width, PARTS * 8) * 8
    begin = part * span
    end = tl.minimum(begin + span, width)
    row_acc = tl.full((ROW_K,), 0, tl.uint64)
    if MAKE_BLOCKS:
        block_acc = tl.full((BLOCK_K,), 0, tl.uint64)
    h = tl.arange(0, H)
    d = tl.arange(0, 128)
    q = tl.load(
        Q + query * QS0 + h[:, None] * QS1 + d[None, :] * QS2,
        h[:, None] < HEADS,
        other=0,
    )
    weights = tl.load(W + query * WS0 + h * WS1, h < HEADS, other=0)
    for start in range(begin, end, STEP):
        col = start + tl.arange(0, B)
        if CANDIDATES >= 0:
            block = tl.load(
                Candidates + query * CS0 + (col // 8) * CS1,
                col < tl.minimum(start + STEP, end),
                other=-1,
            ).to(tl.int64)
            logical = tl.where(block >= 0, block * 8 + col % 8, -1)
        else:
            logical = col.to(tl.int64)
        valid = (
            (col < tl.minimum(start + STEP, end)) & (logical >= 0) & (logical < visible)
        )
        page = tl.load(Table + query * TS0 + (logical // 64) * TS1, valid, other=-1).to(
            tl.int64
        )
        valid = valid & (page >= 0) & (page < PAGES)
        base = Cache + page * CP
        data_byte = (logical[None, :] % 64) * 64 + d[:, None] // 2
        byte = tl.load(
            base[None, :] + _planar_offset(data_byte, CR, CB, 68),
            valid[None, :],
            other=0,
        )
        value = _e2m1_decode((byte >> ((d[:, None] % 2) * 4)) & 15)
        scale_byte = tl.load(
            base[None, :]
            + _planar_offset(
                64 * 64 + (logical[None, :] % 64) * 4 + d[:, None] // 32, CR, CB, 68
            ),
            valid[None, :],
            other=0,
        )
        scale = tl.where(
            scale_byte == 0,
            2.0**-127,
            (scale_byte.to(tl.int32) << 23).to(tl.float32, bitcast=True),
        )
        key = (value * scale).to(tl.bfloat16)
        dots = tl.dot(q, key).to(tl.bfloat16).to(tl.float32)
        products = (
            (tl.maximum(dots, 0.0) * weights[:, None].to(tl.float32))
            .to(W.dtype.element_ty)
            .to(tl.float32)
        )
        # Keep each TP shard's rounded head sum, then use a defined rank-order
        # FP32 sum. NCCL's topology/protocol-dependent order is not bitwise fixed.
        scores = tl.full((B,), 0.0, tl.float32)
        for rank in tl.static_range(SHARDS):
            partial = tl.sum(
                tl.where(
                    (h[:, None] >= rank * SHARD_HEADS)
                    & (h[:, None] < (rank + 1) * SHARD_HEADS),
                    products,
                    0.0,
                ),
                0,
            )
            scores = scores + partial.to(W.dtype.element_ty).to(tl.float32)
        scores = scores.to(W.dtype.element_ty).to(tl.float32)
        scores = tl.where(valid, scores, -float("inf"))
        row_acc = _index_merge(row_acc, scores, logical, B, ROW_K)
        if MAKE_BLOCKS:
            block_scores = tl.max(scores.reshape(B // 8, 8), 1)
            block_ids = (start // 8 + tl.arange(0, B // 8)).to(tl.int64)
            block_scores = tl.where(
                (block_ids == (visible - 1) // 8) & (block_scores > -float("inf")),
                float("inf"),
                block_scores,
            )
            block_acc = _index_merge(
                block_acc, block_scores, block_ids, B // 8, BLOCK_K
            )
    offset = (query * PARTS + part) * ROW_K + tl.arange(0, ROW_K)
    scores, ids = _index_unpack(row_acc)
    tl.store(RowScores + offset, scores)
    tl.store(RowIds + offset, ids)
    if MAKE_BLOCKS:
        offset = (query * PARTS + part) * BLOCK_K + tl.arange(0, BLOCK_K)
        scores, ids = _index_unpack(block_acc)
        tl.store(BlockScores + offset, scores)
        tl.store(BlockIds + offset, ids)


def _index_gather_heads(q, weights, process_group):
    if process_group is None:
        return q, weights, 1
    shards = torch.distributed.get_world_size(process_group)
    # One small collective per query tile instead of one per history tile.
    # Promoting BF16 Q to FP32 when weights are FP32 is lossless.
    packed = torch.cat((q.to(weights.dtype), weights.unsqueeze(-1)), dim=-1)
    gathered = torch.empty(
        (shards * q.shape[0], q.shape[1], 129), dtype=weights.dtype, device=q.device
    )
    torch.distributed.all_gather_into_tensor(gathered, packed, group=process_group)
    gathered = gathered.view(shards, *packed.shape).permute(1, 0, 2, 3).flatten(1, 2)
    return gathered[..., :128].to(torch.bfloat16), gathered[..., 128], shards


def _index_finish_parts(scores, ids, k, output, lengths):
    values, positions = scores.topk(k, dim=1, sorted=False)
    _finish_topk(values, ids.gather(1, positions), output, lengths)


def _finish_topk(scores, ids, output, lengths):
    valid = scores > -torch.inf
    ids = ids.masked_fill(~valid, torch.iinfo(torch.int64).max).sort(dim=1).values
    ids = ids.masked_fill(ids == torch.iinfo(torch.int64).max, -1)
    output.fill_(-1)
    output[:, : ids.shape[1]].copy_(ids)
    lengths.copy_(valid.sum(dim=1))


def _index_topk_outputs(
    index_q,
    weights,
    index_cache,
    page_table,
    visible_lens,
    candidate_blocks,
    topk,
    candidate_topk,
    candidate_block_size,
    query_chunk_size,
    score_chunk_size,
    process_group,
    out,
):
    """Validate the shared selection contract and obtain caller-owned outputs."""
    _index_inputs(index_q, weights, index_cache)
    tokens = index_q.shape[0]
    if page_table.ndim != 2 or page_table.shape[0] != tokens:
        raise ValueError(
            "page_table must be [tokens, logical pages], one request table per query"
        )
    _integers(page_table, page_table.shape, "page_table")
    _integers(visible_lens, (tokens,), "visible_lens")
    _same_device(index_q, (page_table, visible_lens, candidate_blocks))
    if (
        not 1 <= topk <= 512
        or not 0 <= candidate_topk <= 2048
        or candidate_block_size != 8
        or query_chunk_size < 1
        or score_chunk_size < 8
        or score_chunk_size % 8
    ):
        raise ValueError(
            "require 1 <= topk <= 512, 0 <= candidate_topk <= 2048, block_size = 8, positive query chunk and score chunk divisible by 8"
        )
    if candidate_blocks is not None:
        if (
            candidate_blocks.ndim != 2
            or candidate_blocks.shape[0] != tokens
            or candidate_blocks.shape[1] > 2048
        ):
            raise ValueError(
                "candidate_blocks must be [tokens, blocks] with at most 2048 blocks"
            )
        _integers(candidate_blocks, candidate_blocks.shape, "candidate_blocks")
        if candidate_topk:
            raise ValueError(
                "reindex consumes candidates; it does not generate new ones"
            )
    shapes = ((tokens, topk), (tokens,), (tokens, candidate_topk), (tokens,))
    if out is None:
        out = tuple(
            torch.empty(shape, dtype=torch.int32, device=index_q.device)
            for shape in shapes
        )
    if len(out) != 4:
        raise ValueError(
            "out must contain row IDs, row lengths, block IDs, block lengths"
        )
    for tensor, shape in zip(out, shapes, strict=True):
        _output(tensor, shape, torch.int32, index_q.device)
    return out


@register_kernel(
    "attention",
    "dsv41_index_topk",
    name="triton_dsv41_index_topk",
    solution="triton",
    capability=_CAPABILITY,
    signatures=_QUERY_SIGNATURES,
    priority=Priority.PORTABLE,
)
def index_topk(
    index_q,
    weights,
    index_cache,
    page_table,
    visible_lens,
    candidate_blocks,
    topk,
    candidate_topk,
    candidate_block_size,
    query_chunk_size,
    score_chunk_size,
    process_group,
    out,
):
    out = _index_topk_outputs(
        index_q,
        weights,
        index_cache,
        page_table,
        visible_lens,
        candidate_blocks,
        topk,
        candidate_topk,
        candidate_block_size,
        query_chunk_size,
        score_chunk_size,
        process_group,
        out,
    )
    tokens = index_q.shape[0]
    row_out, row_lens, block_out, block_lens = out
    row_out.fill_(-1)
    row_lens.zero_()
    block_out.fill_(-1)
    block_lens.zero_()
    if not tokens or not page_table.shape[1]:
        return out
    # ponytail: exact bitonic TopK is deliberately portable. At long *visible*
    # histories a radix selector can replace it without changing this API.
    parts = 16
    tile = max(16, min(256, 1 << (score_chunk_size.bit_length() - 1)))
    row_k = max(tile, triton.next_power_of_2(topk))
    block_k = max(tile // 8, triton.next_power_of_2(max(1, candidate_topk)))
    query_chunk_size = min(query_chunk_size, 256)
    for start in range(0, tokens, query_chunk_size):
        end = min(start + query_chunk_size, tokens)
        q, w, shards = _index_gather_heads(
            index_q[start:end], weights[start:end], process_group
        )
        q = index_q_quantize(q, None)
        table, visible = page_table[start:end], visible_lens[start:end]
        candidates = None if candidate_blocks is None else candidate_blocks[start:end]
        shape = (end - start, parts * row_k)
        row_scores = torch.empty(shape, dtype=torch.float32, device=q.device)
        row_ids = torch.empty(shape, dtype=torch.int64, device=q.device)
        shape = (end - start, parts * block_k if candidate_topk else 0)
        block_scores = torch.empty(shape, dtype=torch.float32, device=q.device)
        block_ids = torch.empty(shape, dtype=torch.int64, device=q.device)
        _index_scan_kernel[(end - start, parts)](
            q,
            w,
            index_cache,
            table,
            visible,
            candidates,
            row_scores,
            row_ids,
            block_scores,
            block_ids,
            *q.stride(),
            *w.stride(),
            *index_cache.stride(),
            *table.stride(),
            visible.stride(0),
            *(candidates.stride() if candidates is not None else (0, 0)),
            index_cache.shape[0],
            table.shape[1],
            -1 if candidates is None else candidates.shape[1],
            q.shape[1],
            index_q.shape[1],
            shards,
            max(16, triton.next_power_of_2(q.shape[1])),
            parts,
            tile,
            min(tile, score_chunk_size),
            row_k,
            block_k,
            bool(candidate_topk),
            enable_fp_fusion=False,
        )
        _index_finish_parts(
            row_scores, row_ids, topk, row_out[start:end], row_lens[start:end]
        )
        if candidate_topk:
            _index_finish_parts(
                block_scores,
                block_ids,
                candidate_topk,
                block_out[start:end],
                block_lens[start:end],
            )
    return out


@triton.jit
def _compressor_pool(
    CONTENT,
    SCORES,
    PREVIOUS,
    TAIL,
    SLOTS,
    ACTIVE,
    OUT,
    NORM,
    EPS: tl.constexpr,
    HAS_NORM: tl.constexpr,
    N: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    P0: tl.constexpr,
    S0: tl.constexpr,
    A0: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    PAGES: tl.constexpr,
):
    row = tl.program_id(0)
    channels = tl.arange(0, 512)
    live = tl.load(ACTIVE + row * A0)
    previous = tl.load(PREVIOUS + row * P0)
    slot = tl.load(SLOTS + row * S0)
    in_current = live & (previous >= 0) & (previous < N)
    in_tail = live & (previous < 0) & (slot >= 2) & (slot < PAGES * 2)
    old_content = tl.load(CONTENT + previous * C0 + channels * C1, in_current, 0.0)
    old_scores = tl.load(SCORES + previous * G0 + channels * G1, in_current, 0.0)
    tail_offset = slot // 2 * T0 + slot % 2 * T1 + channels * T3
    tail_content = tl.load(TAIL + tail_offset, in_tail, 0.0)
    tail_scores = tl.load(TAIL + tail_offset + T2, in_tail, 0.0)
    old_content = tl.where(in_current, old_content, tail_content)
    old_scores = tl.where(in_current, old_scores, tail_scores)
    current = tl.load(CONTENT + row * C0 + channels * C1, live, 0.0)
    scores = tl.load(SCORES + row * G0 + channels * G1, live, 0.0)
    maximum = tl.maximum(old_scores, scores)
    old_exp = libdevice.exp(old_scores - maximum)
    new_exp = libdevice.exp(scores - maximum)
    denominator = old_exp + new_exp
    old_weight = tl.div_rn(old_exp, denominator)
    new_weight = tl.div_rn(new_exp, denominator)
    # Preserve separate FP32 products/addition before the caller's BF16 cast.
    pooled = tl.inline_asm_elementwise(
        "{ .reg .f32 a, b; mul.rn.f32 a, $1, $2; mul.rn.f32 b, $3, $4; add.rn.f32 $0, a, b; }",
        constraints="=f,f,f,f,f",
        args=[old_content, old_weight, current, new_weight],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    pooled = tl.where(live, pooled, 0.0)
    if HAS_NORM:
        # The released compressor rounds pooled inputs to BF16 before RMSNorm.
        pooled = pooled.to(tl.bfloat16).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(pooled * pooled, 0) / 512 + EPS)
        pooled = pooled * inv_rms
        pooled = pooled * tl.load(NORM + channels).to(tl.float32)
    tl.store(OUT + row * O0 + channels * O1, pooled)


@register_kernel(
    "attention",
    "dsv41_compressor_pool",
    name="triton_dsv41_compressor_pool",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.float32))}),
    priority=Priority.PORTABLE,
)
def compressor_pool(
    content, scores, previous, tail, tail_slots, active, out, norm_weight, norm_eps
):
    if (
        content.ndim != 2
        or content.shape[1] != 512
        or scores.shape != content.shape
        or content.dtype != torch.float32
        or scores.dtype != torch.float32
        or tail.ndim != 4
        or tail.shape[1:] != (2, 2, 512)
        or tail.dtype != torch.float32
    ):
        raise ValueError(
            "Compressor pooling requires FP32 [N,512] and [pages,2,2,512] tail"
        )
    count = content.shape[0]
    for value in (previous, tail_slots):
        if value.shape != (count,) or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Compressor previous rows and slots must be integer [N]")
    if active.shape != (count,) or active.dtype != torch.bool:
        raise ValueError("Compressor active mask must be bool [N]")
    if any(
        x.device != content.device for x in (scores, previous, tail, tail_slots, active)
    ):
        raise ValueError("Compressor operands must share one device")
    dtype = torch.float32 if norm_weight is None else torch.bfloat16
    if norm_weight is not None and (
        norm_weight.shape != (512,)
        or norm_weight.device != content.device
        or norm_eps <= 0
    ):
        raise ValueError(
            "Compressor normalization requires a device [512] weight and positive epsilon"
        )
    if out is None:
        out = torch.empty(content.shape, device=content.device, dtype=dtype)
    elif (
        out.shape != content.shape or out.dtype != dtype or out.device != content.device
    ):
        raise ValueError("Compressor output must be FP32 [N,512] on the input device")
    if count and any(
        out.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        for value in (content, scores, tail)
    ):
        raise ValueError("Compressor output must not alias projection or tail storage")
    if count:
        _compressor_pool[(count,)](
            content,
            scores,
            previous,
            tail,
            tail_slots,
            active,
            out,
            norm_weight,
            norm_eps,
            norm_weight is not None,
            count,
            *content.stride(),
            *scores.stride(),
            previous.stride(0),
            tail_slots.stride(0),
            active.stride(0),
            *tail.stride(),
            *out.stride(),
            tail.shape[0],
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _rotate_interleaved(value, partner, cosine, sine, odd):
    first = value.to(tl.float32) * cosine
    second = partner.to(tl.float32) * sine
    return tl.where(odd, first + second, first - second)


@triton.jit
def _swa_rope_insert(
    X,
    POS,
    CS,
    CACHE,
    SLOTS,
    OUT,
    X0: tl.constexpr,
    X1: tl.constexpr,
    POS0: tl.constexpr,
    CS0: tl.constexpr,
    C0: tl.constexpr,
    S0: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    P: tl.constexpr,
    NP: tl.constexpr,
    MAX_POS: tl.constexpr,
    HAS_OUT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 512)
    position = tl.maximum(tl.load(POS + token * POS0).to(tl.int64), 0)
    position_valid = position < MAX_POS
    tl.device_assert(
        position_valid, "SWA RoPE position is outside the cosine/sine table"
    )
    slot = tl.load(SLOTS + token * S0).to(tl.int64)
    live = (slot >= P) & (slot < NP * P)
    x = tl.load(X + token * X0 + d * X1).to(tl.float32)
    # RMSNorm has already rounded to BF16 in the caller's existing primitive.
    normalized = x
    partner = tl.gather(normalized, d ^ 1, axis=0)
    pair = (d - 448) // 2
    rope = d >= 448
    cosine = tl.load(CS + position * CS0 + pair, rope & position_valid, 1.0)
    sine = tl.load(CS + position * CS0 + 32 + pair, rope & position_valid, 0.0)
    rotated = tl.where(
        rope,
        _rotate_interleaved(normalized, partner, cosine, sine, (d & 1) != 0).to(
            tl.bfloat16
        ),
        normalized.to(tl.bfloat16),
    )
    values = tl.reshape(rotated.to(tl.float32), (16, 32))
    amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
    # Use the same group32 IEEE exponent rule as the portable cache codec.
    raw_scale = amax * (1.0 / 448.0)
    bits = raw_scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quantized = tl.reshape(tl.div_rn(values, scale[:, None]), (512,)).to(tl.float8e4nv)
    if HAS_OUT:
        restored = quantized.to(tl.float32) * tl.reshape(
            tl.broadcast_to(scale[:, None], (16, 32)), (512,)
        )
        tl.store(OUT + token * O0 + d * O1, restored.to(tl.bfloat16))
    page = tl.maximum(slot, 0) // P
    row = tl.maximum(slot, 0) % P
    tl.store(
        CACHE + page * C0 + row * 512 + d, quantized.to(tl.uint8, bitcast=True), live
    )
    tl.store(
        CACHE + page * C0 + P * 512 + row * 16 + tl.arange(0, 16),
        exponent.to(tl.uint8),
        live,
    )


@register_kernel(
    "attention",
    "dsv41_swa_rope_scatter",
    name="triton_dsv41_swa_rope_scatter",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits={},
    priority=Priority.PORTABLE,
)
def triton_dsv41_swa_rope_scatter(values, positions, cos_sin_cache, cache, slots, out):
    _cache(cache, "swa")
    _same_device(values, (positions, cos_sin_cache, cache, slots, out))
    if values.dtype != torch.bfloat16 or values.ndim != 2 or values.shape[1] != 512:
        raise ValueError("SWA values must be BF16 [tokens,512]")
    _integers(positions, values.shape[:1], "positions")
    _integers(slots, values.shape[:1], "slots")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("SWA RoPE table must be FP32 [positions,64]")
    if cache.stride(1) != 528 or cache.stride(2) != 1:
        raise ValueError("Fused SWA requires contiguous page bytes")
    if out is not None:
        _output(out, values.shape, torch.bfloat16, values.device)
    page_size = 64
    cache = cache.as_strided((cache.shape[0], page_size * 528), (cache.stride(0), 1))
    if values.shape[0] == 0:
        return
    _swa_rope_insert[(values.shape[0],)](
        values,
        positions,
        cos_sin_cache,
        cache,
        slots,
        out,
        values.stride(0),
        values.stride(1),
        positions.stride(0),
        cos_sin_cache.stride(0),
        cache.stride(0),
        slots.stride(0),
        0 if out is None else out.stride(0),
        0 if out is None else out.stride(1),
        page_size,
        cache.shape[0],
        cos_sin_cache.shape[0],
        out is not None,
        num_warps=4,
        enable_fp_fusion=False,
        debug=True,
    )


def _register_rope(name):
    return register_kernel(
        "attention",
        name,
        name="triton_" + name,
        solution="triton",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
        signatures=frozenset(
            format_signature(values=dense_tensor_format(t))
            for t in (torch.bfloat16, torch.float16)
        ),
        traits={},
        priority=Priority.PORTABLE,
    )


@triton.jit
def _rope_inplace_kernel(
    X,
    POS,
    CS,
    X0,
    X1,
    P0,
    CS0,
    R: tl.constexpr,
    MAX_POS: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    # Keep padding/compressor sentinels in the caller's metadata. Only this
    # frequency-table lookup maps negative positions to the identity row.
    position = tl.maximum(tl.load(POS + token * P0).to(tl.int64), 0)
    valid = position < MAX_POS
    tl.device_assert(valid, "RoPE position is outside the cosine/sine table")
    pair = tl.arange(0, B)
    mask = (pair < R // 2) & valid
    base = X + token * X0 + head * X1
    even = tl.load(base + 2 * pair, mask, 0.0)
    odd = tl.load(base + 2 * pair + 1, mask, 0.0)
    cosine = tl.load(CS + position * CS0 + pair, mask, 0.0)
    sine = tl.load(CS + position * CS0 + R // 2 + pair, mask, 0.0)
    out_even = _rotate_interleaved(even, odd, cosine, sine, False)
    out_odd = _rotate_interleaved(odd, even, cosine, sine, True)
    tl.store(base + 2 * pair, out_even.to(even.dtype), mask)
    tl.store(base + 2 * pair + 1, out_odd.to(odd.dtype), mask)


@_register_rope("dsv41_rope_inplace")
def rope_inplace(values, positions, cache):
    if values.numel() == 0:
        return values
    rotary_dim = cache.shape[1]
    heads = 1 if values.ndim == 2 else values.shape[1]
    _rope_inplace_kernel[(values.shape[0], heads)](
        values[..., -rotary_dim:],
        positions,
        cache,
        values.stride(0),
        0 if values.ndim == 2 else values.stride(1),
        positions.stride(0),
        cache.stride(0),
        rotary_dim,
        cache.shape[0],
        max(triton.next_power_of_2(rotary_dim // 2), 16),
        num_warps=4,
        enable_fp_fusion=False,
        debug=True,
    )
    return values


@triton.jit
def _rope_pad_query(
    X,
    POS,
    CS,
    OUT,
    X0: tl.constexpr,
    X1: tl.constexpr,
    P0: tl.constexpr,
    CS0: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    R: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 512)
    position = tl.maximum(tl.load(POS + token * P0).to(tl.int64), 0)
    valid_position = position < MAX_POS
    tl.device_assert(valid_position, "RoPE position is outside the cosine/sine table")
    live = head < H
    x = tl.load(X + token * X0 + head * X1 + d, live, 0.0)
    partner = tl.gather(x, d ^ 1, axis=0).to(tl.float32)
    tail = d >= 512 - R
    pair = (d - (512 - R)) // 2
    cosine = tl.load(CS + position * CS0 + pair, tail & live & valid_position, 1.0)
    sine = tl.load(
        CS + position * CS0 + R // 2 + pair, tail & live & valid_position, 0.0
    )
    rotated = _rotate_interleaved(x, partner, cosine, sine, (d & 1) != 0)
    value = tl.where(tail, rotated.to(x.dtype), x)
    value = tl.where(live, value, 0.0)
    tl.store(OUT + (token * HP + head) * 512 + d, value)


@_register_rope("dsv41_rope_pad_query")
def rope_pad_query(values, positions, cache):
    n, h, _ = values.shape
    hp = 64 if h <= 64 else 128
    output = torch.empty((n, hp, 512), dtype=values.dtype, device=values.device)
    return _launch_query(values, positions, cache, output, hp)


def _launch_query(values, positions, cache, output, heads_to_write):
    n, h, _ = values.shape
    hp = output.shape[1]
    if n:
        _rope_pad_query[(n, heads_to_write)](
            values,
            positions,
            cache,
            output,
            values.stride(0),
            values.stride(1),
            positions.stride(0),
            cache.stride(0),
            h,
            hp,
            cache.shape[1],
            cache.shape[0],
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
    return output


@triton.jit(do_not_specialize=["X0", "X1"], do_not_specialize_on_alignment=["X0", "X1"])
def _pack_index_queries(X, V, S, X0, X1):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 128)
    x = tl.load(X + row * X0 + d * X1).to(tl.float32)
    grouped = tl.reshape(x, (4, 32))
    amax = tl.maximum(tl.max(tl.abs(grouped), 1), 6.0 * 1.1754943508222875e-38)
    raw = tl.div_rn(amax, 6.0)
    bits = raw.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    normalized = tl.reshape(tl.div_rn(grouped, scale[:, None]), (128,))
    code = _e2m1_encode(normalized).to(tl.int32)
    pairs = tl.reshape(code, (64, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
    word = tl.sum(exponent.to(tl.uint32) << (tl.arange(0, 4) * 8), 0)
    tl.store(V + row * 64 + tl.arange(0, 64), packed)
    tl.store(S + row, word.to(tl.int32))


def pack_index_queries(values):
    flat = values.reshape(-1, 128)
    data = torch.empty(
        (*values.shape[:-1], 64), dtype=torch.uint8, device=values.device
    )
    scales = torch.empty(values.shape[:-1], dtype=torch.int32, device=values.device)
    if flat.shape[0]:
        _pack_index_queries[(flat.shape[0],)](
            flat,
            data,
            scales,
            flat.stride(0),
            flat.stride(1),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return data, scales


@triton.jit(
    do_not_specialize=["T0", "T1", "L0", "TC", "NP", "CAP"],
    do_not_specialize_on_alignment=["T0", "T1", "L0", "TC", "NP", "CAP"],
)
def _safe_metadata(
    T, L, OT, OL, T0, T1, L0, TC, NP, CAP, P: tl.constexpr, B: tl.constexpr
):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    column = tile * B + tl.arange(0, B)
    page = tl.load(T + row * T0 + column * T1, column < TC, other=0).to(tl.int64)
    valid = (page >= 0) & (page < NP)
    tl.store(OT + row * TC + column, tl.where(valid, page, 0).to(tl.int32), column < TC)
    if tile == 0:
        length = tl.minimum(
            tl.maximum(tl.load(L + row * L0), 0), tl.minimum(CAP, TC * P)
        )
        tl.store(OL + row, length.to(tl.int32))


def safe_metadata(table, lengths, pages, capacity, page_size):
    safe_table = torch.empty(table.shape, dtype=torch.int32, device=table.device)
    safe_lengths = torch.empty(lengths.shape, dtype=torch.int32, device=lengths.device)
    if lengths.numel():
        _safe_metadata[(table.shape[0], triton.cdiv(table.shape[1], 256))](
            table,
            lengths,
            safe_table,
            safe_lengths,
            table.stride(0),
            table.stride(1),
            lengths.stride(0),
            table.shape[1],
            pages,
            capacity,
            page_size,
            256,
            num_warps=4,
        )
    return safe_table, safe_lengths


@triton.jit(
    do_not_specialize=["X0", "L0", "T0", "T1", "TC", "NP", "CAP"],
    do_not_specialize_on_alignment=["X0", "L0", "T0", "T1", "TC", "NP", "CAP"],
)
def _clean_logits(
    X,
    L,
    T,
    O,
    X0,
    L0,
    T0,
    T1,
    TC,
    NP,
    CAP,
    P: tl.constexpr,
    PAGED: tl.constexpr,
    B: tl.constexpr,
):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    column = tile * B + tl.arange(0, B)
    valid = (column < CAP) & (column < tl.load(L + row * L0))
    if PAGED:
        page = tl.load(
            T + row * T0 + (column // P) * T1, valid & (column // P < TC), other=0
        ).to(tl.int64)
        valid &= (page >= 0) & (page < NP) & (column // P < TC)
    value = tl.load(X + row * X0 + column, valid, other=-float("inf"))
    tl.store(O + row * CAP + column, value, column < CAP)


def clean_logits(values, lengths, table, pages, capacity, page_size):
    if (
        values.ndim != 2
        or values.shape[0] != lengths.numel()
        or values.shape[1] < capacity
        or values.stride(1) != 1
        or values.dtype != torch.float32
    ):
        raise RuntimeError("DeepGEMM returned incompatible FP32 score geometry")
    output = torch.empty(
        (values.shape[0], capacity), dtype=torch.float32, device=values.device
    )
    if output.numel():
        _clean_logits[(values.shape[0], triton.cdiv(capacity, 1024))](
            values,
            lengths,
            table if table is not None else lengths,
            output,
            values.stride(0),
            lengths.stride(0),
            table.stride(0) if table is not None else 0,
            table.stride(1) if table is not None else 0,
            table.shape[1] if table is not None else 0,
            pages,
            capacity,
            page_size,
            table is not None,
            1024,
            num_warps=4,
        )
    return output


@triton.jit(
    do_not_specialize=["C0", "S0", "N", "NP"],
    do_not_specialize_on_alignment=["C0", "S0", "N", "NP"],
)
def _gather_index_cache(C, S, V, F, C0, S0, N, NP, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    slot = tl.load(S + row * S0, row < N, other=-1).to(tl.int64)
    valid = (row < N) & (slot >= 0) & (slot < NP * P)
    page, offset = slot // P, slot % P
    d = tl.arange(0, 64)
    data = tl.load(
        C + page[:, None] * C0 + offset[:, None] * 64 + d[None, :],
        valid[:, None],
        other=0,
    )
    tl.store(V + row[:, None] * 64 + d[None, :], data, (row < N)[:, None])
    b = tl.arange(0, 4)
    sf = tl.load(
        C + page[:, None] * C0 + P * 64 + offset[:, None] * 4 + b[None, :],
        valid[:, None],
        other=127,
    ).to(tl.uint32)
    word = tl.sum(sf << (b[None, :] * 8), 1)
    tl.store(F + row, word.to(tl.int32), row < N)


def gather_index_cache(cache, slots, page_size):
    data = torch.empty((slots.numel(), 64), dtype=torch.uint8, device=cache.device)
    scales = torch.empty((slots.numel(),), dtype=torch.int32, device=cache.device)
    if slots.numel():
        _gather_index_cache[(triton.cdiv(slots.numel(), 32),)](
            cache,
            slots,
            data,
            scales,
            cache.stride(0),
            slots.stride(0),
            slots.numel(),
            cache.shape[0],
            page_size,
            32,
            num_warps=4,
        )
    return data, scales


@triton.jit(
    do_not_specialize=["L0", "N", "CAP"],
    do_not_specialize_on_alignment=["L0", "N", "CAP"],
)
def _dense_ranges(L, START, END, L0, N, CAP, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    length = tl.load(L + row * L0, row < N, other=0)
    tl.store(START + row, 0, row < N)
    tl.store(END + row, tl.minimum(tl.maximum(length, 0), CAP).to(tl.int32), row < N)


def dense_ranges(lengths, capacity):
    starts = torch.empty(lengths.shape, dtype=torch.int32, device=lengths.device)
    ends = torch.empty_like(starts)
    if lengths.numel():
        _dense_ranges[(triton.cdiv(lengths.numel(), 256),)](
            lengths,
            starts,
            ends,
            lengths.stride(0),
            lengths.numel(),
            capacity,
            256,
            num_warps=4,
        )
    return starts, ends


@triton.jit(
    do_not_specialize=["X0", "X1", "L0", "S0", "W", "SW"],
    do_not_specialize_on_alignment=["X0", "X1", "L0", "S0", "W", "SW"],
)
def _prepare_dense(
    X,
    L,
    S,
    E,
    X0,
    X1,
    L0,
    S0,
    W,
    SW,
    B: tl.constexpr,
):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    p = tile * B + tl.arange(0, B)
    end = tl.minimum(tl.maximum(tl.load(L + row * L0), 0), W)
    value = tl.load(X + row * X0 + p * X1, (p < W) & (p < end), other=-float("inf"))
    tl.store(S + row * S0 + p, value, p < SW)
    if tile == 0:
        tl.store(E + row, end)


@triton.jit(
    do_not_specialize=["X0", "X1", "L0", "S0", "W", "SW"],
    do_not_specialize_on_alignment=["X0", "X1", "L0", "S0", "W", "SW"],
)
def _reduce_blocks(
    X,
    L,
    S,
    E,
    X0,
    X1,
    L0,
    S0,
    W,
    SW,
    BLOCK_SIZE: tl.constexpr,
    GROUP: tl.constexpr,
    PIN_NEWEST: tl.constexpr,
    B: tl.constexpr,
):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    block = tile * B + tl.arange(0, B)
    lane = tl.arange(0, GROUP)
    end = tl.minimum(tl.maximum(tl.load(L + row * L0), 0), W)
    position = block[:, None] * BLOCK_SIZE + lane[None, :]
    value = tl.load(
        X + row * X0 + position * X1,
        (position < end) & (lane[None, :] < BLOCK_SIZE),
        other=-float("inf"),
    )
    scores = tl.max(value, axis=1)
    if PIN_NEWEST:
        scores = tl.where(
            (end > 0) & (block == (end - 1) // BLOCK_SIZE) & (scores > -float("inf")),
            float("inf"),
            scores,
        )
    tl.store(S + row * S0 + block, scores, block < SW)
    if tile == 0:
        tl.store(E + row, (end + BLOCK_SIZE - 1) // BLOCK_SIZE)


@triton.jit(
    do_not_specialize=["C0", "C1", "L0", "N", "W"],
    do_not_specialize_on_alignment=["C0", "C1", "L0", "N", "W"],
)
def _normalize_candidates(
    C,
    L,
    O,
    E,
    C0,
    C1,
    L0,
    N,
    W,
    BLOCK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    end = tl.minimum(tl.maximum(tl.load(L + row * L0), 0), W)
    candidate = tl.load(C + row * C0 + i * C1, i < N, other=-1)
    count = (end + BLOCK_SIZE - 1) // BLOCK_SIZE
    candidate = tl.where(
        (candidate >= 0) & (candidate < count), candidate, 2147483647
    ).to(tl.int32)
    candidate = tl.sort(candidate, descending=False)
    previous = tl.gather(candidate, tl.maximum(i - 1, 0), axis=0)
    valid = (candidate != 2147483647) & ((i == 0) | (candidate != previous))
    tl.store(O + row * N + i, tl.where(valid, candidate, -1), i < N)
    span = i * BLOCK_SIZE + tl.minimum(BLOCK_SIZE, end - candidate * BLOCK_SIZE)
    tl.store(E + row, tl.max(tl.where(valid, span, 0), axis=0))


@triton.jit(
    do_not_specialize=["X0", "X1", "L0", "S0", "W", "NC", "SW"],
    do_not_specialize_on_alignment=["X0", "X1", "L0", "S0", "W", "NC", "SW"],
)
def _gather_candidates(
    X,
    L,
    C,
    S,
    X0,
    X1,
    L0,
    S0,
    W,
    NC,
    SW,
    BLOCK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i = tile * B + tl.arange(0, B)
    candidate = tl.load(C + row * NC + i // BLOCK_SIZE, i < NC * BLOCK_SIZE, other=-1)
    position = candidate * BLOCK_SIZE + i % BLOCK_SIZE
    end = tl.minimum(tl.maximum(tl.load(L + row * L0), 0), W)
    valid = (candidate >= 0) & (position < end) & (i < NC * BLOCK_SIZE)
    value = tl.load(X + row * X0 + position * X1, valid, other=-float("inf"))
    tl.store(S + row * S0 + i, value, i < SW)


@triton.jit(
    do_not_specialize=["I0", "S0", "NC", "W"],
    do_not_specialize_on_alignment=["I0", "S0", "NC", "W"],
)
def _finish_selection(
    I,
    S,
    E,
    C,
    O,
    I0,
    S0,
    NC,
    K: tl.constexpr,
    W,
    BLOCK_SIZE: tl.constexpr,
    HAS_CANDIDATES: tl.constexpr,
    POSITION_SORT: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    selected = tl.load(I + row * I0 + i, i < K, other=-1)
    end = tl.load(E + row)
    valid = (i < K) & (selected >= 0) & (selected < end)
    score = tl.load(S + row * S0 + selected, valid, other=-float("inf"))
    valid &= score > -float("inf")
    position = selected
    if HAS_CANDIDATES:
        candidate = tl.load(C + row * NC + selected // BLOCK_SIZE, valid, other=-1)
        position = candidate * BLOCK_SIZE + selected % BLOCK_SIZE
        valid &= (candidate >= 0) & (position < W)
    position = tl.where(valid, position, 2147483647)
    if POSITION_SORT:
        position = tl.sort(position, descending=False)
    tl.store(O + row * K + i, tl.where(position != 2147483647, position, -1), i < K)


@triton.jit(
    do_not_specialize=["COUNT", "WIDTH", "L0"],
    do_not_specialize_on_alignment=["COUNT", "WIDTH", "L0"],
)
def _clamp_selection_ends(L, E, COUNT, WIDTH, L0, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    value = tl.load(L + row * L0, row < COUNT, other=0)
    tl.store(E + row, tl.minimum(tl.maximum(value, 0), WIDTH), row < COUNT)


def prepare_native_dense_scores(logits, lengths, alignment):
    """Borrow aligned read-only logits; native exclusive ends enforce causality.

    Full aligned logical rows avoid vector-load access beyond tensor storage.
    Unaligned/strided channels keep the masked/aligned copy implementation.
    Unlike prepare_scores, future score values in a borrowed view are unspecified
    to the consumer and must never be accessed beyond the clamped ends.
    """
    if (
        logits.shape[1] % alignment == 0
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and logits.stride(0) % alignment == 0
        and logits.data_ptr() % (alignment * logits.element_size()) == 0
    ):
        ends = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
        if logits.shape[0]:
            _clamp_selection_ends[(triton.cdiv(logits.shape[0], 256),)](
                lengths, ends, logits.shape[0], logits.shape[1], lengths.stride(0), 256
            )
        return logits, ends
    return prepare_scores(logits, lengths, 1, alignment, False)


def prepare_scores(logits, lengths, block_size, alignment, pin_newest):
    """Fuse causal masking/alignment, optionally block maxima and newest pin."""
    width = logits.shape[1]
    logical_width = (width + block_size - 1) // block_size
    stride = (logical_width + alignment - 1) // alignment * alignment
    scores = torch.empty(
        (logits.shape[0], stride), dtype=torch.float32, device=logits.device
    )
    end = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    if block_size == 1 and not pin_newest:
        _prepare_dense[(logits.shape[0], triton.cdiv(stride, 1024))](
            logits,
            lengths,
            scores,
            end,
            logits.stride(0),
            logits.stride(1),
            lengths.stride(0),
            stride,
            width,
            stride,
            1024,
        )
    else:
        _reduce_blocks[(logits.shape[0], triton.cdiv(stride, 128))](
            logits,
            lengths,
            scores,
            end,
            logits.stride(0),
            logits.stride(1),
            lengths.stride(0),
            stride,
            width,
            stride,
            block_size,
            triton.next_power_of_2(block_size),
            pin_newest,
            128,
        )
    return scores, end


def prepare_candidate_scores(logits, lengths, candidates, block_size, alignment):
    """Deduplicate candidate block IDs, then gather a bounded causal score table."""
    count = candidates.shape[1]
    normalized = torch.empty(
        (logits.shape[0], count), dtype=torch.int32, device=logits.device
    )
    end = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    stride = (count * block_size + alignment - 1) // alignment * alignment
    scores = torch.empty(
        (logits.shape[0], stride), dtype=torch.float32, device=logits.device
    )
    _normalize_candidates[(logits.shape[0],)](
        candidates,
        lengths,
        normalized,
        end,
        candidates.stride(0),
        candidates.stride(1),
        lengths.stride(0),
        count,
        logits.shape[1],
        block_size,
        triton.next_power_of_2(count),
        num_warps=8,
    )
    _gather_candidates[(logits.shape[0], triton.cdiv(stride, 1024))](
        logits,
        lengths,
        normalized,
        scores,
        logits.stride(0),
        logits.stride(1),
        lengths.stride(0),
        stride,
        logits.shape[1],
        count,
        stride,
        block_size,
        1024,
    )
    return scores, end, normalized


def finish_selection(
    indices, scores, lengths, candidates, width, block_size, sort_positions
):
    """Filter masked native selections and return logical IDs with -1 padding."""
    output = torch.empty(indices.shape, dtype=torch.int32, device=indices.device)
    _finish_selection[(indices.shape[0],)](
        indices,
        scores,
        lengths,
        candidates if candidates is not None else indices,
        output,
        indices.stride(0),
        scores.stride(0),
        candidates.shape[1] if candidates is not None else 0,
        indices.shape[1],
        width,
        block_size,
        candidates is not None,
        sort_positions,
        triton.next_power_of_2(indices.shape[1]),
        num_warps=8,
    )
    return output
