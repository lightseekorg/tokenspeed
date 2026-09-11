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
from tokenspeed_kernel._triton import tl, triton
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
def _pack_kernel(
    X,
    C,
    Slots,
    XS0,
    XS1,
    CP,
    CR,
    CB,
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
        base = C + (slot // PAGE_ROWS) * CP + (slot % PAGE_ROWS) * CR
        if FORMAT == "swa":
            encoded = (
                tl.clamp(y, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            )
            tl.store(base + d * CB, encoded)
        else:
            codes = _e2m1_encode(y).reshape(D // 2, 2)
            lo, hi = tl.split(codes)
            tl.store(base + tl.arange(0, D // 2) * CB, lo | (hi << 4))
        tl.store(base + (VALUES + tl.arange(0, D // GROUP)) * CB, scale_byte)


@triton.jit
def _gather_kernel(
    C,
    Slots,
    O,
    CP,
    CR,
    CB,
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
    base = C + (slot // PAGE_ROWS) * CP + (slot % PAGE_ROWS) * CR
    byte = tl.load(base + tl.where(VALUES == D, d, d // 2) * CB, valid, other=0)
    if FORMAT == "swa":
        value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        value = _e2m1_decode((byte >> ((d % 2) * 4)) & 15)
    scale_byte = tl.load(base + (VALUES + d // GROUP) * CB, valid, other=0)
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
):
    from tokenspeed_kernel.ops.attention import dsv4_prefill

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
    CR,
    CB,
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
        base = Cache + page * CP + (logical % 64) * CR
        byte = tl.load(base[None, :] + (d[:, None] // 2) * CB, valid[None, :], other=0)
        value = _e2m1_decode((byte >> ((d[:, None] % 2) * 4)) & 15)
        scale_byte = tl.load(
            base[None, :] + (64 + d[:, None] // 32) * CB, valid[None, :], other=0
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
