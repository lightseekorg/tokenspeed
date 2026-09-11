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
    keys = cache_gather(cache, slots, "index", None)
    # Reference einsum materializes BF16 dots; weighting and the head reduction
    # each round to weights' dtype. Keeping FP32 throughout changes TopK ties.
    dots = torch.bmm(q, keys.transpose(1, 2))
    scores = (dots.relu_() * weights.unsqueeze(-1)).sum(dim=1)
    if process_group is not None:
        torch.distributed.all_reduce(scores, group=process_group)
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


def _merge_topk(scores, ids, new_scores, new_ids, k):
    scores = torch.cat((scores, new_scores.float()), dim=1)
    ids = torch.cat((ids, new_ids.expand_as(new_scores)), dim=1)
    values, positions = scores.topk(min(k, scores.shape[1]), dim=1, sorted=False)
    return values, ids.gather(1, positions)


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
    # ponytail: Full remains an O(queries * history) scan, but scratch is only
    # O(query_chunk * (score_chunk * heads + topk + candidate_topk)); replace
    # repeated torch TopK merges with a fused streaming selector if latency binds.
    width = (
        page_table.shape[1] * 64
        if candidate_blocks is None
        else candidate_blocks.shape[1] * 8
    )
    if not width or not page_table.shape[1]:
        return out
    for start in range(0, tokens, query_chunk_size):
        end = min(start + query_chunk_size, tokens)
        q = index_q_quantize(index_q[start:end], None)
        table = page_table[start:end]
        visible = visible_lens[start:end, None]
        row_scores = torch.empty((end - start, 0), dtype=torch.float32, device=q.device)
        row_ids = torch.empty((end - start, 0), dtype=torch.int64, device=q.device)
        block_scores, block_ids = row_scores, row_ids
        for offset in range(0, width, score_chunk_size):
            col = torch.arange(
                offset, min(offset + score_chunk_size, width), device=q.device
            )
            if candidate_blocks is None:
                logical = col.expand(end - start, -1)
            else:
                blocks = candidate_blocks[start:end].to(torch.int64)[:, col // 8]
                logical = torch.where(blocks >= 0, blocks * 8 + col % 8, -1)
            valid = (
                (logical >= 0) & (logical < visible) & (logical < table.shape[1] * 64)
            )
            pages = table.gather(1, (logical // 64).clamp(0, table.shape[1] - 1)).to(
                torch.int64
            )
            slots = pages * 64 + logical % 64
            slots = slots.masked_fill(
                ~valid | (pages < 0) | (pages >= index_cache.shape[0]), -1
            )
            scores = _score(q, weights[start:end], index_cache, slots, process_group)
            row_scores, row_ids = _merge_topk(
                row_scores, row_ids, scores, logical, topk
            )
            if candidate_topk:
                new_blocks = (col[::8] // 8).expand(end - start, -1)
                new_scores = scores.float().unflatten(1, (-1, 8)).amax(dim=-1)
                latest = (visible - 1) // 8
                new_scores = new_scores.masked_fill(
                    (new_blocks == latest) & (visible > 0) & (new_scores > -torch.inf),
                    torch.inf,
                )
                block_scores, block_ids = _merge_topk(
                    block_scores, block_ids, new_scores, new_blocks, candidate_topk
                )
        _finish_topk(row_scores, row_ids, row_out[start:end], row_lens[start:end])
        if candidate_topk:
            _finish_topk(
                block_scores, block_ids, block_out[start:end], block_lens[start:end]
            )
    return out
