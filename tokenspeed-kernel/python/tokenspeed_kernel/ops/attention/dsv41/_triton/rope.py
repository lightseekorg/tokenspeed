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

"""Fused index and KV normalization, rotary embedding and cache writes."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _cache,
    _e2m1_encode,
    _global_e4m3_scale,
    _integers,
    _mxfp4_index_rows,
    _planar_offset,
    _rotate_interleaved,
    _same_device,
    _ue8m0_word,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _normalized_index_keys(
    X,
    W,
    POS,
    CS,
    N,
    X0: tl.constexpr,
    X1: tl.constexpr,
    P0: tl.constexpr,
    CS0: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MAX_POS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)
    x = tl.load(X + rows[:, None] * X0 + d[None, :] * X1, rows[:, None] < N, 0.0).to(
        tl.float32
    )
    rstd = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 128) + EPS)
    weight = tl.load(W + d).to(tl.float32)
    # Preserve the materialized BF16 RMSNorm output before the RoPE products.
    normalized = (x * rstd[:, None] * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(
        normalized, tl.broadcast_to((d ^ 1)[None, :], (ROWS, 128)), axis=1
    )
    position = tl.maximum(tl.load(POS + rows * P0, rows < N, 0).to(tl.int64), 0)
    valid = position < MAX_POS
    tl.device_assert(
        valid | (rows >= N), "RoPE position is outside the cosine/sine table"
    )
    rope = d >= 128 - R
    pair = (d - (128 - R)) // 2
    mask = (rows[:, None] < N) & valid[:, None] & rope[None, :]
    cosine = tl.load(CS + position[:, None] * CS0 + pair[None, :], mask, 1.0)
    sine = tl.load(CS + position[:, None] * CS0 + R // 2 + pair[None, :], mask, 0.0)
    first = normalized * cosine
    second = partner * sine
    rotated = tl.where((d[None, :] & 1) != 0, first + second, first - second)
    result = tl.where(rope[None, :], rotated, normalized)
    return result


@triton.jit(do_not_specialize=["N"])
def _index_key_norm_rope(
    X,
    W,
    POS,
    CS,
    OUT,
    N,
    X0: tl.constexpr,
    X1: tl.constexpr,
    P0: tl.constexpr,
    CS0: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MAX_POS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)
    result = _normalized_index_keys(
        X, W, POS, CS, N, X0, X1, P0, CS0, EPS, R, MAX_POS, ROWS
    )
    tl.store(OUT + rows[:, None] * 128 + d[None, :], result, rows[:, None] < N)


def _validate_index_key_inputs(values, weight, positions, cos_sin_cache, eps):
    if (
        values.ndim != 2
        or values.shape[1] != 128
        or values.dtype != torch.bfloat16
        or not values.is_cuda
    ):
        raise ValueError("Index keys must be GPU BF16 [tokens, 128]")
    if (
        weight.shape != (128,)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or eps <= 0
    ):
        raise ValueError(
            "Index normalization requires contiguous BF16 [128] weights "
            "and positive epsilon"
        )
    if positions.shape != (values.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Index positions must be integer [tokens]")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.stride(1) != 1
        or not 0 < cos_sin_cache.shape[1] <= 128
        or cos_sin_cache.shape[1] % 2
        or cos_sin_cache.shape[0] == 0
    ):
        raise ValueError(
            "Index RoPE requires an FP32 cosine/sine table "
            "with an even dimension up to 128"
        )
    _same_device(values, (weight, positions, cos_sin_cache))


@register_kernel(
    "attention",
    "dsv41_index_key_norm_rope",
    name="triton_dsv41_index_key_norm_rope",
    solution="triton",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def triton_dsv41_index_key_norm_rope(
    values: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize BF16 index keys and rotate their trailing interleaved pairs.

    Args:
        values: GPU BF16 [tokens, 128] index-key projection.
        weight: Contiguous BF16 [128] RMSNorm weight on the same device.
        positions: Integer [tokens] absolute positions; negative values use row 0.
        cos_sin_cache: FP32 [max_position, rotary_dim] cosine then sine table,
            with contiguous columns and a positive even rotary_dim up to 128.
        eps: Positive RMSNorm epsilon.

    Returns:
        Fresh contiguous BF16 rows with RMSNorm's intermediate rounding retained.
        Inputs are preserved.
    """
    _validate_index_key_inputs(values, weight, positions, cos_sin_cache, eps)
    out = torch.empty(values.shape, dtype=values.dtype, device=values.device)
    if values.shape[0]:
        _index_key_norm_rope[(triton.cdiv(values.shape[0], 4),)](
            values,
            weight,
            positions,
            cos_sin_cache,
            out,
            values.shape[0],
            *values.stride(),
            positions.stride(0),
            cos_sin_cache.stride(0),
            eps,
            cos_sin_cache.shape[1],
            cos_sin_cache.shape[0],
            4,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
    return out


@triton.jit
def _index_query_rope_weights(
    Q,
    W,
    POS,
    CS,
    OUT,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    P0: tl.constexpr,
    CS0: tl.constexpr,
    SCALE: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    position = tl.maximum(tl.load(POS + token * P0).to(tl.int64), 0)
    valid = position < MAX_POS
    tl.device_assert(valid, "RoPE position is outside the cosine/sine table")
    pair = tl.arange(0, 32)
    base = Q + token * Q0 + head * Q1 + 64 * Q2
    even = tl.load(base + 2 * pair * Q2, valid, 0.0).to(tl.float32)
    odd = tl.load(base + (2 * pair + 1) * Q2, valid, 0.0).to(tl.float32)
    cosine = tl.load(CS + position * CS0 + pair, valid, 0.0)
    sine = tl.load(CS + position * CS0 + 32 + pair, valid, 0.0)
    # Separate products match the existing RoPE's FP32 operations and BF16 store.
    out_even = even * cosine - odd * sine
    out_odd = odd * cosine + even * sine
    tl.store(base + 2 * pair * Q2, out_even.to(tl.bfloat16), valid)
    tl.store(base + (2 * pair + 1) * Q2, out_odd.to(tl.bfloat16), valid)
    weight = tl.load(W + token * W0 + head * W1).to(tl.float32)
    # Match Torch's FP32 multiplication followed by BF16 rounding.
    scaled = (weight * tl.full((), SCALE, tl.float32)).to(tl.bfloat16)
    tl.store(OUT + token * tl.num_programs(1) + head, scaled)


def _validate_index_query_inputs(query, head_weights, positions, cos_sin_cache):
    if (
        query.ndim != 3
        or query.shape[2] != 128
        or query.dtype != torch.bfloat16
        or not query.is_cuda
        or query.shape[1] < 1
    ):
        raise ValueError("Index queries must be GPU BF16 [tokens, heads, 128]")
    if head_weights.shape != query.shape[:2] or head_weights.dtype != torch.bfloat16:
        raise ValueError("Index head weights must be BF16 [tokens, heads]")
    if positions.shape != query.shape[:1] or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Index positions must be integer [tokens]")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.shape[0] == 0
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("Index RoPE requires an FP32 [max_position, 64] table")
    _same_device(query, (head_weights, positions, cos_sin_cache))


@register_kernel(
    "attention",
    "dsv41_index_query_rope_weights",
    name="triton_dsv41_index_query_rope_weights",
    solution="triton",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def triton_dsv41_index_query_rope_weights(
    query: torch.Tensor,
    head_weights: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate fresh index queries in place and produce rounded head weights.

    Args:
        query: Writable, nonoverlapping GPU BF16 [tokens, heads, 128] projection.
            Only its trailing 64 dimensions are changed; strided rows are valid.
        head_weights: BF16 [tokens, heads] projection before scaling. Arbitrary
            read strides, including broadcast strides, are supported.
        positions: Integer [tokens] positions; negative sentinels use table row 0.
        cos_sin_cache: FP32 [max_position, 64] table with contiguous columns,
            containing 32 cosines followed by 32 sines per position.
        weight_scale: Scalar multiplier, converted to FP32 before multiplication.

    Returns:
        The original query tensor and fresh contiguous BF16 scaled weights.
        Scaling rounds to BF16 after FP32 multiplication.
    """
    _validate_index_query_inputs(query, head_weights, positions, cos_sin_cache)
    if any(
        size > 1 and stride == 0 for size, stride in zip(query.shape, query.stride())
    ):
        raise ValueError("Index query must not have broadcast writes")
    out = torch.empty(query.shape[:2], dtype=query.dtype, device=query.device)
    if query.shape[0]:
        _index_query_rope_weights[(query.shape[0], query.shape[1])](
            query,
            head_weights,
            positions,
            cos_sin_cache,
            out,
            *query.stride(),
            *head_weights.stride(),
            positions.stride(0),
            cos_sin_cache.stride(0),
            weight_scale,
            cos_sin_cache.shape[0],
            num_warps=1,
            enable_fp_fusion=False,
            debug=True,
        )
    return query, out


@triton.jit(do_not_specialize=["N", "CAP"])
def _index_key_norm_rope_scatter(
    X,
    W,
    POS,
    CS,
    CACHE,
    SLOTS,
    N,
    CAP,
    X0: tl.constexpr,
    X1: tl.constexpr,
    P0: tl.constexpr,
    CS0: tl.constexpr,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    S0: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MAX_POS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    values = (
        _normalized_index_keys(X, W, POS, CS, N, X0, X1, P0, CS0, EPS, R, MAX_POS, ROWS)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    groups = tl.reshape(values, (ROWS, 4, 32))
    amax = tl.max(tl.abs(groups), 2)
    # Match cache_scatter's multiplication and the materialized BF16 RoPE.
    scaled_max = tl.maximum(amax, 6 * 2.0**-126) * (1.0 / 6.0)
    packed, exponent = _mxfp4_index_rows(groups, scaled_max, ROWS)
    slots = tl.load(SLOTS + rows * S0, rows < N, -1).to(tl.int64)
    valid = (rows < N) & (slots >= 0) & (slots < CAP)
    base = CACHE + (slots // 64) * CP
    data_offset = (slots % 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    scale_offset = 4096 + (slots % 64)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(
        base[:, None] + _planar_offset(data_offset, CR, CB, 68),
        packed,
        valid[:, None],
    )
    tl.store(
        base[:, None] + _planar_offset(scale_offset, CR, CB, 68),
        exponent.to(tl.uint8),
        valid[:, None],
    )


@register_kernel(
    "attention",
    "dsv41_index_key_norm_rope_scatter",
    name="triton_dsv41_index_key_norm_rope_scatter",
    solution="triton",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def triton_dsv41_index_key_norm_rope_scatter(
    values, weight, positions, cos_sin_cache, eps, cache, slots
):
    """Fuse index-key RMSNorm, RoPE and page-planar MXFP4 storage.

    Args match dsv41.index_key_norm_rope_scatter. Returns None; only valid
    destination slots are modified, including with strided cache views.
    """
    _validate_index_key_inputs(values, weight, positions, cos_sin_cache, eps)
    _cache(cache, "index")
    _integers(slots, (values.shape[0],), "slots")
    _same_device(values, (cache, slots))
    if values.shape[0]:
        _index_key_norm_rope_scatter[(triton.cdiv(values.shape[0], 4),)](
            values,
            weight,
            positions,
            cos_sin_cache,
            cache,
            slots,
            values.shape[0],
            cache.shape[0] * 64,
            *values.stride(),
            positions.stride(0),
            cos_sin_cache.stride(0),
            *cache.stride(),
            slots.stride(0),
            eps,
            cos_sin_cache.shape[1],
            cos_sin_cache.shape[0],
            4,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )


@triton.jit(do_not_specialize=["N"])
def _index_query_rope_quant(
    Q,
    W,
    POS,
    CS,
    DATA,
    SCALES,
    WEIGHTS,
    N,
    H: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    P0: tl.constexpr,
    CS0: tl.constexpr,
    SCALE: tl.constexpr,
    MAX_POS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    token, head = rows // H, rows % H
    d = tl.arange(0, 128)
    q = tl.load(
        Q + token[:, None] * Q0 + head[:, None] * Q1 + d[None, :] * Q2,
        (rows < N)[:, None],
        0,
    ).to(tl.float32)
    partner = tl.gather(q, tl.broadcast_to((d ^ 1)[None, :], (ROWS, 128)), 1)
    position = tl.maximum(tl.load(POS + token * P0, rows < N, 0).to(tl.int64), 0)
    valid = position < MAX_POS
    tl.device_assert(
        valid | (rows >= N), "RoPE position is outside the cosine/sine table"
    )
    pair = (d - 64) // 2
    mask = (rows < N)[:, None] & valid[:, None] & (d >= 64)[None, :]
    cosine = tl.load(CS + position[:, None] * CS0 + pair[None, :], mask, 1.0)
    sine = tl.load(CS + position[:, None] * CS0 + 32 + pair[None, :], mask, 0.0)
    rotated = _rotate_interleaved(q, partner, cosine, sine, (d[None, :] & 1) != 0)
    q = tl.where((d >= 64)[None, :], rotated, q).to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(q, (ROWS, 4, 32))
    amax = tl.maximum(tl.max(tl.abs(groups), 2), 6 * 2.0**-126)
    packed, exponent = _mxfp4_index_rows(groups, tl.div_rn(amax, 6.0), ROWS)
    tl.store(
        DATA + rows[:, None] * 64 + tl.arange(0, 64)[None, :],
        packed,
        (rows < N)[:, None],
    )
    tl.store(SCALES + rows, _ue8m0_word(exponent), rows < N)
    weight = tl.load(W + token * W0 + head * W1, rows < N, 0).to(tl.float32)
    scaled = (weight * tl.full((), SCALE, tl.float32)).to(tl.bfloat16)
    tl.store(WEIGHTS + rows, scaled.to(tl.float32), rows < N)


@register_kernel(
    "attention",
    "dsv41_index_query_rope_quant",
    name="triton_dsv41_index_query_rope_quant",
    solution="triton",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def triton_dsv41_index_query_rope_quant(
    query, head_weights, positions, cos_sin_cache, weight_scale, output_dtype
):
    """Fuse index-Q RoPE, MXFP4 packing and BF16-rounded head-weight scaling.

    Args and returns match dsv41.index_query_rope_quant. Both FP32 Full and
    BF16 sparse head weights are produced directly in this same launch.
    """
    _validate_index_query_inputs(query, head_weights, positions, cos_sin_cache)
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Index weight output must be BF16 or FP32")
    data = torch.empty((*query.shape[:2], 64), dtype=torch.uint8, device=query.device)
    scales = torch.empty(query.shape[:2], dtype=torch.int32, device=query.device)
    weights = torch.empty(query.shape[:2], dtype=output_dtype, device=query.device)
    count = query.shape[0] * query.shape[1]
    if count:
        _index_query_rope_quant[(triton.cdiv(count, 4),)](
            query,
            head_weights,
            positions,
            cos_sin_cache,
            data,
            scales,
            weights,
            count,
            query.shape[1],
            *query.stride(),
            *head_weights.stride(),
            positions.stride(0),
            cos_sin_cache.stride(0),
            weight_scale,
            cos_sin_cache.shape[0],
            4,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
    return (data, scales), weights


@triton.jit(do_not_specialize=["NP"])
def _global_rope_scatter(
    X,
    POS,
    CS,
    CACHE,
    SLOTS,
    NP,
    X0: tl.constexpr,
    X1: tl.constexpr,
    POS0: tl.constexpr,
    CS0: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    S0: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    position = tl.maximum(tl.load(POS + token * POS0).to(tl.int64), 0)
    position_valid = position < MAX_POS
    tl.device_assert(
        position_valid, "Global RoPE position is outside the cosine/sine table"
    )
    slot = tl.load(SLOTS + token * S0).to(tl.int64)
    if slot >= 0 and slot < NP * 64:
        d = tl.arange(0, 512)
        x = tl.load(X + token * X0 + d * X1).to(tl.float32)
        partner = tl.gather(x, d ^ 1, axis=0)
        pair = (d - 448) // 2
        rope = d >= 448
        cosine = tl.load(CS + position * CS0 + pair, rope & position_valid, 1.0)
        sine = tl.load(CS + position * CS0 + 32 + pair, rope & position_valid, 0.0)
        # Match the BF16 materialization between RoPE and the existing codec.
        rotated = tl.where(
            rope,
            _rotate_interleaved(x, partner, cosine, sine, (d & 1) != 0).to(tl.bfloat16),
            x.to(tl.bfloat16),
        ).to(tl.float32)
        grouped = tl.reshape(rotated, (32, 16))
        amax = tl.max(tl.abs(grouped), axis=1)
        scale = _global_e4m3_scale(amax)
        scale_bytes = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        scaled = tl.reshape(tl.div_rn(grouped, scale[:, None]), (512,))
        lo, hi = tl.split(tl.reshape(_e2m1_encode(scaled), (256, 2)))
        base = CACHE + (slot // 64) * C0
        page_row = slot % 64
        payload = page_row * 256 + tl.arange(0, 256)
        tl.store(base + _planar_offset(payload, C1, C2, 288), lo | (hi << 4))
        scales = 64 * 256 + page_row * 32 + tl.arange(0, 32)
        tl.store(base + _planar_offset(scales, C1, C2, 288), scale_bytes)


@register_kernel(
    "attention",
    "dsv41_global_rope_scatter",
    name="triton_dsv41_global_rope_scatter",
    solution="triton",
    signatures=[format_signature(x=dense_tensor_format(torch.bfloat16))],
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def triton_dsv41_global_rope_scatter(
    values: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Rotate global keys, quantize to FP4, and write their page-planar cache.

    Args:
        values: BF16 [tokens, 512] keys before RoPE, preserved by the kernel.
        positions: Integer [tokens] positions; negative values select table row 0.
        cos_sin_cache: FP32 [max_position, 64] cosine then sine table, with
            contiguous columns. Only the trailing 64 key dimensions rotate.
        cache: Uint8 [pages, 64, 288] global cache, supporting strided bytes.
            Each page stores 64 payload rows followed by 64 E4M3 scale rows.
        slots: Integer [tokens] physical cache slots. Negative and out-of-range
            slots skip writes; valid write slots must be unique. As in the raw
            cache_scatter codec, slot zero is valid. Runtime null-page addresses
            must already be translated to -1 by the cache-slot resolver.

    Returns:
        None. Writes cache bytes, retaining the BF16 rounding between RoPE and
        global E2M1/E4M3 quantization. Inputs must be finite in the codec's range.
    """
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != torch.bfloat16:
        raise ValueError("Global keys must be BF16 [tokens, 512]")
    _cache(cache, "global")
    _same_device(values, (positions, cos_sin_cache, cache, slots))
    _integers(positions, values.shape[:1], "positions")
    _integers(slots, values.shape[:1], "slots")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.shape[0] == 0
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError(
            "Global RoPE table must be FP32 [positions, 64] with contiguous columns"
        )
    if any(
        size > 1 and stride == 0 for size, stride in zip(cache.shape, cache.stride())
    ):
        raise ValueError("Global cache must not have broadcast writes")
    if values.shape[0]:
        _global_rope_scatter[(values.shape[0],)](
            values,
            positions,
            cos_sin_cache,
            cache,
            slots,
            cache.shape[0],
            *values.stride(),
            positions.stride(0),
            cos_sin_cache.stride(0),
            *cache.stride(),
            slots.stride(0),
            cos_sin_cache.shape[0],
            num_warps=4 if values.shape[0] <= 512 else 1,
            enable_fp_fusion=False,
            debug=True,
        )
