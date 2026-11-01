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

"""Fused RoPE preserves BF16 rounding and the packed KV cache layout."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv41 import dsv41_key_norm_rope
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import (
    index_key_norm_rope,
    index_query_rope_weights,
)
from tokenspeed_kernel.ops.attention.dsv41.triton import pack_index_inputs, rope_inplace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _key_inputs(tokens, rotary_dim, strided):
    torch.manual_seed(321)
    width = 256 if strided else 128
    values = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
    if strided:
        values = values[:, 1::2]
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens * 2, device="cuda", dtype=torch.int64)[::2] - 3
    angles = (
        torch.arange(16384, device="cuda")[:, None].float()
        * torch.exp(-torch.arange(rotary_dim // 2, device="cuda").float() / 4)[None, :]
    )
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return values, weight, positions, cache


def _key_reference(values, weight, positions, cache, eps, round_normalized):
    return dsv41_key_norm_rope(
        values, weight, positions, cache, eps, round_normalized=round_normalized
    )


@pytest.mark.parametrize(
    "tokens,rotary_dim,strided",
    [
        (0, 64, False),
        (1, 64, False),
        (7, 128, True),
        (129, 64, True),
        (8192, 64, False),
    ],
)
def test_index_key_norm_rope_reference(tokens, rotary_dim, strided):
    values, weight, positions, cache = _key_inputs(tokens, rotary_dim, strided)
    before = values.clone()
    actual = index_key_norm_rope(values, weight, positions, cache, 1e-6)
    expected = _key_reference(values, weight, positions, cache, 1e-6, True)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=8e-3)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    if tokens:
        error = (actual.float() - expected.float()).square().mean().sqrt()
        assert error < 1e-4


def test_index_key_norm_rope_retains_intermediate_bf16_rounding():
    values, weight, positions, cache = _key_inputs(7, 64, False)
    rounded = _key_reference(values, weight, positions, cache, 1e-6, True)
    unrounded = _key_reference(values, weight, positions, cache, 1e-6, False)
    assert torch.count_nonzero(rounded != unrounded) > 50
    actual = index_key_norm_rope(values, weight, positions, cache, 1e-6)
    torch.testing.assert_close(actual, rounded, rtol=0, atol=0)


def test_index_key_norm_rope_graph_refresh():
    values, weight, positions, cache = _key_inputs(17, 64, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        index_key_norm_rope(values, weight, positions, cache, 1e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = index_key_norm_rope(values, weight, positions, cache, 1e-5)
    torch.cuda.current_stream().wait_stream(stream)
    for shift in (127, -512, 2048):
        values.normal_()
        weight.normal_()
        positions.add_(shift)
        expected = _key_reference(values, weight, positions, cache, 1e-5, True)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=8e-3)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return torch.device("cuda:0")


def _table(device):
    angle = (
        torch.arange(257, device=device).float()[:, None]
        * torch.linspace(0.0001, 1, 32, device=device)[None, :]
    )
    return torch.cat((angle.cos(), angle.sin()), dim=-1)


def _assert_bits(actual, expected):
    dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    torch.testing.assert_close(actual.view(dtype), expected.view(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [0, 1, 3, 33, 257])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_index_query_rope_quant_matches_separate_packing(tokens, strided, output_dtype):
    torch.manual_seed(881)
    q = torch.randn(tokens, 32, 256 if strided else 128, device="cuda").bfloat16()
    if strided:
        q = q[..., 1::2]
    weights = torch.randn(1 if strided else tokens, 32, device="cuda").bfloat16()
    if strided:
        weights = weights.expand(tokens, -1)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32) - 1
    table = _table("cuda")
    before = q.clone(), weights.clone()
    rotated = rope_inplace(q.contiguous().clone(), positions, table)
    packed, expected_weights = pack_index_inputs(
        rotated, weights * 0.023456789, output_dtype
    )
    actual, actual_weights = dsv41.index_query_rope_quant(
        q, weights, positions, table, 0.023456789, output_dtype
    )
    for got, expected in zip(actual, packed, strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    _assert_bits(actual_weights, expected_weights)
    _assert_bits(q, before[0])
    _assert_bits(weights, before[1])


@pytest.mark.parametrize(
    "tokens,rotary_dim,page_stride",
    [
        (0, 64, 4608),
        (1, 64, 4608),
        (2, 64, 4608),  # Both slots are invalid.
        (17, 32, 4608),
        (17, 64, 23040),
        (17, 128, 71680),
        (257, 64, 71680),
    ],
)
def test_index_key_norm_rope_scatter_matches_separate_cache_write(
    tokens, rotary_dim, page_stride
):
    values, weight, positions, table = _key_inputs(tokens, rotary_dim, True)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64) + 64
    if tokens > 1:
        slots[0] = -1
        slots[-1] = 8 * 64
    actual = torch.full((8, page_stride), 0xA7, device="cuda", dtype=torch.uint8)
    expected = actual.clone()
    cache = actual.as_strided((8, 64, 68), (page_stride, 68, 1))
    reference_cache = expected.as_strided((8, 64, 68), (page_stride, 68, 1))
    normalized = index_key_norm_rope(values, weight, positions, table, 1e-6)
    dsv41.cache_scatter(normalized, reference_cache, slots, "index")
    before = values.clone()
    dsv41.index_key_norm_rope_scatter(
        values, weight, positions, table, 1e-6, cache, slots
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_bits(values, before)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_fused_index_preprocessing_graph_refresh(output_dtype):
    values, norm_weight, positions, table = _key_inputs(17, 64, True)
    q = torch.randn(17, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(17, 32, device="cuda", dtype=torch.bfloat16)
    storage = torch.zeros(8, 23040, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((8, 64, 68), (23040, 68, 1))
    slots = torch.arange(17, device="cuda", dtype=torch.int64) + 64

    def run():
        dsv41.index_key_norm_rope_scatter(
            values, norm_weight, positions, table, 1e-6, cache, slots
        )
        return dsv41.index_query_rope_quant(
            q, weights, positions, table, 0.023456789, output_dtype
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual, actual_weights = run()
    torch.cuda.current_stream().wait_stream(stream)
    for offset in (0, 127, 3072):
        values.normal_()
        norm_weight.normal_()
        q.normal_()
        weights.normal_()
        positions.copy_(torch.arange(17, device="cuda") + offset)
        slots.copy_(torch.arange(17, device="cuda") + 64 + offset % 128)
        slots[0], slots[-1] = -1, 8 * 64
        storage.fill_(0xA7)
        reference = storage.clone()
        normalized = index_key_norm_rope(values, norm_weight, positions, table, 1e-6)
        dsv41.cache_scatter(
            normalized,
            reference.as_strided(cache.shape, cache.stride()),
            slots,
            "index",
        )
        packed, expected_weights = pack_index_inputs(
            rope_inplace(q.clone(), positions, table),
            weights * 0.023456789,
            output_dtype,
        )
        graph.replay()
        torch.testing.assert_close(storage, reference, rtol=0, atol=0)
        for got, expected in zip(actual, packed, strict=True):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        _assert_bits(actual_weights, expected_weights)


@pytest.mark.parametrize(
    "tokens,heads,layout,scale",
    [
        (0, 32, "contiguous", 0.023456789),
        (1, 4, "contiguous", 128**-0.5 * 32**-0.5),
        (3, 7, "broadcast_weights", 0.023456789),
        (33, 32, "strided", 0.023456789),
        (257, 32, "contiguous", 0.023456789),
    ],
)
def test_index_query_rope_weights_exact(device, tokens, heads, layout, scale):
    torch.manual_seed(413)
    q_storage = torch.randn((tokens, heads * 2, 256), device=device).bfloat16()
    query = q_storage[:, ::2, 1::2]
    w_storage = torch.randn((tokens, heads * 2), device=device).bfloat16()
    weights = w_storage[:, 1::2]
    if layout == "contiguous":
        query = query.contiguous()
        weights = weights.contiguous()
    elif layout == "broadcast_weights":
        weights = torch.randn((heads,), device=device).bfloat16().expand(tokens, -1)
    positions = torch.arange(tokens * 2, device=device, dtype=torch.int64)[::2] % 257
    if tokens:
        positions[0] = -1
        if tokens > 1:
            positions[-1] = 256
    cache_storage = torch.empty((257, 128), device=device)
    cache = cache_storage[:, :64]
    cache.copy_(_table(device))
    q_before, w_before = query.clone(), weights.clone()
    full_before = q_storage.clone()
    expected_q = rope_inplace(query.contiguous().clone(), positions, cache)
    expected_weights = weights * scale
    q_result, output = index_query_rope_weights(query, weights, positions, cache, scale)
    assert q_result is query
    _assert_bits(query, expected_q)
    _assert_bits(output, expected_weights)
    _assert_bits(query[..., :64], q_before[..., :64])
    _assert_bits(weights, w_before)
    if layout != "contiguous":
        _assert_bits(q_storage[:, 1::2], full_before[:, 1::2])
        _assert_bits(q_storage[:, ::2, ::2], full_before[:, ::2, ::2])


def test_index_query_rope_weights_graph_replay(device):
    torch.manual_seed(414)
    tokens, heads = 19, 32
    query = torch.randn((tokens, heads, 128), device=device).bfloat16()
    weights = torch.randn((tokens, heads), device=device).bfloat16()
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    table = _table(device)
    scale = 0.023456789
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        index_query_rope_weights(query, weights, positions, table, scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _, output = index_query_rope_weights(
                query, weights, positions, table, scale
            )
    torch.cuda.current_stream().wait_stream(stream)
    for offset in (0, 17, 101):
        query.normal_()
        weights.normal_()
        positions.copy_(torch.arange(tokens, device=device, dtype=torch.int32) + offset)
        positions[0] = -1
        expected_q = rope_inplace(query.clone(), positions, table)
        expected_weights = weights * scale
        graph.replay()
        _assert_bits(query, expected_q)
        _assert_bits(output, expected_weights)


def test_index_weight_rounding_edges(device):
    # Cover subnormal weights, signed zeros, saturation and ordinary BF16
    # values against Torch's BF16 scalar multiplication, independent of RoPE.
    raw = torch.tensor(
        [
            0.0,
            -0.0,
            2.0**-133,
            -(2.0**-133),
            2.0**-126,
            0.5,
            -0.5,
            3.3895313892515355e38,
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    weights = raw.repeat(4)[None, :]
    query = torch.zeros((1, 32, 128), device=device, dtype=torch.bfloat16)
    positions = torch.zeros(1, device=device, dtype=torch.int64)
    for scale in (0.015625, 0.3333333333, 1.0, 4.0):
        _, output = index_query_rope_weights(
            query, weights, positions, _table(device), scale
        )
        _assert_bits(output, weights * scale)


def _global_inputs(tokens, strided, mode):
    torch.manual_seed(43)
    values = torch.randn(
        tokens, 1024 if strided else 512, device="cuda", dtype=torch.bfloat16
    )
    if strided:
        values = values[:, 1::2]
    if mode == "tiny":
        values.mul_(1e-6)
    elif mode == "large":
        values.clamp_(-2, 2).mul_(512)
    elif mode == "ties":
        ties = torch.tensor(
            [
                -6,
                -5,
                -3.5,
                -2.5,
                -1.75,
                -1.25,
                -0.75,
                -0.25,
                -0.0,
                0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                3.5,
                6,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        values.copy_(ties.repeat(32)[None, :])
    elif mode == "zero":
        values.zero_()
        values[:, ::2] = -0.0
    positions = torch.randint(-8, 32768, (tokens * 2,), device="cuda")[::2]
    if mode == "ties":
        positions.zero_()
    angles = (
        torch.arange(32768, device="cuda")[:, None].float()
        * torch.exp(-torch.arange(32, device="cuda").float() / 4)[None, :]
    )
    rope_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    pages = (2 * tokens + 63) // 64 + 2
    slots = torch.arange(tokens * 2, device="cuda")[::2] + 64
    if tokens > 1:
        slots[::11] = -1
    slots[2::13] = pages * 64
    slots[3::19] = -64
    return values, positions, rope_cache, slots, pages


def _global_cache(pages, strided):
    if strided:
        storage = torch.full((pages, 65, 600), 165, device="cuda", dtype=torch.uint8)
        return storage, storage[:, :64, :576:2]
    storage = torch.full((pages, 64, 288), 165, device="cuda", dtype=torch.uint8)
    return storage, storage


def _global_reference(values, positions, rope_cache, cache, slots):
    # contiguous() may retain unusual strides on an empty tensor.
    rotated = torch.empty(values.shape, device=values.device, dtype=values.dtype)
    rotated.copy_(values)
    dsv41.rope_inplace(rotated, positions, rope_cache, None)
    dsv41.cache_scatter(rotated, cache, slots, "global")


@pytest.mark.parametrize(
    "tokens,strided,mode",
    [
        (0, False, "normal"),
        (1, False, "normal"),
        (4, True, "ties"),
        (129, True, "tiny"),
        (2048, False, "large"),
        (8192, True, "zero"),
    ],
)
def test_global_rope_scatter_matches_materialized_path(tokens, strided, mode):
    values, positions, rope_cache, slots, pages = _global_inputs(tokens, strided, mode)
    original = values.clone()
    reference_storage, reference_cache = _global_cache(pages, strided)
    actual_storage, actual_cache = _global_cache(pages, strided)
    _global_reference(values, positions, rope_cache, reference_cache, slots)
    dsv41.global_rope_scatter(values, positions, rope_cache, actual_cache, slots)
    torch.testing.assert_close(actual_storage, reference_storage, rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    # Runtime null-page addresses reach the codec as negative slots.
    assert torch.all(actual_cache[0] == 165)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_global_rope_scatter_graph_refresh_and_raw_slot_zero(strided, dtype):
    values, positions, rope_cache, slots, pages = _global_inputs(17, strided, "normal")
    positions = positions.to(dtype)
    slots = slots.to(dtype)
    actual_storage, actual_cache = _global_cache(pages, strided)
    reference_storage, reference_cache = _global_cache(pages, strided)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dsv41.global_rope_scatter(values, positions, rope_cache, actual_cache, slots)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            dsv41.global_rope_scatter(
                values, positions, rope_cache, actual_cache, slots
            )
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (0, 17, 123):
        torch.manual_seed(seed)
        values.normal_()
        positions.random_(-8, 32768)
        positions[0] = -1
        positions[1] = -8
        slots.copy_(torch.randperm(pages * 64, device="cuda")[:17])
        # Raw codec slot zero is valid, matching generic cache_scatter; runtime
        # cache_slots translates reserved null pages to -1 before this boundary.
        slots[slots == 0] = -1
        slots[0] = 0
        slots[1] = -1
        slots[2] = pages * 64
        actual_storage.fill_(165)
        reference_storage.fill_(165)
        _global_reference(values, positions, rope_cache, reference_cache, slots)
        graph.replay()
        torch.testing.assert_close(actual_storage, reference_storage, rtol=0, atol=0)


def test_fused_rope_launches_reuse_compilation_across_token_counts():
    from tokenspeed_kernel.ops.attention.dsv41._triton import rope
    from utils import assert_no_triton_compile

    def keys(tokens):
        values, weight, positions, cache = _key_inputs(tokens, 64, True)
        index_key_norm_rope(values, weight, positions, cache, 1e-6)

    def global_keys(tokens):
        values, positions, rope_cache, slots, pages = _global_inputs(
            tokens, True, "normal"
        )
        _, cache = _global_cache(pages, True)
        dsv41.global_rope_scatter(values, positions, rope_cache, cache, slots)

    keys(17)
    global_keys(17)
    # Token counts and cache page counts change every batch; neither may key
    # a new binary. (Above 512 tokens the global launch uses fewer warps.)
    with assert_no_triton_compile(rope._index_key_norm_rope):
        with assert_no_triton_compile(rope._global_rope_scatter):
            for tokens in (1, 3, 64, 129, 500):
                keys(tokens)
                global_keys(tokens)
