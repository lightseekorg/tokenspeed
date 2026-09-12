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

"""GFX950/GFX1250 integration coverage for the standard block-split DSA cache."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from utils import is_cdna4, is_cdna5

if not (is_cdna4() or is_cdna5()):
    pytest.skip(
        "AMD GFX950 or GFX1250 is required for standard-cache Gluon DSA tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.kvcache.triton import (  # isort: skip
    index_k_block_split_scatter,
)

if is_cdna4():
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (  # isort: skip
        gluon_dsa_decode_topk_standard_gfx950 as _decode_topk,
        gluon_dsa_prefill_topk_standard_gfx950 as _prefill_topk,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (  # isort: skip
        _dsa_standard_decode_logits_kernel,
    )
else:
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (  # isort: skip
        gluon_dsa_decode_topk_standard_gfx1250 as _decode_topk,
        gluon_dsa_prefill_topk_standard_gfx1250 as _prefill_topk,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.standard_cache_logits import (  # isort: skip
        _dsa_standard_decode_logits_kernel,
    )

_DEVICE = "cuda"
_PAGE_SIZE = 64
_HEAD_DIM = 128
_TOPK = 512
_SOFTMAX_SCALE = _HEAD_DIM**-0.5


@dataclass(frozen=True)
class _DecodeCase:
    q_len: int
    heads: int
    q_dtype: torch.dtype
    weight_dtype: torch.dtype


_DECODE_CASES = (
    _DecodeCase(1, 32, torch.bfloat16, torch.bfloat16),
    _DecodeCase(2, 64, torch.bfloat16, torch.float32),
    _DecodeCase(3, 32, torch.float8_e4m3fn, torch.bfloat16),
    _DecodeCase(4, 64, torch.float8_e4m3fn, torch.float32),
    _DecodeCase(5, 32, torch.bfloat16, torch.float32),
    _DecodeCase(6, 64, torch.float8_e4m3fn, torch.bfloat16),
)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device=_DEVICE)
    generator.manual_seed(seed)
    return generator


def _prepared_query(
    tokens: int,
    heads: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    values = (
        torch.randn(
            (tokens, heads, _HEAD_DIM),
            device=_DEVICE,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.15
    )
    if dtype == torch.bfloat16:
        query = values.to(torch.bfloat16)
        return query, None, query.float()

    scales = (
        torch.rand(
            (tokens, heads), device=_DEVICE, dtype=torch.float32, generator=generator
        )
        * 0.02
        + 0.01
    )
    query = (values / scales[..., None]).clamp(-448.0, 448.0).to(dtype)
    return query, scales.contiguous(), query.float() * scales[..., None]


def _noncompact_weights(
    tokens: int,
    heads: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    row_stride = 128 + heads
    backing = torch.empty((tokens, row_stride), device=_DEVICE, dtype=dtype)
    weights = backing[:, 128:]
    values = torch.randn(
        (tokens, heads), device=_DEVICE, dtype=torch.float32, generator=generator
    )
    values[:, 0] = values[:, 0].abs() + 0.1
    values[:, 1] = -(values[:, 1].abs() + 0.1)
    weights.copy_(values.to(dtype))
    assert weights.stride() == (row_stride, 1)
    assert weights.storage_offset() == 128
    return weights


def _pack_standard_cache(
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the page-major FP8-plus-scale layout without scorer helpers."""
    assert keys.shape[0] % _PAGE_SIZE == 0
    num_slots = keys.shape[0]
    num_pages = num_slots // _PAGE_SIZE
    values = keys.float()
    scales = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    key_fp8 = (values / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    reference = key_fp8.float() * scales
    packed = torch.zeros((num_slots, 132), device=_DEVICE, dtype=torch.uint8)
    flat = packed.reshape(-1)
    page_bytes = _PAGE_SIZE * 132
    fp8_pages = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, _PAGE_SIZE, _HEAD_DIM),
        (page_bytes, _HEAD_DIM, 1),
    )
    scale_pages = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, _PAGE_SIZE, 1),
        (page_bytes // 4, 1, 1),
        (_PAGE_SIZE * _HEAD_DIM) // 4,
    )
    fp8_pages.copy_(key_fp8.reshape(num_pages, _PAGE_SIZE, _HEAD_DIM))
    scale_pages.copy_(scales.reshape(num_pages, _PAGE_SIZE, 1))
    return packed, reference, key_fp8, scales


def _weighted_relu_scores(
    query: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Independent FP32 oracle for the kernel's weighted per-head ReLU."""
    per_head = keys.float() @ query.float().transpose(0, 1)
    return (per_head.relu() * weights.float()).sum(dim=1) * _SOFTMAX_SCALE


def _expected_decode(
    query: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    q_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = torch.full(
        (query.shape[0], _TOPK), -1, device=_DEVICE, dtype=torch.int32
    )
    expected_lens = torch.empty(query.shape[0], device=_DEVICE, dtype=torch.int32)
    for row in range(query.shape[0]):
        request, q_offset = divmod(row, q_len)
        valid = int(seq_lens[request].item()) - (q_len - 1) + q_offset
        count = min(max(valid, 0), _TOPK)
        expected_lens[row] = count
        if count == 0:
            continue
        positions = torch.arange(valid, device=_DEVICE)
        pages = block_table[request, positions // _PAGE_SIZE].long()
        slots = pages * _PAGE_SIZE + positions.remainder(_PAGE_SIZE)
        scores = _weighted_relu_scores(query[row], weights[row], keys[slots])
        selected = torch.topk(scores, count).indices
        expected[row, :count] = slots[selected].to(torch.int32)
    return expected, expected_lens


def _expected_prefill(
    query: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = torch.full(
        (query.shape[0], _TOPK), -1, device=_DEVICE, dtype=torch.int32
    )
    expected_lens = torch.empty(query.shape[0], device=_DEVICE, dtype=torch.int32)
    for row in range(query.shape[0]):
        start = max(int(row_starts[row].item()), 0)
        end = min(int(row_ends[row].item()), workspace_slots.numel())
        count = min(max(end - start, 0), _TOPK)
        expected_lens[row] = count
        if count == 0:
            continue
        workspace_rows = torch.arange(start, end, device=_DEVICE)
        slots = workspace_slots[workspace_rows].long()
        scores = _weighted_relu_scores(query[row], weights[row], keys[slots])
        selected = torch.topk(scores, count).indices
        expected[row, :count] = workspace_rows[selected].to(torch.int32)
    return expected, expected_lens


def _assert_topk(
    actual: torch.Tensor,
    actual_lens: torch.Tensor,
    expected: torch.Tensor,
    expected_lens: torch.Tensor,
) -> None:
    torch.testing.assert_close(actual_lens.cpu(), expected_lens.cpu())
    for row, count_tensor in enumerate(expected_lens):
        count = int(count_tensor.item())
        torch.testing.assert_close(
            torch.sort(actual[row, :count].cpu()).values,
            torch.sort(expected[row, :count].cpu()).values,
        )
        assert (actual[row, count:] == -1).all()


@pytest.mark.parametrize(
    "case",
    _DECODE_CASES,
    ids=lambda case: (
        f"q{case.q_len}-h{case.heads}-"
        f"{str(case.q_dtype).removeprefix('torch.')}-"
        f"w{str(case.weight_dtype).removeprefix('torch.')}"
    ),
)
def test_standard_cache_decode_matches_weighted_relu_oracle(case: _DecodeCase) -> None:
    generator = _generator(100 + case.q_len)
    seq_lens = torch.tensor((515, 641), device=_DEVICE, dtype=torch.int32)
    page_count = 11
    block_table = torch.tensor(
        (
            (17, 3, 20, 1, 14, 7, 10, 5, 21, 0, 12),
            (9, 18, 2, 16, 6, 13, 4, 19, 8, 15, 11),
        ),
        device=_DEVICE,
        dtype=torch.int32,
    )
    assert block_table.shape == (2, page_count)
    keys = (
        torch.randn(
            (block_table.numel() * _PAGE_SIZE, _HEAD_DIM),
            device=_DEVICE,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.15
    )
    cache, key_reference, _, _ = _pack_standard_cache(keys)
    query, q_scales, query_reference = _prepared_query(
        2 * case.q_len, case.heads, case.q_dtype, generator
    )
    weights = _noncompact_weights(
        2 * case.q_len, case.heads, case.weight_dtype, generator
    )
    if case.heads == 64:
        weights[:, :32].zero_()

    actual, actual_lens = _decode_topk(
        query,
        weights,
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        q_len_per_req=case.q_len,
        index_k_cache=cache,
        q_scales=q_scales,
    )
    expected, expected_lens = _expected_decode(
        query_reference,
        weights,
        key_reference,
        seq_lens,
        block_table,
        case.q_len,
    )

    _assert_topk(actual, actual_lens, expected, expected_lens)


@pytest.mark.parametrize(
    ("heads", "q_dtype", "weight_dtype"),
    (
        (32, torch.bfloat16, torch.float32),
        (64, torch.float8_e4m3fn, torch.bfloat16),
    ),
)
def test_standard_cache_prefill_uses_workspace_rows_not_global_slots(
    heads: int,
    q_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> None:
    generator = _generator(200 + heads)
    num_slots = 22 * _PAGE_SIZE
    keys = (
        torch.randn(
            (num_slots, _HEAD_DIM),
            device=_DEVICE,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.15
    )
    cache, key_reference, _, _ = _pack_standard_cache(keys)
    workspace_rows = 704
    workspace_slots = (
        torch.arange(workspace_rows, device=_DEVICE) * 37 + 41
    ) % num_slots
    workspace_slots[96:160] = workspace_slots[11:75]
    workspace_slots = workspace_slots.to(torch.int64).contiguous()
    assert not torch.equal(
        workspace_slots, torch.arange(workspace_rows, device=_DEVICE)
    )
    assert workspace_slots.unique().numel() < workspace_slots.numel()
    row_starts = torch.tensor((7, 130, 260, 0, 530), device=_DEVICE, dtype=torch.int32)
    row_ends = torch.tensor(
        (535, 642, 600, 503, 530), device=_DEVICE, dtype=torch.int32
    )
    query, q_scales, query_reference = _prepared_query(
        row_starts.numel(), heads, q_dtype, generator
    )
    weights = _noncompact_weights(row_starts.numel(), heads, weight_dtype, generator)

    actual, actual_lens = _prefill_topk(
        query,
        weights,
        workspace_slots,
        row_starts,
        row_ends,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        index_k_cache=cache,
        page_size=_PAGE_SIZE,
        q_scales=q_scales,
        max_logits_bytes=workspace_rows * 4 * 2,
    )
    expected, expected_lens = _expected_prefill(
        query_reference,
        weights,
        key_reference,
        workspace_slots,
        row_starts,
        row_ends,
    )

    _assert_topk(actual, actual_lens, expected, expected_lens)
    assert (actual[-1] == -1).all()


def test_standard_cache_decode_accepts_block_split_writer_output() -> None:
    generator = _generator(301)
    num_slots = 11 * _PAGE_SIZE
    source_keys = (
        torch.randn(
            (num_slots, _HEAD_DIM),
            device=_DEVICE,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.15
    )
    _, source_reference, key_fp8, key_scales = _pack_standard_cache(source_keys)
    cache = torch.zeros((num_slots, 132), device=_DEVICE, dtype=torch.uint8)
    locations = torch.randperm(num_slots, device=_DEVICE, generator=generator)
    index_k_block_split_scatter(
        cache,
        key_fp8,
        key_scales,
        locations,
        page_size=_PAGE_SIZE,
        head_dim=_HEAD_DIM,
        group_size=_HEAD_DIM,
    )
    key_reference = torch.empty_like(source_reference)
    key_reference[locations] = source_reference
    seq_lens = torch.tensor((515,), device=_DEVICE, dtype=torch.int32)
    block_table = torch.randperm(11, device=_DEVICE, generator=generator).reshape(1, 11)
    block_table = block_table.to(torch.int32)
    query, _, query_reference = _prepared_query(1, 32, torch.bfloat16, generator)
    weights = _noncompact_weights(1, 32, torch.float32, generator)

    actual, actual_lens = _decode_topk(
        query,
        weights,
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        index_k_cache=cache,
    )
    expected, expected_lens = _expected_decode(
        query_reference,
        weights,
        key_reference,
        seq_lens,
        block_table,
        1,
    )

    _assert_topk(actual, actual_lens, expected, expected_lens)


def test_standard_cache_decode_returns_empty_result_for_zero_pages() -> None:
    generator = _generator(349)
    query, _, _ = _prepared_query(1, 32, torch.bfloat16, generator)
    weights = _noncompact_weights(1, 32, torch.float32, generator)
    seq_lens = torch.zeros((1,), device=_DEVICE, dtype=torch.int32)
    block_table = torch.empty((1, 0), device=_DEVICE, dtype=torch.int32)
    cache = torch.empty((0, 132), device=_DEVICE, dtype=torch.uint8)
    out = torch.zeros((1, _TOPK), device=_DEVICE, dtype=torch.int32)
    lens_out = torch.full((1,), -1, device=_DEVICE, dtype=torch.int32)

    actual, actual_lens = _decode_topk(
        query,
        weights,
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        index_k_cache=cache,
        out=out,
        lens_out=lens_out,
    )

    assert actual.data_ptr() == out.data_ptr()
    assert actual_lens.data_ptr() == lens_out.data_ptr()
    assert (actual == -1).all()
    assert (actual_lens == 0).all()


def test_standard_cache_decode_logits_cover_empty_and_short_spans() -> None:
    generator = _generator(350)
    seq_lens = torch.tensor((0, 1, 63, 64, 65), device=_DEVICE, dtype=torch.int32)
    block_table = torch.tensor(
        ((0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        device=_DEVICE,
        dtype=torch.int32,
    )
    keys = torch.randn(
        (2 * _PAGE_SIZE, _HEAD_DIM),
        device=_DEVICE,
        dtype=torch.float32,
        generator=generator,
    )
    cache, key_reference, _, _ = _pack_standard_cache(keys)
    query, _, query_reference = _prepared_query(
        seq_lens.numel(), 32, torch.bfloat16, generator
    )
    weights = _noncompact_weights(seq_lens.numel(), 32, torch.float32, generator)
    logits = torch.full(
        (seq_lens.numel(), 2 * _PAGE_SIZE),
        float("nan"),
        device=_DEVICE,
        dtype=torch.float32,
    )

    _dsa_standard_decode_logits_kernel[(seq_lens.numel(), 1)](
        query,
        weights,
        cache.view(torch.float8_e4m3fn),
        cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        weights.stride(0),
        weights.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        _SOFTMAX_SCALE,
        logits.shape[1],
        1,
        PAGE_SIZE=_PAGE_SIZE,
        ROW_BYTES=132,
        PAGE_STRIDE_BYTES=_PAGE_SIZE * cache.stride(0),
        NUM_HEADS=32,
        HEAD_DIM=_HEAD_DIM,
        BLOCK_N=64,
        CHUNK_N=256,
        NUM_WARPS=2,
        Q_IS_FP8=False,
        USE_BUFFER_LOAD=True,
        USE_BUFFER_STORE=True,
        num_warps=2,
        waves_per_eu=4,
    )
    torch.cuda.synchronize()

    assert logits[0].isnan().all()
    for row, length_tensor in enumerate(seq_lens[1:], start=1):
        length = int(length_tensor.item())
        positions = torch.arange(length, device=_DEVICE)
        pages = block_table[row, positions // _PAGE_SIZE].long()
        slots = pages * _PAGE_SIZE + positions.remainder(_PAGE_SIZE)
        expected = _weighted_relu_scores(
            query_reference[row], weights[row], key_reference[slots]
        )
        torch.testing.assert_close(
            logits[row, :length], expected, rtol=2.0e-2, atol=2.0e-2
        )
        assert logits[row, length:].isnan().all()


@pytest.mark.parametrize("q_len", (1, 2, 3, 4, 5, 6))
def test_standard_cache_decode_cuda_graph_replays_changed_inputs(q_len: int) -> None:
    generator = _generator(400 + q_len)
    heads = 32 if q_len % 2 else 64
    q_dtype = torch.bfloat16 if q_len in (1, 2, 5) else torch.float8_e4m3fn
    weight_dtype = torch.bfloat16 if q_len in (1, 3, 6) else torch.float32
    seq_lens = torch.tensor((515,), device=_DEVICE, dtype=torch.int32)
    block_table = torch.randperm(11, device=_DEVICE, generator=generator).reshape(1, 11)
    block_table = block_table.to(torch.int32)
    keys = (
        torch.randn(
            (11 * _PAGE_SIZE, _HEAD_DIM),
            device=_DEVICE,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.15
    )
    cache, _, _, _ = _pack_standard_cache(keys)
    query, q_scales, _ = _prepared_query(q_len, heads, q_dtype, generator)
    weights = _noncompact_weights(q_len, heads, weight_dtype, generator)
    out = torch.empty((q_len, _TOPK), device=_DEVICE, dtype=torch.int32)
    lens_out = torch.empty((q_len,), device=_DEVICE, dtype=torch.int32)

    def invoke() -> None:
        _decode_topk(
            query,
            weights,
            seq_lens,
            block_table,
            page_size=_PAGE_SIZE,
            topk=_TOPK,
            softmax_scale=_SOFTMAX_SCALE,
            q_len_per_req=q_len,
            index_k_cache=cache,
            q_scales=q_scales,
            out=out,
            lens_out=lens_out,
        )

    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()

    replacement_query, replacement_scales, replacement_reference = _prepared_query(
        q_len, heads, q_dtype, _generator(500 + q_len)
    )
    replacement_weights = _noncompact_weights(
        q_len, heads, weight_dtype, _generator(600 + q_len)
    )
    replacement_keys = (
        torch.randn(
            (11 * _PAGE_SIZE, _HEAD_DIM),
            device=_DEVICE,
            dtype=torch.float32,
            generator=_generator(700 + q_len),
        )
        * 0.15
    )
    replacement_cache, replacement_key_reference, _, _ = _pack_standard_cache(
        replacement_keys
    )
    query.copy_(replacement_query)
    weights.copy_(replacement_weights)
    if q_scales is not None:
        assert replacement_scales is not None
        q_scales.copy_(replacement_scales)
    cache.copy_(replacement_cache)
    seq_lens.fill_(509)
    block_table.copy_(torch.roll(block_table, shifts=3, dims=1))
    graph.replay()
    torch.cuda.synchronize()
    expected, expected_lens = _expected_decode(
        replacement_reference,
        weights,
        replacement_key_reference,
        seq_lens,
        block_table,
        q_len,
    )

    _assert_topk(out, lens_out, expected, expected_lens)
