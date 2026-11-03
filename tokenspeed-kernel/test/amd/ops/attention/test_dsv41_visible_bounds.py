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

"""Raw indexer visibility is bounded by consumers before int32 conversion."""

from contextlib import ExitStack

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import triton as implementation
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _index_block_select_kernel,
    index_block_select,
)
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires admitted CUDA/ROCm"
)


def _raw(rows, device):
    values = torch.tensor(
        [-(2**63), -1, 0, 1, 9, 127, 128, 129, 2**32 + 1, 2**63 - 1],
        dtype=torch.int64,
        device=device,
    )
    return values.repeat((rows + 9) // 10)[:rows]


def _fixture(rows, device):
    q = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    weights = torch.randn(rows, 4, device=device, dtype=torch.bfloat16)
    cache = torch.zeros(3, 64, 68, dtype=torch.uint8, device=device)
    keys = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
    dsv41.cache_scatter(keys, cache, torch.arange(64, 192, device=device), "index")
    table = torch.tensor([1, 2], device=device, dtype=torch.int32)[None, :].repeat(
        rows, 1
    )
    return q, weights, cache, table


def _run(fixture, visible, reindex, make_blocks):
    q, weights, cache, table = fixture
    candidates = (
        torch.tensor([0, 3, 5, 11, 15, -1], device=q.device, dtype=torch.int32)[
            None, :
        ].repeat(q.shape[0], 1)
        if reindex
        else None
    )
    return dsv41.index_topk(
        q,
        weights,
        cache,
        table,
        visible,
        candidates,
        16,
        4 if make_blocks else 0,
        8,
        256,
        64,
        None,
        None,
        "gluon",
    )


@requires_gpu
@pytest.mark.parametrize("rows", [6, 12, 24, 48, 96, 192, 257])
@pytest.mark.parametrize(
    "reindex,make_blocks", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_gluon_raw_visibility_matches_preclamped_inputs(
    rows, reindex, make_blocks, strided
):
    if not (current_platform().is_cdna4 or current_platform().is_cdna5):
        pytest.skip("AMD CSA2 indexer")
    torch.manual_seed(541)
    fixture = _fixture(rows, "cuda")
    raw = _raw(rows, "cuda")
    if strided:
        raw = torch.empty(rows * 2, dtype=raw.dtype, device=raw.device)[::2].copy_(raw)
    expected = _run(fixture, raw.clamp(0, 128), reindex, make_blocks)
    actual = _run(fixture, raw, reindex, make_blocks)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("width,capacity", [(32, 128), (128, 128), (131200, 131200)])
def test_block_selector_bounds_single_and_split_passes(width, capacity):
    raw = _raw(10, "cuda")
    logits = torch.randn(10, width, device="cuda")
    columns = torch.arange(width, device="cuda")[None, :]
    logits.masked_fill_(columns >= raw.clamp(0, capacity)[:, None], -torch.inf)
    expected_out = torch.empty(10, 4, dtype=torch.int32, device="cuda")
    expected_lengths = torch.empty(10, dtype=torch.int32, device="cuda")
    actual_out = torch.empty_like(expected_out)
    actual_lengths = torch.empty_like(expected_lengths)
    index_block_select(
        logits,
        raw.clamp(0, capacity),
        4,
        expected_out,
        expected_lengths,
        visible_capacity=capacity,
    )
    index_block_select(
        logits, raw, 4, actual_out, actual_lengths, visible_capacity=capacity
    )
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)


@requires_gpu
def test_visibility_replay_refresh_and_no_capacity_specialization():
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 scorer and selector")
    fixture = _fixture(12, "cuda")
    visible = _raw(12, "cuda")
    _run(fixture, visible, False, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run(fixture, visible, False, True)
    for value in (128, 0, -1, 2**32 + 1, 2**63 - 1):
        visible.fill_(value)
        expected = _run(fixture, visible.clamp(0, 128), False, True)
        graph.replay()
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    logits = torch.randn(12, 128, device="cuda")
    out = torch.empty(12, 4, dtype=torch.int32, device="cuda")
    lens = torch.empty(12, dtype=torch.int32, device="cuda")
    index_block_select(logits, visible, 4, out, lens, visible_capacity=128)
    with assert_no_triton_compile(_index_block_select_kernel):
        for capacity in (1, 7, 129, 256):
            index_block_select(logits, visible, 4, out, lens, visible_capacity=capacity)


@requires_gpu
@pytest.mark.parametrize("width", [128, 131200])
def test_block_visibility_capacity_does_not_recompile(width):
    logits = torch.randn(6, width, device="cuda")
    visible = _raw(6, "cuda")
    out = torch.empty(6, 4, dtype=torch.int32, device="cuda")
    lens = torch.empty(6, dtype=torch.int32, device="cuda")
    index_block_select(logits, visible, 4, out, lens, visible_capacity=width)
    kernels = (
        implementation._index_block_select_kernel,
        implementation._block_keys_pass,
        implementation._block_hist_pass,
        implementation._block_count_pass,
        implementation._block_emit_pass,
    )
    with ExitStack() as stack:
        for kernel in kernels:
            stack.enter_context(assert_no_triton_compile(kernel))
        for capacity in (1, 7, 129, width + 1):
            index_block_select(logits, visible, 4, out, lens, visible_capacity=capacity)


@requires_gpu
@pytest.mark.parametrize("rows", [1, 6, 12])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_row_selector_live_width_strides_match_dense_without_recompile(rows, dtype):
    if current_platform().is_cdna4:
        from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41.indexer import (
            gluon_dsv41_index_topk_select_gfx950 as kernel,
        )
        from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41.indexer import (
            launch_gluon_dsv41_index_topk_select_gfx950 as launch,
        )
    elif current_platform().is_cdna5:
        from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41.topk import (
            gluon_dsv41_index_topk_select_gfx1250 as kernel,
        )
        from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41.topk import (
            launch_gluon_dsv41_index_topk_select_gfx1250 as launch,
        )
    else:
        pytest.skip("AMD CSA2 row selector")

    width, topk = 128, 16
    limits = torch.arange(rows, device="cuda", dtype=dtype).mul_(9).clamp_(0, width)
    maximum = 2**32 + 1 if dtype == torch.int64 else width
    limits[-1] = maximum
    scores = torch.arange(width, 0, -1, device="cuda", dtype=torch.float32)[
        None, :
    ].repeat(rows, 1)
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    lengths = torch.empty(rows, device="cuda", dtype=torch.int32)
    expected = torch.empty_like(out)
    expected_lengths = torch.empty_like(lengths)
    # Warm the same tensor dtypes and static selector configuration.
    launch(scores, None, limits.clamp(0, width), topk, out, lengths)
    for stride in (0, 1, 2, 7):
        if stride == 0:
            strided = torch.full((1,), maximum, device="cuda", dtype=dtype).expand(rows)
        else:
            strided = torch.empty(rows * stride, device="cuda", dtype=dtype)[::stride]
            strided.copy_(limits)
        logits = scores.masked_fill(
            torch.arange(width, device="cuda")[None, :] >= strided[:, None], -torch.inf
        )
        launch(logits, None, strided.clamp(0, width), topk, expected, expected_lengths)
        with assert_no_triton_compile(kernel):
            launch(logits, None, strided, topk, out, lengths)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)


@requires_gpu
def test_block_selector_table_widths_and_query_rows_reuse_binary():
    visible = torch.full((24,), 2**32 + 1, dtype=torch.int64, device="cuda")
    out = torch.empty(24, 4, dtype=torch.int32, device="cuda")
    lens = torch.empty(24, dtype=torch.int32, device="cuda")
    logits = torch.randn(24, 8192, device="cuda")
    # Real table capacities are multiples of 64; preserve their existing
    # alignment information while varying the runtime dimensions and bound.
    index_block_select(
        logits[:6, :128].contiguous(),
        visible[:6],
        4,
        out[:6],
        lens[:6],
        visible_capacity=128,
    )
    with assert_no_triton_compile(_index_block_select_kernel):
        for rows, width in ((12, 256), (24, 8192)):
            scores = logits[:rows, :width].contiguous()
            index_block_select(
                scores,
                visible[:rows],
                4,
                out[:rows],
                lens[:rows],
                visible_capacity=width,
            )
    assert (lens == 4).all()
