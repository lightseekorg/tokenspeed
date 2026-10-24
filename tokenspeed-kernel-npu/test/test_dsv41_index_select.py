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


"""Independent exact selection checks, including graph updates and invalid pages."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_index_scan import _index_score_scan_kernel
from tokenspeed_kernel_npu.ops.dsv41_index_select import (
    block_scores_for_topk,
    select_full_scores,
)


@pytest.mark.parametrize("width", [64, 128, 256, 1024, 16384, 1048576])
def test_native_block_scores_preserve_maxima_topk_and_replay(width):
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(819)
    scores = torch.randn(6, width, device="npu", dtype=torch.bfloat16).float()
    scores[0] = -torch.inf
    scores[1, 1] = torch.nan
    scores[2, 7] = torch.inf
    scores[3] = 0
    scores[3, 1::2] = -0.0
    visible = torch.tensor(
        [0, 17, 63, width, width + 1, -1], device="npu", dtype=torch.int32
    )

    def check(actual):
        columns = torch.arange(width, device="npu")
        source = scores.masked_fill(columns[None, :] >= visible[:, None], -torch.inf)
        expected = source.reshape(6, -1, 8).amax(-1)
        blocks = torch.arange(width // 8, device="npu")[None, :].expand_as(expected)
        latest = blocks.masked_fill(~torch.isfinite(expected), -1).amax(-1)
        expected.masked_fill_(
            (blocks == latest[:, None]) & (latest[:, None] >= 0), torch.inf
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        k = min(2048, width // 8)
        torch.testing.assert_close(
            actual.topk(k).indices, expected.topk(k).indices, atol=0, rtol=0
        )

    check(block_scores_for_topk(scores, visible))
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = block_scores_for_topk(scores, visible)
    for length in (min(width, 12833), 1, width):
        scores.copy_(scores.roll(8, dims=1))
        visible.fill_(length)
        graph.replay()
        torch.npu.synchronize()
        check(output)


def test_block_scores_runtime_shapes_reuse_compilation():
    from tokenspeed_kernel_npu.ops.dsv41_index_select import _block_maxima
    from utils import assert_no_triton_compile

    torch.npu.set_device(0)
    scores = torch.ones((32, 16384), device="npu", dtype=torch.float32)
    visible = torch.full((32,), 64, device="npu", dtype=torch.int32)
    block_scores_for_topk(scores[:1, :64].contiguous(), visible[:1])
    with assert_no_triton_compile(_block_maxima):
        for n, width in ((2, 128), (6, 256), (16, 1024), (32, 16384)):
            visible.fill_(width)
            actual = block_scores_for_topk(scores[:n, :width].contiguous(), visible[:n])
            expected = torch.ones_like(actual)
            expected[:, -1] = torch.inf
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def expected(source, k, block_mode):
    if block_mode:
        grouped = torch.nn.functional.pad(
            source, (0, (-source.numel()) % 8), value=-torch.inf
        ).reshape(-1, 8)
        grouped = torch.where(grouped.isnan(), -torch.inf, grouped)
        source = grouped.amax(-1)
        # torch.amax keeps an arbitrary sign on equal zeros. The accelerator's
        # maximum prefers +0 where either operand is +0.
        positive_zero = (grouped.view(torch.int32) == 0).any(-1)
        source = torch.where((source == 0) & positive_zero, 0.0, source)
        valid = (source > -torch.inf).nonzero().flatten()
        if valid.numel():
            source[valid[-1]] = torch.inf
    bits = source.bfloat16().view(torch.int16).int() & 65535
    keys = torch.where((bits & 32768) != 0, bits ^ 65535, bits ^ 32768)
    valid = (source > -torch.inf).nonzero().flatten()
    selected = (
        valid[torch.argsort(keys[valid], descending=True, stable=True)[:k]]
        .sort()
        .values
    )
    return selected, source[selected]


def check(result, scores, lengths, capacity, row_k, block_k):
    for (values, ids), k, block in zip(result, (row_k, block_k), (False, True)):
        values, ids = values.cpu(), ids.cpu()
        for row, length in enumerate(lengths):
            selected, expected_values = expected(
                scores[row, : min(max(length, 0), capacity)].clone(), k, block
            )
            valid = values[row] > -torch.inf
            got = ids[row, valid].long()
            order = got.argsort()
            assert torch.equal(got[order], selected), (
                row,
                length,
                block,
                got[order],
                selected,
                values[row],
                scores[row, : min(max(length, 0), capacity)],
            )
            torch.testing.assert_close(
                values[row, valid][order], expected_values, atol=0, rtol=0
            )
            assert (ids[row, ~valid] == -1).all()


@pytest.mark.parametrize(
    "capacity,row_k,block_k", [(65, 32, 8), (4096, 512, 0), (65536, 512, 2048)]
)
def test_selection_reference_and_replay(capacity, row_k, block_k):
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(1823)
    storage = capacity + 128
    cpu = torch.randn(6, storage, dtype=torch.bfloat16).float()
    cpu[:, ::7] = -torch.inf
    cpu[1] = 0
    cpu[2, 0::2] = -0.0
    cpu[2, 1::2] = 0.0
    cpu[3] = -torch.inf
    cpu[4, -40:] = torch.nan
    cpu[5, ::13] = torch.inf
    scores = cpu.npu()
    lengths = [-5, 17, 511, capacity, capacity + 100, capacity - 1]
    visible = torch.tensor(lengths, dtype=torch.int32).npu()

    def launch():
        return select_full_scores(
            scores, visible, row_k, block_k, logical_capacity=capacity
        )

    check(launch(), cpu, lengths, capacity, row_k, block_k)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = launch()
    for lens in (
        [capacity] * 6,
        [0, 1, 127, 513, min(capacity, 51200), capacity + 1],
        lengths,
    ):
        cpu = cpu.roll(5, dims=1)
        scores.copy_(cpu)
        visible.copy_(torch.tensor(lens, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        check(result, cpu, lens, capacity, row_k, block_k)


@pytest.mark.parametrize("shards", [1, 2, 8])
def test_full_score_tile_preserves_bits(shards):
    torch.npu.set_device(0)
    torch.manual_seed(452)
    n, width, heads = 8, 4160, 32
    q = torch.randn(n, heads, 128, device="npu", dtype=torch.bfloat16)
    w = torch.randn(n, heads, device="npu", dtype=torch.bfloat16) * 0.1
    cache = torch.randn(68, 64, 128, device="npu", dtype=torch.bfloat16)
    table = torch.randint(1, 68, (n, width // 64), device="npu", dtype=torch.int32)
    table[:, 2] = 0
    table[:, 10] = -1
    visible = torch.tensor(
        [0, 1, 63, 64, 65, 257, width - 1, width + 19], device="npu", dtype=torch.int32
    )
    outputs = []
    for tile in (32, 64):
        out = torch.full((n, width), -torch.inf, device="npu")
        _index_score_scan_kernel[n, 16](
            q,
            w,
            cache,
            table,
            visible,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            w.stride(0),
            w.stride(1),
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            table.stride(0),
            table.stride(1),
            1,
            width,
            cache.shape[0],
            table.shape[1],
            heads,
            heads // shards,
            shards,
            heads,
            16,
            tile,
            tile,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        outputs.append(out.cpu())
    torch.testing.assert_close(*outputs, atol=0, rtol=0)


def test_topk_full_scan_reference_and_replay():
    from tokenspeed_kernel_npu.ops.dsv41_index_device import (
        topk_tile,
    )

    torch.npu.set_device(0)
    torch.manual_seed(2187)
    n, width, heads, topk, blockk = 192, 8192, 32, 512, 2048
    q = torch.randn(n, heads, 128, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(n, heads, device="npu", dtype=torch.bfloat16) * 0.1
    cache = torch.randn(130, 64, 128, device="npu", dtype=torch.bfloat16)
    table = torch.randint(1, 130, (n, width // 64), device="npu", dtype=torch.int32)
    table[:, 2] = 0
    table[:, 10] = -1
    visible = torch.full((n,), width, device="npu", dtype=torch.int32)

    def buffers():
        return (
            torch.empty(n, topk, device="npu", dtype=torch.int32),
            torch.empty(n, device="npu", dtype=torch.int32),
            torch.empty(n, blockk, device="npu", dtype=torch.int32),
            torch.empty(n, device="npu", dtype=torch.int32),
        )

    output = buffers()

    def launch():
        return topk_tile(
            q, weights, cache, table, visible, None, topk, blockk, 2048, 1, output
        )

    def reference():
        capacity = max(width, blockk * 8)
        scores = torch.empty(n, capacity, device="npu")
        _index_score_scan_kernel[n, 16](
            q,
            weights,
            cache,
            table,
            visible,
            scores,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            weights.stride(0),
            weights.stride(1),
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            table.stride(0),
            table.stride(1),
            1,
            capacity,
            cache.shape[0],
            table.shape[1],
            heads,
            heads,
            1,
            heads,
            16,
            32,
            32,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        scores = scores.cpu()
        ref = tuple(
            t.cpu().fill_(-1) if t.ndim == 2 else t.cpu().zero_() for t in buffers()
        )
        for row, length in enumerate(visible.cpu().tolist()):
            for block, k, dest, lengths in (
                (False, topk, ref[0], ref[1]),
                (True, blockk, ref[2], ref[3]),
            ):
                ids, _ = expected(
                    scores[row, : min(max(length, 0), width)].clone(), k, block
                )
                dest[row, : ids.numel()] = ids.int()
                lengths[row] = ids.numel()
        return ref

    launch()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for shift in (0, 257, width):
        visible.copy_(
            (torch.arange(n, device="npu", dtype=torch.int32) * 103 + shift)
            % (width + 1)
        )
        q.mul_(-1)
        graph.replay()
        ref = reference()
        torch.npu.synchronize()
        for actual, expected_tensor in zip(output, ref):
            torch.testing.assert_close(
                actual.cpu(), expected_tensor.cpu(), atol=0, rtol=0
            )


@pytest.mark.parametrize("distribution", ["ties", "wide_range"])
def test_native_keys_long_context_fallback_replay(distribution):
    """Replay across row/block encoding limits, with winners beyond each cap."""
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(9128)
    capacity = 131200
    cpu = torch.randn(6, capacity + 256, dtype=torch.bfloat16).float()
    if distribution == "ties":
        cpu[1].fill_(1)
        cpu[2].fill_(-1)
        cpu[3].fill_(-torch.inf)
        cpu[3, 60000:] = 3
    else:
        cpu[1] = torch.logspace(-35, 35, cpu.shape[1]).bfloat16().float()
        cpu[2] = -cpu[1].flip(0)
        cpu[3, ::13] = torch.inf
    cpu[:, ::23] = -torch.inf
    cpu[4, :2048] = torch.nan
    cpu[5].fill_(-torch.inf)
    # Large winners past each encoding limit must never be silently truncated.
    cpu[:5, 65536:65544] = 100
    cpu[:5, 131072:131080] = 200
    scores = cpu.npu()
    visible = torch.full((6,), capacity, device="npu", dtype=torch.int32)

    def launch():
        return select_full_scores(scores, visible, 512, 2048, logical_capacity=capacity)

    launch()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = launch()
    for lengths in (
        [51200] * 6,
        [65535, 65536, 65537, 131071, 131072, 131073],
        [capacity] * 6,
        [0, 1, 511, 2047, 51200, capacity],
    ):
        visible.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        check(result, cpu, lengths, capacity, 512, 2048)


def test_selection_keys_ignore_default_dtype():
    """Model dtype scopes must not round the exact FP32 ordering keys."""
    torch.npu.set_device(0)
    cpu = torch.ones((2, 4096), dtype=torch.float32)
    cpu[1, ::3] = -torch.inf
    scores = cpu.npu()
    visible = torch.full((2,), 4096, device="npu", dtype=torch.int32)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        result = select_full_scores(scores, visible, 512, 256, logical_capacity=4096)
    finally:
        torch.set_default_dtype(previous)
    check(result, cpu, [4096, 4096], 4096, 512, 256)
