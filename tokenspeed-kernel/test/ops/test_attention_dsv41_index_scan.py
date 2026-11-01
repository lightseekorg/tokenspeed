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

import os
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
from test_attention_dsv41 import device as device
from tokenspeed_kernel.numerics.torch.attention.dsv41 import (
    torch_dsv41_cache_quantize,
    torch_dsv41_gather_candidates,
    torch_dsv41_index_cache_gather,
    torch_dsv41_index_scores,
)
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import triton as implementation
from tokenspeed_kernel.ops.attention.dsv41._deep_gemm.indexer import gather_index_pages
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    pack_index_inputs,
    pack_index_queries,
    prepare_candidate_scores,
)


def _inputs(device, dtype):
    torch.manual_seed(51)
    q = torch.randint(-12, 13, (3, 32, 256), device=device).bfloat16().div_(8)[..., ::2]
    w = (
        torch.tensor(
            [
                0.3125,
                -0.203125,
                0.703125,
                -0.59375,
                1.09375,
                -1.296875,
                0.40625,
                -0.90625,
            ],
            device=device,
            dtype=dtype,
        )
        .repeat(4)
        .expand(3, -1)
    )
    keys = torch.randint(-12, 13, (8192, 128), device=device).bfloat16().div_(8)
    cache = torch.zeros((128, 64, 136), device=device, dtype=torch.uint8)[..., ::2]
    dsv41.cache_scatter(keys, cache, torch.arange(8192, device=device), "index")
    table = torch.empty((3, 256), dtype=torch.int32, device=device)[:, ::2]
    table.copy_(torch.arange(127, -1, -1, device=device))
    table[:, 2:4] = torch.tensor([-1, 999999], device=device)
    return q, w, keys, cache, table, torch.zeros(3, dtype=torch.int32, device=device)


def _check(ids, lengths, scores, rtol, atol):
    assert ids.dtype == lengths.dtype == torch.int32
    for selected, length, score in zip(ids.cpu(), lengths.cpu(), scores, strict=True):
        n = min(selected.numel(), int((score > -torch.inf).sum()))
        assert int(length) == n and (selected[n:] == -1).all()
        chosen = selected[:n].long()
        assert ((chosen >= 0) & (chosen < score.numel())).all()
        assert (chosen[1:] > chosen[:-1]).all()
        torch.testing.assert_close(
            score[chosen].sort().values,
            score.topk(n).values.sort().values,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_index_gather_heads_packed_once(dtype):
    q = torch.arange(3 * 2 * 256).reshape(3, 2, 256).bfloat16()[..., ::2]
    w = torch.arange(-6, 6, dtype=dtype).reshape(3, 4)[:, ::2]
    packed = [
        torch.cat(((q + rank).to(dtype), (w - rank)[..., None]), -1)
        for rank in range(4)
    ]
    group = object()

    def gather(output, tensor, group):
        assert group is sentinel
        torch.testing.assert_close(tensor, packed[0], rtol=0, atol=0)
        output.copy_(torch.cat(packed, dim=0))

    sentinel = group
    with patch.object(
        torch.distributed, "get_world_size", return_value=4
    ), patch.object(
        torch.distributed, "all_gather_single", side_effect=gather
    ) as collective, patch.object(
        torch.distributed, "all_reduce", side_effect=AssertionError
    ):
        got_q, got_w, shards = implementation._index_gather_heads(q, w, group)
        assert shards == 4 and collective.call_count == 1
        torch.testing.assert_close(
            got_q,
            torch.cat([p[..., :128] for p in packed], 1).bfloat16(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            got_w, torch.cat([p[..., 128] for p in packed], 1), rtol=0, atol=0
        )
        local_q, local_w, shards = implementation._index_gather_heads(q, w, None)
        assert (
            local_q is q and local_w is w and shards == 1 and collective.call_count == 1
        )


def test_index_scan_capacity_launches_and_scratch(device):
    q, w, _, cache, table, visible = _inputs(device, torch.bfloat16)
    visible.copy_(torch.tensor([65, 513, 0], device=device))
    results, launches, scratch = [], [], []
    for capacity in (8192, 1048576):
        pages = torch.full((3, capacity // 32), -1, dtype=torch.int32, device=device)[
            :, ::2
        ]
        pages[:, :128].copy_(table)

        def run():
            full = dsv41.index_topk(
                q,
                w,
                cache,
                pages,
                visible,
                None,
                65,
                17,
                8,
                2,
                64,
                None,
                None,
                solution="triton",
            )
            return full + dsv41.index_topk(
                q,
                w,
                cache,
                pages,
                visible,
                full[2],
                65,
                0,
                8,
                2,
                64,
                None,
                None,
                solution="triton",
            )

        run()
        torch.cuda.synchronize()
        with patch.object(
            implementation,
            "_index_finish_parts",
            wraps=implementation._index_finish_parts,
        ) as finish:
            results.append(run())
        scratch.append([tuple(call.args[0].shape) for call in finish.call_args_list])
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            run()
            torch.cuda.synchronize()
        launches.append(
            sum(
                event.device_type == torch.autograd.DeviceType.CUDA
                for event in prof.events()
            )
        )
    assert launches[0] > 0 and launches[0] == launches[1]
    assert (
        scratch[0]
        == scratch[1]
        == [
            (2, 16 * 128),
            (2, 16 * 32),
            (1, 16 * 128),
            (1, 16 * 32),
            (2, 16 * 128),
            (1, 16 * 128),
        ]
    )
    for small, large in zip(*results, strict=True):
        torch.testing.assert_close(small, large, rtol=0, atol=0)


@pytest.fixture(scope="module")
def tp_group():
    if (
        os.environ.get("TOKENSPEED_TEST_TP4") != "1"
        or os.environ.get("WORLD_SIZE") != "4"
    ):
        yield None
        return
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.cuda.set_device(device)
    # One communicator for both dtypes: reinitializing a destroyed default
    # group can reuse stale rendezvous-store keys between unsynchronized ranks.
    torch.distributed.init_process_group(
        "nccl", timeout=timedelta(seconds=120), device_id=device
    )
    try:
        yield torch.distributed.group.WORLD
    finally:
        torch.distributed.destroy_process_group()


def run_index_scan_graph_oracle(
    device, shards, dtype, tp_group, require, solution, rtol, atol
):
    require("attention", "dsv41_index_topk", solution, torch.bfloat16, "x")
    if shards == 4 and (
        os.environ.get("TOKENSPEED_TEST_TP4") != "1"
        or os.environ.get("WORLD_SIZE") != "4"
    ):
        pytest.skip("opt in with TOKENSPEED_TEST_TP4=1 and torchrun --nproc_per_node=4")
    rank = int(os.environ["RANK"]) if shards == 4 else 0
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if shards == 4 else device
    torch.cuda.set_device(device)
    group = tp_group if shards == 4 else None
    graph = None
    try:
        q, w, keys, cache, table, visible = _inputs(device, dtype)
        physical_scores = torch_dsv41_index_scores(q, w, keys, shards)
        local_q, local_w = q.chunk(shards, dim=1)[rank], w.chunk(shards, dim=1)[rank]
        slots = torch.arange(64, device=device).expand(3, -1)
        scores = dsv41.index_score(local_q, local_w, cache, slots, group, None)
        torch.testing.assert_close(
            scores.cpu().float(), physical_scores[:, :64], rtol=0, atol=0
        )

        def run():
            full = dsv41.index_topk(
                local_q,
                local_w,
                cache,
                table,
                visible,
                None,
                65,
                17,
                8,
                2,
                64,
                group,
                None,
                solution=solution,
            )
            return full + dsv41.index_topk(
                local_q,
                local_w,
                cache,
                table,
                visible,
                full[2],
                65,
                0,
                8,
                2,
                64,
                group,
                None,
                solution=solution,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
        torch.cuda.current_stream().wait_stream(stream)
        for lengths in (
            [1, 7, 0],
            [63, 64, 65],
            [511, 512, 513],
            [4095, 4096, 4097],
            [8192, 129, 0],
            [0, 0, 0],
        ):
            q.add_(0.125)
            w[0].neg_()
            physical_scores = torch_dsv41_index_scores(q, w, keys, shards)
            visible.copy_(torch.tensor(lengths, device=device))
            table.copy_(table.flip(1))
            with patch.object(
                torch.distributed,
                "all_gather_single",
                wraps=torch.distributed.all_gather_single,
            ) as gather, patch.object(
                torch.distributed, "all_reduce", side_effect=AssertionError
            ):
                eager = run()
                assert gather.call_count == (4 if shards == 4 else 0)
                if shards == 4:
                    assert [call.args[1].shape for call in gather.call_args_list] == [
                        (2, 8, 129),
                        (1, 8, 129),
                    ] * 2
                graph.replay()
            for got, want in zip(captured, eager, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            logical = torch.arange(8192).expand(3, -1)
            pages = table.cpu().gather(1, logical // 64).long()
            scores = physical_scores.gather(
                1, (pages * 64 + logical % 64).clamp(0, 8191)
            )
            scores.masked_fill_(
                (logical >= visible.cpu()[:, None]) | (pages < 0) | (pages >= 128),
                -torch.inf,
            )
            _check(eager[0], eager[1], scores, rtol=rtol, atol=atol)
            blocks = scores.reshape(3, -1, 8).amax(dim=-1)
            for t, length in enumerate(lengths):
                if length and blocks[t, (length - 1) // 8] > -torch.inf:
                    blocks[t, (length - 1) // 8] = torch.inf
            _check(eager[2], eager[3], blocks, rtol=rtol, atol=atol)
            for t in range(3):
                scores[t].masked_fill_(
                    ~torch.isin(logical[t] // 8, eager[2][t].cpu()), -torch.inf
                )
            _check(eager[4], eager[5], scores, rtol=rtol, atol=atol)
            assert eager[6].shape == (3, 0) and not eager[7].any()
            if shards == 4:
                packed = torch.cat([tensor.flatten() for tensor in captured])
                others = [torch.empty_like(packed) for _ in range(4)]
                torch.distributed.all_gather(others, packed, group=group)
                for other in others:
                    torch.testing.assert_close(other, packed, rtol=0, atol=0)
    finally:
        # NCCL destruction waits for captured graph references to be released.
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("shards", [1, 4], ids=["local", "tp4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_index_scan_graph_oracle(device, shards, dtype, tp_group, require):
    run_index_scan_graph_oracle(
        device, shards, dtype, tp_group, require, "triton", rtol=0, atol=0
    )


def run_index_topk_full_and_reindex(
    device, solution, heads, candidate_topk, topk, require, match_triton=True
):
    require("attention", "dsv41_index_topk", solution, torch.bfloat16, "x")
    torch.manual_seed(17)
    visible = (200, 64, 0)
    tokens, pages = len(visible), 4
    q = torch.randn((tokens, heads, 128), dtype=torch.bfloat16, device=device)
    w = torch.randn((tokens, heads), dtype=torch.bfloat16, device=device)
    keys = torch.randn((pages * 64, 128), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((pages, 64, 68), dtype=torch.uint8, device=device)
    dsv41.cache_scatter(keys, cache, torch.arange(pages * 64, device=device), "index")
    table = torch.arange(pages, dtype=torch.int32, device=device).expand(tokens, -1)
    lens = torch.tensor(visible, dtype=torch.int32, device=device)
    kwargs = dict(
        index_q=q,
        weights=w,
        index_cache=cache,
        page_table=table,
        visible_lens=lens,
        candidate_blocks=None,
        topk=topk,
        candidate_topk=candidate_topk,
        candidate_block_size=8,
        query_chunk_size=2,
        score_chunk_size=64,
        process_group=None,
        out=None,
        solution=solution,
    )
    got = dsv41.index_topk(**kwargs)
    if match_triton:
        want = implementation.index_topk(
            q,
            w,
            cache,
            table,
            lens,
            None,
            topk,
            candidate_topk,
            8,
            2,
            64,
            None,
            None,
            index_metadata=None,
        )
        for actual, expected in zip(got, want, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        physical = torch_dsv41_index_scores(q, w, keys, 1)
        logical = torch.arange(pages * 64)
        pages_idx = table.cpu().gather(1, logical.expand(tokens, -1) // 64).long()
        scores = physical.gather(1, pages_idx * 64 + logical % 64)
        scores.masked_fill_(
            (logical.expand(tokens, -1) >= lens.cpu()[:, None]) | (pages_idx < 0),
            -torch.inf,
        )
        _check(got[0], got[1], scores, rtol=2e-2, atol=2e-2)
        if candidate_topk:
            blocks = scores.reshape(tokens, -1, 8).amax(dim=-1)
            for t, length in enumerate(visible):
                if length and blocks[t, (length - 1) // 8] > -torch.inf:
                    blocks[t, (length - 1) // 8] = torch.inf
            _check(got[2], got[3], blocks, rtol=2e-2, atol=2e-2)
    if candidate_topk:
        kwargs["candidate_blocks"] = got[2]
        kwargs["candidate_topk"] = 0
        reindex = dsv41.index_topk(**kwargs)
        if match_triton:
            want_reindex = implementation.index_topk(
                q,
                w,
                cache,
                table,
                lens,
                got[2],
                topk,
                0,
                8,
                2,
                64,
                None,
                None,
                index_metadata=None,
            )
            for actual, expected in zip(reindex, want_reindex, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            logical = torch.arange(pages * 64)
            physical = torch_dsv41_index_scores(q, w, keys, 1)
            pages_idx = table.cpu().gather(1, logical.expand(tokens, -1) // 64).long()
            scores = physical.gather(1, pages_idx * 64 + logical % 64)
            scores.masked_fill_(
                (logical.expand(tokens, -1) >= lens.cpu()[:, None]) | (pages_idx < 0),
                -torch.inf,
            )
            for t in range(tokens):
                scores[t].masked_fill_(
                    ~torch.isin(logical // 8, got[2][t].cpu()), -torch.inf
                )
            _check(reindex[0], reindex[1], scores, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("heads", [1, 8, 32])
@pytest.mark.parametrize("candidate_topk", [0, 17])
@pytest.mark.parametrize("topk", [65, 512])
def test_index_topk_full_and_reindex(device, heads, candidate_topk, topk, require):
    run_index_topk_full_and_reindex(
        device, "triton", heads, candidate_topk, topk, require
    )


def test_native_indexer_page_contract_and_graph_lengths(device):
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import (
        is_native_indexer_available,
    )
    from tokenspeed_kernel.thirdparty.deep_select import is_deep_select_available

    if (
        torch.cuda.get_device_capability(device)[0] != 10
        or not is_native_indexer_available()
        or not is_deep_select_available()
    ):
        pytest.skip("requires Blackwell DeepGEMM and DeepSelect")
    torch.manual_seed(8841)
    q = torch.randn(1, 32, 128, device=device, dtype=torch.bfloat16)
    weights = -torch.ones(1, 32, device=device, dtype=torch.bfloat16)
    cache = torch.zeros((3, 64, 68), device=device, dtype=torch.uint8)
    keys = torch.randn(192, 128, device=device, dtype=torch.bfloat16)
    dsv41.cache_scatter(keys, cache, torch.arange(192, device=device), "index")
    # The public operator permits page zero. The LCM caller alone maps its
    # reserved null page to -1; absent pages must not outrank negative scores.
    table = torch.tensor([[0, -1, 2, -1]], device=device, dtype=torch.int32)
    visible = torch.tensor([256], device=device, dtype=torch.int32)

    def run():
        return dsv41.index_topk(
            q, weights, cache, table, visible, None, 16, 32, 8, 1, 64, None, None, None
        )

    output = run()
    assert output[1].item() == 16
    assert not ((output[0] >= 64) & (output[0] < 128)).any()
    assert (output[2] // 8 != 3).all()  # no pinned candidate in missing newest page
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for length in (0, 17, 192, 256):
        visible.fill_(length)
        graph.replay()
        expected = run()
        for got, want in zip(captured, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    # Noncontiguous page bytes use the portable reader with the same byte codec.
    strided = torch.zeros((3, 64, 136), device=device, dtype=torch.uint8)[..., ::2]
    dsv41.cache_scatter(keys, strided, torch.arange(192, device=device), "index")
    result = dsv41.index_topk(
        q, weights, strided, table, visible, None, 16, 0, 8, 1, 64, None, None, None
    )
    assert result[1].item() == 16
    assert not ((result[0] >= 64) & (result[0] < 128)).any()


def test_native_broadcast_history_matches_paged_and_refreshes_graph(device):
    from tokenspeed_kernel.ops.attention.dsv41.deep_gemm import (
        is_native_indexer_available,
    )
    from tokenspeed_kernel.thirdparty.deep_select import is_deep_select_available

    if (
        torch.cuda.get_device_capability(device)[0] != 10
        or not is_native_indexer_available()
        or not is_deep_select_available()
    ):
        pytest.skip("requires Blackwell packed native indexer")
    torch.manual_seed(421)
    q = torch.randn(33, 32, 128, device=device, dtype=torch.bfloat16)
    weights = -torch.rand(33, 32, device=device, dtype=torch.bfloat16)
    cache = torch.zeros(6, 64, 68, device=device, dtype=torch.uint8)
    dsv41.cache_scatter(
        torch.randn(384, 128, device=device, dtype=torch.bfloat16),
        cache,
        torch.arange(384, device=device),
        "index",
    )
    table = torch.tensor([[1, 2, -1, 4]], device=device, dtype=torch.int32).expand(
        33, -1
    )
    lengths = torch.arange(33, device=device, dtype=torch.int32) * 8

    def run(pages):
        return dsv41.index_topk(
            q,
            weights,
            cache,
            pages,
            lengths,
            None,
            512,
            64,
            8,
            1024,
            256,
            None,
            None,
            None,
        )

    dense, paged = run(table), run(table.clone())
    for actual, expected in zip(dense, paged, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(table)
    lengths.fill_(193)
    graph.replay()
    for actual, expected in zip(captured, run(table), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "tokens,heads,layout",
    [
        (0, 32, "contiguous"),
        (1, 1, "contiguous"),
        (5, 7, "broadcast"),
        (16, 32, "strided"),
        (65, 32, "contiguous"),
        (129, 32, "strided"),
    ],
)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_index_pack_matches_independent_fp4_oracle(
    device, tokens, heads, layout, weight_dtype, output_dtype
):
    torch.manual_seed(617)
    q = torch.randn(tokens * 2, heads * 2, 256, device=device).bfloat16()[
        ::2, ::2, 1::2
    ]
    weights = torch.randn(tokens * 2, heads * 2, device=device, dtype=weight_dtype)[
        ::2, ::2
    ]
    if layout == "contiguous":
        q, weights = q.contiguous(), weights.contiguous()
    elif layout == "broadcast":
        q = torch.randn(1, 1, 128, device=device).bfloat16().expand(tokens, heads, -1)
        weights = torch.randn(1, 1, device=device, dtype=weight_dtype).expand(
            tokens, heads
        )
    q_before, weights_before = q.clone(), weights.clone()
    (data, scales), output_weights = pack_index_inputs(q, weights, output_dtype)
    assert (
        data.is_contiguous()
        and scales.is_contiguous()
        and output_weights.is_contiguous()
    )
    packed = torch.cat(
        (data.cpu().reshape(-1, 64), scales.cpu().view(torch.uint8).reshape(-1, 4)), 1
    )
    expected, _ = torch_dsv41_cache_quantize(q, "index")
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output_weights,
        weights.to(output_dtype).contiguous(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(q, q_before, rtol=0, atol=0)
    torch.testing.assert_close(weights, weights_before, rtol=0, atol=0)
    if weights.dtype == output_dtype and weights.is_contiguous():
        assert output_weights is weights


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_index_pack_large_inner_stride(device, weight_dtype):
    stride = 1 << 25
    elements = 191 * stride + 1
    required_bytes = elements * torch.bfloat16.itemsize
    if torch.cuda.mem_get_info(device)[0] < required_bytes + (256 << 20):
        pytest.skip("requires space for a large-stride query view")
    storage = torch.empty(elements, dtype=torch.bfloat16, device=device)
    # Keep wrapped int32 reads in bounds so a regression fails by value.
    storage.as_strided((192,), (stride,)).fill_(-17)
    q = storage.as_strided((1, 1, 128), (0, 0, stride), storage_offset=64 * stride)
    q.copy_(torch.arange(128, dtype=torch.bfloat16, device=device).view(1, 1, 128))
    weights = torch.ones(1, 1, dtype=weight_dtype, device=device)
    (data, scales), output_weights = pack_index_inputs(q, weights, weight_dtype)
    expected, _ = torch_dsv41_cache_quantize(q.contiguous(), "index")
    packed = torch.cat(
        (data.cpu().reshape(-1, 64), scales.cpu().view(torch.uint8).reshape(-1, 4)), 1
    )
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    assert output_weights is weights


@pytest.mark.parametrize("width", [128, 131200])
@pytest.mark.parametrize("rows", [1, 3])
def test_block_select_visible_stride_contract(device, width, rows):
    logits = torch.randn(rows, width, device=device)
    visible = torch.full((2 * rows,), width, dtype=torch.int64, device=device)[::2]
    out = torch.empty(rows, 4, dtype=torch.int32, device=device)
    lengths = torch.empty(rows, dtype=torch.int32, device=device)
    if rows > 1:
        with pytest.raises(ValueError, match="contiguous visible row counts"):
            implementation.index_block_select(
                logits, visible, 4, out, lengths, visible_capacity=width
            )
        return
    # A single element has no stride-dependent addressing, in either selector.
    expected_out = torch.empty_like(out)
    expected_lengths = torch.empty_like(lengths)
    implementation.index_block_select(
        logits, visible, 4, out, lengths, visible_capacity=width
    )
    implementation.index_block_select(
        logits,
        visible.clone(),
        4,
        expected_out,
        expected_lengths,
        visible_capacity=width,
    )
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)


def test_index_pack_boundaries_and_graph_refresh(device):
    q = torch.zeros(8, 32, 128, device=device, dtype=torch.bfloat16)
    weights = torch.zeros(8, 32, device=device, dtype=torch.bfloat16)
    pack_index_inputs(q, weights, torch.float32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (data, scales), fp32_weights = pack_index_inputs(q, weights, torch.float32)
    for seed in range(3):
        torch.manual_seed(seed)
        q.normal_()
        q[0].zero_()
        q[1].fill_(6 * 2.0**-126)
        q[2].fill_(2.0**-127)
        q[3].fill_(-(2.0**-127))
        midpoints = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6], device=device
        )
        q[4].copy_(midpoints.repeat(16).expand(32, -1))
        q[5].copy_(-q[4])
        q[6].fill_(-0.0)
        weights.normal_()
        weights[0].fill_(-0.0)
        graph.replay()
        old_data, old_scales = pack_index_queries(q)
        torch.testing.assert_close(data, old_data, rtol=0, atol=0)
        torch.testing.assert_close(scales, old_scales, rtol=0, atol=0)
        torch.testing.assert_close(
            fp32_weights.view(torch.int32),
            weights.float().view(torch.int32),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("target", ["cpu", "cuda"])
@pytest.mark.parametrize("candidate_capacity", [0, 64, 2048])
def test_select_all_causal_pages_and_replay(target, candidate_capacity):
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    # Include a strided table, a partial final block, missing/invalid pages,
    # and a padded query. Invalid entries must compact into a sorted prefix.
    table = torch.full((5, 16), -1, dtype=torch.int32, device=target)[:, ::2]
    table.copy_(torch.tensor([1, 0, -1, 99, 2, 3, 4, 5], device=target))
    lengths = torch.zeros(10, dtype=torch.int32, device=target)[::2]
    out = tuple(
        torch.empty(shape, dtype=torch.int32, device=target)
        for shape in ((5, 512), (5,), (5, candidate_capacity), (5,))
    )

    def run():
        return dsv41.index_select_all(table, lengths, 6, 8, out)

    run()
    graph = None
    if target == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    for visible in ([0, 1, 129, 512, 0], [512, 257, 9, 63, 0]):
        lengths.copy_(torch.tensor(visible, device=target))
        table[4].fill_(-1)
        if graph is None:
            run()
        else:
            graph.replay()
        for row, length in enumerate(visible):
            pages = table[row].cpu().tolist()
            wanted = [i for i in range(length) if 0 < pages[i // 64] < 6]
            blocks = sorted({i // 8 for i in wanted}) if candidate_capacity else []
            assert out[1][row].item() == len(wanted)
            assert out[3][row].item() == len(blocks)
            assert out[0][row].cpu().tolist() == wanted + [-1] * (512 - len(wanted))
            assert out[2][row].cpu().tolist() == blocks + [-1] * (
                candidate_capacity - len(blocks)
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_select_all_tile_slices_reuse_compilation():
    from utils import assert_no_triton_compile

    # Prefill tiles pass row slices of shared buffers; the slice start must
    # not select a compilation through the pointers' alignment.
    table = torch.arange(1, 9, dtype=torch.int32, device="cuda").repeat(9, 1)
    lengths = torch.full((9,), 300, dtype=torch.int32, device="cuda")
    out = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in ((9, 512), (9,), (9, 64), (9,))
    )

    def run(start, stop):
        dsv41.index_select_all(
            table[start:stop],
            lengths[start:stop],
            9,
            8,
            tuple(x[start:stop] for x in out),
        )

    run(0, 4)
    with assert_no_triton_compile(implementation._index_select_all):
        for start, stop in ((1, 4), (3, 9), (5, 6), (2, 7)):
            run(start, stop)
    torch.testing.assert_close(out[1], lengths)


@pytest.mark.parametrize(
    "count,block_size,alignment,sorted_ids",
    [
        (1, 1, 4, False),
        (17, 3, 16, False),
        (128, 8, 64, True),
        (512, 8, 64, False),
        (513, 8, 64, False),
        (512, 8, 8192, True),
        (2048, 8, 64, True),
    ],
)
def test_candidate_scores_causal_duplicates_and_strides(
    device, count, block_size, alignment, sorted_ids
):
    torch.manual_seed(923)
    logits = torch.randn(5, 2302, device=device)[:, ::2]
    lengths = torch.tensor([-5, 0, 17, 1150, 2000], device=device, dtype=torch.int64)
    candidates = torch.randint(-4, 180, (5, count * 2), device=device)[:, ::2]
    if sorted_ids:
        candidates.copy_(
            candidates.masked_fill(candidates < 0, 2**31 - 1).sort().values
        )
        candidates.masked_fill_(candidates == 2**31 - 1, -1)
    expected = torch_dsv41_gather_candidates(
        logits, lengths, candidates, block_size, alignment
    )
    actual = prepare_candidate_scores(
        logits, lengths, candidates, block_size, alignment
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


def test_candidate_scores_graph_refresh(device):
    torch.manual_seed(924)
    logits = torch.randn(3, 4096, device=device)
    lengths = torch.zeros(3, device=device, dtype=torch.int32)
    candidates = torch.full((3, 512), -1, device=device, dtype=torch.int32)
    prepare_candidate_scores(logits, lengths, candidates, 8, 64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = prepare_candidate_scores(logits, lengths, candidates, 8, 64)
    for step in range(4):
        lengths.copy_(torch.tensor([0, 7, 4096], device=device) - step)
        candidates.random_(-5, 512)
        logits.normal_()
        graph.replay()
        expected = torch_dsv41_gather_candidates(logits, lengths, candidates, 8, 64)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


def _gather_inputs(capacity, strided, dtype):
    torch.manual_seed(731 + capacity)
    pages = capacity // 64 + 2
    storage = torch.randint(
        0, 256, (pages, 8704 if strided else 4352), device="cuda", dtype=torch.uint8
    )
    cache = storage[:, :4352]
    table = torch.randperm(pages, device="cuda")[: capacity // 64].to(dtype)
    if strided:
        backing = torch.empty(table.numel() * 2, device="cuda", dtype=dtype)
        backing[1::2] = table
        table = backing[1::2]
    if table.numel():
        table[0] = 0
    if table.numel() > 2:
        table[1::17] = -1
        table[2::31] = pages + 123
    return cache, table


@pytest.mark.parametrize(
    "capacity,strided,dtype",
    [
        (64, False, torch.int32),
        (8192, True, torch.int64),
        (1048576, False, torch.int64),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_prefill_gather_matches_slot_materialization(capacity, strided, dtype):
    cache, pages = _gather_inputs(capacity, strided, dtype)
    gather_index_pages(cache, pages, 64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_index_pages(cache, pages, 64)
    for pattern in ("reverse", "invalid", "zero", "changed_payload"):
        if pattern == "reverse":
            pages.copy_(pages.flip(0))
        elif pattern == "invalid":
            pages.fill_(-1)
        elif pattern == "zero":
            pages.zero_()
        else:
            cache.random_(0, 256)
        graph.replay()
        for got, want in zip(
            actual, torch_dsv41_index_cache_gather(cache, pages), strict=True
        ):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_prefill_gather_page_zero_and_invalid_page_bytes(dtype):
    cache, _ = _gather_inputs(64, True, dtype)
    pages = torch.tensor(
        [0, -1, cache.shape[0], torch.iinfo(dtype).max, 1], device="cuda", dtype=dtype
    )
    data, scales = gather_index_pages(cache, pages, 64)
    expected_data = torch.zeros((320, 64), device="cuda", dtype=torch.uint8)
    expected_scales = torch.full((320,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    for out_page, physical_page in ((0, 0), (4, 1)):
        expected_data[out_page * 64 : (out_page + 1) * 64] = cache[
            physical_page, :4096
        ].reshape(64, 64)
        expected_scales[out_page * 64 : (out_page + 1) * 64] = cache[
            physical_page, 4096:
        ].view(torch.int32)
    torch.testing.assert_close(data, expected_data, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
