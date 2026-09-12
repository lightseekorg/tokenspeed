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
from test_attention_dsv41 import _reference_quantize
from test_attention_dsv41 import device as device
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.triton import dsv41 as implementation


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


def _oracle(q, weights, keys, shards):
    _, q_ref = _reference_quantize(q, "index")
    _, k_ref = _reference_quantize(keys, "index")
    dots = torch.einsum("thd,nd->thn", q_ref.reshape(q.shape), k_ref).bfloat16()
    products = dots.relu() * weights.cpu().unsqueeze(-1)
    scores = torch.zeros((q.shape[0], keys.shape[0]), dtype=torch.float32)
    # Shard sums round first; the final rank-order FP32 sum is NOT an NCCL oracle.
    for partial in products.chunk(shards, dim=1):
        scores += partial.sum(dim=1).float()
    return scores.to(weights.dtype).float()


def _check(ids, lengths, scores):
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
            rtol=0,
            atol=0,
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
        torch.distributed, "all_gather_into_tensor", side_effect=gather
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
                q, w, cache, pages, visible, None, 65, 17, 8, 2, 64, None, None
            )
            return full + dsv41.index_topk(
                q, w, cache, pages, visible, full[2], 65, 0, 8, 2, 64, None, None
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


@pytest.mark.parametrize("shards", [1, 4], ids=["local", "tp4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_index_scan_graph_oracle(device, shards, dtype, tp_group):
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
        physical_scores = _oracle(q, w, keys, shards)
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
            physical_scores = _oracle(q, w, keys, shards)
            visible.copy_(torch.tensor(lengths, device=device))
            table.copy_(table.flip(1))
            with patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                wraps=torch.distributed.all_gather_into_tensor,
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
            _check(eager[0], eager[1], scores)
            blocks = scores.reshape(3, -1, 8).amax(dim=-1)
            for t, length in enumerate(lengths):
                if length and blocks[t, (length - 1) // 8] > -torch.inf:
                    blocks[t, (length - 1) // 8] = torch.inf
            _check(eager[2], eager[3], blocks)
            for t in range(3):
                scores[t].masked_fill_(
                    ~torch.isin(logical[t] // 8, eager[2][t].cpu()), -torch.inf
                )
            _check(eager[4], eager[5], scores)
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


def test_native_indexer_page_contract_and_graph_lengths(device):
    from tokenspeed_kernel.ops.attention.deep_gemm.dsv41 import (
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
            q, weights, cache, table, visible, None, 16, 32, 8, 1, 64, None, None
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
        q, weights, strided, table, visible, None, 16, 0, 8, 1, 64, None, None
    )
    assert result[1].item() == 16
    assert not ((result[0] >= 64) & (result[0] < 128)).any()


def test_native_broadcast_history_matches_paged_and_refreshes_graph(device):
    from tokenspeed_kernel.ops.attention.deep_gemm.dsv41 import (
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
            q, weights, cache, pages, lengths, None, 512, 64, 8, 1024, 256, None, None
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
