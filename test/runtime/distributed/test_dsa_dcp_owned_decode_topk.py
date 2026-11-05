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

"""KVP decode's owned top-k over real collectives, at Kimi-K3.1 geometry.

32 index heads of width 128, top-k 2048, 64-token index pages inside
128-token cache blocks, and 4-row verify windows at bs 1, 7 and 16. About
70% of the longest request's blocks live on rank 0, and some requests are
shorter than the verify window.

Every rank's owned selection (:func:`select_owned_dsa_topk`, which exchanges
only scores) must hold the local slots of the selection merged on every
rank (:func:`select_dsa_topk`) as scheduler slots compacted to the rank's
local slots, for a BF16 index query and, where a candidate scorer takes
one, an FP8 query with its scales. Rows compare as sets: a scorer's
candidate order is unspecified, and the gfx950 one's follows its atomics.
The ranks' counts must add up to the selection's. A forward's captured
metadata build and owned selection, replayed after the lengths and the
table change in place, must select what the eager merge selects for the
new batch.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-2gpu")

_HEADS, _DIM, _TOPK, _PAGE, _BLOCK, _Q_LEN = 32, 128, 2048, 64, 128, 4
_COLUMNS = 512
# Spare tokens past each request's length, so in-place length edits stay
# inside the request's allocated blocks.
_GROWTH = _BLOCK


def _requests(bs: int, generator: torch.Generator) -> list[int]:
    # The longest request first (skewed ownership), then requests no longer
    # than the verify window, a window past a page edge, and one past topk.
    longest = _COLUMNS * _PAGE - 2 * _BLOCK
    special = [longest, 3, _Q_LEN, _PAGE + _Q_LEN, _TOPK + 3, 5 * _BLOCK]
    rest = torch.randint(
        _Q_LEN, 20000, (max(0, bs - len(special)),), generator=generator
    )
    return (special + rest.tolist())[:bs]


def _place_blocks(
    seq_lens: list[int], degree: int, generator: torch.Generator
) -> tuple[list[list[int]], int]:
    """Virtual blocks per request; block ``v`` lives on rank ``(v - 1) % degree``.

    About 70% of the first request's blocks come from rank 0's blocks.
    """
    counts = [-(-(length + _GROWTH) // _BLOCK) for length in seq_lens]
    per_rank = -(-sum(counts) // degree) + counts[0] + 8
    pools = []
    for rank in range(degree):
        owned = torch.arange(per_rank) * degree + rank + 1
        pools.append(owned[torch.randperm(per_rank, generator=generator)].tolist())
    placed = []
    for index, count in enumerate(counts):
        blocks = []
        for slot in range(count):
            if index == 0 and slot < count * 7 // 10:
                owner = 0
            else:
                live = [rank for rank in range(degree) if pools[rank]]
                owner = live[int(torch.randint(len(live), (1,), generator=generator))]
            blocks.append(pools[owner].pop())
        order = torch.randperm(count, generator=generator).tolist()
        placed.append([blocks[i] for i in order])
    return placed, degree * per_rank + 1


def _page_table(placed: list[list[int]]) -> torch.Tensor:
    # Two 64-token index pages per 128-token block; absent pages are 0.
    table = torch.zeros(len(placed), _COLUMNS, dtype=torch.int32)
    for row, blocks in enumerate(placed):
        pages = torch.tensor(blocks, dtype=torch.int32) * 2
        table[row, : 2 * len(blocks)] = torch.stack((pages, pages + 1), 1).flatten()
    return table


def _plane(keys, virtual_blocks, placement, device):
    """This rank's FP8 index-K plane of every virtual slot's key."""
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    local_blocks = -(-(virtual_blocks - 1) // len(placement.group)) + 1
    plane = torch.zeros(
        local_blocks * _BLOCK, _DIM + 4, dtype=torch.uint8, device=device
    )
    values, scales = quantize_fp8(
        keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    slots, owned = resolve_cache_slots(
        torch.arange(keys.shape[0], device=device), placement
    )
    index_k_block_split_scatter(
        plane,
        values,
        scales,
        slots,
        page_size=_PAGE,
        head_dim=_DIM,
        group_size=128,
        write_mask=owned,
    )
    return plane


def _verify_rows(seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Request ids and causal lengths of every verify row, as the DSA layer
    derives them."""
    rows = seq_lens.numel() * _Q_LEN
    requests = torch.arange(rows, device=seq_lens.device, dtype=torch.int32) // _Q_LEN
    causal = (
        seq_lens[requests.long()]
        - (_Q_LEN - 1)
        + torch.arange(rows, device=seq_lens.device) % _Q_LEN
    )
    return requests, causal


def _index_queries(q: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """The BF16 index query, and the FP8 one with its scales where a
    candidate scorer takes it."""
    from tokenspeed_kernel.ops.quantization import quantize_fp8
    from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    rows = q.shape[0]
    q_fp8, q_scales = quantize_fp8(
        q.view(rows, -1), granularity="token_group", group_size=_DIM
    )
    try:
        select_kernel(
            "attention",
            "dsa_index_candidates",
            format_signature(
                q=dense_tensor_format(torch.float8_e4m3fn),
                weights=dense_tensor_format(torch.bfloat16),
            ),
            traits={
                "index_heads": _HEADS,
                "head_dim": _DIM,
                "page_size": _PAGE,
                "index_k_layout": "packed",
            },
        )
    except NoKernelFoundError:
        return [(q, None)]
    return [(q, None), (q_fp8.view(rows, _HEADS, _DIM), q_scales)]


def _assert_same_rows(slots: torch.Tensor, expected: torch.Tensor) -> None:
    """Equal slot sets per row, -1 padding included."""
    torch.testing.assert_close(
        slots.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0
    )


def _worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed_kernel.ops.attention.dsa.triton import local_topk_to_global_slots

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.indexer import (
        build_dcp_index_metadata,
        owned_dsa_topk_fits,
        select_dsa_topk,
        select_owned_dsa_topk,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        CachePlacement,
        compact_owned_cache_slots,
    )

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=300),
    )
    graph = None
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        backend.configure(use_pynccl=False)
        set_global_backend(backend)
        kwargs = dict(
            page_size=_PAGE,
            topk=_TOPK,
            softmax_scale=0.1,
            initial_tokens=0,
            local_tokens=0,
            query_group_size=_Q_LEN,
        )

        def metadata_of(q, weights, plane, table, seq_lens, placement):
            requests, causal = _verify_rows(seq_lens)
            return build_dcp_index_metadata(
                q,
                weights,
                plane,
                table,
                requests,
                causal,
                placement=placement,
                page_size=_PAGE,
            )

        def merged(q, q_scales, weights, plane, table, seq_lens, placement):
            """The selection merged on every rank, its scheduler slots
            compacted to this rank's local slots, and its lengths."""
            metadata = metadata_of(q, weights, plane, table, seq_lens, placement)
            offsets, _ = select_dsa_topk(
                q,
                weights,
                plane,
                table,
                metadata.query_requests,
                metadata.causal_lens,
                placement=placement,
                max_logits_bytes=None,
                metadata=metadata,
                q_scales=q_scales,
                **kwargs,
            )
            slots, lens = local_topk_to_global_slots(
                local_topk_offsets=offsets,
                block_table=table,
                block_size=_PAGE,
                q_len_per_req=_Q_LEN,
            )
            return compact_owned_cache_slots(slots, lens, placement), lens

        for bs in (1, 7, 16):
            # Every rank derives the same batch from the seed.
            generator = torch.Generator().manual_seed(6100 + 13 * bs)
            seq_list = _requests(bs, generator)
            placed, virtual_blocks = _place_blocks(seq_list, degree, generator)
            table = _page_table(placed).to(device)
            seq_lens = torch.tensor(seq_list, dtype=torch.int32, device=device)
            rows = bs * _Q_LEN
            # Seeded, so a failure reproduces; rank 0's draw is broadcast.
            torch.manual_seed(6900 + 1000 * degree + 13 * bs)
            keys = torch.randn(
                virtual_blocks * _BLOCK, _DIM, device=device, dtype=torch.bfloat16
            )
            q = torch.randn(rows, _HEADS, _DIM, device=device, dtype=torch.bfloat16)
            weights = torch.randn(rows, _HEADS, device=device, dtype=torch.bfloat16)
            for tensor in (keys, q, weights):
                dist.broadcast(tensor, 0)
            placement = CachePlacement(_BLOCK, virtual_blocks, group, rank)
            plane = _plane(keys, virtual_blocks, placement, device)
            del keys
            queries = _index_queries(q)
            for query, scales in queries:
                metadata = metadata_of(
                    query, weights, plane, table, seq_lens, placement
                )
                assert owned_dsa_topk_fits(
                    rows, metadata, page_size=_PAGE, max_logits_bytes=None
                )
                assert not owned_dsa_topk_fits(
                    rows, metadata, page_size=_PAGE, max_logits_bytes=64 << 20
                )
                out = torch.full((rows, _TOPK), 7, dtype=torch.int32, device=device)
                lens_out = torch.full((rows,), 7, dtype=torch.int32, device=device)
                owned, counts = select_owned_dsa_topk(
                    query,
                    weights,
                    plane,
                    metadata=metadata,
                    q_scales=scales,
                    out=out,
                    lens_out=lens_out,
                    **kwargs,
                )
                (expected, expected_counts), lens = merged(
                    query, scales, weights, plane, table, seq_lens, placement
                )

                assert owned.data_ptr() == out.data_ptr()
                assert counts.data_ptr() == lens_out.data_ptr()
                _assert_same_rows(owned, expected)
                torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
                # The ranks' shares partition the selection.
                total = counts.clone()
                dist.all_reduce(total)
                torch.testing.assert_close(total, lens, rtol=0, atol=0)
            print(
                f"rank {rank}/{degree}: bs={bs} owned selections of "
                f"{len(queries)} query types matched the merge",
                flush=True,
            )

        # Capture a forward's build and owned selection at bs 16, as a
        # decode graph holds them, and replay it for other batches.
        query, scales = queries[-1]
        out = torch.empty((rows, _TOPK), dtype=torch.int32, device=device)
        lens_out = torch.empty(rows, dtype=torch.int32, device=device)

        def step():
            select_owned_dsa_topk(
                query,
                weights,
                plane,
                metadata=metadata_of(query, weights, plane, table, seq_lens, placement),
                q_scales=scales,
                out=out,
                lens_out=lens_out,
                **kwargs,
            )

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            step()
        original = seq_lens.clone()
        for edit in ("lengths", "rows"):
            delta = torch.randint(
                -5, _GROWTH, (bs,), generator=generator, dtype=torch.int32
            )
            seq_lens.copy_((original + delta.to(device)).clamp_min(1))
            if edit == "rows":
                # Rows now pair with other requests' pages and lengths.
                table.copy_(table.roll(1, 0))
                seq_lens.copy_(seq_lens.roll(1, 0))
            torch.cuda.synchronize()
            dist.barrier(group=coordination)
            graph.replay()
            (expected, expected_counts), _ = merged(
                query, scales, weights, plane, table, seq_lens, placement
            )
            _assert_same_rows(out, expected)
            torch.testing.assert_close(lens_out, expected_counts, rtol=0, atol=0)
        print(f"rank {rank}/{degree}: graph replay matched the merge", flush=True)
        graph.reset()
        graph = None
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        # NCCL shutdown waits for communicators retained by captured graphs.
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_owned_decode_topk_is_each_ranks_share_of_the_merge(tmp_path, degree):
    if not torch.cuda.is_available() or torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    mp.spawn(
        _worker,
        args=(degree, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=degree,
        join=True,
    )
