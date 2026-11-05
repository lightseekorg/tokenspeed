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

"""DCP index top-k at Kimi-K3.1 geometry over real collectives.

32 index heads of width 128, top-k 2048, 64-token index pages inside
128-token cache blocks, and 8-row DSpark verify windows at bs 1, 7 and 64.
The decode table has the width of a 1M max_model_len (or a short one).
Block ownership is skewed: about 70% of the longest request's blocks live
on rank 0. Some requests are shorter than the verify window, and every
window's rows sit in one block, so other ranks hold no local candidates.

Per batch, the one-tile selection with per-forward metadata, the multi-tile
selection with the same metadata (row slices of the planned pages, one or
more budget-bounded exchanges), the multi-tile selection that compacts per
tile and the unsharded selection (degree-1 placement over the whole plane)
must give equal lengths and equal sets, except that the unsharded selection
may keep other tokens of an exact FP32 score tie at the k-th score. They
must also match the native unsharded decode top-k up to ties (Jaccard >=
0.99). The inputs come from fixed seeds. At bs 64, a captured metadata
build read by two selections, as two indexer layers read it (one tile, then
the budget's tiles), must match eager after the lengths and the table change
in place.

A prefill (a long cached prefix with a chunk, and a cold chunk) selects at
the owned width the host table bounds what it selects at the table's width, with the pages planned once per
forward or per tile, merged on every rank or on the row owner, under skewed
ownership and with every page on rank 0 (a prefill-graph capture's dummy,
whose owned width is the table's).

KVP prefill rows (a skewed long prefix, a cold request and a short one, then
the prefill graph's capture batch whose pages are all rank 0's) merged on
their row owner must select the sets the merge on every rank and (up to
such ties) the unsharded selection select, over several budget-bounded
exchanges whose last one does not split evenly over the ranks, and in one
exchange.

Two DSA layers' eager breaks (the model's ``forward_dsa_core``: the narrowing
of a padded replay, the index-K write and the KVP prefill top-k) captured in
a breakable prefill graph from that capture batch and replayed padded for
live prefills above and below the row-owner merge's row gate must select,
at the owned width and at the table's, what the same layers select eagerly
over the live rows.
"""

import dataclasses
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    suite="runtime-2gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the DeepGEMM index-candidate scorer is NVIDIA-only",
)

_HEADS, _DIM, _TOPK, _PAGE, _BLOCK, _Q_LEN = 32, 128, 2048, 64, 128, 8
# calc_padded_blocks(1048576 + 3 * 8, 64): the 1M decode table with verify padding.
_WIDE_COLUMNS = 16386
_SHORT_COLUMNS = 512
# Spare tokens past each length, so in-place length edits stay inside allocated blocks.
_GROWTH = _BLOCK


def _requests(columns: int, bs: int, generator: torch.Generator) -> list[int]:
    capacity = columns * _PAGE - 2 * _BLOCK
    longest = min(capacity, 131072 + 5)
    # Longest first (skews ownership), then sub-window, page-edge and past-topk lengths.
    special = [longest, 3, 7, 8, _PAGE + _Q_LEN, _TOPK + 3, 5 * _BLOCK]
    rest = torch.randint(
        _Q_LEN, min(capacity, 20000), (max(0, bs - len(special)),), generator=generator
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
                choice = torch.randint(len(live), (1,), generator=generator)
                owner = live[int(choice)]
            blocks.append(pools[owner].pop())
        order = torch.randperm(count, generator=generator).tolist()
        placed.append([blocks[i] for i in order])
    return placed, degree * per_rank + 1


def _page_table(placed: list[list[int]], columns: int) -> torch.Tensor:
    # Two 64-token index pages per 128-token block; absent pages are 0.
    table = torch.zeros(len(placed), columns, dtype=torch.int32)
    for row, blocks in enumerate(placed):
        pages = torch.tensor(blocks, dtype=torch.int32) * 2
        table[row, : 2 * len(blocks)] = torch.stack((pages, pages + 1), 1).flatten()
    return table


def _plane(keys, virtual_blocks, placement, device):
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    degree = len(placement.group)
    local_blocks = -(-(virtual_blocks - 1) // degree) + 1
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


def _index_scorer(q, weights, plane, table, requests, causal, placement):
    """The FP32 index scores the selections rank, for chosen query rows.

    ``scorer(rows)`` scores ``rows`` against the unsharded ``plane`` with
    the candidate leaf the selections use, keeping every visible token as a
    candidate, so each token carries the score the selections saw, bit for
    bit. It returns ``[rows, longest causal length]`` scores by logical
    position, -inf past each row's causal length.
    """
    from tokenspeed_kernel.ops.attention.dsa import dsa_index_candidates

    from tokenspeed.runtime.layers.attention.dcp.indexer import _local_index_table

    local_table = _local_index_table(table, placement, _PAGE)

    def scorer(rows: torch.Tensor) -> torch.Tensor:
        width = int(causal[rows].max().clamp_min(1))
        positions, values = dsa_index_candidates(
            q[rows].contiguous(),
            weights[rows].contiguous(),
            plane,
            local_table,
            requests[rows].contiguous(),
            causal[rows].contiguous(),
            page_size=_PAGE,
            # A power of two no smaller than any row's visible tokens.
            topk=1 << (width - 1).bit_length(),
            softmax_scale=0.1,
            initial_tokens=0,
            local_tokens=0,
            solution=None,
        )
        scores = values.new_full((rows.numel(), width + 1), -float("inf"))
        # Invalid candidates (-1, -inf) land in the spare last column.
        scores.scatter_(1, torch.where(positions >= 0, positions, width).long(), values)
        return scores[:, :width]

    return scorer


def _assert_top_k_up_to_ties(row, selections, length, scores):
    """Both ``selections`` (one row's positions, padding -1) are exact top-k
    selections of the row's ``length`` visible tokens under ``scores`` and
    differ only in tokens that score the k-th score T bit for bit."""
    visible = scores[:length]
    k = min(length, _TOPK)
    assert k > 0, f"row {row}: no visible tokens, yet the selections differ"
    # The k-th score, as a one-element tensor to compare bit patterns.
    threshold = visible.topk(k).values[-1:]
    above = int((visible > threshold).sum())
    chosen = []
    for selection in selections:
        positions = selection[selection >= 0].long()
        count = positions.numel()
        assert count == k, f"row {row}: {count} tokens selected, expected {k}"
        assert positions.unique().numel() == k, f"row {row}: a token selected twice"
        past = positions[positions >= length].tolist()
        assert not past, f"row {row}: tokens {past[:8]} past causal length {length}"
        picked = visible[positions]
        below = positions[picked < threshold].tolist()
        assert not below, (
            f"row {row}: tokens {below[:8]} score below the k-th score "
            f"{float(threshold).hex()}"
        )
        missed = above - int((picked > threshold).sum())
        assert not missed, (
            f"row {row}: {missed} tokens above the k-th score "
            f"{float(threshold).hex()} not selected"
        )
        chosen.append(set(positions.tolist()))
    differ = sorted(chosen[0] ^ chosen[1])
    tied = visible[differ].view(torch.int32) == threshold.view(torch.int32)
    assert bool(tied.all()), (
        f"row {row}: tokens {differ[:8]} differ but do not all score the k-th "
        f"score {float(threshold).hex()} exactly"
    )


def _assert_same_sets(actual, expected, causal, scorer=None):
    """Equal lengths and equal sets of selected positions.

    Exact top-k selections may keep different tokens of an exact FP32 score
    tie at the k-th score: the sharded merge keeps the first tied candidate
    in rank-major order, the unsharded selection's persistent top-k the
    first in atomic order. With ``scorer`` (an :func:`_index_scorer` of
    these rows) rows whose sets differ must instead hold every token that
    scores above the row's k-th score, none below it and none twice, and
    differ only in tokens that score it exactly.
    """
    (indices, lens), (expected_indices, expected_lens) = actual, expected
    torch.testing.assert_close(lens, expected_lens, rtol=0, atol=0)
    torch.testing.assert_close(
        lens, causal.clamp(0, _TOPK).to(lens.dtype), rtol=0, atol=0
    )
    # Equal lengths and equal sorted rows (padding -1 first) are equal sets.
    mine = indices.sort(dim=-1).values
    theirs = expected_indices.sort(dim=-1).values
    if scorer is None or mine.shape != theirs.shape:
        torch.testing.assert_close(mine, theirs, rtol=0, atol=0)
        return
    differ = (mine != theirs).any(dim=-1).nonzero().flatten()
    # A few rows at a time: the scores hold each row's every visible token.
    for start in range(0, differ.numel(), 16):
        rows = differ[start : start + 16]
        for row, scores in zip(rows.tolist(), scorer(rows), strict=True):
            _assert_top_k_up_to_ties(
                row, (mine[row], theirs[row]), int(causal[row]), scores
            )


def _assert_matches_native(selected, lens, q, weights, plane, table, seq_lens):
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk, dsa_plan

    # Native top-k bounds rows by seq_len - 7 + j: only requests holding a full window.
    keep = (seq_lens >= _Q_LEN).nonzero().flatten()
    if not keep.numel():
        return
    rows = keep.unsqueeze(1) * _Q_LEN + torch.arange(_Q_LEN, device=keep.device)
    rows = rows.flatten()
    kept_lens = seq_lens[keep].to(torch.int32).contiguous()
    kept_table = table[keep].contiguous()
    seq_lens_2d = kept_lens.repeat_interleave(_Q_LEN).unsqueeze(1).contiguous()
    native, native_lens = dsa_decode_topk(
        q[rows].contiguous(),
        weights[rows].contiguous(),
        kept_lens,
        kept_table,
        page_size=_PAGE,
        topk=_TOPK,
        softmax_scale=0.1,
        batch_invariant=False,
        q_len_per_req=_Q_LEN,
        index_k_cache=plane,
        seq_lens_2d=seq_lens_2d,
        plan=dsa_plan(page_size=_PAGE, seq_lens_2d=seq_lens_2d, solution="deep_gemm"),
        solution="deep_gemm",
        slot_order="selection",
    )
    offsets = selected[rows]
    requests = rows // _Q_LEN
    slots = torch.where(
        offsets >= 0,
        table[requests.unsqueeze(1), offsets.clamp_min(0).long() // _PAGE].long()
        * _PAGE
        + offsets % _PAGE,
        -1,
    )
    torch.testing.assert_close(native_lens, lens[rows], rtol=0, atol=0)
    worst = 1.0
    for dcp_row, native_row in zip(slots.tolist(), native.tolist(), strict=True):
        mine = {slot for slot in dcp_row if slot >= 0}
        theirs = {slot for slot in native_row if slot >= 0}
        if mine or theirs:
            worst = min(worst, len(mine & theirs) / len(mine | theirs))
    # Different selectors may break FP8 score ties differently.
    assert worst >= 0.99, f"Jaccard {worst:.4f} against the native decode top-k"


def _worker(rank, degree, columns, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.indexer import (
        build_dcp_index_metadata,
        select_dsa_topk,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

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
        )
        # 15 rows/tile at 1M width, 8 at 512; at bs 64 the budget bounds exchanges too.
        budget = (64 << 20) if columns == _WIDE_COLUMNS else (1 << 20)
        for bs in (1, 7, 64):
            # Every rank derives the same batch from the seed.
            generator = torch.Generator().manual_seed(4100 + 13 * bs + columns)
            seq_list = _requests(columns, bs, generator)
            placed, virtual_blocks = _place_blocks(seq_list, degree, generator)
            table = _page_table(placed, columns).to(device)
            seq_lens = torch.tensor(seq_list, dtype=torch.int32, device=device)
            rows = bs * _Q_LEN
            # Seeded, so a failure reproduces; rank 0's draw is broadcast.
            torch.manual_seed(4900 + 1000 * degree + 13 * bs + columns)
            keys = torch.randn(
                virtual_blocks * _BLOCK, _DIM, device=device, dtype=torch.bfloat16
            )
            q = torch.randn(rows, _HEADS, _DIM, device=device, dtype=torch.bfloat16)
            weights = torch.randn(rows, _HEADS, device=device, dtype=torch.bfloat16)
            for tensor in (keys, q, weights):
                dist.broadcast(tensor, 0)
            sharded = CachePlacement(_BLOCK, virtual_blocks, group, rank)
            unsharded = CachePlacement(_BLOCK, virtual_blocks, (rank,), 0)
            local = _plane(keys, virtual_blocks, sharded, device)
            full = _plane(keys, virtual_blocks, unsharded, device)
            del keys
            requests, causal = _verify_rows(seq_lens)

            metadata = build_dcp_index_metadata(
                q,
                weights,
                local,
                table,
                requests,
                causal,
                placement=sharded,
                page_size=_PAGE,
            )
            assert metadata.pages is not None
            assert metadata.pages.kernel == "deep_gemm_dsa_index_candidates"
            single = select_dsa_topk(
                q,
                weights,
                local,
                table,
                metadata.query_requests,
                metadata.causal_lens,
                placement=sharded,
                max_logits_bytes=None,
                metadata=metadata,
                **kwargs,
            )
            # The planned pages, row-sliced across a logits budget's tiles.
            planned_tiled = select_dsa_topk(
                q,
                weights,
                local,
                table,
                metadata.query_requests,
                metadata.causal_lens,
                placement=sharded,
                max_logits_bytes=budget,
                metadata=metadata,
                **kwargs,
            )
            tiled = select_dsa_topk(
                q,
                weights,
                local,
                table,
                requests,
                causal,
                placement=sharded,
                max_logits_bytes=budget,
                **kwargs,
            )
            reference = select_dsa_topk(
                q,
                weights,
                full,
                table,
                requests,
                causal,
                placement=unsharded,
                max_logits_bytes=None,
                **kwargs,
            )
            # ~5% of bs-64 draws tie at the k-th score; the kept tokens may differ.
            scorer = _index_scorer(q, weights, full, table, requests, causal, unsharded)
            _assert_same_sets(single, reference, causal, scorer)
            _assert_same_sets(planned_tiled, reference, causal, scorer)
            _assert_same_sets(tiled, reference, causal, scorer)
            _assert_matches_native(*single, q, weights, full, table, seq_lens)
            print(
                f"rank {rank}/{degree}: columns={columns} bs={bs} matched the "
                "unsharded and native top-k",
                flush=True,
            )
        del full, scorer

        # Capture one build at bs 64 read by two layers: one tile, then budgeted tiles.
        def step():
            requests, causal = _verify_rows(seq_lens)
            metadata = build_dcp_index_metadata(
                q,
                weights,
                local,
                table,
                requests,
                causal,
                placement=sharded,
                page_size=_PAGE,
            )
            return [
                select_dsa_topk(
                    q,
                    weights,
                    local,
                    table,
                    metadata.query_requests,
                    metadata.causal_lens,
                    placement=sharded,
                    max_logits_bytes=layer_budget,
                    metadata=metadata,
                    **kwargs,
                )
                for layer_budget in (None, budget)
            ]

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            captured = step()
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
            expected = step()
            causal = _verify_rows(seq_lens)[1]
            for layer, reference in zip(captured, expected, strict=True):
                _assert_same_sets(layer, reference, causal)
        print(f"rank {rank}/{degree}: graph replay matched eager", flush=True)
        graph.reset()
        graph = None
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        # NCCL shutdown waits for communicators retained by captured graphs.
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


def _owned_width_worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.indexer import (
        build_dcp_index_metadata,
        select_dsa_topk,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        CachePlacement,
        max_owned_index_pages,
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
        )
        # Prefixed and cold chunks; a small budget forces several tiles and exchanges.
        prefix_lens, extend_lens = [20000, 0], [1500, 700]
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        budget = 64 << 20
        # Seeded, so a failure reproduces; rank 0's draws are broadcast.
        torch.manual_seed(5300 + degree)
        for ownership in ("skewed", "one_owner"):
            # Every rank derives the same batch from the seed.
            generator = torch.Generator().manual_seed(5200 + degree)
            placed, virtual_blocks = _place_blocks(seq_lens, degree, generator)
            if ownership == "one_owner":
                # Rank 0 owns all pages, as at capture; distinct blocks avoid ties.
                first = 0
                for request, blocks in enumerate(placed):
                    placed[request] = [
                        1 + degree * (first + i) for i in range(len(blocks))
                    ]
                    first += len(blocks)
            # The prefill reads the table up to the longest history.
            width = -(-max(seq_lens) // _PAGE)
            table = _page_table(placed, 2 * max(map(len, placed)))[:, :width]
            sharded = CachePlacement(_BLOCK, virtual_blocks, group, rank)
            bound = max_owned_index_pages(
                table,
                torch.tensor(seq_lens),
                page_size=_PAGE,
                placement=sharded,
            )
            # The DSA leaf's width: at least topk positions, 16-page groups.
            columns = -(-max(bound, _TOPK // _PAGE) // 16) * 16
            table = table.to(device)
            rows = sum(extend_lens)
            keys = torch.randn(
                virtual_blocks * _BLOCK, _DIM, device=device, dtype=torch.bfloat16
            )
            q = torch.randn(rows, _HEADS, _DIM, device=device, dtype=torch.bfloat16)
            weights = torch.randn(rows, _HEADS, device=device, dtype=torch.bfloat16)
            for tensor in (keys, q, weights):
                dist.broadcast(tensor, 0)
            local = _plane(keys, virtual_blocks, sharded, device)
            del keys
            requests, causal = _prefill_rows(
                tuple(zip(prefix_lens, extend_lens)), device
            )

            full = select_dsa_topk(
                q,
                weights,
                local,
                table,
                requests,
                causal,
                placement=sharded,
                max_logits_bytes=budget,
                **kwargs,
            )
            metadata = build_dcp_index_metadata(
                q,
                weights,
                local,
                table,
                requests,
                causal,
                placement=sharded,
                page_size=_PAGE,
                columns=columns,
            )
            assert metadata.columns == min(columns, width)
            if ownership == "one_owner":
                assert metadata.columns == width
            else:
                assert metadata.columns < width
            # Within the memory cap: planned once, every row's pages fit.
            assert metadata.pages is not None
            assert not metadata.pages.overflow.any()
            planned = select_dsa_topk(
                q,
                weights,
                local,
                table,
                metadata.query_requests,
                metadata.causal_lens,
                placement=sharded,
                max_logits_bytes=budget,
                metadata=metadata,
                **kwargs,
            )
            per_tile = select_dsa_topk(
                q,
                weights,
                local,
                table,
                requests,
                causal,
                placement=sharded,
                max_logits_bytes=budget,
                columns=columns,
                **kwargs,
            )
            _assert_same_sets(planned, full, causal)
            _assert_same_sets(per_tile, full, causal)
            # Owned-width tiles join row-owner exchanges, whose budget sizes them too.
            for owned in (dict(metadata=metadata), dict(columns=columns)):
                owner = select_dsa_topk(
                    q,
                    weights,
                    local,
                    table,
                    metadata.query_requests,
                    metadata.causal_lens,
                    placement=sharded,
                    max_logits_bytes=budget,
                    row_owner=True,
                    **owned,
                    **kwargs,
                )
                _assert_same_sets(owner, full, causal)
            print(
                f"rank {rank}/{degree}: {ownership} owned width {metadata.columns} "
                f"of {width} pages matched the table's width",
                flush=True,
            )
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_k31_dcp_prefill_topk_owned_width_matches_full_width(tmp_path, degree):
    """KVP prefill top-k at the owned width the host table bounds (pages
    planned once per forward, or per tile; merged on every rank, or on the
    row owner) selects what the table's width selects, with skewed ownership
    and with every page on one rank."""
    if (
        torch.version.cuda is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < degree
    ):
        pytest.skip(f"requires {degree} CUDA GPUs")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_hopper_plus:
        pytest.skip("the DeepGEMM index scorer requires NVIDIA Hopper or newer")
    mp.spawn(
        _owned_width_worker,
        args=(degree, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=degree,
        join=True,
    )


@pytest.mark.parametrize("columns", [_SHORT_COLUMNS, _WIDE_COLUMNS])
@pytest.mark.parametrize("degree", [2, 4, 8])
def test_k31_dcp_decode_topk_matches_unsharded(tmp_path, degree, columns):
    if (
        torch.version.cuda is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < degree
    ):
        pytest.skip(f"requires {degree} CUDA GPUs")
    from tokenspeed_kernel.platform import current_platform

    # ROCm reports gfx950 as capability (9, 5); the scorer needs NVIDIA.
    if not current_platform().is_hopper_plus:
        pytest.skip("the DeepGEMM index scorer requires NVIDIA Hopper or newer")
    if (
        columns == _WIDE_COLUMNS
        and torch.cuda.get_device_properties(0).total_memory < 32 << 30
    ):
        pytest.skip("the 1M-wide decode table needs 32 GiB per GPU")
    mp.spawn(
        _worker,
        args=(degree, columns, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=degree,
        join=True,
    )


# Skewed long prefix, cold sub-topk request, short one; 1153 rows top the 1024 gate.
_PREFILL = ((6000, 800), (0, 300), (2500, 53))
# 128 rows a tile on 512 columns, two tiles an exchange: five, the last of 129 rows.
_PREFILL_BUDGET = 16 << 20
# Prefill capture batch: one prefix-free request, every page column on one block.
_CAPTURE_ROWS = 2048


def _prefill_rows(
    prefill: tuple[tuple[int, int], ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Request ids and global causal lengths of every prefill row, as the
    DSA layer derives them."""
    requests, causal = [], []
    for request, (prefix, new) in enumerate(prefill):
        requests += [request] * new
        causal += range(prefix + 1, prefix + new + 1)
    return (
        torch.tensor(requests, dtype=torch.int32, device=device),
        torch.tensor(causal, dtype=torch.int32, device=device),
    )


def _row_owner_worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp import indexer
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=300),
    )
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        # PyNccl has no all-to-all, so the row-owner exchange uses torch.distributed.
        backend.configure(use_pynccl=False)
        set_global_backend(backend)

        def selections(table, virtual_blocks, requests, causal, seed):
            """Row-owner, every-rank and unsharded selections of these rows,
            with the budget and without."""
            torch.manual_seed(seed)
            rows = causal.numel()
            keys = torch.randn(
                virtual_blocks * _BLOCK, _DIM, device=device, dtype=torch.bfloat16
            )
            q = torch.randn(rows, _HEADS, _DIM, device=device, dtype=torch.bfloat16)
            weights = torch.randn(rows, _HEADS, device=device, dtype=torch.bfloat16)
            for tensor in (keys, q, weights):
                dist.broadcast(tensor, 0)
            sharded = CachePlacement(_BLOCK, virtual_blocks, group, rank)
            unsharded = CachePlacement(_BLOCK, virtual_blocks, (rank,), 0)
            local = _plane(keys, virtual_blocks, sharded, device)
            full = _plane(keys, virtual_blocks, unsharded, device)

            def select(placement, plane, budget, row_owner):
                return indexer.select_dsa_topk(
                    q,
                    weights,
                    plane,
                    table,
                    requests,
                    causal,
                    placement=placement,
                    page_size=_PAGE,
                    topk=_TOPK,
                    softmax_scale=0.1,
                    initial_tokens=0,
                    local_tokens=0,
                    max_logits_bytes=budget,
                    row_owner=row_owner,
                )

            reference = select(unsharded, full, None, False)
            scorer = _index_scorer(q, weights, full, table, requests, causal, unsharded)
            results = [reference]
            for budget in (_PREFILL_BUDGET, None):
                owner = select(sharded, local, budget, True)
                # Paths may break FP8 score ties differently, so they compare as sets.
                _assert_same_sets(owner, select(sharded, local, budget, False), causal)
                _assert_same_sets(owner, reference, causal, scorer)
                results.append(owner)
            return results

        generator = torch.Generator().manual_seed(4200 + degree)
        seq_lens = [prefix + new for prefix, new in _PREFILL]
        placed, virtual_blocks = _place_blocks(seq_lens, degree, generator)
        table = _page_table(placed, _SHORT_COLUMNS).to(device)
        requests, causal = _prefill_rows(_PREFILL, device)
        rows = causal.numel()
        assert rows >= 1024
        tile_rows = indexer._tile_rows(rows, _SHORT_COLUMNS, _PAGE, _PREFILL_BUDGET)
        exchange_rows = indexer._exchange_rows(
            rows, tile_rows, degree, _TOPK, _PREFILL_BUDGET, row_owner=True
        )
        # At least three exchanges; the last spans tiles and splits unevenly over ranks.
        assert rows > 2 * exchange_rows and rows % exchange_rows > tile_rows
        assert rows % exchange_rows % degree
        selections(table, virtual_blocks, requests, causal, 4300 + degree)
        print(
            f"rank {rank}/{degree}: skewed prefill row-owner selection matched",
            flush=True,
        )

        # Capture batch: rank 0 holds every page, so each row selects all its tokens.
        capture_table = _page_table([[1] * (_SHORT_COLUMNS // 2)], _SHORT_COLUMNS)
        requests = torch.zeros(_CAPTURE_ROWS, dtype=torch.int32, device=device)
        causal = torch.arange(1, _CAPTURE_ROWS + 1, dtype=torch.int32, device=device)
        visible = torch.arange(_TOPK, dtype=torch.int32, device=device)
        every_token = torch.where(visible < causal.unsqueeze(1), visible, -1)
        for indices, lens in selections(
            capture_table.to(device), 2 * degree + 1, requests, causal, 4400 + degree
        ):
            torch.testing.assert_close(lens, causal, rtol=0, atol=0)
            torch.testing.assert_close(
                indices.sort(dim=-1).values,
                every_token.sort(dim=-1).values,
                rtol=0,
                atol=0,
            )
        print(
            f"rank {rank}/{degree}: capture-batch row-owner selection matched",
            flush=True,
        )
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_k31_dcp_prefill_topk_row_owner_matches_replicated(tmp_path, degree):
    if (
        torch.version.cuda is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < degree
    ):
        pytest.skip(f"requires {degree} CUDA GPUs")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_hopper_plus:
        pytest.skip("the DeepGEMM index scorer requires NVIDIA Hopper or newer")
    mp.spawn(
        _row_owner_worker,
        args=(degree, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=degree,
        join=True,
    )


# The output columns of the padded-replay test's DSA layers.
_OUT_COLUMNS = 8


@dataclasses.dataclass(frozen=True)
class _AttentionLayer:
    """The MLA layer handle a DSA break passes to the backend and the pool."""

    layer_id: int


@dataclasses.dataclass
class _ChunkedPrefillMetadata:
    """The extend lengths and kernel page table the prefill top-k reads."""

    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    page_table: torch.Tensor


def _extend_slots(table: torch.Tensor, prefill) -> torch.Tensor:
    """Virtual cache slots of the new rows of ``prefill`` ((prefix, new) per
    request) over the host kernel-page ``table``: the extend write span."""
    requests, positions = [], []
    for request, (prefix, new) in enumerate(prefill):
        requests += [request] * new
        positions += range(prefix, prefix + new)
    requests = torch.tensor(requests, dtype=torch.int64)
    positions = torch.tensor(positions, dtype=torch.int64)
    pages = table[requests, positions // _PAGE].to(torch.int64)
    return pages * _PAGE + positions % _PAGE


class _IndexKPool:
    """The cache pool as a DSA break reads and writes it: one index-K plane
    (``_plane``'s layout) for every layer."""

    def __init__(self, plane: torch.Tensor):
        self.plane = plane
        self.index_k_page_size = _PAGE

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.plane

    def set_index_k_buffer(self, layer_id, loc, keys, *, write_mask, key_norm):
        """The FP8 arm of ``DSATokenToKVPool.set_index_k_buffer``."""
        from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
        from tokenspeed_kernel.ops.quantization import quantize_fp8

        assert key_norm is None
        values, scales = quantize_fp8(
            keys, granularity="token_group", group_size=128, scale_encoding="float32"
        )
        index_k_block_split_scatter(
            self.plane,
            values,
            scales,
            loc,
            page_size=_PAGE,
            head_dim=_DIM,
            group_size=128,
            write_mask=write_mask,
        )


class _KvpPrefillBackend:
    """The attention backend as a DSA break reads it on a pure KVP extend:
    the forward's chunked-prefill metadata, write span and owned index width
    (planned when ``owned_width``), the cache placement and the shared top-k.

    :meth:`begin` stands in for ``init_forward_metadata``: it plans the owned
    width from the host table as ``DSABackend._plan_prefill_index_columns``
    does and clears the shared top-k, so every forward (the capture batch's
    or a live one) starts from its own metadata.
    """

    def __init__(self, placement, owned_width: bool):
        from tokenspeed.runtime.layers.attention.backends.base import SparseTopKShare

        self.placement = placement
        self.owned_width = owned_width
        self.sparse_topk = SparseTopKShare()
        self.forward_decode_metadata = None
        self.chunked_prefill_metadata: _ChunkedPrefillMetadata | None = None
        self._write_slots: torch.Tensor | None = None
        self._index_columns: int | None = None

    def begin(self, table: torch.Tensor, prefill, device: torch.device) -> None:
        """Metadata of a pure extend of ``prefill`` ((prefix, new) per
        request) over the host kernel-page ``table``."""
        from tokenspeed.runtime.layers.attention.dcp.placement import (
            max_owned_index_pages,
        )

        prefix_lens = torch.tensor([prefix for prefix, _ in prefill], dtype=torch.int32)
        new_lens = torch.tensor([new for _, new in prefill], dtype=torch.int32)
        self.chunked_prefill_metadata = _ChunkedPrefillMetadata(
            prefix_lens, new_lens, table.to(device)
        )
        self._write_slots = _extend_slots(table, prefill).to(device)
        self._index_columns = None
        if self.owned_width:
            bound = max_owned_index_pages(
                table, prefix_lens + new_lens, page_size=_PAGE, placement=self.placement
            )
            # The DSA leaf's width: at least topk positions, 16-page groups.
            self._index_columns = -(-max(bound, _TOPK // _PAGE) // 16) * 16
        self.sparse_topk.clear()

    def write_locations(self, layer, mode) -> torch.Tensor:
        return self._write_slots

    def cache_placement(self, layer):
        return self.placement

    def prefill_index_columns(self) -> int | None:
        return self._index_columns


class _IdentityAttnComm:
    """A prefill rank's pre-attention communication (attention DP 1): the
    identity, so a padded replay can narrow the break's rows."""

    def pre_attn_comm(self, hidden_states, ctx):
        return hidden_states

    def needs_pre_attn_all_gather(self) -> bool:
        return False


def _k31_dsa_layer(records: list):
    """``GlmDsaSparseAttentionMixin`` as a Kimi-K3.1 DSA layer runs it in its
    eager break: the mixin's own ``forward_dsa_core`` (the narrowing of a
    padded replay, the index-K write, the prefill top-k with its KVP
    gates and collectives), around an indexer that reads the index query from
    ``q_norm`` and the index key and head weights from ``hidden_states``. The
    sparse core appends each forward's top-k to ``records`` and outputs the
    first query columns."""
    from tokenspeed.runtime.models.glm5 import (
        GlmDsaIndexerOutput,
        GlmDsaSparseAttentionMixin,
    )

    class Indexer:
        def __init__(self):
            self.weights_softmax_scale = 0.1

        def __call__(self, hidden_states, q_norm, positions):
            return GlmDsaIndexerOutput(
                query=q_norm.view(-1, _HEADS, _DIM),
                key=hidden_states[:, :_DIM].contiguous(),
                weights=hidden_states[:, _DIM:].contiguous(),
            )

    class Layer(GlmDsaSparseAttentionMixin):
        def __init__(self):
            self.attn_mqa = _AttentionLayer(layer_id=3)
            self.indexer = Indexer()
            self.index_topk = _TOPK
            self.skip_indexer_topk = False
            self.is_nextn = False
            self.num_local_heads = 1
            self.v_head_dim = _OUT_COLUMNS
            self.records = records

        def q_b_proj(self, q_norm):
            return q_norm[:, :_OUT_COLUMNS], None

        def forward_dsa_sparse_prefill(
            self,
            positions,
            q,
            latent_cache,
            ctx,
            out_cache_loc,
            output,
            *,
            prefill_topk,
        ):
            self.records.append(prefill_topk)
            output.copy_(q)
            return output

    return Layer()


def _padded_replay_worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.execution.breakable_cuda_graph import (
        BreakableCapture,
        active_forward,
        break_point,
    )
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement
    from tokenspeed.runtime.models import glm5
    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=300),
    )
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        comm_backend = AutoBackend()
        # PyNccl has no all-to-all, so the row-owner exchange uses torch.distributed.
        comm_backend.configure(use_pynccl=False)
        set_global_backend(comm_backend)
        # The live prefills take several tiles and exchanges, the capture batch one.
        budget_mb = _PREFILL_BUDGET >> 20
        global_server_args_dict[glm5._INDEXER_PREFILL_MAX_LOGITS_MB_ARG] = budget_mb

        # The live requests' blocks, about 70% of the longest one's on rank 0.
        generator = torch.Generator().manual_seed(4700 + degree)
        seq_lens = [prefix + new for prefix, new in _PREFILL]
        placed, virtual_blocks = _place_blocks(seq_lens, degree, generator)
        table = _page_table(placed, 2 * max(map(len, placed)))
        placement = CachePlacement(_BLOCK, virtual_blocks, group, rank)
        torch.manual_seed(4800 + degree)
        rows = sum(new for _, new in _PREFILL)
        keys = torch.randn(
            virtual_blocks * _BLOCK, _DIM, device=device, dtype=torch.bfloat16
        )
        q = torch.randn(rows, _HEADS, _DIM, device=device, dtype=torch.bfloat16)
        weights = torch.randn(rows, _HEADS, device=device, dtype=torch.bfloat16)
        for tensor in (keys, q, weights):
            dist.broadcast(tensor, 0)
        # PrefillGraph.make_dummy_batch's batch: every page on block 1, owned by rank 0.
        capture_prefill = ((0, _CAPTURE_ROWS),)
        capture_table = _page_table(
            [[1] * (_CAPTURE_ROWS // _BLOCK)], _CAPTURE_ROWS // _PAGE
        )
        capture_causal = _prefill_rows(capture_prefill, device)[1]
        visible = torch.arange(_TOPK, dtype=torch.int32, device=device)
        every_token = torch.where(visible < capture_causal.unsqueeze(1), visible, -1)
        # Live prefills above the row-owner gate, then below it (merged on every rank).
        batches = (
            (_PREFILL, table, 0),
            (_PREFILL[1:], table[1:], _PREFILL[0][1]),
        )
        assert rows >= glm5._KVP_PREFILL_ROW_OWNER_MIN_ROWS > rows - _PREFILL[0][1]

        records = []
        layer = _k31_dsa_layer(records)
        comm = _IdentityAttnComm()

        @break_point
        def sparse_attention(positions, hidden_states, q_norm, latent_cache, ctx):
            # KimiK31SparseMLAAttention._sparse_attention's DSA work.
            return layer.forward_dsa_core(
                positions, hidden_states, q_norm, latent_cache, ctx, comm
            )

        def forward(positions, hidden_states, q_norm, latent_cache, ctx):
            # The first break's handoff is consumed before the second break reuses it.
            hidden_states = hidden_states * 1
            q_norm = q_norm * 1
            first = sparse_attention(
                positions, hidden_states, q_norm, latent_cache, ctx
            )
            first = first * 2
            second = sparse_attention(
                positions, hidden_states, q_norm, latent_cache, ctx
            )
            return first + second

        def extend_ctx(backend, pool, prefill):
            bs = len(prefill)
            return ForwardContext(
                attn_backend=backend,
                token_to_kv_pool=pool,
                bs=bs,
                num_extends=bs,
                input_num_tokens=sum(new for _, new in prefill),
                forward_mode=ForwardMode.EXTEND,
                output_layout=ForwardOutputLayout(bs, bs, 0, 1),
            )

        # The query layout acts in the DSA leaf's sparse core, which this layer replaces.
        selections = {}
        for owned_width in (True, False):
            backend = _KvpPrefillBackend(placement, owned_width)
            pool = _IndexKPool(_plane(keys, virtual_blocks, placement, device))
            # The bucket's static inputs, holding the capture batch's rows.
            static_positions = torch.arange(_CAPTURE_ROWS, device=device)
            static_hidden = torch.randn(
                _CAPTURE_ROWS, _DIM + _HEADS, device=device, dtype=torch.bfloat16
            )
            static_q_norm = torch.randn(
                _CAPTURE_ROWS, _HEADS * _DIM, device=device, dtype=torch.bfloat16
            )
            static_latent = torch.zeros(
                _CAPTURE_ROWS, _OUT_COLUMNS, device=device, dtype=torch.bfloat16
            )
            for tensor in (static_hidden, static_q_norm):
                dist.broadcast(tensor, 0)

            # Capture as PrefillGraph._capture_bucket does, then replay once unpadded.
            backend.begin(capture_table, capture_prefill, device)
            dummy = extend_ctx(backend, pool, capture_prefill)
            capture = BreakableCapture()
            capture.stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                with torch.cuda.stream(capture.stream):
                    forward(
                        static_positions,
                        static_hidden,
                        static_q_norm,
                        static_latent,
                        dummy,
                    )
                torch.cuda.synchronize()
                dist.barrier(group=coordination)
                records.clear()
                with active_forward(dummy):
                    with capture:
                        output = forward(
                            static_positions,
                            static_hidden,
                            static_q_norm,
                            static_latent,
                            dummy,
                        )
                    capture.replay()
            torch.cuda.synchronize()
            assert capture.num_segments == 5
            # Both layers, both passes: every row selects all its tokens, on rank 0.
            assert len(records) == 4
            for record in records:
                torch.testing.assert_close(
                    record.topk_lens, capture_causal, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    record.workspace_indices.sort(dim=-1).values,
                    every_token.sort(dim=-1).values,
                    rtol=0,
                    atol=0,
                )
                if owned_width:
                    owned = record.dcp_index.local_page_table >= 0
                    assert bool(owned.all()) if rank == 0 else not bool(owned.any())
                    assert record.dcp_index.columns == _CAPTURE_ROWS // _PAGE
                    assert not bool(record.dcp_index.pages.overflow.any())
                else:
                    assert record.dcp_index is None
            # The capture batch's index-K writes went to block 1's pages.
            pool.plane = _plane(keys, virtual_blocks, placement, device)

            selections[owned_width] = []
            for prefill, batch_table, first_row in batches:
                live_rows = sum(new for _, new in prefill)
                rows_of_q = slice(first_row, first_row + live_rows)
                slots = _extend_slots(batch_table, prefill).to(device)
                positions = torch.cat(
                    [torch.arange(p, p + n, device=device) for p, n in prefill]
                )
                hidden = torch.cat((keys[slots], weights[rows_of_q]), dim=1)
                q_norm = q[rows_of_q].reshape(live_rows, -1)
                latent = torch.zeros_like(static_latent[:live_rows])
                causal = _prefill_rows(prefill, device)[1]

                # Pad as PrefillGraph.replay: filler past live rows, bucket token count.
                backend.begin(batch_table, prefill, device)
                live = extend_ctx(backend, pool, prefill)
                static_positions[:live_rows].copy_(positions)
                static_positions[live_rows:].zero_()
                static_hidden[:live_rows].copy_(hidden)
                static_hidden[live_rows:].fill_(1e4)
                static_q_norm[:live_rows].copy_(q_norm)
                static_q_norm[live_rows:].fill_(1e4)
                padded = dataclasses.replace(live, input_num_tokens=_CAPTURE_ROWS)
                records.clear()
                with torch.no_grad(), active_forward(padded):
                    capture.replay(valid_rows=live_rows)
                torch.cuda.synchronize()
                replayed, replayed_output = list(records), output.clone()

                # The same layers eagerly over the live rows.
                backend.begin(batch_table, prefill, device)
                records.clear()
                with torch.no_grad():
                    eager_output = forward(
                        positions,
                        hidden,
                        q_norm,
                        latent,
                        extend_ctx(backend, pool, prefill),
                    )
                torch.cuda.synchronize()
                eager = list(records)

                assert len(replayed) == len(eager) == 2
                for got, expected in zip(replayed, eager, strict=True):
                    _assert_same_sets(
                        (got.workspace_indices, got.topk_lens),
                        (expected.workspace_indices, expected.topk_lens),
                        causal,
                    )
                torch.testing.assert_close(
                    replayed_output[:live_rows], eager_output, rtol=0, atol=0
                )
                assert not bool(replayed_output[live_rows:].any())
                if owned_width:
                    # Planned once at the owned width; the next layer reuses it.
                    index = replayed[0].dcp_index
                    width = -(-max(p + n for p, n in prefill) // _PAGE)
                    assert replayed[1].dcp_index is index
                    assert index.columns == min(backend.prefill_index_columns(), width)
                    assert not bool(index.pages.overflow.any())
                else:
                    assert replayed[0].dcp_index is None
                selections[owned_width].append((replayed, causal))
            print(
                f"rank {rank}/{degree}: owned_width={owned_width} padded replays "
                "matched eager",
                flush=True,
            )

        # The owned width keeps every row's selection.
        on, off = selections.values()
        for (on_layers, causal), (off_layers, _) in zip(on, off, strict=True):
            for got, expected in zip(on_layers, off_layers, strict=True):
                _assert_same_sets(
                    (got.workspace_indices, got.topk_lens),
                    (expected.workspace_indices, expected.topk_lens),
                    causal,
                )
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4])
def test_k31_kvp_prefill_padded_replay_matches_eager(tmp_path, degree):
    """Two Kimi-K3.1 DSA layers' breaks captured in a breakable prefill graph
    over a 2048-row bucket from the capture batch, whose pages all sit on
    rank 0's block, then replayed padded for live prefills above and below
    the row-owner merge's row gate. At the owned width and at the table's,
    the capture batch selects every visible token, and each replay selects
    what the same layers select eagerly over the live rows."""
    if (
        torch.version.cuda is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < degree
    ):
        pytest.skip(f"requires {degree} CUDA GPUs")
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_hopper_plus:
        pytest.skip("the DeepGEMM index scorer requires NVIDIA Hopper or newer")
    mp.spawn(
        _padded_replay_worker,
        args=(degree, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=degree,
        join=True,
    )
