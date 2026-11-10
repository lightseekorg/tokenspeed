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

"""DCP candidate and attention-partial exchange with real two-GPU collectives."""

import sys
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="runtime-2gpu")


def _candidate_inputs(rank, rows, width, phase):
    positions = torch.arange(rows * width, dtype=torch.int32).reshape(rows, width)
    offsets = positions + 2**24 + 1 + rank * 100_000
    scores = positions.float() / 128 + rank * 1000
    bits = scores.view(torch.int32)
    bits[0, :6] = torch.tensor(
        [0, -(2**31), 2139095040, -8388608, 2143289345, -4194302], dtype=torch.int32
    )
    offsets[0, -1] = 2**31 - 1 - rank
    if rows > 1:
        offsets[1].fill_(-1)
        if rank == 0:
            offsets[1, :2] = torch.tensor([7, 11])
    if rows > 2:
        offsets[-1].fill_(-1)
    if phase == "empty" or (phase == "concentrated" and rank != 0):
        offsets.fill_(-1)
        scores.fill_(-torch.inf)
    return offsets, scores


def _check_candidates(result, counts, parts, topk):
    offsets = torch.cat([part[0] for part in parts], dim=1)
    scores = torch.cat([part[1] for part in parts], dim=1)
    valid = (offsets >= 0) & ~torch.isnan(scores) & (scores > -torch.inf)
    expected = torch.topk(torch.where(valid, scores, -torch.inf), topk, dim=1).values
    result, counts = result.cpu(), counts.cpu()
    torch.testing.assert_close(
        counts, valid.sum(1).clamp_max(topk).int(), atol=0, rtol=0
    )
    for row in range(offsets.shape[0]):
        lookup = dict(
            zip(offsets[row, valid[row]].tolist(), scores[row, valid[row]].tolist())
        )
        chosen = result[row, : int(counts[row])].tolist()
        assert len(chosen) == len(set(chosen))
        assert (result[row, int(counts[row]) :] == -1).all()
        values = torch.tensor(
            [lookup[item] for item in chosen] + [-torch.inf] * (topk - len(chosen))
        )
        torch.testing.assert_close(
            values.sort(descending=True).values, expected[row], atol=0, rtol=0
        )


def _candidate_worker(rank, degree, rendezvous, use_pynccl):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.indexer import merge_index_candidates

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    graph = None
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        backend.configure(use_pynccl=use_pynccl)
        set_global_backend(backend)
        resources = backend.nccl._get_or_create_resources(group)
        communicator = resources["pynccl_comm"]
        if use_pynccl:
            assert (
                communicator is not None and communicator.available
            ), "PyNCCL initialization failed"
        context = (
            communicator.change_state(
                enable=True, stream=torch.cuda.current_stream(device)
            )
            if use_pynccl
            else nullcontext()
        )
        with context:
            for rows, topk in ((1, 512), (3, 2048)):
                offsets = torch.empty(
                    (rows, topk * 2), dtype=torch.int32, device=device
                )[:, ::2]
                scores = torch.empty(
                    (rows, topk * 2), dtype=torch.float32, device=device
                )[:, ::2]

                def refresh(phase):
                    data = _candidate_inputs(rank, rows, topk, phase)
                    offsets.copy_(data[0])
                    scores.copy_(data[1])
                    torch.cuda.synchronize()
                    dist.barrier(group=coordination)

                refresh("initial")
                for _ in range(3):
                    eager = merge_index_candidates(
                        offsets, scores, topk=topk, group=group
                    )
                _check_candidates(
                    *eager,
                    [_candidate_inputs(r, rows, topk, "initial") for r in group],
                    topk,
                )
                torch.cuda.synchronize()
                dist.barrier(group=coordination)
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream()
                # PyNCCL uses an explicit stream, including during capture.
                capture_context = (
                    communicator.change_state(enable=True, stream=capture_stream)
                    if use_pynccl
                    else nullcontext()
                )
                with capture_context, torch.cuda.graph(graph, stream=capture_stream):
                    selected, counts = merge_index_candidates(
                        offsets, scores, topk=topk, group=group
                    )
                for phase in ("initial", "empty", "concentrated", "initial"):
                    refresh(phase)
                    graph.replay()
                    _check_candidates(
                        selected,
                        counts,
                        [_candidate_inputs(r, rows, topk, phase) for r in group],
                        topk,
                    )
                graph.reset()
                graph = None
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        # NCCL shutdown waits for communicators retained by captured graphs.
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize("use_pynccl", [False, True])
def test_packed_candidates_with_real_collectives_and_graph(tmp_path, use_pynccl):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(_candidate_worker, args=(2, rendezvous, use_pynccl), nprocs=2, join=True)


_OWNER_TOPK = 512
# A row's FP32 logits take 24 KiB: three rows a tile, two tiles a row-owner exchange.
_OWNER_COLUMNS = 96
_OWNER_BUDGET = 3 * _OWNER_COLUMNS * 64 * 4


def _owner_candidates(rank, degree, causal_lens, active, topk):
    """``rank``'s local candidates of the rows ``causal_lens`` names.

    Positions are distinct across rows and ranks, and so are a row's scores,
    which interleave the ranks' candidates in a row-dependent order. Rows
    not divisible by three have fewer valid candidates than ``topk``; every
    fifth row holds a mandatory (+inf) candidate on rank 0. A zero
    ``active`` leaves the rank without candidates.
    """
    row = causal_lens.to(torch.int32).unsqueeze(1)
    column = torch.arange(topk, device=causal_lens.device, dtype=torch.int32)
    count = torch.where(row % 3 == 0, topk, (row * 7 + rank * 3) % topk)
    valid = (column < count) & (active != 0)
    offsets = torch.where(valid, (row * topk + column) * degree + rank, -1)
    # An odd stride permutes the columns of a power-of-two topk.
    scores = (((column * 37 + row) % topk) * degree + rank).float()
    if rank == 0:
        scores = torch.where((row % 5 == 0) & (column == 0), torch.inf, scores)
    return offsets, torch.where(valid, scores, -torch.inf)


def _owner_scorer(
    rank, degree, active, q, weights, index_k_cache, table, requests, causal, **kw
):
    """``dsa_index_candidates`` stand-in returning ``_owner_candidates``."""
    offsets, scores = _owner_candidates(rank, degree, causal, active, kw["topk"])
    # Interleaved records, as the GPU candidate leaves emit them.
    records = torch.stack((offsets, scores.view(torch.int32)), dim=-1)
    return records[..., 0], records[..., 1].view(torch.float32)


def _row_owner_worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
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
        timeout=timedelta(seconds=90),
    )
    graph = None
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        # PyNccl has no all-to-all: the exchange goes through torch.distributed.
        backend.configure(use_pynccl=False)
        set_global_backend(backend)
        # Synthetic candidates replace the scorer so the exchange runs on any GPU.
        active = torch.ones((), dtype=torch.int32, device=device)
        indexer.dsa_index_candidates = partial(_owner_scorer, rank, degree, active)
        placement = CachePlacement(64, _OWNER_COLUMNS + 1, group, rank)
        table = torch.arange(1, _OWNER_COLUMNS + 1, dtype=torch.int32, device=device)
        table = table.unsqueeze(0)
        rows = 17
        causal = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        requests = torch.zeros(rows, dtype=torch.int32, device=device)
        q = torch.zeros(rows, 1, 1, device=device)
        weights = torch.zeros(rows, 1, device=device)
        cache = torch.zeros(1, device=device)

        def select(count, budget, row_owner):
            return indexer.select_dsa_topk(
                q[:count],
                weights[:count],
                cache,
                table,
                requests[:count],
                causal[:count],
                placement=placement,
                page_size=64,
                topk=_OWNER_TOPK,
                softmax_scale=0.1,
                initial_tokens=0,
                local_tokens=0,
                max_logits_bytes=budget,
                row_owner=row_owner,
            )

        def check(result, count, budget, active_ranks):
            # Distinct scores: the row-owner merge selects as replicated, in order.
            replicated = select(count, budget, False)
            for actual, expected in zip(result, replicated, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            cpu_causal = causal[:count].cpu()
            parts = []
            for source in group:
                flag = torch.tensor(int(source in active_ranks))
                parts.append(
                    _owner_candidates(source, degree, cpu_causal, flag, _OWNER_TOPK)
                )
            _check_candidates(*result, parts, _OWNER_TOPK)

        # Budgeted: exchanges of 6, 6, 5 rows; else 17 padded, 16 even, 1 (rank 1 pads).
        tile_rows = indexer._tile_rows(rows, _OWNER_COLUMNS, 64, _OWNER_BUDGET)
        exchange_rows = indexer._exchange_rows(
            rows, tile_rows, degree, _OWNER_TOPK, _OWNER_BUDGET, row_owner=True
        )
        assert (tile_rows, exchange_rows) == (3, 6)
        for count, budget in ((17, _OWNER_BUDGET), (17, None), (16, None), (1, None)):
            for _ in range(2):
                result = select(count, budget, True)
            check(result, count, budget, group)
        torch.cuda.synchronize()
        dist.barrier(group=coordination)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = select(rows, _OWNER_BUDGET, True)
        original = causal.clone()
        # New rows; all candidates on one rank (as at capture); on none; captured rows.
        for shift, active_ranks in ((rows, group), (5, (0,)), (0, ()), (0, group)):
            causal.copy_(original + shift)
            active.fill_(int(rank in active_ranks))
            torch.cuda.synchronize()
            dist.barrier(group=coordination)
            graph.replay()
            check(captured, rows, _OWNER_BUDGET, active_ranks)
        graph.reset()
        graph = None
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        # NCCL shutdown waits for communicators retained by captured graphs.
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


def test_row_owner_candidates_with_real_collectives(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    mp.spawn(
        _row_owner_worker,
        args=(2, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=2,
        join=True,
    )


def _partial_inputs(rank, degree, dtype, phase):
    generator = torch.Generator().manual_seed(710 + rank)
    output = torch.randn(3, degree * 2, 63, generator=generator).to(dtype)
    lse = torch.randn(3, degree * 2, generator=generator) * 3
    lse[1] = -torch.inf
    output[1] = torch.nan
    if rank == 0:
        lse[0, 0] = torch.inf
        output[0, 0] = torch.nan
    if phase == "empty" or (phase == "concentrated" and rank != 0):
        lse.fill_(-torch.inf)
        output.fill_(torch.nan)
    return output, lse


def _check_partials(actual, rank, degree, dtype, phase, sink, keep_all_heads):
    parts = [_partial_inputs(source, degree, dtype, phase) for source in range(degree)]
    lses = torch.stack([part[1] for part in parts]).double()
    lses = torch.where(torch.isposinf(lses), -torch.inf, lses)
    maximum = lses.max(0).values
    mass = (lses - torch.where(torch.isneginf(maximum), 0, maximum)).exp()
    denominator = mass.sum(0)
    weights = mass / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    outputs = torch.stack([part[0] for part in parts]).double()
    outputs = torch.where(torch.isneginf(lses)[..., None], 0, outputs)
    expected = (outputs * weights[..., None]).sum(0)
    if not keep_all_heads:
        expected = expected[:, rank * 2 : (rank + 1) * 2]
    if sink is not None:
        lse = torch.where(denominator == 0, -torch.inf, maximum + denominator.log())
        lse = lse[:, rank * 2 : (rank + 1) * 2]
        factor = torch.where(torch.isneginf(lse), 0, 1 / (1 + (sink.cpu() - lse).exp()))
        expected *= factor[..., None]
    actual = actual.cpu()
    # Include the final dtype's rounding error after the FP32 combine.
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2e-6,
        rtol=torch.finfo(dtype).eps / 2 + 3e-6,
    )
    assert actual.is_contiguous()


def _partial_worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.comm import combine_attention_partials
    from tokenspeed.runtime.utils.env import global_server_args_dict

    # The packed LSE gather can select RSAG, whose workspace uses the same
    # resolved capacity bounds as a serving worker, even in an NCCL test.
    global_server_args_dict.update(
        mapping=Mapping(rank=rank, world_size=degree, attn_tp_size=degree),
        chunked_prefill_size=16,
        max_prefill_tokens=16,
        max_model_len=1024,
        max_num_seqs=4,
        speculative_algorithm=None,
    )

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    graph = None
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        backend.configure(use_pynccl=False)
        set_global_backend(backend)
        cases = [
            (torch.bfloat16, False, False),
            (torch.float16, False, False),
            (torch.float32, False, False),
            (torch.bfloat16, True, False),
            (torch.bfloat16, False, True),
        ]
        for dtype, use_sink, keep_all_heads in cases:
            output, lse = [
                x.to(device) for x in _partial_inputs(rank, degree, dtype, "normal")
            ]
            sink = torch.tensor([-1.0, 0.5], device=device) if use_sink else None

            def combine():
                return combine_attention_partials(
                    output,
                    lse,
                    group=group,
                    rank=rank,
                    sink=sink,
                    keep_all_heads=keep_all_heads,
                )

            for _ in range(2):
                result = combine()
            _check_partials(result, rank, degree, dtype, "normal", sink, keep_all_heads)
            if not keep_all_heads:
                # Records packed into the exchange's lent input combine alike.
                lent = combine_attention_partials(
                    output,
                    lse,
                    group=group,
                    rank=rank,
                    sink=sink,
                    keep_all_heads=False,
                    lend_records=True,
                )
                assert torch.equal(lent.view(torch.uint8), result.view(torch.uint8))
            torch.cuda.synchronize()
            dist.barrier(group=coordination)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = combine()
            for phase in ("empty", "concentrated", "normal"):
                values, masses = _partial_inputs(rank, degree, dtype, phase)
                output.copy_(values)
                lse.copy_(masses)
                torch.cuda.synchronize()
                dist.barrier(group=coordination)
                graph.replay()
                _check_partials(
                    result, rank, degree, dtype, phase, sink, keep_all_heads
                )
            graph.reset()
            graph = None
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


def test_partial_exchange_with_real_collectives_and_graph(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    mp.spawn(
        _partial_worker,
        args=(2, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=2,
        join=True,
    )


# K3.1's sparse prefill under KVP, on 576-wide FP8 latent rows.
_RECORDS_HEADS = 12
_RECORDS_BLOCK = 128
_RECORDS_TOPK = 2048


def _records_virtual_blocks(degree):
    return 1 + 12 * degree


def _records_selection(degree, tokens, device):
    """The selection every rank shares (the indexer's global top-k) over the
    group's virtual slots: an empty first row, stale slots past each length."""
    generator = torch.Generator().manual_seed(4242 + tokens)
    capacity = _records_virtual_blocks(degree) * _RECORDS_BLOCK
    slots = torch.randint(
        _RECORDS_BLOCK,
        capacity,
        (tokens, _RECORDS_TOPK),
        generator=generator,
        dtype=torch.int32,
    )
    lengths = torch.randint(
        0, _RECORDS_TOPK + 1, (tokens,), generator=generator, dtype=torch.int32
    )
    lengths[0] = 0
    return slots.to(device), lengths.to(device)


def _records_values(degree, device):
    """Every virtual slot's BF16 latent row, before the caches' FP8."""
    generator = torch.Generator().manual_seed(77)
    values = torch.randn(
        (_records_virtual_blocks(degree) * _RECORDS_BLOCK, 576), generator=generator
    )
    return values.to(device=device, dtype=torch.bfloat16)


def _records_cache(degree, rank, device):
    """This rank's FP8 cache: the owned virtual blocks at their local slots."""
    from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
        virtual_slots_to_local,
    )

    virtual = _records_virtual_blocks(degree)
    values = _records_values(degree, device)
    local, owned = virtual_slots_to_local(
        torch.arange(values.shape[0], device=device),
        rows_per_page=_RECORDS_BLOCK,
        virtual_block_count=virtual,
        degree=degree,
        rank=rank,
    )
    rows = ((virtual - 2) // degree + 2) * _RECORDS_BLOCK
    cache = torch.zeros((rows, 576), device=device, dtype=torch.bfloat16)
    cache[local[owned]] = values[owned]
    return cache.to(torch.float8_e4m3fn)


def _records_leaf(dsa, group, rank, query_layout):
    """The DSA leaf's head-sharded KVP prefill state on an FP8 cache."""
    leaf = object.__new__(dsa.DSABackend)
    leaf.kernel_page_size = 64
    # Pinned to the Triton core, as on Hopper, so Blackwell does not take trtllm-gen.
    leaf.kernel_solution = "triton"
    leaf.prefill_kernel_plan = None
    leaf.slot_order = "selection"
    leaf.data_type = torch.float8_e4m3fn
    leaf.kv_lora_rank = 512
    leaf.qk_nope_head_dim = 128
    leaf.qk_rope_head_dim = 64
    leaf.kvp_group = group
    leaf.kvp_rank = rank
    leaf.kvp_block_granularity = _RECORDS_BLOCK
    leaf.kvp_virtual_block_count = _records_virtual_blocks(len(group))
    leaf.query_shard_metadata = None
    leaf.step_counter = None
    leaf.num_attention_heads = _RECORDS_HEADS * len(group)
    leaf.num_local_heads = _RECORDS_HEADS
    leaf.kvp_prefill_query_layout = query_layout
    # What configure_runtime's probe picks on an FP8 cache: the Triton records core.
    leaf._dcp_records_core = True
    leaf._dcp_padded_query = False
    return leaf


def _records_worker(rank, degree, rendezvous, use_pynccl):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from types import SimpleNamespace

    from tokenspeed_kernel.platform import current_platform

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.backends.paged import dsa
    from tokenspeed.runtime.layers.attention.dcp.comm import (
        gather_query_heads,
        gather_query_heads_lent,
        gather_query_heads_padded,
        gather_query_heads_rank_major,
    )

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        backend.configure(use_pynccl=use_pynccl)
        set_global_backend(backend)
        communicator = backend.nccl._get_or_create_resources(group)["pynccl_comm"]
        if use_pynccl:
            assert (
                communicator is not None and communicator.available
            ), "PyNCCL initialization failed"
        context = (
            communicator.change_state(
                enable=True, stream=torch.cuda.current_stream(device)
            )
            if use_pynccl
            else nullcontext()
        )
        # The query-layout arm must run, not only agree: count its core's launches.
        records_core = dsa.dsa_prefill_dcp_records
        launches = []

        def counted_records_core(**kwargs):
            launches.append(tuple(kwargs["q"].shape))
            return records_core(**kwargs)

        dsa.dsa_prefill_dcp_records = counted_records_core
        cache = _records_cache(degree, rank, device)
        pool = SimpleNamespace(quant_method=None, get_key_buffer=lambda layer_id: cache)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=_RECORDS_HEADS,
            head_dim=576,
            v_head_dim=512,
            scaling=192**-0.5,
            logit_cap=0.0,
        )
        with context:
            # 1037 rows use the records core; 5 split keys and use packed partials.
            for tokens, records_arm in ((1037, True), (5, False)):
                slots, lengths = _records_selection(degree, tokens, device)
                generator = torch.Generator().manual_seed(900 + rank)
                q = torch.randn((tokens, _RECORDS_HEADS, 576), generator=generator)
                q = q.to(device=device, dtype=torch.bfloat16)
                outputs = []
                for query_layout in (False, True):
                    launches.clear()
                    leaf = _records_leaf(dsa, group, rank, query_layout)
                    outputs.append(
                        leaf.forward_sparse_prefill(
                            q=q,
                            layer=layer,
                            token_to_kv_pool=pool,
                            kv_seq_lens=None,
                            topk_slots=slots,
                            topk_lens=lengths,
                            max_seq_len=cache.shape[0] * degree,
                        )
                    )
                    ran = query_layout and records_arm
                    shape = (degree, tokens, _RECORDS_HEADS, 576)
                    assert launches == ([shape] if ran else [])
                off, on = outputs
                assert off.shape == (tokens, _RECORDS_HEADS * 512)
                # Bit for bit, signed zeros included.
                assert torch.equal(off.view(torch.int16), on.view(torch.int16))
                assert not off[0].any()
            # The rank-major gather holds the token-major gather's bytes.
            generator = torch.Generator().manual_seed(500 + rank)
            q = torch.randn((7, _RECORDS_HEADS, 576), generator=generator)
            q = q.to(device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            rank_major = gather_query_heads_rank_major(q, group).view(torch.uint8)
            token_major = gather_query_heads(q, group).view(torch.uint8)
            assert torch.equal(
                rank_major.permute(1, 0, 2, 3).reshape(token_major.shape),
                token_major,
            )
            # Through the exchange's lent input: the same bytes.
            lent = gather_query_heads_lent(q, group)
            assert lent.dtype == q.dtype
            assert torch.equal(lent.view(torch.uint8), token_major)
            # 48 heads x 2 ranks = 96, like K3.1's 12 x KVP8; trtllm-gen runs 96 as 128.
            wide_heads = 48
            gathered = wide_heads * degree
            q = torch.randn((7, wide_heads, 576), generator=generator)
            q = q.to(device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            padded = gather_query_heads_padded(q, group, 128).view(torch.uint8)
            assert padded.shape == (7, 128, 576)
            token_major = gather_query_heads(q, group).view(torch.uint8)
            assert torch.equal(padded[:, :gathered], token_major)
            assert not padded[:, gathered:].any()
            if current_platform().is_blackwell:
                # trtllm-gen: padded query layout matches the leaf's padding bitwise.
                wide = SimpleNamespace(**{**vars(layer), "tp_q_head_num": wide_heads})
                slots, lengths = _records_selection(degree, 1037, device)
                q = torch.randn((1037, wide_heads, 576), generator=generator)
                q = q.to(device=device, dtype=torch.bfloat16)
                outputs = []
                for query_layout in (False, True):
                    leaf = _records_leaf(dsa, group, rank, query_layout)
                    leaf.kernel_solution = None
                    leaf.num_attention_heads = gathered
                    leaf.num_local_heads = wide_heads
                    leaf._dcp_records_core = False
                    leaf._dcp_padded_query = True
                    outputs.append(
                        leaf.forward_sparse_prefill(
                            q=q,
                            layer=wide,
                            token_to_kv_pool=pool,
                            kv_seq_lens=None,
                            topk_slots=slots,
                            topk_lens=lengths,
                            max_seq_len=cache.shape[0] * degree,
                        )
                    )
                off, on = outputs
                assert off.shape == (1037, wide_heads * 512)
                assert torch.equal(off.view(torch.int16), on.view(torch.int16))
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("use_pynccl", [False, True])
def test_rank_major_core_matches_packed_partials(tmp_path, use_pynccl):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    mp.spawn(
        _records_worker,
        args=(2, f"file://{tmp_path / 'rendezvous'}", use_pynccl),
        nprocs=2,
        join=True,
    )


def _decode_records_leaf(dsa, group, rank, requests, records_core):
    """The DSA leaf's head-sharded KVP decode state on an FP8 cache, verifying
    K3.1's 4 rows per request (the 3.1 drafter's three tokens)."""
    from types import SimpleNamespace

    leaf = _records_leaf(dsa, group, rank, query_layout=False)
    leaf.kernel_solution = None
    leaf.decode_kernel_plan = None
    leaf.index_topk = _RECORDS_TOPK
    leaf.max_context_len = _records_virtual_blocks(len(group)) * _RECORDS_BLOCK
    leaf.is_draft = False
    leaf.spec_num_tokens = 4
    # configure_runtime's probe on gfx950
    # (test_dcp_decode_records_core_on_the_real_k31_selection), or its
    # verdict where another core decodes.
    leaf._dcp_decode_records_core = records_core
    leaf._owned_decode_topk = True
    leaf._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens_k=torch.full(
                (requests,), leaf.max_context_len, dtype=torch.int32, device="cuda"
            ),
            max_seq_len_k=leaf.max_context_len,
            _dsa_kv_seq_lens=None,
        )
    )
    return leaf


def _assert_attends_live_slots(output, q, slots, lengths, degree, scaling):
    """A decode's TP-local heads against FP32 attention over each row's live
    virtual slots, the query and cache in the FP8 the decode cores take,
    within what their FP8 probabilities cost
    (test_gluon_dsa_decode_dcp_records' bounds)."""
    keys = _records_values(degree, q.device).to(torch.float8_e4m3fn).float()
    queries = q.to(torch.float8_e4m3fn).float()
    expected = torch.zeros((q.shape[0], q.shape[1], 512), device=q.device)
    for row in range(q.shape[0]):
        live = slots[row, : int(lengths[row])].long()
        if live.numel():
            scores = queries[row] @ keys[live].T * scaling
            expected[row] = torch.softmax(scores, dim=-1) @ keys[live, :512]
    error = output.float().view_as(expected) - expected
    assert error.abs().max() < 0.15
    assert error.square().mean().sqrt() <= 0.02 * expected.square().mean().sqrt()


def _decode_records_worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from types import SimpleNamespace

    from tokenspeed_kernel.ops.communication._iris import (
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel.ops.communication._iris.pull_exchange import (
        IrisPullExchange,
    )

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.backends.paged import dsa
    from tokenspeed.runtime.layers.attention.dcp.comm import (
        combine_attention_partials,
        gather_query_heads,
        gather_query_heads_lent,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import (
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
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=300))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        backend.configure(use_pynccl=True)
        set_global_backend(backend)
        communicator = backend.nccl._get_or_create_resources(group)["pynccl_comm"]
        context = communicator.change_state(
            enable=True, stream=torch.cuda.current_stream(device)
        )
        # The records arm must run, not only agree: count its core's launches.
        records_core = dsa.dsa_decode_dcp_records
        launches = []

        def counted_records_core(**kwargs):
            launches.append(tuple(kwargs["q"].shape))
            return records_core(**kwargs)

        dsa.dsa_decode_dcp_records = counted_records_core
        cache = _records_cache(degree, rank, device)
        pool = SimpleNamespace(quant_method=None, get_key_buffer=lambda layer_id: cache)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=_RECORDS_HEADS,
            head_dim=576,
            v_head_dim=512,
            scaling=192**-0.5,
            logit_cap=0.0,
        )
        with context:
            # Every launch reaches the records core. 4 requests' rows split
            # by their live keys, which rounds the FP8 probabilities against
            # other running maxima than the token-major decode, so they are
            # held to FP32 attention; 128 requests' rows fill the gfx950
            # core's grid in one split, as the token-major decode's do.
            for requests, exact in ((4, False), (128, True)):
                tokens = requests * 4
                slots, lengths = _records_selection(degree, tokens, device)
                generator = torch.Generator().manual_seed(900 + rank)
                q = torch.randn((tokens, _RECORDS_HEADS, 576), generator=generator)
                q = q.to(device=device, dtype=torch.bfloat16)
                outputs = []
                for probed in (False, True):
                    launches.clear()
                    leaf = _decode_records_leaf(dsa, group, rank, requests, probed)
                    outputs.append(
                        leaf.forward_sparse_decode(
                            q=q,
                            layer=layer,
                            token_to_kv_pool=pool,
                            bs=requests,
                            topk_indices=slots,
                            topk_lens=lengths,
                        )
                    )
                    shape = (degree, tokens, _RECORDS_HEADS, 576)
                    assert launches == ([shape] if probed else [])
                off, on = outputs
                assert off.shape == (tokens, _RECORDS_HEADS * 512)
                if exact:
                    # Bit for bit, signed zeros included.
                    assert torch.equal(off.view(torch.int16), on.view(torch.int16))
                else:
                    for output in outputs:
                        _assert_attends_live_slots(
                            output, q, slots, lengths, degree, layer.scaling
                        )
                assert not off[0].any() and not on[0].any()

                # The records core reads the query where an owned top-k's
                # score gather landed its rider, bit for bit.
                rider = leaf.dcp_decode_query_rider(layer, q)
                assert rider is not None
                gathered = rider.new_empty((degree, *rider.shape))
                dist.all_gather_into_tensor(gathered, rider)
                owned_slots, owned_lens = compact_owned_cache_slots(
                    slots, lengths, leaf.cache_placement(layer)
                )
                launches.clear()
                ridden = leaf.forward_sparse_decode(
                    q=q,
                    layer=layer,
                    token_to_kv_pool=pool,
                    bs=requests,
                    topk_indices=owned_slots,
                    topk_lens=owned_lens,
                    topk_owned=True,
                    gathered_query=gathered,
                )
                assert launches == [shape]
                assert torch.equal(on.view(torch.int16), ridden.view(torch.int16))

        # The drafter's shapes at K3.1's KVP8: 12 heads a rank, 576-wide FP8
        # queries, 512-wide BF16 partials, 16 rows (4 verify rows of 4 requests).
        rows, heads = 16, 12
        exchange = IrisPullExchange(
            IrisGroup(
                get_or_create_context(heap_size=1 << 28),
                dist.group.WORLD,
                rank,
                device,
            ),
            rows * heads * degree * (512 // 2 + 1) * 4,
        )
        exchange.peers.finish_prepare()
        generator = torch.Generator().manual_seed(40 + rank)
        q = torch.randn((rows, heads, 576), generator=generator)
        q = q.to(device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        output = torch.randn((rows, heads * degree, 512), generator=generator)
        output = output.to(device=device, dtype=torch.bfloat16)
        lse = (torch.randn((rows, heads * degree), generator=generator) * 3).to(device)
        lse[1] = -torch.inf
        output[1] = torch.nan
        # References through NCCL, before the pull exchange is installed.
        gathered = gather_query_heads(q, group)
        combined = combine_attention_partials(
            output, lse, group=group, rank=rank, sink=None, keep_all_heads=False
        )
        backend.iris.install_pull_exchange(group, exchange)

        def lent():
            return (
                gather_query_heads_lent(q, group),
                combine_attention_partials(
                    output,
                    lse,
                    group=group,
                    rank=rank,
                    sink=None,
                    keep_all_heads=False,
                    lend_records=True,
                ),
            )

        def check(query, merged):
            assert torch.equal(query.view(torch.uint8), gathered.view(torch.uint8))
            assert torch.equal(merged.view(torch.int16), combined.view(torch.int16))

        epoch = int(exchange.state[0])
        for _ in range(2):
            check(*lent())
        # Both pulled through the exchange.
        assert int(exchange.state[0]) > epoch
        epoch = int(exchange.state[0])
        # Queries and records it did not lend keep NCCL and its results.
        check(
            gather_query_heads(q, group),
            combine_attention_partials(
                output, lse, group=group, rank=rank, sink=None, keep_all_heads=False
            ),
        )
        assert int(exchange.state[0]) == epoch
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 8])
def test_rank_major_decode_core_matches_packed_partials(tmp_path, degree):
    if not torch.cuda.is_available() or torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    from tokenspeed_kernel.platform import ArchVersion, current_platform

    platform = current_platform()
    if not platform.is_amd or platform.arch_version != ArchVersion(9, 5):
        pytest.skip("the DSA decode records core registers on AMD gfx950 GPUs")
    mp.spawn(
        _decode_records_worker,
        args=(degree, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=degree,
        join=True,
    )
