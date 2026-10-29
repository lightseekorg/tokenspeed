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
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-2gpu")


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
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.comm import combine_attention_partials

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
