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

"""Exercise real CUDA graph events, overwritten inputs, and result ownership."""

import pytest
import torch

from tokenspeed.runtime.execution._draft_overlap import DecodeGraphOverlap

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_tail_uses_snapshot_and_completion_covers_both_streams():
    device = torch.device("cuda", torch.cuda.current_device())
    overlap = DecodeGraphOverlap()
    execution = torch.cuda.Stream()
    indices = torch.zeros(128, dtype=torch.int64, device=device)
    generation = torch.zeros(128, dtype=torch.int64, device=device)
    active = torch.ones(128, dtype=torch.bool, device=device)
    prefix = torch.zeros_like(generation)
    published = torch.full((256,), -1, dtype=torch.int64, device=device)
    graphs = {}
    execution.wait_stream(torch.cuda.current_stream())

    def body(rows):
        saved_indices = overlap.snapshot(indices[:rows])
        saved_active = overlap.snapshot(active[:rows])
        saved_generation = generation[:rows].clone()
        prefix[:rows].copy_(saved_generation)
        overlap.record_prefix()
        torch.cuda._sleep(100000)
        published.index_copy_(
            0, saved_indices, torch.where(saved_active, saved_generation, -1)
        )

    with torch.cuda.stream(execution):
        for rows in (1, 4, 128):
            body(rows)
        execution.synchronize()
        for rows in (1, 4, 128):
            graph = torch.cuda.CUDAGraph()
            with overlap.capture(graph), torch.cuda.graph(graph, stream=execution):
                body(rows)
            graphs[rows] = graph
        execution.synchronize()

    observations = []
    with torch.cuda.stream(execution):
        for step in range(12):
            rows = (1, 4, 128)[step % 3]
            slots = torch.arange(rows, device=device).flip(0) + (step % 2) * 128
            indices[:rows].copy_(slots)
            generation[:rows].copy_(torch.arange(rows, device=device) + step + 100)
            live_rows = 3 if rows == 4 else rows
            active[:rows].copy_(torch.arange(rows, device=device) < live_rows)
            overlap.replay(graphs[rows], live_rows=live_rows)
            # The next prepare is intentionally destructive before the graph tail.
            indices.fill_(12)
            active.fill_(False)
            generation.fill_(-100)
            host_prefix = torch.empty(rows, dtype=torch.int64, pin_memory=True)
            host_prefix.copy_(prefix[:rows], non_blocking=True)
            outputs_done = torch.cuda.Event()
            outputs_done.record()
            fence = overlap.fence_outputs(outputs_done)
            overlap.settle()
            host_published = torch.empty(rows, dtype=torch.int64, pin_memory=True)
            host_published.copy_(published.index_select(0, slots), non_blocking=True)
            observations.append((step, rows, host_prefix, host_published, fence))
        execution.synchronize()
    for step, rows, first, final, fence in observations:
        fence.synchronize()
        expected = torch.arange(rows, dtype=torch.int64) + step + 100
        assert torch.equal(first, expected)
        if rows == 4:
            expected[-1] = -1
        assert torch.equal(final, expected)
    overlap.release()


def test_default_stream_reuse_and_release_wait_for_pending_tail():
    """Cache copy/zero work follows the executor stream after DeviceHandle.settle."""
    overlap = DecodeGraphOverlap()
    execution = torch.cuda.Stream()
    default = torch.cuda.current_stream()
    state = torch.zeros(1, dtype=torch.int64, device="cuda")
    published = torch.zeros_like(state)
    execution.wait_stream(default)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(execution):
        published.copy_(state)
        execution.synchronize()
        with overlap.capture(graph), torch.cuda.graph(graph, stream=execution):
            overlap.record_prefix()
            torch.cuda._sleep(100000)
            published.copy_(state)
        execution.synchronize()

    observations = []
    for value in range(1, 5):
        with torch.cuda.stream(execution):
            execution.wait_stream(default)
            state.fill_(value)
            overlap.replay(graph, live_rows=1)
            copied = torch.cuda.Event()
            copied.record()
            fence = overlap.fence_outputs(copied)
        # Match the non-forward caller: settle from the ambient stream,
        # then copy_cache_pages/zero_cache_pages orders it behind execution.
        overlap.settle()
        default.wait_stream(execution)
        host = torch.empty(1, dtype=torch.int64, pin_memory=True)
        host.copy_(published, non_blocking=True)
        state.zero_()
        observations.append((value, host, fence))
    default.synchronize()
    for value, host, fence in observations:
        fence.synchronize()
        assert host.item() == value

    with torch.cuda.stream(execution):
        execution.wait_stream(default)
        state.fill_(99)
        overlap.replay(graph, live_rows=1)
        copied = torch.cuda.Event()
        copied.record()
        result = overlap.fence_outputs(copied)
    # Graph destruction must also drain an unsettled final round.
    overlap.release()
    result.synchronize()
    assert published.item() == 99


def test_admission_reset_and_idle_replay_preserve_tail_state():
    overlap = DecodeGraphOverlap()
    stream = torch.cuda.Stream()
    value = torch.zeros(1, device="cuda")
    published = torch.zeros_like(value)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        published.copy_(value)
        stream.synchronize()
        with overlap.capture(graph), torch.cuda.graph(graph, stream=stream):
            overlap.record_prefix()
            torch.cuda._sleep(100000)
            published.copy_(value)
        stream.synchronize()
        overlap.prepare_batch(((1, 1),), reset_state=False)
        observations = []
        for admission, reset in ((2, False), (2, True)):
            value.fill_(7)
            overlap.replay(graph, live_rows=1)
            done = torch.cuda.Event()
            done.record()
            fence = overlap.fence_outputs(done)
            # Both slot reuse and a same-admission reset must join before writes.
            overlap.prepare_batch(((admission, 1),), reset_state=reset)
            value.zero_()
            host = torch.empty(1, pin_memory=True)
            host.copy_(published, non_blocking=True)
            observations.append((host, fence))
        # Idle graphs stay on this stream and need no extra result fence.
        value.fill_(9)
        overlap.replay(graph, live_rows=0)
        done = torch.cuda.Event()
        done.record()
        overlap.fence_outputs(done).synchronize()
        assert published.item() == 9
        for host, fence in observations:
            fence.synchronize()
            assert host.item() == 7
    overlap.release()
