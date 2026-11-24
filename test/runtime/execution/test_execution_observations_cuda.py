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

"""Real CUDA replay, fence and memory-release tests for the upstream GPU suite."""

from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch

register_cuda_ci(est_time=30, suite="runtime-1gpu")
pytestmark = pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(), reason="requires CUDA"
)


def test_breakable_replay_reports_executed_segments_and_live_values():
    import torch

    from tokenspeed.runtime.execution.breakable_cuda_graph import (
        BreakableCapture,
        break_point,
    )

    source = torch.arange(32, dtype=torch.float32, device="cuda")

    @break_point
    def eager_part(value):
        return value + 2

    def target():
        return eager_part(source * 3).square()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            target()
    stream.synchronize()
    capture = BreakableCapture(stream=stream)
    with capture:
        output = target()
    stream.synchronize()
    for step in (1, 4, 9):
        source.fill_(step)
        count = capture.replay(valid_rows=32)
        torch.cuda.synchronize()
        assert count == 2
        torch.testing.assert_close(
            output, torch.full_like(output, (3 * step + 2) ** 2), rtol=0, atol=0
        )


def test_execution_receipts_allocate_no_gpu_storage():
    import torch

    from tokenspeed.runtime.execution.observations import TargetExecution

    candidates = torch.empty((3, 8), device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(100):
        receipt = TargetExecution("eager", 0, 3, 24, 0).verified(
            candidates, packed=False
        )
        receipt = receipt.replayed(request_capacity=4, query_capacity=32)
        assert receipt.payload((1, 2, 3))[2] == 8
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def test_pre_stop_and_force_are_refreshed_by_graph_and_copied_before_reuse():
    import torch

    from tokenspeed.runtime.execution.observations import VerificationObservations

    observations = VerificationObservations(max_bs=4, device="cuda")
    accepted = torch.empty(4, device="cuda", dtype=torch.int32)
    forced = torch.empty(3, device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def forward():
        saved = observations.record(accepted, num_extends=1, force_single=forced)
        accepted.clamp_(max=2)  # device stop clipping mutates the original lengths
        return saved

    with torch.cuda.stream(stream):
        accepted.fill_(8)
        forced.zero_()
        forward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        saved = forward()
    snapshots = []
    with torch.cuda.stream(stream):
        for value, apply_force in ((7, True), (6, False), (4, True)):
            accepted.fill_(value)
            forced.zero_()
            if apply_force:
                accepted[2].fill_(1)
                forced[1].fill_(True)
            graph.replay()
            host = saved[:3].to("cpu", non_blocking=True)  # one padded row omitted
            kept = accepted[:3].to("cpu", non_blocking=True)
            fence = torch.cuda.Event()
            fence.record()
            snapshots.append((value, apply_force, host, kept, fence))
        # Later launches overwrite the device source before the CPU consumes any result.
        observations.buffer.fill_(-9)
    for value, apply_force, host, kept, fence in snapshots:
        fence.synchronize()
        assert host.tolist() == [
            [-1, 0],
            [value, 0],
            [1 if apply_force else value, int(apply_force)],
        ]
        assert kept.tolist() == [2, 2, 1 if apply_force else 2]


def test_named_observation_scratch_releases_and_restores_at_stable_addresses():
    import subprocess
    import sys

    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    # The child starts with the allocator preload, as a real host does.
    adapter = TorchMemorySaverAdapter.create(enable=True)
    with adapter.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import torch
from tokenspeed.runtime.execution.observations import VerificationObservations
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
adapter = TorchMemorySaverAdapter.create(enable=True)
with adapter.named_region('runtime_states'):
    observation = VerificationObservations(max_bs=4, device='cuda')
    lengths = torch.empty(4, dtype=torch.int32, device='cuda')
    forced = torch.empty(3, dtype=torch.bool, device='cuda')
addresses = (observation.buffer.data_ptr(), lengths.data_ptr(), forced.data_ptr())
for cycle in range(2):
    lengths.fill_(7 - cycle)
    lengths[2].fill_(1)
    forced.zero_()
    forced[1].fill_(True)
    observation.record(lengths, num_extends=1, force_single=forced)
    expected = observation.buffer.cpu().clone()
    torch.cuda.synchronize()
    adapter.pause(tag='weights')
    torch.cuda.empty_cache()
    ordinary = sum(segment['total_size'] for segment in torch.cuda.memory_snapshot()
                   if tuple(segment['segment_pool_id']) == (0, 0))
    assert ordinary == 0, ordinary
    adapter.resume(tag='weights')
    assert addresses == (observation.buffer.data_ptr(), lengths.data_ptr(), forced.data_ptr())
    assert torch.equal(observation.buffer.cpu(), expected)
""",
            ],
            check=True,
        )
