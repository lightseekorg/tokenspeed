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

"""Check staged auxiliary-stream dependencies with real tensor consumers."""

import os
import sys
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.utils.cuda_stream import StreamFork


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams required")
@pytest.mark.parametrize("enable", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_staged_branch_tensor_dependencies(enable, overlap):
    fork = StreamFork(torch.cuda.Stream())
    # Keep real CUDA events, but observe the disabled/serialized contracts
    # that numerical equality alone cannot distinguish from overlapping work.
    fork.fork_event = Mock(wraps=fork.fork_event)
    fork.join_event = Mock(wraps=fork.join_event)
    fork.checkpoint_event = Mock(wraps=fork.checkpoint_event)
    inputs = torch.randn(4096, device="cuda")
    first, tail, main, combined, output = [torch.empty_like(inputs) for _ in range(5)]

    def run():
        with fork.scope(enable=enable, overlap=overlap):
            with fork.branch():
                torch.add(inputs, 1, out=first)
                fork.record_checkpoint()
                torch.mul(first, 2, out=tail)
            fork.join_checkpoint()
            torch.add(first, 3, out=main)
            fork.join()
            torch.add(main, tail, out=combined)
            with fork.branch_after_main():
                torch.add(combined, 5, out=output)
            fork.join()
            # A real main-stream consumer detects a missing branch join.
            return output.mul(3)

    for _ in range(3):
        expected = ((inputs + 1) + 3 + (inputs + 1) * 2 + 5) * 3
        torch.testing.assert_close(run(), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Reusing events inside one graph must preserve both generations.
        captured = [run(), run()]
    for _ in range(5):
        inputs.normal_()
        graph.replay()
        expected = ((inputs + 1) + 3 + (inputs + 1) * 2 + 5) * 3
        for actual in captured:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if enable:
        scopes = fork.checkpoint_event.record.call_count
        assert scopes > 0
        assert fork.join_event.wait.call_count == scopes * (3 if overlap else 5)
    else:
        for event in (fork.fork_event, fork.join_event, fork.checkpoint_event):
            event.record.assert_not_called()
            event.wait.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
