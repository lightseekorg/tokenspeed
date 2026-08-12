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

"""Regression tests for capture-only auxiliary-stream initialization."""

import multiprocessing
import os
import sys
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402


def _run_rocm_aux_stream_capture() -> None:
    """Run in a child so a capture-time HIP failure cannot hang pytest."""
    from tokenspeed.runtime.execution.cuda_graph_wrapper import (
        _capture_execution_path,
        get_is_capture_mode,
    )
    from tokenspeed.runtime.utils.cuda_stream import StreamFork

    torch.backends.cuda.preferred_blas_library("hipblaslt")

    # Qwen3.5-397B shared-expert gate/up projection at decode batch size 1.
    x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((2048, 4096), device="cuda", dtype=torch.bfloat16)
    graph_stream = torch.cuda.Stream()
    stream_fork = StreamFork(torch.cuda.Stream())
    torch.cuda.synchronize()

    def run_once() -> torch.Tensor:
        with stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with fork.branch():
                return F.linear(x, weight)

    graph = torch.cuda.CUDAGraph()
    with _capture_execution_path():
        for _ in range(4):
            with torch.cuda.stream(graph_stream):
                output = run_once()
        torch.cuda.synchronize()

        with torch.cuda.graph(graph, stream=graph_stream):
            output = run_once()

    graph.replay()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise AssertionError("captured hipBLASLt output is not finite")


class CaptureExecutionPathTest(unittest.TestCase):
    def setUp(self):
        from tokenspeed.runtime.execution import cuda_graph_wrapper

        self.wrapper_module = cuda_graph_wrapper
        self.old_capture_mode = cuda_graph_wrapper._is_capture_mode
        self.old_cuda_graph_phase = cuda_graph_wrapper._is_cuda_graph_phase
        cuda_graph_wrapper._is_capture_mode = False
        cuda_graph_wrapper._is_cuda_graph_phase = False

    def tearDown(self):
        self.wrapper_module._is_capture_mode = self.old_capture_mode
        self.wrapper_module._is_cuda_graph_phase = self.old_cuda_graph_phase

    def test_selects_capture_branches_and_restores_state(self):
        wrapper = self.wrapper_module

        with wrapper._capture_execution_path():
            self.assertTrue(wrapper.get_is_capture_mode())
            self.assertTrue(wrapper.get_is_cuda_graph_phase())

        self.assertFalse(wrapper.get_is_capture_mode())
        self.assertFalse(wrapper.get_is_cuda_graph_phase())

    def test_restores_state_after_failure(self):
        wrapper = self.wrapper_module

        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            with wrapper._capture_execution_path():
                raise RuntimeError("warmup failed")

        self.assertFalse(wrapper.get_is_capture_mode())
        self.assertFalse(wrapper.get_is_cuda_graph_phase())


@unittest.skipUnless(
    torch.version.hip is not None and torch.cuda.is_available(),
    "requires a ROCm GPU",
)
class RocmAuxStreamCaptureTest(unittest.TestCase):
    def test_capture_path_prewarms_aux_stream_hipblaslt_handle(self):
        process = multiprocessing.get_context("spawn").Process(
            target=_run_rocm_aux_stream_capture
        )
        process.start()
        process.join(timeout=60)

        if process.is_alive():
            process.kill()
            process.join()
            self.fail("ROCm auxiliary-stream graph capture timed out")

        self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
