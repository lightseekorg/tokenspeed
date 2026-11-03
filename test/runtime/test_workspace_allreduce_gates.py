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

"""Shared-workspace plain all-reduce admission gates.

Covers the lamport dtype sentinel, integer dtypes, hidden alignment, device,
and empty inputs.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the TRT-LLM workspace all-reduce is NVIDIA-only",
)


def _nvidia_cuda_available() -> bool:
    from tokenspeed_kernel.platform import current_platform

    return torch.cuda.is_available() and current_platform().is_nvidia


@unittest.skipUnless(_nvidia_cuda_available(), "requires NVIDIA CUDA")
class TestWorkspaceAllreduceGates(unittest.TestCase):
    """Admission gates of trtllm_workspace_allreduce + manager failure state."""

    def _reduce(self, tensor, use_fp32_lamport: bool):
        from unittest import mock

        from tokenspeed_kernel.ops.communication import trtllm as comm

        manager = mock.Mock()
        manager.initialized = True
        manager.use_fp32_lamport = use_fp32_lamport
        manager.max_token_num = 64
        manager.hidden_dim = 4096
        manager.world_size = 8
        with mock.patch.object(
            comm.dist, "get_process_group_ranks", return_value=[0, 1]
        ), mock.patch.dict(comm._workspace_managers, {(0, 1): manager}, clear=True):
            return comm.trtllm_workspace_allreduce(tensor, mock.Mock(), fp32_acc=False)

    def test_fp32_payload_on_bf16_workspace_falls_back(self):
        tensor = torch.zeros(2, 256, dtype=torch.float32, device="cuda")
        self.assertIsNone(self._reduce(tensor, use_fp32_lamport=False))

    def test_bf16_payload_on_fp32_workspace_falls_back(self):
        tensor = torch.zeros(2, 256, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._reduce(tensor, use_fp32_lamport=True))

    def test_zero_token_falls_back(self):
        tensor = torch.zeros(0, 256, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._reduce(tensor, use_fp32_lamport=False))

    def test_integer_payload_falls_back(self):
        tensor = torch.zeros(2, 256, dtype=torch.int64, device="cuda")
        self.assertIsNone(self._reduce(tensor, use_fp32_lamport=False))

    def test_misaligned_hidden_falls_back(self):
        tensor = torch.zeros(2, 100, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._reduce(tensor, use_fp32_lamport=False))

    def test_cpu_payload_falls_back(self):
        tensor = torch.zeros(2, 256, dtype=torch.bfloat16)
        self.assertIsNone(self._reduce(tensor, use_fp32_lamport=False))

    def test_group_recorded_before_creation_failure(self):
        from unittest import mock

        from tokenspeed_kernel.ops.communication import trtllm as comm

        manager = comm.TrtllmFusionWorkspaceManager()
        sentinel = mock.Mock()
        with mock.patch.object(
            comm, "_skip_ipc_workspace", return_value=False
        ), mock.patch.object(
            comm,
            "trtllm_create_ipc_workspace_for_all_reduce_fusion",
            return_value=(mock.Mock(), mock.Mock()),
        ), mock.patch.object(
            comm, "_try_create_mnnvl_workspace", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                manager.initialize(2, 0, 64, 2048, sentinel)
        # Cleanup must target this group, not default to WORLD, on the next arm.
        self.assertIs(manager.group, sentinel)
        self.assertIsNotNone(manager.ipc_handles)
        self.assertFalse(manager.initialized)


if __name__ == "__main__":
    unittest.main()
