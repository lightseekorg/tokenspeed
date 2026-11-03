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

"""CPU-only tests for the NCCL side of a sleep release (PD hybrid RL): which
engines can suspend their communicators, and a suspend without a process
group."""

from __future__ import annotations

import os
import sys

import pytest
import torch

from tokenspeed.runtime.distributed import nccl_suspend

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize(
    "version, bindings, supported",
    [
        ((2, 29, 6), ("suspend", "resume"), False),
        ((2, 29, 7), (), False),
        ((2, 30, 4), ("suspend",), False),
        ((2, 30, 4), ("resume",), False),
        ((2, 29, 7), ("suspend", "resume"), True),
    ],
)
def test_nccl_region_requires_both_bindings_and_a_supported_version(
    monkeypatch, version, bindings, supported
):
    # Stand in for a torch build's bindings and CUDA runtime, without GPUs.
    backend = type("NcclBackend", (), {name: lambda self: None for name in bindings})
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "ProcessGroupNCCL", backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: version)

    assert nccl_suspend.nccl_suspend_supported() is supported


def test_nccl_suspend_without_a_process_group_is_a_no_op():
    import torch.distributed as dist

    from tokenspeed.runtime.distributed.nccl_suspend import (
        resume_nccl_communicators,
        suspend_nccl_communicators,
    )

    if dist.is_initialized():
        pytest.skip("another test initialized torch.distributed in this process")
    assert suspend_nccl_communicators() == 0
    assert resume_nccl_communicators() == 0
