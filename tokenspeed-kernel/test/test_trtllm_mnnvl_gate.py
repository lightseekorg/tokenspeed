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

"""Tests for the MNNVL capability gate on the trtllm one-shot all-reduce path."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_nvidia and torch.cuda.is_available()),
    reason="trtllm MNNVL gate is NVIDIA/CUDA only",
)


def _probe():
    import tokenspeed_kernel.ops.communication.trtllm as trtllm_mod

    return trtllm_mod, trtllm_mod._mnnvl_locally_available


def test_cross_host_group_requires_fabric(monkeypatch):
    """A group wider than the host's GPUs needs working fabric memory.

    Without it, symm_mem.rendezvous() hangs instead of failing, so the gate
    must reject the workspace up front.
    """
    trtllm_mod, probe = _probe()
    monkeypatch.setattr(trtllm_mod, "fabric_allocation_supported", lambda _: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    assert probe(16) is False


def test_cross_host_group_allowed_with_fabric(monkeypatch):
    trtllm_mod, probe = _probe()
    monkeypatch.setattr(trtllm_mod, "fabric_allocation_supported", lambda _: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    # Still subject to the other capability checks, so only assert that the
    # cross-host rule alone no longer vetoes the group.
    assert probe(16) == probe(8)


def test_intra_host_group_ignores_fabric(monkeypatch):
    """Groups inside one host ride NVLS multicast, so fabric must not gate them."""
    trtllm_mod, probe = _probe()
    monkeypatch.setattr(
        trtllm_mod,
        "fabric_allocation_supported",
        lambda _: pytest.fail("fabric probe must not run for intra-host groups"),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    probe(8)


def test_unsupported_world_size_rejected():
    _, probe = _probe()

    assert probe(3) is False
