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

"""Host-side checks for NVIDIA RSAG signal-pad allocation."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication import triton as communication


@pytest.fixture
def create_rsag_state(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(communication, "_get_available_gpu_memory", lambda device: 1.0)
    monkeypatch.setattr(
        communication.symm_mem,
        "empty",
        lambda shape, *, dtype, device: torch.empty(shape, dtype=dtype, device="cpu"),
    )
    return partial(
        communication.nvidia_create_rsag_state,
        group=dist.ProcessGroup(0, 4),
        rank_in_group=0,
        max_tokens=2,
        hidden_size=8,
        device=torch.device("cuda:0"),
    )


@pytest.fixture
def fixed_signal_pad(monkeypatch):
    class FixedPad:
        def __init__(self):
            self.capacity = 512

        @property
        def signal_pad_size(self):
            return self.capacity

    pad = FixedPad()
    for name in ("get_signal_pad_size", "set_signal_pad_size"):
        monkeypatch.delattr(communication.symm_mem, name, raising=False)
    monkeypatch.setattr(
        communication.symm_mem, "rendezvous", lambda tensor, *, group: pad
    )
    return pad


def test_rsag_accepts_exact_fixed_signal_pad(create_rsag_state, fixed_signal_pad):
    # A 32-CTA grid with four ranks needs 32 * 4 * 4 = 512 bytes.
    create_rsag_state()


def test_rsag_rejects_insufficient_fixed_signal_pad(
    create_rsag_state, fixed_signal_pad
):
    fixed_signal_pad.capacity = 508
    with pytest.raises(
        RuntimeError,
        match="RSAG requires 512 signal-pad bytes, but Torch allocated 508",
    ):
        create_rsag_state()


@pytest.mark.parametrize(
    "initial_bytes,expected_bytes", [(256, 512), (1024, 1024)], ids=["grow", "preserve"]
)
def test_rsag_reserves_signal_pad_before_allocation(
    monkeypatch, create_rsag_state, initial_bytes, expected_bytes
):
    pad_bytes = initial_bytes

    class GlobalPad:
        @property
        def signal_pad_size(self):
            return pad_bytes

    def empty(shape, *, dtype, device):
        assert pad_bytes == expected_bytes
        return torch.empty(shape, dtype=dtype, device="cpu")

    def set_signal_pad_size(size):
        nonlocal pad_bytes
        pad_bytes = size

    def rendezvous(tensor, *, group):
        # On newer Torch the handle follows the global setting, which can
        # change after allocation without changing the allocated capacity.
        set_signal_pad_size(256)
        return GlobalPad()

    monkeypatch.setattr(communication.symm_mem, "empty", empty)
    monkeypatch.setattr(communication.symm_mem, "rendezvous", rendezvous)
    monkeypatch.setattr(
        communication.symm_mem, "get_signal_pad_size", lambda: pad_bytes, raising=False
    )
    monkeypatch.setattr(
        communication.symm_mem,
        "set_signal_pad_size",
        set_signal_pad_size,
        raising=False,
    )
    create_rsag_state()
