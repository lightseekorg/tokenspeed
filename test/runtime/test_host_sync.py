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

"""The data-plane host-sync guard: arming, reporting and explicit exemptions."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution import device as device_module  # noqa: E402
from tokenspeed.runtime.utils.env import envs  # noqa: E402
from tokenspeed.runtime.utils.host_sync import (  # noqa: E402
    allow_host_sync,
    host_to_device,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture
def sync_debug_off():
    yield
    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode(0)


@requires_cuda
def test_error_mode_rejects_the_hidden_syncs_and_passes_the_async_forms(sync_debug_off):
    x = torch.arange(8, device="cuda")
    flags = torch.ones(8, dtype=torch.bool, device="cuda")
    slots = torch.tensor([1, 3], device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    for name, fn in (
        ("cpu", lambda: x.cpu()),
        ("tolist", lambda: x.tolist()),
        ("bool(any)", lambda: bool((x < 0).any())),
        # The scalar is staged through a pageable host tensor: a hidden sync.
        ("indexed scalar assignment", lambda: flags.__setitem__(slots, False)),
        ("pageable upload", lambda: torch.tensor([1, 2, 3], device="cuda")),
    ):
        with pytest.raises(RuntimeError, match="synchronizing"):
            fn()
    # The forms the data plane is allowed to use.
    flags.index_fill_(0, slots, False)
    x.to("cpu", non_blocking=True)
    torch.tensor([1, 2, 3], pin_memory=True).to("cuda", non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    event.synchronize()
    torch.cuda.set_sync_debug_mode(0)
    assert flags.tolist() == [True, False, True, False, True, True, True, True]


@requires_cuda
def test_allow_host_sync_scopes_the_exemption(sync_debug_off):
    x = torch.arange(4, device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    with allow_host_sync("test"):
        assert x.tolist() == [0, 1, 2, 3]
    assert torch.cuda.get_sync_debug_mode() == 2
    with pytest.raises(RuntimeError, match="synchronizing"):
        x.tolist()
    torch.cuda.set_sync_debug_mode(0)
    with allow_host_sync("test"):
        x.tolist()
    assert torch.cuda.get_sync_debug_mode() == 0


@requires_cuda
def test_serving_always_arms_the_check(monkeypatch, sync_debug_off):
    # Unset, it warns.
    monkeypatch.delenv("TOKENSPEED_DATA_PLANE_SYNC_DEBUG", raising=False)
    assert envs.TOKENSPEED_DATA_PLANE_SYNC_DEBUG.get() == "warn"
    device_module.arm_data_plane_sync_debug("cuda")
    assert torch.cuda.get_sync_debug_mode() == 1
    torch.cuda.set_sync_debug_mode(0)
    monkeypatch.setenv("TOKENSPEED_DATA_PLANE_SYNC_DEBUG", "error")
    device_module.arm_data_plane_sync_debug("cuda")
    assert torch.cuda.get_sync_debug_mode() == 2


@pytest.mark.parametrize("mode", ["default", "off", "0", "loud"])
def test_the_check_cannot_be_disabled(monkeypatch, mode):
    monkeypatch.setenv("TOKENSPEED_DATA_PLANE_SYNC_DEBUG", mode)
    with pytest.raises(ValueError, match="cannot be disabled"):
        device_module.arm_data_plane_sync_debug("cuda")


def test_host_to_device_copies_lists_and_strided_tensors():
    assert host_to_device(
        [[1, 2], [3, 4]], dtype=torch.int32, device="cpu"
    ).tolist() == [
        [1, 2],
        [3, 4],
    ]
    table = torch.arange(12).reshape(3, 4)
    out = host_to_device(table[:, 1:3], dtype=torch.int64, device="cpu")
    assert out.is_contiguous()
    assert torch.equal(out, table[:, 1:3])
    assert host_to_device([], dtype=torch.int64, device="cpu").shape == (0,)


@requires_cuda
def test_host_to_device_does_not_wait_for_the_stream(sync_debug_off):
    # Long enough that copying it from a strided source, even a pinned one,
    # makes the driver stage it through pageable memory and wait.
    table = torch.arange(3 * (1 << 18)).reshape(3, -1)
    chunk = table[:, 2 : 2 + (1 << 17)]
    values = list(range(1000))
    # The first pinned buffer of a size comes from cudaHostAlloc, which waits.
    host_to_device(chunk, dtype=torch.int64, device="cuda")
    host_to_device(values, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    torch.cuda._sleep(1 << 30)
    in_flight = torch.cuda.Event()
    in_flight.record()
    torch.cuda.set_sync_debug_mode("error")
    from_tensor = host_to_device(chunk, dtype=torch.int64, device="cuda")
    from_list = host_to_device(values, dtype=torch.int32, device="cuda")
    torch.cuda.set_sync_debug_mode(0)
    assert not in_flight.query()
    torch.cuda.synchronize()
    assert torch.equal(from_tensor.cpu(), chunk)
    assert from_list.tolist() == values


@requires_cuda
def test_host_to_device_refuses_graph_capture():
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with pytest.raises(RuntimeError, match="graph capture"):
            with torch.cuda.graph(graph, stream=stream):
                host_to_device([1, 2], dtype=torch.int32, device="cuda")
