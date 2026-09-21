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

import pytest
import torch

from tokenspeed.runtime.execution import device as device_module
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.host_sync import allow_host_sync

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
def test_arming_follows_the_env(monkeypatch, sync_debug_off):
    monkeypatch.setenv("TOKENSPEED_DATA_PLANE_SYNC_DEBUG", "warn")
    assert envs.TOKENSPEED_DATA_PLANE_SYNC_DEBUG.get() == "warn"
    device_module.arm_data_plane_sync_debug("cuda")
    assert torch.cuda.get_sync_debug_mode() == 1
    monkeypatch.setenv("TOKENSPEED_DATA_PLANE_SYNC_DEBUG", "default")
    torch.cuda.set_sync_debug_mode(0)
    device_module.arm_data_plane_sync_debug("cuda")
    assert torch.cuda.get_sync_debug_mode() == 0


def test_arming_rejects_unknown_modes(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DATA_PLANE_SYNC_DEBUG", "loud")
    with pytest.raises(ValueError, match="default, warn or error"):
        device_module.arm_data_plane_sync_debug("cuda")
