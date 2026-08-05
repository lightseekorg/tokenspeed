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

"""Tests for the MNNVL auto-detection / override on the DeepEP buffer path."""

from __future__ import annotations

import pytest
import tokenspeed_kernel.ops.communication.deep_ep as deep_ep_mod
import tokenspeed_kernel.ops.communication.fabric as fabric_mod

ENV_VAR = "TS_DEEPEP_ALLOW_MNNVL"


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(deep_ep_mod, "_allow_mnnvl_resolved", None)
    deep_ep_mod.DeepEPBuffer._buffer = None
    yield
    deep_ep_mod._allow_mnnvl_resolved = None
    deep_ep_mod.DeepEPBuffer._buffer = None


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
def test_env_override_wins_over_probe(monkeypatch, value, expected):
    monkeypatch.setenv(ENV_VAR, value)
    # Probe must not be consulted when the override is set.
    monkeypatch.setattr(
        deep_ep_mod,
        "fabric_allocation_supported",
        lambda device_index: pytest.fail("probe should not run"),
    )
    assert deep_ep_mod._resolve_allow_mnnvl(0) is expected


@pytest.mark.parametrize("probe_result", [True, False])
def test_unset_env_uses_probe(monkeypatch, probe_result):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(
        deep_ep_mod, "fabric_allocation_supported", lambda device_index: probe_result
    )
    assert deep_ep_mod._resolve_allow_mnnvl(0) is probe_result


def test_resolution_is_cached(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    calls = []

    def probe(device_index):
        calls.append(device_index)
        return False

    monkeypatch.setattr(deep_ep_mod, "fabric_allocation_supported", probe)
    assert deep_ep_mod._resolve_allow_mnnvl(0) is False
    assert deep_ep_mod._resolve_allow_mnnvl(0) is False
    assert len(calls) == 1


class _FakeConfig:
    def get_nvl_buffer_size_hint(self, hidden_bytes, size):
        return 1

    def get_rdma_buffer_size_hint(self, hidden_bytes, size):
        return 1


class _FakeBuffer:
    num_sms = 8
    last_kwargs = None

    @staticmethod
    def get_dispatch_config(size):
        return _FakeConfig()

    @staticmethod
    def get_combine_config(size):
        return _FakeConfig()

    def __init__(self, group, num_nvl_bytes, num_rdma_bytes, **kwargs):
        _FakeBuffer.last_kwargs = kwargs


class _FakeGroup:
    def size(self):
        return 8


@pytest.mark.parametrize("probe_result", [True, False])
def test_get_deepep_buffer_passes_resolved_allow_mnnvl(monkeypatch, probe_result):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(
        deep_ep_mod, "fabric_allocation_supported", lambda device_index: probe_result
    )
    monkeypatch.setattr(deep_ep_mod, "Buffer", _FakeBuffer)
    monkeypatch.setattr(
        deep_ep_mod, "_get_available_gpu_memory", lambda gpu_id, empty_cache=True: 1.0
    )
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)

    deep_ep_mod.DeepEPBuffer.get_deepep_buffer(
        _FakeGroup(),
        hidden_size=8,
        param_bytes=2,
        deepep_mode=deep_ep_mod.DeepEPMode.normal,
    )

    assert _FakeBuffer.last_kwargs["allow_mnnvl"] is probe_result


def test_probe_returns_false_on_this_host_without_imex():
    """Integration probe: no IMEX channels -> fabric allocation must fail.

    Skipped automatically on hosts where the IMEX stack is present (there the
    probe legitimately returns True).
    """
    import os

    if not os.path.exists("/dev/nvidia0"):
        pytest.skip("no NVIDIA GPU visible")
    if os.path.isdir("/dev/nvidia-caps-imex-channels"):
        pytest.skip("IMEX channels present; probe may legitimately succeed")
    assert fabric_mod._probe_fabric_allocation(0) is False
