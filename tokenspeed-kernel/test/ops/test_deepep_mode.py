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

"""DeepEP mode selection and buffer-reuse guards.

Both are pure host-side bookkeeping, so they are exercised without a process
group: picking the wrong leg (or silently reusing an undersized buffer) is a
hang or a corruption on a real cluster, not a test failure.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.communication import deep_ep as deep_ep_module
from tokenspeed_kernel.ops.communication.deep_ep import (
    DeepEPBuffer,
    DeepEPMode,
    _DeepEPDispatcherImplLowLatency,
)


@pytest.mark.parametrize("low_latency", [True, False, None])
def test_pinned_mode_ignores_the_per_forward_decision(low_latency) -> None:
    assert DeepEPMode.normal.resolve(low_latency) is DeepEPMode.normal
    assert DeepEPMode.low_latency.resolve(low_latency) is DeepEPMode.low_latency


def test_auto_mode_follows_the_per_forward_decision() -> None:
    assert DeepEPMode.auto.resolve(True) is DeepEPMode.low_latency
    assert DeepEPMode.auto.resolve(False) is DeepEPMode.normal


def test_auto_mode_requires_a_decision() -> None:
    with pytest.raises(ValueError, match="requires an explicit low_latency"):
        DeepEPMode.auto.resolve(None)


@pytest.mark.parametrize(
    "mode,normal,low_latency",
    [
        (DeepEPMode.normal, True, False),
        (DeepEPMode.low_latency, False, True),
        (DeepEPMode.auto, True, True),
    ],
)
def test_enabled_legs_per_mode(mode, normal, low_latency) -> None:
    assert mode.enable_normal() is normal
    assert mode.enable_low_latency() is low_latency


def test_new_buffer_disables_mnnvl_fabric_handles(monkeypatch) -> None:
    seen = {}

    class Config:
        def get_nvl_buffer_size_hint(self, _hidden_bytes, _group_size):
            return 1024

        def get_rdma_buffer_size_hint(self, _hidden_bytes, _group_size):
            return 2048

    class RecordingBuffer:
        num_sms = 20

        @staticmethod
        def get_dispatch_config(_group_size):
            return Config()

        @staticmethod
        def get_combine_config(_group_size):
            return Config()

        def __init__(self, *_args, **kwargs):
            seen.update(kwargs)
            self.low_latency_mode = kwargs["low_latency_mode"]

    class Group:
        @staticmethod
        def size():
            return 4

    monkeypatch.setattr(deep_ep_module, "Buffer", RecordingBuffer)
    monkeypatch.setattr(
        deep_ep_module, "_get_available_gpu_memory", lambda *_args, **_kwargs: 1.0
    )
    monkeypatch.setattr(DeepEPBuffer, "_buffer", None)

    DeepEPBuffer.get_deepep_buffer(
        group=Group(),
        hidden_size=2048,
        param_bytes=2,
        deepep_mode=DeepEPMode.normal,
        num_max_dispatch_tokens_per_rank=256,
        num_experts=64,
    )

    assert seen["allow_mnnvl"] is False


class _StubBuffer:
    low_latency_mode = True


@pytest.fixture
def allocated_buffer(monkeypatch):
    """Pretend a buffer was already allocated for an auto-mode layer."""
    monkeypatch.setattr(DeepEPBuffer, "_buffer", _StubBuffer(), raising=False)
    monkeypatch.setattr(DeepEPBuffer, "_hidden_size", 2048, raising=False)
    monkeypatch.setattr(DeepEPBuffer, "_num_experts", 64, raising=False)
    monkeypatch.setattr(
        DeepEPBuffer, "_num_max_dispatch_tokens_per_rank", 256, raising=False
    )
    monkeypatch.setattr(DeepEPBuffer, "_deepep_mode", DeepEPMode.auto, raising=False)


def test_reuse_accepts_a_matching_request(allocated_buffer) -> None:
    assert (
        DeepEPBuffer.get_deepep_buffer(
            group=None,
            hidden_size=2048,
            param_bytes=2,
            deepep_mode=DeepEPMode.auto,
            num_max_dispatch_tokens_per_rank=256,
            num_experts=64,
        )
        is DeepEPBuffer._buffer
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"hidden_size": 4096}, "hidden_size"),
        ({"num_experts": 128}, "num_experts"),
        ({"num_max_dispatch_tokens_per_rank": 512}, "tokens per rank"),
    ],
)
def test_reuse_rejects_a_request_the_buffer_cannot_serve(
    allocated_buffer, overrides, match
) -> None:
    kwargs = dict(
        group=None,
        hidden_size=2048,
        param_bytes=2,
        deepep_mode=DeepEPMode.auto,
        num_max_dispatch_tokens_per_rank=256,
        num_experts=64,
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        DeepEPBuffer.get_deepep_buffer(**kwargs)


def test_reuse_rejects_a_mode_whose_legs_were_never_allocated(monkeypatch) -> None:
    monkeypatch.setattr(DeepEPBuffer, "_buffer", _StubBuffer(), raising=False)
    monkeypatch.setattr(DeepEPBuffer, "_hidden_size", 2048, raising=False)
    monkeypatch.setattr(DeepEPBuffer, "_num_experts", 64, raising=False)
    monkeypatch.setattr(
        DeepEPBuffer, "_num_max_dispatch_tokens_per_rank", 256, raising=False
    )
    monkeypatch.setattr(
        DeepEPBuffer, "_deepep_mode", DeepEPMode.low_latency, raising=False
    )
    with pytest.raises(ValueError, match="without the normal-mode legs"):
        DeepEPBuffer.get_deepep_buffer(
            group=None,
            hidden_size=2048,
            param_bytes=2,
            deepep_mode=DeepEPMode.normal,
            num_max_dispatch_tokens_per_rank=256,
            num_experts=64,
        )


@pytest.mark.parametrize("ue8m0_scales", [False, True])
def test_low_latency_dispatch_requests_deepep_packed_ue8m0(
    monkeypatch, ue8m0_scales
) -> None:
    seen = {}

    class RecordingBuffer:
        def low_latency_dispatch(self, *args, **kwargs):
            seen.update(kwargs)
            return "hidden", "count", "handle", "event", "hook"

    impl = _DeepEPDispatcherImplLowLatency(
        return_recv_hook=True,
        use_fp8=True,
        ue8m0_scales=ue8m0_scales,
        group=None,
        router_topk=4,
        permute_fusion=True,
        num_experts=64,
        num_local_experts=16,
        hidden_size=2048,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.low_latency,
        low_latency_max_num_tokens_per_gpu=256,
    )
    monkeypatch.setattr(impl, "_get_buffer", RecordingBuffer)

    impl._dispatch_core(
        torch.empty((1, 2048), dtype=torch.bfloat16),
        torch.zeros((1, 4), dtype=torch.int64),
        use_fp8=True,
    )

    assert seen["round_scale"] is ue8m0_scales
    assert seen["use_ue8m0"] is ue8m0_scales
