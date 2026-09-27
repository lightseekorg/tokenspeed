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

"""Host-side coverage for backend-independent DeepEP buffer preparation."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops import moe
from tokenspeed_kernel.ops.communication import deep_ep


@pytest.mark.parametrize(
    "solution,mode",
    [
        ("marlin", "normal"),
        ("marlin", "low_latency"),
        ("marlin", "auto"),
        ("deep_gemm", "normal"),
        ("deep_gemm", "low_latency"),
        ("deep_gemm", "auto"),
        ("flashinfer_cutedsl", "low_latency"),
    ],
)
def test_common_weight_processing_prepares_every_deepep_solution(
    monkeypatch, solution, mode
):
    events = []
    state = object()
    weights = SimpleNamespace(hidden_size=2048, num_experts=64)
    group = object()
    capacity = None if mode == "normal" else 256

    def preprocess(*, plan, w):
        events.append("weights")
        # Kernel-private packed dimensions must not become the communication width.
        w.w13_weight = torch.empty((16, 32, 1024), dtype=torch.uint8)
        return state

    def prepare(**kwargs):
        assert events == ["weights"]
        assert kwargs == dict(
            group=group,
            hidden_size=2048,
            num_experts=64,
            deepep_mode=mode,
            max_dispatch_tokens_per_rank=capacity,
        )
        events.append("buffer")

    monkeypatch.setattr(deep_ep, "prepare_deepep_buffer", prepare)
    plan = dict(
        a2a_backend="deepep",
        solution=solution,
        weight_preprocessor=preprocess,
        process_group=group,
        deepep_mode=mode,
        deepep_low_latency_max_num_tokens_per_gpu=capacity,
    )
    assert moe.moe_process_weights(plan, weights) is state
    assert events == ["weights", "buffer"]
    assert "_deepep_dispatcher" not in plan


@pytest.mark.parametrize("backend", [None, "none"])
@pytest.mark.parametrize("with_preprocessor", [False, True])
def test_non_deepep_plans_do_not_prepare_a_buffer(
    monkeypatch, backend, with_preprocessor
):
    prepare = mock.Mock()
    monkeypatch.setattr(deep_ep, "prepare_deepep_buffer", prepare)
    result = object()
    preprocessor = mock.Mock(return_value=result) if with_preprocessor else None
    plan = dict(a2a_backend=backend, weight_preprocessor=preprocessor)
    # No communication geometry is required for ordinary plans.
    weights = object()
    assert moe.moe_process_weights(plan, weights) is (
        result if with_preprocessor else None
    )
    prepare.assert_not_called()
    if preprocessor is not None:
        preprocessor.assert_called_once_with(plan=plan, w=weights)


def test_deepep_prepares_even_without_a_weight_preprocessor(monkeypatch):
    prepare = mock.Mock()
    monkeypatch.setattr(deep_ep, "prepare_deepep_buffer", prepare)
    group = object()
    plan = dict(
        a2a_backend="deepep",
        weight_preprocessor=None,
        process_group=group,
        deepep_mode="auto",
        deepep_low_latency_max_num_tokens_per_gpu=32,
    )
    assert (
        moe.moe_process_weights(
            plan, SimpleNamespace(hidden_size=3584, num_experts=896)
        )
        is None
    )
    prepare.assert_called_once_with(
        group=group,
        hidden_size=3584,
        num_experts=896,
        deepep_mode="auto",
        max_dispatch_tokens_per_rank=32,
    )


def test_failed_weight_processing_does_not_start_buffer_collectives(monkeypatch):
    prepare = mock.Mock()
    monkeypatch.setattr(deep_ep, "prepare_deepep_buffer", prepare)
    plan = dict(
        a2a_backend="deepep",
        weight_preprocessor=mock.Mock(side_effect=RuntimeError("bad weights")),
    )
    with pytest.raises(RuntimeError, match="bad weights"):
        moe.moe_process_weights(plan, SimpleNamespace(hidden_size=2048, num_experts=64))
    prepare.assert_not_called()


@pytest.fixture
def buffer_allocations(monkeypatch):
    allocations = []

    class Buffer:
        num_sms = 8

        @staticmethod
        def get_dispatch_config(size):
            return SimpleNamespace(
                get_nvl_buffer_size_hint=lambda hidden, ranks: 100,
                get_rdma_buffer_size_hint=lambda hidden, ranks: 200,
            )

        get_combine_config = get_dispatch_config

        @staticmethod
        def get_low_latency_rdma_size_hint(capacity, hidden, ranks, experts):
            return capacity * hidden * 2

        def __init__(
            self,
            group,
            nvl_bytes,
            rdma_bytes,
            *,
            low_latency_mode,
            num_qps_per_rank,
            allow_mnnvl
        ):
            self.low_latency_mode = low_latency_mode
            allocations.append((self, nvl_bytes, rdma_bytes))

        def clean_low_latency_buffer(self, *args):
            pass

    monkeypatch.setattr(deep_ep, "Buffer", Buffer)
    monkeypatch.setattr(deep_ep, "_get_available_gpu_memory", lambda device: 100.0)
    monkeypatch.setattr(deep_ep, "_resolve_allow_mnnvl", lambda device: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for name in (
        "_buffer",
        "_dispatch_mode",
        "_hidden_size",
        "_num_max_dispatch_tokens_per_rank",
        "_num_experts",
        "_deepep_mode",
    ):
        monkeypatch.setattr(deep_ep.DeepEPBuffer, name, None)
    return allocations


@pytest.mark.parametrize("mode", ["normal", "low_latency", "auto"])
def test_prepare_allocates_once_and_runtime_dispatcher_reuses_it(
    buffer_allocations, mode
):
    group = SimpleNamespace(size=lambda: 4)
    capacity = None if mode == "normal" else 256
    arguments = dict(
        group=group,
        hidden_size=2048,
        num_experts=64,
        deepep_mode=mode,
        max_dispatch_tokens_per_rank=capacity,
    )
    deep_ep.prepare_deepep_buffer(**arguments)
    deep_ep.prepare_deepep_buffer(**arguments)
    assert len(buffer_allocations) == 1
    buffer, nvl_bytes, rdma_bytes = buffer_allocations[0]
    assert nvl_bytes == (0 if mode == "low_latency" else 100)
    assert rdma_bytes == (200 if mode == "normal" else 256 * 2048 * 2)
    config = SimpleNamespace(
        group=group,
        hidden_size=2048,
        num_experts=64,
        world_size=4,
        top_k=2,
        params_dtype=torch.bfloat16,
        low_latency_max_num_tokens_per_gpu=capacity,
    )
    dispatcher = deep_ep.DeepEPDispatcher(
        config,
        deepep_mode=deep_ep.DeepEPMode(mode),
        async_finish=False,
        return_recv_hook=True,
        use_fp8=False,
        ue8m0_scales=False,
    )
    if mode != "low_latency":
        assert dispatcher._get_impl(False)._get_buffer() is buffer
    if mode != "normal":
        assert dispatcher._get_impl(True)._get_buffer() is buffer
    assert len(buffer_allocations) == 1


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"hidden_size": 4096}, "hidden_size"),
        ({"num_experts": 128}, "num_experts"),
        ({"max_dispatch_tokens_per_rank": 512}, "tokens per rank"),
    ],
)
def test_prepare_preserves_buffer_compatibility_checks(
    buffer_allocations, overrides, match
):
    kwargs = dict(
        group=SimpleNamespace(size=lambda: 4),
        hidden_size=2048,
        num_experts=64,
        deepep_mode="auto",
        max_dispatch_tokens_per_rank=256,
    )
    deep_ep.prepare_deepep_buffer(**kwargs)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        deep_ep.prepare_deepep_buffer(**kwargs)
    assert len(buffer_allocations) == 1


@pytest.mark.parametrize("mode", ["auto", "low_latency"])
@pytest.mark.parametrize("capacity", [None, 0, -1])
def test_prepare_rejects_invalid_low_latency_capacity(monkeypatch, mode, capacity):
    acquire = mock.Mock()
    monkeypatch.setattr(deep_ep.DeepEPBuffer, "get_deepep_buffer", acquire)
    with pytest.raises(ValueError, match="positive token capacity"):
        deep_ep.prepare_deepep_buffer(
            group=object(),
            hidden_size=2048,
            num_experts=64,
            deepep_mode=mode,
            max_dispatch_tokens_per_rank=capacity,
        )
    acquire.assert_not_called()


@pytest.mark.parametrize(
    "override,message",
    [
        ({"group": None}, "missing its process_group"),
        ({"hidden_size": 0}, "must be positive"),
        ({"num_experts": 0}, "must be positive"),
    ],
)
def test_invalid_geometry_fails_before_allocation(monkeypatch, override, message):
    acquire = mock.Mock()
    monkeypatch.setattr(deep_ep.DeepEPBuffer, "get_deepep_buffer", acquire)
    arguments = dict(
        group=object(),
        hidden_size=2048,
        num_experts=64,
        deepep_mode="normal",
        max_dispatch_tokens_per_rank=None,
    )
    arguments.update(override)
    with pytest.raises(ValueError, match=message):
        deep_ep.prepare_deepep_buffer(**arguments)
    acquire.assert_not_called()
