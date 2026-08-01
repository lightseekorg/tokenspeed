"""DeepEP dispatch-window overlap seam.

DeepEP's dispatch is split into a send and a receive step, so work queued
between them runs while the tokens are in flight instead of behind the receive.
The MoE path exposes that window as ``overlap_fn``. Ordering is the whole point:
running the callback after the receive would silently cost the overlap, and no
numerical check would notice, so it is asserted here.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_apply
from tokenspeed_kernel.ops.moe.deep_gemm import deepep_fp8

pytestmark = pytest.mark.skipif(
    not hasattr(deepep_fp8, "_apply_low_latency"),
    reason="DeepEP FP8 MoE path needs an NVIDIA platform with DeepGEMM",
)

NUM_LOCAL_EXPERTS = 2
RECV_M = 4
HIDDEN = 256
INTERMEDIATE = 128
TOP_K = 2


class _RecordingDispatcher:
    """Fake dispatcher that records the order of the legs it is driven through."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def dispatch_a(self, x, topk_ids, topk_weights, low_latency=None) -> None:
        del x, topk_ids, topk_weights, low_latency
        self.calls.append("dispatch_a")

    def dispatch_b(self):
        self.calls.append("dispatch_b")
        recv_x = torch.zeros(
            (NUM_LOCAL_EXPERTS, RECV_M, HIDDEN), dtype=torch.float8_e4m3fn
        )
        recv_scales = torch.zeros((NUM_LOCAL_EXPERTS, RECV_M, HIDDEN // 128))
        masked_m = torch.full((NUM_LOCAL_EXPERTS,), RECV_M, dtype=torch.int32)
        return (recv_x, recv_scales), None, None, None, None, None, masked_m

    def combine_a(self, out, topk_ids, topk_weights, low_latency=None) -> None:
        del out, topk_ids, topk_weights, low_latency
        self.calls.append("combine_a")

    def combine_b(self):
        self.calls.append("combine_b")
        return torch.zeros((1, HIDDEN), dtype=torch.bfloat16)


def _weights() -> torch.nn.Module:
    w = torch.nn.Module()
    w.num_experts = NUM_LOCAL_EXPERTS
    w.ep_size = 1
    w.w13_weight = torch.zeros(
        (NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN), dtype=torch.float8_e4m3fn
    )
    w.w13_weight_scale_inv = torch.zeros((NUM_LOCAL_EXPERTS, 1, 1))
    w.w2_weight = torch.zeros(
        (NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE), dtype=torch.float8_e4m3fn
    )
    w.w2_weight_scale_inv = torch.zeros((NUM_LOCAL_EXPERTS, 1, 1))
    return w


@pytest.fixture
def stub_compute(monkeypatch):
    """Replace the GEMM / activation kernels: only the call order is under test."""
    monkeypatch.setattr(
        deepep_fp8, "m_grouped_fp8_gemm_nt_masked", lambda *a, **k: None
    )
    monkeypatch.setattr(
        deepep_fp8, "get_mn_major_tma_aligned_tensor", lambda scales: scales
    )
    monkeypatch.setattr(deepep_fp8, "deep_gemm_requires_ue8m0", lambda: True)
    monkeypatch.setattr(
        deepep_fp8, "fused_swiglu_fp8_ue8m0_masked", lambda *a, **k: None
    )


def _run_low_latency(calls: list[str], overlap_fn) -> None:
    deepep_fp8._apply_low_latency(
        _RecordingDispatcher(calls),
        torch.zeros((1, HIDDEN), dtype=torch.bfloat16),
        _weights(),
        torch.zeros((1, TOP_K)),
        torch.zeros((1, TOP_K), dtype=torch.int64),
        None,
        False,
        overlap_fn=overlap_fn,
    )


def test_overlap_runs_inside_the_dispatch_window(stub_compute) -> None:
    calls: list[str] = []
    _run_low_latency(calls, lambda: calls.append("overlap"))
    assert calls.index("overlap") > calls.index("dispatch_a")
    assert calls.index("overlap") < calls.index("dispatch_b")


def test_overlap_runs_exactly_once(stub_compute) -> None:
    calls: list[str] = []
    _run_low_latency(calls, lambda: calls.append("overlap"))
    assert calls.count("overlap") == 1


def test_without_overlap_the_legs_are_unchanged(stub_compute) -> None:
    calls: list[str] = []
    _run_low_latency(calls, None)
    assert calls == ["dispatch_a", "dispatch_b", "combine_a", "combine_b"]


def test_overlap_only_reaches_all_to_all_plans(monkeypatch) -> None:
    """Plans that own no dispatch legs must not see the window argument."""
    seen: dict = {}

    def fake_select_kernel(*args, **kwargs):
        del args, kwargs

        def kernel(**kw):
            seen.update(kw)

        return kernel

    monkeypatch.setattr("tokenspeed_kernel.ops.moe.select_kernel", fake_select_kernel)
    x = torch.zeros((1, HIDDEN), dtype=torch.bfloat16)

    def overlap_fn() -> None:
        return None

    moe_apply(
        {"apply_kernel_name": None, "a2a_backend": "deepep"},
        x,
        _weights(),
        torch.zeros((1, NUM_LOCAL_EXPERTS)),
        overlap_fn=overlap_fn,
    )
    assert seen["overlap_fn"] is overlap_fn

    seen.clear()
    moe_apply(
        {"apply_kernel_name": None, "a2a_backend": None},
        x,
        _weights(),
        torch.zeros((1, NUM_LOCAL_EXPERTS)),
        overlap_fn=overlap_fn,
    )
    assert "overlap_fn" not in seen
