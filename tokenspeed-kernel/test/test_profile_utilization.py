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

from dataclasses import asdict, replace

import pytest
import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    OpProfile,
    ProfileSession,
    profiled,
)
from tokenspeed_kernel.profile_costs import (
    AttentionMatmulCost,
    CollectiveCost,
    CollectiveTraffic,
    DenseMatmulCost,
    register_compute_costs,
)
from tokenspeed_kernel.profile_utilization import (
    HardwareProfile,
    ScopeTiming,
    analyze_utilization,
)
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, register_kernel


@pytest.mark.parametrize("batched", [False, True])
def test_registered_compute_costs(batched):
    from tokenspeed_kernel.numerics.torch.gemm import torch_bmm, torch_mm

    batch = (2,) if batched else ()
    a, b = torch.randn(*batch, 3, 5), torch.randn(*batch, 4, 5)
    fn = torch_bmm if batched else torch_mm
    session = ProfileSession(run_id="compute", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        result = fn(a, b, None, None, torch.float64)
    torch.testing.assert_close(result, (a @ b.transpose(-1, -2)).double())
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy="unavailable")
    record = session.ops[0]
    assert record.implementation == ("torch_bmm" if batched else "torch_mm")
    report = session.report(costs=costs)
    cost = report["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 2 * (2 if batched else 1) * 3 * 4 * 5
    assert cost["logical_read_bytes"] == (a.numel() + b.numel()) * 4
    assert cost["logical_write_bytes"] == result.numel() * 8
    assert cost["compute_type"] is None
    assert cost["estimated_hbm_bytes"] is None


@pytest.fixture
def fixture(fresh_registry):
    @register_kernel(
        "gemm",
        "mm",
        name="utilization_mm",
        solution="reference",
        signatures=frozenset(),
        profile=OpProfile(cost_rule="mm"),
    )
    def mm(a, b):
        return a @ b

    class Layer:
        @profiled(level="layer")
        def forward(self):
            return mm(torch.ones(2, 3), torch.ones(3, 4))

    layer = Layer()
    session = ProfileSession(run_id="test", model_id="test", rank=0, backend="none")
    session.bind_layer(layer, layer_id=7)
    with session.iteration(iter_id=0):
        layer.forward()
        layer.forward()
    costs = CostRegistry()
    dense = DenseMatmulCost(lhs="a", rhs="b", output="return")
    costs.register(
        "mm",
        lambda op: replace(
            dense(op), compute_type="synthetic", estimated_hbm_bytes=104
        ),
    )
    timings = [
        ScopeTiming(
            r.trace_label, r.cpu_start_ns, r.cpu_start_ns + r.cpu_duration_ns, duration
        )
        for records, duration in (
            (session.ops, 100_000_000),
            (session.layers, 200_000_000),
        )
        for r in records
    ]
    hardware = HardwareProfile("test", {"synthetic": 1000}, 1000, "analytical fixture")
    return session, costs, timings, hardware, layer


def test_metrics_match_analytical_reference_and_repeated_calls(fixture):
    session, costs, timings, hardware, _ = fixture
    rows = analyze_utilization(session.report(costs=costs), timings, hardware)
    assert len(rows) == 4
    ops = [r for r in rows if r["kind"] == "op"]
    layers = [r for r in rows if r["kind"] == "layer"]
    for row in ops:
        assert row["mfu"] == pytest.approx(0.48)
        assert row["estimated_mbu"] == pytest.approx(1.04)  # Never clamp to 100%.
        assert "MFU=48.00% Est.MBU=104.00%" in row["nvtx_label"]
    for row in layers:
        assert row["op_count"] == 1
        assert row["mfu"] == pytest.approx(0.24)
        assert row["estimated_mbu"] == pytest.approx(0.52)
    assert [r["occurrence"] for r in layers] == [0, 1]


def test_nested_layer_counts_each_descendant_op_once(fixture):
    session, costs, _, hardware, child = fixture

    class Parent:
        @profiled(level="layer")
        def forward(self):
            child.forward()
            return child.forward()

    parent = Parent()
    session.bind_layer(parent, layer_id=0)
    with session.iteration(iter_id=1):
        parent.forward()
    timings = [
        ScopeTiming(
            r.trace_label,
            r.cpu_start_ns,
            r.cpu_start_ns + r.cpu_duration_ns,
            400_000_000,
        )
        for r in [*session.ops, *session.layers]
    ]
    rows = analyze_utilization(session.report(costs=costs), timings, hardware)
    row = next(r for r in rows if r["kind"] == "layer" and r["layer_id"] == 0)
    assert row["op_count"] == row["covered_op_count"] == 2
    assert row["useful_flops"] == 96
    assert row["mfu"] == pytest.approx(0.24)


def test_missing_cost_is_unavailable_and_has_reason(fixture):
    session, _, timings, hardware, _ = fixture
    rows = analyze_utilization(session.report(costs=CostRegistry()), timings, hardware)
    for row in rows:
        assert row["mfu"] is None and row["estimated_mbu"] is None
        assert row["covered_op_count"] == 0
        assert "cost rule unavailable" in row["mfu_missing_reason"]
        if row["kind"] == "op":
            assert "MFU=N/A Est.MBU=N/A" in row["nvtx_label"]
        else:
            assert row["nvtx_label"] == f"L{row['layer_id']}"


def test_missing_peaks_and_hbm_estimates_are_independent(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    rows = analyze_utilization(
        report, timings, replace(hardware, compute_flops_per_second={})
    )
    assert rows[0]["mfu"] is None and rows[0]["estimated_mbu"] is not None
    report["analysis"]["ops"][0]["cost"]["estimated_hbm_bytes"] = None
    rows = analyze_utilization(report, timings, hardware)
    assert rows[0]["mfu"] is not None and rows[0]["estimated_mbu"] is None
    assert rows[0]["mbu_missing_reason"] == "HBM traffic estimate unavailable"


def test_unqualified_device_time_invalidates_parent_utilization(fixture):
    session, costs, timings, hardware, _ = fixture
    timings[0] = replace(timings[0], unavailable_reason="different CUDA device")
    rows = analyze_utilization(session.report(costs=costs), timings, hardware)
    for row in (rows[0], rows[2]):
        assert row["mfu"] is None and row["estimated_mbu"] is None
        assert "different CUDA device" in row["mfu_missing_reason"]
    assert rows[1]["mfu"] is not None


def test_mixed_compute_types_do_not_invent_overall_mfu(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    report["analysis"]["ops"][1]["cost"]["compute_type"] = "other"
    # A parent layer encloses both ops; its own event window is the denominator.
    parent = report["layers"][0]
    label = parent["trace_label"]
    timings = [
        (
            replace(t, end_timestamp=max(x.end_timestamp for x in timings))
            if t.trace_label == label
            else t
        )
        for t in timings
    ]
    hardware = replace(
        hardware, compute_flops_per_second={"synthetic": 1000, "other": 2000}
    )
    rows = analyze_utilization(report, timings, hardware)
    row = next(r for r in rows if r["trace_label"] == label)
    assert row["mfu"] is None
    assert row["mfu_by_compute_type"] == pytest.approx(
        {"synthetic": 0.24, "other": 0.12}
    )
    assert row["estimated_mbu"] == pytest.approx(1.04)
    assert "mixed compute types" in row["mfu_missing_reason"]


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_device_time_never_uses_cpu_time(fixture, duration):
    session, costs, timings, hardware, _ = fixture
    timings[0] = replace(timings[0], gpu_duration_ns=duration)
    row = analyze_utilization(session.report(costs=costs), timings, hardware)[0]
    assert row["mfu"] is None and row["estimated_mbu"] is None
    assert "GPU duration unavailable" in row["mfu_missing_reason"]


def test_attribution_gaps_and_duplicate_timings_are_rejected(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    with pytest.raises(ValueError, match="exactly one timing"):
        analyze_utilization(report, timings[1:], hardware)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_utilization(report, timings + timings[:1], hardware)


def test_failed_layer_does_not_report_complete_utilization(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    report["layers"][0]["error"] = "RuntimeError"
    rows = analyze_utilization(report, timings, hardware)
    row = next(r for r in rows if r["kind"] == "layer")
    assert row["mfu"] is None and row["estimated_mbu"] is None
    assert row["mfu_missing_reason"] == "layer failed: RuntimeError"


@pytest.mark.parametrize("peak", [0, -1, float("nan"), float("inf")])
def test_invalid_hardware_peaks_rejected(peak):
    with pytest.raises(ValueError, match="finite and positive"):
        HardwareProfile("test", {"type": peak}, 1000, "test")
    with pytest.raises(ValueError, match="finite and positive"):
        HardwareProfile("test", {}, peak, "test")


@pytest.mark.parametrize(
    "kind,source,role,total,bus",
    [
        ("all_reduce_sum", "input", "a", 24, 24),
        ("reduce_scatter_sum", "input", "a", 24, 12),
        ("all_gather", "output", "return", 32, 16),
    ],
)
def test_collective_bytes_and_busbw(fixture, kind, source, role, total, bus):
    session, _, timings, hardware, _ = fixture
    # Reuse real tensor metadata to check the offline full-array role and units.
    rule = CollectiveCost(kind, role, source, 2, "tp")
    cost = rule(session.ops[0])
    assert cost.communication.total_bytes == total  # dtype size is already included.
    assert cost.communication.bus_bytes == bus
    expected = 448 if kind == "all_reduce_sum" else 224
    assert CollectiveTraffic(kind, 256, 8, "tp").bus_bytes == expected
    costs = CostRegistry()
    costs.register("mm", rule)
    row = analyze_utilization(session.report(costs=costs), timings, hardware)[0]
    assert row["busbw_bytes_per_second"] == pytest.approx(bus / 0.1)
    assert row["algbw_bytes_per_second"] == pytest.approx(total / 0.1)
    assert row["mfu"] is None and row["estimated_mbu"] is None
    assert "Est.MBU=N/A" in row["nvtx_label"]
    assert "MFU=" not in row["nvtx_label"]
    assert "busbw=" not in row["nvtx_label"]


def test_layer_busbw_window_and_unavailable_results(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    report["analysis"]["ops"][0]["cost"] = asdict(
        CollectiveCost("all_reduce_sum", "a", "input", 2, "tp")(session.ops[0])
    )
    layer_label = session.layers[0].trace_label
    timings = [
        (
            replace(t, end_timestamp=max(x.end_timestamp for x in timings))
            if t.trace_label == layer_label
            else t
        )
        for t in timings
    ]
    rows = analyze_utilization(report, timings, hardware)
    assert rows[0]["busbw_bytes_per_second"] == pytest.approx(24 / 0.1)
    assert rows[2]["busbw_bytes_per_second"] == pytest.approx(24 / 0.2)
    assert rows[2]["mfu"] == pytest.approx(48 / (1000 * 0.2))
    invalid = [replace(timings[0], gpu_duration_ns=0), *timings[1:]]
    assert (
        analyze_utilization(report, invalid, hardware)[0]["busbw_bytes_per_second"]
        is None
    )
    report["layers"][0]["error"] = "failed"
    assert (
        analyze_utilization(report, timings, hardware)[2]["busbw_bytes_per_second"]
        is None
    )
    report["layers"][0]["error"] = None
    saved = report["analysis"]["ops"][1]["cost"]
    report["analysis"]["ops"][1].update(cost=None, missing_reason="unknown")
    row = analyze_utilization(report, timings, hardware)[2]
    assert row["busbw_bytes_per_second"] is None
    assert "incomplete cost coverage" in row["busbw_missing_reason"]
    report["analysis"]["ops"][1]["cost"] = saved
    saved["communication"] = asdict(CollectiveTraffic("all_reduce_sum", 32, 2, "dp"))
    row = analyze_utilization(report, timings, hardware)[2]
    assert row["busbw_bytes_per_second"] is None
    assert "mixed communication groups" in row["busbw_missing_reason"]
    assert CollectiveTraffic("all_reduce_sum", 64, 1, "one").bus_bytes == 0
    with pytest.raises(ValueError, match="positive integer"):
        CollectiveTraffic("all_reduce_sum", 64, 0, "empty")


def test_attention_cost_uses_query_heads_and_causal_pairs():
    def attention(q, k, v, *, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=True
        )

    registry = KernelRegistry()
    registry.register(
        KernelSpec(name="test", family="attention", mode="gqa"),
        attention,
        profile=OpProfile(cost_rule="attention"),
    )
    wrapped = registry.get_impl("test")
    session = ProfileSession(
        run_id="attention", model_id="test", rank=0, backend="none"
    )
    cases = (
        (4, 4, "full", 16),
        (4, 4, "causal_lower_right", 10),
        (1, 8, "causal_lower_right", 8),
        (2, 5, "causal_lower_right", 9),
        (5, 2, "causal_lower_right", 3),
        (2, 5, "causal_upper_left", 3),
        (5, 2, "causal_upper_left", 9),
    )
    for iter_id, (query_length, kv_length, mask, pairs) in enumerate(cases):
        attn_mask = None
        if mask != "full":
            offset = kv_length - query_length if mask == "causal_lower_right" else 0
            attn_mask = (
                torch.arange(kv_length)[None, :]
                <= torch.arange(query_length)[:, None] + offset
            )
            assert attn_mask.sum().item() == pairs
        with session.iteration(iter_id=iter_id):
            wrapped(
                torch.randn(2, 4, query_length, 3),
                torch.randn(2, 2, kv_length, 3),
                torch.randn(2, 2, kv_length, 5),
                attn_mask=attn_mask,
            )
        rule = AttentionMatmulCost(q="q", k="k", v="v", output="return", mask=mask)
        cost = rule(session.ops[-1])
        assert cost.useful_flops == 2 * 2 * 4 * pairs * (3 + 5)
        assert cost.executed_flops is None and cost.estimated_hbm_bytes is None
