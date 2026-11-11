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

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from math import prod

import pytest
import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    OpProfile,
    OpRecord,
    ProfileSession,
    TensorDescription,
    current_op_metadata,
    profile_engine_contract,
    profile_session_active,
    record_selected_kernel,
    wrap_op_for_profiling,
)
from tokenspeed_kernel.profile_attention_moe import (
    grouped_experts_cost_v2,
    mla_decode_cost_v2,
    mla_project_value_cost_v2,
)
from tokenspeed_kernel.profile_utilization import (
    HardwareProfile,
    ScopeTiming,
    analyze_utilization,
)
from tokenspeed_kernel.registry import KernelSpec


def description(shape, size=2, dtype="torch.bfloat16"):
    stride = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
    return TensorDescription(tuple(shape), dtype, stride, "cuda:0", prod(shape) * size)


def record(inputs, output, metadata, values, rule):
    return OpRecord(
        0,
        None,
        None,
        0,
        "diagnostic.op",
        "diagnostic_wrapper",
        rule,
        inputs,
        {"return": output},
        0,
        100,
        None,
        "fixture",
        scalar_args=metadata,
        tensor_values=values,
    )


def mla_record(layout, final=136, batch=1):
    width, heads, latent, tail = 8, 8, 512, 64
    if layout == "query_axis":
        rows, axis, lengths = batch, width, (final,) * batch
        ids, positions = tuple(range(batch)), (0,) * batch
    else:
        rows, axis = batch * width, 1
        lengths = tuple(
            final - width + j + 1 for _ in range(batch) for j in range(width)
        )
        ids = tuple(b for b in range(batch) for _ in range(width))
        positions = tuple(j for _ in range(batch) for j in range(width))
    inputs = {
        "q": description((rows, axis, heads, latent + tail)),
        "kv_cache": description((32, 64, 1, latent + tail)),
        "page_table": description((rows, 16), 4, "torch.int32"),
        "cache_seqlens": description((rows,), 4, "torch.int32"),
    }
    metadata = dict(
        mla_logical_batch=batch,
        mla_query_width=width,
        kv_lora_rank=latent,
        qk_rope_head_dim=tail,
        mla_dcp_size=1,
        mla_attention_tp_size=8,
        window_left=-1,
        noncausal_block_size=1,
        mla_cache_policy="logical_request_once.v2",
        mla_layout=layout,
        mla_row_request_ids=ids,
        mla_row_query_positions=positions,
        return_lse=False,
    )
    return record(
        inputs,
        description((rows, axis, heads, latent)),
        metadata,
        {"cache_seqlens": lengths},
        "diagnostic_mla_decode.v2",
    )


@pytest.mark.parametrize("layout", ["query_axis", "flattened"])
def test_mla_cost_matches_known_causal_and_request_union_cost(layout):
    op = mla_record(layout)
    cost = mla_decode_cost_v2(op)
    # Both layouts attend 1060 causal pairs and read the same 136-row KV
    # union; their actual metadata interfaces contain one vs eight rows.
    expected_reads = {"query_axis": 230416, "flattened": 230528}[layout]
    assert cost.useful_flops == 18452480
    assert cost.logical_read_bytes == expected_reads
    assert cost.logical_write_bytes == 65536
    assert cost.estimated_hbm_bytes == expected_reads + 65536
    assert cost.executed_flops is None
    assert cost.compute_type is None
    assert cost.rule == "diagnostic_mla_decode.v2"


def test_mla_layouts_share_useful_flops_and_request_union_not_metadata_rows():
    axis, flat = (
        mla_decode_cost_v2(mla_record(layout)) for layout in ("query_axis", "flattened")
    )
    assert axis.useful_flops == flat.useful_flops
    # Eight actual metadata rows vs one, all with 3 touched page-table entries.
    assert flat.logical_read_bytes - axis.logical_read_bytes == (8 - 1) * (4 + 3 * 4)
    advanced = mla_decode_cost_v2(mla_record("flattened", final=137))
    assert advanced.useful_flops - flat.useful_flops == 2 * 8 * (576 + 512) * 8


@pytest.mark.parametrize(
    "failure", ["missing", "wrong_lengths", "wrong_mapping", "dcp", "policy"]
)
def test_mla_missing_or_incompatible_live_metadata_is_unavailable(failure):
    op = mla_record("flattened")
    metadata, values = dict(op.scalar_args), dict(op.tensor_values)
    if failure == "missing":
        values["cache_seqlens"] = None
    elif failure == "wrong_lengths":
        values["cache_seqlens"] = (136,) * 8
    elif failure == "wrong_mapping":
        metadata.pop("mla_row_query_positions")
    elif failure == "dcp":
        metadata["mla_dcp_size"] = 2
    else:
        metadata["mla_cache_policy"] = "implicit_sharing"
    with pytest.raises(ValueError):
        mla_decode_cost_v2(replace(op, scalar_args=metadata, tensor_values=values))


def test_mla_only_exact_single_leaf_contract_selects_fp32_simt():
    op = mla_record("flattened")
    metadata = {
        **op.scalar_args,
        "selected_kernel_names": ("triton_mla_decode_with_kvcache",),
        "selected_solutions": ("triton",),
        "selected_compute_types": ("fp32_simt",),
        "selected_engine_contract_ids": ("mla.triton_decode.fp32_simt.v1",),
        "selected_compute_type": "fp32_simt",
        "selected_engine_contract_id": "mla.triton_decode.fp32_simt.v1",
    }
    assert (
        mla_decode_cost_v2(replace(op, scalar_args=metadata)).compute_type
        == "fp32_simt"
    )
    for change in (
        {"selected_kernel_names": ("diagnostic_mla_decode",)},
        {"selected_kernel_names": ("triton_mla_decode_with_kvcache", "torch.bmm")},
        {"selected_engine_contract_id": None},
        {"selected_kernel_names": None},
        {"selected_compute_types": (None,)},
        {"selected_solutions": ("unknown",)},
    ):
        assert (
            mla_decode_cost_v2(
                replace(op, scalar_args={**metadata, **change})
            ).compute_type
            is None
        )


def test_portable_contract_cannot_normalize_an_unsupported_query_axis_width():
    op = mla_record("query_axis")
    metadata = {
        **op.scalar_args,
        "selected_kernel_names": ("triton_mla_decode_with_kvcache",),
        "selected_solutions": ("triton",),
        "selected_compute_types": ("fp32_simt",),
        "selected_engine_contract_ids": ("mla.triton_decode.fp32_simt.v1",),
        "selected_compute_type": "fp32_simt",
        "selected_engine_contract_id": "mla.triton_decode.fp32_simt.v1",
    }
    assert mla_decode_cost_v2(replace(op, scalar_args=metadata)).compute_type is None


def expert_record(routes=((2, 7), (2, 3))):
    rows, topk, hidden, useful, allocated, experts = (
        len(routes),
        len(routes[0]),
        4,
        6,
        8,
        2,
    )
    inputs = dict(
        hidden_states=description((rows, hidden)),
        routes=description((rows * topk,), 4, "torch.int32"),
        weights=description((rows, topk), 4, "torch.float32"),
        w13=description((experts, 2 * allocated, hidden), 1, "torch.float8_e4m3fn"),
        w2=description((experts, hidden, allocated), 1, "torch.float8_e4m3fn"),
        scale13=description((experts, 4, 1), 4, "torch.float32"),
        scale2=description((experts, 1, 2), 4, "torch.float32"),
    )
    metadata = dict(
        route_policy="global_expert_id",
        expert_placement="trivial",
        enable_eplb=False,
        redundant_experts=0,
        zero_experts=0,
        all_to_all=False,
        tp_size=1,
        dcp_size=1,
        do_finalize=True,
        shared_projection=False,
        hidden_dim=hidden,
        useful_intermediate=useful,
        allocated_intermediate=allocated,
        local_experts=experts,
        num_experts=8,
        ep_size=4,
        ep_rank=1,
        group_count=1,
        logical_tokens=rows,
        weight_format="fp8",
        scale_block_shape=(4, 4),
    )
    return record(
        inputs,
        description((rows, hidden)),
        metadata,
        {"routes": tuple(i for row in routes for i in row)},
        "diagnostic_grouped_experts.v2",
    )


@pytest.mark.parametrize(
    "routes,flops,reads",
    [(((2, 7), (2, 3)), 432, 288), (((0, 7), (0, 7)), 0, 48)],
)
def test_expert_instances_unique_weights_and_actual_scales_match_known_cost(
    routes, flops, reads
):
    op = expert_record(routes)
    cost = grouped_experts_cost_v2(op)
    # Repeated routes count three products but read each of two local experts
    # once. The empty owner still reads the 48-byte input/route boundary.
    assert cost.useful_flops == flops
    assert cost.logical_read_bytes == reads
    assert cost.logical_write_bytes == 16
    assert cost.estimated_hbm_bytes == reads + 16
    assert cost.executed_flops is None
    assert cost.compute_type is None


def test_missing_scales_leave_useful_flops_known_and_mbu_unavailable():
    op = expert_record()
    known = grouped_experts_cost_v2(op)
    missing = grouped_experts_cost_v2(
        replace(op, inputs={k: v for k, v in op.inputs.items() if k != "scale13"})
    )
    assert missing.useful_flops == known.useful_flops == 6 * 3 * 4 * 6
    assert missing.estimated_hbm_bytes is None
    assert any("incomplete" in assumption for assumption in missing.assumptions)


@pytest.mark.parametrize(
    "change",
    [
        {"zero_experts": 1},
        {"all_to_all": True},
        {"tp_size": 8},
        {"route_policy": "gmoe_aware"},
        {"expert_placement": "dynamic"},
        {"enable_eplb": True},
        {"useful_intermediate": None},
        {"do_finalize": False},
    ],
)
def test_unsupported_expert_policy_or_missing_pre_padding_geometry_is_unavailable(
    change,
):
    op = expert_record()
    with pytest.raises(ValueError):
        grouped_experts_cost_v2(replace(op, scalar_args={**op.scalar_args, **change}))


def test_missing_current_routes_are_not_replaced_by_expected_balance():
    op = expert_record()
    with pytest.raises(ValueError, match="Actual per-call routing IDs unavailable"):
        grouped_experts_cost_v2(replace(op, tensor_values={"routes": None}))


def test_unknown_expert_engine_keeps_flops_and_io_but_not_mfu():
    op = expert_record()
    session = ProfileSession(run_id="test", model_id="fixture", rank=0, backend="none")
    session.ops.append(op)
    registry = CostRegistry()
    registry.register("diagnostic_grouped_experts.v2", grouped_experts_cost_v2)
    report = session.report(costs=registry)
    hardware = HardwareProfile(
        "fixture", {"fp8_e4m3_tensor": 1000}, 1000, "analytical fixture"
    )
    result = analyze_utilization(
        report, [ScopeTiming("fixture", 0, 100, 100)], hardware
    )[0]
    assert result["useful_flops"] == 432
    assert result["estimated_mbu"] is not None
    assert result["mfu"] is None and result["mfu_total"] is None
    assert "no proven compute type" in result["mfu_missing_reason"]


def test_actual_active_record_identity_is_serializable_without_child_or_scalar_overwrite():
    @profile_engine_contract(
        compute_type="fp32_simt", contract_id="mla.triton_decode.fp32_simt.v1"
    )
    def leaf(x):
        record_selected_kernel(
            "triton_mla_decode_with_kvcache",
            "triton",
            implementation=leaf,
        )
        return x + 1

    def diagnostic(x, compute_type):
        assert profile_session_active()
        return leaf(x)

    wrapped = wrap_op_for_profiling(
        diagnostic,
        spec=KernelSpec(name="diagnostic_wrapper", family="test", mode="identity"),
        profile=OpProfile(cost_rule=None, scalar_args=("compute_type",)),
    )
    session = ProfileSession(
        run_id="identity", model_id="fixture", rank=0, backend="none"
    )
    assert not profile_session_active()
    with session.iteration(iter_id=0):
        output = wrapped(torch.tensor([1]), compute_type="caller_owned")
    assert output.item() == 2 and len(session.ops) == 1
    op = session.ops[0]
    assert op.implementation == "diagnostic_wrapper"
    assert op.scalar_args["compute_type"] == "caller_owned"
    assert op.scalar_args["selected_kernel_names"] == (
        "triton_mla_decode_with_kvcache",
    )
    assert op.scalar_args["selected_compute_type"] == "fp32_simt"
    serialized = json.loads(json.dumps(session.report(costs=CostRegistry())))["ops"][0][
        "scalar_args"
    ]
    assert serialized["selected_kernel_names"] == ["triton_mla_decode_with_kvcache"]
    assert serialized["selected_engine_contract_ids"] == [
        "mla.triton_decode.fp32_simt.v1"
    ]
    assert current_op_metadata() is None and not profile_session_active()


def test_multiple_selected_leaves_clear_single_engine_and_keep_order():
    @profile_engine_contract(
        compute_type="fp32_simt", contract_id="mla.triton_decode.fp32_simt.v1"
    )
    def leaf(x):
        return x

    def composite(x):
        record_selected_kernel(
            "triton_mla_decode_with_kvcache",
            "triton",
            implementation=leaf,
        )
        record_selected_kernel("torch.bmm", "torch", implementation=torch.bmm)
        return x

    wrapped = wrap_op_for_profiling(
        composite,
        spec=KernelSpec(name="diagnostic", family="test", mode="composite"),
        profile=OpProfile(cost_rule=None),
    )
    session = ProfileSession(
        run_id="multiple", model_id="fixture", rank=0, backend="none"
    )
    with session.iteration(iter_id=0):
        wrapped(torch.tensor([1]))
    metadata = session.ops[0].scalar_args
    assert metadata["selected_kernel_names"] == (
        "triton_mla_decode_with_kvcache",
        "torch.bmm",
    )
    assert metadata["selected_compute_types"] == ("fp32_simt", None)
    assert (
        metadata["selected_compute_type"] is None
        and metadata["selected_engine_contract_id"] is None
    )
    json.dumps(asdict(session.ops[0]))


def test_torch_value_projection_retains_flops_and_io_without_engine_proof():
    op = record(
        {"attention": description((8, 8, 512)), "weight": description((8, 512, 128))},
        description((8, 1024)),
        {},
        {},
        "diagnostic_mla_project_value.v2",
    )
    cost = mla_project_value_cost_v2(op)
    assert cost.useful_flops == 2 * 8 * 8 * 512 * 128
    assert (
        cost.estimated_hbm_bytes
        == sum(t.logical_bytes for t in op.inputs.values())
        + op.outputs["return"].logical_bytes
    )
    assert cost.compute_type is None


def test_grouped_expert_group_major_routes_with_ep8():
    groups, tokens, topk, hidden, intermediate, local = 8, 8, 12, 512, 1024, 384
    route_rows = tuple(
        tuple(group * local + k for k in range(topk))
        for group in range(groups)
        for _ in range(tokens)
    )
    op = expert_record(route_rows)
    metadata = {
        **op.scalar_args,
        "hidden_dim": hidden,
        "useful_intermediate": intermediate,
        "allocated_intermediate": intermediate,
        "local_experts": local,
        "num_experts": groups * local,
        "ep_size": 8,
        "ep_rank": 0,
        "group_count": groups,
        "logical_tokens": tokens,
        "scale_block_shape": (128, 128),
    }
    rows = groups * tokens
    inputs = dict(
        hidden_states=description((rows, hidden)),
        routes=description((rows * topk,), 4, "torch.int32"),
        weights=description((rows, topk), 4, "torch.float32"),
        w13=description((local, 2 * intermediate, hidden), 1, "torch.float8_e4m3fn"),
        w2=description((local, hidden, intermediate), 1, "torch.float8_e4m3fn"),
        scale13=description((local, 16, 4), 4, "torch.float32"),
        scale2=description((local, 4, 8), 4, "torch.float32"),
    )
    op = replace(
        op,
        inputs=inputs,
        outputs={"return": description((rows, hidden))},
        scalar_args=metadata,
    )
    cost = grouped_experts_cost_v2(op)
    # This rank owns 96 actual route instances but only 12 unique experts.
    # R=65536 hidden +6144 route metadata +18874368 weights +4608 scales.
    assert cost.useful_flops == 301989888
    assert cost.logical_read_bytes == 18950656
    assert cost.logical_write_bytes == 65536
    assert cost.estimated_hbm_bytes == 19016192
    bad_routes = tuple(0 for _ in op.tensor_values["routes"])
    with pytest.raises(ValueError, match="group-major row mapping"):
        grouped_experts_cost_v2(replace(op, tensor_values={"routes": bad_routes}))


@pytest.mark.parametrize(
    "variant",
    ["builtin", "same_name", "copied_wraps", "malformed_marker", "malformed_compute"],
)
def test_mla_engine_proof_belongs_to_selected_callable(variant):
    from functools import wraps

    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import SelectedKernel

    # A CPU leaf exercises the real declaration/registration/selected-callable
    # behavior. No GPU implementation is imported, extracted or executed.
    @profile_engine_contract(
        compute_type="fp32_simt", contract_id="mla.triton_decode.fp32_simt.v1"
    )
    def declared(x):
        return x + 1

    registry = KernelRegistry()
    spec = KernelSpec(
        name="triton_mla_decode_with_kvcache", family="attention", mode="mla_decode"
    )
    registry.register(spec, declared)
    if variant != "builtin":

        def replacement(x):
            return x + 2

        replacement.__name__ = declared.__name__
        if variant == "copied_wraps":
            replacement = wraps(declared)(replacement)
        elif variant == "malformed_marker":
            replacement._tokenspeed_profile_engine_contract = (
                "fp32_simt",
                "mla.triton_decode.fp32_simt.v1",
            )
        elif variant == "malformed_compute":
            replacement._tokenspeed_profile_engine_contract = replace(
                declared._tokenspeed_profile_engine_contract,
                implementation=replacement,
                compute_type="not_an_engine",
            )
        registry.register(spec, replacement)
    kernel = SelectedKernel(spec.name, registry.get_impl(spec.name))

    def diagnostic(x):
        record_selected_kernel(kernel.name, "triton", implementation=kernel.impl)
        return kernel(x)

    wrapped = wrap_op_for_profiling(
        diagnostic,
        spec=KernelSpec(name="diagnostic", family="attention", mode="mla_decode"),
        profile=OpProfile(cost_rule=None),
    )
    session = ProfileSession(run_id="proof", model_id="fixture", rank=0, backend="none")
    with session.iteration(iter_id=0):
        output = wrapped(torch.tensor([1]))
    metadata = session.ops[0].scalar_args
    assert output.item() == (2 if variant == "builtin" else 3)
    assert metadata["selected_compute_type"] == (
        "fp32_simt" if variant == "builtin" else None
    )
    op = mla_record("flattened")
    op = replace(op, scalar_args={**op.scalar_args, **metadata})
    cost_session = ProfileSession(
        run_id="cost", model_id="fixture", rank=0, backend="none"
    )
    cost_session.ops.append(op)
    costs = CostRegistry()
    costs.register("diagnostic_mla_decode.v2", mla_decode_cost_v2)
    result = analyze_utilization(
        cost_session.report(costs=costs),
        [ScopeTiming("fixture", 0, 100, 100)],
        HardwareProfile("fixture", {"fp32_simt": 1000}, 1000, "analytical fixture"),
    )[0]
    assert (result["mfu"] is not None) == (variant == "builtin")


def test_output_bank_conv_pad_reads_only_source_index_and_leaves_output_untouched():
    from tokenspeed_kernel.profile_costs import CausalConv1dUpdateCost

    op = record(
        dict(
            x=description((1, 3072)),
            weight=description((3072, 4)),
            conv_state=description((32, 3072, 3)),
            conv_state_indices=description((1,), 4, "torch.int32"),
            output_state_indices=description((1, 1), 4, "torch.int32"),
        ),
        description((1, 3072)),
        {"pad_slot_id": -1},
        {"conv_state_indices": (-1,), "output_state_indices": (999,)},
        "causal_conv1d_update.v3",
    )
    op = replace(
        op,
        op_name="conv.causal_conv1d_update",
        implementation="triton_causal_conv1d_update",
    )
    rule = CausalConv1dUpdateCost(hbm_policy="logical_io_once")
    cost = rule(op)
    assert (
        cost.useful_flops,
        cost.logical_read_bytes,
        cost.logical_write_bytes,
        cost.estimated_hbm_bytes,
    ) == (0, 4, 0, 4)
    assert (
        CausalConv1dUpdateCost(hbm_policy="unavailable")(op).estimated_hbm_bytes is None
    )
    live = replace(
        op, tensor_values={"conv_state_indices": (0,), "output_state_indices": (-1,)}
    )
    cost = rule(live)
    assert (cost.useful_flops, cost.logical_read_bytes, cost.logical_write_bytes) == (
        24576,
        49160,
        6144,
    )
    for source in (-2, 32):
        with pytest.raises(ValueError, match="neither pad_slot_id"):
            rule(
                replace(
                    op,
                    tensor_values={**op.tensor_values, "conv_state_indices": (source,)},
                )
            )


def include_fixture(tmp_path):
    from tokenspeed_kernel.op_profile import (
        _profile_fingerprint,
        _profile_include_candidate,
        _profile_include_search_roots,
    )

    package = tmp_path / "package"
    header = package / "include/cute/arch/mma_sm90_gmma.hpp"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"CPU fixture primitive, not GPU instruction evidence")
    snapshot = dict(
        modules=[dict(name="deep_gemm", root=str(package))],
        configured_roots={},
        environment={"CPATH": str(tmp_path / "nonhit")},
        working_directory=str(tmp_path),
    )
    entries = _profile_include_search_roots(
        {"deep_gemm": str(package)}, {}, snapshot["environment"], str(tmp_path)
    )
    snapshot["include_search"] = dict(
        schema="tokenspeed.gmma_include_search/v1", entries=entries, complete=True
    )
    snapshot["include_candidates"] = [
        _profile_include_candidate(entry, _profile_fingerprint) for entry in entries
    ]
    return snapshot, hashlib.sha256(header.read_bytes()).hexdigest()


def test_closed_include_search_keeps_negative_results_and_rejects_unknown_inputs(
    tmp_path,
):
    import copy

    from tokenspeed_kernel.op_profile import (
        _profile_fingerprint,
        _profile_include_candidate,
    )
    from tokenspeed_kernel.profile_costs import _include_search_error

    snapshot, digest = include_fixture(tmp_path)
    baseline = copy.deepcopy(snapshot)
    assert snapshot["include_candidates"][1]["presence"] == "proved_absent_ENOENT"
    assert _include_search_error(snapshot, baseline, [digest]) is None
    changes = [
        lambda x: x["include_candidates"][0].update(sha256="0" * 64),
        lambda x: x["include_candidates"][1].update(exception_path="/different/header"),
        lambda x: x["include_candidates"][1].update(
            presence="unavailable", unavailable_reason="PermissionError: EACCES"
        ),
        lambda x: x["environment"].update(
            CPATH=x["environment"]["CPATH"] + ":/unrepresented"
        ),
        lambda x: x["include_candidates"].reverse(),
    ]
    for change in changes:
        bad = copy.deepcopy(snapshot)
        change(bad)
        assert _include_search_error(bad, baseline, [digest]) is not None
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "package/include", target_is_directory=True)
    item = _profile_include_candidate(
        dict(root=str(link), origin="CPATH[0]"), _profile_fingerprint
    )
    assert item["presence"] == "unavailable"


def epoch_resolver_fixture(tmp_path, *, legacy_config):
    """Synthetic CPU proof inputs exercise the real resolver, never GPU truth."""
    import copy
    import shlex
    import time

    from tokenspeed_kernel.op_profile import OpCost
    from tokenspeed_kernel.profile_costs import register_selected_engine_resolver

    observed, primitive = include_fixture(tmp_path)
    lower = time.time_ns() + 1
    run_id = "a" * 16 + "-1"

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    config = dict(
        run_id=run_id,
        output_dir=str(tmp_path),
        hardware="fixture.json",
        layers="layers",
        start_step=1,
        num_steps=1,
        max_scopes=16,
        hbm_policy="logical_io_once",
    )
    driver = dict(path="/trusted/service_tpot.py", sha256="f" * 64, bytes=1)
    body = dict(
        schema="tokenspeed.before_engine_observation/v1",
        run_id=run_id,
        arm="profiled",
        repetition=1,
        case_sha256="a" * 64,
        timeline_config=config,
        timeline_config_sha256=digest(config),
        driver_source=driver,
        process_id=111,
        parent_process_id=110,
        realtime_before_ns=lower - 1,
        realtime_after_ns=lower,
        monotonic_ns=1000,
        clock_identity=dict(
            boot_id="boot", time_namespace="time:[1]", pid_namespace="pid:[1]"
        ),
    )
    observation = dict(body=body, sha256=digest(body))
    from tokenspeed_kernel.op_profile import _profile_config_for_observation

    expanded = _profile_config_for_observation(observation, run_id=run_id)
    if legacy_config:
        expanded["library"] = None
    config_sha = digest(expanded)
    compiler = dict(
        path="/compiler",
        requested="/compiler",
        sha256="2" * 64,
        ctime_ns=lower - 100,
        unavailable_reason=None,
    )
    api = dict(
        path="/approved/api.so",
        sha256="d" * 64,
        ctime_ns=lower - 100,
        unavailable_reason=None,
    )
    root = observed["modules"][0]["root"]
    command = [
        "/compiler",
        "/tmp/123-compile/kernel.cu",
        "-I",
        root + "/include",
        "--output-file",
        "kernel.cubin",
    ]
    meta = dict(
        command=shlex.join(command),
        compiler_info=dict(path="/compiler"),
        config=dict(python_library_root=root, include_dirs=[root + "/include"]),
        compiler_options=[],
    )
    cubin = dict(
        relative_path="sm90.tile/kernel.cubin",
        sha256="e" * 64,
        birthtime_ns=lower + 20,
        ctime_ns=lower + 20,
        mtime_ns=lower + 20,
        inode=2,
        device=1,
        unavailable_reason=None,
    )
    cache_root = str(tmp_path / "jit")
    observed["environment"]["DG_JIT_CACHE_DIR"] = cache_root
    raw = dict(
        process_id=123,
        stat_process_id=123,
        start_ticks=7,
        namespace_pids=[123],
        ancestor_process_ids=[111],
        boot_id="boot",
        time_namespace="time:[1]",
        pid_namespace="pid:[1]",
        monotonic_ns=1001,
        realtime_before_ns=lower + 5,
        realtime_after_ns=lower + 6,
        unavailable_reason=None,
    )
    observed.update(
        schema="tokenspeed.fp8_source_provenance/v1",
        process_id=123,
        run_id=run_id,
        rank=0,
        started_unix_ns=lower + 40,
        finished_unix_ns=lower + 100,
        source_files=[],
        source_context=dict(
            run_id=run_id,
            rank=0,
            profile_config_sha256=config_sha,
            startup_observation=observation,
        ),
        workers=[dict(process_id=123, rank=0, process_identity=raw)],
        compilers=dict(deep_gemm=compiler),
        loaded_extensions=[api],
        startup_driver_source={
            **driver,
            "ctime_ns": lower - 100,
            "unavailable_reason": None,
        },
        caches=[
            dict(
                name="deep_gemm",
                root=cache_root + "/cache",
                complete=True,
                unavailable_reason=None,
                candidates=[
                    cubin,
                    dict(relative_path="sm90.tile/meta.json", metadata=meta),
                ],
            )
        ],
    )
    observed["baseline"] = copy.deepcopy(observed)
    kernel = dict(
        name="fixture_fp8",
        grid=[1, 1, 1],
        block=[32, 1, 1],
        registers_per_thread=8,
        static_shared_memory=0,
        dynamic_shared_memory=0,
        cluster=[1, 1, 1],
    )
    policy = dict(
        api_object_sha256s=[api["sha256"]],
        compiler_sha256s=dict(deep_gemm=[compiler["sha256"]]),
        deep_compiler_options=[],
        deep_command_options=["-I", "placeholder"],
        workflow_assumptions=["synthetic isolated compiler writer"],
    )
    rule = dict(
        contract_id="fixture.fp8.source",
        op_name="fixture.mm",
        implementation="fixture",
        source_sha256s=["b" * 64],
        compute_capability=[9, 0],
        library_prefixes=["api.so"],
        library_sha256s=[api["sha256"]],
        cases=[dict(boundary={}, native_launches=[kernel])],
        components=[dict(numerator=1, denominator=1, compute_type="fp8_e4m3_tensor")],
        engine_evidence_level="synthetic_cpu_fixture",
        module_identity="unproven",
        source_provenance=dict(
            source_contract_sha256="3" * 64,
            source_files=[],
            primitive_sha256s=[primitive],
            compiler_overrides={},
            generation_policy=policy,
            caches=[dict(name="deep_gemm", key_prefixes=["sm90"], cubin_sha256s=[])],
        ),
    )
    definitions = json.dumps(
        dict(
            schema="tokenspeed.reviewed_engine_guards/v1",
            semantic_sources=[],
            bindings=[rule],
        )
    ).encode()
    registry = CostRegistry()
    registry.register(
        "fixture", lambda op: OpCost(8, None, 2, 2, 4, "fixture", (), compute_type=None)
    )
    register_selected_engine_resolver(
        registry,
        definitions=definitions,
        expected_sha256=hashlib.sha256(definitions).hexdigest(),
    )
    op = replace(
        record(
            {"x": description((1,))},
            description((1,)),
            {"profile_source_sha256": "b" * 64},
            {},
            "fixture",
        ),
        op_name="fixture.mm",
        implementation="fixture",
    )
    session = ProfileSession(run_id=run_id, model_id="fixture", rank=0, backend="none")
    session.ops.append(op)
    context = dict(
        complete_join=True,
        capture_sha256="0" * 64,
        replay_identity=dict(process_id=123),
        run_id=run_id,
        rank=0,
        scope_identity=dict(scope_id=1),
        source_node_ids=[1],
        activity_bindings=[
            dict(
                table="CUPTI_ACTIVITY_KIND_KERNEL",
                source_rowid=1,
                node=11,
                source=1,
                kernel={**kernel, "device_id": 0, "context_id": 1},
            )
        ],
        hardware=dict(
            device=dict(compute_capability=[9, 0]),
            selection=dict(observed=dict(uuid="GPU-aaaa")),
        ),
        native_device_uuids={0: dict(uuid="aaaa", compute_capability=[9, 0])},
        profile_config_sha256=config_sha,
        startup_observation_sha256=observation["sha256"],
        loaded_library_manifest=dict(
            schema="tokenspeed.loaded_libraries/v1",
            process_id=123,
            libraries=[api],
            semantic_provenance=observed,
        ),
    )
    return registry, session, context


@pytest.mark.parametrize("legacy_config", [False, True])
def test_before_engine_epoch_binds_real_resolver_and_rejects_bad_lineage(
    tmp_path, legacy_config
):
    import copy

    registry, session, context = epoch_resolver_fixture(
        tmp_path, legacy_config=legacy_config
    )

    def evaluate(value):
        return registry.analyze(
            session, engine_context={session.ops[0].trace_label: value}
        )["ops"][0]

    accepted = evaluate(context)
    assert accepted["engine_proof"]["status"] == "bound"
    assert (
        accepted["cost"]["compute_components"][0]["compute_type"] == "fp8_e4m3_tensor"
    )
    assert (
        accepted["cost"]["useful_flops"],
        accepted["cost"]["logical_read_bytes"],
        accepted["cost"]["logical_write_bytes"],
    ) == (8, 2, 2)
    changes = [
        lambda x: x["source_context"]["startup_observation"].update(sha256="0" * 64),
        lambda x: x["source_context"]["startup_observation"]["body"].update(
            run_id="wrong"
        ),
        lambda x: x["source_context"]["startup_observation"]["body"][
            "timeline_config"
        ].update(run_id="wrong"),
        lambda x: x["startup_driver_source"].update(sha256="0" * 64),
        lambda x: x["caches"][0]["candidates"][0].update(birthtime_ns=1),
        lambda x: x["caches"][0]["candidates"][1]["metadata"].update(
            command="/compiler /tmp/999-compile/kernel.cu -I /unknown --output-file out"
        ),
        lambda x: x["source_context"].update(startup_observation=None),
        lambda x: x["baseline"]["source_context"].update(
            profile_config_sha256="0" * 64
        ),
        lambda x: x["environment"].update(CXXFLAGS="-I/unknown"),
    ]
    for change in changes:
        bad = copy.deepcopy(context)
        change(bad["loaded_library_manifest"]["semantic_provenance"])
        result = evaluate(bad)
        assert result["engine_proof"]["status"] != "bound"
        assert result["cost"]["compute_components"][0]["compute_type"] is None
        assert (
            result["cost"]["useful_flops"] == 8
            and result["cost"]["estimated_hbm_bytes"] == 4
        )
