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

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from math import prod
from pathlib import Path

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
from tokenspeed_kernel.profile_utilization import (
    HardwareProfile,
    ScopeTiming,
    analyze_utilization,
)
from tokenspeed_kernel.registry import KernelSpec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "decode-timeline"))
from case_costs import (
    grouped_experts_cost_v2,
    mla_decode_cost_v2,
    mla_project_value_cost_v2,
)

spec = importlib.util.spec_from_file_location(
    "accepted_cost_reference", ROOT / "outputs/dynamic-cost-design/reference_costs.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


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
def test_mla_cost_matches_accepted_integer_reference(layout):
    op = mla_record(layout)
    cost = mla_decode_cost_v2(op)
    expected = reference.mla_causal_cost(
        (136,),
        width=8,
        heads=8,
        latent=512,
        tail=64,
        page_size=64,
        layout=layout,
        cache_policy="logical_request_once.v2",
        query_bytes=2,
        cache_bytes=2,
        output_bytes=2,
        index_bytes=4,
    )
    for name in (
        "useful_flops",
        "logical_read_bytes",
        "logical_write_bytes",
        "executed_flops",
    ):
        assert getattr(cost, name) == expected[name]
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


@pytest.mark.parametrize("routes", [((2, 7), (2, 3)), ((0, 7), (0, 7))])
def test_expert_instances_unique_weights_and_actual_scales_match_reference(routes):
    op = expert_record(routes)
    cost = grouped_experts_cost_v2(op)
    expected = reference.expert_cost(
        routes,
        first_expert=2,
        local_experts=2,
        total_experts=8,
        hidden=4,
        useful_intermediate=6,
        allocated_intermediate=8,
        activation_bytes=2,
        output_bytes=2,
        id_bytes=4,
        route_weight_bytes=4,
        expert_weight_bytes=1,
        scale_bytes_per_expert=24,
    )
    for name in (
        "useful_flops",
        "logical_read_bytes",
        "logical_write_bytes",
        "executed_flops",
    ):
        assert getattr(cost, name) == expected[name]
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
    assert "compute type unavailable" in result["mfu_missing_reason"]


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


def test_grouped_expert_core_b1_width8_ep8_reference():
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
    expected = reference.expert_cost(
        route_rows,
        first_expert=0,
        local_experts=local,
        total_experts=groups * local,
        hidden=hidden,
        useful_intermediate=intermediate,
        allocated_intermediate=intermediate,
        activation_bytes=2,
        output_bytes=2,
        id_bytes=4,
        route_weight_bytes=4,
        expert_weight_bytes=1,
        scale_bytes_per_expert=384,
    )
    assert expected["local_route_instances"] == 96
    assert cost.useful_flops == expected["useful_flops"] == 301989888
    assert cost.logical_read_bytes == expected["logical_read_bytes"]
    assert cost.logical_write_bytes == expected["logical_write_bytes"]
    bad_routes = tuple(0 for _ in op.tensor_values["routes"])
    with pytest.raises(ValueError, match="group-major row mapping"):
        grouped_experts_cost_v2(replace(op, tensor_values={"routes": bad_routes}))


@pytest.mark.parametrize(
    "variant",
    ["builtin", "same_name", "copied_wraps", "malformed_marker", "malformed_compute"],
)
def test_mla_engine_proof_belongs_to_selected_callable(fresh_registry, variant):
    import ast
    from functools import wraps

    from tokenspeed_kernel.platform import CapabilityRequirement
    from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
    from tokenspeed_kernel.selection import SelectedKernel
    from tokenspeed_kernel.signature import format_signatures

    # Execute the unchanged real registration/declaration, without GPU imports or
    # running its GPU body. Registry, SelectedKernel and OpRecord are real classes.
    path = (
        ROOT
        / "tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/triton.py"
    )
    tree = ast.parse(path.read_text())
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"_FP8_DTYPES", "_PORTABLE_DTYPES", "_PORTABLE_CAPABILITY"}
            for target in node.targets
        )
    ]
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "triton_mla_decode_with_kvcache"
    )
    namespace = dict(
        torch=torch,
        register_kernel=register_kernel,
        Priority=Priority,
        CapabilityRequirement=CapabilityRequirement,
        format_signatures=format_signatures,
        profile_engine_contract=profile_engine_contract,
    )
    exec(
        compile(
            ast.Module(body=[*assignments, function], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    builtin = namespace["triton_mla_decode_with_kvcache"]
    registry = KernelRegistry.get()
    spec = registry.get_by_name("triton_mla_decode_with_kvcache")
    if variant != "builtin":

        def replacement(x):
            return x + 1

        if variant == "copied_wraps":
            replacement = wraps(builtin)(replacement)
        if variant == "malformed_marker":
            replacement._tokenspeed_profile_engine_contract = (
                "fp32_simt",
                "mla.triton_decode.fp32_simt.v1",
            )
        if variant == "malformed_compute":
            replacement._tokenspeed_profile_engine_contract = replace(
                builtin._tokenspeed_profile_engine_contract,
                implementation=replacement,
                compute_type="not_an_engine",
            )
        registry.register(spec, replacement)
    kernel = SelectedKernel(spec.name, registry.get_impl(spec.name))
    helper_path = (
        ROOT
        / "tokenspeed/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/__init__.py"
    )
    helper = next(
        node
        for node in ast.parse(helper_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_record_selected_mla_leaf"
    )
    scope = dict(
        KernelRegistry=KernelRegistry,
        current_op_metadata=current_op_metadata,
        record_selected_kernel=record_selected_kernel,
    )
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(helper_path), "exec"),
        scope,
    )

    def diagnostic(x):
        scope["_record_selected_mla_leaf"](kernel)
        return x if variant == "builtin" else kernel(x)

    wrapped = wrap_op_for_profiling(
        diagnostic,
        spec=KernelSpec(name="diagnostic", family="attention", mode="mla_decode"),
        profile=OpProfile(cost_rule=None),
    )
    session = ProfileSession(run_id="proof", model_id="fixture", rank=0, backend="none")
    with session.iteration(iter_id=0):
        output = wrapped(torch.tensor([1]))
    metadata = session.ops[0].scalar_args
    assert output.item() == (1 if variant == "builtin" else 2)
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
