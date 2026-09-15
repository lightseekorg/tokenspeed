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

"""CPU policy and source-contract checks, not GPU or performance qualification."""

import ast
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/communication"
COMM = ROOT / "python/tokenspeed/runtime/models/kimi_k3_comm.py"
BUCKETS = (256, 384, 512, 768, 832, 896, 960, 1024)


def definitions(path, names, namespace):
    selected = []
    found = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            declared = {node.name}
        elif isinstance(node, ast.Assign):
            declared = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
        else:
            continue
        if declared & names:
            selected.append(node)
            found.update(declared & names)
    assert found == names
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def profile():
    namespace = {"dataclass": dataclass, "__name__": __name__}
    definitions(
        OPS / "medium_fused_rs_up_projection_config.py",
        {"MediumFusedRsUpProjectionTuning"},
        namespace,
    )
    return definitions(
        OPS / "medium_fused_rs_up_projection_serving_config.py",
        {
            "MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS",
            "MEDIUM_FUSED_RS_SERVING_QUALIFIED",
            "medium_fused_rs_serving_config",
            "integrated_fused_rs_serving_config",
        },
        namespace,
    )


def policy():
    names = {
        "K3MoETailTier",
        "TAIL_FUSION_MAX_TOKENS",
        "MULTIMEM_AR_MIN_TOKENS",
        "MULTIMEM_AR_MAX_TOKENS",
        "MNNVL_BT_QUALIFIED_TOKENS",
        "MNNVL_HT_QUALIFIED_TOKENS",
        "MNNVL_DEFERRED_ENABLE_ENV",
        "FUSED_RS_UP_AG_ENABLE_ENV",
        "FUSED_RS_UP_AG_TOKENS",
        "MNNVL_BT_ONLY_ENABLE_ENV",
        "MEDIUM_FUSED_RS_UP_AG_ENABLE_ENV",
        "_mnnvl_bt_only_enabled",
        "_medium_fused_rs_up_ag_enabled",
        "_medium_flags_valid",
        "_fused_rs_up_ag_enabled",
        "_mnnvl_deferred_enabled",
        "select_k3_moe_tail_tier",
        "select_integrated_k3_moe_tail_tier",
        "TailPlan",
    }
    return definitions(
        COMM,
        names,
        {"os": os, "IntEnum": IntEnum, "dataclass": dataclass, "__name__": __name__},
    )


def test_integrated_output_pool_allocates_two_slots_with_independent_layer_plans():
    allocations = []

    def allocate(group, capacity, *, device):
        output = SimpleNamespace(group=group, capacity=capacity, device=device)
        allocations.append(output)
        return output

    def adapter(workspace, capacity, *, output):
        return SimpleNamespace(
            workspace=workspace, capacity=capacity, output=output, _plans={}
        )

    workspace = SimpleNamespace(
        state=SimpleNamespace(group=object(), max_token_num=8192, device="cuda:0")
    )
    namespace = definitions(
        OPS / "medium_fused_rs_up_projection_serving.py",
        {"IntegratedFusedRsOutputPool"},
        {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)
            ),
            "integrated_fused_rs_serving_config": profile()[
                "integrated_fused_rs_serving_config"
            ],
            "allocate_symmetric_up_projection_output": allocate,
            "IntegratedFusedRsUpProjectionServing": adapter,
        },
    )
    pool = namespace["IntegratedFusedRsOutputPool"](workspace, 8192)
    layers = [pool.bind_layer(index) for index in range(2, 94)]
    assert len(allocations) == 2
    assert all(
        layer.output is allocations[index % 2] for index, layer in enumerate(layers)
    )
    assert len({id(layer._plans) for layer in layers}) == 92
    layers[0]._plans[8192] = "first layer weight binding"
    assert not layers[2]._plans
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="layer index"):
            pool.bind_layer(invalid)
    second = namespace["IntegratedFusedRsOutputPool"](workspace, 8192)
    assert all(left is not right for left in pool.outputs for right in second.outputs)
    assert len(allocations) == 4
    namespace["torch"].cuda.is_current_stream_capturing = lambda: True
    with pytest.raises(ValueError, match="precede capture"):
        namespace["IntegratedFusedRsOutputPool"](workspace, 8192)
    assert len(allocations) == 4


def test_integrated_policy_and_profile_cover_continuous_ranges():
    namespace = policy()
    select = namespace["select_integrated_k3_moe_tail_tier"]
    tiers = namespace["K3MoETailTier"]
    get = profile()["integrated_fused_rs_serving_config"]
    for m in range(8194):
        if m <= 32:
            assert select(m) is None
        elif m <= 1024:
            assert select(m) is tiers.MEDIUM_FUSED_RS_UP_AG
            config = get(m)
            assert config.ab_stages == 0
            assert (config.tile_m, config.tile_n) == (64, 64 if m <= 512 else 128)
            assert config.two_cta is False
            assert (config.cluster_m, config.cluster_n) == (1, 1)
        elif m <= 8192:
            assert select(m) is tiers.FUSED_RS_UP_AG
            config = get(m)
            assert (
                config.tile_m,
                config.tile_n,
                config.ab_stages,
                config.c_stages,
            ) == (256, 128, 6, 3)
            assert config.cluster_cap == (38 if m <= 4096 else None)
            assert config.two_cta is True
            assert (config.cluster_m, config.cluster_n) == (2, 1)
        else:
            assert select(m) is tiers.SEPARATE_REDUCE


@pytest.mark.parametrize(
    "m,geometry",
    [
        (256, (128, 64, True, 2, 1)),
        (384, (64, 64, False, 1, 1)),
        (512, (64, 64, False, 1, 1)),
        (768, (64, 128, False, 1, 1)),
        (832, (64, 128, False, 1, 1)),
        (896, (64, 128, False, 1, 1)),
        (960, (64, 128, False, 1, 1)),
        (1024, (64, 128, False, 1, 1)),
    ],
)
def test_profile_is_explicit_provisional_and_fresh(m, geometry):
    namespace = profile()
    assert not namespace["MEDIUM_FUSED_RS_SERVING_QUALIFIED"]
    assert namespace["MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS"] == frozenset(BUCKETS)
    get = namespace["medium_fused_rs_serving_config"]
    value = get(m)
    assert (
        value.tile_m,
        value.tile_n,
        value.two_cta,
        value.cluster_m,
        value.cluster_n,
    ) == geometry
    assert (
        value.ab_stages,
        value.c_stages,
        value.addend_stages,
        value.reduce_vectors,
    ) == (0, 0, 1, 4)
    assert value.scheduler_type == "static_persistent"
    assert value.cluster_cap is None
    assert value == get(m) and value is not get(m)


@pytest.mark.parametrize(
    "m", [True, 0, 32, 255, 257, 800, 1025, 1280, 2048, 4096, 8192, 256.0, "256"]
)
def test_profile_rejects_implicit_or_unlisted_buckets(m):
    with pytest.raises(ValueError):
        profile()["medium_fused_rs_serving_config"](m)


def test_new_flags_default_off_and_do_not_enable_legacy(monkeypatch):
    namespace = policy()
    for key in (
        "MNNVL_DEFERRED_ENABLE_ENV",
        "FUSED_RS_UP_AG_ENABLE_ENV",
        "MNNVL_BT_ONLY_ENABLE_ENV",
        "MEDIUM_FUSED_RS_UP_AG_ENABLE_ENV",
    ):
        monkeypatch.delenv(namespace[key], raising=False)
    assert not namespace["_mnnvl_bt_only_enabled"]()
    assert not namespace["_medium_fused_rs_up_ag_enabled"]()
    for raw in ("true", "yes", "0", ""):
        monkeypatch.setenv(namespace["MNNVL_BT_ONLY_ENABLE_ENV"], raw)
        assert not namespace["_mnnvl_bt_only_enabled"]()
    monkeypatch.setenv(namespace["MNNVL_BT_ONLY_ENABLE_ENV"], "1")
    monkeypatch.setenv(namespace["MEDIUM_FUSED_RS_UP_AG_ENABLE_ENV"], "1")
    assert namespace["_mnnvl_bt_only_enabled"]()
    assert namespace["_medium_fused_rs_up_ag_enabled"]()
    assert not namespace["_mnnvl_deferred_enabled"]()


@pytest.mark.parametrize(
    "bt,medium,legacy",
    [(b, m, l) for b in (False, True) for m in (False, True) for l in (False, True)],
)
def test_flag_scope_truth_table(bt, medium, legacy):
    expected = (not medium or bt) and (not legacy or not (bt or medium))
    assert policy()["_medium_flags_valid"](bt, medium, legacy) is expected


def route(m, bt, medium, prefill, decode):
    namespace = policy()
    namespace.update(
        get_is_cuda_graph_phase=lambda: False,
        get_is_prefill_graph_phase=lambda: prefill,
        _acquire_symm_join_outputs=lambda **kwargs: None,
        allreduce_fusion_lane=lambda *args, **kwargs: None,
    )
    tree = ast.parse(COMM.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "K3MoeTailComm"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "plan"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, method], type_ignores=[])
    )
    exec(compile(module, str(COMM), "exec"), namespace)
    adapter = (
        SimpleNamespace(input_view=lambda count: ("raw", count)) if medium else None
    )
    comm = SimpleNamespace(
        state=SimpleNamespace(
            mnnvl_bt_deferred=(
                SimpleNamespace(supports_num_tokens=lambda count: count <= 1024)
                if bt
                else None
            ),
            mnnvl_ht_deferred=None,
            bt_candidate_tokens=frozenset(BUCKETS),
            multimem_ar_ok=True,
            fused_rs_up_ag_capacity=0,
        ),
        execution_plan=SimpleNamespace(fused_moe_ar=True, join_moe_reduce=False),
        latent_tail=None,
        fused_rs_up_ag=None,
        medium_fused_rs_up_ag=adapter,
        _experts_supports_deferred_finalize=True,
        _shard_up_projection=True,
        mapping=None,
        routed_hidden=3584,
        hidden_size=7168,
    )
    return (
        namespace["plan"](comm, m, None, is_decode=decode),
        namespace["K3MoETailTier"],
    )


@pytest.mark.parametrize("m", BUCKETS)
def test_both_medium_arms_use_identical_bt_policy(m):
    bt, tier = route(m, True, False, True, False)
    fused, _ = route(m, True, True, True, False)
    main, _ = route(m, False, False, True, False)
    assert bt.tier.name == tier.MNNVL_BT_DEFERRED.name
    assert fused.tier.name == tier.MEDIUM_FUSED_RS_UP_AG.name
    assert main.tier.name == tier.MULTIMEM_AR.name
    assert bt.defer_finalize and fused.defer_finalize
    assert bt.symm_outputs is None
    assert fused.symm_outputs == (None, ("raw", m))


@pytest.mark.parametrize(
    "m,prefill,decode",
    [
        (1024, False, False),
        (1024, True, True),
        (800, True, False),
        (1280, True, False),
        (2048, True, False),
        (4096, True, False),
        (8192, True, False),
    ],
)
def test_medium_cannot_promote_eager_decode_or_unlisted_bucket(m, prefill, decode):
    plan, _ = route(m, True, True, prefill, decode)
    assert plan.tier.name not in {
        "MEDIUM_FUSED_RS_UP_AG",
        "MNNVL_HT_DEFERRED",
        "FUSED_RS_UP_AG",
    }
    assert not plan.defer_finalize and plan.symm_outputs is None


def test_medium_adapter_inherits_live_pointer_launch_without_override():
    path = OPS / "medium_fused_rs_up_projection_serving.py"
    owner = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
    )
    assert [ast.unparse(base) for base in owner.bases] == ["FusedRsUpProjectionServing"]
    assert {node.name for node in owner.body if isinstance(node, ast.FunctionDef)} == {
        "__init__",
        "input_view",
        "_prepare",
    }
    prepare = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare"
    )
    text = ast.unparse(prepare)
    assert "is_current_stream_capturing()" in text
    assert "bound._cute_args[1:4]" in text
    assert "bound.admission_records" in text
    assert "fixed_args=bound._cute_args[1:4]" in text
    assert "prepare_medium_fused_rs_up_projection" in text


def test_medium_run_uses_existing_bt_and_never_ht():
    owner = next(
        node
        for node in ast.parse(COMM.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "K3MoeTailComm"
    )
    run = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    branch = next(
        node
        for node in run.body
        if isinstance(node, ast.If)
        and "MEDIUM_FUSED_RS_UP_AG" in ast.unparse(node.test)
    )
    text = ast.unparse(branch)
    assert "self.state.mnnvl_bt_deferred(" in text
    assert "mnnvl_ht_deferred" not in text
    assert "fused_rs_up_ag_finalize" not in text
    assert "self.medium_fused_rs_up_ag(" in text


def test_allocation_and_flag_agreement_precede_medium_launch():
    source = COMM.read_text()
    assert "and not mnnvl_bt_only" in source
    assert "if medium_ok and self.multimem_ar_ok:" in source
    assert "if self.state.medium_fused_rs_up_ag_workspace is not None:" in source
    assert "bt_min != -bt_negative_min" in source
    assert "medium_min != -medium_negative_min" in source
    assert source.index("bt_min != -bt_negative_min") < source.index(
        "if medium_ok and self.multimem_ar_ok:"
    )
