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

"""CPU-only configuration and dispatch guards for the integrated fused tail."""

import ast
import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "python/tokenspeed_kernel/ops/communication"
spec = importlib.util.spec_from_file_location(
    "medium_fused_config_cpu_test", OPS / "medium_fused_rs_up_projection_config.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Tuning = module.MediumFusedRsUpProjectionTuning


def tuning(tile_m, tile_n, two_cta):
    return Tuning(
        tile_m=tile_m,
        tile_n=tile_n,
        two_cta=two_cta,
        cluster_m=2 if two_cta else 1,
        cluster_n=1,
        c_stages=0,
        ab_stages=0,
        addend_stages=1,
        reduce_vectors=4,
        cluster_cap=None,
        scheduler_type="static_persistent",
    )


@pytest.mark.parametrize(
    "geometry",
    [
        (64, 64, False),
        (64, 128, False),
        (128, 64, False),
        (128, 128, False),
        (128, 64, True),
        (128, 128, True),
        (256, 64, True),
        (256, 128, True),
    ],
)
def test_supported_geometry(geometry):
    value = tuning(*geometry)
    value.validate()
    assert value.epilogue_m == value.tile_m // (2 if value.two_cta else 1)
    assert value.epilogue_n == 64
    assert value.enable_pdl is False


@pytest.mark.parametrize(
    "updates",
    [
        {"tile_m": True},
        {"tile_n": 64.0},
        {"two_cta": 1},
        {"cluster_m": 2},
        {"cluster_n": 2},
        {"tile_n": 256},
        {"tile_m": 256},
        {"c_stages": 1},
        {"c_stages": True},
        {"ab_stages": True},
        {"ab_stages": 1},
        {"ab_stages": 22},
        {"ab_stages": 4},
        {"addend_stages": 0},
        {"addend_stages": 2},
        {"reduce_vectors": 3},
        {"reduce_vectors": 8},
        {"cluster_cap": 0},
        {"cluster_cap": True},
        {"scheduler_type": "clc"},
        {"scheduler_type": "nonpersistent"},
        {"scheduler_type": "full_grid", "cluster_cap": 32},
    ],
)
def test_rejects_unimplemented_controls(updates):
    with pytest.raises(ValueError):
        replace(tuning(64, 64, False), **updates).validate()


@pytest.mark.parametrize("group", [1, 2, 4])
def test_small_epilogue_groups(group):
    replace(tuning(64, 64, False), reduce_vectors=group).validate()


def test_wide_epilogue_has_eight_vectors_and_two_stage_requires_one():
    replace(tuning(128, 64, False), reduce_vectors=8).validate()
    replace(tuning(64, 64, False), addend_stages=2, reduce_vectors=1).validate()


def test_all_constructor_controls_are_explicit():
    import dataclasses

    assert len(fields(Tuning)) == 11
    assert all(field.default is dataclasses.MISSING for field in fields(Tuning))


def test_binding_keeps_fused_protocol_and_prelaunch_admission():
    source = (OPS / "medium_fused_rs_up_projection.py").read_text()
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    bound = next(
        node for node in classes if node.name == "BoundMediumFusedRsUpProjection"
    )
    assert [ast.unparse(base) for base in bound.bases] == ["BoundFusedRsUpProjection"]
    launch = next(
        node
        for node in bound.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    assert "super().__call__()" in ast.unparse(launch)
    assert "cuOccupancyMaxActiveClusters" in source
    assert "plan._compile()" in source
    assert source.index("plan._compile()") < source.index("plan._barrier()")
    assert "plan.admission_records" in source
    assert "register_kernel" not in source


def test_medium_kernel_inherits_device_reduction_and_body():
    path = (
        ROOT
        / "python/tokenspeed_kernel/thirdparty/cute_dsl/symmetric_up_projection/fused_shared_rs_medium.py"
    )
    tree = ast.parse(path.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert [ast.unparse(base) for base in klass.bases] == [
        "FusedSharedRsUpProjectionGemm"
    ]
    methods = {node.name for node in klass.body if isinstance(node, ast.FunctionDef)}
    assert methods == {
        "__init__",
        "configure_addend_pipeline",
        "_setup_attributes",
        "_compute_grid",
    }
    # No replacement device load, store, rank barrier or arithmetic is hidden
    # behind this host-geometry specialization.
    assert not any(
        isinstance(node, ast.FunctionDef) and node.decorator_list for node in klass.body
    )


@pytest.mark.parametrize("ab_stages", [2, 3, 4, 6, 21])
@pytest.mark.parametrize("c_stages", [2, 3])
def test_explicit_ab_stage_controls(ab_stages, c_stages):
    # The host schema admits the metadata bound; actual device layouts decide
    # the lower geometry-dependent shared-memory upper bound during compile.
    replace(tuning(64, 64, False), ab_stages=ab_stages, c_stages=c_stages).validate()


def test_zero_ab_stages_keeps_original_layouts():
    path = (
        ROOT
        / "python/tokenspeed_kernel/thirdparty/cute_dsl/symmetric_up_projection/fused_shared_rs_medium.py"
    )
    tree = ast.parse(path.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    setup = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "_setup_attributes"
    )
    assert ast.unparse(setup.body[0]) == "super()._setup_attributes()"
    branch = setup.body[1]
    assert isinstance(branch, ast.If)
    assert ast.unparse(branch.test) == "self.requested_ab_stages"
    assignments = [node for node in ast.walk(branch) if isinstance(node, ast.Assign)]
    targets = {ast.unparse(target) for node in assignments for target in node.targets}
    assert {
        "self.num_ab_stage",
        "self.a_smem_layout_staged",
        "self.b_smem_layout_staged",
    } <= targets
    assert "self.c_smem_layout_staged" not in targets
    assert "self.num_acc_stage" not in targets


@pytest.mark.parametrize(
    "geometry,m,problem",
    [
        ((64, 64, False), 256, 56),
        ((64, 64, False), 257, 70),
        ((64, 64, False), 1024, 224),
        ((64, 128, False), 1024, 112),
        ((128, 64, False), 1024, 112),
        ((128, 64, True), 1024, 112),
        ((128, 128, True), 832, 49),
        ((256, 64, True), 1024, 56),
        ((256, 128, True), 1024, 28),
    ],
)
@pytest.mark.parametrize("scheduler", ["static_persistent", "full_grid"])
def test_planned_cluster_grid(geometry, m, problem, scheduler):
    config = replace(tuning(*geometry), scheduler_type=scheduler)
    result = config.launch_geometry(m, 48)
    launched = problem if scheduler == "full_grid" else min(problem, 48)
    assert result == {
        "problem_clusters": problem,
        "launched_clusters": launched,
        "grid": [config.cluster_m, config.cluster_n, launched],
    }


def test_full_grid_does_not_equate_launch_count_with_residency():
    config = replace(tuning(64, 64, False), scheduler_type="full_grid")
    assert config.launch_geometry(1024, 1)["launched_clusters"] == 224
    with pytest.raises(ValueError):
        config.launch_geometry(1024, 0)
    with pytest.raises(ValueError):
        config.launch_geometry(32, 1)
    with pytest.raises(ValueError):
        replace(
            config, scheduler_type="static_persistent", cluster_cap=2
        ).launch_geometry(1024, 1)


def test_full_grid_reuses_original_scheduler_and_z_indexing():
    path = (
        ROOT
        / "python/tokenspeed_kernel/thirdparty/cute_dsl/symmetric_up_projection/fused_shared_rs_medium.py"
    )
    tree = ast.parse(path.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    grid = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_grid"
    )
    assert isinstance(grid.body[0], ast.If)
    assert ast.unparse(grid.body[0].test) == "self.medium_scheduler_type == 'full_grid'"
    assert ast.unparse(grid.body[-1]) == (
        "return SymmetricUpProjectionGemm._compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn, max_active_clusters)"
    )
    assert "cute.ceil_div(c.shape[0], self.mma_tiler_mn[0])" in ast.unparse(grid)
    assert "cute.ceil_div(c.shape[1], self.mma_tiler_mn[1])" in ast.unparse(grid)
    # No replacement ordinary grid, per-thread scheduler or device method.
    assert not grid.decorator_list


@pytest.mark.parametrize("scheduler", ["static_persistent", "full_grid"])
@pytest.mark.parametrize(
    "m,tile_m,tile_n,cluster_m,capacity",
    [
        (256, 64, 64, 1, 152),
        (1024, 64, 64, 1, 152),
        (1024, 64, 128, 1, 152),
        (832, 128, 128, 2, 76),
    ],
)
def test_actual_host_grid_override_forwards_correct_worker_limit(
    scheduler, m, tile_m, tile_n, cluster_m, capacity
):
    from types import SimpleNamespace

    path = (
        ROOT
        / "python/tokenspeed_kernel/thirdparty/cute_dsl/symmetric_up_projection/fused_shared_rs_medium.py"
    )
    tree = ast.parse(path.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_grid"
    )
    # Execute the actual plain host-policy body with only the CuTe shape helper
    # and inherited scheduler factory substituted; no GPU package is imported.
    namespace = {
        "cute": SimpleNamespace(
            ceil_div=lambda value, divisor: (value + divisor - 1) // divisor
        ),
        "SymmetricUpProjectionGemm": SimpleNamespace(_compute_grid=lambda *args: args),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    instance = SimpleNamespace(
        medium_scheduler_type=scheduler, mma_tiler_mn=(tile_m, tile_n)
    )
    c = SimpleNamespace(shape=(m, 896, 1))
    cta = (tile_m // cluster_m, tile_n, 64)
    cluster = (cluster_m, 1)
    forwarded = namespace["_compute_grid"](instance, c, cta, cluster, capacity)
    expected = (
        ((m + tile_m - 1) // tile_m) * (896 // tile_n)
        if scheduler == "full_grid"
        else capacity
    )
    assert forwarded == (c, cta, cluster, expected)


@pytest.fixture
def integrated_profile():
    # Execute the CPU-only selector without importing CUDA-dependent bindings.
    path = OPS / "medium_fused_rs_up_projection_serving_config.py"
    tree = ast.parse(path.read_text())
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "integrated_fused_rs_serving_config"
    )
    namespace = {"MediumFusedRsUpProjectionTuning": Tuning}
    exec(
        compile(ast.Module(body=[selector], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[selector.name]


@pytest.mark.parametrize(
    "m,tile_m,tile_n,cluster_m,cap",
    [
        (256, 64, 64, 1, None),
        (512, 64, 64, 1, None),
        (768, 64, 128, 1, None),
        (1024, 64, 128, 1, None),
        (1280, 256, 128, 2, 38),
        (2048, 256, 128, 2, 38),
        (3072, 256, 128, 2, 38),
        (4096, 256, 128, 2, 38),
        (6144, 256, 128, 2, None),
        (8192, 256, 128, 2, None),
    ],
)
def test_integrated_profile_replaces_large_endpoint_whitelist(
    integrated_profile, m, tile_m, tile_n, cluster_m, cap
):
    config = integrated_profile(m)
    assert (config.tile_m, config.tile_n, config.cluster_m) == (
        tile_m,
        tile_n,
        cluster_m,
    )
    assert config.cluster_cap == cap
    assert config.ab_stages == (6 if m > 1024 else 0)
    assert config.c_stages == (3 if m > 1024 else 0)
    assert config.addend_stages == 1
    assert config.reduce_vectors == 4
    assert config.scheduler_type == "static_persistent"


def test_shared_serving_base_cannot_select_old_endpoint_policy():
    source = (OPS / "cutedsl_fused_rs_up_projection.py").read_text()
    tree = ast.parse(source)
    base = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FusedRsUpProjectionServingBase"
    )
    assert [ast.unparse(parent) for parent in base.bases] == ["ABC"]
    methods = {
        node.name: node for node in base.body if isinstance(node, ast.FunctionDef)
    }
    assert "__init__" not in methods
    for name in ("input_view", "_prepare"):
        assert "abstractmethod" in [
            ast.unparse(decorator) for decorator in methods[name].decorator_list
        ]
    assert "prepare_fused_rs_up_projection" not in source
    assert "fused_rs_up_projection_config" not in source


@pytest.fixture
def runtime_policy():
    from enum import IntEnum
    from types import SimpleNamespace

    path = ROOT.parent / "python/tokenspeed/runtime/models/kimi_k3_comm.py"
    tree = ast.parse(path.read_text())
    names = {
        "K3MoETailTier",
        "_integrated_tail_applicable",
        "select_integrated_k3_moe_tail_tier",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    comm = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "K3MoeTailComm"
    )
    methods = [
        node
        for node in comm.body
        if isinstance(node, ast.FunctionDef) and node.name in {"plan", "run"}
    ]

    def reject_legacy_dispatch(**kwargs):
        raise AssertionError("supported fused M reached legacy dispatch")

    namespace = {
        "IntEnum": IntEnum,
        "torch": SimpleNamespace(Tensor=object),
        "TailPlan": SimpleNamespace,
        "select_k3_moe_tail_tier": reject_legacy_dispatch,
    }
    exec(
        compile(
            ast.Module(body=[*nodes, *methods], type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    args = dict(
        mapping=SimpleNamespace(
            moe=SimpleNamespace(tp_size=8, ep_size=1, tp_ep_size=8),
            attn=SimpleNamespace(dp_size=1, cp_size=1),
        ),
        hidden_size=7168,
        latent_size=3584,
        top_k=16,
        is_blackwell=True,
        fused_moe_ar=True,
        has_routed_norm=True,
        shard_up_projection=True,
        experts_supports_deferred_finalize=True,
    )
    return namespace, args


@pytest.mark.parametrize("old_value", ["0", "1"])
def test_integrated_route_is_automatic_not_an_environment_opt_in(
    runtime_policy, monkeypatch, old_value
):
    for flag in (
        "TOKENSPEED_K3_INTEGRATED_FUSED_TAIL",
        "TOKENSPEED_K3_FUSED_RS_UP_AG",
        "TOKENSPEED_K3_MNNVL_CUTEDSL",
        "TOKENSPEED_K3_MNNVL_BT_ONLY",
        "TOKENSPEED_K3_MEDIUM_FUSED_RS_UP_AG",
    ):
        monkeypatch.setenv(flag, old_value)
    namespace, args = runtime_policy
    assert namespace["_integrated_tail_applicable"](**args)


@pytest.mark.parametrize(
    "key,value",
    [
        ("is_blackwell", False),
        ("hidden_size", 4096),
        ("latent_size", 4096),
        ("top_k", 8),
        ("fused_moe_ar", False),
        ("has_routed_norm", False),
        ("shard_up_projection", False),
        ("experts_supports_deferred_finalize", False),
    ],
)
def test_automatic_route_requires_supported_model_backend(runtime_policy, key, value):
    namespace, args = runtime_policy
    args[key] = value
    assert not namespace["_integrated_tail_applicable"](**args)


@pytest.mark.parametrize(
    "m,tier",
    [
        (256, "MEDIUM_FUSED_RS_UP_AG"),
        (512, "MEDIUM_FUSED_RS_UP_AG"),
        (1024, "MEDIUM_FUSED_RS_UP_AG"),
        (1280, "FUSED_RS_UP_AG"),
        (2048, "FUSED_RS_UP_AG"),
        (4096, "FUSED_RS_UP_AG"),
        (8192, "FUSED_RS_UP_AG"),
        (16384, "SEPARATE_REDUCE"),
    ],
)
def test_automatic_plan_bypasses_legacy_tail(runtime_policy, m, tier):
    from types import SimpleNamespace

    namespace, _ = runtime_policy
    raw = object()
    finalize = SimpleNamespace(supports_num_tokens=lambda tokens: True)
    owner = SimpleNamespace(
        state=SimpleNamespace(
            integrated_tail=True,
            mnnvl_bt_deferred=finalize,
            mnnvl_ht_deferred=finalize,
        ),
        fused_rs_up_ag=SimpleNamespace(input_view=lambda tokens: raw),
        _experts_supports_deferred_finalize=True,
    )
    result = namespace["plan"](owner, m)
    assert result.tier is getattr(namespace["K3MoETailTier"], tier)
    if m <= 8192:
        assert result.defer_finalize
        assert result.symm_outputs == (None, raw)
        owner.fused_rs_up_ag = None
        with pytest.raises(RuntimeError, match="missing"):
            namespace["plan"](owner, m)
    else:
        assert result.routed_in_fork


def test_only_small_integrated_and_separate_tiers_remain(runtime_policy):
    namespace, _ = runtime_policy
    assert set(namespace["K3MoETailTier"].__members__) == {
        "TAIL_FUSION",
        "MEDIUM_FUSED_RS_UP_AG",
        "FUSED_RS_UP_AG",
        "SEPARATE_REDUCE",
    }


def test_integrated_adapter_has_no_intermediate_serving_profile():
    tree = ast.parse((OPS / "medium_fused_rs_up_projection_serving.py").read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert set(classes) == {
        "IntegratedFusedRsUpProjectionServing",
        "IntegratedFusedRsOutputPool",
    }
    assert [
        ast.unparse(parent)
        for parent in classes["IntegratedFusedRsUpProjectionServing"].bases
    ] == ["FusedRsUpProjectionServingBase"]
    config = ast.parse(
        (OPS / "medium_fused_rs_up_projection_serving_config.py").read_text()
    )
    assert {node.name for node in config.body if isinstance(node, ast.FunctionDef)} == {
        "integrated_fused_rs_serving_config"
    }


@pytest.mark.parametrize(
    "m,tier,protocol",
    [
        (256, "MEDIUM_FUSED_RS_UP_AG", "bt"),
        (1024, "MEDIUM_FUSED_RS_UP_AG", "bt"),
        (2048, "FUSED_RS_UP_AG", "ht"),
        (8192, "FUSED_RS_UP_AG", "ht"),
    ],
)
def test_integrated_run_keeps_bt_ht_and_one_fused_back_half(
    runtime_policy, m, tier, protocol
):
    from types import SimpleNamespace

    namespace, _ = runtime_policy
    calls = []
    normalized, weight, norm_weight, residual, shared, output = (
        object() for _ in range(6)
    )
    deferred = (object(), object(), object())

    def finalize(name, *args):
        calls.append((name, args))
        return normalized

    def fused(*args):
        calls.append(("fused", args))
        return output

    def residual_view(rows, columns):
        assert (rows, columns) == (m, 7168)
        return residual

    owner = SimpleNamespace(
        state=SimpleNamespace(
            mnnvl_bt_deferred=lambda *args: finalize("bt", *args),
            mnnvl_ht_deferred=lambda *args: finalize("ht", *args),
        ),
        routed_norm=SimpleNamespace(weight=norm_weight),
        up_proj=SimpleNamespace(weight=weight),
        fused_rs_up_ag=fused,
    )
    result = namespace["run"](
        owner,
        SimpleNamespace(tier=getattr(namespace["K3MoETailTier"], tier)),
        deferred,
        shared,
        SimpleNamespace(view=residual_view),
        m,
        7168,
        prepared_shared_shard=None,
    )
    assert result is output
    assert calls == [
        (protocol, (*deferred, norm_weight)),
        ("fused", (normalized, weight, residual, shared)),
    ]
