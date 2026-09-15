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

"""CPU-only configuration and boundary guards for the medium-M experiment."""

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
