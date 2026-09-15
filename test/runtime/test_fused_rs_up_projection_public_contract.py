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

"""CPU release checks only: source/configuration/evidence, not GPU emulation."""

import ast
import hashlib
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMM = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/communication"
BENCH = ROOT / "tokenspeed-kernel/test/nvidia/thirdparty"
DATA = ROOT / "docs/public/data/fused-rs-up-ag"


def definitions(path, names, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def config(m):
    return runpy.run_path(COMM / "fused_rs_up_projection_config.py")[
        "fused_rs_up_projection_config"
    ](m)


@pytest.mark.parametrize("m,cap", [(4096, 38), (8192, None)])
def test_endpoint_configuration_equals_historical_identity(m, cap):
    row = json.loads((DATA / f"primary-m{m}-batch1.json").read_text())
    policy = config(m)
    assert policy["tuning"] == row["fused_up"]
    assert policy["cluster_cap"] == cap
    assert policy["ht_stages"] == row["candidate_ht"]["stages"] == 10
    assert policy["raw_publication_warps"] == policy["output_exit_warps"] == 4
    assert policy["residual_issue_order"] == "after_reduce"
    assert policy["staging_order"] == "before_ht"


@pytest.mark.parametrize(
    "m", [True, 0, 256, 1024, 4095, 4097, 8191, 8193, 4096.0, "4096"]
)
def test_unqualified_endpoint_rejected(m):
    with pytest.raises(ValueError):
        config(m)


def test_config_is_fresh_and_typed_tuning_validates():
    policy = config(4096)
    policy["tuning"]["tile_n"] = 64
    assert config(4096)["tuning"]["tile_n"] == 128
    namespace = {"dataclass": dataclass}
    definitions(
        COMM / "mnnvl_cutedsl_symmetric_up_projection.py",
        {"SymmetricUpProjectionTuning", "SymmetricUpProjectionOverlapTuning"},
        namespace,
    )
    definitions(
        COMM / "mnnvl_cutedsl_fused_rs_up_projection.py",
        {"FusedRsUpProjectionTuning", "validate_fused_tuning"},
        namespace,
    )
    for m in (4096, 8192):
        tuning = namespace["FusedRsUpProjectionTuning"](**config(m)["tuning"])
        namespace["validate_fused_tuning"](tuning)
    with pytest.raises(ValueError):
        namespace["FusedRsUpProjectionTuning"](
            **{**config(4096)["tuning"], "reduce_vectors": True}
        ).validate()


def test_ht_change_is_explicit_not_mislabelled_as_isolated_rs():
    factory = runpy.run_path(COMM / "fused_rs_up_projection_config.py")[
        "fused_rs_tail_ht_config"
    ]
    before, after = factory(7), factory(10)
    assert {k for k in before if before[k] != after[k]} == {"stages"}
    with pytest.raises(ValueError):
        factory(8)


def test_facade_rejects_wrong_stream_before_any_plan_work():
    events = []
    stream = SimpleNamespace(cuda_stream=12)
    namespace = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda device: stream)
        )
    }
    cls = definitions(
        COMM / "fused_rs_up_projection.py", {"FusedRsUpProjection"}, namespace
    )["FusedRsUpProjection"]

    class Plan:
        output = SimpleNamespace(tensor=SimpleNamespace(device="test-only"))
        input_view = "stable-input"

        def stage(self, partial):
            events.append(("stage", partial))
            return self.input_view

        def __call__(self):
            events.append(("run",))
            return "stable-output"

    bound = cls(Plan(), config(4096), 12)
    assert bound.stage("partial") == "stable-input"
    assert bound() == "stable-output"
    stream.cuda_stream = 13
    with pytest.raises(ValueError, match="prepared stream"):
        bound.stage("other")
    with pytest.raises(ValueError, match="prepared stream"):
        bound()
    assert events == [("stage", "partial"), ("run",)]


def test_complete_fused_plan_keeps_both_barriers_and_blocks_standalone_producer():
    tree = ast.parse((COMM / "mnnvl_cutedsl_fused_rs_up_projection.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "BoundFusedRsUpProjection"
    )
    call = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    code = ast.unparse(call)
    assert (
        code.index("_rank_barrier_kernel")
        < code.index("producer_only")
        < code.index("self._barrier")
    )
    standalone = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "producer_only"
    )
    assert any(isinstance(n, ast.Raise) for n in ast.walk(standalone))
    assert "num_warps=4" in code


def test_benchmark_admits_before_first_producer_and_times_real_producer():
    text = (BENCH / "bench_fused_rs_up_projection_ag.py").read_text()
    assert text.index("admission = ") < text.index("eager = ")
    assert "kimi3_shared_down_projection(" in text
    assert "out=destination" in text
    assert "20 * 4" in text
    assert "dist.ReduceOp.MAX" in text
    assert "range(128)" in text
    assert "0.046875 + 0.01" in text and "relative_l2 > 0.006" in text
    runtime = (ROOT / "python/tokenspeed/runtime/models/kimi_k3_comm.py").read_text()
    assert "prepare_fused_rs_up_projection" not in runtime


def test_resource_admission_fails_closed_without_compiled_metadata():
    check = runpy.run_path(BENCH / "fused_rs_up_ag_admission.py")["cap_smem_precheck"]
    for cap in (38, 76):
        with pytest.raises(ValueError, match="actual compiled"):
            check({"static": {}, "driver": {}}, "", cap)
    with pytest.raises(ValueError, match="explicit integer"):
        check({}, "", 37)


@pytest.mark.parametrize("scope", ["primary", "producer"])
@pytest.mark.parametrize("m", [4096, 8192])
@pytest.mark.parametrize("batch", [1, 2])
def test_all_paired_statistics_recompute_exactly(scope, m, batch):
    record = json.loads((DATA / f"{scope}-m{m}-batch{batch}.json").read_text())
    calculate = runpy.run_path(BENCH / "fused_rs_up_ag_statistics.py")[
        "paired_statistics"
    ]
    assert record["historical"] and not record["fresh_pr_gpu_rerun"]
    assert not record["instrumented"] and not record["runtime_enabled"]
    assert (
        record["rounds"],
        record["graph_layers"],
        record["warmup"],
        record["iters"],
    ) == (31, 4, 8, 20)
    for name, old in record["comparisons"].items():
        assert len(old["baseline_raw_us"]) == len(old["candidate_raw_us"]) == 31
        recomputed = calculate(old["baseline_raw_us"], old["candidate_raw_us"])
        for key, value in recomputed.items():
            assert old[key] == pytest.approx(value, rel=1e-12)
        assert old["p90_speedup"] >= 1
        assert old["paired_ci95"][0] > 1
        if name in ("frozen_to_selected", "frozen_to_direct"):
            assert old["p50_speedup"] >= 1 / 0.85
        if name == "copied_to_direct":
            assert old["p50_speedup"] >= 1.01


def flag_records(value):
    if isinstance(value, dict):
        if "flag_runs" in value:
            yield value
        else:
            for child in value.values():
                yield from flag_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from flag_records(child)


def test_lossless_continuous_flag_exports_have_expected_counts():
    counts = {"formal": 0, "continuous": 0, "numerical": 0}
    for path in DATA.glob("*.json"):
        group = (
            "formal"
            if path.name.startswith(("primary-", "producer-"))
            else (
                "continuous"
                if path.name.startswith("continuous-")
                else "numerical" if path.name.startswith("numerical-") else None
            )
        )
        if group is None:
            continue
        for record in flag_records(json.loads(path.read_text())):
            assert record["generations"] == 128
            assert (
                sum(length for bit, length in record["flag_runs"])
                == record["num_checks"]
            )
            assert (
                sum(length for bit, length in record["flag_runs"] if bit == 0)
                == record["num_failures"]
                == 0
            )
            assert all(
                bit in (0, 1) and length > 0 for bit, length in record["flag_runs"]
            )
            counts[group] += record["num_checks"]
    assert counts == {"formal": 196608, "continuous": 24576, "numerical": 40960}


@pytest.mark.parametrize("policy", ["cap38", "hardware-capacity"])
def test_numerical_certificate_does_not_relax_original_bounds(policy):
    record = json.loads((DATA / f"numerical-{policy}.json").read_text())
    assert record["all_boundary_M_covered"]
    assert len(record["rows"]) == 10
    checks = 0
    for row in record["rows"]:
        assert row["numerical_failures"] == 0
        assert row["reference_bounds"] == {
            "absolute": 0.046875,
            "relative_to_reference_max": 0.01,
            "relative_l2": 0.006,
        }
        checks += sum(row["numerical_check_counts"].values())
    assert checks == 9600


def test_frozen_source_provenance_survives_formatting():
    manifest = json.loads((DATA / "source-provenance.json").read_text())
    assert not manifest["fresh_pr_gpu_rerun"]
    for path, record in manifest["files"].items():
        actual = (ROOT / path).read_text()
        tree = ast.parse(actual)
        imports = sorted(
            [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))],
            key=ast.dump,
        )
        tree.body = [
            n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ] + imports
        canonical = ast.dump(tree, include_attributes=False)
        assert hashlib.sha256(canonical.encode()).hexdigest() == record.get(
            "ported_python_ast_sha256", record["python_ast_sha256"]
        )
        if "unchanged_device_functions" in record:
            device_nodes = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name in record["unchanged_device_functions"]
            ]
            device_ast = ast.dump(ast.Module(body=device_nodes, type_ignores=[]))
            assert (
                hashlib.sha256(device_ast.encode()).hexdigest()
                == record["device_ast_sha256"]
            )
        assert "Permission is hereby granted" in actual or "License" in actual


def test_public_addition_has_no_internal_iteration_names():
    paths = list(DATA.glob("*.json"))
    paths += [
        ROOT / "docs/design/kimi-k3-fused-rs-up-projection-ag.md",
        ROOT / "docs/guides/kimi-k3-fused-rs-up-projection-ag.md",
    ]
    for path in paths:
        text = path.read_text()
        assert "v" + "30" not in text
        assert "tail" + "15" not in text
        assert "/home/scratch." not in text
