from __future__ import annotations

import ast
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "test" / "ci_system" / "deepseek_v4_flat_differential.py"
MANIFEST_PATH = ROOT / "test" / "ci" / "deepseek-v4-flat-differential-manifest.json"
TASK_PATHS = (
    ROOT / "test" / "ci" / "ut" / "ut-deepseek-v4-flat-differential-flash.yaml",
    ROOT / "test" / "ci" / "ut" / "ut-deepseek-v4-flat-differential-pro-tp8.yaml",
    ROOT / "test" / "ci" / "ut" / "ut-deepseek-v4-flat-differential-pro-tp8-ep.yaml",
)
EVENT_LOOP_PATH = (
    ROOT / "python" / "tokenspeed" / "runtime" / "engine" / "event_loop.py"
)
REQUEST_HANDLER_PATH = (
    ROOT / "python" / "tokenspeed" / "runtime" / "engine" / "request_handler.py"
)
SYNTHETIC_TEST_PATH = ROOT / "test" / "runtime" / "test_deepseek_v4_flat_synthetic.py"
SYNTHETIC_TASK_PATH = (
    ROOT / "test" / "ci" / "ut" / "ut-deepseek-v4-flat-synthetic-b200.yaml"
)
INSTALL_VARIANTS_PATH = ROOT / "test" / "ci_system" / "install_scheduler_variants.sh"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_v4_flat_differential", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_checked_in_manifest_covers_required_topologies_and_matrices():
    manifest = runner.load_manifest(MANIFEST_PATH)

    assert set(manifest["topologies"]) == runner.REQUIRED_TOPOLOGIES
    synthetic = manifest["synthetic_1gpu"]
    assert synthetic["runner"] == "b200-1gpu"
    assert synthetic["required_chain"] == [
        "cpp_flat_scheduler",
        "python_flat_forward_bridge",
        "v4_backend_metadata",
        "v4_cache_writers_readers",
    ]
    assert set(synthetic["physical_data"]) == {
        "swa_kv",
        "compressed_kv",
        "compressor_state",
        "indexer_kv",
        "indexer_compressor_state",
    }
    assert synthetic["status"] == "implemented"
    assert synthetic["acceptance_injection_seam"] == "scheduler_structured_completion"
    assert synthetic["gpu_test"] == "test/runtime/test_deepseek_v4_flat_synthetic.py"
    assert synthetic["ci_task"] == "test/ci/ut/ut-deepseek-v4-flat-synthetic-b200.yaml"
    scenarios = [
        scenario
        for entries in manifest["scenario_sets"].values()
        for scenario in entries
    ]
    assert {scenario["execution_mode"] for scenario in scenarios} == {
        "eager",
        "graph",
    }
    assert {scenario["indexer_layout"] for scenario in scenarios} == {
        "fp8",
        "mxfp4",
    }
    assert (
        manifest["topologies"]["flash-dp4-ep"]["engine_kwargs"]["load_balance_method"]
        == "round_robin"
    )
    graph_prefill = next(
        workload
        for workload in manifest["workloads"]
        if workload["kind"] == "graph_prefill"
    )
    assert graph_prefill["requires_all_dp_extend"] is True
    assert graph_prefill["batched_dispatch"] is True
    assert len(graph_prefill["prompts"]) >= 8
    spec_matrix = {
        (
            scenario["prefix_caching"],
            scenario["overlap_depth"],
            scenario["mtp_steps"],
        )
        for scenario in scenarios
        if scenario["mtp_steps"]
    }
    assert spec_matrix == {
        (prefix, overlap, steps)
        for prefix in (False, True)
        for overlap in (0, 1)
        for steps in (1, 3)
    }


def test_variant_installer_rejects_unsafe_or_unowned_delete_targets(tmp_path):
    env = os.environ.copy()
    env["WORKSPACE"] = str(ROOT)
    env["TOKENSPEED_SCHEDULER_VARIANT_ROOT"] = str(ROOT)
    workspace_result = subprocess.run(
        ["bash", str(INSTALL_VARIANTS_PATH)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert workspace_result.returncode == 2
    assert "Refusing unsafe scheduler variant root" in workspace_result.stderr

    unowned = tmp_path / "scheduler-variants"
    unowned.mkdir()
    sentinel = unowned / "keep-me"
    sentinel.write_text("owned by the test")
    env["TOKENSPEED_SCHEDULER_VARIANT_ROOT"] = str(unowned)
    unowned_result = subprocess.run(
        ["bash", str(INSTALL_VARIANTS_PATH)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert unowned_result.returncode == 2
    assert "without marker" in unowned_result.stderr
    assert sentinel.read_text() == "owned by the test"


def test_scenario_materialization_is_explicit_and_deterministic():
    manifest = runner.load_manifest(MANIFEST_PATH)
    scenario = next(
        scenario
        for scenario in manifest["scenario_sets"]["mtp"]
        if scenario["prefix_caching"]
        and scenario["overlap_depth"] == 1
        and scenario["mtp_steps"] == 3
    )

    kwargs = runner.scenario_engine_kwargs(manifest, "flash-dp4-ep", scenario)

    assert kwargs["seed"] == manifest["seed"]
    assert kwargs["speculative_algorithm"] == "MTP"
    assert kwargs["speculative_num_steps"] == 3
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["disable_overlap_schedule"] is False
    assert kwargs["enforce_eager"] is (scenario["execution_mode"] == "eager")
    assert kwargs["attention_use_fp4_indexer_cache"] is (
        scenario["indexer_layout"] == "mxfp4"
    )


def test_graph_prefill_probe_uses_one_batch_before_waiting_for_outputs():
    manifest = runner.load_manifest(MANIFEST_PATH)
    workload = next(
        workload
        for workload in manifest["workloads"]
        if workload["kind"] == "graph_prefill"
    )

    class FakeEngine:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [
                {"output_ids": [index], "meta_info": {"cached_tokens": 0}}
                for index, _ in enumerate(kwargs["prompt"])
            ]

    engine = FakeEngine()
    token_ids, acceptances, cached = runner._run_workload(engine, workload)

    assert len(engine.calls) == 1
    assert engine.calls[0]["prompt"] == [
        prompt * repeat
        for prompt, repeat in zip(workload["prompts"], workload["prompt_repeat_counts"])
    ]
    assert len(engine.calls[0]["sampling_params"]) == len(workload["prompts"])
    assert token_ids == [[index] for index in range(len(workload["prompts"]))]
    assert acceptances == []
    assert cached == [0] * len(workload["prompts"])


def _valid_results():
    radix = {
        "variant": "radix",
        "topology": "t",
        "scenario": "s",
        "token_ids": {"cold": [[1, 2, 3]]},
        "cached_tokens": {"cold": [0]},
        "acceptances": {},
        "graph_proof": {"execution_mode": "eager", "validated": True},
    }
    flat = {
        "variant": "flat",
        "topology": "t",
        "scenario": "s",
        "token_ids": {"cold": [[1, 2, 3]]},
        "cached_tokens": {"cold": [0]},
        "acceptances": {},
        "graph_proof": {"execution_mode": "eager", "validated": True},
        "flat_proof": {
            "fingerprint": "a" * 64,
            "generation_delta": 1,
            "all_pools_reset": True,
            "post_wake_token_exact": True,
        },
    }
    return radix, flat


def test_exact_output_id_mismatch_fails_closed():
    radix, flat = _valid_results()
    flat["token_ids"]["cold"] = [[1, 9, 3]]

    with pytest.raises(runner.DifferentialGateError, match="output_ids mismatch"):
        runner.compare_variant_results(radix, flat)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("topology", "other", "topology mismatch"),
        ("cached_tokens", {"cold": [4]}, "cached_tokens mismatch"),
        ("acceptances", {"cold": [0.5]}, "acceptance mismatch"),
    ),
)
def test_runtime_parity_metadata_mismatch_fails_closed(field, value, message):
    radix, flat = _valid_results()
    flat[field] = value

    with pytest.raises(runner.DifferentialGateError, match=message):
        runner.compare_variant_results(radix, flat)


def test_missing_or_mismatched_graph_proof_fails_closed():
    radix, flat = _valid_results()
    flat["graph_proof"] = None
    with pytest.raises(runner.DifferentialGateError, match="graph evidence"):
        runner.compare_variant_results(radix, flat)

    radix, flat = _valid_results()
    flat["graph_proof"]["execution_mode"] = "graph"
    with pytest.raises(runner.DifferentialGateError, match="evidence mode mismatch"):
        runner.compare_variant_results(radix, flat)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("fingerprint", None, "fingerprint missing"),
        ("generation_delta", 0, "exactly one generation"),
        ("all_pools_reset", False, "baseline"),
        ("post_wake_token_exact", False, "wake changed"),
    ),
)
def test_missing_flat_runtime_proof_fails_closed(field, value, message):
    radix, flat = _valid_results()
    flat["flat_proof"][field] = value

    with pytest.raises(runner.DifferentialGateError, match=message):
        runner.compare_variant_results(radix, flat)


def test_existing_ci_task_scanner_can_consume_all_differential_targets():
    """Task declarations are JSON documents with a .yaml suffix.

    JSON is a strict YAML subset, so the current pipeline's ``yaml.safe_load``
    consumes these files directly while this CPU-only test can validate the
    wiring without adding PyYAML to a developer venv.
    """

    manifest = runner.load_manifest(MANIFEST_PATH)
    seen_topologies = set()
    for path in TASK_PATHS:
        task = json.loads(path.read_text())
        assert task["api_version"] == "ci.tokenspeed.io/v1"
        assert task["type"] == "ut"
        assert set(task["triggers"]) == {"nightly", "manual"}
        assert "bash test/ci_system/install_scheduler_variants.sh" in task["install"]
        commands = task["ut"]["commands"]
        assert len(commands) == 1
        command = commands[0]
        assert "deepseek_v4_flat_differential.py" in command
        assert str(MANIFEST_PATH.relative_to(ROOT)) in command
        argv = shlex.split(command)
        topology = argv[argv.index("--topology") + 1]
        assert topology in manifest["topologies"]
        seen_topologies.add(topology)
        assert task["runner"]["labels"] == [manifest["topologies"][topology]["runner"]]

    assert seen_topologies == runner.REQUIRED_TOPOLOGIES


def test_synthetic_b200_task_and_source_are_real_chain_not_a_manifest_stub():
    manifest = runner.load_manifest(MANIFEST_PATH)
    contract = manifest["synthetic_1gpu"]
    task = json.loads(SYNTHETIC_TASK_PATH.read_text())
    source = SYNTHETIC_TEST_PATH.read_text()
    tree = ast.parse(source, filename=str(SYNTHETIC_TEST_PATH))

    assert contract["status"] == "implemented"
    assert task["api_version"] == "ci.tokenspeed.io/v1"
    assert task["runner"]["labels"] == [contract["runner"]]
    assert set(task["triggers"]) == {"nightly", "manual"}
    assert "bash test/ci_system/install_scheduler_variants.sh" in task["install"]
    command = task["ut"]["commands"]
    assert len(command) == 1
    assert "scheduler-variants/flat" in command[0]
    assert "--suite deepseek-v4-flat-synthetic" in command[0]

    registrations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_cuda_ci"
    ]
    assert len(registrations) == 1
    registration_kwargs = {
        keyword.arg: keyword.value for keyword in registrations[0].keywords
    }
    assert ast.literal_eval(registration_kwargs["suite"]) == (
        "deepseek-v4-flat-synthetic"
    )

    required_real_calls = {
        "FlatBlockTableStagingBuffers",
        "DeepseekV4AttentionBackend",
        "_group_slot_mapping_from_raw",
        "fused_qnorm_rope_kv_insert",
        "save_deepseek_v4_compressor_state",
        "deepseek_v4_csa_compress_kv_cache_insert",
        "deepseek_v4_hca_compress_kv_cache_insert",
        "deepseek_v4_csa_indexer_cache_insert",
        "dequantize_deepseek_v4_fp8_ds_mla_cache",
        "read_deepseek_v4_indexer_fp8_cache",
    }
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert required_real_calls <= call_names
    mtp_test = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "test_mtp_acceptance_matrix_at_structured_completion_seam"
    )
    parametrizations = {}
    for decorator in mtp_test.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
        ):
            argnames = ast.literal_eval(decorator.args[0])
            parametrizations[argnames] = ast.literal_eval(decorator.args[1])
    assert parametrizations["overlap_depth"] == [0, 1]
    assert parametrizations[("acceptance", "accepted_length")] == [
        ("zero", 0),
        ("partial", 2),
        ("full", 4),
    ]

    confined_targets = {
        ast.literal_eval(node.args[1])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_confined_write"
        and len(node.args) >= 2
    }
    assert confined_targets == {
        "swa_kv",
        "compressed_kv_c4",
        "compressed_kv_c128",
        "compressor_state_c4",
        "compressor_state_c128",
        "indexer_kv",
        "indexer_compressor_state",
    }
    assert "ts.Scheduler" in source
    assert "ts.FLAT_KVCACHE is True" in source
    assert "torch.cuda.get_device_capability(0)" in source
    assert "ts.ForwardEvent.FlatKVCompletion" in source
    assert "flat_block_table_group_ids" in source
    assert 'table_source_kind == "flat"' in source
    assert "torch.count_nonzero(planes[target][0])" in source
    assert "pytest.skip" not in source


def test_internal_state_gate_uses_existing_cold_control_seam():
    event_loop = EVENT_LOOP_PATH.read_text()
    request_handler = REQUEST_HANDLER_PATH.read_text()

    assert "get_internal_state_fn=self._get_internal_state" in event_loop
    assert '"flat_plan_fingerprint"' in event_loop
    assert '"flat_pool_snapshots"' in event_loop
    assert '"flat_kv_generation"' in event_loop
    assert '"cuda_graph_evidence"' in event_loop
    assert "self.get_internal_state_fn()" in request_handler
    assert "GetInternalStateReqOutput(internal_state=internal_state)" in request_handler


def _graph_state(
    *,
    decode=False,
    idle=False,
    prefill=False,
    captured=True,
    drafter=False,
):
    return {
        "cuda_graph_evidence": {
            "decode": {
                "captured_keys": (
                    [{"variant": "default", "batch_size": 1}] if captured else []
                ),
                "decode_replayed": decode,
                "idle_replayed": idle,
            },
            "prefill": {
                "captured_buckets": [1024] if captured else [],
                "replayed_kinds": ["text"] if prefill else [],
            },
            "drafter_present": drafter,
        }
    }


def test_graph_evidence_requires_capture_decode_prefill_and_dp_idle():
    scenario = {"execution_mode": "graph", "mtp_steps": 0}
    topology = {"engine_kwargs": {"data_parallel_size": 4}}
    states = [
        _graph_state(decode=True, prefill=True),
        _graph_state(idle=True, prefill=True),
        _graph_state(idle=True, prefill=True),
        _graph_state(idle=True, prefill=True),
    ]

    proof = runner._assert_graph_evidence(states, scenario=scenario, topology=topology)
    assert proof == {
        "execution_mode": "graph",
        "validated": True,
        "decode_replayed": True,
        "idle_replayed": True,
        "prefill_replayed": True,
    }

    no_idle = [_graph_state(decode=True, prefill=True) for _ in range(4)]
    with pytest.raises(runner.DifferentialGateError, match="idle rank"):
        runner._assert_graph_evidence(no_idle, scenario=scenario, topology=topology)

    no_prefill = [_graph_state(decode=True)]
    with pytest.raises(runner.DifferentialGateError, match="breakable prefill"):
        runner._assert_graph_evidence(
            no_prefill,
            scenario=scenario,
            topology={"engine_kwargs": {}},
        )


def test_graph_evidence_checks_mtp_drafter_and_eager_confinement():
    mtp = {"execution_mode": "graph", "mtp_steps": 3}
    with pytest.raises(runner.DifferentialGateError, match="drafter mode"):
        runner._assert_graph_evidence(
            [_graph_state(decode=True, prefill=True, drafter=False)],
            scenario=mtp,
            topology={"engine_kwargs": {}},
        )

    eager = {"execution_mode": "eager", "mtp_steps": 0}
    assert runner._assert_graph_evidence(
        [_graph_state(captured=False)],
        scenario=eager,
        topology={"engine_kwargs": {}},
    )["validated"]
    with pytest.raises(runner.DifferentialGateError, match="unexpectedly used"):
        runner._assert_graph_evidence(
            [_graph_state(captured=True)],
            scenario=eager,
            topology={"engine_kwargs": {}},
        )


def _pool(pool_id: str, total_blocks: int, bytes_per_block: int):
    return {
        "pool_id": pool_id,
        "total_blocks": total_blocks,
        "bytes_per_block": bytes_per_block,
    }


def _snapshot(pool_id: str, total_blocks: int, bytes_per_block: int):
    return {
        "pool_id": pool_id,
        "total_blocks": total_blocks,
        "usable_blocks": total_blocks - 1,
        "free_blocks": total_blocks - 1,
        "active_blocks": 0,
        "cached_evictable_blocks": 0,
        "pinned_cached_blocks": 0,
        "reserved_blocks": 0,
        "bytes_per_block": bytes_per_block,
    }


def _reset_state(snapshots):
    return {
        "flat_kv_generation": 7,
        "flat_arena_generation": 7,
        "flat_kv_quiescent": True,
        "flat_pool_snapshots": snapshots,
    }


def test_pool_reset_requires_exact_unique_manifest_pool_set():
    pools = [_pool("history", 8, 128), _pool("state", 5, 32)]
    valid = [_snapshot("history", 8, 128), _snapshot("state", 5, 32)]

    assert runner._assert_pools_reset([_reset_state(valid)], pools) == 7

    with pytest.raises(runner.DifferentialGateError, match="snapshot set differs"):
        runner._assert_pools_reset([_reset_state(valid[:1])], pools)

    extra = [*valid, _snapshot("unexpected", 3, 16)]
    with pytest.raises(runner.DifferentialGateError, match="snapshot set differs"):
        runner._assert_pools_reset([_reset_state(extra)], pools)

    duplicate = [valid[0], dict(valid[0])]
    with pytest.raises(runner.DifferentialGateError, match="duplicate pool snapshots"):
        runner._assert_pools_reset([_reset_state(duplicate)], pools)

    duplicate_plan = [pools[0], dict(pools[0])]
    with pytest.raises(runner.DifferentialGateError, match="duplicate pool IDs"):
        runner._assert_pools_reset([_reset_state(valid)], duplicate_plan)
