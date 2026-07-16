import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).with_name("minimax_m3_encoder_graph_ab.py")
REPO_ROOT = Path(__file__).parents[3]
MODULE_NAME = "minimax_m3_encoder_graph_ab"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = benchmark
SPEC.loader.exec_module(benchmark)


def _timing_line(rank: int, encode_ms: float, *, tokens: int = 77) -> str:
    return (
        "\x1b[38;20m[2026-07-15 12:00:00,000  "
        f"ATTN TP RANK {rank}] - INFO - mm_timing encoder_ms "
        f"modality=image items=1 encoder_output_tokens={tokens} "
        f"move_h2d=0.{rank + 1}00 encode={encode_ms:.3f} "
        "per_item_tokens=[77]\x1b[0m\n"
    )


def _graph_startup_log() -> str:
    budgets = ", ".join(str(value) for value in benchmark.EXPECTED_GRAPH_BUDGETS)
    lines = []
    for rank in benchmark.EXPECTED_RANKS:
        lines.append(
            f"[ATTN TP RANK {rank}] EncoderCudaGraphWrapper initialized: "
            f"modality=image, budgets=[{budgets}], max_batch_size=2, "
            "max_metadata_sequences_per_batch=encoder_output_token_budget, "
            "encoder_tp=4\n"
        )
        for budget in benchmark.EXPECTED_GRAPH_BUDGETS:
            lines.append(
                f"[ATTN TP RANK {rank}] Captured encoder cudagraph: "
                f"modality=image, budget={budget}\n"
            )
        lines.append(
            f"[ATTN TP RANK {rank}] Encoder CUDA graph capture complete: "
            "modality=image, 9 budget graphs.\n"
        )
        lines.append(
            f"[ATTN TP RANK {rank}] Installed encoder CUDA graphs for "
            "MiniMaxM3ForCausalLM: "
            "['image_encoder']\n"
        )
    return "".join(lines)


def _response(logprob: float = -0.001) -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "message": {"content": "Dog"},
                "logprobs": {
                    "content": [
                        {"token": "Dog", "logprob": logprob, "top_logprobs": []}
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 254, "completion_tokens": 1},
    }


def _artifact(arm: str, encode_ms: float, e2e_ms: float) -> dict:
    args = {
        "model": benchmark.EXPECTED_MODEL,
        "revision": benchmark.EXPECTED_REVISION,
        "attn_tp_size": 4.0,
        "enable_mm_encoder_cuda_graph": arm == "graph",
        "enable_log_mm_timing": True,
        "enforce_eager": True,
        "disable_prefill_graph": True,
        "enable_prefix_caching": False,
        "enable_output_logprobs": True,
        "sampling_backend": "greedy",
        "language_model_only": False,
        "port": 8000 if arm == "eager" else 9000,
        "rl_control_port": 12345 if arm == "eager" else 23456,
    }
    timing = {
        "critical_path_encode_ms": encode_ms,
        "rank_spread_ms": 0.2,
        "per_rank": {
            str(rank): {"rank": rank, "encode_ms": encode_ms - 0.1 * rank}
            for rank in benchmark.EXPECTED_RANKS
        },
    }
    measured = [
        {
            "index": index,
            "e2e_ms": e2e_ms,
            "timing": timing,
            "response": {"logprob": -0.001},
        }
        for index in range(benchmark.EXPECTED_MEASURE_REQUESTS)
    ]
    return {
        "schema_version": benchmark.SCHEMA_VERSION,
        "benchmark": benchmark.BENCHMARK_NAME,
        "status": "complete",
        "ok": True,
        "arm": arm,
        "launch_id": "launch-1",
        "contract": {
            "concurrency": 1,
            "warmup_requests": benchmark.EXPECTED_WARMUP_REQUESTS,
            "measure_requests": benchmark.EXPECTED_MEASURE_REQUESTS,
        },
        "fixture": {"sha256": benchmark.EXPECTED_FIXTURE_SHA256},
        "reference": {"sha256": "reference-sha"},
        "provenance": {
            key: {"sha256": "a" * 64} for key in benchmark.REQUIRED_PROVENANCE_KEYS
        },
        "server": {"sha": "server-sha", "args": args},
        "warmup": [{}] * benchmark.EXPECTED_WARMUP_REQUESTS,
        "measured": measured,
    }


def _write_artifact(path: Path, artifact: dict) -> Path:
    path.write_text(json.dumps(artifact))
    return path


def test_parses_four_rank_timings_and_uses_slowest_rank() -> None:
    text = "noise\n" + "".join(_timing_line(rank, 8.0 + rank) for rank in (2, 0, 3, 1))

    result = benchmark.validate_encoder_timings(benchmark.find_encoder_timings(text))

    assert list(result["per_rank"]) == ["0", "1", "2", "3"]
    assert result["critical_path_encode_ms"] == 11.0
    assert result["rank_spread_ms"] == 3.0
    assert result["rank_spread_fraction_of_max"] == pytest.approx(3.0 / 11.0)


@pytest.mark.parametrize(
    "text",
    [
        "".join(_timing_line(rank, 8.0) for rank in (0, 1, 2)),
        "".join(_timing_line(rank, 8.0) for rank in (0, 1, 1, 3)),
        "".join(
            _timing_line(rank, 8.0, tokens=76 if rank == 2 else 77)
            for rank in benchmark.EXPECTED_RANKS
        ),
    ],
)
def test_rank_timing_contract_fails_closed(text: str) -> None:
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.validate_encoder_timings(benchmark.find_encoder_timings(text))


def test_compile_detection_ignores_normal_triton_runtime_line() -> None:
    clean = "Custom Triton RSAG symmetric-memory buffer allocated: 0.05 GB\n"
    dirty = clean + "Triton autotuning and compiling kernel for shape=(308, 1176)\n"

    assert benchmark.find_compile_events(clean) == []
    assert benchmark.find_compile_events(dirty)[0]["kind"] == (
        "triton_compile_or_autotune"
    )


def test_request_log_window_requires_matching_finish_marker(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "".join(_timing_line(rank, 8.0 + rank) for rank in benchmark.EXPECTED_RANKS)
    )

    with pytest.raises(benchmark.BenchmarkError, match="finish=False"):
        benchmark.wait_for_request_log_window(
            server_log, 0, 0.01, request_id="chatcmpl-test"
        )

    with server_log.open("a") as handle:
        handle.write("Req: chatcmpl-test Finish! Accept_num_tokens_avg: 0\n")
    _, _, rows = benchmark.wait_for_request_log_window(
        server_log, 0, 0.1, request_id="chatcmpl-test"
    )
    assert len(rows) == 4


def test_log_window_end_offset_matches_bytes_in_returned_window(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_bytes(b"prefix-payload")

    payload, end_offset = benchmark._read_log_window(server_log, len(b"prefix-"))

    assert payload == b"payload"
    assert end_offset == len(b"prefix-payload")


def test_graph_summary_requires_tp4_nine_budget_startup() -> None:
    summary = benchmark.summarize_encoder_graph_log(_graph_startup_log())

    assert summary["initialized"] == 4
    assert summary["captured_total"] == 36
    assert summary["capture_details_total"] == 36
    assert summary["observed_ranks"] == [0, 1, 2, 3]
    assert all(
        summary["per_rank"][str(rank)]["captured_total"] == 9
        for rank in benchmark.EXPECTED_RANKS
    )
    assert benchmark.validate_graph_summary("graph", summary) == []
    assert benchmark.validate_graph_summary("eager", summary)


def test_graph_summary_rejects_balanced_but_wrong_per_rank_completion_counts() -> None:
    log = _graph_startup_log()
    log = log.replace("9 budget graphs.", "8 budget graphs.", 1)
    log = log.replace("9 budget graphs.", "10 budget graphs.", 1)
    summary = benchmark.summarize_encoder_graph_log(log)

    assert sum(summary["capture_complete_graph_counts"]) == 36
    assert any(
        "each TP rank" in failure
        for failure in benchmark.validate_graph_summary("graph", summary)
    )


def test_graph_summary_rejects_all_capture_markers_from_one_rank() -> None:
    log = re.sub(r"ATTN TP RANK [0-3]", "ATTN TP RANK 0", _graph_startup_log())

    summary = benchmark.summarize_encoder_graph_log(log)
    failures = benchmark.validate_graph_summary("graph", summary)

    assert summary["observed_ranks"] == [0]
    assert any("expected [0, 1, 2, 3]" in failure for failure in failures)
    assert any(
        "rank 1 has no encoder graph startup summary" in failure for failure in failures
    )


def test_graph_summary_rejects_detailed_budgets_attributed_to_wrong_rank() -> None:
    lines = []
    for line in _graph_startup_log().splitlines(keepends=True):
        if "Captured encoder cudagraph" in line:
            line = re.sub(r"ATTN TP RANK [0-3]", "ATTN TP RANK 0", line)
        lines.append(line)

    summary = benchmark.summarize_encoder_graph_log("".join(lines))
    failures = benchmark.validate_graph_summary("graph", summary)

    assert summary["observed_ranks"] == [0, 1, 2, 3]
    assert any("rank 1 detailed capture total is 0" in failure for failure in failures)


def test_eager_summary_rejects_detailed_capture_without_completion() -> None:
    summary = benchmark.summarize_encoder_graph_log(
        "[ATTN TP RANK 0] Captured encoder cudagraph: modality=image, budget=128\n"
    )

    assert any(
        "capture_details_total" in failure
        for failure in benchmark.validate_graph_summary("eager", summary)
    )


def test_response_contract_checks_dog_tokens_and_logprob() -> None:
    passed = benchmark.validate_response(_response(-0.0015), -0.001)
    failed = benchmark.validate_response(_response(-0.5), -0.001)

    assert passed["failures"] == []
    assert failed["failures"]
    assert failed["reference_logprob_absolute_delta"] == pytest.approx(0.499)


def test_server_info_requires_graph_flag_and_timing_contract() -> None:
    args = _artifact("graph", 8.0, 20.0)["server"]["args"]

    _, parsed = benchmark._extract_server_info({"server_args": args}, "graph")
    assert parsed["enable_mm_encoder_cuda_graph"] is True

    args["enable_log_mm_timing"] = False
    with pytest.raises(benchmark.BenchmarkError, match="enable_log_mm_timing"):
        benchmark._extract_server_info({"server_args": args}, "graph")


def test_server_info_retries_control_sidecar_startup() -> None:
    class DelayedClient:
        attempts = 0

        def get_json(self, _url, _timeout):
            self.attempts += 1
            if self.attempts < 3:
                raise benchmark.BenchmarkError("connection refused")
            return {"server_args": {}}

    client = DelayedClient()
    raw, attempts = benchmark.fetch_server_info_with_retry(
        client, "http://127.0.0.1:8001/get_server_info", 2.0
    )

    assert raw == {"server_args": {}}
    assert attempts == 3


def test_collect_arm_serializes_sixty_requests_and_retains_raw_rank_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text("")
    reference_path = tmp_path / "reference.json"
    dog_path = tmp_path / "dog.jpg"
    reference = benchmark.Reference(
        path=reference_path,
        sha256="reference-sha",
        raw={"prompt": benchmark.EXPECTED_PROMPT},
        reference_logprob=-0.001,
    )
    monkeypatch.setattr(benchmark, "load_reference", lambda *_: reference)
    monkeypatch.setattr(benchmark, "build_request", lambda *_: {"fixed": True})

    class FakeClient:
        def get_json(self, _url, _timeout):
            return {"server_args": _artifact("eager", 10.0, 30.0)["server"]["args"]}

        def post_json(self, _url, body, _timeout):
            assert body == {"fixed": True}
            with server_log.open("a") as handle:
                handle.write(
                    "".join(
                        _timing_line(rank, 10.0 + rank)
                        for rank in benchmark.EXPECTED_RANKS
                    )
                )
                handle.write("Req: chatcmpl-test Finish! Accept_num_tokens_avg: 0\n")
            return _response()

    output = tmp_path / "eager.json"
    config = benchmark.CollectConfig(
        arm="eager",
        launch_id="launch-1",
        base_url="http://127.0.0.1:8000",
        server_info_base_url="http://127.0.0.1:8001",
        model="minimax-m3",
        dog=dog_path,
        reference=reference_path,
        server_log=server_log,
        server_sha="server-sha",
        output=output,
    )

    artifact = benchmark.collect_arm(config, FakeClient())

    assert artifact["ok"] is True
    assert len(artifact["warmup"]) == 10
    assert len(artifact["measured"]) == 50
    assert artifact["statistics"]["critical_path_encode_ms"]["median"] == 13.0
    assert artifact["measured"][0]["timing"]["per_rank"]["3"]["encode_ms"] == 13.0
    assert artifact["measured"][0]["raw"]["response"] == _response()
    assert artifact["compile_events"]["measurement_count"] == 0
    assert artifact["compile_events"]["measurement_span"]["events"] == []
    assert (
        artifact["compile_events"]["measurement_span"]["server_log_end_offset"]
        > artifact["compile_events"]["measurement_span"]["server_log_start_offset"]
    )
    assert json.loads(output.read_text())["status"] == "complete"


def test_compare_passes_speed_quality_ci_and_ephemeral_port_differences(
    tmp_path: Path,
) -> None:
    eager_paths = []
    graph_paths = []
    for launch_id in ("launch-1", "launch-2"):
        eager_artifact = _artifact("eager", 10.0, 30.0)
        eager_artifact["launch_id"] = launch_id
        eager_paths.append(
            _write_artifact(tmp_path / f"{launch_id}-eager.json", eager_artifact)
        )
        graph_artifact = _artifact("graph", 8.0, 27.0)
        graph_artifact["launch_id"] = launch_id
        graph_paths.append(
            _write_artifact(tmp_path / f"{launch_id}-graph.json", graph_artifact)
        )
    output = tmp_path / "comparison.json"

    report = benchmark.compare_arms(
        eager_paths,
        graph_paths,
        output=output,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert report["ok"] is True
    assert report["statistics"]["critical_path_encode"][
        "graph_over_eager_median_ratio"
    ] == pytest.approx(0.8)
    assert report["statistics"]["critical_path_encode"]["bootstrap"]["ratio_ci"] == {
        "lower": pytest.approx(0.8),
        "upper": pytest.approx(0.8),
    }
    assert all(report["gates"].values())
    assert json.loads(output.read_text())["ok"] is True


def test_compare_single_launch_is_debug_only_and_fails_release_gate(
    tmp_path: Path,
) -> None:
    eager = _write_artifact(tmp_path / "eager.json", _artifact("eager", 10.0, 30.0))
    graph = _write_artifact(tmp_path / "graph.json", _artifact("graph", 8.0, 27.0))

    report = benchmark.compare_arms(
        [eager],
        [graph],
        output=tmp_path / "comparison.json",
        bootstrap_samples=20,
    )

    assert report["ok"] is False
    assert report["gates"]["at_least_two_independent_launch_pairs"] is False
    assert any(
        "at least 2 independent launch pairs" in item for item in report["failures"]
    )


def test_compare_rejects_non_graph_server_argument_difference(tmp_path: Path) -> None:
    eager_artifact = _artifact("eager", 10.0, 30.0)
    graph_artifact = _artifact("graph", 8.0, 27.0)
    graph_artifact["server"]["args"]["max_num_seqs"] = 8
    eager_artifact["server"]["args"]["max_num_seqs"] = 2
    eager = _write_artifact(tmp_path / "eager.json", eager_artifact)
    graph = _write_artifact(tmp_path / "graph.json", graph_artifact)

    report = benchmark.compare_arms(
        [eager],
        [graph],
        output=tmp_path / "comparison.json",
        bootstrap_samples=20,
    )

    assert report["ok"] is False
    assert report["gates"]["server_args_differ_only_by_graph_flag"] is False
    assert "max_num_seqs" in json.dumps(report["server_arg_differences"])


def test_compare_enforces_ten_percent_speed_gate(tmp_path: Path) -> None:
    eager = _write_artifact(tmp_path / "eager.json", _artifact("eager", 10.0, 30.0))
    graph = _write_artifact(tmp_path / "graph.json", _artifact("graph", 9.5, 30.0))

    report = benchmark.compare_arms(
        [eager],
        [graph],
        output=tmp_path / "comparison.json",
        bootstrap_samples=20,
    )

    assert report["ok"] is False
    assert report["gates"]["median_encoder_at_least_ten_percent_faster"] is False
    assert report["gates"]["no_launch_encoder_regression_over_five_percent"] is True


def test_compare_rejects_duplicate_artifact_path(tmp_path: Path) -> None:
    eager = _write_artifact(tmp_path / "eager.json", _artifact("eager", 10.0, 30.0))
    graph_one_artifact = _artifact("graph", 8.0, 27.0)
    graph_one = _write_artifact(tmp_path / "graph-one.json", graph_one_artifact)
    graph_two_artifact = _artifact("graph", 8.0, 27.0)
    graph_two_artifact["launch_id"] = "launch-2"
    graph_two = _write_artifact(tmp_path / "graph-two.json", graph_two_artifact)

    with pytest.raises(benchmark.BenchmarkError, match="duplicate eager artifact"):
        benchmark.compare_arms(
            [eager, eager],
            [graph_one, graph_two],
            output=tmp_path / "comparison.json",
            bootstrap_samples=20,
        )


def test_compare_rejects_duplicate_launch_id(tmp_path: Path) -> None:
    eager_one = _write_artifact(
        tmp_path / "eager-one.json", _artifact("eager", 10.0, 30.0)
    )
    eager_two = _write_artifact(
        tmp_path / "eager-two.json", _artifact("eager", 10.0, 30.0)
    )
    graph_one = _write_artifact(
        tmp_path / "graph-one.json", _artifact("graph", 8.0, 27.0)
    )
    graph_two = _write_artifact(
        tmp_path / "graph-two.json", _artifact("graph", 8.0, 27.0)
    )

    with pytest.raises(benchmark.BenchmarkError, match="duplicate eager launch_id"):
        benchmark.compare_arms(
            [eager_one, eager_two],
            [graph_one, graph_two],
            output=tmp_path / "comparison.json",
            bootstrap_samples=20,
        )


def test_compare_fails_mismatched_hardware_or_package_provenance(
    tmp_path: Path,
) -> None:
    eager_artifact = _artifact("eager", 10.0, 30.0)
    graph_artifact = _artifact("graph", 8.0, 27.0)
    graph_artifact["provenance"]["hardware"]["sha256"] = "b" * 64
    eager = _write_artifact(tmp_path / "eager.json", eager_artifact)
    graph = _write_artifact(tmp_path / "graph.json", graph_artifact)

    report = benchmark.compare_arms(
        [eager],
        [graph],
        output=tmp_path / "comparison.json",
        bootstrap_samples=20,
    )

    assert report["ok"] is False
    assert report["gates"]["hardware_environment_and_packages_match"] is False


def test_multi_launch_comparison_uses_paired_cluster_bootstrap(tmp_path: Path) -> None:
    artifacts = []
    for arm, launch_id, encode_ms in (
        ("eager", "launch-1", 10.0),
        ("eager", "launch-2", 20.0),
        ("graph", "launch-1", 8.0),
        ("graph", "launch-2", 10.0),
    ):
        artifact = _artifact(arm, encode_ms, encode_ms * 3)
        artifact["launch_id"] = launch_id
        artifacts.append(
            _write_artifact(tmp_path / f"{arm}-{launch_id}.json", artifact)
        )

    report = benchmark.compare_arms(
        artifacts[:2],
        artifacts[2:],
        output=tmp_path / "comparison.json",
        bootstrap_samples=200,
        bootstrap_seed=11,
    )

    bootstrap = report["statistics"]["critical_path_encode"]["bootstrap"]
    assert report["ok"] is True
    assert bootstrap["launch_pairs"] == 2
    assert "cluster bootstrap" in bootstrap["method"]
    assert bootstrap["ratio_ci"]["lower"] <= 0.5
    assert bootstrap["ratio_ci"]["upper"] >= 0.8


def test_cli_contract_fixes_default_sample_counts() -> None:
    args = benchmark.parse_args(
        [
            "collect",
            "--arm",
            "eager",
            "--base-url",
            "http://127.0.0.1:8000",
            "--server-info-base-url",
            "http://127.0.0.1:8001",
            "--dog",
            "dog.jpg",
            "--reference",
            "reference.json",
            "--server-log",
            "server.log",
            "--server-sha",
            "abc",
            "--output",
            "eager.json",
        ]
    )

    assert not hasattr(args, "warmup_requests")
    assert not hasattr(args, "measure_requests")


def test_ci_contract_runs_paired_orchestrator_in_one_isolated_job() -> None:
    task = yaml.safe_load(
        (REPO_ROOT / "test/ci/perf/minimax-m3-mxfp8-encoder-graph-ab.yaml").read_text()
    )

    assert task["triggers"] == ["manual"]
    assert task["runner"]["labels"] == ["b200-4gpu"]
    assert task["isolated_python"] is True
    assert task["timeout_minutes"] == 240
    assert "server" not in task
    assert any("--install-published" in command for command in task["install"])
    assert not any("runtime_env_preflight.py" in command for command in task["install"])
    assert task["preflight"] == [
        "python3 test/ci_system/runtime_env_preflight.py "
        "--output .ci-artifacts/minimax-m3-encoder-graph-ab/"
        "runtime-environment.json"
    ]
    command = task["perf"]["command"]
    assert "minimax_m3_encoder_graph_ab_ci.py" in command
    assert "--runtime-environment" in command
    assert "--smg-packages" in command
    assert "--pip-install-report" in command
    assert "--bootstrap-samples 10000" in command
