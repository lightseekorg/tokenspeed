import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import minimax_m3_encoder_graph_ab_ci as orchestrator
import pytest


def test_graph_server_argv_adds_exactly_one_flag() -> None:
    eager = orchestrator.build_server_argv(graph=False)
    graph = orchestrator.build_server_argv(graph=True)
    graph_without_flag = list(graph)
    graph_without_flag.remove("--enable-mm-encoder-cuda-graph")

    assert graph_without_flag == eager
    assert graph.count("--enable-mm-encoder-cuda-graph") == 1
    assert "--enable-log-mm-timing" in eager
    assert "--enforce-eager" in eager
    assert "--disable-prefill-graph" in eager
    assert eager[eager.index("--base-gpu-id") + 1] == "0"
    assert eager[eager.index("--gpu-id-step") + 1] == "1"


def test_physical_gpu_selection_uses_cli_base_step_without_visibility_mask() -> None:
    gpu_indices = orchestrator.derive_gpu_indices(4, 1)
    eager = orchestrator.build_server_argv(
        graph=False,
        base_gpu_id=4,
        gpu_id_step=1,
    )

    assert gpu_indices == (4, 5, 6, 7)
    assert eager[eager.index("--base-gpu-id") + 1] == "4"
    assert eager[eager.index("--gpu-id-step") + 1] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(eager)


def test_physical_gpu_selection_cli_defaults_and_explicit_override() -> None:
    required = [
        "--output-dir",
        "output",
        "--dog",
        "dog.jpg",
        "--reference",
        "reference.json",
        "--runtime-environment",
        "runtime-environment.json",
        "--smg-packages",
        "smg-packages.json",
        "--pip-install-report",
        "pip-install-report.json",
        "--server-sha",
        "deadbeef",
    ]

    defaults = orchestrator.parse_args(required)
    local = orchestrator.parse_args(
        [*required, "--base-gpu-id", "4", "--gpu-id-step", "1"]
    )

    assert (defaults.base_gpu_id, defaults.gpu_id_step) == (0, 1)
    assert (local.base_gpu_id, local.gpu_id_step) == (4, 1)


@pytest.mark.parametrize(
    ("base_gpu_id", "gpu_id_step"),
    [(-1, 1), (0, 0), (0, -1), (True, 1), (0, True), ("0", 1), (0, "1")],
)
def test_physical_gpu_selection_rejects_invalid_base_or_step(
    base_gpu_id, gpu_id_step
) -> None:
    with pytest.raises(orchestrator.OrchestratorError):
        orchestrator.derive_gpu_indices(base_gpu_id, gpu_id_step)


def test_capture_hardware_identity_requires_selected_tp4(monkeypatch) -> None:
    rows = []
    for index in range(8):
        values = [
            str(index),
            f"GPU-{index}",
            "NVIDIA B200",
            "580.126.20",
            f"00000000:{index + 1:02X}:00.0",
            "10.0",
            "Enabled",
            "1000.00",
            "1965",
            "3996",
            "1965",
            "3996",
            "Default",
            "Disabled",
        ]
        rows.append(", ".join(values))
    monkeypatch.setattr(
        orchestrator.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="\n".join(rows) + "\n", stderr=""
        ),
    )

    identity = orchestrator.capture_hardware_identity((4, 5, 6, 7))

    assert [gpu["index"] for gpu in identity["gpus"]] == [4, 5, 6, 7]
    assert len({gpu["uuid"] for gpu in identity["gpus"]}) == 4
    assert identity["gpus"][0]["power.limit"] == "1000.00"


def test_readiness_checks_gateway_and_control_sidecar(monkeypatch) -> None:
    urls = []
    monkeypatch.setattr(
        orchestrator,
        "_url_ready",
        lambda url: urls.append(url) or True,
    )
    process = SimpleNamespace(poll=lambda: None)

    orchestrator.wait_for_server_ready(process, 1.0)

    assert urls == [
        "http://127.0.0.1:8000/readiness",
        "http://127.0.0.1:8001/get_server_info",
    ]


def test_server_log_scan_fails_lifecycle_traceback(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("ready\nTraceback (most recent call last):\nboom\n")

    result = orchestrator._scan_server_log(log)

    assert result["passed"] is False
    assert result["match_count"] == 1


def test_arm_still_runs_root_shutdown_when_collection_fails(
    tmp_path: Path, monkeypatch
) -> None:
    stopped = []
    idle_probes = []
    starts = []
    fake_server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))

    def require_idle(gpu_indices):
        idle_probes.append(tuple(gpu_indices))
        return {}

    def start(command, **kwargs):
        starts.append((command, kwargs))
        return fake_server

    monkeypatch.setattr(orchestrator, "_require_idle_gpus", require_idle)
    monkeypatch.setattr(orchestrator, "start_managed_server", start)
    monkeypatch.setattr(orchestrator, "wait_for_server_ready", lambda *args: None)
    monkeypatch.setattr(
        orchestrator.benchmark,
        "collect_arm",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("collect failed")),
    )

    def stop(server, shutdown, *, cwd):
        stopped.append((server, shutdown, cwd))
        return {"passed": True, "failures": []}

    monkeypatch.setattr(orchestrator, "stop_managed_server", stop)
    monkeypatch.setattr(
        orchestrator,
        "_scan_server_log",
        lambda _path: {"passed": True, "match_count": 0, "matches": []},
    )

    with pytest.raises(orchestrator.OrchestratorError, match="collect failed"):
        orchestrator._run_arm(
            arm="eager",
            launch_id="launch-1",
            work_dir=tmp_path,
            arm_dir=tmp_path / ".ci-artifacts/ab/launch-1/eager",
            dog=tmp_path / "dog.jpg",
            reference=tmp_path / "reference.json",
            server_sha="sha",
            provenance_files=(),
            request_timeout_seconds=1.0,
            base_gpu_id=4,
            gpu_id_step=1,
        )

    assert idle_probes == [(4, 5, 6, 7)]
    assert len(starts) == 1
    assert "--base-gpu-id 4" in starts[0][0]
    assert "--gpu-id-step 1" in starts[0][0]
    assert "CUDA_VISIBLE_DEVICES" not in starts[0][0]
    assert tuple(starts[0][1]["gpu_indices"]) == (4, 5, 6, 7)
    assert len(stopped) == 1
    assert stopped[0][1]["target"] == "root"
    assert stopped[0][1]["gpu_indices"] == [4, 5, 6, 7]


def test_arm_reports_shutdown_exception_after_unconditional_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    fake_server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(orchestrator, "_require_idle_gpus", lambda _indices: {})
    monkeypatch.setattr(
        orchestrator, "start_managed_server", lambda *args, **kwargs: fake_server
    )
    monkeypatch.setattr(orchestrator, "wait_for_server_ready", lambda *args: None)
    monkeypatch.setattr(
        orchestrator.benchmark, "collect_arm", lambda *args, **kwargs: {"ok": True}
    )
    monkeypatch.setattr(
        orchestrator,
        "stop_managed_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop failed")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_scan_server_log",
        lambda _path: {"passed": True, "match_count": 0, "matches": []},
    )

    with pytest.raises(
        orchestrator.OrchestratorError,
        match="shutdown raised RuntimeError: stop failed",
    ):
        orchestrator._run_arm(
            arm="graph",
            launch_id="launch-1",
            work_dir=tmp_path,
            arm_dir=tmp_path / ".ci-artifacts/ab/launch-1/graph",
            dog=tmp_path / "dog.jpg",
            reference=tmp_path / "reference.json",
            server_sha="sha",
            provenance_files=(),
            request_timeout_seconds=1.0,
            base_gpu_id=4,
            gpu_id_step=1,
        )


def test_run_rejects_external_output_before_hardware_or_server(
    tmp_path: Path, monkeypatch
) -> None:
    hardware_called = False

    def capture_hardware(_gpu_indices):
        nonlocal hardware_called
        hardware_called = True
        raise AssertionError("hardware must not be queried")

    monkeypatch.setattr(orchestrator, "capture_hardware_identity", capture_hardware)
    args = argparse.Namespace(
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "external-artifacts",
        dog=tmp_path / "dog.jpg",
        reference=tmp_path / "reference.json",
        runtime_environment=tmp_path / "runtime-environment.json",
        smg_packages=tmp_path / "smg-packages.json",
        pip_install_report=tmp_path / "pip-install.json",
        server_sha="server-sha",
        base_gpu_id=4,
        gpu_id_step=1,
        request_timeout_seconds=1.0,
        bootstrap_samples=20,
        bootstrap_seed=7,
    )

    with pytest.raises(
        orchestrator.OrchestratorError,
        match="--output-dir must resolve inside --work-dir",
    ):
        orchestrator.run(args)

    assert hardware_called is False
    assert not args.output_dir.exists()


def test_run_executes_counterbalanced_launch_pairs_then_comparator(
    tmp_path: Path, monkeypatch
) -> None:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "dog",
            "reference",
            "runtime_environment",
            "smg_packages",
            "pip_install_report",
        )
    }
    for path in paths.values():
        path.write_text("{}")
    hardware = {
        "schema_version": 1,
        "gpus": [{"index": index, "uuid": f"GPU-{index}"} for index in range(4, 8)],
    }
    hardware_probes = []

    def capture_hardware(gpu_indices):
        hardware_probes.append(tuple(gpu_indices))
        return hardware

    monkeypatch.setattr(orchestrator, "capture_hardware_identity", capture_hardware)
    arms = []

    def run_arm(**kwargs):
        arms.append(
            (
                kwargs["launch_id"],
                kwargs["arm"],
                kwargs["base_gpu_id"],
                kwargs["gpu_id_step"],
            )
        )
        return {"ok": True}

    monkeypatch.setattr(orchestrator, "_run_arm", run_arm)
    compared = []

    def compare(eager, graph, **kwargs):
        compared.append((eager, graph))
        return {"ok": True}

    monkeypatch.setattr(orchestrator.benchmark, "compare_arms", compare)
    args = argparse.Namespace(
        work_dir=tmp_path,
        output_dir=tmp_path / ".ci-artifacts/ab",
        dog=paths["dog"],
        reference=paths["reference"],
        runtime_environment=paths["runtime_environment"],
        smg_packages=paths["smg_packages"],
        pip_install_report=paths["pip_install_report"],
        server_sha="server-sha",
        base_gpu_id=4,
        gpu_id_step=1,
        request_timeout_seconds=1.0,
        bootstrap_samples=20,
        bootstrap_seed=7,
    )

    result = orchestrator.run(args)

    assert result["ok"] is True
    assert hardware_probes == [(4, 5, 6, 7)] * len(orchestrator.LAUNCH_SEQUENCE)
    assert [(launch_id, arm) for launch_id, arm, *_ in arms] == list(
        orchestrator.LAUNCH_SEQUENCE
    )
    assert all(base == 4 and step == 1 for _, _, base, step in arms)
    assert result["gpu_selection"] == {
        "base_gpu_id": 4,
        "gpu_id_step": 1,
        "physical_indices": [4, 5, 6, 7],
    }
    assert len(compared) == 1
    eager_paths, graph_paths = compared[0]
    assert eager_paths == [
        args.output_dir / "launch-1/eager/arm.json",
        args.output_dir / "launch-2/eager/arm.json",
    ]
    assert graph_paths == [
        args.output_dir / "launch-1/graph/arm.json",
        args.output_dir / "launch-2/graph/arm.json",
    ]
    assert [
        (row["launch_id"], row["arm"]) for row in result["launch_sequence"]
    ] == list(orchestrator.LAUNCH_SEQUENCE)
