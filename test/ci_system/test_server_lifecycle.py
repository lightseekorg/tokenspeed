import io
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import server_lifecycle as lifecycle
from server_lifecycle import (
    cleanup_stale_managed_servers,
    start_managed_server,
    stop_managed_server,
)


def _clean_gpu_probe(indices):
    return {
        "ok": True,
        "gpus": {
            str(index): {
                "uuid": f"GPU-{index}",
                "memory_used_mib": 0,
                "utilization_percent": 0,
                "compute_pids": [],
            }
            for index in indices
        },
    }


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _shutdown_config(output: str, **overrides):
    config = {
        "target": "root",
        "signal": "SIGTERM",
        "timeout_seconds": 5,
        "expected_exit_code": 0,
        "ports": [_closed_port()],
        "gpu_indices": [0],
        "max_memory_mib": 4,
        "output": output,
    }
    config.update(overrides)
    return config


def _start_script(tmp_path: Path, source: str, gpu_probe=_clean_gpu_probe):
    script = tmp_path / "server.py"
    script.write_text(source)
    log = tmp_path / "server.log"
    server = start_managed_server(
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
        env={"PATH": "/usr/bin:/bin"},
        cwd=tmp_path,
        log_path=log,
        gpu_indices=[0],
        gpu_probe=gpu_probe,
        registry_root=tmp_path / ".managed-server-registry",
    )
    assert server is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log.exists() and "READY" in log.read_text():
            break
        if server.process.poll() is not None:
            raise AssertionError(f"server exited early: {server.process.returncode}")
        time.sleep(0.02)
    else:
        raise AssertionError("server did not become ready")
    return server


def test_managed_shutdown_validates_root_child_ports_gpu_and_artifact(tmp_path):
    server = _start_script(
        tmp_path,
        """
import signal
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([
    sys.executable,
    "-c",
    (
        "import os,signal,sys,time; "
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(open(sys.argv[1], 'w').write('unexpected'), os._exit(9))); "
        "signal.signal(signal.SIGUSR1, lambda *_: os._exit(0)); "
        "open(sys.argv[2], 'w').write('ready'); "
        "time.sleep(60)"
    ),
    "child-received-group-term",
    "child-ready",
])

while not Path("child-ready").exists():
    time.sleep(0.01)

def shutdown(*_):
    child.send_signal(signal.SIGUSR1)
    child.wait(timeout=5)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, shutdown)
print("READY", flush=True)
while True:
    time.sleep(1)
""",
    )
    output = ".ci-artifacts/shutdown.json"

    result = stop_managed_server(
        server,
        _shutdown_config(output),
        cwd=tmp_path,
    )

    artifact = json.loads((tmp_path / output).read_text())
    assert result["passed"] is True
    assert result["fallback_used"] is False
    assert result["root"]["pid"] == server.process.pid
    assert result["root"]["name"].startswith("python")
    assert result["process_group"] == result["root"]["pid"]
    assert result["descendants_before_signal"]
    assert result["process_group_after_shutdown"] == []
    assert result["survivors"] == []
    assert result["open_ports"] == []
    assert result["output_stream_open"] is False
    assert not (tmp_path / "child-received-group-term").exists()
    assert artifact == result


def test_managed_shutdown_fails_on_nonzero_root_exit(tmp_path):
    server = _start_script(
        tmp_path,
        """
import os
import signal
import time

signal.signal(signal.SIGTERM, lambda *_: os._exit(7))
print("READY", flush=True)
while True:
    time.sleep(1)
""",
    )

    result = stop_managed_server(
        server,
        _shutdown_config("shutdown-nonzero.json"),
        cwd=tmp_path,
    )

    assert result["passed"] is False
    assert result["returncode"] == 7
    assert result["fallback_used"] is True
    assert any("exit code 7" in failure for failure in result["failures"])


def test_managed_shutdown_timeout_is_failure_even_after_fallback_cleanup(tmp_path):
    server = _start_script(
        tmp_path,
        """
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("READY", flush=True)
while True:
    time.sleep(1)
""",
    )

    result = stop_managed_server(
        server,
        _shutdown_config("shutdown-timeout.json", timeout_seconds=1),
        cwd=tmp_path,
    )

    assert result["passed"] is False
    assert result["fallback_used"] is True
    assert any(
        "did not exit within 1 seconds" in failure for failure in result["failures"]
    )
    assert server.process.poll() is not None


def test_managed_shutdown_rejects_gpu_compute_residue(tmp_path):
    calls = 0

    def gpu_probe(indices):
        nonlocal calls
        calls += 1
        result = _clean_gpu_probe(indices)
        if calls == 2:
            result["gpus"]["0"]["memory_used_mib"] = 32
            result["gpus"]["0"]["compute_pids"] = [4242]
        return result

    server = _start_script(
        tmp_path,
        """
import signal
import time

signal.signal(signal.SIGTERM, lambda *_: exit(0))
print("READY", flush=True)
while True:
    time.sleep(1)
""",
        gpu_probe=gpu_probe,
    )

    result = stop_managed_server(
        server,
        _shutdown_config("shutdown-gpu.json"),
        cwd=tmp_path,
    )

    assert result["passed"] is False
    assert result["fallback_used"] is True
    assert any("retains compute PIDs" in failure for failure in result["failures"])
    assert any("retains 32 MiB" in failure for failure in result["failures"])


def test_managed_start_rejects_preexisting_gpu_compute_process(tmp_path):
    def dirty_gpu_probe(indices):
        result = _clean_gpu_probe(indices)
        result["gpus"]["0"]["compute_pids"] = [4242]
        return result

    with pytest.raises(RuntimeError, match=r"GPU 0 has compute PIDs \[4242\]"):
        _start_script(tmp_path, "print('SHOULD NOT START')\n", dirty_gpu_probe)


def test_managed_start_rejects_preexisting_gpu_memory(tmp_path):
    def dirty_gpu_probe(indices):
        result = _clean_gpu_probe(indices)
        result["gpus"]["0"]["memory_used_mib"] = 17
        return result

    with pytest.raises(RuntimeError, match="maximum clean baseline is 16 MiB"):
        _start_script(tmp_path, "print('SHOULD NOT START')\n", dirty_gpu_probe)


def test_managed_shutdown_fails_closed_when_server_never_started(tmp_path):
    output = "shutdown-not-started.json"

    result = stop_managed_server(
        None,
        _shutdown_config(output),
        cwd=tmp_path,
    )

    assert result["passed"] is False
    assert result["failures"] == ["managed server was not started"]
    assert json.loads((tmp_path / output).read_text()) == result


def test_managed_start_replaces_invalid_utf8_in_server_output(tmp_path):
    server = _start_script(
        tmp_path,
        """
import os
import signal
import time

signal.signal(signal.SIGTERM, lambda *_: exit(0))
os.write(1, b"\\xffREADY\\n")
while True:
    time.sleep(1)
""",
    )

    result = stop_managed_server(
        server,
        _shutdown_config("shutdown-invalid-utf8.json"),
        cwd=tmp_path,
    )

    assert result["passed"] is True
    assert "\ufffdREADY" in (tmp_path / "server.log").read_text()
    assert result["output_capture_errors"] == []


def test_managed_shutdown_fails_when_log_capture_reported_an_error(tmp_path):
    server = _start_script(
        tmp_path,
        """
import signal
import time

signal.signal(signal.SIGTERM, lambda *_: exit(0))
print("READY", flush=True)
while True:
    time.sleep(1)
""",
    )
    server.output_errors.append("OSError: simulated disk full")

    result = stop_managed_server(
        server,
        _shutdown_config("shutdown-log-error.json"),
        cwd=tmp_path,
    )

    assert result["passed"] is False
    assert result["output_capture_errors"] == ["OSError: simulated disk full"]
    assert any("server log capture failed" in item for item in result["failures"])


def test_tee_output_records_log_write_failure_instead_of_dying_silently():
    class BrokenLog:
        def write(self, _line):
            raise OSError("simulated write failure")

        def flush(self):
            pass

        def close(self):
            pass

    errors = []
    lifecycle._tee_output(io.StringIO("one line\n"), BrokenLog(), errors)

    assert errors == ["OSError: simulated write failure"]


def test_managed_start_aborts_root_when_output_thread_start_raises(
    tmp_path, monkeypatch
):
    spawned = []
    real_popen = lifecycle.subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    def fail_thread_start(_thread):
        raise RuntimeError("simulated thread start failure")

    monkeypatch.setattr(lifecycle.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(lifecycle.threading.Thread, "start", fail_thread_start)

    with pytest.raises(RuntimeError, match="simulated thread start failure"):
        start_managed_server(
            f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'",
            env={"PATH": "/usr/bin:/bin", "RUNNER_NAME": "thread-failure-test"},
            cwd=tmp_path,
            log_path=tmp_path / "thread-failure.log",
            gpu_indices=[0],
            gpu_probe=_clean_gpu_probe,
            registry_root=tmp_path / "registry",
        )

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert not list((tmp_path / "registry").rglob("*.json"))


def test_managed_start_aborts_root_when_signal_registration_raises(
    tmp_path, monkeypatch
):
    spawned = []
    real_popen = lifecycle.subprocess.Popen
    signal_mask_before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    handlers_before = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    def fail_registration(_server):
        raise RuntimeError("simulated signal registration failure")

    monkeypatch.setattr(lifecycle.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(lifecycle, "_register_active_server", fail_registration)

    with pytest.raises(RuntimeError, match="simulated signal registration failure"):
        start_managed_server(
            f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'",
            env={"PATH": "/usr/bin:/bin", "RUNNER_NAME": "register-failure-test"},
            cwd=tmp_path,
            log_path=tmp_path / "register-failure.log",
            gpu_indices=[0],
            gpu_probe=_clean_gpu_probe,
            registry_root=tmp_path / "registry",
        )

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert not list((tmp_path / "registry").rglob("*.json"))
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == signal_mask_before
    assert {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    } == handlers_before


def test_next_setup_recovers_registered_server_after_workdir_is_deleted(tmp_path):
    workdir = tmp_path / "deleted-workdir"
    workdir.mkdir()
    script = workdir / "server.py"
    script.write_text(textwrap.dedent("""
            import signal
            import time

            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("READY", flush=True)
            while True:
                time.sleep(1)
            """))
    registry_root = tmp_path / "durable-registry"
    start_env = {
        "PATH": "/usr/bin:/bin",
        "RUNNER_NAME": "durable-runner",
        "CI_TASK_NAME": "nested-ab-driver",
    }
    server = start_managed_server(
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
        env=start_env,
        cwd=workdir,
        log_path=workdir / "server.log",
        gpu_indices=[0],
        gpu_probe=_clean_gpu_probe,
        registry_root=registry_root,
    )
    assert server is not None
    assert server.registry_path is not None and server.registry_path.exists()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        log_path = workdir / "server.log"
        if log_path.exists() and "READY" in log_path.read_text():
            break
        if server.process.poll() is not None:
            raise AssertionError("durability test server exited before readiness")
        time.sleep(0.02)
    else:
        raise AssertionError("durability test server did not become ready")
    lifecycle._unregister_active_server(server)

    # GitHub's always() cleanup removes WORK_DIR even after cancellation.  The
    # registry remains runner-local and the next (possibly different) task sees
    # the same runner scope inherited by a nested A/B driver.
    import shutil

    shutil.rmtree(workdir)
    recovery_env = {
        "PATH": "/usr/bin:/bin",
        "RUNNER_NAME": "durable-runner",
        "CI_TASK_NAME": "different-next-task",
    }
    recovered = cleanup_stale_managed_servers(recovery_env, registry_root=registry_root)
    server.process.wait(timeout=5)
    server.output_thread.join(timeout=2)

    assert [item["status"] for item in recovered] == ["recovered"]
    assert server.process.poll() is not None
    assert not list(registry_root.rglob("*.json"))


def test_registry_pid_reuse_guard_never_signals_mismatched_identity(
    tmp_path, monkeypatch
):
    env = {"RUNNER_NAME": "pid-reuse-test"}
    registry_root = tmp_path / "registry"
    identity = lifecycle._read_process(os.getpid())
    assert identity is not None
    record = lifecycle._registry_record(
        command="unrelated",
        cwd=tmp_path,
        env=env,
        root_identity=identity,
    )
    record["root"]["start_time_ticks"] += 1
    directory = lifecycle._registry_directory(env, registry_root)
    path = directory / "reused-pid.json"
    lifecycle._write_json_atomic(path, record)

    def fail_if_signalled(*args, **kwargs):
        raise AssertionError("identity-mismatched PID must not be signalled")

    monkeypatch.setattr(lifecycle, "_terminate_identities", fail_if_signalled)
    result = cleanup_stale_managed_servers(env, registry_root=registry_root)

    assert result == [
        {"registry": str(path), "status": "stale_identity", "pid": os.getpid()}
    ]
    assert not path.exists()


def test_sigterm_handler_aborts_active_server_and_removes_registry(tmp_path):
    helper = tmp_path / "signal_helper.py"
    pid_path = tmp_path / "server.pid"
    registry_root = tmp_path / "registry"
    helper.write_text(textwrap.dedent(f"""
            import shlex
            import sys
            import time
            from pathlib import Path

            sys.path.insert(0, {str(Path(lifecycle.__file__).parent)!r})
            from server_lifecycle import start_managed_server

            def clean_gpu(indices):
                return {{
                    "ok": True,
                    "gpus": {{
                        str(index): {{
                            "uuid": f"GPU-{{index}}",
                            "memory_used_mib": 0,
                            "utilization_percent": 0,
                            "compute_pids": [],
                        }}
                        for index in indices
                    }},
                }}

            command = (
                shlex.quote(sys.executable)
                + " -c "
                + shlex.quote("import time; time.sleep(60)")
            )
            server = start_managed_server(
                command,
                env={{"PATH": "/usr/bin:/bin", "RUNNER_NAME": "signal-runner"}},
                cwd=Path({str(tmp_path)!r}),
                log_path=Path({str(tmp_path / 'signal-server.log')!r}),
                gpu_indices=[0],
                gpu_probe=clean_gpu,
                registry_root=Path({str(registry_root)!r}),
            )
            Path({str(pid_path)!r}).write_text(str(server.process.pid))
            while True:
                time.sleep(1)
            """))
    helper_process = subprocess.Popen(
        [sys.executable, str(helper)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not pid_path.exists():
        if helper_process.poll() is not None:
            output = helper_process.stdout.read() if helper_process.stdout else ""
            raise AssertionError(f"signal helper exited early: {output}")
        time.sleep(0.02)
    assert pid_path.exists()
    server_pid = int(pid_path.read_text())

    os.kill(helper_process.pid, signal.SIGTERM)
    helper_process.wait(timeout=10)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and lifecycle._read_process(server_pid):
        time.sleep(0.02)

    assert helper_process.returncode == 128 + signal.SIGTERM
    assert lifecycle._read_process(server_pid) is None
    assert not list(registry_root.rglob("*.json"))


def test_managed_start_restores_parent_and_child_signal_masks(tmp_path):
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    if any(
        signum in original_mask
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
    ):
        pytest.skip("test requires TERM, INT, and USR1 to start unblocked")

    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
    expected_mask = original_mask | {signal.SIGUSR1}
    server = None
    try:
        server = _start_script(
            tmp_path,
            """
import json
import signal
import time
from pathlib import Path

blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
Path("server-signal-mask.json").write_text(
    json.dumps(sorted(int(signum) for signum in blocked))
)
signal.signal(signal.SIGTERM, lambda *_: exit(0))
print("READY", flush=True)
while True:
    time.sleep(1)
""",
        )
        parent_mask_after_start = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        child_mask = {
            signal.Signals(signum)
            for signum in json.loads((tmp_path / "server-signal-mask.json").read_text())
        }
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)

    assert parent_mask_after_start == expected_mask
    assert child_mask == expected_mask
    assert signal.SIGTERM not in child_mask
    assert signal.SIGINT not in child_mask
    try:
        result = stop_managed_server(
            server,
            _shutdown_config("shutdown-signal-mask.json"),
            cwd=tmp_path,
        )
        assert result["passed"] is True
    finally:
        if server is not None and server.process.poll() is None:
            lifecycle.abort_managed_server(server)


def test_sigterm_in_popen_registration_window_cleans_process_and_registry(tmp_path):
    helper = tmp_path / "atomic_start_signal_helper.py"
    child_pid_path = tmp_path / "atomic-child.pid"
    signal_sent_path = tmp_path / "atomic-signal-sent"
    registered_path = tmp_path / "atomic-server-registered"
    registry_root = tmp_path / "atomic-registry"
    helper.write_text(textwrap.dedent(f"""
            import os
            import shlex
            import signal
            import sys
            import time
            from pathlib import Path

            sys.path.insert(0, {str(Path(lifecycle.__file__).parent)!r})
            import server_lifecycle as lifecycle

            def clean_gpu(indices):
                return {{
                    "ok": True,
                    "gpus": {{
                        str(index): {{
                            "uuid": f"GPU-{{index}}",
                            "memory_used_mib": 0,
                            "utilization_percent": 0,
                            "compute_pids": [],
                        }}
                        for index in indices
                    }},
                }}

            real_popen = lifecycle.subprocess.Popen
            real_register = lifecycle._register_active_server

            def signal_after_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                Path({str(child_pid_path)!r}).write_text(str(process.pid))
                os.kill(os.getpid(), signal.SIGTERM)
                Path({str(signal_sent_path)!r}).write_text("sent")
                return process

            def record_registration(server):
                if server.registry_path is None or not server.registry_path.exists():
                    raise AssertionError("server registered before durable registry")
                real_register(server)
                Path({str(registered_path)!r}).write_text("registered")

            lifecycle.subprocess.Popen = signal_after_popen
            lifecycle._register_active_server = record_registration
            command = (
                shlex.quote(sys.executable)
                + " -c "
                + shlex.quote("import time; time.sleep(60)")
            )
            lifecycle.start_managed_server(
                command,
                env={{"PATH": "/usr/bin:/bin", "RUNNER_NAME": "atomic-runner"}},
                cwd=Path({str(tmp_path)!r}),
                log_path=Path({str(tmp_path / 'atomic-server.log')!r}),
                gpu_indices=[0],
                gpu_probe=clean_gpu,
                registry_root=Path({str(registry_root)!r}),
            )
            raise AssertionError("pending SIGTERM was not replayed")
            """))
    helper_process = subprocess.Popen(
        [sys.executable, str(helper)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server_pid = None
    try:
        helper_process.wait(timeout=15)
        output = helper_process.stdout.read() if helper_process.stdout else ""
        assert child_pid_path.exists(), output
        server_pid = int(child_pid_path.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and lifecycle._read_process(server_pid):
            time.sleep(0.02)

        assert helper_process.returncode == 128 + signal.SIGTERM, output
        assert signal_sent_path.exists()
        assert registered_path.exists()
        assert lifecycle._read_process(server_pid) is None
        assert not list(registry_root.rglob("*.json"))
        assert not list(registry_root.rglob("*.tmp"))
    finally:
        if helper_process.poll() is None:
            helper_process.kill()
            helper_process.wait(timeout=5)
        if server_pid is not None:
            identity = lifecycle._read_process(server_pid)
            if identity is not None and identity["process_group"] == server_pid:
                try:
                    os.killpg(server_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
