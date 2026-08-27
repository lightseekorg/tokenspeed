import subprocess
import time
from pathlib import Path


def _run_cleanup_script(script: str, timeout: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, timeout=timeout
    )


def test_stop_worker_pids_forces_process_that_ignores_sigterm():
    helper = Path(__file__).with_name("worker_cleanup.sh")
    script = f"""
source {helper}
bash -c 'trap "" TERM; while true; do sleep 10; done' &
worker=$!
sleep 0.1
stop_worker_pids test-cleanup 1 "$worker"
if kill -0 "$worker" 2>/dev/null; then
  exit 1
fi
"""
    started = time.monotonic()
    result = _run_cleanup_script(script)

    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 4
    assert "forcing 1 processes to stop after 1s" in result.stdout


def test_pid_is_live_rejects_zombies_and_missing_pids(tmp_path):
    helper = Path(__file__).with_name("worker_cleanup.sh")
    zombie_file = tmp_path / "zombie_pid"
    script = f"""
source {helper}
python3 - <<'PY' &
import os, time
child = os.fork()
if child == 0:
    os._exit(0)
open({str(zombie_file)!r}, "w").write(str(child))
time.sleep(8)
os.waitpid(child, 0)
PY
parent=$!
for _ in $(seq 1 50); do
  [[ -s {str(zombie_file)!r} ]] && break
  sleep 0.05
done
zombie=$(cat {str(zombie_file)!r})
if pid_is_live "$zombie"; then
  echo "zombie $zombie reported live" >&2
  kill "$parent" 2>/dev/null || true
  exit 1
fi
if ! pid_is_live "$parent"; then
  echo "parent $parent reported dead" >&2
  exit 1
fi
if pid_is_live 999999; then
  echo "missing pid reported live" >&2
  kill "$parent" 2>/dev/null || true
  exit 1
fi
kill "$parent" 2>/dev/null || true
wait "$parent" 2>/dev/null || true
"""
    result = _run_cleanup_script(script)
    assert result.returncode == 0, result.stderr + result.stdout


def test_free_listen_ports_kills_stale_listener(tmp_path):
    helper = Path(__file__).with_name("worker_cleanup.sh")
    port_file = tmp_path / "port"
    script = f"""
source {helper}
python3 - <<'PY' &
import socket, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0))
open({str(port_file)!r}, "w").write(str(s.getsockname()[1]))
s.listen(1)
time.sleep(8)
PY
server=$!
for _ in $(seq 1 50); do
  [[ -s {str(port_file)!r} ]] && break
  sleep 0.05
done
port=$(cat {str(port_file)!r})
free_listen_ports test-ports "$port"
sleep 0.2
if kill -0 "$server" 2>/dev/null; then
  kill -KILL "$server" 2>/dev/null || true
  echo "listener still alive after free_listen_ports" >&2
  exit 1
fi
"""
    result = _run_cleanup_script(script)
    assert result.returncode == 0, result.stderr + result.stdout


def test_worker_launchers_use_bounded_cleanup():
    ci_dir = Path(__file__).parent
    for name in (
        "serve_qwen35_122b_nvfp4_epd_1e1p2d.sh",
        "serve_qwen35_397b_nvfp4_pd_1p1d.sh",
    ):
        script = (ci_dir / name).read_text()
        assert 'source "$SCRIPT_DIR/worker_cleanup.sh"' in script
        assert "stop_worker_pids" in script
        assert "pid_is_live" in script
        assert "free_listen_ports" in script
        subprocess.run(["bash", "-n", ci_dir / name], check=True)
