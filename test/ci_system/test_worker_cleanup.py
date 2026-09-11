import subprocess
import time
from pathlib import Path


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
    result = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, timeout=5
    )

    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 4
    assert "forcing 1 processes to stop after 1s" in result.stdout


def test_worker_launchers_use_bounded_cleanup():
    ci_dir = Path(__file__).parent
    for name in (
        "serve_qwen35_122b_nvfp4_epd_1e1p2d.sh",
        "serve_qwen35_397b_nvfp4_pd_1p1d.sh",
    ):
        script = (ci_dir / name).read_text()
        assert 'source "$SCRIPT_DIR/worker_cleanup.sh"' in script
        assert "stop_worker_pids" in script
        subprocess.run(["bash", "-n", ci_dir / name], check=True)
