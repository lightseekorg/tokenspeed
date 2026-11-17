"""Exercise an installed wheel and its host extension without GPU dependencies."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import tokenspeed_engine

BOOTSTRAP = """
import json
import os
import sys

def bootstrap(control_fd, outbox, protocol, host_spans=None):
    # Exercise native stdlib modules and a separately loaded extension.
    import _ctypes
    import _socket
    import _sqlite3
    import _ssl
    import zlib
    import threading
    import tokenspeed_scheduler as ts
    import setproctitle
    import tokenspeed_engine
    import tokenspeed_engine.scheduler

    scheduler = ts._ext
    assert scheduler is sys.modules["tokenspeed_engine._scheduler"]
    # The engine's own scheduler API is the same linked module.
    assert tokenspeed_engine.scheduler._ext is scheduler
    config = ts.SchedulerConfig()
    config.prefix_granularity = 16
    config.max_scheduled_tokens = 64
    config.max_batch_size = 4
    config.max_resident_requests = config.max_batch_size
    config.num_device_pages = 64
    config.num_snapshot_pages = 1
    config.cache_groups = [ts.CacheGroupConfig(
        group_id="full_attention", block_granularity=16,
        total_pages=64, attn_kind=ts.AttnKind.Full,
    )]
    assert ts.Scheduler(config).waiting_size() == 0

    os.write(control_fd, json.dumps({
        "prefix": sys.prefix,
        "executable": sys.executable,
        "scheduler": scheduler.__name__,
        "title": setproctitle.getproctitle(),
    }).encode())
    # A failed rank must exit even when Python finalization would wait forever.
    threading.Thread(target=threading.Event().wait, daemon=False).start()
    raise RuntimeError("engine wheel bootstrap verified")
"""


def main(expected_version: str, scheduler_python: str) -> None:
    installed_version = version("tokenspeed-engine")
    assert installed_version == expected_version, installed_version
    assert "tokenspeed_engine._host" not in sys.modules
    assert os.access(tokenspeed_engine.host_binary(), os.X_OK)
    signal_count = 0

    def count_failure_signal(_signum, _frame):
        nonlocal signal_count
        signal_count += 1

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        engine = root / "tokenspeed" / "runtime" / "engine"
        engine.mkdir(parents=True)
        for package in (engine, engine.parent, engine.parent.parent):
            (package / "__init__.py").touch()
        (engine / "rust_host.py").write_text(BOOTSTRAP)
        # A real descriptor carries the result through the host's bootstrap.
        with (root / "result.json").open("w+") as result:
            previous = signal.signal(signal.SIGUSR1, count_failure_signal)
            try:
                completed = subprocess.run(
                    [
                        "tokenspeed::wheel-check",
                        "--python",
                        sys.executable,
                        "--control-fd",
                        str(result.fileno()),
                        "--enable-async-logging",
                        "false",
                    ],
                    # A late-loaded host must work without spare static TLS,
                    # as when CUDA libraries have already used the reserve.
                    env=dict(
                        os.environ,
                        PYTHONPATH=os.pathsep.join(
                            [str(root), str(Path(scheduler_python).resolve())]
                        ),
                        GLIBC_TUNABLES="glibc.rtld.optional_static_tls=0",
                    ),
                    executable=tokenspeed_engine.host_binary(),
                    pass_fds=(result.fileno(),),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            finally:
                signal.signal(signal.SIGUSR1, previous)
            print(completed.stdout, completed.stderr)
            assert completed.returncode == 1, completed.returncode
            assert signal_count == 1, f"expected one SIGUSR1, got {signal_count}"
            assert "engine wheel bootstrap verified" in (
                completed.stdout + completed.stderr
            )
            result.seek(0)
            observed = json.load(result)
        assert observed == {
            "prefix": sys.prefix,
            "executable": sys.executable,
            "scheduler": "tokenspeed_engine._scheduler",
            "title": "tokenspeed::wheel-check",
        }, observed
    print("Installed extensions, scheduler and runtime Python verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_version")
    parser.add_argument("scheduler_python")
    args = parser.parse_args()
    main(args.expected_version, args.scheduler_python)
