"""Exercise an installed wheel and its embedded Python without GPU dependencies."""

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

def bootstrap(control_fd, outbox, protocol):
    # Exercise native stdlib modules and a separately loaded extension.
    import _ctypes
    import _socket
    import _sqlite3
    import _ssl
    import zlib
    import tokenspeed_scheduler_ext
    import tokenspeed_engine

    os.write(control_fd, json.dumps({
        "prefix": sys.prefix,
        "executable": sys.executable,
        "scheduler": tokenspeed_scheduler_ext.__name__,
    }).encode())
    raise RuntimeError("engine wheel bootstrap verified")
"""


def main(expected_version: str) -> None:
    installed_version = version("tokenspeed-engine")
    assert installed_version == expected_version, installed_version
    host = tokenspeed_engine.host_binary()
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
                        str(host),
                        "--python",
                        sys.executable,
                        "--control-fd",
                        str(result.fileno()),
                        "--enable-async-logging",
                        "false",
                    ],
                    env=dict(os.environ, PYTHONPATH=str(root)),
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
            "scheduler": "tokenspeed_scheduler_ext",
        }, observed
    print("Installed extension, embedded scheduler and runtime Python verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_version")
    main(parser.parse_args().expected_version)
