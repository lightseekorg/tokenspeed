"""Exercise an installed wheel and its embedded Python without GPU dependencies."""

import json
import os
import subprocess
import sys
import tempfile
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

def notify_launcher_of_failure():
    pass
"""


def main() -> None:
    host = tokenspeed_engine.host_binary()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        engine = root / "tokenspeed" / "runtime" / "engine"
        engine.mkdir(parents=True)
        for package in (engine, engine.parent, engine.parent.parent):
            (package / "__init__.py").touch()
        (engine / "rust_host.py").write_text(BOOTSTRAP)
        # A real descriptor carries the result through the host's bootstrap.
        with (root / "result.json").open("w+") as result:
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
            print(completed.stdout, completed.stderr)
            assert completed.returncode == 1, completed.returncode
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
    main()
