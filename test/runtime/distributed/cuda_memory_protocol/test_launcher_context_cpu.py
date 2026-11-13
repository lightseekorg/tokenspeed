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

"""Run the actual launcher's environment checks without GPU runtime imports.

Only its process-spawn definitions are loaded; the child and control pipe are
real. Full-runtime pytest remains in test/runtime/test_rust_host.py.
"""

import argparse
import ast
import functools
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from multiprocessing import Pipe
from pathlib import Path


def run(source: Path, *, prepend_hook: bool):
    definitions = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            *[
                node
                for node in ast.parse(source.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"HostProcess", "start_rust_host"}
            ],
        ],
        type_ignores=[],
    )
    launcher = {
        "Pipe": Pipe,
        "functools": functools,
        "os": os,
        "rank_process_title": lambda _mapping: "tokenspeed::tp0",
        "subprocess": subprocess,
        "sys": sys,
    }
    exec(compile(ast.fix_missing_locations(definitions), str(source), "exec"), launcher)

    @contextmanager
    def environment(values):
        original = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    hook = "/test/torch_memory_saver_hook_mode_preload.so"
    other = "/test/other_hook.so"
    override = f"/test/torch_memory_saver_custom.so:{other}"
    # The full ServerArgs imports the GPU stack. This fixture supplies only
    # the launch arguments; it does not replace any process-spawn behavior.
    server_args = argparse.Namespace(
        mapping=argparse.Namespace(rank=0),
        log_level="info",
        disaggregation_mode="null",
        enable_async_logging=False,
    )
    with environment({"LD_PRELOAD": os.environ.get("LD_PRELOAD", "")}):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "ts-host"
            binary.write_text(
                f"#!{sys.executable}\n"
                "import os, sys\n"
                "from multiprocessing.connection import Connection\n"
                "connection = Connection(int(sys.argv[sys.argv.index('--control-fd') + 1]))\n"
                "connection.recv_bytes()\n"
                "connection.send((os.environ.get('LD_PRELOAD'), os.environ.get('LD_LIBRARY_PATH')))\n"
            )
            binary.chmod(0o700)
            for inherited in (None, other, override):
                if inherited is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = inherited
                prepared = f"{hook}:{inherited}" if prepend_hook and inherited else hook
                parent_library_path = os.environ.get("LD_LIBRARY_PATH")
                process, reader = launcher["start_rust_host"](
                    str(binary),
                    server_args,
                    None,
                    environment(
                        {"LD_PRELOAD": prepared, "LD_LIBRARY_PATH": "/test/runtime"}
                    ),
                    None,
                    False,
                )
                try:
                    assert reader.poll(10), "child did not report its environment"
                    observed = reader.recv()
                    assert os.environ.get("LD_PRELOAD") == inherited
                    assert os.environ.get("LD_LIBRARY_PATH") == parent_library_path
                    expected = override if inherited == override else prepared
                    assert observed == (expected, "/test/runtime"), observed
                    process.join(timeout=5)
                    assert process.exitcode == 0
                    print(f"passed: inherited={inherited!r}, prepend={prepend_hook}")
                finally:
                    reader.close()
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rust_host", type=Path)
    parser.add_argument("--context", choices=("overwrite", "prepend"), required=True)
    args = parser.parse_args()
    run(args.rust_host, prepend_hook=args.context == "prepend")
