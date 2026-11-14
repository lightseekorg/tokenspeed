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
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from multiprocessing import Pipe
from pathlib import Path


def preload_cases(*, prepend_hook: bool):
    hook = "/test/torch_memory_saver_hook_mode_preload.so"
    explicit = "/custom/torch_memory_saver_hook_mode_preload_cu12.abi3.so"
    extra, other = "/test/context_only.so", "/test/other_hook.so"
    cases = [
        ("no_override", None, None),
        ("ordinary_preload", [other], None),
        (
            "explicit_hook",
            [other, explicit, "/test/last.so"],
            [other, explicit, "/test/last.so", extra],
        ),
        (
            "legacy_cpp_hook",
            [other, "/custom/torch_memory_saver_cpp.some-postfix.so"],
            [other, "/custom/torch_memory_saver_cpp.some-postfix.so", extra],
        ),
        ("directory_name", ["/test/torch_memory_saver/libhelper.so"], None),
        ("other_library", ["/test/torch_memory_saver_helper.so"], None),
        ("cpp_helper", ["/test/torch_memory_saver_cpp_helper.so"], None),
        ("similar_name", ["/test/not_torch_memory_saver_hook_mode_preload.so"], None),
        (
            "repeated_explicit_hook",
            [explicit, other, explicit],
            [explicit, other, extra],
        ),
        ("repeated_context_hook", None, None),
    ]
    for name, inherited, override in cases:
        configured = [hook, extra, *(inherited if prepend_hook and inherited else [])]
        expected = override if override is not None else configured.copy()
        if name == "repeated_context_hook":
            configured.append(hook)
        for separator in (":", " \t"):
            yield (
                f"{name}-{'prepend' if prepend_hook else 'overwrite'}-{separator!r}",
                separator.join(inherited) if inherited is not None else None,
                separator.join(configured),
                ":".join(expected),
            )


def conflicting_preload_cases():
    first = "/one/torch_memory_saver_hook_mode_preload.so"
    second = "/two/torch_memory_saver_hook_mode_preload_cu12.abi3.so"
    for separator in (":", " \t"):
        yield f"parent-{separator!r}", separator.join([first, second]), first
        yield f"context-{separator!r}", None, separator.join([first, second])


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
                and node.name
                in {"HostProcess", "_merge_memory_saver_preload", "start_rust_host"}
            ],
        ],
        type_ignores=[],
    )
    launcher = {
        "Pipe": Pipe,
        "functools": functools,
        "os": os,
        "re": re,
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
            for name, inherited, prepared, expected in preload_cases(
                prepend_hook=prepend_hook
            ):
                if inherited is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = inherited
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
                    assert re.split(r"[:\s]+", observed[0]) == expected.split(
                        ":"
                    ), observed
                    assert observed[1] == "/test/runtime", observed
                    process.join(timeout=5)
                    assert process.exitcode == 0
                    print(f"passed: {name}")
                finally:
                    reader.close()
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            for name, inherited, prepared in conflicting_preload_cases():
                if inherited is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = inherited
                before = len(list(Path("/dev/fd").iterdir()))
                try:
                    launcher["start_rust_host"](
                        str(binary),
                        server_args,
                        None,
                        environment({"LD_PRELOAD": prepared}),
                        None,
                        False,
                    )
                except ValueError as error:
                    assert "multiple memory-saver preload hooks" in str(error)
                else:
                    raise AssertionError("conflicting hooks were accepted")
                assert os.environ.get("LD_PRELOAD") == inherited
                assert len(list(Path("/dev/fd").iterdir())) == before
                print(f"passed: conflicting-{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rust_host", type=Path)
    parser.add_argument("--context", choices=("overwrite", "prepend"), required=True)
    args = parser.parse_args()
    run(args.rust_host, prepend_hook=args.context == "prepend")
