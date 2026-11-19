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

"""Bound a CI command and dump its children's native stacks before killing it.

Python faulthandler identifies a blocked extension call but cannot distinguish
the CUDA driver, NCCL and IPC waits inside it. The debugger is an explicit
command prefix, e.g. ``--debugger 'sudo -n gdb'`` on a CI runner with ptrace
access. A failed debugger attempt is reported without replacing the test's exit
status or removing its timeout. All children share a process group that is
killed on exit, including when the launcher fails before its workers finish.
"""

import argparse
import os
import shlex
import signal
import subprocess
import time

import psutil


def run(command, *, dump_after, timeout, debugger):
    if not 0 < dump_after < timeout:
        raise ValueError("require 0 < dump_after < timeout")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(command, start_new_session=True)
    try:
        try:
            return process.wait(timeout=dump_after)
        except subprocess.TimeoutExpired:
            pass
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in children:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            print(f"Native stacks for pid {child.pid}:", flush=True)
            try:
                subprocess.run(
                    [
                        *debugger,
                        "--batch",
                        "-ex",
                        "set pagination off",
                        "-ex",
                        "thread apply all bt 25",
                        "--pid",
                        str(child.pid),
                    ],
                    timeout=min(30, remaining),
                    check=True,
                )
            except (OSError, subprocess.SubprocessError) as error:
                print(f"Native stack dump failed: {error}", flush=True)
        try:
            return process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout} s", flush=True)
            return 124
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Command pid {process.pid} survived kill", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-after", type=float, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--debugger", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    raise SystemExit(
        run(
            command,
            dump_after=args.dump_after,
            timeout=args.timeout,
            debugger=shlex.split(args.debugger),
        )
    )
