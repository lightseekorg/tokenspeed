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

"""Collect direct-I/O measurements through the native Disk executor."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from test.cache_gds_calibration.data import positive


def collect_disk(args):
    if (
        args.repeats < 1
        or min(args.block_bytes) < 4096
        or any(x % 4096 for x in args.block_bytes)
    ):
        raise ValueError("positive repeats and 4096-byte aligned block sizes required")
    sizes = list(args.block_bytes)
    rng = random.Random(args.seed)
    with Path(args.output).open("x") as output:
        for repeat in range(args.repeats):
            rng.shuffle(sizes)
            for size in sizes:
                command = [
                    str(Path(args.bench).resolve()),
                    str(Path(args.directory).resolve()),
                    str(args.file_bytes),
                    str(size),
                    str(args.inflight),
                    str(args.queue_depth),
                    str(args.max_io_bytes),
                    args.backend,
                    str(args.warmup_seconds),
                    str(args.seconds),
                ]
                try:
                    result = subprocess.run(
                        command,
                        check=True,
                        text=True,
                        capture_output=True,
                        timeout=args.timeout,
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"native Disk measurement failed: {error.stderr}"
                    ) from error
                raw = json.loads(result.stdout)
                if (
                    not raw["direct_io"]
                    or raw["block_bytes"] != size
                    or raw["backend"] != args.backend
                    or raw["inflight"] != args.inflight
                    or raw["queue_depth"] != args.queue_depth
                    or raw["max_io_bytes"] != args.max_io_bytes
                ):
                    raise ValueError(
                        "benchmark returned incompatible geometry/I/O mode"
                    )
                elapsed = (
                    raw["mean_latency_s"]
                    if args.time_basis == "job-latency"
                    else raw["elapsed_s"] / raw["completed_blocks"]
                )
                row = {
                    "kind": "disk",
                    "deployment": args.deployment,
                    "split": args.split,
                    "read_blocks": 1,
                    "aligned_bytes": size,
                    "seconds": positive(elapsed),
                    "time_basis": args.time_basis,
                    "repeat": repeat,
                    "seed": args.seed,
                    "native": raw,
                }
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                print(
                    f"block_bytes={size} seconds={elapsed:.8g} repeat={repeat}",
                    flush=True,
                )
