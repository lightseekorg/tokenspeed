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

"""Run an explicit completion matrix sequentially and preserve raw samples.

Use one output directory per binary revision. Configuration is passed unchanged
into the Rust workload; a failed case is not recorded as a completed result.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--suite", choices=("stages", "loop"), required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-set", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    cases = json.loads(args.matrix.read_text())
    metadata = {
        "revision": args.revision,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "suite": args.suite,
        "cpu_set": args.cpu_set,
        "allocator_settings": {
            name: os.environ[name]
            for name in ("MIMALLOC_PURGE_DELAY",)
            if name in os.environ
        },
        "cases": cases,
    }
    meta_path = args.output / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("output directory belongs to a different binary or matrix")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    prefix = "TS_FINISH" if args.suite == "stages" else "TS_LOOP"
    test = (
        "benchmark_finish_burst"
        if args.suite == "stages"
        else "finish_event_loop::benchmark_finish_event_loop"
    )
    for case in cases:
        name, config = case["name"], case["config"]
        if not name or Path(name).name != name:
            raise ValueError(f"invalid case name: {name}")
        output = args.output / f"{name}.json"
        if output.exists():
            result = json.loads(output.read_text())
            if (
                result["config"] != config
                or len(result["samples"]) != config["repeats"]
            ):
                raise ValueError(f"incomplete or mismatched result: {output}")
            continue
        env = dict(os.environ)
        env[f"{prefix}_CONFIG"] = json.dumps(config)
        env[f"{prefix}_OUTPUT"] = str(output.resolve())
        command = [
            "/usr/bin/time",
            "-f",
            "max_rss_kib=%M\nelapsed_seconds=%e",
            "-o",
            str(args.output / f"{name}.usage"),
            "taskset",
            "-c",
            args.cpu_set,
            str(binary),
            "--ignored",
            "--exact",
            test,
        ]
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=log,
                check=True,
                timeout=args.timeout_seconds,
            )
        result = json.loads(output.read_text())
        if result["config"] != config or len(result["samples"]) != config["repeats"]:
            raise ValueError(f"workload returned incomplete samples: {name}")
        if args.suite == "stages":
            summary = {
                key: round(
                    statistics.median(s[key]["wall_ms"] for s in result["samples"]), 3
                )
                for key in (
                    "commit",
                    "scheduler_finish",
                    "cleanup_plan",
                    "events",
                    "kv_encode",
                )
            }
        else:
            summary = {
                key: round(statistics.median(s[key] for s in result["samples"]), 3)
                for key in ("control_done_ms", "total_wall_ms", "process_cpu_ms")
            }
        print(f"{name}: {summary}", flush=True)


if __name__ == "__main__":
    main()
