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

"""Run an explicit CPU-only matrix serially, keeping every sample and failure."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def case(
    suite,
    transport,
    batch,
    prompt,
    steps,
    logprobs,
    top_k,
    period,
    delay,
    threads,
    timers,
):
    values = [
        suite,
        transport,
        str(batch),
        str(prompt),
        str(steps),
        logprobs,
        str(top_k),
        str(period),
        str(delay),
        str(threads),
    ]
    name = f"{suite}-{transport}-b{batch}-l{prompt}-s{steps}-{logprobs}-k{top_k}-p{period}-d{delay}-t{threads}-{timers}"
    return {"name": name, "values": values, "timers": timers}


def cases(matrix):
    result = []
    lengths = (8192, 32768, 131072, 262144, 524288, 1048576)
    if matrix == "full":
        for batch in (16, 128, 512):
            for length in lengths:
                for logprobs in ("output", "prompt"):
                    result.append(
                        case(
                            "stages",
                            "none",
                            batch,
                            length,
                            8,
                            logprobs,
                            0,
                            0,
                            0,
                            0,
                            "on",
                        )
                    )
                    result.append(
                        case(
                            "pipeline",
                            "ipc",
                            batch,
                            length,
                            128,
                            logprobs,
                            0,
                            1000,
                            0,
                            4,
                            "off",
                        )
                    )
            for length in (8192, 131072, 1048576):
                result.append(
                    case("stages", "none", batch, length, 8, "none", 0, 0, 0, 0, "on")
                )
        for batch, length, top_k in [
            (16, length, k) for length in (8192, 131072, 1048576) for k in (5, 20)
        ] + [(128, 8192, 5), (128, 131072, 5), (512, 8192, 5)]:
            result.append(
                case("stages", "none", batch, length, 1, "prompt", top_k, 0, 0, 0, "on")
            )
            result.append(
                case(
                    "pipeline", "ipc", batch, length, 1, "prompt", top_k, 0, 0, 4, "off"
                )
            )
        for batch in (16, 128, 512):
            result.append(
                case("pipeline", "ipc", batch, 8192, 1024, "output", 0, 0, 0, 4, "off")
            )
            result.append(
                case(
                    "pipeline", "ipc", batch, 8192, 128, "output", 0, 10000, 0, 4, "off"
                )
            )
        for batch in (128, 512):
            for delay in (0, 5000):
                result.append(
                    case(
                        "pipeline",
                        "ipc",
                        batch,
                        8192,
                        256,
                        "output",
                        0,
                        1000,
                        delay,
                        4,
                        "off",
                    )
                )
        for length, logprobs in (
            (131072, "output"),
            (131072, "prompt"),
            (1048576, "prompt"),
        ):
            result.append(
                case(
                    "pipeline", "tcp", 128, length, 128, logprobs, 0, 1000, 0, 4, "off"
                )
            )
        result.append(
            case("pipeline", "ipc", 512, 8192, 1024, "output", 0, 0, 0, 1, "off")
        )
    elif matrix == "anchors":
        for batch, length in ((128, 131072), (512, 1048576)):
            for mode in ("output", "prompt"):
                result.append(
                    case("stages", "none", batch, length, 8, mode, 0, 0, 0, 0, "on")
                )
        result.extend(
            [
                case("pipeline", "ipc", 512, 8192, 128, "output", 0, 1000, 0, 4, "off"),
                case(
                    "pipeline", "ipc", 128, 131072, 128, "prompt", 0, 1000, 0, 4, "off"
                ),
                case(
                    "pipeline", "ipc", 512, 1048576, 128, "prompt", 0, 1000, 0, 4, "off"
                ),
                case("stages", "none", 16, 131072, 1, "prompt", 20, 0, 0, 0, "on"),
            ]
        )
    elif matrix == "calibration":
        for batch, length, logprobs in (
            (16, 8192, "output"),
            (512, 8192, "output"),
            (128, 131072, "prompt"),
        ):
            for timers in ("off", "on"):
                result.append(
                    case(
                        "pipeline",
                        "ipc",
                        batch,
                        length,
                        256,
                        logprobs,
                        0,
                        0,
                        0,
                        4,
                        timers,
                    )
                )
    assert len({item["name"] for item in result}) == len(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--matrix", choices=("full", "anchors", "calibration"), required=True
    )
    parser.add_argument("--stage-cpu", required=True)
    parser.add_argument("--pipeline-cpus", required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-offset", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    assert (
        args.warmups >= 0
        and args.repeats > 0
        and args.rounds > 0
        and args.timeout_seconds > 0
    )
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    selected = cases(args.matrix)
    print(f"{len(selected)} configurations x {args.rounds} process groups", flush=True)
    for offset in range(args.rounds):
        group = args.round_offset + offset
        for item in (selected if group % 2 == 0 else list(reversed(selected))):
            name = f'{item["name"]}-r{group}'
            output = args.output / f"{name}.json"
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite prior samples: {output}")
            values = item["values"] + [
                str(args.warmups),
                str(args.repeats),
                item["timers"],
            ]
            affinity = args.stage_cpu if values[0] == "stages" else args.pipeline_cpus
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(args.output / f"{name}.time"),
                "taskset",
                "-c",
                affinity,
                str(binary),
                *values,
            ]
            record = {
                "name": name,
                "matrix": args.matrix,
                "group": group,
                "arguments": values,
                "affinity": affinity,
                "binary_sha256": digest,
                "load_before": os.getloadavg(),
            }
            print(f"Start {name}", flush=True)
            start = time.monotonic()
            with output.open("w") as stdout, (args.output / f"{name}.log").open(
                "w"
            ) as stderr:
                with subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, start_new_session=True
                ) as process:
                    try:
                        returncode = process.wait(timeout=args.timeout_seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        returncode = 124
            record.update(
                returncode=returncode,
                elapsed_seconds=time.monotonic() - start,
                load_after=os.getloadavg(),
            )
            if returncode == 0:
                data = json.loads(output.read_text())
                assert len(data["samples"]) == args.repeats
                expected = data["config"]["batch"] * data["config"]["steps"]
                assert all(
                    sample["checked_chunks"] == expected for sample in data["samples"]
                )
            with (args.output / "runs.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f'Finished {name}: code={returncode}, {record["elapsed_seconds"]:.1f}s',
                flush=True,
            )
            if returncode:
                raise RuntimeError(f"Failed process: {name}; inspect its retained log")


if __name__ == "__main__":
    main()
