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

"""Aggregate formal waves and per-rank memory without mixing profiler timings."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize_result(path):
    result = json.loads(path.read_text())
    assert result.get("completed"), f"Incomplete experiment: {path}"
    args = result["arguments"]
    root = Path(args["work_dir"])
    tp = result["config"]["world_size"]
    groups = collections.defaultdict(list)
    for trial in result["trials"]:
        if not trial["warmup"]:
            groups[trial["input_tokens"], trial["concurrency"]].append(trial)
    output = []
    for (length, concurrency), trials in sorted(groups.items()):
        metrics = {}
        for key in ("ttft_ms", "tpot_ms", "output_tokens_per_second", "wall_seconds"):
            values = [t[key] for t in trials]
            metrics[key] = dict(
                median=statistics.median(values),
                min=min(values),
                max=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else None,
            )
        ranks = []
        for rank in range(tp):
            rank_root = root / f"rank-{rank}" if tp > 1 else root
            records = []
            for t in trials:
                step_file = rank_root / t["id"] / "steps.jsonl"
                records.extend(
                    json.loads(s) for s in step_file.read_text().splitlines()
                )
            assert records
            ranks.append(
                dict(
                    rank=rank,
                    peak_allocated_bytes=max(x["peak_allocated"] for x in records),
                    peak_reserved_bytes=max(x["peak_reserved"] for x in records),
                    decode_batch_histogram=dict(
                        collections.Counter(
                            x["batch_size"] for x in records if x["mode"] == "DECODE"
                        )
                    ),
                    observed_forward_count=len(records),
                )
            )
        output.append(
            dict(
                source=str(path),
                model=Path(args["model"]).name,
                tp=tp,
                input_tokens=length,
                concurrency=concurrency,
                output_tokens=args["output_tokens"],
                formal_waves=len(trials),
                metrics=metrics,
                ranks=ranks,
            )
        )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    records = [entry for path in args.inputs for entry in summarize_result(path)]
    args.output.write_text(
        json.dumps(
            dict(
                method="Median of formal concurrent request waves; throughput uses total output tokens / wave wall time; allocator peaks per rank",
                configurations=records,
            ),
            indent=2,
        )
        + "\n"
    )
    for r in records:
        print(
            r["model"],
            r["tp"],
            r["input_tokens"],
            r["concurrency"],
            {k: round(v["median"], 3) for k, v in r["metrics"].items()},
        )


if __name__ == "__main__":
    main()
