"""Split the rocJITsu test files into balanced shards.

Prints one line per shard holding the pytest path arguments for that shard. The
last shard is handed the whole test root minus the files assigned elsewhere, so
a test file with no weight entry is still executed.
"""

import sys
from pathlib import Path


def read_weights(path):
    weights = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seconds, test_file = line.split(None, 1)
        weights[test_file.strip()] = float(seconds)
    if not weights:
        raise SystemExit(f"no weight entries found in {path}")
    return weights


def plan(weights, shards):
    assigned = [[] for _ in range(shards)]
    loads = [0.0] * shards
    # Longest-processing-time-first: the slowest file lands on an empty shard,
    # which keeps the makespan within one file of optimal.
    for test_file, seconds in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        target = loads.index(min(loads))
        assigned[target].append(test_file)
        loads[target] += seconds
    return assigned


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: mi450_sim_shard_plan.py <shards> <weights> <test-root>"
        )
    shards = int(sys.argv[1])
    weights = read_weights(sys.argv[2])
    test_root = sys.argv[3]

    if shards < 1:
        raise SystemExit(f"shard count must be positive, got {shards}")

    # The last shard collects the test root minus the files handed to the other
    # shards, so a weighted file outside the root would silently never run.
    prefix = test_root.rstrip("/") + "/"
    stray = sorted(f for f in weights if not f.startswith(prefix))
    if stray:
        raise SystemExit(
            f"{len(stray)} weighted test files are outside {test_root}, "
            f"starting with {stray[0]}"
        )

    if shards == 1:
        print(test_root)
        return

    # An empty shard would hand pytest no path at all and collect the whole
    # tree, so drop it and let the caller run fewer shards than requested.
    assigned = [shard for shard in plan(weights, shards) if shard]
    explicit = sorted(f for shard in assigned[:-1] for f in shard)
    for shard in assigned[:-1]:
        print(" ".join(sorted(shard)))
    print(" ".join([test_root] + [f"--ignore={f}" for f in explicit]))


if __name__ == "__main__":
    main()
