# TEMPORARY, remove before the PR: forwards the old path to
# benchmarks/amd/gfx950/muse_spark/monokernel/draft_layer_bench.py until the gate and
# harness scripts that run this path move to the new one.
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        *[".."] * 5,
        "benchmarks/amd/gfx950/muse_spark/monokernel"
    ),
)
from draft_layer_bench import *  # noqa: E402,F401,F403

if __name__ == "__main__":
    main()  # noqa: F405
