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

"""Paired rank-MAX statistics, retained from historical qualification."""

import math
import random


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 < q <= 1:
        raise ValueError("nonempty sample and q in (0,1] required")
    return sorted(values)[math.ceil(len(values) * q) - 1]


def paired_statistics(baseline: list[float], candidate: list[float]) -> dict:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples must have the same nonzero length")
    if any(not math.isfinite(x) or x <= 0 for x in baseline + candidate):
        raise ValueError("latencies must be finite and positive")
    speedups = [a / b for a, b in zip(baseline, candidate)]
    logs = [math.log(x) for x in speedups]
    rng = random.Random(19073)
    means = [
        math.exp(sum(rng.choice(logs) for _ in logs) / len(logs)) for _ in range(4000)
    ]
    return {
        "baseline_raw_us": baseline,
        "candidate_raw_us": candidate,
        "paired_speedups": speedups,
        "baseline_p50_us": percentile(baseline, 0.5),
        "candidate_p50_us": percentile(candidate, 0.5),
        "baseline_p90_us": percentile(baseline, 0.9),
        "candidate_p90_us": percentile(candidate, 0.9),
        "p50_speedup": percentile(baseline, 0.5) / percentile(candidate, 0.5),
        "p90_speedup": percentile(baseline, 0.9) / percentile(candidate, 0.9),
        "paired_geomean": math.exp(sum(logs) / len(logs)),
        "paired_ci95": [percentile(means, 0.025), percentile(means, 0.975)],
    }


def performance_gate(stats: dict, minimum: float) -> bool:
    """A point passes only with median gain, no p90 loss and positive CI."""
    return (
        stats["p50_speedup"] >= minimum
        and stats["p90_speedup"] >= 1.0
        and stats["paired_ci95"][0] > 1.0
    )
