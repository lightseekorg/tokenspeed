# Single-GPU prefill compute-budget measurements

Dynamic chunks can reduce the critical path when ranks have equal total work
in different FIFO orders. They do not redistribute work assigned persistently
to one rank, and splitting already balanced work can add cost. Keep the option
experimental and opt-in until the target deployment passes a serving A/B.

## Measurement scope

[Raw measurements and plans](prefill_compute_b300.json) record scheduler
revision `dd07cd323a87eda79d6ffbe370f76a87d803b647`. After merging main
`57c4423a2`, all 21 baseline/candidate plan pairs were regenerated using the
rebuilt scheduler and matched the recorded plans and derived metrics exactly.
The GPU measurements were not repeated after the merge.

The measurement used one NVIDIA B300 SXM6 AC (SM103), with BF16 explicit QKV,
96 heads, QK dimension 192, and V dimension 128. The environment was PyTorch
2.14.0+cu130, CUDA 13.0, tokenspeed-kernel
0.1.4.dev20261025+git8c066202, and tokenspeed-mla 2.0.4. This was an existing
kernel environment, not a rebuild of the merged branch's kernel package.

For each shape, three trials each performed three warmup calls followed by
20 CUDA-event measurements. The reported duration is the median of the three
trial medians. Inputs were reused after warmup. The 76 scheduled shapes had
an average max/min trial spread of 0.87%, with a maximum of 4.36%; these are
observed ranges, not confidence intervals. The wrapper was `CUDAAttention`
from the recorded revision's `benchmarks/prefill_compute.py`. Its split-output
check passed for a 512-query/2048-prefix input split into two 256-query calls,
with `rtol=atol=0.02`; this is not full-model correctness validation.

No synchronized multi-GPU or full-model A/B was run. Projection, latent-cache
expansion, KDA, MoE, communication, HTTP, and online arrivals are excluded.
The single-GPU attention measurements below must not be reported as measured
distributed throughput or TTFT gains.

## Equal tokens need not mean equal attention time

| Query tokens | Cached-prefix tokens | Measured attention time |
| ---: | ---: | ---: |
| 8192 | 0 | 1.728 ms |
| 8192 | 16384 | 7.633 ms |
| 8192 | 65536 | 25.853 ms |

The approximately 15-fold component-time difference motivates prefix-aware
sizing. It does not establish the same ratio for full-model work.

## Derived four-rank critical paths

Real C++ schedulers generated the plans, with one request per batch and an
8192-token cap. Prefixes were warmed before the queue began. Every plan pair
computes identical new tokens on each rank. For each shape, the measured
single-GPU time replaces the predicted cost; the estimate is the sum of the
slowest rank's time per round, padding an idle rank with zero.

The scheduler used the current three-term nonnegative fit, trained on
`q={512,2048,8192}` and `c={0,16384,65536}`:

```text
cost_ns(q, c) = 315023.2079308866 + 0*q + 0.04492530978113779*q*(c+(q+1)/2)
budget = cost_ns(8192, 16384) = 7.852423 ms
```

| Workload | Baseline/candidate rounds | Baseline attention path | Candidate attention path | Derived speedup |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 8 / 8 | 13.805 ms | 13.805 ms | 1.000x |
| Mixed context lengths | 30 / 46 | 476.680 ms | 362.023 ms | 1.317x |
| Rotated prefix lengths | 8 / 16 | 208.922 ms | 136.012 ms | 1.536x |
| Persistent rank skew | 8 / 32 | 208.922 ms | 238.447 ms | 0.876x |
| One active rank | 8 / 32 | 208.922 ms | 238.447 ms | 0.876x |
| Identical long-prefix work on all ranks | 8 / 32 | 208.922 ms | 238.447 ms | 0.876x |
| Identical alternating short/long prefixes | 8 / 20 | 111.363 ms | 126.126 ms | 0.883x |

Workload definitions use `(cached_prefix_tokens, new_tokens)`:

- Uniform, mixed contexts, and rotated prefixes are `workloads(4, 2)` from
  the benchmark. Each rank has the same work, rotated in FIFO order.
- Persistent rank skew: rank 0 has eight `(65536, 8192)` requests; the other
  three ranks each have eight `(0, 8192)` requests.
- One active rank: rank 0 has eight `(65536, 8192)` requests; others are idle.
- Identical long-prefix work: every rank has eight `(65536, 8192)` requests.
- Identical alternating prefixes: every rank repeats
  `[(0, 8192), (65536, 8192)]` four times in the same order.

Persistent skew and already balanced long-prefix work take 14.13% longer in
this estimate. Halving the budget improves the mixed-context estimate to
1.397x, but makes persistent rank skew take 35.31% longer. A smaller budget
or lower round imbalance alone is not a performance win.

Extra full-model and communication work can also consume the positive cases'
margin. Under a hypothetical identical fixed overhead per round, the baseline
budget's mixed-context benefit disappears at 7.17 ms/round, and the rotated
prefix benefit at 9.11 ms/round. These are sensitivity thresholds, not
measured overheads; real overhead can depend on the shape.

## Calibration limitations

The current fit's training median relative error is 8.30%, but its largest
error on six held-out calibration shapes is 668.80%. The held-out queries
are 128 and 4096 at the same three prefix lengths. Absolute least squares
can hide large relative errors on short requests behind small median errors.

The separate scheduled-shape measurement contains 73 shapes absent from
training. An exploratory extra prefix term, `d*c`, improved their prediction
errors, but was not used to produce the scheduling results above:

| Model / fitting objective | Calibration held-out median / max error | Scheduled-shape held-out P95 / max error |
| --- | ---: | ---: |
| Current three terms / absolute residual | 26.53% / 668.80% | 21.84% / 72.44% |
| Current three terms / relative residual | 21.72% / 59.95% | 18.28% / 94.39% |
| Extra prefix term / absolute residual | 5.92% / 30.96% | 10.51% / 23.58% |
| Extra prefix term / relative residual | 2.59% / 14.59% | 12.67% / 31.93% |

These small, single-device samples do not establish a generally better
runtime formula. The relative-residual objective is not best on every set.
Calibration should report length-stratified errors and held-out P95/max
errors, and include the whole rank-local work being balanced.

## Recompute the reported estimates

The JSON records all measured shapes and all plans for budget multipliers
0.5, 1, and 2. From this directory, the following recomputes the baseline
budget's estimates without a GPU:

```python
import itertools
import json
from pathlib import Path

data = json.loads(Path("prefill_compute_b300.json").read_text())
times = {
    (row["q"], row["c"]): row["median_ns"]
    for row in data["scheduled_shape_measurements"]["samples"]
}


def critical_path(plans):
    costs = [[times[tuple(shape)] for shape in rank] for rank in plans]
    return sum(max(row) for row in itertools.zip_longest(*costs, fillvalue=0))


for row in data["analysis"]["evaluations"]:
    if row["budget_multiplier"] == 1:
        print(row["case"], critical_path(row["baseline"]) / critical_path(row["candidate"]))
```

For the next validation stage, retain these negative controls and measure
actual synchronized wall time before a full serving A/B. Persistent work
skew needs request assignment that accounts for remaining work and cache
accessibility; rank-local chunk sizing alone cannot fix that assignment.
