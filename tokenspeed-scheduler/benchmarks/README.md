# Prefill compute-budget reproduction

The benchmark drives real C++ schedulers with one full-attention cache group
and one recurrent-state group. Each DP rank receives the same total work in
a different FIFO order. Prefixes are warmed through real scheduler completion
and publication before the measured queue starts. Every configuration must
compute exactly the same new tokens, or the benchmark fails.

## Cost-model experiment (CPU only)

Install the scheduler from this checkout, then run:

```bash
python tokenspeed-scheduler/benchmarks/prefill_compute.py \
  --mode model --ranks 4 --repeats 2 \
  --coefficients 40 0.01 5000 --output model.json
```

These coefficients are illustrative assumptions, not a measured GPU profile.
The default time budget equals the model's cost for an 8192-token chunk with
a 16384-token prefix; `--budget-ms` overrides it. The report includes the
estimated sum of per-round slowest-rank costs, useful compute fraction,
round count and measured CPU planner time. A lower max/mean alone is not a
performance win: extra rounds and per-chunk overhead also count.

## Synchronized GPU component experiment

Use exclusively allocated SM100/SM103/SM107 GPUs with this checkout's
`tokenspeed-kernel`, `tokenspeed-mla` and scheduler installed:

```bash
torchrun --standalone --nproc-per-node=4 \
  tokenspeed-scheduler/benchmarks/prefill_compute.py \
  --mode cuda --repeats 2 --iterations 5 --output gpu.json
```

This uses K3-shaped explicit-QKV prefill (96 heads, QK dimension 192, V
dimension 128, BF16). Dimensions can be passed explicitly for another model.
It checks split/unsplit attention outputs, measures calibration shapes on
every GPU, fits non-negative coefficients, and warms each scheduled shape
before timing. Each actual DP round executes the local scheduler's attention
shape and a common all-reduce; an idle rank joins the same collective. The
report contains raw calibration samples, fit error, kernel-time imbalance and
median synchronized wall time. The same token limit is used in both arms.

This is a GPU **attention component** benchmark, not K3 serving: projection,
KV expansion/transfers, KDA, MoE and HTTP are excluded. Scheduler plans are
prepared before GPU timing; CPU planner time is reported separately. Confirm
whole-model correctness, throughput and TTFT under the actual deployment
before drawing a serving-performance conclusion. Keep uniformly sized and
low-concurrency workloads as regressions, and repeat trials with both arm
orders when measuring small differences.
