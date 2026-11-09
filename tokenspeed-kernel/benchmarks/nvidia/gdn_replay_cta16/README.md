# Bounded GDN replay experiments

`kernel.py` saves a grouped replay implementation with a 16-CTA launcher bound.
The independent opt-in registry entry retains its 24-CTA baseline. All launch
settings are explicit: CTA budget, state-buffer count, compute-warps, and
register limit. `candidate.json` records the selected benchmark settings.

The baseline registry entry is `gluon_nvidia_gdn_replay_commit`, below the
portable kernel's selection priority. Select it explicitly with
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_REPLAY_COMMIT=gluon_nvidia_gdn_replay_commit`.
A bounded commit launch alone does not create draft overlap; the independent
captured-graph probe supplies the stream ordering and joins.

## Accepted-state computation

For BF16 four-token windows with three value heads per key head, each work
item groups those three heads. Four producer warps prepare normalized keys,
gates, Gram data, and asynchronous state copies in two shared-memory slots.
Eight compute warps consume the prepared work. Shared full/empty barriers
account for every participating thread.

For decay products `A[t]`, the scaled state is `R[t] = S[t] / A[t]`:

```text
q[t] = beta[t] * (v[t] / A[t] - k[t]^T R[t-1])
R[t] = R[t-1] + k[t] q[t]^T
S[accepted] = A[accepted] * R[accepted]
```

The selected path projects the initial state onto the accepted keys, solves
the small triangular recurrence in FP32, and applies the final rank updates.
It reads and computes only the accepted prefix. Short prefixes and very small
cumulative decays retain the ordinary recurrence. Other supported dtypes and
window buckets use the general replay implementation. Conversion to the state
dtype occurs only at the final store; each work item owns complete value heads.

`kernel_stream.py` preserves a separate streamed prototype with one, two, or
three state stages and two token-input slots per stage. Its benchmark setting
selects three stages, 16 compute warps, and a 96-register limit. Three stages
support FP32 state only through a four-token window; one/two stages support
longer windows. Both launchers reject budgets above 16. The CTA sweep below
explicitly selects the JIT entry's grid to compare larger budgets.

## Reproduction

Activate the repository venv and run from the repository root. The isolated
sweep changes only the grid of the same compiled kernel. It uses synthetic
state and payloads, checks the original recurrence oracle, verifies unchanged
source rows, and interleaves timing order. `full4` accepts four tokens in every
request; `mixed` repeats accepted lengths 1/2/3/4.

```bash
PYTHONPATH=tokenspeed-kernel/python:tokenspeed-kernel/test python \
  tokenspeed-kernel/benchmarks/nvidia/gdn_replay_cta16/bench_cta_sweep.py \
  --batches 32 64 --ctas 16 24 32 --layers 36 --patterns full4 mixed \
  --blocks 11 --iterations 30 --gpu 0 --seed 731 \
  --output /tmp/gdn-cta-sweep

PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/benchmarks/nvidia/gdn_replay_cta16/bench.py \
  --action bench --names baseline scaled stream --batches 8 32 \
  --patterns accept1 accept3 full4 mixed --layers 36 \
  --blocks 7 --iterations 30 --gpu 0 --output /tmp/gdn-candidates.json

PYTHONPATH=tokenspeed-kernel/python:tokenspeed-kernel/test pytest -q \
  tokenspeed-kernel/test/nvidia/ops/test_gdn_replay_pipeline.py \
  tokenspeed-kernel/test/nvidia/test_gdn_commit_parameter_analysis.py
```

The grouped configuration uses 384 threads, 216,196 bytes of shared memory,
and one resident CTA per SM on the measured architecture. Compiler versions
can change register allocation; record register/spill counts with timing data.
The synthetic benchmark uses H4/HV12/K128/V128 and BF16 payload/state. Its layer
count is an independent fixture parameter. CUDA Graph replay times are
isolated kernel measurements; the graph-overlap probe measures scheduling
and resource contention separately.

The GPU tests compare BF16/FP16/FP32 state, several token windows, mixed and
clamped acceptance, rejected NaNs, extreme gates, odd strides, graph refresh,
and runtime-scalar compilation reuse. Tolerances remain BF16 rtol/atol
`2e-2/1e-3`, FP16 `1e-3/1e-6`, and FP32 `1e-5/1e-6`; untouched rows remain
bitwise unchanged. `parameter_analysis.py` additionally audits real JIT-cache
growth at the benchmark-to-kernel boundary. Keep generated measurements,
profiler reports, and tuning history outside version control.
