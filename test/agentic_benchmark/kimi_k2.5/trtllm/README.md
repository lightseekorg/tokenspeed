# Agentic Benchmark — TRT-LLM

Sweep `trtllm-serve` against an agentic, multi-turn workload (SWE-Smith) at a fixed
set of attention/MoE parallelism layouts and report per-config throughput, latency,
and KV-cache hit rate.

Tested with TRT-LLM `1.3.0rc13`. Server listens on port **8001**.

## Layout

```
agentic_bench.sh        # main sweep: dataset prep -> for config in CONFIGS: launch, wait, bench, kill
configs/                # one yaml per parallelism layout (passed to trtllm-serve via --config)
collect_outputs.py      # parse a sweep into a flat CSV
outputs/<sweep_ts>/<config>/parallel_<P>_number_<N>/  # per-run evalscope artifacts
```

## Run a sweep

```bash
cd test/agentic_benchmark/trtllm
./agentic_bench.sh
```

The script (1) installs evalscope at the pinned commit, (2) builds the SWE-Smith
multi-turn dataset, (3) iterates each config in `CONFIGS=()`: launch server, poll
`/health` until ready, run `evalscope perf`, kill server, wait for the port to be
free, repeat. Aborts the whole sweep on the first failure (`set -e`).

To narrow the matrix, comment out entries in the `CONFIGS=()` array.

## Configs

Each `configs/*.yaml` is a `trtllm-serve --config` file specifying:

- `tensor_parallel_size`, `enable_attention_dp`, `moe_expert_parallel_size`
- `max_batch_size`, `cuda_graph_config.batch_sizes`
- `kv_cache_config.dtype`
- `moe_config.backend` (TRTLLM for TP-attn, CUTEDSL for DP-attn)
- `attention_dp_config` (DP variants only) — `enable_balance`,
  `enable_kv_cache_aware_routing`, batching/timeout iters
- `speculative_config` — Eagle3 with `lightseekorg/kimi-k2.5-eagle3-mla`

Naming: `attn_<X>_moe_<Y>` where `X ∈ {tp4,tp8,dp8}` and `Y ∈ {tp4,tp8,ep4,ep8}`.
World size = the number after `attn_(tp|dp)`.

## Collect results

```bash
python3 collect_outputs.py outputs/<sweep_ts> -o sweep.csv
```

Emits one row per (config, concurrency) with `Conc.`, `Latency (tps/user)`,
`Throughput (tps/gpu)`, `Approx Cache Hit`, `Decoded Tok/Iter`. `tps/gpu` divides
the system-wide `Total Throughput (tok/s)` by the GPU count inferred from the
config name; the other metrics come straight from `benchmark_summary.json`
(same numbers as evalscope's `performance_summary.txt` Request Metrics table).
