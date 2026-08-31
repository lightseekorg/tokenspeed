# Agentic Benchmark — TokenSpeed (Kimi-K3)

Sweep `ts serve` against the shared agentic multi-turn workload (SWE-Smith)
at a fixed set of K3 attention/MoE parallelism layouts and report per-config
throughput, latency, and KV-cache hit rate. Same dataset recipe, sweep
ladder, and metric conventions as `../kimi_k2.5/tokenspeed`.

Server listens on port **8000**.

## Run a sweep

```bash
cd test/agentic_benchmark/kimi_k3/tokenspeed
./agentic_bench.sh                      # dataset prep -> per-config: launch, wait, bench, kill
python3 collect_outputs.py outputs/<sweep_ts>   # flat CSV, one row per (config, concurrency)
```

To narrow the matrix, comment out entries in the `CONFIGS=()` array.

## Workload sizing

The script builds the dataset with the kimi_k2.5 recipe (first turn 50,000
tokens, +800/turn, 10-15 turns; 71 conversations build). Each turn's
500-token completion joins the next prompt, so final prompts reach ~68.2K
tokens — hence `--max-model-len 80000` in every config.

**Use ONE input file for every run you intend to compare.** The builder's
multi-worker selection is nondeterministic — rebuilding does not reproduce
the file (verified: two same-recipe builds share 0/71 conversations) — so
keep the first `agentic_dataset.json` and reuse it across runs and machines.
For text-identical comparisons with other models, drop in the frozen CI
artifact instead: `https://huggingface.co/datasets/lightseekorg/agentic-dataset`
(the file fetched by `test/ci/perf/kimi-k2.5-nvfp4-evalscope-agentic.yaml`
and the qwen3.5 agentic gates; K2.5-recipe, sizes verified identical under
the K3 tokenizer).

## Configs

`attn_<X>_moe_<Y>`, world size = the number after `attn_(tp|dp)`. All rows
run DSpark speculative decoding by default at util 0.92, kvstore on
(DSPARK+KVStore validated on-machine incl. the retract -> L2 -> restore
path). Note the prefill graph stays enabled: with DSpark this runs the
TP8 rows at ~99.5% memory (measured), so an OOM during a sweep should look
here first. tp4 layouts are omitted: the checkpoint does not fit 4 GPUs.

| config | notes |
|---|---|
| `attn_tp8_moe_ep8` | baseline |
| `attn_tp8_moe_tp8` | MoE TP variant |
Run-to-run noise: sampling is deliberately NOT pinned (matching the
kimi_k2.5/inkling convention — no seed, no greedy), so speculative
acceptance drifts between runs. Measured across identical-config runs:
TPOT +-7%, Decoded Tok/Iter +-13%. Differences smaller than ~10% need
repeated sweeps before ranking two configs.

Cross-model caveat: this directory pins evalscope `acd09b44` (kimi_k2.5 pins
`9d052ca0`). TPOT and Decoded Tok/Iter are verified identical between the two
pins (same formulas and aggregation; the summary keys merely dropped the
"Avg " prefix); the remaining columns are unverified between pins.

To verify the parallelism actually applied, grep the server log:
```bash
grep -A6 "Parallelism configuration" /tmp/tokenspeed_server_<config>.log
```
