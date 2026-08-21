# Agentic Benchmark — TokenSpeed (Kimi-K3)

Sweep `ts serve` against the shared agentic multi-turn workload (SWE-Smith)
at a fixed set of K3 attention/MoE parallelism layouts and report per-config
throughput, latency, and KV-cache hit rate. Same dataset recipe, sweep
ladder, and metric conventions as `../kimi_k2.5/tokenspeed`.

Server listens on port **8000** (DP rendezvous on **4000**).

## Run a sweep

```bash
cd test/agentic_benchmark/kimi_k3/tokenspeed
./agentic_bench.sh                      # dataset prep -> per-config: launch, wait, bench, kill
python3 collect_outputs.py outputs/<sweep_ts>   # flat CSV, one row per (config, concurrency)
```

To narrow the matrix, comment out entries in the `CONFIGS=()` array.

## Workload sizing

First turn 50,000 tokens, +800/turn, 10–15 turns; each turn's 500-token
completion joins the next prompt, so final prompts reach ~68.2K tokens —
hence `--max-model-len 80000` in every config. 71 conversations build with
the K3 tokenizer (the script asserts >= 70 and rejects stale datasets built
with another tokenizer).

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
| `attn_dp8_moe_ep8` | attention DP; DP x DSpark is not yet validated on-machine (see config comments); never pass `--dense-tp-size`; chunked prefill pinned to 2048 (trtllm MoE workspace scales with chunk x 8 gathered tokens) |

**Merge order**: the DP row needs #1152 (dynamic MLA packing, boot) and
#1185 (merge_state head tiling, runtime) merged first.

Cross-model caveat: this directory pins evalscope `acd09b44` (kimi_k2.5 pins
`9d052ca0`); metric implementations may differ between pins.

To verify the parallelism actually applied, grep the server log:
```bash
grep -A6 "Parallelism configuration" /tmp/tokenspeed_server_<config>.log
```
