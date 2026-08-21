# Kimi-K3 agentic decode-throughput benchmark

Ranks K3 parallelism configs by **decode throughput** on an agentic
multi-turn workload. TTFT is recorded but not ranked: the target regime is
long-context agentic serving where turn 2+ is prefix-hit + decode dominated.
Every parallelism change (TP8 baseline, attention-DP, future hybrid) lands
here and is compared on the same dataset, same sweep, same metric.

## Usage

```
./agentic_bench.sh                # builds the dataset once, sweeps all configs
python3 collect_outputs.py outputs/<sweep_ts>   # one CSV ranking table
```

## Workload

swe_smith multi-turn conversations sized for K3: first turn 50,000 tokens,
subsequent turns 800, 10-15 turns. Each turn's 500-token completion joins the
next turn's prompt, so the worst-case final prompt is ~50,000 + 14 x (800+500)
= 68.2K tokens plus chat-template overhead — `--max-model-len 81920` (the
inkling value) covers it in every config, including the attention-DP per-rank
cache budget (32 concurrent x ~69K spread over 8 ranks is ~276K tokens/rank,
well under the ~1.45M pool). Fixed `--max-tokens 500` with `ignore_eos` keeps
decode segments equal-length across configs; prefix caching (default-on) hit
rate is reported.

## Metrics (from collect_outputs.py)

- Latency (tps/user) — decode speed per stream (1000 / TPOT)
- **Output Throughput (tps/gpu) — the ranking metric** (generated tokens only;
  total throughput would be dominated by the 50K prompts)
- Approx Cache Hit, Decoded Tok/Iter

## Fairness rules

1. Each config runs its own best-known serve parameters (utilization, chunk
   size): this ranks deployment shapes, not controlled kernel ablations.
   TP8 = util 0.95 + 8192 chunks; attention-DP = util 0.95 + 2048 chunks
   (the trtllm MoE workspace scales with gathered global tokens = chunk x 8
   under DP, so larger chunks OOM at 0.95).
2. Speculative decoding is OFF in every config: the drafter family raises
   under dp>1, so a spec-on matrix cannot be filled. A TP8+DSpark row can be
   added later as a separate, clearly-labelled entry.
3. Warmup pass before every measured run; one sweep owns the machine.

## Configs

| config | notes |
|---|---|
| `attn_tp8_moe_ep8` | baseline |
| `attn_tp8_moe_tp8` | MoE TP variant (same capacity, different comm shape) |
| `attn_dp8_moe_ep8` | attention DP; requires the dynamic MLA packing fix (#1152) and must not pass `--dense-tp-size` (see config comments) |

tp4 variants are omitted: the K3-NVFP4 checkpoint does not fit a 4-GPU
partition. A long-context dataset variant (first turn 200K+, TP8/hybrid only)
is future work — it is the showcase workload for latent-partitioned hybrids.
