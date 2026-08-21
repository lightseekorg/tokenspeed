# Kimi-K3 agentic benchmark

Ranks K3 parallelism configs on the shared agentic multi-turn workload, with
the same dataset recipe, sweep ladder, and metric conventions as the
`kimi_k2.5` and `inkling` entries so results stay comparable across models.
Every parallelism change (TP8 baseline, attention-DP, future hybrid) lands
here and is compared on the same sweep.

## Usage

```
./agentic_bench.sh                # builds the dataset once, sweeps all configs
python3 collect_outputs.py outputs/<sweep_ts>   # one CSV comparison table
```

## Workload

swe_smith multi-turn conversations sized identically to `kimi_k2.5`: first
turn 50,000 tokens, subsequent turns 800, 10-15 turns (71 conversations build
with the Kimi-K3 tokenizer). Each turn's 500-token completion joins the next
turn's prompt, so the worst-case final prompt is ~50,000 + 14 x (800+500)
= 68.2K tokens plus chat-template overhead — `--max-model-len 80000` (the
kimi_k2.5 value) covers it
in every config, including the attention-DP per-rank cache budget. Fixed
`--max-tokens 500` with `ignore_eos` keeps decode segments equal-length across
configs; prefix caching (default-on) hit rate is reported.

## Metrics (from collect_outputs.py, house columns)

- Latency (tps/user) — decode speed per stream (1000 / TPOT)
- Throughput (tps/gpu)
- Approx Cache Hit, Decoded Tok/Iter

## Fairness rules

1. Each config runs its own best-known serve parameters (utilization, chunk
   size): this ranks deployment shapes, not controlled kernel ablations.
   TP8 = default 8192 chunks; attention-DP = 2048 chunks (the trtllm MoE
   workspace scales with gathered global tokens = chunk x 8 under DP, so
   larger chunks OOM at 0.95 utilization).
2. Speculative decoding is OFF in the parallelism matrix: the DP x
   speculative combination has no CI coverage here, so a spec-on matrix
   cannot be filled honestly. The TP8+DSpark config is the separate,
   clearly-labelled spec row (`_dspark` suffix, runs last) — it follows the
   CI DSpark gates' memory envelope (util 0.92, prefill graph and kvstore
   off), so compare it against TP8 spec-off qualitatively, not as a
   controlled ablation. "Decoded Tok/Iter" is only meaningful there
   (-1.0 elsewhere = not measured).
3. Warmup pass before each config's sweep; one sweep owns the machine.

## Configs

| config | notes |
|---|---|
| `attn_tp8_moe_ep8` | baseline |
| `attn_tp8_moe_ep8_dspark` | + DSpark speculative decoding (spec flags and memory envelope from the CI gates); the spec reference row, runs last |
| `attn_tp8_moe_tp8` | MoE TP variant (same capacity, different comm shape) |
| `attn_dp8_moe_ep8` | attention DP; requires #1152 (dynamic MLA packing, boot) AND #1185 (merge_state head tiling, runtime) — merge this PR after both. Must not pass `--dense-tp-size` (see config comments) |

tp4 variants are omitted: the K3-NVFP4 checkpoint does not fit a 4-GPU
partition.

Cross-model caveat: this directory pins evalscope acd09b44 (the inkling pin);
kimi_k2.5 pins 9d052ca0. Dataset recipe and sweep ladder are identical, but
metric implementations may differ between pins — compare across models with
that in mind.

## Future work

- **PD-disaggregation bench method**: a separate harness for decode-side
  evaluation (output-only throughput as the ranking metric, prefill
  interference isolated, high-concurrency sweep tail). The decode-focused
  metric changes deliberately do NOT live in this directory so the shared
  agentic columns stay comparable across models.
- Long-context dataset variant (200K+ first turn, TP8/hybrid only).
