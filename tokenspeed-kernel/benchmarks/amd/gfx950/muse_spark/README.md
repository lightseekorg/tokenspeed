# Muse Spark kernel benchmarks on gfx950

These cases use the shapes of Muse Spark as TokenSpeed serves it in its MI355X
AgentX submission to InferenceX (`musespark-fp4-mi355x-tokenspeed-agentic-dflash`,
recipe `benchmarks/single_node/srt-slurm-recipes/musespark/tokenspeed/mi355x-fp4/agentic.yaml`).
The submission serves `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` (MXFP4 experts,
BF16 experts in the first and last layers) with the DFlash2 draft, an FP8 E4M3
KV cache without checkpoint scales and the native 1M-token context. It runs two
layouts:

- TP8: one worker on eight GPUs, attention and MoE TP8 (EP1), the decode
  monokernel on, 8192-token prefill chunks, concurrency 1 to 28.
- TP4: attention TP4 with EP4 MoE (64 experts on each GPU), the regular decode
  path, the QKV projection in FP8 at 33 to 128 decode rows
  (`--dense-fp8-small-m qkv`), 24576-token prefill chunks, concurrency 28 to 40.

## Rows per step

A decode step verifies every running request's seven drafts plus its current
token, 8 rows a request. Each concurrency point caps the running requests
(`max-num-seqs`); decode graphs are captured per request count, so 7 requests
at TP8 pad to 8:

| Layout | Concurrency      | Requests | Decode rows                                       |
| ------ | ---------------: | -------: | ------------------------------------------------- |
| TP8    | 1, 6             | 1        | 8, in the decode monokernel                       |
| TP8    | 2, 4, 8, 10, 16  | 2        | 8 or 16, in the decode monokernel                 |
| TP8    | 28               | 8        | 8 or 16 in the monokernel; 24, 32, 40, 48 or 64   |
| TP4    | 28               | 13       | up to 104                                         |
| TP4    | 36               | 20       | up to 160                                         |
| TP4    | 40               | 80       | up to 320 (40 sessions)                           |

Prefill graphs are off, so prefill rows are the uncached tokens themselves.

From 192 rows a forward takes the owned-row layout: each rank keeps
`ceil(rows / TP)` rows for the residual boundaries, norms, router and top-k
(TP8 prefill of 1,600 or 8,192 tokens: 200 or 1,024 rows; TP4 decode of 320
rows: 80; TP4 prefill of 1,600 or 24,576 tokens: 400 or 6,144). Attention, the
experts and the projections whose collectives gather rows still see every row.

## Workload statistics

Context lengths and turn sizes come from the AgentX agentic-coding trace corpus
that Muse Spark replays (`semianalysisai/cc-traces-weka-062126` on Hugging Face,
the one Kimi K3.1 replays). The traces count tokens in 64-token hash blocks, so
the numbers do not depend on the tokenizer:

- Context per request: median 142,016 tokens. Weighted by output tokens, which
  is what decode sees: median 217,344 and p90 629,184.
- Uncached tokens per turn: p25 576, median 1,600 and p75 2,944.
- Accepted tokens per verify step: about 4 (the DFlash2 golden acceptance
  length is 4.05 at seven drafts).

## Coverage

- TP8 decode steps of one or two requests run each decoder layer as one
  persistent launch (`monokernel/`), which this suite does not time. Its
  `layer_bench.py` times them, for example with `--gpus 8 --rows 8
  --requests 2 --kv-dtype fp8 --length 217344`.
- Not covered yet: the multi-rank fused collectives (the sliding attention
  tail, the boundaries, the gather-and-quantize top-k and the route
  completions); TP8 prefill experts, which run the sorted gate/up GEMM and the
  column-split down projection; the router launch shared with the pre-expert
  projection; the boundary that also writes the FP8 rows of TP4's FP8 QKV;
  and the DFlash2 drafter.
