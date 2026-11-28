# Kimi K3.1 Pro kernel benchmarks on gfx950

These cases use the shapes of Kimi K3.1 Pro as TokenSpeed serves it in its
MI355X AgentX submission to InferenceX (`kimik3.1-fp4-mi355x-tokenspeed-agentic-dspark`,
recipe `benchmarks/single_node/srt-slurm-recipes/kimik3.1/tokenspeed/mi355x-fp4/agentic.yaml`).
K3.1 Pro keeps Kimi K3's weight shapes and adds a DSA lightning indexer to its
24 MLA layers, so the directory keeps its `kimi_k3` name. The submission runs
one worker on eight GPUs with attention and MoE TP8 (EP1), MXFP4 experts, an
FP8 KV cache, an mxfp4 Index-K cache, DSpark speculative decoding and
8192-token chunked prefill.

## Rows per step

A decode step verifies every running request's drafts plus its current token.
Each concurrency point caps the running requests (`max-num-seqs`):

| Concurrency | Requests | Verify rows each | Decode rows | KV parallel |
| ----------: | -------: | ---------------: | ----------: | ----------: |
| 1           | 1        | 8                | 8           | 1           |
| 4           | 2        | 8                | 16          | 1           |
| 8           | 8        | 8                | 64          | 8           |
| 16          | 8        | 6                | 48          | 8           |
| 24          | 12       | 6                | 72          | 8           |
| 32          | 16       | 6                | 96          | 8           |
| 48          | 48       | 6                | 288         | 8           |

Prefill runs 8192-token chunks. The KV-parallel points capture prefill graphs
of 256 to 3072 tokens, so a turn of at most 3072 uncached tokens runs padded to
one of them.

## Workload statistics

Context lengths and turn sizes come from the AgentX agentic-coding trace corpus
that Kimi K3.1 replays (`semianalysisai/cc-traces-weka-062126` on Hugging Face):

- Context per request: median 142,016 tokens. Weighted by output tokens, which
  is what decode sees: median 217,344 and p90 629,184.
- Uncached tokens per turn: p25 576, median 1,600 and p75 2,944. They pad to the
  1024, 2048 and 3072-token prefill graphs.
- Accepted tokens per verify step: about 4 (the DSpark golden acceptance length
  is 3.81 at five drafts and 4.16 at seven).

## Coverage

- In a decode step that holds one request (8 rows), the gfx950 MoE layer tail
  runs each MoE layer's work after attention. It is a multi-rank launch that
  this single-GPU suite does not time, so the MoE, o-projection and
  shared/latent GEMM cases start at 16 rows.
- GEMM cases name one registration each and list only the rows it serves under
  normal dispatch. Rows that dispatch elsewhere are not listed, for example 288
  decode rows or the 3808-wide DSA input projection past 16 rows.
- DSA cases time KV-parallel rank 0. Index-K rows and queries are random. The
  other ranks' candidate scores are rank 0's, permuted and slightly perturbed, so
  each rank keeps about `topk / 8` of the selection.
- Not covered yet: DSA prefill (index top-k and the sparse prefill core), the
  RMSNorm GEMV that serves the stacked `[q_b; wq_b]` projection at 8 rows, the
  MoE layer tail, and the DSpark drafter.
