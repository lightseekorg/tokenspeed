# FlashInfer MoE adapters

## NVFP4 routing-map padding

The TRT-LLM NVFP4 entry points use routing kernels that write both valid token
assignments and invalid entries in `permuted_idx_to_token_idx`. Each expert's
last tile owns its unused rows. Threads also fill the trailing unused capacity
and guard with `-1`, so the first GEMM does not gather activations using stale
workspace values. Valid rows are written once, without a preceding full-map
memset or an additional kernel launch.

The launcher passes the native tensor's actual allocated length to routing;
the producers never assume an extra guard exists. It applies to routing from
both logits and precomputed top-k, including different token counts, expert
partitions and GEMM tile sizes. It runs in eager
execution and is recorded inside CUDA graphs, so every replay refreshes padding
even when graphs reuse a memory pool. No host synchronization or extra routing
buffer is needed.

`thirdparty/flashinfer/trtllm_moe.py` builds a source-keyed private JIT module.
The adapter ships as a Python package in both source distributions and wheels;
it does not require a source checkout on `PYTHONPATH`.
It extends the existing block, dynamic-block, cluster, cooperative and offsets
producers at their tile-metadata and padded-count publication sites. Padding,
trailing capacity and valid assignments have disjoint writers. The writes run
before the producers' existing PDL completion triggers and share their stream
and graph dependencies. FlashInfer's routing decisions, GEMMs and
Python API signatures are retained. The small-batch tactic policy below narrows the tuner's candidates for one model.
The installed package and stock JIT modules are unchanged. The first warmup
requires FlashInfer's usual JIT toolchain and compiles the private module;
subsequent processes reuse its cache. Warmup must finish before graph capture.
The private build includes matching routing headers and source transforms;
the installed package is not patched. An unrecognized allocation, producer or
publication layout raises an error rather than silently skipping initialization.
The FlashInfer 0.7 custom-routing translation units use a shared source header;
the private build copies both together so all producers use the patched header.
Review this adapter when updating FlashInfer.
It can be removed once the minimum supported FlashInfer version guarantees
the same initialized-map contract on every routing invocation.

Regression coverage includes live and padded mapping entries, local expert
partitions, PDL on/off, changing routing within one captured shape, graph replay
after workspace corruption, and output equality with the upstream operator.
Expert-partition tests retain the loader's global activation input scales while
sharding expert weights, and select SiTU through FlashInfer's activation enum.

## Qwen3.8 low-batch tactic

For Qwen3.8's 2,560-hidden, 640-intermediate, 512-expert NVFP4 MoE with
TP4/EP4 and top-k 10, the private runner restricts inputs of at most 32 tokens
to FlashInfer's valid tile-32 tactics. If the native launcher does not offer
tile 32 for a profile, all native tactics remain available. Other models and
larger token counts use the full tuner. Only the matching shape gets a separate
tuning-cache key, so a previously cached tile-8 choice cannot bypass this
policy without forcing unrelated shapes to retune. The adapter raises an error
if FlashInfer removes either tuning hook or moves runner construction outside
the cloned entrypoints.

This is a temporary workaround for FlashInfer 0.7's MoE tactic selection.
Remove it when upstream tuning handles the full decode graph.

The policy targets CUDA-graph latency across routing, shared experts and MoE
GEMMs. On the measured BS1 MTP3 graph, FlashInfer 0.7's isolated-kernel tuner
selected tile 8, leaving an idle interval before routing. This policy keeps
tile 32 available for that shape while retaining upstream routing and GEMM
implementations.
