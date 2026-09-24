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
and graph dependencies. FlashInfer's routing decisions, tuning, GEMMs and
Python API signatures are retained.
The installed package and stock JIT modules are unchanged. The first warmup
requires FlashInfer's usual JIT toolchain and compiles the private module;
subsequent processes reuse its cache. Warmup must finish before graph capture.
The private build includes matching routing headers and source transforms;
the installed package is not patched. An unrecognized allocation, producer or
publication layout raises an error rather than silently skipping initialization.
Review this adapter when updating FlashInfer.
It can be removed once the minimum supported FlashInfer version guarantees
the same initialized-map contract on every routing invocation.

Regression coverage includes live and padded mapping entries, local expert
partitions, PDL on/off, changing routing within one captured shape, graph replay
after workspace corruption, and output equality with the upstream operator.
Expert-partition tests retain the loader's global activation input scales while
sharding expert weights, and select SiTU through FlashInfer's activation enum.
