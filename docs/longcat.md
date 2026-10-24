# LongCat inference

LongCat uses the model entry in `python/tokenspeed/runtime/models/longcat_flash.py`.
Model-specific execution adapters live in `python/tokenspeed/runtime/models/longcat/`.
The adapters import kernels through `tokenspeed_kernel.ops.longcat`; CuTe implementation
lives in `tokenspeed_kernel.thirdparty.cute_dsl.longcat`.

The Python package registers LongCat through its `tokenspeed.plugins` entry point.
After updating an existing editable checkout, reinstall `python/` to refresh the
package metadata before loading a LongCat checkpoint.

## Source layout

| Directory | Responsibility |
| --- | --- |
| `ops/longcat/` | Runtime-facing attention, MLP, MoE, router, and selection APIs |
| `thirdparty/cute_dsl/longcat/attention/` | Indexer, sparse MLA, projections, and attention collectives |
| `thirdparty/cute_dsl/longcat/engine/` | Static execution plans, shared workspaces, and layer programs |
| `thirdparty/cute_dsl/longcat/routines/` | Dense and expert GEMM, quantization, communication, and synchronization routines |
| `thirdparty/cute_dsl/longcat/embedding/` | Over-embedding lookup kernel |

Shared GEMM, quantization, sampling, and metadata operators retain their generic
kernel APIs. LongCat model registration does not import its GPU engine adapters;
those load when the corresponding engine is constructed. LongCat LSA lives in
`models/longcat/dsa.py`, sharing the MLA prologue and metadata helpers while
owning its paired selection and eight-row decode contract. The GLM model
keeps its existing implementation. LongCat model profiles declare their cache
geometry and token-history requirement; OE consumes the runtime's
`RequestTokenHistoryView`.

## Decode execution

The fused path targets TP8 with eight verify rows. Attention 0 assigns the indexer
to rank 0 and the 64 attention heads to ranks 1–7. The indexer publishes selected
KV slots through the selection exchange. Workers prepare queries and KV rows,
run sparse MLA, project the output, and participate in the collective.

The layer program composes dense MLP 0 and routing, expert work, attention 1,
and dense MLP 1. The last collective combines the dense/expert results with the
residual and computes the next input normalization. DFlash capture boundaries
select the appropriate layer output and notify the capture sink before the next
layer computes, so incremental draft projection can overlap that layer. The final
tap is notified after the decoder loop.

Unquantized BF16 draft linear layers use the standard kernel dispatch and honor
explicit kernel overrides and the selected numerics mode.

Programs define producer counts, ticket dependencies, and shared-memory lifetimes.
A consumer must acquire the published generation before reading its payload.
A reused slot must be retired after all consumers finish. Workspaces remain at
fixed addresses through CUDA Graph capture and replay.

Attention projection weights are BF16. The FP8 KV path stores E4M3 values and
converts them to FP16 for sparse-MLA QK/PV multiplication, with FP32 accumulation.
The engine requires an unscaled KV cache. Dense and expert MLP kernels use NVFP4
weights and activation packing with their associated scales.

Projection weights and BF16 activations are ordinary matrices loaded through
TMA into the consumers' SMEM layouts. BF16 streaming projections use a
128-byte-swizzled K-stripe layout while retaining their MMA and reduction
order. Folded normalization weights and absorbed-query compositions are
materialized once as BF16 matrices. Expert weight and block-scale layouts
are retained independently of the ordinary backend's weight processing.

Q-B reads the first 1536 columns of the ordinary QKV output with its
2112-element row stride after the Q-A completion ticket. Normalization writes
row-major BF16 outputs directly. NVFP4 activation values and block scales
retain their quantization and readiness contracts.

Attention 1's output projection retains its 128-row output tile, and
the five-stage ring loads each head's activation after its readiness ticket.
The weights stream ahead of those gates; the MMA and collective epilogue
consume the same stage and output layouts.

## Runtime selection

Set `TOKENSPEED_LONGCAT_LAYER_ENGINE=1` to enable the complete fused decode
path. This selects streaming BF16 projections, folded attention-0 preparation,
tcgen05 N32 indexer QK, and the full layer program. Ordinary prefill and idle
forwards use the model's native layers.

LongCat Pro requires one attention-DP group, one KV-partition group and pipeline
parallel size one, including when the fused engine is disabled.
Its primary architecture must be `LongcatCausalLM` or `LongcatFlashForCausalLM`.

The engine configuration requires TP8, one attention-DP group, verify width
eight, `--max-num-seqs 1`, `--numerics auto` and `--mla-lora-scale folded`. Launch with
`--disable-prefix-caching --disable-kvstore`: attention 0's indexer and worker
ranks own different cache planes, so its generated history cannot be published
for ordinary prefill reuse. Engine weights are fixed after initialization;
live updates are rejected. Query-context parallelism, pipeline parallelism and
expert-placement updates are outside this engine configuration.

Use `--disable-cudagraph-memory-reserve` to build the full KV arena before
warmup; the engine sizes its persistent workspaces for the full decode page table.

Keep requests independent. Feeding engine-generated history into a later prompt
can make completed-page deduplication reuse cache planes the engine does not
update, even with prefix matching disabled. Recovery and transitions to ordinary
attention require separate validation.

The index-K cache uses FP8 values with per-row scales. Draft capture uses
the shared `ForwardContext.target_capture_sink` interface. Backend write slots
and the MLA prologue remain the authoritative inputs for ordinary attention.
