# Gluon MLA decode context parallelism

Gluon MLA uses the existing KVP cache layout and the shared DCP exchange. It
does not introduce a cache allocator, model-owned placement policy or a new
scheduling path. The full-attention group is sharded; recurrent state remains
replicated. Aggregated serving requires equal KVP/DCP sizes and separate
prefill/decode rounds.

## Runtime contract

`GluonMLAAttnBackend` inherits the ordinary MLA execution and overrides the
parts that consume sharded history. `DCPDecodeState` in `attention/dcp/decode.py`
is shared with CuTe MLA. It builds compact local page tables and per-query
local lengths from virtual allocation IDs and global query endpoints. Its
buffers are initialized before capture and refreshed in place for eager and
captured execution. Draft frontier changes update lengths against the same
ownership prefix. Cache rebind drops pool-bound state.

The input table stays virtual for cache writes. `cache_placement` lets the
existing attention prologue translate slots and mask nonowners. Kernel reads
use only the compact local table. Ownership is determined by virtual block ID,
not logical token position.

Decode gathers TP query heads, computes partial latent output and natural-log
FP32 LSE over owned pages, and calls the shared partial exchange/merge. Empty
shards are normalized to zero output and negative-infinity LSE before exchange.
Value projection and output gating follow the merge; the projected-value
decode epilogue cannot operate on incomplete attention results.

The initial kernel contract is E4M3 KV, latent width 512, RoPE width 64,
64-token pages and full-history attention. Bitwise numerics and sliding-window
attention are rejected under DCP. A forward consumes either one query or the
configured full verify/block window.

## Cached-prefix prefill

The absorbed cached-extend kernel reads a complete page table and cannot read
the sharded pool directly. Under KVP the Gluon leaf uses ordinary chunked
prefix replay and prepares the shared `PrefixHistoryGather` with the host
mirror of the page table and the planner's exact per-request chunk boundaries.
Those boundaries stay with the prefill metadata so ragged prefix loops and
their owner gathers use the same partition. The model reads each bounded
prefix chunk through
`read_prefix_history`; cache layout stays outside the model. KVP=1 retains the
existing absorbed-extend selection.

## Initial kernels and subsequent optimization

The initial implementation runs each query as a single-query row with its
actual local visible length and processes gathered heads in chunks of at most
16. FP8 queries are promoted losslessly to BF16 for the existing BF16-query /
FP8-KV kernels: rounding softmax probabilities to FP8 separately on each shard
can exceed the long-context error bound. BF16 queries retain their precision.
This composition reuses registered Gluon kernels and deliberately rereads KV.
It establishes the end-to-end contract and makes no speedup claim.

The performance implementation should share KV across queries, accept runtime
`[batch, query]` local lengths and efficiently process 96 gathered heads.
Single-query decode also needs wider head support. Preserve this runtime and
cache contract when replacing the kernel scaffold; do not add another serving
mode. Consider exchange fusion or overlap only after measuring the existing
collectives.

## Validation

`test/runtime/test_dcp_page_metadata.py` tests shared ownership/visibility and
refresh storage. `test/runtime/test_gluon_mla_dcp.py` tests Gluon metadata,
draft frontier updates, block windows, reinitialization and unsupported cache
formats. CPU cases use real metadata operations; when no GPU exists, only
hardware discovery is mocked. GPU cases exercise the device implementation.

Run the distributed correctness check on a held GPU allocation:

```bash
PYTHONPATH=. python -m torch.distributed.run --standalone --nproc-per-node=8 \
  test/runtime/distributed/run_gluon_mla_dcp.py --kvp-size 8
```

The runner uses a real cache pool, Gluon leaf and collectives. It compares
decode, verify and draft outputs against independent dense FP32 attention,
including nonconsecutive ownership, empty shards, graph replay and boundary
crossings. It also reconstructs cached-prefix chunks bit for bit. The FP8
attention checks use the existing native kernel's 5% relative-L2 contract per
query and allow at most one extra percentage point of error against dense
attention compared with the no-DCP kernel. Both BF16 and FP8 queries are
covered. It is not a model accuracy test or a performance benchmark. Also
validate cached-prefix prefill, owner-masked writes, Host L2 restore and full
model logits/tokens before enabling a serving recipe.

## Performance measurements

Use the same build for DCP on/off and keep profiled timings separate from
unprofiled performance results. Measure in this order:

1. Roofline bytes/FLOPs for the actual shape and GPU.
2. Attention kernels, including normalization/merge cost: batch 1/2/4/10/16,
   query widths 1/4, representative full contexts 50K/121K/193K/500K and ragged
   ownership. Compare gathered-head/local-KV shapes with TP-head/full-KV shapes.
3. Eight-GPU query gather and partial exchange at the actual payload sizes,
   including graph capture. Account for every layer and draft pass.
4. Complete captured MLA layer, then the full speculative decode step.
5. Fixed shapes: 50K input/500 output and 200K/1K at C1/C16, including the
   prefix-history gather in TTFT.
6. Agentic trace screening, admission/cache sweeps, then matched one-hour
   comparisons with the same client, trace seed and acceptance assumptions.

Record TPOT mean/p50/p90, TTFT, output tokens/s with explicit units, request
failures, prefix hit, prefill share, queue pressure, cache occupancy, Host L2
loads, acceptance and measured token capacity. Separately verify real
acceptance and model accuracy. Treat C1 DCP as a separate recipe choice: its
collective cost may exceed the bandwidth saving.
