# DeepSeek V4.1 Ascend operators

The runtime imports through `tokenspeed-kernel`. Request state, cache ownership,
prefix reuse and PD transfer remain in the shared runtime and LCM. Eager and
native-graph decode use the same live metadata and caller-owned output buffers.
See the [serving guide](../../../../docs/guides/deepseek-v41-ascend.md) for builds
and launch configuration.

## Cache and attention

Both profiles store KV and Index K in BF16. `bf16-qdq` applies operand rounding:
SWA E4M3/group32 E8M0, global E2M1/group16 E4M3, and Index Q/K E2M1/group32
E8M0. `bf16` skips target QDQ. Temporary byte codecs serve QDQ and DSpark;
packed resident-cache readers are unsupported. Operand QDQ does not reproduce
native low-bit GEMM accumulation.

Invalid writes leave storage untouched; invalid reads return zero. Prefill
reads the old SWA prefix before publishing new rows. The recipe allocates only the selected Index format.

Sparse attention receives full post-RoPE 512-dimensional Q/K and zero rotary
ABI inputs. Selection compaction preserves order and duplicates. The sink
correction operates on BF16-rounded SFA output and masks empty rows.
[Direct paged attention](README.paged-sfa.md) reads the aligned LCM arena;
draft attention retains compact windows. Both PD roles must share cache layout.

## Index selection

The registered `torch_npu` implementation uses `dsv41_index_device.py` for
both prefill and decode. Row selection without candidate publication uses
`custom.npu_lightning_indexer` with FP32 scoring; candidate-pool generation
retains BF16 intermediate rounding. Query count and chunk size only bound
scratch and never switch these scoring rules.

Candidate generation scans full history with Ascend C for supported geometry
and Triton-Ascend otherwise. It preserves BF16 dot/product/shard/final rounding
and shares key loads only between queries with matching physical page IDs.
History partitioning is a runtime parameter. Selection produces row TopK and
TopK maxima of contiguous eight-position blocks, retaining the latest resident
block. Ordered keys preserve earlier-column ties; small batches use native TopK.

## Projection and mHC

Wide attention TP projects local head slices with BF16 operands and FP32
accumulation. Output-group owners sum FP32 partials, round to BF16, then apply
`wo_b`. This changes summation association relative to an unsplit GEMM.

mHC coefficient projection, RMS and Sinkhorn remain FP32. Pre-mix/RMSNorm
preserves the BF16 boundary between mixing and normalization. Post-mix combines
residuals and scaling in FP32 before the BF16 store. RoPE retains separate FP32
products and additions. BF16-input projections can produce FP32 directly,
avoiding materialized FP32 input copies. Weight preparation preserves persistent
buffer addresses needed by graph replay.

## Experts

Native routing uses fixed capacity and device expert counts. GMM consumes only
valid expert rows. Route weights apply before the down projection; combination
accumulates BF16 expert outputs in FP32 without applying weights again.
Router selection preserves NaN-first ordering and lower-ID ties.

The native W4A4 kernel fuses grouped INT4 Cube computation and token scaling.
Its rounding contract is FP19 channel scale, FP16 intermediate, FP32 token
scaling and BF16 output. Device counts define active work on replay. CANN serves
other geometries. See [MoE configuration](../../../../tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/README.ascend.md)
for manifests and kernel selection. Offline conversion is outside this package.

## Engram and DSpark

Engram column projection uses the smaller of Dense TP and attention TP;
all-gather restores key/value order. Device table lookup decodes local
FP8/group32 rows to BF16 and leaves shard combination to shared collectives.
History/hash preparation uses exact integer arithmetic and preserves seed
frontier checks. Accepted history changes only at the shared commit boundary.

DSpark sampling preserves score/token tie order. Window and compressor metadata
consume live request slots, lengths and page tables. Hash-route bounds checks
use device assertions compatible with graph replay.

Unequal HCCL shards use a copy-and-pad kernel for aligned contiguous rows in
groups of up to sixteen ranks. It preserves raw values and zeroes only padding;
equal shards bypass packing and other layouts use the ordinary copy path.
