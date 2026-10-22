# KV cache kernels

## Range clearing

`kvcache.zero_byte_ranges` dispatches by physical placement. Device allocations
use the original Triton helper; pinned host allocations use
`cute_dsl_zero_host_byte_ranges` on the execution device's current stream. CPU
arenas clear synchronously. Host callers retain backing storage until stream
completion, including before a CPU read. Offsets are relative to the tensor view;
the CuTe kernel uses int64 byte addressing and a bounded grid that strides each
range, preserving unaligned tails and neighboring bytes. Range counts, lengths,
and pointers are dynamic arguments and do not trigger new compilations.

This scheduler/lifecycle operation consumes a CPU range list outside CUDA graph
capture. Capturing it fails explicitly rather than retaining a temporary pinned
metadata pointer in a graph. Model forward graph replay is unchanged.

## Sparse KV offloading

`offload.py` is the runtime boundary. NVIDIA uses the registered
`cute_dsl_offload_materialize` implementation and CuTe DSL row helpers.
The private `_cute_dsl` Python package ships in both wheels and source archives.
NVIDIA tests check the GPU vendor before importing CuTe; ROCm also exposes
`torch.cuda`, so GPU availability alone does not identify this backend.
It accepts physical history IDs (positive int32; zero/negative are invalid),
request IDs (int32 or scheduler int64), persistent hot tags/LRU, and preallocated
scratch. An invocation resolves the whole request's Q×K selection union.
Original order, duplicate references, and invalid masks survive in the output.

One CTA per request constructs an open-addressed selection hash. Atomic minimum
of original indices makes duplicate ownership deterministic. After a CTA barrier,
resident scanning selects the minimum physical slot for duplicate tags, protects
selected/current ordinary rows, and stably partitions the old LRU. Unique misses
use first-occurrence order and the oldest unprotected slots. Before publishing
tags, an unconditional device trap rejects capacity overflow; probes are bounded.
The new LRU is remaining old slots, installed misses, then protected old slots.
Reserved/ring rows participate in lookup but cannot be victims.

The hash uses two int32 arrays at a load factor ≤0.5. `hash_geometry` selects
shared tables up to 128 KiB, and budgeted global tables above that threshold.
Global `entry_dest` separates canonical destinations from the final output.
Miss pairs occupy their canonical input positions, with other entries masked;
the static copy grid therefore requires no CPU read of the miss count.

A separate row-parallel CuTe kernel copies mapped pinned host bytes. Integer
32-bit transfers are used for aligned rows, with a byte path for arbitrary tails.
Current/seed/accepted/reset helpers also use CuTe. The caller records ready only
after both resolver and payload copy, and joins write-through before reusing
current slots. Shared tables are rebuilt on every replay; padding rows never
write persistent tags or LRU.

Compiled executors are cached by device, static geometry, and pointer element
types. Batch size, pointers, table width/stride, and sequence/accept counts remain
runtime values. Warmup precedes graph capture; an unwarmed capture fails explicitly.
The runtime also warms admission's int64 request reset before serving.
The residency path uses only CuTe DSL. The former resident/selection sorting,
binary-search lookup, and adjacent-entry miss deduplication have been removed.
Hash ownership merges duplicate loads while preserving all attention references;
upstream top-k ordering remains part of the attention numerical contract.

## Ascend KVP page compaction

Compact KVP tables retain owned physical pages in their original virtual
position order. Ascend uses an inclusive owner scan and a device scatter:
owned column i writes to prefix[i]-1; an unowned column writes zero to
total_owned+i-prefix[i]. These destinations partition the whole output row,
so every entry is written exactly once, including padding. No general sort
or host read of the owned-page count is needed.

With a page-prefix buffer, allocated draft reserve pages are retained and
causal token counts still exclude rows beyond the current endpoint. Without
it, only pages intersecting the endpoint are packed. Both paths refresh the
caller's persistent buffers and support changed tables/lengths in NPU graph
replay. This device-op implementation avoids the synchronous lock setup in
the current Ascend Triton scan launcher during capture. It uses AIV scan and
scatter tasks, rather than the int64 sort that dispatches to AICPU.


### Large Ascend clear grids

`zero_page_fields` caps the flattened Ascend launch grid at 65535 programs
and strides over remaining page/field ranges. The logical page count remains
a runtime argument. The tile dimension is at most 1024 for every nonempty
range list, so dividing the launch limit by it always leaves at least one
range program. Only selected fields are cleared; surrounding arena bytes
must remain unchanged. A small-grid device test exercises both the range
stride and payload-tile stride, including partial tiles and unselected pages.

`copy_state_rows` uses the same conservative 65535-program cap. Large state copies
split along logical rows while retaining global layer indices and physical row
strides. CUDA keeps one launch. The start row is a runtime argument.
