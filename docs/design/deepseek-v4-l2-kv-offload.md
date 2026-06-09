# DeepSeek V4 L2 KV Cache Offload Design

Status: design and implementation plan for `feat/ds-v4-l2-offload`.

This document designs host-side CPU L2 offload for DeepSeek V4 KV cache.
It is intentionally scoped to L2 host memory. L3 storage, cross-node cache
transport, and new third-party runtime dependencies are out of scope.

## Goals

1. Support DeepSeek V4 when KVStore is enabled internally
   (`ServerArgs.enable_kvstore`) without falling back to token-indexed
   `get_cpu_copy` / `load_cpu_copy` paths. Do not add a new CLI flag: the
   user-facing switch today is `--disable-kvstore`, while KVStore defaults on
   for non-decode instances unless disabled.
2. Preserve the existing overlap scheduling contract: scheduler cache loadback
   must not add a global GPU sync, and layer execution should only wait for the
   loaded layer that it actually consumes.
3. Offload the complete V4 paged cache state:
   - SWA KV sliding-window cache.
   - Compressed full-history KV cache.
   - Compressor state sliding-window cache.
   - CSA indexer KV cache that shares compressed-KV page ids.
   - CSA indexer compressor state sliding-window cache.
4. Maximize PCIe utilization by using page-level, batched, pinned host-memory
   transfers on independent D2H and H2D streams.
5. Keep runtime dependency boundaries unchanged. This implementation should not
   add new runtime dependencies or new transfer kernels.

The first production slice must support both history-family prefix reuse and
terminal state prefix reuse. The replay semantics should match the existing
device paged-cache replay path: host hits are first materialized back into
device paged-cache pages, then the normal V4 forward path consumes device
block tables and base logical-page offsets.

## First Principles

The design follows a few hard constraints instead of optimizing for abstraction
generality:

1. Cache correctness is ownership plus visibility. A cache page is reusable only
   when the scheduler can prove which memory tier owns it and that the async
   transfer that produced it has completed on every rank.
2. Model code consumes device cache only. Host cache is a storage tier, not a
   second attention backend. Therefore host hits must become device pages before
   `FlatForwardOperation` is built.
3. The scheduler owns cache identity and lifetime. Python runtime code owns
   bytes and streams. Do not let runtime copy code decide prefix-cache validity.
4. The transfer unit is a V4 paged-cache group page, not a token span and not
   the ordinary KV page.
5. Keep the first implementation simple: use pinned/registered host tensors and
   batched DMA-style copies. Do not introduce a custom transfer kernel in this
   implementation.
6. Synchronization must be local to the data dependency. No scheduler-wide
   loadback barrier, no global `torch.cuda.synchronize()`, and no stream wait
   beyond the producing/consuming copy and layer events.

## Current Code Facts

DeepSeek V4 does not use the ordinary single KV page table.
`DeepseekV4TokenToKVPool` owns several per-group paged buffers. The legacy
token-indexed `get_cpu_copy` / `load_cpu_copy` APIs remain unsupported because
compressed-MQA and indexer buffers are page-shaped and cannot be copied by
token-indexed MHA/MLA logic. L2 support must therefore go through the
group-paged host-pool path described below.

The scheduler already has most of the device-side V4 model:

- `PagedCacheGroupSpec` describes group id, rows per page, entry stride,
  retention, sliding window, and history/state family.
- `PagedCacheSnapshot` records per-group device page ownership plus cursor and
  completeness metadata.
- `HybridPrefixCache` can attach, import, and commit device-side V4 paged cache
  snapshots.
- `FlatForwardOperation` already carries `paged_cache_block_tables` and compact
  logical-page base offsets to Python.

The missing part is host residency for these group pages:

- `TreeNode` currently has ordinary `DeviceResource`, ordinary `HostResource`,
  Mamba device/host slots, and one `PagedCacheSnapshot`. That paged snapshot is
  effectively device-side today.
- `MatchResult::PagedCache` is a single hit structure. It does not distinguish
  device pages from host pages.
- `HybridPrefixCache::augmentMatchPagedCache` matches and imports device
  paged-cache snapshots; `AcquireForRequest` expects page ids it can use as
  device pages.
- `CacheKind` currently distinguishes only `kv` and `mamba`.
- `HostExecutor` is built around one `CachePool.page_size()` per kind. V4 group
  pages have different byte sizes, so they should not be forced through the
  existing kind/page-size path.
- `MemoryExecutor` only creates MHA/MLA host KV pools plus the optional Mamba L2
  pool.

## Reference Patterns

### Current Mamba L2

The Mamba L2 path is the closest in-tree pattern:

- Scheduler distinguishes device and host Mamba state.
- Host memory is a dedicated layer-first pool.
- Writeback is all-layer D2H.
- Loadback is per-layer H2D.
- `LayerDoneCounter` is used by the model path to wait only when a layer reads
  restored state.

This is the correct synchronization shape for V4. The main difference is that
V4 has multiple independent paged groups and some logical groups map to more
than one physical tensor.

### vLLM CPU offload

The local vLLM implementation reinforces several points that should carry over:

- Scheduler-side state should manage cache identity, readiness, eviction, and
  store/load preparation. Worker/runtime code should only execute transfer
  specs.
- Transfers should operate on canonical block/page descriptors rather than
  model-specific token ranges.
- Worker-side code canonicalizes model-specific KV layouts into raw
  `[num_blocks, page_bytes]` tensors and data refs. TokenSpeed should do the
  same for V4 group buffers instead of starting with a model-specific transfer
  kernel.
- Pinned or host-registered memory plus batched pointer descriptors are the
  right first primitive. GPU-to-CPU is mostly DMA bandwidth bound, while
  CPU-to-GPU uses the same batched-copy path in this implementation.
- Store completion makes host cache visible. In-flight store blocks must be
  pinned or otherwise protected so allocator reuse cannot race with async D2H.
- DeepSeek-style hybrid groups need alignment filtering for sliding-window
  groups: SWA pages that can never be used by a later full-history-aligned hit
  should not be eagerly stored.

TokenSpeed should not import vLLM code. The useful part is the separation of
policy metadata from transfer execution, the canonical raw-block descriptor
model, and the completion-before-visible rule.

## DeepSeek V4 Cache Groups

The L2 unit is a V4 paged-cache group page, not an ordinary KV token page.

| Group | Family | Retention | Device buffers | L2 copy payload |
| --- | --- | --- | --- | --- |
| `v4.swa_kv` | state | sliding window | `swa_kv_buffer[layer]` | one uint8 SWA page per layer |
| `v4.c{r}a.compressor_state` | state | sliding window | `compressor_state_buffer[layer]` for ratio `r` | one fp32 state page for each layer using ratio `r` |
| `v4.c{r}a.compressed_kv` | history | full history | `compressed_kv_buffer[layer]` for ratio `r` | one uint8 compressed page for each layer using ratio `r` |
| `v4.c4a.compressed_kv` | history | full history | `compressed_kv_buffer[layer]` and `indexer_kv_buffer[layer]` for ratio 4 | both tensors must be copied under the same logical page id |
| `v4.c4a.indexer_compressor_state` | state | sliding window | `indexer_state_buffer[layer]` for ratio 4 | one fp32 indexer-state page for each ratio-4 layer |

The ratio-4 indexer KV cache is the easiest place to corrupt correctness:
it has no separate page table. A transfer for `v4.c4a.compressed_kv` must copy
both compressed KV and indexer KV payloads for the same page id.

## High-Level Flow

```
request finishes, retracts, or device cache pressure rises
        |
        v
scheduler allocates host pages for every V4 group in the committed snapshot
        |
        v
tree node owns a pending host paged-cache snapshot
        |
        v
HostExecutor write stream copies device group pages to host group pages
        |
        v
WriteBackDoneEvent promotes pending host snapshot after all-rank ack
        |
        v
later request matches host pages when device pages are absent or shorter
        |
        v
scheduler allocates device pages and emits paged-cache LoadBackOp
        |
        v
forward op is built from destination device pages, never host page ids
        |
        v
HostExecutor load stream copies host group pages back per layer
        |
        v
V4 buffer getters wait through LayerDoneCounter before each layer reads cache
```

## Scheduler Design

### Typed paged-cache transfer operations

Do not encode V4 groups as more `CacheKind` enum values, and do not force V4
through the existing `CachePool.page_size()` path. The group ids are already
strings in scheduler configs and forward operations, and each V4 group has its
own page id space and page byte size.

Add a typed transfer payload beside ordinary `TransferPair`:

```cpp
struct PagedCacheTransferPair {
    std::string group_id;
    std::vector<std::int32_t> src_pages;
    std::vector<std::int32_t> dst_pages;
};
```

Then extend `WriteBackOperation`, `LoadBackOperation`, `FlatWriteBackOperation`,
and `FlatLoadBackOperation` with:

```cpp
std::vector<std::vector<PagedCacheTransferPair>> paged_cache_transfers;
```

The outer vector is indexed by op id in the same order as `op_ids`. This keeps
op-level grouping explicit and avoids adding parallel maps whose keys must be
kept aligned by convention. Python bindings can expose the typed object or a
plain list of `(group_id, src_pages, dst_pages)` tuples per op.

Keep the current `src_pages`, `dst_pages`, and `src_pages_by_kind` fields
unchanged for MHA/MLA/Mamba compatibility. V4 runtime code should consume only
`paged_cache_transfers`; legacy KV runtime code should ignore it.

This is the minimal ABI change that preserves the existing kind-pool model while
making V4 group identity explicit.

### Host paged-cache allocators

Add host-side paged group allocators mirroring device paged group allocators.
Each group must have its own host page id space.

Host page counts must be computed from V4 group bytes, not from a single
ordinary `num_host_pages` value:

1. Runtime computes per-group `bytes_per_page` from the V4 layout and layer
   ratios.
2. If `--kvstore-size` is set, distribute the byte budget across groups in
   proportion to desired resident device bytes, then divide by each group page
   size.
3. Otherwise use `ceil(device_group_pages * --kvstore-ratio)` per group.
4. Send the per-group host page counts into scheduler config.

State groups need enough host pages for trailing-window snapshots. History
groups need enough pages for full-history prefix reuse. These capacities should
be logged per group at startup because a single aggregate count is misleading.

Concretely, add a scheduler config field parallel to `paged_cache_groups`, for
example:

```cpp
std::map<std::string, std::int32_t> paged_cache_host_group_pages;
```

Python `scheduler_utils.make_config(...)` should populate it from
`DeepseekV4TokenToKVPoolHost`. The existing scalar `num_host_pages` remains for
ordinary KV host pages and must not be interpreted as V4 host group capacity.
For V4, runtime should still provide a token-page shadow capacity through
`num_host_pages` so the scheduler can attach `HostResource` locks to tree nodes
that own visible host paged snapshots. These shadow pages do not allocate
runtime KV host tensors and are not copied by `HostExecutor`; they exist only
to reuse the existing HostNodeRef pin/eviction machinery for in-flight H2D
sources and D2H destinations. Real V4 host capacity is solely the per-group
`paged_cache_host_group_pages` map.

### Tree-node residency model

The tree needs separate device, visible host, and pending host paged-cache
residency. The current single `PagedCacheSnapshot` can become the device
snapshot, but host pages must not be stored in the same field because
`AcquireForRequest` treats matched page ids as device pages.

Conceptually:

```cpp
class TreeNode {
    std::optional<PagedCacheSnapshot> paged_cache_device_snapshot_;
    std::optional<PagedCacheSnapshot> paged_cache_host_snapshot_;
    std::optional<PagedCacheSnapshot> paged_cache_pending_host_snapshot_;
};
```

All snapshots must preserve:

- `prefix_len_tokens`.
- Per-group page ids.
- Per-group base logical page.
- Raw token cursor.
- Complete history/state family metadata.

The state transitions are:

1. `CommitChunk` or state checkpoint produces a device snapshot under the
   existing accepted-token boundary.
2. Host writeback allocation creates `paged_cache_pending_host_snapshot_` with
   host page ids. Pending snapshots are not visible to prefix matching.
3. `WriteBackOperation` pins the source device snapshot and copies device pages
   into pending host pages.
4. `WriteBackDoneEvent` promotes pending host to visible host only after all
   ranks report success.
5. On failure, pending host pages are released and no visible host metadata is
   changed.
6. Device demotion may detach `paged_cache_device_snapshot_` only when no live
   request or in-flight writeback pins it. It must not detach the visible host
   snapshot.

The implementation can hide these fields behind a small `PagedCacheResidency`
helper if that keeps `TreeNode` cleaner, but the semantics above should remain
explicit.

### Matching and loadback

Extend `HybridPrefixCache::augmentMatchPagedCache` so it can return both:

- best device paged-cache hit, directly usable by a forward op;
- best host paged-cache hit, usable after LoadBack.

The match rules are identical for device and host snapshots:

- History-family groups must be complete at the history alignment boundary.
- State-family groups can be restored only when their trailing-window state is
  complete for the selected node.
- Transport-only continuation state requires exact terminal continuation; if it
  is missing, cap the hit instead of silently reusing partial state.

For DeepSeek V4, Python only passes history-family groups in the prefix-cache
required group set. C++ derives the state-family groups from the registered
allocators: page-aligned state groups may contribute to intermediate snapshots,
while all state groups are also eligible for exact terminal continuation
snapshots.

The first implementation should support both match intents:

- `MatchIntent::PrefixReuse`: import complete history-family groups. If the
  selected replay point also has a complete terminal continuation snapshot,
  import its state-family groups as well; otherwise cap the hit to history-only
  replay instead of importing partial state.
- `MatchIntent::StateRecovery`: require exact terminal state completeness for
  SWA, compressor state, and indexer state. If exact terminal state is absent,
  cap the hit just as the device replay path does.

When host is deeper than device, the scheduler allocates device pages per group,
emits a paged-cache `LoadBackOperation`, and constructs the forward operation
with the destination device pages. The host snapshot is a source only; the
forward operation must never contain host page ids.

The replay sequence is therefore:

1. Match host snapshot and compute usable prefix/state length.
2. Allocate fresh device paged-cache pages for every imported group.
3. Build `(host_page -> device_page)` transfers per group.
4. Attach or import the destination device snapshot for the request.
5. Populate `FlatForwardOperation.paged_cache_block_tables` from destination
   device pages.
6. Submit H2D loadback before model execution.
7. Let V4 layer getters wait only for their producer layer event.

Loadback does not need a scheduler completion event. It is synchronized by
layer counters in the runtime, matching the current Mamba path.

### Writeback and demotion

Writeback should be generated when a committed V4 snapshot becomes worth
preserving on host:

- request finish;
- retract/preempt path;
- device residency demotion when host cache is enabled and pressure requires it.

For every selected node:

1. Read the already committed device `PagedCacheSnapshot`; do not infer pages
   from request token length.
2. Allocate host pages for each group and build a pending host snapshot.
3. Emit `PagedCacheTransferPair` for each group.
4. Keep the device snapshot pinned while the writeback is in flight.
5. On all-rank success, promote pending host to visible host.
6. If the host copy is complete and the device snapshot is no longer pinned by
   live requests, demote only the device snapshot.

On failed writeback, release the newly allocated host pages and keep scheduler
metadata unchanged.

This mirrors the vLLM completion-before-visible rule while staying inside
TokenSpeed's existing event-state model.

Visible host paged snapshots must be attached only to tree nodes that also have
a scheduler `HostResource`. That resource is a shadow token-page lock for V4:
it protects host paged pages from reuse while H2D loadback is in flight, and
ordinary host-resource eviction releases the attached V4 host snapshot.

### Accepted-token boundary

For V4 MTP, only accepted target tokens may become globally reusable cache.
Group pages that contain rejected verify-tail or draft-only state must remain
request-local. The L2 writeback path must consume the same committed snapshot
boundary as prefix-cache publish. It must not copy raw working buffers by
request token length.

## Runtime Design

### Host pool

Add a DeepSeek V4 host pool, for example:

```python
class DeepseekV4TokenToKVPoolHost:
    def __init__(device_pool, host_ratio, host_size_gb, host_group_page_counts):
        ...
```

The host layout should mirror V4 device group buffers:

- `swa_kv_buffer[layer]`: `[host_swa_pages, swa_block_bytes]`, `uint8`, CPU.
- `compressed_kv_buffer[layer]`: `[host_compressed_pages, compressed_page_bytes]`,
  `uint8`, CPU.
- `compressor_state_buffer[layer]`: `[host_state_pages, rows_per_page, state_width * 2]`,
  `float32`, CPU.
- `indexer_kv_buffer[layer]`: `[host_compressed_pages, indexer_page_bytes]`,
  `uint8`, CPU, sharing the compressed group page count.
- `indexer_state_buffer[layer]`: `[host_indexer_state_pages, rows_per_page, state_width * 2]`,
  `float32`, CPU.

Allocate CPU tensors normally, then register them through
`current_platform().register_host_tensor_for_gpu_access`. This keeps the runtime
inside existing platform abstractions and avoids a direct CUDA dependency.

This implementation supports `layer_first` host layout only. It is the natural
layout for per-layer loadback and matches the current V4 device buffer
organization. `page_first` and L3-oriented contiguous page blobs are not in this
scope.

The host pool also reports a token-page shadow `page_num` derived from
history-family host group capacity. This value is only scheduler metadata for
`HostResource` locks; it does not create ordinary KV host buffers and must not
be used as V4 group capacity.

### Transfer pool

Add a V4 transfer pool beside `KVCachePool` and `MambaCachePool`, but do not
force it through the existing single-page-size `CachePool` shape. It should
consume the typed `PagedCacheTransferPair` list and use group ids to build raw
copy descriptors.

```python
class DeepseekV4CachePool:
    supports_layerwise_loadback = True
    def writeback_paged(transfers: list[PagedCacheTransferPair]): ...
    def loadback_paged(
        transfers: list[PagedCacheTransferPair],
        layer_idx: int,
    ): ...
```

`MemoryExecutor.submit` should route `paged_cache_transfers` to this pool when
the device pool is `DeepseekV4TokenToKVPool`.

The existing `CachePool.kind` protocol is keyed by `CacheKind`, which is too
narrow for V4 group ids. Keep the old protocol for legacy pools and add an
explicit V4 path in `HostExecutor` or a small sibling executor for paged-cache
groups. That is less magical than pretending V4 is just another `kv` pool.

### Copy implementation

The first implementation should copy with canonical raw block descriptors:

```python
@dataclass
class PagedCacheTensorRef:
    group_id: str
    layer_id: int
    device_tensor: torch.Tensor  # viewed as [device_pages, page_bytes]
    host_tensor: torch.Tensor    # viewed as [host_pages, page_bytes]
    page_bytes: int
```

For each `PagedCacheTransferPair`, expand `(group_id, src_pages, dst_pages)` to
one or more `PagedCacheTensorRef` copy descriptors. This mirrors vLLM's raw
`[num_blocks, page_bytes]` canonicalization and keeps model-specific layout
knowledge in one V4 host-pool module.

The V4 group-to-buffer mapping is:

- `v4.swa_kv`: copy `swa_kv_buffer[layer]`.
- `v4.c{r}a.compressor_state`: copy `compressor_state_buffer[layer]` where
  layer ratio is `r`.
- `v4.c{r}a.compressed_kv`: copy `compressed_kv_buffer[layer]` where layer
  ratio is `r`.
- `v4.c4a.compressed_kv`: additionally copy `indexer_kv_buffer[layer]`.
- `v4.c4a.indexer_compressor_state`: copy `indexer_state_buffer[layer]`.

Use the existing runtime streams:

- D2H writeback expands all layers for the selected group pages and submits one
  merged batch on `write_stream`.
- H2D loadback expands one layer at a time across all selected groups and
  records the layer completion after that layer's batch completes.

The initial backend should be a batched copy implementation over registered
host memory. Prefer existing platform copy helpers or a small runtime wrapper
around the same primitive shape as vLLM's `cuMemcpyBatchAsync` /
`hipMemcpyBatchAsync` path. A new paged-copy kernel is not part of this design
or implementation plan.

CUDA D2H should prefer driver DMA / copy-engine behavior for large pages.
H2D uses the same batched-copy implementation for now; no Triton/CuteDSL copy
kernel is planned in this scope.

### Layer synchronization

Register a `LayerDoneCounter` on `DeepseekV4TokenToKVPool`. All V4 buffer
accessors that can read restored pages must wait:

- `get_swa_kv_buffer`
- `get_compressed_kv_buffer_2d`
- `get_compressor_state_buffer`
- `get_indexer_kv_buffer_2d`
- `get_indexer_state_buffer`
- `get_key_buffer`, `get_value_buffer`, and `get_kv_buffer` through the SWA
  getter

Use one counter for the whole V4 pool. The load stream should mark a layer done
only after every group payload for that layer has been copied. This avoids a
partial restore where attention sees compressed KV but the indexer still sees
stale state.

Do not call `torch.cuda.synchronize()` in the offload path. The existing
`LayerDoneCounter.update_producer()` may apply bounded backpressure when the
producer ring is reused; that is acceptable because it is not a scheduler-wide
loadback barrier and happens only when outstanding loadbacks exceed the counter
ring.

### Stream ordering

The intended ordering is:

- D2H writeback stream waits once on the current compute stream before reading
  live device cache pages.
- H2D loadback stream is submitted before forward execution.
- Compute stream waits only when a V4 layer getter consumes a loaded layer.
- Writeback is throttled behind loadback when both queues are non-empty, as
  `HostExecutor.flush` already does.

This preserves overlap scheduling:

- lower model layers can execute while upper-layer H2D loadback is still in
  flight;
- D2H writeback can proceed after the producing compute stream reaches the
  recorded point;
- scheduler does not block on loadback completion;
- bounded producer-ring backpressure is allowed when outstanding loadbacks
  exceed available layer-event slots;
- only writeback completion changes host residency metadata.

## Implementation Plan

This section is the scope for the implementation that follows this design. It
does not include transfer-kernel tuning, L3 storage, page-first host layout, or
new KVStore CLI flags.

### 1. Scheduler Residency And Transfer Metadata

Files likely touched:

- `tokenspeed-scheduler/csrc/scheduler/operations/cache.h`
- `tokenspeed-scheduler/bindings/python_module.cpp`
- `tokenspeed-scheduler/csrc/resource/radix_tree/*`
- `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/*`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp`
- `tokenspeed-scheduler/tests/cpp/*`

Tasks:

1. Add typed paged-cache transfer specs to cache ops and Python bindings.
2. Add host paged-cache group allocators and config plumbing.
3. Add device, visible-host, and pending-host paged-cache residency to tree
   nodes or a `PagedCacheResidency` helper.
4. Extend V4 paged-cache matching to report device and host hits for both
   history-family prefix reuse and exact terminal state recovery.
5. Generate LoadBackOp and WriteBackOp with per-group page ids.
6. Attach host snapshots only after all-rank writeback completion.
7. Add only the necessary C++ tests for the new scheduler contract:
   - typed paged transfer fields survive flattening and Python binding shape;
   - pending host snapshot is not matchable before writeback ack, then becomes
     matchable after ack;
   - host V4 hit allocates destination device group pages and forward metadata
     uses those device pages, not host page ids;
   - terminal state recovery from host succeeds when all state-family groups are
     complete and caps the hit when a required state group is missing;
   - ratio-4 compressed group transfer keeps compressed KV and indexer KV tied
     to the same logical page id.
   Do not add broad ordinary KV/Mamba regression tests unless the implementation
   changes their existing cache-op behavior.

### 2. Runtime Host Pool And L2 Transfer

Files likely touched:

- `python/tokenspeed/runtime/cache/executor/memory_executor.py`
- `python/tokenspeed/runtime/cache/executor/host_executor.py`
- `python/tokenspeed/runtime/cache/transfer/pool.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`
- new `python/tokenspeed/runtime/cache/deepseek_v4_cache_host.py`
- new `python/tokenspeed/runtime/cache/transfer/deepseek_v4_pool.py`
- `python/tokenspeed/runtime/engine/event_loop.py`
- `python/tokenspeed/runtime/engine/scheduler_utils.py`

Tasks:

1. Build `DeepseekV4TokenToKVPoolHost` with registered host tensors.
2. Build `DeepseekV4CachePool` with canonical raw block descriptors and route
   typed paged transfers through `MemoryExecutor`.
3. Pass host paged group counts into scheduler config.
4. Flip `DeepseekV4TokenToKVPool.supports_hierarchical_kv_cache` only when the
   host pool path is implemented.
5. Register and consume `LayerDoneCounter` in all V4 cache getters.
6. Implement D2H and H2D with the batched-copy path over registered host memory.
   Do not add a custom transfer kernel.
7. Add only the necessary Python unit tests for runtime transfer/load:
   - host page-count and tensor shape computation for V4 groups;
   - canonical descriptor expansion for SWA, compressed KV, compressor state,
     and ratio-4 indexer payloads;
   - batched copy dispatch receives the expected descriptors for D2H and H2D;
   - layerwise loadback marks a layer complete only after all required V4 group
     descriptors for that layer are submitted.
   Avoid testing the copy backend implementation itself with synthetic bandwidth
   cases; integration validation is enough for transfer timing in this scope.

### 3. Integration, Docs, And Validation

Files likely touched:

- `docs/serving/deepseek-v4.md`
- scheduler/runtime metrics files

Tasks:

1. Document when DeepSeek V4 is supported under the existing KVStore behavior
   and recommend `--kvstore-size` / `--kvstore-ratio` sizing. Do not introduce
   a redundant affirmative enable CLI flag; users opt out with
   `--disable-kvstore`.
2. Add minimal logs and metrics required to validate the implementation:
   - host pages total/free per V4 group;
   - D2H/H2D pages and bytes per group;
   - loadback queue time and transfer time if available from the copy helper;
   - prefix hit tokens from device versus host;
   - hit caps due to missing state groups.
3. Validate functional correctness with GPU prefix cache reset followed by host
   cache hit.
4. Validate terminal state recovery from host and history-family prefix reuse
   from host.
5. Validate mixed prefill/decode does not introduce scheduler-wide loadback
   waits.

## Correctness Invariants

1. A V4 host snapshot is visible to prefix matching only after all group pages
   in that snapshot are written successfully.
2. Pending host snapshots never participate in matching.
3. A host hit must be replayed into destination device pages before
   `FlatForwardOperation` is built. Forward ops never contain host page ids.
4. A V4 forward operation may use loaded device pages only after the matching
   layer's loadback event is complete.
5. `v4.c4a.compressed_kv` host/device page ids cover both compressed KV and
   indexer KV payloads.
6. History groups and state groups are matched with the same completeness rules
   on host and device.
7. Terminal state recovery requires exact state-family completeness; otherwise
   the match must cap rather than reuse partial state.
8. L2 writeback copies committed cache snapshots, not raw request-local working
   ranges.
9. No offload path may call `torch.cuda.synchronize()` or force a scheduler-wide
   loadback completion barrier.
10. All ranks must generate equivalent cache op ids and host snapshot attachment
   decisions; writeback success is committed only after rank agreement.

## Performance Notes

- Use pinned or CUDA-registered host tensors. Pageable host memory should be
  treated as a correctness fallback only.
- Start with batched DMA-style copies over canonical raw page tensors. A custom
  copy kernel is not part of the first implementation.
- Prefer larger merged descriptor batches for D2H; the transfer is bandwidth
  bound and should use copy-engine DMA when page payloads are large.
- Preserve per-layer H2D completion so the first layers can run while later
  layers are still loading.
- Keep writeback lower priority than loadback. A host cache hit that is needed
  for the next forward has higher latency impact than preserving additional
  evictable pages.
- Skip storing sliding-window V4 pages that cannot serve a later replay hit
  because they are outside the history alignment segment. This mirrors the
  vLLM DeepSeek V4 MLA/SWA alignment optimization and avoids wasting host
  bandwidth on dead SWA pages.
- Compact and sort page ids inside a transfer group when doing so does not
  change logical mapping. This improves descriptor locality for the batched
  copy submission.
- Track bytes per group. State groups can have tiny pages and may be dominated
  by launch overhead; compressed/history groups should dominate bandwidth.

## Risks

- Treating all V4 groups as one page id space will silently corrupt cache.
- Forgetting the ratio-4 indexer KV payload will produce sparse-indexer errors
  even when compressed attention appears to work.
- Allowing host matches without complete state groups can skip tokens whose
  compressor or SWA state was never restored.
- Promoting a pending host snapshot before all-rank writeback ack can make
  prefix matching observe bytes that are not actually present on every rank.
- Adding scheduler wait-for-loadback events would break the intended overlap
  model and can regress mixed batches.
- Host memory sizing by token count will be inaccurate because V4 group page
  byte sizes differ by ratio and by state/history type.
- MTP requires accepted-token publish discipline; draft or rejected pages must
  not enter shared L2.
