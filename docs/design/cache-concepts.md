# Cache Concepts: Prefix Matching vs. Storage

This document defines the conceptual layering of the cache subsystem: which
concepts are *logical* (token-based, storage-agnostic) and which are *physical*
(storage-based), and which components are allowed to see which. It is the
reference for naming, code placement, and layering decisions in both the C++
scheduler (`tokenspeed-scheduler`) and the Python runtime.

## Two worlds: logical tokens vs. physical storage

### Logical world (token units, storage-agnostic)

**Naming convention: every `*_granularity` quantity (`prefix_granularity`,
`block_granularity`, `checkpoint_granularity`) and `page_size` is measured in
logical tokens.** A name in this family never counts rows, blocks, or bytes;
conversely, a quantity counting storage must not borrow one of these names.

Two quantities anchor the vocabulary:

* **`prefix_granularity`** — the granularity at which prefixes are hashed and
  matched for cache reuse. It defines the *identity boundary* of cached
  prefixes: two requests share cache only at multiples of this many tokens.
* **`block_granularity`** — the number of tokens covered by one block-table
  slot (one `CacheBlock`) of a cache group. This is the unit in which
  family-agnostic code (admission, capacity planning, PD slot selection)
  addresses the cache.

`block_granularity` is the generic quantity; a group *declares* it through
one of two shapes (Python `CacheGroupSpec`), and the group's **family** is
the name of that shape:

* **Row geometry** (`rows_per_page` × `entry_stride_tokens`, exposed as
  **`page_size`**) — the `history` family: paged KV-cache consumers whose
  blocks physically hold rows of entries (per-token KV, sliding windows,
  compressed entries, compressor input tails). "Page" vocabulary is *only*
  legal here. Retention is `full_history` or `sliding_window`.
* **`checkpoint_granularity`** — the `state` family: snapshot-style groups
  (recurrent/conv state) whose blocks each hold one state snapshot taken
  every this-many tokens. Such a group has no rows and no pages; declaring
  fictional row geometry for it is a bug, not a convention. A checkpoint
  summarizes everything before it, so nothing in it ever slides out:
  retention is always `full_history`.

The two shapes are mutually exclusive and `CacheGroupSpec.__post_init__`
holds each family to its shape, so exactly three `(family, retention)`
combinations exist — the C++ `AttnKind { kFull, kSlidingWindow,
kMambaState }`. A trailing window of token rows (DeepSeek V4's SWA kv and
compressor tails) is a sliding `history` group whatever a kernel calls its
buffer; `state` is not a routing label for "some backend owns this table".

None of these say anything about storage. A slot of `block_granularity = 64`
tokens may be backed by only 16 units of physical storage under compression —
the logical world neither knows nor cares.

A fourth quantity lives outside the logical world entirely:

* **`kernel_page_size`** — the token span of one attention-kernel page, a
  property of the *kernel implementation*, not of the scheduler. All kernel
  page geometry is registered in one file
  (`runtime/layers/attention/kernel_page_sizes.py`): fixed-page kernels pin a
  constant (FlashMLA = 64), constrained kernels choose within a supported
  set (trtllm-mla ∈ {32, 64}), and flexible kernels carry a chosen default.
  `config.kernel_page_size` overrides any default; deriving kernel_page_size
  from `prefix_granularity` is a category error and a bug.

  DeepSeek V4's geometry is fully
  registry-sourced: the compressed full-history chains declare
  `DEEPSEEK_V4_PAGE_SIZE // ratio` rows × ratio-token stride, the SWA
  window declares `V4_KERNEL_BLOCK_ROWS`, and the layout's field byte
  shapes are built from the kernel page constant — nothing derives from
  `prefix_granularity`. The V4 backend's scalar is therefore a true
  registry-sourced `kernel_page_size` (config-overridable), and the
  scheduler grain is free: any positive multiple of the kernel page is
  accepted (asserted at backend construction and in the recipe's
  `check_layout`). The V4 architecture spec defaults P to exactly one kernel
  page by naming `DEEPSEEK_V4_PAGE_SIZE` as its
  `default_prefix_granularity` — the registry constant itself, not a second
  copy of the number.

### Physical world (storage units)

* **`CacheBlock`** (and its aggregation, the **LCM block**) is the unit of
  physical storage. Allocation, refcounting, eviction, and tiering operate on
  physical blocks.
* A `CacheBlock` is attention-agnostic storage: it can be *viewed* by
  KV-cache-based attention (as paged KV entries) and by state-based attention
  (as a state slot). The view is defined by the consumer, not by the block.

`BlockPool` owns the physical placement indexes: the FIFO of empty LCM blocks,
the free child-slot count for each cache group, and per-bucket ordered sets of
partially filled LCM blocks. C++ group ids are dense scheduler indices, so
the scheduler supplies the complete packing vector when it constructs each
pool. The per-group placement records form a vector indexed by group id, and
each `GroupAvailability` stores immutable slots-per-parent geometry. The
coordinator registers each group's shard count before allocation; an ordinary
standalone pool fixes a single bucket on first use. Registration cannot change
the geometry after it is fixed, including after all blocks are freed. The pool
updates its free-slot count and bucket indexes together on every occupancy
transition. A parent with zero occupants is unbound, so its capacity
belongs to the global empty-parent FIFO rather than any group.

The pool knows which child slots are occupied, never who holds them. Whether a
child is pinned by a request table, published by a prefix-cache entry, or held
by an in-flight transfer is a `CacheBlockRef` ownership fact that lives with
the holders; the pool does not track it, and `CacheBlockRef` does not report
it. Anything that needs "held only by the cache" asks `PrefixCacheIndex`
(`ParentIsFullyEvictable`), which is a scan and is therefore reserved for
eviction policy and leak checks, not for per-step accounting.

Admission first checks the indexed free-slot and empty-parent counts. If the
request fits, it does not enumerate eviction candidates. Under memory pressure,
it enumerates candidates and keeps shadow occupancy only for the parents whose
children it tentatively evicts. This makes the common zero-eviction path scale
with cache groups and request demand rather than total cache capacity.

The per-step page gauge composes two O(1)-or-cheaper quantities the same way:
empty parents come from the pool, active parents from the live requests' block
tables, and cache-only residency is the remainder `total - empty - active`.

## Who is allowed to see what

### C++ scheduler: schedules in logical units

Scheduling decisions — admission, prefix matching, chunk alignment, capacity —
are made in **exactly two token-based quantities**: `prefix_granularity` and
the per-group `block_granularity` (`CacheGroupSpec.block_granularity`,
wrapped by the coordinator's `GroupGeometry`; both declaration shapes fold
to it at the bridge — see below). Physical geometry (LCM packing, storage
counts, bytes) is confined
to the scheduler's cache/allocator layer; scheduling, FSM, and
config-consuming code must not reason about it.

**The identifier `page_size` must not appear anywhere in
`tokenspeed-scheduler`.** The scheduler has no kernel pages and no row
geometry — its only slot-span word is `block_granularity`. A `page_size`
showing up there means a paged-KV concept is leaking across the boundary;
name it `block_granularity` (generic span), `prefix_granularity` (identity
span), or keep it on the Python side where the page actually exists.

**Nor does the word "paged" belong in cache-group type names, on either
side of the bridge.** A cache group is not necessarily paged: a snapshot
state group has no rows and no pages (see the two declaration shapes
above), so a `Paged`-prefixed group type is a claim its own contents
contradict. Three types carry one group across the boundary, and none of
them says "paged":

```
Python  CacheGroupSpec     declaration shape (rows | checkpoint) + policy
  ↓     pool_to_cache_groups                      the single folding point
C++     CacheGroupConfig   boundary config, nanobind-exposed (SchedulerConfig.cache_groups)
  ↓     MakeSpecsFromConfig
C++     CacheGroupSpec     scheduling form (block_granularity + kind)
```

The first and third share a name and differ in fields, so always qualify
which side you mean; `CacheGroupConfig` in between is the only one visible
from both.

Declaration-shape vocabulary stops at the bridge; only the generic span
crosses it. Neither `checkpoint_granularity` nor `rows_per_page` /
`entry_stride_tokens` enters `tokenspeed-scheduler` at all: they are
Python-side *declaration shapes* on `CacheGroupSpec`, and the bridge
(`scheduler_utils.pool_to_cache_groups`) folds both to
`CacheGroupSpec.block_granularity` before constructing the C++
`CacheGroupConfig`, whose only span field is `block_granularity`. So a
snapshot group's `block_granularity` equals its `checkpoint_granularity`
numerically, a paged group's equals `rows_per_page × entry_stride_tokens`,
and the scheduler has no "checkpoint", "row" or "stride" word — only "how
many tokens one block-table slot covers". Carrying a fictional
`(rows = P, stride = 1)` across the bridge for a snapshot group would be a
claim its contents contradict, the same way a `Paged` type name would.

`CacheGroupSpec.block_granularity` is **required and explicit**: a positive
divisor of `prefix_granularity`, rejected by `SchedulerConfig::Validate()`
before construction and asserted again at coordinator construction. There is
no zero-means-default fallback — every group states its span.

One flag crosses the bridge beside the token quantities: `replayable`, set
only on a sliding History group that is **regenerated by bounded replay**
instead of being prefix-cached (DeepSeek V4.1's SWA rows and compressor
tails). Before a prefix hit the model re-feeds the group's whole
`sliding_window_tokens`: retention keeps exactly what the queries after the
boundary read, and every retained page must be regenerated, so the window
the scheduler re-feeds is the retention window itself rather than a second
declared number. False means an ordinary cached window; the flag changes
nothing else about the group's geometry.

The block tables the scheduler emits are logically *indexed*: row *i* of a
request's table covers tokens
`[i * block_granularity, (i + 1) * block_granularity)`. The
entry *values*, however, are **`CacheBlock` ids** — handles to the physical
storage the cache layer allocated. The scheduler owns allocation, so its output names that storage directly.
Consumers outside the cache layer treat the ids as opaque.

With DCP virtual-block placement, `CacheBatchMetadata` validates exported IDs
against each group's **virtual** block count. The physical page count bounds
local arena storage only; applying it to the scheduler table would reject valid
remote-owner IDs before the runtime can translate them.

#### Snapshot-state prefill checkpoints

Logical width does not imply dense physical residency. Full-history KV and
retained sliding-window rows materialize every block their kernels read, but a
full-history snapshot-state prefill normally needs only its input, aligned
prefix output, and final continuation checkpoints. When the extent contains an
internal checkpoint, one forward materializes it and the final output, with
capacity for both secured at admission. A completing forward on a decoding
role also reserves growth storage, which is not an additional computed state
([Scheduler §1.2](scheduler.md#12-state-checkpoints-one-forward)). The next
decode admission rolls the expired input block forward, including under overlap
scheduling. The table keeps absolute slot positions while representing other
skipped intermediate checkpoints as null holes (`0`). State consumers may gather only the declared
input/output slots; compacting the row or publishing an unwritten intermediate
checkpoint would break position identity.

Publication requires provenance, not just an allocated block or completed hash.
The request's cache progress keeps every aligned boundary whose state is
written but not yet hashed: the checkpoint a scheduled prefill materializes
(carried through the same ordered-forward contract as its token progress) and
each accepted endpoint a landed result stops on. Verification commits only its
accepted endpoint; a boundary it crosses proves nothing and is never recorded.
Admission, finish and retraction pass the list to the coordinator, which
publishes each recorded boundary inside the newly hashed range and no other —
several in one admission when the overlap schedule lands two results back to
back — before retention can reclaim the slot. Entries leave the list only after
the admission that hashed them succeeds, so a failed attempt retries. The
hashed range comes from `Request::NumComputedTokens()`, which is exact under
any verify width ([Scheduler §5](scheduler.md#5-invariants-a-change-must-preserve)),
so an aligned accepted endpoint is hashed by the very next admission rather
than after a lag of up to the verify width. Remote endpoint-only landings do
not claim an internal prefill checkpoint, only their endpoint when aligned.

Snapshot selection and slot addressing are distinct even within this mapping:
the last internal reusable checkpoint is at
`floor(after / prefix_granularity) * prefix_granularity`, strictly between
`before` and `after`. Its slot is `(checkpoint - 1) / block_granularity`.
The two granularities need not be equal. Selecting the snapshot by a smaller
state-group span would leave the prefix-boundary state unwritten while the
coordinator publishes it. With smaller state blocks, admission also accounts
for the whole materialized suffix through the endpoint and its reserve.

When an extend materializes internal state checkpoints, convolution windows
are assembled directly into their destination blocks by the kernel package.
The prefill path calls that batched write before causal convolution updates
the final continuation state; no write is needed without an internal checkpoint.
At most one internal checkpoint per eligible request is selected. Its token
position and packed-prefix metadata are computed once on the host; the GPU
views share one immutable, pinned asynchronous upload. The reusable
`runtime.utils.tensor.upload_packed` helper aligns each typed view and owns a
fresh staging buffer per call, so preparing the next forward cannot overwrite
an upload still in flight. Checkpoint writers widen page IDs to int64 before
multiplying by pool strides; an int32 page ID can address an element offset
beyond the int32 range. Decode has no such batch
and does not compute checkpoint indices. Recurrent execution partitions the
batch into a body scan and a tail scan when an internal checkpoint is needed.
Otherwise the ordinary prefill scan runs once. `_run_prefill_recurrent` owns
both cases and always returns outputs and final states; its caller writes the
final states to the continuation blocks. Target verification remains separate.
Every request in a checkpoint
batch participates in the body: rows crossing a checkpoint stop at that
boundary, while other rows run to completion. Only crossing rows enter the
packed tail scan, initialized directly
from their body final states. Body and tail outputs are scattered back to
original token order, so every token is evaluated exactly once while the aligned
and final states are both retained. Batch size one uses zero-copy body/tail
views; larger batches use the same batched pack/scatter contract. This split
does not change cache ownership or scheduler metadata.

The body/tail split uses the existing prefill-op state-layout contract. KDA
solutions may retain their original K-major Python adapters; the kernel facade
converts between that layout and the runtime's V-major state slab on both scans.
Native-layout adapter entry points are not required for checkpoint continuation.
KDA's int64 sequence boundaries are still prepared once in runtime metadata and
reused by the original adapters. Preparation uses `Tensor.to(torch.int64)`;
an already-int64 boundary is returned unchanged without allocating a copy.

Scan results may be transposed views in that public state layout. The batched
input packer addresses recurrent-state rows and features using their actual
strides, including on CUDA graph replay; it does not require an extra
contiguous copy or a different kernel adapter.

Speculative KDA verification stores no per-position recurrent states: it
captures each window's raw projections in a compact payload and commits by
replaying the accepted prefix from the committed page. The Kimi-K3 recipe
reserves that workspace before sizing the arena — the transient conv rows
plus the per-layer capture payloads — so speculative state memory does not
disappear from the GPU budget. (Platforms without the replay kernels fall
back to the dense `max_bs * (draft_tokens + 1)` per-position state
workspace, reserved the same way.)

### Python runtime: maps logical to physical, perceives as little as possible

The Python side owns the translation from the scheduler's cache-block tables
to the kernel page tables that attention kernels consume (the
`block_granularity → kernel_page_size` subdivision). This mapping should
happen at **one designated point**; beyond that point, kernels see physical
page tables and nothing upstream sees them at all.

Outside the mapping point, Python code should perceive `prefix_granularity`
and `page_size` as little as possible. If a Python component needs either
value, that is a design smell to justify, not a default to reach for.

Provenance discipline: each quantity is sourced from its own domain and never
laundered through another's name. The contract's `prefix_granularity` comes
from the memory plan, not read back out of pool state. The arena carries
**two** scalars with distinct roles: `CacheArena.prefix_granularity` is the
identity grain, used for contract publication and plan-consistency checks.
The state-checkpoint mapping point additionally uses the contract's identity
grain to select the snapshot required for prefix reuse; this is not kernel
geometry. Other runtime arithmetic must not reach for it.
`CacheArena.kv_page_size` is the KV arena
page span that paged-KV geometry math (row views, slot↔page arithmetic,
scale-tile branching) reads. Both derive from the one plan, which is the
single point of the prefix-page ↔ KV-page convention.
Neither is a statement about per-group CacheBlock geometry: group spans live
in the specs as `block_granularity`, and blocks narrower than P (V4's SWA
window, state checkpoints) are the norm, not the exception. A backend's
`kernel_page_size` comes from the kernel registry or an explicit config
override, never from `prefix_granularity` as a fallback. The CLI flag is
`--prefix-granularity` (`--block-size` remains a deprecated alias).

Layerwise L2 load fences guard the first access to every field owned by a
layer, not just the conventional paged-KV buffers. Model-owned side caches
(for example QSA raw/compressed keys and positions, or PLE state) must wait on
the pool's layerwise load tracker before their first read or write. A wait in a
later attention-buffer accessor is too late: the side-cache kernels would
already be racing the asynchronous H2D restore. Place the fence immediately
before the first cache-field access so independent projections can still
overlap the load.

A batched post-verification commit is the one side-cache writer that issues no
fence of its own. It moves every bound layer's fields in a single launch, so it
cannot fence per layer, and it relies on two invariants instead: it runs after
the whole model forward, by which point every bound layer has already fenced
this step in its own forward, and under pipeline parallelism only the layers
this rank forwards are bound to it. Moving such a commit earlier than the end of
the forward, or binding layers the rank does not forward, breaks the fence
guarantee without any call site visibly dropping a wait.

The Host-transfer workspace resolves transport capability for its buffer
binding before publishing consumer waits. NVIDIA mapped-Host transfers can
use a single full-geometry H2D launch with per-layer ready flags; other paths
publish per-layer events. Device-resident geometry is metadata, not evidence
that a particular completion protocol is supported. Automatic DMA fallback
is limited to unavailable Host-pointer mapping, scoped to that workspace and
buffer binding; validation, allocation, and kernel-launch failures propagate.
Transfer callers explicitly select the backend, staging synchronization,
grid cap, and optional layer flags. For an unflagged layer with no matching
blocks, the transfer boundary validates the loaded block count and returns
before mapping Host pointers or touching the accelerator runtime. Flagged
loads still publish readiness for empty consumers.

Every L2 copy is ordered after the prerequisite stream the caller names per
submission: the stream whose completed work the copy must observe. Writeback
runs on the executor's write stream, ordered after the model executor's
execution stream, where the forwards wrote the source pages. (Page zeroing
likewise orders itself behind that stream inside `zero_cache_pages`; there is
no caller-side fence to remember.)
Each op says how the scheduler guards its Device sources
(`source_pinned`, see `scheduler.md` §2). A pinned op's sources stay cached
and unevictable until the ACK, so its copy overlaps whatever the round does
next and nobody waits on it. An unpinned op's sources may be re-granted in the
same plan, so it is launched first and the fence stream the caller names --
the default stream, where the plan's page zeroing runs -- waits on its
completion event before the zeroing is enqueued — the zeroing, load-backs,
forwards and RDMA triggers behind it inherit the fence (the forward by
waiting on the default stream in its prologue; that wait is one-way, the
zeroing's and the writeback's own waits are what order the default and write
streams behind the forwards). A load-back's prerequisite stream is the
default stream that zeroed its destinations; its start event is recorded
there rather than on whatever stream is current. Every stream the L2
executor orders against is an argument, never ambient thread state. The
two kinds use separate staging lanes: each lane uploads block metadata
asynchronously and records an event after both metadata copies to protect its
pinned CPU staging tables — before refilling them, the next submission on
that lane waits only if that event is incomplete. This does not wait for the
payload transfer or publish a writeback ACK. Device metadata reuse stays
ordered after the previous payload by write-stream FIFO. Even a partially
submitted metadata upload records its retirement event before propagating a
staging failure.

Ready flags are valid only for a full-geometry H2D transfer. Consumers first
wait for the current generation's flag initialization event, then its layer
flag. Workspace reuse also waits for the previous transfer's completion;
layer readiness alone does not retire metadata still read by the transfer.
On submission failure, retirement fences protect reuse without publishing a
successful load ACK. If neither event publication nor stream synchronization
can establish retirement, the executor must reject further loads.

An op is acknowledged only by its copy's completion event, so every op on the
wire carries at least one transfer, and no (group, source, destination)
repeats within one plan: a store skips keys already in flight, a load targets
freshly acquired pages. The scheduler asserts both when batching a plan's ops
and the runtime refuses an op with nothing to copy; neither side dedups or
invents an acknowledgement, because an op that never completes a copy would
hold its tickets forever.

## block vs. page

**`block` is the general concept; `page` is its specialization under
KV-cache-based attention.** A block is one addressable cache unit of a
group — one block-table slot, one `CacheBlock`, spanning `block_granularity`
tokens — regardless of what it holds. A page is a block whose contents are
rows of KV entries consumed by paged attention kernels. Every page is a
block; a state-checkpoint block is not a page.

Every pairing in this document is a corollary of that relation:

* `block_granularity` (generic slot span) vs. `page_size` (the row-geometry
  reading of it);
* `block_table` (generic container) vs. `page_table` (the paged-KV kernel
  table);
* `CacheBlock` (storage unit) vs. "cache page" (that unit viewed as paged
  KV).

Naming rule: reach for a `block` word by default; a `page` word asserts that
the consumer is paged KV-cache attention, and is wrong anywhere that
assertion doesn't hold.

## page_table vs. block_table

The two names are distinct concepts, not synonyms:

* **`page_table`** — exists *only* for KV-cache-based (paged) attention. It
  maps logical token pages to cache pages. State-based attention has no pages
  and therefore no page table.
* **`block_table`** — the table used by state-based attention (e.g. Mamba /
  linear attention state slots), and the generic name for the container that
  carries per-group tables between scheduler and runtime.

Use `page_table` when **(and only when)** the consumer is paged KV-cache
attention; use `block_table` otherwise.

## The prefix layer and the cache group (`csrc/cache/prefix/`, `csrc/cache/allocator/`)

One attention structure (full attention, SWA, Mamba state, …) is one **cache
group**, and a group is built from three single-purpose pieces plus its spec:

```
CacheGroup = CacheGroupSpec          what the group is (kind, slot geometry)
           + GroupAllocator     physical placement  (cache/allocator/)
           + PrefixMatcher        match policy        (cache/prefix/)
           + PrefixCacheIndex     reuse index         (cache/prefix/)
```

`CacheGroup` is the **only place the allocation and prefix-matching concerns
meet**; neither side holds the other's data structures. The group's token
arithmetic (`GroupGeometry`) lives one level up, in the coordinator.

Perception rules per directory:

* **`cache/prefix/` may perceive `prefix_granularity` and the group's slot
  span** — prefix matching is defined in token space, so its hashing,
  key expansion, and window-resume arithmetic legitimately speak tokens.
* **`cache/allocator/` perceives no logical token quantity at all** — no
  `prefix_granularity`, no `block_granularity`, no windows. Its vocabulary
  is blocks: `CacheBlock`, packing (`cache_blocks_per_lcm_block`), pool
  slots, block counts.
* The conversion between the two lives in the **coordinator's
  `GroupGeometry`** (`cache/coordinator/group_geometry.h`).

### `cache/prefix/` — what is reusable

* **`PrefixCacheIndex`** (`prefix_index.h`) — the CacheKey → canonical
  `CacheBlock` index, extracted from the old `GroupAllocator`. It owns
  register/lookup/evict/pin (`Register`, `RegisterFullBlocks`, `Contains`,
  `Find`, `Evict`, `AcquireMatched`, eviction metadata). Indices are
  pool-scoped: one index serves both the Device and Host tiers of its group. A
  cache entry is metadata plus an owning `CacheBlockRef` that publishes an
  existing block for reuse; registration neither allocates nor copies physical
  storage.
* **`PrefixMatcher`** (`prefix_matcher.h`) — the per-attention-kind match
  policy, extracted from the old manager subclasses. `FullAttnMatcher` walks
  left-to-right until the first miss (prefix-closed); `SwaMatcher` scans
  right-to-left for a run backing a resumable boundary (non-closed). Mamba
  needs no matcher of its own: it is `SwaMatcher` at window 2 — "keep the
  live state page plus its snapshot". A matcher only *reads* the group's
  index; it never touches allocation or physical placement.
* **`prefix_hasher.h`** — SHA-256 prefix-page hashing (moved from
  `scheduler/`).

### `cache/allocator/` — where things live (token-free)

`GroupAllocator` is **physical placement only**, and there is exactly one of
it — no subclasses. It moves `CacheBlock`s between the `BlockPool` and
`BlockTable`s (`Acquire`, `AppendHostExtension`, `Free`), resolves kernel
page ids, and executes retention (`ReclaimExpired` punches the first *N*
slots to null holes). It is deliberately token-free: every token quantity is
converted to block counts before it reaches the manager.

The conversion is `GroupGeometry` in the coordinator layer:

* `PlanAcquire(table, demand)` turns a token demand into a token-free
  **`AcquirePlan`** (`cache/core/acquire_plan.h`) — block counts plus the
  bookkeeping values the manager stores verbatim; the manager executes the
  plan without deriving anything.
* `ExpiredBlocksAt(spec, num_computed_tokens)` is the retention *policy*
  (full attention never expires; SWA and Mamba-at-window-2 slide out whole
  pages); the manager only *executes* the resulting block count. This is
  what dissolved the old `SwaManager`/`MambaStateManager` subclasses.

Where reclaim needs to know whether a block is still cached, it takes the
group's `PrefixCacheIndex` as an explicit read-only parameter — the
dependency is visible in the signature, not hidden in shared state. The
reverse direction is symmetric: publishing a table's completed blocks may
replace one with the key's existing canonical block, and that write goes
through a mutable window the allocator hands out
(`GroupAllocator::BlocksToPublish`) to `PrefixCacheIndex::RegisterFullBlocks`.
The index never sees a `BlockTable`; the allocator remains its only mutator.

## The coordinator layer (`csrc/cache/coordinator/`)

The coordinator is the scheduler's *sole* entry point into the cache
subsystem — the facade that hides "multiple attention structures, one shared
physical pool, two storage tiers" behind a token-unit request lifecycle:
probe → admit → publish → free.

### `CacheCoordinator` (`cache_coordinator.h`)

A model may mix attention kinds (full attention, SWA, Mamba state, …); each
becomes one `CacheGroup` (manager + prefix index + matcher). The coordinator
fans every request-level operation out across all groups, which share a
single `BlockPool` of LCM blocks, and folds the results back into one answer.
It holds no per-request state; it only advances the global access-epoch
clock, with each request carrying its issued epoch.

Its responsibilities:

* **Prefix probe and admission.** `ProbePrefix` is a read-only lookup of
  prefix hits per group on both tiers, converged to the common prefix length
  across the *matched* groups; `Admit` then allocates, pins the hit prefix,
  and produces host→device `load_pairs` plus each group's fresh pages. A
  **replayable** group (`CacheGroupSpec::replayable`) is not matched
  at all: it never constrains the boundary and never claims a hit, so its
  table is empty at first admission and the scheduler materializes it as a
  sparse private suffix from the replay window's first token
  ([Scheduler §1.3](scheduler.md#13-bounded-replay)). Probe and admit are
  deliberately split so the probe can be taken once and the admission retried
  against it — the scheduler's same-round retract-and-grant re-runs a failed
  admission after freeing a victim (see `scheduler.md`) without re-probing.
  `ProbeDecodeDevicePrefix` is the PD-decode variant: local history
  pages are reused while final-state groups are restored from the remote
  endpoint snapshot.

  `Admit` takes two inputs of different scope and tense. One `GroupDemand`
  per group says what that group needs for the round ahead: an extent and a
  reserve beyond it. The extent is one of two shapes in different reference
  frames — `DenseGrowth` appends tokens relative to the table's current fill,
  `SparseSuffix` names an absolute token extent and the first slot to
  materialize, leaving the slots below as null holes — so the bounded-replay
  rewrite in `Admit` is a visible conversion from one to the other (hit plus
  growth becomes the absolute extent), not an arithmetic side effect. One
  `RequestProgress` per
  request says what the request has done since the coordinator's previous
  transaction for it: the prefix pages it completed (`CompletedPages`, present
  only when the newly hashed range is non-empty, so "new hashes without a
  boundary kind" cannot be expressed) and its computed-token count for
  retention. Publication fields are request-scoped and therefore live on the
  progress, not replicated onto every group's demand.

  Publication rides inside `Admit` on purpose. "Completed" means scheduled
  stream order for prefill (`NumComputedTokens` is the scheduled window end;
  the FIFO data plane orders any hitter's forward after the writer) and
  landed order for decode (token ids, hence hashes, exist only after
  landing); the next admission is the first point after both, and one rule
  covers both. Inside the transaction it is side-effect free when the
  admission fails and is retried in the same round, it is ordered before
  retention reclaims the slots it publishes, and victim planning sees the
  pre-publication state so a request's own expired, still-unpublished tail
  can fund the same admission. `CacheCompletedBlocks` takes the same
  `RequestProgress` for finish, retraction and remote completion, which
  publish without admitting.
* **Prefix publication.** `CacheFullBlocks` / `CacheCompletedBlocks` register
  computed blocks into the prefix indexes for later requests. Prefix-closed
  groups match first; non-closed groups (SWA, Mamba) match only within the
  boundary the closed groups settled (`match_order_` enforces this).
  For Mamba-state groups, `CacheCompletedBlocks` publishes only explicitly
  listed materialized boundaries inside the newly hashed range; an empty list
  publishes no state snapshots (see
  [Scheduler §1.2](scheduler.md#12-state-checkpoints-one-forward)).
  Replayable groups are outside `match_order_` and skip publication on both
  tiers — never registered, never streamed to Host, never counted by
  `DeviceBoundaryResidency` — because their rows are approximations the
  model regenerates from re-fed tokens, not a function of the prefix alone.
  Retention (`ExpiredBlocksAt`/`ReclaimExpired`) still slides their
  request-private pages out.
* **Two tiers (Device/Host).** Device prefix publication can optionally
  stream to the Host tier (`stream_device_cache_to_host_`); a
  `pending_stores_` queue drives D2H transfers, alongside Host-side
  acquire/contains/pin queries. During prefill, each completed scheduling
  boundary queues all newly published full-attention pages and one checkpoint
  per snapshot-state group; the pending candidates are merged into a batched
  writeback. The first decode admission from `PrefillDone` applies the same
  policy to the final prompt boundary. Ordinary decode still publishes Device
  entries but does not stream full-attention or snapshot-state entries to
  Host. At finish or retraction, all eligible Device-resident non-state pages
  and only the newest Device-resident checkpoint per state group are queued
  before request ownership is released. Ordinary sliding-window entries
  always stream when published. The queue is drained by
  `TierTransferManager::StartPendingStores(guard)`: every store but a
  retraction's snapshot pins its Device sources until the ACK; the snapshot
  store is stream-ordered instead, because its sources are re-granted in the
  same round (`scheduler.md` §2).
* **Reclamation and lifecycle.** `ReclaimExpired`, `Free`,
  `ClearDeviceCache`/`ClearCache`, and `NumNewlyReleasableLcmBlocks` for
  ranking retraction (preemption) victims.
* **Mutation reporting.** `SetCacheMutationSink` reports per-group cache
  insertions/removals; the scheduler folds them into one externally visible
  prefix event. Whether a scheduler-level boundary is fully, partially or not
  resident is the coordinator's answer (`DeviceBoundaryResidency`, read off
  the group indexes), so the scheduler keeps no residency counters of its own
  — only the token descriptor the event carries and whether that event is
  currently out.

`MakeCoordinator` is the factory: one `CacheGroup` per `CacheGroupSpec`
(group_id = index), all sharing one scheduler-level `prefix_granularity`
while each manager may use a smaller cache-page token count.

### `AdmissionPlanner` (`cache_admission.cpp`, anonymous namespace)

The internal capacity planner behind `Admit`. It runs entirely on shadow
occupancy — never mutating the real pool — and answers: *which cached blocks
must be evicted for this admission to fit, while protecting the current
prefix hits?* The algorithm: first check whether existing local holes plus
empty parents fit with zero eviction; otherwise select eviction candidates
until the plan fits. Request-reclaimable candidates are collected and sorted
once; each cache group loads and sorts one epoch of eligible candidates at a
time, skipping epochs whose entries are all protected or already listed for
request reclaim. Selection compares the next candidate from each group with
the next request-reclaimable candidate using one policy: LRU access epoch,
then tier (uncached request-only block → probationary boundary → established
boundary → suffix of a closed prefix). Finally, walk the selected blocks in
reverse and restore every block that is not strictly required, yielding a
minimal eviction set in `victims`.

Each cache group uses a non-owning cursor over its tier's eviction index.
It advances continuously, skips fully pinned epochs, and returns complete
epochs for policy ordering. Cursors exist only during one read-only planning
pass: their index and pool must remain alive, and entries must not be inserted,
erased, or re-keyed during traversal. Commit-time index mutations happen after
the planner is destroyed. A full traversal costs O(N), including pinned entries,
without a separate tree lookup per epoch.

## The cache pipeline: layers → group → pack → bind

Every model family's cache is built by the same four-stage pipeline, and the
stage names are the vocabulary:

```
layers ──group──▶ groups ──pack──▶ CacheLayout ──bind──▶ CacheMemoryPlan
```

* **layers** — the family's layer vocabulary: a `layer_types` label and a
  `group_ids` assignment per layer, target layers then draft layers. The
  labels are the *storage* vocabulary (`AttnConfig`'s `cache_layer_types`,
  resolved once by `configs/base.py:resolve_cache_layer_types`), never the
  checkpoint's compute labels: a label names what the scheduler retains, and
  a layer's compute mask is a separate contract (see *Storage vs.
  visibility* below).
* **`group`** (`recipes/spec.py`) walks those layers **once** and returns
  `(CacheGroupSpec, fields)` pairs — one per distinct group. The pairing is
  the point: a group id is spelled exactly once, in its spec, next to the
  fields that deposit bytes in it.
* **`pack`** (`recipes/plan.py`) decides how one physical parent is laid out:
  plane sizes, per-field offsets and page strides, and how many of each
  group's CacheBlocks share a parent (`cache_blocks_per_lcm_block`). The
  result, `CacheLayout`, is **capacity-independent** — it describes one
  parent, not an allocation.
* **`bind`** (`CacheLayout.bind(num_lcm_blocks)`) multiplies that parent out
  by a count and yields the `CacheMemoryPlan` the arena allocates from and
  the PD wire carries.

`CacheRecipe` (`recipes/base.py`) is a template method: `setup()` is the one
place the four stages appear in order, and a family fills in uniformly named
seams — `layer_types`, `group_ids`, `fields_for_layer`, `prefix_granularity`,
`alignment`, `max_padding_fraction`, `packing`, `check_layout`,
`num_lcm_blocks`, `token_capacity`, `parents_needed`, `workspace_bytes`,
`pool_options`. `groups()` itself is a seam for the two families whose groups
are not per-layer (Inkling appends conv columns; V4 declares each group
whole). No family restates the order of the stages, and `_RECIPES`
(`recipes/setup.py`) is the single family → recipe map.

**No round-trip reconciliation.** The pipeline is arranged so that pairs which
would otherwise need cross-checking cannot differ:

* `setup()` obtains `(spec, fields)` pairs from `groups()` and uses the same
  local tuple for `pack` and spec publication, so both name the same group
  set without a separate cache of declarations;
* a field cannot name a group the plan does not have, because it never names
  one — `pack` carries the declaring group id alongside each field;
* per-group packing is read from the layout, not recomputed, everywhere
  downstream (the C++ bridge, the runtime contract, capacity math).

If you find yourself writing a check that two derived views agree, the
design is wrong: make one of them the source.

### Storage vs. visibility

An attention layer has two contracts that a single `layer_types` string used
to carry at once, and they are kept apart on purpose:

* **Storage** — which cache group the layer's KV rides, hence how long the
  scheduler retains it (`CacheGroupSpec.retention`, `sliding_window_tokens`).
  Owned by the cache plan: the recipe assigns `group_ids` per layer, `pack`
  places the fields, and the model never spells a group id. At executor
  startup `bind_cache_groups` (`layers/paged_attention.py`) reads
  layer → group back from the planned KV fields
  (`CachePool.history_group_by_layer`) and stamps it onto each
  `PagedAttention`; before that the layer's `group_id` is unbound and any read
  raises.
* **Visibility** — how far back the kernel may look. Owned by the layer
  (`PagedAttention.sliding_window_size`, a `window_left` mask the model derives
  from its own config) and read by the backends only as a kernel argument;
  it never influences tables or write locations.

The one relation between them is an inequality, not an equality: **a group
must retain every token its layers can see.** `bind_cache_groups` enforces it —
a full-visibility layer cannot ride a sliding group, and a sliding mask must
fit inside the group's retention window (`window_left + 1 <=
sliding_window_tokens`, matching `GroupGeometry::ExpiredBlocksAt`). Everything
off the diagonal is therefore legal by construction rather than by special
case: a sliding-masked layer on a full-history group (a block drafter, DSA's
sparse compute over a fully retained cache) simply retains more than it reads.

A third contract appears when a group leaves prefix caching. DeepSeek V4.1
declares its SWA rows and compressor tails **replayable**
(`CacheGroupSpec.replayable`, `recipes/deepseek_v41.py`): a prefix
hit never shares them, so the scheduler re-feeds the cached prefix's last
window and the backend regenerates them into the request's private pages.
The rows it re-feeds carry `extend_replay_lens_cpu` down the extend bundle
(`unified_path.md`), and the backend derives two things from it:

* **Visibility for replayed rows** starts at the replay window's first
  token (`V41PrefillSpan.swa_prefix_begin`): a re-fed row may not look back
  into rows that were never regenerated, and the new rows after a full
  window need nothing older than the window, so their view is unchanged.
* **A write floor for the shared groups** (`V41Metadata.global_write_floor`
  = prefix + replay per request): the replayed rows recompute the global KV
  and index rows the hit already holds in shared pages, and `write_global`
  masks every compression group whose last position lies below the floor
  (ratio-aware: a ratio-2 pair straddling the floor is written). The
  shared rows therefore stay exactly what the first computation produced;
  only the private groups are rewritten.

The same backend narrows the CED decoder to each request's prompt tail
(`decoder_view()`); the decoder's SWA rows are decode-only state and, being
in the replayable group, are never expected from a hit either.

Block drafters (DFLASH / DSPARK) write their KV at the target's cache
locations, so their storage *is* a target-owned group whatever mask their
layers apply. Drafts with their own attention layers ride the target's
full-history group: `resolve_cache_layer_types` labels every block-draft layer
full-history, and `check_block_drafter_storage` verifies at startup that the
group the draft bound is one the target's own layers share — a target without
a full-history group has nothing for a block drafter to borrow. DeepSeek
V4.1's same-checkpoint DSpark has no draft attention layers of its own; its
per-stage context rows are extra fields of the target's SWA group on the last
target layer, addressed by the target's SWA slots, so they are transferred
and evicted together with the SWA rows and the draft's 128-row window fits
inside that group's retention. The SWA group is replayable, so a prefix hit
regenerates them with it: the drafter writes one row per row of the target's
`decoder_view()` — each prompt's kept tail, then the decode rows — which is
also the layout of the taps it reads, and the kept tail is exactly the
window the first decode consumes. Under PD the retained tail of the SWA group
ships to the decode node like any sliding window, draft rows included, and
the decode node re-feeds nothing.

Capacity has exactly two shapes, both on the base class. The default is the
flat product (`parents × tightest packing × P`). Families whose per-group
demand decides the pool — K3's state groups riding inside MLA planes, V4's
SWA and compressed chains, GLM-5.3-Flash — size from `parents_needed` and get
the inverse for free from `_capacity_from_parents`, one monotonic binary
search shared by all. `parents_needed` itself is not a Python formula: it
hands the group specs, the layout's virtual packing and `scheduler_limits`
to the scheduler's own `CapacityModel` (`recipes/scheduler_bridge.py` →
`csrc/scheduler/capacity_model.h`), whose `ConcurrentGroupPages` reports each
group's demand at the configured concurrency and whose `LcmBlocksNeededFor`
folds it by packing. The per-request working set — decode reservation,
overlap-protected step, a state group's checkpoints and banked growth, a
sliding group's lookback and resident window — is therefore defined once, in
C++, and the `Scheduler` bounds single requests against the pool with the
same model (`docs/design/scheduler.md` §1.4). No recipe restates any of it.
`scheduler_limits` is the single place a recipe reads the scheduler's
concurrency, role and reserve widths, so demand and capacity cannot size
against different numbers.

The runtime's global `max_num_seqs` is divided across attention DP ranks to
produce each scheduler's rank-local `max_batch_size`. These values limit
simultaneous sequence slots; they do **not** reserve enough history cache for
that many maximum-length requests. Aggregate prompt and decode growth must
still fit the recipe's reported token capacity. When that dynamic pool is
overcommitted, admission can fail even though the batch still has a free
sequence slot.

When admission fails for capacity and no prefill can progress, the scheduler
retracts a resident victim and grants the freed pages to the blocked request
within the same plan build; what stops an overcommitted workload from
repeatedly rebuilding, briefly decoding and re-retracting the same prompt is
the escalating admission headroom each retraction adds to the victim's next
admission. The protocol — victim choice, readmission order, why the release
is safe before the L2 snapshot copies — is `scheduler.md` §2 and §4.

## Virtual block placement within a shared physical plan

A recipe declares each group as a `(CacheGroupSpec, fields)` tuple.
`CacheGroupSpec.shard_count` defaults to 1 (replicated); a larger value assigns
virtual blocks cyclically across that many owners. The memory plan continues
to own local shapes, strides, packing and byte counts. `CacheArena` is the sole
publisher of `CacheRuntimeContract`, whose virtual counts and packing derive
from these physical facts and each spec's `shard_count`. No separate placement
or per-group address-space object is needed.

A recipe's group set never depends on the DCP size. Only groups whose every
reader can attend to a shard may be sharded; a group some consumer must read
whole stays replicated and is declared as its own group at every DCP size, so
prefix matching, transfer and zeroing -- all keyed by group -- see one
topology. DeepSeek V4 shards its compressed-KV chains and keeps the SWA cache,
the compressor states and the indexer's K replicated; the indexer K is its own
full-history group rather than a tenant of the compressed chain it indexes.

Splitting or regrouping fields can change physical packing and parent plane
sizes. Capacity planning therefore uses the resulting physical parent byte
size and each group's declared demand: a replicated full-history group holds
one physical child per token span where a sharded one holds one per
`shard_count` spans, so the replicated groups bound the token capacity.
Virtual placement alone does not impose a fixed parent size across different
group declarations; field alignment and bounds remain the physical planner's
responsibility, and the padding bound applies unchanged.

Translation from virtual to local IDs is one operation with `shard_count` as
a parameter, never a mode: a replicated group translates to itself minus the
null block, so batch metadata refreshes its local read tables, writers mask
unowned rows, and zeroing filters foreign blocks through the same path at
every DCP size. Virtual block 0 is the null block; no path writes to it, at
any DCP size.

Consumers bind a pool's compute view and read its arena's runtime contract.
Views sharing an arena share that contract, rather than accepting separately
injected copies of its geometry. Batch metadata retains the same contract for
address translation.

For physical packing K, N usable parents and D shards, a sharded
group has `1 + N*K` local pages and `1 + N*D*K` virtual blocks. A replicated
group uses one bucket. Existing `group_page_counts` and `group_packing` name
physical quantities; the scheduler bridge explicitly consumes
`virtual_block_counts` and `virtual_packing`. Virtual capacity must never be
used to shape an arena field. Virtual null ID 0 has no owner and is filtered
during translation.

Before zeroing scheduler blocks, the runtime checks IDs against the virtual
bound and translates each group's batch to owned local IDs through the shared
translation API. Translation precedes dispatch to pool views; pools and the
arena receive physical IDs and hold no context rank. The arena's
`zero_blocks()` validates every ID against its group's local page count before
clearing any bytes. Physical page 0 is within that range and is handled like
any other page when explicitly requested.

The allocator receives only an integer `shard_count`, fixed when the
coordinator registers the group in its pools. It counts
all nonnull refs in the request table, including shared prefixes and reserved
headroom. Among available holes in already bound parents, it selects the
least-loaded request bucket (ties by bucket ID), then the most occupied
parent (ties by parent ID), then the lowest child ID. Allocation updates these
occupancies before selecting the next child. Only when all bound-parent holes
are exhausted may it open the next FIFO empty parent. Bucket balance cannot
reserve an extra parent or cause admission failure while another bucket is
available. A failed acquire leaves both placement and request tables unchanged.

`BlockPool` maintains the free-slot count of each group and an ordered parent
index per bucket. Each parent records only its lowest free slot per bucket;
it does not materialize a list of every hole. Physical `occupy` and `Release`
update these indices, including full-to-partial transitions and final-child
release. Shared request/prefix references therefore keep both the parent
binding and its index state alive until the last reference releases the block.
These indices describe physical availability, not request-local owner loads.

An exact acquire first checks `group holes + empty parents * group packing`.
Capacity-first selection can consume all of these slots, so it then allocates
directly without a second, pool-sized shadow planner. Insufficient capacity
returns before changing the indices, occupancy, or FIFO. Ordinary, balanced,
and Host allocation entry points use the same availability updates.

Choosing a block examines the bucket-index heads, not every parent. Updating
the chosen parent's ordering costs one logarithmic-time index update per
bucket the parent has a hole in. Advancing its free-slot cursor only searches
that bucket within that parent. Ordinary and balanced calls share the registered geometry;
changing the shard count or packing is rejected even after every block is
released. No allocation call rebuilds the group's indices for new geometry.
The additional metadata is per-parent bucket minima and at most one tree
entry per available bucket of a partial parent. Request loads remain derived
from `BlockTable` on each actual acquire; this optimization introduces no
request counter that retract, prefix replacement, or table clearing must reset.


## Sparse indexers: model weights, backend dispatch and verification

* The **model** owns indexer weights and top-k selection, passing
  `topk_indices` through `PagedAttention.forward` to the backend.
* The **full-attention backend** owns attention dispatch and full-KV cache
  addressing. `CacheGroupRouter` expands only the groups served by attention
  leaves; QSA's compressed/recent history groups are separate consumers.
* **`QSAIndexerBackend`** owns those two groups' raw block tables, query
  lengths and transient verification workspace. It uses the shared table
  fill at expansion ratio one, preserving block ids and clearing padding,
  and borrows the full-KV address view from the router. Its private
  `QSAVerifyState` exists only for a speculative target with local QSA fields.
* **`Qwen4ExpBackend`** composes an attention backend with optional PLE and
  indexer consumers. Its attention child uses the ordinary hybrid only for
  views with GDN layers; draft views have neither GDN nor PLE.
  The runner's existing post-verify calls dispatch once
  to their respective children; the root neither allocates verify tensors
  nor registers or looks up QSA state. PLE owns its checkpoint metadata
  independently of Mamba and shares only the checkpoint arithmetic. See the
  [execution lifecycle](unified_path.md#backend-package-layout).

LCM owns persistent allocation, prefix matching, transfer and retention,
including QSA's full-KV, compressed and recent cache groups. Verify-state layer
ownership and cache addresses come from the bound plan's layer window,
including under PP and target/draft sharing. Cache recipes reserve verify
workspace before sizing the arena.

Qwen4-Exp selects its cache recipe by model family, including targets with
only full-attention layers. PLE and QSA fields and their verify budget do
not require a GDN component. Recurrent shapes and replay settings are read
only when that component exists; a recurrent layer label without matching
linear-attention geometry still fails during recipe construction.

## Code placement

* Prefix-matching code (prefix hashing, match/lookup, reuse boundaries) lives
  in its **own directory**, isolated from allocator/storage code. Prefix
  matching decides *what* is reusable; the allocator decides *where* things
  live. Neither should be entangled with the other's data structures.
* Allocator/storage code (`CacheBlock`, pools, LCM planning, eviction) is the
  only place physical concepts appear.

## Rules of thumb

1. If a value is in tokens, it belongs to the logical world; if it is in
   blocks-of-storage or bytes, it belongs to the physical world. Never mix the
   two in one interface without an explicit mapping.
2. The C++ scheduler schedules in tokens; physical geometry stays inside its
   cache/allocator layer. Emitted table entries are `CacheBlock` ids, opaque
   to everything outside that layer.
3. There is exactly one logical→physical mapping point in Python. Adding a
   second one is a bug.
4. New attention backends declare which view of `CacheBlock` they need
   (paged KV view or state view); they do not invent new table concepts.

## Current state vs. principles (audit 2026-08-16)

Where the code stands relative to each principle. Every claim below was
re-checked against the code at this date; a ✓ means the principle holds with
no known exception, and the exceptions that remain say so explicitly.

### Principle 1 — prefix matching isolated from the allocator: fixed

`csrc/cache/prefix/` owns the concern: `prefix_index.h`
(`PrefixCacheIndex`, the CacheKey → canonical CacheBlock index),
`prefix_matcher.h` (`FullAttnMatcher`/`SwaMatcher` policy hierarchy; mamba is
`SwaMatcher` at window 2), and `prefix_hasher.h` (moved from `scheduler/`).
`GroupAllocator` (`csrc/cache/allocator/`) is token-free physical placement
only — `GroupGeometry` in the coordinator owns the token arithmetic — and
`CacheGroup` pairs spec + allocator + matcher + index, the only place the
two concerns meet. Remaining known item: `csrc/scheduler/kv_cache_events.cpp`
still hosts a second, independent block-hash implementation for external KV
events (wire-format constrained; unify deliberately if ever).

### Principle 2 — scheduler perceives only logical quantities: fixed, now with hard vocabulary rules

Scheduling and FSM code do no geometry arithmetic. The coordinator exposes
capacity views — `NumActiveLcmBlocks(request_tables)`,
`NumAvailableLcmBlocks`, `TotalLcmBlocks`, `GroupAvailablePages(group)` — and
the config-only `CapacityModel` (`csrc/scheduler/capacity_model.h`) folds
page demand into LCM blocks (`LcmBlocksNeededFor(group_pages)`); the
scheduler treats the counts as opaque capacity units. The null-page
reservation lives in `SchedulerConfig::AllocatorConfig::NumUsableBlocks()`,
and nothing outside the cache layer enumerates LCM block ids.

Enforced:

* the identifiers `page_size`, `rows_per_page`, `entry_stride_tokens` and
  `checkpoint_granularity` are grep-zero across `tokenspeed-scheduler`
  (csrc, tests, python bindings) — the slot span is spelled
  `block_granularity` everywhere, and `CacheGroupConfig` carries it as its
  only span field;
* `CacheGroupSpec.block_granularity` is required and explicit: every group
  states its span, with no zero-means-default fallback, and the coordinator
  asserts a positive divisor of P at construction;
* `SchedulerConfig::Validate()` is the **single** configuration gate: every
  scheduler scalar, every `CacheGroupConfig::Validate()`, and the cross-checks
  between them (P divisibility, PD transfer policy, one-cache-block chunks for
  a recurrent-state group). Its sizing half, `ValidateCapacityInputs()`, is
  the same checks minus the page counts that describe a sized pool; the
  `CapacityModel` runs that half so a pool can be sized before it exists, and
  `Validate()` runs it too, so no rule is stated twice. The `Scheduler` runs
  `Validate()` before constructing any member, because the pools and the
  coordinator assert on the same fields and would otherwise preempt the
  diagnostic. Python callers must also pass `Scheduler(config)` explicitly;
  the binding retains no module-lifetime default configuration. Consequently
  `MakeSpecsFromConfig` is pure translation — it validates nothing;
* the scheduler layer **transports** `cache_blocks_per_lcm_block` rather than
  reasoning with it. It appears in `csrc/scheduler/` only as a config field
  copied into the spec; capacity math stays in tokens and pages and folds to
  LCM blocks inside `CapacityModel::LcmBlocksNeededFor`.

Note on naming: the capacity counts intentionally keep *LCM block* names. An
LCM parent is a byte-uniform storage unit whose token span differs per group
(packing is solved from field byte ratios), so no token-unit name would be
truthful. What Principle 2 requires is that the scheduler not *reason* about
the geometry — opaque physical counts crossing the boundary is the same
carve-out Principle 3 makes for CacheBlock ids.

### Principle 3 — emitted table entries: settled by decision, compliant

Decision (2026-08-13): emitted table entries are **`CacheBlock` ids**, and
that is the accepted contract. The code works this way: rows are logical
(row *i* covers `[i*block_granularity, (i+1)*block_granularity)`), entries
come from `ResolveCacheBlockId`
(`csrc/cache/allocator/group_allocator.h`), and the packing fold lives on
the Python side of the contract (`recipes/cache_runtime.py` validates
`group page counts == num_lcm_blocks * packing + 1`; the bridge in
`engine/scheduler_utils.py` ships the folded counts), so Python's
per-forward mapping only subdivides `block_granularity → kernel_page_size`
and never touches packing. No refactor needed here.

### Principle 4 — page_table vs. block_table: fixed in Python's own naming

* C++ keeps its single generic `BlockTable` container — per this doc's
  vocabulary that is the correct name for the scheduler-side container; the
  `page_table` concept exists only where paged-KV kernel tables exist, i.e.
  in Python.
* The Python residues are cleaned: `FlashMLADecodeMetadata.page_table`, the
  TRT-LLM MLA chunked-prefill metadata's `page_table`, inkling's
  `col_block_table` (conv state), and the base-class group routing
  docstrings. Third-party kernel keyword names
  (`flash_mla`'s `block_table=`, TRT-LLM's `block_tables=`) are an external
  boundary and stay as the kernels spell them.
* The state backend's replay hook names no `page_table` parameter — state
  attention has no page table, so the shared call's keyword is absorbed unused
  via `**kwargs`. `input_buffer.py` carries no table at all anymore: KV write
  locations are backend-owned (`write_locations`), so the runner's input prep
  writes positions and seq_lens only.

### Principle 5 — Python perceives the logical quantities minimally: fixed; one conversion point, one slot invariant

Compliant: the recipes/planner layer *owns* the vocabulary rather than
leaking it; the router's `GroupTableStacks` fill (`backends/group_tables.py`)
is the single expansion primitive; state attention and KV share one
plan/arena/`CacheBlock` view, mirrored by the host tier. Specifically:

* No magic `prefix_granularity == 128` branches: the constraint is named
  `MXFP8_KV_SCALE_TILE_TOKENS`, defined in `recipes/plan.py` beside the scale
  field geometry it fixes and re-exported from `recipes/spec.py` for callers
  that read it as a token span.
* glm5's page→slot arithmetic lives in the mapping layer
  (`attention/page_table.py::build_prefill_kv_workspace_slots`), not in the
  model.
* The mamba backend derives its checkpoint span from the state group's
  `spec.checkpoint_granularity`, which snapshot-state groups declare directly
  — asking such a group for `page_size` is a `TypeError`, since it has no
  rows.
* Spec geometry is shape-checked at construction: row geometry and
  `checkpoint_granularity` are mutually exclusive, both positive, and each
  family is held to its shape and retention — `state` declares
  `checkpoint_granularity` and `full_history`, `history` declares rows
  (`CacheGroupSpec.__post_init__`; the C++ `CacheGroupConfig::Validate`
  refuses a sliding `State` group at the bridge).
* Group consumption is claimed positively, from one declaration: each
  consumer takes exactly the delivered `block_tables` entries for the
  groups it serves. The router builds a leaf for each claimed attention
  group and fails a live batch missing any of them. QSA's indexer claims
  its compressed/recent history groups separately; it does not instantiate
  attention leaves for them. State consumers (Mamba/KDA, PLE, Inkling conv)
  index the dict by their own group ids, as does the V4 backend for the
  several history groups one V4 layer reads at once (its pool view reports
  no `PagedAttention` binding and no router leaf). Mamba's group set comes from
  its recurrent fields, so a PLE checkpoint group cannot arm its verify
  state. `cache_consumer_families` remains the boot-time coverage
  declaration (`validate_scheduler_config`). Extra delivered groups ride
  through untouched; a table for a group the bound pool never published
  fails loudly.
* The logical→physical conversion has ONE home per pool view:
  `CacheGroupRouter` (`backends/router.py`) learns each group's
  `block_granularity` and each leaf's `kernel_page_size` into one
  `CacheGroupGeometry`, expands the bridge's raw block tables into
  kernel-page stacks, and derives every KV write location — extend spans,
  the decode/verify window, and the drafters' published step windows — from
  those same tables (`backends/write_locations.py`, pure functions). Paged
  leaves see kernel vocabulary only. The bridge's per-group table views
  (`CacheBatchMetadata`) are the router's input — block vocabulary in,
  kernel pages out, one expand launch per group. Models and the runner never
  compute locations — `write_locations(layer, mode)` is the single accessor
  (`unified_path.md`, "Write locations have one owner").
  QSA's indexer reuses `GroupTableStacks` with `kernel_page_size` equal to
  each group's `block_granularity`. This ratio-one fill copies stable raw
  table views and clears holes/padding; it does not add another subdivision
  convention or derive its addresses through a dummy attention leaf.
* The slot *arithmetic* itself lives in the mapping layer in exactly two
  spellings of one invariant (`table[req, pos // P] * P + pos % P`, which
  is page-size invariant): the router's stacked window/span math
  (`backends/write_locations.py`, failing to slot 0 — the reserved dummy
  page) and the token-shaped resolve
  (`attention/page_table.py::group_slot_mapping_from_raw` +
  `safe_page_ids` / `mask_invalid_graph_tokens`, failing closed to the
  `-1` skip sentinel for group buffers with no dummy page). DeepSeek-V4 is
  a *composer*, not an owner: its SWA / compressor-state / indexer-state
  writes call the shared token-shaped resolve over its delivered tables;
  only the compression-boundary orchestration (per-ratio strides, boundary
  masks, the fused dsv4 kernel) is V4-specific
  (`kv_cache/hybrid_deepseek_v4.py`).

### Principle 6 — provenance discipline: fixed

* The contract's `prefix_granularity` comes from the memory plan
  (`kv_cache/arena.py` builds the runtime contract from
  `plan.prefix_granularity`, never from pool geometry). ✓
* Field dtypes come from the memory plan. `CacheFieldSpec`/`CacheFieldLayout`
  carry `dtype` (a name; `plan.py` stays torch-free because the plan travels
  the PD wire) and `element_size` derives from it, so byte geometry and dtype
  cannot disagree. Recipes name each field's dtype where they already know it,
  via `cache_dtype_name` or `scatter_stored_dtype_name`; the latter holds the
  one substitution rule — fp8 collapses to `uint8` for fields written by
  elementwise scatter, because `index_put` has no fp8 kernel, while fields
  written through dtype-aware kernels (MXFP8) keep their fp8 view. The
  contract carries no parallel `field_dtypes` tuple. ✓
* The arena owns the allocation and materializes every planned field view in
  its constructor, so `field(field_id)` is a lookup with no dtype argument and
  no lazy-bind state. `CachePool.store_dtype` means one thing: how a pool
  reinterprets *input* tensors before a write. A pool allocates nothing —
  `_bind_layer_planes` walks `plan.fields` once and arranges this view's layer
  window into the per-layer buffers its kernels read, with each subclass
  declaring only its `layer_plane_bindings`. Which planes a layer has is a
  fact of the plan (a state layer plans no `k`/`v`), and page contiguity is a
  plan invariant (`exact_page_stride`). ✓
* Every backend's `kernel_page_size` is registry- or config-sourced
  (`kernel_page_sizes.py`); the registry LCM validator checks explicitly
  configured values, and a backend that resolves its own registry default
  owns the divisibility check for it. The V4 milestone closed the last
  P-derivation: compressed-chain rows, SWA rows, and layout byte shapes all
  build from `DEEPSEEK_V4_PAGE_SIZE`, and V4 accepts any P that is a
  positive multiple of it (e2e-verified against the `bt_v4` baseline;
  GSM8K 1319-question sweep: nospec 0.9651, DSpark 0.9629). ✓
* Two arena scalars, two roles, both derived from the plan:
  `CacheArena.prefix_granularity` (identity grain; contract publication and
  plan checks) and `CacheArena.kv_page_size` (KV arena geometry, read by
  row/slot/tile math in the paged pools and their consumers). Prefix reuse
  also requires the state-checkpoint mapping point to select the snapshot
  at the contract's prefix boundary, independently of its state-block span. ✓
* Cache geometry has one owner and no mirrors. `CacheArena` holds the
  allocation, the field views, the plan, the contract and the geometry
  scalars; `CachePool` is a typed layer window that forwards nothing, so
  consumers write `pool.arena.plan` and cannot read a stale copy off a view.
  What stays per-view is what genuinely differs per view: the dtype these
  bytes are read as (a bf16 draft head over an fp8 target is two views of
  one arena), the layer-window offset, and the per-layer kernel buffers. ✓

### Principle 7 — one pipeline, declared once: fixed

* Every family's cache is built by `CacheRecipe.setup()`, the single place the
  four stages appear in order (see *The cache pipeline* above). A family fills
  in uniformly named seams and marks each with `@override`, so a renamed seam
  fails type checking rather than silently taking the base default. ✓
* The stage names and the function names are the same words: `group`, `pack`,
  `bind`. `pack` is capacity-independent — it describes one physical parent —
  and `bind` is the only place a parent count enters. ✓
* A group id is written once, in its spec, next to the fields that deposit
  bytes in it. `CacheFieldSpec` carries no `group_id` and `CacheGroupSpec` no
  packing: the declaring group is positional, and packing is the layout's
  answer, so neither can be stated twice and disagree. ✓
* The model side names no ids: a `PagedAttention` layer declares only its
  compute mask, and `bind_cache_groups` stamps its group from the pool's plan
  at startup, checking that the group's retention covers the mask (see
  *Storage vs. visibility*). Backends index their learned geometry by the
  bound id with no fallback (`CacheGroupGeometry.granularity_of` raises on
  unknown ids). Block drafters ride a target-owned group whatever mask their
  layers apply: the full-history group for drafts with their own attention
  layers (`check_block_drafter_storage`), the SWA group's extra fields for
  V4.1's same-checkpoint DSpark. ✓
* Capacity has two shapes and no more, and one place to read the scheduler's
  concurrency (see *The cache pipeline* above). ✓
* Kernel geometry does not live under the recipes package. DeepSeek V4's byte
  formulas, cache layout and group-id vocabulary sit in
  `attention/deepseek_v4_geometry.py`, which the backends, ops, model and pool
  read directly; the recipe depends on it, not the reverse. ✓
* One kernel layout has one definition. The interleaved mxfp8 KV-scale planes
  come from `plan.mxfp8_kv_scale_fields`, which also owns the page-span and
  head-dim constraints that layout imposes, so no recipe restates the shape or
  re-checks the constraint. ✓

Verified end to end for this round: DeepSeek V3.2, R1 and V4-Flash, each
× {CUDA graph, eager} × {spec, no spec}, against pre-refactor baselines
(accuracy equal or better; speculative accept length within noise).
