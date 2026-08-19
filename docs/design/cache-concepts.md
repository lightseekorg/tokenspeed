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
one of two family-specific shapes (Python
`CacheGroupSpec`):

* **Row geometry** (`rows_per_page` × `entry_stride_tokens`, exposed as
  **`page_size`**) — for paged KV-cache consumers, whose blocks physically
  hold rows of entries (per-token KV, sliding windows, compressed entries).
  "Page" vocabulary is *only* legal here.
* **`checkpoint_granularity`** — for snapshot-style state groups
  (recurrent/conv state), whose blocks each hold one state snapshot taken
  every this-many tokens. Such a group has no rows and no pages; declaring
  fictional row geometry for it is a bug, not a convention.

The two shapes are mutually exclusive, and the split is by *shape*, not by
family: V4's sliding-window tail and compressor buffers are state-family yet
have real row geometry, so they declare rows.

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
C++     CacheGroupSpec     folded scheduling form (block_granularity only)
```

The first and third share a name and differ in fields, so always qualify
which side you mean; `CacheGroupConfig` in between is the only one visible
from both.

The same boundary holds for `checkpoint_granularity`: the identifier never
enters `tokenspeed-scheduler` at all. It is a Python-side *declaration
shape* on `CacheGroupSpec`, and the bridge
(`scheduler_utils.pool_to_cache_groups`) is the single folding point:
a snapshot declaration folds to `(rows = checkpoint_granularity,
stride = 1)` and crosses into C++ as `CacheGroupSpec.block_granularity` — so
a snapshot group's `block_granularity` equals its `checkpoint_granularity`
numerically, and the scheduler has no "checkpoint" word, only "how many
tokens one block-table slot covers". The row-geometry shape
(`rows_per_page`, `entry_stride_tokens`) folds away at the same point.
Declaration-shape vocabulary stops at the bridge; only the generic span
crosses it.

`CacheGroupSpec.block_granularity` is **required and explicit**: a positive
divisor of `prefix_granularity`, rejected by `SchedulerConfig::Validate()`
before construction and asserted again at coordinator construction. There is
no zero-means-default fallback — every group states its span.

The block tables the scheduler emits are logically *indexed*: row *i* of a
request's table covers tokens
`[i * block_granularity, (i + 1) * block_granularity)`. The
entry *values*, however, are **`CacheBlock` ids** — handles to the physical
storage the cache layer allocated. The scheduler owns allocation, so its output names that storage directly.
Consumers outside the cache layer treat the ids as opaque.

Logical width does not imply dense physical residency. Full-history KV and
retained sliding-window rows materialize every block their kernels read, but a
full-history snapshot-state prefill needs only its input checkpoint, final
output checkpoint. The next decode admission allocates its destination after
prefill scheduling and rolls the expired input page forward, including under
overlap scheduling. Its table therefore keeps absolute slot positions while
representing skipped intermediate checkpoints as null holes (`0`). State
consumers may gather only the declared input/output slots; compacting the row or
publishing an unwritten intermediate checkpoint would break position identity.

Speculative KDA verification keeps its candidate recurrent states in a dense,
graph-stable workspace outside the cache arena. The Kimi-K3 recipe reserves
that workspace before sizing the arena (`max_bs * (draft_tokens + 1)` state
rows), so two rolling persistent pages do not make speculative state memory
disappear from the GPU budget.

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
identity grain, used only for contract publication and plan-consistency
checks — `prefix_granularity` exists to compute prefix hits, and runtime
arithmetic must not reach for it; `CacheArena.kv_page_size` is the KV arena
page span that paged-KV geometry math (row views, slot↔page arithmetic,
scale-tile branching) reads. Both derive from the one plan, which is the
single point of the prefix-page ↔ KV-page convention.
Neither is a statement about per-group CacheBlock geometry: group spans live
in the specs as `block_granularity`, and blocks narrower than P (V4's SWA
window, state checkpoints) are the norm, not the exception. A backend's
`kernel_page_size` comes from the kernel registry or an explicit config
override, never from `prefix_granularity` as a fallback. The CLI flag is
`--prefix-granularity` (`--block-size` remains a deprecated alias).

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
  pool-scoped: one index serves both the Device and Host tiers of its group.
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
dependency is visible in the signature, not hidden in shared state.

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
  across groups; `Admit` then allocates, pins the hit prefix, and produces
  host→device `load_pairs` plus each group's fresh pages. Probe and admit are
  deliberately split so that a failed admission can be retried under a
  hypothetical release (`CanAdmitAfterReleasing`) without mutating cache
  state. `ProbeDecodeDevicePrefix` is the PD-decode variant: local history
  pages are reused while final-state groups are restored from the remote
  endpoint snapshot.
* **Prefix publication.** `CacheFullBlocks` / `CacheCompletedBlocks` register
  computed blocks into the prefix indexes for later requests. Prefix-closed
  groups match first; non-closed groups (SWA, Mamba) match only within the
  boundary the closed groups settled (`match_order_` enforces this).
* **Two tiers (Device/Host).** Device prefix publication can optionally
  stream to the Host tier (`stream_device_cache_to_host_`); a
  `pending_stores_` queue drives D2H transfers, alongside Host-side
  acquire/contains/pin queries.
* **Reclamation and lifecycle.** `ReclaimExpired`, `Free`,
  `ClearDeviceCache`/`ClearCache`, and `NumNewlyReleasableLcmBlocks` for
  ranking retraction (preemption) victims.
* **Mutation reporting.** `SetCacheMutationSink` reports per-group cache
  insertions/removals; the scheduler folds them into one externally visible
  prefix event.

`MakeCoordinator` is the factory: one `CacheGroup` per `CacheGroupSpec`
(group_id = index), all sharing one scheduler-level `prefix_granularity`
while each manager may use a smaller cache-page token count.

### `AdmissionPlanner` (`cache_admission.cpp`, anonymous namespace)

The internal capacity planner behind `Admit`. It runs entirely on shadow
occupancy — never mutating the real pool — and answers: *which cached blocks
must be evicted for this admission to fit, while protecting the current
prefix hits?* The algorithm: first check whether existing local holes plus
empty parents fit with zero eviction; otherwise pop victims from a heap
ordered by eviction policy (LRU access epoch, then tier: uncached
request-only block → probationary boundary → established boundary → suffix of
a closed prefix) until the plan fits; finally walk the victim list in reverse
and restore every victim that is not strictly required, yielding a minimal
eviction set.

## The cache pipeline: layers → group → pack → bind

Every model family's cache is built by the same four-stage pipeline, and the
stage names are the vocabulary:

```
layers ──group──▶ groups ──pack──▶ CacheLayout ──bind──▶ CacheMemoryPlan
```

* **layers** — the family's layer vocabulary: a `layer_types` label and a
  `group_ids` assignment per layer, target layers then draft layers.
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

* the group set in the plan equals the declared one because `pack` consumes
  the `(spec, fields)` pairs and `setup()` publishes the specs from those
  same pairs;
* a field cannot name a group the plan does not have, because it never names
  one — `pack` carries the declaring group id alongside each field;
* per-group packing is read from the layout, not recomputed, everywhere
  downstream (the C++ bridge, the runtime contract, capacity math).

If you find yourself writing a check that two derived views agree, the
design is wrong: make one of them the source.

Capacity has exactly two shapes, both on the base class. The default is the
flat product (`parents × tightest packing × P`). Families whose per-group
demand differs — K3's state groups riding inside MLA planes, V4's SWA and
compressed chains — override `parents_needed` and get the inverse for free
from `_capacity_from_parents`, one monotonic binary search shared by all.
`scheduler_limits` is the single place a recipe reads the scheduler's
concurrency, so demand and capacity cannot size against different numbers.

The runtime's global `max_num_seqs` is divided across attention DP ranks to
produce each scheduler's rank-local `max_batch_size`. These values limit
simultaneous sequence slots; they do **not** reserve enough history cache for
that many maximum-length requests. Aggregate prompt and decode growth must
still fit the recipe's reported token capacity. When that dynamic pool is
overcommitted, admission can fail even though the batch still has a free
sequence slot.

For a fused scheduler without Host L2, such a failure may directly retract a
resident request and release its Device pages. Direct retraction enters a
capacity-drain phase: already-resident `Prefilling`, `PrefillDone`, and
`Decoding` requests continue, while `Submitted` requests (including the
requeued victim) cannot consume the released pages. Admission resumes
automatically after that resident cohort finishes. This prevents an
overcommitted workload from repeatedly rebuilding a prompt, decoding briefly,
and retracting it again. Best-effort L2 writeback, PD Decode recovery, and
ordinary capacity-fit continuous batching do not use this fused direct-drain
path.

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
capacity views — `LcmBlocksNeededFor(group_pages)`,
`NumActiveLcmBlocks(request_tables)`, `NumAvailableLcmBlocks`,
`TotalLcmBlocks`, `GroupAvailablePages(group)` — and the scheduler treats the
counts as opaque capacity units. The null-page reservation lives in
`SchedulerConfig::AllocatorConfig::NumUsableBlocks()`, and nothing outside the
cache layer enumerates LCM block ids.

Enforced:

* the identifier `page_size` is grep-zero across `tokenspeed-scheduler`
  (csrc, tests, python bindings) — the slot span is spelled
  `block_granularity` everywhere;
* `CacheGroupSpec.block_granularity` is required and explicit: every group
  states its span, with no zero-means-default fallback, and the coordinator
  asserts a positive divisor of P at construction;
* `SchedulerConfig::Validate()` is the **single** configuration gate: every
  scheduler scalar, every `CacheGroupConfig::Validate()`, and the cross-checks
  between them (P divisibility, PD transfer policy, one-cache-block chunks for
  a recurrent-state group). The `Scheduler` runs it before constructing any
  member, because the pools and the coordinator assert on the same fields and
  would otherwise preempt the diagnostic. Consequently `MakeSpecsFromConfig`
  is pure translation — it validates nothing;
* the scheduler layer **transports** `cache_blocks_per_lcm_block` rather than
  reasoning with it. It appears in `csrc/scheduler/` only as a config field
  copied into the spec; capacity math stays in tokens and pages and folds to
  LCM blocks inside `LcmBlocksNeededFor`.

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
  TRT-LLM MLA chunked-prefill metadata's `page_table`,
  `_page_table_aliased`, inkling's `col_block_table` (conv state), and the
  `CacheGroupsMixin` docstring. Third-party kernel keyword names
  (`flash_mla`'s `block_table=`, TRT-LLM's `block_tables=`) are an external
  boundary and stay as the kernels spell them.
* The state backend's replay hook names no `page_table` parameter — state
  attention has no page table, so the shared call's keyword is absorbed unused
  via `**kwargs`. And `input_buffer.py::fill_input_buffers` takes a
  unit-neutral `out_loc_table`: the batch-ordered table `out_cache_loc`
  derives from, which is the scheduler's full-history table on the target path
  and the staged draft table on the drafter path.

### Principle 5 — Python perceives the logical quantities minimally: leaks fixed, mapping owners still four

Compliant: the recipes/planner layer *owns* the vocabulary rather than
leaking it; `expand_page_table` (`attention/page_table.py`) is the single
expansion primitive; state attention and KV share one plan/arena/`CacheBlock`
view, mirrored by the host tier. Specifically:

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
  `checkpoint_granularity` are mutually exclusive, both positive, and
  family-gated (`CacheGroupSpec.__post_init__`).

Remaining known item (deliberate, separate project): the mapping *primitive*
is single but the *owners* are four — MLA `CacheBatchMetadata.kernel_table`,
the MHA `CacheGroupsMixin` (eager plus two graph paths),
`DraftPageStaging.publish`, and DeepSeek-V4's bespoke slot mapping — each
with its own caching and validation, and the write-location math triplicated
alongside. Consolidating them means touching every backend family at once;
do it as its own milestone.

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
  plan checks only) and `CacheArena.kv_page_size` (KV arena geometry, read by
  row/slot/tile math in the paged pools and their consumers). Prefix-hit
  computation is the only computational consumer of `prefix_granularity`. ✓
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
