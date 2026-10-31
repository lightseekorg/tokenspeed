# Cache concepts: prefix matching vs. storage

This document defines the conceptual layering of the cache subsystem: which
concepts are *logical* (token-based, storage-agnostic) and which are *physical*
(storage-based), and which components are allowed to see which. It is the
reference for naming, code placement, and layering decisions in both the C++
scheduler (`tokenspeed-scheduler`) and the Python runtime.

## Two worlds: logical tokens vs. physical storage

### Logical world (token units, storage-agnostic)

**Naming convention: every `*_granularity` quantity (`prefix_granularity`,
`block_granularity`) and `page_size` is measured in logical tokens.** A name
in this family never counts rows, blocks, or bytes; conversely, a quantity
counting storage must not borrow one of these names.

Two quantities anchor the vocabulary:

* **`prefix_granularity`** — the granularity at which prefixes are hashed:
  one chain hash per page of this many tokens. A reuse boundary `T` is any
  token count at which some request published its state: every multiple of
  `prefix_granularity` a request completed, plus the **tails** it published
  wherever it stopped writing — its prompt end, a junction cut, its finish or
  retraction (`hash@T`, the same chained hash over the page's first `T mod P`
  tokens). Two requests share cache below the longest published boundary that
  is a prefix of both.
* **`block_granularity`** — the number of tokens covered by one block-table
  slot (one `CacheBlock`) of a history group. This is the unit in which
  kind-agnostic code (admission, capacity planning, PD slot selection)
  addresses the cache.

A group's **attention kind** (Python `CacheGroupSpec.attn_kind`, the C++
`AttnKind`) is its one classification: what its blocks hold and how long they
stay. There are exactly three, and every scheduler decision that depends on
the group's nature switches on this value and nothing else:

* **`full_history`** (`kFull`) — rows of token history that stay resident for
  the request's whole life; prefix-closed and shared.
* **`sliding_window`** (`kSlidingWindow`) — rows of token history of which
  only the trailing `sliding_window_tokens` stay resident; the rest slide out
  (`ExpiredBlocksAt`). A trailing window of token rows (DeepSeek V4.1's SWA
  kv and compressor tails) is this kind whatever a kernel calls its buffer.
* **`state`** (`kState`) — recurrent/conv state whose block holds the state
  after some token count, whichever count a forward happened to end at. A
  state summarizes everything before it, so nothing in it ever slides out and
  it carries no window. A request holds exactly two blocks of every state
  group (see "Recurrent state: two slots, natural checkpoints").

The first two are the **history** groups: their declaration is **row
geometry** (`rows_per_page` × `entry_stride_tokens`, exposed as
**`page_size`**, the group's `block_granularity`), and they are the paged
KV-cache consumers whose blocks physically hold rows of entries (per-token KV,
sliding windows, compressed entries, compressor input tails). "Page"
vocabulary is *only* legal here. A `state` group declares **no span**: no
rows, no pages, no token span per block; declaring row geometry for it is a
bug, not a convention, and its `block_granularity` is a `TypeError`.
`CacheGroupSpec.__post_init__` holds each kind to its shape (a window only on
a sliding group, rows only on a history group, replay only on a sliding
group). `state` is not a routing label for "some backend owns this table".

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
  `config.kernel_page_size` overrides any default; deriving `kernel_page_size`
  from `prefix_granularity` is a category error and a bug.

### Physical world (storage units)

* **`CacheBlock`** (and its aggregation, the **LCM block**) is the unit of
  physical storage. Allocation, refcounting, eviction, and tiering operate on
  physical blocks.
* A `CacheBlock` is attention-agnostic storage: it can be *viewed* by
  KV-cache-based attention (as paged KV entries) and by state-based attention
  (as a state slot). The view is defined by the consumer, not by the block.

`BlockPool` owns the physical placement indexes: the FIFO of empty LCM blocks,
the free child-slot count **per bucket** for each cache group, and per-bucket
ordered sets of partially filled LCM blocks. C++ group ids are dense scheduler
indices, so the scheduler supplies the complete packing vector when it
constructs each pool. The per-group placement records form a vector indexed by
group id, and each `GroupAvailability` stores immutable slots-per-parent
geometry. The coordinator registers each group's shard count in every pool it
owns — Device, Host L2, and the snapshot pool alike — before allocation, so a
block's bucket means the same owner on every tier; an ordinary standalone pool
fixes a single bucket on first use. Registration cannot change the geometry
after it is fixed, including after all blocks are freed. The pool updates its
per-bucket free-slot counts and bucket indexes together on every occupancy
transition. A parent with zero occupants is unbound, so its capacity belongs
to the global empty-parent FIFO rather than any group.

Packing is a property of each pool. Device packing satisfies attention-kernel
strides; compact Host storage uses a parent large enough for the largest single
group payload, then fits as many complete children of each group as that parent
holds. Host L2 and the request-private snapshot pool use this compact geometry.
The scheduler receives their virtual packing explicitly, including each group's
shard count. Enabling either Host pool requires `host_group_packing` with one
value per group, in `cache_groups` order; Device packing is never a fallback.
A block reference's owner pool determines its exported block ID;
the same location can encode different IDs on Device and Host. Placement,
eviction's shadow fit and snapshot sizing all use the destination pool's
packing. Cyclic owner counts stay the same across tiers, so transfers still
keep the Device block's owner bucket. Every empty parent returns to the shared
pool; no capacity is reserved for State or history.
Mirrored ranks must agree on Host packing as well as parent counts. Startup
compares both before allocating pinned memory and refuses incompatible local
payload geometry rather than giving the same block ID different meanings.

The pool knows which child slots are occupied, never who holds them. Whether a
child is pinned by a request table, published by a prefix-cache entry, or held
by an in-flight transfer is a `CacheBlockRef` ownership fact that lives with
the holders; the pool does not track it, and `CacheBlockRef` does not report
it to the pool. Anything that needs "held only by the cache" asks `PrefixCacheIndex`
(`ParentIsFullyEvictable`), which is a scan and is therefore reserved for
eviction policy and leak checks, not for per-step accounting.

Admission first checks the indexed free-slot and empty-parent counts. If the
request fits, it does not enumerate eviction candidates. Under memory pressure,
it enumerates candidates and keeps shadow occupancy only for the parents whose
children it tentatively evicts. This makes the common zero-eviction path scale
with cache groups and request demand rather than total cache capacity.

Replicated groups have one ownership bucket: every Host hit needs a Device
destination in that bucket, without looking up or temporarily owning the Host
block to count it. Sharded groups count destinations by their Host sources'
ownership buckets.

The prefix index remembers which entries an eviction traversal found pinned.
Later admission attempts skip those entries until the last external block owner
releases them. A one-shot ownership notification restores each entry to its
original access epoch, including an epoch changed while it was pinned. This
keeps the same LRU victim order without repeatedly scanning active requests'
blocks. Temporary lookups do not update this secondary index. The notification
belongs to block-reference ownership; the pool still tracks placement alone.

Host duplicate reclamation checks pending stores on a candidate's ancestor
chain once per scan and reuses that answer for other candidates sharing it.
The memo and traversal buffers belong to the scan: the next scan rechecks
store completions and new tree links. This changes traversal cost without
changing candidate eligibility or eviction order.
Collection also keeps each candidate's Device ownership and access epoch for
ranking in the same call, avoiding a second set of key and tree lookups.

The per-step page gauge composes two O(1)-or-cheaper quantities the same way:
empty parents come from the pool, active parents from the live requests' block
tables, and cache-only residency is the remainder `total - empty - active`.

## Who is allowed to see what

### C++ scheduler: schedules in logical units

The scheduler makes its decisions — admission, prefix matching, chunk
alignment, capacity — in **exactly two token-based quantities**:
`prefix_granularity` and the per-group `block_granularity`
(`CacheGroupSpec.block_granularity`, wrapped by the coordinator's
`GroupGeometry`; both declaration shapes fold to it at the bridge — see
below). Physical geometry (LCM packing, storage counts, bytes) is confined
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
Python  CacheGroupSpec     kind + declaration shape (rows | no span)
  ↓     pool_to_cache_groups                      the single folding point
C++     CacheGroupConfig   boundary config, nanobind-exposed (SchedulerConfig.cache_groups)
  ↓     MakeSpecsFromConfig
C++     CacheGroupSpec     scheduling form (block_granularity + kind)
```

The first and third share a name and differ in fields, so always qualify
which side you mean; `CacheGroupConfig` in between is the only one visible
from both.

Declaration-shape vocabulary stops at the bridge; only the kind and the
generic span cross it. Neither `rows_per_page` nor `entry_stride_tokens`
enters `tokenspeed-scheduler` at all: they are the Python-side *declaration
shape* of a history `CacheGroupSpec`, and the bridge
(`scheduler_utils.pool_to_cache_groups`) folds them to
`CacheGroupSpec.block_granularity = rows_per_page × entry_stride_tokens`
before constructing the C++ `CacheGroupConfig`, whose only span field is
`block_granularity`. The scheduler has no "row" or "stride" word — only
"how many tokens one block-table slot covers". A `state` group crosses the
bridge with `block_granularity = 0`: its blocks cover no tokens, and
carrying a fictional span for it would be a claim its contents contradict,
the same way a `Paged` type name would.

A history group's `block_granularity` is **required and explicit**: a
positive divisor of `prefix_granularity`, rejected by
`SchedulerConfig::Validate()` before construction and asserted again at
coordinator construction. There is no zero-means-default fallback — every
paged group states its span, and `0` is the `state` kind's declaration that
it has none.

One flag crosses the bridge beside the kind and the token quantities:
`replayable`, set only on a `sliding_window` group that is **regenerated by
bounded replay**
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

With KVP virtual-block placement, `CacheBatchMetadata` validates exported IDs
against each group's **virtual** block count. The physical page count bounds
local arena storage only; applying it to the scheduler table would reject valid
remote-owner IDs before the runtime can translate them.

#### Recurrent state: two slots, natural checkpoints

A `state` group's block table has **two slots per request** with fixed roles,
not a position-indexed row:

```
slot 0  input checkpoint   a claimed prefix hit (shared, immutable) |
                           the block the request's previous forward wrote |
                           a fresh zeroed block: no history, the zero state |
                           null: read slot 1 instead
slot 1  live               the block this request's forwards write
```

A forward reads slot 0 when it holds a block and slot 1 otherwise, and always
writes slot 1; the rule is the same for prefill, decode, and verify, so the
runtime needs no "first chunk" or "first decode" signal
(`layers/attention/backends/state/checkpoint.py`, `compute_state_block_indices`;
the decode prep tape and `verify_state_blocks` spell the same `where`). A
prefill chunk therefore reads one block and writes another ("write-to-new"),
and a request holds its two blocks from its first admission on: a prompt with
no checkpoint to resume from gets a fresh block in slot 0, zeroed by the plan
like any fresh page, so the zero state is a block like any other and no
rotation ever needs a block the admission did not already grant. At the next
admission the scheduler **rotates**: slot 0 leaves the table (the prefix
index keeps it alive if it was published), slot 1 moves to slot 0, and a
fresh block becomes slot 1. When the prompt end is published (the prefix
cache is on), the first decode step still sees the prompt-end state in slot 0
and writes a fresh slot 1; from the second step on -- from the first, with
nothing to publish -- slot 0 is null and decode evolves slot 1 in place. A
null slot 0 also means "read slot 1" for the rows the runtime fabricates
(capture dummies) and for the decode role's landing row. Speculative verify
commits the accepted state into slot 1 unconditionally
(`commit_state_pages`).

A slot's state block is a whole snapshot, so the token count that keys it
must be exactly the tokens it consumed. The device therefore ends a request
where the control plane does: the sampling epilogue cuts every round's
accepted width at the request's first stop token and its token limit
(`RuntimeStates.apply_stop_conditions`, the stop set the finish check reads),
before the commit, the frontier or the drafter reads it, and marks the slot
stopped. A decode the overlap schedule already dispatched behind that round
runs the request as a padding row—both slots `-1`
(`RuntimeStates.masked_state_tables`): it reads the zero state, writes
nothing, and keeps no token. (Pipeline parallelism serves only the prefill
role, which neither decodes nor overlaps, so its stages, which do not sample,
never run a forward behind a stopped request.) The finish checkpoint is thus the state before
the stop token under every schedule ([Scheduler
§1.2](scheduler.md#12-state-groups-two-slots-natural-checkpoints)); a finish
dispatched as a rotating first decode publishes at the prompt end, where the
prompt-end checkpoint it still holds in slot 0 is the canonical block.

State groups may additionally restrict reusable checkpoints with
`checkpoint_granularity`: intermediate prefill forwards end on its multiples,
and only prefill-produced states at these positions are published or matched.
The existing admission publishes retained checkpoints in stream order. A first
decode admission can deliver the preceding prefill's checkpoint; decode-produced
live states remain private, including at finish and retraction. Retraction
still preserves the private state for the same request's restore. A forward may
span several segments; the existing in-forward checkpoint mechanism saves the
last segment entrance independently of the live endpoint.
This is a numerical continuation constraint, not a block's token span. The
grain is also part of the persistent cache identity and the P/D semantic
contract, so a scan-final state cannot be loaded as an aligned entrance H.
Unrestricted persistent namespaces remain unchanged. Live State
blocks retain the same two-slot ownership; an interior checkpoint uses the
existing separately allocated output block. Checkpoint keys always name the
actual computed frontier; an unfinished tail is never relabeled as an earlier
aligned H. Unrestricted State groups preserve the natural-checkpoint policy
below.

A **natural checkpoint** is a state block that exists because a forward ended
there -- the prompt end (the block rotated out of slot 1 as the first decode
runs), the finish (slot 1 in place), the chunk the junction cut ended, or the
retraction image -- published under the hash of that token count, page
boundary or not. The one interior checkpoint is opt-in
(`--enable-state-checkpoint-in-forward`): the forward that spans a long
prompt's last page boundary before its replay tail also stores the state at
that boundary into a third block of its rows (`STATE_CHECKPOINT_SLOT`,
`InForwardStateCheckpoints`) -- the GDN scan emits it from inside the kernel,
the conv and PLE windows are the inputs before it -- which is then published
like a natural checkpoint
([Scheduler §1.2](scheduler.md#12-state-groups-two-slots-natural-checkpoints)).
No scan is split into body and tail, and no state group declares a token
span: a hit resumes
at the published token count `T` exactly, with its state in slot 0 and the KV
tail of the same instant in the history groups
([Scheduler §1.2](scheduler.md#12-state-groups-two-slots-natural-checkpoints)
for the rotation, publication, and eviction mechanics; the state block and
its KV tail are evicted together, so a stale state is never matched). Resuming a chunked recurrent kernel at a `T` that is not a multiple
of its kernel chunk regroups a floating-point reduction; `numerics.md`
("Prefix hits resume at any token") states the tolerance and the oracle.

Capacity prefill graphs (`backends/state/prefill_capacity.py`) keep this
contract at a captured request count: an unused request slot scans one
padding row past the live tokens, its state-block index is negative so the
masked state writer (`store_state_rows`) drops its result, and the padded
output rows are zeroed by a device-side bound the capture cannot freeze. The
live tokens keep the packing of the unpadded forward, so a capture of any
request count matches eager bit for bit; nothing about padding reaches the
cache or the scheduler. A forward that writes an in-forward State checkpoint
keeps its live metadata instead (`prepare_prefill_metadata` returns False):
the capacity layout carries no checkpoint, so the layer runs at its eager
break of the ordinary prefill graph.

PD moves exactly one block per state group: `latest_snapshot` is slot 1 of
the prefill's final state, landing in the decode's slot 1 (its slot 0 stays
null, so the first decode reads the landed state). When the decode role's
own prefix cache holds the state for the prompt end -- a state hit counts
only at the prompt end exactly, since the prefill's final state supersedes
any shorter one -- `Forward.Batch.state_prefix_cached` is set for that row
and both sides omit the group from the manifest (`ts-pd`'s `layout::manifest`).
The decode admission copies this immutable input checkpoint into its private
live block before consuming the bootstrap result, since that result can finish
the request without another forward. KDA convolution history may use either
channel-major or channels-last storage; the PD schema partitions the channel
axis in each layout, preserving its separate Q/K/V channel segments.

KDA targets with a batched commit kernel use the same deferred-state lifecycle
for ordinary decode (`T=1`) and speculative verification. Verify leaves the
committed recurrent and convolution state unchanged; acceptance selects the
prefix to commit into the live block.

On NVIDIA and AMD, verify saves FP32 corrections plus unnormalized keys and raw gates
in the activation dtype (BF16 or FP16). Recovery recomputes key normalization
and gate decay, then applies the saved corrections to the input checkpoint.
It does not recompute the state-dependent correction or the gate projection.
Original QKV rows are retained only for convolution-history commit. The
persistent convolution pool keeps its cache dtype, independently of activation
dtype. Records and position/generation stamps cover the current verify round;
they are transient workspace, not prefix-cache entries.

Both vendors consume the same post-convolution activation values in verify
and save those unnormalized keys for recovery. NVIDIA uses Triton recovery;
AMD uses Gluon recovery. Neither recomputes the state-dependent correction.
Convolution history is committed separately so in-place pages are safe.
Platforms without a matching batched commit and no-store verify keep the dense
`max_bs * (draft_tokens + 1)` per-position state workspace.

The Kimi-K3 and GLM-5.3-Flash recipes share the backend's payload specification
and reserve transient convolution rows, payloads, descriptors and record
metadata before sizing the arena. Prefill-only engines reserve no decode
records. GLM also reserves its target/draft-shared KPool tails once, so state
memory does not disappear from the GPU budget.

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
from the memory plan, not read back out of pool state, and the plan takes it
from the model's cache recipe (`CacheRecipe.prefix_granularity`, abstract on
the base): no launch option sets it. The arena carries
**two** scalars with distinct roles: `CacheArena.prefix_granularity` is the
identity grain, used for contract publication and plan-consistency checks.
Other runtime arithmetic must not reach for it.
`CacheArena.kv_page_size` is the KV arena
page span that paged-KV geometry math (row views, slot↔page arithmetic,
scale-tile branching) reads. Both derive from the one plan, which is the
single point of the prefix-page ↔ KV-page convention.
Neither is a statement about per-group CacheBlock geometry: group spans live
in the specs as `block_granularity`, and blocks narrower than P (V4.1's SWA
window) or spanning no tokens at all (recurrent state) are the norm, not the
exception. A backend's
`kernel_page_size` comes from the kernel registry or an explicit config
override, never from `prefix_granularity` as a fallback. The dependency runs
the other way only: a recipe sizes P to hold whole pages of the kernels that
read its blocks. The ordinary recipe makes one block span P, so its P is the
lcm of 64, a fixed-page leaf's page (the registry pins it on the side's
config before planning), the MSA sparse page, and the MXFP8 scale tile; the
other recipes state a constant that is a multiple of their group spans.

Layerwise L2 load fences guard the first access to every field owned by a
layer, not just the conventional paged-KV buffers. Model-owned side caches
(for example QSA raw/compressed keys and positions, or PLE state) must wait on
the pool's layerwise load tracker before their first read or write. A wait in a
later attention-buffer accessor is too late: the side-cache kernels would
already be racing the asynchronous H2D restore. Place the fence immediately
before the first cache-field access so independent projections can still
overlap the load. The fences are eager calls, so a captured prefill graph
segment holds none: the latent rows an attention prologue writes there (into a
page a tail hit loads) and the State checkpoint an inline KDA layer reads there
would race the load. `PrefillGraph.replay` therefore waits for the plan's whole
Host load before the first segment; a replayed forward overlaps only its
embedding with that load.

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

Mapped Host field copies use word accesses when both byte addresses and the
payload size are 16-byte aligned. Other fields retain byte accesses, including
packed children whose offsets lose that alignment. Masked copies preserve
field padding, and per-layer readiness is published only after either copy
has completed. The same geometry and completion protocol serve both cases.

Every L2 copy is ordered after the prerequisite stream the caller names per
submission: the stream whose completed work the copy must observe. Writeback
runs on the executor's write stream, ordered after the model executor's
execution stream, where the forwards wrote the source pages. (Page zeroing
likewise orders itself behind that stream inside `zero_cache_pages`; there is
no caller-side fence to remember.)
Each op says how the scheduler guards its Device sources
(`source_pinned`, see `scheduler.md` §2). A pinned op's sources stay cached
and unevictable until the ACK, so its copy overlaps whatever the round does
next and nobody waits on it. An unpinned op's sources (a retraction image's
pages, or what an admission or a restore evicted under the write-back policy)
may be re-granted in the same plan, so it is launched first and the fence
stream the caller names --
the default stream, where the plan's page zeroing runs -- waits on its
completion event before the zeroing is enqueued — the zeroing, load-backs,
forwards and RDMA triggers behind it inherit the fence (the forward by
waiting on the default stream in its prologue; that wait is one-way, the
zeroing's and the writeback's own waits are what order the default and write
streams behind the forwards). A load-back's prerequisite stream is the
default stream that zeroed its destinations; its start event is recorded
there rather than on whatever stream is current. Every stream the Host
cache executor orders against is an argument, never ambient thread state.
Each submission kind uses its own staging lane (`cache/transfer/lanes.py`:
the stream-ordered and pinned write-backs, the snapshot store, and the
restore's two tiers): a lane uploads block metadata asynchronously and
records an event after both metadata copies to protect its pinned CPU
staging tables — before refilling them, the next submission on that lane
waits only if that event is incomplete. This does not wait for the payload
transfer or publish an ACK. Device metadata reuse stays ordered after the
previous payload by stream FIFO. Even a partially submitted metadata upload
records its retirement event before propagating a staging failure.

Ready flags are valid only for a full-geometry H2D transfer. Consumers first
wait for the current generation's flag initialization event, then its layer
flag. Workspace reuse also waits for the previous transfer's completion;
layer readiness alone does not retire metadata still read by the transfer.
On submission failure, retirement fences protect reuse without publishing a
successful load ACK. If neither event publication nor stream synchronization
can establish retirement, the executor must reject further loads.

Only an op's copy completion event acknowledges it, so every op on the
wire carries at least one transfer, and no (group, source, destination)
repeats within one plan: a store skips keys already in flight, a load targets
freshly acquired pages. The scheduler asserts both when batching a plan's ops
and the runtime refuses an op with nothing to copy; neither side dedups or
invents an acknowledgment, because an op that never completes a copy would
hold its tickets forever.
A load-back's source is a published Host entry. L3 may return partial or
missing objects before admission; publication requires complete rows and proven
native retirement. The existing group matcher chooses the reusable boundary.

### Retraction image: two Host tiers, one executor, one ownership translation

Retraction suspends a resident request instead of recomputing it: the
request's cache-group pages and its per-slot Device state are imaged to
pinned Host memory, and readmission allocates Device pages again and copies
the image back, so the request continues exactly where it stopped. The image
is **split by publishability** (the C++ coordinator's `TakeImage` decides per
slot; `scheduler.md` §2 and §4 describe the FSM side):

* the **bulk** — every block the group publishes: those under a
  hash-complete prefix page, plus the partial last page and the live state
  block the retraction publishes as a tail, like a finish does — goes to Host
  L2 as today's stream-ordered retraction store (`WriteBackOp`,
  `source_pinned=false`), and the `Retracted` request pins those Host
  entries until it is restored. The entries stay published, so other
  requests may hit them; they are merely unevictable for that span, so L2's
  evictable capacity shrinks only by the retracted set, never by a separate
  copy of it. A published block is never written again, so the restore
  copies the victim's tail blocks into fresh blocks instead of claiming them;
* the **tail** — every block of a group L2 never holds (a drafter-private
  group, a replayable group's private pages, a state group's unpublished
  slot 0, a partial page the index did not take) and the request's slot-state
  blob — goes to the **retraction snapshot pool**: a small, request-private, pinned pool
  sized like the Host L2 tier (an explicit `--retraction-snapshot-host-gb`,
  else `--retraction-snapshot-ratio` of the Device KV, else derived: by
  default one tail per retractable request, `cache/l2/sizing.py`), never
  evicted, never prefix-indexed, allocated by the scheduler like the Host L2
  pool and addressed by scheduler (virtual) block id. Without the L2 tier
  (`--disable-kvstore`) the whole image goes there, and the derived default
  is a tenth of the Device KV -- an on-by-default pool stays cheap to pin;
  `--retraction-snapshot-ratio 1` makes every resident suspendable.

The runtime side is **one** `HostCacheExecutor` (`cache/l2/executor.py`)
owning two `HostCacheStorage` buffers over the same field geometry, each with
its own `build_transfer_geometry`; the executor only chooses a buffer per row.
Wire ops (`cache/transfer/ops.py`, the runtime image of the scheduler's
`SnapshotStoreOperation`/`SnapshotRestoreOperation`): a `SnapshotOp`
(`op_id, request_id, request_pool_index, snapshot_slot, transfers`) stores
the tail rows and the victim's slot state; a `RestoreOp` adds a `source_tier`
per row (`HostTier.L2` or `HostTier.SNAPSHOT_POOL`) and carries the prefix
key on its L2 rows so the scheduler can publish the restored Device blocks.
The two legs of a store are two ops under **one fence**: the ordered L2 rows,
then the tail rows, then the slot-state export ride the write stream
consecutively behind the execution stream the forwards wrote on, and the
forward thread's default stream waits on the event recorded after all of
them before the plan's zeroing (the victim's Device pages are re-granted in
the same plan). The restore is one op: on the load stream behind the
zeroing, the L2-tier rows, the pool-tier rows, the slot-state import, one
event — no layerwise tracker, because the request is `Restoring` and
unschedulable until the ACK. Three ACK kinds ride the one cache poll
(`WriteBackDone`, `SnapshotDone`, `RestoreDone`) and the control plane
(`ts-control`'s `cache_ops`) replica-intersects them alike; a readmission
needs both store ACKs.

**Slot-state image.** Device state keyed by `req_pool_index` that lives in no
cache group must travel with the pages or the continuation is not exact: the
next step's inputs and the verifier's recorded draft distributions
(`RuntimeStates`), the MTP stash, DSpark's context windows, Eagle's draft
history frontier, the Inkling conv ring and its pending-hydration bit, the
DSA KPool tail, and the sampling backend's per-request state (the sampling
scalars, the penalty token counts and logit bias of the `*_full` backends,
the FlashInfer backends' per-slot coin generator) — everything its rid-flip
reset would otherwise reinitialize, so the import claims the new slot for
the request and penalties and seeded sampling continue exactly. Each owner
implements `SlotStateExporter` (`execution/slot_state.py`) by listing its
per-slot rows once (`slot_state_rows`); the export, the import, and the size
derive from that list. Every node of an attention tree is its own exporter
(`slot_state_exporters` flattens the tree), and `ModelExecutor` lists
`RuntimeStates`, the attention nodes, the drafter, and the sampling backend
in blob order. The Host
cache executor measures them **once**, at construction (`SlotStateLayout`:
sizes are fixed for its lifetime), so a store or restore only slices the
arena row at the recorded offsets and never re-measures an owner on the
forward thread. The executor keeps a pinned arena of
`--retraction-snapshot-max-requests` × `blob_bytes`, indexed by the op's
`snapshot_slot`; the store exports the victim's slot on the write stream
(the slot is reused only behind the fence), the restore imports into the
new slot and lets a request-keyed owner claim it. A request-keyed owner —
the sampling backends, DSpark — prepares a slot at the request's first
forward, so it images a **prepared marker** ahead of its rows: a victim
retracted before any forward on this engine (a PD decode role's request
between its landing and its first decode) still holds the previous
occupant's rows, which are not imaged; the restore then leaves the new slot
unclaimed and the first forward prepares it from the request's own
parameters, as the unretracted request's would have. Token-derived rows —
the committed-token history and the n-gram tail — are not imaged; they are
reseeded from the control plane's token list as on any slot handoff.
`test/runtime/test_slot_state.py` constructs these owners (and a
`ModelExecutor` over them) and enumerates every tensor they or their runtime
components allocate with a slot-sized dimension, failing when one is neither
exported nor declared in the exporter's `token_derived_slot_state` /
`constant_slot_state` (required class attributes, empty or not): backend-
private per-slot state is the exception (`AGENTS.md`), and when it exists it
must have an exporter -- and a builder in that test to be guarded.

**One ownership translation for every leg.** The scheduler places every Host
block — L2 or pool, store or load or restore — in the same residue class as
the Device block it mirrors, so under KVP one rank owns both ends of every
copy it performs. `BlockOwnerTranslation` (`cache/transfer/ownership.py`)
applies the same `owned_local_pages` placement the zeroing and PD paths use
to both ends of every row at the one place rows are formed
(`HostCacheExecutor._append_transfers`): a sharded group's row is kept by the
owning rank alone and translated to local ids on both sides, a mismatch of
owners is an error, and a replicated group translates to the identity. Each
Host tier holds one translation (same Device bound, its own Host bound,
`1 + host_lcm_blocks × packing × shard_count`). An op whose every row belongs
to other ranks is still acknowledged by this rank from an empty copy; the
control plane's replica intersection completes the op only once every owner
has. The
scheduler side (residue-class allocation for every tier transfer, the Host
pools registered with each group's `shard_count`) is what makes the
translation exact, so both Host tiers serve a KV-page-sharded engine. L3
uses the same Host owner translation before Device admission. A sharded
object key names the replica (`attention_tp_rank // shard_count`) rather than
the writer's cyclic owner. Its bytes must be replicated within that owner
subgroup; startup rejects head-partitioned fields in a page-sharded group.
Replicated state groups keep the full rank in their keys.

## block vs. page

**`block` is the general concept; `page` is its specialization under
KV-cache-based attention.** A block is one addressable cache unit of a
group — one block-table slot, one `CacheBlock`, spanning `block_granularity`
tokens when the group has a span — regardless of what it holds. A page is a
block whose contents are rows of KV entries consumed by paged attention
kernels. Every page is a block; a recurrent-state block is not a page.

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
* **`block_table`** — the table used by state-based attention (such as
  Mamba / linear attention state slots), and the generic name for the
  container that carries per-group tables between scheduler and runtime.

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
  policy. `FullAttnMatcher` walks left-to-right until the first miss
  (prefix-closed); `SwaMatcher` scans right-to-left for a run backing a
  resumable boundary (non-closed); `StateMatcher` answers one key per
  boundary — a state checkpoint is a single block under the key of its token
  count, with no run behind it. `Probe` takes the L3 hit set as a required
  argument (`nullptr` when storage is unset) so Host-only matching cannot be
  selected by omitting it. A matcher only *reads* the group's index; it never
  touches allocation or physical placement.
* **`TailTable`** (`tail_table.h`) — the coordinator's index of published
  tails: under the chain hash of the last full page (the *prior*), the
  `(T mod P, hash@T)` pairs published so far in least-recently-used order
  — a publication or a hit moves a tail to the back — capped at
  `kMaxTailsPerPrior` (8) with a counter of the drops
  (`Scheduler.tail_cap_evictions`). The probe enumerates it to find the
  candidate boundaries inside a page; eviction of a tail's block removes its
  entry.
* **`BoundaryTree`** (`boundary_tree.h`) — the coordinator's tree of the
  boundaries it indexes, shared by both tiers: one node per content
  hash either tier holds, linked to the chain hash of the page before it (a
  tail to its prior), counting per tier the *checkpoint boundaries* —
  boundaries where some State group holds a checkpoint on that tier — in its
  subtree, and listing the Host pages no Host page continues (the Host
  leaves). A publication links its new pages and tail — and any node of its
  pages that an entry started without a parent (below) — before any group
  registers under them, so a checkpoint counts in every ancestor as it is
  published; a node lives while either tier holds its hash, an eviction
  write-back is bringing it to the Host, or a child hangs under it, and goes
  with its tail hint once none of these holds. An entry registered with no
  node yet (an L3 GET, a store whose Device entry left before its ACK, a
  restore's republication) starts one without a parent. It is kept in step
  with the indexes: no match reads it, evicting by segments selects along it
  (`AdmissionPlanner`, and the Host's segment plan, `AcquireHostSegments`),
  the collection of dead prefix entries finds them through it, and its walks
  follow parent pointers and insertion-ordered lists, never the map, so
  mirrored ranks agree.
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
  (full attention never expires; SWA slides whole pages out; a State group
  has no token span and never expires — its two slots turn over by
  `StateStep`); the manager only *executes* the resulting block count.

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

* **Prefix probe and admission.** `ProbePromptPrefix` is a read-only lookup
  of prefix hits per group on both tiers, converged to the common boundary
  across the *matched* groups, in two phases per tier: the chain walk over
  the prompt's full pages (`FullAttnMatcher`, prefix-closed groups first,
  then the window groups within that bound), then the **tail phase** -- for
  each offset the `TailTable` lists under the last matched page's hash,
  longest first, the probe hashes the prompt's own tokens up to it, skips a
  candidate whose hash differs (the publisher diverged inside the page), and
  takes the first `hash@T` every matched group holds (a closed group all of
  its blocks below `T`; a sliding group the blocks holding the window
  `[T − W + 1, T)` the first query after `T` attends --
  `CacheCoordinator::SlidingWindowBlocks`, one block more than behind a
  page-aligned boundary when the window's first row falls inside a block;
  the tail's blocks below that window stay holes). The Host tier extends
  the Device under the **Host-suffix rule**
  (`CacheCoordinator::probeHostSuffix`): a hit at `T = p + t` takes the
  Device's KV pages below a split `F` -- the Device's prefix-closed KV coverage `L` in
  full pages, or `p` when lower, lowered further to the page of the first
  block a sliding group's window behind `T` needs and the Device lacks,
  even when no Device window can resume on its own --
  the Host's KV pages `[F, p)`, the KV tail whole from one tier (the
  Device's only when `F == p`), and each State group's checkpoint at `T`
  from the Device when it holds it, else from the Host. So Device KV plus a
  Host checkpoint resumes -- a checkpoint on the Device's last full page, or
  a shorter one under longer Device KV -- and a Host hit never reloads KV
  the Device holds below the split. (A sliding window that lowers the split
  does load the pages `[F, min(L, p))` from the Host though the Device holds
  them: the split is one page for every KV group.) Boundaries are tried
  longest first above the Device-only hit, longer tails in its page
  included; a gap stops a tier's run, so a hit is always one
  contiguous Device prefix plus one Host suffix (KV interleaved
  Device/Host/Device is not stitched together). `PrefixProbe::device` then
  holds the Device part (`full_page_tokens == F`, a Device KV tail and the
  Device's checkpoints), `PrefixProbe::host` the Host part from `F` on (a
  State group's hits count from page 0 on both tiers, since a Host
  checkpoint may sit on the Device's last page); when the Device alone
  resumes furthest the Host tier mirrors its boundary and claims nothing.
  `AdmissionResult::host_prefix_tokens` is a boundary, not a token count:
  the Host's when a Host suffix carries the hit, the Device's otherwise (0
  without a Host pool and on the D role). Such a hit loads the KV beyond the
  Device's prefix (none when only checkpoints come from Host) and adds, over a
  Device-only hit, that boundary less the Device-only one (the scheduler's
  `PrefixHitStats` counts both).
  Storage (L3) queries and reserves under the same rule, with temporary object
  hints standing in for Host entries only, so it never asks for KV
  or a checkpoint the Device's prefix supplies: candidates are full pages,
  matching local `TailTable` entries and the exact reusable request bound.
  It has no shared index for discovering other nonaligned remote
  checkpoints. The probe is bounded
  in tokens, not pages, so a hit may end exactly at the bound. `Admit` then
  allocates, pins the hit prefix, and produces host→device `load_pairs`
  plus each group's fresh pages and `pages_to_copy`. **A table never keeps a
  tail entry's block**: the block holds rows of a page the request later
  publishes under the page's own chain hash, and one block carries one key.
  So every tail block a hit claims becomes the hitter's own -- a
  Device→Device copy of the shared block (`AdmissionResult::pages_to_copy`,
  `(source, destination)` virtual ids in one bucket, the destination in the
  hitter's table and never zeroed) or a fresh block where the probe left a
  hole -- and a Host tail is loaded into fresh private pages like any Host
  hit and never republished on the Device. The owner's side of the same
  rule is **copy-in**: an admission whose publication point the owner
  keeps writing past (`CompletedPages::owner_continues`, the fused role's
  prompt end) registers the owner's tail blocks under `hash@T` and swaps a
  fresh copy of each into the table, so the published bytes are exactly
  what the prefill forward wrote and nothing writes them again. The copies
  are the one thing such an admission may need beyond what the completing
  chunk reserved, so they are best effort: when the plan does not fit with
  them (under page-cyclic sharding a copy may need a parent while other
  buckets still have holes), `Admit` replans without the tail -- the owner
  keeps its partial block private and publishes neither the tail nor the
  state checkpoint keyed by it at that point (its full pages still publish;
  its finish publishes the frontier later). A request whose reserve covers
  its generation therefore never waits on a block it did not reserve, which
  is what keeps a pool full of such requests from wedging at their first
  decodes. Copies are
  fire-and-forget: the runtime orders them on the forward thread's default
  stream after the execution stream's in-flight forwards and before the
  plan's zeroing, so a source another admission of the same round evicts
  and re-grants is read before it is zeroed -- the argument the retraction
  image's stream-ordered legs already rest on. A
  **replayable** group (`CacheGroupSpec::replayable`) is not matched
  at all: it never constrains the boundary and never claims a hit, so its
  table is empty at first admission and the scheduler materializes it as a
  sparse private suffix from the replay window's first token
  ([Scheduler §1.3](scheduler.md#13-bounded-replay)). Probe and admit are
  deliberately split so the probe can be taken once and the admission retried
  against it — the scheduler's same-round retract-and-grant re-runs a failed
  admission after freeing a victim (see `scheduler.md`) without re-probing.
  `Admit` claims exactly the published Device and Host entries the probe
  found. Temporary remote hints are used only while planning and reserving
  a storage read. `ReserveStoragePrefix` matches the hinted boundary and pins
  its Host dependencies; ordinary local probes never expose remote hints as
  resident data. See [asynchronous L3](async-l3-cache.md).
  `ProbeDecodeDevicePrefix` is the PD-decode variant: local history pages
  are reused, and of a State group's checkpoints only the one at exactly the
  prompt end counts (`PrefixProbe::state_at_prompt_end`, claimed as slot 0
  so the peer omits the state transfer; any shorter checkpoint is ignored,
  since this node runs no prefill that could continue from it). Its tail
  phase leaves the partial block a hole: the peer's transfer starts at the
  block containing `hit_tokens` and fills a fresh page whole, so the D role
  never copies it; a tail's full blocks below that page are copied like on
  every role ([Scheduler §3.2](scheduler.md#32-d--decode-worker)).

  `Admit` takes two inputs of different scope and tense. One `GroupDemand`
  per group says what that group needs for the round ahead: an extent and a
  reserve beyond it. The extent is one of three shapes — `DenseGrowth`
  appends tokens relative to the table's current fill, `SparseSuffix` names an
  absolute token extent and the first slot to materialize, leaving the slots
  below as null holes, and `StateStep` names what a State group's two slots
  do (rotate, release, keep, land) — so the bounded-replay rewrite in `Admit`
  is a visible conversion from the first to the second (hit plus growth
  becomes the absolute extent), not an arithmetic side effect. One
  `RequestProgress` per
  request says what the request has done since the coordinator's previous
  transaction for it: what it completed (`CompletedPages`, present only when
  there is something to publish -- newly hashed pages, a tail, or at a final
  point the frontier's state -- so "new hashes without a boundary kind"
  cannot be expressed) and its computed-token count for retention.
  Publication fields are request-scoped and therefore live on the progress,
  not replicated onto every group's demand.

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
  `RequestProgress` for finish, retraction, and remote completion, which
  publish without admitting.
* **Prefix publication.** `CacheFullBlocks` / `CacheCompletedBlocks` register
  computed blocks into the prefix indexes for later requests. Prefix-closed
  groups match first; non-closed groups (SWA, State) match only within the
  boundary the closed groups settled (`match_order_` enforces this).
  A publication point is the request's computed frontier `T`
  (`Request::NumComputedTokens()`); when `T` is not a multiple of
  `prefix_granularity` the partial page holding rows `[kP, T)` is published
  too, as a **tail**: every History group registers its blocks below `T`
  (`ceil((T mod P) / block_granularity)` of them, the last one partial) under
  `hash@T = HashPrefixPage(tokens[kP, T), H_{k-1})` -- the same chained page
  hash over fewer tokens; its `token_count` frame keeps it distinct from the
  page's chain hash and from every other offset. Finish and retraction publish
  the tail in place (`CompletedPages::tail`, `PublicationPoint::kFinal`): the
  block is final for its owner, so it passes to the index like a completed
  page. The coordinator keeps a `TailTable` (`cache/prefix/tail_table.h`)
  of the offsets published under each last-full-page hash, capped at eight
  per prior in least-recently-used order — a hit or a republication keeps a
  tail young, so a hot shared tail outlives any number of one-off prompts
  ending in the same page (the dropped tail's blocks are evicted from the
  Device and the Host tier alike, except those still pinned;
  `Scheduler::TailCapEvictions` counts them) — so a probe can enumerate the
  candidate `T`s without knowing them. A tail's blocks are usable only
  together: evicting one Device block of a tail hash evicts every group's
  entries under it (`evictCachedBlock`), and the table forgets a hash once no
  tier holds an entry for it and no eviction write-back is bringing one to
  the Host: a hint stays while such a store is out, and when the cap drops
  it meanwhile, the store's settlement cancels the tail instead of
  publishing it (the ACK never records a tail). A published partial page is never written
  again: a restore copies a victim's partial page into a fresh block (an
  unkeyed L2 row, not republished) instead of claiming the published one.
  Host-warm first-chunk extensions call `CacheFullBlocks` at admit (the
  layer-wise load makes the bytes available before any forward reads them);
  a load-back's `BlockTransfer.key` is therefore empty and `CompleteLoadBack`
  only drops the op's pins. For L3, `StorageTransferManager::Complete`
  publishes successful rows through `CacheHostBlock` after all-owner native
  retirement, before admission can see them. There is no mixed Host/L3 hash
  to special-case.
  A State group publishes its slot 1 — the state after exactly `T` tokens —
  under the same key as the KV at `T` (the chain hash of the last page, or
  `hash@T` for a tail), and only at a point where the owner stops writing it
  (`CompletedPages::publish_state`: a prompt end, a junction cut, finish,
  retraction); a chunk end or decode step publishes no state (see
  [Scheduler §1.2](scheduler.md#12-state-groups-two-slots-natural-checkpoints)).
  On Device, evicting a History block under a chain hash also evicts the
  State checkpoints under that hash, so Device state does not outlive the
  KV it continues; a Host checkpoint can resume on the Device's KV of its
  page, so evicting a Host KV block takes no checkpoint along. Evicting one
  State group's checkpoint evicts every other State group's checkpoint at
  the same hash, on the Device and on the Host tier alike, since a resume
  needs all of them; it takes no KV along, which still serves the
  boundaries past it.
  A finish that publishes the end state of a request which resumed at a
  Device tail checkpoint marks that input superseded
  (`MarkSupersededStateInput`): its Device entries are registered again at
  access epoch 0, still matchable but first in the eviction order, and an
  admission or a restore that evicts them counts them as request reclaim
  ([Scheduler §1.2](scheduler.md#12-state-groups-two-slots-natural-checkpoints)
  gives why and when a later request unmarks it).
  Replayable groups are outside `match_order_` and skip publication on both
  tiers — never registered, never streamed to Host, never counted by
  `DeviceBoundaryResidency` — because their rows are approximations the
  model regenerates from re-fed tokens, not a function of the prefix alone.
  Retention (`ExpiredBlocksAt`/`ReclaimExpired`) still slides their
  request-private pages out.
* **Two tiers (Device/Host).** What feeds the Host tier from ordinary
  Device prefix entries is an explicit cache policy,
  `SchedulerConfig::host_write_policy`: `Validate` requires it with a Host
  tier and has no default to fall back on. Under `kWriteThrough`, Device
  prefix publication streams to the Host tier
  (`stream_device_cache_to_host_`) on the fused and P roles; the D role's
  Host tier holds retraction images only. Under `kWriteBack` (not on the D
  role, at most 64 groups, no L3: `Validate`)
  nothing streams at publication; what an admission or a restore evicts is
  written back instead (below).
  Under `kWriteThrough` a `pending_stores_` queue drives D2H transfers,
  alongside Host-side acquire/contains/pin queries. Host L2 uses free
  capacity first, then global LRU across cache groups: a compatible child in
  the requested group competes with fully evictable LCM parents across
  groups. When only one group needs
  replacement, its own parents are omitted: the initial allocation pass
  exhausted the requested buckets' holes, so each such parent has a compatible
  child no newer than its newest child and cannot beat single-child eviction.
  For a history group in this case, an ordered cursor collects only enough
  children for each unresolved bucket, then stops after that epoch. The cursor
  is destroyed before eviction mutates the index. Donor groups, State groups,
  and batches with multiple replacement groups retain full candidate scans.
  Cross-group reuse must free a whole parent, ranked by its most recently
  accessed child, so a cold sibling cannot cause newer data to be discarded.
  Equal epochs prefer
  evicting one child; group and physical location break remaining ties. A
  prefix hit refreshes its access epoch, and earlier reuse grants no
  additional protection. Pinned entries, partially pinned foreign parents,
  and blocks allocated earlier in the same batch remain protected. Parent
  candidates are snapshots; selection skips parents that are now empty or
  bound to another group. Parent aggregation reuses coordinator-owned scratch
  storage and resets and examines only touched parents, rather than the whole
  Host arena on every allocation batch. Child reuse preserves the requested
  placement bucket. Host eviction removes each selected entry independently,
  except that a State checkpoint takes the other State groups' checkpoints
  at its hash along; each block retries free capacity before it picks a
  victim, so it places in the room such a cascade freed.
  No group has an eviction preference or a fixed share of the pool; capacity follows the access
  pattern, without guaranteeing that KV and every required state checkpoint
  remain resident together. During prefill, each chunk end queues the
  full-attention pages it published; the prompt end (the first decode's
  admission, or a junction cut) queues its tail and state checkpoint too;
  the candidates are merged into a batched writeback. Ordinary decode
  publishes history-cache Device entries but streams none of them and
  publishes no state. At finish — and at the PD transfer ACK — every entry the
  request published (pages, tail, state slot 1) is queued before request
  ownership is released. Under write-through, ordinary sliding-window
  entries stream whenever published. The queue is drained by
  `TierTransferManager::StartPendingStores(guard, cause)`, and every store it
  issues pins its Device sources until the ACK. A retraction does not use the
  queue: the scheduler publishes the victim's computed prefix the same way,
  `CacheCoordinator::PublishedDataSlots` names the data slots that are
  published entries (with the key, logical index, and boundary kind each was
  published under), `TierTransferManager::SplitRetractionImage` plans their
  Host blocks before anything is acquired or evicted for them — keys already
  Host-cached, or carried by a store in flight that publishes them at its
  ACK, are pinned; of the rest, the Host takes the leading boundaries that
  fit whole
  (`CacheCoordinator::NumHostBoundariesThatFit`) — and
  `StartRetractionStores` carries out that plan: the taken rows become one
  stream-ordered write-back whose Host entries the image pins, and
  `CacheCoordinator::TakeImage` places only the remainder (unpublished slots,
  and the published rows the Host does not take, as planned) in the
  request-private **snapshot pool** on the same stream-ordered footing —
  because the victim's sources are re-granted in the same round
  (`scheduler.md` §2).
  The snapshot pool is a third `BlockPool` the coordinator owns beside Device
  and Host L2: never prefix-indexed, never evicted, its blocks held only by
  the `Retracted`/`Restoring` state that imaged them, and `Validate` requires
  it to be stated on every role (the null page alone means no image ever
  fits, so every capacity block aborts its victim — `scheduler.md` §2).
  The way back is one `SnapshotRestoreOperation` whose rows name their Host
  tier; its L2-tier destinations are republished at the ACK
  (`CompleteSnapshotRestore`) exactly as a prefix load-back's are.
  Under write-back (`write_back_evictions`), each victim set an admission or
  a restore evicts for the cache is recorded as one *segment* as it leaves
  the index (`evictCachedBlockAlone`); the request's own released slots are
  dropped, and so is a block no probe could resume from on the Host
  (`DeadEntriesNotWrittenBack`). A record holds its tree node, tail hint and
  Host presence until its segment settles, and a block the Host holds already
  rides as that entry, never copied again. The invariants: eviction keeps the
  Device's coverage closed from the root (a restore can open it, below) and a
  segment is never split; the plan that evicts drains its segments right
  after the build (`TierTransferManager::StartEvictionStores`), as one
  stream-ordered copy the runtime orders ahead of the plan's reuse of those
  pages, and no record outlives its plan; segments that depend on each other
  are placed together; and a segment publishes whole, at the last ACK it
  waits for, with its Device metadata, or not at all (`SettleEvictedSegment`,
  which also drops what a cap or the loss of every checkpoint boundary under
  it made dead meanwhile).
  A superseded input (above) an admission or a restore evicts is dropped, as
  the reclaim of the request that superseded it.
  Write-through's publication and finish stores and L3 prefetches take Host
  blocks one by one (`AcquireHostBlocks`), a retraction image's leg by the
  plan `SplitRetractionImage` made (`AcquireHostBoundaries`; under the
  write-back policy whole segments first, `scheduler.md` §2), and what the
  write-back policy writes back per segment (`AcquireHostSegments`), and a
  segment arrives whole: every segment's blocks are fitted jointly before
  anything is evicted, a segment is placed whole or not at all, and nothing is
  evicted for a dropped one. Each row's Host block is fitted and placed in its
  Device block's bucket, so under KV parallelism every rank copies only the
  rows it owns; a segment one bucket cannot fill is dropped whole, even when
  another bucket has room. After dead and Device-covered segments, before
  taking Host-only cache, placement reclaims
  unpinned prefix-closed KV copies within a contiguous Device KV prefix,
  independently of Host checkpoints. A Host-only checkpoint can still resume
  on that Device prefix. Nonaligned KV tails stay whole, and sliding-window
  lookback pages at checkpoint boundaries stay on the Host: a missing Device
  window can lower the common tier split and require those closed-group pages
  too. When State groups share the pool, KV still held on the Device has its
  Host duplicate reclaimed first, newer Device copies first: the request keeps
  that Device storage alive. Unheld Device KV can be demoted on the next
  admission. Its duplicate competes with Host-only checkpoints by access epoch
  (Device KV epoch versus Host State epoch), oldest first, with duplicates
  winning ties. This lets unused State age out before newer KV is repeatedly
  copied back. Models without State groups retain duplicate-first reclamation.
  No group has a reserved capacity; every freed parent returns to the shared
  pool and can serve either State or history demand.
  The boundary tree maintains the nodes with settled closed-group duplicates
  as entries arrive and leave. It also propagates complete Device-prefix
  coverage when a page changes, including gaps opened or filled by restores.
  A placement therefore visits duplicate nodes and their bounded sliding
  lookback neighborhoods, rather than rescanning every cached prefix.
  These candidates share the segment plan's shadow and restore pass, so an
  unplaced segment or an unneeded duplicate evicts nothing. Their evictions
  count as `HostEvictionStats::duplicate_kv`. Remaining Host victims are whole,
  never in use, and never take or sit right under a page an eviction write-back is still
  bringing, which would otherwise arrive with nothing under it; dead entries
  go first, Host-only cache last. A Device checkpoint can depend on Host
  history, such as a window missing on the Device: no Host checkpoint at that
  boundary does not make those entries dead. Checkpoints on both tiers count
  when deciding whether Host history is still needed. State groups at one
  boundary may also be split across tiers; an incomplete Host portion is
  still useful when the Device holds the other groups. These entries keep
  their ordinary eviction rank. `AcquireHostSegments` documents the victim
  shapes, the ranking and the restore pass.
  Host-only checkpoint recency follows its State entries, independently of
  the KV entries at that boundary. A longer prefix hit can traverse those KV
  entries while using only a deeper checkpoint; that traversal must not keep
  the shallower, unused State young. KV-only victims retain KV access recency.
  A history-block pin protects that block, independently of the State at the
  same boundary. If it prevents a whole Host segment from leaving, the planner
  may instead reclaim that boundary's unpinned State groups together. It leaves
  every history block in place, still respects pending write-backs, and restores
  the checkpoint when no placed segment needs its space. An actual State pin
  still protects the whole checkpoint. This changes reclaim eligibility, not
  allocation: State and history continue to share the same LCM pools.
  A deeper Device checkpoint cannot restore State at a shorter boundary.
  Aligned Host checkpoints therefore retain their State LRU rank even when
  a deeper Device checkpoint shares their State access epoch. These page
  boundaries remain useful when a later prompt diverges before the deeper
  frontier. An exact Device copy of the same boundary remains a duplicate.
  For a nonaligned input checkpoint, a deeper Device checkpoint with the
  same State access epoch retains the existing continuation priority: old
  input tails must not crowd out other sessions after the request advances.
  This is a retention choice, not a claim that deeper State restores the
  shorter boundary; it does not reserve capacity for State.
  Duplicate-KV planning reads each tier's location, access metadata and pin
  status together through `PrefixCacheIndex::Inspect`. This snapshot holds no
  block reference and does not update recency, so collecting candidates neither
  changes eviction eligibility nor needs a second lookup for Device metadata.
  Its pin status is valid only until ownership changes; a later plan inspects
  the entries again.
* **L3 under flat KV.** Host L2 is one compact pinned byte buffer indexed by
  CacheBlock IDs. D2H ACKs are replica-agreed and publish Host entries without
  waiting for L3. Independent PUT tickets pin resident Host sources at
  dispatch; failure never invalidates a usable L2 entry. A separate GET
  transaction fills Host before Device admission. Runtime workers receive
  immutable CPU byte geometry and no Device buffers or transfer layouts.

  Object keys are
  `{tsl3-<sha256>}_{content_hash}|g{group}|o{page_offset}|r{object_rank}|c0`.
  `object_rank` is the attention-TP rank for replicated groups and the payload
  replica rank (`tp_rank // shard_count`) for cyclic page groups. The literal
  `c0` suffix and `cp_size: 1` namespace field are reserved format fields;
  compatibility is determined by the complete namespace, not these literals.
  Changing its compatibility epoch starts a cold cache.

  The static deployment identity (`storage_deployment_id`) covers the loaded
  target and optional draft checkpoints, packed Host layout, effective HF
  text-config overrides, cache quantization and scale-file bytes, pipeline
  stage, resolved attention backends and KV-producing numerical/parallel
  choices. These include attention-TP, KVP, QCP and linear-attention geometry,
  target/draft compute choices, softmax-skip settings and resolved EAGLE3
  capture layers. The packed-cache signature records target compute dtype and
  field payload geometry; Device arena offsets that depend on GPU capacity
  are excluded. Backend identity includes hybrid full-attention choices and
  MSA's dense sub-backend. Matching layout alone does not prove that two
  producers compute compatible KV.

  Checkpoint identity follows the bytes the loader selects. An immutable
  revision or a validated Hugging Face hub snapshot identifies a checkpoint;
  an unpinned/local checkpoint is fingerprinted from selected weights,
  metadata and local Python modules. An inherited config `_commit_hash` or a
  directory merely named `snapshots/<hex>` is not sufficient. Fingerprinting
  follows imported package directories and symlinks with cycle detection.
  It respects `--load-format`: rank-local `sharded_state` digests are agreed
  across the explicit replica CPU group, `npcache` includes the NumPy files
  actually loaded, and Mistral includes its shard index. `extensible` includes
  the YAML definition, its imported definition module and transitive local
  helpers, even when the weights reside in an immutable hub snapshot. The
  definition path uses the loader's process-working-directory resolution.

  `L3_RUNTIME_COMPAT` is the explicit compatibility epoch for built-in model,
  positional-encoding or kernel changes that alter KV without changing the
  checkpoint, layout or recorded producer choices. A git SHA would split
  caches on unrelated changes; a package version may stay unchanged across
  KV-affecting edits. Static facts are captured once. The namespace
  (`ts-l3`'s `key_prefix`) combines their digest with explicit target/draft
  weight versions, without re-reading mutable launch options or checkpoint
  files on version updates.
  Use resolved `mapping.attn.tp_size`, not `--attn-tp-size` alone: different TP
  widths can have identical per-rank packed geometry while assigning different
  heads to a rank.

  Host eviction leaves remote objects intact. Cross-instance reuse uses
  asynchronous query/GET/PUT tickets. The scheduler applies temporary query
  hints through the group matchers, including sparse State checkpoints, and
  reserves Host destinations before Device admission. After a GET retires on
  every owner, only rows with all-owner full-length success are published.
  Ordinary matching chooses the usable boundary after partial results. A
  query hit is not a lease; the object can disappear before GET. Local Host
  publication likewise does not prove remote existence. C++ owns waiting and
  Ready holds, soft deadlines and pressure admission.

  A full clear or weight update first agrees flush intent across attention DP,
  then probes local clearability across cache-owning ranks and DP. Clearability
  includes outstanding Host copies and storage operations, Ready holds and
  quarantine. Only after agreement does synchronous remote namespace deletion
  run; its result is agreed before any rank clears Device/Host indexes. A
  failed preflight or purge prevents local clearing, and the frontend combines
  every DP worker's reply. The administrative purge is not bounded by the
  preparation soft timeout. Independent jobs sharing the namespace are
  outside this agreement; a fleet-wide flush requires operator coordination.

  With L3 enabled, a weight update requires an explicit `weight_version` and
  successful flush before the GPU load. Automatic per-process version labels
  could collide across independent checkpoints. The new prefix is published
  only after a successful load, and the Engine facade records that version in
  `server_args`. The metadata-only `/update_weight_version` endpoint rejects
  updates because it cannot coordinate worker namespaces or cache clearing.
  See [asynchronous L3](async-l3-cache.md) for ownership and retirement, and
  [server configuration](../configuration/server.md#host-l2-and-mooncake-store-l3)
  for deployment settings.
* **Reclamation and lifecycle.** `ReclaimExpired`, `Free`,
  `ClearDeviceCache`/`ClearCache`, and `NumNewlyReleasableLcmBlocks` for
  ranking retraction (preemption) victims. Device and Host evictions are
  counted by reason (`DeviceEvictionStats`, `HostEvictionStats` in
  `cache_coordinator.h`); the counts are accounting only and never change
  what is evicted. Under the write-back policy (`write_back_evictions`) the
  coordinator also collects *dead prefix entries* on both tiers, entries no
  probe can resume from any more: everything under a page no checkpoint
  boundary is left under on either tier (with State groups), a tail whose
  hint the per-prior cap dropped, and everything at and under a *gap* — a
  page no prefix-closed group holds on either tier, whatever State entries
  or successors it keeps: a prefix-closed group cannot cross it, so not
  even the page's own checkpoints resume. Nothing is dead while an eviction
  write-back is bringing it to the Host. Eviction queues candidates as it goes
  (and a write-back's settlement every boundary it did not publish), and
  `CollectDeadEntries`, once per plan after the build and the write-back drain,
  judges each again by the state it finds (a live request may have published a
  checkpoint under it since), then evicts the dead subtree on both tiers,
  deepest boundary first (`DeviceEvictionStats::dead_entry_collection`); nothing
  a live request's table holds is taken, and what is in use, or has a store
  still on its way into it, waits for a later plan. `ClearDeviceCache` writes
  nothing back and empties the queue: a Device-only clear loses what only the
  Device holds and never collects Host L2. Host placement's dead class
  (`AcquireHostSegments`, above) counts the Host's checkpoint boundaries
  only: a Host page with checkpoint boundaries under it, but only on the
  Device, is dead there (ranked first, counted under
  `HostEvictionStats::dead_entry`), though the collection would keep it.
* **Mutation reporting.** `SetCacheMutationSink` reports per-group cache
  insertions/removals; the scheduler folds them into one externally visible
  prefix event. Whether a scheduler-level boundary is fully, partially, or not
  resident is the coordinator's answer (`DeviceBoundaryResidency`, read off
  the group indexes), so the scheduler keeps no residency counters of its own
  — only the token descriptor the event carries and whether that event is
  currently out. A mutation only marks its boundary for reconcile;
  `DrainKvEvents` reconciles each marked boundary against its residency and
  reports the net change (published exactly while fully resident), then drops
  descriptors of boundaries with no cached child. Nothing is decided
  mid-`Admit`, so one admission may evict a boundary's last cached copy and
  then store the request's own copy without any ordering hazard.
  Admission may pre-hash prompt pages beyond the computed prefix. An early
  junction or retraction tail uses the parent hash at its published frontier,
  even when the request's hash memo already contains later pages.
Replica-agreed D2H ACKs publish Host entries and release Device sources
without waiting for L3. Backup candidates are nonowning; PUT dispatch pins
resident Host sources until all cache owners report native retirement.

`MakeCoordinator` is the factory: one `CacheGroup` per `CacheGroupSpec`
(group_id = index), all sharing one scheduler-level `prefix_granularity`
while each manager may use a smaller cache-page token count.
`enable_l3_storage` is explicit; ordinary Device/Host matching does not
claim remote objects as resident data.

### `AdmissionPlanner` (`cache_admission.cpp`, anonymous namespace)

The internal capacity planner behind `Admit`. It runs entirely on shadow
occupancy — never mutating the real pool — and answers: *which cached blocks
must be evicted for this admission to fit, while protecting the current
prefix hits?* The algorithm: first check whether existing local holes plus
empty parents fit with zero eviction; otherwise select eviction candidates
until the plan fits. Request-reclaimable candidates are collected and sorted
once; each cache group loads and sorts one epoch of eligible candidates at a
time, skipping epochs whose entries are all protected or already listed for
request reclaim. The groups' unread epochs are merged by access time before
inspecting block owners: a newer epoch, even if every entry is pinned, is not
scanned to select an older candidate from another group. Complete equal epochs
still participate in every policy tie-break. Selection compares the next candidate from each group with
the next request-reclaimable candidate using one policy: LRU access epoch,
then tier (uncached request-only block → probationary boundary → established
boundary → suffix of a closed prefix). Each candidate expands into a
*victim set* that is selected, restored and evicted whole. Finally, walk the
selected sets in reverse and restore every set that is not strictly required,
yielding a minimal eviction plan in `victims`. `Admit` and `Restore` evict set
by set, oldest first. The admitted request's own released slots (Admit only; a
restore rebuilds empty tables) and a superseded input, which ranks at epoch 0
with the uncached blocks, count as `DeviceEvictionStats::request_reclaim`;
under the write-back policy every other set is recorded as one segment for the
plan's write-back drain. The shadow occupancy is a class of its own
(`ShadowOccupancy`, `shadow_occupancy.h`) over one pool — holes per bucket,
empty parents, the occupants left in each parent a victim leaves — with the
pool's own fit formula, so a planner on any pool the coordinator registered
its groups in tries victims the same way. Both Host plans fit on it over Host
L2: the segment plan (`AcquireHostSegments`) and the boundary plan
(`NumHostBoundariesThatFit`, `AcquireHostBoundaries`).

A victim set is the candidate block alone unless the coordinator evicts by
*segments* (`write_back_evictions`, the Host write-back policy). Then the set
follows the `BoundaryTree`, so eviction keeps the Device's coverage closed
from the root — no Device page outlives its parent — and what leaves is a
checkpoint boundary with the pages only it depends on, which can be written
back. A restore can open that coverage: it republishes the slots it copies
back from Host L2 entries, but a slot the image kept in the snapshot pool
carries no key and stays the resumed request's own, so a page the image found
on the Host already can return to the index below ancestors the pool kept.
The walk skips what the Device does not hold, so such a page does not leave
with the pages above it; it leaves later as a segment of its own, costing
Device room and hits, never correctness. The victim sets:

* a request's released slots take the other released slots at their hash
  (and, at a tail, the tail's other unpinned blocks); a sliding window's block,
  or a block with no tree node, leaves alone;
* a page or checkpoint more than one Device checkpoint boundary depends on is a
  shared page: its checkpoint (a *nested* checkpoint) leaves alone with every
  State group's at its hash, its other blocks wait for the last checkpoint
  boundary under them (below);
* otherwise the set is a *segment*: from the candidate's boundary, climb while
  the Device holds the parent, it serves no other Device checkpoint boundary and
  none of its blocks is in use, a prefix hit this admission keeps, or a released
  slot; then take every Device block of that subtree, deepest boundary first and
  a boundary's checkpoints before its other blocks. A block under the
  candidate's own boundary counts as a stranded suffix
  (`DeviceEvictionStats::stranded_suffix`). One block in use anywhere in the
  subtree passes the whole set over, never a part of it; a branch of the
  subtree without such a block may still leave as a set of its own, and only
  the block-by-block pass below takes part of it.

The climb reads the real checkpoint-boundary counts, not the shadow's, so two
sibling segments chosen in one plan leave their shared pages on the Device for
later. Only when the candidates run out before the demand fits are the shared
pages tried again, oldest first, counting just the checkpoint boundaries no
selected set takes whole: a page only leaving checkpoint boundaries depend on
then leaves with everything under it, so shared pages never stall an admission
that the segments under them, taken whole, would fund. A set that contains an
earlier one takes it in whole, each block keeping the reason it was chosen for;
sets of one plan are therefore disjoint, and the restore pass returns or keeps
each one whole.

When the shared pages leave the demand unmet as well, an admission whose
request no other one holds pages beside (`Admit`'s
`evict_blocks_past_segments`, from `Request::HoldsPages`) takes the candidates
the segment rules passed over again, block by block: a checkpoint leaves
alone with every State group's at its hash, as a nested one does — at a tail
its eviction also takes the tail's other unpinned blocks, the KV block
included, which the plan does not count; a page leaves with the blocks of its
boundary that can leave (none in use, a prefix hit this admission keeps or a
released slot of the request) and everything the Device holds under it,
climbing nowhere, so the coverage stays closed (a boundary that keeps a block
that cannot leave stays on the Device). Each such set is recorded and written
back like a segment, a block no probe could resume from dropped. Only the
last variant of an admission evicts so: what it may drop — the owner's
copy-in (above), an in-forward State checkpoint's block (`scheduler.md` §1.2)
— is dropped first and never pays for such an eviction. Waiting would never
fund such an admission: the blocks in use are its own, as when it resumed
from a checkpoint whose segment holds the pages under it (`scheduler.md`
§1.4). The pass does not always fund it either: a prefix hit the admission
keeps never leaves, so on a KV-sharded pool of a few pages a request alone
can still wait for good (`scheduler.md` §1.4). Beside another page holder it
waits for that holder's release instead, even where write-through, which
evicts block by block, could fund it at once; and a restore, which falls back
to claiming nothing (below), never takes blocks this way.

A restore of a retracted request (`scheduler.md` §4) protects the blocks it
claims like prefix hits. Under segment eviction a set holding one never
leaves, so the claims can pin the only cache there is to evict: the
request's own pages above a page it claims, or its own checkpoint beside
one. When they leave no plan, the restore plans again claiming nothing;
every imaged slot is a fresh copy, and the request's own cache leaves by
segments like any other.

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
  parent, not an allocation. A parent's bytes are exactly its planes': each
  plane is already a multiple of every tenant's packing, so the parent is not
  rounded up again to the LCM of all the packings, which only added bytes no
  field addresses.
* **`bind`** (`CacheLayout.bind(num_lcm_blocks)`) multiplies that parent out
  by a count and yields the `CacheMemoryPlan` the arena allocates from and
  the PD wire carries.

Cache-layer counts come from the recipe's `layer_types` and `num_draft_layers`,
before capacity is bound, and are published by `CacheSetup`: target cache layers
first, then independent draft cache layers. PP construction gives these values explicit `*_cache_layers` local
names; a drafter that shares target cache contributes no independent cache
layers. Neither count means captured target taps or draft execution depth.

`distributed/pp_stage.py::pp_stage_windows` owns the target execution-window
calculation, shared by pipeline stages, model construction, and PD topology.
The model/cache construction boundary maps those execution windows to explicit
`target_cache_windows` with `pp_stage_cache_windows`: both ends of every
window scale by the model's attention instances per decoder layer
(`ModelProfile.attention_instances_per_layer`; one for K3 and V4.1, which
declare no profile, two for LongCat's paired layer), so a block's cache layers
stay with the stage that executes it. `CacheLayerOwnership` then adds the
final stage's draft cache window. Cache ownership consumes cache-ID windows;
execution partitioning belongs to the distributed layer, and non-PP ownership
covers the complete cache namespace without any per-block arithmetic. Resident
windows, transfer filtering, and producer-field groups all use cache-layer IDs;
PD consumes the resulting `cache_fields_by_stage` and never recomputes windows
from layer counts.
Cache construction resolves ownership into explicit field IDs once:
`cache_fields_by_stage` describes residency, and `producer_fields_by_step`
describes the local readiness barriers. These sets cover resident fields
exactly once. PD never derives placement from target/draft identities,
execution-layer counts or contiguous layer windows. Noncontiguous field sets
are valid. The bootstrap wire carries `cache_fields_by_stage`; all prefill
ranks must register the same complete placement and logical field plan.
Old bootstrap peers without explicit placement must be upgraded together.

`create_attn_components` returns a frozen `AttentionBuild` naming target and
draft backends/pools, cache storage, field placement, readiness, and optional
`logical_plan`. The complete logical plan is retained when PP narrows the
physical arena. The builder passes these values explicitly to PD; none are
attached to the event loop or added to the allocation owner after construction.

`CacheRecipe` (`recipes/base.py`) is a template method: `setup_for_layers()` is
the one place the four stages appear in order. `setup()` delegates to it with
the complete layer window and a standalone capacity validator. A family fills in uniformly named
seams — `layer_types`, `group_ids`, `fields_for_layer`, `prefix_granularity`,
`alignment`, `max_padding_fraction`, `packing`, `check_layout`,
`select_num_lcm_blocks`, `token_capacity`, `parents_needed`, `workspace_bytes`,
`pool_options`, `verify_scratch_in_pool`.
The last answers whether speculative verify stages its scratch in the bound
pool; a CUDA-graph memory probe arena then keeps the serving concurrency.
`groups()` itself is a seam for the two families whose groups are not
per-layer (Inkling appends conv columns; V4.1 declares each group
whole). No family restates the order of the stages, and `_RECIPES`
(`recipes/setup.py`) is the single family → recipe map.

Automatic packing uses powers of two to avoid excessive LCM padding;
exact page-stride constraints and explicit recipe packing take precedence.

**Resident capacity before bind.** A PP owner projects the complete layout to
the planes required by its fields. Whole retained planes, including padding
and shared fields, determine the local bytes per parent. The runtime subtracts
the recipe's fixed workspace, reserves the null parent, and agrees on the
smallest parent limit across the existing world group. Only then does
`select_num_lcm_blocks(layout, max_lcm_blocks=...)` apply model/token limits.
Families do not recalculate memory budgets in that seam. Existing plugins
overriding `num_lcm_blocks` must migrate; the builder rejects that override
instead of silently ignoring it.

The full layout is bound once at the final capacity, so packing, group IDs,
global field IDs and the PD contract remain common. PP subsequently narrows
physical allocation as before, without changing that capacity. A stage with
fewer planes can leave memory unused because another stage limits the shared
capacity. Fixed workspaces are charged once; they do not shrink automatically
with the number of local layers. Offload retains its host-history and fixed
hot-pool constraints and requires a complete field policy.

The agreement callback is supplied by the runtime, keeping recipe tests free
of process-group dependencies. A local budget error is collected before any
rank raises, so peers do not hang in a capacity collective. Probe pools agree
on the largest required working set, not the smallest memory-derived limit;
zero-budget geometry probes keep their existing demand-sized behavior. Final
rebind repeats resident sizing under the graph-adjusted budget.

Ordinary recipes pin one CacheBlock per group in each parent, so a group's
stride does not shrink with its layer count or cache dtype. For example,
equal-sized layers split into groups of one and four need a padding fraction
of `3.0` for the smaller group. The recipe therefore opts out of the padding
ratio limit through `max_padding_fraction`; the common packer still validates
field geometry, strides and block-size limits, and the profiled byte budget
still bounds capacity. Grouping, packing, and allocation are unchanged.

Hybrid recurrent recipes keep their groups balanced instead: every group's page
spans the whole parent, so a parent is fully used only when every group fills
the same planes. Both recipes size a plane segment to one SSM state and give an
attention layer's K and V a segment each, so state pages carry no attention
planes, and deal state layers round-robin. Qwen3.5 keeps its attention layers
in one KV group, the full-history group drafters write through, and deals its
state layers into groups of as many SSM states as that group has segments; a
draft layer of a target attention layer's K/V bytes fills state segments, any
other pads the state groups (`recipes/qwen35.py`). Nemotron-H's Mamba2 runs
are four or five layers long; its recipe splits the attention layers into two
KV groups (`recipes/mamba2.py`).
Qwen3.5's state groups leave segments free, so its conv windows pack into them
and KV pages carry no padding; Nemotron-3 Super's state groups fill every
segment, so its KV pages pad by their share of the conv planes, about 1.5%.

Qwen4-Exp also reuses the attention planes for PLE and QSA groups. A parent
belongs to one cache group at a time, so these groups do not need separate
planes reserved in every attention or recurrent-state parent. Compressed QSA
keeps its exact page stride; PLE and recent QSA state consume the explicit
strides chosen by the common packer.
Groups whose fields do not fit the attention planes, or whose exact strides
require inconsistent packing ratios, retain separate planes.

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

**Recipes do not inspect the hardware.** A row encoding can be a property of
the target — DeepSeek V4.1's packed rows are read natively by FlashMLA above
sm100, while sm90 reads only the wider V4 layout — but the recipe never asks
which machine it is on. The choice is made once where the model's attention
config is generated (`configs/deepseek_v41.py`), recorded on the spec, and
read back by everyone who needs it: the recipe sizes fields and looks up the
packing from it, the backend names its kernel cache formats from it. Row
width forces the packing and the plane, so each format owns its own frozen
`group_packing` / `lcm_block_bytes` tables (`deepseek_v41_geometry.py`); the
geometry module is a table keyed by format name and knows nothing about
architectures. Adding a platform probe below the config layer would give one
parent two possible sizes with no single place that decided which.

### Storage vs. visibility

An attention layer has two contracts that a single `layer_types` string used
to carry at once, and they are kept apart on purpose:

* **Storage** — which cache group the layer's KV rides, hence how long the
  scheduler retains it (`CacheGroupSpec.attn_kind`, `sliding_window_tokens`).
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
sparse compute over a fully retained cache) retains more than it reads.

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
in the replayable group, are never expected from a hit either. Decoder SWA
visibility starts at that retained tail, even when the final chunk already
contains exactly one window and no rows are dropped. Encoder metadata may be
reused only when its visible history also starts there; equal row counts alone
do not make the two windows interchangeable.

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

K3 DSpark context injection uses the draft backend's cache placement to
resolve virtual slots and mask nonowner writes before storing MLA KV. It
therefore follows the same KVP ownership contract as attention writes; the
architecture alone must not reject KVP. Backend capability and shared-group
storage checks still apply.

Capacity has three shapes, all on the base class. The default is the flat
product (`parents × tightest packing × P`). A probe arena is the third and
narrowest: `probe_batch_rows` says how many requests the CUDA-graph probe
fabricates, and the pool takes one parent block per fabricated row, floored at
what a single request needs — it binds before the memory profile has run, so
it cannot size from a budget at all. Families whose per-group demand decides
the pool — K3's state groups riding inside MLA planes, V4.1's
SWA and global KV groups, GLM-5.3-Flash — size from `parents_needed` and get
the inverse for free from `_capacity_from_parents`, one monotonic binary
search shared by all. `parents_needed` itself is not a Python formula: it
hands the group specs, the layout's virtual packing and `scheduler_limits`
to the scheduler's own `CapacityModel` (`recipes/scheduler_bridge.py` →
`csrc/scheduler/capacity_model.h`), whose `ConcurrentGroupPages` reports each
group's demand at the configured concurrency and whose `LcmBlocksNeededFor`
folds it by packing. The per-request working set — decode reservation,
overlap-protected step, the blocks a tail handover holds for one admission, a
State group's two slots, a sliding group's lookback and resident window — is
therefore defined once, in
C++, and the `Scheduler` bounds single requests against the pool with the
same model (`docs/design/scheduler.md` §1.4). No recipe restates any of it.
`scheduler_limits` is the single place a recipe reads the scheduler's
concurrency, role, and reserve widths, so demand and capacity cannot size
against different numbers. Under a probe it reports the probe's fabricated
batch instead: that arena holds a capture, not requests. `probe_batch_rows`
sets both sides: the arena holds at least that many parent blocks (more when
admitting one token per group needs more), and the concurrency is that many
rows, capped at the scheduler's `max_bs`.

The runtime divides its global `max_num_seqs` across attention DP ranks to
produce each scheduler's rank-local `max_batch_size`. These values limit
simultaneous sequence slots; they do **not** reserve enough history cache for
that many maximum-length requests. Aggregate prompt and decode growth must
still fit the recipe's reported token capacity. When that dynamic pool is
overcommitted, admission can fail even though the batch still has a free
sequence slot.

When admission fails for capacity and no prefill can progress, the scheduler
retracts a resident victim and grants the freed pages to the blocked request
within the same plan build; the victim is suspended with an image of its
tables and restored into fresh pages later, losing no work. What stops an
overcommitted workload from repeatedly suspending and restoring the same
request is the escalating admission headroom each retraction adds to the
victim's next admission. The protocol — victim choice, the image's two legs,
readmission order, why the release is safe before the image copies — is
`scheduler.md` §2 and §4. The snapshot pool's size is a configuration input
like the Host cache's (`num_snapshot_pages`, `max_retracted_requests`),
resolved by the runtime at device build (`cache/l2/sizing.py`; the
derivation table is in `docs/configuration/server.md`): it bounds how many
pages of suspended requests may be held at once; when a victim's image does
not fit, the victim is aborted instead of imaged (`scheduler.md` §2), so a
capacity block never waits on the Host. `--retraction-snapshot-ratio 0`
is the pool-less engine, where every capacity block aborts.

## Virtual block placement within a shared physical plan

KV parallelism (KVP, `--kv-parallel-size`) is a storage
layout: a sharded group's pages are dealt page-cyclically to the ranks of the
KVP subgroup by virtual block id, so each rank stores `1/N` of them. Decode
context parallelism (DCP, `--decode-context-parallel-size`) is the decode
compute over those pages -- every rank attends the pages it owns and the
partials are merged by LSE -- and belongs to the attention backends.
Aggregated engines reuse the sharded-cache prefill and DCP decode paths in
separate rounds. This section is about the storage side.

A recipe declares each group as a `(CacheGroupSpec, fields)` tuple.
`CacheGroupSpec.shard_count` defaults to 1 (replicated); a larger value assigns
virtual blocks cyclically across that many owners (the KVP size). The memory plan continues
to own local shapes, strides, packing, and byte counts. `CacheArena` is the sole
publisher of `CacheRuntimeContract`, whose virtual counts and packing derive
from these physical facts and each spec's `shard_count`. No separate placement
or per-group address-space object is needed.

A recipe's group set never depends on the KVP size. Only groups whose every
reader can attend to a shard may be sharded; a group some consumer must read
whole stays replicated and is declared as its own group at every KVP size, so
prefix matching, transfer, and zeroing -- all keyed by group -- see one
topology. DeepSeek V4.1 shards its global KV groups, whose pages carry both
the compressed KV and the index-K rows, and keeps the SWA rows and compressor
tails replicated.

Ordinary GPU MLA and DSA use the same ownership geometry for history storage.
MLA/KDA hybrids shard the MLA history group and keep KDA state replicated.
Before allocating the arena, the hybrid KVP check validates these declared group shard
counts rather than the recipe name or its inheritance. Plugin recipes follow
the same storage contract as built-in recipes. Both pure MLA and MLA/KDA hybrids
require the full-attention backend to declare `supports_mla_dcp`; FlashMLA and
TokenSpeed MLA (CuTe MLA) both declare this capability.
Decode (DCP) gathers query heads, computes attention over owned history, and
merges partials using FP32 natural-log LSE before restoring TP-local heads. MLA
prefill reads its bounded prefix chunks through the backend
(`read_prefix_history`), so the model never sees the sharding. The paged
backend base reads a local cache in place and rebuilds one dealt over owners
by all-gathering each owner's rows as raw cache-dtype bytes, bitwise the
unsharded read; a leaf declaring `supports_mla_dcp` only hands it each
prefill's prefix loops and host page table (`_prepare_prefix_history`), so
planning the gathers never waits on the device. The gathers run on a named
copy of the KVP group, whose communicators are apart from the attention TP
collectives, one window of loops ahead, so each transfer runs under the compute
before its layer instead of queueing with those collectives. GPU DSA sparse
prefill instead combines local sparse-attention partials.
Ascend's generic MLA prefill gathers each layer's complete cache plane into
an owner-major temporary view. That view is not published as persistent cache;
its allocation and communication scale with layer capacity, including unused
rows, rather than the batch's live prefix length. Native token-local prefill
requires an explicit model opt-in and a cache reservation; DSA uses that same
opt-in to prepare full-head projection weights before execution.
Dense MLA uses FlashMLA or CuTe MLA within each backend's device/dtype support;
KVP does not make unsupported kernels portable. Speculative decoding runs
on sharded caches: `OrdinaryRecipe.groups()` shards a draft group like the
target's, and the Kimi K3 recipe shards with its draft layers in the MLA
history group, subject to its backend's `AttnConfig` gate. A verify window's queries
end at different global positions, and the kernel's multi-query causal mask
would subtract a query's global offset from a *local* length, so FlashMLA
and CuTe MLA feed every query's global endpoint through the compact layout
(`CompactDCPLayout.visible_lens` -> `local_visible_lens`, one owned-page
prefix lookup per query, refreshed in place with the table and by the
drafter's seq_lens hooks) and FlashMLA runs one kernel row per query against
its local length. Known cost: a width-`w` FlashMLA verify under KVP issues
`bs * w` single-query rows and reads each request's owned pages `w` times
instead of once. GPU DSA selects causal Top-K before mapping cache slots.
Non-causal DSA block drafts remain unsupported (`AttnConfig` rejects
`draft_block_decode` under GPU DSA KVP).

**Tier transfers under sharding pair blocks of equal residue.** The Host L2
pool and the snapshot pool are registered with the same `shard_count` as the
Device pool, and every transfer the scheduler emits — L2 store, L2 prefix
load, both legs of a retraction image, the restore — acquires its destination
in its source's bucket (`AcquireHostBlocks(group_ids, buckets)`,
`AcquireHostSegments`, `AcquireBlocksInBuckets`, `TakeImage`, `Restore`),
so `(src - 1) % D == (dst - 1) % D` for every row; `TierTransferManager`
asserts it when it builds a batch. A Host block's residue is therefore its
owner exactly as a Device block's is, and one `owned_local_pages`
translation serves both ends of every row: a rank keeps the rows it owns
and translates both ids to local pages, a
replicated group (`D = 1`) owns every row. L3 probes replica-stable keys on
all owners, then reads or writes only the Host blocks this rank owns. A rank
with no local bytes reports neutral success and proven retirement, performs
no native I/O and cannot act as a health recovery probe. The all-owner
intersection still requires the actual owner's success before publication.
L3 reservations distribute successive logical pages across cyclic owners;
Device admission later preserves each restored Host block's residue.
A residue-constrained acquisition can fail while
other buckets have room; a restore then waits like any readmission, and the
balanced allocator keeps ordinary admissions within one block per bucket, so
the imbalance a restore sees is bounded by what its own victim table
contributed.

**No rank reads or writes another rank's Device KV.** A page an eviction
write-back copies may be granted again in the same plan -- zeroed, loaded
into, written by the forward -- and each rank orders its own copies ahead of
all of that with one store fence on its default stream
(`HostCacheExecutor.submit_write_backs`); a rank holding no row of an op
still records the fence and acknowledges the op. One fence per rank is enough
because every path keeps three invariants:

1. Every KV write is the owner's: the attention prologue's owner-masked
   store, page copies and zeroing within one owner's pages, Host rows through
   the owner translation. What KVP, DCP and QCP exchange across ranks is
   copies -- gathered history rows, query heads, attention partials -- never
   a peer's cache.
2. Off the execution stream, a read of a page the plan grants again forks
   from it after the forward's prologue, which waits on the default stream
   and with it on the plan's store fence and zeroing; the history gather's
   all-gather, for one, reads rows the execution stream copied out. Only the
   eviction copies, which the fence orders, read such a page earlier: on the
   write stream, behind the forward that wrote it.
3. A kernel that would read a peer's KV in place (one-sided or
   symmetric-memory access) must also order behind that owner's store fence,
   a cross-rank edge nothing provides today.

PD transfer supports independently sharded prefill and decode roles. Manifests carry scheduler (virtual) IDs on both sides and are bounded
by each side's virtual count, `1 + (page_count - 1) * shard_count`, never by
the physical page count. The route planner (`ts-pd`'s `layout::route`) reads
`shard_count` from the wire `group_specs`: for a sharded group it fans a
decode rank's replica out to the whole KVP subgroup (consecutive attention-TP
ranks), tagging every member with an owner filter `(owner_rank, owner_count)`;
the sender keeps the manifest blocks with `(v - 1) % owner_count ==
owner_rank`, translates them to local pages with the placement zeroing uses
(`owned_local_pages`), and matches them to destination blocks at the same manifest positions.
Only pairs owned by the destination rank are copied, after translating both
virtual IDs to local pages. Every rank of the subgroup therefore
serves every decode rank and none is a control-only dummy; the decode receiver
already counts completions from a rank set. Replicated groups keep the
single-source route. A field that is both head-partitioned and page-sharded is rejected by the planner.

The plan records one decision per (source rank, sharded group), never by
omission: the owner filter when the rank's fragments name the group, an
explicit `None` when they do not -- a rank routed only for another group's
head partition (Kimi K3's KDA state against its MLA subgroup), or a pipeline
stage whose fields miss the group -- so that rank sends nothing for it. The
prefill checks those decisions against its cache groups once, when the decode
registers (`validate_rank_owner_filters`); the sender applies them as given.
A sharded source leaves the equal-TP empty-fragment route (one predicate,
`_uses_whole_copy_route`, decides it for routing and served-rank sets alike),
but the sender folds every fragment with one contiguous span per page -- a
whole field, or a head slice whose rows collapsed into one -- into its group's
pages x fields grid, so KVP, pipeline stages, and replicated fields across
unequal TP all go through the page-gathered WRITE; only a fragment with
several strided rows per page is emitted per page.

The runtime derives compact DCP page tables and local visible lengths from
the scheduler's virtual block tables, without introducing new scheduler-owned
request state. To support draft length changes, the read table retains
allocated reserve pages while per-query visibility bounds limit actual reads.
When drafting advances or reanchors to an accepted prefix, the runtime
translates the global visibility bounds according to page ownership; a change
in the global token count is not the same change in the local token count.

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
every KVP size. Virtual block 0 is the null block; no path writes to it, at
any KVP size.

Query context parallelism (`--query-context-parallel-size`,
`docs/design/unified_path.md`) is orthogonal to placement: KVP decides who
stores a page, QCP decides who computes a row. Allocation, prefix matching,
zeroing, publication, and the P->D route never see the query shard. The
sharded extend arm reads the placement through the same cyclic rule — the
history gather of a request group splits by page owner
(`dcp/placement.py: cyclic_slot_owner`, `owned_history_rows` on the host from
the kernel page table's mirror) and the KV write gathers the rotated rows to
the whole span before the owner-masked store — so a replicated group
(`shard_count` 1) is a local gather and one owner, the same path with the
degree as a parameter. A history gather workspace of one whole history is
reserved from the cache budget by the recipe (`workspace_bytes`) like a
verify workspace; its index-K rows are packed in the plane's own format
(`DSAConfig.index_k_format`: FP8 bytes with scales, or bf16 keys), so the
recipe's bytes and the leaf's allocation follow one row width and the
gather hands the indexer the plane's form, sharded pages or not.

Consumers bind a pool's compute view and read its arena's runtime contract.
Views sharing an arena share that contract, rather than accepting separately
injected copies of its geometry. Batch metadata retains the same contract for
address translation.

For physical packing K, N usable parents, and D shards, a sharded
group has `1 + N*K` local pages and `1 + N*D*K` virtual blocks. A replicated
group uses one bucket. Existing `group_page_counts` and `group_packing` name
physical quantities; the scheduler bridge explicitly consumes
`virtual_block_counts` and `virtual_packing`. Virtual capacity must never be
used to shape an arena field. Virtual null ID 0 has no owner and is filtered
during translation.

Before zeroing or copying scheduler blocks, the runtime checks IDs against
the virtual bound and translates each group's batch to owned local IDs on the
host (`local_pages` and `local_page_pairs_by_group`, plain integer arithmetic
over the same cyclic placement the device translation kernels implement). A
copy pair's source and destination must share an owner -- the scheduler
allocates them in one bucket -- so a rank either copies the whole pair or
nothing; a pair whose two ids resolve to different owners is a contract
violation, not a partial copy. Translation precedes dispatch to pool views;
pools and the arena receive physical IDs and hold no context rank. The
arena's `zero_blocks()` and `copy_blocks()` validate every ID against its
group's local page count before touching any bytes, and `copy_blocks()`
refuses a destination that is another pair's source, since the pairs of one
plan are launched as one unordered kernel. Physical page 0 is within that
range and is handled like any other page when explicitly requested. Host work
on these paths is O(pages): the arena ships the page ids through one pinned
copy and the kernel expands page x field byte ranges from the group's fixed
field table, because a long prompt's admission hands out thousands of pages
and per-range Python here stalls the forward thread while the device sits
idle.

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

Beside that balanced path, `BlockPool::AcquireBlocksInBuckets(group, buckets)`
acquires blocks in **given** buckets — one per requested block — for the
transfers above, for tail copies (a hitter's copy of a claimed tail block, an
owner's copy-in; each in its source's bucket) and for
`GroupAllocator::AcquireShape`, which rebuilds a retracted request's table
from its recorded shape (claimed blocks in place, fresh ones in the imaged
buckets). Its exact check is per bucket with one shared empty-parent budget:
an opened parent gives every bucket `packing / shard_count` slots at once, so
the parents needed are
`max_b ceil((need_b - holes_b)+ / (packing / shard_count))`
(`ParentsNeededForBuckets`, the formula the pool, the coordinator's image
probe and the admission planner's shadow occupancy share). The planner counts
every block an admission places in a given bucket — a Host hit's destination,
a tail copy — as that bucket's demand, and only growth and a hole's fresh
block as dense demand; a group's bucket-exact acquisitions precede its dense
ones. `D = 1` passes bucket 0 everywhere and is the same path. No mutation on
a failed check, and the choice is deterministic on every mirrored rank.

`BlockPool` maintains the free-slot count of each group per bucket and an
ordered parent index per bucket. Each parent records only its lowest free slot per bucket;
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

## Sparse indexers: model weights, backend dispatch, and verification

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
  nor registers or looks up QSA state. PLE owns its state-table metadata
  independently of Mamba and shares only the two-slot block resolve. See the
  [execution lifecycle](unified_path.md#backend-package-layout).

LCM owns persistent allocation, prefix matching, transfer, and retention,
including QSA's full-KV, compressed, and recent cache groups. Verify-state layer
ownership and cache addresses come from the bound plan's layer window,
including under PP and target/draft sharing. Cache recipes reserve verify
workspace before sizing the arena.
Side-stream allocations retain their own workspace owner through execution,
separate from scratch that main-stream operations can overwrite.
Ascend native Prefill keeps receive, send, packed history, and commit slots in
disjoint views of a backend-owned `WorkspacePool`, allocated on first Prefill
use and reused thereafter. Cache binding carries only the bounded configuration;
cache capacity planning does not reserve this runtime allocation. Without
native pool options, the backend retains its ordinary collective path and
creates no native transport.
Native Prefill requires an explicit `chunk_prefill_kv_size`: its page limit is
the scheduler's rounded batch budget, and exceeding it raises before transport.
The null page and alignment are physical overhead, not additional batch capacity.
A Qwen4 PD prefill role retains the target/draft cache layout for transfer,
but reserves no speculative verify workspace: only the decode role verifies.

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
`prefix_matcher.h` (`FullAttnMatcher`/`SwaMatcher`/`StateMatcher` policy
hierarchy), `tail_table.h` (the published tails under each last full page)
and `prefix_hasher.h` (moved from `scheduler/`).
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

* the identifiers `page_size`, `rows_per_page`, and `entry_stride_tokens`
  are grep-zero across `tokenspeed-scheduler` (csrc, tests, python bindings)
  — the slot span is spelled `block_granularity` everywhere, and
  `CacheGroupConfig` carries it as its only span field;
* a history group's `block_granularity` is required and explicit: every
  paged group states its span, with no zero-means-default fallback, and the
  coordinator asserts a positive divisor of P at construction; a `state`
  group crosses the bridge as `0`, the declaration that its blocks span no
  tokens;
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
and never touches packing. No refactor is needed here.

### Principle 4 — page_table vs. block_table: fixed in Python's own naming

* C++ keeps its single generic `BlockTable` container — per this doc's
  vocabulary that is the correct name for the scheduler-side container; the
  `page_table` concept exists only where paged-KV kernel tables exist — that
  is, in Python.
* The Python residues are cleaned: `FlashMLADecodeMetadata.page_table`, the
  TRT-LLM MLA chunked-prefill metadata's `page_table`, Inkling's
  `col_block_table` (conv state), and the base-class group routing
  docstrings. Third-party kernel keyword names
  (`flash_mla`'s `block_table=`, TRT-LLM's `block_tables=`) are an external
  boundary and stay as the kernels spell them.
* The state backend's replay hook names no `page_table` parameter — state
  attention has no page table, so the shared call's keyword is absorbed
  unused through `**kwargs`. `input_buffer.py` carries no table at all
  anymore: KV write locations are backend-owned (`write_locations`), so the
  runner's input prep writes positions and seq_lens only.

### Principle 5 — Python perceives the logical quantities minimally: fixed; one conversion point, one slot invariant

Compliant: the recipes/planner layer *owns* the vocabulary rather than
leaking it; the router's `GroupTableStacks` fill (`backends/paged/group_tables.py`)
is the single expansion primitive; state attention and KV share one
plan/arena/`CacheBlock` view, mirrored by the host tier. Specifically:

* No magic `prefix_granularity == 128` branches: the constraint is named
  `MXFP8_KV_SCALE_TILE_TOKENS`, defined in `recipes/plan.py` beside the scale
  field geometry it fixes and re-exported from `recipes/spec.py` for callers
  that read it as a token span.
* glm5's page→slot arithmetic lives in the mapping layer
  (`attention/page_table.py::build_prefill_kv_workspace_slots`), not in the
  model.
* The state backends read a state group's two table slots and nothing about
  a span — asking such a group for `page_size` or `block_granularity` is a
  `TypeError`, since it has no rows.
* `CacheGroupSpec.__post_init__` shape-checks spec geometry at construction:
  `history` declares positive row geometry, `state` declares none and
  `full_history`, and each family is held to its shape and retention. The
  C++ `CacheGroupConfig::Validate` refuses a sliding `State` group at the
  bridge.
* Group consumption is claimed positively, from one declaration: each
  consumer takes exactly the delivered `block_tables` entries for the
  groups it serves. The router builds a leaf for each claimed attention
  group and fails a live batch missing any of them. QSA's indexer claims
  its compressed/recent history groups separately; it does not instantiate
  attention leaves for them. State consumers (Mamba/KDA, PLE, Inkling conv)
  index the dict by their own group ids, as does the V4.1 backend for the
  several groups one V4.1 layer reads at once (its pool view reports
  no `PagedAttention` binding and no router leaf). Mamba's group set comes from
  its recurrent fields, so a PLE state group cannot arm its verify
  state. `cache_consumer_attn_kinds` remains the boot-time coverage
  declaration (`validate_scheduler_config`). Extra delivered groups ride
  through untouched; a table for a group the bound pool never published
  fails loudly.
* The logical→physical conversion has ONE home per pool view:
  `CacheGroupRouter` (`backends/paged/router.py`) learns each group's
  `block_granularity` and each leaf's `kernel_page_size` into one
  `CacheGroupGeometry`, expands the bridge's raw block tables into
  kernel-page stacks, and derives every KV write location — extend spans,
  the decode/verify window, and the drafters' published step windows — from
  those same tables (`backends/paged/write_locations.py`, pure functions). Paged
  leaves see kernel vocabulary only. The bridge's per-group table views
  (`CacheBatchMetadata`) are the router's input — block vocabulary in,
  kernel pages out, one expand launch per group. Models and the runner never
  compute locations — `write_locations(layer, mode)` addresses authoritative
  history, and `forward_write_locations` supplies the attention prologue's
  writes (`unified_path.md`, "Write locations have one owner"). Sparse
  offloading prepares separate compute writes with `prepare_sparse_kv_access`;
  the router uses those slots for both prologue and leaf dispatch (see
  "Sparse KV offloading and physical storage").
  QSA's indexer reuses `GroupTableStacks` with `kernel_page_size` equal to
  each group's `block_granularity`. This ratio-one fill copies stable raw
  table views and clears holes/padding; it does not add another subdivision
  convention or derive its addresses through a dummy attention leaf.
* The slot *arithmetic* itself lives in the mapping layer as one invariant
  (`table[req, pos // P] * P + pos % P`, which is page-size invariant): the
  router's stacked window/span math (`backends/paged/write_locations.py`,
  failing to slot 0 — the reserved dummy page).

### Principle 6 — provenance discipline: fixed

* The contract's `prefix_granularity` comes from the memory plan
  (`kv_cache/arena.py` builds the runtime contract from
  `plan.prefix_granularity`, never from pool geometry). ✓
* Field dtypes come from the memory plan. `CacheFieldSpec`/`CacheFieldLayout`
  carry `dtype` (a name; `plan.py` stays torch-free because the plan travels
  the PD wire) and `element_size` derives from it, so byte geometry and dtype
  cannot disagree. Recipes name each field's dtype where they already know it,
  through `cache_dtype_name` or `scatter_stored_dtype_name`; the latter
  holds the one substitution rule — fp8 collapses to `uint8` for fields
  written by elementwise scatter, because `index_put` has no fp8 kernel,
  while fields written through dtype-aware kernels (MXFP8) keep their fp8
  view. The contract carries no parallel `field_dtypes` tuple. ✓
* The arena owns the allocation and materializes every planned field view in
  its constructor, so `field(field_id)` is a lookup with no dtype argument and
  no lazy-bind state. Ordinary history fields fold their page axis only when
  their leading extent matches the prefix grain; compressor state keeps its
  page axis even though its leading extent matches the group's row count.
  Offloaded history fields fold by their own physical row grain, preserving
  flat row addressing when that grain differs from the prefix grain.
  These are views of the allocation, never reshaped copies.
  `CachePool.store_dtype` means one thing: how a pool
  reinterprets *input* tensors before a write. A pool allocates nothing —
  `_bind_layer_planes` walks `plan.fields` once and arranges this view's layer
  window into the per-layer buffers its kernels read, with each subclass
  declaring only its `layer_plane_bindings`. Which planes a layer has is a
  fact of the plan (a state layer plans no `k`/`v`), and page contiguity is a
  plan invariant (`exact_page_stride`). ✓
* Every backend's `kernel_page_size` is registry- or config-sourced
  (`kernel_page_sizes.py`); the registry LCM validator checks explicitly
  configured values, and a backend that resolves its own registry default
  owns the divisibility check for it. ✓
* Two arena scalars, two roles, both derived from the plan:
  `CacheArena.prefix_granularity` (identity grain; contract publication and
  plan checks) and `CacheArena.kv_page_size` (KV arena geometry, read by
  row/slot/tile math in the paged pools and their consumers). ✓
* Cache geometry has one owner and no mirrors. `CacheArena` holds the
  allocation, the field views, the plan, the contract, and the geometry
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
* Capacity has three shapes and no more, and one place to read the scheduler's
  concurrency (see *The cache pipeline* above). ✓
* Kernel geometry does not live under the recipes package. DeepSeek V4.1's byte
  formulas, cache layout, and group-id vocabulary sit in
  `attention/deepseek_v41_geometry.py`, which the backend, model, and pool read
  directly; the recipe depends on it, not the reverse. ✓
* One kernel layout has one definition. The interleaved mxfp8 KV-scale planes
  come from `plan.mxfp8_kv_scale_fields`, which also owns the page-span and
  head-dim constraints that layout imposes, so no recipe restates the shape or
  re-checks the constraint. ✓

Verified end to end for this round: DeepSeek V3.2, R1, and V4-Flash, each
× {CUDA graph, eager} × {spec, no spec}, against pre-refactor baselines
(accuracy equal or better; speculative accept length within noise).

### Sparse KV offloading and physical storage

**Configuration and placement.** `ServerArgs.resolve_kv_offload()` normalizes
`--kv-offload-config`: target attention layer IDs, per-request hot capacity,
Host GiB budget and prefetch overlap. The plugin recipe interprets model
options, including the explicit `kv_offload_window_ring` HF override, and
declares a `KVOffloadPolicy` with field IDs, verify width, reserved/ring rows
and selection dependencies. The common `CacheRecipe.setup()` binds request
slots and the extend chunk bound. Recipes without offloading support reject
an enabled configuration.

`CacheStoragePlan` binds the logical `CacheMemoryPlan` to physical regions,
field offsets and offload workspaces. Selected fields have authoritative
history in pinned Host regions; other fields retain their device placement.
`CacheArena` owns these regions and its `SparseKVOffload` engine, which owns
each selected field's hot buffer, GPU tags, LRU metadata, and transfer streams.
Allocation bytes are checked against the storage plan.

`arena.field(field_id)` returns authoritative history;
`arena.compute_field(field_id)` returns the hot buffer for an offloaded field
and the history view for other fields. PD registers the regions and publishes
each field's actual landing address. Transfer contracts validate plan/group
IDs and transfer policies; bootstrap schema identity excludes rank-local
addresses. Block clearing splits logical byte ranges through the physical
bindings. CuTeDSL clears mapped pinned Host ranges on the execution stream,
respecting outstanding GPU work. This Host history belongs to the arena;
the coordinator's L2 prefix cache has a separate ownership contract.

**Fixed hot capacity and admission.** For each offloaded field, let `S` be
the configured request-slot capacity (including null and graph padding), `H`
the ordinary hot rows per request and `R` the reserved/ring rows per request.
`compute_offload_capacity()` fixes the device row count to `S * (H + R)`,
rounded to the storage-page alignment.
Reserved capacity is measured in token rows; page rounding applies to the
complete allocation.

The device cache budget reserves this payload, persistent metadata, and a
conservative temporary-workspace peak before sizing device history. Remaining
device bytes and the Host budget bound the logical LCM capacity, including the
null parent. Probe and serving arenas cover every configured hot partition;
insufficient memory is a startup error. Each C++ request slot owns its fixed
hot partition throughout PD landing and execution, so request-slot allocation
and history-page admission provide the admission checks. Null and padding
requests are masked before installing tags or writing KV.

**Router access.** `KVOffloadAdapter` in `paged/offload_adapter.py` binds
router layers to offloaded fields and groups. It selects the group's kernel
page table and uses `seed_locations` to produce physical history/hot row
pairs. It retains the step's compute read/write mappings; the arena-owned
engine manages payloads and replacement state.

`write_locations(layer, mode)` returns authoritative history destinations,
including those used by Index-K. The model passes its canonical Top-K and
positions to `prepare_sparse_kv_access`, which returns projection destinations:
hot rows for decode, or ordinary history rows
for a field on device. The router substitutes the prepared hot read indices
and write locations at leaf dispatch. An offloaded compute access requires a
prepare for the same step and forward family. Index-K and shared-selection
consumers keep canonical history IDs; models use the returned destinations
without accessing the engine's tags or constructing physical field IDs.

**Lookup and replacement.** Each field has a request-private LRU permutation
over ordinary hot slots. A resolve processes the multi-query selection union
once: protect selected resident rows and ordinary current rows, allocate
unique misses from the oldest unprotected slots, then publish remaining old
slots, newly loaded slots, and protected slots in that order. Hits retain their
relative order. Reserved/ring slots have their own write placement and are
excluded from LRU replacement.

CuTeDSL hash lookup coalesces duplicate history IDs for physical loads. Unique
misses receive destinations in first-occurrence input order, while the mapped
attention inputs preserve order, duplicates, and masks. Hash tables use shared
memory where the kernel geometry permits it; larger tables and destination
scratch use the planned persistent GPU workspace. Slot reuse resets
LRU; accepted-prefix writeback leaves the selection-access replacement state
intact.

**Prefetch and writeback.** The recipe declares selection producer/consumer
pairs. `prefetch_sparse_kv` seeds a selected consumer and installs current-token
write slots on the execution stream, then schedules lookup and Host-to-hot
copies on the prefetch stream. The field's `ready` event covers payload,
mapped indices, tags, and LRU. The consumer still calls
`prepare_sparse_kv_access`, which joins that event before consuming the hot
buffer. Fields without a scheduled prefetch resolve on the execution stream.
LongCat2 shares selection from each even attention layer with the following
odd layer; prefetch applies when that consumer is offloaded and overlap is
enabled. The available overlap depends on intervening compute and stream
dependencies.

The executor calls `prepare_cache_batch` before the target forward and
`writeback_accepted_kv` after verification. Decode writes only accepted target
input rows to authoritative Host history. The write stream waits for forward
work, and the execution stream joins `write_done` before completion or request
slot reuse. Batch lifecycle hooks broadcast to child backends.
`invalidate_cache_residency` fences transfer streams and resets hot tags,
seed flags and LRU for reused slots, preserving authoritative history and
scheduler-owned blocks. PD transfer completion gates the first seed.

**Capacity abort.** Offloading disables the retraction snapshot pool. It
uses the scheduler's existing no-pool path: a quiescent capacity victim is
aborted, its cache blocks and request slot return to their allocators, and
the blocked grant can proceed in the same plan. The runtime reports a
`CapacityAbort` finish to the client. It never suspends or resumes that request.
Explicit snapshot capacity and forced-retraction options are rejected.

Authoritative Host history follows the logical blocks' lifetime. The hot
payload is a fixed allocation; releasing a request slot makes that slot's
partition available rather than freeing the serving arena. Before a new PD
owner uses it, `invalidate_cache_residency` fences transfer streams and
clears tags, seed flags, and LRU. Accepted writeback is joined to execution
before completion, so page and slot reuse cannot race its writes. Offloaded
arenas support neither L2/snapshot copies nor local recovery prefill.

The LongCat2 plugin supports PD decode with PP=1, unsplit KV rows and
KVStore disabled, using the target router directly. Local offloaded prefill
and mixed-arena sleep/wake are unsupported. Eager and CUDA-graph decode
use the same batch, access, and accepted-writeback hooks; execution details are
in `unified_path.md`, "Sparse KV offloading".

## Diagnosing prefix cache misses

For a diagnostic replay, setting `TOKENSPEED_PREFIX_CACHE_DIAGNOSTICS` emits
one `PrefixCacheDiagnostic` line per successful first admission. The snapshot
is taken before admission can evict blocks and includes the request ID, actual
resume boundary, KV-only boundary, group names, prompt chain hashes, and candidate boundary rows
`[tokens, device_missing, host_missing, union_missing]`. Each mask uses bit `i`
for cache group `i`; replayable groups are excluded. Full KV requires the whole
prefix, sliding groups their lookback, and State groups their checkpoint at
that exact boundary. Union availability permits arbitrary Device/Host mixing,
so a zero union mask is an optimistic physical availability bound, not a claim
that the tier-suffix matching rule can resume there. Missing entries do not
by themselves distinguish never-published data from evicted data. The hashes
let an offline analysis confirm shared full pages against completed earlier
turns, independently of trace estimates. The extra walk and logging are for
attribution, not performance measurement; leaving
the environment variable unset omits both and changes no cache policy.
