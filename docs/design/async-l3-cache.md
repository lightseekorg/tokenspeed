# Asynchronous global L3 cache

L3 extends the existing Device → Host cache hierarchy with shared Mooncake
objects. The scheduler prepares Host data before ordinary Device admission.
All requests still use the same matcher, admission planner, H2D operations and
forward path. Selecting `--kvstore-storage-backend mooncake` selects this
asynchronous implementation; there is no second synchronous scheduling mode.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| C++ scheduler and `StorageTransferManager` | Request preparation, prefix boundaries, pressure admission, soft deadlines, operation IDs, credits, Host references, publication, cancellation and quarantine |
| `CacheCoordinator` | Group matching, allocation and pinning; ordinary probes see published Device/Host data only |
| The engine host's L3 executor (`ts-l3`) | Immutable Host jobs, the store client and its registration of the Host buffer, bounded query/read/write queues, completion/pressure facts, and backend health monitoring |
| The control plane's `l3` module (`ts-control`) | Submission of the plans' storage operations; replica agreement on time, permissions and completion facts; feedback to the existing event-loop head |
| `cache/l3/factory.py` (startup) | The deployment identity and the executor's `L3Settings` |
| `HostCacheStorage` | Shared pinned Host allocation and CPU-only block geometry; holds no device buffers or transfer layout |
| `DeviceHandle` / `HostCacheExecutor` | Ordered D2H/H2D and device completion fences |

The C++ cache subsystem selects reusable prefixes, allocates blocks and owns
their references. The executor's workers receive immutable keys and Host byte
ranges; they call neither the scheduler nor GPU APIs or collectives, and no
L3 path takes the Python GIL. Native I/O references remain held until every
cache-owning rank proves retirement. The executor addresses the shared Host
allocation by pointer, so the data plane keeps that allocation for the
process once L3 runs: native I/O that cannot prove its retirement may
outlive the executor.

Device construction returns startup-only `HostCacheResources`: shared storage,
owner translation, CPU cache geometry and resolved attention choices. This
record contains no Device buffers or executor. The separate L3 factory derives
the storage schema and immutable deployment identity, validates sharded payloads,
and resolves the `L3Settings` the engine host starts the executor from: limits
and workers, the identity, the Host allocation's address and geometry, and the
store client's settings. Static checkpoint/layout/producer facts are collected
once; weight-version changes combine that identity with explicit target/draft
versions. Checkpoint agreement uses the caller's explicit CPU group.
Device construction retains responsibility for L2 copies and their Device
buffers, streams and slot-state exporters.

The control plane submits a plan's storage operations to the executor, in
plan order, when it dispatches the plan. The data plane adapts `plan.cache`
once (`CacheOperations`), sets the storage subset aside and passes the Device
subset to `DeviceHandle.execute`; the control plane counts the same Device
operations from the plan. The two executors depend on scheduler tickets rather
than their submission order. Completions enter the existing head feedback
point, including during frozen and DP-idle rounds.

Configuration follows the same ownership boundary. `StorageConfig` is the
read-only scheduling contract for credits, deadlines and pressure watermarks.
`StorageRuntimeConfig` contains that contract plus the executor's worker counts
and backend-health settings. `MooncakeStoreSettings` contains connection
settings and this rank's segment share. The JSON `async` object tunes this one
asynchronous path; it does
not select another execution mode. See
[server configuration](../configuration/server.md#host-l2-and-mooncake-store-l3)
for settings and defaults.

## Request state

A request waiting for QUERY/GET is in the main FSM's `Prefetching` state,
not in `Submitted` with a second readiness flag. It owns no Device pages or
request slot and never enters forward candidates, prefill token reservation
or capacity-retraction demand. A one-shot preparation bit prevents repeated
storage attempts after a miss, pressure bypass or soft timeout; ordinary
admission remains available.

`Prefetching` carries a typed continuation, either `Bootstrapping` or
`Submitted`. The PD bootstrap ACK updates that continuation without ending
the L3 wait. L3 completion or a soft deadline resumes the continuation,
so both dependencies still overlap. A completed read's Ready hold moves with
the pre-admission state, expires independently and remains charged to its
storage ticket until ordinary admission acquires the Host sources. Native
I/O and its memory references may outlive cancellation or a soft deadline;
that lifetime stays in `StorageTransferManager`, independently of the FSM.

`AdmissionResources` carries the input context, `storage_prepared` bit and
optional Ready hold across these pre-admission transitions. The hold names
the storage ticket; the actual Host references remain in
`StorageTransferManager`. `storage_prepared` means the request has considered
L3 once, not that a read succeeded. A Ready hold likewise does not promise
a complete prefix hit.

`plan.cache` contains both Device copies and four Host-only command types:
`StorageQueryOp`, `StorageGetOp`, `StoragePutOp` and `StorageCancelOp`.
Query carries keys, Get carries Host destinations, Put carries Host sources,
and Cancel names the existing operation without creating another ticket.
Their read-only `kind` is fixed by type. They share one L3 executor and
completion protocol; storage completion never substitutes for an H2D fence.

## Read transaction

1. The scheduler probes local Device and Host caches. A local hit covering
   the maximum reusable boundary, or an extension below the configured
   minimum, bypasses L3. Otherwise it considers a bounded extension of that
   prefix, respecting replay/logprob limits and group geometry. It expands
   full prefix boundaries and matching tails from the local `TailTable`, plus
   the exact reusable request bound. Query credits first cover the longest
   boundary whose complete missing dependencies fit: Full history, the SWA
   window and each State checkpoint at that boundary. Shorter fallback
   boundaries contribute their distinct missing keys only when the whole
   additional set fits. Historical State checkpoints cannot crowd out the
   prompt-end checkpoint needed for the longest restore.
2. The storage subset of `plan.cache` carries a query with a monotonically
   increasing operation ID and immutable keys. Querying reserves no Device
   pages or batch slots.
3. After all owners report existence, C++ runs the existing group matcher
   with these temporary hints. It pins warm dependencies only while allocating
   Host destinations, releasing those temporary references before returning
   the reservation. Under shortage it shortens the boundary and re-runs the
   matcher, including sliding-window and snapshot-state requirements.
4. A GET job receives fixed backend keys, Host offsets and lengths. No worker
   reads a mutable namespace or a later scheduler plan.
5. `StorageDone` carries per-key success, confirmed object-failure facts,
   and an operation-wide retirement fact. After the operation has retired on
   every owner, only rows with full-length success on every owner publish
   Host entries. Successful rows may form a shorter legal prefix; ordinary
   matching determines what is actually reusable.
6. A short Ready hold protects the newly read blocks until ordinary admission
   transfers ownership to the existing H2D ticket. Existing Device and Host
   dependencies remain evictable while GET is queued/running and while Ready
   waits. Admission matches the real cache again; lost dependencies can shorten
   reuse or require recomputation. Ready expiry releases the hold. H2D and the
   forward use the same execution path as every Host hit.

Tail candidates carry their exact token count and hash through reservation.
A retired, attached GET records a local tail hint only after publishing Host
data under it. Hints alone never establish a hit: the ordinary group matcher
still requires Full history, the SWA lookback and every State checkpoint at
the same boundary. A partial history block follows the usual tail copy-on-write
path. Under Host shortage the matcher retries the next shorter candidate,
including an aligned boundary immediately below a tail.

Tail discovery is local and bounded. A cold instance can probe the reusable
request bound, but cannot discover an arbitrary older nonaligned checkpoint
in a longer prompt without a local hint. There is no shared tail directory
or scan of every token position. These misses use ordinary computation.

Query planning uses the same matcher masks and key traversal as reservation,
with temporary existence assumptions and no cache-block allocation. It may
skip history below the aligned local Device/Host coverage, but considers only
boundaries beyond the complete local hit. Reservation always matches again
from the Device floor and pins the real Host dependencies before allocation;
eviction between QUERY and GET can shorten the restore, never establish a hit.

`max_host_blocks` counts logical references across all groups and outstanding
storage operations, including recurrent-state checkpoints. It is neither a
byte budget nor a quota per worker. Page sharding changes the transferred bytes,
not this logical count. A read with no legal prefix within the remaining budget
uses ordinary local admission; deployments must budget for the prefix lengths
and concurrency they intend to restore together.
The reservation still checks new destinations plus temporary warm references
against available credits. Once it returns, GET tickets and Ready holds charge
only their destination references; warm dependencies hold no I/O credits.

Retraction restores exact images through `HostCacheExecutor`; it never starts
a fresh L3 preparation. State groups use the current fixed two-slot checkpoint
contract. Runtime compatibility epoch 2 isolates this implementation from
earlier producer and storage contracts. A different runtime epoch starts a
separate namespace even with the same weights; objects are not converted.
The literal `tsl3-` prefix identifies the cache without a second version number;
the namespace hash includes `runtime_compat` as its single compatibility epoch.
Scheduling, queue and query-budget changes do not advance that epoch.

The request's soft deadline starts when the query is issued and includes
queue wait for both query and GET. A soft timeout ends its wait and permits
local admission; it does not retract the request,
raise admission headroom, mark the backend unhealthy or release native I/O
buffers. Detached completions cannot publish data. Another request's local
publication may also satisfy the wait before the remote operation completes.
Deadlines are processed at scheduler ticks; they are not hard bounds on
end-to-end request latency or backend call duration.

Client cancellation enters at the existing loop head. A fused request still
in Submitted or Prefetching can abort immediately when C++ confirms it has no
forward resources. PD requests detach L3 but retain their existing transport
completion fence and output lifecycle. This removes the extra L3 delay; it
does not make PD cancellation immediate or bound it to one forward chunk.

## Write transaction

A D2H completion publishes Host L2 and releases its Device source immediately
after replica agreement. It never waits for Mooncake. C++ queues bounded,
nonowning backup candidates. At PUT dispatch it rechecks Host residency and
acquires source references; evicted candidates are skipped. Queued candidates
therefore cannot pin Host memory indefinitely.

PUT is create-only. A retired failure leaves the local Host entry usable.
An in-flight PUT retains its source references independently of the request,
and source ownership ends only after all owners report retirement. Local Host
publication does not imply that a shared object exists. Different compatible
instances use the same namespace and per-TP-shard key format; DP replica IDs
are deliberately absent from object keys. The Mooncake master owns metadata,
not the cached bytes. Independently mounted data segments can outlive inference
workers: setting `global_segment_size=0` explicitly makes an inference worker
a client without mounting storage of its own.

## Controller queues, monitor and bounds

Host allocation uses the ordinary per-node rank share and actual rounded
pinned sizes for L2, retraction snapshots and slot-state arenas. Ratio-based
L2 sizing is capped to a common count across ranks; an explicit size that
does not fit is rejected. L2 and L3 share this allocation; attaching L3 does
not allocate another arena. This is not an atomic reservation across containers
or independent P/D worlds. Deployment sizing must account for their combined
allocations, Mooncake segments/staging and independent storage owners.
Runtime operation/Host credits bound L3 ownership, not process RSS.

Query, GET and PUT have separate bounded queues and workers. The total
outstanding-operation limit includes queued jobs, native work and undelivered
results. Completed scheduler tickets remain charged while they hold Ready
results or quarantined references. C++ also limits key count, Host references, lookahead tokens,
negative metadata and backup candidates. Each held CacheBlock is charged
conservatively as one potentially pinned LCM parent. Warm references participate
only in the reservation check; quarantine retains the actual I/O references.
This can underuse packed Host capacity but cannot
oversubscribe it.

Backups may occupy at most half of the operation credits and half of the Host
credits, rounded down. Queries and reads may use the entire shared budget.
These are admission limits in the C++ cache owner, not separate allocators or
worker-side decisions. They keep slow PUTs from exhausting the resources a
new prefetch needs; they do not preempt native I/O or guarantee a large read
fits. A one-credit budget admits reads but no backups. Backup candidates stay
bounded and nonowning until admission, so write pressure cannot pin queued
data indefinitely.

Temporary credit exhaustion leaves candidates queued until credits return;
it does not discard a still-resident prefix. Dispatch rechecks Host residency.
A permanently zero write budget rejects candidates immediately. The bounded
queue drops its oldest candidate on overflow; pressure closure and frozen-round
cleanup still discard queued work. Scheduler counters distinguish pressure,
overflow, unavailable Host data, disabled writes and cleanup, exported through
`tokenspeed:l3_dropped_backup_keys` with a fixed `reason` label. These count
candidate occurrences, not distinct objects. Coalescing a candidate already
covered by a pending PUT is not a loss. The existing pressure-only counter
retains its meaning. None of these observations change retirement or read reserves.

The executor's completion poll also returns a small pressure snapshot under the
same lock: the number and payload bytes of queued QUERY/GET/PUT jobs, and the
age of the oldest queued or running job. Query payload bytes are zero. Retired
work is excluded; detached native work remains included until it retires.
Sampling scans at most the configured operation limit and never calls storage
or the GPU. Metrics snapshots remain a separate, lower-frequency observation
(at most once a second, and when storage turns pending or idle), exported by
the engine host (`ts-metrics`).
Local I/O remains labeled by TP rank; exporters do not make storage admission
or retirement decisions.

The control plane MAX-reduces these durations and counts across cache owners
from the round's reports. C++ closes a shared read/write admission gate when
any high watermark is reached and reopens it only when all observations reach
their low watermarks. Local wall-clock timestamps never cross this boundary.
The gate is independent of pause/freeze, backend health and quarantine. It does
not pause forward execution, completion handling or soft deadlines.

Closed admission bypasses a new request's L3 preparation. A request still in
QUERY detaches and uses ordinary local admission without waiting for its soft
deadline; no new GET may acquire destinations. GETs already submitted keep
their original lifetime and deadline, and Ready results remain usable. Host
credit exhaustion also rejects a new QUERY before backend work. Closing the
gate drops nonowning backup candidates and rejects new ones before acquiring
Host references. Already submitted PUTs continue; L2 publication is unaffected.
Recovery does not replay dropped candidates. Native retirement is never inferred
from pressure recovery, and an empty runtime queue cannot clear quarantine.

The L2 executor binds immutable transfer geometry and mapped-buffer address
tables for every write lane and load workspace during construction. Their
initial uploads must not occur on the first serving H2D/D2H submission.
Per-operation block tables still use pinned, asynchronous uploads and their
existing lifetime fences.

Queued cancellation is acknowledged without waiting for an occupied worker.
A cancelled queue item becomes inert. Started I/O has a separate lifetime:
request cancellation is not native retirement. Operation IDs never reuse a
request slot or an earlier operation's ID.

The executor's private health monitor rejects work during cooldown and permits
one recovery probe. It has no separate thread or public control interface.
Health epochs prevent a stale successful call from undoing a newer outage.
Existence misses, safely retired GET misses and retired PUT capacity
rejections do not indicate an outage; exceptions, unretired PUT failures and
unknown native failures do. A soft timeout is
only a scheduling decision. The first version uses fixed deadlines and
credits and explicit pressure watermarks; it does not estimate per-request
transfer benefit or auto-tune limits. Pressure does not report a backend fault,
and a successful backend call does not override C++ pressure admission.

## Completion agreement and ordering

Every attention-TP owner sends its newly completed records once over the
replica's collective mesh. The control plane retains partial-owner records
until the other owners finish.
Each round every owner's report rides the replica's relay exchange, one CPU
all-gather that also carries the TP leader's frontend messages: the owner's
monotonic clock, the three pressure observations, its completions, and an
error when its poll was invalid. The TP leader's clock becomes the tick's and
pressure is MAX-reduced; the forward/storage permissions come from the
control state, which every owner holds alike. The report's size follows its
completions, independently of model attention heads and the number of
owners. The agreed tick is delivered every round, including idle and frozen
rounds.
Results are agreed per operation, so a slow earlier operation cannot block a
later complete one. Per-key validity and retirement are AND-reduced;
confirmed missing/invalid object facts are OR-reduced. A single missing shard
invalidates the shared object, but a peer that did not read contributes no
object-failure evidence. The window is bounded by operation and key credits;
it is not truncated to a prefix of pending operations. An invalid local
report joins the collective as an error before every owner raises the
protocol error: local completion records are validated inside that same error
boundary, before the exchange. Per-key masks have equal lengths and success
cannot also report an object failure. Query/PUT cannot report read failures.
Malformed store results follow the store's explicit failure-retirement
contract.

Page-cyclic groups use the existing Host owner translation. Each worker copies
only its locally owned rows; other rows contribute success and retirement but
no object failure or transferred bytes. Their keys name the payload replica
(`attention_tp_rank // shard_count`), so a new Host allocation may restore the
same object on another cyclic owner. Replicated state groups retain rank-local
keys. Startup rejects head-partitioned fields in a page-sharded group: changing
page owner must not change what bytes an object means. The producer namespace
also includes KVP, QCP and linear-attention TP geometry. Page-sharded layouts
include the ordered group shard counts: identical byte layouts with different
replica numbering must not alias. All-replicated layouts retain their existing
namespace. Host allocation IDs and cyclic owner ranks are never namespace inputs.

The head advance applies that tick and cancellation before completion records,
so an already expired/detached read cannot publish a late result. The scheduler
emits storage commands beside cache and forward operations. Storage dispatch
bypasses the GPU FIFO; ordinary Device-copy and forward ordering is unchanged. Frozen rounds still service the same planner with forward/storage
permissions disabled, allowing completions, expiration and cancellation.

## Retirement and maintenance

For the Mooncake adapter, exact byte counts prove GET success. Nonnegative
short reads are retired failures. `OBJECT_NOT_FOUND` (`-704`) is a metadata
miss before destination transfer submission. PUT `OBJECT_ALREADY_EXISTS`
(`-705`) is a create-only rejection before source transfer and is idempotent
success. PUT `NO_AVAILABLE_HANDLE` (`-200`) is the master's allocation
rejection before any source transfer: a retired per-page failure that leaves
the page unacknowledged for the scheduler to re-offer, never a quarantine.
Other negative transfer errors, exceptions after native submission,
and malformed transfer results do not prove retirement.

`Store::failures_retire` declares the store's guarantee when a call fails or
returns a malformed result: whether it has stopped accessing the supplied Host
buffers.
It is not an operation's success flag. Valid results carry their own `retired`
fact, including safely retired misses. The in-process store guarantees retirement
on failure; Mooncake does not infer it from a failed SDK call. QUERY
accesses no supplied Host payload, so its failure does not retain payload buffers.

Only confirmed missing or invalid GET objects enter C++'s bounded negative
metadata. Health/queue/shutdown rejection, queued cancellation and backend
exceptions carry no object evidence. They still complete the request's attempt
and preserve the retirement contract, but the same keys can be queried and read
by a new request after recovery. Genuine negative metadata retains its existing
bounded FIFO lifetime; expiry/reprobe after external object repair is separate
work. The executor reports facts and does not choose negative-cache policy.

An unretired result quarantines every reference of that job. It is never
published, freed or repaired by an existence recheck. One generic negative
transfer result is enough: the executor rejects all further submissions,
and `close()` reports that a process restart is required. The first version
stops new L3 work for the replica after quarantine and requires process
restart to reclaim that storage safely. Ordinary computation and valid L2
hits can continue within the remaining capacity. A deployment SDK must honor
the adapter's return-code contract; real transport fault tests remain part of
hardware validation.

A full cache clear, namespace rotation or weight update must reject outstanding
storage operations, including Ready holds and quarantine; it must not
synchronously drain QUERY/GET/PUT on the scheduling thread. After replica-agreed
preflight, remote namespace deletion retains the existing synchronous
administrative call on the control thread. This purge can delay request and
control processing until the backend returns; preparation soft timeouts do not
bound or cancel it. A deletion failure prevents local cache clearing.
Asynchronous administrative purge is outside the first-version scope.
L1-only clear may proceed while Host-only storage I/O remains active.
Shutdown closes the L3 executor when the host's event loop ends, before the
data plane's device-copy cleanup. L3 joins its workers with a deadline and then
closes its store; if native retirement remains unknown it refuses store
close/Host deregistration and keeps the store client for the process, as the
data plane keeps the Host allocation.

## Initial scope and validation

### Observability

With Prometheus metrics enabled, L3 exports rank-local query/GET/PUT counters,
queue/backend cumulative durations, queued/running/completed job counts and
their payload-byte totals. Completed jobs remain charged until their facts
are polled. Separate scheduler gauges retain visibility of Ready holds and
quarantined references after runtime completion. Waiting soft timeouts and
Ready expiration are different counters; client detachment increments neither.
Pressure bypasses and dropped backup candidate keys have separate counters;
the admission gate is a mirrored scheduler gauge. These do not count unique
remote objects or saved prefill tokens.

Health state/epoch and unproven native retirement are separate gauges. Reading
health never consumes a recovery probe. Snapshots carry no request IDs or object
keys. Labels identify model, engine role, DP rank and TP rank: I/O is shard-local,
while scheduler resource/deadline values are mirrored and must not be summed
across TP owners. Successful adapter byte counters include idempotent PUT hits
and detached GET results; they measure neither network traffic nor published
cache hits. Metrics do not participate in scheduler decisions or agreement.
The `backend_calls` event counts adapter invocations; a create-only PUT can
finish after its existence check without calling the native PUT API.

### Supported scope

The first version supports PP=1 and the existing Host-KVStore model/cache-group
capabilities, including cyclic KV-page ownership with replica-stable keys.
Existing model, backend and QCP/DCP role restrictions still apply; storage
support does not establish validation of every model/parallel configuration.
Fused and PD-prefill roles can prepare L3 hits. Ordinary PD-decode remote
landing keeps its Device-only prefix contract; local recovery uses the existing
matcher. This version adds no PD handshake, direct L3-to-Device DMA, shared
read deduplication or layer-wise GET publication.

Scheduler tests exercise mixed block granularities, sliding windows, sparse
state, partial results, deadline expiry, cancellation, ID reuse and ownership.
Executor tests (`ts-l3`) cover bounded queues, health epochs, queued
cancellation, shared-store byte round trips, KVP ownership, quarantine and the
Mooncake result codes. Multi-rank tests over the control plane's TCP mesh
(`ts-control`'s `l3` module) cover out-of-order owner agreement and collective
error handling, and its serving tests the executor inside the round. GPU
H2D/D2H and real Mooncake/RDMA validation are distinct from these CPU
protocol tests.

The Mooncake CI workflow runs the executor against a local master and an
independent provider with `MOONCAKE_REQUIRE_MASTER=1`: byte round trips,
objects outliving their writer, KVP owner relocation, tenant isolation and
recovery after a master restart (`ts-l3`'s live tests), plus scheduler tail
round trips through the in-memory and native stores. These CPU tests do not
establish native RDMA or GPU-byte correctness.

`test/runtime/test_l3_flat_kv_e2e.py` checks packed history/state groups through
D2H, asynchronous PUT/GET and H2D, including untouched GPU bytes. Its native
case requires `MOONCAKE_GPU_TEST_CONFIG`, a JSON backend configuration with
`global_segment_size=0`, and an already running master plus an independent
storage owner. It exercises both `direct` and `kernel` I/O settings. Each writer
process exits before a new reader allocates Host/GPU buffers; the reader restores
the objects into different block slots and checks exact bytes, then deletes
them and verifies safely retired misses. Without that environment variable,
the native case skips and only the memory-backend case runs. This transfer test
complements serving tests that verify scheduler admission and model output.

Validation counters must distinguish successful adapter PUT results from new
native writes: create-only PUT can succeed because an object already exists.
Native PUT success can also mean that the SDK skipped an existing object;
successful submitted payload is not proof of new writes or network traffic.
Use SDK transport counters when reporting wire bytes. Serving validation
should use fresh inference processes with empty local caches, verify successful
GETs on every owning rank and the prefix actually reused, and compare outputs
with a recomputation run. Prompts should require information in the restored
prefix. Byte-for-byte transfer tests establish payload preservation separately
from model-output comparisons.

The primary latency measure for L3 prefix reuse is end-to-end time to first
token (TTFT), compared with recomputation when the tested prefix is absent
from both Device and Host caches. Report the prefix actually used by admission,
remaining prefill work and fallback frequency alongside TTFT. Separate remote
DRAM hits, SSD hits with warm or cold file caches, and misses. Concurrent decode
time per output token (TPOT) is an interference check; reducing idle L3 agreement
overhead does not establish a prefill benefit. A storage call overlapping GPU
work also does not establish how much latency was hidden on the request's
critical path. PD measurements must include transfer and decode-side
admission through the first delivered token.

Comparisons use equal Device capacity, warm execution paths and the same
documented wait budget across cache-off, L2-only and L3 runs. Partial hits,
misses and fallback remain in the results. Reset local caches only while idle
without purging the remote namespace; keep reset, cleanup, profiling and
fault injection outside timed cohorts. Check the engine's single-request
limit in both PD roles against input and generation tokens: a hybrid request
must fit every cache group, and context truncation invalidates output
comparisons. Small test cohorts do not establish a production tail guarantee.

See [server configuration](../configuration/server.md#host-l2-and-mooncake-store-l3)
for limits and backend settings, [cache concepts](cache-concepts.md) for
geometry and key compatibility, and [event loop](event-loop.md) for the two
centralized feedback points.

## Disabled storage and scope

Without a storage backend, startup creates no L3 executor or SDK client, the
control plane runs no storage poll or owner agreement, and the scheduler skips storage
preparation and reference-handoff scans. Ordinary Device/Host matching,
admission and execution remain the same path. The runtime requires the
scheduler built from matching source even when storage is disabled.

L3 preparation preserves Decode destination publication and receiver lifetime:
restored Host data enters the existing H2D, forward and PD transfer path.
Storage adds no model-specific forward or attention implementations.
