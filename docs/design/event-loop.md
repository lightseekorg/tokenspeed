# Event loop: design principles

This document records the design principles of the scheduler event loop: the
Rust control plane every engine rank runs (`tokenspeed-engine/crates/ts-control`,
started by the `ts-host` binary) and its boundary to the Python data plane
(`python/tokenspeed/runtime/engine/rust_host.py`). It is the reference for
where new logic belongs, how components feed results back into the scheduler,
and what the loop body itself is allowed to contain. The rules below are
deliberate; treat deviations as bugs in review.

## The process

One engine rank is one `ts-host` process. Its threads:

| thread | runs | holds |
| --- | --- | --- |
| control (`ts-host` main) | `ts_control::event_loop::EventLoop`: the frontend wire, the collectives between ranks, the C++ scheduler, commit post-processing, control operations | the scheduler, every request's state, the collective mesh; never the GIL |
| pump (`ts-pump`) | the Python `DataPlanePump`, one queued operation at a time | the `DeviceHandle`; takes the GIL per operation |
| forward (`tokenspeed::forward`) | `ForwardThread`: everything that touches CUDA | the executor, models, KV pools, CUDA graphs |

Peripheral threads stay off the round: the wire's Tokio runtime
(`ts_control::transport`), the PD transfer peer's workers (`ts-pd`), the
grammar compilers, the KV event publisher, the expert-rebalance solver.

The ranks of a replica (an attention-TP group, across every pipeline stage)
run the same scheduler. They talk over `ts-collective`'s TCP mesh, in fixed
groups (`Ranks`): the replica, the pipeline (one rank per stage), the
attention-DP world, the EP group. The replica's first member owns the frontend
connection and relays each round's messages (`relay.rs`), so every member
admits the same requests in the same round; only it sends outputs. With
attention DP the replicas' schedulers differ but their forwards run in step:
each round every rank shares its batch's size, and a rank with none runs an
idle forward when another has one. A stop request rides those collectives, so
every rank stops after the same round. One rank is the same path with groups
of one.

The launcher (`entrypoints/engine.py`) starts the ranks through
`start_rust_host`, whatever the deployment: the in-process `Engine`, the
gRPC servicer behind `ts serve`, and `ts serve --headless` for an external
frontend. Each rank dials the frontend's handshake socket; the in-process
frontend is `tokenspeed_engine.Frontend` (`ts-py` over `ts-client`) under
`AsyncLLM`. An EPD encode node runs `epd/encode_loop.py` instead: no scheduler,
no event loop.

## Principle 1: the loop runs no GPU work, and cannot reach any

The event loop is the **control plane**. `ForwardThread`
(`execution/forward_thread.py`) is the **data plane**: one thread per rank,
FIFO, everything that touches CUDA. A control-plane round is microseconds, so
the cross-rank collectives that keep the redundant schedulers aligned always
find every rank promptly, however deep the GPUs are in queued work — a stage's
launch-queue backpressure stalls only its own forward thread, never the round.

The control thread cannot run device work by construction: it is Rust, holds
no torch object and never takes the GIL. Its only way to the device is the
`DataPlane` trait (`event_loop.rs`), which `ts-host`'s `PythonDataPlane`
implements by queueing operations for the pump thread. Only `complete`, the
cache poll and a DP idle forward wait for the pump, because the loop needs
their answers; every other operation is fire-and-forget, so a slow pump delays
nothing but its own queue. A round with nothing for the device (no batch, no
cache op, no remote stream: `plan.needs_device()`) does not reach the pump at
all: a loop waiting on a transfer would otherwise hand the interpreter an
empty plan per round and starve the device side's own Python threads of the
GIL.

The pump thread is the control plane's Python half, and it runs under a
thread-local dispatch guard (`maybe_control_plane_guard` in `device.py`, on
by default): any CUDA tensor op from it raises. Event waits — the inbound
channel — and metadata-only view ops pass untouched. Neither submits device
work. `TOKENSPEED_GUARD_CONTROL_PLANE=0` is the escape hatch for a deployment
that trips on an unrouted op (report it).

FIFO describes submission ownership, not a promise that every model kernel
uses one CUDA stream. Model kernels may share main-stream scratch and
persistent kernel protocol state across calls only while those calls are
ordered on that stream. Work deliberately forked to a side stream must use
private storage or establish an ordering edge before touching a shared layout.
Joining the side stream later does not make concurrent reuse safe.

**Visibility**, not discipline, enforces the rest. `build_device_side`
(`execution/device.py`) constructs the model runners, attention backends, KV
pools and executor as its own locals, and returns one `DeviceBuild`, split by
how long the caller may hold each piece:

| | what it is | lifetime |
| --- | --- | --- |
| `DeviceSpecs` | plain values the control plane plans with: cache geometry, cache groups, speculation widths, capability flags | keep forever |
| `DeviceHandle` | the running handle: the complete list of what the control plane may ask of the device side | held by the pump only |
| `transfer` | the PD planner the Rust transfer peer calls (`pd/host_peer.py`), or None outside PD | handed to `ts-pd` at startup |
| `encoder_model_facts` | a callable resolving the encoder facts EPD admission needs (raises on text-only) | consumed at startup, past the EPD gate |

`HostBootstrap` (`rust_host.py`) turns the build into what the Rust side
takes: the scheduler, a plain `HostConfig`, and the pump. Nothing else of the
build crosses.

Encoder facts describe the transferred tensor, independently of the language
embedding width. A model may declare `get_encoder_output_spec()` as a width and
dtype pair, including integer dtypes for discrete visual tokens, without a
loaded vision tower. Without that declaration, the width comes from the language
config and the dtype from the vision tower's dtype or parameters. The resolved
width must be positive. The builder resolves this once; admission keeps only
the resulting values. Converting an encoded payload into language embeddings
remains model work on the forward thread, after all required items and prefill
ranks have completed admission.

`build_device_side` builds the device side **complete**, not built-then-wired:
it constructs the transfer planner from its prepared components and startup
arguments, and reads the engine's role off it once, at construction.

Common MoE weight processing reserves persistent DeepEP communication
storage, before attention/cache construction profiles available memory. The
kernel package owns allocation and compatible reuse. Every DeepEP backend
shares this rule. Runtime orchestration supplies configuration but does not
infer token layouts or capacities from model names. Backend dispatchers reuse
that storage when model execution begins.

Attention construction returns a frozen, named `AttentionBuild` containing its
backends, pools, cache storage, field placement/readiness and optional logical plan.
`build_device_side` consumes this result locally and passes stage field
placement, producer readiness and `logical_plan` explicitly through PD
construction to `get_kv_args`.
These startup dependencies stay within device construction. The control
plane never sees the build result or its ownership/layout fields.

The handle owns BOTH executors of a scheduler plan, and treats their work
identically: a model forward runs asynchronously on the GPU, and handing a
prefill or decode to the peer node is asynchronous work of exactly the same
standing. The plan separates its streams by executor — the `ForwardBatch`
is the model's work, `plan.remote_prefill` is the peer's on a D node (pull
the admitted prompt's KV in), `plan.remote_decode` the peer's on a P node
(the completed prompt decodes over there, so its KV goes out). The remote
streams ride beside whatever forward work the round schedules, occupy no
batch slot, and go out even on rounds with no batch at all — everything
dispatchable dispatches in one round; nothing is ever withheld after
planning (an L3 object is fetched into Host before its request is admitted,
so no planned forward or remote stream depends on a fetch that can miss).
The transfer itself is the Rust peer's (`ts-pd`, over the Mooncake C API):
its workers write KV-pool device memory over RDMA, ordered against forwards
and page zeroing because the pump hands each remote stream to the peer from
the same queue position as the plan's other work. Its control face — request
registration and abort, the per-round agreement on transfer progress — is the
loop's `TransferPeer` (`pd.rs`), where it feeds Principle 3's tail advance.

Consequences:

* The control plane cannot **name** a model runner, backend or KV pool, so it
  cannot pass one implicitly or mutate one a forward is still using.
* A new device interaction that runs once at startup belongs **inside the
  builder**; it belongs on `DeviceHandle` (and the `DataPlane` trait) only
  when the running loop genuinely needs it — the second widens what the loop
  can do to the GPU mid-flight. Hand over the capability, never the object.
* Every method on the handle is a **named operation**, with one registered
  exception: `run_multimodal_work`, because multimodal feature lifecycle is a
  state machine reached from several points (EPD admission's stage/drain
  device half; the release of a request's SHM features). A generic "run this
  closure" slot is the hole this whole design closes, so a second KIND of
  user does not join it — it gets its own name.
  L3 adds named Host-tier operations for the existence probe,
  the prefetch lane's progress and completion (the loop converges them each
  round), namespace and weight-version changes, and cache shutdown. These
  keep the executor and Host buffer hidden; they do not introduce another
  generic work slot. The
  retraction snapshot pool adds no operation at all: its stores ride
  `execute` with the write-backs, its restores with the load-backs, its ACKs
  `poll_cache_results`, and `shutdown_cache` closes it with L2 -- one
  `HostCacheExecutor` behind the handle, two Host buffers inside it. The
  plan's cache ops cross the handle through one adapter,
  `scheduler_utils.cache_ops_from_plan`: the L2 batches (`Cache.WriteBackOp`,
  `Cache.LoadBackOp`) pass through, and the scheduler's batched
  `Cache.SnapshotOp` / `Cache.RestoreOp` become one per-request `SnapshotOp`
  / `RestoreOp` (`cache/transfer/ops.py`) each -- the executor's unit of
  work, one slot exported or imported, one ACK. The control plane counts the
  plan's ops from the same plan (`CacheOps::dispatched`), so what is counted
  is what is submitted; the plan's per-request snapshot store count, one per
  retraction victim, is the round's preemption count, exported as
  `tokenspeed:num_preemptions_total`.
  The `EXPERT_LOAD` profile activity adds two more named operations,
  `reset_expert_load` and `dump_expert_load`: the routing kernels bump the
  expert placement's load counters on the execution stream, so zeroing and
  reading them back ride the forward thread on that stream (the read-back is
  a deliberate, low-rate host wait, like the other `run_*` methods). The
  dump is rank-local by contract -- it runs no collective: a profile stop
  reaches every attention-DP engine independently, so a rank reducing inside
  the request would block in the collective while its peer is still in the
  round's DP agreement. `--init-expert-location` sums the ranks' records
  where it consumes them.
  Online expert rebalancing (`--enable-eplb`) adds two more named
  operations on the same counters: `snapshot_expert_load` (read the window
  to the host and zero it, one stream-ordered step) and
  `apply_expert_placement` (one chunk of layers: P2P over the EP device
  group into a startup-reserved staging buffer, slot copies, then each
  layer's routing tables switched in place after its slots landed, and a
  final execution-stream synchronization so a P2P failure surfaces in the
  op). Both are rank-local on the handle. The rebalance may reduce the
  snapshot over the EP group because its steps ride the same-round gate
  (below), unlike the profile's dump.
  Changing this surface requires updating this contract.
* The role is a **value** (`DeviceRole`), not a class hierarchy. With every
  remote op on its own plan stream, the batch needs no per-role reading at
  all: a `ForwardBatch` is model work, full stop, and a DP rank counts work
  by asking whether the batch has tokens.
* On the per-round path only commit waits, through `PendingExecution.result()`:
  join the forward thread's future (launches issued), then its copy event (D2H
  landed). Dispatch never waits — it submits model forwards and the peer's
  remote prefills/decodes alike fire-and-forget, which is what keeps a
  backpressured stage off the control plane. A raise is for broken invariants
  only: a transfer worker or another Prefill rank can fail a request's
  transfer while its submission waits, and that submission then does nothing
  for the request, whose failure arrives as its transfer state. The handle's
  remaining `run_*` methods do block, deliberately — the DP idle forward, the
  landing of a completed remote prefill, the KV repair after a wake, the RL
  weight sync — but each is a low-rate path whose caller cannot proceed
  without the result (the landing's failure must surface BEFORE the
  scheduler advances the request into decode). A new blocking operation on
  the per-round path is a bug.
* Keep per-forward state resets asynchronous on the execution stream too.
  Use device-side scalar fills for indexed flags: assigning a Python scalar
  through tensor indexing can stage a CPU tensor and introduce a hidden
  synchronous H2D copy, blocking further forward launches.

EPD prefill admission's device half — receive-buffer allocation, the publish
clone/scatter, and the NCCL shard reassembly — crosses through
`DeviceHandle.run_multimodal_work`, so it runs on the forward thread, which
also puts the reassembly broadcasts on the same issuing thread as the model's
collectives and keeps the cross-rank launch order.

With pipeline parallelism, only PP0 receives encoder payloads and reserves
the receive pool. PP0's attention TP group first agrees on completion;
`PipelineEpdPrefillAdmission` then broadcasts admitted/failed request IDs
along the pipeline CPU groups. Every stage submits the same requests in
the same round, while later stages hold no encoder receiver or
payload. Decode does not build this admission controller.

The encoder's TP ranks attach every pixel-SHM segment in a received batch
before any rank submits it to the encode worker. A CPU-group rendezvous
protects this ownership transition: submission consumes and unlinks SHM
on both cache hits and misses, so broadcasting handles alone is insufficient.

The symmetric rule holds on the data plane: **the forward thread never
synchronizes with the device on the per-round path.** Every host
synchronization it performs — `.cpu()`, `.item()`, `.tolist()`,
`bool(tensor)`, `nonzero`, a copy from or to pageable host memory,
`stream.synchronize()` — waits for the whole stream, and the stream holds
the step in flight, so the next step's prologue and graph launch slip
behind the current step's completion and `in_flight_depth` degrades to 0
however it is configured. Results cross back through pinned non-blocking
copies and an event the pump waits on. Torch's sync-debug mode enforces this
rule, and `HostBootstrap.enter_serving` arms it once the rank finished
startup and the frontend handshake (weight loading, tuning, capture, the
transfer and L2 builders all synchronize on purpose):
`TOKENSPEED_DATA_PLANE_SYNC_DEBUG=warn` reports each offending site with its
Python location, `error` raises there. CI runs the serving paths with
`error`; the pump's event wait and non-blocking copies pass unflagged, so a
report is always a real stall.

A JIT compilation on the forward thread is the same kind of stall, only
longer: the round waits 100 ms to seconds while a kernel compiles for a new
compile-cache key. Startup compiles on purpose (graph capture, tuning), so
the bootstrap installs `tokenspeed_kernel.compile_monitor` before building
anything and `enter_serving` marks the end of startup right after arming the
sync-debug mode. The encode loop marks it after its ready message. From then
on the monitor logs every Triton compilation with its duration and cause and
reports by name any compile-time kernel parameter that keeps taking new
values from one call site -- a per-batch value passed as `tl.constexpr`;
each commit reports the compilation counts and durations, which the control
plane exports as `tokenspeed:jit_serving_compiles` /
`tokenspeed:jit_serving_compile_seconds`. `TOKENSPEED_JIT_COMPILE_CHECK=error`
raises there instead, and CI serving jobs run with it.
The same mark closes the kernel package's compile switch
(`compile_monitor.is_serving()`), installed monitor or not. A kernel whose
library compiles once per batch shape outside Triton -- FlashInfer's joint BF16
GEMM and the ll_bf16 router's dot-product kernel compile once per exact row
count -- runs only before it:
graph capture records it, and eager calls after it take a GEMM that never
compiles instead of compiling on the forward thread.

A full garbage collection is a stall of the same kind, for every Python
thread of the process at once. Startup leaves millions of long-lived Python
objects behind (modules, parameters, captured graphs, kernel caches), and
CPython's full (generation 2) collection walks all of them in hundreds of
milliseconds, at a moment each rank's own allocations decide. Since every
round waits on the whole group, each rank's collection stalls every rank. So
`enter_serving` calls `freeze_startup_heap` (`engine/gc_hooks.py`) after the
JIT mark: it collects once, freezes the survivors (`gc.freeze`: no later
collection walks them) and raises the generation-2 threshold far enough that
CPython does not start full collections by itself. The pump's `GcHooks` runs
one every `FULL_COLLECTION_INTERVAL` forwarded rounds, a count that is the
same on every rank, so the ranks collect in the same round and each
collection walks only what serving keeps alive. The young generations still
collect automatically.

### Profiling the issuing thread

Profiling must include the forward thread that issues device work. Batch and
request annotations describe the immutable submitted work, not its later
completion. Collecting them must not read device tensors or change scheduling
and synchronization behavior.

Device work may outlive its host range. Analysis must correlate launches with
device events within each rank and process rather than clip GPU time to host
timestamps. Profile startup rejects unsupported capture capabilities instead
of silently producing an incomplete trace.

### The capture contract

Information crosses to the data plane **only** inside the submitted closure,
and freezes once submitted: no attribute rebinding, no in-place edit, no
releasing a resource the closure captured. Capture plain values or a snapshot,
and bind at capture time rather than closing over a variable the caller later
rebinds. Results cross back **only** through `PendingExecution.result()`.

Between the two Rust and Python halves the same holds by construction: an
operation's arguments are plain values (lists, bytes, the scheduler's plan
moved into a Python object), and a forward's results come back as plain
lists the pump builds from the landed host copies.

The plan's cache ops follow the same capture rule: `DeviceHandle.execute`
adapts them once and the queued submissions read that list, not the live
plan. The L3 prefetch lane is a CPU thread with no stream at all; its outcome
reaches the forward thread through nothing, and the control plane only
through the executor's progress table.

`execution/forward_thread.py` states this in full, including the registered
exceptions — grammar matchers, whose ownership splits by path and whose
overlap the drain rules in Principle 4 break instead, and multimodal items.

## Principle 2: the event loop is a coordinator, nothing more

`EventLoop::round` sequences components. It does not implement them. Domain
logic lives in its own module of `ts-control`, and enters the round as a
single call:

| module | owns |
| --- | --- |
| `ingress` | validating a request against what the engine serves, its sampling defaults and budget, the scheduler request |
| `output` | committing a forward's results: request state, the scheduler's events, the frontend's outputs, logprobs |
| `cache_ops` | the open cache-op count and the replica-agreed acknowledgments |
| `control` | pause and resume, memory release and resume, the weight-operation and internal-operation queues |
| `pd` | the transfer peer as the loop sees it, and the per-round agreement on transfer progress |
| `grammar` | grammar compilation, its cache, and the matchers |
| `eplb` | when an expert rebalance runs, the placement solver, the slot moves |
| `kv_events` | publishing the scheduler's KV cache events |
| `batch_log`, `request_stats` | the operator's per-round and per-request log lines |
| `relay`, `transport` | the replica's relay of frontend messages, and the frontend wire |

The round should read, top to bottom, as the schedule of one scheduling
round, with no feature's internals inlined into it.

Consequences:

* Low-frequency or optional features (pause/resume control, EPD, PD, L3)
  must never make the *normal* scheduling path harder to read. If
  understanding decode throughput requires skipping over your feature's code,
  the feature is in the wrong place.
* A subsystem module owns its state and depends on static configuration and
  what the round passes it; it holds no reference to the loop. The loop
  holds it, calls it at a fixed point, and applies what it returns.

## Principle 3: scheduler feedback is explicit and centralized

The scheduler advances in exactly two places, both in `EventLoop::round`
itself — never in a helper or subsystem. Helpers RETURN their events, or push
them into the round's `events` list. Reading the round alone must reveal
every point where the scheduler's state advances, and why:

* **Head of the round** — completed Host cache ops (`CacheOps::acknowledged`:
  `WriteBackDone`, `LoadBackDone`, `PrefetchDone`, `SnapshotDone`,
  `RestoreDone`, each replica-intersected), through `advance_cache`. These
  must advance *before* `next_execution_plan`; otherwise, cache-gated
  admissions -- a prefetched request's admission, a suspended request's
  restore, which waits for both of its store ACKs, then its own -- are
  delayed by a full round.
* **Tail of the round** — forward results and PD transfer events, funneled
  through the round's single `events` list, through `advance`; all events
  must advance before the *next* round plans.

Anything that produces scheduler events (a new transfer backend, a new async
op kind) either returns events into one of these two points or adds a new
explicit call site in `round` with a comment stating why the existing points
don't fit. It must not advance the scheduler itself.

## Principle 4: correctness never depends on the in-flight depth

`in_flight_depth` parameterizes the loop: 0 (classic synchronous
commit), 1 (the overlap schedule), or `pp_size` (the prefill chunk pipeline).
Dispatched forwards await commit in the `in_flight` queue. The round commits
once the queue exceeds the effective depth (0 when the round dispatched no
new work, so results never wait on future traffic).

The depth is a performance knob only. Any dispatch whose inputs depend on a
pending commit's side effects must commit the queue first, and
`plan_and_dispatch` holds every such rule in one place (currently: an
eager-grammar batch, whose bitmask fill reads matchers only a pending commit
advances). Rounds that run no real forward (a frozen pause, a DP idle forward)
commit the queue fully.

Prefer removing a dependency over registering one. The P-side remote decode
was registered here until the C++ scheduler learned to hold it until its
final chunk's forward result lands (it now rides `plan.remote_decode`,
beside the batch): a request turns `PrefillDone` when its last chunk is
*scheduled*, so the transfer was being planned while that chunk was still in
flight, and satisfying it meant draining the whole queue — under PP,
emptying the chunk pipeline every time a prompt finished prefill. The
dependency was real. The right fix was upstream, not a drain.

Depth ≥ 1 also means the loop plans a round *before* the previous round's
commit, so a batch can contain a request that commit is about to finish.
Anything the data plane frees when a request ends — its grammar matcher, its
shared multimodal features — is released through the pump's queue, behind the
forward that captured it, and the handle orders the device-side release on
the FIFO, not inline.

Only a pipeline's last stage samples. Each of its ranks shares the forward's
results with the other stages of its pipeline before they commit
(`adopt_last_stage`): the tokens, their logprobs and the prompt logprobs, the
NaN guard's flags, the next inputs and where the grammars ended. Every stage
then commits the same result against its own mirrored plan, so its finishes
and aborts are the last stage's.

## Principle 5: publishing drains, once per round

KV event publishing has drain semantics: cache mutations accumulate inside
the C++ scheduler across any number of calls (advance,
`next_execution_plan`), so a single unconditional `drain_kv_events` after the
round's last scheduler mutation publishes everything the round produced, in
order, as one batch. The batch is the round's net change: a block evicted and
cached again within the round produces no event. Do not add per-mutation
publish calls. They only fragment batches.

The same reasoning fixes the metrics call: the loop records scheduler
iteration metrics once per round, from the same pre-dispatch snapshot as the
scheduler stats.

Metric output plugins run in the node's metrics relay process
(`metrics/relay.py`); the rank's `ts-metrics` hands them every update as an
event and writes Prometheus' multiprocess files itself, so the loop never
waits on a plugin. A full relay queue drops events and counts the drops.
Plugin discovery and creation happen outside the loop. The metrics adapter
(`ts-metrics`) sends every value change and re-sends an unchanged gauge at
most once a second; this policy belongs to the adapter, not the scheduler
loop or individual plugins. See
[metric output plugins](../metrics-plugins.md) for event semantics,
registration, and output lifecycle contracts.

Per-request metrics anchor to request lifecycle points instead of the round
snapshot. `tokenspeed:request_queue_time_seconds` stamps each request's
waiting-queue entry when it is submitted to the scheduler — normal admission,
the EPD drain and the pause resume all submit through one funnel — and
observes it once, when the request first appears in a dispatched forward. A
retracted request re-enters the C++ waiting queue without passing the funnel
again, so preemption yields no second observation. The stamp rides the
request's state and dies with it; no cleanup pass exists.

Page gauges count LCM parents, excluding the reserved null parent. The
per-round sampler reads `empty_lcm_blocks()` and `active_lcm_blocks()`.
Cached-only parents are `num_usable_pages - empty - active`. Active and cached
counts are disjoint, even when cache groups pack multiple blocks into a parent.
`available_lcm_blocks()` scans the pool and prefix indexes for reclaimability
and belongs in diagnostics and leak checks, never in the per-round sampler.

## Subsystems in detail

**Pause and memory release** (`control`). A pause in `abort` mode aborts what
runs, `wait` lets it finish; both stop admissions and answer once the
scheduler drained. `keep` freezes everything and answers at once: a frozen
round commits what is dispatched, plans nothing, and still joins the
attention-DP agreement (an idle forward when a peer runs one, but not while
memory is released: the weights are unmapped). A memory release drains like
`wait`, then frees its regions — one release/resume pair per tag, whose
data-plane half (quiescing copies, the transfer engine's registration, the KV
repair, NCCL suspend) runs on the pump — and freezes until they are resumed.
A disaggregated engine refuses new requests while a release is draining or
memory is released: their transfer registration would name memory the
release unmaps. On release the prefill peer deregisters its cache regions and
registers them again on resume under the same session; the decode peer
starts a fresh transfer engine on resume, so its prefill peers see a new
session and register it again. A release is refused while transfers are in
flight, and refused for a paused PD engine with buffered requests, whose
transfers cannot drain until those requests are admitted; their transfer
registration is deferred until resume admits them, and aborted buffered
requests never register one. An empty memory-resume request is a successful
no-op while a drain is pending; it preserves that drain's admission hold
until the owning pause or memory-release operation finishes.

**The same-round gate** (`EventLoop::gate`, every round on every rank).
Weight-update `flush_cache` and standalone `/flush_cache` first MAX-reduce
flush intent across attention DP so every DP rank enters the same
collectives — the frontend sends a flush to every engine separately. The
same reduction gates the weight operations themselves (NCCL group
init/teardown, distributed, tensor, disk and Mooncake loads): each is queued,
and the head completes only in a round where every DP rank holds one of the
same kind at its head (a mismatch is an error — the frontend sends every
operation to every engine in one order). The device call blocks the control
thread in a collective the trainer drives or an SDK read; a DP peer that had
not yet dequeued its copy would keep looping and wait for this rank in the
per-round agreement, a deadlock. One operation completes per round, and the
replica MIN-reduces its result before the loop publishes the L3 weight
version or sends the reply. The same gate carries the online expert
rebalance's *internal* steps (snapshot, commit, chunk apply) from a second
queue with its own columns: frontend operations arrive on different DP
engines in different rounds while the controller queues internal steps at
rank-identical rounds, so one shared queue would make the heads differ in
kind across ranks. When both are ready the frontend operation completes this
round and the internal step the next — still one blocking device call per
round. Because every EP rank is inside a gated step at once, the rebalance
may run a collective over the EP group there (summing the load under
all-to-all EP, broadcasting the committed placement), which the profile's
`dump_expert_load` must not. The manual `POST /rebalance_experts` is a
frontend operation on the first queue that starts the same internal
sequence. The loop refuses a memory release while a rebalance is in progress
and, once drained, the release waits while chunk steps still write the
weights. A cache clear MIN-reduces the scheduler's non-mutating
`can_clear_cache` across the replica and then across attention DP — DP
replicas share Mooncake objects — then MIN-reduces an error-returning L3
namespace deletion, before any rank mutates Device/Host. The frontend ANDs
every engine's reply. Independent TokenSpeed jobs that share a tenant are not
in those groups.

**The expert rebalance** (`eplb`, `--enable-eplb`). The controller counts the
forwards this rank submitted — real or DP-idle, so the count is
rank-identical — and decides when a rebalance's steps are due; the round
queues them for the gate. EP rank 0 derives the placement on a solver thread
of its own, off the round; the other ranks take the broadcast. Nothing in
the round knows a rebalance exists beyond the one call that notes the
round's forward.

**Cache-op acknowledgments** (`cache_ops`). The scheduler is mirrored on
every rank of the replica, so an acknowledgment may only advance once every
rank holds it: the ranks agree on whether any has work, then intersect what
they hold. L3 Host backups finish asynchronously, so a rank-local ACK would
publish a Host block on one mirrored scheduler while a peer still has the op
pending, and a KVP rank that copied a different owned subset of an image
completes at a different time. A failed L3 backup rides the same agreement
(MAX-reduced beside the work flag) and fails every rank after it, instead of
one rank failing while its peers wait in the gather. Every rank stays in
every gather, even when its own part is empty. The poll runs these
collectives only while a cache op is open: every rank counts the same plans
and advances the same intersections, so the open count is the same
everywhere and an idle replica polls nothing.

**L3** (`--kvstore-storage-backend`). The loop owns the two replica-wide L3
decisions. At submission it probes the store for the requests' prefix pages
and registers with the scheduler only the pages every replica rank found
(MIN), so every mirrored scheduler registers the same keys; without L3 it
hashes no prefix and runs no collective, a cost the default path must not
pay. The probe is not a lease, so the L3 pages are fetched before the
request is admitted: the scheduler allocates their Host blocks, emits a
`Cache.PrefetchOp`, and holds the request in a pre-admission state
(`Prefetching`: no Device page, no request-pool row, admission skips past it)
while the executor's lane fills the pages in prefix order, stopping at the
first missing object or at the deadline -- which bounds when the next batch
may start, not a batch in flight: the store's `batch_get_into` has no
timeout, so a hung get holds its active Host pages, namespace deletion and
shutdown. The deadline is counted from submission; an expired queued job
becomes a zero-page miss only when cancellation proves its worker never
started. Active I/O retains its pages until completion. Each round, before
the cache poll, the loop MIN-reduces every in-flight prefetch's done flag and
landed prefix across the replica (in op-id order) and completes a converged
op on the device with the common prefix, so its one
`PrefetchDoneEvent(op_id, landed_pages)` rides that round's poll; the
scheduler publishes the landed prefix in the Host tier and forgets the rest
of the keys, and the request's next admission is an ordinary Host hit whose
Host-to-Device load overlaps its first chunk layer by layer, as any L2 hit's
does. No forward is skipped and nothing is retracted; there is no blacklist
(a later probe hit may register a key again). A backend exception or
malformed existence result is a local miss, a lane fault a landing of
nothing, so every cache-owning rank still enters the replica MIN; raising
would hang healthy peers. Clients are not failed. The D role never
prefetches: its admission probes Device only.

**PD** (`pd`, `ts-pd`). Requests register with the transfer peer at
admission, and every round the replica agrees on which transfers changed
state; their scheduler events join the round's tail feedback, and a failed
transfer aborts its request on every rank together.

**EPD prefill** (`--disaggregation-mode prefill` with encode workers). A
request whose images an encode worker embeds is staged at admission; each
round the replica MIN-reduces which staged requests' embeddings landed on
every rank, submits those, and fails those whose receive failed or timed out
on any.

**Grammars** (`grammar`, `--grammar-backend xgrammar`). A constrained
request waits in the loop's grammar queue until its grammar compiled (off the
round, on compiler threads) on every replica rank, and fails if it timed out
on any or does not compile; every rank admits the same ones in the same
round. The matchers live in Rust (`ts-grammar`); the data plane fills
bitmasks through their handles.

## Anatomy of a round

For orientation, one `EventLoop::round`:

1. The frontend's owner receives (waiting briefly only when the whole engine
   is idle) and relays the round's messages to the replica, with its stop
   request. Admit them in arrival order — a control operation and the
   requests before it keep their order — then apply the pause edges, admit
   the grammar queue and the EPD staged requests that are ready, and run the
   same-round gate.
2. With L3, converge the in-flight prefetches; then, while a cache op is
   open, poll the data plane and **advance the scheduler (head)** with the
   acknowledgments every replica rank holds, so this round's plan sees them.
3. Frozen? Commit what is dispatched and join the attention-DP agreement.
   Otherwise: plan (`next_execution_plan`), count its cache ops, finish the
   requests the plan aborted (`plan.aborts`: a capacity retraction whose
   victim could not be imaged aborted the newest resident inside the plan
   build; the scheduler released its pages in that round and emits nothing
   further for it, so the request finishes toward the client with a capacity
   abort and no event flows back -- the one request finish that is not a
   forward result), record metrics, run the DP agreement, and gather the
   batch's per-round inputs (n-gram windows, the logprob plan, history seeds,
   grammar advances), committing the queue first if the dispatch depends on
   a pending commit (Principle 4).
4. **One `DataPlane::execute(plan, dispatch)` per round** that has device
   work. The pump turns it into one `DeviceHandle.execute(plan, planned)`,
   which submits in an order that is itself a correctness contract for
   same-round page reuse. The handle adapts the plan's cache ops once
   (`cache_ops_from_plan`), starts the L3 prefetches on the executor's CPU
   lane (`PrefetchOp`: no stream, no Device page, nothing on the FIFO) and
   submits, on the FIFO: the Host-bound stores first --
   `WriteBackOp` and `SnapshotOp`, a retraction image's two legs (the
   stream-ordered L2 write-back of the victim's hash-complete pages and the
   snapshot store of its tail pages and slot state) must read the reused
   pages' old bytes, so they ride the write stream consecutively and ONE
   completion event fences the forward thread's default stream here, before
   anything else of the plan is enqueued; an ordinary store's sources are
   pinned by the scheduler until the ACK, so its copy rides the write stream
   and fences nothing -- then the plan's Device-to-Device block copies
   (`pages_to_copy`: a prefix hit's private copy of the block holding its
   boundary, an owner's move off the block it published at its prompt end;
   the copy waits on the execution stream and precedes the zeroing because
   a source may be re-granted and zeroed by this very plan -- or re-granted
   as another copy's destination, which the arena then writes only after the
   pairs reading it; a destination is never zeroed; no op id, no ACK), then
   page zeroing (the new owner's sanitization), then the Device-bound copies
   -- `LoadBackOp` and `RestoreOp`, ordered behind the zeroing: a load-back
   lands on zeroed pages and the forward reads it layer by layer behind the
   layerwise fences, so the loads start first on the load stream; a restore
   queues behind them, overwrites its destination blocks whole (the
   scheduler lists none of them for zeroing), reads both Host tiers under one
   event and arms no layerwise fence, its request being unschedulable until
   the ACK -- then the plan's remote streams to the transfer peer (a D-node
   remote prefill waits on the zeroing fence inside its submission, which
   the FIFO orders after the store fence), then the plan's batch to the
   model. A store names the victim's request-pool slot, which the scheduler
   freed in the same plan build: its next owner first writes it in a
   forward, and every forward is enqueued behind the store fence, so the
   slot-state export reads the victim's bytes; a restore names the resumed
   request's new slot. The rebalance controller notes whether this rank
   forwarded, and so does the pump's garbage-collection hook. Then commit
   from the queue head down to the effective depth (a pipeline's stages
   adopting the last stage's results first) and poll the PD transfers.
5. **Advance the scheduler (tail)** with the round's `events`, publish KV
   events (once), release the data plane's state for the requests that
   ended, send the round's outputs with the scheduler's load, resolve any
   pending pause or release drain, and send the control replies.

## Checklist for extending the loop

* New logic that reacts to scheduler/transfer/cache progress: put it in the
  matching `ts-control` module (or add one), return events, and apply them at
  an existing advance point.
* New device interaction: if it is startup-only, inside `build_device_side`;
  if the running loop needs it, a named `DeviceHandle` operation behind a
  named `DataPlane` operation.
* New reason a dispatch cannot overlap a pending commit: add it beside the
  others in `plan_and_dispatch`.
* New per-round work: a single call at a fixed position in the round
  (rank-identical across ranks if it contains collectives), not a branch of
  feature code.
* Never advance the scheduler or publish KV events from a helper or
  subsystem module.
* Never issue CUDA work from the pump, or hand the control plane something
  that can.
* Never synchronize with the device from the data plane's per-round path;
  run with `TOKENSPEED_DATA_PLANE_SYNC_DEBUG=error` while developing on it.
* Never let a kernel's compile key follow the batch shape; run with
  `TOKENSPEED_JIT_COMPILE_CHECK=error` while developing on the serving path.
* A collective a rank enters must be entered by every rank of its group in
  the same round, decided from rank-identical state (counts the ranks keep
  the same, agreed flags), never from a rank's own view; a rank-local error
  inside such a sequence goes into the agreement instead of out of the round.
