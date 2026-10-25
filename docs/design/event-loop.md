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
| control (`ts-host` main) | `ts_control::event_loop::EventLoop`: the frontend wire, the collectives between ranks, the C++ scheduler, commit post-processing, control operations; the data plane's completion queries | the scheduler, every request's state, the collective mesh; the GIL only during a completion query |
| forward (`ts-forward`) | the Python `HostDataPlane`'s operations, one at a time in queue order: everything that touches CUDA | the `DeviceHandle`, behind it the executor, models, KV pools, CUDA graphs; takes the GIL per operation |

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
every rank stops after the same round; a frontend that is gone (its input
socket closed, which does not reconnect) stops the engine the same way. One
rank is the same path with groups of one.

The launcher (`entrypoints/engine.py`) starts the ranks through
`start_rust_host`, whatever the deployment: the in-process `Engine`, the
gRPC servicer behind `ts serve`, and `ts serve --headless` for an external
frontend. Each rank dials the frontend's handshake socket; the in-process
frontend is `tokenspeed_engine.Frontend` (`ts-py` over `ts-client`) under
`AsyncLLM`. An EPD encode node's ranks run the host's encode mode instead
(`ts-control`'s `encode` loop over `epd/encode_host.py`): no scheduler, no
event loop. Their first rank binds the request channel its frontend pushes
encode requests to; headless, that is the address an engine rank would dial.

## Principle 1: the loop runs no GPU work, and cannot reach any

The event loop is the **control plane**. The forward thread is the **data
plane**: one thread per rank, FIFO, everything that touches CUDA. A
control-plane round is microseconds, so the cross-rank collectives that keep
the redundant schedulers aligned always find every rank promptly, however
deep the GPUs are in queued work — a stage's launch-queue backpressure stalls
only its own forward thread, never the round.

The control thread cannot run device work by construction: it is Rust and
holds no torch object. Its only way to the device is the `DataPlane` trait
(`event_loop.rs`), which `ts-host`'s `PythonDataPlane` (`data_plane.rs`)
implements over the Python `HostDataPlane` (`engine/rust_host.py`): every
operation that issues or orders device work, or changes the data plane's
state, is queued for the forward thread, and fire-and-forget unless the loop
needs its answer (a DP idle forward, the low-rate control operations), so a
slow forward thread delays nothing but its own queue. The completion
queries — a launched forward's results and the cache acknowledgments — read
what landed and touch no stream, and the control thread calls them itself,
taking the GIL for the call: queued,
the round head would wait for every forward still being launched. A commit
waits for the forward thread to have launched its forward, then on the
forward's copy event, which releases the GIL while it waits. Taking the GIL
can wait for the forward thread's Python, up to the interpreter's switch
interval; its launches and NCCL waits release it. A round with nothing for
the device (no batch, no cache op, no remote stream: `plan.needs_device()`)
queues nothing: a loop waiting on a transfer would otherwise hand the forward
thread an empty plan per round, each costing a turn of the GIL.

The control thread's Python calls run under a thread-local dispatch guard
(`maybe_control_plane_guard` in `device.py`, on by default), entered while
it serves: any CUDA tensor op from it raises. Event waits — the inbound
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
| `DeviceHandle` | the running handle: the complete list of what the control plane may ask of the device side | held by the data plane (`HostDataPlane`) only |
| `transfer` | the PD peer's settings and device fences (`pd/host_peer.py`): what the Rust transfer peer starts from, and what hands it each plan's remote stream; None outside PD | handed to `ts-pd` at startup, held by the handle |
| `host_cache` | shared Host resources, the CPU cache layout and the resolved attention choices; no Device buffers or executor | consumed at startup, by the L3 settings (`cache/l3/factory.py`) |
| `encoder_model_facts` | a callable resolving the encoder facts EPD admission needs (raises on text-only) | consumed at startup, past the EPD gate |

`HostBootstrap` (`rust_host.py`) turns the build into what the Rust side
takes: the scheduler, a plain `HostConfig` (with L3, the settings the host
starts its L3 executor from), and the data plane holding the handle. Nothing
else of the build crosses.

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
it constructs the transfer peer from its prepared components and startup
arguments, and reads the engine's role off it once, at construction.

The transfer itself is the host's (`ts-pd`): the routes between unequal rank
sets, the block manifests read off each plan's block tables, the layerwise
chunks and the copies are planned there, over the cache transfer contract the
data plane built at startup. The forward thread stages each plan's remote
stream with the peer before it hands the plan to the data plane; the data
plane hands the stream over (`execute`) once its device prerequisites are
met -- on a Prefill rank behind the drafter's pending round, whose KV goes
too; on a Decode rank behind the plan's page zeroing, as the Prefill ranks'
writes are not ordered with the device's -- and the forward thread checks it
did. On a Prefill rank streaming layerwise, the forward thread arms the
forward's chunks when it hands the plan over and counts the producer steps;
the attention backends only tick the device counter (`StepCounter`).

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
and page zeroing because the forward thread hands each remote stream to the
peer in the plan's own order, behind the forwards and the zeroing. Its control face — request
registration and abort, the per-round agreement on transfer progress — is the
loop's `TransferPeer` (`pd.rs`), where it feeds Principle 3's tail advance.

Consequences:

* The control plane cannot **name** a model runner, backend or KV pool, so it
  cannot pass one implicitly or mutate one a forward is still using.
* A new device interaction that runs once at startup belongs **inside the
  builder**; it belongs on `DeviceHandle` (and the `DataPlane` trait) only
  when the running loop genuinely needs it — the second widens what the loop
  can do to the GPU mid-flight. Hand over the capability, never the object.
* Every method on the handle is a **named operation**; a generic "run this
  closure" slot is the hole this whole design closes. The multimodal feature
  lifecycle needs none: EPD admission and the release of a request's SHM
  features are data-plane operations themselves, run on the forward thread.
  L3 is resolved separately at startup by `cache/l3/factory.py`, using
  the builder's Host resource and CPU-only payload facts. `DeviceBuild` does
  not create an L3 schema, namespace, store or workers, and `DeviceSpecs`
  carries no L3 settings. The factory derives the schema and deployment
  identity from neutral Host resources and model configuration; see
  [asynchronous L3](async-l3-cache.md#ownership). The host runs the
  resulting Host-only executor (`ts-l3`) in Rust; its workers cannot reach
  the device handle or the interpreter. The control plane submits the
  storage subset of `plan.cache` to it independently from device work, and
  its completion facts enter the head feedback point. The
  retraction snapshot pool adds no operation at all: its stores ride
  `execute` with the write-backs, its restores with the load-backs, its ACKs
  `poll_cache_results`, and `shutdown_cache` closes it with L2 -- one
  `HostCacheExecutor` behind the handle, two Host buffers inside it. The
  plan's cache ops are adapted once, by the data plane, through
  `scheduler_utils.cache_ops_from_plan`: the L2 batches (`Cache.WriteBackOp`,
  `Cache.LoadBackOp`) pass through, and the scheduler's batched
  `Cache.SnapshotOp` / `Cache.RestoreOp` become one per-request `SnapshotOp`
  / `RestoreOp` (`cache/transfer/ops.py`) each -- the executor's unit of
  work, one slot exported or imported, one ACK. The control plane counts the
  plan's Device ops from the same plan (`CacheOps::dispatched`) and
  `DeviceHandle.execute` consumes the same Device-op tuple, so what is
  counted is what is submitted; the plan's per-request snapshot store count,
  one per retraction victim, is the round's preemption count, exported as
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
* On the per-round path only commit waits: for the forward thread to have
  launched the forward, then on its copy event (D2H landed). Dispatch never
  waits — it queues model forwards and the peer's remote prefills/decodes
  alike, which is what keeps a backpressured stage off the control plane. A
  raise is for broken invariants only: a transfer worker or another Prefill
  rank can fail a request's transfer while its submission waits, and that
  submission then does nothing for the request, whose failure arrives as its
  transfer state. The loop waits for the DP idle forward and the low-rate
  control operations (memory release and resume, the RL weight sync, the
  expert rebalance's steps, profiling), whose callers cannot proceed without
  the result; on the forward thread, the landing of a completed remote
  prefill and the KV repair after a wake complete before anything queued
  behind them runs. A new blocking operation on the per-round path is a bug.
* Keep per-forward state resets asynchronous on the execution stream too.
  Use device-side scalar fills for indexed flags: assigning a Python scalar
  through tensor indexing can stage a CPU tensor and introduce a hidden
  synchronous H2D copy, blocking further forward launches.

An EPD prefill node's first stage receives each multimodal item's embedding
through the host's embedding receiver (`ts-pd`'s `epd::receiver`): it looks
up and registers with the encode ranks, leases each item a slot of the
receive pools and tells the encode rank where to write, and every round
reports each staged request's state, which the replica MIN-reduces, so a
request is admitted, or failed, on every rank in the same round. Later
pipeline stages hold no receiver and agree with whatever the first decides;
decode builds none. The data plane keeps the device half. It allocates the
pools (one device buffer, after the cache is sized; the host registers it
once). `epd_publish` copies the landed rows out of their slots onto the
items before any forward of the request and, with row sharding, rebuilds
each image's other rows with NCCL broadcasts — on the forward thread, the
same issuing thread as the model's collectives, which keeps the cross-rank
launch order. `epd_released`, a completion query, reports the slots whose
copies finished, so the receiver leases them again.

An EPD encode rank's loop is its own, with the same split. Each round the
attention-TP group's first rank relays the frontend's requests to the
others, in one exchange that also agrees on the send-ring slots free on
every rank, so every rank forms the same vision-tower batches (the tower is
TP-sharded); a room's send concludes on each rank at its own time and only
frees its slot. The data plane admits requests on the forward thread: the
ranks open every pixel-SHM segment of the round behind a CPU-group barrier
before any consumes one, since consuming unlinks it on cache hits and misses
alike. It embeds each batch and copies every item's rows into its send slot;
a completion query on the control thread reports the finished copies, whose
rows the sender (`ts-pd`'s `epd::sender`) then writes to the prefill ranks.

The symmetric rule holds on the data plane: **the forward thread never
synchronizes with the device on the per-round path.** Every host
synchronization it performs — `.cpu()`, `.item()`, `.tolist()`,
`bool(tensor)`, `nonzero`, a copy from or to pageable host memory,
`stream.synchronize()` — waits for the whole stream, and the stream holds
the step in flight, so the next step's prologue and graph launch slip
behind the current step's completion and `in_flight_depth` degrades to 0
however it is configured. Results cross back through pinned non-blocking
copies and an event the commit waits on. Torch's sync-debug mode enforces this
rule, and `HostBootstrap.enter_serving` arms it once the rank finished
startup and the frontend handshake (weight loading, tuning, capture, the
transfer and L2 builders all synchronize on purpose):
`TOKENSPEED_DATA_PLANE_SYNC_DEBUG=warn` reports each offending site with its
Python location, `error` raises there. CI runs the serving paths with
`error`; the commit's event wait and the non-blocking copies pass
unflagged, so a report is always a real stall.

One named exception: a drafter may leave its round pending after the forward
(`BaseDrafter.finish_round`) and finish it in the next forward's `settle`,
before `load_round_inputs` (`complete_round`); every other forward-thread task
finishes it first. A decode settles after the target's decode metadata and
before the draft's, so what a forward queues before its settle runs beside
the round and must leave alone what the round uses: `future_*`, the draft KV
cache and decode metadata, the drafter's buffers, the kernel workspace and
collective state. EAGLE-Harness chains a round's captured graphs on the GPU,
behind the decode graph (`BaseDrafter.replay_decode_graph`), so what the
forward queues after that graph (the verified-state commit, the outputs) runs
beside the round too; an eager round polls a pinned flag to launch each depth.

A JIT compilation on the forward thread is the same kind of stall, only
longer: the round waits 100 ms to seconds while a kernel compiles for a new
compile-cache key. Startup compiles on purpose (graph capture, tuning), so
the bootstrap installs `tokenspeed_kernel.compile_monitor` before building
anything and `enter_serving` marks the end of startup right after arming the
sync-debug mode; an encode rank marks it once it reported ready. From then
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
CPython does not start full collections by itself. The data plane's
`GcHooks` runs one on the forward thread every `FULL_COLLECTION_INTERVAL`
forwarded rounds, a count that is the
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

Information crosses to the data plane **only** as a queued operation's
arguments: plain values (lists, bytes, the scheduler's plan moved into a
Python object), from which the forward thread builds what a forward reads.
What a launched forward reads freezes: no attribute rebinding, no in-place
edit, no releasing a resource it holds while it may still run. Results cross
back **only** as the launched forward's result, which the commit reads once
its copy event landed, as plain lists built from the landed host copies.

The plan's cache ops follow the same capture rule: the data plane adapts
them once into immutable Device and storage tuples, and
`DeviceHandle.execute` runs copies from the supplied Device tuple. The L3
executor's workers receive only Host jobs, with no GPU stream or
forward-thread access. Their results return to the control plane through
the executor's completion poll.

`execution/device.py` states this in full, including the registered
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
| `l3` | the replica's agreement on its storage executors' reports: the clock, the forward and storage permissions, the queue pressure, each storage operation's outcome |
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
  `WriteBackDone`, `LoadBackDone`, `SnapshotDone`, `RestoreDone`, each
  replica-intersected), together with the L3 agreement's storage facts
  (`L3::agree`: the round's tick, the cancellations, the storage operations
  every owner completed), through one `advance_head`. These must advance
  *before* `next_execution_plan`; otherwise, cache-gated admissions -- a
  request whose stored prefix landed in Host, a suspended request's restore,
  which waits for both of its store ACKs, then its own -- are delayed by a
  full round.
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
shared multimodal features — is released through the forward thread's queue,
behind the forwards that read it.

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
round commits what is dispatched and still joins the attention-DP agreement
(an idle forward when a peer runs one, but not while memory is released: the
weights are unmapped; idle forwards count toward the expert rebalance like
real ones, so the ranks' rebalances stay in step). It plans like any round,
but the scheduler, which learns the forward permission from the round's
storage tick, plans only cache and storage work: write-backs and storage
cancels still drain. With L3 every round carries a tick; without, the loop
sends one when a freeze starts and when it ends. A memory release drains like
`wait`, then frees its regions — one release/resume pair per tag, whose
data-plane half (quiescing copies, the transfer engine's registration, the KV
repair, NCCL suspend) runs on the forward thread — and freezes until they are
resumed; a further release drains frozen. A release is refused under a
`keep` pause (its drain would run the frozen requests, and the wake would end
the pause), as a pause is under a release. A drain (`abort`, `wait`, a
release) cancels the admitted requests the scheduler does not hold yet, which
end toward their clients: those compiling their grammars, and those waiting
for their images' embeddings, which end once their receives settle — the
drain waits for those.
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
they hold. A rank-local ACK would publish a Host block on one mirrored
scheduler while a peer still has the op pending, and a KVP rank that copied a
different owned subset of an image completes at a different time. An L3 PUT
has its own completion and does not delay the D2H ACK. Every rank stays in
every gather, even when its own part is empty. The poll runs these
collectives only while a cache op is open: every rank counts the same plans
and advances the same intersections, so the open count is the same
everywhere and an idle replica polls nothing.

**L3** (`l3`, `--kvstore-storage-backend`). The C++ scheduler owns each
request's storage transaction: it plans QUERY, GET, PUT and CANCEL
operations in `plan.cache` ([asynchronous L3](async-l3-cache.md)), which the
control plane submits to the host's Host-only L3 executor (`ts-l3`) when it
dispatches the plan, apart from device work; no L3 path takes the GIL. A
QUERY and
GET run before admission, only with a storage backend: a request waiting for
its preparation is skipped before local prefix hashing, so an idle round does
not rehash a long prompt while native I/O is outstanding. Every round with
L3, each member's executor report (its clock, queue pressure, completions,
and an error, if its poll was invalid) rides the round's relay: one
all-gather carries the first member's messages and every member's report, so
the replica runs one exchange a round, not two. At the head the replica
agrees on the reports (`L3::agree`): rank 0's clock and the worst pressure
become the round's tick, with the forward and storage permissions the
control state gives, which every member holds alike. A storage operation
completes once every owner reported it: per key, success and native
retirement on every owner, an object failure on any. A rank's invalid report
reaches every rank in the same exchange, and every rank fails after
gathering it, so no rank strands a peer. A frozen round plans too, with
forwards and storage forbidden, so detaches and cancels still drain. An
abort of a request still preparing its storage rides the next head: one that
holds no pages yet finishes there (the forward path would never reap it),
any other detaches from its storage work. Without L3 the round runs none of
this. L3 requires PP=1; the D role receives its missing cache from its
prefill peer.

**PD** (`pd`, `ts-pd`). Requests register with the transfer peer at
admission, and every round the replica agrees on which transfers changed
state; their scheduler events join the round's tail feedback, and a failed
transfer aborts its request on every rank together. A peer keeps state only
for the rooms of requests the loop tracks (a sleep waits for those alone),
with one exception on the prefill side: a room the decode node pre-allocated
for before its request registered here, or one this rank failed first (an
EPD embedding timeout), waits unclaimed; one still unclaimed after the
bootstrap timeout is dropped, and the decode ranks that pre-allocated for it
are told it failed. A decode room whose KV lands without the request's first
token fails: the first decode feeds on it.

**EPD prefill** (`--disaggregation-mode prefill` with encode nodes). A
request whose images an encode node embeds is staged at admission; each
round the replica MIN-reduces which staged requests' embeddings landed on
every rank, submits those, and fails those whose receive failed or timed out
on any. The receives settle while paused too: one that lands then is
withheld, like any admission.

**Grammars** (`grammar`, `--grammar-backend xgrammar`). A constrained
request waits in the loop's grammar queue until its grammar compiled (off the
round, on compiler threads) on every replica rank, and fails if it timed out
on any or does not compile; every rank admits the same ones in the same
round. The matchers live in Rust (`ts-grammar`); the data plane fills
bitmasks through their handles, with one C++ fill (`grammar_host.Filler`)
on both paths: the capturable path's host node runs it inside the forward
and advances the matchers there; the eager path runs it before the forward
and the commit advances them.

## Anatomy of a round

For orientation, one `EventLoop::round`:

1. The frontend's owner receives (waiting briefly only when the whole engine
   is idle) and relays the round's messages to the replica, with its stop
   request; with L3 every member's storage executor report rides the same
   exchange. Admit them in arrival order — a control operation and the
   requests before it keep their order — then apply the pause edges, admit
   the grammar queue and the EPD staged requests that are ready, and run the
   same-round gate.
2. Agree the L3 reports into the round's storage tick (without L3, a tick
   only when a freeze starts or ends); while a cache op is open, poll the
   data plane's acknowledgments and intersect them; then **advance the
   scheduler (head)** with both, so this round's plan sees them.
3. Frozen? Commit what is dispatched, plan (the tick forbade forwards, so
   the plan carries cache and storage work only) and dispatch that work,
   and join the attention-DP agreement.
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
   work, after the control thread submitted the plan's storage operations to
   the L3 executor (Host workers: no stream, no Device page). The forward
   thread adapts the plan's cache ops once (`cache_ops_from_plan`), sets the
   storage subset aside and turns the rest into one
   `DeviceHandle.execute(plan, planned, cache_ops=...)`, which submits in an
   order that is itself a correctness contract for same-round page reuse:
   the Host-bound stores first --
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
   forwarded, and so does the data plane's garbage-collection hook. Then commit
   from the queue head down to the effective depth (a pipeline's stages
   adopting the last stage's results first) and poll the PD transfers.
5. **Advance the scheduler (tail)** with the round's `events`, publish KV
   events (once), release the data plane's state for the requests that
   ended, send the round's outputs with the scheduler's load, resolve any
   pending pause or release drain, and send the control replies.

## Startup process ownership

The launcher starts every local rank before waiting for readiness, and
watches their ready channels and process exits together. A rank that fails
startup tells the launcher and exits; it does not finalize its interpreter,
so a lingering Python thread cannot keep it alive. EOF remains a failure
until every rank has reported, even if that rank reported earlier: peers can
still be blocked in startup collectives.

Readiness detection only reports failure. It never joins or terminates
ranks, and introduces no shutdown deadline. The owning launcher stops the
process tree. Child-failure signals unwind the launcher with a nonzero exit
status. Failure, termination and constructor-error cleanup share one
process-local shutdown latch: once cleanup starts, later rank notifications
cannot interrupt it. Signal callbacks do not log or enumerate processes;
cleanup runs outside the callback.

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
* Never issue CUDA work from the control thread, or hand it something that
  can.
* Never synchronize with the device from the data plane's per-round path;
  run with `TOKENSPEED_DATA_PLANE_SYNC_DEBUG=error` while developing on it.
* Never let a kernel's compile key follow the batch shape; run with
  `TOKENSPEED_JIT_COMPILE_CHECK=error` while developing on the serving path.
* A collective a rank enters must be entered by every rank of its group in
  the same round, decided from rank-identical state (counts the ranks keep
  the same, agreed flags), never from a rank's own view; a rank-local error
  inside such a sequence goes into the agreement instead of out of the round.
