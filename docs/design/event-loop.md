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
| control (`ts-host` main) | `ts_control::event_loop::EventLoop`: the frontend wire, the collectives between ranks, the C++ scheduler, commit post-processing, control operations; the data plane's completion queries and, with a shared L2 pool, the round's blocking lookup at the pool's owner | the scheduler, every request's state, the collective mesh; the GIL only during a completion query |
| forward (`ts-forward`) | the Python `HostDataPlane`'s operations, one at a time in queue order: everything that touches CUDA | the `DeviceHandle`, behind it the executor, models, KV pools, CUDA graphs; takes the GIL per operation |

Peripheral threads stay off the round: the wire's Tokio runtime
(`ts_control::transport`, which also encodes each round's outputs), the PD
transfer peer's workers (`ts-pd`), the grammar compilers, the KV event
publisher, the expert-rebalance solver. With a shared L2 pool, so do the
pool client's threads (its writes, its notices, its loads' reads).

Request traces use a bounded, lossy background queue. Plugins register a provider
with `register_trace_provider`, selected explicitly by `--trace-provider NAME`.
The provider maps a request ID to `(trace_id, optional_parent_span_id)` or `None`;
ID policies belong to plugins. It runs only on the export thread and must be
short and nonblocking, without I/O or GPU access. Missing or invalid context
preserves native IDs; span lifecycle and RID sampling remain unchanged.

With `--enable-async-logging`, the Rust tracing writer and Python root logger
also run off the serving threads. Each has a bounded, lossy queue: a slow
output stream must not introduce I/O backpressure into a round. The host
owns Python log cleanup explicitly after serving returns, including errors
and encode ranks. Returning from Rust does not finalize Python, so an
`atexit` hook alone cannot drain those records. Cleanup is bounded because
an output stream can remain stalled.

[Cache observations](cache-observations.md) use the same nonblocking writer
when explicitly enabled. Native callbacks report immutable ownership metadata;
the ordinary ACK poll is followed by a drain of completed CPU copy metadata.
Neither sideband feeds the scheduler or establishes replica-wide completion.
The protocol makes rank scope, operation/admission joins, and diagnostic loss
explicit; incomplete or lossy traces cannot qualify recovery.

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
`start_rust_host`, whatever the deployment: `ts serve` (SMG's gateway on its
direct ZMQ lane, which binds the frontend sockets) and `ts serve --headless`
for an external frontend such as Dynamo's. Each rank dials the frontend's
handshake socket. An EPD encode node's ranks run the host's encode mode instead
(`ts-control`'s `encode` loop over `epd/encode_host.py`): no scheduler, no
event loop. Their first rank dials the frontend the same way and takes
encode requests.

The launcher snapshots each rank's environment inside its start context so
allocator preload hooks reach the child along with the native logging defaults.

Runtime plugins may register `register_post_model_init_hook(name, hook)` through
`tokenspeed.runtime.plugins.registry`. Each language-model engine rank invokes
an immutable snapshot of these synchronous callbacks in name order, once during startup,
after target and local draft loading, multimodal preparation and communication
preparation, and before the first cache profile or allocation. Cache rebuilds
do not invoke them again. Encode-only engines do not run these hooks.
`PostModelInitContext` exposes launch parameters, model configurations, mapping,
the local device and the world CPU process group; it exposes no device model. The group includes pipeline stages without
a draft. Hooks own their synchronization, must run in the same order on every
rank, and propagate failures to startup. Registration joins the plugin loader's
normal recording and rollback.

The host reads its tokenizer and preserves safetensors shapes from the selected
files while loading weights, before this boundary. Dummy/config-only startup
snapshots already-local safetensors headers if present, without downloading a
checkpoint for this metadata. Unreadable optional dummy headers are warned
about and leave an empty snapshot, so an incomplete checkpoint cannot prevent
a weights-free startup. Later disk updates validate against the saved
shapes rather than reopening the initial source. Rank-local state and custom
loaders must supply canonical checkpoint shapes to support this comparison;
their storage layouts need not match a regular checkpoint. Without a snapshot,
new shards are still validated before loading. Explicit reload requests require
the new checkpoint they name.

Plugins must prepare their own metadata before retiring a source. The world
CPU group coordinates one engine only: a source shared by independently
starting engines (including prefill/decode or external frontends and encode
services) requires cross-engine coordination owned by the plugin, or a separate
copy for each engine. A barrier on this group alone cannot protect that source.
Hooks must not retain GPU allocations after startup except in the memory
saver's CPU-backed named regions, preserving zero occupancy on weights release
and restoration on resume.

What an engine registers with tells its frontend what it serves: the
request types it takes (an encode server takes encode requests only), the
field order of the positional structs on its wire, its role and, for a
prefill or encode server, where its bootstrap server listens, and the
checkpoint facts a frontend computes multimodal positions from. A frontend
may name a control operation by method (a control call, as SMG's gateway
calls every scheduler it drives); the transport turns it into the engine's
own operation and answers it under its call id once the operation's reply
comes back, so the loop sees only its own operations.

Scheduler load travels on the same output wire, including an empty batch at
startup and at most once per second without token outputs. Reports carry
running and waiting counts and resident, active, and total LCM blocks; the
active count is at most 100 ms old (see the page gauges below). The
native frontend timestamps receipt and exposes a complete per-rank snapshot
set to the gateway only while every report is less than three seconds old.
The gateway converts these allocator counts to capacity ratios; it must not
interpret an LCM block as a fixed-size token page.

EPD encode-mode metric ownership and units are documented in
[encoder-metrics.md](encoder-metrics.md).

## Request identity and submission failures

The wire identifies a request and its output stream by `rid` alone. A new
request, including a retry after abort, should use a fresh id. Before ingress
can publish a refusal or register data-plane state, the loop discards repeated
ids in the same received round and ids still owned by the output processor.
The original request keeps its state and stream; a warning records the
duplicate. Sending a final error for that duplicate would end the original
stream, since the wire cannot distinguish the two submissions.

Output completion and scheduler retirement are separate fences. After the
output processor forgets a request, the scheduler may still hold it while
forward results drain or until the next plan removes its finished entry.
An id in that interval receives a 409 refusal before data-plane admission.
The check uses the scheduler's existing request lookup; it does not reclaim
an id ahead of its in-flight work. Once both owners have released an id, this
admission guard allows it again.

The native frontend's `abort(rid)` and `cancel(rid)` send the ordinary wire
abort. After a successful send, the frontend detaches the original local
stream if it is still attached and releases its in-flight count. After any
queued outputs, `abort` delivers an empty token chunk with `finish_reason`
set to `"abort"` and the latest reported token counts and verify rounds.
Before the first output, it uses the submitted input ID count for prompt
usage. `cancel` delivers `StreamError::Cancelled`, exposed by Python's
`next_outputs` as a `(rid, RequestCancelled)` exception value. The engine owns
its own load reports and releases slots, cache, and transfer resources at
retirement.

`abort` returns `true` when it sends the wire abort. `cancel` returns `true`
when it detaches the local stream. A finish, stream error or another
cancellation that wins the race keeps its result. A failed send or a dropped
cancellation future leaves any still-active stream attached for retry, even
if the engine already received the abort. Stream drop and send cleanup compare
the original local admission's channel before removing an entry. Fresh request
ids keep delayed wire output or abort messages from reaching a later request.

On disaggregated engines, `bootstrap_room` carries the same one-active-request
rule across distinct request IDs. Peer state is keyed by room, so every new
request — including fan-out children and retries — must use a fresh room. At
admission, before data-plane registration or scheduling, the loop refuses with
409 any request whose room is already reserved by an unscheduled request (same
round, grammar queue, EPD staging, or pause withholding) or tracked by
`TransferPeer` (including draining rooms awaiting `Stopped`). Both checks read
only replica-identical loop state, never worker-thread state such as a prefill
peer's cancellation tombstones, so every replica member makes the same decision;
reusing a previously failed or cancelled room after retirement still fails in
the prefill peer under its tombstone contract. `TransferPeer::register` enforces
the same uniqueness invariant at the peer boundary, and `abort_room` ignores
rooms owned by another active request.

Each accepted request gets a monotonically increasing admission identity
before data-plane registration. The output processor allocates it once;
the Python request record, planned forwards and snapshot/restore operations
carry that same value. External `rid` remains the wire and scheduler lookup
key. Reusing it after retirement creates a new admission, while chunks,
decode steps and retraction keep the existing admission identity. Slot-state
owners compare admissions so a recycled row cannot retain a previous
request's stop criteria or sampling state. Identity allocation is independent
of the sampling seed and does not retain a history of external request IDs.

Cache-op adaptation resolves the admission while the host still holds the
request record and stores it in the immutable operation. No asynchronous
completion resolves a potentially recycled `rid` to infer slot ownership.
The identity is local to the engine lifetime, like its slot-state images;
it is not a cross-engine or persistent-cache key.

The Rust/C++ submission bridge returns `Result`: native validation exceptions
must never escape a `noexcept` bridge. The scheduler validates a complete
submission batch before inserting requests. If it rejects the batch, the
loop finishes and releases those unscheduled requests with a client error
and continues serving; it starts PD transfers only after submission succeeds.
Already running requests remain intact. Other requests in a rejected batch
also receive its error and can be retried with fresh ids.

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

With a shared L2 pool the control thread also calls the pool's client
(`RankClient`, `ts-control/src/shared_l2/client.rs`), which is Rust like the
L3 executor: no call takes the GIL or touches a stream. The lookup sends one
request to the pool's owner and waits for the reply; the other calls hand
work to the client's threads and read what they finished or where they
failed: the poll once at the head of every round, the failure check once
when a forward has landed. A load from the pool is the forward thread's to
hand over, like a Host L2 load (through the shared L2 handle,
`SharedL2Handle.submit_load`, which only queues it for the client's
loader), and its acknowledgment comes back through the data plane's cache
poll, which on a pool rank reads the loader's finished loads in the host and
takes no GIL (`PythonDataPlane::poll_cache_events`,
`ts-host/src/data_plane.rs`).

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
| `shared_l2_client_config` | the rank's client config for the shared L2 pool (`SharedL2ClientConfig`, `cache/shared_l2/client.py`); None without a pool | consumed at startup, by the scheduler's configuration |
| `encoder_model_facts` | a callable resolving the encoder facts EPD admission needs (raises on text-only) | consumed at startup, past the EPD gate |

`HostBootstrap` (`rust_host.py`) turns the build into what the Rust side
takes: the scheduler, a plain `HostConfig` (with L3, the settings the host
starts its L3 executor from), and the data plane holding the handle. Nothing
else of the build crosses.
A rank with a shared L2 pool starts its client of the pool's owner in the
host while it builds: the host hands `bootstrap` a shared L2 handle
(`SharedL2Handle`, `ts-host/src/shared_l2.rs`), as it hands it the PD and peer
outboxes, the build starts the rank's transfer engine and, once it has
joined the pool, the client through it (`cache/shared_l2/rank.py`), and the
loop takes the client from it.

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

Ordinary communication buffers are prepared for the model's execution groups
after model-specific communication preparation and before cache memory
profiling. Large warmup batches can select RCCL while a later small prefix hit
selects Iris, so warmup traversal alone cannot initialize every serving route.
Preparation reuses model-owned capacities and follows the mapping's group-role
order on every rank, so overlapping groups retain symmetric heap offsets.
Pipeline, weight-sharing and named groups keep their separate communication
lifecycles.

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
Each armed producer step gets one tick, after all of its cache writes have
been queued on the producer stream. Splitting a layer's DSA index-cache write
into kernel-sized chunks does not add producer steps: only the final chunk
publishes cache readiness.

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
  One is registered here: with a shared L2 pool, a round makes one blocking
  lookup request before its plan when prompts wait to be looked up
  (`shared_l2_round`). The wait is bounded by the pool client's request
  timeout, and a request that fails or is not answered in time stops the
  engine (the *Shared L2 pool* entry of *Subsystems in detail*).
* Keep per-forward state resets asynchronous on the execution stream too.
  Use device-side scalar fills for indexed flags: assigning a Python scalar
  through tensor indexing can stage a CPU tensor and introduce a hidden
  synchronous H2D copy, blocking further forward launches.

An external-encoder receiver's first stage receives each multimodal item's embedding
through the host's embedding receiver (`ts-pd`'s `epd::receiver`): it looks
up and registers with the encode ranks, leases each item a slot of the
receive pools and tells the encode rank where to write, and every round
reports each staged request's state, which the replica MIN-reduces, so a
request is admitted, or failed, on every rank in the same round. Later
pipeline stages hold no receiver and agree with whatever the first decides;
decode builds none. Aggregated serving opts into the same receive admission with
--enable-external-encoder: it owns a standalone Mooncake transport and remains
a Plain execution role, with no P/D KV peer. A PD Prefill receiver shares its
KV peer transport. The receive pool budget uses the same capability gate as
allocation. The data plane keeps the device half. It allocates the
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
copies and an event the commit waits on; host-built metadata goes the other
way through `host_to_device` (`utils/host_sync.py`), a non-blocking copy from
contiguous pinned memory. Torch's sync-debug mode enforces this
rule, and `HostBootstrap.enter_serving` always arms it once the rank finished
startup and the frontend handshake (weight loading, tuning, capture, the
transfer and L2 builders all synchronize on purpose); there is no switch to
leave it off. `TOKENSPEED_DATA_PLANE_SYNC_DEBUG=warn` (the default) reports
each offending site with its Python location, `error` raises there. CI
serves every `server_smoke`, `eval`, and `perf` task with `error`, which a
task cannot relax; the commit's event wait and the non-blocking copies pass
unflagged, so a report is always a real stall. A deliberate wait off the
per-round path is wrapped in `allow_host_sync` (`utils/host_sync.py`).

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
on the monitor logs every Triton and CuTe DSL compilation with its duration
and cause and reports by name any compile-time kernel parameter that keeps
taking new values from one call site (a per-batch value passed as
`tl.constexpr` or as a static CuTe DSL dimension), a call site that keeps
recompiling specializations the kernel already has, and a direct `@cute.jit`
call, which traces the kernel in Python on every call; each commit reports
the compilation counts and durations, which the control plane exports as
`tokenspeed:jit_serving_compiles` / `tokenspeed:jit_serving_compile_seconds`.
`TOKENSPEED_JIT_COMPILE_CHECK=error` raises there instead, and CI serving
jobs run with it.
The same mark closes the kernel package's compile switch
(`compile_monitor.is_serving()`), installed monitor or not. A kernel whose
library compiles once per batch shape with no key to bucket -- FlashInfer's
joint BF16 GEMM and the ll_bf16 router's dot-product kernel compile once per
exact row count -- runs only before it:
graph capture records it, and eager calls after it take a GEMM that never
compiles instead of compiling on the forward thread. The pinned FlashInfer
TGV tiles for BF16 decode GEMMs on small weights run only before it too,
though they compile once per tile, not per shape: their route was measured
only inside captured graphs, so eager calls after the mark keep cuBLASLt
(`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/README.md`).

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
forwarded rounds, a count that is the same on every rank, so the ranks
collect in the same round and each collection walks only what serving keeps
alive. An EPD encode rank does the same and counts its vision-tower batches,
which every rank of its attention-TP group forms alike. The young
generations still collect automatically.

Graph capture runs before that mark and collects for its own reason: a
collection inside a capture would run CUDA graph destructors on a capturing
stream, so each prefill and drafter capture (`BreakableCapture`) collects on
entry and keeps automatic collection off until it exits. Decode captures
(`torch.cuda.graph`) collect on entry only under PyTorch's
`force_cudagraph_gc`. When the decode or the prefill graph captures,
`ModelExecutor.capture_graphs` freezes the heap once around all captures, so
those collections, and any automatic one during a decode capture, walk only
what the captures created; an enforce-eager boot runs none of this.
`--enable-cudagraph-gc` keeps the heap unfrozen.

### Profiling the issuing thread

Profiling must include the forward thread that issues device work. Batch and
request annotations describe the immutable submitted work, not its later
completion. Collecting them must not read device tensors or change scheduling
and synchronization behavior.

Device work may outlive its host range. Analysis must correlate launches with
device events within each rank and process rather than clip GPU time to host
timestamps. Profile startup rejects unsupported capture capabilities instead
of silently producing an incomplete trace.

The torch profiler sees only torch ops and Python, so a torch profile also
records the host's own spans (`ts_scheduler::profile`): the round's phases,
the scheduler calls and their C++ phases (`csrc/core/profile_span.h`), the
data-plane calls and their waits for the GIL on the control thread, and each
operation on the forward thread. The profiler merges them into the trace it
exports, on their threads' lanes, so a forward thread's operation encloses
the Python it ran; the C++ scheduler's spans are under the `scheduler`
category, the rest under `host`. Work that overlaps on one thread, as Disk
operations in flight and peer landings do, records on lane rows instead
(`profile::Lanes`): each piece holds the lowest free row of its family until
it ends. The recording follows the torch profiler's window and
costs one atomic load per span outside it; it keeps only the spans that began
inside the window, so an operation open as a by-stage profile switches stages
stays out of the next stage's trace. A trace the merge fails on is written
without the host spans rather than failing the forward thread. A round that receives nothing,
hands the data plane and the L3 executor nothing, and commits, acknowledges,
or answers nothing leaves no spans: a loop waiting on a transfer spins through
such rounds. A storage operation's outcome counts as an acknowledgment.

### The capture contract

Information crosses to the data plane **only** as a queued operation's
arguments: plain values (lists, bytes, the scheduler's plan moved into a
Python object), from which the forward thread builds what a forward reads.
What a launched forward reads freezes: no attribute rebinding, no in-place
edit, no releasing a resource it holds while it may still run. Results cross
back **only** as the launched forward's result, which the commit reads once
its copy event landed, as metadata lists and CPU logprob buffer views.

The plan's cache ops follow the same capture rule: the data plane adapts
them once into immutable Device and storage tuples, and
`DeviceHandle.execute` runs copies from the supplied Device tuple. The L3
executor's workers receive only Host jobs, with no GPU stream or
forward-thread access. Their results return to the control plane through
the executor's completion poll.

`execution/device.py` states this in full, including the registered
exceptions — grammar matchers, whose ownership splits by path and whose
overlap the drain rules in Principle 4 break instead, and multimodal items.

A shared L2 pool adds one exception: the pool page behind each proxy Host
block of an answer reaches the forward thread through the pool client's
state, not as an argument of a queued operation. The control thread hands
the client the map when it applies the answers (`map_proxy_blocks`), before it
queues the plan that may load from those blocks, and the forward thread's
hand-over of that plan's load-backs reads it there (through the shared L2
handle, `SharedL2Handle.submit_load`). The scheduler holds a
proxy block a planned load reads until the load is acknowledged, so no later
answer remaps the block first.

Completion queries export large logprob columns as CPU buffer views, after
the result's copy event has landed. The host validates dtype, rank, shape
and token-ID narrowing, then copies into Rust-owned vectors while Python
buffer guards retain the storage. No borrowed array escapes the query or
outlives the result's reuse lease. Buffer guards release on errors and
unwinding too; no manual export/release pair belongs in the event loop.

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
| `shared_l2` | a shared L2 pool's client: as the loop sees it, with the writes in flight there, and the client itself (the owner's wire, its write and notice threads, its loads); and the pool's owner, which no rank runs (the *Shared L2 pool* entry of *Subsystems in detail*) |
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

The scheduler advances at three explicit points in `EventLoop` — never hidden
inside a subsystem. Helpers RETURN their events, or push them into the round's
`events` list. Reading `EventLoop` must reveal every point where the
scheduler's state advances, and why:

* **Head of the round** — completed Host cache ops (`CacheOps::acknowledged`:
  `WriteBackDone`, `LoadBackDone`, `SnapshotDone`, `RestoreDone`, each
  replica-intersected), together with the L3 agreement's storage facts
  (`L3::agree`: the round's tick, the cancellations, the storage operations
  every owner completed), through one `advance_head`. These must advance
  *before* `next_execution_plan`; otherwise, cache-gated admissions -- a
  request whose stored prefix landed in Host, a suspended request's restore,
  which waits for both of its store ACKs, then its own -- are delayed by a
  full round. With a shared L2 pool a finished write is one more kind of
  acknowledgment in the same batch. Its lookups' answers are applied after
  it, right before the plan, in `shared_l2_round`, which `round_with` calls
  (`apply_shared_l2_answers`, one call). An answer is not an event: the proxy
  blocks it creates go back to the pool's client.
* **Tail of the round** — forward results and PD transfer events, funneled
  through the round's single `events` list, through `advance`; all events
  must advance before the *next* round plans.
* **Peer holder synchronization** — in coordinated mode, after a lock-release
  window where `ts-peer-servicer` may have answered `Lookup`/`Pin`/`Release`
  RPCs (`warm_up`, `gate()` when single-member with no in-flight forward or
  unconfirmed device copy batch, and `finish_drain`), `sync_peer_holder_changes`
  drains queued holder events and applies them through `advance_peer` before
  the next plan or cache clear runs. Single-member replicas in autonomous mode
  apply holder exports directly on the peer worker thread without using this
  point.

Anything that produces scheduler events (a new transfer backend, a new async
op kind) either returns events into one of these points or adds a new
explicit call site in `round` with a comment stating why the existing points
don't fit. It must not advance the scheduler itself.

## Principle 4: correctness never depends on the in-flight depth

`in_flight_depth` parameterizes the loop: 0 (classic synchronous
commit), 1 (one forward ahead of its commit: the overlap schedule on a
decoding role), or `pp_size` (the prefill chunk pipeline);
`scheduler_utils.resolve_in_flight_depth` picks it. At PP=1 the prefill role
runs at 1 too, the chunk pipeline's one-stage case: a round's planning,
commit, and transfer polling overlap the chunk on the GPU. It still takes no
overlap schedule, whose decode KV reservation (`overlap_schedule_depth`) it
never needs. `--disable-overlap-schedule` sets the depth to 0 at PP=1.
Dispatched forwards await commit in the `in_flight` queue. The round commits
once the queue exceeds the effective depth (0 when the round dispatched no
new work, so results never wait on future traffic).

The depth is a performance knob only. Any dispatch whose inputs depend on a
pending commit's side effects must commit the queue first, and
`plan_and_dispatch` holds every such rule in one place (currently: an
eager-grammar batch, whose bitmask fill reads matchers only a pending commit
advances). Rounds that run no real forward (a frozen pause, a DP idle forward)
commit the queue fully.

One dependency on uncommitted forwards is handled in a second place, by
waiting for the commits instead of forcing them: a shared L2 pool's write,
whose Device blocks a forward in flight may still be filling (the special
cases of the *Shared L2 pool* entry in *Subsystems in detail*). The write is
no dispatch to the data plane, so `shared_l2`'s `Pool` keeps the op
(`withheld_writes`) until the forwards that were in flight when its plan was
made are committed, which `commit_down_to` reports one by one, and the queue
keeps its depth. It is the only dependency handled outside
`plan_and_dispatch`.

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

The control thread drains an independent, contiguous event snapshot. The
publisher takes ownership of that batch; its thread encodes MessagePack, frees
the snapshot, and sends the payload. It cannot read the scheduler or retain cache
blocks. Timestamps are taken when the control thread submits the batch, and
sequence numbers and replay-buffer updates stay with the publisher in FIFO order.
The Python event-list API and the published wire format remain compatible.

The publisher keeps the existing bounded **batch-count** queue and blocking
backpressure. A full queue delays its caller; accepted batches are never discarded
to make room. Eliminating that wait would require an explicit loss/resynchronization
protocol. A disconnected publisher logs the failed send. Its diagnostic progress
handle counts outstanding snapshots (including a blocked producer and the active
worker), their logical bytes and high-water marks, and enqueue-call time. These
bytes stay charged until the send completes, even after encoding frees a snapshot.
They exclude allocator slack and encoded replay buffers; process RSS measures
those separately. The ownership guard releases counts on failed sends and dropped
queued work as well as normal completion.

The same reasoning fixes the metrics call: the loop records scheduler
iteration metrics once per round, before dispatch and before committing prior
forwards. `num_requests_running` counts the actual forward batch after DP
agreement, excluding idle padding, PD transfer waits and resident rows without
a decode execution permit. Scheduler `sizes.running` remains a state-occupancy
count for routing and must not substitute for batch size. `num_requests_waiting`
adds `sizes.waiting` (admission, prefetch and retraction/restore) and
`sizes.ready_waiting` (KV-ready D requests without a decode permit), plus
admissions withheld during a pause. Frozen rounds use the same recorder with
no forward, so RunningBS becomes zero and queue gauges continue updating. The
storage tick disables the scheduler's forward builder in those rounds. That
builder is the source of capacity aborts, and the no-progress streak resets
when forwards are forbidden, so a frozen plan has no scheduler aborts. Bootstrap,
KV reception and grammar compilation retain their separate queue gauges;
`num_requests_inflight` covers all unfinished requests, including those waits.
Queue subcategories overlap the total and must not be added to it again.

Only ranks with an engine metrics exporter retain per-request timing samples.
Decode request logs, exported latency metrics, and request traces use the
first local decode output as the first-token timestamp. The remote bootstrap
token's arrival is the prefill/queue boundary (`decode_ready_wait`), not the
start of the decode window, so a wait for a decode execution permit is TTFT.
Metrics and tracing are independently enabled; tracing alone retains no
metric samples.
Decode throughput counts tokens appended to the request, including the stop
token when present. A speculative suffix discarded at the stop boundary still
advances the scheduler frontier but contributes no output tokens.

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

Generation throughput counts tokens committed by decode forwards; on a plain
worker the first token normally comes from prefill. Completion counts are
role-local: the bootstrap token appears in both P and D totals. Use decode
completion totals for a disaggregated service's output-token count.
Speculative acceptance has separate verify-slot and request-level views.
`tokenspeed:spec_decode_accept_length` remains weighted by the decode slots in
each verify round. `tokenspeed:request_spec_decode_accept_length` observes one
unweighted sample per ended request with at least one verify round, using
`(output_tokens - 1) / verify_rounds`: the bootstrap token is excluded and each
round's bonus token is included. Requests without a verify round emit no
sample, and draining the output processor cannot duplicate a request's sample.
L3 query hits count the token span of each successful storage key. Cache
groups can cover the same prompt positions, so this measures storage hits
per group rather than unique prompt tokens.

Page gauges count LCM parents, excluding the reserved null parent. The
per-round sampler reads `empty_lcm_blocks()` and `active_lcm_blocks()`.
Cached-only parents are `num_usable_pages - empty - active`. Active and cached
counts are disjoint, even when cache groups pack multiple blocks into a parent.
The used-token gauge prorates the recipe's effective token capacity by active
parents, rounding down; free tokens are the complementary capacity. Separate
resident-token and resident-ratio gauges include reclaimable prefix parents.
A fully free pool reports its full capacity, including packed or capacity-limited
layouts.
`available_lcm_blocks()` scans the pool and prefix indexes for reclaimability
and belongs in diagnostics and leak checks, never in the per-round sampler.
`active_lcm_blocks()` is O(1): the block tables keep the count as they change
(see cache-concepts.md). The loop takes it at most once per 100 ms and the
frontend's load report, the metrics and the batch line share that value; a
round with no request running takes it again, so an engine that goes idle
reports none active at once, and does not keep it, so the first round with
requests takes a fresh one. The
scheduler's other counts (`sizes()`) are a pass over its requests per state:
a round takes them for the load report, and as it planned only when the
metrics or a batch line read them (they share that read); the idle check takes
them only when the loop holds no request, in-flight forward or open cache op.

## Subsystems in detail

**Pause and memory release** (`control`). A pause in `abort` mode aborts what
runs, `wait` lets it finish; both stop admissions and answer once the
scheduler drained, including requests parked for PD retirement: both the
transfer and local forwards must finish even if the output already ended.
`keep` freezes everything and answers at once: a frozen round commits what is
dispatched and still joins the attention-DP agreement
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
drain waits for those. On a prefill engine under `deferred` decode
allocation (`scheduler.md` §3.1) the scheduler holds a request the transfer
peer admitted before its decode destination is allocated, so a `wait` drain
also waits for those destinations: a computed prompt is still prefilling
until its transfer starts. A request the peer has not admitted is
bootstrapping, as under `early`, which a `wait` drain does not wait for.
The bootstrap timeout bounds the wait, from the moment the prompt held its
pages.
A disaggregated engine refuses new requests while a release is draining or
memory is released: their transfer registration would name memory the
release unmaps. On release the prefill peer deregisters its cache regions
from both receiving and sending engines, and registers them again on resume.
The decode peer starts a fresh transfer engine and assigns a new random cache
identity on resume. Registrations carry this identity separately from the
transport address: reusing an IP and port cannot identify the previous cache.
A new registration retires old identities at the same control endpoint or
transport address. Retirement waits for writes already in progress; queued
writes recheck their identity, and late registrations cannot revive it. When
a transport address is reused, the sender replaces its private transfer
engine to discard cached remote memory keys, preserving the public engine
address used by EPD. A release is refused while transfers are in
flight, and refused for a paused PD engine with buffered requests, whose
transfers cannot drain until those requests are admitted; their transfer
registration is deferred until resume admits them, and aborted buffered
requests never register one. An explicit empty release list is refused before
the scheduler drains. Omitting the list releases all configured regions.
An empty memory-resume request is a successful
no-op while a drain is pending; it preserves that drain's admission hold
until the owning pause or memory-release operation finishes.

The following memory-release and NCCL coordination uses protocol 2.

With attention DP, memory release waits for every rank to stop admissions
and drain its requests, forwards, cancelled embedding receives, cache copies,
transfers and queued weight operations. A locally drained rank keeps joining
idle forwards while a peer drains. An idle but unpaused peer prevents release:
it could admit a request in the next round. Release intent rides the head
gate, so every rank joins the tail's readiness agreement when any release is
pending, including ranks without a release of their own. Ordinary rounds add
no collective. An NCCL release also requires a pending NCCL release on every
rank: a previously completed KV release does not authorize that collective.
Other memory regions retain their local release semantics. Readiness is
MIN-reduced over the replica and then attention DP
before any release runs on the forward thread.

Sampler state, cache-arena geometry, capture placeholders and forward metadata
use the existing CPU-backed persistent-buffer regions. Their addresses and
contents survive the `weights` release/resume lifecycle; KV payload bytes keep
the separate discard-and-rebuild policy. These allocation scopes do not add
another execution path or change the draining order.
Per-forward packed cache tables join the same pool when built, including
their shared-storage views retained by the backend after the last decode.
Live forwards and prefill capture use the same managed table factory; target
and draft decode capture prepare their metadata through one explicit entry.

Host-transfer workspace address tables and per-submission device metadata use
the same CPU-backed regions, both at initialization and when a write or load
grows its tables on a transfer stream. Memory release drains those streams;
weights resume restores the tables before any cache transfer can use them.
The immutable Host-transfer geometry tables and capture-boundary barrier
dispatch markers use these regions too, so they cannot pin ordinary allocator
segments while the engine sleeps.
The capture barriers use the default WORLD group, bound eagerly during
distributed initialization; NVIDIA fabric discovery also exercises its
all-gather before model construction, outside any managed allocation region.

The default generator's graph seed and offset join the CPU-backed region when
registered, before CUDA capture begins. They keep their addresses and values
across weights sleep/wake; the graph's private temporary buffers remain
discardable under the separate `cuda_graph` tag.

A resume containing NCCL uses the same-round operation gate: it waits until
every attention-DP engine holds its resume before entering device collectives.
For engines with DP projections, the NCCL release drops captured graphs before
closing the projection workspaces they reference. Each workspace owns its
symmetric storage so it can release it at sleep. Full reclamation on CUDA
requires a PyTorch build that frees symmetric-memory rendezvous pointer tables.
The head gate waits for all memory regions on all ranks to resume, then rebuilds
those workspaces and recaptures graphs before it permits a forward.
Other regions still resume locally. A pending collective resume refuses a
second memory resume until it finishes.

The head gate also agrees whether any DP rank still has released regions.
Until all regions are restored on every rank, it withholds forward permission
through the scheduler's storage tick and suppresses idle forwards. This covers
staggered and partial resumes without building a batch that cannot execute;
control operations and cache acknowledgments continue to progress.

**Coordinated memory transitions (protocol 3)** extend the same control gate
from 12 columns under protocol 2 to 15. The three added values coordinate
call readiness, cancellation and timeout over the replica, or the existing
attention-DP domain when present, including shared MoE TP/EP. All members
must match direction, canonical tags and any transition ID before pause or
allocator changes. Normal and idle forwards can progress while a call waits
for missing participants. Cancellation and timeout refuse the whole observed
batch at that gate; cancellation preserves the current device mappings.

Release checks device prerequisites before its drain and again after the
domain drains requests, forwards, cache work and PD transfers. Every stage
must retire its KV transfers before source KV is released. Readiness is MIN-reduced directly
over this domain. All members must report L1 clearability before any index
is cleared; resume checks local prerequisites, including valid weights before
KV wake. Successful tags and replies are published only after every member
confirms device completion. A device-apply failure or L1-clear failure after
readiness agreement is fatal to the entire domain through the host failure
path, including otherwise healthy ranks; deployment supervision owns recovery.

Protocol selection, startup compatibility, the timeout/retry contract and
NCCL participation requirements are defined in the
[memory-transition serving contract](../serving/ascend-memory-saver.md#memory-transition-protocol).

**The same-round gate** (`EventLoop::gate`, every round on every rank).
Weight-update `flush_cache` and a standalone flush first MAX-reduce
flush intent across attention DP so every DP rank enters the same
collectives — the frontend sends a flush to every engine separately. The
same reduction gates the weight operations themselves (NCCL group
init/teardown, distributed, tensor, disk and Mooncake loads): each is queued,
and the head completes only in a round where every DP rank holds one of the
same kind at its head (a mismatch is an error — the frontend sends every
operation to every engine in one order). The device call blocks the control
thread in a collective the trainer drives or an SDK read; a DP peer that had
not yet dequeued its copy would keep looping and wait for this rank in the
per-round agreement, a deadlock. One operation completes per round. Before a
weight load's cache flush and load, the replica MIN-reduces each rank's device
prerequisites (weights resident, KV released under discard, an SDK update
under discard): a refusal on any rank means no rank flushes or loads, so a
ready rank never enters a loading collective its peer skips, and every rank's
weights and caches stay untouched for a retry. A refused cache flush likewise
loads nothing. After the load the replica MIN-reduces its result before the
loop publishes the L3 weight version or sends the reply, and commits that
final result on all forward threads before replying. Under the discard
policy, local loading success is pending: only the successful replica-wide
commit marks parameters usable. A failed load invalidates every rank,
including ranks that loaded locally; KV wake and forwarding remain blocked
until deployment restart. Discard requires both the model and its resolved
quantization configuration to opt in; quantized post-load derivation must
rebuild codes and scales in place from the complete new checkpoint, never
reuse discarded values or mark remapped storage usable on its own.

An L3 full-clear separates logical invalidation from native memory lifetime.
After the non-storage preflight, healthy I/O is given
`async.clear_cache_drain_timeout_ms` (default 500 ms) through ordinary rounds;
the pending flush replies and weight-operation head are preserved while no new
forward/storage plan starts. Polling, completion agreement and control queries
continue. The budget does not bound the normal synchronous remote purge.
An unsafe active namespace skips that healthy wait; a timeout or unsafe scope
selects retaining clear on every rank. Old tickets detach and keep their exact
Host addresses, while local indexes clear and new L3 work uses a fresh agreed
generation. This path performs no remote purge and proves no native retirement.
A slow healthy timeout still selects this private scope; it does not restore
cross-Prefill sharing with independent services outside the rank agreement.

Retained memory alone does not keep the serving loop busy. Request/Device-copy
readiness and progress-pending storage determine frontend idle waiting and
stall diagnostics; every round still polls and agrees old completions. Metrics
keep historical `l3_io_quarantined` separate from active-namespace
`l3_io_blocked`, backend health and capacity pressure. The historical flag guards
shutdown deregistration; it is not a reason to reject all new-generation I/O.
The same gate carries the online expert
rebalance's *internal* steps (snapshot, commit, chunk apply) from a second
queue with its own columns: frontend operations arrive on different DP
engines in different rounds while the controller queues internal steps at
rank-identical rounds, so one shared queue would make the heads differ in
kind across ranks. When both are ready the frontend operation completes this
round and the internal step the next — still one blocking device call per
round. Because every EP rank is inside a gated step at once, the rebalance
may run a collective over the EP group there (summing the load under
all-to-all EP, broadcasting the committed placement), which the profile's
`dump_expert_load` must not. The manual rebalance (`RebalanceExpertsReqInput`)
is a frontend operation on the first queue that starts the same internal
sequence. The loop refuses a memory release while a rebalance is in progress
and, once drained, the release waits while chunk steps still write the
weights. A cache clear agrees its selected non-mutating preflight across the
replica and attention DP. Without storage it retains ordinary
`can_clear_cache` semantics. With healthy storage, preparation checks normal
indexed Device pins and indexed Host owners after accounting for the storage
references that draining can release. Exact physical-block reference counts,
active parents and occupied-slot coverage are required when local invalidation
must retain old storage references, not for every ordinary clear. The strict
path agrees remote namespace deletion before local mutation; retaining recovery
instead agrees isolation and performs no remote deletion. The frontend combines
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
The data plane acknowledges each op exactly once, one ACK per op of a batch,
and the counts rely on it: `acknowledged` takes each polled ACK off the
in-flight count before it deduplicates, so a repeated ACK breaks the counts.
A poll then fails with `CacheOpError::Unaccounted` (at once, or once the
remaining ops land), or the open count reaches zero while another op is in
flight, and the loop stops polling before that op lands. Only the C++
scheduler tolerates a repeated ACK.

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

**Shared L2 pool** (`shared_l2`, `--shared-l2-config`). The rank's Host tier
is the proxy of one Host L2 pool that every rank shares; its main rule, three
sentences on the lookup, the hit and the write, is stated in the
[scheduler design](scheduler.md#1-admission-is-per-chunk) (*A shared L2 pool
is asked before admission, never by it*).

The C++ scheduler decides what to ask the pool and what to write, the pool's
client (`RankClient`, `ts-control/src/shared_l2/client.rs`) talks to the
owner and copies between the rank's Device blocks and the providers' pages
over the host's transfer engine, and the loop relays between the two. The
rank joins the pool at device build and starts the client then
(`cache/shared_l2/rank.py`, through the host's shared L2 handle). The pool
needs a replica of one member (attention TP 1, no pipeline): the client
answers one rank.

The pool's owner is a process of its own and runs no loop of this document.
Its launcher (`python -m tokenspeed.runtime.cache.shared_l2 owner`) builds
the scheduler's `SharedL2Pool` when the first rank reports its cache geometry
(`cache/shared_l2/owner.py`), and the host extension serves the pool on one
thread with the GIL released (`run_shared_l2_owner` in
`ts-host/src/shared_l2_owner.rs`, over `Owner::serve` in
`ts-control/src/shared_l2/owner.rs`), as a rank's host drives a scheduler
that Python built. That module's header lists the requests the owner
answers.

The loop's part:

* *Lookups.* A round that may forward starts, right before its plan
  (`shared_l2_round`, one call in `round_with`), the lookups of the waiting
  prompts the scheduler has not handed out yet (`start_shared_l2_lookups`)
  and, if there are any, asks the owner about all of them in one request
  (`PoolClient::lookup`). The round waits for the reply, for up to the
  client's request timeout, and applies the answers in one call before it
  plans (`apply_shared_l2_answers`): a prompt taken in at
  the head of a round is admitted by that round's plan, behind the prefix
  the pool holds for it, and a miss, which is only an answer in which the
  owner names no page for the prompt, computes as on a rank without a pool.
  The proxy blocks the answers created go back to the client in one call
  (`map_proxy_blocks`), which maps them to pool pages before any plan can
  load from them. No
  lookup is sent anywhere else. On ranks whose rounds attention DP keeps in
  step, the request lies between the round's two DP collectives, the gate
  and the batch gather, on the asking rank alone: its peers wait for it at
  the gather as they wait for a rank whose commit takes longer.
* *Poll.* The head of every round polls the client
  (`PoolClient::poll_writes`, which never waits), whether or not a write is
  in flight. The poll returns
  the writes the client finished, which reach the scheduler as
  acknowledgments in the head's `advance_head` and let go of the Device
  blocks they read, and raises a failure that the client recorded on its
  own threads for a write or a notice (a release, a touch, a ping), before
  that round plans. An idle rank's loop still runs a round about every
  10 ms (its wait for a request), so it reads a failed ping within about
  that time.
* *Ops.* Each plan's shared L2 ops (write, release, touch) go to the
  client when the plan is dispatched, next to the L3 submit; the client
  carries them out on its own threads. A plan that carries shared L2 ops
  only is not handed to the data plane.

A hit's load is not the loop's: it is a load-back of the plan, which the
forward thread hands to the client's loader before it puts the load's waits
on its streams (`SharedL2LoadExecutor.submit_load_backs`, `Loads::submit` in
`ts-control/src/shared_l2/load.rs`). Handed over first, a load the loader
refuses leaves no stream waiting on a word that nobody sets; the loader
reads nothing before the load's start word anyway, which the stream writes
once the plan's page zeroing is done. The loader then reads the pages layer
by layer and sets each layer's ready word, and the acknowledgment returns
through the loop's poll of the data plane's cache acknowledgments, counted
and agreed between the ranks like any other (`cache_ops`). The host answers
that poll from the loader's finished loads without entering Python
(`PythonDataPlane::poll_cache_events`): load-backs are the only cache ops of
a pool rank that the data plane acknowledges, since its plans carry no
write-back (the pool is written by the client) and no snapshot store or
restore (a pool rank does not retract). The poll raises a failure the client
recorded, like the client's own poll.

The special cases, one sentence each, with where the code is:

* *A pool operation fails.* Every failure of the pool is a loop error and
  stops the engine, as a failed load-back does: a lookup request the owner
  did not answer in time or answered with an error, a reply without one hit
  for each lookup, an answer the scheduler cannot apply (it names no prompt
  that waits for one, lies outside its prompt, or is malformed), a write
  that failed at its reservation, its copy or its commit, a release or a
  touch the owner refused, a ping the owner did not answer, an owner that
  does not know the rank (`PoolError` in `ts-control/src/shared_l2/mod.rs`,
  raised by the lookup, by the poll and by the check when a forward has
  landed; `ApplySharedL2Answers` for an answer). A panic of the client's
  load thread, of a write thread or of its notice thread is that thread's
  failure and stops the engine the same way (`panic_as_failure`, same
  file).
* *A write names blocks a forward in flight may still be filling.* The
  scheduler starts a publication's write in the plan after its chunk's,
  whether or not the chunk's forward landed, and the client reads the
  Device blocks over RDMA, which nothing orders behind the device's
  streams, so a plan's writes are withheld until the forwards that were in
  flight when the plan was made are committed (`Pool::dispatch` and
  `Pool::forward_landed` in `shared_l2/mod.rs`, the second called from
  `commit_down_to`).
* *The lone-request compaction would move a block a queued write names.* A
  publication that waits in the store queue for its write pins nothing, and
  a write started in the compaction's plan would name a block that plan's
  copy has yet to fill, which the client reads outside the device's
  streams, so the scheduler compacts no request while one waits, and a
  plan's own copies fill no block its writes name
  (`LoneRequestRecovery::AdmitFirstChunk` and `AdmitStep`,
  `scheduler/lone_request_recovery.cpp`; [scheduler](scheduler.md) §1.4).
* *A load from the pool fails.* The loader lets the forward that waited on
  the load go, over pages it never filled, and the failure stops the engine
  before the results of that forward are committed or streamed: the loader
  records the failure for the client before it sets the ready words that
  let the forward go (`Loader::run` in `shared_l2/load.rs`, into the
  client's `FailureLatch`), and the loop checks the client when the forward
  has landed, before it commits (`Pool::forward_landed`, called from
  `commit_down_to`).
* *A client thread that does not end at shutdown.* A thread that still
  waits for the owner or for a copy when the rank closes keeps its socket,
  the transfer engine, the registered cache and the signal words in use, and
  a stream may still wait on a ready word it would have set, so the close
  waits for the threads for 10 seconds and then leaves all of those, and
  the streams, as they are for the process's exit to take. When every
  thread has ended, the close itself stops the transfer engine, after them,
  and the rank then synchronizes its streams and frees the words
  (`RankThreads::close` in `shared_l2/client.rs`;
  `SharedL2LoadExecutor.shutdown`).
* *A frozen round.* Its plan forwards nothing and starts no write, but it
  still releases the tickets nothing holds any more, such as one whose last
  holder was a load-back acknowledged at the round's head, so a frozen
  round hands its plan's shared L2 ops to the client too and the owner does
  not keep those pages pinned (`Pool::dispatch` in the frozen branch of
  `round_with`).
* *A cache flush or a stall count while a write is in flight.* A pool write
  the client has not reported is a cache transfer in flight: a flush, which
  clears the rank's Device cache and its proxy Host tier and leaves the pool
  to the ranks that go on loading from it (the pool is emptied at its owner,
  `clear`), is refused like every flush during a transfer, because the
  write reads Device blocks the flush would drop, and the scheduler's stall
  diagnostic counts no stall, so a request that waits behind the write is
  not aborted for no progress (`TierTransferManager::HasAnyInFlight`,
  `cache/tier/transfer_manager.h`).
* *A weight load or a forward reload.* The pool has no weight version, so a
  page written under one set of weights, or by one forward, would be read
  back under another: with a pool the loop refuses the operation when it
  arrives, before it is queued for the gate, so nothing drains and nothing
  is flushed for it (`EventLoop::control`).
* *KV the pool holds past its last checkpoint.* The answer reports how far
  that KV reaches, and the reach joins the Host tier's KV reach, so the
  prompt's first chunk is cut at that junction and the checkpoint cut
  there is where the next prompt sharing that KV, on any rank, resumes
  (`AdmissionResources::shared_l2_kv_tokens`, read by
  `Scheduler::matchPrefixAtAdmission` in `scheduler/operations/forward.cpp`).
* *A configuration the pool does not cover.* It is refused at start and
  never served around: the server arguments refuse it
  (`ServerArgs.validate_cache_options`, `resolve_retraction_snapshot_options`
  and `validate_startup_requests`; `resolve_dspark_prefix_replay_tokens` for
  a DSpark captured-context replay), `SchedulerConfig::Validate` refuses
  again what the scheduler relies on (the P and fused roles only, prefix
  caching, a write-through Host policy, replicated groups, no L3, no
  snapshot pool, so that a pool rank never retracts, and no peer prefix
  sharing, whose holder would export proxy blocks that hold no bytes), a
  rank refuses to start on a device without ordered GPUDirect RDMA writes or
  CUDA stream memory operations (`cache/shared_l2/executor.py`), and the
  owner refuses a rank whose cache geometry or deployment identity differs
  from the first rank's (`Owner::hello` in
  `ts-control/src/shared_l2/owner.rs`); the
  [server configuration](../configuration/server.md#shared-l2-pool) has the
  operator's list.
* *A hit the proxy Host tier has no room for.* It is a fatal check and not
  a miss: the tier holds at most what the pool pins and is sized past the
  pool (`SharedL2Proxy::MirrorAnswer`, `cache/tier/shared_l2_proxy.cpp`).

Known limit: nothing guards a rank process that was frozen for longer than
the owner's client timeout; the
[server configuration](../configuration/server.md#shared-l2-pool) says what
it can read and write after it resumes, and what an operator does about it.

Without a pool the loop makes none of these calls.

**PD source retirement.** A terminal status alone does not release source
pages. The prefill peer counts queued and executing transfer jobs. A failure
agreed by the replica first stops new work on every rank and reports `Failed`,
which parks the request without releasing its resources. Every rank must
retire its jobs before `Stopped` reaches the scheduler. Queued
jobs for failed rooms are dropped; executing jobs finish or return from the
synchronous transport call before retiring. Client cancellation keeps the
request's source ownership until that retirement event. The retirement agreement
is needed only in rounds with a terminal status.

**PD output completion.** A prefill's computed finish is not its client's
final output: the output waits for the transfer's result. A transfer failure
must publish an abort even after computation finished, unless the client
already cancelled its stream or a final output was sent. A numerical error
detected by the NaN guard in the prefill role ends the request when the
prompt's final chunk commits instead of waiting for the transfer; an error
detected in an earlier prompt chunk is remembered until that final chunk
commits. When the final chunk commits, the request's `Abort` event cancels the
transfer to stop the remote decode, and the output processor emits the
terminal abort output unless the client already cancelled. A pause abort still
notifies the client; repeated transfer events do not repeat the final output.
This output fence does not retire source pages.

**PD route refresh.** Before attaching a new room, the decode connection worker
fetches the bootstrap route version. The version changes on bootstrap restart
and source-rank registration, even when the address stays healthy. A changed
version invalidates the cached route and target registrations for future rooms;
existing rooms keep their targets and cancellation retirement fences. Bootstrap
servers without a version are resolved afresh on each connection. This adds
one bootstrap lookup per room, off the scheduler and forward threads. It does
not transparently migrate an in-flight transfer to a replacement prefill peer.

**PD cancellation.** Client aborts and `pause(abort)` cancel the tracked
transfer as well as marking the request's output. A decode receiver sends a
cancellation to every prefill target that could write its landing pages.
Each target stops accepting work for the room and acknowledges only after
its queued and executing writes retire. The receiver waits for every target,
then agrees retirement across its replica before reporting `Stopped` to the
scheduler; `Failed` may already have ended the client request. Failure
notifications and receive timeouts use the same fence:
neither proves that remote writes stopped. A receiver that never advertised
landing pages can retire locally. Cancelled rooms reject delayed registration
and manifests; repeated cancellation is safe.

`Failed` precedes `Stopped`, but both may arrive in the same poll when the
transfer has already retired. Consumers accept either one poll or separate
polls and keep the pages until the retirement event and local forwards complete.

The prefill control receiver handles cancellation even when no native request
claimed the room or its scheduler is paused. Unclaimed rooms therefore need
not wait for the bootstrap timeout after the decode request is cancelled. P
and D must run versions supporting this cancellation protocol. If a target
cannot acknowledge retirement, the receiver retains its pages and the drain
remains pending; a transport failure alone cannot make those pages safe to
reuse. Only a revocation (below) releases them without the acknowledgment.
Cancellation tombstones are retired with their destination cache identities. A
prefill cancellation that never learns a destination identity must retain its
room tombstone until that peer exits, since an arbitrarily late manifest must
not revive the transfer.

These records include every prefill failure and bootstrap timeout, not just
explicit client cancellation. There is currently no count or age limit:
unbound records last for the prefill peer's lifetime, and bound records can
last just as long when a healthy decode cache is never superseded. Sustained
timeouts therefore grow metadata with the number of failed rooms, even after
all associated jobs and network workers retire. Request timeout bounds do not
bound this memory. An age or size eviction would allow a delayed manifest to
revive an ended room; the wire has no enforced message lifetime or retired
attempt epoch to reject it. Bounded retention needs such a protocol guarantee
(or a fail-closed admission policy), rather than assuming that a multiple of
the bootstrap timeout rules out late messages.

A native write error does not prove that its remote DMA stopped. The prefill
peer marks that room uncertain and withholds cancellation acknowledgments even
after its queued and executing calls return. Failure notices may still reach
the receiver, but cannot release its pages. An uncertain room or an unreachable
writer can therefore keep the receiver's drain pending indefinitely; stalled
drains are logged periodically. A parked request also keeps operations that
require a drained engine, such as cache flush, weight replacement and sleep
with memory release, from completing. Recovery requires verified retirement
of the writers or transport invalidation that makes further access to these
registrations impossible; local forwards must also have completed before
pages are released. A timeout can trigger diagnostics or recovery but cannot
authorize page reuse.

**PD receive revocation.** A failed receive whose cancellation some target
has still not acknowledged `waiting_timeout` after it failed, while its
manifest names the current transfer engine, starts a revocation off the
scheduling loop: the writer died, restarted at another control port, or has
a write that may still land. The decode peer registers its cache under a
fresh transfer engine and cache identity and makes it the one transfers use
and registrations name; every receive attached to the replaced transport
fails, whether it published a manifest or still waits for admission; then
the cache is deregistered from the replaced engine. An RDMA NIC checks every
incoming write against the destination's registration, and deregistration
returns only once the NIC no longer uses it (this relies on the Mooncake
library deregistering synchronously on every device), so after it succeeds
no write through the replaced engine's keys lands: the receives whose
manifests named that engine retire without acknowledgments. The fence, not
the timeout, authorizes the reuse; the timeout only picks when to pay for
it. If the deregistration fails, those receives keep waiting for
acknowledgments and no longer trigger revocations of later engines, and
since the replaced engine may still reach the cache, the cache is never
released. Nor is it released while a revocation runs. Each room's
cancellation, and its announcement under `deferred` decode allocation, names
the cache identity its route registered, so a receive that failed before
admission still reaches prefill under an identity it knows. Rooms that were
still connecting, including those waiting for a registration acknowledgment,
connect again under the fresh identity, registering it with their prefill
ranks, and a failed receive never publishes its manifest later. The replaced
engine stays alive for the peer's lifetime without the cache registered, so
its session (host and RPC port) cannot be reused by a later engine that a
revoked writer resolving its old session would reach. Only the `rdma`
transport revokes; with others such receives stay pinned. Each outcome is
logged as an event: `ts_decode_revocation_fenced`,
`ts_decode_revocation_unfenced` (with the pinned rooms), or
`ts_decode_revocation_not_started`; a release refused for an unfenced cache
logs `ts_decode_release_refused_unfenced`. Each revocation costs every
receive attached to that rank at that moment, which its replica peers fail
too through the agreed transfer status, and one idle engine.

**PD failure isolation.** Control messages use one bounded FIFO per destination
endpoint. Connect and send each have a five-second deadline; a failed endpoint
cannot hold another endpoint's registration, manifest or cancellation. An
identical cancellation or status occupies at most one queued/in-flight entry.
Queue refusal fails registration/manifest publication; partial publication
still uses the cancellation fence. Cancellation polling retries refused or
undelivered notifications. Network delivery itself remains best effort: a
locally accepted message is not a remote acknowledgment. A connect or send
failure discards that endpoint's queued backlog and retires its task; it does
not spend another I/O timeout on every queued message. Requests still use
their failure/timeout and cancellation paths. After 60 seconds with no queued
or in-flight message, even a healthy endpoint retires its socket, task and
registry entry. The next send recreates them. Idle retirement and enqueue
share a lock so a new message cannot be stranded on a closing FIFO; an old
task's teardown cannot erase its replacement. Socket retirement does not
retire cache identities, room tombstones or possible remote writes.

A decode cache's registration with a route is therefore acknowledged. Each
registration carries an id; a prefill rank acknowledges it once the
registration is installed, so the manifests that follow find the cache. An
identical registration, such as a resend or the same cache registering again
after a route change, replaces nothing and is installed without waiting for
writes. The decode connection worker resends the registration every second to
the targets that have not acknowledged it, also when their queue refused it,
and records the route as registered only when every target has. A target that
rejects the registration replies with its reason, which fails the room at
once. A room whose route is not acknowledged within 10 seconds fails before
publishing a manifest, so it retires locally, and the next room registers
again; since one worker serves a bootstrap address, an unreachable target
delays that address's later rooms by up to 10 seconds. Recording a route when
its registration is merely queued would leave a lost registration's route
recorded but unknown to prefill: every later room on it would fail and keep
its pages pinned. Prefill ranks still install the id-less registration of an
older decode rank, without acknowledging it; a decode rank needs prefill ranks
that acknowledge, so prefill ranks are upgraded first.

Decode bootstrap lookups are serialized per bootstrap address, with independent
workers for other addresses. Polling (including idle polls) and registration
release a lost address's sender once its route is gone and no live request is
still looking it up. The worker finishes any bounded in-flight HTTP call,
skips terminal queued requests and exits; a later request creates another
worker. A first lookup with no route yet keeps its worker so concurrent
requests for that address stay serialized. Heartbeat failure counters expire
with the routes too. Cache registration on prefill fences only the
cache identities being replaced. A replacement waits off the control receive
path for those native writes, while unrelated registrations and cancellations
continue. Manifests for the replacement wait until it is installed; queued
old writes recheck their identity before reaching native I/O. Synchronous data
writes still use the configured transfer worker pool, and heartbeat probes are
still sequential; these bounds do not provide native write cancellation or
writer incarnation discovery.

Bootstrap timeout and other prefill failures retain the same terminal room
record as explicit cancellation. A late manifest is rejected immediately,
without another unclaimed timeout. Before manifest publication, decode bounds
bootstrap and scheduler admission waiting by `waiting_timeout`. Publication
starts a separate receive budget of the same duration. These phase budgets do
not constitute an upstream end-to-end deadline. Before publication a failed
receiver retires locally and notifies known targets; after publication it must
obtain the normal retirement acknowledgments.

**PD** (`pd`, `ts-pd`). Requests register with the transfer peer when submitted
to the scheduler, before cache admission. Every round the replica agrees on
which transfers changed state; their scheduler events join the round's tail feedback, and a failed
transfer aborts its request on every rank together. The loop releases its
scheduler mutex during the transport poll and reacquires it before handling
the agreed events. Decode can therefore serve a prefix lookup from the very
Prefill whose KV it is waiting for. On contention the autonomous holder can
wait briefly for a round-boundary handoff once per pass, then returns to its
control queue so unrelated releases can retire.
Under `deferred` decode allocation (`scheduler.md` §3.1) the peers keep the
order "prefill pages first": the prefill node states the policy in its route,
and a decode rank that reads it sends `["Announce", room, cache]` to the
room's prefill ranks once connected, again with every heartbeat until each
answered (at most 256 per endpoint and beat; a beat resumes in room order at
the first room the previous one left out, whichever endpoint's). Prefill
requests are schedulable on submission, without waiting for an announcement.
The pages-held marks and the clocks that start at them are local to a rank and
time-based; the members of a replica would have to agree on them, and
`deferred` requires single-member replicas. When the plan carrying a
request's last chunk goes to the device (`ExecutionPlan::completed_prompts`,
the scheduler's `ForwardBatch::CompletesPrompt`) the host marks it
(`PrefillOutbox::mark_pages_held`) and the announcing ranks get the room's
`Bootstrapped` status, as every later announcement does; a room that failed
answers `Failed`. Only then does the decode peer report `Bootstrapped` to its
scheduler: until every expected prefill rank's notice arrived the room is
reported connecting, and the notice is kept beside the room's state, not in
it. The manifests then reach the prefill peer, which reports
`DestinationReady`; a manifest for a room that was not announced, or whose
request does not hold its pages yet, fails the room, so a decode rank that
does not follow the order fails at once. The decode scheduler sees every
noticed request and may allocate them in any order. Timeouts: on the prefill
side the bootstrap timeout bounds what a request waits for from its decode
node. It runs from the pages-held mark to the manifests, whether or not the
request was announced; queueing and computation before that mark consume no budget.
A late announcement receives the pages-held notice immediately. On the decode
side the waiting timeout runs from registration to the notice, restarted at
the notice. A request that fails on the prefill side tells its announcing
ranks as well as those that pre-allocated.
A peer keeps state only
for the rooms of requests the loop tracks (a sleep waits for those alone),
with one exception on the prefill side: a room the decode node pre-allocated
for before its request registered here, or one this rank failed first (an
EPD embedding timeout), waits unclaimed; one still unclaimed after the
bootstrap timeout is dropped, and the decode ranks that pre-allocated for it
are told it failed. A decode room whose KV lands without the request's first
token fails: the first decode feeds on it.

`pd::TransferStoppedEvent` retires the transfer of the parked request. Its
resources still wait for any outstanding local forward results before release:
a transport fence does not fence this engine's execution stream. A late success
retires a parked transfer under the same local-forward condition.

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

## Request-token history seeds

The control plane owns each request's committed token list. `HistoryRows`
tracks slot ownership by admission, not by client request id, and seeds a
new owner's prefix before its first forward. Consecutive chunks on the
same owned row append their inputs without reseeding. Prefix-cache hits,
PD landings, slot reuse and recovery into another slot use this same path.
Only the most recently used slot can hold a request's current history. Moving
to another slot invalidates the old ownership claim: a later restore into
that old slot must seed again, even if no other request has used it.

A model profile may declare `request_token_history_lookback`: the maximum
number of committed tokens it reads behind its write frontier. Omission
(`None`) preserves full-prefix seeds; zero permits an empty seed. The host
uses the maximum bound across target and draft history readers, or full
history if either reader declares no bound. A finite bound is a model
correctness contract, not a serving knob: every history read must stay
within it, including resumed and speculative forwards.

For a prefix of length P and bound L, the seed carries tokens
`[max(0, P-L), P)`, retaining P as `prefix_lengths`. The Python executor
places them at their original absolute positions, never at position zero.
The draft row keeps its existing one-token-left shift: source positions
`[start+1, P)` populate draft positions `[start, P-1)`. The final draft position is not
provided by the seed, just as with full-prefix seeding; the model/drafter
must supply any boundary token it reads. Earlier row contents are
unspecified and must not be read. This reduces copied and uploaded tokens,
not the context-sized history allocation or KV-cache storage. Token history
remains token-derived state, reconstructed rather than included in
retraction snapshots.

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
   data plane's acknowledgments and intersect them; with a shared L2 pool,
   poll its client for the finished writes and for a failure it stored;
   then **advance the scheduler (head)** with all of it, so this round's
   plan sees them.
3. Frozen? Commit what is dispatched, plan (the tick forbade forwards, so
   the plan carries cache and storage work only) and dispatch that work,
   and join the attention-DP agreement.
   Otherwise: with a shared L2 pool, look the waiting prompts not asked yet
   up in one request, wait for the reply and apply the answers
   (`shared_l2_round`); then plan
   (`next_execution_plan`), count its cache ops, finish the
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
   the L3 executor (Host workers: no stream, no Device page) and its shared
   L2 ops to the pool's client. The forward
   thread adapts the plan's cache ops once (`cache_ops_from_plan`), sets the
   storage subset aside and turns the rest into one
   `DeviceHandle.execute(plan, planned, cache_ops=...)`, which submits in an
   order that is itself a correctness contract for same-round page reuse:
   the Host-bound stores first --
   `WriteBackOp` and `SnapshotOp`; the stream-ordered ones, a retraction
   image's two legs (the L2 write-back of the victim's hash-complete pages
   and the snapshot store of its tail pages and slot state) and, under the
   Host write-back policy (`--kvstore-write-policy write_back`), the L2
   write-back of what the plan's admissions and restores evicted, must
   read the reused pages' old bytes, so they ride the write stream
   consecutively and ONE completion event fences the forward thread's
   default stream here, before anything else of the plan is enqueued; an
   ordinary store's sources (a write-through publication's) are pinned by
   the scheduler until the ACK, so its copy rides the write stream and
   fences nothing -- then the plan's Device-to-Device block copies
   (`pages_to_copy`: a prefix hit's private copy of the block holding its
   boundary, an owner's move off the block it published at its prompt end;
   the copy waits on the execution stream and precedes the zeroing because
   a source may be re-granted and zeroed by this very plan -- or re-granted
   as another group's copy destination, which the arena writes only after
   every pair reading those byte ranges; a destination is never zeroed;
   no op id, no ACK), then
   page zeroing (the new owner's sanitization of the fresh pages nothing
   fills whole: a copy destination and a block a remote landing fills are
   left out), then the Device-bound copies
   -- `LoadBackOp` and `RestoreOp`, ordered behind the zeroing and the
   forwards in flight, which may still write a block an abort freed and this
   plan granted (load-back destinations are never zeroed): the forward reads
   a load-back layer by layer behind the
   layerwise fences (a prefill graph replay waits on them before each
   captured segment, which holds none, or, on a pool that does not declare
   every access fenced, for the whole load instead), so the loads start first
   on the load stream, and every plan, with cache ops or without, points
   the fences at its own load or at none; a restore
   queues behind them, overwrites its destination blocks whole (the
   scheduler lists none of them for zeroing), reads both Host tiers under one
   event and arms no layerwise fence, its request being unschedulable until
   the ACK -- then the plan's remote streams to the transfer peer (a D-node
   remote prefill submits only its stream-ordered slot preparation here),
   then the plan's batch to the model, then the D-node remote prefill's
   pull, which waits on the zeroing fence before its CPU-issued RDMA
   trigger. The zeroing orders behind the step in flight, so a pull queued
   ahead of the batch would hold the batch's launch until that step ends. A
   store names the victim's request-pool slot, which the scheduler
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
In particular, a signal can interrupt a buffered stderr write. Logging again
can raise before the callback records a failure or requests shutdown.

While serving generation, a fatal control-loop or forward-thread error
signals the owning launcher once from the thread that detects it. The
signal requires no Python call and precedes the Rust error log and any wait
for the forward thread or device cleanup. The launcher can begin stopping
the local ranks even while the failing rank is still alive. An orderly stop
still drains the forward thread and closes the data plane.

Concurrent reporters wait only for the native signal attempt, never for
another thread's log output. The data plane publishes its first failure
and wakes waiters before logging either the rank failure or a failed signal.
Later distinct failures and cleanup errors remain visible without replacing
the original serving error.
Failures outside generation serving use the same native notifier when
they reach the host's final error handler; their cleanup timing is unchanged.

### Startup requests

Model warmup runs the model alone on a probe cache. It never reaches the
request bookkeeping (input preparation, accepted-state commits), and the
final cache can change kernel specializations it did prepare. With
`--enable-startup-requests` the host closes both gaps before it registers:
the loop starts with no frontend and `EventLoop::warm_up` serves requests of
its own through the ordinary rounds (admission, planning, the data plane,
commits), on every rank of the replica as a frontend's would be.

`startup::Limits::phases` sizes them from the engine's limits alone: one
one-token request, one each of a thirty-second of a prefill chunk, a
sixteenth, one token more, and three sixteenths (the partial forwards that
short prompts and a long prompt's last chunk run, aligned and not), one of a
whole chunk, then `max_num_seqs` requests whose prompts cross one chunk
boundary, each cut to what ingress
admits and so the whole batch fits the cache. Each phase runs until every
rank is idle, then its prefixes are flushed, so serving starts from an
empty cache. Their outputs go nowhere and kernels prepare their compile
variants as in model warmup. Only
then does the host complete the frontend handshake and report ready: no
client request waits behind them or is timed with them.

A disaggregated node is refused (its requests need a peer), and so are an
L3 store (the flush would delete its shared namespace) and a shared L2 pool
(the flush would leave the startup prefixes in the pool).

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

## Frontend load reports

After the round's scheduler mutations and releases, the replica's frontend owner
publishes its current request and cache load. Token outputs carry that snapshot;
a changed snapshot is also sent in an empty output batch. Client cancellation
can reap the final request without producing a final token, so reporting only
nonempty outputs would leave routing and drain checks with a stale running
count. An unchanged idle load is sent at most once per second as a heartbeat,
so the frontend can distinguish an idle engine from a stale connection. The
active page count in it is reused for up to 100 ms (see the page gauges),
far inside the three seconds the frontend considers a report fresh.

Token counts are independent of allocator occupancy. `num_resident_tokens`
sums prompt plus committed output tokens over the output processor's live
requests. `num_waiting_uncached_tokens` sums waiting prompts less the cache
hit observed by their last admission probe, plus unscheduled local prefill
tokens. Before the first probe a waiting prompt counts in full; reporting
does not trigger a probe, and admission always rechecks cache residency.
Both optional wire fields are declared in the handshake and retain `None`
when decoding reports from older engines. New engines report zero when idle.

## Independent execution with shared weights

The device build exports `requires_forward_coordination` as an immutable
capability derived from execution groups. The host hands the control plane a
DP coordination group only when forwards span workers, or when a shared L3
namespace needs agreed flushes. Without one, fully local workers omit the
per-round DP metadata exchange, peer-induced idle forwards and the DP
rendezvous for cache flushes while retaining their DP routing identities.
Sharing immutable expert weights alone does not require forward coordination.
Weight-copy streams and readiness/release events remain behind the model on
the data plane; see [distributed immutable weights](distributed-weights.md).


### Cache and request observation semantics

The legacy used-token convention excludes reclaimable prefix-cache pages.
`kv_token_usage_perc` and `num_used_tokens` therefore use active pages;
`kv_cache_resident_usage_perc` and `num_resident_kv_tokens` preserve physical resident
occupancy. `num_free_tokens` includes reclaimable pages and complements used tokens.

Cache ratios are atomic weighted observations: one admission snapshot carries
`hit / prompt` with weight `prompt`. Device and Host hits are disjoint admitted
prefix spans; Host includes data supplied by L3. Storage-key matches can repeat by
cache group and do not count as admitted L3 hits. Decode-cache observations come
from the decode scheduler's own admission delta, never the prefill response's
cached-token usage. A zero-hit admission must still emit a weighted zero.

Speculative accept length includes the bonus token: `(accepted_draft + slots) /
slots`, weighted by verify request-slots. Its aggregation window belongs to the
reporter. These aggregate observations preserve means and sums, not individual
request/token quantiles. Prometheus sums use the original totals exactly; raw
metric events carry only mean and weight, so reconstructing their product can
introduce floating-point rounding. Python's weighted Prometheus updates use
`prometheus_client` histogram internals; the supported minor version is bounded
in `python/pyproject.toml` and upgrades must pass the reporter tests.
Completion-token samples, in contrast, preserve one
sample per ended engine request (including zero output), separate from the sum;
they retain role boundaries, including prefill's bootstrap output. Frontend
end-to-end samples represent client-facing request distributions.

A registered request whose PD transfer fails uses the normal request-metric
completion path. Its generated-token total and per-request completion sample,
request duration, first-token/decode latencies when available, and stream e2e
sample are recorded once before removal. Invalid requests rejected before
admission retain their existing metric coverage.


PD cancellation retains the same protocol failure status and page-retirement
fence as an error, but does not increment the KV transfer failure counter.
Only actual failure paths increment that counter; cancellation acknowledgements
and retries cannot turn a single failure into another counted error. Timeout
counters remain separate. Replica-wide terminal agreement cancels remaining
work without incrementing failures on unaffected ranks. The failure counter
counts locally detected errors on their detecting ranks, not every replica
rank affected by the same room termination; a rank-local timeout likewise
remains a timeout when other ranks converge on its terminal status.

P's transfer-wait queue uses `prefill_inflight`: PrefillDone plus Decoding,
from the final prefill result landing until PD completion. Computing prefill
chunks and cleanup pins are excluded. D's transfer queue uses RemotePrefilling,
which is also its KV-wait queue; those two exported gauges are aliases, not
additive subqueues. The broad scheduler pending gauge continues to include
restore, permit, and pause waits.


### Output-chunk TPOT and request e2e_TPOT

TPOT observes each nonempty output chunk after the first: elapsed time
since the previous nonempty emitted chunk divided by newly emitted tokens.
The independent e2e_TPOT observes request end minus the first emitted chunk's
time, divided by all output tokens minus the first chunk's token count. The
first chunk may contain several tokens under speculation or stream batching;
subtracting a fixed one is incorrect. Empty outputs do not advance the TPOT
clock, and single-chunk requests have neither metric. These clocks are the
engine's role-local output boundary; the native frontend exposes the same
formulas at its client-facing output boundary, including transport overhead.
Both observations preserve their distributions without window averaging.
With `stream_interval=1`, the raw TPOT rate is the number of noninitial
output chunks: approximately active decode requests times decode steps per
second, summed across ranks. Do not replace a round's distinct samples with
`observe_batched` of their mean: that preserves the sum but loses quantiles.
Capacity checks must cover the relay and the chosen sink separately; a slow
sink can fill the relay queue even when transport alone sustains the rate.
Each output observation pairs its sample destination and timestamp together;
disabled metrics and prefill output do not create an observation.

Admission failures prepare their final output before logging completion, as the
other terminal paths do. Preparing that output consumes a sampled TTFT trace;
completion logging therefore emits no second breakdown for the same request.

### Request-level PD transfer metrics

The first prefill replica member reports one `tokenspeed:kv_handoffs_total`
increment at the replica-agreed terminal transition. A timeout awaiting the
decode landing manifest, or rejected registration/preallocation,
is labeled `handshake_failed`; other failures (including cancelled transfers)
are labeled `failed`. Cleanup and per-rank write retries do not add outcomes.
On success, `tokenspeed:prefill_kv_handoff_seconds` starts at the final prompt
handoff and ends when the replica can retire the transfer. Earlier layerwise
writes can overlap compute. The first decode member observes
`tokenspeed:decode_kv_wait_seconds` from local landing-manifest submission
to replica-agreed full KV arrival; time waiting for an execution slot after
KV arrival is measured separately.

## Recurrent-state materialization feedback

The data plane retires replay for snapshot and abort slots before running
the corresponding plan. Releases use the same forward queue and retire any
remaining replay ahead of later plans. The control plane tracks retirement
fences separately from scheduler cache operations. Peer export, cache clear
and drain wait for replica completion, including releases queued after the
last forward's results.
