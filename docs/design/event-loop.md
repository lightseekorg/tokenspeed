# Event Loop: Design Principles

This document records the design principles of the scheduler event loop
(`python/tokenspeed/runtime/engine/event_loop.py`). It is the reference for
where new logic belongs, how components feed results back into the scheduler,
and what the loop body itself is allowed to contain. The rules below were
established deliberately; treat deviations as bugs in review.

## Principle 1: the loop runs no GPU work, and cannot reach any

The event loop is the **control plane**: ZMQ input, gloo collectives, the C++
scheduler, commit post-processing. `ForwardThread`
(`execution/forward_thread.py`) is the **data plane**: one thread per rank,
FIFO, everything that touches CUDA. A control-plane round is microseconds, so
the cross-rank collectives that keep the redundant schedulers aligned always
find every rank promptly, however deep the GPUs are in queued work — a stage's
launch-queue backpressure stalls only its own forward thread, never the round.

This is enforced by **visibility**, not by discipline. `build_device_side`
(`execution/device.py`) constructs the model runners, attention backends, KV
pools and executor as its own locals, and returns one `DeviceBuild`, split by
how long the caller may hold each piece:

| | what it is | lifetime |
| --- | --- | --- |
| `DeviceSpecs` | plain values the loop plans with: cache geometry, cache groups, speculation widths, capability flags | keep forever |
| `DeviceHandle` | the running handle: the complete list of what the loop may ask of the device side | the only one stored (`self._device`) |
| `transfer` | the PD peer's CONTROL face — bootstrap register/abort, event polling — or None outside PD | held by the PD hooks |
| `encoder_model_facts` | a callable resolving the encoder facts EPD admission needs (raises on text-only) | consumed at startup, past the EPD gate |

The device side is built **complete**, not built-then-wired: the transfer peer
is constructed inside the builder (everything it needs is `server_args` or the
KV pool the builder already owns) and the engine's role is read off it once, at
construction. An earlier shape had the loop assemble the peer and hand it back
through a setter, which left the role mutable after startup for no reason.

The handle owns BOTH executors of a scheduler plan, and treats their work
identically: a model forward runs asynchronously on the GPU, and handing a
prefill or decode to the peer node is asynchronous work of exactly the same
standing. The plan separates its streams by executor — the `ForwardBatch`
is the model's work, `plan.remote_prefill` is the peer's on a D node (pull
the admitted prompt's KV in), `plan.remote_decode` the peer's on a P node
(the completed prompt decodes over there, so its KV goes out). The remote
streams ride beside whatever forward work the round schedules, occupy no
batch slot, and go out even on rounds with no batch at all — everything
dispatchable dispatches in one round. The transfer moves
KV-pool device memory over RDMA rather than through a CUDA kernel, but it
needs the same ordering against forwards and page zeroing — so its execution
face lives behind the handle too, attached once at startup. Its control face
(bootstrap register/abort, event polling, `pop_*`) stays on the control
plane, where it feeds Principle 3's tail advance.

Consequences:

* The loop cannot **name** a model runner, backend or KV pool, so it cannot
  pass one implicitly or mutate one a forward is still using. A test asserts
  this over the AST of `EventLoop.__init__` — locals included, because a name
  it can write is a name a later change can keep.
* A new device interaction belongs **inside the builder** if it runs once at
  startup, and on `DeviceHandle` only if the running loop genuinely needs it —
  the second widens what the loop can do to the GPU mid-flight. Hand over the
  capability, never the object.
* Every method on the handle is a **named operation**, with one registered
  exception: `run_multimodal_work`, because multimodal feature lifecycle is a
  state machine reached from several control-plane points (EPD admission's
  stage/drain device half; the commit-side SHM release). A generic "run this
  closure" slot is the hole this whole design closes, so a second KIND of
  user does not join it — it gets its own name.
* The role is a **value** (`DeviceRole`), not a class hierarchy. Subclassing
  per role forced the handle to publish its own internals so the subclasses
  could call back into it — a reference cycle for about a dozen lines of
  difference. With every remote op on its own plan stream, the batch needs
  no per-role reading at all: a `ForwardBatch` is model work, full stop, and
  a DP rank counts work by simply asking whether the batch has tokens.
* Collaborators are **given** the handle in their constructor. Do not reach it
  through another object (`loop._device...`): a traversal is the seam the next
  change widens, first to the handle and then to whatever it exposes.
* On the per-round path only commit waits, through `PendingExecution.result()`:
  join the forward thread's future (launches issued), then its copy event (D2H
  landed). Dispatch never waits — model forwards and the peer's remote
  prefills/decodes alike are submitted fire-and-forget, which is what keeps a
  backpressured stage off the control plane. The remote submissions' only
  failure surface is a settle at the next round's `execute` (their semantic
  completion arrives through the transfer events; a submission that RAISED
  produces none, so it must not be swallowed). The handle's remaining `run_*`
  methods do block, deliberately — the DP idle forward, the landing of a
  completed remote prefill, the KV repair after a wake, the RL weight sync —
  but each is a low-rate path whose caller cannot proceed without the result
  (the landing's failure must surface BEFORE the scheduler advances the
  request into decode). A new blocking method on the per-round path is a bug.

The rule is also mechanically enforced, on by default: a thread-local
dispatch mode over the loop raises on any CUDA tensor op run from the
control thread. Event waits — the inbound channel — and metadata-only view
ops pass untouched; neither submits device work.
``TOKENSPEED_GUARD_CONTROL_PLANE=0`` is the escape hatch for a deployment
that trips on an unrouted op (report it). EPD prefill admission, the one
known violator, now crosses through ``DeviceHandle.run_multimodal_work``:
receive-buffer allocation, the publish clone/scatter, and the NCCL shard
reassembly all run on the forward thread — which also puts the reassembly
broadcasts on the same issuing thread as the model's collectives, restoring
the cross-rank launch-order guarantee the old non-overlap-loop assumption
provided.

### The capture contract

Information crosses to the data plane **only** inside the submitted closure,
and is frozen once submitted: no attribute rebinding, no in-place edit, no
releasing a resource the closure captured. Capture plain values or a snapshot,
and bind at capture time rather than closing over a variable the caller will
rebind. Results cross back **only** through `PendingExecution.result()`.

`execution/forward_thread.py` states this in full, including the single
registered exception — grammar matchers, whose ownership is split by path and
whose overlap is instead broken by the drain registry in Principle 4.

## Principle 2: the event loop is a coordinator, nothing more

`EventLoop.event_loop` sequences components; it does not implement them.
Domain logic — pause/resume semantics, EPD admission, PD transfer handling,
L2 cache-op tracking, wire handshakes, multimodal batch assembly — lives in
its own module and enters the loop as a **single-line hook**. The loop body
should read, top to bottom, as the schedule of one scheduling round, with no
feature's internals inlined into it.

Consequences:

* Low-frequency or optional features (pause/resume control, EPD, SMG
  transport, kvstore) must never make the *normal* scheduling path harder to
  read. If understanding decode throughput requires skipping over your
  feature's code, the feature is in the wrong place.
* When a feature needs several collaborators of the loop, give it a hooks
  class (see below) instead of weaving branches through the loop and its
  helpers.

## Principle 3: scheduler feedback is explicit and centralized

`advance_scheduler` (`scheduler_utils.py`) is the **only** caller of
`scheduler.advance`, and it is invoked **only explicitly and directly in the
`event_loop` body — never from helpers**. Helpers RETURN their events; the
loop applies them. Reading the loop body alone must reveal every point where
the scheduler's state advances, and why.

There are exactly two call sites, each with a documented reason:

* **Head of the round** — completed L2 cache-op events
  (`_cache_hooks.poll_ready_events()`). These must advance *before*
  `next_execution_plan`, otherwise cache-gated admissions are delayed by a
  full round.
* **Tail of the round** — forward results and PD transfer events, funneled
  through the single `request_changes` list. These can only exist after
  dispatch/commit, and must advance before the *next* round plans.

Anything that produces scheduler events (a new transfer backend, a new async
op kind) either returns events into one of these two points or adds a new
explicit call site in the loop body with a comment stating why the existing
points don't fit. It must not call `advance_scheduler` itself.

## Principle 4: correctness never depends on the in-flight depth

The loop is parameterized by `in_flight_depth`: 0 (classic synchronous
commit), 1 (the overlap schedule), or `pp_size` (the prefill chunk pipeline).
Dispatched forwards await commit in the `in_flight` queue; the tail commits
once the queue exceeds the effective depth (0 when the round dispatched no
new work, so results never wait on future traffic).

The depth is a performance knob only. Any dispatch whose inputs depend on a
pending commit's side effects must drain the queue first, and
`_dispatch_depends_on_pending_commit` is the **single registry** of those
overlap-breaking dependencies (currently: eager-grammar batches). New rules go
there, not into `event_loop`. Rounds that run no real forward (pause/freeze,
DP idle) drain the queue fully.

Prefer removing a dependency over registering one. The P-side remote decode
was registered here until the C++ scheduler learned to hold it until its
final chunk's forward result lands (it now rides `plan.remote_decode`,
beside the batch): a request turns `PrefillDone` when its last chunk is
*scheduled*, so the transfer was being planned while that chunk was still in
flight, and satisfying it meant draining the whole queue — under PP,
emptying the chunk pipeline every time a prompt finished prefill. The
dependency was real; the right fix was upstream, not a drain.

Depth ≥ 1 also means a round is planned *before* the previous round's commit,
so a batch can contain a request that commit is about to finish. Anything the
control plane frees on that commit — a request's shared multimodal features,
for instance — must be released through the handle so the FIFO orders it
behind the forward that captured it, not inline.

## Principle 5: publishing drains, once per round

`_publish_scheduler_kv_events` has drain semantics: KV events accumulate
inside the C++ scheduler across any number of mutations (advance,
`next_execution_plan`), so a single unconditional call at the loop tail
publishes everything the round produced, in order, as one batch. Do not add
per-mutation publish calls; they only fragment batches.

The same reasoning fixes the metrics call: scheduler iteration metrics are
recorded once per round, from the same pre-dispatch snapshot as the
scheduler stats.

## The hooks pattern

Loop-side integration of a subsystem is a small class whose methods are the
subsystem's only entry points from the loop. Two shapes exist:

* **Glue hooks** hold a loop back-reference and act on its collaborators;
  they are stateless (or nearly so) because the real state machine lives in a
  controller the request handler or device drives. The controller DECIDES;
  the hooks ACT with the loop's collaborators. Any capability they need — the
  `DeviceHandle` above all — is injected in their constructor, per Principle 1.
* **Self-contained components** own their state outright and depend only on
  static configuration — they need no loop reference at all. Prefer this
  shape whenever the subsystem doesn't genuinely need the loop's live state.

Current inventory:

| Attribute      | Class / home                                  | Shape          | Loop entry points |
| -------------- | --------------------------------------------- | -------------- | ----------------- |
| `_pause_hooks` | `PauseHooks` — `engine/pause.py`              | glue (PauseController is the state machine) | `apply_transitions`, `withhold_admissions`, `paused_idle_step` |
| `_epd_hooks`   | `EpdPrefillHooks` — `epd/prefill_hooks.py`    | glue (EpdPrefillAdmission decides)          | `try_stage`, `drain_ready_embeddings`, `assert_embeddings_received` |
| `_pd_hooks`    | `PdTransferHooks` — `pd/transfer_hooks.py`    | glue (transfer executors decide)            | `poll_transfer_events` |
| `_cache_hooks` | `L2CacheHooks` — `engine/cache_hooks.py`      | glue-ish (handed the `DeviceHandle`: submission rides `execute`; polling stays control-side event queries) | `count_plan_ops`, `poll_ready_events` |

`_pause_hooks` and `_pd_hooks` are also handed the `DeviceHandle`: both have
work that must land on the data plane — the DP idle forward and the KV repair
after a memory-saver wake, and the device writes a completed remote prefill
lands. `PauseHooks` additionally supplies `reset_caches_for_release` and
`kv_repair_after_wake` to the memory-occupation controller as callbacks; those
are not loop entry points, they fire on release/wake.

Per-round dispatch needs no hooks class at all: the loop hands
`DeviceHandle.execute` the plan and the round's `PlannedForward`, and the
plan's streams already say who runs what — the batch is the model's, the
remote streams are the transfer peer's.

Related placements that follow the same principle without a hooks class: the
SMG startup handshake lives in `zmq_msgpack.connect_msgpack_engine_for_loop`
(wire-schema helpers in `zmq_wire`), multimodal batch-context assembly in
`multimodal/inputs.py::multimodal_context_for_forward` (which also snapshots
each request's multimodal inputs, per Principle 1's capture contract), and
P-side layerwise KV streaming setup, which happens inside the device builder
(the step counter is backend surgery; the sender just receives it). Dest-
contiguous CachePD fragments are 2D-packed in the Prefill transfer path
before Mooncake WRITE — that CUDA copy is data-plane work, not loop work.

All hooks obey Principle 3: they return events or decisions; they never call
`advance_scheduler`.

## Anatomy of a round

For orientation, one iteration of `event_loop`:

1. Receive and admit new requests (`_process_new_requests`), with the pause
   and EPD admission hooks inline as single lines.
2. Poll completed L2 cache ops; **advance the scheduler (head call site)** so
   this round's plan sees them.
3. Frozen (`PAUSED_ALL`)? Drain the in-flight queue and run the paused idle
   step. Otherwise: plan (`next_execution_plan`), derive the forward op,
   record metrics, DP-sync, and gather per-batch state (draining the
   in-flight queue first if the dispatch depends on a pending commit,
   Principle 4).
4. **One `DeviceHandle.execute(plan, planned)` call per round**, in an order
   that is itself a correctness contract for same-round page reuse:
   host-cache write-backs first (a retraction's snapshot copy must read the
   reused pages' old bytes), then page zeroing (the new owner's
   sanitization), then load-backs (they target zeroed pages), then the
   plan's remote streams to the transfer peer (a D-node remote prefill
   waits on the zeroing fence inside its submission, which the FIFO orders
   after the write-backs), then the plan's batch to the model. `planned` is
   None on idle and empty rounds; the plan's own work (hygiene, the remote
   streams) still runs. Then commit from the queue head down to the
   effective depth and poll PD transfer events.
5. **Advance the scheduler (tail call site)** with the round's
   `request_changes`, publish KV events (once), and resolve any pending
   pause/release drain.

## Checklist for extending the loop

* New logic that reacts to scheduler/transfer/cache progress: put it in the
  matching hooks class (or add one), return events, and apply them at an
  existing advance point.
* New device interaction: inside `build_device_side` if it is startup-only,
  `DeviceHandle` if the running loop needs it — and inject the handle into
  whoever needs it rather than traversing to it.
* New reason a dispatch cannot overlap a pending commit: add it to
  `_dispatch_depends_on_pending_commit`.
* New per-round work: add a single-line hook call at a fixed position in the
  loop (rank-identical across ranks if it contains collectives), not a
  branch of feature code.
* Never call `scheduler.advance`, `advance_scheduler`, or the KV event
  publisher from a helper or hooks class.
* Never issue CUDA work, or hold something that can, from the control plane.
