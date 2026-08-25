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
pools and executor as its own locals, and returns three things split by how
long the caller may hold them:

| | what it is | lifetime |
| --- | --- | --- |
| `DeviceSpecs` | plain values the loop plans with: cache geometry, cache groups, speculation widths, capability flags | keep forever |
| `DeviceWiring` | the startup steps that need a real device object: describe the KV to a PD peer, install the layerwise step counter, read the encoder's model facts | a local of `__init__`, dropped when it returns |
| `DeviceHandle` | the running handle: the complete list of what the loop may ask of the device side | the only one stored (`self._device`) |

Consequences:

* The loop cannot **name** a model runner, backend or KV pool, so it cannot
  pass one implicitly or mutate one a forward is still using. A test asserts
  this over the AST of `EventLoop.__init__` — locals included, because a name
  it can write is a name a later change can keep.
* A new device interaction goes on `DeviceWiring` if it runs once at startup,
  and on `DeviceHandle` only if the running loop genuinely needs it — the
  second widens what the loop can do to the GPU mid-flight. The wiring list is
  the one that grows (each new cache tier, transport or accelerator wants a
  hook that needs a pool); sending that growth to an object the loop does not
  hold is the point of the split. Hand over the capability, never the object.
* Collaborators are **given** the handle in their constructor. Do not reach it
  through another object (`loop._device...`): a traversal is the seam the next
  change widens, first to the handle and then to whatever it exposes.
* On the per-round path only commit waits, through `PendingExecution.result()`:
  join the forward thread's future (launches issued), then its copy event (D2H
  landed). Dispatch never waits, which is what keeps a backpressured stage off
  the control plane. The handle's other `run_*` methods do block, deliberately
  — the DP idle forward, the PD receive and landing, the KV repair after a
  wake, the RL weight updates — but each is a low-rate path whose caller
  cannot proceed without the result. A new blocking method on the per-round
  path is a bug.

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
overlap-breaking dependencies (currently: the P-side PD handoff batch and
eager-grammar batches). New rules go there, not into `event_loop`. Rounds
that run no real forward (pause/freeze, DP idle) drain the queue fully.

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
| `_epd_hooks`   | `EpdPrefillHooks` — `epd/prefill_hooks.py`    | glue (EpdPrefillAdmission decides)          | `try_stage`, `drain_ready_embeddings` (and `assert_embeddings_received`, from the P-role dispatcher) |
| `_pd_hooks`    | `PdTransferHooks` — `pd/transfer_hooks.py`    | glue (transfer executors decide)            | `poll_transfer_events` |
| `_cache_hooks` | `L2CacheHooks` — `engine/cache_hooks.py`      | glue-ish (handed the `DeviceHandle`: submission is GPU work; polling stays control-side event queries) | `submit`, `poll_ready_events` |

`_pause_hooks` and `_pd_hooks` are also handed the `DeviceHandle`: both have
work that must land on the data plane — the DP idle forward and the KV repair
after a memory-saver wake, and the device writes a completed remote prefill
lands. `PauseHooks` additionally supplies `reset_caches_for_release` and
`kv_repair_after_wake` to the memory-occupation controller as callbacks; those
are not loop entry points, they fire on release/wake.

Per-round dispatch is not a hooks class but follows the same rule:
`ForwardDispatcher` and its PD subclasses (`engine/forward_dispatch.py`) are
control plane, chosen once per engine role, and hold the handle rather than
anything behind it.

Related placements that follow the same principle without a hooks class: the
SMG startup handshake lives in `zmq_msgpack.connect_msgpack_engine_for_loop`
(wire-schema helpers in `zmq_wire`), multimodal batch-context assembly in
`multimodal/inputs.py::multimodal_context_for_forward` (which also snapshots
each request's multimodal inputs, per Principle 1's capture contract), and
P-side layerwise KV streaming setup in
`DisaggPrefillExecutor.setup_layerwise_transfer` (which takes a `DeviceWiring`
and gets the step counter back, rather than reaching into attention backends).

All hooks obey Principle 3: they return events or decisions; they never call
`advance_scheduler`.

## Anatomy of a round

For orientation, one iteration of `event_loop`:

1. Receive and admit new requests (`_process_new_requests`), with the pause
   and EPD admission hooks inline as single lines.
2. Poll completed L2 cache ops; **advance the scheduler (head call site)** so
   this round's plan sees them.
3. Frozen (`PAUSED_ALL`)? Drain the in-flight queue and run the paused idle
   step. Otherwise: plan (`next_execution_plan`), submit the round's page
   zeroing and then its L2 cache transfers to the data plane (submitted, not
   awaited — the FIFO orders the zeroing before this round's forward, and
   before the host-cache loads that may overwrite the same pages), derive the
   forward op, record metrics, and DP-sync (running an idle forward on idle
   ranks).
4. Non-idle rounds: gather per-batch state, drain the in-flight queue if the
   dispatch depends on a pending commit (Principle 4), dispatch, commit from
   the queue head down to the effective depth, and poll PD transfer events.
5. **Advance the scheduler (tail call site)** with the round's
   `request_changes`, publish KV events (once), and resolve any pending
   pause/release drain.

## Checklist for extending the loop

* New logic that reacts to scheduler/transfer/cache progress: put it in the
  matching hooks class (or add one), return events, and apply them at an
  existing advance point.
* New device interaction: `DeviceWiring` if it is startup-only, `DeviceHandle`
  if the running loop needs it — and inject the handle into whoever needs it
  rather than traversing to it.
* New reason a dispatch cannot overlap a pending commit: add it to
  `_dispatch_depends_on_pending_commit`.
* New per-round work: add a single-line hook call at a fixed position in the
  loop (rank-identical across ranks if it contains collectives), not a
  branch of feature code.
* Never call `scheduler.advance`, `advance_scheduler`, or the KV event
  publisher from a helper or hooks class.
* Never issue CUDA work, or hold something that can, from the control plane.
