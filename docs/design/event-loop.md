# Event Loop: Design Principles

This document records the design principles of the scheduler event loop
(`python/tokenspeed/runtime/engine/event_loop.py`). It is the reference for
where new logic belongs, how components feed results back into the scheduler,
and what the loop body itself is allowed to contain. The rules below were
established deliberately; treat deviations as bugs in review.

## Principle 1: the event loop is a coordinator, nothing more

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

## Principle 2: scheduler feedback is explicit and centralized

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

## Principle 3: correctness never depends on the in-flight depth

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

## Principle 4: publishing drains, once per round

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
  controller the request handler or executor drives. The controller DECIDES;
  the hooks ACT with the loop's collaborators.
* **Self-contained components** own their state outright and depend only on
  static configuration — they need no loop reference at all. Prefer this
  shape whenever the subsystem doesn't genuinely need the loop's live state.

Current inventory:

| Attribute      | Class / home                                  | Shape          | Loop entry points |
| -------------- | --------------------------------------------- | -------------- | ----------------- |
| `_pause_hooks` | `PauseHooks` — `engine/pause.py`              | glue (PauseController is the state machine) | `apply_transitions`, `withhold_admissions`, `paused_idle_step` |
| `_epd_hooks`   | `EpdPrefillHooks` — `epd/prefill_hooks.py`    | glue (EpdPrefillAdmission decides)          | `try_stage`, `drain_ready_embeddings`, `assert_embeddings_received` |
| `_pd_hooks`    | `PdTransferHooks` — `pd/transfer_hooks.py`    | glue (transfer executors decide)            | `poll_transfer_events` |
| `_cache_hooks` | `L2CacheHooks` — `engine/cache_hooks.py`      | self-contained (static config only)         | `submit`, `poll_ready_events` |

Related placements that follow the same principle without a hooks class: the
SMG startup handshake lives in `zmq_msgpack.connect_msgpack_engine_for_loop`
(wire-schema helpers in `zmq_wire`), multimodal batch-context assembly in
`multimodal/inputs.py::multimodal_context_for_forward`, and P-side layerwise
KV streaming setup in `DisaggPrefillExecutor.setup_layerwise_transfer`.

All hooks obey Principle 2: they return events or decisions; they never call
`advance_scheduler`.

## Anatomy of a round

For orientation, one iteration of `event_loop`:

1. Receive and admit new requests (`_process_new_requests`), with the pause
   and EPD admission hooks inline as single lines.
2. Poll completed L2 cache ops; **advance the scheduler (head call site)** so
   this round's plan sees them.
3. Frozen (`PAUSED_ALL`)? Drain the in-flight queue and run the paused idle
   step. Otherwise: plan (`next_execution_plan`), derive the forward op,
   record metrics, and DP-sync (running an idle forward on idle ranks).
4. Non-idle rounds: gather per-batch state, drain the in-flight queue if the
   dispatch depends on a pending commit (Principle 3), dispatch, commit from
   the queue head down to the effective depth, and poll PD transfer events.
5. **Advance the scheduler (tail call site)** with the round's
   `request_changes`, publish KV events (once), and resolve any pending
   pause/release drain.

## Checklist for extending the loop

* New logic that reacts to scheduler/transfer/cache progress: put it in the
  matching hooks class (or add one), return events, and apply them at an
  existing advance point.
* New reason a dispatch cannot overlap a pending commit: add it to
  `_dispatch_depends_on_pending_commit`.
* New per-round work: add a single-line hook call at a fixed position in the
  loop (rank-identical across ranks if it contains collectives), not a
  branch of feature code.
* Never call `scheduler.advance`, `advance_scheduler`, or the KV event
  publisher from a helper or hooks class.
