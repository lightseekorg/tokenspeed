# Fused decoder layers

This document records the invariants of fused decoder layers: decode steps
where one kernel launch runs a whole decoder layer, its cross-GPU exchanges
included, with one workgroup resident on every compute unit. A deviation from
the rules here is a bug unless this document is updated in the same change.
Kernel structure (roles, phases, exchanges) is documented with the kernel in
`tokenspeed-kernel`.

## The problem this solves

A small decode step runs a few dozen short kernels per layer. Each pays a
launch and leaves the GPU idle while it waits on memory or on a peer GPU. A
persistent launch per layer keeps the next weights in flight while waiting,
and replaces kernel boundaries with waits on exactly the data a consumer
needs. Its workgroups wait on each other, which is correct only when all of
them are resident at once, and its exchanges are correct only when every rank
of the group runs the same launches in the same order. The rules below keep
both true without a second execution path.

## Invariants

### One execution path

* A fused layer is another implementation of a decoder layer, chosen with the
  model's existing per-forward layout. The model loop is unchanged: the layer
  calls its prepared kernel, and the value passed to the next fused layer is
  that layer's prepared query (its K/V rows already in the cache).
* The choice depends only on the forward's padded row shape and on the served
  set agreed at startup. Capture records it, eager decode makes the same
  choice for the same shape, and every rank decides alike.
* Everything else runs the regular layer, unchanged: prefill, extend, MIXED
  and IDLE forwards, row shapes outside the served set, and layers the kernel
  does not cover. There is no server flag, and no retry on the regular path
  once a fused launch started.

### Selection and agreement before capture

* The model selects the kernel through the registry
  (`("decoder_layer", "prepare")`) during startup, once per candidate row
  shape, and the prepare compiles every plan variant. Nothing compiles or
  checks inside a captured graph.
* Each rank's refusals (deployment settings, a variant that does not fit one
  workgroup per compute unit, state the kernel cannot allocate) go through
  `agree_served_rows` (`runtime/layers/fused_decoder_layer.py`) over the
  group's CPU group. A refusal on one rank removes the row shape on every
  rank. The startup report lists every declined shape with its rank and
  reason.
* Communication buffers are allocated whenever static selection (platform,
  traits) picks the kernel, identically on every rank; later refusals only
  shrink the served set.

### Residency

All workgroups of a fused launch must be resident together. Anything that can
hold a compute unit while the launch dispatches turns into a fail-stop of the
group. `deployment_decline_reasons` declines the known sources: host KV store
copies run as kernels, `HSA_ENABLE_SDMA=0`, CU masks, a model forward that
forks a side stream, and the settings whose state the fused path cannot keep
(the expert-load recorder, the memory saver). A change that adds kernels on
another stream during decode must extend that list in the same change.

### Fail-stop

* Every wait inside a fused launch has a deadline (60 s), checked on the
  device clock. It is chosen from the longest legal stall of one rank between
  two graph launches, not from the measured skew: the regular path tolerates
  such stalls, and the fused path must not turn them into a crash.
* On expiry the waiting workgroup writes a failure record, copies it to a
  host-mapped page that outlives the process, sets a sticky abort word in
  every peer, and traps. A wait that sees an abort word stops the same way and
  keeps the origin in its record.
* Ranks outside a fused launch are stopped by the launcher when a rank dies.
  Recovery is a restart of the whole group; in-flight requests are lost.

### Launch state

* One owner per group holds the launch state (per-workgroup 64-bit epochs,
  claims, entry resets) and the communication regions it indexes. Each
  region's copy is selected from the epoch, so the regions and the epochs
  share one lifetime.
* The launch state is zeroed once, on every rank, at the first preparation.
  It is not `workspace.zeroed_state` and not owned by a graph: recapture,
  graph release and regular steps never touch it, so epochs stay one
  monotonic sequence across every graph and eager launch.
* The epochs are 64 bits and never wrap; tags, copy parity and stage indices
  derive from them.

### Caches and request state

* A fused layer writes the K/V rows of the layer it prepares at the slots and
  in the format `PagedAttention.prologue_target` names, byte for byte as
  `gqa_prologue` would. It is a listed writer in
  [attention-prologue.md](attention-prologue.md).
* It holds nothing per request across steps: its buffers are sized by rows.
  It needs no cache group and no `SlotStateExporter`, and a restored request
  is served at its next step with no warm-up.

## Deployment contract

* Exclusive GPUs: no other process runs kernels on the group's GPUs.
* No kernels on side streams during decode (the host KV store with the
  `direct` backend or off; SDMA copies on).
* Profiling: kernel tracing only. Counter collection serializes kernels, and
  instrumentation changes a kernel's resources, so an instrumented build must
  pass the same startup checks.
* A failure stops the whole group within the deadline, with the origin's
  record in the logs.
