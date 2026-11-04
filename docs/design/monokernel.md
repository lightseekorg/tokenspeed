# Decode monokernels

A decode monokernel runs a whole decoder layer, its cross-GPU exchanges
included, as one persistent launch per decode step, with one workgroup
resident on every compute unit. Its workgroups hand data to each other through
memory instead of kernel boundaries, so the next weights stay in flight while
a consumer waits on exactly the data it needs. This document records the
interface between a model and its monokernels and the invariants every
monokernel keeps. A deviation from these rules is a bug unless this document
is updated in the same change.

Implementation details live with the implementations:

- [ops/monokernel/README.md](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/monokernel/README.md):
  the kernel op's interface and its leaves.
- [ops/monokernel/_gluon/README.md](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/monokernel/_gluon/README.md):
  the gfx950 layer kernels (roles, hand-offs, plan fields).
- [gfx950/monokernel/README.md](../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/README.md):
  the gfx950 building blocks (launch entry, polls, fail-stop, the MoE
  stream).

## The layer-stack hook

A model's forward hands its layer loop to a layer stack
(`runtime/models/base/layer_stack.py`), which the model class names in
`layer_stack_cls`:

* The model owns everything around the loop (embedding, row layout, final
  norm, logits). The stack owns which implementation runs each layer in a
  step. The default `DecoderLayerStack` runs every layer in order, as the
  loop it replaces did. A model with its own forward defines the stack that
  its forward calls, with the boundary values its layers pass.
* `prepare(model)` runs once at startup, after the weights load and the
  communication heap exists, and before the KV cache is sized or a graph is
  captured. A stack with monokernels selects and prepares them there; the
  default does nothing.
* `run(model, ...)` runs the layers of one step. A stack with monokernels
  picks, in one method, the contiguous range of layers that runs as
  monokernels, from the step's padded shape and the served set alone. An
  empty range is the explicit fallback: every layer then runs as written.
  The first and last monokernel layers of a range take and return the
  regular layers' boundary values, so the layers outside the range and the
  model around the stack do not change.
* A launch may also prepare part of the next layer, so the stack, not a
  layer, owns the prepared monokernel layers.

## Invariants

### One execution path

* A monokernel layer is another implementation of a decoder layer, chosen
  per step by the model's layer stack. The model's forward is the same for
  every step; only the stack's range differs.
* The choice depends only on the forward's padded row shape (requests,
  tokens per request), on whether the forward carries decode metadata, and on
  the served set agreed at startup. Row count alone is not a shape: a
  one-request verify of 8 tokens and a decode of 8 requests differ. Capture
  records the choice, eager decode makes the same choice for the same shape,
  and every rank decides alike.
* Everything else runs the regular layers unchanged: prefill, extend, mixed
  and idle forwards, row shapes outside the served set, and layers no kernel
  covers. There is no server flag, and no retry on the regular path once a
  monokernel launch has started.

### Selection and agreement before capture

* The stack selects the kernel through the registry
  (`("monokernel", "prepare")`) in `prepare`, which compiles and checks
  every plan variant. Nothing compiles or checks inside a captured graph.
* Each rank's refusals (deployment settings, a variant that does not fit one
  workgroup per compute unit, state the kernel cannot allocate) go through
  `agree_served_shapes` (`runtime/layers/monokernel.py`) over the group's
  CPU group. A refusal on one rank removes the row shape on every rank. The
  startup report lists every declined shape with its rank and reason.
* Communication buffers are allocated whenever static selection (platform,
  traits) picks the kernel, identically on every rank, and the communication
  heap is sized for them; later refusals only shrink the served set.

### Residency

All workgroups of a launch must be resident together: a workgroup that cannot
start leaves the others waiting until the deadline stops the group.
`deployment_decline_reasons` declines the known sources of kernels that can
hold a compute unit during the launch: copies that run as kernels on other
streams (the host KV store's kernel copies, a platform setting that turns
asynchronous copies into kernels), compute-unit masks, a model forward that
forks a side stream, and settings whose state the monokernel cannot keep (the
expert-load recorder, the memory saver). A change that adds kernels on another
stream during decode extends that list in the same change (rather than
inspecting each captured graph).

### Fail-stop

* Every wait inside a launch has a deadline on the device clock, set above
  the longest legal stall of one rank between two graph launches, which the
  regular path tolerates.
* On expiry the waiting workgroup writes a failure record, copies it to a
  host-mapped page that outlives the process, sets a sticky abort word in
  every peer, and traps. A wait that sees an abort word stops the same way
  and names the origin in its record.
* The launcher stops the ranks outside a launch when a rank dies. Recovery
  restarts the whole group; in-flight requests are lost.

### Launch state

* One owner per group holds the launch state (per-workgroup 64-bit epochs,
  claims, entry resets) and the communication regions it indexes. Each
  region's copy is chosen from the epoch, so the regions and the epochs share
  one lifetime.
* The launch state is zeroed once, on every rank, at the first preparation.
  It is not `workspace.zeroed_state` and no graph owns it: recapture, graph
  release and regular steps never touch it, so the epochs stay one monotonic
  sequence across every graph and eager launch.
* The epochs are 64 bits and never wrap; tags, copy parity and stage indices
  derive from them.

### Hand-offs inside a launch

Each implementation's README lists its hand-offs. Every one follows these
rules:

* A producer's payload stores complete before the flag or tag that publishes
  them. A single store that carries the value and its tag needs no flag.
* A consumer that may sit behind a different cache from its producer reads
  the payload so that no cached copy from an earlier launch can answer: with
  loads that bypass the caches the writer cannot update, or from memory no
  cache keeps.
* Tags derive from the epoch and repeat after a bounded number of launches.
  Every slot a launch polls is rewritten (or zeroed) in every launch, or
  lives in two copies by epoch parity with the other copy cleared each
  launch, so no slot keeps a tag old enough to repeat.
* A poll never waits behind loads its own wave issued for later work, unless
  the implementation's plan states why they land first, and the
  implementation checks every such wait before launch.

### Caches and request state

* A monokernel writes the K/V rows of the layer whose query it prepares at
  the slots and in the format `PagedAttention.prologue_target` names, byte for
  byte as `gqa_prologue` would. It is a listed writer in
  [attention-prologue.md](attention-prologue.md).
* It holds nothing per request across steps: its buffers are sized by rows.
  It needs no cache group and no `SlotStateExporter`, and a restored request
  is served at its next step with no warm-up.

## Deployment contract

* Exclusive GPUs: no other process runs kernels on the group's GPUs.
* No kernels on side streams during decode (the host KV store with direct
  copies, or off).
* Profiling: kernel tracing only. Counter collection serializes kernels, and
  instrumentation changes a kernel's resources, so an instrumented build must
  pass the same startup checks.
* A failure stops the whole group within the deadline, with the origin's
  record in the logs.
