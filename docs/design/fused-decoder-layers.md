# Fused decoder layers

This document records the invariants of fused decoder layers: decode steps
where one kernel launch runs a whole decoder layer, its cross-GPU exchanges
included, with one workgroup resident on every compute unit. A deviation from
the rules here is a bug unless this document is updated in the same change.
The kernel op's interface is in
[decoder_layer/README.md](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/decoder_layer/README.md);
the gfx950 implementation (roles, hand-offs, plan fields) in
[decoder_layer/_gluon/README.md](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/decoder_layer/_gluon/README.md).

## The problem this solves

A small decode step runs a few dozen short kernels per layer. Each pays a
launch and leaves the GPU idle while it waits on memory or on a peer GPU. A
persistent launch per layer keeps the next weights in flight while waiting,
and replaces kernel boundaries with waits on exactly the data a consumer
needs. Its workgroups wait on each other, which is correct only when all of
them are resident at once, and its exchanges are correct only when every rank
of the group runs the same launches in the same order. The rules below keep
both true without a second execution path.

## The layer-stack hook

A model's forward hands its layer loop to a layer stack
(`runtime/models/base/layer_stack.py`), chosen at construction through the
model class's `layer_stack_cls`:

* The model owns everything around the loop (embedding, row layout, final
  norm, logits); the stack owns which implementation runs each layer for a
  step. The default `DecoderLayerStack` runs every layer in order, exactly as
  `BaseTransformerModel` was written; a model with its own forward defines
  the stack its forward calls, with the boundary values its layers pass.
* `prepare(model)` runs once at startup, after the weights are loaded and
  the communication heap exists, before the KV cache is sized or a graph is
  captured. A stack with fused layers selects and prepares them there; the
  default does nothing.
* `run(model, ...)` runs the layers for one step. A stack with fused layers
  decides in one method, from the step's padded shape and the served set
  alone, which contiguous range of layers runs fused; an empty range is the
  explicit fallback, and every layer then runs as written. Within a range the
  first fused layer takes its query from its attention's regular prologue,
  and the last ends with the next layer's input norm, so the layers before
  and after the range and the model around the stack are unchanged.
* A fused range may span several layers (each launch prepares the next
  layer's query), so the stack, not a layer, owns the prepared fused layers.

## Invariants

### One execution path

* A fused layer is another implementation of a decoder layer, chosen per
  step by the model's layer stack. The model's forward is the same for every
  step; only the stack's range differs.
* The choice depends only on the forward's padded row shape, (requests,
  tokens per request), on whether the forward carries decode metadata, and on
  the served set agreed at startup. Row count alone is not a shape: a
  one-request verify of 8 tokens and a plain decode of 8 requests differ.
  Capture records the choice, eager decode makes the same choice for the same
  shape, and every rank decides alike.
* Everything else runs the regular layers, unchanged: prefill, extend, MIXED
  and IDLE forwards, row shapes outside the served set, and layers the kernel
  does not cover. There is no server flag, and no retry on the regular path
  once a fused launch started.

### Selection and agreement before capture

* The stack selects the kernel through the registry
  (`("decoder_layer", "prepare")`) during `prepare`, and the prepare compiles
  every plan variant. Nothing compiles or checks inside a captured graph.
* Each rank's refusals (deployment settings, a variant that does not fit one
  workgroup per compute unit, state the kernel cannot allocate) go through
  `agree_served_shapes` (`runtime/layers/fused_decoder_layer.py`) over the
  group's CPU group. A refusal on one rank removes the row shape on every
  rank. The startup report lists every declined shape with its rank and
  reason.
* Communication buffers are allocated whenever static selection (platform,
  traits) picks the kernel, identically on every rank, and the communication
  heap is created with room for them; later refusals only shrink the served
  set.

### Residency

All workgroups of a fused launch must be resident together. Anything that can
hold a compute unit while the launch dispatches turns into a fail-stop of the
group. `deployment_decline_reasons` declines the known sources: copies that
run as kernels on other streams (the host KV store's kernel copies, a
platform setting that turns asynchronous copies into kernels), compute-unit
masks, a model forward that forks a side stream, and the settings whose
state the fused path cannot keep (the expert-load recorder, the memory
saver). A change that adds kernels on another stream during decode must
extend that list in the same change. Production declines on these known
sources rather than inspecting each captured graph, which would need graphs
kept for inspection on the serving path.

### Fail-stop

* Every wait inside a fused launch has a deadline, checked on the device
  clock. It is set from the longest legal stall of one rank between two graph
  launches: the regular path tolerates such stalls, and the fused path must
  not turn them into a crash.
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

### Hand-offs inside a launch

Workgroups hand data to each other through memory; the implementation's
README lists each hand-off. Every one follows these rules:

* A producer's payload stores complete before the flag or tag that publishes
  them. A single store that carries the value and its tag together needs no
  flag.
* A consumer that may sit behind a different cache from its producer reads
  the payload so that no cached copy of an earlier launch can answer: loads
  that bypass caches the writer cannot update, or memory no cache keeps.
* Tags derive from the epoch and repeat after a bounded number of launches.
  Every slot a launch polls is rewritten (or zeroed) in every launch, or lives
  in two copies by epoch parity with the other copy cleared each launch, so
  no slot keeps a tag old enough to repeat.
* A poll never waits behind loads its own wave issued for later work unless
  the implementation's plan states why they land first; the plan checker
  checks every such wait.

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
* No kernels on side streams during decode (the host KV store with direct
  copies or off).
* Profiling: kernel tracing only. Counter collection serializes kernels, and
  instrumentation changes a kernel's resources, so an instrumented build must
  pass the same startup checks.
* A failure stops the whole group within the deadline, with the origin's
  record in the logs.
