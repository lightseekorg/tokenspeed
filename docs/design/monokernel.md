# Decode monokernels

A decode monokernel runs a whole decoder layer's decode step as one
persistent launch. Every stage of the layer (norms, projections, attention,
MoE) becomes a set of tasks placed on fixed CTAs. Stages hand values to each
other through tagged mailboxes in a scratch buffer, not through kernel
boundaries. Latency, not FLOPs, limits small-batch decode. One launch per
layer removes many kernel boundaries, and each stage can stream its weights
while earlier stages still compute.

## Where the pieces live

| Layer | Path | Holds |
|---|---|---|
| Kernel boundary | `tokenspeed-kernel/python/tokenspeed_kernel/ops/model/<model>/monokernel/` (see its `README.md`) | The ops that a model calls: `prepare_*` and `*_decode`, through `select_kernel`. `prepare_*` packs a layer once after loading and returns `None` when no kernel serves the layer. The README gives the API, the steps to add a model, and links to each vendor's README. |
| Vendor kernels | The vendor's kernel package, under `ops/<arch>/monokernel/` | The building blocks that every model's kernels share (plans, mailboxes, GEMVs, trace), and one directory for each model's stages and layer kernels. Its README describes the files, the rules for writing a stage, and the performance workflow. |
| Runtime seam | `python/tokenspeed/runtime/models/monokernel.py`, and `<model>_monokernel.py` next to it | The shared part: the layer mixin that selects the launch or the regular layer's code, the state that all launches share (the scratch and the step counter), and the startup error. For each model: its monokernel layer classes (subclasses of the regular layers), the packing from the model's modules with the reasons of a layer that cannot be packed, and the launches. |

## Invariants

**A forward launches every layer or none.** A monokernel layer takes the
regular decoder layer's arguments and returns the layer's output with no
pending state. Startup packs every layer, and whether a layer launches depends
only on the forward (its mode and its rows), not on the layer. So within one
forward every layer launches or none does, and a launch gets the previous
launch's complete output, never a regular layer's pending state. Across
forwards, launches and regular forwards follow each other freely through the
shared caches (next rule). Each model's kernel README gives its boundary
tensors.

A model's boundary layers may keep the regular code in every forward, when
their hand-over to the embedding or the LM head differs from the other
layers'. The model's kernel README names them and why.
Every other layer launches or none does, as above. The first launch after a
boundary layer takes that layer's complete output, and the last launch before
one returns the regular layer's boundary, so no pending state crosses
between a launch and a regular layer. A draft model is a backbone of its
own and follows the same rule as the target: its layers, and its other
launches (a context write, a draft head), are all packed at startup or
startup fails (see **One switch** below).

**Same caches, same layouts.** A monokernel reads and writes the KV cache,
the recurrent state pages, and the other per-request caches exactly as the
regular decode path does. Prefill, the regular decode path, and the monokernel
can follow each other on one request. The current token's own values (such as
its k and v) reach later stages through mailboxes, not through the cache.
A cache write by another CTA in the same launch need not be visible yet (the
caches need not be coherent between CTAs).

**Same math, not the same bits.** A monokernel follows the regular path's
numerics: the same activation quantization for each projection, and rounding
where the regular path's kernels round. Summation orders differ, and so do the
split boundaries of a split softmax, so results are not bitwise equal. Tests
compare each stage with a torch reference of TokenSpeed's math, with tight
tolerances per stage and looser ones per layer.

**One switch, monokernel layer classes.** `--decode-monokernel` is the one
switch for every model. With it, the model's `get_layer` builds each decoder
layer from its monokernel class (`decoder_layer_cls`), a subclass of the
regular layer class with the same parameters, so weight loading and cache
planning do not change. A monokernel layer runs a decode step that its plans
serve as one launch, and every other forward (prefill, larger steps) with the
regular layer's code. Startup fails when the checkpoint's architecture has
no monokernel (`DECODE_MONOKERNEL_ARCHITECTURES` in `server_args.py`), or
when a layer other than a boundary layer cannot be packed after loading. The
switch never serves such a layer without its kernel, and the draft model of a
speculative target follows the same rule: when its layers, context write or
head cannot be packed, startup fails and names each piece and why. The
model's layer loop is the same with and without the switch.

**Deadlock freedom comes from the plan.** Each stage reads only regions that
an earlier stage writes, and the plan checks this when it is constructed.
Every CTA runs its tasks in plan order, and all CTAs are resident (at most one
per compute unit). So a poll waits only on work that some CTA reaches without
waiting on the poller. A task may issue loads for its own weights before it
polls, but it never blocks on a later stage.

**Tags make scratch reuse safe.** A mailbox pair is `(payload, tag)`, and the
tag encodes the step and the layer. The kernel stores and loads each pair as
one single-copy-atomic 64-bit access that no non-coherent cache serves. The
step counter is a device tensor that changes once per forward, inside the
CUDA graph, after the forward's last fused layer. So no write from an earlier
layer or step can satisfy a poll. All layers share the scratch buffer, and
nothing clears it.

**A tensor-parallel group runs one plan.** In a plan of several ranks, every
rank runs the same stages, and a stage can put into another rank's copy of
a peer region. The deadlock rule then holds across ranks: the earliest
unfinished stage on any rank reads only regions that earlier stages on every
rank wrote, and all CTAs of all ranks are resident. The scratch lives on a
symmetric heap, so a region has the same offset on every rank. The scratch is
reserved before the heap exists, and the reservations set the heap's
headroom, so a model that runs monokernels also reserves every other buffer
it later takes from that headroom; otherwise those buffers have only the
small margin. A rank can
start its next launch while another still reads this one, but not two
launches ahead, because finishing the next launch needs every rank's part of
it. So a peer region holds two copies, which the launches pick by a parity
that alternates on every rank. Puts and polls of a peer region use system
scope: another GPU's store does not update the local caches.

**Bulk values go as plain words and one flag.** A tagged pair carries 32 bits
of payload in 64. A producer of many values per task may instead store them
as plain words past the caches, wait until its stores land, and then put one
tagged flag. The consumer polls the flag and then loads the words past the
caches. The tag still guards reuse: a word is read only after this launch's
flag. The choice is a stage's: plain words halve the bytes a consumer
reads, and the producer waits for its stores to land before the flag.

**A step starts and ends in the model.** One step is one forward. The model
calls `start_step()` on its monokernel state before its layers and
`finish_step()` after them. The values that cross layers within a step live in
that state, not in the layers: whether a layer launched, the LM head input
when the last launch also runs the model's final norm or mixer, and a model's
own values (such as the verified lengths of a draft's first step).
`finish_step()` moves the step counter when a layer launched and no launch
moved it, so a step without launches adds no kernel. The regular path needs
none of this: its layers pass values in their return values and through
kernel boundaries.

**Polls throttle their loads.** Many threads spinning on poll loads can starve
the producers' stores. A poll issues all its loads, waits once, and retries
after a short sleep. A retry loads only the pairs whose tag differs, so a wide
poll sees its last pairs one round trip after they land.

Flag polls use `gl.atomic_poll` with the same time bound and fail-stop. They
wait for the whole pair, `(tag << 32) | 1`, not the tag alone, so the plan
names its peer flag regions (`Plan.flags`), and a host that stands in for
absent ranks (the layer bench) writes those pairs as raised flags. A stage
that waits for a flag together with data pairs keeps one tagged poll for
both: two polls in a row cost a round trip on its critical path.

**A stuck poll stops the process.** A poll stops the kernel with an error
after a bounded wait, which is far longer than a launch. A deadlock or a
missing producer then gives a GPU error and the process stops, instead of a
hang. The bound is time, not a retry count: in a
tensor-parallel group, a rank whose host submits a launch seconds late must
only stall its peers. Given a stuck-poll record buffer, a poll that gives up
first writes what it waited for (the tag, the pair, what it saw, the clocks)
where the host can read it after the process stops. A stall leaves records
on every CTA downstream of it; the decoder groups them by wait and prints the
earliest stage in plan order first, which by the deadlock rule is the one
whose producer is missing.

**A launch first reads its own code.** A layer kernel's code can be larger
than the instruction cache, and the weights that stream through the caches
evict it during a launch. Each stage would then wait on its instructions
from memory, one stage after another on the critical path. So a launch's
first tasks may read its own code into the caches as data, and stages that
run once a launch keep compute-light loops rolled, not unrolled.

**Compile-time plans are canonical text.** A plan is a compile-time argument
of the kernel, and Triton's compile hooks serialize every compile-time
argument to JSON. So a kernel takes its plan as its canonical JSON text, never
as a Python object. One plan per row count is a static value, as the
compile-time parameter rule in `AGENTS.md` requires.

**Operands travel in bundles.** Each stage module declares bundles of the
operands its stages read. A kernel takes one argument per bundle and builds
the bundle at its top. The argument is empty when the plan has no stage that
reads the bundle. A bundle of per-row operands has a per-row view, as the
mailbox does. A new operand is a field of one bundle, not an argument threaded
through every helper.

**Rows are a parameter, not a mode.** One launch serves a small, fixed maximum
of token rows. Every region has one copy per row, and the mailbox has a
per-row view in which a stage's indices address that row. There are two kinds
of stage, and every row of a launch computes exactly what a launch of that row
alone does (tests compare them bit for bit):

- A stage that streams weights serves every row in each task, so the stage
  reads a weight once: the GEMV packs the rows into the matrix-unit rows.
  Where each row needs other weights (an expert per row), each row's chain
  runs as a one-row GEMV into its own rows of one accumulator. Such a stage
  keeps the rows as the leading dimension of its tensors (`[rows, ...]`). So
  a reduction, a layout conversion, an epilogue, or a put happens once for all
  rows, not once per row.
- A stage whose work is a request's own state (a recurrence, attention,
  a conv) runs one task per (row, task). Its code is the one-row code on that
  row's view and operands. Tasks past the grid run in rounds. Vector work that
  reads little weight also runs one task per (row, task). Spread over more
  CTAs, it finishes sooner than when batched in one task.

**A speculative verify is a chain of rows, not a mode.** The plan's chain
length is the number of consecutive rows that are one request's tokens (1 in
decode, `k + 1` when the target verifies `k` draft tokens). So a verify is a
multi-row launch. Stages that only read their row stay as they are. A stage
that carries a request's state carries it from row to row. It runs a request's
tokens in order, and it reads every value that an earlier token of the launch
produced (for example its k and v) from that token's mailbox row, not from the
cache, since another CTA's cache write need not be visible.

Later stages can serve fewer rows than the first ones. In a draft model's
first step, every verified row writes its cache entries, but only the row
that the draft samples runs the rest of the layer, as the regular path's
draft narrowing does. Those later stages use a plan of one row, with the
same stages and regions, and the mailbox view of that row. The row is a
runtime value.

A request's operands are passed once a request, not once a row: its
lengths, page tables, and state pages. A bundle's per-row view points at the
request of the row and adds the token's offset to the length. So a verify
needs no kernels to expand its operands to rows.

Where a step's state goes is an operand, not a branch. Decode passes the
state pages. A verify passes the buffers that TokenSpeed's backends already
own for their regular verify, and no state page changes before acceptance.
So the post-verify commit stays the same. Tests check that a verify of
`k + 1` tokens equals that many decode steps.

**Kernels read weights in place where the layout allows.** Where the regular
path's packed weights already have the tile order that the kernel's GEMV
reads, the kernel reads them in place. The packer copies only what does not
have that order. Each vendor README lists the layouts.

## Placement

Placement (each stage's first CTA) is the main tuning knob. A trace records
when each stage starts, has its inputs, has computed, and ends on each CTA.
Judge a placement change on that timeline and on CUDA-graph timings, not on
intuition. The general rule: a task issues the loads that need no mailbox
input before it polls. So put a stage that streams weights on CTAs that are
idle while other stages produce its inputs. Each vendor README gives its
tuning workflow.

## Adding a model

The README of an existing model's kernel boundary
(`tokenspeed-kernel/python/tokenspeed_kernel/ops/model/<model>/monokernel/README.md`)
gives the ops, how a model's runtime seam calls them, the steps, and the
tests. It links to each vendor's README, which gives the building blocks, the
rules for writing a stage, the smallest complete kernel, and the performance
workflow. Each model directory there has a README for its layer kernels.
