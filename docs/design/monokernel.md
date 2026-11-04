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
| Kernel boundary | [`tokenspeed-kernel/python/tokenspeed_kernel/ops/monokernel/`](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/monokernel/README.md) | The ops that a model calls: `prepare_*` and `*_decode`, through `select_kernel`. `prepare_*` packs a layer once after loading and returns `None` when no kernel serves the layer. The README gives the API, the steps to add a model and links to each vendor's README. |
| Vendor kernels | The vendor's kernel package, under `ops/<arch>/monokernel/` | The building blocks that every model's kernels share (plans, mailboxes, GEMVs, trace), and one directory for each model's stages and layer kernels. Its README describes the files, the rules for writing a stage and the performance workflow. |
| Runtime seam | `python/tokenspeed/runtime/models/<model>_monokernel.py` | Eligibility with reasons, packing from the model's modules, the step counter, and one plan for each forward. |

## Invariants

**The layer boundary is the regular layer's boundary.** A fused layer takes
and returns the same tensors as the regular decoder layer in the model's layer
loop, with no pending state between layers. So fused and regular layers mix
freely. A layer that the monokernel does not serve keeps the regular path, and
the step still gives the right result. Each model's kernel README gives its
boundary tensors.

**Same caches, same layouts.** A monokernel reads and writes the KV cache,
the recurrent state pages and the other per-request caches exactly as the
regular decode path does. Prefill, the regular decode path and the monokernel
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

**One switch, one plan per forward.** `--decode-monokernel` is the one
switch for every model. Startup fails when the checkpoint's architecture has
no monokernel (`DECODE_MONOKERNEL_ARCHITECTURES` in `server_args.py`), or
when no layer of the target model qualifies. A layer that does not qualify
keeps the regular path, and startup logs it with the reasons. The switch
never serves silently without kernels. The model asks the seam once per
forward for a step plan, or None for the regular path. Its layer loop then
runs the step's fused layer `i` in place of layer `i` where the step has one.
The decoder layer classes have no monokernel code.

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

**Polls throttle their loads.** Many threads spinning on poll loads can starve
the producers' stores. A poll issues all its loads, waits once and retries
after a short sleep. A retry loads only the pairs whose tag differs, so a wide
poll sees its last pairs one round trip after they land.

**A stuck poll stops the process.** A poll stops the kernel with an error
after a bounded number of retries, which take far longer than a launch. A
deadlock or a missing producer then gives a GPU error and the process stops,
instead of a hang.

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
  a reduction, a layout conversion, an epilogue or a put happens once for all
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
lengths, page tables and state pages. A bundle's per-row view points at the
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
when each stage starts, has its inputs, has computed and ends on each CTA.
Judge a placement change on that timeline and on CUDA-graph timings, not on
intuition. The general rule: a task issues the loads that need no mailbox
input before it polls. So put a stage that streams weights on CTAs that are
idle while other stages produce its inputs. Each vendor README gives its
tuning workflow.

## Adding a model

The kernel boundary's
[README](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/monokernel/README.md)
gives the ops, how a model's runtime seam calls them, the steps and the
tests. It links to each vendor's README, which gives the building blocks, the
rules for writing a stage, the smallest complete kernel and the performance
workflow. Each model directory there has a README for its layer kernels.
