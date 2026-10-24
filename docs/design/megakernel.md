# Decode megakernels

A decode megakernel runs a whole decoder layer's decode step as one
persistent launch. Every stage of the layer (norms, projections, attention,
MoE) becomes a set of tasks placed on fixed CTAs. Stages hand values to each
other through tagged mailboxes in a scratch buffer, not through kernel
boundaries. Latency, not FLOPs, limits small-batch decode. One launch
per layer removes about twenty kernel boundaries, and each stage can stream
its weights while earlier stages still compute.

The first user is Qwen3.8-Flash-Next (`qwen4_exp`) on gfx950, behind
`--decode-megakernel`. Its Gated DeltaNet and QSA attention layers
each have one megakernel.

## Where the pieces live

| Layer | Path | Holds |
|---|---|---|
| Runtime building blocks | `tokenspeed_kernel_amd/ops/gfx950/megakernel/` | `plan.py`: stages, placement, regions, the deadlock check. `schedule.py`: a kernel's view of its plan (which tasks this CTA runs). `mailbox.py`: tagged put / poll, throttling. `mx.py`: MXFP4 x MXFP4 split-K GEMV, MXFP4 and BF16 publishing. `bf16.py`: BF16 x BF16 GEMV. `trace.py`: per-stage timestamps. `weights.py`: tile and scale layouts. |
| Model stages | `.../megakernel/qwen4_exp/` | One module per block (`hc.py`, `ple.py`, `gdn.py`, `qsa.py`, `out_proj.py`, `moe.py`) with its regions, stages and operand bundles. `dims.py`: shared dimensions. `layer.py`: the layer plans, kernels and launchers. `weights.py`: weight packing. |
| Kernel boundary | `tokenspeed_kernel/ops/megakernel/` | `prepare_*` and `*_decode`, through `select_kernel`. `prepare_*` packs once after loading and returns `None` when no kernel serves the layer. `README.md` is the guide for adding a model. |
| Runtime seam | `runtime/models/qwen4_exp_megakernel.py` | `Qwen4ExpMegakernels`: eligibility with reasons, packing from the model's modules, the step counter, and `plan()` once per forward. `MegakernelStep.layer(i)`: the launch that replaces layer `i`. |

## Invariants

**The layer boundary is the hyper-connection streams.** A fused layer takes
the streams `[rows, hc_count * hidden]` and returns new streams with no
pending residual. Every Qwen4-Exp layer and the final mixer already accept
that form, so fused and regular layers mix freely. A layer the megakernel
does not serve keeps the regular path, and the step still gives the right
result. The first layer may take one row `[rows, hidden]` (the embedding)
that stands for all streams: the kernel reads it with stream stride 0,
without a copy.
When the last layer is a QSA layer, its launch also runs the model's final
mixer and writes the LM head's input. The mixer's stages f1 / f2 run on CTAs
that are idle in the MoE tail. The layer then hands its streams to the mixer
through region `h`.

**Same caches, same layouts.** A megakernel reads and writes the KV cache,
the GDN state pages and the QSA indexer caches exactly as the regular decode
path does. Prefill, the regular decode path and the megakernel can follow
each other on one request. The current token's own values (its k, v and
compressed key) reach later stages through mailboxes, not through the cache.
A cache write by another CTA in the same launch need not be visible yet (the
L2 is per XCD).

**Same math, not the same bits.** A megakernel follows the regular path's
numerics: MXFP4 activations for MXFP4 projections, BF16 activations for BF16
projections, and BF16 rounding where the regular path's kernels round. Summation orders differ, and so do the softmax
split boundaries, so results are not bitwise equal. Tests compare each stage
with a torch reference of TokenSpeed's math, with tight tolerances per stage
and looser ones per layer.

**One switch, one plan per forward.** `--decode-megakernel` is the one
switch for every model. Startup fails when the checkpoint's architecture has
no megakernel (`DECODE_MEGAKERNEL_ARCHITECTURES` in `server_args.py`), or
when no layer of the target model qualifies. A layer that does not qualify
keeps the regular path, and startup logs it with the reasons. The switch
never serves silently without kernels. The model asks the seam once per
forward for a `MegakernelStep`, or None for the regular path. Its layer loop
then runs `step.layer(i)` in place of layer `i` where that is not None. The
decoder layer classes have no megakernel code.

**Deadlock freedom comes from the plan.** Each stage reads only regions that
an earlier stage writes, and the `Plan` constructor checks this. Every CTA
runs its tasks in plan order, and all CTAs are resident (one per CU). So a
poll waits only on work that some CTA reaches without waiting on the poller.
A task may issue loads for its own weights before it polls, but it never
blocks on a later stage.

**Tags make scratch reuse safe.** A mailbox pair is `(payload, tag)`, with
`tag = step * 64 + layer + 1`. The kernel stores and loads each pair as one
64-bit agent-scope (`sc1`) access. The step counter is a device tensor that
changes once per forward, inside the CUDA graph: the forward's last launch
moves it on as its last CTA ends (no separate kernel), and otherwise
`MegakernelStep.finish` bumps it after the layers. So no write from an earlier layer
or step can satisfy a poll. All layers share the scratch
buffer, and nothing clears it.

**Polls throttle their loads.** Many waves spinning on `sc1` loads can starve
the producers' stores. A poll issues all its loads, waits once and retries
with `s_sleep`. A retry loads only the pairs whose tag differs, so a wide
poll sees its last pairs one round trip after they land. `pre_poll` lets one
wave spin on hint pairs before a wide poll.

**A stuck poll stops the process.** A poll traps (`s_trap 2`) after
`mailbox.MAX_RETRIES` retries, at least about 8 s, where a launch takes about
100 us. A deadlock or a missing producer then gives a GPU error
(`hipErrorLaunchFailure`) and the process stops, instead of a hang.

**Kernels take compile-time plans as text.** Triton's compile hooks serialize
every compile-time argument to JSON. A kernel therefore takes its plan as
`Plan.key` (a `PlanKey`: the canonical JSON, which also carries the plan
object for the compile-time helpers), never as the dataclass.

**Operands travel in bundles.** Each stage module declares `@aggregate`
bundles of the operands its stages read: `hc.Streams` and `hc.Mixer`,
`moe.Weights`, `gdn.Weights` and `gdn.State`, `qsa.Weights` and `qsa.Cache`,
`ple.Weights` and `ple.Step`. A kernel takes one tuple per bundle and builds
the bundle at its top. The tuple is the host dataclass's `kernel_args()`, in
the bundle's field order. The tuple is `None` when the plan has no stage that
reads the bundle. A bundle
of per-row operands has a `row(r)` view, as the mailbox does. A new operand
is a field of one bundle, not an argument threaded through every helper.

**Rows are a parameter, not a mode.** One launch serves 1 to 4 token rows
(`Plan.rows`). Every region has one copy per row, and
`Mailbox.row(r)` is the view in which a stage's indices address row `r`.
There are two kinds of stage, and every row of a launch computes exactly
what a launch of that row alone does (tests compare them bit for bit):

- A stage that streams weights serves every row in each task, so the stage
  reads a weight once: the GEMV packs the rows into the MFMA rows
  (`mx.gemv` returns `[rows, N]`). Where each row needs other weights (an
  expert per row), each row's chain runs as a one-row GEMV. A select then
  puts that chain's output in its own MFMA rows of one accumulator
  (`mx.gemv_row`). Such a stage keeps the rows as the leading dimension of
  its tensors (`[rows, ...]`, with indices from `Mailbox.row_offset`). So a
  reduction, a layout conversion, an epilogue or a put happens once for all
  rows, not once per row.
- A stage whose work is a request's own state (the recurrence, attention,
  the PLE conv) runs one task per (row, task). Its code is the one-row code
  on that row's view and operands. Tasks past the grid run in rounds. Vector
  work that reads little weight (the mixers' inject logits) also runs one
  task per (row, task). Spread over more CTAs, it finishes sooner than when
  batched in one task.

**A speculative verify is a chain of rows, not a mode.** `Plan.chain` rows
in a row are one request's consecutive tokens (1 in decode, `k + 1` when
the target verifies `k` draft tokens). So a verify is a multi-row launch.
Stages that only read their row stay as they are. A stage that carries a
request's state carries it from row to row:

- B1 builds each token's conv window from the request's earlier rows.
- B3 runs a request's tokens in order on the state it holds in registers.
- The QSA stages read every value that a token of the launch produced
  (block members, compressed keys, k / v) from that token's row. They do
  not read it from the cache, since another CTA's cache write need not be
  visible.

Later stages can serve fewer rows than the first ones. In the MTP draft's
first step, every verified row writes its cache entries, but only the row
that the draft samples runs the rest of the layer, as the regular path's
draft narrowing does. Those later stages use a plan of one row, with the
same stages and regions, and the mailbox view of that row. The row is a
runtime value.

A request's operands are passed once a request, not once a row: its
lengths, page tables and state pages. A bundle's `row(r)` view points at the
request of row `r` and adds the token's offset to the length. So a verify
needs no kernels to expand its operands to rows.

Where a step's state goes is an operand, not a branch. Decode passes the
state pages. A verify passes TokenSpeed's verify scratch rows and replay
payload. It also passes -1 as the output page, so no state page changes
before acceptance. Tests check that a verify of `k + 1` tokens equals that
many decode steps. The model seam passes the verify operands that
TokenSpeed's backends already own for their regular verify, so the
post-verify commit stays the same.

**Kernels read weights in place where the layout allows.** TokenSpeed's MFMA
dense layout and its N16 MoE banks are the tile orders the GEMV reads. The
packer copies only two things: the scales, into the GEMV's dword layout, and
the BF16 hyper-connection weights, which it re-encodes exactly as MXFP4.

## Placement

Placement (each stage's `base` CTA) is the main tuning knob. The trace
buffer (`trace.py`) records four timestamps per stage and CTA: `START`,
`READY` (inputs ready), `COMPUTED` and `END`. Judge a placement change on
that timeline and on CUDA-graph timings, not on intuition. The general rule:
a task issues the loads that need no mailbox input before it polls. So put a
stage that streams weights on CTAs that are idle while other stages produce
its inputs. `tokenspeed_kernel/ops/megakernel/README.md` gives the tuning
workflow and the lessons so far.

Several layers per launch was tried and dropped: the streams handoff
through the mailbox costs about as much as the launch gap it removes.

## Adding a model

The guide is `tokenspeed_kernel/ops/megakernel/README.md`: the steps, the
tests and the performance workflow.
`test_megakernel_example_gfx950.py` is the smallest kernel built from the
shared pieces. The gfx950 building blocks and the rules for writing a stage
are in `tokenspeed_kernel_amd/ops/gfx950/megakernel/README.md`. Each model
directory there has a README for its layer kernels.
