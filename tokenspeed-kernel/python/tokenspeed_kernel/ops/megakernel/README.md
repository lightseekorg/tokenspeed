# Decode megakernels

A decode megakernel runs one decoder layer's decode step as one persistent
launch. The design and its invariants are in
[`docs/design/megakernel.md`](../../../../../docs/design/megakernel.md).
The gfx950 building blocks, and the rules for writing a stage, are in
[`tokenspeed_kernel_amd/ops/gfx950/megakernel/README.md`](../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/megakernel/README.md).
This file is the guide for adding a model's megakernel.

## The ops

A model's megakernel has two ops for each layer kind. The caller selects them
through `select_kernel("megakernel", ...)` with the trait `rows`:

- `prepare_<model>_<layer>(max_rows=, tensors=, dense_tiled=)` packs one
  layer once, after the weights load, into a `FusedDecodeLayer`. It returns
  `None` when no kernel serves the layer on this platform. The caller then
  keeps the regular path for that layer.
- `<model>_<layer>_decode(layer, h, h_out, *, chain=, ...)` runs one step.

The caller owns two values that every layer of a step shares:

- The mailbox scratch buffer (`FusedDecodeLayer.scratch_pairs` int64 pairs,
  the largest over the layers). Nothing clears it.
- The device step counter. It changes once per forward, inside the CUDA
  graph: the forward's last launch moves it on as it ends
  (`advance_step=True`, with a second int32 entry that counts the launch's
  finished CTAs), or the caller advances it after the forward. The launch
  tags its mailbox writes with it.

Every per-step operand is a tensor from the regular path's refresh-in-place
decode metadata (`docs/design/unified_path.md`), so the decode graph
captures the launch. A speculative verify uses the same op: `chain` is the
number of tokens per request (`k + 1` for `k` draft tokens), and the state
operands are the verify buffers that TokenSpeed's backends already own.

A model can also have small portable kernels around its layer megakernels
(`triton.py`). `qwen4_exp_mtp_front` runs the Qwen3.8-Flash-Next MTP draft
input (two Gemma RMSNorms, two BF16 linears and the stream add) for a decode
step's rows in one launch instead of six.

## Adding a model

Use `test/amd/ops/megakernel/test_megakernel_example_gfx950.py` as the
template.
It is a complete megakernel of two stages, about 150 lines long, that uses
the plan, the schedule, the mailbox, an operand bundle and the trace. The
Qwen3.8-Flash-Next kernels
([`gfx950/megakernel/qwen4_exp/`](../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/megakernel/qwen4_exp/README.md)) are the full
example. Their README gives the stages, the placement and the rows and verify
design. Their module docstrings give each stage's math and task split.

1. **List the handoffs.** Write the layer's decode step with the regular
   path's numerics (the activation quantization, and where the regular
   kernels round to BF16). Mark each point where all of a block's outputs
   must be ready before the next block runs: a norm over the full row, a
   softmax over all blocks, a top-k over all experts. Each such point is a
   stage boundary and costs one handoff (0.7 us idle, 2 to 3 us while weights
   stream). Use few, large stages.
2. **Write one stage module per block** in
   `tokenspeed_kernel_amd/.../megakernel/<model>/`. A module has:
   - Its mailbox `REGIONS`, as `(name, pairs per row)`.
   - Its `@aggregate` operand bundles, with a `row(r)` view for per-row
     operands.
   - `stages(bases, rows, ...)`, which declares each stage's tasks and the
     regions it reads and writes. The layer gives the placement.
   - One `@gluon.jit` function per stage, `(mailbox, task, bundles...,
     region names, STAGE)`.

   Put model dimensions that several modules use in `dims.py`.
3. **Compose the layer** in `<model>/layer.py`:
   - A `functools.cache` plan function per layer kind, which concatenates
     the modules' `stages()` with all stage bases in that one function.
   - The `@gluon.jit` kernel, which builds the bundles from tuple arguments
     and runs each stage's tasks in plan order.
   - `launch_<kernel name>`, which checks shapes, strides and row counts.
4. **Pack the weights** in `<model>/weights.py`. Where the regular path's
   weights have the GEMV's tile order, read them in place. The MFMA dense
   layout and the N16 MoE banks have that order. Copy only the scales. Give
   each host dataclass a `kernel_args()` that returns its bundle's fields in
   order.
5. **Register the ops** in this package. `gluon.py` registers the gfx950
   kernels, and `__init__.py` holds the public functions.
6. **Add the runtime seam** in `runtime/models/<model>_megakernel.py`,
   after `qwen4_exp_megakernel.py`. A model object packs the layers after
   loading (`post_quant_warmup`) and keeps the reasons of each layer that it
   does not serve. Its `plan()` runs once per forward and gives a step whose
   `layer(i)` replaces layer `i` in the model's layer loop. Leave the decoder
   layer classes unchanged. Pass the per-step operands by name. Add the
   checkpoint architecture to `DECODE_MEGAKERNEL_ARCHITECTURES` in
   `server_args.py`, so that `--decode-megakernel` accepts the model.
7. **Document the model** in `<model>/README.md`, with the sections of
   `qwen4_exp/README.md`: the kernels and their contract, the stage table,
   the placement, rows and verify, the weights and the tests. Add the
   directory to the model table of the gfx950 megakernel README.

## Tests

| What | How |
|---|---|
| The plan | Construct the layer's plans for every row count and chain length: `Plan` checks the deadlock rule when you construct it. |
| One stage | Run a slice of the real plan, `layer_plan(...).only(("b1", "b3"))`. The regions that the slice reads before it writes become `external`. Fill them with `megakernel_testing.fill` and read the outputs with `payloads`. Compare with a torch reference of the regular path's math, with tight tolerances. Do not write separate test plans, because they drift from the real ones. |
| One layer | The launcher against the regular path's ops, with looser tolerances. |
| Rows | A launch of `R` rows equals `R` launches of one row, bit for bit. |
| Verify | A verify of `k + 1` tokens equals `k + 1` decode steps, bit for bit. |
| Recompiles | `assert_no_triton_compile` across sequence lengths and table widths, after one warmup per row count. |
| The server | GSM8K at concurrency 1 and at the largest row count that the kernels serve, with the model's usual generation length and stop strings. |

GSM8K at concurrency above 1 changes between runs because of the batching
order. If the score changes after a refactor, compare the kernel outputs bit
for bit before and after it (run the old commit from a git worktree on
`PYTHONPATH`, with the same seeded inputs).

## Performance workflow

Placement (each stage's `base`) is the main tuning knob. Make each decision
from a measurement:

1. Record the timeline. A trace buffer records four points per stage and
   CTA: `START`, `READY` (the inputs arrived), `COMPUTED` and `END`. Find
   the critical path.
2. Change the placement:
   - Put a stage that streams weights on CTAs that are idle while other
     stages produce its inputs, so that its loads overlap the wait.
   - CTAs without a task in the first two stages sleep for a short time
     first (`schedule.head_sleep`). Without this, their weight streams
     compete with the critical path.
   - Issue a task's loads that need no mailbox input at the beginning of
     the first task that its CTA runs (see the issue / finish split in the
     gfx950 megakernel README). These early loads gave the largest wins.
   - Give a long serial task its own CTA. The QSA top-k final task on a
     shared CTA was about 20 us late at 128K tokens.
   - For a selection over many items, poll the inputs once into LDS, then
     run chunk tasks plus one final task. Do not poll again for each pass.
3. Time layer kernels cold, with several copies of the weights in
   rotation. Warm replays hit the Infinity Cache and hide 2 to 4 us.
4. Compare the old and new kernels in interleaved runs in one process, also
   for refactors. A change only to the order of address arithmetic moved a
   layer by 0.2 to 0.4 us.
5. Time the server with CUDA-graph events over several fresh servers, run
   alternately. End-to-end TPOT changes by 10 to 25 us between server
   instances.
6. Read the ISA (see the gfx950 megakernel README's checklist) and use
   `rocprofv3 --kernel-trace` for the gaps between launches.

Tried and dropped (besides several layers per launch, see the design doc):

- Per-layer operand tables, which several layers per launch need. Vector
  loads of a table entry make the buffer loads waterfall loops, and a layer
  loop spills. Four layers per launch was 26 us slower per step in serving.
- A 17-way split of QSA attention: no gain.
