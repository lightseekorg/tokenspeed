# Decode monokernels

A decode monokernel runs one decoder layer's decode step as one persistent
launch. The design and its invariants are in
[`docs/design/monokernel.md`](../../../../../docs/design/monokernel.md).
The gfx950 building blocks, and the rules for writing a stage, are in
[`tokenspeed_kernel_amd/ops/gfx950/monokernel/README.md`](../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/README.md).
This file gives the ops, how a model's runtime seam calls them, and the
steps to add a model.

## The ops

A model's monokernel has two ops for each layer kind. The caller selects them
through `select_kernel("monokernel", ...)` with the trait `rows`:

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

A model can also have small portable kernels around its layer monokernels
(`triton.py`). `qwen4_exp_mtp_front` runs the Qwen3.8-Flash-Next MTP draft
input (two Gemma RMSNorms, two BF16 linears and the stream add) for a decode
step's rows in one launch instead of six.

## How a model uses the ops

A model calls the ops from its runtime seam,
`python/tokenspeed/runtime/models/<model>_monokernel.py` (see
`qwen4_exp_monokernel.py`). The decoder layer classes stay unchanged.

- **After loading.** With `--decode-monokernel`, the seam calls `prepare_*`
  for each layer once the weights load and the quantization post-processing
  gives their final layout (the model's `post_quant_warmup` hook, which the
  model loader calls once). It also allocates the scratch buffer and the step
  counter. It keeps the reasons of each layer that no kernel serves and logs
  them at startup. Startup fails when no layer of the target model
  qualifies, so the switch never serves silently without kernels. A
  speculative draft model without such a layer keeps the regular path. The
  checkpoint architecture must be in `DECODE_MONOKERNEL_ARCHITECTURES` in
  `server_args.py`.
- **Each forward.** The model asks the seam once for a step
  (`plan(rows, ctx)`). The step is `None` when no kernel serves the forward
  (prefill, or a row count or chain length past the kernels' plans). The
  forward then runs the regular path.
- **Each layer.** In the model's layer loop, `step.layer(i)` gives the launch
  that replaces layer `i`, or `None`. With `None`, the loop runs the regular
  layer. A launch takes and returns the same tensors as the regular layer, so
  fused and regular layers mix in one step.
- **After the layers.** If no launch moved the step counter on,
  `step.finish()` does.

## Adding a model

1. **List the handoffs.** Write the layer's decode step with the regular
   path's numerics (the activation quantization, and where the regular
   kernels round to BF16). Mark each point where all of a block's outputs
   must be ready before the next block runs: a norm over the full row, a
   softmax over all blocks, a top-k over all experts. Each such point is a
   stage boundary and costs one handoff. Use few, large stages.
2. **Write the layer kernels** in the vendor package. For gfx950, the
   [gfx950 monokernel README](../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/README.md)
   gives the steps: the stage modules, the layer, the weight packing and the
   model's README.
3. **Register the ops** in this package. `gluon.py` registers the gfx950
   kernels, and `__init__.py` holds the public functions.
4. **Add the runtime seam** as in "How a model uses the ops". Pass the
   per-step operands by name.

## Tests

| What | How |
|---|---|
| The plan | Construct the layer's plans for every row count and chain length. The plan checks the deadlock rule when you construct it. |
| One stage | Run a slice of the real plan (the vendor README says how). Compare with a torch reference of the regular path's math, with tight tolerances. Do not write separate test plans, because they drift from the real ones. |
| One layer | The launcher against the regular path's ops, with looser tolerances. |
| Rows | A launch of `R` rows equals `R` launches of one row, bit for bit. |
| Verify | A verify of `k + 1` tokens equals `k + 1` decode steps, bit for bit. |
| Recompiles | `assert_no_triton_compile` across sequence lengths and table widths, after one warmup per row count. |
| The server | GSM8K at concurrency 1 and at the largest row count that the kernels serve, with the model's usual generation length and stop strings. |

GSM8K at concurrency above 1 changes between runs because of the batching
order. If the score changes after a refactor, compare the kernel outputs bit
for bit before and after it (run the old commit from a git worktree on
`PYTHONPATH`, with the same seeded inputs).
