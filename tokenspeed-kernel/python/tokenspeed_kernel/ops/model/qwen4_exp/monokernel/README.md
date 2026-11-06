# Decode monokernels

A decode monokernel runs one decoder layer's decode step as one persistent
launch. The design and its invariants are in
[`docs/design/monokernel.md`](../../../../../../../docs/design/monokernel.md).
The gfx950 building blocks, and the rules for writing a stage, are in
[`tokenspeed_kernel_amd/ops/gfx950/monokernel/README.md`](../../../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/README.md).
This directory holds the Qwen4-Exp ops. Each model has its own
`tokenspeed_kernel/ops/model/<model>/monokernel/`. This file gives the ops,
how a model's runtime seam calls them, and the steps to add a model.

## The ops

A model's monokernel has two ops for each layer kind. The caller selects them
through `select_kernel("monokernel", ...)` with the trait `rows`:

- `prepare_<model>_<layer>(max_rows=, tensors=, dense_tiled=)` packs one
  layer once, after the weights load, into a `FusedDecodeLayer`. It returns
  `None` when no kernel serves the layer on this platform. The runtime seam
  then names the layer in its startup error: the switch never serves a layer
  without its kernel.
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
(`triton.py`). `qwen4_exp_mtp_fuse_inputs` runs the Qwen3.8-Flash-Next MTP draft
input (two Gemma RMSNorms, two BF16 linears and the stream add) for a decode
step's rows in one launch instead of six: the same math as the MTP head's
`_fuse_inputs`.

## How a model uses the ops

A model calls the ops from its runtime seam,
`python/tokenspeed/runtime/models/<model>_monokernel.py` (see
`qwen4_exp_monokernel.py`). The part that every model shares is in
`python/tokenspeed/runtime/models/monokernel.py`.

- **Building the layers.** With `--decode-monokernel`, the model's
  `decoder_layer_cls` gives `get_layer` the monokernel class of each layer
  type: a subclass of the regular layer class with the same parameters, with
  the model's `MonokernelLayer` subclass first in the bases. The model reads
  the switch from its config, which `ModelConfig` sets. The checkpoint
  architecture must be in `DECODE_MONOKERNEL_ARCHITECTURES` in
  `server_args.py`.
- **The layer class.** The model's `MonokernelLayer` subclass gives:
  - `launch_chain(rows, ctx)`: the tokens per request of the launch for a
    forward of `rows` token rows, or `None`. The result must be fixed per
    CUDA graph.
  - `_decode(chain, **forward_args)`: the launch. It takes the regular
    layer's forward arguments and returns the regular layer's outputs.
  - A packing method that calls `prepare_*` and raises
    `NoMonokernel(*reasons)` when the layer cannot be packed. Qwen4-Exp
    layers have `prepare_monokernel(final_mixer)`, and the class attribute
    `FOLDS_FINAL_MIXER` says whether the last layer's launch can also run
    the model's final mixer.

  `MonokernelLayer.forward` runs the regular layer's code when
  `launch_chain` returns `None` (prefill, or a row count or chain length past
  the kernels' plans). It fails when a launch comes before packing, and
  otherwise runs `_decode`.
- **After loading.** The seam packs every layer once the weights load and
  the quantization post-processing gives their final layout (the model's
  `post_quant_warmup` hook, which the model loader calls once). Then it gives
  all layers and the model one `MonokernelState` (`monokernel_state`), which
  holds the scratch buffer and the step counter. Startup fails when a layer cannot be packed,
  and the error (`unserved_error`) names each such layer with its reasons.
- **Each step** (one forward). The model calls
  `monokernel_state.start_step()` before its layers and `finish_step()` after
  them. A launch that also runs the model's final norm or mixer sets
  `lm_head_input` and moves the step counter as it ends
  (`advance_step=True`). Then `finish_step()` returns that output for the LM
  head. After other launches, `finish_step()` moves the step counter. A step
  without launches adds no kernel.

## Adding a model

1. **List the handoffs.** Write the layer's decode step with the regular
   path's numerics (the activation quantization, and where the regular
   kernels round to BF16). Mark each point where all of a block's outputs
   must be ready before the next block runs: a norm over the full row, a
   softmax over all blocks, a top-k over all experts. Each such point is a
   stage boundary and costs one handoff. Use few, large stages.
2. **Write the layer kernels** in the vendor package. For gfx950, the
   [gfx950 monokernel README](../../../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/README.md)
   gives the steps: the stage modules, the layer, the weight packing and the
   model's README.
3. **Register the ops** in `tokenspeed_kernel/ops/model/<model>/monokernel/`,
   as this directory does: `gluon.py` registers the gfx950 kernels, and
   `__init__.py` holds the public functions.
4. **Add the monokernel layer classes** as in "How a model uses the ops".
   Pass the per-step operands by name.

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
