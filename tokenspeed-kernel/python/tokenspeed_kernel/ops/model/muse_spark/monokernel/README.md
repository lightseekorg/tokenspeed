# Muse Spark decode monokernels

This directory holds the Muse Spark ops: the target's decoder layers and its
DFlash2 draft's layers, draft head, and context write. The design and its
invariants are in
[`docs/design/monokernel.md`](../../../../../../../docs/design/monokernel.md).
The general API, how a runtime seam calls the ops, and the steps to add a
model are in the
[Qwen4-Exp ops' README](../../qwen4_exp/monokernel/README.md). The gfx950
kernels, their stages, and their placement are in
[`gfx950/monokernel/muse_spark/README.md`](../../../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/muse_spark/README.md).

## The ops

Every launch runs on all 8 ranks of the TP8 group at once, and the ranks
exchange values through the Iris symmetric heap (`peer_bases`). A prepare
function returns `None` when the platform has no kernel for the layer.

| Op | Packs or runs |
|---|---|
| `prepare_muse_spark_layer(tensors=, max_rows=)`, `muse_spark_layer_decode(layer, *, h, a, h_out, a_out, ...)` | One target decoder layer for the rows of one or more requests: a decode row each, or a speculative verify's consecutive tokens (`chain`). `h` is the rank's owned FP32 residual columns and `a` the whole BF16 attention input. A launch can end with the next layer's q / k / v projection (`proj_out`, read as the next launch's `proj_in`). |
| `prepare_muse_spark_draft_layer`, `muse_spark_draft_layer_decode` | One DFlash2 draft layer for the blocks of one or two requests (8 rows each). |
| `prepare_muse_spark_draft_head`, `muse_spark_draft_head_decode` | The draft head: the vocabulary-parallel top 16 of each draft step and the candidate selector's scores. |
| `prepare_muse_spark_draft_context`, `muse_spark_draft_context_decode` | The context write: `fc` and `hidden_norm` of the target's taps, then every draft layer's K and V into the draft cache. |

The caller owns the values that every launch of a step shares:

- The mailbox scratch buffer, of at least each packed layer's
  `scratch_pairs` int64 pairs, at the same offset of the Iris symmetric heap
  on every rank. Nothing clears it.
- The device step counter, which changes once per forward inside the CUDA
  graph. The target layers' launches of one forward alternate the parity of
  `layer_index`, so a launch's peer writes never land in the buffers of the
  launch before it. A draft layer's launch takes its peer copy from the
  parity of `step * 5 + layer_index`, so every draft forward launches all
  five draft layers and changes the step once.

`compile_only=True` compiles a launch without running it, so every rank
compiles each binary before the first launch that needs all ranks.

## Stuck polls

A poll that waits about a minute stops the launch with a GPU error. Before
it stops, it writes what it waited for into the record buffer of
`muse_spark_stuck_record` (`stuck_rec` of the launches), pinned host memory or
a file that outlives the process. The record's `describe` turns the records
into one line per stuck poll: the stage, region, row and pair, the tags, and
the wait.
