# Inkling2 layer decode monokernel (gfx950)

`layer.py` runs one decoder layer of an Inkling2 target, a local
(sliding-window) one (a mid one or the first of an attention-residual block,
layer 0 included) or a full attention one, with NVFP4 or MXFP8 experts, as
one persistent launch a rank: attention TP4, experts EP4, the verify blocks
of 8 rows of 1, 2, 4 or 8 requests
(`plan.local_layer_plan(dims, rows, 8, experts, boundary, first)` or
`plan.full_layer_plan(dims, rows, 8, experts, splits)`, one plan a row
count, expert format and layer variant). `dims` (`dims.Dims`) are the
rank's dimensions, which the packing reads from the weights' shapes;
`plan.SERVED_DIMS` lists the geometries the stages implement. It uses the building blocks in [`../README.md`](../README.md).

## Which layers launch

Every layer: the mid local layers `6 k + 1 .. 6 k + 4`, the boundary local
layers `6 k` (0, 6, .., 42) and the full attention layers `6 k + 5` (5, 11,
.., 47), with NVFP4 experts below layer 40 and MXFP8 experts from it. Layer
0's block starts from the embedding: its attention input is the embedding's
RMSNorm (`norm_attn`), not a mix. A launch takes the residual state's
pending terms and the set partials from the layer before it and returns the
layer's terms
`(P, A, M)`, as the regular layer's `finalize_layer` leaves them
(`docs/design/monokernel.md`).

## Boundary layers

A boundary layer starts an attention-residual block. Its launch differs from
a mid layer's in three places, each bit for bit the regular path's:

- `mix_attn` writes the rounded candidate `P` as the layer's block (the
  residual state's `blocks[n]`) instead of a pending term.
- `sweep` and `sweep_rest` (`mix.sweep_stage`) compute the new block set's
  partials, `_attnres_partial_kernel`'s recurrence over the `n` stored
  blocks and this launch's candidate (from the `cand` region), one task a
  (row, query) of the set's 14 queries. The regular path runs them on its
  aux stream; here they run on CTAs beside the attention. The layer's own
  MLP query (the set's first) also publishes its partial in `set_acc` and
  `set_ms`.
- `mix_mlp` mixes the one term `A` (the new block's partial so far) over
  that partial (`BOUNDARY`).

Layer 0 is the boundary layer with no block stored before it (`first`): its
block starts from the embedding, the one pending term. `norm_attn`
(`mix.attention_norm_stage`) replaces `mix_attn`: the attention input is the
embedding's `attn_norm` (`rmsnorm_precision`, `_rmsnorm_precision_kernel`'s
reduction), not a mix, and the block is the embedding as is (no rounding
sum: a `-0.0` stays). The sweep then runs over no stored block and the
candidate, as the regular path's set partials of one block.

The boundary and layer 0 are static parameters of the plan (layer 0 by its
`norm_attn` stage) and the boundary one of the layer kernel; mid layers'
plans keep the `set_acc` and `set_ms` regions unused, so every variant keeps
every region at the same offset.

## Full attention layers

A full layer ends an attention-residual block. Its launch differs from a mid
layer's where its regular forward does, each bit for bit:

- `mix_attn` mixes three queries of the block set (`mix.full_attention_mix_stage`):
  the Q, K and V sources, each with its own gain, into `x_attn`, `x_k` and
  `x_v`, as `attention_residual_combine` mixes the layer's 3 query rows.
  Before layers 5 .. 41 the draft tap's `residual_sum` has folded the
  pending terms into one, which is the candidate as it is.
- The projection projects each source (the regular path's one stacked
  GEMM over the three, whose rows do not depend on each other): `qkvr` the
  Q source with the whole full-layer weight (q, rel, gate and sink at the
  local offsets), `kproj` and `vproj` the K and V sources with only their
  value and conv-tap rows (`plan.KV_PROJ`), the columns the regular path
  reads from them.
- `head` (`head.full_head_stage`) scales the clipped Q heads and relative
  logits by the row's `tau` once rounded, and convolves K and V with the
  static taps plus the taps their sources project, lag 0 first,
  `(4 * 2 + 1) ** -0.5` output scale (`_conv_row`'s dynamic order).
- `attn` and `merge` (`attention.full_attention_stage`,
  `attention.full_merge_stage`) take every key over the regular decode's
  split count at the served context length (32 at 262120 tokens, from the
  server's fixed `max_context_len`; the plan's `splits`), each split one or
  more 128-key blocks with the regular kernel's online rescale. A task takes
  a verify block's 8 rows, or half of them where whole blocks would leave
  the grid short of tasks (`plan.SHARED_ATTENTION`;
  `attention.shared_attention_stage`): it reads a split's keys and values
  once for its rows, whose (row, head) pairs are the tile's rows (passes of
  32), each with its row's length mask, relative logits and running sums,
  so every element's MFMA chain and row sum stay its own row's. A task
  whose rows' split lengths differ (a length crossing a split-length step)
  runs each row alone.

The full layer is a static parameter of the plan and the kernel
(`FULL_SPLITS`); local plans keep its regions (`x_k`, `x_v`, `kproj`,
`vproj`, 32 splits of `attn_part`) unused, so every variant keeps every
region at the same offset.

## Looped cores

A looped checkpoint runs its core layers once a pass, each run a visit with
its own cache layer (KV pages and conv ring); the runtime seam addresses
the visit's caches and tags each visit's launches with its visit index.
The first pass's visits launch as above. A later pass gates the previous
pass's message (an attention over the bank of the blocks that pass wrote,
then a low-rank controller) into the Q, K and V inputs before their norms,
and into the MoE input, which mixes the attention output, before its norm.
The regular path computes those gated, normalized inputs, and the visit
runs as two launches (`plan.later_pass_plan`, `LATER_PARTS`):

- the attention part: `in_attn` (`inputs.py`) publishes the Q, K and V
  sources (three, local layers too: the regular path projects a local
  layer's K and V sources with its K and V rows only, `kproj` and `vproj`),
  then `qkvr .. red_attn` as in a full or local layer, writing `A`;
- the MoE part: `in_mlp` publishes the MoE input, then `router .. red_moe`,
  writing `M`.

Each part is a slice of the layer's three-source plan with every plan's
regions. The two parts write disjoint regions and share the visit's tag.
Every rank runs both parts, so the peer copies alternate as for any launch.

## Expert formats

One kernel serves both formats: the format is a static parameter of the
plan (`local_layer_plan(rows, chain, experts)`) and of the stages (`NVFP4`),
so each format compiles its own launch. They differ only in the experts:

| | NVFP4 | MXFP8 |
|---|---|---|
| `quant`, `swiglu` | group-16 E4M3 scales, targets 6 and 4, the row's amax (`xa`, `ha`) | group-32 UE8M0 exponents; `xa` and `ha` unused |
| `ex1`, `ex2` weights | packed E2M1 `[N, K / 2]`, outer factors `wa`, `WD` (`experts.Bank`) | E4M3 `[N, K]` (`experts.Mxfp8Bank`) |
| tile epilogue | `bf16(acc (xa wa) / (1536 WD))` | `bf16(acc)` |
| ring | 4 stages of 26 KB | 2 stages of 42 KB |

Both quantizers publish the exact BF16 values of their codes, which the GEMM
tiles read as their activations. Both plans keep every region at the same
offset: the launches of a step share one scratch and alternate its peer
copies, so a launch that starts while a peer still reads the previous one
(of the other format) must not write over that launch's peer regions.

## Stages

`mix_attn qkvr head attn merge wo | [fold_attn |] red_attn | mix_mlp router topk quant ex1 swiglu ex2 fold | red_moe`

| Stage | Module | Work |
|---|---|---|
| `mix_attn`, `mix_mlp`, `norm_attn` | `mix.py` | Depth mix over the set partials, RMSNorm x gain. `mix_attn` writes `P`; layer 0's `norm_attn` normalizes the embedding instead. |
| `qkvr`, `router`, `wo` | `projection.py` | BF16 GEMVs over every row (weights in `bf16_tiles` order), bit for bit the regular path's split-K projection (`gluon_mm_bf16_fp32_gfx950`): one MFMA chain per K split on its own wave slot, the splits' partials added in split order. `wo` reads the merge's `o` as plain words after each row's KV-head flags (half the bytes of tagged pairs: every `wo` task loads every row), stores its BF16 partial to every rank's peer region and raises a flag a task. |
| `head` | `head.py` | Q / K clips, K / V short convs (ring and in-launch taps), V norm, KV cache write, relative logits; a wave a head, a task a group of a row's heads (`plan.HEAD_SHAPES`: past 16 rows the Q heads of a KV head (6, or 3 at half width), or the 4 K and V heads, so the stage runs in one round). |
| `attn`, `merge` | `attention.py` | Windowed relative attention over 8 splits (the launch's own keys from their rows' mailboxes; a task a split of a row, a CTA's tasks streaming their cached keys and values through two LDS tiles with direct-to-LDS copies, each task's issued before the previous task computes), LSE merge of a KV head's query heads a task (the regular kernel's head block: 8 rows for 6 heads, 4 for 3), output gate and sink (published as plain words with a flag a task): bit for bit the regular Triton split-KV decode at its 128-token page (its MFMA 4x4x4 chains, row-sum layout and online split merge) and `rel_mha_gate_sink`. |
| `red_attn`, `red_moe` | `conv.py` | The rank-ordered FP32 fold of the peer partials fused with the output short conv of the task's channels, ring writes included: bit for bit the regular path's ordered-fold all-reduce and decode sconv. |
| `fold_attn` | `projection.py` | Past 16 rows (`plan.WO_SCATTER`) the attention output's exchange is a reduce-scatter and an all-gather: `wo` sends a 32-column block's partial to its owner rank only, the owner folds the ranks' partials in rank order (the same bits) and sends the folded rows to every rank, and `red_attn` runs the conv on them. Half the bytes over the links, one hop more. |
| `topk` | `route.py` | Sigmoid + bias selection, 8 of the routed experts (384, or 128 at half width), weights from the raw logits. |
| `quant` | `quant.py` | NVFP4 (targets 6 and 4) or MXFP8 activation quantization, as plain words with a flag. |
| `ex1`, `swiglu`, `ex2`, `fold` | `experts.py` | This rank's experts of the launch: w13 / w2 tiles of up to 16 of an expert's rows (an expert of more rows takes several tiles), 16 columns a wave through the LDS ring (NVFP4: 32 columns a wave on the same LDS for the whole waves of tiles past the grid, with a last partial wave of at most the grid left at 16; the ring runs on across a program's tiles, each tile's first K blocks copying during the previous tile's last), capped SwiGLU x route weight with its quantization, slot-order fold of the local routes into the expert exchange. |

## Geometries

`plan.GEOMETRIES` lays out a TP4 rank of Inkling 2 Small (hidden 4608, 12 Q
heads, 384 experts, expert width 2304) and of the Inkling2 geometry of half
its width (hidden 2304, 6 Q heads, 128 experts, expert width 1152). Each
stage follows the regular kernel's launch at the geometry: a row of 2304 is
the mix's 2048-column tile and a masked second tile of 256 live columns,
the norm's 4096-wide tile of 4 warps, and the sweep's 2048-wide tile; 3
query heads a KV head take the regular decode's 4-row head block and its
row-sum layout; the w2 GEMM's K of 1152 is not a whole number of the
ring's 256-wide stages, so `ex2` runs the regular kernel's masked last
stage (its weights and activations past K as +0.0).

## Rows

The verify width (`chain`) is the server's DSPARK block, 8 or 4 rows a
request; a launch serves 1, 2, 4 or 8 requests. The conv ring holds the
taps' history and one verify block (`3 + chain` rows, a compile-time value
of the ring). The shared-row full attention below takes blocks of 8 rows;
other widths run a row a task.

A launch of `8 bs` rows serves a verify batch of `bs` requests, each
request's 8 rows after the previous one's (the plan's chain of 8). The
weight-streaming stages (`qkvr`, `wo`, `router`, the expert tiles) read each
weight once for every row (past 16 rows, a projection's MFMA rows are the
rows, one MFMA a 16-row block on the same weight tile, and `qkvr` and
`router` split K over chained groups of tasks, `plan.PROJECTION_SHAPES`:
each group's tasks load their splits' activations only and add their splits
onto the previous group's FP32 sums, in split order; an expert's tile
holds up to 16 of the rows that route to it, so an expert reads its weights
once for each 16 of its rows); the state stages (`head`, `attn`, `merge`,
`topk`, `quant`, `swiglu`, the mixes) run one task a row and task, in rounds
past the grid, on the row's request: its positions, page-table row, ring
slot and cache write slots. Every row equals the same row of a one-request
launch bit for bit. A padded request of a graph batch has ring slot -1 (no
ring reads or writes), the null page as its pages and write slots, and its
rows at positions 0 .. 7 (the attention's verify floor).

## Numerics

Bit for bit the regular path. Every stage equals the regular kernel it
replaces at that kernel's decode launch, and its test compares the two with
`torch.equal`: the same MFMA shapes and K orders, the same split boundaries
and partial-sum orders, reductions in the regular kernels' layouts, the same
libdevice functions, and FP multiply-adds contracted exactly where those
launches contract them. Attention is identical at the served 128-token KV
page (the regular kernel's KV block is the page). So a verify block's tokens
and acceptance equal the regular path's, also on random-token prompts whose
continuations sit on near ties.

## Tests

`tokenspeed-kernel/test/amd/gfx950/ops/monokernel/test_inkling2_*.py`: each
stage against a torch reference of the regular math (`Plan.only` slices), and
the stages that match a regular kernel bit for bit against that kernel, the
exchanges on four ranks, and the whole layer of either expert format on four
ranks, which must equal its one-stage slices bit for bit, and a launch of
several requests (one of them padded) equals one launch a request.
`test_inkling2_mxfp8_gfx950.py` holds the MXFP8 variants of `quant`, `ex1`,
`swiglu` and `ex2`.
