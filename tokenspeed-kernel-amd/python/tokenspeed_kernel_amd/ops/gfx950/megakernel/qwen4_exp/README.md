# Qwen3.8-Flash-Next decode megakernels

These kernels run one decoder layer of Qwen3.8-Flash-Next (`qwen4_exp`) as one
persistent launch, for 1 to 4 token rows. They use the building blocks in
[`../README.md`](../README.md). That file also gives the launch contract and
the rules for writing a stage.

## Kernels

| Kernel | Layer |
|---|---|
| `gluon_qwen4_exp_delta_layer_gfx950` | A Gated DeltaNet layer, with its PLE block when the layer has one. |
| `gluon_qwen4_exp_qsa_layer_gfx950` | A QSA full-attention layer. The last layer's launch also runs the model's final mixer. |

`layer.py` holds both kernels and their launchers
(`launch_<kernel name>`). `tokenspeed_kernel/ops/megakernel/` registers them
as `prepare_qwen4_exp_*` and `qwen4_exp_*_decode`.

### Contract

- The input is the hyper-connection streams, BF16 `[rows, 4 * HIDDEN]`. For
  the first layer, the input is the embedding, `[rows, HIDDEN]`: one row that
  stands for all four streams. The output is the new streams, with no pending
  residual.
- `rows` is 1 to `MAX_ROWS` (4). `chain` is 1 for decode and `k + 1` for an
  MTP verify of `k` draft tokens. The rows are request-major.
- The kernels read and write the regular path's caches in place: the GDN
  conv and SSM pages, the PLE pages, the paged KV cache and the QSA indexer
  caches.
- A verify writes TokenSpeed's verify buffers and no state page: the GDN and
  PLE verify scratch rows, the ReplaySSM payload and `QSAVerifyState`.
  TokenSpeed's post-verify commit then writes the accepted state.
- An MTP draft step (one row per request, `draft` operand) writes the
  speculative index keys to the QSA indexer's draft scratch, not to the
  recent ring, as the regular draft path does.
- The MTP draft's catch-up step (its first step: one request's verified rows
  as a chain, `accepted` operand) narrows as the regular path does. Every row
  runs up to `q2`: it writes its K / V, and its index key when the row is
  accepted. The stages from `q3` on run only for the live row, the row of the
  last accepted token (`layer.live_plan`). The launch outputs that one row.
- The launchers' docstrings list every operand. Row counts, positions,
  sequence lengths and table widths are runtime values.

### Serving

`python/tokenspeed/runtime/models/qwen4_exp_megakernel.py` is the runtime
seam. The switch `--decode-megakernel` enables it.

- A layer is eligible when it has a sparse MoE with a shared expert, and no
  attention TP or expert parallelism. The seam decides this once, at load,
  and logs every other layer with its reasons. Startup fails when no layer
  of the target model is eligible.
- The seam packs each eligible layer after the weights load.
- A forward uses the kernels when it is a decode step of at most 4 rows,
  padded graph rows included. Each request has one token, or `k + 1` tokens of
  a target verify with ReplaySSM. Other forwards use the regular path.
- The MTP draft model's one QSA layer, with its final mixer, uses the same
  kernel for every draft step: the first step as a chain over one request's
  verified rows (a first step of several requests uses the regular path),
  the next steps as one row each.

## Stages

```
Gated DeltaNet layer:  [p1 p2] a1 a2 | b1 b3                  | c1 | m1 m2 m3 m4 m5
QSA attention layer:   a1 a2 | q1 q2 q3 q4 q4f q5 q6          | c1 | m1 m2 m3 m4 m5 [f1 f2]
```

Tasks are for one row. "+ per row" means that the count grows with each row.
A stage without that note serves every row in each task and reads each weight
once. In a draft catch-up step, the stages from `q3` on serve one row, the
live row.

| Stage | Module | Tasks | Base | Work |
|---|---|---|---|---|
| `p1` | `ple.py` | 200 | 0 | Hashes the token's n-gram ids, gathers the embedding row, and projects the PLE key and value (BF16 `kv_proj`). |
| `p2` | `ple.py` | 80 per row | 176 | The gate, the Gemma norms and the dilated depthwise conv of 128 channels of one stream. Writes the next conv state. |
| `a1`, `m1`, `f1` | `hc.py` | 80 + 8 per row | 0, 0, 57 | Mixer reduce: the per-stream Gemma norm and the low-rank down projection of 16 low-rank columns of one stream, for every row; or one row's inject logit of one stream half. |
| `a2`, `m2`, `f2` | `hc.py` | 80 | 176, 176, 0 | Mixer mix: the up projection of 32 columns of each stream. Publishes the mixed row as BF16 (`m2` also as MXFP4, the experts' input). |
| `b1` | `gdn.py` | 256 | 28 | 64 rows of `in_proj_qkvzba`. Tasks 0 to 159 also run the causal conv. Tasks 160 to 255 also compute one `in_proj_b` / `in_proj_a` logit each. |
| `b3` | `gdn.py` | 48 per row | 28 | The gated delta rule of one value head, then the gated RMS norm. Publishes the result as BF16. |
| `q1` | `qsa.py` | 218 | 28 | 64 rows of `qkv_proj` or `index_qk_proj`. |
| `q2` | `qsa.py` | 31 per row | 246 | Q and K norm and MRoPE, the KV cache write, the index q heads, and the index key with its block compression. |
| `q3` | `qsa.py` | 255 per row | 0 | Block scores `sum_h relu(q_h . k_b)`. Only when there are more complete blocks than `TOPK_BLOCKS`. |
| `q4` | `qsa.py` | 8 per row | 20 | The top `TOPK_BLOCKS` blocks of one chunk of `SELECT_CHUNK` blocks: four 8-bit radix passes in LDS. |
| `q4f` | `qsa.py` | 1 per row | 255 | The top blocks of the chunks' candidates, when there is more than one chunk. |
| `q5` | `qsa.py` | 64 per row | 64 | Attention of one KV head over one of 32 splits. Publishes `acc / l` and the base-2 LSE. |
| `q6` | `qsa.py` | 48 per row | 0 (up to 2 rows), 128 | The LSE merge of 128 dims of a head, then the output gate. Publishes the result as BF16. |
| `c1` | `out_proj.py` | 160 | 96 | The output projection of 16 hidden columns, then the gated residual combine of the 4 streams. |
| `m3` | `moe.py` | 65 (130 past 2 rows) | 28 | Router logits of 8 experts each, for every row or (past 2 rows) for 2 rows. The last block of experts is the shared expert gate. |
| `m4` | `moe.py` | 220 (200 + 20 / C past 1 row) | 93 | Routes, then the gate and up projection of 32 intermediate columns of one routed rank, or of `C` (`shared_blocks`) 32-column blocks of the shared expert, read once for every row. Publishes `silu(gate) * up` as MXFP4. |
| `m5` | `moe.py` | 160 | 96 | Routes, then the down projections of 16 hidden columns of every (row, rank) chain, 16 chains per rank-3 MFMA. Combines each row's chains in rank order into the output streams. |

A layer whose routed experts are BF16 (the MTP draft layer of the MXFP4
checkpoint) runs `m3` to `m5` from `moe_bf16.py`: `m4` has 440 tasks of 16
intermediate columns of one rank, and publishes `bf16(silu(gate) * up)` as
BF16. `m5` is as above, with BF16 MFMAs.

Each module's docstring gives the exact math, the rounding points, and the
mailbox regions. The plan functions `delta_plan` and `qsa_plan` in `layer.py`
hold every base.

## Placement

Each CTA walks the stages in order, and a task issues its weight loads before
it polls. Thus the weights of a stage stream while earlier stages compute, but
only on CTAs that finished their earlier tasks. The bases follow from the
timeline:

- `b1` and `q1` start at CTA 28, on CTAs without a mixer task at one row.
  Their weights stream from the start of the launch.
- The CTAs without a mixer task wait for `HEAD_SLEEP` units at the start.
  Without this wait, their weight streams compete with the mixer's.
- `m1` uses the CTAs of `a1`. The `a1` tasks end long before the MLP half
  starts.
- A CTA that runs two stages issues the loads of the second stage early,
  through one helper:
  - `projection_recurrence_stage` (`b1` and `b3`) issues the recurrent state
    loads with the B1 weights.
  - `up_down_stage` (`m4` and `m5`) issues the first M5 chain group's
    weights during the last M4 GEMV, and the other groups right after M4.

  These early loads gave the largest wins of the tuning.
- `q1` and `q5` do not share such a helper. Next to the 320 KiB of BF16
  weights of a `q1` task, the K / V gather spilled, and the QSA layer was
  1 to 10 us slower (1 to 4 rows).
- A stage polls the inputs that are published early before the ones that
  its producer publishes last: the mix stages' inverse RMS and streams (past
  one row), M5's residual and gates, and C1's inject logits. Each poll round
  trip takes about 1 us, so the last one then carries only the late pairs.
- `q4f` has its own CTA (255). On a shared CTA, it was about 20 us late at
  128K tokens.
- `f1` and `f2` run on CTAs that are idle after the MoE.

## Rows and verify

- The weight stages (`p1`, the mixer mix stages, `b1`, `q1`, `c1`, `m3` to
  `m5`) keep the rows as the leading tensor dimension. Past 2 rows, `m3`
  splits the rows over two tasks per block of experts, so a
  task's per-row work stays short.
- `m4` streams 2 units of weights at a time: a routed task's unit is a row's
  expert, a shared expert task's unit is a block that every row reads.
- `m4`'s two K-split waves of a block each take one contiguous half of K
  (`"n32h"` scales; the LDS rows hold the k-iterations in that order). One
  scale dword then serves two k-iterations of a block pair, which halves
  the scale loads; a wave of an interleaved split needed one byte gather
  per k-iteration.
- `b3` waits for every `b1` task. The z tasks, which also compute a b / a
  logit from the BF16 rows, take the slots before A2's CTAs
  (`gdn.Z_FIRST`), so no slow task starts after A2. Their BF16 poll reads
  consecutive pairs in consecutive lanes: an element-indexed poll issued
  32 strided loads a thread and made these tasks 2 us late.
- Multi-row GEMV operands in LDS end each row in padding
  (`bf16.load_rows`, the mix stage). The MFMA lanes of a multi-row GEMV
  read several rows at once, and rows whose starts share an LDS bank
  conflict: a 4-row B1 GEMV took 1.7 instead of 0.64 us. M5's batch rows
  and scale runs gained nothing from padding.
- `q5` merges the launch's own tokens (from the mailbox) into its K / V
  tiles. Their rows go through LDS once and load in the tiles' layouts; a
  per-token register reduction and layout conversion each cost an LDS round
  trip and spilled about 70 registers (q5 took 12 instead of 4 us at 4
  rows). The q / k / v polls read pairs, one load per pair.
- `m4`'s routing is on the critical path: no routed weight load can issue
  before it. Wave `r` picks row `r`'s top 10 of all 512 keys in inline
  assembly (a DPP wave maximum per rank); the other waves skip it. A row
  split over two waves needed a second merge pass, and the compiler's
  reduction ran on every wave: 1.7 against 0.9 us per routing.
- `m5` loads each chain's scale bytes as one 768-byte run (16-byte loads)
  into LDS before its poll's barrier, then reads each step's bytes there.
  One byte gather per step and chain was two thirds of its memory
  instructions.
- `m5` packs every (row, rank) down projection into chains, 16 per MFMA
  batch (3 batches at 4 rows instead of 4 rows of 16 rank slots), so every
  row's weights are in flight before the poll without register spills.
- The low-rank tasks of the mixer reduce stages (`a1`, `m1`, `f1`) serve
  every row: each reads its 16 columns of the BF16 down projection once. The
  inject logits run one row per task, so the rows' norms run on different
  CTAs.
- The state stages (`p2`, `b3`, `q2` to `q6`) run one task per row and use
  `schedule.row_task`.
- The launchers take the lengths, the page tables and the state pages once a
  request (`[rows // chain]`) and the positions, tokens and output rows once
  a row. `qsa.Cache.row` and `ple.Step.row` point at the row's request;
  `Cache.length` gives the row's own `L + 1`.
- In a verify, each state stage carries the request's state from token to
  token:
  - `b1` builds each token's conv window from the request's earlier rows.
  - `b3` runs the request's tokens in order. It writes `[k | v | a | b]` of
    each token to the ReplaySSM payload.
  - `p2` and `write_context` read the context of the earlier tokens from
    their rows.
  - `q2`, `q3` and `q5` read the compression block members, the compressed
    keys and the K and V of the earlier tokens from their rows. The index key
    goes to TokenSpeed's verify staging, not to the recent ring.
- A launch of `R` rows gives bit for bit the results of `R` one-row launches.
  A verify of `k + 1` tokens gives bit for bit the results of `k + 1` decode
  steps. Draft steps give bit for bit the results of decode steps; only the
  place of the index keys differs. A catch-up step's output is bit for bit
  its live row's decode step output.

## Weights

`weights.py` packs each layer once, after the weights load, into the
dataclasses that the stages read (`DeltaLayerWeights`, `QsaLayerWeights`).

- The BF16 projections (the mixers, `in_proj_qkvzba`, `qkv_proj`,
  `index_qk_proj`, the output projections and PLE `kv_proj`) are copied once into the
  GEMV's tile order (`bf16_tiles`): a wave loads 1 KiB of contiguous weights
  per instruction. The copies add about 6.7 GB per GPU. The regular path
  keeps its row-major weights for prefill.
- MXFP4 dense projections in TokenSpeed's MFMA linear layout are already in
  N16 tile order. `pack_dense` uses them in place and builds only the
  `"k4"` scale dwords.
- The routed MXFP4 experts are TokenSpeed's N16 MoE banks, read in place
  with their `"n32"` scales. BF16 experts (`pack_moe_bf16`) are read in
  place, row-major.
- The shared expert's gate and up blocks alternate like the banks' blocks
  (`alternate_gate_up`). Thus every M4 rank reads its weights the same way.
- Each dataclass's `kernel_args()` gives its bundle's fields in order.

## Files

| File | Contents |
|---|---|
| `layer.py` | The plans, the two layer kernels and their launchers. |
| `dims.py` | The dimensions that several modules use. |
| `hc.py` | The hyper-connection mixer stages. |
| `ple.py` | The PLE stages. |
| `gdn.py` | The Gated DeltaNet stages. |
| `qsa.py` | The QSA attention and indexer stages. |
| `out_proj.py` | The output projection and residual combine. |
| `moe.py` | The router and the expert stages, MXFP4 experts. |
| `moe_bf16.py` | The expert stages, BF16 experts. |
| `weights.py` | Host packing of the layer weights. |

## Tests

Tests are in `tokenspeed-kernel/test/amd/ops/megakernel/`:

| Test | Covers |
|---|---|
| `test_qwen4_exp_hc_gfx950.py` | The mixer stages. |
| `test_qwen4_exp_gdn_gfx950.py` | The Gated DeltaNet stages. |
| `test_qwen4_exp_qsa_gfx950.py` | The QSA stages, and the top-k against TokenSpeed's block selection. |
| `test_qwen4_exp_moe_gfx950.py` | The MoE stages and the `"n32"` scales. |
| `test_qwen4_exp_moe_bf16_gfx950.py` | The MoE stages with BF16 experts. |
| `test_qwen4_exp_layer_gfx950.py` | Both layers (with PLE and the final mixer), the one-row input, rows against one-row launches, verify against decode steps, and draft steps against decode steps. |

Each stage test runs a slice of the real plan (`Plan.only`) and compares it
with a torch reference of TokenSpeed's math.
