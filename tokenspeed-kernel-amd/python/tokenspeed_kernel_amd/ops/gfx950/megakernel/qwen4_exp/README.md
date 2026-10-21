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
- The launchers' docstrings list every operand. Row counts, positions,
  sequence lengths and table widths are runtime values.

### Serving

`python/tokenspeed/runtime/models/qwen4_exp_megakernel.py` is the runtime
seam. The flag `--qwen4-exp-decode-megakernel` enables it.

- A layer is eligible when it has a sparse MoE with a shared expert, and no
  attention TP or expert parallelism. The seam decides this once, at load.
- The seam packs each eligible layer after the weights load.
- A forward uses the kernels when it is a decode step of at most 4 rows,
  padded graph rows included. Each request has one token, or `k + 1` tokens of
  a target verify with ReplaySSM. Other forwards use the regular path.

## Stages

```
Gated DeltaNet layer:  [p1 p2] a1 a2 | b1 b3                  | c1 | m1 m2 m3 m4 m5
QSA attention layer:   a1 a2 | q1 q2 q3 q4 q4f q5 q6          | c1 | m1 m2 m3 m4 m5 [f1 f2]
```

Tasks are for one row. "+ per row" means that the count grows with each row.
A stage without that note serves every row in each task and reads each weight
once.

| Stage | Module | Tasks | Base | Work |
|---|---|---|---|---|
| `p1` | `ple.py` | 200 | 0 | Hashes the token's n-gram ids, gathers the embedding row, and projects the PLE key and value (MXFP4 `kv_proj`). |
| `p2` | `ple.py` | 80 per row | 176 | The gate, the Gemma norms and the dilated depthwise conv of 128 channels of one stream. Writes the next conv state. |
| `a1`, `m1`, `f1` | `hc.py` | 20 + 8 per row | 0, 0, 57 | Mixer reduce: the per-stream Gemma norm, the low-rank down projection and the inject logits, for one K half of the streams. |
| `a2`, `m2`, `f2` | `hc.py` | 80 | 176, 176, 0 | Mixer mix: the up projection of 32 columns of each stream. Publishes the mixed row as MXFP8 and as BF16. |
| `b1` | `gdn.py` | 128 | 28 | 128 rows of `in_proj_qkvz`. Tasks 0 to 79 also run the causal conv. Tasks 80 to 127 also compute two `in_proj_b` / `in_proj_a` logits each. |
| `b3` | `gdn.py` | 48 per row | 28 | The gated delta rule of one value head, then the gated RMS norm. Publishes the result as MXFP8. |
| `q1` | `qsa.py` | 109 | 28 | 128 rows of `qkv_proj` or `index_qk_proj`. |
| `q2` | `qsa.py` | 31 per row | 246 | Q and K norm and MRoPE, the KV cache write, the index q heads, and the index key with its block compression. |
| `q3` | `qsa.py` | 255 per row | 0 | Block scores `sum_h relu(q_h . k_b)`. Only when there are more complete blocks than `TOPK_BLOCKS`. |
| `q4` | `qsa.py` | 8 per row | 20 | The top `TOPK_BLOCKS` blocks of one chunk of `SELECT_CHUNK` blocks: four 8-bit radix passes in LDS. |
| `q4f` | `qsa.py` | 1 per row | 255 | The top blocks of the chunks' candidates, when there is more than one chunk. |
| `q5` | `qsa.py` | 64 per row | 64 | Attention of one KV head over one of 32 splits. Publishes `acc / l` and the base-2 LSE. |
| `q6` | `qsa.py` | 48 per row | 128 | The LSE merge of 128 dims of a head, then the output gate. Publishes the result as MXFP8. |
| `c1` | `out_proj.py` | 160 | 96 | The output projection of 16 hidden columns, then the gated residual combine of the 4 streams. |
| `m3` | `moe.py` | 65 | 28 | Router logits of 8 experts each. The last task computes the shared expert gate. |
| `m4` | `moe.py` | 220 | 93 | Routes, then the gate and up projection of 32 intermediate columns of one rank (10 routed and 1 shared). Publishes `silu(gate) * up` as MXFP8. |
| `m5` | `moe.py` | 160 | 96 | Routes, then all 11 down projections of 16 hidden columns as one rank-3 MFMA. Combines them into the output streams. |

Each module's docstring gives the exact math, the rounding points, and the
mailbox regions. The plan functions `delta_plan` and `qsa_plan` in `layer.py`
hold every base.

## Placement

Each CTA walks the stages in order, and a task issues its weight loads before
it polls. Thus the weights of a stage stream while earlier stages compute, but
only on CTAs that finished their earlier tasks. The bases follow from the
timeline:

- `b1` and `q1` start at CTA 28, on CTAs without a mixer task. Their weights
  stream from the start of the launch.
- The CTAs without a mixer task wait for `HEAD_SLEEP` units at the start.
  Without this wait, their weight streams compete with the mixer's.
- `m1` uses the CTAs of `a1`. The `a1` tasks end long before the MLP half
  starts.
- A CTA that runs two stages issues the loads of the second stage early,
  through one helper:
  - `projection_recurrence_stage` (`b1` and `b3`) issues the recurrent state
    loads with the B1 weights.
  - `projection_attention_stage` (`q1` and `q5`) issues the K and V gather
    with the Q1 weights.
  - `up_down_stage` (`m4` and `m5`) issues the M5 weights during the M4
    GEMV.

  These early loads gave the largest wins of the tuning.
- `q4f` has its own CTA (255). On a shared CTA, it was about 20 us late at
  128K tokens.
- `f1` and `f2` run on CTAs that are idle after the MoE.

## Rows and verify

- The weight stages (`p1`, the mixers, `b1`, `q1`, `c1`, `m3` to `m5`) keep
  the rows as the leading tensor dimension.
- The state stages (`p2`, `b3`, `q2` to `q6`) run one task per row and use
  `schedule.row_task`.
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
  steps.

## Weights

`weights.py` packs each layer once, after the weights load, into the
dataclasses that the stages read (`DeltaLayerWeights`, `QsaLayerWeights`).

- Dense projections in TokenSpeed's MFMA linear layout are already in N16
  tile order. `pack_dense` uses them in place and builds only the `"k4"`
  scale dwords.
- The routed experts are TokenSpeed's N16 MoE banks, read in place with
  their `"n32"` scales.
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
| `moe.py` | The router and the expert stages. |
| `weights.py` | Host packing of the layer weights. |

## Tests

Tests are in `tokenspeed-kernel/test/amd/ops/megakernel/`:

| Test | Covers |
|---|---|
| `test_qwen4_exp_hc_gfx950.py` | The mixer stages. |
| `test_qwen4_exp_gdn_gfx950.py` | The Gated DeltaNet stages. |
| `test_qwen4_exp_qsa_gfx950.py` | The QSA stages, and the top-k against TokenSpeed's block selection. |
| `test_qwen4_exp_moe_gfx950.py` | The MoE stages and the `"n32"` scales. |
| `test_qwen4_exp_layer_gfx950.py` | Both layers (with PLE and the final mixer), the one-row input, rows against one-row launches, and verify against decode steps. |

Each stage test runs a slice of the real plan (`Plan.only`) and compares it
with a torch reference of TokenSpeed's math.
