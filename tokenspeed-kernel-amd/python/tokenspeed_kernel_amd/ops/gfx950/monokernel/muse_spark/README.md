# Muse Spark decode monokernels

These kernels run one decoder layer of Muse Spark Open (`muse_spark`) as one
persistent launch per rank of a TP8 group (attention TP8, MoE TP8 / EP1), for
the rows of a decode step: one decode row, or the 2 to 8 consecutive tokens
of a speculative verify (`Plan.chain`). The row count is a compile-time
parameter up to two requests' rows (16); the runtime serves one request a
launch. They use the building blocks in [`../README.md`](../README.md).
That file also gives the launch contract and the rules for writing a stage.

## Kernels

| Kernel | Layer |
|---|---|
| `gluon_muse_spark_layer_gfx950` | A decoder layer with MXFP4 routed experts: sliding-window (RoPE) or full (NoPE) attention, by the `full` flag. |

`layer.py` holds the kernel, its plans (`layer_plan`) and its launcher
`launch_gluon_muse_spark_layer_gfx950`. `tokenspeed_kernel/ops/model/muse_spark/monokernel/`
registers them as `prepare_muse_spark_layer` and `muse_spark_layer_decode`.

### Contract

- Rank `r` owns hidden columns `1024 r ..`. The launch reads each row's
  FP32 residual of the owned columns (`[rows, 1024]`) and whole BF16
  attention input of the layer (`[rows, 8192]`). It writes each row's owned
  FP32 residual after the layer and whole BF16 attention input of the next
  layer, normed with that layer's input gain.
- Row `t` of request `b` is the token at position `seq_len[b chain] - 1 +
  t` (`seq_len` holds each row's length).
  `layer_plan` builds the plan of a row count and chain; each compiles once.
- The regular path hands over the whole residual row. Thus the first launch
  of a run of fused layers reads its slice of that row, and after the last
  one the caller gathers the 8 slices back into a row (32 KiB).
- The launch reads and writes the regular path's paged KV cache in place:
  BF16 `[slots, 2, 64]` per rank, 64 tokens a page, the page table indexed
  by absolute position on sliding layers too. A sliding layer attends over
  the last 2048 positions. Each row writes its token's slot; the rows read
  the launch's own keys from the projection, not from the cache.
- The step counter, the sequence lengths, the write slots and the page
  table are device tensors. The layer index is a runtime argument. Thus one binary
  serves every layer of a kind, and the decode graph captures the launches.
- The mailbox tag is `step * 64 + layer + 1`. The peer regions' copy is
  `layer & 1`: it alternates from one launch to the next, across steps too,
  since layers 1 to 60 are an even count. A rank one launch ahead thus puts
  only into the copy no slower rank still reads. That needs one region
  layout for every launch in the scratch: the sliding and full plans size
  the attention regions for the full layer's splits, so they lay the scratch
  out alike (a layout per kind let a sliding and a full launch overlap, and
  the group deadlocked under serving).
- The launcher's docstring lists every operand.

### Serving

`python/tokenspeed/runtime/models/muse_spark_monokernel.py` is the runtime
seam. The switch `--decode-monokernel` enables it.

- Layers 1 to 60 are eligible. Layers 0 and 61 hand over to the embedding
  and the LM head, and keep the regular path.
- A model is eligible at attention TP8 and MoE TP8 / EP1, with Muse Spark
  Open's dimensions and the constants the kernel bakes in (`KERNEL_CONSTANTS`),
  BF16 dense projections and norms, and routed experts on a gluon MXFP4 MoE
  kernel. The seam decides this once, at load, and logs every other layer
  with its reasons. Startup fails when no layer is eligible.
- The seam allocates the mailbox scratch on the Iris symmetric heap before
  graph capture.
- A launch traps when its peers' values do not come within about a minute,
  and a cold compile can take longer on one rank than on the others. Thus
  the first launch of each layer and row count only compiles
  (`compile_only`), the ranks meet at a CPU barrier, and then they launch.
- A forward uses the kernels when it is a decode step of one request: one
  token, or a speculative verify of up to 8 tokens (row `t` has length
  `seq - rows + t + 1` for the step's `seq_lens`). Other forwards use the
  regular path.
- A draft model that taps the residual entering a layer (DFlash2's
  `target_layer_ids`) gets whole rows there: the fused run before that layer
  gathers the 8 owned slices, as before a regular layer.

## Stages

One launch has 256 CTAs of 8 warps. The stages, in plan order (each module's
docstring gives its math and task split):

| Module | Stages | Work |
|---|---|---|
| `qkvg.py` | qkvg, gate, qnext | The fused q / k / v / output-gate projection, in two K halves. Between two fused layers (past one row) the q / k / v part moves to the end of the launch before: qnext tasks project the out tasks' rows with the next layer's weights into a global buffer, which the next launch's head and attn tasks read with plain loads, so that launch's attention starts at once. Past 8 rows the CTAs that run both out and qnext issue their qnext weights before their out task, so the weights land while out waits for its inputs. |
| `attention.py` | head, attn, merge | k norm, RoPE and the KV write; split attention with an online softmax, its K and V copied to an LDS ring two chunks ahead (on a full layer at long context, each warp group over its own keys of a chunk, combined once a split; a BF16 cache copied a page at a time; an FP8 cache scores the cached keys with FP8 MFMAs and q rounded once to FP8, as the regular path does); the LSE merge and the gated head norm. |
| `attn_tail.py` | oproj, tail1, tail2 | `o_proj` reduce-scattered to the owners; the post-attention norm, residual gate and pre-FFN norm. |
| `moe_route.py` | router, topk | Partial router logits of the owned columns; the sum, sigmoid scores and top 8. |
| `moe_pool.py` | pool | The pre-expert projection, gathered on every rank. Its input, the whole MoE input every rank's attention tail publishes, arrives as plain words and one flag per tail task (`Mailbox.store_words_peer`, `bf16.load_word_rows`): half the bytes of tagged pairs. |
| `moe_experts.py` | slt, xq, ex1, exr, ex2 | The MXFP4 experts, each rank's 512 of 4096 intermediate values. The rows' distinct experts are slots: each streams once for every row that picked it. slt builds the slot table once after the routing; xq norms and quantizes the expert input once; ex1 streams its first slot's weights as soon as the table is in, then loads its K quarter from xq; exr sums ex1's K quarters and requantizes to MXFP4 once, so ex2 polls 72 words a row and quantizes nothing; ex2 polls its lane's first slots' intermediates at once, then streams the slots' weights ahead with no poll between them. The module docstring gives the placement and batching rules. |
| `warm.py` (shared) | warm | For one row, when the next layer is fused too, CTAs 0 to 63 read its packed QKVG weights into the memory-side cache after their expert tasks, so the next launch's projections read the MALL (earlier, the expert weights would evict them). |
| `moe_tail.py` | comb1, comb2, unpool, out | Combine the routes, the post-expert projection, the post-FFN norm, residual gate and the next layer's input norm. Each out task sums its row's partial sums of squares from every rank into the two norm scales itself. The out tasks of the rank's owned columns also store the FP32 residual and, before a layer a draft taps, the BF16 residual into every rank's tap buffer. |
| `draft_layer.py` | both halves in one plan | A whole DFlash2 draft layer as one launch: the attention half hands its partial to the MLP half's owners through the mailbox (afin in place of scat) and the residual between the two sums stays in registers. The last layer's launch also runs the draft's final norm (fin puts its partial to the owners; fsum, fnorm and fout store the normed rows on every rank). Served with `--decode-monokernel` for one request's draft block (`python/tokenspeed/runtime/models/dflash2_monokernel.py`). |
| `draft_context.py` | fc, fsum, fnorm, kv, kvout | The DFlash2 context write after a verify as one launch: the rank's eighth of `fc` over the target's 5 taps (K split over all CTAs), `hidden_norm` with the sum of squares exchanged, then every draft layer's K and V of the rank's KV head (k RMSNorm and RoPE) into the draft cache. Served with the draft layers. |
| `draft_head.py` | lmh, hproj, merge1, merge2, score | The DFlash2 draft head as one launch: the logits of the rank's 25256-row LM-head shard with each draft step's best 16 a task, the rank's best 16 put to every rank, the sorted best 16 of the 8 ranks (softcapped), and the candidate selector's 16 x 16 edge scores. The best-16 passes are runtime loops: unrolled, they run cold in the instruction cache once a step. Served with the draft layers. |
| `draft_attn.py` | head, attn, merge (with draft_mlp's sum, norm, coefficient, convolution and projection stages around them) | The attention half of a DFlash2 draft layer: the input norm of the ranks' sum, the attention convolution, qkv, the per-head q / k RMSNorm and RoPE with the KV write, non-causal block attention over the window from the paged cache (its partials published as plain words and one flag a task), `o_proj` and the convolution after it, ending in this rank's partial. Served as part of `draft_layer.py`. |
| `draft_mlp.py` | scat, sum1, norm, coef, conv0, gu, act, down, fin | The MLP half of a DFlash2 draft layer (Muse Spark's draft, TP8): the ranks' sum of the attention partials, the residual add and post-attention norm, the convolution coefficients, the convolutions around the MLP, gate/up, `silu(g) u` and down, ending in this rank's partial. Served as part of `draft_layer.py`. |

## Weights

`weights.py` packs each layer once, after the weights load
(`pack_layer` -> `LayerWeights`).

- The BF16 QKVG, `o_proj`, pooler and post-expert weights are copied once
  into the GEMV's tile order (`bf16_tiles`). The pooler and post-expert
  weights are replicated on every rank; a rank copies only the rows it
  computes. The copies add about 4.3 GB per GPU for 60 layers. The regular
  path keeps its weights for prefill.
- The routed MXFP4 experts are read in place, in the layout of the regular
  path's gluon MoE (`preprocess_gluon_mxfp4_gfx950_moe_weights`).
- The router takes the owned columns of the FP32 router weight. The norm
  gains are the FP32 `bf16(w + offset)`. The residual gate coefficients stay
  views of the regular path's tensors.
- `pack_layer` checks every tensor's shape and dtype and reports all
  mismatches in one error. `LayerWeights.kernel_args()` gives the launch's
  operand tuples in order.

## Files

| File | Contents |
|---|---|
| `layer.py` | The plans, the layer kernel and its launcher. |
| `dims.py` | The dimensions that several modules use. |
| `qkvg.py`, `attention.py`, `attn_tail.py` | The attention stages. |
| `moe_route.py`, `moe_pool.py`, `moe_experts.py`, `moe_tail.py` | The MoE stages and the layer boundary. |
| `draft_mlp.py`, `draft_attn.py`, `draft_layer.py`, `draft_head.py`, `draft_context.py` | The DFlash2 draft layer's MLP and attention halves, the whole layer in one plan, the draft head and the context write. |
| `weights.py` | Host packing of the layer weights. |

## Tests

Tests are in `tokenspeed-kernel/test/amd/gfx950/ops/monokernel/`. All but the
QKVG and attention tests need 8 GPUs.

| Test | Covers |
|---|---|
| `test_muse_spark_qkvg_gfx950.py` | The QKVG and gate stages. |
| `test_muse_spark_attention_gfx950.py` | The attention stages, sliding and full. |
| `test_muse_spark_attn_tail_gfx950.py` | The output projection and attention tail. |
| `test_muse_spark_route_gfx950.py` | The router and top 8. |
| `test_muse_spark_pool_gfx950.py` | The pooler. |
| `test_muse_spark_experts_gfx950.py` | The experts, on the regular path's preprocessed weights. |
| `test_muse_spark_moe_tail_gfx950.py` | The combine, post-expert projection and boundary. |
| `test_muse_spark_draft_layer_gfx950.py` | The whole DFlash2 draft layer in one launch across the 8 ranks: the attention half's output, the hand-off to the MLP half (its input is the ranks' sum of every rank's attention partial) and the MLP half's stages; the last layer's final norm. |
| `test_muse_spark_draft_context_gfx950.py` | The context write in one launch across the 8 ranks: every layer's K and V at the rows' slots against fc, hidden_norm and the K/V projection with k RMSNorm and RoPE. |
| `test_muse_spark_draft_head_gfx950.py` | The draft head in one launch across the 8 ranks: a valid top 16 of every step (BF16 ties may pick other tokens of the same value), the same on every rank, sorted with ties by token id, and the selector's scores of those candidates. |
| `test_muse_spark_draft_attn_gfx950.py` | The draft's head stage and block attention on one GPU (window cut, short and empty prefixes), and the whole attention half in one launch across the 8 ranks, each stage on the kernel's own input. |
| `test_muse_spark_draft_mlp_gfx950.py` | The DFlash2 draft's MLP half in one launch across the 8 ranks, each stage on the kernel's own input against `dflash2_draft_reference.py` (the regular path's rounding points). |
| `test_muse_spark_layer_gfx950.py` | The whole layer through the packer and the launcher, a compile-only call that runs nothing, no recompiles across layers, steps and lengths, CUDA graph replays against eager launches, runs of launches that hand the next layer's projection on (qnext) against launches that project their own (the same bits), and the packer's errors. |

Each stage test runs a slice of the real plan, at 1 row and at 8 chained
rows, and compares it with a torch reference of TokenSpeed's math
(`muse_spark_reference.py`).
`test/runtime/test_muse_spark_monokernel_seam.py` covers which forwards and
models the seam serves.
