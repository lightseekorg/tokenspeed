# EAGLE-4 draft decode monokernel

`gluon_eagle4_draft_layer_gfx950` runs one chain draft step of an EAGLE-4
draft (Qwen3.8-27B's: one decoder layer, TP1, BF16) as one persistent
launch: the draft layer and the draft's final norm, for one row per
request, 1 to 16 requests. It uses the building blocks in
[`../README.md`](../README.md), which also gives the launch contract.

`layer.py` holds the kernel, its plans (`layer_plan`), the weight packing
(`pack_eagle4_draft_layer`) and the launcher
`launch_gluon_eagle4_draft_layer_gfx950`.
`tokenspeed_kernel/ops/model/eagle4/monokernel/` registers them as
`prepare_eagle4_draft_layer` and `eagle4_draft_layer_decode`.

## Contract

- Dimensions: hidden 5120, 40 q and 40 KV heads of 128, intermediate 16384
  (SiLU-gated MLP), RMSNorm eps 1e-6, NeoX RoPE over the whole head, a
  sliding window of 4096 keys (the row's own included). No `attn_norm`
  (`eh_proj`), MoE or NoPE.
- Inputs per row: the draft token's embedding and the hidden state it drafts
  from (also the layer's residual), BF16. Output: the final norm's BF16 row,
  which the drafter uses as both the LM head's input and the next step's
  hidden state (`norm_output`).
- The row's length `seq_len` includes its own token; its key and value go to
  `slots[r]` of the BF16 `[slots, 40, 128]` caches and the attention reads
  the window through the request's int32 page table (64 tokens a page,
  absolute positions), as the regular paged decode does.
- The row count is compile-time (one binary each, compiled at the first
  launch of that count, as every monokernel plan); the layer slot, the step,
  the page table's width and every length are runtime arguments.
- The roundings are the regular path's: each projection rounded to BF16, q
  and k rotated in FP32 and rounded once, the attention output rounded to
  BF16, `silu(g) u` rounded to BF16.

## Algorithm

The stages, in plan order (see `layer.py`'s module docstring for each):
inorm (the two input RMSNorms), the q | k | v projection, head (sum of the
K parts, RoPE, the KV write), attn (split-K over the window, online base-2
softmax), merge, the o projection, osum and pnorm (the post-attention
RMSNorm with the residual), gate-up, act, down, dsum and fnorm (the final
RMSNorm). The projections are BF16 GEMVs of 128 output rows by 512-wide K
parts with FP32 partials, summed by the stage that reads them. The
attention splits per row and head (`attn_splits(rows)`) trade rounds over
the 256 CTAs against a fixed cost per task.

On MI355X a launch takes about 168 µs at 1 row, 205 µs at 4, 286 µs at 8
and 415 µs at 16 in isolation (a 300-key window). Served (Qwen3.8-27B with
its EAGLE-4 draft, 7 chain steps), the launch plus its step-counter update
takes about 182 µs a draft step at 1 request and 238 µs at 8, against 187
µs and 243 µs for the regular path's ten kernels it replaces: the regular
path's projections already run near the weights' read bandwidth, so the
draft step (309 µs at 1 request, 371 µs at 8, with the LM head and top-k)
gains under 3%.
