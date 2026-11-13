# sonic-sampler

This directory vendors the kernels and wrappers of
[sonic-sampler](https://github.com/tachyontrails/sonic-sampler) 1.0.0
(Apache-2.0, `LICENSE`), at commit
`c1acbca39b7faa9269b24e73dee2b57fe6215c7e` of the `tokenspeed` branch of
[LorrinWWW/sonic-sampler](https://github.com/LorrinWWW/sonic-sampler/tree/tokenspeed).
That branch adds two commits to the 1.0.0 release
(`b3f8b7dfa052e0fde3020223346039a251b3e0ab`):

- `0371ed8aa59941f99b0ef675c01094f7b7e6c19b`: the batch size is a runtime
  argument, so one kernel serves every batch size instead of compiling one per
  size.
- `c1acbca39b7faa9269b24e73dee2b57fe6215c7e`: with `noise_seeds` and
  `noise_offsets`, the Gumbel and acceptance noise is drawn in-kernel from
  Philox keyed by the slot's seed and `offset * 256 + timestep`, so no noise
  plane is allocated or refreshed per step.

Only the kernels (`ops/`) and their indicator bit layout (`core/flags.py`'s
`Indicator`) are vendored. `tokenspeed_kernel.ops.sampling.sonic` launches
them over the backend's slot state and picks their config; `--sampling-backend
sonic` selects them.

## Local changes

The vendored sources differ from `src/sonic_sampler` at that commit only by:

- the package path: `sonic_sampler.` imports name
  `tokenspeed_kernel.thirdparty.sonic_sampler`;
- Triton imports: `from triton import ...`, `triton.language.extra.libdevice`
  and `triton.language.core.builtin` go through `tokenspeed_kernel._triton`,
  so the kernels compile with `tokenspeed-triton`;
- the top-p nucleus: `ops/base.py`'s `nucleus_masking` keeps a candidate
  while the mass of the candidates ranked above it is below `top_p` of their
  total, so the one crossing `top_p` too; upstream kept one only if the mass
  up to and including it was at most `top_p` (and always the first), which
  drops the crossing one;
- exact fp32 logits (soft-capped heads): `ops/prologue.py` gathers the logit
  manipulators into `transform`, which keeps the logits' dtype
  (`value_dtype`; bf16 logits compile as before), and adds `exact_maximum`,
  `exact_greedy` and `exact_candidates`. Under the `exact` flag of
  `bitpacked_reduction_kernel`, `cumulative_selection_kernel` and
  `cumulative_unpack_kernel`, greedy rows keep each block's fp32 maximum and
  its first column, and sampled rows rank by bf16 keys only to pick their
  top-128 candidates, whose fp32 values are re-read (from the new trailing
  arguments), transformed and re-sorted before masking and the draw;
- vocabulary-sharded logits (a vocab-parallel LM head under TP): the new
  `ops/shard.py` (not upstream) adds `shard_candidates_kernel`, which writes
  each rank's best `MAX_K` (value, global token id) pairs from its reduced
  shard for the ranks' gather, and `merged_candidates` and `merged_first`,
  which merge every rank's gathered pairs inside `cumulative_selection_kernel`
  and `cumulative_unpack_kernel` (new trailing `cand_ptr`, `ranks`, `stride_mr`
  and `stride_mb`) in place of their own unpack. `ops/prologue.py` adds
  `head_logits`, the logits processor's logit scale and softcap, which
  `bitpacked_reduction_kernel` (new trailing `logit_scale` and `softcap`) and
  `exact_candidates` apply to a shard's raw head logits;
- draft-probability verify under tokenspeed's runtime contract:
  `ops/verify.py`'s `chain_speculative_verification_kernel` takes a trailing
  `reject_threshold`. With draft probabilities, a recorded probability above
  it marks a slot with no proposal yet (fresh admission, PD landing): its
  draft is rejected and the rejected position is drawn from the target, as
  the q of that row counts as zero. A draft the target gives no mass is never
  accepted;
- draft probabilities recorded over each rank's vocabulary shard:
  `shard_candidates_kernel` takes trailing `d_ptr`, `stride_d` and `stride_dg`
  and writes each candidate's recorded probability as a third block of its
  pairs; `merged_candidates_with_drafts` (beside `merged_candidates`, both on
  the new `merged_selection`) merges them, `cumulative_unpack_kernel` writes
  them for its candidates (new trailing `m_ptr`), and
  `chain_speculative_verification_kernel` reads them there (new trailing
  `m_ptr`, in place of `d_ptr`) for the accept test and the residual;
- whole-vocabulary draws for rows without top-k, top-p or min-p (not
  upstream): `ops/prologue.py` adds `whole_reduction_kernel`, which stores
  each block's best logit plus the slot's Gumbel noise at each token
  (`whole_block`, reading the block in tiles of at most 4096 columns, in
  `exact_maximum`'s slots), and in a verify with point-draw drafts leaves the
  draft out and stores the block's maximum, exp-sum and the draft's value.
  `bitpacked_reduction_kernel` leaves those rows to it under a new trailing
  `whole`, so the other rows' reduction compiles as before;
  `cumulative_selection_kernel` takes the best over the blocks,
  `shard_candidates_kernel` writes each rank's (`merged_whole` merges them)
  and `cumulative_unpack_kernel` writes a draft row as the draft at its
  probability over the whole vocabulary and the draw without it at
  probability one (`whole_targets`), all under a new trailing `whole`;
- draft-tree draws (not upstream): one draw per tree node in the singular
  setting. `whole_reduction_kernel` and `cumulative_selection_kernel` take a
  new trailing `u_ptr`, each row's noise step (its depth in its tree) in
  place of its step in the window (`whole_block` keys its noise by a new
  `draw_step`), and `decode_counts` reads a singular row's drafted tokens at
  its row (`margin`) instead of its slot, so rows sharing a slot count their
  own paths;
- the Gumbel noise: `ops/base.py`'s `philox_gumbel` computes `log(-log u)`
  from the Philox draw's raw bits (`exponential_log`, on the same counters):
  `u` and `1 - u` come exactly from the bits and `-log u` from a short series below
  `2^-8` and fast logs above, accurate to about `1e-5` where Gumbel-max
  winners draw (`u` near 1), where upstream's fp32 uniform spaces `1 - u` in
  steps of `2^-24`;
- the msgspec host layer is not vendored: `base/`, `core/buffer.py`,
  `interface/` (the `fused_singular`/`fused_multistep` wrappers, the tiling
  and warp-config structs, the class-based interfaces), the packaged dispatch
  tables (`resources/`), and `ops/copy.py`; `core/flags.py` keeps only
  `Indicator` (with `stochastic` and `from_params`), and `core/naive.py` (the
  reference sampler, the only user of `cytoolz`) is left out. With them go
  the kernels only the class-based interfaces launch,
  `ops/topk/bitpack.py:bitpacked_reduction_kernel` and
  `ops/topk/unpack.py:merge_unpack_kernel`, and `__version__`;
- repository formatting (black, isort).

The upstream docs, assets and packaging are not vendored. To update, copy
`src/sonic_sampler/ops` and `core/flags.py` from the new commit, apply the
changes above, and check the launch arguments in
`tokenspeed_kernel.ops.sampling.sonic` against the new wrappers.
