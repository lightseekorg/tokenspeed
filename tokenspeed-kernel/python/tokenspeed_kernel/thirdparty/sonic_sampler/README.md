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
  shard for the ranks' gather and merge, and `merged_candidates`, which
  `cumulative_selection_kernel` and `cumulative_unpack_kernel` read through
  their new trailing `cv_ptr`, `ci_ptr` and `stride_m` arguments in place of
  their own unpack. `ops/prologue.py` adds `head_logits`, the logits
  processor's logit scale and softcap, which `bitpacked_reduction_kernel` (new
  trailing `logit_scale` and `softcap`) and `exact_candidates` apply to a
  shard's raw head logits;
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
