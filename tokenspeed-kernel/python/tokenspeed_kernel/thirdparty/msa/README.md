# MSA (MiniMax Sparse Attention CuTe-DSL kernels)

Vendored copy of the MiniMax MSA sparse-attention Python package (named
`fmha_sm100` upstream; the directory is `msa` here to match the registered
kernel solution). Code is upstream-identical except formatting: the repo's
pre-commit hooks (black/isort) reformat on commit, per the convention used
for every tracked `thirdparty/` tree.

- Upstream: https://github.com/vllm-project/MSA (maintained fork of
  https://github.com/MiniMax-AI/MSA)
- Pinned commit: `890aaa1a37a598ad17ccff0827fea21540d381fa`
  ("Fix CUTLASS DSL 4.6 compatibility (#8)", 2026-07-19) — the same commit
  vLLM pins via `cmake/external_projects/fmha_sm100.cmake`.
- License: MIT (SPDX headers retained in every file; copyright MiniMax).

## What is vendored

The CuTe-DSL sparse-attention stack for the block-sparse prefill attend,
plus the nvcc-JIT dense FMHA used in score-only mode by the prefill indexer:

- `__init__.py`, `sparse.py` — the public import surface
  (`fmha_sm100.sparse` re-exports `sparse_atten_func`, `build_k2q_csr`, ...).
- `cute/` — `interface.py`, `sparse_index_utils.py`, `quantize.py`,
  `fp4_indexer_interface.py`, and the `src/` kernel sources
  (upstream tests, examples, and build scaffolding are excluded).
- `api.py`, `jit.py`, `sparse_fmha_adapter.py`, `csrc/` — the nvcc-JIT dense
  FMHA / indexer-score / top-k path (`fmha_sm100`, `fmha_sm100_plan`,
  `sparse_topk_select`). Consumed by
  `tokenspeed_kernel/ops/attention/msa_score.py` for the prefill indexer's
  OnlyScore scoring + top-k.

NOT vendored: `cutlass/` — upstream pins the full NVIDIA/CUTLASS repo
(`eb61c911`, CUTLASS 4.3.4) as a submodule purely for headers. `jit.py`
carries a local patch (`_find_cutlass_dir`, marked `TokenSpeed patch`)
that resolves headers from `TOKENSPEED_MSA_CUTLASS_DIR`, a package-local
`cutlass/` checkout, or the flashinfer wheel's bundled CUTLASS tree, in
that order. The csrc tree compiles cleanly against flashinfer's CUTLASS
4.5.0 (validated bitwise against the Triton scorer on SM100).

## Runtime requirements and behavior

- SM100 (Blackwell) only; `nvidia-cutlass-dsl>=4.6.0` and
  `quack-kernels>=0.6.1` (both in `requirements/cuda-thirdparty.txt`).
- Importing `.sparse` performs the upstream `sys.path.insert` of the
  `cute/` directory, exposing its top-level module names (`interface`,
  `src`, `quantize`, `sparse_index_utils`, `fp4_indexer_interface`)
  globally — an upstream packaging quirk vLLM ships identically; import it
  lazily (the tokenspeed_kernel wrapper in
  `ops/attention/msa.py` does).
- `cute/src/sm100/build_k2q_csr/` JIT-compiles a small CUDA extension via
  `torch.utils.cpp_extension.load` on first import (needs nvcc; cached in
  `~/.cache/torch_extensions/`). The CuTe kernels JIT-compile per variant
  on first call (cutlass-dsl).
- The dense FMHA path (`api.py`/`jit.py`) nvcc-JIT-compiles per kernel
  variant (~45 s each) into `~/.cache/minfer/fmha_sm100/`
  (`MINFER_FMHA_CACHE_DIR` overrides), loaded through `apache-tvm-ffi`;
  needs `nvcc`, `ninja`, and `jinja2`. `ops/attention/msa_score.py`
  compiles its variants on a background thread and keeps the Triton
  scorer selected until they are ready, so serving never blocks on nvcc.
- FP8 support is identity-scale only: BF16 Q with FP8-E4M3 K/V stages to
  BF16 in-kernel; there are no k/v descale parameters.

## Updating

Re-copy from the upstream fork at a newer commit, let `pre-commit run`
reformat the tree, re-apply the `TokenSpeed patch` block in `jit.py`,
update the pinned commit above, and re-run
`tokenspeed-kernel/test/ops/test_attention_msa.py`. To diff against
upstream, black/isort-format the upstream side first.
