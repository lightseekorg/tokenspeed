# MSA (MiniMax Sparse Attention CuTe-DSL kernels)

MiniMax MSA sparse-attention implementations, integrated under TokenSpeed's
attention family/variant hierarchy. The upstream package is named
`fmha_sm100`; this directory uses the `msa` algorithm variant and registers
CUDA-backed implementations with `solution="cuda"`.

- Upstream: https://github.com/vllm-project/MSA (maintained fork of
  https://github.com/MiniMax-AI/MSA)
- Pinned commit: `890aaa1a37a598ad17ccff0827fea21540d381fa`
  ("Fix CUTLASS DSL 4.6 compatibility (#8)", 2026-07-19) — the same commit
  vLLM pins via `cmake/external_projects/fmha_sm100.cmake`.
- License: MIT (SPDX headers retained in every file; copyright MiniMax).

## What is vendored

The CuTe-DSL sparse-attention stack for the block-sparse prefill attend,
plus the nvcc-JIT dense FMHA used in score-only mode by the prefill indexer:

- `cuda/` - the nvcc-JIT dense FMHA and indexer-score/top-k implementation
  (`fmha_sm100`, `fmha_sm100_plan`, and `sparse_topk_select`).
- `cute_dsl/` - `interface.py`, `sparse_index_utils.py`, `quantize.py`,
  `fp4_indexer_interface.py`, and the `src/` kernel implementation
  (upstream tests, examples, and build scaffolding are excluded).
- `python/csrc/cuda/msa/` - native CUDA sources, headers, and Jinja templates.
  The CSR and decode-schedule pybind extensions are built by `setup.py`; dense
  FMHA templates remain runtime-specialized by `cuda/jit.py`.

NOT vendored: `cutlass/` — upstream pins the full NVIDIA/CUTLASS repo
(`eb61c911`, CUTLASS 4.3.4) as a submodule purely for headers. `cuda/jit.py`
carries a local patch (`_find_cutlass_dir`, marked `TokenSpeed patch`)
that resolves headers from `TOKENSPEED_MSA_CUTLASS_DIR`, a package-local
`cutlass/` checkout, or the flashinfer wheel's bundled CUTLASS tree, in
that order. The csrc tree compiles cleanly against flashinfer's CUTLASS
4.5.0 (validated bitwise against the Triton scorer on SM100).

## Runtime requirements and behavior

- SM100 (Blackwell) only; `nvidia-cutlass-dsl>=4.6.0` and
  `quack-kernels>=0.6.1` (both in `requirements/cuda-thirdparty.txt`).
- CuTe implementation imports are package-relative and do not modify
  `sys.path` or expose top-level module aliases.
- `cuda/k2q_csr/` and `cuda/decode_schedule/` load pybind extensions compiled
  during package installation. Runtime code never invokes
  `torch.utils.cpp_extension.load` for these kernels.
- The CuTe kernels JIT-compile per variant on first call through cutlass-dsl.
- The dense FMHA path (`api.py`/`jit.py`) nvcc-JIT-compiles per kernel
  variant (~45 s each) into `~/.cache/minfer/fmha_sm100/`
  (`MINFER_FMHA_CACHE_DIR` overrides), loaded through `apache-tvm-ffi`;
  needs `nvcc`, `ninja`, and `jinja2`. `cuda/prefill_score.py`
  compiles its variants on a background thread and keeps the Triton
  scorer selected until they are ready, so serving never blocks on nvcc.
- FP8 support is identity-scale only: BF16 Q with FP8-E4M3 K/V stages to
  BF16 in-kernel; there are no k/v descale parameters.

## Updating

Re-copy from the upstream fork at a newer commit, let `pre-commit run`
reformat the tree, re-apply the `TokenSpeed patch` block in `cuda/jit.py`,
update the pinned commit above, and re-run
`tokenspeed-kernel/test/ops/test_attention_msa.py`. To diff against
upstream, black/isort-format the upstream side first.
