# CDNA4 MXFP4 MoE kernels

Gluon MXFP4-weight MoE for AMD CDNA4/gfx950, covering BF16, FP8, and
dynamically quantized MXFP4 activation paths. The package contains two kernel
families that play the same roles (routing, GEMMs, decode) with different
execution strategies:

* **Staged "package" pipeline** — files directly in this directory. Separate
  kernel launches per stage with precomputed top-k
  (`topk_ids`/`topk_weights`) as the interface between them.
* **Fused single-launch family** — the `fused/` subpackage. Routing, GEMM,
  activation, and quantization collapse into as few launches as possible,
  communicating through `RaggedTensorMetadata` plus gather/scatter index
  tensors.

The fused dispatch policy (`fused/moe.py`) is the top of the funnel: it calls
*into* the staged package for the shapes where that wins (package prefill for
`M >= 9`, staged MFMA decode), so the kernel dependency arrow points from
`fused/` to the staged files, never back.

## Staged package files

| File | Role |
| --- | --- |
| `__init__.py` | Package exports: the stage invokers, the top-k route entries, the prefill weight aliases, and the scale gather. |
| `prefill_stage1.py` / `prefill_stage2.py` | Block-ragged MXFP4/MXFP8-activation, MXFP4-weight prefill GEMMs (gdot128-preshuffled weights). |
| `decode_stage1.py` / `decode_stage2.py` | A4W4 direct-MFMA decode kernels plus their invokers: stage 1 fuses SwiGLU, stage 2 fuses the routed-weight top-k combine. |
| `decode_common.py` | Direct CDNA4 MFMA primitives shared by both decode stages: tile addressing, MXFP4 operand loads, `mfma_scaled`. |
| `routing.py` | Fused dense top-k route kernels (softmax, sigmoid-bias, and a prefill variant); output is top-k only, **not** ragged metadata (contrast `fused/routing.py`). |
| `moe_sorting.py` | Block-aligned expert sort feeding the package prefill stages. |
| `situ_decode.py` / `situ_grouped.py` | A16W4 in-situ expert-parallel paths that read the unshuffled MXFP4 weights directly: route-direct warp decode and a contiguous-EP grouped GEMM. |
| `latent_shared_decode.py` | Latent shared-expert decode entry. |
| `quantize_gluon.py` | Leaf Gluon helpers for MXFP4/MXFP8 tile quantization and CDNA4 scale stores shared by staged and fused kernels. |
| `scale_layout.py` | Single source of truth for the CDNA4 MXFP4 scale swizzle (constants, swizzle/predicate/allocator helpers). Leaf module: torch only. |
| `scale.py` | Activation-scale gather into sorted-route order for the package stages. |
| `preprocess.py` / `weight_preprocess.py` | Offline weight interleave, scale swizzle, gdot128 preshuffle, package-prefill aliases. |

## `fused/` subpackage

Organized by approach; every module states which regime owns it:

| Module | Role |
| --- | --- |
| `moe.py` | Model-facing fused entries + dispatch policy (which kernel runs at which batch size). Start reading here. |
| `gemm_api.py` | Per-GEMM entries: dispatch GEMM + fused SwiGLU, combine GEMM, ragged-matmul router. |
| `pipelined_program.py` | Pipelined ragged GEMM program aggregates: `MoEConfig`, async-copy descriptors, pipelined / slice-MN / slice-N shapes. |
| `pipelined_kernel.py` | Pipelined ragged GEMM tile runners, tile compute body, kernel entry. |
| `medium_decode.py` | M=8/16 single-buffer direct-load body (runs under the pipelined kernel's `IS_MEDIUM_DECODE` constexpr switch). |
| `warp_decode.py` | M<=4 FP8×MXFP4 two-stage warp decode (in-kernel top-k, split-K stage 2). |
| `routing.py` | Single-launch route kernels emitting full ragged metadata + gather/scatter + gate scales; torch references; capability predicates. |
| `quantize.py` | FP8 and dynamic-MXFP4 activation quantization (optionally fused with routed gather). |
| `launch.py` | Host launch marshaling for the pipelined kernel; AMDGCN static-profile / spill checks. |
| `tuning.py` | Block autotuning and launch heuristics. |
| `_common.py` / `_layouts.py` | Shared host floor (constants, ragged helpers, wrapped-tensor extraction) and shared constexpr layout factories / device helpers. |

The same-named file at both levels is deliberate: each fills the
routing role for its own family, and the import path says which family you are
in.
