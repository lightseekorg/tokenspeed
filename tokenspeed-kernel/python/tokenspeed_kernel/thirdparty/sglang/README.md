# SGLang small-M FP8 dense GEMM

`fp8_small_m_gemm.py` adapts kernels from [SGLang](https://github.com/sgl-project/sglang)
(Apache-2.0, `LICENSE`) at commit `418f4db7debc`:

- `python/sglang/kernels/ops/gemm/fp8w_small_m_gemm.py`: the split-K FP8
  GEMM (`_fp8_rowmajor_partial`), its scaled reduction (`_sum_scaled`), the
  per-channel weight conversion (`quantize_weight`, here
  `quantize_weight_fp8_channel`) and its tuning table (`CONFIGS`, here
  `FP8W_SMALL_M_CONFIGS`, all 15 entries);
- `python/sglang/srt/rocm_muse_spark/fused_norm_residual.py`: the per-row FP8
  copy its fused norm kernel writes (scale = max(row amax, 1e-20) / 448,
  codes = row / scale), here `fp8_row_codes`, and `_quantize_rows_kernel`,
  which applies it to rows on their own.

`tokenspeed_kernel.ops.gemm.triton_fp8_channel_splitk` registers and launches
the GEMM and adds tuning entries for shapes and row tiles the table lacks;
`tokenspeed_kernel.ops.quantization` registers the row quantizer
(`token_fp8_amax_div`), and `tokenspeed_kernel.ops.residual` reuses
`fp8_row_codes` so the layer boundary's FP8 copy has the same bits.

## Local changes

- Triton comes from `tokenspeed_kernel._triton`.
- `_fp8_rowmajor_partial` and `_sum_scaled` take the row count as a
  non-specialized runtime argument (`do_not_specialize=["M"]`); their bodies
  are unchanged.
- `quantize_weight_fp8_channel` converts the weight to FP32 once, leaves the
  dtype check to its caller, and multiplies the channel amax by FP32
  `1 / 448` where SGLang divides by 448 (the same bits on the GPU, where
  PyTorch evaluates the division that way).
- `fp8_row_codes` and `_quantize_rows_kernel` are new functions holding the
  fused norm kernel's FP8 lines.
