#include "tvm_ffi_utils.h"

#include <cuda_bf16.h>

namespace tokenspeed { namespace lm_head_gemm {
bool launch_lm_head_gemm(__nv_bfloat16* output, __nv_bfloat16 const* mat_a, __nv_bfloat16 const* mat_b,
                         int num_tokens, int hd_in, int hd_out, int tile_n,
                         bool enable_pdl, cudaStream_t stream);
} }

// Signature:
//   output:  bf16 [num_tokens, hd_out]  row-major, contiguous
//   mat_a:   bf16 [num_tokens, hd_in]   row-major, contiguous
//   mat_b:   bf16 [hd_out,   hd_in]     row-major, contiguous
//   tile_n:  8 or 16 (batch-per-CTA). Caller picks based on num_tokens.
//   enable_pdl: programmatic dependent launch flag.
//
// Persistent-CTA vs static-grid is decided internally by the launcher
// based on (num_tokens, hd_out) and the device's SM count.
//
// Supported (hd_in, hd_out) pairs are fixed at compile time; caller must
// check `lm_head_gemm_supported` first and fall back to torch.matmul if
// the shape is unsupported.
void lm_head_gemm(TensorView output, TensorView mat_a, TensorView mat_b, int64_t tile_n,
                  bool enable_pdl) {
  CHECK_INPUT(output);
  CHECK_INPUT(mat_a);
  CHECK_INPUT(mat_b);
  CHECK_INPUT_TYPE(output, dl_bfloat16);
  CHECK_INPUT_TYPE(mat_a, dl_bfloat16);
  CHECK_INPUT_TYPE(mat_b, dl_bfloat16);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be 2D";
  TVM_FFI_ICHECK_EQ(mat_a.ndim(), 2) << "mat_a must be 2D";
  TVM_FFI_ICHECK_EQ(mat_b.ndim(), 2) << "mat_b must be 2D";
  int64_t num_tokens = mat_a.size(0);
  int64_t hd_in = mat_a.size(1);
  int64_t hd_out = mat_b.size(0);
  TVM_FFI_ICHECK_EQ(mat_b.size(1), hd_in) << "mat_b.size(1) must equal mat_a.size(1)";
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens) << "output.size(0) must equal mat_a.size(0)";
  TVM_FFI_ICHECK_EQ(output.size(1), hd_out) << "output.size(1) must equal mat_b.size(0)";

  cudaStream_t stream = get_stream(output.device());
  bool ok = tokenspeed::lm_head_gemm::launch_lm_head_gemm(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
      static_cast<int>(num_tokens), static_cast<int>(hd_in), static_cast<int>(hd_out),
      static_cast<int>(tile_n), enable_pdl, stream);
  TVM_FFI_ICHECK(ok) << "lm_head_gemm: unsupported (hd_in,hd_out,tile_n)=(" << hd_in << "," << hd_out
                     << "," << tile_n << ")";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(lm_head_gemm, lm_head_gemm);
