// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
 * Fused (residual+input) -> RMSNorm -> NVFP4 block-scale quantize, with an
 * optional bf16/fp16 high-precision normed branch for the MoE gate /
 * shared-expert path.
 *
 * Ports the TRT-LLM ws_layernorm warp-specialized kernel into tokenspeed-kernel.
 * Outputs are caller-allocated; the Python wrapper computes the m_padded sizes
 * (m + 31) // 32 * 32 and narrows the views back to [M, ...] on return so the
 * upstream MoE backend sees the natural shapes.
 *
 * Requires SM90 (Hopper) or SM100 (Blackwell). Hidden dim N must be in the
 * [2048, 16384] range and divisible by 16.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <unordered_map>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "layernorm_param.h"
#include "ws_layernorm.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace tokenspeed {
namespace fused_add_rmsnorm_fp4_quant {

// Per-device cache for the warp-specialized counters (tile_ctr, cta_completion_ctr).
// Initialised to zero on first use.
inline tensorrt_llm::kernels::WarpSpecializedCounters* get_or_create_counters(
    int device_id, cudaStream_t stream) {
  static thread_local std::unordered_map<int, tensorrt_llm::kernels::WarpSpecializedCounters*> cache;
  auto it = cache.find(device_id);
  if (it != cache.end()) {
    return it->second;
  }
  tensorrt_llm::kernels::WarpSpecializedCounters* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, sizeof(tensorrt_llm::kernels::WarpSpecializedCounters));
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "fused_add_rmsnorm_fp4_quant: cudaMalloc counters failed: " << cudaGetErrorString(err);
  // Zero-init asynchronously on the same stream the kernel will use, so subsequent
  // launches observe a consistent counters block without an explicit host sync.
  err = cudaMemsetAsync(ptr, 0, sizeof(tensorrt_llm::kernels::WarpSpecializedCounters), stream);
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "fused_add_rmsnorm_fp4_quant: cudaMemsetAsync counters failed: "
      << cudaGetErrorString(err);
  cache[device_id] = ptr;
  return ptr;
}

}  // namespace fused_add_rmsnorm_fp4_quant
}  // namespace tokenspeed

// Fused Add + RMSNorm + FP4 Quantization with caller-allocated outputs.
//
// Inputs:
//   input        : [M, N] fp16/bf16, contiguous
//   residual     : [M, N] fp16/bf16, contiguous (same dtype as input)
//   gamma        : [N]    fp16/bf16, contiguous (same dtype as input)
//   sf_scale     : Optional [1] float32 (= 1 / input_scale for FP4 quant)
//   eps          : RMSNorm epsilon
//   output_hp_norm: when true, also write hp normed values into hp_norm_out
//
// Caller-allocated outputs (sized to m_padded = (M+31)//32*32 by the caller):
//   normed_fp4_out: [m_padded, N/8] int32 (FP4 packed, 8 FP4 values per int32)
//   sf_out        : [m_padded, N/16] uint8 (linear row-major SF; pass directly
//                   to trtllm_fp4_block_scale_moe after slicing to [M, N/16])
//   residual_out  : [m_padded, N] same dtype as input (= input + residual,
//                  pre-norm, used as next layer's residual)
//   hp_norm_out   : Optional [m_padded, N] same dtype as input; only written
//                  when output_hp_norm=true.
void fused_add_rmsnorm_fp4_quant(TensorView input, TensorView residual, TensorView gamma,
                                 Optional<TensorView> sf_scale, double eps, bool output_hp_norm,
                                 TensorView normed_fp4_out, TensorView sf_out,
                                 TensorView residual_out, Optional<TensorView> hp_norm_out) {
  TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "input must be 2-D [M, N]";
  TVM_FFI_ICHECK_EQ(residual.ndim(), 2) << "residual must be 2-D [M, N]";
  TVM_FFI_ICHECK_EQ(gamma.ndim(), 1) << "gamma must be 1-D [N]";

  int64_t const m = input.size(0);
  int64_t const n = input.size(1);
  int64_t const m_padded = (m + 31) / 32 * 32;

  TVM_FFI_ICHECK_EQ(residual.size(0), m) << "residual.size(0) must equal input.size(0)";
  TVM_FFI_ICHECK_EQ(residual.size(1), n) << "residual.size(1) must equal input.size(1)";
  TVM_FFI_ICHECK_EQ(gamma.size(0), n) << "gamma.size(0) must equal hidden dim N";
  TVM_FFI_ICHECK_GE(n, 2048) << "hidden dim N must be >= 2048 (kernel constraint)";
  TVM_FFI_ICHECK_LE(n, 16384) << "hidden dim N must be <= 16384 (kernel constraint)";
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "hidden dim N must be divisible by 16 for FP4 quant";

  // Outputs must already be sized to m_padded by the caller.
  TVM_FFI_ICHECK_EQ(normed_fp4_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(normed_fp4_out.size(0), m_padded);
  TVM_FFI_ICHECK_EQ(normed_fp4_out.size(1), n / 8);
  TVM_FFI_ICHECK_EQ(residual_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual_out.size(0), m_padded);
  TVM_FFI_ICHECK_EQ(residual_out.size(1), n);
  TVM_FFI_ICHECK_EQ(residual_out.dtype(), input.dtype());

  int64_t const sf_linear = m_padded * (n / 16);
  TVM_FFI_ICHECK_GE(sf_out.numel(), sf_linear)
      << "sf_out is too small for linear FP4 SF layout";

  if (output_hp_norm) {
    TVM_FFI_ICHECK(hp_norm_out.has_value())
        << "hp_norm_out must be provided when output_hp_norm=true";
    TVM_FFI_ICHECK_EQ(hp_norm_out.value().ndim(), 2);
    TVM_FFI_ICHECK_EQ(hp_norm_out.value().size(0), m_padded);
    TVM_FFI_ICHECK_EQ(hp_norm_out.value().size(1), n);
    TVM_FFI_ICHECK_EQ(hp_norm_out.value().dtype(), input.dtype());
  }

  float* sf_scale_ptr = nullptr;
  if (sf_scale.has_value()) {
    TVM_FFI_ICHECK_EQ(sf_scale.value().dtype(), dl_float32) << "sf_scale must be float32";
    TVM_FFI_ICHECK_EQ(sf_scale.value().numel(), 1) << "sf_scale must be a scalar [1]";
    sf_scale_ptr = static_cast<float*>(sf_scale.value().data_ptr());
  }

  // SM90+ kernel; surface a clear error early if running on Ampere or older.
  int const device_id = input.device().device_id;
  cudaSetDevice(device_id);
  cudaDeviceProp props;
  cudaError_t prop_err = cudaGetDeviceProperties(&props, device_id);
  TVM_FFI_ICHECK(prop_err == cudaSuccess)
      << "cudaGetDeviceProperties failed: " << cudaGetErrorString(prop_err);
  TVM_FFI_ICHECK_GE(props.major, 9)
      << "fused_add_rmsnorm_fp4_quant requires SM90 (Hopper) or newer; current sm_"
      << props.major << props.minor;

  cudaStream_t const stream = get_stream(input.device());
  static int const multi_processor_count = tensorrt_llm::common::getMultiProcessorCount();
  auto* counters = tokenspeed::fused_add_rmsnorm_fp4_quant::get_or_create_counters(device_id, stream);

#define LAUNCH_FUSED_ADD_RMS_NORM_QUANT(T)                                                       \
  do {                                                                                            \
    using Param = tensorrt_llm::kernels::GeneralFP4AddBiasResidualPreLayerNormParam<T>;          \
    tensorrt_llm::kernels::WarpSpecializedParam<Param> param;                                    \
    param.normed_output = static_cast<uint32_t*>(normed_fp4_out.data_ptr());                     \
    param.output = static_cast<T*>(residual_out.data_ptr());                                     \
    param.input = static_cast<T*>(input.data_ptr());                                             \
    param.sf_scale = sf_scale_ptr;                                                               \
    param.sf_out = static_cast<uint32_t*>(sf_out.data_ptr());                                    \
    param.residual = static_cast<T const*>(residual.data_ptr());                                 \
    param.bias = nullptr;                                                                         \
    param.gamma = static_cast<T const*>(gamma.data_ptr());                                       \
    param.beta = nullptr;                                                                         \
    param.high_precision_normed_output =                                                         \
        output_hp_norm ? static_cast<T*>(hp_norm_out.value().data_ptr()) : nullptr;              \
    param.m = static_cast<int>(m);                                                                \
    param.n = static_cast<int>(n);                                                                \
    param.layernorm_eps = static_cast<float>(eps);                                                \
    param.stream = stream;                                                                        \
    param.counters = counters;                                                                    \
    tensorrt_llm::kernels::invokeWSLayerNorm<Param>(param, /*use_rms_norm=*/true,                \
                                                    multi_processor_count, output_hp_norm);     \
  } while (0)

  switch (encode_dlpack_dtype(input.dtype())) {
    case float16_code: {
      LAUNCH_FUSED_ADD_RMS_NORM_QUANT(half);
      break;
    }
    case bfloat16_code: {
#ifdef ENABLE_BF16
      LAUNCH_FUSED_ADD_RMS_NORM_QUANT(__nv_bfloat16);
#else
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "fused_add_rmsnorm_fp4_quant: BF16 must be enabled (-DENABLE_BF16) for bf16 input.";
#endif
      break;
    }
    default:
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "fused_add_rmsnorm_fp4_quant only supports fp16/bf16 input; got code "
          << input.dtype().code << ", bits " << input.dtype().bits;
  }

#undef LAUNCH_FUSED_ADD_RMS_NORM_QUANT

  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "fused_add_rmsnorm_fp4_quant launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rmsnorm_fp4_quant, fused_add_rmsnorm_fp4_quant);
