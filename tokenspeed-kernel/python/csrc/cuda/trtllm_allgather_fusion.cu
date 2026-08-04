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

#include <string>

#include "flashinfer/comm/trtllm_allgather_fusion.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_allgather_fusion;

using tvm::ffi::Optional;

#define DISPATCH_FLOATING_TYPES_FOR_ALLGATHER(dtype, c_type, ...)                 \
  [&] {                                                                           \
    switch (encode_dlpack_dtype(dtype)) {                                         \
      case float16_code: {                                                        \
        using c_type = half;                                                      \
        return __VA_ARGS__();                                                     \
      }                                                                           \
      case bfloat16_code: {                                                       \
        using c_type = __nv_bfloat16;                                             \
        return __VA_ARGS__();                                                     \
      }                                                                           \
      case float32_code: {                                                        \
        using c_type = float;                                                     \
        return __VA_ARGS__();                                                     \
      }                                                                           \
      default:                                                                    \
        TVM_FFI_LOG_AND_THROW(NotImplementedError)                                \
            << "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_ALLGATHER.";     \
    }                                                                             \
  }()

void trtllm_allgather_fusion(TensorView allgather_in, int64_t world_size,
                             int64_t world_rank, int64_t hidden_size,
                             TensorView workspace_ptrs, bool launch_with_pdl,
                             bool use_oneshot, bool trigger_completion_at_end, bool fp32_acc,
                             int64_t pattern_code, int64_t num_token_current_rank,
                             int64_t num_token_all_group,
                             TensorView allgather_out,
                             Optional<TensorView> x_norm_out, Optional<TensorView> y_norm_out,
                             Optional<TensorView> quant_out, Optional<TensorView> scale_out,
                             Optional<TensorView> x_rms_gamma, Optional<TensorView> y_rms_gamma,
                             Optional<double> x_rms_eps, Optional<double> y_rms_eps,
                             int64_t q_lora_rank, int64_t kv_lora_rank, int64_t qk_rope_head_dim) {
  cudaSetDevice(allgather_in.device().device_id);

  CHECK_CONTIGUOUS(allgather_out);
  if (x_norm_out.has_value()) {
    CHECK_CONTIGUOUS(x_norm_out.value());
  }
  if (y_norm_out.has_value()) {
    CHECK_LAST_DIM_CONTIGUOUS(y_norm_out.value());
  }
  if (quant_out.has_value()) {
    CHECK_CONTIGUOUS(quant_out.value());
  }

  DISPATCH_FLOATING_TYPES_FOR_ALLGATHER(allgather_in.dtype(), c_type, [&] {
    AllGatherFusionParams<c_type> params;
    params.nranks = world_size;
    params.rank = world_rank;
    params.size = num_token_all_group * hidden_size;
    params.hidden_dim = hidden_size;
    params.workspace = reinterpret_cast<void**>(workspace_ptrs.data_ptr());

    params.allgather_in = reinterpret_cast<void*>(allgather_in.data_ptr());
    params.num_token_current_rank = num_token_current_rank;
    params.num_token_all_group = num_token_all_group;
    params.allgather_out = reinterpret_cast<void*>(allgather_out.data_ptr());

    params.x_norm_out = x_norm_out.has_value() ? reinterpret_cast<void*>(x_norm_out.value().data_ptr()) : nullptr;
    params.y_norm_out = y_norm_out.has_value() ? reinterpret_cast<void*>(y_norm_out.value().data_ptr()) : nullptr;
    params.y_norm_stride = y_norm_out.has_value() ? static_cast<int32_t>(y_norm_out.value().stride(0)) : 0;
    params.quant_out = quant_out.has_value() ? reinterpret_cast<void*>(quant_out.value().data_ptr()) : nullptr;
    params.scale_out = scale_out.has_value() ? reinterpret_cast<void*>(scale_out.value().data_ptr()) : nullptr;
    params.x_rms_gamma = x_rms_gamma.has_value() ? reinterpret_cast<void*>(x_rms_gamma.value().data_ptr()) : nullptr;
    params.y_rms_gamma = y_rms_gamma.has_value() ? reinterpret_cast<void*>(y_rms_gamma.value().data_ptr()) : nullptr;
    params.x_rms_eps = x_rms_eps.has_value() ? static_cast<float>(x_rms_eps.value()) : 0.0f;
    params.y_rms_eps = y_rms_eps.has_value() ? static_cast<float>(y_rms_eps.value()) : 0.0f;
    params.q_lora_rank = q_lora_rank;
    params.kv_lora_rank = kv_lora_rank;
    params.qk_rope_head_dim = qk_rope_head_dim;
    params.scale_stride = scale_out.has_value() ? static_cast<int32_t>(scale_out.value().stride(1)) : 0;

    static constexpr int VEC_SIZE = 16 / sizeof(c_type);  // kBytesPerAccess / sizeof(T)
    params.q_lora_access_end = (q_lora_rank + VEC_SIZE - 1) / VEC_SIZE;
    params.kv_lora_start_access = q_lora_rank / VEC_SIZE;
    params.kv_lora_access_end = (q_lora_rank + kv_lora_rank + VEC_SIZE - 1) / VEC_SIZE;
    params.use_oneshot = use_oneshot;
    params.pattern = static_cast<AllGatherFusionPattern>(pattern_code);
    if (params.pattern == AllGatherFusionPattern::kAllGatherfusedRMSFP8BlockWiseQuant) {
      // check not float32
      TVM_FFI_ICHECK_NE(allgather_in.dtype(), dl_float32) << "Only bf16 and half supported";
      // check col-major for scale_out
      if (scale_out.has_value() && scale_out.value().shape()[0] > 0) {
        TVM_FFI_ICHECK(scale_out.value().stride(1) > scale_out.value().stride(0))
            << "scale_out must be col-major (stride[1] > stride[0])";
        TVM_FFI_ICHECK_EQ(scale_out.value().stride(1) % 4, 0)
            << "scale_out stride(1) must be a multiple of 4 for TMA alignment";
      }
      TVM_FFI_ICHECK(use_oneshot) << "kAllGatherfusedRMSFP8BlockWiseQuant only supports oneshot mode now!";
    }
    params.trigger_completion_at_end = trigger_completion_at_end;
    params.stream = get_stream(allgather_in.device());

    auto status = allgather_fusion_op(params, launch_with_pdl);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "allgather_fusion_op failed with error code" << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_allgather_fusion, trtllm_allgather_fusion);
