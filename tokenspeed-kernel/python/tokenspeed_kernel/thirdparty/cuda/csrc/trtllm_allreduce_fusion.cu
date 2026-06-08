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

#include <assert.h>

#include <string>

#include "flashinfer/comm/trtllm_allreduce_fusion.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::trtllm_allreduce_fusion;

using tvm::ffi::Optional;

#define DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(dtype, c_type, ...)             \
  [&] {                                                                       \
    switch (encode_dlpack_dtype(dtype)) {                                     \
      case float16_code: {                                                    \
        using c_type = half;                                                  \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case bfloat16_code: {                                                   \
        using c_type = __nv_bfloat16;                                         \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case float32_code: {                                                    \
        using c_type = float;                                                 \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default:                                                                \
        TVM_FFI_LOG_AND_THROW(NotImplementedError)                            \
            << "Unsupported dtype in DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE."; \
    }                                                                         \
  }()

void trtllm_allreduce_fusion(TensorView allreduce_in, int64_t world_size, int64_t world_rank,
                             int64_t token_num, int64_t hidden_size, TensorView workspace_ptrs,
                             bool launch_with_pdl, bool use_oneshot, bool trigger_completion_at_end,
                             bool fp32_acc, bool residual_reduce_scattered, int64_t pattern_code,
                             Optional<TensorView> allreduce_out, Optional<TensorView> residual_in,
                             Optional<TensorView> residual_out, Optional<TensorView> norm_out,
                             Optional<TensorView> partial_normed_out, Optional<TensorView> quant_out,
                             Optional<TensorView> scale_out, Optional<TensorView> rms_gamma, Optional<double> rms_eps,
                             Optional<TensorView> scale_factor, Optional<int64_t> layout_code,
                             Optional<int64_t> max_sm_to_use) {
  cudaSetDevice(allreduce_in.device().device_id);
  // todo(Yingyi): add dispatch for float and bfloat16

  DISPATCH_FLOATING_TYPES_FOR_ALLREDUCE(allreduce_in.dtype(), c_type, [&] {
    AllReduceFusionParams<c_type> params;
    params.nranks = world_size;
    params.rank = world_rank;
    params.size = token_num * hidden_size;
    params.hidden_dim = hidden_size;
    params.workspace = reinterpret_cast<void**>(workspace_ptrs.data_ptr());

    // todo(Yingyi): update optional params
    // todo(Yingyi): add params check with pattern
    params.allreduce_in = reinterpret_cast<void*>(allreduce_in.data_ptr());
    params.allreduce_out = allreduce_out.has_value()
                               ? reinterpret_cast<void*>(allreduce_out.value().data_ptr())
                               : nullptr;
    params.residual_in =
        residual_in.has_value() ? reinterpret_cast<void*>(residual_in.value().data_ptr()) : nullptr;
    params.residual_out = residual_out.has_value()
                              ? reinterpret_cast<void*>(residual_out.value().data_ptr())
                              : nullptr;
    params.norm_out =
        norm_out.has_value() ? reinterpret_cast<void*>(norm_out.value().data_ptr()) : nullptr;
    params.partial_normed_out =
        partial_normed_out.has_value()? reinterpret_cast<void*>(partial_normed_out.value().data_ptr()) : nullptr;
    params.quant_out =
        quant_out.has_value() ? reinterpret_cast<void*>(quant_out.value().data_ptr()) : nullptr;
    params.scale_out =
        scale_out.has_value() ? reinterpret_cast<void*>(scale_out.value().data_ptr()) : nullptr;
    params.scale_stride = scale_out.has_value() ? int32_t(scale_out.value().stride(1)) : 0;
    params.residual_reduce_scattered = residual_reduce_scattered;
    if (residual_reduce_scattered) {
      TVM_FFI_ICHECK(use_oneshot) << "residual_reduce_scattered only support use_oneshot";
    }
    params.rms_gamma =
        rms_gamma.has_value() ? reinterpret_cast<void*>(rms_gamma.value().data_ptr()) : nullptr;
    params.rms_eps = rms_eps.has_value() ? static_cast<float>(rms_eps.value()) : 0.0f;
    params.scale_factor = scale_factor.has_value()
                              ? reinterpret_cast<float*>(scale_factor.value().data_ptr())
                              : nullptr;
    params.use_oneshot = use_oneshot;
    params.layout = layout_code.has_value() ? static_cast<QuantizationSFLayout>(layout_code.value())
                                            : QuantizationSFLayout::SWIZZLED_128x4;
    params.pattern = static_cast<AllReduceFusionPattern>(pattern_code);
    if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormFP8BlockWiseQuant) {
      // check not float32
      TVM_FFI_ICHECK_NE(allreduce_in.dtype(), dl_float32) << "Only bf16 and half supported";
      // check scale_factor not nullptr
      TVM_FFI_ICHECK(scale_out.has_value()) << "scale_out must be provided";
      // check col-major and alignment
      TVM_FFI_ICHECK_GT(scale_out.value().stride(1), scale_out.value().stride(0))
          << "scale_out must be column-major";
      TVM_FFI_ICHECK_EQ(scale_out.value().stride(1) % 4, 0)
          << "scale_out stride(1) must be a multiple of 4 for TMA alignment";
    }
    params.trigger_completion_at_end = trigger_completion_at_end;
    params.stream = get_stream(allreduce_in.device());
    int max_sm_to_use_;
    if (max_sm_to_use.has_value()) {
      max_sm_to_use_ = static_cast<int>(max_sm_to_use.value());
      TVM_FFI_ICHECK(max_sm_to_use_ <= get_sm_count());
    } else {
      max_sm_to_use_ = get_sm_count();
    }
    auto status = allreduce_fusion_op(params, launch_with_pdl, fp32_acc, max_sm_to_use_);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "allreduce_fusion_op failed with error code" << cudaGetErrorString(status);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_allreduce_fusion, trtllm_allreduce_fusion);
