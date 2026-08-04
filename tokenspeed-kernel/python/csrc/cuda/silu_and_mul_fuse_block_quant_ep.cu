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
 * Materialized from flashinfer JIT template: activation_fuse_block_quant_templ_ep
 * act_func_name = silu
 */
#include <flashinfer/activation.cuh>
#include <cuda_runtime.h>
#include "tvm_ffi_utils.h"

using namespace flashinfer;

__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

void silu_and_mul_fused_block_quant_ep(TensorView out, TensorView scale_out, TensorView input, bool enable_pdl, TensorView num_tokens_per_expert, int64_t num_tokens_hint, int64_t num_experts) {
  int d = input.size(input.ndim() -1) / 2;
  int64_t num_tokens = num_tokens_hint;
  dim3 grid(num_experts, num_tokens_hint);

  cudaSetDevice(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    TVM_FFI_ICHECK(d % vec_size == 0);
    // Check Column Major
    TVM_FFI_ICHECK(scale_out.stride(2) > scale_out.stride(1));
    int32_t scale_stride_expert = int32_t(scale_out.stride(0));
    int32_t scale_stride_col = int32_t(scale_out.stride(2));
    int32_t out_stride_0 = int32_t(out.stride(0));
    int32_t gate_up_stride_0 = int32_t(input.stride(0));
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;
    auto kernel = flashinfer::activation::act_and_mul_post_block_quant_128_kernel_ep<c_type, silu>;
    cudaLaunchKernelEx(&config, kernel, static_cast<__nv_fp8_e4m3*>(out.data_ptr()), static_cast<float*>(scale_out.data_ptr()),
                      static_cast<c_type*>(input.data_ptr()), static_cast<int32_t*> (num_tokens_per_expert.data_ptr()), gate_up_stride_0, out_stride_0, scale_stride_expert, scale_stride_col, d, static_cast<int32_t>(num_experts));


    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "Failed to launch kernel: " << cudaGetErrorString(err);

    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_and_mul_fused_block_quant_ep, silu_and_mul_fused_block_quant_ep);
