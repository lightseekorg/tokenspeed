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

// TVM-FFI entry point for fp32_router_gemm: activation(bf16/fp32) x weight(fp32) -> fp32.
// Custom kernel for M<=32, E=256, H=3072; cuBLAS fallback for larger M.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "tvm_ffi_utils.h"

using tvm::ffi::Tensor;

static constexpr int FP32_NUM_EXPERTS = 256;
static constexpr int FP32_HIDDEN_DIM = 3072;
static constexpr int FP32_MAX_TOKENS = 32;

// Forward declaration — must match fp32_router_gemm.cu
template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a,
                          float const* mat_b, cudaStream_t stream);

// LoopUnroller to dispatch compile-time token count
template <typename InputT, int kBegin, int kEnd>
struct Fp32LoopUnroller {
  static void unroll(int num_tokens, float* output, InputT const* mat_a,
                     float const* mat_b, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeFp32RouterGemm<InputT, kBegin, FP32_NUM_EXPERTS, FP32_HIDDEN_DIM>(
          output, mat_a, mat_b, stream);
    } else {
      Fp32LoopUnroller<InputT, kBegin + 1, kEnd>::unroll(num_tokens, output,
                                                         mat_a, mat_b, stream);
    }
  }
};

template <typename InputT, int kEnd>
struct Fp32LoopUnroller<InputT, kEnd, kEnd> {
  static void unroll(int num_tokens, float* output, InputT const* mat_a,
                     float const* mat_b, cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeFp32RouterGemm<InputT, kEnd, FP32_NUM_EXPERTS, FP32_HIDDEN_DIM>(
          output, mat_a, mat_b, stream);
    } else {
      throw std::invalid_argument(
          "fp32_router_gemm: num_tokens must be in [1, 32]");
    }
  }
};

// ---------------------------------------------------------------------------
// cuBLAS fallback for M > 32
// ---------------------------------------------------------------------------

static inline void check_cublas(cublasStatus_t st) {
  if (st == CUBLAS_STATUS_SUCCESS) return;
  const char* name = "CUBLAS_STATUS_UNKNOWN";
  switch (st) {
    case CUBLAS_STATUS_NOT_INITIALIZED: name = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
    case CUBLAS_STATUS_ALLOC_FAILED: name = "CUBLAS_STATUS_ALLOC_FAILED"; break;
    case CUBLAS_STATUS_INVALID_VALUE: name = "CUBLAS_STATUS_INVALID_VALUE"; break;
    case CUBLAS_STATUS_ARCH_MISMATCH: name = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
    case CUBLAS_STATUS_EXECUTION_FAILED: name = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
    case CUBLAS_STATUS_NOT_SUPPORTED: name = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
    default: break;
  }
  TVM_FFI_ICHECK(false) << "cublas error: " << name << " (" << int(st) << ")";
}

__global__ void fp32_rg_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ in, float* __restrict__ out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

static void cublas_fp32_router_gemm(
    float* output, const void* mat_a, cublasDataType_t a_type,
    const float* mat_b, int m, int n, int k, cudaStream_t stream) {
  static thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    check_cublas(cublasCreate(&handle));
  }
  check_cublas(cublasSetStream(handle, stream));

  float alpha = 1.0f;
  float beta = 0.0f;

  // Column-major: (n x k)^T * (k x m) => (n x m)
  (void)cublasSetMathMode(handle, (a_type == CUDA_R_32F)
                                      ? CUBLAS_PEDANTIC_MATH
                                      : CUBLAS_TENSOR_OP_MATH);
  cublasStatus_t st = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      /*m=*/n, /*n=*/m, /*k=*/k,
      &alpha, mat_b, CUDA_R_32F, k, mat_a, a_type, k,
      &beta, output, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (st == CUBLAS_STATUS_NOT_SUPPORTED) {
    st = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha, mat_b, CUDA_R_32F, k, mat_a, a_type, k,
        &beta, output, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  check_cublas(st);
}

// ---------------------------------------------------------------------------
// SM version helper
// ---------------------------------------------------------------------------

static inline int get_sm_version(int device_id) {
  int sm_major = 0, sm_minor = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return sm_major * 10 + sm_minor;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

void fp32_router_gemm(TensorView output, TensorView mat_a, TensorView mat_b) {
  CHECK_CUDA(output);
  CHECK_CUDA(mat_a);
  CHECK_CUDA(mat_b);
  CHECK_DEVICE(output, mat_a);
  CHECK_DEVICE(output, mat_b);
  CHECK_DIM(2, output);
  CHECK_DIM(2, mat_a);
  CHECK_DIM(2, mat_b);

  TVM_FFI_ICHECK(mat_a.dtype() == dl_bfloat16 || mat_a.dtype() == dl_float32)
      << "mat_a must be bf16 or fp32";
  TVM_FFI_ICHECK(mat_b.dtype() == dl_float32) << "mat_b (weight) must be fp32";
  TVM_FFI_ICHECK(output.dtype() == dl_float32) << "output must be fp32";

  int num_tokens = static_cast<int>(mat_a.size(0));
  int hidden_dim = static_cast<int>(mat_a.size(1));
  int num_experts = static_cast<int>(mat_b.size(0));
  TVM_FFI_ICHECK_EQ(mat_b.size(1), hidden_dim);
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), num_experts);
  TVM_FFI_ICHECK_EQ(hidden_dim, FP32_HIDDEN_DIM) << "expected hidden_dim=3072";
  TVM_FFI_ICHECK_EQ(num_experts, FP32_NUM_EXPERTS) << "expected num_experts=256";

  auto device = mat_a.device();
  int sm = get_sm_version(device.device_id);
  TVM_FFI_ICHECK(sm >= 90) << "requires SM90+";

  cudaStream_t stream = get_stream(device);
  float* out_ptr = static_cast<float*>(output.data_ptr());
  float const* mat_b_ptr = static_cast<float const*>(mat_b.data_ptr());

  // Custom kernel path: M <= 32
  if (num_tokens >= 1 && num_tokens <= FP32_MAX_TOKENS) {
    if (mat_a.dtype() == dl_bfloat16) {
      auto const* mat_a_ptr =
          static_cast<__nv_bfloat16 const*>(mat_a.data_ptr());
      Fp32LoopUnroller<__nv_bfloat16, 1, FP32_MAX_TOKENS>::unroll(
          num_tokens, out_ptr, mat_a_ptr, mat_b_ptr, stream);
    } else {
      auto const* mat_a_ptr = static_cast<float const*>(mat_a.data_ptr());
      Fp32LoopUnroller<float, 1, FP32_MAX_TOKENS>::unroll(
          num_tokens, out_ptr, mat_a_ptr, mat_b_ptr, stream);
    }
    return;
  }

  // cuBLAS fallback for M > 32
  if (mat_a.dtype() == dl_bfloat16) {
    // Cast bf16 -> fp32, then fp32 x fp32 GEMM
    const int64_t numel = static_cast<int64_t>(num_tokens) * hidden_dim;
    Tensor a_fp32_tensor = alloc_tensor({numel}, dl_float32, device);
    float* a_fp32 = static_cast<float*>(a_fp32_tensor.data_ptr());
    dim3 block(256);
    dim3 grid((numel + block.x - 1) / block.x);
    fp32_rg_bf16_to_f32_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(mat_a.data_ptr()), a_fp32, numel);
    cublas_fp32_router_gemm(out_ptr, a_fp32, CUDA_R_32F, mat_b_ptr,
                            num_tokens, num_experts, hidden_dim, stream);
  } else {
    cublas_fp32_router_gemm(out_ptr, mat_a.data_ptr(), CUDA_R_32F, mat_b_ptr,
                            num_tokens, num_experts, hidden_dim, stream);
  }
}
