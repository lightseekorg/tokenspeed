// DSV3 router GEMM entry kernel.
// JIT-only integration for FlashInfer (TVM-FFI).

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "tvm_ffi_utils.h"

using tvm::ffi::Tensor;

template <typename ADtype, typename BDtype, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmFloatOutput(float* output, ADtype const* mat_a, BDtype const* mat_b, bool enable_pdl, cudaStream_t stream);

static inline int get_sm_version(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
  {
    cudaError_t status = cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaDeviceGetAttribute(ComputeCapabilityMajor) failed: " << cudaGetErrorString(status);
  }
  {
    cudaError_t status = cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaDeviceGetAttribute(ComputeCapabilityMinor) failed: " << cudaGetErrorString(status);
  }
  return sm_major * 10 + sm_minor;
}

static inline void check_cublas(cublasStatus_t st) {
  if (st == CUBLAS_STATUS_SUCCESS) return;
  const char* name = "CUBLAS_STATUS_UNKNOWN";
  switch (st) {
    case CUBLAS_STATUS_NOT_INITIALIZED: name = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
    case CUBLAS_STATUS_ALLOC_FAILED: name = "CUBLAS_STATUS_ALLOC_FAILED"; break;
    case CUBLAS_STATUS_INVALID_VALUE: name = "CUBLAS_STATUS_INVALID_VALUE"; break;
    case CUBLAS_STATUS_ARCH_MISMATCH: name = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
    case CUBLAS_STATUS_MAPPING_ERROR: name = "CUBLAS_STATUS_MAPPING_ERROR"; break;
    case CUBLAS_STATUS_EXECUTION_FAILED: name = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
    case CUBLAS_STATUS_INTERNAL_ERROR: name = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
    case CUBLAS_STATUS_NOT_SUPPORTED: name = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
    default: break;
  }
  TVM_FFI_ICHECK(false) << "cublas error: " << name << " (" << int(st) << ")";
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ in, float* __restrict__ out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

static void cublas_fallback_router_gemm(
    float* output, const void* mat_a, cublasDataType_t a_type,
    const void* mat_b, cublasDataType_t b_type,
    int m, int n, int k, cudaStream_t stream) {
  static thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    check_cublas(cublasCreate(&handle));
  }
  check_cublas(cublasSetStream(handle, stream));

  float alpha = 1.0f;
  float beta = 0.0f;

  // Column-major GEMM: (n x k) * (k x m) => (n x m)
  int m_col = n;
  int n_col = m;
  int k_col = k;
  int ldA = k;  // A_col: k x n
  int ldB = k;  // B_col: k x m
  int ldC = n;  // C_col: n x m

  const bool fp32_gemm = (a_type == CUDA_R_32F && b_type == CUDA_R_32F);
  if (fp32_gemm) {
    (void)cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, m_col, n_col, k_col, &alpha, mat_b, b_type, ldA, mat_a,
        a_type, ldB, &beta, output, CUDA_R_32F, ldC, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    check_cublas(st);
  } else {
    (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, m_col, n_col, k_col, &alpha, mat_b, b_type, ldA, mat_a,
        a_type, ldB, &beta, output, CUDA_R_32F, ldC, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st == CUBLAS_STATUS_NOT_SUPPORTED) {
      st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m_col, n_col, k_col, &alpha, mat_b, b_type,
                        ldA, mat_a, a_type, ldB, &beta, output, CUDA_R_32F, ldC, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT);
    }
    check_cublas(st);
  }
}

template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim, typename ADtype, typename BDtype>
struct LoopUnroller {
  static void unroll_float_output(
      int num_tokens, float* output, ADtype const* input, BDtype const* weights, bool enable_pdl, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeRouterGemmFloatOutput<ADtype, BDtype, kBegin, kNumExperts, kHiddenDim>(output, input, weights, enable_pdl, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts, kHiddenDim, ADtype, BDtype>::unroll_float_output(
          num_tokens, output, input, weights, enable_pdl, stream);
    }
  }
};

template <int kEnd, int kNumExperts, int kHiddenDim, typename ADtype, typename BDtype>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim, ADtype, BDtype> {
  static void unroll_float_output(
      int num_tokens, float* output, ADtype const* input, BDtype const* weights, bool enable_pdl, cudaStream_t stream) {
    if (num_tokens == kEnd) {
      invokeRouterGemmFloatOutput<ADtype, BDtype, kEnd, kNumExperts, kHiddenDim>(output, input, weights, enable_pdl, stream);
    } else {
      throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
    }
  }
};

void dsv3_router_gemm(TensorView output, TensorView mat_a, TensorView mat_b, bool enable_pdl) {
  CHECK_CUDA(output);
  CHECK_CUDA(mat_a);
  CHECK_CUDA(mat_b);
  CHECK_DEVICE(output, mat_a);
  CHECK_DEVICE(output, mat_b);

  CHECK_DIM(2, output);
  CHECK_DIM(2, mat_a);
  CHECK_DIM(2, mat_b);

  TVM_FFI_ICHECK(mat_a.dtype() == dl_bfloat16) << "mat_a must be bf16";
  TVM_FFI_ICHECK(mat_b.dtype() == dl_bfloat16 || mat_b.dtype() == dl_float32) << "mat_b must be bf16 or float32";
  TVM_FFI_ICHECK(output.dtype() == dl_float32) << "output must be float32";

  int num_tokens = static_cast<int>(mat_a.size(0));
  int hidden_dim = static_cast<int>(mat_a.size(1));
  int num_experts = static_cast<int>(mat_b.size(0));
  TVM_FFI_ICHECK_EQ(mat_b.size(1), hidden_dim) << "mat_a and mat_b must have same hidden_dim";
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), num_experts);

  auto device = mat_a.device();
  int sm = get_sm_version(device.device_id);
  TVM_FFI_ICHECK(sm >= 90) << "required CUDA ARCH >= SM_90";

  cudaStream_t stream = get_stream(device);

  bool use_custom_kernel = (num_tokens >= 1 && num_tokens <= 8);
  bool supported_experts = (num_experts == 256 || num_experts == 384 || num_experts == 768);
  bool supported_hidden = (hidden_dim == 3072 || hidden_dim == 6144 || hidden_dim == 7168);

  if (use_custom_kernel && supported_experts && supported_hidden) {
    if (mat_b.dtype() == dl_bfloat16) {
      if (num_experts == 256 && hidden_dim == 3072) {
        LoopUnroller<1, 16, 256, 3072, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens,
            static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 384 && hidden_dim == 3072) {
        LoopUnroller<1, 16, 384, 3072, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 768 && hidden_dim == 3072) {
        LoopUnroller<1, 16, 768, 3072, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 256 && hidden_dim == 6144) {
        LoopUnroller<1, 16, 256, 6144, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 384 && hidden_dim == 6144) {
        LoopUnroller<1, 16, 384, 6144, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 768 && hidden_dim == 6144) {
        LoopUnroller<1, 16, 768, 6144, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 256 && hidden_dim == 7168) {
        LoopUnroller<1, 16, 256, 7168, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 384 && hidden_dim == 7168) {
        LoopUnroller<1, 16, 384, 7168, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 768 && hidden_dim == 7168) {
        LoopUnroller<1, 16, 768, 7168, __nv_bfloat16, __nv_bfloat16>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
    } else {
      // mat_b float32
      if (num_experts == 256 && hidden_dim == 7168) {
        LoopUnroller<1, 16, 256, 7168, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 384 && hidden_dim == 7168) {
        LoopUnroller<1, 16, 384, 7168, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 768 && hidden_dim == 7168) {
        LoopUnroller<1, 16, 768, 7168, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 256 && hidden_dim == 6144) {
        LoopUnroller<1, 16, 256, 6144, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 384 && hidden_dim == 6144) {
        LoopUnroller<1, 16, 384, 6144, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 768 && hidden_dim == 6144) {
        LoopUnroller<1, 16, 768, 6144, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 256 && hidden_dim == 3072) {
        LoopUnroller<1, 16, 256, 3072, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 384 && hidden_dim == 3072) {
        LoopUnroller<1, 16, 384, 3072, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
      if (num_experts == 768 && hidden_dim == 3072) {
        LoopUnroller<1, 16, 768, 3072, __nv_bfloat16, float>::unroll_float_output(
            num_tokens, static_cast<float*>(output.data_ptr()),
            static_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
            static_cast<float const*>(mat_b.data_ptr()),
            enable_pdl,
            stream);
        return;
      }
    }
  }

  // Fallback to cuBLAS for other shapes to preserve existing behavior.
  const bool b_is_fp32 = (mat_b.dtype() == dl_float32);

  // If weights are fp32, cast mat_a to fp32 then GEMM (fp32 x fp32 -> fp32).
  // Do not try bf16 x fp32 GEMM first; it can introduce accuracy drift vs torch fp32 reference.
  if (b_is_fp32) {
    const int64_t numel = static_cast<int64_t>(num_tokens) * static_cast<int64_t>(hidden_dim);
    Tensor a_fp32_tensor = alloc_tensor({numel}, dl_float32, device);
    float* a_fp32 = static_cast<float*>(a_fp32_tensor.data_ptr());
    dim3 block(256);
    dim3 grid((numel + block.x - 1) / block.x);
    bf16_to_f32_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(mat_a.data_ptr()), a_fp32, numel);
    cudaError_t s2 = cudaGetLastError();
    TVM_FFI_ICHECK(s2 == cudaSuccess) << "bf16_to_f32_kernel failed: " << cudaGetErrorString(s2);

    cublas_fallback_router_gemm(
        static_cast<float*>(output.data_ptr()),
        a_fp32,
        CUDA_R_32F,
        mat_b.data_ptr(),
        CUDA_R_32F,
        num_tokens,
        num_experts,
        hidden_dim,
        stream);
    return;
  }

  // bf16 x bf16 -> fp32
  cublas_fallback_router_gemm(
      static_cast<float*>(output.data_ptr()),
      mat_a.data_ptr(),
      CUDA_R_16BF,
      mat_b.data_ptr(),
      CUDA_R_16BF,
      num_tokens,
      num_experts,
      hidden_dim,
      stream);
}
