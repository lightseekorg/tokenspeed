// DSV3 router GEMM float-output kernel.
// JIT-only integration for FlashInfer (TVM-FFI).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

// Custom FMA implementation using PTX assembly instructions
__device__ __forceinline__ void fma(float2& d, float2 const& a, float2 const& b, float2 const& c) {
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(d))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
}

// Convert 8 bfloat16 values from a uint4 to float array - optimized conversion
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec, float* dst) {
  __nv_bfloat16* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));
#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}

template <typename ADtype, typename BDtype, int VPT, int kBlockSize, int kNumTokens, int kNumExperts, int kHiddenDim>
__global__ __launch_bounds__(128, 1) void router_gemm_kernel_float_output(float* out, ADtype const* mat_a, BDtype const* mat_b) {
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  BDtype const* b_col = mat_b + n_idx * kHiddenDim;

  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];

    float b_float[VPT];
    if constexpr (std::is_same_v<BDtype, float>) {
      float4* float4_ptr = reinterpret_cast<float4*>(const_cast<BDtype*>(b_col + k_base));
      float4 f4_0 = float4_ptr[0];
      float4 f4_1 = float4_ptr[1];
      b_float[0] = f4_0.x;
      b_float[1] = f4_0.y;
      b_float[2] = f4_0.z;
      b_float[3] = f4_0.w;
      b_float[4] = f4_1.x;
      b_float[5] = f4_1.y;
      b_float[6] = f4_1.z;
      b_float[7] = f4_1.w;
    } else if constexpr (std::is_same_v<BDtype, __nv_bfloat16>) {
      uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);
      bf16_uint4_to_float8<VPT>(b_vec, b_float);
    } else {
      assert(false);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (ki == 0) {
      asm volatile("griddepcontrol.wait;");
    }
#endif

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      uint4 a_vec = *reinterpret_cast<uint4 const*>(mat_a + (m_idx * kHiddenDim) + k_base);
      float a_float[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_float);

#pragma unroll
      for (int k = 0; k < VPT; k++) {
        acc[m_idx] += a_float[k] * b_float[k];
      }
    }
  }

  int const warp_id = tid / kWarpSize;
  int const lane_id = tid % kWarpSize;

#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    float val = acc[m_idx];
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    if (lane_id == 0) {
      sm_reduction[m_idx][warp_id] = val;
    }
  }

  __syncthreads();

  if (tid < kNumWarps) {
#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      float sum = sm_reduction[m_idx][tid];
      for (int i = 1; i < kNumWarps; i++) {
        sum += sm_reduction[m_idx][i];
      }
      sm_reduction[m_idx][0] = sum;
    }
  }

  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      out[m_idx * kNumExperts + n_idx] = sm_reduction[m_idx][0];
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename ADtype, typename BDtype, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmFloatOutput(float* output, ADtype const* mat_a, BDtype const* mat_b, bool enable_pdl, cudaStream_t stream) {
  static_assert(kHiddenDim % 1024 == 0);
  constexpr int VPT = 8;
  constexpr int kBlockSize = 128;
  dim3 grid(kNumExperts);
  dim3 block(kBlockSize);
  auto kernel = router_gemm_kernel_float_output<ADtype, BDtype, VPT, kBlockSize, kNumTokens, kNumExperts, kHiddenDim>;

  cudaLaunchConfig_t config;
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, kernel, output, mat_a, mat_b);
}

// Explicit instantiations for the expected kernel layout.
#define INSTANTIATE_RANGE(ADtype, BDtype, M, K)                                  \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 1, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 2, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 3, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 4, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 5, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 6, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 7, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 8, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 9, M, K>(            \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 10, M, K>(           \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 11, M, K>(           \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 12, M, K>(           \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 13, M, K>(           \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 14, M, K>(           \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 15, M, K>(           \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);                 \
  template void invokeRouterGemmFloatOutput<ADtype, BDtype, 16, M, K>(           \
      float*, ADtype const*, BDtype const*, bool, cudaStream_t);

INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 256, 3072)
INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 384, 3072)
INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 768, 3072)

INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 256, 6144)
INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 384, 6144)
INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 768, 6144)

INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 256, 7168)
INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 384, 7168)
INSTANTIATE_RANGE(__nv_bfloat16, __nv_bfloat16, 768, 7168)

INSTANTIATE_RANGE(__nv_bfloat16, float, 256, 3072)
INSTANTIATE_RANGE(__nv_bfloat16, float, 384, 3072)
INSTANTIATE_RANGE(__nv_bfloat16, float, 768, 3072)

INSTANTIATE_RANGE(__nv_bfloat16, float, 256, 6144)
INSTANTIATE_RANGE(__nv_bfloat16, float, 384, 6144)
INSTANTIATE_RANGE(__nv_bfloat16, float, 768, 6144)

INSTANTIATE_RANGE(__nv_bfloat16, float, 256, 7168)
INSTANTIATE_RANGE(__nv_bfloat16, float, 384, 7168)
INSTANTIATE_RANGE(__nv_bfloat16, float, 768, 7168)
