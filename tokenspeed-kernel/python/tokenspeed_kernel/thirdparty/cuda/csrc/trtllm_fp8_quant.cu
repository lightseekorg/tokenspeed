/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// FP8 E4M3 quantizers adapted from TensorRT-LLM (commit fca831e):
// - scale_1x128_kernel / fp8_1x128_cs from
//   cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh,
//   behind torch.ops.trtllm.fp8_quantize_1x128.
// - dynamicQuantizeMatrixPerToken / invokeComputeScalesAndQuantizeMatrix (PER_TOKEN) from
//   cpp/tensorrt_llm/common/cudaFp8Utils.cu, behind torch.ops.tensorrt_llm.quantize_e4m3_activation.
// The math is unchanged. TensorRT-LLM builds these without --use_fast_math, so
// the divisions use the correctly rounded __fdiv_rn/__frcp_rn intrinsics; the
// results stay identical when this file is compiled with fast math.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "tvm_ffi_utils.h"

namespace tokenspeed_trtllm_fp8 {

constexpr float FP8_E4M3_MAX = 448.0f;
constexpr int CTA_SIZE = 256;
#define TRTLLM_FP8_FINAL_MASK 0xffffffff

// ---------------------------------------------------------------------------
// 1x128 block quantization (fp8_blockscale_gemm_kernel.cuh)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __host__ constexpr T div_up(T a, int b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ inline T cuda_max(T val1, T val2) {
  return (val1 > val2) ? val1 : val2;
}

template <typename T>
__forceinline__ __device__ T find_max_elem_in_warp(T value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = T(std::max(float(value), __shfl_down_sync(0xFFFFFFFF, float(value), offset)));
  }
  value = T(__shfl_sync(0xffffffff, float(value), 0));
  return value;
}

template <typename InputType, typename OutputType, typename ScaleType = float,
          bool USE_UE8M0 = false>
__global__ void scale_1x128_kernel(OutputType* output, ScaleType* scales,
                                   InputType const* const input, int dim_x, int dim_y) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
  size_t scales_along_dim_x = div_up(dim_x, 128);
  size_t scales_along_dim_y = div_up(dim_y, 1);
  size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;
  using Input2Type = typename std::conditional<std::is_same<InputType, half>::value, half2,
                                               __nv_bfloat162>::type;
  for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
       warp_idx < scales_along_dim_x * scales_along_dim_y;
       warp_idx += gridDim.x * blockDim.x / 32) {
    int scales_idx_y = warp_idx / scales_along_dim_x;
    int scales_idx_x = warp_idx % scales_along_dim_x;

    InputType const* input_line = input + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
    InputType input_amax = InputType(0);
    // Each thread reads 2 elements from input_line
    int lane_id = threadIdx.x % 32 * 2;

    Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
#pragma unroll
    for (int i = 0; i < 2; i++) {
      if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
        break;
      } else {
        input_frag2[i] = *((Input2Type*)(input_line) + lane_id / 2);
      }
      input_line += 64;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
        break;
      } else {
        input_amax = InputType(
            __hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
      }
    }

    InputType amax = find_max_elem_in_warp(input_amax);
    amax = cuda_max(amax, InputType(1e-10f));
    ScaleType quant_scale = __fdiv_rn(448.f, ScaleType(amax));
    ScaleType dequant_scale;

    if constexpr (USE_UE8M0) {
      // Round dequant scale to UE8M0 (power of 2)
      ScaleType dequant_scale_raw = __frcp_rn(quant_scale);
      __nv_fp8_e8m0 ue8m0_scale;
      ue8m0_scale.__x =
          __nv_cvt_float_to_e8m0(float(dequant_scale_raw), __NV_SATFINITE, cudaRoundPosInf);
      // Cast back to float automatically decodes E8M0 format
      dequant_scale = ScaleType(static_cast<float>(ue8m0_scale));
      // Recompute quant scale from rounded dequant scale for consistency
      quant_scale = dequant_scale != ScaleType(0.f) ? __frcp_rn(dequant_scale) : 1.f;
    } else {
      dequant_scale = __frcp_rn(quant_scale);
    }

    if (lane_id == 0) {
      scales[(size_t)scales_idx_x * stride_scale_dim_y + scales_idx_y] = dequant_scale;
    }

    OutputType* output_line = output + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
        break;
      } else {
        ScaleType value_1 = ScaleType(input_frag2[i].x) * quant_scale;
        ScaleType value_2 = ScaleType(input_frag2[i].y) * quant_scale;
        output_line[lane_id] = OutputType(value_1);
        output_line[lane_id + 1] = OutputType(value_2);
      }
      output_line += 64;
    }
  }
#endif
}

static int multi_processor_count(int device) {
  // TensorRT-LLM caches the first device's count; the grid only sizes a
  // grid-stride loop, so a per-device cache gives the same results.
  static int counts[64] = {0};
  if (device >= 0 && device < 64 && counts[device] > 0) {
    return counts[device];
  }
  int count = 0;
  cudaError_t status = cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaDeviceGetAttribute(MultiProcessorCount) failed: " << cudaGetErrorString(status);
  if (device >= 0 && device < 64) {
    counts[device] = count;
  }
  return count;
}

void fp8_1x128_cs(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
                  int shape_y, int device, cudaStream_t stream, bool use_ue8m0) {
  int const num_sms = multi_processor_count(device);
  if (use_ue8m0) {
    scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, true>
        <<<num_sms * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
  } else {
    scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, false>
        <<<num_sms * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
  }
}

// ---------------------------------------------------------------------------
// Per-token dynamic quantization (cudaFp8Utils.cu, QuantizeMode::PER_TOKEN)
// ---------------------------------------------------------------------------

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(TRTLLM_FP8_FINAL_MASK, val, mask, 32));
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx

  val = warpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

template <typename T>
__inline__ __device__ T blockAllReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx

  val = warpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

// scale<true>(a, b) in TensorRT-LLM: a / b.
__inline__ __device__ float quantize_value(float a, float b) { return __fdiv_rn(a, b); }

template <typename T_OUT, typename T_S, typename T_IN>
__global__ void dynamicQuantizeMatrixPerToken(T_OUT* output, T_S* quant_ptr, T_IN const* input,
                                              int64_t numel, int64_t lda) {
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  T_IN* shmem = reinterpret_cast<T_IN*>(_shmem);
  constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);
  auto const nrows = numel / lda;
  for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x) {
    float max = 0.f;
    for (int64_t i = threadIdx.x; i < lda; i += blockDim.x) {
      auto const in = input[row * lda + i];
      shmem[i] = in;
      auto val = fabs(static_cast<float>(in));
      max = max > val ? max : val;
    }
    max = blockAllReduceMax<float>(max);  // __syncthreads() called so we can read shmem
    auto const s = (T_S)std::max(__fdiv_rn(max, FP8_E4M3_MAX), min_scaling_factor);
    for (int64_t i = threadIdx.x; i < lda; i += blockDim.x) {
      // true means we are quantizing
      output[row * lda + i] =
          (T_OUT)quantize_value(static_cast<float>(shmem[i]), static_cast<float>(s));
    }
    if (threadIdx.x == 0) {
      quant_ptr[row] = s;
    }
  }
}

// Fallback when a row does not fit in shared memory: computeFP8QuantizeScale
// (PER_TOKEN) followed by scaleMatrix (PER_TOKEN, quantize).
template <typename T_S, typename T_W>
__global__ void computeFP8QuantizeScalePerToken(T_S* quant_ptr, T_W const* weights,
                                                int64_t const size, int64_t const n) {
  constexpr float min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);
  auto const nrows = size / n;
  for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x) {
    float max = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
      auto val = fabs(static_cast<float>(weights[row * n + i]));
      max = max > val ? max : val;
    }
    max = blockReduceMax<float>(max);
    if (threadIdx.x == 0) {
      auto const scale = (T_S)std::max(__fdiv_rn(max, FP8_E4M3_MAX), min_scaling_factor);
      quant_ptr[row] = scale;
    }
  }
}

template <typename T_OUT, typename T_S, typename T_IN>
__global__ void scaleMatrixPerToken(T_OUT* output, T_S const* input_scale, T_IN const* input,
                                    int64_t numel, int64_t lda) {
  for (int64_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += blockDim.x * gridDim.x) {
    output[i] = T_OUT(
        quantize_value(static_cast<float>(input[i]), static_cast<float>(input_scale[i / lda])));
  }
}

template <typename T_OUT, typename T_S, typename T_IN>
void invokeComputeScalesAndQuantizeMatrixPerToken(T_OUT* output, T_S* quant_ptr, T_IN const* input,
                                                  int64_t const numel, int64_t const lda,
                                                  cudaStream_t stream) {
  dim3 grid(numel / lda);
  bool use_shmem = true;
  auto const shmem_size = lda * sizeof(T_IN);
  if (shmem_size >= (48 << 10)) {
    cudaError_t ret =
        cudaFuncSetAttribute(dynamicQuantizeMatrixPerToken<T_OUT, T_S, T_IN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    use_shmem = ret == cudaSuccess;
  }
  if (use_shmem) {
    // ensure the threadblock is as large as possible to increase occupancy
    dim3 block(std::min((lda + 31) / 32 * 32, static_cast<int64_t>(1024)));
    dynamicQuantizeMatrixPerToken<<<grid, block, shmem_size, stream>>>(output, quant_ptr, input,
                                                                       numel, lda);
  } else {
    (void)cudaGetLastError();  // clear the failed cudaFuncSetAttribute
    dim3 block(CTA_SIZE);
    computeFP8QuantizeScalePerToken<<<grid, block, 0, stream>>>(quant_ptr, input, numel, lda);
    scaleMatrixPerToken<<<dim3(1024), dim3(CTA_SIZE), 0, stream>>>(output, quant_ptr, input, numel,
                                                                   lda);
  }
}

}  // namespace tokenspeed_trtllm_fp8

using namespace tokenspeed_trtllm_fp8;

// output: FP8 E4M3 [m_padded, n] with m_padded = ceil(m / 4) * 4 (rows >= m untouched).
// scales: float32 flat buffer of at least ceil(n / 128) * m_padded elements, MN-major
//         with a row stride of m_padded.
// input:  contiguous BF16 [m, n], n % 16 == 0.
void fp8_quantize_1x128(TensorView output, TensorView scales, TensorView input, bool use_ue8m0) {
  CHECK_INPUT_AND_TYPE(input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(output, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(scales, dl_float32);
  CHECK_DIM(2, input);
  CHECK_DEVICE(input, output);
  CHECK_DEVICE(input, scales);
  int64_t const m = input.size(0);
  int64_t const n = input.size(1);
  TVM_FFI_ICHECK(m <= INT32_MAX) << "M must be within int32";
  TVM_FFI_ICHECK(n <= INT32_MAX) << "N must be within int32";
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "input.size(1) must be a multiple of 16, but got " << n;
  int64_t const m_padded = (m + 3) / 4 * 4;
  TVM_FFI_ICHECK(output.ndim() == 2 && output.size(0) == m_padded && output.size(1) == n)
      << "output must be [ceil(m / 4) * 4, n]";
  TVM_FFI_ICHECK_GE(scales.numel(), (n + 127) / 128 * m_padded) << "scales buffer too small";
  if (m == 0) {
    return;
  }
  auto device = input.device();
  fp8_1x128_cs(static_cast<__nv_fp8_e4m3*>(output.data_ptr()), static_cast<float*>(scales.data_ptr()),
               static_cast<__nv_bfloat16 const*>(input.data_ptr()), static_cast<int>(n),
               static_cast<int>(m), device.device_id, get_stream(device), use_ue8m0);
}

// output: FP8 E4M3 with input's shape; scales: input dtype, [..., 1].
// input:  contiguous FP32/FP16/BF16 with at least two dimensions.
void quantize_e4m3_per_token(TensorView output, TensorView scales, TensorView input) {
  CHECK_INPUT(input);
  CHECK_INPUT_AND_TYPE(output, dl_float8_e4m3fn);
  CHECK_INPUT(scales);
  CHECK_DEVICE(input, output);
  CHECK_DEVICE(input, scales);
  TVM_FFI_ICHECK_GE(input.ndim(), 2) << "input must have at least two dimensions";
  TVM_FFI_ICHECK_EQ(output.numel(), input.numel());
  TVM_FFI_ICHECK_EQ(scales.dtype(), input.dtype()) << "scales must use the input dtype";
  int64_t const numel = input.numel();
  int64_t const lda = input.size(-1);
  TVM_FFI_ICHECK(numel != 0) << "input should not be empty tensor";
  TVM_FFI_ICHECK_EQ(scales.numel(), numel / lda);
  auto* out = static_cast<__nv_fp8_e4m3*>(output.data_ptr());
  cudaStream_t stream = get_stream(input.device());
  int64_t const code = encode_dlpack_dtype(input.dtype());
  if (code == float32_code) {
    invokeComputeScalesAndQuantizeMatrixPerToken(out, static_cast<float*>(scales.data_ptr()),
                                                 static_cast<float const*>(input.data_ptr()), numel,
                                                 lda, stream);
  } else if (code == float16_code) {
    invokeComputeScalesAndQuantizeMatrixPerToken(out, static_cast<half*>(scales.data_ptr()),
                                                 static_cast<half const*>(input.data_ptr()), numel,
                                                 lda, stream);
  } else if (code == bfloat16_code) {
    invokeComputeScalesAndQuantizeMatrixPerToken(
        out, static_cast<__nv_bfloat16*>(scales.data_ptr()),
        static_cast<__nv_bfloat16 const*>(input.data_ptr()), numel, lda, stream);
  } else {
    TVM_FFI_ICHECK(false) << "input must be BF16/FP16/FP32";
  }
}
