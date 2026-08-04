/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLASHINFER_ACTIVATION_CUH_
#define FLASHINFER_ACTIVATION_CUH_

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "math.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace activation {

constexpr static float FP8_E4M3_MAX = 448.0f;

template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }
    out_vec.cast_store(out + token_idx * d + idx * vec_size);
  }

  const int64_t remaining_offset = d - d % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    float x = input[offset + remaining_offset + idx],
          y = input[offset + remaining_offset + d + idx];
    out[token_idx * d + remaining_offset + idx] = Activation(x) * y;
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t VEC_SIZE, typename T>
__device__ __forceinline__ vec_t<__nv_fp8_e4m3, VEC_SIZE> block_quant_fp8(
    vec_t<T, VEC_SIZE> input, float* scale_out, int32_t scale_stride_expert,
    int32_t scale_stride_col, int32_t expert_id, int32_t token_id, int32_t access_id_in_token) {
  namespace cg = cooperative_groups;
  float _absmax = 1e-10f;
  vec_t<__nv_fp8_e4m3, VEC_SIZE> quanted_out;
  auto tile_16 = cg::tiled_partition<16>(cg::this_thread_block());
#pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    float val = static_cast<float>(input[i]);
    _absmax = std::max(_absmax, fabs(val));
  }
  _absmax = cg::reduce(tile_16, _absmax, [](float a, float b) { return a > b ? a : b; });

  float scale;
  asm("div.full.f32 %0, %1, %2;" : "=f"(scale) : "f"(_absmax), "f"(FP8_E4M3_MAX));

  // directly write scale to scale_out
  if (tile_16.thread_rank() == 0) {
    int32_t col_idx = int32_t(access_id_in_token / 16);
    // For Column-Major Tensor.
    // scale_stride_expert = scale.stride(0) when grouped gemm, 0 when dense gemm
    // scale_stride_col = scale.stride(-1)
    float* scale_ptr =
        scale_out + expert_id * scale_stride_expert + token_id + col_idx * scale_stride_col;
    *(scale_ptr) = scale;
  }
  tile_16.sync();

  float reverse_scale = 1.0 / scale;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    float x = static_cast<float>(input[i]) * reverse_scale;
    float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
    reinterpret_cast<__nv_fp8_e4m3*>(&quanted_out)[i] = static_cast<__nv_fp8_e4m3>(r);
  }
  return quanted_out;
}

template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_post_block_quant_128_kernel(__nv_fp8_e4m3* __restrict__ out,
                                                        float* __restrict__ scale_out,
                                                        T* __restrict__ input,
                                                        const int scale_stride, const int d) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  static_assert(sizeof(T) == 2);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }
    vec_t<__nv_fp8_e4m3, vec_size> quanted_vec =
        block_quant_fp8<vec_size>(out_vec, scale_out, 0, scale_stride, 0, token_idx, idx);
    quanted_vec.store(out + token_idx * d + idx * vec_size);
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_post_block_quant_128_kernel_ep(
    __nv_fp8_e4m3* __restrict__ out, float* __restrict__ scale_out, T* __restrict__ input,
    int32_t* __restrict__ num_tokens, int gate_up_stride_0, int out_stride_0,
    int scale_stride_expert, int scale_stride_col, const int d, const int num_experts) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // Used in low_latency mode of deep_ep + deepgemm masked grouped gemm
  constexpr uint32_t vec_size = 16 / sizeof(T);
  static_assert(sizeof(T) == 2);
  const int32_t expert_idx = blockIdx.x;
  const int32_t access_id_in_token = threadIdx.x;
  const int32_t access_stride_in_token = blockDim.x;
  // relates to expected_m
  const int32_t token_idx = blockIdx.y;
  int32_t num_tokens_cur_expert = num_tokens[expert_idx];

  for (int32_t t_idx = token_idx; t_idx < num_tokens_cur_expert; t_idx += blockDim.y) {
    const T* gate_up_ptr = input + expert_idx * gate_up_stride_0 + t_idx * 2 * d;
    for (uint32_t idx = access_id_in_token; idx < d / vec_size; idx += access_stride_in_token) {
      vec_t<float, vec_size> gate_vec, up_vec, out_vec;
      gate_vec.cast_load(gate_up_ptr + idx * vec_size);
      up_vec.cast_load(gate_up_ptr + d + idx * vec_size);
#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        out_vec[j] = Activation(gate_vec[j]) * up_vec[j];
      }
      vec_t<__nv_fp8_e4m3, vec_size> quanted_vec = block_quant_fp8<vec_size>(
          out_vec, scale_out, scale_stride_expert, scale_stride_col, expert_idx, t_idx, idx);
      quanted_vec.store(out + expert_idx * out_stride_0 + t_idx * d + idx * vec_size);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace activation
}  // namespace flashinfer

#endif  // FLASHINFER_ACTIVATION_CUH_
