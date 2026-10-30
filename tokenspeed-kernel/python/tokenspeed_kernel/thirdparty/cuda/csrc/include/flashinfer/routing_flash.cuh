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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <cuda/functional>

#include "math.h"
#include "vec_dtypes.cuh"

namespace flashinfer {
namespace routing_flash {

namespace cg = cooperative_groups;

template <int VEC_SIZE, int BLOCK_SIZE>
__device__ void Softmax(float* scores_vec) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage reduce_stroge;
  __shared__ float shared_val;
  float max_score = float(-INFINITY);
  float sum_score = 0.f;

#pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    max_score = scores_vec[i] > max_score ? scores_vec[i] : max_score;
  }
  __syncthreads();
  max_score = BlockReduce(reduce_stroge).Reduce(max_score, ::cuda::maximum<float>{});
  if (threadIdx.x == 0) {
    shared_val = max_score;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    float si = static_cast<float>(scores_vec[i]);
    // float e = expf(si - max_score);
    float e = expf(si - shared_val);
    scores_vec[i] = static_cast<float>(e);
    sum_score += e;
  }
  sum_score = BlockReduce(reduce_stroge).Sum(sum_score);
  if (threadIdx.x == 0) {
    shared_val = sum_score;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    float si = static_cast<float>(scores_vec[i]) / shared_val;
    scores_vec[i] = static_cast<float>(si);
  }
}

template <int VEC_SIZE, int BLOCK_SIZE, typename T>
__global__ void softmax_topk_correction_bias_zero_experts_fuse_kernel_single_warp(
    float* input, float* correction_bias, T* topk_indices, float* topk_weights, int topk,
    int total_num_tokens, int num_experts, int num_experts_real, float scale, bool renorm) {
  static_assert(VEC_SIZE % 4 == 0);
  static_assert(BLOCK_SIZE == 32);
  int64_t token_idx = blockIdx.x;
  int64_t token_access_stride = gridDim.x;
  int64_t thread_idx = threadIdx.x;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  int row_access_offset = thread_idx * VEC_SIZE;
  vec_t<float, VEC_SIZE> bias_vec;
  bias_vec.load(correction_bias + row_access_offset);

  for (int t_idx = token_idx; t_idx < total_num_tokens; t_idx += token_access_stride) {
    vec_t<float, VEC_SIZE> scores_vec;
    int offset = t_idx * num_experts;
    const float* logits_ptr = input + offset;
    scores_vec.load(logits_ptr + row_access_offset);

    // 1. Softmax
    float thread_max_score = float(-INFINITY);
    float row_sum = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      thread_max_score = scores_vec[i] > thread_max_score ? scores_vec[i] : thread_max_score;
    }

    for (int stride = 16; stride > 0; stride >>= 1) {
      thread_max_score =
          max(thread_max_score, __shfl_xor_sync(0xffffffff, thread_max_score, stride));
    }

#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      scores_vec[i] = expf(scores_vec[i] - thread_max_score);
      row_sum += scores_vec[i];
    }

    for (int stride = 16; stride > 0; stride >>= 1) {
      row_sum += __shfl_xor_sync(0xffffffff, row_sum, stride);
    }
    const float reciprocal_row_sum = 1.f / row_sum;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      // Softmax End
      scores_vec[i] = scores_vec[i] * reciprocal_row_sum;
      // apply correction bias
      scores_vec[i] += bias_vec[i];
    }

    // 2. TOPK
    for (int k = 0; k < topk; k++) {
      // Each thread do local max first
      float max_score = scores_vec[0];
      int expert_id = thread_idx * VEC_SIZE;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; i++) {
        if (scores_vec[i] > max_score) {
          max_score = scores_vec[i];
          expert_id = thread_idx * VEC_SIZE + i;
        }
      }

      // Shffule Warp MAX score and related expert id
      for (int stride = 16; stride > 0; stride >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, max_score, stride);
        int other_expert = __shfl_xor_sync(0xffffffff, expert_id, stride);
        if (other_max > max_score || (other_max == max_score && other_expert < expert_id)) {
          max_score = other_max;
          expert_id = other_expert;
        }
      }

      // Now, Every thread has the same k-th max_score and related expert id
      if (k + 1 < topk) {
        // means there's another iter, block out winner score
        if (thread_idx == expert_id / VEC_SIZE) {
          scores_vec[expert_id % VEC_SIZE] = -100000.0;
        }
      }

      // Write back to global mem
      if (thread_idx == expert_id / VEC_SIZE){
        topk_weights[t_idx * topk + k] = (max_score - bias_vec[expert_id % VEC_SIZE]) * scale;
        topk_indices[t_idx * topk + k] = expert_id >= num_experts_real ? -1 : static_cast<T>(expert_id);
      }
    }
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <int VEC_SIZE, int BLOCK_SIZE, typename T>
__global__ void softmax_topk_correction_bias_zero_experts_fuse_kernel(
    float* input, float* correction_bias, T* topk_indices, float* topk_weights, int topk,
    int total_num_tokens, int num_experts, int num_experts_real, float scale, bool renorm) {
  int64_t token_idx = blockIdx.x;
  int64_t token_access_stride = gridDim.x;
  int64_t thread_idx = threadIdx.x;

  __shared__ float s_correction_bias[BLOCK_SIZE * VEC_SIZE];

  const float* bias_ptr = correction_bias + thread_idx * VEC_SIZE;
  vec_t<float, VEC_SIZE> bias_vec;
  bias_vec.load(bias_ptr);
#pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    s_correction_bias[thread_idx * VEC_SIZE + i] = bias_vec[i];
  }
  __syncthreads();

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (int t_idx = token_idx; t_idx < total_num_tokens; t_idx += token_access_stride) {
    vec_t<float, VEC_SIZE> scores_vec;
    int expert_indices[VEC_SIZE];
    int offset = t_idx * num_experts;
    const float* logits_ptr = input + offset;
    scores_vec.load(logits_ptr + thread_idx * VEC_SIZE);
    Softmax<VEC_SIZE, BLOCK_SIZE>(scores_vec.ptr());
    __syncthreads();

#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      expert_indices[i] = thread_idx * VEC_SIZE + i;
      scores_vec[i] = scores_vec[i] + bias_vec[i];
    }

    __syncthreads();

    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, VEC_SIZE, int>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    BlockRadixSort(temp_storage)
        .SortDescending(reinterpret_cast<float (&)[4]>(scores_vec.data[0]), expert_indices);
    __syncthreads();

    for (int i = 0; i < VEC_SIZE; i++) {
      int expert_id = expert_indices[i];
      scores_vec[i] = scores_vec[i] - s_correction_bias[expert_id];
    }

    if (renorm) {
      float sum_score = 0.f;
      for (int i = 0; i < VEC_SIZE; i++) {
        sum_score += scores_vec[i];
      }
      auto warp = cg::tiled_partition<32>(cg::this_thread_block());
      int warp_idx = warp.meta_group_rank();
      float thread_data = (thread_idx < (topk / VEC_SIZE)) ? sum_score : 0;
      sum_score = cg::reduce(warp, thread_data, cg::plus<float>());
      if (warp_idx == 0) {
        for (int i = 0; i < VEC_SIZE; i++) {
          scores_vec[i] /= (sum_score + 1e-10);
        }
      }
      __syncthreads();
    }
    if (thread_idx * VEC_SIZE < topk) {
      // handle zero expert indices and write back
      for (int i = 0; i < VEC_SIZE; i++) {
        expert_indices[i] = (expert_indices[i] >= num_experts_real) ? -1 : expert_indices[i];
        *(topk_indices + t_idx * topk + thread_idx * VEC_SIZE + i) =
            static_cast<T>(expert_indices[i]);
      }
#pragma unroll
      for (int i = 0; i < VEC_SIZE; i++) {
        scores_vec[i] = scores_vec[i] * scale;
      }
      scores_vec.store(topk_weights + t_idx * topk + thread_idx * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
}  // namespace routing_flash
}  // namespace flashinfer
