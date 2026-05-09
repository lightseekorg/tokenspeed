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

#include "flashinfer/routing_flash.cuh"
#include "flashinfer/utils.cuh"
#include "tvm_ffi_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits>

using namespace flashinfer::routing_flash;

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&device_id));
    FLASHINFER_CUDA_CALL(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  }
  return sm_count;
}

void softmax_topk_flash(TensorView input, TensorView correction_bias, TensorView topk_indices,
                        TensorView topk_weights, int64_t num_experts_real, float scaling_factor,
                        bool renormalize) {
  TVM_FFI_ICHECK_EQ(topk_weights.dtype(), dl_float32);
  const int num_experts = input.size(1);
  const int total_num_tokens = input.size(0);
  const int topk = topk_weights.size(1);
  const cudaStream_t stream = get_stream(input.device());

#define NUM_EXPERTS_SWITCH(NUM_EXPERTS_, ...)                                                 \
  [&] {                                                                                       \
    if (NUM_EXPERTS_ == 384) {                                                                \
      constexpr static int NUM_EXPERTS = 384;                                                 \
      return __VA_ARGS__();                                                                   \
    } else if (NUM_EXPERTS_ == 576) {                                                         \
      constexpr static int NUM_EXPERTS = 576;                                                 \
      return __VA_ARGS__();                                                                   \
    } else if (NUM_EXPERTS_ == 768) {                                                         \
      constexpr static int NUM_EXPERTS = 768;                                                 \
      return __VA_ARGS__();                                                                   \
    } else if (NUM_EXPERTS_ == 896) {                                                         \
      constexpr static int NUM_EXPERTS = 896;                                                 \
      return __VA_ARGS__();                                                                   \
    } else {                                                                                  \
      throw std::runtime_error("Not supported num experts: " + std::to_string(NUM_EXPERTS_)); \
    }                                                                                         \
  }()

#define IDTYPE_SWITCH(DTYPE_CODE, IDTYPE, ...)                                    \
  [&] {                                                                           \
    if (DTYPE_CODE == int64_code) {                                               \
      using IDTYPE = int64_t;                                                     \
      return __VA_ARGS__();                                                       \
    } else if (DTYPE_CODE == int32_code) {                                        \
      using IDTYPE = int32_t;                                                     \
      return __VA_ARGS__();                                                       \
    } else {                                                                      \
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported indices dtype."; \
    }                                                                             \
  }()

  NUM_EXPERTS_SWITCH(num_experts, [&] {
    TVM_FFI_ICHECK(NUM_EXPERTS > num_experts_real);
    // Single Warp
    cudaLaunchConfig_t config;
    config.gridDim = min(max(total_num_tokens, 1), 2048);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = true;
    config.numAttrs = 1;
    config.attrs = attrs;
    int64_t indices_dtype_code = encode_dlpack_dtype(topk_indices.dtype());

    IDTYPE_SWITCH(indices_dtype_code, IndexT, [&] {
      if constexpr (NUM_EXPERTS == 576) {
        static constexpr int vec_size = 4;
        TVM_FFI_ICHECK(NUM_EXPERTS % vec_size == 0);
        TVM_FFI_ICHECK(vec_size % 4 == 0);
        TVM_FFI_ICHECK(topk % 4 == 0);

        static constexpr int block_size = NUM_EXPERTS / vec_size;
        config.blockDim = block_size;
        auto kernel =
            flashinfer::routing_flash::softmax_topk_correction_bias_zero_experts_fuse_kernel<
                vec_size, block_size, IndexT>;

        cudaLaunchKernelEx(&config, kernel, static_cast<float*>(input.data_ptr()),
                          static_cast<float*>(correction_bias.data_ptr()), static_cast<IndexT*>(topk_indices.data_ptr()),
                          static_cast<float*>(topk_weights.data_ptr()), topk, total_num_tokens,
                          num_experts, static_cast<int>(num_experts_real),
                          static_cast<float>(scaling_factor), renormalize);
      } else {
        static constexpr int vec_size = (NUM_EXPERTS / 32);
        TVM_FFI_ICHECK(NUM_EXPERTS % vec_size == 0);
        TVM_FFI_ICHECK(vec_size % 4 == 0);

        static constexpr int block_size = 32;
        config.blockDim = block_size;
        auto kernel =
            flashinfer::routing_flash::softmax_topk_correction_bias_zero_experts_fuse_kernel_single_warp<
                vec_size, block_size, IndexT>;

        cudaLaunchKernelEx(&config, kernel, static_cast<float*>(input.data_ptr()),
                          static_cast<float*>(correction_bias.data_ptr()), static_cast<IndexT*>(topk_indices.data_ptr()),
                          static_cast<float*>(topk_weights.data_ptr()), topk, total_num_tokens,
                          num_experts, static_cast<int>(num_experts_real),
                          static_cast<float>(scaling_factor), renormalize);
      }
    });
    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "Failed to launch kernel: " << cudaGetErrorString(err);
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax_topk_flash, softmax_topk_flash);

namespace deepseek_v4_routing {

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  }
}

__device__ __forceinline__ float softplus_sqrt(float value) {
  constexpr float threshold = 20.0f;
  float softplus = value > threshold ? value : log1pf(expf(value));
  return sqrtf(softplus);
}

template <typename InputT, typename IndexT, int NUM_EXPERTS>
__global__ void softplus_sqrt_topk_kernel(const InputT* __restrict__ input,
                                          const float* __restrict__ correction_bias,
                                          IndexT* __restrict__ topk_indices,
                                          float* __restrict__ topk_weights,
                                          int num_rows, int topk, bool renormalize,
                                          float routed_scaling_factor) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  __shared__ float scores[NUM_EXPERTS];
  __shared__ float choice_scores[NUM_EXPERTS];
  __shared__ float reduce_scores[NUM_EXPERTS];
  __shared__ int reduce_indices[NUM_EXPERTS];
  __shared__ float selected_sum;

  if (tid < NUM_EXPERTS) {
    const float score =
        softplus_sqrt(to_float(input[row * NUM_EXPERTS + tid]));
    scores[tid] = score;
    choice_scores[tid] =
        correction_bias == nullptr ? score : score + correction_bias[tid];
  }
  if (tid == 0) {
    selected_sum = 0.0f;
  }
  __syncthreads();

  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    if (tid < NUM_EXPERTS) {
      reduce_scores[tid] = choice_scores[tid];
      reduce_indices[tid] = tid;
    }
    __syncthreads();

    for (int stride = NUM_EXPERTS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_score = reduce_scores[tid + stride];
        const int other_index = reduce_indices[tid + stride];
        const bool take_other =
            other_score > reduce_scores[tid] ||
            (other_score == reduce_scores[tid] &&
             other_index < reduce_indices[tid]);
        if (take_other) {
          reduce_scores[tid] = other_score;
          reduce_indices[tid] = other_index;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int expert = reduce_indices[0];
      const float weight = scores[expert];
      const int out_idx = row * topk + k_idx;
      topk_indices[out_idx] = static_cast<IndexT>(expert);
      topk_weights[out_idx] = weight;
      selected_sum += weight;
      choice_scores[expert] = -std::numeric_limits<float>::infinity();
    }
    __syncthreads();
  }

  if (tid < topk) {
    float scale = routed_scaling_factor;
    if (renormalize) {
      scale /= selected_sum > 0.0f ? selected_sum : 1.0f;
    }
    topk_weights[row * topk + tid] *= scale;
  }
}

template <typename InputT, typename IndexT, typename TokenT, int NUM_EXPERTS>
__global__ void hash_softplus_sqrt_topk_kernel(
    const InputT* __restrict__ input, const TokenT* __restrict__ input_ids,
    const IndexT* __restrict__ hash_indices_table, IndexT* __restrict__ topk_indices,
    float* __restrict__ topk_weights, int num_rows, int topk, bool renormalize,
    float routed_scaling_factor) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  __shared__ float partial[32];
  float weight = 0.0f;

  if (tid < topk) {
    const TokenT token_id = input_ids[row];
    const IndexT expert =
        hash_indices_table[static_cast<int64_t>(token_id) * topk + tid];
    weight = softplus_sqrt(
        to_float(input[row * NUM_EXPERTS + static_cast<int>(expert)]));
    const int out_idx = row * topk + tid;
    topk_indices[out_idx] = expert;
    topk_weights[out_idx] = weight;
  }
  partial[tid] = tid < topk ? weight : 0.0f;
  __syncthreads();

  for (int stride = 16; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid < topk) {
    float scale = routed_scaling_factor;
    if (renormalize) {
      scale /= partial[0] > 0.0f ? partial[0] : 1.0f;
    }
    topk_weights[row * topk + tid] *= scale;
  }
}

}  // namespace deepseek_v4_routing

#define DSV4_DISPATCH_INPUT(DTYPE_CODE, InputT, ...)                                      \
  [&] {                                                                                   \
    if (DTYPE_CODE == float32_code) {                                                     \
      using InputT = float;                                                               \
      return __VA_ARGS__();                                                               \
    } else if (DTYPE_CODE == bfloat16_code) {                                             \
      using InputT = __nv_bfloat16;                                                       \
      return __VA_ARGS__();                                                               \
    } else if (DTYPE_CODE == float16_code) {                                              \
      using InputT = __half;                                                              \
      return __VA_ARGS__();                                                               \
    } else {                                                                              \
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported router logits dtype.";   \
    }                                                                                     \
  }()

#define DSV4_DISPATCH_INDEX(DTYPE_CODE, IndexT, ...)                                      \
  [&] {                                                                                   \
    if (DTYPE_CODE == int32_code) {                                                       \
      using IndexT = int32_t;                                                             \
      return __VA_ARGS__();                                                               \
    } else if (DTYPE_CODE == int64_code) {                                                \
      using IndexT = int64_t;                                                             \
      return __VA_ARGS__();                                                               \
    } else {                                                                              \
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported top-k index dtype.";     \
    }                                                                                     \
  }()

void softplus_sqrt_topk_flash(TensorView input, TensorView correction_bias,
                              TensorView topk_indices, TensorView topk_weights,
                              bool renormalize, float routed_scaling_factor) {
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(correction_bias.ndim(), 1);
  TVM_FFI_ICHECK_EQ(topk_indices.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.dtype(), dl_float32);

  const int num_rows = input.size(0);
  const int num_experts = input.size(1);
  const int topk = topk_weights.size(1);
  TVM_FFI_ICHECK_EQ(num_experts, 256)
      << "DeepSeek V4 fused router currently supports 256 experts only";
  TVM_FFI_ICHECK_EQ(correction_bias.size(0), num_experts);
  TVM_FFI_ICHECK_EQ(topk_indices.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(topk_indices.size(1), topk);
  TVM_FFI_ICHECK(topk > 0 && topk <= 32);

  const cudaStream_t stream = get_stream(input.device());
  const int64_t input_dtype_code = encode_dlpack_dtype(input.dtype());
  const int64_t index_dtype_code = encode_dlpack_dtype(topk_indices.dtype());

  DSV4_DISPATCH_INPUT(input_dtype_code, InputT, [&] {
    DSV4_DISPATCH_INDEX(index_dtype_code, IndexT, [&] {
      deepseek_v4_routing::softplus_sqrt_topk_kernel<InputT, IndexT, 256>
          <<<num_rows, 256, 0, stream>>>(
              static_cast<const InputT*>(input.data_ptr()),
              static_cast<const float*>(correction_bias.data_ptr()),
              static_cast<IndexT*>(topk_indices.data_ptr()),
              static_cast<float*>(topk_weights.data_ptr()), num_rows, topk,
              renormalize, routed_scaling_factor);
      return true;
    });
    return true;
  });

  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "Failed to launch DeepSeek V4 softplus-sqrt top-k kernel: "
      << cudaGetErrorString(err);
}

void hash_softplus_sqrt_topk_flash(TensorView input, TensorView input_ids,
                                   TensorView hash_indices_table,
                                   TensorView topk_indices,
                                   TensorView topk_weights, bool renormalize,
                                   float routed_scaling_factor) {
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input_ids.ndim(), 1);
  TVM_FFI_ICHECK_EQ(hash_indices_table.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_indices.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.dtype(), dl_float32);

  const int num_rows = input.size(0);
  const int num_experts = input.size(1);
  const int topk = topk_weights.size(1);
  TVM_FFI_ICHECK_EQ(num_experts, 256)
      << "DeepSeek V4 fused hash router currently supports 256 experts only";
  TVM_FFI_ICHECK_EQ(input_ids.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(hash_indices_table.size(1), topk);
  TVM_FFI_ICHECK_EQ(topk_indices.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(topk_indices.size(1), topk);
  TVM_FFI_ICHECK(topk > 0 && topk <= 32);

  const cudaStream_t stream = get_stream(input.device());
  const int64_t input_dtype_code = encode_dlpack_dtype(input.dtype());
  const int64_t index_dtype_code = encode_dlpack_dtype(topk_indices.dtype());
  const int64_t token_dtype_code = encode_dlpack_dtype(input_ids.dtype());

  DSV4_DISPATCH_INPUT(input_dtype_code, InputT, [&] {
    DSV4_DISPATCH_INDEX(index_dtype_code, IndexT, [&] {
      DSV4_DISPATCH_INDEX(token_dtype_code, TokenT, [&] {
        deepseek_v4_routing::
            hash_softplus_sqrt_topk_kernel<InputT, IndexT, TokenT, 256>
            <<<num_rows, 32, 0, stream>>>(
                static_cast<const InputT*>(input.data_ptr()),
                static_cast<const TokenT*>(input_ids.data_ptr()),
                static_cast<const IndexT*>(hash_indices_table.data_ptr()),
                static_cast<IndexT*>(topk_indices.data_ptr()),
                static_cast<float*>(topk_weights.data_ptr()), num_rows, topk,
                renormalize, routed_scaling_factor);
        return true;
      });
      return true;
    });
    return true;
  });

  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "Failed to launch DeepSeek V4 hash softplus-sqrt top-k kernel: "
      << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softplus_sqrt_topk_flash, softplus_sqrt_topk_flash);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hash_softplus_sqrt_topk_flash,
                              hash_softplus_sqrt_topk_flash);
