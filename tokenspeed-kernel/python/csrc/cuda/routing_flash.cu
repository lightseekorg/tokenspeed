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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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

// ---------------------------------------------------------------------------
// Warp-level top-k via packed value+index reduction.
// Ported from TRT-LLM moeTopKFuncs.cuh (Apache-2.0).
// ---------------------------------------------------------------------------
namespace warp_topk {
namespace cg = cooperative_groups;
static constexpr int kWARP_SIZE = 32;

__device__ __forceinline__ uint64_t pack_score_idx(float val, int32_t idx) {
  uint32_t v;
  memcpy(&v, &val, 4);
  // IEEE-754 float: flip sign bit so unsigned compare preserves order.
  v = (v & 0x80000000u) ? ~v : (v ^ 0x80000000u);
  return (static_cast<uint64_t>(v) << 32) | static_cast<uint64_t>(0xFFFF - (idx & 0xFFFF));
}

__device__ __forceinline__ void unpack_score_idx(uint64_t packed, float& val, int32_t& idx) {
  idx = 0xFFFF - static_cast<int32_t>(packed & 0xFFFF);
  uint32_t v = static_cast<uint32_t>(packed >> 32);
  v = (v & 0x80000000u) ? (v ^ 0x80000000u) : ~v;
  memcpy(&val, &v, 4);
}

__device__ __forceinline__ uint64_t warp_max_u64(
    cg::thread_block_tile<kWARP_SIZE> const& warp, uint64_t val) {
  return cg::reduce(warp, val, cg::greater<uint64_t>{});
}

template <int K, int N>
__device__ void reduce_topk(
    cg::thread_block_tile<kWARP_SIZE> const& warp,
    float (&out_vals)[K], int32_t (&out_idx)[K],
    float (&scores)[N], int32_t (&indices)[N],
    int actual_k) {
  // Sort candidates per thread (simple insertion sort for small N).
  uint64_t packed[N];
#pragma unroll
  for (int i = 0; i < N; ++i)
    packed[i] = pack_score_idx(scores[i], indices[i]);
  // Odd-even transposition sort.
#pragma unroll
  for (int pass = 0; pass < N; ++pass) {
#pragma unroll
    for (int i = 0; i < N - 1; i += 2)
      if (packed[i] < packed[i + 1]) { auto t = packed[i]; packed[i] = packed[i + 1]; packed[i + 1] = t; }
#pragma unroll
    for (int i = 1; i < N - 1; i += 2)
      if (packed[i] < packed[i + 1]) { auto t = packed[i]; packed[i] = packed[i + 1]; packed[i + 1] = t; }
  }

  uint64_t prev_max = 0;
  for (int k = 0; k < actual_k; ++k) {
    bool dup = k > 0 && prev_max == packed[0];
#pragma unroll
    for (int i = 0; i < N; ++i)
      packed[i] = dup && i == N - 1 ? 0ULL : dup ? packed[i + 1] : packed[i];
    prev_max = warp_max_u64(warp, packed[0]);
    unpack_score_idx(prev_max, out_vals[k], out_idx[k]);
  }
}

} // namespace warp_topk

// ---------------------------------------------------------------------------
// gate_forward_kernel: warp-level fused softplus-sqrt + top-k.
// Ported from TRT-LLM customMoeRoutingKernels.cu (Apache-2.0).
// Supports any nExperts (256, 384, etc.) via template parameter.
// ---------------------------------------------------------------------------
template <int nExperts, int topK, bool hash, typename TokenIdT = int>
__global__ void gate_forward_kernel(
    const float* __restrict__ scores_in,
    const float* __restrict__ bias,
    const TokenIdT* __restrict__ input_ids,
    const int* __restrict__ tid2eid,
    float* __restrict__ out_weights,
    int* __restrict__ out_indices,
    int batch_size, float route_scale) {
  namespace cg = cooperative_groups;
  constexpr int kExpertsPerThread = nExperts / warp_topk::kWARP_SIZE;
  constexpr int kWarpsPerBlock = 4;

  __shared__ float smem_scores[kWarpsPerBlock][nExperts];

  int const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / warp_topk::kWARP_SIZE;
  int const local_warp_id =
      (threadIdx.x / warp_topk::kWARP_SIZE) % kWarpsPerBlock;
  int const lane_id = threadIdx.x % warp_topk::kWARP_SIZE;

  if (global_warp_id >= batch_size) return;

  auto warp = cg::tiled_partition<warp_topk::kWARP_SIZE>(cg::this_thread_block());

  float* my_smem = smem_scores[local_warp_id];
  const float* scores_row = scores_in + global_warp_id * nExperts;

#pragma unroll
  for (int e = 0; e < kExpertsPerThread; ++e) {
    int expert_id = lane_id + e * warp_topk::kWARP_SIZE;
    float s = scores_row[expert_id];
    my_smem[expert_id] = softplus_sqrt(s);
  }
  __syncwarp();

  float my_topk_value = 0.0f;
  int my_topk_index = 0;

  if constexpr (hash) {
    int token_id = static_cast<int>(input_ids[global_warp_id]);
    const int* expert_ids = tid2eid + token_id * topK;
    if (lane_id < topK) {
      int expert_id = expert_ids[lane_id];
      my_topk_index = expert_id;
      my_topk_value = my_smem[expert_id];
    }
  } else {
    float scores[kExpertsPerThread];
    int32_t indices[kExpertsPerThread];
#pragma unroll
    for (int e = 0; e < kExpertsPerThread; ++e) {
      int expert_id = lane_id + e * warp_topk::kWARP_SIZE;
      indices[e] = expert_id;
      scores[e] = my_smem[expert_id] + bias[expert_id];
    }

    float topk_vals[topK];
    int32_t topk_idx[topK];
    warp_topk::reduce_topk<topK, kExpertsPerThread>(
        warp, topk_vals, topk_idx, scores, indices, topK);

    if (lane_id < topK) {
      my_topk_index = topk_idx[lane_id];
      my_topk_value = my_smem[my_topk_index];
    }
  }

  float weight_sum = cg::reduce(warp, my_topk_value, cg::plus<float>{});

  if (lane_id < topK) {
    out_weights[global_warp_id * topK + lane_id] =
        (my_topk_value / weight_sum) * route_scale;
    out_indices[global_warp_id * topK + lane_id] = my_topk_index;
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

template <int nExperts, typename TokenIdT = int>
void launch_gate_forward(
    const float* scores_in, const float* bias, const TokenIdT* input_ids,
    const int* tid2eid, float* out_weights, int* out_indices,
    int batch_size, float route_scale, bool is_hash, cudaStream_t stream) {
  constexpr int kTopK = 6;
  constexpr int warps_per_block = 4;
  constexpr int threads_per_block = warps_per_block * 32;
  int const blocks = (batch_size + warps_per_block - 1) / warps_per_block;
  if (is_hash) {
    deepseek_v4_routing::gate_forward_kernel<nExperts, kTopK, true, TokenIdT>
        <<<blocks, threads_per_block, 0, stream>>>(
            scores_in, nullptr, input_ids, tid2eid,
            out_weights, out_indices, batch_size, route_scale);
  } else {
    deepseek_v4_routing::gate_forward_kernel<nExperts, kTopK, false, TokenIdT>
        <<<blocks, threads_per_block, 0, stream>>>(
            scores_in, bias, nullptr, nullptr,
            out_weights, out_indices, batch_size, route_scale);
  }
}

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
  TVM_FFI_ICHECK(num_experts == 256 || num_experts == 384)
      << "DeepSeek V4 fused router supports 256 or 384 experts, got " << num_experts;
  TVM_FFI_ICHECK_EQ(topk, 6)
      << "gate_forward_kernel is compiled for top_k=6, got " << topk;
  TVM_FFI_ICHECK_EQ(correction_bias.size(0), num_experts);
  TVM_FFI_ICHECK_EQ(topk_indices.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(topk_indices.size(1), topk);

  // The gate_forward_kernel always renormalizes and applies route_scale.
  TVM_FFI_ICHECK(renormalize)
      << "gate_forward_kernel always renormalizes; pass renormalize=true";

  const cudaStream_t stream = get_stream(input.device());

  // gate_forward_kernel requires float32 input.
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_float32)
      << "gate_forward_kernel requires float32 input scores";

  auto* out_w = static_cast<float*>(topk_weights.data_ptr());
  auto* out_i = reinterpret_cast<int*>(topk_indices.data_ptr());
  auto* bias_ptr = static_cast<const float*>(correction_bias.data_ptr());
  auto* scores = static_cast<const float*>(input.data_ptr());

  if (num_experts == 256) {
    launch_gate_forward<256, int>(scores, bias_ptr,
                                  static_cast<const int*>(nullptr), nullptr,
                                  out_w, out_i, num_rows, routed_scaling_factor,
                                  false, stream);
  } else {
    launch_gate_forward<384, int>(scores, bias_ptr,
                                  static_cast<const int*>(nullptr), nullptr,
                                  out_w, out_i, num_rows, routed_scaling_factor,
                                  false, stream);
  }

  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "Failed to launch DeepSeek V4 gate_forward kernel: "
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
  TVM_FFI_ICHECK(num_experts == 256 || num_experts == 384)
      << "DeepSeek V4 fused hash router supports 256 or 384 experts, got " << num_experts;
  TVM_FFI_ICHECK_EQ(topk, 6)
      << "gate_forward_kernel is compiled for top_k=6, got " << topk;
  TVM_FFI_ICHECK_EQ(input_ids.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(hash_indices_table.size(1), topk);
  TVM_FFI_ICHECK_EQ(topk_indices.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(topk_indices.size(1), topk);

  TVM_FFI_ICHECK(renormalize)
      << "gate_forward_kernel always renormalizes; pass renormalize=true";
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_float32)
      << "gate_forward_kernel requires float32 input scores";
  // The hash table is read as int32 (expert ids fit in 32 bits and the model
  // casts the table to int32). Reject other dtypes loudly instead of silently
  // misreading e.g. an int64 table as alternating low/high 32-bit halves.
  TVM_FFI_ICHECK(hash_indices_table.dtype() == dl_int32)
      << "hash_indices_table must be int32";

  const cudaStream_t stream = get_stream(input.device());

  auto* out_w = static_cast<float*>(topk_weights.data_ptr());
  auto* out_i = reinterpret_cast<int*>(topk_indices.data_ptr());
  auto* scores = static_cast<const float*>(input.data_ptr());
  auto* tid2eid = static_cast<const int*>(hash_indices_table.data_ptr());

  // input_ids holds token ids and is commonly int64 (torch.long); also accept
  // int32. Dispatch on the real dtype so the kernel reads the ids correctly.
  auto dispatch = [&](auto* ids) {
    using TokenIdT = std::remove_const_t<std::remove_pointer_t<decltype(ids)>>;
    if (num_experts == 256) {
      launch_gate_forward<256, TokenIdT>(scores, nullptr, ids, tid2eid,
                                         out_w, out_i, num_rows,
                                         routed_scaling_factor, true, stream);
    } else {
      launch_gate_forward<384, TokenIdT>(scores, nullptr, ids, tid2eid,
                                         out_w, out_i, num_rows,
                                         routed_scaling_factor, true, stream);
    }
  };

  if (input_ids.dtype() == dl_int32) {
    dispatch(static_cast<const int32_t*>(input_ids.data_ptr()));
  } else if (input_ids.dtype() == dl_int64) {
    dispatch(static_cast<const int64_t*>(input_ids.data_ptr()));
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "hash router input_ids must be int32 or int64";
  }

  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "Failed to launch DeepSeek V4 hash gate_forward kernel: "
      << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softplus_sqrt_topk_flash, softplus_sqrt_topk_flash);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hash_softplus_sqrt_topk_flash,
                              hash_softplus_sqrt_topk_flash);
