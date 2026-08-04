/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 */
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "../trtllm/common/reduceKernelUtils.cuh"

namespace flashinfer {
namespace minimax_ar {

enum class MinimaxDType : int32_t {
  kHALF = 0,
  kBF16 = 1,
  kFLOAT = 2,
};

// 16-byte aligned 8x bf16 and 8-byte aligned 4x bf16 structs; replaces
// the tensorrt_llm::common types which require ENABLE_BF16/ENABLE_FP8.
struct __align__(16) bfloat168 {
  __nv_bfloat16 array[8];
};

struct __align__(8) bfloat164 {
  __nv_bfloat16 array[4];
};

template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half> {
  static constexpr int value = 8;
  using norm_weight_type = bfloat168;
};

template <>
struct ElemsPerAccess<__nv_bfloat16> {
  static constexpr int value = 8;
  using norm_weight_type = bfloat168;
};

template <>
struct ElemsPerAccess<float> {
  static constexpr int value = 4;
  using norm_weight_type = bfloat164;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

struct MiniMaxReduceRMSParams {
  int nranks{};
  int rank{};
  MinimaxDType dtype{MinimaxDType::kBF16};
  int size_q{};        // numel of Q (num_token * head_dim_q)
  int hidden_dim{};    // head_dim_q
  int size_k{};        // numel of K (num_token * head_dim_k)
  int hidden_dim_k{};  // head_dim_k; requires head_dim_q >= head_dim_k
  // Row strides in float4 units for the QK fast-path kernel inputs. 0 = use
  // tightly-packed default (ThreadsPerRow{Q,K}). Setting these to a larger
  // value lets the kernel read a strided slice (e.g. Q from a fused QKV
  // buffer) without an upstream .contiguous() copy. Outputs always use the
  // packed default since rms_norm_out{,_k} are caller-allocated contiguous.
  int q_row_stride_f4{};
  int k_row_stride_f4{};
  void** workspace{};
  void* allreduce_in{};    // Q input
  void* rms_norm_out{};    // Q output
  void* rms_gamma{};       // Q norm weight (bf16)
  void* allreduce_in_k{};  // K input (nullptr for single-matrix path)
  void* rms_norm_out_k{};  // K output
  void* rms_gamma_k{};     // K norm weight (bf16)
  float rms_eps{};
  cudaStream_t stream{};
  bool trigger_completion_at_end{true};
  bool enable_pdl{false};
};

namespace details {

constexpr int kMinimaxReduceRmsWarpSize = 32;

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
    flag_value = *flag_ptr;
    auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] =
          reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
    }
    clear_buf =
        reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int64_t new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int64_t* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int64_t clear_size;
  int flag_value;
};

__device__ __forceinline__ bool is_neg_zero(float v) {
  return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool is_neg_zero(float4 v) {
  return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) || is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero() {
  float4 vec;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
  }
  return vec;
}

template <int Dim>
__device__ __forceinline__ float rms_rsqrt(float& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v = rsqrtf((v * kInvDim) + eps);
  return v;
}

template <int Dim>
__device__ __forceinline__ float4 rms_rsqrt(float4& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v.x = rsqrtf((v.x * kInvDim) + eps);
  v.y = rsqrtf((v.y * kInvDim) + eps);
  v.z = rsqrtf((v.z * kInvDim) + eps);
  v.w = rsqrtf((v.w * kInvDim) + eps);
  return v;
}

__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
  float4 val;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
               : "l"(addr));
  return val;
}

__device__ __forceinline__ float ld_global_volatile(float* addr) {
  float val;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(val) : "l"(addr));
  return val;
}

template <uint32_t kNumThreads, typename T>
__device__ __forceinline__ void local_warp_reduce_sum(T& value, uint32_t active_mask = 0xffffffffu) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kMinimaxReduceRmsWarpSize);
#pragma unroll
  for (int mask = kNumThreads / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(active_mask, value, mask, kMinimaxReduceRmsWarpSize);
  }
}

template <uint32_t kNumThreads, typename T, int ArraySize = 4>
__device__ __forceinline__ void local_warp_reduce_sum_array(T* value_ptr,
                                                             uint32_t active_mask = 0xffffffffu) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kMinimaxReduceRmsWarpSize);
#pragma unroll
  for (int i = 0; i < ArraySize; ++i) {
#pragma unroll
    for (int mask = kNumThreads / 2; mask > 0; mask >>= 1) {
      value_ptr[i] += __shfl_xor_sync(active_mask, value_ptr[i], mask, kMinimaxReduceRmsWarpSize);
    }
  }
}

constexpr int next_pow2(int val) {
  int result = 1;
  while (result < val) {
    result <<= 1;
  }
  return result;
}

template <typename DType>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(MiniMaxReduceRMSParams const& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> + access_id_in_token;
    access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
    tot_access = params.size_q / kElemsPerAccess<DType>;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

// Single-matrix (Q-only) Lamport AR + RMSNorm. Used when dims don't match the
// Q=6144 / K=1024 fast path.
template <typename DType, int NRanks, bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024)
    minimax_reduce_rms_kernel_lamport(MiniMaxReduceRMSParams params) {
  IndexHelper<DType> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  int tot_tokens = params.size_q / params.hidden_dim;
  float4 clear_vec = get_neg_zero();
  __shared__ float shared_vars_all_ranks;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / kElemsPerAccess<DType>;
  for (int idx = access_id; idx < tot_access; idx += access_stride, token_id += token_stride) {
    alignas(16) DType vals[kElemsPerAccess<DType>];
    float sum_variance = 0.F;
    *reinterpret_cast<float4*>(vals) = reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      sum_variance += static_cast<float>(vals[i]) * static_cast<float>(vals[i]);
    }
    tensorrt_llm::common::blockReduceSumV2<float, 1>(&sum_variance);
    if (is_neg_zero(sum_variance)) {
      sum_variance = 0.F;
    }
    if (threadIdx.x == 0) {
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        reinterpret_cast<float*>(comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] =
            sum_variance;
      }
      bool done = false;
      float vals_all_ranks[NRanks];
      while (!done) {
        done = true;
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
          vals_all_ranks[r] = ld_global_volatile(
              &reinterpret_cast<float*>(comm.data_bufs[params.rank])[(r * tot_tokens) + token_id]);
          done &= !is_neg_zero(vals_all_ranks[r]);
        }
      }
      sum_variance = 0.F;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        sum_variance += vals_all_ranks[r];
      }
      sum_variance =
          rsqrtf(sum_variance / NRanks / static_cast<float>(params.hidden_dim) + params.rms_eps);
      shared_vars_all_ranks = sum_variance;
    }

    __syncthreads();
    sum_variance = shared_vars_all_ranks;

    __nv_bfloat16 norm_weight[kElemsPerAccess<DType>];
    *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight) =
        reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(
            params.rms_gamma)[access_id_in_token];

#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      vals[i] = static_cast<DType>(static_cast<float>(vals[i]) * sum_variance *
                                   static_cast<float>(norm_weight[i]));
    }

    reinterpret_cast<float4*>(params.rms_norm_out)[idx] = *reinterpret_cast<float4*>(vals);
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }
  comm.update(tot_tokens * NRanks * sizeof(float) / sizeof(DType));
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

// Float4/QK fused fast path: processes 4 rows at once and splits Q and K across
// different warps. Used when (hidden_dim_q * nranks == OriginQDim) and
// (hidden_dim_k * nranks == OriginKDim).
template <typename DType, int NRanks, int OriginQDim, int OriginKDim, int TokenPerBlock = 4,
          bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024)
    minimax_reduce_qk_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params) {
  static_assert(TokenPerBlock == 1 || TokenPerBlock == 4, "TokenPerBlock must be 1 or 4");
  constexpr int RankQDim = OriginQDim / NRanks;
  constexpr int RankKDim = OriginKDim / NRanks;
  constexpr int ThreadsPerRowQ = RankQDim / kElemsPerAccess<DType>;
  constexpr int ThreadsPerRowK = RankKDim / kElemsPerAccess<DType>;
  // 0 = caller didn't set a stride; assume tightly-packed inputs.
  int q_in_row_stride =
      params.q_row_stride_f4 == 0 ? ThreadsPerRowQ : params.q_row_stride_f4;
  int k_in_row_stride =
      params.k_row_stride_f4 == 0 ? ThreadsPerRowK : params.k_row_stride_f4;
  constexpr int NumWarpQ =
      (ThreadsPerRowQ + kMinimaxReduceRmsWarpSize - 1) / kMinimaxReduceRmsWarpSize;
  constexpr int NumWarpK =
      (ThreadsPerRowK + kMinimaxReduceRmsWarpSize - 1) / kMinimaxReduceRmsWarpSize;
  int tot_tokens = params.size_q / RankQDim;
  int tot_groups = (tot_tokens + TokenPerBlock - 1) / TokenPerBlock;

  using AccumType = std::conditional_t<TokenPerBlock == 1, float, float4>;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  cg::grid_group grid = cg::this_grid();
  int group_id = grid.cluster_rank();
  int access_id_in_token = cluster.thread_rank();
  int group_stride = grid.num_clusters();
#else
  int group_id = blockIdx.x;
  int access_id_in_token = threadIdx.x;
  int group_stride = gridDim.x;
#endif
  bool is_q = (access_id_in_token < NumWarpQ * kMinimaxReduceRmsWarpSize);
  int q_thread_idx = access_id_in_token;
  int k_thread_idx = (access_id_in_token - (NumWarpQ * kMinimaxReduceRmsWarpSize));
  bool is_valid_token =
      is_q ? (access_id_in_token < ThreadsPerRowQ) : (k_thread_idx < ThreadsPerRowK);
  float4 clear_vec = get_neg_zero();

  __shared__ float block_reduce_sum[TokenPerBlock][kMinimaxReduceRmsWarpSize + 1];
  __shared__ float global_scale_q[TokenPerBlock];
  __shared__ float global_scale_k[TokenPerBlock];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  LamportComm<NRanks> comm(params.workspace, params.rank);

  __nv_bfloat16 norm_weight[kElemsPerAccess<DType>]{};
  if (access_id_in_token < NumWarpQ * kMinimaxReduceRmsWarpSize) {
    if (is_valid_token) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type const*>(
              params.rms_gamma)[access_id_in_token];
    }
  } else {
    if (is_valid_token) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type const*>(
              params.rms_gamma_k)[k_thread_idx];
    }
  }

  for (int g = group_id; g < tot_groups; g += group_stride) {
    alignas(16) DType vals[TokenPerBlock][kElemsPerAccess<DType>]{};
    float warp_sum_variance[TokenPerBlock]{0.F};

    if (is_q) {
#pragma unroll
      for (int row = 0; row < TokenPerBlock; ++row) {
        int token_r = (g * TokenPerBlock) + row;
        if (token_r >= tot_tokens || (!is_valid_token)) {
          continue;
        }
        int idx_r = (token_r * q_in_row_stride) + access_id_in_token;
        *reinterpret_cast<float4*>(&vals[row][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          auto x = static_cast<float>(vals[row][i]);
          warp_sum_variance[row] += x * x;
        }
      }
    } else {
#pragma unroll
      for (int row = 0; row < TokenPerBlock; ++row) {
        int token_r = (g * TokenPerBlock) + row;
        if (token_r >= tot_tokens || (!is_valid_token)) {
          continue;
        }

        int idx_r = (token_r * k_in_row_stride) + k_thread_idx;
        *reinterpret_cast<float4*>(&vals[row][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          auto x = static_cast<float>(vals[row][i]);
          warp_sum_variance[row] += x * x;
        }
      }
    }

    local_warp_reduce_sum_array<kMinimaxReduceRmsWarpSize, float, TokenPerBlock>(warp_sum_variance);
    int line = threadIdx.x & (kMinimaxReduceRmsWarpSize - 1);
    if (line == 0) {
#pragma unroll
      for (int _ = 0; _ < TokenPerBlock; ++_) {
        block_reduce_sum[_][threadIdx.x / kMinimaxReduceRmsWarpSize] = warp_sum_variance[_];
      }
    }
    __syncthreads();
    int tid = threadIdx.x;

    if (tid < kMinimaxReduceRmsWarpSize) {
      constexpr int kNumWarpQPow2 = next_pow2(NumWarpQ) > NRanks ? next_pow2(NumWarpQ) : NRanks;
      float local_sum[TokenPerBlock];
#pragma unroll
      for (int _ = 0; _ < TokenPerBlock; ++_) {
        local_sum[_] = tid < NumWarpQ ? block_reduce_sum[_][tid] : 0.F;
      }
      local_warp_reduce_sum_array<kNumWarpQPow2, float, TokenPerBlock>(local_sum);
      if (tid < NRanks) {
#pragma unroll
        for (int _ = 0; _ < TokenPerBlock; ++_) {
          if (is_neg_zero(local_sum[_])) {
            local_sum[_] = 0.F;
          }
        }

        reinterpret_cast<AccumType*>(
            comm.data_bufs[tid])[(params.rank * tot_groups * 2) + (2 * g)] =
            *reinterpret_cast<AccumType*>(local_sum);
        bool done = false;
        AccumType var_all_ranks;
        while (!done) {
          done = true;
          var_all_ranks = ld_global_volatile(&reinterpret_cast<AccumType*>(
              comm.data_bufs[params.rank])[(tid * tot_groups * 2) + (2 * g)]);
          done &= !is_neg_zero(var_all_ranks);
        }
        constexpr uint32_t kActiveMask = (1 << NRanks) - 1;
        local_warp_reduce_sum_array<NRanks, float, TokenPerBlock>(
            reinterpret_cast<float*>(&var_all_ranks), kActiveMask);
        if (tid == 0) {
          *reinterpret_cast<AccumType*>(global_scale_q) =
              rms_rsqrt<OriginQDim>(var_all_ranks, params.rms_eps);
        }
      }
    } else if (threadIdx.x >= kMinimaxReduceRmsWarpSize * NumWarpQ &&
               threadIdx.x < kMinimaxReduceRmsWarpSize * (NumWarpQ + 1)) {
      constexpr int kNumWarpKPow2 = next_pow2(NumWarpK) > NRanks ? next_pow2(NumWarpK) : NRanks;
      float local_sum[TokenPerBlock];
#pragma unroll
      for (int _ = 0; _ < TokenPerBlock; ++_) {
        local_sum[_] = k_thread_idx < NumWarpK ? block_reduce_sum[_][NumWarpQ + k_thread_idx] : 0.F;
      }
      local_warp_reduce_sum_array<kNumWarpKPow2, float, TokenPerBlock>(local_sum);
      if (k_thread_idx < NRanks) {
#pragma unroll
        for (int _ = 0; _ < TokenPerBlock; ++_) {
          if (is_neg_zero(local_sum[_])) {
            local_sum[_] = 0.F;
          }
        }
        reinterpret_cast<AccumType*>(
            comm.data_bufs[k_thread_idx])[(params.rank * tot_groups * 2) + (2 * g + 1)] =
            *reinterpret_cast<AccumType*>(local_sum);
        bool done = false;
        AccumType var_all_ranks;
        while (!done) {
          done = true;
          var_all_ranks = ld_global_volatile(&reinterpret_cast<AccumType*>(
              comm.data_bufs[params.rank])[(k_thread_idx * tot_groups * 2) + (2 * g + 1)]);
          done &= !is_neg_zero(var_all_ranks);
        }
        constexpr uint32_t kActiveMask = (1 << NRanks) - 1;
        local_warp_reduce_sum_array<NRanks, float, TokenPerBlock>(
            reinterpret_cast<float*>(&var_all_ranks), kActiveMask);
        if (k_thread_idx == 0) {
          *reinterpret_cast<AccumType*>(global_scale_k) =
              rms_rsqrt<OriginKDim>(var_all_ranks, params.rms_eps);
        }
      }
    }
    __syncthreads();
    if (is_q) {
#pragma unroll
      for (int _ = 0; _ < TokenPerBlock; ++_) {
        warp_sum_variance[_] = global_scale_q[_];
      }
#pragma unroll
      for (int r = 0; r < TokenPerBlock; ++r) {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          vals[r][i] = static_cast<DType>(static_cast<float>(vals[r][i]) * warp_sum_variance[r] *
                                          static_cast<float>(norm_weight[i]));
        }
        int token_r = (g * TokenPerBlock) + r;
        if (token_r >= tot_tokens || (!is_valid_token)) {
          continue;
        }
        int idx_r = (token_r * ThreadsPerRowQ) + access_id_in_token;
        reinterpret_cast<float4*>(params.rms_norm_out)[idx_r] =
            *reinterpret_cast<float4*>(&vals[r][0]);
      }
    } else {
#pragma unroll
      for (int _ = 0; _ < TokenPerBlock; ++_) {
        warp_sum_variance[_] = global_scale_k[_];
      }
#pragma unroll
      for (int r = 0; r < TokenPerBlock; ++r) {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          vals[r][i] = static_cast<DType>(static_cast<float>(vals[r][i]) * warp_sum_variance[r] *
                                          static_cast<float>(norm_weight[i]));
        }
        int token_r = (g * TokenPerBlock) + r;
        if (token_r >= tot_tokens || (!is_valid_token)) {
          continue;
        }
        int idx_r = (token_r * ThreadsPerRowK) + k_thread_idx;
        reinterpret_cast<float4*>(params.rms_norm_out_k)[idx_r] =
            *reinterpret_cast<float4*>(&vals[r][0]);
      }
    }
  }

  int clear_access = static_cast<int>(comm.clear_size / (sizeof(float4) / sizeof(DType)));
  int clear_stride = group_stride * blockDim.x;

  for (int idx = group_id * blockDim.x + threadIdx.x; idx < clear_access; idx += clear_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }

  comm.update((2 * tot_groups * TokenPerBlock * sizeof(float) / sizeof(DType) * NRanks));
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

inline int get_sm_count() {
  static int const sm_count = []() {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.multiProcessorCount;
  }();
  return sm_count;
}

inline int get_sm_version() {
  static int const sm_version = []() {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.major * 10 + prop.minor;
  }();
  return sm_version;
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher(MiniMaxReduceRMSParams const& params) {
  int token_num = params.size_q / params.hidden_dim;
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = token_num;
  int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
  int block_size = threads_per_token;
  int grid_size =
      (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = params.enable_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = get_sm_version() >= 90 ? 2 : 0;
  if (params.trigger_completion_at_end) {
    cudaLaunchKernelEx(&cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks, true>, params);
  } else {
    cudaLaunchKernelEx(&cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks, false>, params);
  }
}

template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
void minimax_reduce_rms_kernel_launcher_float4(MiniMaxReduceRMSParams const& params) {
  int token_num = params.size_q / params.hidden_dim;
  int tot_groups = (token_num + 3) / 4;
  if (tot_groups == 0) {
    return;
  }
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = tot_groups;
  int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
  int access_per_row_k =
      (params.allreduce_in_k != nullptr) ? (params.hidden_dim_k / kElemsPerAccess<DType>) : 0;
  auto const divUp = [](int a, int b) { return (a + b - 1) / b * b; };
  int block_size = divUp(access_per_row_q, kMinimaxReduceRmsWarpSize) +
                   ((params.allreduce_in_k != nullptr)
                        ? divUp(access_per_row_k, kMinimaxReduceRmsWarpSize)
                        : 0);
  int grid_size =
      (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = params.enable_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = get_sm_version() >= 90 ? 2 : 0;

  if (params.trigger_completion_at_end) {
    cudaLaunchKernelEx(
        &cfg, minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim, OriginKDim, 4,
                                                          true>,
        params);
  } else {
    cudaLaunchKernelEx(
        &cfg, minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim, OriginKDim, 4,
                                                          false>,
        params);
  }
}

template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params) {
  bool use_float4 = (params.allreduce_in_k != nullptr) &&
                    (params.hidden_dim * params.nranks == 6144) &&
                    (params.hidden_dim_k * params.nranks == 1024);

  if (params.dtype == MinimaxDType::kHALF) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<half, NRanks, 6144, 1024>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<half, NRanks>(params);
    }
  } else if (params.dtype == MinimaxDType::kBF16) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks, 6144, 1024>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<__nv_bfloat16, NRanks>(params);
    }
  } else {
    // float path. The float4 fast-path template with DType=float works too
    // (kElemsPerAccess=4) but the generic path is more conservative.
    minimax_reduce_rms_kernel_launcher<float, NRanks>(params);
  }
}

}  // namespace details

inline void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params) {
  switch (params.nranks) {
    case 2:
      details::dispatch_dtype<2>(params);
      break;
    case 4:
      details::dispatch_dtype<4>(params);
      break;
    case 8:
      details::dispatch_dtype<8>(params);
      break;
    case 16:
      details::dispatch_dtype<16>(params);
      break;
    default:
      // unsupported nranks; caller validates.
      break;
  }
}

}  // namespace minimax_ar
}  // namespace flashinfer
