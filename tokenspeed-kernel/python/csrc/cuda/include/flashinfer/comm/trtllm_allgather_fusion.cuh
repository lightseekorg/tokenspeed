/*
 * Copyright (c) 2025 by FlashInfer team.
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

#ifndef FLASHINFER_TRTLLM_ALLGATHER_FUSION_CUH_
#define FLASHINFER_TRTLLM_ALLGATHER_FUSION_CUH_

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include <cstdint>
#include <cuda/std/optional>
#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../fp4_layout.cuh"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "trtllm_reducescatter_fusion.cuh"

namespace flashinfer {

namespace trtllm_allgather_fusion {

using flashinfer::QuantizationSFLayout;

namespace details {

static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;

}  // namespace details

enum class AllGatherFusionPattern : int {
  kAllGather = 0,
  kAllGatherfusedRMS = 1,
  kAllGatherfusedRMSFP8BlockWiseQuant = 2,
};

enum class QuantType : int {
  kNone = 0,
  kFP8 = 1,
  kFP4 = 2,
  kFP8BlockWise = 3,
};

template <AllGatherFusionPattern Pattern>
struct AllGatherPatternTraits;

#define DEFINE_AG_PATTERN_TRAITS(pattern, hasAllGatherOut, hasFusedRMSNorm, hasNormOut, quantType) \
  template <>                                                                                 \
  struct AllGatherPatternTraits<pattern> {                                                    \
    static constexpr bool kHasAllGatherOut = hasAllGatherOut;                                 \
    static constexpr bool kHasFusedRMSNorm = hasFusedRMSNorm;                                 \
    static constexpr bool kHasNormOut = hasNormOut;                                           \
    static constexpr QuantType kQuantType = quantType;                                        \
  };

DEFINE_AG_PATTERN_TRAITS(AllGatherFusionPattern::kAllGather, true, false, false, QuantType::kNone);
DEFINE_AG_PATTERN_TRAITS(AllGatherFusionPattern::kAllGatherfusedRMS, true, true, true, QuantType::kNone);
DEFINE_AG_PATTERN_TRAITS(AllGatherFusionPattern::kAllGatherfusedRMSFP8BlockWiseQuant, true, true, true, QuantType::kFP8BlockWise);

#undef DEFINE_AG_PATTERN_TRAITS

template <AllGatherFusionPattern Pattern>
constexpr bool HasAllGatherOut = AllGatherPatternTraits<Pattern>::kHasAllGatherOut;
template <AllGatherFusionPattern Pattern>
constexpr bool HasFusedRMSNorm = AllGatherPatternTraits<Pattern>::kHasFusedRMSNorm;
template <AllGatherFusionPattern Pattern>
constexpr bool HasNormOut = AllGatherPatternTraits<Pattern>::kHasNormOut;
template <AllGatherFusionPattern Pattern>
constexpr QuantType GetQuantType = AllGatherPatternTraits<Pattern>::kQuantType;

template <typename T>
struct AllGatherFusionParams {
  int nranks;
  int rank;
  int size;
  int hidden_dim;
  int num_token_current_rank;
  int num_token_all_group;
  void** workspace;
  void* allgather_in;
  void* allgather_out;
  void* x_norm_out;
  void* y_norm_out;
  void* quant_out;
  void* scale_out;
  void* x_rms_gamma;
  void* y_rms_gamma;
  float x_rms_eps;
  float y_rms_eps;
  int q_lora_rank;
  int kv_lora_rank;
  int qk_rope_head_dim;
  int y_norm_stride;
  int scale_stride;
  int q_lora_access_end;      // (q_lora_rank + VEC_SIZE - 1) / VEC_SIZE
  int kv_lora_start_access;   // q_lora_rank / VEC_SIZE
  int kv_lora_access_end;     // (q_lora_rank + kv_lora_rank + VEC_SIZE - 1) / VEC_SIZE
  bool use_oneshot;
  cudaStream_t stream;
  AllGatherFusionPattern pattern;
  bool trigger_completion_at_end = true;
};

template <typename T>
class AIndexHelper {
 public:
  __device__ __forceinline__ AIndexHelper(AllGatherFusionParams<T> const& params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
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
    access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    access_stride = token_stride * params.hidden_dim / VEC_SIZE;
    tot_access = params.size / VEC_SIZE;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

template <AllGatherFusionPattern Pattern, typename T>
class AllGatherFusedOp {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  static constexpr float FP8_E4M3_MAX = 448.0f;

 public:
  __device__ __forceinline__ AllGatherFusedOp(AllGatherFusionParams<T> const& params)
      : m_params(params) {
    if constexpr (Pattern == AllGatherFusionPattern::kAllGatherfusedRMSFP8BlockWiseQuant) {
      // 需要在这里load gamma吗
    }
  }

  __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int offset_idx) {
    if constexpr (HasAllGatherOut<Pattern>) {
      val.store(reinterpret_cast<T*>(m_params.allgather_out) + offset_idx * VEC_SIZE);
    }

    if constexpr (HasFusedRMSNorm<Pattern>) {
      int token_id = offset_idx / (m_params.hidden_dim / VEC_SIZE);
      int access_id_in_token = offset_idx % (m_params.hidden_dim / VEC_SIZE);

      bool is_x_valid = access_id_in_token < m_params.q_lora_access_end;
      bool is_y_valid = access_id_in_token >= m_params.kv_lora_start_access && access_id_in_token < m_params.kv_lora_access_end;

      vec_t<T, VEC_SIZE> val_x, val_y;
      vec_t<T, VEC_SIZE> x_gamma_val, y_gamma_val;
      vec_t<T, VEC_SIZE> norm_out_x, norm_out_y;

      if (is_x_valid) {
        val_x.load(reinterpret_cast<T*>(m_params.allgather_out) + offset_idx * VEC_SIZE);
        x_gamma_val.load(reinterpret_cast<T*>(m_params.x_rms_gamma) + access_id_in_token * VEC_SIZE);
      }
      if (is_y_valid) {
        val_y.load(reinterpret_cast<T*>(m_params.allgather_out) + offset_idx * VEC_SIZE);
        int y_offset_in_token = access_id_in_token - m_params.kv_lora_start_access;
        y_gamma_val.load(reinterpret_cast<T*>(m_params.y_rms_gamma) + y_offset_in_token * VEC_SIZE);
      }

      rms_norm_fused(val_x, val_y, x_gamma_val, y_gamma_val, norm_out_x, norm_out_y, is_x_valid, is_y_valid);

      if (is_x_valid) {
        // x_norm_out [total_tokens, q_lora_rank]
        int x_offset_in_token = access_id_in_token;
        int x_global_offset = token_id * (m_params.q_lora_rank / VEC_SIZE) + x_offset_in_token;
        norm_out_x.store(reinterpret_cast<T*>(m_params.x_norm_out) + x_global_offset * VEC_SIZE);

        // Apply FP8 BlockWise quantization if needed
        if constexpr (GetQuantType<Pattern> == QuantType::kFP8BlockWise) {
          vec_t<__nv_fp8_e4m3, VEC_SIZE> quant_out = block_quant_fp8(
              norm_out_x, reinterpret_cast<float*>(m_params.scale_out), m_params.scale_stride, token_id);

          quant_out.store(reinterpret_cast<__nv_fp8_e4m3*>(m_params.quant_out) +
                          x_global_offset * VEC_SIZE);
        }
      }
      if (is_y_valid) {
        // y_norm_out [total_tokens, kv_lora_rank]
        int y_offset_in_token = access_id_in_token - m_params.kv_lora_start_access;
        int y_global_elem_offset = token_id * m_params.y_norm_stride + y_offset_in_token * VEC_SIZE;
        norm_out_y.store(reinterpret_cast<T*>(m_params.y_norm_out) + y_global_elem_offset);
      }
    }
  }

  __device__ __forceinline__ void rms_norm_fused(vec_t<T, VEC_SIZE> const& val_x,
                                                  vec_t<T, VEC_SIZE> const& val_y,
                                                  vec_t<T, VEC_SIZE> const& x_gamma_val,
                                                  vec_t<T, VEC_SIZE> const& y_gamma_val,
                                                  vec_t<T, VEC_SIZE>& norm_out_x,
                                                  vec_t<T, VEC_SIZE>& norm_out_y,
                                                  bool is_x_valid, bool is_y_valid) {
    __shared__ float s_vals[2];

    float acc_x = 0.f;
    float acc_y = 0.f;

    if (is_x_valid) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const*>(&val_x)[i]);
        acc_x += v * v;
      }
    }

    if (is_y_valid) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const*>(&val_y)[i]);
        acc_y += v * v;
      }
    }

    float acc_vals[2] = {acc_x, acc_y};
    flashinfer::trtllm_reducescatter_fusion::utils::blockReduceSumV2<float, 2>(acc_vals);
    acc_x = acc_vals[0];
    acc_y = acc_vals[1];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    if (cluster.num_blocks() > 1) {
      __shared__ float s_acc[2];
      if (threadIdx.x == 0) {
        s_acc[0] = acc_x;
        s_acc[1] = acc_y;
        acc_x = 0.f;
        acc_y = 0.f;
      }
      cluster.sync();
      if (threadIdx.x == 0) {
        for (int i = 0; i < cluster.num_blocks(); ++i) {
          acc_x += *cluster.map_shared_rank(&s_acc[0], i);
          acc_y += *cluster.map_shared_rank(&s_acc[1], i);
        }
      }
      cluster.sync();
    }
#endif

    if (threadIdx.x == 0) {
      s_vals[0] = rsqrtf(acc_x / m_params.q_lora_rank + m_params.x_rms_eps);
      s_vals[1] = rsqrtf(acc_y / m_params.kv_lora_rank + m_params.y_rms_eps);
    }
    __syncthreads();

    float s_val_x = s_vals[0];
    float s_val_y = s_vals[1];

    if (is_x_valid) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        reinterpret_cast<T*>(&norm_out_x)[i] =
            static_cast<T>(static_cast<float>(reinterpret_cast<T const*>(&val_x)[i]) * s_val_x *
                          static_cast<float>(reinterpret_cast<T const*>(&x_gamma_val)[i]));
      }
    }

    if (is_y_valid) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        reinterpret_cast<T*>(&norm_out_y)[i] =
            static_cast<T>(static_cast<float>(reinterpret_cast<T const*>(&val_y)[i]) * s_val_y *
                          static_cast<float>(reinterpret_cast<T const*>(&y_gamma_val)[i]));
      }
    }
    __syncthreads();
  }

 protected:
  __device__ __forceinline__ vec_t<__nv_fp8_e4m3, VEC_SIZE> block_quant_fp8(
      vec_t<T, VEC_SIZE> normed_res, float* scale_out, int32_t scale_stride, int32_t token_id) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    int32_t access_id_in_token = cluster.thread_rank();

    float _absmax = 1e-10f;
    vec_t<__nv_fp8_e4m3, VEC_SIZE> quanted_out;
    auto tile_16 = cg::tiled_partition<16>(cg::this_thread_block());
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      float val = static_cast<float>(normed_res[i]);
      _absmax = std::max(_absmax, fabs(val));
    }
    _absmax = cg::reduce(tile_16, _absmax, [](float a, float b) { return a > b ? a : b; });

    float scale;
    asm("div.full.f32 %0, %1, %2;" : "=f"(scale) : "f"(_absmax), "f"(FP8_E4M3_MAX));

    // directly write scale to scale_out
    if (tile_16.thread_rank() == 0) {
      int32_t col_idx = int32_t(access_id_in_token / 16);
      int32_t offset = col_idx * scale_stride + token_id;
      *(scale_out + offset) = scale;
    }
    tile_16.sync();

    float reverse_scale = 1.0 / scale;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      float x = static_cast<float>(normed_res[i]) * reverse_scale;
      float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
      reinterpret_cast<__nv_fp8_e4m3*>(&quanted_out)[i] = static_cast<__nv_fp8_e4m3>(r);
    }
    return quanted_out;
  }

 private:
  AllGatherFusionParams<T> const& m_params;
};

template <AllGatherFusionPattern Pattern, typename T, int NRanks,
          bool TriggerCompletionAtEnd = true>
__global__ __launch_bounds__(264, 1) void allgather_fusion_kernel_oneshot_lamport(AllGatherFusionParams<T> params,
                                                        std::array<int, NRanks> begin_tokens,
                                                        std::array<int, NRanks> token_num_per_ranks) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

  AIndexHelper<T> index_helper(params);
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;

  vec_t<T, VEC_SIZE> clear_vec;
  clear_vec.fill(flashinfer::trtllm_reducescatter_fusion::utils::neg_zero_v<T>);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  flashinfer::trtllm_reducescatter_fusion::RLamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / VEC_SIZE;

  int my_begin_token = begin_tokens[params.rank];
  int my_token_num = token_num_per_ranks[params.rank];
  int my_start_access = my_begin_token * params.hidden_dim / VEC_SIZE;
  int my_end_access = (my_begin_token + my_token_num) * params.hidden_dim / VEC_SIZE;

  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    if (idx >= my_start_access && idx < my_end_access) {
      vec_t<T, VEC_SIZE> val;
      val.load(reinterpret_cast<T*>(params.allgather_in) + (idx - my_start_access) * VEC_SIZE);
      flashinfer::trtllm_reducescatter_fusion::utils::remove_neg_zero(val);

#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        val.store(reinterpret_cast<T*>(comm.data_bufs[r]) + idx * VEC_SIZE);
      }
    }
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    // Clear comm buffer that previous kernel used
    clear_vec.store(reinterpret_cast<T*>(comm.clear_buf) + idx * VEC_SIZE);
  }
  __syncthreads();

  AllGatherFusedOp<Pattern, T> fused_op(params);

  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> val;
    bool done = false;
    while (!done) {
      val.load_global_volatile(reinterpret_cast<T*>(comm.data_bufs[params.rank]) +
                               idx * VEC_SIZE);
      done = !flashinfer::trtllm_reducescatter_fusion::utils::has_neg_zero(val);
    }

    fused_op(val, idx);
  }

  // all-gather中, 每个rank上的Lamport buffer只存储一份完整的gathered tensor（tot_access = params.size / VEC_SIZE;）
  comm.update(params.size);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <AllGatherFusionPattern Pattern, typename T, int NRanks,
          bool TriggerCompletionAtEnd = true>
__global__ void allgather_fusion_kernel_twoshot_sync(
    AllGatherFusionParams<T> params, std::array<int, NRanks> begin_tokens,
    std::array<int, NRanks> token_num_per_ranks) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

  AIndexHelper<T> index_helper(params);
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  flashinfer::trtllm_reducescatter_fusion::RSyncComm<NRanks> comm(params.workspace);

  int my_token_num = token_num_per_ranks[params.rank];
  int my_access_count = my_token_num * params.hidden_dim / VEC_SIZE;

  for (int idx = access_id; idx < my_access_count; idx += access_stride) {
    reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx] =
        reinterpret_cast<float4*>(params.allgather_in)[idx];
  }

  flashinfer::trtllm_reducescatter_fusion::RBarrier<NRanks> barrier(params.rank, comm);
  barrier.sync();

  AllGatherFusedOp<Pattern, T> fused_op(params);

  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    int owner_rank = -1;
    int tokens_per_rank = params.size / params.hidden_dim / NRanks;
    int remaining = (params.size / params.hidden_dim) % NRanks;
    int token_idx = idx / (params.hidden_dim / VEC_SIZE);
    int threshold = remaining * (tokens_per_rank + 1);

    if (remaining == 0) {
      owner_rank = token_idx / tokens_per_rank;
    } else {
      if (token_idx < threshold) {
        owner_rank = token_idx / (tokens_per_rank + 1);
      } else {
        owner_rank = remaining + (token_idx - threshold) / tokens_per_rank;
      }
    }

    int owner_start_token = begin_tokens[owner_rank];
    int owner_start_access = owner_start_token * params.hidden_dim / VEC_SIZE;
    int local_idx = idx - owner_start_access;

    vec_t<T, VEC_SIZE> val;
    val.load(reinterpret_cast<T*>(comm.comm_bufs[owner_rank]) + local_idx * VEC_SIZE);
    fused_op(val, idx);
  }

  comm.update(barrier.m_flag_value);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <AllGatherFusionPattern Pattern, typename T, int NRanks,
          bool TriggerCompletionAtEnd = true>
int get_registers_per_thread_oneshot_ag() {
  auto kernel = allgather_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, TriggerCompletionAtEnd>;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  return attr.numRegs;
}

template <AllGatherFusionPattern Pattern, typename T, int NRanks>
int get_registers_per_thread_twoshot_ag() {
  auto kernel = allgather_fusion_kernel_twoshot_sync<Pattern, T, NRanks>;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  return attr.numRegs;
}

template <AllGatherFusionPattern Pattern, typename T, int NRanks>
cudaError_t launch_oneshot_allgather(AllGatherFusionParams<T> const& params,
                                     cudaLaunchConfig_t& cfg,
                                     std::array<int, NRanks> begin_tokens,
                                     std::array<int, NRanks> token_num_per_ranks) {
  bool trigger_completion_at_end = params.trigger_completion_at_end;
  if (trigger_completion_at_end) {
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &cfg,
        allgather_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, true>,
        params, begin_tokens, token_num_per_ranks));
  } else {
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &cfg,
        allgather_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, false>,
        params, begin_tokens, token_num_per_ranks));
  }
  return cudaSuccess;
}

template <AllGatherFusionPattern Pattern, typename T, int NRanks>
cudaError_t launch_twoshot_allgather(AllGatherFusionParams<T> const& params,
                                     cudaLaunchConfig_t& cfg,
                                     std::array<int, NRanks> begin_tokens,
                                     std::array<int, NRanks> token_num_per_ranks) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg, allgather_fusion_kernel_twoshot_sync<Pattern, T, NRanks>, params,
      begin_tokens, token_num_per_ranks));
  return cudaSuccess;
}

template <AllGatherFusionPattern Pattern, typename T, int NRanks>
cudaError_t allgather_fusion_kernel_launcher(AllGatherFusionParams<T> const& params,
                                             bool launch_with_pdl) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  FLASHINFER_CHECK(params.size % params.hidden_dim == 0, "params.size % params.hidden_dim != 0");
  FLASHINFER_CHECK(params.hidden_dim % VEC_SIZE == 0, "params.hidden_dim % VEC_SIZE != 0");

  static int SM = flashinfer::trtllm_reducescatter_fusion::utils::getSMVersion();
  int token_num_all_group = params.num_token_all_group;
  int token_num = params.size / params.hidden_dim;
  bool oneshot = params.use_oneshot;
  int cluster_num = token_num_all_group;
  std::array<int, NRanks> begin_tokens, token_num_per_ranks;

  int remaining_token = token_num % NRanks;
  int token_num_per_rank = token_num / NRanks;

  for (int r = 0; r < NRanks; ++r) {
    begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
    token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
  }

  int threads_per_token = params.hidden_dim / VEC_SIZE;
  int cluster_size;
  if (SM >= 90) {
    cluster_size = 8;
  } else {
    cluster_size = 1;
  }

  while (threads_per_token % cluster_size != 0 && cluster_size > 1) {
    cluster_size /= 2;
  }

  int threads_per_block = threads_per_token / cluster_size;
  while (threads_per_block < 256 && cluster_size >= 2) {
    threads_per_block *= 2;
    cluster_size /= 2;
  }

  int sm_count = flashinfer::trtllm_reducescatter_fusion::get_sm_count();
  int registers_per_thread;

  if (oneshot) {
    if (params.trigger_completion_at_end) {
      registers_per_thread =
          get_registers_per_thread_oneshot_ag<Pattern, T, NRanks, true>();
    } else {
      registers_per_thread =
          get_registers_per_thread_oneshot_ag<Pattern, T, NRanks, false>();
    }
  } else {
    registers_per_thread = get_registers_per_thread_twoshot_ag<Pattern, T, NRanks>();
  }

  static int max_registers = -1;
  if (max_registers < 0) {
    max_registers = flashinfer::trtllm_reducescatter_fusion::utils::getSMRegisters();
  }

  int max_threads_per_block = min(max_registers / registers_per_thread, 1024);

  while (cluster_num * cluster_size > sm_count && cluster_size > 1 &&
         threads_per_block <= max_threads_per_block / 2) {
    threads_per_block *= 2;
    cluster_size /= 2;
  }

  FLASHINFER_CHECK(oneshot || threads_per_block >= params.nranks,
                   "not oneshot, or threads_per_block < nranks");

  int block_size = threads_per_block;
  FLASHINFER_CHECK(block_size <= 1024 && cluster_size > 0,
                   "block_size > 1024 or cluster_size <= 0");

  int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;

  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attribute[2];
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  if (oneshot) {
    FLASHINFER_CUDA_CALL(
        (launch_oneshot_allgather<Pattern, T, NRanks>(params, cfg, begin_tokens, token_num_per_ranks)));
  } else {
    FLASHINFER_CUDA_CALL((launch_twoshot_allgather<Pattern, T, NRanks>(
        params, cfg, begin_tokens, token_num_per_ranks)));
  }

  return cudaSuccess;
}

template <typename T>
cudaError_t allgather_fusion_op(AllGatherFusionParams<T> const& params,
                                bool launch_with_pdl) {
  #define DISPATCH_PATTERN(T, NRanks)                                                          \
    switch (params.pattern) {                                                                  \
      case AllGatherFusionPattern::kAllGather:                                                 \
        return allgather_fusion_kernel_launcher<AllGatherFusionPattern::kAllGather, T, NRanks>(params, launch_with_pdl); \
      case AllGatherFusionPattern::kAllGatherfusedRMS:                                         \
        return allgather_fusion_kernel_launcher<AllGatherFusionPattern::kAllGatherfusedRMS, T, NRanks>(params, launch_with_pdl); \
      case AllGatherFusionPattern::kAllGatherfusedRMSFP8BlockWiseQuant:                        \
        return allgather_fusion_kernel_launcher<AllGatherFusionPattern::kAllGatherfusedRMSFP8BlockWiseQuant, T, NRanks>(params, launch_with_pdl); \
      default:                                                                                 \
        FLASHINFER_CHECK(false, "Unsupported allgather fusion pattern");                       \
    }

  switch (params.nranks) {
    case 2:
      DISPATCH_PATTERN(T, 2);
      break;
    case 4:
      DISPATCH_PATTERN(T, 4);
      break;
    case 8:
      DISPATCH_PATTERN(T, 8);
      break;
    case 16:
      DISPATCH_PATTERN(T, 16);
      break;
    default:
      FLASHINFER_ERROR(
          "allgather_fusion_kernel: unsupported ranks number! Supported ranks: 2, 4, 8, 16.");
  }
  #undef DISPATCH_PATTERN
}

}  // namespace trtllm_allgather_fusion

}  // namespace flashinfer

#endif  // FLASHINFER_TRTLLM_ALLGATHER_FUSION_CUH_
