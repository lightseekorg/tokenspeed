// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
// Copyright (c) 2026 LightSeek Foundation
//
// Vendored/adapted from flashinfer
//   include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh (Apache-2.0 per
//   upstream header): the MNNVL one-shot Lamport protocol (buffer-flags
//   rotation, multicast payload store, local-buffer polling, dirty-buffer
//   clearing) is ported here and grafted onto tokenspeed's vendored
//   AR-fusion epilogues (FusedOp) so that all tokenspeed fusion patterns —
//   including the Kimi-K3 custom patterns kARResidualAttnResCombine and
//   kAllReduceLatentNorm — run on the MNNVL kernel structure.
//
// Protocol summary (single stage, one-shot only):
//   * The workspace is one symmetric-memory allocation holding 3 equally
//     sized "lamport" buffers (triple buffering), pre-filled with the
//     32-bit sentinel 0x80000000 (fp32 -0.0).
//   * All mutable protocol state lives in a 9-word device uint32 array
//     ("buffer_flags"), so the kernel is stateless across launches and is
//     safe under CUDA-graph replay:
//       [0] current buffer index (0..2)
//       [1] dirty buffer index (buffer used two calls ago)
//       [2] bytes per lamport buffer (16B aligned)
//       [3] dirty stage count (layout compat with upstream; always <=1)
//       [4..7] bytes to clear in the dirty buffer (stage 0 used)
//       [8] arrival counter (one arrival per cluster per launch)
//   * Each launch: store the sanitized local shard ONCE through the NVLS
//     multicast VA (slot [token][rank][hidden] in the current buffer),
//     clear the dirty buffer back to the sentinel, poll the LOCAL replica
//     until all ranks' slots are sentinel-free, reduce deterministically
//     in rank order, run the fusion epilogue, then rotate the indices.
//   * Cross-rank safety of the rotation without any global barrier: a rank
//     can only reach call N+2 (which clears buffer (N+2)%3 == (N-1)%3)
//     after its call N+1 poll observed every peer's call-N+1 payload; a
//     peer's call-N+1 payload store strictly follows that peer's own clear
//     of (N-1)%3 (previous kernel on its stream), so the clear can never
//     race a peer that still polls the buffer.

#pragma once

#include "trtllm_allreduce_fusion.cuh"

namespace flashinfer {

namespace trtllm_mnnvl_allreduce_fusion {

using trtllm_allreduce_fusion::AllReduceFusionParams;
using trtllm_allreduce_fusion::AllReduceFusionPattern;
using trtllm_allreduce_fusion::allreduce_sum;
using trtllm_allreduce_fusion::FusedOp;
using trtllm_allreduce_fusion::get_sm_count;
using trtllm_allreduce_fusion::has_neg_zero;
using trtllm_allreduce_fusion::remove_neg_zero;

namespace details {
using trtllm_allreduce_fusion::details::kBytesPerAccess;
// Mirror of the vendored one-shot cap: the MNNVL path is a decode-latency
// kernel; larger payloads use the lamport/twoshot fallback.
static constexpr int kMnnvlOneShotMaxToken = 128;
static constexpr int kMaxClusterSize = 8;
}  // namespace details

// Peer-visible communication pointers for the MNNVL workspace. The kernel
// only needs the local unicast VA (poll target), the multicast VA (single
// payload store fans out to every rank) and the rotation-state array.
struct MnnvlCommArgs {
  void* multicast_ptr;
  void* buffer_ptr_local;
  uint32_t* buffer_flags;
};

// Device-side view over buffer_flags; see the protocol summary above.
// Ported from flashinfer's LamportFlags, specialized to a single stage.
struct MnnvlLamportFlags {
  __device__ __forceinline__ explicit MnnvlLamportFlags(uint32_t* buffer_flags)
      : flags_ptr(buffer_flags), access_ptr(&buffer_flags[8]) {
    uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
    cur_idx = flag.x;
    dirty_idx = flag.y;
    bytes_per_buffer = flag.z;
    dirty_bytes = buffer_flags[4];
  }

  __device__ __forceinline__ void* current_buf(void* base) const {
    return reinterpret_cast<char*>(base) + static_cast<size_t>(cur_idx) * bytes_per_buffer;
  }

  // Order the payload stores of the whole cluster before a single arrival
  // increment (one per cluster). Non-leader CTAs arrive without waiting so
  // they can proceed to the dirty-buffer clear.
  __device__ __forceinline__ void cta_arrive() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    __cluster_barrier_arrive();
    if (cluster.block_rank() == 0 && threadIdx.x < 32) {
      __cluster_barrier_wait();
      if (threadIdx.x == 0) {
#if (__CUDA_ARCH__ >= 1000)
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(access_ptr), "r"(1)
                     : "memory");
#else
        asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(access_ptr), "r"(1)
                     : "memory");
#endif
      }
    }
#endif
  }

  // Grid-strided refill of the dirty buffer with the Lamport sentinel.
  // Assumes a (gridDim.x, gridDim.y) grid of 1D CTAs; all threads take part.
  __device__ __forceinline__ void clear_dirty(void* base) {
    uint32_t global_cta = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t tid = global_cta * blockDim.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * gridDim.y * blockDim.x;
    float4* dirty = reinterpret_cast<float4*>(reinterpret_cast<char*>(base) +
                                              static_cast<size_t>(dirty_idx) * bytes_per_buffer);
    uint32_t num_packed = ceil_div<uint32_t>(dirty_bytes, sizeof(float4));
    float4 const sentinel = make_float4(-0.f, -0.f, -0.f, -0.f);
    for (uint32_t i = tid; i < num_packed; i += num_threads) {
      dirty[i] = sentinel;
    }
  }

  // Rotate: wait until every cluster arrived (i.e. every payload store of
  // this launch is ordered before the rotation), then publish the next
  // buffer index and record how many bytes this launch dirtied.
  __device__ __forceinline__ void wait_and_update(uint32_t bytes_written) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      // One arrival per cluster; the launch geometry is one cluster per
      // token, i.e. gridDim.x clusters in total.
      while (*reinterpret_cast<uint32_t volatile*>(access_ptr) < gridDim.x) {
      }
      uint4* flag = reinterpret_cast<uint4*>(flags_ptr);
      flag[0] = make_uint4((cur_idx + 1) % 3, cur_idx, bytes_per_buffer, 1u);
      flag[1] = make_uint4(bytes_written, 0u, 0u, 0u);
      *access_ptr = 0;
    }
  }

  uint32_t* flags_ptr;
  uint32_t* access_ptr;
  uint32_t cur_idx;
  uint32_t dirty_idx;
  uint32_t bytes_per_buffer;
  uint32_t dirty_bytes;
};

// One-shot MNNVL-structured allreduce with the vendored FusedOp epilogue.
//
// Geometry (mirrors flashinfer's oneshotAllreduceFusionKernel): one cluster
// per token, grid (num_tokens, cluster_size), cluster dim on y; the hidden
// dimension is partitioned EXACTLY across the cluster (host-side checked),
// so every thread is in-bounds and FusedOp's block/cluster reductions see
// full participation.
template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024)
    mnnvl_allreduce_fusion_kernel_oneshot(AllReduceFusionParams<T> params, MnnvlCommArgs comm) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int const token_id = blockIdx.x;
  int const num_tokens = gridDim.x;
  int const packed_idx = cluster.thread_rank();
  int const token_dim = params.hidden_dim;
  int const access_id = token_id * (token_dim / VEC_SIZE) + packed_idx;
  FusedOp<Pattern, T> fused_op(params, access_id, packed_idx);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  // Load upstream-produced inputs (gamma/residual/...) only after
  // gridDepSync, but before releasing PDL dependents that may reuse those
  // producer buffers (same contract as the vendored lamport kernel).
  fused_op.load_upstream_inputs();
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  MnnvlLamportFlags flags(comm.buffer_flags);
  T* stage_mcast = reinterpret_cast<T*>(flags.current_buf(comm.multicast_ptr));
  T* stage_local = reinterpret_cast<T*>(flags.current_buf(comm.buffer_ptr_local));

  // ==================== Broadcast the local shard =========================
  // One multicast STG.128 per thread replaces the NRanks unicast stores of
  // the lamport kernel. Slot layout: [token][rank][hidden].
  vec_t<T, VEC_SIZE> val;
  val.load(reinterpret_cast<T*>(params.allreduce_in) + static_cast<size_t>(access_id) * VEC_SIZE);
  remove_neg_zero<T, VEC_SIZE>(val);
  size_t const slot_base = (static_cast<size_t>(token_id) * NRanks + params.rank) *
                               static_cast<size_t>(token_dim) +
                           static_cast<size_t>(packed_idx) * VEC_SIZE;
  val.store(stage_mcast + slot_base);

  flags.cta_arrive();
  // ============== Clear the buffer dirtied two calls ago ==================
  flags.clear_dirty(comm.buffer_ptr_local);

  // ======================= Poll + reduce ==================================
  // Poll the LOCAL replica (the multicast store above also landed locally)
  // and reduce in fixed rank order: fully deterministic across ranks.
  vec_t<T, VEC_SIZE> vals[NRanks];
  bool done = false;
  while (!done) {
    done = true;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[r].load_global_volatile(stage_local +
                                   (static_cast<size_t>(token_id) * NRanks + r) *
                                       static_cast<size_t>(token_dim) +
                                   static_cast<size_t>(packed_idx) * VEC_SIZE);
      done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);
    }
  }
  vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);

  // ======================= Fusion epilogue ================================
  // Same FusedOp instantiation as the lamport kernel: all tokenspeed
  // patterns (incl. attn-res combine and latent-norm) work unmodified.
  fused_op(sum_val, token_id, /*skip_residual_add=*/false, /*skip_residual_store=*/false,
           /*skip_partial_store=*/true, access_id, access_id);

  flags.wait_and_update(static_cast<uint32_t>(static_cast<size_t>(num_tokens) * NRanks *
                                              static_cast<size_t>(token_dim) * sizeof(T)));
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

// Pick (block_size, cluster_size) such that block_size * cluster_size ==
// hidden_dim / VEC_SIZE exactly (FusedOp's block reductions require full
// participation and whole warps). Returns {-1, -1} when no exact partition
// exists; callers must then fall back to the lamport path.
inline std::tuple<int, int> mnnvl_oneshot_grid_config(int token_num, int hidden_dim,
                                                      int vec_size) {
  if (hidden_dim % vec_size != 0) {
    return {-1, -1};
  }
  int threads_needed = hidden_dim / vec_size;
  int cluster_size = 1;
  while (threads_needed / cluster_size > 1024) {
    cluster_size *= 2;
  }
  if (cluster_size > details::kMaxClusterSize || threads_needed % cluster_size != 0 ||
      (threads_needed / cluster_size) % 32 != 0) {
    return {-1, -1};
  }
  int sm_count = get_sm_count();
  // Widen the cluster (more, smaller CTAs per token) while the partition
  // stays exact/warp-aligned and the grid still fits the GPU.
  while (cluster_size < details::kMaxClusterSize) {
    int candidate = cluster_size * 2;
    int block = threads_needed / candidate;
    if (threads_needed % candidate != 0 || block % 32 != 0 || block < 128 ||
        token_num * candidate > sm_count) {
      break;
    }
    cluster_size = candidate;
  }
  return {threads_needed / cluster_size, cluster_size};
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t mnnvl_launch_oneshot(AllReduceFusionParams<T> const& params, MnnvlCommArgs const& comm,
                                 cudaLaunchConfig_t& cfg) {
  if (params.trigger_completion_at_end) {
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &cfg, mnnvl_allreduce_fusion_kernel_oneshot<Pattern, T, NRanks, Fp32Acc, true>, params,
        comm));
  } else {
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &cfg, mnnvl_allreduce_fusion_kernel_oneshot<Pattern, T, NRanks, Fp32Acc, false>, params,
        comm));
  }
  return cudaSuccess;
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks>
cudaError_t mnnvl_allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const& params,
                                                   MnnvlCommArgs const& comm, bool launch_with_pdl,
                                                   bool fp32_acc) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  FLASHINFER_CHECK(params.size % params.hidden_dim == 0, "params.size % params.hidden_dim != 0");
  int token_num = params.size / params.hidden_dim;
  FLASHINFER_CHECK(token_num <= details::kMnnvlOneShotMaxToken,
                   "mnnvl oneshot supports at most ", details::kMnnvlOneShotMaxToken, " tokens");
  static int SM = trtllm_allreduce_fusion::utils::getSMVersion();
  FLASHINFER_CHECK(SM >= 90, "mnnvl allreduce fusion requires SM90+");
  auto [block_size, cluster_size] =
      mnnvl_oneshot_grid_config(token_num, params.hidden_dim, VEC_SIZE);
  FLASHINFER_CHECK(block_size > 0, "mnnvl allreduce fusion: hidden_dim ", params.hidden_dim,
                   " has no exact cluster partition");

  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attrs[2];
  cfg.gridDim = dim3(token_num, cluster_size, 1);
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = cluster_size;
  attrs[1].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;

  if constexpr (!std::is_same_v<T, float>) {
    if (fp32_acc) {
      return mnnvl_launch_oneshot<Pattern, T, NRanks, true>(params, comm, cfg);
    }
  }
  return mnnvl_launch_oneshot<Pattern, T, NRanks, false>(params, comm, cfg);
}

template <typename T>
cudaError_t mnnvl_allreduce_fusion_op(AllReduceFusionParams<T> const& params,
                                      MnnvlCommArgs const& comm, bool launch_with_pdl,
                                      bool fp32_acc) {
#define MNNVL_DISPATCH_PATTERN(NRanks)                                                       \
  switch (params.pattern) {                                                                  \
    case AllReduceFusionPattern::kAllReduce:                                                 \
      return mnnvl_allreduce_fusion_kernel_launcher<AllReduceFusionPattern::kAllReduce, T,   \
                                                    NRanks>(params, comm, launch_with_pdl,   \
                                                            fp32_acc);                       \
    case AllReduceFusionPattern::kARResidualRMSNorm:                                         \
      return mnnvl_allreduce_fusion_kernel_launcher<AllReduceFusionPattern::kARResidualRMSNorm, \
                                                    T, NRanks>(params, comm, launch_with_pdl, \
                                                               fp32_acc);                    \
    case AllReduceFusionPattern::kARResidualAttnResCombine:                                  \
      return mnnvl_allreduce_fusion_kernel_launcher<                                         \
          AllReduceFusionPattern::kARResidualAttnResCombine, T, NRanks>(params, comm,        \
                                                                        launch_with_pdl,     \
                                                                        fp32_acc);           \
    case AllReduceFusionPattern::kAllReduceLatentNorm:                                       \
      return mnnvl_allreduce_fusion_kernel_launcher<                                         \
          AllReduceFusionPattern::kAllReduceLatentNorm, T, NRanks>(params, comm,             \
                                                                   launch_with_pdl,          \
                                                                   fp32_acc);                \
    default:                                                                                 \
      FLASHINFER_CHECK(false,                                                                \
                       "mnnvl allreduce fusion: unsupported pattern (supported: "            \
                       "kAllReduce, kARResidualRMSNorm, kARResidualAttnResCombine, "         \
                       "kAllReduceLatentNorm)");                                             \
  }

  switch (params.nranks) {
    case 2:
      MNNVL_DISPATCH_PATTERN(2);
      break;
    case 4:
      MNNVL_DISPATCH_PATTERN(4);
      break;
    case 8:
      MNNVL_DISPATCH_PATTERN(8);
      break;
    default:
      FLASHINFER_ERROR("mnnvl allreduce fusion: unsupported world size (supported: 2, 4, 8)");
  }
#undef MNNVL_DISPATCH_PATTERN
  return cudaErrorInvalidValue;
}

}  // namespace trtllm_mnnvl_allreduce_fusion

}  // namespace flashinfer
