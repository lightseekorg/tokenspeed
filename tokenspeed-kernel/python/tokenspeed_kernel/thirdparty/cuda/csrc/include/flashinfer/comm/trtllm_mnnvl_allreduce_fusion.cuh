/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
/*
 * Copyright (c) 2026 LightSeek Foundation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
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
// Protocol summary (one-shot uses one stage; two-shot uses two):
//   * The workspace is one symmetric-memory allocation holding 3 equally
//     sized "lamport" buffers (triple buffering), pre-filled with the
//     32-bit sentinel 0x80000000 (fp32 -0.0).
//   * All mutable protocol state lives in a 9-word device uint32 array
//     ("buffer_flags"), so the kernel is stateless across launches and is
//     safe under CUDA-graph replay:
//       [0] current buffer index (0..2)
//       [1] dirty buffer index (buffer used by the previous call)
//       [2] bytes per lamport buffer (32B aligned: stage 1 sits at half)
//       [3] dirty stage count of the launch that dirtied the buffer
//           (1 = one-shot, 2 = two-shot); consumers key their clear on it
//       [4] bytes dirtied in stage 0, [5] in stage 1 (two-shot only)
//       [8] arrival counter (one arrival per cluster per launch)
//   * Each launch: store the sanitized local shard ONCE through the NVLS
//     multicast VA (slot [token][rank][hidden] in the current buffer),
//     clear the dirty buffer back to the sentinel, poll the LOCAL replica
//     until all ranks' slots are sentinel-free, reduce deterministically
//     in rank order, run the fusion epilogue, then rotate the indices.
//   * Rotation: call N writes buffer N%3 and clears buffer (N-1)%3 -- the
//     buffer used by the immediately preceding call, not two calls back.
//   * Cross-rank safety without a global barrier: completing call N implies a
//     rank observed every peer's call-N payload, so peer writes to a buffer
//     always precede that rank's clear of it one call later, and its clear at
//     call N precedes peer writes at call N+2.
//     For ONE-SHOT the observation is direct: every rank polls all NRanks
//     slots. For TWO-SHOT it is transitive: an owner polls only the A slots of
//     tokens it owns, but for every token it does not own it polls B[t], and
//     B[t] can only have been published after that token's owner observed all
//     NRanks A slots. The closure still covers the whole A region; B is
//     covered because each B[t] a rank reads is either its own store or one it
//     polled.
//     That transitivity is what makes the two-shot rotation safe, and it is
//     fragile: it breaks if a non-owner is ever allowed to skip the B poll
//     (e.g. an "output not needed here" fast path), or if an owner publishes
//     B before all NRanks A slots have landed.

#pragma once

#include <type_traits>

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
// Two-shot serves the prefill-sized calls one-shot cannot; bounded by the
// workspace allocation (two stages of ceil(T/NRanks)*NRanks vectors per slot).
static constexpr int kMnnvlTwoShotMaxToken = 2048;
static constexpr int kMaxClusterSize = 8;
}  // namespace details

// Peer-visible communication pointers for the MNNVL workspace. The kernel
// only needs the local unicast VA (poll target), the multicast VA (single
// payload store fans out to every rank) and the rotation-state array.
struct MnnvlCommArgs {
  void* multicast_ptr;
  void* buffer_ptr_local;
  uint32_t* buffer_flags;
  // Peer unicast bases, indexed by rank. One-shot does not use these (it
  // broadcasts once through the multicast VA); two-shot scatters phase A to
  // the owning rank only, which is the whole point of the extra hop.
  void* const* peer_ptrs;
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
    // Cache, do not re-read: clear_dirty runs AFTER cta_arrive in the one-shot
    // kernel, so a live load could observe this launch's own wait_and_update
    // and mistake the previous launch's stage count for its own.
    dirty_stages = flag.w;
    dirty_bytes = buffer_flags[4];
    dirty_bytes_stage1 = buffer_flags[5];
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

  // --- two-stage helpers -------------------------------------------------
  // Two-shot needs SCATTER and BROADCAST to live in disjoint sub-regions with
  // independent dirty extents (upstream: MNNVLTwoShotStage). One-shot keeps
  // using the whole buffer via the single-stage calls above.
  // Stage stride is a property of the BUFFER, not of a launch: a dirty extent
  // recorded by one launch is cleared by the next, which may carry a different
  // token count. Deriving the offset from the caller's token count (as an
  // earlier version did) made the consumer clear the wrong region.
  __device__ __forceinline__ uint32_t stage_stride() const { return bytes_per_buffer / 2u; }

  __device__ __forceinline__ void* stage_buf(void* base, int stage) const {
    return reinterpret_cast<char*>(current_buf(base)) +
           static_cast<size_t>(stage) * stage_stride();
  }

  __device__ __forceinline__ void clear_dirty_stage(void* base, int stage) {
    uint32_t global_cta = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t tid = global_cta * blockDim.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * gridDim.y * blockDim.x;
    // A single-stage predecessor (one-shot) recorded one extent spanning the
    // whole buffer: stage 0 clears it, stage 1 has nothing to do.
    uint32_t bytes;
    size_t offset;
    if (dirty_stages > 1u) {
      bytes = (stage == 0) ? dirty_bytes : dirty_bytes_stage1;
      offset = static_cast<size_t>(stage) * stage_stride();
    } else {
      bytes = (stage == 0) ? dirty_bytes : 0u;
      offset = 0;
    }
    char* dirty_base = reinterpret_cast<char*>(base) +
                       static_cast<size_t>(dirty_idx) * bytes_per_buffer + offset;
    float4* dirty = reinterpret_cast<float4*>(dirty_base);
    uint32_t num_packed = ceil_div<uint32_t>(bytes, sizeof(float4));
    float4 const sentinel = make_float4(-0.f, -0.f, -0.f, -0.f);
    for (uint32_t i = tid; i < num_packed; i += num_threads) {
      dirty[i] = sentinel;
    }
  }

  __device__ __forceinline__ void wait_and_update2(uint32_t scatter_bytes,
                                                   uint32_t broadcast_bytes) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<uint32_t volatile*>(access_ptr) < gridDim.x) {
      }
      uint4* flag = reinterpret_cast<uint4*>(flags_ptr);
      flag[0] = make_uint4((cur_idx + 1) % 3, cur_idx, bytes_per_buffer, 2u);
      flag[1] = make_uint4(scatter_bytes, broadcast_bytes, 0u, 0u);
      *access_ptr = 0;
    }
  }

  // Grid-strided refill of the dirty buffer with the Lamport sentinel.
  // Assumes a (gridDim.x, gridDim.y) grid of 1D CTAs; all threads take part.
  __device__ __forceinline__ void clear_dirty(void* base) {
    uint32_t global_cta = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t tid = global_cta * blockDim.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * gridDim.y * blockDim.x;
    float4* dirty = reinterpret_cast<float4*>(reinterpret_cast<char*>(base) +
                                              static_cast<size_t>(dirty_idx) * bytes_per_buffer);
    // The previous launch may have been two-shot, which records a per-stage
    // extent (flags[4], flags[5]) at a stage stride this launch does not know.
    // Clearing only flags[4] would leave that launch's second stage dirty and
    // corrupt this one. Clear the whole buffer whenever a staged launch
    // preceded us -- flags[3] carries its stage count.
    // A two-stage predecessor dirtied two DISJOINT ranges: [0, dirty_bytes)
    // and [stride, stride + dirty_bytes_stage1). Clearing their span instead
    // would memset the provably-clean gap between them -- measured at 27.5 MB
    // of waste after a 129-token prefill, making the next decode 2.6x slower.
    // Walk the two ranges as one index space so the grid-stride loop stays
    // balanced.
    uint32_t const n0 = ceil_div<uint32_t>(dirty_bytes, sizeof(float4));
    uint32_t const n1 =
        (dirty_stages > 1u) ? ceil_div<uint32_t>(dirty_bytes_stage1, sizeof(float4)) : 0u;
    uint32_t const stride_packed = stage_stride() / sizeof(float4);
    float4 const sentinel = make_float4(-0.f, -0.f, -0.f, -0.f);
    for (uint32_t i = tid; i < n0 + n1; i += num_threads) {
      uint32_t const slot = (i < n0) ? i : (stride_packed + (i - n0));
      dirty[slot] = sentinel;
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
  uint32_t dirty_bytes_stage1;
  uint32_t dirty_stages;
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

// Two-shot MNNVL allreduce with the vendored FusedOp epilogue.
//
// The communication skeleton follows flashinfer's twoshotAllreduceKernel
// (trtllm_mnnvl_allreduce.cuh): round-robin ownership, shard-local staging,
// chunked reduction, two independently-rotated Lamport stages. Only the
// epilogue differs -- upstream emits a plain all-reduce and runs RMSNorm as a
// separate kernel, whereas tokenspeed needs FusedOp so that the Kimi-K3
// patterns (kARResidualAttnResCombine, kAllReduceLatentNorm) fuse into the
// same launch. Deviating from that skeleton has already cost us once: an
// earlier version broadcast phase A through the multicast VA, which made every
// rank receive the whole staging region and moved MORE bytes than one-shot.
//
// Three properties are load-bearing; changing any of them breaks correctness
// or the memory budget rather than just performance:
//   * ownership is token % NRanks -- balanced, and it is what makes the
//     SCATTER region shard-local (ceil(tokens/NRanks) slots per rank, not
//     tokens), which is the difference between 2 stages and NRanks+1 lanes;
//   * the reduced value is scrubbed before the broadcast store, because a
//     legitimate reduction can round to the -0.0 sentinel (all-0x8001 input
//     under FTZ does exactly this) and a peer would then poll forever;
//   * every cluster arrives exactly once regardless of owner/non-owner role,
//     since ownership depends only on blockIdx.x and is cluster-uniform.
//
// Token-sharded round-robin: rank r owns tokens t where t % NRanks == r. Every rank still
// launches one cluster per token (same geometry as one-shot) because every
// rank needs every token's epilogue output locally.
//
//   phase A: all ranks multicast-store their input slot [token][rank][hidden]
//            (same staging layout as one-shot);
//   reduce : ONLY the owner cluster polls the NRanks slots of its token and
//            reduces -- reduce work per rank drops NRanks-fold vs one-shot;
//   phase B: the owner multicast-stores the REDUCED (pre-epilogue) vector to
//            the B region [token][hidden] appended after the A region;
//   land   : non-owner clusters poll their token's B slot;
//   epilog : every rank runs the identical FusedOp on identical sum bits --
//            bitwise-deterministic, so norm/residual outputs match everywhere
//            without shipping two result arrays.
//
// Buffer per rotation slot holds whichever layout is larger: two-shot's two
// stages of ceil(tokens/NRanks)*NRanks vectors, or one-shot's tokens*NRanks.
// Stage 1 begins at bytes_per_buffer/2 -- a fixed, launch-independent stride.
template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024)
    mnnvl_allreduce_fusion_kernel_twoshot(AllReduceFusionParams<T> params, MnnvlCommArgs comm) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int const token_id = blockIdx.x;
  int const num_tokens = gridDim.x;
  int const packed_idx = cluster.thread_rank();
  int const token_dim = params.hidden_dim;
  int const access_id = token_id * (token_dim / VEC_SIZE) + packed_idx;

  // Round-robin ownership (upstream's destRank/destTokenOffset): balanced by
  // construction, and the owner stores its shard compactly at local_token so
  // the SCATTER region only ever holds this rank's share of the tokens --
  // NRanks tokens' worth, not num_tokens'.
  int const owner = token_id % NRanks;
  int const local_token = token_id / NRanks;
  int const tokens_per_rank = ceil_div<int>(num_tokens, NRanks);
  bool const is_owner = (owner == params.rank);
  FusedOp<Pattern, T> fused_op(params, access_id, packed_idx);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  fused_op.load_upstream_inputs();
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  MnnvlLamportFlags flags(comm.buffer_flags);
  // How many bytes THIS launch dirties in each stage. Offsets come from the
  // buffer-wide stage_stride(); these values only ever describe extents, so a
  // later launch with a different token count still clears the right region.
  uint32_t const scatter_bytes =
      static_cast<uint32_t>(static_cast<size_t>(tokens_per_rank) * NRanks *
                            static_cast<size_t>(token_dim) * sizeof(T));
  T* scatter_dest =
      reinterpret_cast<T*>(flags.stage_buf(comm.peer_ptrs[owner], 0));
  T* scatter_local =
      reinterpret_cast<T*>(flags.stage_buf(comm.buffer_ptr_local, 0));
  T* bcast_write = reinterpret_cast<T*>(flags.stage_buf(comm.multicast_ptr, 1));
  T* bcast_read = reinterpret_cast<T*>(flags.stage_buf(comm.buffer_ptr_local, 1));

  // ============ SCATTER: unicast this rank's slot to the owner ============
  vec_t<T, VEC_SIZE> val;
  val.load(reinterpret_cast<T*>(params.allreduce_in) + static_cast<size_t>(access_id) * VEC_SIZE);
  remove_neg_zero<T, VEC_SIZE>(val);
  size_t const scatter_slot =
      (static_cast<size_t>(local_token) * NRanks + params.rank) * static_cast<size_t>(token_dim) +
      static_cast<size_t>(packed_idx) * VEC_SIZE;
  val.store(scatter_dest + scatter_slot);

  flags.clear_dirty_stage(comm.buffer_ptr_local, 0);

  // ============ REDUCE (owner only) + BROADCAST the sum =================
  vec_t<T, VEC_SIZE> sum_val;
  if (is_owner) {
    // Chunked to bound live registers: a full vals[NRanks] array spills
    // inside the poll loop at NRanks=16. The accumulator stays in fp32 ACROSS
    // chunks and is rounded exactly once (upstream reduceLamportRanksChunked):
    // rounding at each chunk boundary cost 33% relative error on inputs that
    // cancel across the boundary, and made world 16 disagree bitwise with
    // one-shot on the same input.
    constexpr int kRankChunk = NRanks < 8 ? NRanks : 8;
    static_assert(NRanks % kRankChunk == 0,
                  "chunked reduction requires kRankChunk to divide NRanks");
    vec_t<T, VEC_SIZE> chunk[kRankChunk];
    // Accumulator width follows Fp32Acc so two-shot reduces in exactly the
    // same arithmetic as one-shot (allreduce_sum): otherwise a batch crossing
    // switching strategies at the traffic-based dispatch boundary would
    // change the answer's bits.
    using AccT = std::conditional_t<Fp32Acc, float, T>;
    AccT accum[VEC_SIZE];
#pragma unroll
    for (int e = 0; e < VEC_SIZE; ++e) {
      accum[e] = static_cast<AccT>(0.f);
    }
#pragma unroll 1
    for (int base = 0; base < NRanks; base += kRankChunk) {
      bool done = false;
      while (!done) {
        done = true;
#pragma unroll
        for (int j = 0; j < kRankChunk; ++j) {
          chunk[j].load_global_volatile(
              scatter_local +
              (static_cast<size_t>(local_token) * NRanks + base + j) *
                  static_cast<size_t>(token_dim) +
              static_cast<size_t>(packed_idx) * VEC_SIZE);
          done &= !has_neg_zero<T, VEC_SIZE>(chunk[j]);
        }
      }
#pragma unroll
      for (int j = 0; j < kRankChunk; ++j) {
#pragma unroll
        for (int e = 0; e < VEC_SIZE; ++e) {
          accum[e] = static_cast<AccT>(static_cast<float>(accum[e]) +
                                       static_cast<float>(chunk[j][e]));
        }
      }
    }
#pragma unroll
    for (int e = 0; e < VEC_SIZE; ++e) {
      sum_val[e] = static_cast<T>(accum[e]);
    }
    // A reduced value can round to the sentinel; scrub before peers poll it.
    remove_neg_zero<T, VEC_SIZE>(sum_val);
    sum_val.store(bcast_write + static_cast<size_t>(access_id) * VEC_SIZE);
  }

  flags.clear_dirty_stage(comm.buffer_ptr_local, 1);
  flags.cta_arrive();

  if (!is_owner) {
    bool done = false;
    while (!done) {
      sum_val.load_global_volatile(bcast_read + static_cast<size_t>(access_id) * VEC_SIZE);
      done = !has_neg_zero<T, VEC_SIZE>(sum_val);
    }
  }

  // ============ identical epilogue on identical bits ======================
  fused_op(sum_val, token_id, /*skip_residual_add=*/false, /*skip_residual_store=*/false,
           /*skip_partial_store=*/true, access_id, access_id);

  flags.wait_and_update2(
      scatter_bytes,
      static_cast<uint32_t>(static_cast<size_t>(num_tokens) * token_dim * sizeof(T)));
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

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t mnnvl_launch_twoshot(AllReduceFusionParams<T> const& params, MnnvlCommArgs const& comm,
                                 cudaLaunchConfig_t& cfg) {
  if (params.trigger_completion_at_end) {
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &cfg, mnnvl_allreduce_fusion_kernel_twoshot<Pattern, T, NRanks, Fp32Acc, true>, params,
        comm));
  } else {
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &cfg, mnnvl_allreduce_fusion_kernel_twoshot<Pattern, T, NRanks, Fp32Acc, false>, params,
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
  bool const use_twoshot = !params.use_oneshot;
  FLASHINFER_CHECK(token_num <= details::kMnnvlTwoShotMaxToken,
                   "mnnvl allreduce supports at most ", details::kMnnvlTwoShotMaxToken, " tokens");
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

  if (use_twoshot) {
    if constexpr (!std::is_same_v<T, float>) {
      if (fp32_acc) {
        return mnnvl_launch_twoshot<Pattern, T, NRanks, true>(params, comm, cfg);
      }
    }
    return mnnvl_launch_twoshot<Pattern, T, NRanks, false>(params, comm, cfg);
  }
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
    case 16:
      MNNVL_DISPATCH_PATTERN(16);
      break;
    default:
      FLASHINFER_ERROR(
          "mnnvl allreduce fusion: unsupported world size (supported: 2, 4, 8, 16)");
  }
#undef MNNVL_DISPATCH_PATTERN
  return cudaErrorInvalidValue;
}

}  // namespace trtllm_mnnvl_allreduce_fusion

}  // namespace flashinfer
