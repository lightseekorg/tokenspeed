/*
 * Copyright (c) 2024-2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * ------------------------------------------------------------------------
 * Portions of this file (indexerTopK histogram-step + final-candidate selection
 * algorithm) are derived from NVIDIA TensorRT-LLM:
 *   Copyright (c) 2019-2026 NVIDIA CORPORATION
 *   Copyright (c) 2021 NAVER Corp. (CLOVA)
 * licensed under Apache License 2.0.  Original source:
 *   tensorrt_llm/cpp/tensorrt_llm/kernels/indexerTopK.cu
 *
 * Function provenance map (NVIDIA TensorRT-LLM, indexerTopK.cu line ranges):
 *   extractBinIdx                indexerTopK.cu:43-71
 *   isPartialMatch               indexerTopK.cu:73-83
 *   vectorized_process           indexerTopK.cu:99-162
 *   processHistogramStep         indexerTopK.cu:164-362  (multipleBlocksPerRow /
 *                                                        mergeBlocks branches dropped;
 *                                                        stride1!=1 branch dropped)
 *   topKPerRowJob                indexerTopK.cu:365-595  (insertion-sort branch only;
 *                                                        radix-sort branch dropped;
 *                                                        multipleBlocksPerRow /
 *                                                        mergeBlocks dropped;
 *                                                        ascending-by-index sort fused)
 *
 * v2.0_indexer_topk_qo_outermost fork additions (FlashInfer-original):
 *   SparseTopKTransposeKernel        — input layout adapter (reused from v1.6.5)
 *   SparseTopKIdentityFillKernel     — trivial-row fast path  (reused from v1.6.5)
 *   IndexerTopKWithSortKernel        — wraps topKPerRowJob and fuses an
 *                                      ascending-by-index cub::BlockRadixSort over
 *                                      smemOutput before the gmem write, so the
 *                                      output contract matches the v1.6.5 fork
 *                                      (qo_outermost layout, asc-by-index)
 *   SparseTopKSelect                 — top-level dispatcher
 * ------------------------------------------------------------------------
 */
#ifndef FLASHINFER_SPARSE_TOPK_SELECT_CUH_
#define FLASHINFER_SPARSE_TOPK_SELECT_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cub/cub.cuh>
#include <type_traits>

namespace flashinfer {
namespace sparse_topk {

enum class SparseTopKInputLayout : uint32_t {
  kHKT = 0,  // input is (num_qo_heads, max_k_tiles, total_qo_len)
  kTHK = 1,  // input is (total_qo_len, num_qo_heads, max_k_tiles)
};

__device__ __forceinline__ void SparseTopKWaitOnDependentGrids(uint32_t use_pdl) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (use_pdl) {
    cudaGridDependencySynchronize();
  }
#else
  (void)use_pdl;
#endif
}

__device__ __forceinline__ void SparseTopKLaunchDependentGrids(uint32_t use_pdl) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (use_pdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#else
  (void)use_pdl;
#endif
}

__device__ __forceinline__ uint32_t LoadRowNumValidPages(
    const int32_t* __restrict__ num_valid_pages_per_token,
    uint32_t t, uint32_t scalar_num_valid_pages) {
  if (num_valid_pages_per_token == nullptr) return scalar_num_valid_pages;
  const int32_t row_nvp = __ldg(num_valid_pages_per_token + t);
  if (row_nvp <= 0) return 0u;
  const uint32_t row_nvp_u = static_cast<uint32_t>(row_nvp);
  return row_nvp_u < scalar_num_valid_pages ? row_nvp_u : scalar_num_valid_pages;
}

__device__ __forceinline__ bool IsForcedBlock(uint32_t k, uint32_t force_begin,
                                              uint32_t force_end_start,
                                              uint32_t row_num_valid_pages) {
  return (k < force_begin) || (k >= force_end_start && k < row_num_valid_pages);
}

__device__ __forceinline__ int32_t GatherBlockTableValue(
    const int32_t* __restrict__ block_table_row, uint64_t block_table_stride_k,
    int32_t logical_idx) {
  if (logical_idx < 0) return -1;
  if (block_table_row == nullptr) return logical_idx;
  return __ldg(block_table_row + static_cast<size_t>(logical_idx) * block_table_stride_k);
}

// =============================================================================
// v2.0 design overview
// =============================================================================
//
// Replaces the v1.6.5_2cta_per_row_qo_outermost pipeline (transpose +
// Filtered/Multi-CTA + Merge + SortByIndex = 3-4 kernels) with the
// indexerTopK algorithm (transpose + indexerTopK-with-fused-sort = 2 kernels).
//
// Pipeline:
//
//   [in (Hq, K, qo) row-contig fp32]
//          |
//          v  SparseTopKTransposeKernel  (per-head 32x32 SMEM tile transpose)
//          |
//   [transpose_buf (Hq, qo, K) row-contig fp32 — workspace]
//          |
//          v  IndexerTopKWithSortKernel<MAX_TOPK>(grid=num_rows, block=512)
//          |    Stage 0: 10-bit fp16 hist + cub::BlockScan + threshold + classify
//          |    Stage 1/2/3: 10+10+10 bit fp32 (only if Stage 0 didn't finish)
//          |    Final pass: bounded rank-select / warp merge over the staged bin
//          |    Warp-only bitonic sort on smemOutput (ascending by index)
//          |    NEW: write gmem with qo_outermost offset
//          |
//   [out (qo, num_qo_heads, topk) int32, ascending-by-index]
//     If block_table is supplied, selected logical indices are sorted first,
//     then gathered through block_table[t][h][idx] before the write.
//
// Trivial path (max_k_tiles <= topk) routes to SparseTopKIdentityFillKernel,
// same as the v1.6.5 fork — preserves the seamless drop-in contract.
//
// Workspace = transpose_buf only (no descriptors / merge_in / unsorted_tmp /
// counter slabs needed; v1.6.5's workspace formula is preserved as a
// conservative upper bound so callers that allocated v1.6.5-sized workspaces
// still work without any change).

// =============================================================================
// indexerTopK helpers — vendored from TRT-LLM indexerTopK.cu
// =============================================================================

template <int step>
__device__ __forceinline__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 6;          // 10-bit fp16 (was >> 5 for 11-bit)
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 22;                // high 10 bits (was >> 21)
    } else if constexpr (step == 2) {
      return (bits >> 12) & 0x3ff;      // mid 10 bits (was (>> 10) & 0x7ff)
    } else if constexpr (step == 3) {
      return (bits >> 2) & 0x3ff;       // next-low 10 bits, drops bits 0-1
    }
  }
}

template <int shift>
__device__ __forceinline__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

// Map Func over the input data, using vectorized float4 loads when possible.
// stride1==1 only (we always feed transpose_buf, which is row-contig).
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, T const* in, idxT len,
                                   Func f) {
  constexpr int WARP_SIZE = 32;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                       ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
                          sizeof(T))
                       : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    idxT const len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      idxT const real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(WARP_SIZE >= items_per_scalar);
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    idxT const remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

// =============================================================================
// processHistogramStep — vendored from indexerTopK.cu (single-block-per-row only)
// =============================================================================
//
// Differences vs upstream:
//   - Drops `multipleBlocksPerRow` and `mergeBlocks` template branches: v2.0
//     always processes one row per CTA, no inter-CTA merge.
//   - Drops `stride1` runtime branch: v2.0 always feeds transpose_buf, so input
//     is always row-contiguous (stride1 == 1) and we always go through
//     vectorized_process.
//   - `indices` arg removed (was only used in mergeBlocks branch).
template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          typename SmemFinalType, typename SmemOutputType>
__device__ bool processHistogramStep(float const* logits, int rowEnd, uint32_t& logitPattern,
                                     int& thresholdBinIdx, SmemOutputType& smemOutput,
                                     int* smemThresholdBinIdx, int* smemFinalDstIdx,
                                     int* smemFinalBinSize, int* smemFoundTopKValues,
                                     SmemFinalType& smemFinal, int rowStart, int topK,
                                     uint32_t force_begin, uint32_t force_end_start,
                                     uint32_t row_num_valid_pages) {
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }
  __syncthreads();

  // Update pattern.
  const int prevThresholdBinIdx = thresholdBinIdx;
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 22 : 12;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x3ff) << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x3ff) << patternShift;
  }

  auto effective_logit = [&](float logit, int idx) {
    const uint32_t k = static_cast<uint32_t>(rowStart + idx);
    return IsForcedBlock(k, force_begin, force_end_start, row_num_valid_pages) ? FLT_MAX : logit;
  };

  auto matchesCurrentRefinement = [&](float logit) {
    if constexpr (step == 1) {
      return extractBinIdx<0>(logit) == static_cast<uint32_t>(prevThresholdBinIdx);
    } else {
      return isPartialMatch<patternShift>(logit, logitPattern);
    }
  };

  auto distributeToBins = [&](float logit, int idx = 0) {
    logit = effective_logit(logit, idx);
    if (matchesCurrentRefinement(logit)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  // Distribute the elements to the histogram bins.
  vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart,
                     distributeToBins);
  __syncthreads();

  // Reads the value of the starting position in smemOutput accounting.
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;
    __syncthreads();

    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemFinal.histo.data[idx + 1];
      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    if (__syncthreads_or(foundThreshold)) {
      break;
    }
    lastValue = totalSum;
  }
  __syncthreads();

  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    logit = effective_logit(logit, idx);
    if (matchesCurrentRefinement(logit)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        // The element is part of the top-k selection.
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);
        smemOutput[dstIdx] = idx;
      }
      if constexpr (step < 3) {
        // Only fill final items for sorting if the threshold bin fits.
        if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          smemFinal.items.indices[dstIdx] = idx;
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // Elements in the threshold bin share the same 32 bits at step 3.
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            smemOutput[dstIdx] = idx;
          }
        }
      }
    }
  };

  vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart,
                     processBins);
  __syncthreads();

  // Continue to next step if the threshold bin overflows the staging buffer.
  return smemFinalBinSize[0] > kNumFinalItems;
}

// =============================================================================
// Transpose kernel — reused verbatim from v1.6.5_2cta_per_row_qo_outermost
// =============================================================================
//
// Per-head 32x32 SMEM tile transpose.  Source layout has qo as the innermost
// dim (stride 1, K-stride = qo); dest swaps so K becomes innermost (stride 1,
// qo-stride = K).  Heads dim stays outermost in both.
//
// Block: (32, 8) = 256 threads.  Each thread does 4 elements per pass.
// Tile : 32x32 floats with [32][33] padding to remove SMEM bank conflicts.
// Grid : (ceil(qo / 32), ceil(K / 32), num_qo_heads)
//
// Fallback used when XorF4 preconditions (qo % TILE == 0 && K % TILE == 0 &&
// qo % 4 == 0 && K % 4 == 0) are not met.  All decode/spec workloads with
// qo, K divisible by 32 take the faster TransposeXorF4 path.
constexpr int kTransposeTile = 32;
constexpr int kTransposeBlockRows = 8;

__global__ void __launch_bounds__(kTransposeTile* kTransposeBlockRows)
    SparseTopKTransposeKernel(const float* __restrict__ in, float* __restrict__ out, uint32_t K,
                              uint32_t qo, uint32_t use_pdl) {
  __shared__ float tile[kTransposeTile][kTransposeTile + 1];

  const uint32_t q_base = blockIdx.x * kTransposeTile;
  const uint32_t k_base = blockIdx.y * kTransposeTile;
  const uint32_t head = blockIdx.z;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;

  const float* in_head = in + static_cast<size_t>(head) * K * qo;
  float* out_head = out + static_cast<size_t>(head) * qo * K;

  SparseTopKWaitOnDependentGrids(use_pdl);

  const uint32_t q_load = q_base + tx;
#pragma unroll
  for (int dk = 0; dk < kTransposeTile; dk += kTransposeBlockRows) {
    const uint32_t k_load = k_base + ty + dk;
    if (q_load < qo && k_load < K) {
      tile[ty + dk][tx] = in_head[static_cast<size_t>(k_load) * qo + q_load];
    }
  }
  __syncthreads();

  const uint32_t k_store = k_base + tx;
#pragma unroll
  for (int dq = 0; dq < kTransposeTile; dq += kTransposeBlockRows) {
    const uint32_t q_store = q_base + ty + dq;
    if (q_store < qo && k_store < K) {
      out_head[static_cast<size_t>(q_store) * K + k_store] = tile[tx][ty + dq];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    SparseTopKLaunchDependentGrids(use_pdl);
  }
}

// =============================================================================
// SparseTopKTransposeXorF4Kernel — FAST PATH transpose
// =============================================================================
//
// Vendored from paged_attention_doc/sparse_topk_experiments/transpose_bench/
// benchmark_swizzle.py — XOR-swizzle + float4 GMEM I/O, no SMEM padding.
//
// Theory:
//   Standard +1-padding k32: scalar float GMEM loads/stores.
//   This kernel: float4 GMEM loads (4 Q values per thread), scalar SMEM
//   writes (XOR-swizzled), scalar SMEM reads (transposed, conflict-free by
//   XOR), float4 GMEM stores (4 K values per thread).
//
//   The XOR swizzle: value[k][q] stored at S[k * TILE + (q ^ k)].
//     WRITE bank (q^k): for fixed k, varying q → all different banks → no conflict.
//     READ bank ((q^k) for varying k, fixed q): all different banks → no conflict.
//
// Preconditions (caller must ensure):
//   - TILE divides qo and K
//   - qo and K divisible by 4 (for float4 alignment)
//   - input/output base pointers 16-B aligned (cudaMalloc guarantees this for
//     the workspace pointer; transpose_buf offset is at the start so aligned)
//
// Grid: (qo / TILE, K / TILE, num_qo_heads), Block: TILE * TILE / 4 threads.
template <int TILE>
__global__ void __launch_bounds__(TILE * TILE / 4) SparseTopKTransposeXorF4Kernel(
    const float* __restrict__ in, float* __restrict__ out, uint32_t K, uint32_t qo,
    uint32_t use_pdl) {
  constexpr int N_THR = TILE / 4;  // threads along N (qo) direction

  __shared__ float S[TILE * TILE];

  const uint32_t qb = blockIdx.x * TILE;
  const uint32_t kb = blockIdx.y * TILE;
  const uint32_t h  = blockIdx.z;
  const uint32_t tx = threadIdx.x;

  const int tm = tx / N_THR;        // K-tile row index (0..TILE-1)
  const int tn = tx % N_THR;        // Q-tile col group (0..N_THR-1, covers 4 Q each)

  const float* in_head  = in  + static_cast<size_t>(h) * K * qo;
  float*       out_head = out + static_cast<size_t>(h) * qo * K;

  SparseTopKWaitOnDependentGrids(use_pdl);

  // ---- LOAD: float4 GMEM → XOR-swizzled SMEM ---------------------------
  {
    const float4* in4 = reinterpret_cast<const float4*>(
        in_head + static_cast<size_t>(kb + tm) * qo + qb + tn * 4);
    float4 v = *in4;

    const int base = tm * TILE;
    S[base + ((tn * 4 + 0) ^ tm)] = v.x;
    S[base + ((tn * 4 + 1) ^ tm)] = v.y;
    S[base + ((tn * 4 + 2) ^ tm)] = v.z;
    S[base + ((tn * 4 + 3) ^ tm)] = v.w;
  }
  __syncthreads();

  // ---- STORE: scalar swizzled SMEM reads → float4 GMEM -----------------
  {
    const int q_out = qb + tm;
    const int k0    = kb + tn * 4;

    float v0 = S[(tn * 4 + 0) * TILE + (tm ^ (tn * 4 + 0))];
    float v1 = S[(tn * 4 + 1) * TILE + (tm ^ (tn * 4 + 1))];
    float v2 = S[(tn * 4 + 2) * TILE + (tm ^ (tn * 4 + 2))];
    float v3 = S[(tn * 4 + 3) * TILE + (tm ^ (tn * 4 + 3))];

    float4 out_v = {v0, v1, v2, v3};
    float4* out4 = reinterpret_cast<float4*>(
        out_head + static_cast<size_t>(q_out) * K + k0);
    *out4 = out_v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTopKLaunchDependentGrids(use_pdl);
  }
}

// =============================================================================
// Identity-fill kernel — reused verbatim from v1.6.5_2cta_per_row_qo_outermost
// =============================================================================
// Used when max_k_tiles <= topk (trivial row): output = [0, 1, ..., K-1, -1, ..., -1].
// Skips both transpose + indexerTopK to save VRAM and a kernel launch.
//
// v2.5_oob_clamp_in_kernel: also handles num_valid_pages clamp — indices in
// [num_valid_pages, max_k_tiles) become -1 (already sorted to tail by
// construction since the trivial fill is ascending). Pass num_valid_pages =
// max_k_tiles (or any value >= max_k_tiles) to disable clamping.
__global__ void SparseTopKIdentityFillKernel(int32_t* __restrict__ out,
                                             const int32_t* __restrict__ block_table,
                                             uint64_t out_stride_t,
                                             uint64_t out_stride_h,
                                             uint64_t out_stride_k,
                                             uint64_t block_table_stride_t,
                                             uint64_t block_table_stride_h,
                                             uint64_t block_table_stride_k,
                                             uint32_t total_qo_len,
                                             uint32_t num_qo_heads, uint32_t max_k_tiles,
                                             uint32_t topk, uint32_t num_valid_pages,
                                             const int32_t* __restrict__ num_valid_pages_per_token,
                                             uint32_t use_pdl) {
  // grid = (num_rows = total_qo_len * num_qo_heads,), block = (min(topk, 256),)
  // bid encodes (qo_head_idx, t) with t innermost: bid = qo_head_idx * total_qo_len + t
  // out logical layout: (total_qo_len, num_qo_heads, topk)
  //   element (t, qo_head_idx, k) at offset = t*out_stride_t + h*out_stride_h + k*out_stride_k
  const uint32_t bid = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t qo_head_idx = bid / total_qo_len;
  const uint32_t t = bid % total_qo_len;
  int32_t* row =
      out + static_cast<size_t>(t) * out_stride_t +
      static_cast<size_t>(qo_head_idx) * out_stride_h;
  const int32_t* block_table_row =
      block_table == nullptr
          ? nullptr
          : block_table + static_cast<size_t>(t) * block_table_stride_t +
                static_cast<size_t>(qo_head_idx) * block_table_stride_h;
  SparseTopKWaitOnDependentGrids(use_pdl);
  // valid_count = min(max_k_tiles, num_valid_pages); positions [valid_count, topk) → -1
  const uint32_t row_num_valid_pages =
      LoadRowNumValidPages(num_valid_pages_per_token, t, num_valid_pages);
  const uint32_t valid_count =
      (row_num_valid_pages < max_k_tiles) ? row_num_valid_pages : max_k_tiles;
  for (uint32_t i = tx; i < topk; i += blockDim.x) {
    const int32_t logical_idx = (i < valid_count) ? static_cast<int32_t>(i) : -1;
    row[static_cast<size_t>(i) * out_stride_k] =
        GatherBlockTableValue(block_table_row, block_table_stride_k, logical_idx);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    SparseTopKLaunchDependentGrids(use_pdl);
  }
}

// =============================================================================
// Warp-only ascending sort (v2.1 fix replacing the heavy cub::BlockRadixSort)
// =============================================================================
//
// v2.0 used cub::BlockRadixSort<uint32_t, 512, 1> over all 512 threads to
// sort the topk indices ascending — but only ≤64 of the 512 keys are real
// (rest are sentinel ~0u), so 92%+ of the sort work was wasted, and nsys
// measured this as ~13 us per kernel call (largest single component of the
// IndexerTopK kernel duration).
//
// v2.1 replaces it with a warp-only bitonic sort (single warp, 32 lanes ×
// {1, 2} keys depending on MAX_TOPK), modelled on v1.6.5's WarpBitonicSortAsc16.
// Expected cost: < 1 us, savings ~12 us per kernel call.

constexpr int kFinalCandidateSourceBits = 14;
constexpr uint64_t kFinalCandidateSourceMask = (1ull << kFinalCandidateSourceBits) - 1;

// Pack score, staging position, and source index into one sortable key.  The
// high 32 bits preserve numeric float order.  Staging position is next so an
// equal-score tie follows the legacy rank loop, which preferred larger staged
// positions (`i < j` incremented i's rank).  The source index is payload in the
// low 14 bits; max_k_tiles is constrained to < 12288 by the dispatcher.
__device__ __forceinline__ uint64_t MakeFinalCandidateKey(float logit,
                                                         int staging_position,
                                                         int source_index) {
  // Numeric comparison treats -0.0f and +0.0f as equal, so canonicalize zero
  // before converting the score to an integer ordering key.
  const uint32_t raw_bits = __float_as_uint(logit);
  const uint32_t bits = (raw_bits & 0x7fffffffu) == 0 ? 0u : raw_bits;
  const uint32_t ordered_bits =
      (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (static_cast<uint64_t>(ordered_bits) << 32) |
         (static_cast<uint64_t>(staging_position) << kFinalCandidateSourceBits) |
         static_cast<uint32_t>(source_index);
}

struct FinalCandidateKeyGreater {
  __device__ __forceinline__ bool operator()(uint64_t lhs, uint64_t rhs) const {
    return lhs > rhs;
  }
};

// 32-element ascending sort, 32 lanes × 1 key each (for MAX_TOPK ≤ 32).
// Lanes with no real data should pass key = ~0u sentinel, which sorts to the end.
__device__ __forceinline__ void WarpBitonicSortAsc32(uint32_t& key, uint32_t lane) {
  constexpr uint32_t MASK = 0xFFFFFFFFu;
#pragma unroll
  for (int k = 2; k <= 32; k *= 2) {
#pragma unroll
    for (int j = k / 2; j > 0; j /= 2) {
      uint32_t partner = __shfl_xor_sync(MASK, key, j);
      const bool asc_pair = ((lane & k) == 0);
      const bool I_am_lower = ((lane & j) == 0);
      if (I_am_lower) {
        key = asc_pair ? min(key, partner) : max(key, partner);
      } else {
        key = asc_pair ? max(key, partner) : min(key, partner);
      }
    }
  }
}

// 64-element ascending sort, 32 lanes × 2 keys each (for MAX_TOPK == 64).
// BLOCKED layout: lane L holds positions 2L, 2L+1.
// Lanes/slots without real data pass ~0u sentinel.
__device__ __forceinline__ void WarpBitonicSortAsc64(uint32_t* keys, uint32_t lane) {
  constexpr uint32_t MASK = 0xFFFFFFFFu;
  constexpr int KPT = 2;
  constexpr int N = 64;
#pragma unroll
  for (int k = 2; k <= N; k *= 2) {
#pragma unroll
    for (int j = k / 2; j > 0; j /= 2) {
      // For BLOCKED KPT=2 layout: position i = 2L + s
      // (i & k) for k power of 2 ≥ 2: == 0 iff (L & (k/2)) == 0
      // For k == N: (i & N) == 0 always since i < N → asc_pair = true (final merge)
      const bool asc_pair = (k < N) ? ((lane & (k / 2)) == 0) : true;

      if (j >= KPT) {
        // Cross-lane shuffle (mask in lane-space = j / KPT)
        const int mask = j / KPT;
        const bool I_am_lower = ((lane & mask) == 0);
#pragma unroll
        for (int s = 0; s < KPT; ++s) {
          uint32_t my = keys[s];
          uint32_t partner = __shfl_xor_sync(MASK, my, mask);
          if (I_am_lower) {
            keys[s] = asc_pair ? min(my, partner) : max(my, partner);
          } else {
            keys[s] = asc_pair ? max(my, partner) : min(my, partner);
          }
        }
      } else {
        // j == 1, within-thread: compare slot 0 vs slot 1 of same lane
        if (asc_pair) {
          if (keys[0] > keys[1]) {
            uint32_t t = keys[0]; keys[0] = keys[1]; keys[1] = t;
          }
        } else {
          if (keys[0] < keys[1]) {
            uint32_t t = keys[0]; keys[0] = keys[1]; keys[1] = t;
          }
        }
      }
    }
  }
}

// =============================================================================
// IndexerTopKWithSortKernel — fused indexerTopK + asc-by-index sort + qo_outermost write
// =============================================================================
//
// Per-CTA work (1 CTA per (qo_head, qo_pos) row):
//   1. Stage 0: 10-bit fp16 histogram (1024 bins) → cub::BlockScan threshold
//      → classify each element: bin < threshold ⇒ direct emit to smemOutput;
//        bin == threshold ⇒ stage to smemFinal.items if it fits (≤ 2048).
//   2. Stage 1/2/3 (only if needed): 10+10+10 bit fp32 histogram refinement
//      on the elements that landed in the threshold bin.
//   3. If we never fell through to step 3, the staging buffer has a
//      well-defined subset of "ties at the boundary".  Small bins use bounded
//      all-pairs ranking; large bins use a two-level warp merge.
//   4. Warp-only bitonic sort over smemOutput[0..topk-1] → ascending by index.
//   5. Write to gmem with qo_outermost strides:
//        out[t][qo_head_idx][k] at t*out_stride_t + qo_head_idx*out_stride_h + k*out_stride_k
//
// Template params:
//   MAX_TOPK            — round-up template bin (16/32/64), runtime topk ≤ MAX_TOPK
//   kNumThreadsPerBlock — fixed at 512 (matches trtllm canonical config)
//   kNumBins            — fixed at 1024 (10-bit fp16 / fp32 hist)
//   kNumFinalItems      — fixed at 2048 (final-candidate staging capacity)
//
// Dynamic SMEM = topk * sizeof(int32_t).  The module configures the kernel's
// dynamic-SMEM attribute once at load time so graph capture only sees launches.
constexpr uint32_t kSparseTopkMaxK = 64;
constexpr int kIndexerNumThreadsPerBlock = 512;
constexpr int kIndexerNumBins = 1024;  // 10-bit hist (was 2048 / 11-bit)
constexpr int kIndexerNumFinalItems = 2048;
constexpr int kIndexerQuadraticSelectionLimit = 416;

template <uint32_t MAX_TOPK>
__global__ void __launch_bounds__(kIndexerNumThreadsPerBlock) IndexerTopKWithSortKernel(
    const float* __restrict__ in,        // row-contig fp32: (H, T, K) or (T, H, K)
    int32_t* __restrict__ out,           // (qo, num_qo_heads, topk) logical int32
    const int32_t* __restrict__ block_table,
    uint64_t out_stride_t, uint64_t out_stride_h, uint64_t out_stride_k,
    uint64_t block_table_stride_t, uint64_t block_table_stride_h,
    uint64_t block_table_stride_k,
    uint32_t total_qo_len, uint32_t num_qo_heads, uint32_t max_k_tiles,
    uint32_t topk,
    // v2.5_oob_clamp_in_kernel: indices >= num_valid_pages get rewritten to -1
    // and sorted to the tail.  Pass num_valid_pages = max_k_tiles to disable
    // (the comparison `idx >= num_valid_pages` will then never trigger because
    // every valid idx is in [0, max_k_tiles)).
    uint32_t num_valid_pages,
    const int32_t* __restrict__ num_valid_pages_per_token,
    uint32_t force_begin, uint32_t force_end,
    uint32_t input_qo_outermost, uint32_t use_pdl) {
  static_assert(MAX_TOPK <= kSparseTopkMaxK, "MAX_TOPK exceeds supported max");

  constexpr int kNumThreadsPerBlock = kIndexerNumThreadsPerBlock;
  constexpr int kNumBins = kIndexerNumBins;
  constexpr int kNumFinalItems = kIndexerNumFinalItems;
  constexpr int kNumWarps = kNumThreadsPerBlock / 32;
  constexpr int kCandidatesPerThread =
      (kNumFinalItems + kNumThreadsPerBlock - 1) / kNumThreadsPerBlock;
  constexpr int kWarpCandidatesPerLane =
      (kNumWarps * MAX_TOPK + 31) / 32;
  static_assert(kNumThreadsPerBlock % 32 == 0);
  static_assert(kCandidatesPerThread * kNumThreadsPerBlock >= kNumFinalItems);

  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
  using CandidateWarpSort = cub::WarpMergeSort<uint64_t, kCandidatesPerThread, 32>;
  using CandidateGlobalSort = cub::WarpMergeSort<uint64_t, kWarpCandidatesPerLane, 32>;

  struct FinalItems {
    int indices[kNumFinalItems];
    float logits[kNumFinalItems];
  };
  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };
  struct CandidateWarpSortStorage {
    typename CandidateWarpSort::TempStorage warps[kNumWarps];
  };

  // SMEM union — v2.1 dropped the cub::BlockRadixSort sortByIndex member
  // since the asc-by-index sort is now warp-only.  Final-candidate merge-sort
  // storage can alias items after every candidate has been packed in registers.
  __shared__ union {
    FinalItems items;
    Histogram histo;
    CandidateWarpSortStorage candidateWarpSort;
    typename CandidateGlobalSort::TempStorage candidateGlobalSort;
  } smemFinal;

  // Dynamic SMEM holds the top-K accumulator (one int32 per slot, sized by topk).
  extern __shared__ int32_t smemOutput[];

  __shared__ int smemThresholdBinIdx[1];
  __shared__ int smemFinalDstIdx[1];
  __shared__ int smemFinalBinSize[1];
  __shared__ int smemFoundTopKValues[1];
  __shared__ uint64_t smemWarpTopCandidateKeys[kNumWarps * MAX_TOPK];

  // ---- bid -> row decode ---------------------------------------------------
  // HKT transpose path: bid = qo_head_idx * total_qo_len + t.
  // THK direct path:    bid = t * num_qo_heads + qo_head_idx.
  const uint32_t bid = blockIdx.x;
  const uint32_t t = input_qo_outermost ? (bid / num_qo_heads) : (bid % total_qo_len);
  const uint32_t qo_head_idx = input_qo_outermost ? (bid % num_qo_heads) : (bid / total_qo_len);
  // qo_outermost output offset:
  //   out (t, qo_head_idx, k) at offset t*out_stride_t + qo_head_idx*out_stride_h + k*out_stride_k
  const size_t row_offset_out =
      static_cast<size_t>(t) * out_stride_t +
      static_cast<size_t>(qo_head_idx) * out_stride_h;
  const int32_t* block_table_row =
      block_table == nullptr
          ? nullptr
          : block_table + static_cast<size_t>(t) * block_table_stride_t +
                static_cast<size_t>(qo_head_idx) * block_table_stride_h;

  // Per-row input pointer (row stride = max_k_tiles).
  const float* logits = in + static_cast<size_t>(bid) * max_k_tiles;
  const int rowStart = 0;
  const int topK = static_cast<int>(topk);
  SparseTopKWaitOnDependentGrids(use_pdl);
  const uint32_t row_num_valid_pages =
      LoadRowNumValidPages(num_valid_pages_per_token, t, num_valid_pages);
  // Padding is always -inf and can never be a useful sparse-attention page.
  // Restrict every histogram/refinement pass to the valid prefix instead of
  // repeatedly scanning max_k_tiles for rows with a large padded tail.
  const int rowEnd = static_cast<int>(
      row_num_valid_pages < max_k_tiles ? row_num_valid_pages : max_k_tiles);
  const int rowLen = rowEnd - rowStart;
  const uint32_t force_end_start =
      (force_end <= row_num_valid_pages) ? row_num_valid_pages - force_end : 0;

  // If the actual valid page count fits in top-k, every valid page must be
  // emitted exactly once and the rest of the row is invalid padding.  Bypass the
  // histogram path so large -inf padding ties cannot manufacture arbitrary
  // duplicate/OOB selections.
  if (row_num_valid_pages <= static_cast<uint32_t>(topK)) {
    int32_t* row_out = out + row_offset_out;
    const int valid_count = (row_num_valid_pages < static_cast<uint32_t>(rowLen))
                                ? static_cast<int>(row_num_valid_pages)
                                : rowLen;
    for (int rowIt = threadIdx.x; rowIt < valid_count; rowIt += kNumThreadsPerBlock) {
      row_out[static_cast<size_t>(rowIt) * out_stride_k] =
          GatherBlockTableValue(block_table_row, block_table_stride_k, rowIt);
    }
    for (int rowIt = valid_count + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock) {
      row_out[static_cast<size_t>(rowIt) * out_stride_k] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      SparseTopKLaunchDependentGrids(use_pdl);
    }
    return;
  }

  // ---- Trivial path: rowLen <= topK (defensive — host dispatcher already
  //      catches max_k_tiles <= topk via SparseTopKIdentityFillKernel).
  if (rowLen <= topK) {
    int32_t* row_out = out + row_offset_out;
    // OOB clamp: indices >= num_valid_pages → -1
    const int valid_count = (static_cast<int>(row_num_valid_pages) < rowLen)
                                ? static_cast<int>(row_num_valid_pages)
                                : rowLen;
    for (int rowIt = threadIdx.x; rowIt < valid_count; rowIt += kNumThreadsPerBlock) {
      row_out[static_cast<size_t>(rowIt) * out_stride_k] =
          GatherBlockTableValue(block_table_row, block_table_stride_k, rowIt);
    }
    for (int rowIt = valid_count + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock) {
      row_out[static_cast<size_t>(rowIt) * out_stride_k] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      SparseTopKLaunchDependentGrids(use_pdl);
    }
    return;
  }

  // ---- Init scalar SMEM counters -----------------------------------------
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // ---- Stage 0: fp16 10-bit hist -----------------------------------------
  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
          logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
          smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, rowStart, topK,
          force_begin, force_end_start, row_num_valid_pages);

  if (continueToNextStep) {
    // Stage 1: fp32 high 10 bits.
    continueToNextStep =
        processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
            logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
            smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, rowStart, topK,
            force_begin, force_end_start, row_num_valid_pages);
  }
  if (continueToNextStep) {
    // Stage 2: fp32 mid 10 bits.
    continueToNextStep =
        processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
            logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
            smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, rowStart, topK,
            force_begin, force_end_start, row_num_valid_pages);
  }
  if (continueToNextStep) {
    // Stage 3: fp32 low 10 bits.  After this step every remaining tie shares
    // the same 32-bit pattern so we just fill the topk window directly.
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
        logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
        smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, rowStart, topK,
        force_begin, force_end_start, row_num_valid_pages);
  }

  if (!continueToNextStep) {
    // The threshold bin fit within kNumFinalItems.  The all-pairs rank loop is
    // fastest for small bins, but cap it below the measured crossover to avoid
    // an overflow -> refinement -> O(n^2) cliff.  Larger bins use a two-level
    // warp merge: each warp contributes its local top candidates, then warp 0
    // merges at most kNumWarps * topK keys.
    const int baseIdx = smemFoundTopKValues[0];
    const int finalCount = smemFinalDstIdx[0];
    const int numFinalSelections = topK - baseIdx;
    const int selectionLane = threadIdx.x & 31;
    const int selectionWarp = threadIdx.x >> 5;

    if (finalCount <= kIndexerQuadraticSelectionLimit) {
      for (int i = threadIdx.x; i < finalCount; i += kNumThreadsPerBlock) {
        int outIndex = 0;
        const float logit = smemFinal.items.logits[i];
        for (int j = 0; j < finalCount; ++j) {
          const float otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            ++outIndex;
          }
        }
        if (outIndex + baseIdx < topK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
        }
      }
      __syncthreads();
    } else {
      // Pack all shared-memory candidates into registers before merge-sort
      // reuses the same union as temporary storage.
      uint64_t candidateKeys[kCandidatesPerThread];
#pragma unroll
      for (int item = 0; item < kCandidatesPerThread; ++item) {
        const int position = threadIdx.x + item * kNumThreadsPerBlock;
        candidateKeys[item] =
            position < finalCount
                ? MakeFinalCandidateKey(smemFinal.items.logits[position], position,
                                        smemFinal.items.indices[position])
                : 0;
      }
      __syncthreads();

      CandidateWarpSort(smemFinal.candidateWarpSort.warps[selectionWarp])
          .Sort(candidateKeys, FinalCandidateKeyGreater{});

#pragma unroll
      for (int item = 0; item < kCandidatesPerThread; ++item) {
        const int rank = selectionLane * kCandidatesPerThread + item;
        if (rank < numFinalSelections) {
          smemWarpTopCandidateKeys[selectionWarp * numFinalSelections + rank] =
              candidateKeys[item];
        }
      }
      __syncthreads();

      // Merge the per-warp top keys with warp 0.  No item ranked below R within
      // its own warp can enter the block-wide top R.
      if (selectionWarp == 0) {
        const int numWarpCandidates = kNumWarps * numFinalSelections;
        uint64_t warpCandidateKeys[kWarpCandidatesPerLane];
#pragma unroll
        for (int item = 0; item < kWarpCandidatesPerLane; ++item) {
          const int position = selectionLane * kWarpCandidatesPerLane + item;
          warpCandidateKeys[item] =
              position < numWarpCandidates ? smemWarpTopCandidateKeys[position] : 0;
        }

        CandidateGlobalSort(smemFinal.candidateGlobalSort)
            .Sort(warpCandidateKeys, FinalCandidateKeyGreater{});

#pragma unroll
        for (int item = 0; item < kWarpCandidatesPerLane; ++item) {
          const int rank = selectionLane * kWarpCandidatesPerLane + item;
          if (rank < numFinalSelections) {
            smemOutput[baseIdx + rank] = static_cast<int>(
                warpCandidateKeys[item] & kFinalCandidateSourceMask);
          }
        }
      }
    }
  }

  // ---- v2.1 warp-only fused asc-by-index sort -----------------------------
  // After final-candidate selection fills smemOutput[0..topk-1] (unsorted by index),
  // only warp 0 (32 lanes) participates in the sort.  Other 480 threads idle.
  //
  // Why warp-only vs cub::BlockRadixSort<512, 1>:
  //   - cub::BlockRadixSort<uint32_t, 512, 1>.Sort() processes ALL 512 thread
  //     positions even though only ≤64 are real (rest are sentinel ~0u).
  //     Per nsys, that variant cost ~13 us per kernel call.
  //   - WarpBitonicSortAsc{32,64} only sorts the 32/64 real slots.  Estimated
  //     cost: < 1 us per kernel.  Savings: ~12 us per call = ~50% of the
  //     v2.0 IndexerTopKWithSortKernel runtime on n1024_K8192.
  __syncthreads();  // ensure all threads' writes to smemOutput are visible
  if (threadIdx.x == 0) {
    SparseTopKLaunchDependentGrids(use_pdl);
  }

  const uint32_t warp_id = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  if (warp_id != 0) return;  // only warp 0 sorts and writes the output

  int32_t* row_out = out + row_offset_out;

  if constexpr (MAX_TOPK <= 32) {
    // 32 lanes × 1 key.  Lanes >= topk pad with sentinel ~0u (sorts to end).
    // v2.5_oob_clamp_in_kernel: also fold OOB clamp here — indices >=
    // num_valid_pages get rewritten to ~0u, so they sort to the tail and the
    // existing `(key == ~0u) ? -1` write contract converts them to -1.
    uint32_t key = ~0u;
    if (lane < topK) {
      const int32_t idx = smemOutput[lane];
      const bool valid = (idx >= 0) && (static_cast<uint32_t>(idx) < row_num_valid_pages);
      key = valid ? static_cast<uint32_t>(idx) : ~0u;
    }
    WarpBitonicSortAsc32(key, lane);
    if (lane < topK) {
      const int32_t logical_idx = (key == ~0u) ? -1 : static_cast<int32_t>(key);
      row_out[static_cast<size_t>(lane) * out_stride_k] =
          GatherBlockTableValue(block_table_row, block_table_stride_k, logical_idx);
    }
  } else {
    // MAX_TOPK == 64: 32 lanes × 2 keys, BLOCKED layout (lane L holds 2L, 2L+1).
    uint32_t keys[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int pos = static_cast<int>(lane) * 2 + s;
      if (pos < topK) {
        const int32_t idx = smemOutput[pos];
        const bool valid = (idx >= 0) && (static_cast<uint32_t>(idx) < row_num_valid_pages);
        keys[s] = valid ? static_cast<uint32_t>(idx) : ~0u;
      } else {
        keys[s] = ~0u;
      }
    }
    WarpBitonicSortAsc64(keys, lane);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int pos = static_cast<int>(lane) * 2 + s;
      if (pos < topK) {
        const uint32_t k = keys[s];
        const int32_t logical_idx = (k == ~0u) ? -1 : static_cast<int32_t>(k);
        row_out[static_cast<size_t>(pos) * out_stride_k] =
            GatherBlockTableValue(block_table_row, block_table_stride_k, logical_idx);
      }
    }
  }
}

// =============================================================================
// Host-side dispatcher
// =============================================================================

inline cudaError_t ConfigureSparseTopKSelect() {
  static bool s_attr_set = false;
  if (s_attr_set) return cudaSuccess;
  auto kernel = IndexerTopKWithSortKernel<16>;
  constexpr size_t dyn_smem_bytes = 16 * sizeof(int32_t);
  cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(dyn_smem_bytes));
  if (err != cudaSuccess) return err;
  s_attr_set = true;
  return cudaSuccess;
}

inline cudaError_t LaunchSparseTopKKernel(const void* kernel, dim3 grid, dim3 block,
                                          void** args, size_t dyn_smem_bytes,
                                          cudaStream_t stream, bool enable_pdl) {
  if (!enable_pdl) {
    return cudaLaunchKernel(kernel, grid, block, args, dyn_smem_bytes, stream);
  }

#if ((__CUDACC_VER_MAJOR__ >= 12) || \
     ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8)))
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dyn_smem_bytes;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = 1;
  return cudaLaunchKernelExC(&config, kernel, args);
#else
  return cudaLaunchKernel(kernel, grid, block, args, dyn_smem_bytes, stream);
#endif
}

cudaError_t LaunchIndexerTopK(const float* in_row_contig, int32_t* out,
                              const int32_t* block_table,
                              uint64_t out_stride_t, uint64_t out_stride_h,
                              uint64_t out_stride_k,
                              uint64_t block_table_stride_t,
                              uint64_t block_table_stride_h,
                              uint64_t block_table_stride_k,
                              uint32_t total_qo_len, uint32_t num_qo_heads,
                              uint32_t max_k_tiles, uint32_t topk,
                              uint32_t num_valid_pages,
                              const int32_t* num_valid_pages_per_token,
                              uint32_t force_begin, uint32_t force_end,
                              bool input_qo_outermost,
                              cudaStream_t stream, bool enable_pdl) {
  const uint32_t num_rows = total_qo_len * num_qo_heads;
  if (num_rows == 0) return cudaSuccess;

  auto kernel = IndexerTopKWithSortKernel<16>;
  const size_t dyn_smem_bytes = static_cast<size_t>(topk) * sizeof(int32_t);
  const uint32_t input_qo_outermost_u32 = input_qo_outermost ? 1u : 0u;
  const uint32_t use_pdl = enable_pdl ? 1u : 0u;
  void* args[] = {(void*)&in_row_contig, (void*)&out, (void*)&block_table,
                  (void*)&out_stride_t, (void*)&out_stride_h, (void*)&out_stride_k,
                  (void*)&block_table_stride_t, (void*)&block_table_stride_h,
                  (void*)&block_table_stride_k,
                  (void*)&total_qo_len,
                  (void*)&num_qo_heads, (void*)&max_k_tiles, (void*)&topk,
                  (void*)&num_valid_pages, (void*)&num_valid_pages_per_token,
                  (void*)&force_begin, (void*)&force_end, (void*)&input_qo_outermost_u32,
                  (void*)&use_pdl};
  dim3 grid(num_rows);
  dim3 block(kIndexerNumThreadsPerBlock);
  return LaunchSparseTopKKernel((const void*)kernel, grid, block, args, dyn_smem_bytes,
                                stream, enable_pdl);
}

cudaError_t LaunchTransposeAndIndexerTopK(const float* in_strided, float* transposed, int32_t* out,
                                          const int32_t* block_table,
                                          uint64_t out_stride_t, uint64_t out_stride_h,
                                          uint64_t out_stride_k,
                                          uint64_t block_table_stride_t,
                                          uint64_t block_table_stride_h,
                                          uint64_t block_table_stride_k,
                                          uint32_t total_qo_len, uint32_t num_qo_heads,
                                          uint32_t max_k_tiles, uint32_t topk,
                                          uint32_t num_valid_pages,
                                          const int32_t* num_valid_pages_per_token,
                                          uint32_t force_begin, uint32_t force_end,
                                          cudaStream_t stream, bool enable_pdl) {
  const uint32_t num_rows = total_qo_len * num_qo_heads;
  if (num_rows == 0) return cudaSuccess;
  const uint32_t use_pdl = enable_pdl ? 1u : 0u;

  // ---- (1) Transpose: dispatch to TransposeXorF4<32> fast path when
  //          preconditions are met (qo & K both divisible by 32 and 4),
  //          otherwise fall back to the original padded-tile transpose.
  {
    constexpr int kXorTile = 32;
    const bool xor_path_ok = (total_qo_len % kXorTile == 0) &&
                             (max_k_tiles % kXorTile == 0) &&
                             (total_qo_len % 4 == 0) &&
                             (max_k_tiles % 4 == 0);
    if (xor_path_ok) {
      dim3 grid(total_qo_len / kXorTile, max_k_tiles / kXorTile, num_qo_heads);
      dim3 block(kXorTile * kXorTile / 4);  // = 256 threads
      void* tr_args[] = {
          (void*)&in_strided, (void*)&transposed, (void*)&max_k_tiles, (void*)&total_qo_len,
          (void*)&use_pdl};
      cudaError_t err = LaunchSparseTopKKernel(
          (const void*)SparseTopKTransposeXorF4Kernel<kXorTile>, grid, block, tr_args, 0,
          stream, enable_pdl);
      if (err != cudaSuccess) return err;
    } else {
      dim3 grid((total_qo_len + kTransposeTile - 1) / kTransposeTile,
                (max_k_tiles + kTransposeTile - 1) / kTransposeTile, num_qo_heads);
      dim3 block(kTransposeTile, kTransposeBlockRows);
      void* tr_args[] = {
          (void*)&in_strided, (void*)&transposed, (void*)&max_k_tiles, (void*)&total_qo_len,
          (void*)&use_pdl};
      cudaError_t err = LaunchSparseTopKKernel((const void*)SparseTopKTransposeKernel, grid,
                                               block, tr_args, 0, stream, enable_pdl);
      if (err != cudaSuccess) return err;
    }
  }

  // ---- (2) IndexerTopK + fused sort + write to qo_outermost layout -------
  return LaunchIndexerTopK(transposed, out, block_table,
                           out_stride_t, out_stride_h, out_stride_k,
                           block_table_stride_t, block_table_stride_h, block_table_stride_k,
                           total_qo_len, num_qo_heads, max_k_tiles, topk,
                           num_valid_pages, num_valid_pages_per_token,
                           force_begin, force_end,
                           /*input_qo_outermost=*/false, stream, enable_pdl);
}

// =============================================================================
// Workspace size — transpose_buf only: (num_qo_heads, max_k_tiles, total_qo_len) fp32.
// =============================================================================
inline size_t SparseTopKWorkspaceSize(uint32_t total_qo_len, uint32_t num_qo_heads,
                                      uint32_t max_k_tiles, SparseTopKInputLayout input_layout) {
  if (input_layout == SparseTopKInputLayout::kTHK) return 0;
  return static_cast<size_t>(num_qo_heads) * max_k_tiles * total_qo_len;
}

// =============================================================================
// Top-level dispatcher
// =============================================================================
//   in        : (num_qo_heads, max_k_tiles, total_qo_len) contiguous fp32 for kHKT,
//               or (total_qo_len, num_qo_heads, max_k_tiles) contiguous fp32 for kTHK
//   out       : (total_qo_len, num_qo_heads, topk=16) int32. Without block_table,
//               values are logical k_tile indices asc by k_tile index. With
//               block_table, logical indices are sorted first, then gathered
//               from block_table[t][h][idx].
//   workspace : int32, at least SparseTopKWorkspaceSize(...) elements
//   num_valid_pages : indices >= num_valid_pages are emitted as -1 (sorted to
//                     tail).  Pass max_k_tiles (or any value >= max_k_tiles)
//                     to disable clamping.
//
//   * max_k_tiles <= 16      → SparseTopKIdentityFillKernel (trivial)
//   * 16 < max_k_tiles < 12288 and kHKT → Transpose + IndexerTopKWithSortKernel<16>
//   * 16 < max_k_tiles < 12288 and kTHK → IndexerTopKWithSortKernel<16> directly
//   * max_k_tiles >= 12288   → cudaErrorNotSupported
inline cudaError_t SparseTopKSelect(const float* in, int32_t* out, int32_t* workspace,
                                    const int32_t* block_table,
                                    uint64_t out_stride_t, uint64_t out_stride_h,
                                    uint64_t out_stride_k,
                                    uint64_t block_table_stride_t,
                                    uint64_t block_table_stride_h,
                                    uint64_t block_table_stride_k,
                                    uint32_t total_qo_len, uint32_t num_qo_heads,
                                    uint32_t max_k_tiles, uint32_t num_valid_pages,
                                    const int32_t* num_valid_pages_per_token,
                                    SparseTopKInputLayout input_layout,
                                    uint32_t force_begin, uint32_t force_end,
                                    cudaStream_t stream, bool enable_pdl = true) {
  constexpr uint32_t topk = 16;
  const uint32_t num_rows = total_qo_len * num_qo_heads;
  if (num_rows == 0) return cudaSuccess;
  const uint32_t use_pdl = enable_pdl ? 1u : 0u;

  // ---- Trivial path: max_k_tiles <= topk → identity fill -----------------
  if (max_k_tiles <= topk) {
    dim3 grid(num_rows);
    dim3 block(topk);
    void* args[] = {(void*)&out, (void*)&block_table,
                    (void*)&out_stride_t, (void*)&out_stride_h, (void*)&out_stride_k,
                    (void*)&block_table_stride_t, (void*)&block_table_stride_h,
                    (void*)&block_table_stride_k,
                    (void*)&total_qo_len, (void*)&num_qo_heads,
                    (void*)&max_k_tiles, (void*)&topk, (void*)&num_valid_pages,
                    (void*)&num_valid_pages_per_token, (void*)&use_pdl};
    return LaunchSparseTopKKernel((const void*)SparseTopKIdentityFillKernel, grid, block,
                                  args, 0, stream, enable_pdl);
  }

  // ---- IndexerTopK insertion-sort path ------------------------------------
  if (max_k_tiles >= 12288) return cudaErrorNotSupported;

  if (input_layout == SparseTopKInputLayout::kTHK) {
    return LaunchIndexerTopK(in, out, block_table,
                             out_stride_t, out_stride_h, out_stride_k,
                             block_table_stride_t, block_table_stride_h,
                             block_table_stride_k,
                             total_qo_len, num_qo_heads, max_k_tiles, topk,
                             num_valid_pages, num_valid_pages_per_token,
                             force_begin, force_end,
                             /*input_qo_outermost=*/true, stream, enable_pdl);
  }

  float* transpose_buf = reinterpret_cast<float*>(workspace);
  return LaunchTransposeAndIndexerTopK(in, transpose_buf, out, block_table,
                                       out_stride_t, out_stride_h, out_stride_k,
                                       block_table_stride_t, block_table_stride_h,
                                       block_table_stride_k,
                                       total_qo_len, num_qo_heads,
                                       max_k_tiles, topk, num_valid_pages,
                                       num_valid_pages_per_token,
                                       force_begin, force_end, stream, enable_pdl);
}

}  // namespace sparse_topk
}  // namespace flashinfer

#endif  // FLASHINFER_SPARSE_TOPK_SELECT_CUH_
