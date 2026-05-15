/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portions of this file are adapted from vLLM csrc/topk.cu.
 *
 * Modifications copyright:
 *   Copyright (c) 2026 LightSeek Foundation
 */

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {
namespace prefill_topk {

template <int step>
static inline __device__ uint32_t extract_bin_idx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
  }
  return 0;
}

template <int shift>
static inline __device__ bool is_partial_match(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads,
                                   const T* in, IdxT len, Func f) {
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const IdxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (IdxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <int step, int kNumThreadsPerBlock, int kNumBins,
          int kNumFinalItems, typename SmemFinalType, typename SmemOutputType>
__device__ bool process_histogram_step(
    const float* logits, int row_len, uint32_t& logit_pattern,
    int& threshold_bin_idx, SmemOutputType& smem_output,
    int* smem_threshold_bin_idx, int* smem_final_dst_idx,
    int* smem_final_bin_size, int* smem_found_topk_values,
    SmemFinalType& smem_final, int top_k) {
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smem_final.histo.data[idx] = 0;
  }
  __syncthreads();

  constexpr auto pattern_shift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logit_pattern =
        static_cast<uint32_t>(threshold_bin_idx & 0x7ff) << pattern_shift;
  } else if constexpr (step == 3) {
    logit_pattern |=
        static_cast<uint32_t>(threshold_bin_idx & 0x7ff) << pattern_shift;
  }

  auto distribute_to_bins = [&](float logit, int /*idx*/ = 0) {
    if (is_partial_match<pattern_shift>(logit, logit_pattern)) {
      uint32_t bin_idx = extract_bin_idx<step>(logit);
      atomicAdd(&smem_final.histo.data[bin_idx], 1);
    }
  };

  vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits, row_len,
                     distribute_to_bins);
  __syncthreads();

  int last_value = smem_found_topk_values[0];
  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int bin_count = smem_final.histo.data[idx];
    __syncthreads();

    int prefix_sum = 0;
    int total_sum = 0;
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smem_final.histo.scan).ExclusiveSum(bin_count, prefix_sum, total_sum);

    prefix_sum += last_value;
    total_sum += last_value;
    smem_final.histo.data[idx] = prefix_sum;
    __syncthreads();

    bool found_threshold = false;
    if (prefix_sum < top_k) {
      int next_prefix_sum =
          threadIdx.x == kNumThreadsPerBlock - 1
              ? total_sum
              : smem_final.histo.data[idx + 1];
      if (next_prefix_sum >= top_k) {
        smem_threshold_bin_idx[0] = idx;
        smem_final_bin_size[0] = next_prefix_sum - prefix_sum;
        found_threshold = true;
      }
    }

    if (__syncthreads_or(found_threshold)) {
      break;
    }
    last_value = total_sum;
  }
  __syncthreads();

  threshold_bin_idx = smem_threshold_bin_idx[0];

  auto process_bins = [&](float logit, int idx) {
    if (is_partial_match<pattern_shift>(logit, logit_pattern)) {
      uint32_t bin_idx = extract_bin_idx<step>(logit);
      bool should_write_directly =
          (step == 0 && smem_final_bin_size[0] <= kNumFinalItems) ||
          (step >= 1);
      if (bin_idx < threshold_bin_idx && should_write_directly) {
        int dst_idx = atomicAdd(&smem_found_topk_values[0], 1);
        smem_output[dst_idx] = idx;
      }
      if constexpr (step < 3) {
        if (bin_idx == threshold_bin_idx &&
            smem_final_bin_size[0] <= kNumFinalItems) {
          int dst_idx = atomicAdd(&smem_final_dst_idx[0], 1);
          smem_final.items.logits[dst_idx] = logit;
          smem_final.items.indices[dst_idx] = idx;
        }
      } else {
        if (bin_idx == threshold_bin_idx) {
          int dst_idx = atomicAdd(&smem_final.histo.data[bin_idx], 1);
          if (dst_idx < top_k) {
            smem_output[dst_idx] = idx;
          }
        }
      }
    }
  };

  vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits, row_len,
                     process_bins);
  __syncthreads();
  return smem_final_bin_size[0] > kNumFinalItems;
}

template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort>
static __device__ void topk_per_row_job(const float* logits, int row_start,
                                        int row_end, int* out_indices,
                                        int top_k) {
  static constexpr int kNumFinalItems = 2048;
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage =
      std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  struct FinalItems {
    int indices[kNumFinalItems];
    float logits[kNumFinalItems];
  };
  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smem_final;

  extern __shared__ int32_t smem_output[];
  __shared__ int smem_threshold_bin_idx[1];
  __shared__ int smem_final_dst_idx[1];
  __shared__ int smem_final_bin_size[1];
  __shared__ int smem_found_topk_values[1];

  int row_len = row_end - row_start;
  if (row_len <= top_k) {
    for (int row_it = threadIdx.x; row_it < row_len;
         row_it += kNumThreadsPerBlock) {
      out_indices[row_it] = row_it;
    }
    for (int row_it = row_len + threadIdx.x; row_it < top_k;
         row_it += kNumThreadsPerBlock) {
      out_indices[row_it] = -1;
    }
    return;
  }

  if (threadIdx.x == 0) {
    smem_final_dst_idx[0] = 0;
    smem_found_topk_values[0] = 0;
  }
  __syncthreads();

  int threshold_bin_idx = -1;
  uint32_t logit_pattern = 0;
  const float* row_logits = logits + row_start;

  bool continue_to_next_step =
      process_histogram_step<0, kNumThreadsPerBlock, kNumBins,
                             kNumFinalItems>(
          row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
          smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
          smem_found_topk_values, smem_final, top_k);
  if (continue_to_next_step) {
    continue_to_next_step =
        process_histogram_step<1, kNumThreadsPerBlock, kNumBins,
                               kNumFinalItems>(
            row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
            smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
            smem_found_topk_values, smem_final, top_k);
  }
  if (continue_to_next_step) {
    continue_to_next_step =
        process_histogram_step<2, kNumThreadsPerBlock, kNumBins,
                               kNumFinalItems>(
            row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
            smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
            smem_found_topk_values, smem_final, top_k);
  }
  if (continue_to_next_step) {
    process_histogram_step<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
        row_logits, row_len, logit_pattern, threshold_bin_idx, smem_output,
        smem_threshold_bin_idx, smem_final_dst_idx, smem_final_bin_size,
        smem_found_topk_values, smem_final, top_k);
  }

  if (!continue_to_next_step) {
    if constexpr (useRadixSort) {
      float final_logits[kNumFinalItemsPerThread];
      int final_indices[kNumFinalItemsPerThread];
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        final_logits[ii] = -FLT_MAX;
      }
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int src_idx = ii * kNumThreadsPerBlock + threadIdx.x;
        if (src_idx < smem_final_dst_idx[0]) {
          final_logits[ii] = smem_final.items.logits[src_idx];
          final_indices[ii] = smem_final.items.indices[src_idx];
        }
      }
      __syncthreads();
      FinalSort(smem_final.finalSort)
          .SortDescendingBlockedToStriped(final_logits, final_indices);
      int base_idx = smem_found_topk_values[0];
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int src_idx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dst_idx = base_idx + src_idx;
        if (dst_idx < top_k) {
          smem_output[dst_idx] = final_indices[ii];
        }
      }
    } else {
      auto base_idx = smem_found_topk_values[0];
      for (int i = threadIdx.x; i < smem_final_dst_idx[0];
           i += kNumThreadsPerBlock) {
        int out_index = 0;
        auto logit = smem_final.items.logits[i];
        for (int j = 0; j < smem_final_dst_idx[0]; j++) {
          auto other_logit = smem_final.items.logits[j];
          if (logit < other_logit || (logit == other_logit && i < j)) {
            out_index++;
          }
        }
        if (out_index + base_idx < top_k) {
          smem_output[out_index + base_idx] = smem_final.items.indices[i];
        }
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < top_k; i += kNumThreadsPerBlock) {
    out_indices[i] = smem_output[i];
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock)
void topk_per_row_prefill_kernel(const float* logits, const int* row_starts,
                                 const int* row_ends, int* out_indices,
                                 int stride0, const int top_k,
                                 const int offset_index) {
  static constexpr int kNumBins = 2048;
  int row_idx = blockIdx.x + offset_index;
  int row_start = row_starts[row_idx];
  int row_end = row_ends[row_idx];
  out_indices += static_cast<int64_t>(row_idx) * top_k;
  logits += static_cast<int64_t>(row_idx) * stride0;
  topk_per_row_job<kNumThreadsPerBlock, kNumBins, useRadixSort>(
      logits, row_start, row_end, out_indices, top_k);
}

void launch_indexer_topk_prefill(const TensorView& logits,
                                 const TensorView& row_starts,
                                 const TensorView& row_ends,
                                 TensorView& output, int64_t k,
                                 cudaStream_t stream) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  int num_rows = static_cast<int>(logits.size(0));
  int num_insertion_blocks = std::min(num_rows, kSortingAlgorithmThreshold);
  if (num_insertion_blocks > 0) {
    topk_per_row_prefill_kernel<kNumThreadsPerBlock, false>
        <<<num_insertion_blocks, kNumThreadsPerBlock,
           static_cast<size_t>(k) * sizeof(int32_t), stream>>>(
            static_cast<float*>(logits.data_ptr()),
            static_cast<int*>(row_starts.data_ptr()),
            static_cast<int*>(row_ends.data_ptr()),
            static_cast<int*>(output.data_ptr()), static_cast<int>(logits.stride(0)),
            static_cast<int>(k), 0);
  }
  if (num_rows > kSortingAlgorithmThreshold) {
    int num_radix_blocks = num_rows - kSortingAlgorithmThreshold;
    topk_per_row_prefill_kernel<kNumThreadsPerBlock, true>
        <<<num_radix_blocks, kNumThreadsPerBlock,
           static_cast<size_t>(k) * sizeof(int32_t), stream>>>(
            static_cast<float*>(logits.data_ptr()),
            static_cast<int*>(row_starts.data_ptr()),
            static_cast<int*>(row_ends.data_ptr()),
            static_cast<int*>(output.data_ptr()), static_cast<int>(logits.stride(0)),
            static_cast<int>(k), kSortingAlgorithmThreshold);
  }
}

}  // namespace prefill_topk
}  // namespace

void deepseek_v4_indexer_topk_prefill(TensorView logits, TensorView row_starts,
                                      TensorView row_ends, TensorView output,
                                      int64_t k) {
  CHECK_CUDA(logits);
  CHECK_CUDA(row_starts);
  CHECK_CUDA(row_ends);
  CHECK_CUDA(output);
  CHECK_DIM(2, logits);
  CHECK_DIM(1, row_starts);
  CHECK_DIM(1, row_ends);
  CHECK_DIM(2, output);
  CHECK_CONTIGUOUS(logits);
  CHECK_CONTIGUOUS(row_starts);
  CHECK_CONTIGUOUS(row_ends);
  CHECK_CONTIGUOUS(output);
  TVM_FFI_ICHECK(logits.dtype() == dl_float32) << "logits must be float32";
  TVM_FFI_ICHECK(row_starts.dtype() == dl_int32)
      << "row_starts must be int32";
  TVM_FFI_ICHECK(row_ends.dtype() == dl_int32) << "row_ends must be int32";
  TVM_FFI_ICHECK(output.dtype() == dl_int32) << "output must be int32";
  TVM_FFI_ICHECK(logits.stride(1) == 1)
      << "logits last dimension must be contiguous";
  TVM_FFI_ICHECK(row_starts.size(0) == logits.size(0))
      << "row_starts size mismatch";
  TVM_FFI_ICHECK(row_ends.size(0) == logits.size(0))
      << "row_ends size mismatch";
  TVM_FFI_ICHECK(output.size(0) == logits.size(0) && output.size(1) == k)
      << "output size mismatch";
  TVM_FFI_ICHECK(k > 0) << "k must be positive";

  cudaSetDevice(logits.device().device_id);
  cudaStream_t stream = get_stream(logits.device());
  prefill_topk::launch_indexer_topk_prefill(logits, row_starts, row_ends, output,
                                            k, stream);
  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "deepseek_v4_indexer_topk_prefill failed: "
      << cudaGetErrorString(err);
}
