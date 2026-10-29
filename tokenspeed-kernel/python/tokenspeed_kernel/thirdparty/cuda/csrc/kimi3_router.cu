/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Derived from FlashInfer v0.7.2's Cake warp-per-row Kimi-K3 router
// (d84335d8c54e0c8f9f7e22faa15c7e34a58abf94); aligned-plan construction
// removed.

#include <cstdint>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include "tvm_ffi_utils.h"

namespace {

constexpr int NUM_EXPERTS = 896;
constexpr int TOP_K = 16;
constexpr int NUM_WARPS = 7;
constexpr int THREADS = NUM_WARPS * 32;

__global__ __launch_bounds__(THREADS, 4) void cuda_kimi3_sigmoid_bias_topk(
    const float *__restrict__ logits, const float *__restrict__ bias,
    float *__restrict__ topk_weights, int *__restrict__ topk_ids,
    int64_t num_tokens) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  __shared__ int warp_hist[NUM_WARPS * 288];
  __shared__ int winner_ids[NUM_WARPS * TOP_K];
  __shared__ float winner_scores[NUM_WARPS * TOP_K];
  __shared__ float bias_smem[NUM_EXPERTS];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    bias_smem[tid * 4 + i] = bias[tid * 4 + i];
  }
  __syncthreads();

  const int stash_base = warp * TOP_K;
  const int hist_base = warp * 288;
  const int lane_hist_base = hist_base + lane * 9;
  const int scan_bin_base = lane * 8;
  const unsigned int one_u32 = 1;
  const unsigned int lanemask_lt = (one_u32 << lane) - 1;
  for (int64_t token = static_cast<int64_t>(blockIdx.x) * NUM_WARPS + warp;
       token < num_tokens;
       token += static_cast<int64_t>(gridDim.x) * NUM_WARPS) {
    unsigned long long row_base =
        (unsigned long long)token * (unsigned long long)NUM_EXPERTS;
    unsigned int keys[28];
#pragma unroll
    for (int item = 0; item < 28; item++) {
      int col = item * 32 + lane;
      float logit = logits[row_base + (unsigned long long)col];
      float _expf_0 = __expf(-logit);
      float _fdiv_full_0;
      asm volatile("div.full.f32 %0, %1, %2;"
                   : "=f"(_fdiv_full_0)
                   : "f"(1.0f), "f"(1.0f + _expf_0));
      float score = _fdiv_full_0;
      float bias_col = bias_smem[col];
      float ranking = score + bias_col;
      if (ranking == 0.0f) {
        ranking = 0.0f;
      }
      unsigned int bits = __float_as_uint(ranking);
      unsigned int ordered = bits | 2147483648u;
      if (bits >= 2147483648u) {
        ordered = bits ^ 4294967295u;
      }
      keys[item] = ordered;
    }
    int remaining = TOP_K;
    int active_total = NUM_EXPERTS;
    unsigned int threshold = 0;
    unsigned int examined_mask = 0;
    int take_all_equals = 0;
    unsigned int lane_top = 0;
#pragma unroll
    for (int item_top = 0; item_top < 28; item_top++) {
      unsigned int top_digit = keys[item_top] >> 21;
      if (top_digit > lane_top) {
        lane_top = top_digit;
      }
    }
    unsigned int _warp_redux_u32_0;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;"
                 : "=r"(_warp_redux_u32_0)
                 : "r"(lane_top));
    unsigned int warp_top = _warp_redux_u32_0;
    unsigned int lane_top_count = 0;
#pragma unroll
    for (int item_cnt = 0; item_cnt < 28; item_cnt++) {
      if (keys[item_cnt] >> 21 == warp_top) {
        lane_top_count += 1;
      }
    }
    unsigned int _warp_redux_u32_1;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;"
                 : "=r"(_warp_redux_u32_1)
                 : "r"(lane_top_count));
    unsigned int warp_top_count = _warp_redux_u32_1;
    int shift = 13;
    int spec_round = 0;
    if (warp_top_count >= (unsigned int)TOP_K) {
      threshold = warp_top << 21;
      examined_mask = (unsigned int)2047 << 21;
      active_total = (int)warp_top_count;
      if (remaining == active_total) {
        take_all_equals = 1;
      }
    } else {
      threshold = warp_top >> 3 << 24;
      examined_mask = (unsigned int)255 << 24;
      shift = 16;
      spec_round = 1;
    }
    if (take_all_equals == 0) {
#pragma unroll 1
      for (int radix_round = 0; radix_round < 5; radix_round++) {
        __syncwarp();
#pragma unroll
        for (int clear_slot = 0; clear_slot < 8; clear_slot++) {
          warp_hist[lane_hist_base + clear_slot] = 0;
        }
        __syncwarp();
#pragma unroll
        for (int item_hist = 0; item_hist < 28; item_hist++) {
          if ((keys[item_hist] & examined_mask) == threshold) {
            int digit = (int)(keys[item_hist] >> (unsigned int)shift & 255);
            atomicAdd(&warp_hist[hist_base + digit + (digit >> 3)], 1);
          }
        }
        __syncwarp();
        int scan_counts[8];
        int lane_count = 0;
#pragma unroll
        for (int scan_offset = 0; scan_offset < 8; scan_offset++) {
          int scan_count = warp_hist[lane_hist_base + scan_offset];
          scan_counts[scan_offset] = scan_count;
          lane_count += scan_count;
        }
        int inclusive = lane_count;
        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
        int scan_peer = _shfl_up_0;
        if (lane >= 1) {
          inclusive += scan_peer;
        }
        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
        int scan_peer_0 = _shfl_up_1;
        if (lane >= 2) {
          inclusive += scan_peer_0;
        }
        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
        int scan_peer_1 = _shfl_up_2;
        if (lane >= 4) {
          inclusive += scan_peer_1;
        }
        int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
        int scan_peer_2 = _shfl_up_3;
        if (lane >= 8) {
          inclusive += scan_peer_2;
        }
        int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
        int scan_peer_3 = _shfl_up_4;
        if (lane >= 16) {
          inclusive += scan_peer_3;
        }
        int consume_round = 1;
        if (spec_round != 0) {
          spec_round = 0;
          int _shfl_0 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
          active_total = _shfl_0;
          if (active_total < TOP_K) {
            threshold = (unsigned int)0;
            examined_mask = (unsigned int)0;
            active_total = NUM_EXPERTS;
            shift = 24;
            consume_round = 0;
          }
        }
        if (consume_round != 0) {
          int need = active_total - remaining;
          int lane_excl = inclusive - lane_count;
          int run_prefix = lane_excl;
          int bin_below = lane_excl;
          int bin_top = inclusive;
          int bins_before = 0;
#pragma unroll
          for (int scan_offset_1 = 0; scan_offset_1 < 8; scan_offset_1++) {
            int bin_count = scan_counts[scan_offset_1];
            run_prefix += bin_count;
            if (run_prefix <= need) {
              bin_below = run_prefix;
              bins_before += 1;
            } else {
              int _min_0 =
                  ((bin_top) < (run_prefix) ? (bin_top) : (run_prefix));
              bin_top = _min_0;
            }
          }
          unsigned int warp_match_record = 0;
          if (lane_excl <= need && need < inclusive) {
            int chosen_scan_bin = scan_bin_base + bins_before;
            int bin_above = active_total - bin_top;
            int chosen_count = bin_top - bin_below;
            warp_match_record = (unsigned int)(chosen_scan_bin & 255) |
                                (unsigned int)(bin_above & 1023) << 8 |
                                (unsigned int)(chosen_count & 1023) << 18;
          }
          unsigned int _warp_redux_u32_2;
          asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;"
                       : "=r"(_warp_redux_u32_2)
                       : "r"(warp_match_record));
          unsigned int match_record = _warp_redux_u32_2;
          int chosen_bin = (int)(match_record & 255);
          remaining -= (int)(match_record >> 8 & 1023);
          active_total = (int)(match_record >> 18 & 1023);
          threshold |= (unsigned int)chosen_bin << (unsigned int)shift;
          examined_mask |= (unsigned int)255 << (unsigned int)shift;
          if (remaining == active_total) {
            take_all_equals = 1;
            break;
          }
          if (shift == 0) {
            break;
          }
          shift -= 8;
          if (shift < 0) {
            shift = 0;
          }
        }
      }
    }
    unsigned int selected_mask = 0;
    if (take_all_equals != 0) {
#pragma unroll
      for (int item_fast = 0; item_fast < 28; item_fast++) {
        if (threshold <= (keys[item_fast] & examined_mask)) {
          selected_mask |= one_u32 << (unsigned int)item_fast;
        }
      }
    } else {
      int tie_prior = 0;
#pragma unroll
      for (int item_tie = 0; item_tie < 28; item_tie++) {
        int tie_active = 0;
        if ((keys[item_tie] & examined_mask) == threshold) {
          tie_active = 1;
        }
        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, tie_active != 0);
        unsigned int tie_ballot = _vote_0;
        int _popc_0 = __popc(tie_ballot & lanemask_lt);
        int tie_rank = tie_prior + _popc_0;
        if (threshold < (keys[item_tie] & examined_mask)) {
          selected_mask |= one_u32 << (unsigned int)item_tie;
        }
        if (tie_active != 0) {
          if (tie_rank < remaining) {
            selected_mask |= one_u32 << (unsigned int)item_tie;
          }
        }
        int _popc_1 = __popc(tie_ballot);
        tie_prior += _popc_1;
      }
    }
    int compact_prior = 0;
#pragma unroll
    for (int item_compact = 0; item_compact < 28; item_compact++) {
      int chosen = (int)(selected_mask >> (unsigned int)item_compact & 1);
      unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, chosen != 0);
      unsigned int compact_ballot = _vote_1;
      if (chosen != 0) {
        int _popc_2 = __popc(compact_ballot & lanemask_lt);
        int slot = compact_prior + _popc_2;
        int winner_col = item_compact * 32 + lane;
        winner_ids[stash_base + slot] = winner_col;
      }
      int _popc_3 = __popc(compact_ballot);
      compact_prior += _popc_3;
    }
    __syncwarp();
    if (lane < TOP_K) {
      int score_col = winner_ids[stash_base + lane];
      float score_logit = reinterpret_cast<const volatile float *>(
          logits)[row_base + (unsigned long long)score_col];
      float _expf_1 = __expf(-score_logit);
      float _fdiv_full_1;
      asm volatile("div.full.f32 %0, %1, %2;"
                   : "=f"(_fdiv_full_1)
                   : "f"(1.0f), "f"(1.0f + _expf_1));
      winner_scores[stash_base + lane] = _fdiv_full_1;
    }
    __syncwarp();
    if (lane < TOP_K) {
      float selected_sum = 0.0f;
#pragma unroll
      for (int route_sum = 0; route_sum < TOP_K; route_sum++) {
        selected_sum += winner_scores[stash_base + route_sum];
      }
      float selected_norm = 1.0f;
      if (selected_sum > 0.0f) {
        selected_norm = selected_sum;
      }
      int selected_id = winner_ids[stash_base + lane];
      unsigned long long output_index =
          (unsigned long long)token * (unsigned long long)TOP_K +
          (unsigned long long)lane;
      float _fdiv_full_2;
      asm volatile("div.full.f32 %0, %1, %2;"
                   : "=f"(_fdiv_full_2)
                   : "f"(winner_scores[stash_base + lane]), "f"(selected_norm));
      topk_weights[output_index] = _fdiv_full_2;
      topk_ids[output_index] = selected_id;
    }
    __syncwarp();
  }
}

void Run(TensorView logits, TensorView bias, TensorView weights, TensorView ids,
         int64_t max_blocks) {
  CHECK_INPUT_AND_TYPE(logits, dl_float32);
  CHECK_INPUT_AND_TYPE(bias, dl_float32);
  CHECK_INPUT_AND_TYPE(weights, dl_float32);
  CHECK_INPUT_AND_TYPE(ids, dl_int32);
  CHECK_DEVICE(logits, bias);
  CHECK_DEVICE(logits, weights);
  CHECK_DEVICE(logits, ids);
  TVM_FFI_CHECK(logits.ndim() == 2 && logits.size(1) == NUM_EXPERTS &&
                    bias.ndim() == 1 && bias.size(0) == NUM_EXPERTS,
                ValueError)
      << "Kimi-K3 routing expects logits [tokens, 896] and bias [896]";
  const int64_t rows = logits.size(0);
  TVM_FFI_CHECK(weights.ndim() == 2 && weights.size(0) == rows &&
                    weights.size(1) == TOP_K && ids.ndim() == 2 &&
                    ids.size(0) == rows && ids.size(1) == TOP_K,
                ValueError)
      << "Kimi-K3 routing outputs must have shape [tokens, 16]";
  if (rows == 0)
    return;
  const int64_t needed_blocks = (rows + NUM_WARPS - 1) / NUM_WARPS;
  const int blocks =
      static_cast<int>(needed_blocks < max_blocks ? needed_blocks : max_blocks);
  const DLDevice device = logits.device();
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  const auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  cuda_kimi3_sigmoid_bias_topk<<<blocks, THREADS, 0, stream>>>(
      static_cast<const float *>(logits.data_ptr()),
      static_cast<const float *>(bias.data_ptr()),
      static_cast<float *>(weights.data_ptr()),
      static_cast<int *>(ids.data_ptr()), rows);
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << cudaGetErrorString(status);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
