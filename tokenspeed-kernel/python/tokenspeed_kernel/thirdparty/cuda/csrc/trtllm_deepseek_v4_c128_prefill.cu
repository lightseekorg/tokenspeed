/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from TensorRT-LLM commit
 * 54efe57e41f9742d915da977e7701c5cff2b7d6c, including the C128 prefill
 * reduction optimization from 047a3738523013263e9c0b81f5ceaf7a8e81138f:
 *   cpp/tensorrt_llm/kernels/compressorKernels/compressorKernels.cu
 *
 * This source keeps the TensorRT-LLM online-softmax reduction structure while
 * specializing it for TokenSpeed's DeepSeek-V4 C128 combined state and paged
 * fp8_ds_mla cache layouts. TensorRT-LLM framework dispatch and cache wrappers
 * are replaced by a TVM-FFI entry point.
 */

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kCompressRatio = 128;
constexpr int kHeadDim = 512;
constexpr int kHeadBlockDim = 128;
constexpr int kHeadBlocks = kHeadDim / kHeadBlockDim;
constexpr int kStateWidth = kHeadDim;
constexpr int kStateRowWidth = 2 * kStateWidth;
constexpr int kReductionGroups = 4;
constexpr int kWarpSize = 32;
constexpr int kVec = 4;
constexpr int kReductionThreads = kReductionGroups * kWarpSize;
constexpr int kReductionSmemBytes =
    3 * kReductionGroups * kHeadBlockDim * sizeof(float);

constexpr int kRopeDim = 64;
constexpr int kNopeDim = kHeadDim - kRopeDim;
constexpr int kFp8QuantBlock = 64;
constexpr int kFp8ScaleDim = kNopeDim / kFp8QuantBlock + 1;
constexpr int kCacheTokenStride = kNopeDim + 2 * kRopeDim;
constexpr int kPostprocessWarps = 4;
constexpr int kPostprocessThreads = kPostprocessWarps * kWarpSize;
constexpr float kFp8Max = 448.0f;
constexpr float kFp8MinAmax = 1.0e-4f;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// TensorRT-LLM's power-of-two scale rounding expressed without transcendental
// instructions. The positive input is clamped by the caller.
__device__ __forceinline__ int fast_log2_ceil(float value) {
  const uint32_t bits = __float_as_uint(value);
  const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  return exponent + ((bits & 0x007fffffu) != 0u ? 1 : 0);
}

__device__ __forceinline__ float fast_pow2(int exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
}

template <typename WeightT>
__device__ __forceinline__ float load_weight(
    const WeightT* __restrict__ weights,
    int index) {
  return static_cast<float>(weights[index]);
}

template <>
__device__ __forceinline__ float load_weight<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ weights,
    int index) {
  return __bfloat162float(weights[index]);
}

struct C128Boundary {
  int token;
  int64_t position;
  bool valid;
};

__device__ __forceinline__ C128Boundary c128_boundary_for_output(
    const int64_t* __restrict__ positions,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    int request_idx,
    int output_idx) {
  const int query_start = query_start_loc[request_idx];
  const int query_end = query_start_loc[request_idx + 1];
  if (query_start >= query_end) {
    return {-1, 0, false};
  }

  const int64_t start_position = positions[query_start];
  const int64_t first_window = start_position / kCompressRatio;
  const int actual_outputs =
      seq_lens[request_idx] / kCompressRatio -
      static_cast<int>(first_window);
  if (output_idx >= actual_outputs) {
    return {-1, 0, false};
  }

  const int64_t boundary_position =
      (first_window + output_idx + 1) * kCompressRatio - 1;
  const int64_t boundary_token =
      static_cast<int64_t>(query_start) + boundary_position - start_position;
  if (boundary_token < query_start || boundary_token >= query_end) {
    return {-1, 0, false};
  }
  return {static_cast<int>(boundary_token), boundary_position, true};
}

// One CTA computes one 128-wide slice of one C128 output. Four warps reduce
// disjoint 32-row partitions and merge their online-softmax states in 6 KiB of
// shared memory. The result overwrites only scratch[boundary, :512].
__global__ __launch_bounds__(kReductionThreads)
void c128_prefill_reduction_kernel(
    const float* __restrict__ state_cache,
    float* __restrict__ scratch,
    int64_t scratch_row_stride,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ compressor_slot_mapping,
    const int64_t* __restrict__ kv_slot_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int block_table_width,
    const int32_t* __restrict__ block_table_base_offsets,
    bool has_base_offsets,
    int state_block_size,
    int max_outputs) {
  const int request_idx = static_cast<int>(blockIdx.x);
  const int output_idx = static_cast<int>(blockIdx.y);
  const int head_block = static_cast<int>(blockIdx.z);
  if (output_idx >= max_outputs) {
    return;
  }

  const C128Boundary boundary = c128_boundary_for_output(
      positions, query_start_loc, seq_lens, request_idx, output_idx);
  if (!boundary.valid || compressor_slot_mapping[boundary.token] < 0 ||
      kv_slot_mapping[boundary.token] < 0) {
    return;
  }

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int reduction_group = threadIdx.x / kWarpSize;
  const int dim = head_block * kHeadBlockDim + lane * kVec;
  const int base_logical_page =
      has_base_offsets ? block_table_base_offsets[request_idx] : 0;

  float running_max[kVec];
  float running_sum[kVec];
  float running_weighted_sum[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    running_max[i] = -INFINITY;
    running_sum[i] = 0.0f;
    running_weighted_sum[i] = 0.0f;
  }

  const int64_t window_start =
      boundary.position - (kCompressRatio - 1);
#pragma unroll
  for (int local_row = 0; local_row < kCompressRatio / kReductionGroups;
       ++local_row) {
    const int64_t position =
        window_start + reduction_group * (kCompressRatio / kReductionGroups) +
        local_row;
    const int table_index =
        static_cast<int>(position / state_block_size) - base_logical_page;

    int physical_block = -1;
    if (lane == 0 && table_index >= 0 && table_index < block_table_width) {
      physical_block =
          block_table[request_idx * block_table_width + table_index];
    }
    physical_block = __shfl_sync(0xffffffffu, physical_block, 0);
    if (physical_block < 0) {
      continue;
    }

    const int position_in_block = static_cast<int>(position % state_block_size);
    const int64_t row_offset =
        (static_cast<int64_t>(physical_block) * state_block_size +
         position_in_block) *
        kStateRowWidth;
    const float4 kv = *reinterpret_cast<const float4*>(
        state_cache + row_offset + dim);
    const float4 score = *reinterpret_cast<const float4*>(
        state_cache + row_offset + kStateWidth + dim);
    const float kv_values[kVec] = {kv.x, kv.y, kv.z, kv.w};
    const float score_values[kVec] = {score.x, score.y, score.z, score.w};

#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float new_max = fmaxf(running_max[i], score_values[i]);
      const float previous_scale = expf(running_max[i] - new_max);
      const float current_scale = expf(score_values[i] - new_max);
      running_sum[i] = running_sum[i] * previous_scale + current_scale;
      running_weighted_sum[i] =
          running_weighted_sum[i] * previous_scale +
          kv_values[i] * current_scale;
      running_max[i] = new_max;
    }
  }

  extern __shared__ float shared[];
  float* shared_max = shared;
  float* shared_sum =
      shared_max + kReductionGroups * kHeadBlockDim;
  float* shared_weighted_sum =
      shared_sum + kReductionGroups * kHeadBlockDim;
  const int shared_offset =
      reduction_group * kHeadBlockDim + lane * kVec;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    shared_max[shared_offset + i] = running_max[i];
    shared_sum[shared_offset + i] = running_sum[i];
    shared_weighted_sum[shared_offset + i] = running_weighted_sum[i];
  }
  __syncthreads();

  if (reduction_group != 0) {
    return;
  }

#pragma unroll
  for (int group = 1; group < kReductionGroups; ++group) {
    const int group_offset = group * kHeadBlockDim + lane * kVec;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float other_max = shared_max[group_offset + i];
      const float other_sum = shared_sum[group_offset + i];
      if (other_sum == 0.0f) {
        continue;
      }
      if (running_sum[i] == 0.0f) {
        running_max[i] = other_max;
        running_sum[i] = other_sum;
        running_weighted_sum[i] = shared_weighted_sum[group_offset + i];
        continue;
      }
      const float new_max = fmaxf(running_max[i], other_max);
      const float previous_scale = expf(running_max[i] - new_max);
      const float other_scale = expf(other_max - new_max);
      running_sum[i] =
          running_sum[i] * previous_scale +
          other_sum * other_scale;
      running_weighted_sum[i] =
          running_weighted_sum[i] * previous_scale +
          shared_weighted_sum[group_offset + i] * other_scale;
      running_max[i] = new_max;
    }
  }

  float4 output;
  output.x = running_weighted_sum[0] / running_sum[0];
  output.y = running_weighted_sum[1] / running_sum[1];
  output.z = running_weighted_sum[2] / running_sum[2];
  output.w = running_weighted_sum[3] / running_sum[3];
  *reinterpret_cast<float4*>(
      scratch + static_cast<int64_t>(boundary.token) * scratch_row_stride +
      dim) = output;
}

template <typename WeightT>
__global__ __launch_bounds__(kPostprocessThreads)
void c128_prefill_postprocess_scatter_kernel(
    const float* __restrict__ scratch,
    int64_t scratch_row_stride,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ compressor_slot_mapping,
    const int64_t* __restrict__ kv_slot_mapping,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ seq_lens,
    const WeightT* __restrict__ rms_weight,
    float rms_norm_eps,
    const float* __restrict__ cos_sin_cache,
    uint8_t* __restrict__ kv_cache,
    int64_t kv_cache_page_stride,
    int kv_block_size,
    int max_outputs) {
  const int request_idx = static_cast<int>(blockIdx.x);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int output_idx =
      static_cast<int>(blockIdx.y) * kPostprocessWarps + warp_in_block;
  if (output_idx >= max_outputs) {
    return;
  }

  const C128Boundary boundary = c128_boundary_for_output(
      positions, query_start_loc, seq_lens, request_idx, output_idx);
  if (!boundary.valid || compressor_slot_mapping[boundary.token] < 0) {
    return;
  }
  const int64_t kv_slot = kv_slot_mapping[boundary.token];
  if (kv_slot < 0) {
    return;
  }

  constexpr int kElementsPerLane = kHeadDim / kWarpSize;
  float values[kElementsPerLane];
  float square_sum = 0.0f;
  const int lane_start = lane * kElementsPerLane;
#pragma unroll
  for (int i = 0; i < kElementsPerLane; ++i) {
    const float value =
        scratch[static_cast<int64_t>(boundary.token) * scratch_row_stride +
                lane_start + i];
    values[i] = value;
    square_sum += value * value;
  }
  square_sum = warp_reduce_sum(square_sum);
  const float rms_scale =
      rsqrtf(square_sum / static_cast<float>(kHeadDim) + rms_norm_eps);

#pragma unroll
  for (int i = 0; i < kElementsPerLane; ++i) {
    values[i] *= rms_scale * load_weight(rms_weight, lane_start + i);
  }

  const int64_t physical_page = kv_slot / kv_block_size;
  const int position_in_page = static_cast<int>(kv_slot % kv_block_size);
  uint8_t* const page_base =
      kv_cache + physical_page * kv_cache_page_stride;
  uint8_t* const value_base =
      page_base + position_in_page * kCacheTokenStride;
  uint8_t* const scale_base =
      page_base + kv_block_size * kCacheTokenStride +
      position_in_page * kFp8ScaleDim;

  // The 448 NoPE values are quantized in seven 64-value subwarp groups.
  if (lane_start < kNopeDim) {
    float local_amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kElementsPerLane; ++i) {
      const float rounded = __bfloat162float(__float2bfloat16_rn(values[i]));
      local_amax = fmaxf(local_amax, fabsf(rounded));
    }
    constexpr unsigned int kNopeActiveMask = 0x0fffffffu;
    local_amax = fmaxf(
        local_amax,
        __shfl_xor_sync(kNopeActiveMask, local_amax, 2, 4));
    local_amax = fmaxf(
        local_amax,
        __shfl_xor_sync(kNopeActiveMask, local_amax, 1, 4));
    local_amax = fmaxf(local_amax, kFp8MinAmax);

    const int scale_exponent = fast_log2_ceil(local_amax / kFp8Max);
    const float inverse_scale = 1.0f / fast_pow2(scale_exponent);
    alignas(16) uint8_t fp8_bytes[kElementsPerLane];
#pragma unroll
    for (int i = 0; i < kElementsPerLane; ++i) {
      const float rounded = __bfloat162float(__float2bfloat16_rn(values[i]));
      const float scaled =
          fminf(fmaxf(rounded * inverse_scale, -kFp8Max), kFp8Max);
      const __nv_fp8_e4m3 fp8_value(scaled);
      fp8_bytes[i] = *reinterpret_cast<const uint8_t*>(&fp8_value);
    }
    *reinterpret_cast<uint4*>(value_base + lane_start) =
        *reinterpret_cast<const uint4*>(fp8_bytes);

    if ((lane & 3) == 0) {
      const int encoded_exponent =
          max(0, min(255, scale_exponent + 127));
      scale_base[lane / 4] = static_cast<uint8_t>(encoded_exponent);
    }
  }

  // RoPE consumes the original FP32 RMSNorm result, not the BF16-rounded
  // quantization input. One lane owns one interleaved pair.
  const int rope_pair = lane;
  const int rope_even_dim = kNopeDim + 2 * rope_pair;
  const float even =
      scratch[static_cast<int64_t>(boundary.token) * scratch_row_stride +
              rope_even_dim] *
      rms_scale * load_weight(rms_weight, rope_even_dim);
  const float odd =
      scratch[static_cast<int64_t>(boundary.token) * scratch_row_stride +
              rope_even_dim + 1] *
      rms_scale * load_weight(rms_weight, rope_even_dim + 1);
  const int64_t compressed_position =
      (boundary.position / kCompressRatio) * kCompressRatio;
  const float* const cos_sin_row =
      cos_sin_cache + compressed_position * kRopeDim;
  const float cosine = cos_sin_row[rope_pair];
  const float sine = cos_sin_row[kRopeDim / 2 + rope_pair];
  const __nv_bfloat162 rotated = __floats2bfloat162_rn(
      even * cosine - odd * sine,
      odd * cosine + even * sine);
  reinterpret_cast<__nv_bfloat162*>(value_base + kNopeDim)[rope_pair] =
      rotated;

  if (lane == 0) {
    scale_base[kFp8ScaleDim - 1] = 0;
  }
}

void validate_common_tensor(
    const TensorView& reference,
    const TensorView& tensor,
    const char* name) {
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  TVM_FFI_ICHECK_EQ(reference.device().device_type, tensor.device().device_type)
      << name << " must be on the same device as state_cache";
  TVM_FFI_ICHECK_EQ(reference.device().device_id, tensor.device().device_id)
      << name << " must be on the same device as state_cache";
}

}  // namespace

void trtllm_deepseek_v4_c128_prefill_compress_cache(
    TensorView state_cache,
    TensorView scratch,
    TensorView positions,
    TensorView compressor_slot_mapping,
    TensorView query_start_loc,
    TensorView seq_lens,
    TensorView block_table,
    TensorView block_table_base_offsets,
    bool has_base_offsets,
    TensorView rms_norm_weight,
    TensorView cos_sin_cache,
    TensorView kv_cache,
    TensorView kv_slot_mapping,
    int64_t state_block_size,
    int64_t kv_block_size,
    int64_t max_outputs,
    double rms_norm_eps) {
  CHECK_CUDA(state_cache);
  CHECK_CONTIGUOUS(state_cache);
  validate_common_tensor(state_cache, scratch, "scratch");
  validate_common_tensor(state_cache, positions, "positions");
  validate_common_tensor(
      state_cache, compressor_slot_mapping, "compressor_slot_mapping");
  validate_common_tensor(state_cache, query_start_loc, "query_start_loc");
  validate_common_tensor(state_cache, seq_lens, "seq_lens");
  validate_common_tensor(state_cache, block_table, "block_table");
  validate_common_tensor(
      state_cache, block_table_base_offsets, "block_table_base_offsets");
  validate_common_tensor(state_cache, rms_norm_weight, "rms_norm_weight");
  validate_common_tensor(state_cache, cos_sin_cache, "cos_sin_cache");
  validate_common_tensor(state_cache, kv_cache, "kv_cache");
  validate_common_tensor(state_cache, kv_slot_mapping, "kv_slot_mapping");

  CHECK_DIM(3, state_cache);
  CHECK_DIM(2, scratch);
  CHECK_DIM(1, positions);
  CHECK_DIM(1, compressor_slot_mapping);
  CHECK_DIM(1, query_start_loc);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(2, block_table);
  CHECK_DIM(1, rms_norm_weight);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_DIM(2, kv_cache);
  CHECK_DIM(1, kv_slot_mapping);
  CHECK_INPUT_TYPE(state_cache, dl_float32);
  CHECK_INPUT_TYPE(scratch, dl_float32);
  CHECK_INPUT_TYPE(positions, dl_int64);
  CHECK_INPUT_TYPE(compressor_slot_mapping, dl_int64);
  CHECK_INPUT_TYPE(query_start_loc, dl_int32);
  CHECK_INPUT_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_TYPE(block_table, dl_int32);
  TVM_FFI_ICHECK(
      rms_norm_weight.dtype() == dl_bfloat16 ||
      rms_norm_weight.dtype() == dl_float32)
      << "rms_norm_weight must be bfloat16 or float32";
  CHECK_INPUT_TYPE(cos_sin_cache, dl_float32);
  CHECK_INPUT_TYPE(kv_cache, dl_uint8);
  CHECK_INPUT_TYPE(kv_slot_mapping, dl_int64);

  TVM_FFI_ICHECK_GT(state_block_size, 0);
  TVM_FFI_ICHECK_LE(state_block_size, INT_MAX);
  TVM_FFI_ICHECK_GT(kv_block_size, 0);
  TVM_FFI_ICHECK_LE(kv_block_size, INT_MAX);
  TVM_FFI_ICHECK_GT(max_outputs, 0);
  TVM_FFI_ICHECK_LE(max_outputs, 65535);
  TVM_FFI_ICHECK_EQ(state_cache.size(1), state_block_size);
  TVM_FFI_ICHECK_EQ(state_cache.size(2), kStateRowWidth);
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(state_cache.data_ptr()) % alignof(float4), 0)
      << "state_cache must be 16-byte aligned";
  TVM_FFI_ICHECK_GE(scratch.size(1), kHeadDim);
  TVM_FFI_ICHECK_EQ(scratch.stride(0) % kVec, 0)
      << "scratch row stride must preserve 16-byte alignment";
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(scratch.data_ptr()) % alignof(float4), 0)
      << "scratch must be 16-byte aligned";
  TVM_FFI_ICHECK_EQ(positions.size(0), scratch.size(0));
  TVM_FFI_ICHECK_EQ(compressor_slot_mapping.size(0), scratch.size(0));
  TVM_FFI_ICHECK_EQ(kv_slot_mapping.size(0), scratch.size(0));
  TVM_FFI_ICHECK_LE(scratch.size(0), INT_MAX);
  const int64_t batch_size = seq_lens.size(0);
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_LE(batch_size, INT_MAX);
  TVM_FFI_ICHECK_EQ(query_start_loc.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(block_table.size(0), batch_size);
  TVM_FFI_ICHECK_GT(block_table.size(1), 0);
  TVM_FFI_ICHECK_LE(block_table.size(1), INT_MAX);
  if (has_base_offsets) {
    CHECK_DIM(1, block_table_base_offsets);
    CHECK_INPUT_TYPE(block_table_base_offsets, dl_int32);
    TVM_FFI_ICHECK_EQ(block_table_base_offsets.size(0), batch_size);
  }
  TVM_FFI_ICHECK_EQ(rms_norm_weight.size(0), kHeadDim);
  TVM_FFI_ICHECK_GT(cos_sin_cache.size(0), 0);
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1), kRopeDim);
  TVM_FFI_ICHECK_GE(
      kv_cache.size(1),
      kv_block_size * (kCacheTokenStride + kFp8ScaleDim));
  TVM_FFI_ICHECK_EQ(kv_cache.stride(0) % alignof(uint4), 0)
      << "kv_cache page stride must preserve 16-byte alignment";
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(kv_cache.data_ptr()) % alignof(uint4), 0)
      << "kv_cache must be 16-byte aligned";
  TVM_FFI_ICHECK_GE(rms_norm_eps, 0.0);

  const int device_id = state_cache.device().device_id;
  cudaError_t status = cudaSetDevice(device_id);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "failed to select CUDA device: " << cudaGetErrorString(status);
  int major = 0;
  status = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device_id);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "failed to query CUDA capability: " << cudaGetErrorString(status);
  TVM_FFI_ICHECK_EQ(major, 10)
      << "DeepSeek-V4 C128 prefill compressor requires an SM100-family GPU";

  const cudaStream_t stream = get_stream(state_cache.device());
  const dim3 reduction_grid(
      static_cast<unsigned int>(batch_size),
      static_cast<unsigned int>(max_outputs),
      kHeadBlocks);
  c128_prefill_reduction_kernel<<<
      reduction_grid,
      kReductionThreads,
      kReductionSmemBytes,
      stream>>>(
      static_cast<const float*>(state_cache.data_ptr()),
      static_cast<float*>(scratch.data_ptr()),
      scratch.stride(0),
      static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<const int64_t*>(compressor_slot_mapping.data_ptr()),
      static_cast<const int64_t*>(kv_slot_mapping.data_ptr()),
      static_cast<const int32_t*>(query_start_loc.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const int32_t*>(block_table.data_ptr()),
      static_cast<int>(block_table.size(1)),
      static_cast<const int32_t*>(block_table_base_offsets.data_ptr()),
      has_base_offsets,
      static_cast<int>(state_block_size),
      static_cast<int>(max_outputs));
  status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "DeepSeek-V4 C128 prefill reduction launch failed: "
      << cudaGetErrorString(status);

  const dim3 postprocess_grid(
      static_cast<unsigned int>(batch_size),
      static_cast<unsigned int>(
          (max_outputs + kPostprocessWarps - 1) / kPostprocessWarps));
#define LAUNCH_POSTPROCESS(WeightT)                                           \
  c128_prefill_postprocess_scatter_kernel<WeightT>                           \
      <<<postprocess_grid, kPostprocessThreads, 0, stream>>>(                 \
          static_cast<const float*>(scratch.data_ptr()),                     \
          scratch.stride(0),                                                 \
          static_cast<const int64_t*>(positions.data_ptr()),                 \
          static_cast<const int64_t*>(compressor_slot_mapping.data_ptr()),   \
          static_cast<const int64_t*>(kv_slot_mapping.data_ptr()),           \
          static_cast<const int32_t*>(query_start_loc.data_ptr()),           \
          static_cast<const int32_t*>(seq_lens.data_ptr()),                  \
          static_cast<const WeightT*>(rms_norm_weight.data_ptr()),           \
          static_cast<float>(rms_norm_eps),                                  \
          static_cast<const float*>(cos_sin_cache.data_ptr()),               \
          static_cast<uint8_t*>(kv_cache.data_ptr()),                        \
          kv_cache.stride(0),                                                \
          static_cast<int>(kv_block_size),                                   \
          static_cast<int>(max_outputs))

  if (rms_norm_weight.dtype() == dl_bfloat16) {
    LAUNCH_POSTPROCESS(__nv_bfloat16);
  } else {
    LAUNCH_POSTPROCESS(float);
  }
#undef LAUNCH_POSTPROCESS

  status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "DeepSeek-V4 C128 postprocess/scatter launch failed: "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    trtllm_deepseek_v4_c128_prefill_compress_cache,
    trtllm_deepseek_v4_c128_prefill_compress_cache);
