/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * 1a4fc3e05cea6e2843db8ba7a14e0c3a1a2a7018:
 *   cpp/tensorrt_llm/kernels/mlaKernels.cu
 *   cpp/tensorrt_llm/kernels/fusedCatFp4.cu
 *
 * The kernel math and launch layouts are preserved.  TensorRT-LLM framework
 * checks and launch helpers are replaced by TokenSpeed's TVM-FFI boundary.
 */

#include <cmath>
#include <climits>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kIndexerHeadDim = 128;
constexpr int kIndexerRopeDim = 64;
constexpr int kIndexerNopeDim = kIndexerHeadDim - kIndexerRopeDim;

// TensorRT-LLM's MLA RoPE kernel uses 16-byte vector loads and groups up to
// sixteen heads in one CTA.  DeepSeek-V4 uses interleaved/GPT-J RoPE.
constexpr int kRopeBytesPerLoad = 16;
constexpr int kRopeElemsPerLoad = kRopeBytesPerLoad / sizeof(__nv_bfloat16);
constexpr int kRopeHeadsPerBlock = 16;
constexpr int kRopeThreadsPerHead = kIndexerRopeDim / kRopeElemsPerLoad;

union Bf16x8 {
  uint4 vec;
  __nv_bfloat162 pairs[kRopeElemsPerLoad / 2];
};

inline __device__ float2 rotary_embedding_transform(
    const float2 value,
    const float2 coefficient) {
  float2 rotated;
  rotated.x =
      coefficient.x * value.x - coefficient.y * value.y;
  rotated.y =
      coefficient.x * value.y + coefficient.y * value.x;
  return rotated;
}

inline __device__ __nv_bfloat162 rotary_embedding_transform(
    const __nv_bfloat162 value,
    const float2 coefficient) {
  const float2 float_value = __bfloat1622float2(value);
  const float2 rotated = rotary_embedding_transform(float_value, coefficient);
  return __floats2bfloat162_rn(rotated.x, rotated.y);
}

template <typename PositionT>
__global__ void mla_rope_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ data,
    const PositionT* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache,
    int num_heads) {
  const int head_idx = blockIdx.y * kRopeHeadsPerBlock + threadIdx.y;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  if (head_idx >= num_heads || threadIdx.x >= kRopeThreadsPerHead) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  __nv_bfloat16* head_ptr =
      data + (static_cast<int64_t>(blockIdx.x) * num_heads + head_idx) *
                 kIndexerHeadDim;
  const int position = static_cast<int>(position_ids[blockIdx.x]);
  const int elem_offset = threadIdx.x * kRopeElemsPerLoad;
  const int pair_offset = elem_offset / 2;
  const float* cos_ptr =
      cos_sin_cache + static_cast<int64_t>(position) * kIndexerRopeDim +
      pair_offset;
  const float* sin_ptr = cos_ptr + kIndexerRopeDim / 2;

  Bf16x8 values;
  values.vec = *reinterpret_cast<const uint4*>(
      head_ptr + kIndexerNopeDim + elem_offset);

#pragma unroll
  for (int pair_idx = 0; pair_idx < kRopeElemsPerLoad / 2; ++pair_idx) {
    const float2 coefficient{cos_ptr[pair_idx], sin_ptr[pair_idx]};
    values.pairs[pair_idx] =
        rotary_embedding_transform(values.pairs[pair_idx], coefficient);
  }

  *reinterpret_cast<uint4*>(head_ptr + kIndexerNopeDim + elem_offset) =
      values.vec;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

constexpr int kWarpSize = 32;
constexpr int kElemsPerThread = 4;
constexpr int kRowsPerBlock = 8;
constexpr int kFp4BlockSize = 32;
constexpr int kLanesPerFp4Block = kFp4BlockSize / kElemsPerThread;
constexpr float kInvFp4E2m1Max = 1.0f / 6.0f;
// TRT-LLM's source uses 1e-12, but its stated DeepGEMM reference and
// TokenSpeed's existing Q-preparation contract both clamp at 1e-4. Preserve
// that runtime contract across the token-count dispatch boundary.
constexpr float kMinAmax = 1.0e-4f;

union Bf16x4 {
  int2 vec;
  __nv_bfloat162 pairs[2];
};

__device__ __forceinline__ uint32_t quantize_fp4_e2m1(float scaled) {
  const float absolute = fminf(fabsf(scaled), 6.0f);
  const uint32_t magnitude = static_cast<uint32_t>(
      (absolute > 0.25f) + (absolute > 0.75f) + (absolute > 1.25f) +
      (absolute > 1.75f) + (absolute > 2.5f) + (absolute > 3.5f) +
      (absolute > 5.0f));
  const uint32_t sign = (scaled < 0.0f && magnitude != 0u) ? 1u : 0u;
  return (magnitude & 0x7u) | (sign << 3);
}

// Direct adaptation of TensorRT-LLM's fusedCatFp4Kernel.  Each warp owns one
// 128-element row.  Inputs may be non-contiguous split views as long as their
// innermost dimensions are contiguous.
__global__ __launch_bounds__(kWarpSize * kRowsPerBlock)
void fused_cat_fp4_kernel(
    uint8_t* __restrict__ packed_out,
    int32_t* __restrict__ scale_out,
    const __nv_bfloat16* __restrict__ first,
    const __nv_bfloat16* __restrict__ second,
    int rows,
    int first_dim,
    int first_row_stride,
    int second_row_stride) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kRowsPerBlock + warp_in_block;
  if (row >= rows) {
    return;
  }

  const int input_offset = lane * kElemsPerThread;
  const bool from_first = input_offset < first_dim;
  const __nv_bfloat16* input_row =
      (from_first ? first + static_cast<int64_t>(row) * first_row_stride
                  : second + static_cast<int64_t>(row) * second_row_stride);
  const int input_column =
      from_first ? input_offset : input_offset - first_dim;

  Bf16x4 loaded;
  loaded.vec = *reinterpret_cast<const int2*>(input_row + input_column);
  const float2 values01 = __bfloat1622float2(loaded.pairs[0]);
  const float2 values23 = __bfloat1622float2(loaded.pairs[1]);

  float amax = fmaxf(
      fmaxf(fabsf(values01.x), fabsf(values01.y)),
      fmaxf(fabsf(values23.x), fabsf(values23.y)));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, kMinAmax);

  const float ratio = amax * kInvFp4E2m1Max;
  const uint32_t ratio_bits = __float_as_uint(ratio);
  uint32_t exponent_bits = ratio_bits & 0x7f800000u;
  if ((ratio_bits & 0x007fffffu) != 0u) {
    exponent_bits += 0x00800000u;
  }
  const float scale = __uint_as_float(exponent_bits);

  const uint32_t code0 = quantize_fp4_e2m1(values01.x / scale);
  const uint32_t code1 = quantize_fp4_e2m1(values01.y / scale);
  const uint32_t code2 = quantize_fp4_e2m1(values23.x / scale);
  const uint32_t code3 = quantize_fp4_e2m1(values23.y / scale);

  const int output_offset = row * (kIndexerHeadDim / 2) + lane * 2;
  packed_out[output_offset] =
      static_cast<uint8_t>(code0 | (code1 << 4));
  packed_out[output_offset + 1] =
      static_cast<uint8_t>(code2 | (code3 << 4));

  const uint32_t exponent = (__float_as_uint(scale) >> 23) & 0xffu;
  const uint32_t exponent0 =
      __shfl_sync(0xffffffffu, exponent, 0 * kLanesPerFp4Block);
  const uint32_t exponent1 =
      __shfl_sync(0xffffffffu, exponent, 1 * kLanesPerFp4Block);
  const uint32_t exponent2 =
      __shfl_sync(0xffffffffu, exponent, 2 * kLanesPerFp4Block);
  const uint32_t exponent3 =
      __shfl_sync(0xffffffffu, exponent, 3 * kLanesPerFp4Block);
  if (lane == 0) {
    scale_out[row] = static_cast<int32_t>(
        exponent0 | (exponent1 << 8) | (exponent2 << 16) |
        (exponent3 << 24));
  }
}

template <typename PositionT>
cudaError_t launch_mla_rope_inplace(
    __nv_bfloat16* data,
    const PositionT* position_ids,
    const float* cos_sin_cache,
    int num_tokens,
    int num_heads,
    bool enable_pdl,
    cudaStream_t stream) {
  if (num_tokens == 0) {
    return cudaSuccess;
  }
  const dim3 grid(
      num_tokens,
      (num_heads + kRopeHeadsPerBlock - 1) / kRopeHeadsPerBlock);
  const dim3 block(kRopeThreadsPerHead, kRopeHeadsPerBlock);
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.attrs = attrs;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(
      &config,
      mla_rope_inplace_bf16_kernel<PositionT>,
      data,
      position_ids,
      cos_sin_cache,
      num_heads);
}

}  // namespace

void trtllm_deepseek_v4_mla_rope_inplace(
    TensorView data,
    TensorView position_ids,
    TensorView cos_sin_cache,
    bool enable_pdl) {
  CHECK_CUDA(data);
  CHECK_CUDA(position_ids);
  CHECK_CUDA(cos_sin_cache);
  CHECK_DEVICE(data, position_ids);
  CHECK_DEVICE(data, cos_sin_cache);
  CHECK_DIM(3, data);
  CHECK_DIM(1, position_ids);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_INPUT_TYPE(data, dl_bfloat16);
  CHECK_INPUT_TYPE(cos_sin_cache, dl_float32);
  TVM_FFI_ICHECK(
      position_ids.dtype() == dl_int32 || position_ids.dtype() == dl_int64)
      << "position_ids must be int32 or int64";
  CHECK_CONTIGUOUS(data);
  CHECK_CONTIGUOUS(position_ids);
  CHECK_CONTIGUOUS(cos_sin_cache);
  TVM_FFI_ICHECK_EQ(data.size(2), kIndexerHeadDim)
      << "data head_dim must be 128";
  TVM_FFI_ICHECK_EQ(position_ids.size(0), data.size(0))
      << "position_ids must cover every token";
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1), kIndexerRopeDim)
      << "cos_sin_cache width must be 64";
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(data.data_ptr()) % kRopeBytesPerLoad, 0)
      << "data must be 16-byte aligned";

  cudaSetDevice(data.device().device_id);
  const cudaStream_t stream = get_stream(data.device());
  auto* data_ptr = static_cast<__nv_bfloat16*>(data.data_ptr());
  const auto* cos_sin_ptr = static_cast<const float*>(cos_sin_cache.data_ptr());
  const int num_tokens = static_cast<int>(data.size(0));
  const int num_heads = static_cast<int>(data.size(1));
  cudaError_t status;
  if (position_ids.dtype() == dl_int32) {
    status = launch_mla_rope_inplace(
        data_ptr,
        static_cast<const int32_t*>(position_ids.data_ptr()),
        cos_sin_ptr,
        num_tokens,
        num_heads,
        enable_pdl,
        stream);
  } else {
    status = launch_mla_rope_inplace(
        data_ptr,
        static_cast<const int64_t*>(position_ids.data_ptr()),
        cos_sin_ptr,
        num_tokens,
        num_heads,
        enable_pdl,
        stream);
  }

  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "TRT-LLM MLA RoPE launch failed: " << cudaGetErrorString(status);
  status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "TRT-LLM MLA RoPE launch state failed: "
      << cudaGetErrorString(status);
}

void trtllm_deepseek_v4_fused_cat_fp4(
    TensorView first,
    TensorView second,
    TensorView packed_out,
    TensorView scale_out) {
  CHECK_CUDA(first);
  CHECK_CUDA(second);
  CHECK_CUDA(packed_out);
  CHECK_CUDA(scale_out);
  CHECK_DEVICE(first, second);
  CHECK_DEVICE(first, packed_out);
  CHECK_DEVICE(first, scale_out);
  TVM_FFI_ICHECK_GE(first.ndim(), 2) << "first must be at least 2-D";
  TVM_FFI_ICHECK_GE(second.ndim(), 2) << "second must be at least 2-D";
  CHECK_DIM(2, packed_out);
  CHECK_DIM(2, scale_out);
  CHECK_INPUT_TYPE(first, dl_bfloat16);
  CHECK_INPUT_TYPE(second, dl_bfloat16);
  CHECK_INPUT_TYPE(packed_out, dl_uint8);
  CHECK_INPUT_TYPE(scale_out, dl_int32);
  TVM_FFI_ICHECK_EQ(first.stride(-1), 1)
      << "first innermost dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(second.stride(-1), 1)
      << "second innermost dimension must be contiguous";
  CHECK_CONTIGUOUS(packed_out);
  CHECK_CONTIGUOUS(scale_out);

  const int first_dim = static_cast<int>(first.size(-1));
  const int second_dim = static_cast<int>(second.size(-1));
  TVM_FFI_ICHECK_EQ(first_dim + second_dim, kIndexerHeadDim)
      << "input dimensions must concatenate to 128";
  TVM_FFI_ICHECK_EQ(first_dim % kElemsPerThread, 0)
      << "first dimension must be divisible by 4";
  const int64_t first_rows = first.numel() / first_dim;
  const int64_t second_rows = second.numel() / second_dim;
  TVM_FFI_ICHECK_EQ(first_rows, second_rows)
      << "inputs must contain the same number of rows";
  TVM_FFI_ICHECK_LE(first_rows, static_cast<int64_t>(INT32_MAX))
      << "row count exceeds int32 range";
  TVM_FFI_ICHECK_EQ(packed_out.size(0), first_rows);
  TVM_FFI_ICHECK_EQ(packed_out.size(1), kIndexerHeadDim / 2);
  TVM_FFI_ICHECK_EQ(scale_out.size(0), first_rows);
  TVM_FFI_ICHECK_EQ(scale_out.size(1), 1);
  int64_t first_expected_stride = first.stride(-2);
  int64_t second_expected_stride = second.stride(-2);
  if (first_rows > 0) {
    for (int axis = static_cast<int>(first.ndim()) - 2; axis >= 0; --axis) {
      if (first.size(axis) > 1) {
        TVM_FFI_ICHECK_EQ(first.stride(axis), first_expected_stride)
            << "first leading dimensions must be row-linear";
      }
      first_expected_stride *= first.size(axis);
    }
    for (int axis = static_cast<int>(second.ndim()) - 2; axis >= 0; --axis) {
      if (second.size(axis) > 1) {
        TVM_FFI_ICHECK_EQ(second.stride(axis), second_expected_stride)
            << "second leading dimensions must be row-linear";
      }
      second_expected_stride *= second.size(axis);
    }
  }
  TVM_FFI_ICHECK_GE(first.stride(-2), 0);
  TVM_FFI_ICHECK_LE(first.stride(-2), static_cast<int64_t>(INT32_MAX));
  TVM_FFI_ICHECK_GE(second.stride(-2), 0);
  TVM_FFI_ICHECK_LE(second.stride(-2), static_cast<int64_t>(INT32_MAX));
  TVM_FFI_ICHECK_EQ(first.stride(-2) % kElemsPerThread, 0)
      << "first row stride must preserve vector alignment";
  TVM_FFI_ICHECK_EQ(second.stride(-2) % kElemsPerThread, 0)
      << "second row stride must preserve vector alignment";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(first.data_ptr()) % 8, 0)
      << "first pointer must be 8-byte aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(second.data_ptr()) % 8, 0)
      << "second pointer must be 8-byte aligned";

  const int rows = static_cast<int>(first_rows);
  if (rows == 0) {
    return;
  }
  cudaSetDevice(first.device().device_id);
  const cudaStream_t stream = get_stream(first.device());
  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  fused_cat_fp4_kernel<<<blocks, kWarpSize * kRowsPerBlock, 0, stream>>>(
      static_cast<uint8_t*>(packed_out.data_ptr()),
      static_cast<int32_t*>(scale_out.data_ptr()),
      static_cast<const __nv_bfloat16*>(first.data_ptr()),
      static_cast<const __nv_bfloat16*>(second.data_ptr()),
      rows,
      first_dim,
      static_cast<int>(first.stride(-2)),
      static_cast<int>(second.stride(-2)));

  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "TRT-LLM fusedCatFp4 launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    trtllm_deepseek_v4_mla_rope_inplace,
    trtllm_deepseek_v4_mla_rope_inplace);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    trtllm_deepseek_v4_fused_cat_fp4,
    trtllm_deepseek_v4_fused_cat_fp4);
