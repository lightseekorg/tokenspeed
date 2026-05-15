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
//
// DeepSeek V4 fused SWA cache insert and sparse attention/indexer helpers.
//
// Cache layout per paged block:
//   [0, block_size * 576): token data, each token [448 fp8 bytes | 64 bf16/fp16]
//   [block_size * 576, block_size * 584): scale bytes, 8 per token

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kHalfRopeDim = kRopeDim / 2;
constexpr int kNopeDim = kHeadDim - kRopeDim;
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;
constexpr int kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr int kThreads = 256;
constexpr float kFp8Max = 448.0f;

template <int BlockYSize>
__global__ void gather_paged_indexer_mxfp4_cache_kernel(
    const uint8_t* __restrict__ kv_cache,
    uint8_t* __restrict__ values_out,
    uint8_t* __restrict__ scales_out,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    int batch_size,
    int num_tokens,
    int value_bytes,
    int scale_bytes,
    int cache_block_size,
    int64_t cache_block_stride,
    int64_t value_stride,
    int64_t scale_stride,
    int64_t block_table_stride) {
  constexpr int kVecBytes = sizeof(uint4);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * kVecBytes;

  __shared__ int batch_idx[BlockYSize];
  if (threadIdx.x == 0) {
    batch_idx[threadIdx.y] = -1;
  }
  __syncthreads();

  for (int iter = 0; iter < (batch_size + blockDim.x - 1) / blockDim.x;
       ++iter) {
    const int req = iter * blockDim.x + threadIdx.x;
    if (req < batch_size) {
      const int seq_start = cu_seq_lens[req];
      const int seq_end = cu_seq_lens[req + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = req;
      }
    }
  }
  __syncthreads();

  const int req = batch_idx[threadIdx.y];
  if (token_idx >= num_tokens || req < 0) {
    return;
  }

  const int in_req_token_idx = token_idx - cu_seq_lens[req];
  const int block_idx =
      block_table[static_cast<int64_t>(req) * block_table_stride +
                  in_req_token_idx / cache_block_size];
  const int block_offset = in_req_token_idx % cache_block_size;
  const int64_t block_base = static_cast<int64_t>(block_idx) * cache_block_stride;

  if (head_idx < value_bytes) {
    const int64_t value_src =
        block_base + static_cast<int64_t>(block_offset) * value_bytes + head_idx;
    const int64_t value_dst =
        static_cast<int64_t>(token_idx) * value_stride + head_idx;
    *reinterpret_cast<uint4*>(values_out + value_dst) =
        *reinterpret_cast<const uint4*>(kv_cache + value_src);
  }

  if (blockIdx.y == 0 && threadIdx.x == 0) {
    const int64_t scale_src =
        block_base + static_cast<int64_t>(cache_block_size) * value_bytes +
        static_cast<int64_t>(block_offset) * scale_bytes;
    const int64_t scale_dst = static_cast<int64_t>(token_idx) * scale_stride;
    *reinterpret_cast<uint32_t*>(scales_out + scale_dst) =
        *reinterpret_cast<const uint32_t*>(kv_cache + scale_src);
  }
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <>
__device__ __forceinline__ float scalar_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<nv_bfloat16>(nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <>
__device__ __forceinline__ half float_to_scalar<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 float_to_scalar<nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ uint8_t encode_ue8m0_scale(float exponent) {
  float encoded = fminf(fmaxf(exponent + 127.0f, 0.0f), 255.0f);
  return static_cast<uint8_t>(encoded);
}

template <typename scalar_t>
__global__ void fused_qnorm_rope_kv_insert_kernel(
    scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    uint8_t* __restrict__ k_cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    float rms_norm_eps,
    int num_tokens_full,
    int num_tokens_insert,
    int num_heads,
    int cache_block_size,
    int64_t cache_block_stride) {
  const int token_idx = blockIdx.x;
  const int task_idx = blockIdx.y;
  const int tid = threadIdx.x;

  if (token_idx >= num_tokens_full) {
    return;
  }

  __shared__ float values[kHeadDim];
  __shared__ float reduction[kThreads];
  __shared__ float scales[kNumQuantBlocks];

  if (task_idx < num_heads) {
    const int64_t q_offset =
        (static_cast<int64_t>(token_idx) * num_heads + task_idx) * kHeadDim;
    float local_sum = 0.0f;

    for (int dim = tid; dim < kHeadDim; dim += blockDim.x) {
      const float value = scalar_to_float(q[q_offset + dim]);
      values[dim] = value;
      local_sum += value * value;
    }
    reduction[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    const float rms_scale =
        rsqrtf(reduction[0] / static_cast<float>(kHeadDim) + rms_norm_eps);
    const int64_t position = positions[token_idx];
    const float* cos_ptr = cos_sin_cache + position * kRopeDim;
    const float* sin_ptr = cos_ptr + kHalfRopeDim;

    for (int dim = tid; dim < kNopeDim; dim += blockDim.x) {
      q[q_offset + dim] = float_to_scalar<scalar_t>(values[dim] * rms_scale);
    }

    for (int pair = tid; pair < kHalfRopeDim; pair += blockDim.x) {
      const int dim = kNopeDim + pair * 2;
      const float x_even = values[dim] * rms_scale;
      const float x_odd = values[dim + 1] * rms_scale;
      const float cos_v = cos_ptr[pair];
      const float sin_v = sin_ptr[pair];
      q[q_offset + dim] =
          float_to_scalar<scalar_t>(x_even * cos_v - x_odd * sin_v);
      q[q_offset + dim + 1] =
          float_to_scalar<scalar_t>(x_even * sin_v + x_odd * cos_v);
    }
    return;
  }

  if (task_idx != num_heads || token_idx >= num_tokens_insert) {
    return;
  }

  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }

  for (int dim = tid; dim < kHeadDim; dim += blockDim.x) {
    values[dim] =
        scalar_to_float(kv[static_cast<int64_t>(token_idx) * kHeadDim + dim]);
  }
  __syncthreads();

  const int64_t position = positions[token_idx];
  const float* cos_ptr = cos_sin_cache + position * kRopeDim;
  const float* sin_ptr = cos_ptr + kHalfRopeDim;
  for (int pair = tid; pair < kHalfRopeDim; pair += blockDim.x) {
    const int dim = kNopeDim + pair * 2;
    const float x_even = values[dim];
    const float x_odd = values[dim + 1];
    const float cos_v = cos_ptr[pair];
    const float sin_v = sin_ptr[pair];
    values[dim] = x_even * cos_v - x_odd * sin_v;
    values[dim + 1] = x_even * sin_v + x_odd * cos_v;
  }
  __syncthreads();

  // Match the reference cache writer by materializing K at activation dtype
  // before the UE8M0 absmax and final cache write.
  for (int dim = tid; dim < kHeadDim; dim += blockDim.x) {
    values[dim] = scalar_to_float(float_to_scalar<scalar_t>(values[dim]));
  }
  __syncthreads();

  const int64_t block_idx = slot / cache_block_size;
  const int64_t pos_in_block = slot % cache_block_size;
  uint8_t* block_base = k_cache + block_idx * cache_block_stride;
  uint8_t* token_data = block_base + pos_in_block * kTokenDataBytes;
  uint8_t* token_scales =
      block_base + static_cast<int64_t>(cache_block_size) * kTokenDataBytes +
      pos_in_block * kScaleBytesPerToken;

  if (tid < kNumQuantBlocks) {
    float absmax = 0.0f;
    const int start = tid * kQuantBlock;
    for (int i = 0; i < kQuantBlock; ++i) {
      absmax = fmaxf(absmax, fabsf(values[start + i]));
    }
    absmax = fmaxf(absmax, 1.0e-4f);
    const float exponent = ceilf(log2f(absmax / kFp8Max));
    scales[tid] = exponent;
    token_scales[tid] = encode_ue8m0_scale(exponent);
  }
  if (tid == 0) {
    token_scales[kNumQuantBlocks] = 0;
  }
  __syncthreads();

  for (int dim = tid; dim < kNopeDim; dim += blockDim.x) {
    const float inv_scale = exp2f(-scales[dim / kQuantBlock]);
    float scaled = values[dim] * inv_scale;
    scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
    const __nv_fp8_storage_t storage =
        __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
    token_data[dim] = static_cast<uint8_t>(storage);
  }

  scalar_t* rope_tail = reinterpret_cast<scalar_t*>(token_data + kNopeDim);
  for (int dim = tid; dim < kRopeDim; dim += blockDim.x) {
    rope_tail[dim] = float_to_scalar<scalar_t>(values[kNopeDim + dim]);
  }
}

template <typename scalar_t>
void launch_fused_qnorm_rope_kv_insert(
    scalar_t* q,
    const scalar_t* kv,
    uint8_t* k_cache,
    const int64_t* slot_mapping,
    const int64_t* positions,
    const float* cos_sin_cache,
    float rms_norm_eps,
    int num_tokens_full,
    int num_tokens_insert,
    int num_heads,
    int cache_block_size,
    int64_t cache_block_stride,
    cudaStream_t stream) {
  const dim3 grid(num_tokens_full, num_heads + 1);
  fused_qnorm_rope_kv_insert_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
      q, kv, k_cache, slot_mapping, positions, cos_sin_cache, rms_norm_eps,
      num_tokens_full, num_tokens_insert, num_heads, cache_block_size,
      cache_block_stride);
}

}  // namespace

void deepseek_v4_gather_paged_indexer_mxfp4_cache(TensorView kv_cache,
                                                  TensorView values_out,
                                                  TensorView scales_out,
                                                  TensorView block_table,
                                                  TensorView cu_seq_lens,
                                                  int64_t cache_block_size) {
  CHECK_CUDA(kv_cache);
  CHECK_CUDA(values_out);
  CHECK_CUDA(scales_out);
  CHECK_CUDA(block_table);
  CHECK_CUDA(cu_seq_lens);
  CHECK_DIM(2, kv_cache);
  CHECK_DIM(2, values_out);
  CHECK_DIM(2, scales_out);
  CHECK_DIM(2, block_table);
  CHECK_DIM(1, cu_seq_lens);

  TVM_FFI_ICHECK(kv_cache.dtype() == dl_uint8) << "kv_cache must be uint8";
  TVM_FFI_ICHECK(values_out.dtype() == dl_uint8) << "values_out must be uint8";
  TVM_FFI_ICHECK(scales_out.dtype() == dl_uint8) << "scales_out must be uint8";
  TVM_FFI_ICHECK(block_table.dtype() == dl_int32)
      << "block_table must be int32";
  TVM_FFI_ICHECK(cu_seq_lens.dtype() == dl_int32)
      << "cu_seq_lens must be int32";
  TVM_FFI_ICHECK(kv_cache.stride(1) == 1) << "kv_cache last dim must be contiguous";
  TVM_FFI_ICHECK(values_out.stride(1) == 1)
      << "values_out last dim must be contiguous";
  TVM_FFI_ICHECK(scales_out.stride(1) == 1)
      << "scales_out last dim must be contiguous";
  TVM_FFI_ICHECK(cache_block_size > 0) << "cache_block_size must be positive";
  TVM_FFI_ICHECK(cu_seq_lens.size(0) == block_table.size(0) + 1)
      << "cu_seq_lens must have batch_size + 1 entries";

  const int batch_size = static_cast<int>(block_table.size(0));
  const int num_tokens = static_cast<int>(values_out.size(0));
  TVM_FFI_ICHECK(scales_out.size(0) >= num_tokens)
      << "scales_out must cover values_out rows";
  // Output rows may be an exact length or a conservative upper bound, so do
  // not read cu_seq_lens[-1] on host here. The kernel only writes rows covered
  // by device-side cu_seq_lens.
  if (batch_size == 0 || num_tokens == 0) {
    return;
  }
  const int value_bytes = static_cast<int>(values_out.size(1));
  const int scale_bytes = static_cast<int>(scales_out.size(1));
  TVM_FFI_ICHECK(value_bytes > 0 && value_bytes % static_cast<int>(sizeof(uint4)) == 0)
      << "values_out width must be a positive multiple of 16 bytes";
  TVM_FFI_ICHECK(scale_bytes > 0) << "scales_out width must be positive";
  TVM_FFI_ICHECK(scale_bytes == static_cast<int>(sizeof(uint32_t)))
      << "paged indexer MXFP4 gather expects 4 scale bytes per row";
  TVM_FFI_ICHECK(kv_cache.size(1) >= cache_block_size * (value_bytes + scale_bytes))
      << "kv_cache block stride is too small for indexer MXFP4 rows";

  cudaSetDevice(kv_cache.device().device_id);
  const cudaStream_t stream = get_stream(kv_cache.device());
  constexpr int kBlockX = 8;
  constexpr int kVecBytes = sizeof(uint4);
  const int grid_y = (value_bytes + kBlockX * kVecBytes - 1) / (kBlockX * kVecBytes);

#define LAUNCH_PAGED_GATHER(BLOCK_Y)                                            \
  do {                                                                          \
    const dim3 grid((num_tokens + (BLOCK_Y)-1) / (BLOCK_Y), grid_y);            \
    const dim3 block(kBlockX, (BLOCK_Y));                                       \
    gather_paged_indexer_mxfp4_cache_kernel<(BLOCK_Y)>                          \
        <<<grid, block, 0, stream>>>(                                           \
            static_cast<const uint8_t*>(kv_cache.data_ptr()),                   \
            static_cast<uint8_t*>(values_out.data_ptr()),                       \
            static_cast<uint8_t*>(scales_out.data_ptr()),                       \
            static_cast<const int32_t*>(block_table.data_ptr()),                \
            static_cast<const int32_t*>(cu_seq_lens.data_ptr()), batch_size,    \
            num_tokens, value_bytes, scale_bytes,                               \
            static_cast<int>(cache_block_size), kv_cache.stride(0),             \
            values_out.stride(0), scales_out.stride(0), block_table.stride(0)); \
  } while (0)

  if (num_tokens < 32) {
    LAUNCH_PAGED_GATHER(1);
  } else if (num_tokens < 64) {
    LAUNCH_PAGED_GATHER(2);
  } else if (num_tokens < 128) {
    LAUNCH_PAGED_GATHER(4);
  } else if (num_tokens < 256) {
    LAUNCH_PAGED_GATHER(8);
  } else if (num_tokens < 512) {
    LAUNCH_PAGED_GATHER(16);
  } else {
    LAUNCH_PAGED_GATHER(32);
  }
#undef LAUNCH_PAGED_GATHER

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "deepseek_v4_gather_paged_indexer_mxfp4_cache failed: "
      << cudaGetErrorString(status);
}

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    TensorView q,
    TensorView kv,
    TensorView k_cache,
    TensorView slot_mapping,
    TensorView positions,
    TensorView cos_sin_cache,
    double rms_norm_eps,
    int64_t cache_block_size) {
  CHECK_CUDA(q);
  CHECK_CUDA(kv);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(slot_mapping);
  CHECK_CUDA(positions);
  CHECK_CUDA(cos_sin_cache);
  CHECK_DIM(3, q);
  CHECK_DIM(2, kv);
  CHECK_DIM(2, k_cache);
  CHECK_DIM(1, slot_mapping);
  CHECK_DIM(1, positions);
  CHECK_DIM(2, cos_sin_cache);

  TVM_FFI_ICHECK(q.IsContiguous()) << "q must be contiguous";
  TVM_FFI_ICHECK(kv.IsContiguous()) << "kv must be contiguous";
  TVM_FFI_ICHECK(k_cache.stride(1) == 1) << "k_cache last dim must be contiguous";
  TVM_FFI_ICHECK(slot_mapping.IsContiguous()) << "slot_mapping must be contiguous";
  TVM_FFI_ICHECK(positions.IsContiguous()) << "positions must be contiguous";
  TVM_FFI_ICHECK(cos_sin_cache.IsContiguous()) << "cos_sin_cache must be contiguous";
  TVM_FFI_ICHECK(q.dtype() == kv.dtype()) << "q and kv dtype must match";
  TVM_FFI_ICHECK(k_cache.dtype() == dl_uint8) << "k_cache must be uint8";
  TVM_FFI_ICHECK(slot_mapping.dtype() == dl_int64) << "slot_mapping must be int64";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(cos_sin_cache.dtype() == dl_float32)
      << "cos_sin_cache must be float32";
  TVM_FFI_ICHECK(q.size(2) == kHeadDim) << "q must have head_dim=512";
  TVM_FFI_ICHECK(kv.size(1) == kHeadDim) << "kv must have dim=512";
  TVM_FFI_ICHECK(kv.size(0) == q.size(0)) << "q and kv token counts must match";
  TVM_FFI_ICHECK(positions.size(0) == q.size(0))
      << "positions must cover all q rows";
  TVM_FFI_ICHECK(slot_mapping.size(0) <= q.size(0))
      << "slot_mapping cannot be longer than q";
  TVM_FFI_ICHECK(cos_sin_cache.size(1) == kRopeDim)
      << "cos_sin_cache must have width 64";
  TVM_FFI_ICHECK(cache_block_size > 0) << "cache_block_size must be positive";
  TVM_FFI_ICHECK(k_cache.size(1) >= cache_block_size * (kTokenDataBytes + kScaleBytesPerToken))
      << "k_cache block stride is too small for DeepSeek V4 SWA rows";

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  const int num_tokens_full = static_cast<int>(q.size(0));
  const int num_tokens_insert = static_cast<int>(slot_mapping.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int64_t cache_block_stride = k_cache.stride(0);

  if (q.dtype() == dl_float16) {
    launch_fused_qnorm_rope_kv_insert<half>(
        static_cast<half*>(q.data_ptr()), static_cast<const half*>(kv.data_ptr()),
        static_cast<uint8_t*>(k_cache.data_ptr()),
        static_cast<const int64_t*>(slot_mapping.data_ptr()),
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<const float*>(cos_sin_cache.data_ptr()),
        static_cast<float>(rms_norm_eps), num_tokens_full, num_tokens_insert,
        num_heads, static_cast<int>(cache_block_size), cache_block_stride, stream);
  } else if (q.dtype() == dl_bfloat16) {
    launch_fused_qnorm_rope_kv_insert<nv_bfloat16>(
        static_cast<nv_bfloat16*>(q.data_ptr()),
        static_cast<const nv_bfloat16*>(kv.data_ptr()),
        static_cast<uint8_t*>(k_cache.data_ptr()),
        static_cast<const int64_t*>(slot_mapping.data_ptr()),
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<const float*>(cos_sin_cache.data_ptr()),
        static_cast<float>(rms_norm_eps), num_tokens_full, num_tokens_insert,
        num_heads, static_cast<int>(cache_block_size), cache_block_stride, stream);
  } else {
    TVM_FFI_ICHECK(false) << "q/kv dtype must be float16 or bfloat16";
  }

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert failed: "
      << cudaGetErrorString(status);
}
