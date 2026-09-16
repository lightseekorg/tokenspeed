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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace tokenspeed {

template <int NumSplits, int BlockSize>
__launch_bounds__(BlockSize) __global__ void MhcBigFuseKernel(
    const float* __restrict__ projection, const float* __restrict__ square_sum,
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base, float* __restrict__ post_mix,
    float* __restrict__ comb_mix, __nv_bfloat16* __restrict__ layer_input,
    int num_tokens, int hidden_size, float rms_eps, float hc_eps,
    int sinkhorn_iters, const __nv_bfloat16* __restrict__ norm_weight,
    float norm_eps, bool fuse_norm) {
  constexpr int kHc = 4;
  constexpr int kProjection = 24;
  constexpr int kWarp = 32;
  constexpr int kVec = 8;
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;");
#endif

  __shared__ float projection_reduced[kProjection];
  __shared__ float rstd_shared;
  __shared__ float pre_mix[kHc];
  __shared__ float norm_warp_sums[16];
  __shared__ float norm_rstd;
  float comb[kHc];
  float norm_square_acc = 0.0f;
  if (warp == 0) {
    if (lane < kProjection) {
      float projection_sum = 0.0f;
#pragma unroll
      for (int split = 0; split < NumSplits; ++split) {
        const float* row =
            projection + (split * num_tokens + token) * kProjection;
        projection_sum += row[lane];
      }
      projection_reduced[lane] = projection_sum;
    }
    if (lane == 0) {
      float square_sum_reduced = 0.0f;
#pragma unroll
      for (int split = 0; split < NumSplits; ++split) {
        square_sum_reduced += square_sum[split * num_tokens + token];
      }
      rstd_shared = rsqrtf(
          square_sum_reduced / static_cast<float>(kHc * hidden_size) + rms_eps);
    }
    __syncwarp();
    if (lane < kHc) {
      const float rstd = rstd_shared;
      float value =
          projection_reduced[lane] * rstd * hc_scale[0] + hc_base[lane];
      pre_mix[lane] = 1.0f / (1.0f + expf(-value)) + hc_eps;
      value = projection_reduced[kHc + lane] * rstd * hc_scale[1] +
              hc_base[kHc + lane];
      post_mix[token * kHc + lane] = 2.0f / (1.0f + expf(-value));
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        const int index = 2 * kHc + lane * kHc + column;
        comb[column] =
            projection_reduced[index] * rstd * hc_scale[2] + hc_base[index];
      }
    }
  }
  __syncthreads();

  if (warp == 0 && lane < kHc) {
    constexpr unsigned kLaneMask = 0x0f;
    const float row_max = fmaxf(fmaxf(comb[0], comb[1]), fmaxf(comb[2], comb[3]));
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      comb[column] = expf(comb[column] - row_max);
    }
    float inverse_row_sum = 1.0f / (comb[0] + comb[1] + comb[2] + comb[3]);
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      comb[column] = comb[column] * inverse_row_sum + hc_eps;
      float column_sum = comb[column];
      column_sum += __shfl_xor_sync(kLaneMask, column_sum, 1);
      column_sum += __shfl_xor_sync(kLaneMask, column_sum, 2);
      comb[column] *= 1.0f / (column_sum + hc_eps);
    }
    for (int iteration = 1; iteration < sinkhorn_iters; ++iteration) {
      inverse_row_sum = 1.0f / (comb[0] + comb[1] + comb[2] + comb[3] + hc_eps);
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        comb[column] *= inverse_row_sum;
      }
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        float column_sum = comb[column];
        column_sum += __shfl_xor_sync(kLaneMask, column_sum, 1);
        column_sum += __shfl_xor_sync(kLaneMask, column_sum, 2);
        comb[column] *= 1.0f / (column_sum + hc_eps);
      }
    }
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      comb_mix[token * kHc * kHc + lane * kHc + column] = comb[column];
    }
  }

  if (warp > 0) {
    float mix[kHc];
#pragma unroll
    for (int index = 0; index < kHc; ++index) {
      mix[index] = pre_mix[index];
    }
    const __nv_bfloat16* residual_row = residual + token * kHc * hidden_size;
    __nv_bfloat16* output_row = layer_input + token * hidden_size;
    const int apply_tid = tid - kWarp;
    constexpr int kApplyThreads = BlockSize - kWarp;
    for (int hidden = apply_tid * kVec; hidden < hidden_size;
         hidden += kApplyThreads * kVec) {
      float accumulator[kVec] = {};
#pragma unroll
      for (int stream = 0; stream < kHc; ++stream) {
        const uint4 raw = *reinterpret_cast<const uint4*>(
            residual_row + stream * hidden_size + hidden);
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int pair = 0; pair < kVec / 2; ++pair) {
          const float2 value = __bfloat1622float2(pairs[pair]);
          accumulator[pair * 2] += mix[stream] * value.x;
          accumulator[pair * 2 + 1] += mix[stream] * value.y;
        }
      }
      uint4 raw;
      __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
      for (int pair = 0; pair < kVec / 2; ++pair) {
        pairs[pair] = __float22bfloat162_rn(
            make_float2(accumulator[pair * 2], accumulator[pair * 2 + 1]));
        if (fuse_norm) {
          const float2 rounded = __bfloat1622float2(pairs[pair]);
          norm_square_acc += rounded.x * rounded.x + rounded.y * rounded.y;
        }
      }
      *reinterpret_cast<uint4*>(output_row + hidden) = raw;
    }
    if (fuse_norm) {
#pragma unroll
      for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        norm_square_acc +=
            __shfl_down_sync(0xffffffff, norm_square_acc, offset);
      }
      if (lane == 0) norm_warp_sums[warp] = norm_square_acc;
    }
  }
  if (fuse_norm) {
    __syncthreads();
    const __nv_bfloat16* output_row = layer_input + token * hidden_size;
    if (warp == 0) {
      constexpr int kApplyWarps = BlockSize / kWarp - 1;
      float block_sum =
          lane < kApplyWarps ? norm_warp_sums[lane + 1] : 0.0f;
#pragma unroll
      for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
      }
      if (lane == 0) {
        norm_rstd = rsqrtf(block_sum / static_cast<float>(hidden_size) + norm_eps);
      }
    }
    __syncthreads();
    for (int hidden = tid * kVec; hidden < hidden_size;
         hidden += BlockSize * kVec) {
      const uint4 output_raw =
          *reinterpret_cast<const uint4*>(output_row + hidden);
      const uint4 weight_raw =
          *reinterpret_cast<const uint4*>(norm_weight + hidden);
      const __nv_bfloat162* output_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&output_raw);
      const __nv_bfloat162* weight_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&weight_raw);
      uint4 normalized_raw;
      __nv_bfloat162* normalized_pairs =
          reinterpret_cast<__nv_bfloat162*>(&normalized_raw);
#pragma unroll
      for (int pair = 0; pair < kVec / 2; ++pair) {
        const float2 output_value = __bfloat1622float2(output_pairs[pair]);
        const float2 weight_value = __bfloat1622float2(weight_pairs[pair]);
        normalized_pairs[pair] = __float22bfloat162_rn(make_float2(
            output_value.x * norm_rstd * weight_value.x,
            output_value.y * norm_rstd * weight_value.y));
      }
      *reinterpret_cast<uint4*>(layer_input + token * hidden_size + hidden) =
          normalized_raw;
    }
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <int NumSplits>
cudaError_t LaunchMhcBigFuse(
    const float* projection, const float* square_sum, const __nv_bfloat16* residual,
    const float* hc_scale, const float* hc_base, float* post_mix, float* comb_mix,
    __nv_bfloat16* layer_input, int num_tokens, int hidden_size, float rms_eps,
    float hc_eps, int sinkhorn_iters, const __nv_bfloat16* norm_weight,
    float norm_eps, bool fuse_norm, int block_size, bool enable_pdl,
    cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(num_tokens);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = enable_pdl;
  config.attrs = &attribute;
  config.numAttrs = 1;
#define LAUNCH(BlockSize)                                                        \
  do {                                                                           \
    config.blockDim = dim3(BlockSize);                                            \
    auto kernel = MhcBigFuseKernel<NumSplits, BlockSize>;                         \
    return cudaLaunchKernelEx(&config, kernel, projection, square_sum, residual,  \
                              hc_scale, hc_base, post_mix, comb_mix, layer_input, \
                              num_tokens, hidden_size, rms_eps, hc_eps,            \
                              sinkhorn_iters, norm_weight, norm_eps, fuse_norm);   \
  } while (false)
  if (block_size == 128) LAUNCH(128);
  if (block_size == 256) LAUNCH(256);
  LAUNCH(512);
#undef LAUNCH
}

}  // namespace tokenspeed

void mhc_big_fuse(
    TensorView projection, TensorView square_sum, TensorView hc_scale,
    TensorView hc_base, TensorView residual, TensorView layer_input,
    TensorView post_mix, TensorView comb_mix, int64_t hidden_size, double rms_eps,
    double hc_eps, int64_t sinkhorn_iters, int64_t n_splits, int64_t num_tokens,
    TensorView norm_weight, double norm_eps, bool fuse_norm, int64_t block_size,
    bool enable_pdl) {
  CHECK_INPUT(projection);
  CHECK_INPUT(square_sum);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);
  CHECK_INPUT(residual);
  CHECK_INPUT(layer_input);
  CHECK_INPUT(post_mix);
  CHECK_INPUT(comb_mix);
  CHECK_INPUT(norm_weight);
  CHECK_DEVICE(square_sum, projection);
  CHECK_DEVICE(hc_scale, projection);
  CHECK_DEVICE(hc_base, projection);
  CHECK_DEVICE(residual, projection);
  CHECK_DEVICE(layer_input, projection);
  CHECK_DEVICE(post_mix, projection);
  CHECK_DEVICE(comb_mix, projection);
  CHECK_DEVICE(norm_weight, projection);
  CHECK_DIM(3, projection);
  CHECK_DIM(2, square_sum);
  CHECK_DIM(1, hc_scale);
  CHECK_DIM(1, hc_base);
  CHECK_DIM(3, residual);
  CHECK_DIM(2, layer_input);
  CHECK_DIM(2, post_mix);
  CHECK_DIM(2, comb_mix);
  CHECK_DIM(1, norm_weight);
  TVM_FFI_ICHECK_EQ(projection.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(square_sum.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(hc_scale.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(hc_base.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(residual.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(layer_input.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(post_mix.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(comb_mix.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(norm_weight.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(projection.size(0), n_splits);
  TVM_FFI_ICHECK_EQ(projection.size(1), num_tokens);
  TVM_FFI_ICHECK_EQ(projection.size(2), 24);
  TVM_FFI_ICHECK_EQ(square_sum.size(0), n_splits);
  TVM_FFI_ICHECK_EQ(square_sum.size(1), num_tokens);
  TVM_FFI_ICHECK_EQ(residual.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(residual.size(1), 4);
  TVM_FFI_ICHECK_EQ(residual.size(2), hidden_size);
  TVM_FFI_ICHECK_EQ(layer_input.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(layer_input.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(post_mix.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(post_mix.size(1), 4);
  TVM_FFI_ICHECK_EQ(comb_mix.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(comb_mix.size(1), 16);
  TVM_FFI_ICHECK_EQ(hc_scale.size(0), 3);
  TVM_FFI_ICHECK_EQ(hc_base.size(0), 24);
  TVM_FFI_ICHECK_EQ(norm_weight.size(0), hidden_size);
  TVM_FFI_ICHECK_GT(num_tokens, 0);
  TVM_FFI_ICHECK_GT(sinkhorn_iters, 0);
  TVM_FFI_ICHECK_EQ(hidden_size % 8, 0);
  TVM_FFI_ICHECK(block_size == 128 || block_size == 256 || block_size == 512);
  TVM_FFI_ICHECK(n_splits == 1 || n_splits == 2 || n_splits == 4 ||
                 n_splits == 8 || n_splits == 16 || n_splits == 32 ||
                 n_splits == 64)
      << "unsupported n_splits=" << n_splits;

  cudaSetDevice(projection.device().device_id);
  const cudaStream_t stream = get_stream(projection.device());
  cudaError_t status = cudaErrorInvalidValue;
#define DISPATCH(Splits)                                                          \
  status = tokenspeed::LaunchMhcBigFuse<Splits>(                                  \
      static_cast<const float*>(projection.data_ptr()),                           \
      static_cast<const float*>(square_sum.data_ptr()),                           \
      static_cast<const __nv_bfloat16*>(residual.data_ptr()),                     \
      static_cast<const float*>(hc_scale.data_ptr()),                             \
      static_cast<const float*>(hc_base.data_ptr()),                              \
      static_cast<float*>(post_mix.data_ptr()),                                   \
      static_cast<float*>(comb_mix.data_ptr()),                                   \
      static_cast<__nv_bfloat16*>(layer_input.data_ptr()),                        \
      static_cast<int>(num_tokens), static_cast<int>(hidden_size),                \
      static_cast<float>(rms_eps), static_cast<float>(hc_eps),                    \
      static_cast<int>(sinkhorn_iters),                                           \
      static_cast<const __nv_bfloat16*>(norm_weight.data_ptr()),                  \
      static_cast<float>(norm_eps), fuse_norm, static_cast<int>(block_size),       \
      enable_pdl, stream)
  switch (n_splits) {
    case 1: DISPATCH(1); break;
    case 2: DISPATCH(2); break;
    case 4: DISPATCH(4); break;
    case 8: DISPATCH(8); break;
    case 16: DISPATCH(16); break;
    case 32: DISPATCH(32); break;
    case 64: DISPATCH(64); break;
  }
#undef DISPATCH
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "mhc_big_fuse launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_big_fuse, mhc_big_fuse);
