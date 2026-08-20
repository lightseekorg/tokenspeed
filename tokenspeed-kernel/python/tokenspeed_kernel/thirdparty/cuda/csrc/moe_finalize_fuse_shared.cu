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

/*
 * Fused MoE finalize + shared-output add (bf16 output, SM>=90 for PDL).
 *
 * Forked from flashinfer's ``finalizeKernel`` and ``finalizeKernelVecLoad``
 * (trtllm_fused_moe_dev_kernel.cu:639 and :803), stripped of the MoE
 * backend's KernelParams / UsePdl templating, and extended with an
 * optional shared_output residual add on the epilogue side.
 *
 * For each token t, computes:
 *     out[t] = Σ_k expert_weights[t, k] * gemm2_out[permuted_idx(t, k)]
 *            + shared_output[t]                      // if non-null
 *
 * Shared-expert-sink extension (Inkling): when ``expert_weights`` is
 * ``[numTokens, topK + numShared]`` (numShared > 0), ``shared_output`` is
 * the un-weighted per-shared-expert output ``[numShared, numTokens,
 * hiddenDim]`` and the tail weight columns are applied here:
 *     out[t] = Σ_k w[t, k] * gemm2_out[permuted_idx(t, k)]
 *            + Σ_s w[t, topK + s] * shared_output[s, t]
 * The routed and shared weights come from one joint normalization, so
 * fusing both applications keeps the whole combine in a single epilogue.
 *
 * Eliminates the native PyTorch ``routed + shared_output`` add (and the
 * separate ``*= routed_scaling_factor`` kernel when applicable) from
 * ``DeepseekV3MoE.forward``, and gives the downstream allreduce+rmsnorm
 * a clean PDL handoff.
 *
 * Expert-weight dtype is templated on ``TypeExpW`` so we support both the
 * bf16 and fp32 topk-weight paths (DSv3/K2.5 trtllm backends use fp32
 * because their ``_routing_logits_dtype = torch.float32``; other backends
 * use bf16).
 *
 * Expert-weight scale convention: in our target backends
 * (flashinfer trtllm nvfp4 + unquantized), ``apply_routed_scaling_factor_on_output``
 * is True, so the routed scaling factor is already folded into
 * ``expert_weights`` at topk time. This kernel does not apply any
 * additional scale.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <type_traits>

#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "tvm_ffi_utils.h"

namespace tokenspeed {

using BF16 = cutlass::bfloat16_t;

constexpr int FINALIZE_THREADS_PER_BLOCK = 256;
constexpr int MAX_TOPK = 64;
constexpr int MAX_SHARED = 8;

// ---------------------------------------------------------------------------
// General kernel — one CTA per (hidden_chunk, token). Picks up small-to-mid
// workloads where the block count fits in a few waves.
// ---------------------------------------------------------------------------
template <typename TypeExpW>
__global__ void moeFinalizeKernel(int numTokens, int hiddenDim, int hiddenDimPadded, int topK,
                                  int numShared, BF16 const* __restrict__ inPtr,
                                  int const* __restrict__ expandedIdxToPermutedIdx,
                                  TypeExpW const* __restrict__ expertWeightsPtr,
                                  BF16 const* __restrict__ sharedBiasPtr,
                                  BF16* __restrict__ outPtr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  // Row stride: [topK] weights, or [topK | shared] from a joint normalization when numShared > 0.
  int const weightStride = topK + numShared;

  for (int64_t tokenIdx = blockIdx.y; tokenIdx < numTokens; tokenIdx += gridDim.y) {
    for (int64_t hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
         hiddenIdx < hiddenDim; hiddenIdx += blockDim.x * gridDim.x) {
      float acc = 0.0f;
      for (int k = 0; k < topK; k++) {
        int64_t const permutedIdx = expandedIdxToPermutedIdx[tokenIdx * topK + k];
        if (permutedIdx == -1) {
          continue;
        }
        float const scale =
            static_cast<float>(expertWeightsPtr[tokenIdx * weightStride + k]);
        float const val =
            static_cast<float>(inPtr[permutedIdx * hiddenDimPadded + hiddenIdx]);
        acc += scale * val;
      }
      if (sharedBiasPtr != nullptr) {
        if (numShared == 0) {
          // Pre-combined [numTokens, hiddenDim] residual, added verbatim.
          acc += static_cast<float>(sharedBiasPtr[tokenIdx * hiddenDim + hiddenIdx]);
        } else {
          // Un-weighted [numShared, numTokens, hiddenDim] shared outputs; apply tail weight columns here.
          for (int s = 0; s < numShared; s++) {
            float const scale = static_cast<float>(
                expertWeightsPtr[tokenIdx * weightStride + topK + s]);
            float const val = static_cast<float>(
                sharedBiasPtr[(s * int64_t(numTokens) + tokenIdx) * hiddenDim + hiddenIdx]);
            acc += scale * val;
          }
        }
      }
      outPtr[tokenIdx * hiddenDim + hiddenIdx] = static_cast<BF16>(acc);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------
// Vectorized-load kernel — one CTA per token, 128-bit loads, topK unrolled.
// Better at prefill shapes where the general kernel's block count saturates
// many waves and the indirect gather from gemm2_out dominates.
// ---------------------------------------------------------------------------

__device__ inline float4 vectorizedLoadPtx(float4 const* ptr) {
  float4 ret;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "l"(ptr));
  return ret;
}

template <int TopKUnrollFactor>
struct IdxPackedTraits;
template <>
struct IdxPackedTraits<1> {
  using Packed = int;
};
template <>
struct IdxPackedTraits<2> {
  using Packed = int2;
};
template <>
struct IdxPackedTraits<4> {
  using Packed = int4;
};

template <typename TypeExpW, int TopKUnrollFactor>
__global__ void moeFinalizeKernelVecLoad(int numTokens, int hiddenDim, int hiddenDimPadded,
                                         int topK, int numShared, BF16 const* __restrict__ inPtr,
                                         int const* __restrict__ expandedIdxToPermutedIdx,
                                         TypeExpW const* __restrict__ expertWeightsPtr,
                                         BF16 const* __restrict__ sharedBiasPtr,
                                         BF16* __restrict__ outPtr) {
  static_assert(TopKUnrollFactor == 1 || TopKUnrollFactor == 2 || TopKUnrollFactor == 4,
                "TopKUnrollFactor must be 1, 2, or 4");
  using IdxPackedType = typename IdxPackedTraits<TopKUnrollFactor>::Packed;
  using IdxArrayType = cutlass::Array<int, TopKUnrollFactor>;
  using ScaleArrayType = cutlass::Array<TypeExpW, TopKUnrollFactor>;

  // 128 bits per thread → 8 bf16 elements.
  constexpr int FINALIZE_ELEM_PER_THREAD = 8;
  using InputElem = cutlass::Array<BF16, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<BF16, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  int64_t const tokenIdx = blockIdx.x;
  int64_t const startOffset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const numElemsInPaddedCol = hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
  int64_t const numElemsInCol = hiddenDim / FINALIZE_ELEM_PER_THREAD;

  int const weightStride = topK + numShared;

  // Stage the per-token (topK/unroll) indices + scales into smem.
  __shared__ ScaleArrayType scaleArrSmem[MAX_TOPK / TopKUnrollFactor];
  __shared__ IdxArrayType permutedIdxArrSmem[MAX_TOPK / TopKUnrollFactor];
  __shared__ float sharedScaleSmem[MAX_SHARED];

  for (int kChunkIdx = threadIdx.x; kChunkIdx < topK / TopKUnrollFactor; kChunkIdx += blockDim.x) {
    int64_t const expandedIdx = tokenIdx * topK + kChunkIdx * TopKUnrollFactor;
    auto const permutedIdxPacked = reinterpret_cast<IdxPackedType const*>(
        expandedIdxToPermutedIdx)[expandedIdx / TopKUnrollFactor];
    permutedIdxArrSmem[kChunkIdx] =
        *reinterpret_cast<IdxArrayType const*>(&permutedIdxPacked);
#pragma unroll
    for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
      scaleArrSmem[kChunkIdx][ki] =
          expertWeightsPtr[tokenIdx * weightStride + kChunkIdx * TopKUnrollFactor + ki];
    }
  }
  for (int s = threadIdx.x; s < numShared; s += blockDim.x) {
    sharedScaleSmem[s] =
        static_cast<float>(expertWeightsPtr[tokenIdx * weightStride + topK + s]);
  }

  BF16* outputPtr = outPtr + tokenIdx * hiddenDim;
  auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
  auto const* inElemPtr = reinterpret_cast<InputElem const*>(inPtr);
  // numShared==0: pre-combined residual; else shared expert s, token t is row (s*numTokens + t).
  auto const* sharedElemPtr =
      sharedBiasPtr != nullptr
          ? reinterpret_cast<InputElem const*>(sharedBiasPtr + tokenIdx * hiddenDim)
          : nullptr;
  int64_t const sharedExpertElemStride = int64_t(numTokens) * numElemsInCol;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  __syncthreads();

  for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride) {
    ComputeElem threadOutput;
    threadOutput.fill(0.0f);

    for (int kChunkIdx = 0; kChunkIdx < topK / TopKUnrollFactor; kChunkIdx++) {
      IdxArrayType permutedIdxArr = permutedIdxArrSmem[kChunkIdx];
      InputElem inputElemArr[TopKUnrollFactor];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        int const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;
        float4 input =
            vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
        inputElemArr[ki] = *reinterpret_cast<InputElem const*>(&input);
      }
      ScaleArrayType scaleArr = scaleArrSmem[kChunkIdx];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        int const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        float const scale = static_cast<float>(scaleArr[ki]);
        cutlass::NumericArrayConverter<float, BF16, FINALIZE_ELEM_PER_THREAD> toFloat;
        ComputeElem expertResult = toFloat(inputElemArr[ki]);
#pragma unroll
        for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
          threadOutput[e] += scale * expertResult[e];
        }
      }
    }

    if (sharedElemPtr != nullptr) {
      cutlass::NumericArrayConverter<float, BF16, FINALIZE_ELEM_PER_THREAD> toFloat;
      if (numShared == 0) {
        float4 shared =
            vectorizedLoadPtx(reinterpret_cast<float4 const*>(&sharedElemPtr[elemIndex]));
        InputElem sharedElem = *reinterpret_cast<InputElem const*>(&shared);
        ComputeElem sharedFloat = toFloat(sharedElem);
#pragma unroll
        for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
          threadOutput[e] += sharedFloat[e];
        }
      } else {
        for (int s = 0; s < numShared; ++s) {
          float4 shared = vectorizedLoadPtx(reinterpret_cast<float4 const*>(
              &sharedElemPtr[s * sharedExpertElemStride + elemIndex]));
          InputElem sharedElem = *reinterpret_cast<InputElem const*>(&shared);
          ComputeElem sharedFloat = toFloat(sharedElem);
          float const scale = sharedScaleSmem[s];
#pragma unroll
          for (int e = 0; e < FINALIZE_ELEM_PER_THREAD; ++e) {
            threadOutput[e] += scale * sharedFloat[e];
          }
        }
      }
    }

    cutlass::NumericArrayConverter<BF16, float, FINALIZE_ELEM_PER_THREAD> toBF16;
    outElemPtr[elemIndex] = toBF16(threadOutput);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------
// Typed dispatch
// ---------------------------------------------------------------------------
template <typename TypeExpW>
void dispatchFinalize(int numTokens, int hiddenDim, int hiddenDimPadded, int topK, int numShared,
                      BF16 const* inPtr, int const* expandedIdxPtr, void const* weightsPtrVoid,
                      BF16 const* sharedPtr, BF16* outPtr, bool useVecLoad, cudaStream_t stream,
                      cudaLaunchAttribute const* attrs, int numAttrs) {
  auto const* weightsPtr = static_cast<TypeExpW const*>(weightsPtrVoid);
  constexpr int kNumThreads = 256;

  if (!useVecLoad) {
    int const numBlocksX = (hiddenDim + kNumThreads - 1) / kNumThreads;
    int const numBlocksY = std::min(8192, numTokens);
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numBlocksX, numBlocksY);
    config.blockDim = dim3(kNumThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.numAttrs = numAttrs;
    config.attrs = const_cast<cudaLaunchAttribute*>(attrs);

    cudaLaunchKernelEx(&config, moeFinalizeKernel<TypeExpW>, numTokens, hiddenDim, hiddenDimPadded,
                       topK, numShared, inPtr, expandedIdxPtr, weightsPtr, sharedPtr, outPtr);
    return;
  }

  auto launch = [&](auto unroll_tag) {
    constexpr int UNROLL = decltype(unroll_tag)::value;
    cudaLaunchConfig_t config;
    config.gridDim = dim3(numTokens);
    config.blockDim = dim3(FINALIZE_THREADS_PER_BLOCK);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.numAttrs = numAttrs;
    config.attrs = const_cast<cudaLaunchAttribute*>(attrs);
    cudaLaunchKernelEx(&config, moeFinalizeKernelVecLoad<TypeExpW, UNROLL>, numTokens, hiddenDim,
                       hiddenDimPadded, topK, numShared, inPtr, expandedIdxPtr, weightsPtr,
                       sharedPtr, outPtr);
  };
  // Match flashinfer's LAUNCH_TOPK_EXPW dispatch order.
  if (topK % 4 == 0) {
    launch(std::integral_constant<int, 4>{});
  } else if (topK % 2 == 0) {
    launch(std::integral_constant<int, 2>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}

// ---------------------------------------------------------------------------
// Kimi-K3 prepared MoE front: pack precomputed top-k + MXFP8 quant
// ---------------------------------------------------------------------------
//
// FlashInfer needs MXFP8 quantization of routed activations and packed
// (expert id, bf16 weight) pairs. One CTA per token performs both: the first
// 16 lanes pack routes while all 224 lanes quantize one 32-value group per
// lane pair.

constexpr int K3_PREP_HIDDEN = 3584;
constexpr int K3_PREP_GROUP = 32;
constexpr int K3_PREP_GROUPS = K3_PREP_HIDDEN / K3_PREP_GROUP;
constexpr int K3_PREP_TOPK = 16;
constexpr int K3_PREP_THREADS = K3_PREP_GROUPS * 2;
constexpr float K3_FP8_MAX = 448.0f;

constexpr int K3_ROUTE_EXPERTS = 896;
constexpr int K3_ROUTE_VEC = 4;
constexpr int K3_ROUTE_THREADS = K3_ROUTE_EXPERTS / K3_ROUTE_VEC;
constexpr int K3_ROUTE_WARPS = (K3_ROUTE_THREADS + 31) / 32;
constexpr int K3_ROUTE_RADIX_BITS = 8;
constexpr int K3_ROUTE_RADIX_SIZE = 1 << K3_ROUTE_RADIX_BITS;
constexpr int K3_ROUTE_RADIX_ROUNDS = 32 / K3_ROUTE_RADIX_BITS;

struct alignas(16) K3RouteMatch {
  uint32_t bin;
  uint32_t aboveCount;
  uint32_t equalCount;
};

struct K3RouteSmem {
  uint32_t warpSum[3][K3_ROUTE_WARPS];
  K3RouteMatch match[K3_ROUTE_RADIX_ROUNDS];
  uint32_t histogram[K3_ROUTE_RADIX_SIZE];
  int32_t winnerId[K3_PREP_TOPK];
  uint32_t winnerKey[K3_PREP_TOPK];
  float winnerWeight[K3_PREP_TOPK];
};

__device__ __forceinline__ uint32_t k3WarpInclusiveSum(uint32_t value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    uint32_t const peer = __shfl_up_sync(0xffffffffu, value, offset);
    if ((threadIdx.x & 31) >= offset)
      value += peer;
  }
  return value;
}

__device__ __forceinline__ uint32_t
k3BlockExclusiveSum(uint32_t value, uint32_t *__restrict__ warpSum) {
  int const lane = int(threadIdx.x) & 31;
  int const warp = int(threadIdx.x) >> 5;
  uint32_t const inclusive = k3WarpInclusiveSum(value);
  if (lane == 31 || int(threadIdx.x) == K3_ROUTE_THREADS - 1) {
    warpSum[warp] = inclusive;
  }
  __syncthreads();
  uint32_t base = 0;
#pragma unroll
  for (int i = 0; i < K3_ROUTE_WARPS; ++i) {
    if (i < warp)
      base += warpSum[i];
  }
  return base + inclusive - value;
}

__device__ __forceinline__ float k3Sigmoid(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

__device__ __forceinline__ uint32_t k3BiasedToKey(float value) {
  // Match the Triton packed-key router: canonicalize signed zero and order
  // every finite FP32 value monotonically as an unsigned integer.
  if (value == 0.0f)
    value = 0.0f;
  uint32_t const bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ void
k3RouteRow(float const *__restrict__ logits, int64_t logitsRowStride,
           float const *__restrict__ bias,
           __nv_bfloat16 *__restrict__ topkWeights, int64_t weightsRowStride,
           int32_t *__restrict__ topkIds, int64_t idsRowStride,
           int32_t *__restrict__ packedTopk, int64_t packedRowStride,
           float routedScalingFactor, bool renormalize, K3RouteSmem &smem) {
  int const row = int(blockIdx.x);
  int const tid = int(threadIdx.x);
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const firstExpert = tid * K3_ROUTE_VEC;

  uint32_t keys[K3_ROUTE_VEC];
  float weights[K3_ROUTE_VEC];
  auto const *rowLogits = logits + int64_t(row) * logitsRowStride;
#pragma unroll
  for (int i = 0; i < K3_ROUTE_VEC; ++i) {
    int const expert = firstExpert + i;
    float const weight = k3Sigmoid(rowLogits[expert]);
    float selected = weight + bias[expert];
    if (isnan(selected))
      selected = -1.0e30f;
    weights[i] = weight;
    keys[i] = k3BiasedToKey(selected);
  }

  bool active[K3_ROUTE_VEC] = {true, true, true, true};
  uint32_t totalActive = K3_ROUTE_EXPERTS;
  uint32_t quota = K3_PREP_TOPK;
  uint32_t threshold = 0;
  uint32_t examinedMask = 0;
  bool takeAllEquals = false;

  if (tid < K3_ROUTE_RADIX_SIZE / 2) {
    smem.histogram[2 * tid] = 0;
    smem.histogram[2 * tid + 1] = 0;
  }
#pragma unroll
  for (int round = 0; round < K3_ROUTE_RADIX_ROUNDS; ++round) {
    __syncthreads();
    int const shift = 24 - round * K3_ROUTE_RADIX_BITS;
    uint32_t bins[K3_ROUTE_VEC];
#pragma unroll
    for (int i = 0; i < K3_ROUTE_VEC; ++i) {
      bins[i] = (keys[i] >> shift) & 0xffu;
      if (active[i])
        atomicAdd(&smem.histogram[bins[i]], 1u);
    }
    __syncthreads();

    uint32_t pairCount = 0;
    if (tid < K3_ROUTE_RADIX_SIZE / 2) {
      pairCount = smem.histogram[2 * tid] + smem.histogram[2 * tid + 1];
    }
    uint32_t const pairInclusive = k3WarpInclusiveSum(pairCount);
    if (lane == 31 && warp < 4)
      smem.warpSum[0][warp] = pairInclusive;
    __syncthreads();

    if (tid < K3_ROUTE_RADIX_SIZE / 2) {
      uint32_t prefix = pairInclusive;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < warp)
          prefix += smem.warpSum[0][i];
      }
      uint32_t const right = smem.histogram[2 * tid + 1];
      uint32_t const left = smem.histogram[2 * tid];
      uint32_t const aboveRight = totalActive - prefix;
      uint32_t const aboveLeft = aboveRight + right;
      if (aboveRight < quota && aboveRight + right >= quota) {
        smem.match[round] = {static_cast<uint32_t>(2 * tid + 1), aboveRight,
                             right};
      } else if (aboveLeft < quota && aboveLeft + left >= quota) {
        smem.match[round] = {static_cast<uint32_t>(2 * tid), aboveLeft, left};
      }
    }
    __syncthreads();

    K3RouteMatch const split = smem.match[round];
    threshold |= split.bin << shift;
    examinedMask |= 0xffu << shift;
#pragma unroll
    for (int i = 0; i < K3_ROUTE_VEC; ++i) {
      active[i] = active[i] && bins[i] == split.bin;
    }
    totalActive = split.equalCount;
    quota -= split.aboveCount;
    if (quota == totalActive) {
      takeAllEquals = true;
      break;
    }
    if (round + 1 < K3_ROUTE_RADIX_ROUNDS && tid < K3_ROUTE_RADIX_SIZE / 2) {
      smem.histogram[2 * tid] = 0;
      smem.histogram[2 * tid + 1] = 0;
    }
  }

  bool selected[K3_ROUTE_VEC];
  if (takeAllEquals) {
#pragma unroll
    for (int i = 0; i < K3_ROUTE_VEC; ++i) {
      selected[i] = active[i] || (keys[i] & examinedMask) > threshold;
    }
  } else {
    uint32_t activeCount = 0;
#pragma unroll
    for (int i = 0; i < K3_ROUTE_VEC; ++i)
      activeCount += active[i];
    uint32_t rank = k3BlockExclusiveSum(activeCount, smem.warpSum[1]);
#pragma unroll
    for (int i = 0; i < K3_ROUTE_VEC; ++i) {
      bool const equalWinner = active[i] && rank < quota;
      if (active[i])
        ++rank;
      selected[i] = equalWinner || (keys[i] & examinedMask) > threshold;
    }
  }

  uint32_t selectedCount = 0;
#pragma unroll
  for (int i = 0; i < K3_ROUTE_VEC; ++i)
    selectedCount += selected[i];
  uint32_t slot = k3BlockExclusiveSum(selectedCount, smem.warpSum[2]);
#pragma unroll
  for (int i = 0; i < K3_ROUTE_VEC; ++i) {
    if (selected[i] && slot < K3_PREP_TOPK) {
      smem.winnerId[slot] = firstExpert + i;
      smem.winnerKey[slot] = keys[i];
      smem.winnerWeight[slot] = weights[i];
      ++slot;
    }
  }
  __syncthreads();

  if (tid < K3_PREP_TOPK) {
    // Match TokenSpeed's existing Triton top-k order: selection score
    // descending, expert id ascending for ties.
    uint32_t const key = smem.winnerKey[tid];
    int32_t const id = smem.winnerId[tid];
    int rank = 0;
#pragma unroll
    for (int i = 0; i < K3_PREP_TOPK; ++i) {
      rank += smem.winnerKey[i] > key ||
              (smem.winnerKey[i] == key && smem.winnerId[i] < id);
    }
    float denominator = 0.0f;
#pragma unroll
    for (int i = 0; i < K3_PREP_TOPK; ++i) {
      denominator += smem.winnerWeight[i];
    }
    float weight = smem.winnerWeight[tid];
    if (renormalize)
      weight /= denominator > 0.0f ? denominator : 1.0f;
    weight *= routedScalingFactor;
    __nv_bfloat16 const weightBf16 = __float2bfloat16_rn(weight);
    topkWeights[int64_t(row) * weightsRowStride + rank] = weightBf16;
    topkIds[int64_t(row) * idsRowStride + rank] = id;
    uint32_t const bits = __bfloat16_as_ushort(weightBf16);
    packedTopk[int64_t(row) * packedRowStride + rank] =
        static_cast<int32_t>((static_cast<uint32_t>(id) << 16) | bits);
  }
}

__device__ __forceinline__ int k3CastToUe8m0(float x) {
  uint32_t const bits = __float_as_uint(x);
  int exponent = int((bits >> 23) & 0xffu);
  exponent += (bits & 0x7fffffu) != 0;
  return exponent;
}

__device__ __forceinline__ float k3InvUe8m0Scale(int exponent) {
  return __uint_as_float(uint32_t(254 - exponent) << 23);
}

template <typename InputT>
__device__ __forceinline__ float k3LoadQuantValue(InputT const *input,
                                                  int offset) {
  if constexpr (std::is_same_v<InputT, float>) {
    return input[offset];
  } else {
    return __bfloat162float(input[offset]);
  }
}

template <typename InputT>
__device__ __forceinline__ void
k3QuantMxfp8Row(InputT const *__restrict__ x, int64_t xRowStride,
                uint8_t *__restrict__ xQuant, int64_t quantRowStride,
                uint8_t *__restrict__ xScales, int row) {
  int const tid = int(threadIdx.x);
  int const group = tid >> 1;
  int const pairLane = tid & 1;
  int const col = group * K3_PREP_GROUP + pairLane * 16;
  auto const *input = x + int64_t(row) * xRowStride + col;

  float values[16];
  float localAbsMax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float const value = k3LoadQuantValue(input, i);
    values[i] = value;
    localAbsMax = fmaxf(localAbsMax, fabsf(value));
  }
  float const peerAbsMax = __shfl_xor_sync(0xffffffffu, localAbsMax, 1);
  float const absMax = fmaxf(fmaxf(localAbsMax, peerAbsMax), 1.0e-10f);
  int const scaleExponent = k3CastToUe8m0(absMax / K3_FP8_MAX);
  float const inverseScale = k3InvUe8m0Scale(scaleExponent);
  // The packed BF16 path matches FlashInfer/CuTe's BF16 multiply. The FP32
  // path deliberately retains FP32 multiplication, matching SGLang's
  // strided FP32 fused-front consumer.
  float const multiplier =
      std::is_same_v<InputT, float>
          ? inverseScale
          : __bfloat162float(__float2bfloat16_rn(inverseScale));

  uint4 quantized;
  auto *quantBytes = reinterpret_cast<uint8_t *>(&quantized);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float scaled = values[i] * multiplier;
    if constexpr (!std::is_same_v<InputT, float>) {
      scaled = __bfloat162float(__float2bfloat16_rn(scaled));
    }
    scaled = fminf(scaled, K3_FP8_MAX);
    quantBytes[i] = static_cast<uint8_t>(
        __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
  }
  auto *output = xQuant + int64_t(row) * quantRowStride + col;
  *reinterpret_cast<uint4 *>(output) = quantized;
  if (pairLane == 0) {
    xScales[row * K3_PREP_GROUPS + group] = static_cast<uint8_t>(scaleExponent);
  }
}

template <typename InputT, bool EnablePDL>
__global__ __launch_bounds__(K3_ROUTE_THREADS) void k3RoutePackQuantMxfp8Kernel(
    float const *__restrict__ logits, int64_t logitsRowStride,
    float const *__restrict__ bias, InputT const *__restrict__ x,
    int64_t xRowStride, __nv_bfloat16 *__restrict__ topkWeights,
    int64_t weightsRowStride, int32_t *__restrict__ topkIds,
    int64_t idsRowStride, int32_t *__restrict__ packedTopk,
    int64_t packedRowStride, uint8_t *__restrict__ xQuant,
    int64_t quantRowStride, uint8_t *__restrict__ xScales,
    float routedScalingFactor, bool renormalize, int numTokens) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (EnablePDL)
    cudaGridDependencySynchronize();
#endif
  if (int(blockIdx.x) < numTokens) {
    __shared__ K3RouteSmem routeSmem;
    k3RouteRow(logits, logitsRowStride, bias, topkWeights, weightsRowStride,
               topkIds, idsRowStride, packedTopk, packedRowStride,
               routedScalingFactor, renormalize, routeSmem);
  } else {
    k3QuantMxfp8Row(x, xRowStride, xQuant, quantRowStride, xScales,
                    int(blockIdx.x) - numTokens);
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (EnablePDL)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__global__ __launch_bounds__(K3_PREP_THREADS) void k3PackTopkQuantMxfp8Kernel(
    __nv_bfloat16 const *__restrict__ x, int64_t xRowStride,
    int32_t const *__restrict__ topkIds, int64_t idsRowStride,
    uint16_t const *__restrict__ topkWeightBits, int64_t weightsRowStride,
    int32_t *__restrict__ packedTopk, int64_t packedRowStride,
    uint8_t *__restrict__ xQuant, int64_t quantRowStride,
    uint8_t *__restrict__ xScales) {
  int const row = int(blockIdx.x);
  int const tid = int(threadIdx.x);

  if (tid < K3_PREP_TOPK) {
    uint32_t const id =
        static_cast<uint32_t>(topkIds[row * idsRowStride + tid]);
    uint32_t const weight = topkWeightBits[row * weightsRowStride + tid];
    packedTopk[row * packedRowStride + tid] =
        static_cast<int32_t>((id << 16) | weight);
  }

  int const group = tid >> 1;
  int const pairLane = tid & 1;
  int const col = group * K3_PREP_GROUP + pairLane * 16;
  auto const *input = x + int64_t(row) * xRowStride + col;

  float values[16];
  float localAbsMax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float const value = __bfloat162float(input[i]);
    values[i] = value;
    localAbsMax = fmaxf(localAbsMax, fabsf(value));
  }
  float const peerAbsMax = __shfl_xor_sync(0xffffffffu, localAbsMax, 1);
  float const absMax = fmaxf(fmaxf(localAbsMax, peerAbsMax), 1.0e-10f);
  int const scaleExponent = k3CastToUe8m0(absMax / K3_FP8_MAX);
  float const inverseScale = k3InvUe8m0Scale(scaleExponent);
  // The production FlashInfer/CuTe quantizer rounds the power-of-two
  // multiplier and the multiply itself in BF16 before the FP8 conversion.
  float const inverseScaleBf16 =
      __bfloat162float(__float2bfloat16_rn(inverseScale));

  uint4 quantized;
  auto *quantBytes = reinterpret_cast<uint8_t *>(&quantized);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float scaled =
        __bfloat162float(__float2bfloat16_rn(values[i] * inverseScaleBf16));
    // fminf sanitizes NaN/+inf to +448; -inf is saturated by the FP8 cast.
    scaled = fminf(scaled, K3_FP8_MAX);
    quantBytes[i] = static_cast<uint8_t>(
        __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
  }
  auto *output = xQuant + int64_t(row) * quantRowStride + col;
  *reinterpret_cast<uint4 *>(output) = quantized;
  if (pairLane == 0) {
    xScales[row * K3_PREP_GROUPS + group] = static_cast<uint8_t>(scaleExponent);
  }
}

}  // namespace tokenspeed

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void moe_finalize_fuse_shared(TensorView out, TensorView gemm2_out,
                              TensorView expanded_idx_to_permuted_idx,
                              TensorView expert_weights, TensorView shared_output,
                              int64_t top_k, bool enable_pdl) {
  TVM_FFI_ICHECK_EQ(out.ndim(), 2) << "out must be 2-D [numTokens, hiddenDim]";
  TVM_FFI_ICHECK_EQ(gemm2_out.ndim(), 2)
      << "gemm2_out must be 2-D [totalNumPaddedTokens, hiddenDimPadded]";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.ndim(), 1);
  TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2)
      << "expert_weights must be 2-D [numTokens, topK] or [numTokens, topK + numShared]";

  int const numTokens = int(out.size(0));
  int const hiddenDim = int(out.size(1));
  int const hiddenDimPadded = int(gemm2_out.size(1));
  TVM_FFI_ICHECK_LE(top_k, tokenspeed::MAX_TOPK);
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.size(0), numTokens * top_k);
  TVM_FFI_ICHECK_EQ(expert_weights.size(0), numTokens);
  TVM_FFI_ICHECK_GE(expert_weights.size(1), top_k);
  // Weight columns beyond topK are shared-expert-sink weights for the 3-D shared_output rows.
  int const numShared = int(expert_weights.size(1) - top_k);
  TVM_FFI_ICHECK_LE(numShared, tokenspeed::MAX_SHARED);

  bool const hasShared = shared_output.numel() > 0;
  TVM_FFI_ICHECK(numShared == 0 || hasShared)
      << "expert_weights has shared columns but shared_output is empty";
  if (hasShared) {
    if (numShared == 0) {
      TVM_FFI_ICHECK_EQ(shared_output.ndim(), 2);
      TVM_FFI_ICHECK_EQ(shared_output.size(0), numTokens);
      TVM_FFI_ICHECK_EQ(shared_output.size(1), hiddenDim);
    } else {
      TVM_FFI_ICHECK_EQ(shared_output.ndim(), 3)
          << "with shared weight columns, shared_output must be "
             "[numShared, numTokens, hiddenDim]";
      TVM_FFI_ICHECK_EQ(shared_output.size(0), numShared);
      TVM_FFI_ICHECK_EQ(shared_output.size(1), numTokens);
      TVM_FFI_ICHECK_EQ(shared_output.size(2), hiddenDim);
    }
  }

  auto const* inPtr = static_cast<tokenspeed::BF16 const*>(gemm2_out.data_ptr());
  auto const* expandedIdxPtr = static_cast<int const*>(expanded_idx_to_permuted_idx.data_ptr());
  auto const* sharedPtr = hasShared
                              ? static_cast<tokenspeed::BF16 const*>(shared_output.data_ptr())
                              : nullptr;
  auto* outPtr = static_cast<tokenspeed::BF16*>(out.data_ptr());

  cudaSetDevice(out.device().device_id);
  cudaStream_t const stream = get_stream(out.device());

  // Dispatch heuristic (matches flashinfer): few waves → general kernel,
  // many waves → vectorized. The 1184 threshold comes from 148 SMs × 8
  // blocks/SM on Blackwell.
  constexpr int kNumThreads = 256;
  int const numBlocksX = (hiddenDim + kNumThreads - 1) / kNumThreads;
  int const numBlocksY = std::min(8192, numTokens);
  bool const useVecLoad =
      (numBlocksX * numBlocksY) >= 1184 && (hiddenDim % 8 == 0) && (hiddenDimPadded % 8 == 0);

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;

  auto ew_dtype = expert_weights.dtype();
  if (ew_dtype == DLDataType{kDLFloat, 32, 1}) {
    tokenspeed::dispatchFinalize<float>(numTokens, hiddenDim, hiddenDimPadded, int(top_k),
                                        numShared, inPtr, expandedIdxPtr,
                                        expert_weights.data_ptr(), sharedPtr, outPtr, useVecLoad,
                                        stream, attrs, 1);
  } else if (ew_dtype == DLDataType{kDLBfloat, 16, 1}) {
    tokenspeed::dispatchFinalize<tokenspeed::BF16>(
        numTokens, hiddenDim, hiddenDimPadded, int(top_k), numShared, inPtr, expandedIdxPtr,
        expert_weights.data_ptr(), sharedPtr, outPtr, useVecLoad, stream, attrs, 1);
  } else {
    TVM_FFI_ICHECK(false) << "expert_weights dtype must be float32 or bfloat16";
  }

  cudaError_t const err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_finalize_fuse_shared launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_finalize_fuse_shared, moe_finalize_fuse_shared);

void moe_pack_topk_quant_mxfp8(TensorView packed_topk, TensorView x_quant,
                               TensorView x_scales, TensorView x,
                               TensorView topk_ids, TensorView topk_weights) {
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [numTokens, 3584]";
  TVM_FFI_ICHECK_EQ(x.size(1), tokenspeed::K3_PREP_HIDDEN);
  TVM_FFI_ICHECK_EQ(x.dtype(), dl_bfloat16);
  int const numTokens = int(x.size(0));
  TVM_FFI_ICHECK_GT(numTokens, 0);
  TVM_FFI_ICHECK_LE(numTokens, 64)
      << "prepared K3 MoE front is limited to decode batches <= 64";

  TVM_FFI_ICHECK_EQ(topk_ids.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_ids.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(topk_ids.size(1), tokenspeed::K3_PREP_TOPK);
  TVM_FFI_ICHECK_EQ(topk_ids.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(topk_weights.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), tokenspeed::K3_PREP_TOPK);
  TVM_FFI_ICHECK_EQ(topk_weights.dtype(), dl_bfloat16);

  TVM_FFI_ICHECK_EQ(packed_topk.ndim(), 2);
  TVM_FFI_ICHECK_EQ(packed_topk.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(packed_topk.size(1), tokenspeed::K3_PREP_TOPK);
  TVM_FFI_ICHECK_EQ(packed_topk.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(x_quant.ndim(), 2);
  TVM_FFI_ICHECK_EQ(x_quant.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(x_quant.size(1), tokenspeed::K3_PREP_HIDDEN);
  TVM_FFI_ICHECK_EQ(x_quant.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(x_scales.numel(), numTokens * tokenspeed::K3_PREP_GROUPS);
  TVM_FFI_ICHECK_EQ(x_scales.dtype(), dl_uint8);

  TVM_FFI_ICHECK_EQ(x.stride(1), 1);
  TVM_FFI_ICHECK_EQ(topk_ids.stride(1), 1);
  TVM_FFI_ICHECK_EQ(topk_weights.stride(1), 1);
  TVM_FFI_ICHECK_EQ(packed_topk.stride(1), 1);
  TVM_FFI_ICHECK_EQ(x_quant.stride(1), 1);

  cudaSetDevice(x.device().device_id);
  cudaStream_t const stream = get_stream(x.device());
  tokenspeed::k3PackTopkQuantMxfp8Kernel<<<
      numTokens, tokenspeed::K3_PREP_THREADS, 0, stream>>>(
      static_cast<__nv_bfloat16 const *>(x.data_ptr()), x.stride(0),
      static_cast<int32_t const *>(topk_ids.data_ptr()), topk_ids.stride(0),
      static_cast<uint16_t const *>(topk_weights.data_ptr()),
      topk_weights.stride(0), static_cast<int32_t *>(packed_topk.data_ptr()),
      packed_topk.stride(0), static_cast<uint8_t *>(x_quant.data_ptr()),
      x_quant.stride(0), static_cast<uint8_t *>(x_scales.data_ptr()));
  cudaError_t const err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_pack_topk_quant_mxfp8 launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_pack_topk_quant_mxfp8,
                              moe_pack_topk_quant_mxfp8);

void moe_route_pack_quant_mxfp8(TensorView topk_weights, TensorView topk_ids,
                                TensorView packed_topk, TensorView x_quant,
                                TensorView x_scales, TensorView router_logits,
                                TensorView correction_bias, TensorView x,
                                double routed_scaling_factor, bool renormalize,
                                bool enable_pdl) {
  TVM_FFI_ICHECK_EQ(router_logits.ndim(), 2);
  TVM_FFI_ICHECK_EQ(router_logits.size(1), tokenspeed::K3_ROUTE_EXPERTS);
  TVM_FFI_ICHECK_EQ(router_logits.dtype(), dl_float32);
  int const numTokens = int(router_logits.size(0));
  TVM_FFI_ICHECK_GT(numTokens, 0);
  TVM_FFI_ICHECK_LE(numTokens, 64);
  TVM_FFI_ICHECK_EQ(router_logits.stride(1), 1);

  TVM_FFI_ICHECK_EQ(correction_bias.ndim(), 1);
  TVM_FFI_ICHECK_EQ(correction_bias.size(0), tokenspeed::K3_ROUTE_EXPERTS);
  TVM_FFI_ICHECK_EQ(correction_bias.dtype(), dl_float32);

  TVM_FFI_ICHECK_EQ(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(x.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(x.size(1), tokenspeed::K3_PREP_HIDDEN);
  TVM_FFI_ICHECK(x.dtype() == dl_float32 || x.dtype() == dl_bfloat16)
      << "x must be FP32 or BF16";
  TVM_FFI_ICHECK_EQ(x.stride(1), 1);

  TVM_FFI_ICHECK_EQ(topk_weights.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), tokenspeed::K3_PREP_TOPK);
  TVM_FFI_ICHECK_EQ(topk_weights.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(topk_weights.stride(1), 1);
  TVM_FFI_ICHECK_EQ(topk_ids.ndim(), 2);
  TVM_FFI_ICHECK_EQ(topk_ids.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(topk_ids.size(1), tokenspeed::K3_PREP_TOPK);
  TVM_FFI_ICHECK_EQ(topk_ids.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(topk_ids.stride(1), 1);
  TVM_FFI_ICHECK_EQ(packed_topk.ndim(), 2);
  TVM_FFI_ICHECK_EQ(packed_topk.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(packed_topk.size(1), tokenspeed::K3_PREP_TOPK);
  TVM_FFI_ICHECK_EQ(packed_topk.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(packed_topk.stride(1), 1);

  TVM_FFI_ICHECK_EQ(x_quant.ndim(), 2);
  TVM_FFI_ICHECK_EQ(x_quant.size(0), numTokens);
  TVM_FFI_ICHECK_EQ(x_quant.size(1), tokenspeed::K3_PREP_HIDDEN);
  TVM_FFI_ICHECK_EQ(x_quant.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(x_quant.stride(1), 1);
  TVM_FFI_ICHECK_EQ(x_scales.numel(), numTokens * tokenspeed::K3_PREP_GROUPS);
  TVM_FFI_ICHECK_EQ(x_scales.dtype(), dl_uint8);

  cudaSetDevice(router_logits.device().device_id);
  cudaStream_t const stream = get_stream(router_logits.device());
  dim3 const grid(2 * numTokens);
  dim3 const block(tokenspeed::K3_ROUTE_THREADS);
  auto const *logitsPtr = static_cast<float const *>(router_logits.data_ptr());
  auto const *biasPtr = static_cast<float const *>(correction_bias.data_ptr());
  auto *weightsPtr = static_cast<__nv_bfloat16 *>(topk_weights.data_ptr());
  auto *idsPtr = static_cast<int32_t *>(topk_ids.data_ptr());
  auto *packedPtr = static_cast<int32_t *>(packed_topk.data_ptr());
  auto *quantPtr = static_cast<uint8_t *>(x_quant.data_ptr());
  auto *scalesPtr = static_cast<uint8_t *>(x_scales.data_ptr());
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = enable_pdl ? 1 : 0;
  if (x.dtype() == dl_float32) {
    if (enable_pdl) {
      cudaLaunchKernelEx(
          &config, tokenspeed::k3RoutePackQuantMxfp8Kernel<float, true>,
          logitsPtr, router_logits.stride(0), biasPtr,
          static_cast<float const *>(x.data_ptr()), x.stride(0), weightsPtr,
          topk_weights.stride(0), idsPtr, topk_ids.stride(0), packedPtr,
          packed_topk.stride(0), quantPtr, x_quant.stride(0), scalesPtr,
          float(routed_scaling_factor), renormalize, numTokens);
    } else {
      cudaLaunchKernelEx(
          &config, tokenspeed::k3RoutePackQuantMxfp8Kernel<float, false>,
          logitsPtr, router_logits.stride(0), biasPtr,
          static_cast<float const *>(x.data_ptr()), x.stride(0), weightsPtr,
          topk_weights.stride(0), idsPtr, topk_ids.stride(0), packedPtr,
          packed_topk.stride(0), quantPtr, x_quant.stride(0), scalesPtr,
          float(routed_scaling_factor), renormalize, numTokens);
    }
  } else {
    if (enable_pdl) {
      cudaLaunchKernelEx(
          &config, tokenspeed::k3RoutePackQuantMxfp8Kernel<__nv_bfloat16, true>,
          logitsPtr, router_logits.stride(0), biasPtr,
          static_cast<__nv_bfloat16 const *>(x.data_ptr()), x.stride(0),
          weightsPtr, topk_weights.stride(0), idsPtr, topk_ids.stride(0),
          packedPtr, packed_topk.stride(0), quantPtr, x_quant.stride(0),
          scalesPtr, float(routed_scaling_factor), renormalize, numTokens);
    } else {
      cudaLaunchKernelEx(
          &config,
          tokenspeed::k3RoutePackQuantMxfp8Kernel<__nv_bfloat16, false>,
          logitsPtr, router_logits.stride(0), biasPtr,
          static_cast<__nv_bfloat16 const *>(x.data_ptr()), x.stride(0),
          weightsPtr, topk_weights.stride(0), idsPtr, topk_ids.stride(0),
          packedPtr, packed_topk.stride(0), quantPtr, x_quant.stride(0),
          scalesPtr, float(routed_scaling_factor), renormalize, numTokens);
    }
  }
  cudaError_t const err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_route_pack_quant_mxfp8 launch failed: "
      << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_route_pack_quant_mxfp8,
                              moe_route_pack_quant_mxfp8);
