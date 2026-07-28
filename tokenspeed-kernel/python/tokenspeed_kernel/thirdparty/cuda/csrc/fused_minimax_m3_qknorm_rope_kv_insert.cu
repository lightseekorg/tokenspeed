/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Horizontally-fused MiniMax-M3 attention pre-processing kernel.
 *
 * Replaces the per-token Python sequence in
 * ``MiniMaxM3SparseAttention.forward`` / ``MiniMaxM3Attention.forward``:
 *
 *     q  = q_norm(q);  k = k_norm(k);  q, k = rotary_emb(pos, q, k)
 *     index_q = index_q_norm(index_q);  index_k = index_k_norm(index_k)
 *     index_q, index_k = rotary_emb(pos, index_q, index_k)
 *     _insert_kv(k, v, index_k)
 *
 * All branches share head_dim=128 and the *same* partial-NeoX RoPE table
 * (``rotary_dim`` rotated, the trailing dims pass through).  The four norms
 * are Gemma-style RMSNorm (``x * rsqrt(mean(x^2)+eps) * (1 + weight)``) with
 * independent weights.
 *
 * Everything lives in a single fused ``qkv`` tensor.  The sparse layer's
 * fused projection (MinimaxM3QKVParallelLinearWithIndexer) emits, per token::
 *
 *     [ q | k | v | index_q | index_k ]   (the "5 results")
 *
 * while the dense layer emits just ``[ q | k | v ]``.  The kernel reads the
 * index branch straight out of that packed row -- no separate index tensors.
 *
 * One kernel, one grid; each warp owns one (token, head-slot) pair.  Slot
 * enumeration per token:
 *     [0, nq)                         Q  heads   -> norm(q_w)  + RoPE, write
 * qkv [nq, nq+nkv)                    K  heads   -> norm(k_w)  + RoPE, write
 * qkv
 *                                                  (+ insert into key cache)
 *     [nq+nkv, nq+2*nkv)              V  heads   -> insert into value cache
 *     IQ heads (niq)                             -> norm(iq_w) + RoPE, write iq
 *     IK       (1)                               -> norm(ik_w) + RoPE
 *                                                  (+ insert into index cache)
 *
 * The IQ/IK warps address the index_q/index_k sub-blocks *inside* qkv at the
 * fixed physical offsets (nq+2*nkv)*128 and (nq+2*nkv+niq)*128.
 *
 * Dense vs sparse row layout and index-branch processing are separate template
 * choices. Skip-index-topk reuse layers still have sparse rows and insert main
 * K/V cache entries, but compile away index_q/index_k work and index-cache
 * writes.
 *
 * Q/K and (sparse) index_q/index_k are all rewritten in place inside the fused
 * ``qkv`` tensor.  Caches (bf16) are scatter-written by slot.
 */

#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "tvm_ffi_utils.h"

// TokenSpeed vendoring note: this kernel is the vLLM
// fused_minimax_m3_qknorm_rope_kv_insert CUDA kernel, adapted to TokenSpeed's
// TVM-FFI (no-libtorch) build. The device algorithm is unchanged; only the
// includes, scalar<->float conversions, the fp8 helpers, and the host binding
// were ported. NVIDIA-only (ROCm branches are inert).

#ifndef FINAL_MASK
  #define FINAL_MASK 0xffffffffu
#endif

namespace vllm {

// Inlined from vLLM quant_utils (identity-scale path only) to avoid pulling in
// its attention-dtype header tree. Only auto (unquantized) + e4m3/e5m2 used.
enum class Fp8KVCacheDataType { kAuto = 0, kFp8E4M3 = 1, kFp8E5M2 = 2 };

inline Fp8KVCacheDataType get_fp8_kv_cache_data_type(const std::string& s) {
  if (s == "auto" || s.empty()) return Fp8KVCacheDataType::kAuto;
  if (s == "fp8" || s == "fp8_e4m3") return Fp8KVCacheDataType::kFp8E4M3;
  if (s == "fp8_e5m2") return Fp8KVCacheDataType::kFp8E5M2;
  TVM_FFI_ICHECK(false) << "unsupported kv_cache_dtype: " << s;
  return Fp8KVCacheDataType::kAuto;
}

namespace minimax_m3_fused_ops {

namespace {
inline int getSMVersion() {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, dev);
  return props.major * 10 + props.minor;
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Constants (hard-coded for MiniMax-M3-preview).
// ────────────────────────────────────────────────────────────────────────────
constexpr int kHeadDim = 128;
constexpr int kNumLanes = 32;
constexpr int kElemsPerLane = kHeadDim / kNumLanes;  // 4

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

// Gemma RMSNorm over the full head (no-op when ``weight == nullptr``), rounded
// back to scalar_t like the materialized unfused norm output, followed by
// partial NeoX RoPE on the leading ``rotary_dim`` dims. Each lane owns
// ``kElemsPerLane`` contiguous dims [laneId*4, laneId*4+4).
template <typename scalar_t>
__device__ __forceinline__ void normAndRope(
    float (&elems)[kElemsPerLane], int const laneId, float const eps,
    scalar_t const* __restrict__ weight,  // [kHeadDim] or nullptr (no norm)
    bool const do_rope, int const rotary_dim,
    float const* __restrict__ cos_ptr,  // fp32 cos_sin_cache + pos*rotary_dim
    bool const apply_norm) {
  // ── Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + w) ──────────────────
  if (apply_norm) {
    float sumsq = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) sumsq += elems[i] * elems[i];
    sumsq = warpReduceSum(sumsq);
    float const rms_rcp = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      int const dim = laneId * kElemsPerLane + i;
      float const w = 1.0f + static_cast<float>(weight[dim]);
      elems[i] = elems[i] * rms_rcp * w;
    }
  }

  // ── Partial NeoX RoPE on dims [0, rotary_dim) ──────────────────────────
  // half = rotary_dim/2.  Pair (i, i+half) for i in [0, half).  Lane L owns
  // dims [4L, 4L+4); since half is a multiple of 4, a lane lies wholly in the
  // first half (own=x[i]) or second half (own=x[i+half]); its partner lives
  // ``half/4`` lanes away (XOR with that distance).
  if (do_rope) {
    int const half = rotary_dim / 2;
    int const dim0 = laneId * kElemsPerLane;
    bool const in_rope = dim0 < rotary_dim;
    int const lane_xor = half / kElemsPerLane;  // partner-lane distance

    float partner[kElemsPerLane];
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      partner[i] = __shfl_xor_sync(FINAL_MASK, elems[i], lane_xor, 32);
    }
    if (in_rope) {
      bool const first_half = dim0 < half;
      int const i_base = first_half ? dim0 : (dim0 - half);  // cos/sin index
      float const* sin_ptr = cos_ptr + half;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i++) {
        float const c = static_cast<float>(cos_ptr[i_base + i]);
        float const s = static_cast<float>(sin_ptr[i_base + i]);
        if (first_half) {
          elems[i] = elems[i] * c - partner[i] * s;
        } else {
          elems[i] = elems[i] * c + partner[i] * s;
        }
      }
    }
  }
}

// bf16/fp16 <-> float element converters.
template <typename scalar_t>
__device__ __forceinline__ float elemToFloat(scalar_t x) {
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>)
    return __bfloat162float(x);
  else
    return __half2float(x);
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t floatToElem(float x) {
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>)
    return __float2bfloat16(x);
  else
    return __float2half(x);
}

// Load 4 contiguous bf16/fp16 -> 4 fp32 registers.
template <typename scalar_t>
__device__ __forceinline__ void loadElems(scalar_t const* __restrict__ src,
                                          float (&elems)[kElemsPerLane]) {
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++)
    elems[i] = elemToFloat<scalar_t>(src[i]);
}

// Store 4 fp32 registers -> 4 contiguous bf16/fp16.
template <typename scalar_t>
__device__ __forceinline__ void storeElems(
    scalar_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
#pragma unroll
  for (int i = 0; i < kElemsPerLane; i++)
    dst[i] = floatToElem<scalar_t>(elems[i]);
}

// Main K/V cache store. kAuto = unquantized (cache_t == scalar_t); fp8 cache
// dtypes use the scaled-convert path with identity scale.
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ void storeCacheElems(
    cache_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
    // kAuto means unquantized KV cache here: cache_t == scalar_t, so store the
    // model dtype directly. FP8 cache dtypes use the conversion path below.
    storeElems<scalar_t>(reinterpret_cast<scalar_t*>(dst), elems);
  } else {
    // fp8 cache (identity scale 1.0): direct float -> e4m3/e5m2 byte.
    constexpr __nv_fp8_interpretation_t kInterp =
        (kv_dt == Fp8KVCacheDataType::kFp8E4M3) ? __NV_E4M3 : __NV_E5M2;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
      dst[i] = static_cast<cache_t>(
          __nv_cvt_float_to_fp8(elems[i], __NV_SATFINITE, kInterp));
    }
  }
}

// Store 4 fp32 registers -> 4 contiguous E4M3 FP8 bytes (direct cast,
// saturating to ±448). Used for the fp8 indexer-Q / index-K outputs; no scale
// (RMSNorm outputs are O(1) and the score path only needs relative block
// ordering).
__device__ __forceinline__ void storeElemsFp8(
    uint8_t* __restrict__ dst, float const (&elems)[kElemsPerLane]) {
  constexpr float kFp8Max = 448.0f;
#ifndef USE_ROCM
  __nv_fp8x2_storage_t out2[kElemsPerLane / 2];
  #pragma unroll
  for (int i = 0; i < kElemsPerLane / 2; i++) {
    float2 vv = make_float2(elems[2 * i], elems[2 * i + 1]);
    vv.x = fminf(fmaxf(vv.x, -kFp8Max), kFp8Max);
    vv.y = fminf(fmaxf(vv.y, -kFp8Max), kFp8Max);
    out2[i] = __nv_cvt_float2_to_fp8x2(vv, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<uint32_t const*>(out2);
#else
  #pragma unroll
  for (int i = 0; i < kElemsPerLane; i++) {
    float vv = fminf(fmaxf(elems[i], -kFp8Max), kFp8Max);
    dst[i] = rocm_cvt_float_to_fp8_e4m3(vv);
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel
// ────────────────────────────────────────────────────────────────────────────
// Grid: 1D, ceil(num_tokens * slots_per_token / warps_per_block).
// Each warp = one (token, slot).
//
// `kHasIndex`, `kProcessIndex`, and `kInsertKV` are compile-time template
// bools, so branch decisions that distinguish the dense layer from the sparse
// layer (index slots, KV/index inserts, V slots) fold away per instantiation.
// Slots per token:
//     Q : nq                            (always — norm+RoPE)
//     K : nkv                           (always — norm+RoPE; +K-cache insert)
//     V : nkv  only if kInsertKV        (V-cache insert; no warps in dense)
//     IQ: niq  only if kProcessIndex    (norm+RoPE)
//     IK: 1    only if kProcessIndex    (norm+RoPE; +index-cache insert)
// cache_t/kv_dt: main attention KV-cache dtype (auto/fp8). out_idx_t/kFp8Idx:
// indexer index-K cache + index-Q output dtype (scalar_t or e4m3 byte).
// kHasIndex means the qkv row is laid out as sparse [q|k|v|index_q|index_k].
// kProcessIndex controls whether this launch actually norms/ropes the index
// branch and writes index_q/index_k outputs. Skip-index-topk reuse layers keep
// kHasIndex=true but set kProcessIndex=false.
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          typename out_idx_t, bool kHasIndex, bool kInsertKV,
          bool kProcessIndex,
          bool kFp8Idx>
__global__ void fusedMiniMaxM3QNormRopeKVInsertKernel(
    scalar_t* __restrict__ qkv,  // [N, qkv_row] in/out (packs index if sparse)
    scalar_t* __restrict__ q_out,         // [N, nq*128] contiguous, or nullptr
    out_idx_t* __restrict__ index_q_out,  // [N, niq*128]; scalar_t or e4m3 byte
    scalar_t const* __restrict__ q_norm_w,
    scalar_t const* __restrict__ k_norm_w,
    scalar_t const* __restrict__ iq_norm_w,
    scalar_t const* __restrict__ ik_norm_w,
    float const* __restrict__ cos_sin_cache,  // fp32 [max_pos, rotary_dim]
    int64_t const* __restrict__ positions,       // [N] i64
    int64_t const* __restrict__ slot_mapping,    // main K/V slots or nullptr
    int64_t const* __restrict__ index_slot_mapping,  // index K slots/nullptr
    cache_t* __restrict__ k_cache,        // [num_slots, nkv, 128] or nullptr
    cache_t* __restrict__ v_cache,        // [num_slots, nkv, 128] or nullptr
    out_idx_t* __restrict__ index_cache,  // [nb*bs, 128]; scalar_t or e4m3 byte
    float const eps, int const rotary_dim, int const num_tokens, int const nq,
    int const nkv, int const niq, int const block_size,
    // TokenSpeed separate flat k_cache/v_cache strides (in elements) for logical
    // shape [num_slots, nkv, head_dim]. The content (dim) is innermost-contiguous.
    int64_t const kv_s_slot, int64_t const kv_s_head, int64_t const kv_s_dim) {
    // (Pre-Ampere bf16 guard removed for the NVIDIA sm90+ vendored build.)
    int const warpsPerBlock = blockDim.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);

    static_assert(!kProcessIndex || kHasIndex,
                  "index processing requires sparse row layout");

    // Slot layout (compile-time gated: dense has neither V nor index slots).
    int const v_slots = kInsertKV ? nkv : 0;
    int const idx_slots = kProcessIndex ? niq + 1 : 0;
    int const slots_per_token = nq + nkv + v_slots + idx_slots;

    int const tokenIdx = globalWarpIdx / slots_per_token;
    int const slot = globalWarpIdx % slots_per_token;
    if (tokenIdx >= num_tokens) return;

    // Slot boundaries.
    int const k_begin = nq;
    int const v_begin = nq + nkv;             // valid only when kInsertKV
    int const iq_begin = nq + nkv + v_slots;  // index block start
    int const ik_slot = iq_begin + niq;       // valid only when kProcessIndex

    bool const isQ = slot < k_begin;
    bool const isK = slot >= k_begin && slot < v_begin;
    bool isV = false;
    if constexpr (kInsertKV) isV = slot >= v_begin && slot < v_begin + nkv;
    bool isIQ = false, isIK = false;
    if constexpr (kProcessIndex) {
      isIQ = slot >= iq_begin && slot < ik_slot;
      isIK = slot == ik_slot;
    }

    int const dim_base = laneId * kElemsPerLane;
    // Physical row width of qkv: the dense layer packs [q|k|v]; the sparse
    // layer additionally packs [index_q (niq heads) | index_k (1 head)].
    int const qkv_row = (nq + 2 * nkv + (kHasIndex ? (niq + 1) : 0)) * kHeadDim;

    // ── Resolve source pointer + per-branch parameters. ────────────────────
    scalar_t* row_ptr = nullptr;       // in-place output location
    scalar_t const* norm_w = nullptr;  // nullptr -> skip norm (V)
    bool do_rope = true;
    int head = 0;  // kv head index for inserts

    if (isQ) {
      row_ptr =
          qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
      norm_w = q_norm_w;
    } else if (isK) {
      head = slot - k_begin;
      row_ptr =
          qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
      norm_w = k_norm_w;
    } else if (isV) {
      // qkv V section starts at slot index (nq + nkv): slot * kHeadDim is the
      // correct in-tensor offset.
      head = slot - v_begin;
      row_ptr =
          qkv + static_cast<int64_t>(tokenIdx) * qkv_row + slot * kHeadDim;
      norm_w = nullptr;  // V: no norm, no rope
      do_rope = false;
    } else if (isIQ) {
      // index_q sub-block lives at physical offset (nq+2*nkv)*128 in qkv.
      int const ih = slot - iq_begin;
      row_ptr = qkv + static_cast<int64_t>(tokenIdx) * qkv_row +
                (nq + 2 * nkv + ih) * kHeadDim;
      norm_w = iq_norm_w;
    } else if (isIK) {
      // Single shared index key at (nq+2*nkv+niq)*128.
      row_ptr = qkv + static_cast<int64_t>(tokenIdx) * qkv_row +
                (nq + 2 * nkv + niq) * kHeadDim;
      norm_w = ik_norm_w;
    } else {
      return;
    }

    // Store destination.  Q and index_q are gathered into dedicated contiguous
    // output buffers (when provided) so the downstream SM100 sparse kernel's
    // flat TMA descriptor can address them as [tokens*heads, head_dim]; this
    // folds the de-interleaving into the store the kernel already does, instead
    // of a separate q.contiguous() copy.  Everything else stays in place.
    scalar_t* store_ptr = row_ptr;
    if (isQ && q_out != nullptr) {
      store_ptr = q_out + static_cast<int64_t>(tokenIdx) * nq * kHeadDim +
                  slot * kHeadDim;
    } else if (isIQ && index_q_out != nullptr) {
      // bf16 index_q_out: gather here. fp8: written by the explicit fp8 store.
      if constexpr (!kFp8Idx) {
        store_ptr = index_q_out +
                    static_cast<int64_t>(tokenIdx) * niq * kHeadDim +
                    (slot - iq_begin) * kHeadDim;
      }
    }

    // PDL: wait for the predecessor kernel (the qkv-projection GEMM that
    // produces ``qkv``) to finish before touching any global memory.  No-op
    // when PDL is not enabled on the launch.  The CUDA runtime wrapper emits
    // the griddepcontrol.wait PTX with the required memory clobber internally.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    // ── Load -> norm+rope (fp32) -> store back in place. ───────────────────
    float elems[kElemsPerLane];
    loadElems<scalar_t>(row_ptr + dim_base, elems);

    if (!isV) {
      int64_t const pos = positions[tokenIdx];
      float const* cos_ptr = cos_sin_cache + pos * rotary_dim;
      normAndRope<scalar_t>(elems, laneId, eps, norm_w, do_rope, rotary_dim,
                            cos_ptr, /*apply_norm=*/norm_w != nullptr);
      if constexpr (kFp8Idx) {
        // index_q is e4m3 bytes; Q/K (and in-place index_k) stay scalar_t.
        if (isIQ && index_q_out != nullptr) {
          storeElemsFp8(index_q_out +
                            static_cast<int64_t>(tokenIdx) * niq * kHeadDim +
                            (slot - iq_begin) * kHeadDim + dim_base,
                        elems);
        } else {
          storeElems<scalar_t>(store_ptr + dim_base, elems);
        }
      } else {
        storeElems<scalar_t>(store_ptr + dim_base, elems);
      }
    }

    // ── Cache inserts (sparse serving only). ───────────────────────────────
    if constexpr (kInsertKV) {
      // Guard (not early-return) so every thread reaches the PDL trigger below.
      int64_t sm = -1;
      if (isK || isV) {
        sm = slot_mapping[tokenIdx];
      } else if constexpr (kProcessIndex) {
        if (isIK) sm = index_slot_mapping[tokenIdx];
      }
      if (sm >= 0) {  // skip padded / unscheduled tokens
        if (isIK) {
          if constexpr (kFp8Idx) {
            storeElemsFp8(index_cache + sm * kHeadDim + dim_base, elems);
          } else {
            storeElems<scalar_t>(index_cache + sm * kHeadDim + dim_base, elems);
          }
        } else if (isK || isV) {
          // TokenSpeed: separate flat slot-indexed k_cache / v_cache, logical
          // shape [num_slots, nkv, head_dim]. ``sm`` is the direct slot row; the
          // physical layout rides the passed strides (block_size unused here).
          cache_t* dst = isK ? k_cache : v_cache;
          int64_t const off =
              sm * kv_s_slot + head * kv_s_head + dim_base * kv_s_dim;
          storeCacheElems<scalar_t, cache_t, kv_dt>(dst + off, elems);
        }
      }
    }

    // PDL: signal that this kernel is done so a dependent successor may launch
    // early.  No-op when PDL is not enabled on the launch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void launchFusedMiniMaxM3(
    scalar_t* qkv, scalar_t* q_out, void* index_q_out, scalar_t const* q_norm_w,
    scalar_t const* k_norm_w, scalar_t const* iq_norm_w,
    scalar_t const* ik_norm_w, float const* cos_sin_cache,
    int64_t const* positions, int64_t const* slot_mapping,
    int64_t const* index_slot_mapping, cache_t* k_cache, cache_t* v_cache,
    void* index_cache, float const eps, int const rotary_dim,
    int const num_tokens, int const nq, int const nkv, int const niq,
    int const block_size, int64_t const kv_s_slot, int64_t const kv_s_head,
    int64_t const kv_s_dim, bool const has_index, bool const insert_kv,
    bool const process_index, bool const fp8_idx, bool const enable_pdl,
    cudaStream_t stream) {
  // Index outputs are scalar_t (bf16) or e4m3 bytes (uint8_t); reinterpret the
  // void* pointers per instantiation in the LAUNCH macro.
  // Slot count must match the kernel's compile-time gating.
  int const v_slots = insert_kv ? nkv : 0;
  int const idx_slots = process_index ? niq + 1 : 0;
  int const slots_per_token = nq + nkv + v_slots + idx_slots;

  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens) * slots_per_token;
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (grid == 0) return;

#ifndef USE_ROCM
  // PDL: enable programmatic stream serialization when the caller requests it
  // and the hardware supports it (SM90+).  Otherwise leave numAttrs = 0 and
  // launch as a regular kernel via cudaLaunchKernelEx.
  static int const sm_version = getSMVersion();
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (enable_pdl && sm_version >= 90) ? 1 : 0;

  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)                 \
    cudaLaunchKernelEx(                                                        \
        &config,                                                               \
        fusedMiniMaxM3QNormRopeKVInsertKernel<scalar_t, cache_t, kv_dt, OUT_T, \
                                              HAS_INDEX, INSERT,               \
                                              PROCESS_INDEX, FP8>,             \
        qkv, q_out, reinterpret_cast<OUT_T*>(index_q_out), q_norm_w, k_norm_w, \
        iq_norm_w, ik_norm_w, cos_sin_cache, positions, slot_mapping,          \
        index_slot_mapping, k_cache, v_cache,                                  \
        reinterpret_cast<OUT_T*>(index_cache), eps, rotary_dim, num_tokens,    \
        nq, nkv, niq, block_size, kv_s_slot, kv_s_head, kv_s_dim)
#else
  // ROCm: standard kernel launch syntax (no PDL/stream serialization).
  // clang-format off
  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)              \
    fusedMiniMaxM3QNormRopeKVInsertKernel<                                  \
        scalar_t, cache_t, kv_dt, OUT_T, HAS_INDEX, INSERT, PROCESS_INDEX,  \
        FP8><<<grid, kBlockSize, 0, stream>>>(                              \
        qkv, q_out, reinterpret_cast<OUT_T*>(index_q_out), q_norm_w,        \
        k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions,           \
        slot_mapping, index_slot_mapping, k_cache, v_cache,                 \
        reinterpret_cast<OUT_T*>(index_cache), eps, rotary_dim, num_tokens, \
        nq, nkv, niq, block_size, kv_s_slot, kv_s_head, kv_s_dim)
  // clang-format on
#endif

  if (has_index) {
    if (!process_index) {
      if (insert_kv) {
        LAUNCH(true, true, false, false, scalar_t);
      } else {
        LAUNCH(true, false, false, false, scalar_t);
      }
    } else if (insert_kv) {
      if (fp8_idx) {
        LAUNCH(true, true, true, true,
               uint8_t);  // sparse serving, fp8 index outputs
      } else {
        LAUNCH(true, true, true, false, scalar_t);  // sparse serving, bf16
      }
    } else {
      if (fp8_idx) {
        LAUNCH(true, false, true, true,
               uint8_t);  // sparse profiling, fp8 index_q
      } else {
        LAUNCH(true, false, true, false, scalar_t);  // sparse profiling, bf16
      }
    }
  } else {
    // Dense layer: never has an index branch and never inserts here (the
    // generic Attention layer owns the KV insert).
    LAUNCH(false, false, false, false, scalar_t);
  }
#undef LAUNCH
}

}  // namespace minimax_m3_fused_ops
}  // namespace vllm

// ────────────────────────────────────────────────────────────────────────────
// TVM-FFI host binding (ported from vLLM's torch::stable wrapper).
// kv_cache_dtype is passed as an int code: 0=auto, 1=fp8_e4m3, 2=fp8_e5m2.
// ────────────────────────────────────────────────────────────────────────────
using tvm::ffi::Optional;

void fused_minimax_m3_qknorm_rope_kv_insert(
    TensorView qkv, TensorView q_norm_weight, TensorView k_norm_weight,
    TensorView cos_sin_cache, TensorView positions, int64_t num_heads,
    int64_t num_kv_heads, int64_t rotary_dim, double eps,
    Optional<TensorView> index_q_norm_weight,
    Optional<TensorView> index_k_norm_weight, int64_t num_index_heads,
    Optional<TensorView> slot_mapping, Optional<TensorView> index_slot_mapping,
    Optional<TensorView> k_cache, Optional<TensorView> v_cache,
    Optional<TensorView> index_cache,
    int64_t block_size, Optional<TensorView> q_out,
    Optional<TensorView> index_q_out, int64_t kv_cache_dtype,
    bool skip_index_branch, bool enable_pdl) {
  namespace mm = vllm::minimax_m3_fused_ops;
  constexpr int kHeadDim = mm::kHeadDim;

  TVM_FFI_ICHECK(qkv.dtype() == dl_bfloat16 || qkv.dtype() == dl_float16)
      << "qkv must be float16 or bfloat16";
  TVM_FFI_ICHECK(positions.dtype() == dl_int64) << "positions must be int64";
  TVM_FFI_ICHECK(cos_sin_cache.dtype() == dl_float32)
      << "cos_sin_cache must be float32 (RoPE applied at fp32 precision)";
  TVM_FFI_ICHECK(cos_sin_cache.ndim() == 2 &&
                 cos_sin_cache.size(1) == rotary_dim)
      << "cos_sin_cache shape [max_pos, rotary_dim]";
  TVM_FFI_ICHECK(q_norm_weight.dtype() == qkv.dtype() &&
                 k_norm_weight.dtype() == qkv.dtype())
      << "q/k norm weight dtype must match qkv";
  TVM_FFI_ICHECK(q_norm_weight.numel() == kHeadDim &&
                 k_norm_weight.numel() == kHeadDim)
      << "q/k norm weight must have 128 elements";
  TVM_FFI_ICHECK(rotary_dim > 0 && rotary_dim % 8 == 0 && rotary_dim <= kHeadDim)
      << "rotary_dim must be a positive multiple of 8 and <= 128";

  int const num_tokens = static_cast<int>(qkv.size(0));
  int const nq = static_cast<int>(num_heads);
  int const nkv = static_cast<int>(num_kv_heads);
  int const niq = static_cast<int>(num_index_heads);
  bool const has_index = niq > 0;
  bool const insert_kv = k_cache.has_value();
  bool const process_index = has_index && !skip_index_branch;
  vllm::Fp8KVCacheDataType const kv_dt =
      static_cast<vllm::Fp8KVCacheDataType>(kv_cache_dtype);

  int const expected_row =
      (nq + 2 * nkv + (has_index ? niq + 1 : 0)) * kHeadDim;
  TVM_FFI_ICHECK(qkv.size(1) == expected_row)
      << "qkv last dim must be (nq + 2*nkv + niq + 1)*128 sparse / (nq+2*nkv)*128 dense";
  TVM_FFI_ICHECK(!insert_kv || has_index)
      << "insert mode (kv_cache) requires the index branch (sparse layer)";
  TVM_FFI_ICHECK(has_index || !skip_index_branch)
      << "skip_index_branch requires sparse qkv rows";
  if (process_index) {
    TVM_FFI_ICHECK(index_q_norm_weight.has_value() &&
                   index_k_norm_weight.has_value())
        << "index branch requires both index norm weights";
    TVM_FFI_ICHECK(index_q_norm_weight.value().numel() == kHeadDim &&
                   index_k_norm_weight.value().numel() == kHeadDim)
        << "index norm weights must have 128 elements";
  }

  int64_t kv_s_slot = 0, kv_s_head = 0, kv_s_dim = 0;
  if (insert_kv) {
    TVM_FFI_ICHECK(slot_mapping.has_value() &&
                   slot_mapping.value().dtype() == dl_int64)
        << "insert mode requires int64 slot_mapping";
    TVM_FFI_ICHECK(v_cache.has_value()) << "insert mode requires v_cache";
    TensorView kc = k_cache.value();
    TensorView vc = v_cache.value();
    // TokenSpeed separate flat caches: [num_slots, nkv, head_dim].
    TVM_FFI_ICHECK(kc.ndim() == 3 && kc.stride(2) == 1)
        << "k_cache must be [num_slots, nkv, head_dim], contiguous content dim";
    TVM_FFI_ICHECK(vc.ndim() == 3 && vc.stride(0) == kc.stride(0) &&
                   vc.stride(1) == kc.stride(1) && vc.stride(2) == kc.stride(2))
        << "v_cache must share k_cache's [num_slots, nkv, head_dim] layout";
    if (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
      TVM_FFI_ICHECK(kc.dtype() == qkv.dtype() && vc.dtype() == qkv.dtype())
          << "auto k/v_cache dtype must match qkv";
    } else {
      TVM_FFI_ICHECK(kc.dtype() == dl_uint8 && vc.dtype() == dl_uint8)
          << "fp8 k/v_cache must use uint8 storage";
    }
    if (process_index)
      TVM_FFI_ICHECK(index_cache.has_value() &&
                     (index_cache.value().dtype() == qkv.dtype() ||
                      index_cache.value().dtype() == dl_float8_e4m3fn))
          << "index_cache must match qkv dtype or fp8 e4m3";
    kv_s_slot = kc.stride(0);
    kv_s_head = kc.stride(1);
    kv_s_dim = kc.stride(2);
  }

  bool const fp8_idx =
      process_index &&
      ((index_cache.has_value() &&
        index_cache.value().dtype() == dl_float8_e4m3fn) ||
       (index_q_out.has_value() &&
        index_q_out.value().dtype() == dl_float8_e4m3fn));

  void* qkv_ptr = qkv.data_ptr();
  void* q_out_ptr = q_out.has_value() ? q_out.value().data_ptr() : nullptr;
  void* index_q_out_ptr =
      index_q_out.has_value() ? index_q_out.value().data_ptr() : nullptr;
  void const* q_norm_ptr = q_norm_weight.data_ptr();
  void const* k_norm_ptr = k_norm_weight.data_ptr();
  void const* iq_norm_ptr =
      process_index ? index_q_norm_weight.value().data_ptr() : nullptr;
  void const* ik_norm_ptr =
      process_index ? index_k_norm_weight.value().data_ptr() : nullptr;
  void const* csc_ptr = cos_sin_cache.data_ptr();
  int64_t const* pos_ptr = static_cast<int64_t const*>(positions.data_ptr());
  int64_t const* slot_ptr =
      insert_kv ? static_cast<int64_t const*>(slot_mapping.value().data_ptr())
                : nullptr;
  int64_t const* idx_slot_ptr = nullptr;
  if (insert_kv && process_index)
    idx_slot_ptr = static_cast<int64_t const*>(
        (index_slot_mapping.has_value() ? index_slot_mapping.value()
                                        : slot_mapping.value())
            .data_ptr());
  void* k_cache_ptr = insert_kv ? k_cache.value().data_ptr() : nullptr;
  void* v_cache_ptr = insert_kv ? v_cache.value().data_ptr() : nullptr;
  void* index_cache_ptr =
      (insert_kv && process_index) ? index_cache.value().data_ptr() : nullptr;

  cudaSetDevice(qkv.device().device_id);
  cudaStream_t stream = get_stream(qkv.device());

#define TS_CALL_MM3(ST, CACHE_T, KV_DTYPE)                                    \
  mm::launchFusedMiniMaxM3<ST, CACHE_T, KV_DTYPE>(                            \
      reinterpret_cast<ST*>(qkv_ptr), reinterpret_cast<ST*>(q_out_ptr),       \
      index_q_out_ptr, reinterpret_cast<ST const*>(q_norm_ptr),               \
      reinterpret_cast<ST const*>(k_norm_ptr),                                \
      reinterpret_cast<ST const*>(iq_norm_ptr),                               \
      reinterpret_cast<ST const*>(ik_norm_ptr),                               \
      static_cast<float const*>(csc_ptr), pos_ptr, slot_ptr, idx_slot_ptr,   \
      reinterpret_cast<CACHE_T*>(k_cache_ptr),                                \
      reinterpret_cast<CACHE_T*>(v_cache_ptr), index_cache_ptr,              \
      static_cast<float>(eps), static_cast<int>(rotary_dim), num_tokens, nq,  \
      nkv, niq, static_cast<int>(block_size), kv_s_slot, kv_s_head, kv_s_dim, \
      has_index, insert_kv, process_index, fp8_idx, enable_pdl, stream)

#define TS_DISPATCH_KV(ST)                                                    \
  if (kv_dt == vllm::Fp8KVCacheDataType::kAuto)                               \
    TS_CALL_MM3(ST, ST, vllm::Fp8KVCacheDataType::kAuto);                     \
  else if (kv_dt == vllm::Fp8KVCacheDataType::kFp8E4M3)                       \
    TS_CALL_MM3(ST, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);             \
  else                                                                        \
    TS_CALL_MM3(ST, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2)

  if (qkv.dtype() == dl_bfloat16) {
    TS_DISPATCH_KV(__nv_bfloat16);
  } else {
    TS_DISPATCH_KV(__half);
  }
#undef TS_DISPATCH_KV
#undef TS_CALL_MM3
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_minimax_m3_qknorm_rope_kv_insert,
                              fused_minimax_m3_qknorm_rope_kv_insert);
