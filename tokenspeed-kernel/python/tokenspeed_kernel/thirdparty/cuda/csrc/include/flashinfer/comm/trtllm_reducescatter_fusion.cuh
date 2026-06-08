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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include <cstdint>
#include <cuda/std/optional>
#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../fp4_layout.cuh"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace trtllm_reducescatter_fusion {

using flashinfer::QuantizationSFLayout;

namespace details {

static constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
static constexpr int CVT_FP4_SF_VEC_SIZE = 16;
static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

}  // namespace details

enum class ReduceScatterFusionPattern : int {
  kReduceScatter = 0,
  kRSResidualRMSNorm = 1,
  kRSResidualRMSNormFP8Quant = 2,
  kRSResidualRMSNormFP4Quant = 3,
  kRSResidualRMSNormOutFP8Quant = 4,
  kRSResidualRMSNormOutFP4Quant = 5,
  kRSResidualRMSNormFP8BlockWiseQuant = 6,
  kRSAddResidualRMSNormFP8BlockWiseQuant = 7,
  kRSAddResidualRMSNorm = 8,
};

enum class QuantType : int {
  kNone = 0,
  kFP8 = 1,
  kFP4 = 2,
  kFP8BlockWise = 3,
};

template <ReduceScatterFusionPattern Pattern>
struct ReduceScatterPatternTraits;

#define DEFINE_RS_PATTERN_TRAITS(pattern, hasReduceScatterOut, hasResidual, hasResidualOut, \
                                 hasRMSNorm, hasNormOut, hasAdd, quantType)                 \
  template <>                                                                               \
  struct ReduceScatterPatternTraits<pattern> {                                              \
    static constexpr bool kHasReduceScatterOut = hasReduceScatterOut;                       \
    static constexpr bool kHasResidual = hasResidual;                                       \
    static constexpr bool kHasResidualOut = hasResidualOut;                                 \
    static constexpr bool kHasRMSNorm = hasRMSNorm;                                         \
    static constexpr bool kHasNormOut = hasNormOut;                                         \
    static constexpr QuantType kQuantType = quantType;                                      \
    static constexpr bool kHasAdd = hasAdd;                                            \
  };

DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kReduceScatter, true, false, false, false,
                         false, false, QuantType::kNone);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSResidualRMSNorm, false, true, true, true,
                         true, false, QuantType::kNone);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSResidualRMSNormFP8Quant, false, true, true,
                         true, true, false, QuantType::kFP8);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSResidualRMSNormFP4Quant, false, true, true,
                         true, true, false, QuantType::kFP4);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSResidualRMSNormOutFP8Quant, false, true,
                         true, true, true, false, QuantType::kFP8);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSResidualRMSNormOutFP4Quant, false, true,
                         true, true, true, false, QuantType::kFP4);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSResidualRMSNormFP8BlockWiseQuant, false, true,
                         true, true, true, false, QuantType::kFP8BlockWise);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSAddResidualRMSNormFP8BlockWiseQuant, false, true,
                         true, true, true, true, QuantType::kFP8BlockWise);
DEFINE_RS_PATTERN_TRAITS(ReduceScatterFusionPattern::kRSAddResidualRMSNorm, false, true,
                         true, true, true, true, QuantType::kNone);
#undef DEFINE_RS_PATTERN_TRAITS

template <ReduceScatterFusionPattern Pattern>
constexpr bool HasResidual = ReduceScatterPatternTraits<Pattern>::kHasResidual;
template <ReduceScatterFusionPattern Pattern>
constexpr bool HasRMSNorm = ReduceScatterPatternTraits<Pattern>::kHasRMSNorm;
template <ReduceScatterFusionPattern Pattern>
constexpr bool HasReduceScatterOut = ReduceScatterPatternTraits<Pattern>::kHasReduceScatterOut;
template <ReduceScatterFusionPattern Pattern>
constexpr bool HasResidualOut = ReduceScatterPatternTraits<Pattern>::kHasResidualOut;
template <ReduceScatterFusionPattern Pattern>
constexpr bool HasNormOut = ReduceScatterPatternTraits<Pattern>::kHasNormOut;
template <ReduceScatterFusionPattern Pattern>
constexpr QuantType GetQuantType = ReduceScatterPatternTraits<Pattern>::kQuantType;
template <ReduceScatterFusionPattern Pattern>
constexpr bool HasAdd = ReduceScatterPatternTraits<Pattern>::kHasAdd;

template <typename T>
struct ReduceScatterFusionParams {
  int nranks;
  int rank;
  int size;
  int hidden_dim;
  int scale_stride;
  int num_token_current_rank;
  void** workspace;
  void* reducescatter_in;
  void* reducescatter_out;
  void* residual_in;
  void* residual_out;
  void* add_in;
  void* norm_out;
  void* quant_out;
  void* scale_out;
  void* rms_gamma;
  float rms_eps;
  float* scale_factor;
  bool use_oneshot;
  QuantizationSFLayout layout = QuantizationSFLayout::SWIZZLED_128x4;
  cudaStream_t stream;
  ReduceScatterFusionPattern pattern;
  bool trigger_completion_at_end = true;
};

template <int NRanks>
struct RSyncComm {
  __device__ __forceinline__ RSyncComm(void** workspace) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[1];
    flag_value = *flag_ptr;
    for (int r = 0; r < NRanks; ++r) {
      comm_bufs[r] = workspace[r];
      barrier_flags[r] = workspace[NRanks + r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_flag_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = new_flag_value;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  void* comm_bufs[NRanks];
  void* barrier_flags[NRanks];
  int flag_value;
};

template <int NRanks>
struct RLamportComm {
  __device__ __forceinline__ RLamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
    flag_value = *flag_ptr;
    int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     static_cast<int64_t>(data_offset) * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

template <int NRanks>
class RBarrier {
 public:
  __device__ __forceinline__ RBarrier(int rank, RSyncComm<NRanks> const& comm) {
    if (threadIdx.x < NRanks) {
      m_flag_value = comm.flag_value;
      int current_rank = rank;
      int target_rank = threadIdx.x;
      m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + current_rank;
      m_current_flag = reinterpret_cast<int*>(comm.barrier_flags[current_rank]) +
                       blockIdx.x * NRanks + target_rank;
    }
  }

  __device__ __forceinline__ void sync() {
    __syncthreads();
    if (threadIdx.x < NRanks) {
      m_flag_value = next_flag(m_flag_value);
      for (int flag_idx = blockIdx.x; flag_idx < details::kBarrierFlagCount;
           flag_idx += gridDim.x) {
        st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
      }
      while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
      }
    }
    __syncthreads();
  }

 protected:
  __device__ __forceinline__ void st_flag(int* addr, int flag) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
  }

  __device__ __forceinline__ int ld_flag(int* addr) {
    int flag;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr));
    return flag;
  }

  __device__ __forceinline__ int next_flag(int flag) { return flag == 2 ? 0 : flag + 1; }

  __device__ __forceinline__ int prev_flag(int flag) { return flag == 0 ? 2 : flag - 1; }

 public:
  int m_flag_value;

 private:
  int* m_target_flag;
  int* m_current_flag;
};

// Forward declaration of vec_add
template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> vec_add(const vec_t<T, VEC_SIZE>& a,
                                                      const vec_t<T, VEC_SIZE>& b) {
  vec_t<T, VEC_SIZE> ret;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    ret[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
  }
  return ret;
}

namespace maths {
// // ============================== Cast ==============================
template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float2 cuda_cast<float2, int2>(int2 val) {
  return make_float2(val.x, val.y);
}

template <>
__device__ inline float2 cuda_cast<float2, float>(float val) {
  return make_float2(val, val);
}

template <>
__device__ inline float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float2>(float2 val) {
  return __float22half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float>(float val) {
  return __float2half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, half>(half val) {
  return __half2half2(val);
}

// // ============================== Abs ==============================
template <typename T>
__device__ inline T cuda_abs(T val) {
  assert(false);
  return {};
}

template <>
__device__ inline float cuda_abs(float val) {
  return fabs(val);
}

template <>
__device__ inline float2 cuda_abs(float2 val) {
  return make_float2(fabs(val.x), fabs(val.y));
}

template <>
__device__ inline half cuda_abs(half val) {
  return __habs(val);
}

template <>
__device__ inline half2 cuda_abs(half2 val) {
  return __habs2(val);
}

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val) {
  return __habs(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val) {
  return __habs2(val);
}
#endif

// // ============================== Max ==============================
template <typename To, typename Ti>
__device__ inline To cuda_max(Ti val) {
  return cuda_cast<To>(val);
};

template <>
__device__ inline float cuda_max(float2 val) {
  return fmaxf(val.x, val.y);
}

template <>
__device__ inline half cuda_max(half2 val) {
  return __hmax(val.x, val.y);
}

template <>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 val) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  return __hmax(val.x, val.y);
#else
  assert(0);
  asm volatile("brkpt;\n" ::);
  return __nv_bfloat16(0);
#endif
}

// Binary maximum: compute the max of two values.
template <typename T>
__device__ inline T cuda_max(T val1, T val2) {
  return (val1 > val2) ? val1 : val2;
}

template <>
__device__ inline float2 cuda_max(float2 val1, float2 val2) {
  float2 out;
  out.x = fmaxf(val1.x, val2.x);
  out.y = fmaxf(val1.y, val2.y);
  return out;
}

template <>
__device__ inline half2 cuda_max(half2 val1, half2 val2) {
  return __hmax2(val1, val2);
}

template <>
__device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 val1, __nv_bfloat162 val2) {
  return __hmax2(val1, val2);
}

// // ============================== Reciprocal ==============================
// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}
}  // namespace maths

namespace utils {

#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

inline int getSMVersion() {
  int device{-1};
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline int getSMRegisters() {
  int device{-1};
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int regs_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&regs_per_block, cudaDevAttrMaxRegistersPerBlock, device));
  return regs_per_block;
}

inline __device__ int64_t get_sf_out_offset_128x4(std::optional<int> batchIdx, int mIdx, int kIdx,
                                                  std::optional<int> numRows, int numCols) {
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]
  constexpr int kMTileSize = 128;
  constexpr int kKTileSize = 64;
  constexpr int kOuterMTileSize = 32;
  constexpr int kInnerMTileSize = 4;
  constexpr int kInnerKTileSize = 4;

  int32_t mTileIdx = mIdx / kMTileSize;
  int32_t kTileIdx = kIdx / kKTileSize;

  int32_t outerMIdx = (mIdx % kMTileSize) / kInnerMTileSize;
  int32_t innerMIdx = mIdx % kInnerMTileSize;
  int32_t innerKIdx = kIdx % kInnerKTileSize;

  int32_t numMTiles = (numRows.value_or(0) + kMTileSize - 1) / kMTileSize;
  int32_t numKTiles = (numCols + kKTileSize - 1) / kKTileSize;

  int64_t kTileStride = kOuterMTileSize * kInnerMTileSize * kInnerKTileSize;
  int64_t mTileStride = numKTiles * kTileStride;
  int64_t bTileStride = numMTiles * mTileStride;

  int64_t outerMStride = kInnerMTileSize * kInnerKTileSize;
  int64_t innerMStride = kInnerKTileSize;
  int64_t innerKStride = 1;

  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                     kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride +
                     innerKIdx * innerKStride;

  return SFOffset;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx,
                                                       int colIdx, std::optional<int> numRows,
                                                       int numCols, SFType* SFout,
                                                       QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    if (layout == QuantizationSFLayout::SWIZZLED_128x4) {
      int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;

      auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numCols);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == QuantizationSFLayout::LINEAR) {
      int32_t KTileIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;

      int32_t numKTiles = numCols / details::CVT_FP4_SF_VEC_SIZE;
      int64_t mTileStride = numKTiles;

      int64_t BTileStride = numRows.value_or(0) * mTileStride;

      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else {
      return nullptr;
    }
  }
#endif
  return nullptr;
}

__forceinline__ __device__ uint32_t pack_bytes(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
  uint32_t val0 = c0;
  uint32_t val1 = c1;
  uint32_t val2 = c2;
  uint32_t val3 = c3;
  return (val3 << 24) | (val2 << 16) | (val1 << 8) | val0;
}

inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]),
        "f"(array[6]), "f"(array[7]));
  return val;
#else
  uint32_t val;
  __nv_fp4x2_storage_t vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    vals[i] = __nv_cvt_float2_to_fp4x2(*(((float2*)array) + i), __NV_E2M1, cudaRoundNearest);
  }
  val = pack_bytes(vals[0], vals[1], vals[2], vals[3]);
  return val;
#endif
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x),
        "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  uint32_t val;
  __nv_fp4x2_storage_t vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    vals[i] = __nv_cvt_float2_to_fp4x2(array[i], __NV_E2M1, cudaRoundNearest);
  }
  val = pack_bytes(vals[0], vals[1], vals[2], vals[3]);
  return val;
#endif
}

template <typename T, int VEC_SIZE>
__device__ uint32_t cvt_warp_fp16_to_fp4(vec_t<T, VEC_SIZE> val, float scale_factor,
                                        uint8_t* sf_out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = maths::cuda_abs(get_vec2_element(val, 0));

#pragma unroll
  for (int i = 1; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = maths::cuda_max(localMax, maths::cuda_abs(get_vec2_element(val, i)));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = maths::cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  float vecMax = float(maths::cuda_max(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  float SFValue = scale_factor * (vecMax * maths::reciprocal_approximate_ftz(6.0f));
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 12080)
  __nv_fp8_e8m0 tmp;
  tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmp);
  fp8SFVal = tmp.__x;
#else
#error "FP8 E8M0 support requires CUDA 12.8 or newer."
#endif
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
  float outputScale = SFValue != 0 ? maths::reciprocal_approximate_ftz(
                                         SFValue * maths::reciprocal_approximate_ftz(scale_factor))
                                   : 0.0f;

  if (sf_out) {
    // Write the SF to global memory (STG.8).
    *sf_out = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[details::CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<T, half>) {
      fp2Vals[i] = __half22float2(get_vec2_element(val, i));
    } else {
      fp2Vals[i] = __bfloat1622float2(get_vec2_element(val, i));
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
#else
  return 0;
#endif
}

template <typename T>
struct neg_zero {
  static constexpr T value = -T(0);
};

template <>
struct neg_zero<half> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __half value = __half_raw{neg_zero_bits};
};

template <>
struct neg_zero<nv_bfloat16> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __nv_bfloat16 value = __nv_bfloat16_raw{neg_zero_bits};
};

template <>
struct neg_zero<float> {
  static constexpr unsigned int neg_zero_bits = 0x80000000U;
  static constexpr float value = -0.0f;
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T>
__device__ __forceinline__ bool is_negative_zero(T) {
  return false;
}

// float specialization
template <>
__device__ __forceinline__ bool is_negative_zero<float>(float x) {
  return (__float_as_int(x) == 0x80000000);
}

// double specialization
template <>
__device__ __forceinline__ bool is_negative_zero<double>(double x) {
  return (__double_as_longlong(x) == 0x8000000000000000ULL);
}

// __half specialization
template <>
__device__ __forceinline__ bool is_negative_zero<__half>(__half x) {
  return (__half_as_ushort(x) == 0x8000);
}

// __nv_bfloat16 specialization
template <>
__device__ __forceinline__ bool is_negative_zero<__nv_bfloat16>(__nv_bfloat16 x) {
  return (__bfloat16_as_ushort(x) == 0x8000);
}

template <typename T, size_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE; ++i) {
    if (is_negative_zero(vec[i])) {
      return true;
    }
  }
  return false;
}

template <typename T, size_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (size_t i = 0; i < VEC_SIZE; ++i) {
    vec[i] = (is_negative_zero(vec[i])) ? static_cast<T>(0.f) : vec[i];
  }
}

template <typename T>
__device__ __forceinline__ void set_neg_zero(T* addr) {
  vec_t<T, details::kBytesPerAccess / sizeof(T)> val;
  val.fill(neg_zero_v<T>);
  val.store_global_volatile(addr);
}

}  // namespace utils

template <ReduceScatterFusionPattern Pattern, typename T>
class ReduceScatterFusedOp {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  static constexpr float FP8_E4M3_MAX = 448.0f;

 public:
  __device__ __forceinline__ ReduceScatterFusedOp(ReduceScatterFusionParams<T> const& params,
                                                  int access_id, int access_id_in_token,
                                                  int token_count_this_rank = 1)
      : m_params(params), m_access_id(access_id), m_access_id_in_token(access_id_in_token),
        m_token_count_this_rank(token_count_this_rank) {
    if (token_count_this_rank > 0) {
      if constexpr (HasRMSNorm<Pattern>) {
        m_gamma_val.load(reinterpret_cast<T*>(params.rms_gamma) + m_access_id_in_token * VEC_SIZE);
      }
      if (params.add_in) {
        m_add_val.load(reinterpret_cast<T*>(params.add_in) + m_access_id * VEC_SIZE);
      }
      if constexpr (HasResidual<Pattern>) {
        m_residual_val.load(reinterpret_cast<T*>(params.residual_in) + m_access_id * VEC_SIZE);
      }
      if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
        m_scale_factor = 1.f / *(params.scale_factor);
      } else if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
        m_scale_factor = *(params.scale_factor);
      }
    }
  }

  __device__ __forceinline__ void update(int access_id) {
    if (m_access_id != access_id) {
      m_access_id = access_id;
      if constexpr (HasAdd<Pattern>) {
        m_add_val.load(reinterpret_cast<T*>(m_params.add_in) + m_access_id * VEC_SIZE);
      }
      if constexpr (HasResidual<Pattern>) {
        m_residual_val.load(reinterpret_cast<T*>(m_params.residual_in) + m_access_id * VEC_SIZE);
      }
    }
  }

  __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int token_id) {

    if constexpr (HasReduceScatterOut<Pattern>) {
      val.store(reinterpret_cast<T*>(m_params.reducescatter_out) + m_access_id * VEC_SIZE);
    }

    if constexpr (HasAdd<Pattern>) {
      val = vec_add<T, VEC_SIZE>(val, m_add_val);
    }

    if constexpr (HasResidual<Pattern>) {
      val = vec_add<T, VEC_SIZE>(val, m_residual_val);

      if constexpr (HasResidualOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.residual_out) + m_access_id * VEC_SIZE);
      }
    }

    if constexpr (HasRMSNorm<Pattern>) {
      val = rms_norm(val, m_gamma_val);

      if constexpr (HasNormOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.norm_out) + m_access_id * VEC_SIZE);
      }
    }

#if CUDA_VERSION >= 12080
    if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
      auto sf_out = utils::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2>(
          std::nullopt, token_id, m_access_id_in_token, std::nullopt, m_params.hidden_dim,
          reinterpret_cast<uint32_t*>(m_params.scale_out), m_params.layout);
      reinterpret_cast<uint32_t*>(m_params.quant_out)[m_access_id] =
          utils::cvt_warp_fp16_to_fp4<T, VEC_SIZE>(val, m_scale_factor, sf_out);
    } else
#endif
        if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
      using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
      PackedQuantizedType ret;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] = static_cast<__nv_fp8_e4m3>(
            static_cast<float>(reinterpret_cast<T*>(&val)[i]) * m_scale_factor);
      }
      // PackedQuantizedType ret 是一个包含8个__nv_fp8_e4m3的向量，m_params.quant_out也按照这个格式来存储
      // 将m_params.quant_out的m_access_id位置的8个__nv_fp8_e4m3向量替换为ret
      reinterpret_cast<PackedQuantizedType*>(m_params.quant_out)[m_access_id] = ret;
    } else if constexpr (GetQuantType<Pattern> == QuantType::kFP8BlockWise) {
      vec_t<__nv_fp8_e4m3, VEC_SIZE> quant_out = block_quant_fp8(
          val, reinterpret_cast<float*>(m_params.scale_out), m_params.scale_stride, token_id);
      quant_out.store(reinterpret_cast<__nv_fp8_e4m3*>(m_params.quant_out) +
                      m_access_id * VEC_SIZE);
    } else {
      static_assert(GetQuantType<Pattern> == QuantType::kNone, "Invalid quant type");
    }
  }

 protected:
 __device__ __forceinline__ vec_t<__nv_fp8_e4m3, VEC_SIZE> block_quant_fp8(
      vec_t<T, VEC_SIZE> normed_res, float* scale_out, int32_t scale_stride, int32_t token_id) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    int32_t access_id_in_token = m_access_id_in_token;

    float _absmax = 1e-10f;
    vec_t<__nv_fp8_e4m3, VEC_SIZE> quanted_out;
    auto tile_16 = cg::tiled_partition<16>(cg::this_thread_block());
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      float val = static_cast<float>(normed_res[i]);
      _absmax = std::max(_absmax, fabs(val));
    }
    _absmax = cg::reduce(tile_16, _absmax, [](float a, float b) { return a > b ? a : b; });

    float scale;
    asm("div.full.f32 %0, %1, %2;" : "=f"(scale) : "f"(_absmax), "f"(FP8_E4M3_MAX));

    // directly write scale to scale_out
    if (tile_16.thread_rank() == 0) {
      int32_t col_idx = int32_t(access_id_in_token / 16);
      *(scale_out + col_idx * scale_stride + token_id) = scale;
    }
    tile_16.sync();

    float reverse_scale = 1.0 / scale;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
      float x = static_cast<float>(normed_res[i]) * reverse_scale;
      float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
      reinterpret_cast<__nv_fp8_e4m3*>(&quanted_out)[i] = static_cast<__nv_fp8_e4m3>(r);
    }
    return quanted_out;
  }
  __device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const& residual,
                                                         vec_t<T, VEC_SIZE> const& gamma) {
    __shared__ float s_val;
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = static_cast<float>(reinterpret_cast<T const*>(&residual)[i]);
      acc += v * v;
    }
    utils::blockReduceSumV2<float, 1>(&acc);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    if (cluster.num_blocks() > 1) {
      if (threadIdx.x == 0) {
        s_val = acc;
        acc = 0.f;
      }
      cluster.sync();
      if (threadIdx.x == 0) {
        for (int i = 0; i < cluster.num_blocks(); ++i) {
          acc += *cluster.map_shared_rank(&s_val, i);
        }
      }
      cluster.sync();
    }
#endif
    if (threadIdx.x == 0) {
      s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      reinterpret_cast<T*>(&norm_out)[i] =
          static_cast<T>(static_cast<float>(reinterpret_cast<T const*>(&residual)[i]) * s_val *
                         static_cast<float>(reinterpret_cast<T const*>(&gamma)[i]));
    }
    return norm_out;
  }

 private:
  ReduceScatterFusionParams<T> const& m_params;
  int m_access_id;
  int m_access_id_in_token;
  int m_token_count_this_rank;
  float m_scale_factor;
  vec_t<T, VEC_SIZE> m_residual_val;
  vec_t<T, VEC_SIZE> m_add_val;
  vec_t<T, VEC_SIZE> m_gamma_val;
};

template <typename T>
class RIndexHelper {
 public:
  __device__ __forceinline__ RIndexHelper(ReduceScatterFusionParams<T> const& params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    access_stride = token_stride * params.hidden_dim / VEC_SIZE;
    tot_access = params.size / VEC_SIZE;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

template <typename T, uint32_t VEC_SIZE, int NRanks, bool Fp32Acc>
__device__ __forceinline__ vec_t<T, VEC_SIZE> reducescatter_sum(vec_t<T, VEC_SIZE>* vals) {
  if constexpr (Fp32Acc) {
    static_assert(!std::is_same_v<T, float>);
    float acc_f32[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc_f32[i] = static_cast<float>(reinterpret_cast<T*>(&vals[0])[i]);
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        acc_f32[i] += static_cast<float>(reinterpret_cast<T*>(&vals[r])[i]);
      }
    }
    vec_t<T, VEC_SIZE> acc;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] = static_cast<T>(acc_f32[i]);
    }
    return acc;
  } else {
    vec_t<T, VEC_SIZE> acc = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      acc = vec_add<T, VEC_SIZE>(acc, vals[r]);
    }
    return acc;
  }
}

template <ReduceScatterFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
__global__ void reducescatter_fusion_kernel_oneshot_lamport(ReduceScatterFusionParams<T> params) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  RIndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id; // token id (线程级别)
  int access_id_in_token = index_helper.access_id_in_token; // 以VEC_SIZE为单位,在当前token中的access id(threadIdx.x)
  int token_stride = index_helper.token_stride; // token_id增加步长
  int access_id = index_helper.access_id; // 全局access id = token_id * hidden_dim / VEC_SIZE + access_id_in_token;
  int access_stride = index_helper.access_stride; // token_id增加时，对应access_stride增加步长 = token_stride * hidden_dim / VEC_SIZE
  int tot_access = index_helper.tot_access;
  vec_t<T, VEC_SIZE> clear_vec;
  clear_vec.fill(utils::neg_zero_v<T>);
  int num_tokens = params.size / params.hidden_dim;
  int tokens_per_rank = num_tokens / NRanks;
  int remaining_tokens = num_tokens % NRanks;
  int token_count_this_rank = tokens_per_rank + (params.rank < remaining_tokens ? 1 : 0);
  // int token_count_this_rank = params.num_token_current_rank;
  int start_token_idx = params.rank * tokens_per_rank + (params.rank < remaining_tokens ? params.rank : remaining_tokens);
  int access_per_token = params.hidden_dim / VEC_SIZE;


#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  RLamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / VEC_SIZE;

  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> val;
    val.load(reinterpret_cast<T*>(params.reducescatter_in) + idx * VEC_SIZE);
    utils::remove_neg_zero(val);
    // 计算数据要存放在哪一个target_rank
    int current_token_idx = idx / access_per_token;
    int threshold = remaining_tokens * (tokens_per_rank + 1);
    int target_rank;

    if (remaining_tokens == 0) {
      target_rank = current_token_idx / tokens_per_rank;
    } else {
      if (current_token_idx < threshold) {
        target_rank = current_token_idx / (tokens_per_rank + 1);
      } else {
        target_rank = remaining_tokens + (current_token_idx - threshold) / tokens_per_rank;
      }
    }
    val.store(reinterpret_cast<T*>(comm.data_bufs[target_rank]) +
              (params.rank * tot_access + idx) * VEC_SIZE);
  }

    for (int idx = access_id; idx < clear_access; idx += access_stride) {
      clear_vec.store(reinterpret_cast<T*>(comm.clear_buf) + idx * VEC_SIZE);
    }

    int start_idx = start_token_idx * access_per_token;
    int end_idx = start_idx + token_count_this_rank * access_per_token;

    // idx:输入数据的全局地址(从所有 rank 的view获取); out_idx:输出地址
    for (int idx = access_id + start_idx, out_idx = access_id; idx < end_idx;
        idx += access_stride, out_idx += access_stride) {
      ReduceScatterFusedOp<Pattern, T> fused_op(params, out_idx, access_id_in_token, token_count_this_rank);
      fused_op.update(out_idx);
      vec_t<T, VEC_SIZE> vals[NRanks];
      bool done = false;

      while (!done) {
        done = true;
  #pragma unroll
        for (int r = 0; r < NRanks; ++r) {
          vals[r].load_global_volatile(reinterpret_cast<T*>(comm.data_bufs[params.rank]) +
                                      (r * tot_access + idx) * VEC_SIZE);
          done &= !utils::has_neg_zero(vals[r]);
        }
      }

      vec_t<T, VEC_SIZE> sum_val = reducescatter_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);
      int token_id_local = out_idx / (params.hidden_dim / VEC_SIZE);
      fused_op(sum_val, token_id_local);
    }

    comm.update(params.size * NRanks);

  #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (TriggerCompletionAtEnd) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  #endif
}

template <ReduceScatterFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
__global__ void reducescatter_fusion_kernel_twoshot_sync(
    ReduceScatterFusionParams<T> params, std::array<int, NRanks> begin_tokens,
    std::array<int, NRanks> token_num_per_ranks) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  RIndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;

  int num_tokens = params.size / params.hidden_dim;
  int tokens_per_rank = num_tokens / NRanks;
  int remaining_tokens = num_tokens % NRanks;
  int token_count_this_rank = tokens_per_rank + (params.rank < remaining_tokens ? 1 : 0);
  ReduceScatterFusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token, token_count_this_rank);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  RSyncComm<NRanks> comm(params.workspace);

  // Phase 1: Gather data from all ranks
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
      reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx] =
          reinterpret_cast<float4*>(params.reducescatter_in)[idx];
    }
  }

  RBarrier<NRanks> barrier(params.rank, comm);
  barrier.sync();

  // Phase 2: Reduce for this rank's partition
  int partition_size = tot_access / NRanks;
  int start_idx = params.rank * partition_size;
  int end_idx = (params.rank == NRanks - 1) ? tot_access : (params.rank + 1) * partition_size;

  for (int idx = access_id + start_idx; idx < end_idx; idx += access_stride) {
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[r].load(reinterpret_cast<T*>(comm.comm_bufs[r]) + idx * VEC_SIZE);
    }
    vec_t<T, VEC_SIZE> sum_val = reducescatter_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_val.store(reinterpret_cast<T*>(comm.comm_bufs[r]) + (tot_access + idx) * VEC_SIZE);
    }
  }

  barrier.sync();

  // Phase 3: Write output
  for (int idx = access_id + start_idx, out_idx = access_id; idx < end_idx;
       idx += access_stride, out_idx += access_stride) {
    fused_op.update(out_idx);
    vec_t<T, VEC_SIZE> sum_val;
    sum_val.load(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                 (tot_access + idx) * VEC_SIZE);
    // out_idx is already relative to this rank's output, so token_id_local should be in [0, token_num/world_size)
    int token_id_local = out_idx / (params.hidden_dim / VEC_SIZE);
    fused_op(sum_val, token_id_local);
  }

  comm.update(barrier.m_flag_value);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

inline int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&device_id));
    FLASHINFER_CUDA_CALL(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  }
  return sm_count;
}

template <ReduceScatterFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
cudaError_t launch_oneshot_reducescatter(ReduceScatterFusionParams<T> const& params,
                                         cudaLaunchConfig_t& cfg) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg,
      reducescatter_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, Fp32Acc,
                                                  TriggerCompletionAtEnd>,
      params));

  return cudaSuccess;
}

template <ReduceScatterFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
int get_registers_per_thread_oneshot_rs() {
  auto kernel = reducescatter_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, Fp32Acc,
                                                            TriggerCompletionAtEnd>;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  return attr.numRegs;
}

template <ReduceScatterFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t launch_twoshot_reducescatter(ReduceScatterFusionParams<T> const& params,
                                         cudaLaunchConfig_t& cfg,
                                         std::array<int, NRanks> begin_tokens,
                                         std::array<int, NRanks> token_num_per_ranks) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg, reducescatter_fusion_kernel_twoshot_sync<Pattern, T, NRanks, Fp32Acc>, params,
      begin_tokens, token_num_per_ranks));
  return cudaSuccess;
}

template <ReduceScatterFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
int get_registers_per_thread_twoshot_rs() {
  auto kernel = reducescatter_fusion_kernel_twoshot_sync<Pattern, T, NRanks, Fp32Acc>;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  return attr.numRegs;
}

inline bool use_oneshot_rs(int token_num) { return token_num <= details::kOneShotMaxToken; }

template <ReduceScatterFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t reducescatter_fusion_kernel_launcher(ReduceScatterFusionParams<T> const& params,
                                                 bool launch_with_pdl) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  FLASHINFER_CHECK(params.size % params.hidden_dim == 0, "params.size % params.hidden_dim != 0");
  FLASHINFER_CHECK(params.hidden_dim % VEC_SIZE == 0, "params.hidden_dim % VEC_SIZE != 0");

  static int SM = utils::getSMVersion();
  int token_num = params.size / params.hidden_dim;
  bool oneshot = params.use_oneshot;
  int cluster_num = token_num;
  std::array<int, NRanks> begin_tokens, token_num_per_ranks;

  if (!oneshot) {
    int remaining_token = token_num % NRanks;
    int token_num_per_rank = token_num / NRanks;
    cluster_num = token_num_per_rank;
    if (remaining_token) {
      cluster_num++;
    }
    for (int r = 0; r < NRanks; ++r) {
      begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
      token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
    }
  }

  int threads_per_token = params.hidden_dim / VEC_SIZE;
  int cluster_size;
  if (SM >= 90) {
    cluster_size = 8;
  } else {
    cluster_size = 1;
  }

  while (threads_per_token % cluster_size != 0 && cluster_size > 1) {
    cluster_size /= 2;
  }

  int threads_per_block = threads_per_token / cluster_size;
  while (threads_per_block < 128 && cluster_size >= 2) {
    threads_per_block *= 2;
    cluster_size /= 2;
  }

  int sm_count = get_sm_count();
  int registers_per_thread;

  if (oneshot) {
    if (params.trigger_completion_at_end) {
      registers_per_thread =
          get_registers_per_thread_oneshot_rs<Pattern, T, NRanks, Fp32Acc, true>();
    } else {
      registers_per_thread =
          get_registers_per_thread_oneshot_rs<Pattern, T, NRanks, Fp32Acc, false>();
    }
  } else {
    registers_per_thread = get_registers_per_thread_twoshot_rs<Pattern, T, NRanks, Fp32Acc>();
  }

  static int max_registers = -1;
  if (max_registers < 0) {
    max_registers = utils::getSMRegisters();
  }

  int max_threads_per_block = min(max_registers / registers_per_thread, 1024);
  while (cluster_num * cluster_size > sm_count && cluster_size > 1 &&
         threads_per_block <= max_threads_per_block / 2) {
    threads_per_block *= 2;
    cluster_size /= 2;
  }

  FLASHINFER_CHECK(oneshot || threads_per_block >= params.nranks,
                   "not oneshot, or threads_per_block < nranks");

  int block_size = threads_per_block;
  FLASHINFER_CHECK(block_size <= 1024 && cluster_size > 0,
                   "block_size > 1024 or cluster_size <= 0");

  int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;

  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attribute[2];
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  if (oneshot) {
    bool trigger_completion_at_end = params.trigger_completion_at_end;
    if (trigger_completion_at_end) {
      FLASHINFER_CUDA_CALL(
          (launch_oneshot_reducescatter<Pattern, T, NRanks, Fp32Acc, true>(params, cfg)));
    } else {
      FLASHINFER_CUDA_CALL(
          (launch_oneshot_reducescatter<Pattern, T, NRanks, Fp32Acc, false>(params, cfg)));
    }
  } else {
    FLASHINFER_CUDA_CALL((launch_twoshot_reducescatter<Pattern, T, NRanks, Fp32Acc>(
        params, cfg, begin_tokens, token_num_per_ranks)));
  }

  return cudaSuccess;
}

template <typename T>
cudaError_t reducescatter_fusion_op(ReduceScatterFusionParams<T> const& params,
                                    bool launch_with_pdl, bool fp32_acc) {
#define DISPATCH_ACC_TYPE(T, Pattern, NRanks)                                                      \
  if constexpr (std::is_same_v<T, float>) {                                                        \
    return reducescatter_fusion_kernel_launcher<Pattern, T, NRanks, false>(params, launch_with_pdl); \
  } else {                                                                                         \
    if (fp32_acc) {                                                                                \
      return reducescatter_fusion_kernel_launcher<Pattern, T, NRanks, true>(params, launch_with_pdl); \
    } else {                                                                                       \
      return reducescatter_fusion_kernel_launcher<Pattern, T, NRanks, false>(params, launch_with_pdl); \
    }                                                                                              \
  }

#define DISPATCH_PATTERN(T, NRanks)                                                          \
  switch (params.pattern) {                                                                  \
    case ReduceScatterFusionPattern::kReduceScatter:                                         \
      DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kReduceScatter, NRanks);              \
      break;                                                                                 \
    case ReduceScatterFusionPattern::kRSResidualRMSNorm:                                     \
      DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSResidualRMSNorm, NRanks);          \
      break;                                                                                 \
    case ReduceScatterFusionPattern::kRSResidualRMSNormFP8Quant:                             \
      DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSResidualRMSNormFP8Quant, NRanks);  \
      break;                                                                                 \
    case ReduceScatterFusionPattern::kRSResidualRMSNormFP8BlockWiseQuant:                    \
      DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSResidualRMSNormFP8BlockWiseQuant, NRanks); \
      break;                                                                                  \
    case ReduceScatterFusionPattern::kRSAddResidualRMSNormFP8BlockWiseQuant:                    \
      DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSAddResidualRMSNormFP8BlockWiseQuant, NRanks); \
      break;                                                                                 \
    case ReduceScatterFusionPattern::kRSAddResidualRMSNorm:                                  \
      DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSAddResidualRMSNorm, NRanks);       \
      break;                                                                                 \
    case ReduceScatterFusionPattern::kRSResidualRMSNormFP4Quant:                             \
      if constexpr (!std::is_same_v<T, float> && CUDA_VERSION >= 12080) {                    \
        DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSResidualRMSNormFP4Quant, NRanks); \
      } else {                                                                               \
        FLASHINFER_CHECK(CUDA_VERSION >= 12080, "FP4Quant requires CUDA 12.8 or higher");    \
        FLASHINFER_CHECK(false, "FP4Quant pattern cannot work with DType=float");            \
      }                                                                                      \
      break;                                                                                 \
    case ReduceScatterFusionPattern::kRSResidualRMSNormOutFP8Quant:                          \
      DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSResidualRMSNormOutFP8Quant, NRanks); \
      break;                                                                                 \
    case ReduceScatterFusionPattern::kRSResidualRMSNormOutFP4Quant:                          \
      if constexpr (!std::is_same_v<T, float> && CUDA_VERSION >= 12080) {                    \
        DISPATCH_ACC_TYPE(T, ReduceScatterFusionPattern::kRSResidualRMSNormOutFP4Quant, NRanks); \
      } else {                                                                               \
        FLASHINFER_CHECK(CUDA_VERSION >= 12080, "OutFP4Quant requires CUDA 12.8 or higher"); \
        FLASHINFER_CHECK(false, "OutFP4Quant pattern cannot work with DType=float");         \
      }                                                                                      \
      break;                                                                                 \
    default:                                                                                 \
      FLASHINFER_CHECK(false, "Unsupported reducescatter fusion pattern");                   \
  }

  switch (params.nranks) {
    case 2:
      DISPATCH_PATTERN(T, 2);
      break;
    case 4:
      DISPATCH_PATTERN(T, 4);
      break;
    case 8:
      DISPATCH_PATTERN(T, 8);
      break;
    case 16:
      DISPATCH_PATTERN(T, 16);
      break;
    default:
      FLASHINFER_ERROR(
          "reducescatter_fusion_kernel: unsupported ranks number! Supported ranks: 2, 4, 8, 16.");
  }
}

}  // namespace trtllm_reducescatter_fusion

}  // namespace flashinfer
