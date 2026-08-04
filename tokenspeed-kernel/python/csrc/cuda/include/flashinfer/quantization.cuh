/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_QUANTIZATION_CUH_
#define FLASHINFER_QUANTIZATION_CUH_
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cub/cub.cuh>

#include "utils.cuh"

namespace flashinfer {
namespace quantization {

enum class BitOrder { kBig = 0U, kLittle = 1U };

#define DISPATCH_BITORDER(bitorder, BITORDER, ...)   \
  if (bitorder == BitOrder::kBig) {                  \
    constexpr BitOrder BITORDER = BitOrder::kBig;    \
    __VA_ARGS__                                      \
  } else {                                           \
    constexpr BitOrder BITORDER = BitOrder::kLittle; \
    __VA_ARGS__                                      \
  }

template <BitOrder BITORDER>
__global__ void PackBitsKernel(bool* input, uint8_t* output, int64_t num_elements) {
  int64_t start_offset = static_cast<int64_t>(blockIdx.x) * blockDim.x * 8, tx = threadIdx.x;
  uint8_t ret = 0;
  bool input_vec[8];
  typedef cub::BlockLoad<bool, 256, 8, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
  __shared__ typename BlockLoad::TempStorage temp_storage;

  // This fix the INT32_T overflow issue, which is possible in DiT video models
  // where the kv_len could be 128K.
  // ref:
  // https://github.com/NVIDIA/cub/blob/0fc3c3701632a4be906765b73be20a9ad0da603d/cub/block/block_load.cuh#L711C13-L711C100
  int block_items_end =
      (num_elements - start_offset > INT32_MAX) ? INT32_MAX : num_elements - start_offset;
  BlockLoad(temp_storage).Load(input + start_offset, input_vec, block_items_end, /*default=*/0);

  if constexpr (BITORDER == BitOrder::kBig) {
    ret = (input_vec[0] << 7) | (input_vec[1] << 6) | (input_vec[2] << 5) | (input_vec[3] << 4) |
          (input_vec[4] << 3) | (input_vec[5] << 2) | (input_vec[6] << 1) | input_vec[7];
  } else {
    ret = (input_vec[7] << 7) | (input_vec[6] << 6) | (input_vec[5] << 5) | (input_vec[4] << 4) |
          (input_vec[3] << 3) | (input_vec[2] << 2) | (input_vec[1] << 1) | input_vec[0];
  }
  if (start_offset + tx * 8 < num_elements) output[start_offset / 8 + tx] = ret;
}

template <BitOrder BITORDER, typename IdType>
__global__ void SegmentPackBitsKernel(bool* input, uint8_t* output, IdType* input_indptr,
                                      IdType* output_indptr) {
  int64_t bx = blockIdx.x, tx = threadIdx.x;
  bool input_vec[8];
  typedef cub::BlockLoad<bool, 256, 8, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
  __shared__ typename BlockLoad::TempStorage temp_storage;
  int64_t num_elements = input_indptr[bx + 1] - input_indptr[bx];
  for (uint32_t start_offset = 0; start_offset < num_elements; start_offset += 8 * blockDim.x) {
    uint8_t ret = 0;
    BlockLoad(temp_storage)
        .Load(input + input_indptr[bx] + start_offset, input_vec, num_elements - start_offset,
              /*default=*/0);

    if constexpr (BITORDER == BitOrder::kBig) {
      ret = (input_vec[0] << 7) | (input_vec[1] << 6) | (input_vec[2] << 5) | (input_vec[3] << 4) |
            (input_vec[4] << 3) | (input_vec[5] << 2) | (input_vec[6] << 1) | input_vec[7];
    } else {
      ret = (input_vec[7] << 7) | (input_vec[6] << 6) | (input_vec[5] << 5) | (input_vec[4] << 4) |
            (input_vec[3] << 3) | (input_vec[2] << 2) | (input_vec[1] << 1) | input_vec[0];
    }
    if (start_offset + tx * 8 < num_elements)
      output[output_indptr[bx] + start_offset / 8 + tx] = ret;
  }
}

cudaError_t PackBits(bool* input, uint8_t* output, int64_t num_elements, BitOrder bitorder,
                     cudaStream_t stream) {
  DISPATCH_BITORDER(bitorder, BITORDER, {
    auto kernel = PackBitsKernel<BITORDER>;
    const dim3 nthrs(256);
    const dim3 nblks(ceil_div(num_elements, nthrs.x * 8));
    void* args[] = {&input, &output, &num_elements};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template <typename IdType>
cudaError_t SegmentPackBits(bool* input, uint8_t* output, IdType* input_indptr,
                            IdType* output_indptr, uint32_t batch_size, BitOrder bitorder,
                            cudaStream_t stream) {
  DISPATCH_BITORDER(bitorder, BITORDER, {
    auto kernel = SegmentPackBitsKernel<BITORDER, IdType>;
    const dim3 nthrs(256);
    const dim3 nblks(batch_size);
    void* args[] = {&input, &output, &input_indptr, &output_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

__device__ __forceinline__ void fast_divmod(uint32_t& div, uint32_t& mod, int x, int y,
                                            uint32_t mul, uint32_t shr) {
  if (y == 1) {
    div = x;
    mod = 0;
  } else {
    div = __umulhi((uint32_t)x, mul) >> shr;
    mod = x - div * y;
  }
}

template <typename T>
__device__ __host__ constexpr T div_up(T a, int b) {
  return (a + b - 1) / b;
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

template <>
__inline__ __device__ __nv_bfloat16 warpReduceSum(__nv_bfloat16 val) {
  constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = __hmax(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

__inline__ __device__ uint32_t elect_one_sync([[maybe_unused]] int lane_id) {
  uint32_t pred = 0;
#if __CUDA_ARCH__ >= 900
  uint32_t laneid = 0;
  asm volatile(
      "\n\
    {\n\
        .reg .b32 %rx;\n\
        .reg .pred %px;\n\
        elect.sync %rx|%px, %2;\n\
        @%px mov.s32 %1, 1;\n\
        mov.s32 %0, %rx;\n\
    }\n\
  "
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
#else
  return lane_id == 0;
#endif
  return pred;
}

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&device_id));
    FLASHINFER_CUDA_CALL(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  }
  return sm_count;
}

__host__ __device__ __forceinline__ int64_t compute_padded_offset(int64_t offset, int problem_idx) {
  constexpr int64_t alignment = 32;
  return (offset + problem_idx * (alignment - 1)) / alignment * alignment;
}

template <bool UseBinarySearch, typename InputType, typename OutputType>
__global__ void scale_1x128_kernel(OutputType* output, float* scales, InputType const* input,
                                   int32_t const* problem_m_offsets, int num_problems, int dim_x,
                                   int64_t scale_leading_dim, uint32_t scale_dim_x_mul,
                                   uint32_t scale_dim_x_shr) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  extern __shared__ char shared_memory[];
  int64_t* smem_problem_m_boundaries = reinterpret_cast<int64_t*>(shared_memory);

  // problem_m_offsets[0] is omitted because its value is known to be 0
  for (int i = threadIdx.x; i < num_problems; i += blockDim.x) {
    smem_problem_m_boundaries[i] = problem_m_offsets[i + 1];
    // printf("problem_m_offsets=%d i=%d \n", problem_m_offsets[i + 1], i);
  }
  __syncthreads();

  size_t scales_along_dim_x = div_up(dim_x, 128);
  // printf("scales_along_dim_x=%lld \n", scales_along_dim_x);
  size_t scales_along_dim_y = smem_problem_m_boundaries[num_problems - 1];
  // printf("scales_along_dim_y=%lld \n", scales_along_dim_y);
  size_t total_scales = scales_along_dim_x * scales_along_dim_y;

  int problem_idx = 0;
  int64_t padded_offset = 0;
  int64_t boundary_left, boundary_right;
  if constexpr (UseBinarySearch) {
    boundary_left = smem_problem_m_boundaries[0];
    boundary_right = scales_along_dim_y;
  } else {
    boundary_left = 0;
    boundary_right = smem_problem_m_boundaries[0];
  }

  for (size_t warp_idx = (threadIdx.x + blockIdx.x * blockDim.x) / 32; warp_idx < total_scales;
       warp_idx += (blockDim.x * gridDim.x) / 32) {
    uint32_t scales_idx_y;  // = warp_idx / scales_along_dim_x;
    uint32_t scales_idx_x;  // = warp_idx % scales_along_dim_x;
    fast_divmod(scales_idx_y, scales_idx_x, warp_idx, scales_along_dim_x, scale_dim_x_mul,
                scale_dim_x_shr);

    if constexpr (UseBinarySearch) {
      int idx_right = num_problems - 1;
      int64_t val_right = boundary_right;
      if (scales_idx_y >= boundary_left) {
        while (problem_idx + 1 < idx_right) {
          int idx_mid = (problem_idx + idx_right) >> 1;
          int64_t val_mid = smem_problem_m_boundaries[idx_mid];
          if (scales_idx_y < val_mid) {
            idx_right = idx_mid;
            val_right = val_mid;
          } else {
            problem_idx = idx_mid;
            boundary_left = val_mid;
          }
        }
        padded_offset = compute_padded_offset(boundary_left, problem_idx + 1) - boundary_left;
        boundary_left = val_right;
      }
    } else {
      if (boundary_right <= scales_idx_y) {
        while (problem_idx < num_problems - 1) {
          boundary_left = boundary_right;
          boundary_right = smem_problem_m_boundaries[++problem_idx];
          if (scales_idx_y < boundary_right) {
            break;
          }
        }
        padded_offset = compute_padded_offset(boundary_left, problem_idx) - boundary_left;
      }
    }

    auto warp_offset = (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
    InputType const* input_line = input + warp_offset;
    OutputType* output_line = output + warp_offset;
    auto& scale_output =
        scales[(size_t)scales_idx_x * scale_leading_dim + scales_idx_y + padded_offset];

    int lane_id = threadIdx.x % 32;
    InputType input_frag[4];

    for (int i = 0; i < 4; i++) {
      input_frag[i] =
          (scales_idx_x * 128 + i * 32 + lane_id < dim_x) ? input_line[lane_id] : InputType(0);
      input_line += 32;
    }

    InputType amax =
        warpReduceSum(max(max(fabs(float(input_frag[0])), fabs(float(input_frag[1]))),
                          max(fabs(float(input_frag[2])), fabs(float(input_frag[3])))));

    float scale = amax != InputType(0.f) ? 448.f / float(amax) : 1.f;

    if (elect_one_sync(lane_id)) {
      scale_output = float(1.f / scale);
    }

    for (int i = 0; i < 4; i++) {
      float value = float(input_frag[i]) * scale;
      if (scales_idx_x * 128 + i * 32 + lane_id < dim_x) {
        output_line[lane_id] = OutputType(value);
      }
      output_line += 32;
    }
  }
#endif
}

inline void find_divisor(uint32_t& mul, uint32_t& shr, int x) {
  auto find_log_2 = [](int x, bool round_up = false) {
    auto clz = [](int x) {
      for (int i = 31; i >= 0; --i) {
        if ((1 << i) & x) {
          return 31 - i;
        }
      }
      return 32;
    };

    int a = 31 - clz(x);
    if (round_up) {
      a += (x & (x - 1)) ? 1 : 0;
    }
    return a;
  };

  assert(x != 0);
  if (x == 1) {
    // If dividing by 1, reduced math doesn't work because mul_coeff would need
    // to be 2^32, which doesn't fit into unsigned int.  the div() routine
    // handles this special case separately.
    mul = 0;
    shr = 0;
    // printf("[FIND_DIVISOR] Case x=1: setting mul=0, shr=0\n");
  } else {
    // To express the division N/D in terms of a multiplication, what we first
    // imagine is simply N*(1/D).  However, 1/D will always evaluate to 0 (for
    // D>1), so we need another way.  There's nothing that says we have to use
    // exactly the fraction 1/D; instead it could be any X/Y that reduces to 1/D
    // (i.e., Y=X*D), or at least to "close enough" to it.  If we pick Y that is
    // a power of two, then the N*(X/Y) can be N*X followed by a right-shift by
    // some amount. The power of two we should pick should be at least 2^32,
    // because in the div() routine we'll use umulhi(), which returns only the
    // upper 32 bits -- this being equivalent to a right-shift by 32.  But we
    // might want a higher power of two for better accuracy depending on the
    // magnitude of the denominator. Once we've picked Y, then X [our mul_coeff
    // value] is simply Y/D, rounding up, and we save shift_coeff as whatever
    // further shift we have to do beyond what the umulhi() implies.
    uint32_t p = 31 + find_log_2(x, true);
    uint32_t m = (uint32_t)(((1ull << p) + (uint32_t)x - 1) / (uint32_t)x);

    mul = m;
    shr = p - 32;
    // printf("[FIND_DIVISOR] Case x>1: p=%u, m=%u, shr=%u\n",
    //   p, m, shr);
  }
}

cudaError_t quant_1x128(__nv_bfloat16 const* mat_a, __nv_fp8_e4m3* fp8_mat_a, float* scales_a,
                        int32_t const* problem_m_offsets, int num_problems, int64_t max_shape_m,
                        int64_t max_shape_m_padded, int shape_k, cudaStream_t stream) {
  constexpr int NumThreads = 256;
  int scales_dim_x = div_up(shape_k, 128);
  uint32_t scale_dim_x_mul, scale_dim_x_shr;
  find_divisor(scale_dim_x_mul, scale_dim_x_shr, scales_dim_x);
  // printf("SAFE_CHECK: mul=%u, shr=%u\n", scale_dim_x_mul, scale_dim_x_shr);
  int num_sms = get_sm_count();
  int smem_size = num_problems * sizeof(int64_t);
  int num_blocks =
      std::min(static_cast<int64_t>(num_sms), div_up(max_shape_m * scales_dim_x, NumThreads / 32));
  // Binary search is expected to have lower complexity when max_shape_m is small
  bool use_binary_search =
      static_cast<double>(max_shape_m) * scales_dim_x /
          static_cast<double>(NumThreads * num_blocks / 32) <=
      static_cast<double>(num_problems) / std::log2(static_cast<double>(num_problems));
  auto kernel = use_binary_search ? scale_1x128_kernel<true, __nv_bfloat16, __nv_fp8_e4m3>
                                  : scale_1x128_kernel<false, __nv_bfloat16, __nv_fp8_e4m3>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel<<<num_blocks, NumThreads, smem_size, stream>>>(
      fp8_mat_a, scales_a, mat_a, problem_m_offsets, num_problems, shape_k, max_shape_m_padded,
      scale_dim_x_mul, scale_dim_x_shr);
  return cudaSuccess;
}

}  // namespace quantization
}  // namespace flashinfer

#endif  // FLASHINFER_QUANTIZATION_CUH_
