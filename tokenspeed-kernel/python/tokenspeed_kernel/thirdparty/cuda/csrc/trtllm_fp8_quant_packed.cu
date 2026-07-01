/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 *   cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/
 *   fp8_blockscale_quant_packed.cu
 *
 * TensorRT-LLM introduced the kernel in 096f86eca5 and completed padded-row
 * initialization in d2ba38fccd. TokenSpeed retains the kernel launch/layout,
 * constrains K to complete 128-element groups, exposes per-call PDL through a
 * TVM-FFI adapter, and keeps TokenSpeed's pre-existing post-division scale
 * floor so this performance change does not alter fallback numerics.
 */

#include <climits>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace {

__device__ __forceinline__ float reciprocal_approximate_ftz(float value) {
  float reciprocal;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n"
               : "=f"(reciprocal)
               : "f"(value));
  return reciprocal;
}

// Each warp consumes one row by four 128-element quantization groups. The
// four eight-lane subgroups reduce one amax each, then lane 0 packs their
// UE8M0 bytes into the final DeepGEMM int32 layout.
template <int WarpsPerBlock>
__global__ void fp8_quantize_1x128_packed_kernel(
    __nv_fp8_e4m3* __restrict__ fp8_output,
    int32_t* __restrict__ packed_scale_output,
    const __nv_bfloat16* __restrict__ input,
    int rows,
    int columns,
    int scale_leading_dim) {
  const int packed_scale_column = static_cast<int>(blockIdx.x);
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;
  const int lane_id = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.y) * WarpsPerBlock + warp_id;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const bool row_in_range = row < rows;
  uint32_t packed_scale = 0u;
  if (row_in_range) {
    const int column_base = packed_scale_column * 512 + lane_id * 16;

    constexpr int kLoadElements = sizeof(double4) / sizeof(__nv_bfloat16);
    union Bf16Load {
      double4 vector;
      __nv_bfloat16 values[kLoadElements];
    };

    Bf16Load loaded;
    const bool column_in_range = column_base < columns;
    if (column_in_range) {
      const auto* input_vector = reinterpret_cast<const double4*>(
          input + static_cast<int64_t>(row) * columns + column_base);
      loaded.vector = input_vector[0];
    } else {
      loaded.vector = double4{};
    }

    if (column_in_range && column_base + kLoadElements > columns) {
      const int valid = columns - column_base;
#pragma unroll
      for (int index = 0; index < kLoadElements; ++index) {
        if (index >= valid) {
          loaded.values[index] = __nv_bfloat16(0.0f);
        }
      }
    }

    __nv_bfloat16 max_value = __nv_bfloat16(0.0f);
#pragma unroll
    for (int index = 0; index < kLoadElements; ++index) {
      max_value = __hmax(max_value, __habs(loaded.values[index]));
    }
    float amax = static_cast<float>(max_value);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 8));
    const float raw_scale = fmaxf(
        amax * reciprocal_approximate_ftz(448.0f), 1.0e-10f);
    __nv_fp8_e8m0 ue8m0_scale;
    ue8m0_scale.__x = __nv_cvt_float_to_e8m0(
        raw_scale, __NV_SATFINITE, cudaRoundPosInf);

    constexpr uint32_t kFp32ExponentBias = 127u;
    const float quant_scale = ue8m0_scale.__x == 0
                                  ? 1.0f
                                  : exp2f(static_cast<float>(kFp32ExponentBias) -
                                          static_cast<float>(ue8m0_scale.__x));

    constexpr int kStoreElements = sizeof(float4) / sizeof(__nv_fp8_e4m3);
    union Fp8Store {
      float4 vector;
      __nv_fp8_e4m3 values[kStoreElements];
    };

    Fp8Store quantized;
    quantized.vector = float4{};
#pragma unroll
    for (int index = 0; index < kStoreElements; ++index) {
      quantized.values[index] = __nv_fp8_e4m3(
          static_cast<float>(loaded.values[index]) * quant_scale);
    }
    if (column_in_range) {
      if (column_base + kStoreElements > columns) {
        const int valid = columns - column_base;
#pragma unroll
        for (int index = 0; index < kStoreElements; ++index) {
          if (index >= valid) {
            quantized.values[index] = __nv_fp8_e4m3(0.0f);
          }
        }
      }
      auto* output_vector = reinterpret_cast<float4*>(
          fp8_output + static_cast<int64_t>(row) * columns + column_base);
      output_vector[0] = quantized.vector;
    }

    const uint32_t scale0 = __shfl_sync(
        0xffffffffu, static_cast<uint32_t>(ue8m0_scale.__x), 0);
    const uint32_t scale1 = __shfl_sync(
        0xffffffffu, static_cast<uint32_t>(ue8m0_scale.__x), 8);
    const uint32_t scale2 = __shfl_sync(
        0xffffffffu, static_cast<uint32_t>(ue8m0_scale.__x), 16);
    const uint32_t scale3 = __shfl_sync(
        0xffffffffu, static_cast<uint32_t>(ue8m0_scale.__x), 24);
    if (lane_id == 0) {
      const int scale_columns = (columns + 127) / 128;
      const int scale_column_base = packed_scale_column * 4;
      if (scale_column_base < scale_columns) {
        packed_scale |= scale0;
      }
      if (scale_column_base + 1 < scale_columns) {
        packed_scale |= scale1 << 8;
      }
      if (scale_column_base + 2 < scale_columns) {
        packed_scale |= scale2 << 16;
      }
      if (scale_column_base + 3 < scale_columns) {
        packed_scale |= scale3 << 24;
      }
    }
  }

  if (lane_id == 0 && row < scale_leading_dim) {
    packed_scale_output[
        static_cast<int64_t>(packed_scale_column) * scale_leading_dim + row] =
        static_cast<int32_t>(packed_scale);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

cudaError_t launch_fp8_quantize_1x128_packed(
    __nv_fp8_e4m3* fp8_output,
    int32_t* packed_scale_output,
    const __nv_bfloat16* input,
    int rows,
    int columns,
    int scale_leading_dim,
    bool enable_pdl,
    cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 4;
  const int packed_scale_columns = ((columns + 127) / 128 + 3) / 4;
  const int row_blocks =
      (scale_leading_dim + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const dim3 grid(packed_scale_columns, row_blocks, 1);
  const dim3 block(kWarpsPerBlock * 32, 1, 1);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attributes[1]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.attrs = attributes;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(
      &config,
      fp8_quantize_1x128_packed_kernel<kWarpsPerBlock>,
      fp8_output,
      packed_scale_output,
      input,
      rows,
      columns,
      scale_leading_dim);
}

}  // namespace

void trtllm_fp8_quantize_1x128_packed_ue8m0(
    TensorView input,
    TensorView fp8_output,
    TensorView packed_scale_output,
    bool enable_pdl) {
  CHECK_CUDA(input);
  CHECK_CUDA(fp8_output);
  CHECK_CUDA(packed_scale_output);
  CHECK_DEVICE(input, fp8_output);
  CHECK_DEVICE(input, packed_scale_output);
  CHECK_DIM(2, input);
  CHECK_DIM(2, fp8_output);
  CHECK_DIM(2, packed_scale_output);
  CHECK_INPUT_TYPE(input, dl_bfloat16);
  CHECK_INPUT_TYPE(fp8_output, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(packed_scale_output, dl_int32);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(fp8_output);
  CHECK_CONTIGUOUS(packed_scale_output);

  const int64_t rows64 = input.size(0);
  const int64_t columns64 = input.size(1);
  TVM_FFI_ICHECK_GT(rows64, 0) << "input must contain at least one row";
  TVM_FFI_ICHECK_LE(rows64, static_cast<int64_t>(INT32_MAX));
  TVM_FFI_ICHECK_GT(columns64, 0) << "input must contain at least one column";
  TVM_FFI_ICHECK_LE(columns64, static_cast<int64_t>(INT32_MAX));
  TVM_FFI_ICHECK_EQ(columns64 % 128, 0)
      << "input columns must be divisible by 128";
  TVM_FFI_ICHECK_EQ(fp8_output.size(0), rows64);
  TVM_FFI_ICHECK_EQ(fp8_output.size(1), columns64);

  const int64_t aligned_rows = (rows64 + 3) / 4 * 4;
  const int64_t packed_scale_columns = (columns64 / 128 + 3) / 4;
  TVM_FFI_ICHECK_EQ(packed_scale_output.size(0), packed_scale_columns);
  TVM_FFI_ICHECK_EQ(packed_scale_output.size(1), aligned_rows);
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % sizeof(double4), 0)
      << "input must be 32-byte aligned";
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(fp8_output.data_ptr()) % sizeof(float4), 0)
      << "fp8_output must be 16-byte aligned";

  const int device_id = input.device().device_id;
  cudaError_t status = cudaSetDevice(device_id);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "cudaSetDevice failed: " << cudaGetErrorString(status);

  const cudaStream_t stream = get_stream(input.device());
  status = launch_fp8_quantize_1x128_packed(
      static_cast<__nv_fp8_e4m3*>(fp8_output.data_ptr()),
      static_cast<int32_t*>(packed_scale_output.data_ptr()),
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<int>(rows64),
      static_cast<int>(columns64),
      static_cast<int>(aligned_rows),
      enable_pdl,
      stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "TRT-LLM packed FP8 quantization launch failed: "
      << cudaGetErrorString(status);
  status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "TRT-LLM packed FP8 quantization launch state failed: "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    trtllm_fp8_quantize_1x128_packed_ue8m0,
    trtllm_fp8_quantize_1x128_packed_ue8m0);
