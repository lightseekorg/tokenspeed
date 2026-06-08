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
#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "../../exception.h"

// #ifndef _WIN32 // Linux
// #include <sys/sysinfo.h>
// #endif         // not WIN32
// #include <vector>
// #ifdef _WIN32  // Windows
// #include <windows.h>
// #undef ERROR   // A Windows header file defines ERROR as 0, but it's used in our logger.h enum.
// Logging breaks without
//                // this undef.
// #endif         // WIN32

namespace tensorrt_llm::common {

// // workspace for cublas gemm : 32MB
// #define CUBLAS_WORKSPACE_SIZE 33554432

// typedef struct __align__(4)
// {
//     half x, y, z, w;
// }

// half4;

// /* **************************** type definition ***************************** */

// enum CublasDataType
// {
//     FLOAT_DATATYPE = 0,
//     HALF_DATATYPE = 1,
//     BFLOAT16_DATATYPE = 2,
//     INT8_DATATYPE = 3,
//     FP8_DATATYPE = 4
// };

// enum TRTLLMCudaDataType
// {
//     FP32 = 0,
//     FP16 = 1,
//     BF16 = 2,
//     INT8 = 3,
//     FP8 = 4
// };

// enum class OperationType
// {
//     FP32,
//     FP16,
//     BF16,
//     INT8,
//     FP8
// };

/* **************************** debug tools ********************************* */

inline std::optional<bool> isCudaLaunchBlocking() {
  thread_local bool firstCall = true;
  thread_local std::optional<bool> result = std::nullopt;
  if (firstCall) {
    char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (env != nullptr && std::string(env) == "1") {
      result = true;
    } else {
      result = false;
    }
    firstCall = false;
  }
  return result;
}

inline std::optional<bool> isCapturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  FLASHINFER_CHECK(cudaStreamIsCapturing(stream, &status) == cudaSuccess,
                   "CUDA error in cudaStreamIsCapturing");
  return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
}

inline bool doCheckError(cudaStream_t stream) {
  auto const cudaLaunchBlocking = isCudaLaunchBlocking();
  if (cudaLaunchBlocking.has_value() && cudaLaunchBlocking.value()) {
    return !isCapturing(stream);
  }

#ifndef NDEBUG
  // Debug builds will sync when we're not capturing unless explicitly
  // disabled.
  bool const checkError = cudaLaunchBlocking.value_or(!isCapturing(stream));
#else
  bool const checkError = cudaLaunchBlocking.value_or(false);
#endif

  return checkError;
}

inline void syncAndCheck(cudaStream_t stream, char const* const file, int const line) {
  if (doCheckError(stream)) {
    cudaStreamSynchronize(stream);
    auto error = cudaGetLastError();
    FLASHINFER_CHECK(error == cudaSuccess, "CUDA error in %s: %s", file, cudaGetErrorString(error));
  }
}

#define sync_check_cuda_error(stream) tensorrt_llm::common::syncAndCheck(stream, __FILE__, __LINE__)

template <typename T1, typename T2>
inline size_t divUp(T1 const& a, T2 const& b) {
  auto const tmp_a = static_cast<size_t>(a);
  auto const tmp_b = static_cast<size_t>(b);
  return (tmp_a + tmp_b - 1) / tmp_b;
}

inline int roundUp(int a, int b) { return divUp(a, b) * b; }

template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
          typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T>
struct num_elems;
template <>
struct num_elems<float> {
  static constexpr int value = 1;
};
template <>
struct num_elems<float2> {
  static constexpr int value = 2;
};
template <>
struct num_elems<float4> {
  static constexpr int value = 4;
};
template <>
struct num_elems<half> {
  static constexpr int value = 1;
};
template <>
struct num_elems<half2> {
  static constexpr int value = 2;
};
#ifdef ENABLE_BF16
template <>
struct num_elems<__nv_bfloat16> {
  static constexpr int value = 1;
};
template <>
struct num_elems<__nv_bfloat162> {
  static constexpr int value = 2;
};
#endif
#ifdef ENABLE_FP8
template <>
struct num_elems<__nv_fp8_e4m3> {
  static constexpr int value = 1;
};
template <>
struct num_elems<__nv_fp8x2_e4m3> {
  static constexpr int value = 2;
};
#endif

template <typename T, int num>
struct packed_as;
template <typename T>
struct packed_as<T, 1> {
  using type = T;
};
template <>
struct packed_as<half, 2> {
  using type = half2;
};
template <>
struct packed_as<float, 2> {
  using type = float2;
};
template <>
struct packed_as<int8_t, 2> {
  using type = int16_t;
};
template <>
struct packed_as<int32_t, 2> {
  using type = int2;
};
template <>
struct packed_as<half2, 1> {
  using type = half;
};
template <>
struct packed_as<float2, 1> {
  using type = float;
};
#ifdef ENABLE_BF16
template <>
struct packed_as<__nv_bfloat16, 2> {
  using type = __nv_bfloat162;
};
template <>
struct packed_as<__nv_bfloat162, 1> {
  using type = __nv_bfloat16;
};
#endif
#ifdef ENABLE_FP8
template <>
struct packed_as<__nv_fp8_e4m3, 2> {
  using type = __nv_fp8x2_e4m3;
};
template <>
struct packed_as<__nv_fp8x2_e4m3, 1> {
  using type = __nv_fp8_e4m3;
};
template <>
struct packed_as<__nv_fp8_e5m2, 2> {
  using type = __nv_fp8x2_e5m2;
};
template <>
struct packed_as<__nv_fp8x2_e5m2, 1> {
  using type = __nv_fp8_e5m2;
};
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

inline __device__ float2 operator*(float2 a, float b) { return make_float2(a.x * b, a.y * b); }
inline __device__ float2 operator+(float2 a, float b) { return make_float2(a.x + b, a.y + b); }
inline __device__ float2 operator-(float2 a, float b) { return make_float2(a.x - b, a.y - b); }

}  // namespace tensorrt_llm::common
