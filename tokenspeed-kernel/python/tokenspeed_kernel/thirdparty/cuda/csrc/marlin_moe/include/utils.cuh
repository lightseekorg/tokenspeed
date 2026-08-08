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

// Minimal device-side prelude for the vendored Marlin MoE GEMM kernel.
//
// The original upstream header pulled in a full Tensor/FFI stack; the Marlin
// kernel body only needs the scalar dtype aliases and the device qualifier
// macro, so this trimmed drop-in replaces it.
#pragma once

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <utility>

// Scalar / packed dtype aliases used by marlin/marlin_dtypes.cuh (global
// namespace, matching the upstream layout so the kernel refers to them bare).
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;
using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp32x4_t = float4;

namespace device {
/// Forced-inline device-function qualifier used across the Marlin kernel.
#define SGL_DEVICE __forceinline__ __device__

/// Ceil-division used across the Marlin GEMM (host + device).
template <typename T, typename U>
__host__ __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}
}  // namespace device

// Trimmed host-side helpers the vendored moe_align_kernel.cu still references
// (RuntimeCheck / RuntimeDeviceCheck / LaunchKernel). Behaviour matches the
// original versions for the launch shapes marlin uses (no PDL / cluster),
// routed through TVM-FFI error reporting.
namespace host {

template <typename Cond, typename... Args>
inline void RuntimeCheck(Cond&& condition, Args&&...) {
  TVM_FFI_ICHECK(static_cast<bool>(condition));
}

inline void RuntimeDeviceCheck(cudaError_t error) {
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

// Fatal, never-returns error path (the entry uses it for an unreachable
// moe_block_size). Routed through TVM-FFI so it surfaces as an FFI error.
template <typename... Args>
[[noreturn]] inline void Panic(Args&&...) {
  TVM_FFI_ICHECK(false) << "marlin_moe: unrecoverable error";
  __builtin_unreachable();
}

struct LaunchKernel {
  dim3 grid_dim;
  dim3 block_dim;
  cudaStream_t stream;
  std::size_t dynamic_shared_mem_bytes;

  explicit LaunchKernel(
      dim3 grid, dim3 block, cudaStream_t s, std::size_t smem = 0) noexcept
      : grid_dim(grid), block_dim(block), stream(s), dynamic_shared_mem_bytes(smem) {}

  static cudaStream_t resolve_device(DLDevice device) {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  template <typename T, typename... Args>
  void operator()(T&& kernel, Args&&... args) const {
    cudaLaunchConfig_t config{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = dynamic_shared_mem_bytes;
    config.stream = stream;
    config.numAttrs = 0;
    RuntimeDeviceCheck(
        ::cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
  }
};

}  // namespace host
