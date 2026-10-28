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

// tvm_ffi glue and wavefront helpers shared by the HIP kernels.

#pragma once

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

// The stream torch set current for the tensors' device (a ROCm torch passes kDLROCM tensors).
inline hipStream_t get_stream(DLDevice device) {
  return static_cast<hipStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

inline int64_t get_element_size(const TensorView& x) {
  return (x.dtype().bits * x.dtype().lanes) / 8;
}

// The wavefront's lanes (64 on gfx9).
constexpr int kWaveSize = 64;

// Orders shared-memory accesses across the lanes of one wavefront (CUDA's __syncwarp).
__device__ __forceinline__ void wave_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
