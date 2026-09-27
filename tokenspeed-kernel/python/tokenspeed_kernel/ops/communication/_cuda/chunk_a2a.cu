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

#include "tvm_ffi_utils.h"
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t begin_epoch(uint32_t *control) {
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0)
    epoch = control[0];
  __syncthreads();
  if (epoch == 0)
    asm volatile("trap;");
  if (threadIdx.x == 0)
    atomicAdd(control + 1, 1);
  return epoch;
}
__device__ __forceinline__ void end_epoch(uint32_t *control, uint32_t epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (atomicAdd(control + 1, 0) != gridDim.x) {
    }
    control[1] = 0;
    control[0] = epoch + 1;
  }
}
__device__ __forceinline__ uint4 read4(const uint4 *p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void write4(uint4 *p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void release_flag(uint64_t *p, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(value)
               : "memory");
}
__device__ __forceinline__ uint64_t acquire_flag(const uint64_t *p) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(p)
               : "memory");
  return value;
}
template <bool Inverse>
__global__ __launch_bounds__(1024, 1) void chunk_a2a(
    const uint4 *input, uint4 *output, uint64_t **peers, uint64_t **flag_peers,
    uint32_t *control, int capacity, int rows, int channels, int rank) {
  const uint32_t epoch = begin_epoch(control);
  const int ring = epoch % 3, width = channels / 32, count = rows * width;
  uint4 *buffers[4];
#pragma unroll
  for (int p = 0; p < 4; ++p)
    buffers[p] = reinterpret_cast<uint4 *>(peers[p] + ring * capacity);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x,
            stride = gridDim.x * blockDim.x;
  // A CTA owns a striped chunk of every peer segment.
  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
    uint4 values[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int src =
          Inverse ? p * count + i : row * 4 * width + p * width + col;
      values[p] = input[src];
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int dst =
          Inverse ? row * 4 * width + rank * width + col : rank * count + i;
      write4(buffers[p] + dst, values[p]);
    }
  }
  // EVERY producer thread flushes its own writes before the signaling lanes
  // announce completion. Their fences alone would not order peers' writes.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x < 4) {
    const int p = threadIdx.x;
    release_flag(flag_peers[p] + (ring * 4 + rank) * gridDim.x + blockIdx.x,
                 epoch);
    while (acquire_flag(flag_peers[rank] + (ring * 4 + p) * gridDim.x +
                        blockIdx.x) != epoch) {
    }
  }
  __syncthreads();
  // The acquire and CTA barrier order subsequent payload reads. Volatile
  // vector loads bypass stale L1 lines after remote publication.
  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int dst =
          Inverse ? row * 4 * width + p * width + col : p * count + i;
      output[dst] = read4(buffers[rank] + dst);
    }
  }
  end_epoch(control, epoch);
}
void exchange_chunk(TensorView flags, TensorView input, TensorView output,
                    TensorView peers, TensorView control, int64_t capacity,
                    int64_t rows, int64_t channels, int64_t rank,
                    int64_t blocks, bool inverse) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(peers);
  CHECK_INPUT(control);
  CHECK_INPUT(flags);
  TVM_FFI_ICHECK(channels >= 32 && channels % 32 == 0 && rows > 0);
  TVM_FFI_ICHECK(input.dtype().code == kDLBfloat && input.dtype().bits == 16);
  TVM_FFI_ICHECK_EQ(input.dtype(), output.dtype());
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), input.numel());
  TVM_FFI_ICHECK(rank >= 0 && rank < 4 && blocks > 0);
  TVM_FFI_ICHECK(capacity >= rows * channels / 4 && capacity <= INT32_MAX / 3);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, peers.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, flags.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, control.device().device_id);
  TVM_FFI_ICHECK(peers.dtype().code == kDLInt && peers.dtype().bits == 64 &&
                 peers.numel() == 4);
  TVM_FFI_ICHECK(flags.dtype().code == kDLInt && flags.dtype().bits == 64 &&
                 flags.numel() == 4);
  TVM_FFI_ICHECK(control.dtype().code == kDLInt && control.dtype().bits == 32 &&
                 control.numel() == 2);
  auto stream = get_stream(input.device());
#define CHUNK(INVERSE)                                                         \
  chunk_a2a<INVERSE><<<blocks, 1024, 0, stream>>>(                             \
      static_cast<const uint4 *>(input.data_ptr()),                            \
      static_cast<uint4 *>(output.data_ptr()),                                 \
      static_cast<uint64_t **>(peers.data_ptr()),                              \
      static_cast<uint64_t **>(flags.data_ptr()),                              \
      static_cast<uint32_t *>(control.data_ptr()), capacity, rows, channels,   \
      rank)
  if (inverse) {
    CHUNK(true);
  } else {
    CHUNK(false);
  }
#undef CHUNK
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exchange_chunk, exchange_chunk);
