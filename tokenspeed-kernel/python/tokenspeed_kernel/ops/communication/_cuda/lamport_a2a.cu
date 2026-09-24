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

// Publish readiness and payload in the same naturally aligned 64-bit system
// transaction. Separate flags would require a release fence and more polling.
// Unlike a floating-point sentinel, all payload bit patterns remain valid.
__device__ __forceinline__ void publish(uint64_t *p, uint64_t value) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(p), "l"(value)
               : "memory");
}
__device__ __forceinline__ uint64_t observe(const uint64_t *p) {
  uint64_t value;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(p)
               : "memory");
  return value;
}

template <bool Inverse, int Pipeline>
__global__ __launch_bounds__(256, 1) void lamport_a2a(
    const uint32_t *input, uint32_t *output, uint64_t **peers,
    uint32_t *control, int capacity, int rows, int channels, int rank) {
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0)
    epoch = control[0];
  __syncthreads();
  // Never silently accept stale packets on 32-bit generation wrap. This
  // experimental communicator must be recreated before 2^32 exchanges.
  if (epoch == 0)
    asm volatile("trap;");
  // Every CTA snapshots the epoch before block 0 advances it for the next
  // kernel. Grid <= SM count permits all entry counters to become resident.
  if (threadIdx.x == 0)
    atomicAdd(control + 1, 1);
  const int width = channels / 8; // 32-bit words in one channel shard
  const int count = rows * width;
  const int offset = (epoch % 3) * capacity;
  uint64_t *buffers[4];
#pragma unroll
  for (int peer = 0; peer < 4; ++peer)
    buffers[peer] = peers[peer] + offset;
  const uint64_t *local = buffers[rank];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      const int src =
          Inverse ? peer * count + i : row * 4 * width + peer * width + col;
      const int dst =
          Inverse ? row * 4 * width + rank * width + col : rank * count + i;
      publish(buffers[peer] + dst, (uint64_t(epoch) << 32) | input[src]);
    }
  }
  // No grid barrier separates publish and polling. Every sender CTA completes
  // its bounded publish work before polling; limiting the grid permits all
  // producer CTAs to become resident even while other CTAs wait for packets.
  for (int i = tid; i < 4 * count; i += Pipeline * stride) {
    uint64_t packets[Pipeline];
    bool ready;
    do {
      ready = true;
#pragma unroll
      for (int j = 0; j < Pipeline; ++j) {
        const int index = i + j * stride;
        if (index < 4 * count) {
          packets[j] = observe(local + i + j * stride);
          ready &= uint32_t(packets[j] >> 32) == epoch;
        }
      }
    } while (!ready);
#pragma unroll
    for (int j = 0; j < Pipeline; ++j) {
      const int index = i + j * stride;
      if (index < 4 * count)
        output[index] = uint32_t(packets[j]);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (atomicAdd(control + 1, 0) != gridDim.x) {
    }
    control[1] = 0;
    control[0] = epoch + 1;
  }
}

// Medium messages use more resident warps, avoid staging self-owned words,
// and poll the three remote owners together. The packet layout and generation
// contract are identical to lamport_a2a, so shape changes reuse the same rings.
// Compile-time rank removes dynamic peer-array indexing and self-owner tests
// from the unrolled loops. Every TP4 rank instantiates the same protocol.
template <bool Inverse, int Threads, bool Parallel, int Rank>
__global__ __launch_bounds__(Threads, 1) void paired_a2a(
    const uint32_t *input, uint32_t *output, uint64_t **peers,
    uint32_t *control, int capacity, int rows, int channels) {
  constexpr int rank = Rank;
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0)
    epoch = control[0];
  __syncthreads();
  if (epoch == 0)
    asm volatile("trap;");
  if (threadIdx.x == 0)
    atomicAdd(control + 1, 1);
  const int width = channels / 8, count = rows * width;
  const int tid = blockIdx.x * Threads + threadIdx.x,
            stride = gridDim.x * Threads;
  const uint64_t tag = uint64_t(epoch) << 32;
  uint64_t *buffers[4];
#pragma unroll
  for (int p = 0; p < 4; ++p)
    buffers[p] = peers[p] + (epoch % 3) * capacity;

  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int p = Parallel ? (rank + step) % 4 : step;
      const int src =
          Inverse ? p * count + i : row * 4 * width + p * width + col;
      const int dst =
          Inverse ? row * 4 * width + rank * width + col : rank * count + i;
      if (p == rank)
        output[dst] = input[src];
      else
        publish(buffers[p] + dst, tag | input[src]);
    }
  }
  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
    uint64_t v[3];
    bool ready;
    do {
      ready = true;
#pragma unroll
      for (int step = 0; step < 3; ++step) {
        const int p = (rank + 1 + step) % 4;
        const int dst =
            Inverse ? row * 4 * width + p * width + col : p * count + i;
        v[step] = observe(buffers[rank] + dst);
        ready &= uint32_t(v[step] >> 32) == epoch;
      }
    } while (!ready);
#pragma unroll
    for (int step = 0; step < 3; ++step) {
      const int p = (rank + 1 + step) % 4;
      const int dst =
          Inverse ? row * 4 * width + p * width + col : p * count + i;
      output[dst] = uint32_t(v[step]);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (atomicAdd(control + 1, 0) != gridDim.x) {
    }
    control[1] = 0;
    control[0] = epoch + 1;
  }
}

void exchange(TensorView input, TensorView output, TensorView peers,
              TensorView control, int64_t capacity, int64_t rows,
              int64_t channels, int64_t rank, int64_t blocks, bool inverse) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  auto stream = get_stream(input.device());
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(peers);
  CHECK_INPUT(control);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, peers.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, control.device().device_id);
  TVM_FFI_ICHECK(input.dtype().code == kDLBfloat && input.dtype().bits == 16);
  TVM_FFI_ICHECK_EQ(input.dtype(), output.dtype());
  TVM_FFI_ICHECK(peers.dtype().code == kDLInt && peers.dtype().bits == 64 &&
                 peers.numel() == 4);
  TVM_FFI_ICHECK(control.dtype().code == kDLInt && control.dtype().bits == 32 &&
                 control.numel() == 2);
  TVM_FFI_ICHECK(rows > 0 && channels >= 8 && channels % 8 == 0);
  TVM_FFI_ICHECK(rank >= 0 && rank < 4 && blocks > 0);
  TVM_FFI_ICHECK(capacity >= rows * channels / 2 && capacity <= INT32_MAX / 3);
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), input.numel());
  auto in = static_cast<const uint32_t *>(input.data_ptr());
  auto out = static_cast<uint32_t *>(output.data_ptr());
  auto ptrs = static_cast<uint64_t **>(peers.data_ptr());
  auto ctrl = static_cast<uint32_t *>(control.data_ptr());
#define LAUNCH(INVERSE, PIPELINE)                                              \
  lamport_a2a<INVERSE, PIPELINE><<<blocks, 256, 0, stream>>>(                  \
      in, out, ptrs, ctrl, capacity, rows, channels, rank)
  if (rows * channels * 2 >= (4 << 20) && rows * channels * 2 <= (8 << 20)) {
#define PAIRED(RANK)                                                           \
  if (inverse) {                                                               \
    paired_a2a<true, 1024, true, RANK><<<blocks, 1024, 0, stream>>>(           \
        in, out, ptrs, ctrl, capacity, rows, channels);                        \
  } else {                                                                     \
    paired_a2a<false, 1024, true, RANK><<<blocks, 1024, 0, stream>>>(          \
        in, out, ptrs, ctrl, capacity, rows, channels);                        \
  }
    switch (rank) {
    case 0:
      PAIRED(0);
      break;
    case 1:
      PAIRED(1);
      break;
    case 2:
      PAIRED(2);
      break;
    case 3:
      PAIRED(3);
      break;
    }
#undef PAIRED
  } else if (rows * channels / 2 <= blocks * 256) {
    if (inverse) {
      LAUNCH(true, 1);
    } else {
      LAUNCH(false, 1);
    }
  } else {
    if (inverse) {
      LAUNCH(true, 4);
    } else {
      LAUNCH(false, 4);
    }
  }
#undef LAUNCH
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exchange, exchange);
