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

#include "a2a_fp8.cuh"
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

template <int NRanks, bool Inverse, int Pipeline, bool Quantize, int Threads>
__global__ __launch_bounds__(Threads, 1) void lamport_a2a(
    const uint32_t *input, uint32_t *output, float *scales, uint64_t **peers,
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
  const int width = channels / (2 * NRanks); // 32-bit words per shard
  const int count = rows * width;
  const int offset = (epoch % 3) * capacity;
  uint64_t *buffers[NRanks];
#pragma unroll
  for (int peer = 0; peer < NRanks; ++peer)
    buffers[peer] = peers[peer] + offset;
  const uint64_t *local = buffers[rank];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
#pragma unroll
    for (int peer = 0; peer < NRanks; ++peer) {
      const int src = Inverse ? peer * count + i
                              : row * NRanks * width + peer * width + col;
      const int dst = Inverse ? row * NRanks * width + rank * width + col
                              : rank * count + i;
      publish(buffers[peer] + dst, (uint64_t(epoch) << 32) | input[src]);
    }
  }
  // No grid barrier separates publish and polling. Every sender CTA completes
  // its bounded publish work before polling; limiting the grid permits all
  // producer CTAs to become resident even while other CTAs wait for packets.
  if constexpr (Quantize) {
    // Each half-warp polls a complete 128-element group before its scale.
    // The tagged payload and three-generation ring are shared with BF16 A2A.
    for (int i = 4 * tid; i < NRanks * count; i += 4 * stride) {
      uint64_t packets[4];
      bool ready;
      do {
        ready = true;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          packets[j] = observe(local + i + j);
          ready &= uint32_t(packets[j] >> 32) == epoch;
        }
      } while (!ready);
      const uint32_t words[4] = {uint32_t(packets[0]), uint32_t(packets[1]),
                                 uint32_t(packets[2]), uint32_t(packets[3])};
      quantize_a2a_group(words, output, scales, i,
                         NRanks == 2 ? (2 * rows + 3) / 4 * 4 : NRanks * rows,
                         channels / NRanks, NRanks == 2 && rows % 2 != 0);
    }
    pad_a2a_fp8<NRanks>(output, scales, rows, channels);
  } else {
    for (int i = tid; i < NRanks * count; i += Pipeline * stride) {
      uint64_t packets[Pipeline];
      bool ready;
      do {
        ready = true;
#pragma unroll
        for (int j = 0; j < Pipeline; ++j) {
          const int index = i + j * stride;
          if (index < NRanks * count) {
            packets[j] = observe(local + i + j * stride);
            ready &= uint32_t(packets[j] >> 32) == epoch;
          }
        }
      } while (!ready);
#pragma unroll
      for (int j = 0; j < Pipeline; ++j) {
        const int index = i + j * stride;
        if (index < NRanks * count)
          output[index] = uint32_t(packets[j]);
      }
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
// and poll the remote owners together. The packet layout and generation
// contract are identical to lamport_a2a, so shape changes reuse the same rings.
// Compile-time rank removes dynamic peer-array indexing and self-owner tests
// from the unrolled loops. Every rank instantiates the same protocol.
template <int NRanks, bool Inverse, int Threads, bool Parallel, int Rank>
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
  const int width = channels / (2 * NRanks), count = rows * width;
  const int tid = blockIdx.x * Threads + threadIdx.x,
            stride = gridDim.x * Threads;
  const uint64_t tag = uint64_t(epoch) << 32;
  uint64_t *buffers[NRanks];
#pragma unroll
  for (int p = 0; p < NRanks; ++p)
    buffers[p] = peers[p] + (epoch % 3) * capacity;

  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
#pragma unroll
    for (int step = 0; step < NRanks; ++step) {
      const int p = Parallel ? (rank + step) % NRanks : step;
      const int src =
          Inverse ? p * count + i : row * NRanks * width + p * width + col;
      const int dst = Inverse ? row * NRanks * width + rank * width + col
                              : rank * count + i;
      if (p == rank)
        output[dst] = input[src];
      else
        publish(buffers[p] + dst, tag | input[src]);
    }
  }
  for (int i = tid; i < count; i += stride) {
    const int row = i / width, col = i % width;
    uint64_t v[NRanks - 1];
    bool ready;
    do {
      ready = true;
#pragma unroll
      for (int step = 0; step < NRanks - 1; ++step) {
        const int p = (rank + 1 + step) % NRanks;
        const int dst =
            Inverse ? row * NRanks * width + p * width + col : p * count + i;
        v[step] = observe(buffers[rank] + dst);
        ready &= uint32_t(v[step] >> 32) == epoch;
      }
    } while (!ready);
#pragma unroll
    for (int step = 0; step < NRanks - 1; ++step) {
      const int p = (rank + 1 + step) % NRanks;
      const int dst =
          Inverse ? row * NRanks * width + p * width + col : p * count + i;
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

template <int NRanks>
void exchange_impl(TensorView input, TensorView output, TensorView peers,
                   TensorView control, int64_t capacity, int64_t rows,
                   int64_t channels, int64_t rank, int64_t blocks,
                   bool inverse) {
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
                 peers.numel() == NRanks);
  TVM_FFI_ICHECK(control.dtype().code == kDLInt && control.dtype().bits == 32 &&
                 control.numel() == 2);
  TVM_FFI_ICHECK(rows > 0 && channels > 0 && channels % (2 * NRanks) == 0);
  TVM_FFI_ICHECK(rank >= 0 && rank < NRanks && blocks > 0);
  TVM_FFI_ICHECK(capacity >= rows * channels / 2 && capacity <= INT32_MAX / 3);
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), input.numel());
  auto in = static_cast<const uint32_t *>(input.data_ptr());
  auto out = static_cast<uint32_t *>(output.data_ptr());
  auto ptrs = static_cast<uint64_t **>(peers.data_ptr());
  auto ctrl = static_cast<uint32_t *>(control.data_ptr());
#define LAUNCH(INVERSE, PIPELINE)                                              \
  lamport_a2a<NRanks, INVERSE, PIPELINE, false, 256>                           \
      <<<blocks, 256, 0, stream>>>(in, out, nullptr, ptrs, ctrl, capacity,     \
                                   rows, channels, rank)
  if (rows * channels * 2 >= (4 << 20) && rows * channels * 2 <= (8 << 20)) {
#define PAIRED(RANK)                                                           \
  if (inverse) {                                                               \
    paired_a2a<NRanks, true, 1024, true, RANK><<<blocks, 1024, 0, stream>>>(   \
        in, out, ptrs, ctrl, capacity, rows, channels);                        \
  } else {                                                                     \
    paired_a2a<NRanks, false, 1024, true, RANK><<<blocks, 1024, 0, stream>>>(  \
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
      if constexpr (NRanks >= 4) {
        PAIRED(2);
      }
      break;
    case 3:
      if constexpr (NRanks >= 4) {
        PAIRED(3);
      }
      break;
    case 4:
      if constexpr (NRanks == 8) {
        PAIRED(4);
      }
      break;
    case 5:
      if constexpr (NRanks == 8) {
        PAIRED(5);
      }
      break;
    case 6:
      if constexpr (NRanks == 8) {
        PAIRED(6);
      }
      break;
    case 7:
      if constexpr (NRanks == 8) {
        PAIRED(7);
      }
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
void exchange(TensorView input, TensorView output, TensorView peers,
              TensorView control, int64_t capacity, int64_t rows,
              int64_t channels, int64_t rank, int64_t blocks, bool inverse) {
#define DISPATCH(N)                                                            \
  exchange_impl<N>(input, output, peers, control, capacity, rows, channels,    \
                   rank, blocks, inverse)
  switch (peers.numel()) {
  case 2:
    DISPATCH(2);
    break;
  case 4:
    DISPATCH(4);
    break;
  case 8:
    DISPATCH(8);
    break;
  default:
    TVM_FFI_ICHECK(false) << "Lamport A2A requires TP2, TP4 or TP8";
  }
#undef DISPATCH
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exchange, exchange);

template <int NRanks>
void exchange_fp8_impl(TensorView input, TensorView output, TensorView scales,
                       TensorView peers, TensorView control, int64_t capacity,
                       int64_t rows, int64_t channels, int64_t rank,
                       int64_t blocks) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(scales);
  CHECK_INPUT(peers);
  CHECK_INPUT(control);
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(scales.dtype(), dl_float32);
  TVM_FFI_ICHECK(channels > 0 && channels % (128 * NRanks) == 0 && rows > 0);
  const int64_t padded_rows = (NRanks * rows + 3) / 4 * 4;
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), padded_rows * (channels / NRanks));
  TVM_FFI_ICHECK_EQ(scales.numel(), output.numel() / 128);
  TVM_FFI_ICHECK(rank >= 0 && rank < NRanks && blocks > 0);
  TVM_FFI_ICHECK(capacity >= rows * channels / 2 && capacity <= INT32_MAX / 3);
  TVM_FFI_ICHECK_EQ(peers.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(peers.numel(), NRanks);
  TVM_FFI_ICHECK_EQ(control.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(control.numel(), 2);
  for (auto tensor : {output, scales, peers, control})
    TVM_FFI_ICHECK_EQ(input.device().device_id, tensor.device().device_id);
  auto stream = get_stream(input.device());
#define QUANT(THREADS)                                                         \
  lamport_a2a<NRanks, false, 1, true, THREADS>                                 \
      <<<blocks, THREADS, 0, stream>>>(                                        \
          static_cast<const uint32_t *>(input.data_ptr()),                     \
          static_cast<uint32_t *>(output.data_ptr()),                          \
          static_cast<float *>(scales.data_ptr()),                             \
          static_cast<uint64_t **>(peers.data_ptr()),                          \
          static_cast<uint32_t *>(control.data_ptr()), capacity, rows,         \
          channels, rank)
  if (rows * channels * 2 >= (4 << 20)) {
    QUANT(1024);
  } else {
    QUANT(256);
  }
#undef QUANT
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
void exchange_fp8(TensorView input, TensorView output, TensorView scales,
                  TensorView peers, TensorView control, int64_t capacity,
                  int64_t rows, int64_t channels, int64_t rank,
                  int64_t blocks) {
#define DISPATCH(N)                                                            \
  exchange_fp8_impl<N>(input, output, scales, peers, control, capacity, rows,  \
                       channels, rank, blocks)
  switch (peers.numel()) {
  case 2:
    DISPATCH(2);
    break;
  case 4:
    DISPATCH(4);
    break;
  case 8:
    DISPATCH(8);
    break;
  default:
    TVM_FFI_ICHECK(false) << "Lamport A2A requires TP2, TP4 or TP8";
  }
#undef DISPATCH
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exchange_fp8, exchange_fp8);
