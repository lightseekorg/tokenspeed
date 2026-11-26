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

// Collective progress requires a fully co-resident grid (blocks <= SM count).
// Concurrent blocking kernels must not pin SMs during the exchange.
static int device_sm_count(int device_id) {
  int sm_count = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sm_count,
                                        cudaDevAttrMultiProcessorCount,
                                        device_id) == cudaSuccess);
  return sm_count;
}

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

__device__ __forceinline__ void arrive_red(uint32_t *p) {
  asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

__device__ __forceinline__ void wait_count(uint32_t *p, uint32_t target) {
  uint32_t count;
  do {
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(count)
                 : "l"(p)
                 : "memory");
  } while (count != target);
}

__device__ __forceinline__ uint32_t begin_epoch(uint32_t *control) {
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0)
    epoch = control[0];
  __syncthreads();
  if (epoch == 0)
    asm volatile("trap;");
  if (threadIdx.x == blockDim.x - 1)
    arrive_red(control + 1);
  return epoch;
}

template <int NRanks>
__device__ __forceinline__ void peer_barrier(uint64_t *const *peers, int rank,
                                             int64_t flags, uint64_t gen) {
  __syncthreads();
  __threadfence_system();
  const int peer = threadIdx.x;
  if (peer < NRanks) {
    asm volatile(
        "st.release.sys.global.u64 [%0], %1;" ::"l"(peers[peer] + flags + rank),
        "l"(gen)
        : "memory");
    uint64_t observed;
    do {
      asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                   : "=l"(observed)
                   : "l"(peers[rank] + flags + peer)
                   : "memory");
    } while (observed != gen);
  }
  __syncthreads();
}

template <int NRanks>
__device__ __forceinline__ void end_epoch(uint32_t *control, uint32_t epoch,
                                          uint64_t *const *peers, int rank,
                                          int64_t capacity) {
  if (epoch != UINT32_MAX) {
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
      wait_count(control + 1, gridDim.x);
      control[1] = 0;
      control[0] = epoch + 1u;
    }
    return;
  }

  // Finish all reads and entry arrivals before any rank clears its receive
  // slots.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    arrive_red(control + 2);
  }
  if (blockIdx.x != 0)
    return;
  if (threadIdx.x == 0)
    wait_count(control + 2, gridDim.x);
  const uint64_t gen = uint64_t(control[3]) + 1ULL;
  const int64_t words = 3 * capacity;
  peer_barrier<NRanks>(peers, rank, words, gen);
  for (int64_t i = threadIdx.x; i < words; i += blockDim.x)
    publish(peers[rank] + i, 0);
  __threadfence_system();
  __syncthreads();
  peer_barrier<NRanks>(peers, rank, words + NRanks, gen);
  if (threadIdx.x == 0) {
    control[1] = 0;
    control[2] = 0;
    control[3] = uint32_t(gen);
    control[0] = 1;
  }
}

template <int NRanks, bool Inverse, int Pipeline, bool Quantize,
          bool Prequantized, int Threads>
__global__ __launch_bounds__(Threads, 1) void lamport_a2a(
    const uint32_t *input, uint32_t *output, float *scales, uint64_t **peers,
    uint32_t *control, int64_t capacity, int rows, int channels, int rank,
    const uint32_t *input_scales, int input_rows) {
  const uint32_t epoch = begin_epoch(control);
  const int value_width = channels / ((Prequantized ? 4 : 2) * NRanks);
  const int scale_width = Prequantized ? channels / (128 * NRanks) : 0;
  // FP8 values and scales share one tagged row; both fit in the BF16 ring.
  const int width = value_width + scale_width;
  const int count = rows * width;
  const int64_t offset = (epoch % 3) * capacity;
  uint64_t *buffers[NRanks];
#pragma unroll
  for (int peer = 0; peer < NRanks; ++peer)
    buffers[peer] = peers[peer] + offset;
  const uint64_t *local = buffers[rank];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  // Small BF16 exchanges have spare producer warps. Stripe them across
  // peers so a single warp does not serialize every peer's store.
  constexpr int peer_groups =
      Pipeline == 1 && !Quantize && !Prequantized ? NRanks : 1;
  const int first_peer = (tid / 32) % peer_groups;
  const int first_word =
      peer_groups == 1 ? tid : (tid / (32 * peer_groups)) * 32 + tid % 32;
  for (int i = first_word; i < count; i += stride / peer_groups) {
    const int row = i / width, col = i % width;
#pragma unroll
    for (int group = 0; group < NRanks / peer_groups; ++group) {
      const int peer = first_peer + group * peer_groups;
      uint32_t payload;
      if constexpr (Prequantized) {
        // Every owner publishes the common physical row count, including
        // empty owners. Do not read beyond its logical input/scale rows.
        payload = 0;
        if (row < input_rows) {
          if (col < value_width)
            payload = input[row * NRanks * value_width + peer * value_width + col];
          else
            payload = input_scales[row * NRanks * scale_width +
                                   peer * scale_width + col - value_width];
        }
      } else {
        const int src = Inverse ? peer * count + i
                                : row * NRanks * width + peer * width + col;
        payload = input[src];
      }
      const int dst = Inverse ? row * NRanks * width + rank * width + col
                              : rank * count + i;
      uint64_t *target =
          peer_groups == 1 ? buffers[peer] : peers[peer] + offset;
      publish(target + dst, (uint64_t(epoch) << 32) | payload);
    }
  }
  // No grid barrier separates publish and polling. Every sender CTA completes
  // its bounded publish work before polling; limiting the grid permits all
  // producer CTAs to become resident even while other CTAs wait for packets.
  if constexpr (Quantize) {
    // Each half-warp polls a complete 128-element group before its scale.
    // The tagged payload and three-generation ring are shared with BF16 A2A.
    const int valid_rows = NRanks * rows;
    const int padded_rows = (valid_rows + 3) / 4 * 4;
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
      quantize_a2a_group(words, output, scales, i, padded_rows,
                         channels / NRanks, valid_rows != padded_rows);
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
        if (index < NRanks * count) {
          if constexpr (Prequantized) {
            const int row = index / width, col = index % width;
            if (col < value_width)
              output[row * value_width + col] = uint32_t(packets[j]);
            else
              reinterpret_cast<uint32_t *>(scales)[row * scale_width +
                                                   col - value_width] =
                  uint32_t(packets[j]);
          } else {
            output[index] = uint32_t(packets[j]);
          }
        }
      }
    }
  }
  end_epoch<NRanks>(control, epoch, peers, rank, capacity);
}

// Medium messages use more resident warps, avoid staging self-owned words,
// and poll the remote owners together. The packet layout and generation
// contract are identical to lamport_a2a, so shape changes reuse the same rings.
// Compile-time rank removes dynamic peer-array indexing and self-owner tests
// from the unrolled loops. Every rank instantiates the same protocol.
template <int NRanks, bool Inverse, int Threads, bool Parallel, int Rank>
__global__ __launch_bounds__(Threads, 1) void paired_a2a(
    const uint32_t *input, uint32_t *output, uint64_t **peers,
    uint32_t *control, int64_t capacity, int rows, int channels) {
  constexpr int rank = Rank;
  const uint32_t epoch = begin_epoch(control);
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
  end_epoch<NRanks>(control, epoch, peers, rank, capacity);
}

static uintptr_t effective_addr(TensorView tensor) {
  const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const auto offset = static_cast<uint64_t>(tensor.byte_offset());
  TVM_FFI_ICHECK(offset <= UINTPTR_MAX - base);
  return base + static_cast<uintptr_t>(offset);
}

template <typename T> static T *tensor_ptr(TensorView tensor) {
  return reinterpret_cast<T *>(effective_addr(tensor));
}

static void check_aligned(TensorView tensor, size_t alignment) {
  TVM_FFI_ICHECK(effective_addr(tensor) % alignment == 0);
}

static void check_buffer(TensorView buffer, TensorView input, int64_t words,
                         size_t alignment) {
  CHECK_INPUT(buffer);
  TVM_FFI_ICHECK_EQ(buffer.device().device_id, input.device().device_id);
  TVM_FFI_ICHECK_EQ(buffer.dtype(), dl_int64);
  check_aligned(buffer, alignment);
  TVM_FFI_ICHECK_EQ(buffer.numel(), words);
}

template <int NRanks>
void exchange_impl(TensorView input, TensorView output, TensorView peers,
                   TensorView control, TensorView scratch, int64_t capacity,
                   int64_t rows, int64_t channels, int64_t rank, int64_t blocks,
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
                 control.numel() == 4);
  TVM_FFI_ICHECK(rows > 0 && channels > 0 && channels % (2 * NRanks) == 0);
  TVM_FFI_ICHECK(capacity > 0 && capacity <= (INT32_MAX - 2 * NRanks) / 3);
  TVM_FFI_ICHECK(rows <= capacity / (channels / 2));
  TVM_FFI_ICHECK(rank >= 0 && rank < NRanks);
  TVM_FFI_ICHECK(blocks > 0 &&
                 blocks <= device_sm_count(input.device().device_id));
  check_aligned(peers, alignof(uint64_t *));
  check_aligned(control, alignof(uint32_t));
  check_aligned(input, 4);
  check_aligned(output, 4);
  check_buffer(scratch, input, 3 * capacity + 2 * NRanks, 8);
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), input.numel());
  auto in = tensor_ptr<const uint32_t>(input);
  auto out = tensor_ptr<uint32_t>(output);
  auto ptrs = tensor_ptr<uint64_t *>(peers);
  auto ctrl = tensor_ptr<uint32_t>(control);
#define LAUNCH(INVERSE, PIPELINE)                                              \
  lamport_a2a<NRanks, INVERSE, PIPELINE, false, false, 256>                           \
      <<<blocks, 256, 0, stream>>>(in, out, nullptr, ptrs, ctrl, capacity,     \
                                   rows, channels, rank, nullptr, rows)
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
              TensorView control, TensorView scratch, int64_t capacity,
              int64_t rows, int64_t channels, int64_t rank, int64_t blocks,
              bool inverse) {
#define DISPATCH(N)                                                            \
  exchange_impl<N>(input, output, peers, control, scratch, capacity, rows,     \
                   channels, rank, blocks, inverse)
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
                       TensorView peers, TensorView control, TensorView scratch,
                       int64_t capacity, int64_t rows, int64_t channels,
                       int64_t rank, int64_t blocks) {
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
  TVM_FFI_ICHECK(capacity > 0 && capacity <= (INT32_MAX - 2 * NRanks) / 3);
  TVM_FFI_ICHECK(rows <= capacity / (channels / 2));
  const int64_t padded_rows = (NRanks * rows + 3) / 4 * 4;
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), padded_rows * (channels / NRanks));
  TVM_FFI_ICHECK_EQ(scales.numel(), output.numel() / 128);
  TVM_FFI_ICHECK(rank >= 0 && rank < NRanks);
  TVM_FFI_ICHECK(blocks > 0 &&
                 blocks <= device_sm_count(input.device().device_id));
  check_aligned(peers, alignof(uint64_t *));
  check_aligned(control, alignof(uint32_t));
  check_aligned(input, 4);
  check_aligned(output, 4);
  check_aligned(scales, 4);
  check_buffer(scratch, input, 3 * capacity + 2 * NRanks, 8);
  TVM_FFI_ICHECK_EQ(peers.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(peers.numel(), NRanks);
  TVM_FFI_ICHECK_EQ(control.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(control.numel(), 4);
  for (auto tensor : {output, scales, peers, control})
    TVM_FFI_ICHECK_EQ(input.device().device_id, tensor.device().device_id);
  auto stream = get_stream(input.device());
#define QUANT(THREADS)                                                         \
  lamport_a2a<NRanks, false, 1, true, false, THREADS>                                 \
      <<<blocks, THREADS, 0, stream>>>(                                        \
          tensor_ptr<const uint32_t>(input), tensor_ptr<uint32_t>(output),     \
          tensor_ptr<float>(scales), tensor_ptr<uint64_t *>(peers),            \
          tensor_ptr<uint32_t>(control), capacity, rows, channels, rank,       \
          nullptr, rows)
  if (rows * channels * 2 >= (4 << 20)) {
    QUANT(1024);
  } else {
    QUANT(256);
  }
#undef QUANT
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
void exchange_fp8(TensorView input, TensorView output, TensorView scales,
                  TensorView peers, TensorView control, TensorView scratch,
                  int64_t capacity, int64_t rows, int64_t channels,
                  int64_t rank, int64_t blocks) {
#define DISPATCH(N)                                                            \
  exchange_fp8_impl<N>(input, output, scales, peers, control, scratch,         \
                       capacity, rows, channels, rank, blocks)
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

// Exchange prequantized values and scales without packing or requantization.
// The physical row count is common; input_rows may differ on every owner.
template <int NRanks>
void exchange_prequantized_impl(
    TensorView input, TensorView input_scales, TensorView output,
    TensorView output_scales, TensorView peers, TensorView control,
    TensorView scratch, int64_t capacity, int64_t rows, int64_t channels,
    int64_t rank, int64_t blocks) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  for (auto tensor : {input, input_scales, output, output_scales, peers, control}) {
    CHECK_INPUT(tensor);
    TVM_FFI_ICHECK_EQ(input.device().device_id, tensor.device().device_id);
  }
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(input_scales.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(output_scales.dtype(), dl_float32);
  TVM_FFI_ICHECK(channels > 0 && channels % (128 * NRanks) == 0 && rows > 0);
  TVM_FFI_ICHECK(capacity > 0 && capacity <= (INT32_MAX - 2 * NRanks) / 3);
  TVM_FFI_ICHECK(rows <= capacity / (channels / 2));
  TVM_FFI_ICHECK_EQ(input.numel() % channels, 0);
  const int64_t input_rows = input.numel() / channels;
  TVM_FFI_ICHECK(input_rows <= rows);
  TVM_FFI_ICHECK_EQ(input_scales.numel(), input.numel() / 128);
  TVM_FFI_ICHECK_EQ(output.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output_scales.numel(), output.numel() / 128);
  TVM_FFI_ICHECK(rank >= 0 && rank < NRanks);
  TVM_FFI_ICHECK(blocks > 0 &&
                 blocks <= device_sm_count(input.device().device_id));
  check_aligned(peers, alignof(uint64_t *));
  check_aligned(control, alignof(uint32_t));
  check_aligned(input, 4);
  check_aligned(input_scales, 4);
  check_aligned(output, 4);
  check_aligned(output_scales, 4);
  check_buffer(scratch, input, 3 * capacity + 2 * NRanks, 8);
  TVM_FFI_ICHECK_EQ(peers.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(peers.numel(), NRanks);
  TVM_FFI_ICHECK_EQ(control.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(control.numel(), 4);
  auto stream = get_stream(input.device());
#define PREQUANTIZED(PIPELINE)                                                 \
  lamport_a2a<NRanks, false, PIPELINE, false, true, 256>                         \
      <<<blocks, 256, 0, stream>>>(                                            \
          tensor_ptr<const uint32_t>(input), tensor_ptr<uint32_t>(output),     \
          tensor_ptr<float>(output_scales), tensor_ptr<uint64_t *>(peers),     \
          tensor_ptr<uint32_t>(control), capacity, rows, channels, rank,       \
          tensor_ptr<const uint32_t>(input_scales), input_rows)
  if (rows * channels * 33 / 128 <= blocks * 256) {
    PREQUANTIZED(1);
  } else {
    PREQUANTIZED(4);
  }
#undef PREQUANTIZED
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
void exchange_prequantized(
    TensorView input, TensorView input_scales, TensorView output,
    TensorView output_scales, TensorView peers, TensorView control,
    TensorView scratch, int64_t capacity, int64_t rows, int64_t channels,
    int64_t rank, int64_t blocks) {
#define DISPATCH(N)                                                            \
  exchange_prequantized_impl<N>(input, input_scales, output, output_scales,      \
                                peers, control, scratch, capacity, rows,       \
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exchange_prequantized, exchange_prequantized);
