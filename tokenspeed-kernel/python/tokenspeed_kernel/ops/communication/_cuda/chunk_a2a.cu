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

__device__ __forceinline__ void arrive_red_u64(uint64_t *p) {
  asm volatile("red.release.gpu.global.add.u64 [%0], 1;" ::"l"(p) : "memory");
}

__device__ __forceinline__ void wait_count_u64(uint64_t *p, uint64_t target) {
  uint64_t count;
  do {
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                 : "=l"(count)
                 : "l"(p)
                 : "memory");
  } while (count != target);
}

__device__ __forceinline__ uint64_t begin_epoch(uint64_t *control) {
  __shared__ uint64_t epoch;
  if (threadIdx.x == 0)
    epoch = control[0];
  __syncthreads();
  if (epoch == 0)
    asm volatile("trap;");
  if (threadIdx.x == blockDim.x - 1)
    arrive_red_u64(control + 1);
  return epoch;
}
__device__ __forceinline__ void end_epoch(uint64_t *control, uint64_t epoch) {
  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
    wait_count_u64(control + 1, gridDim.x);
    control[1] = 0;
    control[0] = epoch + 1ULL;
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
template <int NRanks, bool Inverse, bool Quantize>
__global__ __launch_bounds__(1024, 1) void chunk_a2a(
    const uint4 *input, uint4 *output, float *scales, uint64_t **peers,
    uint64_t **flag_peers, uint64_t *control, int64_t capacity, int64_t rows,
    int64_t channels, int rank) {
  const uint64_t epoch = begin_epoch(control);
  const int ring = static_cast<int>(epoch % 3ULL);
  const int64_t offset = int64_t(ring) * capacity;
  const int64_t width = channels / (8 * NRanks), count = rows * width;
  const int64_t valid_rows = NRanks * rows;
  const int64_t padded_rows = (valid_rows + 3) / 4 * 4;
  uint4 *buffers[NRanks];
#pragma unroll
  for (int p = 0; p < NRanks; ++p)
    buffers[p] = reinterpret_cast<uint4 *>(peers[p] + offset);
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x,
                stride = gridDim.x * blockDim.x;
  // A CTA owns a striped chunk of every peer segment.
  for (int64_t i = tid; i < count; i += stride) {
    const int64_t row = i / width, col = i % width;
    uint4 values[NRanks];
#pragma unroll
    for (int p = 0; p < NRanks; ++p) {
      const int64_t src =
          Inverse ? p * count + i : row * NRanks * width + p * width + col;
      values[p] = input[src];
    }
#pragma unroll
    for (int p = 0; p < NRanks; ++p) {
      const int64_t dst = Inverse ? row * NRanks * width + rank * width + col
                                  : rank * count + i;
      write4(buffers[p] + dst, values[p]);
    }
  }
  // EVERY producer thread flushes its own writes before the signaling lanes
  // announce completion. Their fences alone would not order peers' writes.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x < NRanks) {
    const int p = threadIdx.x;
    release_flag(
        flag_peers[p] + (ring * NRanks + rank) * gridDim.x + blockIdx.x, epoch);
    while (acquire_flag(flag_peers[rank] + (ring * NRanks + p) * gridDim.x +
                        blockIdx.x) != epoch) {
    }
  }
  __syncthreads();
  // The acquire and CTA barrier order subsequent payload reads. Volatile
  // vector loads bypass stale L1 lines after remote publication.
  for (int64_t i = tid; i < count; i += stride) {
    const int64_t row = i / width, col = i % width;
#pragma unroll
    for (int p = 0; p < NRanks; ++p) {
      const int64_t dst =
          Inverse ? row * NRanks * width + p * width + col : p * count + i;
      const uint4 value = read4(buffers[rank] + dst);
      if constexpr (Quantize) {
        const uint32_t words[4] = {value.x, value.y, value.z, value.w};
        quantize_a2a_group(words, reinterpret_cast<uint32_t *>(output), scales,
                           4 * dst, padded_rows, channels / NRanks,
                           valid_rows != padded_rows);
      } else {
        output[dst] = value;
      }
    }
  }
  if constexpr (Quantize)
    pad_a2a_fp8<NRanks>(reinterpret_cast<uint32_t *>(output), scales, rows,
                        channels);
  end_epoch(control, epoch);
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
void exchange_chunk_impl(TensorView flags, TensorView input, TensorView output,
                         TensorView peers, TensorView control,
                         TensorView chunk_scratch, TensorView chunk_flags,
                         int64_t capacity, int64_t rows, int64_t channels,
                         int64_t rank, int64_t blocks, bool inverse) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(peers);
  CHECK_INPUT(control);
  CHECK_INPUT(flags);
  TVM_FFI_ICHECK(channels > 0 && channels % (8 * NRanks) == 0 && rows > 0);
  TVM_FFI_ICHECK(capacity > 0 && (capacity & 1) == 0 &&
                 capacity <= INT32_MAX / 3);
  TVM_FFI_ICHECK(rows <= capacity / (channels / 4));
  TVM_FFI_ICHECK(input.dtype().code == kDLBfloat && input.dtype().bits == 16);
  TVM_FFI_ICHECK_EQ(input.dtype(), output.dtype());
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), input.numel());
  TVM_FFI_ICHECK(rank >= 0 && rank < NRanks);
  TVM_FFI_ICHECK(blocks > 0 &&
                 blocks <= device_sm_count(input.device().device_id));
  check_aligned(flags, alignof(uint64_t *));
  check_aligned(peers, alignof(uint64_t *));
  check_aligned(control, alignof(uint64_t));
  check_aligned(input, 16);
  check_aligned(output, 16);
  check_buffer(chunk_scratch, input, 3 * capacity, 16);
  check_buffer(chunk_flags, input, 3 * NRanks * blocks, 8);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, peers.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, flags.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, control.device().device_id);
  TVM_FFI_ICHECK(peers.dtype().code == kDLInt && peers.dtype().bits == 64 &&
                 peers.numel() == NRanks);
  TVM_FFI_ICHECK(flags.dtype().code == kDLInt && flags.dtype().bits == 64 &&
                 flags.numel() == NRanks);
  TVM_FFI_ICHECK(control.dtype().code == kDLInt && control.dtype().bits == 64 &&
                 control.numel() == 2);
  auto stream = get_stream(input.device());
#define CHUNK(INVERSE)                                                         \
  chunk_a2a<NRanks, INVERSE, false><<<blocks, 1024, 0, stream>>>(              \
      tensor_ptr<const uint4>(input), tensor_ptr<uint4>(output), nullptr,      \
      tensor_ptr<uint64_t *>(peers), tensor_ptr<uint64_t *>(flags),            \
      tensor_ptr<uint64_t>(control), capacity, rows, channels, rank)
  if (inverse) {
    CHUNK(true);
  } else {
    CHUNK(false);
  }
#undef CHUNK
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

void exchange_chunk(TensorView flags, TensorView input, TensorView output,
                    TensorView peers, TensorView control,
                    TensorView chunk_scratch, TensorView chunk_flags,
                    int64_t capacity, int64_t rows, int64_t channels,
                    int64_t rank, int64_t blocks, bool inverse) {
#define DISPATCH(N)                                                            \
  exchange_chunk_impl<N>(flags, input, output, peers, control, chunk_scratch,  \
                         chunk_flags, capacity, rows, channels, rank, blocks,  \
                         inverse)
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exchange_chunk, exchange_chunk);

template <int NRanks>
void exchange_chunk_fp8_impl(TensorView flags, TensorView input,
                             TensorView output, TensorView scales,
                             TensorView peers, TensorView control,
                             TensorView chunk_scratch, TensorView chunk_flags,
                             int64_t capacity, int64_t rows, int64_t channels,
                             int64_t rank, int64_t blocks) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(scales);
  CHECK_INPUT(peers);
  CHECK_INPUT(control);
  CHECK_INPUT(flags);
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(scales.dtype(), dl_float32);
  TVM_FFI_ICHECK(channels > 0 && channels % (128 * NRanks) == 0 && rows > 0);
  TVM_FFI_ICHECK(capacity > 0 && (capacity & 1) == 0 &&
                 capacity <= INT32_MAX / 3);
  TVM_FFI_ICHECK(rows <= capacity / (channels / 4));
  const int64_t padded_rows = (NRanks * rows + 3) / 4 * 4;
  TVM_FFI_ICHECK_EQ(input.numel(), rows * channels);
  TVM_FFI_ICHECK_EQ(output.numel(), padded_rows * (channels / NRanks));
  TVM_FFI_ICHECK_EQ(scales.numel(), output.numel() / 128);
  TVM_FFI_ICHECK(rank >= 0 && rank < NRanks);
  TVM_FFI_ICHECK(blocks > 0 &&
                 blocks <= device_sm_count(input.device().device_id));
  check_aligned(flags, alignof(uint64_t *));
  check_aligned(peers, alignof(uint64_t *));
  check_aligned(control, alignof(uint64_t));
  check_aligned(input, 16);
  check_aligned(output, 4);
  check_aligned(scales, 4);
  check_buffer(chunk_scratch, input, 3 * capacity, 16);
  check_buffer(chunk_flags, input, 3 * NRanks * blocks, 8);
  TVM_FFI_ICHECK_EQ(peers.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(peers.numel(), NRanks);
  TVM_FFI_ICHECK_EQ(flags.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(flags.numel(), NRanks);
  TVM_FFI_ICHECK_EQ(control.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(control.numel(), 2);
  for (auto tensor : {output, scales, peers, flags, control})
    TVM_FFI_ICHECK_EQ(input.device().device_id, tensor.device().device_id);
  chunk_a2a<NRanks, false, true>
      <<<blocks, 1024, 0, get_stream(input.device())>>>(
          tensor_ptr<const uint4>(input), tensor_ptr<uint4>(output),
          tensor_ptr<float>(scales), tensor_ptr<uint64_t *>(peers),
          tensor_ptr<uint64_t *>(flags), tensor_ptr<uint64_t>(control),
          capacity, rows, channels, rank);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
void exchange_chunk_fp8(TensorView flags, TensorView input, TensorView output,
                        TensorView scales, TensorView peers, TensorView control,
                        TensorView chunk_scratch, TensorView chunk_flags,
                        int64_t capacity, int64_t rows, int64_t channels,
                        int64_t rank, int64_t blocks) {
#define DISPATCH(N)                                                            \
  exchange_chunk_fp8_impl<N>(flags, input, output, scales, peers, control,     \
                             chunk_scratch, chunk_flags, capacity, rows,       \
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exchange_chunk_fp8, exchange_chunk_fp8);
