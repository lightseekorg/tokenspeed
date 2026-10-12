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

// Adapt the existing TRT-LLM BF16 Lamport transport, but quantize ready groups
// directly instead of materializing the ordinary gathered BF16 output.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include "cuda/csrc/include/flashinfer/comm/trtllm_reducescatter_fusion.cuh"
#include "tvm_ffi_utils.h"

namespace rs = flashinfer::trtllm_reducescatter_fusion;
using flashinfer::vec_t;

template <int NRanks>
__global__ void allgather_fp8_quantize_lamport_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output, float* scales,
    void** workspace, int rank, int rows, int hidden, int padded_rows) {
  constexpr int kVec = 8;
  constexpr int kThreads = 256;
  rs::RLamportComm<NRanks> comm(workspace, rank);
  const int tid = blockIdx.x * kThreads + threadIdx.x;
  const int stride = gridDim.x * kThreads;
  const int total_vecs = NRanks * rows * hidden / kVec;
  const int local_vecs = rows * hidden / kVec;

  // Publish the entire local slice before waiting on peers. A bounded grid
  // avoids consumer CTAs occupying the GPU while unscheduled producers wait.
  for (int i = tid; i < local_vecs; i += stride) {
    vec_t<__nv_bfloat16, kVec> value;
    value.load(input + i * kVec);
    rs::utils::remove_neg_zero(value);
#pragma unroll
    for (int peer = 0; peer < NRanks; ++peer) {
      value.store(reinterpret_cast<__nv_bfloat16*>(comm.data_bufs[peer]) +
                  (rank * local_vecs + i) * kVec);
    }
  }
  vec_t<__nv_bfloat16, kVec> empty;
  empty.fill(rs::utils::neg_zero_v<__nv_bfloat16>);
  // AllGather tracks the clear extent in elements (comm_size is in bytes).
  for (int i = tid; i < comm.clear_size / kVec; i += stride) {
    empty.store(reinterpret_cast<__nv_bfloat16*>(comm.clear_buf) + i * kVec);
  }
  __syncthreads();

  // Sixteen lanes each load eight contiguous BF16 values: one scale per
  // 128-element group. Readiness is checked on the IPC data, never on an
  // ordinary tensor whose contents may be zero or from a previous call.
  const unsigned mask = 0xffffu << ((threadIdx.x & 16) ? 16 : 0);
  for (int i = tid; i < total_vecs; i += stride) {
    vec_t<__nv_bfloat16, kVec> value;
    do {
      value.load_global_volatile(
          reinterpret_cast<__nv_bfloat16*>(comm.data_bufs[rank]) + i * kVec);
    } while (rs::utils::has_neg_zero(value));
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(float(value[j])));
    amax = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(amax)));
    // Match the active prepacked quantizer: aligned M uses TRT-LLM's BF16
    // epsilon clamp and rounded divisions; M-padding uses the existing Triton
    // fallback's neutral zero scale and approximate divisions. Do not let
    // the JIT's fast-math flags change the native quantization contract.
    const bool padded = NRanks * rows != padded_rows;
    const float clamped = fmaxf(amax, float(__nv_bfloat16(1e-10f)));
    const float multiplier = padded ? (amax != 0.f ? 448.f / amax : 1.f)
                                    : __fdiv_rn(448.f, clamped);
    const float dequant = padded ? 1.f / multiplier : __fdiv_rn(1.f, multiplier);
    const int group = i / 16;
    if ((threadIdx.x & 15) == 0) {
      const int groups_per_row = hidden / 128;
      scales[(group % groups_per_row) * padded_rows + group / groups_per_row] =
          dequant;
    }
    vec_t<__nv_fp8_e4m3, kVec> quantized;
#pragma unroll
    for (int j = 0; j < kVec; ++j) quantized[j] = __nv_fp8_e4m3(float(value[j]) * multiplier);
    quantized.store(output + i * kVec);
  }
  // GEMM's MN-major scale contract pads M to a multiple of four. This matters
  // for a TP2 C1 call; padded rows never participate in the transport.
  for (int i = NRanks * rows * hidden + tid; i < padded_rows * hidden; i += stride)
    output[i] = __nv_fp8_e4m3(0.f);
  for (int i = tid; i < (padded_rows - NRanks * rows) * (hidden / 128); i += stride) {
    int padding = padded_rows - NRanks * rows;
    scales[(i / padding) * padded_rows + NRanks * rows + i % padding] = 1.f;
  }
  // Retain the existing three-buffer rotation and clear-size tracking. Calls
  // sharing this state must be serialized; the output buffers are borrowed.
  comm.update(NRanks * rows * hidden);
}

void allgather_fp8_quantize(TensorView input, TensorView output, TensorView scales,
                          TensorView workspace, int64_t rank, int64_t size,
                          int64_t max_blocks) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(scales);
  CHECK_INPUT(workspace);
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(scales.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(workspace.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  const int rows = input.size(0), hidden = input.size(1);
  const int padded_rows = ((size * rows + 3) / 4) * 4;
  TVM_FFI_ICHECK(rows > 0 && rows <= 128 && hidden > 0 && hidden % 128 == 0);
  TVM_FFI_ICHECK(rank >= 0 && rank < size && max_blocks > 0);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  TVM_FFI_ICHECK_EQ(output.size(0), padded_rows);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden);
  TVM_FFI_ICHECK_EQ(scales.ndim(), 2);
  TVM_FFI_ICHECK_EQ(scales.size(0), hidden / 128);
  TVM_FFI_ICHECK_EQ(scales.size(1), padded_rows);
  TVM_FFI_ICHECK_EQ(workspace.numel(), 3 * size + 1);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, scales.device().device_id);
  TVM_FFI_ICHECK_EQ(input.device().device_id, workspace.device().device_id);
  ffi::CUDADeviceGuard guard(input.device().device_id);
  auto stream = get_stream(input.device());
  const int blocks = std::min<int64_t>(max_blocks, (size * rows * hidden + 2047) / 2048);
#define LAUNCH(P) allgather_fp8_quantize_lamport_kernel<P><<<blocks, 256, 0, stream>>>( \
    static_cast<const __nv_bfloat16*>(input.data_ptr()), \
    static_cast<__nv_fp8_e4m3*>(output.data_ptr()), static_cast<float*>(scales.data_ptr()), \
    reinterpret_cast<void**>(workspace.data_ptr()), rank, rows, hidden, padded_rows)
  switch (size) {
    case 2: LAUNCH(2); break;
    case 4: LAUNCH(4); break;
    case 8: LAUNCH(8); break;
    case 16: LAUNCH(16); break;
    default: TVM_FFI_ICHECK(false) << "Unsupported AllGather TP size";
  }
#undef LAUNCH
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(allgather_fp8_quantize, allgather_fp8_quantize);
