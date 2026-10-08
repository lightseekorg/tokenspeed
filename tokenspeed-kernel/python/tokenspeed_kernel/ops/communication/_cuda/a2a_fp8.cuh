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
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Sixteen contiguous lanes collectively own one 128-element group, with eight
// BF16 values per lane. Packet and chunk consumers establish readiness first.
__device__ __forceinline__ void
quantize_a2a_group(const uint32_t (&words)[4], uint32_t *output, float *scales,
                   int word_index, int rows, int channels, bool padded) {
  constexpr int Words = 4, lanes = 16;
  const unsigned mask = 0xffffu << (threadIdx.x & 16);
  float values[2 * Words];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 2 * Words; ++j) {
    values[j] =
        __bfloat162float(__ushort_as_bfloat16(words[j / 2] >> ((j % 2) * 16)));
    amax = fmaxf(amax, fabsf(values[j]));
  }
  amax = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(amax)));
  // The prepared quantizer uses Triton for padded M (approximate reciprocals,
  // unit scale for zero groups), and TRT-LLM otherwise (BF16 epsilon and RN
  // divisions). Match both contracts, as the fused AllGather quantizer does.
  const float multiplier =
      padded ? (amax != 0.f ? 448.f / amax : 1.f)
             : __fdiv_rn(448.f, fmaxf(amax, float(__nv_bfloat16(1e-10f))));
  if ((threadIdx.x & (lanes - 1)) == 0) {
    const int group = word_index / 64;
    const int groups_per_row = channels / 128;
    scales[(group % groups_per_row) * rows + group / groups_per_row] =
        padded ? 1.f / multiplier : __fdiv_rn(1.f, multiplier);
  }
#pragma unroll
  for (int j = 0; j < Words / 2; ++j) {
    uint32_t packed = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v)
      packed |= uint32_t(__nv_fp8_e4m3(values[j * 4 + v] * multiplier).__x)
                << (8 * v);
    output[word_index / 2 + j] = packed;
  }
}

// Prepared FP8 GEMMs require the gathered row count divisible by four. Tail
// rows are local padding, never part of the exchange protocol.
template <int NRanks>
__device__ __forceinline__ void pad_a2a_fp8(uint32_t *output, float *scales,
                                            int rows, int channels) {
  const int valid_rows = NRanks * rows;
  const int padded_rows = (valid_rows + 3) / 4 * 4;
  if (valid_rows != padded_rows) {
    const int shard = channels / NRanks;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < (padded_rows - valid_rows) * shard / 4; i += stride)
      output[valid_rows * shard / 4 + i] = 0;
    for (int i = tid; i < (padded_rows - valid_rows) * (shard / 128);
         i += stride) {
      const int tail = padded_rows - valid_rows;
      scales[(i / tail) * padded_rows + valid_rows + i % tail] = 1.f;
    }
  }
}
