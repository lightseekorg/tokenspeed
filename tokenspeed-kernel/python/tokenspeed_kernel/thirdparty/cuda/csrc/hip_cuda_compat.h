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

// Force-included (`-include`) when a hipified CUDA kernel is built for gfx9:
// the CUDA intrinsics HIP lacks or gives different semantics. The build
// renames each CUDA call `__<op>(` in `HIP_RENAMED_INTRINSICS` (setup.py) to
// `ts_hip_<op>(`, so HIP's own declarations stay untouched.
//
// Warp collectives: the CUDA sources assume 32-lane warps (`lane = tid & 31`,
// full mask 0xffffffff); a gfx9 wavefront is 64 lanes, so it runs two such
// warps in lockstep. Each collective operates on the caller's 32-lane half:
// shuffles use width 32 and a ballot returns the half's 32 bits. HIP's own
// `*_sync` forms take 64-bit masks and span the whole wavefront. The CUDA
// callers always pass the full 32-lane mask, which is ignored. `__syncwarp()`
// keeps HIP's definition (a wavefront barrier, which also orders the other
// half).
#pragma once

#include <hip/hip_runtime.h>

constexpr int ts_hip_warp_width = 32;

template <typename T>
__device__ __forceinline__ T ts_hip_shfl_sync(unsigned, T var, int src_lane) {
  return __shfl(var, src_lane, ts_hip_warp_width);
}

template <typename T>
__device__ __forceinline__ T ts_hip_shfl_xor_sync(unsigned, T var, int lane_mask) {
  return __shfl_xor(var, lane_mask, ts_hip_warp_width);
}

template <typename T>
__device__ __forceinline__ T ts_hip_shfl_up_sync(unsigned, T var, unsigned delta) {
  return __shfl_up(var, delta, ts_hip_warp_width);
}

__device__ __forceinline__ unsigned ts_hip_ballot_sync(unsigned, int predicate) {
  return static_cast<unsigned>(__ballot(predicate) >> (__lane_id() & ts_hip_warp_width));
}

template <typename T>
__device__ __forceinline__ T ts_hip_reduce_max_sync(unsigned, T v) {
  for (int offset = ts_hip_warp_width / 2; offset > 0; offset >>= 1) {
    const T o = __shfl_xor(v, offset, ts_hip_warp_width);
    v = o > v ? o : v;
  }
  return v;
}

template <typename T>
__device__ __forceinline__ T ts_hip_reduce_min_sync(unsigned, T v) {
  for (int offset = ts_hip_warp_width / 2; offset > 0; offset >>= 1) {
    const T o = __shfl_xor(v, offset, ts_hip_warp_width);
    v = o < v ? o : v;
  }
  return v;
}

// CUDA `__ldcg` (cache at L2, not L1): an agent-scope relaxed load, which
// skips the non-coherent vector L1.
template <typename T>
__device__ __forceinline__ T ts_hip_ldcg(const T* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
