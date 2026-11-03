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

// Copies a tile table with TMA bulk copies through a shared-memory ring: one
// thread per CTA keeps STAGES chunks in flight, so few SMs saturate fabric
// reads. Tiles are int64 [n, 3] = (destination, source, length) in 8-byte
// words relative to `base`; CTAs claim tiles from `counter`.
#include "tvm_ffi_utils.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kStages = 6;
constexpr int kChunk = 32 << 10;
// CTAs launch in clusters of this size so they hold whole 8-SM slots, the
// capacity tokenspeed_kernel.sm_reservation subtracts for persistent grids.
constexpr int kClusterSize = 8;

__device__ __forceinline__ uint32_t smem_address(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void wait_parity(uint64_t *barrier,
                                            uint32_t parity) {
  asm volatile("{\n"
               ".reg .pred p;\n"
               "WAIT_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra WAIT_%=;\n"
               "}\n" ::"r"(smem_address(barrier)),
               "r"(parity)
               : "memory");
}

__global__ void __launch_bounds__(32)
    peer_fetch_kernel(uint64_t base, const int64_t *__restrict__ tiles,
                      int *counter, int num_tiles) {
  extern __shared__ __align__(128) uint8_t ring[];
  __shared__ __align__(8) uint64_t full[kStages];
  if (threadIdx.x != 0) {
    return;
  }
  for (int s = 0; s < kStages; ++s) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
        smem_address(&full[s])));
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  uint32_t parity = 0; // bit s: phase to wait for on stage s
  for (int g = atomicAdd(counter, 1); g < num_tiles; g = atomicAdd(counter, 1)) {
    const uint64_t dst = base + tiles[3 * g] * 8;
    const uint64_t src = base + tiles[3 * g + 1] * 8;
    const int64_t bytes = tiles[3 * g + 2] * 8;
    const int64_t chunks = (bytes + kChunk - 1) / kChunk;
    // Chunk i is loaded at step i and stored at step i + kStages - 1.
    for (int64_t i = 0; i < chunks + kStages - 1; ++i) {
      const int64_t j = i - (kStages - 1);
      if (j >= 0) {
        const int s = static_cast<int>(j % kStages);
        wait_parity(&full[s], (parity >> s) & 1u);
        parity ^= 1u << s;
        const int64_t left = bytes - j * kChunk;
        const uint32_t size = static_cast<uint32_t>(left < kChunk ? left : kChunk);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
                         dst + j * kChunk),
                     "r"(smem_address(ring + s * kChunk)), "r"(size)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
      if (i < chunks) {
        const int s = static_cast<int>(i % kStages);
        if (i >= kStages) {
          // Stage s last held chunk i - kStages, stored one group before the
          // newest; wait until that store has read shared memory.
          asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
        }
        const int64_t left = bytes - i * kChunk;
        const uint32_t size = static_cast<uint32_t>(left < kChunk ? left : kChunk);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                         smem_address(&full[s])),
                     "r"(size)
                     : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1], %2, [%3];" ::"r"(smem_address(ring + s * kChunk)),
                     "l"(src + i * kChunk), "r"(size), "r"(smem_address(&full[s]))
                     : "memory");
      }
    }
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

} // namespace

void fetch(TensorView tiles, TensorView counter, int64_t base, int64_t ctas) {
  TVM_FFI_ICHECK(tiles.dtype().code == kDLInt && tiles.dtype().bits == 64);
  TVM_FFI_ICHECK(counter.dtype().code == kDLInt && counter.dtype().bits == 32);
  TVM_FFI_ICHECK(ctas > 0 && ctas % kClusterSize == 0)
      << "peer fetch CTAs must be a positive multiple of " << kClusterSize;
  const int num_tiles = static_cast<int>(tiles.numel() / 3);
  auto stream = get_stream(tiles.device());
  int *claimed = static_cast<int *>(counter.data_ptr());
  TVM_FFI_ICHECK(cudaMemsetAsync(claimed, 0, sizeof(int), stream) == cudaSuccess);
  constexpr int smem = kStages * kChunk;
  TVM_FFI_ICHECK(cudaFuncSetAttribute(
                     peer_fetch_kernel,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(ctas));
  config.blockDim = dim3(32);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute cluster[1];
  cluster[0].id = cudaLaunchAttributeClusterDimension;
  cluster[0].val.clusterDim.x = kClusterSize;
  cluster[0].val.clusterDim.y = 1;
  cluster[0].val.clusterDim.z = 1;
  config.attrs = cluster;
  config.numAttrs = 1;
  TVM_FFI_ICHECK(cudaLaunchKernelEx(&config, peer_fetch_kernel,
                                    static_cast<uint64_t>(base),
                                    static_cast<const int64_t *>(tiles.data_ptr()),
                                    claimed, num_tiles) == cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fetch, fetch);
