/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from SGLang's router front: retain BF16 GEMM and NVFP4 quantization,
// remove routing/shared-expert modes, and expose a checked TVM FFI launcher.
#include "tvm_ffi_utils.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#define DEVICE __forceinline__ __device__
namespace {
using bf16_t = __nv_bfloat16;
constexpr uint32_t kHidden = ROUTER_HIDDEN;
constexpr uint32_t kExperts = ROUTER_EXPERTS;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kExpertsPerCta = 8;
constexpr uint32_t kSfVec = 16;
constexpr uint32_t kBlocksPerRow = kHidden / kSfVec;
constexpr uint32_t kKSplit = 4;
constexpr uint32_t kSplitK = kHidden / kKSplit;
constexpr uint32_t kChainSteps = kSplitK / 16;
constexpr uint32_t kRowBytes = kHidden * 2 + 16;
static_assert(kHidden % 64 == 0 && kHidden >= 256);
static_assert(kExperts % 8 == 0 && kExperts >= 64);

struct Params {
  const bf16_t *x;
  const bf16_t *weight;
  const float *global_scale;
  bf16_t *logits;
  uint8_t *x_fp4;
  uint8_t *x_sf;
  uint32_t num_tokens;
};
template <uint32_t kMaxM> struct GemvSmem {
  static constexpr uint32_t kW = 0;
  static constexpr uint32_t kX = kExpertsPerCta * kRowBytes;
  static constexpr uint32_t kPart = kX + kMaxM * kRowBytes;
  static constexpr uint32_t kBarW = kPart + kKSplit * 16 * kExpertsPerCta * 4;
  static constexpr uint32_t kBarX = kBarW + 8;
  static constexpr uint32_t kBytes = kBarX + 8;
};
DEVICE float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVICE float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }

DEVICE float rcp_approx_ftz(float a) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a));
  return y;
}

DEVICE uint32_t cvt_e4m3(float a) {
  uint32_t out;
  asm("{\n"
      ".reg .b16 pair;\n"
      ".reg .f32 zero;\n"
      "mov.f32 zero, 0f00000000;\n"
      "cvt.rn.satfinite.e4m3x2.f32 pair, zero, %1;\n"
      "cvt.u32.u16 %0, pair;\n"
      "}"
      : "=r"(out)
      : "f"(a));
  return out & 0xFFu;
}

DEVICE float e4m3_to_f32(uint32_t v) {
  float out;
  asm("{\n"
      ".reg .b16 pair;\n"
      ".reg .b32 h2;\n"
      ".reg .b16 lo, hi;\n"
      "cvt.u16.u32 pair, %1;\n"
      "cvt.rn.f16x2.e4m3x2 h2, pair;\n"
      "mov.b32 {lo, hi}, h2;\n"
      "cvt.f32.f16 %0, lo;\n"
      "}"
      : "=f"(out)
      : "r"(v));
  return out;
}

DEVICE uint32_t cvt_e2m1x8(const float (&v)[8]) {
  uint32_t out;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]),
        "f"(v[6]), "f"(v[7]));
  return out;
}

DEVICE uint32_t bf16x2_abs_max(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("{\n"
      ".reg .b32 ta, tb;\n"
      "and.b32 ta, %1, 0x7FFF7FFF;\n"
      "and.b32 tb, %2, 0x7FFF7FFF;\n"
      "max.bf16x2 %0, ta, tb;\n"
      "}"
      : "=r"(out)
      : "r"(a), "r"(b));
  return out;
}

DEVICE uint32_t bf16x2_max(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}

// One 16-element NVFP4 block: e4m3 scale byte and 8 packed bytes (element 2i in
// the low nibble of byte i).
DEVICE void quantize_block(const uint4 &a, const uint4 &b, float gs,
                           uint32_t &sf, uint2 &packed) {
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  const uint32_t m01 = bf16x2_abs_max(w[0], w[1]);
  const uint32_t m23 = bf16x2_abs_max(w[2], w[3]);
  const uint32_t m45 = bf16x2_abs_max(w[4], w[5]);
  const uint32_t m67 = bf16x2_abs_max(w[6], w[7]);
  const uint32_t m = bf16x2_max(bf16x2_max(m01, m23), bf16x2_max(m45, m67));
  const float amax = fmaxf(bf16_lo(m), bf16_hi(m));
  const float scale = __fmul_rn(gs, __fmul_rn(amax, rcp_approx_ftz(6.0f)));
  sf = cvt_e4m3(scale);
  const float sf_f = e4m3_to_f32(sf);
  const float out_scale =
      (sf_f == 0.0f) ? 0.0f
                     : rcp_approx_ftz(__fmul_rn(sf_f, rcp_approx_ftz(gs)));
  float lo[8], hi[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lo[2 * i] = __fmul_rn(bf16_lo(w[i]), out_scale);
    lo[2 * i + 1] = __fmul_rn(bf16_hi(w[i]), out_scale);
    hi[2 * i] = __fmul_rn(bf16_lo(w[4 + i]), out_scale);
    hi[2 * i + 1] = __fmul_rn(bf16_hi(w[4 + i]), out_scale);
  }
  packed.x = cvt_e2m1x8(lo);
  packed.y = cvt_e2m1x8(hi);
}

// D += A * B for one m16n8k16 bf16 tile, fp32 accumulators.
DEVICE void mma_bf16_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                           uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, "
               "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

DEVICE uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVICE void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

DEVICE void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

// One K split of the GEMV for this CTA's experts; writes the fp32 partials to
// s_part.
template <uint32_t kMaxM>
DEVICE void gemv_chain(uint32_t half, uint32_t M, const uint8_t *smem,
                       float *s_part) {
  using L = GemvSmem<kMaxM>;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t j = lane >> 3, r = lane & 7;
  // ldmatrix.x4 for A: matrix j covers tokens (j & 1) * 8 + r and K offset (j
  // >> 1) * 8.
  const uint32_t tok = (j & 1) * 8 + r;
  const uint32_t a_row = tok < M ? tok : 0;
  const uint32_t a_addr = smem_addr(smem + L::kX + a_row * kRowBytes +
                                    (half * kSplitK + (j >> 1) * 8) * 2);
  // ldmatrix.x2 for B: matrix (j & 1) covers experts r and K offset (j & 1)
  // * 8.
  const uint32_t b_row = r < kExpertsPerCta ? r : 0;
  const uint32_t b_addr = smem_addr(smem + L::kW + b_row * kRowBytes +
                                    (half * kSplitK + (j & 1) * 8) * 2);
  // Fragments are loaded kDepth steps ahead so shared-memory latency stays off
  // the MMA chain.
  constexpr uint32_t kDepth = 4;
  uint32_t a[kDepth][4], b[kDepth][2];
#pragma unroll
  for (uint32_t i = 0; i < kDepth; ++i) {
    ldmatrix_x4(a[i], a_addr + i * 32);
    ldmatrix_x2(b[i], b_addr + i * 32);
  }
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (uint32_t step = 0; step < kChainSteps; ++step) {
    const uint32_t slot = step % kDepth;
    mma_bf16_16816(d, a[slot][0], a[slot][1], a[slot][2], a[slot][3],
                   b[slot][0], b[slot][1]);
    if (step + kDepth < kChainSteps) {
      ldmatrix_x4(a[slot], a_addr + (step + kDepth) * 32);
      ldmatrix_x2(b[slot], b_addr + (step + kDepth) * 32);
    }
  }
  // C fragment: d0/d1 = (token g, experts 2c, 2c+1), d2/d3 = token g + 8.
  const uint32_t g = lane >> 2, c = lane & 3;
  if (c < kExpertsPerCta / 2) {
    float *part = s_part + half * 16 * kExpertsPerCta;
    part[g * kExpertsPerCta + 2 * c] = d[0];
    part[g * kExpertsPerCta + 2 * c + 1] = d[1];
    if constexpr (kMaxM > 8) {
      part[(g + 8) * kExpertsPerCta + 2 * c] = d[2];
      part[(g + 8) * kExpertsPerCta + 2 * c + 1] = d[3];
    }
  }
}

DEVICE void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count)
               : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

DEVICE void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
      "r"(bytes)
      : "memory");
}

DEVICE void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n"
               ".reg .pred done;\n"
               "WAIT_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
               "@!done bra WAIT_%=;\n"
               "}" ::"r"(bar),
               "r"(parity)
               : "memory");
}

DEVICE void bulk_load(uint32_t dst, const void *src, uint32_t bytes,
                      uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

// Four warps compute K partials; the other four quantize one input row in
// CTA b < M. Reduction order and the BF16 logits boundary stay explicit.
template <uint32_t kMaxM>
__global__ __launch_bounds__(kThreads, 1) void cuda_router_gemm_nvfp4(
    const __grid_constant__ Params p) {
  using L = GemvSmem<kMaxM>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t M = p.num_tokens;
  const uint32_t expert0 = blockIdx.x * kExpertsPerCta;
  const uint32_t warp = threadIdx.x / 32;
  float *s_part = reinterpret_cast<float *>(smem + L::kPart);
  const uint32_t bar_w = smem_addr(smem + L::kBarW);
  const uint32_t bar_x = smem_addr(smem + L::kBarX);
  if (threadIdx.x == 0) {
    mbar_init(bar_w, 1);
    mbar_init(bar_x, 1);
    mbar_expect_tx(bar_w, kExpertsPerCta * kHidden * 2);
#pragma unroll
    for (uint32_t e = 0; e < kExpertsPerCta; ++e) {
      bulk_load(smem_addr(smem + L::kW + e * kRowBytes),
                p.weight + static_cast<uint64_t>(expert0 + e) * kHidden,
                kHidden * 2, bar_w);
    }
  }
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) {
    mbar_expect_tx(bar_x, M * kHidden * 2);
#pragma unroll
    for (uint32_t row = 0; row < kMaxM; ++row) {
      if (row < M) {
        bulk_load(smem_addr(smem + L::kX + row * kRowBytes),
                  p.x + static_cast<uint64_t>(row) * kHidden, kHidden * 2,
                  bar_x);
      }
    }
  }
  if (warp < kKSplit) {
    mbar_wait(bar_w, 0);
    mbar_wait(bar_x, 0);
    gemv_chain<kMaxM>(warp, M, smem, s_part);
  } else if (blockIdx.x < M) {
    mbar_wait(bar_x, 0);
    const uint8_t *row = smem + L::kX + blockIdx.x * kRowBytes;
    for (uint32_t blk = threadIdx.x - 32 * kKSplit; blk < kBlocksPerRow;
         blk += kThreads - 32 * kKSplit) {
      const uint4 *src =
          reinterpret_cast<const uint4 *>(row + blk * kSfVec * 2);
      uint32_t sf;
      uint2 packed;
      quantize_block(src[0], src[1], p.global_scale[0], sf, packed);
      *reinterpret_cast<uint2 *>(
          p.x_fp4 + static_cast<uint64_t>(blockIdx.x) * (kHidden / 2) +
          blk * 8) = packed;
      p.x_sf[static_cast<uint64_t>(blockIdx.x) * kBlocksPerRow + blk] =
          static_cast<uint8_t>(sf);
    }
  }
  __syncthreads();
  if (threadIdx.x < M * kExpertsPerCta) {
    const uint32_t m = threadIdx.x / kExpertsPerCta;
    const uint32_t e = threadIdx.x % kExpertsPerCta;
    float acc = s_part[m * kExpertsPerCta + e];
#pragma unroll
    for (uint32_t k = 1; k < kKSplit; ++k)
      acc = __fadd_rn(acc, s_part[(k * 16 + m) * kExpertsPerCta + e]);
    p.logits[static_cast<uint64_t>(m) * kExperts + expert0 + e] =
        __float2bfloat16_rn(acc);
  }
}

template <uint32_t Capacity>
cudaError_t launch(const Params &params, cudaStream_t stream) {
  auto kernel = cuda_router_gemm_nvfp4<Capacity>;
  constexpr auto shared = GemvSmem<Capacity>::kBytes;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess)
    return status;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = true;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kExperts / kExpertsPerCta);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, params);
}

void Run(TensorView x, TensorView weight, TensorView scale, TensorView logits,
         TensorView packed, TensorView scales) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(weight, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(scale, dl_float32);
  CHECK_INPUT_AND_TYPE(logits, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(scales, dl_uint8);
  CHECK_DEVICE(x, weight);
  CHECK_DEVICE(x, scale);
  CHECK_DEVICE(x, logits);
  CHECK_DEVICE(x, packed);
  CHECK_DEVICE(x, scales);
  TVM_FFI_CHECK(x.ndim() == 2, ValueError) << "Router input must be 2D";
  const int64_t rows = x.size(0);
  TVM_FFI_CHECK(x.ndim() == 2 && rows >= 1 && rows <= 8 &&
                    x.size(1) == kHidden && weight.ndim() == 2 &&
                    weight.size(0) == kExperts && weight.size(1) == kHidden &&
                    scale.numel() == 1,
                ValueError)
      << "Invalid router inputs";
  TVM_FFI_CHECK(logits.ndim() == 2 && logits.size(0) == rows &&
                    logits.size(1) == kExperts && packed.ndim() == 2 &&
                    packed.size(0) == rows && packed.size(1) == kHidden / 2 &&
                    scales.ndim() == 2 && scales.size(0) == rows &&
                    scales.size(1) == kBlocksPerRow,
                ValueError)
      << "Invalid router outputs";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
                ValueError)
      << "Router bulk loads require 16-byte alignment";
  const auto device = x.device();
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  const auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  Params params{static_cast<const bf16_t *>(x.data_ptr()),
                static_cast<const bf16_t *>(weight.data_ptr()),
                static_cast<const float *>(scale.data_ptr()),
                static_cast<bf16_t *>(logits.data_ptr()),
                static_cast<uint8_t *>(packed.data_ptr()),
                static_cast<uint8_t *>(scales.data_ptr()),
                static_cast<uint32_t>(rows)};
  const cudaError_t status =
      rows == 1 ? launch<1>(params, stream) : launch<8>(params, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << cudaGetErrorString(status);
}
} // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
