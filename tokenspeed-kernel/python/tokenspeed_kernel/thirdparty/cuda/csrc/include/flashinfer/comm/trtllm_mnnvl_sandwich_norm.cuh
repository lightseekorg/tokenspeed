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
//
// The pre/post ("sandwich") norm boundary as a FusedOp epilogue of the MNNVL
// all-reduce kernels (trtllm_mnnvl_allreduce_fusion.cuh, unchanged):
//
//   a            = RMSNorm(AR(in); post_norm_gamma, rms_eps)
//   residual_out = RN(RN(x_scale * a) + RN(residual_scale * residual_in))
//   norm_out     = RMSNorm(residual_out; rms_gamma, rms_eps)
//
// RN rounds to BF16 (nearest even). The operations are those of the unfused
// chain -- the plain all-reduce, then ops/layernorm/triton.py rmsnorm() on the
// sum, then rmsnorm() with the residual, round_residual_sum_bf16 and the scale
// pair -- with the same BF16 rounding points (a, the two scaled addends, their
// sum, the output) and the same FP32 operations: products and sums of squares
// in FP32 without flush-to-zero, the mean and epsilon folded into
// fma(s, RN(1 / hidden), eps) as ptxas folds the Triton kernel's division,
// rsqrt.approx.ftz, then RN((rstd * x) * gamma). Only the order of the
// sum-of-squares reduction differs. Every FP32 operation is inline PTX: the
// kernel package builds with -use_fast_math, which would flush subnormals.
//
// Reduction (the same order at every token count, cluster shape and
// strategy, so the outputs do not depend on the launch geometry):
//   * thread g of the token holds elements 8g..8g+7 and forms
//     (fma(x0,x0,x2*x2) + fma(x4,x4,x6*x6)) + (fma(x1,x1,x3*x3) + fma(x5,x5,x7*x7));
//   * a butterfly (xor 16..1) sums the 32 threads of each warp; warp w of the
//     token covers groups 32w..32w+31 because every CTA is a whole number of
//     warps (the launcher's partition guarantees it);
//   * every warp of every CTA then sums the hidden / 256 warp sums: lane l adds
//     slots l, l + 32, ... in order, and a butterfly sums the lanes.
//
// Exchange of the warp sums (per norm): lane 0 of each warp stores its sum in
// slot w of its own CTA (st.shared) and of every peer CTA of the cluster
// (st.async, which completes bytes on that CTA's transaction mbarrier). Each
// CTA arms its barriers (one per exchange) for the peers' bytes before
// cta_arrive, whose cluster barrier orders the arming before any peer's
// store; the epilogue's first statement closes the phase cta_arrive leaves
// open. A CTA reads the slots after a CTA barrier (its own warps) and a wait
// on its own mbarrier (the peers) followed by the shared::cluster acquire
// fence. No CTA reads a peer's shared memory and no GPU-scope fence is
// issued; a CTA exits only after each of its barriers completed, so every
// peer write into it landed.
//
// Per-token FP8 (kARSandwichResidualRMSNormTokenFP8): the same epilogue also
// writes norm_out quantized per token, by the rule of
// ops/quantization/triton.py's fp8_quantize_per_token applied to the stored
// BF16 norm_out: amax = max |RN(norm_out)| over the token, codes =
// E4M3_satfinite(clamp(RN(x) * min(448 / amax, 448 * 512), +-448)) into
// quant_out, scale = max(amax / 448, 1 / (448 * 512)) into scale_out[token],
// both divisions IEEE-rounded. The token's amax is a third exchange of warp
// values (a maximum, so its order does not matter). norm_out may be null: only
// the codes and the scales are stored then. The BF16 pattern instantiates the
// epilogue without this stage and with its own two-exchange shared layout.
//
// Registers: the MNNVL kernels launch one cluster per token, so once the
// tokens outnumber the SMs the launch runs in waves of resident CTAs, each
// paying the all-reduce's cross-rank round trip. Only the residual stays in
// registers across the all-reduce poll; the two gammas are copied to shared
// memory with cp.async and the slot sum is not unrolled. The two-shot
// kernels up to 8 ranks then keep to 48 registers on sm_100a and sm_103a, so
// two CTAs of up to 21 warps share an SM, as the plain all-reduce's do.

#pragma once

#include <cooperative_groups.h>

#include "trtllm_mnnvl_allreduce_fusion.cuh"

namespace flashinfer {

namespace trtllm_allreduce_fusion {

namespace sandwich {

static constexpr int kVec = 8;  // BF16 elements per 16-byte access
static constexpr int kWarpGroups = 32;
// hidden / 256 warps per token: a cluster of at most 8 CTAs of 1024 threads.
static constexpr int kMaxWarps = 256;
static constexpr int kMaxClusterCtas = 8;
static constexpr int kMaxBlock = 1024;  // the MNNVL launch's largest CTA

__device__ __forceinline__ float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

__device__ __forceinline__ float rcp_rn(float a) {
  float d;
  asm("rcp.rn.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}

// The per-token FP8 stage's comparisons, without flush-to-zero (the package's
// -use_fast_math would make the CUDA math functions flush subnormals).
__device__ __forceinline__ float abs_f32(float a) {
  float d;
  asm("abs.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}

__device__ __forceinline__ float max_f32(float a, float b) {
  float d;
  asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

__device__ __forceinline__ float min_f32(float a, float b) {
  float d;
  asm("min.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float d;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}

// RN(lo) in the low half, RN(hi) in the high half: one conversion for two
// values, the rounding of cvt.rn.bf16.f32.
__device__ __forceinline__ uint32_t cvt_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }

__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }

// Sum of squares of one thread's eight elements.
__device__ __forceinline__ float group_sumsq(float const (&x)[kVec]) {
  float const ax = fma_rn(x[0], x[0], mul_rn(x[2], x[2]));
  float const ay = fma_rn(x[1], x[1], mul_rn(x[3], x[3]));
  float const bx = fma_rn(x[4], x[4], mul_rn(x[6], x[6]));
  float const by = fma_rn(x[5], x[5], mul_rn(x[7], x[7]));
  return add_rn(add_rn(ax, bx), add_rn(ay, by));
}

// Butterfly over the warp: every lane ends with the same value.
__device__ __forceinline__ float warp_sum(float s) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, m));
  }
  return s;
}

__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// One 16-byte global -> shared copy, completed by cp_async_wait_all() in the
// issuing thread (each thread reads only the slots it copied).
__device__ __forceinline__ void cp_async_16(void* smem, void const* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(smem)), "l"(gmem)
               : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory");
}

// The same shared-memory offset in CTA `cta` of the cluster.
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t cta) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(addr), "r"(cta));
  return d;
}

// One arrival (this one) and the warp sums the peers send.
__device__ __forceinline__ void arm_barrier(uint64_t* bar, uint32_t bytes) {
  uint32_t const b = smem_u32(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes)
               : "memory");
}

// Phase 0 completed: every peer's warp sum landed. The acquire is the
// shared::cluster-restricted fence that pairs with st.async's release.
__device__ __forceinline__ void wait_barrier(uint64_t* bar) {
  uint32_t const b = smem_u32(bar);
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n"
        " mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%1], 0;\n"
        " selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(b)
        : "memory");
  }
  asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
}

// kExchanges cross-warp exchanges: the two sums of squares, and with
// per-token FP8 the token's amax.
template <int kExchanges>
struct Shared {
  uint4 gamma[2][kMaxBlock];               // post_norm_gamma and rms_gamma, by thread
  float warp_sums[kExchanges][kMaxWarps];  // per exchange, indexed by the token's warp
  alignas(8) uint64_t bar[kExchanges];     // transaction barriers of the exchanges
};

// One instance per CTA (function-scope __shared__, like blockReduceSumV2).
template <int kExchanges>
__device__ __forceinline__ Shared<kExchanges>& shared() {
  __shared__ Shared<kExchanges> s;
  return s;
}

// Lane 0 of each warp publishes the warp's sum to slot w of every CTA of the
// cluster: st.shared into its own CTA, st.async into the peers (PTX defines
// st.async for a peer CTA's shared memory only).
template <int kExchanges>
__device__ __forceinline__ void push_warp_sum(float s, int norm) {
  namespace cg = cooperative_groups;
  if ((threadIdx.x & 31) != 0) {
    return;
  }
  cg::cluster_group cluster = cg::this_cluster();
  uint32_t const ctas = cluster.num_blocks();
  uint32_t const w = cluster.thread_rank() >> 5;
  float* slot = &shared<kExchanges>().warp_sums[norm][w];
  *slot = s;
  if (ctas > 1) {
    uint32_t const self = cluster.block_rank();
    uint32_t const dst = smem_u32(slot);
    uint32_t const bar = smem_u32(&shared<kExchanges>().bar[norm]);
#pragma unroll
    for (uint32_t c = 0; c < kMaxClusterCtas; ++c) {
      if (c != self && c < ctas) {
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(
                mapa(dst, c)),
            "f"(s), "r"(mapa(bar, c))
            : "memory");
      }
    }
  }
}

// The token's sum of squares, in every thread: the own warps' slots after a
// CTA barrier, the peers' after the transaction barrier, then the canonical
// lane-strided sum and a butterfly.
template <int kExchanges>
__device__ __forceinline__ float token_sumsq(int norm, int warps) {
  namespace cg = cooperative_groups;
  __syncthreads();
  if (cg::this_cluster().num_blocks() > 1) {
    wait_barrier(&shared<kExchanges>().bar[norm]);
  }
  float const* slots = shared<kExchanges>().warp_sums[norm];
  float s = 0.f;
#pragma unroll 1
  for (int w = threadIdx.x & 31; w < warps; w += kWarpGroups) {
    s = add_rn(s, slots[w]);
  }
  return warp_sum(s);
}

// Per-token FP8: the maximum over the warp, then over the token's warp
// values exchanged as the sums are (exchange `slot`); exact in any order.
__device__ __forceinline__ float warp_max(float m) {
#pragma unroll
  for (int k = 16; k > 0; k >>= 1) {
    m = max_f32(m, __shfl_xor_sync(0xffffffffu, m, k));
  }
  return m;
}

template <int kExchanges>
__device__ __forceinline__ float token_max(int slot, int warps) {
  namespace cg = cooperative_groups;
  __syncthreads();
  if (cg::this_cluster().num_blocks() > 1) {
    wait_barrier(&shared<kExchanges>().bar[slot]);
  }
  float const* slots = shared<kExchanges>().warp_sums[slot];
  float m = 0.f;
#pragma unroll 1
  for (int w = threadIdx.x & 31; w < warps; w += kWarpGroups) {
    m = max_f32(m, slots[w]);
  }
  return warp_max(m);
}

// The per-token FP8 rule of ops/quantization/triton.py (fp8_token_scale,
// fp8_token_multiplier, fp8_quantize_by_multiplier); the constants are its
// FP32 constexprs, 1 / (448 * 512) rounded from double as Triton rounds it.
static constexpr float kE4m3Max = 448.f;
static constexpr float kTokenMinScale = static_cast<float>(1.0 / (448.0 * 512.0));
static constexpr float kTokenMaxMultiplier = 448.f * 512.f;

__device__ __forceinline__ float div_rn(float a, float b) {
  float d;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

// E4M3 codes of lo (low byte) and hi (high byte), saturating to +-448.
__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return static_cast<uint32_t>(d);
}

__device__ __forceinline__ float quantize_by_multiplier(float x, float multiplier) {
  return min_f32(max_f32(mul_rn(x, multiplier), -kE4m3Max), kE4m3Max);
}

}  // namespace sandwich

// The MNNVL kernels construct FusedOp from the launch params, call
// load_upstream_inputs() after the grid dependency wait, and call operator()
// once per thread with the reduced 8-element vector of its token.
// kTokenFp8: kARSandwichResidualRMSNormTokenFP8 (the per-token FP8 stage).
template <bool kTokenFp8>
class SandwichNormFusedOp {
  using T = __nv_bfloat16;
  static constexpr int VEC_SIZE = sandwich::kVec;
  static constexpr int kExchanges = kTokenFp8 ? 3 : 2;
  static_assert(VEC_SIZE == details::kBytesPerAccess / sizeof(T), "one 16 B access per thread");

 public:
  __device__ __forceinline__ SandwichNormFusedOp(AllReduceFusionParams<T> const& params,
                                                 int access_id, int access_id_in_token)
      : m_params(params), m_access_id(access_id), m_access_id_in_token(access_id_in_token) {}

  __device__ __forceinline__ void load_upstream_inputs() {
    sandwich::cp_async_16(&sandwich::shared<kExchanges>().gamma[0][threadIdx.x],
                          reinterpret_cast<uint4 const*>(m_params.post_norm_gamma) +
                              m_access_id_in_token);
    sandwich::cp_async_16(
        &sandwich::shared<kExchanges>().gamma[1][threadIdx.x],
        reinterpret_cast<uint4 const*>(m_params.rms_gamma) + m_access_id_in_token);
    m_residual.load(reinterpret_cast<T const*>(m_params.residual_in) + m_access_id * VEC_SIZE);
    // Arm the exchanges before cta_arrive, whose cluster barrier (release,
    // then the waits in cta_arrive and at the start of operator()) orders the
    // arming before any peer's st.async.
    namespace cg = cooperative_groups;
    uint32_t const ctas = cg::this_cluster().num_blocks();
    if (threadIdx.x == 0 && ctas > 1) {
      uint32_t const warps = static_cast<uint32_t>(m_params.hidden_dim) >> 8;
      uint32_t const bytes = (warps - (blockDim.x >> 5)) * static_cast<uint32_t>(sizeof(float));
      sandwich::arm_barrier(&sandwich::shared<kExchanges>().bar[0], bytes);
      sandwich::arm_barrier(&sandwich::shared<kExchanges>().bar[1], bytes);
      if constexpr (kTokenFp8) {
        sandwich::arm_barrier(&sandwich::shared<kExchanges>().bar[2], bytes);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    // A launch that releases its PDL dependents right after this call must
    // not leave the copies in flight.
    if (!m_params.trigger_completion_at_end) {
      sandwich::cp_async_wait_all();
    }
  }

  __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int token_id,
                                             bool /*skip_residual_add*/,
                                             bool /*skip_residual_store*/,
                                             bool /*skip_partial_store*/,
                                             int /*residual_access_id*/,
                                             int /*partial_out_access_id*/) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    // MnnvlLamportFlags::cta_arrive() arrived every thread on the cluster
    // barrier; only block rank 0, threads 0..31 waited. The rest wait here:
    // after it every CTA's exchange barriers are armed. (A cluster of one
    // exchanges through CTA barriers and leaves the phase open, as the
    // vendored patterns do.)
    if (cluster.num_blocks() > 1 && !(cluster.block_rank() == 0 && threadIdx.x < 32)) {
      cluster.barrier_wait();
    }
    sandwich::cp_async_wait_all();
    int const warps = m_params.hidden_dim >> 8;
    float const inv_hidden = sandwich::rcp_rn(static_cast<float>(m_params.hidden_dim));

    // The norm of the all-reduced sum, widened from its packed words.
    float x[VEC_SIZE];
    uint32_t const* val2 = reinterpret_cast<uint32_t const*>(&val);
#pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
      x[2 * i] = sandwich::bf16_lo(val2[i]);
      x[2 * i + 1] = sandwich::bf16_hi(val2[i]);
    }
    sandwich::push_warp_sum<kExchanges>(sandwich::warp_sum(sandwich::group_sumsq(x)), 0);
    // While the sums travel: rp = RN(residual_scale * residual), in pairs.
    uint32_t rp[VEC_SIZE / 2];
    uint32_t const* res2 = reinterpret_cast<uint32_t const*>(&m_residual);
#pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
      rp[i] = sandwich::cvt_bf16x2(
          sandwich::mul_rn(sandwich::bf16_lo(res2[i]), m_params.residual_scale),
          sandwich::mul_rn(sandwich::bf16_hi(res2[i]), m_params.residual_scale));
    }
    float const rstd1 = sandwich::rsqrt_approx_ftz(
        sandwich::fma_rn(sandwich::token_sumsq<kExchanges>(0, warps), inv_hidden,
                         m_params.rms_eps));

    // a = RN((rstd1 * x) * post_gamma), xp = RN(x_scale * a), the sum
    // RN(xp + rp): the residual stream the next norm reads.
    vec_t<T, VEC_SIZE> sum;
    uint32_t* sum2 = reinterpret_cast<uint32_t*>(&sum);
    uint4 const post_gamma = sandwich::shared<kExchanges>().gamma[0][threadIdx.x];
    uint32_t const* pg2 = reinterpret_cast<uint32_t const*>(&post_gamma);
#pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
      uint32_t const a = sandwich::cvt_bf16x2(
          sandwich::mul_rn(sandwich::mul_rn(rstd1, x[2 * i]), sandwich::bf16_lo(pg2[i])),
          sandwich::mul_rn(sandwich::mul_rn(rstd1, x[2 * i + 1]), sandwich::bf16_hi(pg2[i])));
      uint32_t const xp =
          sandwich::cvt_bf16x2(sandwich::mul_rn(sandwich::bf16_lo(a), m_params.x_scale),
                               sandwich::mul_rn(sandwich::bf16_hi(a), m_params.x_scale));
      // One FP32 add of two BF16 values then one BF16 rounding equals a BF16
      // add (the double rounding is innocuous: 24 >= 2 * 8 + 2).
      sum2[i] = sandwich::cvt_bf16x2(
          sandwich::add_rn(sandwich::bf16_lo(xp), sandwich::bf16_lo(rp[i])),
          sandwich::add_rn(sandwich::bf16_hi(xp), sandwich::bf16_hi(rp[i])));
      x[2 * i] = sandwich::bf16_lo(sum2[i]);
      x[2 * i + 1] = sandwich::bf16_hi(sum2[i]);
    }

    // The next norm of the new residual stream; residual_out is stored while
    // its sums travel.
    sandwich::push_warp_sum<kExchanges>(sandwich::warp_sum(sandwich::group_sumsq(x)), 1);
    sum.store(reinterpret_cast<T*>(m_params.residual_out) + m_access_id * VEC_SIZE);
    float const rstd2 = sandwich::rsqrt_approx_ftz(
        sandwich::fma_rn(sandwich::token_sumsq<kExchanges>(1, warps), inv_hidden,
                         m_params.rms_eps));
    vec_t<T, VEC_SIZE> h;
    uint32_t* h2 = reinterpret_cast<uint32_t*>(&h);
    uint4 const gamma = sandwich::shared<kExchanges>().gamma[1][threadIdx.x];
    uint32_t const* g2 = reinterpret_cast<uint32_t const*>(&gamma);
#pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
      h2[i] = sandwich::cvt_bf16x2(
          sandwich::mul_rn(sandwich::mul_rn(rstd2, x[2 * i]), sandwich::bf16_lo(g2[i])),
          sandwich::mul_rn(sandwich::mul_rn(rstd2, x[2 * i + 1]), sandwich::bf16_hi(g2[i])));
    }
    if constexpr (kTokenFp8) {
      // The token's amax of the BF16 output, exchanged while norm_out (if
      // asked for) is stored; then this thread's eight codes and, from the
      // token's first thread, its scale.
      float hv[VEC_SIZE];
      float m = 0.f;
#pragma unroll
      for (int i = 0; i < VEC_SIZE / 2; ++i) {
        hv[2 * i] = sandwich::bf16_lo(h2[i]);
        hv[2 * i + 1] = sandwich::bf16_hi(h2[i]);
        m = sandwich::max_f32(m, sandwich::max_f32(sandwich::abs_f32(hv[2 * i]),
                                                   sandwich::abs_f32(hv[2 * i + 1])));
      }
      sandwich::push_warp_sum<kExchanges>(sandwich::warp_max(m), 2);
      if (m_params.norm_out != nullptr) {
        h.store(reinterpret_cast<T*>(m_params.norm_out) + m_access_id * VEC_SIZE);
      }
      float const amax = sandwich::token_max<kExchanges>(2, warps);
      float const multiplier = sandwich::min_f32(sandwich::div_rn(sandwich::kE4m3Max, amax),
                                                 sandwich::kTokenMaxMultiplier);
      uint32_t c[VEC_SIZE / 2];
#pragma unroll
      for (int i = 0; i < VEC_SIZE / 2; ++i) {
        c[i] = sandwich::cvt_e4m3x2(sandwich::quantize_by_multiplier(hv[2 * i], multiplier),
                                    sandwich::quantize_by_multiplier(hv[2 * i + 1], multiplier));
      }
      uint2 const codes = make_uint2(c[0] | (c[1] << 16), c[2] | (c[3] << 16));
      reinterpret_cast<uint2*>(m_params.quant_out)[m_access_id] = codes;
      if (m_access_id_in_token == 0) {
        reinterpret_cast<float*>(m_params.scale_out)[token_id] =
            sandwich::max_f32(sandwich::div_rn(amax, sandwich::kE4m3Max), sandwich::kTokenMinScale);
      }
    } else {
      h.store(reinterpret_cast<T*>(m_params.norm_out) + m_access_id * VEC_SIZE);
    }
    // No closing cluster barrier: peers only write into this CTA (st.async),
    // and each of its barriers completed above, so every such write landed.
  }

 private:
  AllReduceFusionParams<T> const& m_params;
  int m_access_id;
  int m_access_id_in_token;
  vec_t<T, VEC_SIZE> m_residual;
};

// The MNNVL kernels name FusedOp<Pattern, T>: route the sandwich patterns here.
template <>
class FusedOp<AllReduceFusionPattern::kARSandwichResidualRMSNorm, __nv_bfloat16>
    : public SandwichNormFusedOp<false> {
 public:
  using SandwichNormFusedOp<false>::SandwichNormFusedOp;
};

template <>
class FusedOp<AllReduceFusionPattern::kARSandwichResidualRMSNormTokenFP8, __nv_bfloat16>
    : public SandwichNormFusedOp<true> {
 public:
  using SandwichNormFusedOp<true>::SandwichNormFusedOp;
};

}  // namespace trtllm_allreduce_fusion

}  // namespace flashinfer
