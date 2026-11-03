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

// EAGLE-4 low-rank candidate head: shard top-k and post-all-gather tail, bit for bit as torch.

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

namespace cg = cooperative_groups;

namespace {

constexpr int kCluster = 8;

// torch.topk's order-preserving key (TopKTypeConfig::convert); NaN maps to the maximum.
__device__ __forceinline__ uint32_t radix_key(__nv_bfloat16 v) {
  const uint32_t b = __bfloat16_as_ushort(v);
  const uint32_t key = (b & 0x8000u) ? (b ^ 0xFFFFu) : (b ^ 0x8000u);
  const float f = __bfloat162float(v);
  return (f != f) ? 0xFFFFu : key;
}
__device__ __forceinline__ uint32_t radix_key(float v) {
  const uint32_t b = __float_as_uint(v);
  const uint32_t key = (b >> 31) ? ~b : (b | 0x80000000u);
  return (v != v) ? 0xFFFFFFFFu : key;
}

__device__ __forceinline__ float radix_key_to_float(uint32_t key) {
  const uint32_t b = (key >> 31) ? (key ^ 0x80000000u) : ~key;
  return __uint_as_float(b);
}

// Shard top-k in torch.topk(sorted=False) order: one cluster of kCluster CTAs per row.

// Exclusive block scan of packed 16-bit pairs; call once (s_warp is not reusable afterwards).
template <int THREADS>
__device__ __forceinline__ uint32_t block_exclusive_scan_once(uint32_t v, uint32_t* s_warp) {
  constexpr int WARPS = THREADS / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t incl = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const uint32_t u = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += u;
  }
  if (lane == 31) s_warp[warp] = incl;
  __syncthreads();
  uint32_t before = 0;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) before += (w < warp) ? s_warp[w] : 0u;
  return before + incl - v;
}

// out: bin b of the target-th largest key, keys above b, and for CTAs < rank: keys above b, in b.
__device__ __forceinline__ void find_bin_warp(const int (*hist)[256], int rank, int target, int* out) {
  const int lane = threadIdx.x & 31;
  int gv[8], pv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) gv[i] = pv[i] = 0;
#pragma unroll
  for (int c = 0; c < kCluster; ++c) {
    const int4 lo = reinterpret_cast<const int4*>(&hist[c][lane * 8])[0];
    const int4 hi = reinterpret_cast<const int4*>(&hist[c][lane * 8])[1];
    const int v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      gv[i] += v[i];
      pv[i] += c < rank ? v[i] : 0;
    }
  }
  int tg = 0, tp = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tg += gv[i];
    tp += pv[i];
  }
  // sums over the bins of the higher lanes
  int sg = tg, sp = tp;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int ug = __shfl_down_sync(0xffffffffu, sg, off);
    const int up = __shfl_down_sync(0xffffffffu, sp, off);
    if (lane + off < 32) {
      sg += ug;
      sp += up;
    }
  }
  int above_g = sg - tg, above_p = sp - tp;  // bins > lane * 8 + 7
#pragma unroll
  for (int i = 7; i >= 0; --i) {
    if (above_g < target && above_g + gv[i] >= target) {
      out[0] = lane * 8 + i;
      out[1] = above_g;
      out[2] = above_p;
      out[3] = pv[i];
    }
    above_g += gv[i];
    above_p += pv[i];
  }
}

template <int THREADS, int E>
__global__ void __cluster_dims__(kCluster, 1, 1) __launch_bounds__(THREADS)
    eagle4_shard_topk_cluster_kernel(
        const __nv_bfloat16* __restrict__ scores,
        int64_t stride_sm,
        int N,
        int K,
        int64_t* __restrict__ global_ids,
        int64_t stride_gm,
        int64_t vocab_start) {
  constexpr int WARPS = THREADS / 32;
  constexpr int HISTS = WARPS < 16 ? WARPS : 16;
  static_assert(THREADS >= 256 && THREADS % 32 == 0, "needs >= 256 threads");
  __shared__ int s_wh[HISTS][256];                        // per-warp(-pair) histograms
  __shared__ __align__(16) int s_hist[2][kCluster][256];  // [pass][CTA]: pushed by every CTA
  __shared__ uint32_t s_warp[WARPS];
  __shared__ int s_val[8];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int m = blockIdx.y;
  const int t = threadIdx.x, warp = t >> 5;
  const __nv_bfloat16* row = scores + static_cast<int64_t>(m) * stride_sm;
  const int pos0 = (rank * THREADS + t) * E;
  // Arrive now; wait before the first DSMEM push, which needs every CTA of the cluster running.
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");

  int keys[E];  // radix keys, -1 = no element
#pragma unroll
  for (int i = 0; i < E; ++i) keys[i] = -1;
  const bool full = pos0 + E <= N;
  static_assert(E % 8 == 0, "bf16 scores in whole 16-byte vectors");
  if (full && (reinterpret_cast<uintptr_t>(row + pos0) & 15u) == 0) {
#pragma unroll
    for (int v = 0; v < E / 8; ++v) {
      const uint4 q = *reinterpret_cast<const uint4*>(row + pos0 + v * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
      for (int i = 0; i < 8; ++i) keys[v * 8 + i] = static_cast<int>(radix_key(h[i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < E; ++i)
      if (pos0 + i < N) keys[i] = static_cast<int>(radix_key(row[pos0 + i]));
  }
  for (int i = t; i < HISTS * 256; i += THREADS) (&s_wh[0][0])[i] = 0;
  __syncthreads();

  // Pass 1: histogram of the high byte, pushed into every CTA of the cluster.
#pragma unroll
  for (int i = 0; i < E; ++i)
    if (keys[i] >= 0) atomicAdd(&s_wh[warp % HISTS][keys[i] >> 8], 1);
  __syncthreads();
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
  if (t < 256) {
    int h = 0;
#pragma unroll
    for (int w = 0; w < HISTS; ++w) {
      h += s_wh[w][t];
      s_wh[w][t] = 0;
    }
#pragma unroll
    for (int c = 0; c < kCluster; ++c) cluster.map_shared_rank(&s_hist[0][rank][t], c)[0] = h;
  }
  cluster.sync();
  if (warp == 0) find_bin_warp(s_hist[0], rank, K, s_val);
  __syncthreads();
  const int b1 = s_val[0];         // high byte of the K-th largest key
  const int above1 = s_val[1];     // keys with a larger high byte
  const int pre_above = s_val[2];  // ... of them in the earlier CTAs

  // Pass 2: histogram of the low byte of the keys in bin b1.
#pragma unroll
  for (int i = 0; i < E; ++i)
    if (keys[i] >= 0 && (keys[i] >> 8) == b1) atomicAdd(&s_wh[warp % HISTS][keys[i] & 255], 1);
  __syncthreads();
  if (t < 256) {
    int h = 0;
#pragma unroll
    for (int w = 0; w < HISTS; ++w) h += s_wh[w][t];
#pragma unroll
    for (int c = 0; c < kCluster; ++c) cluster.map_shared_rank(&s_hist[1][rank][t], c)[0] = h;
  }
  // After this barrier no CTA touches another's shared memory, so none has to wait at exit.
  cluster.sync();
  if (warp == 0) find_bin_warp(s_hist[1], rank, K - above1, s_val + 4);
  __syncthreads();
  const int kth = (b1 << 8) | s_val[4];
  const int n_gt = above1 + s_val[5];        // keys above kth
  const int base_gt = pre_above + s_val[6];  // ... of them in the earlier CTAs
  const int base_eq = s_val[7];              // kth ties in the earlier CTAs
  const int remaining = K - n_gt;            // kth ties to take, in position order

  // Pass 3: place this chunk's selected positions.
  int n_gt_t = 0, n_eq_t = 0;
#pragma unroll
  for (int i = 0; i < E; ++i) {
    n_gt_t += keys[i] > kth;
    n_eq_t += keys[i] == kth;
  }
  const uint32_t packed = block_exclusive_scan_once<THREADS>(
      static_cast<uint32_t>(n_gt_t) | (static_cast<uint32_t>(n_eq_t) << 16), s_warp);
  int slot_gt = base_gt + static_cast<int>(packed & 0xFFFFu);
  int rank_eq = base_eq + static_cast<int>(packed >> 16);
  int64_t* grow = global_ids + static_cast<int64_t>(m) * stride_gm;
  if (n_gt_t + n_eq_t > 0) {
#pragma unroll
    for (int i = 0; i < E; ++i) {
      int slot = -1;
      if (keys[i] > kth) {
        slot = slot_gt++;
      } else if (keys[i] == kth) {
        if (rank_eq < remaining) slot = n_gt + rank_eq;
        ++rank_eq;
      }
      if (slot >= 0) grow[slot] = pos0 + i + vocab_start;
    }
  }
}

// Candidate tail: one CTA per row, one thread per gathered candidate (candidate order r * NC + j).

// One torch SmallBitonicSort<32> round, descending, on (code << 6 | index << 1 | valid) per lane.
__device__ __forceinline__ uint32_t bitonic_round(uint32_t g, int lane, int log2_stride, int half) {
  const int stride = 1 << log2_stride;
  const uint32_t other = __shfl_xor_sync(0xffffffffu, g, stride);
  const bool is_a = (lane & stride) == 0;
  const uint32_t a = is_a ? g : other;
  const uint32_t b = is_a ? other : g;
  const int pos = is_a ? lane : (lane ^ stride);
  const int tx = ((pos >> (log2_stride + 1)) << log2_stride) | (pos & (stride - 1));
  const bool flag = half != 0 && (tx & half) != 0;
  const bool swap = (((a >> 6) > (b >> 6)) && (a & 1u)) || !(b & 1u);
  const bool exchange = swap == flag;
  return is_a ? (exchange ? b : a) : (exchange ? a : b);
}

// Two CTAs of 1024 threads per SM: 32 registers per thread.
template <int NC>
__global__ void __launch_bounds__(1024, 2) eagle4_candidate_tail_cuda_kernel(
    const __nv_bfloat16* __restrict__ cand_logits,
    int64_t stride_cl_r,
    int64_t stride_cl_m,
    const int64_t* __restrict__ cand_tokens,
    int64_t stride_ct_r,
    int64_t stride_ct_m,
    float* __restrict__ logits_out,
    int64_t* __restrict__ tokens_out,
    float* __restrict__ topk_probs,
    int64_t* __restrict__ topk_tokens,
    int R,
    int topk) {
  constexpr int KEY_BITS = 16;
  constexpr int ITER = NC / 32;  // slots per lane of warp 0
  __shared__ float s_logit[NC];
  __shared__ int64_t s_tok[NC];
  __shared__ uint16_t s_code[NC];
  __shared__ int s_wg[32];
  __shared__ int s_we[32];
  __shared__ uint32_t s_pack[32];

  const int64_t m = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if (R > 1) {
    const int r = t / NC;
    const int j = t - r * NC;
    const __nv_bfloat16 x = cand_logits[r * stride_cl_r + m * stride_cl_m + j];
    const int64_t tok = cand_tokens[r * stride_ct_r + m * stride_ct_m + j];
    const uint32_t key = radix_key(x);
    // The NC-th largest key, one bit per block-wide count from the top, as torch's radix select.
    uint32_t prefix = 0u;
    int remaining = NC;
#pragma unroll
    for (int bit = KEY_BITS - 1; bit >= 0; --bit) {
      const uint32_t cand = prefix | (1u << bit);
      const uint32_t hi = ~((1u << bit) - 1u);
      const int c = __syncthreads_count((key & hi) == cand);
      if (c >= remaining) {
        prefix = cand;
      } else {
        remaining -= c;
      }
    }
    const uint32_t kth = prefix;
    const int n_gt = NC - remaining;
    const bool gt = key > kth;
    const bool eq = key == kth;
    const unsigned bg = __ballot_sync(0xffffffffu, gt);
    const unsigned be = __ballot_sync(0xffffffffu, eq);
    if (lane == 0) {
      s_wg[warp] = __popc(bg);
      s_we[warp] = __popc(be);
    }
    __syncthreads();
    int pg = 0, pe = 0;
    for (int w = 0; w < warp; ++w) {
      pg += s_wg[w];
      pe += s_we[w];
    }
    const unsigned lt_mask = (1u << lane) - 1u;
    const int rg = pg + __popc(bg & lt_mask);
    const int re = pe + __popc(be & lt_mask);
    if (gt || (eq && re < remaining)) {
      const int dest = gt ? rg : n_gt + re;
      const float f = static_cast<float>(x);
      logits_out[m * NC + dest] = f;
      tokens_out[m * NC + dest] = tok;
      s_logit[dest] = f;
      s_tok[dest] = tok;
    }
  } else {
    const float f = static_cast<float>(cand_logits[m * stride_cl_m + t]);
    logits_out[m * NC + t] = f;
    s_logit[t] = f;
    if (topk > 0) s_tok[t] = cand_tokens[m * stride_ct_m + t];
  }
  if (topk == 0) return;
  __syncthreads();
  if (warp != 0) return;

  // torch softmax_warp_forward (fp32, WARP_SIZE 32, WARP_ITERATIONS = NC / 32), one row.
  float e[ITER];
#pragma unroll
  for (int it = 0; it < ITER; ++it) e[it] = s_logit[lane + it * 32];
  float mx = e[0];
#pragma unroll
  for (int it = 0; it < ITER; ++it) mx = mx > e[it] ? mx : e[it];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float b = __shfl_xor_sync(0xffffffffu, mx, off);
    mx = mx < b ? b : mx;
  }
  float sum = 0.0f;
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    e[it] = expf(e[it] - mx);
    sum += e[it];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum = sum + __shfl_xor_sync(0xffffffffu, sum, off);
  uint32_t pk[ITER];
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    const float p = (sum == 0.0f) ? std::numeric_limits<float>::quiet_NaN() : e[it] / sum;
    pk[it] = radix_key(p);
  }

  // Top-k by repeated warp argmax (largest key, then lowest slot); lane i keeps the i-th.
  unsigned live = (1u << ITER) - 1u;
  uint32_t sel_key = 0, prev = 0;
  int sel_slot = 0;
  bool distinct = true;
  for (int i = 0; i < topk; ++i) {
    uint32_t lk = 0;  // valid keys are >= radix_key(+0.0f) = 0x80000000
#pragma unroll
    for (int it = 0; it < ITER; ++it) lk = ((live >> it) & 1u) && pk[it] > lk ? pk[it] : lk;
    const uint32_t gk = __reduce_max_sync(0xffffffffu, lk);
    uint32_t ls = 0;
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      const uint32_t c = static_cast<uint32_t>(NC - (lane + it * 32));
      ls = ((live >> it) & 1u) && pk[it] == gk && c > ls ? c : ls;
    }
    const int slot = NC - static_cast<int>(__reduce_max_sync(0xffffffffu, ls));
    if ((slot & 31) == lane) live &= ~(1u << (slot >> 5));
    if (lane == i) {
      sel_key = gk;
      sel_slot = slot;
    }
    distinct = distinct && (i == 0 || gk != prev);
    prev = gk;
  }
  if (distinct) {  // torch.max (k = 1) / any sort of distinct keys: descending key order
    if (lane < topk) {
      topk_probs[m * topk + lane] = radix_key_to_float(sel_key);
      topk_tokens[m * topk + lane] = s_tok[sel_slot];
    }
    return;
  }
  // Ties: torch.topk(sorted=True)'s order, gatherTopK placement (above kth, then ties, by slot) ...
  const uint32_t kth = prev;
  const unsigned lt_mask = (1u << lane) - 1u;
  const int n_gt = __popc(__ballot_sync(0xffffffffu, lane < topk && sel_key > kth));
  {
    // code: 32 - (index of the key among the distinct selected keys, largest first)
    const uint32_t up = __shfl_up_sync(0xffffffffu, sel_key, 1);
    const unsigned starts = __ballot_sync(0xffffffffu, lane < topk && (lane == 0 || sel_key != up));
    const int distinct_index = __popc(starts & ((2u << lane) - 1u)) - 1;
    if (lane < topk) s_code[sel_slot] = static_cast<uint16_t>(((32 - distinct_index) << 5) | lane);
  }
  s_pack[lane] = 0;
  __syncwarp();
  int seen_gt = 0, seen_eq = 0;
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    const int slot = lane + it * 32;
    const bool gt = pk[it] > kth;
    const bool eq = pk[it] == kth;
    const unsigned bg = __ballot_sync(0xffffffffu, gt);
    const unsigned be = __ballot_sync(0xffffffffu, eq);
    const int rg = seen_gt + __popc(bg & lt_mask);
    const int re = seen_eq + __popc(be & lt_mask);
    if (gt || (eq && re < topk - n_gt)) {
      const int dest = gt ? rg : n_gt + re;
      s_pack[dest] = (static_cast<uint32_t>(s_code[slot]) << 1) | 1u;
    }
    seen_gt += __popc(bg);
    seen_eq += __popc(be);
  }
  __syncwarp();
  // ... then the SmallBitonicSort<32> network (sizes 2..16, then the final merge).
  uint32_t g = s_pack[lane];
#pragma unroll
  for (int log2_size = 1; log2_size <= 5; ++log2_size) {
#pragma unroll
    for (int log2_stride = log2_size - 1; log2_stride >= 0; --log2_stride) {
      g = bitonic_round(g, lane, log2_stride, log2_size < 5 ? 1 << (log2_size - 1) : 0);
    }
  }
  const int idx = static_cast<int>((g >> 1) & 31u);  // the entry's selection index
  const uint32_t out_key = __shfl_sync(0xffffffffu, sel_key, idx);
  const int out_slot = __shfl_sync(0xffffffffu, sel_slot, idx);
  if (lane < topk) {
    topk_probs[m * topk + lane] = radix_key_to_float(out_key);
    topk_tokens[m * topk + lane] = s_tok[out_slot];
  }
}


template <int THREADS, int E>
void launch_shard_topk(TensorView scores, int k, int64_t vocab_start, TensorView global_ids) {
  const int M = scores.size(0), N = scores.size(1);
  dim3 grid(kCluster, M);
  eagle4_shard_topk_cluster_kernel<THREADS, E><<<grid, THREADS, 0, get_stream(scores.device())>>>(
      static_cast<const __nv_bfloat16*>(scores.data_ptr()), scores.stride(0), N, k,
      static_cast<int64_t*>(global_ids.data_ptr()), global_ids.stride(0), vocab_start);
}

}  // namespace

// global_ids: [M, k] int64 with unit column stride.
void eagle4_shard_topk(TensorView scores, int64_t k, int64_t vocab_start, TensorView global_ids) {
  TVM_FFI_ICHECK(scores.ndim() == 2 && scores.stride(1) == 1);
  TVM_FFI_ICHECK_EQ(scores.dtype(), dl_bfloat16) << "eagle4_shard_topk: bf16 scores";
  TVM_FFI_ICHECK_EQ(global_ids.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(global_ids.stride(1), 1);
  const int64_t M = scores.size(0), N = scores.size(1);
  if (M == 0) return;
  TVM_FFI_ICHECK(k >= 1 && k <= N && N <= kCluster * 1024 * 32 && M <= 65535)
      << "eagle4_shard_topk does not cover " << M << " x " << N << " scores with k=" << k;
  if (N <= kCluster * 512 * 8) {  // TP8 shards of 248320
    launch_shard_topk<512, 8>(scores, k, vocab_start, global_ids);
  } else if (N <= kCluster * 1024 * 8) {  // TP4
    launch_shard_topk<1024, 8>(scores, k, vocab_start, global_ids);
  } else if (N <= kCluster * 1024 * 16) {  // TP2
    launch_shard_topk<1024, 16>(scores, k, vocab_start, global_ids);
  } else {  // TP1
    launch_shard_topk<1024, 32>(scores, k, vocab_start, global_ids);
  }
}

// Inputs [R, M, NC] (unit column stride); tokens_out unused if R == 1, top-k outputs if topk == 0.
void eagle4_candidate_tail(TensorView cand_logits, TensorView cand_tokens, int64_t topk,
                           TensorView logits_out, TensorView tokens_out, TensorView topk_probs,
                           TensorView topk_tokens) {
  TVM_FFI_ICHECK(cand_logits.ndim() == 3 && cand_tokens.ndim() == 3);
  TVM_FFI_ICHECK(cand_logits.stride(2) == 1 && cand_tokens.stride(2) == 1);
  TVM_FFI_ICHECK_EQ(cand_logits.dtype(), dl_bfloat16) << "eagle4_candidate_tail: bf16 logits";
  TVM_FFI_ICHECK_EQ(cand_tokens.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(logits_out.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(tokens_out.dtype(), dl_int64);
  TVM_FFI_ICHECK_EQ(topk_probs.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(topk_tokens.dtype(), dl_int64);
  const int64_t R = cand_logits.size(0), M = cand_logits.size(1), NC = cand_logits.size(2);
  if (M == 0) return;
  TVM_FFI_ICHECK(R >= 1 && R * NC <= 1024 && topk <= 32)
      << "eagle4_candidate_tail does not cover " << R << " x " << NC << " candidates with top_k="
      << topk;
  auto rows_of = [M](const TensorView& t, int64_t width) {
    return t.IsContiguous() && t.size(0) == M && t.size(1) == width;
  };
  TVM_FFI_ICHECK(rows_of(logits_out, NC) && (R == 1 || rows_of(tokens_out, NC)) &&
                 (topk == 0 || (rows_of(topk_probs, topk) && rows_of(topk_tokens, topk))))
      << "eagle4_candidate_tail: contiguous [M, NC] / [M, top_k] outputs";
  decltype(&eagle4_candidate_tail_cuda_kernel<32>) kernel = nullptr;
  switch (NC) {
    case 32: kernel = eagle4_candidate_tail_cuda_kernel<32>; break;
    case 64: kernel = eagle4_candidate_tail_cuda_kernel<64>; break;
    case 128: kernel = eagle4_candidate_tail_cuda_kernel<128>; break;
    case 256: kernel = eagle4_candidate_tail_cuda_kernel<256>; break;
    default:
      TVM_FFI_ICHECK(false) << "eagle4_candidate_tail: NC " << NC << " not in 32, 64, 128, 256";
  }
  const int threads = static_cast<int>(R * NC);  // one per gathered candidate
  kernel<<<static_cast<unsigned>(M), threads, 0, get_stream(cand_logits.device())>>>(
      static_cast<const __nv_bfloat16*>(cand_logits.data_ptr()), cand_logits.stride(0),
      cand_logits.stride(1), static_cast<const int64_t*>(cand_tokens.data_ptr()),
      cand_tokens.stride(0), cand_tokens.stride(1), static_cast<float*>(logits_out.data_ptr()),
      static_cast<int64_t*>(tokens_out.data_ptr()), static_cast<float*>(topk_probs.data_ptr()),
      static_cast<int64_t*>(topk_tokens.data_ptr()), static_cast<int>(R), static_cast<int>(topk));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle4_shard_topk, eagle4_shard_topk);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle4_candidate_tail, eagle4_candidate_tail);
