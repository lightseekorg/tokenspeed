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

// EAGLE-4 draft trees: per-request node pool, the root (node 0) then children in expansion order.

#include <curand_kernel.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cstdint>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

constexpr int MAX_BATCH_SIZE = 512;       // the work deltas' one thread per request row
constexpr float DEPTH_DECAY = 0.9999f;    // draft depth d >= 1's children: path score * DEPTH_DECAY^(d + 1)
constexpr int MAX_TREE_NODES = 64;        // verify tree nodes per request

// The root's children: the first level of the pool, the work and sort lists.
__global__ void update_node_data_kernel(float* __restrict__ pool_scores,
                                        int64_t* __restrict__ pool_tokens,
                                        float* __restrict__ work_scores,
                                        float* __restrict__ sort_scores,
                                        const float* __restrict__ child_scores,
                                        const int64_t* __restrict__ child_tokens, int batch_size,
                                        int node_top_k, int node_stride, int work_stride,
                                        int sort_stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * node_top_k) {
    const int b = idx / node_top_k;
    const int node = idx % node_top_k + 1;  // the root's children are nodes 1..K
    const float score = child_scores[idx];
    const int64_t token = child_tokens[idx];
    pool_scores[b * node_stride + node] = score;
    pool_tokens[b * node_stride + node] = token;
    work_scores[b * work_stride + node] = score;
    sort_scores[b * sort_stride + node] = score;
  }
}

// work_deltas[j]: batch mean of each row's best min(node_count + j, verify_num) merged scores.
template <int kThreads>
__global__ void calculate_work_deltas_kernel(const float* __restrict__ work_scores,
                                             float* __restrict__ work_deltas,
                                             const int* __restrict__ live_rows, int total_cols,
                                             const int* __restrict__ node_count_ptr,
                                             const int* __restrict__ verify_num_ptr) {
  const int node_count = *node_count_ptr;
  if (node_count == 0) return;
  const int j = blockIdx.x;
  const int tid = threadIdx.x;
  const int batch_size = *live_rows;
  const int verify_num = *verify_num_ptr;

  __shared__ float s_reduction_sums[kThreads];

  float row_sum = 0.0f;
  if (tid < batch_size) {
    const int slice_len = node_count + j;
    const int k = min(slice_len, verify_num);

    const float* temp_scores_ptr = work_scores + tid * total_cols;
    const float* topk_scores_ptr = temp_scores_ptr + node_count;

    int pA = 0;
    int pB = 0;
    for (int i = 0; i < k; ++i) {
      float val_a = (pA < node_count) ? temp_scores_ptr[pA] : 0.0f;
      float val_b = (pB < j) ? topk_scores_ptr[pB] : 0.0f;
      if (val_a >= val_b) {
        row_sum += val_a;
        pA++;
      } else {
        row_sum += val_b;
        pB++;
      }
    }
  }

  s_reduction_sums[tid] = row_sum;
  __syncthreads();

  for (unsigned int s = kThreads / 2; s > 0; s >>= 1) {  // 512's extra step adds zeros: 256's bits
    if (tid < s) s_reduction_sums[tid] += s_reduction_sums[tid + s];
    __syncthreads();
  }

  if (tid == 0) work_deltas[j] = s_reduction_sums[0] / batch_size;
}

// Depth update, one CTA per request: sort children, best K become the frontier, all join the pool.
constexpr int DRAFT_DEPTH_THREADS = 768;  // warps 0-4 bookkeep, the other 19 copy hidden rows
constexpr int DRAFT_DEPTH_COPY_THREADS = DRAFT_DEPTH_THREADS - 5 * 32;
constexpr int DRAFT_DEPTH_MAX_CHILDREN = 64;  // width * K, the sort network's size
constexpr int DRAFT_DEPTH_MAX_VERIFY = 256;   // the sort list's merged head
constexpr int DRAFT_DEPTH_MAX_MASK_WORDS = 4;
constexpr int DRAFT_DEPTH_COPY_UNROLL = 4;

// Compare-exchange of positions p and p ^ stride; strict comparison, so ties and NaN never swap.
__device__ __forceinline__ void sort_exchange_lanes(float& v, int& x, int p, int stride, int size,
                                                    int n) {
  const float o = __shfl_xor_sync(0xffffffff, v, stride >> 1);
  const int ox = __shfl_xor_sync(0xffffffff, x, stride >> 1);
  const bool lower = (p & stride) == 0;
  const bool descending = (p & size) == 0;
  const bool swap = ((p | stride) < n) & (lower == descending ? v < o : v > o);
  v = swap ? o : v;
  x = swap ? ox : x;
}

// Descending warp sort of n <= 64 values, lane l holding positions 2l, 2l + 1; pairs past n skip.
__device__ __forceinline__ void bitonic_sort_64_warp(float& v0, int& x0, float& v1, int& x1,
                                                     int n) {
  const int p0 = 2 * (threadIdx.x & 31);
#pragma unroll
  for (int size = 2; size <= DRAFT_DEPTH_MAX_CHILDREN; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 1; stride >>= 1) {
      sort_exchange_lanes(v0, x0, p0, stride, size, n);
      sort_exchange_lanes(v1, x1, p0 + 1, stride, size, n);
    }
    // Stride 1: the pair (2l, 2l + 1) sits in one lane.
    const bool descending = (p0 & size) == 0;
    const bool swap = (p0 + 1 < n) & (descending ? v0 < v1 : v0 > v1);
    const float v = v0;
    const int x = x0;
    v0 = swap ? v1 : v0;
    x0 = swap ? x1 : x0;
    v1 = swap ? v : v1;
    x1 = swap ? x : x1;
  }
}

__global__ void __launch_bounds__(DRAFT_DEPTH_THREADS) draft_depth_update_kernel(
    const float* __restrict__ child_probs, const int64_t* __restrict__ child_tokens,
    const int64_t* __restrict__ expanded, const uint4* __restrict__ child_hidden,
    int64_t* __restrict__ frontier_tokens, float* __restrict__ frontier_scores,
    uint4* __restrict__ frontier_hidden, int64_t* __restrict__ frontier_index,
    int64_t* __restrict__ pool_tokens, float* __restrict__ pool_scores,
    int64_t* __restrict__ pool_prev, int64_t node_stride, float* __restrict__ work_scores,
    int64_t work_stride, float* __restrict__ sort_scores, int64_t sort_stride,
    int64_t* __restrict__ positions, int64_t* __restrict__ slot_starts,
    int64_t* __restrict__ lane_mask, int mask_words, int node_top_k,
    const int* __restrict__ width_ptr, int depth, const int* __restrict__ cumu_count_ptr,
    const int* __restrict__ verify_num_ptr, int num_slots, int vecs_per_row) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int k = node_top_k;
  const int width = *width_ptr;
  const int cumu_count = *cumu_count_ptr;
  const int verify_num = *verify_num_ptr;
  const int n = width * k;
  const int head = min(verify_num, cumu_count);    // sort list entries merged with the children
  const int kept = min(verify_num, cumu_count + n);  // sort list entries after the merge

  __shared__ float s_scores[DRAFT_DEPTH_MAX_CHILDREN];  // children's path scores, sorted
  __shared__ int s_order[DRAFT_DEPTH_MAX_CHILDREN];     // their child positions
  __shared__ int s_parent[DRAFT_DEPTH_MAX_CHILDREN];    // the frontier's parent lanes
  __shared__ int64_t s_tokens[DRAFT_DEPTH_MAX_CHILDREN];  // children's tokens, unsorted
  __shared__ float s_head[DRAFT_DEPTH_MAX_VERIFY];
  __shared__ int64_t s_mask[DRAFT_DEPTH_MAX_CHILDREN * DRAFT_DEPTH_MAX_MASK_WORDS];

  // The request's K lanes: child t of lane t / K at t, lane r's row at r.
  const int64_t lanes = static_cast<int64_t>(b) * k;
  const float* probs = child_probs + lanes * k;
  const int64_t* tokens = child_tokens + lanes * k;
  const int64_t* parents = expanded + lanes;
  float* scores = frontier_scores + lanes;
  int64_t* masks = lane_mask + lanes * mask_words;
  const int64_t node_row = b * node_stride;

  // Reads before the rewrite barrier; all lanes' children are read so loads need not wait for n.
  if (warp == 0) {
    const int p0 = 2 * lane;
    const int p1 = p0 + 1;
    float v0 = p0 < k * k ? probs[p0] * scores[p0 / k] : 0.0f;
    float v1 = p1 < k * k ? probs[p1] * scores[p1 / k] : 0.0f;
    v0 = p0 < n ? v0 : -1e10f;
    v1 = p1 < n ? v1 : -1e10f;
    int x0 = p0;
    int x1 = p1;
    bitonic_sort_64_warp(v0, x0, v1, x1, n);
    s_scores[p0] = v0;
    s_scores[p1] = v1;
    s_order[p0] = x0;
    s_order[p1] = x1;
    __syncwarp();
    for (int i = lane; i < k; i += 32) s_parent[i] = s_order[i] / k;
  } else if (warp == 1) {
    // The children's pool tokens and scores.
    const float depth_decay = powf(DEPTH_DECAY, (float)(depth + 1));
    for (int t = lane; t < k * k; t += 32) {
      const int64_t token = tokens[t];
      const float score = probs[t] * scores[t / k];
      if (t >= n) continue;
      s_tokens[t] = token;
      const int node = cumu_count + t;
      pool_tokens[node_row + node] = token;
      pool_scores[node_row + node] = score * depth_decay;
    }
  } else if (warp == 2) {
    // The children's parent.
    for (int t = lane; t < k * k; t += 32) {
      const int64_t parent = parents[t / k];
      if (t >= n) continue;
      pool_prev[node_row + cumu_count + t] = parent;
    }
  } else if (warp == 3) {
    for (int i = lane; i < head; i += 32) {
      const float v = sort_scores[b * sort_stride + i];
      s_head[i] = v;
      work_scores[b * work_stride + i] = v;
    }
  } else if (warp == 4) {
    for (int t = lane; t < k * mask_words; t += 32) s_mask[t] = masks[t];
    for (int r = lane; r < k; r += 32) positions[lanes + r] += 1;
    if (lane == 0) slot_starts[b] += k;
  }
  __syncthreads();

  if (warp == 0) {
    for (int i = lane; i < k; i += 32) {
      const int child = s_order[i];
      frontier_index[lanes + i] = cumu_count + child;
      scores[i] = s_scores[i];
      frontier_tokens[lanes + i] = s_tokens[child];
      work_scores[b * work_stride + head + i] = s_scores[i];
    }
    if ((depth + 1) * k <= num_slots) {
      for (int t = lane; t < k * mask_words; t += 32) {
        const int i = t / mask_words;
        const int word = t % mask_words;
        const int slot = depth * k + i;
        uint64_t bits = static_cast<uint64_t>(s_mask[s_parent[i] * mask_words + word]);
        if (slot / 64 == word) bits |= uint64_t{1} << (slot % 64);
        masks[t] = static_cast<int64_t>(bits);
      }
    }
  } else if (warp <= 4) {
    // Stable rank of each entry of [head, sorted children]; the best `kept` land at their rank.
    const int total = head + n;
    for (int e = tid - 32; e < total; e += 4 * 32) {
      const float key = e < head ? s_head[e] : s_scores[e - head];
      int rank = 0;
      for (int j = 0; j < total; ++j) {
        const float v = j < head ? s_head[j] : s_scores[j - head];
        rank += (v > key) || (v == key && j < e);
      }
      if (rank < kept) sort_scores[b * sort_stride + rank] = key;
    }
  } else {
    // The frontier's input hidden rows: their parent lanes' output rows.
    const int total_vecs = k * vecs_per_row;
    const uint4* hidden_in = child_hidden + lanes * vecs_per_row;
    uint4* hidden_out = frontier_hidden + lanes * vecs_per_row;
    constexpr int step = DRAFT_DEPTH_COPY_UNROLL * DRAFT_DEPTH_COPY_THREADS;
    for (int base = tid - 5 * 32; base < total_vecs; base += step) {
      uint4 data[DRAFT_DEPTH_COPY_UNROLL];
#pragma unroll
      for (int u = 0; u < DRAFT_DEPTH_COPY_UNROLL; ++u) {
        const int v = base + u * DRAFT_DEPTH_COPY_THREADS;
        if (v < total_vecs) {
          const int i = v / vecs_per_row;
          data[u] = hidden_in[(s_parent[i] - i) * vecs_per_row + v];
        }
      }
#pragma unroll
      for (int u = 0; u < DRAFT_DEPTH_COPY_UNROLL; ++u) {
        const int v = base + u * DRAFT_DEPTH_COPY_THREADS;
        if (v < total_vecs) hidden_out[v] = data[u];
      }
    }
  }
}

// Verify ranking order, torch.argmax's: NaN first, then larger value, then lower index.
__device__ __forceinline__ bool argmax_before(float v, int i, float best_v, int best_i) {
  const bool an = isnan(v);
  const bool bn = isnan(best_v);
  if (an != bn) return an;
  if (!an && v != best_v) return v > best_v;
  return i < best_i;
}

__global__ void stable_topk_desc_kernel(const float* __restrict__ scores, int64_t row_stride,
                                        const int* __restrict__ live_rows,
                                        const int* __restrict__ n_ptr,
                                        const int* __restrict__ k_ptr,
                                        float* __restrict__ out_values,
                                        int64_t* __restrict__ out_indices, int64_t out_stride) {
  __shared__ float key[1024];
  __shared__ int idx[1024];
  const int row = blockIdx.x;
  if (row >= *live_rows) return;
  const int n = *n_ptr;
  const int k = *k_ptr;
  int p = 1;
  while (p < n) p <<= 1;
  const float* in = scores + row * row_stride;
  for (int i = threadIdx.x; i < p; i += blockDim.x) {
    key[i] = i < n ? in[i] : 0.0f;
    idx[i] = i;
  }
  __syncthreads();
  for (int size = 2; size <= p; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < p; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const float vi = key[i], vj = key[j];
          const int ii = idx[i], ij = idx[j];
          const bool j_first = ij < n && (ii >= n || argmax_before(vj, ij, vi, ii));
          const bool swap = ((i & size) == 0) ? j_first : !j_first;
          if (swap) {
            key[i] = vj;
            key[j] = vi;
            idx[i] = ij;
            idx[j] = ii;
          }
        }
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    out_values[row * out_stride + i] = key[i];
    out_indices[row * out_stride + i] = idx[i];
  }
}

// Verify tree into the next round's inputs: slot i < verify_num is ranked node i, later the root.
__global__ void write_verify_tree_kernel(const int64_t* __restrict__ ranked, int64_t ranked_stride,
                                         const int64_t* __restrict__ pool_tokens,
                                         const int64_t* __restrict__ pool_prev,
                                         int64_t node_stride,
                                         const int64_t* __restrict__ rows,
                                         int32_t* __restrict__ input_map, int64_t input_stride,
                                         int32_t* __restrict__ parent_map, int64_t parent_stride,
                                         const int* __restrict__ live_rows,
                                         const int* __restrict__ verify_num_ptr, int num_slots) {
  __shared__ int64_t s_nodes[MAX_TREE_NODES];
  const int b = blockIdx.x;
  if (b >= *live_rows) return;
  const int verify_num = *verify_num_ptr;
  for (int i = threadIdx.x; i < verify_num; i += blockDim.x) {
    s_nodes[i] = ranked[b * ranked_stride + i];
  }
  __syncthreads();
  const int64_t row = rows[b];
  for (int i = threadIdx.x; i < num_slots; i += blockDim.x) {
    int64_t node = 0;
    int32_t parent = -1;
    if (i < verify_num) {
      node = s_nodes[i];
      const int64_t prev = pool_prev[b * node_stride + node];
      for (int j = 0; j < i; ++j) {
        if (s_nodes[j] == prev) parent = j;
      }
    }
    input_map[row * input_stride + i] = static_cast<int32_t>(pool_tokens[b * node_stride + node]);
    parent_map[row * parent_stride + i] = parent;
  }
}


// Sampled trees: K children by Gumbel-top-K from q (cut by top-k / top-p); greedy rows take top-K.
constexpr int SAMPLE_CHILDREN_THREADS = 128;
constexpr int SAMPLE_CHILDREN_PER_THREAD = 8;

__device__ __forceinline__ void block_argmax_128(float& v, int& i, float* red_v, int* red_i) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffff, v, offset);
    const int oi = __shfl_xor_sync(0xffffffff, i, offset);
    if (ov > v || (ov == v && oi < i)) {
      v = ov;
      i = oi;
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    red_v[warp] = v;
    red_i[warp] = i;
  }
  __syncthreads();
  v = red_v[0];
  i = red_i[0];
  for (int w = 1; w < SAMPLE_CHILDREN_THREADS / 32; ++w) {
    if (red_v[w] > v || (red_v[w] == v && red_i[w] < i)) {
      v = red_v[w];
      i = red_i[w];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float block_sum_128(float v, float* red) {
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset);
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) red[warp] = v;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < SAMPLE_CHILDREN_THREADS / 32; ++w) total += red[w];
  __syncthreads();
  return total;
}

__global__ void sample_children_kernel(
    const float* __restrict__ logits_base, int64_t logits_s0, int64_t logits_s1,
    const int64_t* __restrict__ cand_base, int64_t cand_s0, int64_t cand_s1,
    int64_t* __restrict__ tokens_base, int64_t tokens_s0, int64_t tokens_s1,
    float* __restrict__ scores_base, int64_t scores_s0, int64_t scores_s1,
    const float* __restrict__ temps, const int32_t* __restrict__ top_ks,
    const float* __restrict__ top_ps, float* __restrict__ q_store, int32_t* __restrict__ qid_store,
    int32_t* __restrict__ node_slot, int32_t* __restrict__ node_pos, int rows_per_req,
    const int* __restrict__ width_ptr, int slot_base, const int* __restrict__ cumu_start_ptr,
    int num_slots, int max_nodes, int nc, int k, int64_t vocab_limit, uint64_t seed,
    const int64_t* __restrict__ round_counter, int launch) {
  // Request b's row w of its first `width` rows, expanded row m = b * width + w in the draws.
  const int b = blockIdx.x / rows_per_req;
  const int w = blockIdx.x - b * rows_per_req;
  const int width = width_ptr == nullptr ? rows_per_req : *width_ptr;
  if (w >= width) return;
  const int m = b * width + w;
  const int cumu_start = *cumu_start_ptr;
  const float* logits = logits_base + b * logits_s0 + w * logits_s1;
  const int64_t* cand = cand_base + b * cand_s0 + w * cand_s1;
  int64_t* tokens = tokens_base + b * tokens_s0 + w * tokens_s1;
  float* scores = scores_base + b * scores_s0 + w * scores_s1;
  const int tid = threadIdx.x;
  const int64_t node_row = (int64_t)b * max_nodes + cumu_start + (int64_t)w * k;
  const float temp = temps[b];
  if (temp <= 0.0f) {
    if (tid < k) node_slot[node_row + tid] = -1;
    return;
  }
  __shared__ float red_v[SAMPLE_CHILDREN_THREADS / 32];
  __shared__ int red_i[SAMPLE_CHILDREN_THREADS / 32];
  const float inv_temp = 1.0f / temp;
  float x[SAMPLE_CHILDREN_PER_THREAD];
  float x_max = -FLT_MAX;
#pragma unroll
  for (int t = 0; t < SAMPLE_CHILDREN_PER_THREAD; ++t) {
    const int j = tid + t * SAMPLE_CHILDREN_THREADS;
    const bool valid = j < nc && cand[j] < vocab_limit;
    x[t] = valid ? logits[j] * inv_temp : -FLT_MAX;
    x_max = fmaxf(x_max, x[t]);
  }
  int dummy = 0;
  block_argmax_128(x_max, dummy, red_v, red_i);
  float e_sum = 0.0f;
#pragma unroll
  for (int t = 0; t < SAMPLE_CHILDREN_PER_THREAD; ++t) {
    const int j = tid + t * SAMPLE_CHILDREN_THREADS;
    x[t] = (j < nc && x[t] > -FLT_MAX) ? __expf(x[t] - x_max) : 0.0f;
    e_sum += x[t];
  }
  float inv_sum = 1.0f / block_sum_128(e_sum, red_v);
  const int top_k = top_ks[b];
  const float top_p = top_ps[b];
  if (top_k < nc || top_p < 1.0f) {
    __shared__ float s_q[SAMPLE_CHILDREN_THREADS * SAMPLE_CHILDREN_PER_THREAD];
#pragma unroll
    for (int t = 0; t < SAMPLE_CHILDREN_PER_THREAD; ++t) {
      const int j = tid + t * SAMPLE_CHILDREN_THREADS;
      if (j < nc) s_q[j] = x[t] * inv_sum;
    }
    __syncthreads();
    float kept_sum = 0.0f;
#pragma unroll
    for (int t = 0; t < SAMPLE_CHILDREN_PER_THREAD; ++t) {
      const int j = tid + t * SAMPLE_CHILDREN_THREADS;
      if (j >= nc || x[t] <= 0.0f) continue;
      const float q_j = s_q[j];
      int above = 0;
      float mass_above = 0.0f;
      for (int i = 0; i < nc; ++i) {
        const float q_i = s_q[i];
        if (q_i > q_j) {
          ++above;
          mass_above += q_i;
        }
      }
      if (above < top_k && mass_above < top_p)
        kept_sum += x[t];
      else
        x[t] = 0.0f;
    }
    inv_sum = 1.0f / block_sum_128(kept_sum, red_v);
  }
  const int s = slot_base + w;
  const int64_t store_row = ((int64_t)b * num_slots + s) * nc;
  const uint64_t stream = static_cast<uint64_t>(round_counter[0]) * 64 + launch;
  float key[SAMPLE_CHILDREN_PER_THREAD];
#pragma unroll
  for (int t = 0; t < SAMPLE_CHILDREN_PER_THREAD; ++t) {
    const int j = tid + t * SAMPLE_CHILDREN_THREADS;
    if (j < nc) {
      const float q = x[t] * inv_sum;
      q_store[store_row + j] = q;
      qid_store[store_row + j] = static_cast<int32_t>(cand[j]);
      curandStatePhilox4_32_10_t state;
      curand_init(seed, (stream << 20) | ((uint64_t)m << 10) | (uint64_t)j, 0, &state);
      const double u = ((double)curand(&state) + 0.5) * 2.3283064365386963e-10;
      key[t] = q > 0.0f ? (float)((double)logf(q) - log(-log(u)))
                        : (cand[j] < vocab_limit ? -1e30f : -2e30f);
    } else {
      key[t] = -FLT_MAX;
    }
  }
  for (int i = 0; i < k; ++i) {
    float best_v = -FLT_MAX;
    int best_j = nc;
#pragma unroll
    for (int t = 0; t < SAMPLE_CHILDREN_PER_THREAD; ++t) {
      const int j = tid + t * SAMPLE_CHILDREN_THREADS;
      if (j < nc && (key[t] > best_v || (key[t] == best_v && j < best_j))) {
        best_v = key[t];
        best_j = j;
      }
    }
    block_argmax_128(best_v, best_j, red_v, red_i);
#pragma unroll
    for (int t = 0; t < SAMPLE_CHILDREN_PER_THREAD; ++t) {
      if (tid + t * SAMPLE_CHILDREN_THREADS == best_j) key[t] = -FLT_MAX;
    }
    if (tid == 0) {
      tokens[i] = cand[best_j];
      node_slot[node_row + i] = s;
      node_pos[node_row + i] = best_j;
      // A child past the cut's support can never be accepted.
      if (best_v <= -1e29f) scores[i] = 0.0f;
    }
  }
}

// Tree rejection sampling, one CTA per request, one warp per node; greedy rows use the argmax only.
__device__ __forceinline__ int next_child(const int32_t* s_parent, int node, int after,
                                          int num_nodes) {
  for (int j = after + 1; j < num_nodes; ++j) {
    if (s_parent[j] == node) return j;
  }
  return -1;
}

__global__ void tree_reject_sample_kernel(
    float* __restrict__ probas, int64_t probas_stride, const int32_t* __restrict__ tokens,
    const int32_t* __restrict__ parents, const int64_t* __restrict__ q_rows,
    const int32_t* __restrict__ q_slot, const int32_t* __restrict__ q_pos,
    const float* __restrict__ q_store, const int32_t* __restrict__ qid_store,
    const float* __restrict__ coins, int64_t coins_stride, const int32_t* __restrict__ top_ks,
    const int64_t* __restrict__ argmax, int32_t* __restrict__ path,
    int32_t* __restrict__ predicts, int32_t* __restrict__ accept_length,
    int64_t* __restrict__ last_row, int num_nodes, int num_slots, int nc, int vocab) {
  const int b = blockIdx.x;
  __shared__ int32_t s_parent[MAX_TREE_NODES];
  __shared__ int s_accept_child[MAX_TREE_NODES];
  for (int i = threadIdx.x; i < num_nodes; i += blockDim.x) {
    s_parent[i] = parents[(int64_t)b * num_nodes + i];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int num_warps = blockDim.x >> 5;
  const int64_t q_row = q_rows != nullptr ? q_rows[b] : -1;
  for (int node = threadIdx.x >> 5; node < num_nodes; node += num_warps) {
    const int64_t row = (int64_t)b * num_nodes + node;
    float* p = probas + row * probas_stride;
    int child = next_child(s_parent, node, node, num_nodes);
    int accepted = -1;
    const bool greedy = top_ks[row] == 1;
    const int32_t slot =
        (!greedy && q_row >= 0 && child >= 0) ? q_slot[q_row * num_nodes + child] : -1;
    if (slot >= 0) {
      const int64_t store_row = (q_row * num_slots + slot) * nc;
      const float* q_row_ptr = q_store + store_row;
      const int32_t* id_row = qid_store + store_row;
      constexpr int MAX_SAMPLED_CHILDREN = 16;
      int rejected_pos[MAX_SAMPLED_CHILDREN];
      int rejected_count = 0;
      float q_left = 0.0f, p_cand = 0.0f;
      // Argmax before the updates, for a vanishing residual; each touched entry is folded in first.
      float best_v = -INFINITY;
      int best_i = INT_MAX;
      for (int j = lane; j < nc; j += 32) {
        const float q_j = q_row_ptr[j];
        if (q_j > 0.0f) {
          q_left += q_j;
          const int32_t token = id_row[j];
          const float p_j = p[token];
          p_cand += p_j;
          if (static_cast<unsigned>(token) < static_cast<unsigned>(vocab) &&
              argmax_before(p_j, token, best_v, best_i)) {
            best_v = p_j;
            best_i = token;
          }
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        q_left += __shfl_xor_sync(0xffffffff, q_left, offset);
        p_cand += __shfl_xor_sync(0xffffffff, p_cand, offset);
      }
      const float z_out = fmaxf(1.0f - p_cand, 0.0f);
      float z = 1.0f;
      while (child >= 0) {
        const int pos = q_pos[q_row * num_nodes + child];
        const int64_t child_token = tokens[(int64_t)b * num_nodes + child];
        const float q_child = q_row_ptr[pos];
        if (q_child > 0.0f && q_left > 0.0f && z > 0.0f) {
          const float q_i = q_child / q_left;
          const float r_child = p[child_token];
          if (static_cast<uint64_t>(child_token) < static_cast<uint64_t>(vocab) &&
              argmax_before(r_child, static_cast<int>(child_token), best_v, best_i)) {
            best_v = r_child;
            best_i = static_cast<int>(child_token);
          }
          if (coins[(int64_t)b * coins_stride + child] * q_i * z < r_child) {
            accepted = child;
            break;
          }
          const float scale = z / q_left;
          float r_sum = 0.0f, q_sum = 0.0f;
          for (int j = lane; j < nc; j += 32) {
            const float q_j = q_row_ptr[j];
            if (q_j <= 0.0f || j == pos) continue;
            const int32_t token = id_row[j];
            bool gone = false;
            for (int r = 0; r < rejected_count; ++r) gone |= (rejected_pos[r] == j);
            if (gone) {
              r_sum += p[token];
              continue;
            }
            const float r_new = fmaxf(p[token] - scale * q_j, 0.0f);
            p[token] = r_new;
            r_sum += r_new;
            q_sum += q_j;
          }
          for (int offset = 16; offset > 0; offset >>= 1) {
            r_sum += __shfl_xor_sync(0xffffffff, r_sum, offset);
            q_sum += __shfl_xor_sync(0xffffffff, q_sum, offset);
          }
          const float r_y = fmaxf(r_child - scale * q_child, 0.0f);
          if (lane == 0) p[child_token] = r_y;
          __syncwarp();
          const float z_new = z_out + r_sum + r_y;
          if (!(z_new > 1e-6f * z)) {
            if (r_child > 0.0f) {
              accepted = child;
            } else {
              // Bonus = target's argmax: untouched entries keep target values (rare, so scanned).
              for (int j = lane; j < vocab; j += 32) {
                const float v = p[j];
                if (argmax_before(v, j, best_v, best_i)) {
                  best_v = v;
                  best_i = j;
                }
              }
              for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_v = __shfl_xor_sync(0xffffffff, best_v, offset);
                const int other_i = __shfl_xor_sync(0xffffffff, best_i, offset);
                if (argmax_before(other_v, other_i, best_v, best_i)) {
                  best_v = other_v;
                  best_i = other_i;
                }
              }
              if (lane == 0) p[best_i] = 1.0f;
              __syncwarp();
            }
            break;
          }
          z = z_new;
          q_left = q_sum;
          if (rejected_count < MAX_SAMPLED_CHILDREN) rejected_pos[rejected_count++] = pos;
        }
        child = next_child(s_parent, node, child, num_nodes);
      }
    } else if (lane == 0 && greedy) {
      const int64_t best = argmax[row];
      while (child >= 0 && tokens[(int64_t)b * num_nodes + child] != best) {
        child = next_child(s_parent, node, child, num_nodes);
      }
      accepted = child;
    } else if (lane == 0) {
      float total = 1.0f;
      while (child >= 0) {
        const int64_t child_token = tokens[(int64_t)b * num_nodes + child];
        if (static_cast<uint64_t>(child_token) < static_cast<uint64_t>(vocab)) {
          const float child_proba = p[child_token];
          // Accepts with probability p'(child) over the mass its rejected siblings left.
          if (child_proba > 0.0f &&
              coins[(int64_t)b * coins_stride + child] * total < child_proba) {
            accepted = child;
            break;
          }
          total = fmaxf(total - child_proba, 0.0f);
          p[child_token] = 0.0f;
        }
        child = next_child(s_parent, node, child, num_nodes);
      }
    }
    if (lane == 0) s_accept_child[node] = accepted;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t* path_b = path + (int64_t)b * num_nodes;
    int32_t* predicts_b = predicts + (int64_t)b * num_nodes;
    int length = 1;
    int node = 0;
    path_b[0] = 0;
    for (int c = s_accept_child[0]; c >= 0; c = s_accept_child[node]) {
      predicts_b[length - 1] = tokens[(int64_t)b * num_nodes + c];
      path_b[length++] = c;
      node = c;
    }
    for (int i = length; i < num_nodes; ++i) path_b[i] = -1;
    accept_length[b] = length;
    last_row[b] = (int64_t)b * num_nodes + node;
  }
}

// Bonus token: inverse-CDF draw from the last accepted node's unnormalized row; greedy: its argmax.
constexpr int RESIDUAL_SAMPLE_THREADS = 1024;
// Runs of 16 staged in shared memory coalesce the loads; each lane still sums its segment in order.
constexpr int RESIDUAL_SAMPLE_RUN = 16;
static_assert(32 % RESIDUAL_SAMPLE_RUN == 0, "a warp load stages whole runs");
constexpr int RESIDUAL_SAMPLE_SMEM_BYTES =
    RESIDUAL_SAMPLE_THREADS * (RESIDUAL_SAMPLE_RUN + 1) * sizeof(float);

__global__ void __launch_bounds__(RESIDUAL_SAMPLE_THREADS)
    residual_sample_kernel(const float* __restrict__ probas, int64_t probas_stride,
                           const int64_t* __restrict__ last_row, const float* __restrict__ coins,
                           const int32_t* __restrict__ top_ks,
                           const int64_t* __restrict__ argmax,
                           const int32_t* __restrict__ accept_length,
                           int32_t* __restrict__ predicts, int num_nodes, int vocab) {
  const int b = blockIdx.x;
  const int64_t row = last_row[b];
  if (top_ks[row] == 1) {
    if (threadIdx.x == 0) {
      predicts[(int64_t)b * num_nodes + accept_length[b] - 1] = static_cast<int32_t>(argmax[row]);
    }
    return;
  }
  const float* p = probas + row * probas_stride;
  const int seg = (vocab + RESIDUAL_SAMPLE_THREADS - 1) / RESIDUAL_SAMPLE_THREADS;
  const int start = min(threadIdx.x * seg, vocab);
  const int end = min(start + seg, vocab);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  extern __shared__ float s_runs[];
  // Row r of this warp's [32, RUN + 1] tile holds a run of lane r's segment.
  float* runs = s_runs + warp * 32 * (RESIDUAL_SAMPLE_RUN + 1);
  // A warp load stages 32 / RUN runs: lane l fills column l % RUN of run l / RUN.
  const int col = lane % RESIDUAL_SAMPLE_RUN;
  float local = 0.0f;
  int last_positive = -1;
  for (int c = 0; c < seg; c += RESIDUAL_SAMPLE_RUN) {
#pragma unroll
    for (int r0 = 0; r0 < 32; r0 += 32 / RESIDUAL_SAMPLE_RUN) {
      const int r = r0 + lane / RESIDUAL_SAMPLE_RUN;
      const int r_start = min((warp * 32 + r) * seg, vocab);
      const int i = r_start + c + col;
      runs[r * (RESIDUAL_SAMPLE_RUN + 1) + col] =
          i < min(r_start + seg, vocab) ? p[i] : 0.0f;
    }
    __syncwarp();
    const int n = min(RESIDUAL_SAMPLE_RUN, end - start - c);
    for (int j = 0; j < n; ++j) {
      const float v = runs[lane * (RESIDUAL_SAMPLE_RUN + 1) + j];
      local += v;
      if (v > 0.0f) last_positive = start + c + j;
    }
    __syncwarp();
  }
  // Inclusive scan of the segment sums over the block.
  __shared__ float s_warp[RESIDUAL_SAMPLE_THREADS / 32];
  __shared__ int s_last[RESIDUAL_SAMPLE_THREADS / 32];
  __shared__ int s_token;
  float incl = local;
  int last = last_positive;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float u = __shfl_up_sync(0xffffffff, incl, offset);
    if (lane >= offset) incl += u;
  }
  for (int offset = 16; offset > 0; offset >>= 1) last = max(last, __shfl_xor_sync(0xffffffff, last, offset));
  if (lane == 31) s_warp[warp] = incl;
  if (lane == 0) s_last[warp] = last;
  if (threadIdx.x == 0) s_token = INT_MAX;
  __syncthreads();
  float before = 0.0f, total = 0.0f;
  int any_last = -1;
  for (int w = 0; w < RESIDUAL_SAMPLE_THREADS / 32; ++w) {
    if (w < warp) before += s_warp[w];
    total += s_warp[w];
    any_last = max(any_last, s_last[w]);
  }
  const float exclusive = before + incl - local;
  const float target = coins[b] * total;
  // A claiming lane rescans its segment, staged run by run through its warp's tile.
  const bool claim = local > 0.0f && exclusive <= target && target < exclusive + local;
  for (unsigned claims = __ballot_sync(0xffffffff, claim); claims != 0; claims &= claims - 1) {
    const int r = __ffs(claims) - 1;
    const int r_start = __shfl_sync(0xffffffff, start, r);
    const int r_end = __shfl_sync(0xffffffff, end, r);
    float acc = exclusive;
    for (int c = r_start; c < r_end; c += RESIDUAL_SAMPLE_RUN) {
      if (lane < RESIDUAL_SAMPLE_RUN) runs[lane] = c + lane < r_end ? p[c + lane] : 0.0f;
      __syncwarp();
      int token = -1;
      if (lane == r) {
        const int n = min(RESIDUAL_SAMPLE_RUN, r_end - c);
        for (int j = 0; j < n; ++j) {
          acc += runs[j];
          if (acc > target && runs[j] > 0.0f) {
            token = c + j;
            break;
          }
        }
      }
      token = __shfl_sync(0xffffffff, token, r);
      __syncwarp();
      if (token >= 0) {
        // Rounding can make neighbouring segments overlap: the lowest claim wins.
        if (lane == 0) atomicMin(&s_token, token);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Rounding can also leave the target in a gap, or at the very end of the mass.
    const int token = s_token != INT_MAX ? s_token : max(any_last, 0);
    predicts[(int64_t)b * num_nodes + accept_length[b] - 1] = token;
  }
}

// A sampled round's q, candidate ids and parent draws (slot -1: none) into the next round's tables.
__global__ void publish_draft_q_kernel(const int64_t* __restrict__ ranked, int64_t ranked_stride,
                                       const int32_t* __restrict__ node_slot,
                                       const int32_t* __restrict__ node_pos, int64_t node_stride,
                                       const float* __restrict__ q_round,
                                       const int32_t* __restrict__ qid_round,
                                       const int64_t* __restrict__ rows, float* __restrict__ q_tab,
                                       int32_t* __restrict__ qid_tab, int32_t* __restrict__ slot_tab,
                                       int32_t* __restrict__ pos_tab,
                                       const int* __restrict__ live_rows,
                                       const int* __restrict__ verify_num_ptr, int num_nodes,
                                       int64_t store_size) {
  const int b = blockIdx.x;
  if (b >= *live_rows) return;
  const int verify_num = *verify_num_ptr;
  const int64_t row = rows[b];
  for (int i = threadIdx.x; i < num_nodes; i += blockDim.x) {
    int32_t slot = -1, pos = 0;
    if (i > 0 && i < verify_num) {
      const int64_t node = ranked[(int64_t)b * ranked_stride + i];
      slot = node_slot[(int64_t)b * node_stride + node];
      pos = node_pos[(int64_t)b * node_stride + node];
    }
    slot_tab[row * num_nodes + i] = slot;
    pos_tab[row * num_nodes + i] = pos;
  }
  for (int64_t i = threadIdx.x; i < store_size; i += blockDim.x) {
    q_tab[row * store_size + i] = q_round[(int64_t)b * store_size + i];
    qid_tab[row * store_size + i] = qid_round[(int64_t)b * store_size + i];
  }
}

// Host entry points: shapes and strides are checked, dtypes are the caller's.
void eagle_harness_update_node_data(TensorView pool_scores, TensorView pool_tokens,
                                    TensorView work_scores, TensorView sort_scores,
                                    TensorView child_scores, TensorView child_tokens) {
  TVM_FFI_ICHECK_EQ(pool_tokens.stride(0), pool_scores.stride(0));
  TVM_FFI_ICHECK(child_scores.IsContiguous() && child_tokens.IsContiguous())
      << "eagle_harness_update_node_data: contiguous [bs, K] children";
  const int batch_size = child_scores.size(0);
  const int node_top_k = child_scores.size(1);
  const int total_elements = batch_size * node_top_k;
  if (total_elements == 0) return;
  const int threads = 256;
  const int blocks = (total_elements + threads - 1) / threads;
  update_node_data_kernel<<<blocks, threads, 0, get_stream(child_scores.device())>>>(
      static_cast<float*>(pool_scores.data_ptr()), static_cast<int64_t*>(pool_tokens.data_ptr()),
      static_cast<float*>(work_scores.data_ptr()), static_cast<float*>(sort_scores.data_ptr()),
      static_cast<const float*>(child_scores.data_ptr()),
      static_cast<const int64_t*>(child_tokens.data_ptr()), batch_size, node_top_k,
      pool_scores.stride(0), work_scores.stride(0), sort_scores.stride(0));
}

void eagle_harness_work_deltas(TensorView work_scores, TensorView work_deltas,
                               TensorView live_rows, TensorView node_count, TensorView verify_num) {
  TVM_FFI_ICHECK_LE(work_scores.size(0), MAX_BATCH_SIZE);
  const auto launch = [&](auto kernel, int threads) {
    kernel<<<work_deltas.size(0), threads, 0, get_stream(work_scores.device())>>>(
        static_cast<const float*>(work_scores.data_ptr()),
        static_cast<float*>(work_deltas.data_ptr()),
        static_cast<const int*>(live_rows.data_ptr()), work_scores.size(1),
        static_cast<const int*>(node_count.data_ptr()),
        static_cast<const int*>(verify_num.data_ptr()));
  };
  // The rows the live ones are among (a graph's padded batch) fix the block size.
  if (work_scores.size(0) <= MAX_BATCH_SIZE / 2) {
    launch(calculate_work_deltas_kernel<MAX_BATCH_SIZE / 2>, MAX_BATCH_SIZE / 2);
  } else {
    launch(calculate_work_deltas_kernel<MAX_BATCH_SIZE>, MAX_BATCH_SIZE);
  }
}

void eagle_harness_draft_depth_update(
    TensorView child_probs, TensorView child_tokens, TensorView expanded, TensorView child_hidden,
    TensorView frontier_tokens, TensorView frontier_scores, TensorView frontier_hidden,
    TensorView frontier_index, TensorView pool_tokens, TensorView pool_scores, TensorView pool_prev,
    TensorView work_scores, TensorView sort_scores, TensorView positions, TensorView slot_starts,
    TensorView lane_mask, TensorView width, int64_t depth, TensorView cumu_count,
    TensorView verify_num, int64_t num_slots) {
  const int64_t rows = child_probs.size(0);
  const int64_t k = child_probs.size(1);
  // Every tensor holds whole rows of the request rows' K lanes, contiguous.
  for (const TensorView* t : {&child_probs, &child_tokens, &expanded, &child_hidden,
                              &frontier_tokens, &frontier_scores, &frontier_hidden,
                              &frontier_index, &pool_tokens, &pool_scores, &pool_prev,
                              &work_scores, &sort_scores, &positions, &slot_starts, &lane_mask}) {
    TVM_FFI_ICHECK(t->IsContiguous() && t->size(0) == rows)
        << "eagle_harness_draft_depth_update: contiguous [rows, ...] tensors";
  }
  for (const TensorView* t : {&child_probs, &child_tokens, &child_hidden, &frontier_hidden,
                              &lane_mask}) {
    TVM_FFI_ICHECK(t->ndim() == 3 && t->size(1) == k)
        << "eagle_harness_draft_depth_update: [rows, lanes, *] tensors";
  }
  TVM_FFI_ICHECK(child_probs.size(2) == k && child_tokens.size(2) == k &&
                 k * k <= DRAFT_DEPTH_MAX_CHILDREN)
      << "eagle_harness_draft_depth_update: " << k << " lanes of " << child_probs.size(2)
      << " children";
  for (const TensorView* t : {&expanded, &frontier_tokens, &frontier_scores, &frontier_index,
                              &positions}) {
    TVM_FFI_ICHECK(t->ndim() == 2 && t->size(1) == k)
        << "eagle_harness_draft_depth_update: frontier";
  }
  const int64_t num_nodes = pool_tokens.size(1);
  TVM_FFI_ICHECK(pool_scores.size(1) == num_nodes && pool_prev.size(1) == num_nodes)
      << "eagle_harness_draft_depth_update: node pool";
  // The caller sizes the pool and lists for every depth and verify size the policy can reach.
  TVM_FFI_ICHECK(depth >= 1 && sort_scores.size(1) <= DRAFT_DEPTH_MAX_VERIFY &&
                 sort_scores.size(1) + k <= work_scores.size(1))
      << "eagle_harness_draft_depth_update: verify lists";
  const int64_t mask_words = lane_mask.size(2);
  TVM_FFI_ICHECK(mask_words <= DRAFT_DEPTH_MAX_MASK_WORDS && num_slots >= 0 &&
                 num_slots <= 64 * mask_words)
      << "eagle_harness_draft_depth_update: lane masks";
  // Hidden rows move as 16-byte vectors.
  const int64_t elem = get_element_size(child_hidden);
  const int64_t row_bytes = child_hidden.size(2) * elem;
  TVM_FFI_ICHECK(child_hidden.dtype() == frontier_hidden.dtype() &&
                 frontier_hidden.size(2) == child_hidden.size(2) && row_bytes % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(child_hidden.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(frontier_hidden.data_ptr()) % 16 == 0)
      << "eagle_harness_draft_depth_update: hidden rows must be 16-byte aligned";
  // The copy gathers rows across threads: it cannot run in place.
  const uintptr_t in_start = reinterpret_cast<uintptr_t>(child_hidden.data_ptr());
  const uintptr_t out_start = reinterpret_cast<uintptr_t>(frontier_hidden.data_ptr());
  const uintptr_t bytes = static_cast<uintptr_t>(rows * k * row_bytes);
  TVM_FFI_ICHECK(in_start + bytes <= out_start || out_start + bytes <= in_start)
      << "eagle_harness_draft_depth_update: child_hidden must not overlap frontier_hidden";
  if (rows == 0) return;
  draft_depth_update_kernel<<<rows, DRAFT_DEPTH_THREADS, 0, get_stream(child_probs.device())>>>(
      static_cast<const float*>(child_probs.data_ptr()),
      static_cast<const int64_t*>(child_tokens.data_ptr()),
      static_cast<const int64_t*>(expanded.data_ptr()),
      static_cast<const uint4*>(child_hidden.data_ptr()),
      static_cast<int64_t*>(frontier_tokens.data_ptr()),
      static_cast<float*>(frontier_scores.data_ptr()),
      static_cast<uint4*>(frontier_hidden.data_ptr()),
      static_cast<int64_t*>(frontier_index.data_ptr()),
      static_cast<int64_t*>(pool_tokens.data_ptr()), static_cast<float*>(pool_scores.data_ptr()),
      static_cast<int64_t*>(pool_prev.data_ptr()), num_nodes,
      static_cast<float*>(work_scores.data_ptr()), work_scores.size(1),
      static_cast<float*>(sort_scores.data_ptr()), sort_scores.size(1),
      static_cast<int64_t*>(positions.data_ptr()), static_cast<int64_t*>(slot_starts.data_ptr()),
      static_cast<int64_t*>(lane_mask.data_ptr()), static_cast<int>(mask_words),
      static_cast<int>(k), static_cast<const int*>(width.data_ptr()), static_cast<int>(depth),
      static_cast<const int*>(cumu_count.data_ptr()), static_cast<const int*>(verify_num.data_ptr()),
      static_cast<int>(num_slots), static_cast<int>(row_bytes / 16));
}

void eagle_harness_stable_topk_desc(TensorView scores, TensorView live_rows, TensorView n,
                                    TensorView k, TensorView out_values,
                                    TensorView out_indices) {
  TVM_FFI_ICHECK(scores.ndim() == 2 && out_values.ndim() == 2 && out_indices.ndim() == 2 &&
                 scores.stride(1) == 1 && out_values.stride(1) == 1 &&
                 out_indices.stride(1) == 1 && out_values.stride(0) == out_indices.stride(0));
  // The caller keeps 1 <= k <= n <= the score columns, and k within the outputs.
  const int64_t rows = scores.size(0);
  TVM_FFI_ICHECK(scores.size(1) <= 1024 && out_values.size(0) == rows &&
                 out_indices.size(0) == rows && out_values.size(1) == out_indices.size(1))
      << "stable_topk_desc needs <= 1024 score columns and [rows, max_k] outputs";
  if (rows == 0) return;
  stable_topk_desc_kernel<<<rows, 512, 0, get_stream(scores.device())>>>(
      static_cast<const float*>(scores.data_ptr()), scores.stride(0),
      static_cast<const int*>(live_rows.data_ptr()), static_cast<const int*>(n.data_ptr()),
      static_cast<const int*>(k.data_ptr()), static_cast<float*>(out_values.data_ptr()),
      static_cast<int64_t*>(out_indices.data_ptr()), out_values.stride(0));
}

void eagle_harness_write_verify_tree(TensorView ranked, TensorView pool_tokens,
                                    TensorView pool_prev, TensorView rows, TensorView input_map,
                                    TensorView parent_map, TensorView live_rows,
                                    TensorView verify_num) {
  TVM_FFI_ICHECK(ranked.stride(1) == 1 && pool_tokens.stride(1) == 1 && pool_prev.stride(1) == 1);
  TVM_FFI_ICHECK_EQ(pool_tokens.stride(0), pool_prev.stride(0));
  // The caller keeps 1 <= verify_num <= the ranked columns, MAX_TREE_NODES and the slots.
  const int64_t num_slots = input_map.size(1);
  const int batch_size = ranked.size(0);
  if (batch_size == 0) return;
  write_verify_tree_kernel<<<batch_size, 64, 0, get_stream(ranked.device())>>>(
      static_cast<const int64_t*>(ranked.data_ptr()), ranked.stride(0),
      static_cast<const int64_t*>(pool_tokens.data_ptr()),
      static_cast<const int64_t*>(pool_prev.data_ptr()), pool_tokens.stride(0),
      static_cast<const int64_t*>(rows.data_ptr()), static_cast<int32_t*>(input_map.data_ptr()),
      input_map.stride(0), static_cast<int32_t*>(parent_map.data_ptr()), parent_map.stride(0),
      static_cast<const int*>(live_rows.data_ptr()), static_cast<const int*>(verify_num.data_ptr()),
      static_cast<int>(num_slots));
}

void eagle_harness_sample_children(TensorView logits, TensorView cand, TensorView tokens,
                                   TensorView scores, TensorView temps, TensorView top_ks,
                                   TensorView top_ps, TensorView q_store, TensorView qid_store,
                                   TensorView node_slot, TensorView node_pos,
                                   Optional<TensorView> width, int64_t slot_base,
                                   TensorView cumu_start, int64_t vocab_limit, int64_t seed,
                                   TensorView round_counter, int64_t launch) {
  TVM_FFI_ICHECK(logits.ndim() == 3 && cand.ndim() == 3 && tokens.ndim() == 3 &&
                 scores.ndim() == 3 && logits.stride(2) == 1 && cand.stride(2) == 1 &&
                 tokens.stride(2) == 1 && scores.stride(2) == 1)
      << "eagle_harness_sample_children: [requests, rows, *] views";
  const int nc = static_cast<int>(q_store.size(2));
  const int64_t requests = tokens.size(0);
  const int64_t rows_per_req = tokens.size(1);
  const int k = static_cast<int>(tokens.size(2));
  const int64_t rows = requests * rows_per_req;
  TVM_FFI_ICHECK(logits.size(2) == nc && cand.size(2) == nc && qid_store.size(2) == nc)
      << "eagle_harness_sample_children: candidate width";
  TVM_FFI_ICHECK(logits.size(0) == requests && cand.size(0) == requests &&
                 scores.size(0) == requests && logits.size(1) == rows_per_req &&
                 cand.size(1) == rows_per_req && scores.size(1) == rows_per_req &&
                 scores.size(2) == k)
      << "eagle_harness_sample_children: expanded rows";
  TVM_FFI_ICHECK(node_slot.size(1) == node_pos.size(1) &&
                 node_slot.stride(0) == node_pos.stride(0) && node_pos.stride(1) == 1)
      << "eagle_harness_sample_children: node tables";
  // The caller keeps cumu_start + width * K within the node tables.
  TVM_FFI_ICHECK(rows_per_req > 0 && slot_base + rows_per_req <= q_store.size(1))
      << "eagle_harness_sample_children: slot range";
  TVM_FFI_ICHECK(nc <= SAMPLE_CHILDREN_THREADS * SAMPLE_CHILDREN_PER_THREAD && rows <= 1024 &&
                 launch >= 0 && launch < 64)
      << "eagle_harness_sample_children: counter layout";
  if (rows == 0) return;
  sample_children_kernel<<<rows, SAMPLE_CHILDREN_THREADS, 0, get_stream(logits.device())>>>(
      static_cast<const float*>(logits.data_ptr()), logits.stride(0), logits.stride(1),
      static_cast<const int64_t*>(cand.data_ptr()), cand.stride(0), cand.stride(1),
      static_cast<int64_t*>(tokens.data_ptr()), tokens.stride(0), tokens.stride(1),
      static_cast<float*>(scores.data_ptr()), scores.stride(0), scores.stride(1),
      static_cast<const float*>(temps.data_ptr()), static_cast<const int32_t*>(top_ks.data_ptr()),
      static_cast<const float*>(top_ps.data_ptr()), static_cast<float*>(q_store.data_ptr()),
      static_cast<int32_t*>(qid_store.data_ptr()), static_cast<int32_t*>(node_slot.data_ptr()),
      static_cast<int32_t*>(node_pos.data_ptr()), static_cast<int>(rows_per_req),
      width.has_value() ? static_cast<const int*>(width.value().data_ptr()) : nullptr,
      static_cast<int>(slot_base), static_cast<const int*>(cumu_start.data_ptr()),
      static_cast<int>(q_store.size(1)),
      static_cast<int>(node_slot.size(1)), nc, k, vocab_limit,
      static_cast<uint64_t>(seed), static_cast<const int64_t*>(round_counter.data_ptr()),
      static_cast<int>(launch));
}

void eagle_harness_tree_reject_sample(TensorView probas, TensorView tokens, TensorView parents,
                                      Optional<TensorView> q_rows, TensorView q_slot,
                                      TensorView q_pos, TensorView q_store, TensorView qid_store,
                                      TensorView coins, TensorView top_ks, TensorView argmax,
                                      TensorView path, TensorView predicts,
                                      TensorView accept_length, TensorView last_row) {
  TVM_FFI_ICHECK(probas.stride(1) == 1 && coins.stride(1) == 1);
  const int batch_size = static_cast<int>(tokens.size(0));
  const int num_nodes = static_cast<int>(tokens.size(1));
  TVM_FFI_ICHECK_LE(num_nodes, MAX_TREE_NODES);
  // The kernel indexes the trees and their q tables with the trees' row width.
  TVM_FFI_ICHECK(tokens.stride(0) == num_nodes && parents.size(1) == num_nodes &&
                 parents.stride(0) == num_nodes && coins.size(1) >= num_nodes &&
                 probas.size(0) >= static_cast<int64_t>(batch_size) * num_nodes)
      << "eagle_harness_tree_reject_sample: row widths";
  TVM_FFI_ICHECK(top_ks.ndim() == 1 && top_ks.stride(0) == 1 && argmax.ndim() == 1 &&
                 argmax.stride(0) == 1 &&
                 top_ks.size(0) >= static_cast<int64_t>(batch_size) * num_nodes &&
                 argmax.size(0) >= static_cast<int64_t>(batch_size) * num_nodes)
      << "eagle_harness_tree_reject_sample: one top-k and argmax per row";
  if (q_rows.has_value()) {
    TVM_FFI_ICHECK(q_slot.size(1) == num_nodes && q_pos.size(1) == num_nodes &&
                   q_slot.stride(0) == num_nodes && q_pos.stride(0) == num_nodes)
        << "eagle_harness_tree_reject_sample: q table row widths";
  }
  if (batch_size == 0) return;
  const bool sampled = q_rows.has_value();
  const int threads = std::min(1024, 32 * num_nodes);
  tree_reject_sample_kernel<<<batch_size, threads, 0, get_stream(probas.device())>>>(
      static_cast<float*>(probas.data_ptr()), probas.stride(0),
      static_cast<const int32_t*>(tokens.data_ptr()), static_cast<const int32_t*>(parents.data_ptr()),
      sampled ? static_cast<const int64_t*>(q_rows.value().data_ptr()) : nullptr,
      static_cast<const int32_t*>(q_slot.data_ptr()), static_cast<const int32_t*>(q_pos.data_ptr()),
      static_cast<const float*>(q_store.data_ptr()),
      static_cast<const int32_t*>(qid_store.data_ptr()), static_cast<const float*>(coins.data_ptr()),
      coins.stride(0), static_cast<const int32_t*>(top_ks.data_ptr()),
      static_cast<const int64_t*>(argmax.data_ptr()), static_cast<int32_t*>(path.data_ptr()),
      static_cast<int32_t*>(predicts.data_ptr()), static_cast<int32_t*>(accept_length.data_ptr()),
      static_cast<int64_t*>(last_row.data_ptr()), num_nodes, static_cast<int>(q_store.size(1)),
      static_cast<int>(q_store.size(2)), static_cast<int>(probas.size(1)));
}

void eagle_harness_residual_sample(TensorView probas, TensorView last_row, TensorView coins,
                                   TensorView top_ks, TensorView argmax, TensorView accept_length,
                                   TensorView predicts, int64_t num_nodes) {
  TVM_FFI_ICHECK_EQ(probas.stride(1), 1);
  const int batch_size = static_cast<int>(last_row.size(0));
  TVM_FFI_ICHECK(predicts.size(0) >= batch_size * num_nodes && accept_length.size(0) >= batch_size)
      << "eagle_harness_residual_sample: output rows";
  // last_row indexes probas rows, and so the per-row top-k and argmax.
  TVM_FFI_ICHECK(top_ks.ndim() == 1 && top_ks.stride(0) == 1 && argmax.ndim() == 1 &&
                 argmax.stride(0) == 1 && top_ks.size(0) >= probas.size(0) &&
                 argmax.size(0) >= probas.size(0))
      << "eagle_harness_residual_sample: one top-k and argmax per row";
  if (batch_size == 0) return;
  // The opt-in to over 48 KB of dynamic shared memory is per device: once each.
  static std::atomic<uint64_t> opted_in{0};
  const int device = probas.device().device_id;
  TVM_FFI_ICHECK(device < 64) << "eagle_harness_residual_sample: device " << device;
  if (!((opted_in.load(std::memory_order_relaxed) >> device) & 1)) {
    const cudaError_t smem_status = cudaFuncSetAttribute(
        residual_sample_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        RESIDUAL_SAMPLE_SMEM_BYTES);
    TVM_FFI_ICHECK(smem_status == cudaSuccess)
        << "eagle_harness_residual_sample: " << cudaGetErrorString(smem_status);
    opted_in.fetch_or(uint64_t{1} << device, std::memory_order_relaxed);
  }
  residual_sample_kernel<<<batch_size, RESIDUAL_SAMPLE_THREADS, RESIDUAL_SAMPLE_SMEM_BYTES,
                           get_stream(probas.device())>>>(
      static_cast<const float*>(probas.data_ptr()), probas.stride(0),
      static_cast<const int64_t*>(last_row.data_ptr()), static_cast<const float*>(coins.data_ptr()),
      static_cast<const int32_t*>(top_ks.data_ptr()), static_cast<const int64_t*>(argmax.data_ptr()),
      static_cast<const int32_t*>(accept_length.data_ptr()),
      static_cast<int32_t*>(predicts.data_ptr()), static_cast<int>(num_nodes),
      static_cast<int>(probas.size(1)));
}

void eagle_harness_publish_draft_q(TensorView ranked, TensorView node_slot, TensorView node_pos,
                                   TensorView q_round, TensorView qid_round, TensorView rows,
                                   TensorView q_tab, TensorView qid_tab, TensorView slot_tab,
                                   TensorView pos_tab, TensorView live_rows,
                                   TensorView verify_num) {
  TVM_FFI_ICHECK(ranked.stride(1) == 1 && node_slot.stride(1) == 1);
  TVM_FFI_ICHECK_EQ(node_slot.stride(0), node_pos.stride(0));
  // The caller keeps verify_num within the ranked columns and the table slots.
  TVM_FFI_ICHECK(slot_tab.size(1) == pos_tab.size(1))
      << "eagle_harness_publish_draft_q: node tables";
  const int batch_size = static_cast<int>(ranked.size(0));
  const int64_t store_size = q_round.size(1) * q_round.size(2);
  TVM_FFI_ICHECK_EQ(q_tab.size(1) * q_tab.size(2), store_size);
  if (batch_size == 0) return;
  publish_draft_q_kernel<<<batch_size, 256, 0, get_stream(ranked.device())>>>(
      static_cast<const int64_t*>(ranked.data_ptr()), ranked.stride(0),
      static_cast<const int32_t*>(node_slot.data_ptr()),
      static_cast<const int32_t*>(node_pos.data_ptr()), node_slot.stride(0),
      static_cast<const float*>(q_round.data_ptr()), static_cast<const int32_t*>(qid_round.data_ptr()),
      static_cast<const int64_t*>(rows.data_ptr()), static_cast<float*>(q_tab.data_ptr()),
      static_cast<int32_t*>(qid_tab.data_ptr()), static_cast<int32_t*>(slot_tab.data_ptr()),
      static_cast<int32_t*>(pos_tab.data_ptr()), static_cast<const int*>(live_rows.data_ptr()),
      static_cast<const int*>(verify_num.data_ptr()), static_cast<int>(slot_tab.size(1)),
      store_size);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_update_node_data, eagle_harness_update_node_data);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_write_verify_tree, eagle_harness_write_verify_tree);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_work_deltas, eagle_harness_work_deltas);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_draft_depth_update, eagle_harness_draft_depth_update);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_stable_topk_desc, eagle_harness_stable_topk_desc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_sample_children, eagle_harness_sample_children);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_tree_reject_sample, eagle_harness_tree_reject_sample);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_residual_sample, eagle_harness_residual_sample);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_publish_draft_q, eagle_harness_publish_draft_q);
