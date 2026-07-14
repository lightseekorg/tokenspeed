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

// Fused TopK + TopP renorm. Picks one of three branches per row, but launches
// the same kernels every call so the host-side path is deterministic and
// CUDA-graph capturable. Per-row dispatch happens inside the apply kernel via
// topKs[row].

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <vector>

#include "air_top_p.cuh"
#include "air_topk_stable.cuh"
#include "fused_topk_topp.h"

namespace fused_topk_topp {

// PDL helper. B200 (sm_100) supports cudaLaunchAttributeProgrammaticStream
// Serialization — the next kernel can run its prologue (allocate resources,
// fetch args) in parallel with the previous kernel's epilogue. Saves a
// fraction of each launch's overhead. Used for every kernel in this pipeline
// when the runtime PDL toggle is enabled.
template <typename KernelFunc, typename... Args>
static inline void launchKernel(bool enable_pdl, KernelFunc kernel, dim3 grid, dim3 block,
                                size_t smem, cudaStream_t stream, Args... args) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = enable_pdl ? attr : nullptr;
    config.numAttrs = enable_pdl ? 1 : 0;
    cudaLaunchKernelEx(&config, kernel, args...);
}

// Per-row init: mark `counter.skip = 1` for rows whose top_k exceeds the
// MAX_K bound (mode 3.2). The radix kernels in air_topk_stable.cuh check this
// flag at the top and return immediately, so the entire stage 1 pipeline does
// no HBM read or histogram work for those rows.
template <typename T, typename IdxT>
__global__ void initTopKSkipKernel(nv::air_topk_stable::Counter<T, IdxT>* counters,
                                   int batch_size,
                                   int32_t const* __restrict__ top_k_arr,
                                   int max_topk) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    counters[b].skip = (top_k_arr[b] > max_topk) ? 1 : 0;
}

// Unified per-row init kernel: combines the topk-stage skip-flag setter, the
// topk workspace zero-fill, and the topp-stage init kernel into a single
// launch. Saves 2 kernel launches plus a cudaMemsetAsync.
//
// Block layout: dim3(batch_size) × dim3(256). Each block handles one row.
template <typename T, typename IdxT, int NUM_TOPP_BUCKETS, int TOPK_NUM_BUCKETS>
__launch_bounds__(256) __global__ void unifiedInitKernel(
    nv::air_topk_stable::Counter<T, IdxT>* topk_counters,
    IdxT* topk_histograms,
    air_top_p::Counter<T>* topp_counters,
    air_top_p::HisT<T>* topp_histograms,
    air_top_p::IdxT* topp_count_histograms,
    int batch_size,
    int vocab_size,
    T const* __restrict__ probs,
    float const* __restrict__ top_p_arr,
    int32_t const* __restrict__ top_k_arr,
    int32_t max_topk) {
    int const b = blockIdx.x;
    if (b >= batch_size) return;
    int32_t const k = top_k_arr[b];
    float const p = top_p_arr[b];
    bool const topp_active = (k > max_topk);

    // Set topk counter fields (thread 0). The radix kernels reset most fields
    // themselves between passes — only `skip` matters for short-circuiting.
    if (threadIdx.x == 0) {
        auto* tkc = topk_counters + b;
        tkc->k = 0;
        tkc->len = 0;
        tkc->previous_len = 0;
        tkc->kth_value_bits = 0;
        tkc->skip = topp_active ? 1 : 0;
        tkc->filter_cnt = 0;
        tkc->out_cnt = 0;
        tkc->out_back_cnt = 0;
        tkc->finished_block_cnt = 0;
    }

    // Set topp counter fields (thread 0).
    if (threadIdx.x == 0) {
        auto* tpc = topp_counters + b;
        tpc->in = probs + static_cast<size_t>(b) * vocab_size;
        tpc->oriLen = vocab_size;
        tpc->len = topp_active ? vocab_size : 0;
        tpc->previousLen = vocab_size;
        tpc->p = topp_active ? p : 0.0f;
        tpc->totalSum = 0.0f;
        tpc->sum = 0;
        tpc->kthValueBits = 0;
        tpc->finishedBlockCnt = 0;
        tpc->filterCnt = 0;
    }

    // Zero topk histograms (per row).
    IdxT* tk_hist = topk_histograms + static_cast<size_t>(b) * TOPK_NUM_BUCKETS;
    for (int i = threadIdx.x; i < TOPK_NUM_BUCKETS; i += blockDim.x) {
        tk_hist[i] = 0;
    }

    // Zero topp histograms (per row).
    air_top_p::HisT<T>* tp_hist =
        topp_histograms + static_cast<size_t>(b) * NUM_TOPP_BUCKETS;
    air_top_p::IdxT* tp_cnt_hist =
        topp_count_histograms + static_cast<size_t>(b) * NUM_TOPP_BUCKETS;
    for (int i = threadIdx.x; i < NUM_TOPP_BUCKETS; i += blockDim.x) {
        tp_hist[i] = 0;
        tp_cnt_hist[i] = 0;
    }
}

// Compute workspace pointers for the air_topk multi-block path.
template <typename T, typename IdxT, int BitsPerPass>
static void airTopKResolveWorkspace(void* buf, IdxT len, int batch_size,
                                    nv::air_topk_stable::Counter<T, IdxT>*& counters,
                                    IdxT*& histograms, T*& buf1, IdxT*& idx_buf1,
                                    T*& buf2, IdxT*& idx_buf2) {
    using Counter = nv::air_topk_stable::Counter<T, IdxT>;
    constexpr int num_buckets = nv::air_topk_stable::calc_num_buckets<BitsPerPass>();
    IdxT const len_candidates = nv::air_topk_stable::calc_buf_len<T>(len);
    std::vector<size_t> sizes = {
        sizeof(Counter) * static_cast<size_t>(batch_size),
        sizeof(IdxT) * static_cast<size_t>(num_buckets) * static_cast<size_t>(batch_size),
        sizeof(T) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
        sizeof(IdxT) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
        sizeof(T) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
        sizeof(IdxT) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
    };
    auto ptrs = nv::calc_aligned_pointers(buf, sizes);
    counters = static_cast<Counter*>(ptrs[0]);
    histograms = static_cast<IdxT*>(ptrs[1]);
    buf1 = static_cast<T*>(ptrs[2]);
    idx_buf1 = static_cast<IdxT*>(ptrs[3]);
    buf2 = static_cast<T*>(ptrs[4]);
    idx_buf2 = static_cast<IdxT*>(ptrs[5]);
}

// Multi-block radix top-K launcher that mirrors air_topk_stable::
// standalone_stable_radix_topk_ but inserts initTopKSkipKernel between the
// workspace memset and the first radix pass. We can't use the cuVS standalone
// directly because it does the memset internally and gives no hook between
// memset and pass 0; replicating its ~30 lines of host-side logic is the
// simplest way to slot the init kernel in.
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
static void airTopKMultiBlockWithSkip(void* buf, size_t& buf_size, T const* in, int batch_size,
                                      IdxT len, IdxT k, T* out, IdxT* out_idx,
                                      bool select_min, bool fused_last_filter, unsigned grid_dim,
                                      int32_t const* top_k_arr, int max_topk,
                                      cudaStream_t stream, bool enable_pdl,
                                      bool skip_init = false) {
    static_assert(nv::air_topk_stable::calc_num_passes<T, BitsPerPass>() > 1);
    constexpr int num_buckets = nv::air_topk_stable::calc_num_buckets<BitsPerPass>();

    using Counter = nv::air_topk_stable::Counter<T, IdxT>;

    IdxT const len_candidates = nv::air_topk_stable::calc_buf_len<T>(len);
    std::vector<size_t> sizes = {
        sizeof(Counter) * static_cast<size_t>(batch_size),
        sizeof(IdxT) * static_cast<size_t>(num_buckets) * static_cast<size_t>(batch_size),
        sizeof(T) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
        sizeof(IdxT) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
        sizeof(T) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
        sizeof(IdxT) * static_cast<size_t>(len_candidates) * static_cast<size_t>(batch_size),
    };
    size_t const total_size = nv::calc_aligned_size(sizes);
    if (!buf) {
        buf_size = total_size;
        return;
    }

    auto ptrs = nv::calc_aligned_pointers(buf, sizes);
    auto* counters = static_cast<Counter*>(ptrs[0]);
    auto* histograms = static_cast<IdxT*>(ptrs[1]);
    T* buf1 = static_cast<T*>(ptrs[2]);
    IdxT* idx_buf1 = static_cast<IdxT*>(ptrs[3]);
    T* buf2 = static_cast<T*>(ptrs[4]);
    IdxT* idx_buf2 = static_cast<IdxT*>(ptrs[5]);

    if (!skip_init) {
        // Zero counters + histograms (skip flag included → defaults to "no skip").
        cudaMemsetAsync(buf, 0,
                        static_cast<char*>(ptrs[2]) - static_cast<char*>(ptrs[0]), stream);

        // Mark skip rows. Optional: when top_k_arr is null we keep the
        // memset-default of "no skip" for every row (equivalent to standalone).
        if (top_k_arr) {
            int threads = 128;
            int blocks = (batch_size + threads - 1) / threads;
            launchKernel(enable_pdl, initTopKSkipKernel<T, IdxT>, dim3(blocks),
                         dim3(threads), 0, stream, counters, batch_size, top_k_arr, max_topk);
        }
    }

    // Radix passes. Same dispatch as standalone_stable_radix_topk_.
    T const* in_buf = nullptr;
    IdxT const* in_idx_buf = nullptr;
    T* out_buf = nullptr;
    IdxT* out_idx_buf = nullptr;
    dim3 blocks(grid_dim, static_cast<unsigned>(batch_size));
    // Pass 2 (the last pass) reads only the small per-row survivors buffer,
    // so multi-block doesn't help — each row's last block ends up doing all
    // the histogram scan + bucket select + last-filter work anyway. Drop to
    // grid_dim=1 for pass 2 to skip the inter-block atomic and synchronization
    // overhead in the existing radix_kernel.
    dim3 last_pass_blocks(1U, static_cast<unsigned>(batch_size));
    constexpr int num_passes = nv::air_topk_stable::calc_num_passes<T, BitsPerPass>();

    auto kernel =
        nv::air_topk_stable::radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, true>;

    for (int pass = 0; pass < num_passes; ++pass) {
        nv::air_topk_stable::set_buf_pointers(in, static_cast<IdxT const*>(nullptr), buf1,
                                              idx_buf1, buf2, idx_buf2, pass, in_buf,
                                              in_idx_buf, out_buf, out_idx_buf);
        if (fused_last_filter && pass == num_passes - 1) {
            kernel = nv::air_topk_stable::
                radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, true>;
        }
        dim3 pass_blocks = (pass == num_passes - 1) ? last_pass_blocks : blocks;
        launchKernel(enable_pdl, kernel, pass_blocks, dim3(BlockSize), 0, stream, in,
                     static_cast<IdxT const*>(nullptr), in_buf, in_idx_buf, out_buf,
                     out_idx_buf, out, out_idx, counters, histograms, len, k, select_min, pass);
    }

    if (!fused_last_filter) {
        launchKernel(enable_pdl,
                     nv::air_topk_stable::last_filter_kernel<T, IdxT, BitsPerPass, true>,
                     blocks, dim3(BlockSize), 0, stream, in,
                     static_cast<IdxT const*>(nullptr), out_buf, out_idx_buf, out, out_idx, len,
                     k, counters, select_min);
    }
}

// air_topk wrapper that always uses fused_last_filter=true and prioritizes
// smaller indices on ties (matches baseline's deterministic top-k semantics).
// Always goes through the multi-block path so the workspace layout is
// predictable (the unified init kernel writes to a fixed multi-block layout
// regardless of V). For production V=163840 multi-block is always optimal;
// for small V (sanity check), multi-block with grid_dim=1 is correct.
template <typename T, typename IdxT>
static void air_topk_11bits_fused_last(void* buf, size_t& buf_size, T const* in, int batch_size,
                                       IdxT len, IdxT k, T* out, IdxT* out_idx,
                                       int32_t const* top_k_arr, int max_topk,
                                       cudaStream_t stream, bool enable_pdl = true,
                                       bool skip_init = false) {
    constexpr int block_dim = 512;
    constexpr int BitsPerPass = 11;
    constexpr bool greater = true;     // largest values
    constexpr bool fused_last_filter = true;

    int sm_cnt = 0, dev = 0;
    if (buf) {
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev);
    } else {
        // Use a representative SM count for workspace sizing query.
        sm_cnt = 132;
    }
    unsigned grid_dim =
        nv::air_topk_stable::calc_grid_dim<T, IdxT, BitsPerPass, block_dim>(batch_size, len,
                                                                            sm_cnt);
    if (grid_dim == 0U) grid_dim = 1U;
    airTopKMultiBlockWithSkip<T, IdxT, BitsPerPass, block_dim>(
        buf, buf_size, in, batch_size, len, k, out, out_idx,
        !greater, fused_last_filter, grid_dim, top_k_arr, max_topk, stream, enable_pdl,
        skip_init);
}

static size_t airTopKWorkspaceBytes(int batchSize, int vocabSize) {
    size_t ws = 0;
    air_topk_11bits_fused_last<float, int32_t>(nullptr, ws, nullptr, batchSize, vocabSize,
                                              K_TOPK_MAX, nullptr, nullptr,
                                              /*top_k_arr=*/nullptr,
                                              /*max_topk=*/K_TOPK_MAX, 0);
    return ws;
}

// Workspace layout (all 256B-aligned):
//   [airTopkWS] [topKVals: bs*K_TOPK_MAX float] [topKIdx: bs*K_TOPK_MAX int32]
//   [airTopPWS] (includes its own counters + histograms + buf1 + buf2)
size_t getWorkspaceSize(SizeType32 batchSize, SizeType32 vocabSize) {
    std::vector<size_t> sizes = {
        airTopKWorkspaceBytes(batchSize, vocabSize),
        sizeof(float) * static_cast<size_t>(batchSize) * K_TOPK_MAX,
        sizeof(int32_t) * static_cast<size_t>(batchSize) * K_TOPK_MAX,
        air_top_p::getWorkspaceBytes<float>(batchSize, vocabSize),
    };
    return nv::calc_aligned_size(sizes);
}

// Per-row apply kernel — branches on top_k[row]:
//   K_eff <= MAX_K → mode 3.1 / 3.3: sort the K top values, find min(K, P)
//     cutoff, scatter renormalized values into outProbs[row].
//   K_eff >  MAX_K → mode 3.2: use the top-p threshold left in the
//     air_top_p counter, scan the row, renormalize, write outProbs[row].
//
// outProbs is pre-zeroed by an asynchronous memset on a side stream — both
// branches write only the kept positions.
// ROWS_ALIGNED: the caller guarantees every row base of `probs`/`out_probs` is
// 16B-aligned (both bases 16B-aligned and vocab_size % 4 == 0), so the top-p
// V-scan can use float4 loads/stores with no peel. Decided host-side in
// invokeFusedTopKTopP; see the mode 3.2 branch.
template <int BLOCK_SIZE, int ITEMS_PER_THREAD, bool ROWS_ALIGNED>
__launch_bounds__(BLOCK_SIZE) __global__ void applyKernel(
    float const* __restrict__ probs,
    float const* __restrict__ top_k_vals,
    int32_t const* __restrict__ top_k_idx,
    int32_t const* __restrict__ top_ks,
    float const* __restrict__ top_ps,
    air_top_p::Counter<float>* __restrict__ topp_counters,
    float* __restrict__ out_probs,
    int32_t vocab_size,
    int32_t max_k,
    bool enable_pdl) {
    // PDL: wait for preceding stream work before reading top-k outputs.
    if (enable_pdl) {
        cudaGridDependencySynchronize();
    }

    constexpr int MAX_K = BLOCK_SIZE * ITEMS_PER_THREAD;
    const int b = blockIdx.x;
    const int32_t k_raw = top_ks[b];
    const float p = top_ps[b];

    // Packed (val ↓, idx ↑) uint64 sort. Upper 32 bits = twiddle_in(val, false)
    // so ascending uint order == descending val order; lower 32 bits = idx so
    // ties on val break by smaller idx first. This makes the sort tie-break
    // deterministic (matches baseline `is_deterministic=True`), independent
    // of how air_topk_stable's strictly-greater branch happened to order tied
    // items via atomicAdd. The smem footprint matches what the dummy uint64
    // pad used to provide, so SM occupancy is unchanged (still 1 block/SM,
    // which mode 3.2's V-scan needs to keep HBM bandwidth).
    using BlockRadixSort = cub::BlockRadixSort<uint64_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                                cub::NullType, /*RADIX_BITS=*/6>;
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;

    __shared__ union {
        typename BlockRadixSort::TempStorage sort;
        typename BlockScan::TempStorage scan;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    if (k_raw <= max_k) {
        // ── Mode 3.1 / 3.3: top-K (post-process for top-P) ──────────────────
        const int k = max(1, min(k_raw, max_k));

        __shared__ float s_vals[MAX_K];
        __shared__ int32_t s_idx[MAX_K];
        __shared__ float s_cumsum[MAX_K];
        __shared__ int s_cutoff_j;
        __shared__ float s_inv_factor;

        // Build packed keys, sort ascending → descending val, ascending idx on ties.
        uint64_t t_keys[ITEMS_PER_THREAD];
        const int row_base = b * max_k;
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            const int pos = threadIdx.x * ITEMS_PER_THREAD + i;
            float val;
            int32_t idx;
            if (pos < max_k) {
                val = top_k_vals[row_base + pos];
                idx = top_k_idx[row_base + pos];
            } else {
                val = -FLT_MAX;
                idx = INT32_MAX;
            }
            uint32_t v_bits = nv::air_topk_stable::twiddle_in<float>(val, /*select_min=*/false);
            t_keys[i] = (static_cast<uint64_t>(v_bits) << 32) | static_cast<uint32_t>(idx);
        }

        BlockRadixSort(temp_storage.sort).Sort(t_keys);
        __syncthreads();

        // Unpack into thread-local t_vals (for the upcoming BlockScan) and
        // mirror to smem so the scatter step can read by sorted position.
        float t_vals[ITEMS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            const uint64_t key = t_keys[i];
            const uint32_t v_bits = static_cast<uint32_t>(key >> 32);
            const int32_t idx = static_cast<int32_t>(static_cast<uint32_t>(key));
            const float val =
                nv::air_topk_stable::twiddle_out<float>(v_bits, /*select_min=*/false);
            t_vals[i] = val;
            const int pos = threadIdx.x * ITEMS_PER_THREAD + i;
            s_vals[pos] = val;
            s_idx[pos] = idx;
        }

        float t_scan[ITEMS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            const int pos = threadIdx.x * ITEMS_PER_THREAD + i;
            t_scan[i] = (pos < k) ? t_vals[i] : 0.0f;
        }
        __syncthreads();
        BlockScan(temp_storage.scan).InclusiveSum(t_scan, t_scan);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            const int pos = threadIdx.x * ITEMS_PER_THREAD + i;
            s_cumsum[pos] = t_scan[i];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const float sum_topk = s_cumsum[k - 1];
            const float threshold = p * sum_topk;
            int j = k - 1;
            for (int i = 0; i < k; ++i) {
                if (s_cumsum[i] >= threshold) {
                    j = i;
                    break;
                }
            }
            s_cutoff_j = j;
            const float denom = s_cumsum[j];
            s_inv_factor = (denom > 1e-30f) ? 1.0f / denom : 0.0f;
        }
        __syncthreads();

        const int cutoff = s_cutoff_j;
        const float inv = s_inv_factor;
        float* out_row = out_probs + b * vocab_size;
        for (int i = threadIdx.x; i <= cutoff; i += BLOCK_SIZE) {
            const int idx = s_idx[i];
            if (static_cast<unsigned>(idx) < static_cast<unsigned>(vocab_size)) {
                out_row[idx] = s_vals[i] * inv;
            }
        }
    } else {
        // ── Mode 3.2: top-P only (radix top-p threshold) ────────────────────
        // Vectorized V-scan with float4: each thread reads 16B per iteration,
        // 4x fewer transactions than scalar. A float4 access must be naturally
        // aligned, and a 16B-aligned row base is NOT guaranteed:
        //   1. `probs`/`out_probs` may be views whose data_ptr is only 4B-aligned
        //      (any slice or offset gather off a larger buffer), and
        //   2. vocab_size % 4 != 0 pushes every row b >= 1 off a 16B boundary
        //      even when the base allocation is 256B-aligned.
        // Either one used to fault with "misaligned address".
        //
        // Whether any row can be unaligned is a property of the launch (base
        // pointers + vocab_size), so the host picks the instantiation and
        // ROWS_ALIGNED is a compile-time constant here. Keeping it compile-time
        // matters: a runtime branch leaves the unaligned code in the aligned
        // path's register footprint and costs ~2.5% occupancy even on rows that
        // never execute it.
        const float threshold =
            air_top_p::twiddleOut<float>(topp_counters[b].kthValueBits, false);

        float const* in_row = probs + b * vocab_size;
        float* out_row = out_probs + b * vocab_size;

        if constexpr (ROWS_ALIGNED) {
            // Every row base is 16B-aligned and in phase: no peel, no tail.
            const int vec_count = vocab_size >> 2;
            float4 const* in_row_v = reinterpret_cast<float4 const*>(in_row);
            float4* out_row_v = reinterpret_cast<float4*>(out_row);

            float thread_sum = 0.0f;
            for (int i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
                float4 v4 = in_row_v[i];
                if (v4.x >= threshold) thread_sum += v4.x;
                if (v4.y >= threshold) thread_sum += v4.y;
                if (v4.z >= threshold) thread_sum += v4.z;
                if (v4.w >= threshold) thread_sum += v4.w;
            }
            float row_sum = BlockReduce(temp_storage.reduce).Sum(thread_sum);

            __shared__ float s_inv;
            if (threadIdx.x == 0) {
                s_inv = (row_sum > 1e-30f) ? (1.0f / row_sum) : 0.0f;
            }
            __syncthreads();
            const float inv = s_inv;

            // Only write the float4 if at least one lane is non-zero -- keeps the
            // write traffic ~ kept positions x 4B in the common sharp-distribution
            // case (most float4s have all 4 lanes below threshold and stay at
            // their memset-0 value, set on the side stream before this kernel).
            for (int i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
                float4 v4 = in_row_v[i];
                float4 o4;
                o4.x = (v4.x >= threshold) ? v4.x * inv : 0.0f;
                o4.y = (v4.y >= threshold) ? v4.y * inv : 0.0f;
                o4.z = (v4.z >= threshold) ? v4.z * inv : 0.0f;
                o4.w = (v4.w >= threshold) ? v4.w * inv : 0.0f;
                if (o4.x != 0.0f || o4.y != 0.0f || o4.z != 0.0f || o4.w != 0.0f) {
                    out_row_v[i] = o4;
                }
            }
        } else {
            // Some row may be unaligned. Rather than drop the whole row to a
            // scalar loop (measured ~2x slower on this kernel -- the two V x 4B
            // read passes dominate it, and with one block per row the slowest row
            // gates the launch), peel it: a scalar prologue of 0-3 floats walks
            // `in_row` up to a 16B boundary, the bulk stays vectorized, and a
            // scalar tail finishes. Unaligned rows keep full read bandwidth.
            const uintptr_t in_addr = reinterpret_cast<uintptr_t>(in_row);
            const uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_row);

            // A float tensor is always 4B-aligned; if that ever fails, the row
            // cannot be vectorized at all -- peel all of it.
            const bool elem_aligned = ((in_addr | out_addr) & 0x3u) == 0;
            const int head_want =
                elem_aligned ? static_cast<int>(((16u - (in_addr & 15u)) & 15u) >> 2)
                             : vocab_size;
            const int head_end = min(head_want, vocab_size);       // end of prologue
            const int vec_count = (vocab_size - head_end) >> 2;    // # of float4 lanes
            const int tail_start = head_end + vec_count * 4;       // start of tail

            // Stores can only ride along when `out_row` shares `in_row`'s 16B
            // phase; otherwise no single peel aligns both. Then keep the
            // vectorized loads and write kept lanes scalar (writes are sparse).
            const bool store_vec = elem_aligned && ((in_addr ^ out_addr) & 15u) == 0;

            float4 const* in_row_v =
                reinterpret_cast<float4 const*>(in_row + head_end);
            float4* out_row_v = reinterpret_cast<float4*>(out_row + head_end);

            float thread_sum = 0.0f;
            for (int i = threadIdx.x; i < head_end; i += BLOCK_SIZE) {
                float v = in_row[i];
                if (v >= threshold) thread_sum += v;
            }
            for (int i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
                float4 v4 = in_row_v[i];
                if (v4.x >= threshold) thread_sum += v4.x;
                if (v4.y >= threshold) thread_sum += v4.y;
                if (v4.z >= threshold) thread_sum += v4.z;
                if (v4.w >= threshold) thread_sum += v4.w;
            }
            for (int i = tail_start + threadIdx.x; i < vocab_size; i += BLOCK_SIZE) {
                float v = in_row[i];
                if (v >= threshold) thread_sum += v;
            }
            float row_sum = BlockReduce(temp_storage.reduce).Sum(thread_sum);

            __shared__ float s_inv;
            if (threadIdx.x == 0) {
                s_inv = (row_sum > 1e-30f) ? (1.0f / row_sum) : 0.0f;
            }
            __syncthreads();
            const float inv = s_inv;

            for (int i = threadIdx.x; i < head_end; i += BLOCK_SIZE) {
                float v = in_row[i];
                if (v >= threshold) out_row[i] = v * inv;
            }
            for (int i = threadIdx.x; i < vec_count; i += BLOCK_SIZE) {
                float4 v4 = in_row_v[i];
                float4 o4;
                o4.x = (v4.x >= threshold) ? v4.x * inv : 0.0f;
                o4.y = (v4.y >= threshold) ? v4.y * inv : 0.0f;
                o4.z = (v4.z >= threshold) ? v4.z * inv : 0.0f;
                o4.w = (v4.w >= threshold) ? v4.w * inv : 0.0f;
                if (o4.x != 0.0f || o4.y != 0.0f || o4.z != 0.0f || o4.w != 0.0f) {
                    if (store_vec) {
                        out_row_v[i] = o4;
                    } else {
                        float* o = out_row + head_end + i * 4;
                        if (o4.x != 0.0f) o[0] = o4.x;
                        if (o4.y != 0.0f) o[1] = o4.y;
                        if (o4.z != 0.0f) o[2] = o4.z;
                        if (o4.w != 0.0f) o[3] = o4.w;
                    }
                }
            }
            for (int i = tail_start + threadIdx.x; i < vocab_size; i += BLOCK_SIZE) {
                float v = in_row[i];
                if (v >= threshold) out_row[i] = v * inv;
            }
        }
        (void)p;  // unused in this branch
    }
}

void invokeFusedTopKTopP(float const* probs, SizeType32 const* topKs, float const* topPs,
                        float* outProbs, void* workspace, SizeType32 batchSize,
                        SizeType32 vocabSize, cudaStream_t mainStream,
                        cudaStream_t memsetStream, bool enable_pdl) {
    // ── Workspace partitioning ──────────────────────────────────────────────
    size_t airTopkWS = airTopKWorkspaceBytes(batchSize, vocabSize);
    std::vector<size_t> sizes = {
        airTopkWS,
        sizeof(float) * static_cast<size_t>(batchSize) * K_TOPK_MAX,
        sizeof(int32_t) * static_cast<size_t>(batchSize) * K_TOPK_MAX,
        air_top_p::getWorkspaceBytes<float>(batchSize, vocabSize),
    };
    auto ptrs = nv::calc_aligned_pointers(workspace, sizes);
    void* topkWS = ptrs[0];
    float* topKVals = static_cast<float*>(ptrs[1]);
    int32_t* topKIdx = static_cast<int32_t*>(ptrs[2]);
    void* toppWS = ptrs[3];

    // ── Stage 0a: pull the side stream into any in-flight CUDA-graph capture
    //              BEFORE the first side-stream op. A capture only includes
    //              work issued on streams already joined to the capture; if we
    //              memset outProbs on a still-unjoined side stream, the memset
    //              runs eagerly at capture time and never replays — leaving
    //              outProbs polluted with the previous replay's kept-positions
    //              on every subsequent replay. The fork-event also gives the
    //              side stream a happens-after on whatever was queued on the
    //              main stream up to this point (the apply kernel from the
    //              previous call, if any) so its memset doesn't race a stale
    //              reader.
    const cudaStream_t msStream = memsetStream ? memsetStream : mainStream;
    const bool sideStreamActive = memsetStream && memsetStream != mainStream;
    if (sideStreamActive) {
        cudaEvent_t forkEvent;
        cudaEventCreateWithFlags(&forkEvent, cudaEventDisableTiming);
        cudaEventRecord(forkEvent, mainStream);
        cudaStreamWaitEvent(msStream, forkEvent, 0);
        cudaEventDestroy(forkEvent);
    }

    // ── Stage 0b: zero-fill outProbs on side stream (overlaps with stage 1) ─
    cudaMemsetAsync(outProbs, 0,
                    sizeof(float) * static_cast<size_t>(batchSize) * vocabSize, msStream);

    // ── Stage 1a: unified init kernel — sets up topk skip flags, topp counter
    //              fields, and zeros both stages' histograms in one launch.
    constexpr int TOPK_NUM_BUCKETS = nv::air_topk_stable::calc_num_buckets<11>();  // 2048

    nv::air_topk_stable::Counter<float, int32_t>* topkCounters = nullptr;
    int32_t* topkHistograms = nullptr;
    float* topkBuf1 = nullptr;
    int32_t* topkIdxBuf1 = nullptr;
    float* topkBuf2 = nullptr;
    int32_t* topkIdxBuf2 = nullptr;
    airTopKResolveWorkspace<float, int32_t, 11>(topkWS, vocabSize, batchSize, topkCounters,
                                                topkHistograms, topkBuf1, topkIdxBuf1,
                                                topkBuf2, topkIdxBuf2);

    air_top_p::Counter<float>* toppCounters = nullptr;
    air_top_p::HisT<float>* toppHistograms = nullptr;
    air_top_p::IdxT* toppCountHistograms = nullptr;
    float* toppBuf1 = nullptr;
    float* toppBuf2 = nullptr;
    air_top_p::resolveWorkspace<float>(batchSize, vocabSize, toppWS, toppCounters,
                                       toppHistograms, toppCountHistograms, toppBuf1, toppBuf2);

    launchKernel(enable_pdl,
                 unifiedInitKernel<float, int32_t, air_top_p::NUM_BUCKETS, TOPK_NUM_BUCKETS>,
                 dim3(batchSize), dim3(256), 0, mainStream, topkCounters, topkHistograms,
                 toppCounters, toppHistograms, toppCountHistograms, batchSize, vocabSize, probs,
                 topPs, topKs, K_TOPK_MAX);

    // ── Stage 2 on side stream (parallel with stage 1 on main) ──────────────
    // Run the topp radix on the side stream so it overlaps with the topk
    // radix on the main stream. After both finish, the apply kernel runs on
    // the main stream after a stream-event sync. The side stream has to wait
    // for `unifiedInitKernel` to finish (it sets up the topp counters), so
    // we record an event on main here and gate the side stream on it.
    if (sideStreamActive) {
        cudaEvent_t initEvent;
        cudaEventCreateWithFlags(&initEvent, cudaEventDisableTiming);
        cudaEventRecord(initEvent, mainStream);
        cudaStreamWaitEvent(msStream, initEvent, 0);
        cudaEventDestroy(initEvent);
    }
    air_top_p::launchRadixOnly<float>(toppCounters, toppHistograms, toppCountHistograms,
                                     toppBuf1, toppBuf2, batchSize, vocabSize, msStream);

    // ── Stage 1b: deterministic radix top-K on main stream ──────────────────
    // K is pinned to K_TOPK_MAX so the grid configuration is fixed regardless
    // of the per-row top_k values. Per-row short-circuit: rows in mode 3.2
    // (k_user > K_TOPK_MAX) get counter.skip=1 from the unified init, so
    // every block of that grid column returns immediately — no HBM read, no
    // histogram work.
    air_topk_11bits_fused_last<float, int32_t>(topkWS, airTopkWS, probs, batchSize, vocabSize,
                                              K_TOPK_MAX, topKVals, topKIdx,
                                              topKs, K_TOPK_MAX, mainStream,
                                              enable_pdl,
                                              /*skip_init=*/true);

    // ── Sync side stream (memset + topp radix) onto main before apply ───────
    if (sideStreamActive) {
        cudaEvent_t evt;
        cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
        cudaEventRecord(evt, msStream);
        cudaStreamWaitEvent(mainStream, evt, 0);
        cudaEventDestroy(evt);
    }

    // ── Stage 3: per-row apply. BLOCK=128, ITEMS=1 → MAX_K=128 sort window,
    //            and 128 threads handle V=160k via stride loops in the top-p
    //            branch (block reduce sized for 128).
    // Every row base is 16B-aligned iff both tensor bases are and vocab_size is a
    // multiple of 4 (row b starts at base + b*vocab_size floats). Otherwise some
    // row needs the peeled path — see the mode 3.2 branch in applyKernel. This is
    // a launch property, so pick the instantiation here and keep the aligned
    // kernel free of the peel code (and of its register cost).
    const bool rowsAligned = (reinterpret_cast<uintptr_t>(probs) & 0xFu) == 0 &&
                             (reinterpret_cast<uintptr_t>(outProbs) & 0xFu) == 0 &&
                             (vocabSize % 4 == 0);
    auto applyFn = rowsAligned ? applyKernel<128, 1, true> : applyKernel<128, 1, false>;
    launchKernel(enable_pdl, applyFn, dim3(batchSize), dim3(128), 0, mainStream,
                 probs, topKVals, topKIdx, topKs, topPs, toppCounters, outProbs, vocabSize,
                 K_TOPK_MAX, enable_pdl);
}

}  // namespace fused_topk_topp
