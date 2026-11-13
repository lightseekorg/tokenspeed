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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace {

constexpr int kHeads = 16;
constexpr int kHeadDim = 160;
constexpr int kTokensPerTile = 64;
constexpr int kHeadsPerBlock = 4;
constexpr int kLanesPerRow = 8;
constexpr int kRowsPerRound = 32 / kLanesPerRow;
constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ unsigned decode_pair(unsigned packed, float scale) {
    auto half_raw = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3);
    auto value = __half22float2(static_cast<__half2>(half_raw));
    __nv_bfloat162_raw result = __floats2bfloat162_rn(value.x * scale, value.y * scale);
    return static_cast<unsigned>(result.x) | (static_cast<unsigned>(result.y) << 16);
}

__device__ __forceinline__ void decode_four(unsigned packed, float scale,
                                           unsigned &lo, unsigned &hi) {
    lo = decode_pair(packed, scale);
    hi = decode_pair(packed >> 16, scale);
}

__device__ __forceinline__ void decode_sixteen(uint4 raw, float scale,
                                              uint4 &lo, uint4 &hi) {
    decode_four(raw.x, scale, lo.x, lo.y);
    decode_four(raw.y, scale, lo.z, lo.w);
    decode_four(raw.z, scale, hi.x, hi.y);
    decode_four(raw.w, scale, hi.z, hi.w);
}

__device__ __forceinline__ unsigned sort_warp_key(unsigned key, int lane) {
    #pragma unroll
    for (int width = 2; width <= 32; width *= 2) {
        #pragma unroll
        for (int stride = width / 2; stride > 0; stride /= 2) {
            unsigned peer = __shfl_xor_sync(kFullWarp, key, stride);
            bool take_min = ((lane & width) == 0) == ((lane & stride) == 0);
            key = take_min ? min(key, peer) : max(key, peer);
        }
    }
    return key;
}

__device__ __forceinline__ void sort_two_warp_keys(unsigned& left, unsigned& right, int lane) {
    unsigned ascending = sort_warp_key(left, lane);
    unsigned descending = __shfl_xor_sync(kFullWarp, sort_warp_key(right, lane), 31);
    left = min(ascending, descending);
    right = max(ascending, descending);
    #pragma unroll
    for (int stride = 16; stride > 0; stride /= 2) {
        unsigned peer_left = __shfl_xor_sync(kFullWarp, left, stride);
        unsigned peer_right = __shfl_xor_sync(kFullWarp, right, stride);
        bool lower = (lane & stride) == 0;
        left = lower ? min(left, peer_left) : max(left, peer_left);
        right = lower ? min(right, peer_right) : max(right, peer_right);
    }
}

struct LoadedRow {
    uint4 first;
    uint4 tail;
    uint64_t destinations;
};

__device__ __forceinline__ bool has_tail_vector(int lane_in_row) {
    return lane_in_row < kHeadDim / 16 - kLanesPerRow;
}

__device__ __forceinline__ LoadedRow read_row(
    const uint8_t* host, const uint8_t* hot, int id, int cache_row,
    uint64_t destinations, int lane_in_row
) {
    LoadedRow row{{}, {}, destinations};
    if (destinations != 0) {
        const auto* source = host + static_cast<int64_t>(id) * kHeadDim;
        if (cache_row >= 0) {
            source = hot + static_cast<int64_t>(cache_row) * kHeadDim;
        }
        const auto* vectors = reinterpret_cast<const uint4*>(source);
        row.first = vectors[lane_in_row];
        if (has_tail_vector(lane_in_row)) {
            row.tail = vectors[kLanesPerRow + lane_in_row];
        }
    }
    return row;
}

__device__ __forceinline__ void write_row(
    LoadedRow row, uint16_t* out, int base_token, int head, int lane_in_row,
    float scale
) {
    if (row.destinations == 0) {
        return;
    }
    bool has_tail = has_tail_vector(lane_in_row);
    uint4 lo, hi, tail_lo, tail_hi;
    decode_sixteen(row.first, scale, lo, hi);
    if (has_tail) {
        decode_sixteen(row.tail, scale, tail_lo, tail_hi);
    }

    auto store = [&](int token) {
        int output_row = (base_token + token) * kHeads + head;
        auto* dst = reinterpret_cast<uint4*>(out + static_cast<int64_t>(output_row) * kHeadDim);
        dst[lane_in_row * 2] = lo;
        dst[lane_in_row * 2 + 1] = hi;
        if (has_tail) {
            dst[(kLanesPerRow + lane_in_row) * 2] = tail_lo;
            dst[(kLanesPerRow + lane_in_row) * 2 + 1] = tail_hi;
        }
    };
    do {
        store(__ffsll(row.destinations) - 1);
        row.destinations &= row.destinations - 1;
    } while (row.destinations != 0);
}

struct SortedEntry {
    int id;
    int token;
    int leader;
    uint64_t destinations;
};

__device__ __forceinline__ SortedEntry make_sorted_entry(unsigned key, int base_id) {
    int id = base_id + static_cast<int>(key >> 6);
    int token = key & 63;
    unsigned peers = __match_any_sync(kFullWarp, id);
    unsigned bit = 1u << (token & 31);
    unsigned low = __reduce_or_sync(peers, token < 32 ? bit : 0u);
    unsigned high = __reduce_or_sync(peers, token >= 32 ? bit : 0u);
    return {id, token, __ffs(peers) - 1, (static_cast<uint64_t>(high) << 32) | low};
}

__device__ __forceinline__ int hot_slot(int id, int slot_bits) {
    return (static_cast<unsigned>(id) * 2654435761u) >> (32 - slot_bits);
}

__device__ __forceinline__ void stage_row(
    LoadedRow row, uint8_t* staged, int index, int lane_in_row
) {
    auto* dst = reinterpret_cast<uint4*>(staged + static_cast<int64_t>(index) * kHeadDim);
    dst[lane_in_row] = row.first;
    if (has_tail_vector(lane_in_row)) {
        dst[kLanesPerRow + lane_in_row] = row.tail;
    }
}

__global__ void gather_wide_cta(const uint8_t* __restrict__ host,
                               const int64_t* __restrict__ ids,
                               const int32_t* __restrict__ tags,
                               const uint8_t* __restrict__ hot,
                               uint16_t* __restrict__ out,
                               int32_t* __restrict__ miss_slots,
                               float scale, int slot_bits, int vocab_size,
                               uint8_t* __restrict__ staged,
                               int32_t* __restrict__ winners) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row_group = lane / kLanesPerRow;
    int lane_in_row = lane % kLanesPerRow;
    int base_token = blockIdx.x * kTokensPerTile;
    int head = blockIdx.y * kHeadsPerBlock + warp;
    int64_t low = ids[(base_token + lane) * kHeads + head];
    int64_t high = ids[(base_token + lane + 32) * kHeads + head];
    bool valid = low >= 0 && low < vocab_size && high >= 0 && high < vocab_size;
    int low_id = valid ? static_cast<int>(low) : 0;
    int high_id = valid ? static_cast<int>(high) : 0;
    int base_id = __reduce_min_sync(kFullWarp, min(low_id, high_id));
    int max_id = __reduce_max_sync(kFullWarp, max(low_id, high_id));
    if (!__all_sync(kFullWarp, valid) || max_id - base_id >= (1 << 26)) {
        for (int token = row_group; token < kTokensPerTile; token += kRowsPerRound) {
            int index = (base_token + token) * kHeads + head;
            int64_t id = ids[index];
            bool owned = id >= 0 && id < vocab_size;
            int slot = owned ? hot_slot(static_cast<int>(id), slot_bits) : -1;
            int cache_row = owned && tags[slot] == id ? slot : -1;
            int miss = owned && cache_row < 0 ? slot : -1;
            if (lane_in_row == 0) {
                miss_slots[index] = miss;
                if (miss >= 0) {
                    atomicMax(winners + miss, index);
                }
            }
            uint64_t destinations = uint64_t{1} << token;
            auto row = read_row(host, hot, owned ? static_cast<int>(id) : 0,
                                cache_row, owned ? destinations : 0, lane_in_row);
            if (miss >= 0) {
                stage_row(row, staged, index, lane_in_row);
            }
            row.destinations = destinations;
            write_row(row, out, base_token, head, lane_in_row, scale);
        }
        return;
    }
    unsigned key_left = (static_cast<unsigned>(low_id - base_id) << 6) | lane;
    unsigned key_right = (static_cast<unsigned>(high_id - base_id) << 6) | (lane + 32);
    sort_two_warp_keys(key_left, key_right, lane);

    auto left = make_sorted_entry(key_left, base_id);
    auto right = make_sorted_entry(key_right, base_id);
    int left_last = __shfl_sync(kFullWarp, left.id, 31);
    int right_first = __shfl_sync(kFullWarp, right.id, 0);
    bool shared_boundary = left_last == right_first;
    uint64_t boundary_destinations = __shfl_sync(kFullWarp, right.destinations, 0);
    if (shared_boundary && left.id == left_last) {
        left.destinations |= boundary_destinations;
    }
    bool left_leader = lane == left.leader;
    bool right_at_boundary = shared_boundary && right.id == right_first;
    bool right_leader = lane == right.leader && !right_at_boundary;

    int slot_left = hot_slot(left.id, slot_bits);
    int slot_right = hot_slot(right.id, slot_bits);
    int cache_left = tags[slot_left] == left.id ? slot_left : -1;
    int cache_right = tags[slot_right] == right.id ? slot_right : -1;
    miss_slots[(base_token + left.token) * kHeads + head] = cache_left < 0 ? slot_left : -1;
    miss_slots[(base_token + right.token) * kHeads + head] = cache_right < 0 ? slot_right : -1;

    unsigned leaders_left = __ballot_sync(kFullWarp, left_leader);
    unsigned leaders_right = __ballot_sync(kFullWarp, right_leader);
    int count_left = __popc(leaders_left);
    int rounds = (count_left + __popc(leaders_right) + kRowsPerRound - 1) / kRowsPerRound;
    auto read = [&](int round) {
        int rank = round * kRowsPerRound + row_group;
        bool from_left = rank < count_left;
        unsigned leaders = from_left ? leaders_left : leaders_right;
        int local_rank = from_left ? rank : rank - count_left;
        int leader = static_cast<int>(__fns(leaders, 0, local_rank + 1));
        int source_lane = leader >= 0 ? leader : 0;
        int id_left = __shfl_sync(kFullWarp, left.id, source_lane);
        int id_right = __shfl_sync(kFullWarp, right.id, source_lane);
        uint64_t destinations_left = __shfl_sync(kFullWarp, left.destinations, source_lane);
        uint64_t destinations_right = __shfl_sync(kFullWarp, right.destinations, source_lane);
        int cached_left = __shfl_sync(kFullWarp, cache_left, source_lane);
        int cached_right = __shfl_sync(kFullWarp, cache_right, source_lane);
        int cache_row = from_left ? cached_left : cached_right;
        uint64_t destinations = from_left ? destinations_left : destinations_right;
        int id = from_left ? id_left : id_right;
        auto row = read_row(host, hot, id, cache_row,
                            leader >= 0 ? destinations : uint64_t{0}, lane_in_row);
        if (row.destinations != 0 && cache_row < 0) {
            // Only the last occurrence of a repeated ID can win its cache slot.
            int token = 63 - __clzll(row.destinations);
            int index = (base_token + token) * kHeads + head;
            stage_row(row, staged, index, lane_in_row);
            if (lane_in_row == 0) {
                atomicMax(winners + hot_slot(id, slot_bits), index);
            }
        }
        write_row(row, out, base_token, head, lane_in_row, scale);
    };
    for (int round = 0; round < rounds; ++round) {
        read(round);
    }
}

}  // namespace

void ple_gather_staged(int64_t address, TensorView ids, TensorView tags, TensorView hot,
                       TensorView out, TensorView misses, double scale, int64_t tokens,
                       int64_t vocab_size, TensorView staged, TensorView winners) {
    TVM_FFI_ICHECK(tokens > 0 && tokens % kTokensPerTile == 0 &&
                   tokens <= INT32_MAX / kHeads);
    TVM_FFI_ICHECK(vocab_size > 0 && vocab_size <= INT32_MAX);
    int capacity = tags.size(0);
    TVM_FFI_ICHECK(capacity > 1 && (capacity & (capacity - 1)) == 0);
    int slot_bits = __builtin_ctz(static_cast<unsigned>(capacity));
    dim3 grid(tokens / kTokensPerTile, kHeads / kHeadsPerBlock);
    gather_wide_cta<<<grid, kHeadsPerBlock * 32, 0, get_stream(ids.device())>>>(
        reinterpret_cast<const uint8_t*>(address),
        static_cast<const int64_t*>(ids.data_ptr()),
        static_cast<const int32_t*>(tags.data_ptr()),
        static_cast<const uint8_t*>(hot.data_ptr()),
        static_cast<uint16_t*>(out.data_ptr()),
        static_cast<int32_t*>(misses.data_ptr()),
        static_cast<float>(scale), slot_bits, static_cast<int>(vocab_size),
        static_cast<uint8_t*>(staged.data_ptr()), static_cast<int32_t*>(winners.data_ptr()));
    auto status = cudaPeekAtLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ple_gather_staged, ple_gather_staged);
