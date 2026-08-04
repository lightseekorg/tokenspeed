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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/cache_types.h"
#include "core/token_container.h"
#include "resource/allocator/req_pool_allocator.h"
#include "scheduler/request_spec.h"

namespace tokenspeed::fsm {

struct CacheProgress {
    // One source of truth for both the next hash-chain seed and the cumulative
    // history needed to publish a resumable boundary across chunk edges.
    std::vector<std::string> page_hashes;
    std::uint64_t access_epoch{0};
    // Pending closed-prefix boundary; zero once published or when absent.
    std::int32_t promotion_boundary_tokens{0};
};

inline std::vector<std::int32_t> ComputeShiftedInputIds(const TokenContainer* token_container,
                                                        TokenContainer::Window window) {
    const std::int32_t shifted_start = window.begin + 1;
    const std::int32_t shifted_end = std::min(token_container->PrefillSize(), shifted_start + window.size);
    const std::int32_t shifted_size = std::max<std::int32_t>(0, shifted_end - shifted_start);

    std::vector<std::int32_t> shifted;
    shifted.reserve(static_cast<std::size_t>(window.size));
    if (shifted_size > 0) {
        auto slice = token_container->TokenSlice(TokenContainer::Window{shifted_start, shifted_size});
        shifted.insert(shifted.end(), slice.begin(), slice.end());
    }
    shifted.resize(static_cast<std::size_t>(window.size), -1);
    return shifted;
}

struct Submitted {
    Submitted(TokenContainer* token_container, std::int32_t page_size)
        : token_container_{token_container}, page_size_{page_size} {}

    TokenContainer* TokenContainerPtr() const { return token_container_; }
    std::int32_t PageSize() const { return page_size_; }

private:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};
};

struct ForwardState {
    ForwardState(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<ReqPoolIndex> req_pool_index,
                 std::vector<BlockTable> block_tables, CacheProgress cache_progress)
        : token_container_{token_container},
          page_size_{page_size},
          req_pool_index_{std::move(req_pool_index)},
          block_tables_{std::move(block_tables)},
          cache_progress_{std::move(cache_progress)} {}

    ForwardState(const ForwardState&) = delete;
    ForwardState& operator=(const ForwardState&) = delete;
    ForwardState(ForwardState&&) noexcept = default;
    ForwardState& operator=(ForwardState&&) noexcept = default;

    TokenContainer* TokenContainerPtr() const { return token_container_; }
    std::int32_t PageSize() const { return page_size_; }

    std::unique_ptr<ReqPoolIndex> TakeRequestPoolIndex() && { return std::move(req_pool_index_); }
    std::int32_t RequestPoolIndex() const { return req_pool_index_ ? req_pool_index_->slot_ : -1; }

    std::vector<BlockTable>& BlockTables() { return block_tables_; }
    const std::vector<BlockTable>& BlockTables() const { return block_tables_; }
    std::vector<BlockTable> TakeBlockTables() && { return std::move(block_tables_); }

    CacheProgress TakeCacheProgress() && { return std::move(cache_progress_); }
    const CacheProgress& CacheProgressRef() const { return cache_progress_; }

    std::vector<std::int32_t> ActiveLcmBlockIds() const {
        std::vector<std::int32_t> ids;
        for (const BlockTable& table : block_tables_) {
            for (const CacheBlockRef& block_ref : table.Blocks()) {
                if (block_ref) {
                    ids.push_back(block_ref->Location().lcm_block_id);
                }
            }
        }
        return ids;
    }

protected:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};

private:
    std::unique_ptr<ReqPoolIndex> req_pool_index_;
    std::vector<BlockTable> block_tables_;
    CacheProgress cache_progress_;
};

struct Prefilling : public ForwardState {
    Prefilling(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<ReqPoolIndex> req_pool_index,
               TokenContainer::Window window, std::int32_t reserve_num_tokens_in_next_schedule_event,
               std::vector<BlockTable> block_tables, CacheProgress cache_progress)
        : ForwardState(token_container, page_size, std::move(req_pool_index), std::move(block_tables),
                       std::move(cache_progress)),
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::span<const std::int32_t> PrefillInputIds() const { return token_container_->TokenSlice(window); }
    std::vector<std::int32_t> ShiftedInputIds() const { return ComputeShiftedInputIds(token_container_, window); }
    PrefillInfo CurrentPrefillInfo() const {
        return PrefillInfo{
            .input_ids = PrefillInputIds(),
            .shifted_input_ids = ShiftedInputIds(),
            .already_scheduled_len = window.begin,
            .extend_len = window.size,
        };
    }

    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }

    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

struct PrefillDone : public ForwardState {
    PrefillDone(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<ReqPoolIndex> req_pool_index,
                TokenContainer::Window window, std::int32_t reserve_num_tokens_in_next_schedule_event,
                std::vector<BlockTable> block_tables, CacheProgress cache_progress)
        : ForwardState(token_container, page_size, std::move(req_pool_index), std::move(block_tables),
                       std::move(cache_progress)),
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }
    std::span<const std::int32_t> PrefillInputIds() const { return token_container_->TokenSlice(window); }
    std::vector<std::int32_t> ShiftedInputIds() const { return ComputeShiftedInputIds(token_container_, window); }
    PrefillInfo CurrentPrefillInfo() const {
        return PrefillInfo{
            .input_ids = PrefillInputIds(),
            .shifted_input_ids = ShiftedInputIds(),
            .already_scheduled_len = window.begin,
            .extend_len = window.size,
        };
    }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { token_container_->Extend(result_tokens); }

    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

struct Decoding : public ForwardState {
    Decoding(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<ReqPoolIndex> req_pool_index,
             std::int32_t reserve_num_tokens_in_next_schedule_event, std::vector<BlockTable> block_tables,
             CacheProgress cache_progress)
        : ForwardState(token_container, page_size, std::move(req_pool_index), std::move(block_tables),
                       std::move(cache_progress)),
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    std::int32_t ReserveNumTokensInNextScheduleEvent() const {
        _assert(reserve_num_tokens_in_next_schedule_event_ >= 0);
        return reserve_num_tokens_in_next_schedule_event_;
    }
    void SetReserveNumTokensInNextScheduleEvent(std::int32_t value) {
        reserve_num_tokens_in_next_schedule_event_ = value;
    }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { token_container_->Extend(result_tokens); }

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{-1};
};

struct RetractionSnapshot {
    TokenContainer* token_container{};
    std::int32_t page_size{};
    std::int32_t frontier_tokens{};
    std::vector<BlockTable> host_tables;
    CacheProgress cache_progress;
    std::int32_t decode_reserve_tokens{};
};

// The Device state remains owned until the D2H acknowledgement commits the
// complete snapshot. The Host tables are already pinned by strong refs.
struct Retracting {
    Decoding device_state;
    std::vector<BlockTable> host_tables;
};

struct Retracted {
    RetractionSnapshot snapshot;
};

// Host and newly allocated Device tables stay pinned together until H2D ACK.
struct Recovering {
    RetractionSnapshot snapshot;
    std::unique_ptr<ReqPoolIndex> req_pool_index;
    std::vector<BlockTable> device_tables;
};

struct Finished {};

}  // namespace tokenspeed::fsm
