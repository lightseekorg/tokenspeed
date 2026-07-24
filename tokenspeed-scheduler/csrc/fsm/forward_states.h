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
#include <utility>
#include <variant>

#include "core/token_container.h"
#include "resource/allocator/page_allocator.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/local_mamba_allocator.h"
#include "scheduler/operations/cache.h"
#include "resource/page_container.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/types.h"
#include "scheduler/request_spec.h"
#include "cache/cache_types.h"

namespace tokenspeed::fsm {

inline std::vector<std::int32_t> ComputeShiftedInputIds(const TokenContainer* token_container,
                                                        TokenContainer::Window window) {
    const std::int32_t shifted_start = window.begin + 1;
    const std::int32_t shifted_end = std::min(token_container->PrefillSize(), shifted_start + window.size);
    const std::int32_t shifted_size = std::max<std::int32_t>(0, shifted_end - shifted_start);

    std::vector<std::int32_t> shifted;
    shifted.reserve(static_cast<std::size_t>(window.size));
    if (shifted_size > 0) {
        auto slice = token_container->GetTokenSlice(TokenContainer::Window{shifted_start, shifted_size});
        shifted.insert(shifted.end(), slice.begin(), slice.end());
    }
    shifted.resize(static_cast<std::size_t>(window.size), -1);
    return shifted;
}

// Holds no allocator pointers; allocators are owned by the events that allocate.
struct Submitted {
    Submitted(TokenContainer* token_container, std::int32_t page_size)
        : token_container_{token_container}, page_size_{page_size} {}

    TokenContainer* GetTokenContainer() { return token_container_; }
    std::int32_t GetPageSize() const { return page_size_; }

    std::vector<std::int32_t> GetOccupiedPages() const {
        return {};  // prefix matched but not locked until ScheduleEvent
    }

private:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};
};

struct Aborting {
    Aborting(OwnedPages host_pages) : host_pages_{std::move(host_pages)} {}

    ~Aborting() = default;

    Aborting(Aborting&&) noexcept = default;
    Aborting& operator=(Aborting&&) noexcept = default;
    Aborting(const Aborting&) = delete;
    Aborting& operator=(const Aborting&) = delete;

    std::vector<std::int32_t> GetHostPageIds() const { return host_pages_.Ids(); }

    std::vector<std::int32_t> GetOccupiedPages() const { return {}; }

private:
    OwnedPages host_pages_;
};

struct BaseState {
public:
    BaseState(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<DeviceNodeRef>&& device_node_ref,
              std::unique_ptr<LocalKVAllocator>&& local_kv_allocator,
              std::unique_ptr<LocalMambaAllocator>&& local_mamba_allocator = nullptr)
        : token_container_{token_container},
          page_size_{page_size},
          device_node_ref_{std::move(device_node_ref)},
          local_kv_allocator_{std::move(local_kv_allocator)},
          local_mamba_allocator_{std::move(local_mamba_allocator)} {}

    BaseState(const BaseState&) = delete;
    BaseState& operator=(const BaseState&) = delete;
    BaseState(BaseState&&) noexcept = default;
    BaseState& operator=(BaseState&&) noexcept = default;

    PageContainer GetPageContainer() const {
        return PageContainer{device_node_ref_->Node(), local_kv_allocator_.get()};
    }

    const TreeNode* GetDeviceNode() const { return device_node_ref_->Node(); }

    // Null on the flat path (block_tables_ drives allocation), so radix-only publishing branches can skip.
    bool HasDeviceNodeRef() const { return device_node_ref_ != nullptr; }

    std::int32_t TailPageAvailableTokens() const {
        if (!block_tables_.empty()) {
            // flat: group-0 sample, feeding only the radix-era capacity probe (a flat no-op);
            // real admission gates on the coordinator's per-group BlocksNeededFor.
            return block_tables_[0].TailAvailableTokens();
        }
        return local_kv_allocator_->TailPageAvailableTokens();  // radix
    }

    auto GetFullPagedTokens() const { return token_container_->GetFullPagedTokens(page_size_, false); }
    auto GetFullPagedTokens(bool except_last) const {
        return token_container_->GetFullPagedTokens(page_size_, except_last);
    }

    TokenContainer* GetTokenContainer() { return token_container_; }

    const std::int32_t GetPageSize() const { return page_size_; }

    std::unique_ptr<DeviceNodeRef> TakeDeviceNodeRef() && { return std::move(device_node_ref_); }

    std::unique_ptr<LocalKVAllocator> TakeLocalKVAllocator() && { return std::move(local_kv_allocator_); }

    LocalKVAllocator* GetLocalKVAllocatorPtr() { return local_kv_allocator_.get(); }

    LocalMambaAllocator* GetLocalMambaAllocator() { return local_mamba_allocator_.get(); }
    std::unique_ptr<LocalMambaAllocator> TakeLocalMambaAllocator() && { return std::move(local_mamba_allocator_); }

    std::vector<BlockTable>& BlockTables() { return block_tables_; }
    const std::vector<BlockTable>& BlockTables() const { return block_tables_; }
    std::vector<BlockTable> TakeBlockTables() && { return std::move(block_tables_); }
    void SetBlockTables(std::vector<BlockTable> tables) { block_tables_ = std::move(tables); }

protected:
    TokenContainer* token_container_;
    std::int32_t page_size_{};

protected:
    std::unique_ptr<DeviceNodeRef> device_node_ref_;
    std::unique_ptr<LocalKVAllocator> local_kv_allocator_;
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator_;
    std::vector<BlockTable> block_tables_{};  // flat KV-cache path; empty under radix path
};

struct ForwardState : public BaseState {
    ForwardState(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<DeviceNodeRef>&& node_ref,
                 std::unique_ptr<LocalKVAllocator>&& local_kv_allocator, std::unique_ptr<ReqPoolIndex>&& req_pool_index,
                 std::unique_ptr<LocalMambaAllocator>&& local_mamba_allocator = nullptr)
        : BaseState(token_container, page_size, std::move(node_ref), std::move(local_kv_allocator),
                    std::move(local_mamba_allocator)),
          req_pool_index_(std::move(req_pool_index)) {}

    ForwardState(ForwardState&&) noexcept = default;
    ForwardState& operator=(ForwardState&&) noexcept = default;

    std::unique_ptr<ReqPoolIndex> TakeReqPoolIndex() && { return std::move(req_pool_index_); }
    std::int32_t GetReqPoolIndex() const { return req_pool_index_ ? req_pool_index_->slot_ : -1; }

    // Flat: group-0 SAMPLE (ids -- and with per-group block sizes, counts -- differ per group);
    // feeds one-group stats and the radix-era req_to_page fallback, never cross-group accounting
    // (that is GetOccupiedPagesAllGroups).
    std::vector<std::int32_t> GetOccupiedPages() const {
        if (!block_tables_.empty()) {
            return BlockTablePageIds(block_tables_[0]);  // flat: first-group sample (see above)
        }
        return GetPageContainer().Pages();  // radix
    }

    // Flat: real pages of EVERY group's block table (null holes excluded — block 0 is the
    // never-allocated punch placeholder, not occupancy). Block ids are pool-wide and each id
    // is owned by one group, so concatenation has no cross-group duplicates. This is the
    // pool-wide occupancy counterpart of the group-0 sample above.
    std::vector<std::int32_t> GetOccupiedPagesAllGroups() const {
        if (block_tables_.empty()) {
            return GetPageContainer().Pages();  // radix
        }
        std::vector<std::int32_t> ids;
        for (const BlockTable& table : block_tables_) {
            for (const BlockRef& block : table.Blocks()) {
                if (block) {
                    ids.push_back(block->BlockId());
                }
            }
        }
        return ids;
    }

    std::vector<std::int32_t> GetLocalAllocatorPages() const {
        if (!block_tables_.empty()) {
            return BlockTablePageIds(block_tables_[0]);  // flat: first-group sample (see GetOccupiedPages)
        }
        return local_kv_allocator_->Pages();  // radix: tail pages only, not tree-owned prefix pages
    }

private:
    std::unique_ptr<ReqPoolIndex> req_pool_index_;
};

struct Prefilling : public ForwardState {
    Prefilling(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<HostNodeRef>&& host_node_ref,
               std::unique_ptr<DeviceNodeRef>&& device_node_ref, std::unique_ptr<LocalKVAllocator>&& local_kv_allocator,
               std::unique_ptr<ReqPoolIndex>&& req_pool_index, TokenContainer::Window _window,
               std::unique_ptr<LocalMambaAllocator>&& local_mamba_allocator = nullptr)
        : ForwardState(token_container, page_size, std::move(device_node_ref), std::move(local_kv_allocator),
                       std::move(req_pool_index), std::move(local_mamba_allocator)),
          host_node_ref_(std::move(host_node_ref)),
          window(_window) {}

    Prefilling(Prefilling&& state) noexcept = default;
    Prefilling& operator=(Prefilling&&) noexcept = default;

    std::span<const std::int32_t> PrefillInputIds() const { return token_container_->GetTokenSlice(window); }

    std::vector<std::int32_t> ShiftedInputIds() const { return ComputeShiftedInputIds(token_container_, window); }

    PrefillInfo GetPrefillInfo() const {
        return PrefillInfo{
            .input_ids = PrefillInputIds(),
            .shifted_input_ids = ShiftedInputIds(),
            .already_scheduled_len = window.begin,
            .extend_len = window.size,
        };
    }

    std::unique_ptr<HostNodeRef> TakeHostNodeRef() && { return std::move(host_node_ref_); }

    TokenContainer::Window window{};

private:
    std::unique_ptr<HostNodeRef> host_node_ref_{};  // pins host pages until the next state takes ownership
};

// All prefill tokens scheduled (last chunk in flight); the next ScheduleEvent transitions to Decoding.
struct PrefillDone : public ForwardState {
    PrefillDone(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<HostNodeRef>&& host_node_ref,
                std::unique_ptr<DeviceNodeRef>&& device_node_ref,
                std::unique_ptr<LocalKVAllocator>&& local_kv_allocator, std::unique_ptr<ReqPoolIndex>&& req_pool_index,
                TokenContainer::Window _window, std::int32_t reserve_num_tokens_in_next_schedule_event,
                std::unique_ptr<LocalMambaAllocator>&& local_mamba_allocator = nullptr)
        : ForwardState(token_container, page_size, std::move(device_node_ref), std::move(local_kv_allocator),
                       std::move(req_pool_index), std::move(local_mamba_allocator)),
          host_node_ref_(std::move(host_node_ref)),
          window(_window),
          reserve_num_tokens_in_next_schedule_event_(reserve_num_tokens_in_next_schedule_event) {}

    PrefillDone(PrefillDone&& state) noexcept = default;
    PrefillDone& operator=(PrefillDone&&) noexcept = default;

    std::int32_t GetReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }

    std::span<const std::int32_t> PrefillInputIds() const { return token_container_->GetTokenSlice(window); }

    std::vector<std::int32_t> ShiftedInputIds() const { return ComputeShiftedInputIds(token_container_, window); }

    PrefillInfo GetPrefillInfo() const {
        return PrefillInfo{
            .input_ids = PrefillInputIds(),
            .shifted_input_ids = ShiftedInputIds(),
            .already_scheduled_len = window.begin,
            .extend_len = window.size,
        };
    }

    void ExtendResultTokens(const std::vector<std::int32_t> result_tokens) { token_container_->Extend(result_tokens); }

    std::unique_ptr<HostNodeRef> TakeHostNodeRef() && { return std::move(host_node_ref_); }

    TokenContainer::Window window{};

private:
    std::unique_ptr<HostNodeRef> host_node_ref_{};  // pins host pages until the next state takes ownership
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

#if TOKENSPEED_FLAT_KVCACHE
// Rolling page-hash chain: pages [0, num_hashed_pages) are registered, last_hash seeds the
// next increment. tail holds pages [tail_begin_page, num_hashed_pages) back to the fold grid
// (lcm/base) so coarse groups can fold a block completed mid-decode.
struct HashChain {
    std::int32_t num_hashed_pages{0};
    std::string last_hash;
    std::int32_t tail_begin_page{0};
    std::vector<std::string> tail;
};
#endif

struct Decoding : public ForwardState {
    Decoding(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<HostNodeRef>&& host_node_ref,
             std::unique_ptr<DeviceNodeRef>&& node_ref, std::unique_ptr<LocalKVAllocator>&& local_kv_allocator,
             std::unique_ptr<ReqPoolIndex>&& req_pool_index, std::int32_t reserve_num_tokens_in_next_schedule_event,
             std::unique_ptr<LocalMambaAllocator>&& local_mamba_allocator = nullptr)
        : ForwardState(token_container, page_size, std::move(node_ref), std::move(local_kv_allocator),
                       std::move(req_pool_index), std::move(local_mamba_allocator)),
          host_node_ref_(std::move(host_node_ref)),
          reserve_num_tokens_in_next_schedule_event_(reserve_num_tokens_in_next_schedule_event) {}

    Decoding() = delete;
    Decoding(Decoding&& state) noexcept = default;
    Decoding& operator=(Decoding&&) noexcept = default;

    std::int32_t GetReserveNumTokensInNextScheduleEvent() const {
        _assert(reserve_num_tokens_in_next_schedule_event_ >= 0);

        return reserve_num_tokens_in_next_schedule_event_;
    }
    void SetReserveNumTokensInNextScheduleEvent(std::int32_t n) { reserve_num_tokens_in_next_schedule_event_ = n; }

    void ExtendResultTokens(const std::vector<std::int32_t> result_tokens) { token_container_->Extend(result_tokens); }

    std::unique_ptr<HostNodeRef> TakeHostNodeRef() && { return std::move(host_node_ref_); }

#if TOKENSPEED_FLAT_KVCACHE
    HashChain TakeHashChain() && { return std::move(hash_chain_); }
    void SetHashChain(HashChain chain) { hash_chain_ = std::move(chain); }
#endif

private:
    std::unique_ptr<HostNodeRef> host_node_ref_{};  // pins host pages until the next state takes ownership
    std::int32_t reserve_num_tokens_in_next_schedule_event_{-1};
#if TOKENSPEED_FLAT_KVCACHE
    HashChain hash_chain_{};
#endif
};

// Generation finished, host pages allocated, writeback op not yet generated.
struct Draining {
    // Draining captures concrete page pairs while the node->page mapping is still stable, making
    // newWriteBackOperation split-safe (no re-walk after splitChild redistributes pages).
    using PagePair = TransferPair;
    Draining(std::vector<PagePair> pages_to_transfer, std::unique_ptr<DeviceNodeRef>&& device_node_ref,
             std::unique_ptr<HostNodeRef>&& host_node_ref, std::vector<TreeNode*> host_writeback_nodes = {},
             std::vector<TreeNode*> mamba_writeback_nodes = {},
             std::vector<PagedCacheTransferPair> paged_cache_writeback_transfers = {},
             std::vector<TreeNode*> paged_cache_writeback_nodes = {})
        : pages_to_transfer_(std::move(pages_to_transfer)),
          device_node_ref_(std::move(device_node_ref)),
          host_node_ref_(std::move(host_node_ref)),
          host_writeback_nodes_(std::move(host_writeback_nodes)),
          mamba_writeback_nodes_(std::move(mamba_writeback_nodes)),
          paged_cache_writeback_transfers_(std::move(paged_cache_writeback_transfers)),
          paged_cache_writeback_nodes_(std::move(paged_cache_writeback_nodes)) {}

public:
    const std::vector<PagePair>& GetPagesToTransfer() const { return pages_to_transfer_; }

    std::unique_ptr<DeviceNodeRef> TakeDeviceNodeRef() && { return std::move(device_node_ref_); }
    std::unique_ptr<HostNodeRef> TakeHostNodeRef() && { return std::move(host_node_ref_); }
    std::vector<TreeNode*> TakeHostWriteBackNodes() && { return std::move(host_writeback_nodes_); }
    std::vector<TreeNode*> TakeMambaWriteBackNodes() && { return std::move(mamba_writeback_nodes_); }
    const std::vector<PagedCacheTransferPair>& GetPagedCacheWriteBackTransfers() const {
        return paged_cache_writeback_transfers_;
    }
    const std::vector<TreeNode*>& PagedCacheWriteBackNodes() const { return paged_cache_writeback_nodes_; }
    std::vector<TreeNode*> TakePagedCacheWriteBackNodes() && { return std::move(paged_cache_writeback_nodes_); }

private:
    std::vector<PagePair> pages_to_transfer_;         // concrete mixed-kind pairs to copy
    std::unique_ptr<DeviceNodeRef> device_node_ref_;  // keeps matched Device node alive until WritingBack
    std::unique_ptr<HostNodeRef> host_node_ref_;      // keeps pre-allocated Host node alive until WritingBack
    std::vector<TreeNode*> host_writeback_nodes_;     // exact KV host resources newly allocated by this op
    std::vector<TreeNode*> mamba_writeback_nodes_;    // exact Mamba nodes covered by this writeback op
    std::vector<PagedCacheTransferPair> paged_cache_writeback_transfers_;
    std::vector<TreeNode*> paged_cache_writeback_nodes_;
};

// Writeback op executing; both node refs are RAII locks pinning the pages while the transfer is in flight.
struct WritingBack {
    WritingBack(std::unique_ptr<DeviceNodeRef>&& device_node_ref, std::unique_ptr<HostNodeRef>&& host_node_ref,
                std::vector<TreeNode*> host_writeback_nodes = {}, std::vector<TreeNode*> mamba_writeback_nodes = {},
                std::vector<TreeNode*> paged_cache_writeback_nodes = {})
        : device_node_ref_(std::move(device_node_ref)),
          host_node_ref_(std::move(host_node_ref)),
          host_writeback_nodes_(std::move(host_writeback_nodes)),
          mamba_writeback_nodes_(std::move(mamba_writeback_nodes)),
          paged_cache_writeback_nodes_(std::move(paged_cache_writeback_nodes)) {}

    WritingBack(WritingBack&&) noexcept = default;
    WritingBack& operator=(WritingBack&&) noexcept = default;

    std::unique_ptr<HostNodeRef> TakeHostNodeRef() && { return std::move(host_node_ref_); }
    TreeNode* DeviceNode() const { return device_node_ref_ ? device_node_ref_->Node() : nullptr; }
    const std::vector<TreeNode*>& HostWriteBackNodes() const { return host_writeback_nodes_; }
    const std::vector<TreeNode*>& MambaWriteBackNodes() const { return mamba_writeback_nodes_; }
    const std::vector<TreeNode*>& PagedCacheWriteBackNodes() const { return paged_cache_writeback_nodes_; }
    void DropDeviceNodeRef() { device_node_ref_.reset(); }
    void DropHostNodeRef() { host_node_ref_.reset(); }

private:
    std::unique_ptr<DeviceNodeRef> device_node_ref_;  // released after WriteBackDone
    std::unique_ptr<HostNodeRef> host_node_ref_;      // released after WriteBackDone
    std::vector<TreeNode*> host_writeback_nodes_;     // newly attached KV host resources
    std::vector<TreeNode*> mamba_writeback_nodes_;    // pending host Mamba slots published by this op ack
    std::vector<TreeNode*> paged_cache_writeback_nodes_;
};

// Keeps the local KV allocator (tail-page info) and token container for recovery.
struct Retracting : public WritingBack {
    using PagePair = TransferPair;

    Retracting(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<HostNodeRef>&& host_node_ref,
               std::unique_ptr<DeviceNodeRef>&& device_node_ref, std::unique_ptr<LocalKVAllocator>&& local_kv_allocator,
               std::vector<PagePair> pages_to_transfer, std::vector<TreeNode*> mamba_writeback_nodes = {},
               std::unique_ptr<LocalMambaAllocator>&& local_mamba_allocator = nullptr,
               std::vector<PagedCacheTransferPair> paged_cache_writeback_transfers = {},
               std::vector<TreeNode*> paged_cache_writeback_nodes = {},
               std::vector<TreeNode*> host_writeback_nodes = {})
        : WritingBack(std::move(device_node_ref), std::move(host_node_ref), std::move(host_writeback_nodes),
                      std::move(mamba_writeback_nodes), std::move(paged_cache_writeback_nodes)),
          token_container_{token_container},
          page_size_{page_size},
          local_kv_allocator_(std::move(local_kv_allocator)),
          pages_to_transfer_(std::move(pages_to_transfer)),
          local_mamba_allocator_(std::move(local_mamba_allocator)),
          paged_cache_writeback_transfers_(std::move(paged_cache_writeback_transfers)) {}

    Retracting(Retracting&&) noexcept = default;
    Retracting& operator=(Retracting&&) noexcept = default;

    TokenContainer* GetTokenContainer() const { return token_container_; }
    std::unique_ptr<LocalKVAllocator> TakeKVAllocator() && { return std::move(local_kv_allocator_); }
    std::unique_ptr<LocalMambaAllocator> TakeMambaAllocator() && { return std::move(local_mamba_allocator_); }
    std::int32_t GetPageSize() const { return page_size_; }

    const std::vector<PagePair>& GetPagesToTransfer() const { return pages_to_transfer_; }
    const std::vector<PagedCacheTransferPair>& GetPagedCacheWriteBackTransfers() const {
        return paged_cache_writeback_transfers_;
    }
    void ExtendResultTokens(const std::vector<std::int32_t> result_tokens) { token_container_->Extend(result_tokens); }

    std::vector<std::int32_t> GetLocalAllocatorPages() const {
        return local_kv_allocator_ ? local_kv_allocator_->Pages() : std::vector<std::int32_t>{};
    }

private:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};
    std::unique_ptr<LocalKVAllocator> local_kv_allocator_{};
    std::vector<PagePair> pages_to_transfer_{};
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator_{};
    std::vector<PagedCacheTransferPair> paged_cache_writeback_transfers_{};
};

struct Retracted {
    Retracted(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<HostNodeRef>&& host_node_ref,
              std::unique_ptr<LocalKVAllocator> local_kv_allocator,
              std::unique_ptr<LocalMambaAllocator> local_mamba_allocator = nullptr)
        : token_container_{token_container},
          page_size_{page_size},
          host_node_ref_{std::move(host_node_ref)},
          local_kv_allocator_(std::move(local_kv_allocator)),
          local_mamba_allocator_(std::move(local_mamba_allocator)) {}

    TokenContainer* GetTokenContainer() { return token_container_; }
    std::int32_t GetPageSize() const { return page_size_; }
    std::int32_t TailPageAvailableTokens() const {
        return local_kv_allocator_ ? local_kv_allocator_->TailPageAvailableTokens() : 0;
    }
    std::unique_ptr<LocalKVAllocator> TakeKVAllocator() && { return std::move(local_kv_allocator_); }
    std::unique_ptr<LocalMambaAllocator> TakeMambaAllocator() && { return std::move(local_mamba_allocator_); }

    std::vector<std::int32_t> GetLocalAllocatorPages() const {
        return local_kv_allocator_ ? local_kv_allocator_->Pages() : std::vector<std::int32_t>{};
    }

    void ExtendResultTokens(const std::vector<std::int32_t> result_tokens) { token_container_->Extend(result_tokens); }

private:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};
    std::unique_ptr<HostNodeRef> host_node_ref_{};
    std::unique_ptr<LocalKVAllocator> local_kv_allocator_{};
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator_{};
};

struct Finished {};

}  // namespace tokenspeed::fsm
