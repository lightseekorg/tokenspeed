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
#include "resource/page_container.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/types.h"
#include "scheduler/request_spec.h"

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

// Submitted holds only the token container and page size — no allocator pointers.
// Allocators are owned by the events that perform resource allocation.
struct Submitted {
    Submitted(TokenContainer* token_container, std::int32_t page_size)
        : token_container_{token_container}, page_size_{page_size} {}

    TokenContainer* GetTokenContainer() { return token_container_; }
    std::int32_t GetPageSize() const { return page_size_; }

    std::vector<std::int32_t> GetOccupiedPages() const {
        // prefix is matched, but not locked
        // will be locked in ScheduleEvent
        return {};
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

    std::int32_t TailPageAvailableTokens() const { return local_kv_allocator_->TailPageAvailableTokens(); }

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

protected:
    TokenContainer* token_container_;
    std::int32_t page_size_{};

protected:
    std::unique_ptr<DeviceNodeRef> device_node_ref_;
    std::unique_ptr<LocalKVAllocator> local_kv_allocator_;
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator_;
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

    std::vector<std::int32_t> GetOccupiedPages() const { return GetPageContainer().Pages(); }

    // Returns only the pages held by the local KV allocator (tail pages, not radix-tree prefix pages).
    std::vector<std::int32_t> GetLocalAllocatorPages() const { return local_kv_allocator_->Pages(); }

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
    std::unique_ptr<HostNodeRef> host_node_ref_{};      // pins host pages until the next state takes ownership
};

// All prefill tokens have been scheduled (in-flight in the last chunk).
// On the next NextExecutionPlan call, ScheduleEvent transitions this to Decoding.
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

    // As a design decision, SetReserveNumTokensInNextScheduleEvent is not allowed
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
    std::unique_ptr<HostNodeRef> host_node_ref_{};      // pins host pages until the next state takes ownership
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

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

private:
    std::unique_ptr<HostNodeRef> host_node_ref_{};      // pins host pages until the next state takes ownership
    std::int32_t reserve_num_tokens_in_next_schedule_event_{-1};
};

// Request has finished it's generation, and host pages have been allocated,
// ready to WriteBack to l2 Cache, but no ops generated yet.
struct Draining {
    // pages_to_transfer is captured in FinishEvent immediately after alloc_host_node,
    // while the node→page mapping is still stable (before any future splitChild calls
    // can redistribute pages across new prefix/suffix nodes).
    // Storing concrete (device_page, host_page) pairs here makes newWriteBackOperation
    // split-safe: it never needs to re-walk the radix tree.
    using PagePair = std::tuple<std::int32_t, std::int32_t>;
    Draining(std::vector<PagePair> pages_to_transfer, std::unique_ptr<DeviceNodeRef>&& device_node_ref,
             std::unique_ptr<HostNodeRef>&& host_node_ref)
        : pages_to_transfer_(std::move(pages_to_transfer)),
          device_node_ref_(std::move(device_node_ref)),
          host_node_ref_(std::move(host_node_ref)) {}

public:
    // (device_page, host_page) pairs that must be transferred Device→Host.
    const std::vector<PagePair>& GetPagesToTransfer() const { return pages_to_transfer_; }

    std::unique_ptr<DeviceNodeRef> TakeDeviceNodeRef() && { return std::move(device_node_ref_); }
    std::unique_ptr<HostNodeRef> TakeHostNodeRef() && { return std::move(host_node_ref_); }

private:
    std::vector<PagePair> pages_to_transfer_;         // concrete (dev, host) page pairs to copy
    std::unique_ptr<DeviceNodeRef> device_node_ref_;  // keeps matched Device node alive until WritingBack
    std::unique_ptr<HostNodeRef> host_node_ref_;      // keeps pre-allocated Host node alive until WritingBack
};

// WritingBack OP has been generated, executing offload.
// Holds both node refs as RAII locks so the pages are not evicted while the
// async Device→Host transfer is in flight.
struct WritingBack {
    WritingBack(std::unique_ptr<DeviceNodeRef>&& device_node_ref, std::unique_ptr<HostNodeRef>&& host_node_ref)
        : device_node_ref_(std::move(device_node_ref)), host_node_ref_(std::move(host_node_ref)) {}

    WritingBack(WritingBack&&) noexcept = default;
    WritingBack& operator=(WritingBack&&) noexcept = default;

    std::unique_ptr<HostNodeRef> TakeHostNodeRef() && { return std::move(host_node_ref_); }

private:
    std::unique_ptr<DeviceNodeRef> device_node_ref_;  // released after WriteBackDone
    std::unique_ptr<HostNodeRef> host_node_ref_;      // released after WriteBackDone
};

// Need to hold local_kv_allocator(has tail page info), and token container for recovery
struct Retracting : public WritingBack {
    using PagePair = std::tuple<std::int32_t, std::int32_t>;

    Retracting(TokenContainer* token_container, std::int32_t page_size, std::unique_ptr<HostNodeRef>&& host_node_ref,
               std::unique_ptr<DeviceNodeRef>&& device_node_ref, std::unique_ptr<LocalKVAllocator>&& local_kv_allocator,
               std::vector<PagePair> pages_to_transfer,
               std::unique_ptr<LocalMambaAllocator>&& local_mamba_allocator = nullptr)
        : WritingBack(std::move(device_node_ref), std::move(host_node_ref)),
          token_container_{token_container},
          page_size_{page_size},
          local_kv_allocator_(std::move(local_kv_allocator)),
          pages_to_transfer_(std::move(pages_to_transfer)),
          local_mamba_allocator_(std::move(local_mamba_allocator)) {}

    Retracting(Retracting&&) noexcept = default;
    Retracting& operator=(Retracting&&) noexcept = default;

    TokenContainer* GetTokenContainer() const { return token_container_; }
    std::unique_ptr<LocalKVAllocator> TakeKVAllocator() && { return std::move(local_kv_allocator_); }
    std::unique_ptr<LocalMambaAllocator> TakeMambaAllocator() && { return std::move(local_mamba_allocator_); }
    std::int32_t GetPageSize() const { return page_size_; }

    // (device_page, host_page) pairs to transfer, captured at ScheduleRetractEvent time.
    const std::vector<PagePair>& GetPagesToTransfer() const { return pages_to_transfer_; }
    void ExtendResultTokens(const std::vector<std::int32_t> result_tokens) { token_container_->Extend(result_tokens); }

    // Returns only the pages held by the local KV allocator (tail page after retraction insert).
    std::vector<std::int32_t> GetLocalAllocatorPages() const {
        return local_kv_allocator_ ? local_kv_allocator_->Pages() : std::vector<std::int32_t>{};
    }

private:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};
    std::unique_ptr<LocalKVAllocator> local_kv_allocator_{};
    std::vector<PagePair> pages_to_transfer_{};
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator_{};
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

    // Returns only the pages held by the local KV allocator (tail page kept after retraction).
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
