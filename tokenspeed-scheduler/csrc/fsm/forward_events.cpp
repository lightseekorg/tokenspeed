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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/token_container.h"
#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"

namespace {

// Build a flat list of (device_page, host_page) pairs from the given write_diff nodes.
// Both Draining::PagePair and Retracting::PagePair are std::tuple<int32_t, int32_t>.
std::vector<tokenspeed::TransferPair> BuildWriteBackPairs(const std::vector<tokenspeed::TreeNode*>& write_diff) {
    std::vector<tokenspeed::TransferPair> pages_to_transfer;
    for (tokenspeed::TreeNode* n : write_diff) {
        const auto& dev_pages = n->Device().Pages();
        const auto& host_pages = n->Host().Pages();
        for (std::size_t i = 0; i < dev_pages.size(); ++i) {
            pages_to_transfer.push_back(
                tokenspeed::TransferPair{tokenspeed::CacheKind::kKV, dev_pages[i], host_pages[i]});
        }
    }
    return pages_to_transfer;
}

void DemoteWrittenBackDevice(tokenspeed::KVPrefixCache* kv_prefix_cache,
                             tokenspeed::HybridPrefixCache* hybrid_prefix_cache, tokenspeed::TreeNode* device_node) {
    if (kv_prefix_cache == nullptr || device_node == nullptr) return;
    kv_prefix_cache->ReleaseDeviceResourcesPresentOnHost(device_node, [hybrid_prefix_cache](tokenspeed::TreeNode* n) {
        if (hybrid_prefix_cache != nullptr) {
            hybrid_prefix_cache->OnKVDeviceDemote(n);
        }
    });
}

}  // namespace

namespace tokenspeed::fsm {

// Submitted -> PrefillDone / Prefilling
std::variant<PrefillDone, Prefilling> SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
    // Lock node
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    std::int32_t max_matched_pages =
        disable_l2_cache_ ? match_result_.device.DepthInPage()
                          : std::max(match_result_.device.DepthInPage(), match_result_.host.DepthInPage());
    std::int32_t window_begin = max_matched_pages * state.GetPageSize();
    const std::int32_t checkpoint_raw_position = window_begin + tokens_this_round_;
    if (!disable_l2_cache_ && (match_result_.host.DepthInPage() > match_result_.device.DepthInPage())) {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.host.last_node);
        StepCommitRequest materialization_request{
            .materialize_prefix =
                PrefixMaterializationRequest{
                    .compat_match = &match_result_,
                    .require_all_pages = false,
                },
        };
        (void)hybrid_prefix_cache_.StepCommit(std::move(materialization_request));
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.host.last_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
    }

    auto step_result = hybrid_prefix_cache_.StepCommit({
        .request_local_kv =
            RequestLocalKVStateRequest{
                .create_allocator = true,
                .initial_tokens = tokens_this_round_,
                .acquire_tokens = decode_input_tokens_,
            },
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .create_allocator = true,
                .checkpoint_raw_position = checkpoint_raw_position,
            },
    });
    auto local_kv_allocator = std::move(step_result.local_kv_allocator);

    // Allocate req_pool_idx when first-time scheduled
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());

    auto local_mamba_allocator = std::move(step_result.local_mamba_allocator);

    TokenContainer* token_container = state.GetTokenContainer();

    TokenContainer::Window window{.begin = window_begin, .size = tokens_this_round_};

    bool is_last_chunk = (window.begin + window.size) == token_container->PrefillSize();
    if (is_last_chunk && role_ != Role::kD) {
        std::int32_t reserve_num_tokens_in_next_schedule_event = decode_input_tokens_;
        return PrefillDone{token_container,
                           state.GetPageSize(),
                           std::move(host_node_ref),
                           std::move(device_node_ref),
                           std::move(local_kv_allocator),
                           std::move(req_pool_index),
                           window,
                           reserve_num_tokens_in_next_schedule_event,
                           std::move(local_mamba_allocator)};
    } else {
        return Prefilling{token_container,
                          state.GetPageSize(),
                          std::move(host_node_ref),
                          std::move(device_node_ref),
                          std::move(local_kv_allocator),
                          std::move(req_pool_index),
                          window,
                          std::move(local_mamba_allocator)};
    }
}

// Prefilling -> Prefilling / PrefillDone
std::variant<PrefillDone, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    // Only insert pages from the beginning up to the end of the last processed chunk.
    auto paged_tokens = state.GetFullPagedTokens(false);
    std::int32_t end_of_window_pages = (state.window.begin + state.window.size) / state.GetPageSize();
    if (end_of_window_pages < static_cast<std::int32_t>(paged_tokens.size())) {
        paged_tokens.resize(end_of_window_pages);
    }
    StepCommitRequest publication_request{
        .publish_device_prefix =
            DevicePrefixPublicationRequest{
                .full_paged_tokens = &paged_tokens,
                .device_node_ref = &device_node_ref,
                .local_kv_allocator = local_kv_allocator.get(),
                .local_mamba_allocator = local_mamba_allocator.get(),
                .chunk_begin = state.window.begin,
            },
        .request_local_kv =
            RequestLocalKVStateRequest{
                .allocator = local_kv_allocator.get(),
                .acquire_tokens = tokens_this_round_,
            },
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .refresh_checkpoint_allocator = local_mamba_allocator.get(),
                .checkpoint_raw_position = state.window.begin + state.window.size + tokens_this_round_,
            },
    };
    (void)hybrid_prefix_cache_.StepCommit(std::move(publication_request));

    TokenContainer::Window window{.begin = state.window.begin + state.window.size, .size = tokens_this_round_};

    bool is_last_chunk = (window.begin + window.size) == state.GetTokenContainer()->PrefillSize();
    if (is_last_chunk) {
        return PrefillDone{state.GetTokenContainer(),
                           state.GetPageSize(),
                           std::move(host_node_ref),
                           std::move(device_node_ref),
                           std::move(local_kv_allocator),
                           std::move(state).TakeReqPoolIndex(),
                           window,
                           reserve_num_tokens_in_next_schedule_event_,
                           std::move(local_mamba_allocator)};
    } else {
        return Prefilling{state.GetTokenContainer(),
                          state.GetPageSize(),
                          std::move(host_node_ref),
                          std::move(device_node_ref),
                          std::move(local_kv_allocator),
                          std::move(state).TakeReqPoolIndex(),
                          window,
                          std::move(local_mamba_allocator)};
    }
}

// PrefillDone -> Decoding: insert prefill pages into tree, then transition to decode.
Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    // Only insert pages from the beginning up to the end of the last processed chunk.
    auto paged_tokens = state.GetFullPagedTokens(false);
    std::int32_t end_of_window_pages = (state.window.begin + state.window.size) / state.GetPageSize();
    if (end_of_window_pages < static_cast<std::int32_t>(paged_tokens.size())) {
        paged_tokens.resize(end_of_window_pages);
    }
    StepCommitRequest publication_request{
        .publish_device_prefix =
            DevicePrefixPublicationRequest{
                .full_paged_tokens = &paged_tokens,
                .device_node_ref = &device_node_ref,
                .local_kv_allocator = local_kv_allocator.get(),
                .local_mamba_allocator = local_mamba_allocator.get(),
                .chunk_begin = state.window.begin,
            },
        .request_local_kv =
            RequestLocalKVStateRequest{
                .allocator = local_kv_allocator.get(),
                .acquire_tokens = state.GetReserveNumTokensInNextScheduleEvent(),
            },
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .refresh_checkpoint_allocator = local_mamba_allocator.get(),
                .checkpoint_raw_position = state.GetTokenContainer()->Size() + decode_input_tokens_,
            },
    };
    (void)hybrid_prefix_cache_.StepCommit(std::move(publication_request));

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_,          std::move(local_mamba_allocator)};
}

// Decoding -> Decoding: allocate pages for next decode step.
Decoding ScheduleDecodeEvent::operator()(Decoding&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    (void)hybrid_prefix_cache_.StepCommit({
        .request_local_kv =
            RequestLocalKVStateRequest{
                .allocator = local_kv_allocator.get(),
                .acquire_tokens = reserve,
            },
    });

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_,          std::move(local_mamba_allocator)};
}

// Retracted -> Decoding: recover via LoadBack (host → device).
// match_result_ was computed by the caller; alloc_device_node attaches device pages to LoadBack nodes.
Decoding ScheduleDecodeFromRetractedEvent::operator()(Retracted&& state) {
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    if (match_result_.host.DepthInPage() > match_result_.device.DepthInPage()) {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.host.last_node);
        StepCommitRequest materialization_request{
            .materialize_prefix =
                PrefixMaterializationRequest{
                    .compat_match = &match_result_,
                    .require_all_pages = true,
                },
        };
        if (!hybrid_prefix_cache_.StepCommit(std::move(materialization_request)).ok) {
            // Device allocation failed (race between capacity check and actual alloc).
            throw std::logic_error(
                "ScheduleDecodeFromRetractedEvent: failed to allocate device pages for host cache recovery");
        }
        // This is not a typo
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.host.last_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
    }
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_kv_allocator = std::move(state).TakeKVAllocator();
    auto old_mamba_allocator = std::move(state).TakeMambaAllocator();
    old_mamba_allocator.reset();
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    auto step_result = hybrid_prefix_cache_.StepCommit({
        .request_local_kv =
            RequestLocalKVStateRequest{
                .allocator = local_kv_allocator.get(),
                .acquire_tokens = decode_input_tokens_,
            },
        .request_local_mamba =
            RequestLocalMambaStateRequest{
                .create_allocator = true,
                .require_allocator = true,
            },
    });
    auto local_mamba_allocator = std::move(step_result.local_mamba_allocator);
    return Decoding{token_container,
                    page_size,
                    std::move(host_node_ref),
                    std::move(device_node_ref),
                    std::move(local_kv_allocator),
                    std::move(req_pool_index),
                    decode_input_tokens_,
                    std::move(local_mamba_allocator)};
}

// Decode -> Finish / PrefillDone -> Finish
// This transection is triggered by python side Advance
template <typename ForwardStateT>
std::variant<Draining, Finished> FinishEvent::apply(ForwardStateT&& state) {
    auto full_paged_tokens = state.GetFullPagedTokens(true);
    const TreeNode* current_device_node = state.GetDeviceNode();

    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto local_allocator = std::move(state).TakeLocalKVAllocator();
    StepCommitRequest publication_request{
        .publish_finished_request =
            FinishedRequestPublicationRequest{
                .full_paged_tokens = &full_paged_tokens,
                .current_device_node = current_device_node,
                .local_kv_allocator = local_allocator.get(),
                .local_mamba_allocator = local_mamba_allocator.get(),
                .page_hashes = &page_hashes_,
            },
    };
    MatchResult match = hybrid_prefix_cache_.StepCommit(std::move(publication_request)).match_result;
    // local_mamba_allocator dropped here — destructor frees remaining slots

    if (!disable_l2_cache_ && (match.device.DepthInPage() > match.host.DepthInPage())) {
        std::vector<TreeNode*> write_diff = match.NodesWithout<ResourceType::Host>();
        std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match.host.last_node);
        StepCommitRequest materialization_request{
            .materialize_host_writeback =
                HostWritebackMaterializationRequest{
                    .write_diff = &write_diff,
                    .ensure_capacity_before_allocate = true,
                },
        };
        StepCommitResult materialization_result = hybrid_prefix_cache_.StepCommit(std::move(materialization_request));
        if (!materialization_result.ok) {
            return Finished{};
        }
        std::unique_ptr<DeviceNodeRef> device_node_ref = std::make_unique<DeviceNodeRef>(match.device.last_node);
        std::unique_ptr<HostNodeRef> host_node_ref = std::make_unique<HostNodeRef>(match.device.last_node);

        auto pages_to_transfer = BuildWriteBackPairs(write_diff);
        pages_to_transfer.insert(pages_to_transfer.end(),
                                 std::make_move_iterator(materialization_result.cache_transfer_pairs.begin()),
                                 std::make_move_iterator(materialization_result.cache_transfer_pairs.end()));
        return Draining{std::move(pages_to_transfer), std::move(write_diff), std::move(device_node_ref),
                        std::move(host_node_ref), std::move(materialization_result.mamba_writeback_nodes)};
    }
    return Finished{};
}

std::variant<Draining, Finished> FinishEvent::operator()(Decoding&& state) {
    return apply(std::move(state));
}

std::variant<Draining, Finished> FinishEvent::operator()(PrefillDone&& state) {
    return apply(std::move(state));
}

// The request finished (EOS) while its device→host writeback is still in-flight.
// Downcast to WritingBack so that WriteBackDoneEvent takes the existing
// WritingBack → Finished path.  TokenContainer and LocalKVAllocator are
// released here (no longer needed for recovery).
WritingBack FinishEvent::operator()(Retracting&& state) {
    return static_cast<WritingBack&&>(state);
}

// Draining → WritingBack
// Transfer both RAII node-ref locks out of Draining and into WritingBack.
// From this point the request no longer owns match_result; the locks alone
// are enough to keep the Device and Host pages pinned until WriteBackDone.
WritingBack CommitDrainingEvent::operator()(Draining&& state) {
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();
    auto mamba_writeback_nodes = std::move(state).TakeMambaWriteBackNodes();
    return WritingBack{std::move(device_node_ref), std::move(host_node_ref), std::move(mamba_writeback_nodes)};
}

// WritingBack → Finished
// The async Device→Host transfer completed. Dropping the refs releases locks,
// then written-back cache becomes host-only so the next hit must load back.
Finished WriteBackDoneEvent::operator()(WritingBack&& state) {
    TreeNode* device_node = state.DeviceNode();
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->OnMambaHostWriteBackDone(state.MambaWriteBackNodes());
    }
    state.DropDeviceNodeRef();
    DemoteWrittenBackDevice(kv_prefix_cache_, hybrid_prefix_cache_, device_node);
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->DemoteIdleMambaDeviceCopiesPresentOnHost();
    }
    return Finished{};
}

Retracted WriteBackDoneEvent::operator()(Retracting&& state) {
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    TreeNode* device_node = state.DeviceNode();
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->OnMambaHostWriteBackDone(state.MambaWriteBackNodes());
    }
    state.DropDeviceNodeRef();
    DemoteWrittenBackDevice(kv_prefix_cache_, hybrid_prefix_cache_, device_node);
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->DemoteIdleMambaDeviceCopiesPresentOnHost();
    }
    auto host_ref = std::move(static_cast<WritingBack&&>(state)).TakeHostNodeRef();
    std::unique_ptr<LocalKVAllocator> local_device_allocator = std::move(state).TakeKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeMambaAllocator();
    // DeviceNodeRef inside WritingBack base is released here (unique_ptr dtor).
    return Retracted{token_container, page_size, std::move(host_ref), std::move(local_device_allocator),
                     std::move(local_mamba_allocator)};
}

Finished AbortEvent::operator()(Submitted&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Prefetching&& state) {
    return Aborting{std::move(state).TakeHostPages()};
}

Finished AbortEvent::operator()(Draining&&) {
    return Finished{};
}

Finished AbortEvent::operator()(PrefetchDone&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Aborting&& state) {
    return std::move(state);
}

Finished AbortEvent::operator()(Prefilling&&) {
    return Finished{};
}

Finished AbortEvent::operator()(PrefillDone&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Decoding&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Retracting&&) {
    return Finished{};
}

Finished AbortEvent::operator()(Retracted&&) {
    return Finished{};
}

template <typename ForwardStateT>
Retracting ScheduleRetractEvent::applyRetract(ForwardStateT&& state) {
    std::unique_ptr<DeviceNodeRef> device_node_ref = nullptr;
    std::unique_ptr<HostNodeRef> host_node_ref = nullptr;
    std::vector<Retracting::PagePair> pages_to_transfer;
    std::vector<TreeNode*> writeback_nodes;
    std::vector<TreeNode*> mamba_writeback_nodes;

    if (match_result_.device.DepthInPage() > match_result_.host.DepthInPage()) {
        std::vector<TreeNode*> write_diff = match_result_.NodesWithout<ResourceType::Host>();
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
        StepCommitRequest materialization_request{
            .materialize_host_writeback =
                HostWritebackMaterializationRequest{
                    .write_diff = &write_diff,
                    .ensure_capacity_before_allocate = false,
                },
        };
        StepCommitResult materialization_result = hybrid_prefix_cache_.StepCommit(std::move(materialization_request));
        if (!materialization_result.ok) {
            throw std::logic_error("ScheduleRetractEvent: failed to allocate host pages for device cache writeback");
        }
        pages_to_transfer = BuildWriteBackPairs(write_diff);
        pages_to_transfer.insert(pages_to_transfer.end(),
                                 std::make_move_iterator(materialization_result.cache_transfer_pairs.begin()),
                                 std::make_move_iterator(materialization_result.cache_transfer_pairs.end()));
        writeback_nodes = std::move(write_diff);
        mamba_writeback_nodes = std::move(materialization_result.mamba_writeback_nodes);
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    } else {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    }

    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();

    (void)hybrid_prefix_cache_.StepCommit({
        .publish_tree_owned_request_state =
            TreeOwnedRequestStatePublicationRequest{
                .terminal = match_result_.device.last_node,
                .local_mamba_allocator_owner = &local_mamba_allocator,
            },
    });

    return Retracting{token_container,
                      page_size,
                      std::move(host_node_ref),
                      std::move(device_node_ref),
                      std::move(local_allocator),
                      std::move(pages_to_transfer),
                      std::move(writeback_nodes),
                      std::move(mamba_writeback_nodes),
                      std::move(local_mamba_allocator)};
}

Retracting ScheduleRetractEvent::operator()(Decoding&& state) {
    return applyRetract(std::move(state));
}

Retracting ScheduleRetractEvent::operator()(PrefillDone&& state) {
    return applyRetract(std::move(state));
}

}  // namespace tokenspeed::fsm
