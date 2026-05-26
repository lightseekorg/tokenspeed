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

#include "scheduler/scheduler.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/radix_tree.h"
#include "scheduler/execution_event.h"
#include "scheduler/operations/cache.h"
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "scheduler/request_cache_context.h"
#include "scheduler/request_spec.h"
#include "scheduler/types.h"

namespace tokenspeed {

namespace {

std::optional<MambaChunkAllocator> MakeMambaAllocator(const SchedulerConfig& config) {
    if (config.enable_mamba && config.mamba_pool_total_chunks > 0) {
        return std::optional<MambaChunkAllocator>{std::in_place, config.mamba_pool_total_chunks};
    }
    return std::nullopt;
}

MambaChunkAllocator* MambaAdjunctAllocator(std::optional<MambaChunkAllocator>& allocator,
                                           const SchedulerConfig& config) {
    const bool has_mamba_pool = allocator.has_value();
    const bool has_mamba_adjunct = has_mamba_pool && config.role != Role::kD;
    return has_mamba_adjunct ? &*allocator : nullptr;
}

std::optional<MambaHostAllocator> MakeMambaHostAllocator(const SchedulerConfig& config) {
    if (config.enable_mamba && config.mamba_pool_total_chunks > 0 && config.enable_mamba_l2 &&
        config.mamba_l2_host_slots > 0) {
        return std::optional<MambaHostAllocator>{std::in_place, config.mamba_l2_host_slots};
    }
    return std::nullopt;
}

MambaHostAllocator* MambaHostAdjunctAllocator(std::optional<MambaHostAllocator>& host_allocator,
                                              std::optional<MambaChunkAllocator>& device_allocator,
                                              const SchedulerConfig& config) {
    const bool has_mamba_adjunct = device_allocator.has_value() && config.role != Role::kD;
    return has_mamba_adjunct && host_allocator.has_value() ? &*host_allocator : nullptr;
}

struct RequestLocalKVPagesSnapshot {
    std::string request_id;
    std::string state_name;
    std::vector<std::int32_t> pages;
};

bool ValidateDeviceMemoryDiagnostics(const std::vector<RequestLocalKVPagesSnapshot>& request_pages,
                                     const CacheDeviceMemoryDiagnosticsSnapshot& device_snapshot) {
    bool ok = true;
    const std::int32_t total_device = device_snapshot.total_device_pages;
    // page_id -> (owner_req_id, state_name) for duplicate tail-page reporting
    std::unordered_map<std::int32_t, std::pair<std::string, std::string>> page_owner;

    for (const auto& snapshot : request_pages) {
        for (std::int32_t p : snapshot.pages) {
            auto [it, inserted] = page_owner.emplace(p, std::make_pair(snapshot.request_id, snapshot.state_name));
            if (!inserted) {
                spdlog::error("[check_mem] DEVICE TAIL PAGE OVERLAP: page={}  req1={}({})  req2={}({})", p,
                              it->second.first, it->second.second, snapshot.request_id, snapshot.state_name);
                ok = false;
            }
        }
    }

    // Check for duplicate page_ids inside the tree itself.
    for (auto& [page, cnt] : device_snapshot.tree_device_pages) {
        if (cnt > 1) {
            spdlog::error("[check_mem] DEVICE TREE DUPLICATE: page={} appears {} times in radix tree", page, cnt);
            ok = false;
        }
    }

    const std::int32_t tree_device_total = static_cast<std::int32_t>(device_snapshot.tree_device_pages.size());

    std::int32_t req_device_total = 0;
    for (const auto& snapshot : request_pages) {
        req_device_total += static_cast<std::int32_t>(snapshot.pages.size());
    }

    const std::int32_t free_device = device_snapshot.free_device_pages;

    if (tree_device_total + req_device_total + free_device != total_device) {
        spdlog::error("[check_mem] DEVICE PAGE ACCOUNTING MISMATCH: tree={} req={} free={} sum={} total={}",
                      tree_device_total, req_device_total, free_device,
                      tree_device_total + req_device_total + free_device, total_device);
        ok = false;
    }

    // PageAllocator starts from page id 1; 0 is reserved as invalid/null.
    for (const auto& snapshot : request_pages) {
        for (std::int32_t p : snapshot.pages) {
            if (p <= 0 || p > total_device) {
                spdlog::error("[check_mem] INVALID DEVICE PAGE id={} for req={} (valid range [1,{}])", p,
                              snapshot.request_id, total_device);
                ok = false;
            }
        }
    }
    for (const auto& entry : device_snapshot.tree_device_pages) {
        const std::int32_t p = entry.first;
        if (p <= 0 || p > total_device) {
            spdlog::error("[check_mem] INVALID DEVICE PAGE id={} in radix tree (valid range [1,{}])", p, total_device);
            ok = false;
        }
    }

    return ok;
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : config_{std::move(config)},
      device_allocator_{config_.page_size, config_.device_allocator.total_pages},
      host_allocator_{config_.page_size, config_.host_allocator.total_pages},
      mamba_allocator_{MakeMambaAllocator(config_)},
      mamba_host_allocator_{MakeMambaHostAllocator(config_)},
      kv_prefix_cache_{&device_allocator_, &host_allocator_, config_.enable_l3_storage, config_.disable_prefix_cache},
      hybrid_prefix_cache_{kv_prefix_cache_, device_allocator_, MambaAdjunctAllocator(mamba_allocator_, config_),
                           config_.mamba_cache_chunk_size,
                           MambaHostAdjunctAllocator(mamba_host_allocator_, mamba_allocator_, config_)},
      req_pool_allocator_{config_.max_batch_size} {
    if (auto* env = std::getenv("SPDLOG_LEVEL")) {
        std::string level_str{env};
        spdlog::level::level_enum level = spdlog::level::from_str(level_str);
        spdlog::set_level(level);
    }

    if (config_.enable_kv_cache_events) {
        hybrid_prefix_cache_.SetKvEventSink([this](KvCacheEvent event) { kv_events_.push_back(std::move(event)); });
    }
    std::optional<std::span<const std::string>> required_paged_cache_groups;
    if (config_.prefix_cache_adjunct.has_value()) {
        required_paged_cache_groups = std::span<const std::string>{config_.prefix_cache_adjunct->required_groups};
    }
    hybrid_prefix_cache_.ConfigurePagedCacheAdjunct(std::span<const PagedCacheGroupConfig>{config_.paged_cache_groups},
                                                    required_paged_cache_groups);
}

Scheduler::~Scheduler() {
    hybrid_prefix_cache_.SetKvEventSink({});
}

std::vector<KvCacheEvent> Scheduler::DrainKvEvents() {
    std::vector<KvCacheEvent> events;
    events.swap(kv_events_);
    return events;
}

std::vector<std::string> Scheduler::CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match) {
    const std::int32_t page_size = config_.page_size;
    const std::size_t num_pages = input_tokens.size() / page_size;
    std::vector<std::span<const std::int32_t>> token_pages;
    token_pages.reserve(num_pages);
    for (std::size_t i = 0; i < num_pages; ++i) {
        token_pages.emplace_back(input_tokens.data() + i * page_size, page_size);
    }
    if (!apply_match) {
        return ComputePagedHashes(token_pages, "");
    }
    const auto raw_host_seed = hybrid_prefix_cache_.LookupRawHostStorageHashSeed(token_pages);
    const std::int32_t host_matched = raw_host_seed.host_matched_pages;
    if (host_matched >= static_cast<std::int32_t>(num_pages)) {
        return {};
    }

    return ComputePagedHashes(
        std::vector<std::span<const std::int32_t>>(token_pages.begin() + host_matched, token_pages.end()),
        raw_host_seed.prior_hash_seed);
}

void Scheduler::SubmitRequests(const std::vector<RequestSpec>& request_specs) {
    for (const auto& spec : request_specs) {
        auto req = std::make_unique<Request>(spec, config_.page_size, config_.role);
        requests_.emplace(spec.request_id, std::move(req));
    }
}

std::size_t Scheduler::WaitingSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Submitted>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::DecodingSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Decoding>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::PrefillSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Prefilling>() || req->Is<fsm::PrefillDone>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::RetractedSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Retracting>() || req->Is<fsm::Retracted>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::AvailableKvPages() const {
    return hybrid_prefix_cache_.Stats().available_device_pages;
}

std::size_t Scheduler::ActiveKvPages() const {
    std::unordered_set<std::int32_t> active_pages;
    for (const auto& [_, req] : requests_) {
        if (req->Is<fsm::Prefilling>() || req->Is<fsm::PrefillDone>() || req->Is<fsm::Decoding>()) {
            RequestCacheContext cache_context(*req);
            for (std::int32_t page : cache_context.OccupiedPagesSnapshot()) {
                active_pages.insert(page);
            }
        }
    }
    return active_pages.size();
}

std::vector<std::string> Scheduler::PagedCacheGroupIds() const {
    return hybrid_prefix_cache_.Stats().paged_cache_group_ids;
}

std::int32_t Scheduler::PagedCacheGroupTotalPages(const std::string& group_id) const {
    return hybrid_prefix_cache_.Stats({.paged_cache_group_ids = {group_id}}).paged_cache_total_pages.at(group_id);
}

std::int32_t Scheduler::PagedCacheGroupAvailablePages(const std::string& group_id) const {
    return hybrid_prefix_cache_.Stats({.paged_cache_group_ids = {group_id}}).paged_cache_available_pages.at(group_id);
}

std::int64_t Scheduler::PagedCacheGroupFailedAllocCount(const std::string& group_id) const {
    return hybrid_prefix_cache_.Stats({.paged_cache_group_ids = {group_id}})
        .paged_cache_failed_alloc_count.at(group_id);
}

std::vector<std::int32_t> Scheduler::GetRequestPagedCachePageIds(const std::string& request_id,
                                                                 const std::string& group_id) const {
    return hybrid_prefix_cache_.Stats({.request_id = request_id, .paged_cache_group_ids = {group_id}})
        .request_paged_cache_page_ids.at(group_id);
}

std::int32_t Scheduler::GetRequestPagedCacheBaseLogicalPage(const std::string& request_id,
                                                            const std::string& group_id) const {
    return hybrid_prefix_cache_.Stats({.request_id = request_id, .paged_cache_group_ids = {group_id}})
        .request_paged_cache_base_logical_page.at(group_id);
}

std::int32_t Scheduler::GetRequestTokenSize(const std::string& id) const {
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return -1;
    }
    return it->second->TokenSize();
}

std::vector<WriteBackOperation> Scheduler::newWriteBackOperation(
    std::unordered_map<std::string, std::unique_ptr<Request>>& requests) {
    std::vector<WriteBackOperation> ops;
    if (config_.disable_l2_cache) {
        return ops;
    }
    for (auto& [id, req] : requests) {
        if (!req->Is<fsm::Draining>()) continue;
        const auto& pages_to_transfer = req->GetPagesToTransfer<fsm::Draining>();

        if (!pages_to_transfer.empty()) {
            cache_op_id op_id = hybrid_prefix_cache_.AllocateCacheOpId();
            CacheOpSpec spec;
            spec.request_id = id;
            spec.writeback_nodes = req->GetWriteBackNodes<fsm::Draining>();
            cache_op_tracker_[op_id] = std::move(spec);
            ops.push_back(WriteBackOperation{
                op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end())});
            req->Apply(fsm::CommitDrainingEvent{});
        } else {
            req->Apply(fsm::AbortEvent{});
        }
    }
    return ops;
}

ExecutionPlan Scheduler::NextExecutionPlan() {
    ExecutionPlan plan;

    std::vector<WriteBackOperation> write_back_ops;
    write_back_ops = std::move(newWriteBackOperation(requests_));

    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Finished>()) {
            hybrid_prefix_cache_.FinishRequest(id);
        }
    }
    std::erase_if(requests_, [](const auto& req) { return req.second->template Is<fsm::Finished>(); });

    std::vector<Request*> candidates;
    for (auto& [id, req] : requests_) {
        if (!req->Is<fsm::Draining>() && !req->Is<fsm::Prefetching>() && !req->Is<fsm::Retracting>() &&
            !req->Is<fsm::WritingBack>()) {
            candidates.push_back(req.get());
        }
    }

    auto [fwd_ops, cache_ops] = newForwardOperation(candidates);
    plan.With(FlatForwardOperation{std::move(fwd_ops)});

    // Merge retract write-backs (if any) into the Draining write-back list, then emit once.
    if (auto* wb = std::get_if<std::vector<WriteBackOperation>>(&cache_ops)) {
        write_back_ops.insert(write_back_ops.end(), std::make_move_iterator(wb->begin()),
                              std::make_move_iterator(wb->end()));
    }
    if (!write_back_ops.empty()) {
        plan.With(CacheOperation{FlatWriteBackOperation{write_back_ops}});
    }
    if (auto* lb = std::get_if<std::vector<LoadBackOperation>>(&cache_ops)) {
        if (!lb->empty()) {
            plan.With(CacheOperation{FlatLoadBackOperation{*lb}});
        }
    }
    if (std::getenv("DEBUG_MEM")) {
        check_device_mem();
    }
    return plan;
}

void Scheduler::check_device_mem() {
    std::vector<RequestLocalKVPagesSnapshot> request_page_snapshots;

    for (auto& [id, req] : requests_) {
        RequestCacheContext cache_context(*req);
        std::vector<std::int32_t> pages = cache_context.LocalKVPagesSnapshot();
        if (pages.empty()) continue;
        request_page_snapshots.push_back(RequestLocalKVPagesSnapshot{
            .request_id = id,
            .state_name = req->StateName(),
            .pages = std::move(pages),
        });
    }

    auto stats_snapshot = hybrid_prefix_cache_.Stats({.include_device_memory_diagnostics = true});
    if (!stats_snapshot.device_memory_diagnostics.has_value()) {
        throw std::runtime_error("Scheduler::check_device_mem: missing diagnostics snapshot");
    }
    auto device_snapshot = std::move(*stats_snapshot.device_memory_diagnostics);

    // ── 5. Summary ────────────────────────────────────────────────────────────
    if (!ValidateDeviceMemoryDiagnostics(request_page_snapshots, device_snapshot)) {
        throw std::runtime_error("Scheduler::CheckMem: device page accounting check failed");
    }
}

void Scheduler::Advance(const ExecutionEvent& event) {
    auto dispatch = [this](const auto& inner) { handleEvent(inner); };
    for (const auto& item : event.Events()) {
        std::visit([&](const auto& outer) { std::visit(dispatch, outer); }, item);
    }
}

}  // namespace tokenspeed
