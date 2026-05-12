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
#include "resource/radix_tree/tree_node.h"
#include "scheduler/execution_event.h"
#include "scheduler/operations/cache.h"
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/types.h"

namespace {

std::int32_t CeilDivPositive(std::int32_t numer, std::int32_t denom) {
    if (numer <= 0) return 0;
    return (numer + denom - 1) / denom;
}

}  // namespace

namespace tokenspeed {

Scheduler::Scheduler(SchedulerConfig config)
    : config_{std::move(config)},
      device_allocator_{config_.page_size, config_.device_allocator.total_pages},
      host_allocator_{config_.page_size, config_.host_allocator.total_pages},
      kv_prefix_cache_{&device_allocator_, &host_allocator_, config_.enable_l3_storage},
      req_pool_allocator_{config_.max_batch_size} {
    if (auto* env = std::getenv("SPDLOG_LEVEL")) {
        std::string level_str{env};
        spdlog::level::level_enum level = spdlog::level::from_str(level_str);
        spdlog::set_level(level);
    }

    if (config_.enable_kv_cache_events) {
        kv_prefix_cache_.SetKvEventSink([this](KvCacheEvent event) { kv_events_.push_back(std::move(event)); });
    }

    const std::int32_t num_mamba_slots =
        config_.enable_mamba ? config_.mamba_pool_total_chunks : config_.num_mamba_slots;
    if (num_mamba_slots > 0) {
        mamba_allocator_.emplace(num_mamba_slots);
        if (config_.role != Role::kD) {
            hybrid_prefix_cache_.emplace(kv_prefix_cache_, &*mamba_allocator_, config_.mamba_cache_chunk_size);
            kv_prefix_cache_.GetDeviceManager().SetEvictionCallback(
                [this](TreeNode* node) { hybrid_prefix_cache_->OnKVEvict(node); });
        }
    }

    for (const auto& cfg : config_.paged_cache_groups) {
        PagedCacheGroupConfig copy = cfg;
        copy.Validate();
        std::string gid = copy.group_id;
        auto [_, inserted] =
            paged_cache_allocators_.emplace(gid, std::make_unique<PagedCacheGroupAllocator>(std::move(copy)));
        if (!inserted) {
            throw std::invalid_argument("Scheduler: duplicate paged cache group_id: " + gid);
        }
    }
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
    MatchResult result = kv_prefix_cache_.Match(token_pages);
    const std::int32_t host_matched = result.host.DepthInPage();
    if (host_matched >= static_cast<std::int32_t>(num_pages)) {
        return {};
    }
    const auto& hashes = result.host.last_node->PageHashes();
    std::string prior = hashes.empty() ? std::string{} : hashes.back();

    return ComputePagedHashes(
        std::vector<std::span<const std::int32_t>>(token_pages.begin() + host_matched, token_pages.end()), prior);
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
    return device_allocator_.AvailablePages();
}

std::size_t Scheduler::ActiveKvPages() const {
    std::unordered_set<std::int32_t> active_pages;
    for (const auto& [_, req] : requests_) {
        if (req->Is<fsm::Prefilling>() || req->Is<fsm::PrefillDone>() || req->Is<fsm::Decoding>()) {
            for (std::int32_t page : req->GetOccupiedPages()) {
                active_pages.insert(page);
            }
        }
    }
    return active_pages.size();
}

std::vector<std::string> Scheduler::PagedCacheGroupIds() const {
    std::vector<std::string> ids;
    ids.reserve(paged_cache_allocators_.size());
    for (const auto& [gid, _] : paged_cache_allocators_) {
        ids.push_back(gid);
    }
    return ids;
}

std::int32_t Scheduler::PagedCacheGroupTotalPages(const std::string& group_id) const {
    auto it = paged_cache_allocators_.find(group_id);
    if (it == paged_cache_allocators_.end()) {
        throw std::out_of_range("Scheduler::PagedCacheGroupTotalPages: group_id not configured");
    }
    return it->second->TotalPages();
}

std::int32_t Scheduler::PagedCacheGroupAvailablePages(const std::string& group_id) const {
    auto it = paged_cache_allocators_.find(group_id);
    if (it == paged_cache_allocators_.end()) {
        throw std::out_of_range("Scheduler::PagedCacheGroupAvailablePages: group_id not configured");
    }
    return it->second->AvailablePages();
}

std::int64_t Scheduler::PagedCacheGroupFailedAllocCount(const std::string& group_id) const {
    auto it = paged_cache_allocators_.find(group_id);
    if (it == paged_cache_allocators_.end()) {
        throw std::out_of_range("Scheduler::PagedCacheGroupFailedAllocCount: group_id not configured");
    }
    return it->second->FailedAllocCount();
}

std::vector<std::int32_t> Scheduler::GetRequestPagedCachePageIds(const std::string& request_id,
                                                                 const std::string& group_id) const {
    if (paged_cache_allocators_.find(group_id) == paged_cache_allocators_.end()) {
        throw std::out_of_range("Scheduler::GetRequestPagedCachePageIds: group_id not configured");
    }
    auto req_it = request_paged_cache_tables_.find(request_id);
    if (req_it == request_paged_cache_tables_.end()) {
        return {};
    }
    auto group_it = req_it->second.find(group_id);
    if (group_it == req_it->second.end()) {
        return {};
    }
    return group_it->second.PageIds();
}

std::int32_t Scheduler::GetRequestPagedCacheBaseLogicalPage(const std::string& request_id,
                                                            const std::string& group_id) const {
    if (paged_cache_allocators_.find(group_id) == paged_cache_allocators_.end()) {
        throw std::out_of_range("Scheduler::GetRequestPagedCacheBaseLogicalPage: group_id not configured");
    }
    auto req_it = request_paged_cache_tables_.find(request_id);
    if (req_it == request_paged_cache_tables_.end()) {
        return 0;
    }
    auto group_it = req_it->second.find(group_id);
    if (group_it == req_it->second.end()) {
        return 0;
    }
    return group_it->second.BaseLogicalPage();
}

void Scheduler::acquirePagedCachePagesForRequest(const std::string& request_id, std::int32_t first_raw_position_of_op,
                                                 std::int32_t target_raw_tokens_exclusive) {
    if (paged_cache_allocators_.empty()) return;
    auto& tables = request_paged_cache_tables_[request_id];
    for (const auto& [group_id, allocator] : paged_cache_allocators_) {
        auto it = tables.find(group_id);
        if (it == tables.end()) {
            it = tables.emplace(group_id, PagedCacheGroupTable(allocator.get())).first;
        }
        const auto& cfg = allocator->Config();
        if (cfg.retention == PagedCacheGroupConfig::Retention::SlidingWindow && cfg.sliding_window_tokens.has_value()) {
            const std::int32_t lower = std::max(0, first_raw_position_of_op - *cfg.sliding_window_tokens + 1);
            it->second.ReleaseSkipped(lower);
        }
        it->second.Acquire(target_raw_tokens_exclusive);
    }
}

PagedCacheGroupAdmission Scheduler::checkPagedCacheGroupAdmission(
    const std::string& request_id, std::int32_t first_raw_position_of_op, std::int32_t target_raw_tokens_exclusive,
    const std::map<std::string, std::int32_t>& simulated_free) const {
    PagedCacheGroupAdmission result;
    if (paged_cache_allocators_.empty() || target_raw_tokens_exclusive < 0) {
        return result;
    }

    auto req_it = request_paged_cache_tables_.find(request_id);
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        const auto& cfg = allocator->Config();
        const std::int32_t raw_per_page = cfg.RawTokensPerPage();
        if (cfg.entry_stride_tokens <= 0 || cfg.rows_per_page <= 0 || raw_per_page <= 0) {
            continue;
        }

        const std::int32_t entries = CeilDivPositive(target_raw_tokens_exclusive, cfg.entry_stride_tokens);
        const std::int32_t required = (entries + cfg.rows_per_page - 1) / cfg.rows_per_page;

        std::int32_t current_size = 0;
        std::int32_t current_active = 0;
        std::int32_t already_released = 0;
        if (req_it != request_paged_cache_tables_.end()) {
            auto t_it = req_it->second.find(gid);
            if (t_it != req_it->second.end()) {
                current_size = t_it->second.Size();
                current_active = t_it->second.ActivePagesCount();
                already_released = t_it->second.ReleasedPagesCount();
            }
        }

        std::int32_t releasable = 0;
        if (cfg.retention == PagedCacheGroupConfig::Retention::SlidingWindow && cfg.sliding_window_tokens.has_value()) {
            const std::int32_t lower = std::max(0, first_raw_position_of_op - *cfg.sliding_window_tokens + 1);
            const std::int32_t target_releases = lower / raw_per_page;
            releasable = std::max(0, target_releases - already_released);
            releasable = std::min(releasable, current_active);
        }

        // Absolute coverage = already_released (base) + live size.
        const std::int32_t absolute_have = already_released + current_size;
        const std::int32_t new_pages = std::max(0, required - absolute_have);
        std::int32_t free = allocator->AvailablePages();
        auto sf_it = simulated_free.find(gid);
        if (sf_it != simulated_free.end()) {
            free = sf_it->second;
        }

        result.releasable_pages[gid] = releasable;
        result.new_pages_needed[gid] = new_pages;
        if (free + releasable < new_pages) {
            result.ok = false;
        }
    }
    return result;
}

std::map<std::string, std::int32_t> Scheduler::initialPagedCacheGroupSimulatedFree() const {
    std::map<std::string, std::int32_t> out;
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        out[gid] = allocator->AvailablePages();
    }
    return out;
}

void Scheduler::applyPagedCacheGroupAdmissionDebit(std::map<std::string, std::int32_t>& simulated_free,
                                                   const PagedCacheGroupAdmission& admission) {
    for (const auto& [gid, releasable] : admission.releasable_pages) {
        simulated_free[gid] += releasable;
    }
    for (const auto& [gid, new_pages] : admission.new_pages_needed) {
        simulated_free[gid] -= new_pages;
    }
}

void Scheduler::releasePagedCachePagesForRequest(const std::string& request_id) {
    auto it = request_paged_cache_tables_.find(request_id);
    if (it == request_paged_cache_tables_.end()) return;
    for (auto& [_, table] : it->second) {
        table.ReleaseAll();
    }
    request_paged_cache_tables_.erase(it);
}

// Snapshot the per-group page ids the request currently owns into op.
// For sliding groups page_ids are compact (live-only) and a base
// logical-page offset is emitted alongside; full-history groups omit the
// offset (implicit 0).
void Scheduler::populatePagedCachePagesForOp(ForwardOperationBase& op_base) const {
    if (paged_cache_allocators_.empty()) {
        return;
    }
    auto req_it = request_paged_cache_tables_.find(op_base.request_id);
    for (const auto& [gid, allocator] : paged_cache_allocators_) {
        std::vector<std::int32_t> pages;
        std::int32_t base_offset = 0;
        if (req_it != request_paged_cache_tables_.end()) {
            auto table_it = req_it->second.find(gid);
            if (table_it != req_it->second.end()) {
                pages = table_it->second.PageIds();
                base_offset = table_it->second.BaseLogicalPage();
            }
        }
        op_base.paged_cache_pages[gid] = std::move(pages);
        if (allocator->Config().retention == PagedCacheGroupConfig::Retention::SlidingWindow) {
            op_base.paged_cache_page_base_offsets[gid] = base_offset;
        }
    }
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
            cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();
            CacheOpSpec spec;
            spec.request_id = id;
            cache_op_tracker_[op_id] = std::move(spec);
            ops.push_back(WriteBackOperation{op_id, std::vector<std::tuple<std::int32_t, std::int32_t>>(
                                                        pages_to_transfer.begin(), pages_to_transfer.end())});
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
            releasePagedCachePagesForRequest(id);
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
    bool ok = true;
    const std::int32_t total_device = device_allocator_.TotalPages() - 1;
    std::unordered_map<std::string, std::vector<std::int32_t>> req_pages_map;
    // page_id → (owner_req_id, state_name) for duplicate tail-page reporting
    std::unordered_map<std::int32_t, std::pair<std::string, std::string>> page_owner;

    for (auto& [id, req] : requests_) {
        std::string state = req->StateName();
        std::vector<std::int32_t> pages = req->GetLocalAllocatorPages();
        if (pages.empty()) continue;
        req_pages_map[id] = pages;

        for (std::int32_t p : pages) {
            auto [it, inserted] = page_owner.emplace(p, std::make_pair(id, state));
            if (!inserted) {
                spdlog::error("[check_mem] DEVICE TAIL PAGE OVERLAP: page={}  req1={}({})  req2={}({})", p,
                              it->second.first, it->second.second, id, state);
                ok = false;
            }
        }
    }

    // ── 2. Collect pages in radix tree ───────────────────────────────────────
    auto tree_device_pages = kv_prefix_cache_.CollectAllPages<ResourceType::Device>();

    // 2a. Check for duplicate page_ids inside the tree itself
    for (auto& [page, cnt] : tree_device_pages) {
        if (cnt > 1) {
            spdlog::error("[check_mem] DEVICE TREE DUPLICATE: page={} appears {} times in radix tree", page, cnt);
            ok = false;
        }
    }

    std::int32_t tree_device_total = static_cast<std::int32_t>(tree_device_pages.size());

    std::int32_t req_device_total = 0;
    for (auto& [id, pages] : req_pages_map) req_device_total += static_cast<std::int32_t>(pages.size());

    std::int32_t free_device = device_allocator_.AvailablePages();

    if (tree_device_total + req_device_total + free_device != total_device) {
        spdlog::error("[check_mem] DEVICE PAGE ACCOUNTING MISMATCH: tree={} req={} free={} sum={} total={}",
                      tree_device_total, req_device_total, free_device,
                      tree_device_total + req_device_total + free_device, total_device);
        ok = false;
    }

    // ── 4. Per-request: page ids must be in [1, total] ────────────────────
    // PageAllocator starts from page id 1 (0 is reserved as invalid/null).
    for (auto& [id, pages] : req_pages_map) {
        for (std::int32_t p : pages) {
            if (p <= 0 || p > total_device) {
                spdlog::error("[check_mem] INVALID DEVICE PAGE id={} for req={} (valid range [1,{}])", p, id,
                              total_device);
                ok = false;
            }
        }
    }
    for (auto& [p, cnt] : tree_device_pages) {
        if (p <= 0 || p > total_device) {
            spdlog::error("[check_mem] INVALID DEVICE PAGE id={} in radix tree (valid range [1,{}])", p, total_device);
            ok = false;
        }
    }

    // ── 5. Summary ────────────────────────────────────────────────────────────
    if (!ok) {
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
