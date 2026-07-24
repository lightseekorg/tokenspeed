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
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/token_container.h"
#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/radix_tree/node_range.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"

#if TOKENSPEED_FLAT_KVCACHE
#include "cache/forward_cache_ops.h"
#include "scheduler/page_hasher.h"
#endif

namespace {

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

template <tokenspeed::ResourceType RType>
tokenspeed::TreeNode* LastNodeWithResourceOrRoot(tokenspeed::TreeNode* node) {
    for (tokenspeed::TreeNode* candidate = node; candidate != nullptr; candidate = candidate->Parent()) {
        if (candidate->IsRoot()) return candidate;
        if constexpr (RType == tokenspeed::ResourceType::Device) {
            if (candidate->OnDevice()) {
                return candidate;
            }
        } else {
            if (candidate->OnHost()) {
                return candidate;
            }
        }
    }
    return nullptr;
}

struct KVResourceNodes {
    tokenspeed::TreeNode* device_node{nullptr};
    tokenspeed::TreeNode* host_node{nullptr};
    std::int32_t device_page_size{0};
    std::int32_t host_page_size{0};

    std::int32_t DeviceDepthInPage() const {
        return device_node == nullptr ? 0 : device_node->DepthInPage(device_page_size);
    }

    std::int32_t HostDepthInPage() const { return host_node == nullptr ? 0 : host_node->DepthInPage(host_page_size); }

    std::vector<tokenspeed::TreeNode*> NodesWithoutDevice() const {
        std::vector<tokenspeed::TreeNode*> result;
        for (tokenspeed::TreeNode* node : tokenspeed::LeafToRoot(host_node)) {
            if (node->OnDevice()) break;
            result.push_back(node);
        }
        return result;
    }
};

KVResourceNodes GetKVResourceNodes(const tokenspeed::MatchResult& match) {
    return KVResourceNodes{
        LastNodeWithResourceOrRoot<tokenspeed::ResourceType::Device>(match.device.last_node),
        LastNodeWithResourceOrRoot<tokenspeed::ResourceType::Host>(match.host.last_node),
        match.device.page_size,
        match.host.page_size,
    };
}

std::vector<tokenspeed::TreeNode*> MambaNodesForTransferPairs(const std::vector<tokenspeed::TreeNode*>& candidates,
                                                              const std::vector<tokenspeed::TransferPair>& transfers) {
    std::unordered_set<std::int32_t> src_slots;
    for (const auto& transfer : transfers) {
        if (transfer.kind == tokenspeed::CacheKind::kMamba) {
            src_slots.insert(transfer.src);
        }
    }
    std::vector<tokenspeed::TreeNode*> nodes;
    nodes.reserve(src_slots.size());
    for (tokenspeed::TreeNode* node : candidates) {
        if (node != nullptr && node->HasMamba() && src_slots.find(node->MambaSlotIndex()) != src_slots.end()) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

std::vector<tokenspeed::TreeNode*> PagedCacheSnapshotNodesForWriteBack(tokenspeed::TreeNode* last_node) {
    std::vector<tokenspeed::TreeNode*> nodes;
    for (tokenspeed::TreeNode* node : tokenspeed::LeafToRoot(last_node)) {
        if (node == nullptr) continue;
        if (node->HasPagedCacheSnapshot() && !node->HasPagedCacheHostSnapshot() &&
            !node->HasPagedCachePendingHostSnapshot()) {
            nodes.push_back(node);
        }
    }
    return nodes;
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

void ReleaseFailedHostWriteBack(tokenspeed::KVPrefixCache* kv_prefix_cache,
                                tokenspeed::HybridPrefixCache* hybrid_prefix_cache,
                                const std::vector<tokenspeed::TreeNode*>& host_writeback_nodes) {
    if (kv_prefix_cache == nullptr || host_writeback_nodes.empty()) return;
    kv_prefix_cache->ReleaseHostResources(host_writeback_nodes, [hybrid_prefix_cache](tokenspeed::TreeNode* n) {
        if (hybrid_prefix_cache != nullptr) {
            hybrid_prefix_cache->OnKVHostEvict(n);
        }
    });
}

void RollbackHostWriteBack(tokenspeed::KVPrefixCache* kv_prefix_cache,
                           tokenspeed::HybridPrefixCache* hybrid_prefix_cache, tokenspeed::fsm::WritingBack& state) {
    if (hybrid_prefix_cache != nullptr) {
        hybrid_prefix_cache->OnMambaHostWriteBackDone(state.MambaWriteBackNodes(), false);
        hybrid_prefix_cache->OnPagedCacheHostWriteBackDone(state.PagedCacheWriteBackNodes(), false);
    }
    state.DropHostNodeRef();
    ReleaseFailedHostWriteBack(kv_prefix_cache, hybrid_prefix_cache, state.HostWriteBackNodes());
    state.DropDeviceNodeRef();
}

bool ShouldPublishMambaCheckpoint(tokenspeed::HybridPrefixCache* hybrid_cache, std::int32_t chunk_begin,
                                  std::int32_t chunk_size, std::int32_t page_size) {
    if (hybrid_cache == nullptr || chunk_size <= 0 || page_size <= 0) return false;
    const std::int32_t final_len = chunk_begin + chunk_size;
    const std::int32_t last_inserted_len = (final_len / page_size) * page_size;
    if (last_inserted_len <= chunk_begin) return false;
    if (last_inserted_len == final_len) return true;

    const std::int32_t track_len = last_inserted_len - chunk_begin;
    return hybrid_cache->AlignMambaCacheSeqlen(track_len) == track_len;
}

}  // namespace

namespace tokenspeed::fsm {

#if TOKENSPEED_FLAT_KVCACHE
namespace {

// The throw makes it noreturn so callers with non-default-constructible returns compile.
[[noreturn]] void FlatRetractUnsupported() {
    _assert(false, "flat path: retract/writeback/loadback unsupported in C slice");
    throw std::logic_error("flat path: retract/writeback/loadback unsupported in C slice");
}

// Base-page hashes one decode step hands to registration, anchored at begin_page.
struct HashesToRegister {
    std::int32_t begin_page{0};
    std::vector<std::string> hashes;
};

// Seed from a full-prefix hash list; the tail keeps the suffix from the last fold-grid boundary.
HashChain MakeHashChain(const std::vector<std::string>& hashes, std::int32_t fold_align_pages) {
    const std::int32_t n = static_cast<std::int32_t>(hashes.size());
    const std::int32_t tail_begin = n / fold_align_pages * fold_align_pages;
    return HashChain{.num_hashed_pages = n,
                     .last_hash = n > 0 ? hashes.back() : std::string{},
                     .tail_begin_page = tail_begin,
                     .tail = {hashes.begin() + tail_begin, hashes.end()}};
}

// Hash the newly filled pages onto the chain; the returned batch reaches back to the fold grid
// (fold_align_pages = lcm/base) so coarse groups fold the blocks completed this step. The
// re-covered overlap is idempotent (m == 1 degenerates to exactly the fresh pages).
HashesToRegister AdvanceHashChain(HashChain& chain, const std::vector<std::span<const std::int32_t>>& paged,
                                  std::int32_t filled_pages, std::int32_t fold_align_pages) {
    _assert(filled_pages > chain.num_hashed_pages, "caller must pre-check hash-chain progress");
    _assert(filled_pages <= static_cast<std::int32_t>(paged.size()),
            "flat decode hashing: filled pages exceed the container's full pages");
    const std::vector<std::span<const std::int32_t>> fresh(paged.begin() + chain.num_hashed_pages,
                                                           paged.begin() + filled_pages);
    std::vector<std::string> new_hashes = ComputePagedHashes(fresh, chain.last_hash);

    HashesToRegister batch{.begin_page = chain.tail_begin_page, .hashes = std::move(chain.tail)};
    batch.hashes.insert(batch.hashes.end(), new_hashes.begin(), new_hashes.end());

    const std::int32_t tail_begin = filled_pages / fold_align_pages * fold_align_pages;
    chain.num_hashed_pages = filled_pages;
    chain.last_hash = batch.hashes.back();
    chain.tail_begin_page = tail_begin;
    chain.tail.assign(batch.hashes.begin() + (tail_begin - batch.begin_page), batch.hashes.end());
    return batch;
}

}  // namespace
#endif

void InsertHybridCache(HybridPrefixCache* hybrid_cache,
                       const std::vector<std::span<const std::int32_t>>& full_paged_tokens,
                       std::unique_ptr<DeviceNodeRef>& device_node_ref, LocalKVAllocator* local_kv_allocator,
                       LocalMambaAllocator* local_mamba_allocator, std::int32_t chunk_begin, std::int32_t chunk_size,
                       std::int32_t page_size, const std::vector<std::int32_t>* prefix_pages_override) {
    if (hybrid_cache == nullptr) return;

    std::vector<std::int32_t> computed_prefix_pages;
    const std::vector<std::int32_t>* prefix_pages = prefix_pages_override;
    if (prefix_pages == nullptr) {
        computed_prefix_pages = DevicePagesFromRoot(device_node_ref->Node());
        prefix_pages = &computed_prefix_pages;
    }
    std::int32_t new_page_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages->size());
    if (new_page_count <= 0) {
        if (local_mamba_allocator != nullptr && local_mamba_allocator->HasCheckpoint()) {
            local_mamba_allocator->DetachCheckpoint();
        }
        return;
    }

    // Allocator may hold fewer pages than new_page_count by insert time; clamp page-aligned so
    // prefix + inserted == full (a count-only clamp would relocate the overflow into Insert's TakeLast).
    const std::vector<std::span<const std::int32_t>>* tokens_for_insert = &full_paged_tokens;
    std::vector<std::span<const std::int32_t>> clamped_tokens;
    const std::int32_t avail = static_cast<std::int32_t>(local_kv_allocator->PageCount());
    if (new_page_count > avail) {
        const std::int32_t keep = static_cast<std::int32_t>(prefix_pages->size()) + avail;
        clamped_tokens.assign(full_paged_tokens.begin(), full_paged_tokens.begin() + keep);
        tokens_for_insert = &clamped_tokens;
        new_page_count = avail;
    }

    OwnedPages pages_to_insert = local_kv_allocator->TakeFirst(new_page_count);
    auto insert_result = hybrid_cache->GetKVPrefixCache().Insert<ResourceType::Device>(
        *tokens_for_insert, *prefix_pages, std::move(pages_to_insert));

    if (local_mamba_allocator != nullptr && local_mamba_allocator->HasCheckpoint()) {
        const bool publish = ShouldPublishMambaCheckpoint(hybrid_cache, chunk_begin, chunk_size, page_size);
        if (publish) {
            hybrid_cache->InsertMamba(insert_result.last_node, local_mamba_allocator->DetachCheckpoint());
        } else {
            local_mamba_allocator->DetachCheckpoint();
        }
    }
    device_node_ref = std::make_unique<DeviceNodeRef>(insert_result.last_node);
}

// Submitted -> PrefillDone / Prefilling
std::variant<PrefillDone, Prefilling> SchedulePrefillFirstChunkEvent::operator()(Submitted&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "SchedulePrefillFirstChunkEvent: flat path requires a coordinator");
    TokenContainer* token_container = state.GetTokenContainer();

    // Slot first: Allocate() throws on exhaustion; order kept for determinism.
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());

    std::vector<BlockTable> tables(coordinator_->NumGroups());
    const std::int32_t device_hit_tokens = flat_hit_.num_common_tokens;
    const std::int32_t hit_tokens = std::max(device_hit_tokens, flat_host_.num_common_tokens);
    coordinator_->ClaimCommonPrefix(tables, std::move(flat_hit_));
    // Extension appends between claim and fresh acquire so slots stay
    // [device hit | host ext | new pages]; ext pages are FULL, composing with the gate.
    flat_load_pairs_ = LoadHostExtension(*coordinator_, tables, std::move(flat_host_));
    // Host boundary is absolute (floor included); default-constructed (no host pool) it is 0.
    // Loaded pages become device-cached now (SWA holes skipped by the IsNull guard); the sink
    // re-collects them and the drain dedupes against the host index. The extension ends at the
    // page-aligned hit boundary, so the state group's final page is exactly the loaded snapshot.
    coordinator_->CacheFullBlocks(tables, flat_ext_hashes_,
                                  /*first_slot=*/device_hit_tokens / state.GetPageSize(),
                                  /*end_tokens=*/hit_tokens);
    // Role kD cannot defer its decode reserve until RemotePrefillDone: the
    // remote transfer may complete after other requests consume the pool.
    const std::int32_t tokens_to_acquire = tokens_this_round_ + (role_ == Role::kD ? decode_input_tokens_ : 0);
    if (!coordinator_->Acquire(tables, tokens_to_acquire)) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    TokenContainer::Window window{.begin = hit_tokens, .size = tokens_this_round_};
    bool is_last_chunk = (window.begin + window.size) == token_container->PrefillSize();
    if (is_last_chunk && role_ != Role::kD) {
        PrefillDone done{token_container, state.GetPageSize(),  nullptr, nullptr, nullptr, std::move(req_pool_index),
                         window,          decode_input_tokens_, nullptr};
        done.SetBlockTables(std::move(tables));
        return done;
    }
    Prefilling prefilling{token_container, state.GetPageSize(),       nullptr, nullptr,
                          nullptr,         std::move(req_pool_index), window,  nullptr};
    prefilling.SetBlockTables(std::move(tables));
    return prefilling;
#else
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    const KVResourceNodes kv_nodes = GetKVResourceNodes(match_result_);
    if (!disable_l2_cache_ && (kv_nodes.HostDepthInPage() > kv_nodes.DeviceDepthInPage())) {
        host_node_ref = std::make_unique<HostNodeRef>(kv_nodes.host_node);
        kv_prefix_cache_->AllocateResourceOfType<ResourceType::Device>(kv_nodes.NodesWithoutDevice());
        device_node_ref = std::make_unique<DeviceNodeRef>(kv_nodes.host_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(kv_nodes.device_node);
    }

    auto local_kv_allocator = std::make_unique<LocalKVAllocator>(device_allocator_, tokens_this_round_);
    // Reserve token slots for draft multi-step decode.
    local_kv_allocator->Acquire(decode_input_tokens_);

    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());

    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator;
    if (mamba_allocator_ != nullptr) {
        local_mamba_allocator = std::make_unique<LocalMambaAllocator>(mamba_allocator_);
        if (!local_mamba_allocator->AllocateWorking()) {
            throw std::logic_error("SchedulePrefillFirstChunkEvent: failed to allocate Mamba working slot");
        }
        if (!local_mamba_allocator->AllocateCheckpoint()) {
            throw std::logic_error("SchedulePrefillFirstChunkEvent: failed to allocate Mamba checkpoint slot");
        }
    }

    TokenContainer* token_container = state.GetTokenContainer();

    std::int32_t window_begin = matched_prefix_len_tokens_;
    if (window_begin < 0) {
        const std::int32_t max_matched_pages =
            disable_l2_cache_ ? match_result_.device.DepthInPage()
                              : std::max(match_result_.device.DepthInPage(), match_result_.host.DepthInPage());
        window_begin = max_matched_pages * state.GetPageSize();
    }
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
#endif
}

// Prefilling -> Prefilling / PrefillDone
std::variant<PrefillDone, Prefilling> SchedulePrefillEvent::operator()(Prefilling&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "SchedulePrefillEvent: flat path requires a coordinator");
    const std::vector<std::string> hashes = FlatWindowPageHashes(state.GetFullPagedTokens(false), state.GetPageSize(),
                                                                 state.window.begin, state.window.size);
    // Prior chunks 0..k-1 (state.window is the PREVIOUS chunk); the gate credited this same slide value.
    const std::int32_t num_computed_tokens = state.window.begin + state.window.size;

    auto tables = std::move(state).TakeBlockTables();
    if (!PrefillChunk(*coordinator_, tables, hashes, tokens_this_round_, num_computed_tokens)) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    TokenContainer::Window window{.begin = state.window.begin + state.window.size, .size = tokens_this_round_};
    bool is_last_chunk = (window.begin + window.size) == state.GetTokenContainer()->PrefillSize();
    if (is_last_chunk) {
        PrefillDone done{state.GetTokenContainer(),
                         state.GetPageSize(),
                         nullptr,
                         nullptr,
                         nullptr,
                         std::move(state).TakeReqPoolIndex(),
                         window,
                         reserve_num_tokens_in_next_schedule_event_,
                         nullptr};
        done.SetBlockTables(std::move(tables));
        return done;
    }
    Prefilling prefilling{state.GetTokenContainer(),
                          state.GetPageSize(),
                          nullptr,
                          nullptr,
                          nullptr,
                          std::move(state).TakeReqPoolIndex(),
                          window,
                          nullptr};
    prefilling.SetBlockTables(std::move(tables));
    return prefilling;
#else
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
    InsertHybridCache(hybrid_prefix_cache_, paged_tokens, device_node_ref, local_kv_allocator.get(),
                      local_mamba_allocator.get(), state.window.begin, state.window.size, state.GetPageSize());
    local_kv_allocator->Acquire(tokens_this_round_);

    if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr) {
        if (!local_mamba_allocator->AllocateCheckpoint()) {
            throw std::logic_error("SchedulePrefillEvent: failed to allocate Mamba checkpoint slot");
        }
    }

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
#endif
}

// PrefillDone -> Decoding: insert prefill pages into tree, then transition to decode.
Decoding ScheduleDecodeEvent::operator()(PrefillDone&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "ScheduleDecodeEvent: flat path requires a coordinator");
    const std::vector<std::string> hashes = FlatWindowPageHashes(state.GetFullPagedTokens(false), state.GetPageSize(),
                                                                 state.window.begin, state.window.size);
    const std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    // Full prefill length (window end == PrefillSize()); the PrefillDone gate credited the same value.
    const std::int32_t num_computed_tokens = state.window.begin + state.window.size;

    auto tables = std::move(state).TakeBlockTables();
    if (!FinalizePrefillAndReserveDecode(*coordinator_, tables, hashes, reserve, num_computed_tokens)) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    Decoding decoding{state.GetTokenContainer(),           state.GetPageSize(),  nullptr, nullptr, nullptr,
                      std::move(state).TakeReqPoolIndex(), decode_input_tokens_, nullptr};
    decoding.SetBlockTables(std::move(tables));
    decoding.SetHashChain(MakeHashChain(hashes, coordinator_->LcmBlockSize() / coordinator_->BaseBlockSize()));
    return decoding;
#else
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
    InsertHybridCache(hybrid_prefix_cache_, paged_tokens, device_node_ref, local_kv_allocator.get(),
                      local_mamba_allocator.get(), state.window.begin, state.window.size, state.GetPageSize());
    if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr) {
        if (!local_mamba_allocator->AllocateCheckpoint()) {
            throw std::logic_error("ScheduleDecodeEvent: failed to allocate Mamba checkpoint slot");
        }
    }

    std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    local_kv_allocator->Acquire(reserve);

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_,          std::move(local_mamba_allocator)};
#endif
}

// Decoding -> Decoding: allocate pages for next decode step.
Decoding ScheduleDecodeEvent::operator()(Decoding&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "ScheduleDecodeEvent: flat path requires a coordinator");
    const std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    // Size() includes this round's pending decode tail; sliding at Size() would free a page its
    // query still reads. scheduleDecode's gate credited the slide with this same value.
    const std::int32_t num_computed_tokens = state.GetTokenContainer()->Size() - decode_input_tokens_;

    HashChain chain = std::move(state).TakeHashChain();
    const std::int32_t filled_pages = num_computed_tokens / state.GetPageSize();
    // A page fills only once every page_size steps; skip the span walk on the other steps.
    HashesToRegister to_register{.begin_page = chain.num_hashed_pages};
    if (filled_pages > chain.num_hashed_pages) {
        to_register = AdvanceHashChain(chain, state.GetFullPagedTokens(false), filled_pages,
                                       coordinator_->LcmBlockSize() / coordinator_->BaseBlockSize());
    }

    auto tables = std::move(state).TakeBlockTables();
    if (!DecodeStep(*coordinator_, tables, to_register.hashes, to_register.begin_page, reserve, num_computed_tokens)) {
        _assert(false, "flat path: allocation failure unsupported in C slice");
    }

    Decoding decoding{state.GetTokenContainer(),           state.GetPageSize(),  nullptr, nullptr, nullptr,
                      std::move(state).TakeReqPoolIndex(), decode_input_tokens_, nullptr};
    decoding.SetBlockTables(std::move(tables));
    decoding.SetHashChain(std::move(chain));
    return decoding;
#else
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    std::int32_t reserve = state.GetReserveNumTokensInNextScheduleEvent();
    local_kv_allocator->Acquire(reserve);

    return Decoding{state.GetTokenContainer(),     state.GetPageSize(),
                    std::move(host_node_ref),      std::move(device_node_ref),
                    std::move(local_kv_allocator), std::move(state).TakeReqPoolIndex(),
                    decode_input_tokens_,          std::move(local_mamba_allocator)};
#endif
}

// Retracted -> Decoding: recover via LoadBack (host -> device).
Decoding ScheduleDecodeFromRetractedEvent::operator()(Retracted&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    FlatRetractUnsupported();
#else
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    const KVResourceNodes kv_nodes = GetKVResourceNodes(match_result_);
    if (kv_nodes.HostDepthInPage() > kv_nodes.DeviceDepthInPage()) {
        host_node_ref = std::make_unique<HostNodeRef>(kv_nodes.host_node);
        if (!kv_prefix_cache_->AllocateResourceOfType<ResourceType::Device>(kv_nodes.NodesWithoutDevice())) {
            // Device allocation failed (race between capacity check and actual alloc).
            throw std::logic_error(
                "ScheduleDecodeFromRetractedEvent: failed to allocate device pages for host cache recovery");
        }
        // This is not a typo
        device_node_ref = std::make_unique<DeviceNodeRef>(kv_nodes.host_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(kv_nodes.device_node);
    }
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_kv_allocator = std::move(state).TakeKVAllocator();
    auto old_mamba_allocator = std::move(state).TakeMambaAllocator();
    old_mamba_allocator.reset();
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator;
    if (mamba_allocator_ != nullptr) {
        local_mamba_allocator = std::make_unique<LocalMambaAllocator>(mamba_allocator_);
        if (!local_mamba_allocator->AllocateWorking()) {
            throw std::logic_error("ScheduleDecodeFromRetractedEvent: failed to allocate Mamba recovery working slot");
        }
        if (!local_mamba_allocator->AllocateCheckpoint()) {
            throw std::logic_error(
                "ScheduleDecodeFromRetractedEvent: failed to allocate Mamba recovery checkpoint slot");
        }
    }
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    local_kv_allocator->Acquire(decode_input_tokens_);
    return Decoding{token_container,
                    page_size,
                    std::move(host_node_ref),
                    std::move(device_node_ref),
                    std::move(local_kv_allocator),
                    std::move(req_pool_index),
                    decode_input_tokens_,
                    std::move(local_mamba_allocator)};
#endif
}

// Decoding/PrefillDone -> Draining/Finished, driven by the Python side's Advance.
template <typename ForwardStateT>
std::variant<Draining, Finished> FinishEvent::apply(ForwardStateT&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "FinishEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
    return Finished{};
#else
    auto full_paged_tokens = state.GetFullPagedTokens(true);
    std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(state.GetDeviceNode());
    std::int32_t alloc_count =
        static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());

    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();
    auto local_allocator = std::move(state).TakeLocalKVAllocator();
    // Overlap / disagg handoff can grow the token container past the pages this
    // request actually Acquired (the terminal token's Acquire is skipped). Clamp
    // to owned pages, as scheduleRetract does.
    const std::int32_t owned_pages = local_allocator->PageCount();
    if (alloc_count > owned_pages) {
        full_paged_tokens.resize(static_cast<std::size_t>(owned_pages) + prefix_pages.size());
        alloc_count = owned_pages;
    }
    if (alloc_count > 0) {
        // Same page-aligned clamp as InsertHybridCache, keeping prefix + inserted == full.
        const std::int32_t avail = static_cast<std::int32_t>(local_allocator->PageCount());
        if (alloc_count > avail) {
            full_paged_tokens.resize(prefix_pages.size() + avail);
            alloc_count =
                static_cast<std::int32_t>(full_paged_tokens.size()) - static_cast<std::int32_t>(prefix_pages.size());
        }
        OwnedPages alloc_pages = local_allocator->TakeFirst(alloc_count);

        kv_prefix_cache_->Insert<ResourceType::Device>(full_paged_tokens, prefix_pages, std::move(alloc_pages),
                                                       page_hashes_);

        if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr &&
            (local_mamba_allocator->HasCheckpoint() || local_mamba_allocator->HasWorking())) {
            MatchResult post_match = kv_prefix_cache_->Match(full_paged_tokens);
            TreeNode* terminal = post_match.device.last_node;
            if (terminal != nullptr && !terminal->HasMamba()) {
                if (local_mamba_allocator->HasCheckpoint()) {
                    hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachCheckpoint());
                } else {
                    hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachWorking());
                }
            }
        }
    }
    // local_mamba_allocator dropped here — destructor frees remaining slots

    MatchResult match = kv_prefix_cache_->Match(full_paged_tokens);
    if (!disable_l2_cache_ && (match.device.DepthInPage() > match.host.DepthInPage())) {
        std::vector<TreeNode*> write_diff = match.NodesWithout<ResourceType::Host>();
        std::int32_t host_pages_num = 0;
        for (TreeNode* node : write_diff) {
            host_pages_num += node->Device().NumPages();
        }
        std::unique_ptr<HostNodeRef> temp_lock = std::make_unique<HostNodeRef>(match.host.last_node);
        if (!kv_prefix_cache_->EnsureCapacityByEvict<ResourceType::Host>(host_pages_num)) {
            return Finished{};
        }
        kv_prefix_cache_->AllocateResourceOfType<ResourceType::Host>(write_diff);
        std::unique_ptr<DeviceNodeRef> device_node_ref = std::make_unique<DeviceNodeRef>(match.device.last_node);
        std::unique_ptr<HostNodeRef> host_node_ref = std::make_unique<HostNodeRef>(match.device.last_node);

        auto pages_to_transfer = BuildWriteBackPairs(write_diff);
        std::vector<TreeNode*> mamba_writeback_nodes;
        std::vector<PagedCacheTransferPair> paged_cache_writeback_transfers;
        std::vector<TreeNode*> paged_cache_writeback_nodes;
        if (hybrid_prefix_cache_ != nullptr) {
            auto mamba_pairs = hybrid_prefix_cache_->PrepareMambaHostWriteBack(write_diff);
            mamba_writeback_nodes = MambaNodesForTransferPairs(write_diff, mamba_pairs);
            pages_to_transfer.insert(pages_to_transfer.end(), std::make_move_iterator(mamba_pairs.begin()),
                                     std::make_move_iterator(mamba_pairs.end()));
            paged_cache_writeback_nodes = PagedCacheSnapshotNodesForWriteBack(match.device.last_node);
            paged_cache_writeback_transfers =
                hybrid_prefix_cache_->PreparePagedCacheHostWriteBack(paged_cache_writeback_nodes);
        }
        return Draining{std::move(pages_to_transfer),
                        std::move(device_node_ref),
                        std::move(host_node_ref),
                        std::move(write_diff),
                        std::move(mamba_writeback_nodes),
                        std::move(paged_cache_writeback_transfers),
                        std::move(paged_cache_writeback_nodes)};
    }
    return Finished{};
#endif
}

std::variant<Draining, Finished> FinishEvent::operator()(Decoding&& state) {
    return apply(std::move(state));
}

std::variant<Draining, Finished> FinishEvent::operator()(PrefillDone&& state) {
    return apply(std::move(state));
}

// EOS mid-writeback: downcast so WriteBackDoneEvent takes the existing WritingBack -> Finished path.
WritingBack FinishEvent::operator()(Retracting&& state) {
    return static_cast<WritingBack&&>(state);
}

// Draining -> WritingBack: the two RAII node-ref locks alone keep the pages pinned until WriteBackDone.
WritingBack CommitDrainingEvent::operator()(Draining&& state) {
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();
    auto host_writeback_nodes = std::move(state).TakeHostWriteBackNodes();
    auto mamba_writeback_nodes = std::move(state).TakeMambaWriteBackNodes();
    auto paged_cache_writeback_nodes = std::move(state).TakePagedCacheWriteBackNodes();
    return WritingBack{std::move(device_node_ref), std::move(host_node_ref), std::move(host_writeback_nodes),
                       std::move(mamba_writeback_nodes), std::move(paged_cache_writeback_nodes)};
}

// WritingBack -> Finished: written-back cache demotes to host-only, so the next hit must load back.
Finished WriteBackDoneEvent::operator()(WritingBack&& state) {
    TreeNode* device_node = state.DeviceNode();
    if (!success_) {
        RollbackHostWriteBack(kv_prefix_cache_, hybrid_prefix_cache_, state);
        return Finished{};
    }
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->OnMambaHostWriteBackDone(state.MambaWriteBackNodes(), true);
        hybrid_prefix_cache_->OnPagedCacheHostWriteBackDone(state.PagedCacheWriteBackNodes(), true);
    }
    state.DropDeviceNodeRef();
    DemoteWrittenBackDevice(kv_prefix_cache_, hybrid_prefix_cache_, device_node);
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->DemoteIdleMambaDeviceCopiesPresentOnHost();
    }
    return Finished{};
}

std::variant<Retracted, Finished> WriteBackDoneEvent::operator()(Retracting&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    FlatRetractUnsupported();
#else
    if (!success_) {
        RollbackHostWriteBack(kv_prefix_cache_, hybrid_prefix_cache_, state);
        return Finished{};
    }
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    TreeNode* device_node = state.DeviceNode();
    if (hybrid_prefix_cache_ != nullptr) {
        hybrid_prefix_cache_->OnMambaHostWriteBackDone(state.MambaWriteBackNodes(), true);
        hybrid_prefix_cache_->OnPagedCacheHostWriteBackDone(state.PagedCacheWriteBackNodes(), true);
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
#endif
}

Finished AbortEvent::operator()(Submitted&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Prefetching&& state) {
    return Aborting{std::move(state).TakeHostPages()};
}

Finished AbortEvent::operator()(Draining&& state) {
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();
    auto host_writeback_nodes = std::move(state).TakeHostWriteBackNodes();
    auto mamba_writeback_nodes = std::move(state).TakeMambaWriteBackNodes();
    auto paged_cache_writeback_nodes = std::move(state).TakePagedCacheWriteBackNodes();
    auto writeback = WritingBack{std::move(device_node_ref), std::move(host_node_ref), std::move(host_writeback_nodes),
                                 std::move(mamba_writeback_nodes), std::move(paged_cache_writeback_nodes)};
    RollbackHostWriteBack(kv_prefix_cache_, hybrid_prefix_cache_, writeback);
    return Finished{};
}

Finished AbortEvent::operator()(PrefetchDone&&) {
    return Finished{};
}

Aborting AbortEvent::operator()(Aborting&& state) {
    return std::move(state);
}

Finished AbortEvent::operator()(Prefilling&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "AbortEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
#else
    (void)state;
#endif
    return Finished{};
}

Finished AbortEvent::operator()(PrefillDone&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "AbortEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
#else
    (void)state;
#endif
    return Finished{};
}

Finished AbortEvent::operator()(Decoding&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    _assert(coordinator_ != nullptr, "AbortEvent: flat path requires a coordinator");
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
#else
    (void)state;
#endif
    return Finished{};
}

Finished AbortEvent::operator()(Retracted&&) {
    return Finished{};
}

#if TOKENSPEED_FLAT_KVCACHE
template <typename ForwardStateT>
Submitted FlatRetractEvent::applyRetract(ForwardStateT&& state) {
    _assert(coordinator_ != nullptr, "FlatRetractEvent: flat path requires a coordinator");
    TokenContainer* token_container = state.GetTokenContainer();
    const std::int32_t page_size = state.GetPageSize();
    // Generated tokens rebase into the prefill window so the requeued prefill recomputes them.
    token_container->RebasePrefill();
    auto tables = std::move(state).TakeBlockTables();
    FreeRequest(*coordinator_, tables);
    return Submitted{token_container, page_size};
}

Submitted FlatRetractEvent::operator()(Decoding&& state) {
    return applyRetract(std::move(state));
}

Submitted FlatRetractEvent::operator()(PrefillDone&& state) {
    return applyRetract(std::move(state));
}
#endif

template <typename ForwardStateT>
Retracting ScheduleRetractEvent::applyRetract(ForwardStateT&& state) {
#if TOKENSPEED_FLAT_KVCACHE
    FlatRetractUnsupported();
#else
    std::unique_ptr<DeviceNodeRef> device_node_ref = nullptr;
    std::unique_ptr<HostNodeRef> host_node_ref = nullptr;
    std::vector<Retracting::PagePair> pages_to_transfer;
    std::vector<TreeNode*> host_writeback_nodes;
    std::vector<TreeNode*> mamba_writeback_nodes;
    std::vector<PagedCacheTransferPair> paged_cache_writeback_transfers;
    std::vector<TreeNode*> paged_cache_writeback_nodes;

    if (match_result_.device.DepthInPage() > match_result_.host.DepthInPage()) {
        std::vector<TreeNode*> write_diff = match_result_.NodesWithout<ResourceType::Host>();
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
        if (!kv_prefix_cache_->AllocateResourceOfType<ResourceType::Host>(write_diff)) {
            throw std::logic_error("ScheduleRetractEvent: failed to allocate host pages for device cache writeback");
        }
        host_writeback_nodes = write_diff;
        pages_to_transfer = BuildWriteBackPairs(write_diff);
        if (hybrid_prefix_cache_ != nullptr) {
            auto mamba_pairs = hybrid_prefix_cache_->PrepareMambaHostWriteBack(write_diff);
            mamba_writeback_nodes = MambaNodesForTransferPairs(write_diff, mamba_pairs);
            pages_to_transfer.insert(pages_to_transfer.end(), std::make_move_iterator(mamba_pairs.begin()),
                                     std::make_move_iterator(mamba_pairs.end()));
            paged_cache_writeback_nodes = PagedCacheSnapshotNodesForWriteBack(match_result_.device.last_node);
            paged_cache_writeback_transfers =
                hybrid_prefix_cache_->PreparePagedCacheHostWriteBack(paged_cache_writeback_nodes);
        }
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    } else {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.device.last_node);
    }

    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_allocator = std::move(state).TakeLocalKVAllocator();
    auto local_mamba_allocator = std::move(state).TakeLocalMambaAllocator();

    // Save Mamba state into the prefix cache before retract so loadback can recover it.
    if (hybrid_prefix_cache_ != nullptr && local_mamba_allocator != nullptr &&
        (local_mamba_allocator->HasCheckpoint() || local_mamba_allocator->HasWorking())) {
        TreeNode* terminal = match_result_.device.last_node;
        if (terminal != nullptr && !terminal->HasMamba()) {
            if (local_mamba_allocator->HasCheckpoint()) {
                hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachCheckpoint());
            } else {
                hybrid_prefix_cache_->InsertMamba(terminal, local_mamba_allocator->DetachWorking());
            }
        }
        // Once retracted, the recoverable Mamba state is tree-owned and evictable: keep no request-local slots.
        local_mamba_allocator.reset();
    }

    return Retracting{token_container,
                      page_size,
                      std::move(host_node_ref),
                      std::move(device_node_ref),
                      std::move(local_allocator),
                      std::move(pages_to_transfer),
                      std::move(mamba_writeback_nodes),
                      std::move(local_mamba_allocator),
                      std::move(paged_cache_writeback_transfers),
                      std::move(paged_cache_writeback_nodes),
                      std::move(host_writeback_nodes)};
#endif
}

Retracting ScheduleRetractEvent::operator()(Decoding&& state) {
    return applyRetract(std::move(state));
}

Retracting ScheduleRetractEvent::operator()(PrefillDone&& state) {
    return applyRetract(std::move(state));
}

}  // namespace tokenspeed::fsm
