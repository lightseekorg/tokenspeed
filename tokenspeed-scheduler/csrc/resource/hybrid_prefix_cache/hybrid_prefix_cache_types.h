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

#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/types.h"
#include "scheduler/operations/cache.h"

namespace tokenspeed {

class ForwardOperationBase;
class LocalKVAllocator;
class LocalMambaAllocator;
class TreeNode;

struct RecoveryPlan {
    bool recovery_state_available{true};
    TreeNode* protected_recovery_node{nullptr};
    MatchResult compat_match{};
};

enum class AdmissionRequestKind {
    kPrefillFirstChunk,
    kPrefillChunk,
    kDecodeChunk,
    kDecodeFromRetracted,
    kRetract,
};

struct AdmissionRequest {
    std::string request_id{};
    AdmissionRequestKind kind{AdmissionRequestKind::kDecodeChunk};
    std::int32_t device_pages_needed{0};
    std::int32_t host_pages_needed{0};
    std::int32_t tokens_this_round{0};
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};
    const RecoveryPlan* recovery_plan{nullptr};
    const MatchResult* compat_match{nullptr};
    TreeNode* protected_recovery_node{nullptr};
    bool refresh_mamba_checkpoint{false};
};

struct AdmissionVerdict {
    bool admitted{false};
    std::optional<std::int32_t> mamba_branching_seqlen{};
    std::optional<std::int32_t> mamba_cow_src_index{};
    std::vector<TransferPair> cache_transfer_pairs{};
};

struct PrefixMaterializationRequest {
    const MatchResult* compat_match{nullptr};
    bool require_all_pages{false};
};

struct RequestLocalKVStateRequest {
    bool create_allocator{false};
    LocalKVAllocator* allocator{nullptr};
    std::int32_t initial_tokens{0};
    std::int32_t acquire_tokens{0};
};

struct RequestLocalMambaStateRequest {
    bool create_allocator{false};
    bool require_allocator{false};
    LocalMambaAllocator* refresh_checkpoint_allocator{nullptr};
    std::optional<std::int32_t> checkpoint_raw_position{};
};

struct DevicePrefixPublicationRequest {
    const std::vector<std::span<const std::int32_t>>* full_paged_tokens{nullptr};
    std::unique_ptr<DeviceNodeRef>* device_node_ref{nullptr};
    LocalKVAllocator* local_kv_allocator{nullptr};
    LocalMambaAllocator* local_mamba_allocator{nullptr};
    std::optional<std::int32_t> chunk_begin{};
};

struct FinishedRequestPublicationRequest {
    const std::vector<std::span<const std::int32_t>>* full_paged_tokens{nullptr};
    const TreeNode* current_device_node{nullptr};
    LocalKVAllocator* local_kv_allocator{nullptr};
    LocalMambaAllocator* local_mamba_allocator{nullptr};
    const std::vector<std::string>* page_hashes{nullptr};
};

struct DevicePrefixInsertionPlanRequest {
    const std::vector<std::span<const std::int32_t>>* full_paged_tokens{nullptr};
    const TreeNode* current_device_node{nullptr};
};

struct DevicePrefixInsertionRequest {
    const std::vector<std::span<const std::int32_t>>* full_paged_tokens{nullptr};
    const TreeNode* current_device_node{nullptr};
    OwnedPages pages_to_insert{};
};

struct HostWritebackMaterializationRequest {
    const std::vector<TreeNode*>* write_diff{nullptr};
    bool ensure_capacity_before_allocate{false};
};

struct TreeOwnedRequestStatePublicationRequest {
    TreeNode* terminal{nullptr};
    std::unique_ptr<LocalMambaAllocator>* local_mamba_allocator_owner{nullptr};
};

enum class WorkerCompatibilityCommitKind {
    kPrefillFirstChunk,
    kPrefillChunk,
    kDecodeChunk,
    kDecodeFromRetracted,
};

struct WorkerCompatibilityCommitRequest {
    ForwardOperationBase* op_base{nullptr};
    WorkerCompatibilityCommitKind kind{WorkerCompatibilityCommitKind::kDecodeChunk};
    TreeNode* terminal{nullptr};
    const MatchResult* compat_match{nullptr};
    const LocalMambaAllocator* local_mamba_allocator_view{nullptr};
    MatchResult::PagedCache paged_cache_hit{};
    std::int32_t first_raw_position_of_op{0};
    std::int32_t target_raw_tokens_exclusive{0};
    bool commit_tree_prefix_before_acquire{false};
};

struct StepCommitRequest {
    std::optional<PrefixMaterializationRequest> materialize_prefix{};
    std::optional<DevicePrefixPublicationRequest> publish_device_prefix{};
    std::optional<FinishedRequestPublicationRequest> publish_finished_request{};
    std::optional<DevicePrefixInsertionPlanRequest> plan_device_prefix_insertion{};
    std::optional<DevicePrefixInsertionRequest> publish_device_prefix_insertion{};
    std::optional<HostWritebackMaterializationRequest> materialize_host_writeback{};
    std::optional<TreeOwnedRequestStatePublicationRequest> publish_tree_owned_request_state{};
    std::optional<RequestLocalKVStateRequest> request_local_kv{};
    std::optional<RequestLocalMambaStateRequest> request_local_mamba{};
    std::optional<WorkerCompatibilityCommitRequest> worker_metadata{};
};

struct StepCommitResult {
    bool ok{true};
    MatchResult match_result{};
    std::int32_t device_insert_page_count{0};
    std::unique_ptr<LocalKVAllocator> local_kv_allocator{};
    std::unique_ptr<LocalMambaAllocator> local_mamba_allocator{};
    std::vector<TransferPair> cache_transfer_pairs{};
    std::vector<TreeNode*> mamba_writeback_nodes{};
};

struct CacheDeviceMemoryDiagnosticsSnapshot {
    std::unordered_map<std::int32_t, int> tree_device_pages{};
    std::int32_t free_device_pages{0};
    // Usable device pages; page id 0 remains reserved/invalid.
    std::int32_t total_device_pages{0};
};

struct StatsRequest {
    std::optional<std::string> request_id{};
    std::vector<std::string> paged_cache_group_ids{};
    bool include_device_memory_diagnostics{false};
};

struct CacheStatsSnapshot {
    std::size_t available_device_pages{0};
    std::vector<std::string> paged_cache_group_ids{};
    std::map<std::string, std::int32_t> paged_cache_total_pages{};
    std::map<std::string, std::int32_t> paged_cache_available_pages{};
    std::map<std::string, std::int64_t> paged_cache_failed_alloc_count{};
    std::map<std::string, std::vector<std::int32_t>> request_paged_cache_page_ids{};
    std::map<std::string, std::int32_t> request_paged_cache_base_logical_page{};
    std::optional<CacheDeviceMemoryDiagnosticsSnapshot> device_memory_diagnostics{};
};

}  // namespace tokenspeed
