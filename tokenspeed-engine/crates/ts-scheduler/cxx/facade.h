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

// The control thread's view of the C++ scheduler: plain-value requests and
// events in, plain-value copies of what the control plane reads out.

#pragma once

#include <memory>

#include "cache/tier/shared_l2_pool.h"
#include "rust/cxx.h"
#include "scheduler/execution_plan.h"
#include "scheduler/scheduler.h"

namespace ts_scheduler {

struct RequestSpec;
struct CacheEventObserver;
struct ForwardEvent;
struct CacheEvent;
struct CacheOpView;
struct StorageOpView;
struct ForwardBatchView;
struct TransferBatchView;
struct SchedulerAbortView;
struct SchedulerSizes;
struct AdmissionSnapshot;
struct PrefillGateProbe;
struct SharedPrefixPublication;
struct CacheGroupPages;
struct KvEventView;
struct StorageTickView;
struct StorageCancellationView;
struct StorageDoneView;
struct StorageStatsView;
struct DiskOpView;
struct DiskDoneView;
struct DiskStatsView;
struct PrefixHitView;
struct StorageKeyView;
struct SharedL2LookupView;
struct SharedL2AnswerView;
struct SharedL2ProxyBlockView;
struct SharedL2CheckpointView;
struct SharedL2OpView;
struct SharedL2ReservationView;
struct SharedL2PoolStatsView;
struct PeerEventView;
struct PeerAnswerView;
struct PeerLookupView;
struct PeerFetchView;
struct PeerLocalChoice;
struct PinSampleView;
struct PeerStatsView;
struct PeerLeaseView;
struct PeerPageView;

std::uint32_t cache_event_protocol_version();
void install_cache_event_observer(tokenspeed::Scheduler& scheduler, rust::Box<CacheEventObserver> observer);

void submit_requests(tokenspeed::Scheduler& scheduler, rust::Slice<const RequestSpec> specs);
std::unique_ptr<tokenspeed::ExecutionPlan> next_execution_plan(tokenspeed::Scheduler& scheduler,
                                                               bool allow_prefill, bool apply_prefill_watermark);
void advance(tokenspeed::Scheduler& scheduler, rust::Slice<const ForwardEvent> events);
void advance_head(tokenspeed::Scheduler& scheduler, rust::Slice<const StorageTickView> ticks,
                  rust::Slice<const StorageCancellationView> cancellations,
                  rust::Slice<const StorageDoneView> done, rust::Slice<const DiskDoneView> disk_done,
                  rust::Slice<const CacheEvent> cache_events);
void advance_peer(tokenspeed::Scheduler& scheduler, std::int64_t now_us, rust::Slice<const PeerEventView> events);
bool release_peer_pin(const std::shared_ptr<tokenspeed::PeerPinRelease>& handle);
std::shared_ptr<tokenspeed::PeerPinRelease> peer_release_handle(tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease);
bool enable_async_host_peer(tokenspeed::Scheduler& scheduler);
bool enable_async_device_peer(tokenspeed::Scheduler& scheduler, bool adaptive);
rust::Vec<PeerFetchView> reserve_peer_host(tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease, std::int32_t tokens, bool& retry);
bool apply_peer_export(tokenspeed::Scheduler& scheduler, const PeerEventView& event, std::int64_t now_us);
bool pin_peer_offer(tokenspeed::Scheduler& scheduler, const PeerEventView& event, std::int64_t now_us);
bool close_peer_offer(tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease, std::int64_t now_us);
rust::Vec<PeerLookupView> take_peer_lookups(tokenspeed::Scheduler& scheduler);
PeerAnswerView answer_peer_lookup(const tokenspeed::Scheduler& scheduler, const PeerEventView& lookup,
                                  std::size_t pending_new_leases);
bool peer_pages_match(const tokenspeed::Scheduler& scheduler, const PeerEventView& lookup, std::int32_t num_tokens,
                      rust::Slice<const PeerPageView> pages);
bool peer_lease_can_pin(const tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease);
bool peer_lease_held(const tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease);
std::int64_t plan_calls(const tokenspeed::Scheduler& scheduler);
rust::Vec<PinSampleView> take_peer_pin_samples(tokenspeed::Scheduler& scheduler);
PeerStatsView peer_stats(const tokenspeed::Scheduler& scheduler);
std::size_t demote_sessions(tokenspeed::Scheduler& scheduler, rust::Slice<const rust::String> ids, bool ended);
std::size_t demote_session(tokenspeed::Scheduler& scheduler, rust::Str session_id, bool ended);
std::size_t demote_cache_groups(tokenspeed::Scheduler& scheduler, rust::Slice<const rust::String> ids, bool ended);
AdmissionSnapshot admission_snapshot(const tokenspeed::Scheduler& scheduler, std::int32_t token_step);
bool shared_prefix_storage_enabled(const tokenspeed::Scheduler& scheduler);
SharedPrefixPublication shared_prefix_publication(const tokenspeed::Scheduler& scheduler,
                                                  rust::Slice<const std::int32_t> input_ids,
                                                  std::int32_t max_prefix_tokens);
SchedulerSizes sizes(const tokenspeed::Scheduler& scheduler);
std::int32_t active_lcm_blocks(const tokenspeed::Scheduler& scheduler);
PrefillGateProbe probe_prefill_gate(const tokenspeed::Scheduler& scheduler);
CacheGroupPages cache_group_pages(const tokenspeed::Scheduler& scheduler, rust::Str group_id);
std::unique_ptr<tokenspeed::KvEventBatch> drain_kv_events(tokenspeed::Scheduler& scheduler);
std::unique_ptr<tokenspeed::KvEventBatch> make_kv_event_batch(rust::Slice<const KvEventView> events);
std::size_t kv_event_batch_size(const tokenspeed::KvEventBatch& batch);
std::size_t kv_event_batch_bytes(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::uint8_t> kv_event_stored(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::uint8_t> kv_event_has_parent(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::uint64_t> kv_event_parents(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::int32_t> kv_event_block_sizes(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::size_t> kv_event_hash_offsets(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::size_t> kv_event_token_offsets(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::uint64_t> kv_event_hashes(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::int32_t> kv_event_tokens(const tokenspeed::KvEventBatch& batch);
rust::Slice<const std::uint8_t> kv_event_media(const tokenspeed::KvEventBatch& batch);
std::int32_t request_token_size(const tokenspeed::Scheduler& scheduler, rust::Str request_id);
bool can_abort_before_admission(const tokenspeed::Scheduler& scheduler, rust::Str request_id);
bool defers_decode_state(const tokenspeed::Scheduler& scheduler);
bool storage_pending(const tokenspeed::Scheduler& scheduler);
bool storage_current_pending(const tokenspeed::Scheduler& scheduler);
bool storage_progress_pending(const tokenspeed::Scheduler& scheduler);
std::size_t storage_blocking_quarantined(const tokenspeed::Scheduler& scheduler);
bool storage_non_storage_clearable(const tokenspeed::Scheduler& scheduler);
bool can_prepare_storage_clear(const tokenspeed::Scheduler& scheduler);
StorageStatsView storage_stats(const tokenspeed::Scheduler& scheduler);
DiskStatsView disk_stats(const tokenspeed::Scheduler& scheduler);
PrefixHitView prefix_hit_stats(const tokenspeed::Scheduler& scheduler);
bool clear_cache(tokenspeed::Scheduler& scheduler);
void cancel_storage_backups(tokenspeed::Scheduler& scheduler);
void cancel_storage_for_clear(tokenspeed::Scheduler& scheduler);
bool can_clear_cache_retaining_storage(const tokenspeed::Scheduler& scheduler);
bool clear_cache_retaining_storage(tokenspeed::Scheduler& scheduler);
bool storage_has_retained_operations(const tokenspeed::Scheduler& scheduler);
bool clear_l1_cache(tokenspeed::Scheduler& scheduler);
bool can_clear_cache(const tokenspeed::Scheduler& scheduler);
std::int32_t max_single_request_tokens(const tokenspeed::Scheduler& scheduler);
rust::Vec<SharedL2LookupView> start_shared_l2_lookups(tokenspeed::Scheduler& scheduler);
rust::Vec<SharedL2ProxyBlockView> apply_shared_l2_answers(tokenspeed::Scheduler& scheduler, rust::Slice<const SharedL2AnswerView> answers);

rust::Vec<ForwardBatchView> plan_forward_batches(const tokenspeed::ExecutionPlan& plan);
rust::Vec<rust::String> plan_completed_prompts(const tokenspeed::ExecutionPlan& plan);
bool plan_needs_device(const tokenspeed::ExecutionPlan& plan);
bool plan_has_pages_to_copy(const tokenspeed::ExecutionPlan& plan);
rust::Vec<ForwardBatchView> plan_remote_decode(const tokenspeed::ExecutionPlan& plan);
rust::Vec<ForwardBatchView> plan_remote_prefill(const tokenspeed::ExecutionPlan& plan);
rust::Vec<TransferBatchView> plan_forward_transfer(const tokenspeed::ExecutionPlan& plan);
rust::Vec<TransferBatchView> plan_remote_decode_transfer(const tokenspeed::ExecutionPlan& plan);
rust::Vec<TransferBatchView> plan_remote_prefill_transfer(const tokenspeed::ExecutionPlan& plan);
rust::Vec<SchedulerAbortView> plan_aborts(const tokenspeed::ExecutionPlan& plan);
rust::Vec<CacheOpView> plan_cache_ops(const tokenspeed::ExecutionPlan& plan);
rust::Vec<StorageOpView> plan_storage_ops(const tokenspeed::ExecutionPlan& plan);
rust::Vec<DiskOpView> plan_disk_ops(const tokenspeed::ExecutionPlan& plan);
rust::Vec<SharedL2OpView> plan_shared_l2_ops(const tokenspeed::ExecutionPlan& plan);
rust::Vec<PeerLocalChoice> plan_peer_local_choices(const tokenspeed::ExecutionPlan& plan);
rust::Vec<PeerFetchView> plan_peer_fetches(const tokenspeed::ExecutionPlan& plan);

SharedL2AnswerView shared_l2_pool_lookup(tokenspeed::SharedL2Pool& pool, std::uint64_t client,
                                         const SharedL2LookupView& lookup);
void shared_l2_pool_release(tokenspeed::SharedL2Pool& pool, std::uint64_t client, std::uint64_t ticket);
SharedL2ReservationView shared_l2_pool_reserve(tokenspeed::SharedL2Pool& pool, std::uint64_t client,
                                               rust::Slice<const StorageKeyView> keys,
                                               rust::Slice<const std::int32_t> buckets);
void shared_l2_pool_commit(tokenspeed::SharedL2Pool& pool, std::uint64_t client, std::uint64_t ticket,
                           rust::Slice<const SharedL2CheckpointView> checkpoints);
void shared_l2_pool_touch(tokenspeed::SharedL2Pool& pool, rust::Slice<const StorageKeyView> keys);
void shared_l2_pool_disconnect(tokenspeed::SharedL2Pool& pool, std::uint64_t client);
bool shared_l2_pool_clear_cache(tokenspeed::SharedL2Pool& pool);
SharedL2PoolStatsView shared_l2_pool_stats(const tokenspeed::SharedL2Pool& pool);

void set_profile_span_sink(bool installed);

}  // namespace ts_scheduler
