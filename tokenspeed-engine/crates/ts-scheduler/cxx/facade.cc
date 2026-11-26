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

#include "ts-scheduler/cxx/facade.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "core/profile_span.h"
#include "scheduler/execution_event.h"
#include "ts-scheduler/src/lib.rs.h"

namespace ts_scheduler {

namespace {

template <typename T>
rust::Vec<T> to_rust(const std::vector<T>& values) {
    rust::Vec<T> out;
    out.reserve(values.size());
    for (const T& value : values) {
        out.push_back(value);
    }
    return out;
}

std::vector<std::int32_t> to_std(rust::Slice<const std::int32_t> values) {
    return {values.begin(), values.end()};
}

void record_profile_span(const char* name, std::int64_t begin_ns, std::int64_t end_ns) {
    record_scheduler_span(rust::Str(name), begin_ns, end_ns);
}

}  // namespace

namespace {

class RustCacheEventSink final : public tokenspeed::CacheEventSink {
public:
    explicit RustCacheEventSink(rust::Box<CacheEventObserver> observer) : observer_(std::move(observer)) {}

    void OnSlotAssigned(const tokenspeed::SlotOwnershipEvent& event) noexcept override {
        RecordSlot(event, CacheObservationKind::SlotAssigned);
    }

    void OnSlotReleased(const tokenspeed::SlotOwnershipEvent& event) noexcept override {
        RecordSlot(event, CacheObservationKind::SlotReleased);
    }

    void OnPrefixLoadScheduled(const tokenspeed::LoadBackOperation& operation) noexcept override {
        const auto& metadata = operation.telemetry;
        CacheObservation value{};
        value.kind = CacheObservationKind::PrefixLoadScheduled;
        value.slot = metadata.request_pool_index;
        value.slot_generation = metadata.slot_generation;
        value.admission_id = metadata.admission_id;
        value.op_id = operation.op_id;
        value.op_generation = metadata.op_generation;
        Record(value, operation.request_id);
    }

    void OnRestoreScheduled(const tokenspeed::SnapshotRestoreOperation& operation) noexcept override {
        const auto& metadata = operation.telemetry;
        CacheObservation value{};
        value.kind = CacheObservationKind::RestoreScheduled;
        value.slot = operation.request_pool_index;
        value.slot_generation = metadata.slot_generation;
        value.admission_id = metadata.admission_id;
        value.slot_previous_admission_id = metadata.slot_previous_admission_id;
        value.op_id = operation.op_id;
        value.op_generation = metadata.op_generation;
        value.previous_slot = metadata.previous_slot;
        value.previous_slot_generation = metadata.previous_slot_generation;
        value.restore_sequence = metadata.restore_sequence;
        value.committed_tokens = metadata.committed_tokens;
        value.computed_tokens = metadata.computed_tokens;
        value.prompt_tokens = metadata.prompt_tokens;
        Record(value, operation.request_id);
    }

private:
    void RecordSlot(const tokenspeed::SlotOwnershipEvent& event, CacheObservationKind kind) noexcept {
        CacheObservation value{};
        value.kind = kind;
        value.slot = event.slot;
        value.slot_generation = event.generation;
        value.admission_id = event.admission_id;
        value.slot_previous_admission_id = event.previous_admission_id;
        Record(value, event.request_id);
    }

    void Record(CacheObservation value, std::string_view request_id) noexcept {
        // Borrow bytes rather than manufacturing a Rust string: the receiver
        // validates the bounded identity before interpreting it as UTF-8.
        record_cache_observation(*observer_, value,
            rust::Slice<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(request_id.data()), request_id.size()));
    }

    rust::Box<CacheEventObserver> observer_;
};

}  // namespace

std::uint32_t cache_event_protocol_version() {
    return tokenspeed::kCacheEventProtocolVersion;
}

void install_cache_event_observer(tokenspeed::Scheduler& scheduler, rust::Box<CacheEventObserver> observer) {
    scheduler.SetCacheEventSink(std::make_shared<RustCacheEventSink>(std::move(observer)));
}

void submit_requests(tokenspeed::Scheduler& scheduler, rust::Slice<const RequestSpec> specs) {
    std::vector<tokenspeed::RequestSpec> converted;
    {
        // Each prompt is copied across the bridge.
        tokenspeed::ProfileSpan profile_span{"copyRequestSpecs"};
        converted.reserve(specs.size());
        for (const RequestSpec& spec : specs) {
            converted.push_back(tokenspeed::RequestSpec{
                .request_id = std::string(spec.request_id),
                .admission_id = spec.admission_id,
                .session_id = std::string(spec.session_id),
                .cache_group_id = std::string(spec.cache_group_id),
                .shared_prefix_tokens = to_std({spec.shared_prefix_tokens.data(), spec.shared_prefix_tokens.size()}),
                .peer_holder = std::string(spec.peer_holder),
                .tokens = to_std({spec.tokens.data(), spec.tokens.size()}),
                .max_new_tokens = spec.max_new_tokens,
                .max_cached_prefix_tokens = spec.max_cached_prefix_tokens,
            });
        }
    }
    scheduler.SubmitRequests(converted);
}

namespace {

tokenspeed::PeerLeaseId FromView(const PeerLeaseView& lease) {
    return tokenspeed::PeerLeaseId{.reader = std::string(lease.reader), .id = lease.id};
}

PeerLeaseView ToView(const tokenspeed::PeerLeaseId& lease) {
    return PeerLeaseView{.reader = rust::String(lease.reader), .id = lease.id};
}

PeerTier ToView(tokenspeed::CacheTier tier) {
    switch (tier) {
        case tokenspeed::CacheTier::kDevice:
            return PeerTier::Device;
        case tokenspeed::CacheTier::kHost:
            return PeerTier::Host;
        case tokenspeed::CacheTier::kDisk:
            return PeerTier::Disk;
    }
    throw std::invalid_argument("an unknown cache tier");
}

tokenspeed::CacheTier FromView(PeerTier tier) {
    switch (tier) {
        case PeerTier::Device:
            return tokenspeed::CacheTier::kDevice;
        case PeerTier::Host:
            return tokenspeed::CacheTier::kHost;
        case PeerTier::Disk:
            return tokenspeed::CacheTier::kDisk;
    }
    throw std::invalid_argument("an unknown peer page tier");
}

rust::Vec<PeerPageView> ToView(const std::vector<tokenspeed::PeerPage>& pages) {
    rust::Vec<PeerPageView> out;
    out.reserve(pages.size());
    for (const tokenspeed::PeerPage& page : pages) {
        out.push_back(PeerPageView{
            .group_id = page.group_id, .slot = page.slot, .tier = ToView(page.tier), .block_id = page.block_id});
    }
    return out;
}

template <typename T>
rust::Vec<T> ToRustVec(const std::vector<T>& values) {
    rust::Vec<T> out;
    out.reserve(values.size());
    for (const T& value : values) {
        out.push_back(value);
    }
    return out;
}

std::vector<tokenspeed::PeerPage> FromView(rust::Slice<const PeerPageView> pages) {
    std::vector<tokenspeed::PeerPage> out;
    out.reserve(pages.size());
    for (const PeerPageView& page : pages) {
        out.push_back(tokenspeed::PeerPage{
            .group_id = page.group_id, .slot = page.slot, .tier = FromView(page.tier), .block_id = page.block_id});
    }
    return out;
}

tokenspeed::cache::PeerLookup LookupFromView(const PeerEventView& event) {
    std::vector<std::string> hashes;
    hashes.reserve(event.prefix_hashes.size());
    for (const rust::String& hash : event.prefix_hashes) {
        hashes.emplace_back(hash);
    }
    return tokenspeed::cache::PeerLookup{
        .lease = FromView(event.lease),
        .prefix_hashes = std::move(hashes),
        .tokens = std::vector<std::int32_t>(event.tokens.begin(), event.tokens.end()),
        .max_tokens = event.max_tokens,
    };
}

PeerAnswerView ToView(const tokenspeed::PeerAnswer& answer) {
    const PeerAnswerKind kind = answer.kind == tokenspeed::PeerAnswerKind::kLookup ? PeerAnswerKind::Lookup
                                : answer.kind == tokenspeed::PeerAnswerKind::kPin  ? PeerAnswerKind::Pin
                                                                                    : PeerAnswerKind::Release;
    return PeerAnswerView{.lease = ToView(answer.lease),
                          .kind = kind,
                          .num_tokens = answer.num_tokens,
                          .pages = ToView(answer.pages),
                          .held = answer.held};
}

rust::Vec<rust::String> ToRustStrings(const std::vector<std::string>& values) {
    rust::Vec<rust::String> out;
    out.reserve(values.size());
    for (const std::string& value : values) {
        out.push_back(rust::String(value));
    }
    return out;
}

}  // namespace

void advance_peer(tokenspeed::Scheduler& scheduler, std::int64_t now_us, rust::Slice<const PeerEventView> events) {
    // Advance peer_now_us_ first (without expiring) so holder events answered
    // against the pre-tick state (Lookup, Pin, Release) stamp against now_us
    // and promote/release before PeerExports::Expire sweeps stale leases.
    // Expire then runs before reader events (LookupAnswered, FetchDone) so
    // coordinated lookup waits past their deadline still time out before stale
    // answers are delivered, while FetchDone stamps against now_us.
    tokenspeed::ExecutionEvent execution_event;
    execution_event.With(tokenspeed::cache::PeerTick{.now_us = now_us, .expire = false});
    for (const PeerEventView& event : events) {
        switch (event.kind) {
            case PeerEventKind::Lookup:
                execution_event.With(LookupFromView(event));
                break;
            case PeerEventKind::Pin:
                execution_event.With(tokenspeed::cache::PeerPin{
                    .lease = FromView(event.lease),
                    .num_tokens = event.num_tokens,
                    .pages = FromView(rust::Slice<const PeerPageView>(event.pages.data(), event.pages.size()))});
                break;
            case PeerEventKind::Release:
                execution_event.With(
                    tokenspeed::cache::PeerRelease{.lease = FromView(event.lease), .demote = event.demote});
                break;
            case PeerEventKind::LookupAnswered:
            case PeerEventKind::FetchDone:
                break;
        }
    }
    execution_event.With(tokenspeed::cache::PeerTick{.now_us = now_us, .expire = true});
    for (const PeerEventView& event : events) {
        switch (event.kind) {
            case PeerEventKind::Lookup:
            case PeerEventKind::Pin:
            case PeerEventKind::Release:
                break;
            case PeerEventKind::LookupAnswered: {
                unsigned tiers = 0;
                for (const auto& page : event.pages) {
                    tiers |= 1u << static_cast<unsigned>(FromView(page.tier));
                }
                execution_event.With(tokenspeed::cache::PeerLookupAnswered{
                    .lease = FromView(event.lease),
                    .num_tokens = event.num_tokens,
                    .source_mask = tiers,
                    .checkpoint = {event.checkpoint.begin(), event.checkpoint.end()}});
                break;
            }
            case PeerEventKind::TargetRetired:
                scheduler.RetirePeerTarget(FromView(event.lease));
                break;
            case PeerEventKind::FetchFrozen:
            case PeerEventKind::FetchDone:
                execution_event.With(tokenspeed::cache::PeerFetchDone{.request_id = std::string(event.request_id),
                                                                      .num_tokens = event.num_tokens,
                                                                      .dispatch_us = event.dispatch_us,
                                                                      .writable_us = event.writable_us,
                                                                      .terminal_us = event.terminal_us,
                                                                      .lease = FromView(event.lease),
                                                                      .frozen = event.kind == PeerEventKind::FetchFrozen});
                break;
        }
    }
    scheduler.Advance(execution_event);
}

PeerAnswerView answer_peer_lookup(const tokenspeed::Scheduler& scheduler, const PeerEventView& lookup,
                                  std::size_t pending_new_leases) {
    return ToView(scheduler.AnswerPeerLookup(LookupFromView(lookup), pending_new_leases));
}

bool peer_pages_match(const tokenspeed::Scheduler& scheduler, const PeerEventView& lookup, std::int32_t num_tokens,
                      rust::Slice<const PeerPageView> pages) {
    return scheduler.PeerPagesMatch(LookupFromView(lookup), num_tokens, FromView(pages));
}

bool peer_lease_can_pin(const tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease) {
    return scheduler.PeerLeaseCanPin(FromView(lease));
}

bool peer_lease_held(const tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease) {
    return scheduler.PeerLeaseHeld(FromView(lease));
}

std::int64_t plan_calls(const tokenspeed::Scheduler& scheduler) {
    return scheduler.PlanCalls();
}

rust::Vec<PeerLookupView> take_peer_lookups(tokenspeed::Scheduler& scheduler) {
    rust::Vec<PeerLookupView> out;
    for (const tokenspeed::PeerLookupRequest& lookup : scheduler.TakePeerLookups()) {
        out.push_back(PeerLookupView{.request_id = rust::String(lookup.request_id),
                                     .deadline_us = lookup.deadline_us,
                                     .lease = ToView(lookup.lease),
                                     .holder = rust::String(lookup.holder),
                                     .prefix_hashes = ToRustStrings(lookup.prefix_hashes),
                                     .tokens = ToRustVec(lookup.tokens),
                                     .max_tokens = lookup.max_tokens});
    }
    return out;
}

rust::Vec<PinSampleView> take_peer_pin_samples(tokenspeed::Scheduler& scheduler) {
    rust::Vec<PinSampleView> out;
    for (const auto& sample : scheduler.TakePeerPinSamples()) {
        out.push_back(PinSampleView{.held_us = sample.held_us, .remaining_us = sample.remaining_us, .expired = sample.expired});
    }
    return out;
}

PeerStatsView peer_stats(const tokenspeed::Scheduler& scheduler) {
    const tokenspeed::Scheduler::PeerStats stats = scheduler.GetPeerStats();
    return PeerStatsView{.leases = stats.leases,
                         .pinned_blocks = stats.pinned_blocks,
                         .looking_requests = stats.looking_requests,
                         .landing_requests = stats.landing_requests,
                         .landed_tokens = stats.landed_tokens,
                         .failed_landings = stats.failed_landings,
                         .unused_host_landings = stats.unused_host_landings};
}

std::unique_ptr<tokenspeed::ExecutionPlan> next_execution_plan(tokenspeed::Scheduler& scheduler, bool allow_prefill,
                                                            bool apply_prefill_watermark) {
    return std::make_unique<tokenspeed::ExecutionPlan>(
        scheduler.NextExecutionPlan(tokenspeed::Scheduler::PlanOptions{
            .allow_prefill = allow_prefill, .apply_prefill_watermark = apply_prefill_watermark}));
}

void advance(tokenspeed::Scheduler& scheduler, rust::Slice<const ForwardEvent> events) {
    tokenspeed::ExecutionEvent execution_event;
    {
        tokenspeed::ProfileSpan conversion{"SchedulerBridge::convertForwardEvents"};
        for (const ForwardEvent& event : events) {
            std::string request_id(event.request_id);
            switch (event.kind) {
                case ForwardEventKind::ExtendResult:
                    execution_event.With(tokenspeed::forward::ExtendResult{
                        .request_id = std::move(request_id),
                        .tokens = to_std({event.tokens.data(), event.tokens.size()}),
                        .spec_candidate_ids =
                            to_std({event.spec_candidate_ids.data(), event.spec_candidate_ids.size()}),
                    });
                    break;
                case ForwardEventKind::Finish:
                    execution_event.With(tokenspeed::forward::Finish{.request_id = std::move(request_id)});
                    break;
                case ForwardEventKind::Abort:
                    execution_event.With(tokenspeed::forward::Abort{.request_id = std::move(request_id)});
                    break;
                case ForwardEventKind::UpdateReserveNumTokens:
                    execution_event.With(tokenspeed::forward::UpdateReserveNumTokens{
                        .request_id = std::move(request_id),
                        .reserve_num_tokens_in_next_schedule_event = event.reserve_num_tokens,
                    });
                    break;
                case ForwardEventKind::PdBootstrapped:
                    execution_event.With(tokenspeed::pd::BootstrappedEvent(std::move(request_id)));
                    break;
                case ForwardEventKind::PdDestinationReady:
                    execution_event.With(tokenspeed::pd::DestinationReadyEvent(std::move(request_id)));
                    break;
                case ForwardEventKind::PdFailed:
                    execution_event.With(tokenspeed::pd::FailedEvent(std::move(request_id)));
                    break;
                case ForwardEventKind::PdTransferStopped:
                    execution_event.With(tokenspeed::pd::TransferStoppedEvent(std::move(request_id)));
                    break;
                case ForwardEventKind::PdSucceeded:
                    execution_event.With(tokenspeed::pd::SucceededEvent(std::move(request_id)));
                    break;
                case ForwardEventKind::PdRemotePrefillDone:
                    execution_event.With(
                        tokenspeed::pd::RemotePrefillDoneEvent(std::move(request_id), event.bootstrap_token));
                    break;
                default:
                    throw std::invalid_argument("unknown forward event kind");
            }
        }
    }
    scheduler.Advance(execution_event);
    {
        tokenspeed::ProfileSpan destruction{"SchedulerBridge::destroyForwardEvents"};
        execution_event = {};
    }
}

void advance_head(tokenspeed::Scheduler& scheduler, rust::Slice<const StorageTickView> ticks,
                  rust::Slice<const StorageCancellationView> cancellations,
                  rust::Slice<const StorageDoneView> done, rust::Slice<const DiskDoneView> disk_done,
                  rust::Slice<const CacheEvent> cache_events) {
    // One batch, in the order the Python hooks built it: the storage clock,
    // the cancellations, the storage and Disk completions, then the Host
    // copies' ACKs.
    tokenspeed::ExecutionEvent execution_event;
    for (const StorageTickView& tick : ticks) {
        execution_event.With(tokenspeed::cache::StorageTick{
            .now_us = tick.now_us,
            .allow_forward = tick.allow_forward,
            .allow_storage = tick.allow_storage,
            .pressure =
                tokenspeed::StoragePressure{
                    .queued_operations = tick.queued_operations,
                    .queued_bytes = tick.queued_bytes,
                    .oldest_operation_us = tick.oldest_operation_us,
                },
        });
    }
    for (const StorageCancellationView& cancellation : cancellations) {
        std::string request_id(cancellation.request_id);
        if (cancellation.abort) {
            execution_event.With(tokenspeed::forward::Abort{.request_id = std::move(request_id)});
        } else {
            execution_event.With(tokenspeed::cache::StorageDetach{.request_id = std::move(request_id)});
        }
    }
    for (const StorageDoneView& completion : done) {
        execution_event.With(tokenspeed::cache::StorageDone{
            .op_id = completion.op_id,
            .success = std::vector<std::uint8_t>(completion.success.begin(), completion.success.end()),
            .retired = completion.retired,
            .object_failures = std::vector<std::uint8_t>(completion.object_failures.begin(),
                                                         completion.object_failures.end()),
        });
    }
    for (const DiskDoneView& completion : disk_done) {
        execution_event.With(tokenspeed::cache::DiskDone{.op_id = completion.op_id, .success = completion.success});
    }
    for (const CacheEvent& event : cache_events) {
        switch (event.kind) {
            case CacheEventKind::WriteBackDone:
                execution_event.With(tokenspeed::cache::WriteBackDone{.op_id = event.op_id});
                break;
            case CacheEventKind::LoadBackDone:
                execution_event.With(tokenspeed::cache::LoadBackDone{.op_id = event.op_id});
                break;
            case CacheEventKind::SnapshotDone:
                execution_event.With(tokenspeed::cache::SnapshotDone{.op_id = event.op_id});
                break;
            case CacheEventKind::RestoreDone:
                execution_event.With(tokenspeed::cache::RestoreDone{.op_id = event.op_id});
                break;
            case CacheEventKind::StateCommitDone:
                execution_event.With(tokenspeed::cache::StateCommitDone{.op_id = event.op_id});
                break;
            case CacheEventKind::SharedL2WriteDone:
                execution_event.With(tokenspeed::cache::SharedL2WriteDone{.op_id = event.op_id});
                break;
            default:
                throw std::invalid_argument("unknown cache event kind");
        }
    }
    scheduler.Advance(execution_event);
}

std::size_t demote_sessions(tokenspeed::Scheduler& scheduler, rust::Slice<const rust::String> ids, bool ended) {
    std::vector<std::string> session_ids;
    for (const auto& id : ids) session_ids.emplace_back(id);
    return scheduler.DemoteSessions(session_ids, ended);
}

std::size_t demote_cache_groups(tokenspeed::Scheduler& scheduler, rust::Slice<const rust::String> ids, bool ended) {
    std::vector<std::string> group_ids;
    for (const auto& id : ids) group_ids.emplace_back(id);
    return scheduler.DemoteCacheGroups(group_ids, ended);
}

std::size_t demote_session(tokenspeed::Scheduler& scheduler, rust::Str session_id, bool ended) {
    return scheduler.DemoteSession(std::string(session_id), ended);
}

bool shared_prefix_storage_enabled(const tokenspeed::Scheduler& scheduler) {
    return scheduler.SharedPrefixStorageEnabled();
}

SharedPrefixPublication shared_prefix_publication(const tokenspeed::Scheduler& scheduler,
                                                  rust::Slice<const std::int32_t> input_ids,
                                                  std::int32_t max_prefix_tokens) {
    const auto result = scheduler.GetSharedPrefixPublication({input_ids.data(), input_ids.size()}, max_prefix_tokens);
    return SharedPrefixPublication{rust::String(result.status), result.confirmed_prefix_tokens, result.cache_epoch};
}

AdmissionSnapshot admission_snapshot(const tokenspeed::Scheduler& scheduler, std::int32_t token_step) {
    AdmissionSnapshot snapshot;
    snapshot.free_slots = scheduler.FreeRequestSlots();
    snapshot.total_slots = scheduler.TotalRequestSlots();
    snapshot.available_blocks = scheduler.AvailableLcmBlocks();
    snapshot.total_blocks = scheduler.TotalLcmBlocks();
    snapshot.host_free_blocks = scheduler.HostPoolFreeBlocks();
    snapshot.host_total_blocks = scheduler.HostPoolTotalBlocks();
    snapshot.retracted = scheduler.RetractedSize();
    snapshot.waiting = scheduler.WaitingSize();
    snapshot.max_tokens = scheduler.MaxSingleRequestTokens();
    snapshot.capacity_token_step = token_step;
    for (auto blocks : scheduler.AdmissionCapacityCurve(token_step)) snapshot.capacity_blocks.push_back(blocks);
    return snapshot;
}

SchedulerSizes sizes(const tokenspeed::Scheduler& scheduler) {
    return SchedulerSizes{
        .running = scheduler.RunningSize(),
        .allocated_request_slots = static_cast<std::size_t>(scheduler.AllocatedRequestSlots()),
        .active_decode_slots = static_cast<std::size_t>(scheduler.ActiveDecodeSlots()),
        .ready_waiting = scheduler.ReadyWaitingSize(),
        .waiting = scheduler.WaitingSize(),
        .pending_prefill_tokens = scheduler.PendingPrefillTokens(),
        .prefilling = scheduler.PrefillSize(),
        .decoding = scheduler.DecodingSize(),
        .retracted = scheduler.RetractedSize(),
        .bootstrapping = scheduler.BootstrappingSize(),
        .remote_prefilling = scheduler.RemotePrefillSize(),
        .pd_transfer = scheduler.PdTransferSize(),
        .prefill_inflight = scheduler.PrefillInflightSize(),
        .empty_lcm_blocks = scheduler.EmptyLcmBlocks(),
    };
}

std::int32_t active_lcm_blocks(const tokenspeed::Scheduler& scheduler) { return scheduler.ActiveLcmBlocks(); }

PrefillGateProbe probe_prefill_gate(const tokenspeed::Scheduler& scheduler) {
    const tokenspeed::Scheduler::PrefillGateProbe probe = scheduler.ProbePrefillGate();
    return PrefillGateProbe{
        .wants_prefill = probe.wants_prefill,
        .has_decode_work = probe.has_decode_work,
    };
}

CacheGroupPages cache_group_pages(const tokenspeed::Scheduler& scheduler, rust::Str group_id) {
    const std::string id(group_id);
    return CacheGroupPages{
        .total = scheduler.CacheGroupTotalPages(id),
        .available = scheduler.CacheGroupAvailablePages(id),
    };
}

std::unique_ptr<tokenspeed::KvEventBatch> drain_kv_events(tokenspeed::Scheduler& scheduler) {
    auto batch = scheduler.DrainKvEventBatch();
    if (batch.Empty()) return nullptr;
    return std::make_unique<tokenspeed::KvEventBatch>(std::move(batch));
}

std::unique_ptr<tokenspeed::KvEventBatch> make_kv_event_batch(rust::Slice<const KvEventView> events) {
    if (events.empty()) return nullptr;
    auto batch = std::make_unique<tokenspeed::KvEventBatch>();
    for (const auto& event : events) {
        const std::span<const std::uint64_t> hashes{event.block_hashes.data(), event.block_hashes.size()};
        if (event.medium >= tokenspeed::kNumKvEventMedia) throw std::invalid_argument("an unknown KV event medium");
        const auto medium = static_cast<tokenspeed::KvEventMedium>(event.medium);
        if (event.stored) {
            batch->Store(hashes, event.has_parent ? std::optional{event.parent_block_hash} : std::nullopt,
                         {event.token_ids.data(), event.token_ids.size()}, event.block_size, medium);
        } else {
            batch->Remove(hashes, medium);
        }
    }
    return batch;
}

std::size_t kv_event_batch_size(const tokenspeed::KvEventBatch& batch) {
    return batch.Size();
}
std::size_t kv_event_batch_bytes(const tokenspeed::KvEventBatch& batch) {
    return batch.ByteSize();
}
rust::Slice<const std::uint8_t> kv_event_stored(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.Stored();
    return {values.data(), values.size()};
}
rust::Slice<const std::uint8_t> kv_event_has_parent(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.HasParent();
    return {values.data(), values.size()};
}
rust::Slice<const std::uint64_t> kv_event_parents(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.Parents();
    return {values.data(), values.size()};
}
rust::Slice<const std::int32_t> kv_event_block_sizes(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.BlockSizes();
    return {values.data(), values.size()};
}
rust::Slice<const std::size_t> kv_event_hash_offsets(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.HashOffsets();
    return {values.data(), values.size()};
}
rust::Slice<const std::size_t> kv_event_token_offsets(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.TokenOffsets();
    return {values.data(), values.size()};
}
rust::Slice<const std::uint64_t> kv_event_hashes(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.Hashes();
    return {values.data(), values.size()};
}
rust::Slice<const std::int32_t> kv_event_tokens(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.Tokens();
    return {values.data(), values.size()};
}

rust::Slice<const std::uint8_t> kv_event_media(const tokenspeed::KvEventBatch& batch) {
    const auto values = batch.Media();
    return {values.data(), values.size()};
}

std::int32_t request_token_size(const tokenspeed::Scheduler& scheduler, rust::Str request_id) {
    return scheduler.RequestTokenSize(std::string(request_id));
}

bool can_abort_before_admission(const tokenspeed::Scheduler& scheduler, rust::Str request_id) {
    return scheduler.CanAbortBeforeAdmission(std::string(request_id));
}

bool defers_decode_state(const tokenspeed::Scheduler& scheduler) { return scheduler.DefersDecodeState(); }

bool storage_pending(const tokenspeed::Scheduler& scheduler) { return scheduler.StoragePending(); }
bool storage_current_pending(const tokenspeed::Scheduler& scheduler) { return scheduler.StorageCurrentPending(); }
bool storage_progress_pending(const tokenspeed::Scheduler& scheduler) { return scheduler.StorageProgressPending(); }
std::size_t storage_blocking_quarantined(const tokenspeed::Scheduler& scheduler) {
    return scheduler.StorageBlockingQuarantined();
}
bool storage_non_storage_clearable(const tokenspeed::Scheduler& scheduler) {
    return scheduler.StorageNonStorageClearable();
}
bool can_prepare_storage_clear(const tokenspeed::Scheduler& scheduler) {
    return scheduler.CanPrepareStorageClear();
}

StorageStatsView storage_stats(const tokenspeed::Scheduler& scheduler) {
    const tokenspeed::StorageStats stats = scheduler.GetStorageStats();
    return StorageStatsView{
        .operations = stats.operations,
        .held_blocks = stats.held_blocks,
        .retained_operations = stats.retained_operations,
        .retained_host_blocks = stats.retained_host_blocks,
        .retained_capacity_exhausted = stats.retained_capacity_exhausted,
        .retained_capacity_warnings = stats.retained_capacity_warnings,
        .quarantined = stats.quarantined,
        .negative_keys = stats.negative_keys,
        .backup_candidates = stats.backup_candidates,
        .waiting_requests = stats.waiting_requests,
        .ready_requests = stats.ready_requests,
        .wait_timeouts = stats.wait_timeouts,
        .ready_expirations = stats.ready_expirations,
        .pressure_gate_closed = stats.pressure_gate_closed,
        .pressure_bypasses = stats.pressure_bypasses,
        .pressure_dropped_backup_keys = stats.pressure_dropped_backup_keys,
        .overflow_dropped_backup_keys = stats.overflow_dropped_backup_keys,
        .unavailable_dropped_backup_keys = stats.unavailable_dropped_backup_keys,
        .disabled_dropped_backup_keys = stats.disabled_dropped_backup_keys,
        .cleanup_dropped_backup_keys = stats.cleanup_dropped_backup_keys,
    };
}

DiskStatsView disk_stats(const tokenspeed::Scheduler& scheduler) {
    const tokenspeed::DiskStats stats = scheduler.GetDiskStats();
    return DiskStatsView{
        .operations = stats.operations,
        .held_host_blocks = stats.held_host_blocks,
        .queued_cleanings = stats.queued_cleanings,
        .waiting_requests = stats.waiting_requests,
        .ready_requests = stats.ready_requests,
        .disk_blocks = stats.disk_blocks,
        .free_disk_parents = stats.free_disk_parents,
        .written_blocks = stats.written_blocks,
        .read_blocks = stats.read_blocks,
        .failed_reads = stats.failed_reads,
        .failed_writes = stats.failed_writes,
        .dropped_cleanings = stats.dropped_cleanings,
        .wait_timeouts = stats.wait_timeouts,
        .ready_expirations = stats.ready_expirations,
    };
}

PrefixHitView prefix_hit_stats(const tokenspeed::Scheduler& scheduler) {
    const tokenspeed::HostCacheStats stats = scheduler.GetHostCacheStats();
    return PrefixHitView{
        .prompt_tokens = stats.prefix_hits.prompt_tokens,
        .device_hit_tokens = stats.prefix_hits.device_hit_tokens,
        .host_hit_tokens = stats.prefix_hits.host_hit_tokens,
        .peer_hit_tokens = stats.prefix_hits.peer_hit_tokens,
        .l3_hit_tokens = stats.storage_query_hit_tokens,
    };
}

bool clear_cache(tokenspeed::Scheduler& scheduler) { return scheduler.ClearCache(); }

void cancel_storage_backups(tokenspeed::Scheduler& scheduler) { scheduler.CancelStorageBackups(); }
void cancel_storage_for_clear(tokenspeed::Scheduler& scheduler) { scheduler.CancelStorageForClear(); }

bool can_clear_cache_retaining_storage(const tokenspeed::Scheduler& scheduler) {
    return scheduler.CanClearCacheRetainingStorage();
}

bool clear_cache_retaining_storage(tokenspeed::Scheduler& scheduler) {
    return scheduler.ClearCacheRetainingStorage();
}

bool storage_has_retained_operations(const tokenspeed::Scheduler& scheduler) {
    return scheduler.StorageHasRetainedOperations();
}

bool clear_l1_cache(tokenspeed::Scheduler& scheduler) { return scheduler.ClearL1Cache(); }

bool can_clear_cache(const tokenspeed::Scheduler& scheduler) { return scheduler.CanClearCache(); }

std::int32_t max_single_request_tokens(const tokenspeed::Scheduler& scheduler) {
    return scheduler.MaxSingleRequestTokens();
}

namespace {

rust::Vec<StorageKeyView> to_rust(const std::vector<tokenspeed::CacheKey>& keys) {
    rust::Vec<StorageKeyView> out;
    out.reserve(keys.size());
    for (const tokenspeed::CacheKey& key : keys) {
        out.push_back(StorageKeyView{.group_id = key.group_id,
                                     .content_hash = rust::String(key.content_hash.ToString()),
                                     .page_offset = key.page_offset});
    }
    return out;
}

std::vector<std::string> to_std(const rust::Vec<rust::String>& values) {
    std::vector<std::string> out;
    out.reserve(values.size());
    for (const rust::String& value : values) {
        out.emplace_back(value);
    }
    return out;
}

// Keys of the default namespace: a view carries no other.
std::vector<tokenspeed::CacheKey> to_std(rust::Slice<const StorageKeyView> keys) {
    std::vector<tokenspeed::CacheKey> out;
    out.reserve(keys.size());
    for (const StorageKeyView& key : keys) {
        out.push_back(tokenspeed::CacheKey{.group_id = key.group_id,
                                           .content_hash = tokenspeed::ContentHash{std::string(key.content_hash)},
                                           .page_offset = key.page_offset});
    }
    return out;
}

// A view's tail. No tail is no tokens and no hash; one without the other is a
// tail the scheduler and the pool refuse.
std::optional<tokenspeed::TailPublication> to_tail(std::int32_t tail_tokens, const rust::String& tail_hash) {
    if (tail_tokens == 0 && tail_hash.empty()) {
        return std::nullopt;
    }
    return tokenspeed::TailPublication{.tail_tokens = tail_tokens,
                                       .hash = tokenspeed::ContentHash{std::string(tail_hash)}};
}

}  // namespace

rust::Vec<SharedL2LookupView> start_shared_l2_lookups(tokenspeed::Scheduler& scheduler) {
    rust::Vec<SharedL2LookupView> out;
    for (const tokenspeed::SharedL2Lookup& lookup : scheduler.StartSharedL2Lookups()) {
        out.push_back(SharedL2LookupView{.request_id = rust::String(lookup.request_id),
                                         .prefix_hashes = ToRustStrings(lookup.prefix_hashes),
                                         .tokens = to_rust(lookup.tokens),
                                         .max_tokens = lookup.max_tokens});
    }
    return out;
}

rust::Vec<SharedL2ProxyBlockView> apply_shared_l2_answers(tokenspeed::Scheduler& scheduler,
                                                          rust::Slice<const SharedL2AnswerView> answers) {
    std::vector<tokenspeed::SharedL2Answer> converted;
    converted.reserve(answers.size());
    for (const SharedL2AnswerView& answer : answers) {
        converted.push_back(tokenspeed::SharedL2Answer{
            .request_id = std::string(answer.request_id),
            .ticket = answer.ticket,
            .num_tokens = answer.num_tokens,
            .kv_tokens = answer.kv_tokens,
            .tail = to_tail(answer.tail_tokens, answer.tail_hash),
            .keys = to_std({answer.keys.data(), answer.keys.size()}),
            .buckets = to_std({answer.buckets.data(), answer.buckets.size()}),
            .pool_pages = to_std({answer.pool_pages.data(), answer.pool_pages.size()})});
    }
    rust::Vec<SharedL2ProxyBlockView> out;
    for (const tokenspeed::SharedL2ProxyBlock& block : scheduler.ApplySharedL2Answers(converted)) {
        out.push_back(SharedL2ProxyBlockView{.group_id = block.group_id,
                                             .host_block = block.host_block,
                                             .bucket = block.bucket,
                                             .pool_page = block.pool_page});
    }
    return out;
}

namespace {

ForwardBatchView batch_view(const tokenspeed::ForwardBatch& batch) {
    return ForwardBatchView{
        .request_ids = ToRustStrings(batch.request_ids),
        .request_pool_indices = to_rust(batch.request_pool_indices),
        .input_lengths = to_rust(batch.input_lengths),
        .prefill_lengths = to_rust(batch.prefill_lengths),
        .extend_prefix_lens = to_rust(batch.extend_prefix_lens),
        .extend_replay_lens = to_rust(batch.extend_replay_lens),
        .num_extends = batch.NumExtends(),
    };
}

TransferBatchView transfer_view(const tokenspeed::ForwardBatch& batch) {
    TransferBatchView view;
    view.request_ids = ToRustStrings(batch.request_ids);
    view.request_pool_indices = to_rust(batch.request_pool_indices);
    view.input_lengths = to_rust(batch.input_lengths);
    view.prefill_lengths = to_rust(batch.prefill_lengths);
    view.extend_prefix_lens = to_rust(batch.extend_prefix_lens);
    view.state_prefix_cached.reserve(batch.state_prefix_cached.size());
    for (const bool cached : batch.state_prefix_cached) {
        view.state_prefix_cached.push_back(cached);
    }
    view.decode_input_ids = to_rust(batch.decode_input_ids);
    view.spec_candidate_offsets.push_back(0);
    for (const std::vector<std::int32_t>& candidates : batch.spec_candidate_ids) {
        for (const std::int32_t id : candidates) {
            view.spec_candidate_ids.push_back(id);
        }
        view.spec_candidate_offsets.push_back(view.spec_candidate_ids.size());
    }
    view.num_extends = batch.NumExtends();
    for (const auto& [group_id, table] : batch.block_tables) {
        view.block_tables.push_back(BlockTableView{
            .group_id = rust::String(group_id),
            .columns = table.Columns(),
            .blocks = to_rust(table.AbsoluteRows()),
        });
    }
    return view;
}

rust::Vec<TransferBatchView> optional_transfer(const std::optional<tokenspeed::ForwardBatch>& batch) {
    rust::Vec<TransferBatchView> out;
    if (batch.has_value()) {
        out.push_back(transfer_view(*batch));
    }
    return out;
}

rust::Vec<ForwardBatchView> optional_batch(const std::optional<tokenspeed::ForwardBatch>& batch) {
    rust::Vec<ForwardBatchView> out;
    if (batch.has_value()) {
        out.push_back(batch_view(*batch));
    }
    return out;
}

}  // namespace

rust::Vec<ForwardBatchView> plan_forward_batches(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<ForwardBatchView> out;
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation);
        if (batch != nullptr) {
            out.push_back(batch_view(*batch));
        }
    }
    return out;
}

rust::Vec<rust::String> plan_completed_prompts(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<rust::String> out;
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation);
        if (batch == nullptr) {
            continue;
        }
        for (std::size_t row = 0; row < batch->NumExtends(); ++row) {
            if (batch->CompletesPrompt(row)) {
                out.push_back(rust::String(batch->request_ids[row]));
            }
        }
    }
    return out;
}

bool plan_needs_device(const tokenspeed::ExecutionPlan& plan) {
    // A peer landing's reads start from the fence the data plane reports for
    // its plan, which may have no other device work: a prefill replica's
    // landing prepays no decode room, so it zeroes nothing.
    // The commit fence is acknowledged from the device, behind the commit, and
    // a retraction whose image was refused may leave its state folds as the
    // plan's only device work.
    if (plan.remote_decode.has_value() || plan.remote_prefill.has_value() || !plan.peer_fetches.empty() ||
        plan.state_commit_fence.has_value() || !plan.state_folds.empty()) {
        return true;
    }
    for (const auto& [group_id, pages] : plan.pages_to_zero) {
        if (!pages.empty()) {
            return true;
        }
    }
    for (const auto& [group_id, pairs] : plan.pages_to_copy) {
        if (!pairs.empty()) {
            return true;
        }
    }
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        if (const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation)) {
            if (!batch->request_ids.empty()) {
                return true;
            }
            continue;
        }
        // Storage and Disk operations run on the host's executors and shared
        // L2 ops on its pool client, not the device.
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        if (cache == nullptr || (!std::holds_alternative<tokenspeed::StorageOperation>(*cache) &&
                                 !std::holds_alternative<tokenspeed::DiskOperation>(*cache) &&
                                 !std::holds_alternative<tokenspeed::SharedL2Operation>(*cache))) {
            return true;
        }
    }
    return false;
}

bool plan_has_pages_to_copy(const tokenspeed::ExecutionPlan& plan) {
    for (const auto& [group_id, pairs] : plan.pages_to_copy) {
        if (!pairs.empty()) {
            return true;
        }
    }
    return false;
}

rust::Vec<ForwardBatchView> plan_remote_decode(const tokenspeed::ExecutionPlan& plan) {
    return optional_batch(plan.remote_decode);
}

rust::Vec<ForwardBatchView> plan_remote_prefill(const tokenspeed::ExecutionPlan& plan) {
    return optional_batch(plan.remote_prefill);
}

rust::Vec<TransferBatchView> plan_forward_transfer(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<TransferBatchView> out;
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        if (const auto* batch = std::get_if<tokenspeed::ForwardBatch>(&operation)) {
            out.push_back(transfer_view(*batch));
            break;
        }
    }
    return out;
}

rust::Vec<TransferBatchView> plan_remote_decode_transfer(const tokenspeed::ExecutionPlan& plan) {
    return optional_transfer(plan.remote_decode);
}

rust::Vec<TransferBatchView> plan_remote_prefill_transfer(const tokenspeed::ExecutionPlan& plan) {
    return optional_transfer(plan.remote_prefill);
}

rust::Vec<SchedulerAbortView> plan_aborts(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<SchedulerAbortView> out;
    out.reserve(plan.aborts.size());
    for (const tokenspeed::SchedulerAbort& abort : plan.aborts) {
        out.push_back(SchedulerAbortView{
            .request_id = rust::String(abort.request_id),
            .detail = rust::String(abort.detail),
        });
    }
    return out;
}

rust::Vec<CacheOpView> plan_cache_ops(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<CacheOpView> out;
    // One acknowledgment per op id, whatever the batch kind: an L2 batch
    // lists its ops, and a snapshot or restore batch has one op per request.
    // Storage and Disk operations are the host executors', completed through
    // their agreements instead.
    auto append = [&out](CacheOpKind kind, const std::vector<std::uint32_t>& op_ids) {
        for (std::uint32_t op_id : op_ids) {
            out.push_back(CacheOpView{.kind = kind, .op_id = op_id});
        }
    };
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        if (cache == nullptr) {
            continue;
        }
        std::visit(
            [&append](const auto& batch) {
                using Batch = std::decay_t<decltype(batch)>;
                if constexpr (std::is_same_v<Batch, tokenspeed::WriteBackBatch>) {
                    append(CacheOpKind::WriteBack, batch.op_ids);
                } else if constexpr (std::is_same_v<Batch, tokenspeed::LoadBackBatch>) {
                    append(CacheOpKind::LoadBack, batch.op_ids);
                } else if constexpr (std::is_same_v<Batch, tokenspeed::SnapshotStoreBatch>) {
                    append(CacheOpKind::SnapshotStore, batch.op_ids);
                } else if constexpr (std::is_same_v<Batch, tokenspeed::SnapshotRestoreBatch>) {
                    append(CacheOpKind::SnapshotRestore, batch.op_ids);
                } else {
                    // Shared L2 ops are the pool client's: none is
                    // acknowledged here.
                    static_assert(std::is_same_v<Batch, tokenspeed::StorageOperation> ||
                                      std::is_same_v<Batch, tokenspeed::DiskOperation> ||
                                      std::is_same_v<Batch, tokenspeed::SharedL2Operation>,
                                  "a new cache batch kind");
                }
            },
            *cache);
    }
    if (plan.state_commit_fence) {
        out.push_back(CacheOpView{.kind = CacheOpKind::StateCommit, .op_id = *plan.state_commit_fence});
    }
    return out;
}

rust::Vec<StorageOpView> plan_storage_ops(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<StorageOpView> out;
    const auto keys = [](const std::vector<tokenspeed::CacheKey>& cache_keys) {
        rust::Vec<StorageKeyView> views;
        views.reserve(cache_keys.size());
        for (const tokenspeed::CacheKey& key : cache_keys) {
            views.push_back(StorageKeyView{.group_id = key.group_id,
                                           .content_hash = rust::String(key.content_hash.ToString()),
                                           .page_offset = key.page_offset});
        }
        return views;
    };
    const auto blocks = [](const std::vector<std::int32_t>& host_blocks) {
        rust::Vec<std::int32_t> out;
        out.reserve(host_blocks.size());
        for (const std::int32_t block : host_blocks) {
            out.push_back(block);
        }
        return out;
    };
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        if (cache == nullptr) {
            continue;
        }
        const auto* storage = std::get_if<tokenspeed::StorageOperation>(cache);
        if (storage == nullptr) {
            continue;
        }
        std::visit(
            [&](const auto& op) {
                using Op = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<Op, tokenspeed::StorageCancelOperation>) {
                    out.push_back(StorageOpView{
                        .kind = StorageOpKind::Cancel, .op_id = op.op_id, .keys = {}, .host_blocks = {}});
                } else if constexpr (std::is_same_v<Op, tokenspeed::StorageQueryOperation>) {
                    out.push_back(StorageOpView{
                        .kind = StorageOpKind::Query, .op_id = op.op_id, .keys = keys(op.keys), .host_blocks = {}});
                } else {
                    const StorageOpKind kind = std::is_same_v<Op, tokenspeed::StorageGetOperation>
                                                   ? StorageOpKind::Get
                                                   : StorageOpKind::Put;
                    out.push_back(StorageOpView{.kind = kind,
                                                .op_id = op.op_id,
                                                .keys = keys(op.keys),
                                                .host_blocks = blocks(op.host_blocks)});
                }
            },
            *storage);
    }
    return out;
}

rust::Vec<DiskOpView> plan_disk_ops(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<DiskOpView> out;
    const auto copy = [](const auto& values) {
        rust::Vec<std::decay_t<decltype(values[0])>> out;
        out.reserve(values.size());
        for (const auto value : values) {
            out.push_back(value);
        }
        return out;
    };
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        const auto* disk = cache == nullptr ? nullptr : std::get_if<tokenspeed::DiskOperation>(cache);
        if (disk == nullptr) {
            continue;
        }
        out.push_back(DiskOpView{
            .kind = disk->kind == tokenspeed::DiskOpKind::kRead ? DiskOpKind::Read : DiskOpKind::Write,
            .op_id = disk->op_id,
            .group_ids = copy(disk->group_ids),
            .host_blocks = copy(disk->host_blocks),
            .disk_blocks = copy(disk->disk_blocks),
        });
    }
    return out;
}

rust::Vec<SharedL2OpView> plan_shared_l2_ops(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<SharedL2OpView> out;
    for (const tokenspeed::Operation& operation : plan.Operations()) {
        const auto* cache = std::get_if<tokenspeed::CacheOperation>(&operation);
        if (cache == nullptr) {
            continue;
        }
        const auto* shared_l2_operation = std::get_if<tokenspeed::SharedL2Operation>(cache);
        if (shared_l2_operation == nullptr) {
            continue;
        }
        std::visit(
            [&out](const auto& op) {
                using Op = std::decay_t<decltype(op)>;
                SharedL2OpView view{};
                if constexpr (std::is_same_v<Op, tokenspeed::SharedL2WriteOperation>) {
                    view.kind = SharedL2OpKind::Write;
                    view.op_id = op.op_id;
                    view.keys = to_rust(op.keys);
                    view.buckets = to_rust(op.buckets);
                    view.device_blocks = to_rust(op.device_blocks);
                    view.checkpoints.reserve(op.checkpoints.size());
                    for (const tokenspeed::CheckpointTree::Checkpoint& checkpoint : op.checkpoints) {
                        view.checkpoints.push_back(SharedL2CheckpointView{
                            .tokens = to_rust(checkpoint.tokens),
                            .prefix_hashes = ToRustStrings(checkpoint.prefix_hashes),
                            .tail_tokens = checkpoint.tail ? checkpoint.tail->tail_tokens : 0,
                            .tail_hash =
                                rust::String(checkpoint.tail ? checkpoint.tail->hash.ToString() : std::string{}),
                            .access_epoch = checkpoint.access_epoch,
                        });
                    }
                } else if constexpr (std::is_same_v<Op, tokenspeed::SharedL2ReleaseOperation>) {
                    view.kind = SharedL2OpKind::Release;
                    view.ticket = op.ticket;
                } else {
                    static_assert(std::is_same_v<Op, tokenspeed::SharedL2TouchOperation>,
                                  "a new shared L2 operation kind");
                    view.kind = SharedL2OpKind::Touch;
                    view.keys = to_rust(op.keys);
                }
                out.push_back(std::move(view));
            },
            *shared_l2_operation);
    }
    return out;
}

SharedL2AnswerView shared_l2_pool_lookup(tokenspeed::SharedL2Pool& pool, std::uint64_t client,
                                         const SharedL2LookupView& lookup) {
    const std::vector<std::string> prefix_hashes = to_std(lookup.prefix_hashes);
    tokenspeed::PromptPrefix prompt;
    prompt.prefix_hashes = prefix_hashes;
    prompt.tokens = {lookup.tokens.data(), lookup.tokens.size()};
    prompt.max_tokens = lookup.max_tokens;
    const tokenspeed::SharedL2Pool::Hit hit = pool.Lookup(client, prompt);
    std::vector<tokenspeed::CacheKey> keys;
    std::vector<std::int32_t> buckets;
    std::vector<std::int32_t> pool_pages;
    keys.reserve(hit.pages.size());
    buckets.reserve(hit.pages.size());
    pool_pages.reserve(hit.pages.size());
    for (const tokenspeed::SharedL2Pool::Page& page : hit.pages) {
        keys.push_back(page.key);
        buckets.push_back(page.bucket);
        pool_pages.push_back(page.pool_page);
    }
    return SharedL2AnswerView{.request_id = lookup.request_id,
                              .ticket = hit.ticket,
                              .num_tokens = hit.num_tokens,
                              .kv_tokens = hit.kv_tokens,
                              .tail_tokens = hit.tail ? hit.tail->tail_tokens : 0,
                              .tail_hash = rust::String(hit.tail ? hit.tail->hash.ToString() : std::string{}),
                              .keys = to_rust(keys),
                              .buckets = to_rust(buckets),
                              .pool_pages = to_rust(pool_pages)};
}

void shared_l2_pool_release(tokenspeed::SharedL2Pool& pool, std::uint64_t client, std::uint64_t ticket) {
    pool.Release(client, ticket);
}

SharedL2ReservationView shared_l2_pool_reserve(tokenspeed::SharedL2Pool& pool, std::uint64_t client,
                                               rust::Slice<const StorageKeyView> keys,
                                               rust::Slice<const std::int32_t> buckets) {
    const tokenspeed::SharedL2Pool::Reservation reservation =
        pool.Reserve(client, to_std(keys), {buckets.data(), buckets.size()});
    return SharedL2ReservationView{.ticket = reservation.ticket, .pages = to_rust(reservation.pages)};
}

void shared_l2_pool_commit(tokenspeed::SharedL2Pool& pool, std::uint64_t client, std::uint64_t ticket,
                           rust::Slice<const SharedL2CheckpointView> checkpoints) {
    std::vector<tokenspeed::CheckpointTree::Checkpoint> converted;
    converted.reserve(checkpoints.size());
    for (const SharedL2CheckpointView& checkpoint : checkpoints) {
        converted.push_back({.tokens = to_std({checkpoint.tokens.data(), checkpoint.tokens.size()}),
                             .prefix_hashes = to_std(checkpoint.prefix_hashes),
                             .tail = to_tail(checkpoint.tail_tokens, checkpoint.tail_hash),
                             .access_epoch = checkpoint.access_epoch});
    }
    pool.Commit(client, ticket, converted);
}

void shared_l2_pool_touch(tokenspeed::SharedL2Pool& pool, rust::Slice<const StorageKeyView> keys) {
    pool.Touch(to_std(keys));
}

void shared_l2_pool_disconnect(tokenspeed::SharedL2Pool& pool, std::uint64_t client) { pool.Disconnect(client); }

bool shared_l2_pool_clear_cache(tokenspeed::SharedL2Pool& pool) { return pool.ClearCache(); }

SharedL2PoolStatsView shared_l2_pool_stats(const tokenspeed::SharedL2Pool& pool) {
    const tokenspeed::SharedL2Pool::Stats stats = pool.GetStats();
    return SharedL2PoolStatsView{.num_lcm_blocks = stats.num_lcm_blocks,
                                 .empty_lcm_blocks = stats.empty_lcm_blocks,
                                 .cached_blocks = stats.cached_blocks,
                                 .pinned_blocks = stats.pinned_blocks,
                                 .reserved_blocks = stats.reserved_blocks,
                                 .open_lookups = stats.open_lookups,
                                 .evicted_blocks = stats.evicted_blocks,
                                 .checkpoints = stats.checkpoints};
}

void set_profile_span_sink(bool installed) {
    tokenspeed::SetProfileSpanSink(installed ? &record_profile_span : nullptr);
}

static PeerFetchView FetchView(const tokenspeed::PeerFetchOperation& fetch) {
    return PeerFetchView{.request_id = rust::String(fetch.request_id),
                                    .lease = ToView(fetch.lease),
                                    .holder = rust::String(fetch.holder),
                                    .begin_tokens = fetch.begin_tokens,
                                    .end_tokens = fetch.end_tokens,
                                    .destinations = ToView(fetch.destinations),
                                    .checkpoint = ToRustVec(fetch.checkpoint),
                                    .prefix_hashes = ToRustStrings(fetch.prefix_hashes),
                                    .tokens = ToRustVec(fetch.tokens)};
}

bool release_peer_pin(const std::shared_ptr<tokenspeed::PeerPinRelease>& handle) { return handle && handle->Release(); }

std::shared_ptr<tokenspeed::PeerPinRelease> peer_release_handle(tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease) {
    return scheduler.PeerReleaseHandle(FromView(lease));
}

bool enable_async_host_peer(tokenspeed::Scheduler& scheduler) { return scheduler.EnableAsyncHostPeer(); }
bool enable_async_device_peer(tokenspeed::Scheduler& scheduler, bool adaptive) { return scheduler.EnableAsyncDevicePeer(adaptive); }

rust::Vec<PeerFetchView> reserve_peer_host(tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease, std::int32_t tokens, bool& retry) {
    rust::Vec<PeerFetchView> out;
    if (auto fetch = scheduler.ReservePeerHost(FromView(lease), tokens, retry)) out.push_back(FetchView(*fetch));
    return out;
}

bool pin_peer_offer(tokenspeed::Scheduler& scheduler, const PeerEventView& event, std::int64_t now_us) {
    auto checkpoint = LookupFromView(event);
    return scheduler.PinPeerOffer(checkpoint, tokenspeed::cache::PeerPin{
        .lease = checkpoint.lease, .num_tokens = event.num_tokens,
        .pages = FromView(rust::Slice<const PeerPageView>(event.pages.data(), event.pages.size()))}, now_us);
}

bool close_peer_offer(tokenspeed::Scheduler& scheduler, const PeerLeaseView& lease, std::int64_t now_us) {
    return scheduler.ClosePeerOffer(FromView(lease), now_us);
}

bool apply_peer_export(tokenspeed::Scheduler& scheduler, const PeerEventView& event, std::int64_t now_us) {
    switch (event.kind) {
        case PeerEventKind::Lookup: return scheduler.ApplyPeerExport(LookupFromView(event), now_us);
        case PeerEventKind::Pin: {
            std::optional<tokenspeed::cache::PeerLookup> checkpoint;
            if (event.max_tokens > 0) checkpoint.emplace(LookupFromView(event));
            scheduler.ApplyPeerExport(tokenspeed::cache::PeerPin{
            .lease = FromView(event.lease), .num_tokens = event.num_tokens,
            .pages = FromView(rust::Slice<const PeerPageView>(event.pages.data(), event.pages.size()))}, now_us, checkpoint ? &*checkpoint : nullptr);
            return scheduler.PeerLeaseHeld(FromView(event.lease));
        }
        case PeerEventKind::Release: {
            const bool held = scheduler.PeerLeaseHeld(FromView(event.lease));
            scheduler.ApplyPeerExport(tokenspeed::cache::PeerRelease{
            .lease = FromView(event.lease), .demote = event.demote}, now_us); return held;
        }
        default: throw std::invalid_argument("expected holder event");
    }
}

rust::Vec<PeerLocalChoice> plan_peer_local_choices(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<PeerLocalChoice> out;
    for (const auto& [id, reason] : plan.peer_local_choices) out.push_back({rust::String(id), rust::String(reason)});
    return out;
}

rust::Vec<PeerFetchView> plan_peer_fetches(const tokenspeed::ExecutionPlan& plan) {
    rust::Vec<PeerFetchView> out;
    for (const tokenspeed::PeerFetchOperation& fetch : plan.peer_fetches) {
        out.push_back(FetchView(fetch));
    }
    return out;
}

}  // namespace ts_scheduler
