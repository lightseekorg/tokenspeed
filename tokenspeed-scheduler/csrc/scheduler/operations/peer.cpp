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

// Peer prefix sharing (docs/rfcs/0004-peer-prefix-sharing.md): the holder
// answers lookups, pins and releases from its own cache; the reader asks
// once per prompt, admits a landing for what its holder can add, and either
// keeps the landed prefix or is admitted again from its local hit.

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

#include "cache/prefix/prefix_hasher.h"
#include "core/profile_span.h"
#include "fsm/peer_events.h"
#include "fsm/storage_events.h"
#include "scheduler/scheduler.h"

namespace tokenspeed {

bool Scheduler::PeerPagesMatch(const cache::PeerLookup& lookup, std::int32_t num_tokens,
                               std::span<const PeerPage> pages) const {
    if (!PeerExports::Matches(coordinator_, lookup, num_tokens, pages)) {
        return false;
    }
    // A decoded endpoint a retraction published holds its deferred state only
    // after the runtime acknowledged the commit (ExecutionPlan::
    // state_commit_fence); local forwards order behind it, a remote read
    // does not.
    if (std::ranges::any_of(pages, [&](const PeerPage& page) {
            return page.tier == CacheTier::kDevice && awaitsStateCommit(page.group_id, page.block_id);
        })) {
        return false;
    }
    if (!tier_transfers_.HasLoadBacksInFlight()) {
        return true;
    }
    // Local forwards wait on layerwise H2D fences; a remote reader cannot.
    // A PIN may export a load's Device destinations only after every rank
    // acknowledged it. Other Device pages and Host sources remain readable.
    const auto loading = tier_transfers_.LoadBackDestinations();
    return std::ranges::none_of(pages, [&](const PeerPage& page) {
        return page.tier == CacheTier::kDevice && loading[page.group_id].contains(page.block_id);
    });
}
bool Scheduler::awaitsStateCommit(std::uint32_t group, std::int32_t page) const {
    const std::pair key{group, page};
    return uncommitted_state_pages_.contains(key) ||
           std::ranges::any_of(state_commit_fences_, [&](const auto& fence) { return fence.second.contains(key); });
}

namespace {

std::int64_t Deadline(std::int64_t now_us, std::int64_t lifetime_us) {
    return now_us + std::min(lifetime_us, std::numeric_limits<std::int64_t>::max() - now_us);
}

}  // namespace

Scheduler::PeerStats Scheduler::GetPeerStats() const {
    PeerStats stats{.landed_tokens = peer_landed_tokens_, .failed_landings = peer_failed_landings_};
    if (peer_exports_) {
        stats.leases = peer_exports_->NumLeases();
        stats.pinned_blocks = peer_exports_->NumPinnedBlocks();
    }
    for (const auto& request : requests_) {
        stats.looking_requests += request->Is<fsm::PeerLooking>();
        stats.landing_requests += request->Is<fsm::PeerLanding>();
    }
    if (peer_host_landings_) stats.landing_requests += peer_host_landings_->NumLandings();
    return stats;
}

void Scheduler::handleEvent(const cache::PeerTick& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer tick without configured peer sharing");
    if (event.now_us < peer_now_us_) throw std::invalid_argument("the peer clock moved backwards");
    peer_now_us_ = event.now_us;
    peer_exports_->Expire(peer_now_us_, async_host_peer_ || async_device_peer_);
    std::erase_if(peer_host_grants_, [](const auto& entry) { return entry.second.expired(); });
    for (const auto& request : requests_) {
        if (const auto* looking = request->GetIf<fsm::PeerLooking>();
            looking != nullptr && !async_host_peer_ && !async_device_peer_ && peer_now_us_ >= looking->deadline_us) {
            spdlog::debug("[Peer] rid={} lookup {} unanswered; admitted from the local hit", request->Id(),
                          looking->lease.id);
            request->Apply(fsm::FinishPeerLookupEvent{std::nullopt});
        }
        // A prefix landed into Host is held for the admission on the peer
        // clock; past it the Host keeps it as any other entry.
        auto* admission = request->AdmissionResourcesPtr();
        if (peer_host_landings_ && admission != nullptr && admission->storage_hold &&
            admission->storage_hold->lease.IssuedBy(*peer_host_landings_) &&
            peer_now_us_ >= admission->storage_hold->deadline_us) {
            spdlog::debug("[Peer] rid={} Host landing's ready hold expired before admission", request->Id());
            admission->storage_hold.reset();
        }
    }
}

void Scheduler::handleEvent(const cache::PeerLookup& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer lookup without configured peer sharing");
    peer_exports_->Lookup(coordinator_, event, peer_now_us_);
}

void Scheduler::handleEvent(const cache::PeerPin& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer pin without configured peer sharing");
    peer_exports_->Pin(coordinator_, event, peer_now_us_);
}

void Scheduler::handleEvent(const cache::PeerRelease& event) {
    if (!peer_exports_) throw std::invalid_argument("a peer release without configured peer sharing");
    peer_exports_->Release(coordinator_, event, peer_now_us_);
}

void Scheduler::handleEvent(const cache::PeerLookupAnswered& event) {
    for (const auto& request : requests_) {
        const auto* looking = request->GetIf<fsm::PeerLooking>();
        if (looking == nullptr || looking->lease != event.lease) {
            continue;
        }
        if (async_host_peer_ && event.num_tokens > 0) {
            if (looking->host_grant) looking->host_grant->lookup_answered = true;
            return;
        }
        request->Apply(fsm::FinishPeerLookupEvent{event.num_tokens > 0
                                                      ? std::optional{fsm::PeerHit{.lease = event.lease,
                                                                                   .num_tokens = event.num_tokens,
                                                                                   .source_mask = event.source_mask,
                                                                                   .checkpoint = event.checkpoint}}
                                                      : std::nullopt});
        return;
    }
    // The request gave the wait up or ended: the soft pin expires at the holder.
}

void Scheduler::DispatchWait::Observe(std::int64_t us) {
    recent[next] = us;
    next = (next + 1) % recent.size();
    if (next == 0) ready = true;
}

double Scheduler::DispatchWait::Estimate() const {
    if (!ready) return 0.0;
    auto samples = recent;
    std::ranges::sort(samples);
    return static_cast<double>(samples[samples.size() / 2]);
}

void Scheduler::recordPeerDispatch(Request& request) {
    const auto& sample = *request.peer_dispatch_sample;
    const auto after_release = peer_now_us_ - sample.release_us;
    device_dispatch_wait_[sample.batch_class].Observe(sample.fence_wait_us + after_release);
    spdlog::info("[PeerFeedback] rid={} release_to_prefill_round_us={}", request.Id(), after_release);
    request.peer_dispatch_sample.reset();
}

void Scheduler::handleEvent(const cache::PeerFetchDone& event) {
    Request* request = findRequest(event.request_id);
    if (auto attempt = device_attempts_.extract(event.request_id);
        !attempt.empty() && (!grant_first_peer_ || event.grant_source != 0)) {
        const auto& a = attempt.mapped();
        const auto sample_path = a.path + (event.grant_source & 3);
        auto& cost = device_landing_cost_[sample_path];
        const bool ok = event.num_tokens == a.end_tokens;
        if (event.dispatch_us > 0 && event.dispatch_us <= event.writable_us && event.writable_us <= event.terminal_us &&
            event.terminal_us <= peer_now_us_) {
            const auto protocol_us = event.terminal_us - event.writable_us;
            if (ok) {
                const double observed = std::clamp(protocol_us / a.nominal_us, 1.0 / 64.0, 64.0);
                cost.samples = std::min(8, cost.samples + 1);
                cost.scale += (observed - cost.scale) / cost.samples;
                if (request) {
                    request->peer_dispatch_sample = Request::PeerDispatchSample{event.writable_us - event.dispatch_us,
                                                                                event.terminal_us, a.batch_class};
                }
            }
            cost.failure_us += 0.125 * ((ok ? 0.0 : protocol_us) - cost.failure_us);
            spdlog::info("[PeerFeedback] rid={} path={} ok={} fence_us={} protocol_us={} consume_us={} scale={:.4f}",
                         event.request_id, sample_path, ok, event.writable_us - event.dispatch_us, protocol_us,
                         peer_now_us_ - event.terminal_us, cost.scale);
        }
        cost.success_rate += 0.125 * ((ok ? 1.0 : 0.0) - cost.success_rate);
    }
    if (peer_host_landings_) {
        const auto* prefetch = request == nullptr ? nullptr : request->GetIf<fsm::Prefetching>();
        const bool waiting = prefetch != nullptr && prefetch->lease.IssuedBy(*peer_host_landings_);
        auto* active = request ? request->GetIf<fsm::PeerLooking>() : nullptr;
        if (active && active->host_grant && active->host_grant->landing) {
            active->admission.storage_hold =
                fsm::StorageReadyHold{.lease = std::move(active->host_grant->landing), .deadline_us = peer_now_us_};
        }
        if (const std::optional<std::int32_t> landed =
                peer_host_landings_->Finish(event.request_id, event.num_tokens)) {
            const auto* owner = request ? request->GetIf<fsm::PeerLooking>() : nullptr;
            if (*landed > 0 && (!async_host_peer_ || (owner && owner->admission.storage_hold))) {
                peer_landed_tokens_ += static_cast<std::uint64_t>(*landed);
            } else {
                ++peer_failed_landings_;
            }
            // A request that let go of the landing has nothing to step.
            auto* looking = request ? request->GetIf<fsm::PeerLooking>() : nullptr;
            if (async_host_peer_ && looking && looking->admission.storage_hold &&
                looking->admission.storage_hold->lease.IssuedBy(*peer_host_landings_)) {
                if (*landed > 0) {
                    looking->admission.storage_hold->deadline_us =
                        Deadline(peer_now_us_, config_.peer_config->host_ready_timeout_us);
                } else {
                    looking->admission.storage_hold.reset();
                }
                request->Apply(fsm::FinishPeerLookupEvent{std::nullopt});
            } else if (waiting) {
                request->Apply(fsm::FinishPrefetchEvent{
                    *landed > 0 ? std::optional{Deadline(peer_now_us_, config_.peer_config->host_ready_timeout_us)}
                                : std::nullopt});
            }
            return;
        }
    }
    if (request == nullptr) {
        return;
    }
    if (request->Is<fsm::PdTransferring>()) {
        // It ended while landing: the reads have stopped, its pages go back.
        handleEvent(pd::TransferStoppedEvent{event.request_id});
        return;
    }
    const auto* landing = request->GetIf<fsm::PeerLanding>();
    if (landing == nullptr) {
        throw std::logic_error("PeerFetchDone for a request that is not landing; state=" + request->StateName());
    }
    const std::int32_t end = landing->window.begin + landing->window.size;
    if (event.num_tokens == end) {
        peer_landed_tokens_ += static_cast<std::uint64_t>(landing->window.size);
        request->Apply(fsm::PeerLandedEvent{});
        return;
    }
    spdlog::debug("[Peer] rid={} landing confirmed {} of [{}, {}); admitted again from the local hit", request->Id(),
                  event.num_tokens, landing->window.begin, end);
    ++peer_failed_landings_;
    request->Apply(fsm::PeerLandingFailedEvent{&coordinator_});
}

std::vector<PeerPage> Scheduler::peerLandingDestinations(std::span<const BlockTable> tables, std::int32_t hit_tokens,
                                                         std::int32_t end_tokens) const {
    std::vector<PeerPage> destinations;
    for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
        const BlockTable& table = tables[static_cast<std::size_t>(i)];
        const std::span<const CacheBlockRef> blocks = table.Blocks();
        const auto add = [&](std::int32_t slot) {
            destinations.push_back(PeerPage{
                .group_id = static_cast<std::uint32_t>(i),
                .slot = slot,
                .tier = CacheTier::kDevice,
                .block_id = coordinator_.Allocator(i).ResolveCacheBlockId(*blocks[static_cast<std::size_t>(slot)]),
            });
        };
        if (coordinator_.GroupIsState(static_cast<std::size_t>(i))) {
            add(kPeerStateSlot);
            continue;
        }
        const std::int32_t block_granularity = coordinator_.GroupBlockGranularity(i);
        const std::int32_t past_end = std::min(CeilDiv(end_tokens, block_granularity), table.NumBlocks());
        for (std::int32_t slot = hit_tokens / block_granularity; slot < past_end; ++slot) {
            if (blocks[static_cast<std::size_t>(slot)]) {
                add(slot);
            }
        }
    }
    return destinations;
}

void Scheduler::excludeLandedPages(AdmissionResult& admission, std::span<const PeerPage> destinations) const {
    const auto landed = [&](std::size_t group, std::int32_t block_id) {
        return std::ranges::any_of(
            destinations, [&](const PeerPage& page) { return page.group_id == group && page.block_id == block_id; });
    };
    for (std::size_t i = 0; i < admission.new_page_ids.size(); ++i) {
        std::erase_if(admission.new_page_ids[i], [&](std::int32_t block_id) { return landed(i, block_id); });
        for (const auto& [source, destination] : admission.pages_to_copy[i]) {
            _assert(!landed(i, destination), "a landing's destination is no copy destination");
        }
    }
    for (const BlockTransfer& pair : admission.load_pairs) {
        _assert(!landed(pair.group_id, coordinator_.Allocator(static_cast<std::int32_t>(pair.group_id))
                                           .ResolveCacheBlockId(*pair.destination)),
                "a landing's destination is no load destination");
    }
}

bool Scheduler::EnableAsyncDevicePeer(bool adaptive, bool grant_first) {
    if (!peer_exports_ || peer_host_landings_) return false;
    async_device_peer_ = true;
    adaptive_device_peer_ = adaptive;
    grant_first_peer_ = grant_first;
    return true;
}

bool Scheduler::EnableAsyncHostPeer() {
    if (!peer_host_landings_) return false;
    async_host_peer_ = true;
    return true;
}

bool Scheduler::ApplyPeerExport(const cache::PeerLookup& event, std::int64_t now_us) {
    peer_exports_->Expire(now_us, /*retain_hard=*/true);
    // A duplicate lookup must not release pages still being read.
    if (peer_exports_->Held(event.lease)) return false;
    peer_exports_->Lookup(coordinator_, event, now_us);
    return peer_exports_->HasLease(event.lease);
}
void Scheduler::ApplyPeerExport(const cache::PeerPin& event, std::int64_t now_us, const cache::PeerLookup* checkpoint) {
    peer_exports_->Expire(now_us, /*retain_hard=*/true);
    peer_exports_->Pin(coordinator_, event, now_us, checkpoint);
}
void Scheduler::ApplyPeerExport(const cache::PeerRelease& event, std::int64_t now_us) {
    peer_exports_->Release(coordinator_, event, now_us);
}

std::optional<PeerFetchOperation> Scheduler::ReservePeerHost(const PeerLeaseId& lease, std::int32_t num_tokens,
                                                             bool& retry) {
    retry = false;
    if (!async_host_peer_ || !forward_allowed_) return std::nullopt;
    const auto it = peer_host_grants_.find(lease);
    if (it == peer_host_grants_.end()) return std::nullopt;
    auto grant = it->second.lock();
    if (!grant || num_tokens <= 0 || num_tokens > grant->lookup.max_tokens ||
        peer_host_landings_->ContainsRequest(grant->lookup.request_id))
        return std::nullopt;
    const auto grain = config_.prefix_granularity;
    const auto hashes = std::span(grant->lookup.prefix_hashes).first(num_tokens / grain);
    const auto tokens = std::span(grant->lookup.tokens).first(num_tokens);
    std::optional<TailPublication> tail;
    if (auto count = num_tokens % grain) {
        const ContentHash prior = hashes.empty() ? ContentHash{} : ContentHash{hashes.back()};
        tail = TailPublication{.tail_tokens = count, .hash = HashPrefixPage(tokens.last(count), prior)};
    }
    const PromptPrefix prompt{.prefix_hashes = hashes,
                              .tokens = tokens,
                              .max_tokens = num_tokens,
                              .tail = tail,
                              .shared_prefix_tokens = grant->shared_prefix};
    const auto probe = coordinator_.Prober().ProbePromptPrefix(prompt);
    const auto local = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    if (local >= num_tokens) return std::nullopt;
    auto started =
        peer_host_landings_->Start(prompt, grant->lookup.request_id, lease, grant->lookup.holder, local, &retry);
    if (!started) return std::nullopt;
    peer_host_grants_.erase(it);
    grant->landing = std::move(started->lease);
    return std::move(started->fetch);
}

std::optional<PeerHostLandings::Started> Scheduler::reservePeerHost(Request& request, const fsm::PeerHit& hit,
                                                                    std::optional<TailPublication>& tail) {
    const std::int32_t grain = config_.prefix_granularity;
    const auto hashes = request.PromptPrefixHashes(hit.num_tokens / grain);
    if (const std::int32_t tail_tokens = hit.num_tokens % grain; tail_tokens > 0) {
        const ContentHash prior = hashes.empty() ? ContentHash{} : ContentHash{hashes.back()};
        tail = TailPublication{
            .tail_tokens = tail_tokens,
            .hash = HashPrefixPage(request.TokenSlice(hit.num_tokens - tail_tokens, tail_tokens), prior)};
    }
    const PromptPrefix prompt{.prefix_hashes = hashes,
                              .tokens = request.TokenSlice(0, hit.num_tokens),
                              .max_tokens = hit.num_tokens,
                              .tail = tail,
                              .shared_prefix_tokens = request.SharedPrefixTokens()};
    const PrefixProbe probe = coordinator_.Prober().ProbePromptPrefix(prompt);
    const std::int32_t local = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    return local < hit.num_tokens
               ? peer_host_landings_->Start(prompt, request.Id(), hit.lease, request.PeerHolder(), local)
               : std::nullopt;
}

void Scheduler::startPeerHostLanding(Request& request) {
    auto& admission = request.GetIf<fsm::Submitted>()->admission;
    const fsm::PeerHit hit = *std::exchange(admission.peer_hit, std::nullopt);
    std::optional<TailPublication> tail;
    auto started = reservePeerHost(request, hit, tail);
    if (!started) {
        // The local hit grew to the answer, or the Host cannot take the whole
        // answered prefix now: admitted from the local hit; the soft pin
        // lapses at the holder.
        spdlog::debug("[Peer] rid={} Host landing of {} tokens not started; admitted from the local hit", request.Id(),
                      hit.num_tokens);
        return;
    }
    spdlog::debug("[Peer] rid={} lands [{}, {}) from {} into Host ({} blocks)", request.Id(),
                  started->fetch.begin_tokens, hit.num_tokens, request.PeerHolder(),
                  started->fetch.destinations.size());
    peer_host_fetches_.push_back(std::move(started->fetch));
    // The runtime always reports the landing (PeerFetchDone), after its
    // answers' deadlines or its quarantine: no soft deadline of its own.
    const auto hashes = request.PromptPrefixHashes(hit.num_tokens / config_.prefix_granularity);
    request.Apply(fsm::StartPrefetchEvent{std::move(started->lease),
                                          std::numeric_limits<std::int64_t>::max(),
                                          {hashes.begin(), hashes.end()},
                                          hit.num_tokens,
                                          std::move(tail)});
}

void Scheduler::preparePeer(Request& request) {
    auto* submitted = request.GetIf<fsm::Submitted>();
    if (submitted == nullptr) {
        return;
    }
    // A prompt its holder answered lands into Host before its admission; not
    // while frozen, so a flush or a weight update waits for no new landing.
    if (peer_host_landings_ && submitted->admission.peer_hit && forward_allowed_) {
        startPeerHostLanding(request);
        return;
    }
    if (submitted->admission.peer_prepared) {
        return;
    }
    submitted->admission.peer_prepared = true;
    const PeerConfig& limits = *config_.peer_config;
    const std::string& holder = request.PeerHolder();
    // The D role computes no prompt: what it lacks comes from its PD peer,
    // so it never reads from a holder, whatever the request names.
    if (holder.empty() || holder == limits.name || config_.role == Role::kD) {
        return;
    }
    const std::int32_t grain = config_.prefix_granularity;
    const std::int32_t max_tokens = cacheablePrefixTokens(request);
    const std::span<const std::string> hashes = request.PromptPrefixHashes(max_tokens / grain);
    const std::span<const std::int32_t> tokens = request.TokenSlice(0, max_tokens);
    const PrefixProbe probe =
        coordinator_.Prober().ProbePromptPrefix({.prefix_hashes = hashes,
                                                 .tokens = tokens,
                                                 .max_tokens = max_tokens,
                                                 .shared_prefix_tokens = request.SharedPrefixTokens()});
    const std::int32_t local = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    if (max_tokens - local < limits.min_fetch_tokens) {
        return;
    }
    PeerLeaseId lease{.reader = limits.name, .id = ++next_peer_lease_};
    if (grant_first_peer_) {
        // A router hint authorizes no I/O. The normal prefill queue decides whether
        // this request can own pages this round. A missing estimate uses local compute.
        const auto estimate = std::min(max_tokens - local, request.PeerExpectedTokens());
        if (estimate >= limits.min_fetch_tokens) {
            submitted->admission.peer_hit = fsm::PeerHit{.lease = lease, .num_tokens = local + estimate};
        }
        return;
    }
    peer_lookups_.push_back(PeerLookupRequest{.request_id = request.Id(),
                                              .deadline_us = Deadline(peer_now_us_, limits.lookup_timeout_us),
                                              .lease = lease,
                                              .holder = holder,
                                              .prefix_hashes = {hashes.begin(), hashes.end()},
                                              .tokens = {tokens.begin(), tokens.end()},
                                              .max_tokens = max_tokens});
    spdlog::debug("[Peer] rid={} lookup {} to {} local_tokens={} bound_tokens={}", request.Id(), lease.id, holder,
                  local, max_tokens);
    const bool grant_host = async_host_peer_ && !submitted->admission.storage_hold;
    request.Apply(fsm::StartPeerLookupEvent{lease, Deadline(peer_now_us_, limits.lookup_timeout_us)});
    if (grant_host) {
        const auto shared = request.SharedPrefixTokens();
        auto grant = std::make_shared<fsm::PeerHostGrant>();
        grant->lookup = peer_lookups_.back();
        grant->shared_prefix.assign(shared.begin(), shared.end());
        request.GetIf<fsm::PeerLooking>()->host_grant = grant;
        peer_host_grants_.emplace(lease, grant);
    }
}

void Scheduler::schedulePeerLandings(AdmissionFeedback& feedback, PlanBuild& build,
                                     std::span<Request* const> candidates, bool new_prompts_sealed) {
    // Host landings start before admission (preparePeer).
    if (!peer_exports_ || peer_host_landings_ || new_prompts_sealed) {
        return;
    }
    ProfileSpan profile_span{"Scheduler::schedulePeerLandings"};
    const std::int64_t kv_token_limit =
        config_.max_kv_page_num == 0 ? std::numeric_limits<std::int64_t>::max()
                                     : static_cast<std::int64_t>(build.kv_page_budget) * config_.prefix_granularity;
    for (Request* request : candidates) {
        if (req_pool_allocator_.AvailableSlots() == 0 || lone_requests_.ActedInPlan()) {
            return;
        }
        auto* submitted = request->GetIf<fsm::Submitted>();
        if (submitted == nullptr || !submitted->admission.peer_hit || build.Scheduled(*request)) {
            continue;
        }
        if (!prefillAllowed(*request, build)) {
            return;
        }
        fsm::PeerHit hit = *submitted->admission.peer_hit;
        std::size_t path = 0;
        double nominal_us = 0;
        unsigned allowed_sources = 0;
        bool probe = false;
        std::size_t batch_class = 0;
        double local_wait_us = 0.0;
        if (adaptive_device_peer_) {
            const auto match = matchPrefixAtAdmission(request);
            if (match.waits_for_eviction_store) continue;
            const int local = std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens);
            if (grant_first_peer_) {
                // Honor the same round's local token/slot budget. Never reserve from
                // the backlog pass after the batch filled, or wait for more pages.
                if (build.operations.size() + build.plan.peer_fetches.size() >=
                    static_cast<size_t>(config_.max_batch_size))
                    return;
                const auto budget = std::max(0, build.token_budget - build.granted_peer_tokens);
                hit.num_tokens = local + std::min(std::max(0, hit.num_tokens - local), budget);
            }
            const int extra = std::max(0, hit.num_tokens - local);
            const auto host = [](const std::string& endpoint) {
                return std::string_view(endpoint).substr(0, endpoint.rfind(':'));
            };
            const bool remote_node = host(request->PeerHolder()) != host(config_.peer_config->name);
            path = (hit.source_mask & 3) + (remote_node ? 4 : 0);
            // Cold-start estimates for Lite17B. Never use the preceding forward
            // as overlap: HBM grants are made only in this round.
            constexpr double compute_us_per_token = 80.0;
            constexpr double margin_us = 20000.0;
            const double serialization = 1.0 + device_attempts_.size();
            nominal_us = (80000.0 + extra * (remote_node ? 30.0 : 15.0)) * serialization;
            // When the batch is already full, A cannot compute until its
            // next turn. Only work actually placed in this plan can hide a read.
            const double scheduled_tokens = std::max(0, config_.max_scheduled_tokens - build.token_budget);
            const double wait_for_local = build.Full(config_.max_batch_size) && !build.operations.empty()
                                              ? scheduled_tokens * compute_us_per_token
                                              : 0.0;
            batch_class = build.Full(config_.max_batch_size) ? 2 : !build.operations.empty();
            local_wait_us = device_dispatch_wait_[batch_class].Estimate();
            if (extra < config_.peer_config->min_fetch_tokens) continue;
            const auto worth_reading = [&](std::size_t candidate_path) {
                const auto& cost = device_landing_cost_[candidate_path];
                const double wait = local_wait_us + nominal_us * cost.scale;
                const double savings = cost.success_rate * extra * compute_us_per_token;
                const auto worthwhile = [&](double estimate) {
                    return savings >
                           cost.success_rate * std::max(0.0, estimate - wait_for_local) + cost.failure_us + margin_us;
                };
                if (worthwhile(wait)) return true;
                if (worthwhile(wait * 0.25) && device_attempts_.empty() &&
                    std::hash<std::string>{}(request->Id()) % 16 == 0) {
                    probe = true;
                    return true;
                }
                return false;
            };
            bool take = false;
            if (grant_first_peer_) {
                path = remote_node ? 4 : 0;
                for (unsigned tiers : {1u, 2u, 3u}) {
                    if (worth_reading(path + tiers)) allowed_sources |= 1u << tiers;
                }
                take = allowed_sources != 0;
            } else {
                take = worth_reading(path);
            }
            if (!take) {
                // Let regular admission try. If it cannot commit, keep the
                // hit for another round; no holder is pinned while waiting.
                continue;
            }
        }
        feedback.admission_failed = false;
        const auto remaining = grant_first_peer_ ? std::max(0, build.token_budget - build.granted_peer_tokens)
                                                 : std::numeric_limits<std::int32_t>::max();
        FirstChunkOutcome outcome =
            schedulePrefillFirstChunk(build.plan, feedback, request, remaining, config_.decode_input_tokens,
                                      kv_token_limit, /*yield_tokens=*/0, /*must_complete=*/false, hit.num_tokens);
        if (!outcome.event) {
            if (feedback.admission_failed && !grant_first_peer_) {
                // It waits for room, keeping its hit; its soft pin may lapse.
                feedback.NoteCapacityBlocked(request);
                return;
            }
            // Nothing worth landing (the local hit grew, or the landing would
            // pass the KV budget): it is admitted locally.
            submitted->admission.peer_hit.reset();
            continue;
        }
        std::vector<BlockTransfer> load_pairs = outcome.event->TakeLoadPairs();
        request->Apply(std::move(*outcome.event));
        if (!load_pairs.empty()) {
            build.load_backs.push_back(startPrefixLoad(*request, std::move(load_pairs)));
        }
        build.scheduled.insert(request);

        const auto& landing = *request->GetIf<fsm::PeerLanding>();
        PeerFetchOperation fetch{.request_id = request->Id(),
                                 .lease = hit.lease,
                                 .holder = request->PeerHolder(),
                                 .begin_tokens = landing.window.begin,
                                 .end_tokens = landing.window.begin + landing.window.size,
                                 .destinations = std::move(outcome.peer_destinations)};
        spdlog::debug("[Peer] rid={} lands [{}, {}) from {} ({} blocks)", request->Id(), fetch.begin_tokens,
                      fetch.end_tokens, fetch.holder, fetch.destinations.size());
        if (adaptive_device_peer_) {
            spdlog::info("[PeerChoice] rid={} read extra={} path={} nominal_us={:.0f} local_wait_us={:.0f} reason={}",
                         request->Id(), hit.num_tokens - fetch.begin_tokens, path, nominal_us, local_wait_us,
                         probe ? "probe" : "score");
            device_attempts_.emplace(request->Id(), DeviceAttempt{path, nominal_us, batch_class, hit.num_tokens});
        }
        if (grant_first_peer_) {
            build.granted_peer_tokens += fetch.end_tokens - fetch.begin_tokens;
            const auto hashes = request->PromptPrefixHashes(fetch.end_tokens / config_.prefix_granularity);
            const auto tokens = request->TokenSlice(0, fetch.end_tokens);
            peer_lookups_.push_back(
                PeerLookupRequest{.request_id = request->Id(),
                                  .deadline_us = Deadline(peer_now_us_, config_.peer_config->lookup_timeout_us),
                                  .lease = fetch.lease,
                                  .holder = fetch.holder,
                                  .prefix_hashes = {hashes.begin(), hashes.end()},
                                  .tokens = {tokens.begin(), tokens.end()},
                                  .max_tokens = fetch.end_tokens,
                                  .allowed_sources = allowed_sources});
        }
        if (!hit.checkpoint.empty()) {
            fetch.checkpoint = std::move(hit.checkpoint);
            const auto hashes = request->PromptPrefixHashes(fetch.end_tokens / config_.prefix_granularity);
            const auto tokens = request->TokenSlice(0, fetch.end_tokens);
            fetch.prefix_hashes.assign(hashes.begin(), hashes.end());
            fetch.tokens.assign(tokens.begin(), tokens.end());
        }
        build.plan.peer_fetches.push_back(std::move(fetch));
    }
}

}  // namespace tokenspeed
