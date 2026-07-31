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

#include <stdexcept>
#include <utility>
#include <vector>

#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "fsm/pd_events.h"
#include "scheduler/outside_events/inc.h"
#include "utils.h"

namespace tokenspeed {

namespace {

void FreeAll(std::vector<CacheBlockRef>&& block_refs) {
    for (auto it = block_refs.rbegin(); it != block_refs.rend(); ++it) {
        it->reset();
    }
}

}  // namespace

void Scheduler::handleEvent(const cache::PrefetchDone&) {
    // Legacy storage-prefetch acknowledgements are irrelevant to the
    // coordinator cache and intentionally have no state transition.
}

void Scheduler::handleEvent(const pd::BootstrappedEvent& event) {
    Request* request = find_request(event.request_id);
    if (request != nullptr && request->Is<fsm::Bootstrapping>()) {
        request->Apply(fsm::BootstrappedEvent{});
    }
}

void Scheduler::handleEvent(const pd::FailedEvent& event) {
    Request* request = find_request(event.request_id);
    if (request == nullptr || request->Is<fsm::Finished>()) {
        return;
    }
    pending_forward_results_.erase(event.request_id);
    pd_transfer_pins_.erase(event.request_id);
    request->Apply(fsm::AbortEvent{&coordinator_});
}

void Scheduler::handleEvent(const pd::SucceededEvent& event) {
    Request* request = find_request(event.request_id);
    if (request == nullptr || request->Is<fsm::Finished>()) {
        return;
    }
    if (!request->Is<fsm::PrefillDone>() && !request->Is<fsm::Decoding>()) {
        throw std::logic_error("PD SucceededEvent received in state " +
                               request->StateName());
    }
    pending_forward_results_.erase(event.request_id);
    pd_transfer_pins_.erase(event.request_id);
    request->Apply(fsm::FinishEvent{&coordinator_});
}

void Scheduler::handleEvent(const pd::RemotePrefillDoneEvent& event) {
    Request* request = find_request(event.request_id);
    if (request == nullptr) {
        return;
    }
    if (request->Is<fsm::Prefilling>()) {
        if (event.bootstrap_token < 0) {
            throw std::invalid_argument(
                "PD RemotePrefillDoneEvent requires a non-negative bootstrap token");
        }
        pd_transfer_pins_.erase(event.request_id);
        request->Apply(fsm::RemotePrefillDoneEvent{event.bootstrap_token});
        return;
    }
    if (request->Is<fsm::PrefillDone>() ||
        request->Is<fsm::Decoding>() ||
        request->Is<fsm::Finished>()) {
        return;
    }
    throw std::logic_error(
        "PD RemotePrefillDoneEvent received before destination admission; state=" +
        request->StateName());
}

void Scheduler::handleEvent(const forward::Finish& event) {
    if (config_.enable_pd_cache &&
        pd_transfer_pins_.contains(event.request_id)) {
        throw std::logic_error(
            "PD Finish received while transfer pages are pinned");
    }
    pending_forward_results_.erase(event.request_id);
    if (Request* request = find_request(event.request_id)) {
        request->Apply(fsm::FinishEvent{&coordinator_});
    }
}

void Scheduler::handleEvent(const forward::UpdateReserveNumTokens& event) {
    if (Request* request = find_request(event.request_id)) {
        request->Apply(fsm::UpdateReserveNumTokensEvent{
            event.reserve_num_tokens_in_next_schedule_event});
    }
}

void Scheduler::handleEvent(const forward::ExtendResult& event) {
    if (auto it = pending_forward_results_.find(event.request_id);
        it != pending_forward_results_.end() && --it->second <= 0) {
        pending_forward_results_.erase(it);
    }
    if (Request* request = find_request(event.request_id)) {
        request->Apply(fsm::ExtendResultEvent{event.tokens});
    }
}

void Scheduler::handleEvent(const forward::Abort& event) {
    pending_forward_results_.erase(event.request_id);
    pd_transfer_pins_.erase(event.request_id);
    if (Request* request = find_request(event.request_id)) {
        request->Apply(fsm::AbortEvent{&coordinator_});
    }
}

void Scheduler::handleEvent(const cache::WriteBackDone& event) {
    std::vector<StoreTicket> tickets = store_ops_.Retire(event.op_id);
    for (StoreTicket& ticket : tickets) {
        if (event.success) {
            coordinator_.CacheHostBlock(ticket.host_block_ref, ticket.key);
        }
    }
    for (auto it = tickets.rbegin(); it != tickets.rend(); ++it) {
        it->device_block_ref.reset();
        it->host_block_ref.reset();
    }
}

void Scheduler::handleEvent(const cache::LoadBackDone& event) {
    auto it = load_ops_.find(event.op_id);
    if (it == load_ops_.end()) {
        return;
    }
    _assert(event.success, "host loadback failed: host bytes integrity");
    FreeAll(std::move(it->second.host_pins));
    FreeAll(std::move(it->second.device_blocks));
    load_ops_.erase(it);
}

}  // namespace tokenspeed
