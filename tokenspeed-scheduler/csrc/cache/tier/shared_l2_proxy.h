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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/shared_l2.h"
#include "cache/tier/shared_l2_lookup_hold.h"

namespace tokenspeed {

// A rank's proxy of a shared L2 pool: the Host entries that stand for the
// pool pages its lookups pinned, and the commands that tell the owner what
// the rank no longer needs. All methods run on the scheduler thread. Writes
// to the pool are Device-to-Host stores and belong to TierTransferManager.
class SharedL2Proxy {
public:
    explicit SharedL2Proxy(CacheCoordinator& coordinator);

    // Starts the lookup of a waiting prompt; the runtime sends it and
    // answers it under the hold's id.
    SharedL2LookupHold StartLookup(const std::string& request_id);
    // The request a lookup still waits for; null once it was answered,
    // abandoned or its request left.
    const std::string* PendingRequest(std::uint64_t lookup_id) const;
    // Mirrors the pool's answer to hold's lookup into proxy Host entries,
    // which the hold pins; tail_prior precedes the hit's tail. Returns the
    // entries created -- a key another lookup's proxy already stands for is
    // shared, and stays under that lookup's ticket.
    std::vector<SharedL2ProxyPage> ApplyAnswer(const SharedL2LookupResult& result, const ContentHash& tail_prior,
                                               SharedL2LookupHold& hold);
    // An answer no lookup waits for: what it pinned is released.
    void DropAnswer(const SharedL2LookupResult& result);
    // The keys Device hits used since the last call go to the owner.
    void Touch(std::vector<CacheKey> keys);
    // The runtime's lease on what the lookups pinned at the owner, as it last
    // reported it. While it is lost the owner may have dropped the pins, so no
    // load may start from a proxy entry: admission probes the Device alone
    // (Scheduler::matchPrefixAtAdmission).
    void SetLeaseLost(bool lost) { lease_lost_ = lost; }
    bool LeaseLost() const { return lease_lost_; }
    // Drops proxy entries nothing holds any more -- no waiting prompt, no
    // load-back -- and releases each ticket left without one. A ticket's
    // entries go from the answer's last one back, up to the first one still
    // held; while the lease is lost, every entry nothing holds goes.
    void ReleaseUnheld();
    std::vector<SharedL2Operation> TakeCommands() { return std::exchange(commands_, {}); }

    // Cumulative counts and the pending lookups and unreleased tickets;
    // the caller fills in the writes.
    SharedL2Stats Stats() const;

private:
    friend class SharedL2LookupHold;

    // A hold ended: a lookup still pending is abandoned.
    void endLookup(std::uint64_t lookup_id) noexcept;

    // The proxy entries created under one owner ticket, in the answer's
    // order; the ticket is released once the last one is dropped.
    struct TicketEntries {
        std::uint64_t id{0};
        std::vector<CacheKey> keys;
    };

    CacheCoordinator& coordinator_;
    // Lookups not answered yet, with their requests; only a hold adds and
    // removes them.
    std::unordered_map<std::uint64_t, std::string> pending_;
    std::vector<TicketEntries> tickets_;
    std::vector<SharedL2Operation> commands_;
    bool lease_lost_{false};
    std::uint64_t next_lookup_id_{0};
    std::uint64_t num_lookups_{0};
    std::uint64_t num_hits_{0};
    std::uint64_t num_abandoned_{0};
    std::uint64_t num_proxy_overflows_{0};
};

}  // namespace tokenspeed
