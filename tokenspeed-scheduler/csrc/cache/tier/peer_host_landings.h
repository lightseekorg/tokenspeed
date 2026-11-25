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

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/peer.h"
#include "cache/tier/storage_lease.h"

namespace tokenspeed {

// A reader's peer landings into Host (PeerLandingTier::kHost,
// docs/rfcs/0006-disk-tier.md): before its admission, a prompt the holder
// answered for reads the holder's pages into Host blocks reserved here, as
// an L3 GET's destinations, and waits in Prefetching. Once every member read
// them and the holder vouched for them (PeerFetchDone at the landing's end),
// they are published as Host entries the request holds as its Ready hold
// until its admission takes them as an ordinary Host hit. Anything else
// returns the blocks unpublished, and the request is admitted from its local
// hit. All methods run on the scheduler thread.
class PeerHostLandings : public StorageLeaseIssuer {
public:
    PeerHostLandings(CacheCoordinator& coordinator, std::int32_t max_host_blocks);
    PeerHostLandings(const PeerHostLandings&) = delete;
    PeerHostLandings& operator=(const PeerHostLandings&) = delete;

    struct Started {
        StorageLease lease;
        // The landing for the plan: the reader's Host destinations.
        PeerFetchOperation fetch;
    };
    // Reserves Host blocks for the keys of `prompt` (bounded at the holder's
    // answered boundary) that the reader's Host lacks above its Device hit,
    // and starts the landing of `request_id` from `holder` under `lease`.
    // Nothing when the reservation cannot resume the prompt at exactly that
    // boundary within the credits: a partial landing is never kept.
    std::optional<Started> Start(const PromptPrefix& prompt, const std::string& request_id, const PeerLeaseId& lease,
                                 const std::string& holder, std::int32_t local_tokens, bool* retry = nullptr);
    // Whether the landing finished; an unknown one counts as finished.
    bool ContainsRequest(const std::string& rid) const { return by_request_.contains(rid); }
    bool Completed(std::uint64_t id) const;
    // The runtime ended `request_id`'s landing at `num_tokens`: at its end,
    // every member read every page under a pin the holder vouched for, and
    // the pages are published. Returns the tokens landed past the local hit,
    // 0 when the landing fell short, or nothing when no landing into Host
    // has that request.
    std::optional<std::int32_t> Finish(const std::string& request_id, std::int32_t num_tokens);

    // Landings in flight or held for their requests' admissions.
    std::size_t NumLandings() const { return landings_.size(); }
    // Whether a landing is still in flight.
    bool HasProgressPending() const;
    std::size_t NumHeldHostBlocks() const;

private:
    struct Landing {
        std::string request_id;
        std::int32_t begin_tokens{0};
        std::int32_t end_tokens{0};
        std::vector<CacheKey> keys;
        std::vector<CacheBlockRef> host_blocks;
        ContentHash tail_prior;
        std::optional<TailPublication> tail;
        bool detached{false};
        bool completed{false};
    };

    std::uint64_t nextId();
    // A landing whose request let go keeps its blocks until the runtime
    // reports it: a read may still be writing them.
    void detach(std::uint64_t id) noexcept override;

    CacheCoordinator& coordinator_;
    const std::int32_t max_host_blocks_;
    std::map<std::uint64_t, Landing> landings_;
    std::unordered_map<std::string, std::uint64_t> by_request_;
    std::uint64_t next_id_{1};
};

}  // namespace tokenspeed
