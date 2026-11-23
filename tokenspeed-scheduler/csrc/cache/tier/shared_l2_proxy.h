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
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/cache_block_ref.h"
#include "cache/tier/shared_l2.h"

namespace tokenspeed {

// A rank's proxy of a shared L2 pool: the Host entries that stand for the
// pool pages its lookups pinned, and the operations that tell the owner what
// the rank no longer needs. All methods run on the scheduler thread. Writes
// to the pool are Device-to-Host stores and belong to TierTransferManager.
class SharedL2Proxy {
public:
    explicit SharedL2Proxy(CacheCoordinator& coordinator);

    // Mirrors the pool's answer to a waiting prompt's lookup into proxy Host
    // entries; tail_prior precedes the hit's tail. hit_blocks takes the
    // references that pin the hit's entries: dropping them lets the entries
    // go. Returns the entries created -- a key another lookup's proxy already
    // stands for is shared, and stays under that lookup's ticket.
    std::vector<SharedL2ProxyBlock> MirrorAnswer(const SharedL2Answer& answer, const ContentHash& tail_prior,
                                                 std::vector<CacheBlockRef>& hit_blocks);
    // This plan's operations for the owner: one touch for the keys Device
    // hits used since the last plan (none without any), and, after evicting
    // the proxy entries nothing holds any more -- no waiting prompt, no
    // load-back --, one release for each ticket left without an entry. A
    // ticket's entries go from the answer's last one back, up to the first
    // one still held.
    std::vector<SharedL2Operation> NextOperations(std::vector<CacheKey> device_hits);

private:
    // The proxy entries created under one owner ticket, in the answer's
    // order; the ticket is released once the last one is dropped. A ticket
    // whose every page stands under earlier tickets has none.
    struct TicketEntries {
        std::uint64_t ticket{0};
        std::vector<CacheKey> keys;
    };

    CacheCoordinator& coordinator_;
    std::vector<TicketEntries> tickets_;
};

}  // namespace tokenspeed
