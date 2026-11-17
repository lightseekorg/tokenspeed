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
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "cache/core/cache_types.h"

namespace tokenspeed {

// A rank whose Host L2 is a pool shared with other ranks: the pool's owner
// process keeps the index and its provider processes the bytes, and the
// rank's Host tier is a proxy of the pool -- entries that stand for the pool
// pages a lookup pinned, with no bytes of their own. Admission, junction cuts
// and load-back then run the Host L2 code unchanged; the runtime answers the
// rank's lookups from the owner and reads and writes the pool's pages.
struct SharedL2Config {
    // The pool writes in flight at once and the blocks one write carries:
    // together, the Device blocks pool writes pin. Publications beyond them
    // wait in the store queue, pinning nothing.
    std::int32_t max_writes;
    std::int32_t max_write_keys;
    void Validate() const {
        if (max_writes <= 0 || max_write_keys <= 0) {
            throw std::invalid_argument("shared L2 needs positive write limits");
        }
    }
};

// A State checkpoint's lineage, which the pool's owner records with the
// checkpoint's State block: its tokens, the chain hashes of its full pages
// and its tail. `publisher` is the access epoch of the request that published
// it on this rank.
struct SharedL2Publication {
    std::vector<std::int32_t> tokens;
    std::vector<std::string> prefix_hashes;
    std::optional<TailPublication> tail;
    std::uint64_t publisher{0};

    bool operator==(const SharedL2Publication&) const = default;
};

// A waiting prompt's question to the pool: its full-page hashes and its
// tokens up to max_tokens (the owner hashes tails from them).
struct SharedL2Lookup {
    std::uint64_t lookup_id;
    std::string request_id;
    std::vector<std::string> prefix_hashes;
    std::vector<std::int32_t> tokens;
    std::int32_t max_tokens;
};

// The pool's answer: its longest reusable prefix of the prompt, pinned under
// `ticket` (0 pins nothing), each key with its bucket and its pool page.
// kv_tokens is how far the pool's KV reaches, past the checkpoint num_tokens
// resumes from. A miss is an answer that names no key.
struct SharedL2LookupResult {
    std::uint64_t lookup_id;
    std::uint64_t ticket;
    std::int32_t num_tokens;
    std::int32_t kv_tokens;
    std::optional<TailPublication> tail;
    std::vector<CacheKey> keys;
    std::vector<std::int32_t> buckets;
    std::vector<std::int32_t> pages;
};

// A proxy Host block and the pool page whose bytes it stands for: a
// load-back from the block reads the page.
struct SharedL2ProxyPage {
    std::uint32_t group_id;
    std::int32_t block;
    std::int32_t bucket;
    std::int32_t page;
};

// Copies Device blocks into pool pages the owner grants and commits them
// with the lineage of the checkpoints among them. Its Device blocks stay
// pinned until the runtime reports cache::SharedL2WriteDone.
struct SharedL2WriteOperation {
    std::uint64_t write_id;
    std::vector<CacheKey> keys;
    std::vector<std::int32_t> buckets;
    std::vector<std::int32_t> blocks;
    std::vector<SharedL2Publication> publications;

    bool operator==(const SharedL2WriteOperation&) const = default;
};

// Releases a lookup's pins at the owner; no completion.
struct SharedL2ReleaseOperation {
    std::uint64_t ticket;

    bool operator==(const SharedL2ReleaseOperation&) const = default;
};

// The pool's copies of entries Device hits used, for the owner to refresh;
// no completion.
struct SharedL2TouchOperation {
    std::vector<CacheKey> keys;

    bool operator==(const SharedL2TouchOperation&) const = default;
};

using SharedL2Operation = std::variant<SharedL2WriteOperation, SharedL2ReleaseOperation, SharedL2TouchOperation>;

// Scheduler-owned observations of a shared L2 rank, cumulative unless noted.
struct SharedL2Stats {
    std::uint64_t lookups{0};
    // Answers that pinned something at the owner, whether or not the prompt
    // then loaded from it: the proxy overflows are among them.
    std::uint64_t hits{0};
    // Answers the proxy Host tier had no room to mirror, taken as misses.
    std::uint64_t proxy_overflows{0};
    // Now: owner tickets not released, writes not completed.
    std::uint64_t tickets{0};
    std::uint64_t writes_in_flight{0};
};

namespace cache {

// The runtime committed a pool write: the write's Device blocks are
// released. A write that fails stops the runtime instead.
struct SharedL2WriteDone {
    std::uint64_t write_id;
};

}  // namespace cache
}  // namespace tokenspeed
