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
#include <string>
#include <variant>
#include <vector>

#include "cache/core/cache_types.h"

namespace tokenspeed {

// A rank's shared L2 pool as its Host L2: a proxy Host tier mirrors the pool's hits,
// and admission, load-back and the junction run the Host L2 code unchanged.
struct SharedL2Config {
    std::int32_t max_writes;
    std::int32_t max_write_keys;
    void Validate() const;
};

// A published State checkpoint's lineage, recorded by the pool's checkpoint
// ancestry once its blocks are stored.
struct SharedL2Publication {
    std::vector<std::int32_t> tokens;
    std::vector<ContentHash> prefix_hashes;
    std::optional<TailPublication> tail;
};

// A waiting prompt's question to the pool: its full-page hashes and tokens up
// to max_tokens (tails are hashed from them).
struct SharedL2Lookup {
    std::string request_id;
    std::vector<ContentHash> prefix_hashes;
    std::vector<std::int32_t> tokens;
    std::int32_t max_tokens;
};

// The pool's longest reusable prefix, pinned under ticket (0 pins nothing);
// kv_tokens is how far its KV reaches, past the checkpoint num_tokens resumes from.
struct SharedL2LookupResult {
    std::string request_id;
    std::uint64_t ticket;
    std::int32_t num_tokens;
    std::int32_t kv_tokens;
    std::optional<TailPublication> tail;
    std::vector<CacheKey> keys;
    std::vector<std::int32_t> pages;
};

// A proxy Host block and the pool page whose bytes it stands for.
struct SharedL2ProxyPage {
    std::uint32_t group_id;
    std::int32_t block;
    std::int32_t page;
};

// Releases a lookup's pins; no completion.
struct SharedL2ReleaseOperation {
    std::uint64_t ticket;
};

// Reserves pool pages, copies Device blocks into them and commits them with
// the lineage of the checkpoints among them.
struct SharedL2WriteOperation {
    std::uint64_t op_id;
    std::vector<CacheKey> keys;
    std::vector<std::int32_t> blocks;
    std::vector<SharedL2Publication> publications;
};

// Entries Device hits used; no completion.
struct SharedL2TouchOperation {
    std::vector<CacheKey> keys;
};

using SharedL2Operation = std::variant<SharedL2ReleaseOperation, SharedL2WriteOperation, SharedL2TouchOperation>;

namespace cache {

struct SharedL2WriteDone {
    std::uint64_t op_id;
};

}  // namespace cache
}  // namespace tokenspeed
