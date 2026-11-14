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
#include <span>
#include <string>

#include "cache/prefix/prefix_hasher.h"

namespace tokenspeed {

// A page's hash under a prior given as text, returned as text: one page of
// the chains ComputePrefixHashes and ExtendPrefixHashes build. Production
// hashes a single page under a ContentHash prior; tests build their
// expected text chains, and pin the hash framing, page by page with this.
inline std::string HashPrefixPage(std::span<const std::int32_t> tokens, const std::string& prior_hash,
                                  std::span<const std::string> extra_keys = {}) {
    const auto prior_bytes = HexToBytes(prior_hash);
    return DigestToHex(HashPrefixPageDigest(tokens, prior_bytes, extra_keys));
}

}  // namespace tokenspeed
