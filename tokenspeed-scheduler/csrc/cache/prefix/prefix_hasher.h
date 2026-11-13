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

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <openssl/sha.h>

#include "core/profile_span.h"
#include "utils.h"

namespace tokenspeed {

// Append each byte of [bytes, bytes+n) to out as two lowercase hex characters.
inline void AppendHexBytes(std::string& out, const uint8_t* bytes, std::size_t n) {
    static constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(kHex[bytes[i] >> 4]);
        out.push_back(kHex[bytes[i] & 0xf]);
    }
}

// Absorb a uint32_t into the hash as 4 little-endian bytes. Encodes each token id
// and the count/length prefixes that frame extra_keys.
inline void Sha256UpdateU32LE(SHA256_CTX& ctx, uint32_t v) {
    uint8_t buf[4] = {
        static_cast<uint8_t>(v),
        static_cast<uint8_t>(v >> 8),
        static_cast<uint8_t>(v >> 16),
        static_cast<uint8_t>(v >> 24),
    };
    SHA256_Update(&ctx, buf, 4);
}

// Feed token IDs as little-endian bytes using bounded stack storage.
inline void Sha256UpdateTokensExplicitLE(SHA256_CTX& ctx, std::span<const std::int32_t> tokens) {
    constexpr std::size_t kChunkTokens = 64;
    uint8_t buffer[kChunkTokens * sizeof(std::uint32_t)];
    for (std::size_t start = 0; start < tokens.size();) {
        const std::size_t count = std::min<std::size_t>(kChunkTokens, tokens.size() - start);
        for (std::size_t i = 0; i < count; ++i) {
            const auto token = static_cast<std::uint32_t>(tokens[start + i]);
            buffer[4 * i] = static_cast<uint8_t>(token);
            buffer[4 * i + 1] = static_cast<uint8_t>(token >> 8);
            buffer[4 * i + 2] = static_cast<uint8_t>(token >> 16);
            buffer[4 * i + 3] = static_cast<uint8_t>(token >> 24);
        }
        SHA256_Update(&ctx, buffer, count * sizeof(std::uint32_t));
        start += count;
    }
}

// Encode a 32-byte SHA-256 digest as a 64-char lowercase hex string.
inline std::string DigestToHex(const unsigned char* digest) {
    std::string out;
    out.reserve(SHA256_DIGEST_LENGTH * 2);
    AppendHexBytes(out, digest, SHA256_DIGEST_LENGTH);
    return out;
}

// Decode a hex string back into its raw bytes (inverse of DigestToHex).
inline std::vector<uint8_t> HexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        uint8_t hi = (hex[i] >= 'a')   ? static_cast<uint8_t>(hex[i] - 'a' + 10)
                     : (hex[i] >= 'A') ? static_cast<uint8_t>(hex[i] - 'A' + 10)
                                       : static_cast<uint8_t>(hex[i] - '0');
        uint8_t lo = (hex[i + 1] >= 'a')   ? static_cast<uint8_t>(hex[i + 1] - 'a' + 10)
                     : (hex[i + 1] >= 'A') ? static_cast<uint8_t>(hex[i + 1] - 'A' + 10)
                                           : static_cast<uint8_t>(hex[i + 1] - '0');
        bytes.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return bytes;
}

// extra_keys: per-page list of distinguishing keys (e.g. LoRA name, cache salt);
// the caller encodes each value, this function owns only the framing.
//
// The whole input is prefix-framed -- [prior_len][prior][token_count][tokens]
// [extra_count][extra...] -- so every section is self-delimiting and no two
// distinct (prior, tokens, extra_keys) triples can hash the same byte stream:
// prior_len separates page 0 from a chained page, token_count keeps tokens from
// bleeding into extra_keys, and per-key length prefixes prevent re-splitting.
// Feed order is prior_hash -> tokens -> extra_keys.
inline std::string HashPrefixPage(std::span<const std::int32_t> tokens, const std::string& prior_hash,
                                  std::span<const std::string> extra_keys = {}) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    std::vector<uint8_t> prior_bytes = HexToBytes(prior_hash);
    Sha256UpdateU32LE(ctx, static_cast<uint32_t>(prior_bytes.size()));
    if (!prior_bytes.empty()) {
        SHA256_Update(&ctx, prior_bytes.data(), prior_bytes.size());
    }

    Sha256UpdateU32LE(ctx, static_cast<uint32_t>(tokens.size()));
    if (!tokens.empty()) {
        if constexpr (std::endian::native == std::endian::little) {
            SHA256_Update(&ctx, tokens.data(), tokens.size_bytes());
        } else {
            Sha256UpdateTokensExplicitLE(ctx, tokens);
        }
    }

    // extra_keys is the terminal block, so an empty list can be skipped without
    // ambiguity (a non-empty list always writes a count >= 1 first).
    if (!extra_keys.empty()) {
        Sha256UpdateU32LE(ctx, static_cast<uint32_t>(extra_keys.size()));
        for (const std::string& key : extra_keys) {
            Sha256UpdateU32LE(ctx, static_cast<uint32_t>(key.size()));
            SHA256_Update(&ctx, key.data(), key.size());
        }
    }

    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &ctx);
    return DigestToHex(digest);
}

inline std::vector<std::string> ComputePrefixHashes(
    std::span<const std::span<const std::int32_t>> prefix_pages, const std::string& prior,
    std::span<const std::span<const std::string>> extra_keys_per_page = {}) {
    ProfileSpan profile_span{"ComputePrefixHashes"};
    std::vector<std::string> hashes;
    hashes.reserve(prefix_pages.size());
    std::string current_prior = prior;
    for (std::size_t i = 0; i < prefix_pages.size(); ++i) {
        std::span<const std::string> extra =
            (i < extra_keys_per_page.size()) ? extra_keys_per_page[i] : std::span<const std::string>{};
        std::string h = HashPrefixPage(prefix_pages[i], current_prior, extra);
        hashes.push_back(h);
        current_prior = h;
    }
    return hashes;
}

}  // namespace tokenspeed
