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
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <openssl/sha.h>

#include "cache/core/content_hash.h"
#include "cache/core/hex_codec.h"
#include "utils.h"

namespace tokenspeed {

// Append each byte of [bytes, bytes+n) to out as two lowercase hex characters.
inline void AppendHexBytes(std::string& out, const uint8_t* bytes, std::size_t n) {
    const std::size_t size = out.size();
    out.resize(size + 2 * n);
    WriteLowercaseHex({bytes, n}, out.data() + size);
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
    std::string out(SHA256_DIGEST_LENGTH * 2, '0');
    WriteLowercaseHex({digest, SHA256_DIGEST_LENGTH}, out.data());
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
using PrefixDigest = std::array<unsigned char, SHA256_DIGEST_LENGTH>;

inline PrefixDigest HashPrefixPageDigest(std::span<const std::int32_t> tokens, std::span<const uint8_t> prior_bytes,
                                         std::span<const std::string> extra_keys) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);

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

    PrefixDigest digest;
    SHA256_Final(digest.data(), &ctx);
    return digest;
}

inline std::string HashPrefixPage(std::span<const std::int32_t> tokens, const std::string& prior_hash,
                                  std::span<const std::string> extra_keys = {}) {
    const auto prior_bytes = HexToBytes(prior_hash);
    return DigestToHex(HashPrefixPageDigest(tokens, prior_bytes, extra_keys).data());
}

// The same page hash under a prior held as an identity, returned as one: it
// chains exactly the bytes HashPrefixPage(tokens, prior.ToString()) decodes,
// taking a canonical prior's 32 bytes directly instead of through its text.
inline ContentHash HashPrefixPage(std::span<const std::int32_t> tokens, const ContentHash& prior) {
    if (const ContentHash::Digest* digest = prior.digest()) {
        return ContentHash{HashPrefixPageDigest(tokens, *digest, {})};
    }
    return ContentHash{HashPrefixPageDigest(tokens, HexToBytes(prior.ToString()), {})};
}

// Appends the hashes of prefix_pages to hashes, chained on the raw seed
// bytes. Later pages chain the raw digest; an empty seed remains distinct
// from a 32-byte all-zero digest.
inline void AppendPrefixHashChain(std::vector<std::string>& hashes,
                                  std::span<const std::span<const std::int32_t>> prefix_pages,
                                  std::span<const uint8_t> seed,
                                  std::span<const std::span<const std::string>> extra_keys_per_page) {
    // Grow as a range insert would: a chain may also be extended a few pages
    // at a time.
    const std::size_t size = hashes.size() + prefix_pages.size();
    if (size > hashes.capacity()) hashes.reserve(std::max(size, 2 * hashes.size()));
    std::span<const uint8_t> current_prior = seed;
    PrefixDigest digest;
    for (std::size_t i = 0; i < prefix_pages.size(); ++i) {
        std::span<const std::string> extra =
            (i < extra_keys_per_page.size()) ? extra_keys_per_page[i] : std::span<const std::string>{};
        digest = HashPrefixPageDigest(prefix_pages[i], current_prior, extra);
        hashes.push_back(DigestToHex(digest.data()));
        current_prior = digest;
    }
}

inline std::vector<std::string> ComputePrefixHashes(
    std::span<const std::span<const std::int32_t>> prefix_pages, const std::string& prior,
    std::span<const std::span<const std::string>> extra_keys_per_page = {}) {
    std::vector<std::string> hashes;
    if (prefix_pages.empty()) return hashes;
    // Decode the caller's seed once.
    const auto seed = HexToBytes(prior);
    AppendPrefixHashChain(hashes, prefix_pages, seed, extra_keys_per_page);
    return hashes;
}

// Extends hashes, the chain of a token sequence's leading pages, in place by
// next_pages, the pages that follow them. A throw leaves a valid, shorter
// chain (basic guarantee).
inline void ExtendPrefixHashes(std::vector<std::string>& hashes,
                               std::span<const std::span<const std::int32_t>> next_pages) {
    if (next_pages.empty()) return;
    // Decoded before hashes grows: the seed is its own last element.
    const auto seed = hashes.empty() ? std::vector<uint8_t>{} : HexToBytes(hashes.back());
    AppendPrefixHashChain(hashes, next_pages, seed, {});
}

}  // namespace tokenspeed
