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

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <openssl/sha.h>

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

// Encode a 32-byte SHA-256 digest as a 64-char lowercase hex string.
inline std::string DigestToHex(const unsigned char* digest) {
    std::string out;
    out.reserve(SHA256_DIGEST_LENGTH * 2);
    AppendHexBytes(out, digest, SHA256_DIGEST_LENGTH);
    return out;
}

// Decode a hex string into caller-owned storage. HashPageSegments uses this
// overload for its fixed-size prior digest so chaining pages never allocates a
// temporary byte vector.
inline std::size_t HexToBytes(std::string_view hex, std::span<uint8_t> bytes) {
    const std::size_t byte_count = hex.size() / 2;
    _assert(byte_count <= bytes.size(), "hex output does not fit caller-owned storage");
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        uint8_t hi = (hex[i] >= 'a')   ? static_cast<uint8_t>(hex[i] - 'a' + 10)
                     : (hex[i] >= 'A') ? static_cast<uint8_t>(hex[i] - 'A' + 10)
                                       : static_cast<uint8_t>(hex[i] - '0');
        uint8_t lo = (hex[i + 1] >= 'a')   ? static_cast<uint8_t>(hex[i + 1] - 'a' + 10)
                     : (hex[i + 1] >= 'A') ? static_cast<uint8_t>(hex[i + 1] - 'A' + 10)
                                           : static_cast<uint8_t>(hex[i + 1] - '0');
        bytes[i / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return byte_count;
}

// Decode a hex string back into its raw bytes (inverse of DigestToHex).
inline std::vector<uint8_t> HexToBytes(std::string_view hex) {
    std::vector<uint8_t> bytes(hex.size() / 2);
    HexToBytes(hex, bytes);
    return bytes;
}

// Hash one logical page whose tokens may straddle two stable backing spans.
// The split is deliberately not represented in the byte stream: token_count is
// the sum of both spans and tokens are absorbed first->second, exactly as if the
// caller had copied them into one contiguous page. This lets completion hashing
// join stable request tokens with newly returned tokens without an O(tokens)
// staging copy.
//
// extra_keys: per-page list of distinguishing keys (e.g. LoRA name, cache salt);
// the caller encodes each value, this function owns only the framing.
//
// The whole input is prefix-framed -- [prior_len][prior][token_count][tokens]
// [extra_count][extra...] -- so every section is self-delimiting and no two
// distinct (prior, tokens, extra_keys) triples can hash the same byte stream:
// prior_len separates page 0 from a chained page, token_count keeps tokens from
// bleeding into extra_keys, and per-key length prefixes prevent re-splitting.
// Feed order is prior_hash -> tokens -> extra_keys.
inline std::string HashPageSegments(std::span<const std::int32_t> first_tokens,
                                    std::span<const std::int32_t> second_tokens, const std::string& prior_hash,
                                    std::span<const std::string> extra_keys = {}) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    std::array<uint8_t, SHA256_DIGEST_LENGTH> inline_prior_bytes{};
    std::vector<uint8_t> overflow_prior_bytes;
    const std::size_t prior_byte_count = prior_hash.size() / 2;
    std::span<uint8_t> prior_bytes;
    if (prior_byte_count <= inline_prior_bytes.size()) {
        prior_bytes = std::span<uint8_t>{inline_prior_bytes}.first(prior_byte_count);
    } else {
        overflow_prior_bytes.resize(prior_byte_count);
        prior_bytes = overflow_prior_bytes;
    }
    HexToBytes(prior_hash, prior_bytes);
    Sha256UpdateU32LE(ctx, static_cast<uint32_t>(prior_byte_count));
    if (prior_byte_count != 0) {
        SHA256_Update(&ctx, prior_bytes.data(), prior_byte_count);
    }

    Sha256UpdateU32LE(ctx, static_cast<uint32_t>(first_tokens.size() + second_tokens.size()));
    for (std::int32_t t : first_tokens) {
        Sha256UpdateU32LE(ctx, static_cast<uint32_t>(t));
    }
    for (std::int32_t t : second_tokens) {
        Sha256UpdateU32LE(ctx, static_cast<uint32_t>(t));
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

inline std::string HashPage(std::span<const std::int32_t> tokens, const std::string& prior_hash,
                            std::span<const std::string> extra_keys = {}) {
    return HashPageSegments(tokens, {}, prior_hash, extra_keys);
}

inline std::vector<std::string> ComputePagedHashes(
    const std::vector<std::span<const std::int32_t>>& paged_tokens, const std::string& prior,
    const std::vector<std::span<const std::string>>& extra_keys_per_page = {}) {
    std::vector<std::string> hashes;
    hashes.reserve(paged_tokens.size());
    std::string current_prior = prior;
    for (std::size_t i = 0; i < paged_tokens.size(); ++i) {
        std::span<const std::string> extra =
            (i < extra_keys_per_page.size()) ? extra_keys_per_page[i] : std::span<const std::string>{};
        std::string h = HashPage(paged_tokens[i], current_prior, extra);
        hashes.push_back(h);
        current_prior = h;
    }
    return hashes;
}

// Content hashes for the full pages covered by the processed window
// [0, window_begin + window_size), truncating any tail page past the window.
// paged_tokens holds the request's full pages (partial tail already excluded).
inline std::vector<std::string> FlatWindowPageHashes(std::vector<std::span<const std::int32_t>> paged_tokens,
                                                     std::int32_t page_size, std::int32_t window_begin,
                                                     std::int32_t window_size) {
    const std::int32_t end_of_window_pages = (window_begin + window_size) / page_size;
    if (end_of_window_pages < static_cast<std::int32_t>(paged_tokens.size())) {
        paged_tokens.resize(end_of_window_pages);
    }
    return ComputePagedHashes(paged_tokens, "");
}

// ---- group_id packing / unpacking ----
// group_id is NOT mixed into the SHA stream: it labels which KV cache group the
// content belongs to, not the content itself. Content hash is computed first,
// then the group_id is appended as a 4-byte big-endian value in 8 hex chars
// (64-hex content hash -> 72-hex key).
inline constexpr std::size_t kGroupIdHexLen = 8;  // 4-byte group_id as hex

// Fold m consecutive base content hashes into one coarse-block content hash
// (group block_size = m * base), coarsening the once-computed chain per group
// without re-hashing tokens. first_base is base_hashes[0]'s global base-page
// index; only complete blocks on the group grid are emitted, so
// idx = (m - first_base%m) % m skips a leading remainder. Chained via HashPage
// so order matters and no two runs collide.
inline std::vector<std::string> FoldBaseHashes(std::span<const std::string> base_hashes, std::int32_t first_base,
                                               std::int32_t m) {
    _assert(m >= 1, "fold factor must be >= 1");
    std::vector<std::string> out;
    out.reserve(base_hashes.size() / m + 1);
    std::int32_t idx = (m - first_base % m) % m;
    for (; idx + m <= static_cast<std::int32_t>(base_hashes.size()); idx += m) {
        std::string running;
        for (std::int32_t k = 0; k < m; ++k) {
            running = HashPage(std::span<const std::int32_t>{}, running, base_hashes.subspan(idx + k, 1));
        }
        out.push_back(running);
    }
    return out;
}

inline std::string MakeKeyWithGroupId(const std::string& block_hash, uint32_t group_id) {
    std::string key = block_hash;
    uint8_t b[4] = {
        static_cast<uint8_t>(group_id >> 24),
        static_cast<uint8_t>(group_id >> 16),
        static_cast<uint8_t>(group_id >> 8),
        static_cast<uint8_t>(group_id),
    };
    AppendHexBytes(key, b, 4);
    return key;
}

// Fold base content hashes into the group's coarse blocks (m = group_block_size
// / base), then wrap each with group_id. m == 1 keeps each base hash verbatim:
// FoldBaseHashes(m==1) would re-hash through HashPage, so the bypass is what
// keeps a uniform-block_size group's keys unchanged.
inline std::vector<std::string> MakeFoldedGroupKeys(std::span<const std::string> base_hashes, std::uint32_t group_id,
                                                    std::int32_t m, std::int32_t first_base = 0) {
    _assert(m >= 1, "fold factor must be >= 1");
    std::vector<std::string> keys;
    if (m == 1) {
        keys.reserve(base_hashes.size());
        for (const std::string& h : base_hashes) {
            keys.push_back(MakeKeyWithGroupId(h, group_id));
        }
        return keys;
    }
    std::vector<std::string> folded = FoldBaseHashes(base_hashes, first_base, m);
    keys.reserve(folded.size());
    for (const std::string& h : folded) {
        keys.push_back(MakeKeyWithGroupId(h, group_id));
    }
    return keys;
}

// Recover the content hash (strip the trailing group_id hex characters).
inline std::string GetBlockHashFromKey(const std::string& key) {
    if (key.size() < kGroupIdHexLen) {
        return {};
    }
    return key.substr(0, key.size() - kGroupIdHexLen);
}

// Recover the group_id (trailing group_id hex characters -> 4-byte big-endian).
inline uint32_t GetGroupIdFromHashKey(const std::string& key) {
    if (key.size() < kGroupIdHexLen) {
        return 0;
    }
    std::array<uint8_t, kGroupIdHexLen / 2> b{};
    HexToBytes(std::string_view{key}.substr(key.size() - kGroupIdHexLen), b);
    return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
           (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
}

// Compute the content hashes once, then wrap each page's hash with group_id. The
// content hash is group-independent, so multiple groups reuse it; the chain still
// links on the bare content hash, so group_id never leaks into the prefix chain.
inline std::vector<std::string> ComputePagedHashesWithGroup(
    const std::vector<std::span<const std::int32_t>>& paged_tokens, const std::string& prior, uint32_t group_id,
    const std::vector<std::span<const std::string>>& extra_keys_per_page = {}) {
    std::vector<std::string> keys = ComputePagedHashes(paged_tokens, prior, extra_keys_per_page);
    for (std::string& k : keys) {
        k = MakeKeyWithGroupId(k, group_id);
    }
    return keys;
}

}  // namespace tokenspeed
