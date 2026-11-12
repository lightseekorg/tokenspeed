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

#include <gtest/gtest.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache/core/content_hash.h"
#include "cache/core/hex_codec.h"
#include "cache/prefix/prefix_hasher.h"
#include "core/token_container.h"
#include "scheduler/request.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;
using key_span = std::span<const std::string>;

// HashPrefixPage frames the input as [prior_len][prior][token_count][tokens][extra...];
// all-empty input is two zero u32s (8 zero bytes), whose SHA-256 is below.
constexpr const char* kEmptyFramedSha256 = "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc";

token_span Tokens(const std::vector<std::int32_t>& v) {
    return token_span(v.data(), v.size());
}
key_span Keys(const std::vector<std::string>& v) {
    return key_span(v.data(), v.size());
}

TEST(RequestPromptHashesTest, RepeatedQueriesKeepTheFullInputPageChain) {
    Request request(RequestSpec{.request_id = "prompt", .tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9}},
                    /*prefix_granularity=*/4, Role::kFused);
    const std::string first = HashPrefixPage(Tokens({1, 2, 3, 4}), "");
    const std::string second = HashPrefixPage(Tokens({5, 6, 7, 8}), first);
    const std::vector<std::string> expected{first, second};
    for (int retry = 0; retry < 3; ++retry) {
        const auto hashes = request.PromptPrefixHashes();
        EXPECT_EQ(std::vector<std::string>(hashes.begin(), hashes.end()), expected);
    }
}

TEST(RequestPromptHashesTest, AnAlignedInputKeepsItsLastFullPageForDecodeRoleMatching) {
    Request request(RequestSpec{.request_id = "decode", .tokens = {1, 2, 3, 4}},
                    /*prefix_granularity=*/2, Role::kD);
    const std::string first = HashPrefixPage(Tokens({1, 2}), "");
    const std::string second = HashPrefixPage(Tokens({3, 4}), first);
    const auto hashes = request.PromptPrefixHashes();
    EXPECT_EQ(std::vector<std::string>(hashes.begin(), hashes.end()), (std::vector<std::string>{first, second}));
}

TEST(RequestPromptHashesTest, AShortInputHasNoFullPageHashes) {
    Request request(RequestSpec{.request_id = "short", .tokens = {1, 2, 3}},
                    /*prefix_granularity=*/4, Role::kFused);
    EXPECT_TRUE(request.PromptPrefixHashes().empty());
    EXPECT_TRUE(request.PromptPrefixHashes().empty());
}

TEST(RequestPromptHashesTest, BoundedStorageThenAdmissionAndPublicationShareTheSameChain) {
    Request request(RequestSpec{.request_id = "bounded", .tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9}},
                    /*prefix_granularity=*/2, Role::kFused);
    const auto expected = ComputePrefixHashes(request.FullPrefixPages(false), "");
    const auto bounded = request.PromptPrefixHashes(2);
    EXPECT_EQ(std::vector<std::string>(bounded.begin(), bounded.end()),
              (std::vector<std::string>{expected[0], expected[1]}));
    std::vector<std::string> completed;
    request.AppendPrefixHashes(completed, 1);
    EXPECT_EQ(completed, (std::vector<std::string>{expected[0]}));
    // Cross the cached prefix: publication computes just the missing suffix.
    request.AppendPrefixHashes(completed, 3);
    EXPECT_EQ(completed, (std::vector<std::string>{expected[0], expected[1], expected[2]}));
    const auto admitted = request.PromptPrefixHashes();
    EXPECT_EQ(std::vector<std::string>(admitted.begin(), admitted.end()), expected);
    request.AppendPrefixHashes(completed, 4);
    EXPECT_EQ(completed, expected);
    const auto shorter = request.PromptPrefixHashes(1);
    EXPECT_EQ(shorter.front(), expected.front());
    EXPECT_EQ(request.PromptPrefixHashes().size(), 4);
}

TEST(RequestPromptHashesTest, ABoundedSpanStaysValidAfterLaterCallsExtendTheChain) {
    std::vector<std::int32_t> tokens(33);
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<std::int32_t>(i);
    Request request(RequestSpec{.request_id = "stable", .tokens = tokens}, /*prefix_granularity=*/2, Role::kFused);
    const auto expected = ComputePrefixHashes(request.FullPrefixPages(false), "");
    ASSERT_EQ(expected.size(), 16);
    const auto bounded = request.PromptPrefixHashes(1);
    const auto middle = request.PromptPrefixHashes(5);
    const auto full = request.PromptPrefixHashes();
    // Each earlier span is a prefix of the same storage, still readable.
    EXPECT_EQ(bounded.data(), full.data());
    EXPECT_EQ(middle.data(), full.data());
    EXPECT_EQ(std::vector<std::string>(bounded.begin(), bounded.end()), (std::vector<std::string>{expected[0]}));
    EXPECT_EQ(std::vector<std::string>(middle.begin(), middle.end()),
              (std::vector<std::string>(expected.begin(), expected.begin() + 5)));
    EXPECT_EQ(std::vector<std::string>(full.begin(), full.end()), expected);
}

TEST(RequestPromptHashesTest, PublicationWithoutAdmissionHashesOnlyCompletedPages) {
    Request request(RequestSpec{.request_id = "uncached", .tokens = {1, 2, 3, 4, 5}},
                    /*prefix_granularity=*/2, Role::kFused);
    std::vector<std::string> completed;
    request.AppendPrefixHashes(completed, 0);
    EXPECT_TRUE(completed.empty());
    request.AppendPrefixHashes(completed, 1);
    EXPECT_EQ(completed, (std::vector<std::string>{HashPrefixPage(Tokens({1, 2}), "")}));
    request.AppendPrefixHashes(completed, 2);
    EXPECT_EQ(completed, ComputePrefixHashes(request.FullPrefixPages(false), ""));
}

// Decode publishes one completed page at a time: the completed chain grows
// geometrically, whether it copies prompt hashes or hashes pages itself.
TEST(RequestPromptHashesTest, PageByPageAppendsGrowTheCompletedChainGeometrically) {
    constexpr std::int32_t kPages = 256;
    std::vector<std::int32_t> tokens(2 * kPages);
    std::iota(tokens.begin(), tokens.end(), 0);
    for (const bool prompt_hashed : {false, true}) {
        Request request(RequestSpec{.request_id = "grow", .tokens = tokens}, /*prefix_granularity=*/2, Role::kFused);
        if (prompt_hashed) {
            ASSERT_EQ(request.PromptPrefixHashes().size(), kPages);
        }
        std::vector<std::string> completed;
        int reallocations = 0;
        for (std::int32_t pages = 1; pages <= kPages; ++pages) {
            // The storage moves exactly when it reallocates; capacity() may
            // report less than was allocated (_GLIBCXX_DEBUG_PEDANTIC).
            const std::string* storage = completed.data();
            request.AppendPrefixHashes(completed, pages);
            reallocations += completed.data() != storage ? 1 : 0;
        }
        EXPECT_EQ(completed, ComputePrefixHashes(request.FullPrefixPages(false), ""));
        // Doubling from one page reallocates log2(256) + 1 times.
        EXPECT_LE(reallocations, 9) << "prompt hashed: " << prompt_hashed;
    }
}

// ---- hex helpers --------------------------------------------------------

TEST(PageHasherHexTest, AppendHexBytesIsLowercaseTwoCharsPerByte) {
    std::string out;
    const uint8_t bytes[] = {0x00, 0x0f, 0xa3, 0xff};
    AppendHexBytes(out, bytes, 4);
    EXPECT_EQ(out, "000fa3ff");
}

TEST(PageHasherHexTest, HexToBytesIsInverseOfAppend) {
    const uint8_t bytes[] = {0x00, 0x01, 0x7e, 0x80, 0xab, 0xff};
    std::string hex;
    AppendHexBytes(hex, bytes, sizeof(bytes));
    std::vector<uint8_t> decoded = HexToBytes(hex);
    ASSERT_EQ(decoded.size(), sizeof(bytes));
    for (std::size_t i = 0; i < sizeof(bytes); ++i) {
        EXPECT_EQ(decoded[i], bytes[i]) << "byte " << i;
    }
}

TEST(PageHasherHexTest, HexToBytesAcceptsUppercase) {
    EXPECT_EQ(HexToBytes("ABCDEF"), HexToBytes("abcdef"));
}

// The per-character reading HexToBytes has always given every text; chained
// hashes under an opaque prior depend on it.
std::vector<std::uint8_t> ReferenceHexToBytes(const std::string& hex) {
    const auto nibble = [](char c) {
        return (c >= 'a')   ? static_cast<std::uint8_t>(c - 'a' + 10)
               : (c >= 'A') ? static_cast<std::uint8_t>(c - 'A' + 10)
                            : static_cast<std::uint8_t>(c - '0');
    };
    std::vector<std::uint8_t> bytes;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        bytes.push_back(static_cast<std::uint8_t>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
    }
    return bytes;
}

TEST(PageHasherHexTest, HexToBytesReadsCanonicalAndOpaqueTextLikeTheReference) {
    // Every character pair, alone and inside canonical text of each word
    // position, with and without an odd last character.
    const std::string canonical = "0123456789abcdef0123456789abcdef";
    for (unsigned high = 0; high < 256; ++high) {
        for (unsigned low = 0; low < 256; ++low) {
            const std::string pair{static_cast<char>(high), static_cast<char>(low)};
            ASSERT_EQ(HexToBytes(pair), ReferenceHexToBytes(pair)) << high << " " << low;
            std::string text = canonical;
            text.replace(2 * ((high ^ low) % 16), 2, pair);
            ASSERT_EQ(HexToBytes(text), ReferenceHexToBytes(text)) << high << " " << low;
            ASSERT_EQ(HexToBytes(text + "g"), ReferenceHexToBytes(text + "g")) << high << " " << low;
        }
    }
    for (std::size_t length = 0; length <= 9; ++length) {
        EXPECT_EQ(HexToBytes(canonical.substr(0, length)), ReferenceHexToBytes(canonical.substr(0, length)));
    }
}

// Per-character spelling the word-at-a-time codec must reproduce.
std::string ReferenceLowercaseHex(std::span<const std::uint8_t> bytes) {
    std::string text;
    for (const std::uint8_t byte : bytes) {
        text += "0123456789abcdef"[byte >> 4];
        text += "0123456789abcdef"[byte & 15];
    }
    return text;
}

TEST(PageHasherHexTest, EveryByteValueAndShortTailEncodesLikeTheReference) {
    std::vector<std::uint8_t> all(256);
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint8_t>(i);
    std::string text(2 * all.size(), '?');
    WriteLowercaseHex(all, text.data());
    EXPECT_EQ(text, ReferenceLowercaseHex(all));
    // Every length, so each short last word is covered, written in place
    // without touching the characters after it.
    for (std::size_t length = 0; length <= 9; ++length) {
        const std::span<const std::uint8_t> bytes = std::span<const std::uint8_t>(all).subspan(250 - length, length);
        std::string out(2 * length + 3, '?');
        WriteLowercaseHex(bytes, out.data());
        EXPECT_EQ(out, ReferenceLowercaseHex(bytes) + "???") << "length " << length;
        std::string appended = "ab";
        AppendHexBytes(appended, bytes.data(), bytes.size());
        EXPECT_EQ(appended, "ab" + ReferenceLowercaseHex(bytes)) << "length " << length;
    }
}

TEST(PageHasherHexTest, DecodingAcceptsExactlyLowercaseDigitsOfTheExpectedLength) {
    std::mt19937 random(9137);
    for (std::size_t length = 0; length <= 9; ++length) {
        std::vector<std::uint8_t> bytes(length);
        for (auto& byte : bytes) byte = static_cast<std::uint8_t>(random());
        const std::string text = ReferenceLowercaseHex(bytes);
        std::vector<std::uint8_t> decoded(length);
        ASSERT_TRUE(ReadLowercaseHex(text, decoded)) << "length " << length;
        EXPECT_EQ(decoded, bytes);
        std::vector<std::uint8_t> longer(length + 1);
        EXPECT_FALSE(ReadLowercaseHex(text, longer));
        EXPECT_FALSE(ReadLowercaseHex(text + "0", decoded)) << "odd length";
        for (std::size_t position = 0; position < text.size(); ++position) {
            for (unsigned value = 0; value < 256; ++value) {
                auto changed = text;
                changed[position] = static_cast<char>(value);
                const bool digit = (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                ASSERT_EQ(ReadLowercaseHex(changed, decoded), digit)
                    << "length " << length << " position " << position << " byte " << value;
                if (digit) {
                    EXPECT_EQ(ReferenceLowercaseHex(decoded), changed);
                }
            }
        }
    }
}

TEST(PageHasherHexTest, DigestTextRoundTripsThroughBothDirections) {
    std::mt19937 random(5);
    for (int trial = 0; trial < 64; ++trial) {
        ContentHash::Digest digest;
        for (auto& byte : digest) byte = static_cast<std::uint8_t>(random());
        const std::string text = DigestToHex(digest);
        EXPECT_EQ(text, ReferenceLowercaseHex(digest));
        ContentHash::Digest decoded;
        ASSERT_TRUE(ReadLowercaseHex(text, decoded));
        EXPECT_EQ(decoded, digest);
        EXPECT_EQ(HexToBytes(text), std::vector<std::uint8_t>(digest.begin(), digest.end()));
    }
}

// ---- HashPrefixPage -----------------------------------------------------------

TEST(HashPrefixPageTest, EmptyPageMatchesKnownSha256) {
    std::vector<std::int32_t> none;
    EXPECT_EQ(HashPrefixPage(Tokens(none), ""), kEmptyFramedSha256);
}

TEST(HashPrefixPageTest, OutputIs64HexChars) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::string h = HashPrefixPage(Tokens(toks), "");
    EXPECT_EQ(h.size(), 64u);
    EXPECT_EQ(h.find_first_not_of("0123456789abcdef"), std::string::npos);
}

TEST(HashPrefixPageTest, Deterministic) {
    std::vector<std::int32_t> toks = {7, 8, 9};
    EXPECT_EQ(HashPrefixPage(Tokens(toks), "seed"), HashPrefixPage(Tokens(toks), "seed"));
}

TEST(HashPrefixPageTest, DifferentTokensDifferentHash) {
    std::vector<std::int32_t> a = {1, 2, 3};
    std::vector<std::int32_t> b = {1, 2, 4};
    EXPECT_NE(HashPrefixPage(Tokens(a), ""), HashPrefixPage(Tokens(b), ""));
}

TEST(HashPrefixPageTest, TokenOrderMatters) {
    std::vector<std::int32_t> a = {1, 2};
    std::vector<std::int32_t> b = {2, 1};
    EXPECT_NE(HashPrefixPage(Tokens(a), ""), HashPrefixPage(Tokens(b), ""));
}

TEST(HashPrefixPageTest, PriorHashChangesOutput) {
    std::vector<std::int32_t> toks = {5, 6};
    std::string no_prior = HashPrefixPage(Tokens(toks), "");
    std::string with_prior = HashPrefixPage(Tokens(toks), no_prior);
    EXPECT_NE(no_prior, with_prior);
}

// A tail hash (hash@T over the page's first T mod P tokens, same prior) is
// the same function as the page hash; the token_count frame keeps it apart
// from the full page and from every other offset under that prior.
TEST(HashPrefixPageTest, TailHashesAreDistinctFromThePageAndFromEachOther) {
    const std::vector<std::int32_t> page = {1, 2, 3, 4};
    const std::string prior = HashPrefixPage(Tokens({9, 9, 9, 9}), "");
    const std::string full = HashPrefixPage(Tokens(page), prior);
    std::vector<std::string> tails;
    for (std::size_t t = 1; t < page.size(); ++t) {
        tails.push_back(HashPrefixPage(token_span(page.data(), t), prior));
        EXPECT_NE(tails.back(), full);
        EXPECT_NE(tails.back(), HashPrefixPage(token_span(page.data(), t), "")) << "the prior frames the tail";
    }
    EXPECT_NE(tails[0], tails[1]);
    EXPECT_NE(tails[1], tails[2]);
    // The page's own chain hash is what a tail's prior names, so a tail of
    // the next page never collides with a tail of this one.
    EXPECT_NE(HashPrefixPage(token_span(page.data(), 1), full), tails[0]);
}

// The identity overload chains the bytes the text overload decodes from the
// prior's text: an empty prior, a canonical digest (decoded from its text by
// one, read directly by the other) and opaque text, including uppercase hex,
// odd lengths and embedded nulls.
TEST(HashPrefixPageTest, AnIdentityPriorHashesLikeItsText) {
    const std::vector<std::int32_t> tokens = {7, -1, 0, 2147483647, 42};
    const std::string canonical = HashPrefixPage(Tokens({1, 2, 3}), "");
    std::string uppercase = canonical;
    for (char& c : uppercase) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    const std::vector<std::string> priors{
        "", canonical, std::string(64, '0'), uppercase, "opaque-key", std::string("a\0b", 3), std::string(63, 'f')};
    for (const std::string& prior : priors) {
        for (const token_span page : {Tokens(tokens), token_span{}}) {
            const ContentHash hash = HashPrefixPage(page, ContentHash{prior});
            EXPECT_EQ(hash.ToString(), HashPrefixPage(page, prior)) << prior;
            ASSERT_NE(hash.digest(), nullptr) << "a page hash is a canonical identity";
        }
    }
}

TEST(HashPrefixPageTest, EmptyExtraKeysEqualsTwoArgForm) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::vector<std::string> empty;
    EXPECT_EQ(HashPrefixPage(Tokens(toks), "p"), HashPrefixPage(Tokens(toks), "p", Keys(empty)));
}

TEST(HashPrefixPageTest, ExtraKeysChangeOutput) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::vector<std::string> keys = {"lora-A"};
    EXPECT_NE(HashPrefixPage(Tokens(toks), "p"), HashPrefixPage(Tokens(toks), "p", Keys(keys)));
}

TEST(HashPrefixPageTest, FramingDisambiguatesKeySplits) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> split_a = {"ab", "c"};
    std::vector<std::string> split_b = {"a", "bc"};
    EXPECT_NE(HashPrefixPage(Tokens(toks), "", Keys(split_a)), HashPrefixPage(Tokens(toks), "", Keys(split_b)));
}

TEST(HashPrefixPageTest, FramingDisambiguatesKeyCount) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> one = {"abc"};
    std::vector<std::string> two = {"a", "bc"};
    EXPECT_NE(HashPrefixPage(Tokens(toks), "", Keys(one)), HashPrefixPage(Tokens(toks), "", Keys(two)));
}

// A 32-byte prior reinterpreted as 8 LE tokens in page 0 must not produce the
// same stream as a chained page carrying that digest as prior.
TEST(HashPrefixPageTest, FramingDisambiguatesEmptyPriorFromChainedPage) {
    const std::string prior = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    std::vector<uint8_t> pb = HexToBytes(prior);
    ASSERT_EQ(pb.size(), 32u);

    std::vector<std::int32_t> toks(8);
    for (std::size_t i = 0; i < 8; ++i) {
        toks[i] = static_cast<std::int32_t>(
            static_cast<uint32_t>(pb[4 * i]) | (static_cast<uint32_t>(pb[4 * i + 1]) << 8) |
            (static_cast<uint32_t>(pb[4 * i + 2]) << 16) | (static_cast<uint32_t>(pb[4 * i + 3]) << 24));
    }
    std::vector<std::int32_t> none;
    std::string as_page0 = HashPrefixPage(Tokens(toks), "");
    std::string as_chained = HashPrefixPage(Tokens(none), prior);
    EXPECT_NE(as_page0, as_chained);
}

// A 4-byte extra key must not produce the same stream as count(1) + len(4) +
// the key's LE int32 folded back into the token list.
TEST(HashPrefixPageTest, FramingDisambiguatesTokensFromExtraKeys) {
    std::vector<std::int32_t> short_toks = {9, 8};
    std::vector<std::string> one_key = {"wxyz"};

    // count=1, len=4, then "wxyz" (0x7a797877 little-endian) as a trailing token.
    std::vector<std::int32_t> long_toks = {9, 8, 1, 4, 0x7a797877};
    std::vector<std::string> no_keys;

    EXPECT_NE(HashPrefixPage(Tokens(short_toks), "", Keys(one_key)),
              HashPrefixPage(Tokens(long_toks), "", Keys(no_keys)));
}

// Serialize the framed bytes independently of the streaming implementation.
TEST(HashPrefixPageTest, PreservesFramedBytesAndChainingAcrossTokenBatchBoundaries) {
    std::mt19937 random(871923);
    std::string prior;
    for (const std::size_t length : {0u, 1u, 2u, 15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u, 4096u}) {
        std::vector<std::int32_t> tokens(length);
        for (auto& token : tokens) {
            const auto bits = static_cast<std::uint32_t>(random());
            std::memcpy(&token, &bits, sizeof(token));
        }
        if (length >= 3) {
            tokens[0] = std::numeric_limits<std::int32_t>::min();
            tokens[1] = std::numeric_limits<std::int32_t>::max();
            tokens[2] = -1;
        }
        for (const auto& keys : std::vector<std::vector<std::string>>{{}, {""}, {std::string("a\0b", 3), "salt"}}) {
            std::vector<unsigned char> bytes;
            const auto append_u32 = [&](std::uint32_t value) {
                for (unsigned shift = 0; shift < 32; shift += 8) {
                    bytes.push_back(static_cast<unsigned char>(value >> shift));
                }
            };
            const auto prior_bytes = HexToBytes(prior);
            append_u32(static_cast<std::uint32_t>(prior_bytes.size()));
            bytes.insert(bytes.end(), prior_bytes.begin(), prior_bytes.end());
            append_u32(static_cast<std::uint32_t>(tokens.size()));
            const auto token_begin = bytes.size();
            for (const auto token : tokens) {
                append_u32(static_cast<std::uint32_t>(token));
            }
            const auto token_end = bytes.size();
            if (!keys.empty()) {
                append_u32(static_cast<std::uint32_t>(keys.size()));
                for (const auto& key : keys) {
                    append_u32(static_cast<std::uint32_t>(key.size()));
                    bytes.insert(bytes.end(), key.begin(), key.end());
                }
            }
            ContentHash::Digest digest;
            SHA256(bytes.data(), bytes.size(), digest.data());
            const auto expected = DigestToHex(digest);
            EXPECT_EQ(HashPrefixPage(tokens, prior, keys), expected) << "token count " << length;
            SHA256_CTX explicit_ctx;
            SHA256_Init(&explicit_ctx);
            SHA256_Update(&explicit_ctx, bytes.data(), token_begin);
            Sha256UpdateTokensExplicitLE(explicit_ctx, tokens);
            SHA256_Update(&explicit_ctx, bytes.data() + token_end, bytes.size() - token_end);
            SHA256_Final(digest.data(), &explicit_ctx);
            EXPECT_EQ(DigestToHex(digest), expected) << "explicit encoding, token count " << length;
            prior = expected;
        }
    }
}

// ---- ComputePrefixHashes (chaining) -------------------------------------

TEST(ComputePrefixHashesTest, EmptyPagesDoNotInspectAnUnusedPrior) {
    EXPECT_TRUE(ComputePrefixHashes({}, "not a hexadecimal prior").empty());
}

TEST(ComputePrefixHashesTest, MatchesManualRollingChain) {
    std::vector<std::int32_t> p0 = {1, 2};
    std::vector<std::int32_t> p1 = {3, 4};
    std::vector<std::int32_t> p2 = {5, 6};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1), Tokens(p2)};

    std::vector<std::string> got = ComputePrefixHashes(pages, "root");

    std::string h0 = HashPrefixPage(Tokens(p0), "root");
    std::string h1 = HashPrefixPage(Tokens(p1), h0);
    std::string h2 = HashPrefixPage(Tokens(p2), h1);

    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], h0);
    EXPECT_EQ(got[1], h1);
    EXPECT_EQ(got[2], h2);
}

TEST(ComputePrefixHashesTest, SamePageDifferentPrefixDiffers) {
    std::vector<std::int32_t> same = {9, 9};
    std::vector<std::int32_t> other = {1, 1};
    std::vector<token_span> a = {Tokens(same), Tokens(same)};
    std::vector<token_span> b = {Tokens(other), Tokens(same)};

    std::vector<std::string> ha = ComputePrefixHashes(a, "");
    std::vector<std::string> hb = ComputePrefixHashes(b, "");
    EXPECT_NE(ha[0], hb[0]);
    EXPECT_NE(ha[1], hb[1]);
}

TEST(ComputePrefixHashesTest, MissingExtraKeysPerPageTreatedAsEmpty) {
    std::vector<std::int32_t> p0 = {1};
    std::vector<std::int32_t> p1 = {2};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1)};

    std::vector<std::string> k0 = {"salt"};
    std::vector<key_span> extra = {Keys(k0)};

    std::vector<std::string> got = ComputePrefixHashes(pages, "", extra);

    std::string h0 = HashPrefixPage(Tokens(p0), "", Keys(k0));
    std::string h1 = HashPrefixPage(Tokens(p1), h0);
    EXPECT_EQ(got[0], h0);
    EXPECT_EQ(got[1], h1);
}

TEST(ComputePrefixHashesTest, IncrementalChainEqualsOneShot) {
    // 12-token stream, prefix_granularity 2 -> 6 pages.
    std::vector<std::int32_t> tokens(12);
    for (std::int32_t i = 0; i < 12; ++i) {
        tokens[i] = 100 + i;
    }
    std::vector<token_span> pages;
    for (std::size_t start = 0; start < tokens.size(); start += 2) {
        pages.push_back(token_span(tokens.data() + start, 2));
    }

    const std::vector<std::string> one_shot = ComputePrefixHashes(pages, "");

    const std::vector<token_span> head(pages.begin(), pages.begin() + 3);
    const std::vector<token_span> tail(pages.begin() + 3, pages.end());
    std::vector<std::string> incremental = ComputePrefixHashes(head, "");
    const std::vector<std::string> rest = ComputePrefixHashes(tail, incremental.back());
    incremental.insert(incremental.end(), rest.begin(), rest.end());

    EXPECT_EQ(incremental, one_shot);
}

TEST(ComputePrefixHashesTest, ExtendingInPlaceEqualsOneShotAcrossReallocation) {
    std::vector<std::int32_t> tokens(14);
    for (std::int32_t i = 0; i < 14; ++i) {
        tokens[i] = -7 * i;
    }
    std::vector<token_span> pages;
    for (std::size_t start = 0; start < tokens.size(); start += 2) {
        pages.push_back(token_span(tokens.data() + start, 2));
    }
    const std::vector<std::string> one_shot = ComputePrefixHashes(pages, "");

    // From nothing, the chain starts with the empty prior.
    std::vector<std::string> chain;
    ExtendPrefixHashes(chain, std::span(pages).first(1));
    EXPECT_EQ(chain, std::vector<std::string>{one_shot[0]});
    ExtendPrefixHashes(chain, {});
    EXPECT_EQ(chain.size(), 1u);
    // Each extension seeds from the chain's own last element, also when
    // growing the vector moves that element.
    for (const auto [begin, end] : {std::pair{1, 2}, std::pair{2, 5}, std::pair{5, 7}}) {
        chain.shrink_to_fit();
        ExtendPrefixHashes(chain, std::span(pages).subspan(begin, end - begin));
        EXPECT_EQ(chain, std::vector<std::string>(one_shot.begin(), one_shot.begin() + end));
    }
}

// A step hashes the pages it completed, sliced alone rather than out of a view
// of every page of the context.
TEST(PrefixPagesTest, SlicesOnlyTheAskedCompletePages) {
    const TokenContainer tokens{std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7}};

    const std::vector<token_span> pages = tokens.PrefixPages(/*prefix_granularity=*/2, 1, 3);

    ASSERT_EQ(pages.size(), 2u);
    EXPECT_EQ(std::vector<std::int32_t>(pages[0].begin(), pages[0].end()), (std::vector<std::int32_t>{3, 4}));
    EXPECT_EQ(std::vector<std::int32_t>(pages[1].begin(), pages[1].end()), (std::vector<std::int32_t>{5, 6}));
    EXPECT_TRUE(tokens.PrefixPages(2, 3, 3).empty());
    EXPECT_THROW(tokens.PrefixPages(2, 3, 4), std::runtime_error) << "token 7 starts a partial page";
    const TokenContainer aligned{std::vector<std::int32_t>{1, 2, 3, 4}};
    EXPECT_EQ(aligned.FullPrefixPages(2, /*except_last=*/false).size(), 2u);
    EXPECT_EQ(aligned.FullPrefixPages(2, /*except_last=*/true).size(), 1u) << "the last token's page is left out";
}

}  // namespace
}  // namespace tokenspeed::test
