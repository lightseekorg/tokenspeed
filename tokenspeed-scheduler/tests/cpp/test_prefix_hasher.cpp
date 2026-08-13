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

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "cache/prefix/prefix_hasher.h"

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

    EXPECT_NE(HashPrefixPage(Tokens(short_toks), "", Keys(one_key)), HashPrefixPage(Tokens(long_toks), "", Keys(no_keys)));
}

// ---- ComputePrefixHashes (chaining) -------------------------------------

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
    // 12-token stream, page_size 2 -> 6 pages.
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

TEST(ComputePrefixHashesTest, AdvancePrefixHashesReturnsOnlyNewPages) {
    std::vector<std::int32_t> tokens(12);
    for (std::int32_t i = 0; i < 12; ++i) {
        tokens[i] = 100 + i;
    }
    std::vector<token_span> pages;
    for (std::size_t start = 0; start < tokens.size(); start += 2) {
        pages.push_back(token_span(tokens.data() + start, 2));
    }

    const std::vector<std::string> one_shot = ComputePrefixHashes(pages, "");
    const std::vector<std::string> first = AdvancePrefixHashes(pages, 0, "", 2);
    const std::vector<std::string> second = AdvancePrefixHashes(pages, 2, first.back(), 5);

    EXPECT_EQ(first, std::vector<std::string>(one_shot.begin(), one_shot.begin() + 2));
    EXPECT_EQ(second, std::vector<std::string>(one_shot.begin() + 2, one_shot.begin() + 5));
}

}  // namespace
}  // namespace tokenspeed::test
