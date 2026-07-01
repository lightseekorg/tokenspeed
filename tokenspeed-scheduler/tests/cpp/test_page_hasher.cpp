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

#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;
using key_span = std::span<const std::string>;

// HashPage frames the whole input as [prior_len][prior][token_count][tokens]
// [extra...]. With no prior, no tokens and no extra_keys that is two zero u32s
// (8 zero bytes), whose SHA-256 is the vector below -- pins the helpers to real
// SHA-256 output over the framed stream.
constexpr const char* kEmptyFramedSha256 = "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc";

token_span Tokens(const std::vector<std::int32_t>& v) { return token_span(v.data(), v.size()); }
key_span Keys(const std::vector<std::string>& v) { return key_span(v.data(), v.size()); }

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

// ---- HashPage -----------------------------------------------------------

TEST(HashPageTest, EmptyPageMatchesKnownSha256) {
    std::vector<std::int32_t> none;
    EXPECT_EQ(HashPage(Tokens(none), ""), kEmptyFramedSha256);
}

TEST(HashPageTest, OutputIs64HexChars) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::string h = HashPage(Tokens(toks), "");
    EXPECT_EQ(h.size(), 64u);
    EXPECT_EQ(h.find_first_not_of("0123456789abcdef"), std::string::npos);
}

TEST(HashPageTest, Deterministic) {
    std::vector<std::int32_t> toks = {7, 8, 9};
    EXPECT_EQ(HashPage(Tokens(toks), "seed"), HashPage(Tokens(toks), "seed"));
}

TEST(HashPageTest, DifferentTokensDifferentHash) {
    std::vector<std::int32_t> a = {1, 2, 3};
    std::vector<std::int32_t> b = {1, 2, 4};
    EXPECT_NE(HashPage(Tokens(a), ""), HashPage(Tokens(b), ""));
}

TEST(HashPageTest, TokenOrderMatters) {
    std::vector<std::int32_t> a = {1, 2};
    std::vector<std::int32_t> b = {2, 1};
    EXPECT_NE(HashPage(Tokens(a), ""), HashPage(Tokens(b), ""));
}

TEST(HashPageTest, PriorHashChangesOutput) {
    std::vector<std::int32_t> toks = {5, 6};
    std::string no_prior = HashPage(Tokens(toks), "");
    std::string with_prior = HashPage(Tokens(toks), no_prior);
    EXPECT_NE(no_prior, with_prior);
}

TEST(HashPageTest, EmptyExtraKeysEqualsTwoArgForm) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::vector<std::string> empty;
    EXPECT_EQ(HashPage(Tokens(toks), "p"), HashPage(Tokens(toks), "p", Keys(empty)));
}

TEST(HashPageTest, ExtraKeysChangeOutput) {
    std::vector<std::int32_t> toks = {1, 2, 3};
    std::vector<std::string> keys = {"lora-A"};
    EXPECT_NE(HashPage(Tokens(toks), "p"), HashPage(Tokens(toks), "p", Keys(keys)));
}

// Length-prefix framing must prevent boundary-shift collisions: the same
// concatenated bytes split differently must hash differently.
TEST(HashPageTest, FramingDisambiguatesKeySplits) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> split_a = {"ab", "c"};
    std::vector<std::string> split_b = {"a", "bc"};
    EXPECT_NE(HashPage(Tokens(toks), "", Keys(split_a)), HashPage(Tokens(toks), "", Keys(split_b)));
}

// A single key must not collide with the same content carried as two keys.
TEST(HashPageTest, FramingDisambiguatesKeyCount) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> one = {"abc"};
    std::vector<std::string> two = {"a", "bc"};
    EXPECT_NE(HashPage(Tokens(toks), "", Keys(one)), HashPage(Tokens(toks), "", Keys(two)));
}

// Regression -- prior_len framing. Before whole-input framing, an empty prior
// wrote zero bytes, so a page-0 whose leading tokens LE-encode to a 32-byte
// digest produced the same stream as a chained page carrying that digest as
// prior. Reinterpret a 32-byte prior as 8 little-endian tokens and assert the
// two forms no longer collide.
TEST(HashPageTest, FramingDisambiguatesEmptyPriorFromChainedPage) {
    const std::string prior = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    std::vector<uint8_t> pb = HexToBytes(prior);
    ASSERT_EQ(pb.size(), 32u);

    std::vector<std::int32_t> toks(8);
    for (std::size_t i = 0; i < 8; ++i) {
        toks[i] = static_cast<std::int32_t>(static_cast<uint32_t>(pb[4 * i]) |
                                            (static_cast<uint32_t>(pb[4 * i + 1]) << 8) |
                                            (static_cast<uint32_t>(pb[4 * i + 2]) << 16) |
                                            (static_cast<uint32_t>(pb[4 * i + 3]) << 24));
    }
    std::vector<std::int32_t> none;
    // page 0: no prior, 8 tokens that reproduce the digest bytes.
    std::string as_page0 = HashPage(Tokens(toks), "");
    // chained page: that digest as prior, no tokens.
    std::string as_chained = HashPage(Tokens(none), prior);
    EXPECT_NE(as_page0, as_chained);
}

// Regression -- token_count framing. Before whole-input framing the token block
// had no count prefix, so it could bleed into the extra_keys frame: a 4-byte key
// carried as one extra_key produced the same stream as the bytes count(1) +
// len(4) + key folded back into the token list. The key "wxyz" LE-decodes to the
// int32 below; assert the two forms no longer collide.
TEST(HashPageTest, FramingDisambiguatesTokensFromExtraKeys) {
    std::vector<std::int32_t> short_toks = {9, 8};
    std::vector<std::string> one_key = {"wxyz"};

    // count=1, len=4, then "wxyz" (0x7a797877 little-endian) as a trailing token.
    std::vector<std::int32_t> long_toks = {9, 8, 1, 4, 0x7a797877};
    std::vector<std::string> no_keys;

    EXPECT_NE(HashPage(Tokens(short_toks), "", Keys(one_key)),
              HashPage(Tokens(long_toks), "", Keys(no_keys)));
}

// ---- ComputePagedHashes (chaining) -------------------------------------

TEST(ComputePagedHashesTest, MatchesManualRollingChain) {
    std::vector<std::int32_t> p0 = {1, 2};
    std::vector<std::int32_t> p1 = {3, 4};
    std::vector<std::int32_t> p2 = {5, 6};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1), Tokens(p2)};

    std::vector<std::string> got = ComputePagedHashes(pages, "root");

    std::string h0 = HashPage(Tokens(p0), "root");
    std::string h1 = HashPage(Tokens(p1), h0);
    std::string h2 = HashPage(Tokens(p2), h1);

    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], h0);
    EXPECT_EQ(got[1], h1);
    EXPECT_EQ(got[2], h2);
}

TEST(ComputePagedHashesTest, SamePageDifferentPrefixDiffers) {
    std::vector<std::int32_t> same = {9, 9};
    std::vector<std::int32_t> other = {1, 1};
    std::vector<token_span> a = {Tokens(same), Tokens(same)};
    std::vector<token_span> b = {Tokens(other), Tokens(same)};

    std::vector<std::string> ha = ComputePagedHashes(a, "");
    std::vector<std::string> hb = ComputePagedHashes(b, "");
    // identical first page -> identical hash; differing prefix -> 2nd page differs.
    EXPECT_NE(ha[0], hb[0]);
    EXPECT_NE(ha[1], hb[1]);
}

TEST(ComputePagedHashesTest, MissingExtraKeysPerPageTreatedAsEmpty) {
    std::vector<std::int32_t> p0 = {1};
    std::vector<std::int32_t> p1 = {2};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1)};

    // extra_keys only provided for page 0; page 1 should fall back to empty.
    std::vector<std::string> k0 = {"salt"};
    std::vector<key_span> extra = {Keys(k0)};

    std::vector<std::string> got = ComputePagedHashes(pages, "", extra);

    std::string h0 = HashPage(Tokens(p0), "", Keys(k0));
    std::string h1 = HashPage(Tokens(p1), h0);
    EXPECT_EQ(got[0], h0);
    EXPECT_EQ(got[1], h1);
}

// ---- group_id pack / unpack --------------------------------------------

TEST(GroupIdTest, KeyIsContentHashPlusEightHex) {
    std::string content(64, 'a');
    std::string key = MakeKeyWithGroupId(content, 7);
    EXPECT_EQ(key.size(), 72u);
    EXPECT_EQ(key.substr(0, 64), content);
    EXPECT_EQ(key.substr(64), "00000007");  // big-endian
}

TEST(GroupIdTest, BigEndianByteOrder) {
    std::string key = MakeKeyWithGroupId(std::string(64, 'a'), 0x01020304u);
    EXPECT_EQ(key.substr(64), "01020304");
}

TEST(GroupIdTest, RoundTrip) {
    std::string content(64, 'd');
    for (uint32_t gid : {0u, 1u, 7u, 255u, 256u, 0xdeadbeefu, 0xffffffffu}) {
        std::string key = MakeKeyWithGroupId(content, gid);
        EXPECT_EQ(GetBlockHashFromKey(key), content) << "gid " << gid;
        EXPECT_EQ(GetGroupIdFromHashKey(key), gid) << "gid " << gid;
    }
}

TEST(GroupIdTest, ContentHashIndependentOfGroup) {
    std::string content(64, 'c');
    std::string k0 = MakeKeyWithGroupId(content, 0);
    std::string k1 = MakeKeyWithGroupId(content, 1);
    EXPECT_EQ(GetBlockHashFromKey(k0), GetBlockHashFromKey(k1));
    EXPECT_NE(k0, k1);
}

TEST(GroupIdTest, ShortKeyDecodesDefensively) {
    EXPECT_EQ(GetBlockHashFromKey("abc"), "");
    EXPECT_EQ(GetGroupIdFromHashKey("abc"), 0u);
}

// ---- ComputePagedHashesWithGroup ---------------------------------------

TEST(ComputePagedHashesWithGroupTest, EqualsBareHashesWrappedWithGroup) {
    std::vector<std::int32_t> p0 = {1, 2};
    std::vector<std::int32_t> p1 = {3, 4};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1)};

    std::vector<std::string> bare = ComputePagedHashes(pages, "r");
    std::vector<std::string> grouped = ComputePagedHashesWithGroup(pages, "r", 42);

    ASSERT_EQ(grouped.size(), bare.size());
    for (std::size_t i = 0; i < bare.size(); ++i) {
        EXPECT_EQ(grouped[i], MakeKeyWithGroupId(bare[i], 42)) << "page " << i;
        // group_id rides outside the chain: stripping it recovers the bare hash.
        EXPECT_EQ(GetBlockHashFromKey(grouped[i]), bare[i]) << "page " << i;
    }
}

TEST(ComputePagedHashesWithGroupTest, GroupDoesNotLeakIntoPrefixChain) {
    std::vector<std::int32_t> p0 = {1, 2};
    std::vector<std::int32_t> p1 = {3, 4};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1)};

    std::vector<std::string> g0 = ComputePagedHashesWithGroup(pages, "r", 0);
    std::vector<std::string> g9 = ComputePagedHashesWithGroup(pages, "r", 9);

    // Same content, different group: only the trailing group_id hex differs.
    for (std::size_t i = 0; i < g0.size(); ++i) {
        EXPECT_EQ(GetBlockHashFromKey(g0[i]), GetBlockHashFromKey(g9[i])) << "page " << i;
    }
}

}  // namespace
}  // namespace tokenspeed::test
