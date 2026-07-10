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

// HashPage frames the input as [prior_len][prior][token_count][tokens][extra...];
// all-empty input is two zero u32s (8 zero bytes), whose SHA-256 is below.
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

TEST(HashPageTest, FramingDisambiguatesKeySplits) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> split_a = {"ab", "c"};
    std::vector<std::string> split_b = {"a", "bc"};
    EXPECT_NE(HashPage(Tokens(toks), "", Keys(split_a)), HashPage(Tokens(toks), "", Keys(split_b)));
}

TEST(HashPageTest, FramingDisambiguatesKeyCount) {
    std::vector<std::int32_t> toks = {1};
    std::vector<std::string> one = {"abc"};
    std::vector<std::string> two = {"a", "bc"};
    EXPECT_NE(HashPage(Tokens(toks), "", Keys(one)), HashPage(Tokens(toks), "", Keys(two)));
}

// A 32-byte prior reinterpreted as 8 LE tokens in page 0 must not produce the
// same stream as a chained page carrying that digest as prior.
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
    std::string as_page0 = HashPage(Tokens(toks), "");
    std::string as_chained = HashPage(Tokens(none), prior);
    EXPECT_NE(as_page0, as_chained);
}

// A 4-byte extra key must not produce the same stream as count(1) + len(4) +
// the key's LE int32 folded back into the token list.
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
    EXPECT_NE(ha[0], hb[0]);
    EXPECT_NE(ha[1], hb[1]);
}

TEST(ComputePagedHashesTest, MissingExtraKeysPerPageTreatedAsEmpty) {
    std::vector<std::int32_t> p0 = {1};
    std::vector<std::int32_t> p1 = {2};
    std::vector<token_span> pages = {Tokens(p0), Tokens(p1)};

    std::vector<std::string> k0 = {"salt"};
    std::vector<key_span> extra = {Keys(k0)};

    std::vector<std::string> got = ComputePagedHashes(pages, "", extra);

    std::string h0 = HashPage(Tokens(p0), "", Keys(k0));
    std::string h1 = HashPage(Tokens(p1), h0);
    EXPECT_EQ(got[0], h0);
    EXPECT_EQ(got[1], h1);
}

TEST(ComputePagedHashesTest, IncrementalChainEqualsOneShot) {
    // 12-token stream, page_size 2 -> 6 pages.
    std::vector<std::int32_t> tokens(12);
    for (std::int32_t i = 0; i < 12; ++i) {
        tokens[i] = 100 + i;
    }
    std::vector<token_span> pages;
    for (std::size_t start = 0; start < tokens.size(); start += 2) {
        pages.push_back(token_span(tokens.data() + start, 2));
    }

    const std::vector<std::string> one_shot = ComputePagedHashes(pages, "");

    const std::vector<token_span> head(pages.begin(), pages.begin() + 3);
    const std::vector<token_span> tail(pages.begin() + 3, pages.end());
    std::vector<std::string> incremental = ComputePagedHashes(head, "");
    const std::vector<std::string> rest = ComputePagedHashes(tail, incremental.back());
    incremental.insert(incremental.end(), rest.begin(), rest.end());

    EXPECT_EQ(incremental, one_shot);
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

    for (std::size_t i = 0; i < g0.size(); ++i) {
        EXPECT_EQ(GetBlockHashFromKey(g0[i]), GetBlockHashFromKey(g9[i])) << "page " << i;
    }
}

// ---- FoldBaseHashes ----------------------------------------------------

TEST(FoldBaseHashesTest, IdentityWhenGroupEqualsBase) {
    std::vector<std::string> base = {"aa", "bb", "cc"};
    auto folded = FoldBaseHashes(base, /*first_base=*/0, /*m=*/1);
    ASSERT_EQ(folded.size(), 3u);
    EXPECT_EQ(folded[0], HashPage(std::span<const std::int32_t>{}, "", std::vector<std::string>{"aa"}));
}

TEST(FoldBaseHashesTest, FoldsMConsecutiveIntoOneOrderSensitive) {
    std::vector<std::string> base = {"a0", "a1", "a2", "a3", "a4", "a5"};
    auto folded = FoldBaseHashes(base, 0, 2);
    ASSERT_EQ(folded.size(), 3u);
    std::vector<std::string> swapped = {"a1", "a0", "a2", "a3", "a4", "a5"};
    auto folded2 = FoldBaseHashes(swapped, 0, 2);
    EXPECT_NE(folded[0], folded2[0]);
    EXPECT_EQ(folded[1], folded2[1]);
}

TEST(FoldBaseHashesTest, DropsIncompleteTrailingGroupBlock) {
    std::vector<std::string> base = {"a0", "a1", "a2", "a3", "a4"};
    auto folded = FoldBaseHashes(base, 0, 2);
    EXPECT_EQ(folded.size(), 2u);
}

TEST(FoldBaseHashesTest, FirstBaseOffsetShiftsFoldWindow) {
    // first_base=1, m=2: first_base%m==1 -> drop 1 leading page, [a2,a3] folds into 1 block
    std::vector<std::string> base = {"a1", "a2", "a3"};
    auto folded = FoldBaseHashes(base, /*first_base=*/1, /*m=*/2);
    ASSERT_EQ(folded.size(), 1u);
}

// ---- MakeFoldedGroupKeys ----------------------------------------------

TEST(MakeFoldedGroupKeysTest, MEqualsOneIsByteIdenticalToRawPerBaseKey) {
    std::vector<std::string> base = {"aa", "bb", "cc"};
    auto keys = MakeFoldedGroupKeys(base, /*group_id=*/7, /*m=*/1);
    ASSERT_EQ(keys.size(), 3u);
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_EQ(keys[i], MakeKeyWithGroupId(base[i], 7)) << "page " << i;
    }
}

TEST(MakeFoldedGroupKeysTest, MTwoFoldsThenWrapsGroupId) {
    std::vector<std::string> base = {"a0", "a1", "a2", "a3"};
    auto keys = MakeFoldedGroupKeys(base, /*group_id=*/3, /*m=*/2);
    auto folded = FoldBaseHashes(base, /*first_base=*/0, /*m=*/2);
    ASSERT_EQ(keys.size(), 2u);
    ASSERT_EQ(folded.size(), 2u);
    for (std::size_t i = 0; i < folded.size(); ++i) {
        EXPECT_EQ(keys[i], MakeKeyWithGroupId(folded[i], 3)) << "coarse block " << i;
    }
}

TEST(MakeFoldedGroupKeysTest, FirstBaseOffsetShiftsFoldedKeys) {
    // first_base=1, m=2 -> drop 1 leading base page, [a2,a3] fold into 1 coarse key.
    std::vector<std::string> base = {"a1", "a2", "a3"};
    auto keys = MakeFoldedGroupKeys(base, /*group_id=*/0, /*m=*/2, /*first_base=*/1);
    auto folded = FoldBaseHashes(base, /*first_base=*/1, /*m=*/2);
    ASSERT_EQ(keys.size(), 1u);
    EXPECT_EQ(keys[0], MakeKeyWithGroupId(folded[0], 0));
}

}  // namespace
}  // namespace tokenspeed::test
