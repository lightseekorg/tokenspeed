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
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "scheduler/kv_cache_events.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;
using key_span = std::span<const std::string>;

token_span Tokens(const std::vector<std::int32_t>& v) {
    return token_span(v.data(), v.size());
}
key_span Keys(const std::vector<std::string>& v) {
    return key_span(v.data(), v.size());
}

std::vector<token_span> Pages(const std::vector<std::vector<std::int32_t>>& pages) {
    std::vector<token_span> spans;
    spans.reserve(pages.size());
    for (const std::vector<std::int32_t>& page : pages) {
        spans.push_back(Tokens(page));
    }
    return spans;
}

TEST(LoraNamespaceHashTest, NoKeysIsByteIdenticalToUnnamespacedChain) {
    // The whole design rests on this: a base-model request must hash exactly as
    // it did before namespacing existed, or enabling LoRA would invalidate every
    // cache entry already on disk.
    const std::vector<std::vector<std::int32_t>> pages = {{1, 2}, {3, 4}, {5, 6}};
    const std::vector<token_span> spans = Pages(pages);

    EXPECT_EQ(ComputeNamespacedPagedHashes(spans, "", {}), ComputePagedHashes(spans, ""));
}

TEST(LoraNamespaceHashTest, DifferentAdaptersDiverge) {
    const std::vector<std::vector<std::int32_t>> pages = {{1, 2}, {3, 4}};
    const std::vector<token_span> spans = Pages(pages);
    const std::vector<std::string> key_a = {"adapter-a"};
    const std::vector<std::string> key_b = {"adapter-b"};

    const std::vector<std::string> hashes_a = ComputeNamespacedPagedHashes(spans, "", Keys(key_a));
    const std::vector<std::string> hashes_b = ComputeNamespacedPagedHashes(spans, "", Keys(key_b));
    const std::vector<std::string> hashes_base = ComputeNamespacedPagedHashes(spans, "", {});

    ASSERT_EQ(hashes_a.size(), 2u);
    for (std::size_t i = 0; i < hashes_a.size(); ++i) {
        EXPECT_NE(hashes_a[i], hashes_b[i]) << "page " << i << " collides across adapters";
        EXPECT_NE(hashes_a[i], hashes_base[i]) << "page " << i << " collides with the base model";
    }
}

TEST(LoraNamespaceHashTest, EveryPageCarriesTheKeyNotJustPageZero) {
    // Chaining would make a page-0 key enough for a chain that starts at page 0.
    // Keying every page is what makes a chain that starts *mid-request* isolated
    // too -- see AdvanceFromMidChainStaysIsolated below.
    const std::vector<std::vector<std::int32_t>> pages = {{1, 2}};
    const std::vector<token_span> spans = Pages(pages);
    const std::vector<std::string> key = {"adapter-a"};

    // Same tokens, same non-empty prior: only the key can distinguish them.
    const std::string prior = ComputePagedHashes(Pages({{9, 9}}), "").back();
    EXPECT_NE(ComputeNamespacedPagedHashes(spans, prior, Keys(key)), ComputeNamespacedPagedHashes(spans, prior, {}));
}

TEST(LoraNamespaceHashTest, IncrementalExtensionMatchesOneShot) {
    const std::vector<std::vector<std::int32_t>> pages = {{1, 2}, {3, 4}, {5, 6}};
    const std::vector<token_span> spans = Pages(pages);
    const std::vector<std::string> key = {"adapter-a"};

    const std::vector<std::string> one_shot = ComputeNamespacedPagedHashes(spans, "", Keys(key));

    // Seed pages [0,1), then extend to page 3 the way appendCompletedPageHashes does.
    std::vector<std::string> incremental = ComputeNamespacedPagedHashes(Pages({{1, 2}}), "", Keys(key));
    const std::vector<std::string> extended =
        AdvancePagedHashes(spans, /*first_page=*/1, incremental.back(), /*past_end_page=*/3, Keys(key));
    incremental.insert(incremental.end(), extended.begin(), extended.end());

    EXPECT_EQ(incremental, one_shot);
}

TEST(LoraNamespaceHashTest, AdvanceDroppingKeysBreaksIsolation) {
    // Guards the regression this change exists to prevent: if AdvancePagedHashes
    // is called without the adapter keys, the extended pages fall back into the
    // base-model namespace even though the request owns an adapter.
    const std::vector<std::vector<std::int32_t>> pages = {{1, 2}, {3, 4}};
    const std::vector<token_span> spans = Pages(pages);
    const std::vector<std::string> key = {"adapter-a"};

    const std::string prior = ComputeNamespacedPagedHashes(Pages({{1, 2}}), "", Keys(key)).back();
    const std::vector<std::string> with_keys = AdvancePagedHashes(spans, 1, prior, 2, Keys(key));
    const std::vector<std::string> without_keys = AdvancePagedHashes(spans, 1, prior, 2);

    EXPECT_NE(with_keys, without_keys);
}

TEST(LoraNamespaceHashTest, AdvanceFromMidChainStaysIsolated) {
    // A chain can start at page 0 with an empty prior on the incremental path:
    // a prompt shorter than one page produces no admission-time hashes, so the
    // first hash a LoRA request ever computes comes from AdvancePagedHashes.
    // That page must still be namespaced.
    const std::vector<std::vector<std::int32_t>> pages = {{1, 2}};
    const std::vector<token_span> spans = Pages(pages);
    const std::vector<std::string> key = {"adapter-a"};

    EXPECT_NE(AdvancePagedHashes(spans, /*first_page=*/0, "", /*past_end_page=*/1, Keys(key)),
              AdvancePagedHashes(spans, /*first_page=*/0, "", /*past_end_page=*/1));
}

// ---- published KV-cache event hashes --------------------------------------
// The event stream carries its own block hash lineage, separate from the page
// hashes above. Namespacing only the page hashes would leave two adapters
// publishing identical block hashes for the same prefix, so a cache-aware
// consumer could hand one adapter's block to the other.

TEST(LoraNamespaceKvEventTest, NoKeysIsByteIdenticalToUnnamespacedBlockHash) {
    const std::vector<std::int32_t> tokens = {1, 2, 3, 4};
    EXPECT_EQ(HashKvBlock(Tokens(tokens), std::nullopt, {}), HashKvBlock(Tokens(tokens), std::nullopt));
    EXPECT_EQ(HashKvBlock(Tokens(tokens), std::optional<std::uint64_t>{7}, {}),
              HashKvBlock(Tokens(tokens), std::optional<std::uint64_t>{7}));
}

TEST(LoraNamespaceKvEventTest, DifferentAdaptersPublishDifferentBlockHashes) {
    const std::vector<std::int32_t> tokens = {1, 2, 3, 4};
    const std::vector<std::string> key_a = {"1"};
    const std::vector<std::string> key_b = {"2"};

    const std::uint64_t base = HashKvBlock(Tokens(tokens), std::nullopt);
    const std::uint64_t a = HashKvBlock(Tokens(tokens), std::nullopt, Keys(key_a));
    const std::uint64_t b = HashKvBlock(Tokens(tokens), std::nullopt, Keys(key_b));

    EXPECT_NE(a, b) << "adapters collide in the published event stream";
    EXPECT_NE(a, base) << "adapter collides with the base model";
    EXPECT_NE(b, base) << "adapter collides with the base model";
}

TEST(LoraNamespaceKvEventTest, NamespaceSurvivesTheParentChain) {
    // The lineage is chained like the page hashes, so a divergence at the first
    // block must keep later blocks apart even though their tokens match.
    const std::vector<std::int32_t> first = {1, 2};
    const std::vector<std::int32_t> second = {3, 4};
    const std::vector<std::string> key_a = {"1"};
    const std::vector<std::string> key_b = {"2"};

    const std::uint64_t a0 = HashKvBlock(Tokens(first), std::nullopt, Keys(key_a));
    const std::uint64_t b0 = HashKvBlock(Tokens(first), std::nullopt, Keys(key_b));
    EXPECT_NE(HashKvBlock(Tokens(second), a0, Keys(key_a)), HashKvBlock(Tokens(second), b0, Keys(key_b)));
}

}  // namespace
}  // namespace tokenspeed::test
