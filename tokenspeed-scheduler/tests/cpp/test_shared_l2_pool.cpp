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

#include <algorithm>
#include <numeric>
#include <set>

#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/shared_l2_pool.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kPage = 4;
constexpr std::uint32_t kFull = 0;
constexpr std::uint32_t kState = 1;

// A prompt published at its end: Full KV pages, the partial page as a tail,
// and the State checkpoint after its last token.
struct Prompt {
    std::vector<std::int32_t> tokens;
    std::vector<ContentHash> hashes;
    std::optional<TailPublication> tail;

    explicit Prompt(std::vector<std::int32_t> input) : tokens{std::move(input)} {
        ContentHash prior;
        for (std::size_t i = 0; i + kPage <= tokens.size(); i += kPage) {
            prior = HashPrefixPage(std::span{tokens}.subspan(i, kPage), prior);
            hashes.push_back(prior);
        }
        if (tokens.size() % kPage != 0) {
            tail = TailPublication{.tail_tokens = static_cast<std::int32_t>(tokens.size() % kPage),
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(hashes.size() * kPage), prior)};
        }
    }
    ContentHash End() const { return tail ? tail->hash : hashes.back(); }
    std::vector<CacheKey> Keys(std::vector<std::uint32_t> state_groups) const {
        std::vector<CacheKey> keys;
        for (const ContentHash& hash : hashes) keys.push_back(CacheKey{.group_id = kFull, .content_hash = hash});
        if (tail) keys.push_back(CacheKey{.group_id = kFull, .content_hash = tail->hash});
        for (std::uint32_t group : state_groups) keys.push_back(CacheKey{.group_id = group, .content_hash = End()});
        return keys;
    }
    SharedL2Pool::Publication Publication() const { return {.tokens = tokens, .prefix_hashes = hashes, .tail = tail}; }
    CacheCoordinator::PromptPrefix Prefix() const {
        return {.prefix_hashes = hashes, .tokens = tokens, .max_tokens = static_cast<std::int32_t>(tokens.size())};
    }
};

std::vector<std::int32_t> Sequence(std::int32_t count, std::int32_t first = 1) {
    std::vector<std::int32_t> tokens(count);
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
}

std::vector<CacheGroupSpec> HybridSpecs() {
    return {CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = kPage},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}

bool AllGranted(const SharedL2Pool::Reservation& reservation) {
    return std::ranges::all_of(reservation.pages, [](std::int32_t page) { return page >= 0; });
}

void Write(SharedL2Pool& pool, SharedL2Pool::ClientId client, const std::vector<CacheKey>& keys,
           std::optional<SharedL2Pool::Publication> publication) {
    const SharedL2Pool::Reservation reservation = pool.Reserve(client, keys);
    ASSERT_TRUE(AllGranted(reservation));
    std::vector<SharedL2Pool::Publication> publications;
    if (publication) publications.push_back(std::move(*publication));
    pool.Commit(client, reservation.ticket, publications);
}

bool HoldsAll(const SharedL2Pool& pool, const std::vector<CacheKey>& keys) {
    return std::ranges::all_of(keys, [&](const CacheKey& key) { return pool.Contains(key); });
}

TEST(SharedL2Pool, OneClientResumesWhereAnotherPublished) {
    SharedL2Pool pool(HybridSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Prompt first{Sequence(10)};
    Write(pool, 1, first.Keys({kState}), first.Publication());

    std::vector<std::int32_t> longer = first.tokens;
    for (std::int32_t token : Sequence(6, 50)) longer.push_back(token);
    const SharedL2Pool::Hit hit = pool.Probe(2, Prompt{longer}.Prefix());

    EXPECT_EQ(hit.num_tokens, 10);
    ASSERT_TRUE(hit.tail);
    EXPECT_EQ(hit.tail->hash, first.tail->hash);
    std::vector<CacheKey> keys;
    std::set<std::pair<std::uint32_t, std::int32_t>> pages;
    for (const SharedL2Pool::Slot& slot : hit.slots) {
        keys.push_back(slot.key);
        pages.emplace(slot.key.group_id, slot.page);
    }
    std::vector<CacheKey> expected = first.Keys({kState});
    std::ranges::sort(keys, {}, &CacheKey::content_hash);
    std::ranges::sort(expected, {}, &CacheKey::content_hash);
    EXPECT_EQ(keys, expected);
    EXPECT_EQ(pages.size(), hit.slots.size());
    EXPECT_EQ(pool.GetStats().pinned_blocks, 4);
    pool.Release(2, hit.ticket);
    EXPECT_EQ(pool.GetStats().pinned_blocks, 0);
}

TEST(SharedL2Pool, ACheckpointWithoutItsStateBlocksGivesNoHit) {
    SharedL2Pool pool(HybridSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Prompt first{Sequence(10)};
    Write(pool, 1, first.Keys({}), first.Publication());
    EXPECT_EQ(pool.Probe(2, first.Prefix()).num_tokens, 0);
}

TEST(SharedL2Pool, PinnedSlotsAreNotEvictedUntilReleased) {
    // Full packs two blocks per LCM block: the first prompt fills two Full and
    // one State LCM block, leaving one LCM block and one Full slot free.
    SharedL2Pool pool(HybridSpecs(), kPage, 4, /*headroom_lcm_blocks=*/0);
    const Prompt first{Sequence(10)};
    Write(pool, 1, first.Keys({kState}), first.Publication());
    const SharedL2Pool::Hit hit = pool.Probe(2, first.Prefix());
    ASSERT_EQ(hit.num_tokens, 10);

    const Prompt second{Sequence(16, 100)};
    const std::vector<CacheKey> second_keys = second.Keys({});
    const SharedL2Pool::Reservation crowded = pool.Reserve(1, second_keys);
    EXPECT_EQ(std::ranges::count(crowded.pages, SharedL2Pool::kNoSlot), 1);
    EXPECT_TRUE(HoldsAll(pool, first.Keys({kState})));
    pool.Commit(1, crowded.ticket, {});

    pool.Release(2, hit.ticket);
    const auto refused = std::ranges::find(crowded.pages, SharedL2Pool::kNoSlot) - crowded.pages.begin();
    const SharedL2Pool::Reservation retry = pool.Reserve(1, std::span{second_keys}.subspan(refused, 1));
    EXPECT_GE(retry.pages.front(), 0);
    EXPECT_FALSE(HoldsAll(pool, first.Keys({kState})));
    pool.Disconnect(1);
}

TEST(SharedL2Pool, EachBlockIsWrittenOnce) {
    SharedL2Pool pool(HybridSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Prompt prompt{Sequence(8)};
    const std::vector<CacheKey> keys = prompt.Keys({kState});

    const SharedL2Pool::Reservation first = pool.Reserve(1, keys);
    EXPECT_TRUE(AllGranted(first));
    const SharedL2Pool::Reservation racing = pool.Reserve(2, keys);
    EXPECT_EQ(racing.ticket, 0u);
    EXPECT_EQ(std::ranges::count(racing.pages, SharedL2Pool::kHeld), static_cast<std::ptrdiff_t>(keys.size()));

    pool.Disconnect(1);
    const SharedL2Pool::Reservation second = pool.Reserve(2, keys);
    EXPECT_TRUE(AllGranted(second));
    pool.Commit(2, second.ticket, std::vector{prompt.Publication()});
    EXPECT_TRUE(HoldsAll(pool, keys));
    EXPECT_EQ(pool.Reserve(1, keys).ticket, 0u);
}

TEST(SharedL2Pool, ADisconnectedClientLeavesNothingHeld) {
    SharedL2Pool pool(HybridSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Prompt first{Sequence(8)};
    Write(pool, 1, first.Keys({kState}), first.Publication());
    const Prompt second{Sequence(8, 100)};
    ASSERT_GT(pool.Probe(2, first.Prefix()).num_tokens, 0);
    ASSERT_NE(pool.Reserve(2, second.Keys({kState})).ticket, 0u);

    pool.Disconnect(2);

    const SharedL2Pool::Stats stats = pool.GetStats();
    EXPECT_EQ(stats.pinned_blocks, 0);
    EXPECT_EQ(stats.reserved_blocks, 0u);
    EXPECT_NE(pool.Reserve(3, second.Keys({kState})).ticket, 0u);
}

std::vector<CacheGroupSpec> TwoStateSpecs() {
    return {CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}

std::vector<CacheKey> StateKeys(const Prompt& prompt) {
    return {CacheKey{.group_id = 0, .content_hash = prompt.End()},
            CacheKey{.group_id = 1, .content_hash = prompt.End()}};
}

// An unrelated checkpoint, then three turns of one conversation, each
// published by a different client: the pool sees them as one lineage.
struct Lineage {
    Prompt unrelated{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};

    explicit Lineage(SharedL2Pool& pool) {
        SharedL2Pool::ClientId client = 0;
        for (const Prompt* prompt : {&unrelated, &a, &b, &c}) {
            Write(pool, ++client, StateKeys(*prompt), prompt->Publication());
        }
    }
};

TEST(SharedL2Pool, EvictionDemotesACheckpointTwoNewerOnesSupersede) {
    SharedL2Pool pool(TwoStateSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Lineage lineage(pool);
    const std::vector<CacheKey> next = StateKeys(Prompt{Sequence(16)});
    const SharedL2Pool::Reservation reservation = pool.Reserve(9, next);
    EXPECT_TRUE(AllGranted(reservation));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.unrelated)));
    EXPECT_TRUE(std::ranges::none_of(StateKeys(lineage.a), [&](const CacheKey& key) { return pool.Contains(key); }));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.b)));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.c)));
    pool.Disconnect(9);
}

TEST(SharedL2Pool, ASupersededCheckpointsRemainingBlockGoesBeforeAnOlderOne) {
    SharedL2Pool pool(TwoStateSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Lineage lineage(pool);
    const std::vector<CacheKey> next = StateKeys(Prompt{Sequence(16)});
    const SharedL2Pool::Reservation first = pool.Reserve(9, std::span{next}.first(1));
    ASSERT_GE(first.pages.front(), 0);
    ASSERT_FALSE(pool.Contains(StateKeys(lineage.a).front()));
    // a stays superseded with one State block gone: its other block goes
    // before the older, unrelated checkpoint.
    const SharedL2Pool::Reservation second = pool.Reserve(9, std::span{next}.last(1));
    ASSERT_GE(second.pages.front(), 0);
    EXPECT_FALSE(pool.Contains(StateKeys(lineage.a).back()));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.unrelated)));
    pool.Disconnect(9);
}

TEST(SharedL2Pool, APoolMissingAPromptsFirstPageHasNoHit) {
    SharedL2Pool pool(HybridSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Prompt prompt{Sequence(12)};
    std::vector<CacheKey> keys = prompt.Keys({kState});
    keys.erase(keys.begin());
    Write(pool, 1, keys, prompt.Publication());

    EXPECT_EQ(pool.Probe(2, prompt.Prefix()).num_tokens, 0);
}

TEST(SharedL2Pool, AReservationRestoresTheHeadroomOldestFirst) {
    SharedL2Pool pool(TwoStateSpecs(), kPage, 10, /*headroom_lcm_blocks=*/4);
    std::vector<Prompt> prompts;
    for (std::int32_t i = 0; i < 5; ++i) prompts.emplace_back(Sequence(4, 100 * (i + 1)));
    for (std::int32_t i = 0; i < 4; ++i) Write(pool, 1, StateKeys(prompts[i]), prompts[i].Publication());
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 2);

    const SharedL2Pool::Reservation reservation = pool.Reserve(2, StateKeys(prompts[4]));
    ASSERT_TRUE(AllGranted(reservation));
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 4);
    EXPECT_FALSE(pool.Contains(StateKeys(prompts[0]).front()));
    EXPECT_FALSE(pool.Contains(StateKeys(prompts[1]).front()));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(prompts[2])));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(prompts[3])));
    pool.Disconnect(2);
}

TEST(SharedL2Pool, AClearWaitsForHeldSlotsThenEmptiesThePool) {
    SharedL2Pool pool(HybridSpecs(), kPage, 8, /*headroom_lcm_blocks=*/0);
    const Prompt prompt{Sequence(8)};
    Write(pool, 1, prompt.Keys({kState}), prompt.Publication());
    const SharedL2Pool::Hit hit = pool.Probe(2, prompt.Prefix());
    ASSERT_GT(hit.num_tokens, 0);
    EXPECT_FALSE(pool.Clear());
    EXPECT_TRUE(HoldsAll(pool, prompt.Keys({kState})));
    pool.Release(2, hit.ticket);
    EXPECT_TRUE(pool.Clear());
    EXPECT_EQ(pool.Probe(2, prompt.Prefix()).num_tokens, 0);
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 8);
}

TEST(SharedL2Pool, AHeadroomRefillReachesItsTargetFromAnyEmptyCount) {
    SharedL2Pool pool(TwoStateSpecs(), kPage, 10, /*headroom_lcm_blocks=*/6);
    std::vector<Prompt> prompts;
    for (std::int32_t i = 0; i < 4; ++i) prompts.emplace_back(Sequence(4, 100 * (i + 1)));
    for (std::int32_t i = 0; i < 3; ++i) Write(pool, 1, StateKeys(prompts[i]), prompts[i].Publication());
    ASSERT_EQ(pool.GetStats().free_lcm_blocks, 4);

    // Two empty blocks remain after this reservation, below half the headroom:
    // the refill evicts the two oldest checkpoints to get back to six.
    const SharedL2Pool::Reservation reservation = pool.Reserve(2, StateKeys(prompts[3]));
    ASSERT_TRUE(AllGranted(reservation));
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 6);
    EXPECT_FALSE(pool.Contains(StateKeys(prompts[0]).front()));
    EXPECT_FALSE(pool.Contains(StateKeys(prompts[1]).front()));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(prompts[2])));
    pool.Disconnect(2);
}

TEST(SharedL2Pool, ATouchedEntryOutlivesAnOlderUntouchedOne) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kPage}};
    SharedL2Pool pool(specs, kPage, 2, /*headroom_lcm_blocks=*/0);
    const Prompt older{Sequence(4)}, newer{Sequence(4, 100)}, incoming{Sequence(4, 200)};
    Write(pool, 1, older.Keys({}), std::nullopt);
    Write(pool, 1, newer.Keys({}), std::nullopt);
    pool.Touch(older.Keys({}));

    const SharedL2Pool::Reservation reservation = pool.Reserve(2, incoming.Keys({}));
    EXPECT_TRUE(AllGranted(reservation));
    EXPECT_TRUE(HoldsAll(pool, older.Keys({})));
    EXPECT_FALSE(HoldsAll(pool, newer.Keys({})));
    pool.Disconnect(2);
}

}  // namespace
}  // namespace tokenspeed::test
