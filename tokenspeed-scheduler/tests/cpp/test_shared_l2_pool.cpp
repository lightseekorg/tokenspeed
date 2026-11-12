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
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/tier/shared_l2_pool.h"
#include "cache_test_access.h"

namespace tokenspeed {

struct SharedL2PoolTestAccess {
    static const CacheCoordinator& Cache(const SharedL2Pool& pool) { return pool.cache_; }
};

namespace test {
namespace {

constexpr std::int32_t kPage = 4;
constexpr std::uint32_t kFull = 0;
constexpr std::uint32_t kState = 1;
// Large enough that no test's lookups move the retention window.
constexpr std::int64_t kRetention = 1000;

// A prompt published at its end: Full KV pages, the partial page as a tail,
// and the State checkpoint after its last token.
struct Prompt {
    std::vector<std::int32_t> tokens;
    std::vector<std::string> hashes;
    std::optional<TailPublication> tail;

    explicit Prompt(std::vector<std::int32_t> input) : tokens{std::move(input)} {
        ContentHash prior;
        for (std::size_t i = 0; i + kPage <= tokens.size(); i += kPage) {
            prior = HashPrefixPage(std::span{tokens}.subspan(i, kPage), prior);
            hashes.push_back(prior.ToString());
        }
        if (tokens.size() % kPage != 0) {
            tail = TailPublication{.tail_tokens = static_cast<std::int32_t>(tokens.size() % kPage),
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(hashes.size() * kPage), prior)};
        }
    }
    ContentHash End() const { return tail ? tail->hash : ContentHash{hashes.back()}; }
    // The Full group's keys, then the checkpoint's in each of state_groups.
    std::vector<CacheKey> Keys(std::vector<std::uint32_t> state_groups) const {
        std::vector<CacheKey> keys;
        for (const std::string& hash : hashes) {
            keys.push_back(CacheKey{.group_id = kFull, .content_hash = ContentHash{hash}});
        }
        if (tail) keys.push_back(CacheKey{.group_id = kFull, .content_hash = tail->hash});
        for (std::uint32_t group : state_groups) keys.push_back(CacheKey{.group_id = group, .content_hash = End()});
        return keys;
    }
    CheckpointTree::Lineage Publication(std::uint64_t publisher) const {
        return {.tokens = tokens, .prefix_hashes = hashes, .tail = tail, .access_epoch = publisher};
    }
    CacheCoordinator::PromptPrefix Prefix() const {
        return {.prefix_hashes = hashes, .tokens = tokens, .max_tokens = static_cast<std::int32_t>(tokens.size())};
    }
};

std::vector<std::int32_t> Sequence(std::int32_t count, std::int32_t first = 1) {
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(count));
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
}

std::vector<CacheGroupSpec> HybridSpecs() {
    return {CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = kPage},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}

std::vector<CacheGroupSpec> TwoStateSpecs() {
    return {CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}

// Two Full blocks per pool parent, one State block.
struct HybridPool : SharedL2Pool {
    HybridPool(std::int32_t num_lcm_blocks, std::int32_t headroom_lcm_blocks = 0)
        : SharedL2Pool(HybridSpecs(), kPage, {2, 1}, num_lcm_blocks, headroom_lcm_blocks, kRetention,
                       /*checkpoint_retention_leaf_distance=*/0) {}
};

struct TwoStatePool : SharedL2Pool {
    TwoStatePool(std::int32_t num_lcm_blocks, std::int32_t headroom_lcm_blocks, std::int64_t retention = kRetention)
        : SharedL2Pool(TwoStateSpecs(), kPage, {1, 1}, num_lcm_blocks, headroom_lcm_blocks, retention,
                       /*checkpoint_retention_leaf_distance=*/0) {}
};

std::vector<CacheKey> StateKeys(const Prompt& prompt) {
    return {CacheKey{.group_id = 0, .content_hash = prompt.End()},
            CacheKey{.group_id = 1, .content_hash = prompt.End()}};
}

SharedL2Pool::Reservation Reserve(SharedL2Pool& pool, SharedL2Pool::ClientId client, std::uint64_t write_id,
                                  std::span<const CacheKey> keys) {
    const std::vector<std::int32_t> buckets(keys.size(), 0);
    return pool.Reserve(client, write_id, keys, buckets);
}

bool AllGranted(const SharedL2Pool::Reservation& reservation) {
    return std::ranges::all_of(reservation.pages, [](std::int32_t page) { return page >= 0; });
}

// One write of `keys`, every page granted; write ids are unique per test.
void Write(SharedL2Pool& pool, SharedL2Pool::ClientId client, const std::vector<CacheKey>& keys,
           std::optional<CheckpointTree::Lineage> publication) {
    static std::uint64_t last_write_id = 0;
    const SharedL2Pool::Reservation reservation = Reserve(pool, client, ++last_write_id, keys);
    ASSERT_TRUE(AllGranted(reservation));
    std::vector<CheckpointTree::Lineage> publications;
    if (publication) publications.push_back(std::move(*publication));
    ASSERT_TRUE(pool.Commit(client, reservation.ticket, publications));
}

bool HoldsAll(const SharedL2Pool& pool, const std::vector<CacheKey>& keys) {
    return std::ranges::all_of(keys, [&](const CacheKey& key) { return pool.Contains(key); });
}

bool HoldsNone(const SharedL2Pool& pool, const std::vector<CacheKey>& keys) {
    return std::ranges::none_of(keys, [&](const CacheKey& key) { return pool.Contains(key); });
}

TEST(SharedL2Pool, OneClientResumesWhereAnotherPublished) {
    HybridPool pool(8);
    const Prompt first{Sequence(10)};
    Write(pool, 1, first.Keys({kState}), first.Publication(1));

    std::vector<std::int32_t> longer = first.tokens;
    for (std::int32_t token : Sequence(6, 50)) longer.push_back(token);
    const SharedL2Pool::Hit hit = pool.Probe(2, /*lookup_id=*/1, Prompt{longer}.Prefix());

    EXPECT_EQ(hit.num_tokens, 10);
    EXPECT_EQ(hit.kv_tokens, 10);
    ASSERT_TRUE(hit.tail);
    EXPECT_EQ(hit.tail->hash, first.tail->hash);
    std::vector<CacheKey> keys;
    std::set<std::pair<std::uint32_t, std::int32_t>> pages;
    for (const SharedL2Pool::Slot& slot : hit.slots) {
        keys.push_back(slot.key);
        pages.emplace(slot.key.group_id, slot.page);
        EXPECT_GE(slot.page, 1);
    }
    std::vector<CacheKey> expected = first.Keys({kState});
    const auto order = [](const CacheKey& key) { return std::pair{key.group_id, key.content_hash.ToString()}; };
    std::ranges::sort(keys, {}, order);
    std::ranges::sort(expected, {}, order);
    EXPECT_EQ(keys, expected);
    EXPECT_EQ(pages.size(), hit.slots.size());
    EXPECT_EQ(pool.GetStats().pinned_blocks, 4);
    EXPECT_EQ(pool.GetStats().lookups, 1u);
    pool.Release(2, hit.ticket);
    EXPECT_EQ(pool.GetStats().pinned_blocks, 0);
    EXPECT_EQ(pool.GetStats().lookups, 0u);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(SharedL2PoolTestAccess::Cache(pool)).empty());
}

TEST(SharedL2Pool, ACheckpointWithoutItsStateBlocksGivesNoHit) {
    HybridPool pool(8);
    const Prompt first{Sequence(10)};
    Write(pool, 1, first.Keys({}), first.Publication(1));
    const SharedL2Pool::Hit hit = pool.Probe(2, 1, first.Prefix());
    EXPECT_EQ(hit.num_tokens, 0);
    EXPECT_EQ(hit.ticket, 0u);
    EXPECT_TRUE(hit.slots.empty());
    // The KV is there: a junction for the rank to cut its first chunk at.
    EXPECT_EQ(hit.kv_tokens, 10);
    EXPECT_EQ(pool.GetStats().checkpoints, 0u);
    EXPECT_EQ(pool.GetStats().pinned_blocks, 0);
}

TEST(SharedL2Pool, KvPastTheLastCheckpointIsReportedWithTheHit) {
    HybridPool pool(8);
    const Prompt checkpoint{Sequence(4)}, longer{Sequence(12)};
    Write(pool, 1, checkpoint.Keys({kState}), checkpoint.Publication(1));
    // A chunk end: two more KV pages with no State checkpoint behind them.
    std::vector<CacheKey> more = longer.Keys({});
    more.erase(more.begin());
    Write(pool, 1, more, std::nullopt);

    const SharedL2Pool::Hit hit = pool.Probe(2, 1, Prompt{Sequence(16)}.Prefix());
    EXPECT_EQ(hit.num_tokens, 4);
    EXPECT_EQ(hit.kv_tokens, 12);
    // Only the hit is pinned: the pages past the checkpoint are recomputed.
    EXPECT_EQ(hit.slots.size(), 2u);
    pool.Release(2, hit.ticket);
}

TEST(SharedL2Pool, BlocksThatArriveBeforeTheirLineageAreLinkedByALaterPublication) {
    HybridPool pool(8);
    const Prompt first{Sequence(8)}, second{Sequence(14)};
    // The first turn's blocks land without their lineage (its rank no longer
    // knew it): they match, and rank by their own use.
    Write(pool, 1, first.Keys({kState}), std::nullopt);
    EXPECT_EQ(pool.GetStats().checkpoints, 0u);
    const SharedL2Pool::Hit hit = pool.Probe(2, 1, second.Prefix());
    EXPECT_EQ(hit.num_tokens, 8);
    pool.Release(2, hit.ticket);

    // The next turn's publication runs through the first turn's pages.
    std::vector<CacheKey> more = second.Keys({kState});
    more.erase(more.begin(), more.begin() + 2);
    Write(pool, 2, more, second.Publication(1));
    EXPECT_EQ(pool.GetStats().checkpoints, 1u);
    EXPECT_EQ(pool.Probe(2, 2, Prompt{Sequence(20)}.Prefix()).num_tokens, 14);
    const CacheCoordinator& cache = SharedL2PoolTestAccess::Cache(pool);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(cache).empty());
    // Both checkpoints count under the conversation's first page.
    const BoundaryTree::Node* root =
        CacheCoordinatorTestAccess::BoundaryTreeOf(cache).Find(ContentHash{first.hashes.front()});
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->parent, nullptr);
    EXPECT_EQ(root->checkpoint_boundaries_in_subtree[BoundaryTree::kHost], 2);
    pool.Disconnect(2);
}

TEST(SharedL2Pool, APoolMissingAPromptsFirstPageHasNoHit) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(12)};
    std::vector<CacheKey> keys = prompt.Keys({kState});
    keys.erase(keys.begin());
    Write(pool, 1, keys, prompt.Publication(1));

    const SharedL2Pool::Hit hit = pool.Probe(2, 1, prompt.Prefix());
    EXPECT_EQ(hit.num_tokens, 0);
    EXPECT_EQ(hit.kv_tokens, 0);
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(SharedL2PoolTestAccess::Cache(pool)).empty());
}

TEST(SharedL2Pool, PinnedSlotsAreNotEvictedUntilReleased) {
    // Full packs two blocks per LCM block: the first prompt fills two Full and
    // one State LCM block, leaving one LCM block and one Full slot free.
    HybridPool pool(4);
    const Prompt first{Sequence(10)};
    Write(pool, 1, first.Keys({kState}), first.Publication(1));
    const SharedL2Pool::Hit hit = pool.Probe(2, 1, first.Prefix());
    ASSERT_EQ(hit.num_tokens, 10);

    const Prompt second{Sequence(16, 100)};
    const std::vector<CacheKey> second_keys = second.Keys({});
    const SharedL2Pool::Reservation crowded = Reserve(pool, 1, 10, second_keys);
    EXPECT_EQ(std::ranges::count(crowded.pages, SharedL2Pool::kNoSlot), 1);
    EXPECT_TRUE(HoldsAll(pool, first.Keys({kState})));
    EXPECT_TRUE(pool.Commit(1, crowded.ticket, {}));

    pool.Release(2, hit.ticket);
    const auto refused = std::ranges::find(crowded.pages, SharedL2Pool::kNoSlot) - crowded.pages.begin();
    const SharedL2Pool::Reservation retry = Reserve(pool, 1, 11, std::span{second_keys}.subspan(refused, 1));
    EXPECT_GE(retry.pages.front(), 1);
    EXPECT_FALSE(HoldsAll(pool, first.Keys({kState})));
    pool.Disconnect(1);
}

TEST(SharedL2Pool, EachBlockIsWrittenOnce) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(8)};
    const std::vector<CacheKey> keys = prompt.Keys({kState});

    const SharedL2Pool::Reservation first = Reserve(pool, 1, 1, keys);
    EXPECT_TRUE(AllGranted(first));
    EXPECT_EQ(pool.GetStats().reserved_blocks, keys.size());
    const SharedL2Pool::Reservation racing = Reserve(pool, 2, 1, keys);
    EXPECT_EQ(racing.ticket, 0u);
    EXPECT_EQ(std::ranges::count(racing.pages, SharedL2Pool::kHeld), static_cast<std::ptrdiff_t>(keys.size()));

    pool.Disconnect(1);
    EXPECT_EQ(pool.GetStats().reserved_blocks, 0u);
    const SharedL2Pool::Reservation second = Reserve(pool, 2, 2, keys);
    EXPECT_TRUE(AllGranted(second));
    EXPECT_TRUE(pool.Commit(2, second.ticket, std::vector{prompt.Publication(1)}));
    EXPECT_TRUE(HoldsAll(pool, keys));
    EXPECT_EQ(Reserve(pool, 1, 2, keys).ticket, 0u);

    // A key named twice in one reservation gets one page.
    const Prompt other{Sequence(4, 100)};
    const std::vector<CacheKey> twice{other.Keys({}).front(), other.Keys({}).front()};
    const SharedL2Pool::Reservation repeated = Reserve(pool, 1, 3, twice);
    EXPECT_GE(repeated.pages[0], 1);
    EXPECT_EQ(repeated.pages[1], SharedL2Pool::kHeld);
    pool.Disconnect(1);
}

TEST(SharedL2Pool, ADisconnectedClientLeavesNothingHeld) {
    HybridPool pool(8);
    const Prompt first{Sequence(8)};
    Write(pool, 1, first.Keys({kState}), first.Publication(1));
    const Prompt second{Sequence(8, 100)};
    ASSERT_GT(pool.Probe(2, 1, first.Prefix()).num_tokens, 0);
    ASSERT_NE(Reserve(pool, 2, 1, second.Keys({kState})).ticket, 0u);

    pool.Disconnect(2);

    const SharedL2Pool::Stats stats = pool.GetStats();
    EXPECT_EQ(stats.pinned_blocks, 0);
    EXPECT_EQ(stats.reserved_blocks, 0u);
    EXPECT_EQ(stats.lookups, 0u);
    EXPECT_NE(Reserve(pool, 3, 1, second.Keys({kState})).ticket, 0u);
    pool.Disconnect(3);
}

// An unrelated checkpoint, then three turns of one conversation, each
// published by a different client: the pool sees them as one lineage.
struct Lineage {
    Prompt unrelated{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};

    explicit Lineage(SharedL2Pool& pool) {
        SharedL2Pool::ClientId client = 0;
        for (const Prompt* prompt : {&unrelated, &a, &b, &c}) {
            Write(pool, ++client, StateKeys(*prompt), prompt->Publication(1));
        }
    }
};

TEST(SharedL2Pool, EvictionTakesACheckpointLaterTurnsSupersedeBeforeAnOlderLeaf) {
    for (const std::int64_t retention : {std::int64_t{0}, kRetention}) {
        SCOPED_TRACE(retention);
        TwoStatePool pool(8, /*headroom_lcm_blocks=*/0, retention);
        const Lineage lineage(pool);
        EXPECT_EQ(pool.GetStats().checkpoints, 4u);
        const std::vector<CacheKey> next = StateKeys(Prompt{Sequence(16)});
        const SharedL2Pool::Reservation reservation = Reserve(pool, 9, 1, next);
        EXPECT_TRUE(AllGranted(reservation));
        EXPECT_EQ(pool.GetStats().evicted_blocks, 2u);
        if (retention == 0) {
            // Each entry by its own use: the oldest goes.
            EXPECT_TRUE(HoldsNone(pool, StateKeys(lineage.unrelated)));
            EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.a)));
        } else {
            // A lies two turns behind the newest checkpoint of its branch.
            EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.unrelated)));
            EXPECT_TRUE(HoldsNone(pool, StateKeys(lineage.a)));
        }
        EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.b)));
        EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.c)));
        pool.Disconnect(9);
    }
}

TEST(SharedL2Pool, AnEvictedCheckpointTakesItsOtherStateBlocksAlong) {
    TwoStatePool pool(8, /*headroom_lcm_blocks=*/0);
    const Lineage lineage(pool);
    // One page is asked for; a checkpoint resumes only with every State
    // group's block, so both of A's leave.
    const std::vector<CacheKey> next = StateKeys(Prompt{Sequence(16)});
    const SharedL2Pool::Reservation reservation = Reserve(pool, 9, 1, std::span{next}.first(1));
    ASSERT_GE(reservation.pages.front(), 1);
    EXPECT_TRUE(HoldsNone(pool, StateKeys(lineage.a)));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(lineage.unrelated)));
    EXPECT_EQ(pool.GetStats().checkpoints, 3u);
    pool.Disconnect(9);
}

TEST(SharedL2Pool, OneRequestsCheckpointsAreNoTurnApart) {
    // A and B come from one request of one client (an in-forward checkpoint
    // and its prompt end), C from the next turn: A is one turn behind the
    // newest checkpoint, as B is.
    TwoStatePool pool(8, /*headroom_lcm_blocks=*/0);
    const Prompt a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)}, d{Sequence(16)};
    Write(pool, 2, StateKeys(a), a.Publication(7));
    Write(pool, 2, StateKeys(b), b.Publication(7));
    Write(pool, 2, StateKeys(c), c.Publication(9));
    const CheckpointTree& tree = SharedL2PoolTestAccess::Cache(pool).Checkpoints();
    EXPECT_EQ(tree.Distance(a.End()), 1);
    EXPECT_EQ(tree.Distance(b.End()), 1);
    EXPECT_EQ(tree.Distance(c.End()), 0);
    // Another client's request under the same epoch is another request.
    Write(pool, 3, StateKeys(d), d.Publication(9));
    EXPECT_EQ(tree.Distance(c.End()), 1);
    EXPECT_EQ(tree.Distance(a.End()), 2);
}

TEST(SharedL2Pool, AReservationRestoresTheHeadroomOldestFirst) {
    TwoStatePool pool(10, /*headroom_lcm_blocks=*/4);
    std::vector<Prompt> prompts;
    for (std::int32_t i = 0; i < 5; ++i) prompts.emplace_back(Sequence(4, 100 * (i + 1)));
    for (std::int32_t i = 0; i < 4; ++i) Write(pool, 1, StateKeys(prompts[i]), prompts[i].Publication(i));
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 2);

    const SharedL2Pool::Reservation reservation = Reserve(pool, 2, 1, StateKeys(prompts[4]));
    ASSERT_TRUE(AllGranted(reservation));
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 4);
    EXPECT_TRUE(HoldsNone(pool, StateKeys(prompts[0])));
    EXPECT_TRUE(HoldsNone(pool, StateKeys(prompts[1])));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(prompts[2])));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(prompts[3])));
    pool.Disconnect(2);
}

TEST(SharedL2Pool, AHeadroomRefillReachesItsTargetFromAnyEmptyCount) {
    TwoStatePool pool(10, /*headroom_lcm_blocks=*/6);
    std::vector<Prompt> prompts;
    for (std::int32_t i = 0; i < 4; ++i) prompts.emplace_back(Sequence(4, 100 * (i + 1)));
    for (std::int32_t i = 0; i < 3; ++i) Write(pool, 1, StateKeys(prompts[i]), prompts[i].Publication(i));
    ASSERT_EQ(pool.GetStats().free_lcm_blocks, 4);

    // Two empty blocks remain after this reservation, below half the headroom:
    // the refill evicts the two oldest checkpoints to get back to six.
    const SharedL2Pool::Reservation reservation = Reserve(pool, 2, 1, StateKeys(prompts[3]));
    ASSERT_TRUE(AllGranted(reservation));
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 6);
    EXPECT_TRUE(HoldsNone(pool, StateKeys(prompts[0])));
    EXPECT_TRUE(HoldsNone(pool, StateKeys(prompts[1])));
    EXPECT_TRUE(HoldsAll(pool, StateKeys(prompts[2])));
    pool.Disconnect(2);
}

TEST(SharedL2Pool, AClearWaitsForHeldSlotsThenEmptiesThePool) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(8)};
    Write(pool, 1, prompt.Keys({kState}), prompt.Publication(1));
    const SharedL2Pool::Hit hit = pool.Probe(2, 1, prompt.Prefix());
    ASSERT_GT(hit.num_tokens, 0);
    EXPECT_FALSE(pool.Clear());
    EXPECT_TRUE(HoldsAll(pool, prompt.Keys({kState})));
    pool.Release(2, hit.ticket);

    const Prompt writing{Sequence(4, 100)};
    const SharedL2Pool::Reservation reservation = Reserve(pool, 2, 1, writing.Keys({}));
    EXPECT_FALSE(pool.Clear());
    pool.AbandonWrite(2, 1);

    EXPECT_TRUE(pool.Clear());
    EXPECT_EQ(pool.Probe(2, 2, prompt.Prefix()).num_tokens, 0);
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 8);
    EXPECT_EQ(pool.GetStats().checkpoints, 0u);
    EXPECT_FALSE(pool.Commit(2, reservation.ticket, {}));
}

TEST(SharedL2Pool, ATouchedEntryOutlivesAnOlderUntouchedOne) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kPage}};
    SharedL2Pool pool(specs, kPage, {1}, 2, /*headroom_lcm_blocks=*/0, kRetention, 0);
    const Prompt older{Sequence(4)}, newer{Sequence(4, 100)}, incoming{Sequence(4, 200)};
    Write(pool, 1, older.Keys({}), std::nullopt);
    Write(pool, 1, newer.Keys({}), std::nullopt);
    // A key the pool does not hold is skipped.
    std::vector<CacheKey> touched = older.Keys({});
    touched.push_back(incoming.Keys({}).front());
    pool.Touch(touched);

    const SharedL2Pool::Reservation reservation = Reserve(pool, 2, 1, incoming.Keys({}));
    EXPECT_TRUE(AllGranted(reservation));
    EXPECT_TRUE(HoldsAll(pool, older.Keys({})));
    EXPECT_FALSE(HoldsAll(pool, newer.Keys({})));
    pool.Disconnect(2);
}

TEST(SharedL2Pool, AReservationKeepsTheKeysItAnswersHeld) {
    // The pool is full and its oldest entries are a prompt's two KV pages; the
    // checkpoint behind them is not there. A rank that computed the prompt
    // again offers the pages with the checkpoint: the pages are held, and the
    // room for the State block comes from the oldest entries after them.
    HybridPool pool(3);
    const Prompt prompt{Sequence(8)}, other{Sequence(16, 100)};
    Write(pool, 1, prompt.Keys({}), std::nullopt);
    Write(pool, 1, other.Keys({}), std::nullopt);
    ASSERT_EQ(pool.GetStats().free_lcm_blocks, 0);

    const SharedL2Pool::Reservation reservation = Reserve(pool, 2, 1, prompt.Keys({kState}));
    EXPECT_EQ(reservation.pages[0], SharedL2Pool::kHeld);
    EXPECT_EQ(reservation.pages[1], SharedL2Pool::kHeld);
    ASSERT_GE(reservation.pages[2], 1);
    EXPECT_TRUE(HoldsAll(pool, prompt.Keys({})));
    ASSERT_TRUE(pool.Commit(2, reservation.ticket, std::vector{prompt.Publication(1)}));
    const SharedL2Pool::Hit hit = pool.Probe(3, 1, prompt.Prefix());
    EXPECT_EQ(hit.num_tokens, 8);
    pool.Release(3, hit.ticket);

    // Held keys are refreshed also when the reservation asks for no page.
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kPage}};
    SharedL2Pool small(specs, kPage, {1}, 2, /*headroom_lcm_blocks=*/0, kRetention, 0);
    const Prompt older{Sequence(4)}, newer{Sequence(4, 100)}, incoming{Sequence(4, 200)};
    Write(small, 1, older.Keys({}), std::nullopt);
    Write(small, 1, newer.Keys({}), std::nullopt);
    EXPECT_EQ(Reserve(small, 2, 1, older.Keys({})).ticket, 0u);
    EXPECT_TRUE(AllGranted(Reserve(small, 2, 2, incoming.Keys({}))));
    EXPECT_TRUE(HoldsAll(small, older.Keys({})));
    EXPECT_TRUE(HoldsNone(small, newer.Keys({})));
    small.Disconnect(2);
}

TEST(SharedL2Pool, ReleaseAndAbandonLookupCanBeRepeated) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(10)};
    Write(pool, 1, prompt.Keys({kState}), prompt.Publication(1));

    const SharedL2Pool::Hit first = pool.Probe(2, /*lookup_id=*/1, prompt.Prefix());
    const SharedL2Pool::Hit second = pool.Probe(2, /*lookup_id=*/2, prompt.Prefix());
    ASSERT_NE(first.ticket, 0u);
    ASSERT_NE(second.ticket, first.ticket);
    EXPECT_EQ(pool.GetStats().lookups, 2u);

    pool.Release(2, first.ticket);
    pool.Release(2, first.ticket);
    pool.Release(2, /*ticket=*/12345);
    pool.Release(/*client=*/77, first.ticket);
    // The other lookup still pins the pages.
    EXPECT_EQ(pool.GetStats().lookups, 1u);
    EXPECT_EQ(pool.GetStats().pinned_blocks, 4);

    pool.AbandonLookup(2, 2);
    EXPECT_EQ(pool.GetStats().pinned_blocks, 0);
    pool.AbandonLookup(2, 2);
    // A lookup whose ticket was released is abandoned without effect.
    pool.AbandonLookup(2, 1);
    pool.Release(2, second.ticket);
    EXPECT_EQ(pool.GetStats().lookups, 0u);
    // Another client's lookup with the same id is its own.
    const SharedL2Pool::Hit other = pool.Probe(3, /*lookup_id=*/2, prompt.Prefix());
    EXPECT_NE(other.ticket, 0u);
    pool.AbandonLookup(2, 2);
    EXPECT_EQ(pool.GetStats().pinned_blocks, 4);
    pool.Disconnect(3);
}

TEST(SharedL2Pool, ALookupAbandonedBeforeItArrivesPinsNothing) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(10)};
    Write(pool, 1, prompt.Keys({kState}), prompt.Publication(1));

    pool.AbandonLookup(2, /*lookup_id=*/5);
    const SharedL2Pool::Hit late = pool.Probe(2, /*lookup_id=*/5, prompt.Prefix());
    EXPECT_EQ(late.ticket, 0u);
    EXPECT_EQ(late.num_tokens, 0);
    EXPECT_TRUE(late.slots.empty());
    EXPECT_EQ(pool.GetStats().pinned_blocks, 0);
    // Only that lookup: the next one hits.
    const SharedL2Pool::Hit next = pool.Probe(2, /*lookup_id=*/6, prompt.Prefix());
    EXPECT_EQ(next.num_tokens, 10);
    pool.Release(2, next.ticket);

    // The newest abandoned ids are remembered; older ones are forgotten.
    for (std::uint64_t id = 100; id < 100 + SharedL2Pool::kMaxAbandoned + 1; ++id) pool.AbandonLookup(2, id);
    EXPECT_NE(pool.Probe(2, 100, prompt.Prefix()).ticket, 0u);
    EXPECT_EQ(pool.Probe(2, 101, prompt.Prefix()).ticket, 0u);
    pool.Disconnect(2);
}

TEST(SharedL2Pool, AnAbandonedWriteFreesItsPagesAndKeys) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(8)};
    const std::vector<CacheKey> keys = prompt.Keys({kState});

    const SharedL2Pool::Reservation reservation = Reserve(pool, 1, /*write_id=*/7, keys);
    ASSERT_TRUE(AllGranted(reservation));
    EXPECT_LT(pool.GetStats().free_lcm_blocks, 8);
    pool.AbandonWrite(1, 7);
    pool.AbandonWrite(1, 7);
    EXPECT_EQ(pool.GetStats().reserved_blocks, 0u);
    EXPECT_EQ(pool.GetStats().free_lcm_blocks, 8);
    EXPECT_TRUE(HoldsNone(pool, keys));
    // Its late commit finds nothing reserved; another client writes the keys.
    EXPECT_FALSE(pool.Commit(1, reservation.ticket, std::vector{prompt.Publication(1)}));
    EXPECT_TRUE(HoldsNone(pool, keys));
    Write(pool, 2, keys, prompt.Publication(1));

    // A committed write is not undone.
    pool.AbandonWrite(2, 0);
    const SharedL2Pool::Reservation committed = Reserve(pool, 2, /*write_id=*/8, Prompt{Sequence(4, 100)}.Keys({}));
    ASSERT_TRUE(pool.Commit(2, committed.ticket, {}));
    pool.AbandonWrite(2, 8);
    EXPECT_TRUE(pool.Contains(Prompt{Sequence(4, 100)}.Keys({}).front()));
    EXPECT_FALSE(pool.Commit(2, committed.ticket, {}));
    EXPECT_FALSE(pool.Commit(2, /*ticket=*/0, {}));
    EXPECT_FALSE(pool.Commit(/*client=*/77, committed.ticket, {}));
}

TEST(SharedL2Pool, AWriteAbandonedBeforeItArrivesReservesNothing) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(8)};
    const std::vector<CacheKey> keys = prompt.Keys({kState});

    pool.AbandonWrite(1, /*write_id=*/3);
    const SharedL2Pool::Reservation late = Reserve(pool, 1, 3, keys);
    EXPECT_EQ(late.ticket, 0u);
    EXPECT_EQ(std::ranges::count(late.pages, SharedL2Pool::kHeld), static_cast<std::ptrdiff_t>(keys.size()));
    EXPECT_EQ(pool.GetStats().reserved_blocks, 0u);
    EXPECT_TRUE(AllGranted(Reserve(pool, 1, 4, keys)));
    pool.Disconnect(1);
}

TEST(SharedL2Pool, ABucketIsCarriedPerKeyAndCheckedAgainstItsGroup) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(10)};
    const std::vector<CacheKey> keys = prompt.Keys({kState});
    // Replicated groups have one bucket.
    std::vector<std::int32_t> buckets(keys.size(), 0);
    buckets.back() = 1;
    EXPECT_THROW(pool.Reserve(1, 1, keys, buckets), std::invalid_argument);
    EXPECT_THROW(pool.Reserve(1, 1, keys, std::span{buckets}.first(1)), std::invalid_argument);
    EXPECT_EQ(pool.GetStats().reserved_blocks, 0u);

    Write(pool, 1, keys, prompt.Publication(1));
    const SharedL2Pool::Hit hit = pool.Probe(2, 1, prompt.Prefix());
    ASSERT_EQ(hit.slots.size(), keys.size());
    for (const SharedL2Pool::Slot& slot : hit.slots) EXPECT_EQ(slot.bucket, 0);
    pool.Release(2, hit.ticket);

    std::vector<CacheGroupSpec> sharded = HybridSpecs();
    sharded.front().shard_count = 2;
    EXPECT_THROW(SharedL2Pool(sharded, kPage, {2, 1}, 8, 0, kRetention, 0), std::invalid_argument);
    EXPECT_THROW(SharedL2Pool(HybridSpecs(), kPage, {2}, 8, 0, kRetention, 0), std::invalid_argument);
}

TEST(SharedL2Pool, MisusedCallsAreRefusedBeforeAnythingChanges) {
    HybridPool pool(8);
    const Prompt prompt{Sequence(10)};
    const std::vector<CacheKey> keys = prompt.Keys({kState});
    const SharedL2Pool::Reservation reservation = Reserve(pool, 1, 1, keys);

    // Tokens that do not cover the pages and the tail.
    CheckpointTree::Lineage short_tokens = prompt.Publication(1);
    short_tokens.tokens.pop_back();
    EXPECT_THROW(pool.Commit(1, reservation.ticket, std::vector{short_tokens}), std::invalid_argument);
    EXPECT_EQ(pool.GetStats().reserved_blocks, keys.size());
    EXPECT_TRUE(pool.Commit(1, reservation.ticket, std::vector{prompt.Publication(1)}));

    const std::vector<CacheKey> unknown_group{CacheKey{.group_id = 2, .content_hash = prompt.End()}};
    EXPECT_THROW(Reserve(pool, 1, 2, unknown_group), std::invalid_argument);
    EXPECT_THROW(pool.Touch(unknown_group), std::invalid_argument);
    const std::vector<CacheKey> other_namespace{
        CacheKey{.namespace_id = 1, .group_id = kFull, .content_hash = prompt.End()}};
    EXPECT_THROW(Reserve(pool, 1, 2, other_namespace), std::invalid_argument);
    EXPECT_THROW(HybridPool(8, /*headroom_lcm_blocks=*/8), std::invalid_argument);
    // A group's pages are int32 block ids: a pool one page past that range is
    // refused, so the headroom's blocks, fewer than the pool's, fit as well.
    EXPECT_THROW(SharedL2Pool(HybridSpecs(), kPage, {1 << 16, 1}, 1 << 15, 0, kRetention, 0), std::runtime_error);

    // A lookup whose tokens do not reach its bound, or whose bound is negative.
    CacheCoordinator::PromptPrefix lookup = prompt.Prefix();
    lookup.max_tokens += 1;
    EXPECT_THROW(pool.Probe(2, 1, lookup), std::invalid_argument);
    lookup.max_tokens = -kPage;
    EXPECT_THROW(pool.Probe(2, 1, lookup), std::invalid_argument);
    EXPECT_EQ(pool.GetStats().lookups, 0u);
    lookup.max_tokens = static_cast<std::int32_t>(prompt.tokens.size());
    const SharedL2Pool::Hit hit = pool.Probe(2, 1, lookup);
    EXPECT_EQ(hit.num_tokens, 10);
    pool.Release(2, hit.ticket);
}

TEST(SharedL2Pool, EveryLookupCountsOneRequestOfTheRetentionWindow) {
    // The window is four requests. Until four lookups have arrived, every
    // entry counts as used within it, and A, which B supersedes, leaves before
    // the oldest entry. From the fourth on A ranks as if last used four
    // requests ago, or at its own use, which is older: the oldest entry leaves
    // first. A lookup counts whether it hits or not.
    for (const std::uint64_t lookups : {std::uint64_t{3}, std::uint64_t{4}}) {
        SCOPED_TRACE(lookups);
        TwoStatePool pool(8, /*headroom_lcm_blocks=*/0, /*retention=*/4);
        const Prompt stale{{90, 91, 92, 93}}, recent{{80, 81, 82, 83}}, a{Sequence(4)}, b{Sequence(8)};
        std::uint64_t publisher = 0;
        for (const Prompt* prompt : {&stale, &recent, &a, &b}) {
            Write(pool, 1, StateKeys(*prompt), prompt->Publication(++publisher));
        }
        for (std::uint64_t lookup = 1; lookup <= lookups; ++lookup) {
            if (lookup % 2 == 1) {
                EXPECT_EQ(pool.Probe(2, lookup, Prompt{Sequence(4, 500)}.Prefix()).ticket, 0u);
                continue;
            }
            const SharedL2Pool::Hit hit = pool.Probe(2, lookup, recent.Prefix());
            ASSERT_NE(hit.ticket, 0u);
            pool.Release(2, hit.ticket);
        }
        ASSERT_TRUE(AllGranted(Reserve(pool, 3, 1, StateKeys(Prompt{Sequence(4, 600)}))));
        EXPECT_EQ(HoldsNone(pool, StateKeys(a)), lookups < 4);
        EXPECT_EQ(HoldsNone(pool, StateKeys(stale)), lookups == 4);
        EXPECT_TRUE(HoldsAll(pool, StateKeys(recent)));
        EXPECT_TRUE(HoldsAll(pool, StateKeys(b)));
        pool.Disconnect(3);
    }
}

// A Host tier of a rank's own coordinator, fed the pool's sequence: a new
// request takes the Host blocks its checkpoint will be stored in (the pool's
// reservation; none for a checkpoint the Host holds), publishes the
// checkpoint on the Device, and its store lands (the pool's commit); the
// Device is then cleared, so the Host tier decides alone, as the pool does.
struct LocalHostTier {
    BlockPool device{64, {1, 1}};
    BlockPool host;
    CacheCoordinator cache;

    LocalHostTier(std::int32_t num_lcm_blocks, std::int64_t retention, std::int32_t leaf_distance)
        : host{num_lcm_blocks, {1, 1}},
          cache{MakeCoordinator(TwoStateSpecs(), kPage, device, false, &host, nullptr, false, false, retention,
                                leaf_distance)} {}

    void Publish(const Prompt& point) {
        const std::vector<CacheKey> keys = StateKeys(point);
        CacheCoordinator::HostAllocationBatch batch;
        if (!HoldsAll(point)) {
            const std::vector<std::uint32_t> groups{0, 1};
            const std::vector<std::int32_t> buckets{0, 0};
            batch = cache.AcquireHostBlocks(groups, buckets);
            ASSERT_EQ(batch.stats.allocated, keys.size());
        }
        std::vector<BlockTable> tables(2);
        auto admission = AdmitForTest(cache, tables, 0);
        ASSERT_TRUE(admission);
        cache.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.tokens = point.tokens,
                                                              .prefix_hashes = point.hashes,
                                                              .tail = point.tail,
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = static_cast<std::int32_t>(point.tokens.size())},
            admission->access_epoch);
        cache.Free(tables);
        for (std::size_t i = 0; i < batch.blocks.size(); ++i) {
            cache.CacheHostBlock(batch.blocks[i], keys[i], std::nullopt);
        }
        if (point.tail) {
            cache.CacheHostTail(point.hashes.empty() ? ContentHash{} : ContentHash{point.hashes.back()}, *point.tail);
        }
        admission.reset();
        ASSERT_TRUE(cache.ClearDeviceCache());
    }
    // A new request resumes at the probe's Host hit and leaves.
    void Resume(const CacheCoordinator::PromptPrefix& prefix) {
        {
            std::vector<BlockTable> tables(2);
            auto admission =
                AdmitForTest(cache, tables, cache.ProbePromptPrefix(prefix), GroupDemand{.extent = DenseGrowth{0}});
            ASSERT_TRUE(admission);
            admission.reset();
            cache.Free(tables);
        }
        ASSERT_TRUE(cache.ClearDeviceCache());
    }
    bool HoldsAll(const Prompt& point) const {
        return std::ranges::all_of(StateKeys(point),
                                   [&](const CacheKey& key) { return cache.ContainsHostCachedBlock(key); });
    }
    bool HoldsNone(const Prompt& point) const {
        return std::ranges::none_of(StateKeys(point),
                                    [&](const CacheKey& key) { return cache.ContainsHostCachedBlock(key); });
    }
};

// The same publications, hits, touches and checkpoints computed and offered
// again reach a pool and a rank's own Host tier; whatever room either then
// makes, both keep the same checkpoints. The windows compared never move.
TEST(SharedL2Pool, EvictsAsARanksOwnHostTierGivenTheSameSequence) {
    constexpr std::int32_t kBlocks = 24;
    std::size_t evictions_compared = 0;
    std::size_t reoffers_held = 0;
    for (const std::int64_t retention : {std::int64_t{0}, kRetention}) {
        for (const std::int32_t leaf_distance : {0, 1}) {
            for (const std::uint32_t seed : {1u, 2u, 3u, 4u, 5u}) {
                SCOPED_TRACE(testing::Message()
                             << "retention " << retention << ", leaf distance " << leaf_distance << ", seed " << seed);
                SharedL2Pool pool(TwoStateSpecs(), kPage, {1, 1}, kBlocks, /*headroom_lcm_blocks=*/0, retention,
                                  leaf_distance);
                LocalHostTier local(kBlocks, retention, leaf_distance);
                // The local tier's blocks held as the pool's reservations hold theirs.
                std::vector<CacheBlockRef> local_reserved;
                std::mt19937 random(seed);
                // Three conversations; a turn extends an earlier turn of its
                // conversation by up to a page and a half.
                std::vector<Prompt> published;
                std::vector<std::vector<std::size_t>> conversations(3);
                std::uint64_t request = 0;
                const auto compare = [&] {
                    for (const Prompt& point : published) {
                        ASSERT_EQ(HoldsAll(pool, StateKeys(point)), local.HoldsAll(point)) << point.tokens.size();
                        ASSERT_EQ(HoldsNone(pool, StateKeys(point)), local.HoldsNone(point)) << point.tokens.size();
                    }
                };
                for (std::int32_t step = 0; step < 40; ++step) {
                    const std::size_t conversation = random() % conversations.size();
                    std::vector<std::size_t>& turns = conversations[conversation];
                    const std::uint32_t action = random() % 12;
                    if (action < 6 || turns.empty()) {
                        std::vector<std::int32_t> tokens =
                            turns.empty() ? Sequence(4, static_cast<std::int32_t>(1000 * (conversation + 1)))
                                          : published[turns[random() % turns.size()]].tokens;
                        const std::int32_t grown = 1 + static_cast<std::int32_t>(random() % 6);
                        for (std::int32_t i = 0; i < grown; ++i) {
                            tokens.push_back(100000 + 100 * step + i);
                        }
                        const Prompt point{std::move(tokens)};
                        const SharedL2Pool::Reservation reservation = Reserve(pool, 1, ++request, StateKeys(point));
                        ASSERT_TRUE(AllGranted(reservation));
                        ASSERT_TRUE(pool.Commit(1, reservation.ticket, std::vector{point.Publication(request)}));
                        local.Publish(point);
                        turns.push_back(published.size());
                        published.push_back(point);
                    } else if (action < 9) {
                        // A new request resumes from a published turn, if it is still held.
                        const Prompt& point = published[turns[random() % turns.size()]];
                        std::vector<std::int32_t> longer = point.tokens;
                        longer.push_back(7);
                        const Prompt prompt{std::move(longer)};
                        CacheCoordinator::PromptPrefix prefix = prompt.Prefix();
                        prefix.max_tokens = static_cast<std::int32_t>(point.tokens.size());
                        const SharedL2Pool::Hit hit = pool.Probe(2, ++request, prefix);
                        pool.Release(2, hit.ticket);
                        ASSERT_EQ(hit.num_tokens, local.cache.ProbePromptPrefix(prefix).host.num_common_tokens);
                        if (hit.num_tokens > 0) {
                            local.Resume(prefix);
                        }
                    } else if (action == 9) {
                        const std::vector<CacheKey> keys = StateKeys(published[turns[random() % turns.size()]]);
                        pool.Touch(keys);
                        local.cache.TouchHostBlocks(keys);
                    } else {
                        // A request computes a published turn again and offers its
                        // checkpoint again: held, it is refreshed; evicted
                        // meanwhile, it is written again.
                        const Prompt& point = published[turns[random() % turns.size()]];
                        const SharedL2Pool::Reservation reservation = Reserve(pool, 1, ++request, StateKeys(point));
                        ASSERT_EQ(reservation.ticket == 0, local.HoldsAll(point));
                        if (reservation.ticket != 0) {
                            ASSERT_TRUE(AllGranted(reservation));
                            ASSERT_TRUE(pool.Commit(1, reservation.ticket, std::vector{point.Publication(request)}));
                        }
                        reoffers_held += reservation.ticket == 0 ? 1 : 0;
                        local.Publish(point);
                    }
                    compare();
                }
                evictions_compared += pool.GetStats().evicted_blocks;
                // Then room is made two blocks at a time until nothing is left.
                for (std::int32_t round = 0; round < kBlocks / 2; ++round) {
                    ASSERT_TRUE(
                        AllGranted(Reserve(pool, 3, ++request, StateKeys(Prompt{Sequence(4, 900000 + 10 * round)}))));
                    const std::vector<std::uint32_t> groups{0, 1};
                    const std::vector<std::int32_t> buckets{0, 0};
                    CacheCoordinator::HostAllocationBatch batch = local.cache.AcquireHostBlocks(groups, buckets);
                    ASSERT_EQ(batch.stats.allocated, 2u);
                    for (CacheBlockRef& block : batch.blocks) {
                        local_reserved.push_back(std::move(block));
                    }
                    compare();
                }
                EXPECT_EQ(pool.GetStats().cached_blocks, 0);
                evictions_compared += pool.GetStats().evicted_blocks;
                EXPECT_EQ(pool.GetStats().evicted_blocks, local.cache.HostEvictions().lru.blocks);
                pool.Disconnect(3);
            }
        }
    }
    // The sequences did put both under pressure, and offered held checkpoints again.
    EXPECT_GT(evictions_compared, 400u);
    EXPECT_GT(reoffers_held, 40u);
}

// A Device-only coordinator and a Host-only one holding the same entries:
// the Host-only probe of an empty Device is the Device's own probe.
struct TierPair {
    BlockPool device{32, {2, 1}};
    BlockPool unused_device{1, {2, 1}};
    BlockPool host{32, {2, 1}};
    CacheCoordinator on_device{
        MakeCoordinator(HybridSpecs(), kPage, device, false, nullptr, nullptr, false, false, kRetention)};
    CacheCoordinator on_host{
        MakeCoordinator(HybridSpecs(), kPage, unused_device, false, &host, nullptr, false, false, kRetention)};

    // Publishes the prompt's pages and tail on both, with its checkpoint when
    // `state`: a chunk end otherwise.
    void Publish(const Prompt& prompt, bool state) {
        std::vector<BlockTable> tables(2);
        auto admission = AdmitForTest(on_device, tables, static_cast<std::int32_t>(prompt.tokens.size()));
        ASSERT_TRUE(admission);
        on_device.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.tokens = prompt.tokens,
                                                              .prefix_hashes = prompt.hashes,
                                                              .tail = prompt.tail,
                                                              .publish_state = state,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = static_cast<std::int32_t>(prompt.tokens.size())},
            admission->access_epoch);
        on_device.Free(tables);
        for (const CacheKey& key :
             prompt.Keys(state ? std::vector<std::uint32_t>{kState} : std::vector<std::uint32_t>{})) {
            if (on_host.ContainsHostCachedBlock(key)) continue;
            CacheBlockRef block = on_host.AcquireHostBlock(key.group_id, 0);
            ASSERT_TRUE(block);
            on_host.CacheHostBlock(block, key, std::nullopt);
        }
        if (prompt.tail) {
            on_host.CacheHostTail(prompt.hashes.empty() ? ContentHash{} : ContentHash{prompt.hashes.back()},
                                  *prompt.tail);
        }
    }
};

TEST(SharedL2Pool, AProbeOfAnEmptyDeviceMatchesTheHostTierAlone) {
    TierPair tiers;
    // A conversation with a checkpoint inside a page, a page-aligned one, a
    // chunk end past them (KV with no checkpoint) and a branch.
    tiers.Publish(Prompt{Sequence(6)}, /*state=*/true);
    tiers.Publish(Prompt{Sequence(12)}, /*state=*/true);
    tiers.Publish(Prompt{Sequence(22)}, /*state=*/false);
    std::vector<std::int32_t> branch = Sequence(8);
    for (std::int32_t token : Sequence(7, 300)) branch.push_back(token);
    tiers.Publish(Prompt{branch}, /*state=*/true);
    tiers.Publish(Prompt{Sequence(9, 700)}, /*state=*/false);

    std::vector<std::vector<std::int32_t>> queries{Sequence(30), Sequence(22),      Sequence(13),    Sequence(12),
                                                   Sequence(7),  Sequence(6),       Sequence(5),     Sequence(3),
                                                   branch,       Sequence(20, 700), Sequence(8, 900)};
    std::vector<std::int32_t> past_branch = branch;
    past_branch.push_back(1);
    queries.push_back(past_branch);
    std::vector<std::int32_t> diverging = Sequence(10);
    diverging.push_back(999);
    queries.push_back(diverging);
    for (const std::vector<std::int32_t>& tokens : queries) {
        for (const std::int32_t bound :
             {static_cast<std::int32_t>(tokens.size()), static_cast<std::int32_t>(tokens.size()) - 1}) {
            SCOPED_TRACE(testing::Message()
                         << tokens.size() << " tokens from " << tokens.front() << ", bound " << bound);
            const Prompt prompt{tokens};
            CacheCoordinator::PromptPrefix prefix = prompt.Prefix();
            prefix.max_tokens = bound;
            const CacheCoordinator::PrefixProbe::Tier device = tiers.on_device.ProbePromptPrefix(prefix).device;
            const CacheCoordinator::PrefixProbe host_probe = tiers.on_host.ProbePromptPrefix(prefix);
            const CacheCoordinator::PrefixProbe::Tier& host = host_probe.host;
            EXPECT_EQ(host_probe.device.num_common_tokens, 0);
            EXPECT_EQ(host.num_common_tokens, device.num_common_tokens);
            EXPECT_EQ(host.kv_tokens, device.kv_tokens);
            if (device.num_common_tokens == 0) {
                continue;
            }
            EXPECT_EQ(host.full_page_tokens, device.full_page_tokens);
            ASSERT_EQ(host.tail.has_value(), device.tail.has_value());
            if (device.tail) {
                EXPECT_EQ(host.tail->tail_tokens, device.tail->tail_tokens);
                EXPECT_EQ(host.tail->hash, device.tail->hash);
                EXPECT_EQ(host.tail->group_keys, device.tail->group_keys);
            }
            ASSERT_EQ(host.per_group.size(), device.per_group.size());
            for (std::size_t group = 0; group < device.per_group.size(); ++group) {
                EXPECT_EQ(host.per_group[group].hits, device.per_group[group].hits) << "group " << group;
                if (device.tail) {
                    EXPECT_EQ(host.tail->per_group[group].hits, device.tail->per_group[group].hits)
                        << "group " << group;
                }
            }
        }
    }
}

// A rank's store candidates, which its pool writes take a bounded number of
// at a time: they leave oldest first across takes, new ones queue behind
// those left, and a flush drops the ones no take took.
TEST(SharedL2Pool, BoundedTakesOfARanksStoreCandidatesKeepTheirOrder) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kPage}};
    BlockPool device{16, {1}};
    BlockPool host{4, {1}};
    CacheCoordinator cache{MakeCoordinator(specs, kPage, device, false, &host, nullptr,
                                           /*stream_device_cache_to_host=*/true, false, kRetention)};
    const Prompt prompt{Sequence(10 * kPage)};
    std::vector<BlockTable> tables(1);
    auto admission = AdmitForTest(cache, tables, 10 * kPage);
    ASSERT_TRUE(admission);
    cache.CacheCompletedBlocks(
        tables,
        RequestProgress{.completed_pages = CompletedPages{.tokens = prompt.tokens,
                                                          .prefix_hashes = prompt.hashes,
                                                          .boundary_kind = CacheBoundaryKind::kEndpoint},
                        .num_computed_tokens = 10 * kPage},
        admission->access_epoch);
    cache.Free(tables);
    admission.reset();
    ASSERT_FALSE(cache.HasPendingStores());

    std::vector<std::string> taken;
    const auto take = [&](std::size_t max_candidates) {
        for (const CacheCoordinator::StoreCandidate& candidate : cache.TakePendingStores(max_candidates)) {
            taken.push_back(candidate.key.content_hash.ToString());
        }
    };
    cache.QueueCachedBlocksForStore(prompt.hashes);
    take(3);
    cache.QueueCachedBlocksForStore(std::span{prompt.hashes}.first(2));
    take(3);
    take(2);
    EXPECT_EQ(taken, std::vector<std::string>(prompt.hashes.begin(), prompt.hashes.begin() + 8));
    EXPECT_TRUE(cache.HasPendingStores());

    // Pages 8 and 9 and the two queued again are left.
    ASSERT_TRUE(cache.ClearDeviceCache());
    EXPECT_EQ(cache.StoreCandidatesDroppedByFlush(), 4u);
    EXPECT_FALSE(cache.HasPendingStores());
    EXPECT_TRUE(cache.TakePendingStores(1).empty());
}

// A rank's Host tier standing for the pool, each waiting prompt's answer
// applied as SharedL2Proxy applies it: the blocks the tier lacks registered,
// the tail recorded, every block held for the prompt. Tails are kept per
// preceding page up to TailTable::kMaxTailsPerPrior, as on a rank's own
// tiers: a ninth prompt ending inside the same page drops the hint of the one
// answered first, which then matches nothing at its admission. Its blocks
// stay while it holds them and leave once it lets go.
TEST(SharedL2Pool, ATailPastThePerPageCapCostsTheOldestWaitingPromptItsHit) {
    BlockPool device{4, {2, 1}};
    BlockPool host{32, {2, 1}};
    CacheCoordinator rank{
        MakeCoordinator(HybridSpecs(), kPage, device, false, &host, nullptr, false, false, kRetention)};
    std::vector<Prompt> prompts;
    std::vector<std::vector<CacheBlockRef>> held;
    for (std::int32_t i = 0; i <= TailTable::kMaxTailsPerPrior; ++i) {
        std::vector<std::int32_t> tokens = Sequence(kPage);
        tokens.push_back(100 + i);
        const Prompt& prompt = prompts.emplace_back(std::move(tokens));
        std::vector<CacheBlockRef>& blocks = held.emplace_back();
        for (const CacheKey& key : prompt.Keys({kState})) {
            CacheBlockRef block = rank.AcquireHostCachedBlock(key);
            if (!block) {
                block = rank.AcquireHostBlock(key.group_id, 0);
                ASSERT_TRUE(block);
                rank.CacheHostBlock(block, key, std::nullopt);
            }
            blocks.push_back(std::move(block));
        }
        rank.CacheHostTail(ContentHash{prompt.hashes.back()}, *prompt.tail);
    }

    for (std::size_t i = 1; i < prompts.size(); ++i) {
        EXPECT_EQ(rank.ProbePromptPrefix(prompts[i].Prefix()).host.num_common_tokens, kPage + 1) << i;
    }
    const Prompt& oldest = prompts.front();
    const CacheCoordinator::PrefixProbe probe = rank.ProbePromptPrefix(oldest.Prefix());
    EXPECT_EQ(probe.host.num_common_tokens, 0);
    EXPECT_EQ(probe.host.kv_tokens, kPage);
    const std::vector<CacheKey> keys = oldest.Keys({kState});
    EXPECT_TRUE(std::ranges::all_of(keys, [&](const CacheKey& key) { return rank.ContainsHostCachedBlock(key); }));
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(rank).empty());

    held.front().clear();
    // The page is the other prompts' too; the tail's block and the checkpoint go.
    EXPECT_FALSE(rank.DropHostEntry(keys[0]));
    EXPECT_TRUE(rank.DropHostEntry(keys[1]));
    EXPECT_TRUE(rank.DropHostEntry(keys[2]));
    EXPECT_FALSE(rank.ContainsHostCachedBlock(keys[1]));
    EXPECT_FALSE(rank.ContainsHostCachedBlock(keys[2]));
    EXPECT_TRUE(CacheCoordinatorTestAccess::BoundaryTreeMismatches(rank).empty());
    held.clear();
}

}  // namespace
}  // namespace test
}  // namespace tokenspeed
