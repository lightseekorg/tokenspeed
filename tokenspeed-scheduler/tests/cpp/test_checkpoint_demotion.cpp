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

// Superseded checkpoint demotion
// (SchedulerConfig::demote_superseded_state_checkpoints): a finish that
// publishes an in-forward State checkpoint demotes the one its conversation
// published two turns back -- the prefix index's dating, the checkpoint
// lineage, the coordinator's keys, write-back drops and Host placement, and
// whole P- and fused-role conversations.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache/prefix/prefix_index.h"
#include "cache/tier/transfer_manager.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"
#include "scheduler/checkpoint_lineage.h"

namespace tokenspeed::test {
namespace {

CacheKey Key(const std::string& hash, std::uint32_t group = 0) {
    return CacheKey{.group_id = group, .content_hash = ContentHash{hash}, .page_offset = 0};
}

// The names of `names`' entries in `pool`'s eviction order.
std::vector<std::string> EvictionOrder(const PrefixCacheIndex& index, const BlockPool& pool,
                                       const std::vector<std::pair<CacheBlockLocation, std::string>>& names) {
    std::vector<std::string> order;
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    while (index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch)) {
        for (const auto& candidate : batch) {
            for (const auto& [location, name] : names) {
                if (location == candidate.location) {
                    order.push_back(name);
                }
            }
        }
        batch.clear();
    }
    return order;
}

TEST(CheckpointDemotionIndexTest, DemoteDatesBackNeverForwardSkipsHeldEntriesAndLeadsTheEvictionOrder) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(0);
    std::vector<std::pair<CacheBlockLocation, std::string>> names;
    for (const auto& [name, epoch] :
         std::vector<std::pair<std::string, std::uint64_t>>{{"a", 10}, {"b", 20}, {"c", 30}}) {
        CacheBlockRef block = pool.AcquireBlock(0);
        names.emplace_back(block->Location(), name);
        index.Register(pool, block, Key(name), epoch, -1, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    }
    CacheBlockRef held = pool.AcquireBlock(0);
    index.Register(pool, held, Key("held"), 40, -1, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    EXPECT_FALSE(index.Demote(pool, Key("held"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/0))
        << "a request holds it";

    EXPECT_TRUE(index.Demote(pool, Key("c"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/0));
    EXPECT_FALSE(index.Demote(pool, Key("c"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/0))
        << "demoted already";
    EXPECT_FALSE(index.Demote(pool, Key("a"), 15, /*write_back_holds=*/0)) << "never dates forward";
    EXPECT_FALSE(index.Demote(pool, Key("missing"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/0));
    EXPECT_EQ(index.Inspect(pool, Key("c"))->metadata.eviction.last_access_epoch,
              CacheCoordinator::kDemotedAccessEpoch);
    EXPECT_TRUE(index.Inspect(pool, Key("c"))->metadata.demoted);
    EXPECT_FALSE(index.Inspect(pool, Key("a"))->metadata.demoted);
    EXPECT_EQ(index.Inspect(pool, Key("c"))->metadata.max_access_epoch, 30u) << "the newest touch stays";
    EXPECT_EQ(EvictionOrder(index, pool, names), (std::vector<std::string>{"c", "a", "b"}));

    // A later publication or hit dates it forward again and ends the demotion.
    CacheBlockRef again = index.Find(pool, Key("c"));
    index.Register(pool, again, Key("c"), 50, -1, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    again.reset();
    EXPECT_FALSE(index.Inspect(pool, Key("c"))->metadata.demoted);
    EXPECT_EQ(EvictionOrder(index, pool, names), (std::vector<std::string>{"a", "b", "c"}));
}

// A session's demotion dates its entries back without marking them demoted.
// A demotion marks one a session's demotion dated back already, and any touch
// ends the mark, even one the session's demotion keeps at the demoted epoch.
// MarkDemoted marks an entry at the epoch, whatever holds it, and no other.
TEST(CheckpointDemotionIndexTest, OnlyADemotionMarksAnEntryAndATouchEndsTheMark) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(0);
    const auto session = std::make_shared<SessionCacheTag>();
    session->id = "s";
    for (const char* name : {"a", "b"}) {
        CacheBlockRef block = pool.AcquireBlock(0);
        index.Register(pool, block, Key(name), 10, -1, CacheBoundaryKind::kEndpoint, nullptr, session);
    }
    CacheBlockRef dated = pool.AcquireBlock(0);
    index.Register(pool, dated, Key("dated"), 20, -1, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    dated.reset();
    const auto demoted = [&](const char* name) { return index.Inspect(pool, Key(name))->metadata.demoted; };
    session->demoted = true;
    ASSERT_EQ(index.DemoteSession(pool, session), 2u);
    EXPECT_EQ(index.Inspect(pool, Key("a"))->metadata.eviction.last_access_epoch,
              CacheCoordinator::kDemotedAccessEpoch);
    EXPECT_FALSE(demoted("a"));
    EXPECT_FALSE(demoted("b"));

    EXPECT_TRUE(index.Demote(pool, Key("a"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/0));
    EXPECT_TRUE(demoted("a"));
    PrefixMatch hit =
        index.AcquireMatched(pool, std::vector<CacheKey>{Key("a")}, 0, GroupPrefixProbe{.hits = {1}}, 30, session);
    EXPECT_FALSE(demoted("a"));
    EXPECT_EQ(index.Inspect(pool, Key("a"))->metadata.eviction.last_access_epoch, CacheCoordinator::kDemotedAccessEpoch)
        << "the session is demoted still";

    CacheBlockRef held = index.Find(pool, Key("b"));
    index.MarkDemoted(pool, Key("b"), CacheCoordinator::kDemotedAccessEpoch);
    index.MarkDemoted(pool, Key("dated"), CacheCoordinator::kDemotedAccessEpoch);
    EXPECT_TRUE(demoted("b"));
    EXPECT_FALSE(demoted("dated"));
    EXPECT_EQ(index.Inspect(pool, Key("dated"))->metadata.eviction.last_access_epoch, 20u);
}

// The holds of an eviction write-back that publishes into the entry do not
// keep it from a demotion; any other hold does.
TEST(CheckpointDemotionIndexTest, DemoteDiscountsOnlyTheWriteBackHolds) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(0);
    CacheBlockRef write_back = pool.AcquireBlock(0);
    index.Register(pool, write_back, Key("h"), 10, -1, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    CacheBlockRef other = write_back;
    EXPECT_FALSE(index.Demote(pool, Key("h"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/1))
        << "another hold beside the write-back's";
    other.reset();
    EXPECT_FALSE(index.Demote(pool, Key("h"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/0));
    EXPECT_TRUE(index.Demote(pool, Key("h"), CacheCoordinator::kDemotedAccessEpoch, /*write_back_holds=*/1));
    EXPECT_EQ(index.Inspect(pool, Key("h"))->metadata.eviction.last_access_epoch,
              CacheCoordinator::kDemotedAccessEpoch);
}

TEST(CheckpointLineageTest, RecordsPublicationsAndCountsResumesUpToTwo) {
    CheckpointLineage lineage;
    EXPECT_EQ(lineage.FindPublication(ContentHash{"c"}), nullptr);
    lineage.RecordPublication(ContentHash{"c"}, {.resumed_tokens = 20, .access_epoch = 7});
    ASSERT_NE(lineage.FindPublication(ContentHash{"c"}), nullptr);
    EXPECT_EQ(lineage.FindPublication(ContentHash{"c"})->resumed_tokens, 20);
    lineage.RecordPublication(ContentHash{"c"}, {.resumed_tokens = 0, .access_epoch = 9});
    EXPECT_EQ(lineage.FindPublication(ContentHash{"c"})->resumed_tokens, 0)
        << "a later publication replaces the earlier";
    EXPECT_EQ(lineage.FindPublication(ContentHash{"c"})->access_epoch, 9u);

    EXPECT_EQ(lineage.Resumes(ContentHash{"b"}), 0);
    lineage.NoteResume(ContentHash{"b"});
    EXPECT_EQ(lineage.Resumes(ContentHash{"b"}), 1);
    lineage.NoteResume(ContentHash{"b"});
    lineage.NoteResume(ContentHash{"b"});
    EXPECT_EQ(lineage.Resumes(ContentHash{"b"}), 2) << "more than one";
}

// Both maps forget their oldest key past kMaxEntries; a key recorded again
// keeps its place.
TEST(CheckpointLineageTest, ForgetsTheOldestKeysPastItsBound) {
    CheckpointLineage lineage;
    lineage.RecordPublication(ContentHash{"first"}, {.resumed_tokens = 4, .access_epoch = 1});
    lineage.RecordPublication(ContentHash{"second"}, {.resumed_tokens = 8, .access_epoch = 2});
    lineage.NoteResume(ContentHash{"first"});
    for (std::size_t i = 0; i + 2 < CheckpointLineage::kMaxEntries; ++i) {
        lineage.RecordPublication(ContentHash{"p" + std::to_string(i)}, {});
        lineage.NoteResume(ContentHash{"p" + std::to_string(i)});
    }
    lineage.RecordPublication(ContentHash{"first"}, {.resumed_tokens = 12, .access_epoch = 3});
    ASSERT_NE(lineage.FindPublication(ContentHash{"first"}), nullptr) << "kMaxEntries keys fit";
    lineage.RecordPublication(ContentHash{"newest"}, {});
    EXPECT_EQ(lineage.FindPublication(ContentHash{"first"}), nullptr) << "re-recording kept the oldest place";
    EXPECT_NE(lineage.FindPublication(ContentHash{"second"}), nullptr);
    EXPECT_EQ(lineage.Resumes(ContentHash{"first"}), 1);
    lineage.NoteResume(ContentHash{"p-last"});
    lineage.NoteResume(ContentHash{"p-newest"});
    EXPECT_EQ(lineage.Resumes(ContentHash{"first"}), 0) << "forgotten";
}

// One KV group and one State group, P = 2, on Device and Host.
CacheCoordinator MakeTwoGroupCoordinator(BlockPool& device, BlockPool* host, bool write_back) {
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    return MakeCoordinator(specs, /*prefix_granularity=*/2, device, /*enable_l3_storage=*/false, host,
                           /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false, write_back,
                           /*checkpoint_retention_requests=*/0);
}

// A demoted checkpoint still on the Device is not queued for Host while
// demotion is on: free Host space would take it back. Off, epoch 0 skips
// nothing, as for a superseded input.
TEST(CheckpointDemotionCoordinatorTest, ADemotedCheckpointIsNotQueuedForHost) {
    BlockPool device{8, {1, 1}};
    BlockPool host{8, {1, 1}};
    CacheCoordinator coordinator = MakeTwoGroupCoordinator(device, &host, /*write_back=*/false);
    for (const std::uint32_t group : {0u, 1u}) {
        CacheBlockRef block = device.AcquireBlock(group);
        coordinator.GroupPrefixIndex(static_cast<std::int32_t>(group))
            .Register(device, block, Key("h0", group), /*access_epoch=*/3, /*logical_block_index=*/0,
                      CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    }
    const std::vector<std::string> hashes{"h0"};
    const std::vector<CacheKey> keys = coordinator.BoundaryCheckpointKeys(hashes, /*kept_boundaries=*/{});
    ASSERT_EQ(keys, std::vector<CacheKey>{Key("h0", 1)}) << "the State entry alone, never the KV page";
    EXPECT_EQ(coordinator.DemoteEntries(keys), 1);
    EXPECT_EQ(coordinator.DemoteEntries(keys), 0) << "dated back already";
    coordinator.QueueCachedBlocksForStore(hashes);
    EXPECT_EQ(coordinator.TakePendingStores().size(), 2u) << "demotion off: nothing is skipped";
    coordinator.SetCheckpointDemotion(true);
    coordinator.QueueCachedBlocksForStore(hashes);
    const std::vector<CacheCoordinator::StoreCandidate> pending = coordinator.TakePendingStores();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].key, Key("h0", 0));
}

// What a demotion of the checkpoint at 20 takes, P = 4, with a sliding window
// of two blocks: its State entry and the window behind it, [12, 20), less the
// blocks a kept boundary resumes with -- 24's window is [16, 24), 28's
// [20, 28).
TEST(CheckpointDemotionCoordinatorTest, BoundaryKeysLeaveTheWindowAKeptBoundaryResumesWith) {
    BlockPool device{8, {1, 1, 1}};
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 8,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 4},
    };
    const CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const std::vector<std::string> chain{"p0", "p1", "p2", "p3", "p4"};
    // As group:hash/offset; the window block [12, 16) is keyed by p3.
    const auto keys = [&](const std::vector<std::int32_t>& kept_boundaries) {
        std::vector<std::string> names;
        for (const CacheKey& key : coordinator.BoundaryCheckpointKeys(chain, kept_boundaries)) {
            names.push_back(std::to_string(key.group_id) + ":" + key.content_hash.ToString() + "/" +
                            std::to_string(key.page_offset));
        }
        return names;
    };
    EXPECT_EQ(keys({}), (std::vector<std::string>{"1:p4/0", "2:p3/0", "2:p4/0"}));
    EXPECT_EQ(keys({24, 28}), (std::vector<std::string>{"1:p4/0", "2:p3/0"})) << "24 resumes with [16, 20)";
    EXPECT_EQ(keys({28}), (std::vector<std::string>{"1:p4/0", "2:p3/0", "2:p4/0"}));
}

// Demotion under the write-back policy, on the coordinator: one KV group (four
// blocks per parent, P = 4) and one State group, the Host as large as the
// Device. Demotion is on unless a test turns it off.
class CheckpointDemotionWriteBackTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kP = 4;
    static constexpr std::uint32_t kFull = 0;
    static constexpr std::uint32_t kState = 1;
    static constexpr std::array kSpecs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };

    CheckpointDemotionWriteBackTest() { coordinator_.SetCheckpointDemotion(true); }

    // Admits a request over `tokens`, a whole number of pages, and publishes
    // them with the State group's checkpoint at the end; returns the pages'
    // hashes.
    std::vector<std::string> Publish(const std::vector<std::int32_t>& tokens) {
        return PublishInSession(tokens, /*session_id=*/{});
    }
    // As Publish, by a request of the session `session_id` (none when empty),
    // whose entries carry it.
    std::vector<std::string> PublishInSession(const std::vector<std::int32_t>& tokens, const std::string& session_id) {
        const auto num_tokens = static_cast<std::int32_t>(tokens.size());
        std::vector<BlockTable> tables(2);
        const std::array demands{
            GroupDemand{.table = &tables[kFull], .extent = DenseGrowth{num_tokens}},
            GroupDemand{.table = &tables[kState], .extent = StateStep{StateStep::Kind::kRotate}},
        };
        // Held until the publication: it registers the session.
        const std::optional<CacheCoordinator::AdmissionResult> admitted =
            coordinator_.Admit(coordinator_.ProbePrefix({}), demands, RequestProgress{}, /*copy_in_blocks=*/{},
                               /*copy_in_point=*/std::nullopt, /*request_access_epoch=*/std::nullopt,
                               /*evict_blocks_past_segments=*/false, session_id, /*cache_group_id=*/{});
        if (!admitted) {
            ADD_FAILURE() << "the admission failed";
            return {};
        }
        std::vector<std::span<const std::int32_t>> pages;
        for (std::int32_t page = 0; page < num_tokens / kP; ++page) {
            pages.emplace_back(tokens.data() + static_cast<std::ptrdiff_t>(page * kP), static_cast<std::size_t>(kP));
        }
        std::vector<std::string> hashes = ComputePrefixHashes(pages, "");
        coordinator_.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.tokens = tokens,
                                                              .prefix_hashes = hashes,
                                                              .first_new_prefix_page = 0,
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = num_tokens},
            admitted->access_epoch);
        coordinator_.Free(tables);
        return hashes;
    }

    // Holds every free slot and empty parent of `pool`, so what comes next
    // is funded by evictions.
    void HoldFreeCapacity(BlockPool& pool) {
        for (CacheBlockRef& block : pool.AcquireBlocks(kFull, pool.NumFreeSlots(kFull))) {
            held_.push_back(std::move(block));
        }
        for (CacheBlockRef& block : pool.AcquireBlocks(kState, pool.NumEmptyLcmBlocks())) {
            held_.push_back(std::move(block));
        }
    }

    // An admission of one KV block on a full Device, released at once.
    void AdmitOneBlockOnAFullDevice() {
        HoldFreeCapacity(device_);
        std::vector<BlockTable> tables(2);
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{kP}}, RequestProgress{},
                                 StateStep{.kind = StateStep::Kind::kKeep}));
        coordinator_.Free(tables);
        held_.clear();
    }

    // The keys the plan's admissions recorded for their write-back drain.
    static std::vector<CacheKey> RecordedKeys(const std::vector<CacheCoordinator::EvictedSegment>& segments) {
        std::vector<CacheKey> keys;
        for (const CacheCoordinator::EvictedSegment& segment : segments) {
            for (const CacheCoordinator::EvictedBlock& member : segment.members) {
                keys.push_back(member.key);
            }
        }
        return keys;
    }

    std::optional<std::uint64_t> HostEpoch(const CacheKey& key) const {
        const std::optional<PrefixCacheIndex::CachedBlockSnapshot> entry =
            coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id)).Inspect(host_, key);
        return entry ? std::optional{entry->metadata.eviction.last_access_epoch} : std::nullopt;
    }
    bool HostEntryDemoted(const CacheKey& key) const {
        const std::optional<PrefixCacheIndex::CachedBlockSnapshot> entry =
            coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id)).Inspect(host_, key);
        return entry && entry->metadata.demoted;
    }

    std::vector<CacheKey> CheckpointKeys(const std::vector<std::string>& hashes) const {
        return coordinator_.BoundaryCheckpointKeys(hashes, /*kept_boundaries=*/{});
    }

    // Demotions, drops and demoted landings leave the boundary tree in step
    // with the indexes (CacheCoordinatorTestAccess::BoundaryTreeMismatches):
    // checked before a round clears the cache, and when a test ends.
    void ExpectTreeInStep() const {
        const std::vector<std::string> mismatches = CacheCoordinatorTestAccess::BoundaryTreeMismatches(coordinator_);
        EXPECT_TRUE(mismatches.empty()) << mismatches.size() << " mismatches, first: " << mismatches.front();
    }
    bool ClearRound() {
        ExpectTreeInStep();
        return coordinator_.ClearCache();
    }
    void TearDown() override { ExpectTreeInStep(); }

    BlockPool device_{32, {4, 1}};
    BlockPool host_{32, {4, 1}};
    CacheCoordinator coordinator_ =
        MakeCoordinator(kSpecs, kP, device_, /*enable_l3_storage=*/false, &host_, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true,
                        /*checkpoint_retention_requests=*/0);
    TierTransferManager transfers_{coordinator_};
    // Declared after the coordinator, so released before the pools.
    std::vector<CacheBlockRef> held_;
};

// The checkpoint at 8 sits under the one at 16, so the admission takes it
// alone, first: it carries epoch 0. Demoted, it leaves the Device and is
// dropped, counted as the admission's victim; nothing goes to the Host. With
// demotion off, epoch 0 reads as a superseded input's: dropped as well,
// counted as the request's reclaim.
TEST_F(CheckpointDemotionWriteBackTest, ADemotedCheckpointIsDroppedNeverWrittenBack) {
    for (const bool demotion : {true, false}) {
        SCOPED_TRACE(demotion);
        coordinator_.SetCheckpointDemotion(demotion);
        const std::vector<std::string> shorter = Publish(MakeTokens(8));
        const std::vector<std::string> longer = Publish(MakeTokens(16));
        ASSERT_EQ(coordinator_.DemoteEntries(CheckpointKeys(shorter)), 1);
        const CacheCoordinator::DeviceEvictionStats before = coordinator_.DeviceEvictions();
        const std::uint64_t dropped_before = coordinator_.DemotedEntriesNotWrittenBack();
        AdmitOneBlockOnAFullDevice();

        EXPECT_FALSE(coordinator_.GroupPrefixIndex(kState).Contains(device_, Key(shorter[1], kState)));
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(kState).Contains(device_, Key(longer[3], kState)));
        EXPECT_TRUE(RecordedKeys(coordinator_.TakeEvictedSegments()).empty()) << "nothing to write back";
        const CacheCoordinator::DeviceEvictionStats& after = coordinator_.DeviceEvictions();
        EXPECT_EQ(after.admission.blocks - before.admission.blocks, demotion ? 1u : 0u);
        EXPECT_EQ(after.request_reclaim.blocks - before.request_reclaim.blocks, demotion ? 0u : 1u);
        EXPECT_EQ(coordinator_.DemotedEntriesNotWrittenBack() - dropped_before, demotion ? 1u : 0u);
        ASSERT_TRUE(ClearRound());
    }
}

// Whatever evicts a demoted entry -- here a stranded suffix under another
// victim -- records nothing; the dated checkpoint deeper on its chain is
// recorded.
TEST_F(CheckpointDemotionWriteBackTest, NoEvictionCauseWritesADemotedEntryBack) {
    const std::vector<std::string> shorter = Publish(MakeTokens(8));
    const std::vector<std::string> longer = Publish(MakeTokens(16));
    ASSERT_EQ(coordinator_.DemoteEntries(CheckpointKeys(shorter)), 1);
    const auto location_of = [&](const std::string& hash) {
        return *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, kState, Key(hash, kState));
    };
    CacheCoordinator::EvictedSegment segment;
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(coordinator_, kState, location_of(shorter[1]),
                                                           &CacheCoordinator::DeviceEvictionStats::stranded_suffix,
                                                           &segment));
    EXPECT_TRUE(segment.members.empty());
    EXPECT_EQ(coordinator_.DemotedEntriesNotWrittenBack(), 1u);
    EXPECT_EQ(coordinator_.DeviceEvictions().stranded_suffix.blocks, 1u) << "counted under what evicted it";
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(coordinator_, kState, location_of(longer[3]),
                                                           &CacheCoordinator::DeviceEvictionStats::stranded_suffix,
                                                           &segment));
    ASSERT_EQ(segment.members.size(), 1u) << "a dated checkpoint is written back";
    EXPECT_EQ(segment.members[0].key, Key(longer[3], kState));
    EXPECT_EQ(coordinator_.DemotedEntriesNotWrittenBack(), 1u);
    coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
}

// The checkpoint at 8 is written back before it is demoted: its copy is
// still out when the demotion comes. Its segment publishes it on the Host
// demoted, at the demoted epoch; without the demotion, at its own.
TEST_F(CheckpointDemotionWriteBackTest, ACheckpointDemotedWhileItsCopyIsOutLandsDemoted) {
    for (const bool demoted_in_flight : {true, false}) {
        SCOPED_TRACE(demoted_in_flight);
        const std::vector<std::string> shorter = Publish(MakeTokens(8));
        Publish(MakeTokens(16));
        AdmitOneBlockOnAFullDevice();
        const std::optional<WriteBackOperation> store = transfers_.StartEvictionStores();
        ASSERT_TRUE(store.has_value()) << "the checkpoint at 8 goes to the Host";
        ASSERT_EQ(store->transfers.size(), 1u);
        ASSERT_EQ(coordinator_.DemoteEntries(CheckpointKeys(shorter)), 0) << "neither tier holds it yet";
        if (demoted_in_flight) {
            EXPECT_EQ(transfers_.DemoteInFlightEvictions(CheckpointKeys(shorter)), 1);
        }
        transfers_.CompleteWriteBack(store->op_id);
        const std::optional<std::uint64_t> epoch = HostEpoch(Key(shorter[1], kState));
        ASSERT_TRUE(epoch.has_value());
        EXPECT_EQ(*epoch == CacheCoordinator::kDemotedAccessEpoch, demoted_in_flight);
        EXPECT_EQ(HostEntryDemoted(Key(shorter[1], kState)), demoted_in_flight);
        ASSERT_TRUE(ClearRound());
    }
}

// The Host held the checkpoint at 8 already when an admission evicted its
// boundary: the segment copies the two KV pages and holds that Host entry
// until they land, and its settlement keeps the newer of the two dates. A
// demotion while the copy is out dates the Host entry back at once; without
// it the entry keeps its date.
TEST_F(CheckpointDemotionWriteBackTest, AHostEntryTheCopyHoldsIsDemotedWithIt) {
    for (const bool demoted_in_flight : {true, false}) {
        SCOPED_TRACE(demoted_in_flight);
        const std::vector<std::string> shorter = Publish(MakeTokens(8));
        const CacheKey checkpoint = Key(shorter[1], kState);
        CacheBlockRef block = host_.AcquireBlock(kState);
        coordinator_.CacheHostBlock(block, checkpoint, std::nullopt);
        block.reset();
        const std::optional<std::uint64_t> dated = HostEpoch(checkpoint);
        AdmitOneBlockOnAFullDevice();
        const std::optional<WriteBackOperation> store = transfers_.StartEvictionStores();
        ASSERT_TRUE(store.has_value());
        ASSERT_EQ(store->transfers.size(), 2u) << "the KV pages; the Host holds the checkpoint";
        ASSERT_EQ(coordinator_.DemoteEntries(CheckpointKeys(shorter)), 0)
            << "the Device entry is gone, the copy holds the Host entry";
        if (demoted_in_flight) {
            EXPECT_EQ(transfers_.DemoteInFlightEvictions(CheckpointKeys(shorter)), 1) << "the Host entry";
            EXPECT_EQ(HostEpoch(checkpoint), CacheCoordinator::kDemotedAccessEpoch);
        }
        transfers_.CompleteWriteBack(store->op_id);
        EXPECT_EQ(HostEpoch(checkpoint),
                  demoted_in_flight ? std::optional{CacheCoordinator::kDemotedAccessEpoch} : dated);
        EXPECT_EQ(HostEntryDemoted(checkpoint), demoted_in_flight) << "the settlement keeps the demotion";
        ASSERT_TRUE(ClearRound());
    }
}

// Two conversations, X older than D, each checkpoint on the Host alone. A
// Host segment plan needing one victim takes X's without demotion; with D's
// checkpoint demoted, D's first, whatever its pages' dates. Also while the
// Device still holds D's pages, unheld: their Host copies are then safe
// duplicates and leave before any Host-only cache, in X's place; D's demoted
// checkpoint boundary still goes first, ahead of those copies.
TEST_F(CheckpointDemotionWriteBackTest, HostSegmentPlacementTakesADemotedCheckpointFirst) {
    for (const bool d_pages_on_device : {false, true}) {
        for (const bool demote : {true, false}) {
            SCOPED_TRACE(testing::Message() << "d_pages_on_device=" << d_pages_on_device << " demote=" << demote);
            const std::vector<std::string> x = Publish(MakeTokens(16, 100));
            const std::vector<std::string> d = Publish(MakeTokens(16, 1));
            for (const std::vector<std::string>* pages : {&x, &d}) {
                for (const std::string& page : *pages) {
                    CacheBlockRef block = host_.AcquireBlock(kFull);
                    coordinator_.CacheHostBlock(block, Key(page, kFull), std::nullopt);
                }
                CacheBlockRef block = host_.AcquireBlock(kState);
                coordinator_.CacheHostBlock(block, Key(pages->back(), kState), std::nullopt);
                if (pages == &d && d_pages_on_device) {
                    // The checkpoint alone leaves the Device; D's pages stay.
                    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(
                        coordinator_, kState,
                        *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, kState, Key(d.back(), kState))));
                    continue;
                }
                for (const std::string& page : *pages) {
                    CacheCoordinatorTestAccess::EvictCachedBlock(
                        coordinator_, kFull,
                        *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, kFull, Key(page, kFull)));
                }
            }
            if (demote) {
                ASSERT_EQ(coordinator_.DemoteEntries(CheckpointKeys(d)), 1) << "the Host's checkpoint";
            }
            HoldFreeCapacity(host_);
            const std::array<std::uint32_t, 1> groups{kState};
            const std::array<std::int32_t, 1> buckets{0};
            const std::array<CacheCoordinator::HostSegmentRows, 1> segments{
                CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
            CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);
            ASSERT_TRUE(static_cast<bool>(batch.blocks[0]));
            EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(d.back(), kState)), !demote);
            EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(x.back(), kState)), demote || d_pages_on_device);
            EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(d.front(), kFull)), !demote && !d_pages_on_device)
                << "D's pages leave with its demoted checkpoint, or as Device duplicates without demotion";
            batch.blocks.clear();
            held_.clear();
            ASSERT_TRUE(ClearRound());
        }
    }
}

// As in HostSegmentPlacementTakesADemotedCheckpointFirst, X older than D,
// each checkpoint on the Host alone and D's demoted, but both conversations'
// pages stay on the Device, where a request holds D's: the Host copies of D's
// pages are held duplicates, X's unheld ones. The held copies leave before
// D's demoted checkpoint and X's unheld ones after it, all before X's
// checkpoint: a plan needing one parent takes D's copies alone, one needing
// two D's checkpoint as well, and one needing three X's copies too.
TEST_F(CheckpointDemotionWriteBackTest, HostSegmentPlacementTakesADemotedCheckpointBetweenHeldAndUnheldDuplicates) {
    for (const std::size_t parents : {1u, 2u, 3u}) {
        SCOPED_TRACE(testing::Message() << "parents=" << parents);
        const std::vector<std::string> x = Publish(MakeTokens(16, 100));
        const std::vector<std::string> d = Publish(MakeTokens(16, 1));
        for (const std::vector<std::string>* pages : {&x, &d}) {
            for (const std::string& page : *pages) {
                CacheBlockRef block = host_.AcquireBlock(kFull);
                coordinator_.CacheHostBlock(block, Key(page, kFull), std::nullopt);
            }
            CacheBlockRef block = host_.AcquireBlock(kState);
            coordinator_.CacheHostBlock(block, Key(pages->back(), kState), std::nullopt);
            // The checkpoint alone leaves the Device; the pages stay.
            ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(
                coordinator_, kState,
                *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, kState, Key(pages->back(), kState))));
        }
        ASSERT_EQ(coordinator_.DemoteEntries(CheckpointKeys(d)), 1) << "the Host's checkpoint";
        std::vector<CacheBlockRef> request;
        for (const std::string& page : d) {
            request.push_back(coordinator_.GroupPrefixIndex(kFull).Find(device_, Key(page, kFull)));
            ASSERT_TRUE(static_cast<bool>(request.back()));
        }
        HoldFreeCapacity(host_);
        const std::vector<std::uint32_t> groups(parents, kState);
        const std::vector<std::int32_t> buckets(parents, 0);
        const std::array<CacheCoordinator::HostSegmentRows, 1> segments{
            CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = parents}};
        CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);
        ASSERT_EQ(batch.stats.allocated, parents);
        for (const std::string& page : d) {
            EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(page, kFull))) << "a held duplicate, first";
        }
        EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(d.back(), kState)), parents < 2)
            << "the demoted checkpoint, after the held duplicates";
        for (const std::string& page : x) {
            EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(page, kFull)), parents < 3)
                << "an unheld duplicate, after the demoted checkpoint";
        }
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(x.back(), kState))) << "Host-only cache, last";
        batch.blocks.clear();
        request.clear();
        held_.clear();
        ASSERT_TRUE(ClearRound());
    }
}

// A session's demotion (DemoteSessions) dates the session's entries back to
// the demoted epoch without marking them demoted, so with demotion on they
// keep the rules they have with it off: a finish queues them for the Host, an
// admission that evicts one counts it as the request's reclaim, as it does a
// superseded input, and a stranded suffix is written back. They still do once
// the session's next request has made the session current again, which leaves
// their dates as they are.
TEST_F(CheckpointDemotionWriteBackTest, ASessionsDemotedEntriesKeepTheirOwnRules) {
    const auto location_of = [&](const CacheKey& key) {
        return *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, key.group_id, key);
    };
    for (const bool demotion : {false, true}) {
        for (const bool resumed : {false, true}) {
            SCOPED_TRACE(testing::Message() << "demotion=" << demotion << " resumed=" << resumed);
            coordinator_.SetCheckpointDemotion(demotion);
            const std::vector<std::string> shorter = PublishInSession(MakeTokens(8), "s");
            const std::vector<std::string> longer = PublishInSession(MakeTokens(16), "s");
            ASSERT_EQ(coordinator_.DemoteSession("s", /*ended=*/false), 6u) << "four pages and two checkpoints";
            if (resumed) {
                PublishInSession(MakeTokens(4, 500), "s");
            }
            const CacheKey shorter_checkpoint = Key(shorter[1], kState);
            const CacheKey longer_checkpoint = Key(longer[3], kState);
            ASSERT_EQ(coordinator_.GroupPrefixIndex(kState)
                          .Inspect(device_, shorter_checkpoint)
                          ->metadata.eviction.last_access_epoch,
                      CacheCoordinator::kDemotedAccessEpoch);
            coordinator_.QueueCachedBlocksForStore(shorter);
            EXPECT_EQ(coordinator_.TakePendingStores().size(), 3u) << "the two pages and the checkpoint";

            const CacheCoordinator::DeviceEvictionStats before = coordinator_.DeviceEvictions();
            CacheCoordinator::EvictedSegment segment;
            ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(coordinator_, kState, location_of(longer_checkpoint),
                                                                   &CacheCoordinator::DeviceEvictionStats::admission,
                                                                   &segment));
            EXPECT_TRUE(segment.members.empty());
            const CacheCoordinator::DeviceEvictionStats& after = coordinator_.DeviceEvictions();
            EXPECT_EQ(after.request_reclaim.blocks - before.request_reclaim.blocks, 1u);
            EXPECT_EQ(after.admission.blocks, before.admission.blocks);
            ASSERT_TRUE(CacheCoordinatorTestAccess::EvictAloneInto(
                coordinator_, kState, location_of(shorter_checkpoint),
                &CacheCoordinator::DeviceEvictionStats::stranded_suffix, &segment));
            ASSERT_EQ(segment.members.size(), 1u) << "the stranded suffix is written back";
            EXPECT_EQ(segment.members[0].key, shorter_checkpoint);
            EXPECT_EQ(coordinator_.DemotedEntriesNotWrittenBack(), 0u);
            coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
            ASSERT_TRUE(ClearRound());
        }
    }
}

// A demotion still marks an entry a session's demotion dated back already:
// with demotion on, a finish then queues the pages alone for the Host.
TEST_F(CheckpointDemotionWriteBackTest, ADemotionMarksAnEntryASessionsDemotionDatedBack) {
    const std::vector<std::string> hashes = PublishInSession(MakeTokens(8), "s");
    ASSERT_EQ(coordinator_.DemoteSession("s", /*ended=*/false), 3u);
    EXPECT_EQ(coordinator_.DemoteEntries(CheckpointKeys(hashes)), 1);
    coordinator_.QueueCachedBlocksForStore(hashes);
    EXPECT_EQ(coordinator_.TakePendingStores().size(), 2u) << "the pages alone";
}

// As in HostSegmentPlacementTakesADemotedCheckpointFirst, X older than D and
// D's pages still on the Device, unheld, but D's session demoted instead of
// D's checkpoint (its Host copies carry the session, as an eviction
// write-back's do). With demotion on as off, D's checkpoint ranks by its date
// among the Host-only cache: the Host copies of D's pages, Device duplicates,
// leave before it, and both checkpoints stay. So they do once the session's
// next request has made the session current again.
TEST_F(CheckpointDemotionWriteBackTest, HostSegmentPlacementRanksASessionsDemotedCheckpointByItsDate) {
    for (const bool demotion : {false, true}) {
        for (const bool resumed : {false, true}) {
            SCOPED_TRACE(testing::Message() << "demotion=" << demotion << " resumed=" << resumed);
            coordinator_.SetCheckpointDemotion(demotion);
            const std::vector<std::string> x = Publish(MakeTokens(16, 100));
            const std::vector<std::string> d = PublishInSession(MakeTokens(16, 1), "d");
            for (const std::vector<std::string>* pages : {&x, &d}) {
                std::vector<CacheKey> keys;
                for (const std::string& page : *pages) {
                    keys.push_back(Key(page, kFull));
                }
                keys.push_back(Key(pages->back(), kState));
                for (const CacheKey& key : keys) {
                    CacheBlockRef block = host_.AcquireBlock(key.group_id);
                    coordinator_.CacheHostBlock(block, key,
                                                coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id))
                                                    .Inspect(device_, key)
                                                    ->metadata);
                }
            }
            for (const std::string& page : x) {
                CacheCoordinatorTestAccess::EvictCachedBlock(
                    coordinator_, kFull,
                    *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, kFull, Key(page, kFull)));
            }
            ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(
                coordinator_, kState,
                *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, kState, Key(d.back(), kState))));
            ASSERT_EQ(coordinator_.DemoteSession("d", /*ended=*/false), 9u)
                << "D's pages on both tiers and its Host checkpoint";
            if (resumed) {
                PublishInSession(MakeTokens(4, 900), "d");
            }
            HoldFreeCapacity(host_);
            const std::array<std::uint32_t, 1> groups{kState};
            const std::array<std::int32_t, 1> buckets{0};
            const std::array<CacheCoordinator::HostSegmentRows, 1> segments{
                CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
            CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);
            ASSERT_TRUE(static_cast<bool>(batch.blocks[0]));
            EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(d.back(), kState)));
            EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(x.back(), kState)));
            EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(d.front(), kFull)));
            batch.blocks.clear();
            held_.clear();
            ASSERT_TRUE(ClearRound());
        }
    }
}

// Whole conversations on the P role (the fused role where a suite says so):
// in-forward checkpoints, and checkpoint retention off, so entries rank by use
// alone, as in the superseded-input scenarios. One KV group "full" and one
// State group "state0", P = 4 and an 8-token chunk budget: a 22-token prompt
// writes its checkpoint at 20 inside the forward [16, 22).
class CheckpointDemotionSuite : public SchedulerTestSuite {
protected:
    static constexpr std::int32_t kP = 4;

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.role = role_;
        cfg.prefix_granularity = kP;
        cfg.device_allocator.total_pages = device_pages_;
        cfg.host_allocator.total_pages = host_pages_;
        cfg.disable_l2_cache = host_pages_ <= 1;
        if (host_pages_ > 1) {
            cfg.host_write_policy = write_policy_;
        }
        cfg.max_scheduled_tokens = 8;
        cfg.max_batch_size = 2;
        cfg.decode_input_tokens = 1;
        cfg.state_checkpoint_in_forward = in_forward_;
        cfg.demote_superseded_state_checkpoints = demote_;
        cfg.checkpoint_retention_requests = 0;
        for (const std::string& id : group_ids_) {
            const bool is_state = id.starts_with("state");
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = id,
                .block_granularity = is_state ? 0 : kP,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = is_state ? AttnKind::kState : AttnKind::kFull,
                .transfer_policy = is_state ? CacheTransferPolicy::LatestSnapshot : CacheTransferPolicy::FullSuffix,
            });
        }
        if (window_tokens_ > 0) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = "swa",
                .block_granularity = kP,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = AttnKind::kSlidingWindow,
                .sliding_window_tokens = window_tokens_,
                .transfer_policy = CacheTransferPolicy::FullSuffix,
            });
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Rebuilds the engine with demotion as given and `device_pages` and
    // `host_pages` (each with its null page; 1 or less: no Host L2).
    void Rebuild(bool demote, std::int32_t device_pages, std::int32_t host_pages) {
        demote_ = demote;
        device_pages_ = device_pages;
        host_pages_ = host_pages;
        config_ = WithHostPackingFromDevice(MakeConfig());
        scheduler_ = std::make_unique<Scheduler>(config_);
    }

    void SendBootstrapped(const std::string& id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{id});
        scheduler_->Advance(std::move(event));
    }
    void SendPdSucceeded(const std::string& id) {
        ExecutionEvent event;
        event.With(pd::SucceededEvent{id});
        scheduler_->Advance(std::move(event));
    }
    void AckLoads(const ExecutionPlan& plan) {
        for (const CacheOperation& operation : ExtractCacheOpsOfKind<LoadBackBatch>(plan)) {
            for (const std::uint32_t op_id : std::get<LoadBackBatch>(operation).op_ids) {
                SendLoadBackDone(op_id);
            }
        }
    }

    // Prefills `id` alone to its end, every copy acknowledged unless
    // `held_write_backs` collects their op ids; returns where its first chunk
    // resumed. The P role's hand-off is left pending.
    std::int32_t Prefill(const std::string& id, const std::vector<std::int32_t>& tokens,
                         std::vector<std::uint32_t>* held_write_backs = nullptr) {
        Submit(RequestSpec{.request_id = id, .tokens = tokens, .max_new_tokens = 32});
        if (role_ == Role::kP) {
            SendBootstrapped(id);
        }
        std::int32_t first_prefix = -1;
        std::int32_t scheduled = 0;
        const auto total = static_cast<std::int32_t>(tokens.size());
        for (int round = 0; scheduled < total && round < 32; ++round) {
            const ExecutionPlan plan = PlanOnce();
            AckLoads(plan);
            if (held_write_backs == nullptr) {
                AckWriteBacks(plan);
            } else {
                for (const CacheOperation& operation : ExtractCacheOpsOfKind<WriteBackBatch>(plan)) {
                    const auto& op_ids = std::get<WriteBackBatch>(operation).op_ids;
                    held_write_backs->insert(held_write_backs->end(), op_ids.begin(), op_ids.end());
                }
            }
            const ForwardBatch* batch = FindForwardBatch(plan);
            if (batch == nullptr || batch->request_ids.empty()) {
                continue;
            }
            EXPECT_EQ(batch->request_ids, std::vector<std::string>{id});
            if (first_prefix < 0) {
                first_prefix = batch->extend_prefix_lens.at(0);
            }
            scheduled = batch->extend_prefix_lens.at(0) + batch->input_lengths.at(0);
            SendForwardDone(id, scheduled == total ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
        }
        return first_prefix;
    }

    // The P role's hand-off and its ACK, or the fused role's first decode and
    // finish; then the finish's stores.
    void Finish(const std::string& id) {
        AckWriteBacks(PlanOnce());
        if (role_ == Role::kP) {
            SendPdSucceeded(id);
        } else {
            SendForwardDone(id, {901});
            SendFinish(id);
        }
        AckWriteBacks(PlanOnce());
        AckWriteBacks(PlanOnce());
    }

    // One whole turn; returns where its first chunk resumed.
    std::int32_t Turn(const std::string& id, const std::vector<std::int32_t>& tokens) {
        const std::int32_t resumed = Prefill(id, tokens);
        Finish(id);
        return resumed;
    }

    // Where `tokens` would resume on this engine; the request is then aborted.
    std::int32_t ProbeResume(const std::string& id, std::vector<std::int32_t> tokens) {
        Submit(RequestSpec{.request_id = id, .tokens = std::move(tokens), .max_new_tokens = 32});
        if (role_ == Role::kP) {
            SendBootstrapped(id);
        }
        const ExecutionPlan plan = PlanOnce();
        AckLoads(plan);
        AckWriteBacks(plan);
        const ForwardBatch* batch = FindForwardBatch(plan);
        const std::int32_t prefix =
            batch == nullptr || batch->request_ids.empty() ? -1 : batch->extend_prefix_lens.at(0);
        SendAbortEvent(id);
        if (role_ == Role::kP) {
            // Cancellation retains P's pages until the transfer and the
            // probe's dispatched forward have both retired.
            ExecutionEvent event;
            event.With(pd::TransferStoppedEvent{id});
            scheduler_->Advance(std::move(event));
            if (prefix >= 0) {
                SendForwardDone(id);
            }
            EXPECT_FALSE(scheduler_->PdTransferPinned(id));
        }
        AckWriteBacks(PlanOnce());
        return prefix;
    }

    static std::vector<std::int32_t> Extend(std::vector<std::int32_t> tokens, std::int32_t first, std::int32_t n) {
        for (std::int32_t i = 0; i < n; ++i) {
            tokens.push_back(first + i);
        }
        return tokens;
    }
    static std::vector<std::int32_t> First(std::vector<std::int32_t> tokens, std::int32_t n) {
        tokens.resize(static_cast<std::size_t>(n));
        return tokens;
    }

    // Turn 2 of conversation A: turn 1, [1..22], rendered anew from its 22nd
    // token on, and four new tokens; it resumes at turn 1's C(20) and writes
    // C(24).
    static std::vector<std::int32_t> TurnTwo() { return Extend(MakeTokens(21), 500, 4); }
    // Turn 3: turn 2 rendered anew from its 25th token on, and five new
    // tokens; it resumes at C(24) and writes C(28), which demotes C(20).
    static std::vector<std::int32_t> TurnThree() { return Extend(First(TurnTwo(), 24), 600, 5); }
    void ThreeTurns() {
        EXPECT_EQ(Turn("a1", MakeTokens(22)), 0);
        EXPECT_EQ(Turn("a2", TurnTwo()), 20);
        EXPECT_EQ(Turn("a3", TurnThree()), 24);
    }
    // The prompts the scenarios probe with: A's next turn (resumes at C(28)),
    // a turn diverging after turn 2 (at C(24)), a fork of turn 1 (at C(20)),
    // and the next turn of an idle conversation X of 22 tokens from 5000 (at
    // its C(20)).
    static std::vector<std::int32_t> ANext() { return Extend(First(TurnThree(), 28), 700, 5); }
    static std::vector<std::int32_t> ADiverging() { return Extend(First(TurnTwo(), 24), 800, 5); }
    static std::vector<std::int32_t> AFork() { return Extend(MakeTokens(21), 900, 4); }
    static std::vector<std::int32_t> XTurnOne() { return MakeTokens(22, 5000); }
    static std::vector<std::int32_t> XNext() { return Extend(MakeTokens(21, 5000), 950, 4); }

    Role role_{Role::kP};
    bool demote_{true};
    bool in_forward_{true};
    std::int32_t device_pages_{257};
    std::int32_t host_pages_{0};
    SchedulerConfig::HostWritePolicy write_policy_{SchedulerConfig::HostWritePolicy::kWriteThrough};
    std::vector<std::string> group_ids_{"full", "state0"};
    // A sliding-window group "swa" of this many tokens when positive.
    std::int32_t window_tokens_{0};
};

// An idle conversation X, then A's three turns, all stored to the Host
// (write-through), which then has one free block. A filler turn needs two:
// without demotion the Host's oldest entry leaves, one X's next turn needs;
// with it the checkpoint two turns back, which only a fork of A's first turn
// would resume at. A's next turn and a turn diverging after turn 2 resume
// from the Host either way.
TEST_F(CheckpointDemotionSuite, TheCheckpointTwoTurnsBackLeavesTheHostBeforeAnIdleConversation) {
    for (const bool demote : {false, true}) {
        SCOPED_TRACE(demote);
        Rebuild(demote, /*device_pages=*/257, /*host_pages=*/18);
        ASSERT_EQ(Turn("x", XTurnOne()), 0);
        ThreeTurns();
        // X: five pages and its checkpoint at 20, its prompt end being dead
        // (the in-forward checkpoint covers it); A: as many for turn 1, then a
        // page and a checkpoint a turn.
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 16);
        ASSERT_EQ(scheduler_->HostPoolFreeBlocks(), 1);
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), demote ? 2u : 0u) << "C(20)'s State on both tiers";
        Turn("filler", MakeTokens(4, 9000));
        EXPECT_EQ(scheduler_->GetHostCacheStats().host_evictions.lru.blocks, 1u);
        const std::pair<const char*, std::vector<std::int32_t>> probes[] = {
            {"a-next", ANext()}, {"a-diverging", ADiverging()}, {"a-fork", AFork()}, {"x-next", XNext()}};
        for (const auto& [probe, tokens] : probes) {
            SCOPED_TRACE(probe);
            ASSERT_TRUE(scheduler_->ClearL1Cache());
            const std::int32_t resumed = ProbeResume(probe, tokens);
            if (probe == std::string{"a-next"}) {
                EXPECT_EQ(resumed, 28);
            } else if (probe == std::string{"a-diverging"}) {
                EXPECT_EQ(resumed, 24) << "the checkpoint one turn back stays";
            } else if (probe == std::string{"a-fork"}) {
                EXPECT_EQ(resumed, demote ? 0 : 20);
            } else {
                EXPECT_EQ(resumed, demote ? 20 : 0) << "the idle conversation stays whole under demotion";
            }
        }
    }
}

// The same pressure on the Device alone: X and A's three turns leave four
// parents free, and a contender needs five. Without demotion its victims are
// X's oldest entries, its checkpoint at 20 among them (its prompt end is dead,
// never published), so a prompt extending X's resumes at nothing; with
// demotion the one victim is C(20)'s State entry, and that prompt resumes at
// X's checkpoint.
TEST_F(CheckpointDemotionSuite, TheCheckpointTwoTurnsBackLeavesTheDeviceBeforeAnIdleConversation) {
    for (const bool demote : {false, true}) {
        for (const char* probe : {"a-next", "a-diverging", "a-fork", "x-extended"}) {
            SCOPED_TRACE(testing::Message() << "demote=" << demote << " probe=" << probe);
            // Each probe runs on a fresh copy of the state: in a pool this
            // small a probe's own admission evicts.
            Rebuild(demote, /*device_pages=*/21, /*host_pages=*/0);
            ASSERT_EQ(Turn("x", XTurnOne()), 0);
            ThreeTurns();
            ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 4);
            Turn("contender", MakeTokens(8, 9000));
            EXPECT_EQ(scheduler_->GetHostCacheStats().device_evictions.admission.blocks, demote ? 1u : 2u);
            const std::string name{probe};
            if (name == "a-next") {
                EXPECT_EQ(ProbeResume(probe, ANext()), 28);
            } else if (name == "a-diverging") {
                EXPECT_EQ(ProbeResume(probe, ADiverging()), 24);
            } else if (name == "a-fork") {
                EXPECT_EQ(ProbeResume(probe, AFork()), demote ? 0 : 20);
            } else {
                EXPECT_EQ(ProbeResume(probe, Extend(XTurnOne(), 960, 4)), demote ? 20 : 0);
            }
        }
    }
}

// The same Device pressure under the write-back policy, the Host large: the
// contender's admission writes back what it evicts. Without demotion that is
// X's whole chain, a victim set that leaves whole: its five pages and its
// checkpoint at 20 (its prompt end is dead, never published); with demotion
// C(20)'s State entry, dropped instead of written back, so nothing reaches the
// Host.
TEST_F(CheckpointDemotionSuite, UnderWriteBackTheDemotedCheckpointIsDroppedNotWrittenBack) {
    write_policy_ = SchedulerConfig::HostWritePolicy::kWriteBack;
    for (const bool demote : {false, true}) {
        SCOPED_TRACE(demote);
        Rebuild(demote, /*device_pages=*/21, /*host_pages=*/65);
        ASSERT_EQ(Turn("x", XTurnOne()), 0);
        ThreeTurns();
        ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 4);
        ASSERT_EQ(scheduler_->HostPoolCachedBlocks(), 0) << "write-back stores nothing at publication";
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), demote ? 1u : 0u);
        Turn("contender", MakeTokens(8, 9000));
        const HostCacheStats stats = scheduler_->GetHostCacheStats();
        EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), demote ? 0 : 6);
        EXPECT_EQ(stats.eviction_stores.blocks[1], demote ? 0u : 1u) << "X's checkpoint at 20, or no State block";
        EXPECT_EQ(stats.demoted_entries_not_written_back, demote ? 1u : 0u) << "C(20)'s State block";
        EXPECT_EQ(stats.device_evictions.admission.blocks, demote ? 1u : 6u) << "counted as the admission's";
        EXPECT_EQ(ProbeResume("x-next", XNext()), 20);
    }
}

// What the Host holds once write-back evicted a whole conversation: A's three
// turns on a Device four parents larger than they take, then three
// contenders. From the Host alone A's next turn and a turn diverging after
// turn 2 resume either way, and a fork of turn 1 resumes at C(20) only without
// demotion: its demoted State entry was dropped, never written back.
TEST_F(CheckpointDemotionSuite, UnderWriteBackTheHostGetsAllButTheDemotedCheckpoint) {
    write_policy_ = SchedulerConfig::HostWritePolicy::kWriteBack;
    for (const bool demote : {false, true}) {
        SCOPED_TRACE(demote);
        Rebuild(demote, /*device_pages=*/15, /*host_pages=*/129);
        ThreeTurns();
        ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 4);
        for (const std::int32_t contender : {1, 2, 3}) {
            Turn("contender" + std::to_string(contender), MakeTokens(22, 1000 * (contender + 8)));
        }
        EXPECT_EQ(scheduler_->GetHostCacheStats().demoted_entries_not_written_back, demote ? 1u : 0u);
        const std::pair<const char*, std::vector<std::int32_t>> probes[] = {
            {"a-next", ANext()}, {"a-diverging", ADiverging()}, {"a-fork", AFork()}};
        for (const auto& [probe, tokens] : probes) {
            SCOPED_TRACE(probe);
            ASSERT_TRUE(scheduler_->ClearL1Cache());
            const std::int32_t resumed = ProbeResume(probe, tokens);
            if (probe == std::string{"a-next"}) {
                EXPECT_EQ(resumed, 28);
            } else if (probe == std::string{"a-diverging"}) {
                EXPECT_EQ(resumed, 24);
            } else {
                EXPECT_EQ(resumed, demote ? 0 : 20) << "C(20) never reaches the Host under demotion";
            }
        }
    }
}

// Under the write-back policy, turn 3's own admission evicts C(20) and the
// test holds that copy's ACK past turn 3's finish. (The prompt ends being
// dead, C(20) is the one cached entry turn 3 does not hold.) The demotion then
// finds C(20) on neither tier but on its way to the Host, and dates it back
// there: it lands demoted, still resumable.
TEST_F(CheckpointDemotionSuite, ACheckpointOnItsWayToTheHostIsDemotedThere) {
    write_policy_ = SchedulerConfig::HostWritePolicy::kWriteBack;
    for (const bool demote : {false, true}) {
        SCOPED_TRACE(demote);
        Rebuild(demote, /*device_pages=*/12, /*host_pages=*/65);
        ASSERT_EQ(Turn("a1", MakeTokens(22)), 0);
        ASSERT_EQ(Turn("a2", TurnTwo()), 20);
        ASSERT_EQ(scheduler_->EmptyLcmBlocks(), 3);
        std::vector<std::uint32_t> held;
        ASSERT_EQ(Prefill("a3", TurnThree(), &held), 24);
        ASSERT_FALSE(held.empty()) << "the write-back of C(20)";
        Finish("a3");
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), demote ? 1u : 0u) << "C(20), on its way to the Host";
        for (const std::uint32_t op_id : held) {
            SendWriteBackDone(op_id);
        }
        // The Device kept the pages under it.
        EXPECT_EQ(ProbeResume("a-fork", AFork()), 20) << "on the Host, demoted rather than dropped";
    }
}

// A third turn that renders turn 2's new tokens anew resumes at C(20), which
// turn 1 wrote resuming at nothing: nothing lies two turns back, and both
// newer checkpoints stay resumable.
TEST_F(CheckpointDemotionSuite, ATurnResumingOneTurnBackDemotesNothing) {
    Rebuild(/*demote=*/true, /*device_pages=*/257, /*host_pages=*/65);
    ASSERT_EQ(Turn("a1", MakeTokens(22)), 0);
    ASSERT_EQ(Turn("a2", TurnTwo()), 20);
    const std::vector<std::int32_t> a3 = Extend(Extend(MakeTokens(21), 550, 4), 600, 4);
    ASSERT_EQ(Turn("a3", a3), 20);
    EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), 0u);
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    EXPECT_EQ(ProbeResume("a3-next", Extend(First(a3, 28), 700, 5)), 28);
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    EXPECT_EQ(ProbeResume("a-diverging", ADiverging()), 24);
}

// A shared prefix's junction checkpoint is no in-forward checkpoint, so no
// turn that resumed from it demotes it, while each user's own chain demotes
// its first turn's checkpoint. User 3's second turn finds the junction two
// turns back with only user 3 resumed there, so the exact counts check it:
// user 4's hit at 12 would end a demotion before the final probe.
TEST_F(CheckpointDemotionSuite, ASharedJunctionIsNeverDemoted) {
    Rebuild(/*demote=*/true, /*device_pages=*/257, /*host_pages=*/129);
    const std::vector<std::int32_t> shared = MakeTokens(12);
    ASSERT_EQ(Turn("u1", Extend(shared, 1000, 10)), 0);  // C(20) of u1
    ASSERT_EQ(Turn("u2", Extend(shared, 2000, 10)), 0);  // KV to 12: the first chunk ends at the junction 12
    std::uint64_t demoted = 0;
    for (const std::int32_t user : {3, 4, 5}) {
        SCOPED_TRACE(user);
        const std::vector<std::int32_t> first = Extend(shared, 1000 * user, 10);
        ASSERT_EQ(Turn("u" + std::to_string(user), first), 12);
        const std::vector<std::int32_t> second = Extend(First(first, 21), 100 * user, 8);
        ASSERT_EQ(Turn("u" + std::to_string(user) + "b", second), 20);
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), demoted) << "the junction two turns back stays";
        ASSERT_EQ(Turn("u" + std::to_string(user) + "c", Extend(First(second, 28), 50 * user, 5)), 28);
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries() - demoted, 2u) << "the user's own C(20) on both tiers";
        demoted = scheduler_->DemotedCheckpointEntries();
    }
    ASSERT_TRUE(scheduler_->ClearL1Cache());
    EXPECT_EQ(ProbeResume("u9", Extend(shared, 9000, 10)), 12) << "the junction at 12 still serves a new user";
}

// A 21-token system prompt S: a cold request S + 1 writes its in-forward
// checkpoint at 20, inside S. A conversation's first two turns both resume
// there, so its third, resuming at its own C(24), finds C(20) shared and
// leaves it.
TEST_F(CheckpointDemotionSuite, ACheckpointTwoRequestsResumedFromStays) {
    Rebuild(/*demote=*/true, /*device_pages=*/257, /*host_pages=*/0);
    const std::vector<std::int32_t> system = MakeTokens(21);
    ASSERT_EQ(Turn("c1", Extend(system, 1000, 1)), 0);
    ASSERT_EQ(Turn("u2a", Extend(system, 2000, 1)), 20);
    const std::vector<std::int32_t> u2b = Extend(Extend(system, 2100, 1), 3000, 4);
    ASSERT_EQ(Turn("u2b", u2b), 20);
    ASSERT_EQ(Turn("u2c", Extend(First(u2b, 24), 4000, 5)), 24);
    EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), 0u);
}

// Between A's turns 2 and 3 a 20-token prompt, A's first 20 tokens, ends at
// C(20) and publishes its end state there: only turn 2 resumed at C(20), but
// another request used it since, so turn 3 leaves it.
TEST_F(CheckpointDemotionSuite, ACheckpointAnotherRequestRepublishedStays) {
    Rebuild(/*demote=*/true, /*device_pages=*/257, /*host_pages=*/0);
    ASSERT_EQ(Turn("a1", MakeTokens(22)), 0);
    ASSERT_EQ(Turn("a2", TurnTwo()), 20);
    ASSERT_EQ(Turn("b", MakeTokens(20)), 0);
    ASSERT_EQ(Turn("a3", TurnThree()), 24);
    EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), 0u);
}

// With a one-page sliding window the demotion also takes the window block
// behind C(20), [16, 20), never the windows the newer checkpoints resume with.
TEST_F(CheckpointDemotionSuite, TheWindowBehindTheCheckpointIsDemotedWithIt) {
    window_tokens_ = kP;
    for (const bool demote : {false, true}) {
        SCOPED_TRACE(demote);
        Rebuild(demote, /*device_pages=*/257, /*host_pages=*/0);
        ThreeTurns();
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), demote ? 2u : 0u) << "C(20)'s State and its window";
        EXPECT_EQ(ProbeResume("a-next", ANext()), 28);
        EXPECT_EQ(ProbeResume("a-diverging", ADiverging()), 24);
        EXPECT_EQ(ProbeResume("a-fork", AFork()), 20) << "demoted, not evicted";
    }
}

// A window two pages wide: the one behind C(20) is [12, 20), the one C(24)
// resumes with [16, 24). The demotion takes [12, 16) alone, so pressure that
// evicts everything demoted leaves a turn diverging after turn 2 its window.
TEST_F(CheckpointDemotionSuite, TheCheckpointOneTurnBackKeepsItsWindow) {
    window_tokens_ = 2 * kP;
    for (const bool demote : {false, true}) {
        SCOPED_TRACE(demote);
        Rebuild(demote, /*device_pages=*/257, /*host_pages=*/0);
        ThreeTurns();
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), demote ? 2u : 0u) << "C(20)'s State and [12, 16)";
        EXPECT_EQ(ProbeResume("a-diverging", ADiverging()), 24);
    }
}

// The fused role publishes the in-forward checkpoint at the first decode and
// finishes after decoding; its finish demotes two turns back as well.
TEST_F(CheckpointDemotionSuite, AFusedFinishDemotesTheCheckpointTwoTurnsBack) {
    role_ = Role::kFused;
    for (const bool demote : {false, true}) {
        SCOPED_TRACE(demote);
        Rebuild(demote, /*device_pages=*/257, /*host_pages=*/0);
        ThreeTurns();
        EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), demote ? 1u : 0u);
        EXPECT_EQ(ProbeResume("a-next", ANext()), 28);
        EXPECT_EQ(ProbeResume("a-fork", AFork()), 20) << "demoted, not evicted";
    }
}

// Only in-forward checkpoints are ever demoted: the flag requires them,
// except on the D role, which computes no prompt and takes the flag as a
// no-op, as does a model without State groups.
TEST_F(CheckpointDemotionSuite, TheFlagRequiresInForwardCheckpointsAndActsOnlyWithStateGroups) {
    in_forward_ = false;
    EXPECT_THROW(Rebuild(/*demote=*/true, /*device_pages=*/257, /*host_pages=*/0), std::invalid_argument);
    role_ = Role::kD;
    EXPECT_NO_THROW(Rebuild(/*demote=*/true, /*device_pages=*/257, /*host_pages=*/0));
    role_ = Role::kP;
    in_forward_ = true;
    group_ids_ = {"full"};
    Rebuild(/*demote=*/true, /*device_pages=*/257, /*host_pages=*/0);
    ThreeTurns();
    EXPECT_EQ(scheduler_->DemotedCheckpointEntries(), 0u);
}

}  // namespace
}  // namespace tokenspeed::test
