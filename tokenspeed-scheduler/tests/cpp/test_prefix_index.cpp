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
#include <array>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_index.h"
#include "cache/prefix/prefix_matcher.h"
#include "cache/prefix/flat_pointer_index.h"

namespace tokenspeed::test {
namespace {

constexpr std::uint32_t kGroupId = 0;

CacheKey KeyOf(const std::string& content_hash) {
    return CacheKey{.group_id = kGroupId, .content_hash = content_hash, .page_offset = 0};
}

// Registers one freshly acquired block and drops the local reference, leaving
// the index as its only owner so the entry is evictable.
CacheBlockLocation Cache(PrefixCacheIndex& index, BlockPool& pool, const std::string& content_hash,
                         std::uint64_t access_epoch) {
    CacheBlockRef block = pool.AcquireBlock(kGroupId);
    EXPECT_TRUE(block);
    const CacheBlockLocation location = block->Location();
    index.Register(pool, block, KeyOf(content_hash), access_epoch, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr, /*session=*/nullptr);
    return location;
}

// Walks the whole eviction order one access epoch at a time, exactly the way
// admission consumes it.
std::vector<CacheBlockLocation> DrainEvictionOrder(const PrefixCacheIndex& index, const BlockPool& pool) {
    std::vector<CacheBlockLocation> order;
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    while (index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch)) {
        for (const PrefixCacheIndex::EvictionCandidate& candidate : batch) {
            order.push_back(candidate.location);
        }
        batch.clear();
    }
    return order;
}

TEST(PrefixCacheIndexEvictionOrderTest, DeliversTheOldestAccessEpochFirst) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation newest = Cache(index, pool, "newest", /*access_epoch=*/30);
    const CacheBlockLocation oldest = Cache(index, pool, "oldest", /*access_epoch=*/10);
    const CacheBlockLocation middle = Cache(index, pool, "middle", /*access_epoch=*/20);

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{oldest, middle, newest}));
}

TEST(PrefixCacheIndexInspectionTest, ReportsCurrentOwnershipWithoutPinningOrTouchingTheEntry) {
    BlockPool pool(2, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation location = Cache(index, pool, "entry", /*access_epoch=*/10);
    const CacheKey key = KeyOf("entry");
    const auto initial = index.Inspect(pool, key);
    ASSERT_TRUE(initial);
    EXPECT_EQ(initial->location, location);
    EXPECT_EQ(initial->metadata.last_access_epoch, 10);
    EXPECT_FALSE(initial->metadata.was_acquired);
    EXPECT_FALSE(initial->pinned);

    auto holder = index.Find(pool, key);
    ASSERT_TRUE(holder);
    EXPECT_EQ(holder.use_count(), 2);
    ASSERT_TRUE(index.Inspect(pool, key));
    EXPECT_TRUE(index.Inspect(pool, key)->pinned);
    EXPECT_EQ(holder.use_count(), 2);
    index.Register(pool, holder, key, /*access_epoch=*/30, /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                   /*newly_cached=*/nullptr, /*session=*/nullptr);
    EXPECT_EQ(index.Inspect(pool, key)->metadata.last_access_epoch, 30);
    holder = {};
    EXPECT_FALSE(index.Inspect(pool, key)->pinned);

    // Keeping a snapshot alive must not prevent actual eviction.
    EXPECT_EQ(index.Evict(pool, location), key);
    EXPECT_FALSE(index.Inspect(pool, key));
}

TEST(PrefixCacheIndexInspectionTest, KeepsPoolsSeparateAndDoesNotClaimMissingKeys) {
    BlockPool device(1, {1});
    BlockPool host(1, {1});
    PrefixCacheIndex index(kGroupId);
    const auto location = Cache(index, device, "entry", /*access_epoch=*/7);
    ASSERT_TRUE(index.Inspect(device, KeyOf("entry")));
    EXPECT_EQ(index.Inspect(device, KeyOf("entry"))->location, location);
    EXPECT_FALSE(index.Inspect(host, KeyOf("entry")));
    EXPECT_FALSE(index.Inspect(device, KeyOf("absent")));
}

TEST(PrefixCacheIndexInspectionTest, EvictionKeepsItsKeyAndMetadataAfterTheEntryIsGone) {
    BlockPool pool(1, {1});
    PrefixCacheIndex index(0);
    const auto key = KeyOf(std::string(64, 'a'));
    const auto location = Cache(index, pool, key.content_hash, 17);
    EXPECT_THROW(
        index.EvictWithMetadata(
            pool, location, [](const auto&, const auto&, auto*) { throw std::runtime_error("publication refused"); }),
        std::runtime_error);
    EXPECT_TRUE(index.Contains(pool, key));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
    PrefixCacheIndex::CachedBlockMetadata recorded;
    auto removed =
        index.EvictWithMetadata(pool, location, [&](const auto&, const auto& metadata, auto*) { recorded = metadata; });
    ASSERT_TRUE(removed);
    EXPECT_EQ(*removed, key);
    EXPECT_EQ(recorded.last_access_epoch, 17);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_FALSE(index.Contains(pool, location));
    const auto replacement = Cache(index, pool, "replacement", 18);
    EXPECT_EQ(replacement, location);
    EXPECT_EQ(*removed, key);
    EXPECT_EQ(index.KeyOf(pool, replacement), KeyOf("replacement"));
}

// The callback sees the boundary-tree node the entry holds, still counting
// the entry: here two groups' entries under one hash, then the last of them,
// which takes the node along. An index never bound to a tree shows none.
TEST(PrefixCacheIndexInspectionTest, EvictionShowsTheTreeNodeStillCountingTheEntry) {
    BlockPool pool(2, {1, 1});
    BoundaryTree tree(BoundaryTree::Config{.prefix_granularity = 8,
                                           .closed_blocks_per_page = 2,
                                           .num_state_groups = 0,
                                           .has_prefix_closed_groups = true,
                                           .has_host_tier = false,
                                           .queue_dead_candidates = false},
                      /*checkpoints=*/nullptr);
    std::vector<PrefixCacheIndex> indexes;
    indexes.reserve(2);
    std::vector<CacheBlockLocation> locations;
    for (std::uint32_t group = 0; group < 2; ++group) {
        PrefixCacheIndex& index = indexes.emplace_back(group);
        index.TrackBoundaries(tree, pool, /*host_pool=*/nullptr, /*state=*/false, /*closed=*/true);
        CacheBlockRef block = pool.AcquireBlock(group);
        locations.push_back(block->Location());
        index.Register(pool, block, CacheKey{.group_id = group, .content_hash = "page"}, /*access_epoch=*/1,
                       /*logical_block_index=*/0, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr,
                       /*session=*/nullptr);
    }
    ASSERT_NE(tree.Find("page"), nullptr);
    ASSERT_EQ(tree.Find("page")->Entries(BoundaryTree::kDevice), 2);

    for (std::uint32_t group = 0; group < 2; ++group) {
        bool saw_the_hashs_node = false;
        std::int32_t device_entries = 0;
        const auto removed = indexes[group].EvictWithMetadata(
            pool, locations[group], [&](const CacheKey& key, const auto&, BoundaryTree::Node* node) {
                saw_the_hashs_node = node != nullptr && node == tree.Find(key.content_hash);
                device_entries = node == nullptr ? 0 : node->Entries(BoundaryTree::kDevice);
            });
        ASSERT_TRUE(removed) << "group " << group;
        EXPECT_TRUE(saw_the_hashs_node) << "group " << group;
        EXPECT_EQ(device_entries, 2 - static_cast<std::int32_t>(group)) << "group " << group;
    }
    EXPECT_EQ(tree.Find("page"), nullptr);

    BlockPool unbound_pool(1, {1});
    PrefixCacheIndex unbound(kGroupId);
    const CacheBlockLocation location = Cache(unbound, unbound_pool, "page", /*access_epoch=*/1);
    bool saw_a_node = true;
    ASSERT_TRUE(unbound.EvictWithMetadata(unbound_pool, location,
                                          [&](const auto&, const auto&, auto* node) { saw_a_node = node != nullptr; }));
    EXPECT_FALSE(saw_a_node);
}

TEST(PrefixCacheIndexInspectionTest, PackedLocationsAcrossSegmentsSurviveEvictionAndReuse) {
    BlockPool pool(600, {2});
    PrefixCacheIndex index(0);
    std::vector<CacheBlockLocation> locations;
    for (int i = 0; i < 1200; ++i) locations.push_back(Cache(index, pool, std::to_string(i), 1));
    for (int i = 0; i < 1200; i += 2) {
        EXPECT_EQ(index.Evict(pool, locations[i]), KeyOf(std::to_string(i)));
        EXPECT_FALSE(index.Contains(pool, locations[i]));
        EXPECT_EQ(index.KeyOf(pool, locations[i + 1]), KeyOf(std::to_string(i + 1)));
    }
    for (int i = 0; i < 600; ++i) {
        const auto location = Cache(index, pool, "new-" + std::to_string(i), 2);
        EXPECT_EQ(index.KeyOf(pool, location), KeyOf("new-" + std::to_string(i)));
    }
    EXPECT_EQ(index.NumEntries(pool), 1200);
}

TEST(SegmentedLocationIndex, SparseLargeLocationsAndEmptySegmentsCanBeReused) {
    SegmentedLocationIndex<int*> index(/*packing=*/64, /*parents=*/10000000, /*empty=*/nullptr);
    int first = 1, last = 2;
    const CacheBlockLocation low{1, 0}, high{9000000, 63};
    index.Set(low, &first);
    index.Set(high, &last);
    EXPECT_EQ(index.Find(low), &first);
    EXPECT_EQ(index.Find(high), &last);
    EXPECT_EQ(index.Find({0, 0}), nullptr);
    EXPECT_EQ(index.Find({9000000, 64}), nullptr);
    EXPECT_EQ(index.Find({10000001, 0}), nullptr);
    index.Erase(high);
    EXPECT_EQ(index.Find(high), nullptr);
    EXPECT_EQ(index.Find(low), &first);
    index.Set(high, &first);
    EXPECT_EQ(index.Find(high), &first);
}

TEST(ContentHash, PreservesOpaqueBytesEmptyIdentityAndLexicalOrder) {
    const std::vector<std::string> text{"",      std::string(64, '0'),   std::string(64, 'a'), std::string(64, 'A'),
                                        "entry", std::string("a\0b", 3), std::string(63, 'f'), std::string(64, 'x')};
    std::vector<ContentHash> hashes;
    for (const auto& value : text) {
        ContentHash hash(value);
        EXPECT_EQ(hash.ToString(), value);
        EXPECT_EQ(hash.empty(), value.empty());
        hashes.push_back(std::move(hash));
    }
    EXPECT_NE(hashes[0], hashes[1]);
    EXPECT_NE(hashes[2], hashes[3]);
    auto ordered_text = text;
    std::ranges::sort(ordered_text);
    std::ranges::sort(hashes);
    for (std::size_t i = 0; i < hashes.size(); ++i) EXPECT_EQ(hashes[i].ToString(), ordered_text[i]);
    EXPECT_EQ(ContentHash(ContentHash::Digest{}), ContentHash(std::string(64, '0')));
}

TEST(ContentHash, CanonicalTextIsTheDigestFromEveryConstructor) {
    std::mt19937 random(41);
    for (int trial = 0; trial < 32; ++trial) {
        ContentHash::Digest digest;
        for (auto& byte : digest) byte = static_cast<std::uint8_t>(random());
        const ContentHash expected(digest);
        const std::string text = expected.ToString();
        ASSERT_EQ(text.size(), 64u);
        EXPECT_TRUE(std::ranges::all_of(text, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }));
        for (const ContentHash& hash : {ContentHash(text), ContentHash(std::string(text)), ContentHash(text.c_str()),
                                        ContentHash(std::string_view(text))}) {
            EXPECT_EQ(hash, expected);
            EXPECT_EQ(hash.Hash(), expected.Hash());
            EXPECT_EQ(hash.ToString(), text);
            EXPECT_FALSE(hash.empty());
        }
    }
}

TEST(ContentHash, OpaqueKeysKeepTheirBytesAndEquality) {
    const std::string canonical =
        ContentHash(ContentHash::Digest{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}).ToString();
    std::string uppercase = canonical;
    std::ranges::transform(uppercase, uppercase.begin(), [](char c) { return c >= 'a' ? c - 'a' + 'A' : c; });
    std::string high_bit = canonical;
    high_bit[5] = static_cast<char>('0' | 0x80);
    const std::vector<std::string> opaque{"",
                                          "0",
                                          canonical.substr(0, 63),
                                          canonical + "0",
                                          canonical.substr(0, 62),
                                          uppercase,
                                          canonical.substr(0, 63) + "G",
                                          canonical.substr(0, 63) + "g",
                                          high_bit,
                                          std::string(64, '\0'),
                                          "not a digest"};
    for (const auto& text : opaque) {
        for (const ContentHash& hash :
             {ContentHash(text), ContentHash(std::string(text)), ContentHash(std::string_view(text))}) {
            EXPECT_EQ(hash.ToString(), text);
            EXPECT_EQ(hash, ContentHash(text));
            EXPECT_NE(hash, ContentHash(canonical));
            EXPECT_EQ(hash.empty(), text.empty());
        }
    }
    EXPECT_NE(ContentHash(uppercase), ContentHash(canonical)) << "uppercase text is not the digest";
}

TEST(ContentHash, OneChangedCharacterStaysCanonicalExactlyForLowercaseDigits) {
    const ContentHash::Digest digest{0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10, 0x0f, 0x1e, 0x2d,
                                     0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4, 0xc3, 0xd2,
                                     0xe1, 0xf0, 0x00, 0xff, 0x11, 0xee, 0x22, 0xdd, 0x33, 0xcc};
    const std::string canonical = ContentHash(digest).ToString();
    for (std::size_t position = 0; position < canonical.size(); ++position) {
        for (unsigned value = 0; value < 256; ++value) {
            auto text = canonical;
            text[position] = static_cast<char>(value);
            const ContentHash hash(text);
            const auto with_nibble = [&](int nibble) {
                auto changed = digest;
                auto& byte = changed[position / 2];
                byte =
                    static_cast<std::uint8_t>(position % 2 == 0 ? (byte & 0x0f) | nibble << 4 : (byte & 0xf0) | nibble);
                return changed;
            };
            int nibble = -1;
            if (value >= '0' && value <= '9') nibble = static_cast<int>(value - '0');
            if (value >= 'a' && value <= 'f') nibble = static_cast<int>(value - 'a' + 10);
            if (nibble < 0) {
                // Opaque: it keeps its bytes and equals no digest, not even
                // the one an uppercase digit would spell.
                EXPECT_EQ(hash.ToString(), text) << position << " " << value;
                EXPECT_NE(hash, ContentHash(digest)) << position << " " << value;
                if (value >= 'A' && value <= 'F') {
                    EXPECT_NE(hash, ContentHash(with_nibble(static_cast<int>(value - 'A' + 10))))
                        << position << " " << value;
                }
                continue;
            }
            const auto changed = with_nibble(nibble);
            EXPECT_EQ(hash, ContentHash(changed)) << position << " " << value;
            EXPECT_EQ(hash.Hash(), ContentHash(changed).Hash());
        }
    }
}

TEST(ContentHash, EveryNoncanonicalByteRemainsOpaqueAtEveryPosition) {
    const std::string canonical(64, 'a');
    for (std::size_t position = 0; position < canonical.size(); ++position) {
        for (unsigned byte = 0; byte < 256; ++byte) {
            auto text = canonical;
            text[position] = static_cast<char>(byte);
            EXPECT_EQ(ContentHash(text).ToString(), text);
        }
    }
}

TEST(FlatPointerIndex, CollidingKeysSurviveGrowthWrappedClustersAndChurn) {
    struct Entry {
        int key;
    };
    struct KeyOfEntry {
        const int& operator()(const Entry* entry) const { return entry->key; }
    };
    struct CollisionHash {
        std::size_t operator()(int) const { return 15; }
    };
    std::array<Entry, 256> entries;
    FlatPointerIndex<const Entry*, int, KeyOfEntry, CollisionHash> index{/*empty=*/nullptr};
    std::map<int, const Entry*> expected;
    for (int i = 0; i < 256; ++i) {
        entries[i].key = i;
        ASSERT_TRUE(index.Insert(&entries[i]));
        expected.emplace(i, &entries[i]);
    }
    std::mt19937 random(8731);
    for (int step = 0; step < 4096; ++step) {
        const int key = random() % entries.size();
        if (random() % 2) {
            EXPECT_EQ(index.Insert(&entries[key]), expected.emplace(key, &entries[key]).second);
        } else {
            const auto found = expected.find(key);
            const Entry* removed = found == expected.end() ? nullptr : found->second;
            EXPECT_EQ(index.Extract(key), removed);
            expected.erase(key);
        }
        EXPECT_EQ(index.size(), expected.size());
        if (step % 31 == 0) {
            for (int i = 0; i < 256; ++i) {
                const auto found = expected.find(i);
                EXPECT_EQ(index.Find(i), found == expected.end() ? nullptr : found->second);
            }
        }
    }
    for (const auto& [key, entry] : expected) {
        EXPECT_EQ(index.Find(key), entry);
        EXPECT_TRUE(index.Erase(key));
    }
    EXPECT_TRUE(index.empty());
    EXPECT_FALSE(index.Erase(0));
    EXPECT_TRUE(index.Insert(&entries[0]));
    EXPECT_EQ(index.Find(0), &entries[0]);
}

// At every capacity, half the keys are homed at the last slot, a quarter at the
// one before it and a quarter at slot 3: clusters of equal and of mixed hashes
// wrap around the end of the table into the keys homed in front of it while
// the table grows.
TEST(FlatPointerIndex, ExtractObjectRemovesOnlyThatObjectAmongCollidingKeys) {
    struct Entry {
        int key;
    };
    struct KeyOfEntry {
        const int& operator()(const Entry* entry) const { return entry->key; }
    };
    struct WrappingHash {
        std::size_t operator()(int key) const {
            constexpr std::array<std::size_t, 4> homes{~std::size_t{0}, ~std::size_t{0} - 1, ~std::size_t{0}, 3};
            return homes[static_cast<std::size_t>(key) % homes.size()];
        }
    };
    using Index = FlatPointerIndex<const Entry*, int, KeyOfEntry, WrappingHash>;
    std::array<Entry, 300> entries;
    Index index{/*empty=*/nullptr};
    const Entry absent{0};
    EXPECT_EQ(index.ExtractObject(absent.key, absent), nullptr);
    std::map<int, const Entry*> expected;
    for (int i = 0; i < 300; ++i) {
        entries[i].key = i;
        ASSERT_TRUE(index.Insert(&entries[i]));
        expected.emplace(i, &entries[i]);
    }
    const auto expect_contents = [&] {
        ASSERT_EQ(index.size(), expected.size());
        for (int i = 0; i < 300; ++i) {
            const auto found = expected.find(i);
            EXPECT_EQ(index.Find(i), found == expected.end() ? nullptr : found->second) << i;
        }
    };

    // An equal key on another object names nothing; the indexed one stays.
    const Entry impostor{entries[120].key};
    EXPECT_EQ(index.ExtractObject(impostor.key, impostor), nullptr);
    expect_contents();
    // One object out of a cluster of equal hashes, taken from its middle.
    EXPECT_EQ(index.ExtractObject(entries[120].key, entries[120]), &entries[120]);
    expected.erase(120);
    expect_contents();
    EXPECT_EQ(index.ExtractObject(entries[120].key, entries[120]), nullptr);

    std::mt19937 random(5309);
    for (int step = 0; step < 6000; ++step) {
        const int key = static_cast<int>(random() % entries.size());
        const bool present = expected.contains(key);
        switch (random() % 3) {
            case 0:
                EXPECT_EQ(index.Insert(&entries[key]), !present);
                expected.emplace(key, &entries[key]);
                break;
            case 1:
                EXPECT_EQ(index.ExtractObject(entries[key].key, entries[key]), present ? &entries[key] : nullptr);
                expected.erase(key);
                break;
            default:
                EXPECT_EQ(index.Extract(key), present ? &entries[key] : nullptr);
                expected.erase(key);
                break;
        }
        if (step % 97 == 0) expect_contents();
    }
    expect_contents();
    for (const auto& [key, entry] : expected) EXPECT_EQ(index.ExtractObject(key, *entry), entry);
    EXPECT_TRUE(index.empty());
}

// The owning list's end() is the empty handle: every handle the index stores,
// returns or skips over compares against it as an iterator of the same list.
TEST(FlatPointerIndex, ListIteratorsUseTheListEndForAbsentKeys) {
    using List = std::list<int>;
    struct KeyOfNode {
        const int& operator()(List::iterator node) const { return *node; }
    };
    struct ShortHash {
        std::size_t operator()(int key) const { return static_cast<std::size_t>(key % 5); }
    };
    List nodes;
    FlatPointerIndex<List::iterator, int, KeyOfNode, ShortHash> index{nodes.end()};
    EXPECT_EQ(index.Find(3), nodes.end());
    for (int key = 0; key < 40; ++key) {
        nodes.push_back(key);
        ASSERT_TRUE(index.Insert(std::prev(nodes.end())));
    }
    EXPECT_EQ(index.Find(40), nodes.end());
    for (auto node = nodes.begin(); node != nodes.end();) {
        if (*node % 3 == 0) {
            EXPECT_EQ(index.ExtractObject(*node, *node), node);
            node = nodes.erase(node);
        } else {
            ++node;
        }
    }
    for (int key = 0; key < 40; ++key) {
        const auto node = index.Find(key);
        if (key % 3 == 0) {
            EXPECT_EQ(node, nodes.end()) << key;
        } else {
            ASSERT_NE(node, nodes.end()) << key;
            EXPECT_EQ(*node, key);
        }
    }
}

TEST(PrefixCacheIndexInspectionTest, CompleteDigestIdentitySurvivesCommonLeadingBytes) {
    BlockPool pool(32, {1});
    PrefixCacheIndex index(0);
    std::vector<CacheBlockLocation> locations;
    for (int i = 0; i < 16; ++i) {
        std::string hash(64, '0');
        hash.back() = "0123456789abcdef"[i];
        locations.push_back(Cache(index, pool, hash, i + 1));
    }
    for (int i = 15; i >= 0; --i) {
        std::string hash(64, '0');
        hash.back() = "0123456789abcdef"[i];
        EXPECT_EQ(index.KeyOf(pool, locations[i]), KeyOf(hash));
        EXPECT_EQ(index.Evict(pool, locations[i]), KeyOf(hash));
        EXPECT_FALSE(index.Contains(pool, KeyOf(hash)));
    }
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 32);
}

TEST(PrefixCacheIndexInspectionTest, LongDigestsKeepNamespacesAndOffsetsDistinctAfterEviction) {
    constexpr std::int32_t count = 128;
    BlockPool pool(count, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheKey> keys;
    std::vector<CacheBlockLocation> locations;
    for (std::int32_t i = 0; i < count; ++i) {
        std::string digest(64, 'a');
        digest.back() = "0123456789abcdef"[i / 8];
        keys.push_back(CacheKey{.namespace_id = static_cast<CacheNamespaceId>(i % 2),
                                .group_id = kGroupId,
                                .content_hash = std::move(digest),
                                .page_offset = (i / 2) % 4});
        CacheBlockRef block = pool.AcquireBlock(kGroupId);
        ASSERT_TRUE(block);
        locations.push_back(block->Location());
        index.Register(pool, block, keys.back(), static_cast<std::uint64_t>(i + 1),
                       /*logical_block_index=*/i, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr,
                       /*session=*/nullptr);
    }

    for (std::int32_t i = 0; i < count; ++i) {
        const auto entry = index.Inspect(pool, keys[i]);
        ASSERT_TRUE(entry);
        EXPECT_EQ(entry->location, locations[i]);
        EXPECT_EQ(entry->metadata.logical_block_index, i);
        CacheKey missing = keys[i];
        missing.namespace_id += 2;
        EXPECT_FALSE(index.Contains(pool, missing));
        missing = keys[i];
        missing.page_offset += 4;
        EXPECT_FALSE(index.Contains(pool, missing));
        missing = keys[i];
        auto missing_hash = missing.content_hash.ToString();
        missing_hash.front() = 'b';
        missing.content_hash = std::move(missing_hash);
        EXPECT_FALSE(index.Contains(pool, missing));
    }

    std::vector<CacheBlockLocation> remaining;
    for (std::int32_t i = 0; i < count; ++i) {
        if (i % 2 == 0) {
            EXPECT_EQ(index.Evict(pool, locations[i]), keys[i]);
        } else {
            remaining.push_back(locations[i]);
        }
    }
    for (std::int32_t i = 0; i < count; ++i) {
        EXPECT_EQ(index.Contains(pool, keys[i]), i % 2 != 0);
    }
    EXPECT_EQ(DrainEvictionOrder(index, pool), remaining);
}

TEST(PrefixCacheIndexEvictionOrderTest, DeliversOneEpochPerBatchSortedByLocation) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(kGroupId);
    Cache(index, pool, "later", /*access_epoch=*/9);
    const CacheBlockLocation first = Cache(index, pool, "same-a", /*access_epoch=*/7);
    const CacheBlockLocation second = Cache(index, pool, "same-b", /*access_epoch=*/7);
    const CacheBlockLocation third = Cache(index, pool, "same-c", /*access_epoch=*/7);

    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));

    ASSERT_EQ(batch.size(), 3u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    EXPECT_EQ(batch[2].location, third);
    EXPECT_TRUE(std::ranges::all_of(batch, [](const PrefixCacheIndex::EvictionCandidate& candidate) {
        return candidate.metadata.last_access_epoch == 7;
    }));
}

TEST(PrefixCacheIndexEvictionOrderTest, SkipsPinnedEntriesWithoutLosingTheRest) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation unpinned = Cache(index, pool, "unpinned", /*access_epoch=*/20);

    CacheBlockRef pinned = pool.AcquireBlock(kGroupId);
    ASSERT_TRUE(pinned);
    index.Register(pool, pinned, KeyOf("pinned"), /*access_epoch=*/10, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr, /*session=*/nullptr);

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{unpinned}));

    pinned.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool).size(), 2u);
}

TEST(PrefixCacheIndexEvictionOrderTest, EvictionCapacityBoundIncludesReleasedAndRepublishedEntries) {
    BlockPool pool(3, {1});
    BlockPool other_pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    Cache(index, pool, "first", 1);
    Cache(index, pool, "second", 2);
    auto held = pool.AcquireBlock(kGroupId);
    ASSERT_TRUE(held);
    index.Register(pool, held, KeyOf("held"), 3, 0, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    EXPECT_EQ(index.PotentiallyEvictableEntries(other_pool), 0u);
    EXPECT_GE(index.PotentiallyEvictableEntries(pool), 2u);
    EXPECT_EQ(DrainEvictionOrder(index, pool).size(), 2u);
    EXPECT_GE(index.PotentiallyEvictableEntries(pool), 2u);

    held.reset();
    EXPECT_GE(index.PotentiallyEvictableEntries(pool), 3u);
    EXPECT_EQ(DrainEvictionOrder(index, pool).size(), 3u);
    held = index.Find(pool, KeyOf("held"));
    EXPECT_EQ(DrainEvictionOrder(index, pool).size(), 2u);
    index.Register(pool, held, KeyOf("held"), 5, 0, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    held.reset();
    EXPECT_GE(index.PotentiallyEvictableEntries(pool), 3u);

    for (const CacheBlockLocation location : DrainEvictionOrder(index, pool)) {
        ASSERT_TRUE(index.Evict(pool, location));
    }
    EXPECT_EQ(index.PotentiallyEvictableEntries(pool), 0u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);
}

TEST(PrefixCacheIndexEvictionOrderTest, ReleasedPinnedEntryUsesItsLatestPublishedEpoch) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const auto middle = Cache(index, pool, "middle", 20);
    const auto last = Cache(index, pool, "last", 30);
    auto pinned = pool.AcquireBlock(kGroupId);
    const auto location = pinned->Location();
    index.Register(pool, pinned, KeyOf("pinned"), 40, 0, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{middle, last}));
    index.Register(pool, pinned, KeyOf("pinned"), 10, 0, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    pinned.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{location, middle, last}));
    auto repinned = index.Find(pool, KeyOf("pinned"));
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{middle, last}));
    repinned.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{location, middle, last}));
}

TEST(PrefixCacheIndexEvictionOrderTest, MovingIndexKeepsReleasedEntriesEvictable) {
    BlockPool pool(2, {1});
    PrefixCacheIndex source(kGroupId);
    auto pinned = pool.AcquireBlock(kGroupId);
    const auto location = pinned->Location();
    source.Register(pool, pinned, KeyOf("pinned"), 10, 0, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    EXPECT_TRUE(DrainEvictionOrder(source, pool).empty());
    PrefixCacheIndex moved(std::move(source));
    PrefixCacheIndex assigned(kGroupId);
    Cache(assigned, pool, "discarded", 1);
    assigned = std::move(moved);
    pinned.reset();
    EXPECT_EQ(DrainEvictionOrder(assigned, pool), (std::vector<CacheBlockLocation>{location}));
    EXPECT_TRUE(assigned.Evict(pool, location));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);
}

TEST(PrefixCacheIndexEvictionOrderTest, SharedIndexesRestoreTheRemainingCacheOwner) {
    BlockPool pool(1, {1});
    auto first = std::make_unique<PrefixCacheIndex>(kGroupId);
    PrefixCacheIndex second(kGroupId);
    auto block = pool.AcquireBlock(kGroupId);
    const auto location = block->Location();
    first->Register(pool, block, KeyOf("shared"), 10, 0, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    second.Register(pool, block, KeyOf("shared"), 20, 0, CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
    block.reset();
    EXPECT_TRUE(DrainEvictionOrder(*first, pool).empty());
    EXPECT_TRUE(DrainEvictionOrder(second, pool).empty());
    first.reset();
    EXPECT_EQ(DrainEvictionOrder(second, pool), (std::vector<CacheBlockLocation>{location}));
}

TEST(PrefixCacheIndexEvictionOrderTest, RepeatedOwnershipAndEpochChangesMatchFullScan) {
    BlockPool pool(64, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pins(64);
    for (int i = 0; i < 64; ++i) {
        Cache(index, pool, std::to_string(i), i % 7);
    }
    std::mt19937 random(20261030);
    for (int iteration = 0; iteration < 600; ++iteration) {
        const auto slot = random() % pins.size();
        const auto key = KeyOf(std::to_string(slot));
        if (pins[slot]) {
            pins[slot].reset();
        } else if (iteration % 2 == 0) {
            auto match = index.AcquireMatched(pool, std::vector<CacheKey>{key}, 0, GroupPrefixProbe{.hits = {1}},
                                              random() % 13, nullptr);
            pins[slot] = std::move(match.blocks.front());
        } else {
            pins[slot] = index.Find(pool, key);
        }
        if (iteration % 3 == 0) {
            auto block = index.Find(pool, key);
            index.Register(pool, block, key, random() % 13, static_cast<std::int32_t>(slot),
                           CacheBoundaryKind::kEndpoint, nullptr, /*session=*/nullptr);
        }
        auto expected = index.EvictableCandidates(pool);
        std::ranges::sort(expected, [](const auto& lhs, const auto& rhs) {
            if (lhs.metadata.last_access_epoch != rhs.metadata.last_access_epoch) {
                return lhs.metadata.last_access_epoch < rhs.metadata.last_access_epoch;
            }
            if (lhs.location.lcm_block_id != rhs.location.lcm_block_id) {
                return lhs.location.lcm_block_id < rhs.location.lcm_block_id;
            }
            return lhs.location.slot_index < rhs.location.slot_index;
        });
        std::vector<CacheBlockLocation> expected_locations;
        for (const auto& candidate : expected) {
            expected_locations.push_back(candidate.location);
        }
        EXPECT_EQ(DrainEvictionOrder(index, pool), expected_locations) << iteration;
    }
}

TEST(PrefixCacheIndexEvictionOrderTest, BoundedTraversalLeavesNewerEpochsForTheNextCall) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(kGroupId);
    CacheBlockRef pinned = pool.AcquireBlock(kGroupId);
    index.Register(pool, pinned, KeyOf("pinned"), 1, -1, CacheBoundaryKind::kChunk, nullptr, /*session=*/nullptr);
    const CacheBlockLocation first = Cache(index, pool, "first", 4);
    const CacheBlockLocation second = Cache(index, pool, "same-epoch", 4);
    const CacheBlockLocation later = Cache(index, pool, "later", 9);

    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 1u);
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 1u);
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, 3, batch));
    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 4u);
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, 4, batch));
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 9u);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, 8, batch));
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 9u);
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, 9, batch));
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].location, later);
    EXPECT_FALSE(index.PeekEvictionEpoch(pool, cursor));
}

TEST(PrefixCacheIndexEvictionOrderTest, AZeroEpochAndAnAbsentTierAreDistinct) {
    BlockPool pool(1, {1});
    BlockPool empty_pool(1, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation first = Cache(index, pool, "superseded", 0);
    PrefixCacheIndex::EvictionCursor cursor;
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 0u);
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, 0, batch));
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].location, first);
    PrefixCacheIndex::EvictionCursor empty_cursor;
    EXPECT_FALSE(index.PeekEvictionEpoch(empty_pool, empty_cursor));
    EXPECT_FALSE(index.NextEvictionEpoch(empty_pool, empty_cursor, 0, batch));
    EXPECT_EQ(batch.size(), 1u) << "an exhausted tier does not clear prior output";
}

TEST(PrefixCacheIndexEvictionOrderTest, SkipsKnownPinnedEpochsUntilTheirOwnersReleaseThem) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation held = Cache(index, pool, "held", 1);
    const CacheBlockLocation other = Cache(index, pool, "other", 8);
    CacheBlockRef owner = index.Find(pool, KeyOf("held"));
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{other}));

    {
        PrefixCacheIndex::EvictionCursor cursor;
        EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 8u);
        std::vector<PrefixCacheIndex::EvictionCandidate> batch;
        EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, 7, batch));
        EXPECT_TRUE(batch.empty());
        ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, 8, batch));
        ASSERT_EQ(batch.size(), 1u);
        EXPECT_EQ(batch.front().location, other);
        EXPECT_FALSE(index.PeekEvictionEpoch(pool, cursor));
    }
    index.Register(pool, owner, KeyOf("held"), 3, -1, CacheBoundaryKind::kChunk, nullptr, /*session=*/nullptr);
    owner.reset();
    PrefixCacheIndex::EvictionCursor released;
    EXPECT_EQ(index.PeekEvictionEpoch(pool, released), 3u);
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{held, other}));
}

TEST(PrefixCacheIndexEvictionOrderTest, AnEpochBatchRanksAMovedEntryAtItsNewLocationWithoutHoldingItsSession) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(kGroupId);
    const auto session = std::make_shared<SessionCacheTag>();
    CacheBlockRef block = pool.AcquireBlock(kGroupId);
    const CacheBlockLocation from = block->Location();
    index.Register(pool, block, KeyOf("moved"), /*access_epoch=*/5, /*logical_block_index=*/3,
                   CacheBoundaryKind::kEndpoint, /*newly_cached=*/nullptr, session);
    block.reset();
    const CacheBlockLocation older = Cache(index, pool, "older", /*access_epoch=*/2);
    // Acquired at a newer epoch and released, then moved while no traversal
    // has set it aside: its order node is re-keyed in the live order.
    index.AcquireMatched(pool, std::vector<CacheKey>{KeyOf("moved")}, 0, GroupPrefixProbe{.hits = {1}},
                         /*access_epoch=*/7, session);
    CacheBlockRef copy = pool.AcquireBlock(kGroupId);
    const CacheBlockLocation to = copy->Location();
    index.Relocate(pool, from, copy);
    copy.reset();
    ASSERT_FALSE(pool.IsOccupied(from));

    const long session_owners = session.use_count();
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].location, older);
    EXPECT_EQ(batch[0].metadata.last_access_epoch, 2u);
    EXPECT_EQ(batch[0].metadata.logical_block_index, -1);
    EXPECT_FALSE(batch[0].metadata.was_acquired);
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(batch[1].location, to);
    EXPECT_EQ(batch[1].metadata.last_access_epoch, 7u);
    EXPECT_EQ(batch[1].metadata.logical_block_index, 3);
    EXPECT_EQ(batch[1].metadata.boundary_kind, CacheBoundaryKind::kEndpoint);
    EXPECT_TRUE(batch[1].metadata.was_acquired);
    EXPECT_EQ(session.use_count(), session_owners) << "a retained batch keeps no session alive";
    EXPECT_EQ(index.MetadataFor(pool, to)->session, session);
    EXPECT_EQ(index.Evict(pool, to), KeyOf("moved"));
}

TEST(PrefixCacheIndexCommonOrderTest, TracksBothTiersAndUsesOnlySourceRecency) {
    BlockPool source(4, {1});
    BlockPool peer(4, {1});
    PrefixCacheIndex index(kGroupId);
    const auto a = Cache(index, source, "a", 10);
    const auto b = Cache(index, source, "b", 20);
    Cache(index, source, "source-only", 100);
    const auto peer_b = Cache(index, peer, "b", 90);
    index.TrackCommonEntries(source, peer);
    const auto common_keys = [&] {
        std::vector<std::string> keys;
        PrefixCacheIndex::CommonCursor cursor;
        while (const auto entry = index.NextCommonRecentEntry(source, cursor)) {
            keys.push_back(entry->key->content_hash);
        }
        return keys;
    };
    EXPECT_EQ(common_keys(), (std::vector<std::string>{"b"}));
    const auto peer_a = Cache(index, peer, "a", 200);
    Cache(index, peer, "peer-only", 300);
    EXPECT_EQ(common_keys(), (std::vector<std::string>{"b", "a"}));

    CacheBlockRef source_pin = index.Find(source, KeyOf("a"));
    CacheBlockRef peer_pin = index.Find(peer, KeyOf("b"));
    index.Register(source, source_pin, KeyOf("a"), 30, -1, CacheBoundaryKind::kChunk, nullptr, /*session=*/nullptr);
    EXPECT_EQ(common_keys(), (std::vector<std::string>{"a", "b"}));
    {
        PrefixCacheIndex::CommonCursor cursor;
        const auto first = index.NextCommonRecentEntry(source, cursor);
        const auto second = index.NextCommonRecentEntry(source, cursor);
        ASSERT_TRUE(first);
        ASSERT_TRUE(second);
        EXPECT_EQ(first->source.location, a);
        EXPECT_EQ(first->peer.location, peer_a);
        EXPECT_TRUE(first->source.pinned);
        EXPECT_FALSE(first->peer.pinned);
        EXPECT_EQ(second->source.location, b);
        EXPECT_EQ(second->peer.location, peer_b);
        EXPECT_FALSE(second->source.pinned);
        EXPECT_TRUE(second->peer.pinned);
        // Borrowed snapshots must not prevent removing the peer copy.
        EXPECT_EQ(index.Evict(peer, peer_a), KeyOf("a"));
    }
    EXPECT_EQ(common_keys(), (std::vector<std::string>{"b"}));
    Cache(index, peer, "a", 500);
    EXPECT_EQ(common_keys(), (std::vector<std::string>{"a", "b"}));
    source_pin.reset();
    peer_pin.reset();
    EXPECT_EQ(index.Evict(source, b), KeyOf("b"));
    EXPECT_EQ(common_keys(), (std::vector<std::string>{"a"}));
    Cache(index, source, "b", 0);
    EXPECT_EQ(common_keys(), (std::vector<std::string>{"a", "b"}));
}

TEST(PrefixCacheIndexCommonOrderTest, KeepsEvictionAndCommonOrderCurrentWhenPinnedEntriesChangeAge) {
    BlockPool source(2, {1});
    BlockPool peer(2, {1});
    PrefixCacheIndex index(kGroupId);
    index.TrackCommonEntries(source, peer);
    const auto held = Cache(index, source, "held", 4);
    const auto other = Cache(index, source, "other", 7);
    Cache(index, peer, "held", 40);
    Cache(index, peer, "other", 70);

    CacheBlockRef pin = index.Find(source, KeyOf("held"));
    EXPECT_EQ(DrainEvictionOrder(index, source), (std::vector<CacheBlockLocation>{other}));
    index.Register(source, pin, KeyOf("held"), 9, -1, CacheBoundaryKind::kChunk, nullptr, /*session=*/nullptr);
    {
        PrefixCacheIndex::CommonCursor cursor;
        const auto first = index.NextCommonRecentEntry(source, cursor);
        const auto second = index.NextCommonRecentEntry(source, cursor);
        ASSERT_TRUE(first);
        ASSERT_TRUE(second);
        EXPECT_EQ(*first->key, KeyOf("held"));
        EXPECT_EQ(first->source.metadata.last_access_epoch, 9u);
        EXPECT_EQ(first->peer.metadata.last_access_epoch, 40u);
        EXPECT_TRUE(first->source.pinned);
        EXPECT_FALSE(first->peer.pinned);
        EXPECT_EQ(*second->key, KeyOf("other"));
        EXPECT_FALSE(index.NextCommonRecentEntry(source, cursor));
    }
    pin.reset();
    EXPECT_EQ(DrainEvictionOrder(index, source), (std::vector<CacheBlockLocation>{other, held}));

    // A superseded entry can move back to epoch zero while pinned. Releasing
    // its owner must restore that age to the eviction order as well.
    pin = index.Find(source, KeyOf("held"));
    EXPECT_EQ(DrainEvictionOrder(index, source), (std::vector<CacheBlockLocation>{other}));
    index.Register(source, pin, KeyOf("held"), 0, -1, CacheBoundaryKind::kChunk, nullptr, /*session=*/nullptr);
    pin.reset();
    EXPECT_EQ(DrainEvictionOrder(index, source), (std::vector<CacheBlockLocation>{held, other}));
    PrefixCacheIndex::CommonCursor cursor;
    const auto first = index.NextCommonRecentEntry(source, cursor);
    const auto second = index.NextCommonRecentEntry(source, cursor);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(*first->key, KeyOf("other"));
    EXPECT_EQ(*second->key, KeyOf("held"));
    EXPECT_EQ(second->source.metadata.last_access_epoch, 0u);
    EXPECT_FALSE(second->source.pinned);
    EXPECT_FALSE(index.NextCommonRecentEntry(source, cursor));
}

// The common order follows a relocated source entry: the pair stays, listed
// at the new location, and leaves with the entry.
TEST(PrefixCacheIndexCommonOrderTest, KeepsTheCommonOrderCurrentWhenAnEntryMoves) {
    BlockPool source(3, {1});
    BlockPool peer(1, {1});
    PrefixCacheIndex index(kGroupId);
    index.TrackCommonEntries(source, peer);
    const CacheBlockLocation from = Cache(index, source, "moved", 4);
    Cache(index, source, "other", 7);
    Cache(index, peer, "moved", 40);
    CacheBlockRef pin = index.Find(source, KeyOf("moved"));
    CacheBlockRef copy = source.AcquireBlock(kGroupId);
    const CacheBlockLocation to = copy->Location();
    index.Relocate(source, from, copy);
    pin = std::move(copy);
    {
        PrefixCacheIndex::CommonCursor cursor;
        const auto common = index.NextCommonRecentEntry(source, cursor);
        ASSERT_TRUE(common);
        EXPECT_EQ(*common->key, KeyOf("moved"));
        EXPECT_EQ(common->source.location, to);
        EXPECT_EQ(common->source.metadata.last_access_epoch, 4u);
        EXPECT_TRUE(common->source.pinned);
        EXPECT_FALSE(index.NextCommonRecentEntry(source, cursor));
    }
    pin.reset();
    EXPECT_EQ(index.Evict(source, to), KeyOf("moved"));
    PrefixCacheIndex::CommonCursor cursor;
    EXPECT_FALSE(index.NextCommonRecentEntry(source, cursor));
}

TEST(PrefixCacheIndexSnapshotTest, VisitsBothOwnershipKindsWithoutTouchingOrPinningEntries) {
    BlockPool pool(3, {1});
    BlockPool absent(1, {1});
    PrefixCacheIndex index(kGroupId);
    const auto old = Cache(index, pool, "old", 0);
    const auto held = Cache(index, pool, "held", 6);
    const auto recent = Cache(index, pool, "recent", 9);
    CacheBlockRef pin = index.Find(pool, KeyOf("held"));
    std::vector<std::string> seen;
    index.ForEachEntry(pool, [&](const CacheKey& key, const PrefixCacheIndex::CachedBlockSnapshot& entry) {
        seen.push_back(key.content_hash);
        EXPECT_EQ(entry.pinned, key == KeyOf("held"));
        EXPECT_FALSE(entry.metadata.was_acquired);
        if (key == KeyOf("old")) {
            EXPECT_EQ(entry.location, old);
            EXPECT_EQ(entry.metadata.last_access_epoch, 0u);
        } else if (key == KeyOf("held")) {
            EXPECT_EQ(entry.location, held);
            EXPECT_EQ(entry.metadata.last_access_epoch, 6u);
        } else {
            EXPECT_EQ(key, KeyOf("recent"));
            EXPECT_EQ(entry.location, recent);
            EXPECT_EQ(entry.metadata.last_access_epoch, 9u);
        }
    });
    std::ranges::sort(seen);
    EXPECT_EQ(seen, (std::vector<std::string>{"held", "old", "recent"}));
    EXPECT_EQ(pin.use_count(), 2u);
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{old, recent}));
    EXPECT_FALSE(index.Evict(pool, held));
    pin = {};
    {
        CacheBlockRef block = index.Find(pool, KeyOf("held"));
        index.Register(pool, block, KeyOf("held"), 10, -1, CacheBoundaryKind::kChunk, nullptr, /*session=*/nullptr);
    }
    index.ForEachEntry(pool, [&](const CacheKey& key, const PrefixCacheIndex::CachedBlockSnapshot& entry) {
        EXPECT_FALSE(entry.pinned);
        if (key == KeyOf("held")) {
            EXPECT_EQ(entry.metadata.last_access_epoch, 10u);
        }
    });
    EXPECT_EQ(index.Evict(pool, old), KeyOf("old"));
    EXPECT_EQ(index.Evict(pool, held), KeyOf("held"));
    index.ForEachEntry(absent, [&](const CacheKey&, const PrefixCacheIndex::CachedBlockSnapshot&) {
        ADD_FAILURE() << "an absent tier contains no entries";
    });
}

TEST(PrefixCacheIndexEvictionOrderTest, SkipsPinnedEpochsAndReturnsTheWholeNextEpoch) {
    BlockPool pool(5, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pinned;
    for (std::uint64_t epoch : {1u, 2u, 3u}) {
        CacheBlockRef block = pool.AcquireBlock(kGroupId);
        index.Register(pool, block, KeyOf("pinned-" + std::to_string(epoch)), epoch,
                       /*logical_block_index=*/-1, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr,
                       /*session=*/nullptr);
        pinned.push_back(std::move(block));
    }
    const CacheBlockLocation first = Cache(index, pool, "first", /*access_epoch=*/3);
    const CacheBlockLocation second = Cache(index, pool, "second", /*access_epoch=*/3);
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    EXPECT_TRUE(batch.empty());
}

TEST(PrefixCacheIndexEvictionOrderTest, ExhaustsTheMaximumEpochWithoutWrapping) {
    BlockPool pool(1, {1});
    PrefixCacheIndex index(kGroupId);
    Cache(index, pool, "last", std::numeric_limits<std::uint64_t>::max());
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    ASSERT_EQ(batch.size(), 1u);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    EXPECT_TRUE(batch.empty());
}

TEST(PrefixCacheIndexEvictionOrderTest, ReRegisteringAtAnOlderEpochMovesTheEntryEarlier) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    CacheBlockRef refreshed = pool.AcquireBlock(kGroupId);
    ASSERT_TRUE(refreshed);
    const CacheBlockLocation moved = refreshed->Location();
    index.Register(pool, refreshed, KeyOf("moved"), /*access_epoch=*/30, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr, /*session=*/nullptr);
    const CacheBlockLocation settled = Cache(index, pool, "settled", /*access_epoch=*/20);

    // A request that continues under its original epoch re-registers its pages
    // at an epoch older than entries published since, so the order cannot be
    // maintained by append alone.
    index.Register(pool, refreshed, KeyOf("moved"), /*access_epoch=*/5, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr, /*session=*/nullptr);
    refreshed.reset();

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{moved, settled}));
}

TEST(PrefixCacheIndexEvictionOrderTest, AcquiringAMatchMovesTheEntryToTheRequestEpoch) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation matched = Cache(index, pool, "matched", /*access_epoch=*/10);
    const CacheBlockLocation untouched = Cache(index, pool, "untouched", /*access_epoch=*/20);

    const std::vector<CacheKey> keys{KeyOf("matched")};
    const GroupPrefixProbe probe{.hits = {1}};
    PrefixMatch match =
        index.AcquireMatched(pool, keys, /*begin_blocks=*/0, probe, /*access_epoch=*/40, /*session=*/nullptr);
    ASSERT_EQ(match.blocks.size(), 1u);
    match.blocks.clear();

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{untouched, matched}));
}

// Hits re-keyed to a new epoch, and entries published under it, append at
// the end of the order and become its insertion hint. The order stays exact
// when that last entry leaves it right away (re-keyed again, evicted, or set
// aside by a traversal while pinned), and when appends and restores follow.
TEST(PrefixCacheIndexEvictionOrderTest, AppendsAtTheNewestEpochAndRemovesTheLastEntry) {
    BlockPool pool(10, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockLocation> cached;
    std::vector<CacheKey> keys;
    for (int i = 0; i < 6; ++i) {
        cached.push_back(Cache(index, pool, std::to_string(i), /*access_epoch=*/1 + i));
        keys.push_back(KeyOf(std::to_string(i)));
    }
    for (int i = 0; i < 4; ++i) {
        index.AcquireMatched(pool, std::vector<CacheKey>{keys[i]}, /*begin_blocks=*/0, GroupPrefixProbe{.hits = {1}},
                             /*access_epoch=*/10 + i, /*session=*/nullptr);
    }
    EXPECT_EQ(DrainEvictionOrder(index, pool),
              (std::vector<CacheBlockLocation>{cached[4], cached[5], cached[0], cached[1], cached[2], cached[3]}));

    index.AcquireMatched(pool, std::vector<CacheKey>{keys[3]}, 0, GroupPrefixProbe{.hits = {1}}, 20, nullptr);
    const CacheBlockLocation evicted = Cache(index, pool, "evicted", 30);
    ASSERT_TRUE(index.Evict(pool, evicted));
    const CacheBlockLocation mid = Cache(index, pool, "mid", 15);
    EXPECT_EQ(DrainEvictionOrder(index, pool),
              (std::vector<CacheBlockLocation>{cached[4], cached[5], cached[0], cached[1], cached[2], mid, cached[3]}));

    const CacheBlockLocation pinned = Cache(index, pool, "pinned", 40);
    CacheBlockRef pin = index.Find(pool, KeyOf("pinned"));
    EXPECT_EQ(DrainEvictionOrder(index, pool),
              (std::vector<CacheBlockLocation>{cached[4], cached[5], cached[0], cached[1], cached[2], mid, cached[3]}));
    const CacheBlockLocation before_pinned = Cache(index, pool, "before-pinned", 35);
    const CacheBlockLocation appended = Cache(index, pool, "appended", 50);
    pin.reset();
    const std::vector<CacheBlockLocation> order = DrainEvictionOrder(index, pool);
    EXPECT_EQ(order, (std::vector<CacheBlockLocation>{cached[4], cached[5], cached[0], cached[1], cached[2], mid,
                                                      cached[3], before_pinned, pinned, appended}));

    for (const CacheBlockLocation location : order) {
        ASSERT_TRUE(index.Evict(pool, location));
    }
    EXPECT_EQ(index.NumEntries(pool), 0);
    EXPECT_EQ(index.PotentiallyEvictableEntries(pool), 0u);
}

// Pages freed suffix first are reused, and published, in descending location
// order: each lands just before the previous one, inside the order or at its
// end.
TEST(PrefixCacheIndexEvictionOrderTest, PublishingInDescendingLocationOrderKeepsTheOrder) {
    BlockPool pool(10, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation oldest = Cache(index, pool, "oldest", /*access_epoch=*/1);
    const CacheBlockLocation newest = Cache(index, pool, "newest", /*access_epoch=*/9);
    std::vector<CacheBlockRef> blocks;
    for (int i = 0; i < 8; ++i) {
        blocks.push_back(pool.AcquireBlock(kGroupId));
    }
    std::ranges::sort(blocks, {}, [](const CacheBlockRef& block) {
        const CacheBlockLocation location = block->Location();
        return std::pair{location.lcm_block_id, location.slot_index};
    });
    std::vector<CacheBlockLocation> locations;
    for (const CacheBlockRef& block : blocks) {
        locations.push_back(block->Location());
    }
    for (int i = 7; i >= 0; --i) {
        index.Register(pool, blocks[i], KeyOf(std::to_string(i)), /*access_epoch=*/i < 4 ? 5 : 10,
                       /*logical_block_index=*/i, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr,
                       /*session=*/nullptr);
    }
    blocks.clear();

    EXPECT_EQ(DrainEvictionOrder(index, pool),
              (std::vector<CacheBlockLocation>{oldest, locations[0], locations[1], locations[2], locations[3], newest,
                                               locations[4], locations[5], locations[6], locations[7]}));
}

TEST(PrefixCacheIndexEvictionOrderTest, AcquiredOwnersRestoreOrderOnLastReleaseAndUnwind) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const auto hit = Cache(index, pool, "hit", 10);
    const auto other = Cache(index, pool, "other", 20);
    const std::vector<CacheKey> keys{KeyOf("hit")};
    const GroupPrefixProbe probe{.hits = {1}};
    CacheBlockRef retained;
    try {
        auto match = index.AcquireMatched(pool, keys, 0, probe, 40, nullptr);
        retained = match.blocks.front();
        EXPECT_FALSE(index.Evict(pool, hit));
        auto older = index.AcquireMatched(pool, keys, 0, probe, 5, nullptr);
        throw 7;
    } catch (int) {
    }
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{other}));
    retained.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{hit, other}));
    EXPECT_EQ(index.Evict(pool, hit), KeyOf("hit"));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);
}

TEST(PrefixCacheIndexEvictionOrderTest, DestroyingIndexCancelsAnAcquiredOwnersNotification) {
    BlockPool pool(1, {1});
    PrefixMatch match;
    {
        PrefixCacheIndex index(kGroupId);
        Cache(index, pool, "hit", 10);
        match = index.AcquireMatched(pool, std::vector<CacheKey>{KeyOf("hit")}, 0, GroupPrefixProbe{.hits = {1}}, 20,
                                     nullptr);
        EXPECT_EQ(match.blocks.front().use_count(), 2);
        EXPECT_TRUE(DrainEvictionOrder(index, pool).empty());
    }
    EXPECT_EQ(match.blocks.front().use_count(), 1);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
    match.blocks.clear();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(PrefixCacheIndexInspectionTest, ProbeLocationsRecheckIdentityAfterSlotReuse) {
    BlockPool pool(2, {1});
    PrefixCacheIndex index(kGroupId);
    const auto original = Cache(index, pool, "hit", 10);
    const auto other = Cache(index, pool, "other", 20);
    const std::vector<CacheKey> keys{KeyOf("unused"), KeyOf("hit")};
    FullAttnMatcher matcher;
    const auto probe = matcher.Probe(index, pool, keys, 1, 2, nullptr);
    ASSERT_EQ(probe.hits, (std::vector<std::uint8_t>{1}));
    // A probe is non-owning: it neither prevents eviction nor keeps the old
    // control object alive when the location is reused for another identity.
    ASSERT_TRUE(index.Evict(pool, original));
    EXPECT_EQ(Cache(index, pool, "replacement", 30), original);
    ASSERT_TRUE(index.Evict(pool, other));
    EXPECT_EQ(Cache(index, pool, "hit", 40), other);
    EXPECT_EQ(index.MatchedLocations(pool, keys, 1, probe), (std::vector<CacheBlockLocation>{other}));
    auto match = index.AcquireMatched(pool, keys, 1, probe, 50, nullptr);
    ASSERT_EQ(match.blocks.size(), 1u);
    EXPECT_EQ(match.blocks.front()->Location(), other);
    EXPECT_FALSE(index.Evict(pool, other));
    match.blocks.clear();
    EXPECT_EQ(index.Evict(pool, other), KeyOf("hit"));
}

TEST(PrefixCacheIndexInspectionTest, ProbeLocationsDoNotConfusePoolsOrRelocatedEntries) {
    BlockPool device(3, {1});
    BlockPool host(3, {1});
    PrefixCacheIndex index(kGroupId);
    const auto from = Cache(index, device, "hit", 10);
    Cache(index, host, "other", 10);
    const auto host_hit = Cache(index, host, "hit", 20);
    const std::vector<CacheKey> keys{KeyOf("hit")};
    FullAttnMatcher matcher;
    const auto probe = matcher.Probe(index, device, keys, 0, 1, nullptr);
    auto copy = device.AcquireBlock(kGroupId);
    const auto to = copy->Location();
    index.Relocate(device, from, copy);
    copy.reset();
    EXPECT_EQ(index.MatchedLocations(device, keys, 0, probe), (std::vector<CacheBlockLocation>{to}));
    EXPECT_EQ(index.MatchedLocations(host, keys, 0, probe), (std::vector<CacheBlockLocation>{host_hit}));
    auto match = index.AcquireMatched(host, keys, 0, probe, 30, nullptr);
    ASSERT_EQ(match.blocks.size(), 1u);
    EXPECT_TRUE(match.blocks.front().IsOwnedBy(host));
    EXPECT_EQ(match.blocks.front()->Location(), host_hit);
}

TEST(PrefixCacheIndexEvictionOrderTest, EvictedEntriesLeaveTheOrder) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation evicted = Cache(index, pool, "evicted", /*access_epoch=*/10);
    const CacheBlockLocation kept = Cache(index, pool, "kept", /*access_epoch=*/20);

    ASSERT_TRUE(index.Evict(pool, evicted).has_value());

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{kept}));
    EXPECT_EQ(index.NumEntries(pool), 1);
}

// A lone request's compaction copies a published block and moves its entry
// onto the copy: the entry keeps its age, acquisition mark, pin and its hold
// on the boundary-tree node (which keeps its boundary), nothing is left at
// the old location, and the eviction order lists it where it is now once
// the pin goes.
TEST(PrefixCacheIndexEvictionOrderTest, RelocatesAnEntryOntoItsCopyWithItsAgePinAndHold) {
    BlockPool pool(4, {1});
    BoundaryTree tree(BoundaryTree::Config{.prefix_granularity = 8,
                                           .closed_blocks_per_page = 1,
                                           .num_state_groups = 0,
                                           .has_prefix_closed_groups = true,
                                           .has_host_tier = false,
                                           .queue_dead_candidates = false},
                      /*checkpoints=*/nullptr);
    PrefixCacheIndex index(kGroupId);
    index.TrackBoundaries(tree, pool, /*host_pool=*/nullptr, /*state=*/false, /*closed=*/true);
    const std::vector<std::string> hashes{"moved"};
    BoundaryTree::Hold publication =
        tree.LinkPublication(CompletedPages{.prefix_hashes = hashes, .first_new_prefix_page = 0});
    const CacheBlockLocation from = Cache(index, pool, "moved", /*access_epoch=*/10);
    const CacheBlockLocation other = Cache(index, pool, "other", /*access_epoch=*/20);
    publication.reset();
    const BoundaryTree::Node* node = tree.Find("moved");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->boundary_tokens, 8);

    // Pinned by a hit, and set aside by a traversal that found it pinned.
    CacheBlockRef pin =
        index
            .AcquireMatched(pool, std::vector<CacheKey>{KeyOf("moved")}, 0, GroupPrefixProbe{.hits = {1}},
                            /*access_epoch=*/10, /*session=*/nullptr)
            .blocks.front();
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{other}));
    CacheBlockRef copy = pool.AcquireBlock(kGroupId);
    const CacheBlockLocation to = copy->Location();
    index.Relocate(pool, from, copy);
    pin = std::move(copy);  // the pin follows; nothing holds the old block
    EXPECT_FALSE(pool.IsOccupied(from));
    EXPECT_FALSE(index.Contains(pool, from));
    EXPECT_EQ(index.KeyOf(pool, to), KeyOf("moved"));
    const auto moved = index.Inspect(pool, KeyOf("moved"));
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->location, to);
    EXPECT_EQ(moved->metadata.last_access_epoch, 10u);
    EXPECT_TRUE(moved->metadata.was_acquired);
    EXPECT_TRUE(moved->pinned);
    EXPECT_EQ(tree.Find("moved"), node);
    EXPECT_EQ(node->Entries(BoundaryTree::kDevice), 1);
    EXPECT_EQ(node->boundary_tokens, 8);
    // Set aside while pinned; back in age order, at the new location, once
    // the pin goes.
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{other}));
    pin.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{to, other}));
    EXPECT_EQ(index.Evict(pool, to), KeyOf("moved"));
    EXPECT_EQ(tree.Find("moved"), nullptr);
}

TEST(PrefixCacheIndexEvictionOrderTest, MatchesTheEvictableSetSeenByAFullScan) {
    BlockPool pool(64, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pinned;
    for (int i = 0; i < 40; ++i) {
        // Interleave epochs so registration order and eviction order differ.
        const std::uint64_t access_epoch = static_cast<std::uint64_t>((i * 17) % 23 + 1);
        if (i % 5 == 0) {
            CacheBlockRef block = pool.AcquireBlock(kGroupId);
            ASSERT_TRUE(block);
            index.Register(pool, block, KeyOf("pinned-" + std::to_string(i)), access_epoch,
                           /*logical_block_index=*/-1, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr,
                           /*session=*/nullptr);
            pinned.push_back(std::move(block));
            continue;
        }
        Cache(index, pool, "entry-" + std::to_string(i), access_epoch);
    }

    std::vector<CacheBlockLocation> streamed = DrainEvictionOrder(index, pool);
    std::vector<CacheBlockLocation> scanned;
    for (const PrefixCacheIndex::EvictionCandidate& candidate : index.EvictableCandidates(pool)) {
        scanned.push_back(candidate.location);
    }
    ASSERT_EQ(streamed.size(), scanned.size());

    const auto by_location = [](CacheBlockLocation lhs, CacheBlockLocation rhs) {
        return lhs.lcm_block_id != rhs.lcm_block_id ? lhs.lcm_block_id < rhs.lcm_block_id
                                                    : lhs.slot_index < rhs.slot_index;
    };
    std::ranges::sort(streamed, by_location);
    std::ranges::sort(scanned, by_location);
    EXPECT_EQ(streamed, scanned);

    pinned.clear();
}

}  // namespace
}  // namespace tokenspeed::test

namespace tokenspeed::test {
TEST(PrefixCacheSessionTest, TouchExistingMergesOwnershipBeforeRefreshingDemotedCopies) {
    for (const bool demoted : {false, true}) {
        for (const std::string visitor_id : {"first", "second", ""}) {
            SCOPED_TRACE(demoted);
            SCOPED_TRACE(visitor_id);
            BlockPool device(1, {1});
            BlockPool host(1, {1});
            PrefixCacheIndex index(kGroupId);
            auto owner = std::make_shared<SessionCacheTag>();
            owner->id = "first";
            SessionCacheOwner visitor;
            if (visitor_id == "first") {
                visitor = owner;
            } else if (!visitor_id.empty()) {
                visitor = std::make_shared<SessionCacheTag>();
                visitor->id = visitor_id;
            }
            const std::vector<CacheKey> keys{KeyOf("entry")};
            auto host_block = host.AcquireBlock(kGroupId);
            index.Register(host, host_block, keys[0], 10, 0, CacheBoundaryKind::kChunk, nullptr, owner);
            host_block.reset();
            if (demoted) {
                owner->demoted = true;
                ASSERT_EQ(index.DemoteSession(host, owner), 1u);
            }
            const std::vector<CacheBlockRef> sources{device.AcquireBlock(kGroupId)};
            index.TouchExisting(host, keys, sources, 20, visitor);
            const auto entry = index.Inspect(host, keys[0]);
            ASSERT_TRUE(entry);
            EXPECT_FALSE(entry->pinned);
            EXPECT_EQ(entry->metadata.session, visitor == owner ? owner : nullptr);
            EXPECT_EQ(entry->metadata.last_access_epoch, demoted && visitor == owner ? 0u : 20u);
            EXPECT_EQ(index.DemoteSession(host, owner), !demoted && visitor == owner ? 1u : 0u);
            if (visitor != owner) {
                EXPECT_EQ(index.Inspect(host, keys[0])->metadata.last_access_epoch, 20u);
            }
        }
    }
}

TEST(PrefixCacheSessionTest, TouchExistingKeepsNewerEpochsAndDoesNotInsertPinOrTouchMissingSources) {
    BlockPool device(2, {1});
    BlockPool host(3, {1});
    PrefixCacheIndex index(kGroupId);
    auto owner = std::make_shared<SessionCacheTag>();
    owner->id = "first";
    const std::vector<CacheKey> keys{KeyOf("present"), KeyOf("no-source"), KeyOf("absent")};
    for (std::size_t i = 0; i < 2; ++i) {
        auto block = host.AcquireBlock(kGroupId);
        index.Register(host, block, keys[i], 30, 0, CacheBoundaryKind::kChunk, nullptr, owner);
    }
    const std::vector<CacheBlockRef> sources{device.AcquireBlock(kGroupId), {}, device.AcquireBlock(kGroupId)};
    index.TouchExisting(host, keys, sources, 20, /*session=*/nullptr);
    const auto present = index.Inspect(host, keys[0]);
    const auto skipped = index.Inspect(host, keys[1]);
    ASSERT_TRUE(present);
    ASSERT_TRUE(skipped);
    EXPECT_EQ(present->metadata.last_access_epoch, 30u);
    EXPECT_FALSE(present->metadata.session);
    EXPECT_EQ(skipped->metadata.last_access_epoch, 30u);
    EXPECT_EQ(skipped->metadata.session, owner);
    EXPECT_FALSE(index.Contains(host, keys[2]));
    EXPECT_EQ(host.NumEmptyLcmBlocks(), 1);
    EXPECT_FALSE(present->pinned);
    EXPECT_FALSE(skipped->pinned);
    EXPECT_FALSE(present->metadata.was_acquired);
    EXPECT_EQ(sources[0].use_count(), 1);
    EXPECT_EQ(sources[2].use_count(), 1);
}

TEST(PrefixCacheSessionTest, DemotionAgesExclusiveIdleBlocksButProtectsSharedAndActiveBlocks) {
    BlockPool pool(5, {1});
    PrefixCacheIndex index(kGroupId);
    auto first = std::make_shared<SessionCacheTag>();
    first->id = "first";
    auto second = std::make_shared<SessionCacheTag>();
    second->id = "second";
    auto publish = [&](const std::string& key, const SessionCacheOwner& session, std::uint64_t epoch) {
        auto block = pool.AcquireBlock(kGroupId);
        index.Register(pool, block, KeyOf(key), epoch, 0, CacheBoundaryKind::kChunk, nullptr, session);
        return block;
    };
    auto exclusive = publish("exclusive", first, 20);
    auto exclusive_location = exclusive->Location();
    exclusive = {};
    auto shared = publish("shared", first, 30);
    index.Register(pool, shared, KeyOf("shared"), 40, 0, CacheBoundaryKind::kChunk, nullptr, second);
    auto shared_location = shared->Location();
    shared = {};
    auto active = publish("active", first, 50);
    auto active_location = active->Location();
    auto other = publish("other", second, 10);
    auto other_location = other->Location();
    other = {};
    first->demoted = true;
    EXPECT_EQ(index.DemoteSession(pool, first), 1u);
    EXPECT_EQ(index.MetadataFor(pool, exclusive_location)->last_access_epoch, 0u);
    EXPECT_EQ(index.MetadataFor(pool, shared_location)->last_access_epoch, 40u);
    EXPECT_EQ(index.MetadataFor(pool, active_location)->last_access_epoch, 50u);
    EXPECT_EQ(index.MetadataFor(pool, other_location)->last_access_epoch, 10u);
    EXPECT_EQ(DrainEvictionOrder(index, pool),
              (std::vector<CacheBlockLocation>{exclusive_location, other_location, shared_location}));
    EXPECT_FALSE(index.Evict(pool, active_location).has_value());
    active = {};
    EXPECT_EQ(index.DemoteSession(pool, first), 1u);
    EXPECT_EQ(index.DemoteSession(pool, first), 0u);
    EXPECT_TRUE(index.Contains(pool, KeyOf("exclusive")));
    first->demoted = false;
    auto hit = index.AcquireMatched(pool, std::vector<CacheKey>{KeyOf("exclusive")}, 0, GroupPrefixProbe{.hits = {1}},
                                    60, first);
    EXPECT_EQ(index.MetadataFor(pool, exclusive_location)->last_access_epoch, 60u);
    hit.blocks.clear();
    EXPECT_EQ(DrainEvictionOrder(index, pool),
              (std::vector<CacheBlockLocation>{active_location, other_location, shared_location, exclusive_location}));
}
}  // namespace tokenspeed::test
