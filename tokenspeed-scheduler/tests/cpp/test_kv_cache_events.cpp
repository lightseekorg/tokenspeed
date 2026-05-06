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

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "integration_test_helper.h"
#include "resource/allocator/page_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "scheduler/kv_cache_events.h"

namespace tokenspeed::test {
namespace {

class KVPrefixCacheEventTestSuite : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 2;

    PageAllocator device_allocator_{kPageSize, 3};
    PageAllocator host_allocator_{kPageSize, 3};
    KVPrefixCache cache_{&device_allocator_, &host_allocator_};
    std::vector<KvCacheEvent> events_;

    void SetUp() override {
        cache_.SetKvEventSink([this](KvCacheEvent event) { events_.push_back(std::move(event)); });
    }

    void InsertDevicePages(const token_vec_t& tokens, std::int32_t page_count) {
        cache_.Insert<ResourceType::Device>(tokens, {}, device_allocator_.Allocate(page_count));
    }
};

const KvBlockStoredEvent& AsStored(const KvCacheEvent& event) {
    return std::get<KvBlockStoredEvent>(event);
}

const KvBlockRemovedEvent& AsRemoved(const KvCacheEvent& event) {
    return std::get<KvBlockRemovedEvent>(event);
}

}  // namespace

TEST_F(KVPrefixCacheEventTestSuite, InsertOnePageEmitsBlockStored) {
    const token_vec_t tokens = MakeAlignedTokens(1, kPageSize);
    InsertDevicePages(tokens, 1);

    ASSERT_EQ(events_.size(), 1u);
    const auto& stored = AsStored(events_[0]);
    const std::uint64_t expected_hash = HashKvBlock(std::span<const std::int32_t>(tokens.data(), kPageSize));
    EXPECT_EQ(stored.block_hashes, std::vector<std::uint64_t>{expected_hash});
    EXPECT_EQ(stored.parent_block_hash, std::nullopt);
    EXPECT_EQ(stored.token_ids, tokens);
    EXPECT_EQ(stored.block_size, kPageSize);
}

TEST_F(KVPrefixCacheEventTestSuite, InsertSameTokensDoesNotDuplicateBlockStored) {
    const token_vec_t tokens = MakeAlignedTokens(1, kPageSize);
    InsertDevicePages(tokens, 1);
    events_.clear();

    InsertDevicePages(tokens, 1);

    EXPECT_TRUE(events_.empty());
}

TEST_F(KVPrefixCacheEventTestSuite, InsertTwoPagesEmitsRollingParentHash) {
    const token_vec_t tokens = MakeAlignedTokens(2, kPageSize);
    InsertDevicePages(tokens, 2);

    ASSERT_EQ(events_.size(), 2u);
    const auto& first = AsStored(events_[0]);
    const auto& second = AsStored(events_[1]);
    const std::uint64_t first_hash = HashKvBlock(std::span<const std::int32_t>(tokens.data(), kPageSize));
    const std::uint64_t second_hash =
        HashKvBlock(std::span<const std::int32_t>(tokens.data() + kPageSize, kPageSize), first_hash);

    EXPECT_EQ(first.block_hashes, std::vector<std::uint64_t>{first_hash});
    EXPECT_EQ(first.parent_block_hash, std::nullopt);
    EXPECT_EQ(second.block_hashes, std::vector<std::uint64_t>{second_hash});
    EXPECT_EQ(second.parent_block_hash, first_hash);
    EXPECT_EQ(second.token_ids, token_vec_t(tokens.begin() + kPageSize, tokens.end()));
}

TEST_F(KVPrefixCacheEventTestSuite, DeviceEvictionEmitsBlockRemoved) {
    const token_vec_t tokens = MakeAlignedTokens(2, kPageSize);
    InsertDevicePages(tokens, 2);
    const std::uint64_t first_hash = AsStored(events_[0]).block_hashes[0];
    const std::uint64_t second_hash = AsStored(events_[1]).block_hashes[0];
    events_.clear();

    ASSERT_TRUE(cache_.EnsureCapacityByEvict<ResourceType::Device>(1));

    ASSERT_EQ(events_.size(), 1u);
    EXPECT_EQ(AsRemoved(events_[0]).block_hashes, (std::vector<std::uint64_t>{first_hash, second_hash}));
}

TEST_F(KVPrefixCacheEventTestSuite, AlreadyRemovedBlocksAreNotRemovedTwice) {
    const token_vec_t tokens = MakeAlignedTokens(2, kPageSize);
    InsertDevicePages(tokens, 2);
    events_.clear();

    ASSERT_TRUE(cache_.EnsureCapacityByEvict<ResourceType::Device>(1));
    events_.clear();
    ASSERT_TRUE(cache_.EnsureCapacityByEvict<ResourceType::Device>(1));

    EXPECT_TRUE(events_.empty());
}

class SchedulerKvCacheEventTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_l3_storage = false;
        cfg.enable_kv_cache_events = true;
        return cfg;
    }
};

TEST_F(SchedulerKvCacheEventTestSuite, DrainKvEventsReturnsAndClearsSchedulerEvents) {
    auto spec = MakeRequestSpec("r1", 1);
    Submit(spec);
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();

    SendFinish("r1");
    auto events = scheduler_->DrainKvEvents();

    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(AsStored(events[0]).token_ids, spec.tokens);
    EXPECT_TRUE(scheduler_->DrainKvEvents().empty());
}

}  // namespace tokenspeed::test
