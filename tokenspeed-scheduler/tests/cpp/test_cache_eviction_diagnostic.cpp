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
#include <spdlog/sinks/ostream_sink.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>

#include "cache/coordinator/cache_coordinator.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

class CacheEvictionDiagnosticTest : public SchedulerTestSuite {
protected:
    void SetUp() override {
        for (std::size_t i = 0; i < kVariables.size(); ++i) {
            if (const char* value = std::getenv(kVariables[i])) previous_environment_[i] = value;
            unsetenv(kVariables[i]);
        }
        setenv(kVariable, hash_.c_str(), 1);
        previous_logger_ = spdlog::default_logger();
        auto logger = std::make_shared<spdlog::logger>("cache-eviction-test",
                                                       std::make_shared<spdlog::sinks::ostream_sink_mt>(output_));
        logger->set_pattern("%v");
        logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(std::move(logger));
    }
    void TearDown() override {
        spdlog::set_default_logger(std::move(previous_logger_));
        for (std::size_t i = 0; i < kVariables.size(); ++i) {
            if (previous_environment_[i])
                setenv(kVariables[i], previous_environment_[i]->c_str(), 1);
            else
                unsetenv(kVariables[i]);
        }
    }
    bool Has(std::initializer_list<std::string> fields) const {
        std::istringstream input{output_.str()};
        for (std::string line; std::getline(input, line);) {
            if (std::all_of(fields.begin(), fields.end(),
                            [&](const std::string& field) { return line.find(field) != std::string::npos; })) {
                return true;
            }
        }
        return false;
    }
    template <typename Body>
    void InFreshProcess(Body body) {
        // Process configuration is immutable. Re-exec rather than adding a
        // diagnostic reset API or inheriting another test's initialized flag.
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        ASSERT_EXIT((body(), std::exit(::testing::Test::HasFailure() ? 1 : 0)), ::testing::ExitedWithCode(0), "");
    }
    void PublishAndClear() {
        BlockPool device{4, {1}};
        const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kFull, .block_granularity = 4}};
        auto cache = MakeCoordinator(specs, 4, device, false, nullptr, nullptr, false, false, 0);
        const CacheKey key{.content_hash = ContentHash{hash_}};
        auto block = device.AcquireBlock(0);
        cache.CacheDeviceBlock(block, key, 0, CacheBoundaryKind::kEndpoint, 1);
        block.reset();
        EXPECT_NO_THROW(EXPECT_TRUE(cache.ClearDeviceCache()));
        EXPECT_FALSE(CacheCoordinatorTestAccess::DeviceLocationOf(cache, 0, key));
        EXPECT_EQ(cache.Evictor().DeviceEvictions().flush.blocks, 1u);
    }
    SchedulerConfig MakeConfig() override {
        auto config = SchedulerTestSuite::MakeConfig();
        config.disable_l2_cache = true;
        return config;
    }
    void SelectRequest(const char* boundary) {
        unsetenv(kVariable);
        setenv("TOKENSPEED_CACHE_EVICTION_REQUEST_TAG", "watch", 1);
        setenv("TOKENSPEED_CACHE_EVICTION_PREFIX_TOKENS", boundary, 1);
        SchedulerTestSuite::SetUp();
    }
    void RunRequest(const RequestSpec& spec) {
        Submit(spec);
        const auto plan = PlanOnce();
        EXPECT_TRUE(std::any_of(plan.Operations().begin(), plan.Operations().end(),
                                [](const auto& op) { return std::holds_alternative<ForwardBatch>(op); }));
        SendForwardDone(spec.request_id, {42});
        SendFinish(spec.request_id);
        PlanOnce();
        EXPECT_TRUE(scheduler_->ClearCache());
        EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    }
    static constexpr auto kVariable = "TOKENSPEED_CACHE_EVICTION_HASH";
    static constexpr std::array kVariables{kVariable, "TOKENSPEED_CACHE_EVICTION_REQUEST_TAG",
                                           "TOKENSPEED_CACHE_EVICTION_PREFIX_TOKENS",
                                           "TOKENSPEED_PREFIX_CACHE_DIAGNOSTICS"};
    const std::string hash_ = std::string(64, 'a');
    std::array<std::optional<std::string>, kVariables.size()> previous_environment_;
    std::shared_ptr<spdlog::logger> previous_logger_;
    std::ostringstream output_;
};

TEST_F(CacheEvictionDiagnosticTest, HostLruIdentifiesLastStateCopiesAndFlushKeepsItsOwnReason) {
    InFreshProcess([&] {
        BlockPool device{8, {1, 1, 1}};
        BlockPool host{8, {1, 1, 1}};
        const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kFull, .block_granularity = 4},
                               CacheGroupSpec{.attn_kind = AttnKind::kState},
                               CacheGroupSpec{.attn_kind = AttnKind::kState}};
        auto cache = MakeCoordinator(specs, 4, device, false, &host, nullptr, false, true, 0);
        for (std::uint32_t group = 0; group < specs.size(); ++group) {
            const CacheKey key{.group_id = group, .content_hash = ContentHash{hash_}};
            auto block = device.AcquireBlock(group);
            cache.CacheDeviceBlock(block, key, 0, CacheBoundaryKind::kEndpoint, 1);
            auto copy = cache.Host().AcquireHostBlock(group, 0);
            cache.Host().CacheHostBlock(copy, key,
                                        PrefixCacheIndex::CachedBlockMetadata{.eviction = {.last_access_epoch = 29}});
        }
        const CacheKey state{.group_id = 1, .content_hash = ContentHash{hash_}};
        const auto location = CacheCoordinatorTestAccess::DeviceLocationOf(cache, 1, state);
        ASSERT_TRUE(location);
        EXPECT_TRUE(
            cache.Evictor().EvictCachedBlock(1, *location, &CacheEvictor::DeviceEvictionStats::admission, nullptr));
        EXPECT_TRUE(cache.Host().ContainsHostCachedBlock(state));
        EXPECT_TRUE(Has({"tier=device", "reason=admission", "group_id=1", "last_access_epoch=1", "device_present=false",
                         "host_present=true"}));
        EXPECT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(cache, state));
        for (std::uint32_t group : {1u, 2u}) {
            EXPECT_FALSE(
                cache.Host().ContainsHostCachedBlock(CacheKey{.group_id = group, .content_hash = ContentHash{hash_}}));
            EXPECT_TRUE(Has({"tier=host", "reason=lru", "group_id=" + std::to_string(group), "hash=" + hash_,
                             "demoted=false", "last_access_epoch=29", "device_present=false", "host_present=false",
                             "eviction_writeback_pending=false"}));
        }
        // The history page survived the State cascade. A later explicit clear
        // must report flush, not the earlier checkpoint's LRU reason.
        EXPECT_TRUE(cache.Host().ContainsHostCachedBlock(CacheKey{.content_hash = ContentHash{hash_}}));
        EXPECT_TRUE(cache.ClearCache());
        EXPECT_TRUE(Has({"tier=host", "reason=flush", "group_id=0", "device_present=false", "host_present=false"}));
    });
}

TEST_F(CacheEvictionDiagnosticTest, DeviceEvictionDistinguishesPendingWritebackFromLostIndexedCopies) {
    InFreshProcess([&] {
        BlockPool device{8, {1}};
        BlockPool host{8, {1}};
        const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kFull, .block_granularity = 4}};
        auto cache = MakeCoordinator(specs, 4, device, false, &host, nullptr, false, true, 0);
        const CacheKey key{.content_hash = ContentHash{hash_}};
        auto block = device.AcquireBlock(0);
        cache.CacheDeviceBlock(block, key, 0, CacheBoundaryKind::kEndpoint, 1);
        block.reset();
        CacheEvictor::EvictedSegment segment;
        ASSERT_TRUE(CacheCoordinatorTestAccess::RecordDeviceEviction(cache, key, segment));
        ASSERT_EQ(segment.members.size(), 1u);
        EXPECT_FALSE(cache.Host().ContainsHostCachedBlock(key));
        EXPECT_TRUE(Has({"tier=device", "reason=admission", "device_present=false", "host_present=false",
                         "eviction_writeback_pending=true"}));
        segment.members[0].host_block = cache.Host().AcquireHostBlock(0, 0);
        EXPECT_EQ(cache.Host().SettleEvictedSegment(std::move(segment), true).size(), 1u);
        EXPECT_TRUE(cache.Host().ContainsHostCachedBlock(key));
    });
}

TEST_F(CacheEvictionDiagnosticTest, SupersededEvictionReportsTheReasonUsedByAccounting) {
    InFreshProcess([&] {
        BlockPool device{4, {1}};
        const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kFull, .block_granularity = 4}};
        auto cache = MakeCoordinator(specs, 4, device, false, nullptr, nullptr, false, false, 0);
        const CacheKey key{.content_hash = ContentHash{hash_}};
        auto block = device.AcquireBlock(0);
        cache.CacheDeviceBlock(block, key, 0, CacheBoundaryKind::kEndpoint, CacheStore::kSupersededAccessEpoch);
        block.reset();
        CacheEvictor::EvictedSegment segment;
        ASSERT_TRUE(CacheCoordinatorTestAccess::RecordDeviceEviction(cache, key, segment));
        EXPECT_TRUE(segment.members.empty());
        EXPECT_EQ(cache.Evictor().DeviceEvictions().request_reclaim.blocks, 1u);
        EXPECT_EQ(cache.Evictor().DeviceEvictions().admission.blocks, 0u);
        EXPECT_TRUE(Has({"reason=request_reclaim", "demoted=false", "last_access_epoch=0", "device_present=false",
                         "eviction_writeback_pending=false"}));
    });
}

TEST_F(CacheEvictionDiagnosticTest, DisabledDiagnosticStillClearsTheCache) {
    InFreshProcess([&] {
        unsetenv(kVariable);
        PublishAndClear();
        EXPECT_TRUE(output_.str().empty());
    });
}

TEST_F(CacheEvictionDiagnosticTest, UnselectedEvictionStillClearsTheCache) {
    InFreshProcess([&] {
        setenv(kVariable, std::string(64, 'b').c_str(), 1);
        PublishAndClear();
        EXPECT_TRUE(output_.str().empty());
        // The first use fixed this process's selection.
        setenv(kVariable, hash_.c_str(), 1);
        PublishAndClear();
        EXPECT_TRUE(output_.str().empty());
    });
}

TEST_F(CacheEvictionDiagnosticTest, InvalidSelectionReportsOnceAndDoesNotInterruptEviction) {
    InFreshProcess([&] {
        setenv(kVariable, "invalid", 1);
        PublishAndClear();
        PublishAndClear();
        const auto text = output_.str();
        ASSERT_NE(text.find("TOKENSPEED_CACHE_EVICTION_HASH requires"), std::string::npos);
        EXPECT_EQ(text.find("[CacheEvictionDiagnostic]"), text.rfind("[CacheEvictionDiagnostic]"));
        EXPECT_EQ(text.find("tier="), std::string::npos);
    });
}

TEST_F(CacheEvictionDiagnosticTest, FailedOutputAndErrorHandlerCannotInterruptEviction) {
    InFreshProcess([&] {
        // A real stream failure, with the logger configured to propagate it.
        output_.setstate(std::ios_base::badbit);
        EXPECT_THROW(output_.exceptions(std::ios_base::badbit), std::ios_base::failure);
        spdlog::default_logger()->set_error_handler(
            [](const std::string& message) { throw std::runtime_error(message); });
        PublishAndClear();
    });
}

TEST_F(CacheEvictionDiagnosticTest, DemotionMetadataSurvivesDeviceAndHostRemoval) {
    InFreshProcess([&] {
        BlockPool device{8, {1}};
        BlockPool host{8, {1}};
        const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kFull, .block_granularity = 4}};
        auto cache = MakeCoordinator(specs, 4, device, false, &host, nullptr, false, true, 0);
        const CacheKey key{.content_hash = ContentHash{hash_}};
        auto block = device.AcquireBlock(0);
        cache.CacheDeviceBlock(block, key, 0, CacheBoundaryKind::kEndpoint, 17);
        auto copy = cache.Host().AcquireHostBlock(0, 0);
        cache.Host().CacheHostBlock(copy, key, std::nullopt);
        block.reset();
        copy.reset();
        auto& index = cache.Store().Groups()[0].Index();
        ASSERT_TRUE(index.Demote(device, key, 0, 0));
        ASSERT_TRUE(index.Demote(host, key, 0, 0));
        EXPECT_TRUE(cache.ClearCache());
        EXPECT_TRUE(Has({"tier=device", "demoted=true", "last_access_epoch=0", "host_present=true"}));
        EXPECT_TRUE(Has({"tier=host", "reason=flush", "demoted=true", "last_access_epoch=0"}));
        EXPECT_FALSE(index.Contains(device, key));
        EXPECT_FALSE(index.Contains(host, key));
    });
}

TEST_F(CacheEvictionDiagnosticTest, RequestTagBindsCurrentReplayAtTheRequestedBoundary) {
    // The cache-busting first token changes each replay's content hash; the
    // selector stays the same and no old hash is supplied to either process.
    for (const auto first_token : {101, 202}) {
        InFreshProcess([&] {
            SelectRequest("4");
            RunRequest(RequestSpec{.request_id = "other-watchful", .tokens = {1, 2, 3, 4, 5}});
            RunRequest(RequestSpec{.request_id = "short-watch", .tokens = {1}});
            EXPECT_FALSE(Has({"event=bind"}));
            const RequestSpec spec{.request_id = "request-watch-gateway", .tokens = {first_token, 2, 3, 4, 5}};
            Request expected(spec, 2, Role::kFused);
            const auto hashes = expected.PromptPrefixHashes();
            RunRequest(spec);
            EXPECT_TRUE(Has(
                {"event=bind", "request=request-watch-gateway", "tag=watch", "prefix_tokens=4", "hash=" + hashes[1]}));
            EXPECT_TRUE(Has({"event=evict", "reason=flush", "hash=" + hashes[1], "demoted=false"}));
            EXPECT_FALSE(Has({"event=evict", "hash=" + hashes[0]}));
            const auto first_binding = output_.str().find("event=bind");
            RunRequest(RequestSpec{.request_id = "again-watch", .tokens = spec.tokens});
            EXPECT_EQ(first_binding, output_.str().rfind("event=bind"));
            EXPECT_TRUE(Has({"event=evict", "last_access_epoch="}));
        });
    }
}

TEST_F(CacheEvictionDiagnosticTest, AmbiguousTagDisablesObservationWithoutInterruptingRequests) {
    InFreshProcess([&] {
        SelectRequest("4");
        RunRequest(RequestSpec{.request_id = "first-watch", .tokens = {1, 2, 3, 4, 5}});
        ASSERT_TRUE(Has({"event=bind"}));
        output_.str("");
        RunRequest(RequestSpec{.request_id = "different-watch", .tokens = {9, 2, 3, 4, 5}});
        EXPECT_TRUE(Has({"different prefix content", "diagnostics disabled"}));
        EXPECT_FALSE(Has({"event=evict"}));
        RunRequest(RequestSpec{.request_id = "original-watch", .tokens = {1, 2, 3, 4, 5}});
        EXPECT_FALSE(Has({"event=evict"}));
    });
}

TEST_F(CacheEvictionDiagnosticTest, InvalidTokenBoundaryNeverInterruptsRequests) {
    for (const char* boundary : {"3", "0", "-1", "2147483648", "4extra"}) {
        InFreshProcess([&] {
            SelectRequest(boundary);
            RunRequest(RequestSpec{.request_id = "request-watch", .tokens = {1, 2, 3, 4, 5}});
            EXPECT_FALSE(Has({"event=bind"}));
            EXPECT_FALSE(Has({"event=evict"}));
            EXPECT_TRUE(Has({"prefix tokens"}));
        });
    }
}

TEST_F(CacheEvictionDiagnosticTest, ConflictingSelectorsDoNotChooseOneSilently) {
    InFreshProcess([&] {
        setenv("TOKENSPEED_CACHE_EVICTION_REQUEST_TAG", "watch", 1);
        setenv("TOKENSPEED_CACHE_EVICTION_PREFIX_TOKENS", "4", 1);
        PublishAndClear();
        EXPECT_TRUE(Has({"cannot be combined"}));
        EXPECT_FALSE(Has({"event=evict"}));
    });
}

TEST_F(CacheEvictionDiagnosticTest, FailedBindingOutputCannotInterruptRequests) {
    InFreshProcess([&] {
        SelectRequest("4");
        output_.setstate(std::ios_base::badbit);
        EXPECT_THROW(output_.exceptions(std::ios_base::badbit), std::ios_base::failure);
        spdlog::default_logger()->set_error_handler(
            [](const std::string& message) { throw std::runtime_error(message); });
        RunRequest(RequestSpec{.request_id = "request-watch", .tokens = {1, 2, 3, 4, 5}});
    });
}

TEST_F(CacheEvictionDiagnosticTest, RejectedBatchDoesNotBindOrDisableAValidSelection) {
    InFreshProcess([&] {
        SelectRequest("4");
        EXPECT_THROW(Submit(std::vector<RequestSpec>{{.request_id = "rejected-watch", .tokens = {9, 2, 3, 4}},
                                                     {.request_id = "empty", .tokens = {}}}),
                     std::invalid_argument);
        EXPECT_FALSE(Has({"event=bind"}));
        const RequestSpec spec{.request_id = "accepted-watch", .tokens = {101, 2, 3, 4}};
        Request expected(spec, PrefixGranularity(), Role::kFused);
        const auto hash = expected.PromptPrefixHashes()[1];
        RunRequest(spec);
        EXPECT_TRUE(Has({"event=bind", "request=accepted-watch", "hash=" + hash}));
        EXPECT_TRUE(Has({"event=evict", "reason=flush", "hash=" + hash}));
        EXPECT_FALSE(Has({"diagnostics disabled"}));
    });
}

class CacheEvictionQueuedRequestTest : public CacheEvictionDiagnosticTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = CacheEvictionDiagnosticTest::MakeConfig();
        config.device_allocator.total_pages = 7;
        config.cache_groups[0].total_pages = 7;
        config.max_batch_size = 1;
        config.max_resident_requests = 1;
        config.enable_short_prefill_first = false;
        return config;
    }
};

TEST_F(CacheEvictionQueuedRequestTest, RecordsEvictionWhileTheSelectedRequestWaitsForASlot) {
    InFreshProcess([&] {
        SelectRequest("4");
        const RequestSpec selected{.request_id = "queued-watch", .tokens = {101, 2, 3, 4}};
        Request expected(selected, PrefixGranularity(), Role::kFused);
        const auto hash = expected.PromptPrefixHashes()[1];
        // Retain the completed prefixes so the blocker must evict the older
        // target while the tagged request is still waiting for the only slot.
        for (const auto& spec : std::vector<RequestSpec>{{.request_id = "seed", .tokens = selected.tokens},
                                                         {.request_id = "filler", .tokens = {21, 22, 23, 24}}}) {
            Submit(spec);
            const auto plan = PlanOnce();
            ASSERT_NE(FindForwardBatch(plan), nullptr);
            ASSERT_EQ(FindForwardBatch(plan)->request_ids, std::vector<std::string>{spec.request_id});
            SendForwardDone(spec.request_id, {42});
            SendFinish(spec.request_id);
            PlanOnce();
        }
        Submit(std::vector<RequestSpec>{{.request_id = "blocker", .tokens = {11, 12, 13, 14}}, selected});
        EXPECT_TRUE(Has({"event=bind", "request=queued-watch", "hash=" + hash}));
        const auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        EXPECT_EQ(FindForwardBatch(plan)->request_ids, std::vector<std::string>{"blocker"});
        EXPECT_EQ(scheduler_->WaitingSize(), 1u);
        EXPECT_TRUE(Has({"event=evict", "tier=device", "reason=admission", "hash=" + hash, "device_present=false",
                         "host_present=false"}));
        SendForwardDone("blocker", {42});
        SendFinish("blocker");
        const auto selected_plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(selected_plan), nullptr);
        EXPECT_EQ(FindForwardBatch(selected_plan)->request_ids, std::vector<std::string>{selected.request_id});
        SendForwardDone(selected.request_id, {42});
        SendFinish(selected.request_id);
        PlanOnce();
        EXPECT_TRUE(scheduler_->ClearCache());
        EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    });
}

class CacheEvictionPromptEndTest : public CacheEvictionDiagnosticTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = CacheEvictionDiagnosticTest::MakeConfig();
        for (int i = 0; i < 3; ++i) {
            config.cache_groups.push_back(CacheGroupConfig{.group_id = "state_" + std::to_string(i),
                                                           .block_granularity = 0,
                                                           .total_pages = config.device_allocator.total_pages,
                                                           .attn_kind = AttnKind::kState});
        }
        return config;
    }
};

TEST_F(CacheEvictionPromptEndTest, BindsTheCompletePromptPageAndLogsAllPublishedGroups) {
    InFreshProcess([&] {
        SelectRequest("4");
        const RequestSpec spec{.request_id = "prompt-end-watch", .tokens = {101, 2, 3, 4}};
        Request expected(spec, 2, Role::kFused);
        const auto hashes = expected.PromptPrefixHashes();
        RunRequest(spec);
        EXPECT_TRUE(Has({"event=bind", "request=prompt-end-watch", "prefix_tokens=4", "hash=" + hashes[1]}));
        // Recomputing the last input token excludes this page from matching,
        // but finish publishes both its Full page and all three State entries.
        for (int group = 0; group < 4; ++group) {
            EXPECT_TRUE(Has({"event=evict", "tier=device", "reason=flush", "hash=" + hashes[1],
                             "group_id=" + std::to_string(group), "device_present=false"}));
        }
        EXPECT_FALSE(Has({"event=evict", "hash=" + hashes[0]}));
    });
}

}  // namespace
}  // namespace tokenspeed::test
