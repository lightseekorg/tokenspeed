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
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "fsm/forward_states.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {

// FSM states constructed mid-transition take ownership of a HostNodeRef from
// the calling event. After the caller's local goes out of scope the state must
// remain the sole owner so eviction cannot reclaim host pages while a LoadBack
// DMA is still in flight. Each test below builds a state, drops the caller's
// local, and asserts the host lock is still held.

class HostNodeRefLifetimeTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 2;

    HostNodeRefLifetimeTest()
        : device_alloc_{kPageSize, /*total_pages=*/3},
          host_alloc_{kPageSize, /*total_pages=*/16},
          cache_{&device_alloc_, &host_alloc_, /*enable_l3_storage=*/false} {}

    // Insert one Device + Host page for tokens [1,2] and return the host node.
    TreeNode* MakeHostNode() {
        token_vec_t tokens = {1, 2};
        std::vector<std::span<const std::int32_t>> token_pages = {
            std::span<const std::int32_t>{tokens.data(), tokens.size()},
        };

        cache_.Insert<ResourceType::Device>(token_pages, /*prefix_pages=*/{}, device_alloc_.Allocate(1));
        cache_.Insert<ResourceType::Host>(token_pages, /*prefix_pages=*/{}, host_alloc_.Allocate(1));

        MatchResult match = cache_.Match(tokens);
        EXPECT_EQ(match.host.DepthInPage(), 1);
        return match.host.last_node;
    }

    static std::unique_ptr<HostNodeRef> MakeHostRef(TreeNode* node) {
        auto ref = std::make_unique<HostNodeRef>(node);
        EXPECT_GE(node->Host().RefCount(), 1);
        return ref;
    }

    PageAllocator device_alloc_;
    PageAllocator host_alloc_;
    KVPrefixCache cache_;
};

// Draining is the canonical state that already stores host_node_ref; this test
// pins the convention for the other states to follow.
TEST_F(HostNodeRefLifetimeTest, Draining_HoldsHostLockAfterCallerExits) {
    TreeNode* node = MakeHostNode();
    auto host_ref = MakeHostRef(node);
    const std::int32_t lock_before = node->Host().RefCount();

    fsm::Draining drain{
        /*pages_to_transfer=*/{},
        /*device_node_ref=*/std::unique_ptr<DeviceNodeRef>{},
        /*host_node_ref=*/std::move(host_ref),
    };
    host_ref.reset();

    EXPECT_EQ(node->Host().RefCount(), lock_before);
}

TEST_F(HostNodeRefLifetimeTest, Prefilling_HoldsHostLockAfterCallerExits) {
    TreeNode* node = MakeHostNode();
    auto host_ref = MakeHostRef(node);
    const std::int32_t lock_before = node->Host().RefCount();

    fsm::Prefilling prefilling{
        /*token_container=*/nullptr,
        /*page_size=*/kPageSize,
        /*host_node_ref=*/std::move(host_ref),
        /*device_node_ref=*/std::unique_ptr<DeviceNodeRef>{},
        /*local_kv_allocator=*/std::unique_ptr<LocalKVAllocator>{},
        /*req_pool_index=*/std::unique_ptr<ReqPoolIndex>{},
        /*window=*/TokenContainer::Window{},
    };
    host_ref.reset();

    EXPECT_EQ(node->Host().RefCount(), lock_before);
}

TEST_F(HostNodeRefLifetimeTest, PrefillDone_HoldsHostLockAfterCallerExits) {
    TreeNode* node = MakeHostNode();
    auto host_ref = MakeHostRef(node);
    const std::int32_t lock_before = node->Host().RefCount();

    fsm::PrefillDone prefill_done{
        /*token_container=*/nullptr,
        /*page_size=*/kPageSize,
        /*host_node_ref=*/std::move(host_ref),
        /*device_node_ref=*/std::unique_ptr<DeviceNodeRef>{},
        /*local_kv_allocator=*/std::unique_ptr<LocalKVAllocator>{},
        /*req_pool_index=*/std::unique_ptr<ReqPoolIndex>{},
        /*window=*/TokenContainer::Window{},
        /*reserve_num_tokens_in_next_schedule_event=*/0,
    };
    host_ref.reset();

    EXPECT_EQ(node->Host().RefCount(), lock_before);
}

TEST_F(HostNodeRefLifetimeTest, Decoding_HoldsHostLockAfterCallerExits) {
    TreeNode* node = MakeHostNode();
    auto host_ref = MakeHostRef(node);
    const std::int32_t lock_before = node->Host().RefCount();

    fsm::Decoding decoding{
        /*token_container=*/nullptr,
        /*page_size=*/kPageSize,
        /*host_node_ref=*/std::move(host_ref),
        /*node_ref=*/std::unique_ptr<DeviceNodeRef>{},
        /*local_kv_allocator=*/std::unique_ptr<LocalKVAllocator>{},
        /*req_pool_index=*/std::unique_ptr<ReqPoolIndex>{},
        /*reserve_num_tokens_in_next_schedule_event=*/0,
    };
    host_ref.reset();

    EXPECT_EQ(node->Host().RefCount(), lock_before);
}

}  // namespace tokenspeed::test
