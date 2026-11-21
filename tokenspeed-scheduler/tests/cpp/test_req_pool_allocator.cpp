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

#include <utility>

#include "resource/allocator/req_pool_allocator.h"
#include "cache_event_test_sink.h"

namespace tokenspeed::test {

TEST(ReqPoolAllocatorTest, RAIIReleasesSlotOnDestruction) {
    ReqPoolAllocator allocator(1);
    {
        auto index = allocator.Allocate();
        EXPECT_TRUE(index.valid());
        EXPECT_EQ(allocator.AvailableSlots(), 0);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 1);
}

TEST(ReqPoolAllocatorTest, MoveAssignmentReleasesPreviouslyOwnedSlot) {
    ReqPoolAllocator allocator(2);
    {
        auto dst = allocator.Allocate();
        auto src = allocator.Allocate();
        ASSERT_EQ(allocator.AvailableSlots(), 0);

        dst = std::move(src);

        EXPECT_TRUE(dst.valid());
        EXPECT_EQ(src.slot_, -1);
        EXPECT_EQ(allocator.AvailableSlots(), 1);
    }
    EXPECT_EQ(allocator.AvailableSlots(), 2);
}

TEST(ReqPoolAllocatorTest, OwnershipMovesWithoutASecondAssignmentAndReleasesExactlyOnce) {
    RecordingCacheEventSink sink;
    ReqPoolAllocator allocator(1);
    allocator.SetEventSink(&sink);
    {
        auto original = allocator.Allocate();
        original.BindOwner("same-rid", 11);
        const auto generation = original.generation();
        auto moved = std::move(original);
        EXPECT_FALSE(original.valid());
        EXPECT_EQ(moved.generation(), generation);
        EXPECT_EQ(sink.slots.size(), 1u);
        EXPECT_THROW(moved.BindOwner("other", 12), std::logic_error);
    }
    ASSERT_EQ(sink.slots.size(), 2u);
    EXPECT_TRUE(sink.slots[0].assigned);
    EXPECT_FALSE(sink.slots[1].assigned);
    EXPECT_EQ(sink.slots[1].generation, 1u);
    EXPECT_EQ(sink.slots[1].admission_id, 11u);
    auto reused = allocator.Allocate();
    reused.BindOwner("same-rid", 22);
    ASSERT_EQ(sink.slots.size(), 3u);
    EXPECT_EQ(sink.slots[2].slot, sink.slots[0].slot);
    EXPECT_EQ(sink.slots[2].generation, 2u);
    EXPECT_EQ(sink.slots[2].previous_admission_id, 11u);
    EXPECT_EQ(sink.slots[2].admission_id, 22u);
}

TEST(ReqPoolAllocatorTest, MoveAssignmentReleasesTheDestinationOwnerBeforeTakingTheSource) {
    RecordingCacheEventSink sink;
    ReqPoolAllocator allocator(2);
    allocator.SetEventSink(&sink);
    {
        auto first = allocator.Allocate();
        first.BindOwner("a", 11);
        auto second = allocator.Allocate();
        second.BindOwner("b", 22);
        const auto second_slot = second.slot_;
        first = std::move(second);
        ASSERT_EQ(sink.slots.size(), 3u);
        EXPECT_FALSE(sink.slots.back().assigned);
        EXPECT_EQ(sink.slots.back().admission_id, 11u);
        EXPECT_EQ(first.slot_, second_slot);
    }
    ASSERT_EQ(sink.slots.size(), 4u);
    EXPECT_FALSE(sink.slots.back().assigned);
    EXPECT_EQ(sink.slots.back().admission_id, 22u);
    EXPECT_EQ(allocator.AvailableSlots(), 2);
}

TEST(ReqPoolAllocatorTest, ObservedOwnershipRequiresAnIdentityAndCannotBeEnabledAfterReuse) {
    RecordingCacheEventSink sink;
    ReqPoolAllocator allocator(1);
    allocator.SetEventSink(&sink);
    {
        auto slot = allocator.Allocate();
        EXPECT_THROW(slot.BindOwner("a", 0), std::invalid_argument);
        EXPECT_TRUE(sink.slots.empty());
        slot.BindOwner("a", 1);
    }
    EXPECT_THROW(allocator.SetEventSink(nullptr), std::logic_error);
}

TEST(ReqPoolAllocatorTest, ObservationOwnsTheRequestIdUntilTheLastHandleMoveReleasesIt) {
    RecordingCacheEventSink sink;
    ReqPoolAllocator allocator(1);
    allocator.SetEventSink(&sink);
    const std::string expected_id(256, 'a');
    {
        auto slot = allocator.Allocate();
        {
            std::string submitted_id = expected_id;
            slot.BindOwner(submitted_id, 11);
            submitted_id.assign(256, 'b');
        }
        auto moved = std::move(slot);
        slot = std::move(moved);
        EXPECT_EQ(slot.generation(), 1u);
        EXPECT_EQ(slot.previous_admission_id(), 0u);
        ASSERT_EQ(sink.slots.size(), 1u);
        EXPECT_EQ(sink.slots.front().request_id, expected_id);
    }
    ASSERT_EQ(sink.slots.size(), 2u);
    EXPECT_FALSE(sink.slots.back().assigned);
    EXPECT_EQ(sink.slots.back().request_id, expected_id);
    EXPECT_EQ(sink.slots.back().admission_id, 11u);

    // An allocation that never binds must not release the previous owner
    // again, even though this physical slot keeps its generation history.
    {
        auto unbound = allocator.Allocate();
        EXPECT_EQ(unbound.generation(), 2u);
        EXPECT_EQ(unbound.previous_admission_id(), 11u);
    }
    EXPECT_EQ(sink.slots.size(), 2u);
    {
        auto successor = allocator.Allocate();
        successor.BindOwner("next", 22);
        EXPECT_EQ(successor.generation(), 3u);
        EXPECT_EQ(successor.previous_admission_id(), 11u);
    }
    ASSERT_EQ(sink.slots.size(), 4u);
    EXPECT_EQ(sink.slots.back().request_id, "next");
    EXPECT_EQ(sink.slots.back().generation, 3u);
    EXPECT_EQ(sink.slots.back().previous_admission_id, 11u);
}

TEST(ReqPoolAllocatorTest, MovingBetweenAllocatorsPreservesEachOwnersObservation) {
    RecordingCacheEventSink first_sink;
    RecordingCacheEventSink second_sink;
    ReqPoolAllocator first_allocator(1);
    ReqPoolAllocator second_allocator(1);
    first_allocator.SetEventSink(&first_sink);
    second_allocator.SetEventSink(&second_sink);
    {
        auto first = first_allocator.Allocate();
        first.BindOwner("first", 11);
        auto second = second_allocator.Allocate();
        second.BindOwner("second", 22);
        first = std::move(second);
        EXPECT_EQ(first_allocator.AvailableSlots(), 1);
        EXPECT_EQ(second_allocator.AvailableSlots(), 0);
        ASSERT_EQ(first_sink.slots.size(), 2u);
        EXPECT_FALSE(first_sink.slots.back().assigned);
        EXPECT_EQ(first_sink.slots.back().request_id, "first");
        EXPECT_EQ(second_sink.slots.size(), 1u);
    }
    ASSERT_EQ(second_sink.slots.size(), 2u);
    EXPECT_FALSE(second_sink.slots.back().assigned);
    EXPECT_EQ(second_sink.slots.back().request_id, "second");
    EXPECT_EQ(second_sink.slots.back().admission_id, 22u);
    EXPECT_EQ(second_allocator.AvailableSlots(), 1);
}

TEST(ReqPoolAllocatorTest, UnobservedOwnershipStillKeepsGenerationAndBindingSemantics) {
    ReqPoolAllocator allocator(1);
    {
        auto slot = allocator.Allocate();
        slot.BindOwner("first", 11);
        auto moved = std::move(slot);
        EXPECT_EQ(moved.generation(), 1u);
        EXPECT_THROW(moved.BindOwner("second", 22), std::logic_error);
        EXPECT_FALSE(slot.valid());
        EXPECT_EQ(slot.generation(), 0u);
    }
    auto reused = allocator.Allocate();
    EXPECT_EQ(reused.generation(), 2u);
    EXPECT_EQ(reused.previous_admission_id(), 11u);
    reused.BindOwner("second", 22);
}

}  // namespace tokenspeed::test
