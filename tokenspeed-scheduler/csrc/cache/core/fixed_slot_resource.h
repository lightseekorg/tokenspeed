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

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <vector>

#include "utils.h"

namespace tokenspeed {

// Storage for one fixed node type on the scheduler control thread. Unlike a
// general pool resource, freeing a node never searches the allocated chunks:
// the freed slot itself links the free list. The first request fixes the slot
// size and alignment; requests of any other shape, which a container may make
// besides its nodes, go to the upstream new/delete resource. Chunks of
// slots_per_chunk slots are allocated on demand and, with every freed slot,
// retained until this resource is destroyed: about
// ceil(high_water_slots / slots_per_chunk) * slots_per_chunk * aligned slot
// size bytes, plus one handle per chunk. The owning container destroys its
// objects before this resource.
class FixedSlotResource final : public std::pmr::memory_resource {
public:
    explicit FixedSlotResource(std::size_t slots_per_chunk) : slots_per_chunk_{slots_per_chunk} {
        FatalCheck(slots_per_chunk > 0, "a slot chunk must hold at least one object");
    }
    // Containers keep a pointer to their resource, and their nodes live in
    // its chunks: a copy or a move would strand them.
    FixedSlotResource(const FixedSlotResource&) = delete;
    FixedSlotResource& operator=(const FixedSlotResource&) = delete;
    FixedSlotResource(FixedSlotResource&&) = delete;
    FixedSlotResource& operator=(FixedSlotResource&&) = delete;

private:
    struct FreeSlot {
        FreeSlot* next;
    };
    struct ChunkDeleter {
        std::size_t bytes;
        std::size_t alignment;
        void operator()(std::byte* pointer) const noexcept {
            std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
        }
    };
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (stride_ == 0) {
            bytes_ = bytes;
            alignment_ = alignment;
            storage_alignment_ = std::max(alignment, alignof(FreeSlot));
            const auto size = std::max(bytes, sizeof(FreeSlot));
            if (size > std::numeric_limits<std::size_t>::max() - (storage_alignment_ - 1)) throw std::bad_alloc{};
            stride_ = (size + storage_alignment_ - 1) / storage_alignment_ * storage_alignment_;
        }
        if (bytes != bytes_ || alignment != alignment_) {
            return std::pmr::new_delete_resource()->allocate(bytes, alignment);
        }
        if (free_ == nullptr) {
            if (stride_ > std::numeric_limits<std::size_t>::max() / slots_per_chunk_) throw std::bad_alloc{};
            const auto chunk_bytes = stride_ * slots_per_chunk_;
            auto* memory =
                static_cast<std::byte*>(std::pmr::new_delete_resource()->allocate(chunk_bytes, storage_alignment_));
            std::unique_ptr<std::byte, ChunkDeleter> chunk(memory, {chunk_bytes, storage_alignment_});
            // Prepare the owning handle before publishing any free slots.
            chunks_.push_back(std::move(chunk));
            for (std::size_t i = 0; i < slots_per_chunk_; ++i) {
                free_ = new (memory + i * stride_) FreeSlot{free_};
            }
        }
        auto* slot = free_;
        free_ = slot->next;
        return slot;
    }
    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) noexcept override {
        // A deallocation names the shape it was allocated with.
        if (bytes != bytes_ || alignment != alignment_) {
            std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
            return;
        }
        free_ = new (pointer) FreeSlot{free_};
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }

    std::size_t slots_per_chunk_;
    std::size_t bytes_{0};
    std::size_t alignment_{0};
    std::size_t storage_alignment_{0};
    std::size_t stride_{0};
    FreeSlot* free_{nullptr};
    std::vector<std::unique_ptr<std::byte, ChunkDeleter>> chunks_;
};

}  // namespace tokenspeed
