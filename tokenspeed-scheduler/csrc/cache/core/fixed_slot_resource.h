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
#include <bit>
#include <cstddef>
#include <limits>
#include <list>
#include <memory>
#include <memory_resource>
#include <new>
#include <vector>

#include "utils.h"

#if defined(__SANITIZE_ADDRESS__)
#define TOKENSPEED_SCHEDULER_ADDRESS_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define TOKENSPEED_SCHEDULER_ADDRESS_SANITIZER 1
#endif
#endif
#if defined(TOKENSPEED_SCHEDULER_ADDRESS_SANITIZER)
#include <sanitizer/asan_interface.h>
#endif

namespace tokenspeed {

// The size and alignment of the objects a FixedSlotResource pools.
struct SlotShape {
    std::size_t bytes;
    std::size_t alignment;

    bool operator==(const SlotShape&) const noexcept = default;

    // A `T` allocated directly.
    template <typename T>
    static constexpr SlotShape Of() noexcept {
        return SlotShape{sizeof(T), alignof(T)};
    }
    // The node a std::pmr::list<T> allocates for each element.
    template <typename T>
    static SlotShape ListNodeOf();
};

template <typename T>
SlotShape SlotShape::ListNodeOf() {
    // The standard does not name a list's node type, so ask a list for one.
    // A node holds its element in storage of the element's size and
    // alignment, so a list of plain bytes of that shape, which needs no T,
    // allocates nodes of the same shape.
    struct alignas(T) Element {
        std::byte bytes[sizeof(T)];
    };
    class Recorder final : public std::pmr::memory_resource {
    public:
        SlotShape node{0, 0};

    private:
        void* do_allocate(std::size_t bytes, std::size_t alignment) override {
            node = SlotShape{bytes, alignment};
            return std::pmr::new_delete_resource()->allocate(bytes, alignment);
        }
        void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) noexcept override {
            std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
        }
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
    };
    Recorder recorder;
    std::pmr::list<Element>{&recorder}.emplace_back();
    return recorder.node;
}

// Storage for one fixed node type on the scheduler control thread. Unlike a
// general pool resource, freeing a node never searches the allocated chunks:
// the freed slot itself links the free list. The owner names the pooled slot
// shape at construction; requests of any other shape, which a container may
// make besides its nodes, go to the upstream resource. Chunks of
// slots_per_chunk slots come from upstream on demand and, with every freed
// slot, are retained until this resource is destroyed: about
// ceil(high_water_slots / slots_per_chunk) * slots_per_chunk * aligned slot
// size bytes, plus one handle per chunk. The owning container destroys its
// objects before this resource. Under AddressSanitizer every free slot is
// poisoned, its free-list link included, so reading a freed node through a
// stale pointer is reported even though the memory stays allocated.
class FixedSlotResource final : public std::pmr::memory_resource {
public:
    FixedSlotResource(SlotShape slot, std::size_t slots_per_chunk, std::pmr::memory_resource* upstream)
        : slot_{slot}, slots_per_chunk_{slots_per_chunk}, upstream_{upstream} {
        FatalCheck(slot.bytes > 0 && std::has_single_bit(slot.alignment),
                   "a slot has a size and a power-of-two alignment");
        FatalCheck(slots_per_chunk > 0, "a slot chunk must hold at least one object");
        FatalCheck(upstream != nullptr, "slot chunks need an upstream resource");
        storage_alignment_ = std::max(slot.alignment, alignof(FreeSlot));
        const auto size = std::max(slot.bytes, sizeof(FreeSlot));
        constexpr auto kMaxBytes = std::numeric_limits<std::size_t>::max();
        FatalCheck(size <= kMaxBytes - (storage_alignment_ - 1), "a slot's size overflows");
        stride_ = (size + storage_alignment_ - 1) / storage_alignment_ * storage_alignment_;
        FatalCheck(stride_ <= kMaxBytes / slots_per_chunk, "a slot chunk's size overflows");
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
        std::pmr::memory_resource* upstream;
        std::size_t bytes;
        std::size_t alignment;
        void operator()(std::byte* pointer) const noexcept {
            unpoison(pointer, bytes);
            upstream->deallocate(pointer, bytes, alignment);
        }
    };
    static void poison([[maybe_unused]] void* memory, [[maybe_unused]] std::size_t bytes) noexcept {
#if defined(TOKENSPEED_SCHEDULER_ADDRESS_SANITIZER)
        __asan_poison_memory_region(memory, bytes);
#endif
    }
    static void unpoison([[maybe_unused]] void* memory, [[maybe_unused]] std::size_t bytes) noexcept {
#if defined(TOKENSPEED_SCHEDULER_ADDRESS_SANITIZER)
        __asan_unpoison_memory_region(memory, bytes);
#endif
    }
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes != slot_.bytes || alignment != slot_.alignment) {
            return upstream_->allocate(bytes, alignment);
        }
        if (free_ == nullptr) {
            const auto chunk_bytes = stride_ * slots_per_chunk_;
            auto* memory = static_cast<std::byte*>(upstream_->allocate(chunk_bytes, storage_alignment_));
            std::unique_ptr<std::byte, ChunkDeleter> chunk(memory, {upstream_, chunk_bytes, storage_alignment_});
            // Prepare the owning handle before publishing any free slots.
            chunks_.push_back(std::move(chunk));
            for (std::size_t i = 0; i < slots_per_chunk_; ++i) {
                free_ = new (memory + i * stride_) FreeSlot{free_};
                poison(free_, stride_);
            }
        }
        auto* slot = free_;
        unpoison(slot, stride_);
        free_ = slot->next;
        return slot;
    }
    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) noexcept override {
        // A deallocation names the shape it was allocated with.
        if (bytes != slot_.bytes || alignment != slot_.alignment) {
            upstream_->deallocate(pointer, bytes, alignment);
            return;
        }
        free_ = new (pointer) FreeSlot{free_};
        poison(pointer, stride_);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }

    SlotShape slot_;
    std::size_t slots_per_chunk_;
    std::pmr::memory_resource* upstream_;
    std::size_t storage_alignment_{0};
    std::size_t stride_{0};
    FreeSlot* free_{nullptr};
    std::vector<std::unique_ptr<std::byte, ChunkDeleter>> chunks_;
};

}  // namespace tokenspeed
