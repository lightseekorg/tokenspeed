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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// The Device LCM parents that request block tables hold, counted as the
// tables change so the active-page gauge reads it without walking them. Each
// slot of a table bound to it (BlockTable::CountActiveParentsIn) owns a
// ParentRef to its block's parent; a parent is active while any ParentRef to
// it lives, however many tables or slots share it. The coordinator owns it;
// the BlockPool never sees it.
class ActiveParents {
public:
    explicit ActiveParents(const BlockPool& pool)
        : pool_{&pool}, refs_(static_cast<std::size_t>(pool.NumLcmBlocks()) + 1, 0) {}
    ActiveParents(const ActiveParents&) = delete;
    ActiveParents& operator=(const ActiveParents&) = delete;

    // Distinct parents with at least one live ParentRef.
    std::int32_t Count() const noexcept { return num_active_; }

private:
    friend class ParentRef;

    std::uint32_t& refs(std::int32_t lcm_block_id) noexcept {
        FatalCheck(lcm_block_id > 0 && static_cast<std::size_t>(lcm_block_id) < refs_.size(),
                   "LCM block id is outside the counted pool");
        return refs_[static_cast<std::size_t>(lcm_block_id)];
    }

    // The pool whose parents are counted; a bound table holds its blocks only.
    const BlockPool* pool_;
    // refs_[id]: live ParentRefs to LCM parent `id`.
    std::vector<std::uint32_t> refs_;
    std::int32_t num_active_{0};
};

// One table slot's share in its block's LCM parent: the parent counts as
// active in `parents` while this lives. Empty for a null slot or an unbound
// table.
class ParentRef {
public:
    ParentRef() noexcept = default;
    ParentRef(ActiveParents* parents, const CacheBlockRef& block) noexcept {
        if (parents != nullptr && block) {
            FatalCheck(&block->OwnerPool() == parents->pool_, "a bound table holds a block of another pool");
            parents_ = parents;
            lcm_block_id_ = block->Location().lcm_block_id;
            retain();
        }
    }
    ParentRef(const ParentRef& other) noexcept : parents_{other.parents_}, lcm_block_id_{other.lcm_block_id_} {
        retain();
    }
    ParentRef(ParentRef&& other) noexcept
        : parents_{std::exchange(other.parents_, nullptr)}, lcm_block_id_{std::exchange(other.lcm_block_id_, 0)} {}
    ParentRef& operator=(ParentRef other) noexcept {
        std::swap(parents_, other.parents_);
        std::swap(lcm_block_id_, other.lcm_block_id_);
        return *this;
    }
    ~ParentRef() {
        if (parents_ == nullptr) {
            return;
        }
        std::uint32_t& count = parents_->refs(lcm_block_id_);
        FatalCheck(count > 0, "a ParentRef outlived its parent's count");
        if (--count == 0) {
            --parents_->num_active_;
        }
    }

private:
    void retain() noexcept {
        if (parents_ != nullptr && parents_->refs(lcm_block_id_)++ == 0) {
            ++parents_->num_active_;
        }
    }

    ActiveParents* parents_{nullptr};
    std::int32_t lcm_block_id_{0};
};

}  // namespace tokenspeed
