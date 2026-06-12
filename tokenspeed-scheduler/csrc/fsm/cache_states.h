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

#include <cstdint>
#include <memory>
#include <vector>

#include "core/token_container.h"
#include "resource/allocator/owned_pages.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed::fsm {

struct Prefetching {
    Prefetching(TokenContainer* token_container, std::int32_t page_size, OwnedPages host_pages,
                std::unique_ptr<HostNodeRef>&& host_lock, std::int32_t prefetch_start_page)
        : token_container_{token_container},
          page_size_{page_size},
          host_pages_{std::move(host_pages)},
          host_lock_{std::move(host_lock)},
          prefetch_start_page_{prefetch_start_page} {}

    ~Prefetching() = default;

    Prefetching(Prefetching&&) noexcept = default;
    Prefetching& operator=(Prefetching&&) noexcept = default;
    Prefetching(const Prefetching&) = delete;
    Prefetching& operator=(const Prefetching&) = delete;

    TokenContainer* GetTokenContainer() const { return token_container_; }
    std::int32_t GetPageSize() const { return page_size_; }

    std::vector<std::int32_t> GetHostPageIds() const { return host_pages_.Ids(); }

    OwnedPages TakeHostPages() && { return std::move(host_pages_); }

    TreeNode* GetHostLockNode() const { return host_lock_ ? host_lock_->Node() : nullptr; }
    std::int32_t GetPrefetchStartPage() const { return prefetch_start_page_; }

private:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};
    OwnedPages host_pages_;
    std::unique_ptr<HostNodeRef> host_lock_;
    std::int32_t prefetch_start_page_{};
};

struct PrefetchDone {
    PrefetchDone(TokenContainer* token_container, std::int32_t page_size)
        : token_container_{token_container}, page_size_{page_size} {}

    ~PrefetchDone() = default;

    PrefetchDone(PrefetchDone&&) noexcept = default;
    PrefetchDone& operator=(PrefetchDone&&) noexcept = default;
    PrefetchDone(const PrefetchDone&) = delete;
    PrefetchDone& operator=(const PrefetchDone&) = delete;

    TokenContainer* GetTokenContainer() const { return token_container_; }
    std::int32_t GetPageSize() const { return page_size_; }

    std::vector<std::int32_t> GetOccupiedPages() const { return {}; }

private:
    TokenContainer* token_container_{};
    std::int32_t page_size_{};
};

}  // namespace tokenspeed::fsm
