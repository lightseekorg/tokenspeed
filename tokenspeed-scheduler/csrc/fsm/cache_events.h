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
#include <string>
#include <utility>
#include <vector>

#include "fsm/base_event.h"
#include "fsm/states.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed {

class PageAllocator;
namespace fsm {

struct Aborting;
struct Prefetching;
struct Submitted;

struct SchedulePrefetchEvent : InvalidTransitionHandler<SchedulePrefetchEvent> {
    using InvalidTransitionHandler<SchedulePrefetchEvent>::operator();

    SchedulePrefetchEvent() = default;
    SchedulePrefetchEvent(std::int32_t num_pages_to_fetch, std::int32_t prefetch_start_page,
                          std::vector<std::string> rolling_page_hashes, PageAllocator* host_allocator,
                          std::unique_ptr<HostNodeRef> host_node_ref)
        : num_pages_to_fetch_{num_pages_to_fetch},
          prefetch_start_page_{prefetch_start_page},
          rolling_page_hashes_{std::move(rolling_page_hashes)},
          host_allocator_{host_allocator},
          host_node_ref_{std::move(host_node_ref)} {}

    State operator()(Submitted&& state);

    std::vector<std::string> TakeRollingPageHashes() { return std::move(rolling_page_hashes_); }

private:
    std::int32_t num_pages_to_fetch_{};
    std::int32_t prefetch_start_page_{};
    PageAllocator* host_allocator_{};
    std::unique_ptr<HostNodeRef> host_node_ref_;
    std::vector<std::string> rolling_page_hashes_;
};

struct PrefetchDoneEvent : InvalidTransitionHandler<PrefetchDoneEvent> {
    using InvalidTransitionHandler<PrefetchDoneEvent>::operator();

    PrefetchDoneEvent(std::int32_t completed_num_pages, std::int32_t inserted_num_pages)
        : completed_num_pages_{completed_num_pages}, inserted_num_pages_{inserted_num_pages} {}

    State operator()(Prefetching&& state);
    State operator()(Aborting&& state);

private:
    std::int32_t completed_num_pages_{};
    std::int32_t inserted_num_pages_{};
};

}  // namespace fsm
}  // namespace tokenspeed
