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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "fsm/cache_events.h"
#include "fsm/cache_states.h"
#include "fsm/forward_states.h"
#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/radix_tree/tree_node.h"

namespace tokenspeed::fsm {

State SchedulePrefetchEvent::operator()(Submitted&& state) {
    return Prefetching{state.GetTokenContainer(), state.GetPageSize(), host_allocator_->Allocate(num_pages_to_fetch_),
                       std::make_unique<HostNodeRef>(host_match_node_)};
}

State PrefetchDoneEvent::operator()(Prefetching&& state) {
    OwnedPages pages = std::move(state).TakeHostPages();
    std::int32_t num_pages = pages.Size();

    auto completed = std::min(completed_num_pages_, num_pages);
    auto inserted = std::min(inserted_num_pages_, completed);
    auto overlapping = completed - inserted;

    if (overlapping > 0) {
        pages.TakeFirst(overlapping);
    }
    if (completed < num_pages) {
        pages.TakeLast(num_pages - completed);
    }
    auto inserted_pages = pages.Ids();
    // Inserted pages are now owned by the prefix cache tree; keep the RAII
    // wrapper from returning them to the host allocator on destruction.
    pages.ReleaseOwnershipByID(inserted_pages);

    return PrefetchDone{state.GetTokenContainer(), state.GetPageSize()};
}

State PrefetchDoneEvent::operator()(Aborting&& state) {
    return Finished{};
}

}  // namespace tokenspeed::fsm
