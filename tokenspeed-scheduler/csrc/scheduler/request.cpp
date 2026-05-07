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

#include "scheduler/request.h"

#include <stdexcept>

#include "fsm/cache_states.h"
#include "fsm/pd_states.h"

namespace tokenspeed {

Request::Request(const RequestSpec& spec, std::int32_t page_size, Role role)
    : id_{spec.request_id},
      lora_id_{spec.lora_id},
      token_container_{spec.tokens},
      page_size_{page_size},
      state_{role == Role::kFused ? fsm::State{fsm::Submitted{&token_container_, page_size}}
                                  : fsm::State{fsm::Bootstrapping{&token_container_, page_size}}},
      storage_info_{spec.rolling_hashes, spec.storage_hit_pages} {}

PrefillInfo Request::GetPrefillInfo() const {
    return std::visit(Overloaded{
        []<typename T>(const T& s) -> PrefillInfo
            requires(std::same_as<T, fsm::Prefilling> || std::same_as<T, fsm::PrefillDone>)
        { return s.GetPrefillInfo(); },
        [this](const auto&) -> PrefillInfo {
            throw std::logic_error("Request::GetPrefillInfo: expected state=Prefilling or PrefillDone; got state=" +
                                   StateName());
        },
        },
        state_);
}

TreeNode* Request::GetHostLockNode() const {
    auto* s = std::get_if<fsm::Prefetching>(&state_);
    if (s == nullptr) {
        throw std::logic_error("Request::GetHostLockNode: expected state=Prefetching; got state=" + StateName());
    }
    return s->GetHostLockNode();
}

std::vector<std::int32_t> Request::GetHostPageIds() const {
    auto* s = std::get_if<fsm::Prefetching>(&state_);
    if (s == nullptr) {
        throw std::logic_error("Request::GetHostPageIds: expected state=Prefetching; got state=" + StateName());
    }
    return s->GetHostPageIds();
}

}  // namespace tokenspeed
