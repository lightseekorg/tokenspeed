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

#include "fsm/pd_events.h"

#include <utility>
#include <vector>

#include "core/token_container.h"
#include "fsm/pd_states.h"

namespace tokenspeed {
namespace fsm {

Submitted BootstrappedEvent::operator()(Bootstrapping&& state) {
    return Submitted{state.token_container, state.page_size};
}

Finished SucceededEvent::operator()(Decoding&& /*state*/) {
    return Finished{};
}

PrefillDone RemotePrefillDoneEvent::operator()(Prefilling&& state) {
    TokenContainer::Window w = state.window;
    auto prefill_done = PrefillDone{
        state.GetTokenContainer(),
        state.GetPageSize(),
        std::move(state).TakeHostNodeRef(),
        std::move(state).TakeDeviceNodeRef(),
        std::move(state).TakeLocalKVAllocator(),
        std::move(state).TakeReqPoolIndex(),
        w,
        0,  // reserve_num_tokens_in_next_schedule_event
        std::move(state).TakeLocalMambaAllocator(),
    };

    prefill_done.ExtendResultTokens({bootstrap_token});
    return prefill_done;
}

}  // namespace fsm
}  // namespace tokenspeed
