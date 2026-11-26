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

#include "scheduler/prefill_policy.h"

#include <algorithm>
#include <stdexcept>

#include "fsm/forward_states.h"
#include "scheduler/request.h"
#include "scheduler/types.h"

namespace tokenspeed {
namespace {

constexpr std::int32_t kMaxPrefillBypasses = 8;

PrefillCandidate popFifo(PrefillCandidates& pending) {
    if (pending.fifo.empty()) return {nullptr, 0};
    Request* request = pending.fifo.front();
    pending.fifo = pending.fifo.subspan(1);
    return {request, 0};
}

class FifoPrefillPolicy final : public PrefillPolicy {
public:
    bool RanksNewPrompts() const override { return false; }
    PrefillCandidate PopNext(PrefillCandidates& pending) const override { return popFifo(pending); }
    void FirstAdmissionSucceeded(std::span<Request* const>, const Request&) const override {}
};

class ShortPrefillFirstPolicy final : public PrefillPolicy {
public:
    bool RanksNewPrompts() const override { return true; }
    PrefillCandidate PopNext(PrefillCandidates& pending) const override {
        // Resident chunks keep submission order under either policy.
        if (!pending.fifo.empty()) return popFifo(pending);
        if (pending.entries.empty()) return {nullptr, 0};
        // Stable ties and protected requests retain submission order. Age is
        // read after every successful first admission, including within a batch.
        const auto selected = std::min_element(
            pending.entries.begin(), pending.entries.end(), [](const PrefillCandidate& a, const PrefillCandidate& b) {
                const bool aged_a = a.request->GetIf<fsm::Submitted>()->prefill_bypasses >= kMaxPrefillBypasses;
                const bool aged_b = b.request->GetIf<fsm::Submitted>()->prefill_bypasses >= kMaxPrefillBypasses;
                if (aged_a != aged_b) return aged_a;
                return !aged_a && a.remaining_tokens < b.remaining_tokens;
            });
        const PrefillCandidate result = *selected;
        pending.entries.erase(selected);
        return result;
    }
    void FirstAdmissionSucceeded(std::span<Request* const> fifo, const Request& admitted) const override {
        // Only the oldest waiting prompt ages; aging every earlier prompt
        // would turn short-first back into FIFO after eight admissions.
        for (Request* older : fifo) {
            if (older == &admitted) break;
            if (auto* waiting = older->GetIf<fsm::Submitted>()) {
                waiting->prefill_bypasses = std::min(kMaxPrefillBypasses, waiting->prefill_bypasses + 1);
                break;
            }
        }
    }
};

}  // namespace

std::unique_ptr<PrefillPolicy> MakePrefillPolicy(PrefillPolicyKind kind) {
    switch (kind) {
        case PrefillPolicyKind::kFifo:
            return std::make_unique<FifoPrefillPolicy>();
        case PrefillPolicyKind::kShortPrefillFirst:
            return std::make_unique<ShortPrefillFirstPolicy>();
    }
    throw std::invalid_argument("Scheduler: prefill_policy must be fifo or short-prefill-first");
}

}  // namespace tokenspeed
