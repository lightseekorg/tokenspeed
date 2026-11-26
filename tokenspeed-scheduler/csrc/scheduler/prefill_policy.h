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
#include <span>
#include <vector>

namespace tokenspeed {

class Request;
enum class PrefillPolicyKind : std::uint8_t;

// Work estimates belong to a plan-building pass, never to the policy.
struct PrefillCandidate {
    Request* request;
    std::int32_t remaining_tokens;
};

struct PrefillCandidates {
    // New short-first prompts use ranked estimates. FIFO and resident chunks
    // consume the original queue without copying or probing it.
    std::vector<PrefillCandidate> entries;
    std::span<Request* const> fifo;
};

// Startup-selected prefill ordering. Admission, chunk sizing, cache ownership
// and per-request progress remain in the shared Scheduler.
class PrefillPolicy {
public:
    virtual ~PrefillPolicy() = default;
    virtual bool RanksNewPrompts() const = 0;
    virtual PrefillCandidate PopNext(PrefillCandidates& pending) const = 0;
    virtual void FirstAdmissionSucceeded(std::span<Request* const> fifo, const Request& admitted) const = 0;
};

std::unique_ptr<PrefillPolicy> MakePrefillPolicy(PrefillPolicyKind kind);

}  // namespace tokenspeed
