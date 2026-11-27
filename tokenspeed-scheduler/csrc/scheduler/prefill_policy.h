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
#include <optional>
#include <span>

namespace tokenspeed {

class Request;
struct SchedulerConfig;
enum class PrefillPolicyKind : std::uint8_t;

enum class PrefillCompletion : std::uint8_t { kAllowPartial, kRequireComplete };

struct PrefillAllocation {
    Request* request;
    // Total forward input budget, including replay. Admission may shorten it
    // at a cache junction or checkpoint; it never exceeds this quota.
    std::int32_t max_tokens;
    std::int32_t yield_tokens;
    // Requiring completion is supported for a new prompt's first chunk.
    PrefillCompletion completion;
};

struct PrefillBudget {
    std::int32_t tokens;
    std::int32_t kv_pages;
    std::int32_t request_slots;
    std::int32_t batch_slots;
    bool stopped;
};

enum class PrefillAdmissionStatus : std::uint8_t {
    kScheduled,
    kPeerScheduled,
    kUnavailable,
    kCapacityBlocked,
    kStopped,
};

struct PrefillAdmission {
    PrefillAdmissionStatus status;
    std::int32_t input_tokens{0};
    bool incomplete{false};
    bool chunk_gave_way{false};
};

// Borrowed for one round. Policies choose requests, quotas and stop conditions;
// this context reuses existing prefix checks, serial admission and progress.
// Normal selections resolve peer work and the prefill gate before probing
// carry candidates. Admit performs local admission; complete-only carry calls
// it directly. Estimates are advisory: earlier admissions can invalidate a prefix.
class PrefillSchedulingContext {
public:
    virtual ~PrefillSchedulingContext() = default;
    virtual std::span<Request* const> Candidates() const = 0;
    virtual const SchedulerConfig& Config() const = 0;
    virtual PrefillBudget Budget() const = 0;
    virtual bool Eligible(const Request& request) const = 0;
    // Resolve peer work before probing local carry candidates. True means
    // this candidate must be skipped by the local prefill phase. False still
    // requires the prefill gate and local admission checks.
    virtual bool TryPeerLanding(Request& request) = 0;
    virtual bool PrefillAllowed(const Request& request) const = 0;
    virtual std::int32_t EstimateRemaining(Request& request) = 0;
    virtual std::optional<std::int32_t> WholePromptTokens(Request& request, Request* resident_holder) = 0;
    virtual PrefillAdmission Admit(const PrefillAllocation& allocation) = 0;
};

class PrefillPolicy {
public:
    virtual ~PrefillPolicy() = default;
    virtual void Schedule(PrefillSchedulingContext& context) = 0;
    // Also dispatched for a first admission granted by capacity recovery.
    virtual void FirstAdmissionSucceeded(std::span<Request* const> fifo, const Request& admitted) = 0;
};

std::unique_ptr<PrefillPolicy> MakePrefillPolicy(PrefillPolicyKind kind);

}  // namespace tokenspeed
