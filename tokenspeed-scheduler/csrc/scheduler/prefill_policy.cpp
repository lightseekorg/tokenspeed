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
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/profile_span.h"
#include "fsm/forward_states.h"
#include "scheduler/request.h"
#include "scheduler/types.h"
#include "utils.h"

namespace tokenspeed {
namespace {

constexpr std::int32_t kMaxPrefillBypasses = 8;
constexpr std::int32_t kMaxYieldProbesPerRound = 8;

struct WholePrompt {
    std::size_t index;
    Request* request;
    std::int32_t tokens;
    std::int32_t pages;
};

// The serial policies probe a bounded FIFO list once for carry and FIFO fill.
std::vector<WholePrompt> wholePromptsAfter(PrefillSchedulingContext& context, std::size_t from,
                                           Request* resident_holder) {
    const auto& config = context.Config();
    const auto candidates = context.Candidates();
    std::vector<WholePrompt> prompts;
    std::int32_t probes = 0;
    for (std::size_t index = from; index < candidates.size() && probes < kMaxYieldProbesPerRound; ++index) {
        Request* request = candidates[index];
        if (!request->Is<fsm::Submitted>() || !context.Eligible(*request)) continue;
        ++probes;
        const auto tokens = context.WholePromptTokens(*request, resident_holder);
        if (tokens && *tokens <= std::max(config.prefill_chunk_yield_tokens, context.Budget().tokens)) {
            prompts.push_back({index, request, *tokens, CeilDiv(request->PrefillSize(), config.prefix_granularity)});
        }
    }
    return prompts;
}

std::vector<WholePrompt> carriedPrompts(PrefillSchedulingContext& context, std::span<const WholePrompt> prompts,
                                        std::size_t from, bool holder_is_new, std::int32_t holder_max_pages) {
    const auto& config = context.Config();
    const auto budget = context.Budget();
    const std::int64_t slots = std::int64_t{budget.request_slots} - (holder_is_new ? 1 : 0);
    const std::int64_t page_budget = std::int64_t{budget.kv_pages} - holder_max_pages;
    std::vector<WholePrompt> carried;
    std::int64_t tokens = 0;
    std::int64_t pages = 0;
    for (const auto& prompt : prompts) {
        if (prompt.index < from || prompt.tokens > config.prefill_chunk_yield_tokens ||
            !prompt.request->Is<fsm::Submitted>() || !context.Eligible(*prompt.request))
            continue;
        const auto count = static_cast<std::int64_t>(carried.size()) + 1;
        if (tokens + prompt.tokens > config.prefill_chunk_yield_tokens || count + 1 > budget.batch_slots ||
            count > slots || (config.max_kv_page_num > 0 && pages + prompt.pages > page_budget))
            break;
        tokens += prompt.tokens;
        pages += prompt.pages;
        carried.push_back(prompt);
    }
    return carried;
}

struct CarryResult {
    std::int32_t admitted{0};
    bool stopped{false};
};

CarryResult carryWholePrompts(PrefillSchedulingContext& context, std::span<const WholePrompt> prompts) {
    ProfileSpan profile_span{"Scheduler::carryWholePrompts"};
    CarryResult result;
    for (const auto& prompt : prompts) {
        const auto budget = context.Budget();
        if (budget.stopped) return {result.admitted, true};
        if (!prompt.request->Is<fsm::Submitted>() || !context.Eligible(*prompt.request) ||
            prompt.tokens > budget.tokens)
            continue;
        const auto admission = context.Admit({prompt.request, budget.tokens, 0, PrefillCompletion::kRequireComplete});
        if (admission.status == PrefillAdmissionStatus::kScheduled)
            ++result.admitted;
        else if (admission.status == PrefillAdmissionStatus::kCapacityBlocked ||
                 admission.status == PrefillAdmissionStatus::kStopped)
            return {result.admitted, true};
    }
    return result;
}

class FifoSelection {
public:
    explicit FifoSelection(PrefillSchedulingContext& context) : pending_{context.Candidates()} {}
    Request* Next(PrefillSchedulingContext&) {
        if (pending_.empty()) return nullptr;
        Request* request = pending_.front();
        pending_ = pending_.subspan(1);
        return request;
    }
    void Fill(PrefillSchedulingContext& context, std::optional<std::vector<WholePrompt>>& whole_prompts,
              std::size_t from) {
        if (!whole_prompts) whole_prompts = wholePromptsAfter(context, from, nullptr);
        const auto behind =
            std::ranges::find_if(*whole_prompts, [from](const auto& prompt) { return prompt.index >= from; });
        carryWholePrompts(context, std::span<const WholePrompt>(behind, whole_prompts->end()));
    }

private:
    std::span<Request* const> pending_;
};

class ShortPrefillSelection {
public:
    explicit ShortPrefillSelection(PrefillSchedulingContext&) {}
    Request* Next(PrefillSchedulingContext& context) {
        Load(context);
        std::erase_if(pending_, [&context](const auto& prompt) {
            return !prompt.request->template Is<fsm::Submitted>() || !context.Eligible(*prompt.request);
        });
        if (pending_.empty()) return nullptr;
        // Stable ties and protected requests retain submission order. Age is
        // read after every successful first admission, including within a batch.
        const auto selected = std::min_element(pending_.begin(), pending_.end(), [](const auto& a, const auto& b) {
            const bool aged_a =
                a.request->template GetIf<fsm::Submitted>()->admission.prefill_bypasses >= kMaxPrefillBypasses;
            const bool aged_b =
                b.request->template GetIf<fsm::Submitted>()->admission.prefill_bypasses >= kMaxPrefillBypasses;
            if (aged_a != aged_b) return aged_a;
            return !aged_a && a.remaining_tokens < b.remaining_tokens;
        });
        Request* request = selected->request;
        pending_.erase(selected);
        return request;
    }
    void Fill(PrefillSchedulingContext& context, std::optional<std::vector<WholePrompt>>&, std::size_t) {
        Load(context);  // A resident's admission has now published its progress.
        while (!context.Budget().stopped && context.Budget().request_slots > 0) {
            std::erase_if(pending_,
                          [&context](const auto& prompt) { return prompt.remaining_tokens > context.Budget().tokens; });
            Request* request = Next(context);
            if (!request) return;
            ProfileSpan profile_span{"Scheduler::carryWholePrompts"};
            const auto result =
                context.Admit({request, context.Budget().tokens, 0, PrefillCompletion::kRequireComplete});
            if (result.status == PrefillAdmissionStatus::kCapacityBlocked ||
                result.status == PrefillAdmissionStatus::kStopped)
                return;
        }
    }

private:
    struct Candidate {
        Request* request;
        std::int32_t remaining_tokens;
    };
    void Load(PrefillSchedulingContext& context) {
        if (loaded_) return;
        loaded_ = true;
        for (Request* request : context.Candidates()) {
            if (request->Is<fsm::Submitted>() && context.Eligible(*request)) {
                pending_.push_back({request, context.EstimateRemaining(*request)});
            }
        }
    }
    bool loaded_{false};
    std::vector<Candidate> pending_;
};

// Shared serial policy: residents first; a partial keeps the line and only
// P-role whole prompts may use its remaining budget. New policies need not
// use this algorithm: Schedule can submit any safe selection and quota.
template <typename Selection, typename NewPrompts>
bool scheduleTier(PrefillSchedulingContext& context, Selection& selection, NewPrompts& new_prompts, bool resident,
                  std::optional<std::vector<WholePrompt>>& whole_prompts) {
    const auto& config = context.Config();
    while (true) {
        const auto budget = context.Budget();
        if (budget.stopped || (!resident && budget.request_slots == 0)) return false;
        Request* request = selection.Next(context);
        if (!request) return true;
        if ((resident ? !request->Is<fsm::Prefilling>() : !request->Is<fsm::Submitted>()) ||
            !context.Eligible(*request))
            continue;
        if (context.TryPeerLanding(*request)) continue;
        if (!context.PrefillAllowed(*request)) return false;
        std::size_t from = 0;
        if (config.role == Role::kP && !resident) {
            const auto candidates = context.Candidates();
            from = static_cast<std::size_t>(std::ranges::find(candidates, request) - candidates.begin()) + 1;
        }
        std::vector<WholePrompt> carried;
        std::int32_t yield_tokens = 0;
        if (config.prefill_chunk_yield_tokens > 0 &&
            request->PrefillRoundsGivenWay() < config.prefill_chunk_yield_max_rounds &&
            (resident ? request->UnscheduledPrefillSize() : request->PrefillSize()) > budget.tokens) {
            if (!whole_prompts) whole_prompts = wholePromptsAfter(context, from, resident ? request : nullptr);
            const std::int64_t holder_end =
                resident ? std::int64_t{request->PrefillSize() - request->UnscheduledPrefillSize()} + budget.tokens
                         : request->PrefillSize();
            const auto holder_pages =
                CheckedInt32(holder_end / config.prefix_granularity + (holder_end % config.prefix_granularity != 0));
            carried = carriedPrompts(context, *whole_prompts, from, !resident, holder_pages);
            for (const auto& prompt : carried) yield_tokens += prompt.tokens;
        }
        const auto result = context.Admit({request, budget.tokens, yield_tokens, PrefillCompletion::kAllowPartial});
        if (result.status == PrefillAdmissionStatus::kStopped) return false;
        if (result.status == PrefillAdmissionStatus::kCapacityBlocked && resident) return false;
        if (result.status != PrefillAdmissionStatus::kScheduled || !result.incomplete) continue;
        const auto carry = carryWholePrompts(context, carried);
        if (carry.admitted > 0 || result.chunk_gave_way) request->NotePrefillGaveWay();
        if (config.role == Role::kP && !carry.stopped && !context.Budget().stopped) {
            new_prompts.Fill(context, whole_prompts, from);
        }
        return false;
    }
}

template <typename Selection>
void scheduleSerial(PrefillSchedulingContext& context) {
    FifoSelection residents{context};
    Selection new_prompts{context};
    std::optional<std::vector<WholePrompt>> whole_prompts;
    if (scheduleTier(context, residents, new_prompts, true, whole_prompts)) {
        scheduleTier(context, new_prompts, new_prompts, false, whole_prompts);
    }
}

class FifoPrefillPolicy final : public PrefillPolicy {
public:
    void Schedule(PrefillSchedulingContext& context) override { scheduleSerial<FifoSelection>(context); }
    void FirstAdmissionSucceeded(std::span<Request* const>, const Request&) override {}
};

class ShortPrefillFirstPolicy final : public PrefillPolicy {
public:
    void Schedule(PrefillSchedulingContext& context) override { scheduleSerial<ShortPrefillSelection>(context); }
    void FirstAdmissionSucceeded(std::span<Request* const> fifo, const Request& admitted) override {
        // Only the oldest waiting prompt ages; aging every earlier prompt
        // would turn short-first back into FIFO after eight admissions.
        for (Request* older : fifo) {
            if (older == &admitted) break;
            if (auto* waiting = older->GetIf<fsm::Submitted>()) {
                waiting->admission.prefill_bypasses =
                    std::min(kMaxPrefillBypasses, waiting->admission.prefill_bypasses + 1);
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
