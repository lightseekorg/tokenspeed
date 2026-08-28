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

#include <concepts>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/token_container.h"
#include "fsm/forward_states.h"
#include "fsm/states.h"
#include "scheduler/request_spec.h"
#include "utils.h"

namespace tokenspeed {

class Request {
public:
    Request(const RequestSpec& spec, std::int32_t prefix_granularity, Role role);

    const std::string& Id() const { return id_; }

    // Decode headroom an admission must secure before the prefill starts:
    // one safe-step window up front, plus one more per retraction suffered
    // -- being retracted means the previous admission was still too
    // optimistic. Capped by the generation budget the request could ever
    // use, which makes the escalation terminate.
    //
    // A request with no declared budget (max_new_tokens == 0) demands none,
    // and keeps demanding none however often it is retracted. That looks
    // like an omission but is the only safe reading: the declared budget is
    // what bounds the escalation, and without one there is nothing to stop
    // it from demanding more than the pool can ever hold -- at which point
    // the request could not be readmitted at all. An undeclared budget
    // stays optimistic and relies on the victim policy (which exempts a
    // request already recovering) to make progress.
    std::int32_t AdmissionHeadroom(std::int32_t safe_steps) const {
        return std::min(max_new_tokens_, safe_steps * (1 + retraction_count_));
    }
    void NoteRetracted() { ++retraction_count_; }

    // True when the last admission's headroom already covers every token
    // this request could still generate. Retracting such a request is pure
    // thrash -- its readmission must take back exactly what the retraction
    // freed -- so the victim policy skips it. (An undeclared budget is
    // never covered: nothing was reserved for it.)
    bool ReserveCoversGeneration(std::int32_t safe_steps) const {
        return max_new_tokens_ > 0 && AdmissionHeadroom(safe_steps) >= max_new_tokens_;
    }

    // Whether any generated token exists, whatever state currently holds it.
    // Survives retraction's RebasePrefill (which folds generated tokens into
    // the prefill window): the comparison is against the SUBMITTED prompt.
    bool HasGeneratedOutput() const { return TokenSize() > submitted_prompt_size_; }

    template <typename Event>
    void Apply(Event&& event) {
        state_ = std::visit(
            [&event](auto&& state) -> fsm::State { return fsm::ToState(std::forward<Event>(event)(std::move(state))); },
            std::move(state_));
    }

    template <typename State>
    bool Is() const {
        return std::holds_alternative<State>(state_);
    }

    template <typename State>
    const State* GetIf() const {
        return std::get_if<State>(&state_);
    }

    // Forwards scheduled for this request whose results have not come back.
    // Every state that holds pages answers this; the page-less ones
    // (Submitted, Retracted, Finished) owe nothing by construction.
    std::int32_t ResultsInFlight() const {
        return std::visit(Overloaded{
                              [](const std::derived_from<fsm::ForwardState> auto& s) { return s.ResultsInFlight(); },
                              [](const auto&) { return 0; },
                          },
                          state_);
    }

    void TrackScheduledForward() {
        std::visit(Overloaded{
                       [](std::derived_from<fsm::ForwardState> auto& s) { s.TrackScheduledForward(); },
                       [](auto&) {},
                   },
                   state_);
    }

    void NoteResultLanded() {
        std::visit(Overloaded{
                       [](std::derived_from<fsm::ForwardState> auto& s) { s.ResultLanded(); },
                       [](auto&) {},
                   },
                   state_);
    }

    std::vector<std::span<const std::int32_t>> FullPrefixPages(bool except_last) const {
        return token_container_.FullPrefixPages(prefix_granularity_, except_last);
    }

    std::int32_t TokenSize() const { return token_container_.Size(); }
    std::int32_t LastToken() const { return token_container_.LastToken(); }
    // P role: the drafter candidates that arrived with the final chunk's
    // ExtendResult, held until the remote-decode operation carries them out.
    void StoreSpecCandidates(std::vector<std::int32_t> ids) { spec_candidate_ids_ = std::move(ids); }
    std::vector<std::int32_t> TakeSpecCandidates() { return std::exchange(spec_candidate_ids_, {}); }
    std::int32_t PrefillSize() const { return token_container_.PrefillSize(); }
    PrefillInfo CurrentPrefillInfo() const;

    std::int32_t UnscheduledPrefillSize() const {
        return std::visit(Overloaded{
                              [](const fsm::Submitted&) -> std::int32_t { return -1; },
                              [this](const fsm::Prefilling& state) -> std::int32_t {
                                  return PrefillSize() - (state.window.begin + state.window.size);
                              },
                              [](const auto&) -> std::int32_t { return 0; },
                          },
                          state_);
    }

    std::int32_t RequestPoolIndex() const { return forwardState("RequestPoolIndex").RequestPoolIndex(); }

    const std::vector<BlockTable>& BlockTablesRef() const { return forwardState("BlockTablesRef").BlockTables(); }

    std::vector<BlockTable>& BlockTablesRef() { return forwardState("BlockTablesRef").BlockTables(); }

    fsm::CacheProgress CacheProgress() const { return forwardState("CacheProgress").CacheProgressRef(); }

    std::int32_t ReserveNumTokensInNextScheduleEvent() const {
        return std::visit(
            Overloaded{
                [](const fsm::PrefillDone& state) { return state.ReserveNumTokensInNextScheduleEvent(); },
                [](const fsm::PrefillAwaitingResult& state) { return state.ReserveNumTokensInNextScheduleEvent(); },
                [](const fsm::Decoding& state) { return state.ReserveNumTokensInNextScheduleEvent(); },
                [this](const auto&) -> std::int32_t {
                    throw std::logic_error(
                        "Request::ReserveNumTokensInNextScheduleEvent: expected PrefillDone or Decoding; got " +
                        StateName());
                },
            },
            state_);
    }

    std::string StateName() const {
        return std::visit(Overloaded{
                              [](const fsm::Bootstrapping&) -> std::string { return "Bootstrapping"; },
                              [](const fsm::Submitted&) -> std::string { return "Submitted"; },
                              [](const fsm::Prefilling&) -> std::string { return "Prefilling"; },
                              [](const fsm::RemotePrefilling&) -> std::string { return "RemotePrefilling"; },
                              [](const fsm::PrefillAwaitingResult&) -> std::string { return "PrefillAwaitingResult"; },
                              [](const fsm::PrefillDone&) -> std::string { return "PrefillDone"; },
                              [](const fsm::Decoding&) -> std::string { return "Decoding"; },
                              [](const fsm::Retracted&) -> std::string { return "Retracted"; },
                              [](const fsm::Finished&) -> std::string { return "Finished"; },
                          },
                          state_);
    }

private:
    fsm::ForwardState& forwardState(const char* operation);
    const fsm::ForwardState& forwardState(const char* operation) const;

    std::string id_;
    TokenContainer token_container_;
    std::int32_t submitted_prompt_size_{0};
    std::int32_t max_new_tokens_{0};
    std::int32_t retraction_count_{0};
    std::vector<std::int32_t> spec_candidate_ids_;
    std::int32_t prefix_granularity_{};
    fsm::State state_;
};

}  // namespace tokenspeed
