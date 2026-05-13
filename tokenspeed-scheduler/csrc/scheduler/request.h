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
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include "core/token_container.h"
#include "fsm/forward_states.h"
#include "fsm/states.h"
#include "resource/allocator/owned_pages.h"
#include "scheduler/request_spec.h"
#include "utils.h"

namespace tokenspeed {

class TreeNode;

namespace fsm {
struct Bootstrapping;
struct PrefetchDone;
struct Prefetching;
}  // namespace fsm

class Request {
public:
    Request(const RequestSpec& spec, std::int32_t page_size, Role role);

    std::string Id() const { return id_; }
    std::int32_t LoraId() const { return lora_id_; }

    // Keep Apply the only non-const function in Request
    // The wrapper lambda converts any concrete state type returned by event's operator()
    // into fsm::State, allowing operator() to return specific state types instead of State.
    template <typename Event>
    void Apply(Event&& event) {
        state_ = std::visit(
            [&event](auto&& s) -> fsm::State {
                // Call the event handler; result may be a concrete state or a
                // sub-variant<A, B, ...> (e.g. FinishEvent returns variant<Draining,Finished>).
                // Use if-constexpr to lift sub-variants into the full State type.
                auto result = std::forward<Event>(event)(std::move(s));
                using ResultT = std::remove_cvref_t<decltype(result)>;
                if constexpr (std::is_convertible_v<ResultT, fsm::State>) {
                    return fsm::State{std::move(result)};
                } else {
                    // result is a sub-variant: visit it to extract the concrete type.
                    return std::visit([](auto&& inner) -> fsm::State { return fsm::State{std::move(inner)}; },
                                      std::move(result));
                }
            },
            std::move(state_));
    }

    template <typename S>
    bool Is() const {
        return std::holds_alternative<S>(state_);
    }

    TreeNode* GetHostLockNode() const;
    // Returns the host pages allocated in Prefetching state (only valid when Is<Prefetching>()).
    std::vector<std::int32_t> GetHostPageIds() const;

    const StorageInfo& GetStorageInfo() const { return storage_info_; }

    std::vector<std::span<const std::int32_t>> GetFullPagedTokens(bool except_last) const {
        return token_container_.GetFullPagedTokens(page_size_, except_last);
    }

    std::int32_t TokenSize() const { return token_container_.Size(); }
    std::int32_t GetLastToken() const { return token_container_.LastToken(); }

    PrefillInfo GetPrefillInfo() const;

    std::int32_t PrefillSize() const { return token_container_.PrefillSize(); }

    std::int32_t UnScheduledPrefillSize() const {
        return std::visit(Overloaded{
            []<typename T>(const T& s) -> std::int32_t
                requires(std::same_as<T, fsm::Submitted>)
                        { return -1; },
                        [this]<typename T>(const T& s) -> std::int32_t
                            requires(std::same_as<T, fsm::Prefilling>)
            { return PrefillSize() - (s.window.begin + s.window.size); },
            [](const auto&) -> std::int32_t { return 0; },
            },
            state_);
    }

    std::int32_t TailPageAvailableTokens() const {
        return std::visit(
            [](const auto& s) -> std::int32_t {
                using T = std::decay_t<decltype(s)>;
                if constexpr (std::is_base_of_v<fsm::ForwardState, T>) {
                    return s.TailPageAvailableTokens();
                }
                return 0;
            },
            state_);
    }

    std::int32_t GetReqPoolIndex() const {
        return std::visit(Overloaded{
            []<typename T>(const T& s) -> std::int32_t
                requires std::derived_from<T, fsm::ForwardState>
            { return s.GetReqPoolIndex(); },
            [this](const auto&) -> std::int32_t {
                throw std::logic_error("Request::GetReqPoolIndex: expected a forward state; got state=" + StateName());
            },
            },
            state_);
    }

    std::vector<std::int32_t> GetOccupiedPages() const {
        return std::visit(Overloaded{
            []<typename T>(const T& s) -> std::vector<std::int32_t>
                requires(std::derived_from<T, fsm::ForwardState> || std::derived_from<T, fsm::Submitted> ||
                         std::same_as<T, fsm::Aborting> || std::same_as<T, fsm::PrefetchDone>)
            { return s.GetOccupiedPages(); },
            [this](const auto&) -> std::vector<std::int32_t> {
                throw std::logic_error(
                    "Request::GetOccupiedPages: expected state=Submitted, PrefetchDone, Aborting, or forward; got "
                    "state=" +
                    StateName());
            },
            },
            state_);
    }

    const TreeNode* GetDeviceNode() const {
        return std::visit(Overloaded{
            []<typename T>(const T& s) -> const TreeNode*
                requires(std::derived_from<T, fsm::BaseState>)
            { return s.GetDeviceNode(); },
            [this](const auto&) -> const TreeNode* {
                throw std::logic_error("Request::GetDeviceNode: expected a base request state; got state=" +
                                       StateName());
            },
            },
            state_);
    }

    // Returns ONLY the pages held by the local KV allocator (tail pages exclusively
    // owned by this request, NOT shared radix-tree prefix pages).
    // Valid for ForwardState (Prefilling/PrefillDone/Decoding), Retracting and Retracted.
    // Returns empty vector for all other states.
    std::vector<std::int32_t> GetLocalAllocatorPages() const {
        return std::visit(Overloaded{
            []<typename T>(const T& s) -> std::vector<std::int32_t>
                requires(std::derived_from<T, fsm::ForwardState>)
            { return s.GetLocalAllocatorPages(); },
            [](const fsm::Retracting& s) -> std::vector<std::int32_t> { return s.GetLocalAllocatorPages(); },
            [](const fsm::Retracted& s) -> std::vector<std::int32_t> { return s.GetLocalAllocatorPages(); },
            [](const auto&) -> std::vector<std::int32_t> { return {}; },
            },
            state_);
    }

    OwnedPages TakeFullPages() {
        return std::visit(Overloaded{
            []<typename T>(T& s) -> OwnedPages
                requires(std::same_as<T, fsm::Decoding> || std::same_as<T, fsm::PrefillDone>)
            { return s.GetLocalKVAllocatorPtr()->TakeFullPages(); },
            [this](auto&) -> OwnedPages {
                throw std::logic_error("Request::TakeFullPages: expected state=Decoding or PrefillDone; got state=" +
                                       StateName());
            },
            },
            state_);
    }

    OwnedPages TakeFirstPages(std::int32_t n) {
        return std::visit(Overloaded{
            [n]<typename T>(T& s) -> OwnedPages
                requires(std::same_as<T, fsm::Decoding> || std::same_as<T, fsm::PrefillDone>)
            { return s.GetLocalKVAllocatorPtr()->TakeFirst(n); },
            [this](auto&) -> OwnedPages {
                throw std::logic_error("Request::TakeFirstPages: expected state=Decoding or PrefillDone; got state=" +
                                       StateName());
            },
            },
            state_);
    }

    // Returns a human-readable name for the current FSM state.
    std::string StateName() const {
        return std::visit(Overloaded{
                              [](const fsm::Bootstrapping&) -> std::string { return "Bootstrapping"; },
                              [](const fsm::Submitted&) -> std::string { return "Submitted"; },
                              [](const fsm::Prefetching&) -> std::string { return "Prefetching"; },
                              [](const fsm::PrefetchDone&) -> std::string { return "PrefetchDone"; },
                              [](const fsm::Aborting&) -> std::string { return "Aborting"; },
                              [](const fsm::Prefilling&) -> std::string { return "Prefilling"; },
                              [](const fsm::PrefillDone&) -> std::string { return "PrefillDone"; },
                              [](const fsm::Decoding&) -> std::string { return "Decoding"; },
                              [](const fsm::Draining&) -> std::string { return "Draining"; },
                              [](const fsm::WritingBack&) -> std::string { return "WritingBack"; },
                              [](const fsm::Retracting&) -> std::string { return "Retracting"; },
                              [](const fsm::Retracted&) -> std::string { return "Retracted"; },
                              [](const fsm::Finished&) -> std::string { return "Finished"; },
                          },
                          state_);
    }

    std::int32_t GetReserveNumTokensInNextScheduleEvent() const {
        return std::visit(Overloaded{
            []<typename T>(const T& s) -> std::int32_t
                requires(std::same_as<T, fsm::PrefillDone> || std::same_as<T, fsm::Decoding>)
            { return s.GetReserveNumTokensInNextScheduleEvent(); },
            [this](const auto&) -> std::int32_t {
                throw std::logic_error(
                    "Request::GetReserveNumTokensInNextScheduleEvent: expected state=PrefillDone or Decoding; got "
                    "state=" +
                    StateName());
            },
            },
            state_);
    }

    LocalMambaAllocator* GetLocalMambaAllocator() {
        return std::visit(Overloaded{
            []<typename T>(T& s) -> LocalMambaAllocator*
                requires(std::derived_from<T, fsm::BaseState>)
            { return s.GetLocalMambaAllocator(); },
            [](auto&) -> LocalMambaAllocator* { return nullptr; },
            },
            state_);
    }

    // Returns the (device_page, host_page) pairs that were captured immediately after
    // host pages were allocated inside the state transition event.
    // S must be fsm::Draining (captured in FinishEvent) or
    //           fsm::Retracting (captured in ScheduleRetractEvent).
    template <typename S>
        requires(std::same_as<S, fsm::Draining> || std::same_as<S, fsm::Retracting>)
    const std::vector<typename S::PagePair>& GetPagesToTransfer() const {
        return std::visit(Overloaded{
            []<typename T>(const T& s) -> const std::vector<typename S::PagePair>&
                requires(std::same_as<T, S>)
            { return s.GetPagesToTransfer(); },
            [this](const auto&) -> const std::vector<typename S::PagePair>& {
                throw std::logic_error("Request::GetPagesToTransfer: expected state=" +
                                       std::string(detail::TypeName<S>()) + "; got state=" + StateName());
            },
            },
            state_);
    }

private:
    std::string id_;
    std::int32_t lora_id_{0};
    TokenContainer token_container_;
    std::int32_t page_size_;
    fsm::State state_;
    StorageInfo storage_info_;
};

using ConstRequestVector = std::vector<const Request*>;

}  // namespace tokenspeed
