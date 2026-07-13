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
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tokenspeed {
namespace forward {
struct FlatKVGroupCompletion {
    std::string group_id;
    std::uint32_t completed_domain_mask{};
    // One exclusive raw end for every set bit in completed_domain_mask,
    // packed in ascending bit order.
    std::vector<std::int32_t> domain_valid_ends;
};

// Fence-ready, host-only POD. CUDA events, tensors, streams, and pointers are
// deliberately excluded from the scheduler ABI.
struct FlatKVCompletion {
    std::string request_id;
    std::uint64_t table_generation{};
    std::uint64_t dispatch_seq{};
    std::int32_t accepted_raw_end{};
    std::int32_t protected_raw_end{};
    std::vector<FlatKVGroupCompletion> groups;
};

struct ExtendResult {
    std::string request_id;
    // Tokens whose KV has already become stable request history.
    // token placeholder should be removed in python
    std::vector<std::int32_t> tokens;
    // Present only after the executor has made every required KV producer
    // domain host-visible. nullopt preserves the legacy/radix event ABI.
    std::optional<FlatKVCompletion> flat_kv_completion;
};

struct Finish {
    std::string request_id;
};

struct UpdateReserveNumTokens {
    std::string request_id;
    std::int32_t reserve_num_tokens_in_next_schedule_event;
};

struct Abort {
    std::string request_id;
};
}  // namespace forward

using ForwardEvent =
    std::variant<forward::ExtendResult, forward::Finish, forward::Abort, forward::UpdateReserveNumTokens>;

}  // namespace tokenspeed
