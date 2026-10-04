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
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace tokenspeed {

struct RequestSpec {
    std::string request_id;
    std::vector<std::int32_t> tokens;
    std::int32_t max_new_tokens{0};
    // Upper bound on the prompt prefix the first admission's probe may claim
    // from the prefix cache, in tokens. A request that returns prompt logprobs
    // from position `s` sets this to `s`, so every position at or after `s` is
    // recomputed and produces logits (a cached position has no logits). The
    // default places no bound beyond the ordinary replay tail. A readmission
    // after retraction relaxes it to the positions whose forward results had
    // landed before the retraction: their logits already exist.
    std::int32_t max_cached_prefix_tokens{std::numeric_limits<std::int32_t>::max()};
};

// One prefill chunk's model inputs. The input covers `extend_len` tokens
// starting at prompt position `already_scheduled_len`; the first `replay_len`
// of them sit inside a prefix hit and are re-fed only to regenerate the
// replayable cache groups (bounded replay). Progress is
// `already_scheduled_len + extend_len` regardless of replay_len.
struct PrefillInfo {
    std::span<const std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::int32_t already_scheduled_len{0};
    std::int32_t extend_len{0};
    std::int32_t replay_len{0};
};

}  // namespace tokenspeed
