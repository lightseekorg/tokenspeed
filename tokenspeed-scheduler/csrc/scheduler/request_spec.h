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

#include <span>
#include <cstdint>
#include <string>
#include <vector>

namespace tokenspeed {

struct RequestSpec {
    std::string request_id;
    std::vector<std::int32_t> tokens;
    std::vector<std::string> rolling_hashes;
    std::int32_t storage_hit_pages{0};
    // 0 = base model (no adapter).  >0 = LoRA adapter integer ID from
    // LoraRegistry.  The prefix cache is namespaced per lora_id so adapters
    // never share KV pages with different LoRA weights.
    std::int32_t lora_id{0};
};

struct PrefillInfo {
    std::span<const std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::int32_t already_scheduled_len;
    std::int32_t extend_len;
};

struct StorageInfo {
    std::vector<std::string> rolling_hashes;
    std::int32_t hit_pages{0};
};

}  // namespace tokenspeed
