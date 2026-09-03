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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tokenspeed {

struct ForwardOperationBase {
    std::string request_id;
    std::int32_t request_pool_index{-1};
    std::int32_t input_length{0};
    std::int32_t prefill_length{0};

    // Per-group block tables. Rows use absolute logical-page indexing; null
    // holes are page 0 and rows are not compacted.
    std::map<std::string, std::vector<std::int32_t>> block_tables;
};

struct PrefillOperation : public ForwardOperationBase {
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::int32_t extend_prefix_len{0};
};

struct DecodeOperation : public ForwardOperationBase {
    std::int32_t decode_input_id = -1;
    // P-role remote decode only: drafter candidates for the decode side
    // (candidate[0] == decode_input_id). Empty everywhere else.
    std::vector<std::int32_t> spec_candidate_ids;
};

using ForwardOperation = std::variant<PrefillOperation, DecodeOperation>;

struct ForwardBatch {
    std::vector<std::string> request_ids;
    std::vector<std::int32_t> request_pool_indices;
    std::vector<std::int32_t> input_lengths;
    // Per-request total number of prompt tokens (Request::PrefillSize()).
    std::vector<std::int32_t> prefill_lengths;

    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::vector<std::int32_t> extend_prefix_lens;
    std::vector<std::int32_t> decode_input_ids;
    // Parallel to decode_input_ids (one entry per decode row); rows without
    // candidates hold an empty vector.
    std::vector<std::vector<std::int32_t>> spec_candidate_ids;

    // Per-group block tables: dict[group_id] =
    // [num_reqs, max_pages_in_batch] padded with -1. Each row is absolute
    // (null hole = 0, no compaction); there is no base-offset companion.
    std::map<std::string, std::vector<std::vector<std::int32_t>>> block_tables;
    // Contiguous row-major copy of block_tables ([rows * cols], -1
    // padded), exposed zero-copy to Python as a 2-D ndarray -- the nested
    // vectors above cost one PyLong per page id at every attribute access.
    std::map<std::string, std::vector<std::int32_t>> block_tables_contig;
    explicit ForwardBatch(std::vector<ForwardOperation> ops) {
        std::stable_partition(ops.begin(), ops.end(),
                              [](const ForwardOperation& a) { return std::holds_alternative<PrefillOperation>(a); });
        for (auto& op : ops) {
            std::visit(
                [this](auto& inner) {
                    request_ids.push_back(std::move(inner.request_id));
                    request_pool_indices.push_back(inner.request_pool_index);
                    input_lengths.push_back(inner.input_length);
                    prefill_lengths.push_back(inner.prefill_length);
                    for (auto& [gid, pages] : inner.block_tables) {
                        block_tables[gid];
                    }
                },
                op);
            if (auto* prefill = std::get_if<PrefillOperation>(&op)) {
                input_ids.insert(input_ids.end(), prefill->input_ids.begin(), prefill->input_ids.end());
                shifted_input_ids.insert(shifted_input_ids.end(), prefill->shifted_input_ids.begin(),
                                         prefill->shifted_input_ids.end());
                extend_prefix_lens.push_back(prefill->extend_prefix_len);
            } else if (auto* decode = std::get_if<DecodeOperation>(&op)) {
                decode_input_ids.push_back(decode->decode_input_id);
                spec_candidate_ids.push_back(std::move(decode->spec_candidate_ids));
            }
        }
        const std::size_t num_reqs = request_ids.size();
        for (auto& [_, table] : block_tables) {
            table.assign(num_reqs, std::vector<std::int32_t>{});
        }
        std::size_t row = 0;
        for (auto& op : ops) {
            std::visit(
                [&](auto& inner) {
                    for (auto& [gid, pages] : inner.block_tables) {
                        block_tables[gid][row] = std::move(pages);
                    }
                },
                op);
            ++row;
        }
        for (auto& [gid, table] : block_tables) {
            const std::size_t rows = table.size();
            std::size_t columns = 0;
            for (const auto& request_table : table) {
                columns = std::max(columns, request_table.size());
            }
            auto& contiguous = block_tables_contig[gid];
            contiguous.reserve(rows * columns);
            for (auto& request_table : table) {
                request_table.resize(columns, -1);
                contiguous.insert(contiguous.end(), request_table.begin(), request_table.end());
            }
        }
    }

    bool empty() const { return request_ids.empty(); }
    std::size_t NumExtends() const { return extend_prefix_lens.size(); }
};

}  // namespace tokenspeed
