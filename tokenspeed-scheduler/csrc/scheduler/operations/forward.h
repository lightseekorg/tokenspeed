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

#include "cache/allocator/group_allocator.h"

namespace tokenspeed {

// Borrowed only while the scheduler builds one batch. A scheduled request is
// not retracted or scheduled again in that build, so its table stays stable.
// ForwardBatch resolves these rows before any source can change; no borrow
// escapes into the execution plan.
struct ForwardBlockTable {
    const GroupAllocator& allocator;
    const BlockTable& table;
    std::int32_t checkpoint_page_id{0};

    std::size_t NumPages() const { return static_cast<std::size_t>(table.NumBlocks()) + (checkpoint_page_id != 0); }
};

struct ForwardOperationBase {
    std::string request_id;
    std::int32_t request_pool_index{-1};
    std::int32_t input_length{0};
    std::int32_t prefill_length{0};

    // Per-group block tables. A History group's row uses absolute
    // logical-page indexing; null holes are page 0 and rows are not
    // compacted. A State group's row is its two slots [input, live]: a
    // prefill row reads slot 0 (0 = the zero state) and writes slot 1; a
    // decode row reads slot 0 when it is set, else slot 1, and writes slot 1.
    // A prefill that writes an in-forward State checkpoint appends a third
    // entry, the block that receives the state at its checkpoint.
    std::map<std::string, ForwardBlockTable> block_tables;
};

struct PrefillOperation : public ForwardOperationBase {
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    // The prompt positions already cached: the hit boundary, any token count.
    std::int32_t extend_prefix_len{0};
    // The D role's remote admission only: every State group's checkpoint at
    // exactly the prompt end was found locally and sits in slot 0, so the
    // peer omits the latest-snapshot transfer. False everywhere else.
    bool state_prefix_cached{false};
    // Leading input rows that re-feed already computed prompt tokens
    // (bounded replay). Positions [extend_prefix_len, extend_prefix_len +
    // extend_replay_len) regenerate replayable cache groups only; every
    // other group already holds their rows and must not be rewritten.
    std::int32_t extend_replay_len{0};
    // Tokens of this chunk before its in-forward State checkpoint
    // (SchedulerConfig::state_checkpoint_in_forward), 0 for none: every State
    // group writes the state after position extend_prefix_len +
    // extend_state_checkpoint_len into its row's third entry, besides the
    // final state into slot 1.
    std::int32_t extend_state_checkpoint_len{0};
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
    // Parallel to extend_prefix_lens: PrefillOperation::extend_replay_len.
    std::vector<std::int32_t> extend_replay_lens;
    // Parallel to extend_prefix_lens: PrefillOperation::state_prefix_cached.
    std::vector<bool> state_prefix_cached;
    // Parallel to extend_prefix_lens: PrefillOperation::extend_state_checkpoint_len.
    std::vector<std::int32_t> extend_state_checkpoint_lens;
    std::vector<std::int32_t> decode_input_ids;
    // Parallel to decode_input_ids (one entry per decode row); rows without
    // candidates hold an empty vector.
    std::vector<std::vector<std::int32_t>> spec_candidate_ids;

    // Per-group block tables: dict[group_id] =
    // [num_reqs, max_pages_in_batch] padded with -1. Each row is absolute
    // (null hole = 0, no compaction); there is no base-offset companion.
    // One owned row-major buffer per group, exposed zero-copy to Python.
    std::map<std::string, std::vector<std::int32_t>> block_tables_contig;
    // Per-group row spans parallel to block_tables_contig: [num_reqs, 2]
    // (released, pages). A row's columns before `released` are the released
    // prefix's null holes and its columns from `pages` on are -1 padding, so
    // a consumer can move [released, pages) and rebuild the rest unread.
    std::map<std::string, std::vector<std::int32_t>> block_table_spans;
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
                },
                op);
            if (auto* prefill = std::get_if<PrefillOperation>(&op)) {
                input_ids.insert(input_ids.end(), prefill->input_ids.begin(), prefill->input_ids.end());
                shifted_input_ids.insert(shifted_input_ids.end(), prefill->shifted_input_ids.begin(),
                                         prefill->shifted_input_ids.end());
                extend_prefix_lens.push_back(prefill->extend_prefix_len);
                extend_replay_lens.push_back(prefill->extend_replay_len);
                state_prefix_cached.push_back(prefill->state_prefix_cached);
                extend_state_checkpoint_lens.push_back(prefill->extend_state_checkpoint_len);
            } else if (auto* decode = std::get_if<DecodeOperation>(&op)) {
                decode_input_ids.push_back(decode->decode_input_id);
                spec_candidate_ids.push_back(std::move(decode->spec_candidate_ids));
            }
        }
        std::map<std::string, std::size_t> columns_by_group;
        for (auto& op : ops) {
            std::visit(
                [&](auto& inner) {
                    for (const auto& [gid, source] : inner.block_tables) {
                        auto& columns = columns_by_group[gid];
                        columns = std::max(columns, source.NumPages());
                    }
                },
                op);
        }
        for (const auto& group_columns : columns_by_group) {
            // Named rather than structured bindings: the lambda below captures them.
            const std::string& gid = group_columns.first;
            const std::size_t columns = group_columns.second;
            auto& contiguous = block_tables_contig[gid];
            contiguous.reserve(request_ids.size() * columns);
            auto& spans = block_table_spans[gid];
            spans.reserve(2 * request_ids.size());
            for (const auto& op : ops) {
                std::visit(
                    [&](const auto& inner) {
                        std::int32_t released = 0;
                        std::size_t pages = 0;
                        if (const auto it = inner.block_tables.find(gid); it != inner.block_tables.end()) {
                            const auto& source = it->second;
                            // The prefix AppendBlockTablePageIds writes as null holes.
                            released = source.table.ReclaimedPrefixBlocks();
                            source.allocator.AppendBlockTablePageIds(source.table, contiguous);
                            if (source.checkpoint_page_id != 0) {
                                contiguous.push_back(source.checkpoint_page_id);
                            }
                            pages = source.NumPages();
                        }
                        contiguous.insert(contiguous.end(), columns - pages, -1);
                        spans.push_back(released);
                        spans.push_back(static_cast<std::int32_t>(pages));
                    },
                    op);
            }
        }
    }

    // The legacy nested representation is materialized only when requested.
    std::map<std::string, std::vector<std::vector<std::int32_t>>> BlockTables() const {
        std::map<std::string, std::vector<std::vector<std::int32_t>>> out;
        for (const auto& [gid, contiguous] : block_tables_contig) {
            auto& rows = out[gid];
            const std::size_t columns = request_ids.empty() ? 0 : contiguous.size() / request_ids.size();
            rows.reserve(request_ids.size());
            for (std::size_t row = 0; row < request_ids.size(); ++row) {
                const auto begin = contiguous.begin() + row * columns;
                rows.emplace_back(begin, begin + columns);
            }
        }
        return out;
    }

    bool empty() const { return request_ids.empty(); }
    std::size_t NumExtends() const { return extend_prefix_lens.size(); }

    bool operator==(const ForwardBatch&) const = default;
};

}  // namespace tokenspeed
