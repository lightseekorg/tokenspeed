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
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "cache/allocator/group_allocator.h"
#include "core/profile_span.h"

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

// Like block tables, token windows borrow their request only until this
// Plan builds ForwardBatch. Advance cannot mutate tokens during that build.
// The batch copies both windows into its own storage before returning.
struct PrefillOperation : public ForwardOperationBase {
    std::span<const std::int32_t> input_ids;
    // The available shifted prompt window; the batch pads its missing suffix
    // with -1 instead of reading generated tokens beyond PrefillSize().
    std::span<const std::int32_t> shifted_input_ids;
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

// Immutable row windows owned by the batch. The full absolute table is only
// materialized for a consumer that needs it, and its buffer never moves after
// publication. Copies share the snapshot and its thread-safe materialization.
class ForwardBlockTableRows {
public:
    ForwardBlockTableRows(std::size_t columns, std::vector<std::int32_t> spans, std::vector<std::int32_t> windows)
        : data_(std::make_shared<Data>(columns, std::move(spans), std::move(windows))) {
        _assert(data_->spans.size() % 2 == 0, "block-table spans must contain pairs");
        _assert(Rows() == 0 ? data_->windows.empty() : data_->windows.size() % Rows() == 0,
                "block-table windows must contain complete rows");
        _assert(WindowColumns() <= Columns(), "block-table window exceeds its table");
        for (std::size_t row = 0; row < Rows(); ++row) {
            const auto released = data_->spans[2 * row];
            const auto pages = data_->spans[2 * row + 1];
            _assert(0 <= released && released <= pages && static_cast<std::size_t>(pages) <= Columns(),
                    "block-table span exceeds its table");
            _assert(static_cast<std::size_t>(pages - released) <= WindowColumns(),
                    "block-table window must cover every live entry");
        }
    }

    std::size_t Rows() const { return data_->spans.size() / 2; }
    std::size_t Columns() const { return data_->columns; }
    std::size_t WindowColumns() const { return Rows() == 0 ? 0 : data_->windows.size() / Rows(); }
    const std::vector<std::int32_t>& Spans() const { return data_->spans; }
    const std::vector<std::int32_t>& WindowRows() const { return data_->windows; }

    const std::vector<std::int32_t>& AbsoluteRows() const {
        if (WindowColumns() == Columns()) return data_->windows;
        std::call_once(data_->materialize, [&] {
            // call_once may retry after an allocation failure.
            data_->absolute.clear();
            data_->absolute.reserve(Rows() * Columns());
            for (std::size_t row = 0; row < Rows(); ++row) {
                const auto pages = static_cast<std::size_t>(data_->spans[2 * row + 1]);
                const auto first = pages > WindowColumns() ? pages - WindowColumns() : 0;
                data_->absolute.insert(data_->absolute.end(), first, 0);
                if (WindowColumns() != 0) {
                    const auto begin = data_->windows.begin() + row * WindowColumns();
                    data_->absolute.insert(data_->absolute.end(), begin, begin + WindowColumns());
                }
                data_->absolute.insert(data_->absolute.end(), Columns() - first - WindowColumns(), -1);
            }
        });
        return data_->absolute;
    }

    bool operator==(const ForwardBlockTableRows& other) const {
        return data_ == other.data_ ||
               (Columns() == other.Columns() && Spans() == other.Spans() && WindowRows() == other.WindowRows());
    }

private:
    struct Data {
        Data(std::size_t columns, std::vector<std::int32_t> spans, std::vector<std::int32_t> windows)
            : columns(columns), spans(std::move(spans)), windows(std::move(windows)) {}

        std::size_t columns;
        std::vector<std::int32_t> spans;
        std::vector<std::int32_t> windows;
        mutable std::once_flag materialize;
        mutable std::vector<std::int32_t> absolute;
    };
    std::shared_ptr<const Data> data_;
};

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

    // Logical indexing stays absolute: released prefixes are zero and row
    // padding is -1. The owned transport windows omit the long zero prefixes.
    std::map<std::string, ForwardBlockTableRows> block_tables;
    explicit ForwardBatch(std::vector<ForwardOperation> ops) {
        // Flattens every row's whole block table, which at long contexts
        // costs a plan more than choosing its batch.
        ProfileSpan profile_span{"ForwardBatch::ForwardBatch"};
        std::stable_partition(ops.begin(), ops.end(),
                              [](const ForwardOperation& a) { return std::holds_alternative<PrefillOperation>(a); });
        std::size_t prefill_rows = 0;
        std::size_t input_tokens = 0;
        std::size_t shifted_tokens = 0;
        for (const auto& op : ops) {
            if (const auto* prefill = std::get_if<PrefillOperation>(&op)) {
                ++prefill_rows;
                input_tokens += prefill->input_ids.size();
                shifted_tokens += prefill->input_ids.size();
            }
        }
        request_ids.reserve(ops.size());
        request_pool_indices.reserve(ops.size());
        input_lengths.reserve(ops.size());
        prefill_lengths.reserve(ops.size());
        input_ids.reserve(input_tokens);
        shifted_input_ids.reserve(shifted_tokens);
        extend_prefix_lens.reserve(prefill_rows);
        extend_replay_lens.reserve(prefill_rows);
        state_prefix_cached.reserve(prefill_rows);
        extend_state_checkpoint_lens.reserve(prefill_rows);
        decode_input_ids.reserve(ops.size() - prefill_rows);
        spec_candidate_ids.reserve(ops.size() - prefill_rows);
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
                _assert(prefill->shifted_input_ids.size() <= prefill->input_ids.size(),
                        "shifted prompt window exceeds its input window");
                shifted_input_ids.insert(shifted_input_ids.end(),
                                         prefill->input_ids.size() - prefill->shifted_input_ids.size(), -1);
                extend_prefix_lens.push_back(prefill->extend_prefix_len);
                extend_replay_lens.push_back(prefill->extend_replay_len);
                state_prefix_cached.push_back(prefill->state_prefix_cached);
                extend_state_checkpoint_lens.push_back(prefill->extend_state_checkpoint_len);
            } else if (auto* decode = std::get_if<DecodeOperation>(&op)) {
                decode_input_ids.push_back(decode->decode_input_id);
                spec_candidate_ids.push_back(std::move(decode->spec_candidate_ids));
            }
        }
        std::map<std::string, std::pair<std::size_t, std::size_t>> columns_by_group;
        for (auto& op : ops) {
            std::visit(
                [&](auto& inner) {
                    for (const auto& [gid, source] : inner.block_tables) {
                        auto& [columns, window_columns] = columns_by_group[gid];
                        const auto pages = source.NumPages();
                        columns = std::max(columns, pages);
                        window_columns = std::max(
                            window_columns,
                            pages - static_cast<std::size_t>(source.allocator.LeadingReleasedBlocks(source.table)));
                    }
                },
                op);
        }
        for (const auto& group_columns : columns_by_group) {
            // Named rather than structured bindings: the lambda below captures them.
            const std::string& gid = group_columns.first;
            const auto [columns, window_columns] = group_columns.second;
            std::vector<std::int32_t> windows;
            windows.reserve(request_ids.size() * window_columns);
            std::vector<std::int32_t> spans;
            spans.reserve(2 * request_ids.size());
            for (const auto& op : ops) {
                std::visit(
                    [&](const auto& inner) {
                        std::int32_t released = 0;
                        std::size_t pages = 0;
                        std::size_t first = 0;
                        if (const auto it = inner.block_tables.find(gid); it != inner.block_tables.end()) {
                            const auto& source = it->second;
                            released = source.allocator.LeadingReleasedBlocks(source.table);
                            pages = source.NumPages();
                            first = pages > window_columns ? pages - window_columns : 0;
                            source.allocator.AppendBlockTablePageIds(source.table, windows, first);
                            if (source.checkpoint_page_id != 0) {
                                windows.push_back(source.checkpoint_page_id);
                            }
                        }
                        windows.insert(windows.end(), window_columns - (pages - first), -1);
                        spans.push_back(released);
                        spans.push_back(static_cast<std::int32_t>(pages));
                    },
                    op);
            }
            block_tables.emplace(gid, ForwardBlockTableRows(columns, std::move(spans), std::move(windows)));
        }
    }

    // The legacy nested representation is materialized only when requested.
    std::map<std::string, std::vector<std::vector<std::int32_t>>> BlockTables() const {
        std::map<std::string, std::vector<std::vector<std::int32_t>>> out;
        for (const auto& [gid, table] : block_tables) {
            const auto& contiguous = table.AbsoluteRows();
            auto& rows = out[gid];
            const std::size_t columns = table.Columns();
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
