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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "cache/cache_types.h"

namespace tokenspeed {

// Dispatch-time host POD. This is an input to the executor, never evidence
// that device KV is ready for prefix publication.
struct FlatKVCompletionInput {
    std::uint64_t table_generation{};
    std::uint64_t dispatch_seq{};
    std::int32_t dispatch_raw_start{};
    std::int32_t dispatch_raw_end{};
    std::int32_t protected_raw_end{};
};

struct ForwardOperationBase {
    std::string request_id;
    std::int32_t request_pool_index;
    std::int32_t input_length;
    // All pages currently occupied by this request (existing + newly allocated).
    std::vector<int32_t> occupied_pages;
    // Index into occupied_pages where the emitted page-table refresh begins.
    // On the radix path this may precede newly allocated pages when publishing
    // canonicalizes a duplicate physical page.
    std::int32_t begin;
    // Number of page-table entries to refresh from occupied_pages[begin].
    std::int32_t size;

    std::int32_t prefill_length;

    // Per-request model-defined paged cache pages. For sliding-window groups
    // the vector is COMPACT — it contains only live pages (released-from-front
    // entries are absent). Use paged_cache_page_base_offsets to recover
    // absolute logical-page indexing: column c here = absolute logical page
    // base_offset + c. For full-history groups base_offset is implicitly 0
    // and the key may be omitted from paged_cache_page_base_offsets.
    std::map<std::string, std::vector<std::int32_t>> paged_cache_pages;
    // Per-request, per-sliding-group base logical-page offset.
    std::map<std::string, std::int32_t> paged_cache_page_base_offsets;

    // Production flat path: an allocation-free, index-aligned view of the
    // request's live tables. It is consumed synchronously by
    // FlatForwardOperation before NextExecutionPlan returns; request state is
    // never exposed through the public execution-plan ABI.
    std::span<const BlockTable> flat_block_table_view;
    std::span<const KvCacheGroupSchema> flat_cache_schema;

    // Populated by the flat scheduler after the table mutation is committed.
    // Radix operations leave this null so their batched ABI remains empty.
    std::optional<FlatKVCompletionInput> flat_kv_completion_input;

    // Mamba extension (default: inactive)
    std::int32_t mamba_working_idx{-1};
    std::int32_t mamba_checkpoint_dst_idx{-1};
    std::int32_t mamba_cow_src_idx{-1};
    std::int32_t mamba_branching_seqlen{-1};
};

struct PrefillOperation : public ForwardOperationBase {
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::int32_t extend_prefix_len;
};

struct DecodeOperation : public ForwardOperationBase {
    std::int32_t decode_input_id = -1;
    // For retraction recover
    std::int32_t hist_token_len = -1;
};

using ForwardOperation = std::variant<PrefillOperation, DecodeOperation>;

struct FlatBlockTableExport {
    using LegacyRows = std::vector<std::vector<std::int32_t>>;

    struct CopyResult {
        std::size_t rows{};
        std::size_t cols{};
    };

    // One allocation each at the exact batch rectangle sizes. ``values`` is
    // row-major and already padded with -1; ``bases`` is row-aligned with it.
    std::vector<std::int32_t> values;
    std::vector<std::int32_t> bases;
    std::size_t rows{};
    std::size_t cols{};

    FlatBlockTableExport() = default;

    FlatBlockTableExport(std::size_t row_count, std::size_t col_count)
        : values(CheckedElementCount(row_count, col_count), -1),
          bases(row_count, 0),
          rows(row_count),
          cols(col_count) {}

    std::span<const std::int32_t> Row(std::size_t row) const {
        if (row >= rows) {
            throw std::out_of_range("flat block table export row is out of range");
        }
        AssertShape();
        return std::span<const std::int32_t>{values}.subspan(row * cols, cols);
    }

    std::size_t size() const noexcept { return rows; }
    bool empty() const noexcept { return rows == 0; }

    // Explicit compatibility boundary for the legacy Python property only.
    // The scheduler/binding hot path must use Row()/CopyTo() and never call it.
    LegacyRows MaterializeRows() const {
        LegacyRows out;
        out.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            const std::span<const std::int32_t> source = Row(row);
            out.emplace_back(source.begin(), source.end());
        }
        return out;
    }

    CopyResult CopyTo(std::span<std::int32_t> table_destination, std::span<std::int32_t> base_destination,
                      std::int64_t page_id_upper_bound) const {
        AssertShape();
        if (page_id_upper_bound < 2) {
            throw std::invalid_argument("flat block table page-id upper bound must reserve page 0");
        }
        if (rows > base_destination.size()) {
            throw std::invalid_argument("flat block table staging row capacity exceeded");
        }
        if (values.size() > table_destination.size()) {
            throw std::invalid_argument("flat block table staging column capacity exceeded");
        }

        for (std::size_t row = 0; row < rows; ++row) {
            const std::int32_t base = bases[row];
            if (base < 0) {
                throw std::invalid_argument("flat block table contains a negative logical base");
            }
            base_destination[row] = base;

            const std::size_t row_offset = row * cols;
            for (std::size_t col = 0; col < cols; ++col) {
                const std::int32_t page_id = values[row_offset + col];
                if (page_id < -1 || page_id >= page_id_upper_bound) {
                    throw std::invalid_argument("flat block table contains a page id outside [-1, total_blocks)");
                }
                table_destination[row_offset + col] = page_id;
            }
        }
        return CopyResult{.rows = rows, .cols = cols};
    }

private:
    static std::size_t CheckedElementCount(std::size_t rows, std::size_t cols) {
        if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
            throw std::overflow_error("flat block table export shape overflows size_t");
        }
        return rows * cols;
    }

    void AssertShape() const {
        if (bases.size() != rows || values.size() != CheckedElementCount(rows, cols)) {
            throw std::logic_error("flat block table export payload disagrees with its shape");
        }
    }
};

struct FlatForwardOperation {
    // Pool-set generation observed at dispatch. Python stamps every exported
    // table source with it so a pre-reset operation cannot replay after wake.
    std::uint64_t cache_generation{};
    std::vector<std::string> request_ids;
    std::vector<std::int32_t> request_pool_indices;
    std::vector<std::int32_t> input_lengths;
    // Per-request total number of prompt tokens (Request::PrefillSize()).
    std::vector<std::int32_t> prefill_lengths;

    std::vector<std::vector<std::int32_t>> occupied_pages;
    std::vector<std::int32_t> begins;
    std::vector<std::int32_t> sizes;

    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::vector<std::int32_t> extend_prefix_lens;
    std::vector<std::int32_t> decode_input_ids;
    std::vector<std::int32_t> hist_token_lens;

    // Mamba extension (SoA)
    std::vector<std::int32_t> mamba_working_indices;
    std::vector<std::int32_t> mamba_checkpoint_dst_indices;
    std::vector<std::int32_t> mamba_cow_src_indices;
    std::vector<std::int32_t> mamba_branching_seqlens;

    // Per-group paged cache block tables: dict[group_id] = [num_reqs,
    // max_live_pages_for_group_in_this_batch] padded with -1. For sliding
    // groups each row is COMPACT (released-from-front pages are absent);
    // pair with paged_cache_block_table_base_offsets to recover absolute
    // logical-page indexing. For full-history groups rows are absolute and
    // the offset is implicitly 0 (key omitted from the offsets map).
    std::map<std::string, std::vector<std::vector<std::int32_t>>> paged_cache_block_tables;
    // Per-group [num_reqs] base logical-page offsets, only present for
    // sliding-window groups. Missing key ⇔ offset is 0 for every row.
    std::map<std::string, std::vector<std::int32_t>> paged_cache_block_table_base_offsets;

    // Per-group contiguous export owner. Every value owns one exact row-major
    // [rows, cols] rectangle and its [rows] bases; no nested row vectors survive
    // batching. Python exposes this buffer through the mainline zero-copy
    // ndarray API; legacy properties materialize nested containers lazily.
    std::map<std::string, FlatBlockTableExport> flat_block_tables;

    // Row-aligned with request_ids. Empty on radix; on flat, every KV-writing
    // row has exactly one ready-completion seed.
    std::vector<FlatKVCompletionInput> flat_kv_completion_inputs;

    explicit FlatForwardOperation(std::vector<ForwardOperation> ops) {
        std::span<const KvCacheGroupSchema> canonical_flat_schema;
        std::vector<std::size_t> flat_widths;
        bool have_flat_schema = false;
        bool miss_flat_schema = false;
        bool have_completion_inputs = false;
        bool miss_completion_input = false;
        std::size_t total_input_ids = 0;
        std::size_t total_shifted_input_ids = 0;
        std::size_t num_prefills = 0;
        std::size_t num_decodes = 0;
        for (const auto& op : ops) {
            std::visit(
                [&](const auto& inner) {
                    const bool uses_live_view =
                        !inner.flat_block_table_view.empty() || !inner.flat_cache_schema.empty();
                    if (uses_live_view) {
                        // Every production row borrows the coordinator-owned
                        // schema and is index-aligned with its request tables.
                        if (inner.flat_block_table_view.size() != inner.flat_cache_schema.size()) {
                            std::terminate();
                        }
                        if (!have_flat_schema) {
                            canonical_flat_schema = inner.flat_cache_schema;
                            flat_widths.reserve(canonical_flat_schema.size());
                            for (std::size_t i = 0; i < inner.flat_block_table_view.size(); ++i) {
                                flat_widths.push_back(
                                    static_cast<std::size_t>(inner.flat_block_table_view[i].NumBlocks()));
                            }
                            have_flat_schema = true;
                        } else {
                            if (inner.flat_cache_schema.data() != canonical_flat_schema.data() ||
                                inner.flat_cache_schema.size() != canonical_flat_schema.size()) {
                                std::terminate();
                            }
                            for (std::size_t i = 0; i < inner.flat_block_table_view.size(); ++i) {
                                flat_widths[i] =
                                    std::max(flat_widths[i],
                                             static_cast<std::size_t>(inner.flat_block_table_view[i].NumBlocks()));
                            }
                        }
                    } else {
                        miss_flat_schema = true;
                    }
                    for (const auto& [group_id, _] : inner.paged_cache_pages) {
                        paged_cache_block_tables.try_emplace(group_id);
                    }
                    for (const auto& [group_id, _] : inner.paged_cache_page_base_offsets) {
                        paged_cache_block_table_base_offsets.try_emplace(group_id);
                    }
                    have_completion_inputs = have_completion_inputs || inner.flat_kv_completion_input.has_value();
                    miss_completion_input = miss_completion_input || !inner.flat_kv_completion_input.has_value();
                    if constexpr (std::same_as<std::decay_t<decltype(inner)>, PrefillOperation>) {
                        ++num_prefills;
                        total_input_ids += inner.input_ids.size();
                        total_shifted_input_ids += inner.shifted_input_ids.size();
                    } else {
                        ++num_decodes;
                    }
                },
                op);
        }
        if (have_flat_schema && miss_flat_schema) {
            std::terminate();
        }
        if (have_completion_inputs && miss_completion_input) {
            throw std::invalid_argument(
                "FlatForwardOperation: completion inputs must be present for every request row or none");
        }
        const std::size_t num_reqs = ops.size();
        request_ids.reserve(num_reqs);
        request_pool_indices.reserve(num_reqs);
        input_lengths.reserve(num_reqs);
        prefill_lengths.reserve(num_reqs);
        occupied_pages.reserve(num_reqs);
        begins.reserve(num_reqs);
        sizes.reserve(num_reqs);
        mamba_working_indices.reserve(num_reqs);
        mamba_checkpoint_dst_indices.reserve(num_reqs);
        mamba_cow_src_indices.reserve(num_reqs);
        mamba_branching_seqlens.reserve(num_reqs);
        flat_kv_completion_inputs.reserve(have_completion_inputs ? num_reqs : 0);
        input_ids.reserve(total_input_ids);
        shifted_input_ids.reserve(total_shifted_input_ids);
        extend_prefix_lens.reserve(num_prefills);
        decode_input_ids.reserve(num_decodes);
        hist_token_lens.reserve(num_decodes);
        std::vector<FlatBlockTableExport*> flat_exports;
        flat_exports.reserve(flat_widths.size());
        for (std::size_t i = 0; i < flat_widths.size(); ++i) {
            const auto [export_it, inserted] =
                flat_block_tables.try_emplace(canonical_flat_schema[i].group_id, num_reqs, flat_widths[i]);
            if (!inserted) {
                std::terminate();
            }
            // std::map preserves references across insertion. Cache the owners
            // in canonical group order so row export does no tree/string work.
            flat_exports.push_back(&export_it->second);
        }
        for (auto& [_, table] : paged_cache_block_tables) {
            table.assign(num_reqs, std::vector<std::int32_t>{});
        }
        for (auto& [_, offsets] : paged_cache_block_table_base_offsets) {
            offsets.assign(num_reqs, 0);
        }

        auto append_row = [this, &flat_exports](ForwardOperation& op, std::size_t row) {
            std::visit(
                [this, row, &flat_exports](auto& inner) {
                    request_ids.push_back(std::move(inner.request_id));
                    request_pool_indices.push_back(inner.request_pool_index);
                    input_lengths.push_back(inner.input_length);
                    prefill_lengths.push_back(inner.prefill_length);
                    occupied_pages.push_back(std::move(inner.occupied_pages));
                    begins.push_back(inner.begin);
                    sizes.push_back(inner.size);
                    mamba_working_indices.push_back(inner.mamba_working_idx);
                    mamba_checkpoint_dst_indices.push_back(inner.mamba_checkpoint_dst_idx);
                    mamba_cow_src_indices.push_back(inner.mamba_cow_src_idx);
                    mamba_branching_seqlens.push_back(inner.mamba_branching_seqlen);
                    for (std::size_t i = 0; i < inner.flat_block_table_view.size(); ++i) {
                        FlatBlockTableExport& export_owner = *flat_exports[i];
                        const BlockTable& table = inner.flat_block_table_view[i];
                        if (static_cast<std::size_t>(table.NumBlocks()) > export_owner.cols) {
                            std::terminate();
                        }
                        std::size_t destination = row * export_owner.cols;
                        for (const BlockRef& block : table.Blocks()) {
                            export_owner.values[destination++] = block ? block->BlockId() : 0;
                        }
                        export_owner.bases[row] = table.BaseLogicalPage();
                    }
                    for (auto& [gid, pages] : inner.paged_cache_pages) {
                        paged_cache_block_tables.at(gid)[row] = std::move(pages);
                    }
                    for (const auto& [gid, offset] : inner.paged_cache_page_base_offsets) {
                        paged_cache_block_table_base_offsets.at(gid)[row] = offset;
                    }
                    if (inner.flat_kv_completion_input.has_value()) {
                        flat_kv_completion_inputs.push_back(std::move(*inner.flat_kv_completion_input));
                    }
                    if constexpr (std::same_as<std::decay_t<decltype(inner)>, PrefillOperation>) {
                        input_ids.insert(input_ids.end(), inner.input_ids.begin(), inner.input_ids.end());
                        shifted_input_ids.insert(shifted_input_ids.end(), inner.shifted_input_ids.begin(),
                                                 inner.shifted_input_ids.end());
                        extend_prefix_lens.push_back(inner.extend_prefix_len);
                    } else {
                        decode_input_ids.push_back(inner.decode_input_id);
                        hist_token_lens.push_back(inner.hist_token_len);
                    }
                },
                op);
        };

        // Preserve stable prefill-before-decode order without stable_partition:
        // moving the heavy variants would allocate a temporary buffer and walk
        // every row before the real SoA/table export.
        std::size_t row = 0;
        for (auto& op : ops) {
            if (std::holds_alternative<PrefillOperation>(op)) {
                append_row(op, row++);
            }
        }
        for (auto& op : ops) {
            if (std::holds_alternative<DecodeOperation>(op)) {
                append_row(op, row++);
            }
        }
        if (row != num_reqs) {
            std::terminate();
        }
        padRectangularMinusOne(paged_cache_block_tables);
    }

    bool empty() const { return request_ids.empty(); }
    std::size_t num_extends() const { return extend_prefix_lens.size(); }

    std::map<std::string, std::vector<std::vector<std::int32_t>>> MaterializeFlatBlockTables() const {
        std::map<std::string, std::vector<std::vector<std::int32_t>>> out;
        for (const auto& [group_id, export_owner] : flat_block_tables) {
            out.emplace(group_id, export_owner.MaterializeRows());
        }
        return out;
    }

    std::map<std::string, std::vector<std::int32_t>> MaterializeFlatBlockTableBaseOffsets() const {
        std::map<std::string, std::vector<std::int32_t>> out;
        for (const auto& [group_id, export_owner] : flat_block_tables) {
            out.emplace(group_id, export_owner.bases);
        }
        return out;
    }

private:
    template <typename Key>
    static void padRectangularMinusOne(std::map<Key, std::vector<std::vector<std::int32_t>>>& tables) {
        for (auto& [_, table] : tables) {
            std::int32_t max_cols = 0;
            for (const auto& row : table) {
                max_cols = std::max(max_cols, static_cast<std::int32_t>(row.size()));
            }
            for (auto& row : table) {
                row.resize(max_cols, -1);
            }
        }
    }
};

}  // namespace tokenspeed
