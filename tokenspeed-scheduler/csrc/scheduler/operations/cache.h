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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "core/types.h"

namespace tokenspeed {

struct CacheTransfer {
    GroupId group_id{0};
    std::int32_t source_page{-1};
    std::int32_t destination_page{-1};

    bool operator==(const CacheTransfer&) const = default;
};

struct CacheTransferHash {
    std::size_t operator()(const CacheTransfer& transfer) const {
        std::size_t seed = std::hash<GroupId>{}(transfer.group_id);
        const auto combine = [&seed](std::int32_t value) {
            const std::size_t hash = std::hash<std::int32_t>{}(value);
            seed ^= hash + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        };
        combine(transfer.source_page);
        combine(transfer.destination_page);
        return seed;
    }
};

struct WriteBackOperation {
    cache_op_id op_id{0};
    std::vector<CacheTransfer> transfers;  // DEVICE→HOST.

    WriteBackOperation() = default;
    WriteBackOperation(cache_op_id op_id, std::vector<CacheTransfer> transfers)
        : op_id{op_id}, transfers{std::move(transfers)} {}
};

struct WriteBackBatch {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<GroupId>> group_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;

    explicit WriteBackBatch(const std::vector<WriteBackOperation>& ops) {
        std::unordered_set<CacheTransfer, CacheTransferHash> seen;
        for (const auto& op : ops) {
            std::vector<GroupId> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    operation_groups.push_back(transfer.group_id);
                    operation_sources.push_back(transfer.source_page);
                    operation_destinations.push_back(transfer.destination_page);
                }
            }

            op_ids.push_back(op.op_id);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
        }
    }
};

struct LoadBackOperation {
    cache_op_id op_id{0};
    std::vector<CacheTransfer> transfers;  // HOST→DEVICE.

    LoadBackOperation() = default;
    LoadBackOperation(cache_op_id op_id, std::vector<CacheTransfer> transfers)
        : op_id{op_id}, transfers{std::move(transfers)} {}
};

struct LoadBackBatch {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<GroupId>> group_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;

    explicit LoadBackBatch(const std::vector<LoadBackOperation>& ops) {
        std::unordered_set<CacheTransfer, CacheTransferHash> seen;
        for (const auto& op : ops) {
            std::vector<GroupId> operation_groups;
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    operation_groups.push_back(transfer.group_id);
                    operation_sources.push_back(transfer.source_page);
                    operation_destinations.push_back(transfer.destination_page);
                }
            }

            op_ids.push_back(op.op_id);
            group_ids.push_back(std::move(operation_groups));
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
        }
    }
};

using CacheOperation = std::variant<LoadBackBatch, WriteBackBatch>;

}  // namespace tokenspeed
