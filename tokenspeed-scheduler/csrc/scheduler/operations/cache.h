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

struct TransferPair {
    std::int32_t source{-1};
    std::int32_t destination{-1};

    bool operator==(const TransferPair& other) const {
        return source == other.source && destination == other.destination;
    }
};

struct TransferPairHash {
    std::size_t operator()(const TransferPair& pair) const {
        const std::size_t source_hash = std::hash<std::int32_t>{}(pair.source);
        const std::size_t destination_hash = std::hash<std::int32_t>{}(pair.destination);
        return source_hash ^ (destination_hash + 0x9e3779b9 + (source_hash << 6) + (source_hash >> 2));
    }
};

struct WriteBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;  // DEVICE→HOST.

    WriteBackOperation() = default;
    WriteBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers)
        : op_id{op_id}, transfers{std::move(transfers)} {}
};

struct WriteBackBatch {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;

    explicit WriteBackBatch(const std::vector<WriteBackOperation>& ops) {
        std::unordered_set<TransferPair, TransferPairHash> seen;
        for (const auto& op : ops) {
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    operation_sources.push_back(transfer.source);
                    operation_destinations.push_back(transfer.destination);
                }
            }

            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
        }
    }
};

struct LoadBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;  // HOST→DEVICE.

    LoadBackOperation() = default;
    LoadBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers)
        : op_id{op_id}, transfers{std::move(transfers)} {}
};

struct LoadBackBatch {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;

    explicit LoadBackBatch(const std::vector<LoadBackOperation>& ops) {
        std::unordered_set<TransferPair, TransferPairHash> seen;
        for (const auto& op : ops) {
            std::vector<std::int32_t> operation_sources;
            std::vector<std::int32_t> operation_destinations;
            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    operation_sources.push_back(transfer.source);
                    operation_destinations.push_back(transfer.destination);
                }
            }

            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(operation_sources));
            dst_pages.push_back(std::move(operation_destinations));
        }
    }
};

using CacheOperation = std::variant<LoadBackBatch, WriteBackBatch>;

}  // namespace tokenspeed
