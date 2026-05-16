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
#include <functional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "resource/types.h"

namespace tokenspeed {

struct CacheOperationBase {
    cache_op_id op_id = 0;
    std::vector<std::int32_t> src_pages;
    std::vector<std::int32_t> dst_pages;
};

struct PrefetchOperation : public CacheOperationBase {
    std::string request_id;
    std::vector<std::string> rolling_page_hashes;
};
struct BackUpOperation : public CacheOperationBase {
    std::vector<std::string> rolling_page_hashes;
};
struct WriteBackOperation {
    cache_op_id op_id{0};
    std::vector<CacheTransferUnit> transfers;
    bool is_retract{false};
};

struct FlatWriteBackOperation {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    std::vector<std::vector<std::int32_t>> transfer_kinds;
    std::vector<std::vector<std::int32_t>> src_indices;
    std::vector<std::vector<std::int32_t>> dst_indices;
    std::vector<bool> is_retract;

    explicit FlatWriteBackOperation(const std::vector<WriteBackOperation>& ops) {
        struct UnitHash {
            std::size_t operator()(const CacheTransferUnit& unit) const {
                std::size_t h0 = std::hash<std::int32_t>{}(static_cast<std::int32_t>(unit.kind));
                std::size_t h1 = std::hash<std::int32_t>{}(unit.src);
                std::size_t h2 = std::hash<std::int32_t>{}(unit.dst);
                return h0 ^ (h1 << 16) ^ (h2 << 1);
            }
        };
        struct UnitEq {
            bool operator()(const CacheTransferUnit& lhs, const CacheTransferUnit& rhs) const {
                return lhs.kind == rhs.kind && lhs.src == rhs.src && lhs.dst == rhs.dst;
            }
        };
        std::unordered_set<CacheTransferUnit, UnitHash, UnitEq> seen;
        for (const auto& op : ops) {
            std::vector<std::int32_t> src_pages_this_op;
            std::vector<std::int32_t> dst_pages_this_op;
            std::vector<std::int32_t> kinds_this_op;
            std::vector<std::int32_t> src_this_op;
            std::vector<std::int32_t> dst_this_op;
            for (const auto& unit : op.transfers) {
                if (!seen.insert(unit).second) continue;
                kinds_this_op.push_back(static_cast<std::int32_t>(unit.kind));
                src_this_op.push_back(unit.src);
                dst_this_op.push_back(unit.dst);
                if (unit.kind == CacheTransferKind::KV) {
                    src_pages_this_op.push_back(unit.src);
                    dst_pages_this_op.push_back(unit.dst);
                }
            }
            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(src_pages_this_op));
            dst_pages.push_back(std::move(dst_pages_this_op));
            transfer_kinds.push_back(std::move(kinds_this_op));
            src_indices.push_back(std::move(src_this_op));
            dst_indices.push_back(std::move(dst_this_op));
            is_retract.push_back(op.is_retract);
        }
    }
};

struct LoadBackOperation {
    cache_op_id op_id{0};
    std::vector<CacheTransferUnit> transfers;
};

struct FlatLoadBackOperation {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    std::vector<std::vector<std::int32_t>> transfer_kinds;
    std::vector<std::vector<std::int32_t>> src_indices;
    std::vector<std::vector<std::int32_t>> dst_indices;

    explicit FlatLoadBackOperation(const std::vector<LoadBackOperation>& ops) {
        struct UnitHash {
            std::size_t operator()(const CacheTransferUnit& unit) const {
                std::size_t h0 = std::hash<std::int32_t>{}(static_cast<std::int32_t>(unit.kind));
                std::size_t h1 = std::hash<std::int32_t>{}(unit.src);
                std::size_t h2 = std::hash<std::int32_t>{}(unit.dst);
                return h0 ^ (h1 << 16) ^ (h2 << 1);
            }
        };
        struct UnitEq {
            bool operator()(const CacheTransferUnit& lhs, const CacheTransferUnit& rhs) const {
                return lhs.kind == rhs.kind && lhs.src == rhs.src && lhs.dst == rhs.dst;
            }
        };
        std::unordered_set<CacheTransferUnit, UnitHash, UnitEq> seen;
        for (const auto& op : ops) {
            std::vector<std::int32_t> src_pages_this_op;
            std::vector<std::int32_t> dst_pages_this_op;
            std::vector<std::int32_t> kinds_this_op;
            std::vector<std::int32_t> src_this_op;
            std::vector<std::int32_t> dst_this_op;
            for (const auto& unit : op.transfers) {
                if (!seen.insert(unit).second) continue;
                kinds_this_op.push_back(static_cast<std::int32_t>(unit.kind));
                src_this_op.push_back(unit.src);
                dst_this_op.push_back(unit.dst);
                if (unit.kind == CacheTransferKind::KV) {
                    src_pages_this_op.push_back(unit.src);
                    dst_pages_this_op.push_back(unit.dst);
                }
            }
            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(src_pages_this_op));
            dst_pages.push_back(std::move(dst_pages_this_op));
            transfer_kinds.push_back(std::move(kinds_this_op));
            src_indices.push_back(std::move(src_this_op));
            dst_indices.push_back(std::move(dst_this_op));
        }
    }
};

using CacheOperation = std::variant<PrefetchOperation, FlatLoadBackOperation, BackUpOperation, FlatWriteBackOperation>;

}  // namespace tokenspeed
