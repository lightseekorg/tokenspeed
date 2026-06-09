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
#include <map>
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
enum class CacheKind : std::int32_t { kKV = 0, kMamba = 1 };

inline const char* CacheKindName(CacheKind kind) {
    switch (kind) {
        case CacheKind::kKV:
            return "kv";
        case CacheKind::kMamba:
            return "mamba";
    }
    return "unknown";
}

struct TransferPair {
    CacheKind kind{CacheKind::kKV};
    std::int32_t src{-1};
    std::int32_t dst{-1};

    bool operator==(const TransferPair& other) const {
        return kind == other.kind && src == other.src && dst == other.dst;
    }
};

inline std::vector<TransferPair> ToTransferPairs(CacheKind kind,
                                                 const std::vector<std::tuple<std::int32_t, std::int32_t>>& pages) {
    std::vector<TransferPair> transfers;
    transfers.reserve(pages.size());
    for (const auto& page : pages) {
        transfers.push_back(TransferPair{kind, std::get<0>(page), std::get<1>(page)});
    }
    return transfers;
}

struct TransferPairHash {
    std::size_t operator()(const TransferPair& pair) const {
        std::size_t h0 = std::hash<std::int32_t>{}(static_cast<std::int32_t>(pair.kind));
        std::size_t h1 = std::hash<std::int32_t>{}(pair.src);
        std::size_t h2 = std::hash<std::int32_t>{}(pair.dst);
        return h0 ^ (h1 << 1) ^ (h2 << 32) ^ (h2 >> 32);
    }
};

struct PagedCacheTransferPair {
    std::string group_id;
    std::vector<std::int32_t> src_pages;
    std::vector<std::int32_t> dst_pages;

    bool operator==(const PagedCacheTransferPair& other) const {
        return group_id == other.group_id && src_pages == other.src_pages && dst_pages == other.dst_pages;
    }
};

struct WriteBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;                        // DEVICE→HOST by cache kind.
    std::vector<PagedCacheTransferPair> paged_cache_transfers;  // DEVICE→HOST by paged-cache group.
    bool is_retract{false};

    WriteBackOperation() = default;
    WriteBackOperation(cache_op_id op_id, std::vector<std::tuple<std::int32_t, std::int32_t>> pages_to_transfer,
                       bool is_retract = false)
        : op_id{op_id}, transfers{ToTransferPairs(CacheKind::kKV, pages_to_transfer)}, is_retract{is_retract} {}
    WriteBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers, bool is_retract = false)
        : op_id{op_id}, transfers{std::move(transfers)}, is_retract{is_retract} {}
    WriteBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers,
                       std::vector<PagedCacheTransferPair> paged_cache_transfers, bool is_retract = false)
        : op_id{op_id},
          transfers{std::move(transfers)},
          paged_cache_transfers{std::move(paged_cache_transfers)},
          is_retract{is_retract} {}
};

struct FlatWriteBackOperation {
    std::vector<cache_op_id> op_ids;
    // Backward-compatible KV-only view.
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    // Generic view keyed by CacheKindName(kind), currently "kv" and "mamba".
    std::map<std::string, std::vector<std::vector<std::int32_t>>> src_pages_by_kind;
    std::map<std::string, std::vector<std::vector<std::int32_t>>> dst_pages_by_kind;
    // Indexed by op_ids; each item preserves the op's paged-cache group transfer specs.
    std::vector<std::vector<PagedCacheTransferPair>> paged_cache_transfers;
    std::vector<bool> is_retract;

    explicit FlatWriteBackOperation(const std::vector<WriteBackOperation>& ops) {
        std::unordered_set<TransferPair, TransferPairHash> seen;
        for (const auto& op : ops) {
            std::map<std::string, std::vector<std::int32_t>> src_this_op;
            std::map<std::string, std::vector<std::int32_t>> dst_this_op;
            src_this_op[CacheKindName(CacheKind::kKV)];
            dst_this_op[CacheKindName(CacheKind::kKV)];
            src_this_op[CacheKindName(CacheKind::kMamba)];
            dst_this_op[CacheKindName(CacheKind::kMamba)];

            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    const std::string kind_name = CacheKindName(transfer.kind);
                    src_this_op[kind_name].push_back(transfer.src);
                    dst_this_op[kind_name].push_back(transfer.dst);
                }
            }

            op_ids.push_back(op.op_id);
            src_pages.push_back(src_this_op[CacheKindName(CacheKind::kKV)]);
            dst_pages.push_back(dst_this_op[CacheKindName(CacheKind::kKV)]);
            for (auto& [kind, pages] : src_this_op) {
                src_pages_by_kind[kind].push_back(std::move(pages));
            }
            for (auto& [kind, pages] : dst_this_op) {
                dst_pages_by_kind[kind].push_back(std::move(pages));
            }
            paged_cache_transfers.push_back(op.paged_cache_transfers);
            is_retract.push_back(op.is_retract);
        }
    }
};

struct LoadBackOperation {
    cache_op_id op_id{0};
    std::vector<TransferPair> transfers;                        // HOST→DEVICE by cache kind.
    std::vector<PagedCacheTransferPair> paged_cache_transfers;  // HOST→DEVICE by paged-cache group.

    LoadBackOperation() = default;
    LoadBackOperation(cache_op_id op_id, std::vector<std::tuple<std::int32_t, std::int32_t>> pages_to_transfer)
        : op_id{op_id}, transfers{ToTransferPairs(CacheKind::kKV, pages_to_transfer)} {}
    LoadBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers)
        : op_id{op_id}, transfers{std::move(transfers)} {}
    LoadBackOperation(cache_op_id op_id, std::vector<TransferPair> transfers,
                      std::vector<PagedCacheTransferPair> paged_cache_transfers)
        : op_id{op_id}, transfers{std::move(transfers)}, paged_cache_transfers{std::move(paged_cache_transfers)} {}
};

struct FlatLoadBackOperation {
    std::vector<cache_op_id> op_ids;
    // Backward-compatible KV-only view.
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    // Generic view keyed by CacheKindName(kind), currently "kv" and "mamba".
    std::map<std::string, std::vector<std::vector<std::int32_t>>> src_pages_by_kind;
    std::map<std::string, std::vector<std::vector<std::int32_t>>> dst_pages_by_kind;
    // Indexed by op_ids; each item preserves the op's paged-cache group transfer specs.
    std::vector<std::vector<PagedCacheTransferPair>> paged_cache_transfers;

    explicit FlatLoadBackOperation(const std::vector<LoadBackOperation>& ops) {
        std::unordered_set<TransferPair, TransferPairHash> seen;
        for (const auto& op : ops) {
            std::map<std::string, std::vector<std::int32_t>> src_this_op;
            std::map<std::string, std::vector<std::int32_t>> dst_this_op;
            src_this_op[CacheKindName(CacheKind::kKV)];
            dst_this_op[CacheKindName(CacheKind::kKV)];
            src_this_op[CacheKindName(CacheKind::kMamba)];
            dst_this_op[CacheKindName(CacheKind::kMamba)];

            for (const auto& transfer : op.transfers) {
                if (seen.insert(transfer).second) {
                    const std::string kind_name = CacheKindName(transfer.kind);
                    src_this_op[kind_name].push_back(transfer.src);
                    dst_this_op[kind_name].push_back(transfer.dst);
                }
            }

            op_ids.push_back(op.op_id);
            src_pages.push_back(src_this_op[CacheKindName(CacheKind::kKV)]);
            dst_pages.push_back(dst_this_op[CacheKindName(CacheKind::kKV)]);
            for (auto& [kind, pages] : src_this_op) {
                src_pages_by_kind[kind].push_back(std::move(pages));
            }
            for (auto& [kind, pages] : dst_this_op) {
                dst_pages_by_kind[kind].push_back(std::move(pages));
            }
            paged_cache_transfers.push_back(op.paged_cache_transfers);
        }
    }
};

using CacheOperation = std::variant<PrefetchOperation, FlatLoadBackOperation, BackUpOperation, FlatWriteBackOperation>;

}  // namespace tokenspeed
