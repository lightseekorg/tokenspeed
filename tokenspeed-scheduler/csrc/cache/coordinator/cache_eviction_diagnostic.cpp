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

#include "cache/coordinator/cache_eviction_diagnostic.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "cache/core/hex_codec.h"

namespace tokenspeed {
namespace {

void diagnosticError(std::string_view message) noexcept {
    try {
        spdlog::error("[CacheEvictionDiagnostic] {}", message);
    } catch (...) {
        // The output or its error handler can itself fail. Diagnostics must
        // not interrupt eviction, including after a block has been recycled.
    }
}

// One process-wide selection, shared by the request hook and eviction logs.
// The mutex only serves an enabled request-tag selection; no cache objects or
// request lifetimes are retained here.
class Selection {
public:
    Selection() noexcept {
        try {
            const char* hash = std::getenv("TOKENSPEED_CACHE_EVICTION_HASH");
            const char* tag = std::getenv("TOKENSPEED_CACHE_EVICTION_REQUEST_TAG");
            const char* tokens = std::getenv("TOKENSPEED_CACHE_EVICTION_PREFIX_TOKENS");
            if (hash == nullptr && tag == nullptr && tokens == nullptr) return;
            if (hash != nullptr) {
                ContentHash::Digest digest;
                if (tag != nullptr || tokens != nullptr || !ReadLowercaseHex(std::string_view{hash}, digest)) {
                    throw std::invalid_argument(
                        "TOKENSPEED_CACHE_EVICTION_HASH requires one lowercase SHA-256 hash "
                        "and cannot be combined with request selection");
                }
                hash_ = digest;
            } else {
                if (tag == nullptr || tokens == nullptr) {
                    throw std::invalid_argument(
                        "request selection requires TOKENSPEED_CACHE_EVICTION_REQUEST_TAG "
                        "and TOKENSPEED_CACHE_EVICTION_PREFIX_TOKENS");
                }
                request_tag_ = tag;
                if (request_tag_.empty() || !std::all_of(request_tag_.begin(), request_tag_.end(), [](char c) {
                        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
                    })) {
                    throw std::invalid_argument("request tag must contain only letters, digits or underscores");
                }
                const std::string_view value{tokens};
                const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), prefix_tokens_);
                if (error != std::errc{} || end != value.data() + value.size() || prefix_tokens_ <= 0) {
                    throw std::invalid_argument("prefix tokens must be a positive int32 token count");
                }
            }
            enabled_ = true;
        } catch (const std::exception& error) {
            diagnosticError(error.what());
        } catch (...) {
            diagnosticError("failed to configure eviction diagnostics; disabled");
        }
    }

    bool Matches(const ContentHash& hash) noexcept {
        if (!enabled_) return false;
        try {
            const auto* digest = hash.digest();
            if (request_tag_.empty()) return digest != nullptr && hash_ == *digest;
            const std::lock_guard lock{mutex_};
            return !failed_ && digest != nullptr && hash_ == *digest;
        } catch (...) {
            diagnosticError("failed to select an eviction; diagnostic coverage is incomplete");
            return false;
        }
    }

    void Bind(std::string_view request_id, std::span<const std::string> hashes,
              std::int32_t prefix_granularity) noexcept {
        if (!enabled_ || request_tag_.empty()) return;
        // A tag is a whole dash-delimited ID component, including when a
        // gateway appends its own ID. The engine knows no benchmark identities.
        bool tagged = false;
        for (std::size_t start = 0; start < request_id.size();) {
            const auto end = request_id.find('-', start);
            if (request_id.substr(start, end == std::string_view::npos ? end : end - start) == request_tag_) {
                tagged = true;
                break;
            }
            if (end == std::string_view::npos) break;
            start = end + 1;
        }
        if (!tagged) return;
        try {
            bool bound = false;
            ContentHash::Digest digest;
            {
                const std::lock_guard lock{mutex_};
                if (failed_) return;
                if (prefix_granularity <= 0 || prefix_tokens_ % prefix_granularity != 0) {
                    failed_ = true;
                    throw std::invalid_argument("selected prefix tokens must align to the prefix granularity");
                }
                const auto pages = static_cast<std::size_t>(prefix_tokens_ / prefix_granularity);
                if (pages > hashes.size()) return;
                if (!ReadLowercaseHex(hashes[pages - 1], digest)) {
                    failed_ = true;
                    throw std::invalid_argument("selected prompt hash is not lowercase SHA-256");
                }
                if (hash_ && hash_ != digest) {
                    failed_ = true;
                    throw std::invalid_argument(
                        "request tag selected different prefix content; diagnostics disabled; "
                        "use one prefix family and a fresh process per replay");
                }
                bound = !hash_;
                hash_ = digest;
            }
            if (bound) {
                spdlog::info("[CacheEvictionDiagnostic] event=bind request={} tag={} prefix_tokens={} hash={}",
                             request_id, request_tag_, prefix_tokens_,
                             hashes[static_cast<std::size_t>(prefix_tokens_ / prefix_granularity) - 1]);
            }
        } catch (const std::exception& error) {
            diagnosticError(error.what());
        } catch (...) {
            diagnosticError("failed to report request binding; diagnostic coverage is incomplete");
        }
    }

private:
    bool enabled_{false};
    bool failed_{false};
    std::string request_tag_;
    std::int32_t prefix_tokens_{0};
    std::optional<ContentHash::Digest> hash_;
    std::mutex mutex_;
};

Selection& selection() {
    static Selection selected;
    return selected;
}

std::string_view evictionReason(CacheEvictor::DeviceEvictionCounts CacheEvictor::DeviceEvictionStats::*reason) {
    if (reason == &CacheEvictor::DeviceEvictionStats::admission) return "admission";
    if (reason == &CacheEvictor::DeviceEvictionStats::request_reclaim) return "request_reclaim";
    if (reason == &CacheEvictor::DeviceEvictionStats::stranded_suffix) return "stranded_suffix";
    if (reason == &CacheEvictor::DeviceEvictionStats::restore) return "restore";
    if (reason == &CacheEvictor::DeviceEvictionStats::flush) return "flush";
    if (reason == &CacheEvictor::DeviceEvictionStats::tail_cap) return "tail_cap";
    if (reason == &CacheEvictor::DeviceEvictionStats::dead_entry_collection) return "dead_entry_collection";
    if (reason == &CacheEvictor::DeviceEvictionStats::compaction) return "compaction";
    throw std::invalid_argument("unknown Device eviction reason");
}

std::string_view evictionReason(CacheEvictor::HostEvictionCounts CacheEvictor::HostEvictionStats::*reason) {
    if (reason == &CacheEvictor::HostEvictionStats::lru) return "lru";
    if (reason == &CacheEvictor::HostEvictionStats::duplicate) return "duplicate";
    if (reason == &CacheEvictor::HostEvictionStats::superseded) return "superseded";
    if (reason == &CacheEvictor::HostEvictionStats::dead_entry) return "dead_entry";
    if (reason == &CacheEvictor::HostEvictionStats::tail_cap) return "tail_cap";
    if (reason == &CacheEvictor::HostEvictionStats::flush) return "flush";
    if (reason == &CacheEvictor::HostEvictionStats::duplicate_kv) return "duplicate_kv";
    throw std::invalid_argument("unknown Host eviction reason");
}

template <typename Reason>
void logEviction(const CacheKey& key, Reason reason, const PrefixCacheIndex::CachedBlockMetadata& metadata,
                 CacheTier tier, bool other_tier_present,
                 const std::unordered_set<CacheKey, CacheKeyHash>& pending_keys) noexcept {
    if (!selection().Matches(key.content_hash)) return;
    try {
        spdlog::info(
            "[CacheEvictionDiagnostic] event=evict tier={} reason={} namespace_id={} group_id={} hash={} "
            "page_offset={} "
            "demoted={} last_access_epoch={} "
            "device_present={} host_present={} eviction_writeback_pending={}",
            tier == CacheTier::kDevice ? "device" : "host", evictionReason(reason), key.namespace_id, key.group_id,
            key.content_hash.ToString(), key.page_offset, metadata.demoted, metadata.eviction.last_access_epoch,
            tier == CacheTier::kHost && other_tier_present, tier == CacheTier::kDevice && other_tier_present,
            pending_keys.contains(key));
    } catch (...) {
        diagnosticError("failed to record an eviction; diagnostic coverage is incomplete");
    }
}

}  // namespace

void BindCacheEvictionDiagnostic(std::string_view request_id, std::span<const std::string> hashes,
                                 std::int32_t prefix_granularity) noexcept {
    selection().Bind(request_id, hashes, prefix_granularity);
}

void LogCacheEviction(const CacheKey& key,
                      CacheEvictor::DeviceEvictionCounts CacheEvictor::DeviceEvictionStats::*reason,
                      const PrefixCacheIndex::CachedBlockMetadata& metadata, bool host_present,
                      const std::unordered_set<CacheKey, CacheKeyHash>& pending_keys) noexcept {
    logEviction(key, reason, metadata, CacheTier::kDevice, host_present, pending_keys);
}

void LogCacheEviction(const CacheKey& key, CacheEvictor::HostEvictionCounts CacheEvictor::HostEvictionStats::*reason,
                      const PrefixCacheIndex::CachedBlockMetadata& metadata, bool device_present,
                      const std::unordered_set<CacheKey, CacheKeyHash>& pending_keys) noexcept {
    logEviction(key, reason, metadata, CacheTier::kHost, device_present, pending_keys);
}

}  // namespace tokenspeed
