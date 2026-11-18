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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/core/session_cache_tag.h"

namespace tokenspeed {

// The cache owners of the requests the coordinator admits: a session id or a
// cache group id names one owner tag, live while any entry or request holds
// it, and each request's access epoch is registered with its owner for as
// long as the request lives, so every admission, publication and restore that
// epoch dates is charged to that owner (OwnerForAccess). Demoting owners marks
// their tags; the coordinator then demotes the entries they own.
class SessionRegistry {
public:
    // An access epoch's registration with its owner, held by the request the
    // epoch dates: constructing it registers, dropping it unregisters.
    // Move-only; the registry must outlive it.
    class Access {
    public:
        Access() noexcept = default;
        Access(SessionRegistry& registry, std::uint64_t access_epoch, SessionCacheOwner owner);
        Access(const Access&) = delete;
        Access& operator=(const Access&) = delete;
        Access(Access&& other) noexcept;
        Access& operator=(Access&& other) noexcept;
        ~Access();

    private:
        void release() noexcept;

        SessionRegistry* registry_{nullptr};
        std::uint64_t access_epoch_{0};
    };

    // The tags demoted by DemoteSessions or DemoteCacheGroups, and their
    // owners, held alive while the entries they own are demoted.
    struct DemotedOwners {
        std::vector<SessionCacheOwner> owners;
        std::unordered_set<const SessionCacheTag*> tags;
    };

    SessionRegistry() = default;
    // Access handles point at the registry.
    SessionRegistry(const SessionRegistry&) = delete;
    SessionRegistry& operator=(const SessionRegistry&) = delete;
    SessionRegistry(SessionRegistry&&) = delete;
    SessionRegistry& operator=(SessionRegistry&&) = delete;

    // The owner registered under an access epoch; nullptr when none is.
    SessionCacheOwner OwnerForAccess(std::uint64_t access_epoch) const {
        const auto it = access_.find(access_epoch);
        return it == access_.end() ? nullptr : it->second;
    }
    // The live tag of a session or a cache group, a fresh one when the last
    // ended or none exists; a demoted live tag is promoted again.
    SessionCacheOwner AcquireSessionOwner(const std::string& session_id) {
        return acquireOwnerTag(session_id, session_tags_);
    }
    SessionCacheOwner AcquireCacheGroupOwner(const std::string& cache_group_id) {
        return acquireOwnerTag(cache_group_id, cache_group_tags_);
    }
    // Marks the live tags of the given sessions or cache groups demoted, and
    // ended too when `ended`.
    DemotedOwners DemoteSessions(const std::vector<std::string>& ids, bool ended) {
        return demote(ids, ended, session_tags_);
    }
    DemotedOwners DemoteCacheGroups(const std::vector<std::string>& ids, bool ended) {
        return demote(ids, ended, cache_group_tags_);
    }
    // The cache group ids with a tag on record, live or expired and not yet
    // swept (accounting only).
    std::size_t NumCacheGroupTags() const { return cache_group_tags_.size(); }

private:
    using OwnerTags = std::unordered_map<std::string, std::weak_ptr<SessionCacheTag>>;

    SessionCacheOwner acquireOwnerTag(const std::string& id, OwnerTags& tags);
    static DemotedOwners demote(const std::vector<std::string>& ids, bool ended, OwnerTags& tags);

    std::unordered_map<std::uint64_t, SessionCacheOwner> access_;
    OwnerTags session_tags_;
    OwnerTags cache_group_tags_;
    std::size_t tags_since_sweep_{0};
};

// The handle a request holds for its access epoch's registration.
using SessionAccess = SessionRegistry::Access;

}  // namespace tokenspeed
