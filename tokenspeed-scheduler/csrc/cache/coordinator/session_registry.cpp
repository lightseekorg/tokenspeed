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

#include "cache/coordinator/session_registry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"

namespace tokenspeed {

SessionRegistry::Access::Access(SessionRegistry& registry, std::uint64_t access_epoch, SessionCacheOwner owner)
    : registry_{&registry}, access_epoch_{access_epoch} {
    const bool inserted = registry.access_.emplace(access_epoch, std::move(owner)).second;
    _assert(inserted, "an access epoch registers one session");
}

SessionRegistry::Access::Access(Access&& other) noexcept
    : registry_{std::exchange(other.registry_, nullptr)}, access_epoch_{std::exchange(other.access_epoch_, 0)} {}

SessionRegistry::Access& SessionRegistry::Access::operator=(Access&& other) noexcept {
    if (this != &other) {
        release();
        registry_ = std::exchange(other.registry_, nullptr);
        access_epoch_ = std::exchange(other.access_epoch_, 0);
    }
    return *this;
}

SessionRegistry::Access::~Access() {
    release();
}

void SessionRegistry::Access::release() noexcept {
    if (SessionRegistry* registry = std::exchange(registry_, nullptr)) {
        registry->access_.erase(access_epoch_);
    }
}

SessionCacheOwner SessionRegistry::acquireOwnerTag(const std::string& id, OwnerTags& tags) {
    auto& weak = tags[id];
    if (auto tag = weak.lock(); tag && !tag->ended) {
        tag->demoted = false;
        return tag;
    }
    auto tag = std::make_shared<SessionCacheTag>();
    tag->id = id;
    weak = tag;
    if (++tags_since_sweep_ >= 4096) {
        tags_since_sweep_ = 0;
        std::erase_if(session_tags_, [](const auto& entry) { return entry.second.expired(); });
        std::erase_if(cache_group_tags_, [](const auto& entry) { return entry.second.expired(); });
    }
    return tag;
}

SessionRegistry::DemotedOwners SessionRegistry::demote(const std::vector<std::string>& ids, bool ended,
                                                       OwnerTags& tags) {
    DemotedOwners demoted;
    for (const auto& id : ids) {
        auto it = tags.find(id);
        if (it == tags.end()) continue;
        auto tag = it->second.lock();
        if (!tag) {
            tags.erase(it);
            continue;
        }
        tag->demoted = true;
        tag->ended = tag->ended || ended;
        demoted.tags.insert(tag.get());
        demoted.owners.push_back(std::move(tag));
    }
    return demoted;
}

}  // namespace tokenspeed
