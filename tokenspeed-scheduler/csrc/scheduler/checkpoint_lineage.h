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
#include <deque>
#include <unordered_map>

#include "cache/core/cache_types.h"

namespace tokenspeed {

// What superseded checkpoint demotion reads
// (SchedulerConfig::demote_superseded_state_checkpoints,
// Scheduler::demoteSupersededCheckpoint): for every in-forward State
// checkpoint a finishing request published past the point it resumed from,
// that point and the request's access epoch; and how many first admissions
// resumed at each page boundary. Both are keyed by the boundary's chain hash
// and bounded: an entry with kMaxEntries newer ones belongs to a long finished
// conversation, and a forgotten boundary reads as one never seen, which no
// demotion takes.
class CheckpointLineage {
public:
    static constexpr std::size_t kMaxEntries = std::size_t{1} << 16;

    struct Publication {
        // The page boundary the publishing request resumed from, in tokens;
        // 0 when it resumed at none.
        std::int32_t resumed_tokens{0};
        // The publishing request's access epoch: its admission acquired the
        // resumed boundary with it.
        std::uint64_t access_epoch{0};
    };

    // A checkpoint published again takes the newer publication and keeps its
    // place in the bound.
    void RecordPublication(const ContentHash& checkpoint, Publication publication) {
        publications_.FindOrInsert(checkpoint, publication) = publication;
    }
    const Publication* FindPublication(const ContentHash& checkpoint) const { return publications_.Find(checkpoint); }

    // A first admission resumed at this page boundary. Only one or more than
    // one matters, so the count stops at 2.
    void NoteResume(const ContentHash& boundary) {
        std::int32_t& resumes = resumes_.FindOrInsert(boundary, 0);
        resumes = std::min(resumes + 1, 2);
    }
    // The first admissions that resumed there: 0 (none, or forgotten), 1, or 2
    // for more than one.
    std::int32_t Resumes(const ContentHash& boundary) const {
        const std::int32_t* resumes = resumes_.Find(boundary);
        return resumes == nullptr ? 0 : *resumes;
    }

private:
    // A map that forgets its oldest key once it holds more than kMaxEntries.
    template <typename Value>
    class BoundedMap {
    public:
        Value& FindOrInsert(const ContentHash& key, const Value& initial) {
            const auto [it, inserted] = entries_.try_emplace(key, initial);
            if (inserted) {
                order_.push_back(&it->first);
                if (order_.size() > kMaxEntries) {
                    entries_.erase(entries_.find(*order_.front()));
                    order_.pop_front();
                }
            }
            return it->second;
        }
        const Value* Find(const ContentHash& key) const {
            const auto it = entries_.find(key);
            return it == entries_.end() ? nullptr : &it->second;
        }

    private:
        std::unordered_map<ContentHash, Value> entries_;
        // The keys in insertion order, pointing into entries_, whose nodes
        // stay put until the front's is erased.
        std::deque<const ContentHash*> order_;
    };

    BoundedMap<Publication> publications_;
    BoundedMap<std::int32_t> resumes_;
};

}  // namespace tokenspeed
