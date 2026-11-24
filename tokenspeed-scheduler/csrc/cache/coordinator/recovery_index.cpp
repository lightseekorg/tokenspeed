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

#include "cache/coordinator/recovery_index.h"

#include "core/profile_span.h"

#include <algorithm>
#include <array>
#include <bit>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/recovery_route.h"

namespace tokenspeed {
namespace {
enum Column : std::size_t { kDevice, kHost, kHostPending, kDisk, kHostDisk, kHostDiskPending, kColumns };
using Counts = std::array<std::int32_t, kColumns>;
std::size_t PhysicalColumn(CacheTier tier) {
    switch (tier) {
        case CacheTier::kDevice:
            return kDevice;
        case CacheTier::kHost:
            return kHost;
        case CacheTier::kDisk:
            return kDisk;
    }
    FatalCheck(false, "unknown recovery tier");
    return kDevice;
}
void UpdateUnions(Counts& values, bool pending) {
    values[kHostPending] = values[kHost] || pending;
    values[kHostDisk] = values[kHost] || values[kDisk];
    values[kHostDiskPending] = values[kHostPending] || values[kDisk];
}
}  // namespace
struct RecoveryIndex::Impl {
    struct RangeKey {
        CacheKey last;
        unsigned level;
        bool operator==(const RangeKey&) const = default;
    };
    struct RangeHash {
        std::size_t operator()(const RangeKey& key) const {
            return CacheKeyHash{}(key.last) ^ (std::size_t{key.level} * 0x9e3779b9U);
        }
    };
    struct Record;
    struct Node {
        Impl& owner;
        RangeKey key;
        std::int32_t first;
        std::int32_t past_end;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        std::optional<CacheKey> previous;
        Counts counts{};
        bool pending{false};
        std::vector<Node*> parents;
        std::vector<Record*> watchers;
        std::list<Node*>::iterator leaf_position{};
        bool listed{false};
        std::uint64_t visited{0};
        Node(Impl& owner, RangeKey key, std::int32_t first, std::int32_t end)
            : owner{owner}, key{key}, first{first}, past_end{end} {}
        ~Node() {
            if (left) {
                std::erase(left->parents, this);
                if (right) std::erase(right->parents, this);
            } else if (listed) {
                owner.leaves.erase(leaf_position);
            }
            owner.ranges.erase(key);
        }
    };
    struct Record {
        ContentHash hash;
        std::int32_t tokens;
        std::vector<std::vector<std::shared_ptr<Node>>> roots;
        Entry entry;
        bool counted{false};
        std::uint64_t visited{0};
        Record(ContentHash hash, std::int32_t tokens, std::size_t groups) : hash{hash}, tokens{tokens}, roots(groups) {}
        ~Record() {
            for (const auto& group : roots)
                for (const auto& root : group) std::erase(root->watchers, this);
        }
    };

    const CacheStore& store;
    // Declared before records: registration roots are destroyed while both
    // node registries still exist. Parents are non-owning; children own ranges.
    std::unordered_map<RangeKey, std::weak_ptr<Node>, RangeHash> ranges;
    std::list<Node*> leaves;
    std::unordered_map<ContentHash, std::unique_ptr<Record>> records;
    std::uint64_t epoch{0};
    std::uint64_t route_updates{0};
    std::uint64_t range_visits{0};
    std::uint64_t disk_route_losses{0};
    std::size_t ready{0}, device_ready{0}, pending{0}, disk_recoverable{0}, broken{0};
    explicit Impl(const CacheStore& store) : store{store} {}

    CacheKey KeyAt(const Record& record, std::size_t group, std::int32_t block) const {
        if (store.GroupIsState(group))
            return CacheKey{.group_id = static_cast<std::uint32_t>(group), .content_hash = record.hash};
        const auto per_page = store.BlocksPerBoundary(static_cast<std::int32_t>(group));
        const auto page = block / per_page;
        return CacheKey{.group_id = static_cast<std::uint32_t>(group),
                        .content_hash = page < record.tokens / store.PrefixGranularity()
                                            ? store.Checkpoints().PrefixHashAt(record.hash, page)
                                            : record.hash,
                        .page_offset = block % per_page};
    }

    std::shared_ptr<Node> Intern(const Record& record, std::size_t group, std::int32_t first, unsigned level) {
        const auto count = std::int32_t{1} << level;
        const auto end = first + count;  // bounded by the validated interval's exclusive end
        const RangeKey key{KeyAt(record, group, end - 1), level};
        if (auto it = ranges.find(key); it != ranges.end()) {
            auto node = it->second.lock();
            FatalCheck(node && node->first == first, "one recovery range has one logical interval");
            return node;
        }
        auto node = std::make_shared<Node>(*this, key, first, end);
        if (level > 0) {
            node->left = Intern(record, group, first, level - 1);
            node->right = Intern(record, group, first + count / 2, level - 1);
            node->left->parents.push_back(node.get());
            node->right->parents.push_back(node.get());
            for (std::size_t i = 0; i < node->counts.size(); ++i)
                node->counts[i] = node->left->counts[i] + node->right->counts[i];
        } else {
            if (!store.GroupIsState(group) && first > 0) node->previous = KeyAt(record, group, first - 1);
            node->counts[0] = store.Groups()[group].Index().Contains(store.DevicePool(), key.last);
            node->counts[1] = store.HostPool() && store.Groups()[group].Index().Contains(*store.HostPool(), key.last);
            node->pending = store.HasReservedHostKey(key.last);
            node->counts[kDisk] =
                store.DiskPool() && store.Groups()[group].Index().Contains(*store.DiskPool(), key.last);
            UpdateUnions(node->counts, node->pending);
            node->leaf_position = leaves.insert(leaves.end(), node.get());
            node->listed = true;
        }
        ranges.emplace(key, node);
        return node;
    }

    void AddRange(Record& record, std::size_t group, std::int32_t first, std::int32_t end) {
        while (first < end) {
            auto count = std::bit_floor(static_cast<std::uint32_t>(end - first));
            while (static_cast<std::uint32_t>(first) % count != 0) count >>= 1;
            record.roots[group].push_back(Intern(record, group, first, std::countr_zero(count)));
            first += static_cast<std::int32_t>(count);
        }
    }

    std::int32_t Missing(const Node& node, std::int32_t first, std::int32_t end, std::size_t tier) {
        ++range_visits;
        if (node.past_end <= first || node.first >= end || node.counts[tier] == node.past_end - node.first) return end;
        if (!node.left) return node.first;
        const auto at = Missing(*node.left, first, end, tier);
        return at != end ? at : Missing(*node.right, first, end, tier);
    }

    template <typename MissingRange, typename ReadCounts>
    Entry Resolve(const Record& record, bool allow_copy, const MissingRange& missing, const ReadCounts& counts) const {
        const auto resolve = [&](std::size_t host_column, bool copy) {
            return ResolveRecoveryRanges(store, record.tokens, store.HostPool() != nullptr, true,
                                         [&](std::size_t group, std::int32_t first, std::int32_t end, CacheTier tier) {
                                             const auto column = tier == CacheTier::kDevice ? kDevice : host_column;
                                             for (const auto& root : record.roots[group]) {
                                                 const auto at = missing(*root, first, end, column, copy);
                                                 if (at != end) return at;
                                             }
                                             return end;
                                         });
        };
        Entry result;
        auto route = resolve(kHost, false);
        result.ready = route.has_value();
        result.device_ready = route && route->device_only;
        bool include_pending = false;
        if (!route) {
            route = resolve(kHostPending, false);
            include_pending = route.has_value();
        }
        if (!route && store.DiskPool()) route = resolve(kHostDisk, false);
        if (!route && store.DiskPool()) {
            route = resolve(kHostDiskPending, false);
            include_pending = route.has_value();
        }
        if (!route && allow_copy) {
            route = resolve(store.DiskPool() ? kHostDiskPending : kHostPending, true);
            include_pending = route.has_value();
        }
        result.pending = !result.ready && include_pending;
        if (!route) return result;
        const auto add = [&](std::size_t group, std::int32_t first, std::int32_t end, CacheTier tier) {
            if (first >= end) return;
            auto& paths = tier == CacheTier::kDevice ? result.device_paths
                          : tier == CacheTier::kHost ? result.host_paths
                                                     : result.disk_paths;
            paths.push_back({KeyAt(record, group, first), KeyAt(record, group, end - 1)});
            if (tier == CacheTier::kDisk) {
                result.disk_recoverable = true;
                if (store.UsesGds())
                    result.disk_read_cost +=
                        static_cast<double>(end - first) * store.EvictionConfig().disk_block_costs.at(group);
            }
        };
        // Split only at residency changes in shared ranges, not once per
        // history key. Disk is a source for the logical Host suffix.
        const auto host_paths = [&](auto&& self, const Node& node, std::size_t group, std::int32_t first,
                                    std::int32_t end) -> void {
            first = std::max(first, node.first);
            end = std::min(end, node.past_end);
            if (first >= end) return;
            const auto& values = counts(node);
            const auto host = values[include_pending ? kHostPending : kHost];
            const auto length = node.past_end - node.first;
            if (host == length || values[kDisk] == 0) {
                add(group, first, end, CacheTier::kHost);
            } else if (host == 0 && values[kDisk] == length) {
                add(group, first, end, CacheTier::kDisk);
            } else {
                self(self, *node.left, group, first, end);
                self(self, *node.right, group, first, end);
            }
        };
        for (const auto& range : route->dependencies) {
            if (range.tier == CacheTier::kDevice) {
                add(range.group, range.first, range.past_end, range.tier);
            } else {
                for (const auto& root : record.roots[range.group])
                    host_paths(host_paths, *root, range.group, range.first, range.past_end);
            }
        }
        return result;
    }

    void Count(const Entry& entry, bool adding) {
        const auto update = [&](std::size_t& total, bool value) {
            if (adding)
                total += value;
            else
                total -= value;
        };
        update(ready, entry.ready);
        update(device_ready, entry.device_ready);
        update(pending, entry.pending);
        update(disk_recoverable, entry.disk_recoverable);
        update(broken, !entry.Available());
    }

    void Refresh(Record& record) {
        ++route_updates;
        if (record.counted) Count(record.entry, false);
        record.entry = Resolve(
            record, false,
            [&](const Node& node, auto first, auto end, auto column, bool) {
                return Missing(node, first, end, column);
            },
            [](const Node& node) -> const Counts& { return node.counts; });
        Count(record.entry, true);
        record.counted = true;
    }

    void Changed(Node& leaf, bool disk_loss) {
        // One event visits each shared ancestor and affected point once, in
        // increasing range height. No work depends on unrelated checkpoints.
        ++epoch;
        FatalCheck(epoch != 0, "recovery mutation epoch overflow");
        std::array<std::vector<Node*>, 32> levels;
        std::vector<Record*> dirty;
        levels[0].push_back(&leaf);
        leaf.visited = epoch;
        for (auto& level : levels) {
            for (Node* node : level) {
                if (node->left)
                    for (std::size_t i = 0; i < node->counts.size(); ++i)
                        node->counts[i] = node->left->counts[i] + node->right->counts[i];
                for (Record* record : node->watchers) {
                    if (record->visited == epoch) continue;
                    record->visited = epoch;
                    dirty.push_back(record);
                }
                for (Node* parent : node->parents) {
                    if (parent->visited == epoch) continue;
                    parent->visited = epoch;
                    levels[parent->key.level].push_back(parent);
                }
            }
        }
        for (Record* record : dirty) {
            const bool available = record->entry.Available();
            Refresh(*record);
            if (disk_loss && available && !record->entry.Available()) ++disk_route_losses;
        }
    }

    void Set(const CacheKey& key, std::size_t column, bool present) {
        const auto it = ranges.find(RangeKey{key, 0});
        if (it == ranges.end()) return;
        const auto leaf = it->second.lock();
        const auto previous = leaf->counts;
        if (column == kHostPending)
            leaf->pending = present;
        else
            leaf->counts[column] = present;
        UpdateUnions(leaf->counts, leaf->pending);
        if (leaf->counts != previous) Changed(*leaf, column == kDisk && !present);
    }
};

struct RecoveryIndex::View::State {
    const Impl* index;
    std::unordered_map<const Impl::Node*, tokenspeed::Counts> counts;
    std::unordered_map<ContentHash, Entry> routes;

    explicit State(const Impl& index) : index{&index} {}
    const tokenspeed::Counts& Counts(const Impl::Node& node) const {
        const auto found = counts.find(&node);
        return found == counts.end() ? node.counts : found->second;
    }
    std::int32_t Missing(const Impl::Node& node, std::int32_t first, std::int32_t end, std::size_t column,
                         bool allow_device_copy) const {
        if (node.past_end <= first || node.first >= end || Counts(node)[column] == node.past_end - node.first ||
            (column != 0 && allow_device_copy && node.counts[0] == node.past_end - node.first))
            return end;
        if (!node.left) return node.first;
        const auto at = Missing(*node.left, first, end, column, allow_device_copy);
        return at != end ? at : Missing(*node.right, first, end, column, allow_device_copy);
    }
};

RecoveryIndex::View::View(const RecoveryIndex& index) : state_{std::make_unique<State>(*index.impl_)} {}
RecoveryIndex::View::~View() = default;
RecoveryIndex::View::View(const View& other) : state_{std::make_unique<State>(*other.state_)} {}
RecoveryIndex::View& RecoveryIndex::View::operator=(const View& other) {
    if (this != &other) state_ = std::make_unique<State>(*other.state_);
    return *this;
}
RecoveryIndex::View::View(View&&) noexcept = default;
RecoveryIndex::View& RecoveryIndex::View::operator=(View&&) noexcept = default;

bool RecoveryIndex::View::Present(const CacheKey& key, CacheTier tier) const {
    const auto found = state_->index->ranges.find(Impl::RangeKey{key, 0});
    if (found == state_->index->ranges.end()) return false;
    return state_->Counts(*found->second.lock())[PhysicalColumn(tier)] != 0;
}

void RecoveryIndex::View::Apply(std::span<const Change> changes) {
    ProfileSpan profile_span{"RecoveryIndex::View::Apply"};
    std::array<std::unordered_set<const Impl::Node*>, 32> levels;
    std::unordered_set<const Impl::Record*> affected;
    for (const auto& change : changes) {
        const auto found = state_->index->ranges.find(Impl::RangeKey{change.key, 0});
        if (found == state_->index->ranges.end()) continue;
        const auto node = found->second.lock();
        auto values = state_->Counts(*node);
        if (values[PhysicalColumn(change.tier)] == change.present) continue;
        values[PhysicalColumn(change.tier)] = change.present;
        UpdateUnions(values, node->pending);
        state_->counts[node.get()] = values;
        levels[0].insert(node.get());
    }
    for (auto& level : levels) {
        for (const auto* node : level) {
            affected.insert(node->watchers.begin(), node->watchers.end());
            if (node->left) {
                auto& values = state_->counts[node];
                for (std::size_t c = 0; c < values.size(); ++c)
                    values[c] = state_->Counts(*node->left)[c] + state_->Counts(*node->right)[c];
            }
            for (const auto* parent : node->parents) levels[parent->key.level].insert(parent);
        }
    }
    for (const auto* record : affected) {
        state_->routes[record->hash] = state_->index->Resolve(
            *record, false,
            [&](const Impl::Node& node, auto first, auto end, auto column, bool copy) {
                return state_->Missing(node, first, end, column, copy);
            },
            [&](const Impl::Node& node) -> const tokenspeed::Counts& { return state_->Counts(node); });
    }
}

const RecoveryIndex::Entry& RecoveryIndex::View::At(const ContentHash& hash) const {
    const auto found = state_->routes.find(hash);
    return found == state_->routes.end() ? state_->index->records.at(hash)->entry : found->second;
}

RecoveryIndex::Entry RecoveryIndex::View::Resolve(const ContentHash& hash, bool allow_device_copy) const {
    if (!allow_device_copy) return At(hash);
    const auto& record = *state_->index->records.at(hash);
    return state_->index->Resolve(
        record, allow_device_copy,
        [&](const Impl::Node& node, auto first, auto end, auto column, bool copy) {
            return state_->Missing(node, first, end, column, copy);
        },
        [&](const Impl::Node& node) -> const tokenspeed::Counts& { return state_->Counts(node); });
}

RecoveryIndex::RecoveryIndex(const CacheStore& store) : impl_{std::make_unique<Impl>(store)} {}
RecoveryIndex::~RecoveryIndex() = default;
RecoveryIndex::Residency::Residency(RecoveryIndex& index, const CacheKey& key, CacheTier tier)
    : index_{&index}, key_{key}, tier_{tier} {
    index.Apply(RecoveryResidencyEvent{key, tier, true});
}
RecoveryIndex::Residency::Residency(Residency&& other) noexcept
    : index_{std::exchange(other.index_, nullptr)}, key_{other.key_}, tier_{other.tier_} {}
RecoveryIndex::Residency& RecoveryIndex::Residency::operator=(Residency&& other) noexcept {
    if (this != &other) {
        Reset();
        index_ = std::exchange(other.index_, nullptr);
        key_ = other.key_;
        tier_ = other.tier_;
    }
    return *this;
}
RecoveryIndex::Residency::~Residency() {
    Reset();
}
void RecoveryIndex::Residency::Reset() noexcept {
    if (index_) {
        index_->Apply(RecoveryResidencyEvent{key_, tier_, false});
        index_ = nullptr;
    }
}
RecoveryIndex::Registration::Registration(RecoveryIndex& index, const ContentHash& hash)
    : index_{&index}, hash_{hash} {}
RecoveryIndex::Registration::Registration(Registration&& other) noexcept
    : index_{std::exchange(other.index_, nullptr)}, hash_{other.hash_} {}
RecoveryIndex::Registration& RecoveryIndex::Registration::operator=(Registration&& other) noexcept {
    if (this != &other) {
        Reset();
        index_ = std::exchange(other.index_, nullptr);
        hash_ = other.hash_;
    }
    return *this;
}
RecoveryIndex::Registration::~Registration() {
    Reset();
}
void RecoveryIndex::Registration::Reset() noexcept {
    if (index_) {
        index_->Remove(hash_);
        index_ = nullptr;
    }
}
RecoveryIndex::Registration RecoveryIndex::Register(const ContentHash& hash, std::int32_t tokens) {
    ProfileSpan profile_span{"RecoveryIndex::Register"};
    auto& impl = *impl_;
    auto record = std::make_unique<Impl::Record>(hash, tokens, impl.store.Groups().size());
    for (auto i : impl.store.MatchOrder()) {
        if (impl.store.GroupIsState(i)) {
            impl.AddRange(*record, i, 0, 1);
            continue;
        }
        const auto grain = impl.store.Geometry()[i].BlockGranularity();
        const auto window = impl.store.Groups()[i].Matcher().IsPrefixClosed()
                                ? CacheStore::BlockRange{0, tokens / grain + (tokens % grain != 0)}
                                : impl.store.SlidingWindowBlocks(static_cast<std::int32_t>(i), tokens);
        std::vector<CacheStore::BlockRange> required{impl.store.SinkBlocksBelow(static_cast<std::int32_t>(i), window),
                                                     window};
        const auto tail = tokens % impl.store.PrefixGranularity();
        if (tail > 0) {
            const auto base = (tokens - tail) / grain;
            required.push_back({base + impl.store.FirstTailBlockNeeded(static_cast<std::int32_t>(i), tail),
                                base + impl.store.TailBlocksFor(static_cast<std::int32_t>(i), tail)});
        }
        std::ranges::sort(required, {}, &CacheStore::BlockRange::first);
        std::int32_t first = 0, end = 0;
        for (auto range : required) {
            if (range.first >= range.past_end) continue;
            if (range.first > end) {
                impl.AddRange(*record, i, first, end);
                first = range.first;
            }
            end = std::max(end, range.past_end);
        }
        impl.AddRange(*record, i, first, end);
    }
    for (const auto& group : record->roots)
        for (const auto& root : group) root->watchers.push_back(record.get());
    impl.Refresh(*record);
    const auto [it, inserted] = impl.records.emplace(hash, std::move(record));
    FatalCheck(inserted, "one recovery registration per checkpoint");
    return Registration{*this, hash};
}
void RecoveryIndex::Remove(const ContentHash& hash) noexcept {
    ProfileSpan profile_span{"RecoveryIndex::Remove"};
    const auto& entry = impl_->records.at(hash)->entry;
    impl_->Count(entry, false);
    impl_->records.erase(hash);
}
void RecoveryIndex::Apply(const RecoveryResidencyEvent& event) {
    ProfileSpan profile_span{"RecoveryIndex::Residency"};
    impl_->Set(event.key, PhysicalColumn(event.tier), event.present);
}
void RecoveryIndex::Apply(const RecoveryPromiseEvent& event) {
    ProfileSpan profile_span{"RecoveryIndex::Promise"};
    impl_->Set(event.key, kHostPending, event.reserved);
}
const RecoveryIndex::Entry& RecoveryIndex::At(const ContentHash& hash) const {
    return impl_->records.at(hash)->entry;
}
std::vector<RecoveryIndex::Leaf> RecoveryIndex::Leaves() const {
    std::vector<Leaf> result;
    result.reserve(impl_->leaves.size());
    for (const auto* node : impl_->leaves) result.push_back({node->key.last, node->previous});
    return result;
}
RecoveryIndex::Stats RecoveryIndex::GetStats() const {
    return {impl_->records.size(),   impl_->ranges.size(),    impl_->leaves.size(),
            impl_->route_updates,    impl_->range_visits,     impl_->ready,
            impl_->device_ready,     impl_->pending,          impl_->broken,
            impl_->disk_recoverable, impl_->disk_route_losses};
}
}  // namespace tokenspeed
