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

#include "scheduler/kv_event_ledger.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

#include "utils.h"

namespace tokenspeed {

class KvEventFrameWriter {
public:
    // Writes through `cursor`; the caller sized the storage for everything it
    // will write (Written() tells how much it did).
    explicit KvEventFrameWriter(std::uint8_t* cursor) : begin_{cursor}, p_{cursor} {}

    std::size_t Written() const { return static_cast<std::size_t>(p_ - begin_); }
    void Nil() { *p_++ = 0xc0; }
    void Bool(bool value) { *p_++ = value ? 0xc3 : 0xc2; }
    void Int(std::int64_t value) {
        if (value >= 0) {
            Uint(static_cast<std::uint64_t>(value));
        } else if (value >= -32) {
            *p_++ = static_cast<std::uint8_t>(value);
        } else if (value >= INT8_MIN) {
            *p_++ = 0xd0;
            *p_++ = static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
        } else if (value >= INT16_MIN) {
            *p_++ = 0xd1;
            BigEndian(static_cast<std::uint16_t>(static_cast<std::int16_t>(value)), 2);
        } else if (value >= INT32_MIN) {
            *p_++ = 0xd2;
            BigEndian(static_cast<std::uint32_t>(static_cast<std::int32_t>(value)), 4);
        } else {
            *p_++ = 0xd3;
            BigEndian(static_cast<std::uint64_t>(value), 8);
        }
    }
    void Uint(std::uint64_t value) {
        if (value < 128) {
            *p_++ = static_cast<std::uint8_t>(value);
        } else if (value <= UINT8_MAX) {
            *p_++ = 0xcc;
            *p_++ = static_cast<std::uint8_t>(value);
        } else if (value <= UINT16_MAX) {
            *p_++ = 0xcd;
            BigEndian(value, 2);
        } else if (value <= UINT32_MAX) {
            *p_++ = 0xce;
            BigEndian(value, 4);
        } else {
            *p_++ = 0xcf;
            BigEndian(value, 8);
        }
    }
    void Float64(double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        *p_++ = 0xcb;
        BigEndian(bits, 8);
    }
    void Str(std::string_view value) {
        if (value.size() < 32) {
            *p_++ = static_cast<std::uint8_t>(0xa0 | value.size());
        } else if (value.size() <= UINT8_MAX) {
            *p_++ = 0xd9;
            *p_++ = static_cast<std::uint8_t>(value.size());
        } else if (value.size() <= UINT16_MAX) {
            *p_++ = 0xda;
            BigEndian(value.size(), 2);
        } else {
            *p_++ = 0xdb;
            BigEndian(value.size(), 4);
        }
        std::memcpy(p_, value.data(), value.size());
        p_ += value.size();
    }
    void Array(std::size_t count) {
        if (count < 16) {
            *p_++ = static_cast<std::uint8_t>(0x90 | count);
        } else if (count <= UINT16_MAX) {
            *p_++ = 0xdc;
            BigEndian(count, 2);
        } else {
            *p_++ = 0xdd;
            BigEndian(count, 4);
        }
    }

private:
    void BigEndian(std::uint64_t value, int bytes) {
        for (int i = bytes - 1; i >= 0; --i) {
            *p_++ = static_cast<std::uint8_t>(value >> (8 * i));
        }
    }
    std::uint8_t* begin_;
    std::uint8_t* p_;
};
namespace {

constexpr std::string_view kStoredTag = "BlockStored";
constexpr std::string_view kRemovedTag = "BlockRemoved";

// The first 16 hex digits of a digest are 64 uniformly random bits already.
// The first sixteen hex characters of the digest as two words, mixed: the
// digest is uniform, so the words carry sixty-four random bits between them.
std::uint64_t HashOfDigest(std::string_view digest) {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::memcpy(&a, digest.data(), sizeof(a));
    std::memcpy(&b, digest.data() + sizeof(a), sizeof(b));
    std::uint64_t value = (a * 0x9e3779b97f4a7c15ull) ^ (b * 0xc2b2ae3d27d4eb4full);
    value ^= value >> 32;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 29;
    return value;
}

std::size_t IndexSize(std::size_t capacity) {
    return std::bit_ceil(std::max<std::size_t>(capacity * 2, 16));
}

// msgpack, written as msgspec writes it: the smallest format for each value.

// The reader DecodeFrame uses: the formats the writer emits, plus the other
// integer widths a hand-written producer might choose.
class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> data) : data_{data} {}

    bool AtEnd() const { return pos_ >= data_.size(); }
    std::uint8_t Peek() const {
        if (pos_ >= data_.size()) {
            throw std::invalid_argument("KV event frame: truncated");
        }
        return data_[pos_];
    }
    bool IsNil() const { return Peek() == 0xc0; }
    void Nil() { Expect(0xc0); }

    std::size_t Array() {
        const std::uint8_t byte = Next();
        if ((byte & 0xf0) == 0x90) {
            return byte & 0x0f;
        }
        if (byte == 0xdc) {
            return static_cast<std::size_t>(BigEndian(2));
        }
        if (byte == 0xdd) {
            return static_cast<std::size_t>(BigEndian(4));
        }
        throw std::invalid_argument("KV event frame: an array was expected");
    }
    std::int64_t Int() {
        const std::uint8_t byte = Next();
        if (byte < 0x80) {
            return byte;
        }
        if (byte >= 0xe0) {
            return static_cast<std::int8_t>(byte);
        }
        switch (byte) {
            case 0xcc:
                return static_cast<std::int64_t>(BigEndian(1));
            case 0xcd:
                return static_cast<std::int64_t>(BigEndian(2));
            case 0xce:
                return static_cast<std::int64_t>(BigEndian(4));
            case 0xcf:
                return static_cast<std::int64_t>(BigEndian(8));
            case 0xd0:
                return static_cast<std::int8_t>(BigEndian(1));
            case 0xd1:
                return static_cast<std::int16_t>(BigEndian(2));
            case 0xd2:
                return static_cast<std::int32_t>(BigEndian(4));
            case 0xd3:
                return static_cast<std::int64_t>(BigEndian(8));
            default:
                throw std::invalid_argument("KV event frame: an integer was expected");
        }
    }
    std::uint64_t Uint() {
        const std::uint8_t byte = Next();
        if (byte < 0x80) {
            return byte;
        }
        switch (byte) {
            case 0xcc:
                return BigEndian(1);
            case 0xcd:
                return BigEndian(2);
            case 0xce:
                return BigEndian(4);
            case 0xcf:
                return BigEndian(8);
            default:
                throw std::invalid_argument("KV event frame: an unsigned integer was expected");
        }
    }
    double Float64() {
        Expect(0xcb);
        const std::uint64_t bits = BigEndian(8);
        double value = 0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    std::string_view Str() {
        const std::uint8_t byte = Next();
        std::size_t length = 0;
        if ((byte & 0xe0) == 0xa0) {
            length = byte & 0x1f;
        } else if (byte == 0xd9) {
            length = static_cast<std::size_t>(BigEndian(1));
        } else if (byte == 0xda) {
            length = static_cast<std::size_t>(BigEndian(2));
        } else if (byte == 0xdb) {
            length = static_cast<std::size_t>(BigEndian(4));
        } else {
            throw std::invalid_argument("KV event frame: a string was expected");
        }
        if (pos_ + length > data_.size()) {
            throw std::invalid_argument("KV event frame: a string runs past the end");
        }
        const std::string_view value{reinterpret_cast<const char*>(data_.data() + pos_), length};
        pos_ += length;
        return value;
    }
    // Skips one value of any format the writer emits.
    void Skip() {
        const std::uint8_t byte = Peek();
        if (byte == 0xc0) {
            Nil();
        } else if (byte == 0xcb) {
            (void)Float64();
        } else if ((byte & 0xe0) == 0xa0 || byte == 0xd9 || byte == 0xda || byte == 0xdb) {
            (void)Str();
        } else if ((byte & 0xf0) == 0x90 || byte == 0xdc || byte == 0xdd) {
            const std::size_t count = Array();
            for (std::size_t i = 0; i < count; ++i) {
                Skip();
            }
        } else {
            (void)Int();
        }
    }

private:
    std::uint8_t Next() {
        if (pos_ >= data_.size()) {
            throw std::invalid_argument("KV event frame: truncated");
        }
        return data_[pos_++];
    }
    void Expect(std::uint8_t byte) {
        if (Next() != byte) {
            throw std::invalid_argument("KV event frame: unexpected format byte");
        }
    }
    std::uint64_t BigEndian(int bytes) {
        std::uint64_t value = 0;
        for (int i = 0; i < bytes; ++i) {
            value = (value << 8) | Next();
        }
        return value;
    }

    std::span<const std::uint8_t> data_;
    std::size_t pos_{0};
};

AttnKind KindOf(std::string_view name) {
    if (name == KvCacheSpecKind(AttnKind::kSlidingWindow)) {
        return AttnKind::kSlidingWindow;
    }
    if (name == KvCacheSpecKind(AttnKind::kState)) {
        return AttnKind::kState;
    }
    return AttnKind::kFull;
}

}  // namespace

KvEventLedger::KvEventLedger(std::vector<GroupKind> groups, std::int32_t prefix_granularity, std::size_t capacity)
    : groups_{std::move(groups)}, prefix_granularity_{prefix_granularity} {
    FatalCheck(!groups_.empty() && groups_.size() <= kMaxGroups, "KV events track one to 64 cache groups");
    FatalCheck(prefix_granularity_ > 0, "KV events need a positive prefix granularity");
    capacity = std::max<std::size_t>(capacity, 64);
    slots_.resize(capacity);
    resident_.assign(capacity * groups_.size(), 0);
    blocks_.assign(capacity * groups_.size(), 0);
    tokens_.assign(capacity * static_cast<std::size_t>(prefix_granularity_), 0);
    index_.assign(IndexSize(capacity), 0);
    for (std::size_t i = 0; i < capacity; ++i) {
        slots_[i].next_free = i + 1 < capacity ? static_cast<std::uint32_t>(i + 1) : kNoSlot;
    }
    free_head_ = 0;
    marked_.reserve(capacity);
    stores_.reserve(capacity);
    removals_.reserve(capacity);
    frame_.resize(64 * 1024);  // sized once; a frame is a prefix of it
}

std::uint32_t KvEventLedger::find(std::string_view content_hash) const {
    if (content_hash.size() != kHashLength) {
        return kNoSlot;
    }
    const std::size_t mask = index_.size() - 1;
    const std::uint64_t digest_hash = HashOfDigest(content_hash);
    for (std::size_t i = digest_hash & mask;; i = (i + 1) & mask) {
        const std::uint32_t entry = index_[i];
        if (entry == 0) {
            return kNoSlot;
        }
        const Slot& slot = slots_[entry - 1];
        if (slot.digest_hash == digest_hash && std::memcmp(slot.hash.data(), content_hash.data(), kHashLength) == 0) {
            return entry - 1;
        }
    }
}

std::uint32_t KvEventLedger::insert(std::string_view content_hash, std::uint64_t digest_hash) {
    FatalCheck(content_hash.size() == kHashLength, "a KV event boundary is named by a SHA-256 digest in hex");
    if (free_head_ == kNoSlot) {
        grow();
    }
    const std::uint32_t slot_index = free_head_;
    Slot& slot = slots_[slot_index];
    free_head_ = slot.next_free;
    slot = Slot{};
    std::memcpy(slot.hash.data(), content_hash.data(), kHashLength);
    slot.digest_hash = digest_hash;
    slot.live = true;
    std::fill(resident(slot_index).begin(), resident(slot_index).end(), 0);
    std::fill(blocks(slot_index).begin(), blocks(slot_index).end(), 0);
    const std::size_t mask = index_.size() - 1;
    std::size_t i = digest_hash & mask;
    while (index_[i] != 0) {
        i = (i + 1) & mask;
    }
    index_[i] = slot_index + 1;
    ++live_;
    return slot_index;
}

void KvEventLedger::erase(std::uint32_t slot_index) {
    Slot& slot = slots_[slot_index];
    const std::size_t mask = index_.size() - 1;
    std::size_t i = slot.digest_hash & mask;
    while (index_[i] != slot_index + 1) {
        i = (i + 1) & mask;
    }
    // Linear probing with backward shift: no tombstones, the index never fills with dead entries.
    std::size_t hole = i;
    for (std::size_t j = (hole + 1) & mask; index_[j] != 0; j = (j + 1) & mask) {
        const std::size_t home = slots_[index_[j] - 1].digest_hash & mask;
        const bool movable = hole <= j ? (home <= hole || home > j) : (home <= hole && home > j);
        if (movable) {
            index_[hole] = index_[j];
            hole = j;
        }
    }
    index_[hole] = 0;
    slot.live = false;
    slot.marked = false;
    slot.next_free = free_head_;
    free_head_ = slot_index;
    --live_;
}

void KvEventLedger::grow() {
    const std::size_t old_capacity = slots_.size();
    const std::size_t capacity = old_capacity * 2;
    slots_.resize(capacity);
    resident_.resize(capacity * groups_.size(), 0);
    blocks_.resize(capacity * groups_.size(), 0);
    tokens_.resize(capacity * static_cast<std::size_t>(prefix_granularity_), 0);
    for (std::size_t i = old_capacity; i < capacity; ++i) {
        slots_[i].next_free = i + 1 < capacity ? static_cast<std::uint32_t>(i + 1) : kNoSlot;
    }
    free_head_ = static_cast<std::uint32_t>(old_capacity);
    index_.assign(IndexSize(capacity), 0);
    rebuildIndex();
    ++growths_;
    spdlog::info("KV event ledger grew to {} boundaries (growth {})", capacity, growths_);
}

void KvEventLedger::rebuildIndex() {
    const std::size_t mask = index_.size() - 1;
    for (std::size_t s = 0; s < slots_.size(); ++s) {
        if (!slots_[s].live) {
            continue;
        }
        std::size_t i = slots_[s].digest_hash & mask;
        while (index_[i] != 0) {
            i = (i + 1) & mask;
        }
        index_[i] = static_cast<std::uint32_t>(s + 1);
    }
}

void KvEventLedger::mark(std::uint32_t slot_index) {
    Slot& slot = slots_[slot_index];
    if (!slot.marked) {
        slot.marked = true;
        marked_.push_back(slot_index);
    }
}

void KvEventLedger::Register(std::string_view content_hash, std::uint64_t block_hash,
                             std::optional<std::uint64_t> parent_hash, std::span<const std::int32_t> tokens,
                             std::int32_t depth, std::span<const std::int32_t> blocks_per_group) {
    FatalCheck(blocks_per_group.size() == groups_.size(), "one block count per cache group");
    FatalCheck(tokens.size() > 0 && static_cast<std::int32_t>(tokens.size()) <= prefix_granularity_,
               "a KV event boundary spans one to prefix_granularity tokens");
    std::uint32_t slot_index = find(content_hash);
    if (slot_index == kNoSlot) {
        slot_index = insert(content_hash, HashOfDigest(content_hash));
        Slot& slot = slots_[slot_index];
        slot.block_hash = block_hash;
        slot.has_parent = parent_hash.has_value();
        slot.parent_hash = parent_hash.value_or(0);
        slot.depth = depth;
        slot.tokens = static_cast<std::int32_t>(tokens.size());
        std::copy(tokens.begin(), tokens.end(), this->tokens(slot_index).begin());
        const std::span<std::uint16_t> block_counts = blocks(slot_index);
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            FatalCheck(blocks_per_group[g] >= 0 && blocks_per_group[g] <= UINT16_MAX, "a group's block count fits");
            block_counts[g] = static_cast<std::uint16_t>(blocks_per_group[g]);
        }
    } else {
        const Slot& slot = slots_[slot_index];
        FatalCheck(slot.block_hash == block_hash && slot.depth == depth,
                   "one cache content hash mapped to different KV event blocks");
    }
    // The next drain drops a descriptor whose publication never landed.
    mark(slot_index);
}

void KvEventLedger::Mutate(std::string_view content_hash, std::uint32_t group, bool stored) {
    const std::uint32_t slot_index = find(content_hash);
    FatalCheck(slot_index != kNoSlot, "cache mutation on a KV event boundary with no token descriptor");
    FatalCheck(group < groups_.size(), "a cache mutation names a tracked group");
    Slot& slot = slots_[slot_index];
    std::uint16_t& count = resident(slot_index)[group];
    const std::uint16_t needed = blocks(slot_index)[group];
    if (stored) {
        ++count;
        ++slot.resident_total;
    } else {
        FatalCheck(count > 0 && slot.resident_total > 0, "an eviction follows a store of the same block");
        --count;
        --slot.resident_total;
    }
    const std::uint64_t bit = std::uint64_t{1} << group;
    if (needed > 0 && count >= needed) {
        slot.complete_mask |= bit;
    } else {
        slot.complete_mask &= ~bit;
    }
    mark(slot_index);
}

std::optional<KvEventLedger::Descriptor> KvEventLedger::Find(std::string_view content_hash) const {
    const std::uint32_t slot_index = find(content_hash);
    if (slot_index == kNoSlot) {
        return std::nullopt;
    }
    return Descriptor{.depth = slots_[slot_index].depth, .tokens = slots_[slot_index].tokens};
}

std::span<const std::uint8_t> KvEventLedger::Drain(double timestamp, std::int64_t dp_rank) {
    stores_.clear();
    removals_.clear();
    for (std::size_t order = 0; order < marked_.size(); ++order) {
        const std::uint32_t slot_index = marked_[order];
        Slot& slot = slots_[slot_index];
        slot.marked = false;
        const std::uint64_t newly = slot.complete_mask & ~slot.published_mask;
        const std::uint64_t dropped = slot.published_mask & ~slot.complete_mask;
        if (dropped != 0) {
            removals_.push_back(Pending{.depth = slot.depth,
                                        .order = static_cast<std::uint32_t>(order),
                                        .slot = slot_index,
                                        .mask = dropped,
                                        .holding = 0});
        }
        if (newly != 0) {
            stores_.push_back(Pending{.depth = slot.depth,
                                      .order = static_cast<std::uint32_t>(order),
                                      .slot = slot_index,
                                      .mask = newly,
                                      .holding = slot.complete_mask});
        }
        slot.published_mask = slot.complete_mask;
    }
    if (stores_.empty() && removals_.empty()) {
        // Descriptors of boundaries nothing holds any more leave now.
        for (const std::uint32_t slot_index : marked_) {
            const Slot& slot = slots_[slot_index];
            if (slot.live && slot.resident_total == 0 && slot.published_mask == 0) {
                erase(slot_index);
            }
        }
        marked_.clear();
        return {};
    }
    // Consumers resolve a store's parent against what they already hold:
    // removals leaf-first, stores parent-first, both in mark order within a
    // depth. The order key makes the in-place sort stable (std::stable_sort
    // would allocate its buffer on every drain).
    std::sort(removals_.begin(), removals_.end(), [](const Pending& a, const Pending& b) {
        return a.depth != b.depth ? a.depth > b.depth : a.order < b.order;
    });
    std::sort(stores_.begin(), stores_.end(), [](const Pending& a, const Pending& b) {
        return a.depth != b.depth ? a.depth < b.depth : a.order < b.order;
    });
    // Storage for the whole frame first, then one cursor writes it: an
    // upper bound per event (every slot at its widest format), no per-byte
    // growth checks.
    std::size_t bound = 32 + removals_.size() * 64;
    for (const Pending& pending : stores_) {
        bound += 128 + 5 * static_cast<std::size_t>(slots_[pending.slot].tokens);
    }
    if (frame_.size() < bound) {
        frame_.resize(bound);  // grows (rarely); the bytes past a frame are never read
    }
    KvEventFrameWriter writer{frame_.data()};
    writer.Array(3);
    writer.Float64(timestamp);
    writer.Array(removals_.size() + stores_.size());
    for (const Pending& pending : removals_) {
        encodeRemoval(writer, pending.slot, pending.mask);
    }
    for (const Pending& pending : stores_) {
        encodeStore(writer, pending.slot, pending.mask, pending.holding);
    }
    writer.Int(dp_rank);
    FatalCheck(writer.Written() <= bound, "a KV event frame stays within its size bound");
    const std::span<const std::uint8_t> frame{frame_.data(), writer.Written()};
    for (const std::uint32_t slot_index : marked_) {
        const Slot& slot = slots_[slot_index];
        if (slot.live && slot.resident_total == 0 && slot.published_mask == 0) {
            erase(slot_index);
        }
    }
    marked_.clear();
    return frame;
}

void KvEventLedger::encodeStore(KvEventFrameWriter& writer, std::uint32_t slot_index, std::uint64_t newly,
                                std::uint64_t holding) {
    // The slots of the reference engine's BlockStored, in its order, then
    // this engine's group mask: the adapter, medium and placement fields it
    // does not use ride as nil. group_idx, the kind and the window name the
    // lowest group that newly holds the boundary, so a consumer that reads
    // one group per event sees that group's store; the mask names every
    // group that holds it after this drain.
    const Slot& slot = slots_[slot_index];
    const auto group = static_cast<std::size_t>(std::countr_zero(newly));
    const GroupKind& kind = groups_[group];
    writer.Array(16);
    writer.Str(kStoredTag);
    writer.Array(1);
    writer.Uint(slot.block_hash);
    if (slot.has_parent) {
        writer.Uint(slot.parent_hash);
    } else {
        writer.Nil();
    }
    const std::span<std::int32_t> token_ids = tokens(slot_index);
    writer.Array(token_ids.size());
    for (const std::int32_t token : token_ids) {
        writer.Int(token);
    }
    writer.Int(prefix_granularity_);
    writer.Nil();  // lora_id
    writer.Nil();  // medium
    writer.Nil();  // lora_name
    writer.Nil();  // extra_keys
    writer.Uint(group);
    writer.Str(KvCacheSpecKind(kind.attn_kind));
    if (kind.attn_kind == AttnKind::kSlidingWindow) {
        writer.Int(kind.sliding_window);
    } else {
        writer.Nil();
    }
    writer.Nil();  // locality
    writer.Nil();  // ownership
    writer.Nil();  // session_id
    writer.Uint(holding);
}

void KvEventLedger::encodeRemoval(KvEventFrameWriter& writer, std::uint32_t slot_index, std::uint64_t dropped) {
    // BlockRemoved's slots: hashes, medium, group_idx, locality, ownership,
    // then the mask of the groups that dropped the boundary.
    const Slot& slot = slots_[slot_index];
    writer.Array(7);
    writer.Str(kRemovedTag);
    writer.Array(1);
    writer.Uint(slot.block_hash);
    writer.Nil();  // medium
    writer.Uint(static_cast<std::uint64_t>(std::countr_zero(dropped)));
    writer.Nil();  // locality
    writer.Nil();  // ownership
    writer.Uint(dropped);
}

std::vector<KvCacheEvent> KvEventLedger::DecodeFrame(std::span<const std::uint8_t> frame) {
    std::vector<KvCacheEvent> events;
    if (frame.empty()) {
        return events;
    }
    Reader reader{frame};
    if (reader.Array() != 3) {
        throw std::invalid_argument("KV event frame: a batch has three members");
    }
    (void)reader.Float64();
    const std::size_t count = reader.Array();
    events.reserve(count);
    for (std::size_t e = 0; e < count; ++e) {
        const std::size_t slots = reader.Array();
        const std::string_view tag = reader.Str();
        if (tag == kStoredTag) {
            KvBlockStoredEvent event;
            const std::size_t hashes = reader.Array();
            for (std::size_t h = 0; h < hashes; ++h) {
                event.block_hashes.push_back(reader.Uint());
            }
            if (reader.IsNil()) {
                reader.Nil();
            } else {
                event.parent_block_hash = reader.Uint();
            }
            const std::size_t tokens = reader.Array();
            for (std::size_t t = 0; t < tokens; ++t) {
                event.token_ids.push_back(static_cast<std::int32_t>(reader.Int()));
            }
            event.block_size = static_cast<std::int32_t>(reader.Int());
            // lora_id, medium, lora_name, extra_keys
            for (int i = 0; i < 4; ++i) {
                reader.Skip();
            }
            event.group_idx = static_cast<std::int32_t>(reader.Uint());
            event.attn_kind = KindOf(reader.Str());
            if (reader.IsNil()) {
                reader.Nil();
            } else {
                event.sliding_window = static_cast<std::int32_t>(reader.Int());
            }
            // locality, ownership, session_id
            for (int i = 0; i < 3; ++i) {
                reader.Skip();
            }
            event.group_mask = reader.Uint();
            for (std::size_t s = 16; s < slots; ++s) {
                reader.Skip();
            }
            events.emplace_back(std::move(event));
        } else if (tag == kRemovedTag) {
            KvBlockRemovedEvent event;
            const std::size_t hashes = reader.Array();
            for (std::size_t h = 0; h < hashes; ++h) {
                event.block_hashes.push_back(reader.Uint());
            }
            reader.Skip();  // medium
            event.group_idx = static_cast<std::int32_t>(reader.Uint());
            reader.Skip();  // locality
            reader.Skip();  // ownership
            event.group_mask = reader.Uint();
            for (std::size_t s = 7; s < slots; ++s) {
                reader.Skip();
            }
            events.emplace_back(std::move(event));
        } else {
            throw std::invalid_argument("KV event frame: unknown event tag " + std::string(tag));
        }
    }
    (void)reader.Int();  // attn_dp_rank
    return events;
}

}  // namespace tokenspeed
