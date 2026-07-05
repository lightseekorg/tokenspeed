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

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace tokenspeed {

// Per-block metadata. ref_cnt_==0 does NOT destroy the block: it re-enters the free list hash-intact,
// so it is both prefix-reusable and an eviction candidate.
class CacheBlock {
public:
    explicit CacheBlock(std::int32_t block_id) : block_id_{block_id} {}

    // Copy deleted (free_pos_ points into BlockPool::free_); move defaulted only for
    // vector::emplace_back and never actually runs (blocks_ is reserve()'d, never reallocates).
    CacheBlock(const CacheBlock&) = delete;
    CacheBlock& operator=(const CacheBlock&) = delete;
    CacheBlock(CacheBlock&&) = default;
    CacheBlock& operator=(CacheBlock&&) = default;

    std::int32_t BlockId() const { return block_id_; }
    std::int32_t RefCount() const { return ref_cnt_; }
    bool IsNull() const { return is_null_; }
    bool IsCached() const { return !block_hash_.empty(); }

    // BlockHashWithGroupId key (page_hasher.h); empty when nothing is cached.
    const std::string& BlockHash() const { return block_hash_; }

    void IncrRef() { ++ref_cnt_; }
    void DecrRef() {
        _assert(ref_cnt_ >= 1, "ref_cnt must >= 1 on DecrRef");
        --ref_cnt_;
    }

private:
    friend class BlockPool;

    void SetHash(std::string hash) {
        _assert(block_hash_.empty(), "block already has a hash");
        block_hash_ = std::move(hash);
    }
    void ResetHash() { block_hash_.clear(); }

    std::int32_t block_id_{0};
    std::int32_t ref_cnt_{0};
    std::string block_hash_{};
    bool is_null_{false};
    // Valid only while in_free_: this block's node in BlockPool::free_, for O(1) removal on a prefix hit.
    bool in_free_{false};
    std::list<CacheBlock*>::iterator free_pos_{};
};

// Flat prefix-cache block pool; owns all blocks for their whole lifetime, the free list and hash map only track state.
class BlockPool {
public:
    explicit BlockPool(std::int32_t total_num_blocks, bool enable_caching = true)
        : total_num_blocks_{total_num_blocks}, enable_caching_{enable_caching} {
        _assert(total_num_blocks > 0, "total_num_blocks must be > 0");
        blocks_.reserve(total_num_blocks);
        for (std::int32_t i = 0; i < total_num_blocks; ++i) {
            blocks_.emplace_back(i);
        }
        // Block 0 is the null placeholder: never cached, ref not tracked; all others start free.
        null_block_ = &blocks_[0];
        null_block_->is_null_ = true;
        for (std::int32_t i = 1; i < total_num_blocks; ++i) {
            pushToFree(&blocks_[i]);
        }
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    std::int32_t TotalBlocks() const { return total_num_blocks_; }
    std::int32_t NumFreeBlocks() const { return static_cast<std::int32_t>(free_.size()); }
    CacheBlock* NullBlock() { return null_block_; }

    // nullptr on miss; does NOT change ref counts -- callers TouchBlock() the result to claim it.
    CacheBlock* GetCachedBlock(const std::string& block_hash_with_group) {
        if (!enable_caching_) {
            return nullptr;
        }
        auto it = cached_hash_to_blocks_.find(block_hash_with_group);
        if (it == cached_hash_to_blocks_.end() || it->second.empty()) {
            return nullptr;
        }
        return it->second.front();
    }

    // Claim a new reference, pulling a ref-0 eviction candidate out of the free list first (vllm BlockPool.touch).
    void TouchBlock(CacheBlock* block) {
        if (block->is_null_) {
            return;
        }
        if (block->ref_cnt_ == 0) {
            removeFromFree(block);
        }
        block->IncrRef();
    }

    // All-or-nothing (empty on shortfall): fresh ref-1 blocks from the LRU head, evicting popped cached content.
    std::vector<CacheBlock*> AllocateBlocks(std::int32_t num) {
        std::vector<CacheBlock*> out;
        if (num <= 0 || static_cast<std::int32_t>(free_.size()) < num) {
            return out;
        }
        out.reserve(num);
        for (std::int32_t i = 0; i < num; ++i) {
            CacheBlock* block = popFromFree();
            if (block->IsCached()) {
                evictCachedBlock(block);
            }
            block->IncrRef();
            out.push_back(block);
        }
        return out;
    }

    // Blocks reaching ref 0 return hash-intact, in reverse so a chain's tail (more prefix tokens) evicts first.
    void FreeBlocks(const std::vector<CacheBlock*>& blocks) {
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            CacheBlock* block = *it;
            if (block->is_null_) {
                continue;
            }
            block->DecrRef();
            if (block->ref_cnt_ == 0) {
                pushToFree(block);
            }
        }
    }

    void CacheFullBlocks(CacheBlock* block, const std::string& block_hash_with_group) {
        if (!enable_caching_ || block->is_null_) {
            return;
        }
        block->SetHash(block_hash_with_group);
        cached_hash_to_blocks_[block_hash_with_group].push_back(block);
    }

    std::int32_t NumCachedFreeBlocks() const {
        std::int32_t n = 0;
        for (const auto& [key, blocks] : cached_hash_to_blocks_) {
            for (const CacheBlock* b : blocks) {
                if (b->ref_cnt_ == 0) {
                    ++n;
                }
            }
        }
        return n;
    }

private:
    // std::list gives the O(1) stored-iterator erase a prefix cache needs and a vector/deque stack cannot.
    void pushToFree(CacheBlock* block) {
        block->free_pos_ = free_.insert(free_.end(), block);
        block->in_free_ = true;
    }

    CacheBlock* popFromFree() {
        CacheBlock* block = free_.front();
        free_.pop_front();
        block->in_free_ = false;
        return block;
    }

    void removeFromFree(CacheBlock* block) {
        _assert(block->in_free_, "block is not in the free list");
        free_.erase(block->free_pos_);
        block->in_free_ = false;
    }

    void evictCachedBlock(CacheBlock* block) {
        auto it = cached_hash_to_blocks_.find(block->block_hash_);
        if (it != cached_hash_to_blocks_.end()) {
            std::erase(it->second, block);
            if (it->second.empty()) {
                cached_hash_to_blocks_.erase(it);
            }
        }
        block->ResetHash();
    }

    std::int32_t total_num_blocks_{0};
    bool enable_caching_{true};
    std::vector<CacheBlock> blocks_{};
    // LRU-ordered: front = next to evict.
    std::list<CacheBlock*> free_{};

    // One key may map to several physical duplicates: never de-duplicated so handed-out block ids stay stable.
    std::unordered_map<std::string, std::vector<CacheBlock*>> cached_hash_to_blocks_{};
    CacheBlock* null_block_{nullptr};
};

}  // namespace tokenspeed
