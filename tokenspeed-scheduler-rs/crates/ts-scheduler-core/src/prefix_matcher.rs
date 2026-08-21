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

//! Per-attention-kind prefix-match policy.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/prefix/prefix_matcher.h`. A
//! matcher only reads the group's `PrefixCacheIndex`; it never touches
//! allocation or physical placement. The C++ virtual hierarchy is modeled as a
//! closed enum over the two concrete policies.

use std::cell::RefCell;
use std::rc::Rc;

use crate::block_pool::BlockPool;
use crate::cache_types::{CacheKey, GroupPrefixProbe};
use crate::prefix_index::PrefixCacheIndex;

/// Prefix-match policy of one attention group.
#[derive(Debug, Clone, Copy)]
pub enum PrefixMatcher {
    Full(FullAttnMatcher),
    Swa(SwaMatcher),
}

impl PrefixMatcher {
    /// True when every match is a hole-free run from the prefix start, so a
    /// shorter boundary always remains valid.
    pub fn is_prefix_closed(&self) -> bool {
        match self {
            PrefixMatcher::Full(_) => true,
            PrefixMatcher::Swa(_) => false,
        }
    }

    /// Cached pages a resumable boundary needs behind it, in group pages.
    pub fn boundary_lookback_pages(&self) -> i32 {
        match self {
            PrefixMatcher::Full(_) => 0,
            PrefixMatcher::Swa(m) => m.pages_needed_to_resume(),
        }
    }

    /// Probes `keys[begin_blocks, begin_blocks + max_blocks)` against the
    /// index. `probe.hits[i]` marks `keys[begin_blocks + i]`; holes are 0.
    pub fn probe(
        &self,
        index: &PrefixCacheIndex,
        pool: &Rc<RefCell<BlockPool>>,
        keys: &[CacheKey],
        begin_blocks: i32,
        max_blocks: i32,
    ) -> GroupPrefixProbe {
        match self {
            PrefixMatcher::Full(m) => m.probe(index, pool, keys, begin_blocks, max_blocks),
            PrefixMatcher::Swa(m) => m.probe(index, pool, keys, begin_blocks, max_blocks),
        }
    }
}

/// Full attention: a hit is a contiguous run with no holes, so the lookup
/// walks left-to-right until the first miss.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullAttnMatcher;

impl FullAttnMatcher {
    pub fn probe(
        &self,
        index: &PrefixCacheIndex,
        pool: &Rc<RefCell<BlockPool>>,
        keys: &[CacheKey],
        begin_blocks: i32,
        max_blocks: i32,
    ) -> GroupPrefixProbe {
        let end_blocks = keys.len().min(max_blocks.max(0) as usize) as i32;
        let mut probe = GroupPrefixProbe::default();
        for j in begin_blocks..end_blocks {
            if !index.contains(pool, &keys[j as usize]) {
                break;
            }
            probe.hits.push(1);
        }
        probe
    }
}

/// Sliding window (and, with `window == 2`, mamba state snapshots): non-closed
/// — shortening a match can cut its trailing run below the window, so match
/// bound-first.
#[derive(Debug, Clone, Copy)]
pub struct SwaMatcher {
    block_granularity: i32,
    sliding_window: i32,
}

impl SwaMatcher {
    pub fn new(block_granularity: i32, sliding_window: i32) -> Self {
        assert!(block_granularity > 0, "block_granularity must be > 0");
        assert!(sliding_window > 0, "sliding_window must be > 0");
        Self {
            block_granularity,
            sliding_window,
        }
    }

    pub fn probe(
        &self,
        index: &PrefixCacheIndex,
        pool: &Rc<RefCell<BlockPool>>,
        keys: &[CacheKey],
        begin_blocks: i32,
        max_blocks: i32,
    ) -> GroupPrefixProbe {
        let end_blocks = keys.len().min(max_blocks.max(0) as usize) as i32;
        let mut probe = GroupPrefixProbe::default();
        if begin_blocks >= end_blocks {
            return probe;
        }
        // W == 1: no lookback, so every boundary is resumable with no cached
        // page at all.
        if self.pages_needed_to_resume() == 0 {
            probe.hits.resize((end_blocks - begin_blocks) as usize, 0);
            return probe;
        }
        let probe_fn = |i: i32| index.contains(pool, &keys[i as usize]);
        let (boundary, hits_begin) =
            self.find_resumable_boundary(&probe_fn, begin_blocks, end_blocks);
        if boundary == begin_blocks {
            return probe;
        }
        probe.hits.resize((boundary - begin_blocks) as usize, 0);
        for i in hits_begin..boundary {
            probe.hits[(i - begin_blocks) as usize] = 1;
        }
        probe
    }

    /// Cached pages a boundary needs behind it: they cover the window's last
    /// `(window - 1)` tokens.
    fn pages_needed_to_resume(&self) -> i32 {
        (self.sliding_window - 1 + self.block_granularity - 1) / self.block_granularity
    }

    /// Core scan shared by device and host lookup: the highest boundary backed
    /// by enough consecutive probe hits — `pages_needed_to_resume()`, or fewer
    /// bottoming out at `begin_blocks`.
    fn find_resumable_boundary(
        &self,
        probe: &impl Fn(i32) -> bool,
        begin_blocks: i32,
        end_blocks: i32,
    ) -> (i32, i32) {
        let pages_needed = self.pages_needed_to_resume();
        let mut boundary = end_blocks;
        while boundary > begin_blocks {
            let mut hits_begin = boundary;
            while hits_begin > begin_blocks && probe(hits_begin - 1) {
                hits_begin -= 1;
                if boundary - hits_begin >= pages_needed {
                    return (boundary, hits_begin); // enough pages behind the boundary
                }
            }
            if hits_begin == begin_blocks && hits_begin < boundary {
                return (boundary, hits_begin); // fewer, but nothing below begin_blocks is needed
            }
            // The miss at hits_begin-1 cuts every boundary in
            // (hits_begin-1, boundary] short -- retry below it.
            boundary = hits_begin - 1;
        }
        (begin_blocks, begin_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_types::{CacheBoundaryKind, CacheKey};

    fn pool(n: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(n)))
    }

    fn key(hash: &str) -> CacheKey {
        CacheKey {
            namespace_id: 0,
            group_id: 0,
            content_hash: hash.to_string(),
            page_offset: 0,
        }
    }

    /// Registers one cached block per key and returns the local refs (pinning
    /// them so they stay alive for the probe).
    fn seed(
        index: &mut PrefixCacheIndex,
        p: &Rc<RefCell<BlockPool>>,
        hashes: &[&str],
    ) -> Vec<crate::cache_block_ref::CacheBlockRef> {
        let mut refs = Vec::new();
        for (i, h) in hashes.iter().enumerate() {
            let mut b = p.borrow_mut().acquire_block(p, 0, 1).expect("block");
            index.register(
                p,
                &mut b,
                key(h),
                1,
                i as i32,
                CacheBoundaryKind::Chunk,
                None,
            );
            refs.push(b);
        }
        refs
    }

    #[test]
    fn full_attn_probe_stops_at_first_miss() {
        let p = pool(8);
        let mut index = PrefixCacheIndex::new(0);
        let refs = seed(&mut index, &p, &["a", "b", "d", "e"]);
        let keys: Vec<CacheKey> = vec![key("a"), key("b"), key("c"), key("d")];
        let matcher = PrefixMatcher::Full(FullAttnMatcher);
        let probe = matcher.probe(&index, &p, &keys, 0, 4);
        assert_eq!(probe.hits, vec![1, 1]); // c is a miss -> stop
        let _ = refs;
    }

    #[test]
    fn full_attn_probe_respects_begin_and_max() {
        let p = pool(8);
        let mut index = PrefixCacheIndex::new(0);
        let refs = seed(&mut index, &p, &["a", "b", "c"]);
        let keys: Vec<CacheKey> = vec![key("a"), key("b"), key("c")];
        let matcher = PrefixMatcher::Full(FullAttnMatcher);
        // max_blocks is an absolute end (C++ end = min(keys, max(0, max_blocks))):
        // begin=1, end=2 -> probes only b -> [1].
        let probe = matcher.probe(&index, &p, &keys, 1, 2);
        assert_eq!(probe.hits, vec![1]);
        // begin=1, end=3 -> probes b, c -> [1, 1].
        let probe = matcher.probe(&index, &p, &keys, 1, 3);
        assert_eq!(probe.hits, vec![1, 1]);
        // max 0 -> no probe range
        let probe = matcher.probe(&index, &p, &keys, 0, 0);
        assert!(probe.hits.is_empty());
        let _ = refs;
    }

    #[test]
    fn swa_probe_w_1_needs_no_cached_pages() {
        let p = pool(8);
        let index = PrefixCacheIndex::new(0);
        let keys: Vec<CacheKey> = vec![key("a"), key("b"), key("c")];
        let matcher = PrefixMatcher::Swa(SwaMatcher::new(1, 1));
        // pages_needed = (1-1+1-1)/1 = 0 -> all-zero hits, boundary at end.
        let probe = matcher.probe(&index, &p, &keys, 0, 3);
        assert_eq!(probe.hits, vec![0, 0, 0]);
    }

    #[test]
    fn swa_probe_finds_trailing_run_for_window() {
        // block_granularity=1, window=3 -> pages_needed = (3-1)/1 = 2.
        let p = pool(8);
        let mut index = PrefixCacheIndex::new(0);
        // Cache a, b, d (skip c). Window 3 needs the last 2 pages behind a boundary.
        let refs = seed(&mut index, &p, &["a", "b", "d"]);
        let keys: Vec<CacheKey> = vec![key("a"), key("b"), key("c"), key("d")];
        let matcher = PrefixMatcher::Swa(SwaMatcher::new(1, 3));
        let probe = matcher.probe(&index, &p, &keys, 0, 4);
        // Right-to-left scan: boundary=4 needs 2 hits behind it -> only d (1)
        // so it retries below the miss at c; boundary=2 is backed by a,b
        // (2 consecutive hits) -> hits cover [0, 2) = [1, 1].
        assert_eq!(probe.hits, vec![1, 1]);
        let _ = refs;
    }

    #[test]
    fn swa_probe_full_window_run() {
        let p = pool(8);
        let mut index = PrefixCacheIndex::new(0);
        let refs = seed(&mut index, &p, &["a", "b", "c"]);
        let keys: Vec<CacheKey> = vec![key("a"), key("b"), key("c")];
        let matcher = PrefixMatcher::Swa(SwaMatcher::new(1, 3));
        let probe = matcher.probe(&index, &p, &keys, 0, 3);
        // boundary=3 needs only the trailing 2 pages (b,c) -> hits [0,1,1].
        assert_eq!(probe.hits, vec![0, 1, 1]);
        let _ = refs;
    }

    #[test]
    fn boundary_lookback_pages_reflects_window() {
        let matcher = PrefixMatcher::Swa(SwaMatcher::new(4, 9));
        // (9-1 + 4-1)/4 = 11/4 = 2
        assert_eq!(matcher.boundary_lookback_pages(), 2);
        assert!(!matcher.is_prefix_closed());
        assert!(PrefixMatcher::Full(FullAttnMatcher).is_prefix_closed());
        assert_eq!(
            PrefixMatcher::Full(FullAttnMatcher).boundary_lookback_pages(),
            0
        );
    }
}
