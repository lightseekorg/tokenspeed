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

//! One cache group's prefix-reuse index: `CacheKey -> canonical CacheBlock`.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/prefix/prefix_index.h`. It
//! decides WHAT is reusable; placement and allocation stay in `BlockPool` and
//! `GroupAllocator`. Indices are pool-scoped because the same group serves the
//! Device and Host tiers.
//!
//! The C++ implementation used a `std::list<CacheEntry>` plus two non-owning
//! `unordered_map` secondary indices (list iterators). Rust replaces the list
//! with a compact `Vec<CacheEntry>` whose elements never move: erase uses
//! `swap_remove` and repairs the moved element's map indices, so all stored
//! indices stay valid. Entry iteration order is not semantically load-bearing
//! (eviction selection uses `last_access_epoch`, never list position).

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::block_pool::BlockPool;
use crate::block_table::BlockTable;
use crate::cache_block_ref::{CacheBlockLocation, CacheBlockRef};
use crate::cache_types::{CacheBoundaryKind, CacheKey, GroupPrefixProbe, PrefixMatch};

/// Pointer-identity key for pool-scoped entries. Two handles to the same
/// `BlockPool` instance compare equal; different pools never collide.
#[derive(Clone)]
pub(crate) struct PoolKey(pub(crate) Rc<RefCell<BlockPool>>);

impl PartialEq for PoolKey {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for PoolKey {}
impl Hash for PoolKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

/// Read-only admission snapshot from one index lookup; owns no block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedBlockMetadata {
    pub last_access_epoch: u64,
    pub logical_block_index: i32,
    pub boundary_kind: CacheBoundaryKind,
    pub was_acquired: bool,
}

struct CacheEntry {
    key: CacheKey,
    block_ref: CacheBlockRef,
    last_access_epoch: u64,
    /// Position in the request's logical prefix. Host-only entries may not
    /// have a device-table position yet.
    logical_block_index: i32,
    boundary_kind: CacheBoundaryKind,
    /// Set only after a successful request admission acquires this entry.
    was_acquired: bool,
}

/// Per-pool entry set: `entries` owns each entry once; the maps are non-owning
/// secondary indices (Vec positions) for key and location lookup.
#[derive(Default)]
struct CacheEntries {
    entries: Vec<CacheEntry>,
    by_key: HashMap<CacheKey, usize>,
    by_location: HashMap<CacheBlockLocation, usize>,
}

/// The prefix-reuse index for one cache group.
pub struct PrefixCacheIndex {
    group_id: u32,
    cache_entries_by_pool: HashMap<PoolKey, CacheEntries>,
}

impl PrefixCacheIndex {
    pub fn new(group_id: u32) -> Self {
        Self {
            group_id,
            cache_entries_by_pool: HashMap::new(),
        }
    }

    pub fn group_id(&self) -> u32 {
        self.group_id
    }

    /// Registers `block_ref` under `key`. If `key` already has a canonical
    /// block, `block_ref` is replaced with a reference to that block.
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &mut self,
        pool: &Rc<RefCell<BlockPool>>,
        block_ref: &mut CacheBlockRef,
        key: CacheKey,
        access_epoch: u64,
        logical_block_index: i32,
        boundary_kind: CacheBoundaryKind,
        newly_cached: Option<&mut Vec<(CacheKey, CacheBlockRef)>>,
    ) {
        assert!(
            !block_ref.is_null() && block_ref.is_owned_by(pool),
            "cache block must belong to the target pool"
        );
        self.validate_key(&key);
        let cache_entries = self.cache_entries_mut(pool);
        let existing = find_entry_by_location(
            cache_entries,
            block_ref.location().expect("non-null block has a location"),
        );
        if let Some(existing) = existing {
            assert!(
                cache_entries.entries[existing].key == key,
                "one cache block location cannot change cache key"
            );
            if cache_entries.entries[existing].boundary_kind < boundary_kind {
                cache_entries.entries[existing].boundary_kind = boundary_kind;
            }
            cache_entries.entries[existing].last_access_epoch = access_epoch;
            return;
        }
        let canonical = find_entry_by_key(cache_entries, &key);
        if let Some(canonical) = canonical {
            if cache_entries.entries[canonical].boundary_kind < boundary_kind {
                cache_entries.entries[canonical].boundary_kind = boundary_kind;
            }
            cache_entries.entries[canonical].last_access_epoch = access_epoch;
            *block_ref = cache_entries.entries[canonical].block_ref.clone();
            return;
        }

        let entry_index = cache_entries.entries.len();
        cache_entries.entries.push(CacheEntry {
            key: key.clone(),
            block_ref: block_ref.clone(),
            last_access_epoch: access_epoch,
            logical_block_index,
            boundary_kind,
            was_acquired: false,
        });
        cache_entries.by_key.insert(key.clone(), entry_index);
        cache_entries.by_location.insert(
            block_ref.location().expect("non-null block has a location"),
            entry_index,
        );
        if let Some(newly_cached) = newly_cached {
            newly_cached.push((key, block_ref.clone()));
        }
    }

    /// Registers the non-null table blocks `[first_slot, first_slot + keys)`.
    #[allow(clippy::too_many_arguments)]
    pub fn register_full_blocks(
        &mut self,
        pool: &Rc<RefCell<BlockPool>>,
        table: &mut BlockTable,
        keys: &[CacheKey],
        access_epoch: u64,
        first_slot: i32,
        boundary_kind: CacheBoundaryKind,
        newly_cached: Option<&mut Vec<(CacheKey, CacheBlockRef)>>,
    ) {
        assert!(first_slot >= 0, "first_slot must be >= 0");
        assert!(
            first_slot as i64 + keys.len() as i64 <= table.num_blocks() as i64,
            "key range exceeds table size"
        );
        let mut newly_cached = newly_cached;
        for (j, key) in keys.iter().enumerate() {
            let mut block_ref = std::mem::take(&mut table.blocks[first_slot as usize + j]);
            if block_ref.is_null() {
                continue;
            }
            self.register(
                pool,
                &mut block_ref,
                key.clone(),
                access_epoch,
                first_slot + j as i32,
                boundary_kind,
                newly_cached.as_deref_mut(),
            );
            table.blocks[first_slot as usize + j] = block_ref;
        }
    }

    /// Whether `key` is cached in `pool` (location-based membership).
    pub fn contains(&self, pool: &Rc<RefCell<BlockPool>>, key: &CacheKey) -> bool {
        self.find_cache_entries(pool)
            .is_some_and(|cache_entries| find_entry_by_key(cache_entries, key).is_some())
    }

    /// Whether `location` is a cached entry in `pool`.
    pub fn contains_location(
        &self,
        pool: &Rc<RefCell<BlockPool>>,
        location: CacheBlockLocation,
    ) -> bool {
        self.find_cache_entries(pool)
            .is_some_and(|cache_entries| find_entry_by_location(cache_entries, location).is_some())
    }

    /// Any-tier lookup that also checks identity, not just location.
    pub fn contains_ref(&self, block_ref: &CacheBlockRef) -> bool {
        if block_ref.is_null() {
            return false;
        }
        let location = block_ref.location().expect("non-null block has a location");
        self.cache_entries_by_pool.values().any(|entries| {
            find_entry_by_location(entries, location)
                .is_some_and(|idx| entries.entries[idx].block_ref == *block_ref)
        })
    }

    /// The canonical block for `key` in `pool`, or a null ref.
    pub fn find(&self, pool: &Rc<RefCell<BlockPool>>, key: &CacheKey) -> CacheBlockRef {
        let Some(cache_entries) = self.find_cache_entries(pool) else {
            return CacheBlockRef::default();
        };
        match find_entry_by_key(cache_entries, key) {
            Some(idx) => cache_entries.entries[idx].block_ref.clone(),
            None => CacheBlockRef::default(),
        }
    }

    /// Admission metadata for a cached location, if any.
    pub fn metadata_for(
        &self,
        pool: &Rc<RefCell<BlockPool>>,
        location: CacheBlockLocation,
    ) -> Option<CachedBlockMetadata> {
        let cache_entries = self.find_cache_entries(pool)?;
        let idx = find_entry_by_location(cache_entries, location)?;
        let entry = &cache_entries.entries[idx];
        Some(CachedBlockMetadata {
            last_access_epoch: entry.last_access_epoch,
            logical_block_index: entry.logical_block_index,
            boundary_kind: entry.boundary_kind,
            was_acquired: entry.was_acquired,
        })
    }

    /// Number of cached entries in `pool`.
    pub fn num_entries(&self, pool: &Rc<RefCell<BlockPool>>) -> i32 {
        self.find_cache_entries(pool)
            .map_or(0, |entries| entries.entries.len() as i32)
    }

    /// Number of cached entries in `pool` pinned by more than the index itself.
    pub fn num_pinned_entries(&self, pool: &Rc<RefCell<BlockPool>>) -> i32 {
        let Some(entries) = self.find_cache_entries(pool) else {
            return 0;
        };
        entries
            .entries
            .iter()
            .filter(|entry| entry.block_ref.use_count() > 1)
            .count() as i32
    }

    /// Locations whose blocks are evictable (only the index owns them).
    pub fn evictable_locations(&self, pool: &Rc<RefCell<BlockPool>>) -> Vec<CacheBlockLocation> {
        self.evictable_locations_after_releasing(pool, &[])
    }

    /// Locations evictable once `released_locations` lose their request
    /// references (used to discount in-flight Store tickets).
    pub fn evictable_locations_after_releasing(
        &self,
        pool: &Rc<RefCell<BlockPool>>,
        released_locations: &[CacheBlockLocation],
    ) -> Vec<CacheBlockLocation> {
        let Some(entries) = self.find_cache_entries(pool) else {
            return Vec::new();
        };
        let mut locations = Vec::new();
        for entry in &entries.entries {
            let location = entry
                .block_ref
                .location()
                .expect("cached block has a location");
            let released_owners = released_locations
                .iter()
                .filter(|&&l| l == location)
                .count() as u32;
            if entry.block_ref.use_count() == 1 + released_owners {
                locations.push(location);
            }
        }
        locations
    }

    /// Evict the cached entry at `location`, returning its key. Fails (None)
    /// when the location is not a unique cached entry.
    pub fn evict(
        &mut self,
        pool: &Rc<RefCell<BlockPool>>,
        location: CacheBlockLocation,
    ) -> Option<CacheKey> {
        let cache_entries = self.cache_entries_mut_opt(pool)?;
        let idx = find_entry_by_location(cache_entries, location)?;
        if !cache_entries.entries[idx].block_ref.unique() {
            return None;
        }
        let key = cache_entries.entries[idx].key.clone();
        erase_entry(cache_entries, idx);
        Some(key)
    }

    /// True when every occupied child of the LCM parent is an unpinned entry
    /// of this index, i.e. evicting the parent loses only reusable cache.
    pub fn parent_is_fully_evictable(
        &self,
        pool: &Rc<RefCell<BlockPool>>,
        lcm_block_id: i32,
        cache_blocks_per_lcm_block: i32,
    ) -> bool {
        let pool_ref = pool.borrow();
        if pool_ref.occupied_count(lcm_block_id) == 0 {
            return false;
        }
        drop(pool_ref);
        let Some(cache_entries) = self.find_cache_entries(pool) else {
            return false;
        };
        for slot in 0..cache_blocks_per_lcm_block {
            let location = CacheBlockLocation {
                lcm_block_id,
                slot_index: slot,
            };
            let pool_ref = pool.borrow();
            if !pool_ref.is_occupied(location) {
                continue;
            }
            drop(pool_ref);
            let Some(idx) = find_entry_by_location(cache_entries, location) else {
                return false;
            };
            if !cache_entries.entries[idx].block_ref.unique() {
                return false;
            }
        }
        true
    }

    /// Pins the probed hits: marks them acquired at `access_epoch` and returns
    /// owning references aligned with `probe.hits`.
    pub fn acquire_matched(
        &mut self,
        pool: &Rc<RefCell<BlockPool>>,
        keys: &[CacheKey],
        begin_blocks: i32,
        probe: &GroupPrefixProbe,
        access_epoch: u64,
    ) -> PrefixMatch {
        assert!(
            begin_blocks >= 0 && begin_blocks as usize + probe.hits.len() <= keys.len(),
            "matched block range is out of bounds"
        );
        let mut matched = PrefixMatch {
            blocks: vec![CacheBlockRef::default(); probe.hits.len()],
        };
        let mut cache_entries = self.cache_entries_mut_opt(pool);
        for (i, hit) in probe.hits.iter().enumerate() {
            if *hit == 0 {
                continue;
            }
            // The index is only required when a hit is actually acquired
            // (C++ asserts cache_index != nullptr inside this loop).
            let cache_entries = cache_entries
                .as_mut()
                .expect("cached pool disappeared between match probe and acquisition");
            let idx = find_entry_by_key(cache_entries, &keys[begin_blocks as usize + i])
                .expect("cached block disappeared between match probe and acquisition");
            cache_entries.entries[idx].was_acquired = true;
            cache_entries.entries[idx].last_access_epoch = access_epoch;
            matched.blocks[i] = cache_entries.entries[idx].block_ref.clone();
        }
        matched
    }

    /// Locations of the probed hits (non-owning).
    pub fn matched_locations(
        &self,
        pool: &Rc<RefCell<BlockPool>>,
        keys: &[CacheKey],
        begin_blocks: i32,
        probe: &GroupPrefixProbe,
    ) -> Vec<CacheBlockLocation> {
        assert!(
            begin_blocks >= 0 && begin_blocks as usize + probe.hits.len() <= keys.len(),
            "matched block range is out of bounds"
        );
        let mut locations = Vec::with_capacity(probe.hits.iter().filter(|&&h| h == 1).count());
        let cache_entries = self.find_cache_entries(pool);
        for (i, hit) in probe.hits.iter().enumerate() {
            if *hit == 0 {
                continue;
            }
            let cache_entries = cache_entries
                .as_ref()
                .expect("cached pool disappeared between match probes");
            let idx = find_entry_by_key(cache_entries, &keys[begin_blocks as usize + i])
                .expect("cached block disappeared between match probes");
            locations.push(
                cache_entries.entries[idx]
                    .block_ref
                    .location()
                    .expect("cached block has a location"),
            );
        }
        locations
    }

    fn validate_key(&self, key: &CacheKey) {
        assert!(
            key.group_id == self.group_id,
            "cache key group does not match index"
        );
        assert!(
            !key.content_hash.is_empty(),
            "cache key content hash must not be empty"
        );
    }

    fn cache_entries_mut(&mut self, pool: &Rc<RefCell<BlockPool>>) -> &mut CacheEntries {
        self.cache_entries_by_pool
            .entry(PoolKey(pool.clone()))
            .or_default()
    }

    fn cache_entries_mut_opt(
        &mut self,
        pool: &Rc<RefCell<BlockPool>>,
    ) -> Option<&mut CacheEntries> {
        self.cache_entries_by_pool.get_mut(&PoolKey(pool.clone()))
    }

    fn find_cache_entries(&self, pool: &Rc<RefCell<BlockPool>>) -> Option<&CacheEntries> {
        self.cache_entries_by_pool.get(&PoolKey(pool.clone()))
    }
}

fn find_entry_by_key(entries: &CacheEntries, key: &CacheKey) -> Option<usize> {
    entries.by_key.get(key).copied()
}

fn find_entry_by_location(entries: &CacheEntries, location: CacheBlockLocation) -> Option<usize> {
    entries.by_location.get(&location).copied()
}

/// Remove `index` from `entries`, repairing the map indices of the element
/// moved into the vacated slot by `swap_remove`.
fn erase_entry(entries: &mut CacheEntries, index: usize) {
    let erased_key = entries.entries[index].key.clone();
    let erased_location = entries.entries[index]
        .block_ref
        .location()
        .expect("cached block has a location");
    entries.by_key.remove(&erased_key);
    entries.by_location.remove(&erased_location);
    let last = entries.entries.len() - 1;
    if index != last {
        entries.entries.swap_remove(index);
        let moved = &entries.entries[index];
        entries.by_key.insert(moved.key.clone(), index);
        entries.by_location.insert(
            moved
                .block_ref
                .location()
                .expect("cached block has a location"),
            index,
        );
    } else {
        entries.entries.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_types::{CacheBoundaryKind, CacheKey};

    fn pool(n: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(n)))
    }

    fn key(group: u32, hash: &str, offset: i32) -> CacheKey {
        CacheKey {
            namespace_id: 0,
            group_id: group,
            content_hash: hash.to_string(),
            page_offset: offset,
        }
    }

    fn block(p: &Rc<RefCell<BlockPool>>, group: u32) -> CacheBlockRef {
        p.borrow_mut().acquire_block(p, group, 1).expect("block")
    }

    #[test]
    fn register_new_entry_then_find_by_key() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b = block(&p, 1);
        index.register(
            &p,
            &mut b,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        assert_eq!(index.num_entries(&p), 1);
        assert!(index.contains(&p, &key(1, "h1", 0)));
        assert!(index.contains_location(&p, b.location().unwrap()));
        let found = index.find(&p, &key(1, "h1", 0));
        assert_eq!(found, b);
        assert_eq!(found.use_count(), 3); // index + local b + found
    }

    #[test]
    fn register_replaces_with_canonical_block() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b1 = block(&p, 1);
        index.register(
            &p,
            &mut b1,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        // Second block with the same key becomes a reference to the canonical block.
        let mut b2 = block(&p, 1);
        index.register(
            &p,
            &mut b2,
            key(1, "h1", 0),
            2,
            1,
            CacheBoundaryKind::Chunk,
            None,
        );
        assert_eq!(b2, b1);
        assert_eq!(b2.location(), b1.location());
        // The original location of b2 is released: only one occupied slot total.
        assert_eq!(p.borrow().num_occupied_slots(), 1);
    }

    #[test]
    fn register_same_location_requires_same_key() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b = block(&p, 1);
        index.register(
            &p,
            &mut b,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        // Same location, same key: updates epoch, no new entry.
        let mut b_again = b.clone();
        index.register(
            &p,
            &mut b_again,
            key(1, "h1", 0),
            5,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        assert_eq!(index.num_entries(&p), 1);
        assert_eq!(
            index
                .metadata_for(&p, b.location().unwrap())
                .unwrap()
                .last_access_epoch,
            5
        );
    }

    #[test]
    #[should_panic(expected = "one cache block location cannot change cache key")]
    fn register_same_location_different_key_panics() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b = block(&p, 1);
        index.register(
            &p,
            &mut b,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        let mut b2 = b.clone();
        index.register(
            &p,
            &mut b2,
            key(1, "h2", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
    }

    #[test]
    fn evict_only_unique_unpinned_entries() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b = block(&p, 1);
        index.register(
            &p,
            &mut b,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        // Pinned by the local ref: evict fails.
        assert!(index.evict(&p, b.location().unwrap()).is_none());
        drop(b);
        // Unique now (index is the only owner): evict succeeds.
        let evicted = index.evict(&p, index.evictable_locations(&p)[0]);
        assert_eq!(evicted, Some(key(1, "h1", 0)));
        assert_eq!(index.num_entries(&p), 0);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
    }

    #[test]
    fn evictable_locations_after_releasing_discounts_ticket_refs() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b = block(&p, 1);
        index.register(
            &p,
            &mut b,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        // Owned by index + local b (2 owners) -> not evictable.
        assert!(index.evictable_locations(&p).is_empty());
        // Discounting the local ref as a released owner makes it evictable.
        let loc = b.location().unwrap();
        assert_eq!(
            index.evictable_locations_after_releasing(&p, &[loc]),
            vec![loc]
        );
    }

    #[test]
    fn acquire_matched_marks_was_acquired_and_returns_owning_refs() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b = block(&p, 1);
        index.register(
            &p,
            &mut b,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        let probe = GroupPrefixProbe { hits: vec![1] };
        let keys = vec![key(1, "h1", 0)];
        let matched = index.acquire_matched(&p, &keys, 0, &probe, 9);
        assert_eq!(matched.num_hit_blocks(), 1);
        assert_eq!(matched.blocks[0], b);
        assert!(
            index
                .metadata_for(&p, b.location().unwrap())
                .unwrap()
                .was_acquired
        );
        assert_eq!(
            index
                .metadata_for(&p, b.location().unwrap())
                .unwrap()
                .last_access_epoch,
            9
        );
    }

    #[test]
    fn matched_locations_returns_hit_locations_in_order() {
        let p = pool(4);
        let mut index = PrefixCacheIndex::new(1);
        let mut b1 = block(&p, 1);
        let mut b2 = block(&p, 1);
        index.register(
            &p,
            &mut b1,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        index.register(
            &p,
            &mut b2,
            key(1, "h2", 0),
            1,
            1,
            CacheBoundaryKind::Chunk,
            None,
        );
        let probe = GroupPrefixProbe {
            hits: vec![1, 0, 1],
        };
        let keys = vec![key(1, "h1", 0), key(1, "hx", 0), key(1, "h2", 0)];
        let locs = index.matched_locations(&p, &keys, 0, &probe);
        assert_eq!(locs, vec![b1.location().unwrap(), b2.location().unwrap()]);
    }

    #[test]
    fn parent_is_fully_evictable_requires_all_children_unpinned() {
        let p = pool(2);
        let mut index = PrefixCacheIndex::new(1);
        let mut b1 = block(&p, 1);
        let mut b2 = block(&p, 1);
        index.register(
            &p,
            &mut b1,
            key(1, "h1", 0),
            1,
            0,
            CacheBoundaryKind::Chunk,
            None,
        );
        index.register(
            &p,
            &mut b2,
            key(1, "h2", 0),
            1,
            1,
            CacheBoundaryKind::Chunk,
            None,
        );
        // Both children pinned by locals -> not fully evictable.
        assert!(!index.parent_is_fully_evictable(&p, 1, 1));
        drop(b1);
        drop(b2);
        // Index is the only owner of every occupied child.
        assert!(index.parent_is_fully_evictable(&p, 1, 1));
    }
}
