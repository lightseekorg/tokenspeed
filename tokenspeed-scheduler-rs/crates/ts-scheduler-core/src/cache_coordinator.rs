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

//! Multi-group prefix-cache coordination over one shared `BlockPool`.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.{h,cpp}`
//! and `cache_admission.cpp`. Holds no per-request state; the request access
//! clock is global, while each request carries its issued epoch.
//!
//! Deviation from C++: the cache-mutation sink is a `Box<dyn FnMut>`. The
//! scheduler must wire it through a shared buffer (e.g. `Rc<RefCell<Vec<..>>>`)
//! instead of a closure capturing the scheduler itself, to avoid a
//! self-referential borrow (see `docs/design/rust-port.md` §5).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::block_pool::BlockPoolHandle;
use crate::block_table::BlockTable;
use crate::cache_block_ref::{CacheBlockLocation, CacheBlockRef};
use crate::cache_group::CacheGroup;
use crate::cache_types::{
    AttnKind, CacheBoundaryKind, CacheGroupSpec, CacheKey, GroupDemand, GroupPrefixProbe,
    PrefixMatch, DEFAULT_CACHE_NAMESPACE_ID,
};
use crate::group_allocator::GroupAllocator;
use crate::group_geometry::GroupGeometry;
use crate::prefix_index::PrefixCacheIndex;
use crate::prefix_matcher::PrefixMatcher;

/// Which physical tier a coordinator operation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    Device,
    Host,
}

/// `num_common_tokens` is in tokens, aligned to the shared prefix granularity.
/// `per_group[i]` is group i's `PrefixMatch` at exactly that length.
#[derive(Debug)]
pub struct CoordinatorMatch {
    pub num_common_tokens: i32,
    pub per_group: Vec<PrefixMatch>,
}

/// Result of one prefix probe (read-only admission snapshot).
#[derive(Debug, Clone, Default)]
pub struct PrefixProbe {
    pub group_keys: Vec<Vec<CacheKey>>,
    pub device: Tier,
    pub host: Tier,
}

/// Per-tier probe outcome.
#[derive(Debug, Clone, Default)]
pub struct Tier {
    pub num_common_tokens: i32,
    /// Common coverage after all prefix-closed groups, before a window/state
    /// group shortens the resumable boundary.
    pub prefix_closed_tokens: i32,
    pub per_group: Vec<GroupPrefixProbe>,
}

/// Outcome of a successful admission.
#[derive(Debug)]
pub struct AdmissionResult {
    pub device_prefix_tokens: i32,
    pub host_prefix_tokens: i32,
    /// Longer prefix-closed coverage worth materializing for non-closed groups.
    pub promotion_boundary_tokens: i32,
    pub access_epoch: u64,
    pub load_pairs: Vec<crate::cache_types::BlockTransfer>,
    /// Fresh device child pages appended by ordinary Acquire, aligned by
    /// group_id. Cache hits and host-loaded destinations are excluded.
    pub new_page_ids: Vec<Vec<i32>>,
}

/// A cache mutation reported through the sink (KV-event tracking).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMutation {
    Stored,
    Removed,
}

/// Pending device->host store candidate.
#[derive(Debug)]
pub struct StoreCandidate {
    pub key: CacheKey,
}

struct AcquiredPrefix {
    device: CoordinatorMatch,
    host: Option<CoordinatorMatch>,
}

/// One cache group per spec (group_id = index), sharing one scheduler prefix
/// domain P while each group may use a smaller cache-page token count.
pub struct CacheCoordinator {
    groups: Vec<CacheGroup>,
    geometry: Vec<GroupGeometry>,
    /// Closed groups first, so non-closed groups match against a settled bound.
    match_order: Vec<usize>,
    pool: BlockPoolHandle,
    host_pool: Option<BlockPoolHandle>,
    stream_device_cache_to_host: bool,
    prefix_granularity: i32,
    next_access_epoch: u64,
    pending_stores: Vec<StoreCandidate>,
    cache_mutation_sink: Option<Box<dyn FnMut(CacheKey, CacheMutation)>>,
}
impl CacheCoordinator {
    /// The Host pool is available to explicit tier operations. `stream`
    /// controls whether ordinary Device prefix publication also feeds Host.
    pub fn new(
        groups: Vec<CacheGroup>,
        prefix_granularity: i32,
        pool: BlockPoolHandle,
        host_pool: Option<BlockPoolHandle>,
        stream_device_cache_to_host: bool,
    ) -> Self {
        assert!(
            prefix_granularity > 0,
            "coordinator needs positive prefix_granularity"
        );
        let mut geometry = Vec::with_capacity(groups.len());
        let mut match_order = Vec::new();
        for (i, group) in groups.iter().enumerate() {
            assert!(
                group.id() == i as u32,
                "cache group id must equal its group index"
            );
            let group_block_granularity = group.spec().block_granularity;
            assert!(
                group_block_granularity > 0 && prefix_granularity % group_block_granularity == 0,
                "group block_granularity must be a positive divisor of the prefix granularity"
            );
            assert!(
                group.allocator().cache_blocks_per_lcm_block()
                    == group.spec().cache_blocks_per_lcm_block,
                "group allocator packing must match its group spec"
            );
            geometry.push(GroupGeometry::new(group_block_granularity));
            if group.matcher().is_prefix_closed() {
                match_order.push(i);
            }
        }
        for (i, group) in groups.iter().enumerate() {
            if !group.matcher().is_prefix_closed() {
                match_order.push(i);
            }
        }
        let stream = stream_device_cache_to_host && host_pool.is_some();
        Self {
            groups,
            geometry,
            match_order,
            pool,
            host_pool,
            stream_device_cache_to_host: stream,
            prefix_granularity,
            next_access_epoch: 0,
            pending_stores: Vec::new(),
            cache_mutation_sink: None,
        }
    }

    pub fn num_groups(&self) -> i32 {
        self.groups.len() as i32
    }

    pub fn prefix_granularity(&self) -> i32 {
        self.prefix_granularity
    }

    pub fn has_mamba_state_group(&self) -> bool {
        self.groups
            .iter()
            .any(|group| group.spec().kind == AttnKind::MambaState)
    }

    pub fn allocator(&self, i: usize) -> &GroupAllocator {
        self.groups[i].allocator()
    }

    pub fn group_kind(&self, i: usize) -> AttnKind {
        self.groups[i].spec().kind
    }

    pub fn group_prefix_index(&self, i: usize) -> &PrefixCacheIndex {
        self.groups[i].index()
    }

    pub fn group_matcher(&self, i: usize) -> &PrefixMatcher {
        self.groups[i].matcher()
    }

    pub fn group_is_prefix_closed(&self, i: usize) -> bool {
        self.groups[i].matcher().is_prefix_closed()
    }

    pub fn group_boundary_lookback_pages(&self, i: usize) -> i32 {
        self.groups[i].matcher().boundary_lookback_pages()
    }

    pub fn group_block_granularity(&self, i: usize) -> i32 {
        self.geometry[i].block_granularity()
    }

    pub fn group_blocks_needed_for(&self, i: usize, table: &BlockTable, num_tokens: i32) -> i32 {
        self.geometry[i].blocks_needed_for(table, num_tokens)
    }

    pub fn group_has_reclaimable_blocks_at(
        &self,
        i: usize,
        table: &BlockTable,
        num_computed_tokens: i32,
    ) -> bool {
        let group = &self.groups[i];
        !group
            .allocator()
            .reclaimable_block_locations_at(
                group.index(),
                table,
                self.group_expired_blocks_at(i, num_computed_tokens),
                &[],
            )
            .is_empty()
    }

    pub fn group_blocks_reclaimable_at(
        &self,
        i: usize,
        table: &BlockTable,
        num_computed_tokens: i32,
        count_uncached: bool,
    ) -> i32 {
        let group = &self.groups[i];
        group.allocator().blocks_reclaimable_at(
            group.index(),
            table,
            self.group_expired_blocks_at(i, num_computed_tokens),
            count_uncached,
        )
    }

    /// ProbePrefix is read-only. Cache state must not change before its result
    /// is passed to `admit`.
    pub fn probe_prefix(&self, content_hashes: &[String]) -> PrefixProbe {
        assert!(
            content_hashes.len() <= (i32::MAX as usize) / self.prefix_granularity as usize,
            "prefix length exceeds int32 token range"
        );
        let num_prefix_pages = content_hashes.len() as i32;
        let mut out = PrefixProbe {
            group_keys: self.build_group_keys(content_hashes),
            device: Tier::default(),
            host: Tier::default(),
        };
        out.device = self.probe_tier_with_keys(
            CacheTier::Device,
            &out.group_keys,
            &self.match_order,
            num_prefix_pages,
            0,
        );
        if self.host_pool.is_some() {
            let device_common = out.device.num_common_tokens;
            out.host = self.probe_tier_with_keys(
                CacheTier::Host,
                &out.group_keys,
                &self.match_order,
                num_prefix_pages,
                device_common,
            );
        }
        out
    }

    /// Decode-side PD reuses local history pages, while final-state groups are
    /// restored from the remote endpoint snapshot. Their aligned null holes do
    /// not count as cache hits.
    pub fn probe_decode_device_prefix(&self, content_hashes: &[String]) -> PrefixProbe {
        assert!(
            content_hashes.len() <= (i32::MAX as usize) / self.prefix_granularity as usize,
            "prefix length exceeds int32 token range"
        );
        let num_prefix_pages = content_hashes.len() as i32;
        let history_match_order: Vec<usize> = self
            .match_order
            .iter()
            .copied()
            .filter(|&i| self.groups[i].spec().kind != AttnKind::MambaState)
            .collect();
        let mut out = PrefixProbe {
            group_keys: self.build_group_keys(content_hashes),
            device: Tier::default(),
            host: Tier::default(),
        };
        let mut device = self.probe_tier_with_keys(
            CacheTier::Device,
            &out.group_keys,
            &history_match_order,
            num_prefix_pages,
            0,
        );
        let covered_tokens = device.num_common_tokens as i64;
        assert!(
            covered_tokens >= 0,
            "decode destination state coverage is negative"
        );
        for i in 0..self.groups.len() {
            if self.groups[i].spec().kind == AttnKind::MambaState {
                let num_holes = covered_tokens / self.geometry[i].block_granularity() as i64;
                assert!(
                    num_holes <= out.group_keys[i].len() as i64,
                    "decode destination state hole count is outside the probed range"
                );
                device.per_group[i].hits.resize(num_holes as usize, 0);
            }
        }
        out.device = device;
        out
    }

    pub fn promotion_boundary_tokens(&self, prefix: &PrefixProbe) -> i32 {
        let matched_tokens = prefix
            .device
            .num_common_tokens
            .max(prefix.host.num_common_tokens);
        let prefix_closed_tokens = prefix
            .device
            .prefix_closed_tokens
            .max(prefix.host.prefix_closed_tokens);
        if prefix_closed_tokens > matched_tokens {
            prefix_closed_tokens
        } else {
            0
        }
    }

    /// Admission feasibility without mutation, optionally discounting in-flight
    /// Store tickets held in `pending_store_releases`.
    pub fn can_admit_after_releasing(
        &self,
        prefix: &PrefixProbe,
        demands: &[GroupDemand<'_>],
        pending_store_releases: &[(u32, CacheBlockLocation)],
    ) -> bool {
        let mut victims = Vec::new();
        AdmissionPlanner::new(
            &self.groups,
            &self.geometry,
            &self.pool,
            demands,
            prefix,
            pending_store_releases,
            &mut victims,
        )
        .plan()
    }

    /// Admit a probed prefix against the group demands. Returns `None` when
    /// capacity is unavailable (the probe is left intact for a
    /// hypothetical-release check). `request_access_epoch` continues an
    /// existing request's epoch; `None` starts a new request.
    pub fn admit<'a>(
        &mut self,
        prefix: PrefixProbe,
        demands: &mut [GroupDemand<'a>],
        request_access_epoch: Option<u64>,
    ) -> Option<AdmissionResult> {
        assert!(
            demands.len() == self.groups.len(),
            "demands/groups size mismatch"
        );
        for demand in demands.iter() {
            assert!(
                demand.table.is_some(),
                "group demand requires a block table"
            );
            assert!(
                demand.new_prefix_hash_begin >= 0
                    && demand.new_prefix_hash_begin as usize <= demand.prefix_hashes.len(),
                "new page hash begin is outside the hash history"
            );
            let has_new_prefix_hashes =
                (demand.new_prefix_hash_begin as usize) < demand.prefix_hashes.len();
            assert!(
                demand.completed_boundary_kind.is_some() == has_new_prefix_hashes,
                "completed boundary kind must match newly completed page hashes"
            );
        }
        let candidate = plan_admission(&self.groups, &self.geometry, &self.pool, prefix, &*demands);
        let plan = candidate?;
        if let Some(epoch) = request_access_epoch {
            assert!(
                epoch > 0 && epoch <= self.next_access_epoch,
                "request access epoch was not issued by this coordinator"
            );
        }
        let access_epoch = match request_access_epoch {
            Some(epoch) => epoch,
            None => {
                self.next_access_epoch += 1;
                self.next_access_epoch
            }
        };
        let promotion_boundary_tokens = self.promotion_boundary_tokens(&plan.prefix);
        let mut acquired_prefix = self.acquire_prefix(plan.prefix, access_epoch);
        let mut result = AdmissionResult {
            device_prefix_tokens: acquired_prefix.device.num_common_tokens,
            host_prefix_tokens: acquired_prefix
                .host
                .as_ref()
                .map_or(0, |h| h.num_common_tokens),
            promotion_boundary_tokens,
            access_epoch,
            load_pairs: Vec::new(),
            new_page_ids: vec![Vec::new(); self.groups.len()],
        };
        if acquired_prefix.device.num_common_tokens > 0 {
            for (i, demand) in demands.iter_mut().enumerate() {
                let table = demand
                    .table
                    .as_deref_mut()
                    .expect("group demand requires a block table");
                let hit = std::mem::take(&mut acquired_prefix.device.per_group[i]);
                self.groups[i].allocator().claim_hit_blocks(table, hit);
            }
        }
        let mut prospective_victims: Vec<(u32, CacheBlockLocation)> =
            Vec::with_capacity(plan.victims.len());
        // A reclaimable table block may still be pinned by both the request and
        // cache here. Evict what is already free, then slide the request tables
        // and retry the blocks whose request reference has just been released.
        for victim in plan.victims {
            if !self.evict_cached_block(victim.0, victim.1) {
                prospective_victims.push(victim);
            }
        }
        for (i, demand) in demands.iter_mut().enumerate() {
            if demand.completed_boundary_kind.is_some() {
                self.cache_device_completed_blocks_for_group(i, demand, access_epoch);
            }
            if demand.num_computed_tokens >= 0 {
                let expired = self.group_expired_blocks_at(i, demand.num_computed_tokens);
                self.groups[i].allocator().reclaim_expired(
                    demand
                        .table
                        .as_deref_mut()
                        .expect("group demand requires a block table"),
                    expired,
                );
            }
        }
        for (group_id, location) in prospective_victims {
            if !self.evict_cached_block(group_id, location) {
                assert!(
                    !self.pool.borrow().is_occupied(location),
                    "admission victim changed before acquisition"
                );
            }
        }
        // Extract host extension blocks per group so the loops below can move
        // them into the table and load_pairs.
        let mut host_blocks_per_group: Vec<Vec<CacheBlockRef>> = match acquired_prefix.host {
            Some(host) => host.per_group.into_iter().map(|pm| pm.blocks).collect(),
            None => vec![Vec::new(); self.groups.len()],
        };
        for i in 0..self.groups.len() {
            let demand = &mut demands[i];
            let plan = {
                let table_ref = demand
                    .table
                    .as_deref()
                    .expect("group demand requires a block table");
                self.geometry[i].plan_acquire_demand(table_ref, demand)
            };
            let table = demand
                .table
                .as_deref_mut()
                .expect("group demand requires a block table");
            let host_blocks = std::mem::take(&mut host_blocks_per_group[i]);
            if !host_blocks.is_empty() {
                self.groups[i].allocator().append_host_extension(
                    &self.pool,
                    table,
                    host_blocks,
                    &mut result.load_pairs,
                );
            }
            let first_new_block = table.num_blocks();
            let acquired = self.groups[i].allocator().acquire(&self.pool, table, &plan);
            assert!(acquired, "admission plan no longer fits the block pool");
            for block in first_new_block..table.num_blocks() {
                let block_ref = &table.blocks[block as usize];
                if block_ref.is_null() {
                    continue;
                }
                result.new_page_ids[i].push(self.groups[i].allocator().resolve_cache_block_id(
                    block_ref.location().expect("non-null block has a location"),
                ));
            }
        }
        Some(result)
    }
}
impl CacheCoordinator {
    pub fn num_newly_releasable_lcm_blocks(&self, tables: &[BlockTable]) -> i32 {
        assert!(
            tables.len() == self.groups.len(),
            "release estimate requires one table per cache group"
        );
        let mut released_by_group: Vec<HashMap2<CacheBlockLocation, ReleasedRefs>> =
            std::iter::repeat_with(HashMap2::default)
                .take(self.groups.len())
                .collect();
        let mut referenced_parents = HashSet::new();
        for (group_id, table) in tables.iter().enumerate() {
            for block_ref in table.blocks() {
                if block_ref.is_null() {
                    continue;
                }
                let location = block_ref.location().expect("non-null block has a location");
                let refs = released_by_group[group_id].entry(location).or_default();
                refs.block_ref = Some(block_ref);
                refs.count += 1;
                referenced_parents.insert(location.lcm_block_id);
            }
        }
        let mut count = 0;
        for parent_id in referenced_parents {
            let group_id = self
                .pool
                .borrow()
                .bound_group(parent_id)
                .expect("request table references an unbound LCM block");
            let index = self.groups[group_id as usize].index();
            let mut parent_becomes_reclaimable = true;
            for location in self.pool.borrow().occupied_locations(parent_id) {
                let Some(released) = released_by_group[group_id as usize].get(&location) else {
                    parent_becomes_reclaimable = false;
                    break;
                };
                let owners = released
                    .block_ref
                    .expect("released ref recorded")
                    .use_count();
                assert!(
                    owners >= released.count,
                    "request-owned reference count exceeds total owners"
                );
                let retained_owners = owners - released.count;
                let allowed_owners = if index.contains_location(&self.pool, location) {
                    1
                } else {
                    0
                };
                if retained_owners != allowed_owners {
                    parent_becomes_reclaimable = false;
                    break;
                }
            }
            if parent_becomes_reclaimable {
                count += 1;
            }
        }
        count
    }

    pub fn num_available_lcm_blocks(&self) -> i32 {
        let mut available = 0;
        for parent_id in 1..=self.pool.borrow().num_lcm_blocks() {
            let group_id = self.pool.borrow().bound_group(parent_id);
            if group_id.is_none()
                || self.groups[group_id.unwrap() as usize]
                    .index()
                    .parent_is_fully_evictable(
                        &self.pool,
                        parent_id,
                        self.groups[group_id.unwrap() as usize]
                            .allocator()
                            .cache_blocks_per_lcm_block(),
                    )
            {
                available += 1;
            }
        }
        available
    }

    pub fn total_lcm_blocks(&self) -> i32 {
        self.pool.borrow().num_lcm_blocks()
    }

    pub fn num_free_host_lcm_blocks(&self) -> i32 {
        self.host_pool
            .as_ref()
            .map_or(0, |p| p.borrow().num_empty_lcm_blocks())
    }

    /// LCM blocks required to place `group_pages[g]` pages for every group g.
    pub fn lcm_blocks_needed_for(&self, group_pages: &[i64]) -> i64 {
        assert!(
            group_pages.len() == self.groups.len(),
            "page demand requires one entry per cache group"
        );
        let mut prefix_blocks = 0i64;
        for (i, pages) in group_pages.iter().enumerate() {
            assert!(*pages >= 0, "group page demand must be non-negative");
            let packing = self.groups[i].allocator().cache_blocks_per_lcm_block() as i64;
            prefix_blocks += (*pages + packing - 1) / packing;
        }
        prefix_blocks
    }

    /// Distinct LCM blocks referenced by the given per-request table sets.
    pub fn num_active_lcm_blocks(&self, request_tables: &[&[BlockTable]]) -> usize {
        let mut active = HashSet::new();
        for tables in request_tables {
            for table in tables.iter() {
                for block_ref in table.blocks() {
                    if !block_ref.is_null() {
                        active.insert(
                            block_ref
                                .location()
                                .expect("non-null block has a location")
                                .lcm_block_id,
                        );
                    }
                }
            }
        }
        active.len()
    }

    /// Free pages (group page units) this group could still place, counting its
    /// partially filled parents and every empty parent.
    pub fn group_available_pages(&self, group_index: usize) -> i32 {
        assert!(
            group_index < self.groups.len(),
            "cache group index out of range"
        );
        let slots_per_parent = self.groups[group_index]
            .allocator()
            .cache_blocks_per_lcm_block();
        let mut available = self.pool.borrow().num_empty_lcm_blocks() * slots_per_parent;
        for id in 1..=self.pool.borrow().num_lcm_blocks() {
            if self.pool.borrow().bound_group(id) == Some(group_index as u32) {
                available += slots_per_parent - self.pool.borrow().occupied_count(id);
            }
        }
        available
    }

    /// Registers an exact range, used for transferred prefix blocks and tests.
    pub fn cache_full_blocks(
        &mut self,
        tables: &mut [BlockTable],
        content_hashes: &[String],
        access_epoch: u64,
        first_slot: i32,
        boundary_kind: CacheBoundaryKind,
    ) {
        assert!(
            tables.len() == self.groups.len(),
            "tables/groups size mismatch"
        );
        if content_hashes.is_empty() {
            return; // hot decode rounds usually fill no page
        }
        for (i, table) in tables.iter_mut().enumerate() {
            let keys = self.keys_for_group(content_hashes, self.groups[i].id());
            let pages_per_prefix_hash =
                self.prefix_granularity / self.geometry[i].block_granularity();
            self.cache_full_blocks_for_group(
                CacheTier::Device,
                i,
                table,
                &keys,
                first_slot * pages_per_prefix_hash,
                access_epoch,
                boundary_kind,
            );
        }
    }

    /// Registers the completed page range per group's boundary contract.
    pub fn cache_completed_blocks(
        &mut self,
        tables: &mut [BlockTable],
        prefix_hashes: &[String],
        access_epoch: u64,
        first_new_prefix_page: i32,
        num_computed_tokens: i32,
        boundary_kind: CacheBoundaryKind,
    ) {
        assert!(
            tables.len() == self.groups.len(),
            "tables/groups size mismatch"
        );
        assert!(
            first_new_prefix_page >= 0 && (first_new_prefix_page as usize) < prefix_hashes.len(),
            "completed page range must be non-empty"
        );
        for (i, table) in tables.iter_mut().enumerate() {
            let mut demand = GroupDemand {
                table: Some(table),
                prefix_hashes,
                new_prefix_hash_begin: first_new_prefix_page,
                completed_boundary_kind: Some(boundary_kind),
                num_computed_tokens,
                ..GroupDemand::default()
            };
            self.cache_device_completed_blocks_for_group(i, &mut demand, access_epoch);
        }
    }

    pub fn reclaim_expired(&mut self, tables: &mut [BlockTable], num_computed_tokens: i32) {
        assert!(
            tables.len() == self.groups.len(),
            "tables/groups size mismatch"
        );
        for (i, table) in tables.iter_mut().enumerate() {
            self.groups[i]
                .allocator()
                .reclaim_expired(table, self.group_expired_blocks_at(i, num_computed_tokens));
        }
    }

    pub fn consume_reserved_tokens(&mut self, tables: &mut [BlockTable], num_tokens: i32) {
        assert!(
            tables.len() == self.groups.len(),
            "tables/groups size mismatch"
        );
        for (i, table) in tables.iter_mut().enumerate() {
            self.groups[i]
                .allocator()
                .consume_reserved_tokens(table, num_tokens);
        }
    }

    pub fn free(&mut self, tables: &mut [BlockTable]) {
        assert!(
            tables.len() == self.groups.len(),
            "tables/groups size mismatch"
        );
        for (i, table) in tables.iter_mut().enumerate() {
            self.groups[i].allocator().free(table);
        }
    }

    /// Clears only the Device prefix index. Returns false without mutation
    /// when any cached block still has an owner outside its prefix index.
    pub fn clear_device_cache(&mut self) -> bool {
        let mut cached_locations: Vec<(u32, CacheBlockLocation)> = Vec::new();
        for group in &self.groups {
            let index = group.index();
            let group_locations = index.evictable_locations(&self.pool);
            if group_locations.len() != index.num_entries(&self.pool) as usize {
                return false;
            }
            for location in group_locations {
                cached_locations.push((group.id(), location));
            }
        }
        self.pending_stores.clear();
        for (group_id, location) in cached_locations {
            assert!(
                self.evict_cached_block(group_id, location),
                "clearable Device cache entry disappeared"
            );
        }
        true
    }

    /// Clears both Device and Host prefix indexes. Returns false without
    /// mutation when either tier still has a pinned cached block.
    pub fn clear_cache(&mut self) -> bool {
        if self.host_pool.is_none() {
            return self.clear_device_cache();
        }
        let mut host_locations: Vec<(u32, CacheBlockLocation)> = Vec::new();
        {
            let host_pool = self.host_pool.as_ref().expect("host pool present");
            for group in &self.groups {
                let index = group.index();
                let group_locations = index.evictable_locations(host_pool);
                if group_locations.len() != index.num_entries(host_pool) as usize {
                    return false;
                }
                for location in group_locations {
                    host_locations.push((group.id(), location));
                }
            }
        }
        // ClearDeviceCache performs its complete pin check before mutation.
        if !self.clear_device_cache() {
            return false;
        }
        for (group_id, location) in host_locations {
            assert!(
                self.groups[group_id as usize]
                    .index_mut()
                    .evict(
                        self.host_pool.as_ref().expect("host pool present"),
                        location
                    )
                    .is_some(),
                "clearable Host cache entry disappeared"
            );
        }
        true
    }

    /// Retry ordinary D2H Store for already-published Device cache entries.
    /// Missing keys and an absent Host tier are silently skipped.
    pub fn queue_cached_blocks_for_store(&mut self, prefix_hashes: &[String]) {
        if self.host_pool.is_none() {
            return;
        }
        for group in &self.groups {
            for key in self.keys_for_group(prefix_hashes, group.id()) {
                if group.index().contains(&self.pool, &key) {
                    self.pending_stores.push(StoreCandidate { key });
                }
            }
        }
    }

    pub fn take_pending_stores(&mut self) -> Vec<StoreCandidate> {
        std::mem::take(&mut self.pending_stores)
    }

    pub fn acquire_device_cached_block(&self, key: &CacheKey) -> CacheBlockRef {
        if key.group_id >= self.groups.len() as u32 {
            return CacheBlockRef::default();
        }
        self.groups[key.group_id as usize]
            .index()
            .find(&self.pool, key)
    }

    /// Acquire a host-tier block for `group_id`, evicting the least valuable
    /// host cache when the pool is full (same-group child first, then a whole
    /// parent from the least valuable group).
    pub fn acquire_host_block(&mut self, group_id: u32) -> CacheBlockRef {
        let host_pool = self
            .host_pool
            .as_ref()
            .expect("AcquireHostBlock requires a host pool");
        assert!(
            group_id < self.groups.len() as u32,
            "Host block group id out of range"
        );
        let target = self.groups[group_id as usize].allocator();
        let packing = target.cache_blocks_per_lcm_block();
        if let Some(block_ref) = host_pool
            .borrow_mut()
            .acquire_block(host_pool, group_id, packing)
        {
            return block_ref;
        }

        // Reusing one child of an already-bound parent destroys less cache
        // than rebinding a complete parent from another group.
        let local_victims = self.groups[group_id as usize]
            .index()
            .evictable_locations(host_pool);
        if let Some(victim) = local_victims.iter().min_by(|&&a, &&b| {
            self.host_cache_value(host_pool, group_id, a)
                .cmp(&self.host_cache_value(host_pool, group_id, b))
        }) {
            assert!(
                self.groups[group_id as usize]
                    .index_mut()
                    .evict(host_pool, *victim)
                    .is_some(),
                "selected Host child is not evictable"
            );
            let block_ref = host_pool
                .borrow_mut()
                .acquire_block(host_pool, group_id, packing)
                .expect("evicting a same-group Host child did not free a placement");
            return block_ref;
        }

        let mut victim_parent: Option<i32> = None;
        let mut victim_value: Option<HostCacheValue> = None;
        for parent_id in 1..=host_pool.borrow().num_lcm_blocks() {
            let Some(bound_group) = host_pool.borrow().bound_group(parent_id) else {
                continue;
            };
            if !self.groups[bound_group as usize]
                .index()
                .parent_is_fully_evictable(
                    host_pool,
                    parent_id,
                    self.groups[bound_group as usize]
                        .allocator()
                        .cache_blocks_per_lcm_block(),
                )
            {
                continue;
            }
            let mut parent_value: Option<HostCacheValue> = None;
            for slot in 0..self.groups[bound_group as usize]
                .allocator()
                .cache_blocks_per_lcm_block()
            {
                let location = CacheBlockLocation {
                    lcm_block_id: parent_id,
                    slot_index: slot,
                };
                if !host_pool.borrow().is_occupied(location) {
                    continue;
                }
                let child_value = self.host_cache_value(host_pool, bound_group, location);
                parent_value = Some(match parent_value {
                    Some(pv) => pv.max(child_value),
                    None => child_value,
                });
            }
            assert!(
                parent_value.is_some(),
                "evictable Host parent has no children"
            );
            if victim_value.is_none() || parent_value < victim_value {
                victim_parent = Some(parent_id);
                victim_value = parent_value;
            }
        }
        let Some(victim_parent) = victim_parent else {
            return CacheBlockRef::default();
        };
        let bound_group = host_pool
            .borrow()
            .bound_group(victim_parent)
            .expect("victim parent must be bound");
        for slot in 0..self.groups[bound_group as usize]
            .allocator()
            .cache_blocks_per_lcm_block()
        {
            let location = CacheBlockLocation {
                lcm_block_id: victim_parent,
                slot_index: slot,
            };
            if host_pool.borrow().is_occupied(location) {
                assert!(
                    self.groups[bound_group as usize]
                        .index_mut()
                        .evict(host_pool, location)
                        .is_some(),
                    "selected Host parent changed before eviction"
                );
            }
        }
        host_pool
            .borrow_mut()
            .acquire_block(host_pool, group_id, packing)
            .expect("evicting a Host parent did not free a placement")
    }

    pub fn streams_device_cache_to_host(&self) -> bool {
        self.stream_device_cache_to_host
    }

    pub fn contains_host_cached_block(&self, key: &CacheKey) -> bool {
        let Some(host_pool) = self.host_pool.as_ref() else {
            return false;
        };
        assert!(
            key.group_id < self.groups.len() as u32,
            "host cache key group id out of range"
        );
        self.groups[key.group_id as usize]
            .index()
            .contains(host_pool, key)
    }

    pub fn is_host_cached_block(&self, location: CacheBlockLocation) -> bool {
        let Some(host_pool) = self.host_pool.as_ref() else {
            return false;
        };
        self.groups
            .iter()
            .any(|group| group.index().contains_location(host_pool, location))
    }

    pub fn num_host_cached_blocks(&self) -> i32 {
        let Some(host_pool) = self.host_pool.as_ref() else {
            return 0;
        };
        self.groups
            .iter()
            .map(|group| group.index().num_entries(host_pool))
            .sum()
    }

    pub fn num_pinned_host_cached_blocks(&self) -> i32 {
        let Some(host_pool) = self.host_pool.as_ref() else {
            return 0;
        };
        self.groups
            .iter()
            .map(|group| group.index().num_pinned_entries(host_pool))
            .sum()
    }

    pub fn cache_host_block(&mut self, block_ref: &mut CacheBlockRef, key: &CacheKey) {
        let host_pool = self
            .host_pool
            .as_ref()
            .expect("CacheHostBlock requires a host pool");
        assert!(
            key.group_id < self.groups.len() as u32,
            "CacheHostBlock group id out of range"
        );
        self.next_access_epoch += 1;
        self.groups[key.group_id as usize].index_mut().register(
            host_pool,
            block_ref,
            key.clone(),
            self.next_access_epoch,
            -1,
            CacheBoundaryKind::Chunk,
            None,
        );
    }

    /// Reports real device-cache entry insertions and removals.
    pub fn set_cache_mutation_sink(
        &mut self,
        sink: Option<Box<dyn FnMut(CacheKey, CacheMutation)>>,
    ) {
        self.cache_mutation_sink = sink;
    }
}

/// `(was_acquired, last_access_epoch, group_id, lcm_block_id, slot_index)` —
/// lexicographic victim preference for host eviction (C++ `HostCacheValue`).
type HostCacheValue = (bool, u64, u32, i32, i32);

/// Per-location released-reference bookkeeping (C++ `ReleasedRefs`).
#[derive(Default)]
struct ReleasedRefs<'a> {
    block_ref: Option<&'a CacheBlockRef>,
    count: u32,
}

/// HashMap alias used by `num_newly_releasable_lcm_blocks`.
type HashMap2<K, V> = std::collections::HashMap<K, V>;
impl CacheCoordinator {
    fn host_cache_value(
        &self,
        host_pool: &BlockPoolHandle,
        candidate_group: u32,
        location: CacheBlockLocation,
    ) -> HostCacheValue {
        let metadata = self.groups[candidate_group as usize]
            .index()
            .metadata_for(host_pool, location)
            .expect("evictable Host block has no cache metadata");
        (
            metadata.was_acquired,
            metadata.last_access_epoch,
            candidate_group,
            location.lcm_block_id,
            location.slot_index,
        )
    }

    fn group_expired_blocks_at(&self, i: usize, num_computed_tokens: i32) -> i32 {
        self.geometry[i].expired_blocks_at(self.groups[i].spec(), num_computed_tokens)
    }

    fn keys_for_group(&self, content_hashes: &[String], group_id: u32) -> Vec<CacheKey> {
        assert!(
            group_id < self.groups.len() as u32,
            "cache key group id out of range"
        );
        let group_block_granularity = self.geometry[group_id as usize].block_granularity();
        let pages_per_prefix_hash = self.prefix_granularity / group_block_granularity;
        let mut keys = Vec::with_capacity(content_hashes.len() * pages_per_prefix_hash as usize);
        for content_hash in content_hashes {
            for offset in 0..pages_per_prefix_hash {
                keys.push(CacheKey {
                    namespace_id: DEFAULT_CACHE_NAMESPACE_ID,
                    group_id,
                    content_hash: content_hash.clone(),
                    page_offset: offset,
                });
            }
        }
        keys
    }

    fn build_group_keys(&self, content_hashes: &[String]) -> Vec<Vec<CacheKey>> {
        let mut group_keys = Vec::with_capacity(self.groups.len());
        for i in 0..self.groups.len() {
            group_keys.push(self.keys_for_group(content_hashes, self.groups[i].id()));
        }
        group_keys
    }

    fn tier_pool(&self, tier: CacheTier) -> &BlockPoolHandle {
        match tier {
            CacheTier::Device => &self.pool,
            CacheTier::Host => self
                .host_pool
                .as_ref()
                .expect("Host cache tier is not configured"),
        }
    }

    fn probe_tier_with_keys(
        &self,
        tier: CacheTier,
        group_keys: &[Vec<CacheKey>],
        match_order: &[usize],
        num_prefix_pages: i32,
        floor_tokens: i32,
    ) -> Tier {
        let pool = self.tier_pool(tier).clone();
        let mut out = Tier {
            num_common_tokens: 0,
            prefix_closed_tokens: 0,
            per_group: vec![GroupPrefixProbe::default(); self.groups.len()],
        };
        if match_order.is_empty() {
            return out;
        }
        let process = |i: usize, bound_tokens: i32| -> i32 {
            let group_block_granularity = self.geometry[i].block_granularity();
            out.per_group[i] = self.groups[i].matcher().probe(
                self.groups[i].index(),
                &pool,
                &group_keys[i],
                floor_tokens / group_block_granularity,
                bound_tokens / group_block_granularity,
            );
            floor_tokens + out.per_group[i].hits.len() as i32 * group_block_granularity
        };
        let boundary = sweep_then_converge(
            match_order,
            &self.groups,
            num_prefix_pages * self.prefix_granularity,
            self.prefix_granularity,
            process,
        );
        // A matcher can find a q-aligned resume point above the final P-aligned
        // boundary. Only acquire the CacheBlocks covered by the shared boundary.
        for (i, probe) in out.per_group.iter_mut().enumerate() {
            let group_block_granularity = self.geometry[i].block_granularity();
            let covered_pages = (boundary.common_tokens - floor_tokens) / group_block_granularity;
            if probe.hits.len() > covered_pages as usize {
                probe.hits.resize(covered_pages as usize, 0);
            }
        }
        out.num_common_tokens = boundary.common_tokens;
        out.prefix_closed_tokens = boundary.prefix_closed_tokens;
        out
    }

    fn acquire_tier_with_keys(
        &mut self,
        tier: CacheTier,
        group_keys: &[Vec<CacheKey>],
        floor_tokens: i32,
        probe: Tier,
        access_epoch: u64,
    ) -> CoordinatorMatch {
        let pool = self.tier_pool(tier).clone();
        let mut out = CoordinatorMatch {
            num_common_tokens: probe.num_common_tokens,
            per_group: Vec::with_capacity(self.groups.len()),
        };
        for (i, keys) in group_keys.iter().enumerate() {
            let floor_pages = floor_tokens / self.geometry[i].block_granularity();
            let matched = self.groups[i].index_mut().acquire_matched(
                &pool,
                keys,
                floor_pages,
                &probe.per_group[i],
                access_epoch,
            );
            out.per_group.push(matched);
        }
        out
    }

    fn acquire_prefix(&mut self, prefix: PrefixProbe, access_epoch: u64) -> AcquiredPrefix {
        let device = self.acquire_tier_with_keys(
            CacheTier::Device,
            &prefix.group_keys,
            0,
            prefix.device,
            access_epoch,
        );
        let host = if self.host_pool.is_some() && !prefix.host.per_group.is_empty() {
            let device_common = device.num_common_tokens;
            Some(self.acquire_tier_with_keys(
                CacheTier::Host,
                &prefix.group_keys,
                device_common,
                prefix.host,
                access_epoch,
            ))
        } else {
            None
        };
        AcquiredPrefix { device, host }
    }

    #[allow(clippy::too_many_arguments)]
    fn cache_full_blocks_for_group(
        &mut self,
        tier: CacheTier,
        group_index: usize,
        table: &mut BlockTable,
        keys: &[CacheKey],
        first_cache_block: i32,
        access_epoch: u64,
        boundary_kind: CacheBoundaryKind,
    ) {
        let mut newly_cached: Vec<(CacheKey, CacheBlockRef)> = Vec::new();
        let inserted: Option<&mut Vec<(CacheKey, CacheBlockRef)>> = match tier {
            CacheTier::Device
                if self.stream_device_cache_to_host || self.cache_mutation_sink.is_some() =>
            {
                Some(&mut newly_cached)
            }
            _ => None,
        };
        let pool = self.tier_pool(tier).clone();
        self.groups[group_index].index_mut().register_full_blocks(
            &pool,
            table,
            keys,
            access_epoch,
            first_cache_block,
            boundary_kind,
            inserted,
        );
        if tier == CacheTier::Host {
            return;
        }
        for (key, _block_ref) in newly_cached {
            if let Some(sink) = self.cache_mutation_sink.as_mut() {
                sink(key.clone(), CacheMutation::Stored);
            }
            if !self.stream_device_cache_to_host {
                continue;
            }
            self.pending_stores.push(StoreCandidate { key });
        }
    }

    fn cache_device_completed_blocks_for_group(
        &mut self,
        group_index: usize,
        demand: &mut GroupDemand<'_>,
        access_epoch: u64,
    ) {
        self.cache_completed_blocks_for_group(CacheTier::Device, group_index, demand, access_epoch);
    }

    fn cache_completed_blocks_for_group(
        &mut self,
        tier: CacheTier,
        group_index: usize,
        demand: &mut GroupDemand<'_>,
        access_epoch: u64,
    ) {
        let pages_per_prefix_hash =
            self.prefix_granularity / self.geometry[group_index].block_granularity();
        let matcher_is_prefix_closed = self.groups[group_index].matcher().is_prefix_closed();
        let kind = self.groups[group_index].spec().kind;
        let num_computed_tokens = demand.num_computed_tokens;
        let boundary_kind = demand
            .completed_boundary_kind
            .expect("completed boundary kind present");
        if matcher_is_prefix_closed {
            let keys = self.keys_for_group(
                &demand.prefix_hashes[demand.new_prefix_hash_begin as usize..],
                self.groups[group_index].id(),
            );
            let table = demand
                .table
                .as_deref_mut()
                .expect("group demand requires a block table");
            self.cache_full_blocks_for_group(
                tier,
                group_index,
                table,
                &keys,
                demand.new_prefix_hash_begin * pages_per_prefix_hash,
                access_epoch,
                boundary_kind,
            );
            return;
        }
        if num_computed_tokens < 0 {
            return;
        }
        // Mamba can publish only a state checkpoint that the kernel materialized
        // exactly at this boundary. SWA pages are ordinary KV, so an unaligned
        // endpoint can still publish its trailing complete-page boundary.
        if kind == AttnKind::MambaState && num_computed_tokens % self.prefix_granularity != 0 {
            return;
        }
        let boundary_cache_block = demand.prefix_hashes.len() as i32 * pages_per_prefix_hash;
        let lookback = self.groups[group_index]
            .matcher()
            .boundary_lookback_pages()
            .min(boundary_cache_block);
        if lookback == 0 {
            return;
        }
        let first_cache_block = boundary_cache_block - lookback;
        let all_keys = self.keys_for_group(demand.prefix_hashes, self.groups[group_index].id());
        let keys = all_keys[first_cache_block as usize..].to_vec();
        let table = demand
            .table
            .as_deref_mut()
            .expect("group demand requires a block table");
        self.cache_full_blocks_for_group(
            tier,
            group_index,
            table,
            &keys,
            first_cache_block,
            access_epoch,
            boundary_kind,
        );
    }

    fn evict_cached_block(&mut self, group_id: u32, location: CacheBlockLocation) -> bool {
        let removed = self.groups[group_id as usize]
            .index_mut()
            .evict(&self.pool, location);
        match removed {
            Some(key) => {
                if let Some(sink) = self.cache_mutation_sink.as_mut() {
                    sink(key, CacheMutation::Removed);
                }
                true
            }
            None => false,
        }
    }
}

/// Build a coordinator from plain specs, one `CacheGroup` per spec.
pub fn make_coordinator(
    specs: &[CacheGroupSpec],
    prefix_granularity: i32,
    pool: BlockPoolHandle,
    host_pool: Option<BlockPoolHandle>,
    stream_device_cache_to_host: bool,
) -> CacheCoordinator {
    assert!(
        !specs.is_empty(),
        "MakeCoordinator requires at least one spec"
    );
    assert!(prefix_granularity > 0, "prefix_granularity must be > 0");
    assert!(
        specs.len() <= i32::MAX as usize,
        "number of cache groups exceeds int32 range"
    );
    let mut groups = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let group_id = i as u32;
        assert!(
            spec.cache_blocks_per_lcm_block > 0,
            "cache_blocks_per_lcm_block must be > 0"
        );
        let group_block_granularity = spec.block_granularity;
        assert!(
            group_block_granularity > 0 && prefix_granularity % group_block_granularity == 0,
            "group block_granularity must be a positive divisor of the prefix granularity"
        );
        let allocator = GroupAllocator::new(spec.cache_blocks_per_lcm_block, group_id);
        let matcher = match spec.kind {
            AttnKind::Full => PrefixMatcher::Full(crate::prefix_matcher::FullAttnMatcher),
            AttnKind::MambaState => PrefixMatcher::Swa(crate::prefix_matcher::SwaMatcher::new(
                group_block_granularity,
                GroupGeometry::MAMBA_STATE_WINDOW,
            )),
            AttnKind::SlidingWindow => {
                assert!(
                    spec.sliding_window > 0,
                    "sliding window group requires a positive window"
                );
                PrefixMatcher::Swa(crate::prefix_matcher::SwaMatcher::new(
                    group_block_granularity,
                    spec.sliding_window,
                ))
            }
        };
        groups.push(CacheGroup::new(spec.clone(), allocator, matcher));
    }
    CacheCoordinator::new(
        groups,
        prefix_granularity,
        pool,
        host_pool,
        stream_device_cache_to_host,
    )
}

struct ConvergedBoundary {
    common_tokens: i32,
    prefix_closed_tokens: i32,
}

/// Sweep match order left-to-right, then re-run non-closed groups until the
/// bound stabilizes. The `process` callback probes one group at `bound_tokens`
/// and returns that group's extent (in tokens).
fn sweep_then_converge<F>(
    order: &[usize],
    groups: &[CacheGroup],
    bound_tokens: i32,
    align_tokens: i32,
    mut process: F,
) -> ConvergedBoundary
where
    F: FnMut(usize, i32) -> i32,
{
    let align_down = |tokens: i32| tokens - tokens % align_tokens;
    let mut bound_tokens = align_down(bound_tokens);
    let mut prefix_closed_tokens = 0;
    for &i in order {
        let extent = process(i, bound_tokens);
        bound_tokens = bound_tokens.min(align_down(extent));
        if groups[i].matcher().is_prefix_closed() {
            prefix_closed_tokens = bound_tokens;
        }
    }
    loop {
        let mut changed = false;
        for &i in order {
            let extent = process(i, bound_tokens);
            if groups[i].matcher().is_prefix_closed() || extent <= bound_tokens {
                continue;
            }
            bound_tokens = bound_tokens.min(align_down(extent));
            changed = true;
        }
        if !changed {
            break;
        }
    }
    ConvergedBoundary {
        common_tokens: bound_tokens,
        prefix_closed_tokens,
    }
}

struct AdmissionPlan {
    prefix: PrefixProbe,
    victims: Vec<(u32, CacheBlockLocation)>,
}
/// Admission feasibility planning with shadow occupancy; never mutates the
/// real pool. Victim selection mirrors the C++ heap: pop the candidate with
/// the smallest eviction key until the plan fits, then restore newest-first
/// and keep only the victims that are truly required.
struct AdmissionPlanner<'a, 'b, 'x> {
    groups: &'a [CacheGroup],
    geometry: &'a [GroupGeometry],
    pool: &'a BlockPoolHandle,
    demands: &'b [GroupDemand<'x>],
    prefix: &'a PrefixProbe,
    pending_store_releases: &'a [(u32, CacheBlockLocation)],
    victims: &'a mut Vec<(u32, CacheBlockLocation)>,
    remaining_occupied: Vec<i32>,
    local_free_slots: Vec<i64>,
    blocks_needed: Vec<i64>,
    empty_parent_count: i64,
    victim_candidates: BinaryHeap<VictimCandidate>,
}

impl<'a, 'b, 'x> AdmissionPlanner<'a, 'b, 'x> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        groups: &'a [CacheGroup],
        geometry: &'a [GroupGeometry],
        pool: &'a BlockPoolHandle,
        demands: &'b [GroupDemand<'x>],
        prefix: &'a PrefixProbe,
        pending_store_releases: &'a [(u32, CacheBlockLocation)],
        victims: &'a mut Vec<(u32, CacheBlockLocation)>,
    ) -> Self {
        Self {
            groups,
            geometry,
            pool,
            demands,
            prefix,
            pending_store_releases,
            victims,
            remaining_occupied: vec![0; pool.borrow().num_lcm_blocks() as usize + 1],
            local_free_slots: vec![0; groups.len()],
            blocks_needed: vec![0; groups.len()],
            empty_parent_count: 0,
            victim_candidates: BinaryHeap::new(),
        }
    }

    fn plan(&mut self) -> bool {
        self.victims.clear();
        self.initialize_capacity();
        if self.fits() {
            return true;
        }
        self.collect_candidates();
        while !self.fits() {
            let Some(candidate) = self.victim_candidates.pop() else {
                return false;
            };
            self.remove_occupant(candidate.group_id, candidate.location);
            self.victims.push((candidate.group_id, candidate.location));
        }
        // Once removing an eviction prefix fits, keeping the entire unpopped
        // tail also fits. Tentatively restore the prefix newest-first using
        // only the planner's shadow occupancy.
        let mut required_victims = Vec::with_capacity(self.victims.len());
        for i in (0..self.victims.len()).rev() {
            let (group_id, location) = self.victims[i];
            self.restore_occupant(group_id, location);
            if !self.fits() {
                self.remove_occupant(group_id, location);
                required_victims.push((group_id, location));
            }
        }
        required_victims.reverse();
        *self.victims = required_victims;
        true
    }

    fn initialize_capacity(&mut self) {
        assert!(
            self.demands.len() == self.groups.len(),
            "demands/groups size mismatch"
        );
        for i in 0..self.groups.len() {
            let demand = &self.demands[i];
            assert!(
                demand.table.is_some(),
                "group demand requires a block table"
            );
            let device_blocks = self.geometry[i].blocks_needed_for_demand(
                demand
                    .table
                    .as_deref()
                    .expect("group demand requires a block table"),
                demand,
            );
            let host_blocks = if self.prefix.host.per_group.is_empty() {
                0
            } else {
                self.prefix.host.per_group[i]
                    .hits
                    .iter()
                    .filter(|&&h| h == 1)
                    .count() as i64
            };
            self.blocks_needed[i] = device_blocks as i64 + host_blocks;
        }
        for parent_id in 1..=self.pool.borrow().num_lcm_blocks() {
            let group_id = self.pool.borrow().bound_group(parent_id);
            let Some(group_id) = group_id else {
                self.empty_parent_count += 1;
                continue;
            };
            assert!(
                group_id < self.groups.len() as u32,
                "LCM parent has invalid group binding"
            );
            let occupied = self.pool.borrow().occupied_count(parent_id);
            let slots = self.groups[group_id as usize]
                .allocator()
                .cache_blocks_per_lcm_block();
            assert!(
                0 < occupied && occupied <= slots,
                "bound LCM parent has invalid occupancy"
            );
            self.remaining_occupied[parent_id as usize] = occupied;
            self.local_free_slots[group_id as usize] += (slots - occupied) as i64;
        }
    }

    fn collect_candidates(&mut self) {
        let mut protected_locations = HashSet::new();
        for i in 0..self.groups.len() {
            let hits = self.groups[i].index().matched_locations(
                self.pool,
                &self.prefix.group_keys[i],
                0,
                &self.prefix.device.per_group[i],
            );
            protected_locations.extend(hits);
        }
        let mut candidates = HashSet::new();
        for i in 0..self.groups.len() {
            let group_id = i as u32;
            let mut group_pending_store_releases = Vec::new();
            for &(released_group_id, location) in self.pending_store_releases {
                if released_group_id == group_id {
                    group_pending_store_releases.push(location);
                }
            }
            for location in self.groups[i]
                .index()
                .evictable_locations_after_releasing(self.pool, &group_pending_store_releases)
            {
                self.add_candidate(&mut candidates, &protected_locations, group_id, location);
            }
            if self.demands[i].num_computed_tokens >= 0 {
                let expired_blocks = self.geometry[i]
                    .expired_blocks_at(self.groups[i].spec(), self.demands[i].num_computed_tokens);
                let table = self.demands[i]
                    .table
                    .as_deref()
                    .expect("group demand requires a block table");
                for location in self.groups[i].allocator().reclaimable_block_locations_at(
                    self.groups[i].index(),
                    table,
                    expired_blocks,
                    &group_pending_store_releases,
                ) {
                    self.add_candidate(&mut candidates, &protected_locations, group_id, location);
                }
            }
        }
    }

    fn add_candidate(
        &mut self,
        candidates: &mut HashSet<CacheBlockLocation>,
        protected_locations: &HashSet<CacheBlockLocation>,
        group_id: u32,
        location: CacheBlockLocation,
    ) {
        if protected_locations.contains(&location) || !candidates.insert(location) {
            return;
        }
        let metadata = self.groups[group_id as usize]
            .index()
            .metadata_for(self.pool, location);
        let last_access_epoch = metadata.map_or(0, |m| m.last_access_epoch);
        let logical_block_index = metadata.map_or(-1, |m| m.logical_block_index);
        let boundary_kind = metadata.map_or(CacheBoundaryKind::Chunk, |m| m.boundary_kind);
        let is_prefix_closed = self.groups[group_id as usize].matcher().is_prefix_closed();
        let is_probationary_boundary = !is_prefix_closed
            && boundary_kind == CacheBoundaryKind::Chunk
            && !metadata.is_some_and(|m| m.was_acquired)
            && logical_block_index >= 0;
        let eviction_tier = if last_access_epoch == 0 {
            EvictionTier::Uncached
        } else if is_probationary_boundary {
            EvictionTier::ProbationaryBoundary
        } else if is_prefix_closed {
            EvictionTier::ClosedPrefix
        } else {
            EvictionTier::EstablishedBoundary
        };
        let mut position_rank = 0i64;
        if is_probationary_boundary {
            // Retain the longer unproven frontier.
            position_rank = logical_block_index as i64;
        } else if is_prefix_closed && logical_block_index >= 0 {
            // Reclaim a closed prefix from its suffix.
            position_rank = -(logical_block_index as i64);
        }
        self.victim_candidates.push(VictimCandidate {
            group_id,
            location,
            last_access_epoch,
            eviction_tier,
            position_rank,
        });
    }

    fn remove_occupant(&mut self, group_id: u32, location: CacheBlockLocation) {
        assert!(
            self.pool.borrow().bound_group(location.lcm_block_id) == Some(group_id),
            "released admission location belongs to another group"
        );
        let occupied = &mut self.remaining_occupied[location.lcm_block_id as usize];
        assert!(*occupied > 0, "admission released the same location twice");
        let slots = self.groups[group_id as usize]
            .allocator()
            .cache_blocks_per_lcm_block();
        if *occupied == 1 {
            self.local_free_slots[group_id as usize] -= (slots - 1) as i64;
            *occupied = 0;
            self.empty_parent_count += 1;
        } else {
            *occupied -= 1;
            self.local_free_slots[group_id as usize] += 1;
        }
    }

    fn restore_occupant(&mut self, group_id: u32, location: CacheBlockLocation) {
        let occupied = &mut self.remaining_occupied[location.lcm_block_id as usize];
        let slots = self.groups[group_id as usize]
            .allocator()
            .cache_blocks_per_lcm_block();
        if *occupied == 0 {
            assert!(
                self.empty_parent_count > 0,
                "restoring an admission victim underflowed empty parents"
            );
            self.empty_parent_count -= 1;
            *occupied = 1;
            self.local_free_slots[group_id as usize] += (slots - 1) as i64;
        } else {
            assert!(
                *occupied < slots,
                "restoring an admission victim overflowed its parent"
            );
            *occupied += 1;
            self.local_free_slots[group_id as usize] -= 1;
        }
    }

    fn fits(&self) -> bool {
        let mut parents_needed = 0i64;
        for i in 0..self.groups.len() {
            let remaining = (self.blocks_needed[i] - self.local_free_slots[i]).max(0);
            let slots = self.groups[i].allocator().cache_blocks_per_lcm_block() as i64;
            parents_needed += (remaining + slots - 1) / slots;
        }
        parents_needed <= self.empty_parent_count
    }
}

/// Eviction priority tiers (declaration order = C++ enum order = eviction
/// order preference).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EvictionTier {
    Uncached,
    ProbationaryBoundary,
    EstablishedBoundary,
    ClosedPrefix,
}

/// One candidate victim with its eviction key. `Ord` is reversed so the
/// `BinaryHeap` pops the smallest eviction key first (oldest epoch first),
/// matching the C++ `make_heap`/`pop_heap` with `evictedAfter`.
#[derive(Debug, Clone, Copy)]
struct VictimCandidate {
    group_id: u32,
    location: CacheBlockLocation,
    last_access_epoch: u64,
    eviction_tier: EvictionTier,
    position_rank: i64,
}

impl VictimCandidate {
    fn eviction_key(&self) -> (u64, EvictionTier, i64, u32, i32, i32) {
        (
            self.last_access_epoch,
            self.eviction_tier,
            self.position_rank,
            self.group_id,
            self.location.lcm_block_id,
            self.location.slot_index,
        )
    }
}

impl PartialEq for VictimCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.eviction_key() == other.eviction_key()
    }
}
impl Eq for VictimCandidate {}
impl PartialOrd for VictimCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for VictimCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: BinaryHeap pops the max, and we want the min eviction key.
        other.eviction_key().cmp(&self.eviction_key())
    }
}

fn plan_admission<'a, 'x>(
    groups: &'a [CacheGroup],
    geometry: &'a [GroupGeometry],
    pool: &'a BlockPoolHandle,
    prefix: PrefixProbe,
    demands: &'a [GroupDemand<'x>],
) -> Option<AdmissionPlan> {
    assert!(
        demands.len() == groups.len(),
        "demands/groups size mismatch"
    );
    let mut victims: Vec<(u32, CacheBlockLocation)> = Vec::new();
    let planned = {
        let mut planner =
            AdmissionPlanner::new(groups, geometry, pool, demands, &prefix, &[], &mut victims);
        planner.plan()
    };
    if !planned {
        return None;
    }
    Some(AdmissionPlan { prefix, victims })
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::block_pool::BlockPool;

    fn pool(n: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(n)))
    }

    fn hash(h: &str) -> String {
        h.to_string()
    }

    fn full_spec(block_granularity: i32) -> CacheGroupSpec {
        CacheGroupSpec {
            kind: AttnKind::Full,
            cache_blocks_per_lcm_block: 1,
            block_granularity,
            ..CacheGroupSpec::default()
        }
    }

    fn demand_for<'a>(
        table: &'a mut BlockTable,
        num_tokens: i32,
        prefix_hashes: &'a [String],
        new_begin: i32,
        computed: i32,
    ) -> GroupDemand<'a> {
        GroupDemand {
            table: Some(table),
            num_tokens,
            prefix_hashes,
            new_prefix_hash_begin: new_begin,
            completed_boundary_kind: (new_begin as usize)
                .lt(&prefix_hashes.len())
                .then_some(CacheBoundaryKind::Chunk),
            num_computed_tokens: computed,
            reserve_tokens: 0,
            materialized_suffix_start: -1,
        }
    }

    #[test]
    fn make_coordinator_builds_groups_and_match_order() {
        let p = pool(8);
        let c = make_coordinator(&[full_spec(4)], 4, p, None, false);
        assert_eq!(c.num_groups(), 1);
        assert_eq!(c.prefix_granularity(), 4);
        assert!(c.group_is_prefix_closed(0));
        assert_eq!(c.group_block_granularity(0), 4);
    }

    #[test]
    fn probe_prefix_with_no_cache_is_empty() {
        let p = pool(8);
        let c = make_coordinator(&[full_spec(4)], 4, p, None, false);
        let prefix = c.probe_prefix(&[hash("a"), hash("b")]);
        assert_eq!(prefix.device.num_common_tokens, 0);
        assert!(prefix.device.per_group[0].hits.is_empty());
    }

    #[test]
    fn admit_without_cache_allocates_pages() {
        let p = pool(8);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), None, false);
        let mut table = BlockTable::default();
        let hashes = vec![hash("a"), hash("b")];
        // First chunk: no pages completed yet (new_begin == hashes.len()).
        let mut demands = vec![demand_for(&mut table, 8, &hashes, 2, 8)];
        let prefix = c.probe_prefix(&hashes);
        let result = c.admit(prefix, &mut demands, None).expect("admission");
        assert_eq!(result.device_prefix_tokens, 0);
        assert_eq!(result.access_epoch, 1);
        assert_eq!(result.new_page_ids[0].len(), 2);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.available_tokens(), 0);
        // Nothing is cached until the forward completes.
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 0);
    }

    #[test]
    fn admit_caches_completed_blocks_and_hits_on_next_probe() {
        let p = pool(8);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), None, false);
        let hashes = vec![hash("a"), hash("b")];
        // First request schedules both pages; nothing is cached yet.
        let mut table = BlockTable::default();
        {
            let mut demands = vec![demand_for(&mut table, 8, &hashes, 2, 8)];
            let prefix = c.probe_prefix(&hashes);
            c.admit(prefix, &mut demands, None).expect("admission");
        }
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 0);
        // The forward completes: register the completed pages (this is what the
        // scheduler's cache operation does after an ExtendResult).
        c.cache_completed_blocks(
            std::slice::from_mut(&mut table),
            &hashes,
            1,
            0,
            8,
            CacheBoundaryKind::Chunk,
        );
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 2);
        // A second request with the same prefix hits both pages.
        let mut table2 = BlockTable::default();
        let mut demands2 = vec![demand_for(&mut table2, 0, &hashes, 2, 8)];
        let prefix2 = c.probe_prefix(&hashes);
        assert_eq!(prefix2.device.num_common_tokens, 8);
        let result = c.admit(prefix2, &mut demands2, None).expect("admission");
        assert_eq!(result.device_prefix_tokens, 8);
        assert_eq!(table2.num_blocks(), 2); // claimed hits
        assert!(result.new_page_ids[0].is_empty());
    }

    #[test]
    fn admit_returns_none_when_capacity_unavailable() {
        let p = pool(1);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), None, false);
        let mut table = BlockTable::default();
        let hashes = vec![hash("a"), hash("b")];
        let mut demands = vec![demand_for(&mut table, 8, &hashes, 0, 8)];
        let prefix = c.probe_prefix(&hashes);
        // One LCM parent cannot hold two blocks.
        assert!(c.admit(prefix, &mut demands, None).is_none());
    }

    #[test]
    fn cache_full_blocks_registers_exact_range() {
        let p = pool(8);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), None, false);
        // Acquire 2 blocks directly.
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 0, 1, 2);
        let mut table = BlockTable::from_blocks(blocks, 0);
        let hashes = vec![hash("a"), hash("b")];
        c.cache_full_blocks(
            std::slice::from_mut(&mut table),
            &hashes,
            1,
            0,
            CacheBoundaryKind::Chunk,
        );
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 2);
        assert!(c.probe_prefix(&hashes).device.num_common_tokens == 8);
    }

    #[test]
    fn clear_cache_removes_all_entries() {
        let p = pool(8);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), None, false);
        let hashes = vec![hash("a"), hash("b")];
        {
            // Register blocks while a request table still pins them; clear must
            // fail, then succeed once the table releases its references.
            let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 0, 1, 2);
            let mut table = BlockTable::from_blocks(blocks, 0);
            c.cache_full_blocks(
                std::slice::from_mut(&mut table),
                &hashes,
                1,
                0,
                CacheBoundaryKind::Chunk,
            );
            assert!(!c.clear_device_cache());
        }
        assert!(c.clear_device_cache());
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 0);
    }

    #[test]
    fn sink_reports_stored_and_removed() {
        let p = pool(8);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), None, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let events2 = events.clone();
        c.set_cache_mutation_sink(Some(Box::new(move |key, mutation| {
            events2
                .borrow_mut()
                .push((key.content_hash.clone(), mutation));
        })));
        let hashes = vec![hash("a"), hash("b")];
        {
            let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 0, 1, 2);
            let mut table = BlockTable::from_blocks(blocks, 0);
            c.cache_full_blocks(
                std::slice::from_mut(&mut table),
                &hashes,
                1,
                0,
                CacheBoundaryKind::Chunk,
            );
        }
        assert_eq!(events.borrow().len(), 2);
        assert!(events
            .borrow()
            .iter()
            .all(|(_, m)| *m == CacheMutation::Stored));
        c.clear_device_cache();
        assert_eq!(events.borrow().len(), 4);
        assert!(events
            .borrow()
            .iter()
            .skip(2)
            .all(|(_, m)| *m == CacheMutation::Removed));
    }

    #[test]
    fn admission_evicts_oldest_cached_blocks_when_capacity_is_short() {
        let p = pool(4);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), None, false);
        let ha = vec![hash("a"), hash("b")];
        let hc = vec![hash("c"), hash("d")];
        let he = vec![hash("e"), hash("f")];
        // Request 1 caches a,b (epoch 1); request 2 caches c,d (epoch 2).
        {
            let mut table = BlockTable::default();
            let mut demands = vec![demand_for(&mut table, 8, &ha, 2, 8)];
            c.admit(c.probe_prefix(&ha), &mut demands, None)
                .expect("admit1");
            c.cache_completed_blocks(
                std::slice::from_mut(&mut table),
                &ha,
                1,
                0,
                8,
                CacheBoundaryKind::Chunk,
            );
        }
        {
            let mut table = BlockTable::default();
            let mut demands = vec![demand_for(&mut table, 8, &hc, 2, 8)];
            c.admit(c.probe_prefix(&hc), &mut demands, None)
                .expect("admit2");
            c.cache_completed_blocks(
                std::slice::from_mut(&mut table),
                &hc,
                2,
                0,
                8,
                CacheBoundaryKind::Chunk,
            );
        }
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 4);
        // Request 3 needs 2 more parents; the pool is full, so the two oldest
        // cached entries (a,b, epoch 1) must be evicted.
        let mut table3 = BlockTable::default();
        let mut demands3 = vec![demand_for(&mut table3, 8, &he, 2, 8)];
        let result = c
            .admit(c.probe_prefix(&he), &mut demands3, None)
            .expect("admit3");
        assert_eq!(result.new_page_ids[0].len(), 2);
        // a,b are gone; c,d remain; e,f are allocated but not yet cached.
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 2);
        assert!(!c.probe_prefix(&ha).device.per_group[0].hits.contains(&1));
        assert!(c.probe_prefix(&hc).device.per_group[0]
            .hits
            .iter()
            .all(|&h| h == 1));
        // Forward 3 completes: register e,f; the index now holds c,d,e,f.
        c.cache_completed_blocks(
            std::slice::from_mut(&mut table3),
            &he,
            3,
            0,
            8,
            CacheBoundaryKind::Chunk,
        );
        assert_eq!(c.group_prefix_index(0).num_entries(&p), 4);
        assert!(c.probe_prefix(&he).device.per_group[0]
            .hits
            .iter()
            .all(|&h| h == 1));
    }

    #[test]
    fn host_tier_probe_and_store() {
        let p = pool(8);
        let hp = pool(8);
        let mut c = make_coordinator(&[full_spec(4)], 4, p.clone(), Some(hp.clone()), true);
        let hashes = vec![hash("a")];
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 0, 1, 1);
        let mut table = BlockTable::from_blocks(blocks, 0);
        c.cache_full_blocks(
            std::slice::from_mut(&mut table),
            &hashes,
            1,
            0,
            CacheBoundaryKind::Chunk,
        );
        // Streaming device -> host queues a store candidate.
        let stores = c.take_pending_stores();
        assert_eq!(stores.len(), 1);
        // Manually place the host block and register it.
        let mut host_block = hp
            .borrow_mut()
            .acquire_block(&hp, 0, 1)
            .expect("host block");
        let key = stores[0].key.clone();
        c.cache_host_block(&mut host_block, &key);
        assert!(c.contains_host_cached_block(&key));
        assert_eq!(c.num_host_cached_blocks(), 1);
        assert!(c.probe_prefix(&hashes).host.num_common_tokens == 4);
    }
}
