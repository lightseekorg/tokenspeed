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

//! The scheduler control plane.
//!
//! Ported from `tokenspeed-scheduler/csrc/scheduler/scheduler.{h,cpp}` and
//! `operations/forward.cpp`. The C++ design stores `Request*` pointers and a
//! raw `CacheCoordinator&`; Rust re-looks-up requests by id inside each helper
//! and shares the coordinator / request-pool / KV-event state through
//! `Rc<RefCell<..>>` handles, keeping every field borrow disjoint.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use crate::block_pool::{BlockPool, BlockPoolHandle};
use crate::block_table::BlockTable;
use crate::cache_config::CacheGroupConfig;
use crate::cache_coordinator::{CacheCoordinator, CacheMutation};
use crate::cache_ops::{align_prefill_chunk, build_block_tables, make_specs_from_config};
use crate::cache_types::{CacheBoundaryKind, CacheGroupSpec, CacheKey};
use crate::events::{
    Event, ExecutionEvent, ExecutionPlan, ForwardBatch, ForwardOperation, Operation,
};
use crate::fsm;
use crate::fsm::{
    CacheProgress, FsmEvent, PrefillSource, ScheduleDecode, SchedulePrefill,
    SchedulePrefillFirstChunk,
};
use crate::kv_events::{hash_kv_block, KvBlockRemovedEvent, KvBlockStoredEvent, KvCacheEvent};
use crate::prefix_hasher::advance_prefix_hashes;
use crate::req_pool_allocator::{ReqPoolAllocator, ReqPoolAllocatorHandle};
use crate::request::Request;
use crate::request_spec::RequestSpec;
use crate::tier::CacheOperation;
use crate::tier::{LoadBackOperation, TierTransferManager, WriteBackOperation};
use crate::types::{Role, SchedulerConfig};

/// KV-event bookkeeping shared between the coordinator's mutation sink and the
/// scheduler (C++ used a callback into `this`; Rust shares this state through
/// `Rc<RefCell<..>>` to avoid a self-referential borrow).
#[derive(Default)]
struct KvEventState {
    events: Vec<KvCacheEvent>,
    hash_progress: HashMap<String, Vec<u64>>,
    pages: HashMap<CacheKey, KvBlockStoredEvent>,
    child_counts: HashMap<CacheKey, i32>,
    entries_per_event_boundary: i32,
}

/// Per-request forward-result accounting.
struct PlanBuildContext<'a> {
    plan: &'a mut ExecutionPlan,
    admission_failed: bool,
    waits_for_store_ack: bool,
    capacity_blocker: Option<String>,
}

impl<'a> PlanBuildContext<'a> {
    fn new(plan: &'a mut ExecutionPlan) -> Self {
        Self {
            plan,
            admission_failed: false,
            waits_for_store_ack: false,
            capacity_blocker: None,
        }
    }
}

/// The scheduler control plane.
pub struct Scheduler {
    config: SchedulerConfig,
    req_pool_allocator: ReqPoolAllocatorHandle,
    // The device/host pools are owned by the coordinator handle; keeping extra
    // fields here would only duplicate ownership (C++ stored raw references).
    coordinator: Rc<RefCell<CacheCoordinator>>,
    tier_transfers: TierTransferManager,
    cache_group_ids: Vec<String>,
    max_single_request_tokens: i32,
    kv: Rc<RefCell<KvEventState>>,

    pending_forward_results: HashMap<String, i32>,
    pd_transfer_pins: HashSet<String>,
    recovery_queue: VecDeque<String>,
    recovery_barrier: Option<String>,
    fused_capacity_drain: bool,

    requests: HashMap<String, Request>,
}

/// Build a coordinator handle for the scheduler.
fn build_coordinator(
    config: &SchedulerConfig,
    device_pool: &BlockPoolHandle,
    host_pool: Option<&BlockPoolHandle>,
) -> CacheCoordinator {
    let specs = make_specs_from_config(config);
    make_coordinator_inner(
        &specs,
        config.prefix_granularity,
        device_pool.clone(),
        host_pool.cloned(),
        config.streams_device_cache_to_host(),
    )
}

fn make_coordinator_inner(
    specs: &[CacheGroupSpec],
    prefix_granularity: i32,
    pool: BlockPoolHandle,
    host_pool: Option<BlockPoolHandle>,
    stream: bool,
) -> CacheCoordinator {
    crate::cache_coordinator::make_coordinator(specs, prefix_granularity, pool, host_pool, stream)
}
impl Scheduler {
    /// Build a scheduler from a validated configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        config
            .validate()
            .expect("Scheduler configuration must validate before construction");
        let req_pool_allocator =
            Rc::new(RefCell::new(ReqPoolAllocator::new(config.max_batch_size)));
        let block_pool = Rc::new(RefCell::new(BlockPool::new(
            config.device_allocator.num_usable_blocks(),
        )));
        let host_pool = if config.has_host_cache() {
            Some(Rc::new(RefCell::new(BlockPool::new(
                config.host_allocator.num_usable_blocks(),
            ))))
        } else {
            None
        };
        let coordinator = Rc::new(RefCell::new(build_coordinator(
            &config,
            &block_pool,
            host_pool.as_ref(),
        )));
        let mut cache_group_ids = Vec::with_capacity(config.cache_groups.len());
        let mut entries_per_event_boundary = 0i32;
        for group in &config.cache_groups {
            cache_group_ids.push(group.group_id.clone());
            let child_entries = config.prefix_granularity / group.block_granularity();
            entries_per_event_boundary = entries_per_event_boundary
                .checked_add(child_entries)
                .expect("Scheduler: cache entries per event boundary exceed int32 range");
        }
        let max_single_request_tokens = {
            let coord = coordinator.borrow();
            calculate_max_single_request_tokens(&config, &coord, coord.total_lcm_blocks())
        };
        let kv = Rc::new(RefCell::new(KvEventState {
            entries_per_event_boundary,
            ..KvEventState::default()
        }));
        let scheduler = Self {
            config,
            req_pool_allocator,
            coordinator: coordinator.clone(),
            tier_transfers: TierTransferManager::new(coordinator),
            cache_group_ids,
            max_single_request_tokens,
            kv,
            pending_forward_results: HashMap::new(),
            pd_transfer_pins: HashSet::new(),
            recovery_queue: VecDeque::new(),
            recovery_barrier: None,
            fused_capacity_drain: false,
            requests: HashMap::new(),
        };
        if scheduler.config.enable_kv_cache_events {
            let kv = scheduler.kv.clone();
            scheduler
                .coordinator
                .borrow_mut()
                .set_cache_mutation_sink(Some(Box::new(move |key, mutation| {
                    handle_cache_mutation(&kv, key, mutation, scheduler_kv_entries(&kv));
                })));
        }
        scheduler
    }

    /// Register requests after validation (empty tokens / duplicate ids /
    /// negative max_new_tokens / token-limit overflow all rejected).
    pub fn submit_requests(&mut self, request_specs: &[RequestSpec]) {
        let mut request_ids = HashSet::new();
        let mut pending_requests: Vec<(String, Request)> = Vec::with_capacity(request_specs.len());
        for spec in request_specs {
            if spec.tokens.is_empty() {
                panic!("Scheduler: request tokens must be non-empty");
            }
            if self.requests.contains_key(&spec.request_id)
                || !request_ids.insert(spec.request_id.clone())
            {
                panic!("Scheduler: duplicate request id '{}'", spec.request_id);
            }
            if spec.max_new_tokens < 0 {
                panic!("Scheduler: max_new_tokens must be non-negative");
            }
            let generation_reserve = if self.config.role == Role::P {
                0
            } else {
                spec.max_new_tokens.max(self.config.decode_input_tokens) as i64
            };
            let token_limit = spec.tokens.len() as i64 + generation_reserve;
            if token_limit > i32::MAX as i64 {
                panic!("Scheduler: request token limit exceeds int32 range");
            }
            if token_limit > self.max_single_request_tokens as i64 {
                panic!("Scheduler: request token limit exceeds cache capacity");
            }
            pending_requests.push((
                spec.request_id.clone(),
                Request::new(spec, self.config.prefix_granularity, self.config.role),
            ));
        }
        for (id, request) in pending_requests {
            assert!(
                self.requests.insert(id, request).is_none(),
                "validated request id became duplicate before insertion"
            );
        }
    }

    pub fn waiting_size(&self) -> usize {
        self.requests
            .values()
            .filter(|r| r.is::<fsm::Submitted>() || r.is::<fsm::Retracted>())
            .count()
    }

    pub fn decoding_size(&self) -> usize {
        self.requests
            .values()
            .filter(|r| r.is::<fsm::Decoding>())
            .count()
    }

    pub fn prefill_size(&self) -> usize {
        self.requests
            .values()
            .filter(|r| r.is::<fsm::Prefilling>() || r.is::<fsm::PrefillDone>())
            .count()
    }

    pub fn available_kv_pages(&self) -> usize {
        self.coordinator.borrow().num_available_lcm_blocks() as usize
    }

    pub fn active_kv_pages(&self) -> usize {
        let request_tables: Vec<&[BlockTable]> = self
            .requests
            .values()
            .filter(|r| {
                r.is::<fsm::Prefilling>() || r.is::<fsm::PrefillDone>() || r.is::<fsm::Decoding>()
            })
            .map(|r| r.block_tables_ref())
            .collect();
        self.coordinator
            .borrow()
            .num_active_lcm_blocks(&request_tables)
    }

    pub fn request_token_size(&self, id: &str) -> i32 {
        self.requests.get(id).map_or(-1, |r| r.token_size())
    }

    pub fn max_single_request_tokens(&self) -> i32 {
        self.max_single_request_tokens
    }

    pub fn cache_group_total_pages(&self, group_id: &str) -> i32 {
        self.config.cache_groups[self.group_index(group_id)].total_pages
    }

    pub fn cache_group_available_pages(&self, group_id: &str) -> i32 {
        self.coordinator
            .borrow()
            .group_available_pages(self.group_index(group_id))
    }

    pub fn pd_transfer_pinned(&self, request_id: &str) -> bool {
        self.pd_transfer_pins.contains(request_id)
    }

    pub fn pool_free_blocks(&self) -> i32 {
        self.coordinator.borrow().num_available_lcm_blocks()
    }

    pub fn host_pool_cached_blocks(&self) -> i32 {
        self.coordinator.borrow().num_host_cached_blocks()
    }

    pub fn host_pool_free_blocks(&self) -> i32 {
        self.coordinator.borrow().num_free_host_lcm_blocks()
    }

    pub fn host_pool_pinned_blocks(&self) -> i32 {
        self.coordinator.borrow().num_pinned_host_cached_blocks()
    }

    pub fn drain_kv_events(&mut self) -> Vec<KvCacheEvent> {
        std::mem::take(&mut self.kv.borrow_mut().events)
    }

    pub fn clear_l1_cache(&mut self) -> bool {
        self.clear_cache_internal(false)
    }

    pub fn clear_cache(&mut self) -> bool {
        self.clear_cache_internal(true)
    }

    fn clear_cache_internal(&mut self, include_host: bool) -> bool {
        let has_live_request = self.requests.values().any(|r| !r.is::<fsm::Finished>());
        let has_pending_forward_results = !self.pending_forward_results.is_empty();
        let has_pd_transfers = !self.pd_transfer_pins.is_empty();
        let has_tier_transfers = self.tier_transfers.has_any_in_flight();
        if has_live_request || has_pending_forward_results || has_pd_transfers || has_tier_transfers
        {
            return false;
        }
        let cleared = if include_host {
            self.coordinator.borrow_mut().clear_cache()
        } else {
            self.coordinator.borrow_mut().clear_device_cache()
        };
        cleared
    }

    fn group_index(&self, group_id: &str) -> usize {
        self.cache_group_ids
            .iter()
            .position(|g| g == group_id)
            .unwrap_or_else(|| panic!("Scheduler: unknown cache group '{group_id}'"))
    }
}

/// Helper accessor used by the KV sink (avoids holding the RefCell borrow).
fn scheduler_kv_entries(kv: &Rc<RefCell<KvEventState>>) -> i32 {
    kv.borrow().entries_per_event_boundary
}

/// KV mutation callback invoked by the coordinator (through a shared handle).
fn handle_cache_mutation(
    kv: &Rc<RefCell<KvEventState>>,
    key: CacheKey,
    mutation: CacheMutation,
    entries_per_event_boundary: i32,
) {
    let mut kv = kv.borrow_mut();
    let prefix_key = CacheKey {
        namespace_id: key.namespace_id,
        group_id: 0,
        content_hash: key.content_hash,
        page_offset: 0,
    };
    match mutation {
        CacheMutation::Stored => {
            let child_count = kv.child_counts.entry(prefix_key.clone()).or_insert(0);
            assert!(
                *child_count < entries_per_event_boundary,
                "duplicate child entry for one KV event boundary"
            );
            *child_count += 1;
            if *child_count == entries_per_event_boundary {
                let page = kv
                    .pages
                    .get(&prefix_key)
                    .expect("cached KV event boundary has no token descriptor")
                    .clone();
                kv.events.push(KvCacheEvent::BlockStored(page));
            }
        }
        CacheMutation::Removed => {
            assert!(
                kv.child_counts.contains_key(&prefix_key),
                "removed KV event boundary was not registered"
            );
            let count = kv.child_counts[&prefix_key];
            assert!(count > 0, "removed KV event boundary was not registered");
            if count == entries_per_event_boundary {
                let page = kv
                    .pages
                    .get(&prefix_key)
                    .expect("removed KV event boundary has no token descriptor");
                let block_hashes = page.block_hashes.clone();
                kv.events
                    .push(KvCacheEvent::BlockRemoved(KvBlockRemovedEvent {
                        block_hashes,
                    }));
            }
            let new_count = count - 1;
            if new_count == 0 {
                kv.child_counts.remove(&prefix_key);
                kv.pages.remove(&prefix_key);
            } else {
                kv.child_counts.insert(prefix_key.clone(), new_count);
            }
        }
    }
}

/// Binary search for the largest single-request token limit that fits.
fn calculate_max_single_request_tokens(
    config: &SchedulerConfig,
    coord: &CacheCoordinator,
    usable_lcm_blocks: i32,
) -> i32 {
    let mut low = 0i64;
    let mut high = i32::MAX as i64;
    while low < high {
        let candidate = low + (high - low + 1) / 2;
        if single_request_lcm_blocks_required(config, coord, candidate) <= usable_lcm_blocks as i64
        {
            low = candidate;
        } else {
            high = candidate - 1;
        }
    }
    low as i32
}

/// LCM blocks required to place the worst single request of `token_limit` tokens.
fn single_request_lcm_blocks_required(
    config: &SchedulerConfig,
    coord: &CacheCoordinator,
    token_limit: i64,
) -> i64 {
    assert!(
        token_limit >= 0,
        "single-request token limit must be non-negative"
    );
    let decode_width = if config.role == Role::P {
        0
    } else {
        config.decode_input_tokens as i64
    };
    let protected_tokens = config.overlap_schedule_depth as i64 * decode_width;
    let max_prompt_tokens = (token_limit - decode_width).max(0);
    let chunk_tokens = config.max_scheduled_tokens as i64;

    let mut group_pages = vec![0i64; coord.num_groups() as usize];
    for (i, pages_slot) in group_pages.iter_mut().enumerate() {
        let block_granularity = coord.group_block_granularity(i) as i64;
        let group = &config.cache_groups[i];
        let local_prefill_peak = |coord: &CacheCoordinator,
                                  i: usize,
                                  max_prompt: i64,
                                  chunk: i64,
                                  decode: i64,
                                  prot: i64,
                                  grain: i64|
         -> i64 {
            if group.is_snapshot_state_group() {
                if token_limit == 0 {
                    return 0;
                }
                let input_lookback = if max_prompt > chunk {
                    coord.group_boundary_lookback_pages(i) as i64
                } else {
                    0
                };
                return 2i64.max(input_lookback + 1);
            }
            let first_prompt = max_prompt.min(chunk);
            let mut pages = ceil_div(first_prompt + decode + prot, grain);
            if max_prompt > chunk {
                let later_prompt = (max_prompt - chunk).min(chunk);
                let lookback_pages = coord.group_boundary_lookback_pages(i) as i64;
                pages = pages.max(lookback_pages + ceil_div(chunk, grain));
                pages = pages.max(lookback_pages + ceil_div(later_prompt + decode + prot, grain));
            }
            pages
        };
        let child_pages = if coord.group_is_prefix_closed(i) {
            ceil_div(token_limit + protected_tokens, block_granularity)
        } else if config.role == Role::D {
            let latest_snapshot = config.enable_pd_cache
                && group.transfer_policy
                    == crate::cache_config::CacheTransferPolicy::LatestSnapshot;
            if latest_snapshot {
                let snapshot_pages = if token_limit == 0 { 0 } else { 1 };
                // One recovery chunk plus its lookback must fit even when old
                // State checkpoints are evictable.
                snapshot_pages.max(local_prefill_peak(
                    coord,
                    i,
                    max_prompt_tokens,
                    chunk_tokens,
                    decode_width,
                    protected_tokens,
                    block_granularity,
                ))
            } else if config.enable_pd_cache
                && group.retention == crate::cache_config::Retention::SlidingWindow
            {
                let dense_pages = ceil_div(token_limit + protected_tokens, block_granularity);
                let window_pages = ceil_div(
                    group
                        .sliding_window_tokens
                        .expect("sliding group has a window") as i64
                        - 1
                        + decode_width
                        + protected_tokens
                        + block_granularity
                        - 1,
                    block_granularity,
                );
                dense_pages.min(coord.group_boundary_lookback_pages(i) as i64 + window_pages)
            } else {
                ceil_div(token_limit + protected_tokens, block_granularity)
            }
        } else {
            local_prefill_peak(
                coord,
                i,
                max_prompt_tokens,
                chunk_tokens,
                decode_width,
                protected_tokens,
                block_granularity,
            )
        };
        *pages_slot = child_pages;
    }
    coord.lcm_blocks_needed_for(&group_pages)
}

fn ceil_div(value: i64, divisor: i64) -> i64 {
    assert!(
        value >= 0 && divisor > 0,
        "ceilDiv requires non-negative value and positive divisor"
    );
    (value + divisor - 1) / divisor
}
// ── Shared scheduling helpers (C++ anonymous namespace in forward.cpp) ─────

struct CompletedPrefixPages {
    first_new_prefix_page: i32,
    boundary_kind: Option<CacheBoundaryKind>,
}

/// One row per table; each demand borrows its table.
fn make_group_demands<'a>(
    tables: &'a mut [BlockTable],
    prototype: crate::cache_types::GroupDemand<'a>,
) -> Vec<crate::cache_types::GroupDemand<'a>> {
    let mut demands = Vec::with_capacity(tables.len());
    for table in tables.iter_mut() {
        demands.push(crate::cache_types::GroupDemand {
            table: Some(table),
            ..prototype
        });
    }
    demands
}

fn make_snapshot_state_prefill_sparse(
    demands: &mut [crate::cache_types::GroupDemand<'_>],
    cache_groups: &[CacheGroupConfig],
    coord: &CacheCoordinator,
    after_tokens: i32,
) {
    assert!(
        demands.len() == cache_groups.len(),
        "demands/cache groups size mismatch"
    );
    assert!(
        after_tokens > 0,
        "snapshot-state prefill requires a positive endpoint"
    );
    for (i, demand) in demands.iter_mut().enumerate() {
        if !cache_groups[i].is_snapshot_state_group() {
            continue;
        }
        let block_granularity = coord.group_block_granularity(i);
        demand.num_tokens = after_tokens;
        demand.materialized_suffix_start = (after_tokens - 1) / block_granularity;
    }
}

fn defer_snapshot_state_decode_reservation(
    demands: &mut [crate::cache_types::GroupDemand<'_>],
    cache_groups: &[CacheGroupConfig],
) {
    assert!(
        demands.len() == cache_groups.len(),
        "demands/cache groups size mismatch"
    );
    for (i, demand) in demands.iter_mut().enumerate() {
        if cache_groups[i].is_snapshot_state_group() {
            demand.reserve_tokens = 0;
        }
    }
}

fn append_completed_prefix_hashes(
    prefix_hashes: &mut Vec<String>,
    prefix_pages: &[Vec<i32>],
    filled_prefix_pages: i32,
) {
    let first_new_prefix_page = prefix_hashes.len() as i32;
    assert!(
        filled_prefix_pages > first_new_prefix_page,
        "caller must pre-check page-hash progress"
    );
    let previous_hash = prefix_hashes.last().cloned().unwrap_or_default();
    let refs: Vec<&[i32]> = prefix_pages.iter().map(|p| p.as_slice()).collect();
    let new_hashes = advance_prefix_hashes(
        &refs,
        first_new_prefix_page as usize,
        &previous_hash,
        filled_prefix_pages as usize,
    );
    prefix_hashes.extend(new_hashes);
}

fn can_consume_reserved_tokens_in_place(
    coord: &CacheCoordinator,
    tables: &[BlockTable],
    num_tokens: i32,
    num_computed_tokens: i32,
) -> bool {
    for (i, table) in tables.iter().enumerate() {
        if coord.group_blocks_needed_for(i, table, num_tokens) != 0
            || coord.group_has_reclaimable_blocks_at(i, table, num_computed_tokens)
        {
            return false;
        }
    }
    true
}

fn consume_completed_boundary_kind(
    cache_progress: &mut CacheProgress,
    num_computed_tokens: i32,
    prefill_size: i32,
) -> CacheBoundaryKind {
    if cache_progress.promotion_boundary_tokens > 0
        && num_computed_tokens >= cache_progress.promotion_boundary_tokens
    {
        let reached_exactly = num_computed_tokens == cache_progress.promotion_boundary_tokens;
        cache_progress.promotion_boundary_tokens = 0;
        if reached_exactly {
            return CacheBoundaryKind::Promoted;
        }
    }
    if num_computed_tokens == prefill_size {
        CacheBoundaryKind::Endpoint
    } else {
        CacheBoundaryKind::Chunk
    }
}

fn update_completed_prefix_hashes(
    request: &Request,
    cache_progress: &mut CacheProgress,
    num_computed_tokens: i32,
    prefix_granularity: i32,
) -> CompletedPrefixPages {
    let first_new_prefix_page = cache_progress.prefix_hashes.len() as i32;
    let filled_prefix_pages = num_computed_tokens / prefix_granularity;
    if filled_prefix_pages > cache_progress.prefix_hashes.len() as i32 {
        let pages = request.full_prefix_pages(false);
        append_completed_prefix_hashes(
            &mut cache_progress.prefix_hashes,
            &pages,
            filled_prefix_pages,
        );
    }
    let boundary_kind = if first_new_prefix_page < cache_progress.prefix_hashes.len() as i32 {
        Some(consume_completed_boundary_kind(
            cache_progress,
            num_computed_tokens,
            request.prefill_size(),
        ))
    } else {
        None
    };
    CompletedPrefixPages {
        first_new_prefix_page,
        boundary_kind,
    }
}

struct AdmissionMatch {
    probe: crate::cache_coordinator::PrefixProbe,
    candidate_prefix_hashes: Vec<String>,
    extension_hashes: Vec<String>,
    prefix_hashes: Vec<String>,
}
impl Scheduler {
    fn match_prefix_at_admission(&self, request_id: &str) -> AdmissionMatch {
        let request = self.requests.get(request_id).expect("request exists");
        let prefix_granularity = self.coordinator.borrow().prefix_granularity();
        let replay_tokens = self.config.prefix_replay_tokens.max(1);
        let max_cacheable_tokens = (request.prefill_size() - replay_tokens).max(0);
        let probe_prefix_pages = max_cacheable_tokens / prefix_granularity;
        let candidate_prefix_pages = ((request.prefill_size() - 1) / prefix_granularity).max(0);
        let mut prefix_pages = request.full_prefix_pages(false);
        prefix_pages.truncate(candidate_prefix_pages as usize);
        let page_refs: Vec<&[i32]> = prefix_pages.iter().map(|p| p.as_slice()).collect();
        let hashes = crate::prefix_hasher::compute_prefix_hashes(&page_refs, "", &[]);
        let probe_hashes: Vec<String> = hashes
            .iter()
            .take(hashes.len().min(probe_prefix_pages as usize))
            .cloned()
            .collect();

        let mut match_result = AdmissionMatch {
            probe: crate::cache_coordinator::PrefixProbe::default(),
            candidate_prefix_hashes: hashes.clone(),
            extension_hashes: Vec::new(),
            prefix_hashes: Vec::new(),
        };
        let is_retracted = request.is::<fsm::Retracted>();
        let probe = |coord: &Rc<RefCell<CacheCoordinator>>,
                     hashes: &[String],
                     is_retracted: bool|
         -> crate::cache_coordinator::PrefixProbe {
            if self.config.role == Role::D && !is_retracted {
                coord.borrow().probe_decode_device_prefix(hashes)
            } else {
                coord.borrow().probe_prefix(hashes)
            }
        };
        if self.config.disable_prefix_cache && !is_retracted {
            match_result.probe = probe(&self.coordinator, &[], is_retracted);
            return match_result;
        }
        match_result.probe = probe(&self.coordinator, &probe_hashes, is_retracted);
        let hit_prefix_pages = match_result
            .probe
            .device
            .num_common_tokens
            .max(match_result.probe.host.num_common_tokens)
            / prefix_granularity;
        match_result.prefix_hashes = hashes
            .iter()
            .take(hit_prefix_pages as usize)
            .cloned()
            .collect();
        let extension_pages = (match_result.probe.host.num_common_tokens
            - match_result.probe.device.num_common_tokens)
            .max(0)
            / prefix_granularity;
        let extension_begin = match_result.probe.device.num_common_tokens / prefix_granularity;
        match_result.extension_hashes = hashes
            .iter()
            .skip(extension_begin as usize)
            .take(extension_pages as usize)
            .cloned()
            .collect();
        match_result
    }

    fn admit(
        &mut self,
        context: &mut PlanBuildContext,
        prefix: crate::cache_coordinator::PrefixProbe,
        demands: &mut [crate::cache_types::GroupDemand<'_>],
        request_access_epoch: Option<u64>,
    ) -> Option<crate::cache_coordinator::AdmissionResult> {
        let result =
            self.coordinator
                .borrow_mut()
                .admit(prefix.clone(), demands, request_access_epoch);
        match result {
            Some(r) => {
                assert!(
                    r.new_page_ids.len() == self.cache_group_ids.len(),
                    "admission fresh-page groups must match scheduler config"
                );
                for (i, page_ids) in r.new_page_ids.iter().enumerate() {
                    context
                        .plan
                        .pages_to_zero
                        .entry(self.cache_group_ids[i].clone())
                        .or_default()
                        .extend(page_ids.iter().copied());
                }
                Some(r)
            }
            None => {
                context.admission_failed = true;
                if self.tier_transfers.has_stores_in_flight() {
                    let pending_store_releases =
                        self.tier_transfers.device_locations_released_on_store_ack();
                    context.waits_for_store_ack = context.waits_for_store_ack
                        || self.coordinator.borrow().can_admit_after_releasing(
                            &prefix,
                            demands,
                            &pending_store_releases,
                        );
                }
                None
            }
        }
    }

    fn admit_with_kv_event_tracking(
        &mut self,
        context: &mut PlanBuildContext,
        request_id: &str,
        cache_progress: &CacheProgress,
        new_prefix_hash_begin: i32,
        demands: &mut [crate::cache_types::GroupDemand<'_>],
    ) -> bool {
        let event_keys = self.register_kv_event_prefix_pages(
            request_id,
            &cache_progress.prefix_hashes,
            new_prefix_hash_begin,
        );
        let probe = self.coordinator.borrow().probe_prefix(&[]);
        let admitted = self
            .admit(context, probe, demands, Some(cache_progress.access_epoch))
            .is_some();
        self.discard_uncached_kv_event_pages(&event_keys);
        admitted
    }

    fn schedule_prefill_first_chunk(
        &mut self,
        context: &mut PlanBuildContext,
        request_id: &str,
        remaining: i32,
        decode_input_tokens: i32,
    ) -> Option<SchedulePrefillFirstChunk> {
        if self.req_pool_allocator.borrow().available_slots() == 0 {
            return None;
        }
        let match_result = self.match_prefix_at_admission(request_id);
        let coord = self.coordinator.borrow();
        let hit_tokens = match_result
            .probe
            .device
            .num_common_tokens
            .max(match_result.probe.host.num_common_tokens);
        let promotion_boundary_tokens = coord.promotion_boundary_tokens(&match_result.probe);
        assert!(
            promotion_boundary_tokens == 0
                || (promotion_boundary_tokens % coord.prefix_granularity() == 0
                    && promotion_boundary_tokens > hit_tokens
                    && promotion_boundary_tokens
                        < self.requests.get(request_id).unwrap().prefill_size()),
            "promotion boundary must be page-aligned and inside the unmatched prompt"
        );
        let request = self.requests.get(request_id).unwrap();
        let unscheduled = request.prefill_size() - hit_tokens;
        let mut tokens_this_round = remaining.min(unscheduled);
        if coord.has_mamba_state_group() || promotion_boundary_tokens > 0 {
            tokens_this_round = align_prefill_chunk(
                hit_tokens,
                unscheduled,
                remaining,
                coord.prefix_granularity(),
                promotion_boundary_tokens,
            );
            if tokens_this_round == 0 {
                return None;
            }
        }
        let completes_prefill = tokens_this_round == unscheduled;
        let decode_reserve = if completes_prefill {
            decode_input_tokens
        } else {
            0
        };
        let source = if self.config.role == Role::D && request.is::<fsm::Submitted>() {
            PrefillSource::Remote
        } else {
            PrefillSource::Local
        };
        let num_groups = coord.num_groups() as usize;
        let prefix_granularity = coord.prefix_granularity();
        drop(coord);
        let mut tables: Vec<BlockTable> = (0..num_groups).map(|_| BlockTable::default()).collect();
        let mut demands = make_group_demands(
            &mut tables,
            crate::cache_types::GroupDemand {
                table: None,
                num_tokens: tokens_this_round,
                prefix_hashes: &[],
                new_prefix_hash_begin: 0,
                completed_boundary_kind: None,
                num_computed_tokens: -1,
                reserve_tokens: decode_reserve,
                materialized_suffix_start: -1,
            },
        );
        if source == PrefillSource::Local {
            let coord = self.coordinator.borrow();
            make_snapshot_state_prefill_sparse(
                &mut demands,
                &self.config.cache_groups,
                &coord,
                hit_tokens + tokens_this_round,
            );
            drop(coord);
        }
        if self.config.enable_pd_cache && source == PrefillSource::Remote {
            let coord = self.coordinator.borrow();
            for (i, demand) in demands.iter_mut().enumerate() {
                let group = &self.config.cache_groups[i];
                let block_granularity = coord.group_block_granularity(i);
                let prefill_size = self.requests.get(request_id).unwrap().prefill_size();
                if group.transfer_policy == crate::cache_config::CacheTransferPolicy::LatestSnapshot
                {
                    demand.num_tokens = prefill_size;
                    demand.materialized_suffix_start = (prefill_size - 1) / block_granularity;
                } else if group.retention == crate::cache_config::Retention::SlidingWindow {
                    let retained_begin = (prefill_size
                        - group
                            .sliding_window_tokens
                            .expect("sliding group has a window")
                        + 1)
                    .max(0);
                    demand.num_tokens = prefill_size;
                    demand.materialized_suffix_start =
                        (hit_tokens / block_granularity).max(retained_begin / block_granularity);
                }
            }
            drop(coord);
        }
        defer_snapshot_state_decode_reservation(&mut demands, &self.config.cache_groups);
        let event_keys = self.register_kv_event_prefix_pages(
            request_id,
            &match_result.candidate_prefix_hashes,
            0,
        );
        let admission = self.admit(context, match_result.probe, &mut demands, None);
        let Some(admission) = admission else {
            context.capacity_blocker = Some(request_id.to_string());
            self.discard_uncached_kv_event_pages(&event_keys);
            return None;
        };
        assert!(
            admission.promotion_boundary_tokens == promotion_boundary_tokens,
            "promotion boundary changed between probe and admission"
        );
        if !match_result.extension_hashes.is_empty() {
            let mut coord = self.coordinator.borrow_mut();
            coord.cache_full_blocks(
                &mut tables,
                &match_result.extension_hashes,
                admission.access_epoch,
                admission.device_prefix_tokens / prefix_granularity,
                CacheBoundaryKind::Chunk,
            );
        }
        self.discard_uncached_kv_event_pages(&event_keys);
        Some(SchedulePrefillFirstChunk {
            tokens_this_round,
            reserve_num_tokens_in_next_schedule_event: decode_reserve,
            req_pool_allocator: self.req_pool_allocator.clone(),
            source,
            block_tables: tables,
            hit_tokens,
            cache_progress: CacheProgress {
                prefix_hashes: match_result.prefix_hashes,
                access_epoch: admission.access_epoch,
                promotion_boundary_tokens: admission.promotion_boundary_tokens,
            },
            load_pairs: admission.load_pairs,
        })
    }
}

impl Scheduler {
    fn schedule_prefill(
        &mut self,
        context: &mut PlanBuildContext,
        request_id: &str,
        remaining: i32,
        reserve_num_tokens_in_next_schedule_event: i32,
    ) -> Option<SchedulePrefill> {
        let request = self.requests.get(request_id).unwrap();
        let unscheduled = request.unscheduled_prefill_size();
        let first_pos = request.prefill_size() - unscheduled;
        let mut cache_progress = request.cache_progress();
        let mut tokens_this_round = remaining.min(unscheduled);
        let coord = self.coordinator.borrow();
        if coord.has_mamba_state_group() || cache_progress.promotion_boundary_tokens > 0 {
            tokens_this_round = align_prefill_chunk(
                first_pos,
                unscheduled,
                remaining,
                coord.prefix_granularity(),
                cache_progress.promotion_boundary_tokens,
            );
            if tokens_this_round == 0 {
                return None;
            }
        }
        let prefix_granularity = coord.prefix_granularity();
        drop(coord);
        let completes_prefill = tokens_this_round == unscheduled;
        let decode_reserve = if completes_prefill {
            reserve_num_tokens_in_next_schedule_event
        } else {
            0
        };
        let previous = request.current_prefill_info();
        let num_computed_tokens = previous.already_scheduled_len + previous.extend_len;
        let completed = update_completed_prefix_hashes(
            request,
            &mut cache_progress,
            num_computed_tokens,
            prefix_granularity,
        );
        let is_local_prefill =
            self.requests.get(request_id).unwrap().prefill_source() == PrefillSource::Local;
        let mut owned_tables = std::mem::take(
            self.requests
                .get_mut(request_id)
                .unwrap()
                .block_tables_ref_mut(),
        );
        let mut demands = make_group_demands(
            &mut owned_tables,
            crate::cache_types::GroupDemand {
                table: None,
                num_tokens: tokens_this_round,
                prefix_hashes: &cache_progress.prefix_hashes,
                new_prefix_hash_begin: completed.first_new_prefix_page,
                completed_boundary_kind: completed.boundary_kind,
                num_computed_tokens,
                reserve_tokens: decode_reserve,
                materialized_suffix_start: -1,
            },
        );
        if is_local_prefill {
            let coord = self.coordinator.borrow();
            make_snapshot_state_prefill_sparse(
                &mut demands,
                &self.config.cache_groups,
                &coord,
                first_pos + tokens_this_round,
            );
            drop(coord);
        }
        defer_snapshot_state_decode_reservation(&mut demands, &self.config.cache_groups);
        let admitted = self.admit_with_kv_event_tracking(
            context,
            request_id,
            &cache_progress,
            completed.first_new_prefix_page,
            &mut demands,
        );
        *self
            .requests
            .get_mut(request_id)
            .unwrap()
            .block_tables_ref_mut() = owned_tables;
        if !admitted {
            context.capacity_blocker = Some(request_id.to_string());
            return None;
        }
        Some(SchedulePrefill {
            tokens_this_round,
            reserve_num_tokens_in_next_schedule_event: decode_reserve,
            cache_progress,
        })
    }

    fn schedule_decode(
        &mut self,
        context: &mut PlanBuildContext,
        request_id: &str,
    ) -> Option<ScheduleDecode> {
        let reserve_tokens = self
            .requests
            .get(request_id)
            .unwrap()
            .reserve_num_tokens_in_next_schedule_event();
        let mut cache_progress = self.requests.get(request_id).unwrap().cache_progress();
        let num_computed_tokens = {
            let request = self.requests.get(request_id).unwrap();
            if request.is::<fsm::PrefillDone>() {
                let previous = request.current_prefill_info();
                previous.already_scheduled_len + previous.extend_len
            } else {
                request.token_size() - self.config.decode_input_tokens
            }
        };
        let prefix_granularity = self.coordinator.borrow().prefix_granularity();
        let completed = {
            let request = self.requests.get(request_id).unwrap();
            update_completed_prefix_hashes(
                request,
                &mut cache_progress,
                num_computed_tokens,
                prefix_granularity,
            )
        };
        let can_consume = {
            let coord = self.coordinator.borrow();
            let tables = self.requests.get(request_id).unwrap().block_tables_ref();
            completed.first_new_prefix_page == cache_progress.prefix_hashes.len() as i32
                && can_consume_reserved_tokens_in_place(
                    &coord,
                    tables,
                    reserve_tokens,
                    num_computed_tokens,
                )
        };
        if can_consume {
            let tables = self
                .requests
                .get_mut(request_id)
                .unwrap()
                .block_tables_ref_mut();
            self.coordinator
                .borrow_mut()
                .consume_reserved_tokens(tables, reserve_tokens);
        } else {
            let mut owned_tables = std::mem::take(
                self.requests
                    .get_mut(request_id)
                    .unwrap()
                    .block_tables_ref_mut(),
            );
            let mut demands = make_group_demands(
                &mut owned_tables,
                crate::cache_types::GroupDemand {
                    table: None,
                    num_tokens: reserve_tokens,
                    prefix_hashes: &cache_progress.prefix_hashes,
                    new_prefix_hash_begin: completed.first_new_prefix_page,
                    completed_boundary_kind: completed.boundary_kind,
                    num_computed_tokens,
                    reserve_tokens: 0,
                    materialized_suffix_start: -1,
                },
            );
            let admitted = self.admit_with_kv_event_tracking(
                context,
                request_id,
                &cache_progress,
                completed.first_new_prefix_page,
                &mut demands,
            );
            *self
                .requests
                .get_mut(request_id)
                .unwrap()
                .block_tables_ref_mut() = owned_tables;
            if !admitted {
                context.capacity_blocker = Some(request_id.to_string());
                return None;
            }
        }
        Some(ScheduleDecode {
            decode_input_tokens: self.config.decode_input_tokens,
            cache_progress,
        })
    }
}
impl Scheduler {
    fn apply_prefill_event(
        &mut self,
        request_id: &str,
        event: FsmEvent,
    ) -> crate::events::PrefillOperation {
        let source = match &event {
            FsmEvent::SchedulePrefillFirstChunk(ev) => ev.source,
            _ => self.requests.get(request_id).unwrap().prefill_source(),
        };
        {
            let request = self.requests.get_mut(request_id).unwrap();
            let mut coord = self.coordinator.borrow_mut();
            request.apply(event, &mut coord);
        }
        let request = self.requests.get(request_id).unwrap();
        let info = request.current_prefill_info();
        let mut operation = crate::events::PrefillOperation {
            base: crate::events::ForwardOperationBase {
                request_id: request.id().to_string(),
                request_pool_index: request.request_pool_index(),
                input_length: info.extend_len,
                prefill_length: request.prefill_size(),
                block_tables: Default::default(),
            },
            input_ids: info.input_ids,
            shifted_input_ids: info.shifted_input_ids,
            extend_prefix_len: info.already_scheduled_len,
            local_prefill: source == PrefillSource::Local,
        };
        let coord = self.coordinator.borrow();
        operation.base.block_tables =
            build_block_tables(&coord, request.block_tables_ref(), &self.cache_group_ids);
        operation
    }

    fn apply_decode_event(
        &mut self,
        request_id: &str,
        event: FsmEvent,
    ) -> crate::events::DecodeOperation {
        {
            let request = self.requests.get_mut(request_id).unwrap();
            let mut coord = self.coordinator.borrow_mut();
            request.apply(event, &mut coord);
        }
        let request = self.requests.get(request_id).unwrap();
        let mut operation = crate::events::DecodeOperation {
            base: crate::events::ForwardOperationBase {
                request_id: request.id().to_string(),
                request_pool_index: request.request_pool_index(),
                input_length: self.config.decode_input_tokens,
                prefill_length: request.prefill_size(),
                block_tables: Default::default(),
            },
            decode_input_id: -1,
        };
        let coord = self.coordinator.borrow();
        operation.base.block_tables =
            build_block_tables(&coord, request.block_tables_ref(), &self.cache_group_ids);
        operation
    }

    fn apply_event_and_build_operation_first_chunk(
        &mut self,
        request_id: &str,
        event: SchedulePrefillFirstChunk,
        load_back_operations: &mut Vec<LoadBackOperation>,
    ) -> crate::events::PrefillOperation {
        let load_pairs = event.load_pairs.clone();
        let operation =
            self.apply_prefill_event(request_id, FsmEvent::SchedulePrefillFirstChunk(event));
        if load_pairs.is_empty() {
            return operation;
        }
        load_back_operations.push(self.tier_transfers.start_prefix_load(load_pairs));
        operation
    }

    fn apply_event_and_build_operation_prefill(
        &mut self,
        request_id: &str,
        event: SchedulePrefill,
    ) -> crate::events::PrefillOperation {
        self.apply_prefill_event(request_id, FsmEvent::SchedulePrefill(event))
    }

    fn apply_event_and_build_operation_decode(
        &mut self,
        request_id: &str,
        event: ScheduleDecode,
    ) -> crate::events::DecodeOperation {
        let needs_bootstrap_token = self
            .requests
            .get(request_id)
            .unwrap()
            .is::<fsm::PrefillDone>()
            && self.config.role == Role::D;
        let bootstrap_token = if needs_bootstrap_token {
            self.requests.get(request_id).unwrap().last_token()
        } else {
            -1
        };
        let mut operation = self.apply_decode_event(request_id, FsmEvent::ScheduleDecode(event));
        if needs_bootstrap_token {
            operation.decode_input_id = bootstrap_token;
        }
        operation
    }

    fn begin_retraction(&mut self, request_id: &str) -> Option<WriteBackOperation> {
        let mut cache_progress = self.requests.get(request_id).unwrap().cache_progress();
        let num_computed_tokens =
            self.requests.get(request_id).unwrap().token_size() - self.config.decode_input_tokens;
        let prefix_granularity = self.coordinator.borrow().prefix_granularity();
        let completed = {
            let request = self.requests.get(request_id).unwrap();
            update_completed_prefix_hashes(
                request,
                &mut cache_progress,
                num_computed_tokens,
                prefix_granularity,
            )
        };
        if let Some(boundary_kind) = completed.boundary_kind {
            let tables = self
                .requests
                .get_mut(request_id)
                .unwrap()
                .block_tables_ref_mut();
            let mut coord = self.coordinator.borrow_mut();
            coord.cache_completed_blocks(
                tables,
                &cache_progress.prefix_hashes,
                cache_progress.access_epoch,
                completed.first_new_prefix_page,
                num_computed_tokens,
                boundary_kind,
            );
        }
        self.coordinator
            .borrow_mut()
            .queue_cached_blocks_for_store(&cache_progress.prefix_hashes);
        let write_back = self.tier_transfers.start_pending_stores();
        {
            let request = self.requests.get_mut(request_id).unwrap();
            let mut coord = self.coordinator.borrow_mut();
            request.apply(FsmEvent::Retraction, &mut coord);
        }
        self.recovery_queue.push_back(request_id.to_string());
        write_back
    }

    fn retract_for_capacity(
        &mut self,
        context: &PlanBuildContext,
        candidates: &[String],
        write_back_operations: &mut Vec<WriteBackOperation>,
    ) {
        if (self.config.role != Role::Fused && self.config.role != Role::D)
            || !context.admission_failed
            || !self.pending_forward_results.is_empty()
            || !self.pd_transfer_pins.is_empty()
            || context.waits_for_store_ack
            || self.tier_transfers.has_load_backs_in_flight()
        {
            return;
        }
        if self.config.role == Role::D {
            let mut victim: Option<String> = None;
            let mut victim_rank: Option<(i32, i32, String)> = None;
            for id in candidates {
                let request = self.requests.get(id).unwrap();
                if !request.is::<fsm::Decoding>() {
                    continue;
                }
                let tables = request.block_tables_ref();
                let rank = (
                    -self
                        .coordinator
                        .borrow()
                        .num_newly_releasable_lcm_blocks(tables),
                    request.token_size(),
                    id.clone(),
                );
                if victim_rank.is_none() || rank < *victim_rank.as_ref().unwrap() {
                    victim = Some(id.clone());
                    victim_rank = Some(rank);
                }
            }
            let victim = victim.expect(
                "cache admission failed without a retractable Decode request or asynchronous capacity release",
            );
            self.recovery_barrier = context.capacity_blocker.clone();
            if self.recovery_barrier.is_none()
                || self.recovery_barrier.as_deref() == Some(victim.as_str())
            {
                let waiting = candidates.iter().find(|id| {
                    *id != &victim && {
                        let request = self.requests.get(*id).unwrap();
                        request.is::<fsm::Submitted>() || request.is::<fsm::Prefilling>()
                    }
                });
                self.recovery_barrier = waiting.cloned();
            }
            if let Some(operation) = self.begin_retraction(&victim) {
                write_back_operations.push(operation);
            }
            return;
        }
        let mut request_to_retract: Option<String> = None;
        for id in candidates {
            let request = self.requests.get(id).unwrap();
            if (request.is::<fsm::Decoding>() || request.is::<fsm::PrefillDone>())
                && (request_to_retract.is_none()
                    || request.token_size()
                        > self
                            .requests
                            .get(request_to_retract.as_ref().unwrap())
                            .unwrap()
                            .token_size())
            {
                request_to_retract = Some(id.clone());
            }
        }
        let request_to_retract =
            request_to_retract.expect("cache admission failed without a retractable request");
        if self.config.has_host_cache()
            && self
                .requests
                .get(&request_to_retract)
                .unwrap()
                .is::<fsm::Decoding>()
        {
            if let Some(operation) = self.begin_retraction(&request_to_retract) {
                write_back_operations.push(operation);
            }
            return;
        }
        {
            let request = self.requests.get_mut(&request_to_retract).unwrap();
            let mut coord = self.coordinator.borrow_mut();
            request.apply(FsmEvent::Retract, &mut coord);
        }
        if !self.config.has_host_cache() {
            self.fused_capacity_drain = true;
        }
    }
}
impl Scheduler {
    /// Priority used to order scheduling candidates (C++ `buildForwardOperations` lambda).
    fn request_priority(&self, request_id: &str) -> i32 {
        let request = self.requests.get(request_id).unwrap();
        let recovery_front =
            !self.recovery_queue.is_empty() && request_id == self.recovery_queue.front().unwrap();
        let local_decode_prefill =
            request.is::<fsm::Prefilling>() && request.prefill_source() == PrefillSource::Local;
        if self.config.role == Role::P && request.is::<fsm::PrefillDone>() {
            return 0;
        }
        if self.config.role == Role::D
            && (local_decode_prefill
                || request.is::<fsm::PrefillDone>()
                || (recovery_front && request.is::<fsm::Decoding>()))
        {
            return 0;
        }
        if self.recovery_barrier.as_deref() == Some(request_id) {
            return 1;
        }
        if request.is::<fsm::Retracted>()
            && !self.recovery_queue.is_empty()
            && request_id == self.recovery_queue.front().unwrap()
        {
            return 2;
        }
        if request.is::<fsm::Prefilling>() {
            return 3;
        }
        if request.is::<fsm::Submitted>() {
            return 4;
        }
        if request.is::<fsm::Decoding>() || request.is::<fsm::PrefillDone>() {
            return if self.config.enable_mixed_prefill_decode {
                3
            } else {
                5
            };
        }
        10
    }

    fn build_forward_operations(
        &mut self,
        plan: &mut ExecutionPlan,
        candidates: Vec<String>,
        write_back_operations: &mut Vec<WriteBackOperation>,
    ) -> (Vec<ForwardOperation>, Vec<LoadBackOperation>) {
        let mut context = PlanBuildContext::new(plan);
        while let Some(front) = self.recovery_queue.front().cloned() {
            let finished = self
                .requests
                .get(&front)
                .is_none_or(|r| r.is::<fsm::Finished>());
            if !finished {
                break;
            }
            self.recovery_queue.pop_front();
        }
        if let Some(barrier) = self.recovery_barrier.clone() {
            let gone = self
                .requests
                .get(&barrier)
                .is_none_or(|r| r.is::<fsm::Finished>() || r.is::<fsm::Retracted>());
            if gone {
                self.recovery_barrier = None;
            }
        }
        if self.fused_capacity_drain {
            let has_resident = candidates.iter().any(|id| {
                let r = self.requests.get(id).unwrap();
                r.is::<fsm::Prefilling>() || r.is::<fsm::PrefillDone>() || r.is::<fsm::Decoding>()
            });
            if !has_resident {
                self.fused_capacity_drain = false;
            }
        }
        let mut candidates = candidates;
        candidates.sort_by(|a, b| {
            let pa = self.request_priority(a);
            let pb = self.request_priority(b);
            pa.cmp(&pb).then_with(|| a.cmp(b))
        });

        let build_prefill_handoff_batch = self.config.role == Role::P
            && !candidates.is_empty()
            && self
                .requests
                .get(&candidates[0])
                .unwrap()
                .is::<fsm::PrefillDone>();
        let has_local_prefill = candidates.iter().any(|id| {
            let r = self.requests.get(id).unwrap();
            (r.is::<fsm::Prefilling>() && r.prefill_source() == PrefillSource::Local)
                || (self.config.role != Role::D && r.is::<fsm::Submitted>())
                || (r.is::<fsm::Retracted>()
                    && !self.recovery_queue.is_empty()
                    && *self.recovery_queue.front().unwrap() == *id)
        });
        let state_prefill_reserve = if !build_prefill_handoff_batch
            && self.config.enable_mixed_prefill_decode
            && self.coordinator.borrow().has_mamba_state_group()
            && has_local_prefill
        {
            self.coordinator.borrow().prefix_granularity()
        } else {
            0
        };

        let mut operations: Vec<ForwardOperation> = Vec::new();
        let mut load_back_operations: Vec<LoadBackOperation> = Vec::new();
        let mut token_budget = self.config.max_scheduled_tokens;
        let mut pushed_prefill = false;
        let mut pushed_decode = false;
        let push_operation = |s: &mut Scheduler,
                              operation: ForwardOperation,
                              token_budget: &mut i32,
                              pushed_prefill: &mut bool,
                              pushed_decode: &mut bool,
                              operations: &mut Vec<ForwardOperation>| {
            let request_id = match &operation {
                ForwardOperation::Prefill(p) => p.base.request_id.clone(),
                ForwardOperation::Decode(d) => d.base.request_id.clone(),
            };
            if s.recovery_barrier.as_deref() == Some(request_id.as_str()) {
                s.recovery_barrier = None;
            }
            match &operation {
                ForwardOperation::Prefill(p) => {
                    if s.config.role != Role::D || p.local_prefill {
                        *token_budget -= p.base.input_length;
                    }
                    *pushed_prefill = true;
                }
                ForwardOperation::Decode(d) => {
                    if s.config.role != Role::D {
                        *token_budget -= d.base.input_length;
                    }
                    *pushed_decode = true;
                }
            }
            operations.push(operation);
        };
        let track_pending_forward_result = |s: &mut Scheduler, request_id: &str| {
            let r = s.requests.get(request_id).unwrap();
            if r.is::<fsm::PrefillDone>() || r.is::<fsm::Decoding>() {
                *s.pending_forward_results
                    .entry(request_id.to_string())
                    .or_insert(0) += 1;
            }
        };

        for request_id in &candidates {
            if token_budget <= 0 || operations.len() == self.config.max_batch_size as usize {
                break;
            }
            if self.fused_capacity_drain
                && self
                    .requests
                    .get(request_id)
                    .unwrap()
                    .is::<fsm::Submitted>()
            {
                continue;
            }
            if build_prefill_handoff_batch
                && !self
                    .requests
                    .get(request_id)
                    .unwrap()
                    .is::<fsm::PrefillDone>()
            {
                break;
            }
            let is_prefilling = self
                .requests
                .get(request_id)
                .unwrap()
                .is::<fsm::Prefilling>();
            let prefill_local = self
                .requests
                .get(request_id)
                .unwrap()
                .is::<fsm::Prefilling>()
                && self.requests.get(request_id).unwrap().prefill_source() == PrefillSource::Local;
            if is_prefilling && (self.config.role != Role::D || prefill_local) {
                if self.config.role == Role::D && pushed_decode {
                    break;
                }
                let reserve = if self.config.role == Role::P {
                    0
                } else {
                    self.config.decode_input_tokens
                };
                if let Some(event) =
                    self.schedule_prefill(&mut context, request_id, token_budget, reserve)
                {
                    let operation = self.apply_event_and_build_operation_prefill(request_id, event);
                    let op_id = operation.base.request_id.clone();
                    push_operation(
                        self,
                        ForwardOperation::Prefill(operation),
                        &mut token_budget,
                        &mut pushed_prefill,
                        &mut pushed_decode,
                        &mut operations,
                    );
                    if self.config.enable_pd_cache && self.config.role != Role::D {
                        self.pd_transfer_pins.insert(request_id.clone());
                    }
                    track_pending_forward_result(self, request_id);
                    if self.config.role == Role::D {
                        break;
                    }
                    if self
                        .requests
                        .get(request_id)
                        .unwrap()
                        .is::<fsm::Prefilling>()
                    {
                        break;
                    }
                    let _ = op_id;
                } else if context.admission_failed {
                    break;
                }
                continue;
            }
            if self
                .requests
                .get(request_id)
                .unwrap()
                .is::<fsm::Retracted>()
                && !self.recovery_queue.is_empty()
                && *self.recovery_queue.front().unwrap() == *request_id
            {
                if self.config.role == Role::D && pushed_decode {
                    break;
                }
                if let Some(event) = self.schedule_prefill_first_chunk(
                    &mut context,
                    request_id,
                    token_budget,
                    self.config.decode_input_tokens,
                ) {
                    let operation = self.apply_event_and_build_operation_first_chunk(
                        request_id,
                        event,
                        &mut load_back_operations,
                    );
                    push_operation(
                        self,
                        ForwardOperation::Prefill(operation),
                        &mut token_budget,
                        &mut pushed_prefill,
                        &mut pushed_decode,
                        &mut operations,
                    );
                    track_pending_forward_result(self, request_id);
                    if self.config.role == Role::D
                        || self
                            .requests
                            .get(request_id)
                            .unwrap()
                            .is::<fsm::Prefilling>()
                    {
                        break;
                    }
                    continue;
                }
                break;
            }
            if self
                .requests
                .get(request_id)
                .unwrap()
                .is::<fsm::Submitted>()
            {
                if self.config.role == Role::D && pushed_decode {
                    break;
                }
                let decode_input_tokens = if self.config.role == Role::P {
                    0
                } else {
                    self.config.decode_input_tokens
                };
                let prefill_budget = if self.config.role == Role::D {
                    self.requests.get(request_id).unwrap().prefill_size()
                } else {
                    token_budget
                };
                if let Some(event) = self.schedule_prefill_first_chunk(
                    &mut context,
                    request_id,
                    prefill_budget,
                    decode_input_tokens,
                ) {
                    let operation = self.apply_event_and_build_operation_first_chunk(
                        request_id,
                        event,
                        &mut load_back_operations,
                    );
                    push_operation(
                        self,
                        ForwardOperation::Prefill(operation),
                        &mut token_budget,
                        &mut pushed_prefill,
                        &mut pushed_decode,
                        &mut operations,
                    );
                    if self.config.enable_pd_cache {
                        self.pd_transfer_pins.insert(request_id.clone());
                    }
                    track_pending_forward_result(self, request_id);
                    if self
                        .requests
                        .get(request_id)
                        .unwrap()
                        .is::<fsm::Prefilling>()
                    {
                        break;
                    }
                }
                continue;
            }
            let is_prefill_done = self
                .requests
                .get(request_id)
                .unwrap()
                .is::<fsm::PrefillDone>();
            let is_decoding = self.requests.get(request_id).unwrap().is::<fsm::Decoding>();
            if is_prefill_done || (is_decoding && self.config.role != Role::P) {
                if (self.config.role == Role::D || !self.config.enable_mixed_prefill_decode)
                    && pushed_prefill
                {
                    break;
                }
                if token_budget < state_prefill_reserve + self.config.decode_input_tokens {
                    continue;
                }
                if let Some(event) = self.schedule_decode(&mut context, request_id) {
                    let operation = self.apply_event_and_build_operation_decode(request_id, event);
                    push_operation(
                        self,
                        ForwardOperation::Decode(operation),
                        &mut token_budget,
                        &mut pushed_prefill,
                        &mut pushed_decode,
                        &mut operations,
                    );
                    track_pending_forward_result(self, request_id);
                }
            }
        }
        if operations.is_empty() && context.admission_failed {
            self.retract_for_capacity(&context, &candidates, write_back_operations);
        }
        (operations, load_back_operations)
    }

    /// Build the next execution plan.
    pub fn next_execution_plan(&mut self) -> ExecutionPlan {
        let finished: Vec<String> = self
            .requests
            .iter()
            .filter(|(_, r)| r.is::<fsm::Finished>())
            .map(|(id, _)| id.clone())
            .collect();
        for id in finished {
            self.requests.remove(&id);
            self.kv.borrow_mut().hash_progress.remove(&id);
        }
        let candidates: Vec<String> = self
            .requests
            .iter()
            .filter(|(_, r)| {
                r.is::<fsm::Submitted>()
                    || r.is::<fsm::Prefilling>()
                    || r.is::<fsm::PrefillDone>()
                    || r.is::<fsm::Decoding>()
                    || r.is::<fsm::Retracted>()
            })
            .map(|(id, _)| id.clone())
            .collect();
        let mut plan = ExecutionPlan::new();
        let mut write_back_operations = Vec::new();
        let (forward_operations, load_back_operations) =
            self.build_forward_operations(&mut plan, candidates, &mut write_back_operations);
        plan.with(Operation::Forward(ForwardBatch::new(forward_operations)));
        if self.config.streams_device_cache_to_host() {
            if let Some(store) = self.tier_transfers.start_pending_stores() {
                write_back_operations.push(store);
            }
        }
        if !write_back_operations.is_empty() {
            plan.with(Operation::Cache(CacheOperation::WriteBack(
                crate::tier::WriteBackBatch::new(&write_back_operations),
            )));
        }
        if !load_back_operations.is_empty() {
            plan.with(Operation::Cache(CacheOperation::LoadBack(
                crate::tier::LoadBackBatch::new(&load_back_operations),
            )));
        }
        plan
    }
}
impl Scheduler {
    /// Deliver a batch of outside events.
    pub fn advance(&mut self, event: &ExecutionEvent) {
        for item in event.events() {
            match item {
                Event::WriteBackDone(e) => self.tier_transfers.complete_write_back(e.op_id),
                Event::LoadBackDone(e) => self.tier_transfers.complete_load_back(e.op_id),
                Event::ExtendResult(e) => self.handle_extend_result(&e.request_id, &e.tokens),
                Event::Finish(e) => self.handle_finish(&e.request_id),
                Event::Abort(e) => self.handle_abort(&e.request_id),
                Event::UpdateReserveNumTokens(e) => self.handle_update_reserve(
                    &e.request_id,
                    e.reserve_num_tokens_in_next_schedule_event,
                ),
                Event::Bootstrapped(e) => self.handle_bootstrapped(&e.request_id),
                Event::Failed(e) => self.handle_failed(&e.request_id),
                Event::Succeeded(e) => self.handle_succeeded(&e.request_id),
                Event::RemotePrefillDone(e) => {
                    self.handle_remote_prefill_done(&e.request_id, e.bootstrap_token)
                }
            }
        }
    }

    fn handle_bootstrapped(&mut self, request_id: &str) {
        let request = self.requests.get_mut(request_id);
        if let Some(request) = request {
            if request.is::<fsm::Bootstrapping>() {
                let mut coord = self.coordinator.borrow_mut();
                request.apply(FsmEvent::Bootstrapped, &mut coord);
            }
        }
    }

    fn handle_failed(&mut self, request_id: &str) {
        let finished = self
            .requests
            .get(request_id)
            .is_some_and(|r| r.is::<fsm::Finished>());
        if finished {
            return;
        }
        self.pending_forward_results.remove(request_id);
        self.pd_transfer_pins.remove(request_id);
        if let Some(request) = self.requests.get_mut(request_id) {
            let mut coord = self.coordinator.borrow_mut();
            request.apply(FsmEvent::Abort, &mut coord);
        }
    }

    fn handle_succeeded(&mut self, request_id: &str) {
        let request = self.requests.get(request_id);
        if request.is_none() || request.unwrap().is::<fsm::Finished>() {
            return;
        }
        let request = request.unwrap();
        if !request.is::<fsm::PrefillDone>() && !request.is::<fsm::Decoding>() {
            panic!(
                "PD SucceededEvent received in state {}",
                request.state_name()
            );
        }
        self.pending_forward_results.remove(request_id);
        self.pd_transfer_pins.remove(request_id);
        if let Some(request) = self.requests.get_mut(request_id) {
            let mut coord = self.coordinator.borrow_mut();
            request.apply(FsmEvent::Finish, &mut coord);
        }
    }

    fn handle_remote_prefill_done(&mut self, request_id: &str, bootstrap_token: i32) {
        let request = self.requests.get(request_id);
        let Some(request) = request else {
            return;
        };
        if request.is::<fsm::Prefilling>() {
            assert!(
                bootstrap_token >= 0,
                "PD RemotePrefillDoneEvent requires a non-negative bootstrap token"
            );
            self.pd_transfer_pins.remove(request_id);
            if let Some(request) = self.requests.get_mut(request_id) {
                let mut coord = self.coordinator.borrow_mut();
                request.apply(FsmEvent::RemotePrefillDone(bootstrap_token), &mut coord);
            }
            return;
        }
        if request.is::<fsm::PrefillDone>()
            || request.is::<fsm::Decoding>()
            || request.is::<fsm::Finished>()
        {
            return;
        }
        panic!(
            "PD RemotePrefillDoneEvent received before destination admission; state={}",
            request.state_name()
        );
    }

    fn handle_finish(&mut self, request_id: &str) {
        if self.config.enable_pd_cache && self.pd_transfer_pins.contains(request_id) {
            panic!("PD Finish received while transfer pages are pinned");
        }
        self.pending_forward_results.remove(request_id);
        if self.requests.contains_key(request_id) {
            let is_forward = {
                let r = self.requests.get(request_id).unwrap();
                r.is::<fsm::PrefillDone>() || r.is::<fsm::Decoding>()
            };
            if is_forward {
                self.publish_completed_pages(request_id);
            }
            if let Some(request) = self.requests.get_mut(request_id) {
                let mut coord = self.coordinator.borrow_mut();
                request.apply(FsmEvent::Finish, &mut coord);
            }
        }
    }

    fn publish_completed_pages(&mut self, request_id: &str) {
        let stable_prefix_pages = self
            .requests
            .get(request_id)
            .unwrap()
            .full_prefix_pages(true);
        let mut progress = self.requests.get(request_id).unwrap().cache_progress();
        let first_new_prefix_page = progress.prefix_hashes.len() as i32;
        let num_stable_prefix_pages = stable_prefix_pages.len() as i32;
        assert!(
            first_new_prefix_page <= num_stable_prefix_pages,
            "cache progress exceeds completed request pages"
        );
        if first_new_prefix_page == num_stable_prefix_pages {
            return;
        }
        let previous_hash = progress.prefix_hashes.last().cloned().unwrap_or_default();
        let refs: Vec<&[i32]> = stable_prefix_pages.iter().map(|p| p.as_slice()).collect();
        let new_hashes = advance_prefix_hashes(
            &refs,
            first_new_prefix_page as usize,
            &previous_hash,
            num_stable_prefix_pages as usize,
        );
        progress.prefix_hashes.extend(new_hashes);
        let event_keys = self.register_kv_event_prefix_pages(
            request_id,
            &progress.prefix_hashes,
            first_new_prefix_page,
        );
        let token_size_minus_1 = self.requests.get(request_id).unwrap().token_size() - 1;
        let tables = self
            .requests
            .get_mut(request_id)
            .unwrap()
            .block_tables_ref_mut();
        let mut coord = self.coordinator.borrow_mut();
        coord.cache_completed_blocks(
            tables,
            &progress.prefix_hashes,
            progress.access_epoch,
            first_new_prefix_page,
            token_size_minus_1,
            CacheBoundaryKind::Endpoint,
        );
        drop(coord);
        self.discard_uncached_kv_event_pages(&event_keys);
    }

    fn handle_update_reserve(&mut self, request_id: &str, value: i32) {
        if let Some(request) = self.requests.get_mut(request_id) {
            let mut coord = self.coordinator.borrow_mut();
            request.apply(FsmEvent::UpdateReserveNumTokens(value), &mut coord);
        }
    }

    fn handle_extend_result(&mut self, request_id: &str, tokens: &[i32]) {
        if let Some(count) = self.pending_forward_results.get_mut(request_id) {
            *count -= 1;
            if *count <= 0 {
                self.pending_forward_results.remove(request_id);
            }
        }
        if let Some(request) = self.requests.get_mut(request_id) {
            let mut coord = self.coordinator.borrow_mut();
            request.apply(FsmEvent::ExtendResult(tokens.to_vec()), &mut coord);
        }
    }

    fn handle_abort(&mut self, request_id: &str) {
        self.pending_forward_results.remove(request_id);
        self.pd_transfer_pins.remove(request_id);
        if let Some(request) = self.requests.get_mut(request_id) {
            let mut coord = self.coordinator.borrow_mut();
            request.apply(FsmEvent::Abort, &mut coord);
        }
    }

    fn register_kv_event_prefix_pages(
        &mut self,
        request_id: &str,
        prefix_hashes: &[String],
        first_page: i32,
    ) -> Vec<CacheKey> {
        if !self.config.enable_kv_cache_events {
            return Vec::new();
        }
        assert!(
            first_page >= 0 && first_page as usize <= prefix_hashes.len(),
            "KV event page range is invalid"
        );
        let token_pages = self
            .requests
            .get(request_id)
            .unwrap()
            .full_prefix_pages(false);
        assert!(
            prefix_hashes.len() <= token_pages.len(),
            "KV event hashes exceed the request's complete pages"
        );
        let mut kv = self.kv.borrow_mut();
        let hash_snapshot: Vec<u64> = {
            let progress = kv.hash_progress.entry(request_id.to_string()).or_default();
            for i in progress.len()..prefix_hashes.len() {
                let parent_hash = if i == 0 { None } else { Some(progress[i - 1]) };
                progress.push(hash_kv_block(&token_pages[i], parent_hash));
            }
            progress.clone()
        };
        let mut registered_keys = Vec::with_capacity(prefix_hashes.len() - first_page as usize);
        for i in first_page as usize..prefix_hashes.len() {
            let key = CacheKey {
                namespace_id: 0,
                group_id: 0,
                content_hash: prefix_hashes[i].clone(),
                page_offset: 0,
            };
            let parent_hash = if i == 0 {
                None
            } else {
                Some(hash_snapshot[i - 1])
            };
            let event = KvBlockStoredEvent {
                block_hashes: vec![hash_snapshot[i]],
                parent_block_hash: parent_hash,
                token_ids: token_pages[i].clone(),
                block_size: self.config.prefix_granularity,
            };
            if let Some(existing) = kv.pages.get(&key) {
                assert!(
                    existing.block_hashes[0] == hash_snapshot[i],
                    "one cache content hash mapped to different KV event blocks"
                );
            } else {
                kv.pages.insert(key.clone(), event);
            }
            registered_keys.push(key);
        }
        registered_keys
    }

    fn discard_uncached_kv_event_pages(&mut self, keys: &[CacheKey]) {
        let mut kv = self.kv.borrow_mut();
        for key in keys {
            if !kv.child_counts.contains_key(key) {
                kv.pages.remove(key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_config::{CacheGroupConfig, CacheGroupFamily, CacheTransferPolicy, Retention};
    use crate::events::forward;
    use crate::types::AllocatorConfig;

    fn config() -> SchedulerConfig {
        SchedulerConfig {
            prefix_granularity: 4,
            host_allocator: AllocatorConfig { total_pages: 0 },
            device_allocator: AllocatorConfig { total_pages: 1025 },
            cache_groups: vec![CacheGroupConfig {
                group_id: "kv".into(),
                rows_per_page: 4,
                entry_stride_tokens: 1,
                total_pages: 1024,
                cache_blocks_per_lcm_block: 1,
                retention: Retention::FullHistory,
                sliding_window_tokens: None,
                family: CacheGroupFamily::History,
                transfer_policy: CacheTransferPolicy::Unspecified,
            }],
            max_scheduled_tokens: 64,
            max_batch_size: 8,
            decode_input_tokens: 1,
            ..SchedulerConfig::default()
        }
    }

    fn forward_batch(plan: &ExecutionPlan) -> &ForwardBatch {
        plan.operations()
            .iter()
            .find_map(|op| match op {
                Operation::Forward(f) => Some(f),
                _ => None,
            })
            .expect("plan contains a forward batch")
    }

    #[test]
    fn prefill_then_decode_then_finish() {
        let mut scheduler = Scheduler::new(config());
        let spec = RequestSpec {
            request_id: "r1".into(),
            tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
            max_new_tokens: 4,
        };
        scheduler.submit_requests(&[spec]);
        assert_eq!(scheduler.waiting_size(), 1);

        // First plan schedules the whole prefill (budget 64 covers 8 tokens).
        let plan = scheduler.next_execution_plan();
        let batch = forward_batch(&plan);
        assert_eq!(batch.request_ids, vec!["r1"]);
        assert_eq!(batch.input_lengths, vec![8]);
        assert_eq!(batch.num_extends(), 1);
        assert_eq!(scheduler.prefill_size(), 1);
        assert_eq!(scheduler.waiting_size(), 0);

        // Second plan schedules a decode step.
        let plan2 = scheduler.next_execution_plan();
        let batch2 = forward_batch(&plan2);
        assert_eq!(batch2.request_ids, vec!["r1"]);
        assert_eq!(batch2.decode_input_ids, vec![-1]);
        assert_eq!(scheduler.decoding_size(), 1);

        // Advance with an extend result, then finish.
        let mut ev = ExecutionEvent::new();
        ev.with(Event::ExtendResult(forward::ExtendResult {
            request_id: "r1".into(),
            tokens: vec![9],
        }));
        scheduler.advance(&ev);
        assert_eq!(scheduler.request_token_size("r1"), 9);

        let mut fin = ExecutionEvent::new();
        fin.with(Event::Finish(forward::Finish {
            request_id: "r1".into(),
        }));
        scheduler.advance(&fin);
        // Finished requests are dropped on the next plan build.
        assert_eq!(scheduler.decoding_size(), 0);
        let plan3 = scheduler.next_execution_plan();
        assert!(forward_batch(&plan3).is_empty());
        assert_eq!(scheduler.request_token_size("r1"), -1);
    }

    #[test]
    #[should_panic(expected = "Scheduler: request tokens must be non-empty")]
    fn submit_rejects_empty_tokens() {
        let mut scheduler = Scheduler::new(config());
        scheduler.submit_requests(&[RequestSpec {
            request_id: "r1".into(),
            tokens: vec![],
            max_new_tokens: 0,
        }]);
    }

    #[test]
    #[should_panic(expected = "Scheduler: duplicate request id 'r1'")]
    fn submit_rejects_duplicate_id() {
        let mut scheduler = Scheduler::new(config());
        let spec = RequestSpec {
            request_id: "r1".into(),
            tokens: vec![1, 2, 3, 4],
            max_new_tokens: 0,
        };
        scheduler.submit_requests(&[spec.clone(), spec]);
    }

    #[test]
    fn advance_abort_releases_request() {
        let mut scheduler = Scheduler::new(config());
        let spec = RequestSpec {
            request_id: "r1".into(),
            tokens: vec![1, 2, 3, 4],
            max_new_tokens: 4,
        };
        scheduler.submit_requests(&[spec]);
        let _plan = scheduler.next_execution_plan();
        let mut ev = ExecutionEvent::new();
        ev.with(Event::Abort(forward::Abort {
            request_id: "r1".into(),
        }));
        scheduler.advance(&ev);
        assert_eq!(scheduler.decoding_size(), 0);
        assert_eq!(scheduler.prefill_size(), 0);
        let plan = scheduler.next_execution_plan();
        assert!(forward_batch(&plan).is_empty());
    }
}
