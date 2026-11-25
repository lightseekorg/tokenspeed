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

//! Complete-prefix SSD lookup and source preparation, independent of admission.
//!
//! These methods run on I/O workers. They own only bounded source leases and
//! immutable metadata; ordinary peer admission owns all destination pages.
//! No L3 object list is published into a router's hot-cache index.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use ts_l3::{keys::object_key, layout::HostLayout};
use ts_mooncake::offload::{self, Ready, Replica};
use ts_scheduler::Scheduler;

use crate::peer::{prepared::PreparedObject, read::ReadOutcome};

/// A bounded query, shape, or source-lifetime failure. No prefix is published.
#[derive(Debug, thiserror::Error)]
#[error("L3 prepared prefix: {0}")]
pub struct Error(pub String);

impl From<offload::Error> for Error {
    fn from(error: offload::Error) -> Self {
        Self(error.to_string())
    }
}

/// One rank's dependency at a complete resumable boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Object {
    /// Persistent object key, including namespace and shard identity.
    pub key: String,
    /// Reader member owning this destination shard.
    pub rank: usize,
    /// Cache group index.
    pub group_id: u32,
    /// Logical page slot, including the ordinary State slot.
    pub slot: i32,
    /// Full packed object size, including fields and padding.
    pub bytes: u64,
}

/// Immutable complete dependencies. Construction never reserves L1 or L2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixPlan {
    /// Exact complete boundary, not a raw Full-attention overlap.
    pub num_tokens: i32,
    /// Replica member count, including owners with no missing fields.
    pub members: usize,
    /// Dependencies across all members; shared physical objects may repeat.
    pub objects: Vec<Object>,
}

impl PrefixPlan {
    /// Use the scheduler's existing Full/SWA/State dependency planner and the
    /// persistent L3 key format. Call while holding the scheduler only for
    /// metadata construction, then release it before any network operation.
    pub fn from_scheduler(
        scheduler: &Scheduler,
        tokens: &[i32],
        boundary: i32,
        key_prefix: &str,
        layouts: &[HostLayout],
        max_keys: usize,
    ) -> Result<Option<Self>, Error> {
        if layouts.is_empty() || key_prefix.is_empty() || max_keys == 0 || max_keys > 1024 {
            return Err(Error("invalid prefix metadata budget or namespace".into()));
        }
        let metadata = scheduler
            .storage_prefix_plan(tokens, boundary, max_keys)
            .map_err(|e| Error(e.to_string()))?;
        Self::from_metadata(metadata, key_prefix, layouts, max_keys)
    }

    /// Preserve the request's content hashes; never rehash a peer's media or
    /// tenant-sensitive prefix as plain token IDs. The boundary is full-page.
    pub fn from_prompt(
        scheduler: &Scheduler,
        prompt: &super::wire::Prompt,
        boundary: i32,
        key_prefix: &str,
        layouts: &[HostLayout],
        max_keys: usize,
    ) -> Result<Option<Self>, Error> {
        if layouts.is_empty() || key_prefix.is_empty() || max_keys == 0 || max_keys > 1024 {
            return Err(Error("invalid prefix metadata budget or namespace".into()));
        }
        let metadata = scheduler
            .storage_prefix_plan_from_hashes(
                &prompt.tokens,
                &prompt.prefix_hashes,
                boundary,
                max_keys,
            )
            .map_err(|e| Error(e.to_string()))?;
        Self::from_metadata(metadata, key_prefix, layouts, max_keys)
    }

    fn from_metadata(
        metadata: ts_scheduler::StoragePrefixPlanView,
        key_prefix: &str,
        layouts: &[HostLayout],
        max_keys: usize,
    ) -> Result<Option<Self>, Error> {
        if metadata.max_tokens <= 0 {
            return Ok(None);
        }
        let mut objects = Vec::new();
        let mut keys = HashSet::new();
        for (rank, layout) in layouts.iter().enumerate() {
            for object in &metadata.objects {
                let key = &object.key;
                let shard = layout
                    .key_rank(key.group_id)
                    .map_err(|e| Error(e.to_string()))?;
                let key_string = object_key(
                    key_prefix,
                    &key.content_hash,
                    key.group_id,
                    key.page_offset,
                    shard,
                );
                keys.insert(key_string.clone());
                if keys.len() > max_keys {
                    return Ok(None);
                }
                let bytes = layout
                    .block_bytes(key.group_id)
                    .map_err(|e| Error(e.to_string()))? as u64;
                objects.push(Object {
                    key: key_string,
                    rank,
                    group_id: key.group_id,
                    slot: object.slot,
                    bytes,
                });
            }
        }
        Ok(Some(Self {
            num_tokens: metadata.max_tokens,
            members: layouts.len(),
            objects,
        }))
    }
}

/// Backend source ownership before the read. Dropping this releases unused
/// staging; implementations must not expose addresses before `begin_read`.
pub trait PreparedSource: Send {
    /// The source ownership once reads may have been submitted.
    type Reading: ReadingSource;
    /// Check the remaining source TTL before any native read is submitted.
    fn begin_read(self, budget: Duration) -> Result<Self::Reading, Error>;
}

/// An issued source lease. Drop must NOT release it before native retirement.
pub trait ReadingSource: Send {
    /// The registered source descriptors in preparation key order.
    fn source(&self) -> &Ready;
    /// Called only after every reader retired; confirms the lease was alive.
    fn release_retired(self) -> Result<bool, Error>;
}

impl PreparedSource for offload::PreparedLease {
    type Reading = offload::ReadingLease;
    fn begin_read(self, budget: Duration) -> Result<Self::Reading, Error> {
        offload::PreparedLease::begin_read(self, budget).map_err(Into::into)
    }
}
impl ReadingSource for offload::ReadingLease {
    fn source(&self) -> &Ready {
        offload::ReadingLease::source(self)
    }
    fn release_retired(self) -> Result<bool, Error> {
        self.retired().release().map_err(Into::into)
    }
}

/// Only the storage network is abstracted; planning and lifetime logic are shared.
pub trait Backend: Send + Sync {
    /// Prepared source ownership.
    type Prepared: PreparedSource;
    /// Complete SSD replicas, in input key order. Empty rows are misses.
    fn query(&self, keys: &[String]) -> Result<Vec<Vec<Replica>>, Error>;
    /// Prepare exactly these objects at one provider without reader destinations.
    fn prepare(
        &self,
        endpoint: &str,
        keys: &[String],
        bytes: &[u64],
        generations: &[[u64; 2]],
    ) -> Result<Self::Prepared, Error>;
}

/// The deployed Mooncake SDK, selected explicitly at startup.
pub struct Mooncake {
    /// Native RPC adapter loaded separately from the transfer engine.
    pub client: Arc<offload::Client>,
    /// Master RPC endpoint.
    pub master: String,
    /// Object tenant.
    pub tenant: String,
    /// Per-RPC deadline, including connection setup.
    pub timeout: Duration,
}

impl Backend for Mooncake {
    type Prepared = offload::PreparedLease;
    fn query(&self, keys: &[String]) -> Result<Vec<Vec<Replica>>, Error> {
        self.client
            .query(&self.master, &self.tenant, keys, self.timeout)
            .map_err(Into::into)
    }
    fn prepare(
        &self,
        endpoint: &str,
        keys: &[String],
        bytes: &[u64],
        generations: &[[u64; 2]],
    ) -> Result<Self::Prepared, Error> {
        self.client
            .prepare(
                endpoint,
                &self.tenant,
                keys,
                bytes,
                generations,
                self.timeout,
            )
            .map_err(Into::into)
    }
}

/// One bounded request's source work. These are not inference memory limits.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Unique object keys across all candidate boundaries.
    pub max_query_keys: usize,
    /// Candidate complete boundaries, longest first.
    pub max_candidates: usize,
    /// Bytes staged for this request, deduplicated across owners.
    pub max_prepare_bytes: u64,
    /// Bytes in each provider prepare RPC, within the provider's arena limit.
    pub max_source_batch_bytes: u64,
    /// Provider batches per request, bounding RPC and lease bookkeeping.
    pub max_source_batches: usize,
}

/// Source staging reserved by this reader, including unknown native lifetimes.
#[derive(Clone)]
pub struct SourceBudget(Arc<Mutex<BudgetState>>);

struct BudgetState {
    limit: u64,
    held: u64,
    quarantined: u64,
}
struct Credit {
    budget: SourceBudget,
    bytes: u64,
}

impl SourceBudget {
    /// An explicit byte budget shared by all preparations on this worker.
    pub fn new(limit: u64) -> Result<Self, Error> {
        if limit == 0 {
            return Err(Error("zero source staging budget".into()));
        }
        Ok(Self(Arc::new(Mutex::new(BudgetState {
            limit,
            held: 0,
            quarantined: 0,
        }))))
    }
    /// (held bytes, bytes withheld after unproven retirement).
    pub fn usage(&self) -> (u64, u64) {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        (state.held, state.quarantined)
    }
    /// Advisory remaining bytes; only reserve performs admission.
    pub fn available(&self) -> u64 {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.limit - state.held
    }
    fn reserve(&self, bytes: u64) -> Option<Credit> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let held = state.held.checked_add(bytes)?;
        if held > state.limit {
            return None;
        }
        state.held = held;
        Some(Credit {
            budget: self.clone(),
            bytes,
        })
    }
}
impl Credit {
    fn quarantine(mut self) {
        let mut state = self.budget.0.lock().unwrap_or_else(|e| e.into_inner());
        state.quarantined += self.bytes;
        // Keep the charge until worker restart. A timer is not DMA retirement.
        self.bytes = 0;
    }
}
impl Drop for Credit {
    fn drop(&mut self) {
        self.budget.0.lock().unwrap_or_else(|e| e.into_inner()).held -= self.bytes;
    }
}

/// A queried complete boundary. It owns no source lease or destination pages.
pub struct Hit {
    plan: PrefixPlan,
    unique: Vec<(String, Replica)>,
    bytes: u64,
}
impl Hit {
    /// Prepare only the pages ordinary admission actually needs. Query may
    /// have included pages that became local while the metadata was in flight.
    pub fn restrict_to(
        mut self,
        destinations: &[ts_scheduler::PeerPageSlotView],
    ) -> Result<Self, Error> {
        for destination in destinations {
            for rank in 0..self.plan.members {
                if !self.plan.objects.iter().any(|object| {
                    object.rank == rank
                        && object.group_id == destination.group_id
                        && object.slot == destination.slot
                }) {
                    return Err(Error(
                        "admitted destination missing from storage plan".into(),
                    ));
                }
            }
        }
        self.plan.objects.retain(|object| {
            destinations
                .iter()
                .any(|page| page.group_id == object.group_id && page.slot == object.slot)
        });
        if self.plan.objects.is_empty() {
            return Err(Error("empty storage preparation".into()));
        }
        let keys: HashSet<_> = self
            .plan
            .objects
            .iter()
            .map(|object| object.key.as_str())
            .collect();
        self.unique.retain(|(key, _)| keys.contains(key.as_str()));
        self.bytes = self
            .unique
            .iter()
            .try_fold(0u64, |sum, (_, replica)| sum.checked_add(replica.bytes))
            .ok_or_else(|| Error("prefix byte count overflow".into()))?;
        Ok(self)
    }

    /// Exact boundary to consider in a routing/admission cost estimate.
    pub fn num_tokens(&self) -> i32 {
        self.plan.num_tokens
    }
    /// Bytes copied to all reader owners, including replicated destinations.
    pub fn transfer_bytes(&self) -> Result<u64, Error> {
        self.plan.objects.iter().try_fold(0u64, |n, object| {
            n.checked_add(object.bytes)
                .ok_or_else(|| Error("prefix transfer byte count overflow".into()))
        })
    }
    /// Physical bytes to stage, deduplicated across logical owners.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Query only the supplied bounded complete-boundary candidates. A Full hit
/// without its SWA/State/owner dependencies cannot become a hit.
pub fn query<B: Backend>(
    backend: &B,
    candidates: &[PrefixPlan],
    limits: Limits,
) -> Result<Option<Hit>, Error> {
    if limits.max_query_keys == 0
        || limits.max_query_keys > 1024
        || limits.max_candidates == 0
        || limits.max_prepare_bytes == 0
        || limits.max_source_batch_bytes == 0
        || limits.max_source_batch_bytes > (1 << 31)
        || limits.max_source_batches == 0
        || candidates.len() > limits.max_candidates
    {
        return Err(Error("invalid or exceeded source limits".into()));
    }
    let mut keys = Vec::new();
    let mut sizes = HashMap::new();
    let mut previous = i32::MAX;
    let mut members = None;
    for plan in candidates {
        if plan.num_tokens <= 0
            || plan.num_tokens >= previous
            || plan.members == 0
            || members.is_some_and(|n| n != plan.members)
            || plan.objects.is_empty()
        {
            return Err(Error(
                "invalid complete-boundary candidate order/shape".into(),
            ));
        }
        previous = plan.num_tokens;
        members = Some(plan.members);
        let mut slots = HashSet::new();
        for object in &plan.objects {
            if object.rank >= plan.members
                || object.slot < 0
                || object.bytes == 0
                || object.key.is_empty()
                || !slots.insert((object.rank, object.group_id, object.slot))
            {
                return Err(Error("invalid or duplicated prefix dependency".into()));
            }
            if let Some(bytes) = sizes.insert(object.key.clone(), object.bytes) {
                if bytes != object.bytes {
                    return Err(Error("conflicting object sizes".into()));
                }
            } else {
                keys.push(object.key.clone());
                if keys.len() > limits.max_query_keys {
                    return Err(Error("query key budget exceeded".into()));
                }
            }
        }
        if (0..plan.members).any(|rank| !plan.objects.iter().any(|o| o.rank == rank)) {
            return Err(Error("missing owner's dependency plan".into()));
        }
    }
    if keys.is_empty() {
        return Ok(None);
    }
    let rows = backend.query(&keys)?;
    if rows.len() != keys.len() {
        return Err(Error("query result shape mismatch".into()));
    }
    let locations: HashMap<_, _> = keys.into_iter().zip(rows).collect();
    for plan in candidates {
        let mut selected = BTreeMap::new();
        let mut complete = true;
        for object in &plan.objects {
            let location = locations[&object.key]
                .iter()
                .filter(|r| r.bytes == object.bytes && !r.endpoint.is_empty())
                .min_by(|a, b| a.endpoint.cmp(&b.endpoint));
            let Some(replica) = location else {
                complete = false;
                break;
            };
            selected.insert(object.key.clone(), replica.clone());
        }
        if !complete {
            continue;
        }
        let bytes = selected
            .values()
            .try_fold(0u64, |sum, replica| sum.checked_add(replica.bytes))
            .ok_or_else(|| Error("prefix byte count overflow".into()))?;
        if bytes > limits.max_prepare_bytes {
            continue;
        }
        return Ok(Some(Hit {
            plan: plan.clone(),
            bytes,
            unique: selected.into_iter().collect(),
        }));
    }
    Ok(None)
}

struct Batch<S> {
    keys: Vec<String>,
    source: S,
}

/// Prepared source buffers waiting for ordinary L1 admission. Dropping an
/// unadmitted prefix releases sources and credits; it never touches L1/L2.
pub struct PreparedPrefix<P: PreparedSource> {
    plan: PrefixPlan,
    batches: Vec<Batch<P>>,
    credit: Credit,
}

/// Stage a queried hit. Errors unwind all earlier source batches. The caller
/// must discard stale hits and reacquire on a later attempt, not publish them.
pub fn prepare<B: Backend>(
    backend: &B,
    hit: Hit,
    limits: Limits,
    budget: &SourceBudget,
) -> Result<Option<PreparedPrefix<B::Prepared>>, Error> {
    let Some(credit) = budget.reserve(hit.bytes) else {
        return Ok(None);
    };
    if hit.bytes > limits.max_prepare_bytes
        || limits.max_source_batch_bytes == 0
        || limits.max_source_batch_bytes > (1 << 31)
        || limits.max_source_batches == 0
    {
        return Err(Error("invalid preparation limits".into()));
    }
    let mut by_owner: BTreeMap<String, Vec<(String, Replica)>> = BTreeMap::new();
    for (key, replica) in hit.unique {
        by_owner
            .entry(replica.endpoint.clone())
            .or_default()
            .push((key, replica));
    }
    let mut jobs = Vec::new();
    for (endpoint, objects) in by_owner {
        let mut keys = Vec::new();
        let mut sizes = Vec::new();
        let mut generations = Vec::new();
        let mut total = 0u64;
        for (key, replica) in objects {
            let bytes = replica.bytes;
            if bytes > limits.max_source_batch_bytes {
                return Err(Error("object exceeds source batch capacity".into()));
            }
            if keys.len() == 1024 || total > limits.max_source_batch_bytes - bytes {
                jobs.push((
                    endpoint.clone(),
                    std::mem::take(&mut keys),
                    std::mem::take(&mut sizes),
                    std::mem::take(&mut generations),
                ));
                total = 0;
            }
            keys.push(key);
            sizes.push(bytes);
            generations.push(replica.generation);
            total += bytes;
        }
        if !keys.is_empty() {
            jobs.push((endpoint, keys, sizes, generations));
        }
    }
    if jobs.len() > limits.max_source_batches {
        return Err(Error("source batch budget exceeded".into()));
    }
    let mut batches = Vec::new();
    for (endpoint, keys, sizes, generations) in jobs {
        let source = backend.prepare(&endpoint, &keys, &sizes, &generations)?;
        batches.push(Batch { keys, source });
    }
    Ok(Some(PreparedPrefix {
        plan: hit.plan,
        batches,
        credit,
    }))
}

impl<P: PreparedSource> PreparedPrefix<P> {
    /// Boundary prepared for admission. It is not published in the reader cache.
    pub fn num_tokens(&self) -> i32 {
        self.plan.num_tokens
    }
    /// After ordinary admission and its Device fence, check every source TTL
    /// before exposing any addresses. On partial failure no read was submitted.
    pub fn begin_read(self, budget: Duration) -> Result<ReadingPrefix<P::Reading>, Error> {
        let mut batches: Vec<Batch<P::Reading>> = Vec::new();
        for batch in self.batches {
            match batch.source.begin_read(budget) {
                Ok(source) => batches.push(Batch {
                    keys: batch.keys,
                    source,
                }),
                Err(error) => {
                    for batch in batches {
                        let _ = batch.source.release_retired();
                    }
                    return Err(error);
                }
            }
        }
        Ok(ReadingPrefix {
            plan: self.plan,
            batches,
            credit: Some(self.credit),
            retired: false,
        })
    }
}

/// Sources possibly in use by native readers. Dropping without `finish` keeps
/// leases at their providers and quarantines credits. The caller must also
/// keep destination pages until native retirement or terminate the worker.
pub struct ReadingPrefix<R: ReadingSource> {
    plan: PrefixPlan,
    batches: Vec<Batch<R>>,
    credit: Option<Credit>,
    retired: bool,
}

/// Facts to feed into ordinary landing agreement; only Complete publishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completion {
    /// Every owner read and retired, and every provider confirmed its live lease.
    Complete,
    /// Safe to retry/recompute: no writes remain, but some bytes are not vouched for.
    FailedRetired,
    /// Destination ownership must be retained; do not turn a timeout into success.
    Unretired,
}

impl<R: ReadingSource> ReadingPrefix<R> {
    /// The exact boundary represented by these sources.
    pub fn num_tokens(&self) -> i32 {
        self.plan.num_tokens
    }
    /// Immutable sources for one rank's admitted destinations. This includes
    /// all dependencies; ordinary admission may already hold some of them.
    pub fn objects(&self, rank: usize) -> Result<Vec<PreparedObject>, Error> {
        if rank >= self.plan.members {
            return Err(Error("unknown read owner".into()));
        }
        let mut sources = HashMap::new();
        for batch in &self.batches {
            let source = batch.source.source();
            if source.addresses.len() != batch.keys.len() || source.session.is_empty() {
                return Err(Error("prepared source shape mismatch".into()));
            }
            for (key, address) in batch.keys.iter().zip(&source.addresses) {
                sources.insert(key, (source.session.as_str(), *address));
            }
        }
        self.plan
            .objects
            .iter()
            .filter(|o| o.rank == rank)
            .map(|o| {
                let (session, address) = sources
                    .get(&o.key)
                    .ok_or_else(|| Error("missing prepared dependency".into()))?;
                if *address == 0 || address.checked_add(o.bytes).is_none() {
                    return Err(Error("invalid prepared address range".into()));
                }
                Ok(PreparedObject {
                    group_id: o.group_id,
                    slot: o.slot,
                    session: (*session).into(),
                    address: *address,
                    bytes: o.bytes,
                })
            })
            .collect()
    }
    /// Apply the agreed outcomes, one per reader member. A malformed/missing
    /// report cannot prove retirement. Release all retired sources even if one
    /// provider has restarted, expired its lease or failed its release RPC.
    pub fn finish(mut self, outcomes: &[ReadOutcome]) -> Completion {
        if outcomes.len() != self.plan.members || outcomes.iter().any(|o| !o.retired()) {
            return Completion::Unretired;
        }
        self.retired = true;
        let mut complete = outcomes.iter().all(|o| o.succeeded());
        for batch in self.batches.drain(..) {
            complete &= batch.source.release_retired().unwrap_or(false);
        }
        if complete {
            Completion::Complete
        } else {
            Completion::FailedRetired
        }
    }
}
impl<R: ReadingSource> Drop for ReadingPrefix<R> {
    fn drop(&mut self) {
        if !self.retired {
            if let Some(credit) = self.credit.take() {
                credit.quarantine();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer::{
        geometry::{DeviceField, HostGroup, HostPool, RankGeometry},
        prepared::read_spans,
        read::{Arena, PageReader},
    };
    use ts_scheduler::PeerPageView;

    // Substitute only the remote storage provider. Source bytes are copied by
    // the same scatter planner used for native reads, using a real CPU arena.
    #[derive(Default)]
    struct State {
        next: u64,
        live: HashSet<u64>,
        calls: usize,
        fail_call: Option<usize>,
        expire_call: Option<usize>,
    }
    #[derive(Clone)]
    struct Store {
        state: Arc<Mutex<State>>,
        values: BTreeMap<String, (String, Vec<u8>)>,
        arena: Arena,
    }
    struct Prepared {
        store: Store,
        id: u64,
        ready: Ready,
        expires: bool,
        release: bool,
    }
    struct Reading(Prepared);
    impl Drop for Prepared {
        fn drop(&mut self) {
            if self.release {
                self.store.state.lock().unwrap().live.remove(&self.id);
            }
        }
    }
    impl PreparedSource for Prepared {
        type Reading = Reading;
        fn begin_read(mut self, _: Duration) -> Result<Reading, Error> {
            if self.expires {
                return Err(Error("source TTL too short".into()));
            }
            self.release = false;
            Ok(Reading(self))
        }
    }
    impl ReadingSource for Reading {
        fn source(&self) -> &Ready {
            &self.0.ready
        }
        fn release_retired(self) -> Result<bool, Error> {
            Ok(self.0.store.state.lock().unwrap().live.remove(&self.0.id))
        }
    }
    impl Backend for Store {
        type Prepared = Prepared;
        fn query(&self, keys: &[String]) -> Result<Vec<Vec<Replica>>, Error> {
            Ok(keys
                .iter()
                .map(|key| {
                    self.values
                        .get(key)
                        .map(|(endpoint, bytes)| {
                            vec![Replica {
                                endpoint: endpoint.clone(),
                                bytes: bytes.len() as u64,
                                generation: [1, key.len() as u64],
                            }]
                        })
                        .unwrap_or_default()
                })
                .collect())
        }
        fn prepare(
            &self,
            endpoint: &str,
            keys: &[String],
            sizes: &[u64],
            generations: &[[u64; 2]],
        ) -> Result<Prepared, Error> {
            let mut state = self.state.lock().unwrap();
            state.calls += 1;
            if state.fail_call == Some(state.calls) {
                return Err(Error("source arena full".into()));
            }
            state.next += 1;
            let id = state.next;
            let expires = state.expire_call == Some(state.calls);
            state.live.insert(id);
            let mut addresses = Vec::new();
            assert_eq!(generations.len(), keys.len());
            for (i, (key, bytes)) in keys.iter().zip(sizes).enumerate() {
                assert_eq!(generations[i], [1, key.len() as u64]);
                let (owner, value) = &self.values[key];
                assert_eq!(owner, endpoint);
                assert_eq!(value.len() as u64, *bytes);
                let address = id * 1024 + i as u64 * 64;
                self.arena.write(address, value);
                addresses.push(address);
            }
            Ok(Prepared {
                store: self.clone(),
                id,
                expires,
                release: true,
                ready: Ready {
                    batch_id: id,
                    addresses,
                    session: endpoint.into(),
                    ttl_ms: 60_000,
                },
            })
        }
    }
    fn store() -> Store {
        Store {
            state: Arc::new(Mutex::new(State::default())),
            arena: Arena::new(100_000),
            values: BTreeMap::from([
                ("full".into(), ("owner-a".into(), vec![11; 16])),
                ("state".into(), ("owner-b".into(), vec![22; 16])),
            ]),
        }
    }
    const LIMITS: Limits = Limits {
        max_query_keys: 8,
        max_candidates: 4,
        max_prepare_bytes: 1024,
        max_source_batch_bytes: 64,
        max_source_batches: 4,
    };
    fn plan(members: usize) -> PrefixPlan {
        PrefixPlan {
            num_tokens: 128,
            members,
            objects: (0..members)
                .flat_map(|rank| {
                    [
                        Object {
                            key: "full".into(),
                            rank,
                            group_id: 0,
                            slot: 0,
                            bytes: 16,
                        },
                        Object {
                            key: "state".into(),
                            rank,
                            group_id: 1,
                            slot: 0,
                            bytes: 16,
                        },
                    ]
                })
                .collect(),
        }
    }
    fn prepared(store: &Store, budget: &SourceBudget, members: usize) -> PreparedPrefix<Prepared> {
        let hit = query(store, &[plan(members)], LIMITS).unwrap().unwrap();
        prepare(store, hit, LIMITS, budget).unwrap().unwrap()
    }

    #[test]
    fn a_raw_full_hit_without_the_checkpoint_is_not_a_resumable_hit() {
        let mut backend = store();
        backend.values.remove("state");
        assert!(query(&backend, &[plan(2)], LIMITS).unwrap().is_none());
        assert_eq!(backend.state.lock().unwrap().calls, 0);
    }
    #[test]
    fn only_a_complete_shorter_candidate_can_replace_an_incomplete_longer_one() {
        let backend = store();
        let mut long = plan(1);
        long.num_tokens = 256;
        long.objects[1].key = "state-at-256".into();
        let hit = query(&backend, &[long, plan(1)], LIMITS).unwrap().unwrap();
        assert_eq!(hit.num_tokens(), 128);
        assert_eq!(hit.bytes(), 32);
    }
    #[test]
    fn query_checks_all_owners_and_exact_object_sizes() {
        let backend = store();
        let mut candidate = plan(2);
        candidate.objects[3].key = "state-owner-1-missing".into();
        assert!(query(&backend, &[candidate], LIMITS).unwrap().is_none());
        let mut candidate = plan(1);
        candidate.objects[0].bytes = 15;
        assert!(query(&backend, &[candidate], LIMITS).unwrap().is_none());
    }
    #[test]
    fn queued_sources_are_bounded_and_cancellation_releases_them() {
        let backend = store();
        let budget = SourceBudget::new(32).unwrap();
        let ready = prepared(&backend, &budget, 2);
        assert_eq!(budget.usage(), (32, 0), "shared objects charged once");
        let hit = query(&backend, &[plan(1)], LIMITS).unwrap().unwrap();
        assert!(prepare(&backend, hit, LIMITS, &budget).unwrap().is_none());
        assert_eq!(backend.state.lock().unwrap().live.len(), 2);
        drop(ready);
        assert_eq!(budget.usage(), (0, 0));
        assert!(backend.state.lock().unwrap().live.is_empty());
    }
    #[test]
    fn partial_prepare_or_ttl_failure_releases_all_unused_sources() {
        for expire in [false, true] {
            let backend = store();
            let budget = SourceBudget::new(64).unwrap();
            if expire {
                backend.state.lock().unwrap().expire_call = Some(2);
            } else {
                backend.state.lock().unwrap().fail_call = Some(2);
            }
            let hit = query(&backend, &[plan(1)], LIMITS).unwrap().unwrap();
            let ready = prepare(&backend, hit, LIMITS, &budget);
            if expire {
                assert!(ready
                    .unwrap()
                    .unwrap()
                    .begin_read(Duration::from_secs(1))
                    .is_err());
            } else {
                assert!(ready.is_err());
            }
            assert!(backend.state.lock().unwrap().live.is_empty());
            assert_eq!(budget.usage(), (0, 0));
        }
    }
    #[test]
    fn unretired_or_missing_owner_reports_never_release_or_recycle_the_sources() {
        for outcomes in [
            vec![ReadOutcome::Unretired],
            vec![],
            vec![ReadOutcome::Complete, ReadOutcome::Complete],
        ] {
            let backend = store();
            let budget = SourceBudget::new(64).unwrap();
            let read = prepared(&backend, &budget, 1)
                .begin_read(Duration::from_secs(1))
                .unwrap();
            assert_eq!(read.finish(&outcomes), Completion::Unretired);
            assert_eq!(backend.state.lock().unwrap().live.len(), 2);
            assert_eq!(budget.usage(), (32, 32));
        }
    }
    #[test]
    fn source_expiry_or_owner_read_failure_prevents_publication() {
        for expire in [false, true] {
            let backend = store();
            let budget = SourceBudget::new(64).unwrap();
            let read = prepared(&backend, &budget, 2)
                .begin_read(Duration::from_secs(1))
                .unwrap();
            if expire {
                backend.state.lock().unwrap().live.remove(&1);
            }
            let outcome = if expire {
                ReadOutcome::Complete
            } else {
                ReadOutcome::FailedRetired
            };
            assert_eq!(
                read.finish(&[ReadOutcome::Complete, outcome]),
                Completion::FailedRetired
            );
            assert!(backend.state.lock().unwrap().live.is_empty());
            assert_eq!(budget.usage(), (0, 0));
        }
    }
    #[test]
    fn prepared_objects_from_multiple_owners_land_directly_and_release_together() {
        let backend = store();
        let budget = SourceBudget::new(64).unwrap();
        let read = prepared(&backend, &budget, 2)
            .begin_read(Duration::from_secs(1))
            .unwrap();
        for rank in 0..2 {
            let base = 50_000 + rank as u64 * 1024;
            let geometry = RankGeometry {
                session: "reader".into(),
                push: None,
                host: Some(HostPool {
                    address: 0,
                    groups: (0..2)
                        .map(|_| HostGroup {
                            parent_bytes: 16,
                            packing: 1,
                            block_bytes: 16,
                            field_offsets: vec![("state-or-kv".into(), 0)],
                        })
                        .collect(),
                }),
                device: (0..2)
                    .map(|group_id| DeviceField {
                        group_id,
                        field_id: "state-or-kv".into(),
                        address: base + group_id as u64 * 256,
                        page_stride: 32,
                        bytes: 16,
                    })
                    .collect(),
            };
            let destinations: Vec<_> = (0..2)
                .map(|group_id| PeerPageView {
                    group_id,
                    slot: 0,
                    tier: ts_scheduler::PeerTier::Device,
                    block_id: 1,
                })
                .collect();
            let sources = read.objects(rank).unwrap();
            for (session, spans) in read_spans(&sources, &geometry, &destinations).unwrap() {
                assert_eq!(
                    backend.arena.read_outcome(&session, &spans),
                    ReadOutcome::Complete
                );
            }
            for group in 0..2 {
                assert_eq!(
                    backend.arena.read_bytes(base + group * 256 + 32, 16),
                    vec![11 * (group as u8 + 1); 16]
                );
                assert_eq!(
                    backend.arena.read_bytes(base + group * 256 + 48, 16),
                    vec![0; 16],
                    "padding untouched"
                );
            }
        }
        assert_eq!(
            read.finish(&[ReadOutcome::Complete; 2]),
            Completion::Complete
        );
        assert!(backend.state.lock().unwrap().live.is_empty());
        assert_eq!(budget.usage(), (0, 0));
    }
    #[test]
    fn bounded_queries_and_prepares_reject_before_staging() {
        let backend = store();
        assert!(query(
            &backend,
            &[plan(1)],
            Limits {
                max_query_keys: 1,
                ..LIMITS
            }
        )
        .is_err());
        assert!(query(&backend, &[plan(1), plan(1)], LIMITS).is_err());
        assert!(query(
            &backend,
            &[plan(1)],
            Limits {
                max_prepare_bytes: 31,
                ..LIMITS
            }
        )
        .unwrap()
        .is_none());
        let hit = query(&backend, &[plan(1)], LIMITS).unwrap().unwrap();
        let budget = SourceBudget::new(64).unwrap();
        assert!(prepare(
            &backend,
            hit,
            Limits {
                max_source_batches: 1,
                ..LIMITS
            },
            &budget
        )
        .is_err());
        assert_eq!(backend.state.lock().unwrap().calls, 0);
        assert_eq!(budget.usage(), (0, 0));
    }
}
