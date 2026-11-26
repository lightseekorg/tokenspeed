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

//! The C++ scheduler shared by the native host and Python data plane.
//!
//! The scheduler core and its nanobind module are compiled into the host.
//! [`python_module`] imports that module into the running interpreter, so
//! the data plane and Rust use the same build: an [`ExecutionPlan`] the control thread gets from the
//! scheduler is the same C++ type the Python side reads, and crosses over by
//! moving into a Python object ([`ExecutionPlan::into_python`]).
//!
//! The scheduler itself is constructed in Python (its configuration is
//! derived there from the device build) and driven from Rust through
//! [`Scheduler::from_python`], without the GIL. The cache index of a shared
//! L2 pool's owner is built and driven the same way
//! ([`SharedL2Pool::from_python`]).

// The crate is the FFI boundary to C++ and the interpreter.
#![allow(unsafe_code)]

pub mod cache_events;
pub mod profile;

pub use cache_events::{CacheEventObserver, CacheObservationLosses};

use std::ptr::NonNull;

use pyo3::{ffi as pyffi, prelude::*};

use crate::profile::record_scheduler_span;

#[cxx::bridge(namespace = "ts_scheduler")]
mod ffi {
    /// Native scheduler observations, separate from scheduler feedback events.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CacheObservationKind {
        /// A slot acquired a request admission owner.
        SlotAssigned,
        /// The owning request-slot handle released its row.
        SlotReleased,
        /// A request's image was scheduled for restoration.
        RestoreScheduled,
        /// A request's acquired Host prefix was scheduled for copying.
        PrefixLoadScheduled,
    }

    /// Fixed-size metadata. Fields outside the selected kind are unused.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CacheObservation {
        /// The owner transition or scheduled operation represented.
        kind: CacheObservationKind,
        /// Destination or owned request-pool row index.
        slot: i32,
        /// Allocation generation of the owned row.
        slot_generation: u64,
        /// The immutable admission identity allocated by the output processor.
        admission_id: u64,
        /// Prior owner of this row, zero before its first owner.
        slot_previous_admission_id: u64,
        /// Wire operation index, scoped by operation kind and wrap generation.
        op_id: u32,
        /// Generation across 32-bit operation-index wraparound.
        op_generation: u64,
        /// The request's row before its image was captured.
        previous_slot: i32,
        /// Allocation generation of that previous row.
        previous_slot_generation: u64,
        /// One-based restore sequence within the admission.
        restore_sequence: u64,
        /// Submitted prompt plus accepted generated tokens at the snapshot.
        committed_tokens: i32,
        /// Exclusive cache frontier at the same snapshot.
        computed_tokens: i32,
        /// Submitted prompt token count.
        prompt_tokens: i32,
    }

    /// A request to admit (`tokenspeed::RequestSpec`).
    #[derive(Debug, Clone, PartialEq)]
    struct RequestSpec {
        /// Request id.
        request_id: String,
        /// The output processor's immutable identity for this admission.
        admission_id: u64,
        /// Router-only cache retention identity, empty when untracked.
        session_id: String,
        /// Shared cache owner, empty to retain session ownership.
        cache_group_id: String,
        /// Candidate shared-store checkpoints within this prompt.
        shared_prefix_tokens: Vec<i32>,
        /// The replica expected to hold this prompt's prefix, empty for
        /// none (peer prefix sharing).
        peer_holder: String,
        /// Prompt token ids.
        tokens: Vec<i32>,
        /// Generation limit the scheduler reserves for.
        max_new_tokens: i32,
        /// Most prompt tokens a prefix-cache hit may cover; `i32::MAX` is
        /// no bound.
        max_cached_prefix_tokens: i32,
    }

    /// The kind of a [`ForwardEvent`].
    #[derive(Debug)]
    enum ForwardEventKind {
        /// Tokens joined the request's history.
        ExtendResult,
        /// The request finished.
        Finish,
        /// The request was aborted.
        Abort,
        /// The next round's token reservation changed.
        UpdateReserveNumTokens,
        /// PD: the request's peers are connected (`pd::BootstrappedEvent`).
        PdBootstrapped,
        /// PD, P role under deferred decode allocation: every decode rank
        /// allocated the request's destination
        /// (`pd::DestinationReadyEvent`).
        PdDestinationReady,
        /// PD: the request's transfer failed; it may still be stopping
        /// (`pd::FailedEvent`).
        PdFailed,
        /// PD, P role: the request's KV reached the decode node
        /// (`pd::SucceededEvent`).
        PdSucceeded,
        /// PD, D role: the request's KV landed, with the prefill node's
        /// first token (`pd::RemotePrefillDoneEvent`).
        PdRemotePrefillDone,
        /// PD: a failed or cancelled transfer stopped on every rank; the
        /// request's pages may return (`pd::TransferStoppedEvent`).
        PdTransferStopped,
    }

    /// A forward event fed back to the scheduler. Fields other kinds do not
    /// use are left empty.
    #[derive(Debug, Clone, PartialEq)]
    struct ForwardEvent {
        /// What happened.
        kind: ForwardEventKind,
        /// The request it happened to.
        request_id: String,
        /// ExtendResult: tokens whose KV became request history.
        tokens: Vec<i32>,
        /// ExtendResult, P role's final chunk: the drafter's candidates.
        spec_candidate_ids: Vec<i32>,
        /// UpdateReserveNumTokens: tokens to reserve in the next round.
        reserve_num_tokens: i32,
        /// PdRemotePrefillDone: the prefill node's first token, -1 if none.
        bootstrap_token: i32,
    }

    /// The control plane's copy of a forward batch's per-request columns.
    #[derive(Debug, Clone, PartialEq)]
    struct ForwardBatchView {
        /// Request ids, one row each.
        request_ids: Vec<String>,
        /// Each row's request slot.
        request_pool_indices: Vec<i32>,
        /// Tokens each row feeds this forward.
        input_lengths: Vec<i32>,
        /// Prompt tokens each row has prefilled, including this forward.
        prefill_lengths: Vec<i32>,
        /// Extend rows: tokens already in the cache ahead of this chunk.
        extend_prefix_lens: Vec<i32>,
        /// Extend rows: cached tokens re-fed ahead of the chunk.
        extend_replay_lens: Vec<i32>,
        /// Extend (prefill) rows come first; the rest are decodes.
        num_extends: usize,
    }

    /// One cache group's block table over a batch's rows.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct BlockTableView {
        /// The group.
        group_id: String,
        /// Block-table slots per row.
        columns: usize,
        /// The rows' scheduler block ids, row-major, padded with -1.
        blocks: Vec<i32>,
    }

    /// What a PD transfer reads off a batch: its rows, their PD columns
    /// and their block tables.
    #[derive(Debug, Clone, PartialEq)]
    struct TransferBatchView {
        /// Request ids, one row each; extend rows first.
        request_ids: Vec<String>,
        /// Each row's request slot.
        request_pool_indices: Vec<i32>,
        /// Tokens each row feeds this forward.
        input_lengths: Vec<i32>,
        /// Each row's prompt length.
        prefill_lengths: Vec<i32>,
        /// Extend rows: tokens already in the cache ahead of this chunk.
        extend_prefix_lens: Vec<i32>,
        /// Extend rows, the D role's admission: every state group's
        /// prompt-end checkpoint was found locally.
        state_prefix_cached: Vec<bool>,
        /// Decode rows: the token each feeds; on the P role's remote decode,
        /// the request's first output token.
        decode_input_ids: Vec<i32>,
        /// Decode rows' drafter candidates, concatenated.
        spec_candidate_ids: Vec<i32>,
        /// Where each decode row's candidates start, and the end.
        spec_candidate_offsets: Vec<usize>,
        /// Extend rows.
        num_extends: usize,
        /// Every cache group's table.
        block_tables: Vec<BlockTableView>,
    }

    /// A request the scheduler finished while building a plan.
    #[derive(Debug, Clone, PartialEq)]
    struct SchedulerAbortView {
        /// The aborted request.
        request_id: String,
        /// Why, for the client.
        detail: String,
    }

    /// The kind of a plan's cache op batch.
    #[derive(Debug)]
    enum CacheOpKind {
        /// Device to Host L2 (`WriteBackBatch`).
        WriteBack,
        /// Host L2 to Device (`LoadBackBatch`).
        LoadBack,
        /// A retraction image's store (`SnapshotStoreBatch`).
        SnapshotStore,
        /// A retracted request's restore (`SnapshotRestoreBatch`).
        SnapshotRestore,
        /// The commit of the deferred decode state a plan's retractions
        /// published (`ExecutionPlan::state_commit_fence`).
        StateCommit,
    }

    /// The kind of a plan's storage operation.
    #[derive(Debug)]
    enum StorageOpKind {
        /// Which keys the store holds (`StorageQueryOperation`).
        Query,
        /// Fetch keys into Host blocks (`StorageGetOperation`).
        Get,
        /// Store Host blocks under keys (`StoragePutOperation`).
        Put,
        /// Cancel an earlier operation (`StorageCancelOperation`).
        Cancel,
    }

    /// A content-addressed cache key that a storage operation, a shared L2
    /// answer or a shared L2 op names (`CacheKey`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StorageKeyView {
        /// The cache group, by index.
        group_id: u32,
        /// The prefix page's content hash.
        content_hash: String,
        /// The block's ordinal within its prefix page.
        page_offset: i32,
    }

    /// A storage operation a plan starts on the cache owners' executors.
    #[derive(Debug, Clone, PartialEq)]
    struct StorageOpView {
        /// What it does.
        kind: StorageOpKind,
        /// Its id; a cancel names the operation it cancels.
        op_id: u64,
        /// The keys it names; none for a cancel.
        keys: Vec<StorageKeyView>,
        /// A get's destination or a put's source Host block per key; none
        /// for a query or a cancel.
        host_blocks: Vec<i32>,
    }

    /// One cache op a plan puts in flight; each is acknowledged on its own.
    #[derive(Debug, Clone, PartialEq)]
    struct CacheOpView {
        /// Its batch's kind.
        kind: CacheOpKind,
        /// The op id its acknowledgment names.
        op_id: u32,
    }

    /// The kind of a [`CacheEvent`].
    #[derive(Debug)]
    enum CacheEventKind {
        /// A write-back landed (`cache::WriteBackDone`).
        WriteBackDone,
        /// A load-back landed (`cache::LoadBackDone`).
        LoadBackDone,
        /// A snapshot store finished (`cache::SnapshotDone`).
        SnapshotDone,
        /// A restore finished (`cache::RestoreDone`).
        RestoreDone,
        /// The runtime committed a plan's retracted decode state
        /// (`cache::StateCommitDone`).
        StateCommitDone,
        /// A shared L2 pool write finished (`cache::SharedL2WriteDone`).
        SharedL2WriteDone,
    }

    /// A cache op's acknowledgment, fed back to the scheduler.
    #[derive(Debug, Clone, PartialEq)]
    struct CacheEvent {
        /// What finished.
        kind: CacheEventKind,
        /// The op it acknowledges.
        op_id: u32,
    }

    /// The round's storage clock, permissions and queue pressure
    /// (`cache::StorageTick`), as the cache owners agreed on them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct StorageTickView {
        /// The clock, in microseconds.
        now_us: i64,
        /// Whether the scheduler may plan forwards.
        allow_forward: bool,
        /// Whether it may start storage operations.
        allow_storage: bool,
        /// `StoragePressure`: operations queued in the executors.
        queued_operations: i64,
        /// `StoragePressure`: bytes queued in the executors.
        queued_bytes: i64,
        /// `StoragePressure`: the oldest queued operation's age.
        oldest_operation_us: i64,
    }

    /// A cancelled request on the storage path: detached from its storage
    /// work (`cache::StorageDetach`), or aborted outright when it holds no
    /// pages yet (`forward::Abort`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StorageCancellationView {
        /// The request.
        request_id: String,
        /// Abort it; else detach it.
        abort: bool,
    }

    /// A storage operation every cache owner completed (`cache::StorageDone`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StorageDoneView {
        /// The operation.
        op_id: u64,
        /// Per key: it succeeded on every owner.
        success: Vec<u8>,
        /// Every owner retired the native I/O.
        retired: bool,
        /// Per key: an owner confirmed the object missing or invalid.
        object_failures: Vec<u8>,
    }

    /// The scheduler's storage observations (`StorageStats`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    struct StorageStatsView {
        /// Storage operations in flight.
        operations: u64,
        /// Host blocks storage operations hold.
        held_blocks: u64,
        /// Old-generation tickets still charged to the operation limit.
        retained_operations: u64,
        /// Host references retained by old-generation tickets.
        retained_host_blocks: u64,
        /// A global operation or Host-reference limit is full with retained work.
        retained_capacity_exhausted: bool,
        /// Capacity warnings emitted on transitions into exhaustion, cumulative.
        retained_capacity_warnings: u64,
        /// Host blocks quarantined after unproven retirement.
        quarantined: u64,
        /// Keys known to be absent from the store.
        negative_keys: u64,
        /// Published blocks waiting to be backed up.
        backup_candidates: u64,
        /// Requests waiting on their storage preparation.
        waiting_requests: u64,
        /// Requests whose preparation is ready.
        ready_requests: u64,
        /// Preparations that timed out waiting, cumulative.
        wait_timeouts: u64,
        /// Ready preparations that expired, cumulative.
        ready_expirations: u64,
        /// The pressure gate is closed.
        pressure_gate_closed: bool,
        /// Preparations bypassed by pressure, cumulative.
        pressure_bypasses: u64,
        /// Backup candidates dropped by pressure, cumulative.
        pressure_dropped_backup_keys: u64,
        /// Backup candidates dropped on overflow, cumulative.
        overflow_dropped_backup_keys: u64,
        /// Backup candidates dropped with Host memory unavailable.
        unavailable_dropped_backup_keys: u64,
        /// Backup candidates dropped with writes disabled, cumulative.
        disabled_dropped_backup_keys: u64,
        /// Backup candidates dropped on cleanup, cumulative.
        cleanup_dropped_backup_keys: u64,
    }

    /// The kind of a plan's Disk operation (`DiskOpKind`).
    #[derive(Debug)]
    enum DiskOpKind {
        /// Disk to Host: a prefetch.
        Read,
        /// Host to Disk: a cleaning write.
        Write,
    }

    /// A Disk operation a plan starts on every cache owner's disk executor
    /// (`DiskOperation`, `docs/rfcs/0006-disk-tier.md`): row i moves group
    /// `group_ids[i]`'s block between Host block `host_blocks[i]` and Disk
    /// block `disk_blocks[i]`.
    #[derive(Debug, Clone, PartialEq)]
    struct DiskOpView {
        /// Which way.
        kind: DiskOpKind,
        /// Its id.
        op_id: u64,
        /// Each row's cache group, by index.
        group_ids: Vec<u32>,
        /// Each row's Host block.
        host_blocks: Vec<i32>,
        /// Each row's Disk block.
        disk_blocks: Vec<i32>,
    }

    /// A Disk operation every cache owner finished (`cache::DiskDone`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct DiskDoneView {
        /// The operation.
        op_id: u64,
        /// Every owner moved every row.
        success: bool,
    }

    /// The scheduler's Disk tier observations (`DiskStats`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    struct DiskStatsView {
        /// Disk operations in flight or holding a request's Host blocks.
        operations: u64,
        /// Host blocks Disk operations hold.
        held_host_blocks: u64,
        /// Host entries waiting to be cleaned.
        queued_cleanings: u64,
        /// Requests waiting for their prefetch.
        waiting_requests: u64,
        /// Requests holding a finished prefetch for their admission.
        ready_requests: u64,
        /// Disk entries.
        disk_blocks: u64,
        /// Empty Disk parents.
        free_disk_parents: u64,
        /// Blocks cleaned to Disk, cumulative.
        written_blocks: u64,
        /// Blocks prefetched from Disk, cumulative.
        read_blocks: u64,
        /// Failed prefetches, cumulative.
        failed_reads: u64,
        /// Failed cleaning writes, cumulative.
        failed_writes: u64,
        /// Cleaning candidates dropped, cumulative.
        dropped_cleanings: u64,
        /// Prefetches that timed out waiting, cumulative.
        wait_timeouts: u64,
        /// Ready prefetches that expired, cumulative.
        ready_expirations: u64,
    }

    /// Prefix-cache hits by tier, cumulative since start (`HostCacheStats`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    struct PrefixHitView {
        /// Prompt tokens of every admitted request.
        prompt_tokens: u64,
        /// Tokens served from the Device prefix cache (L1).
        device_hit_tokens: u64,
        /// Tokens the Host loads (L2 including anything L3 supplied).
        host_hit_tokens: u64,
        /// Remote Device prefix used beyond the local hit.
        peer_hit_tokens: u64,
        /// Token spans of successful storage keys (L3), counted per cache group.
        l3_hit_tokens: u64,
    }

    #[derive(Debug, Clone, PartialEq)]
    /// Snapshot in physical allocator units. The curve is a conservative
    /// fresh-request bound from the scheduler's own multi-group capacity model.
    struct AdmissionSnapshot {
        free_slots: i32,
        total_slots: i32,
        available_blocks: i32,
        total_blocks: i32,
        host_free_blocks: i32,
        host_total_blocks: i32,
        retracted: usize,
        waiting: usize,
        max_tokens: i32,
        capacity_token_step: i32,
        capacity_blocks: Vec<i64>,
    }

    /// What the prefill-delay vote reads from one rank
    /// (`Scheduler::ProbePrefillGate`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PrefillGateProbe {
        /// A resident chunk, or new prefill work at or below the watermark.
        wants_prefill: bool,
        /// Decode work the round would otherwise run.
        has_decode_work: bool,
    }

    /// Bounded historical shared-store publication proof.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SharedPrefixPublication {
        status: String,
        confirmed_prefix_tokens: i32,
        cache_epoch: u64,
    }

    /// Request and page counts.
    #[derive(Debug, Clone, PartialEq)]
    struct SchedulerSizes {
        /// Requests in the running batch: local prefill forwards, prompts
        /// whose prefill is done, and decodes. Remote prefilling, waiting and
        /// retracted requests are not running.
        running: usize,
        /// Resident requests holding a request-pool slot.
        allocated_request_slots: usize,
        /// Decode permits held by requests or their in-flight operations.
        active_decode_slots: usize,
        /// Resident requests whose KV is ready but that have no decode permit.
        ready_waiting: usize,
        /// Requests awaiting admission or restore.
        waiting: usize,
        /// Waiting prompts less observed cache hits, plus unscheduled prefill tokens.
        pending_prefill_tokens: i64,
        /// Requests prefilling.
        prefilling: usize,
        /// Requests decoding.
        decoding: usize,
        /// Requests suspended with an image.
        retracted: usize,
        /// PD: requests waiting for their peers to connect.
        bootstrapping: usize,
        /// PD, D role: prefilling requests whose KV the prefill node sends.
        remote_prefilling: usize,
        /// PD: requests whose pages a transfer still pins.
        pd_transfer: usize,
        /// P-role prompts with final output ready, awaiting PD completion.
        prefill_inflight: usize,
        /// Cache blocks holding nothing.
        empty_lcm_blocks: i32,
    }

    /// A prefix-cache mutation (`KvBlockStoredEvent` / `KvBlockRemovedEvent`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct KvEventView {
        /// Blocks were stored; else removed.
        stored: bool,
        /// The blocks' hashes, in chain order.
        block_hashes: Vec<u64>,
        /// Stored: the chain's parent has a hash.
        has_parent: bool,
        /// Stored: the parent's hash, when `has_parent`.
        parent_block_hash: u64,
        /// Stored: the blocks' tokens.
        token_ids: Vec<i32>,
        /// Stored: tokens per block.
        block_size: i32,
        /// The memory the event concerns, as a `KvEventMedium` value: 0 the
        /// Device, 1 the Host cache, 2 the Disk tier.
        medium: u8,
    }

    /// A peer lease's name: its reader and the reader's own counter
    /// (`tokenspeed::PeerLeaseId`).
    #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct PeerLeaseView {
        /// The reader replica's name.
        reader: String,
        /// The reader's counter.
        id: u64,
    }

    /// One block of a peer prefix (`tokenspeed::PeerPage`): a holder's
    /// cached block, or a reader's destination.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PeerPageView {
        /// The cache group.
        group_id: u32,
        /// The block's slot in the group's request table.
        slot: i32,
        /// Where the block is.
        tier: PeerTier,
        /// The scheduler block id on its tier.
        block_id: i32,
    }

    /// Where a peer page is (`tokenspeed::CacheTier`): a holder lists a page
    /// from its Device, its Host pool or its Disk tier; a reader lands into
    /// its Device or its Host pool.
    #[derive(Debug)]
    enum PeerTier {
        /// The Device cache arena.
        Device,
        /// The Host pool.
        Host,
        /// The Disk tier's file (`docs/rfcs/0006-disk-tier.md`).
        Disk,
    }

    /// A reader's lookup, to send to its holder (`tokenspeed::PeerLookupRequest`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PeerLookupView {
        /// Request owning this lookup.
        request_id: String,
        /// Deadline on the agreed scheduler clock.
        deadline_us: i64,
        /// The lease the lookup opens.
        lease: PeerLeaseView,
        /// The holder's name.
        holder: String,
        /// The prompt's full-page chain hashes below `max_tokens`.
        prefix_hashes: Vec<String>,
        /// The prompt's tokens up to `max_tokens`.
        tokens: Vec<i32>,
        /// The bound on the prefix.
        max_tokens: i32,
    }

    /// What a holder's answer answers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PeerAnswerKind {
        /// A lookup's answer.
        Lookup,
        /// A pin's answer.
        Pin,
        /// A release's answer.
        Release,
    }

    /// A holder's answer, to send to the lease's reader (`tokenspeed::PeerAnswer`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PeerAnswerView {
        /// The lease.
        lease: PeerLeaseView,
        /// What it answers.
        kind: PeerAnswerKind,
        /// Lookup and pin: the boundary held, 0 for none.
        num_tokens: i32,
        /// Pin: the pinned blocks.
        pages: Vec<PeerPageView>,
        /// Release: whether the hard pin held for its whole life.
        held: bool,
    }

    /// Why a hinted request entered local prefill.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PeerLocalChoice {
        /// Scheduler request ID.
        request_id: String,
        /// Bounded scheduler decision code.
        reason: String,
    }

    /// A peer fetch admitted by the scheduler.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PeerFetchView {
        /// The landing request.
        request_id: String,
        /// The lease it reads under.
        lease: PeerLeaseView,
        /// The holder's name.
        holder: String,
        /// Where the landed tokens begin: the local hit.
        begin_tokens: i32,
        /// Where they end: the holder's boundary.
        end_tokens: i32,
        /// The blocks it fills.
        destinations: Vec<PeerPageView>,
        /// Original holder answer for an exact re-PIN after admission.
        checkpoint: Vec<u8>,
        /// Prefix hashes at the original lookup boundary.
        prefix_hashes: Vec<String>,
        /// Prompt tokens through that boundary for exact checkpoint revalidation.
        tokens: Vec<i32>,
    }

    /// The kind of a [`PeerEventView`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PeerEventKind {
        /// Holder: a reader's lookup (`cache::PeerLookup`).
        Lookup,
        /// Holder: a reader's pin (`cache::PeerPin`).
        Pin,
        /// Holder: a reader's release (`cache::PeerRelease`).
        Release,
        /// Reader: a holder's answer to a lookup (`cache::PeerLookupAnswered`).
        LookupAnswered,
        /// Reader: a landing's outcome (`cache::PeerFetchDone`).
        FetchDone,
        /// Business fallback while the independently retained target may still be written.
        FetchFrozen,
        /// All members certified retirement of a previously frozen target.
        TargetRetired,
    }

    /// A peer event the replica agreed on. Fields other kinds do not use
    /// are left empty.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PeerEventView {
        /// What happened.
        kind: PeerEventKind,
        /// The lease it concerns (all but FetchDone).
        lease: PeerLeaseView,
        /// Lookup: the prompt's full-page chain hashes.
        prefix_hashes: Vec<String>,
        /// Lookup: the prompt's tokens up to `max_tokens`.
        tokens: Vec<i32>,
        /// Lookup: the bound on the prefix.
        max_tokens: i32,
        /// Pin: the pages the holder's answer named.
        pages: Vec<PeerPageView>,
        /// Release: the reader has the whole prefix.
        demote: bool,
        /// Pin: the boundary the holder's answer named; LookupAnswered: the
        /// holder's boundary; FetchDone: the boundary confirmed.
        num_tokens: i32,
        /// Reader-local monotonic times for adaptive Device feedback (zero when absent).
        dispatch_us: i64,
        /// First instant when the device fence and the fetch plan are both ready.
        writable_us: i64,
        /// RELEASE arrival on success, or protocol retirement on failure.
        terminal_us: i64,
        /// Reader-local original lookup answer.
        checkpoint: Vec<u8>,
        /// FetchDone: the landing request.
        request_id: String,
    }

    /// Duration and deadline headroom of a completed hard pin.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct PinSampleView {
        /// Time between applying and releasing the hard pin, in microseconds.
        held_us: i64,
        /// Signed microseconds left until the deadline at release.
        remaining_us: i64,
        /// Whether the holder reclaimed the pin on expiry.
        expired: bool,
    }

    /// The scheduler's peer sharing observations.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct PeerStatsView {
        /// Holder: leases held.
        leases: usize,
        /// Holder: blocks hard pins hold.
        pinned_blocks: usize,
        /// Reader: requests waiting for a lookup's answer.
        looking_requests: usize,
        /// Reader: requests landing.
        landing_requests: usize,
        /// Reader: tokens landed so far.
        landed_tokens: u64,
        /// Reader: landings whose transfer failed to complete the prefix.
        failed_landings: u64,
        /// Reader: successful Host landings whose request gave up the reservation.
        unused_host_landings: u64,
    }

    /// One cache group's page counts.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct CacheGroupPages {
        /// Pages the group has.
        total: i32,
        /// Pages it can still hand out.
        available: i32,
    }

    /// A waiting prompt's question to the shared L2 pool (`SharedL2Lookup`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SharedL2LookupView {
        /// The prompt's request, which its answer names.
        request_id: String,
        /// The chain hashes of the prompt's full pages below `max_tokens`.
        prefix_hashes: Vec<String>,
        /// The prompt's first `max_tokens` tokens.
        tokens: Vec<i32>,
        /// The most tokens a hit may cover.
        max_tokens: i32,
    }

    /// The pool's answer to a lookup (`SharedL2Answer`): its longest
    /// reusable prefix of the prompt, pinned under `ticket`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SharedL2AnswerView {
        /// The request whose lookup it answers.
        request_id: String,
        /// What the owner pinned the hit under; 0 pins nothing (a miss).
        ticket: u64,
        /// The tokens the hit resumes from.
        num_tokens: i32,
        /// How far the pool's KV reaches.
        kv_tokens: i32,
        /// Tokens of the hit's partial last page; 0 when it ends on a page.
        tail_tokens: i32,
        /// The partial last page's hash, with `tail_tokens`.
        tail_hash: String,
        /// The hit's keys.
        keys: Vec<StorageKeyView>,
        /// Each key's shard.
        buckets: Vec<i32>,
        /// Each key's pool page.
        pool_pages: Vec<i32>,
    }

    /// A proxy Host block an answer created, and the pool page a load-back
    /// from it reads (`SharedL2ProxyBlock`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SharedL2ProxyBlockView {
        /// The cache group, by index.
        group_id: u32,
        /// The proxy Host block.
        host_block: i32,
        /// The page's shard.
        bucket: i32,
        /// The pool page, within the group.
        pool_page: i32,
    }

    /// A State checkpoint, which a pool write commits with the
    /// checkpoint's block (`CheckpointTree::Checkpoint`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SharedL2CheckpointView {
        /// The checkpoint's tokens.
        tokens: Vec<i32>,
        /// The chain hashes of its full pages.
        prefix_hashes: Vec<String>,
        /// Tokens of its partial last page; 0 when it ends on a page.
        tail_tokens: i32,
        /// The partial last page's hash, with `tail_tokens`.
        tail_hash: String,
        /// The access epoch of the request that published it on this rank.
        access_epoch: u64,
    }

    /// The kind of a plan's shared L2 op.
    #[derive(Debug)]
    enum SharedL2OpKind {
        /// Copy Device blocks into the pool (`SharedL2WriteOperation`).
        Write,
        /// Unpin what a lookup hit (`SharedL2ReleaseOperation`).
        Release,
        /// Refresh the pool's copies of keys (`SharedL2TouchOperation`).
        Touch,
    }

    /// A shared L2 op a plan carries, for the rank's pool client.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SharedL2OpView {
        /// What it does.
        kind: SharedL2OpKind,
        /// A write's id; unused by the other kinds.
        op_id: u32,
        /// A release's ticket; unused by the other kinds.
        ticket: u64,
        /// A write's or a touch's keys.
        keys: Vec<StorageKeyView>,
        /// A write's shard per key.
        buckets: Vec<i32>,
        /// A write's source Device block per key.
        device_blocks: Vec<i32>,
        /// The checkpoints among a write's blocks.
        checkpoints: Vec<SharedL2CheckpointView>,
    }

    /// What a pool reservation granted (`SharedL2Pool::Reservation`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SharedL2ReservationView {
        /// What the pool reserves the pages under; 0 reserves nothing.
        ticket: u64,
        /// One per key, in order: the page to write, or why there is none
        /// (-1 the pool holds the key, -2 no page is free, -3 another
        /// reservation writes it).
        pages: Vec<i32>,
    }

    /// A pool's counters (`SharedL2Pool::Stats`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SharedL2PoolStatsView {
        /// LCM blocks the pool has.
        num_lcm_blocks: i32,
        /// LCM blocks no page is taken from.
        empty_lcm_blocks: i32,
        /// Blocks the pool caches.
        cached_blocks: i32,
        /// Cached blocks a lookup pins.
        pinned_blocks: i32,
        /// Blocks reserved for a write not committed yet.
        reserved_blocks: usize,
        /// Lookup tickets not released yet.
        open_lookups: usize,
        /// Blocks evicted so far.
        evicted_blocks: u64,
        /// State checkpoints the pool holds.
        checkpoints: usize,
    }

    unsafe extern "C++" {
        include!("ts-scheduler/cxx/facade.h");

        #[namespace = "tokenspeed"]
        type Scheduler;
        #[namespace = "tokenspeed"]
        type ExecutionPlan;

        fn cache_event_protocol_version() -> u32;

        fn install_cache_event_observer(
            scheduler: Pin<&mut Scheduler>,
            observer: Box<CacheEventObserver>,
        ) -> Result<()>;

        fn submit_requests(scheduler: Pin<&mut Scheduler>, specs: &[RequestSpec]) -> Result<()>;
        fn next_execution_plan(
            scheduler: Pin<&mut Scheduler>,
            allow_prefill: bool,
            apply_prefill_watermark: bool,
        ) -> UniquePtr<ExecutionPlan>;
        fn advance(scheduler: Pin<&mut Scheduler>, events: &[ForwardEvent]) -> Result<()>;
        fn advance_head(
            scheduler: Pin<&mut Scheduler>,
            ticks: &[StorageTickView],
            cancellations: &[StorageCancellationView],
            done: &[StorageDoneView],
            disk_done: &[DiskDoneView],
            cache_events: &[CacheEvent],
        ) -> Result<()>;
        fn advance_peer(
            scheduler: Pin<&mut Scheduler>,
            now_us: i64,
            events: &[PeerEventView],
        ) -> Result<()>;
        #[namespace = "tokenspeed"]
        type PeerPinRelease;
        fn peer_release_handle(
            scheduler: Pin<&mut Scheduler>,
            lease: &PeerLeaseView,
        ) -> SharedPtr<PeerPinRelease>;
        fn release_peer_pin(handle: &SharedPtr<PeerPinRelease>) -> bool;
        fn enable_async_host_peer(scheduler: Pin<&mut Scheduler>) -> bool;
        fn enable_async_device_peer(scheduler: Pin<&mut Scheduler>, adaptive: bool) -> bool;
        fn reserve_peer_host(
            scheduler: Pin<&mut Scheduler>,
            lease: &PeerLeaseView,
            tokens: i32,
            retry: &mut bool,
        ) -> Result<Vec<PeerFetchView>>;
        fn apply_peer_export(
            scheduler: Pin<&mut Scheduler>,
            event: &PeerEventView,
            now_us: i64,
        ) -> Result<bool>;
        fn pin_peer_offer(
            scheduler: Pin<&mut Scheduler>,
            event: &PeerEventView,
            now_us: i64,
        ) -> Result<bool>;
        fn close_peer_offer(
            scheduler: Pin<&mut Scheduler>,
            lease: &PeerLeaseView,
            now_us: i64,
        ) -> Result<bool>;
        fn take_peer_lookups(scheduler: Pin<&mut Scheduler>) -> Vec<PeerLookupView>;
        fn answer_peer_lookup(
            scheduler: &Scheduler,
            lookup: &PeerEventView,
            pending_new_leases: usize,
        ) -> PeerAnswerView;
        fn peer_pages_match(
            scheduler: &Scheduler,
            lookup: &PeerEventView,
            num_tokens: i32,
            pages: &[PeerPageView],
        ) -> bool;
        fn peer_lease_can_pin(scheduler: &Scheduler, lease: &PeerLeaseView) -> bool;
        fn peer_lease_held(scheduler: &Scheduler, lease: &PeerLeaseView) -> bool;
        fn plan_calls(scheduler: &Scheduler) -> i64;
        fn take_peer_pin_samples(scheduler: Pin<&mut Scheduler>) -> Vec<PinSampleView>;
        fn peer_stats(scheduler: &Scheduler) -> PeerStatsView;
        fn demote_sessions(
            scheduler: Pin<&mut Scheduler>,
            session_ids: &[String],
            ended: bool,
        ) -> usize;
        fn demote_session(scheduler: Pin<&mut Scheduler>, session_id: &str, ended: bool) -> usize;
        fn demote_cache_groups(
            scheduler: Pin<&mut Scheduler>,
            cache_group_ids: &[String],
            ended: bool,
        ) -> usize;
        fn admission_snapshot(scheduler: &Scheduler, token_step: i32) -> AdmissionSnapshot;
        fn shared_prefix_storage_enabled(scheduler: &Scheduler) -> bool;
        fn shared_prefix_publication(
            scheduler: &Scheduler,
            input_ids: &[i32],
            max_prefix_tokens: i32,
        ) -> Result<SharedPrefixPublication>;
        fn sizes(scheduler: &Scheduler) -> SchedulerSizes;
        fn active_lcm_blocks(scheduler: &Scheduler) -> i32;
        fn probe_prefill_gate(scheduler: &Scheduler) -> PrefillGateProbe;
        fn cache_group_pages(scheduler: &Scheduler, group_id: &str) -> CacheGroupPages;
        #[namespace = "tokenspeed"]
        type KvEventBatch;
        fn drain_kv_events(scheduler: Pin<&mut Scheduler>) -> UniquePtr<KvEventBatch>;
        fn make_kv_event_batch(events: &[KvEventView]) -> UniquePtr<KvEventBatch>;
        fn kv_event_batch_size(batch: &KvEventBatch) -> usize;
        fn kv_event_batch_bytes(batch: &KvEventBatch) -> usize;
        fn kv_event_stored(batch: &KvEventBatch) -> &[u8];
        fn kv_event_has_parent(batch: &KvEventBatch) -> &[u8];
        fn kv_event_parents(batch: &KvEventBatch) -> &[u64];
        fn kv_event_block_sizes(batch: &KvEventBatch) -> &[i32];
        fn kv_event_hash_offsets(batch: &KvEventBatch) -> &[usize];
        fn kv_event_token_offsets(batch: &KvEventBatch) -> &[usize];
        fn kv_event_hashes(batch: &KvEventBatch) -> &[u64];
        fn kv_event_tokens(batch: &KvEventBatch) -> &[i32];
        fn kv_event_media(batch: &KvEventBatch) -> &[u8];
        fn request_token_size(scheduler: &Scheduler, request_id: &str) -> i32;
        fn can_abort_before_admission(scheduler: &Scheduler, request_id: &str) -> bool;
        fn defers_decode_state(scheduler: &Scheduler) -> bool;
        fn storage_pending(scheduler: &Scheduler) -> bool;
        fn storage_current_pending(scheduler: &Scheduler) -> bool;
        fn storage_progress_pending(scheduler: &Scheduler) -> bool;
        fn storage_blocking_quarantined(scheduler: &Scheduler) -> usize;
        fn storage_non_storage_clearable(scheduler: &Scheduler) -> bool;
        fn can_prepare_storage_clear(scheduler: &Scheduler) -> bool;
        fn storage_stats(scheduler: &Scheduler) -> StorageStatsView;
        fn disk_stats(scheduler: &Scheduler) -> DiskStatsView;
        fn prefix_hit_stats(scheduler: &Scheduler) -> PrefixHitView;
        fn clear_cache(scheduler: Pin<&mut Scheduler>) -> bool;
        fn cancel_storage_backups(scheduler: Pin<&mut Scheduler>);
        fn cancel_storage_for_clear(scheduler: Pin<&mut Scheduler>);
        fn can_clear_cache_retaining_storage(scheduler: &Scheduler) -> bool;
        fn clear_cache_retaining_storage(scheduler: Pin<&mut Scheduler>) -> bool;
        fn storage_has_retained_operations(scheduler: &Scheduler) -> bool;
        fn clear_l1_cache(scheduler: Pin<&mut Scheduler>) -> bool;
        fn can_clear_cache(scheduler: &Scheduler) -> bool;
        fn max_single_request_tokens(scheduler: &Scheduler) -> i32;
        fn start_shared_l2_lookups(scheduler: Pin<&mut Scheduler>) -> Vec<SharedL2LookupView>;
        fn apply_shared_l2_answers(
            scheduler: Pin<&mut Scheduler>,
            answers: &[SharedL2AnswerView],
        ) -> Result<Vec<SharedL2ProxyBlockView>>;

        fn plan_forward_batches(plan: &ExecutionPlan) -> Vec<ForwardBatchView>;
        fn plan_completed_prompts(plan: &ExecutionPlan) -> Vec<String>;
        fn plan_needs_device(plan: &ExecutionPlan) -> bool;
        fn plan_has_pages_to_copy(plan: &ExecutionPlan) -> bool;
        fn plan_remote_decode(plan: &ExecutionPlan) -> Vec<ForwardBatchView>;
        fn plan_remote_prefill(plan: &ExecutionPlan) -> Vec<ForwardBatchView>;
        fn plan_forward_transfer(plan: &ExecutionPlan) -> Vec<TransferBatchView>;
        fn plan_remote_decode_transfer(plan: &ExecutionPlan) -> Vec<TransferBatchView>;
        fn plan_remote_prefill_transfer(plan: &ExecutionPlan) -> Vec<TransferBatchView>;
        fn plan_aborts(plan: &ExecutionPlan) -> Vec<SchedulerAbortView>;
        fn plan_cache_ops(plan: &ExecutionPlan) -> Vec<CacheOpView>;
        fn plan_storage_ops(plan: &ExecutionPlan) -> Vec<StorageOpView>;
        fn plan_disk_ops(plan: &ExecutionPlan) -> Vec<DiskOpView>;
        fn plan_shared_l2_ops(plan: &ExecutionPlan) -> Vec<SharedL2OpView>;
        fn plan_peer_local_choices(plan: &ExecutionPlan) -> Vec<PeerLocalChoice>;
        fn plan_peer_fetches(plan: &ExecutionPlan) -> Vec<PeerFetchView>;

        #[namespace = "tokenspeed"]
        type SharedL2Pool;
        fn shared_l2_pool_lookup(
            pool: Pin<&mut SharedL2Pool>,
            client: u64,
            lookup: &SharedL2LookupView,
        ) -> Result<SharedL2AnswerView>;
        fn shared_l2_pool_release(
            pool: Pin<&mut SharedL2Pool>,
            client: u64,
            ticket: u64,
        ) -> Result<()>;
        fn shared_l2_pool_reserve(
            pool: Pin<&mut SharedL2Pool>,
            client: u64,
            keys: &[StorageKeyView],
            buckets: &[i32],
        ) -> Result<SharedL2ReservationView>;
        fn shared_l2_pool_commit(
            pool: Pin<&mut SharedL2Pool>,
            client: u64,
            ticket: u64,
            checkpoints: &[SharedL2CheckpointView],
        ) -> Result<()>;
        fn shared_l2_pool_touch(
            pool: Pin<&mut SharedL2Pool>,
            keys: &[StorageKeyView],
        ) -> Result<()>;
        fn shared_l2_pool_disconnect(pool: Pin<&mut SharedL2Pool>, client: u64);
        fn shared_l2_pool_clear_cache(pool: Pin<&mut SharedL2Pool>) -> bool;
        fn shared_l2_pool_stats(pool: &SharedL2Pool) -> SharedL2PoolStatsView;

        /// Install the scheduler's profile span sink, which forwards to
        /// [`record_scheduler_span`], or remove it.
        fn set_profile_span_sink(installed: bool);
    }

    extern "Rust" {
        type CacheEventObserver;
        fn record_cache_observation(
            observer: &CacheEventObserver,
            value: CacheObservation,
            request_id: &[u8],
        );
        /// The C++ scheduler's finished spans, while the sink is installed.
        fn record_scheduler_span(name: &str, begin_ns: i64, end_ns: i64);
    }
}

pub use ffi::{
    BlockTableView, CacheEvent, CacheEventKind, CacheGroupPages, CacheObservation,
    CacheObservationKind, CacheOpKind, CacheOpView, DiskDoneView, DiskOpKind, DiskOpView,
    DiskStatsView, ForwardBatchView, ForwardEvent, ForwardEventKind, KvEventView, PeerAnswerKind,
    PeerAnswerView, PeerEventKind, PeerEventView, PeerFetchView, PeerLeaseView, PeerLookupView,
    PeerPageView, PeerStatsView, PeerTier, PinSampleView, PrefillGateProbe, PrefixHitView,
    RequestSpec, SchedulerAbortView, SchedulerSizes, SharedL2AnswerView, SharedL2CheckpointView,
    SharedL2LookupView, SharedL2OpKind, SharedL2OpView, SharedL2PoolStatsView,
    SharedL2ProxyBlockView, SharedL2ReservationView, StorageCancellationView, StorageDoneView,
    StorageKeyView, StorageOpKind, StorageOpView, StorageStatsView, StorageTickView,
    TransferBatchView,
};

fn record_cache_observation(
    observer: &CacheEventObserver,
    value: CacheObservation,
    request_id: &[u8],
) {
    observer.record(value, request_id);
}

/// Version implemented by the linked native ownership producer.
pub fn cache_event_protocol_version() -> u32 {
    ffi::cache_event_protocol_version()
}

/// What advances the scheduler at the head of a round: with L3 or the Disk
/// tier, the agreed storage clock, the cancellations and the storage and
/// Disk completions; then the Host copies' acknowledgments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StorageRound {
    /// The agreed clock and permissions; `None` without L3 or a Disk tier.
    pub tick: Option<StorageTickView>,
    /// Requests cancelled since the last round, in order.
    pub cancellations: Vec<StorageCancellationView>,
    /// Storage operations every owner completed, by op id.
    pub done: Vec<StorageDoneView>,
    /// Disk operations every owner finished, by op id.
    pub disk_done: Vec<DiskDoneView>,
}

extern "C" {
    fn PyInit_tokenspeed_scheduler_ext() -> *mut pyffi::PyObject;
    fn ts_scheduler_from_python(object: *mut pyffi::PyObject) -> *mut ffi::Scheduler;
    fn ts_shared_l2_pool_from_python(object: *mut pyffi::PyObject) -> *mut ffi::SharedL2Pool;
    fn ts_scheduler_plan_into_python(plan: *mut ffi::ExecutionPlan) -> *mut pyffi::PyObject;
}

// Execute through a Python frame, as a normal extension import does. nanobind
// installs a Python-level atexit hook using the current frame's globals; native
// Rust callers do not otherwise have such a frame.
#[pyo3::pyfunction]
fn execute_scheduler_module(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // SAFETY: the callable holds the GIL, and the linked initializer returns
    // this module's static definition. FromDefAndSpec2 has created the module.
    unsafe {
        let definition = PyInit_tokenspeed_scheduler_ext().cast::<pyffi::PyModuleDef>();
        if definition.is_null() || pyffi::PyModule_ExecDef(module.as_ptr(), definition) != 0 {
            return Err(PyErr::fetch(module.py()));
        }
    }
    Ok(())
}

/// Import the scheduler linked into the host, once per interpreter.
///
/// nanobind uses multi-phase initialization. The same module owns the Python
/// types and the C++ objects driven by the Rust control loop.
pub fn python_module(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
    const NAME: &str = "tokenspeed_engine._scheduler";
    let modules = py.import("sys")?.getattr("modules")?;
    if modules.contains(NAME)? {
        return Ok(modules.get_item(NAME)?.cast_into()?);
    }
    let spec = py
        .import("importlib.util")?
        .call_method1("spec_from_loader", (NAME, py.None()))?;
    // SAFETY: the GIL is held. nanobind >=2.13 returns a static PyModuleDef,
    // and FromDefAndSpec2 returns a new module reference owned by Bound.
    let module = unsafe {
        let definition = PyInit_tokenspeed_scheduler_ext().cast::<pyffi::PyModuleDef>();
        if definition.is_null() {
            return Err(PyErr::fetch(py));
        }
        let api = if cfg!(feature = "abi3") {
            3
        } else {
            pyffi::PYTHON_API_VERSION
        };
        let module = Bound::from_owned_ptr_or_err(
            py,
            pyffi::PyModule_FromDefAndSpec2(definition, spec.as_ptr(), api),
        )?;
        module.cast_into::<pyo3::types::PyModule>()?
    };
    let scope = pyo3::types::PyDict::new(py);
    scope.set_item("module", &module)?;
    scope.set_item(
        "execute",
        pyo3::wrap_pyfunction!(execute_scheduler_module, py)?,
    )?;
    py.run(c"execute(module)", Some(&scope), None)?;
    modules.set_item(NAME, &module)?;
    Ok(module)
}

/// A scheduler owned by a Python `tokenspeed_scheduler.Scheduler` object and
/// driven from Rust.
///
/// Python must not call into the scheduler while this exists; the control
/// thread is its only user.
pub struct Scheduler {
    raw: NonNull<ffi::Scheduler>,
    /// Keeps the C++ scheduler alive.
    _owner: Py<PyAny>,
}

// SAFETY: the scheduler has no thread affinity; exclusive access goes through
// `&mut self`.
unsafe impl Send for Scheduler {}

// Unpin and owner-side revocation synchronize inside the ticket.
/// A revocable handle to one native lease's existing strong references.
pub struct PeerReleaseHandle(cxx::SharedPtr<ffi::PeerPinRelease>);
impl PeerReleaseHandle {
    /// Returns after all of this lease's strong references were dropped.
    pub fn release(&self) -> bool {
        ffi::release_peer_pin(&self.0)
    }
}
use ffi::PeerPinRelease;
unsafe impl Send for PeerPinRelease {}
unsafe impl Sync for PeerPinRelease {}

impl Scheduler {
    /// A revocable, thread-safe release handle for an existing hard pin.
    pub fn peer_release_handle(&mut self, lease: &PeerLeaseView) -> PeerReleaseHandle {
        PeerReleaseHandle(ffi::peer_release_handle(self.pinned(), lease))
    }

    /// Drive the scheduler a Python `tokenspeed_scheduler.Scheduler` owns.
    pub fn from_python(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        // SAFETY: the GIL is held (`Bound`), and the returned pointer lives as
        // long as `object`, which `_owner` keeps alive.
        let raw = unsafe { ts_scheduler_from_python(object.as_ptr()) };
        let raw = NonNull::new(raw).ok_or_else(|| PyErr::fetch(object.py()))?;
        Ok(Self {
            raw,
            _owner: object.clone().unbind(),
        })
    }

    fn pinned(&mut self) -> std::pin::Pin<&mut ffi::Scheduler> {
        // SAFETY: `raw` is valid (see `from_python`) and `&mut self` makes this
        // the only reference. The C++ object never moves.
        unsafe { std::pin::Pin::new_unchecked(self.raw.as_mut()) }
    }

    fn get(&self) -> &ffi::Scheduler {
        // SAFETY: as in `pinned`.
        unsafe { self.raw.as_ref() }
    }

    /// Install diagnostics before any admission; native ownership retains the observer.
    pub fn install_cache_event_observer(
        &mut self,
        observer: CacheEventObserver,
    ) -> Result<(), cxx::Exception> {
        ffi::install_cache_event_observer(self.pinned(), Box::new(observer))
    }

    /// Queue requests for admission. Validation errors reject the entire
    /// batch without inserting any request.
    pub fn submit_requests(&mut self, specs: &[RequestSpec]) -> Result<(), cxx::Exception> {
        let _span = profile::span("Scheduler::submit_requests");
        ffi::submit_requests(self.pinned(), specs)
    }

    /// Plan the next round. `allow_prefill` gates the fused role's prefill
    /// work; `apply_prefill_watermark` checks pressure before new admissions
    /// when delay is enabled. Resident chunks ignore pressure, and PD roles
    /// ignore both options (see `docs/design/prefill-delay.md`).
    pub fn next_execution_plan(
        &mut self,
        allow_prefill: bool,
        apply_prefill_watermark: bool,
    ) -> ExecutionPlan {
        let _span = profile::span("Scheduler::next_execution_plan");
        ExecutionPlan(ffi::next_execution_plan(
            self.pinned(),
            allow_prefill,
            apply_prefill_watermark,
        ))
    }

    /// Feed forward results back. Fails, applying none of `events`' later
    /// entries, on an event the request's state does not allow.
    pub fn advance(&mut self, events: &[ForwardEvent]) -> Result<(), cxx::Exception> {
        let _span = profile::span("Scheduler::advance");
        ffi::advance(self.pinned(), events)
    }

    /// Feed the round head's storage events and cache-op acknowledgments
    /// back, as one event batch.
    pub fn advance_head(
        &mut self,
        storage: &StorageRound,
        cache_events: &[CacheEvent],
    ) -> Result<(), cxx::Exception> {
        let _span = profile::span("Scheduler::advance_head");
        let ticks: &[StorageTickView] = storage.tick.as_slice();
        ffi::advance_head(
            self.pinned(),
            ticks,
            &storage.cancellations,
            &storage.done,
            &storage.disk_done,
            cache_events,
        )
    }

    /// With a shared L2 pool: the lookups of the waiting prompts not handed
    /// out yet, each once. A prompt is admitted only once its lookup is
    /// answered; one whose Device prefix covers what it may reuse asks
    /// nothing.
    pub fn start_shared_l2_lookups(&mut self) -> Vec<SharedL2LookupView> {
        let _span = profile::span("Scheduler::start_shared_l2_lookups");
        ffi::start_shared_l2_lookups(self.pinned())
    }

    /// Settle the lookups with the pool's answers, in order. Returns the
    /// proxy Host blocks created, for the pool's loader. Fails on an answer
    /// the scheduler cannot take: one no lookup waits for, or that does not
    /// fit its prompt; the answers after it are not applied.
    pub fn apply_shared_l2_answers(
        &mut self,
        answers: &[SharedL2AnswerView],
    ) -> Result<Vec<SharedL2ProxyBlockView>, cxx::Exception> {
        let _span = profile::span("Scheduler::apply_shared_l2_answers");
        ffi::apply_shared_l2_answers(self.pinned(), answers)
    }

    /// Feed the round head's peer events back: the agreed peer clock, then
    /// the events in the order the replica agreed on (peer prefix sharing,
    /// `docs/rfcs/0004-peer-prefix-sharing.md`).
    pub fn advance_peer(
        &mut self,
        now_us: i64,
        events: &[PeerEventView],
    ) -> Result<(), cxx::Exception> {
        let _span = profile::span("Scheduler::advance_peer");
        ffi::advance_peer(self.pinned(), now_us, events)
    }

    /// Enable independent Device reads with the supplied admission behavior.
    pub fn enable_async_device_peer(&mut self, adaptive: bool) -> bool {
        ffi::enable_async_device_peer(self.pinned(), adaptive)
    }
    /// Enable independent reservations into the Host landing tier.
    pub fn enable_async_host_peer(&mut self) -> bool {
        ffi::enable_async_host_peer(self.pinned())
    }

    /// Reserve unpublished Host pages for a live reader lease.
    pub fn reserve_peer_host(
        &mut self,
        lease: &PeerLeaseView,
        tokens: i32,
        retry: &mut bool,
    ) -> Result<Vec<PeerFetchView>, cxx::Exception> {
        ffi::reserve_peer_host(self.pinned(), lease, tokens, retry)
    }

    /// Change only the holder's export ledger while holding the cache mutex.
    pub fn apply_peer_export(
        &mut self,
        event: &PeerEventView,
        now_us: i64,
    ) -> Result<bool, cxx::Exception> {
        ffi::apply_peer_export(self.pinned(), event, now_us)
    }

    /// Acquire one exact v1 member ticket, isolated from legacy timers.
    pub fn pin_peer_offer(
        &mut self,
        event: &PeerEventView,
        now_us: i64,
    ) -> Result<bool, cxx::Exception> {
        ffi::pin_peer_offer(self.pinned(), event, now_us)
    }

    /// Release this member's exact v1 ticket without demoting the source.
    pub fn close_peer_offer(
        &mut self,
        lease: &PeerLeaseView,
        now_us: i64,
    ) -> Result<bool, cxx::Exception> {
        ffi::close_peer_offer(self.pinned(), lease, now_us)
    }

    /// A reader's lookups since the last take, to send to their holders.
    pub fn take_peer_lookups(&mut self) -> Vec<PeerLookupView> {
        ffi::take_peer_lookups(self.pinned())
    }

    /// A holder's answer to `lookup` (a [`PeerEventKind::Lookup`] event),
    /// read from the cache as it stands; it changes nothing. Bounded by the
    /// `pending_new_leases` unsynced lookup hits already queued in the holder
    /// servicer.
    pub fn answer_peer_lookup_bounded(
        &self,
        lookup: &PeerEventView,
        pending_new_leases: usize,
    ) -> PeerAnswerView {
        ffi::answer_peer_lookup(self.get(), lookup, pending_new_leases)
    }

    /// Whether `lookup`'s prompt, up to `num_tokens`, still resolves to
    /// exactly `pages`.
    pub fn peer_pages_match(
        &self,
        lookup: &PeerEventView,
        num_tokens: i32,
        pages: &[PeerPageView],
    ) -> bool {
        ffi::peer_pages_match(self.get(), lookup, num_tokens, pages)
    }

    /// Whether `lease` still holds an active soft pin eligible to become hard.
    pub fn peer_lease_can_pin(&self, lease: &PeerLeaseView) -> bool {
        ffi::peer_lease_can_pin(self.get(), lease)
    }

    /// Whether `lease`'s pin holds its pages now.
    pub fn peer_lease_held(&self, lease: &PeerLeaseView) -> bool {
        ffi::peer_lease_held(self.get(), lease)
    }

    /// Plans built so far: no cache entry leaves between two plans.
    pub fn plan_calls(&self) -> i64 {
        ffi::plan_calls(self.get())
    }

    /// Drain hard pins released or expired since the last sample.
    pub fn take_peer_pin_samples(&mut self) -> Vec<PinSampleView> {
        ffi::take_peer_pin_samples(self.pinned())
    }

    /// Current holder and reader state.
    pub fn peer_stats(&self) -> PeerStatsView {
        ffi::peer_stats(self.get())
    }

    /// Apply one batch with a single scan of each cache index.
    pub fn demote_sessions(&mut self, session_ids: &[String], ended: bool) -> usize {
        ffi::demote_sessions(self.pinned(), session_ids, ended)
    }

    /// Lower only these groups' exclusive cache entries; never invalidates active blocks.
    pub fn demote_cache_groups(&mut self, cache_group_ids: &[String], ended: bool) -> usize {
        ffi::demote_cache_groups(self.pinned(), cache_group_ids, ended)
    }

    /// Lower only this session's exclusive cache entries in the eviction order.
    pub fn demote_session(&mut self, session_id: &str, ended: bool) -> usize {
        ffi::demote_session(self.pinned(), session_id, ended)
    }

    /// Whether this rank has the complete prefix publication path configured.
    pub fn shared_prefix_storage_enabled(&self) -> bool {
        ffi::shared_prefix_storage_enabled(self.get())
    }

    /// Query exact prefix receipts without waiting or acquiring cache references.
    pub fn shared_prefix_publication(
        &self,
        input_ids: &[i32],
        max_prefix_tokens: i32,
    ) -> Result<ffi::SharedPrefixPublication, cxx::Exception> {
        ffi::shared_prefix_publication(self.get(), input_ids, max_prefix_tokens)
    }

    /// Poll native capacity; token_step must be positive.
    pub fn admission_snapshot(&self, token_step: i32) -> ffi::AdmissionSnapshot {
        ffi::admission_snapshot(self.get(), token_step)
    }

    /// Request and physical block counts: a pass over the requests per
    /// state, and the empty-block count, which the pool keeps.
    pub fn sizes(&self) -> SchedulerSizes {
        let _span = profile::span("Scheduler::sizes");
        ffi::sizes(self.get())
    }

    /// Distinct LCM blocks held by running requests' block tables. The
    /// tables keep the count as they change, so reading it walks nothing.
    pub fn active_lcm_blocks(&self) -> i32 {
        let _span = profile::span("Scheduler::active_lcm_blocks");
        ffi::active_lcm_blocks(self.get())
    }

    /// What the prefill-delay vote should publish for this rank.
    pub fn probe_prefill_gate(&self) -> PrefillGateProbe {
        let _span = profile::span("Scheduler::probe_prefill_gate");
        ffi::probe_prefill_gate(self.get())
    }

    /// One cache group's page counts.
    pub fn cache_group_pages(&self, group_id: &str) -> CacheGroupPages {
        ffi::cache_group_pages(self.get(), group_id)
    }

    /// The request's token count, or a negative value for a request the
    /// scheduler does not hold.
    pub fn request_token_size(&self, request_id: &str) -> i32 {
        ffi::request_token_size(self.get(), request_id)
    }

    /// Whether the request holds no pages yet, so an abort can finish it
    /// without the forward or PD completion fence.
    pub fn can_abort_before_admission(&self, request_id: &str) -> bool {
        ffi::can_abort_before_admission(self.get(), request_id)
    }

    /// Whether a cache group defers its decode state to a request's end
    /// (`CacheGroupConfig::deferred_decode_state`): a finish only the host
    /// saw must then end with `ForwardEventKind::Abort`, which publishes
    /// nothing.
    pub fn defers_decode_state(&self) -> bool {
        ffi::defers_decode_state(self.get())
    }

    /// Whether storage operations are in flight.
    pub fn storage_pending(&self) -> bool {
        ffi::storage_pending(self.get())
    }

    /// Current-generation tickets remain, including unknown retirement.
    pub fn storage_current_pending(&self) -> bool {
        ffi::storage_current_pending(self.get())
    }

    /// Current storage work can still advance through completion or deadlines.
    /// Retained and terminal-unknown tickets keep their resources but do not count.
    pub fn storage_progress_pending(&self) -> bool {
        ffi::storage_progress_pending(self.get())
    }

    /// Current-generation quarantined tickets blocking new storage work.
    pub fn storage_blocking_quarantined(&self) -> usize {
        ffi::storage_blocking_quarantined(self.get())
    }

    /// Ordinary indexed-pin preflight after storage references drain.
    pub fn storage_non_storage_clearable(&self) -> bool {
        ffi::storage_non_storage_clearable(self.get())
    }

    /// Ordinary clear preparation, discounting only real storage pins on
    /// indexed Host entries. Recovery requires its separate exact preflight.
    pub fn can_prepare_storage_clear(&self) -> bool {
        ffi::can_prepare_storage_clear(self.get())
    }

    /// The scheduler's storage observations.
    pub fn storage_stats(&self) -> StorageStatsView {
        ffi::storage_stats(self.get())
    }

    /// The Disk tier's observations; all zero without one.
    pub fn disk_stats(&self) -> DiskStatsView {
        ffi::disk_stats(self.get())
    }

    /// Cumulative prefix-cache hits by tier.
    pub fn prefix_hit_stats(&self) -> PrefixHitView {
        ffi::prefix_hit_stats(self.get())
    }

    /// The prefix-cache mutations since the last drain, in order; empty
    /// unless the scheduler was configured to record them.
    pub fn drain_kv_events(&mut self) -> KvEventBatch {
        let _span = profile::span("Scheduler::drain_kv_events");
        KvEventBatch(ffi::drain_kv_events(self.pinned()))
    }

    /// Drop the Device L1 and Host L2 prefix indexes (the public flush).
    /// Refused (`false`) while cache ops are in flight, entries are pinned or
    /// a request is suspended with an image.
    pub fn clear_cache(&mut self) -> bool {
        ffi::clear_cache(self.pinned())
    }

    /// Abandon best-effort PUTs before a flush. The next plan carries any
    /// runtime cancellations; dispatched sources stay pinned until retired.
    pub fn cancel_storage_backups(&mut self) {
        ffi::cancel_storage_backups(self.pinned());
    }

    /// After replica-agreed non-storage preflight, detach request storage
    /// leases and backups so healthy work can drain without attached holds.
    pub fn cancel_storage_for_clear(&mut self) {
        ffi::cancel_storage_for_clear(self.pinned());
    }

    /// Whether a full clear may retain only the storage tickets' Host blocks.
    /// Other cache pins, Device transfers and suspended requests still refuse.
    pub fn can_clear_cache_retaining_storage(&self) -> bool {
        ffi::can_clear_cache_retaining_storage(self.get())
    }

    /// Clear cache indexes while preserving detached old storage references.
    /// The runtime must move subsequent work to a fresh cache generation.
    pub fn clear_cache_retaining_storage(&mut self) -> bool {
        ffi::clear_cache_retaining_storage(self.pinned())
    }

    /// Old storage references survived a clear and still consume capacity.
    pub fn storage_has_retained_operations(&self) -> bool {
        ffi::storage_has_retained_operations(self.get())
    }

    /// Drop the Device L1 prefix index only, under the same refusals.
    pub fn clear_l1_cache(&mut self) -> bool {
        ffi::clear_l1_cache(self.pinned())
    }

    /// Whether current references permit a full clear, without cancelling
    /// backups or changing any state.
    pub fn can_clear_cache(&self) -> bool {
        ffi::can_clear_cache(self.get())
    }

    /// The most tokens one request can hold in the cache.
    pub fn max_single_request_tokens(&self) -> i32 {
        ffi::max_single_request_tokens(self.get())
    }
}

/// A shared L2 pool owned by a Python `tokenspeed_scheduler.SharedL2Pool`
/// object and driven from Rust: the cache index the pool's owner keeps over
/// the providers' pages.
///
/// Python must not call into the pool while this exists; the owner's serving
/// thread is its only user.
pub struct SharedL2Pool {
    raw: NonNull<ffi::SharedL2Pool>,
    /// Keeps the C++ pool alive.
    _owner: Py<PyAny>,
}

// SAFETY: the pool has no thread affinity; exclusive access goes through
// `&mut self`.
unsafe impl Send for SharedL2Pool {}

impl SharedL2Pool {
    /// Drive the pool a Python `tokenspeed_scheduler.SharedL2Pool` owns.
    pub fn from_python(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        // SAFETY: the GIL is held (`Bound`), and the returned pointer lives as
        // long as `object`, which `_owner` keeps alive.
        let raw = unsafe { ts_shared_l2_pool_from_python(object.as_ptr()) };
        let raw = NonNull::new(raw).ok_or_else(|| PyErr::fetch(object.py()))?;
        Ok(Self {
            raw,
            _owner: object.clone().unbind(),
        })
    }

    fn pinned(&mut self) -> std::pin::Pin<&mut ffi::SharedL2Pool> {
        // SAFETY: `raw` is valid (see `from_python`) and `&mut self` makes this
        // the only reference. The C++ object never moves.
        unsafe { std::pin::Pin::new_unchecked(self.raw.as_mut()) }
    }

    /// The pool's longest reusable prefix of `lookup`'s prompt, pinned for
    /// `client` under the answer's ticket until the client releases it; a
    /// miss pins nothing (ticket 0). Fails on tokens short of the lookup's
    /// bound.
    pub fn lookup(
        &mut self,
        client: u64,
        lookup: &SharedL2LookupView,
    ) -> Result<SharedL2AnswerView, cxx::Exception> {
        ffi::shared_l2_pool_lookup(self.pinned(), client, lookup)
    }

    /// Unpin what a lookup hit under `ticket`. Fails for a ticket no lookup
    /// of `client` holds.
    pub fn release(&mut self, client: u64, ticket: u64) -> Result<(), cxx::Exception> {
        ffi::shared_l2_pool_release(self.pinned(), client, ticket)
    }

    /// Reserve a page for each of `keys` that the pool neither holds nor
    /// has reserved, in the key's bucket, for `client` to write and commit.
    /// Fails on a key or a bucket that is not of the pool.
    pub fn reserve(
        &mut self,
        client: u64,
        keys: &[StorageKeyView],
        buckets: &[i32],
    ) -> Result<SharedL2ReservationView, cxx::Exception> {
        ffi::shared_l2_pool_reserve(self.pinned(), client, keys, buckets)
    }

    /// Publish the blocks reserved under `ticket`, with the checkpoints
    /// whose State block is among them. Fails, with no change, for a ticket
    /// under which `client` reserves nothing or a checkpoint whose tokens do
    /// not cover its pages and tail.
    pub fn commit(
        &mut self,
        client: u64,
        ticket: u64,
        checkpoints: &[SharedL2CheckpointView],
    ) -> Result<(), cxx::Exception> {
        ffi::shared_l2_pool_commit(self.pinned(), client, ticket, checkpoints)
    }

    /// Refresh the pool's copies of `keys`: a rank used its own.
    pub fn touch(&mut self, keys: &[StorageKeyView]) -> Result<(), cxx::Exception> {
        ffi::shared_l2_pool_touch(self.pinned(), keys)
    }

    /// End everything `client` holds: its lookups' pins and its
    /// reservations.
    pub fn disconnect(&mut self, client: u64) {
        ffi::shared_l2_pool_disconnect(self.pinned(), client);
    }

    /// Empty the pool. Refused (`false`) while a client holds pins or
    /// reservations.
    pub fn clear_cache(&mut self) -> bool {
        ffi::shared_l2_pool_clear_cache(self.pinned())
    }

    /// The pool's counters.
    pub fn stats(&self) -> SharedL2PoolStatsView {
        // SAFETY: as in `pinned`.
        ffi::shared_l2_pool_stats(unsafe { self.raw.as_ref() })
    }
}

/// An owned, immutable cache-event snapshot in contiguous native columns.
/// It has no reference to a request, scheduler, or cache allocation and may be
/// moved to the publisher for encoding and destruction without holding the GIL.
/// A null native pointer is the allocation-free empty batch.
pub struct KvEventBatch(cxx::UniquePtr<ffi::KvEventBatch>);

impl Default for KvEventBatch {
    fn default() -> Self {
        Self(cxx::UniquePtr::null())
    }
}

// SAFETY: this uniquely owned snapshot contains only vectors of scalar data.
// Native mutation is not exposed, and no scheduler/Python/Device pointer is held.
unsafe impl Send for KvEventBatch {}

/// One borrowed row; its slices live as long as the batch that owns them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvEvent<'a> {
    /// A stored boundary, otherwise a removal.
    pub stored: bool,
    /// Block hashes, including all hashes in a compatible multi-hash event.
    pub block_hashes: &'a [u64],
    /// Whether the parent hash is present (zero itself is a valid hash).
    pub has_parent: bool,
    /// The parent when `has_parent` is true.
    pub parent_block_hash: u64,
    /// Token ids carried by a store.
    pub token_ids: &'a [i32],
    /// Tokens in a full block.
    pub block_size: i32,
    /// The memory the event concerns.
    pub medium: KvEventMedium,
}

/// The memory a KV event concerns (`tokenspeed::KvEventMedium`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvEventMedium {
    /// The GPU cache.
    Device,
    /// The pinned Host cache (Host L2).
    Host,
    /// The Disk tier (`docs/rfcs/0006-disk-tier.md`).
    Disk,
}
impl KvEventBatch {
    /// Copy independent event rows into an owned batch, preserving wire fields.
    /// Serving receives the already-flat native snapshot from `drain_kv_events`.
    ///
    /// Panics if an absent parent carries a hash, or a removal carries store
    /// fields that the wire cannot represent.
    pub fn from_events(events: &[KvEventView]) -> Self {
        for event in events {
            assert!(
                event.has_parent || event.parent_block_hash == 0,
                "an absent parent must not carry a hash"
            );
            assert!(
                event.stored
                    || (!event.has_parent && event.token_ids.is_empty() && event.block_size == 0),
                "a removal must not carry a store payload"
            );
        }
        Self(ffi::make_kv_event_batch(events))
    }
    /// Number of events in this snapshot.
    pub fn len(&self) -> usize {
        self.0.as_ref().map_or(0, ffi::kv_event_batch_size)
    }
    /// Whether this snapshot has no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Logical column bytes; allocated capacity and allocator metadata are excluded.
    pub fn byte_size(&self) -> usize {
        self.0.as_ref().map_or(0, ffi::kv_event_batch_bytes)
    }
    /// Iterate without allocating per-row vectors or copying token/hash data.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = KvEvent<'_>> {
        let columns = self.0.as_ref();
        let stored = columns.map_or(&[][..], ffi::kv_event_stored);
        let parents = columns.map_or(&[][..], ffi::kv_event_parents);
        let has_parent = columns.map_or(&[][..], ffi::kv_event_has_parent);
        let block_sizes = columns.map_or(&[][..], ffi::kv_event_block_sizes);
        let hash_offsets = columns.map_or(&[][..], ffi::kv_event_hash_offsets);
        let token_offsets = columns.map_or(&[][..], ffi::kv_event_token_offsets);
        let hashes = columns.map_or(&[][..], ffi::kv_event_hashes);
        let tokens = columns.map_or(&[][..], ffi::kv_event_tokens);
        let media = columns.map_or(&[][..], ffi::kv_event_media);
        (0..stored.len()).map(move |i| KvEvent {
            stored: stored[i] != 0,
            block_hashes: &hashes[hash_offsets[i]..hash_offsets[i + 1]],
            has_parent: has_parent[i] != 0,
            parent_block_hash: parents[i],
            token_ids: &tokens[token_offsets[i]..token_offsets[i + 1]],
            block_size: block_sizes[i],
            medium: match media[i] {
                0 => KvEventMedium::Device,
                1 => KvEventMedium::Host,
                _ => KvEventMedium::Disk,
            },
        })
    }
}
impl std::fmt::Debug for KvEventBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// One round's plan.
pub struct ExecutionPlan(cxx::UniquePtr<ffi::ExecutionPlan>);

// SAFETY: a plan is plain data with no thread affinity.
unsafe impl Send for ExecutionPlan {}

impl ExecutionPlan {
    /// The model batches, in plan order.
    pub fn forward_batches(&self) -> Vec<ForwardBatchView> {
        let _span = profile::span("ExecutionPlan::forward_batches");
        ffi::plan_forward_batches(&self.0)
    }

    /// The requests whose prompt's last chunk the plan's batches carry
    /// (`ForwardBatch::CompletesPrompt`).
    pub fn completed_prompts(&self) -> Vec<String> {
        ffi::plan_completed_prompts(&self.0)
    }

    /// Whether the data plane has anything to do for this plan: a batch,
    /// a Host copy (cache op), pages to zero or copy, or a remote stream.
    /// Storage operations are the L3 executor's and shared L2 ops
    /// the pool client's, not the data plane's.
    pub fn needs_device(&self) -> bool {
        ffi::plan_needs_device(&self.0)
    }

    /// Whether the plan relocates or copies any Device cache blocks
    /// (`pages_to_copy`), whose destinations enter `PrefixCacheIndex` during
    /// `next_execution_plan` before `copy_cache_pages` runs on the device.
    pub fn has_pages_to_copy(&self) -> bool {
        ffi::plan_has_pages_to_copy(&self.0)
    }

    /// P role: the prompts whose KV goes to their decode node now.
    pub fn remote_decode(&self) -> Option<ForwardBatchView> {
        ffi::plan_remote_decode(&self.0).into_iter().next()
    }

    /// D role: the prompts whose pages were admitted for their prefill
    /// node's KV.
    pub fn remote_prefill(&self) -> Option<ForwardBatchView> {
        ffi::plan_remote_prefill(&self.0).into_iter().next()
    }

    /// The first model batch, with what a PD transfer reads off it (the P
    /// role's layerwise streaming).
    pub fn forward_transfer(&self) -> Option<TransferBatchView> {
        ffi::plan_forward_transfer(&self.0).into_iter().next()
    }

    /// [`ExecutionPlan::remote_decode`], with what a PD transfer reads off
    /// it.
    pub fn remote_decode_transfer(&self) -> Option<TransferBatchView> {
        ffi::plan_remote_decode_transfer(&self.0).into_iter().next()
    }

    /// [`ExecutionPlan::remote_prefill`], with what a PD transfer reads off
    /// it.
    pub fn remote_prefill_transfer(&self) -> Option<TransferBatchView> {
        ffi::plan_remote_prefill_transfer(&self.0)
            .into_iter()
            .next()
    }

    /// Requests the scheduler finished while planning.
    pub fn aborts(&self) -> Vec<SchedulerAbortView> {
        ffi::plan_aborts(&self.0)
    }

    /// The cache ops the plan puts in flight, in plan order.
    pub fn cache_ops(&self) -> Vec<CacheOpView> {
        let _span = profile::span("ExecutionPlan::cache_ops");
        ffi::plan_cache_ops(&self.0)
    }

    /// The storage operations the plan starts, in plan order; the data plane
    /// runs them on the cache owners' storage executors.
    pub fn storage_ops(&self) -> Vec<StorageOpView> {
        ffi::plan_storage_ops(&self.0)
    }

    /// The Disk operations the plan starts, in plan order; the control plane
    /// runs them on the cache owners' disk executors.
    pub fn disk_ops(&self) -> Vec<DiskOpView> {
        ffi::plan_disk_ops(&self.0)
    }

    /// The shared L2 ops the plan carries, in plan order; the
    /// rank's pool client runs them.
    pub fn shared_l2_ops(&self) -> Vec<SharedL2OpView> {
        ffi::plan_shared_l2_ops(&self.0)
    }

    /// Hinted requests this plan admitted locally, with their last decision.
    pub fn peer_local_choices(&self) -> Vec<ffi::PeerLocalChoice> {
        ffi::plan_peer_local_choices(&self.0)
    }

    /// Reads granted by the scheduler.
    pub fn peer_fetches(&self) -> Vec<PeerFetchView> {
        ffi::plan_peer_fetches(&self.0)
    }

    /// Hand the plan to Python as a `tokenspeed_scheduler.ExecutionPlan`.
    pub fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let raw = self.0.into_raw();
        // SAFETY: the GIL is held, and the function takes ownership of `raw`
        // whether it succeeds or not.
        let object = unsafe { ts_scheduler_plan_into_python(raw) };
        // SAFETY: a non-null result is a new reference.
        unsafe { Bound::from_owned_ptr_or_err(py, object) }.map(Bound::unbind)
    }
}
