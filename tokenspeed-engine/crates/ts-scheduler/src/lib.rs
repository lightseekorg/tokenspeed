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

//! The C++ scheduler for a binary that embeds the Python data plane.
//!
//! The scheduler core and its nanobind module are compiled into the binary.
//! [`register_builtin_module`] makes the module a built-in of the embedded
//! interpreter, so the data plane's `import tokenspeed_scheduler` resolves to
//! this build: an [`ExecutionPlan`] the control thread gets from the
//! scheduler is the same C++ type the Python side reads, and crosses over by
//! moving into a Python object ([`ExecutionPlan::into_python`]).
//!
//! The scheduler itself is constructed in Python (its configuration is
//! derived there from the device build) and driven from Rust through
//! [`Scheduler::from_python`], without the GIL.

// The crate is the FFI boundary to C++ and the interpreter.
#![allow(unsafe_code)]

use std::ptr::NonNull;

use pyo3::{ffi as pyffi, prelude::*};

#[cxx::bridge(namespace = "ts_scheduler")]
mod ffi {
    /// A request to admit (`tokenspeed::RequestSpec`).
    #[derive(Debug, Clone, PartialEq)]
    struct RequestSpec {
        /// Request id.
        request_id: String,
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
        /// PD: the request's transfer failed (`pd::FailedEvent`).
        PdFailed,
        /// PD, P role: the request's KV reached the decode node
        /// (`pd::SucceededEvent`).
        PdSucceeded,
        /// PD, D role: the request's KV landed, with the prefill node's
        /// first token (`pd::RemotePrefillDoneEvent`).
        PdRemotePrefillDone,
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

    /// A content-addressed cache key a storage operation names (`CacheKey`).
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

    /// Request and page counts.
    #[derive(Debug, Clone, PartialEq)]
    struct SchedulerSizes {
        /// Requests awaiting admission or restore.
        waiting: usize,
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
        /// Cache blocks holding nothing.
        empty_lcm_blocks: i32,
        /// Cache blocks held by running requests.
        active_lcm_blocks: i32,
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
    }

    /// One cache group's page counts.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct CacheGroupPages {
        /// Pages the group has.
        total: i32,
        /// Pages it can still hand out.
        available: i32,
    }

    unsafe extern "C++" {
        include!("ts-scheduler/cxx/facade.h");

        #[namespace = "tokenspeed"]
        type Scheduler;
        #[namespace = "tokenspeed"]
        type ExecutionPlan;

        fn submit_requests(scheduler: Pin<&mut Scheduler>, specs: &[RequestSpec]);
        fn next_execution_plan(scheduler: Pin<&mut Scheduler>) -> UniquePtr<ExecutionPlan>;
        fn advance(scheduler: Pin<&mut Scheduler>, events: &[ForwardEvent]) -> Result<()>;
        fn advance_head(
            scheduler: Pin<&mut Scheduler>,
            ticks: &[StorageTickView],
            cancellations: &[StorageCancellationView],
            done: &[StorageDoneView],
            cache_events: &[CacheEvent],
        ) -> Result<()>;
        fn sizes(scheduler: &Scheduler) -> SchedulerSizes;
        fn cache_group_pages(scheduler: &Scheduler, group_id: &str) -> CacheGroupPages;
        fn drain_kv_events(scheduler: Pin<&mut Scheduler>) -> Vec<KvEventView>;
        fn request_token_size(scheduler: &Scheduler, request_id: &str) -> i32;
        fn can_abort_before_admission(scheduler: &Scheduler, request_id: &str) -> bool;
        fn storage_pending(scheduler: &Scheduler) -> bool;
        fn storage_stats(scheduler: &Scheduler) -> StorageStatsView;
        fn clear_cache(scheduler: Pin<&mut Scheduler>) -> bool;
        fn clear_l1_cache(scheduler: Pin<&mut Scheduler>) -> bool;
        fn can_clear_cache(scheduler: &Scheduler) -> bool;
        fn max_single_request_tokens(scheduler: &Scheduler) -> i32;

        fn plan_forward_batches(plan: &ExecutionPlan) -> Vec<ForwardBatchView>;
        fn plan_needs_device(plan: &ExecutionPlan) -> bool;
        fn plan_remote_decode(plan: &ExecutionPlan) -> Vec<ForwardBatchView>;
        fn plan_remote_prefill(plan: &ExecutionPlan) -> Vec<ForwardBatchView>;
        fn plan_forward_transfer(plan: &ExecutionPlan) -> Vec<TransferBatchView>;
        fn plan_remote_decode_transfer(plan: &ExecutionPlan) -> Vec<TransferBatchView>;
        fn plan_remote_prefill_transfer(plan: &ExecutionPlan) -> Vec<TransferBatchView>;
        fn plan_aborts(plan: &ExecutionPlan) -> Vec<SchedulerAbortView>;
        fn plan_cache_ops(plan: &ExecutionPlan) -> Vec<CacheOpView>;
        fn plan_storage_ops(plan: &ExecutionPlan) -> Vec<StorageOpView>;
    }
}

pub use ffi::{
    BlockTableView, CacheEvent, CacheEventKind, CacheGroupPages, CacheOpKind, CacheOpView,
    ForwardBatchView, ForwardEvent, ForwardEventKind, KvEventView, RequestSpec, SchedulerAbortView,
    SchedulerSizes, StorageCancellationView, StorageDoneView, StorageKeyView, StorageOpKind,
    StorageOpView, StorageStatsView, StorageTickView, TransferBatchView,
};

/// What advances the scheduler at the head of a round: with L3, the agreed
/// storage clock, the cancellations and the storage completions; then the
/// Host copies' acknowledgments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StorageRound {
    /// The agreed clock and permissions; `None` without L3.
    pub tick: Option<StorageTickView>,
    /// Requests cancelled since the last round, in order.
    pub cancellations: Vec<StorageCancellationView>,
    /// Storage operations every owner completed, by op id.
    pub done: Vec<StorageDoneView>,
}

extern "C" {
    fn PyInit_tokenspeed_scheduler_ext() -> *mut pyffi::PyObject;
    fn ts_scheduler_from_python(object: *mut pyffi::PyObject) -> *mut ffi::Scheduler;
    fn ts_scheduler_plan_into_python(plan: *mut ffi::ExecutionPlan) -> *mut pyffi::PyObject;
}

/// The built-in module's name; the package imports it as a submodule.
const MODULE_NAME: &str = "tokenspeed_scheduler_ext";
const PACKAGE_MODULE_NAME: &str = "tokenspeed_scheduler.tokenspeed_scheduler_ext";

/// Register the scheduler's module as a built-in of the interpreter this
/// process is about to start.
///
/// # Panics
///
/// If the interpreter is already running: built-ins are fixed at start.
pub fn register_builtin_module() {
    // SAFETY: plain reads of interpreter state; registration only touches the
    // inittab, which is valid before initialization.
    unsafe {
        assert_eq!(
            pyffi::Py_IsInitialized(),
            0,
            "register the scheduler module before the interpreter starts"
        );
        let status = pyffi::PyImport_AppendInittab(
            c"tokenspeed_scheduler_ext".as_ptr(),
            Some(PyInit_tokenspeed_scheduler_ext),
        );
        assert_eq!(status, 0, "PyImport_AppendInittab failed");
    }
}

/// Make `import tokenspeed_scheduler` use the built-in module rather than
/// any compiled copy on `sys.path`. Call once, right after the interpreter
/// starts and before anything imports the package.
pub fn alias_builtin_module(py: Python<'_>) -> PyResult<()> {
    let module = py.import(MODULE_NAME)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(PACKAGE_MODULE_NAME, module)
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

impl Scheduler {
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

    /// Queue requests for admission.
    pub fn submit_requests(&mut self, specs: &[RequestSpec]) {
        ffi::submit_requests(self.pinned(), specs);
    }

    /// Plan the next round.
    pub fn next_execution_plan(&mut self) -> ExecutionPlan {
        ExecutionPlan(ffi::next_execution_plan(self.pinned()))
    }

    /// Feed forward results back. Fails, applying none of `events`' later
    /// entries, on an event the request's state does not allow.
    pub fn advance(&mut self, events: &[ForwardEvent]) -> Result<(), cxx::Exception> {
        ffi::advance(self.pinned(), events)
    }

    /// Feed the round head's storage events and cache-op acknowledgments
    /// back, as one event batch.
    pub fn advance_head(
        &mut self,
        storage: &StorageRound,
        cache_events: &[CacheEvent],
    ) -> Result<(), cxx::Exception> {
        let ticks: &[StorageTickView] = storage.tick.as_slice();
        ffi::advance_head(
            self.pinned(),
            ticks,
            &storage.cancellations,
            &storage.done,
            cache_events,
        )
    }

    /// Request and page counts.
    pub fn sizes(&self) -> SchedulerSizes {
        ffi::sizes(self.get())
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

    /// Whether storage operations are in flight.
    pub fn storage_pending(&self) -> bool {
        ffi::storage_pending(self.get())
    }

    /// The scheduler's storage observations.
    pub fn storage_stats(&self) -> StorageStatsView {
        ffi::storage_stats(self.get())
    }

    /// The prefix-cache mutations since the last drain, in order; empty
    /// unless the scheduler was configured to record them.
    pub fn drain_kv_events(&mut self) -> Vec<KvEventView> {
        ffi::drain_kv_events(self.pinned())
    }

    /// Drop the Device L1 and Host L2 prefix indexes (the public flush).
    /// Refused (`false`) while cache ops are in flight, entries are pinned or
    /// a request is suspended with an image.
    pub fn clear_cache(&mut self) -> bool {
        ffi::clear_cache(self.pinned())
    }

    /// Drop the Device L1 prefix index only, under the same refusals.
    pub fn clear_l1_cache(&mut self) -> bool {
        ffi::clear_l1_cache(self.pinned())
    }

    /// Whether [`Scheduler::clear_cache`] would succeed now; changes nothing.
    pub fn can_clear_cache(&self) -> bool {
        ffi::can_clear_cache(self.get())
    }

    /// The most tokens one request can hold in the cache.
    pub fn max_single_request_tokens(&self) -> i32 {
        ffi::max_single_request_tokens(self.get())
    }
}

/// One round's plan.
pub struct ExecutionPlan(cxx::UniquePtr<ffi::ExecutionPlan>);

// SAFETY: a plan is plain data with no thread affinity.
unsafe impl Send for ExecutionPlan {}

impl ExecutionPlan {
    /// The model batches, in plan order.
    pub fn forward_batches(&self) -> Vec<ForwardBatchView> {
        ffi::plan_forward_batches(&self.0)
    }

    /// Whether the data plane has anything to do for this plan: a batch,
    /// a Host copy (cache op), pages to zero or copy, or a remote stream.
    /// Storage operations are the L3 executor's, not the data plane's.
    pub fn needs_device(&self) -> bool {
        ffi::plan_needs_device(&self.0)
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
        ffi::plan_cache_ops(&self.0)
    }

    /// The storage operations the plan starts, in plan order; the data plane
    /// runs them on the cache owners' storage executors.
    pub fn storage_ops(&self) -> Vec<StorageOpView> {
        ffi::plan_storage_ops(&self.0)
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
