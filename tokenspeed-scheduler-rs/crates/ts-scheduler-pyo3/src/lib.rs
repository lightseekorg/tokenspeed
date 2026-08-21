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

//! Python bindings for the Rust scheduler — pyo3 port of
//! `tokenspeed-scheduler/bindings/python_module.cpp` (nanobind).
//!
//! The Python API surface must stay byte-identical: same module
//! (`tokenspeed_scheduler.tokenspeed_scheduler_ext`), same classes, same field
//! names, defaults, and validation messages (the pytest suite is the gate).
//!
//! This crate is the white-listed `unsafe` exception; all scheduler logic lives
//! in `ts-scheduler-core` (`#![forbid(unsafe_code)]`).
//!
//! useless_conversion fires on pyo3 #[pymethods] macro expansions of
//! PyResult<T> return types; it is a macro artifact, not our code.

#![allow(clippy::useless_conversion)]

use numpy::PyArray2;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule as PyModuleType};
use std::collections::BTreeMap;
use ts_scheduler_core::{
    cache_config::{CacheGroupConfig, CacheGroupFamily, CacheTransferPolicy, Retention},
    events::{self, Event, ExecutionEvent, ExecutionPlan, ForwardBatch, Operation},
    kv_events::KvCacheEvent,
    request_spec::RequestSpec,
    tier::{CacheOperation, LoadBackBatch, WriteBackBatch},
    types::{Role, SchedulerConfig},
    Scheduler,
};

// ─── Enums ──────────────────────────────────────────────────────────────────

/// Scheduler role (C++ `Role`); exposed as `SchedulerConfig.Role`.
#[pyclass(eq, eq_int, name = "Role")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyRole {
    P,
    D,
    Fused,
}

impl From<PyRole> for Role {
    fn from(role: PyRole) -> Self {
        match role {
            PyRole::P => Role::P,
            PyRole::D => Role::D,
            PyRole::Fused => Role::Fused,
        }
    }
}

impl From<Role> for PyRole {
    fn from(role: Role) -> Self {
        match role {
            Role::P => PyRole::P,
            Role::D => PyRole::D,
            Role::Fused => PyRole::Fused,
        }
    }
}

/// Cache retention policy.
#[pyclass(eq, eq_int, name = "CacheRetention")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyCacheRetention {
    FullHistory,
    SlidingWindow,
}

impl From<PyCacheRetention> for Retention {
    fn from(r: PyCacheRetention) -> Self {
        match r {
            PyCacheRetention::FullHistory => Retention::FullHistory,
            PyCacheRetention::SlidingWindow => Retention::SlidingWindow,
        }
    }
}

impl From<Retention> for PyCacheRetention {
    fn from(r: Retention) -> Self {
        match r {
            Retention::FullHistory => PyCacheRetention::FullHistory,
            Retention::SlidingWindow => PyCacheRetention::SlidingWindow,
        }
    }
}

/// Cache group family.
#[pyclass(eq, eq_int, name = "CacheGroupFamily")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyCacheGroupFamily {
    History,
    State,
}

impl From<PyCacheGroupFamily> for CacheGroupFamily {
    fn from(f: PyCacheGroupFamily) -> Self {
        match f {
            PyCacheGroupFamily::History => CacheGroupFamily::History,
            PyCacheGroupFamily::State => CacheGroupFamily::State,
        }
    }
}

impl From<CacheGroupFamily> for PyCacheGroupFamily {
    fn from(f: CacheGroupFamily) -> Self {
        match f {
            CacheGroupFamily::History => PyCacheGroupFamily::History,
            CacheGroupFamily::State => PyCacheGroupFamily::State,
        }
    }
}

/// Tier transfer policy.
#[pyclass(eq, eq_int, name = "CacheTransferPolicy")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyCacheTransferPolicy {
    Unspecified,
    FullSuffix,
    LatestSnapshot,
}

impl From<PyCacheTransferPolicy> for CacheTransferPolicy {
    fn from(p: PyCacheTransferPolicy) -> Self {
        match p {
            PyCacheTransferPolicy::Unspecified => CacheTransferPolicy::Unspecified,
            PyCacheTransferPolicy::FullSuffix => CacheTransferPolicy::FullSuffix,
            PyCacheTransferPolicy::LatestSnapshot => CacheTransferPolicy::LatestSnapshot,
        }
    }
}

impl From<CacheTransferPolicy> for PyCacheTransferPolicy {
    fn from(p: CacheTransferPolicy) -> Self {
        match p {
            CacheTransferPolicy::Unspecified => PyCacheTransferPolicy::Unspecified,
            CacheTransferPolicy::FullSuffix => PyCacheTransferPolicy::FullSuffix,
            CacheTransferPolicy::LatestSnapshot => PyCacheTransferPolicy::LatestSnapshot,
        }
    }
}

// ─── SchedulerConfig ────────────────────────────────────────────────────────

/// Writable scheduler configuration.
#[pyclass(name = "SchedulerConfig", module = "tokenspeed_scheduler", unsendable)]
pub struct PySchedulerConfig {
    pub inner: SchedulerConfig,
    /// Shared handles behind `cache_groups`, so in-place Python mutation of a
    /// returned group writes through to the stored config.
    cache_groups: Vec<std::rc::Rc<std::cell::RefCell<CacheGroupConfig>>>,
}

impl PySchedulerConfig {
    /// Flush the shared group handles into `inner.cache_groups` (called right
    /// before constructing a scheduler).
    fn sync(&mut self) {
        self.inner.cache_groups = self
            .cache_groups
            .iter()
            .map(|rc| rc.borrow().clone())
            .collect();
    }
}

#[pymethods]
impl PySchedulerConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: SchedulerConfig::default(),
            cache_groups: Vec::new(),
        }
    }

    #[getter]
    fn prefix_granularity(&self) -> i32 {
        self.inner.prefix_granularity
    }
    #[setter]
    fn set_prefix_granularity(&mut self, v: i32) {
        self.inner.prefix_granularity = v;
    }

    #[getter]
    fn max_scheduled_tokens(&self) -> i32 {
        self.inner.max_scheduled_tokens
    }
    #[setter]
    fn set_max_scheduled_tokens(&mut self, v: i32) {
        self.inner.max_scheduled_tokens = v;
    }

    #[getter]
    fn max_batch_size(&self) -> i32 {
        self.inner.max_batch_size
    }
    #[setter]
    fn set_max_batch_size(&mut self, v: i32) {
        self.inner.max_batch_size = v;
    }

    #[getter]
    fn decode_input_tokens(&self) -> i32 {
        self.inner.decode_input_tokens
    }
    #[setter]
    fn set_decode_input_tokens(&mut self, v: i32) {
        self.inner.decode_input_tokens = v;
    }

    #[getter]
    fn overlap_schedule_depth(&self) -> i32 {
        self.inner.overlap_schedule_depth
    }
    #[setter]
    fn set_overlap_schedule_depth(&mut self, v: i32) {
        self.inner.overlap_schedule_depth = v;
    }

    #[getter]
    fn role(&self) -> PyRole {
        self.inner.role.into()
    }
    #[setter]
    fn set_role(&mut self, v: PyRole) {
        self.inner.role = v.into();
    }

    #[getter]
    fn enable_pd_cache(&self) -> bool {
        self.inner.enable_pd_cache
    }
    #[setter]
    fn set_enable_pd_cache(&mut self, v: bool) {
        self.inner.enable_pd_cache = v;
    }

    #[getter]
    fn num_device_pages(&self) -> i32 {
        self.inner.device_allocator.total_pages
    }
    #[setter]
    fn set_num_device_pages(&mut self, v: i32) {
        self.inner.device_allocator.total_pages = v;
    }

    #[getter]
    fn num_host_pages(&self) -> i32 {
        self.inner.host_allocator.total_pages
    }
    #[setter]
    fn set_num_host_pages(&mut self, v: i32) {
        self.inner.host_allocator.total_pages = v;
    }

    #[getter]
    fn cache_groups<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let items: Vec<PyCacheGroupConfig> = self
            .cache_groups
            .iter()
            .map(|rc| PyCacheGroupConfig { inner: rc.clone() })
            .collect();
        let objs: Vec<Bound<'_, PyAny>> = items
            .into_iter()
            .map(|g| {
                Py::new(py, g)
                    .expect("cache group")
                    .into_bound(py)
                    .into_any()
            })
            .collect();
        PyList::new_bound(py, objs)
    }
    #[setter]
    fn set_cache_groups(&mut self, v: Vec<PyRef<'_, PyCacheGroupConfig>>) {
        self.cache_groups = v.iter().map(|g| g.inner.clone()).collect();
        self.sync();
    }

    #[getter]
    fn disable_l2_cache(&self) -> bool {
        self.inner.disable_l2_cache
    }
    #[setter]
    fn set_disable_l2_cache(&mut self, v: bool) {
        self.inner.disable_l2_cache = v;
    }

    #[getter]
    fn enable_l3_storage(&self) -> bool {
        self.inner.enable_l3_storage
    }
    #[setter]
    fn set_enable_l3_storage(&mut self, v: bool) {
        self.inner.enable_l3_storage = v;
    }

    #[getter]
    fn enable_kv_cache_events(&self) -> bool {
        self.inner.enable_kv_cache_events
    }
    #[setter]
    fn set_enable_kv_cache_events(&mut self, v: bool) {
        self.inner.enable_kv_cache_events = v;
    }

    #[getter]
    fn enable_mixed_prefill_decode(&self) -> bool {
        self.inner.enable_mixed_prefill_decode
    }
    #[setter]
    fn set_enable_mixed_prefill_decode(&mut self, v: bool) {
        self.inner.enable_mixed_prefill_decode = v;
    }

    #[getter]
    fn disable_prefix_cache(&self) -> bool {
        self.inner.disable_prefix_cache
    }
    #[setter]
    fn set_disable_prefix_cache(&mut self, v: bool) {
        self.inner.disable_prefix_cache = v;
    }

    #[getter]
    fn prefix_replay_tokens(&self) -> i32 {
        self.inner.prefix_replay_tokens
    }
    #[setter]
    fn set_prefix_replay_tokens(&mut self, v: i32) {
        self.inner.prefix_replay_tokens = v;
    }
}
// ─── CacheGroupConfig ───────────────────────────────────────────────────────

/// Writable cache-group configuration. Backed by `Rc<RefCell<..>>` so that
/// `cfg.cache_groups[i].field = v` mutations made from Python write through to
/// the config actually used to construct the scheduler (the C++ nanobind
/// getter returned a reference into the config's vector).
#[pyclass(name = "CacheGroupConfig", module = "tokenspeed_scheduler", unsendable)]
#[derive(Clone)]
pub struct PyCacheGroupConfig {
    pub inner: std::rc::Rc<std::cell::RefCell<CacheGroupConfig>>,
}

impl From<CacheGroupConfig> for PyCacheGroupConfig {
    fn from(inner: CacheGroupConfig) -> Self {
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(inner)),
        }
    }
}

#[pymethods]
impl PyCacheGroupConfig {
    #[new]
    #[pyo3(signature = (group_id, rows_per_page, entry_stride_tokens, total_pages,
        retention = PyCacheRetention::FullHistory, sliding_window_tokens = None,
        family = PyCacheGroupFamily::History, cache_blocks_per_lcm_block = 1,
        transfer_policy = PyCacheTransferPolicy::Unspecified))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        group_id: String,
        rows_per_page: i32,
        entry_stride_tokens: i32,
        total_pages: i32,
        retention: PyCacheRetention,
        sliding_window_tokens: Option<i32>,
        family: PyCacheGroupFamily,
        cache_blocks_per_lcm_block: i32,
        transfer_policy: PyCacheTransferPolicy,
    ) -> Self {
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(CacheGroupConfig {
                group_id,
                rows_per_page,
                entry_stride_tokens,
                total_pages,
                cache_blocks_per_lcm_block,
                retention: retention.into(),
                sliding_window_tokens,
                family: family.into(),
                transfer_policy: transfer_policy.into(),
            })),
        }
    }

    #[getter]
    fn group_id(&self) -> String {
        self.inner.borrow().group_id.clone()
    }
    #[setter]
    fn set_group_id(&mut self, v: String) {
        self.inner.borrow_mut().group_id = v;
    }

    #[getter]
    fn rows_per_page(&self) -> i32 {
        self.inner.borrow().rows_per_page
    }
    #[setter]
    fn set_rows_per_page(&mut self, v: i32) {
        self.inner.borrow_mut().rows_per_page = v;
    }

    #[getter]
    fn entry_stride_tokens(&self) -> i32 {
        self.inner.borrow().entry_stride_tokens
    }
    #[setter]
    fn set_entry_stride_tokens(&mut self, v: i32) {
        self.inner.borrow_mut().entry_stride_tokens = v;
    }

    #[getter]
    fn total_pages(&self) -> i32 {
        self.inner.borrow().total_pages
    }
    #[setter]
    fn set_total_pages(&mut self, v: i32) {
        self.inner.borrow_mut().total_pages = v;
    }

    #[getter]
    fn cache_blocks_per_lcm_block(&self) -> i32 {
        self.inner.borrow().cache_blocks_per_lcm_block
    }
    #[setter]
    fn set_cache_blocks_per_lcm_block(&mut self, v: i32) {
        self.inner.borrow_mut().cache_blocks_per_lcm_block = v;
    }

    #[getter]
    fn retention(&self) -> PyCacheRetention {
        self.inner.borrow().retention.into()
    }
    #[setter]
    fn set_retention(&mut self, v: PyCacheRetention) {
        self.inner.borrow_mut().retention = v.into();
    }

    #[getter]
    fn sliding_window_tokens(&self) -> Option<i32> {
        self.inner.borrow().sliding_window_tokens
    }
    #[setter]
    fn set_sliding_window_tokens(&mut self, v: Option<i32>) {
        self.inner.borrow_mut().sliding_window_tokens = v;
    }

    #[getter]
    fn family(&self) -> PyCacheGroupFamily {
        self.inner.borrow().family.into()
    }
    #[setter]
    fn set_family(&mut self, v: PyCacheGroupFamily) {
        self.inner.borrow_mut().family = v.into();
    }

    #[getter]
    fn transfer_policy(&self) -> PyCacheTransferPolicy {
        self.inner.borrow().transfer_policy.into()
    }
    #[setter]
    fn set_transfer_policy(&mut self, v: PyCacheTransferPolicy) {
        self.inner.borrow_mut().transfer_policy = v.into();
    }

    /// Validate; raises `ValueError` with the C++ message on the first violation.
    fn validate(&self) -> PyResult<()> {
        self.inner
            .borrow()
            .validate()
            .map_err(PyValueError::new_err)
    }
}

// ─── RequestSpec ────────────────────────────────────────────────────────────

/// Inbound request specification.
#[pyclass(name = "RequestSpec", module = "tokenspeed_scheduler")]
#[derive(Clone)]
pub struct PyRequestSpec {
    pub inner: RequestSpec,
}

impl From<RequestSpec> for PyRequestSpec {
    fn from(inner: RequestSpec) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRequestSpec {
    #[new]
    fn new() -> Self {
        Self {
            inner: RequestSpec::default(),
        }
    }

    #[getter]
    fn request_id(&self) -> String {
        self.inner.request_id.clone()
    }
    #[setter]
    fn set_request_id(&mut self, v: String) {
        self.inner.request_id = v;
    }

    #[getter]
    fn tokens(&self) -> Vec<i32> {
        self.inner.tokens.clone()
    }
    #[setter]
    fn set_tokens(&mut self, v: Vec<i32>) {
        self.inner.tokens = v;
    }

    #[getter]
    fn max_new_tokens(&self) -> i32 {
        self.inner.max_new_tokens
    }
    #[setter]
    fn set_max_new_tokens(&mut self, v: i32) {
        self.inner.max_new_tokens = v;
    }
}

// ─── ForwardEvent classes ───────────────────────────────────────────────────

/// `ForwardEvent.ExtendResult`
#[pyclass(name = "ExtendResult", module = "tokenspeed_scheduler.ForwardEvent")]
#[derive(Clone)]
pub struct PyExtendResult {
    pub request_id: String,
    pub tokens: Vec<i32>,
}

/// `ForwardEvent.Finish`
#[pyclass(name = "Finish", module = "tokenspeed_scheduler.ForwardEvent")]
#[derive(Clone)]
pub struct PyFinish {
    pub request_id: String,
}

/// `ForwardEvent.Abort`
#[pyclass(name = "Abort", module = "tokenspeed_scheduler.ForwardEvent")]
#[derive(Clone)]
pub struct PyAbort {
    pub request_id: String,
}

/// `ForwardEvent.UpdateReserveNumTokens`
#[pyclass(
    name = "UpdateReserveNumTokens",
    module = "tokenspeed_scheduler.ForwardEvent"
)]
#[derive(Clone)]
pub struct PyUpdateReserveNumTokens {
    pub request_id: String,
    pub reserve_num_tokens_in_next_schedule_event: i32,
}

// ─── Cache / PD / KV event classes ──────────────────────────────────────────

/// `Cache.WriteBackDoneEvent`
#[pyclass(name = "WriteBackDoneEvent", module = "tokenspeed_scheduler.Cache")]
#[derive(Clone)]
pub struct PyWriteBackDone {
    pub op_id: u32,
}

/// `Cache.LoadBackDoneEvent`
#[pyclass(name = "LoadBackDoneEvent", module = "tokenspeed_scheduler.Cache")]
#[derive(Clone)]
pub struct PyLoadBackDone {
    pub op_id: u32,
}

/// `PD.BootstrappedEvent`
#[pyclass(name = "BootstrappedEvent", module = "tokenspeed_scheduler.PD")]
#[derive(Clone)]
pub struct PyBootstrappedEvent {
    pub request_id: String,
}

/// `PD.FailedEvent`
#[pyclass(name = "FailedEvent", module = "tokenspeed_scheduler.PD")]
#[derive(Clone)]
pub struct PyFailedEvent {
    pub request_id: String,
}

/// `PD.SucceededEvent`
#[pyclass(name = "SucceededEvent", module = "tokenspeed_scheduler.PD")]
#[derive(Clone)]
pub struct PySucceededEvent {
    pub request_id: String,
}

/// `PD.RemotePrefillDoneEvent`
#[pyclass(name = "RemotePrefillDoneEvent", module = "tokenspeed_scheduler.PD")]
#[derive(Clone)]
pub struct PyRemotePrefillDoneEvent {
    pub request_id: String,
    pub bootstrap_token: i32,
}

/// `KVEvent.BlockStored`
#[pyclass(name = "BlockStored", module = "tokenspeed_scheduler.KVEvent")]
#[derive(Clone)]
pub struct PyBlockStored {
    pub block_hashes: Vec<u64>,
    pub parent_block_hash: Option<u64>,
    pub token_ids: Vec<i32>,
    pub block_size: i32,
}

/// `KVEvent.BlockRemoved`
#[pyclass(name = "BlockRemoved", module = "tokenspeed_scheduler.KVEvent")]
#[derive(Clone)]
pub struct PyBlockRemoved {
    pub block_hashes: Vec<u64>,
}

// ─── ExecutionEvent ─────────────────────────────────────────────────────────

/// Collects outside events for one `Scheduler.advance` call.
#[pyclass(name = "ExecutionEvent", module = "tokenspeed_scheduler")]
#[derive(Clone, Default)]
pub struct PyExecutionEvent {
    pub inner: Vec<Event>,
}

#[pymethods]
impl PyExecutionEvent {
    #[new]
    fn new() -> Self {
        Self { inner: Vec::new() }
    }

    fn add_event<'py>(
        mut slff: PyRefMut<'py, Self>,
        event: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Ok(ev) = event.extract::<PyRef<'_, PyExtendResult>>() {
            slff.inner
                .push(Event::ExtendResult(events::forward::ExtendResult {
                    request_id: ev.request_id.clone(),
                    tokens: ev.tokens.clone(),
                }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyFinish>>() {
            slff.inner.push(Event::Finish(events::forward::Finish {
                request_id: ev.request_id.clone(),
            }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyAbort>>() {
            slff.inner.push(Event::Abort(events::forward::Abort {
                request_id: ev.request_id.clone(),
            }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyUpdateReserveNumTokens>>() {
            slff.inner.push(Event::UpdateReserveNumTokens(
                events::forward::UpdateReserveNumTokens {
                    request_id: ev.request_id.clone(),
                    reserve_num_tokens_in_next_schedule_event: ev
                        .reserve_num_tokens_in_next_schedule_event,
                },
            ));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyWriteBackDone>>() {
            slff.inner
                .push(Event::WriteBackDone(events::cache::WriteBackDone {
                    op_id: ev.op_id,
                }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyLoadBackDone>>() {
            slff.inner
                .push(Event::LoadBackDone(events::cache::LoadBackDone {
                    op_id: ev.op_id,
                }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyBootstrappedEvent>>() {
            slff.inner
                .push(Event::Bootstrapped(events::pd::BootstrappedEvent {
                    request_id: ev.request_id.clone(),
                }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyFailedEvent>>() {
            slff.inner.push(Event::Failed(events::pd::FailedEvent {
                request_id: ev.request_id.clone(),
            }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PySucceededEvent>>() {
            slff.inner
                .push(Event::Succeeded(events::pd::SucceededEvent {
                    request_id: ev.request_id.clone(),
                }));
            return Ok(slff);
        }
        if let Ok(ev) = event.extract::<PyRef<'_, PyRemotePrefillDoneEvent>>() {
            slff.inner.push(Event::RemotePrefillDone(
                events::pd::RemotePrefillDoneEvent {
                    request_id: ev.request_id.clone(),
                    bootstrap_token: ev.bootstrap_token,
                },
            ));
            return Ok(slff);
        }
        Err(PyTypeError::new_err("add_event: unsupported event type"))
    }
}
// ─── ForwardEvent / Cache / PD / KV constructors and properties ────────────

#[pymethods]
impl PyExtendResult {
    #[new]
    fn new() -> Self {
        Self {
            request_id: String::new(),
            tokens: Vec::new(),
        }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
    #[setter]
    fn set_request_id(&mut self, v: String) {
        self.request_id = v;
    }
    #[getter]
    fn tokens(&self) -> Vec<i32> {
        self.tokens.clone()
    }
    #[setter]
    fn set_tokens(&mut self, v: Vec<i32>) {
        self.tokens = v;
    }
}

#[pymethods]
impl PyFinish {
    #[new]
    fn new() -> Self {
        Self {
            request_id: String::new(),
        }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
    #[setter]
    fn set_request_id(&mut self, v: String) {
        self.request_id = v;
    }
}

#[pymethods]
impl PyAbort {
    #[new]
    fn new() -> Self {
        Self {
            request_id: String::new(),
        }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
    #[setter]
    fn set_request_id(&mut self, v: String) {
        self.request_id = v;
    }
}

#[pymethods]
impl PyUpdateReserveNumTokens {
    #[new]
    fn new() -> Self {
        Self {
            request_id: String::new(),
            reserve_num_tokens_in_next_schedule_event: 0,
        }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
    #[setter]
    fn set_request_id(&mut self, v: String) {
        self.request_id = v;
    }
    #[getter]
    fn reserve_num_tokens_in_next_schedule_event(&self) -> i32 {
        self.reserve_num_tokens_in_next_schedule_event
    }
    #[setter]
    fn set_reserve_num_tokens_in_next_schedule_event(&mut self, v: i32) {
        self.reserve_num_tokens_in_next_schedule_event = v;
    }
}

#[pymethods]
impl PyWriteBackDone {
    #[new]
    fn new() -> Self {
        Self { op_id: 0 }
    }
    #[getter]
    fn op_id(&self) -> u32 {
        self.op_id
    }
    #[setter]
    fn set_op_id(&mut self, v: u32) {
        self.op_id = v;
    }
}

#[pymethods]
impl PyLoadBackDone {
    #[new]
    fn new() -> Self {
        Self { op_id: 0 }
    }
    #[getter]
    fn op_id(&self) -> u32 {
        self.op_id
    }
    #[setter]
    fn set_op_id(&mut self, v: u32) {
        self.op_id = v;
    }
}

#[pymethods]
impl PyBootstrappedEvent {
    #[new]
    fn new(request_id: String) -> Self {
        Self { request_id }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
}

#[pymethods]
impl PyFailedEvent {
    #[new]
    fn new(request_id: String) -> Self {
        Self { request_id }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
}

#[pymethods]
impl PySucceededEvent {
    #[new]
    fn new(request_id: String) -> Self {
        Self { request_id }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
}

#[pymethods]
impl PyRemotePrefillDoneEvent {
    #[new]
    fn new(request_id: String, bootstrap_token: i32) -> Self {
        Self {
            request_id,
            bootstrap_token,
        }
    }
    #[getter]
    fn request_id(&self) -> String {
        self.request_id.clone()
    }
    #[getter]
    fn bootstrap_token(&self) -> i32 {
        self.bootstrap_token
    }
    #[setter]
    fn set_bootstrap_token(&mut self, v: i32) {
        self.bootstrap_token = v;
    }
}

#[pymethods]
impl PyBlockStored {
    #[getter]
    fn kind(&self) -> &'static str {
        "BlockStored"
    }
    #[getter]
    fn block_hashes(&self) -> Vec<u64> {
        self.block_hashes.clone()
    }
    #[getter]
    fn parent_block_hash(&self) -> Option<u64> {
        self.parent_block_hash
    }
    #[getter]
    fn token_ids(&self) -> Vec<i32> {
        self.token_ids.clone()
    }
    #[getter]
    fn block_size(&self) -> i32 {
        self.block_size
    }
}

#[pymethods]
impl PyBlockRemoved {
    #[getter]
    fn kind(&self) -> &'static str {
        "BlockRemoved"
    }
    #[getter]
    fn block_hashes(&self) -> Vec<u64> {
        self.block_hashes.clone()
    }
}

// ─── Forward.Batch ──────────────────────────────────────────────────────────

/// One forward batch produced by `Scheduler.next_execution_plan`.
#[pyclass(name = "Batch", module = "tokenspeed_scheduler.Forward")]
#[derive(Clone)]
pub struct PyForwardBatch {
    pub inner: ForwardBatch,
}

#[pymethods]
impl PyForwardBatch {
    #[getter]
    fn request_ids(&self) -> Vec<String> {
        self.inner.request_ids.clone()
    }
    #[getter]
    fn request_pool_indices(&self) -> Vec<i32> {
        self.inner.request_pool_indices.clone()
    }
    #[getter]
    fn input_lengths(&self) -> Vec<i32> {
        self.inner.input_lengths.clone()
    }
    #[getter]
    fn prefill_lengths(&self) -> Vec<i32> {
        self.inner.prefill_lengths.clone()
    }
    #[getter]
    fn input_ids(&self) -> Vec<i32> {
        self.inner.input_ids.clone()
    }
    #[getter]
    fn shifted_input_ids(&self) -> Vec<i32> {
        self.inner.shifted_input_ids.clone()
    }
    #[getter]
    fn extend_prefix_lens(&self) -> Vec<i32> {
        self.inner.extend_prefix_lens.clone()
    }
    #[getter]
    fn decode_input_ids(&self) -> Vec<i32> {
        self.inner.decode_input_ids.clone()
    }
    #[getter]
    fn block_tables(&self) -> BTreeMap<String, Vec<Vec<i32>>> {
        self.inner.block_tables.clone()
    }
    fn is_local_prefill(&self) -> bool {
        self.inner.is_local_prefill()
    }
    fn num_extends(&self) -> usize {
        self.inner.num_extends()
    }
    /// Zero-copy 2-D int32 views per group (port returns copies; the C++
    /// nanobind version borrowed the backing buffer).
    fn block_tables_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new_bound(py);
        for (gid, buf) in &self.inner.block_tables_contig {
            let rows = self.inner.request_ids.len();
            let columns = buf.len().checked_div(rows).unwrap_or(0);
            let chunks: Vec<Vec<i32>> = buf.chunks(columns.max(1)).map(|c| c.to_vec()).collect();
            let arr = PyArray2::from_vec2_bound(py, &chunks)
                .map_err(|e| PyValueError::new_err(format!("block table array: {e:?}")))?;
            out.set_item(gid.as_str(), arr)?;
        }
        Ok(out)
    }
}

// ─── Cache operations ───────────────────────────────────────────────────────

/// `Cache.LoadBackOp`
#[pyclass(name = "LoadBackOp", module = "tokenspeed_scheduler.Cache")]
#[derive(Clone)]
pub struct PyLoadBackBatch {
    pub inner: LoadBackBatch,
}

#[pymethods]
impl PyLoadBackBatch {
    #[getter]
    fn op_ids(&self) -> Vec<u32> {
        self.inner.op_ids.clone()
    }
    #[getter]
    fn group_ids(&self) -> Vec<Vec<u32>> {
        self.inner.group_ids.clone()
    }
    #[getter]
    fn src_pages(&self) -> Vec<Vec<i32>> {
        self.inner.src_pages.clone()
    }
    #[getter]
    fn dst_pages(&self) -> Vec<Vec<i32>> {
        self.inner.dst_pages.clone()
    }
}

/// `Cache.WriteBackOp`
#[pyclass(name = "WriteBackOp", module = "tokenspeed_scheduler.Cache")]
#[derive(Clone)]
pub struct PyWriteBackBatch {
    pub inner: WriteBackBatch,
}

#[pymethods]
impl PyWriteBackBatch {
    #[getter]
    fn op_ids(&self) -> Vec<u32> {
        self.inner.op_ids.clone()
    }
    #[getter]
    fn group_ids(&self) -> Vec<Vec<u32>> {
        self.inner.group_ids.clone()
    }
    #[getter]
    fn src_pages(&self) -> Vec<Vec<i32>> {
        self.inner.src_pages.clone()
    }
    #[getter]
    fn dst_pages(&self) -> Vec<Vec<i32>> {
        self.inner.dst_pages.clone()
    }
}

// ─── ExecutionPlan ──────────────────────────────────────────────────────────

/// The plan produced by `Scheduler.next_execution_plan`.
#[pyclass(name = "ExecutionPlan", module = "tokenspeed_scheduler")]
pub struct PyExecutionPlan {
    pub inner: ExecutionPlan,
}

#[pymethods]
impl PyExecutionPlan {
    #[new]
    fn new() -> Self {
        Self {
            inner: ExecutionPlan::new(),
        }
    }
    #[getter]
    fn forward<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let items: Vec<PyForwardBatch> = self
            .inner
            .operations()
            .iter()
            .filter_map(|op| match op {
                Operation::Forward(f) => Some(PyForwardBatch { inner: f.clone() }),
                _ => None,
            })
            .collect();
        let objs: Vec<Bound<'_, PyAny>> = items
            .into_iter()
            .map(|f| {
                Py::new(py, f)
                    .expect("forward batch")
                    .into_bound(py)
                    .into_any()
            })
            .collect();
        PyList::new_bound(py, objs)
    }
    #[getter]
    fn cache<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
        for op in self.inner.operations() {
            if let Operation::Cache(c) = op {
                match c {
                    CacheOperation::LoadBack(b) => {
                        items.push(
                            Py::new(py, PyLoadBackBatch { inner: b.clone() })?
                                .into_bound(py)
                                .into_any(),
                        );
                    }
                    CacheOperation::WriteBack(b) => {
                        items.push(
                            Py::new(py, PyWriteBackBatch { inner: b.clone() })?
                                .into_bound(py)
                                .into_any(),
                        );
                    }
                }
            }
        }
        Ok(PyList::new_bound(py, items))
    }
    #[getter]
    fn pages_to_zero(&self) -> BTreeMap<String, Vec<i32>> {
        self.inner.pages_to_zero.clone()
    }
}

// ─── Scheduler ──────────────────────────────────────────────────────────────

/// The scheduler control plane (unsendable: tied to the creating thread, like
/// the single-threaded C++ scheduler).
#[pyclass(name = "Scheduler", module = "tokenspeed_scheduler", unsendable)]
pub struct PyScheduler {
    pub inner: Scheduler,
}

/// Convert a panic payload into a `ValueError` message.
fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "Scheduler: internal error".to_string()
    }
}

#[pymethods]
impl PyScheduler {
    #[new]
    fn new(mut config: PyRefMut<'_, PySchedulerConfig>) -> PyResult<Self> {
        config.sync();
        config.inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Scheduler::new(config.inner.clone()),
        })
    }

    fn submit_requests(&mut self, request_specs: Vec<PyRef<'_, PyRequestSpec>>) -> PyResult<()> {
        let specs: Vec<RequestSpec> = request_specs.iter().map(|s| s.inner.clone()).collect();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.submit_requests(&specs);
        }))
        .map_err(|e| PyValueError::new_err(panic_message(e)))
    }

    fn next_execution_plan(&mut self) -> PyExecutionPlan {
        PyExecutionPlan {
            inner: self.inner.next_execution_plan(),
        }
    }

    fn advance(&mut self, event: PyRef<'_, PyExecutionEvent>) -> PyResult<()> {
        let mut core = ExecutionEvent::new();
        for e in event.inner.clone() {
            core.with(e);
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.advance(&core);
        }))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(panic_message(e)))
    }

    fn drain_kv_events<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let events = self.inner.drain_kv_events();
        let mut items: Vec<Bound<'py, PyAny>> = Vec::new();
        for event in events {
            match event {
                KvCacheEvent::BlockStored(e) => {
                    items.push(
                        Py::new(
                            py,
                            PyBlockStored {
                                block_hashes: e.block_hashes,
                                parent_block_hash: e.parent_block_hash,
                                token_ids: e.token_ids,
                                block_size: e.block_size,
                            },
                        )?
                        .into_bound(py)
                        .into_any(),
                    );
                }
                KvCacheEvent::BlockRemoved(e) => {
                    items.push(
                        Py::new(
                            py,
                            PyBlockRemoved {
                                block_hashes: e.block_hashes,
                            },
                        )?
                        .into_bound(py)
                        .into_any(),
                    );
                }
            }
        }
        Ok(PyList::new_bound(py, items))
    }

    fn waiting_size(&self) -> usize {
        self.inner.waiting_size()
    }
    fn decoding_size(&self) -> usize {
        self.inner.decoding_size()
    }
    fn prefilling_size(&self) -> usize {
        self.inner.prefill_size()
    }
    fn pd_transfer_pinned(&self, request_id: String) -> bool {
        self.inner.pd_transfer_pinned(&request_id)
    }
    fn available_kv_pages(&self) -> usize {
        self.inner.available_kv_pages()
    }
    fn active_kv_pages(&self) -> usize {
        self.inner.active_kv_pages()
    }
    fn request_token_size(&self, id: String) -> i32 {
        self.inner.request_token_size(&id)
    }
    fn max_single_request_tokens(&self) -> i32 {
        self.inner.max_single_request_tokens()
    }
    fn clear_l1_cache(&mut self) -> bool {
        self.inner.clear_l1_cache()
    }
    fn clear_cache(&mut self) -> bool {
        self.inner.clear_cache()
    }
    fn cache_group_total_pages(&self, group_id: String) -> i32 {
        self.inner.cache_group_total_pages(&group_id)
    }
    fn cache_group_available_pages(&self, group_id: String) -> i32 {
        self.inner.cache_group_available_pages(&group_id)
    }
}

// ─── Module ─────────────────────────────────────────────────────────────────

#[pymodule]
fn tokenspeed_scheduler_ext(m: &Bound<'_, PyModuleType>) -> PyResult<()> {
    m.add_class::<PySchedulerConfig>()?;
    m.add_class::<PyRole>()?;
    m.add_class::<PyCacheRetention>()?;
    m.add_class::<PyCacheGroupFamily>()?;
    m.add_class::<PyCacheTransferPolicy>()?;
    m.add_class::<PyCacheGroupConfig>()?;
    m.add_class::<PyRequestSpec>()?;
    m.add_class::<PyScheduler>()?;
    m.add_class::<PyExecutionEvent>()?;
    m.add_class::<PyExecutionPlan>()?;

    // Expose Role as a nested enum of SchedulerConfig (C++ nanobind nests it).
    let role = m.getattr("Role")?;
    let config_type = m.getattr("SchedulerConfig")?;
    config_type.setattr("Role", &role)?;

    let forward_event = PyModuleType::new_bound(m.py(), "tokenspeed_scheduler.ForwardEvent")?;
    forward_event.add_class::<PyExtendResult>()?;
    forward_event.add_class::<PyFinish>()?;
    forward_event.add_class::<PyAbort>()?;
    forward_event.add_class::<PyUpdateReserveNumTokens>()?;
    m.add("ForwardEvent", forward_event)?;

    let forward = PyModuleType::new_bound(m.py(), "tokenspeed_scheduler.Forward")?;
    forward.add_class::<PyForwardBatch>()?;
    m.add("Forward", forward)?;

    let cache = PyModuleType::new_bound(m.py(), "tokenspeed_scheduler.Cache")?;
    cache.add_class::<PyWriteBackDone>()?;
    cache.add_class::<PyLoadBackDone>()?;
    cache.add_class::<PyLoadBackBatch>()?;
    cache.add_class::<PyWriteBackBatch>()?;
    m.add("Cache", cache)?;

    let pd = PyModuleType::new_bound(m.py(), "tokenspeed_scheduler.PD")?;
    pd.add_class::<PyBootstrappedEvent>()?;
    pd.add_class::<PyFailedEvent>()?;
    pd.add_class::<PySucceededEvent>()?;
    pd.add_class::<PyRemotePrefillDoneEvent>()?;
    m.add("PD", pd)?;

    let kv_event = PyModuleType::new_bound(m.py(), "tokenspeed_scheduler.KVEvent")?;
    kv_event.add_class::<PyBlockStored>()?;
    kv_event.add_class::<PyBlockRemoved>()?;
    m.add("KVEvent", kv_event)?;
    Ok(())
}
