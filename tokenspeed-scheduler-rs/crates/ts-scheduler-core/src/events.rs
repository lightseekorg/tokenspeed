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

//! Execution events, plans, and forward operations.
//!
//! Ported from `tokenspeed-scheduler/csrc/scheduler/execution_event.h`,
//! `execution_plan.h`, `operations/forward.h`, and `outside_events/*.h`.

use std::collections::BTreeMap;

use crate::tier::CacheOperation;

/// Events produced by the Python execution plane.
pub mod forward {
    /// A forward step produced these tokens; their KV is now stable history.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ExtendResult {
        pub request_id: String,
        pub tokens: Vec<i32>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Finish {
        pub request_id: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct UpdateReserveNumTokens {
        pub request_id: String,
        pub reserve_num_tokens_in_next_schedule_event: i32,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Abort {
        pub request_id: String,
    }
}

/// Events produced by the tier-transfer runtime.
pub mod cache {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct WriteBackDone {
        pub op_id: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct LoadBackDone {
        pub op_id: u32,
    }
}

/// Events produced by the PD runtime.
pub mod pd {
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BootstrappedEvent {
        pub request_id: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct FailedEvent {
        pub request_id: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct SucceededEvent {
        pub request_id: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RemotePrefillDoneEvent {
        pub request_id: String,
        pub bootstrap_token: i32,
    }
}

/// Any outside event delivered to `Scheduler::advance`.
#[derive(Debug, Clone)]
pub enum Event {
    WriteBackDone(cache::WriteBackDone),
    LoadBackDone(cache::LoadBackDone),
    ExtendResult(forward::ExtendResult),
    Finish(forward::Finish),
    Abort(forward::Abort),
    UpdateReserveNumTokens(forward::UpdateReserveNumTokens),
    Bootstrapped(pd::BootstrappedEvent),
    Failed(pd::FailedEvent),
    Succeeded(pd::SucceededEvent),
    RemotePrefillDone(pd::RemotePrefillDoneEvent),
}

/// A batch of outside events delivered in one `advance` call.
#[derive(Debug, Default)]
pub struct ExecutionEvent {
    events: Vec<Event>,
}

impl ExecutionEvent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(&mut self, event: Event) -> &mut Self {
        self.events.push(event);
        self
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }
}

/// Common fields of a forward operation.
#[derive(Debug, Clone, Default)]
pub struct ForwardOperationBase {
    pub request_id: String,
    pub request_pool_index: i32,
    pub input_length: i32,
    pub prefill_length: i32,
    /// Per-group block tables; rows use absolute logical-page indexing; null
    /// holes are empty refs and rows are not compacted.
    pub block_tables: BTreeMap<String, Vec<i32>>,
}

#[derive(Debug, Clone, Default)]
pub struct PrefillOperation {
    pub base: ForwardOperationBase,
    pub input_ids: Vec<i32>,
    pub shifted_input_ids: Vec<i32>,
    pub extend_prefix_len: i32,
    pub local_prefill: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DecodeOperation {
    pub base: ForwardOperationBase,
    pub decode_input_id: i32,
}

/// One request's forward work: prefill or decode.
#[derive(Debug, Clone)]
pub enum ForwardOperation {
    Prefill(PrefillOperation),
    Decode(DecodeOperation),
}

/// A batch of forward operations produced by one `NextExecutionPlan`.
#[derive(Debug, Clone)]
pub struct ForwardBatch {
    pub request_ids: Vec<String>,
    pub request_pool_indices: Vec<i32>,
    pub input_lengths: Vec<i32>,
    /// Per-request total number of prompt tokens.
    pub prefill_lengths: Vec<i32>,
    pub input_ids: Vec<i32>,
    pub shifted_input_ids: Vec<i32>,
    pub extend_prefix_lens: Vec<i32>,
    pub decode_input_ids: Vec<i32>,
    /// Per-group block tables: group_id -> [num_reqs, max_pages_in_batch]
    /// padded with empty rows; each row is absolute (null hole = empty).
    pub block_tables: BTreeMap<String, Vec<Vec<i32>>>,
    /// Contiguous row-major copy of `block_tables` (`[rows * cols]`, -1
    /// padded), exposed zero-copy to Python as a 2-D ndarray.
    pub block_tables_contig: BTreeMap<String, Vec<i32>>,
    pub local_prefill: bool,
}

impl ForwardBatch {
    pub fn new(ops: Vec<ForwardOperation>) -> Self {
        // Stable partition: prefills first, then decodes (C++ std::stable_partition).
        let mut ops = ops;
        let _split = partition_prefills_first(&mut ops);
        let mut batch = ForwardBatch {
            request_ids: Vec::new(),
            request_pool_indices: Vec::new(),
            input_lengths: Vec::new(),
            prefill_lengths: Vec::new(),
            input_ids: Vec::new(),
            shifted_input_ids: Vec::new(),
            extend_prefix_lens: Vec::new(),
            decode_input_ids: Vec::new(),
            block_tables: BTreeMap::new(),
            block_tables_contig: BTreeMap::new(),
            local_prefill: false,
        };
        let mut prefill_source: Option<bool> = None;
        for op in ops.iter() {
            match op {
                ForwardOperation::Prefill(p) => {
                    batch.request_ids.push(p.base.request_id.clone());
                    batch.request_pool_indices.push(p.base.request_pool_index);
                    batch.input_lengths.push(p.base.input_length);
                    batch.prefill_lengths.push(p.base.prefill_length);
                    for gid in p.base.block_tables.keys() {
                        batch.block_tables.entry(gid.clone()).or_default();
                    }
                }
                ForwardOperation::Decode(d) => {
                    batch.request_ids.push(d.base.request_id.clone());
                    batch.request_pool_indices.push(d.base.request_pool_index);
                    batch.input_lengths.push(d.base.input_length);
                    batch.prefill_lengths.push(d.base.prefill_length);
                    for gid in d.base.block_tables.keys() {
                        batch.block_tables.entry(gid.clone()).or_default();
                    }
                }
            }
        }
        for op in ops.iter() {
            match op {
                ForwardOperation::Prefill(p) => {
                    if let Some(prev) = prefill_source {
                        assert!(
                            prev == p.local_prefill,
                            "one ForwardBatch cannot mix local and remote prefills"
                        );
                    }
                    prefill_source = Some(p.local_prefill);
                    batch.local_prefill = p.local_prefill;
                    batch.input_ids.extend_from_slice(&p.input_ids);
                    batch
                        .shifted_input_ids
                        .extend_from_slice(&p.shifted_input_ids);
                    batch.extend_prefix_lens.push(p.extend_prefix_len);
                }
                ForwardOperation::Decode(d) => {
                    batch.decode_input_ids.push(d.decode_input_id);
                }
            }
        }
        let num_reqs = batch.request_ids.len();
        for table in batch.block_tables.values_mut() {
            table.resize(num_reqs, Vec::new());
        }
        for (row, op) in ops.iter().enumerate() {
            match op {
                ForwardOperation::Prefill(p) => {
                    for (gid, pages) in &p.base.block_tables {
                        batch.block_tables.get_mut(gid).expect("group registered")[row] =
                            pages.clone();
                    }
                }
                ForwardOperation::Decode(d) => {
                    for (gid, pages) in &d.base.block_tables {
                        batch.block_tables.get_mut(gid).expect("group registered")[row] =
                            pages.clone();
                    }
                }
            }
        }
        for (gid, table) in &batch.block_tables {
            let rows = table.len();
            let columns = table.iter().map(|r| r.len()).max().unwrap_or(0);
            let mut contiguous = Vec::with_capacity(rows * columns);
            for request_table in table {
                let mut rt = request_table.clone();
                rt.resize(columns, -1);
                contiguous.extend(rt);
            }
            batch.block_tables_contig.insert(gid.clone(), contiguous);
        }
        batch
    }

    pub fn is_empty(&self) -> bool {
        self.request_ids.is_empty()
    }

    pub fn num_extends(&self) -> usize {
        self.extend_prefix_lens.len()
    }

    pub fn is_local_prefill(&self) -> bool {
        self.num_extends() > 0 && self.local_prefill
    }
}

/// Stable-partition `ops` so prefills precede decodes; returns the split point.
fn partition_prefills_first(ops: &mut [ForwardOperation]) -> usize {
    let mut next_prefill = 0;
    for i in 0..ops.len() {
        if matches!(ops[i], ForwardOperation::Prefill(_)) {
            ops.swap(i, next_prefill);
            next_prefill += 1;
        }
    }
    next_prefill
}

/// One row per group of `Operation` produced by a plan.
#[derive(Debug, Default)]
pub struct ExecutionPlan {
    operations: Vec<Operation>,
    /// Cache child pages newly assigned in this plan. Group identity is
    /// required because one LCM parent can still contain live sibling children.
    /// The runtime clears these exact byte ranges before transfers/forward.
    pub pages_to_zero: BTreeMap<String, Vec<i32>>,
}

impl ExecutionPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(&mut self, operation: Operation) -> &mut Self {
        self.operations.push(operation);
        self
    }

    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
}

/// One executable unit in an execution plan.
#[derive(Debug, Clone)]
pub enum Operation {
    Cache(CacheOperation),
    Forward(ForwardBatch),
}
