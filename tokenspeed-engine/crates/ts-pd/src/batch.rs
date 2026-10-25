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

//! A plan's batch as a transfer reads it: the rows of a remote stream or a
//! forward, with their block tables.

use std::sync::{Arc, Mutex};

use crate::layout::manifest::BlockTables;

/// A batch's rows (the scheduler's `TransferBatchView`); extend rows first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchRows {
    /// Request ids.
    pub request_ids: Vec<String>,
    /// Each row's request slot.
    pub request_pool_indices: Vec<i32>,
    /// Tokens each row feeds this forward.
    pub input_lengths: Vec<i32>,
    /// Each row's prompt length.
    pub prefill_lengths: Vec<i32>,
    /// Extend rows: tokens already in the cache ahead of the chunk.
    pub extend_prefix_lens: Vec<i32>,
    /// Extend rows, a Decode admission: every state group's prompt-end
    /// checkpoint was found locally.
    pub state_prefix_cached: Vec<bool>,
    /// Decode rows: on a Prefill node's remote decode, each request's first
    /// output token.
    pub decode_input_ids: Vec<i32>,
    /// Decode rows: the drafter's candidates, empty for none.
    pub spec_candidate_ids: Vec<Vec<i32>>,
    /// Extend rows.
    pub num_extends: usize,
    /// Every cache group's table.
    pub tables: BlockTables,
}

/// A plan's remote stream, staged by the forward thread when it hands the
/// plan to the data plane, until the data plane hands it over once the
/// stream's device prerequisites are met.
#[derive(Debug, Clone, Default)]
pub(crate) struct Staged(Arc<Mutex<Option<BatchRows>>>);

impl Staged {
    pub(crate) fn stage(&self, rows: BatchRows) -> Result<(), String> {
        let mut staged = self.0.lock().expect("the staging lock is poisoned");
        if staged.is_some() {
            return Err(
                "a plan's remote stream was staged before the previous was handed over".to_owned(),
            );
        }
        *staged = Some(rows);
        Ok(())
    }

    pub(crate) fn take(&self) -> Result<BatchRows, String> {
        self.0
            .lock()
            .expect("the staging lock is poisoned")
            .take()
            .ok_or_else(|| {
                "the data plane handed over a remote stream the plan has none of".to_owned()
            })
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.0
            .lock()
            .expect("the staging lock is poisoned")
            .is_some()
    }
}
