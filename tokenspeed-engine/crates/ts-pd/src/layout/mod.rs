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

//! Transfer planning over the ranks' cache layouts, ported from the Python
//! PD planner (`pd/cache_protocol.py`, `pd/transfer_plan.py` and the
//! Mooncake prefill planning).
//!
//! * [`contract`]: a rank's cache transfer contract, as the wire carries it.
//! * [`manifest`]: one request's blocks per group, read off the scheduler's
//!   block tables, and the blocks one layerwise chunk makes ready.
//! * [`route`]: which Prefill ranks serve which Decode rank, with which row
//!   fragments of which fields, across unequal TP sizes, pipeline stages and
//!   KVP page sharding.
//! * [`registration`]: a Decode rank's registration and pre-allocation
//!   messages, and the route a Prefill rank plans for a registration.
//! * [`copies`]: the copies one transfer makes, as the transfer engine takes
//!   them.
//!
//! Nothing here holds state or moves bytes; the peers do.

pub mod contract;
pub mod copies;
pub mod manifest;
pub mod registration;
pub mod route;

/// A layout, manifest, route or message that breaks the cache transfer
/// contract.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct LayoutError(pub String);

impl LayoutError {
    /// An error with `message`.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
