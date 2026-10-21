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

//! The PD transfer peer as the scheduling loop sees it: requests register at
//! admission, and every round the replica agrees on which transfers changed
//! state.

use ts_collective::{Group, Mesh};

/// Where a request's peer node reaches it: the prefill deployment's
/// bootstrap server and the request's room.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapInfo {
    /// The bootstrap server's host.
    pub host: String,
    /// Its port.
    pub port: u16,
    /// The room both nodes know the request by.
    pub room: u64,
}

/// A D-role request whose KV landed.
#[derive(Debug, Clone, PartialEq)]
pub struct RemotePrefill {
    /// The request.
    pub rid: String,
    /// The prefill node's first token, -1 if it sent none.
    pub bootstrap_token: i32,
    /// Prompt tokens the prefill node served from its cache.
    pub cached_tokens: i64,
    /// The first token's logprob, if the prefill node sent one.
    pub bootstrap_logprob: Option<f64>,
    /// The prefill node's drafter candidates, if any.
    pub spec_candidate_ids: Option<Vec<i32>>,
    /// The request's slot here.
    pub request_pool_index: i32,
}

/// A transfer whose state the replica agreed changed.
#[derive(Debug, Clone, PartialEq)]
pub enum PdEvent {
    /// The request's peers are connected.
    Bootstrapped(String),
    /// The request's transfer failed.
    Failed(String),
    /// P role: the request's KV reached the decode node.
    Succeeded(String),
    /// D role: the request's KV landed.
    RemotePrefillDone(RemotePrefill),
}

/// A PD failure the engine cannot serve past.
#[derive(Debug, thiserror::Error)]
pub enum PdError {
    /// The replica could not agree.
    #[error(transparent)]
    Collective(#[from] ts_collective::Error),
    /// The transfer peer broke.
    #[error("PD transfer: {0}")]
    Peer(String),
}

/// The transfer peer's control face.
pub trait TransferPeer {
    /// Track an admitted request, before the scheduler can plan it.
    fn register(&mut self, rid: &str, bootstrap: &BootstrapInfo) -> Result<(), PdError>;

    /// The transfers whose state changed since the last poll, as every
    /// member of `replica` sees them: a rank reports the least advanced
    /// state any member holds. Every member calls this in the same rounds.
    fn poll(&mut self, mesh: &mut Mesh, replica: &Group) -> Result<Vec<PdEvent>, PdError>;

    /// P role: a committed prompt's usage, sent with its completion: the
    /// prompt tokens served from the cache and the first token's logprob.
    fn record_usage(&mut self, rid: &str, cached_tokens: i64, bootstrap_logprob: Option<f64>);

    /// A transfer is still tracked: it may yet read or write the cache.
    fn has_in_flight(&self) -> bool;

    /// EPD, P role: tell the decode node waiting on room `room` that the
    /// request failed before it could register (its embeddings never
    /// arrived), so its receiver fails instead of waiting. Best effort.
    fn abort_room(&mut self, rid: &str, room: u64);
}
