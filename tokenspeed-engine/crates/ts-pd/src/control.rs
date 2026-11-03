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

//! The PD transfer peer, and the EPD embedding receiver, as the scheduling
//! loop sees them: requests register at admission, and every round the
//! replica agrees on which transfers changed state.

use std::time::{Duration, Instant};

use ts_collective::{Group, Mesh};

/// How long a failed or cancelled transfer may take to stop before its peer
/// warns that the request's pages are still pinned.
pub(crate) const DRAIN_STALL: Duration = Duration::from_secs(30);

/// A failed or cancelled transfer whose readers or writers may still touch
/// its request's pages ([`PdEvent::Stopped`] has not been reported).
pub(crate) struct Draining {
    pub(crate) rid: String,
    pub(crate) room: u64,
    pub(crate) since: Instant,
}

/// The transfers draining for [`DRAIN_STALL`] or longer, as log entries; at
/// most once per [`DRAIN_STALL`] (`logged_at`). One that never stops (a
/// write whose DMA may still land, a peer lost mid-drain) keeps its pages
/// pinned and holds off a cache release.
pub(crate) fn stalled_drains(
    draining: &[Draining],
    logged_at: &mut Option<Instant>,
    now: Instant,
) -> Vec<String> {
    if logged_at.is_some_and(|at| now - at < DRAIN_STALL) {
        return Vec::new();
    }
    let stalled: Vec<String> = draining
        .iter()
        .filter(|drain| now - drain.since >= DRAIN_STALL)
        .take(8)
        .map(|drain| {
            format!(
                "{}(room={}, age={:.0}s)",
                drain.rid,
                drain.room,
                (now - drain.since).as_secs_f64()
            )
        })
        .collect();
    if !stalled.is_empty() {
        *logged_at = Some(now);
    }
    stalled
}

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
    /// The prefill node's first token: a landing without one is a failed
    /// transfer, as the request's first decode feeds on it.
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
    /// The request's transfer failed. Its readers or writers may still be
    /// stopping: [`PdEvent::Stopped`] follows.
    Failed(String),
    /// A failed or cancelled transfer stopped on every member: no reader or
    /// writer touches the request's pages any more.
    Stopped(String),
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

    /// The scheduling loop ended `rid` while its transfer may still run:
    /// stop it. Unless it succeeds first, [`PdEvent::Stopped`] reports it
    /// once every reader and writer stopped. Every member calls this for the
    /// same requests in the same rounds.
    fn cancel(&mut self, rid: &str);

    /// A transfer is still tracked or stopping: it may yet read or write
    /// the cache.
    fn has_in_flight(&self) -> bool;

    /// EPD, P role: tell the decode node waiting on room `room` that the
    /// request failed before it could register (its embeddings never
    /// arrived), so its receiver fails instead of waiting. Best effort.
    fn abort_room(&mut self, rid: &str, room: u64);
}

/// EPD: one multimodal item of a request whose embedding an encode node
/// computes, as the request names it (`item.encode_handshake`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingItem {
    /// The item's position among the request's multimodal items.
    pub index: usize,
    /// The encode node's bootstrap server.
    pub host: String,
    /// Its port.
    pub port: u16,
    /// The room the encode node ships the item's embedding under.
    pub room: u64,
    /// The embedding's rows: the item's tokens.
    pub rows: i64,
}

/// EPD: where one received item's rows landed, for the data plane to
/// publish onto the item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandedItem {
    /// The item's position among the request's multimodal items.
    pub index: usize,
    /// The receive slot holding the rows; `None` when this rank received
    /// none (an empty shard).
    pub slot: Option<usize>,
    /// The first row this rank received within the image.
    pub row_start: i64,
    /// The rows it received, packed at the slot's start.
    pub row_count: i64,
    /// The image's rows.
    pub rows: i64,
    /// The rows are this rank's shard: the ranks rebuild the rest.
    pub sharded: bool,
}

/// EPD: a request whose items' embeddings all landed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandedRequest {
    /// The request.
    pub rid: String,
    /// Its items, in the request's order.
    pub items: Vec<LandedItem>,
}

/// EPD, a prefill node's first stage: receiving the encode nodes'
/// embeddings of the requests waiting for them, as the scheduling loop
/// drives it. Every member of the replica stages the same requests in the
/// same order and settles the same codes.
pub trait EmbeddingReceiver {
    /// Start receiving `rid`'s items' embeddings.
    fn stage(&mut self, rid: &str, items: &[EmbeddingItem]);

    /// Every staged request's state on this rank, in staging order: 0
    /// failed or timed out, 1 receiving, 2 landed.
    fn poll(&mut self) -> Vec<i64>;

    /// Act on the replica-agreed `codes` (`poll` order): forget the settled
    /// requests, and return the landed ones' items for the data plane to
    /// publish. Their slots stay leased until [`EmbeddingReceiver::release`].
    fn settle(&mut self, codes: &[i64]) -> Vec<LandedRequest>;

    /// The data plane copied these slots' rows out: they take new receives.
    fn release(&mut self, slots: &[usize]);

    /// Whether a landed slot still waits for its rows to be copied out.
    fn awaiting_release(&self) -> bool;
}

/// EPD, an encode rank: one item's embedding in its send slot, as the
/// transfer engine reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingChunk {
    /// The room the prefill ranks receive it under.
    pub room: u64,
    /// The rows' address.
    pub source: u64,
    /// The image's rows.
    pub rows: i64,
    /// Elements per row.
    pub hidden: i64,
    /// The rows' dtype (`str(torch.dtype)`).
    pub dtype: String,
    /// The rows' bytes.
    pub nbytes: u64,
    /// The deepstack rows' address; 0 without deepstack.
    pub deepstack_source: u64,
    /// Elements per deepstack row; 0 without deepstack.
    pub deepstack_width: i64,
    /// The deepstack rows' bytes.
    pub deepstack_nbytes: u64,
}

/// EPD, an encode rank: sending each item's embedding to the prefill ranks
/// that wait for it, as the scheduling loop drives it.
pub trait EmbeddingSender {
    /// `room`'s request arrived; false when another request holds the room.
    fn begin(&mut self, room: u64) -> bool;

    /// Hand over a begun room's rows: they are written once every prefill
    /// rank of the block asked, and must stay in place until the room
    /// concludes.
    fn send(&mut self, chunk: EmbeddingChunk);

    /// Conclude a begun room failed before its rows were handed over,
    /// telling the prefill ranks that asked.
    fn fail(&mut self, room: u64, reason: &str);

    /// The begun rooms that concluded since the last call, sent or failed:
    /// their rows' slots take new rows.
    fn concluded(&mut self) -> Vec<u64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draining(rid: &str, age: Duration, now: Instant) -> Draining {
        Draining {
            rid: rid.to_owned(),
            room: 7,
            since: now - age,
        }
    }

    #[test]
    fn a_drain_past_the_stall_is_reported_once_per_interval() {
        let now = Instant::now() + 2 * DRAIN_STALL;
        let list = [
            draining("young", DRAIN_STALL / 2, now),
            draining("stuck", DRAIN_STALL, now),
        ];
        let mut logged_at = None;
        let stalled = stalled_drains(&list, &mut logged_at, now);
        assert_eq!(stalled.len(), 1);
        assert!(stalled[0].starts_with("stuck(room=7, age="));
        assert_eq!(logged_at, Some(now));
        // Throttled until a whole interval passed since the report.
        assert!(stalled_drains(&list, &mut logged_at, now + DRAIN_STALL / 2).is_empty());
        assert_eq!(
            stalled_drains(&list, &mut logged_at, now + DRAIN_STALL).len(),
            2
        );
        // Nothing past the stall: nothing reported, the throttle untouched.
        let mut fresh = None;
        assert!(stalled_drains(&list[..1], &mut fresh, now).is_empty());
        assert_eq!(fresh, None);
    }
}
