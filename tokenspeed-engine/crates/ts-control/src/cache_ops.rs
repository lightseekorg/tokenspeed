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

//! Cache-op accounting and rank-synchronized acknowledgment.
//!
//! A plan's cache ops -- the Host L2 tier's write-backs and load-backs and
//! the retraction snapshot's stores and restores -- run on the data plane
//! from the same plan; each is acknowledged on its own. While any is open,
//! the loop polls the data plane at the head of every round and advances the
//! acknowledgments before it plans, so a cache-gated admission waits no
//! extra round.
//!
//! The scheduler is mirrored on every rank of the replica (the attention-TP
//! group), so an acknowledgment may only advance once every rank holds it:
//! the ranks agree on whether any has work, then intersect what they hold.
//! Completions are not lock-step across ranks, so whether to enter those
//! collectives is decided from the open count, which every rank keeps the
//! same, never from a rank's own view.

use std::collections::BTreeSet;

use ts_collective::{Group, Mesh, Reduce};
use ts_scheduler::{CacheEvent, CacheEventKind, CacheOpView};

use crate::codec::{Reader, Writer};

/// The ranks disagree on the cache ops in flight, or cannot reach each other.
#[derive(Debug, thiserror::Error)]
pub enum CacheOpError {
    /// More acknowledgments arrived than ops are in flight.
    #[error(
        "{acknowledged} cache acknowledgments arrived with only {in_flight} cache ops in flight"
    )]
    Unaccounted {
        /// Acknowledgments in the poll.
        acknowledged: usize,
        /// Ops in flight before it.
        in_flight: usize,
    },
    /// A peer's acknowledgments did not decode.
    #[error("rank {0} sent malformed cache acknowledgments")]
    Malformed(usize),
    /// The replica could not agree.
    #[error(transparent)]
    Collective(#[from] ts_collective::Error),
}

/// The name `cache_event_to_payload` gives a kind; the replica orders the
/// acknowledgments it agrees on by op id, then by this name.
fn kind_name(kind: CacheEventKind) -> &'static str {
    match kind {
        CacheEventKind::WriteBackDone => "WriteBackDoneEvent",
        CacheEventKind::LoadBackDone => "LoadBackDoneEvent",
        CacheEventKind::SnapshotDone => "SnapshotDoneEvent",
        CacheEventKind::RestoreDone => "RestoreDoneEvent",
        _ => unreachable!("a cache event kind the bridge does not define"),
    }
}

const KINDS: [CacheEventKind; 4] = [
    CacheEventKind::WriteBackDone,
    CacheEventKind::LoadBackDone,
    CacheEventKind::SnapshotDone,
    CacheEventKind::RestoreDone,
];

type Key = (u32, &'static str);

fn key(event: &CacheEvent) -> Key {
    (event.op_id, kind_name(event.kind))
}

/// Acknowledgments as one rank shares them: kind, op id.
fn encode(events: &[CacheEvent]) -> Vec<u8> {
    let mut out = Writer::new();
    out.count(events.len());
    for event in events {
        let code = KINDS
            .iter()
            .position(|kind| *kind == event.kind)
            .expect("a bridge kind");
        out.u8(code as u8);
        out.u32(event.op_id);
    }
    out.finish()
}

fn decode(bytes: &[u8]) -> Option<Vec<CacheEvent>> {
    let mut reader = Reader::new(bytes);
    let count = reader.count()?;
    let events = (0..count)
        .map(|_| {
            Some(CacheEvent {
                kind: *KINDS.get(usize::from(reader.u8()?))?,
                op_id: reader.u32()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    reader.finish(events)
}

/// The cache ops dispatched and not acknowledged yet.
#[derive(Debug, Default)]
pub struct CacheOps {
    /// Ops whose acknowledgment has not advanced the scheduler; the same on
    /// every rank (same plans, same agreed acknowledgments).
    open: usize,
    /// Ops this rank has not seen land.
    in_flight: usize,
    /// Acknowledgments this rank holds that the replica has not agreed on,
    /// in the order they landed.
    pending: Vec<CacheEvent>,
}

impl CacheOps {
    /// Nothing in flight.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ops the scheduler has not been told finished.
    pub fn open(&self) -> usize {
        self.open
    }

    /// Count the ops a plan puts in flight. Call it with the plan the data
    /// plane runs, once: a plan counted but never run leaves its ops open
    /// forever.
    pub fn dispatched(&mut self, ops: &[CacheOpView]) {
        self.open += ops.len();
        self.in_flight += ops.len();
    }

    /// Take one poll's acknowledgments, in the order they landed, and return
    /// those every rank of `replica` holds, for the scheduler. Every member
    /// must call this in the same rounds, which it does while
    /// [`CacheOps::open`] is non-zero.
    pub fn acknowledged(
        &mut self,
        polled: Vec<CacheEvent>,
        mesh: &mut Mesh,
        replica: &Group,
    ) -> Result<Vec<CacheEvent>, CacheOpError> {
        if polled.len() > self.in_flight {
            return Err(CacheOpError::Unaccounted {
                acknowledged: polled.len(),
                in_flight: self.in_flight,
            });
        }
        self.in_flight -= polled.len();
        for event in polled {
            match self
                .pending
                .iter_mut()
                .find(|held| key(held) == key(&event))
            {
                Some(held) => *held = event,
                None => self.pending.push(event),
            }
        }

        if replica.size() == 1 {
            // Nothing to agree on: what landed is ready, in landing order.
            let ready = std::mem::take(&mut self.pending);
            self.open -= ready.len();
            return Ok(ready);
        }
        // Completions are not lock-step across ranks, so the ranks agree on
        // whether any has work before gathering.
        let local_work = self.in_flight != 0 || !self.pending.is_empty();
        let agreed = mesh.all_reduce(replica, &[i64::from(local_work)], Reduce::Max)?;
        if agreed[0] == 0 {
            return Ok(Vec::new());
        }
        // Every member enters the gather, even with nothing of its own.
        let gathered = mesh.all_gather(replica, &encode(&self.pending))?;
        let mut members = Vec::with_capacity(gathered.len());
        for (index, bytes) in gathered.iter().enumerate() {
            members.push(decode(bytes).ok_or(CacheOpError::Malformed(replica.ranks()[index]))?);
        }
        let common: BTreeSet<Key> = members
            .iter()
            .map(|events| events.iter().map(key).collect::<BTreeSet<Key>>())
            .reduce(|common, keys| common.intersection(&keys).copied().collect())
            .unwrap_or_default();
        // Ordered by (op id, kind name); the payload is the first member's.
        let mut ready: Vec<CacheEvent> = members
            .swap_remove(0)
            .into_iter()
            .filter(|event| common.contains(&key(event)))
            .collect();
        ready.sort_by_key(key);
        self.pending.retain(|event| !common.contains(&key(event)));
        self.open -= ready.len();
        Ok(ready)
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use ts_collective::Listener;
    use ts_scheduler::CacheOpKind;

    use super::*;

    fn op(kind: CacheOpKind, op_id: u32) -> CacheOpView {
        CacheOpView { kind, op_id }
    }

    fn ack(kind: CacheEventKind, op_id: u32) -> CacheEvent {
        CacheEvent { kind, op_id }
    }

    fn alone() -> (Mesh, Group) {
        let listener = Listener::bind("127.0.0.1").unwrap();
        let address = format!("127.0.0.1:{}", listener.port().unwrap());
        let mesh = Mesh::connect(0, listener, &[address], Duration::from_secs(5)).unwrap();
        (mesh, Group::new(0, vec![0], 0).unwrap())
    }

    #[test]
    fn one_rank_advances_what_landed_in_landing_order() {
        let (mut mesh, replica) = alone();
        let mut ops = CacheOps::new();
        ops.dispatched(&[
            op(CacheOpKind::WriteBack, 1),
            op(CacheOpKind::SnapshotStore, 2),
            op(CacheOpKind::SnapshotStore, 3),
        ]);
        let polled = vec![
            ack(CacheEventKind::SnapshotDone, 3),
            ack(CacheEventKind::WriteBackDone, 1),
        ];
        let ready = ops
            .acknowledged(polled.clone(), &mut mesh, &replica)
            .unwrap();
        assert_eq!(ready, polled);
        assert_eq!(ops.open(), 1);
    }

    #[test]
    fn an_acknowledgment_nothing_dispatched_is_an_error() {
        let (mut mesh, replica) = alone();
        let mut ops = CacheOps::new();
        ops.dispatched(&[op(CacheOpKind::LoadBack, 7)]);
        let polled = vec![
            ack(CacheEventKind::LoadBackDone, 7),
            ack(CacheEventKind::LoadBackDone, 8),
        ];
        let error = ops.acknowledged(polled, &mut mesh, &replica).unwrap_err();
        assert!(matches!(
            error,
            CacheOpError::Unaccounted {
                acknowledged: 2,
                in_flight: 1
            }
        ));
    }

    /// Two replica ranks see the same ops land in different polls.
    #[test]
    fn a_replica_advances_only_what_every_rank_holds() {
        let listeners: Vec<_> = (0..2)
            .map(|_| Listener::bind("127.0.0.1").unwrap())
            .collect();
        let addresses: Vec<String> = listeners
            .iter()
            .map(|listener| format!("127.0.0.1:{}", listener.port().unwrap()))
            .collect();
        // Per rank, what each of three polls returns.
        let polls = [
            vec![
                vec![
                    ack(CacheEventKind::WriteBackDone, 2),
                    ack(CacheEventKind::SnapshotDone, 1),
                ],
                vec![],
                vec![ack(CacheEventKind::RestoreDone, 4)],
            ],
            vec![
                vec![ack(CacheEventKind::SnapshotDone, 1)],
                vec![
                    ack(CacheEventKind::RestoreDone, 4),
                    ack(CacheEventKind::WriteBackDone, 2),
                ],
                vec![],
            ],
        ];
        let threads: Vec<_> = listeners
            .into_iter()
            .zip(polls)
            .enumerate()
            .map(|(rank, (listener, polls))| {
                let addresses = addresses.clone();
                thread::spawn(move || {
                    let mut mesh =
                        Mesh::connect(rank, listener, &addresses, Duration::from_secs(5)).unwrap();
                    let replica = Group::new(0, vec![0, 1], rank).unwrap();
                    let mut ops = CacheOps::new();
                    ops.dispatched(&[
                        op(CacheOpKind::SnapshotStore, 1),
                        op(CacheOpKind::WriteBack, 2),
                        op(CacheOpKind::SnapshotRestore, 4),
                    ]);
                    let rounds: Vec<Vec<CacheEvent>> = polls
                        .into_iter()
                        .map(|polled| ops.acknowledged(polled, &mut mesh, &replica).unwrap())
                        .collect();
                    (rounds, ops.open())
                })
            })
            .collect();
        for thread in threads {
            let (rounds, open) = thread.join().unwrap();
            assert_eq!(
                rounds,
                [
                    vec![ack(CacheEventKind::SnapshotDone, 1)],
                    vec![ack(CacheEventKind::WriteBackDone, 2)],
                    vec![ack(CacheEventKind::RestoreDone, 4)],
                ]
            );
            assert_eq!(open, 0);
        }
    }
}
