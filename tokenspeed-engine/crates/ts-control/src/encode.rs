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

//! An EPD encode node's loop, which an encode rank's host runs in place of
//! the language model's scheduling loop.
//!
//! The encode node embeds each request's one item with the vision tower and
//! ships the embedding to the prefill ranks waiting for it under the item's
//! room ([`EmbeddingSender`]). Each round the attention-TP group's first
//! rank relays the frontend's requests to the others, in one exchange that
//! also agrees on the send slots every rank has free, so every rank forms
//! the same vision-tower batches (the tower is TP-sharded); each rank then
//! sends to its own block of prefill ranks.
//!
//! A request is admitted by the data plane, whose embedding cache may hold
//! its item; the others wait in arrival order for a batch under the token
//! and item budgets. Each batched item leases a slot of the send ring, which
//! the data plane copies its rows into; once the copy finished, the rows go
//! to the sender, and the slot returns when the room concludes. Only what is
//! the same on every rank decides a batch: a room's conclusion, which each
//! rank sees at its own time, only frees its slot.

use std::{
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use bytes::Bytes;
use ts_collective::{Group, Mesh};
use ts_pd::{EmbeddingChunk, EmbeddingSender};
use ts_wire::EncodeRequest;

use crate::{
    codec::{Reader, Writer},
    event_loop::DataPlaneError,
};

/// How long the first rank waits for a request while no rank has work.
const IDLE_WAIT: Duration = Duration::from_millis(50);

/// How long it waits while work is in flight but the last round moved
/// nothing.
const BUSY_WAIT: Duration = Duration::from_millis(1);

/// One item's rows, copied into its send slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedRows {
    /// The image's rows.
    pub rows: i64,
    /// Elements per row.
    pub hidden: i64,
    /// The rows' dtype (`str(torch.dtype)`).
    pub dtype: String,
    /// The rows' bytes.
    pub nbytes: u64,
    /// Elements per deepstack row; 0 without deepstack.
    pub deepstack_width: i64,
    /// The deepstack rows' bytes.
    pub deepstack_nbytes: u64,
}

/// An encode rank's device side, as the loop drives it. Every rank of the
/// attention-TP group makes the same calls in the same order.
pub trait EncodeDataPlane {
    /// Take encode requests (each one's wire frames, under its admission
    /// number), and say for each whether the embedding cache holds its
    /// first item.
    fn admit(&mut self, requests: &[(u64, Vec<Bytes>)]) -> Result<Vec<bool>, DataPlaneError>;

    /// Embed these admitted items in one vision-tower forward (those the
    /// cache held skip it) and copy each one's rows into its send slot:
    /// `(admission, slot)` each.
    fn encode(&mut self, batch: &[(u64, usize)]) -> Result<(), DataPlaneError>;

    /// Forget admitted requests that will not be encoded.
    fn release(&mut self, admissions: &[u64]) -> Result<(), DataPlaneError>;

    /// The encoded items whose rows reached their slots since the last
    /// call: each one's rows, or why it failed.
    #[allow(clippy::type_complexity)]
    fn encoded(&mut self) -> Result<Vec<(u64, Result<EncodedRows, String>)>, DataPlaneError>;
}

/// Where the first rank takes the frontend's requests from.
pub trait Intake {
    /// The messages that arrived, each as its frames, waiting up to `wait`
    /// for the first.
    fn receive(&mut self, wait: Duration) -> Vec<Vec<Bytes>>;
}

/// The send ring: device buffers of equal slots the transfer engine reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ring {
    /// Its slots.
    pub slots: usize,
    /// A slot's bytes for an item's rows.
    pub slot_bytes: u64,
    /// The rows' buffer.
    pub base: u64,
    /// A slot's bytes for an item's deepstack rows; 0 without deepstack.
    pub deepstack_slot_bytes: u64,
    /// The deepstack rows' buffer; 0 without deepstack.
    pub deepstack_base: u64,
}

/// What an encode rank's loop runs with.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// The most item rows (vision tokens) a forward embeds; one larger item
    /// still runs alone.
    pub max_batch_tokens: i64,
    /// The most items a forward embeds.
    pub max_batch_items: usize,
    /// The send ring.
    pub ring: Ring,
}

/// A failure that ends an encode rank's loop.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    /// The data plane failed.
    #[error(transparent)]
    DataPlane(#[from] DataPlaneError),
    /// The ranks could not exchange the round.
    #[error(transparent)]
    Collective(#[from] ts_collective::Error),
    /// A rank's part of the round did not decode.
    #[error("an encode round's exchange is cut short")]
    Exchange,
}

/// One admitted request, until its rows go to the sender.
#[derive(Debug)]
struct Admitted {
    room: u64,
    /// Its item's rows, the batch budget's unit.
    rows: i64,
    /// Another request held its room when it arrived: it is embedded like
    /// any other (batches are the same on every rank), but not sent.
    duplicate: bool,
}

/// The round as every rank agrees on it.
#[derive(Debug)]
struct Round {
    stop: bool,
    messages: Vec<Vec<Bytes>>,
    /// The send slots free on every rank.
    free: usize,
    /// Some rank has work in flight.
    busy: bool,
}

/// An encode rank's loop.
pub struct EncodeLoop<D: EncodeDataPlane, S: EmbeddingSender> {
    config: EncodeConfig,
    data_plane: D,
    sender: S,
    /// The frontend's requests, on the group's first rank.
    intake: Option<Box<dyn Intake + Send>>,
    mesh: Mesh,
    /// The attention-TP group.
    group: Group,
    next_admission: u64,
    requests: HashMap<u64, Admitted>,
    /// Admissions the cache held, waiting for a slot.
    hits: VecDeque<u64>,
    /// Admissions waiting for a forward, in arrival order.
    misses: VecDeque<u64>,
    /// Free send slots, the lowest last.
    free: Vec<usize>,
    /// The slot each item in flight holds, by admission.
    leased: HashMap<u64, usize>,
    /// The admission whose rows a room's send reads.
    sending: HashMap<u64, u64>,
    /// Items handed to the data plane whose rows have not come back.
    encoding: usize,
    /// What the first rank waits for a request before the next round.
    wait: Duration,
}

impl<D: EncodeDataPlane, S: EmbeddingSender> EncodeLoop<D, S> {
    /// A loop over the attention-TP `group`; `intake` on its first rank.
    pub fn new(
        config: EncodeConfig,
        data_plane: D,
        sender: S,
        intake: Option<Box<dyn Intake + Send>>,
        mesh: Mesh,
        group: Group,
    ) -> Self {
        let free = (0..config.ring.slots).rev().collect();
        Self {
            config,
            data_plane,
            sender,
            intake,
            mesh,
            group,
            next_admission: 0,
            requests: HashMap::new(),
            hits: VecDeque::new(),
            misses: VecDeque::new(),
            free,
            leased: HashMap::new(),
            sending: HashMap::new(),
            encoding: 0,
            wait: IDLE_WAIT,
        }
    }

    /// Serve until `stop` is set on the group's first rank.
    pub fn run(&mut self, stop: &AtomicBool) -> Result<(), EncodeError> {
        loop {
            let received = match self.intake.as_mut() {
                Some(intake) => intake.receive(self.wait),
                None => Vec::new(),
            };
            let round = self.exchange(stop.load(Ordering::Relaxed), received)?;
            if round.stop {
                return Ok(());
            }
            let mut moved = !round.messages.is_empty();
            self.admit(round.messages)?;
            moved |= self.batch(round.free)?;
            moved |= self.hand_over()?;
            moved |= self.reclaim();
            self.wait = if !round.busy && !self.busy() {
                IDLE_WAIT
            } else if moved {
                Duration::ZERO
            } else {
                BUSY_WAIT
            };
        }
    }

    /// This rank has work in flight.
    fn busy(&self) -> bool {
        !self.requests.is_empty() || self.encoding > 0 || !self.leased.is_empty()
    }

    /// Agree on the round: the first rank's messages and stop request, the
    /// slots free everywhere and whether any rank is busy.
    fn exchange(&mut self, stop: bool, messages: Vec<Vec<Bytes>>) -> Result<Round, EncodeError> {
        let (free, busy) = (self.free.len(), self.busy());
        if self.group.size() == 1 {
            return Ok(Round {
                stop,
                messages,
                free,
                busy,
            });
        }
        let mut part = Writer::new();
        part.flag(busy);
        part.count(free);
        if self.group.index() == 0 {
            part.flag(stop);
            part.count(messages.len());
            for frames in &messages {
                part.count(frames.len());
                for frame in frames {
                    part.bytes(frame);
                }
            }
        }
        let parts = self.mesh.all_gather(&self.group, &part.finish())?;
        let mut round = Round {
            stop: false,
            messages: Vec::new(),
            free,
            busy: false,
        };
        for (member, part) in parts.iter().enumerate() {
            let mut reader = Reader::new(part);
            round.busy |= reader.flag().ok_or(EncodeError::Exchange)?;
            round.free = round.free.min(reader.count().ok_or(EncodeError::Exchange)?);
            if member == 0 {
                round.stop = reader.flag().ok_or(EncodeError::Exchange)?;
                let count = reader.count().ok_or(EncodeError::Exchange)?;
                for _ in 0..count {
                    let frames = reader.count().ok_or(EncodeError::Exchange)?;
                    let frames = (0..frames)
                        .map(|_| reader.bytes().map(Bytes::copy_from_slice))
                        .collect::<Option<Vec<_>>>()
                        .ok_or(EncodeError::Exchange)?;
                    round.messages.push(frames);
                }
            }
            reader.finish(()).ok_or(EncodeError::Exchange)?;
        }
        Ok(round)
    }

    /// Admit the round's requests. A request carries one item, which is one
    /// room; one that does not fails its room.
    fn admit(&mut self, messages: Vec<Vec<Bytes>>) -> Result<(), EncodeError> {
        if messages.is_empty() {
            return Ok(());
        }
        let mut admitted = Vec::with_capacity(messages.len());
        let mut taken = Vec::with_capacity(messages.len());
        for frames in messages {
            let request: EncodeRequest = match ts_wire::decode_frames(&frames) {
                Ok(request) => request,
                Err(error) => {
                    tracing::error!(%error, "dropping an encode request that does not decode");
                    continue;
                }
            };
            self.next_admission += 1;
            let admission = self.next_admission;
            let room = request.bootstrap_room;
            let item = match request.items.as_slice() {
                [item] => Some(Admitted {
                    room,
                    rows: item
                        .offsets
                        .iter()
                        .flatten()
                        .map(|&(start, end)| end - start + 1)
                        .sum(),
                    duplicate: !self.sender.begin(room),
                }),
                items => {
                    tracing::error!(
                        request_id = request.request_id,
                        items = items.len(),
                        "an encode request carries one item"
                    );
                    if self.sender.begin(room) {
                        self.sender.fail(room, "an encode request carries one item");
                    }
                    None
                }
            };
            if item.as_ref().is_some_and(|item| item.duplicate) {
                tracing::error!(
                    request_id = request.request_id,
                    room,
                    "another encode request holds the room; its embedding is not sent"
                );
            }
            // The data plane still takes a refused request, so its
            // shared-memory pixels are consumed.
            admitted.push((admission, frames));
            taken.push((admission, item));
        }
        if admitted.is_empty() {
            return Ok(());
        }
        let cached = self.data_plane.admit(&admitted)?;
        if cached.len() != admitted.len() {
            return Err(DataPlaneError(format!(
                "{} cache answers for {} admitted requests",
                cached.len(),
                admitted.len()
            ))
            .into());
        }
        let mut refused = Vec::new();
        for ((admission, item), cached) in taken.into_iter().zip(cached) {
            match item {
                Some(item) => {
                    self.requests.insert(admission, item);
                    if cached {
                        self.hits.push_back(admission);
                    } else {
                        self.misses.push_back(admission);
                    }
                }
                None => refused.push(admission),
            }
        }
        if !refused.is_empty() {
            self.data_plane.release(&refused)?;
        }
        Ok(())
    }

    /// Form at most one forward from what waits, within the `free` slots
    /// every rank has: the items the cache held first (they skip the
    /// tower), then those waiting in arrival order under the budgets.
    fn batch(&mut self, free: usize) -> Result<bool, EncodeError> {
        let mut batch = Vec::new();
        while batch.len() < free {
            let Some(admission) = self.hits.pop_front() else {
                break;
            };
            batch.push(admission);
        }
        let (mut tokens, mut items) = (0, 0);
        while batch.len() < free {
            let Some(&admission) = self.misses.front() else {
                break;
            };
            let rows = self.requests[&admission].rows;
            if items > 0
                && (tokens + rows > self.config.max_batch_tokens
                    || items >= self.config.max_batch_items)
            {
                break;
            }
            self.misses.pop_front();
            batch.push(admission);
            tokens += rows;
            items += 1;
        }
        if batch.is_empty() {
            return Ok(false);
        }
        let leased: Vec<(u64, usize)> = batch
            .into_iter()
            .map(|admission| {
                let slot = self
                    .free
                    .pop()
                    .expect("every rank has the agreed slots free");
                self.leased.insert(admission, slot);
                (admission, slot)
            })
            .collect();
        self.encoding += leased.len();
        self.data_plane.encode(&leased)?;
        Ok(true)
    }

    /// Hand the items whose rows reached their slots to the sender.
    fn hand_over(&mut self) -> Result<bool, EncodeError> {
        let encoded = self.data_plane.encoded()?;
        let moved = !encoded.is_empty();
        let ring = self.config.ring;
        for (admission, outcome) in encoded {
            self.encoding = self.encoding.saturating_sub(1);
            let (Some(request), Some(&slot)) = (
                self.requests.remove(&admission),
                self.leased.get(&admission),
            ) else {
                return Err(DataPlaneError(format!(
                    "the data plane encoded admission {admission}, which it was not given"
                ))
                .into());
            };
            if request.duplicate {
                self.free_slot(admission);
                continue;
            }
            match outcome {
                Ok(rows) => {
                    self.sending.insert(request.room, admission);
                    self.sender.send(EmbeddingChunk {
                        room: request.room,
                        source: ring.base + slot as u64 * ring.slot_bytes,
                        rows: rows.rows,
                        hidden: rows.hidden,
                        dtype: rows.dtype,
                        nbytes: rows.nbytes,
                        deepstack_source: if rows.deepstack_width > 0 {
                            ring.deepstack_base + slot as u64 * ring.deepstack_slot_bytes
                        } else {
                            0
                        },
                        deepstack_width: rows.deepstack_width,
                        deepstack_nbytes: rows.deepstack_nbytes,
                    });
                }
                Err(reason) => {
                    self.free_slot(admission);
                    self.sender.fail(request.room, &reason);
                }
            }
        }
        Ok(moved)
    }

    /// Free the slots of the rooms that concluded.
    fn reclaim(&mut self) -> bool {
        let concluded = self.sender.concluded();
        for room in &concluded {
            if let Some(admission) = self.sending.remove(room) {
                self.free_slot(admission);
            }
        }
        !concluded.is_empty()
    }

    fn free_slot(&mut self, admission: u64) {
        if let Some(slot) = self.leased.remove(&admission) {
            self.free.push(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{Arc, Mutex},
    };

    use ts_collective::Listener;
    use ts_wire::{Dict, Modality, MultimodalDataItem};

    use super::*;

    fn request(request_id: &str, room: u64, offsets: Vec<Vec<(i64, i64)>>) -> Vec<Bytes> {
        ts_wire::encode_frames(&EncodeRequest {
            rid: None,
            http_worker_ipc: None,
            request_id: request_id.to_owned(),
            bootstrap_host: "10.0.0.1".to_owned(),
            bootstrap_port: 9001,
            bootstrap_room: room,
            items: offsets
                .into_iter()
                .map(|offsets| MultimodalDataItem {
                    modality: Modality::Image,
                    hash: Some(room),
                    pad_value: None,
                    offsets: Some(offsets),
                    feature: None,
                    feature_shm: None,
                    model_specific_data: Dict::default(),
                    encoded: None,
                    encoded_deepstack: None,
                    encode_handshake: None,
                })
                .collect(),
        })
    }

    #[derive(Default)]
    struct Log {
        admitted: Vec<u64>,
        batches: Vec<Vec<u64>>,
        released: Vec<u64>,
    }

    /// Caches nothing but room 1000's item; rows land one call after the
    /// forward.
    struct FakeDevice {
        log: Arc<Mutex<Log>>,
        rooms: HashMap<u64, u64>,
        pending: Vec<(u64, usize)>,
    }

    impl EncodeDataPlane for FakeDevice {
        fn admit(&mut self, requests: &[(u64, Vec<Bytes>)]) -> Result<Vec<bool>, DataPlaneError> {
            let mut cached = Vec::new();
            for (admission, frames) in requests {
                let request: EncodeRequest = ts_wire::decode_frames(frames).unwrap();
                self.rooms.insert(*admission, request.bootstrap_room);
                self.log.lock().unwrap().admitted.push(*admission);
                cached.push(request.bootstrap_room == 1000);
            }
            Ok(cached)
        }

        fn encode(&mut self, batch: &[(u64, usize)]) -> Result<(), DataPlaneError> {
            self.log
                .lock()
                .unwrap()
                .batches
                .push(batch.iter().map(|&(admission, _)| admission).collect());
            self.pending.extend_from_slice(batch);
            Ok(())
        }

        fn release(&mut self, admissions: &[u64]) -> Result<(), DataPlaneError> {
            self.log
                .lock()
                .unwrap()
                .released
                .extend_from_slice(admissions);
            Ok(())
        }

        fn encoded(&mut self) -> Result<Vec<(u64, Result<EncodedRows, String>)>, DataPlaneError> {
            Ok(std::mem::take(&mut self.pending)
                .into_iter()
                .map(|(admission, _)| {
                    // Room 13's item does not fit its slot.
                    let outcome = if self.rooms[&admission] == 13 {
                        Err("too large".to_owned())
                    } else {
                        Ok(EncodedRows {
                            rows: 4,
                            hidden: 8,
                            dtype: "torch.bfloat16".to_owned(),
                            nbytes: 64,
                            deepstack_width: 0,
                            deepstack_nbytes: 0,
                        })
                    };
                    (admission, outcome)
                })
                .collect())
        }
    }

    #[derive(Default)]
    struct Sends {
        begun: HashSet<u64>,
        sent: Vec<EmbeddingChunk>,
        failed: Vec<u64>,
        concluded: Vec<u64>,
    }

    /// Concludes a sent room `delay` calls of `concluded` later.
    struct FakeSender {
        sends: Arc<Mutex<Sends>>,
        delay: usize,
        waiting: Vec<(u64, usize)>,
    }

    impl EmbeddingSender for FakeSender {
        fn begin(&mut self, room: u64) -> bool {
            self.sends.lock().unwrap().begun.insert(room)
        }

        fn send(&mut self, chunk: EmbeddingChunk) {
            self.waiting.push((chunk.room, self.delay));
            self.sends.lock().unwrap().sent.push(chunk);
        }

        fn fail(&mut self, room: u64, _reason: &str) {
            let mut sends = self.sends.lock().unwrap();
            sends.failed.push(room);
            sends.begun.remove(&room);
        }

        fn concluded(&mut self) -> Vec<u64> {
            let mut done = Vec::new();
            self.waiting.retain_mut(|(room, delay)| {
                if *delay == 0 {
                    done.push(*room);
                    false
                } else {
                    *delay -= 1;
                    true
                }
            });
            let mut sends = self.sends.lock().unwrap();
            for room in &done {
                sends.begun.remove(room);
            }
            sends.concluded.extend(&done);
            done
        }
    }

    struct Queue(Arc<Mutex<VecDeque<Vec<Vec<Bytes>>>>>);

    impl Intake for Queue {
        fn receive(&mut self, _wait: Duration) -> Vec<Vec<Bytes>> {
            self.0.lock().unwrap().pop_front().unwrap_or_default()
        }
    }

    fn config(slots: usize) -> EncodeConfig {
        EncodeConfig {
            max_batch_tokens: 10,
            max_batch_items: 3,
            ring: Ring {
                slots,
                slot_bytes: 1 << 20,
                base: 0x1000_0000,
                deepstack_slot_bytes: 0,
                deepstack_base: 0,
            },
        }
    }

    struct Rank {
        encode: EncodeLoop<FakeDevice, FakeSender>,
        log: Arc<Mutex<Log>>,
        sends: Arc<Mutex<Sends>>,
    }

    /// `ranks` loops over one mesh, the first taking `rounds` of requests;
    /// each rank's sends conclude after its own delay.
    fn ranks(slots: usize, delays: &[usize], rounds: Vec<Vec<Vec<Bytes>>>) -> Vec<Rank> {
        let listeners: Vec<Listener> = delays
            .iter()
            .map(|_| Listener::bind("127.0.0.1").unwrap())
            .collect();
        let addresses: Vec<String> = listeners
            .iter()
            .map(|listener| format!("127.0.0.1:{}", listener.port().unwrap()))
            .collect();
        let size = delays.len();
        let intake = Arc::new(Mutex::new(VecDeque::from(rounds)));
        let handles: Vec<_> = listeners
            .into_iter()
            .zip(delays)
            .enumerate()
            .map(|(rank, (listener, &delay))| {
                let addresses = addresses.clone();
                let intake = intake.clone();
                std::thread::spawn(move || {
                    let mesh =
                        Mesh::connect(rank, listener, &addresses, Duration::from_secs(5)).unwrap();
                    let group = Group::new(0, (0..size).collect(), rank).unwrap();
                    let log = Arc::new(Mutex::new(Log::default()));
                    let sends = Arc::new(Mutex::new(Sends::default()));
                    let encode = EncodeLoop::new(
                        config(slots),
                        FakeDevice {
                            log: log.clone(),
                            rooms: HashMap::new(),
                            pending: Vec::new(),
                        },
                        FakeSender {
                            sends: sends.clone(),
                            delay,
                            waiting: Vec::new(),
                        },
                        (rank == 0).then(|| Box::new(Queue(intake)) as Box<dyn Intake + Send>),
                        mesh,
                        group,
                    );
                    Rank { encode, log, sends }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    }

    /// Run every rank `rounds` rounds, in lockstep.
    fn run(ranks: &mut [Rank], rounds: usize) {
        std::thread::scope(|scope| {
            for rank in ranks.iter_mut() {
                scope.spawn(move || {
                    let stop = AtomicBool::new(false);
                    for round in 0..rounds {
                        // The first rank asks to stop in the last round.
                        stop.store(round + 1 == rounds, Ordering::Relaxed);
                        let received = match rank.encode.intake.as_mut() {
                            Some(intake) => intake.receive(Duration::ZERO),
                            None => Vec::new(),
                        };
                        let round = rank
                            .encode
                            .exchange(stop.load(Ordering::Relaxed), received)
                            .unwrap();
                        if round.stop {
                            return;
                        }
                        rank.encode.admit(round.messages).unwrap();
                        rank.encode.batch(round.free).unwrap();
                        rank.encode.hand_over().unwrap();
                        rank.encode.reclaim();
                    }
                });
            }
        });
    }

    #[test]
    fn ranks_batch_alike_while_their_sends_conclude_apart() {
        // Two slots; rank 1's sends conclude three rounds after rank 0's do,
        // so rank 0 has slots free earlier but waits for rank 1.
        let rounds = vec![vec![
            request("a", 1, vec![vec![(0, 3)]]),
            request("b", 2, vec![vec![(10, 13)]]),
            request("c", 3, vec![vec![(20, 23)]]),
            request("d", 4, vec![vec![(30, 33)]]),
        ]];
        let mut ranks = ranks(2, &[0, 3], rounds);
        run(&mut ranks, 30);
        let batches: Vec<Vec<Vec<u64>>> = ranks
            .iter()
            .map(|rank| rank.log.lock().unwrap().batches.clone())
            .collect();
        assert_eq!(batches[0], batches[1]);
        // Two slots bound each forward.
        assert_eq!(batches[0], [vec![1, 2], vec![3, 4]]);
        for rank in &ranks {
            let sends = rank.sends.lock().unwrap();
            let rooms: Vec<u64> = sends.sent.iter().map(|chunk| chunk.room).collect();
            assert_eq!(rooms, [1, 2, 3, 4]);
            // Each item read from its own slot of the ring.
            let sources: HashSet<u64> = sends.sent[..2].iter().map(|chunk| chunk.source).collect();
            assert_eq!(sources.len(), 2);
            assert_eq!(rank.encode.free.len(), 2);
        }
    }

    #[test]
    fn forwards_keep_arrival_order_within_their_budgets_and_cache_hits_skip_ahead() {
        let rounds = vec![
            vec![
                // Six rows each: two would pass the ten-row budget.
                request("a", 1, vec![vec![(0, 5)]]),
                request("b", 2, vec![vec![(0, 5)]]),
                request("c", 3, vec![vec![(0, 1)]]),
            ],
            vec![request("hit", 1000, vec![vec![(0, 99)]])],
        ];
        let mut ranks = ranks(8, &[0], rounds);
        run(&mut ranks, 10);
        let log = ranks[0].log.lock().unwrap();
        // Round one: a alone (b would pass the budget); round two: the hit
        // first, then b and c.
        assert_eq!(log.batches, [vec![1], vec![4, 2, 3]]);
    }

    #[test]
    fn refused_and_failed_items_free_their_slots() {
        let rounds = vec![vec![
            // Two items in one request: refused, its room failed.
            request("two", 7, vec![vec![(0, 1)], vec![(2, 3)]]),
            // Does not fit its slot: failed after the forward.
            request("big", 13, vec![vec![(0, 1)]]),
            request("ok", 21, vec![vec![(0, 1)]]),
            // The same room again: embedded, not sent.
            request("again", 21, vec![vec![(0, 1)]]),
        ]];
        let mut ranks = ranks(4, &[0], rounds);
        run(&mut ranks, 10);
        let log = ranks[0].log.lock().unwrap();
        assert_eq!(log.admitted, [1, 2, 3, 4]);
        assert_eq!(log.released, [1]);
        assert_eq!(log.batches, [vec![2, 3, 4]]);
        let sends = ranks[0].sends.lock().unwrap();
        assert_eq!(sends.failed, [7, 13]);
        assert_eq!(
            sends
                .sent
                .iter()
                .map(|chunk| chunk.room)
                .collect::<Vec<_>>(),
            [21]
        );
        assert_eq!(ranks[0].encode.free.len(), 4);
        assert!(!ranks[0].encode.busy());
    }
}
