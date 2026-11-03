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

//! An encode rank's sender of the embeddings it computes.
//!
//! The rank registers with its node's bootstrap server; each prefill rank
//! that pulls from it registers once (no state is kept for that) and asks
//! for each item's rows ([`TransferInfo`]). One thread applies those
//! messages and expires what waited too long; workers write each ready
//! room's rows to every prefill rank of its block and push each a status.
//! The scheduling loop names a room when its request arrives
//! ([`EmbeddingSender::begin`]), hands over the item's rows once they sit in
//! their send slot ([`EmbeddingSender::send`]), and learns which rooms
//! concluded, whose slots it then reuses.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, MutexGuard,
    },
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use serde_json::json;

use super::{is_registration, status_frames, TransferInfo};
use crate::{
    control::{EmbeddingChunk, EmbeddingSender},
    http,
    status::TransferPoll,
    Copy, PeerConfig, StartError, Transport,
};

/// How often the message thread checks for shutdown and expiry.
const SOCKET_POLL: Duration = Duration::from_millis(100);

/// How long the rank keeps trying to register with the bootstrap server,
/// which rank 0 may still be starting.
const REGISTER_ATTEMPTS: usize = 60;
const REGISTER_RETRY: Duration = Duration::from_millis(500);

/// What an encode rank's sender starts from.
#[derive(Debug, Clone)]
pub struct SenderConfig {
    /// The transfer engine; its regions are the send slots.
    pub peer: PeerConfig,
    /// The bootstrap server's `/route`.
    pub bootstrap_url: String,
    /// The encode node's TP size, which the prefill ranks' block size
    /// derives from.
    pub tp_size: i64,
    /// How long a handed-over item waits for every prefill rank of its block
    /// to ask, and a room's requests for the item's arrival.
    pub bootstrap_timeout: Duration,
    /// Rooms written concurrently.
    pub transfer_workers: usize,
}

/// The rows of `chunk` one prefill rank asked for: `(source, bytes,
/// deepstack source, deepstack bytes)`. Row strides follow from the chunk's
/// byte counts; a full request reproduces the whole chunk.
pub fn shard_payload(chunk: &EmbeddingChunk, info: &TransferInfo) -> (u64, u64, u64, u64) {
    let rows = chunk.rows.max(1) as u64;
    let row_bytes = chunk.nbytes / rows;
    let start = info.row_start.max(0) as u64;
    let count = info.rows.max(0) as u64;
    let (mut deep_source, mut deep_bytes) = (0, 0);
    if chunk.deepstack_width > 0 && chunk.deepstack_nbytes > 0 {
        let deep_row_bytes = chunk.deepstack_nbytes / rows;
        deep_source = chunk.deepstack_source + start * deep_row_bytes;
        deep_bytes = count * deep_row_bytes;
    }
    (
        chunk.source + start * row_bytes,
        count * row_bytes,
        deep_source,
        deep_bytes,
    )
}

/// Check a room's full set of requests against its chunk before anything is
/// written (the write itself is unchecked); the reason it fails, if it does.
///
/// Each request must match the chunk's width and dtype, name the image's
/// row count as its span (the token-count check of both ends), stay inside
/// the chunk's rows, and agree on deepstack. The set is either all full
/// copies or shards that tile a contiguous range without gaps or overlaps
/// (an encode rank serves a contiguous block of the prefill ranks' shards).
pub fn validate_fanout(infos: &[TransferInfo], chunk: &EmbeddingChunk) -> Option<String> {
    for info in infos {
        if info.hidden != chunk.hidden || info.dtype != chunk.dtype {
            return Some("embedding shape/dtype contract violated".to_owned());
        }
        if info.span != chunk.rows {
            return Some(format!(
                "image token-count contract violated: receiver expects {} rows, encode has {}",
                info.span, chunk.rows
            ));
        }
        if info.row_start < 0 || info.rows < 0 || info.row_start + info.rows > chunk.rows {
            return Some(format!(
                "embedding shard out of range: rows [{}, {}) of {}",
                info.row_start,
                info.row_start + info.rows,
                chunk.rows
            ));
        }
        if info.has_deepstack && chunk.deepstack_width == 0 {
            return Some(
                "receiver expects deepstack but the chunk carries none (encode-side cache hit \
                 without its deepstack half?)"
                    .to_owned(),
            );
        }
        if chunk.deepstack_width > 0 && !info.has_deepstack {
            return Some(
                "chunk carries deepstack but the receiver did not allocate for it".to_owned(),
            );
        }
    }
    if infos
        .iter()
        .all(|info| info.row_start == 0 && info.rows == chunk.rows)
    {
        return None;
    }
    let mut shards: Vec<(i64, i64)> = infos
        .iter()
        .filter(|info| info.rows > 0)
        .map(|info| (info.row_start, info.rows))
        .collect();
    shards.sort_unstable();
    for pair in shards.windows(2) {
        let ((start, count), (next, _)) = (pair[0], pair[1]);
        if start + count != next {
            return Some(format!(
                "embedding shard frames do not tile contiguously: [{start}, {}) then [{next}, ...)",
                start + count
            ));
        }
    }
    None
}

/// A room ready to write: its rows and every prefill rank's request.
type Ready = (EmbeddingChunk, Vec<TransferInfo>);

/// One room as the sender tracks it.
#[derive(Debug)]
struct Room {
    /// The prefill ranks' requests, by transfer session.
    infos: HashMap<String, TransferInfo>,
    /// When the room appeared: an orphan's (requests without a request)
    /// expiry counts from here.
    since: Instant,
    /// The scheduling loop named it: its request arrived.
    begun: bool,
    /// Its chunk, handed over and waiting for the rest of the fanout, and
    /// how long it may wait.
    parked: Option<(EmbeddingChunk, Instant)>,
    /// A worker writes it.
    sending: bool,
}

impl Room {
    fn new(now: Instant) -> Self {
        Self {
            infos: HashMap::new(),
            since: now,
            begun: false,
            parked: None,
            sending: false,
        }
    }

    /// Every prefill rank of the block asked.
    fn complete(&self) -> bool {
        self.infos
            .values()
            .map(|info| info.fanout)
            .max()
            .is_some_and(|fanout| self.infos.len() as i64 >= fanout)
    }
}

#[derive(Default)]
struct State {
    rooms: HashMap<u64, Room>,
    /// Failed rooms and until when a late request gets the failure at once.
    tombstones: HashMap<u64, Instant>,
    /// Rooms that concluded since the scheduling loop last asked.
    concluded: Vec<u64>,
}

struct Shared {
    transport: Arc<Transport>,
    engine_rank: i64,
    bootstrap_timeout: Duration,
    state: Mutex<State>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("the EPD sender lock is poisoned")
    }

    fn push(&self, info: &TransferInfo, status: TransferPoll) {
        self.transport.sockets.send(
            &info.endpoint,
            info.port,
            status_frames(info.room, status, self.engine_rank),
        );
    }

    /// Conclude `room` failed, telling every prefill rank that asked.
    fn fail(&self, state: &mut State, room: u64, reason: &str) {
        let entry = state.rooms.remove(&room);
        tracing::error!(room, reason, "EPD embedding send failed");
        if let Some(entry) = &entry {
            for info in entry.infos.values() {
                self.push(info, TransferPoll::Failed);
            }
        }
        state
            .tombstones
            .insert(room, Instant::now() + self.bootstrap_timeout);
        if entry.is_some_and(|entry| entry.begun) {
            state.concluded.push(room);
        }
    }

    /// A prefill rank's message.
    fn handle(&self, frames: &[Bytes], queues: &[mpsc::Sender<Ready>]) {
        // A registration needs no state: the rank's requests carry where to
        // push its statuses.
        if is_registration(frames) {
            return;
        }
        let info = match TransferInfo::from_frames(frames) {
            Ok(info) => info,
            Err(error) => {
                tracing::error!(%error, "dropping a malformed EPD transfer request");
                return;
            }
        };
        let mut state = self.state();
        if state.tombstones.contains_key(&info.room) {
            self.push(&info, TransferPoll::Failed);
            return;
        }
        let room = info.room;
        let entry = state
            .rooms
            .entry(room)
            .or_insert_with(|| Room::new(Instant::now()));
        entry.infos.insert(info.session_id.clone(), info);
        if entry.complete() {
            if let Some((chunk, _)) = entry.parked.take() {
                entry.sending = true;
                let infos = entry.infos.values().cloned().collect();
                dispatch(queues, chunk, infos);
            }
        }
    }

    /// Fail what waited too long: a parked chunk whose fanout never
    /// completed, and drop the requests of a room no request arrived for.
    fn expire(&self) {
        let now = Instant::now();
        let mut state = self.state();
        state.tombstones.retain(|_, until| *until > now);
        let parked: Vec<u64> = state
            .rooms
            .iter()
            .filter(|(_, entry)| {
                entry
                    .parked
                    .as_ref()
                    .is_some_and(|(_, until)| *until <= now)
            })
            .map(|(&room, _)| room)
            .collect();
        for room in parked {
            let reason = format!(
                "the prefill ranks did not all ask within {:?}",
                self.bootstrap_timeout
            );
            self.fail(&mut state, room, &reason);
        }
        let timeout = self.bootstrap_timeout;
        state.rooms.retain(|&room, entry| {
            let orphan = !entry.begun && now.duration_since(entry.since) > timeout;
            if orphan {
                tracing::warn!(
                    room,
                    "dropping EPD transfer requests no encode request claimed"
                );
            }
            !orphan
        });
    }

    /// Write one room's rows to every prefill rank that asked, then
    /// conclude it.
    fn transfer(&self, chunk: &EmbeddingChunk, infos: &[TransferInfo]) -> Result<(), String> {
        if let Some(reason) = validate_fanout(infos, chunk) {
            return Err(reason);
        }
        let write = |info: &TransferInfo| -> Result<(), String> {
            let (source, bytes, deep_source, deep_bytes) = shard_payload(chunk, info);
            if bytes == 0 {
                // An empty shard: its request only counted toward the
                // fanout.
                return Ok(());
            }
            let mut sources = vec![source];
            let mut destinations = vec![info.embedding_ptr];
            let mut lengths = vec![bytes];
            if deep_bytes > 0 && info.deepstack_ptr != 0 {
                sources.push(deep_source);
                destinations.push(info.deepstack_ptr);
                lengths.push(deep_bytes);
            }
            self.transport
                .write(
                    &info.session_id,
                    &[Copy::Spans {
                        sources,
                        destinations,
                        lengths,
                    }],
                )
                .map_err(|error| {
                    format!("Mooncake transfer to {} failed: {error}", info.session_id)
                })
        };
        // Each prefill rank is its own session: write them concurrently.
        let written: Vec<Result<(), String>> = if infos.len() == 1 {
            vec![write(&infos[0])]
        } else {
            thread::scope(|scope| {
                let handles: Vec<_> = infos
                    .iter()
                    .map(|info| scope.spawn(move || write(info)))
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|_| Err("a write panicked".to_owned()))
                    })
                    .collect()
            })
        };
        if let Some(error) = written.into_iter().find_map(Result::err) {
            return Err(error);
        }
        for info in infos {
            self.push(info, TransferPoll::Success);
        }
        Ok(())
    }

    fn run_worker(&self, queue: &mpsc::Receiver<Ready>) {
        while let Ok((chunk, infos)) = queue.recv() {
            let sent = self.transfer(&chunk, &infos);
            let mut state = self.state();
            match sent {
                Ok(()) => {
                    state.rooms.remove(&chunk.room);
                    state.concluded.push(chunk.room);
                }
                Err(reason) => self.fail(&mut state, chunk.room, &reason),
            }
        }
    }
}

fn dispatch(queues: &[mpsc::Sender<Ready>], chunk: EmbeddingChunk, infos: Vec<TransferInfo>) {
    // One worker per room keeps a room's writes in one place.
    let queue = &queues[(chunk.room % queues.len() as u64) as usize];
    if queue.send((chunk, infos)).is_err() {
        tracing::error!("the EPD transfer workers stopped");
    }
}

/// An encode rank's sender of the embeddings it computes.
pub struct Sender {
    shared: Arc<Shared>,
    queues: Arc<Vec<mpsc::Sender<Ready>>>,
    stop: Arc<AtomicBool>,
}

impl Sender {
    /// Start the transfer engine with the send slots registered, open the
    /// control socket, start the threads and register with the bootstrap
    /// server.
    pub fn start(config: SenderConfig) -> Result<Self, StartError> {
        let (incoming, messages) = mpsc::channel::<Vec<Bytes>>();
        let transport = Transport::start(&config.peer, incoming)?;
        let port = transport.sockets.port();
        let shared = Arc::new(Shared {
            transport,
            engine_rank: config.peer.engine_rank,
            bootstrap_timeout: config.bootstrap_timeout,
            state: Mutex::new(State::default()),
        });
        let mut queues = Vec::with_capacity(config.transfer_workers.max(1));
        for index in 0..config.transfer_workers.max(1) {
            let (queue, chunks) = mpsc::channel();
            let shared = shared.clone();
            thread::Builder::new()
                .name(format!("ts-epd-send-{index}"))
                .spawn(move || shared.run_worker(&chunks))?;
            queues.push(queue);
        }
        let queues = Arc::new(queues);
        let stop = Arc::new(AtomicBool::new(false));
        {
            let shared = shared.clone();
            let queues = queues.clone();
            let stop = stop.clone();
            thread::Builder::new()
                .name("ts-epd-requests".to_owned())
                .spawn(move || {
                    let mut swept = Instant::now();
                    loop {
                        match messages.recv_timeout(SOCKET_POLL) {
                            Ok(frames) => shared.handle(&frames, &queues),
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if stop.load(Ordering::Relaxed) {
                                    return;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        }
                        if swept.elapsed() >= SOCKET_POLL {
                            swept = Instant::now();
                            shared.expire();
                        }
                    }
                })?;
        }
        let payload = json!({
            // The bootstrap server's role for a node others read from.
            "role": "Prefill",
            "world_size": config.tp_size,
            "dp_size": 1,
            "rank_ip": config.peer.local_ip,
            "rank_port": port,
            "engine_rank": config.peer.engine_rank,
        });
        let mut registered = Err(String::new());
        for _ in 0..REGISTER_ATTEMPTS {
            registered = http::put_json(&config.bootstrap_url, &payload, Duration::from_secs(5));
            if registered.is_ok() {
                break;
            }
            thread::sleep(REGISTER_RETRY);
        }
        if let Err(error) = registered {
            // Without the registration no prefill rank finds this one.
            tracing::error!(
                url = config.bootstrap_url,
                error,
                "the encode rank failed to register with the bootstrap server"
            );
        }
        tracing::info!(
            port,
            session = shared.transport.session_id(),
            "EPD embedding sender up"
        );
        Ok(Self {
            shared,
            queues,
            stop,
        })
    }

    /// The control socket's port.
    pub fn port(&self) -> u16 {
        self.shared.transport.sockets.port()
    }
}

impl Drop for Sender {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl EmbeddingSender for Sender {
    fn begin(&mut self, room: u64) -> bool {
        let mut state = self.shared.state();
        let now = Instant::now();
        let entry = state.rooms.entry(room).or_insert_with(|| Room::new(now));
        if entry.begun {
            return false;
        }
        entry.begun = true;
        true
    }

    fn send(&mut self, chunk: EmbeddingChunk) {
        let mut state = self.shared.state();
        let Some(entry) = state.rooms.get_mut(&chunk.room) else {
            // Failed while it was being encoded.
            return;
        };
        if entry.complete() {
            entry.sending = true;
            let infos = entry.infos.values().cloned().collect();
            dispatch(&self.queues, chunk, infos);
        } else {
            let until = Instant::now() + self.shared.bootstrap_timeout;
            entry.parked = Some((chunk, until));
        }
    }

    fn fail(&mut self, room: u64, reason: &str) {
        let mut state = self.shared.state();
        if state.rooms.get(&room).is_some_and(|entry| entry.sending) {
            // A worker writes it already and concludes it either way.
            return;
        }
        self.shared.fail(&mut state, room, reason);
    }

    fn concluded(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.shared.state().concluded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(rows: i64, deepstack_width: i64) -> EmbeddingChunk {
        EmbeddingChunk {
            room: 7,
            source: 0x10_0000,
            rows,
            hidden: 64,
            dtype: "torch.bfloat16".to_owned(),
            nbytes: (rows * 64 * 2) as u64,
            deepstack_source: if deepstack_width > 0 { 0x20_0000 } else { 0 },
            deepstack_width,
            deepstack_nbytes: (rows * deepstack_width * 2) as u64,
        }
    }

    fn info(rows: i64, row_start: i64, span: i64) -> TransferInfo {
        TransferInfo {
            room: 7,
            endpoint: "1.2.3.4".to_owned(),
            port: 5555,
            session_id: format!("s{row_start}"),
            embedding_ptr: 0x1000,
            deepstack_ptr: 0,
            rows,
            hidden: 64,
            dtype: "torch.bfloat16".to_owned(),
            has_deepstack: false,
            fanout: 4,
            row_start,
            span,
        }
    }

    #[test]
    fn a_full_request_takes_the_whole_chunk_and_a_shard_its_rows() {
        let deep = chunk(10, 128);
        let full = TransferInfo {
            has_deepstack: true,
            ..info(10, 0, 10)
        };
        assert_eq!(
            shard_payload(&deep, &full),
            (
                deep.source,
                deep.nbytes,
                deep.deepstack_source,
                deep.deepstack_nbytes
            )
        );
        let shard = TransferInfo {
            has_deepstack: true,
            ..info(3, 5, 10)
        };
        let (row, deep_row) = (deep.nbytes / 10, deep.deepstack_nbytes / 10);
        assert_eq!(
            shard_payload(&deep, &shard),
            (
                deep.source + 5 * row,
                3 * row,
                deep.deepstack_source + 5 * deep_row,
                3 * deep_row
            )
        );
    }

    #[test]
    fn a_fanout_that_breaks_the_contract_is_refused() {
        let plain = chunk(10, 0);
        let gap = [info(3, 0, 10), info(3, 5, 10)];
        let overlap = [info(5, 0, 10), info(5, 3, 10)];
        let out_of_range = [info(6, 5, 10)];
        let bad_dtype = [TransferInfo {
            dtype: "torch.float32".to_owned(),
            ..info(10, 0, 10)
        }];
        let mixed = [info(10, 0, 10), info(5, 5, 10)];
        // The token-count check holds even for a lone full request.
        let short_span = [info(6, 0, 6)];
        for infos in [
            &gap[..],
            &overlap,
            &out_of_range,
            &bad_dtype,
            &mixed,
            &short_span,
        ] {
            assert!(validate_fanout(infos, &plain).is_some(), "{infos:?}");
        }
        assert_eq!(validate_fanout(&[info(10, 0, 10)], &plain), None);
        assert_eq!(
            validate_fanout(
                &[
                    info(4, 0, 10),
                    info(3, 4, 10),
                    info(3, 7, 10),
                    info(0, 10, 10)
                ],
                &plain
            ),
            None
        );
        // Deepstack must agree on both ends.
        let deep = chunk(10, 128);
        let wants_deep = [TransferInfo {
            has_deepstack: true,
            deepstack_ptr: 0x2000,
            ..info(10, 0, 10)
        }];
        assert!(validate_fanout(&wants_deep, &plain).is_some());
        assert!(validate_fanout(&[info(10, 0, 10)], &deep).is_some());
        assert_eq!(validate_fanout(&wants_deep, &deep), None);
    }
}
