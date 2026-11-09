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

//! KV cache event publishing (`pd/kv_events.py`): each round's prefix-cache
//! mutations, drained from the scheduler once per round as one encoded
//! frame, go out as one `KVEventBatch` on a ZMQ PUB socket, with an optional
//! ROUTER socket that replays recent batches to a subscriber that missed
//! them.
//!
//! The scheduler encodes the frame (`Scheduler::drain_kv_event_frame`), so
//! the control thread copies bytes into a bounded ring and never encodes,
//! allocates or waits on a socket; the publisher thread sends. A ring that
//! is full overwrites its oldest frame: the sequence number that frame was
//! given at push time is never sent, the relay reads the gap and asks the
//! replay, and what the replay no longer holds it resyncs from a snapshot.
//!
//! The wire, unchanged from the Python publisher's: `[topic, seq, payload]`
//! frames, the sequence number 8 bytes big-endian, the payload the msgpack
//! array of the batch; a replay request is `[client, "", start_seq]`,
//! answered with every buffered batch from `start_seq` and an end marker of
//! sequence -1.

use std::{
    collections::VecDeque,
    sync::{mpsc as std_mpsc, Arc, Mutex, PoisonError},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use tokio::sync::Notify;
use zeromq::{PubSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

/// How long shutdown waits for queued batches to go out
/// (`ZmqEventPublisher.SHUTDOWN_TIMEOUT`).
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);

/// The replay stream's end marker.
const END_SEQ: [u8; 8] = (-1i64).to_be_bytes();

/// A publisher that could not start.
#[derive(Debug, thiserror::Error)]
pub enum KvEventsError {
    /// A socket failed to bind or connect.
    #[error("KV event publisher: {0}")]
    Zmq(#[from] ZmqError),
    /// The publisher thread did not start.
    #[error("KV event publisher: {0}")]
    Io(#[from] std::io::Error),
    /// The endpoint is not one this publisher serves.
    #[error("KV event publisher: unsupported endpoint {0}")]
    Endpoint(String),
}

/// `KVEventsConfig`'s publishing fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvEventsConfig {
    /// The PUB address: bound when it is a wildcard or IPC address, else
    /// connected. With attention DP each rank adds its rank to the port.
    pub endpoint: String,
    /// The ROUTER address replay requests arrive on, if any.
    pub replay_endpoint: Option<String>,
    /// Batches kept for replay.
    pub buffer_steps: usize,
    /// Frames the ring holds for the publisher thread; when it is full, the
    /// oldest frame is overwritten and its sequence number skipped.
    pub max_queue_size: usize,
    /// The topic frame every batch carries.
    pub topic: String,
}

/// `offset_endpoint_port`: a DP rank's endpoint, its TCP port moved by the
/// rank.
pub fn offset_endpoint_port(endpoint: &str, dp_rank: i64) -> Result<String, KvEventsError> {
    if dp_rank == 0 {
        return Ok(endpoint.to_owned());
    }
    if !endpoint.contains("tcp") {
        return Err(KvEventsError::Endpoint(endpoint.to_owned()));
    }
    match endpoint.rsplit_once(':') {
        Some((base, port)) => {
            let port: i64 = port
                .parse()
                .map_err(|_| KvEventsError::Endpoint(endpoint.to_owned()))?;
            Ok(format!("{base}:{}", port + dp_rank))
        }
        None => Ok(endpoint.to_owned()),
    }
}

/// The batch timestamp the frame carries: seconds since the epoch.
pub fn now_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// Bind addresses as libzmq reads them: `*` is every interface.
fn bind_address(endpoint: &str) -> String {
    endpoint.replace("://*:", "://0.0.0.0:")
}

/// The frames waiting for the publisher thread, oldest first, each with the
/// sequence number it was given when pushed. Fixed capacity: the slots keep
/// their buffers, so a push copies and allocates nothing once every slot
/// has grown to the frames it sees; a push into a full ring drops the oldest
/// frame.
struct Ring {
    slots: Vec<(u64, Vec<u8>)>,
    head: usize,
    len: usize,
    next_seq: u64,
    dropped: u64,
    closed: bool,
}

impl Ring {
    fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity.max(1)).map(|_| (0, Vec::new())).collect(),
            head: 0,
            len: 0,
            next_seq: 0,
            dropped: 0,
            closed: false,
        }
    }

    fn push(&mut self, frame: &[u8]) -> u64 {
        let capacity = self.slots.len();
        if self.len == capacity {
            self.head = (self.head + 1) % capacity;
            self.len -= 1;
            self.dropped += 1;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let (slot_seq, slot) = &mut self.slots[(self.head + self.len) % capacity];
        *slot_seq = seq;
        slot.clear();
        slot.extend_from_slice(frame);
        self.len += 1;
        seq
    }

    /// Copies the oldest frame into `out` and returns its sequence number.
    fn pop_into(&mut self, out: &mut Vec<u8>) -> Option<u64> {
        if self.len == 0 {
            return None;
        }
        let (seq, frame) = &self.slots[self.head];
        out.clear();
        out.extend_from_slice(frame);
        let seq = *seq;
        self.head = (self.head + 1) % self.slots.len();
        self.len -= 1;
        Some(seq)
    }
}

struct Shared {
    ring: Mutex<Ring>,
    wake: Notify,
}

impl Shared {
    fn ring(&self) -> std::sync::MutexGuard<'_, Ring> {
        self.ring.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One rank's publisher; dropping it sends what is queued, for up to a
/// second.
pub struct KvEventPublisher {
    shared: Arc<Shared>,
    dp_rank: i64,
    done: std_mpsc::Receiver<()>,
}

impl KvEventPublisher {
    /// Start publishing for attention-DP rank `dp_rank`.
    pub fn start(config: &KvEventsConfig, dp_rank: i64) -> Result<Self, KvEventsError> {
        let endpoint = offset_endpoint_port(&config.endpoint, dp_rank)?;
        let replay_endpoint = config
            .replay_endpoint
            .as_deref()
            .map(|endpoint| offset_endpoint_port(endpoint, dp_rank))
            .transpose()?;
        if endpoint.starts_with("inproc://") {
            return Err(KvEventsError::Endpoint(endpoint));
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let mut publisher = PubSocket::new();
        let binds =
            endpoint.contains('*') || endpoint.contains("::") || endpoint.starts_with("ipc://");
        runtime.block_on(async {
            if binds {
                publisher.bind(&bind_address(&endpoint)).await.map(drop)
            } else {
                publisher.connect(&endpoint).await
            }
        })?;
        let replay = match &replay_endpoint {
            Some(address) => {
                let mut router = RouterSocket::new();
                runtime.block_on(router.bind(&bind_address(address)))?;
                Some(router)
            }
            None => None,
        };
        tracing::info!(endpoint, replay = ?replay_endpoint, "publishing KV cache events");
        let shared = Arc::new(Shared {
            ring: Mutex::new(Ring::new(config.max_queue_size)),
            wake: Notify::new(),
        });
        let (done_tx, done) = std_mpsc::channel();
        let state = PublisherState {
            publisher,
            replay,
            topic: Bytes::from(config.topic.clone().into_bytes()),
            buffer: VecDeque::new(),
            buffer_steps: config.buffer_steps,
            shared: Arc::clone(&shared),
            scratch: Vec::new(),
        };
        thread::Builder::new()
            .name("ts-kv-events".to_owned())
            .spawn(move || {
                runtime.block_on(state.run());
                let _ = done_tx.send(());
            })?;
        Ok(Self {
            shared,
            dp_rank,
            done,
        })
    }

    /// The attention-DP rank the frames name.
    pub fn dp_rank(&self) -> i64 {
        self.dp_rank
    }

    /// Hands one round's frame to the publisher thread. Never waits: a full
    /// ring drops its oldest frame, whose skipped sequence number tells the
    /// relay to resync.
    pub fn publish(&self, frame: &[u8]) {
        if frame.is_empty() {
            return;
        }
        let dropped = {
            let mut ring = self.shared.ring();
            let before = ring.dropped;
            ring.push(frame);
            (ring.dropped != before).then_some(ring.dropped)
        };
        self.shared.wake.notify_one();
        if let Some(dropped) = dropped {
            if dropped == 1 || dropped % 1000 == 0 {
                tracing::warn!(
                    dropped,
                    "the KV event publisher fell behind; the oldest queued batch was dropped"
                );
            }
        }
    }
}

impl Drop for KvEventPublisher {
    fn drop(&mut self) {
        // The thread sends what is queued once the ring closes.
        self.shared.ring().closed = true;
        self.shared.wake.notify_one();
        if self.done.recv_timeout(SHUTDOWN_TIMEOUT).is_err() {
            tracing::warn!("KV event batches still queued at shutdown");
        }
    }
}

struct PublisherState {
    publisher: PubSocket,
    replay: Option<RouterSocket>,
    topic: Bytes,
    buffer: VecDeque<(u64, Bytes)>,
    buffer_steps: usize,
    shared: Arc<Shared>,
    scratch: Vec<u8>,
}

impl PublisherState {
    async fn run(mut self) {
        loop {
            loop {
                let seq = self.shared.ring().pop_into(&mut self.scratch);
                match seq {
                    Some(seq) => {
                        let payload = Bytes::copy_from_slice(&self.scratch);
                        self.send(seq, payload).await;
                    }
                    None => break,
                }
            }
            if self.shared.ring().closed {
                return;
            }
            let wake = &self.shared.wake;
            let replay = &mut self.replay;
            tokio::select! {
                () = wake.notified() => {}
                request = recv_replay(replay) => match request {
                    Ok(request) => self.serve_replay(request).await,
                    Err(error) => tracing::warn!(%error, "KV event replay socket failed"),
                },
            }
        }
    }

    async fn send(&mut self, seq: u64, payload: Bytes) {
        let mut message = ZmqMessage::from(self.topic.clone());
        message.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
        message.push_back(payload.clone());
        if let Err(error) = self.publisher.send(message).await {
            tracing::warn!(%error, "publishing a KV event batch failed");
        }
        self.buffer.push_back((seq, payload));
        while self.buffer.len() > self.buffer_steps {
            self.buffer.pop_front();
        }
    }

    async fn serve_replay(&mut self, request: ZmqMessage) {
        let frames = request.into_vec();
        let [client, _, start] = frames.as_slice() else {
            tracing::warn!(frames = frames.len(), "invalid KV event replay request");
            return;
        };
        let mut start_seq = [0u8; 8];
        let tail = start.len().min(8);
        start_seq[8 - tail..].copy_from_slice(&start[start.len() - tail..]);
        let start_seq = u64::from_be_bytes(start_seq);
        let Some(router) = self.replay.as_mut() else {
            return;
        };
        let reply = |seq: Bytes, payload: Bytes| {
            let mut message = ZmqMessage::from(client.clone());
            message.push_back(Bytes::new());
            message.push_back(seq);
            message.push_back(payload);
            message
        };
        for (seq, payload) in &self.buffer {
            if *seq >= start_seq {
                let message = reply(Bytes::copy_from_slice(&seq.to_be_bytes()), payload.clone());
                if let Err(error) = router.send(message).await {
                    tracing::warn!(%error, "replaying KV event batches failed");
                    return;
                }
            }
        }
        let _ = router
            .send(reply(Bytes::copy_from_slice(&END_SEQ), Bytes::new()))
            .await;
    }
}

async fn recv_replay(replay: &mut Option<RouterSocket>) -> Result<ZmqMessage, ZmqError> {
    match replay {
        Some(router) => router.recv().await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dp_rank_moves_the_tcp_port() {
        assert_eq!(
            offset_endpoint_port("tcp://*:5557", 0).unwrap(),
            "tcp://*:5557"
        );
        assert_eq!(
            offset_endpoint_port("tcp://*:5557", 2).unwrap(),
            "tcp://*:5559"
        );
        assert!(offset_endpoint_port("ipc:///tmp/x", 1).is_err());
    }

    #[test]
    fn a_full_ring_drops_its_oldest_frame_and_keeps_the_sequence_numbers() {
        let mut ring = Ring::new(2);
        assert_eq!(ring.push(b"a"), 0);
        assert_eq!(ring.push(b"b"), 1);
        assert_eq!(ring.push(b"c"), 2);
        assert_eq!(ring.dropped, 1);
        let mut out = Vec::new();
        assert_eq!(ring.pop_into(&mut out), Some(1));
        assert_eq!(out, b"b");
        assert_eq!(ring.pop_into(&mut out), Some(2));
        assert_eq!(out, b"c");
        assert_eq!(ring.pop_into(&mut out), None);
        // The slots keep their buffers: a later push reuses them.
        let capacity_before: Vec<usize> =
            ring.slots.iter().map(|(_, slot)| slot.capacity()).collect();
        assert_eq!(ring.push(b"d"), 3);
        let capacity_after: Vec<usize> =
            ring.slots.iter().map(|(_, slot)| slot.capacity()).collect();
        assert_eq!(capacity_before, capacity_after);
    }
}
