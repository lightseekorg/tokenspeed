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
//! mutations, drained from the scheduler once per round, go out as one
//! `KVEventBatch` on a ZMQ PUB socket, with an optional ROUTER socket that
//! replays recent batches to a subscriber that missed them.
//!
//! The wire, unchanged from the Python publisher's: `[topic, seq, payload]`
//! frames, the sequence number 8 bytes big-endian, the payload the msgpack
//! array of the batch; a replay request is `[client, "", start_seq]`,
//! answered with every buffered batch from `start_seq` and an end marker of
//! sequence -1.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::mpsc as std_mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use ts_scheduler::{profile, KvEvent, KvEventBatch, KvEventMedium};
use zeromq::{PubSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

mod stats;
use stats::PendingBatch;
pub use stats::{KvEventProgress, KvEventStats};

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
    /// An environment-configured KV event knob is malformed or out of range.
    #[error("KV event publisher configuration: {0}")]
    Config(String),
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
    /// Batches queued for the publisher thread before `publish` waits.
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

/// The medium of a Host event, as vLLM's events name the pinned Host cache.
const MEDIUM_HOST: &str = "CPU_PINNED";
/// The medium of a Disk tier event, as routers name local disk.
const MEDIUM_DISK: &str = "DISK";

/// Default maximum blocks merged into one wire `BlockStored` or `BlockRemoved`
/// event (aligned with Dynamo's `DEFAULT_MAX_BATCH_BLOCKS = 128`).
///
/// Override via `TOKENSPEED_KV_EVENT_COALESCE_MAX_BLOCKS` (`0` or `1` disables
/// coalescing; valid range `0..=1024`).
pub const DEFAULT_KV_EVENT_COALESCE_MAX_BLOCKS: usize = 128;
const MAX_KV_EVENT_COALESCE_MAX_BLOCKS: usize = 1024;

fn parse_kv_event_coalesce_max_blocks_value(raw: &str) -> Result<usize, String> {
    let parsed = raw.trim().parse::<usize>().map_err(|_| {
        format!(
            "TOKENSPEED_KV_EVENT_COALESCE_MAX_BLOCKS must be an integer in 0..={MAX_KV_EVENT_COALESCE_MAX_BLOCKS} (0 or 1 disables coalescing), got {raw:?}"
        )
    })?;
    if parsed > MAX_KV_EVENT_COALESCE_MAX_BLOCKS {
        return Err(format!(
            "TOKENSPEED_KV_EVENT_COALESCE_MAX_BLOCKS must be an integer in 0..={MAX_KV_EVENT_COALESCE_MAX_BLOCKS} (0 or 1 disables coalescing), got {parsed}"
        ));
    }
    Ok(parsed.max(1))
}

fn kv_event_coalesce_max_blocks_from_env() -> Result<usize, KvEventsError> {
    match std::env::var("TOKENSPEED_KV_EVENT_COALESCE_MAX_BLOCKS") {
        Ok(raw) => parse_kv_event_coalesce_max_blocks_value(&raw).map_err(KvEventsError::Config),
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_KV_EVENT_COALESCE_MAX_BLOCKS),
        Err(error) => Err(KvEventsError::Config(format!(
            "TOKENSPEED_KV_EVENT_COALESCE_MAX_BLOCKS: {error}"
        ))),
    }
}

fn encode_medium(encoder: &mut ts_wire::msgpack::Encoder, medium: KvEventMedium) {
    // vLLM's layout: nil is the GPU cache.
    match medium {
        KvEventMedium::Device => encoder.nil(),
        KvEventMedium::Host => encoder.str(MEDIUM_HOST),
        KvEventMedium::Disk => encoder.str(MEDIUM_DISK),
    }
}

fn medium_slot(medium: KvEventMedium) -> usize {
    match medium {
        KvEventMedium::Device => 0,
        KvEventMedium::Host => 1,
        KvEventMedium::Disk => 2,
    }
}

enum WireEvent<'a> {
    Removed {
        medium: KvEventMedium,
        hashes: Vec<u64>,
    },
    Stored {
        medium: KvEventMedium,
        parent: Option<u64>,
        block_size: i32,
        hashes: Vec<u64>,
        token_slices: Vec<&'a [i32]>,
        total_tokens: usize,
    },
}

struct StoredChain<'a> {
    parent: Option<u64>,
    block_size: i32,
    hashes: Vec<u64>,
    token_slices: Vec<&'a [i32]>,
    total_tokens: usize,
    seen: HashSet<u64>,
}

fn has_unique_store_hashes(parent: Option<u64>, block_hashes: &[u64]) -> bool {
    let mut seen = HashSet::with_capacity(block_hashes.len() + usize::from(parent.is_some()));
    if let Some(p) = parent {
        seen.insert(p);
    }
    block_hashes.iter().all(|&h| seen.insert(h))
}

fn is_coalesceable_store(event: &KvEvent<'_>) -> bool {
    let Ok(block_size) = usize::try_from(event.block_size) else {
        return false;
    };
    if block_size == 0 || event.block_hashes.is_empty() {
        return false;
    }
    if event.block_hashes.len().checked_mul(block_size) != Some(event.token_ids.len()) {
        return false;
    }
    let parent = event.has_parent.then_some(event.parent_block_hash);
    has_unique_store_hashes(parent, event.block_hashes)
}

/// Coalesces one round's KV cache mutations into bounded multi-block wire
/// events (`<= max_blocks` blocks each) without introducing any batching delay.
///
/// Strict invariants preserved:
/// - Never reorders across contiguous `Removed` / `Stored` phase boundaries.
/// - Within a contiguous `Removed` phase, groups by medium in `Disk -> Host -> Device`
///   order while preserving exact per-medium leaf-first order, chunked to `max_blocks`.
/// - Within a contiguous `Stored` phase, groups by medium in `Device -> Host -> Disk`
///   order and merges only strictly linear `parent -> child` chains with identical
///   positive `block_size`, exact `block_hashes.len() * block_size == token_ids.len()`,
///   and zero duplicate/self-referencing hashes (branching and short tail blocks start
///   or remain separate events, preserving topological parent-first order).
fn coalesce_wire_events<'a>(raw: &[KvEvent<'a>], max_blocks: usize) -> Vec<WireEvent<'a>> {
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < raw.len() {
        let is_stored = raw[idx].stored;
        let seg_start = idx;
        while idx < raw.len() && raw[idx].stored == is_stored {
            idx += 1;
        }
        let segment = &raw[seg_start..idx];

        if !is_stored {
            let mut by_medium: [Vec<u64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            let mut empty_media: [usize; 3] = [0; 3];
            for ev in segment {
                let slot = medium_slot(ev.medium);
                if ev.block_hashes.is_empty() {
                    empty_media[slot] += 1;
                } else {
                    by_medium[slot].extend_from_slice(ev.block_hashes);
                }
            }
            for medium in [
                KvEventMedium::Disk,
                KvEventMedium::Host,
                KvEventMedium::Device,
            ] {
                let slot = medium_slot(medium);
                for _ in 0..empty_media[slot] {
                    out.push(WireEvent::Removed {
                        medium,
                        hashes: Vec::new(),
                    });
                }
                for chunk in by_medium[slot].chunks(max_blocks) {
                    out.push(WireEvent::Removed {
                        medium,
                        hashes: chunk.to_vec(),
                    });
                }
            }
            continue;
        }

        let mut by_medium: [Vec<&KvEvent<'a>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for ev in segment {
            by_medium[medium_slot(ev.medium)].push(ev);
        }
        for medium in [
            KvEventMedium::Device,
            KvEventMedium::Host,
            KvEventMedium::Disk,
        ] {
            let events_in_medium = &by_medium[medium_slot(medium)];
            if events_in_medium.is_empty() {
                continue;
            }
            let mut chains: Vec<StoredChain<'a>> = Vec::new();
            let mut open_tail: HashMap<u64, usize> = HashMap::new();

            for &&ev in events_in_medium {
                let parent = ev.has_parent.then_some(ev.parent_block_hash);
                let coalesceable = is_coalesceable_store(&ev);
                let merged_into = if coalesceable {
                    parent
                        .and_then(|p| open_tail.get(&p).copied())
                        .filter(|&chain_idx| {
                            let chain = &chains[chain_idx];
                            chain.block_size == ev.block_size
                                && chain.hashes.len() + ev.block_hashes.len() <= max_blocks
                                && ev.block_hashes.iter().all(|h| !chain.seen.contains(h))
                        })
                } else {
                    None
                };

                if let Some(chain_idx) = merged_into {
                    let parent_hash = parent.expect("matched parent exists");
                    open_tail.remove(&parent_hash);
                    let chain = &mut chains[chain_idx];
                    chain.hashes.extend_from_slice(ev.block_hashes);
                    chain.seen.extend(ev.block_hashes.iter().copied());
                    chain.token_slices.push(ev.token_ids);
                    chain.total_tokens += ev.token_ids.len();
                    if chain.hashes.len() < max_blocks {
                        let new_tail = *chain.hashes.last().expect("non-empty chain");
                        open_tail.entry(new_tail).or_insert(chain_idx);
                    }
                } else {
                    let mut seen = HashSet::new();
                    if coalesceable {
                        seen.reserve(ev.block_hashes.len() + usize::from(parent.is_some()));
                        if let Some(p) = parent {
                            seen.insert(p);
                        }
                        seen.extend(ev.block_hashes.iter().copied());
                    }
                    let chain_idx = chains.len();
                    chains.push(StoredChain {
                        parent,
                        block_size: ev.block_size,
                        hashes: ev.block_hashes.to_vec(),
                        token_slices: vec![ev.token_ids],
                        total_tokens: ev.token_ids.len(),
                        seen,
                    });
                    if coalesceable && ev.block_hashes.len() < max_blocks {
                        let tail = *ev.block_hashes.last().expect("non-empty store");
                        open_tail.entry(tail).or_insert(chain_idx);
                    }
                }
            }

            for chain in chains {
                out.push(WireEvent::Stored {
                    medium,
                    parent: chain.parent,
                    block_size: chain.block_size,
                    hashes: chain.hashes,
                    token_slices: chain.token_slices,
                    total_tokens: chain.total_tokens,
                });
            }
        }
    }
    out
}

/// One round's events as the publisher's payload (`KVEventBatch`): the
/// time, the events and this rank's attention-DP rank.
pub fn encode_batch(timestamp: f64, events: &KvEventBatch, dp_rank: i64) -> Vec<u8> {
    let mut encoder = ts_wire::msgpack::Encoder::new();
    encoder.array_len(3);
    encoder.float(timestamp);
    encoder.array_len(events.len());
    for event in events.iter() {
        // vLLM's layout: a store's `lora_id` (always nil) and `medium`, a
        // removal's `medium`; nil is the GPU cache.
        if event.stored {
            encoder.array_len(7);
            encoder.str("BlockStored");
            hashes(&mut encoder, event.block_hashes);
            if event.has_parent {
                encoder.uint(event.parent_block_hash);
            } else {
                encoder.nil();
            }
            encoder.array_len(event.token_ids.len());
            for &token in event.token_ids {
                encoder.int(i64::from(token));
            }
            encoder.int(i64::from(event.block_size));
            encoder.nil();
            encode_medium(&mut encoder, event.medium);
        } else {
            encoder.array_len(3);
            encoder.str("BlockRemoved");
            hashes(&mut encoder, event.block_hashes);
            encode_medium(&mut encoder, event.medium);
        }
    }
    encoder.int(dp_rank);
    encoder.into_single_frame()
}

/// Encodes one round's events after per-round coalescing into bounded
/// multi-block `BlockStored` / `BlockRemoved` wire events (`<= max_blocks` each).
/// Passing `max_blocks <= 1` produces the exact raw encoding of [`encode_batch`].
pub fn encode_coalesced_batch(
    timestamp: f64,
    events: &KvEventBatch,
    dp_rank: i64,
    max_blocks: usize,
) -> Vec<u8> {
    let max_blocks = max_blocks.min(MAX_KV_EVENT_COALESCE_MAX_BLOCKS);
    if max_blocks <= 1 || events.len() <= 1 {
        return encode_batch(timestamp, events, dp_rank);
    }
    let raw: Vec<KvEvent<'_>> = events.iter().collect();
    let coalesced = coalesce_wire_events(&raw, max_blocks);

    let mut encoder = ts_wire::msgpack::Encoder::new();
    encoder.array_len(3);
    encoder.float(timestamp);
    encoder.array_len(coalesced.len());
    for event in &coalesced {
        match event {
            WireEvent::Stored {
                medium,
                parent,
                block_size,
                hashes: block_hashes,
                token_slices,
                total_tokens,
            } => {
                encoder.array_len(7);
                encoder.str("BlockStored");
                hashes(&mut encoder, block_hashes);
                match parent {
                    Some(p) => encoder.uint(*p),
                    None => encoder.nil(),
                }
                encoder.array_len(*total_tokens);
                for slice in token_slices {
                    for &token in *slice {
                        encoder.int(i64::from(token));
                    }
                }
                encoder.int(i64::from(*block_size));
                encoder.nil();
                encode_medium(&mut encoder, *medium);
            }
            WireEvent::Removed {
                medium,
                hashes: block_hashes,
            } => {
                encoder.array_len(3);
                encoder.str("BlockRemoved");
                hashes(&mut encoder, block_hashes);
                encode_medium(&mut encoder, *medium);
            }
        }
    }
    encoder.int(dp_rank);
    encoder.into_single_frame()
}

fn hashes(encoder: &mut ts_wire::msgpack::Encoder, hashes: &[u64]) {
    encoder.array_len(hashes.len());
    for &hash in hashes {
        encoder.uint(hash);
    }
}

/// Bind addresses as libzmq reads them: `*` is every interface.
pub(crate) fn bind_address(endpoint: &str) -> String {
    endpoint.replace("://*:", "://0.0.0.0:")
}

struct Publication {
    timestamp: f64,
    events: KvEventBatch,
}

/// One rank's publisher; dropping it sends what is queued, for up to a
/// second.
pub struct KvEventPublisher {
    queue: Option<tokio::sync::mpsc::Sender<PendingBatch<Publication>>>,
    progress: KvEventProgress,
    done: std_mpsc::Receiver<()>,
}

impl KvEventPublisher {
    /// Start publishing for attention-DP rank `dp_rank`.
    pub fn start(config: &KvEventsConfig, dp_rank: i64) -> Result<Self, KvEventsError> {
        let coalesce_max_blocks = kv_event_coalesce_max_blocks_from_env()?;
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
        let (queue, batches) =
            tokio::sync::mpsc::channel::<PendingBatch<Publication>>(config.max_queue_size.max(1));
        let (done_tx, done) = std_mpsc::channel();
        let state = PublisherState {
            publisher,
            replay,
            topic: Bytes::from(config.topic.clone().into_bytes()),
            buffer: VecDeque::new(),
            buffer_steps: config.buffer_steps,
            dp_rank,
            next_seq: 0,
            coalesce_max_blocks,
        };
        thread::Builder::new()
            .name("ts-kv-events".to_owned())
            .spawn(move || {
                runtime.block_on(state.run(batches));
                let _ = done_tx.send(());
            })?;
        Ok(Self {
            queue: Some(queue),
            progress: KvEventProgress::default(),
            done,
        })
    }

    /// Observe queued and in-flight data without extending worker lifetime.
    pub fn progress(&self) -> KvEventProgress {
        self.progress.clone()
    }

    /// Move one round's independent snapshot to the publisher. Encoding and
    /// snapshot destruction run on its thread. This call still waits when the
    /// bounded queue is full: accepted events retain FIFO/replay semantics.
    pub fn publish(&self, events: KvEventBatch) {
        if events.is_empty() {
            return;
        }
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |elapsed| elapsed.as_secs_f64());
        if let Some(queue) = &self.queue {
            let bytes = events.byte_size();
            let pending = self
                .progress
                .track(Publication { timestamp, events }, bytes);
            let start = Instant::now();
            let result = queue.blocking_send(pending);
            self.progress.enqueued(start.elapsed());
            if result.is_err() {
                tracing::warn!("the KV event publisher stopped; dropping a batch");
            }
        }
    }
}

impl Drop for KvEventPublisher {
    fn drop(&mut self) {
        // The thread sends what is queued once the channel closes.
        self.queue = None;
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
    dp_rank: i64,
    next_seq: u64,
    coalesce_max_blocks: usize,
}

impl PublisherState {
    async fn run(mut self, mut batches: tokio::sync::mpsc::Receiver<PendingBatch<Publication>>) {
        loop {
            tokio::select! {
                batch = batches.recv() => match batch {
                    Some(PendingBatch {payload, lease}) => {
                        let Publication {timestamp, events} = payload;
                        let encoded = {
                            let _span = profile::span("KvEvents::encode");
                            encode_coalesced_batch(
                                timestamp,
                                &events,
                                self.dp_rank,
                                self.coalesce_max_blocks,
                            )
                        };
                        {
                            let _span = profile::span("KvEvents::releaseBatch");
                            drop(events);
                        }
                        {
                            let _span = profile::span("KvEvents::send");
                            self.send(encoded).await;
                        }
                        lease.complete();
                    },
                    None => return,
                },
                request = recv_replay(&mut self.replay) => match request {
                    Ok(request) => self.serve_replay(request).await,
                    Err(error) => tracing::warn!(%error, "KV event replay socket failed"),
                },
            }
        }
    }

    async fn send(&mut self, payload: Vec<u8>) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let payload = Bytes::from(payload);
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
    fn each_medium_encodes_as_routers_name_it() {
        use ts_scheduler::KvEventView;
        let removal = |medium| {
            let batch = KvEventBatch::from_events(&[KvEventView {
                stored: false,
                block_hashes: vec![7],
                has_parent: false,
                parent_block_hash: 0,
                token_ids: Vec::new(),
                block_size: 0,
                medium,
            }]);
            encode_batch(1.0, &batch, 0)
        };
        let contains =
            |bytes: &[u8], text: &[u8]| bytes.windows(text.len()).any(|window| window == text);
        assert!(!contains(&removal(0), b"CPU_PINNED") && !contains(&removal(0), b"DISK"));
        assert!(contains(&removal(1), MEDIUM_HOST.as_bytes()));
        assert!(contains(&removal(2), MEDIUM_DISK.as_bytes()));
    }

    #[test]
    fn a_full_queue_blocks_until_capacity_returns_and_preserves_owned_batches() {
        use std::sync::mpsc::TryRecvError;
        use ts_scheduler::KvEventView;
        let batch = |hash| {
            KvEventBatch::from_events(&[KvEventView {
                stored: false,
                block_hashes: vec![hash],
                has_parent: false,
                parent_block_hash: 0,
                token_ids: Vec::new(),
                block_size: 0,
                medium: 0,
            }])
        };
        let (queue, mut receiver) = tokio::sync::mpsc::channel(1);
        let (done_tx, done) = std_mpsc::channel();
        let publisher = KvEventPublisher {
            queue: Some(queue),
            progress: KvEventProgress::default(),
            done,
        };
        let progress = publisher.progress();
        publisher.publish(batch(11));
        let (returned_tx, returned_rx) = std_mpsc::channel();
        let sender = thread::spawn(move || {
            publisher.publish(batch(29));
            returned_tx.send(()).unwrap();
            publisher
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while progress.snapshot().pending_batches != 2 {
            assert!(
                Instant::now() < deadline,
                "second publication never entered the queue"
            );
            thread::yield_now();
        }
        assert_eq!(returned_rx.try_recv(), Err(TryRecvError::Empty));
        let first = receiver.blocking_recv().unwrap();
        assert_eq!(
            first.payload.events.iter().next().unwrap().block_hashes,
            [11]
        );
        returned_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let second = receiver.blocking_recv().unwrap();
        assert_eq!(
            second.payload.events.iter().next().unwrap().block_hashes,
            [29]
        );
        drop(first);
        drop(second);
        assert_eq!(progress.snapshot().pending_batches, 0);
        assert_eq!(progress.snapshot().peak_pending_batches, 2);
        done_tx.send(()).unwrap();
        drop(sender.join().unwrap());
    }

    #[test]
    fn coalesces_long_linear_chains_into_bounded_chunks() {
        use ts_scheduler::KvEventView;
        let mut views = Vec::new();
        for i in 1..=300u64 {
            views.push(KvEventView {
                stored: true,
                block_hashes: vec![i],
                has_parent: i > 1,
                parent_block_hash: i.saturating_sub(1),
                token_ids: vec![(i * 2) as i32, (i * 2 + 1) as i32],
                block_size: 2,
                medium: 0,
            });
        }
        let batch = KvEventBatch::from_events(&views);
        let raw: Vec<_> = batch.iter().collect();
        let coalesced = coalesce_wire_events(&raw, 128);
        assert_eq!(coalesced.len(), 3);
        let lens: Vec<usize> = coalesced
            .iter()
            .map(|e| match e {
                WireEvent::Stored { hashes, .. } => hashes.len(),
                WireEvent::Removed { .. } => panic!("expected stored"),
            })
            .collect();
        assert_eq!(lens, [128, 128, 44]);
        let parents: Vec<Option<u64>> = coalesced
            .iter()
            .map(|e| match e {
                WireEvent::Stored { parent, .. } => *parent,
                WireEvent::Removed { .. } => unreachable!(),
            })
            .collect();
        assert_eq!(parents, [None, Some(128), Some(256)]);

        // max_blocks = 1 matches encode_batch byte-for-byte.
        assert_eq!(
            encode_coalesced_batch(1.5, &batch, 0, 1),
            encode_batch(1.5, &batch, 0)
        );
    }

    #[test]
    fn coalesces_depth_interleaved_multi_request_and_multi_tier_events() {
        use ts_scheduler::KvEventView;
        let store = |h, parent: Option<u64>, medium| KvEventView {
            stored: true,
            block_hashes: vec![h],
            has_parent: parent.is_some(),
            parent_block_hash: parent.unwrap_or(0),
            token_ids: vec![h as i32, (h + 1000) as i32],
            block_size: 2,
            medium,
        };
        let remove = |h, medium| KvEventView {
            stored: false,
            block_hashes: vec![h],
            has_parent: false,
            parent_block_hash: 0,
            token_ids: Vec::new(),
            block_size: 0,
            medium,
        };
        // Mimics KvEventPrefixView::ApplyMutations output:
        // Removed sorted (depth DESC, medium DESC), then Stored sorted (depth ASC, medium ASC)
        // across two concurrent chains (10->11->12 and 20->21) plus a branch (11->32).
        let views = vec![
            remove(92, 2),
            remove(92, 1),
            remove(91, 2),
            remove(91, 1),
            remove(90, 0),
            store(10, None, 0),
            store(10, None, 1),
            store(20, None, 0),
            store(11, Some(10), 0),
            store(11, Some(10), 1),
            store(21, Some(20), 0),
            store(12, Some(11), 0),
            store(12, Some(11), 2),
            store(32, Some(11), 0),
        ];
        let batch = KvEventBatch::from_events(&views);
        let raw: Vec<_> = batch.iter().collect();
        let coalesced = coalesce_wire_events(&raw, 128);

        // Expect 3 Removed (Disk, Host, Device) + 5 Stored:
        //   Device: [10,11,12] (parent=None), [20,21] (parent=None), [32] (parent=11)
        //   Host:   [10,11] (parent=None)
        //   Disk:   [12] (parent=11)
        assert_eq!(coalesced.len(), 8);
        match &coalesced[0] {
            WireEvent::Removed { medium, hashes } => {
                assert_eq!(*medium, KvEventMedium::Disk);
                assert_eq!(hashes, &[92, 91]);
            }
            _ => panic!("expected Disk removal first"),
        }
        match &coalesced[1] {
            WireEvent::Removed { medium, hashes } => {
                assert_eq!(*medium, KvEventMedium::Host);
                assert_eq!(hashes, &[92, 91]);
            }
            _ => panic!("expected Host removal second"),
        }
        match &coalesced[2] {
            WireEvent::Removed { medium, hashes } => {
                assert_eq!(*medium, KvEventMedium::Device);
                assert_eq!(hashes, &[90]);
            }
            _ => panic!("expected Device removal third"),
        }
        match &coalesced[3] {
            WireEvent::Stored {
                medium,
                parent,
                hashes,
                total_tokens,
                ..
            } => {
                assert_eq!(*medium, KvEventMedium::Device);
                assert_eq!(*parent, None);
                assert_eq!(hashes, &[10, 11, 12]);
                assert_eq!(*total_tokens, 6);
            }
            _ => panic!("expected Device chain 1"),
        }
        match &coalesced[4] {
            WireEvent::Stored {
                medium,
                parent,
                hashes,
                ..
            } => {
                assert_eq!(*medium, KvEventMedium::Device);
                assert_eq!(*parent, None);
                assert_eq!(hashes, &[20, 21]);
            }
            _ => panic!("expected Device chain 2"),
        }
        match &coalesced[5] {
            WireEvent::Stored {
                medium,
                parent,
                hashes,
                ..
            } => {
                assert_eq!(*medium, KvEventMedium::Device);
                assert_eq!(*parent, Some(11));
                assert_eq!(hashes, &[32]);
            }
            _ => panic!("expected Device branch after parent chain"),
        }
        match &coalesced[6] {
            WireEvent::Stored {
                medium,
                parent,
                hashes,
                ..
            } => {
                assert_eq!(*medium, KvEventMedium::Host);
                assert_eq!(*parent, None);
                assert_eq!(hashes, &[10, 11]);
            }
            _ => panic!("expected Host chain"),
        }
        match &coalesced[7] {
            WireEvent::Stored {
                medium,
                parent,
                hashes,
                ..
            } => {
                assert_eq!(*medium, KvEventMedium::Disk);
                assert_eq!(*parent, Some(11));
                assert_eq!(hashes, &[12]);
            }
            _ => panic!("expected Disk continuation"),
        }
    }

    #[test]
    fn never_merges_short_tails_duplicate_hashes_or_across_phases() {
        use ts_scheduler::KvEventView;
        let views = vec![
            KvEventView {
                stored: true,
                block_hashes: vec![1],
                has_parent: false,
                parent_block_hash: 0,
                token_ids: vec![10, 11],
                block_size: 2,
                medium: 0,
            },
            // Short tail (1 token vs block_size=2): must stay standalone.
            KvEventView {
                stored: true,
                block_hashes: vec![2],
                has_parent: true,
                parent_block_hash: 1,
                token_ids: vec![12],
                block_size: 2,
                medium: 0,
            },
            // Phase boundary: Remove after Store must NOT be hoisted before Store.
            KvEventView {
                stored: false,
                block_hashes: vec![2],
                has_parent: false,
                parent_block_hash: 0,
                token_ids: Vec::new(),
                block_size: 0,
                medium: 0,
            },
            // Self-referencing / duplicate hash chain must not merge into itself.
            KvEventView {
                stored: true,
                block_hashes: vec![5],
                has_parent: false,
                parent_block_hash: 0,
                token_ids: vec![50, 51],
                block_size: 2,
                medium: 0,
            },
            KvEventView {
                stored: true,
                block_hashes: vec![5],
                has_parent: true,
                parent_block_hash: 5,
                token_ids: vec![50, 51],
                block_size: 2,
                medium: 0,
            },
        ];
        let batch = KvEventBatch::from_events(&views);
        let raw: Vec<_> = batch.iter().collect();
        let coalesced = coalesce_wire_events(&raw, 128);
        assert_eq!(coalesced.len(), 5);
        assert!(matches!(coalesced[0], WireEvent::Stored { ref hashes, .. } if hashes == &[1]));
        assert!(
            matches!(coalesced[1], WireEvent::Stored { ref hashes, total_tokens: 1, .. } if hashes == &[2])
        );
        assert!(matches!(coalesced[2], WireEvent::Removed { ref hashes, .. } if hashes == &[2]));
        assert!(matches!(coalesced[3], WireEvent::Stored { ref hashes, .. } if hashes == &[5]));
        assert!(matches!(coalesced[4], WireEvent::Stored { ref hashes, .. } if hashes == &[5]));
    }

    #[test]
    fn kv_event_coalesce_env_parser_validates_bounds_and_zero_disable() {
        assert_eq!(parse_kv_event_coalesce_max_blocks_value("0").unwrap(), 1);
        assert_eq!(parse_kv_event_coalesce_max_blocks_value("1").unwrap(), 1);
        assert_eq!(
            parse_kv_event_coalesce_max_blocks_value("128").unwrap(),
            128
        );
        assert_eq!(
            parse_kv_event_coalesce_max_blocks_value("1024").unwrap(),
            1024
        );
        assert!(parse_kv_event_coalesce_max_blocks_value("").is_err());
        assert!(parse_kv_event_coalesce_max_blocks_value("abc").is_err());
        assert!(parse_kv_event_coalesce_max_blocks_value("-1").is_err());
        assert!(parse_kv_event_coalesce_max_blocks_value("1025").is_err());
    }
}
