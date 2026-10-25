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
    collections::VecDeque,
    sync::mpsc as std_mpsc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use ts_scheduler::KvEventView;
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

/// One round's events as the publisher's payload (`KVEventBatch`): the
/// time, the events and this rank's attention-DP rank.
pub fn encode_batch(timestamp: f64, events: &[KvEventView], dp_rank: i64) -> Vec<u8> {
    let mut encoder = ts_wire::msgpack::Encoder::new();
    encoder.array_len(3);
    encoder.float(timestamp);
    encoder.array_len(events.len());
    for event in events {
        if event.stored {
            encoder.array_len(5);
            encoder.str("BlockStored");
            hashes(&mut encoder, &event.block_hashes);
            if event.has_parent {
                encoder.uint(event.parent_block_hash);
            } else {
                encoder.nil();
            }
            encoder.array_len(event.token_ids.len());
            for &token in &event.token_ids {
                encoder.int(i64::from(token));
            }
            encoder.int(i64::from(event.block_size));
        } else {
            encoder.array_len(2);
            encoder.str("BlockRemoved");
            hashes(&mut encoder, &event.block_hashes);
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
fn bind_address(endpoint: &str) -> String {
    endpoint.replace("://*:", "://0.0.0.0:")
}

/// One rank's publisher; dropping it sends what is queued, for up to a
/// second.
pub struct KvEventPublisher {
    queue: Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
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
        let (queue, batches) = tokio::sync::mpsc::channel::<Vec<u8>>(config.max_queue_size.max(1));
        let (done_tx, done) = std_mpsc::channel();
        let state = PublisherState {
            publisher,
            replay,
            topic: Bytes::from(config.topic.clone().into_bytes()),
            buffer: VecDeque::new(),
            buffer_steps: config.buffer_steps,
            next_seq: 0,
        };
        thread::Builder::new()
            .name("ts-kv-events".to_owned())
            .spawn(move || {
                runtime.block_on(state.run(batches));
                let _ = done_tx.send(());
            })?;
        Ok(Self {
            queue: Some(queue),
            dp_rank,
            done,
        })
    }

    /// Queue one round's events; waits while the queue is full.
    pub fn publish(&self, events: &[KvEventView]) {
        if events.is_empty() {
            return;
        }
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |elapsed| elapsed.as_secs_f64());
        let payload = encode_batch(timestamp, events, self.dp_rank);
        if let Some(queue) = &self.queue {
            if queue.blocking_send(payload).is_err() {
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
    next_seq: u64,
}

impl PublisherState {
    async fn run(mut self, mut batches: tokio::sync::mpsc::Receiver<Vec<u8>>) {
        loop {
            tokio::select! {
                batch = batches.recv() => match batch {
                    Some(payload) => self.send(payload).await,
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
}
