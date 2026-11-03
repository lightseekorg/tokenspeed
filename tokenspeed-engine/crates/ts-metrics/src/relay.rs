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

//! The feed to the node's metrics relay (`metrics/relay.py`), which runs the
//! output plugins outside the rank process (RFC 0002, "Metrics").
//!
//! Producers never block: events queue for one sender thread, and a full
//! queue drops the event and counts the drop, as `MetricEventSink.emit`
//! promises its callers.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread,
};

use zeromq::{PushSocket, Socket, SocketSend, ZmqMessage};

use crate::MetricsError;

/// Events waiting for the sender thread before new ones are dropped.
const QUEUE_CAPACITY: usize = 16 * 1024;

/// One metric update, as `MetricEvent` carries it.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricEvent {
    /// Wall-clock time of the update.
    pub timestamp_ns: u64,
    /// The metric's event name (a counter's ends in `_total`).
    pub metric: String,
    /// `inc`, `set` or `observe`.
    pub operation: &'static str,
    /// A counter's delta, a gauge's value or an observation.
    pub value: f64,
    /// Label names and values.
    pub labels: Vec<(String, String)>,
    /// Samples the observation stands for.
    pub weight: i64,
}

impl MetricEvent {
    /// The relay's message: a msgpack array of the fields in order, the
    /// labels as a map.
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = ts_wire::msgpack::Encoder::new();
        encoder.array_len(6);
        encoder.uint(self.timestamp_ns);
        encoder.str(&self.metric);
        encoder.str(self.operation);
        encoder.float(self.value);
        encoder.map_len(self.labels.len());
        for (name, value) in &self.labels {
            encoder.str(name);
            encoder.str(value);
        }
        encoder.int(self.weight);
        encoder.into_single_frame()
    }
}

/// The connection to the relay.
#[derive(Debug)]
pub(crate) struct RelaySender {
    queue: SyncSender<Vec<u8>>,
    dropped: AtomicU64,
}

impl RelaySender {
    /// Connect to the relay bound at `endpoint` and start the sender.
    pub(crate) fn connect(endpoint: &str) -> Result<Self, MetricsError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let mut socket = PushSocket::new();
        runtime.block_on(socket.connect(endpoint))?;
        let (queue, events) = mpsc::sync_channel::<Vec<u8>>(QUEUE_CAPACITY);
        thread::Builder::new()
            .name("ts-metrics-relay".to_owned())
            .spawn(move || {
                while let Ok(event) = events.recv() {
                    if let Err(error) = runtime.block_on(socket.send(ZmqMessage::from(event))) {
                        tracing::warn!(%error, "metrics relay send failed; stopping the feed");
                        return;
                    }
                }
            })?;
        Ok(Self {
            queue,
            dropped: AtomicU64::new(0),
        })
    }

    pub(crate) fn emit(&self, event: &MetricEvent) {
        match self.queue.try_send(event.encode()) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(_)) => {
                let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if dropped.is_power_of_two() {
                    tracing::warn!(dropped, "metrics relay queue full; dropping events");
                }
            }
        }
    }

    /// Events dropped on a full queue so far.
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}
