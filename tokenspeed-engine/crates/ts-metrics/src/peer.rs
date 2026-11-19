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

//! Logical peer attempts are observed by the DP owner. Transfer workers report physical shards.
use super::{EngineLabels, Exporter, Metric};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// Holds a shard's outstanding bytes from enqueue until retirement or cancellation.
pub struct QueuedPeerRead {
    metrics: Arc<PeerMetrics>,
    holder: String,
    member: usize,
    bytes: u64,
    queued: Instant,
}

impl QueuedPeerRead {
    /// Record a completed transport call. Dropping the guard also retires its bytes.
    pub fn finish(self, started: Instant, elapsed: Duration, ok: bool) {
        self.metrics.shard(
            &self.holder,
            self.member,
            started.duration_since(self.queued).as_secs_f64(),
            elapsed.as_secs_f64(),
            self.bytes,
            ok,
        );
    }
}

impl Drop for QueuedPeerRead {
    fn drop(&mut self) {
        self.metrics
            .outstanding(&self.holder, self.member, self.bytes, false);
    }
}

const SECONDS: &[f64] = &[
    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 15., 30., 60., 120.,
];
const TOKENS: &[f64] = &[0., 1., 32., 128., 512., 2048., 8192., 32768., 131072.];

/// Metrics for a peer-enabled replica. Physical series identify the reader member and target.
#[derive(Debug)]
pub struct PeerMetrics {
    model: String,
    app: String,
    dp: String,
    consumed: Metric,
    error: Metric,
    stage: Metric,
    remaining: Metric,
    snapshot: Metric,
    totals: Metric,
    physical: Metric,
    bytes: Metric,
    outstanding: Metric,
    loads: Mutex<HashMap<(String, usize), u64>>,
}

impl PeerMetrics {
    /// Define peer series without creating samples for absent peers.
    pub fn new(exporter: &Arc<Exporter>, labels: &EngineLabels) -> Self {
        let base = &["model_name", "app_key", "dp_rank"];
        let balance = &["model_name", "app_key", "dp_rank", "state"];
        let shard = &["model_name", "app_key", "dp_rank", "holder", "member"];
        let signed = [-131072., -32768., -8192., -2048., -512., -128., -32., -1.]
            .into_iter()
            .chain(TOKENS.iter().copied())
            .collect::<Vec<_>>();
        Self {
            model: labels.model_name.clone(),
            app: labels.app_key.clone(),
            dp: labels.dp_rank.to_string(),
            consumed: Metric::histogram(
                exporter,
                "tokenspeed:peer_consumed_tokens",
                "Remote-only prefix tokens used by a committed prefill, per hinted request including misses.",
                base, TOKENS,
            ),
            error: Metric::histogram(
                exporter,
                "tokenspeed:peer_estimation_error_tokens",
                "Router expected additional peer tokens minus actual consumption for the same request.",
                &["model_name","app_key","dp_rank","outcome"], &signed,
            ),
            remaining: Metric::histogram(
                exporter,
                "tokenspeed:peer_hard_pin_remaining_seconds",
                "Time until the hard-pin deadline when it is released or reclaimed.",
                &["model_name","app_key","dp_rank","outcome"], &[-1.,-0.01,0.,0.01,0.1,0.5,1.,5.,15.,30.,60.,120.],
            ),
            stage: Metric::histogram(
                exporter,
                "tokenspeed:peer_stage_seconds",
                "Logical request stage duration. The count by stage and outcome counts completed stage visits.",
                &["model_name","app_key","dp_rank","stage","outcome"], SECONDS,
            ),
            snapshot: Metric::gauge(
                exporter,
                "tokenspeed:peer_scheduler_state",
                "Current holder or reader scheduler state.",
                balance,
            ),
            totals: Metric::counter(
                exporter,
                "tokenspeed:peer_scheduler_events",
                "Cumulative scheduler events by kind.",
                balance,
            ),
            physical: Metric::histogram(
                exporter,
                "tokenspeed:peer_shard_seconds",
                "Per-rank READ queue wait or transfer service time.",
                &["model_name","app_key","dp_rank","holder","member","stage","outcome"], SECONDS,
            ),
            bytes: Metric::counter(
                exporter,
                "tokenspeed:peer_read_bytes",
                "Payload bytes of successful peer READ calls.",
                shard,
            ),
            outstanding: Metric::gauge(
                exporter,
                "tokenspeed:peer_outstanding_read_bytes",
                "Bytes queued or executing on this reader member for this holder.",
                shard,
            ),
            loads: Mutex::new(HashMap::new()),
        }
    }

    /// One terminal sample for a hinted request. Omitted predictions have no error sample.
    pub fn usage(&self, expected: Option<u64>, actual: u64, outcome: &str) {
        let l = [self.model.as_str(), &self.app, &self.dp];
        self.consumed.observe(&l, actual as f64);
        if let Some(expected) = expected {
            self.error.observe(
                &[l[0], l[1], l[2], outcome],
                expected as f64 - actual as f64,
            );
        }
    }
    /// One stage visit on the replica owner.
    pub fn stage(&self, stage: &str, outcome: &str, seconds: f64) {
        self.stage
            .observe(&[&self.model, &self.app, &self.dp, stage, outcome], seconds);
    }
    /// Actual hard pin eviction or release in the native scheduler clock.
    pub fn pin(&self, held_us: i64, remaining_us: i64, expired: bool) {
        let outcome = if expired { "expired" } else { "released" };
        self.stage("hard_pin", outcome, held_us as f64 / 1e6);
        self.remaining.observe(
            &[&self.model, &self.app, &self.dp, outcome],
            remaining_us as f64 / 1e6,
        );
    }

    /// Current scheduler state and changes to native cumulative counters.
    pub fn snapshot(&self, state: &str, value: usize) {
        self.snapshot
            .set(&[&self.model, &self.app, &self.dp, state], value as f64);
    }
    /// Positive increment of a native cumulative counter.
    pub fn increment(&self, state: &str, delta: u64) {
        if delta != 0 {
            self.totals
                .inc(&[&self.model, &self.app, &self.dp, state], delta as f64);
        }
    }
    /// Account for a shard from the time it is queued.
    pub fn queue_read(
        self: &Arc<Self>,
        holder: String,
        member: usize,
        bytes: u64,
    ) -> QueuedPeerRead {
        self.outstanding(&holder, member, bytes, true);
        QueuedPeerRead {
            metrics: Arc::clone(self),
            holder,
            member,
            bytes,
            queued: Instant::now(),
        }
    }

    /// Change outstanding byte load under the same lock as publishing it.
    fn outstanding(&self, holder: &str, member: usize, bytes: u64, submit: bool) {
        let mut loads = self.loads.lock().expect("peer load lock");
        let key = (holder.to_owned(), member);
        let n = loads.entry(key.clone()).or_default();
        if submit {
            *n += bytes;
        } else {
            debug_assert!(*n >= bytes, "outstanding peer bytes underflow");
            *n = n.saturating_sub(bytes);
        }
        self.outstanding.set(
            &[
                &self.model,
                &self.app,
                &self.dp,
                holder,
                &member.to_string(),
            ],
            *n as f64,
        );
        if *n == 0 {
            loads.remove(&key);
        }
    }
    /// Agreed PIN until the local device fence allows this rank to queue the READ.
    pub fn fence(&self, holder: &str, member: usize, seconds: f64) {
        self.physical.observe(
            &[
                &self.model,
                &self.app,
                &self.dp,
                holder,
                &member.to_string(),
                "device_fence",
                "ok",
            ],
            seconds,
        );
    }

    /// Separate physical queue delay from time spent in the transport.
    fn shard(&self, holder: &str, member: usize, queued: f64, elapsed: f64, bytes: u64, ok: bool) {
        let rank = member.to_string();
        let l = [&*self.model, &*self.app, &*self.dp, holder, &rank];
        self.physical
            .observe(&[l[0], l[1], l[2], holder, &rank, "queue", "ok"], queued);
        self.physical.observe(
            &[
                l[0],
                l[1],
                l[2],
                holder,
                &rank,
                "read",
                if ok { "ok" } else { "error" },
            ],
            elapsed,
        );
        if ok {
            self.bytes.inc(&l, bytes as f64);
        }
    }
}
