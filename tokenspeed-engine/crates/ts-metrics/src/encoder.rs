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

//! Encode-only EPD observations, exported through the engine's standard outputs.
use crate::{EngineLabels, Exporter, Metric};
use std::sync::Arc;
const ENCODE_LABELS: [&str; 3] = ["model_name", "app_key", "dp_rank"];
const TRANSFER_LABELS: [&str; 3] = ["model_name", "app_key", "tp_rank"];
const SECONDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0,
];
const COUNTS: &[f64] = &[
    1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0,
];

/// Logical encode-loop observations owned by the first attention-TP rank.
#[derive(Debug)]
pub struct EncoderMetrics {
    labels: [String; 3],
    arrivals: Metric,
    invalid: Metric,
    duplicates: Metric,
    encoded: Metric,
    failed: Metric,
    queue: Metric,
    running: Metric,
    sending: Metric,
    oldest: Metric,
    queue_time: Metric,
    execution_time: Metric,
    request_time: Metric,
    batch_tokens: Metric,
    batch_items: Metric,
    images: Metric,
    cache_lookups: Metric,
    cache_hits: Metric,
}
impl EncoderMetrics {
    /// Create logical encode metrics using the standard engine labels.
    pub fn new(exporter: &Arc<Exporter>, labels: &EngineLabels) -> Self {
        let values = [
            labels.model_name.clone(),
            labels.app_key.clone(),
            labels.dp_rank.to_string(),
        ];
        let counter = |name, help| Metric::counter(exporter, name, help, &ENCODE_LABELS);
        let gauge = |name, help| Metric::gauge(exporter, name, help, &ENCODE_LABELS);
        let histogram =
            |name, help, buckets| Metric::histogram(exporter, name, help, &ENCODE_LABELS, buckets);
        Self {
            labels: values,
            arrivals: counter(
                "tokenspeed:encoder_requests",
                "Encode wire requests received by TP0.",
            ),
            invalid: counter(
                "tokenspeed:encoder_invalid_requests",
                "Malformed or invalid encode requests.",
            ),
            duplicates: counter(
                "tokenspeed:encoder_duplicate_requests",
                "Encode requests whose room is already owned.",
            ),
            encoded: counter(
                "tokenspeed:encoder_encoded_requests",
                "Items whose embeddings reached their send slots.",
            ),
            failed: counter(
                "tokenspeed:encoder_failed_requests",
                "Items that failed vision encoding or ring-slot copy.",
            ),
            queue: gauge(
                "tokenspeed:encoder_queue_length",
                "Items waiting for a send slot or encoder batch.",
            ),
            running: gauge(
                "tokenspeed:encoder_running_requests",
                "Items submitted to the data plane and not copy-complete.",
            ),
            sending: gauge(
                "tokenspeed:encoder_sending_requests",
                "Items handed to the sender and retaining a ring slot.",
            ),
            oldest: gauge(
                "tokenspeed:encoder_queue_oldest_seconds",
                "Age of the oldest item in either FIFO queue.",
            ),
            queue_time: histogram(
                "tokenspeed:encoder_queue_time_seconds",
                "Admission to data-plane submission.",
                SECONDS,
            ),
            execution_time: histogram(
                "tokenspeed:encoder_execution_time_seconds",
                "Data-plane submission through observed device-copy completion.",
                SECONDS,
            ),
            request_time: histogram(
                "tokenspeed:encoder_request_time_seconds",
                "Admission through observed device-copy completion.",
                SECONDS,
            ),
            batch_tokens: histogram(
                "tokenspeed:encoder_batch_tokens",
                "Vision tokens in a dispatched batch.",
                COUNTS,
            ),
            batch_items: histogram(
                "tokenspeed:encoder_batch_items",
                "Items in a dispatched batch.",
                COUNTS,
            ),
            images: histogram(
                "tokenspeed:encoder_image_count",
                "Image items in one decoded encode request.",
                COUNTS,
            ),
            cache_lookups: counter(
                "tokenspeed:encoder_cache_lookups",
                "Whole-item embedding cache lookups.",
            ),
            cache_hits: counter(
                "tokenspeed:encoder_cache_hits",
                "Whole-item embedding cache hits.",
            ),
        }
    }
    fn labels(&self) -> [&str; 3] {
        [&self.labels[0], &self.labels[1], &self.labels[2]]
    }
    pub fn arrival(&self) {
        self.arrivals.inc(&self.labels(), 1.0);
    }
    pub fn invalid(&self) {
        self.invalid.inc(&self.labels(), 1.0);
    }
    pub fn duplicate(&self) {
        self.duplicates.inc(&self.labels(), 1.0);
    }
    pub fn cache_lookup(&self, hit: bool) {
        self.cache_lookups.inc(&self.labels(), 1.0);
        if hit {
            self.cache_hits.inc(&self.labels(), 1.0);
        }
    }
    pub fn image_count(&self, count: usize) {
        self.images.observe(&self.labels(), count as f64);
    }
    pub fn snapshot(&self, queued: usize, running: usize, sending: usize, oldest: f64) {
        let labels = self.labels();
        self.queue.set(&labels, queued as f64);
        self.running.set(&labels, running as f64);
        self.sending.set(&labels, sending as f64);
        self.oldest.set(&labels, oldest);
    }
    pub fn dispatched(&self, seconds: f64) {
        self.queue_time.observe(&self.labels(), seconds);
    }
    pub fn batch(&self, items: usize, tokens: i64) {
        let labels = self.labels();
        self.batch_items.observe(&labels, items as f64);
        self.batch_tokens.observe(&labels, tokens.max(0) as f64);
    }
    pub fn finished(&self, execution: f64, total: f64, success: bool) {
        let labels = self.labels();
        self.execution_time.observe(&labels, execution);
        self.request_time.observe(&labels, total);
        if success {
            self.encoded.inc(&labels, 1.0);
        } else {
            self.failed.inc(&labels, 1.0);
        }
    }
}

/// Physical embedding writes, owned separately by every encode TP rank.
#[derive(Debug)]
pub struct EncoderTransferMetrics {
    labels: [String; 3],
    completed: Metric,
    failures: Metric,
    timeouts: Metric,
    latency: Metric,
    bytes: Metric,
}
impl EncoderTransferMetrics {
    pub fn new(exporter: &Arc<Exporter>, model_name: &str, app_key: &str, tp_rank: usize) -> Self {
        let labels = [
            model_name.to_owned(),
            app_key.to_owned(),
            tp_rank.to_string(),
        ];
        let counter = |name, help| Metric::counter(exporter, name, help, &TRANSFER_LABELS);
        Self {
            labels,
            completed: counter(
                "tokenspeed:encoder_transfer_completed",
                "Embedding rooms written to their complete prefill fanout.",
            ),
            failures: counter(
                "tokenspeed:encoder_transfer_failures",
                "Rooms that failed at the physical transfer boundary.",
            ),
            timeouts: counter(
                "tokenspeed:encoder_transfer_timeouts",
                "Rooms that timed out waiting for the prefill fanout.",
            ),
            latency: Metric::histogram(
                exporter,
                "tokenspeed:encoder_transfer_seconds",
                "Successful physical room-write wall time.",
                &TRANSFER_LABELS,
                SECONDS,
            ),
            bytes: counter(
                "tokenspeed:encoder_transfer_size_bytes",
                "Successfully written embedding bytes including fanout.",
            ),
        }
    }
    fn labels(&self) -> [&str; 3] {
        [&self.labels[0], &self.labels[1], &self.labels[2]]
    }
    pub fn completed(&self, seconds: f64, bytes: u64) {
        let labels = self.labels();
        self.completed.inc(&labels, 1.0);
        self.latency.observe(&labels, seconds);
        self.bytes.inc(&labels, bytes as f64);
    }
    pub fn failed(&self, timeout: bool) {
        let labels = self.labels();
        self.failures.inc(&labels, 1.0);
        if timeout {
            self.timeouts.inc(&labels, 1.0);
        }
    }
}
