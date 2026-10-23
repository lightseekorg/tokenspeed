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

//! The engine's metrics (`metrics/collector.py`), produced in Rust (RFC
//! 0002, "Metrics").
//!
//! Every metric keeps the name, labels and buckets the Python engine gave
//! it, and reaches the same two outputs:
//!
//! * Prometheus: the series go to this process's multiprocess files in
//!   `PROMETHEUS_MULTIPROC_DIR` (`prom`), which the engine's `/metrics`
//!   endpoint already aggregates.
//! * Output plugins: each update is a `MetricEvent` for the node's relay
//!   process (`relay`), which runs the plugins; a gauge left unchanged is
//!   re-sent at most once a second.
//!
//! Metrics never fail serving: an output that errors is logged and skipped.

mod prom;
mod relay;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub use relay::MetricEvent;

use crate::{prom::PromFiles, relay::RelaySender};

/// A failure to set an output up.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    /// A file or thread could not be created.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The relay could not be reached.
    #[error("metrics relay: {0}")]
    Relay(#[from] zeromq::ZmqError),
}

/// Where the metrics go.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetricsConfig {
    /// `PROMETHEUS_MULTIPROC_DIR`, when the Prometheus reporter is selected.
    pub prometheus_dir: Option<PathBuf>,
    /// The node's relay endpoint, when an output plugin is selected.
    pub relay_endpoint: Option<String>,
}

/// The process's outputs, shared by every metric.
#[derive(Debug)]
pub struct Exporter {
    prom: Option<Mutex<PromFiles>>,
    relay: Option<RelaySender>,
}

impl Exporter {
    /// The outputs `config` selects.
    pub fn new(config: &MetricsConfig) -> Result<Arc<Self>, MetricsError> {
        let relay = match &config.relay_endpoint {
            Some(endpoint) => Some(RelaySender::connect(endpoint)?),
            None => None,
        };
        Ok(Arc::new(Self {
            prom: config
                .prometheus_dir
                .clone()
                .map(|directory| Mutex::new(PromFiles::new(directory))),
            relay,
        }))
    }

    /// Events the relay feed dropped on a full queue.
    pub fn dropped_events(&self) -> u64 {
        self.relay.as_ref().map_or(0, RelaySender::dropped)
    }

    fn write(&self, prefix: &'static str, key: &str, value: f64) {
        let Some(prom) = &self.prom else {
            return;
        };
        let mut files = prom.lock().expect("metrics files lock");
        if let Err(error) = files.write(prefix, key, value) {
            tracing::warn!(%error, prefix, "writing a Prometheus multiprocess file failed");
        }
    }

    fn emit(&self, event: impl FnOnce() -> MetricEvent) {
        if let Some(relay) = &self.relay {
            relay.emit(&event());
        }
    }

    fn emits(&self) -> bool {
        self.relay.is_some()
    }
}

/// How a raw-output gauge update that changes nothing is throttled.
const GAUGE_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Counter,
    Gauge,
    Histogram,
}

#[derive(Debug)]
struct Series {
    labels: Vec<(String, String)>,
    /// Counter total or gauge value.
    value: f64,
    /// Histogram: observations per bucket (not cumulative), the last
    /// `+Inf`; and their sum.
    buckets: Vec<f64>,
    sum: f64,
    /// Prometheus keys: the sample's, then each bucket's.
    key: String,
    bucket_keys: Vec<String>,
    last_gauge: Option<(f64, Instant)>,
}

/// One metric family: a counter, gauge or histogram with fixed label names.
#[derive(Debug)]
pub struct Metric {
    name: &'static str,
    /// The name its raw events carry: a counter's gains `_total`.
    event_name: String,
    help: &'static str,
    labelnames: Vec<&'static str>,
    kind: Kind,
    /// Histogram bounds, `+Inf` last.
    bounds: Vec<f64>,
    exporter: Arc<Exporter>,
    series: Mutex<HashMap<Vec<String>, Series>>,
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64)
}

impl Metric {
    fn new(
        exporter: &Arc<Exporter>,
        kind: Kind,
        name: &'static str,
        help: &'static str,
        labelnames: &[&'static str],
        bounds: &[f64],
    ) -> Self {
        let event_name = if kind == Kind::Counter && !name.ends_with("_total") {
            format!("{name}_total")
        } else {
            name.to_owned()
        };
        let mut bounds = bounds.to_vec();
        if kind == Kind::Histogram && bounds.last() != Some(&f64::INFINITY) {
            bounds.push(f64::INFINITY);
        }
        Self {
            name,
            event_name,
            help,
            labelnames: labelnames.to_vec(),
            kind,
            bounds,
            exporter: Arc::clone(exporter),
            series: Mutex::new(HashMap::new()),
        }
    }

    /// A counter (`MetricFactory.counter`).
    pub fn counter(
        exporter: &Arc<Exporter>,
        name: &'static str,
        help: &'static str,
        labelnames: &[&'static str],
    ) -> Self {
        Self::new(exporter, Kind::Counter, name, help, labelnames, &[])
    }

    /// A gauge aggregated across processes by its maximum (`livemax`, the
    /// mode every engine gauge uses).
    pub fn gauge(
        exporter: &Arc<Exporter>,
        name: &'static str,
        help: &'static str,
        labelnames: &[&'static str],
    ) -> Self {
        Self::new(exporter, Kind::Gauge, name, help, labelnames, &[])
    }

    /// A histogram over `buckets` (`+Inf` is added).
    pub fn histogram(
        exporter: &Arc<Exporter>,
        name: &'static str,
        help: &'static str,
        labelnames: &[&'static str],
        buckets: &[f64],
    ) -> Self {
        Self::new(exporter, Kind::Histogram, name, help, labelnames, buckets)
    }

    fn with_series<R>(&self, values: &[&str], update: impl FnOnce(&mut Series) -> R) -> R {
        assert_eq!(
            values.len(),
            self.labelnames.len(),
            "metric {} takes labels {:?}",
            self.name,
            self.labelnames
        );
        let mut series = self.series.lock().expect("metric series lock");
        let key: Vec<String> = values.iter().map(|&value| value.to_owned()).collect();
        let mut created = false;
        let entry = series.entry(key).or_insert_with(|| {
            created = true;
            let labels: Vec<(&str, &str)> = self
                .labelnames
                .iter()
                .copied()
                .zip(values.iter().copied())
                .collect();
            let sample = match self.kind {
                Kind::Counter => format!("{}_total", self.name),
                Kind::Gauge => self.name.to_owned(),
                Kind::Histogram => format!("{}_sum", self.name),
            };
            let bucket_keys = if self.kind == Kind::Histogram {
                let bucket = format!("{}_bucket", self.name);
                self.bounds
                    .iter()
                    .map(|&bound| {
                        let le = prom::le_label(bound);
                        let mut with_le = labels.clone();
                        with_le.push(("le", &le));
                        prom::sample_key(self.name, &bucket, &with_le, self.help)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Series {
                labels: labels
                    .iter()
                    .map(|&(name, value)| (name.to_owned(), value.to_owned()))
                    .collect(),
                value: 0.0,
                buckets: vec![0.0; self.bounds.len()],
                sum: 0.0,
                key: prom::sample_key(self.name, &sample, &labels, self.help),
                bucket_keys,
                last_gauge: None,
            }
        });
        if created && self.kind == Kind::Histogram {
            // As `prometheus_client` does: every bucket of a new series is
            // stored, so the scrape shows the empty ones too.
            self.exporter.write("histogram", &entry.key, 0.0);
            for key in &entry.bucket_keys {
                self.exporter.write("histogram", key, 0.0);
            }
        }
        update(entry)
    }

    fn event(
        &self,
        operation: &'static str,
        value: f64,
        labels: &[(String, String)],
        weight: i64,
    ) -> MetricEvent {
        MetricEvent {
            timestamp_ns: now_ns(),
            metric: self.event_name.clone(),
            operation,
            value,
            labels: labels.to_vec(),
            weight,
        }
    }

    /// Add `amount` (non-negative) to a counter.
    pub fn inc(&self, labels: &[&str], amount: f64) {
        assert_eq!(self.kind, Kind::Counter, "{} is not a counter", self.name);
        assert!(amount >= 0.0, "counters cannot decrease");
        self.with_series(labels, |series| {
            series.value += amount;
            self.exporter.write("counter", &series.key, series.value);
            self.exporter
                .emit(|| self.event("inc", amount, &series.labels, 1));
        });
    }

    /// Set a gauge.
    pub fn set(&self, labels: &[&str], value: f64) {
        assert_eq!(self.kind, Kind::Gauge, "{} is not a gauge", self.name);
        self.with_series(labels, |series| {
            series.value = value;
            self.exporter.write("gauge_livemax", &series.key, value);
            if !self.exporter.emits() {
                return;
            }
            let now = Instant::now();
            // Idle rounds must not queue a duplicate event each.
            if let Some((last, at)) = series.last_gauge {
                if last == value && now.duration_since(at) < GAUGE_SNAPSHOT_INTERVAL {
                    return;
                }
            }
            self.exporter
                .emit(|| self.event("set", value, &series.labels, 1));
            series.last_gauge = Some((value, now));
        });
    }

    /// Record one histogram observation.
    pub fn observe(&self, labels: &[&str], value: f64) {
        self.observe_weighted(labels, value, value, 1);
    }

    /// Record a per-token interval: `interval` over `num_tokens` tokens,
    /// as `num_tokens` observations of their mean that sum to `interval`.
    pub fn observe_batched(&self, labels: &[&str], interval: f64, num_tokens: i64) {
        if num_tokens <= 0 {
            return;
        }
        self.observe_weighted(labels, interval / num_tokens as f64, interval, num_tokens);
    }

    fn observe_weighted(&self, labels: &[&str], value: f64, total: f64, weight: i64) {
        assert_eq!(
            self.kind,
            Kind::Histogram,
            "{} is not a histogram",
            self.name
        );
        self.with_series(labels, |series| {
            series.sum += total;
            self.exporter.write("histogram", &series.key, series.sum);
            if let Some(index) = self.bounds.iter().position(|&bound| value <= bound) {
                series.buckets[index] += weight as f64;
                self.exporter.write(
                    "histogram",
                    &series.bucket_keys[index],
                    series.buckets[index],
                );
            }
            self.exporter
                .emit(|| self.event("observe", value, &series.labels, weight));
        });
    }
}

/// The engine's labels: `model_name`, `app_key` and the attention-DP rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineLabels {
    /// `--served-model-name`.
    pub model_name: String,
    /// `--app-key`, empty when unset.
    pub app_key: String,
    /// The attention-DP rank.
    pub dp_rank: i64,
}

const ENGINE_LABELS: [&str; 3] = ["model_name", "app_key", "dp_rank"];

/// The scheduler loop's metrics (`EngineMetrics`), on the attention-TP
/// group's first rank.
#[derive(Debug)]
pub struct EngineMetrics {
    labels: [String; 3],
    num_requests_running: Metric,
    num_requests_waiting: Metric,
    kv_cache_usage: Metric,
    iteration_tokens: Metric,
    spec_decode_num_accepted_tokens: Metric,
    spec_decode_num_draft_tokens: Metric,
    spec_decode_num_drafts: Metric,
    jit_serving_compiles: Metric,
    jit_serving_compile_seconds: Metric,
    num_nan_aborted_requests: Metric,
    num_capacity_aborted_requests: Metric,
    num_preemptions: Metric,
    request_queue_time: Metric,
    /// Serving-time compile totals already exported.
    exported_compiles: u64,
    exported_compile_seconds: f64,
}

impl EngineMetrics {
    /// The metrics, exported through `exporter`.
    pub fn new(exporter: &Arc<Exporter>, labels: &EngineLabels) -> Self {
        let names = &ENGINE_LABELS;
        Self {
            labels: [
                labels.model_name.clone(),
                labels.app_key.clone(),
                labels.dp_rank.to_string(),
            ],
            num_requests_running: Metric::gauge(
                exporter,
                "tokenspeed:num_requests_running",
                "Requests with scheduler-side generation state (decode path).",
                names,
            ),
            num_requests_waiting: Metric::gauge(
                exporter,
                "tokenspeed:num_requests_waiting",
                "Requests waiting in the C++ scheduler queue.",
                names,
            ),
            kv_cache_usage: Metric::gauge(
                exporter,
                "tokenspeed:kv_cache_usage_perc",
                "Fraction of device KV pages in use (0-1).",
                names,
            ),
            iteration_tokens: Metric::histogram(
                exporter,
                "tokenspeed:iteration_tokens_total",
                "Tokens scheduled in one scheduler forward step.",
                names,
                &[
                    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
                    2048.0, 4096.0, 8192.0,
                ],
            ),
            spec_decode_num_accepted_tokens: Metric::counter(
                exporter,
                "tokenspeed:spec_decode_num_accepted_tokens",
                "Accepted speculative draft tokens (excludes the bonus token sampled after verify).",
                names,
            ),
            spec_decode_num_draft_tokens: Metric::counter(
                exporter,
                "tokenspeed:spec_decode_num_draft_tokens",
                "Draft tokens proposed across verify steps.",
                names,
            ),
            spec_decode_num_drafts: Metric::counter(
                exporter,
                "tokenspeed:spec_decode_num_drafts",
                "Number of speculative verify rounds (per request-slot).",
                names,
            ),
            jit_serving_compiles: Metric::counter(
                exporter,
                "tokenspeed:jit_serving_compiles",
                "Triton kernel compilations after startup; each one stalls the forward thread.",
                names,
            ),
            jit_serving_compile_seconds: Metric::counter(
                exporter,
                "tokenspeed:jit_serving_compile_seconds",
                "Wall time spent in Triton kernel compilations after startup.",
                names,
            ),
            num_nan_aborted_requests: Metric::counter(
                exporter,
                "tokenspeed:num_nan_aborted_requests",
                "Requests terminated by the NaN guard (NaN in logits or an out-of-vocab sampled \
                 token id).",
                names,
            ),
            num_capacity_aborted_requests: Metric::counter(
                exporter,
                "tokenspeed:num_capacity_aborted_requests",
                "Requests the scheduler aborted because a capacity retraction could not image its \
                 victim (retraction snapshot pool or image rows exhausted); clients may resubmit \
                 them.",
                names,
            ),
            num_preemptions: Metric::counter(
                exporter,
                "tokenspeed:num_preemptions",
                "Requests preempted by a capacity retraction: the scheduler suspended a resident \
                 request (its pages imaged to the Host snapshot pool; it later resumes by \
                 restore, nothing is recomputed) to free Device pages for a blocked admission. \
                 One increment per retraction victim per round, counted from the plan's \
                 snapshot-store ops; forced retractions from --debug-force-retraction-interval \
                 count too. A retraction whose victim cannot be imaged aborts it instead — see \
                 tokenspeed:num_capacity_aborted_requests.",
                names,
            ),
            request_queue_time: Metric::histogram(
                exporter,
                "tokenspeed:request_queue_time_seconds",
                "Wall time from submission to the scheduler's waiting queue to first dispatch \
                 into a forward batch, observed once per request. Grammar-compile, pause-hold, \
                 and EPD embedding waits precede submission and are excluded, as are \
                 tokenizer-manager and IPC wait.",
                names,
                &[
                    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0,
                    60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
                ],
            ),
            exported_compiles: 0,
            exported_compile_seconds: 0.0,
        }
    }

    fn labels(&self) -> [&str; 3] {
        [&self.labels[0], &self.labels[1], &self.labels[2]]
    }

    /// One planning round's snapshot (`record_scheduler_iteration`):
    /// requests running and waiting, the KV pages in use, and the tokens
    /// the round forwards.
    pub fn record_scheduler_iteration(
        &self,
        running: i64,
        waiting: i64,
        active_pages: i64,
        total_pages: i64,
        iteration_tokens: i64,
    ) {
        let labels = self.labels();
        let ratio = if total_pages > 0 {
            active_pages as f64 / total_pages as f64
        } else {
            0.0
        };
        self.num_requests_running.set(&labels, running as f64);
        self.num_requests_waiting.set(&labels, waiting as f64);
        self.kv_cache_usage.set(&labels, ratio);
        if iteration_tokens > 0 {
            self.iteration_tokens
                .observe(&labels, iteration_tokens as f64);
        }
    }

    /// A verify step's speculation counts over its decode slots.
    pub fn record_spec_decode_step(
        &self,
        num_decode_slots: i64,
        accepted_draft_tokens: i64,
        draft_width: i64,
    ) {
        if num_decode_slots <= 0 {
            return;
        }
        let labels = self.labels();
        self.spec_decode_num_drafts
            .inc(&labels, num_decode_slots as f64);
        self.spec_decode_num_draft_tokens
            .inc(&labels, (num_decode_slots * draft_width) as f64);
        self.spec_decode_num_accepted_tokens
            .inc(&labels, accepted_draft_tokens.max(0) as f64);
    }

    /// The data plane's serving-time compilation totals; exports what they
    /// gained since the last call.
    pub fn record_jit_compiles(&mut self, serving_compiles: u64, serving_seconds: f64) {
        if serving_compiles <= self.exported_compiles {
            return;
        }
        let labels = self.labels();
        self.jit_serving_compiles
            .inc(&labels, (serving_compiles - self.exported_compiles) as f64);
        self.jit_serving_compile_seconds.inc(
            &labels,
            (serving_seconds - self.exported_compile_seconds).max(0.0),
        );
        self.exported_compiles = serving_compiles;
        self.exported_compile_seconds = serving_seconds;
    }

    /// A request the NaN guard terminated.
    pub fn record_nan_abort(&self) {
        self.num_nan_aborted_requests.inc(&self.labels(), 1.0);
    }

    /// A request the scheduler aborted for capacity.
    pub fn record_capacity_abort(&self) {
        self.num_capacity_aborted_requests.inc(&self.labels(), 1.0);
    }

    /// A round's retraction victims.
    pub fn record_preemptions(&self, count: i64) {
        if count > 0 {
            self.num_preemptions.inc(&self.labels(), count as f64);
        }
    }

    /// A request's wait from submission to its first forward.
    pub fn observe_request_queue_time(&self, seconds: f64) {
        self.request_queue_time.observe(&self.labels(), seconds);
    }
}

/// PD transfer metrics (`KVTransferMetrics`), on every rank of a
/// disaggregated engine.
#[derive(Debug)]
pub struct KvTransferMetrics {
    labels: [String; 2],
    latency: Metric,
    timeouts: Metric,
    failures: Metric,
}

impl KvTransferMetrics {
    /// The metrics for `model_name` / `app_key`, exported through `exporter`.
    pub fn new(exporter: &Arc<Exporter>, model_name: &str, app_key: &str) -> Self {
        let names = ["model_name", "app_key"];
        Self {
            labels: [model_name.to_owned(), app_key.to_owned()],
            latency: Metric::histogram(
                exporter,
                "tokenspeed:kv_transfer_latency_seconds",
                "Wall time of one KV cache transfer send to a peer.",
                &names,
                &[
                    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
                    30.0, 60.0,
                ],
            ),
            timeouts: Metric::counter(
                exporter,
                "tokenspeed:kv_transfer_timeouts",
                "KV transfers that timed out, by transfer stage.",
                &["model_name", "app_key", "stage"],
            ),
            failures: Metric::counter(
                exporter,
                "tokenspeed:kv_transfer_failures",
                "KV transfers aborted on failure: send failures on the prefill node, and rooms \
                 failed on prefill-node loss on the decode node. Timeouts are counted separately.",
                &names,
            ),
        }
    }

    /// One send's wall time.
    pub fn observe_latency(&self, seconds: f64) {
        self.latency
            .observe(&[&self.labels[0], &self.labels[1]], seconds);
    }

    /// A transfer that timed out in `stage` (`bootstrapping` on the prefill
    /// node, `waiting_for_input` on the decode node).
    pub fn record_timeout(&self, stage: &str) {
        self.timeouts
            .inc(&[&self.labels[0], &self.labels[1], stage], 1.0);
    }

    /// A transfer aborted on failure.
    pub fn record_failure(&self) {
        self.failures.inc(&[&self.labels[0], &self.labels[1]], 1.0);
    }
}

/// The L3 metrics' labels: the engine's, plus the attention-TP rank and the
/// engine's role (`--disaggregation-mode`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3Labels {
    /// `--served-model-name`.
    pub model_name: String,
    /// `--app-key`, empty when unset.
    pub app_key: String,
    /// The attention-DP rank.
    pub dp_rank: i64,
    /// The attention-TP rank.
    pub tp_rank: i64,
    /// `null`, `prefill` or `decode`.
    pub engine_role: String,
}

const L3_LABELS: [&str; 5] = ["model_name", "app_key", "dp_rank", "tp_rank", "engine_role"];

/// One operation kind's counters in an L3 executor snapshot
/// (`OperationSnapshot`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct L3OperationSnapshot {
    /// `query`, `get`, `put` or `cancel`.
    pub kind: String,
    /// Cumulative event counts, by event.
    pub events: Vec<(String, f64)>,
    /// Keys reported successful, cumulative.
    pub success_keys: f64,
    /// Keys reported failed, cumulative.
    pub failed_keys: f64,
    /// Successful adapter bytes, cumulative.
    pub success_bytes: f64,
    /// Queue time, cumulative seconds.
    pub queue_seconds: f64,
    /// Adapter call time, cumulative seconds.
    pub backend_seconds: f64,
    /// Jobs now, by state.
    pub operations: Vec<(String, f64)>,
    /// Job bytes now, by state.
    pub bytes: Vec<(String, f64)>,
}

/// An L3 executor snapshot (`ExecutorSnapshot`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct L3ExecutorSnapshot {
    /// Per operation kind.
    pub operations: Vec<L3OperationSnapshot>,
    /// Backend health: `healthy`, `open` or `half_open`.
    pub health_state: String,
    /// The backend health epoch.
    pub health_epoch: f64,
    /// Native retirement unproven: new I/O disabled until restart.
    pub unsafe_io: bool,
}

/// The scheduler's storage observations (`StorageStats`), mirrored on every
/// rank of a replica.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct L3SchedulerStats {
    /// Storage operations the scheduler has in flight.
    pub operations: f64,
    /// Host blocks storage operations hold.
    pub held_blocks: f64,
    /// Host blocks quarantined after unproven retirement.
    pub quarantined: f64,
    /// Keys known to be absent from the store.
    pub negative_keys: f64,
    /// Published blocks waiting to be backed up.
    pub backup_candidates: f64,
    /// Requests waiting on their storage preparation.
    pub waiting_requests: f64,
    /// Requests whose preparation is ready.
    pub ready_requests: f64,
    /// Preparations that timed out waiting, cumulative.
    pub wait_timeouts: f64,
    /// Ready preparations that expired, cumulative.
    pub ready_expirations: f64,
    /// The pressure gate is closed.
    pub pressure_gate_closed: bool,
    /// Preparations bypassed by pressure, cumulative.
    pub pressure_bypasses: f64,
    /// Backup candidates dropped by pressure, cumulative.
    pub pressure_dropped_backup_keys: f64,
    /// Backup candidates dropped on overflow, cumulative.
    pub overflow_dropped_backup_keys: f64,
    /// Backup candidates dropped with Host memory unavailable, cumulative.
    pub unavailable_dropped_backup_keys: f64,
    /// Backup candidates dropped with writes disabled, cumulative.
    pub disabled_dropped_backup_keys: f64,
    /// Backup candidates dropped on cleanup, cumulative.
    pub cleanup_dropped_backup_keys: f64,
}

/// The L3 storage tier's metrics (`L3Metrics`), on every rank with L3 and
/// `--enable-metrics`. Snapshots carry cumulative counters; each export adds
/// only what grew since the last one.
#[derive(Debug)]
pub struct L3Metrics {
    labels: [String; 5],
    events: Metric,
    keys: Metric,
    bytes: Metric,
    seconds: Metric,
    operations: Metric,
    held_bytes: Metric,
    health: Metric,
    epoch: Metric,
    unsafe_io: Metric,
    resources: Metric,
    expirations: Metric,
    pressure_bypasses: Metric,
    pressure_drops: Metric,
    backup_drops: Metric,
    /// Counter totals already exported, by counter and extra labels.
    exported: HashMap<(&'static str, Vec<String>), f64>,
}

impl L3Metrics {
    /// The metrics, exported through `exporter`.
    pub fn new(exporter: &Arc<Exporter>, labels: &L3Labels) -> Self {
        let with = |extra: &[&'static str]| -> Vec<&'static str> {
            L3_LABELS.iter().chain(extra).copied().collect()
        };
        Self {
            labels: [
                labels.model_name.clone(),
                labels.app_key.clone(),
                labels.dp_rank.to_string(),
                labels.tp_rank.to_string(),
                labels.engine_role.clone(),
            ],
            events: Metric::counter(
                exporter,
                "tokenspeed:l3_io_events",
                "Local executor events; backend_calls counts adapter invocations, not native \
                 transfers.",
                &with(&["kind", "event"]),
            ),
            keys: Metric::counter(
                exporter,
                "tokenspeed:l3_io_keys",
                "Reported per-key outcomes, including query misses and rejected/cancelled jobs.",
                &with(&["kind", "result"]),
            ),
            bytes: Metric::counter(
                exporter,
                "tokenspeed:l3_io_adapter_bytes",
                "Successful adapter bytes, including existing PUT objects; not network bytes.",
                &with(&["kind"]),
            ),
            seconds: Metric::counter(
                exporter,
                "tokenspeed:l3_io_seconds",
                "Cumulative local queue and adapter call durations.",
                &with(&["kind", "stage"]),
            ),
            operations: Metric::gauge(
                exporter,
                "tokenspeed:l3_io_operations",
                "Local jobs, including completed facts not yet polled.",
                &with(&["kind", "state"]),
            ),
            held_bytes: Metric::gauge(
                exporter,
                "tokenspeed:l3_io_buffer_bytes",
                "Job byte ranges; excludes C++ Ready and quarantined ownership after poll.",
                &with(&["kind", "state"]),
            ),
            health: Metric::gauge(
                exporter,
                "tokenspeed:l3_backend_state",
                "Backend health: healthy=0, open=1, half_open=2. Quarantine is separate.",
                &L3_LABELS,
            ),
            epoch: Metric::gauge(
                exporter,
                "tokenspeed:l3_backend_epoch",
                "Local backend health epoch.",
                &L3_LABELS,
            ),
            unsafe_io: Metric::gauge(
                exporter,
                "tokenspeed:l3_io_quarantined",
                "Local native retirement unproven; new I/O disabled until restart.",
                &L3_LABELS,
            ),
            resources: Metric::gauge(
                exporter,
                "tokenspeed:l3_scheduler_resources",
                "Mirrored C++ resource counts; do not sum across TP ranks.",
                &with(&["resource"]),
            ),
            expirations: Metric::counter(
                exporter,
                "tokenspeed:l3_preparation_expirations",
                "Mirrored scheduler deadline expirations; do not sum across TP ranks.",
                &with(&["phase"]),
            ),
            pressure_bypasses: Metric::counter(
                exporter,
                "tokenspeed:l3_pressure_bypasses",
                "Mirrored preparation attempts bypassed by pressure; do not sum TP ranks.",
                &L3_LABELS,
            ),
            pressure_drops: Metric::counter(
                exporter,
                "tokenspeed:l3_pressure_dropped_backup_keys",
                "Mirrored backup candidates dropped by pressure; do not sum TP ranks.",
                &L3_LABELS,
            ),
            backup_drops: Metric::counter(
                exporter,
                "tokenspeed:l3_dropped_backup_keys",
                "Mirrored backup candidates discarded by reason; do not sum TP ranks.",
                &with(&["reason"]),
            ),
            exported: HashMap::new(),
        }
    }

    fn label_values<'a>(&'a self, extra: &[&'a str]) -> Vec<&'a str> {
        self.labels
            .iter()
            .map(String::as_str)
            .chain(extra.iter().copied())
            .collect()
    }

    /// Export a cumulative counter's growth since its last export; a total
    /// that went backwards restarted, and counts in full.
    fn increment(&mut self, counter: &'static str, value: f64, extra: &[&str]) {
        let key = (
            counter,
            extra.iter().map(|&label| label.to_owned()).collect(),
        );
        let previous = self.exported.get(&key).copied().unwrap_or(0.0);
        let delta = if value >= previous {
            value - previous
        } else {
            value
        };
        if delta > 0.0 {
            let metric = match counter {
                "events" => &self.events,
                "keys" => &self.keys,
                "bytes" => &self.bytes,
                "seconds" => &self.seconds,
                "expirations" => &self.expirations,
                "pressure_bypasses" => &self.pressure_bypasses,
                "pressure_drops" => &self.pressure_drops,
                "backup_drops" => &self.backup_drops,
                _ => unreachable!("an L3 counter"),
            };
            metric.inc(&self.label_values(extra), delta);
        }
        self.exported.insert(key, value);
    }

    /// Export one snapshot and the scheduler's observations; no request or
    /// key identifier becomes a label.
    pub fn record(&mut self, io: &L3ExecutorSnapshot, scheduler: &L3SchedulerStats) {
        for row in &io.operations {
            let kind = row.kind.as_str();
            for (event, value) in &row.events {
                self.increment("events", *value, &[kind, event.as_str()]);
            }
            self.increment("keys", row.success_keys, &[kind, "success"]);
            self.increment("keys", row.failed_keys, &[kind, "failure"]);
            self.increment("bytes", row.success_bytes, &[kind]);
            self.increment("seconds", row.queue_seconds, &[kind, "queue"]);
            self.increment("seconds", row.backend_seconds, &[kind, "backend"]);
            for (state, value) in &row.operations {
                self.operations
                    .set(&self.label_values(&[kind, state.as_str()]), *value);
            }
            for (state, value) in &row.bytes {
                self.held_bytes
                    .set(&self.label_values(&[kind, state.as_str()]), *value);
            }
        }
        let health = match io.health_state.as_str() {
            "healthy" => 0.0,
            "open" => 1.0,
            _ => 2.0,
        };
        self.health.set(&self.label_values(&[]), health);
        self.epoch.set(&self.label_values(&[]), io.health_epoch);
        self.unsafe_io
            .set(&self.label_values(&[]), f64::from(u8::from(io.unsafe_io)));
        for (resource, value) in [
            ("operations", scheduler.operations),
            ("held_blocks", scheduler.held_blocks),
            ("quarantined", scheduler.quarantined),
            ("negative_keys", scheduler.negative_keys),
            ("backup_candidates", scheduler.backup_candidates),
            ("waiting_requests", scheduler.waiting_requests),
            ("ready_requests", scheduler.ready_requests),
            (
                "pressure_gate_closed",
                f64::from(u8::from(scheduler.pressure_gate_closed)),
            ),
        ] {
            self.resources.set(&self.label_values(&[resource]), value);
        }
        self.increment("expirations", scheduler.wait_timeouts, &["waiting"]);
        self.increment("expirations", scheduler.ready_expirations, &["ready"]);
        self.increment("pressure_bypasses", scheduler.pressure_bypasses, &[]);
        self.increment(
            "pressure_drops",
            scheduler.pressure_dropped_backup_keys,
            &[],
        );
        for (reason, count) in [
            ("pressure", scheduler.pressure_dropped_backup_keys),
            ("overflow", scheduler.overflow_dropped_backup_keys),
            (
                "host_unavailable",
                scheduler.unavailable_dropped_backup_keys,
            ),
            ("write_disabled", scheduler.disabled_dropped_backup_keys),
            ("cleanup", scheduler.cleanup_dropped_backup_keys),
        ] {
            self.increment("backup_drops", count, &[reason]);
        }
    }
}
