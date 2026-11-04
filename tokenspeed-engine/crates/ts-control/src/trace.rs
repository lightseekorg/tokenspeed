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

//! OTLP trace export: one span per request, built retrospectively at its
//! release.
//!
//! The timestamps a trace needs are the request stats' own — when it was
//! received, scheduled, prefilled, first tokened and finished — so the span
//! is assembled once, when the request leaves the processor, and handed to
//! the export thread over a bounded channel. The serving loop never blocks:
//! a full channel drops the span and counts it, the metrics relay's policy.
//! Trace ids derive from the request id (the same request traces to the same
//! id, so logs and traces cross-reference), span ids from a process counter.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError},
        Arc,
    },
    thread,
    time::Duration,
    time::Instant,
};

use serde_json::{json, Value};

use crate::output::Role;

/// How many spans may wait for the export thread.
const CHANNEL_SPANS: usize = 4096;

/// How many spans one OTLP request carries at most, and how long the
/// export thread waits to fill a batch.
const BATCH_SPANS: usize = 128;
const BATCH_WAIT: Duration = Duration::from_millis(100);

/// One drop warning per this many dropped spans: visible, not noisy.
const DROPPED_WARN_EVERY: u64 = 1024;

/// One finished request, as a span's whole content: wall-clock seconds
/// (`wall_seconds`), zero until the point was reached.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceRecord {
    /// The request's id, the span's `gen_ai.response.id`.
    pub rid: String,
    /// Whether this span covers local prefill, decode, or both.
    pub role: Role,
    /// The continual-prompting session, when the request named one.
    pub session: Option<String>,
    /// Request creation supplied by the frontend: the span's start.
    pub created: f64,
    /// First dispatched into a forward batch.
    pub scheduled: f64,
    /// The prompt's last chunk committed.
    pub prefill_done: f64,
    /// The first output token committed.
    pub first_token: f64,
    /// Finished or aborted: the span's end.
    pub finish: f64,
    /// The prompt's tokens.
    pub prompt_tokens: i64,
    /// The prompt's prefix-cache hit tokens.
    pub cached_tokens: i64,
    /// The generated tokens.
    pub output_tokens: i64,
    /// The finish reason's wire name.
    pub finish_reason: &'static str,
    /// Rounds the request's decode waited behind other requests' prefill.
    pub preempt_count: i64,
    /// Their seconds.
    pub preempt_seconds: f64,
}

/// Optional extension evaluated on the export thread, never on the control loop.
/// None preserves the native root trace. Returned IDs are validated before use.
pub type TraceProvider = Arc<dyn Fn(&str) -> Option<(String, Option<String>)> + Send + Sync>;

/// Sends request spans to an OTLP/HTTP endpoint on a thread of its own.
#[derive(Debug)]
pub struct TracePublisher {
    sender: SyncSender<TraceRecord>,
    sample_ratio: f64,
    dropped: AtomicU64,
    /// The export thread's endpoint and model, when it has one.
    endpoint: Option<(String, String)>,
}

impl TracePublisher {
    /// Export to `endpoint` (an OTLP/HTTP receiver, e.g. a collector's
    /// `/v1/traces`), keeping `sample_ratio` of requests — 0 disables.
    pub fn new(
        endpoint: String,
        sample_ratio: f64,
        model_name: String,
        provider: Option<TraceProvider>,
    ) -> Arc<Self> {
        let (sender, receiver) = sync_channel(CHANNEL_SPANS);
        let publisher = Self::with_sender(Some((endpoint, model_name)), sample_ratio, sender);
        if let Some((endpoint, model_name)) = publisher.endpoint.clone() {
            thread::Builder::new()
                .name("ts-trace".to_owned())
                .spawn(move || export_loop(receiver, endpoint, model_name, provider))
                .expect("spawning the trace export thread");
        }
        publisher
    }

    /// The publisher over a caller-made channel; the test path (and the
    /// constructor's). `endpoint` starts the export thread only when set.
    fn with_sender(
        endpoint: Option<(String, String)>,
        sample_ratio: f64,
        sender: SyncSender<TraceRecord>,
    ) -> Arc<Self> {
        Arc::new(Self {
            sender,
            sample_ratio: sample_ratio.clamp(0.0, 1.0),
            dropped: AtomicU64::new(0),
            endpoint,
        })
    }

    /// Whether this request is in the sample. The rid decides, the way
    /// OTel's TraceIdRatioBased decides, so both roles of a disaggregated
    /// request reach the same verdict and re-running draws the same.
    pub fn sampled(&self, rid: &str) -> bool {
        if self.sample_ratio >= 1.0 {
            return true;
        }
        if self.sample_ratio <= 0.0 {
            return false;
        }
        let draw = mix(rid_hash(rid, 0xd6e8_feb8_6659_fd93));
        ((draw >> 64) as f64 / u64::MAX as f64) < self.sample_ratio
    }

    /// Hand one span to the export thread, dropping it when the channel is
    /// full — a trace is best effort, never a delay.
    pub fn emit(&self, record: TraceRecord) {
        match self.sender.try_send(record) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                if self.dropped.fetch_add(1, Ordering::Relaxed) % DROPPED_WARN_EVERY
                    == DROPPED_WARN_EVERY - 1
                {
                    let dropped = self.dropped.load(Ordering::Relaxed);
                    tracing::warn!(dropped, "trace spans dropped: the export queue is full");
                }
            }
        }
    }

    /// A publisher over a channel the caller reads, with no export
    /// thread: the tests' path.
    #[cfg(test)]
    pub(crate) fn for_tests(
        sample_ratio: f64,
    ) -> (Arc<Self>, std::sync::mpsc::Receiver<TraceRecord>) {
        let (sender, receiver) = sync_channel(CHANNEL_SPANS);
        (Self::with_sender(None, sample_ratio, sender), receiver)
    }

    /// Spans the export thread could not take.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// The export loop: batch what arrived, POST one OTLP/HTTP JSON request,
/// repeat. A failed POST logs and drops its batch; traces never queue up
/// behind an unreachable collector.
fn export_loop(
    receiver: Receiver<TraceRecord>,
    endpoint: String,
    model_name: String,
    provider: Option<TraceProvider>,
) {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(5))
        .build();
    loop {
        let first = match receiver.recv() {
            Ok(record) => record,
            Err(_) => return,
        };
        let mut batch = vec![first];
        let deadline = Instant::now() + BATCH_WAIT;
        while batch.len() < BATCH_SPANS {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            match receiver.recv_timeout(left) {
                Ok(record) => batch.push(record),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    export(&agent, &endpoint, &model_name, &batch, provider.as_ref());
                    return;
                }
            }
        }
        export(&agent, &endpoint, &model_name, &batch, provider.as_ref());
    }
}

fn apply_trace_context(span: &mut Value, rid: &str, provider: &TraceProvider) {
    let Some((trace_id, parent_span_id)) = provider(rid) else {
        return;
    };
    let valid = |id: &str, size: usize| {
        id.len() == size
            && id.bytes().all(|b| b.is_ascii_hexdigit())
            && id.bytes().any(|b| b != b'0')
    };
    if !valid(&trace_id, 32) || parent_span_id.as_ref().is_some_and(|id| !valid(id, 16)) {
        tracing::warn!("trace provider returned invalid IDs; retaining native context");
        return;
    }
    if parent_span_id.as_ref().is_some_and(|parent| {
        span["spanId"]
            .as_str()
            .is_some_and(|own| parent.eq_ignore_ascii_case(own))
    }) {
        tracing::warn!(
            "trace provider returned the current span as parent; retaining native context"
        );
        return;
    }
    span["traceId"] = Value::String(trace_id.to_ascii_lowercase());
    if let Some(parent_span_id) = parent_span_id {
        span["parentSpanId"] = Value::String(parent_span_id.to_ascii_lowercase());
    }
}

/// One POST. The body is OTLP/HTTP JSON — every OTLP/HTTP receiver takes
/// it — so the export needs no protobuf or opentelemetry dependency.
fn export(
    agent: &ureq::Agent,
    endpoint: &str,
    model_name: &str,
    batch: &[TraceRecord],
    provider: Option<&TraceProvider>,
) {
    let mut request = otlp_request(model_name, batch);
    if let Some(provider) = provider {
        for (span, record) in request["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array_mut()
            .expect("native span array")
            .iter_mut()
            .zip(batch)
        {
            apply_trace_context(span, &record.rid, provider);
        }
    }
    let body = match serde_json::to_vec(&request) {
        Ok(body) => body,
        Err(error) => {
            tracing::warn!(%error, "encoding a trace batch failed");
            return;
        }
    };
    if let Err(error) = agent
        .post(endpoint)
        .set("Content-Type", "application/json")
        .send_bytes(&body)
    {
        tracing::warn!(%error, endpoint, "exporting a trace batch failed");
    }
}

/// The `ExportTraceServiceRequest` one batch becomes. Integers are strings
/// and ids hex, as protobuf's JSON mapping writes 64-bit and bytes fields.
fn otlp_request(model_name: &str, batch: &[TraceRecord]) -> Value {
    let spans: Vec<Value> = batch.iter().map(span).collect();
    json!({
        "resourceSpans": [{
            "resource": {"attributes": [
                string_attribute("service.name", "tokenspeed"),
                string_attribute("gen_ai.system", "tokenspeed"),
                string_attribute("gen_ai.request.model", model_name),
            ]},
            "scopeSpans": [{
                "scope": {"name": "tokenspeed"},
                "spans": spans,
            }],
        }],
    })
}

/// One request as a span: the lifecycle points become time-stamped events,
/// the counts become GenAI-conventioned attributes.
fn span(record: &TraceRecord) -> Value {
    let mut events = Vec::new();
    for (name, at) in [
        ("tokenspeed.request.scheduled", record.scheduled),
        ("tokenspeed.request.prefill_done", record.prefill_done),
        ("tokenspeed.request.first_token", record.first_token),
    ] {
        if at > 0.0 {
            events.push(json!({
                "name": name,
                "timeUnixNano": nanos(at),
            }));
        }
    }
    let mut attributes = vec![
        string_attribute("gen_ai.response.id", &record.rid),
        int_attribute("gen_ai.usage.input_tokens", record.prompt_tokens),
        int_attribute("gen_ai.usage.output_tokens", record.output_tokens),
        int_attribute("tokenspeed.prompt_cached_tokens", record.cached_tokens),
        json!({
            "key": "gen_ai.response.finish_reasons",
            "value": {"arrayValue": {"values": [
                {"stringValue": record.finish_reason}
            ]}}
        }),
        int_attribute("tokenspeed.preempt_count", record.preempt_count),
        double_attribute("tokenspeed.preempt_seconds", record.preempt_seconds),
    ];
    attributes.push(string_attribute(
        "tokenspeed.timing_scope",
        "request_created_to_engine_release",
    ));
    attributes.push(string_attribute(
        "tokenspeed.engine_role",
        match record.role {
            Role::Plain => "plain",
            Role::Prefill => "prefill",
            Role::Decode => "decode",
        },
    ));
    // These are host-observed lifecycle intervals, not GPU execution times.
    // Creation is frontend-supplied; queue time can include frontend preparation
    // and, on D, remote prefill. Neither is a scheduler-only measurement.
    for (name, start, end) in [
        ("tokenspeed.ttft_ms", record.created, record.first_token),
        ("tokenspeed.total_ms", record.created, record.finish),
        ("tokenspeed.queue_ms", record.created, record.scheduled),
    ] {
        if let Some(ms) = elapsed_ms(start, end) {
            attributes.push(double_attribute(name, ms));
        }
    }
    // A D worker does not observe remote prefill dispatch. A P worker's
    // post-token lifetime is KV-transfer waiting, not decode processing.
    if record.role != Role::Decode {
        if let Some(ms) = elapsed_ms(record.scheduled, record.prefill_done) {
            attributes.push(double_attribute("tokenspeed.prefill_process_ms", ms));
        }
    }
    if record.role != Role::Prefill {
        if let Some(ms) = elapsed_ms(record.first_token, record.finish) {
            attributes.push(double_attribute("tokenspeed.decode_process_ms", ms));
            if record.output_tokens > 1 {
                attributes.push(double_attribute(
                    "tokenspeed.tpot_ms",
                    ms / (record.output_tokens - 1) as f64,
                ));
            }
        }
    }
    if let Some(session) = &record.session {
        // The GenAI session convention: same id, one conversation in the
        // tracing UI, each request its own trace.
        attributes.push(string_attribute("session.id", session));
    }
    // A request the engine serves is a SERVER span, and one that aborted
    // or failed its transfer is an error trace the UI surfaces.
    let error = matches!(record.finish_reason, "abort" | "unknown");
    json!({
        "traceId": trace_id(&record.rid),
        "spanId": span_id(&record.rid),
        "name": "tokenspeed.request",
        "kind": 2,
        "startTimeUnixNano": nanos(record.created),
        "endTimeUnixNano": nanos(record.finish),
        "attributes": attributes,
        "events": events,
        "status": if error {
            json!({"code": 2, "message": record.finish_reason})
        } else {
            json!({})
        },
    })
}

/// Zero marks an unreached boundary; clock reversals must not become zero latency.
fn elapsed_ms(start: f64, end: f64) -> Option<f64> {
    let ms = (end - start) * 1000.0;
    (start.is_finite() && end.is_finite() && start > 0.0 && end >= start && ms.is_finite())
        .then_some(ms)
}

fn string_attribute(key: &str, value: &str) -> Value {
    json!({"key": key, "value": {"stringValue": value}})
}

fn int_attribute(key: &str, value: i64) -> Value {
    json!({"key": key, "value": {"intValue": value.to_string()}})
}

fn double_attribute(key: &str, value: f64) -> Value {
    json!({"key": key, "value": {"doubleValue": value}})
}

/// Wall-clock seconds as nanoseconds since the epoch, the string proto
/// JSON gives 64-bit fields.
fn nanos(seconds: f64) -> String {
    (seconds.max(0.0) * 1e9).round().to_string()
}

/// The request's trace id: 32 hex characters from a 128-bit mix of the rid,
/// so the same request id traces to the same id across ranks and logs.
fn trace_id(rid: &str) -> String {
    let mut out = String::with_capacity(32);
    for salt in [0xa076_1d64_78bd_642f, 0xe703_7ed1_a0b4_28db] {
        let half = mix(rid_hash(rid, salt));
        out.push_str(&format!("{:016x}", half as u64));
    }
    out
}

/// The request's span id, 16 hex characters: the same mix over the rid and
/// the process counter, so every span differs.
fn span_id(rid: &str) -> String {
    let salt = 0x8ebc_6af0_9c88_c6e3_u128;
    let counter = u128::from(SPANS.fetch_add(1, Ordering::Relaxed));
    format!("{:016x}", (mix(rid_hash(rid, salt) ^ counter) as u64))
}

/// `rid` folded to 128 bits; `salt` separates the derivations. It seeds the
/// low bits — a wrapping multiply never mixes high bits down into the halves
/// a u64 read takes.
fn rid_hash(rid: &str, salt: u128) -> u128 {
    let mut hash: u128 = 0x243f_6a88_85a3_08d3_1319_8a2e_0370_7344 ^ salt;
    for byte in rid.as_bytes() {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834);
    }
    hash
}

/// The final scramble (splitmix64's, over 128 bits).
fn mix(value: u128) -> u128 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9_b495_b30f_b1d3_0a7e);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb_8c9f_b3d9_c764_51a3);
    z ^ (z >> 31)
}

/// The process-wide span counter.
static SPANS: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> TraceRecord {
        TraceRecord {
            rid: "chatcmpl-1".to_owned(),
            role: Role::Plain,
            session: Some("sess-7".to_owned()),
            created: 1_700_000_000.0,
            scheduled: 1_700_000_000.5,
            prefill_done: 1_700_000_001.0,
            first_token: 1_700_000_001.25,
            finish: 1_700_000_002.0,
            prompt_tokens: 100,
            cached_tokens: 64,
            output_tokens: 8,
            finish_reason: "stop",
            preempt_count: 1,
            preempt_seconds: 0.25,
        }
    }

    #[test]
    fn parent_context_preserves_span_identity_and_lifecycle() {
        let mut value = span(&record());
        let before = value.clone();
        let provider: TraceProvider = Arc::new(|_| {
            Some((
                "ABCDEF0123456789ABCDEF0123456789".into(),
                Some("ABCDEF0123456789".into()),
            ))
        });
        apply_trace_context(&mut value, "rid", &provider);
        assert_eq!(value["traceId"], "abcdef0123456789abcdef0123456789");
        assert_eq!(value["parentSpanId"], "abcdef0123456789");
        value.as_object_mut().unwrap().remove("parentSpanId");
        value["traceId"] = before["traceId"].clone();
        assert_eq!(value, before);
    }

    #[test]
    fn provider_can_generate_a_root_trace_without_a_parent() {
        let mut value = span(&record());
        let provider: TraceProvider = Arc::new(|_| Some(("a".repeat(32), None)));
        apply_trace_context(&mut value, "ordinary-rid", &provider);
        assert_eq!(value["traceId"], "a".repeat(32));
        assert!(value.get("parentSpanId").is_none());
    }

    #[test]
    fn a_span_cannot_be_its_own_parent() {
        let mut value = span(&record());
        let before = value.clone();
        let own = value["spanId"].as_str().unwrap().to_ascii_uppercase();
        let provider: TraceProvider = Arc::new(move |_| Some(("a".repeat(32), Some(own.clone()))));
        apply_trace_context(&mut value, "rid", &provider);
        assert_eq!(value, before);
    }

    #[test]
    fn missing_or_invalid_context_preserves_native_span() {
        let before = span(&record());
        for context in [
            None,
            Some(("bad".into(), Some("1234567890abcdef".into()))),
            Some(("0".repeat(32), Some("1234567890abcdef".into()))),
            Some(("1".repeat(32), Some("0".repeat(16)))),
            Some(("1".repeat(32), Some("g".repeat(16)))),
        ] {
            let provider: TraceProvider = Arc::new(move |_| context.clone());
            let mut value = before.clone();
            apply_trace_context(&mut value, "rid", &provider);
            assert_eq!(value, before);
        }
    }

    #[test]
    fn a_record_becomes_a_genai_conventioned_span() {
        let span = span(&record());
        assert_eq!(span["name"], "tokenspeed.request");
        assert_eq!(span["kind"], 2);
        assert_eq!(span["status"], serde_json::json!({}));
        assert_eq!(span["traceId"], trace_id(&record().rid));
        assert!(span.get("parentSpanId").is_none());
        assert_eq!(span["spanId"].as_str().unwrap().len(), 16);
        assert_eq!(span["startTimeUnixNano"], "1700000000000000000");
        assert_eq!(span["endTimeUnixNano"], "1700000002000000000");
        let attrs = span["attributes"].as_array().unwrap();
        assert!(attrs.contains(&json!({
            "key": "gen_ai.response.finish_reasons",
            "value": {"arrayValue": {"values": [{"stringValue": "stop"}]}}
        })));
        assert!(attrs.contains(&string_attribute("session.id", "sess-7")));
        assert!(attrs.contains(&int_attribute("gen_ai.usage.input_tokens", 100)));
        assert!(attrs.contains(&int_attribute("gen_ai.usage.output_tokens", 8)));
        let events = span["events"].as_array().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0]["name"], "tokenspeed.request.scheduled");
        assert_eq!(events[0]["timeUnixNano"], "1700000000500000000");
    }

    fn timing(value: &Value, name: &str) -> Option<f64> {
        value["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|attribute| attribute["key"] == format!("tokenspeed.{name}_ms"))
            .and_then(|attribute| attribute["value"]["doubleValue"].as_f64())
    }

    #[test]
    fn span_reports_host_lifecycle_durations() {
        let value = span(&record());
        for (name, expected) in [
            ("ttft", 1250.0),
            ("total", 2000.0),
            ("queue", 500.0),
            ("prefill_process", 500.0),
            ("decode_process", 750.0),
            ("tpot", 750.0 / 7.0),
        ] {
            assert_eq!(timing(&value, name), Some(expected), "{name}");
        }
        assert!(value["attributes"]
            .as_array()
            .unwrap()
            .contains(&string_attribute(
                "tokenspeed.timing_scope",
                "request_created_to_engine_release",
            )));
    }

    #[test]
    fn phase_durations_follow_the_engine_role() {
        let mut record = record();
        for (role, name) in [(Role::Prefill, "prefill"), (Role::Decode, "decode")] {
            record.role = role;
            let value = span(&record);
            assert!(value["attributes"]
                .as_array()
                .unwrap()
                .contains(&string_attribute("tokenspeed.engine_role", name),));
            assert_eq!(timing(&value, "total"), Some(2000.0));
            assert_eq!(timing(&value, "ttft"), Some(1250.0));
            assert_eq!(timing(&value, "queue"), Some(500.0));
            if role == Role::Prefill {
                assert_eq!(timing(&value, "prefill_process"), Some(500.0));
                assert_eq!(timing(&value, "decode_process"), None);
                assert_eq!(timing(&value, "tpot"), None);
            } else {
                assert_eq!(timing(&value, "prefill_process"), None);
                assert_eq!(timing(&value, "decode_process"), Some(750.0));
                assert_eq!(timing(&value, "tpot"), Some(750.0 / 7.0));
            }
        }
    }

    #[test]
    fn abort_before_dispatch_reports_only_total_duration() {
        let mut record = record();
        record.scheduled = 0.0;
        record.prefill_done = 0.0;
        record.first_token = 0.0;
        record.output_tokens = 0;
        record.finish_reason = "abort";
        let value = span(&record);
        assert_eq!(timing(&value, "total"), Some(2000.0));
        for name in ["ttft", "queue", "prefill_process", "decode_process", "tpot"] {
            assert_eq!(timing(&value, name), None, "{name}");
        }
    }

    #[test]
    fn tpot_requires_more_than_one_output_token() {
        for count in [0, 1] {
            let mut record = record();
            record.output_tokens = count;
            assert_eq!(timing(&span(&record), "tpot"), None);
        }
        let mut record = record();
        record.finish = record.first_token;
        assert_eq!(timing(&span(&record), "tpot"), Some(0.0));
    }

    #[test]
    fn invalid_boundaries_omit_only_affected_durations() {
        for invalid in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1_700_000_003.0,
        ] {
            let mut record = record();
            record.created = invalid;
            let value = span(&record);
            for name in ["ttft", "total", "queue"] {
                assert_eq!(timing(&value, name), None, "{name}: {invalid}");
            }
            assert_eq!(timing(&value, "decode_process"), Some(750.0));
        }
        let mut record = record();
        record.finish = record.first_token - 0.25;
        let value = span(&record);
        assert_eq!(timing(&value, "decode_process"), None);
        assert_eq!(timing(&value, "tpot"), None);
        assert_eq!(timing(&value, "ttft"), Some(1250.0));
    }

    #[test]
    fn a_request_without_a_session_carries_no_session_attribute() {
        let mut record = record();
        record.session = None;
        let span = span(&record);
        let attrs = span["attributes"].as_array().unwrap();
        assert!(!attrs
            .iter()
            .any(|attribute| attribute["key"] == "session.id"));
    }

    #[test]
    fn unreached_points_emit_no_events() {
        let mut record = record();
        record.scheduled = 0.0;
        record.prefill_done = 0.0;
        record.first_token = 0.0;
        let span = span(&record);
        assert_eq!(span["events"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn the_same_rid_traces_to_the_same_id() {
        assert_eq!(trace_id("chatcmpl-1"), trace_id("chatcmpl-1"));
        assert_ne!(trace_id("chatcmpl-1"), trace_id("chatcmpl-2"));
        assert_ne!(trace_id("chatcmpl-1"), span_id("chatcmpl-1"));
        // The two halves derive independently — a repeated half would halve
        // the id's entropy.
        let id = trace_id("chatcmpl-1");
        assert_ne!(&id[..16], &id[16..], "the trace id's halves differ");
    }

    #[test]
    fn a_full_channel_drops_instead_of_blocking() {
        let (sender, receiver) = sync_channel(1);
        let publisher = TracePublisher::with_sender(None, 1.0, sender);
        publisher.emit(record());
        publisher.emit(record());
        assert_eq!(publisher.dropped(), 1);
        assert!(receiver.recv_timeout(Duration::from_millis(10)).is_ok());
    }

    #[test]
    fn sampling_decides_by_the_request_id() {
        let (sender, _) = sync_channel(CHANNEL_SPANS);
        let off = TracePublisher::with_sender(None, 0.0, sender.clone());
        assert!(!off.sampled("chatcmpl-1"));
        let all = TracePublisher::with_sender(None, 1.0, sender.clone());
        for index in 0..100 {
            assert!(all.sampled(&format!("r{index}")));
        }
        // A ratio between: the kept share holds, and one request's verdict
        // is stable — both roles of a disaggregated request agree.
        let (sender, _) = sync_channel(CHANNEL_SPANS);
        let half = TracePublisher::with_sender(None, 0.5, sender);
        let kept: usize = (0..1000)
            .filter(|index| half.sampled(&format!("r{index}")))
            .count();
        assert!((350..=650).contains(&kept), "kept {kept} of 1000");
        assert_eq!(half.sampled("chatcmpl-1"), half.sampled("chatcmpl-1"));
    }

    /// The full export path against a real socket: the export thread, its
    /// batching, ureq's POST and the OTLP/JSON body, as a receiver sees
    /// them.
    #[test]
    fn a_live_export_posts_one_otlp_json_batch() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        let received = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buffer = [0u8; 8192];
            let mut head = Vec::new();
            // Read until the header's end; a peer that closes early fails
            // the test instead of hanging it.
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).expect("read the head");
                assert!(read > 0, "the peer closed before the header ended");
                head.extend_from_slice(&buffer[..read]);
            }
            let head = String::from_utf8_lossy(&head).to_string();
            let length: usize = head
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .expect("content length")
                .trim()
                .parse()
                .expect("a number");
            let header_end = head.find("\r\n\r\n").expect("the end") + 4;
            let mut body = head.as_bytes()[header_end..].to_vec();
            while body.len() < length {
                let n = stream.read(&mut buffer).expect("read the body");
                body.extend_from_slice(&buffer[..n]);
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .expect("reply");
            (head, body)
        });

        let publisher = TracePublisher::new(
            format!("http://{address}/v1/traces"),
            1.0,
            "m".to_owned(),
            Some(Arc::new(|rid| {
                assert_eq!(rid, "chatcmpl-1");
                Some(("1".repeat(32), Some("2".repeat(16))))
            })),
        );
        publisher.emit(record());
        let (head, body) = received.join().expect("the receiver thread");

        let request_line = head.lines().next().expect("the request line");
        assert!(
            request_line.starts_with("POST /v1/traces "),
            "{request_line}"
        );
        assert!(
            head.contains("content-type: application/json")
                || head.contains("Content-Type: application/json")
        );
        let batch: Value = serde_json::from_slice(&body).expect("the OTLP body");
        let spans = &batch["resourceSpans"][0]["scopeSpans"][0]["spans"];
        let span = &spans[0];
        assert_eq!(timing(span, "ttft"), Some(1250.0));
        assert_eq!(timing(span, "total"), Some(2000.0));
        assert_eq!(timing(span, "tpot"), Some(750.0 / 7.0));
        assert_eq!(timing(span, "queue"), Some(500.0));
        assert_eq!(timing(span, "prefill_process"), Some(500.0));
        assert_eq!(timing(span, "decode_process"), Some(750.0));
        assert_eq!(span["name"], "tokenspeed.request");
        assert_eq!(span["kind"], 2);
        assert_eq!(span["status"], serde_json::json!({}));
        assert_eq!(span["traceId"], "1".repeat(32));
        assert_eq!(span["parentSpanId"], "2".repeat(16));
        let attributes = span["attributes"].as_array().unwrap();
        assert!(attributes.contains(&string_attribute("gen_ai.response.id", "chatcmpl-1")));
        assert!(attributes.contains(&int_attribute("gen_ai.usage.input_tokens", 100)));
        assert!(attributes.contains(&int_attribute("gen_ai.usage.output_tokens", 8)));
        let resource = &batch["resourceSpans"][0]["resource"]["attributes"];
        assert!(resource
            .as_array()
            .unwrap()
            .contains(&string_attribute("gen_ai.request.model", "m")));
    }

    #[test]
    fn the_request_wraps_spans_in_one_resource() {
        let body = otlp_request("m", &[record()]);
        let resource = &body["resourceSpans"][0]["resource"]["attributes"];
        assert!(resource
            .as_array()
            .unwrap()
            .contains(&string_attribute("gen_ai.request.model", "m")));
        let spans = &body["resourceSpans"][0]["scopeSpans"][0]["spans"];
        assert_eq!(spans.as_array().unwrap().len(), 1);
    }
}
