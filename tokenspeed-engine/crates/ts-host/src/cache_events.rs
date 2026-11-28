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

//! Opt-in, bounded cache ownership and physical-copy observations.
//!
//! This is a diagnostic sideband, never scheduler feedback. Both producers
//! share the host's existing nonblocking log writer and immutable identity.

use std::{
    io::{self, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Mutex, OnceLock, TryLockError,
    },
};

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyString},
};
use serde_json::{json, Map, Value};
use tracing_appender::non_blocking::NonBlocking;
use ts_scheduler::{
    cache_events::MAX_CACHE_EVENT_REQUEST_BYTES, CacheEventObserver, CacheObservation,
    CacheObservationKind, CacheObservationLosses, Scheduler,
};

const MAX_INSTANCE_BYTES: usize = 128;
const MAX_RECORD_BYTES: usize = 8192;
const MAX_COMPLETIONS: usize = 4096;
const MAX_PENDING_RECORDS: usize = 4096;

#[derive(Clone)]
pub(crate) struct Settings {
    instance: String,
}

impl Settings {
    pub(crate) fn from_env(async_logging: bool) -> Result<Option<Self>, String> {
        let read = |name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(_) => Err(format!("{name} must be valid UTF-8")),
        };
        let enabled = read("TOKENSPEED_CACHE_EVENT_LOG")?;
        if enabled.as_deref() != Some("1") {
            return Self::parse(enabled.as_deref(), None, async_logging);
        }
        Self::parse(
            enabled.as_deref(),
            read("TOKENSPEED_CACHE_EVENT_INSTANCE")?.as_deref(),
            async_logging,
        )
    }

    fn parse(
        enabled: Option<&str>,
        instance: Option<&str>,
        async_logging: bool,
    ) -> Result<Option<Self>, String> {
        match enabled {
            None | Some("0") => return Ok(None),
            Some("1") => (),
            _ => return Err("TOKENSPEED_CACHE_EVENT_LOG must be 0 or 1".to_owned()),
        }
        if !async_logging {
            return Err("TOKENSPEED_CACHE_EVENT_LOG=1 requires --enable-async-logging".to_owned());
        }
        let instance = instance.filter(|value| {
            !value.is_empty() && value.len() <= MAX_INSTANCE_BYTES
                && value.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        }).ok_or_else(|| "TOKENSPEED_CACHE_EVENT_INSTANCE must contain 1..128 ASCII letters, digits, '.', '_' or '-'".to_owned())?;
        Ok(Some(Self {
            instance: instance.to_owned(),
        }))
    }
}

/// Query the linked native scheduler, without bootstrapping a model.
pub(crate) fn protocol() -> String {
    format!(
        "tokenspeed-cache-events-v{}",
        ts_scheduler::cache_event_protocol_version()
    )
}

/// The scheduler's sink owns a clone until its last slot has been released.
/// The final clone writes a terminal summary before the host drains logging.
#[derive(Clone)]
pub(crate) struct CacheEvents(Arc<Context>);

struct Context {
    instance: String,
    role: &'static str,
    dp_rank: i64,
    physical_rank: usize,
    writer: NonBlocking,
    pending: mpsc::SyncSender<Value>,
    publication: Mutex<Publication>,
    pending_losses: AtomicU64,
    lost_observations: AtomicU64,
    reported_log_losses: AtomicUsize,
    native_losses: OnceLock<CacheObservationLosses>,
    execution_enabled: AtomicBool,
}

/// One publisher assigns sequences and enqueues complete lines in that order.
/// Producers only try-send/try-lock: a concurrent native destructor must not
/// wait on another callback, Python, or output I/O.
struct Publication {
    pending: mpsc::Receiver<Value>,
    next_sequence: u64,
    execution_sequence: u64,
    publication_order: u64,
}

impl CacheEvents {
    pub(crate) fn install(
        settings: Settings,
        engine_role: &str,
        dp_rank: i64,
        physical_rank: usize,
        writer: NonBlocking,
        scheduler: &mut Scheduler,
    ) -> Result<Self, String> {
        let role = match engine_role {
            "null" => "fused",
            "prefill" => "prefill",
            "decode" => "decode",
            _ => {
                return Err("cache observations require a fused, prefill or decode host".to_owned())
            }
        };
        if dp_rank < 0 {
            return Err("cache observations require a nonnegative DP rank".to_owned());
        }
        let (pending, records) = mpsc::sync_channel(MAX_PENDING_RECORDS);
        let context = Self(Arc::new(Context {
            instance: settings.instance,
            role,
            dp_rank,
            physical_rank,
            writer,
            pending,
            publication: Mutex::new(Publication {
                pending: records,
                next_sequence: 0,
                execution_sequence: 0,
                publication_order: 0,
            }),
            pending_losses: AtomicU64::new(0),
            lost_observations: AtomicU64::new(0),
            reported_log_losses: AtomicUsize::new(0),
            native_losses: OnceLock::new(),
            execution_enabled: AtomicBool::new(false),
        }));
        let sink = context.clone();
        let (observer, losses) = CacheEventObserver::new(move |event, request_id| {
            sink.0.native(event, request_id);
        });
        let _ = context.0.native_losses.set(losses);
        scheduler
            .install_cache_event_observer(observer)
            .map_err(|error| error.to_string())?;
        context
            .0
            .emit(json!({"event": "trace_started", "scope": "physical_rank"}));
        Ok(context)
    }

    /// Shares publication ordering and loss accounting with cache observations.
    pub(crate) fn execution_sink(&self) -> Arc<dyn ts_control::execution_observation::Sink> {
        assert!(!self.0.execution_enabled.swap(true, Ordering::Relaxed));
        self.0.emit_execution(json!({"event": "trace_started"}));
        Arc::new(ExecutionSink(self.clone()))
    }

    /// Called under the ordinary completion query's GIL, after its ACK poll.
    /// Invalid diagnostics invalidate evidence without swallowing those ACKs.
    pub(crate) fn poll(&self, plane: &Bound<'_, PyAny>) {
        self.0.report_losses();
        if self.poll_python(plane).is_err() {
            self.0.loss("invalid_completion_poll", 1);
        }
    }

    fn poll_python(&self, plane: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = plane.call_method0("poll_cache_observations")?;
        let (records, dropped): (Bound<'_, PyList>, u64) = result.extract()?;
        if dropped > 0 {
            self.0.loss("python_observation_overflow", dropped);
        }
        if records.len() > MAX_COMPLETIONS {
            self.0
                .loss("completion_batch_too_large", records.len() as u64);
            return Ok(());
        }
        for record in records.iter() {
            match completion(&record) {
                Ok(record) => self.0.emit(record),
                Err(_) => self.0.loss("invalid_completion_record", 1),
            }
        }
        Ok(())
    }
}

struct ExecutionSink(CacheEvents);

impl std::fmt::Debug for ExecutionSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ExecutionSink")
    }
}

impl ts_control::execution_observation::Sink for ExecutionSink {
    fn emit(&self, record: Value) {
        if record["event"] == "observation_loss" {
            self.0 .0.lost_observations.fetch_add(1, Ordering::Relaxed);
        }
        self.0 .0.emit_execution(record);
    }
    fn context(&self) -> ts_control::execution_observation::Context {
        ts_control::execution_observation::Context {
            instance_id: self.0 .0.instance.clone(),
            role: self.0 .0.role.into(),
            dp_rank: self.0 .0.dp_rank,
            physical_rank: self.0 .0.physical_rank as u64,
        }
    }
    fn valid(&self) -> bool {
        self.0 .0.lost_observations.load(Ordering::Relaxed) == 0
            && self.0 .0.writer.error_counter().dropped_lines() == 0
    }
}

impl Context {
    fn emit_execution(&self, mut record: Value) {
        record["_execution"] = true.into();
        record["scope"] = "local_physical".into();
        self.emit(record);
    }

    fn native(&self, event: &CacheObservation, request_id: &str) {
        if event.admission_id == 0 || event.slot < 0 || event.slot_generation == 0 {
            self.loss("invalid_native_identity", 1);
            return;
        }
        let (name, kind) = match event.kind {
            CacheObservationKind::SlotAssigned => ("slot_assigned", "slot"),
            CacheObservationKind::SlotReleased => ("slot_released", "slot"),
            CacheObservationKind::RestoreScheduled => ("restore_scheduled", "snapshot_restore"),
            CacheObservationKind::PrefixLoadScheduled => ("prefix_load_scheduled", "prefix_load"),
            _ => {
                self.loss("unknown_native_kind", 1);
                return;
            }
        };
        let mut record = json!({
            "event": name, "kind": kind, "scope": "scheduler_replica",
            "request_id": request_id, "admission_id": event.admission_id,
            "slot": event.slot, "slot_generation": event.slot_generation,
        });
        if kind == "slot" {
            record["slot_previous_admission_id"] = event.slot_previous_admission_id.into();
        } else {
            record["op_id"] = event.op_id.into();
            record["op_generation"] = event.op_generation.into();
        }
        if event.kind == CacheObservationKind::RestoreScheduled {
            record["previous_slot"] = event.previous_slot.into();
            record["previous_slot_generation"] = event.previous_slot_generation.into();
            record["slot_previous_admission_id"] = event.slot_previous_admission_id.into();
            record["restore_sequence"] = event.restore_sequence.into();
            record["committed_tokens"] = event.committed_tokens.into();
            record["computed_tokens"] = event.computed_tokens.into();
            record["prompt_tokens"] = event.prompt_tokens.into();
        }
        self.emit(record);
    }

    fn loss(&self, reason: &'static str, count: u64) {
        self.lost_observations.fetch_add(count, Ordering::Relaxed);
        self.emit(json!({
            "event": "observation_loss", "scope": "physical_rank",
            "reason": reason, "dropped_count": count, "evidence_valid": false,
        }));
    }

    fn report_losses(&self) {
        self.publish_pending();
        let pending_losses = self.pending_losses.swap(0, Ordering::Relaxed);
        if pending_losses > 0 {
            // Counted at rejection even if this loss record is itself dropped.
            self.emit(json!({
                "event": "observation_loss", "scope": "physical_rank",
                "reason": "publication_queue_overflow", "dropped_count": pending_losses,
                "evidence_valid": false,
            }));
        }
        if let Some(losses) = self.native_losses.get() {
            let lost = losses.take();
            if lost > 0 {
                self.loss("native_observation_rejected", lost);
            }
        }
        let lost = self.writer.error_counter().dropped_lines();
        let previous = self.reported_log_losses.swap(lost, Ordering::Relaxed);
        if lost > previous {
            self.loss("log_queue_overflow", (lost - previous) as u64);
        }
    }

    fn emit(&self, record: Value) {
        if self.pending.try_send(record).is_err() {
            self.lost_observations.fetch_add(1, Ordering::Relaxed);
            self.pending_losses.fetch_add(1, Ordering::Relaxed);
        }
        self.publish_pending();
    }

    fn publish_pending(&self) {
        let mut publication = match self.publication.try_lock() {
            Ok(publication) => publication,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        // Bound each callback's work even if other producers keep arriving.
        // A poll or the last owner's Drop drains any records left behind.
        for _ in 0..MAX_PENDING_RECORDS {
            let Ok(record) = publication.pending.try_recv() else {
                break;
            };
            let execution = record["_execution"] == true;
            let sequence = if execution {
                let value = publication.execution_sequence;
                publication.execution_sequence += 1;
                value
            } else {
                let value = publication.next_sequence;
                publication.next_sequence += 1;
                value
            };
            let order = publication.publication_order;
            publication.publication_order += 1;
            self.write_record(record, sequence, order, execution);
        }
    }

    fn write_record(&self, mut record: Value, sequence: u64, order: u64, execution: bool) {
        record
            .as_object_mut()
            .expect("observation object")
            .remove("_execution");
        record["protocol"] = if execution {
            ts_control::execution_observation::PROTOCOL.into()
        } else {
            protocol().into()
        };
        record["publication_order"] = order.into();
        record["instance_id"] = self.instance.as_str().into();
        record["role"] = self.role.into();
        record["dp_rank"] = self.dp_rank.into();
        record["physical_rank"] = self.physical_rank.into();
        record["record_sequence"] = sequence.into();
        record["log_dropped_lines"] = self.writer.error_counter().dropped_lines().into();
        let mut buffer = RecordBuffer(Vec::with_capacity(MAX_RECORD_BYTES));
        // Only fixed scalar fields and bounded identity strings enter this
        // writer. Even a future schema mistake cannot emit a partial object.
        let encoded = buffer
            .write_all(if execution {
                b"[execution-event] "
            } else {
                b"[cache-event] "
            })
            .and_then(|()| serde_json::to_writer(&mut buffer, &record).map_err(io::Error::other))
            .and_then(|()| buffer.write_all(b"\n"));
        if encoded.is_err() {
            // This fixed-size loss record always fits the same bound. Keep
            // the skipped sequence too, in case the loss record is dropped.
            self.loss("record_too_large", 1);
            return;
        }
        if self.writer.clone().write_all(&buffer.0).is_err() {
            self.lost_observations.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.report_losses();
        if self.execution_enabled.load(Ordering::Relaxed) {
            self.emit_execution(json!({"event": "trace_finished",
                "lost_observations": self.lost_observations.load(Ordering::Relaxed),
                "evidence_valid": self.lost_observations.load(Ordering::Relaxed) == 0
                    && self.writer.error_counter().dropped_lines() == 0}));
        }
        self.emit(json!({
            "event": "trace_finished", "scope": "physical_rank",
            "lost_observations": self.lost_observations.load(Ordering::Relaxed),
            "evidence_valid": self.lost_observations.load(Ordering::Relaxed) == 0
                && self.writer.error_counter().dropped_lines() == 0,
        }));
    }
}

struct RecordBuffer(Vec<u8>);

impl Write for RecordBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_RECORD_BYTES - self.0.len() {
            return Err(io::Error::other("cache event exceeds record bound"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn required<'py>(record: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    record
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err("missing cache observation field"))
}

/// Copy only known scalar fields, never an arbitrary Python JSON object.
fn completion(record: &Bound<'_, PyAny>) -> PyResult<Value> {
    let record = record.cast::<PyDict>()?;
    let kind = required(record, "kind")?;
    let kind = kind.cast::<PyString>()?;
    if kind.len()? > 32 {
        return Err(PyValueError::new_err("invalid cache observation kind"));
    }
    let kind = kind.to_str()?;
    let event = match kind {
        "snapshot_restore" => "restore_completed",
        "prefix_load" => "prefix_load_completed",
        _ => return Err(PyValueError::new_err("invalid cache observation kind")),
    };
    let rid = required(record, "request_id")?;
    let rid = rid.cast::<PyString>()?;
    if rid.len()? > MAX_CACHE_EVENT_REQUEST_BYTES {
        return Err(PyValueError::new_err("invalid cache observation identity"));
    }
    let rid = rid.to_str()?;
    if rid.is_empty() || rid.len() > MAX_CACHE_EVENT_REQUEST_BYTES {
        return Err(PyValueError::new_err("invalid cache observation identity"));
    }
    let slot: i32 = required(record, "request_pool_index")?.extract()?;
    let admission: u64 = required(record, "admission_id")?.extract()?;
    let generation: u64 = required(record, "slot_generation")?.extract()?;
    if slot < 0 || admission == 0 || generation == 0 {
        return Err(PyValueError::new_err("invalid cache observation ownership"));
    }
    let op_id: u32 = required(record, "op_id")?.extract()?;
    let mut result = Map::new();
    result.insert("event".into(), event.into());
    result.insert("scope".into(), "local_physical".into());
    result.insert("kind".into(), kind.into());
    result.insert("request_id".into(), rid.into());
    result.insert("slot".into(), slot.into());
    result.insert("admission_id".into(), admission.into());
    result.insert("slot_generation".into(), generation.into());
    result.insert("op_id".into(), op_id.into());
    for key in [
        "op_generation",
        "l2_rows",
        "l2_bytes",
        "pool_rows",
        "pool_bytes",
        "slot_state_image_bytes",
    ] {
        let value: u64 = required(record, key)?.extract()?;
        result.insert(key.into(), value.into());
    }
    if kind == "snapshot_restore" {
        for key in [
            "previous_slot_generation",
            "slot_previous_admission_id",
            "restore_sequence",
        ] {
            let value: u64 = required(record, key)?.extract()?;
            result.insert(key.into(), value.into());
        }
        for key in [
            "previous_slot",
            "committed_tokens",
            "computed_tokens",
            "prompt_tokens",
        ] {
            let value: i32 = required(record, key)?.extract()?;
            if value < 0 {
                return Err(PyValueError::new_err("invalid cache observation frontier"));
            }
            result.insert(key.into(), value.into());
        }
    }
    // Identity/context fields supplied by Python must not shadow host labels.
    let expected_fields = if kind == "snapshot_restore" { 19 } else { 12 };
    if record.len() != expected_fields {
        return Err(PyValueError::new_err("unexpected cache observation fields"));
    }
    Ok(result.into())
}

#[cfg(test)]
#[path = "tests/cache_events.rs"]
mod tests;
