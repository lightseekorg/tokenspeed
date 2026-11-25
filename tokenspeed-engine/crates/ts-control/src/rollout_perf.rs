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

//! Request boundaries returned to the frontend, independently of logging.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use ts_wire::{Dict, Value};

use crate::request::RequestState;

struct Clock {
    origin: Instant,
    host_id: String,
    process_instance_id: String,
}

fn clock() -> &'static Clock {
    static CLOCK: OnceLock<Clock> = OnceLock::new();
    CLOCK.get_or_init(|| Clock {
        origin: Instant::now(),
        host_id: std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .or_else(|_| std::fs::read_to_string("/etc/hostname"))
            .map(|host| host.trim().to_owned())
            .unwrap_or_else(|_| "unknown".to_owned()),
        process_instance_id: format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ),
    })
}

/// One observed boundary. Monotonic values are comparable only within this process.
#[derive(Debug, Clone)]
pub(crate) struct EventStamp {
    wall: SystemTime,
    mono: Instant,
    seq: u64,
}

impl EventStamp {
    pub(crate) fn now() -> Self {
        // Establish the process origin before sampling any local boundary.
        clock();
        Self::at(SystemTime::now(), Instant::now())
    }

    pub(crate) fn at(wall: SystemTime, mono: Instant) -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        Self {
            wall,
            mono,
            seq: SEQUENCE.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub(crate) fn wall_seconds(&self) -> f64 {
        self.wall
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    fn value(&self) -> Value {
        let clock = clock();
        map([
            (
                "event_id",
                Value::Str(format!("{}:{}", clock.process_instance_id, self.seq)),
            ),
            (
                "ts_unix_ns",
                Value::Str(
                    self.wall
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                        .to_string(),
                ),
            ),
            (
                "ts_mono_ns",
                self.mono
                    .checked_duration_since(clock.origin)
                    .map_or(Value::Nil, |time| Value::Str(time.as_nanos().to_string())),
            ),
            ("host_id", Value::Str(clock.host_id.clone())),
            (
                "process_instance_id",
                Value::Str(clock.process_instance_id.clone()),
            ),
            ("source_seq", Value::Str(self.seq.to_string())),
            ("clock_sync", Value::from("unknown")),
            ("clock_uncertainty_ns", Value::Nil),
        ])
    }
}

fn map<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::from(key), value))
            .collect(),
    )
}

/// The request owns its observed boundaries until its terminal output is assembled.
#[derive(Debug, Clone, Default)]
pub(crate) struct RequestPerf {
    pub(crate) accepted: Option<EventStamp>,
    kv_ready: Option<EventStamp>,
    kv_output_tokens: Option<usize>,
}

impl RequestPerf {
    /// Record readiness once; retraction or later chunks never move the first boundary.
    pub(crate) fn kv_ready(&mut self, at: EventStamp, output_tokens: usize) {
        if self.kv_ready.is_none() {
            self.kv_ready = Some(at);
            self.kv_output_tokens = Some(output_tokens);
        }
    }

    /// Build the terminal wire metadata, retaining missing boundaries as null.
    pub(crate) fn finish(
        &self,
        rid: &str,
        role: &str,
        ranks: Option<(i64, i64)>,
        state: &RequestState,
        finished: &EventStamp,
    ) -> Dict<String, Value> {
        let optional_stamp =
            |stamp: &Option<EventStamp>| stamp.as_ref().map_or(Value::Nil, EventStamp::value);
        Dict(vec![
            ("wire_request_id".to_owned(), Value::from(rid)),
            ("role".to_owned(), Value::from(role)),
            (
                "dp_rank".to_owned(),
                ranks.map_or(Value::Nil, |r| Value::from(r.0)),
            ),
            (
                "tp_rank".to_owned(),
                ranks.map_or(Value::Nil, |r| Value::from(r.1)),
            ),
            (
                "events".to_owned(),
                map([
                    ("accepted", optional_stamp(&self.accepted)),
                    ("kv_ready", optional_stamp(&self.kv_ready)),
                    ("finished", finished.value()),
                ]),
            ),
            (
                "kv_output_tokens".to_owned(),
                self.kv_output_tokens
                    .map_or(Value::Nil, |n| Value::from(n as i64)),
            ),
            (
                "completion_tokens".to_owned(),
                Value::from(state.output_ids.len() as i64),
            ),
            (
                "finish_reason".to_owned(),
                state
                    .finished_reason
                    .as_ref()
                    .map_or(Value::Nil, |reason| Value::from(reason.wire_type())),
            ),
            (
                "e2e_latency_s".to_owned(),
                if state.created_time > 0.0 {
                    Value::from((finished.wall_seconds() - state.created_time).max(0.0))
                } else {
                    Value::Nil
                },
            ),
        ])
    }
}
