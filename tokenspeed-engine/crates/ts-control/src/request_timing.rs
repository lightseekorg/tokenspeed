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
    sync::OnceLock,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use ts_wire::{RequestClock, RequestTimestamp, RequestTiming, RequestTimingEvents};

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
}

impl EventStamp {
    pub(crate) fn now() -> Self {
        // Establish the process origin before sampling any local boundary.
        clock();
        Self::at(SystemTime::now(), Instant::now())
    }

    pub(crate) fn at(wall: SystemTime, mono: Instant) -> Self {
        Self { wall, mono }
    }

    pub(crate) fn wall_seconds(&self) -> f64 {
        self.wall
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    fn timestamp(&self) -> RequestTimestamp {
        RequestTimestamp {
            ts_unix_ns: self
                .wall
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .to_string(),
            ts_mono_ns: self
                .mono
                .checked_duration_since(clock().origin)
                .map(|time| time.as_nanos().to_string()),
        }
    }
}

/// The request owns its observed boundaries until its terminal output is assembled.
#[derive(Debug, Clone, Default)]
pub(crate) struct RequestTimingState {
    pub(crate) accepted: Option<EventStamp>,
    kv_ready: Option<EventStamp>,
    kv_output_tokens: Option<usize>,
}

impl RequestTimingState {
    /// Record readiness once; retraction or later chunks never move the first boundary.
    pub(crate) fn kv_ready(&mut self, at: EventStamp, output_tokens: usize) {
        if self.kv_ready.is_none() {
            self.kv_ready = Some(at);
            self.kv_output_tokens = Some(output_tokens);
        }
    }

    /// Return the terminal timestamps in one clock context, retaining missing boundaries.
    pub(crate) fn finish(
        &self,
        role: &str,
        ranks: Option<(i64, i64)>,
        created_time: f64,
        finished: &EventStamp,
    ) -> RequestTiming {
        let clock = clock();
        RequestTiming {
            role: role.to_owned(),
            dp_rank: ranks.map(|r| r.0),
            tp_rank: ranks.map(|r| r.1),
            clock: RequestClock {
                host_id: clock.host_id.clone(),
                process_instance_id: clock.process_instance_id.clone(),
                clock_sync: "unknown".to_owned(),
                clock_uncertainty_ns: None,
            },
            events: RequestTimingEvents {
                accepted: self.accepted.as_ref().map(EventStamp::timestamp),
                kv_ready: self.kv_ready.as_ref().map(EventStamp::timestamp),
                finished: finished.timestamp(),
            },
            kv_output_tokens: self.kv_output_tokens.map(|n| n as i64),
            e2e_latency_s: (created_time > 0.0)
                .then(|| (finished.wall_seconds() - created_time).max(0.0)),
        }
    }
}
