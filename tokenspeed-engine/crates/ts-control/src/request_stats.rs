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

//! The per-request finish line: with
//! `--enable-log-request-stats` a timing summary, otherwise the average
//! accepted tokens per verify.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::request::{FinishReason, RequestState};

/// Wall-clock seconds, the clock the request's `created_time` uses.
pub fn wall_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// A request's lifecycle timestamps, wall-clock seconds, 0 until reached.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestTimings {
    scheduled: f64,
    prefill_done: f64,
    first_token: f64,
    finish: f64,
    /// Rounds in which other requests' prefill interrupted this decode, and
    /// their seconds.
    preempt_count: i64,
    preempt_time: f64,
    preempted_last_step: bool,
}

impl RequestTimings {
    /// Its first forward dispatched.
    pub fn mark_scheduled(&mut self, now: f64) {
        if self.scheduled == 0.0 {
            self.scheduled = now;
        }
    }

    /// Its prompt completed.
    pub fn mark_prefill_done(&mut self, now: f64) {
        if self.prefill_done == 0.0 {
            self.prefill_done = now;
        }
    }

    /// Its first token committed.
    pub fn mark_first_token(&mut self, now: f64) {
        if self.first_token == 0.0 {
            self.first_token = now;
        }
    }

    /// A decode step took `step` seconds; `prefilling_others` when other
    /// requests prefilled in it. Each run of such steps counts once.
    pub fn record_decode_step(&mut self, step: f64, prefilling_others: bool) {
        if prefilling_others {
            if !self.preempted_last_step {
                self.preempt_count += 1;
            }
            self.preempt_time += step;
        }
        self.preempted_last_step = prefilling_others;
    }
}

/// Python's `repr` of a float in the ranges these lines print.
fn py_float(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e16 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

fn round(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}

fn ms(end: f64, start: f64) -> f64 {
    if end > 0.0 && start > 0.0 {
        round((end - start) * 1e3, 2)
    } else {
        0.0
    }
}

/// The accepted tokens per verify a finished request averaged
/// (`accept_draft_tokens`): its tokens but the first over its verify rounds.
pub fn accept_draft_tokens(state: &RequestState) -> Option<f64> {
    (state.spec_verify_ct > 0)
        .then(|| (state.output_ids.len() as f64 - 1.0) / state.spec_verify_ct as f64)
}

/// `Req: {rid} Finish! Accept_num_tokens_avg: {avg}`.
pub fn accept_length_line(rid: &str, state: &RequestState) -> String {
    let average = accept_draft_tokens(state).map_or_else(|| "0".to_owned(), py_float);
    format!("Req: {rid} Finish! Accept_num_tokens_avg: {average}")
}

/// `Req: {rid} Finish! RequestStats(...)`, for a
/// request that ended at `finish` with `spec_num_tokens`-wide verify.
pub fn stats_line(
    rid: &str,
    state: &RequestState,
    timings: &RequestTimings,
    finish: f64,
    spec_num_tokens: Option<usize>,
) -> String {
    let prompt = state.prompt_len;
    let output = state.output_ids.len() as i64;
    let decode_window = if timings.first_token > 0.0 {
        finish - timings.first_token
    } else {
        0.0
    };
    let decode_tps = if decode_window > 0.0 && output > 1 {
        round((output - 1) as f64 / decode_window, 1)
    } else {
        0.0
    };
    let (acc_len, acc_rate) = match (spec_num_tokens, accept_draft_tokens(state)) {
        (Some(width), Some(accepted)) => {
            let rate = if width > 1 {
                round((accepted - 1.0).max(0.0) / (width - 1) as f64, 4)
            } else {
                0.0
            };
            (py_float(round(accepted, 2)), py_float(rate))
        }
        _ => ("None".to_owned(), "None".to_owned()),
    };
    let status = if matches!(state.finished_reason, Some(FinishReason::Abort { .. })) {
        "aborted"
    } else {
        "finished"
    };
    let reason = state
        .finished_reason
        .as_ref()
        .map_or("unknown", FinishReason::wire_type);
    let hit_rate = if prompt > 0 {
        round(state.cached_tokens as f64 / prompt as f64, 4)
    } else {
        0.0
    };
    let created = state.created_time;
    format!(
        "Req: {rid} Finish! RequestStats(status='{status}', reason='{reason}', \
         prompt_tokens={prompt}, cache_tokens={}, output_tokens={output}, cache_hit_rate={}, \
         queue_ms={}, prefill_ms={}, ttft_ms={}, total_ms={}, preempt_ms={}, preempt_count={}, \
         decode_tps={}, acc_len={acc_len}, acc_rate={acc_rate}, recv_ts={}, commit_ts={}, \
         finish_ts={})",
        state.cached_tokens,
        py_float(hit_rate),
        py_float(ms(timings.scheduled, created)),
        py_float(ms(timings.prefill_done, timings.scheduled)),
        py_float(ms(timings.first_token, created)),
        py_float(ms(finish, created)),
        py_float(round(timings.preempt_time * 1e3, 2)),
        timings.preempt_count,
        py_float(decode_tps),
        py_float(round(created, 3)),
        py_float(round(timings.scheduled, 3)),
        py_float(round(finish, 3)),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn the_stats_line_is_the_python_repr() {
        let mut state = RequestState::new(vec![1, 2, 3, 4], 8, BTreeSet::new(), true, None, false);
        state.created_time = 100.0;
        state.cached_tokens = 2;
        state.output_ids = vec![5, 6, 7];
        state.spec_verify_ct = 2;
        state.finished_reason = Some(FinishReason::Length(3));
        let mut timings = RequestTimings::default();
        timings.mark_scheduled(100.5);
        timings.mark_prefill_done(100.75);
        timings.mark_first_token(100.75);
        timings.record_decode_step(0.25, true);
        timings.record_decode_step(0.25, true);
        timings.record_decode_step(0.25, false);
        assert_eq!(
            stats_line("r", &state, &timings, 101.0, Some(4)),
            "Req: r Finish! RequestStats(status='finished', reason='length', prompt_tokens=4, \
             cache_tokens=2, output_tokens=3, cache_hit_rate=0.5, queue_ms=500.0, \
             prefill_ms=250.0, ttft_ms=750.0, total_ms=1000.0, preempt_ms=500.0, \
             preempt_count=1, decode_tps=8.0, acc_len=1.0, acc_rate=0.0, recv_ts=100.0, \
             commit_ts=100.5, finish_ts=101.0)"
        );
        assert_eq!(
            accept_length_line("r", &state),
            "Req: r Finish! Accept_num_tokens_avg: 1.0"
        );
    }
}
