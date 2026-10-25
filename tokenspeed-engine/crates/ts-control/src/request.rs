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

//! A request's generation state.

use std::collections::BTreeSet;

use ts_wire::{
    codec::Dict,
    output::{FinishDetails, PromptTopLogprobs, TopLogprobRows, TopLogprobs},
    value::Value,
};

/// Why an abort happened, as the frontend sees it (`ABORT_CODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortCode {
    /// A PD or EPD transfer failed.
    TransferFailed = 521,
    /// Any other cause.
    UnknownError = 522,
    /// The NaN guard caught corrupted numbers.
    NumericalError = 523,
    /// A capacity retraction could not image its victim; resubmitting
    /// succeeds.
    CapacityAbort = 524,
}

/// Why a request ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// It generated its token budget.
    Length(i64),
    /// It sampled one of its stop tokens.
    MatchedToken(i32),
    /// It was aborted, by the client or the engine.
    Abort {
        /// What the frontend reports.
        message: String,
        /// The kind of abort.
        err_type: AbortCode,
        /// The HTTP status the frontend answers with (400 for a client
        /// error), or `None` for an internal error.
        status_code: Option<u16>,
    },
}

impl FinishReason {
    /// An abort of no particular kind.
    pub fn abort(message: impl Into<String>) -> Self {
        FinishReason::Abort {
            message: message.into(),
            err_type: AbortCode::UnknownError,
            status_code: None,
        }
    }

    /// The finish type the output wire carries.
    pub fn wire_type(&self) -> &'static str {
        match self {
            FinishReason::Length(_) => "length",
            FinishReason::MatchedToken(_) => "stop",
            FinishReason::Abort { .. } => "abort",
        }
    }

    /// The abort's message, if this is one.
    pub fn abort_message(&self) -> Option<&str> {
        match self {
            FinishReason::Abort { message, .. } => Some(message),
            _ => None,
        }
    }

    /// The reason in full (`BaseFinishReason.to_json()`), keys in its order.
    pub fn details(&self) -> FinishDetails {
        let entry = |key: &str, value: Value| (key.to_owned(), value);
        let text = |value: &str| Value::Str(value.to_owned());
        Dict(match self {
            FinishReason::Length(length) => vec![
                entry("type", text("length")),
                entry("length", Value::from(*length)),
            ],
            FinishReason::MatchedToken(token) => vec![
                entry("type", text("stop")),
                entry("matched", Value::from(i64::from(*token))),
            ],
            FinishReason::Abort {
                message,
                err_type,
                status_code,
            } => {
                let mut entries = vec![
                    entry("type", text("abort")),
                    entry("message", text(message)),
                    entry("err_type", Value::from(*err_type as i64)),
                ];
                if let Some(code) = status_code {
                    entries.push(entry("status_code", Value::from(i64::from(*code))));
                }
                entries
            }
        })
    }
}

/// One position's top-k alternatives: their logprobs and token ids.
type Alternatives = (Vec<f64>, Vec<i32>);

/// A request's prompt logprobs once its prompt completed, as the frontend
/// receives them.
#[derive(Debug, Clone, PartialEq)]
pub struct InputLogprobs {
    /// `None` for the first position, then each position's logprob given
    /// its prefix.
    pub values: Vec<Option<f64>>,
    /// The prompt tokens from the start position on.
    pub ids: Vec<i32>,
    /// The top-k alternatives at each position (`None` for the first), when
    /// the request asked for them.
    pub top: Option<PromptTopLogprobs>,
}

/// What a request returns beyond each sampled token's logprob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogprobRequest {
    /// The first prompt position whose logprob it returns; -1 for the last
    /// one, which needs no logits.
    pub start: i64,
    /// Top-k alternatives per position, 0 for none.
    pub top_logprobs_num: usize,
    /// This engine runs the prompt rows (not the PD decode role, whose
    /// prefill node returns the prompt logprobs).
    pub computes_prompt_logprobs: bool,
}

/// What the device needs to cut a request's verify rounds (`StopCriteria`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopCriteria {
    /// Tokens that end the request, sorted; empty under `ignore_eos`.
    pub stop_token_ids: Vec<i32>,
    /// Prompt length plus the token budget.
    pub token_limit: i64,
}

/// The device's per-request stop table width (`MAX_STOP_TOKEN_IDS`).
pub const MAX_STOP_TOKEN_IDS: usize = 16;

/// One admitted request.
#[derive(Debug, Clone)]
pub struct RequestState {
    /// Prompt token ids.
    pub prompt_ids: Vec<i32>,
    /// Prompt length.
    pub prompt_len: i64,
    /// The token budget, already clamped to the context.
    pub max_new_tokens: i64,
    /// Tokens that end the request: its own stop ids and the model's EOS
    /// ids; none under `ignore_eos`.
    pub stop_token_ids: BTreeSet<i32>,
    /// Stream every step rather than every [`FORCED_STREAM_INTERVAL`].
    pub stream: bool,
    /// The request's own streaming interval, if it set one.
    pub stream_interval: Option<i64>,
    /// Return each sampled token's logprob.
    pub return_logprob: bool,
    /// The first prompt position whose logprob it returns, resolved.
    pub logprob_start_len: i64,
    /// Top-k alternatives it returns per position, 0 for none.
    pub top_logprobs_num: usize,
    /// It returns prompt logprobs (logprobs on, on a role that computes the
    /// prompt).
    pub returns_input_logprobs: bool,

    /// Prompt tokens computed so far.
    pub computed_length: i64,
    /// Generated tokens.
    pub output_ids: Vec<i32>,
    /// Their logprobs, when requested.
    pub output_logprobs: Vec<f64>,
    /// Their top-k alternatives' logprobs and ids, when requested.
    pub output_top_logprobs: Option<TopLogprobs>,
    /// The prompt positions' logprobs as the chunks deliver them, over
    /// `[logprob_start_len, prompt_len - 1)`; `None` when none is wanted.
    pub input_logprob_rows: Option<Vec<Option<f64>>>,
    /// Their top-k alternatives, when requested.
    pub input_top_logprob_rows: Option<Vec<Option<Alternatives>>>,
    /// The prompt logprobs once the prompt completed.
    pub input_logprobs: Option<InputLogprobs>,
    /// They were sent.
    pub input_logprobs_sent: bool,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: i64,
    /// Speculative verify rounds this request took part in.
    pub spec_verify_ct: i64,
    /// Once it finished with speculative decoding, its accepted tokens per
    /// verify round.
    pub accept_draft_tokens: Option<f64>,
    /// A NaN flag on a prefill chunk that owed no token: the request aborts
    /// once its prompt completes.
    pub numerical_error_detected: bool,
    /// Set once the request ended.
    pub finished_reason: Option<FinishReason>,
    /// The client aborted it (or a pause did).
    pub to_abort: bool,
    /// An aborted request whose client still waits for a final output
    /// (pause-initiated aborts).
    pub abort_notify_client: bool,

    /// When it reached the scheduler's waiting queue, until its first
    /// forward observes the wait.
    pub submitted_at: Option<std::time::Instant>,
    /// When the frontend received it (wall-clock seconds), 0 if it did not
    /// say.
    pub created_time: f64,
    /// Its lifecycle timestamps, with `--enable-log-request-stats`.
    pub timings: Option<crate::request_stats::RequestTimings>,
    /// The matcher of its grammar, once compiled.
    pub grammar: Option<RequestGrammar>,

    /// Which admission this is: unique per processor, so a client reusing a
    /// finished request's id is a different request.
    pub admission: u64,
    /// Generated tokens already sent.
    pub send_token_offset: usize,
    /// The final output was sent.
    pub finished_output: bool,
}

/// A request's grammar matcher, shared with the data plane by `handle`.
#[derive(Clone)]
pub struct RequestGrammar {
    /// The data plane's handle on it.
    pub handle: u64,
    /// The matcher.
    pub matcher: ts_grammar::table::SharedMatcher,
}

impl std::fmt::Debug for RequestGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestGrammar")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

/// Non-streaming requests are still flushed every this many tokens.
pub const FORCED_STREAM_INTERVAL: i64 = 50;

impl RequestState {
    /// A request with nothing generated yet.
    pub fn new(
        prompt_ids: Vec<i32>,
        max_new_tokens: i64,
        stop_token_ids: BTreeSet<i32>,
        stream: bool,
        stream_interval: Option<i64>,
        return_logprob: bool,
    ) -> Self {
        let prompt_len = prompt_ids.len() as i64;
        Self {
            prompt_len,
            prompt_ids,
            max_new_tokens,
            stop_token_ids,
            stream,
            stream_interval,
            return_logprob,
            logprob_start_len: (prompt_len - 1).max(0),
            top_logprobs_num: 0,
            returns_input_logprobs: false,
            computed_length: 0,
            output_ids: Vec::new(),
            output_logprobs: Vec::new(),
            output_top_logprobs: None,
            input_logprob_rows: None,
            input_top_logprob_rows: None,
            input_logprobs: None,
            input_logprobs_sent: false,
            cached_tokens: 0,
            spec_verify_ct: 0,
            accept_draft_tokens: None,
            numerical_error_detected: false,
            finished_reason: None,
            to_abort: false,
            abort_notify_client: false,
            submitted_at: None,
            created_time: 0.0,
            timings: None,
            grammar: None,
            admission: 0,
            send_token_offset: 0,
            finished_output: false,
        }
    }

    /// Set what the request returns beyond each sampled token's logprob
    /// (`return_logprob` already on).
    pub fn set_logprobs(&mut self, request: LogprobRequest) {
        let last_position = (self.prompt_len - 1).max(0);
        // -1, the default, is the last position.
        self.logprob_start_len = if request.start < 0 {
            last_position
        } else {
            request.start.min(last_position)
        };
        self.top_logprobs_num = request.top_logprobs_num;
        self.returns_input_logprobs = self.return_logprob && request.computes_prompt_logprobs;
        let scored = (last_position - self.logprob_start_len) as usize;
        if self.wants_input_logprobs() {
            self.input_logprob_rows = Some(vec![None; scored]);
            if self.top_logprobs_num > 0 {
                self.input_top_logprob_rows = Some(vec![None; scored]);
            }
        }
        if self.return_logprob && self.top_logprobs_num > 0 {
            self.output_top_logprobs = Some((Vec::new(), Vec::new()));
        }
    }

    /// Whether some prompt position needs logits for its logprob: the
    /// default start, the last prompt position, needs none.
    pub fn wants_input_logprobs(&self) -> bool {
        self.returns_input_logprobs && self.logprob_start_len < self.prompt_len - 1
    }

    /// Store one chunk's prompt logprobs for positions `[start, start + n)`
    /// and their top-k alternatives (most likely first), cut to the
    /// request's own count.
    pub fn record_input_logprobs(
        &mut self,
        start: i64,
        logprobs: &[f64],
        top: Option<TopLogprobRows<'_>>,
    ) -> Result<(), String> {
        let rows = self
            .input_logprob_rows
            .as_mut()
            .ok_or("prompt logprobs recorded for a request that wants none")?;
        let offset = start - self.logprob_start_len;
        if offset < 0 || offset as usize + logprobs.len() > rows.len() {
            return Err(format!(
                "prompt logprob rows [{start}, {}) fall outside [{}, {})",
                start + logprobs.len() as i64,
                self.logprob_start_len,
                self.prompt_len - 1
            ));
        }
        let offset = offset as usize;
        for (row, logprob) in rows[offset..].iter_mut().zip(logprobs) {
            *row = Some(*logprob);
        }
        if let Some(top_rows) = self.input_top_logprob_rows.as_mut() {
            let (values, ids) = top.ok_or(
                "prompt logprobs recorded without the top-k alternatives the request asked for",
            )?;
            let k = self.top_logprobs_num;
            for (row, (row_values, row_ids)) in
                top_rows[offset..].iter_mut().zip(values.iter().zip(ids))
            {
                *row = Some((
                    row_values[..k.min(row_values.len())].to_vec(),
                    row_ids[..k.min(row_ids.len())].to_vec(),
                ));
            }
        }
        Ok(())
    }

    /// Assemble the prompt logprobs once the prompt's final chunk
    /// committed; a second call keeps them.
    pub fn finalize_input_logprobs(&mut self) -> Result<(), String> {
        if !self.returns_input_logprobs || self.input_logprobs.is_some() {
            return Ok(());
        }
        let rows = self.input_logprob_rows.take().unwrap_or_default();
        let missing = rows.iter().filter(|row| row.is_none()).count();
        if missing > 0 {
            // Every scored position is recomputed by a chunk that commits
            // before the prompt completes, so a gap is a planning bug.
            return Err(format!(
                "prompt logprobs incomplete: {missing} of {} positions have no value",
                rows.len()
            ));
        }
        let start = self.logprob_start_len as usize;
        let mut values = Vec::with_capacity(rows.len() + 1);
        values.push(None);
        values.extend(rows);
        let ids = self.prompt_ids[start..].to_vec();
        if values.len() != ids.len() {
            return Err(format!(
                "prompt logprobs have {} values for {} tokens",
                values.len(),
                ids.len()
            ));
        }
        let top = if self.top_logprobs_num > 0 {
            let tops = self.input_top_logprob_rows.take().unwrap_or_default();
            if tops.len() != values.len() - 1 || tops.iter().any(Option::is_none) {
                return Err(format!(
                    "prompt top-k logprobs incomplete: {} of {} positions have alternatives",
                    tops.iter().filter(|top| top.is_some()).count(),
                    values.len() - 1
                ));
            }
            // The first position has no alternatives, like its value.
            let (top_values, top_ids): (Vec<_>, Vec<_>) = tops.into_iter().flatten().unzip();
            Some((
                std::iter::once(None)
                    .chain(top_values.into_iter().map(Some))
                    .collect(),
                std::iter::once(None)
                    .chain(top_ids.into_iter().map(Some))
                    .collect(),
            ))
        } else {
            None
        };
        self.input_logprobs = Some(InputLogprobs { values, ids, top });
        Ok(())
    }

    /// Whether the request ended.
    pub fn finished(&self) -> bool {
        self.finished_reason.is_some()
    }

    /// Whether the whole prompt is computed.
    pub fn prefill_finished(&self) -> bool {
        self.computed_length >= self.prompt_len
    }

    /// Where the device cuts this request's verify rounds.
    pub fn stop_criteria(&self) -> StopCriteria {
        StopCriteria {
            stop_token_ids: self.stop_token_ids.iter().copied().collect(),
            token_limit: self.prompt_len + self.max_new_tokens,
        }
    }

    /// Abort the request; `notify_client` keeps streaming its final output.
    /// End the request with an abort; `status_code` is the HTTP status the
    /// frontend answers with (400 for the request's own error), or `None`.
    pub fn abort(&mut self, message: &str, notify_client: bool, status_code: Option<u16>) {
        self.to_abort = true;
        self.abort_notify_client = notify_client;
        self.finished_reason = Some(FinishReason::Abort {
            message: message.to_owned(),
            err_type: AbortCode::UnknownError,
            status_code,
        });
    }

    /// End the request if its last token ended it.
    pub fn check_finished(&mut self, skip_grammar_termination: bool) {
        if self.finished() {
            return;
        }
        // A grammar's end comes first. Where the data plane advances the
        // matcher, the commit names the terminating token itself, so the
        // check would fire again on every later token.
        if !skip_grammar_termination {
            if let (Some(grammar), Some(&last)) = (&self.grammar, self.output_ids.last()) {
                if grammar
                    .matcher
                    .lock()
                    .expect("matcher lock")
                    .is_terminated()
                {
                    self.finished_reason = Some(FinishReason::MatchedToken(last));
                    return;
                }
            }
        }
        if self.output_ids.len() as i64 >= self.max_new_tokens {
            self.finished_reason = Some(FinishReason::Length(self.max_new_tokens));
            return;
        }
        if let Some(&last) = self.output_ids.last() {
            if self.stop_token_ids.contains(&last) {
                self.finished_reason = Some(FinishReason::MatchedToken(last));
            }
        }
    }

    /// Whether this step's output is due to the frontend.
    pub fn output_due(&self, default_stream_interval: i64) -> bool {
        if self.finished() {
            return !self.finished_output;
        }
        let generated = self.output_ids.len() as i64;
        if self.stream {
            // A zero interval would divide by zero; it means every token.
            let interval = self
                .stream_interval
                .unwrap_or(default_stream_interval)
                .max(1);
            if interval > 1 {
                generated % interval == 1
            } else {
                generated % interval == 0
            }
        } else {
            generated == 1 || generated % FORCED_STREAM_INTERVAL == 0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(max_new_tokens: i64, stop: &[i32]) -> RequestState {
        RequestState::new(
            vec![1, 2, 3],
            max_new_tokens,
            stop.iter().copied().collect(),
            true,
            None,
            false,
        )
    }

    #[test]
    fn the_budget_ends_a_request_before_a_stop_token() {
        let mut request = state(2, &[7]);
        request.output_ids = vec![1, 7];
        request.check_finished(false);
        assert_eq!(request.finished_reason, Some(FinishReason::Length(2)));
    }

    #[test]
    fn a_stop_token_ends_a_request() {
        let mut request = state(10, &[7]);
        request.output_ids = vec![1];
        request.check_finished(false);
        assert!(!request.finished());
        request.output_ids.push(7);
        request.check_finished(false);
        assert_eq!(request.finished_reason, Some(FinishReason::MatchedToken(7)));
    }

    #[test]
    fn an_abort_is_final() {
        let mut request = state(10, &[]);
        request.abort("AbortReq from client", false, None);
        request.output_ids = vec![1; 20];
        request.check_finished(false);
        assert_eq!(
            request.finished_reason,
            Some(FinishReason::abort("AbortReq from client"))
        );
        assert_eq!(request.finished_reason.unwrap().wire_type(), "abort");
    }

    #[test]
    fn stream_intervals_follow_the_python_schedule() {
        let due = |stream: bool, interval: Option<i64>, generated: usize| {
            let mut request = RequestState::new(
                vec![1, 2, 3],
                1000,
                BTreeSet::new(),
                stream,
                interval,
                false,
            );
            request.output_ids = vec![0; generated];
            request.output_due(1)
        };
        assert!((1..5).all(|n| due(true, None, n)));
        // An interval above one fires on its first token and every interval after.
        assert_eq!(
            (1..8).map(|n| due(true, Some(3), n)).collect::<Vec<_>>(),
            [true, false, false, true, false, false, true]
        );
        // Non-streaming requests: the first token, then every 50.
        assert!(due(false, None, 1) && !due(false, None, 2) && due(false, None, 50));
    }
}
