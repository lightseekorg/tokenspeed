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

//! Committing a forward's results: request state, the scheduler's feedback
//! events and the frontend's outputs.

use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use ts_scheduler::{ForwardBatchView, ForwardEvent, ForwardEventKind, SchedulerAbortView};

use ts_wire::output::FinishDetails;

use crate::{
    codec::{Reader, Writer},
    request::{AbortCode, FinishReason, InputLogprobs, RequestState},
    request_stats::{self, wall_seconds, RequestTimings},
};

const NUMERICAL_ERROR: &str = "Request terminated: numerical corruption (NaN logits or \
     out-of-vocab sample) detected during generation.";

/// How long an abort for a request not registered yet is kept.
const PENDING_ABORT_TTL: Duration = Duration::from_secs(300);

/// One forward's results, read once its device-to-host copy landed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ForwardResult {
    /// Sampled tokens: each prefill row's kept tokens, then each decode row's
    /// whole verify window (one token without speculation).
    pub output_tokens: Vec<i32>,
    /// Tokens each row keeps.
    pub output_lengths: Vec<i32>,
    /// The tokens' logprobs in the same layout; `None` when the forward
    /// computed none.
    pub output_logprobs: Option<Vec<f64>>,
    /// Per-row NaN-guard flags; `None` with the guard off.
    pub nan_flags: Option<Vec<i32>>,
    /// P role: each row's sampled token and the drafter's candidates, the
    /// bootstrap payload a completed prompt's remote decode carries; `None`
    /// on the other roles.
    pub next_input_ids: Option<Vec<Vec<i32>>>,
    /// The data plane's serving-time kernel compilations and their seconds
    /// so far, when the engine exports metrics; this rank's own, so not
    /// shared across a pipeline.
    pub serving_compiles: Option<(u64, f64)>,
    /// Where the data plane advanced the grammar matchers (the capturable
    /// path): per row, the index of the kept token that ended its grammar,
    /// or -1. `None` when the commit advances them.
    pub grammar_terminated_at: Option<Vec<i32>>,
    /// The prompt logprobs of the rows the forward's [`InputLogprobPlan`]
    /// named, flat in plan order; `None` when it gathered none.
    pub input_token_logprobs: Option<Vec<f64>>,
    /// Those rows' top-k alternatives (values and ids per row, most likely
    /// first), when the plan asked for them.
    pub input_top_logprobs: Option<TopLogprobs>,
    /// The top-k alternatives of each prefill row's sampled token, in
    /// prefill output order, when the forward asked for them.
    pub output_top_logprobs: Option<TopLogprobs>,
}

pub use ts_wire::output::TopLogprobs;

/// Which prompt rows of one forward need their next-token logprob
/// (`InputLogprobPlan`): one triple per extend row, so the plan stays
/// O(rows) whatever the prompt length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputLogprobPlan {
    /// Per extend row, the flat input-row index of its first gathered row.
    pub row_starts: Vec<i64>,
    /// Per extend row, how many consecutive rows are gathered; 0 for none.
    pub counts: Vec<i64>,
    /// Per extend row, the prompt position of its first gathered row.
    pub position_starts: Vec<i64>,
    /// Alternatives every gathered row also returns: the most any planned
    /// request asks for.
    pub top_k: usize,
}

/// The logprobs a forward computes beyond its sampled tokens'.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogprobPlan {
    /// The prompt rows it gathers, `None` when no row needs one.
    pub input: Option<InputLogprobPlan>,
    /// Alternatives each prefill row's sampled token is reported with: the
    /// most any extend row's request asks for, 0 for none.
    pub sampled_top_k: usize,
}

impl ForwardResult {
    /// The result as a pipeline's last stage shares it. The top-k
    /// alternatives are not shared: the ingress refuses them under pipeline
    /// parallelism.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Writer::new();
        out.i32s(&self.output_tokens);
        out.i32s(&self.output_lengths);
        out.option(self.output_logprobs.as_deref(), Writer::f64s);
        out.option(self.nan_flags.as_deref(), Writer::i32s);
        out.option(self.next_input_ids.as_deref(), |out, rows| {
            out.count(rows.len());
            for row in rows {
                out.i32s(row);
            }
        });
        out.option(self.grammar_terminated_at.as_deref(), Writer::i32s);
        out.option(self.input_token_logprobs.as_deref(), Writer::f64s);
        out.finish()
    }

    /// A result [`ForwardResult::to_bytes`] wrote, or `None` if `bytes` is
    /// not one.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(bytes);
        let output_tokens = reader.i32s()?;
        let output_lengths = reader.i32s()?;
        let output_logprobs = reader.option(Reader::f64s)?;
        let nan_flags = reader.option(Reader::i32s)?;
        let next_input_ids = reader.option(|reader| {
            let rows = reader.count()?;
            (0..rows).map(|_| reader.i32s()).collect()
        })?;
        let grammar_terminated_at = reader.option(Reader::i32s)?;
        let input_token_logprobs = reader.option(Reader::f64s)?;
        reader.finish(Self {
            output_tokens,
            output_lengths,
            output_logprobs,
            nan_flags,
            next_input_ids,
            serving_compiles: None,
            grammar_terminated_at,
            input_token_logprobs,
            // A pipeline computes no alternatives: the ingress refuses them.
            input_top_logprobs: None,
            output_top_logprobs: None,
        })
    }
}

/// Engram's input windows for one forward (`NGramInputs`): per row, the token
/// at the row's position followed by its `context_len` predecessors, newest
/// first, -1 where the request has no token.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NgramWindows {
    /// One window per row.
    pub tokens: Vec<Vec<i32>>,
    /// Each row's position: an extend row's prefix length, a decode row's
    /// newest committed token.
    pub positions: Vec<i64>,
}

/// The committed prefixes a forward seeds into its slots' token-history
/// rows (`RequestHistorySeeds`), for a model that reads each request's
/// token history.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HistorySeeds {
    /// The slots to seed.
    pub slots: Vec<i32>,
    /// Each slot's committed prefix length.
    pub prefix_lengths: Vec<i64>,
    /// Each slot's committed prefix.
    pub tokens: Vec<Vec<i32>>,
}

/// Which request's committed tokens each history row holds.
///
/// Every forward appends its inputs to its slot's row at the committed
/// frontier, so a row holds a request's tokens from position 0 once it
/// started there at position 0 or was seeded there, until another request
/// runs in the slot. A forward seeds its committed prefix otherwise: after
/// a prefix-cache hit, a slot handoff, a retracted request's restore into
/// another slot, or a PD decode landing. Ownership is per admission, not
/// per request id, which clients may reuse.
#[derive(Debug, Default)]
pub struct HistoryRows {
    owners: HashMap<i32, u64>,
}

impl HistoryRows {
    /// Record `batch`'s slot ownership and the prefixes it must seed, or
    /// `None` when no row needs one. Called once per dispatched forward, in
    /// dispatch order.
    pub fn seeds(
        &mut self,
        batch: &ForwardBatchView,
        outputs: &OutputProcessor,
    ) -> Result<Option<HistorySeeds>, CommitError> {
        let mut seeds = HistorySeeds::default();
        for (i, rid) in batch.request_ids.iter().enumerate() {
            let slot = batch.request_pool_indices[i];
            let state = outputs
                .state(rid)
                .ok_or_else(|| CommitError::History(rid.clone()))?;
            let held = self.owners.insert(slot, state.admission) == Some(state.admission);
            let total = state.prompt_ids.len() + state.output_ids.len();
            let boundary = if i < batch.num_extends {
                batch.extend_prefix_lens[i].max(0) as usize
            } else {
                total.saturating_sub(1)
            };
            if boundary == 0 || held {
                continue;
            }
            if boundary > total {
                return Err(CommitError::History(rid.clone()));
            }
            let prompt = &state.prompt_ids;
            let prefix: Vec<i32> = if boundary <= prompt.len() {
                prompt[..boundary].to_vec()
            } else {
                prompt
                    .iter()
                    .chain(&state.output_ids[..boundary - prompt.len()])
                    .copied()
                    .collect()
            };
            seeds.slots.push(slot);
            seeds.prefix_lengths.push(boundary as i64);
            seeds.tokens.push(prefix);
        }
        Ok((!seeds.slots.is_empty()).then_some(seeds))
    }
}

/// One request's output to the frontend this step.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamedOutput {
    /// Request id.
    pub rid: String,
    /// Tokens generated since the last output.
    pub output_ids: Vec<i32>,
    /// The finish type once the request ended.
    pub finish_reason: Option<&'static str>,
    /// Prompt length.
    pub prompt_tokens: i64,
    /// Tokens generated so far.
    pub completion_tokens: i64,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: i64,
    /// Logprobs of `output_ids`; empty when not requested.
    pub output_logprobs: Vec<f64>,
    /// The finish reason in full once the request ended.
    pub finish_details: Option<FinishDetails>,
    /// With speculative decoding: verify rounds so far, and once finished
    /// the accepted tokens per round.
    pub spec: Option<(i64, Option<f64>)>,
    /// The prompt logprobs, in the first output after the prompt completed.
    pub input_logprobs: Option<InputLogprobs>,
    /// The top-k alternatives of `output_ids`, when the request asked for
    /// them.
    pub output_top_logprobs: Option<TopLogprobs>,
}

/// What committing a forward produced.
#[derive(Debug, Default)]
pub struct Committed {
    /// Feedback for the scheduler, in order.
    pub events: Vec<ForwardEvent>,
    /// Outputs for the frontend.
    pub outputs: Vec<StreamedOutput>,
    /// Requests that ended and left the processor.
    pub released: Vec<String>,
    /// What it counted, for the metrics.
    pub stats: CommitStats,
}

/// What a commit counted, for the metrics (`EngineMetrics`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitStats {
    /// Requests the NaN guard terminated.
    pub nan_aborts: i64,
    /// Requests the scheduler aborted for capacity.
    pub capacity_aborts: i64,
    /// A verify step's decode rows whose prompt had completed.
    pub spec_decode_slots: i64,
    /// Their accepted draft tokens: the kept tokens but each row's last.
    pub spec_accepted_tokens: i64,
}

impl CommitStats {
    /// Add `other`'s counts.
    pub fn add(&mut self, other: CommitStats) {
        self.nan_aborts += other.nan_aborts;
        self.capacity_aborts += other.capacity_aborts;
        self.spec_decode_slots += other.spec_decode_slots;
        self.spec_accepted_tokens += other.spec_accepted_tokens;
    }
}

/// A result that breaks an invariant the engine relies on; serving cannot
/// safely continue.
#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    /// A sampled token its grammar's bitmask should have masked out.
    #[error("request {rid}: the grammar rejects sampled token {token}")]
    GrammarRejected {
        /// The request.
        rid: String,
        /// The token.
        token: i32,
    },
    /// The result's columns do not match the batch.
    #[error("forward result for {rows} rows has {found} {column}")]
    Shape {
        /// Rows in the batch.
        rows: usize,
        /// The mismatched column.
        column: &'static str,
        /// Its length.
        found: usize,
    },
    /// A decode row kept no token although its request runs.
    #[error("decode request {0} kept no token")]
    EmptyDecode(String),
    /// The P role's bootstrap token disagrees with the drafter's candidates.
    #[error("request {rid} sampled {sampled}, but its first speculative candidate is {candidate}")]
    BootstrapMismatch {
        /// The request.
        rid: String,
        /// The token sampled.
        sampled: i32,
        /// The first candidate.
        candidate: i32,
    },
    /// An n-gram window reaches past the request's tokens.
    #[error("the n-gram window of {0} reaches past its tokens")]
    NgramWindow(String),
    /// A token-history seed reaches past the request's tokens.
    #[error("the token-history prefix of {0} reaches past its tokens")]
    History(String),
    /// The prompt logprobs do not fit the request's positions.
    #[error("request {rid}: {detail}")]
    InputLogprobs {
        /// The request.
        rid: String,
        /// What is wrong.
        detail: String,
    },
    /// Speculation committed past the per-request KV extent.
    #[error(
        "request {rid} committed {total} tokens, past the physical KV extent of {limit}; \
         the next KV write would run out of bounds"
    )]
    PhysicalExtent {
        /// The request.
        rid: String,
        /// Prompt plus output after this step.
        total: i64,
        /// The extent.
        limit: i64,
    },
}

/// `values[start..start + len]`, cut at the end of `values`.
fn window<T: Clone>(values: &[T], start: usize, len: usize) -> Vec<T> {
    window_ref(values, start, len).to_vec()
}

/// [`window`] without the copy.
fn window_ref<T>(values: &[T], start: usize, len: usize) -> &[T] {
    let end = (start + len).min(values.len());
    &values[start.min(end)..end]
}

fn event(kind: ForwardEventKind, rid: &str, tokens: Vec<i32>, reserve: i32) -> ForwardEvent {
    ForwardEvent {
        kind,
        request_id: rid.to_owned(),
        tokens,
        spec_candidate_ids: Vec::new(),
        reserve_num_tokens: reserve,
        bootstrap_token: -1,
    }
}

fn extend_result(rid: &str, tokens: Vec<i32>) -> ForwardEvent {
    event(ForwardEventKind::ExtendResult, rid, tokens, 0)
}

/// What the engine does in a disaggregated deployment, as its output
/// processing sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// No disaggregation.
    Plain,
    /// Prefills prompts another node decodes: no output streams until the
    /// KV reached the decode node.
    Prefill,
    /// Decodes prompts another node prefilled.
    Decode,
}

fn finish(rid: &str, nan_detected: bool) -> ForwardEvent {
    // Abort rather than Finish keeps numerically corrupted KV out of the
    // prefix caches.
    let kind = if nan_detected {
        ForwardEventKind::Abort
    } else {
        ForwardEventKind::Finish
    };
    event(kind, rid, Vec::new(), 0)
}

/// Which finish lines a processor logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FinishLog {
    /// This rank logs them: the attention-TP group's first rank.
    pub enabled: bool,
    /// `--enable-log-request-stats`: the timing summary replaces the
    /// accept-length line.
    pub request_stats: bool,
}

/// The admitted requests and their outputs.
pub struct OutputProcessor {
    states: HashMap<String, RequestState>,
    /// Aborts for requests not registered yet, oldest first.
    pending_aborts: VecDeque<(String, Instant)>,
    /// Verify width with speculation.
    spec_num_tokens: Option<usize>,
    /// The server's streaming interval.
    stream_interval: i64,
    /// Tokens a request's buffers hold; committing past it is a bug.
    physical_context_len: Option<i64>,
    role: Role,
    finish_log: FinishLog,
    /// When the last commit ran, for the request stats' decode steps.
    last_commit_at: f64,
    /// Requests registered so far, numbering each admission.
    admissions: u64,
}

impl OutputProcessor {
    /// A processor for `spec_num_tokens`-wide verify (or none), the
    /// server's `stream_interval` and the engine's `role`, logging the
    /// finish lines `finish_log` selects.
    pub fn new(
        spec_num_tokens: Option<usize>,
        stream_interval: i64,
        physical_context_len: Option<i64>,
        role: Role,
        finish_log: FinishLog,
    ) -> Self {
        Self {
            states: HashMap::new(),
            pending_aborts: VecDeque::new(),
            spec_num_tokens,
            stream_interval,
            physical_context_len,
            role,
            finish_log,
            last_commit_at: 0.0,
            admissions: 0,
        }
    }

    /// Give `rid` its grammar's matcher.
    pub fn attach_grammar(&mut self, rid: &str, grammar: crate::request::RequestGrammar) {
        if let Some(state) = self.states.get_mut(rid) {
            state.grammar = Some(grammar);
        }
    }

    /// Stamp the first dispatch of `rids`, for the request stats.
    pub fn mark_scheduled(&mut self, rids: &[String]) {
        if !self.finish_log.request_stats {
            return;
        }
        let now = wall_seconds();
        for rid in rids {
            if let Some(timings) = self.states.get_mut(rid).and_then(|s| s.timings.as_mut()) {
                timings.mark_scheduled(now);
            }
        }
    }

    /// Log a request's stats line as it leaves the processor.
    fn log_end(&self, rid: &str, state: &RequestState) {
        if !self.finish_log.enabled {
            return;
        }
        if let Some(timings) = &state.timings {
            tracing::info!(
                "{}",
                request_stats::stats_line(
                    rid,
                    state,
                    timings,
                    wall_seconds(),
                    self.spec_num_tokens
                )
            );
        }
    }

    /// The engine's role.
    pub fn role(&self) -> Role {
        self.role
    }

    /// The verify width with speculation.
    pub fn spec_num_tokens(&self) -> Option<usize> {
        self.spec_num_tokens
    }

    /// When `rid` reached the scheduler's waiting queue, the first time this
    /// is asked after it did: a retraction re-queues
    /// a request without re-stamping it.
    pub fn take_submitted_at(&mut self, rid: &str) -> Option<Instant> {
        self.states.get_mut(rid)?.submitted_at.take()
    }

    /// Stamp `rid`'s arrival in the scheduler's waiting queue.
    pub fn mark_submitted(&mut self, rid: &str, at: Instant) {
        if let Some(state) = self.states.get_mut(rid) {
            state.submitted_at = Some(at);
        }
    }

    /// Requests admitted and not ended.
    pub fn num_running(&self) -> usize {
        self.states.len()
    }

    /// A running request's state.
    pub fn state(&self, rid: &str) -> Option<&RequestState> {
        self.states.get(rid)
    }

    /// Admit a request. An abort that arrived before it applies now.
    pub fn register(&mut self, rid: String, mut state: RequestState) {
        self.admissions += 1;
        state.admission = self.admissions;
        if let Some(index) = self.pending_aborts.iter().position(|(id, _)| *id == rid) {
            self.pending_aborts.remove(index);
            state.abort("AbortReq from client", false, None);
        }
        if self.role == Role::Decode {
            // The prompt was computed on the prefill node.
            state.computed_length = state.prompt_len;
        }
        if self.finish_log.request_stats {
            state.timings = Some(RequestTimings::default());
        }
        self.states.insert(rid, state);
    }

    /// P role: the request's KV reached the decode node. Its output is final:
    /// what the prefill node generated, finished by length unless it already
    /// ended. The scheduler finishes it on the event itself.
    pub fn finish_prefill(&mut self, rid: &str) -> Committed {
        let mut committed = Committed::default();
        let Some(mut state) = self.states.remove(rid) else {
            return committed;
        };
        if !state.finished() {
            state.finished_reason = Some(FinishReason::Length(state.output_ids.len() as i64));
        }
        state.finished_output = false;
        if !state.to_abort || state.abort_notify_client {
            committed.outputs = self.stream(&mut [(rid.to_owned(), &mut state)]);
        }
        self.log_end(rid, &state);
        committed.released.push(rid.to_owned());
        committed
    }

    /// D role: the prefill node's KV landed. Its first token joins the
    /// output, which streams with the next commit.
    pub fn remote_prefill_done(
        &mut self,
        rid: &str,
        bootstrap_token: i32,
        cached_tokens: i64,
        bootstrap_logprob: Option<f64>,
    ) {
        let Some(state) = self.states.get_mut(rid) else {
            return;
        };
        state.cached_tokens = cached_tokens;
        state.output_ids.push(bootstrap_token);
        if state.return_logprob {
            match bootstrap_logprob {
                Some(logprob) => state.output_logprobs.push(logprob),
                None => tracing::warn!(
                    rid,
                    "the request returns logprobs but the prefill node sent none for its first token"
                ),
            }
        }
        // The one token the decode node never samples itself still advances
        // its own matcher, or every decode step would mask from a state one
        // token behind.
        if let Some(grammar) = &state.grammar {
            grammar
                .matcher
                .lock()
                .expect("matcher lock")
                .accept_token(bootstrap_token);
        }
        state.check_finished(false);
    }

    /// D role: end a request that needs no decode step after its remote
    /// prefill: the prefill node's
    /// token already finished it, or it was aborted while the KV moved.
    pub fn finish_remote_prefill_only(&mut self, rid: &str) -> Committed {
        let mut committed = Committed::default();
        let Some(state) = self.states.get(rid) else {
            return committed;
        };
        if !state.finished() {
            return committed;
        }
        let mut state = self.states.remove(rid).expect("present above");
        if !state.to_abort || state.abort_notify_client {
            committed.outputs = self.stream(&mut [(rid.to_owned(), &mut state)]);
        }
        let kind = if state.to_abort {
            ForwardEventKind::Abort
        } else {
            ForwardEventKind::Finish
        };
        committed.events.push(event(kind, rid, Vec::new(), 0));
        if let Some(timings) = state.timings.as_mut() {
            let now = wall_seconds();
            timings.mark_prefill_done(now);
            if !state.output_ids.is_empty() {
                timings.mark_first_token(now);
            }
        }
        self.log_end(rid, &state);
        committed.released.push(rid.to_owned());
        committed
    }

    /// A request's PD transfer failed. The scheduler already released it
    /// on the event; the client gets an abort, unless it ended before and
    /// tore down its stream.
    pub fn fail_transfer(&mut self, rid: &str) -> Committed {
        let mut committed = Committed::default();
        let Some(mut state) = self.states.remove(rid) else {
            return committed;
        };
        if !state.finished() {
            state.abort("PD/EPD remote transfer failed or timed out", false, None);
            state.finished_output = false;
            committed.outputs = self.stream(&mut [(rid.to_owned(), &mut state)]);
        } else if state.abort_notify_client {
            state.finished_output = false;
            committed.outputs = self.stream(&mut [(rid.to_owned(), &mut state)]);
        }
        committed.released.push(rid.to_owned());
        committed
    }

    /// Abort a request now, or when it registers if it has not yet.
    pub fn mark_abort(&mut self, rid: &str, notify_client: bool) {
        if let Some(state) = self.states.get_mut(rid) {
            let message = if notify_client {
                "Aborted by pause"
            } else {
                "AbortReq from client"
            };
            state.abort(message, notify_client, None);
            return;
        }
        self.sweep_pending_aborts();
        self.pending_aborts
            .push_back((rid.to_owned(), Instant::now()));
    }

    /// Drop aborts that waited too long for their request.
    pub fn sweep_pending_aborts(&mut self) {
        while let Some((_, at)) = self.pending_aborts.front() {
            if at.elapsed() < PENDING_ABORT_TTL {
                break;
            }
            self.pending_aborts.pop_front();
        }
    }

    /// The final output of a request that ended before admission (an
    /// invalid request).
    pub fn publish_finished_at_admission(
        &mut self,
        rid: &str,
        mut state: RequestState,
    ) -> Vec<StreamedOutput> {
        if let Some(index) = self.pending_aborts.iter().position(|(id, _)| id == rid) {
            self.pending_aborts.remove(index);
        }
        state.finished_output = false;
        self.stream(&mut [(rid.to_owned(), &mut state)])
    }

    /// The ids of the requests registered and not ended.
    pub fn rids(&self) -> Vec<String> {
        self.states.keys().cloned().collect()
    }

    /// End a request that finished before it reached the scheduler (aborted
    /// while a pause withheld it): a pause's abort
    /// notifies its client, a client's own abort does not.
    pub fn reap_unscheduled(&mut self, rid: &str) -> Committed {
        let mut committed = Committed::default();
        let Some(mut state) = self.states.remove(rid) else {
            return committed;
        };
        if state.abort_notify_client {
            state.finished_output = false;
            committed.outputs = self.stream(&mut [(rid.to_owned(), &mut state)]);
        }
        committed.released.push(rid.to_owned());
        committed
    }

    /// Fail a request that never reached the scheduler (an EPD receive that
    /// failed): its client gets an abort with `reason`.
    pub fn fail_unscheduled(
        &mut self,
        rid: &str,
        reason: &str,
        status_code: Option<u16>,
    ) -> Committed {
        let mut committed = Committed::default();
        let Some(mut state) = self.states.remove(rid) else {
            return committed;
        };
        if !state.finished() {
            state.abort(reason, true, status_code);
        }
        if !state.to_abort || state.abort_notify_client {
            state.finished_output = false;
            committed.outputs = self.stream(&mut [(rid.to_owned(), &mut state)]);
        }
        self.log_end(rid, &state);
        committed.released.push(rid.to_owned());
        committed
    }

    /// Finish the requests the scheduler aborted while planning.
    pub fn finish_scheduler_aborts(&mut self, aborts: &[SchedulerAbortView]) -> Committed {
        let mut committed = Committed::default();
        for abort in aborts {
            let Some(mut state) = self.states.remove(&abort.request_id) else {
                continue;
            };
            if !state.finished() {
                state.finished_reason = Some(FinishReason::Abort {
                    message: format!("Request aborted by the scheduler: {}", abort.detail),
                    err_type: AbortCode::CapacityAbort,
                    status_code: None,
                });
                committed.stats.capacity_aborts += 1;
                committed
                    .outputs
                    .extend(self.stream(&mut [(abort.request_id.clone(), &mut state)]));
            } else if state.abort_notify_client {
                committed
                    .outputs
                    .extend(self.stream(&mut [(abort.request_id.clone(), &mut state)]));
            }
            self.log_end(&abort.request_id, &state);
            committed.released.push(abort.request_id.clone());
        }
        committed
    }

    /// The logprobs `batch` computes beyond its sampled tokens'.
    ///
    /// An extend row feeds prompt positions `[prefix, prefix + length)`; its
    /// request wants position `p` when `logprob_start_len <= p <
    /// prompt_len - 1` (the last position predicts the first generated
    /// token, which is the output logprob's job) until its prompt logprobs
    /// are final. The targets, each row's next prompt token, are read on
    /// the device from the shifted input ids.
    pub fn logprob_plan(&self, batch: &ForwardBatchView) -> LogprobPlan {
        let mut plan = LogprobPlan::default();
        let mut input = InputLogprobPlan {
            row_starts: Vec::with_capacity(batch.num_extends),
            counts: Vec::with_capacity(batch.num_extends),
            position_starts: Vec::with_capacity(batch.num_extends),
            top_k: 0,
        };
        let mut row_offset = 0i64;
        for i in 0..batch.num_extends {
            let state = self.states.get(&batch.request_ids[i]);
            let chunk_start = i64::from(batch.extend_prefix_lens[i]);
            let chunk_len = i64::from(batch.input_lengths[i]);
            let mut row = (0, 0, 0);
            if let Some(state) = state {
                if state.return_logprob {
                    plan.sampled_top_k = plan.sampled_top_k.max(state.top_logprobs_num);
                }
                let lo = chunk_start.max(state.logprob_start_len);
                let hi = (chunk_start + chunk_len).min(state.prompt_len - 1);
                if state.wants_input_logprobs() && state.input_logprobs.is_none() && lo < hi {
                    row = (row_offset + lo - chunk_start, hi - lo, lo);
                    input.top_k = input.top_k.max(state.top_logprobs_num);
                }
            }
            input.row_starts.push(row.0);
            input.counts.push(row.1);
            input.position_starts.push(row.2);
            row_offset += chunk_len;
        }
        if input.counts.iter().any(|&count| count > 0) {
            plan.input = Some(input);
        }
        plan
    }

    /// Place a forward's prompt logprobs into their requests: the flat
    /// result follows the plan's extend rows, each starting at the prompt
    /// position the plan recorded.
    fn record_input_logprobs(
        &mut self,
        batch: &ForwardBatchView,
        plan: &InputLogprobPlan,
        result: &ForwardResult,
    ) -> Result<(), CommitError> {
        let Some(values) = &result.input_token_logprobs else {
            return Ok(());
        };
        let rows: i64 = plan.counts.iter().sum();
        if values.len() as i64 != rows {
            return Err(CommitError::Shape {
                rows: rows as usize,
                column: "input_token_logprobs",
                found: values.len(),
            });
        }
        let mut cursor = 0usize;
        for (i, &count) in plan.counts.iter().enumerate() {
            let count = count as usize;
            if count == 0 {
                continue;
            }
            let rid = &batch.request_ids[i];
            if let Some(state) = self.states.get_mut(rid) {
                if state.input_logprob_rows.is_some() {
                    let top = result.input_top_logprobs.as_ref().map(|(values, ids)| {
                        (
                            window_ref(values, cursor, count),
                            window_ref(ids, cursor, count),
                        )
                    });
                    state
                        .record_input_logprobs(
                            plan.position_starts[i],
                            &values[cursor..cursor + count],
                            top,
                        )
                        .map_err(|detail| CommitError::InputLogprobs {
                            rid: rid.clone(),
                            detail,
                        })?;
                }
            }
            cursor += count;
        }
        Ok(())
    }

    /// The n-gram windows of `batch` (`ngram_inputs_for_forward`), from the
    /// tokens committed so far: a decode row's in-flight predecessors stay
    /// on the device.
    pub fn ngram_windows(
        &self,
        batch: &ForwardBatchView,
        context_len: usize,
    ) -> Result<NgramWindows, CommitError> {
        let mut windows = NgramWindows::default();
        for (i, rid) in batch.request_ids.iter().enumerate() {
            let state = self
                .states
                .get(rid)
                .ok_or_else(|| CommitError::NgramWindow(rid.clone()))?;
            let prompt_len = state.prompt_ids.len() as i64;
            let total = prompt_len + state.output_ids.len() as i64;
            let start = if i < batch.num_extends {
                let start = i64::from(batch.extend_prefix_lens[i]);
                if start < 0 || start + i64::from(batch.input_lengths[i]) > total {
                    return Err(CommitError::NgramWindow(rid.clone()));
                }
                start
            } else {
                if total == 0 {
                    return Err(CommitError::NgramWindow(rid.clone()));
                }
                total - 1
            };
            let token = |p: i64| -> i32 {
                if p < 0 || p >= total {
                    -1
                } else if p < prompt_len {
                    state.prompt_ids[p as usize]
                } else {
                    state.output_ids[(p - prompt_len) as usize]
                }
            };
            windows.tokens.push(
                (0..=context_len as i64)
                    .map(|back| token(start - back))
                    .collect(),
            );
            windows.positions.push(start);
        }
        Ok(windows)
    }

    /// Commit one forward's results; `input_logprobs` is the prompt-row
    /// plan it was dispatched with.
    pub fn commit(
        &mut self,
        batch: &ForwardBatchView,
        input_logprobs: Option<&InputLogprobPlan>,
        result: &ForwardResult,
    ) -> Result<Committed, CommitError> {
        let rows = batch.request_ids.len();
        let check = |column, found| {
            if found == rows {
                Ok(())
            } else {
                Err(CommitError::Shape {
                    rows,
                    column,
                    found,
                })
            }
        };
        check("output_lengths", result.output_lengths.len())?;
        if let Some(flags) = &result.nan_flags {
            check("nan_flags", flags.len())?;
        }

        // The request stats' clock: a decode step lasts from the previous
        // commit, and a round with prefill rows interrupts the decodes in it.
        let (stats_now, step, prefilling_others) = if self.finish_log.request_stats {
            let now = wall_seconds();
            let step = if self.last_commit_at > 0.0 {
                now - self.last_commit_at
            } else {
                0.0
            };
            self.last_commit_at = now;
            (now, step, batch.num_extends > 0)
        } else {
            (0.0, 0.0, false)
        };

        if let Some(plan) = input_logprobs {
            self.record_input_logprobs(batch, plan, result)?;
        }

        // Replayed rows re-feed cached positions: the hit reaches the end of
        // the replay window.
        for i in 0..batch.num_extends {
            if let Some(state) = self.states.get_mut(&batch.request_ids[i]) {
                let hit = i64::from(batch.extend_prefix_lens[i] + batch.extend_replay_lens[i]);
                state.cached_tokens += (hit - state.computed_length).max(0);
            }
        }

        let mut committed = Committed::default();
        let mut streamed: Vec<String> = Vec::new();
        let mut pt = 0usize;
        for i in 0..rows {
            let rid = &batch.request_ids[i];
            let output_length = result.output_lengths[i].max(0) as usize;
            let mut model_ids = window(&result.output_tokens, pt, output_length);
            let mut model_logprobs = result
                .output_logprobs
                .as_ref()
                .map(|lps| window(lps, pt, output_length));
            let is_decode = i >= batch.num_extends;
            // The prefill rows' sampled-token alternatives, in prefill output
            // order.
            let model_top = match &result.output_top_logprobs {
                Some((values, ids)) if !is_decode => Some((
                    window_ref(values, pt, output_length),
                    window_ref(ids, pt, output_length),
                )),
                _ => None,
            };
            // A decode row stores its whole verify window however many tokens
            // it keeps; a prefill row stores what it samples.
            pt += if is_decode {
                self.spec_num_tokens.unwrap_or(1)
            } else {
                output_length
            };
            // A chunk that does not complete the prompt owes no token; the
            // sampled one is garbage.
            let mid_chunk = !is_decode
                && batch.extend_prefix_lens[i] + batch.input_lengths[i] < batch.prefill_lengths[i];
            let nan_flag = result.nan_flags.as_ref().is_some_and(|flags| flags[i] != 0);

            let Some(state) = self.states.get_mut(rid) else {
                // It ended while this forward was in flight. The forward still
                // landed; its tokens keep the scheduler's frontier exact.
                committed.events.push(extend_result(
                    rid,
                    if mid_chunk { Vec::new() } else { model_ids },
                ));
                continue;
            };
            if i < batch.num_extends {
                state.computed_length =
                    i64::from(batch.input_lengths[i] + batch.extend_prefix_lens[i]);
            } else {
                state.computed_length += i64::from(batch.input_lengths[i]);
            }
            if mid_chunk {
                // The chunk's KV landed: report it so the scheduler stops
                // counting the forward against the pages.
                if nan_flag {
                    state.numerical_error_detected = true;
                }
                committed.events.push(extend_result(rid, Vec::new()));
                continue;
            }
            if !state.prefill_finished() {
                continue;
            }
            if is_decode && output_length == 0 {
                return Err(CommitError::EmptyDecode(rid.clone()));
            }
            if !is_decode {
                state
                    .finalize_input_logprobs()
                    .map_err(|detail| CommitError::InputLogprobs {
                        rid: rid.clone(),
                        detail,
                    })?;
            }
            if let Some(timings) = state.timings.as_mut() {
                timings.mark_prefill_done(stats_now);
                if is_decode {
                    timings.record_decode_step(step, prefilling_others);
                }
            }
            if is_decode && self.spec_num_tokens.is_some() {
                committed.stats.spec_decode_slots += 1;
                committed.stats.spec_accepted_tokens += output_length as i64 - 1;
            }

            let nan_detected = nan_flag || state.numerical_error_detected;
            if nan_detected && !state.finished() {
                committed.stats.nan_aborts += 1;
                state.finished_reason = Some(FinishReason::Abort {
                    message: NUMERICAL_ERROR.to_owned(),
                    err_type: AbortCode::NumericalError,
                    status_code: None,
                });
                // One sanitized token keeps the accounting of a mid-step finish.
                model_ids.truncate(1);
                if let Some(lps) = model_logprobs.as_mut() {
                    lps.truncate(1);
                }
                tracing::warn!(rid, "request terminated: NaN detected in logits");
            }
            if is_decode && self.spec_num_tokens.is_some() {
                state.spec_verify_ct += 1;
                if let Some(limit) = self.physical_context_len {
                    let total =
                        state.prompt_len + state.output_ids.len() as i64 + output_length as i64;
                    if total > limit {
                        return Err(CommitError::PhysicalExtent {
                            rid: rid.clone(),
                            total,
                            limit,
                        });
                    }
                }
            }

            // Where the data plane advanced the matcher it names the token
            // that ended the grammar; otherwise the commit advances it.
            let hostfunc = result.grammar_terminated_at.is_some();
            let terminated_at = match (&result.grammar_terminated_at, &state.grammar) {
                (Some(indices), Some(_)) => indices.get(i).copied().unwrap_or(-1),
                _ => -1,
            };
            for (j, &token) in model_ids.iter().enumerate() {
                state.output_ids.push(token);
                if !hostfunc {
                    if let Some(grammar) = &state.grammar {
                        let mut matcher = grammar.matcher.lock().expect("matcher lock");
                        if !matcher.is_terminated() && !matcher.accept_token(token) {
                            return Err(CommitError::GrammarRejected {
                                rid: rid.clone(),
                                token,
                            });
                        }
                    }
                }
                if state.return_logprob {
                    if let Some(lp) = model_logprobs.as_ref().and_then(|lps| lps.get(j)) {
                        state.output_logprobs.push(*lp);
                    }
                }
                let finished_before = state.finished();
                if let Some((top_values, top_ids)) = state.output_top_logprobs.as_mut() {
                    match model_top.filter(|(values, _)| j < values.len()) {
                        Some((values, ids)) => {
                            let k = state.top_logprobs_num;
                            top_values.push(values[j][..k.min(values[j].len())].to_vec());
                            top_ids.push(ids[j][..k.min(ids[j].len())].to_vec());
                        }
                        // The ingress takes a top-k request only when every
                        // token is sampled in a prefill, which reports the
                        // alternatives; end it rather than ship a token
                        // without them.
                        None if !finished_before => {
                            state.finished_reason = Some(FinishReason::abort(
                                "top-k logprobs are not computed for this step (decode steps \
                                 return none)",
                            ));
                        }
                        None => {}
                    }
                }
                if terminated_at == j as i32 && !state.finished() {
                    // Ahead of length and stop tokens at the same step, not of an abort.
                    state.finished_reason = Some(FinishReason::MatchedToken(token));
                } else {
                    state.check_finished(hostfunc);
                }
                if state.finished() {
                    state.accept_draft_tokens = Some(if state.spec_verify_ct > 0 {
                        (state.output_ids.len() as f64 - 1.0) / state.spec_verify_ct as f64
                    } else {
                        0.0
                    });
                    if self.finish_log.enabled && !self.finish_log.request_stats {
                        tracing::info!("{}", request_stats::accept_length_line(rid, state));
                    }
                    break;
                }
            }
            if !state.output_ids.is_empty() {
                if let Some(timings) = state.timings.as_mut() {
                    timings.mark_first_token(stats_now);
                }
            }
            // P role: the final chunk's drafter candidates ride its result,
            // so the remote decode the scheduler emits is self-contained.
            let mut spec_candidate_ids = Vec::new();
            if self.role == Role::Prefill && !model_ids.is_empty() && !nan_detected {
                if let Some(candidates) =
                    result.next_input_ids.as_ref().and_then(|rows| rows.get(i))
                {
                    if let Some(&first) = candidates.first() {
                        if first != model_ids[0] {
                            return Err(CommitError::BootstrapMismatch {
                                rid: rid.clone(),
                                sampled: model_ids[0],
                                candidate: first,
                            });
                        }
                    }
                    spec_candidate_ids = candidates.clone();
                }
            }
            // The scheduler gets every kept token: the device consumed the
            // whole round even when a condition only this side checks ended
            // the request earlier.
            let mut extend = extend_result(rid, model_ids);
            extend.spec_candidate_ids = spec_candidate_ids;
            committed.events.push(extend);

            if self.role == Role::Prefill {
                // A numerically corrupted prompt's KV must never transfer:
                // the abort stops the remote decode. Otherwise the transfer
                // owns the pages until it succeeds or fails, and the output
                // goes out then.
                if nan_detected {
                    committed.events.push(finish(rid, true));
                    self.release(rid, &mut committed);
                }
                continue;
            }
            if state.to_abort && state.finished() {
                committed.events.push(finish(rid, nan_detected));
                if state.abort_notify_client {
                    streamed.push(rid.clone());
                } else {
                    self.release(rid, &mut committed);
                }
                continue;
            }
            streamed.push(rid.clone());
            if state.finished() {
                committed.events.push(finish(rid, nan_detected));
            } else if is_decode {
                committed.events.push(event(
                    ForwardEventKind::UpdateReserveNumTokens,
                    rid,
                    Vec::new(),
                    output_length as i32,
                ));
            }
        }

        let mut states: Vec<(String, RequestState)> = streamed
            .iter()
            .filter_map(|rid| self.states.remove_entry(rid))
            .collect();
        let mut refs: Vec<(String, &mut RequestState)> = states
            .iter_mut()
            .map(|(rid, state)| (rid.clone(), state))
            .collect();
        committed.outputs = self.stream(&mut refs);
        for (rid, state) in states {
            if state.finished() {
                self.log_end(&rid, &state);
                committed.released.push(rid);
            } else {
                self.states.insert(rid, state);
            }
        }
        Ok(committed)
    }

    fn release(&mut self, rid: &str, committed: &mut Committed) {
        if let Some(state) = self.states.remove(rid) {
            self.log_end(rid, &state);
            committed.released.push(rid.to_owned());
        }
    }

    /// The outputs due for `states` this step (`stream_output`).
    fn stream(&self, states: &mut [(String, &mut RequestState)]) -> Vec<StreamedOutput> {
        let mut outputs = Vec::new();
        for (rid, state) in states.iter_mut() {
            if !state.output_due(self.stream_interval) {
                continue;
            }
            if state.finished() {
                state.finished_output = true;
            }
            let sent = state.send_token_offset;
            let output_ids = state.output_ids[sent..].to_vec();
            state.send_token_offset = state.output_ids.len();
            let n_new = output_ids.len();
            let output_logprobs = if state.return_logprob {
                let start = state.output_logprobs.len().saturating_sub(n_new);
                state.output_logprobs[start..].to_vec()
            } else {
                Vec::new()
            };
            // Prompt logprobs ship once, in the first output after the
            // prompt's final chunk committed.
            let input_logprobs = if state.input_logprobs_sent {
                None
            } else {
                state.input_logprobs.clone()
            };
            state.input_logprobs_sent |= input_logprobs.is_some();
            let output_top_logprobs = state.output_top_logprobs.as_ref().map(|(values, ids)| {
                (
                    values[values.len().saturating_sub(n_new)..].to_vec(),
                    ids[ids.len().saturating_sub(n_new)..].to_vec(),
                )
            });
            outputs.push(StreamedOutput {
                rid: rid.clone(),
                output_ids,
                finish_reason: state.finished_reason.as_ref().map(FinishReason::wire_type),
                prompt_tokens: state.prompt_len,
                completion_tokens: state.output_ids.len() as i64,
                cached_tokens: state.cached_tokens,
                output_logprobs,
                finish_details: state.finished_reason.as_ref().map(FinishReason::details),
                spec: self
                    .spec_num_tokens
                    .map(|_| (state.spec_verify_ct, state.accept_draft_tokens)),
                input_logprobs,
                output_top_logprobs,
            });
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn batch(rows: &[(&str, i32, i32, i32)], num_extends: usize) -> ForwardBatchView {
        // (rid, input_length, extend_prefix_len, prefill_length)
        ForwardBatchView {
            request_ids: rows.iter().map(|r| r.0.to_owned()).collect(),
            request_pool_indices: (1..=rows.len() as i32).collect(),
            input_lengths: rows.iter().map(|r| r.1).collect(),
            prefill_lengths: rows.iter().map(|r| r.3).collect(),
            extend_prefix_lens: rows[..num_extends].iter().map(|r| r.2).collect(),
            extend_replay_lens: vec![0; num_extends],
            num_extends,
        }
    }

    fn result(tokens: &[i32], lengths: &[i32]) -> ForwardResult {
        ForwardResult {
            output_tokens: tokens.to_vec(),
            output_lengths: lengths.to_vec(),
            output_logprobs: None,
            nan_flags: None,
            next_input_ids: None,
            serving_compiles: None,
            grammar_terminated_at: None,
            input_token_logprobs: None,
            input_top_logprobs: None,
            output_top_logprobs: None,
        }
    }

    const NO_FINISH_LOG: FinishLog = FinishLog {
        enabled: false,
        request_stats: false,
    };

    fn processor(spec_num_tokens: Option<usize>) -> OutputProcessor {
        OutputProcessor::new(spec_num_tokens, 1, Some(4096), Role::Plain, NO_FINISH_LOG)
    }

    fn register(p: &mut OutputProcessor, rid: &str, prompt: i64, max_new: i64, stop: &[i32]) {
        let stops: BTreeSet<i32> = stop.iter().copied().collect();
        p.register(
            rid.to_owned(),
            RequestState::new(vec![0; prompt as usize], max_new, stops, true, None, false),
        );
    }

    fn kinds(committed: &Committed) -> Vec<(String, ForwardEventKind, Vec<i32>)> {
        committed
            .events
            .iter()
            .map(|e| (e.request_id.clone(), e.kind, e.tokens.clone()))
            .collect()
    }

    /// A request scoring its prompt from `start`, with `top_k` alternatives.
    fn register_scored(
        p: &mut OutputProcessor,
        rid: &str,
        prompt: Vec<i32>,
        start: i64,
        top_k: usize,
        computes_prompt_logprobs: bool,
    ) {
        let mut state = RequestState::new(prompt, 4, BTreeSet::new(), true, None, true);
        state.set_logprobs(crate::request::LogprobRequest {
            start,
            top_logprobs_num: top_k,
            computes_prompt_logprobs,
        });
        p.register(rid.to_owned(), state);
    }

    /// The forward result of a plan's rows: each row's logprob is
    /// `-position`, its alternatives `k` values below it with ids from 100.
    fn scored(plan: &InputLogprobPlan, tokens: &[i32]) -> ForwardResult {
        let positions: Vec<i64> = plan
            .counts
            .iter()
            .zip(&plan.position_starts)
            .flat_map(|(&count, &start)| start..start + count)
            .collect();
        let top = |p: &i64| {
            (
                (0..plan.top_k).map(|r| -(*p as f64) - r as f64).collect(),
                (0..plan.top_k).map(|r| 100 + r as i32).collect(),
            )
        };
        ForwardResult {
            input_token_logprobs: Some(positions.iter().map(|&p| -(p as f64)).collect()),
            input_top_logprobs: (plan.top_k > 0).then(|| positions.iter().map(top).unzip()),
            ..result(tokens, &vec![1; tokens.len()])
        }
    }

    /// A request with `prompt` and the tokens it already generated.
    fn register_history(p: &mut OutputProcessor, rid: &str, prompt: &[i32], output: &[i32]) {
        let mut state = RequestState::new(prompt.to_vec(), 8, BTreeSet::new(), true, None, false);
        state.output_ids = output.to_vec();
        p.register(rid.to_owned(), state);
    }

    /// One forward over `rows` of (rid, slot, extend prefix length), the
    /// first `num_extends` of them extends.
    fn history_batch(rows: &[(&str, i32, i32)], num_extends: usize) -> ForwardBatchView {
        ForwardBatchView {
            request_ids: rows.iter().map(|r| r.0.to_owned()).collect(),
            request_pool_indices: rows.iter().map(|r| r.1).collect(),
            input_lengths: vec![1; rows.len()],
            prefill_lengths: vec![0; rows.len()],
            extend_prefix_lens: rows[..num_extends].iter().map(|r| r.2).collect(),
            extend_replay_lens: vec![0; num_extends],
            num_extends,
        }
    }

    #[test]
    fn history_seeds_cover_the_prompt_and_output_prefix() {
        let mut p = processor(None);
        register_history(&mut p, "a", &[10, 11, 12, 13], &[20, 21]);
        register_history(&mut p, "b", &[30, 31], &[]);
        let seeds = HistoryRows::default()
            .seeds(&history_batch(&[("a", 1, 5), ("b", 2, 2)], 2), &p)
            .unwrap();
        assert_eq!(
            seeds,
            Some(HistorySeeds {
                slots: vec![1, 2],
                prefix_lengths: vec![5, 2],
                tokens: vec![vec![10, 11, 12, 13, 20], vec![30, 31]],
            })
        );
    }

    #[test]
    fn a_fresh_extend_seeds_nothing_and_claims_its_row() {
        let mut p = processor(None);
        register_history(&mut p, "a", &[10, 11, 12, 13], &[20, 21]);
        let mut rows = HistoryRows::default();
        assert_eq!(
            rows.seeds(&history_batch(&[("a", 1, 0)], 1), &p).unwrap(),
            None
        );
        // Its later chunks and decodes extend the row it started.
        assert_eq!(
            rows.seeds(&history_batch(&[("a", 1, 4)], 1), &p).unwrap(),
            None
        );
        assert_eq!(
            rows.seeds(&history_batch(&[("a", 1, 0)], 0), &p).unwrap(),
            None
        );
    }

    #[test]
    fn a_remote_prefill_landing_seeds_its_first_decode() {
        // A PD decode node's first local forward for a request is a decode
        // over a prompt prefilled elsewhere: the row holds nothing of it.
        let mut p = processor(None);
        register_history(&mut p, "b", &[30, 31, 32], &[40]);
        let mut rows = HistoryRows::default();
        let landing = history_batch(&[("b", 2, 0)], 0);
        assert_eq!(
            rows.seeds(&landing, &p).unwrap(),
            Some(HistorySeeds {
                slots: vec![2],
                prefix_lengths: vec![3],
                tokens: vec![vec![30, 31, 32]],
            })
        );
        assert_eq!(rows.seeds(&landing, &p).unwrap(), None);
    }

    #[test]
    fn a_slot_handoff_and_a_recovery_reseed() {
        let mut p = processor(None);
        register_history(&mut p, "a", &[10, 11, 12, 13], &[20, 21]);
        register_history(&mut p, "b", &[30, 31], &[]);
        let mut rows = HistoryRows::default();
        rows.seeds(&history_batch(&[("a", 1, 0)], 1), &p).unwrap();
        // "b" takes slot 1 over with a prefix-cache hit: the row holds "a".
        let handoff = rows.seeds(&history_batch(&[("b", 1, 2)], 1), &p).unwrap();
        assert_eq!(handoff.unwrap().slots, [1]);
        // "a" recovers into slot 1 after "b" used it: reseed its prefix.
        let recovery = rows.seeds(&history_batch(&[("a", 1, 4)], 1), &p).unwrap();
        assert_eq!(recovery.unwrap().prefix_lengths, [4]);
    }

    #[test]
    fn a_reused_request_id_does_not_inherit_the_row() {
        // Clients may reuse a finished request's id; at batch size 1 the
        // new request lands in the same slot. Ownership follows the
        // admission, not the id.
        let mut p = processor(None);
        register_history(&mut p, "a", &[10, 11, 12, 13], &[]);
        let mut rows = HistoryRows::default();
        rows.seeds(&history_batch(&[("a", 1, 0)], 1), &p).unwrap();
        p.states.remove("a");
        register_history(&mut p, "a", &[50, 51, 52, 53], &[]);
        assert_eq!(
            rows.seeds(&history_batch(&[("a", 1, 3)], 1), &p).unwrap(),
            Some(HistorySeeds {
                slots: vec![1],
                prefix_lengths: vec![3],
                tokens: vec![vec![50, 51, 52]],
            })
        );
    }

    #[test]
    fn a_seed_past_the_request_tokens_is_an_engine_error() {
        let mut p = processor(None);
        register_history(&mut p, "b", &[30, 31], &[]);
        let past = history_batch(&[("b", 2, 3)], 1);
        assert!(matches!(
            HistoryRows::default().seeds(&past, &p),
            Err(CommitError::History(_))
        ));
    }

    #[test]
    fn prompt_logprobs_accumulate_over_chunks_and_ship_once() {
        let mut p = processor(None);
        register_scored(&mut p, "r", vec![5, 6, 7, 8, 9, 10], 1, 0, true);
        // Two chunks, [0, 3) then [3, 6); positions 1..5 are scored.
        let first = batch(&[("r", 3, 0, 6)], 1);
        let plan = p.logprob_plan(&first).input.unwrap();
        assert_eq!(
            (
                plan.row_starts.clone(),
                plan.counts.clone(),
                plan.position_starts.clone()
            ),
            (vec![1], vec![2], vec![1])
        );
        let c = p.commit(&first, Some(&plan), &scored(&plan, &[0])).unwrap();
        assert!(c.outputs.is_empty());
        let second = batch(&[("r", 3, 3, 6)], 1);
        let plan = p.logprob_plan(&second).input.unwrap();
        // The last prompt position predicts the first token: not scored.
        assert_eq!(
            (plan.counts.clone(), plan.position_starts.clone()),
            (vec![2], vec![3])
        );
        let c = p
            .commit(&second, Some(&plan), &scored(&plan, &[11]))
            .unwrap();
        let logprobs = c.outputs[0].input_logprobs.clone().unwrap();
        assert_eq!(
            logprobs.values,
            [None, Some(-1.0), Some(-2.0), Some(-3.0), Some(-4.0)]
        );
        assert_eq!(logprobs.ids, [6, 7, 8, 9, 10]);
        // Finalized: no later forward gathers rows, no later output ships them.
        assert!(p.logprob_plan(&batch(&[("r", 1, 0, 6)], 0)).input.is_none());
        let c = p
            .commit(&batch(&[("r", 1, 0, 6)], 0), None, &result(&[12], &[1]))
            .unwrap();
        assert!(c.outputs[0].input_logprobs.is_none());
    }

    #[test]
    fn a_re_prefill_overwrites_the_same_prompt_positions() {
        let mut p = processor(None);
        register_scored(&mut p, "r", vec![5, 6, 7, 8], 0, 0, true);
        let chunk = batch(&[("r", 2, 0, 4)], 1);
        let plan = p.logprob_plan(&chunk).input.unwrap();
        p.commit(&chunk, Some(&plan), &scored(&plan, &[0])).unwrap();
        // The same chunk again (a retraction re-prefill): its rows land on
        // the positions they score, not after the earlier ones.
        let plan = p.logprob_plan(&chunk).input.unwrap();
        assert_eq!(plan.position_starts, [0]);
        p.commit(&chunk, Some(&plan), &scored(&plan, &[0])).unwrap();
        let last = batch(&[("r", 2, 2, 4)], 1);
        let plan = p.logprob_plan(&last).input.unwrap();
        let c = p.commit(&last, Some(&plan), &scored(&plan, &[9])).unwrap();
        assert_eq!(
            c.outputs[0].input_logprobs.as_ref().unwrap().values,
            [None, Some(0.0), Some(-1.0), Some(-2.0)]
        );
    }

    #[test]
    fn a_gap_in_the_prompt_logprobs_is_an_engine_error() {
        let mut p = processor(None);
        register_scored(&mut p, "r", vec![5, 6, 7, 8], 0, 0, true);
        // The whole prompt in one forward whose plan gathered no rows.
        assert!(matches!(
            p.commit(&batch(&[("r", 4, 0, 4)], 1), None, &result(&[9], &[1])),
            Err(CommitError::InputLogprobs { .. })
        ));
    }

    #[test]
    fn top_k_alternatives_ride_the_prompt_rows_and_the_sampled_token() {
        let mut p = processor(None);
        register_scored(&mut p, "r", vec![5, 6, 7], 0, 2, true);
        let prompt = batch(&[("r", 3, 0, 3)], 1);
        let plan = p.logprob_plan(&prompt);
        assert_eq!(plan.sampled_top_k, 2);
        let input = plan.input.unwrap();
        let mut forward = scored(&input, &[9]);
        forward.output_top_logprobs = Some((vec![vec![-0.5, -1.5, -2.5]], vec![vec![9, 4, 3]]));
        let c = p.commit(&prompt, Some(&input), &forward).unwrap();
        let output = &c.outputs[0];
        let (values, ids) = output.input_logprobs.as_ref().unwrap().top.clone().unwrap();
        assert_eq!(
            values,
            [None, Some(vec![0.0, -1.0]), Some(vec![-1.0, -2.0])]
        );
        assert_eq!(ids, [None, Some(vec![100, 101]), Some(vec![100, 101])]);
        // Each request keeps its own count of the sampled token's.
        assert_eq!(
            output.output_top_logprobs,
            Some((vec![vec![-0.5, -1.5]], vec![vec![9, 4]]))
        );
        // A later token comes from a decode step, which computes none.
        let c = p
            .commit(&batch(&[("r", 1, 0, 3)], 0), None, &result(&[10], &[1]))
            .unwrap();
        assert_eq!(
            c.outputs[0].finish_reason,
            Some("abort"),
            "a token without the alternatives its request asked for ends it"
        );
    }

    #[test]
    fn the_decode_role_ships_no_prompt_logprobs() {
        let mut p = OutputProcessor::new(None, 1, None, Role::Decode, NO_FINISH_LOG);
        register_scored(&mut p, "r", vec![5, 6, 7], 0, 0, false);
        assert!(!p.state("r").unwrap().wants_input_logprobs());
        p.remote_prefill_done("r", 9, 0, Some(-0.25));
        let decode = ForwardResult {
            output_logprobs: Some(vec![-0.5]),
            ..result(&[10], &[1])
        };
        let c = p
            .commit(&batch(&[("r", 1, 0, 3)], 0), None, &decode)
            .unwrap();
        let output = &c.outputs[0];
        assert!(output.input_logprobs.is_none());
        // The prefill node's token carries its logprob.
        assert_eq!(output.output_ids, [9, 10]);
        assert_eq!(output.output_logprobs, [-0.25, -0.5]);
    }

    #[test]
    fn the_prefill_role_ships_its_prompt_logprobs_with_its_final_output() {
        let mut p = OutputProcessor::new(None, 1, None, Role::Prefill, NO_FINISH_LOG);
        register_scored(&mut p, "r", vec![5, 6, 7], 0, 0, true);
        let prompt = batch(&[("r", 3, 0, 3)], 1);
        let plan = p.logprob_plan(&prompt).input.unwrap();
        let c = p
            .commit(&prompt, Some(&plan), &scored(&plan, &[9]))
            .unwrap();
        // Nothing streams until the KV reached the decode node.
        assert!(c.outputs.is_empty());
        let c = p.finish_prefill("r");
        assert_eq!(
            c.outputs[0].input_logprobs.as_ref().unwrap().values,
            [None, Some(0.0), Some(-1.0)]
        );
    }

    #[test]
    fn ngram_windows_follow_the_committed_tokens() {
        let mut p = OutputProcessor::new(None, 1, None, Role::Plain, NO_FINISH_LOG);
        p.register(
            "r".to_owned(),
            RequestState::new(vec![10, 11, 12, 13], 8, BTreeSet::new(), true, None, false),
        );
        // A chunk starting at position 2 of the prompt.
        let windows = p.ngram_windows(&batch(&[("r", 2, 2, 4)], 1), 3).unwrap();
        assert_eq!(windows.tokens, [vec![12, 11, 10, -1]]);
        assert_eq!(windows.positions, [2]);
        // After the prompt completes and a token is generated, a decode row
        // sits on the newest committed token.
        p.commit(&batch(&[("r", 4, 0, 4)], 1), None, &result(&[20], &[1]))
            .unwrap();
        let windows = p.ngram_windows(&batch(&[("r", 1, 0, 4)], 0), 3).unwrap();
        assert_eq!(windows.tokens, [vec![20, 13, 12, 11]]);
        assert_eq!(windows.positions, [4]);
        assert!(matches!(
            p.ngram_windows(&batch(&[("other", 1, 0, 4)], 0), 3),
            Err(CommitError::NgramWindow(_))
        ));
    }

    #[test]
    fn a_result_survives_the_pipeline_broadcast() {
        let full = ForwardResult {
            output_tokens: vec![5, -1, 7],
            output_lengths: vec![2, 1],
            output_logprobs: Some(vec![-0.25, -1.5, f64::NEG_INFINITY]),
            nan_flags: Some(vec![0, 1]),
            next_input_ids: Some(vec![vec![5, 6], vec![]]),
            serving_compiles: None,
            grammar_terminated_at: Some(vec![-1, 0]),
            input_token_logprobs: Some(vec![-0.5, -2.0]),
            ..ForwardResult::default()
        };
        let bare = ForwardResult {
            output_tokens: vec![9],
            output_lengths: vec![1],
            ..Default::default()
        };
        for result in [full, bare] {
            let bytes = result.to_bytes();
            assert_eq!(ForwardResult::from_bytes(&bytes).as_ref(), Some(&result));
            assert_eq!(ForwardResult::from_bytes(&bytes[..bytes.len() - 1]), None);
        }
    }

    #[test]
    fn a_request_streams_until_its_stop_token() {
        let mut p = processor(None);
        register(&mut p, "r", 4, 10, &[2]);

        let c = p
            .commit(&batch(&[("r", 4, 0, 4)], 1), None, &result(&[7], &[1]))
            .unwrap();
        assert_eq!(
            kinds(&c),
            [("r".into(), ForwardEventKind::ExtendResult, vec![7])]
        );
        assert_eq!(
            (c.outputs[0].output_ids.clone(), c.outputs[0].finish_reason),
            (vec![7], None)
        );
        assert_eq!(c.outputs[0].prompt_tokens, 4);

        let c = p
            .commit(&batch(&[("r", 1, 0, 4)], 0), None, &result(&[8], &[1]))
            .unwrap();
        assert_eq!(c.events[1].kind, ForwardEventKind::UpdateReserveNumTokens);
        assert_eq!(c.events[1].reserve_num_tokens, 1);

        let c = p
            .commit(&batch(&[("r", 1, 0, 4)], 0), None, &result(&[2], &[1]))
            .unwrap();
        assert_eq!(
            kinds(&c),
            [
                ("r".into(), ForwardEventKind::ExtendResult, vec![2]),
                ("r".into(), ForwardEventKind::Finish, vec![]),
            ]
        );
        assert_eq!(
            (c.outputs[0].finish_reason, c.outputs[0].completion_tokens),
            (Some("stop"), 3)
        );
        assert_eq!(c.released, ["r"]);
        assert_eq!(p.num_running(), 0);
    }

    #[test]
    fn a_mid_prompt_chunk_reports_its_kv_and_outputs_nothing() {
        let mut p = processor(None);
        register(&mut p, "r", 10, 10, &[]);
        let c = p
            .commit(&batch(&[("r", 6, 0, 10)], 1), None, &result(&[99], &[1]))
            .unwrap();
        assert_eq!(
            kinds(&c),
            [("r".into(), ForwardEventKind::ExtendResult, vec![])]
        );
        assert!(c.outputs.is_empty());
        assert_eq!(p.state("r").unwrap().computed_length, 6);

        let c = p
            .commit(&batch(&[("r", 4, 6, 10)], 1), None, &result(&[5], &[1]))
            .unwrap();
        assert_eq!(c.outputs[0].output_ids, [5]);
    }

    #[test]
    fn the_budget_finishes_with_length() {
        let mut p = processor(None);
        register(&mut p, "r", 2, 1, &[]);
        let c = p
            .commit(&batch(&[("r", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        assert_eq!(c.outputs[0].finish_reason, Some("length"));
        assert_eq!(c.events[1].kind, ForwardEventKind::Finish);
    }

    #[test]
    fn a_client_abort_finishes_silently_at_the_next_commit() {
        let mut p = processor(None);
        register(&mut p, "r", 2, 10, &[]);
        p.commit(&batch(&[("r", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        p.mark_abort("r", false);
        let c = p
            .commit(&batch(&[("r", 1, 0, 2)], 0), None, &result(&[4], &[1]))
            .unwrap();
        assert_eq!(c.events[1].kind, ForwardEventKind::Finish);
        assert!(c.outputs.is_empty(), "the client tore down already");
        assert_eq!(c.released, ["r"]);
    }

    #[test]
    fn an_abort_before_registration_applies_on_register() {
        let mut p = processor(None);
        p.mark_abort("early", false);
        register(&mut p, "early", 2, 10, &[]);
        assert!(p.state("early").unwrap().to_abort);
    }

    #[test]
    fn a_nan_aborts_the_request_and_keeps_one_token() {
        let mut p = processor(Some(3));
        register(&mut p, "r", 2, 10, &[]);
        p.commit(&batch(&[("r", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        let mut nan = result(&[4, 5, 6], &[3]);
        nan.nan_flags = Some(vec![1]);
        let c = p.commit(&batch(&[("r", 3, 0, 2)], 0), None, &nan).unwrap();
        assert_eq!(
            kinds(&c),
            [
                ("r".into(), ForwardEventKind::ExtendResult, vec![4]),
                ("r".into(), ForwardEventKind::Abort, vec![]),
            ]
        );
        assert_eq!(c.outputs[0].finish_reason, Some("abort"));
    }

    /// A matcher for `a` then `b` over `ts-grammar`'s nine-token vocabulary.
    fn grammar() -> crate::request::RequestGrammar {
        let info = ts_grammar::TokenizerInfo::from_json(include_str!(
            "../../ts-grammar/tests/fixtures/toy_tokenizer_info.json"
        ))
        .unwrap();
        let compiled = ts_grammar::Compiler::new(&info, 1)
            .unwrap()
            .compile(ts_grammar::GrammarKind::Regex, "ab", true)
            .unwrap();
        let (handle, matcher) = ts_grammar::table::register(compiled.matcher(8).unwrap());
        crate::request::RequestGrammar { handle, matcher }
    }

    #[test]
    fn the_token_that_ended_the_grammar_on_the_data_plane_finishes_the_request() {
        let mut p = processor(Some(3));
        register(&mut p, "r", 2, 10, &[]);
        p.attach_grammar("r", grammar());
        p.commit(&batch(&[("r", 2, 0, 2)], 1), None, &result(&[0], &[1]))
            .unwrap();
        let mut verify = result(&[1, 8, 5], &[3]);
        verify.grammar_terminated_at = Some(vec![1]);
        let c = p
            .commit(&batch(&[("r", 3, 0, 2)], 0), None, &verify)
            .unwrap();
        assert_eq!(c.outputs[0].output_ids, [1, 8]);
        assert_eq!(c.outputs[0].finish_reason, Some("stop"));
    }

    #[test]
    fn an_abort_outranks_the_grammar_ending_at_the_same_token() {
        // A NaN abort, and one a pause set before the commit.
        for pause in [false, true] {
            let mut p = processor(None);
            register(&mut p, "r", 2, 10, &[]);
            p.attach_grammar("r", grammar());
            p.commit(&batch(&[("r", 2, 0, 2)], 1), None, &result(&[0], &[1]))
                .unwrap();
            let mut step = result(&[1], &[1]);
            step.grammar_terminated_at = Some(vec![0]);
            if pause {
                p.mark_abort("r", true);
            } else {
                step.nan_flags = Some(vec![1]);
            }
            let c = p.commit(&batch(&[("r", 1, 0, 2)], 0), None, &step).unwrap();
            assert_eq!(c.outputs[0].finish_reason, Some("abort"), "pause: {pause}");
        }
    }

    #[test]
    fn decode_rows_step_over_whole_verify_windows() {
        // Two decode rows with a 3-token window: row 0 keeps 2, row 1 keeps 3.
        let mut p = processor(Some(3));
        for rid in ["a", "b"] {
            register(&mut p, rid, 1, 100, &[]);
            p.commit(&batch(&[(rid, 1, 0, 1)], 1), None, &result(&[0], &[1]))
                .unwrap();
        }
        let c = p
            .commit(
                &batch(&[("a", 3, 0, 1), ("b", 3, 0, 1)], 0),
                None,
                &result(&[10, 11, 0, 20, 21, 22], &[2, 3]),
            )
            .unwrap();
        assert_eq!(c.outputs[0].output_ids, [10, 11]);
        assert_eq!(c.outputs[1].output_ids, [20, 21, 22]);
        assert_eq!(p.state("a").unwrap().spec_verify_ct, 1);
    }

    #[test]
    fn a_verify_step_past_the_physical_extent_is_an_engine_error() {
        // A 4-token verify window over buffers holding 8 tokens: a request
        // at 4 committed tokens fits one more window exactly, not two.
        let mut p = OutputProcessor::new(Some(4), 1, Some(8), Role::Plain, NO_FINISH_LOG);
        register(&mut p, "r", 3, 100, &[]);
        p.commit(&batch(&[("r", 3, 0, 3)], 1), None, &result(&[1], &[1]))
            .unwrap();
        let window = batch(&[("r", 4, 0, 3)], 0);
        p.commit(&window, None, &result(&[2, 3, 4, 5], &[4]))
            .unwrap();
        assert!(matches!(
            p.commit(&window, None, &result(&[6, 7, 8, 9], &[4])),
            Err(CommitError::PhysicalExtent {
                total: 12,
                limit: 8,
                ..
            })
        ));
    }

    #[test]
    fn a_request_that_ended_in_flight_still_reports_its_tokens() {
        let mut p = processor(None);
        let c = p
            .commit(&batch(&[("gone", 1, 0, 1)], 0), None, &result(&[5], &[1]))
            .unwrap();
        assert_eq!(
            kinds(&c),
            [("gone".into(), ForwardEventKind::ExtendResult, vec![5])]
        );
        assert!(c.outputs.is_empty());
    }

    #[test]
    fn a_decode_row_without_a_token_is_an_engine_error() {
        let mut p = processor(None);
        register(&mut p, "r", 1, 10, &[]);
        p.commit(&batch(&[("r", 1, 0, 1)], 1), None, &result(&[1], &[1]))
            .unwrap();
        let error = p
            .commit(&batch(&[("r", 1, 0, 1)], 0), None, &result(&[0], &[0]))
            .unwrap_err();
        assert!(matches!(error, CommitError::EmptyDecode(_)));
    }

    #[test]
    fn sampled_logprobs_ride_with_their_tokens() {
        let mut p = processor(None);
        p.register(
            "r".to_owned(),
            RequestState::new(vec![0], 10, BTreeSet::new(), true, None, true),
        );
        let mut first = result(&[5], &[1]);
        first.output_logprobs = Some(vec![-0.25]);
        let c = p
            .commit(&batch(&[("r", 1, 0, 1)], 1), None, &first)
            .unwrap();
        assert_eq!(c.outputs[0].output_logprobs, [-0.25]);
    }
}
