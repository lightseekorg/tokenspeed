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

use std::sync::Arc;
use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant, SystemTime},
};

use ts_scheduler::{profile, ForwardBatchView, ForwardEvent, ForwardEventKind, SchedulerAbortView};

use ts_wire::output::FinishDetails;

use crate::{
    codec::{Reader, Writer},
    request::{AbortCode, FinishReason, InputLogprobs, RequestState},
    request_stats::{self, wall_seconds, RequestTimings},
    request_timing::EventStamp,
};

const NUMERICAL_ERROR: &str = "Request terminated: numerical corruption (NaN logits or \
     out-of-vocab sample) detected during generation.";

/// How long an abort for a request not registered yet is kept.
const PENDING_ABORT_TTL: Duration = Duration::from_secs(300);

/// A row's [`ForwardResult::request_ends`] entry once the device ended its
/// request, in that forward or an earlier one (`REQUEST_ENDED` in
/// `tokenspeed_kernel.ops.metadata.stop_conditions`); 0 while it runs.
pub const REQUEST_ENDED: i32 = 1;
/// The entry when the device ended the request but could not complete a
/// decode state it defers (`REQUEST_ENDED_INCOMPLETE`).
pub const REQUEST_ENDED_INCOMPLETE: i32 = 2;

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
    /// Selected tokens' cumulative probability mass in the same layout;
    /// `None` when the diagnostic is disabled.
    pub output_token_cumulative_probs: Option<Vec<f64>>,
    /// Per-row NaN-guard flags; `None` with the guard off.
    pub nan_flags: Option<Vec<i32>>,
    /// Per row, whether the device ended the request ([`REQUEST_ENDED`],
    /// [`REQUEST_ENDED_INCOMPLETE`]) or not (0): what a cache that defers
    /// decode state (`OutputProcessor::defers_decode_state`) publishes on.
    pub request_ends: Option<Vec<i32>>,
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
    /// parallelism. Nor are the cumulative probabilities: the engine refuses
    /// that diagnostic under pipeline parallelism.
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
        out.option(self.request_ends.as_deref(), Writer::i32s);
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
        let request_ends = reader.option(Reader::i32s)?;
        reader.finish(Self {
            output_tokens,
            output_lengths,
            output_logprobs,
            // Not shared: the engine refuses the diagnostic under a pipeline.
            output_token_cumulative_probs: None,
            nan_flags,
            request_ends,
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
    /// Each slot's committed prefix, or only its last tokens when the model
    /// reads a bounded lookback: they end at the prefix length.
    pub tokens: Vec<Vec<i32>>,
}

/// Which request's committed tokens each history row holds.
///
/// Every forward appends its inputs to its slot's row at the committed
/// frontier, so a row holds the request's readable committed tokens until
/// another request runs in the slot. With a bounded lookback, positions
/// before the seeded tail remain unspecified. A forward seeds its committed
/// prefix after a prefix-cache hit, a slot handoff, a retracted request's restore into
/// another slot, or a PD decode landing. Ownership is per admission, not
/// per request id, which clients may reuse. Only the request's latest slot
/// holds its current history; returning to an earlier slot must seed again.
///
/// A model that reads at most `lookback` committed tokens behind the
/// frontier (an n-gram hash, say) needs only those seeded: a 200K-token
/// prefix-cache hit then ships a few tokens to the device, not the prefix.
#[derive(Debug, Default)]
pub struct HistoryRows {
    owners: HashMap<i32, u64>,
    lookback: Option<usize>,
}

impl HistoryRows {
    /// Rows whose seeds carry the last `lookback` committed tokens, or the
    /// whole prefix with None.
    pub fn new(lookback: Option<usize>) -> Self {
        Self {
            owners: HashMap::new(),
            lookback,
        }
    }

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
            if !held {
                // An old row becomes stale as this request advances elsewhere.
                // Forget it on a move, including before a prefix-zero extend.
                self.owners
                    .retain(|other_slot, owner| *other_slot == slot || *owner != state.admission);
            }
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
            let start = self.lookback.map_or(0, |n| boundary.saturating_sub(n));
            let prompt = &state.prompt_ids;
            let prefix: Vec<i32> = if boundary <= prompt.len() {
                prompt[start..boundary].to_vec()
            } else {
                prompt
                    .iter()
                    .chain(&state.output_ids[..boundary - prompt.len()])
                    .skip(start)
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
    /// Reasoning tokens generated so far, excluding structural markers.
    pub reasoning_tokens: i64,
    /// Logprobs of `output_ids`; empty when not requested.
    pub output_logprobs: Vec<f64>,
    /// Cumulative probability mass of `output_ids`; empty without diagnostics
    /// or when the request did not ask for logprobs.
    pub output_token_cumulative_probs: Vec<f64>,
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
    /// Terminal request timing metadata; absent from intermediate outputs.
    pub request_timing: Option<ts_wire::RequestTiming>,
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
    /// A per-token diagnostic column does not match the sampled tokens.
    #[error("forward result has {tokens} tokens but {found} {column}")]
    TokenColumn {
        /// Tokens in the result.
        tokens: usize,
        /// The mismatched column.
        column: &'static str,
        /// Its length.
        found: usize,
    },
    /// A decode row kept no token although its request runs.
    #[error("decode request {0} kept no token")]
    EmptyDecode(String),
    /// The device ended a request but could not complete the decode state
    /// it defers: the state its finish would publish lacks accepted tokens.
    #[error("request {0} ended with a deferred decode state the device did not complete")]
    IncompleteDecodeState(String),
    /// A cache that defers decode state needs the device's request ends.
    #[error("forward result lacks {0}")]
    MissingColumn(&'static str),
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

fn finish(rid: &str, publishable: bool) -> ForwardEvent {
    // Abort rather than Finish keeps the request's pages out of the prefix
    // caches: numerically corrupted KV, or a deferred decode state the device
    // did not end the request with (`OutputProcessor::defers_decode_state`).
    let kind = if publishable {
        ForwardEventKind::Finish
    } else {
        ForwardEventKind::Abort
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
    /// Percentage of PD rooms to trace through their first output.
    pub ttft_percentage: u8,
}

/// Per-request latency samples and token totals since the last drain.
#[derive(Debug, Default)]
pub struct RequestMetrics {
    /// Detail for admission/no-attempt terminal samples.
    pub peer_local_reasons: Vec<String>,
    /// Expected and actually consumed remote tokens per finished hinted request.
    pub peer_usage: Vec<(Option<u64>, u64, &'static str)>,
    /// Unfinished lookup/admission intervals closed by request completion.
    pub peer_stops: Vec<(String, Option<Instant>, Option<Instant>, bool)>,
    /// Time from request creation to its first output token.
    pub ttft_seconds: Vec<f64>,
    /// Time from request creation to its end (role-specific end boundary).
    pub request_seconds: Vec<f64>,
    /// Decode phase: first token to request end (plain and decode roles).
    pub decode_seconds: Vec<f64>,
    /// Gap between consecutive nonempty output blocks, per token in the later block.
    pub tpot_seconds: Vec<f64>,
    /// First actual output to request end, per token after the first output block.
    pub e2e_tpot_seconds: Vec<f64>,
    /// Output tokens the requests produced, including the bootstrap token.
    pub completion_tokens: u64,
    /// One output-token count per ended request, including zero-token ends.
    pub completion_token_samples: Vec<u64>,
    /// Mean output tokens per verify round, including bonus, per ended request.
    pub spec_decode_accept_length_samples: Vec<f64>,
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
    /// A cache group defers its decode state to a request's end
    /// (`Scheduler::defers_decode_state`, KDA): the device completes it in
    /// the forward its stop criteria end, and says so in
    /// `ForwardResult::request_ends`. A finish the device did not end (a
    /// grammar's end under `ignore_eos`, a client abort) ends with an abort,
    /// which publishes nothing; a state the device could not complete stops
    /// the engine.
    defers_decode_state: bool,
    /// Actual DP and attention-TP identity; unknown until supplied by the host.
    timing_ranks: Option<(i64, i64)>,
    finish_log: FinishLog,
    /// When the last commit ran, for the request stats' decode steps.
    last_commit_at: f64,
    /// Identities allocated so far, before data-plane admission.
    admissions: u64,
    metrics_enabled: bool,
    /// Metric deltas since the event loop last drained them.
    peer_stops: Vec<(String, Option<Instant>, Option<Instant>, bool)>,
    peer_local_reasons: Vec<String>,
    peer_samples: Vec<(Option<u64>, u64, &'static str)>,
    generated_tokens: u64,
    prompt_tokens: u64,
    cache_hit_tokens: u64,
    ttft_seconds: Vec<f64>,
    request_seconds: Vec<f64>,
    decode_seconds: Vec<f64>,
    tpot_seconds: Vec<f64>,
    e2e_tpot_seconds: Vec<f64>,
    completion_tokens: u64,
    completion_token_samples: Vec<u64>,
    spec_decode_accept_length_samples: Vec<f64>,
    /// The OTLP trace exporter, when tracing is on. Presence also widens
    /// the request-stats clock: the trace needs the same timestamps.
    trace: Option<Arc<crate::trace::TracePublisher>>,
}

impl OutputProcessor {
    /// A processor for `spec_num_tokens`-wide verify (or none), the
    /// server's `stream_interval` and the engine's `role`, whose cache
    /// `defers_decode_state` or not, logging the finish lines `finish_log`
    /// selects.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec_num_tokens: Option<usize>,
        stream_interval: i64,
        physical_context_len: Option<i64>,
        role: Role,
        defers_decode_state: bool,
        finish_log: FinishLog,
        metrics_enabled: bool,
        trace: Option<Arc<crate::trace::TracePublisher>>,
    ) -> Self {
        Self {
            states: HashMap::new(),
            pending_aborts: VecDeque::new(),
            spec_num_tokens,
            stream_interval,
            physical_context_len,
            role,
            defers_decode_state,
            timing_ranks: None,
            finish_log,
            last_commit_at: 0.0,
            admissions: 0,
            metrics_enabled,
            trace,
            peer_stops: Vec::new(),
            peer_local_reasons: Vec::new(),
            peer_samples: Vec::new(),
            generated_tokens: 0,
            prompt_tokens: 0,
            cache_hit_tokens: 0,
            ttft_seconds: Vec::new(),
            request_seconds: Vec::new(),
            decode_seconds: Vec::new(),
            tpot_seconds: Vec::new(),
            e2e_tpot_seconds: Vec::new(),
            completion_tokens: 0,
            completion_token_samples: Vec::new(),
            spec_decode_accept_length_samples: Vec::new(),
        }
    }

    /// Attach the actual DP and attention-TP ranks to terminal request metadata.
    pub fn with_timing_ranks(mut self, dp_rank: i64, tp_rank: i64) -> Self {
        self.timing_ranks = Some((dp_rank, tp_rank));
        self
    }

    /// Per-request latency samples and token totals since the last drain.
    pub fn take_request_metrics(&mut self) -> RequestMetrics {
        RequestMetrics {
            peer_stops: std::mem::take(&mut self.peer_stops),
            peer_local_reasons: std::mem::take(&mut self.peer_local_reasons),
            peer_usage: std::mem::take(&mut self.peer_samples),
            ttft_seconds: std::mem::take(&mut self.ttft_seconds),
            request_seconds: std::mem::take(&mut self.request_seconds),
            decode_seconds: std::mem::take(&mut self.decode_seconds),
            tpot_seconds: std::mem::take(&mut self.tpot_seconds),
            e2e_tpot_seconds: std::mem::take(&mut self.e2e_tpot_seconds),
            completion_tokens: std::mem::take(&mut self.completion_tokens),
            completion_token_samples: std::mem::take(&mut self.completion_token_samples),
            spec_decode_accept_length_samples: std::mem::take(
                &mut self.spec_decode_accept_length_samples,
            ),
        }
    }

    /// Generated tokens committed since the last drain.
    pub fn take_generated_tokens(&mut self) -> u64 {
        std::mem::take(&mut self.generated_tokens)
    }

    /// Prompt and prefix-cache tokens of requests that finished since the
    /// last drain.
    pub fn take_request_tokens(&mut self) -> (u64, u64) {
        (
            std::mem::take(&mut self.prompt_tokens),
            std::mem::take(&mut self.cache_hit_tokens),
        )
    }

    /// Give `rid` its grammar's matcher.
    pub fn attach_grammar(&mut self, rid: &str, grammar: crate::request::RequestGrammar) {
        if let Some(state) = self.states.get_mut(rid) {
            state.grammar = Some(grammar);
        }
    }

    /// Stamp the first dispatch of `rids`, for the request stats.
    pub fn mark_scheduled(&mut self, rids: &[String]) {
        let stats = self.finish_log.request_stats || self.trace.is_some();
        if !stats && self.finish_log.ttft_percentage == 0 {
            return;
        }
        let now = stats.then(wall_seconds);
        for rid in rids {
            if let Some(state) = self.states.get_mut(rid) {
                if let (Some(timings), Some(now)) = (state.timings.as_mut(), now) {
                    timings.mark_scheduled(now);
                }
                if let Some(trace) = state.ttft.as_mut() {
                    if trace.scheduled_ms.is_none() {
                        trace.scheduled_ms = Some(trace.ms());
                    }
                }
            }
        }
    }

    /// Complete terminal response metadata and collect existing request stats.
    fn complete_request(
        &mut self,
        rid: &str,
        state: &RequestState,
        outputs: &mut [StreamedOutput],
    ) {
        self.complete_response(rid, state, outputs);
        if self.metrics_enabled {
            if let Some(usage) = &state.peer_usage {
                if usage.lookup_at.is_some() || usage.admission_at.is_some() {
                    self.peer_stops.push((
                        rid.to_owned(),
                        usage.lookup_at,
                        usage.admission_at,
                        state.to_abort,
                    ));
                }
                let actual = usage.actual.unwrap_or(0);
                let outcome = if actual > 0 {
                    "used"
                } else if state.to_abort {
                    "aborted"
                } else {
                    usage.disposition
                };
                if matches!(outcome, "admission" | "not_attempted") {
                    self.peer_local_reasons.push(
                        usage
                            .local_reason
                            .clone()
                            .unwrap_or_else(|| format!("{outcome}_unknown")),
                    );
                }
                self.peer_samples.push((usage.expected, actual, outcome));
            }
            // Cache and prompt totals feed the derived cache metrics.
            self.prompt_tokens += state.prompt_len.max(0) as u64;
            self.cache_hit_tokens += state.cached_tokens.max(0) as u64;
            let completion_tokens = state.output_ids.len() as u64;
            self.completion_tokens += completion_tokens;
            self.completion_token_samples.push(completion_tokens);
            if let Some(accept_length) = request_stats::accept_draft_tokens(state) {
                self.spec_decode_accept_length_samples.push(accept_length);
            }
            self.collect_request_latencies(state);
        }
    }

    /// Sample completion once and attach timing to the terminal response, when emitted.
    fn complete_response(
        &mut self,
        rid: &str,
        state: &RequestState,
        outputs: &mut [StreamedOutput],
    ) {
        self.collect_stream_end(state, wall_seconds());
        if let Some(trace) = &state.ttft {
            trace.finish(rid, state);
        }
        // Keep the existing backend completion boundary even with logging disabled.
        let finished = EventStamp::now();
        let finished_at = finished.wall_seconds();
        if let Some(output) = outputs
            .last_mut()
            .filter(|output| output.rid == rid && output.finish_reason.is_some())
        {
            let role = match self.role {
                Role::Plain => "plain",
                Role::Prefill => "prefill",
                Role::Decode => "decode",
            };
            output.request_timing = Some(state.request_timing.finish(
                role,
                self.timing_ranks,
                state.created_time,
                &finished,
            ));
        }
        self.log_finished(rid, state, finished_at);
    }

    /// Log the completion boundary already sampled by the response lifecycle.
    fn log_finished(&self, rid: &str, state: &RequestState, finished_at: f64) {
        if self.finish_log.enabled && tracing::enabled!(tracing::Level::INFO) {
            let stats = state
                .timings
                .as_ref()
                .filter(|_| self.finish_log.request_stats)
                .map(|timings| {
                    request_stats::stats_line(
                        rid,
                        state,
                        timings,
                        finished_at,
                        self.spec_num_tokens,
                    )
                });
            tracing::info!(
                rid,
                role = ?self.role,
                prompt_tokens = state.prompt_len,
                completion_tokens = state.output_ids.len(),
                finish_reason = state.finished_reason.as_ref().map(FinishReason::wire_type),
                abort_reason = state.finished_reason.as_ref().and_then(FinishReason::abort_message),
                cached_tokens = state.cached_tokens,
                e2e_latency_s = (state.created_time > 0.0)
                    .then(|| (finished_at - state.created_time).max(0.0)),
                request_stats = stats.as_deref(),
                "event=ts_request_finished"
            );
        }
    }

    /// Record one request's latency samples, with this role's boundaries.
    ///
    /// A prefill role's request ends when its KV transfer completes, so its
    /// request time includes the transfer; a decode role's request ends
    /// after generation, and its decode time starts at its first local
    /// decode output, after the bootstrap token landed. Both report the
    /// role-local views fluentllm's P/D servers do.
    fn collect_request_latencies(&mut self, state: &RequestState) {
        let created = state.created_time;
        if created <= 0.0 {
            return;
        }
        let finish = wall_seconds();
        self.request_seconds.push((finish - created).max(0.0));
        let first = state.first_token_time;
        if first <= 0.0 {
            return;
        }
        self.ttft_seconds.push((first - created).max(0.0));
        if self.role == Role::Prefill {
            return;
        }
        let decode = (finish - first).max(0.0);
        self.decode_seconds.push(decode);
    }

    /// Finish the stream metric even when a terminal path publishes no final block.
    /// These are engine output boundaries; TTFT and RequestTime keep their own clocks.
    fn collect_stream_end(&mut self, state: &RequestState, finished: f64) {
        if !self.metrics_enabled || self.role == Role::Prefill {
            return;
        }
        if let Some(seconds) = state
            .stream_output_timing
            .as_ref()
            .and_then(|timing| timing.e2e_tpot(finished))
        {
            self.e2e_tpot_seconds.push(seconds);
        }
    }

    /// Trace one released request: the span the request's own timestamps
    /// assemble, handed to the export thread when the sample keeps it.
    fn trace_end(&self, rid: &str, state: &RequestState) {
        let Some(trace) = &self.trace else {
            return;
        };
        if !trace.sampled(rid) {
            return;
        }
        let (scheduled, prefill_done, first_token, preempt_count, preempt_seconds) = state
            .timings
            .as_ref()
            .map_or((0.0, 0.0, 0.0, 0, 0.0), |timings| timings.trace_points());
        trace.emit(crate::trace::TraceRecord {
            rid: rid.to_owned(),
            role: self.role,
            session: state.session_id.clone(),
            created: state.created_time,
            scheduled,
            prefill_done,
            first_token,
            finish: crate::request_stats::wall_seconds(),
            prompt_tokens: state.prompt_len,
            cached_tokens: state.cached_tokens,
            output_tokens: state.output_ids.len() as i64,
            finish_reason: state
                .finished_reason
                .as_ref()
                .map_or("unknown", FinishReason::wire_type),
            preempt_count,
            preempt_seconds,
        });
    }

    /// The engine's role.
    pub fn role(&self) -> Role {
        self.role
    }

    /// The verify width with speculation.
    pub fn spec_num_tokens(&self) -> Option<usize> {
        self.spec_num_tokens
    }

    /// Whether this rank can collect sampled PD first-output summaries.
    pub(crate) fn ttft_sampling_enabled(&self) -> bool {
        self.finish_log.enabled
            && self.finish_log.ttft_percentage > 0
            && self.role != Role::Plain
            && tracing::enabled!(target: "ts_control::request_stats", tracing::Level::INFO)
    }

    /// When `rid` reached the scheduler's waiting queue, the first time this
    /// is asked after it did: a retraction re-queues
    /// a request without re-stamping it.
    pub fn take_submitted_at(&mut self, rid: &str) -> Option<Instant> {
        self.states.get_mut(rid)?.submitted_at.take()
    }

    /// Observe KV-ready waiting once, only for a request that starts decode.
    pub fn take_decode_ready_at(&mut self, rid: &str) -> Option<Instant> {
        let state = self.states.get_mut(rid)?;
        let at = state.decode_ready_at.take();
        if state.to_abort || state.finished() {
            return None;
        }
        at
    }

    /// Stamp `rid`'s arrival in the scheduler's waiting queue.
    pub fn mark_submitted(&mut self, rid: &str, at: Instant) {
        if let Some(state) = self.states.get_mut(rid) {
            state.submitted_at = Some(at);
            if let Some(trace) = state.ttft.as_mut() {
                let now = trace.ms();
                trace.submitted_ms.get_or_insert(now);
            }
        }
    }

    /// Stamp the first plan selecting a sampled request, before DP agreement.
    /// Remote admission reserves the D landing pages; it is not a forward.
    pub fn mark_planned(&mut self, rids: &[String], remote_admission: bool) {
        if self.finish_log.ttft_percentage == 0 {
            return;
        }
        for rid in rids {
            if let Some(trace) = self
                .states
                .get_mut(rid)
                .and_then(|state| state.ttft.as_mut())
            {
                let now = trace.ms();
                if remote_admission {
                    trace.remote_admitted_ms.get_or_insert(now);
                } else {
                    trace.planned_ms.get_or_insert(now);
                }
            }
        }
    }

    /// Requests admitted and not ended.
    pub fn num_running(&self) -> usize {
        self.states.len()
    }

    /// Usage attribution for a running peer-hinted request.
    pub(crate) fn peer_usage(&mut self, rid: &str) -> Option<&mut crate::request::PeerUsage> {
        self.states.get_mut(rid)?.peer_usage.as_mut()
    }

    /// Prompt plus committed output tokens of every request the rank holds.
    pub fn num_resident_tokens(&self) -> i64 {
        self.states
            .values()
            .map(|state| state.prompt_len + state.output_ids.len() as i64)
            .sum()
    }

    /// A running request's state.
    pub fn state(&self, rid: &str) -> Option<&RequestState> {
        self.states.get(rid)
    }

    pub(crate) fn sample_ttft(
        &self,
        state: &mut RequestState,
        bootstrap: Option<&ts_pd::BootstrapInfo>,
    ) {
        if self.ttft_sampling_enabled() {
            state.ttft = bootstrap.and_then(|b| {
                request_stats::TtftTrace::sample(self.finish_log.ttft_percentage, self.role, b)
            });
        }
    }

    pub(crate) fn bootstrapped(&mut self, rid: &str) {
        if self.finish_log.ttft_percentage > 0 {
            if let Some(trace) = self.states.get_mut(rid).and_then(|s| s.ttft.as_mut()) {
                trace.bootstrapped(rid);
            }
        }
    }

    /// Give a new request its identity before either plane registers it.
    pub fn assign_admission(&mut self, state: &mut RequestState) {
        assert_eq!(
            state.admission, 0,
            "request already has an admission identity"
        );
        self.admissions = self
            .admissions
            .checked_add(1)
            .expect("admission identity exhausted");
        state.admission = self.admissions;
    }

    /// Register a request whose identity was assigned before data-plane admission.
    /// An abort that arrived before it applies now.
    pub fn register(&mut self, rid: String, mut state: RequestState) {
        assert_ne!(state.admission, 0, "assign an identity before registration");
        state.request_timing.accepted = Some(EventStamp::now());
        if let Some(index) = self.pending_aborts.iter().position(|(id, _)| *id == rid) {
            self.pending_aborts.remove(index);
            state.abort("AbortReq from client", false, None);
        }
        if self.role == Role::Decode {
            // The prompt was computed on the prefill node.
            state.computed_length = state.prompt_len;
        }
        if self.finish_log.request_stats || self.trace.is_some() {
            state.timings = Some(RequestTimings::default());
        }
        if self.finish_log.enabled {
            tracing::info!(
                rid,
                role = ?self.role,
                session_id = state.session_id.as_deref(),
                input_tokens = state.prompt_len,
                max_new_tokens = state.max_new_tokens,
                "event=ts_request_accepted"
            );
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
        if let Some(trace) = &mut state.ttft {
            trace.transfer_done_ms = Some(trace.ms());
        }
        if !state.finished() {
            state.finished_reason = Some(FinishReason::Length(state.output_ids.len() as i64));
        }
        state.finished_output = false;
        if !state.to_abort || state.abort_notify_client {
            committed.outputs = Self::stream_state(
                rid,
                &mut state,
                self.stream_interval,
                self.spec_num_tokens,
                self.metrics_enabled,
                Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
            )
            .into_iter()
            .collect();
        }
        self.complete_request(rid, &state, &mut committed.outputs);
        self.trace_end(rid, &state);
        committed.released.push(rid.to_owned());
        committed
    }

    /// D role: the prefill node's KV landed. Its bootstrap token joins the
    /// output, which streams with the next local commit. The landing closes
    /// the queue/prefill boundary but does not start the decode window: the
    /// first local decode output is the role's first token, so a wait for a
    /// decode permit stays in TTFT rather than TPOT.
    pub fn remote_prefill_done(
        &mut self,
        rid: &str,
        bootstrap_token: i32,
        cached_tokens: i64,
        bootstrap_logprob: Option<f64>,
        kv_ready_at: (SystemTime, Instant),
    ) {
        let Some(state) = self.states.get_mut(rid) else {
            return;
        };
        if self.metrics_enabled {
            state.decode_ready_at.get_or_insert_with(Instant::now);
        }
        if let Some(trace) = &mut state.ttft {
            trace.transfer_done_ms = Some(trace.ms());
        }
        state.cached_tokens = cached_tokens;
        state.output_ids.push(bootstrap_token);
        state.request_timing.kv_ready(
            EventStamp::at(kv_ready_at.0, kv_ready_at.1),
            state.output_ids.len(),
        );
        if let Some(tokens) = &mut state.reasoning_tokens {
            tokens.push([bootstrap_token]);
        }
        if let Some(timings) = state.timings.as_mut() {
            // The remote bootstrap token is not this role's first output.
            // Its landing closes the prefill/queue boundary; the first
            // local decode commit stamps `first_token_time`, so the wait for
            // a decode permit is TTFT, not decode time.
            timings.mark_prefill_done(wall_seconds());
        }
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
            if !grammar
                .matcher
                .lock()
                .expect("matcher lock")
                .accept_token(bootstrap_token)
            {
                tracing::warn!(
                    rid,
                    token = bootstrap_token,
                    "grammar rejected the remote bootstrap token"
                );
            }
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
            committed.outputs = Self::stream_state(
                rid,
                &mut state,
                self.stream_interval,
                self.spec_num_tokens,
                self.metrics_enabled,
                Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
            )
            .into_iter()
            .collect();
        }
        let kind = if state.to_abort {
            ForwardEventKind::Abort
        } else {
            ForwardEventKind::Finish
        };
        committed.events.push(event(kind, rid, Vec::new(), 0));
        self.complete_request(rid, &state, &mut committed.outputs);
        self.trace_end(rid, &state);
        committed.released.push(rid.to_owned());
        committed
    }

    /// A request's PD transfer failed. A computed prefill still needs its
    /// final output; only a client abort or an already-sent final stays silent.
    /// The scheduler keeps its pages until the transfer and local forwards retire.
    pub fn fail_transfer(&mut self, rid: &str) -> Committed {
        let mut committed = Committed::default();
        let Some(mut state) = self.states.remove(rid) else {
            return committed;
        };
        if !state.to_abort && !state.finished_output {
            state.abort("PD/EPD remote transfer failed or timed out", true, None);
        }
        if state.abort_notify_client {
            committed.outputs = Self::stream_state(
                rid,
                &mut state,
                self.stream_interval,
                self.spec_num_tokens,
                self.metrics_enabled,
                Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
            )
            .into_iter()
            .collect();
        }
        self.complete_request(rid, &state, &mut committed.outputs);
        self.trace_end(rid, &state);
        committed.released.push(rid.to_owned());
        committed
    }

    /// Abort a request now, or when it registers if it has not yet.
    pub fn mark_abort(&mut self, rid: &str, notify_client: bool) {
        tracing::debug!(
            rid,
            notify_client,
            registered = self.states.contains_key(rid),
            "abort requested"
        );
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
        let mut outputs: Vec<_> = Self::stream_state(
            rid,
            &mut state,
            self.stream_interval,
            self.spec_num_tokens,
            self.metrics_enabled,
            Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
        )
        .into_iter()
        .collect();
        self.complete_response(rid, &state, &mut outputs);
        outputs
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
            committed.outputs = Self::stream_state(
                rid,
                &mut state,
                self.stream_interval,
                self.spec_num_tokens,
                self.metrics_enabled,
                Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
            )
            .into_iter()
            .collect();
        }
        self.complete_response(rid, &state, &mut committed.outputs);
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
            tracing::warn!(rid, reason, status_code, "request failed before scheduling");
            state.abort(reason, true, status_code);
        }
        if !state.to_abort || state.abort_notify_client {
            state.finished_output = false;
            committed.outputs = Self::stream_state(
                rid,
                &mut state,
                self.stream_interval,
                self.spec_num_tokens,
                self.metrics_enabled,
                Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
            )
            .into_iter()
            .collect();
        }
        self.complete_request(rid, &state, &mut committed.outputs);
        self.trace_end(rid, &state);
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
                committed.outputs.extend(Self::stream_state(
                    abort.request_id.as_str(),
                    &mut state,
                    self.stream_interval,
                    self.spec_num_tokens,
                    self.metrics_enabled,
                    Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
                ));
            } else if state.abort_notify_client {
                committed.outputs.extend(Self::stream_state(
                    abort.request_id.as_str(),
                    &mut state,
                    self.stream_interval,
                    self.spec_num_tokens,
                    self.metrics_enabled,
                    Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
                ));
            }
            self.complete_request(&abort.request_id, &state, &mut committed.outputs);
            self.trace_end(&abort.request_id, &state);
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
                if state.wants_input_logprobs() && !state.input_logprobs_finalized() && lo < hi {
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
        let _span = profile::span("OutputProcessor::commit");
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
        if let Some(ends) = &result.request_ends {
            check("request_ends", ends.len())?;
        } else if self.defers_decode_state {
            return Err(CommitError::MissingColumn("request_ends"));
        }
        if let Some(probs) = &result.output_token_cumulative_probs {
            if probs.len() != result.output_tokens.len() {
                return Err(CommitError::TokenColumn {
                    tokens: result.output_tokens.len(),
                    column: "output_token_cumulative_probs",
                    found: probs.len(),
                });
            }
        }

        // The request stats' clock: a decode step lasts from the previous
        // commit, and a round with prefill rows interrupts the decodes in it.
        let (stats_now, step, prefilling_others) =
            if self.finish_log.request_stats || self.trace.is_some() {
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
                if let Some(usage) = &mut state.peer_usage {
                    // Replayed positions execute again. Count only the remote interval
                    // below the first committed forward's actual starting position.
                    usage.actual.get_or_insert_with(|| {
                        usage.window.map_or(0, |(begin, end)| {
                            batch.extend_prefix_lens[i]
                                .min(end)
                                .saturating_sub(begin)
                                .max(0) as u64
                        })
                    });
                }
                state.cached_tokens += (hit - state.computed_length).max(0);
            }
        }

        // Most rows push one scheduler event and one streamed output.
        let mut committed = Committed {
            events: Vec::with_capacity(rows),
            ..Committed::default()
        };
        let mut streamed: Vec<&str> = Vec::with_capacity(rows);
        let mut pt = 0usize;
        for i in 0..rows {
            let rid = &batch.request_ids[i];
            let output_length = result.output_lengths[i].max(0) as usize;
            let mut model_ids = window(&result.output_tokens, pt, output_length);
            let mut model_logprobs = result
                .output_logprobs
                .as_ref()
                .map(|lps| window(lps, pt, output_length));
            let mut model_cumulative_probs = result
                .output_token_cumulative_probs
                .as_ref()
                .map(|probs| window(probs, pt, output_length));
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
            let request_end = result.request_ends.as_ref().map_or(0, |ends| ends[i]);
            if request_end == REQUEST_ENDED_INCOMPLETE {
                return Err(CommitError::IncompleteDecodeState(rid.to_owned()));
            }

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
                if let Some(trace) = &mut state.ttft {
                    trace.prefill_chunks += 1;
                }
                tracing::debug!(
                    rid,
                    prefix_tokens = batch.extend_prefix_lens[i],
                    cached_tokens = state.cached_tokens,
                    chunk_tokens = batch.input_lengths[i],
                    prompt_tokens = batch.prefill_lengths[i],
                    last_chunk = !mid_chunk,
                    "prefill chunk committed"
                );
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
            if !is_decode {
                if let Some(trace) = &mut state.ttft {
                    if trace.prompt_done_ms.is_none() {
                        trace.prompt_done_ms = Some(trace.ms());
                    }
                }
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
                if let Some(probs) = model_cumulative_probs.as_mut() {
                    probs.truncate(1);
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
            let before = state.output_ids.len();
            for (j, &token) in model_ids.iter().enumerate() {
                state.output_ids.push(token);
                if let Some(tokens) = &mut state.reasoning_tokens {
                    tokens.push([token]);
                }
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
                    if let Some(prob) = model_cumulative_probs
                        .as_ref()
                        .and_then(|probs| probs.get(j))
                    {
                        // The column stays indexed like `output_ids`: an
                        // earlier token that came without a value (a remote
                        // prefill's bootstrap token) keeps its place as NaN.
                        let before = state.output_ids.len() - 1;
                        state.output_token_cumulative_probs.resize(
                            before.max(state.output_token_cumulative_probs.len()),
                            f64::NAN,
                        );
                        state.output_token_cumulative_probs.push(*prob);
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
            if self.role == Role::Plain && !is_decode && !state.output_ids.is_empty() {
                state
                    .request_timing
                    .kv_ready(EventStamp::now(), state.output_ids.len());
            }
            if is_decode && self.metrics_enabled {
                self.generated_tokens += (state.output_ids.len() - before) as u64;
            }
            // The role's first output token. On a decode role this is the
            // first local decode commit, after any wait for a decode permit;
            // a remote prefill's bootstrap token alone does not stamp it.
            if (self.metrics_enabled || state.timings.is_some())
                && state.first_token_time == 0.0
                && !state.output_ids.is_empty()
            {
                let first = if stats_now > 0.0 {
                    stats_now
                } else {
                    wall_seconds()
                };
                state.first_token_time = first;
                if let Some(timings) = state.timings.as_mut() {
                    timings.mark_first_token(first);
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

            if self.role == Role::Prefill && !nan_detected {
                // A numerically corrupted prompt's KV must never transfer:
                // it falls through to emit its terminal abort (unless the
                // client already cancelled) and send Abort to stop the remote
                // decode. Otherwise the transfer owns the pages until it
                // succeeds or fails, and the output goes out then.
                continue;
            }
            let publishable =
                !nan_detected && (!self.defers_decode_state || request_end == REQUEST_ENDED);
            if state.to_abort && state.finished() {
                committed.events.push(finish(rid, publishable));
                if state.abort_notify_client {
                    streamed.push(rid);
                } else {
                    self.release(rid, &mut committed);
                }
                continue;
            }
            streamed.push(rid);
            if state.finished() {
                committed.events.push(finish(rid, publishable));
            } else if is_decode && state.decode_reserve != Some(output_length as i32) {
                // The scheduler keeps a request's decode reservation until it
                // is sent another (a retraction's image carries it too), so
                // only a change goes out, not an event per row every round.
                state.decode_reserve = Some(output_length as i32);
                committed.events.push(event(
                    ForwardEventKind::UpdateReserveNumTokens,
                    rid,
                    Vec::new(),
                    output_length as i32,
                ));
            }
        }

        committed.outputs = self.stream(&streamed, &mut committed.released);
        Ok(committed)
    }

    fn release(&mut self, rid: &str, committed: &mut Committed) {
        if let Some(state) = self.states.remove(rid) {
            self.complete_request(rid, &state, &mut committed.outputs);
            self.trace_end(rid, &state);
            committed.released.push(rid.to_owned());
        }
    }

    /// Pair the sample destination with its output clock, only when collected.
    fn tpot_observation(
        metrics_enabled: bool,
        role: Role,
        samples: &mut Vec<f64>,
    ) -> Option<(&mut Vec<f64>, f64)> {
        (metrics_enabled && role != Role::Prefill).then(|| (samples, wall_seconds()))
    }

    /// The output due from one request this step, at its stream interval,
    /// if any is. A free function so the commit streams states in place
    /// without lifting them out of the map.
    fn stream_state(
        rid: &str,
        state: &mut RequestState,
        stream_interval: i64,
        spec_num_tokens: Option<usize>,
        metrics_enabled: bool,
        tpot_observation: Option<(&mut Vec<f64>, f64)>,
    ) -> Option<StreamedOutput> {
        if !state.output_due(stream_interval) {
            return None;
        }
        // The normal path stamps `first_token_time` at the first local decode
        // commit. A decode request that finishes at landing has no such
        // commit: its bootstrap token is the only output. This fallback
        // covers that case (and any terminal stream before a commit) without
        // making a non-streaming request wait for its forced flush.
        if (metrics_enabled || state.timings.is_some())
            && state.first_token_time == 0.0
            && !state.output_ids.is_empty()
        {
            let first = wall_seconds();
            state.first_token_time = first;
            if let Some(timings) = state.timings.as_mut() {
                timings.mark_first_token(first);
            }
        }
        if state.finished() {
            state.finished_output = true;
        }
        if let Some(trace) = state.ttft.take() {
            trace.finish(rid, state);
        }
        tracing::debug!(
            rid,
            prompt_tokens = state.prompt_len,
            cached_tokens = state.cached_tokens,
            output_tokens = state.output_ids.len(),
            finished = state.finished(),
            "stream output"
        );
        let sent = state.send_token_offset;
        let output_ids = state.output_ids[sent..].to_vec();
        state.send_token_offset = state.output_ids.len();
        let n_new = output_ids.len();
        if n_new > 0 {
            if let Some((samples, now)) = tpot_observation {
                if let Some(timing) = &mut state.stream_output_timing {
                    samples.push((now - timing.last_output).max(0.0) / n_new as f64);
                    timing.last_output = now;
                    timing.output_tokens += n_new;
                } else {
                    state.stream_output_timing = Some(crate::request::StreamOutputTiming {
                        first_output: now,
                        last_output: now,
                        first_chunk_tokens: n_new,
                        output_tokens: n_new,
                    });
                }
            }
        }
        let output_logprobs = if state.return_logprob {
            let start = state.output_logprobs.len().saturating_sub(n_new);
            state.output_logprobs[start..].to_vec()
        } else {
            Vec::new()
        };
        // Indexed like `output_ids` (see the commit), so the new tokens'
        // values start where theirs do. Once the request has any value,
        // every output carries one per token, NaN where none arrived, so
        // the column never misaligns with the tokens.
        let output_token_cumulative_probs =
            if state.return_logprob && !state.output_token_cumulative_probs.is_empty() {
                let mut values = state
                    .output_token_cumulative_probs
                    .get(sent..)
                    .map_or_else(Vec::new, <[f64]>::to_vec);
                values.resize(n_new, f64::NAN);
                values
            } else {
                Vec::new()
            };
        // Prompt logprobs ship once, in the first output after the
        // prompt's final chunk committed.
        let input_logprobs = if state.input_logprobs_sent {
            None
        } else {
            state.input_logprobs.take()
        };
        state.input_logprobs_sent |= input_logprobs.is_some();
        let output_top_logprobs = state.output_top_logprobs.as_ref().map(|(values, ids)| {
            (
                values[values.len().saturating_sub(n_new)..].to_vec(),
                ids[ids.len().saturating_sub(n_new)..].to_vec(),
            )
        });
        Some(StreamedOutput {
            request_timing: None,
            rid: rid.to_owned(),
            output_ids,
            finish_reason: state.finished_reason.as_ref().map(FinishReason::wire_type),
            prompt_tokens: state.prompt_len,
            completion_tokens: state.output_ids.len() as i64,
            cached_tokens: state.cached_tokens,
            reasoning_tokens: state.reasoning_tokens.map_or(0, |tokens| tokens.count()),
            output_logprobs,
            output_token_cumulative_probs,
            finish_details: state.finished_reason.as_ref().map(FinishReason::details),
            spec: spec_num_tokens.map(|_| (state.spec_verify_ct, state.accept_draft_tokens)),
            input_logprobs,
            output_top_logprobs,
        })
    }

    /// The outputs due from `rids` this step, each request's at its stream
    /// interval; a finished request leaves the processor, into `released`,
    /// whether or not an output was due.
    fn stream(&mut self, rids: &[&str], released: &mut Vec<String>) -> Vec<StreamedOutput> {
        let interval = self.stream_interval;
        let spec = self.spec_num_tokens;
        let metrics_enabled = self.metrics_enabled;
        let mut outputs = Vec::with_capacity(rids.len());
        for &rid in rids {
            let Some(state) = self.states.get_mut(rid) else {
                continue;
            };
            if let Some(output) = Self::stream_state(
                rid,
                state,
                interval,
                spec,
                metrics_enabled,
                Self::tpot_observation(self.metrics_enabled, self.role, &mut self.tpot_seconds),
            ) {
                outputs.push(output);
            }
            if state.finished() {
                let (_, state) = self.states.remove_entry(rid).expect("present above");
                self.complete_request(rid, &state, &mut outputs);
                self.trace_end(rid, &state);
                released.push(rid.to_owned());
            }
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn request_timing_returns_actual_prefill_output_count_once_without_logs() {
        for stream in [true, false] {
            for replay_prefill in [true, false] {
                let mut p = processor(None).with_timing_ranks(2, 0);
                let mut state =
                    RequestState::new(vec![1, 2, 3], 4, BTreeSet::new(), stream, None, false);
                state.created_time = wall_seconds() - 5.0;
                let created = state.created_time;
                register_state(&mut p, "timing".to_owned(), state);
                let first = p
                    .commit(
                        &batch(&[("timing", 3, 0, 3)], 1),
                        None,
                        &result(&[4, 5, 6], &[3]),
                    )
                    .unwrap();
                assert!(first
                    .outputs
                    .iter()
                    .all(|output| output.request_timing.is_none()));
                let first_boundary = p
                    .state("timing")
                    .unwrap()
                    .request_timing
                    .finish("plain", Some((2, 0)), created, &EventStamp::now())
                    .events
                    .kv_ready
                    .unwrap();
                // A resumed prefill must preserve the original readiness boundary.
                let next = if replay_prefill {
                    batch(&[("timing", 1, 3, 4)], 1)
                } else {
                    batch(&[("timing", 1, 0, 3)], 0)
                };
                let final_output = p.commit(&next, None, &result(&[7], &[1])).unwrap();
                let output = &final_output.outputs[0];
                let timing = output.request_timing.as_ref().unwrap();
                assert_eq!(timing.kv_output_tokens, Some(3));
                assert_eq!(output.completion_tokens, 4);
                assert_eq!(output.rid, "timing");
                assert_eq!(output.finish_reason, Some("length"));
                assert_eq!(timing.role, "plain");
                assert_eq!((timing.dp_rank, timing.tp_rank), (Some(2), Some(0)));
                let accepted = timing.events.accepted.as_ref().unwrap();
                let kv_ready = timing.events.kv_ready.as_ref().unwrap();
                let finished = &timing.events.finished;
                assert_eq!(kv_ready, &first_boundary);
                let mono_ns = |stamp: &ts_wire::RequestTimestamp| {
                    stamp.ts_mono_ns.as_ref().unwrap().parse::<u128>().unwrap()
                };
                assert!(mono_ns(accepted) <= mono_ns(kv_ready));
                assert!(mono_ns(kv_ready) <= mono_ns(finished));
                let finished_s = finished.ts_unix_ns.parse::<u128>().unwrap() as f64 / 1e9;
                assert!((timing.e2e_latency_s.unwrap() - (finished_s - created)).abs() < 1e-6);
                assert_eq!(timing.clock.clock_sync, "unknown");
                assert_eq!(timing.clock.clock_uncertainty_ns, None);
                assert!(!timing.clock.process_instance_id.is_empty());
                assert_eq!(p.num_running(), 0);
            }
        }
    }

    #[test]
    fn request_timing_preserves_pd_landing_time_for_one_token_request() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            false,
            None,
        );
        register(&mut p, "one", 3, 1, &[]);
        let observed = (SystemTime::now(), Instant::now());
        p.remote_prefill_done("one", 9, 0, None, observed);
        let c = p.finish_remote_prefill_only("one");
        let timing = c.outputs[0].request_timing.as_ref().unwrap();
        assert_eq!(timing.kv_output_tokens, Some(1));
        assert_eq!(c.outputs[0].completion_tokens, 1);
        assert_eq!(timing.role, "decode");
        assert_eq!((timing.dp_rank, timing.tp_rank), (None, None));
        assert_eq!(
            timing.events.kv_ready.as_ref().unwrap().ts_unix_ns,
            observed
                .0
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                .to_string(),
        );
        assert!(p.finish_remote_prefill_only("one").outputs.is_empty());
    }

    #[test]
    fn request_timing_transfer_failure_keeps_missing_kv_boundary() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "failed", 3, 8, &[]);
        let c = p.fail_transfer("failed");
        let timing = c.outputs[0].request_timing.as_ref().unwrap();
        assert!(timing.events.accepted.is_some());
        assert_eq!(timing.kv_output_tokens, None);
        assert_eq!(timing.events.kv_ready, None);
        assert_eq!(c.outputs[0].finish_reason, Some("abort"));
        assert_eq!(timing.e2e_latency_s, None);
        let metrics = p.take_request_metrics();
        assert_eq!(metrics.completion_token_samples, vec![0]);
        assert_eq!(p.take_request_tokens(), (3, 0));
        assert!(p.fail_transfer("failed").outputs.is_empty());
        assert!(p.take_request_metrics().completion_token_samples.is_empty());
    }

    #[test]
    fn request_timing_rejection_has_no_registration_or_readiness() {
        let mut p = processor(None);
        let mut state = RequestState::new(vec![1], 1, BTreeSet::new(), true, None, false);
        state.abort("invalid request", true, None);
        let outputs = p.publish_finished_at_admission("invalid", state);
        let timing = outputs[0].request_timing.as_ref().unwrap();
        assert_eq!(timing.events.accepted, None);
        assert_eq!(timing.events.kv_ready, None);
        assert_eq!(timing.kv_output_tokens, None);
        assert_eq!(outputs[0].finish_reason, Some("abort"));
    }

    #[test]
    fn request_timing_prefill_reports_its_local_completion() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Prefill,
            false,
            NO_FINISH_LOG,
            false,
            None,
        );
        register(&mut p, "prefill", 3, 8, &[]);
        let c = p
            .commit(
                &batch(&[("prefill", 3, 0, 3)], 1),
                None,
                &result(&[7], &[1]),
            )
            .unwrap();
        assert!(c.outputs.is_empty());
        let c = p.finish_prefill("prefill");
        let timing = c.outputs[0].request_timing.as_ref().unwrap();
        assert_eq!(timing.role, "prefill");
        assert!(timing.events.accepted.is_some());
        assert_eq!(timing.events.kv_ready, None);
        assert_eq!(timing.kv_output_tokens, None);
        assert_eq!(c.outputs[0].completion_tokens, 1);
        assert!(p.finish_prefill("prefill").outputs.is_empty());
    }

    #[test]
    fn request_timing_attaches_to_each_terminal_row_and_cancellation_stays_silent() {
        let mut p = processor(None);
        for rid in ["a", "b", "cancelled"] {
            register(&mut p, rid, 3, 1, &[]);
        }
        p.mark_abort("cancelled", false);
        let cancelled = p.reap_unscheduled("cancelled");
        assert!(cancelled.outputs.is_empty());
        let c = p
            .commit(
                &batch(&[("a", 3, 0, 3), ("b", 3, 0, 3)], 2),
                None,
                &result(&[7, 8], &[1, 1]),
            )
            .unwrap();
        assert_eq!(c.outputs.len(), 2);
        for output in &c.outputs {
            assert!(output.request_timing.is_some());
            assert_eq!(output.completion_tokens, 1);
        }
        assert_eq!(p.num_running(), 0);
    }

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
            output_token_cumulative_probs: None,
            nan_flags: None,
            // The device ends no request unless a test says so.
            request_ends: Some(vec![0; lengths.len()]),
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
        ttft_percentage: 0,
    };

    fn processor(spec_num_tokens: Option<usize>) -> OutputProcessor {
        OutputProcessor::new(
            spec_num_tokens,
            1,
            Some(4096),
            Role::Plain,
            false,
            NO_FINISH_LOG,
            true,
            None,
        )
    }

    fn register_state(p: &mut OutputProcessor, rid: String, mut state: RequestState) {
        p.assign_admission(&mut state);
        p.register(rid, state);
    }

    fn register(p: &mut OutputProcessor, rid: &str, prompt: i64, max_new: i64, stop: &[i32]) {
        let stops: BTreeSet<i32> = stop.iter().copied().collect();
        register_state(
            p,
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
        register_state(p, rid.to_owned(), state);
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
        register_state(p, rid.to_owned(), state);
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
    fn commits_accumulate_reasoning_usage() {
        let grammar = crate::grammar::ReasoningGrammar {
            continues: String::new(),
            may_open: String::new(),
            schema_slot: String::new(),
            markers: Some((1, 2)),
            implicit_closer: None,
            boundary: None,
        };
        let mut state = RequestState::new(vec![1], 10, BTreeSet::new(), true, None, false);
        state.reasoning_tokens = Some(crate::request::ReasoningTokens::new(
            &grammar,
            &state.prompt_ids,
        ));
        let mut p = processor(None);
        register_state(&mut p, "r".to_owned(), state);

        for (i, (token, expected)) in [(10, 1), (11, 2), (2, 2), (12, 2)].into_iter().enumerate() {
            let input_length = i as i32 + 1;
            let committed = p
                .commit(
                    &batch(&[("r", input_length, 0, 1)], usize::from(i == 0)),
                    None,
                    &result(&[token], &[1]),
                )
                .unwrap();
            assert_eq!(committed.outputs[0].reasoning_tokens, expected);
        }
    }

    #[test]
    fn ttft_sampling_depends_on_pd_summary_settings_and_the_log_filter() {
        use tracing::Level;

        for (level, role, enabled, percentage, request_stats, expected) in [
            (Level::INFO, Role::Prefill, true, 100, false, true),
            (Level::INFO, Role::Decode, true, 100, false, true),
            (Level::INFO, Role::Decode, true, 1, false, true),
            (Level::INFO, Role::Prefill, true, 0, false, false),
            (Level::INFO, Role::Decode, true, 0, true, false),
            (Level::DEBUG, Role::Decode, true, 0, false, false),
            (Level::WARN, Role::Decode, true, 100, true, false),
            (Level::INFO, Role::Decode, false, 100, true, false),
            (Level::INFO, Role::Plain, true, 100, true, false),
            (Level::DEBUG, Role::Decode, true, 100, false, true),
        ] {
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(level)
                .with_writer(std::io::sink)
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                let processor = OutputProcessor::new(
                    None,
                    1,
                    None,
                    role,
                    false,
                    FinishLog {
                        enabled,
                        request_stats,
                        ttft_percentage: percentage,
                    },
                    false,
                    None,
                );
                let mut state =
                    RequestState::new(vec![8, 9], 4, BTreeSet::new(), true, None, false);
                // Room 4 is selected even at one percent sampling.
                processor.sample_ttft(
                    &mut state,
                    Some(&ts_pd::BootstrapInfo {
                        room: 4,
                        host: "p".into(),
                        port: 8080,
                    }),
                );
                assert_eq!(state.ttft.is_some(), expected);
                assert_eq!(processor.ttft_sampling_enabled(), expected);
            });
        }
    }

    #[test]
    fn sampled_rooms_emit_bootstrap_and_first_output_once() {
        use std::{io::Read, os::unix::net::UnixStream};
        let (writer, mut reader) = UnixStream::pair().unwrap();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.try_clone().unwrap())
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            for role in [Role::Prefill, Role::Decode] {
                let mut processor = OutputProcessor::new(
                    None,
                    1,
                    None,
                    role,
                    false,
                    FinishLog {
                        enabled: true,
                        request_stats: false,
                        ttft_percentage: 1,
                    },
                    false,
                    None,
                );
                // Python zlib.crc32 of the decimal room, modulo 1000:
                // room 4 lands in bucket 8, room 5 is outside the sample.
                for (room, rid) in [(4, "kept"), (5, "skipped")] {
                    let mut state =
                        RequestState::new(vec![8, 9], 1, BTreeSet::new(), true, None, false);
                    let bootstrap = ts_pd::BootstrapInfo {
                        room,
                        host: "p".into(),
                        port: 8080,
                    };
                    processor.sample_ttft(&mut state, Some(&bootstrap));
                    register_state(&mut processor, rid.into(), state);
                    processor.mark_submitted(rid, Instant::now());
                    processor.bootstrapped(rid);
                    processor.bootstrapped(rid);
                    if role == Role::Prefill {
                        processor.mark_planned(&[rid.into()], false);
                        processor.mark_scheduled(&[rid.into()]);
                        let committed = processor
                            .commit(&batch(&[(rid, 2, 0, 2)], 1), None, &result(&[10], &[1]))
                            .unwrap();
                        assert!(committed.outputs.is_empty());
                        if rid == "kept" {
                            let trace = processor.state(rid).unwrap().ttft.as_ref().unwrap();
                            assert!(trace.transfer_done_ms.is_none());
                        }
                        processor.finish_prefill(rid);
                    } else {
                        processor.mark_planned(&[rid.into()], true);
                        processor.remote_prefill_done(
                            rid,
                            10,
                            1,
                            None,
                            (SystemTime::now(), Instant::now()),
                        );
                        processor.finish_remote_prefill_only(rid);
                    }
                }
            }
        });
        let mut log = String::new();
        reader.read_to_string(&mut log).unwrap();
        let records = |marker| {
            log.lines()
                .filter_map(|line| line.split_once(marker))
                .map(|(_, data)| serde_json::from_str::<serde_json::Value>(data).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(records("TTFTBootstrap ").len(), 1, "{log}");
        let summaries = records("TTFTBreakdown ");
        assert_eq!(summaries.len(), 2, "{log}");
        for summary in &summaries {
            assert_eq!(summary["rid"], "kept");
            assert_eq!(summary["room"], 4);
            assert_eq!(summary["status"], "first_output");
            assert!(summary["phases_ms"]["bootstrap"].is_number());
            assert!(summary["milestones_ms"]["submitted"].is_number());
        }
        assert_eq!(summaries[0]["prefill_chunks"], 1);
        assert!(summaries[0]["phases_ms"]["prefill"].is_number());
        assert!(summaries[0]["phases_ms"]["ready_to_plan"].is_number());
        assert!(summaries[0]["phases_ms"]["plan_to_dispatch"].is_number());
        assert!(summaries[1]["phases_ms"]["remote_admission"].is_number());
        assert!(summaries[1]["milestones_ms"]["first_planned"].is_null());
        assert!(summaries[0]["phases_ms"]["transfer_observe"].is_number());
        assert!(summaries[0]["phases_ms"]["output_pack"].is_number());
        assert!(summaries[1]["phases_ms"]["output_after_landing"].is_number());
    }

    #[test]
    fn admission_failure_emits_one_ttft_breakdown() {
        use std::{io::Read, os::unix::net::UnixStream};
        for role in [Role::Prefill, Role::Decode] {
            let (writer, mut reader) = UnixStream::pair().unwrap();
            let subscriber = tracing_subscriber::fmt()
                .without_time()
                .with_ansi(false)
                .with_writer(move || writer.try_clone().unwrap())
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                let mut processor = OutputProcessor::new(
                    None,
                    1,
                    None,
                    role,
                    false,
                    FinishLog {
                        enabled: true,
                        request_stats: false,
                        ttft_percentage: 100,
                    },
                    true,
                    None,
                );
                let mut state =
                    RequestState::new(vec![8, 9], 4, BTreeSet::new(), true, None, false);
                processor.sample_ttft(
                    &mut state,
                    Some(&ts_pd::BootstrapInfo {
                        room: 4,
                        host: "p".into(),
                        port: 8080,
                    }),
                );
                state.abort("invalid request", true, Some(400));
                let outputs = processor.publish_finished_at_admission("rejected", state);
                assert_eq!(outputs.len(), 1);
                assert!(processor.rids().is_empty());
                let metrics = processor.take_request_metrics();
                assert!(metrics.tpot_seconds.is_empty());
                assert!(metrics.e2e_tpot_seconds.is_empty());
            });
            let mut log = String::new();
            reader.read_to_string(&mut log).unwrap();
            assert_eq!(log.matches("TTFTBreakdown ").count(), 1, "{log}");
            assert_eq!(log.matches("event=ts_request_finished").count(), 1, "{log}");
            assert!(log.contains("\"status\":\"abort\""), "{log}");
        }
    }

    #[test]
    fn sampled_milestones_keep_the_first_submission_and_each_plan_kind() {
        let subscriber = tracing_subscriber::fmt()
            .with_writer(std::io::sink)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            let mut processor = OutputProcessor::new(
                None,
                1,
                None,
                Role::Decode,
                false,
                FinishLog {
                    enabled: true,
                    request_stats: false,
                    ttft_percentage: 100,
                },
                false,
                None,
            );
            let mut state = RequestState::new(vec![8, 9], 4, BTreeSet::new(), true, None, false);
            processor.sample_ttft(
                &mut state,
                Some(&ts_pd::BootstrapInfo {
                    room: 4,
                    host: "p".into(),
                    port: 8080,
                }),
            );
            register_state(&mut processor, "r".into(), state);
            processor.mark_submitted("r", Instant::now());
            processor.bootstrapped("r");
            processor.mark_planned(&["r".into()], true);
            let trace = processor.state("r").unwrap().ttft.as_ref().unwrap();
            let first_submit = trace.submitted_ms.unwrap();
            let remote_admitted = trace.remote_admitted_ms.unwrap();
            assert!(trace.planned_ms.is_none());

            processor.remote_prefill_done("r", 10, 0, None, (SystemTime::now(), Instant::now()));
            processor.mark_planned(&["r".into()], false);
            processor.mark_scheduled(&["r".into()]);
            let trace = processor.state("r").unwrap().ttft.as_ref().unwrap();
            let first_plan = trace.planned_ms.unwrap();
            let first_dispatch = trace.scheduled_ms.unwrap();
            assert!(remote_admitted <= trace.transfer_done_ms.unwrap());
            assert!(trace.transfer_done_ms.unwrap() <= first_plan);
            assert!(first_plan <= first_dispatch);

            processor.mark_submitted("r", Instant::now());
            processor.mark_planned(&["r".into()], true);
            processor.mark_planned(&["r".into()], false);
            processor.mark_scheduled(&["r".into()]);
            let trace = processor.state("r").unwrap().ttft.as_ref().unwrap();
            assert_eq!(trace.submitted_ms, Some(first_submit));
            assert_eq!(trace.remote_admitted_ms, Some(remote_admitted));
            assert_eq!(trace.planned_ms, Some(first_plan));
            assert_eq!(trace.scheduled_ms, Some(first_dispatch));
        });
    }

    #[test]
    fn request_latency_samples_use_the_prefill_boundaries() {
        let mut p = OutputProcessor::new(
            None,
            1,
            Some(4096),
            Role::Prefill,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "a", 4, 8, &[]);
        let mut state = p.states.remove("a").expect("registered above");
        state.created_time = 100.0;
        state.first_token_time = 100.25;
        state.output_ids = vec![7];
        p.complete_request("a", &state, &mut []);
        let metrics = p.take_request_metrics();
        assert_eq!(metrics.ttft_seconds, vec![0.25]);
        assert_eq!(metrics.request_seconds.len(), 1);
        // A prefill role has no decode phase of its own.
        assert!(metrics.decode_seconds.is_empty());
        assert!(metrics.tpot_seconds.is_empty());
        assert_eq!(metrics.completion_tokens, 1);
        assert_eq!(p.take_request_metrics().completion_tokens, 0);
    }

    fn stream_at(p: &mut OutputProcessor, rid: &str, now: f64) -> Option<StreamedOutput> {
        OutputProcessor::stream_state(
            rid,
            p.states.get_mut(rid).unwrap(),
            p.stream_interval,
            p.spec_num_tokens,
            p.metrics_enabled,
            (p.metrics_enabled && p.role != Role::Prefill).then_some((&mut p.tpot_seconds, now)),
        )
    }

    #[test]
    fn stream_tpot_uses_emitted_chunks_and_ignores_empty_outputs() {
        let mut p = processor(Some(4));
        p.stream_interval = 3;
        register(&mut p, "r", 4, 20, &[]);
        p.states.get_mut("r").unwrap().output_ids.extend([10, 11]);
        assert_eq!(stream_at(&mut p, "r", 100.0).unwrap().output_ids, [10, 11]);
        p.states.get_mut("r").unwrap().output_ids.push(12);
        assert!(stream_at(&mut p, "r", 101.0).is_none());
        p.states.get_mut("r").unwrap().output_ids.extend([13, 14]);
        assert_eq!(
            stream_at(&mut p, "r", 106.0).unwrap().output_ids,
            [12, 13, 14]
        );
        p.stream_interval = 1;
        assert!(stream_at(&mut p, "r", 110.0).unwrap().output_ids.is_empty());
        p.states.get_mut("r").unwrap().output_ids.extend([15, 16]);
        assert_eq!(stream_at(&mut p, "r", 112.0).unwrap().output_ids, [15, 16]);
        p.states.get_mut("r").unwrap().finished_reason = Some(FinishReason::Length(7));
        assert!(stream_at(&mut p, "r", 118.0).unwrap().output_ids.is_empty());
        let state = p.states.remove("r").unwrap();
        p.collect_stream_end(&state, 118.0);
        let samples = p.take_request_metrics();
        assert_eq!(samples.tpot_seconds, [2.0, 3.0]);
        assert_eq!(samples.e2e_tpot_seconds, [18.0 / 5.0]);
        let drained = p.take_request_metrics();
        assert!(drained.tpot_seconds.is_empty());
        assert!(drained.e2e_tpot_seconds.is_empty());
    }

    #[test]
    fn disabled_and_prefill_outputs_do_not_keep_stream_metric_state() {
        for (role, enabled) in [
            (Role::Plain, false),
            (Role::Decode, false),
            (Role::Prefill, true),
        ] {
            let mut p =
                OutputProcessor::new(None, 1, None, role, false, NO_FINISH_LOG, enabled, None);
            register(&mut p, "r", 4, 8, &[]);
            p.states.get_mut("r").unwrap().output_ids.push(10);
            stream_at(&mut p, "r", 100.0).unwrap();
            p.states.get_mut("r").unwrap().output_ids.push(11);
            stream_at(&mut p, "r", 104.0).unwrap();
            let state = p.states.remove("r").unwrap();
            assert!(state.stream_output_timing.is_none());
            p.collect_stream_end(&state, 106.0);
            let samples = p.take_request_metrics();
            assert!(samples.tpot_seconds.is_empty());
            assert!(samples.e2e_tpot_seconds.is_empty());
        }
    }

    #[test]
    fn decode_bootstrap_clock_starts_when_it_is_emitted_and_abort_keeps_sent_tokens() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "r", 4, 8, &[]);
        p.remote_prefill_done("r", 7, 0, None, (SystemTime::now(), Instant::now()));
        assert!(p.state("r").unwrap().stream_output_timing.is_none());
        assert_eq!(stream_at(&mut p, "r", 100.0).unwrap().output_ids, [7]);
        p.states.get_mut("r").unwrap().output_ids.extend([8, 9]);
        assert_eq!(stream_at(&mut p, "r", 106.0).unwrap().output_ids, [8, 9]);
        // The client disappears before this token can be sent.
        p.states.get_mut("r").unwrap().output_ids.push(10);
        p.mark_abort("r", false);
        let before = wall_seconds();
        assert!(p.finish_remote_prefill_only("r").outputs.is_empty());
        let after = wall_seconds();
        let samples = p.take_request_metrics();
        assert_eq!(samples.tpot_seconds, [3.0]);
        assert_eq!(samples.e2e_tpot_seconds.len(), 1);
        // Only the two tokens actually sent after the bootstrap count.
        assert!(samples.e2e_tpot_seconds[0] >= (before - 100.0) / 2.0);
        assert!(samples.e2e_tpot_seconds[0] <= (after - 100.0) / 2.0);
        assert!(p.take_request_metrics().e2e_tpot_seconds.is_empty());
    }

    #[test]
    fn committed_streams_collect_e2e_on_normal_and_transfer_failure_ends() {
        for transfer_failed in [false, true] {
            let mut p = OutputProcessor::new(
                None,
                1,
                None,
                Role::Decode,
                false,
                NO_FINISH_LOG,
                true,
                None,
            );
            register(&mut p, "r", 4, 3, &[]);
            p.states.get_mut("r").unwrap().created_time = wall_seconds() - 1.0;
            p.remote_prefill_done("r", 7, 0, None, (SystemTime::now(), Instant::now()));
            stream_at(&mut p, "r", wall_seconds()).unwrap();
            let second = p
                .commit(&batch(&[("r", 1, 0, 4)], 0), None, &result(&[8], &[1]))
                .unwrap();
            assert_eq!(second.outputs[0].output_ids, [8]);
            let ended = if transfer_failed {
                p.fail_transfer("r")
            } else {
                p.commit(&batch(&[("r", 1, 0, 4)], 0), None, &result(&[9], &[1]))
                    .unwrap()
            };
            assert_eq!(ended.released, ["r"]);
            let samples = p.take_request_metrics();
            assert_eq!(
                samples.tpot_seconds.len(),
                if transfer_failed { 1 } else { 2 }
            );
            assert_eq!(samples.e2e_tpot_seconds.len(), 1);
            let completed = if transfer_failed { 2 } else { 3 };
            assert_eq!(samples.completion_tokens, completed);
            assert_eq!(samples.completion_token_samples, [completed]);
            assert_eq!(samples.request_seconds.len(), 1);
            assert_eq!(samples.ttft_seconds.len(), 1);
            assert_eq!(samples.decode_seconds.len(), 1);
            assert_eq!(p.take_request_tokens(), (4, 0));
        }
    }

    #[test]
    fn one_chunk_requests_and_zero_token_finishes_have_no_tpot() {
        for tokens in [vec![], vec![10], vec![10, 11, 12]] {
            let mut p = processor(None);
            register(&mut p, "r", 4, 8, &[]);
            let state = p.states.get_mut("r").unwrap();
            state.output_ids = tokens;
            state.finished_reason = Some(FinishReason::Length(state.output_ids.len() as i64));
            stream_at(&mut p, "r", 100.0).unwrap();
            let state = p.states.remove("r").unwrap();
            p.collect_stream_end(&state, 105.0);
            let samples = p.take_request_metrics();
            assert!(samples.tpot_seconds.is_empty());
            assert!(samples.e2e_tpot_seconds.is_empty());
        }
    }

    #[test]
    fn request_acceptance_samples_include_bonus_and_drain_once_per_request() {
        let mut p = processor(Some(3));

        register(&mut p, "two_rounds", 1, 5, &[]);
        p.commit(
            &batch(&[("two_rounds", 1, 0, 1)], 1),
            None,
            &result(&[0], &[1]),
        )
        .unwrap();
        p.commit(
            &batch(&[("two_rounds", 3, 0, 1)], 0),
            None,
            &result(&[1, 2], &[2]),
        )
        .unwrap();
        assert_eq!(
            p.commit(
                &batch(&[("two_rounds", 3, 0, 1)], 0),
                None,
                &result(&[3, 4], &[2]),
            )
            .unwrap()
            .released,
            ["two_rounds"]
        );

        register(&mut p, "one_round", 1, 4, &[]);
        p.commit(
            &batch(&[("one_round", 1, 0, 1)], 1),
            None,
            &result(&[10], &[1]),
        )
        .unwrap();
        assert_eq!(
            p.commit(
                &batch(&[("one_round", 3, 0, 1)], 0),
                None,
                &result(&[11, 12, 13], &[3]),
            )
            .unwrap()
            .released,
            ["one_round"]
        );

        // Finishing during prefill has no verify round and emits no sample.
        register(&mut p, "no_verify", 1, 1, &[]);
        assert_eq!(
            p.commit(
                &batch(&[("no_verify", 1, 0, 1)], 1),
                None,
                &result(&[20], &[1]),
            )
            .unwrap()
            .released,
            ["no_verify"]
        );

        let samples = p.take_request_metrics();
        // Three verify rounds across two requests produce exactly two samples.
        // Each value includes every round's bonus token: (5 - 1) / 2 and 4 - 1.
        assert_eq!(samples.spec_decode_accept_length_samples, [2.0, 3.0]);
        assert_eq!(samples.completion_token_samples, [5, 4, 1]);
        assert!(p
            .take_request_metrics()
            .spec_decode_accept_length_samples
            .is_empty());
    }

    #[test]
    fn completion_samples_preserve_request_counts_zeroes_and_drain_boundaries() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "one", 4, 1, &[]);
        p.remote_prefill_done("one", 7, 0, None, (SystemTime::now(), Instant::now()));
        assert_eq!(p.finish_remote_prefill_only("one").released, ["one"]);

        register(&mut p, "two", 4, 2, &[]);
        p.remote_prefill_done("two", 7, 0, None, (SystemTime::now(), Instant::now()));
        let committed = p
            .commit(&batch(&[("two", 1, 0, 4)], 0), None, &result(&[8], &[1]))
            .unwrap();
        assert_eq!(committed.released, ["two"]);

        register(&mut p, "zero", 4, 2, &[]);
        p.mark_abort("zero", false);
        assert_eq!(p.finish_remote_prefill_only("zero").released, ["zero"]);
        let metrics = p.take_request_metrics();
        assert_eq!(metrics.completion_tokens, 3);
        assert_eq!(metrics.completion_token_samples, [1, 2, 0]);
        let drained = p.take_request_metrics();
        assert_eq!(drained.completion_tokens, 0);
        assert!(drained.completion_token_samples.is_empty());
        // Reaping an already removed request cannot duplicate its sample.
        assert!(p.finish_remote_prefill_only("zero").released.is_empty());
        assert!(p.take_request_metrics().completion_token_samples.is_empty());
    }

    #[test]
    fn decode_ready_wait_starts_at_kv_landing_and_is_observed_once() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "a", 4, 8, &[]);
        assert!(p.take_decode_ready_at("a").is_none());
        let before = Instant::now();
        p.remote_prefill_done("a", 7, 0, None, (SystemTime::now(), Instant::now()));
        let after = Instant::now();
        let ready = p.take_decode_ready_at("a").unwrap();
        assert!(before <= ready && ready <= after);
        assert!(p.take_decode_ready_at("a").is_none());

        register(&mut p, "cancelled", 4, 8, &[]);
        p.remote_prefill_done("cancelled", 7, 0, None, (SystemTime::now(), Instant::now()));
        p.states.get_mut("cancelled").unwrap().to_abort = true;
        assert!(p.take_decode_ready_at("cancelled").is_none());

        register(&mut p, "finished", 4, 1, &[]);
        p.remote_prefill_done("finished", 7, 0, None, (SystemTime::now(), Instant::now()));
        p.finish_remote_prefill_only("finished");
        assert!(p.take_decode_ready_at("finished").is_none());
    }

    #[test]
    fn decode_ready_wait_does_not_stamp_without_metrics() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            false,
            None,
        );
        register(&mut p, "a", 4, 8, &[]);
        p.remote_prefill_done("a", 7, 0, None, (SystemTime::now(), Instant::now()));
        assert!(p.take_decode_ready_at("a").is_none());
    }

    #[test]
    fn a_decode_landing_seeds_the_bootstrap_without_starting_the_decode_window() {
        let mut p = OutputProcessor::new(
            None,
            1,
            Some(4096),
            Role::Decode,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "a", 4, 8, &[]);
        let grammar = crate::grammar::ReasoningGrammar {
            continues: String::new(),
            may_open: String::new(),
            schema_slot: String::new(),
            markers: Some((0, 2)),
            implicit_closer: None,
            boundary: None,
        };
        let state = p.states.get_mut("a").unwrap();
        state.reasoning_tokens = Some(crate::request::ReasoningTokens::new(
            &grammar,
            &state.prompt_ids,
        ));
        p.remote_prefill_done("a", 7, 0, None, (SystemTime::now(), Instant::now()));
        let state = p.state("a").expect("still registered");
        assert_eq!(state.first_token_time, 0.0);
        assert_eq!(state.output_ids, vec![7]);
        assert_eq!(state.reasoning_tokens.unwrap().count(), 1);
    }

    #[test]
    fn decode_request_stats_start_at_the_first_local_output_without_an_exporter() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            FinishLog {
                enabled: true,
                request_stats: true,
                ttft_percentage: 0,
            },
            false,
            None,
        );
        register(&mut p, "r", 3, 8, &[]);
        p.remote_prefill_done("r", 9, 0, None, (SystemTime::now(), Instant::now()));
        assert_eq!(p.state("r").unwrap().first_token_time, 0.0);
        p.commit(&batch(&[("r", 1, 0, 3)], 0), None, &result(&[10], &[1]))
            .unwrap();
        let first = p.state("r").unwrap().first_token_time;
        assert!(first > 0.0);
        let state = p.states.get_mut("r").unwrap();
        state.created_time = first - 1.0;
        let line = request_stats::stats_line(
            "r",
            state,
            state.timings.as_ref().unwrap(),
            first + 2.0,
            None,
        );
        assert!(line.contains("ttft_ms=1000.0,"), "{line}");
        assert!(line.contains("prefill_ms=0.0,"), "{line}");
        assert!(line.contains("decode_tps=0.5,"), "{line}");
    }

    #[test]
    fn disabled_metrics_do_not_retain_finished_requests() {
        let mut p = OutputProcessor::new(
            None,
            1,
            Some(4096),
            Role::Plain,
            false,
            NO_FINISH_LOG,
            false,
            None,
        );
        let mut state = RequestState::new(vec![1], 4, BTreeSet::new(), true, None, false);
        state.created_time = 100.0;
        state.first_token_time = 101.0;
        state.output_ids = vec![2, 3];
        p.complete_request("done", &state, &mut []);
        let samples = p.take_request_metrics();
        assert!(samples.ttft_seconds.is_empty());
        assert!(samples.request_seconds.is_empty());
        assert!(samples.decode_seconds.is_empty());
        assert!(samples.tpot_seconds.is_empty());
    }

    #[test]
    fn peer_usage_counts_only_the_remote_prefix_used_at_the_first_commit() {
        let mut p = processor(None);
        register(&mut p, "r", 12, 1, &[]);
        p.states.get_mut("r").unwrap().peer_usage = Some(crate::request::PeerUsage {
            expected: Some(8),
            window: Some((4, 10)),
            disposition: "unused_after_landing",
            ..Default::default()
        });
        p.commit(&batch(&[("r", 2, 8, 12)], 1), None, &result(&[0], &[1]))
            .unwrap();
        p.commit(&batch(&[("r", 2, 10, 12)], 1), None, &result(&[7], &[1]))
            .unwrap();
        assert_eq!(
            p.take_request_metrics().peer_usage,
            vec![(Some(8), 4, "used")]
        );
        assert!(p.take_request_metrics().peer_usage.is_empty());
    }

    #[test]
    fn a_hinted_request_reports_one_terminal_local_reason() {
        let mut p = processor(None);
        for (outcome, reason, expected) in [
            (
                "admission",
                Some("adaptive_recompute"),
                Some("adaptive_recompute"),
            ),
            ("not_attempted", None, Some("not_attempted_unknown")),
            ("pin", Some("adaptive_recompute"), None),
        ] {
            register(&mut p, "r", 12, 1, &[]);
            let mut state = p.states.remove("r").unwrap();
            state.peer_usage = Some(crate::request::PeerUsage {
                disposition: outcome,
                local_reason: reason.map(str::to_owned),
                ..Default::default()
            });
            p.complete_request("r", &state, &mut []);
            assert_eq!(
                p.take_request_metrics().peer_local_reasons,
                expected.into_iter().collect::<Vec<_>>()
            );
            assert!(p.take_request_metrics().peer_local_reasons.is_empty());
        }
    }

    #[test]
    fn finished_requests_feed_prompt_and_cache_totals() {
        let mut p = processor(None);
        register(&mut p, "a", 12, 4, &[]);
        let mut state = p.states.remove("a").expect("registered above");
        state.cached_tokens = 3;
        p.complete_request("a", &state, &mut []);
        assert_eq!(p.take_request_tokens(), (12, 3));
        // Draining is destructive: the next round must not report it again.
        assert_eq!(p.take_request_tokens(), (0, 0));
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
    fn a_request_returning_to_an_earlier_slot_reseeds_its_decode_history() {
        let mut p = processor(None);
        register_history(&mut p, "a", &[10, 11, 12, 13], &[20, 21]);
        let mut rows = HistoryRows::new(Some(3));
        rows.seeds(&history_batch(&[("a", 1, 0)], 1), &p).unwrap();
        rows.seeds(&history_batch(&[("a", 2, 0)], 0), &p).unwrap();
        // The request advanced in slot 2 before a second restore returned
        // it to slot 1. Slot 1 still names it, but holds an older history.
        p.states.get_mut("a").unwrap().output_ids.extend([22, 23]);
        let restored = history_batch(&[("a", 1, 0)], 0);
        assert_eq!(
            rows.seeds(&restored, &p).unwrap(),
            Some(HistorySeeds {
                slots: vec![1],
                prefix_lengths: vec![7],
                tokens: vec![vec![20, 21, 22]],
            })
        );
        assert_eq!(rows.seeds(&restored, &p).unwrap(), None);
    }

    #[test]
    fn a_request_returning_to_an_earlier_slot_reseeds_its_prefill_history() {
        let mut p = processor(None);
        register_history(&mut p, "a", &[10, 11, 12, 13, 14, 15], &[]);
        let mut rows = HistoryRows::new(Some(3));
        rows.seeds(&history_batch(&[("a", 1, 0)], 1), &p).unwrap();
        rows.seeds(&history_batch(&[("a", 2, 2)], 1), &p).unwrap();
        let restored = history_batch(&[("a", 1, 4)], 1);
        assert_eq!(
            rows.seeds(&restored, &p).unwrap(),
            Some(HistorySeeds {
                slots: vec![1],
                prefix_lengths: vec![4],
                tokens: vec![vec![11, 12, 13]],
            })
        );
        assert_eq!(rows.seeds(&restored, &p).unwrap(), None);
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
    fn a_bounded_lookback_seeds_only_the_prefix_tail() {
        let mut p = processor(None);
        register_history(&mut p, "a", &[10, 11, 12, 13], &[20, 21]);
        register_history(&mut p, "b", &[30, 31], &[]);
        let seeds = HistoryRows::new(Some(3))
            .seeds(&history_batch(&[("a", 1, 5), ("b", 2, 2)], 2), &p)
            .unwrap();
        // The tail ends at the prefix length and may span prompt and
        // output; a prefix shorter than the lookback ships whole.
        assert_eq!(
            seeds,
            Some(HistorySeeds {
                slots: vec![1, 2],
                prefix_lengths: vec![5, 2],
                tokens: vec![vec![12, 13, 20], vec![30, 31]],
            })
        );
        let in_prompt = HistoryRows::new(Some(2))
            .seeds(&history_batch(&[("a", 1, 3)], 1), &p)
            .unwrap();
        assert_eq!(in_prompt.unwrap().tokens, [vec![11, 12]]);
    }

    #[test]
    fn bounded_seeds_preserve_landing_reuse_and_recovery() {
        for lookback in [0, 1, 3, 100] {
            let mut p = processor(None);
            register_history(&mut p, "a", &[10, 11, 12, 13], &[20, 21]);
            register_history(&mut p, "b", &[30, 31], &[]);
            let mut rows = HistoryRows::new(Some(lookback));
            // A PD landing excludes the last output: it is the decode input.
            let landing = history_batch(&[("a", 1, 0)], 0);
            let seed = rows.seeds(&landing, &p).unwrap().unwrap();
            let prefix = [10, 11, 12, 13, 20];
            assert_eq!(seed.prefix_lengths, [5]);
            assert_eq!(seed.tokens[0], prefix[5_usize.saturating_sub(lookback)..]);
            assert_eq!(rows.seeds(&landing, &p).unwrap(), None);
            // A prefix hit takes over the slot, then the original request recovers.
            let handoff = rows
                .seeds(&history_batch(&[("b", 1, 2)], 1), &p)
                .unwrap()
                .unwrap();
            assert_eq!(
                handoff.tokens[0],
                [30, 31][2_usize.saturating_sub(lookback)..]
            );
            let recovery = rows
                .seeds(&history_batch(&[("a", 1, 4)], 1), &p)
                .unwrap()
                .unwrap();
            assert_eq!(
                recovery.tokens[0],
                prefix[4_usize.saturating_sub(lookback)..4]
            );
            // Finishing and reusing the client id must still create a new owner.
            p.states.remove("a");
            register_history(&mut p, "a", &[50, 51, 52, 53], &[]);
            let reused = rows
                .seeds(&history_batch(&[("a", 1, 3)], 1), &p)
                .unwrap()
                .unwrap();
            assert_eq!(
                reused.tokens[0],
                [50, 51, 52][3_usize.saturating_sub(lookback)..]
            );
        }
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
    fn replay_after_sending_prompt_logprobs_does_not_regather_or_resend_them() {
        let mut p = processor(None);
        register_scored(&mut p, "r", vec![5, 6, 7, 8], 0, 0, true);
        let prompt = batch(&[("r", 4, 0, 4)], 1);
        let plan = p.logprob_plan(&prompt).input.unwrap();
        let first = p
            .commit(&prompt, Some(&plan), &scored(&plan, &[9]))
            .unwrap();
        assert_eq!(
            first.outputs[0].input_logprobs.as_ref().unwrap().ids,
            [5, 6, 7, 8]
        );
        assert!(p.state("r").unwrap().input_logprobs.is_none());
        assert!(p.state("r").unwrap().input_logprobs_finalized());
        // The request remains live, but a later retraction can replay its
        // prefill. The sent marker outlives the buffers transferred above.
        assert!(p.logprob_plan(&prompt).input.is_none());
        let replay = p.commit(&prompt, None, &result(&[10], &[1])).unwrap();
        assert!(replay.outputs[0].input_logprobs.is_none());
        assert!(p.logprob_plan(&prompt).input.is_none());
        // The first output owns its data independently of the live state.
        drop(p);
        assert_eq!(
            first.outputs[0].input_logprobs.as_ref().unwrap().values,
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
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register_scored(&mut p, "r", vec![5, 6, 7], 0, 0, false);
        assert!(!p.state("r").unwrap().wants_input_logprobs());
        p.remote_prefill_done("r", 9, 0, Some(-0.25), (SystemTime::now(), Instant::now()));
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
    fn a_bootstrap_token_without_a_cumulative_probability_keeps_the_column_aligned() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register_scored(&mut p, "r", vec![5, 6, 7], 0, 0, false);
        // The prefill node sends the first token's logprob, but no cumulative
        // probability.
        p.remote_prefill_done("r", 9, 0, Some(-0.25), (SystemTime::now(), Instant::now()));
        let decode = ForwardResult {
            output_logprobs: Some(vec![-0.5]),
            output_token_cumulative_probs: Some(vec![0.375]),
            ..result(&[10], &[1])
        };
        let c = p
            .commit(&batch(&[("r", 1, 0, 3)], 0), None, &decode)
            .unwrap();
        let output = &c.outputs[0];
        assert_eq!(output.output_ids, [9, 10]);
        assert_eq!(output.output_logprobs, [-0.25, -0.5]);
        // One value per token: the bootstrap token's place holds NaN.
        let probs = &output.output_token_cumulative_probs;
        assert_eq!(probs.len(), 2);
        assert!(probs[0].is_nan());
        assert_eq!(probs[1], 0.375);
        // The next output starts at the next token.
        let c = p
            .commit(
                &batch(&[("r", 1, 0, 3)], 0),
                None,
                &ForwardResult {
                    output_logprobs: Some(vec![-0.75]),
                    output_token_cumulative_probs: Some(vec![0.625]),
                    ..result(&[11], &[1])
                },
            )
            .unwrap();
        assert_eq!(c.outputs[0].output_ids, [11]);
        assert_eq!(c.outputs[0].output_token_cumulative_probs, [0.625]);
    }

    #[test]
    fn the_prefill_role_ships_its_prompt_logprobs_with_its_final_output() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Prefill,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
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
    fn a_failed_transfer_ends_a_prefill_that_finished_computing() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Prefill,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "r", 3, 1, &[]);
        let committed = p
            .commit(&batch(&[("r", 3, 0, 3)], 1), None, &result(&[9], &[1]))
            .unwrap();
        assert!(committed.outputs.is_empty());
        let state = p.state("r").unwrap();
        assert!(state.finished());
        assert!(!state.finished_output);
        assert!(!state.to_abort);

        // The transfer can fail before the next round applies pause(abort).
        let failed = p.fail_transfer("r");
        assert_eq!(failed.outputs.len(), 1);
        assert_eq!(failed.outputs[0].finish_reason, Some("abort"));
        assert_eq!(failed.outputs[0].output_ids, [9]);
        assert_eq!(failed.released, ["r"]);
        assert!(p.state("r").is_none());
        assert!(p.fail_transfer("r").outputs.is_empty());
        assert!(p.finish_prefill("r").outputs.is_empty());
    }

    #[test]
    fn a_failed_transfer_aborts_an_unfinished_request() {
        let mut p = processor(None);
        register(&mut p, "r", 3, 4, &[]);
        let failed = p.fail_transfer("r");
        assert_eq!(failed.outputs.len(), 1);
        assert_eq!(failed.outputs[0].finish_reason, Some("abort"));
        assert_eq!(failed.released, ["r"]);
    }

    #[test]
    fn a_failed_transfer_preserves_client_and_pause_abort_notifications() {
        for max_new in [1, 4] {
            for notify in [false, true] {
                let mut p = OutputProcessor::new(
                    None,
                    1,
                    None,
                    Role::Prefill,
                    false,
                    NO_FINISH_LOG,
                    true,
                    None,
                );
                register(&mut p, "r", 3, max_new, &[]);
                p.commit(&batch(&[("r", 3, 0, 3)], 1), None, &result(&[9], &[1]))
                    .unwrap();
                p.mark_abort("r", notify);

                let failed = p.fail_transfer("r");
                assert_eq!(failed.outputs.len(), usize::from(notify));
                if notify {
                    assert_eq!(failed.outputs[0].finish_reason, Some("abort"));
                }
                assert_eq!(failed.released, ["r"]);
                assert!(p.state("r").is_none());
            }
        }
    }

    #[test]
    fn a_nan_on_the_prefill_role_emits_a_terminal_abort() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Prefill,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "r", 3, 4, &[]);
        let mut nan = result(&[9], &[1]);
        nan.nan_flags = Some(vec![1]);

        let committed = p.commit(&batch(&[("r", 3, 0, 3)], 1), None, &nan).unwrap();
        assert_eq!(
            kinds(&committed),
            [
                ("r".into(), ForwardEventKind::ExtendResult, vec![9]),
                ("r".into(), ForwardEventKind::Abort, vec![]),
            ]
        );
        assert_eq!(committed.stats.nan_aborts, 1);
        assert_eq!(committed.outputs.len(), 1);
        assert_eq!(committed.outputs[0].finish_reason, Some("abort"));
        assert_eq!(committed.outputs[0].output_ids, [9]);
        assert_eq!(
            committed.outputs[0].finish_details,
            Some(
                FinishReason::Abort {
                    message: NUMERICAL_ERROR.to_owned(),
                    err_type: AbortCode::NumericalError,
                    status_code: None,
                }
                .details()
            )
        );
        assert_eq!(committed.released, ["r"]);
        assert!(p.state("r").is_none());
        assert!(p.fail_transfer("r").outputs.is_empty());
        assert!(p.finish_prefill("r").outputs.is_empty());
    }

    #[test]
    fn a_mid_chunk_nan_on_the_prefill_role_aborts_when_the_prompt_completes() {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Prefill,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
        register(&mut p, "r", 6, 4, &[]);

        let mut first = result(&[0], &[1]);
        first.nan_flags = Some(vec![1]);
        let mid = p
            .commit(&batch(&[("r", 3, 0, 6)], 1), None, &first)
            .unwrap();
        assert_eq!(
            kinds(&mid),
            [("r".into(), ForwardEventKind::ExtendResult, vec![])]
        );
        assert!(mid.outputs.is_empty());
        assert!(mid.released.is_empty());
        assert!(p.state("r").is_some());

        let mut last = result(&[7], &[1]);
        last.nan_flags = Some(vec![0]);
        let done = p.commit(&batch(&[("r", 3, 3, 6)], 1), None, &last).unwrap();
        assert_eq!(
            kinds(&done),
            [
                ("r".into(), ForwardEventKind::ExtendResult, vec![7]),
                ("r".into(), ForwardEventKind::Abort, vec![]),
            ]
        );
        assert_eq!(done.stats.nan_aborts, 1);
        assert_eq!(done.outputs.len(), 1);
        assert_eq!(done.outputs[0].finish_reason, Some("abort"));
        assert_eq!(done.outputs[0].output_ids, [7]);
        assert_eq!(
            done.outputs[0].finish_details,
            Some(
                FinishReason::Abort {
                    message: NUMERICAL_ERROR.to_owned(),
                    err_type: AbortCode::NumericalError,
                    status_code: None,
                }
                .details()
            )
        );
        assert_eq!(done.released, ["r"]);
        assert!(p.state("r").is_none());
    }

    #[test]
    fn a_nan_on_the_prefill_role_preserves_client_and_pause_abort_notifications() {
        for notify in [false, true] {
            let mut p = OutputProcessor::new(
                None,
                1,
                None,
                Role::Prefill,
                false,
                NO_FINISH_LOG,
                true,
                None,
            );
            register(&mut p, "r", 3, 4, &[]);
            p.mark_abort("r", notify);
            let mut nan = result(&[9], &[1]);
            nan.nan_flags = Some(vec![1]);

            let committed = p.commit(&batch(&[("r", 3, 0, 3)], 1), None, &nan).unwrap();
            assert_eq!(
                kinds(&committed),
                [
                    ("r".into(), ForwardEventKind::ExtendResult, vec![9]),
                    ("r".into(), ForwardEventKind::Abort, vec![]),
                ]
            );
            assert_eq!(committed.stats.nan_aborts, 0);
            assert_eq!(committed.outputs.len(), usize::from(notify));
            if notify {
                assert_eq!(committed.outputs[0].finish_reason, Some("abort"));
                assert_eq!(committed.outputs[0].output_ids, [9]);
            }
            assert_eq!(committed.released, ["r"]);
            assert!(p.state("r").is_none());
            assert!(p.fail_transfer("r").outputs.is_empty());
        }
    }

    #[test]
    fn ngram_windows_follow_the_committed_tokens() {
        let mut p =
            OutputProcessor::new(None, 1, None, Role::Plain, false, NO_FINISH_LOG, true, None);
        register_state(
            &mut p,
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
            request_ends: Some(vec![REQUEST_ENDED, 0]),
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
    fn a_finished_request_traces_its_lifecycle() {
        // The span one request becomes: its own tokens, its finish reason,
        // and the timestamps its timings kept — which tracing alone, without
        // the stats log, is enough to switch on.
        let (trace, received) = crate::trace::TracePublisher::for_tests(1.0);
        let mut p = OutputProcessor::new(
            None,
            1,
            Some(4096),
            Role::Plain,
            false,
            NO_FINISH_LOG,
            false,
            Some(trace),
        );
        register(&mut p, "r", 4, 10, &[2]);
        p.states.get_mut("r").unwrap().session_id = Some("sess-7".to_owned());
        p.mark_scheduled(&["r".to_owned()]);
        let c = p
            .commit(&batch(&[("r", 4, 0, 4)], 1), None, &result(&[7], &[1]))
            .unwrap();
        assert!(c.outputs[0].finish_reason.is_none());
        let c = p
            .commit(&batch(&[("r", 1, 0, 4)], 0), None, &result(&[2], &[1]))
            .unwrap();
        assert_eq!(c.released, ["r"]);
        let record = received
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the finished request traced");
        assert_eq!(
            (
                record.rid.as_str(),
                record.prompt_tokens,
                record.output_tokens
            ),
            ("r", 4, 2)
        );
        assert_eq!(record.finish_reason, "stop");
        assert_eq!(record.session.as_deref(), Some("sess-7"));
        assert!(record.scheduled > 0.0, "tracing stamps the schedule point");
        assert!(record.first_token >= record.scheduled);
        assert!(record.finish >= record.first_token);
        // Unsampled requests build no record at all.
        let (trace, received) = crate::trace::TracePublisher::for_tests(0.0);
        let mut p = OutputProcessor::new(
            None,
            1,
            Some(4096),
            Role::Plain,
            false,
            NO_FINISH_LOG,
            false,
            Some(trace),
        );
        register(&mut p, "r", 4, 10, &[2]);
        p.commit(&batch(&[("r", 4, 0, 4)], 1), None, &result(&[2], &[1]))
            .unwrap();
        assert!(received
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
    }

    #[test]
    fn decode_trace_starts_at_the_first_local_output() {
        for (max_new, metrics_enabled) in [(1, false), (1, true), (2, false), (2, true)] {
            let (trace, received) = crate::trace::TracePublisher::for_tests(1.0);
            let mut p = OutputProcessor::new(
                None,
                1,
                Some(4096),
                Role::Decode,
                false,
                NO_FINISH_LOG,
                metrics_enabled,
                Some(trace),
            );
            register(&mut p, "r", 4, max_new, &[]);
            p.states.get_mut("r").unwrap().created_time = wall_seconds();
            p.remote_prefill_done("r", 7, 2, None, (SystemTime::now(), Instant::now()));
            let (_, prefill_done, first_token, _, _) = p
                .state("r")
                .unwrap()
                .timings
                .as_ref()
                .unwrap()
                .trace_points();
            assert!(prefill_done > 0.0, "stamp KV landing");
            assert_eq!(first_token, 0.0, "the remote token does not start decode");
            if max_new == 1 {
                assert_eq!(p.finish_remote_prefill_only("r").released, ["r"]);
            } else {
                assert!(p.finish_remote_prefill_only("r").released.is_empty());
                p.mark_scheduled(&["r".to_owned()]);
                let c = p
                    .commit(&batch(&[("r", 1, 0, 4)], 0), None, &result(&[8], &[1]))
                    .unwrap();
                assert_eq!(c.released, ["r"]);
            }
            let record = received.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(record.role, Role::Decode);
            assert!(record.first_token > 0.0, "stamp the first local output");
            assert!(record.first_token >= prefill_done);
            assert_eq!(record.prefill_done, prefill_done);
            assert_eq!(record.output_tokens, max_new);
            if max_new == 1 {
                assert_eq!(record.scheduled, 0.0);
            } else {
                assert!(record.scheduled <= record.first_token);
            }
            let metrics = p.take_request_metrics();
            if metrics_enabled {
                assert_eq!(metrics.ttft_seconds, [record.first_token - record.created]);
                assert_eq!(metrics.request_seconds.len(), 1);
                assert_eq!(metrics.decode_seconds.len(), 1);
                // Bootstrap and decode tokens are emitted together in this test.
                assert!(metrics.tpot_seconds.is_empty());
                assert!(metrics.e2e_tpot_seconds.is_empty());
                assert_eq!(metrics.completion_tokens, max_new as u64);
                assert_eq!(metrics.completion_token_samples, [max_new as u64]);
                assert_eq!(p.take_request_tokens(), (4, 2));
            } else {
                assert!(metrics.ttft_seconds.is_empty());
                assert!(metrics.request_seconds.is_empty());
                assert!(metrics.decode_seconds.is_empty());
                assert!(metrics.tpot_seconds.is_empty());
                assert_eq!(metrics.completion_tokens, 0);
                assert!(metrics.completion_token_samples.is_empty());
                assert_eq!(p.take_request_tokens(), (0, 0));
            }
        }
    }

    #[test]
    fn remote_prefill_abort_does_not_invent_completion_boundaries() {
        let (trace, received) = crate::trace::TracePublisher::for_tests(1.0);
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            false,
            NO_FINISH_LOG,
            false,
            Some(trace),
        );
        register(&mut p, "r", 4, 2, &[]);
        p.mark_abort("r", true);
        assert_eq!(p.finish_remote_prefill_only("r").released, ["r"]);
        let record = received.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(record.finish_reason, "abort");
        assert_eq!(record.first_token, 0.0);
        assert_eq!(record.prefill_done, 0.0);
        assert_eq!(record.output_tokens, 0);
    }

    #[test]
    fn prefill_trace_keeps_its_role_through_transfer_completion() {
        let (trace, received) = crate::trace::TracePublisher::for_tests(1.0);
        let mut p = OutputProcessor::new(
            None,
            1,
            Some(4096),
            Role::Prefill,
            false,
            NO_FINISH_LOG,
            false,
            Some(trace),
        );
        register(&mut p, "r", 4, 2, &[]);
        p.mark_scheduled(&["r".to_owned()]);
        let c = p
            .commit(&batch(&[("r", 4, 0, 4)], 1), None, &result(&[7], &[1]))
            .unwrap();
        assert!(c.released.is_empty());
        let (_, _, first_token, _, _) = p
            .state("r")
            .unwrap()
            .timings
            .as_ref()
            .unwrap()
            .trace_points();
        assert_eq!(p.finish_prefill("r").released, ["r"]);
        let record = received.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(record.role, Role::Prefill);
        assert_eq!(record.first_token, first_token);
        assert!(record.prefill_done >= record.scheduled);
        assert!(record.finish >= first_token);
    }

    #[test]
    fn a_decode_reservation_reaches_the_scheduler_only_when_it_changes() {
        // Verify width 3: each decode row stores three tokens and keeps some.
        let mut p = processor(Some(3));
        register(&mut p, "r", 4, 100, &[]);
        p.commit(&batch(&[("r", 4, 0, 4)], 1), None, &result(&[7], &[1]))
            .unwrap();
        let reserves: Vec<Vec<i32>> = [2, 2, 3, 3, 3, 1]
            .into_iter()
            .map(|kept| {
                let committed = p
                    .commit(
                        &batch(&[("r", 3, 0, 4)], 0),
                        None,
                        &result(&[8, 9, 10], &[kept]),
                    )
                    .unwrap();
                committed
                    .events
                    .iter()
                    .filter(|event| event.kind == ForwardEventKind::UpdateReserveNumTokens)
                    .map(|event| event.reserve_num_tokens)
                    .collect()
            })
            .collect();
        assert_eq!(
            reserves,
            [vec![2], vec![], vec![3], vec![], vec![], vec![1]]
        );
    }

    #[test]
    fn resident_tokens_include_waiting_prompts_and_committed_output() {
        let mut p = processor(None);
        register_history(&mut p, "decoding", &[1, 2, 3], &[7, 8]);
        register_history(&mut p, "waiting", &[4, 5, 6, 7, 8], &[]);
        assert_eq!(p.num_resident_tokens(), 10);
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
        assert_eq!(p.num_resident_tokens(), 5);

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
        assert_eq!(p.num_resident_tokens(), 0);
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
    fn a_finished_request_leaves_even_when_no_output_is_due() {
        // Release follows the finish, not the output: with its final output
        // already marked sent, the finishing commit streams nothing and still
        // releases the request.
        let mut p = processor(None);
        register(&mut p, "r", 2, 2, &[]);
        p.commit(&batch(&[("r", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        p.states.get_mut("r").unwrap().finished_output = true;
        let c = p
            .commit(&batch(&[("r", 1, 0, 2)], 0), None, &result(&[4], &[1]))
            .unwrap();
        assert_eq!(c.events[1].kind, ForwardEventKind::Finish);
        assert!(c.outputs.is_empty());
        assert_eq!(c.released, ["r"]);
        assert_eq!(p.num_running(), 0);
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

    /// A processor whose cache defers its decode state to a request's end.
    fn deferring_processor(spec_num_tokens: Option<usize>) -> OutputProcessor {
        OutputProcessor::new(
            spec_num_tokens,
            1,
            Some(4096),
            Role::Plain,
            true,
            NO_FINISH_LOG,
            true,
            None,
        )
    }

    #[test]
    fn a_finish_the_device_did_not_see_publishes_nothing_when_state_is_deferred() {
        // A grammar's end with no stop token (as under ignore_eos) and a
        // client abort end where the device's stop criteria do not, so the
        // deferred decode state is incomplete: the scheduler gets an abort.
        // The client still sees the grammar's ordinary finish.
        let mut p = deferring_processor(Some(3));
        register(&mut p, "g", 2, 10, &[]);
        p.attach_grammar("g", grammar());
        p.commit(&batch(&[("g", 2, 0, 2)], 1), None, &result(&[0], &[1]))
            .unwrap();
        let mut verify = result(&[1, 8, 5], &[3]);
        verify.grammar_terminated_at = Some(vec![1]);
        let c = p
            .commit(&batch(&[("g", 3, 0, 2)], 0), None, &verify)
            .unwrap();
        assert_eq!(c.outputs[0].finish_reason, Some("stop"));
        assert_eq!(
            kinds(&c),
            [
                ("g".into(), ForwardEventKind::ExtendResult, vec![1, 8, 5]),
                ("g".into(), ForwardEventKind::Abort, vec![]),
            ]
        );

        register(&mut p, "a", 2, 10, &[]);
        p.commit(&batch(&[("a", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        p.mark_abort("a", false);
        let c = p
            .commit(&batch(&[("a", 1, 0, 2)], 0), None, &result(&[4], &[1]))
            .unwrap();
        assert_eq!(c.events[1].kind, ForwardEventKind::Abort);
    }

    /// A result whose every row the device reports with `end`.
    fn ended(tokens: &[i32], lengths: &[i32], end: i32) -> ForwardResult {
        ForwardResult {
            request_ends: Some(vec![end; lengths.len()]),
            ..result(tokens, lengths)
        }
    }

    #[test]
    fn a_finish_the_device_saw_publishes_when_state_is_deferred() {
        // A stop token and a spent budget end the request on the device too,
        // which reports it with the deferred state complete.
        let mut p = deferring_processor(None);
        register(&mut p, "s", 2, 10, &[7]);
        p.commit(&batch(&[("s", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        let c = p
            .commit(
                &batch(&[("s", 1, 0, 2)], 0),
                None,
                &ended(&[7], &[1], REQUEST_ENDED),
            )
            .unwrap();
        assert_eq!(c.events[1].kind, ForwardEventKind::Finish);

        register(&mut p, "l", 2, 1, &[]);
        let c = p
            .commit(
                &batch(&[("l", 2, 0, 2)], 1),
                None,
                &ended(&[3], &[1], REQUEST_ENDED),
            )
            .unwrap();
        assert_eq!(c.outputs[0].finish_reason, Some("length"));
        assert_eq!(c.events[1].kind, ForwardEventKind::Finish);
    }

    #[test]
    fn a_deferred_state_publishes_on_the_device_word_alone() {
        // The stop token ends the request here, but the device reports no
        // end (its stop check saw another token): the state it holds may lack
        // the finish, so the scheduler gets an abort.
        let mut p = deferring_processor(None);
        register(&mut p, "s", 2, 10, &[7]);
        p.commit(&batch(&[("s", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        let c = p
            .commit(&batch(&[("s", 1, 0, 2)], 0), None, &result(&[7], &[1]))
            .unwrap();
        assert_eq!(c.outputs[0].finish_reason, Some("stop"));
        assert_eq!(c.events[1].kind, ForwardEventKind::Abort);
    }

    #[test]
    fn a_state_the_device_could_not_complete_is_an_engine_error() {
        // The device ended the request but rejected a layer's commit: the
        // state its finish would publish lacks accepted tokens.
        let mut p = deferring_processor(None);
        register(&mut p, "s", 2, 10, &[7]);
        p.commit(&batch(&[("s", 2, 0, 2)], 1), None, &result(&[3], &[1]))
            .unwrap();
        let error = p
            .commit(
                &batch(&[("s", 1, 0, 2)], 0),
                None,
                &ended(&[7], &[1], REQUEST_ENDED_INCOMPLETE),
            )
            .unwrap_err();
        assert!(matches!(error, CommitError::IncompleteDecodeState(rid) if rid == "s"));
    }

    #[test]
    fn a_deferred_state_needs_the_device_request_ends() {
        let mut p = deferring_processor(None);
        register(&mut p, "s", 2, 10, &[7]);
        let bare = ForwardResult {
            request_ends: None,
            ..result(&[3], &[1])
        };
        let error = p
            .commit(&batch(&[("s", 2, 0, 2)], 1), None, &bare)
            .unwrap_err();
        assert!(matches!(error, CommitError::MissingColumn("request_ends")));
        // A cache without deferred state needs no report.
        let mut plain = processor(None);
        register(&mut plain, "s", 2, 10, &[7]);
        plain
            .commit(&batch(&[("s", 2, 0, 2)], 1), None, &bare)
            .unwrap();
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
        let mut p = OutputProcessor::new(
            Some(4),
            1,
            Some(8),
            Role::Plain,
            false,
            NO_FINISH_LOG,
            true,
            None,
        );
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
        register_state(
            &mut p,
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

    #[test]
    fn cumulative_probabilities_follow_streamed_tokens_and_logprob_requests() {
        let mut p = processor(None);
        for (rid, logprobs) in [("scored", true), ("plain", false)] {
            register_state(
                &mut p,
                rid.to_owned(),
                RequestState::new(vec![0], 4, BTreeSet::new(), true, Some(2), logprobs),
            );
        }
        for (step, token) in [5, 6, 7, 8].into_iter().enumerate() {
            let mut forward = result(&[token, token + 10], &[1, 1]);
            forward.output_logprobs = Some(vec![-(token as f64), -10.0]);
            forward.output_token_cumulative_probs = Some(vec![token as f64 / 10.0, 0.9]);
            let c = p
                .commit(
                    &batch(
                        &[("scored", 1, 0, 1), ("plain", 1, 0, 1)],
                        if step == 0 { 2 } else { 0 },
                    ),
                    None,
                    &forward,
                )
                .unwrap();
            if step == 1 {
                assert!(c.outputs.is_empty());
                continue;
            }
            let scored = c
                .outputs
                .iter()
                .find(|output| output.rid == "scored")
                .unwrap();
            let expected = match step {
                0 => vec![5],
                2 => vec![6, 7],
                _ => vec![8],
            };
            assert_eq!(scored.output_ids, expected);
            assert_eq!(
                scored.output_logprobs,
                expected.iter().map(|&id| -(id as f64)).collect::<Vec<_>>()
            );
            assert_eq!(
                scored.output_token_cumulative_probs,
                expected
                    .iter()
                    .map(|&id| id as f64 / 10.0)
                    .collect::<Vec<_>>()
            );
            let plain = c
                .outputs
                .iter()
                .find(|output| output.rid == "plain")
                .unwrap();
            assert!(plain.output_logprobs.is_empty());
            assert!(plain.output_token_cumulative_probs.is_empty());
        }
    }

    #[test]
    fn a_stop_token_cuts_cumulative_probabilities_with_its_verify_window() {
        let mut p = processor(Some(3));
        register_state(
            &mut p,
            "r".to_owned(),
            RequestState::new(vec![0], 10, BTreeSet::from([7]), true, None, true),
        );
        let mut first = result(&[5], &[1]);
        first.output_logprobs = Some(vec![-0.5]);
        first.output_token_cumulative_probs = Some(vec![0.1]);
        p.commit(&batch(&[("r", 1, 0, 1)], 1), None, &first)
            .unwrap();
        let mut verify = result(&[6, 7, 8], &[3]);
        verify.output_logprobs = Some(vec![-0.6, -0.7, -0.8]);
        verify.output_token_cumulative_probs = Some(vec![0.2, 0.3, 0.4]);
        let c = p
            .commit(&batch(&[("r", 3, 0, 1)], 0), None, &verify)
            .unwrap();
        assert_eq!(c.outputs[0].output_ids, [6, 7]);
        assert_eq!(p.take_generated_tokens(), 2);
        p.commit(&batch(&[("r", 3, 0, 1)], 0), None, &verify)
            .unwrap();
        assert_eq!(p.take_generated_tokens(), 0);
        assert_eq!(c.outputs[0].output_logprobs, [-0.6, -0.7]);
        assert_eq!(c.outputs[0].output_token_cumulative_probs, [0.2, 0.3]);
    }

    #[test]
    fn cumulative_probabilities_must_cover_the_result_tokens() {
        let mut p = processor(None);
        register(&mut p, "r", 1, 2, &[]);
        let mut forward = result(&[5], &[1]);
        forward.output_token_cumulative_probs = Some(vec![]);
        assert!(matches!(
            p.commit(&batch(&[("r", 1, 0, 1)], 1), None, &forward),
            Err(CommitError::TokenColumn {
                column: "output_token_cumulative_probs",
                ..
            })
        ));
    }
}
