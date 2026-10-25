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

//! Per-step outputs from the engine to a frontend (`engine/io_struct.py`).

use crate::{
    codec::Dict,
    error::{Error, Result},
    value::Value,
};

wire_struct! {
    /// One step's output for a frontend that detokenizes itself.
    ///
    /// Every column is parallel to `rids`. The load fields at the tail are
    /// the scheduler's latest snapshot; an older producer omits them and
    /// they decode as zero.
    #[derive(Debug, Clone, PartialEq)]
    pub struct BatchTokenIDOutSlim: array("BatchTokenIDOutSlim") {
        /// Request ids. Always present on this wire.
        pub rids: Option<Vec<String>> = None,
        /// Token ids generated this step, per request.
        pub output_ids: Vec<Vec<i32>>,
        /// `""` while running, else the finish type (`"stop"`, `"length"`,
        /// `"abort"`).
        pub finished_reasons: Vec<String>,
        /// Prompt length.
        pub prompt_tokens: Vec<i64>,
        /// Tokens generated so far.
        pub completion_tokens: Vec<i64>,
        /// Prompt tokens served from the prefix cache.
        pub cached_tokens: Vec<i64>,
        /// Logprobs of this step's tokens; empty for requests without logprobs.
        pub output_token_logprobs_val: Vec<Vec<f64>>,
        /// The token ids `output_token_logprobs_val` refers to.
        pub output_token_logprobs_idx: Vec<Vec<i32>>,
        /// The producing DP rank's engine index.
        pub engine_index: i64 = 0,
        /// Running requests.
        pub num_running: i64 = 0,
        /// Waiting requests.
        pub num_waiting: i64 = 0,
        /// KV pages in use by active requests.
        pub kv_active_pages: i64 = 0,
        /// KV pages in total.
        pub kv_total_pages: i64 = 0,
        /// The finish reason in full (`BaseFinishReason.to_json()`: the
        /// matched token, an abort's message and codes), `None` while
        /// running; one entry per request.
        pub finish_reason_details: Vec<Option<FinishDetails>> = Vec::new(),
        /// With speculative decoding, each request's verify rounds so far;
        /// empty without.
        pub spec_verify_ct: Vec<i64> = Vec::new(),
        /// With speculative decoding, each finished request's accepted
        /// tokens per verify round, `None` before it finishes; empty
        /// without.
        pub accept_draft_tokens: Vec<Option<f64>> = Vec::new(),
        /// Prompt logprobs, shipped once per request: `None` for the first
        /// position, then each position's logprob given its prefix. This and
        /// the top-k columns hold one entry per request in a frame where
        /// some request carries one, and are empty for the whole frame
        /// otherwise.
        pub input_token_logprobs_val: Vec<Vec<Option<f64>>> = Vec::new(),
        /// The prompt tokens `input_token_logprobs_val` refers to.
        pub input_token_logprobs_idx: Vec<Vec<i32>> = Vec::new(),
        /// The top-k alternatives' logprobs at each prompt position (`None`
        /// for the first), most likely first.
        pub input_top_logprobs_val: Vec<Vec<Option<Vec<f64>>>> = Vec::new(),
        /// Their token ids.
        pub input_top_logprobs_idx: Vec<Vec<Option<Vec<i32>>>> = Vec::new(),
        /// The top-k alternatives' logprobs of each token this step.
        pub output_top_logprobs_val: Vec<Vec<Vec<f64>>> = Vec::new(),
        /// Their token ids.
        pub output_top_logprobs_idx: Vec<Vec<Vec<i32>>> = Vec::new(),
    }
}

/// A finish reason in full, as a map (`"type"`, then `"matched"`,
/// `"length"` or `"message"`/`"err_type"`/`"status_code"`).
pub type FinishDetails = Dict<String, Value>;

/// Top-k alternatives per token: their logprobs and token ids, most likely
/// first.
pub type TopLogprobs = (Vec<Vec<f64>>, Vec<Vec<i32>>);

/// Top-k alternatives per prompt position, `None` for the first.
pub type PromptTopLogprobs = (Vec<Option<Vec<f64>>>, Vec<Option<Vec<i32>>>);

/// [`TopLogprobs`] borrowed from a batch.
pub type TopLogprobRows<'a> = (&'a [Vec<f64>], &'a [Vec<i32>]);

/// [`PromptTopLogprobs`] borrowed from a batch.
pub type PromptTopLogprobRows<'a> = (&'a [Option<Vec<f64>>], &'a [Option<Vec<i32>>]);

/// One request's slice of a [`BatchTokenIDOutSlim`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestOutput<'a> {
    /// Request id.
    pub rid: &'a str,
    /// Token ids generated this step.
    pub output_ids: &'a [i32],
    /// The finish type, or `None` while the request runs.
    pub finish_reason: Option<&'a str>,
    /// Prompt length.
    pub prompt_tokens: i64,
    /// Tokens generated so far.
    pub completion_tokens: i64,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: i64,
    /// Logprobs of this step's tokens; empty when not requested.
    pub output_token_logprobs: &'a [f64],
    /// The finish reason in full, once the request ended (from a producer
    /// that sends it).
    pub finish_details: Option<&'a FinishDetails>,
    /// Verify rounds so far, with speculative decoding.
    pub spec_verify_ct: Option<i64>,
    /// Accepted tokens per verify round, once finished with speculative
    /// decoding.
    pub accept_draft_tokens: Option<f64>,
    /// Prompt logprobs and their tokens, in the frame that ships them.
    pub input_token_logprobs: Option<(&'a [Option<f64>], &'a [i32])>,
    /// The prompt positions' top-k alternatives, in the frame that ships
    /// them.
    pub input_top_logprobs: Option<PromptTopLogprobRows<'a>>,
    /// This step's tokens' top-k alternatives, when the request asked for
    /// them.
    pub output_top_logprobs: Option<TopLogprobRows<'a>>,
}

impl BatchTokenIDOutSlim {
    /// The per-request rows, after checking that every column has one entry
    /// per request.
    pub fn requests(&self) -> Result<impl Iterator<Item = RequestOutput<'_>>> {
        let rids = self.rids.as_deref().ok_or(Error::MissingField("rids"))?;
        let len = rids.len();
        let columns = [
            ("output_ids", self.output_ids.len()),
            ("finished_reasons", self.finished_reasons.len()),
            ("prompt_tokens", self.prompt_tokens.len()),
            ("completion_tokens", self.completion_tokens.len()),
            ("cached_tokens", self.cached_tokens.len()),
            (
                "output_token_logprobs_val",
                self.output_token_logprobs_val.len(),
            ),
            (
                "output_token_logprobs_idx",
                self.output_token_logprobs_idx.len(),
            ),
        ];
        if let Some((name, found)) = columns.iter().find(|(_, found)| *found != len) {
            return Err(Error::Invalid(format!(
                "column `{name}` has {found} entries for {len} requests"
            )));
        }
        // The tail's columns are either per request or empty for the frame.
        let optional = [
            ("finish_reason_details", self.finish_reason_details.len()),
            ("spec_verify_ct", self.spec_verify_ct.len()),
            ("accept_draft_tokens", self.accept_draft_tokens.len()),
            (
                "input_token_logprobs_val",
                self.input_token_logprobs_val.len(),
            ),
            (
                "input_token_logprobs_idx",
                self.input_token_logprobs_idx.len(),
            ),
            ("input_top_logprobs_val", self.input_top_logprobs_val.len()),
            ("input_top_logprobs_idx", self.input_top_logprobs_idx.len()),
            (
                "output_top_logprobs_val",
                self.output_top_logprobs_val.len(),
            ),
            (
                "output_top_logprobs_idx",
                self.output_top_logprobs_idx.len(),
            ),
        ];
        if let Some((name, found)) = optional
            .iter()
            .find(|(_, found)| *found != 0 && *found != len)
        {
            return Err(Error::Invalid(format!(
                "column `{name}` has {found} entries for {len} requests"
            )));
        }
        fn pair<'a, A, B>(
            values: &'a [Vec<A>],
            ids: &'a [Vec<B>],
            i: usize,
        ) -> Option<(&'a [A], &'a [B])> {
            Some((values.get(i)?.as_slice(), ids.get(i)?.as_slice())).filter(|(v, _)| !v.is_empty())
        }
        Ok((0..len).map(move |i| RequestOutput {
            rid: &rids[i],
            output_ids: &self.output_ids[i],
            finish_reason: Some(self.finished_reasons[i].as_str()).filter(|r| !r.is_empty()),
            prompt_tokens: self.prompt_tokens[i],
            completion_tokens: self.completion_tokens[i],
            cached_tokens: self.cached_tokens[i],
            output_token_logprobs: &self.output_token_logprobs_val[i],
            finish_details: self.finish_reason_details.get(i).and_then(Option::as_ref),
            spec_verify_ct: self.spec_verify_ct.get(i).copied(),
            accept_draft_tokens: self.accept_draft_tokens.get(i).copied().flatten(),
            input_token_logprobs: pair(
                &self.input_token_logprobs_val,
                &self.input_token_logprobs_idx,
                i,
            ),
            input_top_logprobs: pair(
                &self.input_top_logprobs_val,
                &self.input_top_logprobs_idx,
                i,
            ),
            output_top_logprobs: pair(
                &self.output_top_logprobs_val,
                &self.output_top_logprobs_idx,
                i,
            ),
        }))
    }
}
