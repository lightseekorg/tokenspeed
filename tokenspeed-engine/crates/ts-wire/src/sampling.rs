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

//! Sampling parameters (`sampling/sampling_params.py`).

use crate::{
    codec::{Dict, StrOrList},
    value::Value,
};

wire_struct! {
    /// `SamplingParams`, an untagged array-layout struct nested in a request.
    ///
    /// This is the data only. The Python class also normalizes on
    /// construction and decode (`__post_init__`: greedy for a near-zero
    /// temperature, `stop` copied to `stop_strs`, ...) unless
    /// `is_normalized` is set; decoding here applies none of that.
    #[derive(Debug, Clone, PartialEq)]
    pub struct SamplingParams: array + Default {
        /// Generation length limit; `None` leaves it to the engine.
        pub max_new_tokens: Option<i64> = None,
        /// API input for stop strings; normalization copies it to `stop_strs`.
        pub stop: Option<StrOrList> = None,
        /// Token ids that end generation (a Python `set`).
        pub stop_token_ids: Option<Vec<i64>> = None,
        /// Sampling temperature.
        pub temperature: f64 = 1.0,
        /// Nucleus sampling mass.
        pub top_p: f64 = 1.0,
        /// Top-k cutoff; `-1` disables it before normalization.
        pub top_k: i64 = -1,
        /// Minimum probability relative to the most likely token.
        pub min_p: f64 = 0.0,
        /// Frequency penalty.
        pub frequency_penalty: f64 = 0.0,
        /// Presence penalty.
        pub presence_penalty: f64 = 0.0,
        /// Repetition penalty.
        pub repetition_penalty: f64 = 1.0,
        /// Tokens generated before stop conditions apply.
        pub min_new_tokens: i64 = 0,
        /// JSON schema constraint.
        pub json_schema: Option<String> = None,
        /// Regex constraint.
        pub regex: Option<String> = None,
        /// EBNF grammar constraint.
        pub ebnf: Option<String> = None,
        /// Structural tag constraint.
        pub structural_tag: Option<String> = None,
        /// Keep generating past end-of-sequence tokens.
        pub ignore_eos: bool = false,
        /// Detokenization option, carried for the frontend.
        pub skip_special_tokens: bool = true,
        /// Detokenization option, carried for the frontend.
        pub spaces_between_special_tokens: bool = true,
        /// Keep the matched stop string in the output.
        pub no_stop_trim: bool = false,
        /// Reasoning-token budget.
        pub thinking_budget: Option<i64> = None,
        /// Free-form parameters for custom logit processors.
        pub custom_params: Option<Dict<String, Value>> = None,
        /// Streaming chunk interval in tokens.
        pub stream_interval: Option<i64> = None,
        /// Per-token logit offsets, keyed by token id as a string.
        pub logit_bias: Option<Dict<String, f64>> = None,
        /// Sampling seed.
        pub seed: Option<i64> = None,
        /// Output logprobs: `None` off, `0` the sampled token's logprob.
        pub logprobs: Option<i64> = None,
        /// OpenAI `n`; the engine accepts only `1`.
        pub n: i64 = 1,
        /// Stop strings after normalization.
        pub stop_strs: Option<StrOrList> = None,
        /// Longest stop string, set by normalization.
        pub stop_str_max_len: i64 = 0,
        /// Whether normalization already ran on the sender.
        pub is_normalized: bool = false,
    }
}
