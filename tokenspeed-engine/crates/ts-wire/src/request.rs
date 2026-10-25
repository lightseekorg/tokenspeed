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

//! Requests from a frontend to the engine (`engine/io_struct.py`).

use crate::{
    codec::{Dict, StrOrList},
    multimodal::MultimodalInputs,
    sampling::SamplingParams,
    value::Value,
};

wire_struct! {
    /// Continual-prompting session of a request. Array layout, untagged.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct SessionParams: array + Default {
        /// Session id.
        pub id: Option<String> = None,
        /// The request in the session to continue from.
        pub rid: Option<String> = None,
        /// Token offset to continue from.
        pub offset: Option<i64> = None,
        /// Replace the continued request instead of branching.
        pub replace: Option<bool> = None,
    }
}

wire_struct! {
    /// A tokenized generate request, the payload of an ADD message.
    ///
    /// The first two fields come from the Python `BaseReq` base class.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TokenizedGenerateReqInput: array("TokenizedGenerateReqInput") {
        /// Request id. Generate requests always carry a single string.
        pub rid: Option<StrOrList> = None,
        /// In-process HTTP worker address; unused across processes.
        pub http_worker_ipc: Option<String> = None,
        /// Prompt text, when the frontend has it.
        pub input_text: Option<String> = None,
        /// Prompt token ids; `None` on the input-embeds path.
        pub input_ids: Option<Vec<i32>> = None,
        /// Sampling parameters.
        pub sampling_params: SamplingParams,
        /// Return the sampled tokens' logprobs.
        pub return_logprob: bool = false,
        /// First prompt position to return logprobs for; `-1` is the last
        /// prompt token only.
        pub logprob_start_len: i64 = -1,
        /// Top-k alternatives per position.
        pub top_logprobs_num: i64 = 0,
        /// Token ids to score per position.
        pub token_ids_logprob: Option<Vec<i32>> = None,
        /// Stream outputs as they are generated.
        pub stream: bool = false,
        /// Prompt embeddings (nested float lists).
        pub input_embeds: Option<Vec<Value>> = None,
        /// Continual-prompting session.
        pub session_params: Option<SessionParams> = None,
        /// Serialized custom logit processor.
        pub custom_logit_processor: Option<String> = None,
        /// Return hidden states.
        pub return_hidden_states: bool = false,
        /// Creation time, in seconds.
        pub created_time: f64 = 0.0,
        /// PD: prefill bootstrap host.
        pub bootstrap_host: Option<String> = None,
        /// PD: prefill bootstrap port.
        pub bootstrap_port: Option<i64> = None,
        /// PD: transfer room id; frontends mint it as a random 64-bit value.
        pub bootstrap_room: Option<u64> = None,
        /// Additional token-id streams for multi-stream models.
        pub input_multi_ids: Option<Vec<Vec<i32>>> = None,
        /// Model-specific per-request inputs.
        pub input_extra_infos: Option<Vec<Dict<Value, Value>>> = None,
        /// Prompt ids before multimodal placeholder substitution.
        pub input_ids_unpadded: Option<Vec<i32>> = None,
        /// Multimodal payload.
        pub multimodal_inputs: Option<MultimodalInputs> = None,
        /// Set by a validating transport: the engine finishes the request
        /// with an abort carrying this reason.
        pub validation_error: Option<String> = None,
        /// Pin to an attention-DP rank.
        pub data_parallel_rank: Option<i64> = None,
    }
}

impl TokenizedGenerateReqInput {
    /// A request with every optional field at its schema default.
    pub fn new(rid: impl Into<String>, sampling_params: SamplingParams) -> Self {
        Self {
            rid: Some(StrOrList::One(rid.into())),
            http_worker_ipc: None,
            input_text: None,
            input_ids: None,
            sampling_params,
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: None,
            stream: false,
            input_embeds: None,
            session_params: None,
            custom_logit_processor: None,
            return_hidden_states: false,
            created_time: 0.0,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            input_multi_ids: None,
            input_extra_infos: None,
            input_ids_unpadded: None,
            multimodal_inputs: None,
            validation_error: None,
            data_parallel_rank: None,
        }
    }
}
