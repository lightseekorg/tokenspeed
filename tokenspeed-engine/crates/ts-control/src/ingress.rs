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

//! Admitting a generate request off the msgpack wire: validating it against
//! what this engine serves, resolving its sampling defaults and budget, and
//! turning it into the scheduler's request.
//!
//! An invalid request is not dropped, which would leave the frontend's stream
//! open forever: it is admitted finished with an abort carrying the reason.

use std::collections::BTreeSet;

use ts_pd::{BootstrapInfo, EmbeddingItem};
use ts_scheduler::RequestSpec;
use ts_wire::{SamplingParams, StrOrList, TokenizedGenerateReqInput};

use crate::{
    request::{LogprobRequest, RequestState, MAX_STOP_TOKEN_IDS},
    sampling,
};

/// What the engine fixed at startup and every admission checks against.
#[derive(Debug, Clone)]
pub struct IngressLimits {
    /// Token ids in the vocabulary.
    pub vocab_size: i64,
    /// The model's EOS ids, which end every request not ignoring EOS.
    pub eos_token_ids: Vec<i32>,
    /// The longest request the engine admits, prompt and output.
    pub max_model_len: i64,
    /// The longest prompt.
    pub max_req_input_len: i64,
    /// `--enable-output-logprobs`.
    pub enable_output_logprobs: bool,
    /// A disaggregated engine: every request names its peer node.
    pub requires_bootstrap: bool,
    /// The model runs its vision tower: requests may carry multimodal
    /// inputs.
    pub multimodal: bool,
    /// Grammar-constrained decoding is on (`--grammar-backend xgrammar`).
    pub grammar: bool,
    /// Which logprobs beyond the sampled tokens' the engine returns.
    pub logprobs: LogprobServability,
}

/// Most top-k alternatives a request may ask for per position.
pub const MAX_TOP_LOGPROBS: i64 = 20;

/// Which logprobs an engine returns beyond the sampled tokens'
/// (`LogprobServability`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogprobServability {
    /// The model keeps every prompt row's activations
    /// (`DeviceSpecs.supports_prompt_logprobs`).
    pub supports_prompt_logprobs: bool,
    /// `--logprob-order`; the top-k alternatives need `torch`.
    pub logprob_order: String,
    /// The engine runs pipeline stages.
    pub pipeline_parallel: bool,
    /// The engine is one role of a disaggregated deployment.
    pub disaggregated: bool,
    /// LM-head TP under attention DP: the prompt-logprob chunk loop would
    /// run the logits exchange a per-rank number of times.
    pub lm_head_tp_under_dp: bool,
    /// This engine runs the prompt rows (not the PD decode role).
    pub computes_prompt_logprobs: bool,
}

impl LogprobServability {
    /// Why the engine cannot compute this prompt's logprobs, or `None`.
    fn prompt_logprobs_refusal(
        &self,
        input_ids: &[i32],
        has_multimodal_inputs: bool,
        vocab_size: i64,
    ) -> Option<String> {
        if !self.supports_prompt_logprobs {
            return Some(
                "logprob_start_len >= 0 (prompt logprobs) is not supported by this engine: \
                 the model narrows its prefill rows, so it cannot score every prompt \
                 position. Use logprob_start_len=-1."
                    .to_owned(),
            );
        }
        if has_multimodal_inputs {
            return Some(
                "logprob_start_len >= 0 (prompt logprobs) is not supported for multimodal \
                 prompts; use logprob_start_len=-1."
                    .to_owned(),
            );
        }
        let outside = input_ids
            .iter()
            .any(|&id| id < 0 || i64::from(id) >= vocab_size);
        outside.then(|| {
            format!(
                "prompt logprobs need every prompt token id inside the vocabulary \
                 [0, {vocab_size}); input_ids contains one outside it."
            )
        })
    }

    /// Why the engine cannot return `top_logprobs_num` alternatives per
    /// position, or `None`: they are computed at the prompt rows and the
    /// token a prefill samples, so every token must be sampled in the
    /// prefill.
    fn top_logprobs_refusal(
        &self,
        top_logprobs_num: i64,
        max_new_tokens: Option<i64>,
    ) -> Option<String> {
        if !(0 < top_logprobs_num && top_logprobs_num <= MAX_TOP_LOGPROBS) {
            return Some(format!(
                "top_logprobs_num must be between 1 and {MAX_TOP_LOGPROBS}, got \
                 {top_logprobs_num}."
            ));
        }
        if max_new_tokens.unwrap_or(0) > 1 {
            return Some(
                "top_logprobs_num > 0 needs max_new_tokens <= 1: top-k logprobs are \
                 computed for the prompt and the token a prefill samples, not for decode \
                 steps."
                    .to_owned(),
            );
        }
        if self.logprob_order != "torch" {
            return Some(format!(
                "top_logprobs_num > 0 needs --logprob-order torch, not {}.",
                self.logprob_order
            ));
        }
        if self.pipeline_parallel {
            return Some(
                "top_logprobs_num > 0 is not supported with pipeline parallelism.".to_owned(),
            );
        }
        if self.disaggregated {
            return Some(
                "top_logprobs_num > 0 is not supported with disaggregated prefill and decode."
                    .to_owned(),
            );
        }
        None
    }
}

/// An admitted request.
#[derive(Debug, Clone)]
pub struct Admission {
    /// Request id.
    pub rid: String,
    /// What the scheduler admits.
    pub spec: RequestSpec,
    /// The control plane's state.
    pub state: RequestState,
    /// The resolved sampling parameters, for the data plane.
    pub sampling_params: SamplingParams,
    /// PD: where the request's peer node reaches it.
    pub bootstrap: Option<BootstrapInfo>,
    /// Its multimodal inputs, for the data plane, which owns the feature
    /// records: the wire frames of a generate request carrying only them.
    pub multimodal: Option<Vec<bytes::Bytes>>,
    /// EPD: the items whose embeddings an encode node computes (an item
    /// names its encode handshake), which must land before it is scheduled.
    pub embeddings: Vec<EmbeddingItem>,
    /// The grammar its output follows, compiled before it is scheduled.
    pub grammar: Option<crate::grammar::GrammarKey>,
}

/// The outcome of admitting one request.
#[derive(Debug, Clone)]
pub enum Admitted {
    /// Schedule it.
    Run(Box<Admission>),
    /// It is already finished with an abort; send the client its final
    /// output.
    Refused {
        /// Request id.
        rid: String,
        /// The finished state.
        state: Box<RequestState>,
    },
}

/// Admit `request`, or `None` if it carries no single request id to answer.
pub fn admit(mut request: TokenizedGenerateReqInput, limits: &IngressLimits) -> Option<Admitted> {
    let rid = match request.rid.take() {
        Some(StrOrList::One(rid)) => rid,
        _ => {
            tracing::warn!("dropping a generate request without a single request id");
            return None;
        }
    };
    let view = request_view(&request, limits);
    let bootstrap = bootstrap_info(&request);
    let embeddings = request
        .multimodal_inputs
        .as_ref()
        .map_or(Ok(Vec::new()), embedding_items);
    let multimodal = request.multimodal_inputs.take().map(|inputs| {
        // Only the data plane reads them; it decodes them as the wire does.
        let carrier = TokenizedGenerateReqInput {
            multimodal_inputs: Some(inputs),
            ..TokenizedGenerateReqInput::new(&rid, SamplingParams::default())
        };
        ts_wire::EngineRequest::Add(Box::new(carrier)).to_frames()
    });
    let mut params = request.sampling_params;
    sampling::resolve(&mut params, &rid);
    // Checked as the client sent them: the context clamp below is the
    // engine's, so `min_new_tokens` is bounded by the budget asked for, not
    // the one the context leaves (the request then finishes at its length).
    let invalid = request
        .validation_error
        .clone()
        .or_else(|| embeddings.as_ref().err().cloned())
        .or_else(|| refusal(&view, &params, limits));

    let prompt_len = request.input_ids.as_ref().map_or(0, Vec::len) as i64;
    let stop_token_ids: BTreeSet<i32> = if params.ignore_eos {
        BTreeSet::new()
    } else {
        let own = params.stop_token_ids.iter().flatten().map(|&id| id as i32);
        own.chain(limits.eos_token_ids.iter().copied()).collect()
    };
    // Clamp the budget to the context (the admitted length limit is one
    // under the model's).
    let max_req_len = limits.max_model_len - 1;
    let max_new_tokens = params
        .max_new_tokens
        .unwrap_or(1 << 30)
        .min(max_req_len - prompt_len - 1);
    params.max_new_tokens = Some(max_new_tokens);
    let mut state = RequestState::new(
        request.input_ids.clone().unwrap_or_default(),
        max_new_tokens,
        stop_token_ids,
        request.stream,
        params.stream_interval,
        request.return_logprob,
    );
    state.created_time = request.created_time;

    if invalid.is_none() && request.return_logprob {
        state.set_logprobs(LogprobRequest {
            start: view.logprob_start_len,
            top_logprobs_num: view.top_logprobs_num.max(0) as usize,
            computes_prompt_logprobs: limits.logprobs.computes_prompt_logprobs,
        });
    }
    let refusal = invalid
        .map(|reason| format!("Invalid request: {reason}"))
        .or_else(|| {
            request
                .session_params
                .as_ref()
                .and_then(|session| session.id.as_ref())
                .map(|id| format!("Invalid request: session id {id} does not exist"))
        })
        .or_else(|| {
            (state.wants_input_logprobs() && limits.logprobs.lm_head_tp_under_dp).then(|| {
                "Invalid request: prompt logprobs (logprob_start_len) are not available \
                 with --lm-head-tp-size > 1 under attention DP"
                    .to_owned()
            })
        })
        .or_else(|| too_many_stop_token_ids(&state, &params))
        .or_else(|| {
            (limits.requires_bootstrap && bootstrap.is_none()).then(|| {
                "Invalid request: a disaggregated engine needs the request's bootstrap \
                 host, port and room"
                    .to_owned()
            })
        });
    if let Some(message) = refusal {
        tracing::warn!(rid, %message, "refusing request");
        // Every refusal is the request's own error, which the frontend
        // answers as a client one (400).
        state.abort(&message, true, Some(400));
        return Some(Admitted::Refused {
            rid,
            state: Box::new(state),
        });
    }

    let spec = RequestSpec {
        request_id: rid.clone(),
        tokens: request.input_ids.unwrap_or_default(),
        max_new_tokens: max_new_tokens.clamp(0, i64::from(i32::MAX)) as i32,
        // Prompt logprobs need logits from their start on, so the admission
        // probe may not match those positions from the prefix cache.
        max_cached_prefix_tokens: if state.wants_input_logprobs() {
            state.logprob_start_len as i32
        } else {
            i32::MAX
        },
    };
    let grammar = crate::grammar::GrammarKey::of(&params);
    Some(Admitted::Run(Box::new(Admission {
        rid,
        spec,
        state,
        sampling_params: params,
        bootstrap,
        multimodal,
        embeddings: embeddings.unwrap_or_default(),
        grammar,
    })))
}

/// EPD: the items naming an encode node's handshake (`bootstrap_room`,
/// `bootstrap_host`, `bootstrap_port`), with their embeddings' rows: one per
/// token of the item's ranges (`_item_token_count`).
fn embedding_items(inputs: &ts_wire::MultimodalInputs) -> Result<Vec<EmbeddingItem>, String> {
    let mut items = Vec::new();
    for (index, item) in inputs.mm_items.iter().enumerate() {
        let Some(handshake) = &item.encode_handshake else {
            continue;
        };
        let field = |key: &str| handshake.get(&ts_wire::Value::from(key));
        let integer = |key: &str| match field(key) {
            Some(ts_wire::Value::Int(value)) => Some(*value),
            _ => None,
        };
        let malformed = || format!("item {index}'s encode handshake is malformed");
        let host = match field("bootstrap_host") {
            Some(ts_wire::Value::Str(host)) => host.clone(),
            _ => return Err(malformed()),
        };
        let port = integer("bootstrap_port")
            .and_then(|port| port.narrow::<u16>("a port").ok())
            .ok_or_else(malformed)?;
        let room = integer("bootstrap_room")
            .and_then(|room| room.narrow::<u64>("a room").ok())
            .ok_or_else(malformed)?;
        let rows = item
            .offsets
            .iter()
            .flatten()
            .map(|&(start, end)| end - start + 1)
            .sum();
        items.push(EmbeddingItem {
            index,
            host,
            port,
            room,
            rows,
        });
    }
    Ok(items)
}

/// The request's PD bootstrap fields, if it names all three.
fn bootstrap_info(request: &TokenizedGenerateReqInput) -> Option<BootstrapInfo> {
    Some(BootstrapInfo {
        host: request.bootstrap_host.clone()?,
        port: u16::try_from(request.bootstrap_port?).ok()?,
        room: request.bootstrap_room?,
    })
}

/// The request fields the checks read, taken before the sampling parameters
/// move out.
struct RequestView {
    input_len: Option<usize>,
    /// The prompt's smallest and largest token ids.
    input_id_range: Option<(i32, i32)>,
    has_multimodal_inputs: bool,
    return_logprob: bool,
    top_logprobs_num: i64,
    has_token_ids_logprob: bool,
    logprob_start_len: i64,
    /// The budget the request asked for, before it is clamped.
    max_new_tokens: Option<i64>,
    unsupported: Option<&'static str>,
}

fn request_view(request: &TokenizedGenerateReqInput, limits: &IngressLimits) -> RequestView {
    let unsupported = if request.multimodal_inputs.is_some() && !limits.multimodal {
        Some("multimodal inputs need a multimodal model (not --language-model-only)")
    } else if request.input_embeds.is_some() {
        Some("input embeddings are not supported by this engine host")
    } else if request.custom_logit_processor.is_some() {
        Some("custom logit processors are not supported by this engine host")
    } else if request.return_hidden_states {
        Some("hidden states are not supported by this engine host")
    } else {
        None
    };
    let input_ids = request.input_ids.as_deref().unwrap_or_default();
    let input_id_range = input_ids
        .iter()
        .min()
        .zip(input_ids.iter().max())
        .map(|(&min, &max)| (min, max));
    RequestView {
        input_len: request.input_ids.as_ref().map(Vec::len),
        input_id_range,
        has_multimodal_inputs: request.multimodal_inputs.is_some(),
        return_logprob: request.return_logprob,
        top_logprobs_num: request.top_logprobs_num,
        has_token_ids_logprob: request
            .token_ids_logprob
            .as_ref()
            .is_some_and(|ids| !ids.is_empty()),
        logprob_start_len: request.logprob_start_len,
        max_new_tokens: request.sampling_params.max_new_tokens,
        unsupported,
    }
}

/// Why a request cannot run, or `None`.
fn refusal(
    request: &RequestView,
    params: &SamplingParams,
    limits: &IngressLimits,
) -> Option<String> {
    if params.n != 1 {
        return Some(format!(
            "n={} is not supported on the msgpack wire; the frontend must fan out n > 1 itself",
            params.n
        ));
    }
    if let Some(reason) = request.unsupported {
        return Some(reason.to_owned());
    }
    let Some(input_len) = request.input_len else {
        return Some("a request needs prompt token ids".to_owned());
    };
    if input_len as i64 > limits.max_req_input_len {
        return Some(format!(
            "the prompt has {input_len} tokens; this engine admits at most {}",
            limits.max_req_input_len
        ));
    }
    if request.return_logprob {
        if !limits.enable_output_logprobs {
            return Some(
                "logprobs were requested but the server was started without \
                 --enable-output-logprobs"
                    .to_owned(),
            );
        }
        if request.top_logprobs_num != 0 {
            if let Some(reason) = limits
                .logprobs
                .top_logprobs_refusal(request.top_logprobs_num, request.max_new_tokens)
            {
                return Some(reason);
            }
        }
        if request.has_token_ids_logprob {
            return Some("token_ids_logprob is not supported yet".to_owned());
        }
        let start = request.logprob_start_len;
        let input_len = input_len as i64;
        if start < -1 {
            return Some(format!(
                "logprob_start_len must be -1 or >= 0, got {start}."
            ));
        }
        if start >= input_len {
            return Some(format!(
                "logprob_start_len ({start}) must be smaller than the prompt length ({input_len})."
            ));
        }
        if start >= 0 && start < input_len - 1 {
            let (min, max) = request.input_id_range.unwrap_or((0, 0));
            if let Some(reason) = limits.logprobs.prompt_logprobs_refusal(
                &[min, max],
                request.has_multimodal_inputs,
                limits.vocab_size,
            ) {
                return Some(reason);
            }
        }
    }
    let constrained = [
        &params.json_schema,
        &params.regex,
        &params.ebnf,
        &params.structural_tag,
    ];
    if !limits.grammar && constrained.iter().any(|grammar| grammar.is_some()) {
        return Some(
            "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not \
             supported when the server is launched with --grammar-backend none"
                .to_owned(),
        );
    }
    let has_stop_strings =
        matches!(&params.stop_strs, Some(StrOrList::Many(stops)) if !stops.is_empty());
    if has_stop_strings {
        return Some(
            "stop strings are not matched on the msgpack wire; the frontend matches them \
             and sends stop_token_ids only"
                .to_owned(),
        );
    }
    sampling::verify(params, limits.vocab_size).err()
}

/// More stop tokens than the sampler's stop table holds.
fn too_many_stop_token_ids(state: &RequestState, params: &SamplingParams) -> Option<String> {
    let count = state.stop_token_ids.len();
    if count <= MAX_STOP_TOKEN_IDS {
        return None;
    }
    let own: BTreeSet<i32> = params
        .stop_token_ids
        .iter()
        .flatten()
        .map(|&id| id as i32)
        .collect();
    let model = state.stop_token_ids.difference(&own).count();
    Some(format!(
        "Invalid request: {count} stop token ids exceed the limit of {MAX_STOP_TOKEN_IDS}, \
         which includes the model's {model} EOS ids; pass at most {} other stop_token_ids",
        MAX_STOP_TOKEN_IDS - model
    ))
}

#[cfg(test)]
mod tests {
    use ts_wire::{Modality, MultimodalDataItem, MultimodalInputs, Value};

    use super::*;
    use crate::request::FinishReason;

    fn limits() -> IngressLimits {
        IngressLimits {
            vocab_size: 1000,
            eos_token_ids: vec![2],
            max_model_len: 100,
            max_req_input_len: 90,
            enable_output_logprobs: false,
            requires_bootstrap: false,
            multimodal: false,
            grammar: false,
            logprobs: LogprobServability {
                supports_prompt_logprobs: true,
                logprob_order: "torch".to_owned(),
                pipeline_parallel: false,
                disaggregated: false,
                lm_head_tp_under_dp: false,
                computes_prompt_logprobs: true,
            },
        }
    }

    fn request(
        rid: &str,
        input_ids: Vec<i32>,
        params: SamplingParams,
    ) -> TokenizedGenerateReqInput {
        TokenizedGenerateReqInput {
            input_ids: Some(input_ids),
            stream: true,
            ..TokenizedGenerateReqInput::new(rid, params)
        }
    }

    fn refused_message(admitted: Admitted) -> String {
        match admitted {
            Admitted::Refused { state, .. } => match state.finished_reason {
                Some(FinishReason::Abort {
                    message,
                    status_code,
                    ..
                }) => {
                    // Every refusal is a client error.
                    assert_eq!(status_code, Some(400), "{message}");
                    message
                }
                other => panic!("expected an abort, got {other:?}"),
            },
            Admitted::Run(admission) => panic!("expected a refusal, admitted {}", admission.rid),
        }
    }

    #[test]
    fn an_admitted_request_has_its_budget_and_stop_tokens() {
        let params = SamplingParams {
            max_new_tokens: Some(8),
            stop_token_ids: Some(vec![9]),
            ..Default::default()
        };
        let Some(Admitted::Run(admission)) = admit(request("r1", vec![1, 2, 3], params), &limits())
        else {
            panic!("expected admission");
        };
        assert_eq!(admission.spec.tokens, [1, 2, 3]);
        assert_eq!(admission.spec.max_new_tokens, 8);
        assert_eq!(admission.state.stop_token_ids, BTreeSet::from([2, 9]));
        assert_eq!(admission.state.stop_criteria().token_limit, 11);
        assert!(admission.sampling_params.is_normalized);
    }

    #[test]
    fn the_budget_is_clamped_to_the_context() {
        let Some(Admitted::Run(admission)) = admit(
            request("r1", vec![0; 50], SamplingParams::default()),
            &limits(),
        ) else {
            panic!("expected admission");
        };
        // max_model_len - 1 - prompt - 1
        assert_eq!(admission.spec.max_new_tokens, 48);
    }

    #[test]
    fn min_new_tokens_is_checked_against_the_budget_asked_for() {
        // The context leaves 8 tokens of the 50 asked for: the request runs
        // to its length rather than being refused over a budget the client
        // never sent.
        let params = SamplingParams {
            max_new_tokens: Some(50),
            min_new_tokens: 20,
            ..Default::default()
        };
        let Some(Admitted::Run(admission)) = admit(request("r1", vec![0; 90], params), &limits())
        else {
            panic!("expected admission");
        };
        assert_eq!(admission.spec.max_new_tokens, 8);
        assert_eq!(admission.sampling_params.max_new_tokens, Some(8));

        let params = SamplingParams {
            max_new_tokens: Some(10),
            min_new_tokens: 20,
            ..Default::default()
        };
        assert!(
            refused_message(admit(request("r2", vec![1], params), &limits()).unwrap())
                .contains("min_new_tokens must be in (0, max_new_tokens(10)]")
        );
    }

    #[test]
    fn ignore_eos_drops_every_stop_token() {
        let params = SamplingParams {
            ignore_eos: true,
            stop_token_ids: Some(vec![9]),
            ..Default::default()
        };
        let Some(Admitted::Run(admission)) = admit(request("r1", vec![1], params), &limits())
        else {
            panic!("expected admission");
        };
        assert!(admission.state.stop_token_ids.is_empty());
    }

    #[test]
    fn invalid_requests_finish_with_the_reason() {
        let bad_top_p = SamplingParams {
            top_p: 0.0,
            ..Default::default()
        };
        assert!(
            refused_message(admit(request("a", vec![1], bad_top_p), &limits()).unwrap())
                .contains("top_p")
        );

        let with_stop = SamplingParams {
            stop: Some(StrOrList::One("x".into())),
            ..Default::default()
        };
        assert!(
            refused_message(admit(request("b", vec![1], with_stop), &limits()).unwrap())
                .contains("stop strings")
        );

        let logprobs = TokenizedGenerateReqInput {
            return_logprob: true,
            ..request("c", vec![1], SamplingParams::default())
        };
        assert!(refused_message(admit(logprobs, &limits()).unwrap())
            .contains("--enable-output-logprobs"));

        let too_long = request("d", vec![0; 91], SamplingParams::default());
        assert!(refused_message(admit(too_long, &limits()).unwrap()).contains("at most 90"));

        let fan_out = SamplingParams {
            n: 2,
            ..Default::default()
        };
        assert!(
            refused_message(admit(request("e", vec![1], fan_out), &limits()).unwrap())
                .contains("n=2")
        );

        let constrained = SamplingParams {
            regex: Some("[a-z]+".into()),
            ..Default::default()
        };
        assert!(
            refused_message(admit(request("g", vec![1], constrained), &limits()).unwrap())
                .contains("--grammar-backend none")
        );

        let many_stops = SamplingParams {
            stop_token_ids: Some((10..30).collect()),
            ..Default::default()
        };
        let message = refused_message(admit(request("f", vec![1], many_stops), &limits()).unwrap());
        assert!(
            message.contains("21 stop token ids") && message.contains("at most 15"),
            "{message}"
        );
    }

    #[test]
    fn prompt_logprobs_bound_the_cached_prefix() {
        let limits = IngressLimits {
            enable_output_logprobs: true,
            ..limits()
        };
        let scored = TokenizedGenerateReqInput {
            return_logprob: true,
            logprob_start_len: 1,
            top_logprobs_num: 3,
            ..request(
                "a",
                vec![5, 6, 7, 8],
                SamplingParams {
                    max_new_tokens: Some(1),
                    ..Default::default()
                },
            )
        };
        let Some(Admitted::Run(admission)) = admit(scored, &limits) else {
            panic!("expected admission");
        };
        assert_eq!(admission.spec.max_cached_prefix_tokens, 1);
        let state = &admission.state;
        assert!(state.wants_input_logprobs());
        assert_eq!(state.input_logprob_rows.as_ref().map(Vec::len), Some(2));
        assert_eq!(state.input_top_logprob_rows.as_ref().map(Vec::len), Some(2));
        assert!(state.output_top_logprobs.is_some());

        // The default start scores only the last position: nothing to compute.
        let sampled_only = TokenizedGenerateReqInput {
            return_logprob: true,
            ..request("b", vec![5, 6, 7], SamplingParams::default())
        };
        let Some(Admitted::Run(admission)) = admit(sampled_only, &limits) else {
            panic!("expected admission");
        };
        assert_eq!(admission.spec.max_cached_prefix_tokens, i32::MAX);
        assert!(!admission.state.wants_input_logprobs());
        assert!(admission.state.returns_input_logprobs);
    }

    #[test]
    fn logprobs_the_engine_cannot_return_are_refused() {
        let limits = IngressLimits {
            enable_output_logprobs: true,
            ..limits()
        };
        let top_k = |rid: &str, k: i64, max_new_tokens: i64| TokenizedGenerateReqInput {
            return_logprob: true,
            top_logprobs_num: k,
            ..request(
                rid,
                vec![1, 2],
                SamplingParams {
                    max_new_tokens: Some(max_new_tokens),
                    ..Default::default()
                },
            )
        };
        assert!(refused_message(admit(top_k("a", 21, 1), &limits).unwrap())
            .contains("between 1 and 20"));
        assert!(refused_message(admit(top_k("b", 2, 2), &limits).unwrap())
            .contains("max_new_tokens <= 1"));
        let disaggregated = IngressLimits {
            logprobs: LogprobServability {
                disaggregated: true,
                ..limits.logprobs.clone()
            },
            ..limits.clone()
        };
        assert!(
            refused_message(admit(top_k("c", 2, 1), &disaggregated).unwrap())
                .contains("disaggregated")
        );

        let prompt = |rid: &str, ids: Vec<i32>| TokenizedGenerateReqInput {
            return_logprob: true,
            logprob_start_len: 0,
            ..request(rid, ids, SamplingParams::default())
        };
        assert!(
            refused_message(admit(prompt("d", vec![1, 5000]), &limits).unwrap())
                .contains("inside the vocabulary")
        );
        let narrowing = IngressLimits {
            logprobs: LogprobServability {
                supports_prompt_logprobs: false,
                ..limits.logprobs.clone()
            },
            ..limits.clone()
        };
        assert!(
            refused_message(admit(prompt("e", vec![1, 2]), &narrowing).unwrap())
                .contains("narrows its prefill rows")
        );
        let lm_head_tp = IngressLimits {
            logprobs: LogprobServability {
                lm_head_tp_under_dp: true,
                ..limits.logprobs.clone()
            },
            ..limits.clone()
        };
        assert!(
            refused_message(admit(prompt("f", vec![1, 2]), &lm_head_tp).unwrap())
                .contains("--lm-head-tp-size")
        );
    }

    #[test]
    fn sessions_and_requests_without_a_bootstrap_are_refused() {
        let session = TokenizedGenerateReqInput {
            session_params: Some(ts_wire::SessionParams {
                id: Some("s-1".to_owned()),
                ..Default::default()
            }),
            ..request("a", vec![1], SamplingParams::default())
        };
        assert!(refused_message(admit(session, &limits()).unwrap()).contains("session id s-1"));

        let disaggregated = IngressLimits {
            requires_bootstrap: true,
            ..limits()
        };
        let plain = request("b", vec![1], SamplingParams::default());
        assert!(refused_message(admit(plain, &disaggregated).unwrap()).contains("bootstrap"));
    }

    #[test]
    fn too_many_stop_tokens_are_a_client_error() {
        let many_stops = SamplingParams {
            stop_token_ids: Some((10..30).collect()),
            ..Default::default()
        };
        let Some(Admitted::Refused { state, .. }) =
            admit(request("a", vec![1], many_stops), &limits())
        else {
            panic!("expected a refusal");
        };
        let details = state.finished_reason.expect("finished").details();
        assert_eq!(
            details
                .0
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>(),
            ["type", "message", "err_type", "status_code"]
        );
        assert_eq!(details.0[3].1, Value::from(400));
    }

    #[test]
    fn a_request_without_a_single_id_is_dropped() {
        let mut no_id = request("x", vec![1], SamplingParams::default());
        no_id.rid = None;
        assert!(admit(no_id, &limits()).is_none());
    }

    fn image(
        offsets: Vec<(i64, i64)>,
        handshake: Option<Vec<(&str, Value)>>,
    ) -> MultimodalDataItem {
        MultimodalDataItem {
            modality: Modality::Image,
            hash: None,
            pad_value: None,
            offsets: Some(offsets),
            feature: None,
            feature_shm: None,
            model_specific_data: ts_wire::Dict::default(),
            encoded: None,
            encoded_deepstack: None,
            encode_handshake: handshake.map(|fields| {
                ts_wire::Dict(
                    fields
                        .into_iter()
                        .map(|(key, value)| (Value::from(key), value))
                        .collect(),
                )
            }),
        }
    }

    fn with_images(items: Vec<MultimodalDataItem>) -> TokenizedGenerateReqInput {
        TokenizedGenerateReqInput {
            multimodal_inputs: Some(MultimodalInputs {
                mm_items: items,
                im_token_id: None,
                video_token_id: None,
                mrope_positions: None,
                mrope_position_delta: None,
                mrope_position_delta_scalar: None,
                mrope_position_delta_repeated_cache: None,
            }),
            ..request("r1", vec![0; 16], SamplingParams::default())
        }
    }

    #[test]
    fn epd_items_name_their_encode_node_and_rows() {
        let limits = IngressLimits {
            multimodal: true,
            ..limits()
        };
        // A room past i64 (rooms are random 64-bit), two ranges of one
        // image, and an item the prefill node encodes itself.
        let room = (1u64 << 63) + 5;
        let request = with_images(vec![
            image(vec![(0, 1)], None),
            image(
                vec![(2, 4), (8, 9)],
                Some(vec![
                    ("bootstrap_host", Value::from("10.0.0.3")),
                    ("bootstrap_port", Value::from(8998)),
                    (
                        "bootstrap_room",
                        Value::Int(ts_wire::msgpack::Integer::Unsigned(room)),
                    ),
                ]),
            ),
        ]);
        let Some(Admitted::Run(admission)) = admit(request, &limits) else {
            panic!("expected admission");
        };
        assert_eq!(
            admission.embeddings,
            [EmbeddingItem {
                index: 1,
                host: "10.0.0.3".to_owned(),
                port: 8998,
                room,
                rows: 5,
            }]
        );
        // A handshake without its port cannot be received: refused.
        let request = with_images(vec![image(
            vec![(0, 1)],
            Some(vec![
                ("bootstrap_host", Value::from("10.0.0.3")),
                ("bootstrap_room", Value::from(7)),
            ]),
        )]);
        assert_eq!(
            refused_message(admit(request, &limits).expect("answered")),
            "Invalid request: item 0's encode handshake is malformed"
        );
    }
}
