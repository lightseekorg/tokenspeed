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

//! The requests an engine serves itself at startup, before a frontend is
//! attached ([`crate::event_loop::EventLoop::warm_up`]).

use ts_wire::{EngineRequest, SamplingParams, TokenizedGenerateReqInput};

/// One startup phase: `batch` requests of `prompt_len` tokens, served
/// together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase {
    /// Requests served together.
    pub batch: usize,
    /// Prompt tokens per request.
    pub prompt_len: usize,
}

/// The engine limits the startup requests are sized from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most requests a forward runs.
    pub max_batch_size: usize,
    /// The most prompt tokens a forward runs.
    pub chunk_tokens: usize,
    /// The longest prompt ingress admits.
    pub max_prompt_len: usize,
    /// The longest request, prompt and output.
    pub max_model_len: usize,
    /// Tokens the cache holds.
    pub token_capacity: usize,
    /// Tokens per cache page.
    pub page_tokens: usize,
    /// Tokens a decode round runs per request: the verify width with
    /// speculative decoding, else 1.
    pub decode_width: usize,
}

/// Full-batch decode rounds each startup request's output covers.
const DECODE_ROUNDS: usize = 2;

impl Limits {
    /// Each startup request's output: [`DECODE_ROUNDS`] decode rounds, a
    /// partly accepted round and the draft round still pending.
    pub fn output_tokens(&self) -> usize {
        (DECODE_ROUNDS + 2) * self.decode_width + 1
    }

    /// The startup phases, each a shape serving reaches: a one-token
    /// request; requests of a thirty-second of a prefill chunk, a sixteenth,
    /// one token more, and three sixteenths (the partial forwards that short
    /// prompts and the last chunk of a long one run, aligned and not); a
    /// request of a whole chunk; then a full batch whose prompts cross one
    /// chunk boundary, so its prefill rounds also mix a finishing request
    /// with a starting one before every request decodes.
    /// Prompts are cut to what ingress admits and so the whole batch, with
    /// its output and the pages both round up to, fits the cache at once;
    /// a phase that cannot fit is left out.
    pub fn phases(&self) -> Vec<Phase> {
        let output = self.output_tokens();
        let longest = self
            .max_prompt_len
            .min(self.max_model_len.saturating_sub(output));
        let fits = |batch: usize| {
            (self.token_capacity / batch.max(1)).saturating_sub(output + 2 * self.page_tokens)
        };
        let mut phases: Vec<Phase> = [
            (1, 1),
            (1, self.chunk_tokens / 32),
            (1, self.chunk_tokens / 16),
            (1, self.chunk_tokens / 16 + 1),
            (1, self.chunk_tokens * 3 / 16),
            (1, self.chunk_tokens),
            (self.max_batch_size, self.chunk_tokens + 1),
        ]
        .into_iter()
        .map(|(batch, prompt_len)| Phase {
            batch,
            prompt_len: prompt_len.min(longest).min(fits(batch)),
        })
        .filter(|phase| phase.batch > 0 && phase.prompt_len > 0)
        .collect();
        phases.dedup();
        phases
    }
}

/// The id prefix of every startup request.
pub const RID_PREFIX: &str = "__startup_";

/// Each phase's requests, generating `output_tokens` each. Every prompt in
/// a phase is distinct, so prefix sharing cannot skip the prefill a phase
/// is there to run. Sampling is greedy as a request normalized by a
/// frontend is: temperature 1 with a top-k of 1.
pub fn requests(phases: &[Phase], output_tokens: usize) -> Vec<Vec<EngineRequest>> {
    let output_tokens = i64::try_from(output_tokens).expect("a startup output fits i64");
    phases
        .iter()
        .enumerate()
        .map(|(phase, &Phase { batch, prompt_len })| {
            (0..batch)
                .map(|row| {
                    let params = SamplingParams {
                        max_new_tokens: Some(output_tokens),
                        temperature: 1.0,
                        top_k: 1,
                        ignore_eos: true,
                        ..Default::default()
                    };
                    let token = i32::try_from(row + 1).expect("a startup batch fits a token id");
                    EngineRequest::Add(Box::new(TokenizedGenerateReqInput {
                        input_ids: Some(vec![token; prompt_len]),
                        ..TokenizedGenerateReqInput::new(
                            format!("{RID_PREFIX}{phase}_{row}"),
                            params,
                        )
                    }))
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(token_capacity: usize) -> Limits {
        Limits {
            max_batch_size: 48,
            chunk_tokens: 8192,
            max_prompt_len: 1 << 20,
            max_model_len: 1 << 20,
            token_capacity,
            page_tokens: 128,
            decode_width: 4,
        }
    }

    fn phase(batch: usize, prompt_len: usize) -> Phase {
        Phase { batch, prompt_len }
    }

    #[test]
    fn a_roomy_cache_runs_every_shape_at_its_limit() {
        assert_eq!(
            limits(897_024).phases(),
            [
                phase(1, 1),
                phase(1, 256),
                phase(1, 512),
                phase(1, 513),
                phase(1, 1536),
                phase(1, 8192),
                phase(48, 8193)
            ]
        );
    }

    #[test]
    fn the_full_batch_is_cut_to_what_the_cache_holds_at_once() {
        // 302592 / 48 = 6304 tokens per request, less 17 of output and two
        // pages.
        assert_eq!(limits(302_592).phases()[6], phase(48, 6031));
        assert_eq!(
            limits(48 * 200).phases(),
            [
                phase(1, 1),
                phase(1, 256),
                phase(1, 512),
                phase(1, 513),
                phase(1, 1536),
                phase(1, 8192)
            ]
        );
    }

    #[test]
    fn a_tiny_chunk_runs_each_shape_once() {
        let limits = Limits {
            chunk_tokens: 4,
            ..limits(1 << 30)
        };
        assert_eq!(limits.phases(), [phase(1, 1), phase(1, 4), phase(48, 5)]);
    }

    #[test]
    fn prompts_stay_within_what_ingress_admits() {
        let limits = Limits {
            max_prompt_len: 4000,
            max_model_len: 4010,
            ..limits(1 << 30)
        };
        assert_eq!(
            limits.phases(),
            [
                phase(1, 1),
                phase(1, 256),
                phase(1, 512),
                phase(1, 513),
                phase(1, 1536),
                phase(1, 3993),
                phase(48, 3993)
            ]
        );
    }
}
