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

//! Sampling parameters as the engine's ingress resolves them
//! (`sampling/sampling_params.py`).
//!
//! The Python class normalizes on construction and on decode
//! (`__post_init__`), then the msgpack ingress derives a missing seed and
//! normalizes the stop strings. [`resolve`] does all three; [`verify`] is the
//! range check. A fixture written by the Python implementation pins both.

use ts_wire::{SamplingParams, StrOrList};

/// Temperatures below this are greedy.
pub const SAMPLING_EPS: f64 = 1e-6;
/// `top_k` once disabled (the API's `-1`).
pub const TOP_K_DISABLED: i64 = 1 << 30;
/// Exclusive upper bound of an enabled `top_k`, set by the fused kernel.
pub const TOP_K_FUSED_MAX: i64 = 128;

/// Resolve parameters as received for request `rid`: the class's own
/// normalization, a seed derived from the request id when none was given, and
/// the stop strings as a list.
pub fn resolve(params: &mut SamplingParams, rid: &str) {
    post_init(params);
    if params.seed.is_none() {
        params.seed = Some(i64::from(crc32fast::hash(rid.as_bytes())));
    }
    normalize(params);
}

/// `__post_init__`: skipped once the sender already normalized.
fn post_init(params: &mut SamplingParams) {
    if params.is_normalized {
        return;
    }
    params.stop_strs = params.stop.clone();
    params.stop_token_ids = match params.stop_token_ids.take() {
        Some(mut ids) if !ids.is_empty() => {
            // A Python set.
            ids.sort_unstable();
            ids.dedup();
            Some(ids)
        }
        _ => None,
    };
    if params.temperature < SAMPLING_EPS {
        params.temperature = 1.0;
        params.top_k = 1;
    }
    if params.top_k == 1 {
        // Greedy ignores randomness; a pinned seed keeps identical greedy
        // requests identical where the sampler breaks exact ties randomly.
        params.seed = Some(0);
    }
    if params.top_k == -1 {
        params.top_k = TOP_K_DISABLED;
    }
}

/// `normalize(tokenizer=None)`: without a tokenizer a stop string's length
/// bound is its character count.
fn normalize(params: &mut SamplingParams) {
    let stops = match params.stop_strs.take() {
        None => Vec::new(),
        Some(StrOrList::One(stop)) => vec![stop],
        Some(StrOrList::Many(stops)) => stops,
    };
    params.stop_str_max_len = stops
        .iter()
        .map(|stop| stop.chars().count() as i64)
        .max()
        .unwrap_or(0);
    params.stop_strs = Some(StrOrList::Many(stops));
    params.is_normalized = true;
}

/// `verify(vocab_size)`: the reason the parameters are invalid, if they are.
pub fn verify(params: &SamplingParams, vocab_size: i64) -> Result<(), String> {
    let p = params;
    if p.temperature < 0.0 {
        return Err(format!(
            "temperature must be non-negative, got {}.",
            p.temperature
        ));
    }
    if !(p.top_p > 0.0 && p.top_p <= 1.0) {
        return Err(format!("top_p must be in (0, 1], got {}.", p.top_p));
    }
    if !(0.0..=1.0).contains(&p.min_p) {
        return Err(format!("min_p must be in [0, 1], got {}.", p.min_p));
    }
    if p.top_k < -1 || p.top_k == 0 {
        return Err(format!(
            "top_k must be -1 (disable), or at least 1, got {}.",
            p.top_k
        ));
    }
    if p.top_k != TOP_K_DISABLED && p.top_k >= TOP_K_FUSED_MAX {
        return Err(format!(
            "top_k must be < {TOP_K_FUSED_MAX} (fused kernel limit) or -1 (disable), got {}.",
            p.top_k
        ));
    }
    if !(-2.0..=2.0).contains(&p.frequency_penalty) {
        return Err(format!(
            "frequency_penalty must be in [-2, 2], got {}.",
            p.frequency_penalty
        ));
    }
    if !(-2.0..=2.0).contains(&p.presence_penalty) {
        return Err(format!(
            "presence_penalty must be in [-2, 2], got {}.",
            p.presence_penalty
        ));
    }
    if !(p.repetition_penalty > 0.0 && p.repetition_penalty <= 2.0) {
        return Err(format!(
            "repetition_penalty must be in (0, 2], got {}.",
            p.repetition_penalty
        ));
    }
    if p.min_new_tokens < 0 {
        return Err(format!(
            "min_new_tokens must be in (0, max_new_tokens], got {}.",
            p.min_new_tokens
        ));
    }
    if let Some(max_new_tokens) = p.max_new_tokens {
        if max_new_tokens < 0 {
            return Err(format!(
                "max_new_tokens must be at least 0, got {max_new_tokens}."
            ));
        }
        if p.min_new_tokens > max_new_tokens {
            return Err(format!(
                "min_new_tokens must be in (0, max_new_tokens({max_new_tokens})], got {}.",
                p.min_new_tokens
            ));
        }
    }
    if let Some(bias) = &p.logit_bias {
        for (token_id, _) in &bias.0 {
            let in_vocab = token_id
                .parse::<i64>()
                .is_ok_and(|id| (0..vocab_size).contains(&id));
            if !in_vocab {
                return Err(format!(
                    "logit_bias must has keys in [0, {}], got {token_id}.",
                    vocab_size - 1
                ));
            }
        }
    }
    if p.logprobs.is_some_and(|logprobs| logprobs != 0) {
        return Err(format!(
            "logprobs={} is not supported; use logprobs=0 (the sampled token's logprob).",
            p.logprobs.unwrap_or_default()
        ));
    }
    let grammars = [&p.json_schema, &p.regex, &p.ebnf];
    if grammars.iter().filter(|g| g.is_some()).count() > 1 {
        return Err("Only one of regex, json_schema, or ebnf can be set.".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_collapses_temperature_and_pins_the_seed() {
        let mut params = SamplingParams {
            temperature: 0.0,
            seed: Some(5),
            ..Default::default()
        };
        resolve(&mut params, "r");
        assert_eq!(
            (params.temperature, params.top_k, params.seed),
            (1.0, 1, Some(0))
        );
        assert!(params.is_normalized);
    }

    #[test]
    fn a_missing_seed_comes_from_the_request_id() {
        let mut params = SamplingParams {
            temperature: 0.5,
            ..Default::default()
        };
        resolve(&mut params, "r-text");
        // zlib.crc32(b"r-text")
        assert_eq!(params.seed, Some(3568737799));
    }

    #[test]
    fn stop_string_bounds_count_characters() {
        let mut params = SamplingParams {
            stop: Some(StrOrList::One("çdé".to_owned())),
            ..Default::default()
        };
        resolve(&mut params, "r");
        assert_eq!(params.stop_str_max_len, 3);
        assert_eq!(
            params.stop_strs,
            Some(StrOrList::Many(vec!["çdé".to_owned()]))
        );
    }
}
