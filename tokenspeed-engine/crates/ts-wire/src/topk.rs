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

//! Binary full-vocabulary top-k rows shared by engine, PD and frontend.

use crate::error::{Error, Result};

/// Bytes per row: sampled ID, tail FP32 bits, K IDs, K FP32 logprobs.
pub fn row_bytes(k: usize) -> usize {
    (2 + 2 * k) * 4
}

/// Validate an accepted output row and clamp only tiny positive tail roundoff.
/// The row is little endian and must match the emitted token and logprob.
pub fn validate_row(
    row: &mut [u8], token: i32, logprob: f64, k: usize, vocab_size: i64,
) -> Result<()> {
    let invalid = |reason: &str| Error::Invalid(format!("full-vocabulary top-k: {reason}"));
    if !(1..=128).contains(&k) || row.len() != row_bytes(k) {
        return Err(invalid("invalid row width"));
    }
    let int = |index: usize| i32::from_le_bytes(row[index * 4..index * 4 + 4].try_into().unwrap());
    let float = |index: usize| f32::from_bits(int(index) as u32);
    if int(0) != token || token < 0 || i64::from(token) >= vocab_size {
        return Err(invalid("sampled ID does not match the emitted token"));
    }
    if !logprob.is_finite() || logprob > 0.0 {
        return Err(invalid("invalid sampled logprob"));
    }
    let tail = float(1);
    if tail.is_nan() || tail > 1e-4 || (k as i64 == vocab_size && tail != f32::NEG_INFINITY) {
        return Err(invalid("invalid tail log mass"));
    }
    for j in 0..k {
        let id = int(2 + j);
        let value = float(2 + k + j);
        if id < 0 || i64::from(id) >= vocab_size || (0..j).any(|i| int(2 + i) == id) {
            return Err(invalid("head IDs must be distinct vocabulary IDs"));
        }
        if !value.is_finite() || value > 0.0 {
            return Err(invalid("head logprobs must be finite and nonpositive"));
        }
        if id == token && (f64::from(value) - logprob).abs() > 1e-6 {
            return Err(invalid("head and sampled logprobs disagree"));
        }
    }
    if tail > 0.0 {
        row[4..8].copy_from_slice(&0f32.to_le_bytes());
    }
    Ok(())
}
