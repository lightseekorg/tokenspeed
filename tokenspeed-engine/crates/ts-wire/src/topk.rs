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
    row: &mut [u8],
    token: i32,
    logprob: f64,
    k: usize,
    vocab_size: i64,
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
    let mut head_ids = [0i32; 128];
    for j in 0..k {
        let id = int(2 + j);
        let value = float(2 + k + j);
        if id < 0 || i64::from(id) >= vocab_size {
            return Err(invalid("head IDs must be distinct vocabulary IDs"));
        }
        head_ids[j] = id;
        if !value.is_finite() || value > 0.0 {
            return Err(invalid("head logprobs must be finite and nonpositive"));
        }
        if id == token && (f64::from(value) - logprob).abs() > 1e-6 {
            return Err(invalid("head and sampled logprobs disagree"));
        }
    }
    head_ids[..k].sort_unstable();
    if head_ids[..k].windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("head IDs must be distinct vocabulary IDs"));
    }
    if tail > 0.0 {
        row[4..8].copy_from_slice(&0f32.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(token: i32, tail: f32, head: &[(i32, f32)]) -> Vec<u8> {
        [token, tail.to_bits() as i32]
            .into_iter()
            .chain(head.iter().map(|&(id, _)| id))
            .chain(head.iter().map(|&(_, value)| value.to_bits() as i32))
            .flat_map(i32::to_le_bytes)
            .collect()
    }

    fn valid() -> Vec<u8> {
        row(3, -1.0, &[(3, -0.5), (4, -2.0)])
    }

    #[test]
    fn valid_rows_preserve_bytes_and_signed_zero() {
        for tail in [-1.0, -0.0, f32::NEG_INFINITY] {
            let mut packed = row(3, tail, &[(3, -0.0), (4, -200.0)]);
            let original = packed.clone();
            validate_row(&mut packed, 3, -0.0, 2, 8).unwrap();
            assert_eq!(packed, original);
        }
        let mut packed = valid();
        validate_row(&mut packed, 3, -0.5 + 0.5e-6, 2, 8).unwrap();
        assert!(validate_row(&mut packed, 3, -0.5 + 2e-6, 2, 8).is_err());
    }

    #[test]
    fn malformed_rows_and_invalid_ids_are_refused() {
        for k in [0, 129] {
            assert!(validate_row(&mut valid(), 3, -0.5, k, 256).is_err());
        }
        for len in [0, row_bytes(2) - 1, row_bytes(2) + 1] {
            assert!(validate_row(&mut vec![0; len], 3, -0.5, 2, 8).is_err());
        }
        assert!(validate_row(&mut valid(), 4, -0.5, 2, 8).is_err());
        for token in [-1, 8] {
            let mut packed = row(token, -1.0, &[(3, -0.5), (4, -2.0)]);
            assert!(validate_row(&mut packed, token, -0.5, 2, 8).is_err());
        }
        for id in [-1, 3, 8] {
            let mut packed = row(3, -1.0, &[(3, -0.5), (id, -2.0)]);
            assert!(validate_row(&mut packed, 3, -0.5, 2, 8).is_err());
        }
    }

    #[test]
    fn head_validation_handles_unsorted_ids_and_a_distant_duplicate_at_max_k() {
        let mut head: Vec<_> = (0..128).rev().map(|id| (id, -0.5)).collect();
        let original = row(17, -1.0, &head);
        let mut packed = original.clone();
        validate_row(&mut packed, 17, -0.5, 128, 256).unwrap();
        assert_eq!(packed, original);
        head[127].0 = head[0].0;
        assert!(validate_row(&mut row(17, -1.0, &head), 17, -0.5, 128, 256).is_err());
    }

    #[test]
    fn invalid_head_sample_and_tail_values_are_refused() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.1] {
            assert!(validate_row(&mut valid(), 3, value, 2, 8).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.1] {
            let mut packed = row(3, -1.0, &[(3, -0.5), (4, value)]);
            assert!(validate_row(&mut packed, 3, -0.5, 2, 8).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, 1e-3] {
            let mut packed = row(3, value, &[(3, -0.5), (4, -2.0)]);
            assert!(validate_row(&mut packed, 3, -0.5, 2, 8).is_err());
        }
    }

    #[test]
    fn tiny_positive_tail_roundoff_is_clamped_to_positive_zero_bits() {
        let mut packed = row(3, 1e-5, &[(3, -0.5), (4, -2.0)]);
        let original = packed.clone();
        validate_row(&mut packed, 3, -0.5, 2, 8).unwrap();
        assert_eq!(&packed[4..8], &0f32.to_le_bytes());
        assert_eq!(&packed[..4], &original[..4]);
        assert_eq!(&packed[8..], &original[8..]);
    }

    #[test]
    fn full_vocabulary_head_requires_an_empty_tail() {
        for tail in [-1.0, -0.0, 0.0] {
            let mut packed = row(0, tail, &[(0, -0.5), (1, -2.0)]);
            assert!(validate_row(&mut packed, 0, -0.5, 2, 2).is_err());
        }
        let mut packed = row(0, f32::NEG_INFINITY, &[(0, -0.5), (1, -2.0)]);
        validate_row(&mut packed, 0, -0.5, 2, 2).unwrap();
    }
}
