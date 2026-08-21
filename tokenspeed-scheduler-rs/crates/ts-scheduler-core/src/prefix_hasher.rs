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

//! SHA-256 prefix-hash chain over token pages.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/prefix/prefix_hasher.h`.
//! Byte-for-byte compatible framing (OpenSSL `SHA256_*` → `sha2` crate):
//!
//! ```text
//! [prior_len u32le][prior bytes][token_count u32le][tokens u32le...]
//! [extra_count u32le][per key: len u32le + key bytes...]
//! ```
//!
//! The whole input is prefix-framed so every section is self-delimiting and no
//! two distinct `(prior, tokens, extra_keys)` triples hash the same byte stream.

use sha2::{Digest, Sha256};

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Append each byte as two lowercase hex characters.
pub fn append_hex_bytes(out: &mut String, bytes: &[u8]) {
    out.reserve(bytes.len() * 2);
    for b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
}

/// Decode a hex string back into raw bytes (inverse of [`digest_to_hex`]).
pub fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.bytes();
    while let (Some(hi), Some(lo)) = (iter.next(), iter.next()) {
        let hi = hex_digit(hi);
        let lo = hex_digit(lo);
        bytes.push((hi << 4) | lo);
    }
    bytes
}

fn hex_digit(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("invalid hex digit: {:?}", c as char),
    }
}

/// Encode a digest as a lowercase hex string.
pub fn digest_to_hex(digest: &[u8]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    append_hex_bytes(&mut out, digest);
    out
}

/// Absorb a `u32` into the hash as four little-endian bytes.
fn sha256_update_u32_le(ctx: &mut Sha256, v: u32) {
    ctx.update([v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8]);
}

/// Hash one prefix page: `prior` is the previous page's hash (empty for the
/// first page), `tokens` the page tokens, `extra_keys` optional per-page
/// distinguishing keys (e.g. LoRA name, cache salt).
pub fn hash_prefix_page(tokens: &[i32], prior_hash: &str, extra_keys: &[&str]) -> String {
    let mut ctx = Sha256::new();

    let prior_bytes = hex_to_bytes(prior_hash);
    sha256_update_u32_le(&mut ctx, prior_bytes.len() as u32);
    if !prior_bytes.is_empty() {
        ctx.update(&prior_bytes);
    }

    sha256_update_u32_le(&mut ctx, tokens.len() as u32);
    for token in tokens {
        sha256_update_u32_le(&mut ctx, *token as u32);
    }

    // extra_keys is the terminal block, so an empty list can be skipped without
    // ambiguity (a non-empty list always writes a count >= 1 first).
    if !extra_keys.is_empty() {
        sha256_update_u32_le(&mut ctx, extra_keys.len() as u32);
        for key in extra_keys {
            sha256_update_u32_le(&mut ctx, key.len() as u32);
            ctx.update(key.as_bytes());
        }
    }

    let digest = ctx.finalize();
    digest_to_hex(&digest)
}

/// Compute the hash chain over `prefix_pages`, seeded by `prior` (the hash of
/// the page before the first one; usually empty).
pub fn compute_prefix_hashes(
    prefix_pages: &[&[i32]],
    prior: &str,
    extra_keys_per_page: &[&[&str]],
) -> Vec<String> {
    let mut hashes = Vec::with_capacity(prefix_pages.len());
    let mut current_prior = prior.to_string();
    for (i, page) in prefix_pages.iter().enumerate() {
        let extra = extra_keys_per_page.get(i).copied().unwrap_or(&[]);
        let hash = hash_prefix_page(page, &current_prior, extra);
        hashes.push(hash);
        current_prior = hashes.last().expect("just pushed").clone();
    }
    hashes
}

/// Continue an existing hash chain and return only `[first_page, past_end_page)`.
pub fn advance_prefix_hashes(
    prefix_pages: &[&[i32]],
    first_page: usize,
    prior: &str,
    past_end_page: usize,
) -> Vec<String> {
    assert!(first_page < past_end_page, "hash range must be non-empty");
    assert!(
        past_end_page <= prefix_pages.len(),
        "hash range exceeds the available full pages"
    );
    compute_prefix_hashes(&prefix_pages[first_page..past_end_page], prior, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors computed with an independent SHA-256 implementation
    // (.NET System.Security.Cryptography), which implements the same standard
    // as OpenSSL used by the C++ scheduler.

    #[test]
    fn empty_page_with_empty_prior() {
        // bytes: [0u32 prior_len][0u32 count]
        assert_eq!(
            hash_prefix_page(&[], "", &[]),
            "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
        );
    }

    #[test]
    fn tokens_with_empty_prior() {
        // bytes: [0u32][3u32][1,2,3 as u32le]
        assert_eq!(
            hash_prefix_page(&[1, 2, 3], "", &[]),
            "a452f93a8b397e453162a0ee3b3408c00b5ddb4587f936b4ce2b66659feaedaf"
        );
    }

    #[test]
    fn chained_page_absorbs_prior_digest() {
        // prior = hex-decoded digest of the previous page, prefixed with its
        // 32-byte length.
        let prior = "a452f93a8b397e453162a0ee3b3408c00b5ddb4587f936b4ce2b66659feaedaf";
        assert_eq!(
            hash_prefix_page(&[4, 5], prior, &[]),
            "37a58e214fcc09dceb07aa0f4ec9b1f8e644e9b5c855c8f0725d37749f9c4386"
        );
    }

    #[test]
    fn extra_keys_are_framed_after_tokens() {
        // tokens [1]; extras ["loraA", "b"]
        assert_eq!(
            hash_prefix_page(&[1], "", &["loraA", "b"]),
            "0139d024eb5c28ad07c2b3dfc4b05aca2fa8d80155b0ffe853cf7e19bea47130"
        );
    }

    #[test]
    fn negative_tokens_are_two_complement_u32() {
        // tokens [-1] == u32 0xFFFFFFFF; bytes: [0u32][1u32][0xFFFFFFFF]
        let mut ctx = Sha256::new();
        ctx.update([0u8, 0, 0, 0]); // prior length 0
        ctx.update([1u8, 0, 0, 0]); // token count 1
        ctx.update([0xFF, 0xFF, 0xFF, 0xFF]); // token -1 as u32 LE
        let expected = digest_to_hex(&ctx.finalize());
        assert_eq!(hash_prefix_page(&[-1], "", &[]), expected);
    }

    #[test]
    fn compute_prefix_hashes_chains_pages() {
        let pages: Vec<&[i32]> = vec![&[1, 2, 3], &[4, 5]];
        let hashes = compute_prefix_hashes(&pages, "", &[]);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], hash_prefix_page(&[1, 2, 3], "", &[]));
        assert_eq!(hashes[1], hash_prefix_page(&[4, 5], &hashes[0], &[]));
    }

    #[test]
    fn advance_prefix_hashes_returns_subrange() {
        let pages: Vec<&[i32]> = vec![&[1], &[2], &[3]];
        let all = compute_prefix_hashes(&pages, "", &[]);
        let tail = advance_prefix_hashes(&pages, 1, &all[0], 3);
        assert_eq!(tail, vec![all[1].clone(), all[2].clone()]);
    }

    #[test]
    fn hex_round_trips() {
        let digest = hash_prefix_page(&[7, 8], "", &[]);
        assert_eq!(digest_to_hex(&hex_to_bytes(&digest)), digest);
        assert_eq!(digest.len(), 64);
    }

    #[test]
    #[should_panic(expected = "hash range must be non-empty")]
    fn advance_prefix_hashes_rejects_empty_range() {
        advance_prefix_hashes(&[&[1]], 0, "", 0);
    }
}
