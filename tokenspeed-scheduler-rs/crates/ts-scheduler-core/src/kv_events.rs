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

//! KV cache events emitted for PD / external consumers.
//!
//! Ported from `tokenspeed-scheduler/csrc/scheduler/kv_cache_events.{h,cpp}`.
//! `hash_kv_block` reproduces the C++ FNV-1a framing byte-for-byte (used by the
//! wire protocol, so it must not change without a coordinated runtime change).

/// Kind of a KV cache mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheEventKind {
    BlockStored,
    BlockRemoved,
}

/// A block stored in the KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBlockStoredEvent {
    pub block_hashes: Vec<u64>,
    pub parent_block_hash: Option<u64>,
    pub token_ids: Vec<i32>,
    pub block_size: i32,
}

/// A block removed from the KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBlockRemovedEvent {
    pub block_hashes: Vec<u64>,
}

/// Any KV cache event drained from the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheEvent {
    BlockStored(KvBlockStoredEvent),
    BlockRemoved(KvBlockRemovedEvent),
}

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

fn mix_byte(hash: &mut u64, byte: u8) {
    *hash ^= u64::from(byte);
    *hash = hash.wrapping_mul(FNV_PRIME);
}

fn mix_u64(hash: &mut u64, value: u64) {
    for i in 0..8 {
        mix_byte(hash, ((value >> (i * 8)) & 0xff) as u8);
    }
}

fn mix_i32(hash: &mut u64, value: i32) {
    let raw = value as u32;
    for i in 0..4 {
        mix_byte(hash, ((raw >> (i * 8)) & 0xff) as u8);
    }
}

/// FNV-1a hash of a KV block: `[has_parent][parent u64 LE bytes][count u64][tokens]`.
pub fn hash_kv_block(token_ids: &[i32], parent_hash: Option<u64>) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    mix_byte(&mut hash, if parent_hash.is_some() { 1 } else { 0 });
    if let Some(parent) = parent_hash {
        mix_u64(&mut hash, parent);
    }
    mix_u64(&mut hash, token_ids.len() as u64);
    for token_id in token_ids {
        mix_i32(&mut hash, *token_id);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_kv_block_reproduces_fnv_framing() {
        // Verify determinism and parent/child sensitivity:
        let a = hash_kv_block(&[1, 2, 3], None);
        let b = hash_kv_block(&[1, 2, 3], None);
        assert_eq!(a, b);
        assert_ne!(a, hash_kv_block(&[1, 2, 4], None));
        assert_ne!(a, hash_kv_block(&[1, 2, 3], Some(7)));
        // The empty hash is the FNV offset basis after mixing a 0 parent flag
        // and a 0 count.
        let empty = hash_kv_block(&[], None);
        let mut expect = FNV_OFFSET_BASIS;
        mix_byte(&mut expect, 0);
        mix_u64(&mut expect, 0);
        assert_eq!(empty, expect);
    }
}
