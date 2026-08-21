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

//! Leaf cache types shared across the scheduler.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/core/cache_types.h`. The
//! demand/match types that reference `BlockTable`/`CacheBlockRef` land with the
//! coordinator batch (see `docs/design/rust-port.md` §4).

/// Attention flavor of a cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnKind {
    Full,
    SlidingWindow,
    MambaState,
}

/// Why a resumable cache boundary was retained. Declaration order is its
/// monotonic promotion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheBoundaryKind {
    Chunk,
    Endpoint,
    Promoted,
}

/// Namespace id for cache keys (LoRA/context separation).
pub type CacheNamespaceId = u32;
/// Content hash of one prefix page, hex-encoded (64 chars for SHA-256).
pub type ContentHash = String;

/// Default cache namespace.
pub const DEFAULT_CACHE_NAMESPACE_ID: CacheNamespaceId = 0;

/// Identifies one cached block: namespace + group + content hash + the
/// ordinal of this group's page within the enclosing prefix page.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub namespace_id: CacheNamespaceId,
    pub group_id: u32,
    pub content_hash: ContentHash,
    /// Ordinal of this group's page within the enclosing prefix page, NOT a
    /// token/byte offset inside a page.
    pub page_offset: i32,
}

impl Default for CacheKey {
    fn default() -> Self {
        Self {
            namespace_id: DEFAULT_CACHE_NAMESPACE_ID,
            group_id: 0,
            content_hash: ContentHash::new(),
            page_offset: 0,
        }
    }
}

/// Static geometry of one attention group, in logical-page units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheGroupSpec {
    pub kind: AttnKind,
    /// Only `SlidingWindow` uses this value.
    pub sliding_window: i32,
    /// Number of this group's CacheBlocks packed into one physical LCM block.
    /// Affects placement only, not the scheduler-wide prefix granularity.
    pub cache_blocks_per_lcm_block: i32,
    /// Tokens represented by one CacheBlock in this group; must be a positive
    /// divisor of the coordinator-wide prefix granularity.
    pub block_granularity: i32,
}

impl Default for CacheGroupSpec {
    fn default() -> Self {
        Self {
            kind: AttnKind::Full,
            sliding_window: 0,
            cache_blocks_per_lcm_block: 1,
            block_granularity: 0,
        }
    }
}

/// Per-group input for one admission. `prefix_hashes` is the request's
/// cumulative completed prefix-page history; `new_prefix_hash_begin` is the
/// start of the hashes appended since the previous admission.
/// `completed_boundary_kind` is present exactly when that suffix is non-empty.
/// Non-closed groups select the trailing pages required to resume
/// `num_computed_tokens`.
#[derive(Default)]
pub struct GroupDemand<'a> {
    /// The group's block table (asserted non-null by the coordinator).
    pub table: Option<&'a mut crate::block_table::BlockTable>,
    pub num_tokens: i32,
    pub prefix_hashes: &'a [String],
    pub new_prefix_hash_begin: i32,
    pub completed_boundary_kind: Option<CacheBoundaryKind>,
    pub num_computed_tokens: i32,
    pub reserve_tokens: i32,
    /// -1 materializes the ordinary dense suffix. A non-negative value keeps
    /// earlier logical slots as null holes and materializes only this suffix.
    /// Snapshot-state local prefill uses an absolute endpoint here; Decode-side
    /// PD also uses it for latest snapshots and retained sliding tails.
    pub materialized_suffix_start: i32,
}

/// Match result for one group: owned [`CacheBlockRef`]s aligned with the probe
/// hits (holes are null refs).
#[derive(Debug, Default)]
pub struct PrefixMatch {
    pub blocks: Vec<crate::cache_block_ref::CacheBlockRef>,
}

impl PrefixMatch {
    /// Number of non-null (hit) blocks.
    pub fn num_hit_blocks(&self) -> i32 {
        self.blocks.iter().filter(|b| !b.is_null()).count() as i32
    }
}

/// Non-owning match shape. A nonzero slot is acquired only after the
/// coordinator converges every group to the final common boundary.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GroupPrefixProbe {
    pub hits: Vec<u8>,
}

/// Pinned source/destination blocks for one asynchronous cache transfer.
#[derive(Debug, Clone)]
pub struct BlockTransfer {
    pub group_id: u32,
    pub source: crate::cache_block_ref::CacheBlockRef,
    pub destination: crate::cache_block_ref::CacheBlockRef,
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cache_key_uses_default_namespace_and_zero_fields() {
        let key = CacheKey::default();
        assert_eq!(key.namespace_id, DEFAULT_CACHE_NAMESPACE_ID);
        assert_eq!(key.group_id, 0);
        assert!(key.content_hash.is_empty());
        assert_eq!(key.page_offset, 0);
    }

    #[test]
    fn cache_key_eq_and_hash_cover_all_fields() {
        let a = CacheKey {
            namespace_id: 1,
            group_id: 2,
            content_hash: "abc".into(),
            page_offset: 3,
        };
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, a.clone());
        // Changing any single field breaks equality.
        assert_ne!(
            a,
            CacheKey {
                namespace_id: 9,
                ..a.clone()
            }
        );
        assert_ne!(
            a,
            CacheKey {
                group_id: 9,
                ..a.clone()
            }
        );
        assert_ne!(
            a,
            CacheKey {
                content_hash: "xyz".into(),
                ..a.clone()
            }
        );
        assert_ne!(
            a,
            CacheKey {
                page_offset: 9,
                ..a.clone()
            }
        );
    }
}
