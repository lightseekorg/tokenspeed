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

//! One cache group's token -> page arithmetic helpers shared by the scheduler
//! and the FSM (C++ `scheduler/operations/cache.{h,cpp}`).

use std::collections::BTreeMap;

use crate::block_table::BlockTable;
use crate::cache_config::CacheGroupConfig;
use crate::cache_coordinator::CacheCoordinator;
use crate::cache_types::{AttnKind, CacheGroupSpec};
use crate::types::SchedulerConfig;

/// One `CacheGroupSpec` per config cache_group (group_id = index); all groups
/// share `config.prefix_granularity`. The caller must have accepted `config`
/// through `SchedulerConfig::validate()` first.
pub fn make_specs_from_config(config: &SchedulerConfig) -> Vec<CacheGroupSpec> {
    let mut specs = Vec::with_capacity(config.cache_groups.len());
    for group in &config.cache_groups {
        if group.is_snapshot_state_group() {
            specs.push(CacheGroupSpec {
                kind: AttnKind::MambaState,
                sliding_window: 0,
                cache_blocks_per_lcm_block: group.cache_blocks_per_lcm_block,
                block_granularity: group.block_granularity(),
            });
            continue;
        }
        // family=State also covers linear-attention groups with a trailing
        // window; those translate like any other sliding group.
        let is_swa = group.retention == crate::cache_config::Retention::SlidingWindow;
        specs.push(CacheGroupSpec {
            kind: if is_swa {
                AttnKind::SlidingWindow
            } else {
                AttnKind::Full
            },
            sliding_window: if is_swa {
                group
                    .sliding_window_tokens
                    .expect("sliding groups validate their window")
            } else {
                0
            },
            cache_blocks_per_lcm_block: group.cache_blocks_per_lcm_block,
            block_granularity: group.block_granularity(),
        });
    }
    specs
}

/// Align a prefill chunk to the cache-prefix granularity (C++ `AlignPrefillChunk`).
pub fn align_prefill_chunk(
    first_pos: i32,
    unscheduled: i32,
    token_budget: i32,
    prefix_granularity: i32,
    promotion_boundary_tokens: i32,
) -> i32 {
    assert!(
        first_pos >= 0 && unscheduled >= 0 && token_budget >= 0,
        "prefill positions must be non-negative"
    );
    assert!(prefix_granularity > 0, "prefix_granularity must be > 0");
    let mut chunk_size = unscheduled.min(token_budget);
    if promotion_boundary_tokens > first_pos {
        chunk_size = chunk_size.min(promotion_boundary_tokens - first_pos);
    }
    if chunk_size == unscheduled {
        return chunk_size;
    }
    let prefix_page_offset = first_pos % prefix_granularity;
    if prefix_page_offset != 0 {
        let tokens_to_boundary = prefix_granularity - prefix_page_offset;
        return if token_budget >= tokens_to_boundary {
            tokens_to_boundary
        } else {
            0
        };
    }
    chunk_size - chunk_size % prefix_granularity
}

/// Release every request-owned page (no-op when the request never got tables).
pub fn free_request(coordinator: &mut CacheCoordinator, tables: &mut [BlockTable]) {
    if tables.is_empty() {
        return; // request never got tables, or a failure path already released them
    }
    coordinator.free(tables);
}

/// One row per config group_id; each group allocator resolves the LCM placement
/// to the kernel-visible page id.
pub fn build_block_tables(
    coordinator: &CacheCoordinator,
    tables: &[BlockTable],
    group_ids: &[String],
) -> BTreeMap<String, Vec<i32>> {
    assert!(
        tables.len() == group_ids.len(),
        "BuildBlockTables: tables/group_ids size mismatch"
    );
    assert!(
        tables.len() == coordinator.num_groups() as usize,
        "BuildBlockTables: tables/coordinator size mismatch"
    );
    let mut out = BTreeMap::new();
    for (i, table) in tables.iter().enumerate() {
        out.insert(
            group_ids[i].clone(),
            coordinator.allocator(i).block_table_page_ids(table),
        );
    }
    out
}

/// Convenience: one `CacheGroupConfig` -> a `CacheGroupSpec` (used by tests and
/// the pyo3 boundary; the scheduler uses [`make_specs_from_config`]).
pub fn spec_from_group(group: &CacheGroupConfig) -> CacheGroupSpec {
    if group.is_snapshot_state_group() {
        CacheGroupSpec {
            kind: AttnKind::MambaState,
            sliding_window: 0,
            cache_blocks_per_lcm_block: group.cache_blocks_per_lcm_block,
            block_granularity: group.block_granularity(),
        }
    } else {
        let is_swa = group.retention == crate::cache_config::Retention::SlidingWindow;
        CacheGroupSpec {
            kind: if is_swa {
                AttnKind::SlidingWindow
            } else {
                AttnKind::Full
            },
            sliding_window: if is_swa {
                group
                    .sliding_window_tokens
                    .expect("sliding groups validate their window")
            } else {
                0
            },
            cache_blocks_per_lcm_block: group.cache_blocks_per_lcm_block,
            block_granularity: group.block_granularity(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_config::{CacheGroupConfig, CacheGroupFamily, Retention};

    fn group(id: &str, rows: i32, stride: i32) -> CacheGroupConfig {
        CacheGroupConfig {
            group_id: id.to_string(),
            rows_per_page: rows,
            entry_stride_tokens: stride,
            total_pages: 65,
            cache_blocks_per_lcm_block: 1,
            retention: Retention::FullHistory,
            sliding_window_tokens: None,
            family: CacheGroupFamily::History,
            transfer_policy: crate::cache_config::CacheTransferPolicy::Unspecified,
        }
    }

    #[test]
    fn make_specs_from_config_maps_groups() {
        let cfg = crate::types::SchedulerConfig {
            prefix_granularity: 64,
            cache_groups: vec![group("kv", 16, 4), {
                let mut g = group("state", 64, 1);
                g.family = CacheGroupFamily::State;
                g
            }],
            ..crate::types::SchedulerConfig::default()
        };
        let specs = make_specs_from_config(&cfg);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].kind, AttnKind::Full);
        assert_eq!(specs[0].block_granularity, 64);
        // Snapshot-state group -> MambaState.
        assert_eq!(specs[1].kind, AttnKind::MambaState);
        assert_eq!(specs[1].block_granularity, 64);
    }

    #[test]
    fn align_prefill_chunk_pads_to_boundary() {
        // Unaligned start pads to the page boundary when the budget is below
        // the unscheduled total.
        assert_eq!(align_prefill_chunk(2, 10, 5, 4, 0), 2); // 4-2 = 2 tokens to boundary
                                                            // Aligned start floors to the granularity.
        assert_eq!(align_prefill_chunk(4, 10, 9, 4, 0), 8);
        // Full chunk stays as-is.
        assert_eq!(align_prefill_chunk(0, 4, 10, 4, 0), 4);
        // Promotion boundary caps the chunk.
        assert_eq!(align_prefill_chunk(0, 10, 10, 4, 6), 4);
        // Budget too small to reach boundary -> 0.
        assert_eq!(align_prefill_chunk(2, 10, 1, 4, 0), 0);
    }
}
