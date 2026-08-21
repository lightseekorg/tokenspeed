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

//! Scheduler configuration and validation.
//!
//! Ported from `tokenspeed-scheduler/csrc/scheduler/types.{h,cpp}`. Validation
//! error messages match the C++ `std::invalid_argument` text character for
//! character (Python tests assert on them).

use crate::cache_config::{CacheGroupConfig, CacheTransferPolicy};

/// Scheduler role: prefill-only, decode-only, or fused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    P,
    D,
    Fused,
}

/// Block-pool sizing for one tier. Page 0 is the null placeholder, so usable =
/// total - 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AllocatorConfig {
    pub total_pages: i32,
}

impl AllocatorConfig {
    pub fn num_usable_blocks(&self) -> i32 {
        self.total_pages - 1
    }
}

/// Full scheduler configuration. `validate` is the single validation entry
/// point; the scheduler runs it before constructing any member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub prefix_granularity: i32,
    pub host_allocator: AllocatorConfig,
    pub device_allocator: AllocatorConfig,
    pub cache_groups: Vec<CacheGroupConfig>,
    pub max_scheduled_tokens: i32,
    pub max_batch_size: i32,
    pub decode_input_tokens: i32,
    /// Number of scheduler iterations that may be dispatched before the
    /// accepted decode length is committed. Only 0 (non-overlapped) and 1
    /// (one-step overlapped) are supported.
    pub overlap_schedule_depth: i32,
    pub disable_l2_cache: bool,
    pub enable_l3_storage: bool,
    pub enable_kv_cache_events: bool,
    pub enable_mixed_prefill_decode: bool,
    pub role: Role,
    pub enable_pd_cache: bool,
    pub disable_prefix_cache: bool,
    /// Minimum prompt tail that must be recomputed after a prefix-cache hit.
    pub prefix_replay_tokens: i32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            prefix_granularity: 0,
            host_allocator: AllocatorConfig::default(),
            device_allocator: AllocatorConfig::default(),
            cache_groups: Vec::new(),
            max_scheduled_tokens: 0,
            max_batch_size: 0,
            decode_input_tokens: 1,
            overlap_schedule_depth: 0,
            disable_l2_cache: false,
            enable_l3_storage: false,
            enable_kv_cache_events: false,
            enable_mixed_prefill_decode: false,
            role: Role::Fused,
            enable_pd_cache: false,
            disable_prefix_cache: false,
            prefix_replay_tokens: 0,
        }
    }
}

impl SchedulerConfig {
    pub fn has_host_cache(&self) -> bool {
        !self.disable_l2_cache && self.host_allocator.total_pages > 1
    }

    /// Decode uses Host cache only for best-effort Retraction and recovery.
    pub fn streams_device_cache_to_host(&self) -> bool {
        self.has_host_cache() && self.role != Role::D
    }

    /// Validate every scalar, every group's own invariants, and the
    /// cross-checks between them. Returns the first violation message (the C++
    /// `Validate()` throws `std::invalid_argument`).
    pub fn validate(&self) -> Result<(), String> {
        if self.prefix_granularity <= 0 {
            return Err("Scheduler: prefix_granularity must be > 0".to_string());
        }
        if self.device_allocator.total_pages <= 1 {
            return Err(
                "Scheduler: device cache must contain a null page and usable capacity".to_string(),
            );
        }
        if self.cache_groups.is_empty() {
            return Err("Scheduler: at least one cache group is required".to_string());
        }
        if self.decode_input_tokens < 0 {
            return Err("Scheduler: decode_input_tokens must be >= 0".to_string());
        }
        if self.max_scheduled_tokens <= 0 {
            return Err("Scheduler: max_scheduled_tokens must be > 0".to_string());
        }
        if !(0..=1).contains(&self.overlap_schedule_depth) {
            return Err("Scheduler: overlap_schedule_depth must be 0 or 1".to_string());
        }
        if self.overlap_schedule_depth > 0 && self.decode_input_tokens == 0 {
            return Err(
                "Scheduler: overlapped decode requires decode_input_tokens > 0".to_string(),
            );
        }
        if self.prefix_replay_tokens < 0 {
            return Err("Scheduler: prefix_replay_tokens must be >= 0".to_string());
        }
        if self.enable_l3_storage {
            return Err(
                "Scheduler: L3 storage is not supported by the cache coordinator".to_string(),
            );
        }
        for group in &self.cache_groups {
            validate_group(self, group)?;
            // A recurrent state advances one whole checkpoint at a time, so a
            // chunk must be able to cover one cache block.
            if group.is_snapshot_state_group()
                && self.max_scheduled_tokens < self.prefix_granularity
            {
                return Err(
                    "Scheduler: Mamba max_scheduled_tokens must cover one cache block".to_string(),
                );
            }
        }
        Ok(())
    }
}

fn validate_group(config: &SchedulerConfig, group: &CacheGroupConfig) -> Result<(), String> {
    group.validate()?;
    let where_ = format!("Cache group '{}': ", group.group_id);
    if config.prefix_granularity % group.block_granularity() != 0 {
        return Err(format!(
            "{where_}block_granularity must divide the scheduler prefix_granularity"
        ));
    }
    if !config.enable_pd_cache {
        return Ok(());
    }
    // A group's transfer policy is dictated by the destination layout the
    // scheduler builds for it, so it cannot be chosen independently.
    let expected = if group.is_snapshot_state_group() {
        CacheTransferPolicy::LatestSnapshot
    } else {
        CacheTransferPolicy::FullSuffix
    };
    if group.transfer_policy == CacheTransferPolicy::Unspecified {
        return Err(format!(
            "{where_}PD cache requires an explicit transfer_policy"
        ));
    }
    if group.transfer_policy != expected {
        return Err(format!(
            "{where_}transfer_policy does not match its scheduler destination layout"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_config::{CacheGroupFamily, Retention};

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
            transfer_policy: CacheTransferPolicy::Unspecified,
        }
    }

    fn valid() -> SchedulerConfig {
        SchedulerConfig {
            prefix_granularity: 64,
            host_allocator: AllocatorConfig { total_pages: 0 },
            device_allocator: AllocatorConfig { total_pages: 1025 },
            cache_groups: vec![group("kv", 16, 4)],
            max_scheduled_tokens: 128,
            max_batch_size: 8,
            ..SchedulerConfig::default()
        }
    }

    #[test]
    fn validate_accepts_well_formed_config() {
        assert!(valid().validate().is_ok());
    }

    #[test]
    fn validate_rejects_each_violation() {
        let mut cfg = valid();
        cfg.prefix_granularity = 0;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: prefix_granularity must be > 0"
        );
        cfg.prefix_granularity = 64;

        cfg.device_allocator.total_pages = 1;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: device cache must contain a null page and usable capacity"
        );
        cfg.device_allocator.total_pages = 1025;

        cfg.cache_groups.clear();
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: at least one cache group is required"
        );
        cfg.cache_groups.push(group("kv", 16, 4));

        cfg.max_scheduled_tokens = 0;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: max_scheduled_tokens must be > 0"
        );
        cfg.max_scheduled_tokens = 128;

        cfg.overlap_schedule_depth = 2;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: overlap_schedule_depth must be 0 or 1"
        );
        cfg.overlap_schedule_depth = 1;
        cfg.decode_input_tokens = 0;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: overlapped decode requires decode_input_tokens > 0"
        );
        cfg.decode_input_tokens = 1;
        cfg.overlap_schedule_depth = 0;

        cfg.enable_l3_storage = true;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: L3 storage is not supported by the cache coordinator"
        );
        cfg.enable_l3_storage = false;
    }

    #[test]
    fn validate_group_checks_block_granularity_divisor() {
        let mut cfg = valid();
        cfg.cache_groups[0].rows_per_page = 15; // 15*4=60 does not divide 64
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': block_granularity must divide the scheduler prefix_granularity"
        );
    }

    #[test]
    fn validate_pd_cache_requires_matching_transfer_policy() {
        let mut cfg = valid();
        cfg.enable_pd_cache = true;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': PD cache requires an explicit transfer_policy"
        );
        cfg.cache_groups[0].transfer_policy = CacheTransferPolicy::FullSuffix;
        assert!(cfg.validate().is_ok());
        cfg.cache_groups[0].transfer_policy = CacheTransferPolicy::LatestSnapshot;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': transfer_policy does not match its scheduler destination layout"
        );
    }

    #[test]
    fn mamba_group_requires_chunk_covering_one_block() {
        let mut cfg = valid();
        cfg.cache_groups[0].family = CacheGroupFamily::State;
        cfg.max_scheduled_tokens = 32;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Scheduler: Mamba max_scheduled_tokens must cover one cache block"
        );
        cfg.max_scheduled_tokens = 64;
        assert!(cfg.validate().is_ok());
    }
}
