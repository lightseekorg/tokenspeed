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

//! Cache group configuration and validation.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/core/cache_config.{h,cpp}`.
//! Validation error messages must match the C++ `std::invalid_argument` text
//! character for character: Python tests assert on them.

/// Logical cache-group family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheGroupFamily {
    History,
    State,
}

/// Transfer policy for tier / PD cache movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTransferPolicy {
    Unspecified,
    FullSuffix,
    LatestSnapshot,
}

/// Retention policy of one cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    FullHistory,
    SlidingWindow,
}

/// Configuration of one attention cache group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheGroupConfig {
    pub group_id: String,
    pub rows_per_page: i32,
    pub entry_stride_tokens: i32,
    pub total_pages: i32,
    /// Number of this group's CacheBlocks packed into one physical LCM block.
    pub cache_blocks_per_lcm_block: i32,
    pub retention: Retention,
    pub sliding_window_tokens: Option<i32>,
    pub family: CacheGroupFamily,
    pub transfer_policy: CacheTransferPolicy,
}

impl Default for CacheGroupConfig {
    fn default() -> Self {
        Self {
            group_id: String::new(),
            rows_per_page: 0,
            entry_stride_tokens: 0,
            total_pages: 0,
            cache_blocks_per_lcm_block: 1,
            retention: Retention::FullHistory,
            sliding_window_tokens: None,
            family: CacheGroupFamily::History,
            transfer_policy: CacheTransferPolicy::Unspecified,
        }
    }
}

impl CacheGroupConfig {
    /// Tokens covered by one CacheBlock of this group.
    pub fn block_granularity(&self) -> i32 {
        self.rows_per_page * self.entry_stride_tokens
    }

    /// A `State` group WITHOUT `SlidingWindow` retention keeps one
    /// recurrent-state checkpoint per block instead of a token history. Note
    /// `family == State` alone is not enough: it also covers linear-attention
    /// sliding groups.
    pub fn is_snapshot_state_group(&self) -> bool {
        matches!(self.family, CacheGroupFamily::State) && self.retention != Retention::SlidingWindow
    }

    /// Validate every invariant, returning the first violation message (the
    /// C++ `Validate()` throws `std::invalid_argument` with the same text).
    pub fn validate(&self) -> Result<(), String> {
        if self.group_id.is_empty() {
            return Err("CacheGroupConfig: group_id must be non-empty".to_string());
        }
        // Every remaining message names the group: a model mixes several of
        // them, and the offending one is the only actionable part.
        let where_ = format!("Cache group '{}': ", self.group_id);
        if self.rows_per_page <= 0 {
            return Err(format!("{where_}rows_per_page must be > 0"));
        }
        if self.entry_stride_tokens <= 0 {
            return Err(format!("{where_}entry_stride_tokens must be > 0"));
        }
        if self.total_pages < 1 {
            return Err(format!("{where_}total_pages must include the null page"));
        }
        if self.cache_blocks_per_lcm_block <= 0 {
            return Err(format!("{where_}cache_blocks_per_lcm_block must be > 0"));
        }
        if self.retention == Retention::SlidingWindow
            && self.sliding_window_tokens.is_none_or(|v| v <= 0)
        {
            return Err(format!(
                "{where_}sliding_window_tokens must be > 0 for sliding groups"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> CacheGroupConfig {
        CacheGroupConfig {
            group_id: "kv".to_string(),
            rows_per_page: 16,
            entry_stride_tokens: 4,
            total_pages: 65,
            ..CacheGroupConfig::default()
        }
    }

    #[test]
    fn block_granularity_is_rows_times_stride() {
        assert_eq!(valid().block_granularity(), 64);
    }

    #[test]
    fn snapshot_state_group_requires_state_family_without_sliding() {
        let mut cfg = valid();
        cfg.family = CacheGroupFamily::State;
        assert!(cfg.is_snapshot_state_group());
        cfg.retention = Retention::SlidingWindow;
        cfg.sliding_window_tokens = Some(128);
        assert!(!cfg.is_snapshot_state_group());
        cfg.retention = Retention::FullHistory;
        cfg.family = CacheGroupFamily::History;
        assert!(!cfg.is_snapshot_state_group());
    }

    #[test]
    fn validate_accepts_defaults_for_non_sliding_groups() {
        assert!(valid().validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_group_id() {
        let mut cfg = valid();
        cfg.group_id.clear();
        assert_eq!(
            cfg.validate().unwrap_err(),
            "CacheGroupConfig: group_id must be non-empty"
        );
    }

    #[test]
    fn validate_messages_name_the_group() {
        let mut cfg = valid();
        cfg.rows_per_page = 0;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': rows_per_page must be > 0"
        );
        cfg.rows_per_page = 16;
        cfg.entry_stride_tokens = -1;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': entry_stride_tokens must be > 0"
        );
        cfg.entry_stride_tokens = 4;
        cfg.total_pages = 0;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': total_pages must include the null page"
        );
        cfg.total_pages = 65;
        cfg.cache_blocks_per_lcm_block = 0;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': cache_blocks_per_lcm_block must be > 0"
        );
    }

    #[test]
    fn validate_requires_sliding_window_tokens_for_sliding_groups() {
        let mut cfg = valid();
        cfg.retention = Retention::SlidingWindow;
        assert_eq!(
            cfg.validate().unwrap_err(),
            "Cache group 'kv': sliding_window_tokens must be > 0 for sliding groups"
        );
        cfg.sliding_window_tokens = Some(0);
        assert!(cfg.validate().is_err());
        cfg.sliding_window_tokens = Some(128);
        assert!(cfg.validate().is_ok());
    }
}
