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

//! The per-round batch lines operators read: every
//! prefill or mixed round, and every `decode_log_interval`-th decode round
//! with the decode throughput since the previous line.

use std::{collections::HashSet, time::Instant};

use ts_scheduler::ForwardBatchView;

/// Prefill ids remembered to count each request's cache hit once; the set is
/// cleared past this.
const MAX_SEEN_PREFILL_IDS: usize = 100_000;

/// The logger's fixed settings.
#[derive(Debug, Clone)]
pub struct BatchLogConfig {
    /// This rank speaks for its scheduler: one attention-TP rank per
    /// attention-DP scheduler, on the first pipeline stage. Other ranks still
    /// count.
    pub enabled: bool,
    /// Decode rounds between two decode lines.
    pub decode_log_interval: u64,
    /// Usable KV pages, the total the metrics publish too.
    pub num_total_pages: i64,
    /// Draft steps per verify; 0 without speculation.
    pub spec_num_steps: i64,
    /// The attention-DP rank printed on every line.
    pub dp_rank: i64,
    /// A disaggregated engine: lines carry the PD request states.
    pub disaggregated: bool,
    /// State-family cache groups, whose page use a decode line reports at
    /// debug level.
    pub cache_state_group_ids: Vec<String>,
    /// `TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS`: log every verify step's accept
    /// lengths.
    pub log_spec_accept_lengths: bool,
}

/// The scheduler's counts a line reads, sampled when the round planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundStats {
    /// Requests in the waiting queue.
    pub queued: usize,
    /// KV pages held by running requests.
    pub active_pages: i64,
    /// KV pages holding only cache entries.
    pub cached_pages: i64,
    /// PD: requests bootstrapping, prefilling, remote-prefilling, decoding
    /// and pinned by a transfer.
    pub pd: [usize; 5],
}

/// One rank's batch logger.
#[derive(Debug)]
pub struct BatchLogger {
    config: BatchLogConfig,
    step: u64,
    seen_prefill_ids: HashSet<String>,
    /// Tokens committed by decode rounds since the last decode line.
    num_generated_tokens: i64,
    num_decode_steps: i64,
    last_decode_line: Instant,
}

impl BatchLogger {
    /// A logger with `config`.
    pub fn new(config: BatchLogConfig) -> Self {
        Self {
            config,
            step: 0,
            seen_prefill_ids: HashSet::new(),
            num_generated_tokens: 0,
            num_decode_steps: 0,
            last_decode_line: Instant::now(),
        }
    }

    fn queue_suffix(&self, stats: &RoundStats) -> String {
        if !self.config.disaggregated {
            return format!(", #queue-req: {}", stats.queued);
        }
        // Every PD request waits in Bootstrapping until its peer is ready, a
        // state the waiting count leaves out; to the operator it is queued.
        let [bootstrap, prefill, remote_prefill, decode, pinned] = stats.pd;
        format!(
            ", #queue-req: {}, #req-state(bootstrap/prefill/remote-prefill/decode/pd-pinned): \
             {bootstrap}/{prefill}/{remote_prefill}/{decode}/{pinned}",
            stats.queued + bootstrap
        )
    }

    /// Log the round being dispatched. `context_length` is a running
    /// request's tokens (prompt and generated); `group_pages` a cache group's
    /// total and available pages. Both are read only for an emitted line.
    pub fn log_dispatch(
        &mut self,
        batch: &ForwardBatchView,
        stats: &RoundStats,
        context_length: impl Fn(&str) -> i64,
        group_pages: impl Fn(&str) -> (i32, i32),
    ) {
        self.step += 1;
        if !self.config.enabled {
            return;
        }
        let num_extends = batch.num_extends;
        let bs = batch.request_ids.len();
        if num_extends > 0 {
            self.log_extend(batch, stats);
        } else if self.step % self.config.decode_log_interval.max(1) == 0 {
            self.log_decode(batch, bs, stats, context_length, group_pages);
        }
    }

    fn log_extend(&mut self, batch: &ForwardBatchView, stats: &RoundStats) {
        let num_extends = batch.num_extends;
        let bs = batch.request_ids.len();
        let mode = if num_extends == bs { "Prefill" } else { "Mix" };
        let total_tokens: i64 = batch.input_lengths.iter().map(|&n| i64::from(n)).sum();
        let cached_tokens: i64 = batch.request_ids[..num_extends]
            .iter()
            .zip(&batch.extend_prefix_lens)
            .filter(|(rid, _)| !self.seen_prefill_ids.contains(*rid))
            .map(|(_, &prefix)| i64::from(prefix))
            .sum();
        if self.seen_prefill_ids.len() > MAX_SEEN_PREFILL_IDS {
            self.seen_prefill_ids.clear();
        }
        self.seen_prefill_ids
            .extend(batch.request_ids[..num_extends].iter().cloned());
        tracing::info!(
            "{mode} batch. #dp-rank: {}, #new-seq: {num_extends}, #new-token: {total_tokens}, \
             #cached-token: {cached_tokens}, #running-req: {bs}{}",
            self.config.dp_rank,
            self.queue_suffix(stats)
        );
    }

    fn log_decode(
        &mut self,
        batch: &ForwardBatchView,
        bs: usize,
        stats: &RoundStats,
        context_length: impl Fn(&str) -> i64,
        group_pages: impl Fn(&str) -> (i32, i32),
    ) {
        let now = Instant::now();
        let gap = now.duration_since(self.last_decode_line).as_secs_f64();
        let throughput = if gap > 0.0 {
            self.num_generated_tokens as f64 / gap
        } else {
            0.0
        };
        let avg_accept = if self.num_decode_steps > 0 {
            self.num_generated_tokens as f64 / self.num_decode_steps as f64
        } else {
            0.0
        };
        let avg_seq_len = if bs > 0 {
            batch
                .request_ids
                .iter()
                .map(|rid| context_length(rid))
                .sum::<i64>() as f64
                / bs as f64
        } else {
            0.0
        };
        let total = self.config.num_total_pages;
        let page_ratio = if total > 0 {
            stats.active_pages as f64 / total as f64
        } else {
            0.0
        };
        let spec = if self.config.spec_num_steps > 0 {
            format!(
                ", avg_accept_len: {avg_accept:.2}, accept_rate: {:.2}",
                (avg_accept - 1.0) / self.config.spec_num_steps as f64
            )
        } else {
            String::new()
        };
        tracing::info!(
            "Decode batch. #dp-rank: {}, #running-req: {bs}, avg_seq_len: {avg_seq_len:.1}, \
             #pages(active/cached/total): {}/{}/{total}, page ratio: {page_ratio:.2}, gen \
             throughput (token/s): {throughput:.2}{spec}{}",
            self.config.dp_rank,
            stats.active_pages,
            stats.cached_pages,
            self.queue_suffix(stats)
        );
        if !self.config.cache_state_group_ids.is_empty() && tracing::enabled!(tracing::Level::DEBUG)
        {
            // State groups are sized apart from the KV groups, so the line's
            // single page ratio hides one that binds.
            let parts: Vec<String> = self
                .config
                .cache_state_group_ids
                .iter()
                .map(|group| {
                    let (total, available) = group_pages(group);
                    format!(
                        "{group}: used={}/{total}, available={available}",
                        total - available
                    )
                })
                .collect();
            tracing::debug!(
                "Cache state group pages. #dp-rank: {}, {}",
                self.config.dp_rank,
                parts.join("; ")
            );
        }
        self.num_generated_tokens = 0;
        self.num_decode_steps = 0;
        self.last_decode_line = now;
    }

    /// Fold a committed decode round's kept tokens into the throughput.
    pub fn record_decode(&mut self, output_lengths: &[i32]) {
        self.num_generated_tokens += output_lengths.iter().map(|&n| i64::from(n)).sum::<i64>();
        self.num_decode_steps += output_lengths.len() as i64;
        if self.config.log_spec_accept_lengths
            && self.config.enabled
            && self.config.spec_num_steps > 0
        {
            let drafts: Vec<i32> = output_lengths.iter().map(|&n| (n - 1).max(0)).collect();
            tracing::info!(
                "Spec verify step. #dp-rank: {}, accept_lengths={output_lengths:?}, \
                 accepted_draft_tokens={drafts:?}",
                self.config.dp_rank
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(rids: &[&str], num_extends: usize, prefixes: &[i32]) -> ForwardBatchView {
        ForwardBatchView {
            request_ids: rids.iter().map(|&rid| rid.to_owned()).collect(),
            request_pool_indices: (0..rids.len() as i32).collect(),
            input_lengths: vec![4; rids.len()],
            prefill_lengths: vec![8; rids.len()],
            extend_prefix_lens: prefixes.to_vec(),
            extend_replay_lens: vec![0; num_extends],
            num_extends,
        }
    }

    #[test]
    fn a_request_s_cache_hit_counts_on_its_first_chunk_only() {
        let mut logger = BatchLogger::new(BatchLogConfig {
            enabled: true,
            decode_log_interval: 40,
            num_total_pages: 100,
            spec_num_steps: 0,
            dp_rank: 0,
            disaggregated: false,
            cache_state_group_ids: Vec::new(),
            log_spec_accept_lengths: false,
        });
        let stats = RoundStats {
            queued: 0,
            active_pages: 1,
            cached_pages: 0,
            pd: [0; 5],
        };
        logger.log_dispatch(&batch(&["a"], 1, &[4]), &stats, |_| 0, |_| (0, 0));
        assert!(logger.seen_prefill_ids.contains("a"));
        logger.record_decode(&[1, 3]);
        assert_eq!(
            (logger.num_generated_tokens, logger.num_decode_steps),
            (4, 2)
        );
        assert_eq!(logger.step, 1);
    }
}
