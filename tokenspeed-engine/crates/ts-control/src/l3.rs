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

//! The L3 storage tier as the loop sees it: this rank's storage executor and
//! the replica's agreement on what the executors report, every round
//! (`docs/design/async-l3-cache.md`).
//!
//! The C++ scheduler owns readiness, deadlines and Host memory lifetime; it
//! plans storage operations (query, get, put, cancel) that each cache owner
//! runs on its Host-only executor (`ts-l3`), which the loop starts with the
//! plan. The ranks of a replica mirror one scheduler, so each round they
//! share their executors' reports and agree on the storage clock and the
//! worst pressure, and on each operation's outcome once every owner
//! reported it. The reports ride the round's relay ([`crate::relay::join`]):
//! one collective a round for both. The permissions to run forwards and
//! start storage work come from the control state, which every member holds
//! alike. A fault on one rank, even in its own poll, fails every rank in the
//! same round instead of stranding peers in the next collective.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use ts_l3::{Executor, Operation, StorageKey};
pub use ts_l3::{StorageCompletion, StoragePoll, StoragePressure};
use ts_metrics::L3SchedulerStats;
use ts_scheduler::{
    ExecutionPlan, Scheduler, StorageCancellationView, StorageDoneView, StorageOpKind,
    StorageRound, StorageStatsView, StorageTickView,
};

use crate::codec::{Reader, Writer};

/// The scheduler's storage credits (`StorageConfig`) the agreement checks
/// reports against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLimits {
    /// Operations in flight.
    pub max_operations: usize,
    /// Keys one operation names.
    pub max_keys_per_operation: usize,
}

/// The replica could not agree on its storage reports.
#[derive(Debug, thiserror::Error)]
pub enum L3Error {
    /// A rank's poll or report was invalid; every rank fails with it.
    #[error("L3 replica polling failed: {0:?}")]
    Polling(Vec<String>),
    /// The reports break the protocol.
    #[error("L3: {0}")]
    Protocol(&'static str),
}

/// What one rank shares each round.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Report {
    now_us: i64,
    pressure: StoragePressure,
    error: Option<String>,
    completions: Vec<StorageCompletion>,
}

/// Owner reports kept until every owner of the operation reported.
type OwnerReport = (Vec<bool>, bool, Vec<bool>);

/// The agreement, with what persists between rounds.
#[derive(Debug)]
pub struct L3 {
    limits: StorageLimits,
    /// Per operation, each owner's report, until every owner delivered one;
    /// a delivered report is never sent again.
    reports: BTreeMap<u64, BTreeMap<usize, OwnerReport>>,
    /// Requests cancelled since the last round, in first-cancel order.
    cancellations: Vec<StorageCancellationView>,
    /// When the metrics were last sampled, and whether storage was pending.
    metrics_sampled: Option<(Instant, bool)>,
}

impl L3 {
    /// No reports held and nothing cancelled.
    pub fn new(limits: StorageLimits) -> Self {
        Self {
            limits,
            reports: BTreeMap::new(),
            cancellations: Vec::new(),
            metrics_sampled: None,
        }
    }

    /// A request was aborted: queue its cancellation for the next round's
    /// head. Returns whether no future forward will reap its output (it held
    /// no pages yet, so the scheduler finishes it at the head), in which case
    /// the caller reaps it now. A request the scheduler does not hold is not
    /// the storage path's.
    pub fn cancel(&mut self, scheduler: &Scheduler, request_id: &str) -> bool {
        if scheduler.request_token_size(request_id) < 0 {
            return false;
        }
        let abort = scheduler.can_abort_before_admission(request_id);
        let cancellation = StorageCancellationView {
            request_id: request_id.to_owned(),
            abort,
        };
        match self
            .cancellations
            .iter_mut()
            .find(|queued| queued.request_id == request_id)
        {
            Some(queued) => *queued = cancellation,
            None => self.cancellations.push(cancellation),
        }
        abort
    }

    /// The report this rank shares this round: its executor's `poll`, or why
    /// the poll is invalid. A bad record must not strand peers already
    /// exchanging reports, so it becomes the report's error instead of a
    /// local failure.
    pub fn report(&self, poll: StoragePoll) -> Vec<u8> {
        encode(&self.validated(poll))
    }

    /// One round's agreement on every replica member's [`L3::report`], in
    /// member order: the first member's clock and the worst pressure, the
    /// permissions (this rank's, which every member holds alike), the queued
    /// cancellations, and the operations every owner has now completed
    /// (success and retirement ANDed, object failures ORed across owners).
    /// Every member calls this every round.
    pub fn agree(
        &mut self,
        reports: &[Vec<u8>],
        allow_forward: bool,
        allow_storage: bool,
    ) -> Result<StorageRound, L3Error> {
        let reports = reports
            .iter()
            .map(|bytes| decode(bytes).ok_or(L3Error::Protocol("a malformed owner report")))
            .collect::<Result<Vec<_>, _>>()?;
        if reports.is_empty() {
            return Err(L3Error::Protocol("a round without reports"));
        }
        let errors: Vec<String> = reports
            .iter()
            .filter_map(|report| report.error.clone())
            .collect();
        if !errors.is_empty() {
            return Err(L3Error::Polling(errors));
        }
        for (rank, report) in reports.iter().enumerate() {
            if report.completions.len() > self.limits.max_operations {
                return Err(L3Error::Protocol("a peer exceeded its report credits"));
            }
            for completion in &report.completions {
                if completion.success.len() > self.limits.max_keys_per_operation {
                    return Err(L3Error::Protocol("a peer exceeded its key credits"));
                }
                let value = (
                    completion.success.clone(),
                    completion.retired,
                    completion.object_failures.clone(),
                );
                let row = self.reports.entry(completion.op_id).or_default();
                match row.get(&rank) {
                    Some(held) if *held != value => {
                        return Err(L3Error::Protocol("conflicting owner reports"));
                    }
                    _ => {
                        row.insert(rank, value);
                    }
                }
            }
        }
        if self.reports.len() > self.limits.max_operations {
            return Err(L3Error::Protocol(
                "the aggregation exceeded its operation credits",
            ));
        }
        let worst = |field: fn(&StoragePressure) -> i64| {
            reports
                .iter()
                .map(|report| field(&report.pressure))
                .max()
                .unwrap_or_default()
        };
        let tick = StorageTickView {
            now_us: reports[0].now_us,
            allow_forward,
            allow_storage,
            queued_operations: worst(|pressure| pressure.queued_operations),
            queued_bytes: worst(|pressure| pressure.queued_bytes),
            oldest_operation_us: worst(|pressure| pressure.oldest_operation_us),
        };
        let owners = reports.len();
        let mut done = Vec::new();
        for (&op_id, row) in &self.reports {
            if row.len() != owners {
                continue;
            }
            let count = row[&0].0.len();
            if row.values().any(|(success, _, _)| success.len() != count) {
                return Err(L3Error::Protocol("mirrored operation shapes differ"));
            }
            done.push(StorageDoneView {
                op_id,
                success: (0..count)
                    .map(|key| u8::from(row.values().all(|(success, _, _)| success[key])))
                    .collect(),
                retired: row.values().all(|(_, retired, _)| *retired),
                // A confirmed miss on any owner invalidates the object; an
                // owner that rejected the work is no such evidence.
                object_failures: (0..count)
                    .map(|key| u8::from(row.values().any(|(_, _, failed)| failed[key])))
                    .collect(),
            });
        }
        for completion in &done {
            self.reports.remove(&completion.op_id);
        }
        Ok(StorageRound {
            tick: Some(tick),
            cancellations: std::mem::take(&mut self.cancellations),
            done,
        })
    }

    /// Whether to sample the storage metrics now: at most once a second,
    /// and whenever storage turns pending or idle.
    pub fn metrics_due(&mut self, pending: bool) -> bool {
        let due = match self.metrics_sampled {
            None => true,
            Some((at, was_pending)) => {
                at.elapsed() >= Duration::from_secs(1) || was_pending != pending
            }
        };
        if due {
            self.metrics_sampled = Some((Instant::now(), pending));
        }
        due
    }

    /// `poll` as this rank reports it: with its records, or with why they
    /// are invalid in their place.
    fn validated(&self, poll: StoragePoll) -> Report {
        let StoragePoll {
            now_us,
            completions,
            pressure,
        } = poll;
        let invalid = if [
            pressure.queued_operations,
            pressure.queued_bytes,
            pressure.oldest_operation_us,
        ]
        .iter()
        .any(|&value| value < 0)
        {
            Some("Invalid L3 pressure observation".to_owned())
        } else if completions.len() > self.limits.max_operations {
            Some("L3 completion window exceeded operation credits".to_owned())
        } else if completions.iter().any(|completion| {
            completion.op_id == 0
                || completion.success.len() > self.limits.max_keys_per_operation
                || completion.object_failures.len() != completion.success.len()
                || completion
                    .success
                    .iter()
                    .zip(&completion.object_failures)
                    .any(|(&ok, &failed)| ok && failed)
        }) {
            Some("Malformed L3 completion".to_owned())
        } else {
            None
        };
        match invalid {
            Some(error) => Report {
                now_us,
                pressure: StoragePressure::default(),
                error: Some(error.chars().take(512).collect()),
                completions: Vec::new(),
            },
            None => Report {
                now_us,
                pressure,
                error: None,
                completions,
            },
        }
    }
}

/// The tier on this rank: the executor running the plans' storage
/// operations, and the replica's agreement on what the executors report.
pub struct Tier {
    /// This rank's executor.
    pub executor: Executor,
    /// The agreement.
    pub agreement: L3,
}

impl Tier {
    /// The tier over `executor`: the agreement checks the reports against
    /// the executor's credits, the scheduler's.
    pub fn new(executor: Executor) -> Self {
        let config = executor.config();
        Self {
            executor,
            agreement: L3::new(StorageLimits {
                max_operations: config.max_operations,
                max_keys_per_operation: config.max_keys_per_operation,
            }),
        }
    }

    /// Start `plan`'s storage operations; none waits for I/O.
    pub fn submit(&self, plan: &ExecutionPlan) -> Result<(), ts_l3::ExecutorError> {
        let operations = plan
            .storage_ops()
            .into_iter()
            .map(|op| {
                let keys = || -> Vec<StorageKey> {
                    op.keys
                        .iter()
                        .map(|key| StorageKey {
                            group: key.group_id,
                            content_hash: key.content_hash.clone(),
                            page_offset: key.page_offset,
                        })
                        .collect()
                };
                let operation = match op.kind {
                    StorageOpKind::Query => Operation::Query {
                        op_id: op.op_id,
                        keys: keys(),
                    },
                    StorageOpKind::Get => Operation::Get {
                        op_id: op.op_id,
                        keys: keys(),
                        host_blocks: op.host_blocks.clone(),
                    },
                    StorageOpKind::Put => Operation::Put {
                        op_id: op.op_id,
                        keys: keys(),
                        host_blocks: op.host_blocks.clone(),
                    },
                    StorageOpKind::Cancel => Operation::Cancel { op_id: op.op_id },
                    _ => return Err(ts_l3::ExecutorError::Invalid("its kind")),
                };
                Ok(operation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if operations.is_empty() {
            return Ok(());
        }
        self.executor.submit(&operations)
    }

    /// This round's report of this rank's executor; see [`L3::report`].
    pub fn report(&self) -> Vec<u8> {
        self.agreement.report(self.executor.poll())
    }

    /// The round's agreement on the members' reports; see [`L3::agree`].
    pub fn agree(
        &mut self,
        reports: &[Vec<u8>],
        allow_forward: bool,
        allow_storage: bool,
    ) -> Result<StorageRound, L3Error> {
        self.agreement.agree(reports, allow_forward, allow_storage)
    }
}

/// The scheduler's storage observations, as the metrics take them.
pub fn scheduler_stats(stats: StorageStatsView) -> L3SchedulerStats {
    L3SchedulerStats {
        operations: stats.operations as f64,
        held_blocks: stats.held_blocks as f64,
        quarantined: stats.quarantined as f64,
        negative_keys: stats.negative_keys as f64,
        backup_candidates: stats.backup_candidates as f64,
        waiting_requests: stats.waiting_requests as f64,
        ready_requests: stats.ready_requests as f64,
        wait_timeouts: stats.wait_timeouts as f64,
        ready_expirations: stats.ready_expirations as f64,
        pressure_gate_closed: stats.pressure_gate_closed,
        pressure_bypasses: stats.pressure_bypasses as f64,
        pressure_dropped_backup_keys: stats.pressure_dropped_backup_keys as f64,
        overflow_dropped_backup_keys: stats.overflow_dropped_backup_keys as f64,
        unavailable_dropped_backup_keys: stats.unavailable_dropped_backup_keys as f64,
        disabled_dropped_backup_keys: stats.disabled_dropped_backup_keys as f64,
        cleanup_dropped_backup_keys: stats.cleanup_dropped_backup_keys as f64,
    }
}

/// A report as one rank shares it.
fn encode(report: &Report) -> Vec<u8> {
    let mut out = Writer::new();
    out.i64(report.now_us);
    out.i64(report.pressure.queued_operations);
    out.i64(report.pressure.queued_bytes);
    out.i64(report.pressure.oldest_operation_us);
    out.option(report.error.as_deref(), |out, error| {
        out.bytes(error.as_bytes())
    });
    out.count(report.completions.len());
    for completion in &report.completions {
        out.u64(completion.op_id);
        out.flag(completion.retired);
        out.flags(&completion.success);
        out.flags(&completion.object_failures);
    }
    out.finish()
}

fn decode(bytes: &[u8]) -> Option<Report> {
    let mut reader = Reader::new(bytes);
    let now_us = reader.i64()?;
    let pressure = StoragePressure {
        queued_operations: reader.i64()?,
        queued_bytes: reader.i64()?,
        oldest_operation_us: reader.i64()?,
    };
    let error = reader.option(|reader| String::from_utf8(reader.bytes()?.to_vec()).ok())?;
    let count = reader.count()?;
    let completions = (0..count)
        .map(|_| {
            Some(StorageCompletion {
                op_id: reader.u64()?,
                retired: reader.flag()?,
                success: reader.flags()?,
                object_failures: reader.flags()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    reader.finish(Report {
        now_us,
        pressure,
        error,
        completions,
    })
}

#[cfg(test)]
mod tests {
    use std::thread;

    use ts_collective::{Group, Listener, Mesh};

    use super::*;

    const LIMITS: StorageLimits = StorageLimits {
        max_operations: 8,
        max_keys_per_operation: 4,
    };

    fn completion(
        op_id: u64,
        success: &[bool],
        retired: bool,
        failed: &[bool],
    ) -> StorageCompletion {
        StorageCompletion {
            op_id,
            success: success.to_vec(),
            retired,
            object_failures: failed.to_vec(),
        }
    }

    fn poll(now_us: i64, completions: Vec<StorageCompletion>) -> StoragePoll {
        StoragePoll {
            now_us,
            completions,
            ..StoragePoll::default()
        }
    }

    /// Run `polls` on every rank of a replica of `polls.len()`, exchanging
    /// the reports over a real mesh as the round's relay does; each rank's
    /// round outcomes.
    fn replica(polls: Vec<Vec<StoragePoll>>) -> Vec<Vec<Result<StorageRound, String>>> {
        let listeners: Vec<_> = polls
            .iter()
            .map(|_| Listener::bind("127.0.0.1").unwrap())
            .collect();
        let addresses: Vec<String> = listeners
            .iter()
            .map(|listener| format!("127.0.0.1:{}", listener.port().unwrap()))
            .collect();
        let size = polls.len();
        let threads: Vec<_> = listeners
            .into_iter()
            .zip(polls)
            .enumerate()
            .map(|(rank, (listener, rounds))| {
                let addresses = addresses.clone();
                thread::spawn(move || {
                    let mut mesh =
                        Mesh::connect(rank, listener, &addresses, Duration::from_secs(5)).unwrap();
                    let group = Group::new(0, (0..size).collect(), rank).unwrap();
                    let mut l3 = L3::new(LIMITS);
                    let mut outcomes = Vec::new();
                    for poll in rounds {
                        let reports = mesh.all_gather(&group, &l3.report(poll)).unwrap();
                        let outcome = l3
                            .agree(&reports, true, true)
                            .map_err(|error| error.to_string());
                        let failed = outcome.is_err();
                        outcomes.push(outcome);
                        if failed {
                            break;
                        }
                    }
                    outcomes
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    }

    #[test]
    fn owners_complete_out_of_order_and_converge_on_a_fault() {
        let rank0 = vec![
            poll(
                10,
                vec![completion(1, &[true, true], true, &[false, false])],
            ),
            poll(
                20,
                vec![completion(2, &[true, true], true, &[false, false])],
            ),
            poll(30, vec![]),
            // An op id 0 completion: malformed.
            poll(40, vec![completion(0, &[true], true, &[false])]),
        ];
        let rank1 = vec![
            poll(
                11,
                vec![completion(2, &[true, false], true, &[false, true])],
            ),
            poll(21, vec![]),
            poll(
                31,
                vec![completion(1, &[false, false], false, &[false, false])],
            ),
            poll(41, vec![]),
        ];
        for outcomes in replica(vec![rank0, rank1]) {
            // Both operations incomplete: only the common tick, rank 0's clock.
            let first = outcomes[0].as_ref().unwrap();
            assert_eq!(first.tick.unwrap().now_us, 10);
            assert!(first.done.is_empty());
            let second = outcomes[1].as_ref().unwrap();
            assert_eq!(
                second.done,
                [StorageDoneView {
                    op_id: 2,
                    success: vec![1, 0],
                    retired: true,
                    object_failures: vec![0, 1],
                }]
            );
            let third = outcomes[2].as_ref().unwrap();
            assert_eq!(third.done.len(), 1);
            assert_eq!(third.done[0].op_id, 1);
            assert!(!third.done[0].retired);
            // Every rank observes the peer's polling failure.
            let failure = outcomes[3].as_ref().unwrap_err();
            assert!(failure.contains("Malformed L3 completion"), "{failure}");
        }
    }

    #[test]
    fn idle_rounds_agree_on_the_first_members_clock() {
        let rounds = |rank: usize| {
            (0..3)
                .map(|round| poll(100 * round + rank as i64, vec![]))
                .collect::<Vec<_>>()
        };
        for outcomes in replica((0..3).map(rounds).collect()) {
            let clocks: Vec<i64> = outcomes
                .into_iter()
                .map(|outcome| outcome.unwrap().tick.unwrap().now_us)
                .collect();
            assert_eq!(clocks, [0, 100, 200]);
        }
    }

    #[test]
    fn the_tick_carries_the_permissions_it_is_given() {
        let mut l3 = L3::new(LIMITS);
        let reports = [l3.report(poll(5, vec![]))];
        let tick = l3.agree(&reports, false, true).unwrap().tick.unwrap();
        assert!(!tick.allow_forward && tick.allow_storage);
        let tick = l3.agree(&reports, true, false).unwrap().tick.unwrap();
        assert!(tick.allow_forward && !tick.allow_storage);
    }

    #[test]
    fn the_worst_owner_pressure_wins_and_an_invalid_one_fails_every_rank() {
        let pressured = |queued_operations, queued_bytes, oldest_operation_us| StoragePoll {
            pressure: StoragePressure {
                queued_operations,
                queued_bytes,
                oldest_operation_us,
            },
            ..StoragePoll::default()
        };
        let rank0 = vec![pressured(9, 0, 0), pressured(0, 0, 0), pressured(-1, 0, 0)];
        let rank1 = vec![
            pressured(0, 4096, 700_000),
            pressured(0, 0, 0),
            pressured(0, 0, 0),
        ];
        for outcomes in replica(vec![rank0, rank1]) {
            let tick = outcomes[0].as_ref().unwrap().tick.unwrap();
            assert_eq!(
                (
                    tick.queued_operations,
                    tick.queued_bytes,
                    tick.oldest_operation_us
                ),
                (9, 4096, 700_000)
            );
            let tick = outcomes[1].as_ref().unwrap().tick.unwrap();
            assert_eq!(
                (
                    tick.queued_operations,
                    tick.queued_bytes,
                    tick.oldest_operation_us
                ),
                (0, 0, 0)
            );
            let failure = outcomes[2].as_ref().unwrap_err();
            assert!(
                failure.contains("Invalid L3 pressure observation"),
                "{failure}"
            );
        }
    }

    #[test]
    fn a_malformed_completion_fails_every_rank_before_publication() {
        for bad in [
            completion(7, &[true], true, &[]),
            completion(7, &[true], true, &[true]),
            completion(0, &[true], true, &[false]),
            completion(7, &[true; 5], true, &[false; 5]),
        ] {
            let outcomes = replica(vec![vec![poll(1, vec![bad])], vec![poll(1, vec![])]]);
            for outcome in outcomes {
                let failure = outcome[0].as_ref().unwrap_err();
                assert!(failure.contains("L3 replica polling failed"), "{failure}");
            }
        }
    }

    #[test]
    fn a_rejection_is_not_a_confirmed_peer_miss() {
        let outcomes = replica(vec![
            vec![poll(
                1,
                vec![completion(
                    1,
                    &[false, false, true],
                    true,
                    &[false, false, false],
                )],
            )],
            vec![poll(
                1,
                vec![completion(
                    1,
                    &[true, false, true],
                    true,
                    &[false, true, false],
                )],
            )],
        ]);
        for outcome in outcomes {
            let done = &outcome[0].as_ref().unwrap().done[0];
            assert_eq!(done.success, [0, 0, 1]);
            assert_eq!(done.object_failures, [0, 1, 0]);
            assert!(done.retired);
        }
    }

    #[test]
    fn a_conflicting_owner_report_fails_every_rank() {
        // Rank 1 reports operation 5 twice, differently, while rank 0 has
        // not finished it: the held report and the new one disagree.
        let outcomes = replica(vec![
            vec![poll(1, vec![]), poll(2, vec![])],
            vec![
                poll(1, vec![completion(5, &[true], true, &[false])]),
                poll(2, vec![completion(5, &[false], true, &[false])]),
            ],
        ]);
        for outcome in outcomes {
            assert!(outcome[0].as_ref().unwrap().done.is_empty());
            let failure = outcome[1].as_ref().unwrap_err();
            assert!(failure.contains("conflicting"), "{failure}");
        }
    }

    #[test]
    fn reports_round_trip_their_encoding() {
        let report = Report {
            now_us: -3,
            pressure: StoragePressure {
                queued_operations: 1,
                queued_bytes: 2_i64.pow(33),
                oldest_operation_us: 7,
            },
            error: Some("boom".to_owned()),
            completions: vec![completion(9, &[true, false], false, &[false, true])],
        };
        assert_eq!(decode(&encode(&report)), Some(report.clone()));
        let mut cut = encode(&report);
        cut.pop();
        assert_eq!(decode(&cut), None);
        // A report that does not decode fails the round.
        let mut l3 = L3::new(LIMITS);
        let failure = l3.agree(&[cut], true, true).unwrap_err();
        assert!(failure.to_string().contains("malformed"), "{failure}");
    }
}
