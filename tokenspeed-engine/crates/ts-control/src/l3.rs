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

//! The L3 storage tier as the loop sees it: the replica's agreement on its
//! storage executors' reports, every round (`docs/design/async-l3-cache.md`).
//!
//! The C++ scheduler owns readiness, deadlines and Host memory lifetime; it
//! plans storage operations (query, get, put, cancel) that each cache owner's
//! data plane runs on its Host-only executor. The ranks of a replica mirror
//! one scheduler, so they agree each round on the storage clock and the
//! permissions to run forwards and start storage work, and on each
//! operation's outcome once every owner reported it. A fault on one rank,
//! even in its own poll, fails every rank in the same round instead of
//! stranding peers in the next collective.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use ts_collective::{Group, Mesh};
use ts_metrics::L3SchedulerStats;
use ts_scheduler::{
    Scheduler, StorageCancellationView, StorageDoneView, StorageRound, StorageStatsView,
    StorageTickView,
};

/// The scheduler's storage credits (`StorageConfig`) the agreement checks
/// reports against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLimits {
    /// Operations in flight.
    pub max_operations: usize,
    /// Keys one operation names.
    pub max_keys_per_operation: usize,
}

/// One operation this rank's executor finished (`StorageCompletion`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageCompletion {
    /// The operation.
    pub op_id: u64,
    /// Per key: it succeeded here.
    pub success: Vec<bool>,
    /// The native I/O retired: nothing still touches its Host memory.
    pub retired: bool,
    /// Per key: the object is confirmed missing or invalid here.
    pub object_failures: Vec<bool>,
}

/// The executor's queue pressure (`StoragePressure`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoragePressure {
    /// Operations queued.
    pub queued_operations: i64,
    /// Bytes queued.
    pub queued_bytes: i64,
    /// The oldest queued operation's age, in microseconds.
    pub oldest_operation_us: i64,
}

/// One poll of this rank's storage executor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoragePoll {
    /// The storage clock, in microseconds; rank 0's is the replica's.
    pub now_us: i64,
    /// Operations finished since the last poll.
    pub completions: Vec<StorageCompletion>,
    /// Queue pressure.
    pub pressure: StoragePressure,
    /// Why the poll failed here, if it did: every rank still joins the
    /// agreement, and every rank fails with it.
    pub error: Option<String>,
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
    /// The ranks could not reach each other.
    #[error(transparent)]
    Collective(#[from] ts_collective::Error),
}

/// What one rank shares each round.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Report {
    now_us: i64,
    allow_forward: bool,
    allow_storage: bool,
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

    /// One round's agreement over `replica`: the clock, the permissions and
    /// the worst pressure, the queued cancellations, and the operations every
    /// owner has now completed (success and retirement ANDed, object failures
    /// ORed across owners). Every member calls this every round.
    pub fn round(
        &mut self,
        poll: StoragePoll,
        allow_forward: bool,
        allow_storage: bool,
        mesh: &mut Mesh,
        replica: &Group,
    ) -> Result<StorageRound, L3Error> {
        let local = self.validated(poll, allow_forward, allow_storage);
        let reports = if replica.size() == 1 {
            vec![local]
        } else {
            mesh.all_gather(replica, &encode(&local))?
                .iter()
                .map(|bytes| decode(bytes).ok_or(L3Error::Protocol("a malformed owner report")))
                .collect::<Result<Vec<_>, _>>()?
        };
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
            allow_forward: reports.iter().all(|report| report.allow_forward),
            allow_storage: reports.iter().all(|report| report.allow_storage),
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

    /// The report this rank shares: its poll, or why the poll is invalid.
    /// A bad record must not strand peers already exchanging reports, so it
    /// becomes the report's error instead of a local failure.
    fn validated(&self, poll: StoragePoll, allow_forward: bool, allow_storage: bool) -> Report {
        let StoragePoll {
            now_us,
            completions,
            pressure,
            error,
        } = poll;
        let invalid = if error.is_some() {
            error
        } else if [
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
                allow_forward,
                allow_storage,
                pressure: StoragePressure::default(),
                error: Some(error.chars().take(512).collect()),
                completions: Vec::new(),
            },
            None => Report {
                now_us,
                allow_forward,
                allow_storage,
                pressure,
                error: None,
                completions,
            },
        }
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

fn put_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_flags(out: &mut Vec<u8>, flags: &[bool]) {
    out.extend_from_slice(&(flags.len() as u32).to_le_bytes());
    out.extend(flags.iter().map(|&flag| u8::from(flag)));
}

/// A report as one rank shares it.
fn encode(report: &Report) -> Vec<u8> {
    let mut out = Vec::new();
    put_i64(&mut out, report.now_us);
    out.push(u8::from(report.allow_forward));
    out.push(u8::from(report.allow_storage));
    put_i64(&mut out, report.pressure.queued_operations);
    put_i64(&mut out, report.pressure.queued_bytes);
    put_i64(&mut out, report.pressure.oldest_operation_us);
    let error = report.error.as_deref().unwrap_or("");
    out.push(u8::from(report.error.is_some()));
    out.extend_from_slice(&(error.len() as u32).to_le_bytes());
    out.extend_from_slice(error.as_bytes());
    out.extend_from_slice(&(report.completions.len() as u32).to_le_bytes());
    for completion in &report.completions {
        out.extend_from_slice(&completion.op_id.to_le_bytes());
        out.push(u8::from(completion.retired));
        put_flags(&mut out, &completion.success);
        put_flags(&mut out, &completion.object_failures);
    }
    out
}

/// A cursor over a shared report.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, len: usize) -> Option<&[u8]> {
        if self.0.len() < len {
            return None;
        }
        let (head, rest) = self.0.split_at(len);
        self.0 = rest;
        Some(head)
    }

    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|bytes| bytes[0])
    }

    fn flag(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn u32(&mut self) -> Option<usize> {
        let value = u32::from_le_bytes(self.take(4)?.try_into().ok()?);
        usize::try_from(value).ok()
    }

    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn flags(&mut self) -> Option<Vec<bool>> {
        let len = self.u32()?;
        (0..len).map(|_| self.flag()).collect()
    }
}

fn decode(bytes: &[u8]) -> Option<Report> {
    let mut reader = Reader(bytes);
    let now_us = reader.i64()?;
    let allow_forward = reader.flag()?;
    let allow_storage = reader.flag()?;
    let pressure = StoragePressure {
        queued_operations: reader.i64()?,
        queued_bytes: reader.i64()?,
        oldest_operation_us: reader.i64()?,
    };
    let has_error = reader.flag()?;
    let error_len = reader.u32()?;
    let error = String::from_utf8(reader.take(error_len)?.to_vec()).ok()?;
    let count = reader.u32()?;
    let completions = (0..count)
        .map(|_| {
            let op_id = reader.u64()?;
            let retired = reader.flag()?;
            Some(StorageCompletion {
                op_id,
                retired,
                success: reader.flags()?,
                object_failures: reader.flags()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    reader.0.is_empty().then_some(Report {
        now_us,
        allow_forward,
        allow_storage,
        pressure,
        error: has_error.then_some(error),
        completions,
    })
}

#[cfg(test)]
mod tests {
    use std::thread;

    use ts_collective::Listener;

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

    /// Run `rounds` on every rank of a replica of `polls.len()`; each rank's
    /// round outcomes.
    fn replica(
        polls: Vec<Vec<(StoragePoll, bool, bool)>>,
    ) -> Vec<Vec<Result<StorageRound, String>>> {
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
                    for (poll, allow_forward, allow_storage) in rounds {
                        let outcome = l3
                            .round(poll, allow_forward, allow_storage, &mut mesh, &group)
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
            (
                poll(
                    10,
                    vec![completion(1, &[true, true], true, &[false, false])],
                ),
                true,
                true,
            ),
            (
                poll(
                    20,
                    vec![completion(2, &[true, true], true, &[false, false])],
                ),
                true,
                true,
            ),
            (poll(30, vec![]), true, true),
            (
                StoragePoll {
                    error: Some("injected local poll fault".to_owned()),
                    ..poll(40, vec![])
                },
                false,
                false,
            ),
        ];
        let rank1 = vec![
            (
                poll(
                    11,
                    vec![completion(2, &[true, false], true, &[false, true])],
                ),
                true,
                true,
            ),
            (poll(21, vec![]), true, true),
            (
                poll(
                    31,
                    vec![completion(1, &[false, false], false, &[false, false])],
                ),
                true,
                true,
            ),
            (poll(41, vec![]), false, false),
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
            assert!(failure.contains("injected local poll fault"), "{failure}");
        }
    }

    #[test]
    fn idle_rounds_agree_the_clock_and_the_permissions() {
        let rounds = |rank: usize| {
            (0..3)
                .map(|round| {
                    (
                        poll(100 * round + rank as i64, vec![]),
                        rank == 0,
                        rank == 1,
                    )
                })
                .collect::<Vec<_>>()
        };
        for outcomes in replica((0..3).map(rounds).collect()) {
            let mut previous = -1;
            for outcome in outcomes {
                let tick = outcome.unwrap().tick.unwrap();
                assert!(tick.now_us > previous);
                assert!(!tick.allow_forward && !tick.allow_storage);
                previous = tick.now_us;
            }
        }
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
        let rank0 = vec![
            (pressured(9, 0, 0), true, true),
            (pressured(0, 0, 0), true, true),
            (pressured(-1, 0, 0), true, true),
        ];
        let rank1 = vec![
            (pressured(0, 4096, 700_000), true, true),
            (pressured(0, 0, 0), true, true),
            (pressured(0, 0, 0), true, true),
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
            let outcomes = replica(vec![
                vec![(poll(1, vec![bad]), true, true)],
                vec![(poll(1, vec![]), true, true)],
            ]);
            for outcome in outcomes {
                let failure = outcome[0].as_ref().unwrap_err();
                assert!(failure.contains("L3 replica polling failed"), "{failure}");
            }
        }
    }

    #[test]
    fn a_rejection_is_not_a_confirmed_peer_miss() {
        let outcomes = replica(vec![
            vec![(
                poll(
                    1,
                    vec![completion(
                        1,
                        &[false, false, true],
                        true,
                        &[false, false, false],
                    )],
                ),
                true,
                true,
            )],
            vec![(
                poll(
                    1,
                    vec![completion(
                        1,
                        &[true, false, true],
                        true,
                        &[false, true, false],
                    )],
                ),
                true,
                true,
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
            vec![(poll(1, vec![]), true, true), (poll(2, vec![]), true, true)],
            vec![
                (
                    poll(1, vec![completion(5, &[true], true, &[false])]),
                    true,
                    true,
                ),
                (
                    poll(2, vec![completion(5, &[false], true, &[false])]),
                    true,
                    true,
                ),
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
            allow_forward: true,
            allow_storage: false,
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
    }
}
