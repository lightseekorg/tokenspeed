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

//! The Disk tier as the loop sees it (`docs/rfcs/0006-disk-tier.md`): this
//! rank's disk executor and the replica's agreement on what the executors
//! finished, every round.
//!
//! The C++ scheduler plans each Disk operation by block ids: a cleaning
//! write from Host blocks to Disk blocks, a prefetch read the other way.
//! Each cache owner runs its rows on its own file and Host pool (`ts-disk`),
//! started with the plan. The ranks of a replica mirror one scheduler, so
//! each round their reports ride the relay, and an operation completes once
//! every owner finished it, successfully only if every owner did. The
//! storage clock is the first member's; it times the scheduler's prefetch
//! deadlines, and the permissions come from the control state.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use ts_disk::{Direction, DiskFile, Executor, ExecutorConfig, Io, Job, Memory, ALIGNMENT};
use ts_l3::{HostBuffer, HostLayout};
use ts_metrics::{DiskExecutorStats, DiskSchedulerStats};
use ts_scheduler::{
    DiskDoneView, DiskOpKind, DiskStatsView, ExecutionPlan, StorageRound, StorageTickView,
};

use crate::codec::{Reader, Writer};

/// Everything a rank's Disk tier starts from, as the data plane resolves it.
#[derive(Debug, Clone)]
pub struct DiskSettings {
    /// This rank's file.
    pub file: PathBuf,
    /// Its size; at least the Disk pool's.
    pub file_bytes: u64,
    /// How the executor runs.
    pub executor: ExecutorConfig,
    /// The Host pool's bytes.
    pub host: HostBuffer,
    /// The Host pool's geometry and this rank's place among the owners.
    pub host_layout: HostLayout,
    /// The Disk pool's geometry: the Host's packing over the Disk's parents.
    pub disk_layout: HostLayout,
    /// The scheduler's operation credits (`DiskConfig::max_operations`).
    pub max_operations: usize,
    /// With peer sharing, the bytes of the staging ring this rank's pushes
    /// read Disk blocks into (`peer::push`); 0 without. Rounded up to whole
    /// slots of the largest block.
    pub push_ring_bytes: usize,
}

/// A Disk tier that cannot start, or an operation it cannot run.
#[derive(Debug, thiserror::Error)]
pub enum DiskError {
    /// The settings are inconsistent.
    #[error("Disk tier settings: {0}")]
    Settings(String),
    /// The file did not open or allocate.
    #[error(transparent)]
    File(#[from] ts_disk::FileError),
    /// The executor did not start.
    #[error(transparent)]
    Start(#[from] ts_disk::StartError),
    /// A plan's operation names blocks the layouts cannot place.
    #[error("a Disk operation's block: {0}")]
    Layout(#[from] ts_l3::LayoutError),
    /// A plan's operation is inconsistent.
    #[error("a Disk operation: {0}")]
    Operation(&'static str),
    /// The executor refused a job.
    #[error(transparent)]
    Submit(#[from] ts_disk::SubmitError),
    /// The replica's reports break the protocol.
    #[error("Disk agreement: {0}")]
    Protocol(&'static str),
}

/// What one rank shares each round: its clock and the operations it
/// finished since its last report, with whether every row of each moved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Report {
    now_us: i64,
    finished: Vec<(u64, bool)>,
}

impl Report {
    fn encode(&self) -> Vec<u8> {
        let mut out = Writer::new();
        out.i64(self.now_us);
        out.count(self.finished.len());
        for &(op_id, success) in &self.finished {
            out.u64(op_id);
            out.flag(success);
        }
        out.finish()
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut input = Reader::new(bytes);
        let now_us = input.i64()?;
        let count = input.count()?;
        let mut finished = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            finished.push((input.u64()?, input.flag()?));
        }
        input.finish(Self { now_us, finished })
    }
}

/// The replica's agreement on its members' reports, with the reports of
/// operations not every owner finished yet.
#[derive(Debug)]
struct Agreement {
    max_operations: usize,
    /// Per operation, each owner's outcome until every owner reported.
    reports: BTreeMap<u64, BTreeMap<usize, bool>>,
}

impl Agreement {
    fn agree(
        &mut self,
        reports: &[Vec<u8>],
        allow_forward: bool,
        allow_storage: bool,
    ) -> Result<StorageRound, DiskError> {
        let reports = reports
            .iter()
            .map(|bytes| Report::decode(bytes).ok_or(DiskError::Protocol("a malformed report")))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(first) = reports.first() else {
            return Err(DiskError::Protocol("a round without reports"));
        };
        for (rank, report) in reports.iter().enumerate() {
            for &(op_id, success) in &report.finished {
                let row = self.reports.entry(op_id).or_default();
                if row.insert(rank, success).is_some() {
                    return Err(DiskError::Protocol("an owner reported an operation twice"));
                }
            }
        }
        if self.reports.len() > self.max_operations {
            return Err(DiskError::Protocol("more operations than the credits"));
        }
        let owners = reports.len();
        let finished: Vec<u64> = self
            .reports
            .iter()
            .filter(|(_, row)| row.len() == owners)
            .map(|(&op_id, _)| op_id)
            .collect();
        let disk_done = finished
            .into_iter()
            .map(|op_id| {
                let row = self.reports.remove(&op_id).expect("a finished operation");
                DiskDoneView {
                    op_id,
                    success: row.values().all(|&success| success),
                }
            })
            .collect();
        Ok(StorageRound {
            tick: Some(StorageTickView {
                now_us: first.now_us,
                allow_forward,
                allow_storage,
                queued_operations: 0,
                queued_bytes: 0,
                oldest_operation_us: 0,
            }),
            cancellations: Vec::new(),
            done: Vec::new(),
            disk_done,
        })
    }
}

/// This rank's Disk tier: the executor and the replica's agreement.
pub struct DiskTier {
    /// Shared with the rank's peer pushes, which read into `ring`.
    executor: Arc<Executor>,
    ring: Option<Arc<Memory>>,
    host: HostBuffer,
    host_layout: HostLayout,
    disk_layout: HostLayout,
    /// Operations this rank finished since its last report; executor
    /// threads append to it.
    finished: Arc<Mutex<Vec<(u64, bool)>>>,
    agreement: Agreement,
    clock: Instant,
    /// When the metrics were last sampled, and whether Disk work was pending.
    metrics_sampled: Option<(Instant, bool)>,
}

impl DiskTier {
    /// Open the file and start the executor over the Host pool.
    pub fn start(settings: &DiskSettings) -> Result<Self, DiskError> {
        let alignment = ALIGNMENT;
        let host = settings.host;
        let aligned = |value: usize| value % alignment == 0;
        if !aligned(host.base) || host.len != settings.host_layout.bytes() {
            return Err(DiskError::Settings(format!(
                "the Host pool at {:#x}, {} bytes, is unaligned or does not match its layout ({} bytes)",
                host.base,
                host.len,
                settings.host_layout.bytes()
            )));
        }
        if settings.disk_layout.groups != settings.host_layout.groups
            || settings.disk_layout.lcm_block_bytes != settings.host_layout.lcm_block_bytes
        {
            return Err(DiskError::Settings(
                "the Disk pool must pack like the Host pool".to_owned(),
            ));
        }
        if !aligned(settings.host_layout.lcm_block_bytes)
            || settings
                .host_layout
                .groups
                .iter()
                .any(|group| !aligned(group.block_bytes))
        {
            return Err(DiskError::Settings(format!(
                "every Host block and parent must be a multiple of {alignment} bytes for direct I/O"
            )));
        }
        if (settings.disk_layout.bytes() as u64) > settings.file_bytes {
            return Err(DiskError::Settings(format!(
                "the {}-byte file is smaller than the {}-byte Disk pool",
                settings.file_bytes,
                settings.disk_layout.bytes()
            )));
        }
        let file = DiskFile::create(&settings.file, settings.file_bytes)?;
        let ring = (settings.push_ring_bytes > 0)
            .then(|| {
                let slot = push_slot_bytes(&settings.host_layout);
                Memory::map(settings.push_ring_bytes.div_ceil(slot) * slot)
            })
            .transpose()
            .map_err(|error| DiskError::Settings(format!("the push staging ring: {error}")))?
            .map(Arc::new);
        let mut regions = vec![(host.base, host.len)];
        regions.extend(ring.iter().map(|ring| (ring.address(), ring.size())));
        let executor = Arc::new(Executor::start(file, &regions, settings.executor)?);
        tracing::info!(
            file = %settings.file.display(),
            bytes = settings.file_bytes,
            backend = ?settings.executor.backend,
            "Disk tier started"
        );
        Ok(Self {
            executor,
            ring,
            host,
            host_layout: settings.host_layout.clone(),
            disk_layout: settings.disk_layout.clone(),
            finished: Arc::new(Mutex::new(Vec::new())),
            agreement: Agreement {
                max_operations: settings.max_operations,
                reports: BTreeMap::new(),
            },
            clock: Instant::now(),
            metrics_sampled: None,
        })
    }

    /// Start `plan`'s Disk operations on this rank; none waits for I/O.
    /// Returns whether the plan had any.
    pub fn submit(&self, plan: &ExecutionPlan) -> Result<bool, DiskError> {
        let ops = plan.disk_ops();
        for op in &ops {
            if op.group_ids.len() != op.host_blocks.len()
                || op.group_ids.len() != op.disk_blocks.len()
            {
                return Err(DiskError::Operation("its columns disagree"));
            }
            let mut ios = Vec::with_capacity(op.group_ids.len());
            for row in 0..op.group_ids.len() {
                let group = op.group_ids[row];
                let host = self.host_layout.owned_range(group, op.host_blocks[row])?;
                let disk = self.disk_layout.owned_range(group, op.disk_blocks[row])?;
                match (host, disk) {
                    (Some(host), Some(disk)) => ios.push(Io {
                        file_offset: disk.offset as u64,
                        address: self.host.address(host),
                        len: host.size,
                    }),
                    // Another owner's row of a page-sharded group.
                    (None, None) => {}
                    _ => {
                        return Err(DiskError::Operation(
                            "a row's Host and Disk blocks have different owners",
                        ))
                    }
                }
            }
            let op_id = op.op_id;
            if ios.is_empty() {
                self.finished.lock().push((op_id, true));
                continue;
            }
            let direction = match op.kind {
                DiskOpKind::Read => Direction::Read,
                DiskOpKind::Write => Direction::Write,
                _ => return Err(DiskError::Operation("its kind")),
            };
            let finished = self.finished.clone();
            self.executor
                .submit(Job { direction, ios }, move |outcome| {
                    if let Err(error) = &outcome {
                        tracing::warn!(op_id, %error, "Disk operation failed");
                    }
                    finished.lock().push((op_id, outcome.is_ok()));
                })?;
        }
        Ok(!ops.is_empty())
    }

    /// This round's report: the clock and what finished since the last.
    pub fn report(&self) -> Vec<u8> {
        Report {
            now_us: i64::try_from(self.clock.elapsed().as_micros()).unwrap_or(i64::MAX),
            finished: std::mem::take(&mut *self.finished.lock()),
        }
        .encode()
    }

    /// The round's agreement on every replica member's [`DiskTier::report`],
    /// in member order: the first member's clock, the permissions, and the
    /// operations every owner has now finished. Every member calls this
    /// every round.
    pub fn agree(
        &mut self,
        reports: &[Vec<u8>],
        allow_forward: bool,
        allow_storage: bool,
    ) -> Result<StorageRound, DiskError> {
        self.agreement.agree(reports, allow_forward, allow_storage)
    }

    /// The executor's queues and cumulative totals, as metrics.
    pub fn executor_stats(&self) -> DiskExecutorStats {
        let pressure = self.executor.pressure();
        let stats = self.executor.stats();
        DiskExecutorStats {
            queued_read_bytes: pressure.queued_read_bytes as f64,
            queued_write_bytes: pressure.queued_write_bytes as f64,
            inflight_bytes: pressure.inflight_bytes as f64,
            read_bytes: stats.read_bytes as f64,
            write_bytes: stats.write_bytes as f64,
            failed_jobs: stats.failed_jobs as f64,
        }
    }

    /// Whether to sample the metrics now: at most once a second, and
    /// whenever Disk work turns pending or idle.
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

    /// What this rank's peer pushes read from: the executor, the file's
    /// layout, the staging ring and its slot size. None without a ring.
    pub fn push_source(&self) -> Option<(Arc<Executor>, HostLayout, Arc<Memory>, usize)> {
        self.ring.as_ref().map(|ring| {
            (
                self.executor.clone(),
                self.disk_layout.clone(),
                ring.clone(),
                push_slot_bytes(&self.host_layout),
            )
        })
    }

    /// Cancel what is queued and wait for what runs to retire.
    pub fn close(&mut self, timeout: Duration) -> Result<(), ts_disk::executor::CloseError> {
        self.executor.close(timeout)
    }
}

/// A push staging slot: the largest block, already a multiple of the
/// direct-I/O alignment (`DiskTier::start` checks).
fn push_slot_bytes(layout: &HostLayout) -> usize {
    layout
        .groups
        .iter()
        .map(|group| group.block_bytes)
        .max()
        .unwrap_or(ALIGNMENT)
        .max(ALIGNMENT)
}

/// The scheduler's Disk observations, as metrics.
pub fn scheduler_stats(stats: DiskStatsView) -> DiskSchedulerStats {
    DiskSchedulerStats {
        operations: stats.operations as f64,
        held_host_blocks: stats.held_host_blocks as f64,
        queued_cleanings: stats.queued_cleanings as f64,
        waiting_requests: stats.waiting_requests as f64,
        ready_requests: stats.ready_requests as f64,
        disk_blocks: stats.disk_blocks as f64,
        free_disk_parents: stats.free_disk_parents as f64,
        written_blocks: stats.written_blocks as f64,
        read_blocks: stats.read_blocks as f64,
        failed_reads: stats.failed_reads as f64,
        failed_writes: stats.failed_writes as f64,
        dropped_cleanings: stats.dropped_cleanings as f64,
        wait_timeouts: stats.wait_timeouts as f64,
        ready_expirations: stats.ready_expirations as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(now_us: i64, finished: &[(u64, bool)]) -> Vec<u8> {
        Report {
            now_us,
            finished: finished.to_vec(),
        }
        .encode()
    }

    #[test]
    fn an_operation_completes_once_every_owner_finished_it() {
        let mut agreement = Agreement {
            max_operations: 8,
            reports: BTreeMap::new(),
        };
        let first = agreement
            .agree(&[report(5, &[(1, true)]), report(9, &[])], true, true)
            .unwrap();
        assert_eq!(first.tick.unwrap().now_us, 5, "the first member's clock");
        assert!(first.disk_done.is_empty());
        let second = agreement
            .agree(
                &[
                    report(6, &[(2, true)]),
                    report(10, &[(1, true), (2, false)]),
                ],
                true,
                true,
            )
            .unwrap();
        assert_eq!(
            second.disk_done,
            vec![
                DiskDoneView {
                    op_id: 1,
                    success: true
                },
                DiskDoneView {
                    op_id: 2,
                    success: false
                }
            ]
        );
    }

    #[test]
    fn a_repeated_or_malformed_report_breaks_the_round() {
        let mut agreement = Agreement {
            max_operations: 8,
            reports: BTreeMap::new(),
        };
        agreement
            .agree(&[report(1, &[(3, true)]), report(1, &[])], true, true)
            .unwrap();
        assert!(agreement
            .agree(&[report(2, &[(3, true)]), report(2, &[])], true, true)
            .is_err());
        assert!(agreement.agree(&[vec![1, 2, 3]], true, true).is_err());
    }

    #[test]
    fn reports_round_trip() {
        let bytes = report(-1, &[(7, true), (u64::MAX, false)]);
        assert_eq!(
            Report::decode(&bytes),
            Some(Report {
                now_us: -1,
                finished: vec![(7, true), (u64::MAX, false)]
            })
        );
    }
}
