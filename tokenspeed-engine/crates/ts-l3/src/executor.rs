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

//! The executor: bounded queues of storage jobs and the worker threads that
//! run them against the store (`docs/design/async-l3-cache.md`).
//!
//! The scheduler plans each operation and holds its Host blocks until every
//! cache owner proves the native I/O retired; the executor only runs what it
//! is given, in a bounded window, and reports per-key facts. It never calls
//! the scheduler, a collective or the GPU. A job that cannot run (a full
//! queue, a backend held open after failures, shutdown) completes at once as
//! a retired failure; a backend that cannot prove retirement disables new
//! I/O on this rank until restart.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use ts_metrics::{L3ExecutorSnapshot, L3OperationSnapshot};

use crate::{
    keys,
    layout::{HostLayout, LayoutError, Range},
    store::{HostBuffer, IoResult, Store},
};

/// The executor's limits: the scheduler's storage credits and the workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorConfig {
    /// Operations the scheduler keeps in flight (`StorageConfig`).
    pub max_operations: usize,
    /// Keys one operation names (`StorageConfig`).
    pub max_keys_per_operation: usize,
    /// Threads running existence queries.
    pub query_workers: usize,
    /// Threads running gets.
    pub read_workers: usize,
    /// Threads running puts.
    pub write_workers: usize,
    /// Failed backend calls that open the backend.
    pub failure_threshold: u32,
    /// How long an open backend refuses calls before one probe.
    pub recovery_interval: Duration,
}

/// A monotonic clock, in seconds.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// The process's monotonic clock, in seconds since its first reading.
pub fn monotonic_clock() -> Clock {
    let start = Instant::now();
    Arc::new(move || start.elapsed().as_secs_f64())
}

/// A content-addressed key of a scheduler storage operation (`CacheKey`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageKey {
    /// The cache group.
    pub group: u32,
    /// The prefix page's content hash.
    pub content_hash: String,
    /// The block's ordinal within its prefix page.
    pub page_offset: i32,
}

/// A storage operation a plan starts (`StorageOperation`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    /// Which keys the store holds.
    Query {
        /// Its id.
        op_id: u64,
        /// The keys.
        keys: Vec<StorageKey>,
    },
    /// Fetch keys into Host blocks.
    Get {
        /// Its id.
        op_id: u64,
        /// The keys.
        keys: Vec<StorageKey>,
        /// Per key, the destination Host block.
        host_blocks: Vec<i32>,
    },
    /// Store Host blocks under keys.
    Put {
        /// Its id.
        op_id: u64,
        /// The keys.
        keys: Vec<StorageKey>,
        /// Per key, the source Host block.
        host_blocks: Vec<i32>,
    },
    /// Cancel an earlier operation; it gets no completion of its own.
    Cancel {
        /// The operation cancelled.
        op_id: u64,
    },
}

/// One operation this rank finished (`StorageCompletion`).
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
    /// The oldest operation not yet finished, its age in microseconds.
    pub oldest_operation_us: i64,
}

/// One poll of this rank's storage executor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoragePoll {
    /// The storage clock, in microseconds; the replica takes rank 0's.
    pub now_us: i64,
    /// The operations finished since the last poll.
    pub completions: Vec<StorageCompletion>,
    /// The queues now.
    pub pressure: StoragePressure,
}

/// A fault in what the executor was given.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    /// A key or block the Host layout cannot place.
    #[error(transparent)]
    Layout(#[from] LayoutError),
    /// A descriptor no scheduler would plan.
    #[error("invalid L3 job: {0}")]
    Invalid(&'static str),
    /// More operations in flight than the scheduler's credits.
    #[error("the scheduler exceeded the L3 operation credits")]
    Credits,
    /// A namespace change while operations are in flight.
    #[error("L3 operations must retire before the namespace changes")]
    Busy,
    /// Shutdown could not prove every native I/O retired.
    #[error("L3 I/O retirement is unproven; the Host pool is retained; restart required")]
    Unretired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Query,
    Get,
    Put,
}

const KINDS: [Kind; 3] = [Kind::Query, Kind::Get, Kind::Put];

impl Kind {
    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Query => "query",
            Kind::Get => "get",
            Kind::Put => "put",
        }
    }
}

/// A job as the workers run it: object keys and, for a get or put, this
/// rank's Host range per key (None where another rank owns the block).
#[derive(Debug, Clone)]
struct Job {
    op_id: u64,
    kind: Kind,
    keys: Vec<String>,
    ranges: Vec<Option<Range>>,
}

impl Job {
    /// The positions with native work on this rank.
    fn local(&self) -> Vec<usize> {
        match self.kind {
            Kind::Query => (0..self.keys.len()).collect(),
            Kind::Get | Kind::Put => (0..self.keys.len())
                .filter(|&i| self.ranges[i].is_some())
                .collect(),
        }
    }

    fn num_bytes(&self) -> usize {
        self.ranges.iter().flatten().map(|range| range.size).sum()
    }
}

#[derive(Debug)]
struct JobState {
    job: Arc<Job>,
    submitted_at: f64,
    num_bytes: usize,
    started_at: Option<f64>,
    cancelled: bool,
    completed: bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct Counters {
    accepted: u64,
    backend_calls: u64,
    completed: u64,
    cancelled_queued: u64,
    cancelled_running: u64,
    rejected: u64,
    unretired: u64,
    success_keys: u64,
    failed_keys: u64,
    success_bytes: u64,
    queue_seconds: f64,
    backend_seconds: f64,
}

#[derive(Debug, Default)]
struct State {
    jobs: HashMap<u64, JobState>,
    results: VecDeque<StorageCompletion>,
    queues: [VecDeque<u64>; 3],
    counters: [Counters; 3],
    stopping: bool,
    unsafe_io: bool,
    last_submitted: u64,
    /// Workers still running.
    running_workers: usize,
}

/// A permission to call the backend, from one health epoch.
#[derive(Debug, Clone, Copy)]
struct HealthToken {
    epoch: u64,
    recovery: bool,
}

#[derive(Debug, Default)]
struct HealthState {
    epoch: u64,
    failures: u32,
    unavailable_until: Option<f64>,
    probe_inflight: bool,
}

/// The backend's availability: after `threshold` failures it opens for the
/// recovery interval, then admits one probe whose outcome closes it again
/// or reopens it. Reports from an older epoch change nothing.
struct Health {
    threshold: u32,
    interval: f64,
    clock: Clock,
    state: Mutex<HealthState>,
}

impl Health {
    fn lock(&self) -> MutexGuard<'_, HealthState> {
        self.state.lock().expect("L3 health lock")
    }

    fn begin(&self) -> Option<HealthToken> {
        let mut state = self.lock();
        match state.unavailable_until {
            None => Some(HealthToken {
                epoch: state.epoch,
                recovery: false,
            }),
            Some(until) if (self.clock)() < until || state.probe_inflight => None,
            Some(_) => {
                state.probe_inflight = true;
                Some(HealthToken {
                    epoch: state.epoch,
                    recovery: true,
                })
            }
        }
    }

    fn failure(&self, token: HealthToken) {
        let mut state = self.lock();
        if token.epoch != state.epoch {
            return;
        }
        state.failures += 1;
        if token.recovery || state.failures >= self.threshold {
            state.epoch += 1;
            state.unavailable_until = Some((self.clock)() + self.interval);
            state.probe_inflight = false;
        }
    }

    fn success(&self, token: HealthToken) {
        let mut state = self.lock();
        if token.epoch != state.epoch {
            return;
        }
        state.failures = 0;
        if token.recovery {
            state.epoch += 1;
            state.unavailable_until = None;
            state.probe_inflight = false;
        }
    }

    /// `healthy`, `open` or `half_open`, and the epoch, without taking the
    /// probe permission.
    fn snapshot(&self) -> (&'static str, u64) {
        let state = self.lock();
        let name = match (state.unavailable_until, state.probe_inflight) {
            (None, _) => "healthy",
            (Some(_), true) => "half_open",
            (Some(_), false) => "open",
        };
        (name, state.epoch)
    }
}

/// The namespace objects are named in: the deployment, and whether a draft
/// shares the weight versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Namespace {
    /// The deployment's identity (`storage_deployment_id`).
    pub deployment_id: String,
    /// The draft model's cache is stored too; its weights follow the
    /// target's version.
    pub has_draft: bool,
}

impl Namespace {
    fn prefix(&self, weight_version: &str) -> String {
        let draft = if self.has_draft { weight_version } else { "" };
        keys::key_prefix(&self.deployment_id, weight_version, draft)
    }
}

struct Shared {
    store: Box<dyn Store>,
    host: HostBuffer,
    layout: HostLayout,
    namespace: Namespace,
    config: ExecutorConfig,
    clock: Clock,
    health: Health,
    state: Mutex<State>,
    /// Per kind: a job was queued, or shutdown began.
    work: [Condvar; 3],
    /// A worker exited.
    exited: Condvar,
    prefix: Mutex<String>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("L3 executor lock")
    }

    /// Complete `op_id` with `result` (under the state lock).
    fn finish(&self, state: &mut State, op_id: u64, result: IoResult) {
        let now = (self.clock)();
        let Some(job_state) = state.jobs.get_mut(&op_id) else {
            return;
        };
        job_state.completed = true;
        let job = Arc::clone(&job_state.job);
        let queued_until_now = job_state
            .started_at
            .is_none()
            .then_some(now - job_state.submitted_at);
        let counters = &mut state.counters[job.kind.index()];
        counters.completed += 1;
        for position in job.local() {
            if result.success[position] {
                counters.success_keys += 1;
                if let Some(range) = job.ranges.get(position).copied().flatten() {
                    counters.success_bytes += range.size as u64;
                }
            } else {
                counters.failed_keys += 1;
            }
        }
        if let Some(queued) = queued_until_now {
            counters.queue_seconds += queued;
        }
        if !result.retired {
            counters.unretired += 1;
            state.unsafe_io = true;
        }
        state.results.push_back(StorageCompletion {
            op_id,
            success: result.success,
            retired: result.retired,
            object_failures: result.object_failures,
        });
    }

    /// A retired failure of every key: the job never ran.
    fn rejected(keys: usize) -> IoResult {
        IoResult {
            success: vec![false; keys],
            retired: true,
            object_failures: vec![false; keys],
        }
    }

    fn submit(&self, job: Job) -> Result<(), ExecutorError> {
        if job.keys.is_empty() || job.keys.len() > self.config.max_keys_per_operation {
            return Err(ExecutorError::Invalid("its key count"));
        }
        let mut state = self.lock();
        if job.op_id <= state.last_submitted {
            return Err(ExecutorError::Invalid("its operation id does not increase"));
        }
        state.last_submitted = job.op_id;
        if state.jobs.len() >= self.config.max_operations {
            // The scheduler bounds every accepted operation: exceeding it is
            // a protocol fault, not a completion to drop silently.
            return Err(ExecutorError::Credits);
        }
        let (op_id, kind, num_keys) = (job.op_id, job.kind, job.keys.len());
        let num_bytes = job.num_bytes();
        state.counters[kind.index()].accepted += 1;
        state.jobs.insert(
            op_id,
            JobState {
                job: Arc::new(job),
                submitted_at: (self.clock)(),
                num_bytes,
                started_at: None,
                cancelled: false,
                completed: false,
            },
        );
        let queue = &state.queues[kind.index()];
        if state.stopping || state.unsafe_io || queue.len() >= self.config.max_operations {
            state.counters[kind.index()].rejected += 1;
            self.finish(&mut state, op_id, Self::rejected(num_keys));
            return Ok(());
        }
        state.queues[kind.index()].push_back(op_id);
        drop(state);
        self.work[kind.index()].notify_one();
        Ok(())
    }

    fn cancel(&self, op_id: u64) {
        let mut state = self.lock();
        let Some(job_state) = state.jobs.get_mut(&op_id) else {
            return;
        };
        if job_state.completed || job_state.cancelled {
            return;
        }
        job_state.cancelled = true;
        let started = job_state.started_at.is_some();
        let (kind, keys) = (job_state.job.kind, job_state.job.keys.len());
        let counters = &mut state.counters[kind.index()];
        if started {
            counters.cancelled_running += 1;
        } else {
            counters.cancelled_queued += 1;
            // Retire at once, even with every worker blocked in native I/O:
            // the queue entry cannot start after this, and its worker skips
            // it.
            self.finish(&mut state, op_id, Self::rejected(keys));
        }
    }

    /// One worker of `kind`'s queue, until shutdown drains it.
    fn work(&self, kind: Kind) {
        loop {
            let mut state = self.lock();
            let op_id = loop {
                if let Some(op_id) = state.queues[kind.index()].pop_front() {
                    break Some(op_id);
                }
                if state.stopping {
                    break None;
                }
                state = self.work[kind.index()]
                    .wait(state)
                    .expect("L3 executor lock");
            };
            let Some(op_id) = op_id else {
                state.running_workers -= 1;
                drop(state);
                self.exited.notify_all();
                return;
            };
            let (stopping, unsafe_io) = (state.stopping, state.unsafe_io);
            let Some(job_state) = state.jobs.get_mut(&op_id) else {
                continue;
            };
            if job_state.completed {
                continue;
            }
            let job = Arc::clone(&job_state.job);
            if job_state.cancelled || stopping || unsafe_io {
                state.counters[kind.index()].rejected += 1;
                self.finish(&mut state, op_id, Self::rejected(job.keys.len()));
                continue;
            }
            let now = (self.clock)();
            let queued = now - job_state.submitted_at;
            job_state.started_at = Some(now);
            state.counters[kind.index()].queue_seconds += queued;
            drop(state);
            let result = self.run(&job);
            let mut state = self.lock();
            self.finish(&mut state, op_id, result);
        }
    }

    /// Run `job` against the store, outside the state lock.
    fn run(&self, job: &Job) -> IoResult {
        let local = job.local();
        let keys = job.keys.len();
        if local.is_empty() {
            // No native work on this owner: it neither probes the backend's
            // health nor charges bytes it did not move.
            return IoResult {
                success: vec![true; keys],
                retired: true,
                object_failures: vec![false; keys],
            };
        }
        let local_keys: Vec<String> = local.iter().map(|&i| job.keys[i].clone()).collect();
        let local_ranges: Vec<Range> = match job.kind {
            Kind::Query => Vec::new(),
            Kind::Get | Kind::Put => local
                .iter()
                .map(|&i| job.ranges[i].expect("a local position has a range"))
                .collect(),
        };
        let result = match self.health.begin() {
            None => {
                self.lock().counters[job.kind.index()].rejected += 1;
                Self::rejected(local.len())
            }
            Some(token) => {
                self.lock().counters[job.kind.index()].backend_calls += 1;
                let started = (self.clock)();
                let called = match job.kind {
                    Kind::Query => self.store.batch_exists(&local_keys).map(|flags| IoResult {
                        object_failures: vec![false; flags.len()],
                        success: flags,
                        retired: true,
                    }),
                    Kind::Get => self
                        .store
                        .batch_get_into(&local_keys, self.host, &local_ranges),
                    Kind::Put => self
                        .store
                        .batch_put_from(&local_keys, self.host, &local_ranges),
                };
                let checked = called.map_err(|error| error.0).and_then(|result| {
                    let valid = result.success.len() == local.len()
                        && result.object_failures.len() == local.len()
                        && !result
                            .success
                            .iter()
                            .zip(&result.object_failures)
                            .any(|(&ok, &failed)| ok && failed)
                        && (job.kind == Kind::Get || !result.object_failures.contains(&true));
                    if valid {
                        Ok(result)
                    } else {
                        Err("malformed L3 result".to_owned())
                    }
                });
                let result = match checked {
                    Ok(result) => {
                        // A retired failure is an object or capacity fact,
                        // never backend unavailability.
                        if job.kind == Kind::Query || result.retired {
                            self.health.success(token);
                        } else {
                            self.health.failure(token);
                        }
                        result
                    }
                    Err(error) => {
                        tracing::error!(
                            kind = job.kind.name(),
                            op_id = job.op_id,
                            %error,
                            "L3 operation failed"
                        );
                        self.health.failure(token);
                        IoResult {
                            success: vec![false; local.len()],
                            retired: job.kind == Kind::Query || self.store.failures_retire(),
                            object_failures: vec![false; local.len()],
                        }
                    }
                };
                self.lock().counters[job.kind.index()].backend_seconds += (self.clock)() - started;
                result
            }
        };
        // Keys without native work here are neutral successes.
        let mut success = vec![true; keys];
        let mut object_failures = vec![false; keys];
        for (k, &position) in local.iter().enumerate() {
            success[position] = result.success[k];
            object_failures[position] = result.object_failures[k];
        }
        IoResult {
            success,
            retired: result.retired,
            object_failures,
        }
    }
}

/// The Host-only L3 executor of one rank. Dropping it closes it (see
/// [`Executor::close`]), waiting a few seconds for the workers.
pub struct Executor {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    closed: bool,
}

impl Executor {
    /// Start the workers over `store`, which reads and writes `host`, laid
    /// out as `layout`; objects are named in `namespace` under
    /// `weight_version`.
    pub fn start(
        store: Box<dyn Store>,
        host: HostBuffer,
        layout: HostLayout,
        namespace: Namespace,
        weight_version: &str,
        config: ExecutorConfig,
        clock: Clock,
    ) -> Self {
        assert_eq!(
            host.len,
            layout.bytes(),
            "the Host pool and its layout disagree"
        );
        let prefix = namespace.prefix(weight_version);
        let workers = config.query_workers + config.read_workers + config.write_workers;
        let shared = Arc::new(Shared {
            store,
            host,
            layout,
            namespace,
            config,
            health: Health {
                threshold: config.failure_threshold,
                interval: config.recovery_interval.as_secs_f64(),
                clock: Arc::clone(&clock),
                state: Mutex::new(HealthState::default()),
            },
            clock,
            state: Mutex::new(State {
                running_workers: workers,
                ..State::default()
            }),
            work: [Condvar::new(), Condvar::new(), Condvar::new()],
            exited: Condvar::new(),
            prefix: Mutex::new(prefix),
        });
        let mut threads = Vec::with_capacity(workers);
        for (kind, count) in [
            (Kind::Query, config.query_workers),
            (Kind::Get, config.read_workers),
            (Kind::Put, config.write_workers),
        ] {
            for index in 0..count {
                let shared = Arc::clone(&shared);
                threads.push(
                    std::thread::Builder::new()
                        .name(format!("l3-{}-{index}", kind.name()))
                        .spawn(move || shared.work(kind))
                        .expect("spawning an L3 worker"),
                );
            }
        }
        Self {
            shared,
            threads,
            closed: false,
        }
    }

    /// The limits and workers it started with.
    pub fn config(&self) -> ExecutorConfig {
        self.shared.config
    }

    /// The namespace prefix objects are named under now.
    pub fn key_prefix(&self) -> String {
        self.shared.prefix.lock().expect("L3 prefix lock").clone()
    }

    fn object_keys(&self, keys: &[StorageKey]) -> Result<Vec<String>, ExecutorError> {
        let prefix = self.key_prefix();
        keys.iter()
            .map(|key| {
                Ok(keys::object_key(
                    &prefix,
                    &key.content_hash,
                    key.group,
                    key.page_offset,
                    self.shared.layout.key_rank(key.group)?,
                ))
            })
            .collect()
    }

    /// Freeze `keys`' object names and this rank's Host ranges, in the
    /// operation's order; a key another rank owns gets no range, and its
    /// completion here stays neutral.
    fn ranged_job(
        &self,
        op_id: u64,
        kind: Kind,
        keys: &[StorageKey],
        host_blocks: &[i32],
    ) -> Result<Job, ExecutorError> {
        if keys.len() != host_blocks.len() {
            return Err(ExecutorError::Invalid("a Host block per key"));
        }
        let ranges = keys
            .iter()
            .zip(host_blocks)
            .map(|(key, &block)| self.shared.layout.owned_range(key.group, block))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Job {
            op_id,
            kind,
            keys: self.object_keys(keys)?,
            ranges,
        })
    }

    /// Start a plan's storage operations, in order; none waits for I/O.
    pub fn submit(&self, operations: &[Operation]) -> Result<(), ExecutorError> {
        for operation in operations {
            let job = match operation {
                Operation::Cancel { op_id } => {
                    self.shared.cancel(*op_id);
                    continue;
                }
                Operation::Query { op_id, keys } => Job {
                    op_id: *op_id,
                    kind: Kind::Query,
                    keys: self.object_keys(keys)?,
                    ranges: Vec::new(),
                },
                Operation::Get {
                    op_id,
                    keys,
                    host_blocks,
                } => self.ranged_job(*op_id, Kind::Get, keys, host_blocks)?,
                Operation::Put {
                    op_id,
                    keys,
                    host_blocks,
                } => self.ranged_job(*op_id, Kind::Put, keys, host_blocks)?,
            };
            self.shared.submit(job)?;
        }
        Ok(())
    }

    /// The operations finished since the last poll and the queues' pressure,
    /// under one lock and without I/O. A running cancelled job still counts:
    /// detaching its request does not retire its native I/O.
    pub fn poll(&self) -> StoragePoll {
        let mut state = self.shared.lock();
        let completions: Vec<StorageCompletion> = state.results.drain(..).collect();
        for completion in &completions {
            state.jobs.remove(&completion.op_id);
        }
        let now = (self.shared.clock)();
        let mut pressure = StoragePressure::default();
        let mut oldest = now;
        for job_state in state.jobs.values().filter(|job| !job.completed) {
            oldest = oldest.min(job_state.submitted_at);
            if job_state.started_at.is_none() {
                pressure.queued_operations += 1;
                pressure.queued_bytes += job_state.num_bytes as i64;
            }
        }
        pressure.oldest_operation_us = ((now - oldest) * 1_000_000.0).max(0.0) as i64;
        StoragePoll {
            now_us: (now * 1_000_000.0) as i64,
            completions,
            pressure,
        }
    }

    /// Nothing in flight, and no unproven retirement.
    pub fn is_idle(&self) -> bool {
        let state = self.shared.lock();
        state.jobs.is_empty() && !state.unsafe_io
    }

    /// The counters, without backend or GPU work. Completed jobs count until
    /// a poll delivers them.
    pub fn snapshot(&self) -> L3ExecutorSnapshot {
        let state = self.shared.lock();
        let operations = KINDS
            .iter()
            .map(|&kind| {
                let counters = state.counters[kind.index()];
                let mut phases = [(0.0, 0.0); 3];
                for job_state in state.jobs.values().filter(|job| job.job.kind == kind) {
                    let phase = if job_state.completed {
                        2
                    } else if job_state.started_at.is_some() {
                        1
                    } else {
                        0
                    };
                    phases[phase].0 += 1.0;
                    phases[phase].1 += job_state.num_bytes as f64;
                }
                let by_phase = |pick: fn(&(f64, f64)) -> f64| -> Vec<(String, f64)> {
                    ["queued", "running", "completed"]
                        .iter()
                        .zip(&phases)
                        .map(|(name, phase)| ((*name).to_owned(), pick(phase)))
                        .collect()
                };
                L3OperationSnapshot {
                    kind: kind.name().to_owned(),
                    events: [
                        ("accepted", counters.accepted),
                        ("backend_calls", counters.backend_calls),
                        ("completed", counters.completed),
                        ("cancelled_queued", counters.cancelled_queued),
                        ("cancelled_running", counters.cancelled_running),
                        ("rejected", counters.rejected),
                        ("unretired", counters.unretired),
                    ]
                    .into_iter()
                    .map(|(name, value)| (name.to_owned(), value as f64))
                    .collect(),
                    success_keys: counters.success_keys as f64,
                    failed_keys: counters.failed_keys as f64,
                    success_bytes: counters.success_bytes as f64,
                    queue_seconds: counters.queue_seconds,
                    backend_seconds: counters.backend_seconds,
                    operations: by_phase(|phase| phase.0),
                    bytes: by_phase(|phase| phase.1),
                }
            })
            .collect();
        drop(state);
        let (health_state, health_epoch) = self.shared.health.snapshot();
        L3ExecutorSnapshot {
            operations,
            health_state: health_state.to_owned(),
            health_epoch: health_epoch as f64,
            unsafe_io: self.shared.lock().unsafe_io,
        }
    }

    /// Switch the namespace to `weight_version`, after the replica agreed to
    /// clear the caches.
    pub fn set_weight_version(&self, weight_version: &str) -> Result<(), ExecutorError> {
        if !self.is_idle() {
            return Err(ExecutorError::Busy);
        }
        *self.shared.prefix.lock().expect("L3 prefix lock") =
            self.shared.namespace.prefix(weight_version);
        Ok(())
    }

    /// Delete the namespace's objects once idle: true when they are gone.
    pub fn delete_namespace(&self) -> bool {
        if !self.is_idle() {
            return false;
        }
        let prefix = format!("{}_", self.key_prefix());
        self.shared.store.remove_by_prefix(&prefix)
    }

    /// Stop the workers within `timeout` and release the store; a later
    /// call does nothing. If a worker is still in native I/O, or some I/O
    /// never proved its retirement, the store stays connected for the rest
    /// of the process and the caller must keep the Host pool alive too.
    pub fn close(&mut self, timeout: Duration) -> Result<(), ExecutorError> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.lock();
        state.stopping = true;
        for work in &self.shared.work {
            work.notify_all();
        }
        while state.running_workers > 0 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            state = self
                .shared
                .exited
                .wait_timeout(state, left)
                .expect("L3 executor lock")
                .0;
        }
        let retired = state.running_workers == 0 && !state.unsafe_io;
        drop(state);
        if !retired {
            // Only process exit releases an unretired native transport: the
            // store, and with it the registration of the pool, stay.
            std::mem::forget(Arc::clone(&self.shared));
            return Err(ExecutorError::Unretired);
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        self.shared.store.close();
        Ok(())
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if let Err(error) = self.close(Duration::from_secs(5)) {
            tracing::error!(%error, "closing the L3 executor");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{layout::GroupLayout, store::MemoryStore, store::StoreError};

    /// A one-shot latch (`threading.Event`).
    #[derive(Default)]
    struct Event {
        set: Mutex<bool>,
        changed: Condvar,
    }

    impl Event {
        fn set(&self) {
            *self.set.lock().unwrap() = true;
            self.changed.notify_all();
        }

        fn wait(&self, timeout: Duration) -> bool {
            let guard = self.set.lock().unwrap();
            *self
                .changed
                .wait_timeout_while(guard, timeout, |set| !*set)
                .unwrap()
                .0
        }
    }

    type ExistsFn = dyn Fn(&[String]) -> Result<Vec<bool>, StoreError> + Send + Sync;
    type IoFn =
        dyn Fn(&[String], HostBuffer, &[Range]) -> Result<IoResult, StoreError> + Send + Sync;
    /// The gets and puts a store ran, in order, with their keys.
    type Calls = Arc<Mutex<Vec<(&'static str, Vec<String>)>>>;

    /// A memory store whose calls a test may replace, recording the gets
    /// and puts it runs and how often it is closed.
    #[derive(Default)]
    struct TestStore {
        memory: Arc<MemoryStore>,
        exists: Option<Box<ExistsFn>>,
        get: Option<Box<IoFn>>,
        put: Option<Box<IoFn>>,
        /// Clearing a namespace fails.
        remove_fails: bool,
        failures_retire: bool,
        calls: Calls,
        closes: Arc<AtomicUsize>,
    }

    impl TestStore {
        fn new() -> Self {
            Self {
                failures_retire: true,
                ..Self::default()
            }
        }
    }

    impl Store for TestStore {
        fn failures_retire(&self) -> bool {
            self.failures_retire
        }

        fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError> {
            match &self.exists {
                Some(exists) => exists(keys),
                None => self.memory.batch_exists(keys),
            }
        }

        fn batch_get_into(
            &self,
            keys: &[String],
            host: HostBuffer,
            ranges: &[Range],
        ) -> Result<IoResult, StoreError> {
            self.calls.lock().unwrap().push(("get", keys.to_vec()));
            match &self.get {
                Some(get) => get(keys, host, ranges),
                None => self.memory.batch_get_into(keys, host, ranges),
            }
        }

        fn batch_put_from(
            &self,
            keys: &[String],
            host: HostBuffer,
            ranges: &[Range],
        ) -> Result<IoResult, StoreError> {
            self.calls.lock().unwrap().push(("put", keys.to_vec()));
            match &self.put {
                Some(put) => put(keys, host, ranges),
                None => self.memory.batch_put_from(keys, host, ranges),
            }
        }

        fn remove_by_prefix(&self, prefix: &str) -> bool {
            !self.remove_fails && self.memory.remove_by_prefix(prefix)
        }

        fn close(&self) {
            self.closes.fetch_add(1, Ordering::SeqCst);
        }
    }

    const CONFIG: ExecutorConfig = ExecutorConfig {
        max_operations: 32,
        max_keys_per_operation: 256,
        query_workers: 2,
        read_workers: 2,
        write_workers: 1,
        failure_threshold: 1,
        recovery_interval: Duration::from_secs(30),
    };

    /// Group 0 has 8-byte blocks, group 1 16-byte ones, one each per LCM
    /// block; with `shard_count` owners of group 0, this is owner
    /// `owner_rank`.
    fn layout(shard_count: usize, owner_rank: usize, tp_rank: usize) -> HostLayout {
        HostLayout {
            groups: vec![
                GroupLayout {
                    packing: 1,
                    block_bytes: 8,
                    shard_count,
                },
                GroupLayout {
                    packing: 1,
                    block_bytes: 16,
                    shard_count: 1,
                },
            ],
            lcm_block_bytes: 24,
            num_lcm_blocks: 4,
            owner_rank,
            tp_rank,
        }
    }

    /// An executor over its own Host pool.
    struct Harness {
        executor: Option<Executor>,
        pool: Vec<u8>,
    }

    impl Harness {
        fn start(
            store: TestStore,
            config: ExecutorConfig,
            layout: HostLayout,
            clock: Clock,
        ) -> Self {
            let mut pool = vec![0_u8; layout.bytes()];
            let host = HostBuffer {
                base: pool.as_mut_ptr() as usize,
                len: pool.len(),
            };
            let executor = Executor::start(
                Box::new(store),
                host,
                layout,
                Namespace {
                    deployment_id: "deployment".to_owned(),
                    has_draft: false,
                },
                "v1",
                config,
                clock,
            );
            Self {
                executor: Some(executor),
                pool,
            }
        }

        fn new(store: TestStore, config: ExecutorConfig) -> Self {
            Self::start(store, config, layout(1, 0, 0), monotonic_clock())
        }

        fn executor(&self) -> &Executor {
            self.executor.as_ref().unwrap()
        }

        /// The next `count` completions, within three seconds.
        fn wait(&self, count: usize) -> Vec<StorageCompletion> {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut completions = Vec::new();
            while completions.len() < count {
                assert!(Instant::now() < deadline, "{completions:?}");
                completions.extend(self.executor().poll().completions);
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(completions.len(), count, "{completions:?}");
            completions
        }

        fn close(&mut self, timeout: Duration) -> Result<(), ExecutorError> {
            match self.executor.take() {
                Some(mut executor) => executor.close(timeout),
                None => Ok(()),
            }
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            // Stop the workers before the pool goes; a test that expects an
            // unretired close left nothing to stop.
            let _ = self.close(Duration::from_secs(2));
        }
    }

    fn key(hash: &str) -> StorageKey {
        StorageKey {
            group: 0,
            content_hash: hash.to_owned(),
            page_offset: 0,
        }
    }

    fn query(op_id: u64, hashes: &[&str]) -> Operation {
        Operation::Query {
            op_id,
            keys: hashes.iter().map(|hash| key(hash)).collect(),
        }
    }

    fn get(op_id: u64, hash: &str, block: i32) -> Operation {
        Operation::Get {
            op_id,
            keys: vec![key(hash)],
            host_blocks: vec![block],
        }
    }

    fn put(op_id: u64, hash: &str, block: i32) -> Operation {
        Operation::Put {
            op_id,
            keys: vec![key(hash)],
            host_blocks: vec![block],
        }
    }

    fn row(snapshot: &L3ExecutorSnapshot, kind: &str) -> L3OperationSnapshot {
        snapshot
            .operations
            .iter()
            .find(|row| row.kind == kind)
            .cloned()
            .unwrap()
    }

    fn value(pairs: &[(String, f64)], name: &str) -> f64 {
        pairs.iter().find(|(key, _)| key == name).unwrap().1
    }

    fn fake_clock(now: &Arc<Mutex<f64>>) -> Clock {
        let now = Arc::clone(now);
        Arc::new(move || *now.lock().unwrap())
    }

    #[test]
    fn stale_reports_cannot_reopen_a_failed_recovery() {
        let now = Arc::new(Mutex::new(1.0));
        let health = Health {
            threshold: 1,
            interval: 10.0,
            clock: fake_clock(&now),
            state: Mutex::default(),
        };
        let old = health.begin().unwrap();
        let failing = health.begin().unwrap();
        health.failure(failing);
        health.success(old);
        assert!(health.begin().is_none());
        *now.lock().unwrap() = 12.0;
        let probe = health.begin().unwrap();
        assert!(probe.recovery);
        assert!(health.begin().is_none());
        health.failure(probe);
        // This success belongs to the failed epoch.
        health.success(probe);
        assert!(health.begin().is_none());
        assert_eq!(health.snapshot().0, "open");
        *now.lock().unwrap() = 23.0;
        let recovered = health.begin().unwrap();
        assert!(recovered.recovery && health.begin().is_none());
        health.success(recovered);
        assert!(!health.begin().unwrap().recovery);
    }

    #[test]
    fn a_snapshot_cannot_take_the_recovery_probe() {
        let now = Arc::new(Mutex::new(0.0));
        let health = Health {
            threshold: 1,
            interval: 1.0,
            clock: fake_clock(&now),
            state: Mutex::default(),
        };
        health.failure(health.begin().unwrap());
        *now.lock().unwrap() = 2.0;
        for _ in 0..3 {
            assert_eq!(health.snapshot().0, "open");
        }
        let probe = health.begin().unwrap();
        assert!(probe.recovery);
        assert_eq!(health.snapshot().0, "half_open");
        health.success(probe);
        assert_eq!(health.snapshot().0, "healthy");
    }

    #[test]
    fn a_slow_query_blocks_neither_another_worker_nor_the_submitter() {
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let slow = keys::object_key(&keys::key_prefix("deployment", "v1", ""), "slow", 0, 0, 0);
        let store = TestStore {
            exists: Some(Box::new(move |keys| {
                if keys == [slow.clone()] {
                    s.set();
                    assert!(r.wait(Duration::from_secs(3)));
                }
                Ok(vec![true; keys.len()])
            })),
            ..TestStore::new()
        };
        let harness = Harness::new(store, CONFIG);
        harness.executor().submit(&[query(1, &["slow"])]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        harness.executor().submit(&[query(2, &["fast"])]).unwrap();
        let fast = &harness.wait(1)[0];
        assert_eq!((fast.op_id, fast.success.as_slice()), (2, &[true][..]));
        release.set();
        assert_eq!(harness.wait(1)[0].op_id, 1);
    }

    #[test]
    fn a_cancelled_queued_get_never_reaches_the_store() {
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let store = TestStore {
            get: Some(Box::new(move |keys, _, _| {
                s.set();
                assert!(r.wait(Duration::from_secs(3)));
                Ok(IoResult {
                    success: vec![true; keys.len()],
                    retired: true,
                    object_failures: vec![false; keys.len()],
                })
            })),
            ..TestStore::new()
        };
        let calls = Arc::clone(&store.calls);
        let harness = Harness::new(
            store,
            ExecutorConfig {
                read_workers: 1,
                ..CONFIG
            },
        );
        harness.executor().submit(&[get(1, "a", 1)]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        harness
            .executor()
            .submit(&[get(2, "b", 2), Operation::Cancel { op_id: 2 }])
            .unwrap();
        let cancelled = &harness.wait(1)[0];
        assert_eq!(cancelled.op_id, 2);
        assert!(cancelled.retired);
        assert_eq!(cancelled.success, [false]);
        assert_eq!(cancelled.object_failures, [false]);
        // The only worker is still inside a's I/O.
        assert_eq!(calls.lock().unwrap().len(), 1);
        release.set();
        assert_eq!(harness.wait(1)[0].op_id, 1);
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_failing_query_degrades_without_killing_its_worker() {
        let store = TestStore {
            exists: Some(Box::new(|_| Err(StoreError("unavailable".to_owned())))),
            ..TestStore::new()
        };
        let harness = Harness::new(store, CONFIG);
        harness.executor().submit(&[query(1, &["a"])]).unwrap();
        let failed = &harness.wait(1)[0];
        assert!(failed.retired);
        assert_eq!(failed.success, [false]);
        harness.executor().submit(&[query(2, &["b"])]).unwrap();
        assert_eq!(harness.wait(1)[0].op_id, 2);
    }

    #[test]
    fn unproven_retirement_disables_io_and_refuses_a_clean_close() {
        for success in [false, true] {
            let store = TestStore {
                get: Some(Box::new(move |_, _, _| {
                    Ok(IoResult {
                        success: vec![success],
                        retired: false,
                        object_failures: vec![false],
                    })
                })),
                ..TestStore::new()
            };
            let closes = Arc::clone(&store.closes);
            let mut harness = Harness::new(store, CONFIG);
            harness.executor().submit(&[get(1, "a", 1)]).unwrap();
            assert!(!harness.wait(1)[0].retired);
            assert!(!harness.executor().is_idle());
            assert_eq!(harness.executor().snapshot().health_state, "open");
            harness.executor().submit(&[get(2, "b", 2)]).unwrap();
            // No new I/O starts.
            let refused = &harness.wait(1)[0];
            assert!(refused.retired);
            assert_eq!(refused.success, [false]);
            assert!(matches!(
                harness.close(Duration::from_secs(2)),
                Err(ExecutorError::Unretired)
            ));
            // The store stays connected: only process exit releases it.
            assert_eq!(closes.load(Ordering::SeqCst), 0);
            // The pool must outlive the leaked workers.
            std::mem::forget(std::mem::take(&mut harness.pool));
        }
    }

    #[test]
    fn independent_executors_share_objects_through_the_store() {
        let memory = Arc::new(MemoryStore::new());
        let writer = Harness::new(
            TestStore {
                memory: Arc::clone(&memory),
                ..TestStore::new()
            },
            CONFIG,
        );
        let reader = Harness::new(
            TestStore {
                memory: Arc::clone(&memory),
                ..TestStore::new()
            },
            CONFIG,
        );
        // Block 1 of group 0 is the pool's first 8 bytes.
        let pool = writer.pool.as_ptr() as *mut u8;
        // SAFETY: the pool is live and no job runs yet.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::copy_nonoverlapping(b"abcdefgh".as_ptr(), pool, 8);
        }
        writer.executor().submit(&[put(1, "shared", 1)]).unwrap();
        assert_eq!(writer.wait(1)[0].success, [true]);
        // Block 3 sits at the third LCM block.
        reader.executor().submit(&[get(1, "shared", 3)]).unwrap();
        assert_eq!(reader.wait(1)[0].success, [true]);
        assert_eq!(&reader.pool[48..56], b"abcdefgh");
        assert_eq!(&reader.pool[..8], &[0; 8]);
    }

    #[test]
    fn operation_credits_bound_what_is_in_flight() {
        let harness = Harness::new(
            TestStore::new(),
            ExecutorConfig {
                max_operations: 1,
                ..CONFIG
            },
        );
        harness.executor().submit(&[query(1, &["a"])]).unwrap();
        // Completed but not polled still counts.
        std::thread::sleep(Duration::from_millis(20));
        assert!(matches!(
            harness.executor().submit(&[query(2, &["b"])]),
            Err(ExecutorError::Credits)
        ));
        harness.wait(1);
    }

    #[test]
    fn the_key_limit_and_increasing_ids_are_enforced() {
        let harness = Harness::new(
            TestStore::new(),
            ExecutorConfig {
                max_keys_per_operation: 2,
                ..CONFIG
            },
        );
        assert!(matches!(
            harness.executor().submit(&[query(1, &["a", "b", "c"])]),
            Err(ExecutorError::Invalid(_))
        ));
        harness.executor().submit(&[query(1, &["a", "b"])]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [false, false]);
        assert!(matches!(
            harness.executor().submit(&[query(1, &["a"])]),
            Err(ExecutorError::Invalid(_))
        ));
    }

    #[test]
    fn a_retired_get_miss_is_an_object_failure_not_a_backend_fault() {
        let harness = Harness::new(TestStore::new(), CONFIG);
        harness.executor().submit(&[get(1, "evicted", 1)]).unwrap();
        let miss = &harness.wait(1)[0];
        assert!(miss.retired);
        assert_eq!(miss.success, [false]);
        assert_eq!(miss.object_failures, [true]);
        assert_eq!(harness.executor().snapshot().health_state, "healthy");
    }

    #[test]
    fn snapshots_count_cancelled_queued_work_and_late_completions() {
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let store = TestStore {
            get: Some(Box::new(move |keys, _, _| {
                s.set();
                assert!(r.wait(Duration::from_secs(3)));
                Ok(IoResult {
                    success: vec![true; keys.len()],
                    retired: true,
                    object_failures: vec![false; keys.len()],
                })
            })),
            ..TestStore::new()
        };
        let harness = Harness::new(
            store,
            ExecutorConfig {
                read_workers: 1,
                ..CONFIG
            },
        );
        let executor = harness.executor();
        executor.submit(&[get(1, "running", 1)]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        // A group-1 block is 16 bytes.
        executor
            .submit(&[Operation::Get {
                op_id: 2,
                keys: vec![StorageKey {
                    group: 1,
                    content_hash: "queued".to_owned(),
                    page_offset: 0,
                }],
                host_blocks: vec![1],
            }])
            .unwrap();
        let snapshot = row(&executor.snapshot(), "get");
        assert_eq!(
            snapshot.operations,
            [
                ("queued".to_owned(), 1.0),
                ("running".to_owned(), 1.0),
                ("completed".to_owned(), 0.0)
            ]
        );
        assert_eq!(value(&snapshot.bytes, "queued"), 16.0);
        assert_eq!(value(&snapshot.bytes, "running"), 8.0);
        executor
            .submit(&[
                Operation::Cancel { op_id: 2 },
                Operation::Cancel { op_id: 2 },
                Operation::Cancel { op_id: 1 },
            ])
            .unwrap();
        harness.wait(1);
        let cancelled = row(&executor.snapshot(), "get");
        assert_eq!(value(&cancelled.events, "cancelled_queued"), 1.0);
        assert_eq!(value(&cancelled.events, "cancelled_running"), 1.0);
        assert_eq!(value(&cancelled.events, "backend_calls"), 1.0);
        assert_eq!(value(&cancelled.operations, "running"), 1.0);
        release.set();
        let deadline = Instant::now() + Duration::from_secs(3);
        while value(&row(&executor.snapshot(), "get").operations, "completed") != 1.0 {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let completed = row(&executor.snapshot(), "get");
        assert_eq!(value(&completed.bytes, "completed"), 8.0);
        // I/O success does not imply publication.
        assert_eq!(completed.success_bytes, 8.0);
        assert_eq!((completed.success_keys, completed.failed_keys), (1.0, 1.0));
        assert!(completed.queue_seconds >= 0.0 && completed.backend_seconds > 0.0);
        harness.wait(1);
        let drained = row(&executor.snapshot(), "get");
        assert!(drained.operations.iter().all(|(_, count)| *count == 0.0));
    }

    #[test]
    fn an_open_backend_rejects_jobs_visibly_without_calling_it() {
        let harness = Harness::new(TestStore::new(), CONFIG);
        let executor = harness.executor();
        let shared = &executor.shared;
        shared.health.failure(shared.health.begin().unwrap());
        executor.submit(&[put(1, "rejected", 1)]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [false]);
        let put_row = row(&executor.snapshot(), "put");
        assert_eq!(value(&put_row.events, "rejected"), 1.0);
        assert_eq!(value(&put_row.events, "backend_calls"), 0.0);
        assert_eq!((put_row.success_bytes, put_row.backend_seconds), (0.0, 0.0));
    }

    #[test]
    fn pressure_tracks_the_queue_and_cancelled_native_work_until_it_retires() {
        let now = Arc::new(Mutex::new(1.0));
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let store = TestStore {
            get: Some(Box::new(move |keys, _, _| {
                s.set();
                assert!(r.wait(Duration::from_secs(5)));
                Ok(IoResult {
                    success: vec![true; keys.len()],
                    retired: true,
                    object_failures: vec![false; keys.len()],
                })
            })),
            ..TestStore::new()
        };
        let harness = Harness::start(
            store,
            ExecutorConfig {
                read_workers: 1,
                ..CONFIG
            },
            layout(1, 0, 0),
            fake_clock(&now),
        );
        let executor = harness.executor();
        executor.submit(&[get(1, "active", 1)]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        executor.submit(&[get(2, "queued", 2)]).unwrap();
        *now.lock().unwrap() = 1.75;
        let poll = executor.poll();
        assert!(poll.completions.is_empty());
        assert_eq!(
            poll.pressure,
            StoragePressure {
                queued_operations: 1,
                queued_bytes: 8,
                oldest_operation_us: 750_000
            }
        );
        executor.submit(&[Operation::Cancel { op_id: 2 }]).unwrap();
        let poll = executor.poll();
        assert_eq!(
            poll.completions.iter().map(|c| c.op_id).collect::<Vec<_>>(),
            [2]
        );
        assert_eq!(
            poll.pressure,
            StoragePressure {
                queued_operations: 0,
                queued_bytes: 0,
                oldest_operation_us: 750_000
            }
        );
        // Cancelling running native work does not retire it.
        executor.submit(&[Operation::Cancel { op_id: 1 }]).unwrap();
        assert_eq!(executor.poll().pressure.oldest_operation_us, 750_000);
        assert_eq!(executor.snapshot().health_state, "healthy");
        release.set();
        assert!(harness.wait(1)[0].retired);
        assert_eq!(executor.poll().pressure, StoragePressure::default());
        assert!(executor.is_idle());
    }

    #[test]
    fn query_pressure_counts_metadata_jobs_without_payload_bytes() {
        let now = Arc::new(Mutex::new(1.0));
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let store = TestStore {
            exists: Some(Box::new(move |keys| {
                s.set();
                assert!(r.wait(Duration::from_secs(5)));
                Ok(vec![false; keys.len()])
            })),
            ..TestStore::new()
        };
        let harness = Harness::start(
            store,
            ExecutorConfig {
                query_workers: 1,
                ..CONFIG
            },
            layout(1, 0, 0),
            fake_clock(&now),
        );
        let executor = harness.executor();
        executor.submit(&[query(1, &["a"])]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        executor.submit(&[query(2, &["b"])]).unwrap();
        *now.lock().unwrap() = 2.0;
        assert_eq!(
            executor.poll().pressure,
            StoragePressure {
                queued_operations: 1,
                queued_bytes: 0,
                oldest_operation_us: 1_000_000
            }
        );
        release.set();
        assert!(harness.wait(2).iter().all(|c| c.retired));
        assert_eq!(executor.poll().pressure.oldest_operation_us, 0);
    }

    #[test]
    fn a_rejected_or_failed_get_reports_no_object_failure() {
        // Through the open backend, a full queue and a failing store.
        for failure in ["health", "queue", "error"] {
            let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
            let (s, r) = (Arc::clone(&started), Arc::clone(&release));
            let blocking = failure == "queue";
            let store = TestStore {
                get: (failure != "health").then(|| {
                    Box::new(move |keys: &[String], _: HostBuffer, _: &[Range]| {
                        if blocking && keys[0].contains("_blocker") {
                            s.set();
                            assert!(r.wait(Duration::from_secs(3)));
                        }
                        if blocking {
                            Ok(IoResult {
                                success: vec![true; keys.len()],
                                retired: true,
                                object_failures: vec![false; keys.len()],
                            })
                        } else {
                            Err(StoreError("unavailable".to_owned()))
                        }
                    }) as Box<IoFn>
                }),
                ..TestStore::new()
            };
            let harness = Harness::new(
                store,
                ExecutorConfig {
                    max_operations: 2,
                    read_workers: 1,
                    ..CONFIG
                },
            );
            let executor = harness.executor();
            let target = match failure {
                "health" => {
                    let shared = &executor.shared;
                    shared.health.failure(shared.health.begin().unwrap());
                    1
                }
                "queue" => {
                    // A stale entry of a cancelled job keeps its queue slot
                    // until a worker passes it.
                    executor.submit(&[get(1, "blocker", 1)]).unwrap();
                    assert!(started.wait(Duration::from_secs(1)));
                    executor
                        .submit(&[get(2, "stale", 2), Operation::Cancel { op_id: 2 }])
                        .unwrap();
                    harness.wait(1);
                    executor
                        .submit(&[get(3, "stale", 3), Operation::Cancel { op_id: 3 }])
                        .unwrap();
                    harness.wait(1);
                    4
                }
                _ => 1,
            };
            executor.submit(&[get(target, "present", 4)]).unwrap();
            let result = &harness.wait(1)[0];
            assert_eq!(result.op_id, target);
            assert!(result.retired);
            assert_eq!(result.success, [false]);
            assert_eq!(result.object_failures, [false]);
            if failure != "error" {
                let get_row = row(&executor.snapshot(), "get");
                let calls = if failure == "queue" { 1.0 } else { 0.0 };
                assert_eq!(value(&get_row.events, "backend_calls"), calls);
            }
            release.set();
        }
    }

    #[test]
    fn malformed_facts_cannot_publish() {
        for (success, object_failures) in [
            (vec![true], vec![]),
            (vec![true], vec![true]),
            (vec![], vec![false]),
        ] {
            let store = TestStore {
                get: Some(Box::new(move |_, _, _| {
                    Ok(IoResult {
                        success: success.clone(),
                        retired: true,
                        object_failures: object_failures.clone(),
                    })
                })),
                ..TestStore::new()
            };
            let harness = Harness::new(store, CONFIG);
            harness.executor().submit(&[get(1, "a", 1)]).unwrap();
            let result = &harness.wait(1)[0];
            assert_eq!(result.success, [false]);
            assert_eq!(result.object_failures, [false]);
            assert_eq!(harness.executor().snapshot().health_state, "open");
        }
    }

    #[test]
    fn a_put_never_reports_object_failures() {
        let store = TestStore {
            get: None,
            ..TestStore::new()
        };
        let harness = Harness::new(store, CONFIG);
        harness.executor().submit(&[put(1, "a", 1)]).unwrap();
        let result = &harness.wait(1)[0];
        assert_eq!(
            (result.success.as_slice(), result.retired),
            (&[true][..], true)
        );
        assert_eq!(result.object_failures, [false]);
    }

    #[test]
    fn close_waits_for_native_retirement_then_closes_the_store_once() {
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let memory = Arc::new(MemoryStore::new());
        let reader = Arc::clone(&memory);
        let store = TestStore {
            memory: Arc::clone(&memory),
            get: Some(Box::new(move |keys, host, ranges| {
                s.set();
                assert!(r.wait(Duration::from_secs(3)));
                reader.batch_get_into(keys, host, ranges)
            })),
            ..TestStore::new()
        };
        let closes = Arc::clone(&store.closes);
        let mut harness = Harness::new(store, CONFIG);
        let executor = harness.executor();
        let object = keys::object_key(&executor.key_prefix(), "a", 0, 0, 0);
        let mut source = b"abcdefgh".to_vec();
        memory
            .batch_put_from(
                &[object],
                HostBuffer {
                    base: source.as_mut_ptr() as usize,
                    len: 8,
                },
                &[Range { offset: 0, size: 8 }],
            )
            .unwrap();
        executor.submit(&[get(1, "a", 1)]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        // The poll does not wait for the store.
        assert!(executor.poll().completions.is_empty());
        release.set();
        assert_eq!(harness.wait(1)[0].success, [true]);
        assert_eq!(&harness.pool[..8], b"abcdefgh");
        harness.close(Duration::from_secs(2)).unwrap();
        harness.close(Duration::from_secs(2)).unwrap();
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_close_timeout_keeps_the_store_and_the_pool() {
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let store = TestStore {
            get: Some(Box::new(move |keys, _, _| {
                s.set();
                assert!(r.wait(Duration::from_secs(3)));
                Ok(IoResult {
                    success: vec![true; keys.len()],
                    retired: true,
                    object_failures: vec![false; keys.len()],
                })
            })),
            ..TestStore::new()
        };
        let closes = Arc::clone(&store.closes);
        let mut harness = Harness::new(store, CONFIG);
        harness.executor().submit(&[get(1, "a", 1)]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        assert!(matches!(
            harness.close(Duration::ZERO),
            Err(ExecutorError::Unretired)
        ));
        assert_eq!(closes.load(Ordering::SeqCst), 0);
        // The worker still writes the pool after the close.
        std::mem::forget(std::mem::take(&mut harness.pool));
        release.set();
    }

    #[test]
    fn a_namespace_changes_only_while_idle() {
        let (started, release) = (Arc::new(Event::default()), Arc::new(Event::default()));
        let (s, r) = (Arc::clone(&started), Arc::clone(&release));
        let store = TestStore {
            exists: Some(Box::new(move |keys| {
                s.set();
                assert!(r.wait(Duration::from_secs(3)));
                Ok(vec![false; keys.len()])
            })),
            ..TestStore::new()
        };
        let harness = Harness::new(store, CONFIG);
        let executor = harness.executor();
        let first = executor.key_prefix();
        assert_eq!(first, keys::key_prefix("deployment", "v1", ""));
        executor.submit(&[query(1, &["a"])]).unwrap();
        assert!(started.wait(Duration::from_secs(1)));
        assert!(matches!(
            executor.set_weight_version("v2"),
            Err(ExecutorError::Busy)
        ));
        assert!(!executor.delete_namespace());
        release.set();
        harness.wait(1);
        executor.set_weight_version("v2").unwrap();
        assert_eq!(
            executor.key_prefix(),
            keys::key_prefix("deployment", "v2", "")
        );
        assert!(executor.delete_namespace());
    }

    #[test]
    fn clearing_the_namespace_deletes_its_objects_and_keeps_its_name() {
        let harness = Harness::new(TestStore::new(), CONFIG);
        let executor = harness.executor();
        executor.submit(&[put(1, "h", 1)]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [true]);
        let prefix = executor.key_prefix();
        assert!(executor.delete_namespace());
        assert_eq!(executor.key_prefix(), prefix);
        executor.submit(&[query(2, &["h"])]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [false]);
    }

    #[test]
    fn a_failed_clear_keeps_the_namespace_and_its_objects() {
        let store = TestStore {
            remove_fails: true,
            ..TestStore::new()
        };
        let harness = Harness::new(store, CONFIG);
        let executor = harness.executor();
        executor.submit(&[put(1, "h", 1)]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [true]);
        let prefix = executor.key_prefix();
        assert!(!executor.delete_namespace());
        assert_eq!(executor.key_prefix(), prefix);
        executor.submit(&[query(2, &["h"])]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [true]);
    }

    #[test]
    fn a_new_weight_version_publishes_under_a_new_namespace() {
        let store = TestStore::new();
        let memory = Arc::clone(&store.memory);
        let harness = Harness::new(store, CONFIG);
        let executor = harness.executor();
        executor.submit(&[put(1, "h", 1)]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [true]);
        let old = keys::object_key(&executor.key_prefix(), "h", 0, 0, 0);
        executor.set_weight_version("v2").unwrap();
        // The new checkpoint's pages are not the old one's.
        executor.submit(&[query(2, &["h"])]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [false]);
        executor.submit(&[put(3, "h", 1)]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [true]);
        let new = keys::object_key(&executor.key_prefix(), "h", 0, 0, 0);
        assert_ne!(new, old);
        assert_eq!(memory.batch_exists(&[old, new]).unwrap(), [true, true]);
    }

    #[test]
    fn a_rejected_put_retires_and_keeps_the_backend_in_service() {
        let store = TestStore::new();
        let memory = Arc::clone(&store.memory);
        let pressure = keys::object_key(
            &keys::key_prefix("deployment", "v1", ""),
            "pressure",
            0,
            0,
            0,
        );
        let rejected = pressure.clone();
        let store = TestStore {
            // A full store rejects `pressure`'s allocation before reading
            // its bytes.
            put: Some(Box::new(move |keys, host, ranges| {
                let mut success = Vec::new();
                for (key, range) in keys.iter().zip(ranges) {
                    success.push(
                        *key != rejected
                            && memory
                                .batch_put_from(
                                    std::slice::from_ref(key),
                                    host,
                                    std::slice::from_ref(range),
                                )?
                                .success[0],
                    );
                }
                Ok(IoResult {
                    object_failures: vec![false; keys.len()],
                    success,
                    retired: true,
                })
            })),
            ..store
        };
        let memory = Arc::clone(&store.memory);
        let harness = Harness::new(store, CONFIG);
        let executor = harness.executor();
        executor
            .submit(&[Operation::Put {
                op_id: 1,
                keys: vec![key("pressure"), key("saved")],
                host_blocks: vec![1, 2],
            }])
            .unwrap();
        let done = harness.wait(1).remove(0);
        assert_eq!(done.success, [false, true]);
        assert!(done.retired);
        assert_eq!(memory.batch_exists(&[pressure]).unwrap(), [false]);
        // Neither a quarantine nor an open circuit: the scheduler offers
        // the page again later.
        let snapshot = executor.snapshot();
        assert_eq!(snapshot.health_state, "healthy");
        assert!(!snapshot.unsafe_io);
        executor.submit(&[put(2, "retry", 3)]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [true]);
    }

    #[test]
    fn page_keys_follow_the_shard_and_state_keys_follow_the_rank() {
        let store = Arc::new(MemoryStore::new());
        let executors: Vec<Harness> = [0, 1, 2]
            .into_iter()
            .map(|rank| {
                Harness::start(
                    TestStore {
                        memory: Arc::clone(&store),
                        ..TestStore::new()
                    },
                    CONFIG,
                    layout(2, rank % 2, rank),
                    monotonic_clock(),
                )
            })
            .collect();
        let keys_of = |harness: &Harness, group: u32| {
            harness
                .executor()
                .object_keys(&[StorageKey {
                    group,
                    content_hash: "h".to_owned(),
                    page_offset: 0,
                }])
                .unwrap()
        };
        let (a, b, other) = (&executors[0], &executors[1], &executors[2]);
        assert_eq!(keys_of(a, 0), keys_of(b, 0));
        assert_ne!(keys_of(a, 0), keys_of(other, 0));
        assert_ne!(keys_of(a, 1), keys_of(b, 1));
    }

    #[test]
    fn a_sharded_block_moves_to_its_new_owner_through_the_store() {
        let store = Arc::new(MemoryStore::new());
        let harnesses = |base: &str| -> Vec<Harness> {
            let _ = base;
            (0..2)
                .map(|rank| {
                    Harness::start(
                        TestStore {
                            memory: Arc::clone(&store),
                            ..TestStore::new()
                        },
                        CONFIG,
                        layout(2, rank, rank),
                        monotonic_clock(),
                    )
                })
                .collect()
        };
        let writers = harnesses("writer");
        // Block 2 of group 0 is rank 1's local block 1: the pool's first 8
        // bytes.
        let pool = writers[1].pool.as_ptr() as *mut u8;
        // SAFETY: the pool is live and no job runs yet.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::copy_nonoverlapping(b"KV01KV01".as_ptr(), pool, 8);
        }
        for writer in &writers {
            writer.executor().submit(&[put(1, "h", 2)]).unwrap();
            let fact = &writer.wait(1)[0];
            assert!(fact.success == [true] && fact.retired);
        }
        assert_eq!(store.keys().len(), 1);
        // Block 3 belongs to rank 0 as its local block 2, at the second LCM
        // block.
        let readers = harnesses("reader");
        for reader in &readers {
            reader.executor().submit(&[get(1, "h", 3)]).unwrap();
            let fact = &reader.wait(1)[0];
            assert!(fact.success == [true] && fact.retired);
            assert_eq!(fact.object_failures, [false]);
        }
        assert_eq!(&readers[0].pool[24..32], b"KV01KV01");
        assert!(readers[1].pool.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn a_non_owner_is_neutral_and_moves_no_bytes() {
        let store = TestStore::new();
        let calls = Arc::clone(&store.calls);
        let harness = Harness::start(store, CONFIG, layout(2, 1, 1), monotonic_clock());
        // Block 3 belongs to rank 0.
        harness.executor().submit(&[get(1, "h", 3)]).unwrap();
        let fact = &harness.wait(1)[0];
        assert!(fact.success == [true] && fact.retired && fact.object_failures == [false]);
        assert!(calls.lock().unwrap().is_empty());
        let get_row = row(&harness.executor().snapshot(), "get");
        assert_eq!((get_row.success_keys, get_row.success_bytes), (0.0, 0.0));
    }

    #[test]
    fn a_job_without_native_work_cannot_recover_an_open_backend() {
        let store = TestStore {
            exists: Some(Box::new(|_| {
                Err(StoreError("metadata unavailable".to_owned()))
            })),
            ..TestStore::new()
        };
        let calls = Arc::clone(&store.calls);
        let harness = Harness::start(store, CONFIG, layout(2, 1, 1), monotonic_clock());
        let executor = harness.executor();
        executor.submit(&[query(1, &["h"])]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [false]);
        let before = executor.snapshot();
        assert_ne!(before.health_state, "healthy");
        executor.submit(&[get(2, "h", 3)]).unwrap();
        assert_eq!(harness.wait(1)[0].success, [true]);
        let after = executor.snapshot();
        assert_eq!(
            (after.health_state, after.health_epoch),
            (before.health_state, before.health_epoch)
        );
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn mixed_owner_results_keep_their_logical_order() {
        let store = TestStore::new();
        let memory = Arc::clone(&store.memory);
        let calls = Arc::clone(&store.calls);
        let harness = Harness::start(store, CONFIG, layout(2, 0, 0), monotonic_clock());
        let executor = harness.executor();
        let keys = [
            StorageKey {
                group: 0,
                content_hash: "absent-peer".to_owned(),
                page_offset: 0,
            },
            StorageKey {
                group: 0,
                content_hash: "absent-local".to_owned(),
                page_offset: 0,
            },
            StorageKey {
                group: 1,
                content_hash: "state".to_owned(),
                page_offset: 0,
            },
        ];
        let names = executor.object_keys(&keys).unwrap();
        let mut payload = vec![7_u8; 16];
        memory
            .batch_put_from(
                &names[2..],
                HostBuffer {
                    base: payload.as_mut_ptr() as usize,
                    len: 16,
                },
                &[Range {
                    offset: 0,
                    size: 16,
                }],
            )
            .unwrap();
        // Block 2 of the sharded group is rank 1's; block 3 is rank 0's.
        executor
            .submit(&[Operation::Get {
                op_id: 1,
                keys: keys.to_vec(),
                host_blocks: vec![2, 3, 1],
            }])
            .unwrap();
        let fact = &harness.wait(1)[0];
        assert_eq!(fact.success, [true, false, true]);
        assert_eq!(fact.object_failures, [false, true, false]);
        assert!(fact.retired);
        let get_row = row(&executor.snapshot(), "get");
        assert_eq!(
            (
                get_row.success_keys,
                get_row.failed_keys,
                get_row.success_bytes
            ),
            (1.0, 1.0, 16.0)
        );
        assert_eq!(calls.lock().unwrap()[0].1, names[1..].to_vec());
    }

    #[test]
    fn a_failing_store_and_its_cooldown_keep_non_owners_neutral() {
        let store = TestStore {
            get: Some(Box::new(|_, _, _| {
                Err(StoreError("read backend unavailable".to_owned()))
            })),
            ..TestStore::new()
        };
        let calls = Arc::clone(&store.calls);
        let harness = Harness::start(
            store,
            ExecutorConfig {
                recovery_interval: Duration::from_secs(60),
                ..CONFIG
            },
            layout(2, 0, 0),
            monotonic_clock(),
        );
        let executor = harness.executor();
        let keys = vec![key("peer"), key("local")];
        for op_id in [1, 2] {
            executor
                .submit(&[Operation::Get {
                    op_id,
                    keys: keys.clone(),
                    host_blocks: vec![2, 3],
                }])
                .unwrap();
            let fact = &harness.wait(1)[0];
            assert_eq!(fact.success, [true, false]);
            assert!(fact.retired && fact.object_failures == [false, false]);
        }
        assert_eq!(calls.lock().unwrap().len(), 1);
        let get_row = row(&executor.snapshot(), "get");
        assert_eq!(
            (
                get_row.failed_keys,
                get_row.success_keys,
                get_row.success_bytes
            ),
            (2.0, 0.0, 0.0)
        );
    }
}
