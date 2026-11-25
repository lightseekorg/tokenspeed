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

//! Peer pushes of a holder's Disk pages (`docs/rfcs/0006-disk-tier.md`, Peer
//! sharing): a reader cannot read a page off the holder's drive, so it asks
//! the holder rank of its own index to push it. The holder reads each block
//! with its Disk executor into a slot of a staging ring registered with that
//! executor and the transfer engine, and writes the block's fields into the
//! reader's landing as soon as the read completes, so drive reads and network
//! writes overlap. Neither side's scheduler takes part.
//!
//! Every rank runs one endpoint ([`PushEndpoint`]): it receives the pushes a
//! holder serves and the answers its own reads wait for. A push ends with
//! one answer: done, with whether every block landed. A reader that stops
//! waiting cancels it on the same channel; the holder answers once the
//! job's writes retired, and posts none past the job's budget either way,
//! so the reader's quarantine of a failed landing outlasts every write
//! (`landing::Timeouts`).

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Weak,
    },
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use parking_lot::{Condvar, Mutex};
use ts_disk::{Direction, Executor, Io, Job, JobId, JobReport, Memory};
use ts_l3::HostLayout;
use ts_pd::{NetError, Sockets};
use ts_scheduler::profile::{self, Lanes};

use crate::{
    codec::{Reader, Writer},
    peer::{geometry::Spans, net::split_endpoint},
};

const PUSH: &[u8] = b"p";
const CANCEL: &[u8] = b"c";
const DONE: &[u8] = b"d";
const CANCELLED: &[u8] = b"x";

/// How long a reader waits for a cancelled push's acknowledgment.
const CANCEL_WAIT: Duration = Duration::from_secs(5);

/// One field copy of a pushed block: `length` bytes at `offset` within the
/// holder's Disk block, to the reader's `address`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushSpan {
    /// From the block's start.
    pub offset: u64,
    /// Where it lands on the reader.
    pub address: u64,
    /// Its bytes.
    pub length: u64,
}

/// One Disk block to push: its group, its block id on the holder, and its
/// fields' copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushItem {
    /// The cache group, by index.
    pub group: u32,
    /// The holder's Disk block id.
    pub block: i32,
    /// The copies.
    pub spans: Vec<PushSpan>,
}

/// What moves a push's bytes: the transfer engine's WRITEs from the holder's
/// staging ring to the reader's registered memory.
pub trait PageWriter: Send + Sync {
    /// Copy `spans` (local sources, the peer's destinations) into the memory
    /// registered under `session`; returns once every copy retired, or the
    /// failure.
    fn write(&self, session: &str, spans: &Spans) -> Result<(), String>;
}

/// The transfer engine's one-sided WRITEs.
pub struct EngineWriter(pub Arc<ts_mooncake::Engine>);

impl PageWriter for EngineWriter {
    fn write(&self, session: &str, spans: &Spans) -> Result<(), String> {
        self.0
            .write(session, &spans.sources, &spans.destinations, &spans.lengths)
            .map_err(|error| error.to_string())
    }
}

/// What a reader's reads push through: its endpoint, in production.
pub trait PagePusher: Send + Sync {
    /// Ask the holder rank at `endpoint` to push `items` into the memory
    /// registered under `session` within `budget`; returns once the holder
    /// answered, or, past the budget, once it acknowledged a cancellation or
    /// that wait ended too.
    fn push(
        &self,
        endpoint: &str,
        session: &str,
        budget: Duration,
        items: Vec<PushItem>,
    ) -> Result<(), String>;
}

/// A holder rank's Disk pages, as its pushes read them.
pub struct PushSource {
    /// The rank's Disk executor; the staging ring is registered with it.
    pub executor: Arc<Executor>,
    /// Where a Disk block lies in the file: packed like the Host pool.
    pub disk_layout: HostLayout,
    /// The staging ring, registered with the executor and the transfer
    /// engine.
    pub ring: Arc<Memory>,
    /// Bytes of one slot, at least the largest block, a multiple of the
    /// direct-I/O alignment.
    pub slot_bytes: usize,
    /// What writes a slot's fields to the reader.
    pub writer: Arc<dyn PageWriter>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Message {
    Push {
        job: u64,
        reply_to: String,
        session: String,
        budget_us: u64,
        items: Vec<PushItem>,
    },
    Cancel {
        job: u64,
        reply_to: String,
    },
    Done {
        job: u64,
        ok: bool,
    },
    Cancelled {
        job: u64,
    },
}

impl Message {
    fn frames(&self) -> Vec<Bytes> {
        let mut out = Writer::new();
        let kind = match self {
            Message::Push {
                job,
                reply_to,
                session,
                budget_us,
                items,
            } => {
                out.u64(*job);
                out.bytes(reply_to.as_bytes());
                out.bytes(session.as_bytes());
                out.u64(*budget_us);
                out.count(items.len());
                for item in items {
                    out.u32(item.group);
                    out.i64(i64::from(item.block));
                    out.count(item.spans.len());
                    for span in &item.spans {
                        out.u64(span.offset);
                        out.u64(span.address);
                        out.u64(span.length);
                    }
                }
                PUSH
            }
            Message::Cancel { job, reply_to } => {
                out.u64(*job);
                out.bytes(reply_to.as_bytes());
                CANCEL
            }
            Message::Done { job, ok } => {
                out.u64(*job);
                out.flag(*ok);
                DONE
            }
            Message::Cancelled { job } => {
                out.u64(*job);
                CANCELLED
            }
        };
        vec![Bytes::from_static(kind), Bytes::from(out.finish())]
    }

    fn decode(frames: &[Bytes]) -> Option<Self> {
        let [kind, payload] = frames else {
            return None;
        };
        let mut reader = Reader::new(payload);
        let text = |reader: &mut Reader<'_>| String::from_utf8(reader.bytes()?.to_vec()).ok();
        let message = match kind.as_ref() {
            PUSH => {
                let job = reader.u64()?;
                let reply_to = text(&mut reader)?;
                let session = text(&mut reader)?;
                let budget_us = reader.u64()?;
                let count = reader.count()?;
                let mut items = Vec::with_capacity(count.min(1 << 16));
                for _ in 0..count {
                    let group = reader.u32()?;
                    let block = i32::try_from(reader.i64()?).ok()?;
                    let spans_count = reader.count()?;
                    let mut spans = Vec::with_capacity(spans_count.min(1 << 16));
                    for _ in 0..spans_count {
                        spans.push(PushSpan {
                            offset: reader.u64()?,
                            address: reader.u64()?,
                            length: reader.u64()?,
                        });
                    }
                    items.push(PushItem {
                        group,
                        block,
                        spans,
                    });
                }
                Message::Push {
                    job,
                    reply_to,
                    session,
                    budget_us,
                    items,
                }
            }
            CANCEL => Message::Cancel {
                job: reader.u64()?,
                reply_to: text(&mut reader)?,
            },
            DONE => Message::Done {
                job: reader.u64()?,
                ok: reader.flag()?,
            },
            CANCELLED => Message::Cancelled { job: reader.u64()? },
            _ => return None,
        };
        reader.finish(message)
    }
}

/// Maximum staging slots a single push job may hold simultaneously, preventing
/// one large push from monopolizing the entire ring and starving sibling workers.
const MAX_INFLIGHT_SLOTS_PER_JOB: usize = 8;

fn max_inflight_slots_per_job(total_slots: usize) -> usize {
    (total_slots / 2).clamp(1, MAX_INFLIGHT_SLOTS_PER_JOB)
}

/// The ring's free slots.
struct Slots {
    total: usize,
    max_per_job: usize,
    free: Mutex<Vec<usize>>,
    returned: Condvar,
}

impl Slots {
    fn take(&self, deadline: Instant, cancelled: &AtomicBool) -> Option<SlotGuard<'_>> {
        let mut free = self.free.lock();
        loop {
            if cancelled.load(Ordering::Acquire) {
                return None;
            }
            if let Some(slot) = free.pop() {
                return Some(SlotGuard { slots: self, slot });
            }
            if Instant::now() >= deadline {
                return None;
            }
            if self.returned.wait_until(&mut free, deadline).timed_out() {
                if cancelled.load(Ordering::Acquire) {
                    return None;
                }
                return free.pop().map(|slot| SlotGuard { slots: self, slot });
            }
        }
    }

    fn try_take(&self) -> Option<SlotGuard<'_>> {
        self.free
            .lock()
            .pop()
            .map(|slot| SlotGuard { slots: self, slot })
    }

    fn give_back(&self, slot: usize) {
        self.free.lock().push(slot);
        self.returned.notify_one();
    }

    fn free_count(&self) -> usize {
        self.free.lock().len()
    }

    fn wake_all(&self) {
        let _guard = self.free.lock();
        self.returned.notify_all();
    }
}

/// RAII guard returning a staging slot on scope exit or panic.
struct SlotGuard<'a> {
    slots: &'a Slots,
    slot: usize,
}

impl SlotGuard<'_> {
    fn index(&self) -> usize {
        self.slot
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        self.slots.give_back(self.slot);
    }
}

/// Maximum completed or cancelled push job keys retained to fence duplicate or
/// late-arriving Push messages after a Cancel.
const MAX_FENCED_JOBS: usize = 16_384;
/// Upper bound on concurrent push worker threads per holder.
const MAX_PUSH_WORKERS: usize = 16;
/// Upper bound on queued pushes waiting for a worker before fast-failing.
const MAX_QUEUED_PUSHES: usize = 64;
/// Cap remote budgets so `Instant::now() + budget` can never overflow.
const MAX_PUSH_BUDGET: Duration = Duration::from_secs(3600);

/// Scopes a push job id to the reader endpoint that generated it so two
/// readers with colliding incarnations never interfere on the same holder.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct JobKey {
    reply_to: String,
    job: u64,
}

/// Bounded FIFO set of `(reply_to, job)` keys that may never start again.
#[derive(Default)]
struct FencedJobs {
    set: HashSet<JobKey>,
    order: VecDeque<JobKey>,
}

impl FencedJobs {
    fn contains(&self, key: &JobKey) -> bool {
        self.set.contains(key)
    }

    fn insert(&mut self, key: JobKey) {
        if self.set.insert(key.clone()) {
            self.order.push_back(key);
            while self.order.len() > MAX_FENCED_JOBS {
                if let Some(oldest) = self.order.pop_front() {
                    self.set.remove(&oldest);
                }
            }
        }
    }
}

/// Validates that every span has positive length, stays within the block's
/// byte range, and does not wrap the destination address space.
fn validate_spans(spans: &[PushSpan], block_bytes: u64) -> bool {
    spans.iter().all(|span| {
        span.length > 0
            && span
                .offset
                .checked_add(span.length)
                .is_some_and(|end| end <= block_bytes)
            && span.address.checked_add(span.length).is_some()
    })
}

enum CancelAck {
    Pending(Option<String>),
    Finished,
}

/// A push the holder runs, and whether the reader cancelled it.
struct Running {
    cancelled: AtomicBool,
    /// State machine preventing lost `Cancelled` ACKs when Cancel races finish.
    ack: Mutex<CancelAck>,
}

impl Running {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            ack: Mutex::new(CancelAck::Pending(None)),
        }
    }

    /// Marks the job cancelled; returns `Some(reply_to)` if the job already
    /// finished so the caller must send `Cancelled` immediately.
    fn cancel(&self, reply_to: String) -> Option<String> {
        self.cancelled.store(true, Ordering::Release);
        let mut ack = self.ack.lock();
        match &mut *ack {
            CancelAck::Pending(slot) => {
                *slot = Some(reply_to);
                None
            }
            CancelAck::Finished => Some(reply_to),
        }
    }

    /// Transitions to `Finished` and returns any pending cancellation target.
    fn finish(&self) -> Option<String> {
        let mut ack = self.ack.lock();
        match std::mem::replace(&mut *ack, CancelAck::Finished) {
            CancelAck::Pending(reply_to) => reply_to,
            CancelAck::Finished => None,
        }
    }
}

struct QueuedPush {
    guard: JobGuard,
    arrived: Instant,
    session: String,
    budget: Duration,
    items: Vec<PushItem>,
}

/// Owns a push job's registration in `HolderState.running`; dropping it always
/// fences the job and sends `Done` (and `Cancelled` if requested).
struct JobGuard {
    state: Arc<HolderState>,
    sockets: Weak<Sockets>,
    key: JobKey,
    running: Arc<Running>,
    ok: bool,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        {
            let mut running_map = self.state.running.lock();
            running_map.remove(&self.key);
            self.state.fenced.lock().insert(self.key.clone());
        }
        let ack = self.running.finish();
        if let Some(sockets) = self.sockets.upgrade() {
            send(
                &sockets,
                &self.key.reply_to,
                &Message::Done {
                    job: self.key.job,
                    ok: self.ok,
                },
            );
            if let Some(acknowledge) = ack {
                send(
                    &sockets,
                    &acknowledge,
                    &Message::Cancelled { job: self.key.job },
                );
            }
        }
    }
}

/// Shared holder state accessed by worker threads and the dispatcher.
struct HolderState {
    source: PushSource,
    slots: Slots,
    running: Mutex<HashMap<JobKey, Arc<Running>>>,
    fenced: Mutex<FencedJobs>,
}

/// The holder's half: a bounded worker pool serving Disk pushes.
struct Holder {
    state: Arc<HolderState>,
    queue: Mutex<Option<mpsc::SyncSender<QueuedPush>>>,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Holder {
    fn new(source: PushSource) -> Result<Self, String> {
        let slots = source.ring.size() / source.slot_bytes.max(1);
        if slots == 0 || source.slot_bytes % ts_disk::ALIGNMENT != 0 {
            return Err("the staging ring holds no aligned slot".to_owned());
        }
        if source
            .disk_layout
            .groups
            .iter()
            .any(|group| group.block_bytes > source.slot_bytes)
        {
            return Err("a Disk block is larger than a staging slot".to_owned());
        }
        let state = Arc::new(HolderState {
            source,
            slots: Slots {
                total: slots,
                max_per_job: max_inflight_slots_per_job(slots),
                free: Mutex::new((0..slots).collect()),
                returned: Condvar::new(),
            },
            running: Mutex::new(HashMap::new()),
            fenced: Mutex::new(FencedJobs::default()),
        });
        let worker_count = slots.clamp(1, MAX_PUSH_WORKERS);
        let (tx, rx) = mpsc::sync_channel::<QueuedPush>(MAX_QUEUED_PUSHES);
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let (state, rx) = (state.clone(), rx.clone());
            let worker = thread::Builder::new()
                .name(format!("ts-peer-push-job-{index}"))
                .spawn(move || loop {
                    let Ok(mut queued) = rx.lock().recv() else {
                        return;
                    };
                    let started = Instant::now();
                    let deadline = queued.arrived + queued.budget;
                    let ok = state.run(
                        &queued.session,
                        deadline,
                        &queued.items,
                        &queued.guard.running,
                    );
                    queued.guard.ok = ok;
                    tracing::debug!(
                        job = queued.guard.key.job,
                        blocks = queued.items.len(),
                        ok,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "peer push ended"
                    );
                })
                .map_err(|error| format!("starting peer push worker: {error}"))?;
            workers.push(worker);
        }
        Ok(Self {
            state,
            queue: Mutex::new(Some(tx)),
            workers: Mutex::new(workers),
        })
    }

    fn enqueue(
        &self,
        sockets: Weak<Sockets>,
        key: JobKey,
        session: String,
        budget: Duration,
        items: Vec<PushItem>,
    ) {
        let arrived = Instant::now();
        if self.state.source.executor.broken() {
            tracing::warn!(
                job = key.job,
                reader = key.reply_to,
                "rejecting peer push because holder Disk executor is broken"
            );
            if let Some(sockets) = sockets.upgrade() {
                send(
                    &sockets,
                    &key.reply_to,
                    &Message::Done {
                        job: key.job,
                        ok: false,
                    },
                );
            }
            return;
        }
        let running = Arc::new(Running::new());
        {
            let mut running_map = self.state.running.lock();
            if running_map.contains_key(&key) || self.state.fenced.lock().contains(&key) {
                tracing::warn!(
                    job = key.job,
                    reader = key.reply_to,
                    "rejecting duplicate or fenced peer push"
                );
                if let Some(sockets) = sockets.upgrade() {
                    send(
                        &sockets,
                        &key.reply_to,
                        &Message::Done {
                            job: key.job,
                            ok: false,
                        },
                    );
                }
                return;
            }
            running_map.insert(key.clone(), running.clone());
        }
        let guard = JobGuard {
            state: self.state.clone(),
            sockets,
            key,
            running,
            ok: false,
        };
        let queued = QueuedPush {
            guard,
            arrived,
            session,
            budget: budget.min(MAX_PUSH_BUDGET),
            items,
        };
        let queue = self.queue.lock();
        if let Some(tx) = queue.as_ref() {
            if let Err(rejected) = tx.try_send(queued) {
                match rejected {
                    mpsc::TrySendError::Full(_) => {
                        tracing::warn!("peer push queue full; rejecting");
                    }
                    mpsc::TrySendError::Disconnected(_) => {
                        tracing::warn!("peer push queue closed; rejecting");
                    }
                }
            }
        }
    }

    fn cancel(&self, sockets: &Sockets, key: JobKey) {
        let running = {
            let running_map = self.state.running.lock();
            if let Some(active) = running_map.get(&key) {
                Some(active.clone())
            } else {
                self.state.fenced.lock().insert(key.clone());
                None
            }
        };
        match running {
            Some(running) => {
                let immediate = running.cancel(key.reply_to);
                self.state.slots.wake_all();
                if let Some(reply_to) = immediate {
                    send(sockets, &reply_to, &Message::Cancelled { job: key.job });
                }
            }
            None => send(sockets, &key.reply_to, &Message::Cancelled { job: key.job }),
        }
    }

    /// Cancels all active jobs, closes the queue, and joins every worker.
    fn shutdown(&self) {
        self.queue.lock().take();
        {
            let running_map = self.state.running.lock();
            for running in running_map.values() {
                running.cancelled.store(true, Ordering::Release);
            }
        }
        self.state.slots.wake_all();
        for worker in self.workers.lock().drain(..) {
            let _ = worker.join();
        }
    }
}

impl HolderState {
    /// Runs one push to its end: every block read into a slot and written to
    /// the reader, unless the arrival-anchored `deadline` passes, the reader
    /// cancels or an I/O fails. Returns whether every block actually landed via
    /// RDMA write.
    fn run(&self, session: &str, deadline: Instant, items: &[PushItem], job: &Running) -> bool {
        // On a profile timeline: the push, and each block's write inside it.
        static PUSHES: Lanes = Lanes::new("Peer push");
        let lane = PUSHES.hold();
        let begin_ns = profile::now_ns();
        if job.cancelled.load(Ordering::Acquire) || Instant::now() >= deadline {
            return false;
        }
        if self.source.executor.broken() {
            return false;
        }

        // Pre-validate every item before issuing any disk read or RDMA write,
        // and group MR-split PushItems by unique `(group, block)` so each Disk
        // block is read from NVMe once regardless of how many destination MRs
        // its fields span on the reader.
        struct UniqueBlock<'a> {
            range: ts_l3::Range,
            batches: Vec<&'a [PushSpan]>,
        }
        let mut blocks: Vec<UniqueBlock<'_>> = Vec::with_capacity(items.len());
        let mut block_index: HashMap<(u32, i32), usize> = HashMap::with_capacity(items.len());
        for item in items {
            match self.source.disk_layout.owned_range(item.group, item.block) {
                Ok(Some(range)) if validate_spans(&item.spans, range.size as u64) => {
                    if item.spans.is_empty() {
                        continue;
                    }
                    if let Some(&idx) = block_index.get(&(item.group, item.block)) {
                        blocks[idx].batches.push(&item.spans);
                    } else {
                        let idx = blocks.len();
                        block_index.insert((item.group, item.block), idx);
                        blocks.push(UniqueBlock {
                            range,
                            batches: vec![&item.spans],
                        });
                    }
                }
                _ => {
                    tracing::warn!(
                        group = item.group,
                        block = item.block,
                        "peer push item has invalid block or out-of-bounds spans"
                    );
                    return false;
                }
            }
        }

        let ring = self.source.ring.address();
        let (done, completions) = mpsc::channel::<(usize, usize, bool)>();
        let mut next = 0;
        let mut written = 0;
        let mut inflight: HashMap<usize, (SlotGuard<'_>, JobId)> = HashMap::new();
        let mut cancelled_reads = false;
        let mut ok = true;
        let stopped =
            |ok: bool| !ok || job.cancelled.load(Ordering::Acquire) || Instant::now() >= deadline;
        loop {
            // Reads go ahead of the writes, bounded per job so one large push
            // never monopolizes every staging slot in the shared ring.
            while next < blocks.len() && inflight.len() < self.slots.max_per_job && !stopped(ok) {
                let slot = if inflight.is_empty() {
                    self.slots.take(deadline, &job.cancelled)
                } else {
                    self.slots.try_take()
                };
                let Some(slot_guard) = slot else { break };
                let slot_idx = slot_guard.index();
                let range = blocks[next].range;
                let done = done.clone();
                let index = next;
                let submitted = self.source.executor.submit(
                    Job {
                        direction: Direction::Read,
                        ios: vec![Io {
                            file_offset: range.offset as u64,
                            address: ring + slot_idx * self.source.slot_bytes,
                            len: range.size,
                        }],
                    },
                    move |report: JobReport| {
                        let _ = done.send((index, slot_idx, report.outcome.is_ok()));
                    },
                );
                match submitted {
                    Ok(job_id) => {
                        inflight.insert(slot_idx, (slot_guard, job_id));
                        next += 1;
                    }
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if stopped(ok) && !cancelled_reads {
                cancelled_reads = true;
                for &(_, job_id) in inflight.values() {
                    self.source.executor.cancel(job_id);
                }
            }
            if inflight.is_empty() {
                break;
            }
            let (index, slot_idx, read) = match completions.recv_timeout(Duration::from_millis(50))
            {
                Ok(item) => item,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self.source.executor.broken() {
                        tracing::error!(
                            inflight = inflight.len(),
                            "holder Disk executor broke down during peer push; retiring push without reusing unretired staging slots"
                        );
                        // The kernel may still write into these staging slots;
                        // leak their SlotGuards so no future push reuses them.
                        for (_, (slot_guard, _)) in inflight.drain() {
                            std::mem::forget(slot_guard);
                        }
                        ok = false;
                        break;
                    }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };
            let Some((_slot_guard, _job_id)) = inflight.remove(&slot_idx) else {
                continue;
            };
            ok &= read;
            if !stopped(ok) {
                let base = (ring + slot_idx * self.source.slot_bytes) as u64;
                let mut block_ok = true;
                for batch_spans in &blocks[index].batches {
                    if stopped(ok) {
                        block_ok = false;
                        ok = false;
                        break;
                    }
                    let spans = Spans {
                        sources: batch_spans.iter().map(|span| base + span.offset).collect(),
                        destinations: batch_spans.iter().map(|span| span.address).collect(),
                        lengths: batch_spans.iter().map(|span| span.length).collect(),
                        host_bytes: 0,
                    };
                    let write_ns = profile::now_ns();
                    if let Err(error) = self.source.writer.write(session, &spans) {
                        tracing::warn!(%error, slot = slot_idx, "peer push write failed");
                        block_ok = false;
                        ok = false;
                    }
                    if let Some(lane) = &lane {
                        lane.record(profile::HOST, "write".into(), write_ns, profile::now_ns());
                    }
                    if !block_ok {
                        break;
                    }
                }
                if block_ok {
                    written += 1;
                }
            } else {
                ok = false;
            }
        }
        let landed = ok && written == blocks.len() && !job.cancelled.load(Ordering::Acquire);
        if let Some(lane) = &lane {
            let bytes: u64 = items
                .iter()
                .flat_map(|item| &item.spans)
                .map(|span| span.length)
                .sum();
            let outcome = if landed { "landed" } else { "stopped" };
            lane.record(
                profile::HOST,
                format!(
                    "Peer push to {session}: {} blocks, {:.1} MiB, {outcome}",
                    blocks.len(),
                    bytes as f64 / (1 << 20) as f64
                )
                .into(),
                begin_ns,
                profile::now_ns(),
            );
        }
        landed
    }
}

/// A rank's push endpoint: it serves the pushes of a holder with a Disk tier
/// and receives the answers its own reads' pushes wait for.
pub struct PushEndpoint {
    name: String,
    sockets: Option<Arc<Sockets>>,
    holder: Arc<Mutex<Option<Arc<Holder>>>>,
    waiters: Arc<Mutex<HashMap<u64, mpsc::Sender<Message>>>>,
    next_job: AtomicU64,
    dispatcher: Option<thread::JoinHandle<()>>,
}

/// An endpoint that could not start.
#[derive(Debug, thiserror::Error)]
pub enum PushError {
    /// The socket did not bind.
    #[error("the peer push endpoint: {0}")]
    Net(#[from] NetError),
    /// A thread did not start.
    #[error("starting the peer push thread: {0}")]
    Thread(#[from] std::io::Error),
    /// The holder's settings are inconsistent.
    #[error("peer push settings: {0}")]
    Settings(String),
}

impl PushEndpoint {
    /// Bind on `bind_host` at `port` (0: the OS picks); other ranks reach it
    /// at `advertised_host`. `incarnation` keeps job ids of a restarted rank
    /// apart.
    pub fn bind(
        bind_host: &str,
        port: u16,
        advertised_host: &str,
        incarnation: u64,
    ) -> Result<Self, PushError> {
        let (incoming, received) = mpsc::channel::<Vec<Bytes>>();
        let sockets = Arc::new(Sockets::start_at(bind_host, port, incoming)?);
        let name = crate::peer::net::join_endpoint(advertised_host, sockets.port());
        let holder: Arc<Mutex<Option<Arc<Holder>>>> = Arc::new(Mutex::new(None));
        let waiters: Arc<Mutex<HashMap<u64, mpsc::Sender<Message>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let dispatcher = {
            let (sockets_weak, holder, waiters) =
                (Arc::downgrade(&sockets), holder.clone(), waiters.clone());
            thread::Builder::new()
                .name("ts-peer-push".to_owned())
                .spawn(move || {
                    for frames in received {
                        let Some(sockets) = sockets_weak.upgrade() else {
                            break;
                        };
                        match Message::decode(&frames) {
                            Some(Message::Push {
                                job,
                                reply_to,
                                session,
                                budget_us,
                                items,
                            }) => {
                                let serving = holder.lock().clone();
                                match serving {
                                    Some(h) => h.enqueue(
                                        sockets_weak.clone(),
                                        JobKey { reply_to, job },
                                        session,
                                        Duration::from_micros(budget_us),
                                        items,
                                    ),
                                    None => {
                                        send(&sockets, &reply_to, &Message::Done { job, ok: false })
                                    }
                                }
                            }
                            Some(Message::Cancel { job, reply_to }) => {
                                let serving = holder.lock().clone();
                                match serving {
                                    Some(h) => h.cancel(&sockets, JobKey { reply_to, job }),
                                    None => send(&sockets, &reply_to, &Message::Cancelled { job }),
                                }
                            }
                            Some(answer @ (Message::Done { .. } | Message::Cancelled { .. })) => {
                                let job = match answer {
                                    Message::Done { job, .. } | Message::Cancelled { job } => job,
                                    _ => unreachable!(),
                                };
                                if let Some(waiter) = waiters.lock().get(&job) {
                                    let _ = waiter.send(answer);
                                }
                            }
                            None => tracing::warn!("a malformed peer push message dropped"),
                        }
                    }
                })?
        };
        Ok(Self {
            name,
            sockets: Some(sockets),
            holder,
            waiters,
            next_job: AtomicU64::new(incarnation << 32),
            dispatcher: Some(dispatcher),
        })
    }

    /// Where other ranks reach this endpoint.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Serve pushes of `source`'s Disk pages from now on.
    pub fn serve(&self, source: PushSource) -> Result<(), PushError> {
        let next = Arc::new(Holder::new(source).map_err(PushError::Settings)?);
        let prev = self.holder.lock().replace(next);
        if let Some(prev) = prev {
            prev.shutdown();
        }
        Ok(())
    }

    /// `(free, total)` staging-ring slot counts on the active holder, if serving.
    pub fn push_ring_slot_counts(&self) -> Option<(usize, usize)> {
        let holder = self.holder.lock().clone()?;
        let free = holder.state.slots.free_count();
        Some((free, holder.state.slots.total))
    }

    /// Number of currently registered active push jobs on the holder.
    pub fn active_push_jobs(&self) -> usize {
        self.holder
            .lock()
            .as_ref()
            .map_or(0, |holder| holder.state.running.lock().len())
    }
}

impl Drop for PushEndpoint {
    fn drop(&mut self) {
        if let Some(holder) = self.holder.lock().take() {
            holder.shutdown();
        }
        self.sockets.take();
        if let Some(dispatcher) = self.dispatcher.take() {
            let _ = dispatcher.join();
        }
    }
}

impl PagePusher for PushEndpoint {
    fn push(
        &self,
        endpoint: &str,
        session: &str,
        budget: Duration,
        items: Vec<PushItem>,
    ) -> Result<(), String> {
        let sockets = self
            .sockets
            .as_ref()
            .ok_or_else(|| "push endpoint is closed".to_owned())?;
        let budget = budget.min(MAX_PUSH_BUDGET);
        let job = self.next_job.fetch_add(1, Ordering::Relaxed);
        let (answered, answers) = mpsc::channel();
        self.waiters.lock().insert(job, answered);
        let _waiting = Waiting {
            waiters: &self.waiters,
            job,
        };
        let request = Message::Push {
            job,
            reply_to: self.name.clone(),
            session: session.to_owned(),
            budget_us: u64::try_from(budget.as_micros()).unwrap_or(u64::MAX),
            items,
        };
        let (host, port) = split_endpoint(endpoint).map_err(|error| error.to_string())?;
        sockets
            .send(host, port, request.frames())
            .map_err(|error| error.to_string())?;
        // The holder's answer may trail its budget by a network round trip.
        match answers.recv_timeout(budget.saturating_add(Duration::from_secs(1))) {
            Ok(Message::Done { ok: true, .. }) => return Ok(()),
            Ok(Message::Done { ok: false, .. }) => {
                return Err("the holder could not push".to_owned());
            }
            _ => {}
        }
        let cancel = Message::Cancel {
            job,
            reply_to: self.name.clone(),
        };
        let _ = sockets.send(host, port, cancel.frames());
        let deadline = Instant::now() + CANCEL_WAIT;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            match answers.recv_timeout(remaining) {
                Ok(Message::Done { ok: true, .. }) => return Ok(()),
                Ok(Message::Cancelled { .. } | Message::Done { .. }) => {
                    return Err("the push did not finish in its budget".to_owned());
                }
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        Err(
            "the push did not finish in its budget, and its cancellation went unacknowledged"
                .to_owned(),
        )
    }
}

/// Unregisters a push's waiter however its wait ends.
struct Waiting<'a> {
    waiters: &'a Mutex<HashMap<u64, mpsc::Sender<Message>>>,
    job: u64,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.waiters.lock().remove(&self.job);
    }
}

fn send(sockets: &Sockets, endpoint: &str, message: &Message) {
    let sent = split_endpoint(endpoint)
        .map_err(|error| error.to_string())
        .and_then(|(host, port)| {
            sockets
                .send(host, port, message.frames())
                .map_err(|error| error.to_string())
        });
    if let Err(error) = sent {
        tracing::warn!(endpoint, %error, "peer push answer not sent");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_codec_validation_fencing_and_cancel_race() {
        for msg in [
            Message::Push {
                job: (7 << 32) + 1,
                reply_to: "10.0.0.3:7200".into(),
                session: "10.0.0.3:12345".into(),
                budget_us: 2_000_000,
                items: vec![PushItem {
                    group: 1,
                    block: 42,
                    spans: vec![PushSpan {
                        offset: 0,
                        address: 1 << 40,
                        length: 4096,
                    }],
                }],
            },
            Message::Cancel {
                job: 3,
                reply_to: "[::1]:7200".into(),
            },
            Message::Done { job: 3, ok: true },
            Message::Done { job: 4, ok: false },
            Message::Cancelled { job: 3 },
        ] {
            assert_eq!(Message::decode(&msg.frames()), Some(msg));
        }
        assert_eq!(
            Message::decode(&[Bytes::from_static(b"?"), Bytes::new()]),
            None
        );

        assert!(validate_spans(
            &[PushSpan {
                offset: 0,
                address: 1 << 20,
                length: 4096
            }],
            4096
        ));
        assert!(!validate_spans(
            &[PushSpan {
                offset: 0,
                address: 1 << 20,
                length: 0
            }],
            4096
        ));
        assert!(!validate_spans(
            &[PushSpan {
                offset: 2048,
                address: 1 << 20,
                length: 4096
            }],
            4096
        ));
        assert!(!validate_spans(
            &[PushSpan {
                offset: u64::MAX - 16,
                address: 1 << 20,
                length: 32
            }],
            4096
        ));
        assert!(!validate_spans(
            &[PushSpan {
                offset: 0,
                address: u64::MAX - 16,
                length: 32
            }],
            4096
        ));

        let mut fenced = FencedJobs::default();
        let a = JobKey {
            reply_to: "10.0.0.1:7200".into(),
            job: 42,
        };
        let b = JobKey {
            reply_to: "10.0.0.2:7200".into(),
            job: 42,
        };
        fenced.insert(a.clone());
        assert!(fenced.contains(&a) && !fenced.contains(&b));
        for job in 0..(MAX_FENCED_JOBS as u64 + 2) {
            fenced.insert(JobKey {
                reply_to: "10.0.0.3:7200".into(),
                job,
            });
        }
        assert!(!fenced.contains(&a));
        assert!(fenced.contains(&JobKey {
            reply_to: "10.0.0.3:7200".into(),
            job: MAX_FENCED_JOBS as u64 + 1
        }));

        let r1 = Running::new();
        assert_eq!(r1.cancel("r1".into()), None);
        assert_eq!(r1.finish(), Some("r1".into()));
        let r2 = Running::new();
        assert_eq!(r2.finish(), None);
        assert_eq!(r2.cancel("r2".into()), Some("r2".into()));
    }

    #[test]
    fn split_mr_dedup_slot_reclaim_and_queue_overflow() {
        use std::{path::PathBuf, sync::atomic::AtomicUsize};
        use ts_disk::{Backend, DiskFile, ExecutorConfig, Memory};
        use ts_l3::{GroupLayout, HostLayout};

        const BLOCK: usize = 4096;
        struct TempDisk(PathBuf);
        impl Drop for TempDisk {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }

        let dir = std::env::var_os("TS_DISK_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let temp = TempDisk(dir.join(format!(
            "ts-push-{}-{}.bin",
            std::process::id(),
            profile::now_ns()
        )));
        let disk_file = DiskFile::create(&temp.0, (4 * BLOCK) as u64).unwrap();
        let ring = Arc::new(Memory::map(BLOCK).unwrap());
        let mut seed_mem = Memory::map(2 * BLOCK).unwrap();
        let pattern: Vec<u8> = (0..BLOCK).map(|i| ((i * 7 + 3) & 0xff) as u8).collect();
        seed_mem.write_at(0, &pattern);
        seed_mem.write_at(BLOCK, &[0xBB; BLOCK]);

        let executor = Arc::new(
            ts_disk::Executor::start(
                disk_file,
                &[
                    (ring.address(), ring.size()),
                    (seed_mem.address(), seed_mem.size()),
                ],
                ExecutorConfig {
                    backend: Backend::Psync,
                    queue_depth: 2,
                    write_queue_depth: 1,
                    max_io_bytes: 1 << 20,
                    write_bytes_per_second: 1 << 30,
                },
            )
            .unwrap(),
        );
        let (tx, rx) = mpsc::channel();
        executor
            .submit(
                Job {
                    direction: Direction::Write,
                    ios: vec![Io {
                        file_offset: 0,
                        address: seed_mem.address(),
                        len: 2 * BLOCK,
                    }],
                },
                move |r| {
                    let _ = tx.send(r.outcome);
                },
            )
            .unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), Ok(()));

        type RecordedWrite = Vec<(u64, Vec<u8>)>;
        struct MockWriter {
            ring: Arc<Memory>,
            fail: AtomicBool,
            calls: Mutex<Vec<RecordedWrite>>,
            hold: (std::sync::Mutex<bool>, std::sync::Condvar),
            entered: AtomicUsize,
        }
        impl PageWriter for MockWriter {
            fn write(&self, _: &str, spans: &Spans) -> Result<(), String> {
                self.entered.fetch_add(1, Ordering::SeqCst);
                let held = self.hold.0.lock().unwrap();
                let _held = self.hold.1.wait_while(held, |b| *b).unwrap();
                if self.fail.load(Ordering::Acquire) {
                    return Err("simulated RDMA fault".into());
                }
                let mut batch = Vec::new();
                for ((&src, &dst), &len) in spans
                    .sources
                    .iter()
                    .zip(&spans.destinations)
                    .zip(&spans.lengths)
                {
                    let mut buf = vec![0u8; len as usize];
                    self.ring
                        .read_at(src as usize - self.ring.address(), &mut buf);
                    batch.push((dst, buf));
                }
                self.calls.lock().push(batch);
                Ok(())
            }
        }

        let writer = Arc::new(MockWriter {
            ring: ring.clone(),
            fail: AtomicBool::new(false),
            calls: Mutex::new(Vec::new()),
            hold: (std::sync::Mutex::new(false), std::sync::Condvar::new()),
            entered: AtomicUsize::new(0),
        });
        let holder = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 1).unwrap();
        holder
            .serve(PushSource {
                executor: executor.clone(),
                disk_layout: HostLayout {
                    groups: vec![GroupLayout {
                        packing: 1,
                        block_bytes: BLOCK,
                        shard_count: 1,
                    }],
                    lcm_block_bytes: BLOCK,
                    num_lcm_blocks: 4,
                    owner_rank: 0,
                    tp_rank: 0,
                },
                writer: writer.clone(),
                ring: ring.clone(),
                slot_bytes: BLOCK,
            })
            .unwrap();
        let reader = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 2).unwrap();

        // 1. 3 MR-split PushItems for block 1 -> 1 deduplicated NVMe read + 3 MR write batches.
        let split = vec![
            PushItem {
                group: 0,
                block: 1,
                spans: vec![PushSpan {
                    offset: 0,
                    address: 100_000,
                    length: 1024,
                }],
            },
            PushItem {
                group: 0,
                block: 1,
                spans: vec![PushSpan {
                    offset: 1024,
                    address: 200_000,
                    length: 1024,
                }],
            },
            PushItem {
                group: 0,
                block: 1,
                spans: vec![PushSpan {
                    offset: 2048,
                    address: 300_000,
                    length: 2048,
                }],
            },
        ];
        assert_eq!(
            reader.push(holder.name(), "s1", Duration::from_secs(5), split),
            Ok(())
        );
        assert_eq!(executor.stats().read_jobs, 1);
        assert_eq!(writer.calls.lock().len(), 3);

        // 2. Write fault to one unreachable reader immediately returns the staging slot
        //    so healthy readers are never starved.
        writer.fail.store(true, Ordering::Release);
        let one = |blk| {
            vec![PushItem {
                group: 0,
                block: blk,
                spans: vec![PushSpan {
                    offset: 0,
                    address: 400_000,
                    length: BLOCK as u64,
                }],
            }]
        };
        assert!(reader
            .push(holder.name(), "s2", Duration::from_secs(2), one(1))
            .is_err());
        assert_eq!(holder.push_ring_slot_counts(), Some((1, 1)));
        writer.fail.store(false, Ordering::Release);
        assert_eq!(
            reader.push(holder.name(), "s3", Duration::from_secs(2), one(2)),
            Ok(())
        );

        // 3. Overflowing 1 worker + MAX_QUEUED_PUSHES rejects cleanly without blocking.
        *writer.hold.0.lock().unwrap() = true;
        writer.entered.store(0, Ordering::SeqCst);
        std::thread::scope(|scope| {
            let mut hs = Vec::new();
            for i in 0..(1 + MAX_QUEUED_PUSHES) {
                let (r, t) = (&reader, holder.name().to_owned());
                hs.push(
                    scope.spawn(move || {
                        r.push(&t, &format!("q{i}"), Duration::from_secs(5), one(1))
                    }),
                );
            }
            let wait = Instant::now();
            while writer.entered.load(Ordering::Acquire) < 1
                || holder.active_push_jobs() < 1 + MAX_QUEUED_PUSHES
            {
                assert!(wait.elapsed() < Duration::from_secs(5));
                thread::sleep(Duration::from_millis(2));
            }
            assert!(reader
                .push(holder.name(), "ovf", Duration::from_secs(5), one(1))
                .is_err());
            *writer.hold.0.lock().unwrap() = false;
            writer.hold.1.notify_all();
            for h in hs {
                assert_eq!(h.join().unwrap(), Ok(()));
            }
        });
    }

    #[test]
    fn cancel_ack_late_done_ok_and_wake_all() {
        let (htx, hrx) = mpsc::channel();
        let mock = Arc::new(Sockets::start_at("127.0.0.1", 0, htx).unwrap());
        let addr = format!("127.0.0.1:{}", mock.port());
        let reader = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 41).unwrap();
        let m2 = mock.clone();
        let driver = thread::spawn(move || {
            let Message::Push {
                job: j1,
                reply_to: r1,
                ..
            } = Message::decode(&hrx.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap()
            else {
                panic!()
            };
            assert_eq!(
                Message::decode(&hrx.recv_timeout(Duration::from_secs(5)).unwrap()),
                Some(Message::Cancel {
                    job: j1,
                    reply_to: r1.clone()
                })
            );
            send(&m2, &r1, &Message::Cancelled { job: j1 });
            let Message::Push {
                job: j2,
                reply_to: r2,
                ..
            } = Message::decode(&hrx.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap()
            else {
                panic!()
            };
            let _ = hrx.recv_timeout(Duration::from_secs(5)).unwrap();
            send(&m2, &r2, &Message::Done { job: j2, ok: true });
        });
        let items = vec![PushItem {
            group: 0,
            block: 1,
            spans: vec![PushSpan {
                offset: 0,
                address: 1000,
                length: 4096,
            }],
        }];
        assert!(reader
            .push(&addr, "s", Duration::from_millis(50), items.clone())
            .is_err());
        assert_eq!(
            reader.push(&addr, "s", Duration::from_millis(50), items),
            Ok(())
        );
        driver.join().unwrap();

        let slots = Arc::new(Slots {
            total: 1,
            max_per_job: 1,
            free: Mutex::new(Vec::new()),
            returned: Condvar::new(),
        });
        let cancelled = Arc::new(AtomicBool::new(false));
        let (sw, cw) = (slots.clone(), cancelled.clone());
        let w = thread::spawn(move || {
            sw.take(Instant::now() + Duration::from_secs(30), &cw)
                .is_none()
        });
        thread::yield_now();
        cancelled.store(true, Ordering::Release);
        slots.wake_all();
        assert!(w.join().unwrap());
    }
}
