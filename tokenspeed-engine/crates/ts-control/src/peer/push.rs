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
/// Offer-bound v2 push frame carrying `(holder_incarnation, offer_id)` for v1
/// readers; legacy readers continue speaking `PUSH` without wire breakage.
const PUSH_V2: &[u8] = b"P";
const CANCEL: &[u8] = b"c";
const DONE: &[u8] = b"d";
const CANCELLED: &[u8] = b"x";
const UNKNOWN: &[u8] = b"u";

/// How long a reader waits for a cancelled push's acknowledgment (legacy sync push).
const CANCEL_WAIT: Duration = Duration::from_secs(5);
/// How often a multi-holder push polls the reader's seal flag and cancel timers.
const SEAL_POLL_INTERVAL: Duration = Duration::from_millis(20);
/// Grace window before an unacknowledged multi-holder cancel signals early freeze
/// so the business request falls back locally while cancels retry in background.
const CANCEL_FREEZE_WAIT: Duration = Duration::from_secs(1);
/// Initial retry interval for unacknowledged cancels, doubling up to `CANCEL_RETRY_MAX`.
const CANCEL_RETRY_INITIAL: Duration = Duration::from_millis(250);
const CANCEL_RETRY_MAX: Duration = Duration::from_secs(5);
/// Upper bound on background cancel retries before concluding the holder is
/// permanently gone (e.g. crashed/restarted onto a new ephemeral port) and
/// resolving as `PushOutcome::Unknown` so the landing stays safely frozen
/// without leaking reader state or spinning forever.
const CANCEL_RETRY_BUDGET: Duration = Duration::from_secs(60);

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

/// One holder rank's Disk push request within a multi-holder transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushTarget {
    /// The holder rank's push endpoint.
    pub endpoint: String,
    /// This member's transfer session, which the holder writes into.
    pub session: String,
    /// Expected holder incarnation authorizing `offer_id` (ignored when `offer_id == 0`).
    pub holder_incarnation: u64,
    /// The v1 offer id authorizing this push on the holder (`0` when untagged).
    pub offer_id: u64,
    /// The Disk blocks and destination spans to push.
    pub items: Vec<PushItem>,
}

impl PushTarget {
    /// Payload bytes this target pushes.
    pub fn bytes(&self) -> u64 {
        self.items
            .iter()
            .flat_map(|item| &item.spans)
            .map(|span| span.length)
            .sum()
    }
}

/// Retirement evidence for one or more holder Disk pushes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    /// Every pushed block landed and every remote write retired.
    Complete,
    /// Not all blocks landed, but every holder proved no writes remain in
    /// flight (or nothing was submitted).
    Aborted(String),
    /// Retirement could not be established (a remote write failed or a
    /// cancellation went unacknowledged). Keep landing and source pinned.
    Unknown(String),
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

    /// Run one or more Disk pushes under a shared cancellation flag and
    /// submission deadline, distinguishing cleanly retired aborts from unknown
    /// remote write retirement.
    fn push_many(
        &self,
        targets: Vec<PushTarget>,
        budget: Duration,
        submit_until: Instant,
        sealed: &AtomicBool,
    ) -> PushOutcome {
        for target in targets {
            if sealed.load(Ordering::SeqCst) || Instant::now() >= submit_until {
                return PushOutcome::Aborted("peer push sealed before submission".to_owned());
            }
            let remaining = submit_until.saturating_duration_since(Instant::now());
            let job_budget = budget.min(remaining);
            if job_budget.is_zero() {
                return PushOutcome::Aborted("peer push submission window elapsed".to_owned());
            }
            if let Err(error) =
                self.push(&target.endpoint, &target.session, job_budget, target.items)
            {
                return PushOutcome::Unknown(error);
            }
        }
        if sealed.load(Ordering::SeqCst) {
            PushOutcome::Aborted("peer push sealed".to_owned())
        } else {
            PushOutcome::Complete
        }
    }

    /// Authorize holder-side Disk pushes tagged with `(incarnation, offer_id)`
    /// once the offer's blocks are pinned on this member.
    fn activate_offer(&self, _incarnation: u64, _offer_id: u64) {}

    /// Revoke authorization for `(incarnation, offer_id)` and signal any
    /// queued/in-flight holder-side Disk pushes tagged with it to stop. Returns
    /// `true` only once zero pushes for that offer remain in flight (so
    /// scheduler Disk pins may be safely released).
    fn cancel_offer(&self, _incarnation: u64, _offer_id: u64) -> bool {
        true
    }

    /// Start one or more Disk pushes without blocking the caller thread.
    /// Sets `cancelling` to `true` once an unacknowledged cancel exceeds its
    /// grace period (so the reader can freeze early and fall back locally while
    /// cancels retry in the background) and invokes `finish` once every holder
    /// reports completion, clean abort, or unknown write retirement.
    fn start_push_many(
        &self,
        targets: Vec<PushTarget>,
        budget: Duration,
        submit_until: Instant,
        sealed: Arc<AtomicBool>,
        _cancelling: Arc<AtomicBool>,
        finish: Box<dyn FnOnce(PushOutcome) + Send>,
    ) {
        let outcome = self.push_many(targets, budget, submit_until, &sealed);
        finish(outcome);
    }
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
        /// `Some((holder_incarnation, offer_id))` encodes as `PUSH_V2`; `None`
        /// encodes as legacy `PUSH` for wire compatibility.
        offer: Option<(u64, u64)>,
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
    Unknown {
        job: u64,
    },
}

impl Message {
    fn frames(&self) -> Vec<Bytes> {
        let mut out = Writer::new();
        let kind = match self {
            Message::Push {
                job,
                offer,
                reply_to,
                session,
                budget_us,
                items,
            } => {
                out.u64(*job);
                let tag = if let Some((incarnation, offer_id)) = offer {
                    out.u64(*incarnation);
                    out.u64(*offer_id);
                    PUSH_V2
                } else {
                    PUSH
                };
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
                tag
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
            Message::Unknown { job } => {
                out.u64(*job);
                UNKNOWN
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
            PUSH | PUSH_V2 => {
                let job = reader.u64()?;
                let offer = if kind.as_ref() == PUSH_V2 {
                    Some((reader.u64()?, reader.u64()?))
                } else {
                    None
                };
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
                    offer,
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
            UNKNOWN => Message::Unknown { job: reader.u64()? },
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
    #[cfg(test)]
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
        // Wake every blocked waiter: a single wakeup could land on a waiter
        // that just saw `cancelled` and returns `None` without popping.
        self.returned.notify_all();
    }

    fn free_count(&self) -> usize {
        self.free.lock().len()
    }

    fn wake_all(&self) {
        let _guard = self.free.lock();
        self.returned.notify_all();
    }
}

/// RAII guard returning a staging slot on normal scope exit; leaks on panic
/// so an in-flight NVMe DMA read cannot write into a reused slot.
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
        if !std::thread::panicking() {
            self.slots.give_back(self.slot);
        }
    }
}

/// Maximum completed or cancelled clean push job keys retained to fence
/// duplicate or late-arriving Push messages after a Cancel.
const MAX_FENCED_JOBS: usize = 16_384;
/// Maximum `UnknownRetirement` job keys stored explicitly; if exceeded, the
/// holder conservatively answers `Unknown` for any unrecognized Cancel so an
/// unknown RDMA write retirement is never downgraded to `Cancelled`.
const MAX_UNKNOWN_FENCED_JOBS: usize = 65_536;
/// Upper bound on concurrent push worker threads per holder.
const MAX_PUSH_WORKERS: usize = 16;
/// Upper bound on queued pushes waiting for a worker before fast-failing.
/// Sized for 64 concurrent TP8/KVP8 landings (8 reader ranks per holder rank).
const MAX_QUEUED_PUSHES: usize = 512;
/// Cap remote budgets so `Instant::now() + budget` can never overflow.
const MAX_PUSH_BUDGET: Duration = Duration::from_secs(3600);

/// Scopes a push job id to the reader endpoint that generated it so two
/// readers with colliding incarnations never interfere on the same holder.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct JobKey {
    reply_to: String,
    job: u64,
}

/// Set of `(reply_to, job)` keys that may never start again.
///
/// Clean completions/cancellations use a bounded FIFO (`MAX_FENCED_JOBS`).
/// `UnknownRetirement` entries are NEVER FIFO-evicted by normal push traffic so
/// a late `Cancel` retry can never misclassify an RDMA write fault as cleanly
/// cancelled (`Cancelled`), which would cause premature reader unfreeze.
#[derive(Default)]
struct FencedJobs {
    clean: HashSet<JobKey>,
    order: VecDeque<JobKey>,
    unknown: HashSet<JobKey>,
    unknown_overflowed: bool,
}

impl FencedJobs {
    fn contains(&self, key: &JobKey) -> bool {
        self.unknown.contains(key) || self.clean.contains(key)
    }

    fn unknown(&self, key: &JobKey) -> bool {
        self.unknown.contains(key) || (self.unknown_overflowed && !self.clean.contains(key))
    }

    fn insert(&mut self, key: JobKey, unknown: bool) {
        if unknown {
            self.clean.remove(&key);
            if self.unknown.contains(&key) {
                return;
            }
            if self.unknown.len() < MAX_UNKNOWN_FENCED_JOBS {
                self.unknown.insert(key);
            } else {
                self.unknown_overflowed = true;
            }
            return;
        }
        if self.unknown.contains(&key) || !self.clean.insert(key.clone()) {
            return;
        }
        self.order.push_back(key);
        while self.order.len() > MAX_FENCED_JOBS {
            if let Some(oldest) = self.order.pop_front() {
                self.clean.remove(&oldest);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunOutcome {
    Landed,
    RetiredFailed,
    UnknownRetirement,
}

enum CancelAck {
    Pending(Option<String>),
    Finished { unknown: bool },
}

/// A push the holder runs, and whether the reader cancelled it.
struct Running {
    offer: Option<(u64, u64)>,
    cancelled: AtomicBool,
    /// State machine preventing lost `Cancelled`/`Unknown` ACKs when Cancel races finish.
    ack: Mutex<CancelAck>,
}

impl Running {
    fn new(offer: Option<(u64, u64)>) -> Self {
        Self {
            offer,
            cancelled: AtomicBool::new(false),
            ack: Mutex::new(CancelAck::Pending(None)),
        }
    }

    /// Marks the job cancelled; returns `Some((reply_to, unknown))` if the job
    /// already finished so the caller must answer immediately.
    fn cancel(&self, reply_to: String) -> Option<(String, bool)> {
        self.cancelled.store(true, Ordering::Release);
        let mut ack = self.ack.lock();
        match &mut *ack {
            CancelAck::Pending(slot) => {
                *slot = Some(reply_to);
                None
            }
            CancelAck::Finished { unknown } => Some((reply_to, *unknown)),
        }
    }

    /// Transitions to `Finished` and returns any pending cancellation target.
    fn finish(&self, unknown: bool) -> Option<String> {
        let mut ack = self.ack.lock();
        match std::mem::replace(&mut *ack, CancelAck::Finished { unknown }) {
            CancelAck::Pending(reply_to) => reply_to,
            CancelAck::Finished { .. } => None,
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
/// fences the job and sends `Done`/`Unknown` (and `Cancelled`/`Unknown` if requested).
struct JobGuard {
    state: Arc<HolderState>,
    sockets: Weak<Sockets>,
    key: JobKey,
    running: Arc<Running>,
    outcome: RunOutcome,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        let unknown = matches!(self.outcome, RunOutcome::UnknownRetirement);
        {
            let mut running_map = self.state.running.lock();
            running_map.remove(&self.key);
            self.state.fenced.lock().insert(self.key.clone(), unknown);
        }
        let ack = self.running.finish(unknown);
        if let Some(sockets) = self.sockets.upgrade() {
            let reply = match self.outcome {
                RunOutcome::Landed => Message::Done {
                    job: self.key.job,
                    ok: true,
                },
                RunOutcome::RetiredFailed => Message::Done {
                    job: self.key.job,
                    ok: false,
                },
                RunOutcome::UnknownRetirement => Message::Unknown { job: self.key.job },
            };
            send(&sockets, &self.key.reply_to, &reply);
            if let Some(acknowledge) = ack {
                let cancel_reply = if unknown {
                    Message::Unknown { job: self.key.job }
                } else {
                    Message::Cancelled { job: self.key.job }
                };
                send(&sockets, &acknowledge, &cancel_reply);
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
    active_offers: Mutex<HashSet<(u64, u64)>>,
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
                #[cfg(test)]
                total: slots,
                max_per_job: max_inflight_slots_per_job(slots),
                free: Mutex::new((0..slots).collect()),
                returned: Condvar::new(),
            },
            running: Mutex::new(HashMap::new()),
            fenced: Mutex::new(FencedJobs::default()),
            active_offers: Mutex::new(HashSet::new()),
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
                    // Isolate a per-job panic so a fixed-size worker pool is never
                    // permanently drained by an unexpected fault in a single push
                    // (e.g. an external writer/FFI panic). During unwind, `SlotGuard`
                    // intentionally leaks any staging slot with in-flight NVMe DMA
                    // rather than reusing it, while `JobGuard` still fails the job
                    // back to the reader.
                    let outcome =
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            state.run(
                                &queued.session,
                                deadline,
                                &queued.items,
                                &queued.guard.running,
                            )
                        })) {
                            Ok(outcome) => outcome,
                            Err(_) => {
                                tracing::error!(
                                job = queued.guard.key.job,
                                worker = index,
                                free_slots = state.slots.free_count(),
                                "peer push worker caught panic; failing job without retiring worker"
                            );
                                RunOutcome::UnknownRetirement
                            }
                        };
                    queued.guard.outcome = outcome;
                    tracing::debug!(
                        job = queued.guard.key.job,
                        blocks = queued.items.len(),
                        ?outcome,
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
        offer: Option<(u64, u64)>,
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
        let running = Arc::new(Running::new(offer));
        {
            let mut running_map = self.state.running.lock();
            let offer_unauthorized = match offer {
                Some(pair) => !self.state.active_offers.lock().contains(&pair),
                None => false,
            };
            if offer_unauthorized
                || running_map.contains_key(&key)
                || self.state.fenced.lock().contains(&key)
            {
                tracing::warn!(
                    job = key.job,
                    ?offer,
                    reader = key.reply_to,
                    "rejecting duplicate, fenced, or unauthorized-offer peer push"
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
            outcome: RunOutcome::RetiredFailed,
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
        let mut fenced_unknown = false;
        let running = {
            let running_map = self.state.running.lock();
            if let Some(active) = running_map.get(&key) {
                Some(active.clone())
            } else {
                let mut fenced = self.state.fenced.lock();
                fenced_unknown = fenced.unknown(&key);
                if !fenced.contains(&key) {
                    fenced.insert(key.clone(), false);
                }
                None
            }
        };
        match running {
            Some(running) => {
                let immediate = running.cancel(key.reply_to);
                self.state.slots.wake_all();
                if let Some((reply_to, unknown)) = immediate {
                    let msg = if unknown {
                        Message::Unknown { job: key.job }
                    } else {
                        Message::Cancelled { job: key.job }
                    };
                    send(sockets, &reply_to, &msg);
                }
            }
            None => {
                let msg = if fenced_unknown {
                    Message::Unknown { job: key.job }
                } else {
                    Message::Cancelled { job: key.job }
                };
                send(sockets, &key.reply_to, &msg);
            }
        }
    }

    /// Authorizes `(incarnation, offer_id)` to serve Disk pushes once pinned.
    fn activate_offer(&self, incarnation: u64, offer_id: u64) {
        if offer_id == 0 {
            return;
        }
        self.state
            .active_offers
            .lock()
            .insert((incarnation, offer_id));
    }

    /// Revokes authorization for `(incarnation, offer_id)` and signals all
    /// queued/active jobs tagged with it to stop. Returns `true` only once zero
    /// jobs for that offer remain in `running` (so Disk block pins may be released).
    fn cancel_offer(&self, incarnation: u64, offer_id: u64) -> bool {
        if offer_id == 0 {
            return true;
        }
        let pair = (incarnation, offer_id);
        let drained = {
            let running_map = self.state.running.lock();
            self.state.active_offers.lock().remove(&pair);
            let mut active = false;
            for running in running_map.values() {
                if running.offer == Some(pair) {
                    running.cancelled.store(true, Ordering::Release);
                    active = true;
                }
            }
            !active
        };
        self.state.slots.wake_all();
        drained
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
    /// cancels or an I/O fails. Reports whether every block landed, or whether
    /// an RDMA write error left remote write retirement unknown.
    fn run(
        &self,
        session: &str,
        deadline: Instant,
        items: &[PushItem],
        job: &Running,
    ) -> RunOutcome {
        // On a profile timeline: the push, and each block's write inside it.
        static PUSHES: Lanes = Lanes::new("Peer push");
        let lane = PUSHES.hold();
        let begin_ns = profile::now_ns();
        if job.cancelled.load(Ordering::Acquire) || Instant::now() >= deadline {
            return RunOutcome::RetiredFailed;
        }
        if self.source.executor.broken() {
            return RunOutcome::RetiredFailed;
        }

        // Pre-validate every item before issuing any disk read or RDMA write,
        // and group PushItems by unique `(group, block)` so each Disk
        // block is read from NVMe once even if multiple items refer to it.
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
                    return RunOutcome::RetiredFailed;
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
        let mut write_failed = false;
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
                        write_failed = true;
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
        if write_failed {
            RunOutcome::UnknownRetirement
        } else if landed {
            RunOutcome::Landed
        } else {
            RunOutcome::RetiredFailed
        }
    }
}

#[derive(Clone, Copy)]
struct CancelTiming {
    freeze_wait: Duration,
    retry_initial: Duration,
    retry_max: Duration,
    retry_budget: Duration,
}

impl Default for CancelTiming {
    fn default() -> Self {
        Self {
            freeze_wait: CANCEL_FREEZE_WAIT,
            retry_initial: CANCEL_RETRY_INITIAL,
            retry_max: CANCEL_RETRY_MAX,
            retry_budget: CANCEL_RETRY_BUDGET,
        }
    }
}

struct NewPushGroup {
    group_id: u64,
    requests: Vec<(u64, String, u16, Message)>,
    budget: Duration,
    sealed: Arc<AtomicBool>,
    cancelling: Arc<AtomicBool>,
    finish: Box<dyn FnOnce(PushOutcome) + Send>,
}

enum GroupPhase {
    Waiting {
        wait_deadline: Instant,
    },
    Cancelling {
        freeze_at: Instant,
        next_retry: Instant,
        retry_interval: Duration,
        give_up_at: Instant,
        freeze_signaled: bool,
    },
}

struct PushGroup {
    pending: HashMap<u64, (String, u16)>,
    sealed: Arc<AtomicBool>,
    cancelling: Arc<AtomicBool>,
    abort_reason: Option<String>,
    unknown_reason: Option<String>,
    phase: GroupPhase,
    finish: Option<Box<dyn FnOnce(PushOutcome) + Send>>,
}

impl PushGroup {
    fn enter_cancelling(
        &mut self,
        sockets: &Sockets,
        reply_to: &str,
        now: Instant,
        timing: CancelTiming,
    ) {
        if matches!(self.phase, GroupPhase::Cancelling { .. }) {
            return;
        }
        for (&job, (host, port)) in &self.pending {
            let cancel = Message::Cancel {
                job,
                reply_to: reply_to.to_owned(),
            };
            let _ = sockets.send(host, *port, cancel.frames());
        }
        let freeze_signaled = timing.freeze_wait.is_zero();
        if freeze_signaled {
            self.cancelling.store(true, Ordering::Release);
        }
        self.phase = GroupPhase::Cancelling {
            freeze_at: now + timing.freeze_wait,
            next_retry: now + timing.retry_initial,
            retry_interval: timing.retry_initial,
            give_up_at: now + timing.retry_budget,
            freeze_signaled,
        };
    }

    fn complete(mut self) {
        let outcome = if let Some(reason) = self.unknown_reason {
            PushOutcome::Unknown(reason)
        } else if let Some(reason) = self.abort_reason {
            PushOutcome::Aborted(reason)
        } else if self.sealed.load(Ordering::SeqCst) {
            PushOutcome::Aborted("peer push sealed".to_owned())
        } else {
            PushOutcome::Complete
        };
        if let Some(finish) = self.finish.take() {
            finish(outcome);
        }
    }
}

fn drain_new_groups(
    new_groups_rx: &mpsc::Receiver<NewPushGroup>,
    sockets: &Sockets,
    reply_to: &str,
    timing: CancelTiming,
    groups: &mut HashMap<u64, PushGroup>,
    job_to_group: &mut HashMap<u64, u64>,
) {
    for new in new_groups_rx.try_iter() {
        let now = Instant::now();
        let mut pending = HashMap::with_capacity(new.requests.len());
        let mut abort_reason = None;
        for (job, host, port, request) in new.requests {
            if new.sealed.load(Ordering::SeqCst) {
                abort_reason = Some("peer push sealed during submission".to_owned());
                break;
            }
            match sockets.send(&host, port, request.frames()) {
                Ok(()) => {
                    pending.insert(job, (host, port));
                    job_to_group.insert(job, new.group_id);
                }
                Err(error) => {
                    abort_reason = Some(error.to_string());
                    break;
                }
            }
        }
        let mut group = PushGroup {
            pending,
            sealed: new.sealed,
            cancelling: new.cancelling,
            abort_reason,
            unknown_reason: None,
            phase: GroupPhase::Waiting {
                wait_deadline: now + new.budget.saturating_add(Duration::from_secs(1)),
            },
            finish: Some(new.finish),
        };
        if group.pending.is_empty() {
            group.complete();
            continue;
        }
        if group.abort_reason.is_some() {
            group.enter_cancelling(sockets, reply_to, now, timing);
        }
        groups.insert(new.group_id, group);
    }
}

fn step_groups(
    sockets: &Sockets,
    reply_to: &str,
    timing: CancelTiming,
    groups: &mut HashMap<u64, PushGroup>,
    job_to_group: &mut HashMap<u64, u64>,
) {
    let now = Instant::now();
    let mut done_ids = Vec::new();
    for (&group_id, group) in groups.iter_mut() {
        if group.pending.is_empty() {
            done_ids.push(group_id);
            continue;
        }
        match &mut group.phase {
            GroupPhase::Waiting { wait_deadline } => {
                if group.sealed.load(Ordering::SeqCst) {
                    group
                        .abort_reason
                        .get_or_insert_with(|| "peer push sealed".to_owned());
                    group.enter_cancelling(sockets, reply_to, now, timing);
                } else if now >= *wait_deadline {
                    group
                        .abort_reason
                        .get_or_insert_with(|| "the push did not finish in its budget".to_owned());
                    group.enter_cancelling(sockets, reply_to, now, timing);
                }
            }
            GroupPhase::Cancelling {
                freeze_at,
                next_retry,
                retry_interval,
                give_up_at,
                freeze_signaled,
            } => {
                if !*freeze_signaled && now >= *freeze_at {
                    group.cancelling.store(true, Ordering::Release);
                    *freeze_signaled = true;
                }
                if now >= *give_up_at {
                    // Holder never acknowledged cancellation within the retry
                    // budget (e.g. crashed or restarted onto a new ephemeral
                    // port). Resolve as Unknown so the landing stays safely
                    // frozen while reclaiming reader dispatcher resources.
                    group.unknown_reason.get_or_insert_with(|| {
                        "peer push cancellation went unacknowledged; remote write retirement unknown"
                            .to_owned()
                    });
                    done_ids.push(group_id);
                    continue;
                }
                if now >= *next_retry {
                    for (&job, (host, port)) in &group.pending {
                        let cancel = Message::Cancel {
                            job,
                            reply_to: reply_to.to_owned(),
                        };
                        let _ = sockets.send(host, *port, cancel.frames());
                    }
                    *retry_interval = (*retry_interval * 2).min(timing.retry_max);
                    *next_retry = now + *retry_interval;
                }
            }
        }
    }
    for group_id in done_ids {
        if let Some(group) = groups.remove(&group_id) {
            for job in group.pending.keys() {
                job_to_group.remove(job);
            }
            group.complete();
        }
    }
}

/// A rank's push endpoint: it serves the pushes of a holder with a Disk tier
/// and receives the answers its own reads' pushes wait for.
pub struct PushEndpoint {
    name: String,
    sockets: Option<Arc<Sockets>>,
    holder: Arc<Mutex<Option<Arc<Holder>>>>,
    waiters: Arc<Mutex<HashMap<u64, mpsc::Sender<Message>>>>,
    new_groups: Option<mpsc::Sender<NewPushGroup>>,
    wake: Option<mpsc::Sender<Vec<Bytes>>>,
    #[allow(dead_code)]
    cancel_timing: Arc<Mutex<CancelTiming>>,
    next_job: AtomicU64,
    next_group: AtomicU64,
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
        let wake = incoming.clone();
        let (new_groups_tx, new_groups_rx) = mpsc::channel::<NewPushGroup>();
        let sockets = Arc::new(Sockets::start_at(bind_host, port, incoming)?);
        let name = crate::peer::net::join_endpoint(advertised_host, sockets.port());
        let holder: Arc<Mutex<Option<Arc<Holder>>>> = Arc::new(Mutex::new(None));
        let waiters: Arc<Mutex<HashMap<u64, mpsc::Sender<Message>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cancel_timing = Arc::new(Mutex::new(CancelTiming::default()));
        let dispatcher = {
            let (sockets_weak, holder, waiters, cancel_timing, local_name) = (
                Arc::downgrade(&sockets),
                holder.clone(),
                waiters.clone(),
                cancel_timing.clone(),
                name.clone(),
            );
            thread::Builder::new()
                .name("ts-peer-push".to_owned())
                .spawn(move || {
                    let mut groups: HashMap<u64, PushGroup> = HashMap::new();
                    let mut job_to_group: HashMap<u64, u64> = HashMap::new();
                    loop {
                        let recv_res = if groups.is_empty() {
                            received
                                .recv()
                                .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
                        } else {
                            received.recv_timeout(SEAL_POLL_INTERVAL)
                        };
                        let Some(sockets) = sockets_weak.upgrade() else {
                            break;
                        };
                        let timing = *cancel_timing.lock();
                        drain_new_groups(
                            &new_groups_rx,
                            &sockets,
                            &local_name,
                            timing,
                            &mut groups,
                            &mut job_to_group,
                        );
                        match recv_res {
                            Ok(frames) if !frames.is_empty() => match Message::decode(&frames) {
                                Some(Message::Push {
                                    job,
                                    offer,
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
                                            offer,
                                            session,
                                            Duration::from_micros(budget_us),
                                            items,
                                        ),
                                        None => send(
                                            &sockets,
                                            &reply_to,
                                            &Message::Done { job, ok: false },
                                        ),
                                    }
                                }
                                Some(Message::Cancel { job, reply_to }) => {
                                    let serving = holder.lock().clone();
                                    match serving {
                                        Some(h) => h.cancel(&sockets, JobKey { reply_to, job }),
                                        None => {
                                            send(&sockets, &reply_to, &Message::Cancelled { job })
                                        }
                                    }
                                }
                                Some(
                                    answer @ (Message::Done { .. }
                                    | Message::Cancelled { .. }
                                    | Message::Unknown { .. }),
                                ) => {
                                    let job = match answer {
                                        Message::Done { job, .. }
                                        | Message::Cancelled { job }
                                        | Message::Unknown { job } => job,
                                        _ => unreachable!(),
                                    };
                                    if let Some(group_id) = job_to_group.remove(&job) {
                                        if let Some(group) = groups.get_mut(&group_id) {
                                            group.pending.remove(&job);
                                            let in_cancelling = matches!(
                                                group.phase,
                                                GroupPhase::Cancelling { .. }
                                            );
                                            match answer {
                                                Message::Done { ok: true, .. } => {}
                                                Message::Done { ok: false, .. } => {
                                                    if !in_cancelling {
                                                        group.abort_reason.get_or_insert_with(
                                                            || {
                                                                "the holder could not push"
                                                                    .to_owned()
                                                            },
                                                        );
                                                    }
                                                }
                                                Message::Cancelled { .. } => {
                                                    if !in_cancelling {
                                                        group.abort_reason.get_or_insert_with(
                                                            || {
                                                                "the holder cancelled the push"
                                                                    .to_owned()
                                                            },
                                                        );
                                                    }
                                                }
                                                Message::Unknown { .. } => {
                                                    group.unknown_reason.get_or_insert_with(|| {
                                                        "the holder push retired with unknown write state"
                                                            .to_owned()
                                                    });
                                                }
                                                _ => {}
                                            }
                                            if !group.pending.is_empty()
                                                && (group.abort_reason.is_some()
                                                    || group.unknown_reason.is_some())
                                            {
                                                group.enter_cancelling(
                                                    &sockets,
                                                    &local_name,
                                                    Instant::now(),
                                                    timing,
                                                );
                                            }
                                            if group.pending.is_empty() {
                                                if let Some(done_group) = groups.remove(&group_id) {
                                                    done_group.complete();
                                                }
                                            }
                                        }
                                    } else if let Some(waiter) = waiters.lock().get(&job) {
                                        let _ = waiter.send(answer);
                                    }
                                }
                                None => tracing::warn!("a malformed peer push message dropped"),
                            },
                            Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                        if !groups.is_empty() {
                            step_groups(
                                &sockets,
                                &local_name,
                                timing,
                                &mut groups,
                                &mut job_to_group,
                            );
                        }
                    }
                })?
        };
        Ok(Self {
            name,
            sockets: Some(sockets),
            holder,
            waiters,
            new_groups: Some(new_groups_tx),
            wake: Some(wake),
            cancel_timing,
            next_job: AtomicU64::new(incarnation << 32),
            next_group: AtomicU64::new(1),
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

    /// Authorize pushes tagged with `(incarnation, offer_id)` once pinned.
    pub fn activate_offer(&self, incarnation: u64, offer_id: u64) {
        if let Some(holder) = self.holder.lock().as_ref() {
            holder.activate_offer(incarnation, offer_id);
        }
    }

    /// Revoke authorization for `(incarnation, offer_id)` and cancel queued/in-flight
    /// pushes. Returns `true` once zero jobs for that offer remain in flight.
    pub fn cancel_offer(&self, incarnation: u64, offer_id: u64) -> bool {
        self.holder
            .lock()
            .as_ref()
            .is_none_or(|holder| holder.cancel_offer(incarnation, offer_id))
    }

    /// Override cancel freeze/retry timings in unit and integration tests.
    #[cfg(test)]
    pub fn set_cancel_timing_for_test(
        &self,
        freeze_wait: Duration,
        retry_initial: Duration,
        retry_max: Duration,
        retry_budget: Duration,
    ) {
        *self.cancel_timing.lock() = CancelTiming {
            freeze_wait,
            retry_initial,
            retry_max,
            retry_budget,
        };
    }

    /// `(free, total)` staging-ring slot counts on the active holder, if serving.
    #[cfg(test)]
    pub fn push_ring_slot_counts(&self) -> Option<(usize, usize)> {
        let holder = self.holder.lock().clone()?;
        let free = holder.state.slots.free_count();
        Some((free, holder.state.slots.total))
    }

    /// Number of currently registered active push jobs on the holder.
    #[cfg(test)]
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
        self.new_groups.take();
        self.wake.take();
        self.sockets.take();
        if let Some(dispatcher) = self.dispatcher.take() {
            let _ = dispatcher.join();
        }
    }
}

impl PagePusher for PushEndpoint {
    fn activate_offer(&self, incarnation: u64, offer_id: u64) {
        PushEndpoint::activate_offer(self, incarnation, offer_id);
    }

    fn cancel_offer(&self, incarnation: u64, offer_id: u64) -> bool {
        PushEndpoint::cancel_offer(self, incarnation, offer_id)
    }

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
            offer: None,
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
            Ok(Message::Unknown { .. }) => {
                return Err("the holder push retired with unknown write state".to_owned());
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
                Ok(Message::Unknown { .. }) => {
                    return Err("the holder push retired with unknown write state".to_owned());
                }
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

    fn start_push_many(
        &self,
        targets: Vec<PushTarget>,
        budget: Duration,
        submit_until: Instant,
        sealed: Arc<AtomicBool>,
        cancelling: Arc<AtomicBool>,
        finish: Box<dyn FnOnce(PushOutcome) + Send>,
    ) {
        if targets.is_empty() {
            let outcome = if sealed.load(Ordering::SeqCst) {
                PushOutcome::Aborted("peer push sealed".to_owned())
            } else {
                PushOutcome::Complete
            };
            finish(outcome);
            return;
        }
        let (Some(new_groups), Some(wake)) = (self.new_groups.as_ref(), self.wake.as_ref()) else {
            finish(PushOutcome::Aborted("push endpoint is closed".to_owned()));
            return;
        };
        if self.sockets.is_none() {
            finish(PushOutcome::Aborted("push endpoint is closed".to_owned()));
            return;
        }
        if sealed.load(Ordering::SeqCst) {
            finish(PushOutcome::Aborted(
                "peer push sealed before submission".to_owned(),
            ));
            return;
        }
        let Some(remaining) = submit_until.checked_duration_since(Instant::now()) else {
            finish(PushOutcome::Aborted(
                "peer push submission window elapsed".to_owned(),
            ));
            return;
        };
        let effective_budget = budget.min(remaining).min(MAX_PUSH_BUDGET);
        let budget_us = u64::try_from(effective_budget.as_micros()).unwrap_or(u64::MAX);
        if budget_us == 0 {
            finish(PushOutcome::Aborted("peer push budget is zero".to_owned()));
            return;
        }
        let mut requests = Vec::with_capacity(targets.len());
        for target in targets {
            let (host, port) = match split_endpoint(&target.endpoint) {
                Ok((host, port)) => (host.to_owned(), port),
                Err(error) => {
                    finish(PushOutcome::Aborted(error.to_string()));
                    return;
                }
            };
            let job = self.next_job.fetch_add(1, Ordering::Relaxed);
            requests.push((
                job,
                host,
                port,
                Message::Push {
                    job,
                    offer: (target.offer_id != 0)
                        .then_some((target.holder_incarnation, target.offer_id)),
                    reply_to: self.name.clone(),
                    session: target.session,
                    budget_us,
                    items: target.items,
                },
            ));
        }
        let group_id = self.next_group.fetch_add(1, Ordering::Relaxed);
        let group = NewPushGroup {
            group_id,
            requests,
            budget: effective_budget,
            sealed,
            cancelling,
            finish,
        };
        match new_groups.send(group) {
            Ok(()) => {
                let _ = wake.send(Vec::new());
            }
            Err(mpsc::SendError(group)) => {
                (group.finish)(PushOutcome::Aborted("push endpoint is closed".to_owned()));
            }
        }
    }

    fn push_many(
        &self,
        targets: Vec<PushTarget>,
        budget: Duration,
        submit_until: Instant,
        sealed: &AtomicBool,
    ) -> PushOutcome {
        let (tx, rx) = mpsc::channel();
        let sealed_bridge = Arc::new(AtomicBool::new(sealed.load(Ordering::SeqCst)));
        let cancelling = Arc::new(AtomicBool::new(false));
        self.start_push_many(
            targets,
            budget,
            submit_until,
            sealed_bridge.clone(),
            cancelling,
            Box::new(move |outcome| {
                let _ = tx.send(outcome);
            }),
        );
        loop {
            if sealed.load(Ordering::SeqCst) {
                sealed_bridge.store(true, Ordering::SeqCst);
            }
            match rx.recv_timeout(SEAL_POLL_INTERVAL) {
                Ok(outcome) => return outcome,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return PushOutcome::Unknown("push endpoint dispatcher closed".to_owned());
                }
            }
        }
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
        let legacy_push = Message::Push {
            job: (7 << 32) + 1,
            offer: None,
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
        };
        assert_eq!(legacy_push.frames()[0].as_ref(), PUSH);
        let v2_push = Message::Push {
            job: (7 << 32) + 2,
            offer: Some((99, 42)),
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
        };
        assert_eq!(v2_push.frames()[0].as_ref(), PUSH_V2);
        for msg in [
            legacy_push,
            v2_push,
            Message::Cancel {
                job: 3,
                reply_to: "[::1]:7200".into(),
            },
            Message::Done { job: 3, ok: true },
            Message::Done { job: 4, ok: false },
            Message::Cancelled { job: 3 },
            Message::Unknown { job: 5 },
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
        fenced.insert(a.clone(), false);
        assert!(fenced.contains(&a) && !fenced.unknown(&a) && !fenced.contains(&b));
        fenced.insert(b.clone(), true);
        assert!(fenced.contains(&b) && fenced.unknown(&b));
        for job in 0..(MAX_FENCED_JOBS as u64 + 100) {
            fenced.insert(
                JobKey {
                    reply_to: "10.0.0.3:7200".into(),
                    job,
                },
                false,
            );
        }
        assert!(!fenced.contains(&a));
        // Unknown-retirement entries must never be evicted by clean FIFO traffic.
        assert!(fenced.contains(&b) && fenced.unknown(&b));
        assert!(fenced.contains(&JobKey {
            reply_to: "10.0.0.3:7200".into(),
            job: MAX_FENCED_JOBS as u64 + 99
        }));

        let r1 = Running::new(None);
        assert_eq!(r1.cancel("r1".into()), None);
        assert_eq!(r1.finish(false), Some("r1".into()));
        let r2 = Running::new(None);
        assert_eq!(r2.finish(false), None);
        assert_eq!(r2.cancel("r2".into()), Some(("r2".into(), false)));
        let r3 = Running::new(None);
        assert_eq!(r3.finish(true), None);
        assert_eq!(r3.cancel("r3".into()), Some(("r3".into(), true)));
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

        // 1. 3 PushItems naming block 1 -> 1 deduplicated NVMe read + 3 MR write batches.
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

    #[test]
    fn slot_guard_leaks_on_panic_and_returns_normally() {
        let slots = Slots {
            total: 2,
            max_per_job: 1,
            free: Mutex::new(vec![0, 1]),
            returned: Condvar::new(),
        };
        {
            let guard = slots.try_take().unwrap();
            assert_eq!(slots.free_count(), 1);
            let _ = guard.index();
        }
        assert_eq!(slots.free_count(), 2);

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = slots.try_take().unwrap();
            assert_eq!(slots.free_count(), 1);
            panic!("simulated panic with in-flight DMA");
        }));
        assert!(panicked.is_err());
        assert_eq!(slots.free_count(), 1);
    }

    #[test]
    fn worker_survives_writer_panic_leaks_slot_and_serves_next_push() {
        use std::{path::PathBuf, sync::atomic::AtomicUsize};
        use ts_disk::{Backend, DiskFile, ExecutorConfig, Memory};
        use ts_l3::{GroupLayout, HostLayout};

        const BLOCK: usize = 4096;
        // One more slot than MAX_PUSH_WORKERS so panicking MAX_PUSH_WORKERS
        // times proves both that every panicked job leaks its slot and that
        // workers survive rather than exhausting the pool.
        const TOTAL_SLOTS: usize = MAX_PUSH_WORKERS + 1;

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
            "ts-push-panic-{}-{}.bin",
            std::process::id(),
            profile::now_ns()
        )));
        let disk_file = DiskFile::create(&temp.0, (2 * BLOCK) as u64).unwrap();
        let ring = Arc::new(Memory::map(TOTAL_SLOTS * BLOCK).unwrap());
        let mut seed_mem = Memory::map(BLOCK).unwrap();
        seed_mem.write_at(0, &[0xAB; BLOCK]);

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
                        len: BLOCK,
                    }],
                },
                move |r| {
                    let _ = tx.send(r.outcome);
                },
            )
            .unwrap();
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), Ok(()));

        struct PanicNWriter {
            remaining_panics: std::sync::atomic::AtomicIsize,
            ok_writes: AtomicUsize,
        }
        impl PageWriter for PanicNWriter {
            fn write(&self, _: &str, _: &Spans) -> Result<(), String> {
                // Signed, so the countdown may run past zero without wrapping.
                if self.remaining_panics.fetch_sub(1, Ordering::SeqCst) > 0 {
                    panic!("simulated writer panic");
                }
                self.ok_writes.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let writer = Arc::new(PanicNWriter {
            remaining_panics: std::sync::atomic::AtomicIsize::new(MAX_PUSH_WORKERS as isize),
            ok_writes: AtomicUsize::new(0),
        });
        let holder = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 11).unwrap();
        holder
            .serve(PushSource {
                executor,
                disk_layout: HostLayout {
                    groups: vec![GroupLayout {
                        packing: 1,
                        block_bytes: BLOCK,
                        shard_count: 1,
                    }],
                    lcm_block_bytes: BLOCK,
                    num_lcm_blocks: 2,
                    owner_rank: 0,
                    tp_rank: 0,
                },
                writer: writer.clone(),
                ring,
                slot_bytes: BLOCK,
            })
            .unwrap();
        let reader = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 12).unwrap();
        let one = vec![PushItem {
            group: 0,
            block: 1,
            spans: vec![PushSpan {
                offset: 0,
                address: 100_000,
                length: BLOCK as u64,
            }],
        }];

        for i in 0..MAX_PUSH_WORKERS {
            assert!(reader
                .push(
                    holder.name(),
                    &format!("p{i}"),
                    Duration::from_secs(5),
                    one.clone()
                )
                .is_err());
        }
        assert_eq!(holder.push_ring_slot_counts(), Some((1, TOTAL_SLOTS)));
        assert_eq!(
            reader.push(holder.name(), "survivor", Duration::from_secs(5), one),
            Ok(())
        );
        assert_eq!(writer.ok_writes.load(Ordering::SeqCst), 1);
        assert_eq!(holder.push_ring_slot_counts(), Some((1, TOTAL_SLOTS)));
    }

    #[test]
    fn slots_give_back_wakes_live_waiter_when_sibling_cancelled() {
        let slots = Arc::new(Slots {
            total: 1,
            max_per_job: 1,
            free: Mutex::new(vec![0]),
            returned: Condvar::new(),
        });
        let held = slots.try_take().expect("initial single slot");

        // Park multiple cancelled waiters alongside one live waiter so a single
        // `notify_one` could land on a cancelled waiter and leave the live one
        // asleep until its long deadline.
        let cancelled = Arc::new(AtomicBool::new(false));
        let live_cancel = Arc::new(AtomicBool::new(false));
        let deadline = Instant::now() + Duration::from_secs(5);

        let mut cancelled_handles = Vec::new();
        for _ in 0..8 {
            let s = slots.clone();
            let c = cancelled.clone();
            cancelled_handles.push(thread::spawn(move || s.take(deadline, &c).is_some()));
        }
        let s_live = slots.clone();
        let c_live = live_cancel.clone();
        let live_handle = thread::spawn(move || {
            let start = Instant::now();
            let got = s_live.take(deadline, &c_live);
            (got.is_some(), start.elapsed())
        });

        // Give all waiters time to enter `wait_until`.
        thread::sleep(Duration::from_millis(40));
        // Mark the sibling waiters cancelled without calling `wake_all`, then
        // return the single slot via `give_back` (`notify_all`).
        cancelled.store(true, Ordering::Release);
        drop(held);

        let (live_got, elapsed) = live_handle.join().unwrap();
        assert!(
            live_got,
            "live waiter must acquire the returned slot despite cancelled siblings"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "live waiter should wake immediately on give_back, took {elapsed:?}"
        );
        for h in cancelled_handles {
            assert!(!h.join().unwrap());
        }
        assert_eq!(slots.free_count(), 1);
    }

    #[test]
    fn push_many_parallel_complete_clean_abort_and_unknown_retirement() {
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

        struct TestWriter {
            ring: Arc<Memory>,
            fail: AtomicBool,
            panic_next: AtomicBool,
            written: Mutex<Vec<(u64, Vec<u8>)>>,
            hold: (std::sync::Mutex<bool>, std::sync::Condvar),
            entered: AtomicUsize,
        }
        impl PageWriter for TestWriter {
            fn write(&self, _: &str, spans: &Spans) -> Result<(), String> {
                self.entered.fetch_add(1, Ordering::SeqCst);
                let held = self.hold.0.lock().unwrap();
                let _held = self.hold.1.wait_while(held, |b| *b).unwrap();
                if self.panic_next.swap(false, Ordering::SeqCst) {
                    panic!("simulated push_many writer panic");
                }
                if self.fail.load(Ordering::Acquire) {
                    return Err("simulated RDMA write failure".into());
                }
                for ((&src, &dst), &len) in spans
                    .sources
                    .iter()
                    .zip(&spans.destinations)
                    .zip(&spans.lengths)
                {
                    let mut buf = vec![0u8; len as usize];
                    self.ring
                        .read_at(src as usize - self.ring.address(), &mut buf);
                    self.written.lock().push((dst, buf));
                }
                Ok(())
            }
        }

        let dir = std::env::var_os("TS_DISK_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let spawn_holder = |tag: &str, fill: u8, inc: u64| {
            let temp = TempDisk(dir.join(format!(
                "ts-push-many-{tag}-{}-{}.bin",
                std::process::id(),
                profile::now_ns()
            )));
            let disk_file = DiskFile::create(&temp.0, (2 * BLOCK) as u64).unwrap();
            let ring = Arc::new(Memory::map(4 * BLOCK).unwrap());
            let mut seed = Memory::map(BLOCK).unwrap();
            seed.write_at(0, &vec![fill; BLOCK]);
            let executor = Arc::new(
                ts_disk::Executor::start(
                    disk_file,
                    &[(ring.address(), ring.size()), (seed.address(), seed.size())],
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
                            address: seed.address(),
                            len: BLOCK,
                        }],
                    },
                    move |r| {
                        let _ = tx.send(r.outcome);
                    },
                )
                .unwrap();
            assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), Ok(()));
            let writer = Arc::new(TestWriter {
                ring: ring.clone(),
                fail: AtomicBool::new(false),
                panic_next: AtomicBool::new(false),
                written: Mutex::new(Vec::new()),
                hold: (std::sync::Mutex::new(false), std::sync::Condvar::new()),
                entered: AtomicUsize::new(0),
            });
            let ep = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", inc).unwrap();
            ep.serve(PushSource {
                executor,
                disk_layout: HostLayout {
                    groups: vec![GroupLayout {
                        packing: 1,
                        block_bytes: BLOCK,
                        shard_count: 1,
                    }],
                    lcm_block_bytes: BLOCK,
                    num_lcm_blocks: 2,
                    owner_rank: 0,
                    tp_rank: 0,
                },
                writer: writer.clone(),
                ring,
                slot_bytes: BLOCK,
            })
            .unwrap();
            (ep, writer, temp)
        };

        let (h1, w1, _t1) = spawn_holder("h1", 0x11, 101);
        let (h2, w2, _t2) = spawn_holder("h2", 0x22, 102);
        let reader = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 103).unwrap();
        let target = |ep: &PushEndpoint, block: i32, dst: u64| PushTarget {
            endpoint: ep.name().to_owned(),
            session: "reader-sess".into(),
            holder_incarnation: 0,
            offer_id: 0,
            items: vec![PushItem {
                group: 0,
                block,
                spans: vec![PushSpan {
                    offset: 16,
                    address: dst,
                    length: 64,
                }],
            }],
        };

        // 1. Parallel multi-holder fan-out succeeds and delivers both holders' bytes.
        let sealed = AtomicBool::new(false);
        let out = reader.push_many(
            vec![target(&h1, 1, 1000), target(&h2, 1, 2000)],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            &sealed,
        );
        assert_eq!(out, PushOutcome::Complete);
        assert_eq!(w1.written.lock().as_slice(), &[(1000, vec![0x11; 64])]);
        assert_eq!(w2.written.lock().as_slice(), &[(2000, vec![0x22; 64])]);

        // 2. Clean holder refusal (invalid block id) -> Aborted (not Unknown).
        let out = reader.push_many(
            vec![target(&h1, 99, 1000)],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            &sealed,
        );
        assert!(matches!(out, PushOutcome::Aborted(_)), "got {out:?}");

        // 3. Sealing while one holder is paused cancels cleanly -> Aborted.
        *w1.hold.0.lock().unwrap() = true;
        w1.entered.store(0, Ordering::SeqCst);
        let sealed_cancel = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                reader.push_many(
                    vec![target(&h1, 1, 3000)],
                    Duration::from_secs(5),
                    Instant::now() + Duration::from_secs(10),
                    &sealed_cancel,
                )
            });
            let wait = Instant::now();
            while w1.entered.load(Ordering::Acquire) < 1 {
                assert!(wait.elapsed() < Duration::from_secs(5));
                thread::sleep(Duration::from_millis(2));
            }
            sealed_cancel.store(true, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(40));
            *w1.hold.0.lock().unwrap() = false;
            w1.hold.1.notify_all();
            let res = handle.join().unwrap();
            assert!(matches!(res, PushOutcome::Aborted(_)), "got {res:?}");
        });
        assert_eq!(h1.push_ring_slot_counts(), Some((4, 4)));

        // 4. RDMA write error -> Unknown retirement (not Aborted).
        w1.fail.store(true, Ordering::Release);
        let out = reader.push_many(
            vec![target(&h1, 1, 4000), target(&h2, 1, 5000)],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            &sealed,
        );
        assert!(matches!(out, PushOutcome::Unknown(_)), "got {out:?}");
        w1.fail.store(false, Ordering::Release);

        // 5. Writer panic -> Unknown retirement and slot leaked safely.
        w2.panic_next.store(true, Ordering::SeqCst);
        let out = reader.push_many(
            vec![target(&h2, 1, 6000)],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            &sealed,
        );
        assert!(matches!(out, PushOutcome::Unknown(_)), "got {out:?}");
        assert_eq!(h2.push_ring_slot_counts(), Some((3, 4)));

        // 6. Holder active-offer whitelist & drain-aware cancel_offer:
        //    - Unauthorized (never activated) or wrong-incarnation offer pushes
        //      are rejected before touching NVMe.
        //    - Active offer push is cancelled by `cancel_offer`, which reports
        //      `false` (not drained) until the worker exits and `true` once clear.
        let mut offer_target = target(&h1, 1, 7000);
        offer_target.holder_incarnation = 101;
        offer_target.offer_id = 77;
        let entered_before_unauth = w1.entered.load(Ordering::Acquire);
        let unauth_res = reader.push_many(
            vec![offer_target.clone()],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            &sealed,
        );
        assert!(
            matches!(unauth_res, PushOutcome::Aborted(_)),
            "got {unauth_res:?}"
        );
        assert_eq!(w1.entered.load(Ordering::Acquire), entered_before_unauth);

        h1.activate_offer(101, 77);
        let mut wrong_inc_target = offer_target.clone();
        wrong_inc_target.holder_incarnation = 999;
        let wrong_inc_res = reader.push_many(
            vec![wrong_inc_target],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            &sealed,
        );
        assert!(
            matches!(wrong_inc_res, PushOutcome::Aborted(_)),
            "got {wrong_inc_res:?}"
        );
        assert_eq!(w1.entered.load(Ordering::Acquire), entered_before_unauth);

        *w1.hold.0.lock().unwrap() = true;
        w1.entered.store(0, Ordering::SeqCst);
        std::thread::scope(|scope| {
            let t = offer_target.clone();
            let handle = scope.spawn(|| {
                reader.push_many(
                    vec![t],
                    Duration::from_secs(5),
                    Instant::now() + Duration::from_secs(10),
                    &sealed,
                )
            });
            let wait = Instant::now();
            while w1.entered.load(Ordering::Acquire) < 1 {
                assert!(wait.elapsed() < Duration::from_secs(5));
                thread::sleep(Duration::from_millis(2));
            }
            // In-flight worker still holds the block -> not drained yet.
            assert!(!h1.cancel_offer(101, 77));
            *w1.hold.0.lock().unwrap() = false;
            w1.hold.1.notify_all();
            let res = handle.join().unwrap();
            assert!(matches!(res, PushOutcome::Aborted(_)), "got {res:?}");
        });
        assert!(h1.cancel_offer(101, 77));
        let entered_before_late = w1.entered.load(Ordering::Acquire);
        let late_res = reader.push_many(
            vec![offer_target],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            &sealed,
        );
        assert!(
            matches!(late_res, PushOutcome::Aborted(_)),
            "got {late_res:?}"
        );
        assert_eq!(w1.entered.load(Ordering::Acquire), entered_before_late);
    }

    #[test]
    fn background_cancel_retry_signals_freeze_early_and_recovers_on_late_ack() {
        for (late_unknown, expect_unknown) in [(false, false), (true, true)] {
            let (in_tx, in_rx) = mpsc::channel::<Vec<Bytes>>();
            let fake_holder = Sockets::start_at("127.0.0.1", 0, in_tx).unwrap();
            let fake_ep = format!("127.0.0.1:{}", fake_holder.port());
            let reader = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 200).unwrap();
            reader.set_cancel_timing_for_test(
                Duration::from_millis(40),
                Duration::from_millis(25),
                Duration::from_millis(50),
                Duration::from_secs(10),
            );
            let sealed = Arc::new(AtomicBool::new(false));
            let cancelling = Arc::new(AtomicBool::new(false));
            let (done_tx, done_rx) = mpsc::channel();
            reader.start_push_many(
                vec![PushTarget {
                    endpoint: fake_ep,
                    session: "s".into(),
                    holder_incarnation: 99,
                    offer_id: 9,
                    items: vec![PushItem {
                        group: 0,
                        block: 1,
                        spans: vec![PushSpan {
                            offset: 0,
                            address: 1024,
                            length: 64,
                        }],
                    }],
                }],
                Duration::from_secs(5),
                Instant::now() + Duration::from_secs(10),
                sealed.clone(),
                cancelling.clone(),
                Box::new(move |outcome| {
                    let _ = done_tx.send(outcome);
                }),
            );
            // Receive initial Push and do not answer it yet.
            let frames = in_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let Some(Message::Push {
                job,
                offer,
                reply_to,
                ..
            }) = Message::decode(&frames)
            else {
                panic!("expected Push");
            };
            assert_eq!(offer, Some((99, 9)));

            // Seal the reader push -> reader sends Cancel, waits freeze_wait,
            // signals `cancelling`, and keeps retrying Cancel until answered.
            sealed.store(true, Ordering::SeqCst);
            let mut cancels_seen = 0usize;
            let start = Instant::now();
            while cancels_seen < 3 || !cancelling.load(Ordering::Acquire) {
                assert!(
                    start.elapsed() < Duration::from_secs(5),
                    "timed out waiting for cancel retries (seen={cancels_seen}, cancelling={})",
                    cancelling.load(Ordering::Acquire)
                );
                if let Ok(cf) = in_rx.recv_timeout(Duration::from_millis(20)) {
                    if matches!(Message::decode(&cf), Some(Message::Cancel { job: j, .. }) if j == job)
                    {
                        cancels_seen += 1;
                    }
                }
            }
            // Still pending (neither Aborted nor Unknown yet) before the holder replies.
            assert!(done_rx.try_recv().is_err());

            // Now the slow/recovered holder sends its late answer.
            let reply = if late_unknown {
                Message::Unknown { job }
            } else {
                Message::Cancelled { job }
            };
            send(&fake_holder, &reply_to, &reply);
            let outcome = done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            if expect_unknown {
                assert!(
                    matches!(outcome, PushOutcome::Unknown(_)),
                    "got {outcome:?}"
                );
            } else {
                assert!(
                    matches!(outcome, PushOutcome::Aborted(_)),
                    "got {outcome:?}"
                );
            }
        }
    }

    #[test]
    fn background_cancel_retry_gives_up_as_unknown_after_budget() {
        let (in_tx, in_rx) = mpsc::channel::<Vec<Bytes>>();
        let fake_holder = Sockets::start_at("127.0.0.1", 0, in_tx).unwrap();
        let fake_ep = format!("127.0.0.1:{}", fake_holder.port());
        let reader = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 201).unwrap();
        reader.set_cancel_timing_for_test(
            Duration::from_millis(20),
            Duration::from_millis(20),
            Duration::from_millis(40),
            Duration::from_millis(120),
        );
        let sealed = Arc::new(AtomicBool::new(false));
        let cancelling = Arc::new(AtomicBool::new(false));
        let (done_tx, done_rx) = mpsc::channel();
        reader.start_push_many(
            vec![PushTarget {
                endpoint: fake_ep,
                session: "s".into(),
                holder_incarnation: 55,
                offer_id: 11,
                items: vec![PushItem {
                    group: 0,
                    block: 1,
                    spans: vec![PushSpan {
                        offset: 0,
                        address: 1024,
                        length: 64,
                    }],
                }],
            }],
            Duration::from_secs(5),
            Instant::now() + Duration::from_secs(10),
            sealed.clone(),
            cancelling.clone(),
            Box::new(move |outcome| {
                let _ = done_tx.send(outcome);
            }),
        );
        let _push_frames = in_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        sealed.store(true, Ordering::SeqCst);
        let outcome = done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            matches!(outcome, PushOutcome::Unknown(_)),
            "expected Unknown once cancel retry budget expires, got {outcome:?}"
        );
        assert!(cancelling.load(Ordering::Acquire));

        // Sub-microsecond push budget must abort cleanly rather than send budget_us=0.
        let (zero_tx, zero_rx) = mpsc::channel();
        reader.start_push_many(
            vec![PushTarget {
                endpoint: format!("127.0.0.1:{}", fake_holder.port()),
                session: "s".into(),
                holder_incarnation: 55,
                offer_id: 12,
                items: vec![PushItem {
                    group: 0,
                    block: 1,
                    spans: vec![PushSpan {
                        offset: 0,
                        address: 1024,
                        length: 64,
                    }],
                }],
            }],
            Duration::from_nanos(500),
            Instant::now() + Duration::from_secs(10),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Box::new(move |outcome| {
                let _ = zero_tx.send(outcome);
            }),
        );
        let zero_out = zero_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            matches!(zero_out, PushOutcome::Aborted(_)),
            "sub-microsecond budget must abort, got {zero_out:?}"
        );
    }
}
