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
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
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
/// Maximum elapsed holder monotonic time past `activate_offer` during which an
/// offer-bound `PUSH_V2` (`P`) frame is accepted. Covers the reader's 30 s
/// `SUBMIT_WINDOW` plus network transit slack, while rejecting stalled-TCP
/// retransmits arriving after a partition heals even if the offer pin has not
/// yet been retired.
const OFFER_PUSH_ACCEPT_WINDOW: Duration = Duration::from_secs(35);
/// Absolute upper bound past `activate_offer` on new Disk read / RDMA WRITE
/// submissions for any offer-bound `PUSH_V2` job. Strictly below
/// `CANCEL_RETRY_BUDGET` (60 s) by 15 s (plus reader `quarantine_us`), so no
/// holder can start an RDMA write after a silent-holder landing unfreezes.
const OFFER_PUSH_DEADLINE: Duration = Duration::from_secs(45);

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

    /// Whether this endpoint's holder can currently serve Disk-tier pushes.
    /// Returns `false` if no Disk holder is attached, the Disk executor broke,
    /// or the unknown-retirement fence overflowed and quarantined Disk serving.
    fn can_serve_disk(&self) -> bool {
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
    total: usize,
    max_per_job: usize,
    quarantined: AtomicUsize,
    free: Mutex<Vec<usize>>,
    returned: Condvar,
}

impl Slots {
    fn has_available_slots(&self) -> bool {
        self.quarantined.load(Ordering::Acquire) < self.total
    }

    fn take(&self, deadline: Instant, cancelled: &AtomicBool) -> Option<SlotGuard<'_>> {
        let mut free = self.free.lock();
        loop {
            if cancelled.load(Ordering::Acquire) || !self.has_available_slots() {
                return None;
            }
            if let Some(slot) = free.pop() {
                return Some(SlotGuard { slots: self, slot });
            }
            if Instant::now() >= deadline {
                return None;
            }
            if self.returned.wait_until(&mut free, deadline).timed_out() {
                if cancelled.load(Ordering::Acquire) || !self.has_available_slots() {
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

    fn mark_quarantined(&self) {
        let prev = self.quarantined.fetch_add(1, Ordering::AcqRel);
        if prev + 1 >= self.total {
            self.wake_all();
        }
    }

    fn free_count(&self) -> usize {
        self.free.lock().len()
    }

    fn wake_all(&self) {
        let _guard = self.free.lock();
        self.returned.notify_all();
    }
}

/// RAII guard returning a staging slot on normal scope exit; leaks on panic or
/// explicit [`SlotGuard::quarantine`] so an unretired NVMe DMA read or RDMA
/// WRITE cannot access a reused slot.
struct SlotGuard<'a> {
    slots: &'a Slots,
    slot: usize,
}

impl SlotGuard<'_> {
    fn index(&self) -> usize {
        self.slot
    }

    /// Permanently retains this slot without returning it to the free pool.
    fn quarantine(self) {
        self.slots.mark_quarantined();
        std::mem::forget(self);
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.slots.mark_quarantined();
        } else {
            self.slots.give_back(self.slot);
        }
    }
}

/// Maximum completed or cancelled clean push job keys retained to fence
/// duplicate or late-arriving Push messages after a Cancel.
const MAX_FENCED_JOBS: usize = 16_384;
/// Maximum `UnknownRetirement` job keys stored explicitly; if exceeded, the
/// holder quarantines Disk serving and records per-`(reply_to, incarnation)`
/// high-water marks so overflowed unknown retirements still answer `Unknown`
/// without falsely freezing brand-new jobs from healthy readers.
const MAX_UNKNOWN_FENCED_JOBS: usize = 65_536;
/// Maximum distinct `(reply_to, incarnation)` high-water mark entries tracked
/// after `unknown` overflows before falling back to conservative `Unknown`.
const MAX_UNKNOWN_HWM_PEERS: usize = 4_096;
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
/// Clean completions/cancellations use a bounded FIFO (`MAX_FENCED_JOBS`) with
/// sequence-tagged queue entries so upgrading a key to `unknown` never shrinks
/// the live clean window or lets a ghost queue entry evict a re-inserted key.
/// `UnknownRetirement` entries are NEVER FIFO-evicted by normal push traffic;
/// if `unknown` reaches `MAX_UNKNOWN_FENCED_JOBS`, Disk serving is quarantined
/// and overflowed unknown keys are bounded per `(reply_to, incarnation)` high-
/// water mark so replays stay `Unknown` while fresh jobs fail cleanly (`Done`).
#[derive(Default)]
struct FencedJobs {
    clean: HashMap<JobKey, u64>,
    order: VecDeque<(JobKey, u64)>,
    unknown: HashSet<JobKey>,
    /// Keys cancelled after `unknown_overflowed` became true whose retirement
    /// may have been unknown: fenced against late `Push`es without ever
    /// entering `clean` (so repeat `Cancel`s stay `Unknown`).
    overflow_cancelled: HashMap<JobKey, u64>,
    overflow_order: VecDeque<(JobKey, u64)>,
    next_seq: u64,
    /// Per `(reply_to, incarnation = (job >> 32) as u32)` maximum `job` id among
    /// unknown retirements that could not fit in `unknown`.
    unknown_hwm: HashMap<String, HashMap<u32, u64>>,
    unknown_hwm_entries: usize,
    /// Reader endpoints that experienced at least one v2 unknown retirement.
    /// Keeps `hwm_overflowed` from sending unrecognized `u` frames in reply to
    /// stray `CANCEL` frames from legacy `p`-only peers or freezing unseen v2
    /// peers that have no unknown-retirement history here.
    v2_peers: HashSet<String>,
    /// Fails closed if `v2_peers` itself reaches capacity under double overflow.
    v2_peers_overflowed: bool,
    unknown_overflowed: bool,
    hwm_overflowed: bool,
}

impl FencedJobs {
    fn contains(&self, key: &JobKey) -> bool {
        self.unknown.contains(key)
            || self.clean.contains_key(key)
            || self.overflow_cancelled.contains_key(key)
    }

    fn is_unknown_overflowed(&self) -> bool {
        self.unknown_overflowed
    }

    fn note_v2_peer(&mut self, reply_to: &str) {
        if !self.v2_peers.contains(reply_to) {
            if self.v2_peers.len() < MAX_UNKNOWN_FENCED_JOBS {
                self.v2_peers.insert(reply_to.to_owned());
            } else {
                self.v2_peers_overflowed = true;
            }
        }
    }

    /// Whether `key` is known or conservatively suspected to have retired with
    /// unknown RDMA write state.
    fn may_be_unknown(&self, key: &JobKey) -> bool {
        if self.unknown.contains(key) {
            return true;
        }
        if !self.unknown_overflowed || self.clean.contains_key(key) {
            return false;
        }
        let incarnation = (key.job >> 32) as u32;
        if self
            .unknown_hwm
            .get(key.reply_to.as_str())
            .and_then(|incarnations| incarnations.get(&incarnation))
            .is_some_and(|&max_job| key.job <= max_job)
        {
            return true;
        }
        self.hwm_overflowed
            && (self.v2_peers_overflowed || self.v2_peers.contains(key.reply_to.as_str()))
    }

    fn alloc_seq(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        seq
    }

    fn record_unknown_overflow(&mut self, key: JobKey) {
        self.note_v2_peer(&key.reply_to);
        if !self.unknown_overflowed {
            self.unknown_overflowed = true;
            tracing::error!(
                unknown_fenced = self.unknown.len(),
                job = key.job,
                reader = %key.reply_to,
                "peer push unknown-retirement fence capacity reached; quarantining holder Disk pushes"
            );
        }
        let incarnation = (key.job >> 32) as u32;
        if let Some(incarnations) = self.unknown_hwm.get_mut(key.reply_to.as_str()) {
            if let Some(max_job) = incarnations.get_mut(&incarnation) {
                *max_job = (*max_job).max(key.job);
                return;
            }
            if self.unknown_hwm_entries < MAX_UNKNOWN_HWM_PEERS {
                incarnations.insert(incarnation, key.job);
                self.unknown_hwm_entries += 1;
            } else {
                self.hwm_overflowed = true;
            }
            return;
        }
        if self.unknown_hwm_entries < MAX_UNKNOWN_HWM_PEERS {
            let mut incarnations = HashMap::new();
            incarnations.insert(incarnation, key.job);
            self.unknown_hwm.insert(key.reply_to, incarnations);
            self.unknown_hwm_entries += 1;
        } else {
            self.hwm_overflowed = true;
        }
    }

    fn insert_overflow_cancelled(&mut self, key: JobKey) {
        if self.unknown.contains(&key)
            || self.clean.contains_key(&key)
            || self.overflow_cancelled.contains_key(&key)
        {
            return;
        }
        let seq = self.alloc_seq();
        self.overflow_cancelled.insert(key.clone(), seq);
        self.overflow_order.push_back((key, seq));
        while self.overflow_cancelled.len() > MAX_FENCED_JOBS {
            let Some((oldest, oldest_seq)) = self.overflow_order.pop_front() else {
                break;
            };
            if self.overflow_cancelled.get(&oldest) == Some(&oldest_seq) {
                self.overflow_cancelled.remove(&oldest);
            }
        }
        if self.overflow_order.len() > MAX_FENCED_JOBS * 2 {
            self.overflow_order
                .retain(|(k, s)| self.overflow_cancelled.get(k) == Some(s));
        }
    }

    fn insert(&mut self, key: JobKey, unknown: bool) {
        if unknown {
            self.note_v2_peer(&key.reply_to);
            self.clean.remove(&key);
            self.overflow_cancelled.remove(&key);
            if self.unknown.contains(&key) {
                return;
            }
            if self.unknown.len() < MAX_UNKNOWN_FENCED_JOBS {
                self.unknown.insert(key);
            } else {
                self.record_unknown_overflow(key);
            }
            return;
        }
        if self.unknown.contains(&key) || self.clean.contains_key(&key) {
            return;
        }
        self.overflow_cancelled.remove(&key);
        let seq = self.alloc_seq();
        self.clean.insert(key.clone(), seq);
        self.order.push_back((key, seq));
        while self.clean.len() > MAX_FENCED_JOBS {
            let Some((oldest, oldest_seq)) = self.order.pop_front() else {
                break;
            };
            if self.clean.get(&oldest) == Some(&oldest_seq) {
                self.clean.remove(&oldest);
            }
        }
        if self.order.len() > MAX_FENCED_JOBS * 2 {
            self.order.retain(|(k, s)| self.clean.get(k) == Some(s));
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
    deadline: Instant,
    session: String,
    items: Vec<PushItem>,
}

/// Owns a push job's registration in `HolderState.running`; dropping it always
/// fences the job and sends `Done`/`Unknown` (and `Cancelled`/`Unknown` if requested).
struct JobGuard {
    state: Arc<HolderState>,
    sockets: Weak<Sockets>,
    key: JobKey,
    running: Arc<Running>,
    /// True for v2 `PUSH_V2` (`P`) callers that understand `UNKNOWN` (`u`)
    /// replies; false for legacy `PUSH` (`p`) callers that only speak `d`/`x`.
    supports_unknown: bool,
    outcome: RunOutcome,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        let unknown =
            self.supports_unknown && matches!(self.outcome, RunOutcome::UnknownRetirement);
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
                RunOutcome::UnknownRetirement if self.supports_unknown => {
                    Message::Unknown { job: self.key.job }
                }
                RunOutcome::UnknownRetirement => Message::Done {
                    job: self.key.job,
                    ok: false,
                },
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
    /// Active `(incarnation, offer_id)` authorizations mapped to the holder
    /// monotonic `Instant` when the offer was activated on PIN.
    active_offers: Mutex<HashMap<(u64, u64), Instant>>,
    offer_accept_window: Duration,
    offer_push_deadline: Duration,
}

/// Bounded FIFO push queue supporting O(Q) removal of cancelled queued items
/// without head-of-line blocking behind long-running worker jobs.
struct PushQueueInner {
    items: VecDeque<QueuedPush>,
    closed: bool,
}

struct PushQueue {
    inner: Mutex<PushQueueInner>,
    available: Condvar,
}

impl PushQueue {
    fn new() -> Self {
        Self {
            inner: Mutex::new(PushQueueInner {
                items: VecDeque::new(),
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    fn try_push(&self, item: QueuedPush) -> Option<(QueuedPush, bool)> {
        let mut inner = self.inner.lock();
        if inner.closed {
            return Some((item, true));
        }
        if inner.items.len() >= MAX_QUEUED_PUSHES {
            return Some((item, false));
        }
        inner.items.push_back(item);
        self.available.notify_one();
        None
    }

    fn pop(&self) -> Option<QueuedPush> {
        let mut inner = self.inner.lock();
        loop {
            if let Some(item) = inner.items.pop_front() {
                return Some(item);
            }
            if inner.closed {
                return None;
            }
            self.available.wait(&mut inner);
        }
    }

    /// Removes and returns all queued pushes matching `pred` so the caller can
    /// drop their `JobGuard`s outside the queue lock.
    fn drain_matching(&self, mut pred: impl FnMut(&QueuedPush) -> bool) -> Vec<QueuedPush> {
        let mut inner = self.inner.lock();
        let mut removed = Vec::new();
        let mut i = 0;
        while i < inner.items.len() {
            if pred(&inner.items[i]) {
                if let Some(item) = inner.items.remove(i) {
                    removed.push(item);
                }
            } else {
                i += 1;
            }
        }
        removed
    }

    fn close(&self) -> Vec<QueuedPush> {
        let mut inner = self.inner.lock();
        inner.closed = true;
        let remaining: Vec<_> = inner.items.drain(..).collect();
        self.available.notify_all();
        remaining
    }
}

/// The holder's half: a bounded worker pool serving Disk pushes.
struct Holder {
    state: Arc<HolderState>,
    queue: Arc<PushQueue>,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Holder {
    fn new(source: PushSource) -> Result<Self, String> {
        Self::new_with_offer_windows(source, OFFER_PUSH_ACCEPT_WINDOW, OFFER_PUSH_DEADLINE)
    }

    fn new_with_offer_windows(
        source: PushSource,
        offer_accept_window: Duration,
        offer_push_deadline: Duration,
    ) -> Result<Self, String> {
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
                quarantined: AtomicUsize::new(0),
                free: Mutex::new((0..slots).collect()),
                returned: Condvar::new(),
            },
            running: Mutex::new(HashMap::new()),
            fenced: Mutex::new(FencedJobs::default()),
            active_offers: Mutex::new(HashMap::new()),
            offer_accept_window,
            offer_push_deadline,
        });
        let worker_count = slots.clamp(1, MAX_PUSH_WORKERS);
        let queue = Arc::new(PushQueue::new());
        let mut workers = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let (state, queue) = (state.clone(), queue.clone());
            let worker = thread::Builder::new()
                .name(format!("ts-peer-push-job-{index}"))
                .spawn(move || loop {
                    let Some(mut queued) = queue.pop() else {
                        return;
                    };
                    let started = Instant::now();
                    let deadline = queued.deadline;
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
            queue,
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
        let supports_unknown = offer.is_some();
        let active_offer = offer.filter(|&(_, id)| id != 0);
        let running = Arc::new(Running::new(active_offer));
        enum Reject {
            InFlightDuplicate,
            FencedUnknown,
            Quarantined,
            Clean,
        }
        let mut offer_deadline: Option<Instant> = None;
        let reject = {
            let mut running_map = self.state.running.lock();
            if running_map.contains_key(&key) {
                Some(Reject::InFlightDuplicate)
            } else {
                let mut fenced = self.state.fenced.lock();
                if fenced.may_be_unknown(&key) {
                    if supports_unknown {
                        Some(Reject::FencedUnknown)
                    } else {
                        Some(Reject::Clean)
                    }
                } else if fenced.is_unknown_overflowed() || !self.state.slots.has_available_slots()
                {
                    // Holder Disk serving is quarantined (unknown-fence overflow or
                    // all staging slots quarantined by unretired RDMA writes), and
                    // `key` is provably not an earlier unknown job: record it in
                    // `clean` and reject cleanly (`Done`) so healthy readers never
                    // freeze destination memory.
                    fenced.insert(key.clone(), false);
                    Some(Reject::Quarantined)
                } else {
                    let offer_unauthorized = match active_offer {
                        Some(pair) => {
                            let active = self.state.active_offers.lock();
                            match active.get(&pair).copied() {
                                Some(activated_at)
                                    if arrived.saturating_duration_since(activated_at)
                                        < self.state.offer_accept_window =>
                                {
                                    offer_deadline =
                                        Some(activated_at + self.state.offer_push_deadline);
                                    false
                                }
                                _ => true,
                            }
                        }
                        None => false,
                    };
                    if offer_unauthorized || fenced.contains(&key) {
                        Some(Reject::Clean)
                    } else {
                        running_map.insert(key.clone(), running.clone());
                        None
                    }
                }
            }
        };
        if let Some(reason) = reject {
            match reason {
                Reject::InFlightDuplicate => {
                    // Never send `Done { ok: false }` for an active job: the
                    // reader would treat it as cleanly retired and could reuse
                    // destination memory while the worker is still writing.
                    tracing::warn!(
                        job = key.job,
                        ?offer,
                        reader = key.reply_to,
                        "dropping duplicate peer push for in-flight job"
                    );
                }
                Reject::FencedUnknown => {
                    tracing::warn!(
                        job = key.job,
                        ?offer,
                        reader = key.reply_to,
                        "rejecting peer push for job with unknown retirement"
                    );
                    if let Some(sockets) = sockets.upgrade() {
                        send(&sockets, &key.reply_to, &Message::Unknown { job: key.job });
                    }
                }
                Reject::Quarantined => {
                    tracing::warn!(
                        job = key.job,
                        ?offer,
                        reader = key.reply_to,
                        "rejecting peer push cleanly because holder Disk serving is quarantined after unknown-fence overflow"
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
                }
                Reject::Clean => {
                    tracing::warn!(
                        job = key.job,
                        ?offer,
                        reader = key.reply_to,
                        "rejecting fenced or unauthorized-offer peer push"
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
                }
            }
            return;
        }
        let guard = JobGuard {
            state: self.state.clone(),
            sockets,
            key,
            running,
            supports_unknown,
            outcome: RunOutcome::RetiredFailed,
        };
        let mut deadline = arrived + budget.min(MAX_PUSH_BUDGET);
        if let Some(cap) = offer_deadline {
            deadline = deadline.min(cap);
        }
        let queued = QueuedPush {
            guard,
            deadline,
            session,
            items,
        };
        // Drop any rejected `QueuedPush` (and its `JobGuard`) AFTER releasing
        // the queue lock to avoid nested lock acquisition under `queue.inner`.
        if let Some((rejected, closed)) = self.queue.try_push(queued) {
            if closed {
                tracing::warn!("peer push queue closed; rejecting");
            } else {
                tracing::warn!("peer push queue full; rejecting");
            }
            drop(rejected);
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
                fenced_unknown = fenced.may_be_unknown(&key);
                if !fenced.contains(&key) {
                    if fenced_unknown {
                        fenced.insert_overflow_cancelled(key.clone());
                    } else {
                        fenced.insert(key.clone(), false);
                    }
                }
                None
            }
        };
        match running {
            Some(running) => {
                let immediate = running.cancel(key.reply_to.clone());
                // Evict any still-queued copy immediately so it replies
                // `Cancelled` without waiting behind preceding NVMe jobs.
                let evicted = self.queue.drain_matching(|q| q.guard.key == key);
                drop(evicted);
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

    fn can_serve_disk(&self) -> bool {
        !self.state.source.executor.broken()
            && !self.state.fenced.lock().is_unknown_overflowed()
            && self.state.slots.has_available_slots()
    }

    /// Authorizes `(incarnation, offer_id)` to serve Disk pushes once pinned.
    fn activate_offer(&self, incarnation: u64, offer_id: u64) {
        if offer_id == 0 {
            return;
        }
        self.state
            .active_offers
            .lock()
            .entry((incarnation, offer_id))
            .or_insert_with(Instant::now);
    }

    /// Revokes authorization for `(incarnation, offer_id)`, evicts still-queued
    /// pushes immediately, and signals active workers tagged with it to stop.
    /// Returns `true` only once zero jobs for that offer remain in `running`
    /// (so Disk block pins may be released safely).
    fn cancel_offer(&self, incarnation: u64, offer_id: u64) -> bool {
        if offer_id == 0 {
            return true;
        }
        let pair = (incarnation, offer_id);
        {
            let running_map = self.state.running.lock();
            self.state.active_offers.lock().remove(&pair);
            for running in running_map.values() {
                if running.offer == Some(pair) {
                    running.cancelled.store(true, Ordering::Release);
                }
            }
        }
        // Evict queued jobs for this offer outside locks so their `JobGuard`s
        // retire immediately without waiting behind unrelated queued pushes.
        let evicted = self
            .queue
            .drain_matching(|q| q.guard.running.offer == Some(pair));
        drop(evicted);
        self.state.slots.wake_all();
        !self
            .state
            .running
            .lock()
            .values()
            .any(|running| running.offer == Some(pair))
    }

    /// Cancels all active jobs, closes the queue, and joins every worker.
    fn shutdown(&self) {
        let remaining = self.queue.close();
        {
            let running_map = self.state.running.lock();
            for running in running_map.values() {
                running.cancelled.store(true, Ordering::Release);
            }
        }
        drop(remaining);
        self.state.slots.wake_all();
        for worker in self.workers.lock().drain(..) {
            let _ = worker.join();
        }
    }
}

impl HolderState {
    /// Gather-packs consecutive spans whose remote destination addresses are
    /// contiguous (`dst[i+1] == dst[i] + len[i]`, e.g. `D(TP1, Disk) -> P(TP8)`
    /// head-sharded 256B DFLASH SWA slices) into already-registered staging
    /// memory, collapsing `1,536 x 256B` RDMA WRITE WQEs into `12 x 32KB`
    /// without changing the wire protocol.
    ///
    /// Safety across reader memory regions: `PeerReadPlan::plan` partitions
    /// `PushItem`s by destination registered memory region, so every
    /// `batch_spans` lies strictly inside a single reader MR and coalescing
    /// contiguous destinations can never span adjacent reader registrations.
    fn build_write_spans<'a>(
        &'a self,
        ring: usize,
        base: u64,
        range_size: usize,
        allow_in_place: bool,
        batch_spans: &[PushSpan],
    ) -> (Spans, Option<SlotGuard<'a>>) {
        static COALESCE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *COALESCE_ENABLED.get_or_init(|| {
            std::env::var("TOKENSPEED_PEER_PUSH_COALESCE")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true)
        });
        let raw = || {
            (
                Spans {
                    sources: batch_spans.iter().map(|span| base + span.offset).collect(),
                    destinations: batch_spans.iter().map(|span| span.address).collect(),
                    lengths: batch_spans.iter().map(|span| span.length).collect(),
                    host_bytes: 0,
                },
                None,
            )
        };
        if !enabled || batch_spans.len() < 2 {
            return raw();
        }
        // Identify maximal runs of consecutive spans targeting contiguous remote
        // addresses.
        let mut runs: Vec<(usize, usize, u64)> = Vec::new();
        let mut pack_bytes: usize = 0;
        let mut has_merged = false;
        let mut i = 0;
        while i < batch_spans.len() {
            let start = i;
            let mut total = batch_spans[i].length;
            let mut end_addr = match batch_spans[i].address.checked_add(batch_spans[i].length) {
                Some(a) => a,
                None => return raw(),
            };
            i += 1;
            while i < batch_spans.len() && batch_spans[i].address == end_addr {
                total = match total.checked_add(batch_spans[i].length) {
                    Some(t) => t,
                    None => return raw(),
                };
                end_addr = match end_addr.checked_add(batch_spans[i].length) {
                    Some(a) => a,
                    None => return raw(),
                };
                i += 1;
            }
            if i - start > 1 {
                has_merged = true;
                pack_bytes = match usize::try_from(total)
                    .ok()
                    .and_then(|n| pack_bytes.checked_add(n))
                {
                    Some(n) => n,
                    None => return raw(),
                };
            }
            runs.push((start, i, total));
        }
        if !has_merged || pack_bytes == 0 {
            return raw();
        }
        let slot_bytes = self.source.slot_bytes;
        let src_slot_off = match usize::try_from(base).ok().and_then(|b| b.checked_sub(ring)) {
            Some(off) => off,
            None => return raw(),
        };
        // Fast path 1 (zero extra slot contention, zero locks): pack into unused
        // tail of the same registered staging slot when `range_size + pack_bytes <= slot_bytes`.
        // Fast path 2 (zero extra slot contention, works even when `range_size == slot_bytes`):
        // in-place monotonic compaction to offset 0 of the same slot when this is the sole batch
        // for the block and all spans are strictly ascending and non-overlapping in source offset.
        // Fast path 3: borrow a free staging slot if available.
        let can_in_place = allow_in_place
            && batch_spans.windows(2).all(|w| {
                w[0].offset
                    .checked_add(w[0].length)
                    .is_some_and(|e| e <= w[1].offset)
            });
        let (pack_ring_off, in_place_all, pack_guard) = if range_size
            .checked_add(pack_bytes)
            .is_some_and(|end| end <= slot_bytes)
        {
            (src_slot_off + range_size, false, None)
        } else if can_in_place {
            (src_slot_off, true, None)
        } else if pack_bytes <= slot_bytes {
            match self.slots.try_take() {
                Some(guard) => (guard.index() * slot_bytes, false, Some(guard)),
                None => return raw(),
            }
        } else {
            return raw();
        };

        let mut sources = Vec::with_capacity(runs.len());
        let mut destinations = Vec::with_capacity(runs.len());
        let mut lengths = Vec::with_capacity(runs.len());
        let mut cursor: usize = 0;
        for (start, end, total) in runs {
            if !in_place_all && end - start == 1 {
                let span = batch_spans[start];
                sources.push(base + span.offset);
                destinations.push(span.address);
                lengths.push(span.length);
            } else {
                let run_src = (ring + pack_ring_off + cursor) as u64;
                for span in &batch_spans[start..end] {
                    let src_off = src_slot_off + span.offset as usize;
                    let dst_off = pack_ring_off + cursor;
                    let len = span.length as usize;
                    if in_place_all {
                        self.source.ring.move_within(src_off, dst_off, len);
                    } else {
                        self.source.ring.copy_within(src_off, dst_off, len);
                    }
                    cursor += len;
                }
                sources.push(run_src);
                destinations.push(batch_spans[start].address);
                lengths.push(total);
            }
        }
        (
            Spans {
                sources,
                destinations,
                lengths,
                host_bytes: 0,
            },
            pack_guard,
        )
    }

    /// Runs one push to its end: every block read into a slot and written to
    /// the reader, unless `deadline` passes (arrival-anchored `arrived + budget`
    /// for legacy pushes, and clamped to `activated_at + offer_push_deadline`
    /// for offer-bound `PUSH_V2` pushes), the reader cancels, or an I/O fails.
    /// Reports whether every block landed, or whether an RDMA write error left
    /// remote write retirement unknown.
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
        struct InflightRead<'a> {
            slot: Option<SlotGuard<'a>>,
            executor: &'a ts_disk::Executor,
            job_id: JobId,
            completed: bool,
        }
        impl Drop for InflightRead<'_> {
            fn drop(&mut self) {
                if !self.completed && std::thread::panicking() {
                    self.executor.cancel(self.job_id);
                }
            }
        }
        let mut inflight: HashMap<usize, InflightRead<'_>> = HashMap::new();
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
                        inflight.insert(
                            slot_idx,
                            InflightRead {
                                slot: Some(slot_guard),
                                executor: &self.source.executor,
                                job_id,
                                completed: false,
                            },
                        );
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
                for entry in inflight.values() {
                    self.source.executor.cancel(entry.job_id);
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
                        // quarantine their SlotGuards so no future push reuses them.
                        for (_, mut entry) in inflight.drain() {
                            entry.completed = true;
                            if let Some(slot_guard) = entry.slot.take() {
                                slot_guard.quarantine();
                            }
                        }
                        ok = false;
                        break;
                    }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };
            let Some(mut entry) = inflight.remove(&slot_idx) else {
                continue;
            };
            entry.completed = true;
            let mut slot_guard = entry.slot.take();
            ok &= read;
            if !stopped(ok) {
                let base = (ring + slot_idx * self.source.slot_bytes) as u64;
                let range_size = blocks[index].range.size;
                let mut block_ok = true;
                let allow_in_place = blocks[index].batches.len() == 1;
                for batch_spans in &blocks[index].batches {
                    if stopped(ok) {
                        block_ok = false;
                        ok = false;
                        break;
                    }
                    let (spans, pack_slot) =
                        self.build_write_spans(ring, base, range_size, allow_in_place, batch_spans);
                    let write_ns = profile::now_ns();
                    if let Err(error) = self.source.writer.write(session, &spans) {
                        tracing::warn!(%error, slot = slot_idx, "peer push write failed");
                        // An RDMA WRITE error does not guarantee the transport
                        // stopped reading local source buffers (e.g. Mooncake
                        // deferred batch cleanup). Quarantine every staging slot
                        // referenced by `spans` so no subsequent push overwrites
                        // memory while a WRITE may remain in flight.
                        let slot_start = base;
                        let slot_end = base + self.source.slot_bytes as u64;
                        if spans
                            .sources
                            .iter()
                            .any(|&src| src >= slot_start && src < slot_end)
                        {
                            if let Some(guard) = slot_guard.take() {
                                guard.quarantine();
                            }
                        }
                        if let Some(pack_guard) = pack_slot {
                            let pack_start =
                                (ring + pack_guard.index() * self.source.slot_bytes) as u64;
                            let pack_end = pack_start + self.source.slot_bytes as u64;
                            if spans
                                .sources
                                .iter()
                                .any(|&src| src >= pack_start && src < pack_end)
                            {
                                pack_guard.quarantine();
                            }
                        }
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
    submit_until: Instant,
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
        let mut pending = HashMap::with_capacity(new.requests.len());
        let mut abort_reason = None;
        let mut wait_deadline = Instant::now() + new.budget.saturating_add(Duration::from_secs(1));
        for (job, host, port, mut request) in new.requests {
            if new.sealed.load(Ordering::SeqCst) {
                abort_reason = Some("peer push sealed during submission".to_owned());
                break;
            }
            // Re-check the local submission window and re-clamp budget_us at
            // actual wire-send time so dispatcher queueing delay can never push
            // a request past `submit_until` or inflate the holder's deadline.
            let now = Instant::now();
            let Some(remaining) = new.submit_until.checked_duration_since(now) else {
                abort_reason = Some("peer push submission window elapsed".to_owned());
                break;
            };
            let clamped = new.budget.min(remaining).min(MAX_PUSH_BUDGET);
            let budget_us = u64::try_from(clamped.as_micros()).unwrap_or(u64::MAX);
            if budget_us == 0 {
                abort_reason = Some("peer push budget is zero".to_owned());
                break;
            }
            if let Message::Push {
                budget_us: wire_budget,
                ..
            } = &mut request
            {
                *wire_budget = budget_us;
            }
            let target_deadline = now + clamped.saturating_add(Duration::from_secs(1));
            wait_deadline = wait_deadline.min(target_deadline);
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
        let now = Instant::now();
        let mut group = PushGroup {
            pending,
            sealed: new.sealed,
            cancelling: new.cancelling,
            abort_reason,
            unknown_reason: None,
            phase: GroupPhase::Waiting { wait_deadline },
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

    #[cfg(test)]
    fn serve_with_offer_windows(
        &self,
        source: PushSource,
        accept_window: Duration,
        push_deadline: Duration,
    ) -> Result<(), PushError> {
        let next = Arc::new(
            Holder::new_with_offer_windows(source, accept_window, push_deadline)
                .map_err(PushError::Settings)?,
        );
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

    /// Whether the active holder can currently serve Disk-tier pushes.
    pub fn can_serve_disk(&self) -> bool {
        self.holder
            .lock()
            .as_ref()
            .is_some_and(|holder| holder.can_serve_disk())
    }

    /// Simulate an unknown-retirement fence overflow for `key` in tests.
    #[cfg(test)]
    fn force_unknown_overflow_for_test(&self, reply_to: &str, job: u64) {
        if let Some(holder) = self.holder.lock().as_ref() {
            holder.state.fenced.lock().record_unknown_overflow(JobKey {
                reply_to: reply_to.to_owned(),
                job,
            });
        }
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

    fn can_serve_disk(&self) -> bool {
        PushEndpoint::can_serve_disk(self)
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
                    // Multi-holder callers always speak `PUSH_V2` (`P`), which
                    // also signals support for `UNKNOWN` (`u`) replies; an
                    // `offer_id == 0` simply omits offer-whitelist enforcement.
                    offer: Some((target.holder_incarnation, target.offer_id)),
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
            submit_until,
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
        assert!(fenced.contains(&a) && !fenced.may_be_unknown(&a) && !fenced.contains(&b));
        fenced.insert(b.clone(), true);
        assert!(fenced.contains(&b) && fenced.may_be_unknown(&b));
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
        assert!(fenced.contains(&b) && fenced.may_be_unknown(&b));
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
        let ring = Arc::new(Memory::map(2 * BLOCK).unwrap());
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

        // 2. Write fault to one unreachable reader quarantines the faulted slot
        //    so unretired writes cannot read overwritten memory, while healthy
        //    readers continue on the remaining staging slot.
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
        assert_eq!(holder.push_ring_slot_counts(), Some((1, 2)));
        writer.fail.store(false, Ordering::Release);
        assert_eq!(
            reader.push(holder.name(), "s3", Duration::from_secs(2), one(2)),
            Ok(())
        );

        // 3. Overflowing 2 workers + MAX_QUEUED_PUSHES rejects cleanly without blocking.
        *writer.hold.0.lock().unwrap() = true;
        writer.entered.store(0, Ordering::SeqCst);
        let (qtx, qrx) = mpsc::channel::<Vec<Bytes>>();
        let raw_client = Sockets::start_at("127.0.0.1", 0, qtx).unwrap();
        let reply_ep = format!("127.0.0.1:{}", raw_client.port());
        let (h_host, h_port) = split_endpoint(holder.name()).unwrap();
        let total = 2 + MAX_QUEUED_PUSHES;
        for i in 0..total {
            let msg = Message::Push {
                job: (900u64 << 32) | (i as u64 + 1),
                offer: None,
                reply_to: reply_ep.clone(),
                session: format!("q{i}"),
                budget_us: 30_000_000,
                items: one(1),
            };
            raw_client.send(h_host, h_port, msg.frames()).unwrap();
        }
        let wait = Instant::now();
        while writer.entered.load(Ordering::Acquire) < 1 || holder.active_push_jobs() < total {
            assert!(wait.elapsed() < Duration::from_secs(15));
            thread::sleep(Duration::from_millis(2));
        }
        assert!(reader
            .push(holder.name(), "ovf", Duration::from_secs(5), one(1))
            .is_err());
        *writer.hold.0.lock().unwrap() = false;
        writer.hold.1.notify_all();
        for _ in 0..total {
            let reply = Message::decode(&qrx.recv_timeout(Duration::from_secs(15)).unwrap());
            assert!(
                matches!(reply, Some(Message::Done { ok: true, .. })),
                "expected Done{{ok:true}}, got {reply:?}"
            );
        }
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
            quarantined: AtomicUsize::new(0),
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
            quarantined: AtomicUsize::new(0),
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
            quarantined: AtomicUsize::new(0),
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

    #[test]
    fn duplicate_inflight_push_is_silent_unknown_stays_unknown_and_queued_cancel_evicts_immediately(
    ) {
        use std::{path::PathBuf, sync::atomic::AtomicUsize};
        use ts_disk::{Backend, DiskFile, ExecutorConfig, Memory};
        use ts_l3::{GroupLayout, HostLayout};

        // 1. P1-2: After `unknown_overflowed`, cancelling an unrecognized job
        //    within the overflowed peer/incarnation HWM must NEVER insert it
        //    into `clean` (repeat cancels must stay Unknown), while fresh jobs
        //    beyond the HWM or from other readers are cleanly classified.
        let mut fenced = FencedJobs::default();
        let overflowed = JobKey {
            reply_to: "10.0.0.9:7200".into(),
            job: (7u64 << 32) | 999,
        };
        fenced.record_unknown_overflow(overflowed.clone());
        assert!(fenced.is_unknown_overflowed());
        assert!(fenced.may_be_unknown(&overflowed));
        fenced.insert_overflow_cancelled(overflowed.clone());
        assert!(fenced.contains(&overflowed));
        assert!(
            fenced.may_be_unknown(&overflowed),
            "overflow_cancelled key must remain unknown on repeat cancel"
        );
        // Fresh job from the same reader beyond the overflow HWM is NOT unknown.
        let fresh_same_reader = JobKey {
            reply_to: "10.0.0.9:7200".into(),
            job: (7u64 << 32) | 1000,
        };
        assert!(!fenced.may_be_unknown(&fresh_same_reader));
        // Fresh job from a different reader or different incarnation is NOT unknown.
        let fresh_other_inc = JobKey {
            reply_to: "10.0.0.9:7200".into(),
            job: (8u64 << 32) | 1,
        };
        let fresh_other_reader = JobKey {
            reply_to: "10.0.0.10:7200".into(),
            job: (7u64 << 32) | 1,
        };
        assert!(!fenced.may_be_unknown(&fresh_other_inc));
        assert!(!fenced.may_be_unknown(&fresh_other_reader));
        // Clean completions after overflow remain cleanly classified.
        fenced.insert(fresh_same_reader.clone(), false);
        assert!(fenced.contains(&fresh_same_reader) && !fenced.may_be_unknown(&fresh_same_reader));

        // 2. P1-1 & P2-1: Single-slot (1-worker) holder proves:
        //    - duplicate in-flight Push receives NO reply until the active job finishes;
        //    - queued Push behind a blocked worker is evicted immediately by Cancel
        //      and by `cancel_offer` without waiting for the blocked worker;
        //    - duplicate Push for a job that retired Unknown receives `Unknown`.
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
            "ts-push-audit-{}-{}.bin",
            std::process::id(),
            profile::now_ns()
        )));
        let disk_file = DiskFile::create(&temp.0, (2 * BLOCK) as u64).unwrap();
        let ring = Arc::new(Memory::map(BLOCK).unwrap()); // 1 slot -> 1 worker
        let mut seed = Memory::map(BLOCK).unwrap();
        seed.write_at(0, &[0x5A; BLOCK]);
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

        struct HoldWriter {
            hold: (std::sync::Mutex<bool>, std::sync::Condvar),
            fail: AtomicBool,
            entered: AtomicUsize,
        }
        impl PageWriter for HoldWriter {
            fn write(&self, _: &str, _: &Spans) -> Result<(), String> {
                self.entered.fetch_add(1, Ordering::SeqCst);
                let held = self.hold.0.lock().unwrap();
                let _held = self.hold.1.wait_while(held, |b| *b).unwrap();
                if self.fail.load(Ordering::Acquire) {
                    return Err("forced write fault".into());
                }
                Ok(())
            }
        }
        let writer = Arc::new(HoldWriter {
            hold: (std::sync::Mutex::new(true), std::sync::Condvar::new()),
            fail: AtomicBool::new(false),
            entered: AtomicUsize::new(0),
        });
        let holder = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 301).unwrap();
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
        holder.activate_offer(500, 1);
        holder.activate_offer(500, 2);

        let (rtx, rrx) = mpsc::channel::<Vec<Bytes>>();
        let raw_reader = Sockets::start_at("127.0.0.1", 0, rtx).unwrap();
        let reply_ep = format!("127.0.0.1:{}", raw_reader.port());
        let (h_host, h_port) = split_endpoint(holder.name()).unwrap();
        let items = vec![PushItem {
            group: 0,
            block: 1,
            spans: vec![PushSpan {
                offset: 0,
                address: 2048,
                length: 64,
            }],
        }];

        // Send job 1 (occupies the single worker and blocks inside HoldWriter).
        let p1 = Message::Push {
            job: 1,
            offer: Some((500, 1)),
            reply_to: reply_ep.clone(),
            session: "s".into(),
            budget_us: 10_000_000,
            items: items.clone(),
        };
        raw_reader.send(h_host, h_port, p1.frames()).unwrap();
        let wait = Instant::now();
        while writer.entered.load(Ordering::Acquire) < 1 {
            assert!(wait.elapsed() < Duration::from_secs(5));
            thread::sleep(Duration::from_millis(2));
        }

        // Send duplicate in-flight job 1 -> must NOT produce Done{ok:false}!
        raw_reader.send(h_host, h_port, p1.frames()).unwrap();
        assert!(
            rrx.recv_timeout(Duration::from_millis(120)).is_err(),
            "duplicate in-flight Push must not send any premature reply"
        );

        // Queue job 2 (offer 2) behind blocked job 1, then cancel_offer(500, 2):
        // must evict job 2 IMMEDIATELY and report drained=true while job 1 is still blocked!
        let p2 = Message::Push {
            job: 2,
            offer: Some((500, 2)),
            reply_to: reply_ep.clone(),
            session: "s".into(),
            budget_us: 10_000_000,
            items: items.clone(),
        };
        raw_reader.send(h_host, h_port, p2.frames()).unwrap();
        let wait_q = Instant::now();
        while holder.active_push_jobs() < 2 {
            assert!(wait_q.elapsed() < Duration::from_secs(5));
            thread::sleep(Duration::from_millis(2));
        }
        assert!(
            holder.cancel_offer(500, 2),
            "queued-only offer must drain immediately without waiting behind blocked worker"
        );
        let msg2 = Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(msg2, Some(Message::Done { job: 2, ok: false }));

        // Now fail job 1 with RDMA write error -> Unknown retirement.
        writer.fail.store(true, Ordering::Release);
        *writer.hold.0.lock().unwrap() = false;
        writer.hold.1.notify_all();
        let msg1 = Message::decode(&rrx.recv_timeout(Duration::from_secs(5)).unwrap());
        assert_eq!(msg1, Some(Message::Unknown { job: 1 }));

        // Replaying job 1 after Unknown retirement must answer Unknown (never Done{ok:false}),
        // even after quarantining job 1's sole staging slot degraded `can_serve_disk()`.
        assert!(!holder.can_serve_disk());
        raw_reader.send(h_host, h_port, p1.frames()).unwrap();
        let replay = Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(replay, Some(Message::Unknown { job: 1 }));
        drop(holder);

        // Fresh 2-slot holder so `legacy_push`'s write fault quarantines 1 slot
        // while keeping `can_serve_disk() == true` until step 3 forces unknown overflow.
        let ring2 = Arc::new(Memory::map(2 * BLOCK).unwrap());
        let temp2 = TempDisk(dir.join(format!(
            "ts-push-audit2-{}-{}.bin",
            std::process::id(),
            profile::now_ns()
        )));
        let disk_file2 = DiskFile::create(&temp2.0, (2 * BLOCK) as u64).unwrap();
        let executor2 = Arc::new(
            ts_disk::Executor::start(
                disk_file2,
                &[
                    (ring2.address(), ring2.size()),
                    (seed.address(), seed.size()),
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
        let holder = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 302).unwrap();
        holder
            .serve(PushSource {
                executor: executor2,
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
                ring: ring2,
                slot_bytes: BLOCK,
            })
            .unwrap();
        let (h_host, h_port) = split_endpoint(holder.name()).unwrap();
        let _keep_temp2 = temp2;

        // Legacy frame `p` (`offer: None`) hitting an RDMA write fault must reply
        // `Done { ok: false }` (and `Cancelled` on Cancel), NEVER unrecognized frame `u`.
        let legacy_job = (77u64 << 32) | 5;
        let legacy_push = Message::Push {
            job: legacy_job,
            offer: None,
            reply_to: reply_ep.clone(),
            session: "s".into(),
            budget_us: 10_000_000,
            items: items.clone(),
        };
        assert_eq!(legacy_push.frames()[0].as_ref(), PUSH);
        raw_reader
            .send(h_host, h_port, legacy_push.frames())
            .unwrap();
        assert_eq!(
            Message::decode(&rrx.recv_timeout(Duration::from_secs(5)).unwrap()),
            Some(Message::Done {
                job: legacy_job,
                ok: false,
            })
        );
        let legacy_cancel = Message::Cancel {
            job: legacy_job,
            reply_to: reply_ep.clone(),
        };
        raw_reader
            .send(h_host, h_port, legacy_cancel.frames())
            .unwrap();
        assert_eq!(
            Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap()),
            Some(Message::Cancelled { job: legacy_job })
        );
        // Verify production cancel retry budget bounds silent-holder recovery (~60s + 5s quarantine < 70s).
        assert_eq!(CANCEL_RETRY_BUDGET, Duration::from_secs(60));

        // 3. Unknown-fence overflow quarantines Disk serving (`can_serve_disk() == false`),
        //    still answers `Unknown` for replays within the overflowed peer/incarnation HWM,
        //    but cleanly rejects fresh jobs (`Done { ok: false }` + `Cancelled`) without
        //    freezing healthy readers' destination blocks.
        assert!(holder.can_serve_disk());
        let ovf_job = (42u64 << 32) | 10;
        holder.force_unknown_overflow_for_test(&reply_ep, ovf_job);
        assert!(!holder.can_serve_disk());
        holder.activate_offer(500, 3);

        // Replay of overflowed job (job <= HWM) -> Unknown on Push and Cancel.
        let replay_ovf = Message::Push {
            job: ovf_job,
            offer: Some((500, 3)),
            reply_to: reply_ep.clone(),
            session: "s".into(),
            budget_us: 10_000_000,
            items: items.clone(),
        };
        raw_reader
            .send(h_host, h_port, replay_ovf.frames())
            .unwrap();
        assert_eq!(
            Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap()),
            Some(Message::Unknown { job: ovf_job })
        );
        let cancel_ovf = Message::Cancel {
            job: ovf_job,
            reply_to: reply_ep.clone(),
        };
        raw_reader
            .send(h_host, h_port, cancel_ovf.frames())
            .unwrap();
        assert_eq!(
            Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap()),
            Some(Message::Unknown { job: ovf_job })
        );

        // Fresh job from the same reader beyond HWM -> clean Done{ok:false} + Cancelled.
        let fresh_job = (42u64 << 32) | 11;
        let fresh_push = Message::Push {
            job: fresh_job,
            offer: Some((500, 3)),
            reply_to: reply_ep.clone(),
            session: "s".into(),
            budget_us: 10_000_000,
            items: items.clone(),
        };
        raw_reader
            .send(h_host, h_port, fresh_push.frames())
            .unwrap();
        assert_eq!(
            Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap()),
            Some(Message::Done {
                job: fresh_job,
                ok: false
            })
        );
        let cancel_fresh = Message::Cancel {
            job: fresh_job,
            reply_to: reply_ep.clone(),
        };
        raw_reader
            .send(h_host, h_port, cancel_fresh.frames())
            .unwrap();
        assert_eq!(
            Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap()),
            Some(Message::Cancelled { job: fresh_job })
        );

        // Fresh job from another reader (different reply_to) -> clean Done{ok:false}.
        let (other_tx, other_rx) = mpsc::channel::<Vec<Bytes>>();
        let other_reader = Sockets::start_at("127.0.0.1", 0, other_tx).unwrap();
        let other_ep = format!("127.0.0.1:{}", other_reader.port());
        let other_push = Message::Push {
            job: (42u64 << 32) | 1,
            offer: Some((500, 3)),
            reply_to: other_ep,
            session: "s".into(),
            budget_us: 10_000_000,
            items,
        };
        other_reader
            .send(h_host, h_port, other_push.frames())
            .unwrap();
        assert_eq!(
            Message::decode(&other_rx.recv_timeout(Duration::from_secs(2)).unwrap()),
            Some(Message::Done {
                job: (42u64 << 32) | 1,
                ok: false
            })
        );
    }

    #[test]
    fn upgrading_clean_fence_to_unknown_preserves_full_clean_window() {
        let mut fenced = FencedJobs::default();
        let key = |job: u64| JobKey {
            reply_to: "10.0.0.1:7200".into(),
            job,
        };
        // Fill `clean` halfway, upgrade all of those keys to `unknown`, then
        // insert `MAX_FENCED_JOBS` live clean keys: none of the live clean keys
        // may be prematurely evicted by ghosts left in `order`.
        for j in 0..(MAX_FENCED_JOBS as u64) {
            fenced.insert(key(j), false);
            fenced.insert(key(j), true);
        }
        assert_eq!(fenced.clean.len(), 0);
        let base = MAX_FENCED_JOBS as u64;
        for j in base..(base + MAX_FENCED_JOBS as u64) {
            fenced.insert(key(j), false);
        }
        assert_eq!(fenced.clean.len(), MAX_FENCED_JOBS);
        assert!(fenced.contains(&key(base)));
        assert!(!fenced.may_be_unknown(&key(base)));
        assert!(fenced.contains(&key(base + MAX_FENCED_JOBS as u64 - 1)));
        assert!(fenced.order.len() <= MAX_FENCED_JOBS * 2);
    }

    #[test]
    fn legacy_p_peer_cancel_stays_cancelled_even_after_hwm_overflow() {
        let mut fenced = FencedJobs::default();
        // Fill unknown set and overflow both `unknown` and `unknown_hwm` with v2 peers.
        for j in 0..(MAX_UNKNOWN_FENCED_JOBS as u64) {
            fenced.insert(
                JobKey {
                    reply_to: "10.0.0.1:7200".into(),
                    job: j,
                },
                true,
            );
        }
        for inc in 1..=(MAX_UNKNOWN_HWM_PEERS as u32 + 1) {
            fenced.insert(
                JobKey {
                    reply_to: "10.0.0.1:7200".into(),
                    job: (u64::from(inc) << 32) | 5,
                },
                true,
            );
        }
        assert!(fenced.is_unknown_overflowed());
        assert!(fenced.hwm_overflowed);

        // Known v2 peer whose new incarnation overflowed `unknown_hwm` stays conservatively Unknown.
        assert!(fenced.may_be_unknown(&JobKey {
            reply_to: "10.0.0.1:7200".into(),
            job: (u64::from(MAX_UNKNOWN_HWM_PEERS as u32 + 10) << 32) | 1,
        }));
        // Legacy `p`-only peer (or an unseen v2 peer with no unknown-retirement history)
        // is NOT in `v2_peers`, so it neither receives an unrecognized `u` (`Unknown`)
        // frame nor freezes freshly allocated destination blocks.
        assert!(!fenced.may_be_unknown(&JobKey {
            reply_to: "10.0.0.99:7200".into(),
            job: (9u64 << 32) | 1,
        }));

        // Once `v2_peers` itself reaches capacity, `v2_peers_overflowed` fails closed so
        // an unknown retirement can never be downgraded to `Cancelled`.
        for peer in 0..(MAX_UNKNOWN_FENCED_JOBS as u64) {
            fenced.note_v2_peer(&format!("10.1.{}.{}:7200", peer >> 8, peer & 0xff));
        }
        assert!(fenced.v2_peers_overflowed);
        assert!(fenced.may_be_unknown(&JobKey {
            reply_to: "10.0.0.99:7200".into(),
            job: (9u64 << 32) | 1,
        }));
    }

    #[test]
    fn delayed_push_past_offer_accept_window_is_rejected_and_exec_deadline_is_capped() {
        use std::path::PathBuf;
        use std::sync::atomic::AtomicUsize;
        use ts_disk::{Backend, DiskFile, ExecutorConfig};

        assert!(
            OFFER_PUSH_ACCEPT_WINDOW >= super::super::v1::SUBMIT_WINDOW,
            "offer push acceptance window must cover SUBMIT_WINDOW"
        );
        assert!(
            OFFER_PUSH_DEADLINE + Duration::from_secs(10) < CANCEL_RETRY_BUDGET,
            "holder absolute push deadline must strictly precede reader cancel retry budget expiry"
        );
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
            "ts-push-offer-win-{}-{}.bin",
            std::process::id(),
            profile::now_ns()
        )));
        let disk_file = DiskFile::create(&temp.0, (2 * BLOCK) as u64).unwrap();
        let ring = Arc::new(Memory::map(BLOCK).unwrap());
        let executor = Arc::new(
            ts_disk::Executor::start(
                disk_file,
                &[(ring.address(), ring.size())],
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

        struct CountingWriter {
            calls: AtomicUsize,
            delay: Duration,
        }
        impl PageWriter for CountingWriter {
            fn write(&self, _: &str, _: &Spans) -> Result<(), String> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if !self.delay.is_zero() {
                    thread::sleep(self.delay);
                }
                Ok(())
            }
        }
        let writer = Arc::new(CountingWriter {
            calls: AtomicUsize::new(0),
            delay: Duration::from_millis(60),
        });
        let make_source = || PushSource {
            executor: executor.clone(),
            disk_layout: HostLayout {
                groups: vec![ts_l3::GroupLayout {
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
            ring: ring.clone(),
            slot_bytes: BLOCK,
        };
        let holder = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 302).unwrap();
        holder
            .serve_with_offer_windows(
                make_source(),
                Duration::from_millis(80),
                Duration::from_millis(120),
            )
            .unwrap();

        let (rtx, rrx) = mpsc::channel::<Vec<Bytes>>();
        let raw_reader = Sockets::start_at("127.0.0.1", 0, rtx).unwrap();
        let reply_ep = format!("127.0.0.1:{}", raw_reader.port());
        let (h_host, h_port) = split_endpoint(holder.name()).unwrap();
        let make_items = |n: usize| -> Vec<PushItem> {
            (0..n)
                .map(|_| PushItem {
                    group: 0,
                    block: 1,
                    spans: vec![PushSpan {
                        offset: 0,
                        address: 2048,
                        length: 64,
                    }],
                })
                .collect()
        };

        // 1. Activate offer 10, sleep past accept_window (80ms) WITHOUT closing
        //    the offer (simulating stalled-TCP retransmit after partition heal
        //    while retirement_bound_ms is unset or long). Must reject cleanly
        //    with Done{ok:false} and ZERO RDMA writes.
        holder.activate_offer(600, 10);
        thread::sleep(Duration::from_millis(110));
        let late_push = Message::Push {
            job: 10,
            offer: Some((600, 10)),
            reply_to: reply_ep.clone(),
            session: "s".into(),
            budget_us: 10_000_000,
            items: make_items(1),
        };
        raw_reader.send(h_host, h_port, late_push.frames()).unwrap();
        let reply = Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(reply, Some(Message::Done { job: 10, ok: false }));
        assert_eq!(
            writer.calls.load(Ordering::SeqCst),
            0,
            "stalled-TCP Push arriving past offer accept window must not execute any RDMA write"
        );

        // 2. Re-serve with a generous accept window (10s) and tight
        //    `offer_push_deadline` (120ms), activate offer 11, and submit a
        //    multi-block push with a huge 10s wire budget; the push must be
        //    accepted (writing >= 1 block) and then aborted by the activation-
        //    anchored execution cap long before the 10s wire budget elapses.
        holder
            .serve_with_offer_windows(
                make_source(),
                Duration::from_secs(10),
                Duration::from_millis(120),
            )
            .unwrap();
        holder.activate_offer(600, 11);
        let started = Instant::now();
        let capped_push = Message::Push {
            job: 11,
            offer: Some((600, 11)),
            reply_to: reply_ep,
            session: "s".into(),
            budget_us: 10_000_000,
            items: make_items(20),
        };
        raw_reader
            .send(h_host, h_port, capped_push.frames())
            .unwrap();
        let reply2 = Message::decode(&rrx.recv_timeout(Duration::from_secs(2)).unwrap());
        assert_eq!(reply2, Some(Message::Done { job: 11, ok: false }));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "offer_push_deadline must cap execution well below the 10s wire budget (elapsed {:?})",
            started.elapsed()
        );
        let calls = writer.calls.load(Ordering::SeqCst);
        assert!(
            (1..20).contains(&calls),
            "capped push must start writing (>= 1) then stop before all 20 blocks once offer_push_deadline elapses (got {calls})"
        );
    }

    #[test]
    fn build_write_spans_coalesces_tail_inplace_and_borrowed_slot() {
        use std::path::PathBuf;
        use ts_disk::{Backend, DiskFile, ExecutorConfig};

        struct TempDisk(PathBuf);
        impl Drop for TempDisk {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        struct NoopWriter;
        impl PageWriter for NoopWriter {
            fn write(&self, _: &str, _: &Spans) -> Result<(), String> {
                Ok(())
            }
        }

        let make_state = |slot_bytes: usize, num_slots: usize| {
            let dir = std::env::var_os("TS_DISK_TEST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            let temp = TempDisk(dir.join(format!(
                "ts-push-coalesce-{}-{}-{}.bin",
                std::process::id(),
                slot_bytes,
                profile::now_ns()
            )));
            let disk_file = DiskFile::create(&temp.0, 8192).unwrap();
            let mut ring = Memory::map(slot_bytes * num_slots).unwrap();
            // Fill slot 0 with 16 distinct 256-byte chunks.
            for chunk in 0..(slot_bytes / 256) {
                let byte = ((chunk * 17 + 3) & 0xff) as u8;
                ring.write_at(chunk * 256, &vec![byte; 256]);
            }
            let ring = Arc::new(ring);
            let executor = Arc::new(
                ts_disk::Executor::start(
                    disk_file,
                    &[(ring.address(), ring.size())],
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
            let source = PushSource {
                executor,
                disk_layout: HostLayout {
                    groups: vec![ts_l3::GroupLayout {
                        packing: 1,
                        block_bytes: 4096,
                        shard_count: 1,
                    }],
                    lcm_block_bytes: 4096,
                    num_lcm_blocks: 2,
                    owner_rank: 0,
                    tp_rank: 0,
                },
                writer: Arc::new(NoopWriter),
                ring: ring.clone(),
                slot_bytes,
            };
            let state = HolderState {
                source,
                slots: Slots {
                    total: num_slots,
                    max_per_job: num_slots,
                    quarantined: AtomicUsize::new(0),
                    free: Mutex::new((0..num_slots).rev().collect()),
                    returned: Condvar::new(),
                },
                running: Mutex::new(HashMap::new()),
                fenced: Mutex::new(FencedJobs::default()),
                active_offers: Mutex::new(HashMap::new()),
                offer_accept_window: OFFER_PUSH_ACCEPT_WINDOW,
                offer_push_deadline: OFFER_PUSH_DEADLINE,
            };
            (state, ring, temp)
        };

        // Four strided 256B slices (chunks 1, 3, 5, 7) targeting one contiguous 1024B remote range.
        let strided: Vec<PushSpan> = (0u64..4)
            .map(|i| PushSpan {
                offset: (2 * i + 1) * 256,
                address: 0x20_000 + i * 256,
                length: 256,
            })
            .collect();
        let mut expected = Vec::with_capacity(1024);
        for chunk in [1usize, 3, 5, 7] {
            expected.extend(std::iter::repeat_n(((chunk * 17 + 3) & 0xff) as u8, 256));
        }

        // Fast path 1: unused tail of the same slot (`range_size + pack_bytes <= slot_bytes`).
        let (state1, ring1, _t1) = make_state(8192, 1);
        let ring_addr1 = ring1.address();
        let (spans1, guard1) =
            state1.build_write_spans(ring_addr1, ring_addr1 as u64, 4096, false, &strided);
        assert!(guard1.is_none());
        assert_eq!(spans1.lengths, vec![1024]);
        assert_eq!(spans1.destinations, vec![0x20_000]);
        assert_eq!(spans1.sources, vec![(ring_addr1 + 4096) as u64]);
        let mut got1 = vec![0u8; 1024];
        ring1.read_at(4096, &mut got1);
        assert_eq!(got1, expected);

        // Fast path 2: in-place monotonic compaction (`range_size == slot_bytes`, `allow_in_place = true`).
        let (state2, ring2, _t2) = make_state(4096, 1);
        let ring_addr2 = ring2.address();
        let (spans2, guard2) =
            state2.build_write_spans(ring_addr2, ring_addr2 as u64, 4096, true, &strided);
        assert!(guard2.is_none());
        assert_eq!(spans2.lengths, vec![1024]);
        assert_eq!(spans2.destinations, vec![0x20_000]);
        assert_eq!(spans2.sources, vec![ring_addr2 as u64]);
        let mut got2 = vec![0u8; 1024];
        ring2.read_at(0, &mut got2);
        assert_eq!(got2, expected);

        // Fast path 3: borrow a spare staging slot (`range_size == slot_bytes`, `allow_in_place = false`).
        let (state3, ring3, _t3) = make_state(4096, 2);
        let ring_addr3 = ring3.address();
        let _busy0 = state3.slots.try_take().unwrap(); // hold slot 0 like the active read
        let (spans3, guard3) =
            state3.build_write_spans(ring_addr3, ring_addr3 as u64, 4096, false, &strided);
        assert!(guard3.is_some());
        assert_eq!(spans3.lengths, vec![1024]);
        assert_eq!(spans3.destinations, vec![0x20_000]);
        assert_eq!(spans3.sources, vec![(ring_addr3 + 4096) as u64]);
        let mut got3 = vec![0u8; 1024];
        ring3.read_at(4096, &mut got3);
        assert_eq!(got3, expected);

        // Non-contiguous remote destinations bypass coalescing untouched.
        let disjoint = vec![
            PushSpan {
                offset: 0,
                address: 0x20_000,
                length: 256,
            },
            PushSpan {
                offset: 512,
                address: 0x21_000,
                length: 256,
            },
        ];
        let (spans4, guard4) =
            state1.build_write_spans(ring_addr1, ring_addr1 as u64, 4096, true, &disjoint);
        assert!(guard4.is_none());
        assert_eq!(spans4.lengths, vec![256, 256]);
        assert_eq!(spans4.destinations, vec![0x20_000, 0x21_000]);
    }

    #[test]
    fn write_error_quarantines_original_and_borrowed_slots_and_degrades_disk_serving() {
        use std::path::PathBuf;
        use ts_disk::{Backend, DiskFile, ExecutorConfig};

        struct TempDisk(PathBuf);
        impl Drop for TempDisk {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        struct FailingWriter;
        impl PageWriter for FailingWriter {
            fn write(&self, _: &str, _: &Spans) -> Result<(), String> {
                Err("transport write fault".into())
            }
        }

        let make_holder = |slot_bytes: usize, num_slots: usize| {
            let dir = std::env::var_os("TS_DISK_TEST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            let temp = TempDisk(dir.join(format!(
                "ts-push-write-err-{}-{}-{}.bin",
                std::process::id(),
                num_slots,
                profile::now_ns()
            )));
            let disk_file = DiskFile::create(&temp.0, 8192).unwrap();
            let ring = Arc::new(Memory::map(slot_bytes * num_slots).unwrap());
            let executor = Arc::new(
                ts_disk::Executor::start(
                    disk_file,
                    &[(ring.address(), ring.size())],
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
            let holder = Holder::new(PushSource {
                executor,
                disk_layout: HostLayout {
                    groups: vec![ts_l3::GroupLayout {
                        packing: 1,
                        block_bytes: 4096,
                        shard_count: 1,
                    }],
                    lcm_block_bytes: 4096,
                    num_lcm_blocks: 2,
                    owner_rank: 0,
                    tp_rank: 0,
                },
                writer: Arc::new(FailingWriter),
                ring,
                slot_bytes,
            })
            .unwrap();
            (holder, temp)
        };

        // Path 1: single batch referencing the original staging slot.
        // Failing `writer.write()` must quarantine the original slot and
        // degrade `can_serve_disk()` once all slots are quarantined.
        let (holder1, _t1) = make_holder(4096, 2);
        assert!(holder1.can_serve_disk());
        assert_eq!(holder1.state.slots.free_count(), 2);
        let item_orig = vec![PushItem {
            group: 0,
            block: 1,
            spans: vec![PushSpan {
                offset: 0,
                address: 0x10_000,
                length: 256,
            }],
        }];
        let running = Running::new(None);
        assert_eq!(
            holder1.state.run(
                "peer",
                Instant::now() + Duration::from_secs(5),
                &item_orig,
                &running,
            ),
            RunOutcome::UnknownRetirement
        );
        assert_eq!(holder1.state.slots.free_count(), 1);
        assert!(holder1.can_serve_disk());
        // Second failing write quarantines the last remaining slot -> degrades Disk serving.
        assert_eq!(
            holder1.state.run(
                "peer",
                Instant::now() + Duration::from_secs(5),
                &item_orig,
                &running,
            ),
            RunOutcome::UnknownRetirement
        );
        assert_eq!(holder1.state.slots.free_count(), 0);
        assert!(!holder1.can_serve_disk());
        holder1.shutdown();

        // Path 2: two batches on the same full-slot block (`range_size == slot_bytes`,
        // `allow_in_place = false`) force Fast path 3 to borrow a second staging slot.
        // Batch 0 mixes a contiguous merged run (in the borrowed pack slot) with a
        // disjoint span (still in the original read slot), so `writer.write()`
        // references BOTH slots; both must be quarantined on write failure.
        let (holder2, _t2) = make_holder(4096, 3);
        assert_eq!(holder2.state.slots.free_count(), 3);
        let items_borrowed_and_orig = vec![
            PushItem {
                group: 0,
                block: 1,
                spans: vec![
                    // Merged into borrowed pack slot:
                    PushSpan {
                        offset: 0,
                        address: 0x20_000,
                        length: 256,
                    },
                    PushSpan {
                        offset: 512,
                        address: 0x20_100,
                        length: 256,
                    },
                    // Disjoint single span kept directly in original read slot:
                    PushSpan {
                        offset: 1024,
                        address: 0x30_000,
                        length: 256,
                    },
                ],
            },
            // Second item on the same `(group, block)` sets `allow_in_place = false`.
            PushItem {
                group: 0,
                block: 1,
                spans: vec![PushSpan {
                    offset: 2048,
                    address: 0x40_000,
                    length: 256,
                }],
            },
        ];
        assert_eq!(
            holder2.state.run(
                "peer",
                Instant::now() + Duration::from_secs(5),
                &items_borrowed_and_orig,
                &running,
            ),
            RunOutcome::UnknownRetirement
        );
        assert_eq!(
            holder2.state.slots.free_count(),
            1,
            "both the original read slot and the borrowed pack slot must be quarantined"
        );
        assert!(holder2.can_serve_disk());
        holder2.shutdown();

        // Path 3: borrowed pack slot holds ALL spans of batch 0 (none left in the
        // original read slot) -> only the borrowed pack slot is quarantined while
        // the unreferenced original read slot returns cleanly to the pool.
        let (holder3, _t3) = make_holder(4096, 3);
        let items_borrowed_only = vec![
            PushItem {
                group: 0,
                block: 1,
                spans: vec![
                    PushSpan {
                        offset: 0,
                        address: 0x20_000,
                        length: 256,
                    },
                    PushSpan {
                        offset: 512,
                        address: 0x20_100,
                        length: 256,
                    },
                ],
            },
            PushItem {
                group: 0,
                block: 1,
                spans: vec![PushSpan {
                    offset: 2048,
                    address: 0x40_000,
                    length: 256,
                }],
            },
        ];
        assert_eq!(
            holder3.state.run(
                "peer",
                Instant::now() + Duration::from_secs(5),
                &items_borrowed_only,
                &running,
            ),
            RunOutcome::UnknownRetirement
        );
        assert_eq!(
            holder3.state.slots.free_count(),
            2,
            "borrowed pack slot must be quarantined while unreferenced read slot returns"
        );
        holder3.shutdown();
    }
}
