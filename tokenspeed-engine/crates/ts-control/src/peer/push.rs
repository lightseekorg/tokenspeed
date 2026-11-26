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
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use parking_lot::{Condvar, Mutex};
use ts_disk::{Direction, Executor, Io, Job, JobReport, Memory};
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

/// The ring's free slots.
struct Slots {
    free: Mutex<Vec<usize>>,
    returned: Condvar,
}

impl Slots {
    fn take(&self, deadline: Instant) -> Option<usize> {
        let mut free = self.free.lock();
        loop {
            if let Some(slot) = free.pop() {
                return Some(slot);
            }
            if self.returned.wait_until(&mut free, deadline).timed_out() {
                return free.pop();
            }
        }
    }

    fn try_take(&self) -> Option<usize> {
        self.free.lock().pop()
    }

    fn give_back(&self, slot: usize) {
        self.free.lock().push(slot);
        self.returned.notify_one();
    }
}

/// A push the holder runs, and whether the reader cancelled it.
struct Running {
    cancelled: AtomicBool,
    /// Where the cancellation's acknowledgment goes once the job ended.
    acknowledge: Mutex<Option<String>>,
}

/// The holder's half: the pushes it runs.
struct Holder {
    source: PushSource,
    slots: Slots,
    running: Mutex<HashMap<u64, Arc<Running>>>,
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
        Ok(Self {
            source,
            slots: Slots {
                free: Mutex::new((0..slots).collect()),
                returned: Condvar::new(),
            },
            running: Mutex::new(HashMap::new()),
        })
    }

    /// Runs one push to its end: every block read into a slot and written to
    /// the reader, unless the budget passes, the reader cancels or an I/O
    /// fails. Returns whether every block landed.
    fn run(&self, session: &str, budget: Duration, items: &[PushItem], job: &Running) -> bool {
        // On a profile timeline: the push, and each block's write inside it.
        static PUSHES: Lanes = Lanes::new("Peer push");
        let lane = PUSHES.hold();
        let begin_ns = profile::now_ns();
        let deadline = Instant::now() + budget;
        let ring = self.source.ring.address();
        let (done, completions) = mpsc::channel::<(usize, usize, bool)>();
        let mut next = 0;
        let mut in_flight = 0;
        let mut ok = true;
        let stopped =
            |ok: bool| !ok || job.cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline;
        loop {
            // Reads go ahead of the writes, as many as the ring has slots for.
            while next < items.len() && !stopped(ok) {
                let slot = if in_flight == 0 {
                    self.slots.take(deadline)
                } else {
                    self.slots.try_take()
                };
                let Some(slot) = slot else { break };
                let item = &items[next];
                let range = match self.source.disk_layout.owned_range(item.group, item.block) {
                    Ok(Some(range)) => range,
                    _ => {
                        self.slots.give_back(slot);
                        ok = false;
                        break;
                    }
                };
                let done = done.clone();
                let index = next;
                let submitted = self.source.executor.submit(
                    Job {
                        direction: Direction::Read,
                        ios: vec![Io {
                            file_offset: range.offset as u64,
                            address: ring + slot * self.source.slot_bytes,
                            len: range.size,
                        }],
                    },
                    move |report: JobReport| {
                        let _ = done.send((index, slot, report.outcome.is_ok()));
                    },
                );
                if submitted.is_err() {
                    self.slots.give_back(slot);
                    ok = false;
                    break;
                }
                next += 1;
                in_flight += 1;
            }
            if in_flight == 0 {
                break;
            }
            let Ok((index, slot, read)) = completions.recv() else {
                break;
            };
            in_flight -= 1;
            ok &= read;
            if !stopped(ok) {
                let base = (ring + slot * self.source.slot_bytes) as u64;
                let spans = Spans {
                    sources: items[index]
                        .spans
                        .iter()
                        .map(|span| base + span.offset)
                        .collect(),
                    destinations: items[index].spans.iter().map(|span| span.address).collect(),
                    lengths: items[index].spans.iter().map(|span| span.length).collect(),
                    host_bytes: 0,
                };
                let write_ns = profile::now_ns();
                if let Err(error) = self.source.writer.write(session, &spans) {
                    tracing::warn!(%error, "peer push write failed");
                    ok = false;
                }
                if let Some(lane) = &lane {
                    lane.record(profile::HOST, "write".into(), write_ns, profile::now_ns());
                }
            }
            self.slots.give_back(slot);
        }
        let landed = ok && next == items.len() && !job.cancelled.load(Ordering::Relaxed);
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
                    items.len(),
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
    sockets: Arc<Sockets>,
    holder: Arc<Mutex<Option<Arc<Holder>>>>,
    waiters: Arc<Mutex<HashMap<u64, mpsc::Sender<Message>>>>,
    next_job: AtomicU64,
    _dispatcher: thread::JoinHandle<()>,
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
            let (sockets, holder, waiters) = (sockets.clone(), holder.clone(), waiters.clone());
            thread::Builder::new()
                .name("ts-peer-push".to_owned())
                .spawn(move || {
                    for frames in received {
                        match Message::decode(&frames) {
                            Some(Message::Push {
                                job,
                                reply_to,
                                session,
                                budget_us,
                                items,
                            }) => {
                                let serving = holder.lock().clone();
                                start_job(
                                    serving,
                                    sockets.clone(),
                                    job,
                                    reply_to,
                                    session,
                                    Duration::from_micros(budget_us),
                                    items,
                                );
                            }
                            Some(Message::Cancel { job, reply_to }) => {
                                let running = holder
                                    .lock()
                                    .as_ref()
                                    .and_then(|holder| holder.running.lock().get(&job).cloned());
                                match running {
                                    Some(running) => {
                                        running.cancelled.store(true, Ordering::Relaxed);
                                        *running.acknowledge.lock() = Some(reply_to);
                                    }
                                    // Ended or never started: no write follows.
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
            sockets,
            holder,
            waiters,
            next_job: AtomicU64::new(incarnation << 32),
            _dispatcher: dispatcher,
        })
    }

    /// Where other ranks reach this endpoint.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Serve pushes of `source`'s Disk pages from now on.
    pub fn serve(&self, source: PushSource) -> Result<(), PushError> {
        *self.holder.lock() = Some(Arc::new(Holder::new(source).map_err(PushError::Settings)?));
        Ok(())
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
        self.sockets
            .send(host, port, request.frames())
            .map_err(|error| error.to_string())?;
        // The holder's answer may trail its budget by a network round trip.
        match answers.recv_timeout(budget + Duration::from_secs(1)) {
            Ok(Message::Done { ok: true, .. }) => return Ok(()),
            Ok(Message::Done { ok: false, .. }) => {
                return Err("the holder could not push".to_owned())
            }
            _ => {}
        }
        let cancel = Message::Cancel {
            job,
            reply_to: self.name.clone(),
        };
        let _ = self.sockets.send(host, port, cancel.frames());
        let wait_end = Instant::now() + CANCEL_WAIT;
        while let Some(remaining) = wait_end.checked_duration_since(Instant::now()) {
            match answers.recv_timeout(remaining) {
                Ok(Message::Cancelled { .. } | Message::Done { .. }) => {
                    return Err("the push did not finish in its budget".to_owned())
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

/// Runs a push on its own thread and answers it: a holder without a Disk
/// tier answers at once that nothing landed.
fn start_job(
    holder: Option<Arc<Holder>>,
    sockets: Arc<Sockets>,
    job: u64,
    reply_to: String,
    session: String,
    budget: Duration,
    items: Vec<PushItem>,
) {
    let Some(holder) = holder else {
        send(&sockets, &reply_to, &Message::Done { job, ok: false });
        return;
    };
    let running = Arc::new(Running {
        cancelled: AtomicBool::new(false),
        acknowledge: Mutex::new(None),
    });
    holder.running.lock().insert(job, running.clone());
    let spawned = thread::Builder::new()
        .name("ts-peer-push-job".to_owned())
        .spawn({
            let (holder, sockets, reply_to) = (holder.clone(), sockets.clone(), reply_to.clone());
            move || {
                let started = Instant::now();
                let ok = holder.run(&session, budget, &items, &running);
                tracing::debug!(
                    job,
                    blocks = items.len(),
                    ok,
                    elapsed_us = started.elapsed().as_micros() as u64,
                    "peer push ended"
                );
                holder.running.lock().remove(&job);
                send(&sockets, &reply_to, &Message::Done { job, ok });
                if let Some(acknowledge) = running.acknowledge.lock().take() {
                    send(&sockets, &acknowledge, &Message::Cancelled { job });
                }
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "peer push not started");
        holder.running.lock().remove(&job);
        send(&sockets, &reply_to, &Message::Done { job, ok: false });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_message_survives_the_wire() {
        for message in [
            Message::Push {
                job: (7 << 32) + 1,
                reply_to: "10.0.0.3:7200".into(),
                session: "10.0.0.3:12345".into(),
                budget_us: 2_000_000,
                items: vec![PushItem {
                    group: 1,
                    block: 42,
                    spans: vec![
                        PushSpan {
                            offset: 0,
                            address: 1 << 40,
                            length: 4096,
                        },
                        PushSpan {
                            offset: 4096,
                            address: (1 << 40) + 65536,
                            length: 2048,
                        },
                    ],
                }],
            },
            Message::Cancel {
                job: 3,
                reply_to: "[::1]:7200".into(),
            },
            Message::Done { job: 3, ok: true },
            Message::Cancelled { job: 3 },
        ] {
            assert_eq!(Message::decode(&message.frames()), Some(message));
        }
        assert_eq!(
            Message::decode(&[Bytes::from_static(b"?"), Bytes::new()]),
            None
        );
    }
}
