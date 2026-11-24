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

//! A storage source broker using the ordinary peer lookup/admission protocol.
//!
//! All metadata queries and source preparation run on this I/O thread. Lookup
//! returns metadata only; PIN prepares and exposes registered source addresses
//! after the reader's ordinary admission. No Device or Host cache page lives here.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use ts_scheduler::{PeerAnswerKind, PeerAnswerView, PeerLeaseView, PeerPageSlotView, PeerPageView};

use crate::peer::{
    read::ReadOutcome,
    storage::{
        self, Backend, Completion, Error, Limits, PrefixPlan, PreparedSource, ReadingPrefix,
        SourceBudget,
    },
    wire::{Prompt, Reply, Request, Stamp},
};

/// Planner callbacks inspect scheduler metadata only. The source worker must
/// release any scheduler guard before invoking the backend.
/// Complete prefix dependency plans for the specified identity contract.
pub type Planner = Arc<dyn Fn(&Prompt, &str) -> Result<Vec<PrefixPlan>, Error> + Send + Sync>;
/// The reader's actual all-group local match; holds no pages or source lease.
pub type LocalPrefix = Arc<dyn Fn(&Prompt) -> Result<i32, Error> + Send + Sync>;

/// Explicit broker limits. The endpoint is not a hot-cache inventory publisher.
#[derive(Clone)]
pub struct Settings {
    /// Same deployment and weight identity as the reader; a new incarnation per startup.
    pub stamp: Stamp,
    /// Persistent object weight generation, passed to the metadata planner.
    pub weight_version: String,
    /// Reader replica members whose dependencies each plan includes.
    pub members: usize,
    /// Source query and preparation budgets per lookup.
    pub limits: Limits,
    /// Bounded outstanding logical requests (waiting and reading).
    pub max_records: usize,
    /// Minimum complete prefix extension worth querying and preparing.
    pub min_extra_tokens: i32,
    /// Maximum time a record is retained without a reader message.
    pub record_lifetime: Duration,
    /// Remaining source lifetime required before exposing addresses.
    pub read_budget: Duration,
}

enum State<P: PreparedSource> {
    Queried(storage::Hit),
    Reading(ReadingPrefix<P::Reading>),
}
struct Record<P: PreparedSource> {
    prompt: Prompt,
    reused_pages: Vec<PeerPageSlotView>,
    destinations: Option<Vec<PeerPageSlotView>>,
    boundary: i32,
    expires: Instant,
    state: State<P>,
}

// Storage-only metadata stays inside this process; hot-peer wire messages
// and deployed providers do not need a protocol change.
struct Command {
    request: Request,
    reused_pages: Vec<PeerPageSlotView>,
    destinations: Option<Vec<PeerPageSlotView>>,
}

/// Owns the broker thread. Drop stops it and releases unused source leases on
/// that I/O thread. Submitted leases without retirement retain their source TTL.
pub struct Service {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    budget: SourceBudget,
    namespace: Arc<Mutex<Option<(String, String)>>>,
    name: String,
    requests: mpsc::SyncSender<Command>,
    replies: mpsc::Receiver<Reply>,
}

impl Service {
    /// Start an in-process storage worker. Queries retain metadata only;
    /// preparation begins on Pin, after the reader has admitted Device pages.
    pub fn start<B: Backend + 'static>(
        mut settings: Settings,
        planner: Planner,
        local_prefix: LocalPrefix,
        backend: B,
        budget: SourceBudget,
    ) -> Result<Self, Error> {
        if settings.members == 0
            || settings.members > 1024
            || settings.max_records == 0
            || settings.max_records > 1024
            || settings.min_extra_tokens <= 0
            || settings.record_lifetime.is_zero()
            || settings.read_budget.is_zero()
            || settings.stamp.name.is_empty()
            || settings.stamp.identity.is_empty()
        {
            return Err(Error("invalid storage worker settings".into()));
        }
        let name = settings.stamp.name.clone();
        let (requests, incoming) = mpsc::sync_channel(settings.max_records * 4);
        let (outgoing, replies) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let worker_budget = budget.clone();
        let namespace = Arc::new(Mutex::new(None::<(String, String)>));
        let updates = namespace.clone();
        let worker = thread::Builder::new()
            .name("ts-storage-source".into())
            .spawn(move || {
                let mut records: HashMap<PeerLeaseView, Record<B::Prepared>> = HashMap::new();
                while !stopped.load(Ordering::Acquire) {
                    if let Some((identity, version)) =
                        updates.lock().unwrap_or_else(|e| e.into_inner()).take()
                    {
                        records.clear();
                        settings.stamp.identity = identity;
                        settings.weight_version = version;
                    }
                    let now = Instant::now();
                    records.retain(|_, record| now < record.expires);
                    let request = match incoming.recv_timeout(Duration::from_millis(20)) {
                        Ok(request) => request,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    };
                    let (_, reply) = handle(
                        request,
                        &settings,
                        &planner,
                        &local_prefix,
                        &backend,
                        &worker_budget,
                        &mut records,
                    );
                    if outgoing.send(reply).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| Error(e.to_string()))?;
        Ok(Self {
            stop,
            worker: Some(worker),
            budget,
            namespace,
            name,
            requests,
            replies,
        })
    }

    /// Logical holder name used for dispatch within this process.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Bounded and nonblocking: the control loop never waits for storage I/O.
    pub fn request(&self, request: Request) -> Result<(), Error> {
        self.requests
            .try_send(Command {
                request,
                reused_pages: Vec::new(),
                destinations: None,
            })
            .map_err(|e| Error(e.to_string()))
    }

    /// Query the dependencies not supplied by the same prompt's hot prefix.
    pub fn lookup(
        &self,
        lease: PeerLeaseView,
        prompt: Prompt,
        reused_pages: Vec<PeerPageSlotView>,
    ) -> Result<(), Error> {
        self.requests
            .try_send(Command {
                request: Request::Lookup { lease, prompt },
                reused_pages,
                destinations: None,
            })
            .map_err(|e| Error(e.to_string()))
    }

    /// Prepare only ordinarily admitted destinations, after the source split.
    pub fn prepare(&self, lease: PeerLeaseView, pages: &[PeerPageView]) -> Result<(), Error> {
        self.requests
            .try_send(Command {
                request: Request::Pin { lease },
                reused_pages: Vec::new(),
                destinations: Some(
                    pages
                        .iter()
                        .map(|page| PeerPageSlotView {
                            group_id: page.group_id,
                            slot: page.slot,
                        })
                        .collect(),
                ),
            })
            .map_err(|e| Error(e.to_string()))
    }

    /// Completed replies, drained without waiting by the control thread.
    pub fn poll(&self) -> impl Iterator<Item = Reply> + '_ {
        self.replies.try_iter()
    }

    /// Queue a namespace change after the host's ordinary flush. Old replies
    /// retain their old stamp, so the reader rejects them while this is pending.
    pub fn set_namespace(&self, identity: String, weight_version: String) {
        *self.namespace.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((identity, weight_version));
    }

    /// Reserved and quarantined source bytes, including prepared waiting work.
    pub fn source_usage(&self) -> (u64, u64) {
        self.budget.usage()
    }
}
impl Drop for Service {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn validate_candidates(
    candidates: &[PrefixPlan],
    prompt: &Prompt,
    members: usize,
) -> Result<(), Error> {
    if candidates
        .iter()
        .any(|p| p.members != members || p.num_tokens > prompt.max_tokens)
    {
        return Err(Error("planner boundary/owner mismatch".into()));
    }
    Ok(())
}

fn answer(
    settings: &Settings,
    lease: PeerLeaseView,
    kind: PeerAnswerKind,
    num_tokens: i32,
    held: bool,
) -> (String, Reply) {
    (
        lease.reader.clone(),
        Reply::Answer {
            stamp: settings.stamp.clone(),
            answer: PeerAnswerView {
                lease,
                kind,
                num_tokens,
                held,
                pages: Vec::new(),
            },
        },
    )
}

fn handle<B: Backend>(
    command: Command,
    settings: &Settings,
    planner: &Planner,
    local_prefix: &LocalPrefix,
    backend: &B,
    budget: &SourceBudget,
    records: &mut HashMap<PeerLeaseView, Record<B::Prepared>>,
) -> (String, Reply) {
    let Command {
        request,
        reused_pages,
        destinations,
    } = command;
    match request {
        Request::Layout { reply_to } => (
            reply_to,
            Reply::StorageLayout {
                stamp: settings.stamp.clone(),
                members: settings.members as u32,
            },
        ),
        Request::Lookup { lease, prompt } => {
            if let Some(record) = records.get(&lease) {
                // Retransmission is idempotent. Reusing a live id for another
                // prompt must never replace a source underneath its reader.
                let boundary = if record.prompt == prompt && record.reused_pages == reused_pages {
                    record.boundary
                } else {
                    0
                };
                return answer(settings, lease, PeerAnswerKind::Lookup, boundary, false);
            }
            if records.len() >= settings.max_records {
                return answer(settings, lease, PeerAnswerKind::Lookup, 0, false);
            }
            let queried = (|| {
                let local = local_prefix(&prompt)?;
                if local < 0 || local > prompt.max_tokens {
                    return Err(Error("invalid local prefix".into()));
                }
                if prompt.max_tokens - local < settings.min_extra_tokens {
                    return Ok(None);
                }
                let mut candidates = planner(&prompt, &settings.weight_version)?;
                validate_candidates(&candidates, &prompt, settings.members)?;
                for candidate in &mut candidates {
                    candidate.objects.retain(|object| {
                        !reused_pages.iter().any(|page| {
                            page.group_id == object.group_id && page.slot == object.slot
                        })
                    });
                }
                candidates.retain(|candidate| !candidate.objects.is_empty());
                let Some(hit) = storage::query(backend, &candidates, settings.limits)? else {
                    return Ok(None);
                };
                if hit.num_tokens() - local < settings.min_extra_tokens
                    || hit.bytes() > budget.available()
                {
                    return Ok(None);
                }
                Ok(Some(hit))
            })();
            let boundary = match queried {
                Ok(Some(hit)) => {
                    let boundary = hit.num_tokens();
                    records.insert(
                        lease.clone(),
                        Record {
                            prompt,
                            reused_pages,
                            destinations: None,
                            boundary,
                            expires: Instant::now() + settings.record_lifetime,
                            state: State::Queried(hit),
                        },
                    );
                    boundary
                }
                Ok(None) => 0,
                Err(error) => {
                    tracing::warn!(%error, "L3 metadata lookup unavailable");
                    0
                }
            };
            answer(settings, lease, PeerAnswerKind::Lookup, boundary, false)
        }
        Request::Pin { lease } => {
            let Some(mut record) = records.remove(&lease) else {
                return answer(settings, lease, PeerAnswerKind::Pin, 0, false);
            };
            if matches!(record.state, State::Reading(_)) && record.destinations != destinations {
                // An idempotent retry cannot change sources beneath a read.
                records.insert(lease.clone(), record);
                return answer(settings, lease, PeerAnswerKind::Pin, 0, false);
            }
            let reading = match record.state {
                State::Queried(hit) => match destinations.as_ref() {
                    Some(pages) => hit.restrict_to(pages),
                    None => Ok(hit),
                }
                .and_then(|hit| storage::prepare(backend, hit, settings.limits, budget))
                .and_then(|ready| ready.ok_or_else(|| Error("L3 source budget unavailable".into())))
                .and_then(|ready| ready.begin_read(settings.read_budget)),
                State::Reading(reading) => Ok(reading),
            };
            let Ok(reading) = reading else {
                return answer(settings, lease, PeerAnswerKind::Pin, 0, false);
            };
            let ranks = (0..settings.members)
                .map(|rank| reading.objects(rank))
                .collect::<Result<Vec<_>, _>>();
            let Ok(ranks) = ranks else {
                // A duplicate PIN may mean the reader already started. Do not
                // manufacture retirement on malformed descriptors; keep TTL.
                return answer(settings, lease, PeerAnswerKind::Pin, 0, false);
            };
            record.state = State::Reading(reading);
            record.destinations = destinations;
            record.expires = Instant::now() + settings.record_lifetime;
            let boundary = record.boundary;
            records.insert(lease.clone(), record);
            (
                lease.reader.clone(),
                Reply::Prepared {
                    stamp: settings.stamp.clone(),
                    lease,
                    num_tokens: boundary,
                    ranks,
                },
            )
        }
        Request::Release { lease, demote } => {
            let complete = match records.remove(&lease).map(|r| r.state) {
                Some(State::Reading(reading)) => {
                    // The reader sends release only after all members proved
                    // retirement; false means some retired read was unsuccessful.
                    let outcome = if demote {
                        ReadOutcome::Complete
                    } else {
                        ReadOutcome::FailedRetired
                    };
                    reading.finish(&vec![outcome; settings.members]) == Completion::Complete
                }
                _ => false,
            };
            answer(settings, lease, PeerAnswerKind::Release, 0, complete)
        }
    }
}
