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

//! A prefill rank's client of the shared L2 pool's owner, once the rank has
//! joined the pool.
//!
//! * A lookup asks the owner, in one request on the control thread, for the
//!   pool's hit of every waiting prompt, and waits for the answer. The owner
//!   pins each hit's pages, which the scheduler mirrors into proxy Host
//!   blocks; the client keeps the pool page behind each for the rank's loads
//!   ([`Loads`]).
//! * A write is queued for the write threads: one reserves pool pages for
//!   the write's blocks, copies the blocks there ([`PageWriter`]) and
//!   commits them.
//! * A release and a touch are notices, which one thread sends in order; it
//!   pings the owner when it has had nothing to send for the keepalive
//!   interval.
//!
//! Nothing is sent twice, and every failure stops the rank's engine: a
//! request the owner does not answer in time or answers with an error (an
//! owner that does not know the rank answers so), an answer that is not one
//! hit of the wire's shape, on pages the pool has, for each lookup, a
//! reservation that does not answer each key of the write with a page the
//! pool has or with none, a copy that fails, a commit the owner does not
//! carry out. A lookup returns its failure; a write's, a notice's and a
//! load's are kept ([`FailureLatch`]) for the loop's next poll or check.

use std::{
    num::NonZeroUsize,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use ts_scheduler::{
    SharedL2AnswerView, SharedL2LookupView, SharedL2OpKind, SharedL2OpView, SharedL2ProxyBlockView,
    StorageKeyView,
};

use crate::{
    peer::{
        geometry::Spans,
        read::{Arena, PageReader},
    },
    shared_l2::{
        geometry::{PageCopy, PoolGeometry},
        load::{Loads, ProxyPages, SignalWords},
        panic_as_failure,
        wire::OwnerConnection,
        FailureLatch, PoolClient, PoolError,
    },
};

/// What moves a write's bytes. A load's bytes come through a peer read's
/// [`PageReader`], over the same transfer engine.
pub trait PageWriter: Send + Sync {
    /// Copy `spans` to the memory registered under `session`: span i copies
    /// `lengths[i]` bytes from the rank's `sources[i]` to the session's
    /// `destinations[i]`. Returns once every copy retired, or the failure.
    /// May be called concurrently for separate writes, as the transfer
    /// engine's entry points allow.
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

/// Within test memory a write is the copy a read is.
impl PageWriter for Arena {
    fn write(&self, session: &str, spans: &Spans) -> Result<(), String> {
        self.read(session, spans)
    }
}

/// What a joined rank starts its pool client from.
pub struct ClientSettings {
    /// The owner's address.
    pub owner_address: String,
    /// The rank's id at the owner, from its hello.
    pub client_id: u64,
    /// How long a request waits for the owner's answer. One not answered by
    /// then has failed.
    pub request_timeout: Duration,
    /// How long the rank sends no notice before it pings the owner, which
    /// drops a client it has not heard for its client timeout.
    pub keepalive: Duration,
    /// Threads reading one load's providers in parallel.
    pub read_workers: NonZeroUsize,
    /// Threads writing to the pool.
    pub write_workers: NonZeroUsize,
    /// The pool and the rank's Device cache.
    pub geometry: PoolGeometry,
}

/// What the notice thread tells the owner.
enum Notice {
    /// Unpin what a lookup hit under this ticket.
    Release(u64),
    /// The rank used its own copies of these blocks.
    Touch(Vec<StorageKeyView>),
    /// The rank is there: it had nothing to send for the keepalive interval.
    Ping,
}

/// The client as the loop drives it, on the control thread.
pub struct RankClient {
    owner: OwnerConnection,
    geometry: Arc<PoolGeometry>,
    proxy_pages: Arc<ProxyPages>,
    failure: Arc<FailureLatch>,
    /// The plan's write ops; `None` tells a write thread to stop.
    writes: mpsc::Sender<Option<SharedL2OpView>>,
    /// The plan's notices; `None` tells the notice thread to stop.
    notices: mpsc::Sender<Option<Notice>>,
    /// The ids of the writes the write threads finished.
    finished_writes: mpsc::Receiver<u32>,
}

/// The client's threads and the rank's loads: what the forward thread hands
/// its loads to, and what the rank's shutdown closes.
pub struct RankThreads {
    loads: Loads,
    writes: mpsc::Sender<Option<SharedL2OpView>>,
    notices: mpsc::Sender<Option<Notice>>,
    write_workers: usize,
    /// The write and notice threads; empty once closed.
    threads: Mutex<Vec<thread::JoinHandle<()>>>,
    /// What the threads read and write through, kept past them: released
    /// last, by the close that saw every thread end.
    transfer: Mutex<Option<Transfer>>,
}

/// What the client's threads read and write through: the transfer engine
/// in a server.
struct Transfer {
    _reader: Arc<dyn PageReader>,
    _writer: Arc<dyn PageWriter>,
}

/// Start a joined rank's client: its connections to the owner, its write
/// and notice threads and its loads, which read through `reader`, write
/// through `writer` and signal layers through `signal_words`.
pub fn start(
    settings: ClientSettings,
    reader: Arc<dyn PageReader>,
    writer: Arc<dyn PageWriter>,
    signal_words: Arc<dyn SignalWords>,
) -> Result<(RankClient, RankThreads), PoolError> {
    // A copy divides by its group's packing and indexes its fields' layers
    // without a check of its own.
    let num_layers = settings.geometry.num_layers;
    for (group, page_group) in settings.geometry.groups.iter().enumerate() {
        if page_group.packing == 0 {
            return Err(PoolError(format!(
                "shared L2 cache group {group} has no page in an LCM block"
            )));
        }
        if let Some(field) = page_group
            .fields
            .iter()
            .find(|field| field.layer >= num_layers)
        {
            return Err(PoolError(format!(
                "a field of shared L2 cache group {group} is read by layer {}, and the rank \
                 has {num_layers} layers",
                field.layer
            )));
        }
    }
    let spawn_failed = |error: std::io::Error| {
        PoolError(format!(
            "starting a shared L2 client thread failed: {error}"
        ))
    };
    // One connection per thread that asks the owner: the owner reads a
    // thread's requests in the order the thread sent them.
    let connect = || {
        OwnerConnection::connect(
            &settings.owner_address,
            settings.client_id,
            settings.request_timeout,
        )
    };
    let owner = connect()?;
    let geometry = Arc::new(settings.geometry);
    let failure = Arc::new(FailureLatch::default());
    let proxy_pages = Arc::new(ProxyPages::default());
    let loads = Loads::start(
        geometry.clone(),
        proxy_pages.clone(),
        reader.clone(),
        signal_words,
        failure.clone(),
        settings.read_workers,
    )
    .map_err(spawn_failed)?;

    let (writes, incoming_writes) = mpsc::channel::<Option<SharedL2OpView>>();
    let (finished, finished_writes) = mpsc::channel();
    let incoming_writes = Arc::new(Mutex::new(incoming_writes));
    // A thread that started ends when a later step fails: its queue's
    // sender is dropped with this function's locals.
    let mut threads = Vec::with_capacity(settings.write_workers.get() + 1);
    for index in 0..settings.write_workers.get() {
        let mut worker = WriteWorker {
            owner: connect()?,
            geometry: geometry.clone(),
            writer: writer.clone(),
        };
        let (incoming_writes, finished, failure) =
            (incoming_writes.clone(), finished.clone(), failure.clone());
        threads.push(
            thread::Builder::new()
                .name(format!("ts-shared-l2-write-{index}"))
                .spawn(move || loop {
                    // Release the queue lock before the write.
                    let write = incoming_writes.lock().recv();
                    let Ok(Some(write)) = write else {
                        return;
                    };
                    match panic_as_failure(|| worker.write(&write)) {
                        Ok(()) => {
                            let _ = finished.send(write.op_id);
                        }
                        Err(error) => {
                            tracing::error!(
                                %error,
                                write = write.op_id,
                                "shared L2 write failed"
                            );
                            failure.record(error);
                        }
                    }
                })
                .map_err(spawn_failed)?,
        );
    }

    let (notices, incoming_notices) = mpsc::channel();
    let (mut notice_owner, keepalive) = (connect()?, settings.keepalive);
    let notice_failure = failure.clone();
    threads.push(
        thread::Builder::new()
            .name("ts-shared-l2-notice".to_owned())
            .spawn(move || loop {
                let notice = match incoming_notices.recv_timeout(keepalive) {
                    Ok(Some(notice)) => notice,
                    Err(mpsc::RecvTimeoutError::Timeout) => Notice::Ping,
                    Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
                };
                let sent = panic_as_failure(|| match notice {
                    Notice::Release(ticket) => notice_owner.release(ticket),
                    Notice::Touch(keys) => notice_owner.touch(&keys),
                    Notice::Ping => notice_owner.ping(),
                });
                if let Err(error) = sent {
                    tracing::error!(%error, "shared L2 notice failed");
                    notice_failure.record(error);
                }
            })
            .map_err(spawn_failed)?,
    );

    Ok((
        RankClient {
            owner,
            geometry,
            proxy_pages,
            failure,
            writes: writes.clone(),
            notices: notices.clone(),
            finished_writes,
        },
        RankThreads {
            loads,
            writes,
            notices,
            write_workers: settings.write_workers.get(),
            threads: Mutex::new(threads),
            transfer: Mutex::new(Some(Transfer {
                _reader: reader,
                _writer: writer,
            })),
        },
    ))
}

/// What a write thread works with.
struct WriteWorker {
    owner: OwnerConnection,
    geometry: Arc<PoolGeometry>,
    writer: Arc<dyn PageWriter>,
}

impl WriteWorker {
    /// Carry out a write op: reserve pool pages for its `keys`, copy
    /// its `blocks`, the keys' Device blocks, to them, and commit them with
    /// its `checkpoints`, the State checkpoints among them.
    fn write(&mut self, write: &SharedL2OpView) -> Result<(), PoolError> {
        // No reservation: every block is in the pool, is being written by
        // another rank, or has no room.
        let Some((ticket, pages)) = self.owner.reserve(write)? else {
            return Ok(());
        };
        // A key without a page (negative) is one of those: skipped. Pages
        // start at 1: a page 0 is no reply of the owner's, and fails the
        // copy.
        let copies: Vec<PageCopy> = (0..write.keys.len())
            .filter(|&index| pages[index] >= 0)
            .map(|index| PageCopy {
                group_id: write.keys[index].group_id,
                bucket: write.buckets[index],
                pool_page: pages[index],
                device_block: write.device_blocks[index],
            })
            .collect();
        let spans_by_provider = self.geometry.write_spans(&copies)?;
        for (provider, spans) in self.geometry.providers.iter().zip(&spans_by_provider) {
            if spans.is_empty() {
                continue;
            }
            self.writer
                .write(&provider.session, spans)
                .map_err(|error| {
                    PoolError(format!(
                        "shared L2 write to {} failed: {error}",
                        provider.session
                    ))
                })?;
        }
        self.owner.commit(ticket, &write.checkpoints)
    }
}

impl PoolClient for RankClient {
    fn lookup(
        &mut self,
        lookups: &[SharedL2LookupView],
    ) -> Result<Vec<SharedL2AnswerView>, PoolError> {
        let answers = self.owner.lookup(lookups)?;
        // A load reads a hit's pages at their addresses: each has one.
        for answer in &answers {
            for ((key, bucket), page) in answer
                .keys
                .iter()
                .zip(&answer.buckets)
                .zip(&answer.pool_pages)
            {
                self.geometry.page_address(key.group_id, *bucket, *page)?;
            }
        }
        Ok(answers)
    }

    fn poll_writes(&mut self) -> Result<Vec<u32>, PoolError> {
        self.failure.check()?;
        Ok(self.finished_writes.try_iter().collect())
    }

    fn map_proxy_blocks(&mut self, proxy_blocks: &[SharedL2ProxyBlockView]) {
        self.proxy_pages.record(proxy_blocks);
    }

    fn submit(&mut self, ops: &[SharedL2OpView]) -> Result<(), PoolError> {
        let stopped = || PoolError("the shared L2 client's threads have stopped".to_owned());
        // Each kind goes out in plan order.
        for op in ops {
            match op.kind {
                SharedL2OpKind::Write => {
                    self.writes.send(Some(op.clone())).map_err(|_| stopped())?
                }
                SharedL2OpKind::Release => self
                    .notices
                    .send(Some(Notice::Release(op.ticket)))
                    .map_err(|_| stopped())?,
                SharedL2OpKind::Touch => self
                    .notices
                    .send(Some(Notice::Touch(op.keys.clone())))
                    .map_err(|_| stopped())?,
                kind => {
                    return Err(PoolError(format!(
                        "no client call for the shared L2 op {kind:?}"
                    )))
                }
            }
        }
        Ok(())
    }

    fn check_failure(&self) -> Result<(), PoolError> {
        self.failure.check()
    }
}

impl RankThreads {
    /// The rank's loads.
    pub fn loads(&self) -> &Loads {
        &self.loads
    }

    /// Stop the client's threads and the loads once what is queued ran,
    /// wait up to `wait` for them, and then release what they read and
    /// wrote through, on the calling thread: the transfer engine stops
    /// there, after its last user. Returns whether every thread ended. One
    /// that still waits for the owner or for a copy keeps its socket, the
    /// transfer engine and the registered memory in use: then nothing is
    /// released, and all of it stays as it is for the process's exit to
    /// take.
    pub fn close(&self, wait: Duration) -> bool {
        for _ in 0..self.write_workers {
            let _ = self.writes.send(None);
        }
        let _ = self.notices.send(None);
        let mut threads = self.threads.lock();
        threads.extend(self.loads.stop());
        let deadline = Instant::now() + wait;
        while !threads.iter().all(thread::JoinHandle::is_finished) {
            if Instant::now() >= deadline {
                let running: Vec<_> = threads
                    .iter()
                    .filter(|thread| !thread.is_finished())
                    .filter_map(|thread| thread.thread().name().map(str::to_owned))
                    .collect();
                tracing::warn!(
                    ?running,
                    ?wait,
                    "shared L2 client threads did not stop; the transfer engine and the \
                     owner's sockets stay open"
                );
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
        for thread in threads.drain(..) {
            // A thread that panicked has ended too.
            let _ = thread.join();
        }
        // Past every thread's own reference: the last ones.
        drop(self.transfer.lock().take());
        true
    }
}

#[cfg(test)]
#[path = "../../tests/unit/shared_l2_client.rs"]
mod tests;
