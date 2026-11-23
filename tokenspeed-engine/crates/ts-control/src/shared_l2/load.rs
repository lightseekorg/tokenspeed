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

//! A rank's loads from a shared L2 pool: pool pages read into Device blocks
//! layer by layer, like a Host L2 load.
//!
//! The scheduler's Host tier is a proxy of the pool, so a load-back names
//! proxy Host blocks, each standing for a pool page ([`ProxyPages`]). A load
//! has signal words ([`SignalWords`]): a start word, which the forward
//! thread's stream writes once the plan's page zeroing is done, and a ready
//! word per layer, which the layer's stream waits on before it reads its
//! cache. The forward thread hands a plan's load over ([`Loads::submit`])
//! before it puts any of those on a stream (`cache/shared_l2/executor.py`):
//! a load that is refused here then leaves no stream waiting on a word
//! nobody sets, and the loader reads nothing before the start word anyway.
//! One loader thread runs the loads in the order they came: it waits for
//! the start word, reads a layer's fields from every provider in parallel,
//! and sets the layer's ready word once all of them are there.
//!
//! The loader makes no device call: while the device waits on a ready word,
//! the forward thread may be inside a device call that holds the driver
//! until the device drains.

use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Duration,
};

use parking_lot::Mutex;
use ts_scheduler::SharedL2ProxyBlockView;

use crate::{
    peer::{geometry::Spans, read::PageReader},
    shared_l2::{
        geometry::{PageCopy, PoolGeometry},
        panic_as_failure, FailureLatch, PoolError,
    },
};

/// How often the loader looks at a load's start word.
const START_WORD_POLL: Duration = Duration::from_micros(20);

/// The signal words of a rank's loads, a start word and a ready word per
/// layer for each load in flight: 32-bit words in host memory the device
/// maps (`HostSignals` in `cache/shared_l2/executor.py`). A stream waits on
/// a word or writes it with a stream memory operation; the loader reads and
/// sets it as plain memory.
pub trait SignalWords: Send + Sync {
    /// How many words there are.
    fn num_words(&self) -> usize;
    /// The value of `word`.
    fn get(&self, word: usize) -> u32;
    /// Set `word` to `value`.
    fn set(&self, word: usize, value: u32);
}

/// Whether a word holds load `seq`'s value or a later load's.
fn reached(value: u32, seq: u32) -> bool {
    value.wrapping_sub(seq) < 0x8000_0000
}

/// A proxy Host block of a cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ProxyBlock {
    group_id: u32,
    host_block: i32,
}

/// The pool page a proxy Host block stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PoolPage {
    bucket: i32,
    pool_page: i32,
}

/// The pool page behind each proxy Host block. The control thread records
/// the pages of an answer before the plan that may load from them; the
/// forward thread reads them when it hands that plan's load over.
#[derive(Default)]
pub struct ProxyPages(Mutex<HashMap<ProxyBlock, PoolPage>>);

impl ProxyPages {
    /// Record the pool page each proxy block of an answer stands for. A
    /// proxy block the scheduler uses again stands for its new page.
    pub fn record(&self, proxy_blocks: &[SharedL2ProxyBlockView]) {
        let mut proxy_pages = self.0.lock();
        for proxy_block in proxy_blocks {
            proxy_pages.insert(
                ProxyBlock {
                    group_id: proxy_block.group_id,
                    host_block: proxy_block.host_block,
                },
                PoolPage {
                    bucket: proxy_block.bucket,
                    pool_page: proxy_block.pool_page,
                },
            );
        }
    }

    /// The pool page a proxy Host block stands for.
    fn page(&self, proxy_block: ProxyBlock) -> Option<PoolPage> {
        self.0.lock().get(&proxy_block).copied()
    }
}

/// One block of a plan's load: the proxy Host block whose pool page is
/// read, and the Device block it fills.
#[derive(Debug, Clone, Copy)]
pub struct LoadRow {
    /// The cache group, by index.
    pub group_id: u32,
    /// The proxy Host block.
    pub host_block: i32,
    /// The Device block.
    pub device_block: i32,
}

/// One plan's load, its proxy blocks resolved to pool pages.
struct Job {
    /// The load-back operations the load carries out, acknowledged together.
    op_ids: Vec<u32>,
    /// Every block to load.
    copies: Vec<PageCopy>,
    /// The load's start word; its layers' ready words follow it.
    start_word: usize,
    /// The value that marks this load in its words.
    seq: u32,
}

/// One provider's share of a layer, for a reader thread.
struct Read {
    session: String,
    spans: Spans,
    done: mpsc::Sender<Result<(), String>>,
}

/// The rank's loads: plans' loads in, the finished operations out.
pub struct Loads {
    geometry: Arc<PoolGeometry>,
    proxy_pages: Arc<ProxyPages>,
    failure: Arc<FailureLatch>,
    signal_words: Arc<dyn SignalWords>,
    /// `None` once stopped.
    jobs: Mutex<Option<mpsc::Sender<Job>>>,
    finished_loads: Mutex<mpsc::Receiver<Vec<u32>>>,
    stopped: Arc<AtomicBool>,
    threads: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Loads {
    /// Start the loader and `read_workers` readers, which read one layer's
    /// providers in parallel over `reader`. A failed load is recorded in
    /// `failure`.
    pub fn start(
        geometry: Arc<PoolGeometry>,
        proxy_pages: Arc<ProxyPages>,
        reader: Arc<dyn PageReader>,
        signal_words: Arc<dyn SignalWords>,
        failure: Arc<FailureLatch>,
        read_workers: NonZeroUsize,
    ) -> std::io::Result<Self> {
        let (jobs, incoming) = mpsc::channel::<Job>();
        let (reads, incoming_reads) = mpsc::channel::<Read>();
        let (finished, finished_loads) = mpsc::channel();
        let stopped = Arc::new(AtomicBool::new(false));
        let incoming_reads = Arc::new(Mutex::new(incoming_reads));
        // Own each thread as soon as it starts: a later spawn failure drops
        // the senders, which ends the ones that already started.
        let mut threads = Vec::with_capacity(1 + read_workers.get());
        for index in 0..read_workers.get() {
            let (reader, incoming_reads) = (reader.clone(), incoming_reads.clone());
            threads.push(
                thread::Builder::new()
                    .name(format!("ts-shared-l2-read-{index}"))
                    .spawn(move || loop {
                        // Release the queue lock before the blocking read.
                        let read = incoming_reads.lock().recv();
                        let Ok(read) = read else {
                            return;
                        };
                        let outcome = reader.read(&read.session, &read.spans).map_err(|error| {
                            format!("shared L2 read from {} failed: {error}", read.session)
                        });
                        // The loader waits for every read it queued.
                        let _ = read.done.send(outcome);
                    })?,
            );
        }
        let loader = Loader {
            geometry: geometry.clone(),
            signal_words: signal_words.clone(),
            failure: failure.clone(),
            stopped: stopped.clone(),
            reads,
            finished,
        };
        threads.push(
            thread::Builder::new()
                .name("ts-shared-l2-load".to_owned())
                .spawn(move || {
                    for job in incoming {
                        loader.run(job);
                    }
                })?,
        );
        Ok(Self {
            geometry,
            proxy_pages,
            failure,
            signal_words,
            jobs: Mutex::new(Some(jobs)),
            finished_loads: Mutex::new(finished_loads),
            stopped,
            threads: Mutex::new(threads),
        })
    }

    /// Queue a plan's load; never waits. `rows` are the blocks to load,
    /// `start_word` is the load's start word, which its layers' ready words
    /// follow, and `seq` the value that marks the load in them. The proxy
    /// blocks are resolved now: one may stand for another page once its
    /// lookup's pin is released.
    pub fn submit(
        &self,
        op_ids: Vec<u32>,
        rows: &[LoadRow],
        start_word: usize,
        seq: u32,
    ) -> Result<(), PoolError> {
        // The load's words end with its last layer's ready word.
        if start_word
            .checked_add(self.geometry.num_layers)
            .is_none_or(|last_word| last_word >= self.signal_words.num_words())
        {
            return Err(PoolError(format!(
                "a shared L2 load names start word {start_word} of {} words",
                self.signal_words.num_words()
            )));
        }
        let copies = rows
            .iter()
            .map(|row| {
                let proxy_block = ProxyBlock {
                    group_id: row.group_id,
                    host_block: row.host_block,
                };
                let pool_page = self.proxy_pages.page(proxy_block).ok_or_else(|| {
                    PoolError(format!(
                        "a shared L2 load reads proxy Host block {} of group {}, which stands \
                         for no pool page",
                        row.host_block, row.group_id
                    ))
                })?;
                Ok(PageCopy {
                    group_id: row.group_id,
                    bucket: pool_page.bucket,
                    pool_page: pool_page.pool_page,
                    device_block: row.device_block,
                })
            })
            .collect::<Result<Vec<_>, PoolError>>()?;
        let job = Job {
            op_ids,
            copies,
            start_word,
            seq,
        };
        let stopped = || PoolError("the shared L2 loads have stopped".to_owned());
        let jobs = self.jobs.lock();
        jobs.as_ref()
            .ok_or_else(stopped)?
            .send(job)
            .map_err(|_| stopped())
    }

    /// The load-back operations finished since the last poll; never waits.
    /// Fails once a load, or other pool work of the rank, has failed.
    pub fn poll(&self) -> Result<Vec<u32>, PoolError> {
        self.failure.check()?;
        Ok(self.finished_loads.lock().try_iter().flatten().collect())
    }

    /// Stop taking loads and hand back the loader's threads, which end once
    /// the loads already queued ran; one still waiting for its start word
    /// fails.
    pub fn stop(&self) -> Vec<thread::JoinHandle<()>> {
        self.stopped.store(true, Ordering::Relaxed);
        self.jobs.lock().take();
        std::mem::take(&mut *self.threads.lock())
    }
}

impl Drop for Loads {
    fn drop(&mut self) {
        // The threads end by themselves; each keeps what it still uses.
        self.stop();
    }
}

/// What the loader thread works with.
struct Loader {
    geometry: Arc<PoolGeometry>,
    signal_words: Arc<dyn SignalWords>,
    failure: Arc<FailureLatch>,
    stopped: Arc<AtomicBool>,
    reads: mpsc::Sender<Read>,
    finished: mpsc::Sender<Vec<u32>>,
}

impl Loader {
    fn run(&self, job: Job) {
        match panic_as_failure(|| self.load(&job)) {
            Ok(()) => {
                let _ = self.finished.send(job.op_ids);
            }
            Err(error) => {
                tracing::error!(%error, "shared L2 load failed");
                // A failed load stops the engine, and so does one whose
                // thread panicked. The forward that waits on it is let go,
                // so that the stop is not a hang, but over pages the load
                // never filled: the failure is recorded first, and the loop
                // checks it once that forward has landed, before it commits
                // the forward's results.
                self.failure.record(error);
                for layer in 0..self.geometry.num_layers {
                    self.signal_words.set(job.start_word + 1 + layer, job.seq);
                }
            }
        }
    }

    fn load(&self, job: &Job) -> Result<(), PoolError> {
        // The start word lands once the plan's page zeroing is done.
        while !reached(self.signal_words.get(job.start_word), job.seq) {
            if self.stopped.load(Ordering::Relaxed) {
                return Err(PoolError("the rank shut down before the load".to_owned()));
            }
            thread::sleep(START_WORD_POLL);
        }
        let layers = self.geometry.load_spans(&job.copies)?;
        for (layer, spans_by_provider) in layers.into_iter().enumerate() {
            let (done, outcomes) = mpsc::channel();
            let mut queued = 0;
            for (provider, spans) in self.geometry.providers.iter().zip(spans_by_provider) {
                if spans.is_empty() {
                    continue;
                }
                let read = Read {
                    session: provider.session.clone(),
                    spans,
                    done: done.clone(),
                };
                if self.reads.send(read).is_err() {
                    return Err(PoolError("the shared L2 readers have stopped".to_owned()));
                }
                queued += 1;
            }
            drop(done);
            // Every read of the layer retires before the layer is let go or
            // the load fails: none still writes a block afterwards.
            let mut failed = None;
            for _ in 0..queued {
                let outcome = outcomes
                    .recv()
                    .unwrap_or_else(|_| Err("a shared L2 reader stopped".to_owned()));
                if let Err(error) = outcome {
                    failed.get_or_insert(error);
                }
            }
            if let Some(error) = failed {
                return Err(PoolError(error));
            }
            self.signal_words.set(job.start_word + 1 + layer, job.seq);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Condvar;

    use super::*;
    use crate::{
        peer::read::Arena,
        shared_l2::{
            geometry::{PageField, PageGroup, Provider},
            testing::{wait_until, PlainSignals},
        },
    };

    const KV: u32 = 0;
    const STATE: u32 = 1;
    /// Device block ids 1..8 and the null block.
    const BLOCKS: i32 = 9;
    const STATE_DEVICE: u64 = 1_000;
    const LAYERS: usize = 2;

    /// A rank's Device cache and a pool of 8 LCM blocks on two providers,
    /// all in one arena: a KV block's k (12 bytes, layer 0) and v (16 bytes,
    /// layer 1) fill a pool LCM block of 28 bytes; two State blocks (12
    /// bytes, layer 1) fit in one.
    fn geometry() -> Arc<PoolGeometry> {
        Arc::new(PoolGeometry {
            lcm_block_bytes: 28,
            providers: vec![
                Provider {
                    session: "p0".to_owned(),
                    base_address: 10_000,
                    num_lcm_blocks: 3,
                    first_lcm_block: 0,
                },
                Provider {
                    session: "p1".to_owned(),
                    base_address: 20_000,
                    num_lcm_blocks: 5,
                    first_lcm_block: 3,
                },
            ],
            groups: vec![
                PageGroup {
                    packing: 1,
                    page_bytes: 28,
                    fields: vec![
                        PageField {
                            device_address: 0,
                            device_stride: 32,
                            bytes: 12,
                            offset_in_page: 0,
                            layer: 0,
                        },
                        PageField {
                            device_address: 16,
                            device_stride: 32,
                            bytes: 16,
                            offset_in_page: 12,
                            layer: 1,
                        },
                    ],
                },
                PageGroup {
                    packing: 2,
                    page_bytes: 12,
                    fields: vec![PageField {
                        device_address: STATE_DEVICE,
                        device_stride: 16,
                        bytes: 12,
                        offset_in_page: 0,
                        layer: 1,
                    }],
                },
            ],
            num_layers: LAYERS,
        })
    }

    /// Pool pages of both groups on both providers, and the Device blocks a
    /// writer holds them in: `(group, page, block)`. Two of the State pages
    /// share an LCM block.
    const STORED: [(u32, i32, i32); 6] = [
        (KV, 1, 3),
        (KV, 3, 5),
        (KV, 8, 1),
        (STATE, 3, 4),
        (STATE, 4, 8),
        (STATE, 9, 2),
    ];

    /// Where a block's fields are in the Device cache, with their sizes.
    fn fields(group: u32, block: i32) -> Vec<(u64, usize)> {
        let block = block as u64;
        if group == KV {
            vec![(block * 32, 12), (block * 32 + 16, 16)]
        } else {
            vec![(STATE_DEVICE + block * 16, 12)]
        }
    }

    /// A writer fills its blocks and copies them to the pool's pages.
    fn store(arena: &Arena) {
        let geometry = geometry();
        let mut copies = Vec::new();
        for (group, page, block) in STORED {
            for (field, (address, bytes)) in fields(group, block).into_iter().enumerate() {
                let seed = (10 * page + 50 * field as i32) as u8 + group as u8;
                let payload: Vec<u8> = (0..bytes as u8).map(|byte| seed + byte).collect();
                arena.write(address, &payload);
            }
            copies.push(PageCopy {
                group_id: group,
                bucket: 0,
                pool_page: page,
                device_block: block,
            });
        }
        let writes = geometry.write_spans(&copies).unwrap();
        for (provider, spans) in geometry.providers.iter().zip(&writes) {
            // Within test memory a write is the copy a read is.
            arena.read(&provider.session, spans).unwrap();
        }
    }

    /// A reader loads the stored pages into other Device blocks: every
    /// page behind a proxy Host block of its own.
    fn loaded() -> (Vec<SharedL2ProxyBlockView>, Vec<LoadRow>) {
        STORED
            .iter()
            .enumerate()
            .map(|(index, &(group_id, page, block))| {
                let proxy = 100 + index as i32;
                (
                    SharedL2ProxyBlockView {
                        group_id,
                        host_block: proxy,
                        bucket: 0,
                        pool_page: page,
                    },
                    LoadRow {
                        group_id,
                        host_block: proxy,
                        device_block: BLOCKS - block,
                    },
                )
            })
            .unzip()
    }

    fn assert_loaded(arena: &Arena) {
        for (group, _, block) in STORED {
            for ((written, bytes), (read, _)) in fields(group, block)
                .into_iter()
                .zip(fields(group, BLOCKS - block))
            {
                assert_eq!(
                    arena.read_bytes(read, bytes),
                    arena.read_bytes(written, bytes),
                    "group {group} block {block}"
                );
            }
        }
    }

    /// The arena behind a reader that records each read's session and can
    /// fail its reads or hold those of one length.
    struct Reader {
        arena: Arena,
        sessions: Mutex<Vec<String>>,
        failing: AtomicBool,
        held_length: Mutex<Option<u64>>,
        resumed: Condvar,
    }

    impl PageReader for Reader {
        fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
            self.sessions.lock().push(session.to_owned());
            let mut held = self.held_length.lock();
            while held.is_some_and(|length| spans.lengths.contains(&length)) {
                self.resumed.wait(&mut held);
            }
            drop(held);
            if self.failing.load(Ordering::Relaxed) {
                return Err("down".to_owned());
            }
            self.arena.read(session, spans)
        }
    }

    /// A rank's loads over the pool: two word sets, as a tracker's two
    /// event sets have.
    struct Rank {
        loads: Loads,
        signal_words: Arc<PlainSignals>,
        /// Whether a failure was recorded when each word was set.
        failed_at_set: Arc<Mutex<Vec<bool>>>,
        reader: Arc<Reader>,
        failure: Arc<FailureLatch>,
        rows: Vec<LoadRow>,
    }

    impl Rank {
        fn new() -> Self {
            Self::over(|signal_words| signal_words)
        }

        /// A rank whose loader reaches its signal words through `words`.
        fn over(words: impl FnOnce(Arc<PlainSignals>) -> Arc<dyn SignalWords>) -> Self {
            let reader = Arc::new(Reader {
                arena: Arena::new(30_000),
                sessions: Mutex::default(),
                failing: AtomicBool::new(false),
                held_length: Mutex::default(),
                resumed: Condvar::new(),
            });
            store(&reader.arena);
            let failure = Arc::new(FailureLatch::default());
            let failed_at_set = Arc::new(Mutex::new(Vec::new()));
            let (failed, at_set) = (failure.clone(), failed_at_set.clone());
            // Two sets of a start word and a ready word per layer.
            let signal_words = Arc::new(PlainSignals::observed(2 * (LAYERS + 1), move || {
                at_set.lock().push(failed.check().is_err())
            }));
            let proxy_pages = Arc::new(ProxyPages::default());
            let (pages, rows) = loaded();
            proxy_pages.record(&pages);
            let loads = Loads::start(
                geometry(),
                proxy_pages,
                reader.clone(),
                words(signal_words.clone()),
                failure.clone(),
                NonZeroUsize::new(2).unwrap(),
            )
            .unwrap();
            Self {
                loads,
                signal_words,
                failed_at_set,
                reader,
                failure,
                rows,
            }
        }

        /// Hand over the load of every stored page on a word set; the start
        /// word lands with `started`.
        fn load(&self, op_id: u32, set: usize, seq: u32, started: bool) {
            let start_word = set * (LAYERS + 1);
            if started {
                self.signal_words.words[start_word].store(seq, Ordering::Relaxed);
            }
            self.loads
                .submit(vec![op_id], &self.rows, start_word, seq)
                .unwrap();
        }

        fn word(&self, word: usize) -> u32 {
            self.signal_words.get(word)
        }

        fn resume(&self) {
            *self.reader.held_length.lock() = None;
            self.reader.resumed.notify_all();
        }
    }

    fn join(threads: Vec<thread::JoinHandle<()>>) {
        wait_until(|| threads.iter().all(thread::JoinHandle::is_finished));
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn a_word_is_reached_by_its_load_and_by_later_ones() {
        assert!(reached(5, 5) && reached(6, 5) && !reached(4, 5));
        // The sequence wraps.
        assert!(reached(0, u32::MAX) && !reached(u32::MAX, 0));
    }

    #[test]
    fn a_load_restores_every_field_from_the_provider_that_holds_it() {
        let rank = Rank::new();
        let before = rank.reader.arena.read_bytes(0, 2_000);
        rank.load(7, 0, 1, true);

        let mut done = Vec::new();
        wait_until(|| {
            done.extend(rank.loads.poll().unwrap());
            !done.is_empty()
        });
        assert_eq!(done, [7]);
        assert_eq!(rank.loads.poll().unwrap(), Vec::<u32>::new());
        assert_loaded(&rank.reader.arena);
        assert_eq!((rank.word(1), rank.word(2)), (1, 1));
        // Each layer read both providers, and the first layer came first.
        let sessions = rank.reader.sessions.lock().clone();
        assert_eq!(sessions.len(), 4);
        let mut layers = [sessions[..2].to_vec(), sessions[2..].to_vec()];
        layers.iter_mut().for_each(|layer| layer.sort());
        assert_eq!(layers, [["p0", "p1"], ["p0", "p1"]]);
        // Only the loaded blocks changed, and of those only the payloads.
        let mut expected = before;
        for (group, _, block) in STORED {
            for ((written, bytes), (read, _)) in fields(group, block)
                .into_iter()
                .zip(fields(group, BLOCKS - block))
            {
                let payload = expected[written as usize..written as usize + bytes].to_vec();
                expected[read as usize..read as usize + bytes].copy_from_slice(&payload);
            }
        }
        assert_eq!(rank.reader.arena.read_bytes(0, 2_000), expected);
    }

    #[test]
    fn a_layer_is_ready_once_its_fields_are_there() {
        let rank = Rank::new();
        // The second layer's 16-byte field stays on its providers.
        *rank.reader.held_length.lock() = Some(16);
        rank.load(7, 0, 1, true);

        wait_until(|| rank.word(1) == 1);
        // The first layer's word is set with its field loaded and no other.
        let (group, _, block) = STORED[0];
        let (k, v) = (fields(group, block)[0], fields(group, block)[1]);
        let (loaded_k, loaded_v) = (
            fields(group, BLOCKS - block)[0],
            fields(group, BLOCKS - block)[1],
        );
        let arena = &rank.reader.arena;
        assert_eq!(
            arena.read_bytes(loaded_k.0, k.1),
            arena.read_bytes(k.0, k.1)
        );
        assert_ne!(
            arena.read_bytes(loaded_v.0, v.1),
            arena.read_bytes(v.0, v.1)
        );
        assert_eq!(rank.word(2), 0);
        assert_eq!(rank.loads.poll().unwrap(), Vec::<u32>::new());

        rank.resume();
        wait_until(|| rank.word(2) == 1);
        assert_loaded(arena);
    }

    #[test]
    fn a_load_starts_once_its_start_word_landed() {
        let rank = Rank::new();
        rank.load(7, 0, 1, false);
        // The plan's page zeroing may still be running: nothing is read.
        thread::sleep(Duration::from_millis(50));
        assert!(rank.reader.sessions.lock().is_empty());
        assert_eq!(rank.word(1), 0);

        rank.signal_words.words[0].store(1, Ordering::Relaxed);
        wait_until(|| rank.word(2) == 1);
        assert_loaded(&rank.reader.arena);
    }

    #[test]
    fn consecutive_loads_run_in_the_order_they_came() {
        let rank = Rank::new();
        // The second load's start word lands first: it still runs second.
        rank.load(1, 0, 1, false);
        rank.load(2, 1, 2, true);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(rank.word(LAYERS + 1 + 1), 0);
        rank.signal_words.words[0].store(1, Ordering::Relaxed);
        for (op_id, seq) in [(3, 3), (4, 4), (5, 5)] {
            rank.load(op_id, op_id as usize % 2, seq, true);
        }

        let mut done = Vec::new();
        wait_until(|| {
            done.extend(rank.loads.poll().unwrap());
            done.len() == 5
        });
        assert_eq!(done, [1, 2, 3, 4, 5]);
        // Each word set holds its last load's value.
        assert_eq!((rank.word(1), rank.word(2)), (4, 4));
        assert_eq!((rank.word(4), rank.word(5)), (5, 5));
    }

    #[test]
    fn a_failed_read_is_recorded_before_the_forward_is_let_go() {
        let rank = Rank::new();
        rank.reader.failing.store(true, Ordering::Relaxed);
        rank.load(7, 0, 1, true);

        // The forward is let go: every layer's word is set.
        wait_until(|| rank.word(1) == 1 && rank.word(2) == 1);
        // The failure was there before any of them.
        assert_eq!(*rank.failed_at_set.lock(), [true, true]);
        let error = rank.failure.check().unwrap_err();
        assert!(error.0.contains("shared L2 read from p"), "{error}");
        assert!(error.0.contains("down"), "{error}");
        // The load is not reported, and it stays failed: the engine is to
        // stop.
        for _ in 0..2 {
            assert_eq!(rank.loads.poll().unwrap_err().0, error.0);
        }
    }

    /// Signal words whose start words cannot be read: the load thread
    /// panics where it waits for one.
    struct Unreadable(Arc<PlainSignals>);

    impl SignalWords for Unreadable {
        fn num_words(&self) -> usize {
            self.0.num_words()
        }

        fn get(&self, _: usize) -> u32 {
            panic!("the start word is unreadable")
        }

        fn set(&self, word: usize, value: u32) {
            self.0.set(word, value);
        }
    }

    #[test]
    fn a_panic_of_the_load_thread_is_a_failed_load() {
        let rank = Rank::over(|signal_words| Arc::new(Unreadable(signal_words)));
        rank.load(7, 0, 1, true);
        // A load queued behind the one that panics.
        rank.load(8, 1, 2, true);

        // The forwards of both are let go: every layer's word is set.
        wait_until(|| (1..=2).all(|word| rank.word(word) == 1));
        wait_until(|| (4..=5).all(|word| rank.word(word) == 2));
        // The failure was there before any of them, and it names the thread
        // and the panic.
        assert_eq!(*rank.failed_at_set.lock(), [true; 4]);
        let error = rank.loads.poll().unwrap_err();
        assert_eq!(
            error.0,
            "the shared L2 thread ts-shared-l2-load panicked: the start word is unreadable"
        );
    }

    #[test]
    fn a_load_of_a_block_or_of_words_the_rank_does_not_have_is_refused() {
        let rank = Rank::new();
        let submit = |rows: &[LoadRow], start_word| rank.loads.submit(vec![7], rows, start_word, 1);
        let unmapped = LoadRow {
            group_id: KV,
            host_block: 99,
            device_block: 1,
        };
        let error = submit(&[unmapped], 0).unwrap_err();
        assert!(error.0.contains("stands for no pool page"), "{error}");
        // The second word set ends at word 5.
        let error = submit(&[], 4).unwrap_err();
        assert!(error.0.contains("start word 4 of 6 words"), "{error}");
        let error = submit(&[], usize::MAX).unwrap_err();
        assert!(error.0.contains("of 6 words"), "{error}");
        submit(&[], 3).unwrap();
    }

    #[test]
    fn a_proxy_block_used_again_stands_for_its_new_page() {
        let proxy_pages = ProxyPages::default();
        let page = |group_id, block, page| SharedL2ProxyBlockView {
            group_id,
            host_block: block,
            bucket: 0,
            pool_page: page,
        };
        proxy_pages.record(&[page(0, 11, 3), page(1, 12, 9)]);
        let stands_for = |group_id, host_block| {
            proxy_pages
                .page(ProxyBlock {
                    group_id,
                    host_block,
                })
                .map(|pool_page| pool_page.pool_page)
        };
        assert_eq!(stands_for(0, 11), Some(3));
        assert_eq!(stands_for(1, 12), Some(9));
        proxy_pages.record(&[page(0, 11, 4)]);
        assert_eq!(stands_for(0, 11), Some(4));
        assert_eq!(stands_for(0, 99), None);
    }

    #[test]
    fn a_stop_ends_the_load_in_flight_and_fails_one_that_never_started() {
        let rank = Rank::new();
        *rank.reader.held_length.lock() = Some(16);
        rank.load(7, 0, 1, true);
        // The second load's start word never lands.
        rank.load(8, 1, 2, false);
        wait_until(|| rank.word(1) == 1);

        let threads = rank.loads.stop();
        assert_eq!(threads.len(), 3);
        let error = rank.loads.submit(vec![9], &[], 0, 3).unwrap_err();
        assert!(error.0.contains("have stopped"), "{error}");
        rank.resume();
        join(threads);

        // The load in flight ended; the one that never started failed, and
        // its forward is let go.
        assert_loaded(&rank.reader.arena);
        assert_eq!((rank.word(4), rank.word(5)), (2, 2));
        let error = rank.failure.check().unwrap_err();
        assert!(error.0.contains("shut down before the load"), "{error}");
    }
}
