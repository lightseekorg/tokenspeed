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

//! Progress the same shard FSM while the scheduler waits for device work.
//! The foreground holds the scheduler mutex when entering and leaving a window.
//! A worker can change cache ownership only after every member locked its own
//! scheduler at the same point in the same plan.
use super::{
    coordinated::Control, read::Reads, Group, Mesh, Mutex, PeerError, PeerRound, Scheduler,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub(crate) enum CacheWait {
    Execute = 1,
    DataParallel = 2,
    Complete = 3,
}

type Point = [i64; 3];

pub(crate) struct Window<'a>(&'a Mutex<Option<Point>>);
impl Drop for Window<'_> {
    fn drop(&mut self) {
        *self.0.lock() = None;
    }
}

pub(super) struct Progress {
    point: Arc<Mutex<Option<Point>>>,
    stop: Arc<AtomicBool>,
    interrupt: Option<ts_collective::Interrupt>,
    thread: Option<thread::JoinHandle<()>>,
    results: mpsc::Receiver<Result<PeerRound, PeerError>>,
}

impl Progress {
    pub fn start(
        mut mesh: Option<(Mesh, Group)>,
        control: Arc<Mutex<Control>>,
        scheduler: Arc<Mutex<Scheduler>>,
        reads: Option<Arc<Mutex<Reads>>>,
        origin: Instant,
    ) -> Result<Self, PeerError> {
        let width = control.lock().layout.members.len();
        if mesh.as_ref().map_or(1, |(_, group)| group.size()) != width {
            return Err(PeerError::Topology);
        }
        let leader = mesh.as_ref().is_none_or(|(_, group)| group.index() == 0);
        let interrupt = mesh
            .as_ref()
            .map(|(m, _)| m.interrupt_handle())
            .transpose()?;
        let stop = Arc::new(AtomicBool::new(false));
        let point = Arc::new(Mutex::new(None::<Point>));
        let (tx, results) = mpsc::channel();
        let point_worker = point.clone();
        let stopped = stop.clone();
        let wake = control.lock().wake();
        let thread = thread::Builder::new()
            .name("ts-peer-shards".into())
            .spawn(move || {
                if let Some(w) = wake {
                    w.0.set(thread::current())
                        .expect("one registered peer worker");
                }
                let mut run = || -> Result<(), PeerError> {
                    while !stopped.load(Ordering::Acquire) {
                        if leader {
                            thread::park_timeout(Duration::from_millis(1));
                            if stopped.load(Ordering::Acquire) {
                                break;
                            }
                            if point_worker.lock().is_none() || !control.lock().has_work() {
                                thread::park();
                                continue;
                            }
                        }
                        // Followers sleep in the broadcast without holding cache or protocol locks.
                        if let Some((m, group)) = mesh.as_mut() {
                            m.broadcast(group, 0, Vec::new())?;
                        }
                        let mut cache = scheduler.try_lock();
                        let point = cache.as_ref().and_then(|_| *point_worker.lock());
                        let mut row = point.unwrap_or_default().to_vec();
                        row.push(origin.elapsed().as_micros() as i64);
                        let now = if let Some((m, group)) = mesh.as_mut() {
                            let rows = m.all_gather_i64(group, &row)?;
                            if point.is_none() || !rows.iter().all(|r| r[..3] == row[..3]) {
                                continue;
                            }
                            rows[0][3]
                        } else if point.is_some() {
                            row[3]
                        } else {
                            continue;
                        };
                        let mut c = control.lock();
                        let report = c.report(Vec::new());
                        let reports = if let Some((m, g)) = mesh.as_mut() {
                            m.all_gather(g, &report)?
                        } else {
                            vec![report]
                        };
                        let (_, reports) = Control::unpack(&reports)?;
                        let mut round = PeerRound {
                            now_us: now,
                            ..PeerRound::default()
                        };
                        let reads = reads.as_ref().map(|r| r.lock());
                        c.advance(
                            reports,
                            cache.as_deref_mut().unwrap(),
                            reads.as_deref(),
                            &mut round,
                        )?;
                        if (!round.events.is_empty() || !round.landed.is_empty())
                            && tx.send(Ok(round)).is_err()
                        {
                            break;
                        }
                    }
                    Ok(())
                };
                if let Err(error) = run() {
                    if !stopped.load(Ordering::Acquire) {
                        let _ = tx.send(Err(error));
                    }
                }
            })?;
        Ok(Self {
            point,
            stop,
            interrupt,
            thread: Some(thread),
            results,
        })
    }

    pub fn open(&self, epoch: u64, phase: CacheWait, forward: u64) -> Window<'_> {
        *self.point.lock() = Some([epoch as i64, phase as i64, forward as i64]);
        if let Some(t) = &self.thread {
            t.thread().unpark();
        }
        Window(&self.point)
    }

    pub fn drain(&self, round: &mut PeerRound) -> Result<(), PeerError> {
        loop {
            match self.results.try_recv() {
                Ok(Ok(result)) => {
                    round.decisions.extend(result.decisions);
                    round.events.extend(result.events);
                    round.landed.extend(result.landed);
                }
                Ok(Err(e)) => return Err(e),
                Err(mpsc::TryRecvError::Empty) => return Ok(()),
                Err(mpsc::TryRecvError::Disconnected) => return Err(PeerError::WorkerExited),
            }
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(i) = &self.interrupt {
            i.shutdown();
        }
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}
