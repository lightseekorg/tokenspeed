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

//! The data plane: the Python `HostDataPlane` (`engine/rust_host.py`),
//! driven from two threads.
//!
//! The forward thread runs every operation that issues or orders device
//! work or changes the data plane's state, one at a time in the order the
//! control thread queued them: a request is admitted before a plan names
//! it, and a round's plan is launched before the previous forward is
//! committed. It takes the GIL per operation and waits for the next one
//! without it.
//!
//! The control thread runs the completion queries itself: a launched
//! forward's results (it waits on the forward's copy event, which releases
//! the GIL) and the cache acknowledgments. Each takes the GIL for its call
//! only, under a dispatch guard that rejects device work, and none waits
//! behind the forward thread's queue: the round head never waits for
//! launches in flight. The L3 tier's storage executor is the host's own
//! (`ts-l3`), outside the data plane.

use std::{
    collections::HashMap,
    sync::{mpsc, Arc, Condvar, Mutex, MutexGuard},
    thread,
};

use bytes::Bytes;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyList, PyTuple},
};
use ts_control::eplb::SlotMoves;
use ts_control::{
    event_loop::{DataPlane, DataPlaneError, Dispatch, DpRound},
    ingress::Admission,
    output::{ForwardResult, HistorySeeds, LogprobPlan, NgramWindows, TopLogprobs},
};
use ts_pd::LandedRequest;
use ts_scheduler::{CacheEvent, CacheEventKind, ExecutionPlan};
use ts_wire::{control::ProfileReq, ControlRequest, Wire};

use crate::pd::Streams;

/// A forward the forward thread launched, by launch order.
pub type ForwardId = u64;

/// One round's plan and what its forward needs besides.
struct Execution {
    plan: ExecutionPlan,
    forward: Option<ForwardId>,
    dp: Option<DpRound>,
    ngram: Option<NgramWindows>,
    grammar_advance: Option<Vec<bool>>,
    logprobs: Option<LogprobPlan>,
    history: Option<HistorySeeds>,
}

/// EPD: one landed item as the data plane's `epd_publish` takes it: the
/// item's index, its slot or None, the first row received, how many, the
/// image's rows, and whether they are this rank's shard.
type PyLandedItem = (usize, Option<usize>, i64, i64, i64, bool);

/// An operation for the forward thread.
enum Op {
    Admit(Vec<AdmitRow>),
    Execute(Box<Execution>),
    AttachGrammars(Vec<(String, u64, Vec<i32>)>),
    /// A committed forward, for the profiler's step count.
    Committed(Py<PyAny>),
    /// Answers once everything queued before it ran.
    Barrier {
        reply: mpsc::SyncSender<Result<(), String>>,
    },
    Release(Vec<String>),
    Land {
        spec_candidates: Option<(i32, Vec<i32>)>,
        cache_slot: Option<i32>,
    },
    IdleForward {
        dp: DpRound,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
    /// A weight operation or profile request, as its msgpack control struct.
    UpdateWeights {
        payload: Bytes,
        reply: mpsc::SyncSender<Result<(bool, String), String>>,
    },
    Profile {
        payload: Bytes,
        reply: mpsc::SyncSender<Result<(bool, String), String>>,
    },
    ReleaseMemory {
        tags: Vec<String>,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
    ResumeMemory {
        tags: Vec<String>,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
    SnapshotExpertLoad {
        reply: mpsc::SyncSender<Result<ExpertLoad, String>>,
    },
    /// EPD: landed embeddings to publish onto their items.
    EpdPublish(Vec<LandedRequest>),
    ApplyExpertPlacement {
        layer_ids: Vec<usize>,
        rows: Vec<Vec<i32>>,
        moves: Vec<SlotMoves>,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
}

/// Route counters `[layers, physical]` and the placement they were counted
/// under.
type ExpertLoad = (Vec<Vec<i64>>, Vec<Vec<i32>>);

struct AdmitRow {
    rid: String,
    sampling_params: Bytes,
    stop_token_ids: Vec<i32>,
    token_limit: i64,
    multimodal: Option<Vec<Bytes>>,
}

/// What the two threads share. The lock is a leaf: nobody takes the GIL
/// while holding it.
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    /// Signalled when a forward was launched or the data plane failed.
    changed: Condvar,
}

#[derive(Default)]
struct State {
    /// The first failure on either thread; every later call reports it.
    failure: Option<String>,
    /// Launched forwards, until their commit takes them.
    launched: HashMap<ForwardId, Py<PyAny>>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn failure(&self) -> Option<String> {
        self.state().failure.clone()
    }

    /// Record a failure, unless an earlier one is recorded, and wake the
    /// control thread if it waits for a launch.
    fn fail(&self, failure: String) {
        self.state().failure.get_or_insert(failure);
        self.changed.notify_all();
    }

    fn launch(&self, forward: ForwardId, launched: Py<PyAny>) {
        self.state().launched.insert(forward, launched);
        self.changed.notify_all();
    }

    /// The launched forward `forward`, once the forward thread launched it.
    fn take_launched(&self, forward: ForwardId) -> Result<Py<PyAny>, String> {
        let mut state = self.state();
        loop {
            if let Some(failure) = &state.failure {
                return Err(failure.clone());
            }
            if let Some(launched) = state.launched.remove(&forward) {
                return Ok(launched);
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

/// Fails the data plane when the forward thread ends, so the control thread
/// never waits for a launch that cannot come (a panic included).
struct ExitNotice(Arc<Shared>);

impl Drop for ExitNotice {
    fn drop(&mut self) {
        self.0.fail("the forward thread exited".to_owned());
    }
}

/// The data plane: the forward thread and the control thread's queries.
pub struct PythonDataPlane {
    /// The Python `HostDataPlane`.
    plane: Py<PyAny>,
    /// The control thread's dispatch guard, entered while the plane serves.
    guard: Option<Py<PyAny>>,
    ops: Option<mpsc::Sender<Op>>,
    shared: Arc<Shared>,
    next_forward: ForwardId,
    thread: Option<thread::JoinHandle<()>>,
}

impl PythonDataPlane {
    /// Start the forward thread over a Python `HostDataPlane`, and enter the
    /// dispatch guard on the calling thread, the control thread, which must
    /// also drop the data plane. With PD, the forward thread stages each
    /// plan's remote stream with the transfer peer (`streams`).
    pub fn start(plane: Py<PyAny>, streams: Option<Streams>) -> Result<Self, String> {
        let guard = Python::attach(|py| {
            plane
                .call_method0(py, "control_thread_guard")
                .and_then(|guard| guard.call_method0(py, "__enter__").map(|_| guard))
                .map_err(|error| describe(py, &error))
        })?;
        let (ops, queue) = mpsc::channel();
        let shared = Arc::new(Shared::default());
        let mut data_plane = Self {
            plane: Python::attach(|py| plane.clone_ref(py)),
            guard: Some(guard),
            ops: Some(ops),
            shared: shared.clone(),
            next_forward: 0,
            thread: None,
        };
        let thread = thread::Builder::new()
            .name("ts-forward".to_owned())
            .spawn(move || {
                let _exit = ExitNotice(shared.clone());
                Python::attach(|py| {
                    let plane = plane.bind(py);
                    if let Err(error) = plane.call_method0("enter_forward_thread") {
                        let error = describe(py, &error);
                        tracing::error!("the forward thread could not start:\n{error}");
                        shared.fail(error);
                        return;
                    }
                    forward_loop(py, plane, queue, &shared, streams.as_ref());
                })
            })
            .map_err(|error| format!("starting the forward thread: {error}"))?;
        data_plane.thread = Some(thread);
        Ok(data_plane)
    }

    fn failed(&self) -> Result<(), DataPlaneError> {
        match self.shared.failure() {
            Some(failure) => Err(DataPlaneError(failure)),
            None => Ok(()),
        }
    }

    fn queue(&self, op: Op) -> Result<(), DataPlaneError> {
        self.failed()?;
        self.ops
            .as_ref()
            .expect("open until drop")
            .send(op)
            .map_err(|_| DataPlaneError("the forward thread exited".to_owned()))
    }

    /// Queue an operation the forward thread answers, and wait for the
    /// answer.
    fn ask<T>(
        &self,
        op: impl FnOnce(mpsc::SyncSender<Result<T, String>>) -> Op,
    ) -> Result<T, DataPlaneError> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.queue(op(reply))?;
        match answer.recv() {
            Ok(answer) => answer.map_err(DataPlaneError),
            Err(_) => {
                self.failed()?;
                Err(DataPlaneError("the forward thread exited".to_owned()))
            }
        }
    }

    /// Run a completion query on this, the control thread; its failure is
    /// the data plane's.
    fn query<T>(
        &self,
        call: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>,
    ) -> Result<T, DataPlaneError> {
        self.failed()?;
        Python::attach(|py| call(self.plane.bind(py)).map_err(|error| describe(py, &error)))
            .map_err(|failure| self.fail(failure))
    }

    fn fail(&self, failure: String) -> DataPlaneError {
        tracing::error!("the data plane failed:\n{failure}");
        self.shared.fail(failure.clone());
        DataPlaneError(failure)
    }
}

/// A control struct as one msgpack frame.
fn control_payload<T: Wire>(value: &T) -> Result<Bytes, DataPlaneError> {
    let frames = ts_wire::encode_frames(value);
    let [payload] = <[Bytes; 1]>::try_from(frames).map_err(|frames| {
        DataPlaneError(format!(
            "a control operation encodes to {} frames",
            frames.len()
        ))
    })?;
    Ok(payload)
}

impl DataPlane for PythonDataPlane {
    type Pending = ForwardId;

    fn admit(&mut self, requests: &[Admission]) -> Result<(), DataPlaneError> {
        let mut rows = Vec::with_capacity(requests.len());
        for request in requests {
            let frames = ts_wire::encode_frames(&request.sampling_params);
            let [sampling_params] = <[Bytes; 1]>::try_from(frames).map_err(|frames| {
                DataPlaneError(format!(
                    "the sampling parameters of {} encode to {} frames",
                    request.rid,
                    frames.len()
                ))
            })?;
            let stop = request.state.stop_criteria();
            rows.push(AdmitRow {
                rid: request.rid.clone(),
                sampling_params,
                stop_token_ids: stop.stop_token_ids,
                token_limit: stop.token_limit,
                multimodal: request.multimodal.clone(),
            });
        }
        self.queue(Op::Admit(rows))
    }

    fn execute(
        &mut self,
        plan: ExecutionPlan,
        dispatch: &Dispatch<'_>,
    ) -> Result<Option<ForwardId>, DataPlaneError> {
        // The forward thread launches a forward exactly when the plan has a
        // batch and the rank does not idle, by the same rule the loop chose
        // `forward` with.
        let forward = dispatch.forward.map(|_| {
            self.next_forward += 1;
            self.next_forward
        });
        let idle_forward = dispatch.dp.is_some_and(|dp| dp.idle_forward);
        self.queue(Op::Execute(Box::new(Execution {
            plan,
            forward,
            dp: dispatch.dp.cloned(),
            ngram: dispatch.ngram.cloned(),
            grammar_advance: dispatch.grammar_advance.map(<[bool]>::to_vec),
            logprobs: dispatch.logprobs.cloned(),
            history: dispatch.history.cloned(),
        })))?;
        if idle_forward {
            // An idle forward leaves nothing to commit, so no later call
            // would wait for it; the round does. A rank with nothing of its
            // own then cannot run ahead of its device.
            self.ask(|reply| Op::Barrier { reply })?;
        }
        Ok(forward)
    }

    fn complete(&mut self, forward: ForwardId) -> Result<ForwardResult, DataPlaneError> {
        let launched = self.shared.take_launched(forward).map_err(DataPlaneError)?;
        let result = self.query(|plane| complete(plane, &launched))?;
        // The profiler counts committed forwards, on the thread it traces.
        self.queue(Op::Committed(launched))?;
        Ok(result)
    }

    fn poll_cache_events(&mut self) -> Result<Vec<CacheEvent>, DataPlaneError> {
        let polled: Vec<(String, u32)> = self.query(|plane| {
            plane
                .call_method0("poll_cache_events")
                .and_then(|events| events.extract())
        })?;
        polled
            .into_iter()
            .map(|(kind, op_id)| {
                let kind = match kind.as_str() {
                    "WriteBackDoneEvent" => CacheEventKind::WriteBackDone,
                    "LoadBackDoneEvent" => CacheEventKind::LoadBackDone,
                    "SnapshotDoneEvent" => CacheEventKind::SnapshotDone,
                    "RestoreDoneEvent" => CacheEventKind::RestoreDone,
                    _ => return Err(self.fail(format!("unknown cache event {kind}"))),
                };
                Ok(CacheEvent { kind, op_id })
            })
            .collect()
    }

    fn release(&mut self, rids: &[String]) -> Result<(), DataPlaneError> {
        self.queue(Op::Release(rids.to_vec()))
    }

    fn land(
        &mut self,
        spec_candidates: Option<(i32, Vec<i32>)>,
        cache_slot: Option<i32>,
    ) -> Result<(), DataPlaneError> {
        self.queue(Op::Land {
            spec_candidates,
            cache_slot,
        })
    }

    fn idle_forward(&mut self, dp: &DpRound) -> Result<(), DataPlaneError> {
        let dp = dp.clone();
        self.ask(|reply| Op::IdleForward { dp, reply })
    }

    fn update_weights(&mut self, op: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        let payload = control_payload(op)?;
        self.ask(|reply| Op::UpdateWeights { payload, reply })
    }

    fn profile(&mut self, request: &ProfileReq) -> Result<(bool, String), DataPlaneError> {
        let payload = control_payload(&ControlRequest::Profile(request.clone()))?;
        self.ask(|reply| Op::Profile { payload, reply })
    }

    fn release_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError> {
        let tags = tags.to_vec();
        self.ask(|reply| Op::ReleaseMemory { tags, reply })
    }

    fn resume_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError> {
        let tags = tags.to_vec();
        self.ask(|reply| Op::ResumeMemory { tags, reply })
    }

    fn attach_grammars(
        &mut self,
        grammars: &[(String, u64, Vec<i32>)],
    ) -> Result<(), DataPlaneError> {
        self.queue(Op::AttachGrammars(grammars.to_vec()))
    }

    fn epd_publish(&mut self, landed: &[LandedRequest]) -> Result<(), DataPlaneError> {
        self.queue(Op::EpdPublish(landed.to_vec()))
    }

    /// A completion query: the copies' events, read on this thread.
    fn epd_released(&mut self) -> Result<Vec<usize>, DataPlaneError> {
        self.query(|plane| {
            plane
                .call_method0("epd_released")
                .and_then(|slots| slots.extract())
        })
    }

    fn snapshot_expert_load(&mut self) -> Result<ExpertLoad, DataPlaneError> {
        self.ask(|reply| Op::SnapshotExpertLoad { reply })
    }

    fn apply_expert_placement(
        &mut self,
        layer_ids: &[usize],
        rows: &[Vec<i32>],
        moves: &[SlotMoves],
    ) -> Result<(), DataPlaneError> {
        let (layer_ids, rows, moves) = (layer_ids.to_vec(), rows.to_vec(), moves.to_vec());
        self.ask(|reply| Op::ApplyExpertPlacement {
            layer_ids,
            rows,
            moves,
            reply,
        })
    }
}

impl Drop for PythonDataPlane {
    fn drop(&mut self) {
        // Closing the queue ends the forward thread once it ran what is
        // queued.
        self.ops.take();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("the forward thread panicked");
            }
        }
        let guard = self.guard.take();
        let launched = std::mem::take(&mut self.shared.state().launched);
        Python::attach(|py| {
            if let Some(guard) = guard {
                if let Err(error) =
                    guard.call_method1(py, "__exit__", (py.None(), py.None(), py.None()))
                {
                    tracing::warn!(
                        "the control thread's guard did not exit: {}",
                        describe(py, &error)
                    );
                }
            }
            // Forwards launched but never committed.
            drop(launched);
        });
    }
}

/// A Python exception with its traceback, for the log.
pub fn describe(py: Python<'_>, error: &PyErr) -> String {
    let formatted = py
        .import("traceback")
        .and_then(|traceback| {
            traceback.call_method1(
                "format_exception",
                (error.get_type(py), error.value(py), error.traceback(py)),
            )
        })
        .and_then(|lines| lines.extract::<Vec<String>>());
    match formatted {
        Ok(lines) => lines.concat(),
        Err(_) => error.to_string(),
    }
}

fn forward_loop(
    py: Python<'_>,
    plane: &Bound<'_, PyAny>,
    mut queue: mpsc::Receiver<Op>,
    shared: &Shared,
    streams: Option<&Streams>,
) {
    loop {
        // Wait without the GIL: the control thread and the transfer peer's
        // threads need it.
        let (returned, op) = py.detach(move || {
            let op = queue.recv();
            (queue, op)
        });
        queue = returned;
        let Ok(op) = op else {
            // Closed: finish what a forward left pending before the caches
            // close behind this thread.
            if shared.failure().is_none() {
                if let Err(error) = plane.call_method0("exit_forward_thread") {
                    let error = describe(py, &error);
                    tracing::error!("the forward thread could not finish:\n{error}");
                    shared.fail(error);
                }
            }
            return;
        };
        if let Some(failure) = shared.failure() {
            // Nothing runs after a failure; a waiting caller learns of it.
            match op {
                Op::Barrier { reply }
                | Op::IdleForward { reply, .. }
                | Op::ReleaseMemory { reply, .. }
                | Op::ResumeMemory { reply, .. }
                | Op::ApplyExpertPlacement { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::UpdateWeights { reply, .. } | Op::Profile { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::SnapshotExpertLoad { reply } => {
                    let _ = reply.send(Err(failure));
                }
                Op::Admit(_)
                | Op::Execute(_)
                | Op::Committed(_)
                | Op::Release(_)
                | Op::Land { .. }
                | Op::EpdPublish(_)
                | Op::AttachGrammars(_) => {}
            }
            continue;
        }
        if let Err(error) = run(py, plane, op, shared, streams) {
            tracing::error!("the data plane failed:\n{error}");
            shared.fail(error);
        }
    }
}

/// Run one operation; a failure ends the data plane.
fn run(
    py: Python<'_>,
    plane: &Bound<'_, PyAny>,
    op: Op,
    shared: &Shared,
    streams: Option<&Streams>,
) -> Result<(), String> {
    let python = |error: PyErr| describe(py, &error);
    match op {
        Op::Admit(rows) => {
            let rows = rows
                .into_iter()
                .map(|row| {
                    PyTuple::new(
                        py,
                        [
                            row.rid.into_pyobject(py)?.into_any(),
                            PyBytes::new(py, &row.sampling_params).into_any(),
                            PyList::new(py, row.stop_token_ids)?.into_any(),
                            row.token_limit.into_pyobject(py)?.into_any(),
                            match &row.multimodal {
                                Some(frames) => PyList::new(
                                    py,
                                    frames.iter().map(|frame| PyBytes::new(py, frame)),
                                )?
                                .into_any(),
                                None => py.None().into_bound(py),
                            },
                        ],
                    )
                })
                .collect::<PyResult<Vec<_>>>()
                .map_err(python)?;
            plane.call_method1("admit", (rows,)).map_err(python)?;
        }
        Op::Execute(execution) => {
            let Execution {
                plan,
                forward,
                dp,
                ngram,
                grammar_advance,
                logprobs,
                history,
            } = *execution;
            // The plan's remote stream goes out from the data plane's
            // execution, once its device prerequisites are met.
            if let Some(streams) = streams {
                streams.before_execute(&plan, forward.is_some())?;
            }
            let plan = plan.into_python(py).map_err(python)?;
            // `(prompt rows (row starts, counts, positions, top k) or None,
            // sampled top k)`, None for a plan without a batch.
            let logprobs = logprobs.map(|plan| {
                (
                    plan.input.map(|input| {
                        (
                            input.row_starts,
                            input.counts,
                            input.position_starts,
                            input.top_k,
                        )
                    }),
                    plan.sampled_top_k,
                )
            });
            let ngram = ngram.map(|windows| (windows.tokens, windows.positions));
            let dp = dp.map(|dp| {
                (
                    dp.num_tokens,
                    dp.batch_sizes,
                    dp.num_extends,
                    dp.idle_forward,
                )
            });
            let history = history.map(|seeds| (seeds.slots, seeds.prefix_lengths, seeds.tokens));
            let launched = plane
                .call_method1(
                    "execute",
                    (plan, dp, ngram, grammar_advance, logprobs, history),
                )
                .map_err(python)?;
            if let Some(streams) = streams {
                streams.after_execute()?;
            }
            match (forward, launched.is_none()) {
                (Some(forward), false) => shared.launch(forward, launched.unbind()),
                (None, true) => {}
                (Some(_), true) => {
                    return Err("the plan's batch ran no forward".to_owned());
                }
                (None, false) => {
                    return Err("a plan without a batch ran a forward".to_owned());
                }
            }
        }
        Op::Committed(launched) => {
            plane
                .call_method1("note_committed", (launched,))
                .map_err(python)?;
        }
        Op::Barrier { reply } => {
            answer(reply, Ok(()))?;
        }
        Op::Release(rids) => {
            plane.call_method1("release", (rids,)).map_err(python)?;
        }
        Op::Land {
            spec_candidates,
            cache_slot,
        } => {
            plane
                .call_method1("land", (spec_candidates, cache_slot))
                .map_err(python)?;
        }
        Op::IdleForward { dp, reply } => {
            let ran = plane
                .call_method1(
                    "idle_forward",
                    ((dp.num_tokens, dp.batch_sizes, dp.num_extends),),
                )
                .map(drop)
                .map_err(python);
            answer(reply, ran)?;
        }
        Op::UpdateWeights { payload, reply } => {
            let done = plane
                .call_method1("update_weights", (PyBytes::new(py, &payload),))
                .and_then(|result| result.extract())
                .map_err(python);
            answer(reply, done)?;
        }
        Op::Profile { payload, reply } => {
            let done = plane
                .call_method1("profile", (PyBytes::new(py, &payload),))
                .and_then(|result| result.extract())
                .map_err(python);
            answer(reply, done)?;
        }
        Op::ReleaseMemory { tags, reply } => {
            let done = plane
                .call_method1("release_memory", (tags,))
                .map(drop)
                .map_err(python);
            answer(reply, done)?;
        }
        Op::ResumeMemory { tags, reply } => {
            let done = plane
                .call_method1("resume_memory", (tags,))
                .map(drop)
                .map_err(python);
            answer(reply, done)?;
        }
        Op::AttachGrammars(grammars) => {
            plane
                .call_method1("attach_grammars", (grammars,))
                .map_err(python)?;
        }
        Op::EpdPublish(landed) => {
            let landed: Vec<(String, Vec<PyLandedItem>)> = landed
                .into_iter()
                .map(|request| {
                    let items = request
                        .items
                        .into_iter()
                        .map(|item| {
                            (
                                item.index,
                                item.slot,
                                item.row_start,
                                item.row_count,
                                item.rows,
                                item.sharded,
                            )
                        })
                        .collect();
                    (request.rid, items)
                })
                .collect();
            plane
                .call_method1("epd_publish", (landed,))
                .map_err(python)?;
        }
        Op::SnapshotExpertLoad { reply } => {
            let done = plane
                .call_method0("snapshot_expert_load")
                .and_then(|load| load.extract())
                .map_err(python);
            answer(reply, done)?;
        }
        Op::ApplyExpertPlacement {
            layer_ids,
            rows,
            moves,
            reply,
        } => {
            // Each layer's moves as (recv, send, local_copy, free_rider).
            type Pairs = Vec<(usize, usize)>;
            let moves: Vec<(Pairs, Pairs, Pairs, Pairs)> = moves
                .into_iter()
                .map(|m| (m.recv, m.send, m.local_copy, m.free_rider))
                .collect();
            let done = plane
                .call_method1("apply_expert_placement", (layer_ids, rows, moves))
                .map(drop)
                .map_err(python);
            answer(reply, done)?;
        }
    }
    Ok(())
}

/// Send an answer; a failure is also the forward thread's own.
fn answer<T>(
    reply: mpsc::SyncSender<Result<T, String>>,
    answer: Result<T, String>,
) -> Result<(), String> {
    let failure = answer.as_ref().err().cloned();
    // The control thread may be gone; the failure is recorded anyway.
    let _ = reply.send(answer);
    match failure {
        Some(failure) => Err(failure),
        None => Ok(()),
    }
}

type Completed = (
    Vec<i32>,
    Vec<i32>,
    Option<Vec<f64>>,
    Option<Vec<i32>>,
    Option<Vec<Vec<i32>>>,
    Option<(u64, f64)>,
    Option<Vec<i32>>,
    Option<(Option<Vec<f64>>, Option<TopLogprobs>, Option<TopLogprobs>)>,
);

/// A launched forward's results, once they landed.
fn complete(plane: &Bound<'_, PyAny>, launched: &Py<PyAny>) -> PyResult<ForwardResult> {
    let (
        output_tokens,
        output_lengths,
        output_logprobs,
        nan_flags,
        next_input_ids,
        serving_compiles,
        grammar_terminated_at,
        logprobs,
    ): Completed = plane.call_method1("complete", (launched,))?.extract()?;
    let (input_token_logprobs, input_top_logprobs, output_top_logprobs) =
        logprobs.unwrap_or_default();
    Ok(ForwardResult {
        output_tokens,
        output_lengths,
        output_logprobs,
        nan_flags,
        next_input_ids,
        serving_compiles,
        grammar_terminated_at,
        input_token_logprobs,
        input_top_logprobs,
        output_top_logprobs,
    })
}

#[cfg(test)]
mod tests {
    //! The two-thread contract, against a Python stand-in for `HostDataPlane`
    //! that records which thread made each call.

    use std::{ffi::CString, sync::Once, time::Duration};

    use pyo3::types::PyDict;
    use ts_scheduler::{ForwardBatchView, Scheduler};

    use super::*;

    /// Start the interpreter once per test process, with the scheduler
    /// module built in and the repository's `tokenspeed_scheduler` package
    /// importable.
    fn interpreter() {
        static START: Once = Once::new();
        START.call_once(|| {
            // The test binary is not in a Python prefix; point at the one the
            // crate was built against.
            std::env::set_var("PYTHONHOME", env!("TS_HOST_TEST_PYTHONHOME"));
            ts_scheduler::register_builtin_module();
            Python::initialize();
            Python::attach(|py| {
                ts_scheduler::alias_builtin_module(py).unwrap();
                let package_dir = concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../../tokenspeed-scheduler/python"
                );
                py.import("sys")
                    .unwrap()
                    .getattr("path")
                    .unwrap()
                    .call_method1("insert", (0, package_dir))
                    .unwrap();
            });
        });
    }

    /// The stand-in, and an idle scheduler to draw plans from. A launch
    /// returns the next token, from 100; the launch numbered `held` waits
    /// for `unblock`, and every launch raises once `failing` is set.
    const PLANE: &str = r#"
import threading

import tokenspeed_scheduler as ts


class Guard:
    def __init__(self, plane):
        self.plane = plane

    def __enter__(self):
        self.plane.note("guard enter")

    def __exit__(self, *exc):
        self.plane.note("guard exit")


class Plane:
    def __init__(self):
        self.calls = []
        self.next_token = 100
        self.held = None
        self.unblock = threading.Event()
        self.failing = False

    def note(self, name):
        self.calls.append((name, threading.get_ident()))

    def control_thread_guard(self):
        return Guard(self)

    def enter_forward_thread(self):
        self.note("enter_forward_thread")

    def exit_forward_thread(self):
        self.note("exit_forward_thread")

    def execute(self, plan, dp, ngram, grammar_advance, logprobs, history):
        self.note("execute")
        token = self.next_token
        self.next_token += 1
        if token == self.held:
            self.unblock.wait()
        if self.failing:
            raise RuntimeError("the launch failed")
        return token

    def complete(self, launched):
        self.note("complete")
        return ([launched], [1], None, None, None, None, None, None)

    def note_committed(self, launched):
        self.note("note_committed")

    def release(self, rids):
        self.note("release")

    def poll_cache_events(self):
        self.note("poll_cache_events")
        return [("WriteBackDoneEvent", 3)]


cfg = ts.SchedulerConfig()
cfg.prefix_granularity = 16
cfg.max_scheduled_tokens = 64
cfg.max_batch_size = 4
cfg.num_device_pages = 64
cfg.num_snapshot_pages = 1
cfg.cache_groups = [
    ts.CacheGroupConfig(
        group_id="full_attention",
        block_granularity=16,
        total_pages=64,
        attn_kind=ts.AttnKind.Full,
    )
]
result = (Plane(), ts.Scheduler(cfg))
"#;

    fn plane() -> (Py<PyAny>, Scheduler) {
        interpreter();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(&CString::new(PLANE).unwrap(), Some(&globals), None)
                .unwrap_or_else(|error| panic!("{}", describe(py, &error)));
            let (plane, scheduler): (Py<PyAny>, Bound<'_, PyAny>) = globals
                .get_item("result")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            (plane, Scheduler::from_python(&scheduler).unwrap())
        })
    }

    fn start(plane: &Py<PyAny>) -> PythonDataPlane {
        PythonDataPlane::start(Python::attach(|py| plane.clone_ref(py)), None).unwrap()
    }

    /// A one-row batch: the stand-in launches whatever it is handed.
    fn batch() -> ForwardBatchView {
        ForwardBatchView {
            request_ids: vec!["r".to_owned()],
            request_pool_indices: vec![0],
            input_lengths: vec![1],
            prefill_lengths: vec![1],
            extend_prefix_lens: Vec::new(),
            extend_replay_lens: Vec::new(),
            num_extends: 0,
        }
    }

    fn set(plane: &Py<PyAny>, name: &str, value: i64) {
        Python::attach(|py| plane.setattr(py, name, value).unwrap());
    }

    /// The calls the stand-in recorded, with the thread each ran on.
    fn calls(plane: &Py<PyAny>) -> Vec<(String, u64)> {
        Python::attach(|py| plane.getattr(py, "calls").unwrap().extract(py).unwrap())
    }

    fn this_thread() -> u64 {
        Python::attach(|py| {
            py.import("threading")
                .unwrap()
                .call_method0("get_ident")
                .unwrap()
                .extract()
                .unwrap()
        })
    }

    #[test]
    fn the_forward_thread_runs_the_queue_and_the_control_thread_its_queries() {
        let (plane, mut scheduler) = plane();
        let batch = batch();
        let dispatch = Dispatch {
            forward: Some(&batch),
            ..Dispatch::default()
        };
        let mut data_plane = start(&plane);
        let forward = data_plane
            .execute(scheduler.next_execution_plan(), &dispatch)
            .unwrap()
            .unwrap();
        assert_eq!(data_plane.complete(forward).unwrap().output_tokens, [100]);
        let polled = data_plane.poll_cache_events().unwrap();
        assert_eq!(polled.len(), 1);
        data_plane.release(&["r".to_owned()]).unwrap();
        // Answered once everything queued before it ran.
        data_plane.ask(|reply| Op::Barrier { reply }).unwrap();
        drop(data_plane);

        let control = this_thread();
        let calls = calls(&plane);
        let on = |thread: u64| -> Vec<&str> {
            calls
                .iter()
                .filter(|(_, ran_on)| *ran_on == thread)
                .map(|(name, _)| name.as_str())
                .collect()
        };
        assert_eq!(
            on(control),
            ["guard enter", "complete", "poll_cache_events", "guard exit"]
        );
        let forward_thread = calls
            .iter()
            .find(|(name, _)| name == "enter_forward_thread")
            .unwrap()
            .1;
        assert_ne!(forward_thread, control);
        assert_eq!(
            on(forward_thread),
            [
                "enter_forward_thread",
                "execute",
                "note_committed",
                "release",
                "exit_forward_thread"
            ]
        );
    }

    #[test]
    fn a_commit_does_not_wait_behind_a_later_launch() {
        let (plane, mut scheduler) = plane();
        let batch = batch();
        let dispatch = Dispatch {
            forward: Some(&batch),
            ..Dispatch::default()
        };
        set(&plane, "held", 101);
        let mut data_plane = start(&plane);
        let first = data_plane
            .execute(scheduler.next_execution_plan(), &dispatch)
            .unwrap()
            .unwrap();
        // The forward thread will sit in this launch until released.
        let second = data_plane
            .execute(scheduler.next_execution_plan(), &dispatch)
            .unwrap()
            .unwrap();
        // A commit queued behind it would wait forever; release the held
        // launch after a while regardless, and fail the test then.
        let (cancel, cancelled) = mpsc::channel::<()>();
        let held = Python::attach(|py| plane.clone_ref(py));
        let watchdog = thread::spawn(move || {
            let timed_out = cancelled.recv_timeout(Duration::from_secs(10)).is_err();
            Python::attach(|py| held.getattr(py, "unblock")?.call_method0(py, "set")).unwrap();
            timed_out
        });
        assert_eq!(data_plane.complete(first).unwrap().output_tokens, [100]);
        cancel.send(()).unwrap();
        assert!(
            !watchdog.join().unwrap(),
            "the commit waited for the launch"
        );
        assert_eq!(data_plane.complete(second).unwrap().output_tokens, [101]);
    }

    #[test]
    fn a_failed_launch_fails_the_commit_waiting_for_it() {
        let (plane, mut scheduler) = plane();
        let batch = batch();
        let dispatch = Dispatch {
            forward: Some(&batch),
            ..Dispatch::default()
        };
        Python::attach(|py| plane.setattr(py, "failing", true).unwrap());
        let mut data_plane = start(&plane);
        let forward = data_plane
            .execute(scheduler.next_execution_plan(), &dispatch)
            .unwrap()
            .unwrap();
        let error = data_plane.complete(forward).unwrap_err();
        assert!(error.0.contains("the launch failed"), "{}", error.0);
        // Every later call reports the failure, on either thread's side.
        assert!(data_plane.poll_cache_events().is_err());
        assert!(data_plane.release(&["r".to_owned()]).is_err());
    }
}
