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

//! The data plane behind a pump thread.
//!
//! The control thread never takes the GIL: each data-plane call queues an
//! operation, and the pump thread runs the queue in order against the
//! Python `DataPlanePump` (`engine/rust_host.py`). Only `complete`, the
//! cache poll and an idle forward wait for the pump, because the loop needs
//! their answers or, for the idle forward, its completion. The queue keeps
//! the loop's order: a request is admitted before a plan names it, and a
//! round's plan is dispatched before the previous forward is waited for.

use std::{
    collections::HashMap,
    sync::{mpsc, Arc, Mutex},
    thread,
};

use bytes::Bytes;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyList, PyTuple},
};
use ts_control::eplb::SlotMoves;
use ts_control::{
    cache_ops::CachePoll,
    event_loop::{DataPlane, DataPlaneError, Dispatch, DpRound},
    ingress::Admission,
    output::{ForwardResult, HistorySeeds, LogprobPlan, NgramWindows, TopLogprobs},
};
use ts_scheduler::{CacheEvent, CacheEventKind, CacheKeyView, ExecutionPlan};
use ts_wire::{control::ProfileReq, ControlRequest, Wire};

/// A forward the pump dispatched, by dispatch order.
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

enum Op {
    Admit(Vec<AdmitRow>),
    Execute(Box<Execution>),
    AttachGrammars(Vec<(String, u64, Vec<i32>)>),
    Complete {
        forward: ForwardId,
        reply: mpsc::SyncSender<Result<ForwardResult, String>>,
    },
    PollCache {
        reply: mpsc::SyncSender<Result<CachePoll, String>>,
    },
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
    /// One of the L3 storage tier's operations.
    L3 {
        request: L3Request,
        reply: mpsc::SyncSender<Result<L3Reply, String>>,
    },
    SnapshotExpertLoad {
        reply: mpsc::SyncSender<Result<ExpertLoad, String>>,
    },
    EpdStage(Vec<String>),
    EpdPoll {
        reply: mpsc::SyncSender<Result<Vec<i64>, String>>,
    },
    EpdSettle {
        codes: Vec<i64>,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
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

/// The L3 operations (`DataPlanePump.l3_*`).
enum L3Request {
    Query(Vec<CacheKeyView>),
    Progress,
    Complete { op_id: u32, landed_pages: i64 },
    DeleteNamespace,
    SetWeightVersion(String),
}

enum L3Reply {
    Exists(Vec<bool>),
    Progress(Vec<(u32, bool, i64)>),
    Done(bool),
}

struct AdmitRow {
    rid: String,
    sampling_params: Bytes,
    stop_token_ids: Vec<i32>,
    token_limit: i64,
    room: Option<u64>,
    multimodal: Option<Vec<Bytes>>,
}

/// The data plane, run by a pump thread.
pub struct PythonDataPlane {
    ops: Option<mpsc::Sender<Op>>,
    /// The first failure on the pump; every later call reports it.
    failure: Arc<Mutex<Option<String>>>,
    next_forward: ForwardId,
    thread: Option<thread::JoinHandle<()>>,
}

impl PythonDataPlane {
    /// Start the pump thread over a Python `DataPlanePump`.
    pub fn start(pump: Py<PyAny>) -> std::io::Result<Self> {
        let (ops, queue) = mpsc::channel();
        let failure = Arc::new(Mutex::new(None));
        let pump_failure = failure.clone();
        let thread = thread::Builder::new()
            .name("ts-pump".to_owned())
            .spawn(move || {
                Python::attach(|py| {
                    let pump = pump.bind(py);
                    // The pump issues no device work itself: everything
                    // crosses through the handle (Principle 1 of
                    // `docs/design/event-loop.md`), which a dispatch guard
                    // on this thread enforces.
                    let guard = match pump
                        .call_method0("control_plane_guard")
                        .and_then(|guard| guard.call_method0("__enter__").map(|_| guard))
                    {
                        Ok(guard) => guard,
                        Err(error) => {
                            let error = describe(py, &error);
                            tracing::error!("the pump could not enter its guard:\n{error}");
                            lock(&pump_failure).get_or_insert(error);
                            return;
                        }
                    };
                    pump_loop(py, pump, queue, &pump_failure);
                    if let Err(error) =
                        guard.call_method1("__exit__", (py.None(), py.None(), py.None()))
                    {
                        tracing::warn!("the pump's guard did not exit: {}", describe(py, &error));
                    }
                })
            })?;
        Ok(Self {
            ops: Some(ops),
            failure,
            next_forward: 0,
            thread: Some(thread),
        })
    }

    fn failed(&self) -> Result<(), DataPlaneError> {
        match lock(&self.failure).as_ref() {
            Some(failure) => Err(DataPlaneError(failure.clone())),
            None => Ok(()),
        }
    }

    fn queue(&self, op: Op) -> Result<(), DataPlaneError> {
        self.failed()?;
        self.ops
            .as_ref()
            .expect("open until drop")
            .send(op)
            .map_err(|_| DataPlaneError("the pump thread exited".to_owned()))
    }

    /// Queue an operation the pump answers, and wait for the answer.
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
                Err(DataPlaneError("the pump thread exited".to_owned()))
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
                room: request.bootstrap.as_ref().map(|bootstrap| bootstrap.room),
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
        // The pump runs a forward exactly when the plan has a batch and the
        // rank does not idle, by the same rule the loop chose `forward` with.
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
        self.ask(|reply| Op::Complete { forward, reply })
    }

    fn poll_cache_events(&mut self) -> Result<CachePoll, DataPlaneError> {
        self.ask(|reply| Op::PollCache { reply })
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

    fn query_l3_storage(&mut self, pages: &[CacheKeyView]) -> Result<Vec<bool>, DataPlaneError> {
        match self.l3(L3Request::Query(pages.to_vec()))? {
            L3Reply::Exists(exists) => Ok(exists),
            _ => Err(DataPlaneError("an L3 probe got another reply".to_owned())),
        }
    }

    fn l3_prefetch_progress(&mut self) -> Result<Vec<(u32, bool, i64)>, DataPlaneError> {
        match self.l3(L3Request::Progress)? {
            L3Reply::Progress(progress) => Ok(progress),
            _ => Err(DataPlaneError(
                "an L3 progress read got another reply".to_owned(),
            )),
        }
    }

    fn complete_l3_prefetch(
        &mut self,
        op_id: u32,
        landed_pages: i64,
    ) -> Result<(), DataPlaneError> {
        self.l3(L3Request::Complete {
            op_id,
            landed_pages,
        })
        .map(drop)
    }

    fn delete_l3_namespace(&mut self) -> Result<bool, DataPlaneError> {
        match self.l3(L3Request::DeleteNamespace)? {
            L3Reply::Done(deleted) => Ok(deleted),
            _ => Err(DataPlaneError("an L3 delete got another reply".to_owned())),
        }
    }

    fn set_l3_weight_version(&mut self, version: &str) -> Result<(), DataPlaneError> {
        self.l3(L3Request::SetWeightVersion(version.to_owned()))
            .map(drop)
    }

    fn attach_grammars(
        &mut self,
        grammars: &[(String, u64, Vec<i32>)],
    ) -> Result<(), DataPlaneError> {
        self.queue(Op::AttachGrammars(grammars.to_vec()))
    }

    fn epd_stage(&mut self, rids: &[String]) -> Result<(), DataPlaneError> {
        self.queue(Op::EpdStage(rids.to_vec()))
    }

    fn epd_poll(&mut self) -> Result<Vec<i64>, DataPlaneError> {
        self.ask(|reply| Op::EpdPoll { reply })
    }

    fn epd_settle(&mut self, codes: &[i64]) -> Result<(), DataPlaneError> {
        let codes = codes.to_vec();
        self.ask(|reply| Op::EpdSettle { codes, reply })
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

impl PythonDataPlane {
    fn l3(&mut self, request: L3Request) -> Result<L3Reply, DataPlaneError> {
        self.ask(|reply| Op::L3 { request, reply })
    }
}

impl Drop for PythonDataPlane {
    fn drop(&mut self) {
        // Closing the queue ends the pump once it ran what is queued.
        self.ops.take();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("the pump thread panicked");
            }
        }
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

fn pump_loop(
    py: Python<'_>,
    pump: &Bound<'_, PyAny>,
    mut queue: mpsc::Receiver<Op>,
    failure: &Mutex<Option<String>>,
) {
    let mut pending: HashMap<ForwardId, Py<PyAny>> = HashMap::new();
    loop {
        // Wait without the GIL: the data plane's own threads need it.
        let (returned, op) = py.detach(move || {
            let op = queue.recv();
            (queue, op)
        });
        queue = returned;
        let Ok(op) = op else {
            return;
        };
        if let Some(failure) = lock(failure).clone() {
            // Nothing runs after a failure; a waiting caller learns of it.
            match op {
                Op::Complete { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::PollCache { reply } => {
                    let _ = reply.send(Err(failure));
                }
                Op::Barrier { reply }
                | Op::IdleForward { reply, .. }
                | Op::ReleaseMemory { reply, .. }
                | Op::ResumeMemory { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::UpdateWeights { reply, .. } | Op::Profile { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::L3 { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::SnapshotExpertLoad { reply } => {
                    let _ = reply.send(Err(failure));
                }
                Op::EpdPoll { reply } => {
                    let _ = reply.send(Err(failure));
                }
                Op::EpdSettle { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::ApplyExpertPlacement { reply, .. } => {
                    let _ = reply.send(Err(failure));
                }
                Op::Admit(_)
                | Op::Execute { .. }
                | Op::Release(_)
                | Op::Land { .. }
                | Op::EpdStage(_)
                | Op::AttachGrammars(_) => {}
            }
            continue;
        }
        if let Err(error) = run(py, pump, op, &mut pending) {
            tracing::error!("the data plane failed:\n{error}");
            lock(failure).get_or_insert(error);
        }
    }
}

/// Run one operation; a failure ends the data plane.
fn run(
    py: Python<'_>,
    pump: &Bound<'_, PyAny>,
    op: Op,
    pending: &mut HashMap<ForwardId, Py<PyAny>>,
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
                            row.room.into_pyobject(py)?.into_any(),
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
            pump.call_method1("admit", (rows,)).map_err(python)?;
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
            let dispatched = pump
                .call_method1(
                    "execute",
                    (plan, dp, ngram, grammar_advance, logprobs, history),
                )
                .map_err(python)?;
            match (forward, dispatched.is_none()) {
                (Some(forward), false) => {
                    pending.insert(forward, dispatched.unbind());
                }
                (None, true) => {}
                (Some(_), true) => {
                    return Err("the plan's batch ran no forward".to_owned());
                }
                (None, false) => {
                    return Err("a plan without a batch ran a forward".to_owned());
                }
            }
        }
        Op::Complete { forward, reply } => {
            answer(reply, complete(py, pump, pending.remove(&forward)))?;
        }
        Op::PollCache { reply } => {
            answer(reply, poll_cache(py, pump))?;
        }
        Op::Barrier { reply } => {
            answer(reply, Ok(()))?;
        }
        Op::Release(rids) => {
            pump.call_method1("release", (rids,)).map_err(python)?;
        }
        Op::Land {
            spec_candidates,
            cache_slot,
        } => {
            pump.call_method1("land", (spec_candidates, cache_slot))
                .map_err(python)?;
        }
        Op::IdleForward { dp, reply } => {
            let ran = pump
                .call_method1(
                    "idle_forward",
                    ((dp.num_tokens, dp.batch_sizes, dp.num_extends),),
                )
                .map(drop)
                .map_err(python);
            answer(reply, ran)?;
        }
        Op::UpdateWeights { payload, reply } => {
            let done = pump
                .call_method1("update_weights", (PyBytes::new(py, &payload),))
                .and_then(|result| result.extract())
                .map_err(python);
            answer(reply, done)?;
        }
        Op::Profile { payload, reply } => {
            let done = pump
                .call_method1("profile", (PyBytes::new(py, &payload),))
                .and_then(|result| result.extract())
                .map_err(python);
            answer(reply, done)?;
        }
        Op::ReleaseMemory { tags, reply } => {
            let done = pump
                .call_method1("release_memory", (tags,))
                .map(drop)
                .map_err(python);
            answer(reply, done)?;
        }
        Op::ResumeMemory { tags, reply } => {
            let done = pump
                .call_method1("resume_memory", (tags,))
                .map(drop)
                .map_err(python);
            answer(reply, done)?;
        }
        Op::L3 { request, reply } => {
            let done = match request {
                L3Request::Query(pages) => {
                    let pages: Vec<(u32, String, i32)> = pages
                        .into_iter()
                        .map(|key| (key.group_id, key.content_hash, key.page_offset))
                        .collect();
                    pump.call_method1("query_l3_storage", (pages,))
                        .and_then(|exists| exists.extract())
                        .map(L3Reply::Exists)
                }
                L3Request::Progress => pump
                    .call_method0("l3_prefetch_progress")
                    .and_then(|progress| progress.extract())
                    .map(L3Reply::Progress),
                L3Request::Complete {
                    op_id,
                    landed_pages,
                } => pump
                    .call_method1("complete_l3_prefetch", (op_id, landed_pages))
                    .map(|_| L3Reply::Done(true)),
                L3Request::DeleteNamespace => pump
                    .call_method0("delete_l3_namespace")
                    .and_then(|deleted| deleted.extract())
                    .map(L3Reply::Done),
                L3Request::SetWeightVersion(version) => pump
                    .call_method1("set_l3_weight_version", (version,))
                    .map(|_| L3Reply::Done(true)),
            };
            answer(reply, done.map_err(python))?;
        }
        Op::AttachGrammars(grammars) => {
            pump.call_method1("attach_grammars", (grammars,))
                .map_err(python)?;
        }
        Op::EpdStage(rids) => {
            pump.call_method1("epd_stage", (rids,)).map_err(python)?;
        }
        Op::EpdPoll { reply } => {
            let done = pump
                .call_method0("epd_poll")
                .and_then(|codes| codes.extract())
                .map_err(python);
            answer(reply, done)?;
        }
        Op::EpdSettle { codes, reply } => {
            let done = pump
                .call_method1("epd_settle", (codes,))
                .map(drop)
                .map_err(python);
            answer(reply, done)?;
        }
        Op::SnapshotExpertLoad { reply } => {
            let done = pump
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
            let done = pump
                .call_method1("apply_expert_placement", (layer_ids, rows, moves))
                .map(drop)
                .map_err(python);
            answer(reply, done)?;
        }
    }
    Ok(())
}

/// Send an answer; a failure is also the pump's own.
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

/// The acknowledgments landed since the last poll, as
/// `cache_event_to_payload` names them.
fn poll_cache(py: Python<'_>, pump: &Bound<'_, PyAny>) -> Result<CachePoll, String> {
    let (polled, l3_backup_failed): (Vec<(String, u32, i32)>, bool) = pump
        .call_method0("poll_cache_events")
        .and_then(|events| events.extract())
        .map_err(|error| describe(py, &error))?;
    let events = polled
        .into_iter()
        .map(|(kind, op_id, landed_pages)| {
            let kind = match kind.as_str() {
                "WriteBackDoneEvent" => CacheEventKind::WriteBackDone,
                "LoadBackDoneEvent" => CacheEventKind::LoadBackDone,
                "PrefetchDoneEvent" => CacheEventKind::PrefetchDone,
                "SnapshotDoneEvent" => CacheEventKind::SnapshotDone,
                "RestoreDoneEvent" => CacheEventKind::RestoreDone,
                _ => return Err(format!("unknown cache event {kind}")),
            };
            Ok(CacheEvent {
                kind,
                op_id,
                landed_pages,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(CachePoll {
        events,
        l3_backup_failed,
    })
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

fn complete(
    py: Python<'_>,
    pump: &Bound<'_, PyAny>,
    dispatched: Option<Py<PyAny>>,
) -> Result<ForwardResult, String> {
    let dispatched = dispatched.ok_or("completing a forward that was never dispatched")?;
    let (
        output_tokens,
        output_lengths,
        output_logprobs,
        nan_flags,
        next_input_ids,
        serving_compiles,
        grammar_terminated_at,
        logprobs,
    ): Completed = pump
        .call_method1("complete", (dispatched,))
        .and_then(|result| result.extract())
        .map_err(|error| describe(py, &error))?;
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
