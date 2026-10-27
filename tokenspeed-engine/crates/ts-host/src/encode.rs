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

//! An EPD encode rank: the host's encode mode (`epd/encode_host.py` is its
//! Python half).
//!
//! The embedded interpreter loads the vision tower and allocates the send
//! ring. The host then starts the embedding sender over the ring (the
//! attention-TP group's first rank also the node's bootstrap server and the
//! request intake), connects the group's mesh, and runs `ts-control`'s
//! encode loop, whose data-plane calls go to the forward thread; the
//! completion query runs on this, the control thread.

use std::{
    sync::{
        atomic::AtomicBool,
        mpsc::{self, RecvTimeoutError, TryRecvError},
        Arc, Mutex, MutexGuard,
    },
    thread,
    time::Duration,
};

use bytes::Bytes;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyList, PyTuple},
};
use ts_collective::{Group, Listener, Mesh};
use ts_control::{
    encode::{EncodeConfig, EncodeDataPlane, EncodeLoop, EncodedRows, Intake, Ring},
    event_loop::DataPlaneError,
};
use ts_pd::{
    bootstrap::{BootstrapServer, Registry},
    epd::sender::{Sender, SenderConfig},
    PeerConfig,
};
use zeromq::{Socket, SocketRecv};

use crate::{data_plane::describe, python_error, HostError, HANDSHAKE_TIMEOUT};

/// The attention-TP group's collective id.
const TP_GROUP: u32 = 0;

/// What `encode_host.EncodeSettings` carries.
struct Settings {
    rank: usize,
    tp_ranks: Vec<usize>,
    collective_bind_host: String,
    intake: Option<String>,
    bootstrap_port: Option<u16>,
    sender: SenderConfig,
    config: EncodeConfig,
}

impl Settings {
    fn from_python(settings: &Bound<'_, PyAny>) -> PyResult<Self> {
        let get = |name: &str| settings.getattr(name);
        let slots: usize = get("ring_slots")?.extract()?;
        let slot_bytes: u64 = get("ring_slot_bytes")?.extract()?;
        let base: u64 = get("ring_base")?.extract()?;
        let deepstack_base: u64 = get("ring_deepstack_base")?.extract()?;
        let ring_bytes = slots * slot_bytes as usize;
        let mut regions = vec![(base as usize, ring_bytes)];
        if deepstack_base != 0 {
            regions.push((deepstack_base as usize, ring_bytes));
        }
        let rank: usize = get("rank")?.extract()?;
        let tp_ranks: Vec<usize> = get("tp_ranks")?.extract()?;
        Ok(Self {
            rank,
            collective_bind_host: get("collective_bind_host")?.extract()?,
            intake: get("intake")?.extract()?,
            bootstrap_port: get("bootstrap_port")?.extract()?,
            sender: SenderConfig {
                peer: PeerConfig {
                    library: get("library")?.extract::<String>()?.into(),
                    local_ip: get("local_ip")?.extract()?,
                    protocol: get("protocol")?.extract()?,
                    ib_device: get("ib_device")?.extract()?,
                    regions,
                    engine_rank: rank as i64,
                    metrics: None,
                },
                bootstrap_url: get("bootstrap_url")?.extract()?,
                tp_size: tp_ranks.len() as i64,
                bootstrap_timeout: Duration::from_secs_f64(get("bootstrap_timeout_s")?.extract()?),
                transfer_workers: get("transfer_workers")?.extract()?,
            },
            config: EncodeConfig {
                max_batch_tokens: get("max_batch_tokens")?.extract()?,
                max_batch_items: get("max_batch_items")?.extract()?,
                ring: Ring {
                    slots,
                    slot_bytes,
                    base,
                    deepstack_slot_bytes: if deepstack_base != 0 { slot_bytes } else { 0 },
                    deepstack_base,
                },
            },
            tp_ranks,
        })
    }
}

/// Serve an encode rank built by `bootstrap` (an `EncodeBootstrap`) until
/// `stop` is set on the group's first rank.
pub fn serve(bootstrap: Py<PyAny>, stop: &AtomicBool) -> Result<(), HostError> {
    let (settings, data_plane) = Python::attach(|py| {
        let built = (|| {
            let bootstrap = bootstrap.bind(py);
            let settings = Settings::from_python(&bootstrap.getattr("config")?)?;
            Ok::<_, PyErr>((settings, bootstrap.getattr("data_plane")?.unbind()))
        })();
        built.map_err(|error| python_error(py, error))
    })?;

    // The first rank hosts the node's bootstrap server, which every rank's
    // sender registers with.
    let _server = settings
        .bootstrap_port
        .map(|port| BootstrapServer::start(port, Registry::Addresses))
        .transpose()?;
    let sender = Sender::start(settings.sender)?;

    let listener = Listener::bind(&settings.collective_bind_host)?;
    let port = listener.port()?;
    let addresses: Vec<String> = Python::attach(|py| {
        bootstrap
            .call_method1(py, "collective_addresses", (port,))
            .and_then(|addresses| addresses.extract(py))
            .map_err(|error| python_error(py, error))
    })?;
    let mesh = Mesh::connect(settings.rank, listener, &addresses, HANDSHAKE_TIMEOUT)?;
    let group = Group::new(TP_GROUP, settings.tp_ranks.clone(), settings.rank)?;
    let intake = settings
        .intake
        .as_deref()
        .map(ZmqIntake::bind)
        .transpose()?
        .map(|intake| Box::new(intake) as Box<dyn Intake + Send>);

    for method in ["report_ready", "enter_serving"] {
        Python::attach(|py| {
            bootstrap
                .call_method0(py, method)
                .map(drop)
                .map_err(|error| python_error(py, error))
        })?;
    }
    let data_plane = PythonEncodePlane::start(data_plane).map_err(HostError::DataPlane)?;
    tracing::info!(
        rank = settings.rank,
        tp_ranks = ?settings.tp_ranks,
        slots = settings.config.ring.slots,
        "encoding"
    );
    let mut encode = EncodeLoop::new(settings.config, data_plane, sender, intake, mesh, group);
    let served = encode.run(stop);
    // Stops the forward thread and the sender's threads.
    drop(encode);
    Ok(served?)
}

/// The first rank's PULL socket for the frontend's encode requests.
struct ZmqIntake {
    messages: mpsc::Receiver<Vec<Bytes>>,
    _runtime: tokio::runtime::Runtime,
}

impl ZmqIntake {
    /// Bind `endpoint` (`tcp://host:port`), which the frontend connects to.
    fn bind(endpoint: &str) -> Result<Self, HostError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("ts-encode-intake")
            .enable_all()
            .build()?;
        let mut pull = zeromq::PullSocket::new();
        runtime
            .block_on(pull.bind(endpoint))
            .map_err(|error| HostError::Interpreter(format!("binding {endpoint}: {error}")))?;
        let (incoming, messages) = mpsc::channel();
        runtime.spawn(async move {
            loop {
                match pull.recv().await {
                    Ok(message) => {
                        if incoming.send(message.into_vec()).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        tracing::warn!(%error, "the encode request socket closed");
                        return;
                    }
                }
            }
        });
        Ok(Self {
            messages,
            _runtime: runtime,
        })
    }
}

impl Intake for ZmqIntake {
    fn receive(&mut self, wait: Duration) -> Vec<Vec<Bytes>> {
        let first = if wait.is_zero() {
            self.messages.try_recv().map_err(|error| match error {
                TryRecvError::Empty => RecvTimeoutError::Timeout,
                TryRecvError::Disconnected => RecvTimeoutError::Disconnected,
            })
        } else {
            self.messages.recv_timeout(wait)
        };
        let Ok(first) = first else {
            return Vec::new();
        };
        let mut received = vec![first];
        received.extend(self.messages.try_iter());
        received
    }
}

/// An operation for the forward thread.
enum Op {
    Admit {
        requests: Vec<(u64, Vec<Bytes>)>,
        reply: mpsc::SyncSender<Result<Vec<bool>, String>>,
    },
    Encode(Vec<(u64, usize)>),
    Release(Vec<u64>),
}

/// The first failure of the data plane; nothing runs after it.
#[derive(Default)]
struct Failure(Mutex<Option<String>>);

impl Failure {
    fn get(&self) -> MutexGuard<'_, Option<String>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set(&self, failure: String) {
        self.get().get_or_insert(failure);
    }
}

/// The encode data plane: the forward thread and the control thread's
/// completion query, over a Python `EncodeDataPlane`.
struct PythonEncodePlane {
    plane: Py<PyAny>,
    ops: Option<mpsc::Sender<Op>>,
    failure: Arc<Failure>,
    thread: Option<thread::JoinHandle<()>>,
}

impl PythonEncodePlane {
    fn start(plane: Py<PyAny>) -> Result<Self, String> {
        let (ops, queue) = mpsc::channel::<Op>();
        let failure = Arc::new(Failure::default());
        let forward_plane = Python::attach(|py| plane.clone_ref(py));
        let thread = {
            let failure = failure.clone();
            thread::Builder::new()
                .name("ts-forward".to_owned())
                .spawn(move || {
                    Python::attach(|py| forward_loop(py, forward_plane.bind(py), queue, &failure));
                    failure.set("the forward thread exited".to_owned());
                })
                .map_err(|error| format!("starting the forward thread: {error}"))?
        };
        Ok(Self {
            plane,
            ops: Some(ops),
            failure,
            thread: Some(thread),
        })
    }

    fn failed(&self) -> Result<(), DataPlaneError> {
        match self.failure.get().clone() {
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
}

impl Drop for PythonEncodePlane {
    fn drop(&mut self) {
        self.ops.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn forward_loop(
    py: Python<'_>,
    plane: &Bound<'_, PyAny>,
    mut queue: mpsc::Receiver<Op>,
    failure: &Failure,
) {
    if let Err(error) = plane.call_method0("enter_forward_thread") {
        failure.set(describe(py, &error));
        return;
    }
    loop {
        // Wait without the GIL: the control thread's query needs it.
        let (returned, op) = py.detach(move || {
            let op = queue.recv();
            (queue, op)
        });
        queue = returned;
        let Ok(op) = op else {
            return;
        };
        if let Some(failed) = failure.get().clone() {
            if let Op::Admit { reply, .. } = op {
                let _ = reply.send(Err(failed));
            }
            continue;
        }
        let ran: PyResult<()> = (|| match op {
            Op::Admit { requests, reply } => {
                let requests = requests
                    .iter()
                    .map(|(admission, frames)| {
                        let frames =
                            PyList::new(py, frames.iter().map(|frame| PyBytes::new(py, frame)))?;
                        PyTuple::new(
                            py,
                            [admission.into_pyobject(py)?.into_any(), frames.into_any()],
                        )
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let answer = plane
                    .call_method1("admit", (requests,))
                    .and_then(|cached| cached.extract::<Vec<bool>>());
                let _ = reply.send(
                    answer
                        .as_ref()
                        .map(Clone::clone)
                        .map_err(|error| describe(py, error)),
                );
                answer.map(drop)
            }
            Op::Encode(batch) => plane.call_method1("encode", (batch,)).map(drop),
            Op::Release(admissions) => plane.call_method1("release", (admissions,)).map(drop),
        })();
        if let Err(error) = ran {
            let error = describe(py, &error);
            tracing::error!("the encode data plane failed:\n{error}");
            failure.set(error);
        }
    }
}

/// One row of `EncodeDataPlane.encoded`.
type EncodedRow = (u64, Option<String>, i64, i64, String, u64, i64, u64);

impl EncodeDataPlane for PythonEncodePlane {
    fn admit(&mut self, requests: &[(u64, Vec<Bytes>)]) -> Result<Vec<bool>, DataPlaneError> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.queue(Op::Admit {
            requests: requests.to_vec(),
            reply,
        })?;
        match answer.recv() {
            Ok(answer) => answer.map_err(DataPlaneError),
            Err(_) => {
                self.failed()?;
                Err(DataPlaneError("the forward thread exited".to_owned()))
            }
        }
    }

    fn encode(&mut self, batch: &[(u64, usize)]) -> Result<(), DataPlaneError> {
        self.queue(Op::Encode(batch.to_vec()))
    }

    fn release(&mut self, admissions: &[u64]) -> Result<(), DataPlaneError> {
        self.queue(Op::Release(admissions.to_vec()))
    }

    fn encoded(&mut self) -> Result<Vec<(u64, Result<EncodedRows, String>)>, DataPlaneError> {
        self.failed()?;
        let rows: Vec<EncodedRow> = Python::attach(|py| {
            self.plane
                .call_method0(py, "encoded")
                .and_then(|rows| rows.extract(py))
                .map_err(|error| describe(py, &error))
        })
        .map_err(|failure| {
            self.failure.set(failure.clone());
            DataPlaneError(failure)
        })?;
        Ok(rows
            .into_iter()
            .map(
                |(
                    admission,
                    failure,
                    rows,
                    hidden,
                    dtype,
                    nbytes,
                    deepstack_width,
                    deepstack_nbytes,
                )| {
                    let outcome = match failure {
                        Some(reason) => Err(reason),
                        None => Ok(EncodedRows {
                            rows,
                            hidden,
                            dtype,
                            nbytes,
                            deepstack_width,
                            deepstack_nbytes,
                        }),
                    };
                    (admission, outcome)
                },
            )
            .collect())
    }
}
