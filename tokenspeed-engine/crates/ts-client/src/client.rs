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

//! Request submission, routing and output streams.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::Duration,
};

use bytes::Bytes;
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::timeout,
};
use tracing::{debug, warn};
use ts_wire::{
    output::{FinishDetails, PromptTopLogprobs, TopLogprobs},
    BatchTokenIDOutSlim, ControlOutput, ControlRequest, EngineOutput, EngineRequest, RequestOutput,
    StrOrList, TokenizedGenerateReqInput,
};
use zeromq::{PullSocket, RouterRecvHalf, RouterSendHalf, SocketRecv, SocketSend, ZmqMessage};

use crate::{
    config::{ClientConfig, LoadBalance, Routing},
    error::{Error, Result, StreamError},
    handshake::{self, EngineInfo},
};

/// One step of a request's output.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputChunk {
    /// Token ids generated this step.
    pub output_ids: Vec<i32>,
    /// The finish type (`"stop"`, `"length"`, `"abort"`) on the last chunk.
    pub finish_reason: Option<String>,
    /// Prompt length.
    pub prompt_tokens: i64,
    /// Tokens generated so far.
    pub completion_tokens: i64,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: i64,
    /// Logprobs of this step's tokens; empty when not requested.
    pub output_token_logprobs: Vec<f64>,
    /// The finish reason in full on the last chunk (the matched token, an
    /// abort's message and codes), from an engine that sends it.
    pub finish_details: Option<FinishDetails>,
    /// Verify rounds so far, with speculative decoding.
    pub spec_verify_ct: Option<i64>,
    /// Accepted tokens per verify round once finished, with speculative
    /// decoding.
    pub accept_draft_tokens: Option<f64>,
    /// The prompt logprobs (`None` for the first position) and their
    /// tokens, in the one chunk that carries them.
    pub input_token_logprobs: Option<(Vec<Option<f64>>, Vec<i32>)>,
    /// The prompt positions' top-k alternatives, beside the prompt
    /// logprobs.
    pub input_top_logprobs: Option<PromptTopLogprobs>,
    /// This step's tokens' top-k alternatives, when requested.
    pub output_top_logprobs: Option<TopLogprobs>,
}

impl From<&RequestOutput<'_>> for OutputChunk {
    fn from(row: &RequestOutput<'_>) -> Self {
        fn owned<A: Clone, B: Clone>(pair: Option<(&[A], &[B])>) -> Option<(Vec<A>, Vec<B>)> {
            pair.map(|(a, b)| (a.to_vec(), b.to_vec()))
        }
        Self {
            output_ids: row.output_ids.to_vec(),
            finish_reason: row.finish_reason.map(str::to_owned),
            prompt_tokens: row.prompt_tokens,
            completion_tokens: row.completion_tokens,
            cached_tokens: row.cached_tokens,
            output_token_logprobs: row.output_token_logprobs.to_vec(),
            finish_details: row.finish_details.cloned(),
            spec_verify_ct: row.spec_verify_ct,
            accept_draft_tokens: row.accept_draft_tokens,
            input_token_logprobs: owned(row.input_token_logprobs),
            input_top_logprobs: owned(row.input_top_logprobs),
            output_top_logprobs: owned(row.output_top_logprobs),
        }
    }
}

/// An engine's latest scheduler load, as reported with its outputs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineLoad {
    /// Running requests.
    pub num_running: i64,
    /// Waiting requests.
    pub num_waiting: i64,
    /// KV pages in use by active requests.
    pub kv_active_pages: i64,
    /// KV pages in total.
    pub kv_total_pages: i64,
}

type StreamItem = std::result::Result<OutputChunk, StreamError>;

struct Engine {
    info: EngineInfo,
    identity: Bytes,
    /// This client's requests on the engine that have not finished.
    in_flight: AtomicUsize,
    load: Mutex<EngineLoad>,
}

struct Pending {
    engine: usize,
    tx: mpsc::UnboundedSender<StreamItem>,
}

/// A control operation awaiting every engine's reply.
struct ControlWaiter {
    replies: Vec<ControlOutput>,
    done: oneshot::Sender<Vec<ControlOutput>>,
}

struct Shared {
    engines: Vec<Engine>,
    first_engine_index: u16,
    routing: Routing,
    send_timeout: Duration,
    input: RouterSendHalf,
    requests: Mutex<HashMap<String, Pending>>,
    /// The control operation in flight per reply tag: replies carry no
    /// engine or request id, so one operation of a kind is out at a time.
    controls: Mutex<HashMap<&'static str, ControlWaiter>>,
    /// Serializes the control operations of each kind.
    control_turns: Mutex<HashMap<&'static str, Arc<tokio::sync::Mutex<()>>>>,
    /// Set once the engines are gone; no request is accepted after.
    closed: Mutex<Option<StreamError>>,
    round_robin: AtomicUsize,
    runtime: Handle,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding one of these locks leaves plain data behind.
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Shared {
    fn closed(&self) -> Option<StreamError> {
        lock(&self.closed).clone()
    }

    fn route(&self, request: &TokenizedGenerateReqInput) -> Result<usize> {
        let count = self.engines.len();
        let pinned = |rank: i64| {
            usize::try_from(rank)
                .ok()
                .filter(|rank| *rank < count)
                .ok_or_else(|| {
                    Error::InvalidRequest(format!(
                        "data_parallel_rank {rank} is outside 0..{count}"
                    ))
                })
        };
        match self.routing {
            Routing::BootstrapRoom => {
                let room = request.bootstrap_room.ok_or_else(|| {
                    Error::InvalidRequest(
                        "disaggregated serving needs a bootstrap_room on every request".to_owned(),
                    )
                })?;
                match request.data_parallel_rank {
                    Some(rank) => pinned(rank),
                    None => Ok((room % count as u64) as usize),
                }
            }
            Routing::Aggregated(policy) => {
                if let Some(rank) = request.data_parallel_rank {
                    return pinned(rank);
                }
                let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % count;
                Ok(match policy {
                    LoadBalance::RoundRobin => start,
                    // Scanning from the rotating start spreads ties.
                    LoadBalance::LeastInFlight => (0..count)
                        .map(|offset| (start + offset) % count)
                        .min_by_key(|engine| {
                            self.engines[*engine].in_flight.load(Ordering::Relaxed)
                        })
                        .expect("at least one engine"),
                    LoadBalance::MinimumCacheUsage => (0..count)
                        .map(|offset| (start + offset) % count)
                        .min_by_key(|engine| {
                            let engine = &self.engines[*engine];
                            (
                                lock(&engine.load).kv_active_pages,
                                engine.in_flight.load(Ordering::Relaxed),
                            )
                        })
                        .expect("at least one engine"),
                })
            }
        }
    }

    async fn send(&self, engine: usize, frames: Vec<Bytes>) -> Result<()> {
        let mut message = ZmqMessage::from(self.engines[engine].identity.clone());
        for frame in frames {
            message.push_back(frame);
        }
        let mut input = self.input.clone();
        match timeout(self.send_timeout, input.send(message)).await {
            Ok(sent) => Ok(sent?),
            Err(_) => Err(Error::SendTimeout {
                engine_index: self.engines[engine].info.index,
                timeout: self.send_timeout,
            }),
        }
    }

    /// Stop tracking a request; returns its engine if it was in flight.
    fn forget(&self, rid: &str) -> Option<usize> {
        let pending = lock(&self.requests).remove(rid)?;
        self.engines[pending.engine]
            .in_flight
            .fetch_sub(1, Ordering::Relaxed);
        Some(pending.engine)
    }

    fn dispatch(&self, batch: &BatchTokenIDOutSlim) {
        let slot = batch
            .engine_index
            .checked_sub(i64::from(self.first_engine_index))
            .and_then(|slot| usize::try_from(slot).ok())
            .filter(|slot| *slot < self.engines.len());
        match slot {
            Some(slot) => {
                *lock(&self.engines[slot].load) = EngineLoad {
                    num_running: batch.num_running,
                    num_waiting: batch.num_waiting,
                    kv_active_pages: batch.kv_active_pages,
                    kv_total_pages: batch.kv_total_pages,
                };
            }
            None => debug!(
                engine_index = batch.engine_index,
                "output from an unknown engine index"
            ),
        }

        let rows = match batch.requests() {
            Ok(rows) => rows,
            Err(error) => {
                warn!(%error, "dropping a malformed output batch");
                return;
            }
        };
        let mut requests = lock(&self.requests);
        for row in rows {
            let Some(pending) = requests.get(row.rid) else {
                // Cancelled while its output was in flight.
                debug!(rid = row.rid, "output for a request that is not in flight");
                continue;
            };
            // A closed receiver means the stream is being dropped; its drop
            // forgets the request.
            let _ = pending.tx.send(Ok(OutputChunk::from(&row)));
            if row.finish_reason.is_some() {
                let pending = requests.remove(row.rid).expect("present above");
                self.engines[pending.engine]
                    .in_flight
                    .fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    fn dispatch_control(&self, output: ControlOutput) {
        let mut controls = lock(&self.controls);
        let tag = output.tag();
        let Some(waiter) = controls.get_mut(tag) else {
            warn!(tag, "a control reply nobody waits for");
            return;
        };
        waiter.replies.push(output);
        if waiter.replies.len() == self.engines.len() {
            let waiter = controls.remove(tag).expect("present above");
            let _ = waiter.done.send(waiter.replies);
        }
    }

    /// End every stream and refuse new requests.
    fn close(&self, reason: StreamError) {
        lock(&self.closed).get_or_insert_with(|| reason.clone());
        // Dropping a waiter ends its operation with the reason.
        lock(&self.controls).clear();
        let drained: Vec<Pending> = lock(&self.requests).drain().map(|(_, p)| p).collect();
        for pending in drained {
            self.engines[pending.engine]
                .in_flight
                .fetch_sub(1, Ordering::Relaxed);
            let _ = pending.tx.send(Err(reason.clone()));
        }
    }
}

/// A connection to one engine group: every attention-DP rank's engine.
///
/// Dropping the client ends every open stream with
/// [`StreamError::ClientClosed`].
pub struct EngineClient {
    shared: Arc<Shared>,
    tasks: Vec<JoinHandle<()>>,
}

/// The client's sockets, bound and waiting for the engines.
///
/// Binding first lets a launcher bind `tcp://<host>:0`, read the port the OS
/// chose from [`ClientListener::handshake_address`], and start the engines
/// with it.
pub struct ClientListener {
    config: ClientConfig,
    bound: handshake::Bound,
}

impl ClientListener {
    /// Bind the handshake, input and output sockets.
    pub async fn bind(config: ClientConfig) -> Result<Self> {
        let bound = handshake::bind(&config).await?;
        Ok(Self { config, bound })
    }

    /// The handshake address the engines must dial, as bound.
    pub fn handshake_address(&self) -> &str {
        &self.bound.handshake_address
    }

    /// Wait until every engine has registered, within the configured
    /// handshake timeout.
    ///
    /// Must be called within a Tokio runtime; the client's background tasks
    /// run on it.
    pub async fn accept(self) -> Result<EngineClient> {
        let config = self.config;
        let connected = handshake::accept(self.bound, &config).await?;
        let (input, input_recv) = connected.input.split();
        let engines = connected
            .engines
            .into_iter()
            .map(|info| Engine {
                identity: Bytes::copy_from_slice(&ts_wire::engine_identity(info.index)),
                info,
                in_flight: AtomicUsize::new(0),
                load: Mutex::new(EngineLoad::default()),
            })
            .collect();
        let shared = Arc::new(Shared {
            engines,
            first_engine_index: config.first_engine_index,
            routing: config.routing,
            send_timeout: config.send_timeout,
            input,
            requests: Mutex::new(HashMap::new()),
            controls: Mutex::new(HashMap::new()),
            control_turns: Mutex::new(HashMap::new()),
            closed: Mutex::new(None),
            round_robin: AtomicUsize::new(0),
            runtime: Handle::current(),
        });
        let tasks = vec![
            tokio::spawn(receive_outputs(shared.clone(), connected.output)),
            tokio::spawn(drain_input(input_recv)),
        ];
        Ok(EngineClient { shared, tasks })
    }
}

impl EngineClient {
    /// Bind the frontend sockets and wait until every engine has registered.
    /// See [`ClientListener`] to learn the bound address before waiting.
    pub async fn connect(config: ClientConfig) -> Result<Self> {
        ClientListener::bind(config).await?.accept().await
    }

    /// The engines, in index order, with the configuration each registered.
    pub fn engines(&self) -> Vec<EngineInfo> {
        self.shared.engines.iter().map(|e| e.info.clone()).collect()
    }

    /// Each engine's latest reported load, in index order.
    pub fn loads(&self) -> Vec<EngineLoad> {
        self.shared.engines.iter().map(|e| *lock(&e.load)).collect()
    }

    /// This client's unfinished requests per engine, in index order.
    pub fn in_flight(&self) -> Vec<usize> {
        self.shared
            .engines
            .iter()
            .map(|e| e.in_flight.load(Ordering::Relaxed))
            .collect()
    }

    /// Submit a request and return its output stream.
    ///
    /// The request's `rid` must be a single string, unique among requests
    /// in flight. Dropping the stream before its last chunk aborts the
    /// request on the engine.
    pub async fn generate(&self, request: TokenizedGenerateReqInput) -> Result<RequestStream> {
        let rid = match &request.rid {
            Some(StrOrList::One(rid)) => rid.clone(),
            _ => {
                return Err(Error::InvalidRequest(
                    "a generate request needs a single string rid".to_owned(),
                ))
            }
        };
        if let Some(reason) = self.shared.closed() {
            return Err(Error::Unavailable(reason));
        }
        let engine = self.shared.route(&request)?;
        let (tx, rx) = mpsc::unbounded_channel();
        {
            // Register before sending: outputs can arrive before `send`
            // returns.
            let mut requests = lock(&self.shared.requests);
            if requests.contains_key(&rid) {
                return Err(Error::DuplicateRequestId(rid));
            }
            requests.insert(rid.clone(), Pending { engine, tx });
        }
        self.shared.engines[engine]
            .in_flight
            .fetch_add(1, Ordering::Relaxed);
        let frames = EngineRequest::Add(Box::new(request)).to_frames();
        if let Err(error) = self.shared.send(engine, frames).await {
            self.shared.forget(&rid);
            return Err(error);
        }
        Ok(RequestStream {
            shared: self.shared.clone(),
            rid,
            rx,
            done: false,
        })
    }

    /// Run a control operation on every engine and return their replies,
    /// one per engine, in arrival order. Operations of one kind run one at
    /// a time; some (a `wait` pause, a weight load) answer only when done,
    /// so callers bound the wait themselves.
    pub async fn control(&self, request: ControlRequest) -> Result<Vec<ControlOutput>> {
        let tag = request.reply_tag();
        let turn = lock(&self.shared.control_turns)
            .entry(tag)
            .or_default()
            .clone();
        let _turn = turn.lock().await;
        if let Some(reason) = self.shared.closed() {
            return Err(Error::Unavailable(reason));
        }
        let (done, replies) = oneshot::channel();
        lock(&self.shared.controls).insert(
            tag,
            ControlWaiter {
                replies: Vec::new(),
                done,
            },
        );
        let frames = EngineRequest::Control(Box::new(request)).to_frames();
        for engine in 0..self.shared.engines.len() {
            if let Err(error) = self.shared.send(engine, frames.clone()).await {
                lock(&self.shared.controls).remove(tag);
                return Err(error);
            }
        }
        replies.await.map_err(|_| {
            Error::Unavailable(self.shared.closed().unwrap_or(StreamError::ClientClosed))
        })
    }

    /// Ask the engine to abort a request. Its stream then ends with an
    /// `"abort"` chunk. Returns `false` if the request is not in flight.
    pub async fn abort(&self, rid: &str) -> Result<bool> {
        let engine = match lock(&self.shared.requests).get(rid) {
            Some(pending) => pending.engine,
            None => return Ok(false),
        };
        let frames = EngineRequest::Abort(vec![rid.to_owned()]).to_frames();
        self.shared.send(engine, frames).await?;
        Ok(true)
    }
}

impl Drop for EngineClient {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
        self.shared.close(StreamError::ClientClosed);
    }
}

/// The outputs of one request, in order.
pub struct RequestStream {
    shared: Arc<Shared>,
    rid: String,
    rx: mpsc::UnboundedReceiver<StreamItem>,
    done: bool,
}

impl RequestStream {
    /// The request id.
    pub fn rid(&self) -> &str {
        &self.rid
    }

    /// The next chunk; `None` after the chunk with a finish reason or an
    /// error.
    pub async fn next(&mut self) -> Option<StreamItem> {
        if self.done {
            return None;
        }
        let item = self.rx.recv().await;
        self.done = match &item {
            Some(Ok(chunk)) => chunk.finish_reason.is_some(),
            Some(Err(_)) | None => true,
        };
        item
    }
}

impl Drop for RequestStream {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let Some(engine) = self.shared.forget(&self.rid) else {
            return;
        };
        let shared = self.shared.clone();
        let frames = EngineRequest::Abort(vec![self.rid.clone()]).to_frames();
        let rid = self.rid.clone();
        self.shared.runtime.spawn(async move {
            if let Err(error) = shared.send(engine, frames).await {
                warn!(rid, %error, "failed to abort a dropped request");
            }
        });
    }
}

async fn receive_outputs(shared: Arc<Shared>, mut output: PullSocket) {
    loop {
        let message = match output.recv().await {
            Ok(message) => message,
            Err(error) => {
                // A peer error here is an engine's output connection
                // breaking, as when the process is killed.
                warn!(%error, "output socket failed");
                shared.close(StreamError::Transport(error.to_string()));
                return;
            }
        };
        match EngineOutput::from_frames(&message.into_vec()) {
            Ok(EngineOutput::Batch(batch)) => shared.dispatch(&batch),
            Ok(EngineOutput::Control(output)) => shared.dispatch_control(*output),
            Ok(EngineOutput::Dead) => {
                warn!("an engine shut down");
                shared.close(StreamError::EngineDead);
                return;
            }
            Err(error) => warn!(%error, "dropping a malformed output message"),
        }
    }
}

/// Engines send nothing on the input socket after registering. Receiving
/// keeps the socket's peer bookkeeping current.
async fn drain_input(mut input: RouterRecvHalf) {
    loop {
        match input.recv().await {
            Ok(message) => warn!(
                frames = message.len(),
                "unexpected message on the input socket"
            ),
            Err(error) => {
                debug!(%error, "input socket receive ended");
                return;
            }
        }
    }
}
