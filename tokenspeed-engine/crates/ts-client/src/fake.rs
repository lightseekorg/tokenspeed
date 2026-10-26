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

//! A stand-in engine for a frontend's tests (feature `fake-engine`).
//!
//! [`FakeEngine`] speaks the engine side of the wire over real ZMQ sockets
//! the way `ts-host` does: it dials the frontend's handshake socket,
//! completes HELLO, INIT and READY, registers with the
//! [`EngineCoreReadyResponse`] the test chooses, then hands the test every
//! request typed ([`EngineRequest`]) and sends what the test scripts
//! ([`EngineOutput`]). [`FakeEngine::serve`] runs a trivial scheduler instead,
//! for tests that only need tokens to flow. [`FakeEncodeIntake`] binds an
//! encode node's request intake and hands the test every [`EncodeRequest`]
//! pushed to it. Nothing here needs a device.
//!
//! Every wait ends after [`FAKE_DEADLINE`] with an error naming the step, so
//! a frontend that never answers costs a test a message, not a hung run.

use std::{
    collections::HashSet,
    future::Future,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use bytes::Bytes;
use tokio::{sync::mpsc, task::JoinHandle, time::timeout};
use ts_wire::{
    control::{FlushCacheReqOutput, IsSchedulerPausedReqOutput},
    decode_frames, encode_frames,
    handshake::{STATUS_HELLO, STATUS_READY},
    BatchTokenIDOutSlim, ControlOutput, ControlRequest, EncodeRequest, EngineCoreReadyResponse,
    EngineOutput, EngineRequest, HandshakeInitMessage, ReadyMessage, StrOrList,
};
use zeromq::{
    util::PeerIdentity, DealerSocket, PullSocket, PushSocket, Socket, SocketOptions, SocketRecv,
    SocketSend, ZmqMessage,
};

use crate::{
    client::EngineLoad,
    error::{Error, Result},
};

/// How long any wait of a fake lasts before it fails naming the step.
pub const FAKE_DEADLINE: Duration = Duration::from_secs(30);

/// A unique `ipc://` address per call, so tests run in parallel without
/// port races.
pub fn ipc_address(name: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ts-fake-engine-{}-{name}-{n}.sock",
        std::process::id()
    ));
    format!("ipc://{}", path.display())
}

async fn within<T>(step: &str, future: impl Future<Output = T>) -> Result<T> {
    timeout(FAKE_DEADLINE, future)
        .await
        .map_err(|_| Error::Handshake(format!("fake engine: no {step} within {FAKE_DEADLINE:?}")))
}

fn options(index: u16) -> SocketOptions {
    let mut options = SocketOptions::default();
    options.peer_identity(
        PeerIdentity::try_from(ts_wire::engine_identity(index).to_vec()).expect("two bytes"),
    );
    options
}

fn message(frames: Vec<Bytes>) -> ZmqMessage {
    ZmqMessage::try_from(frames).expect("a message has frames")
}

/// One request's row of a step's output, for [`batch`].
#[derive(Debug, Clone, PartialEq)]
pub struct Row<'a> {
    /// Request id.
    pub rid: &'a str,
    /// Token ids generated this step.
    pub output_ids: Vec<i32>,
    /// The finish type (`"stop"`, `"length"`, `"abort"`), or `None` while
    /// the request runs.
    pub finish_reason: Option<&'a str>,
    /// Prompt length.
    pub prompt_tokens: i64,
    /// Tokens generated so far.
    pub completion_tokens: i64,
}

/// A slim output batch of `rows` from engine `engine_index`, reporting
/// `load`; no logprobs, no finish details.
pub fn batch(engine_index: u16, rows: Vec<Row<'_>>, load: EngineLoad) -> BatchTokenIDOutSlim {
    let n = rows.len();
    let mut out = BatchTokenIDOutSlim {
        rids: Some(Vec::with_capacity(n)),
        output_ids: Vec::with_capacity(n),
        finished_reasons: Vec::with_capacity(n),
        prompt_tokens: Vec::with_capacity(n),
        completion_tokens: Vec::with_capacity(n),
        cached_tokens: vec![0; n],
        output_token_logprobs_val: vec![Vec::new(); n],
        output_token_logprobs_idx: vec![Vec::new(); n],
        engine_index: i64::from(engine_index),
        num_running: load.num_running,
        num_waiting: load.num_waiting,
        kv_active_pages: load.kv_active_pages,
        kv_total_pages: load.kv_total_pages,
        finish_reason_details: vec![None; n],
        spec_verify_ct: Vec::new(),
        accept_draft_tokens: Vec::new(),
        input_token_logprobs_val: Vec::new(),
        input_token_logprobs_idx: Vec::new(),
        input_top_logprobs_val: Vec::new(),
        input_top_logprobs_idx: Vec::new(),
        output_top_logprobs_val: Vec::new(),
        output_top_logprobs_idx: Vec::new(),
    };
    for row in rows {
        out.rids
            .as_mut()
            .expect("set above")
            .push(row.rid.to_owned());
        out.output_ids.push(row.output_ids);
        out.finished_reasons
            .push(row.finish_reason.unwrap_or("").to_owned());
        out.prompt_tokens.push(row.prompt_tokens);
        out.completion_tokens.push(row.completion_tokens);
    }
    out
}

/// An engine that completed the handshake and registered; the test drives
/// it with [`FakeEngine::recv`] and [`FakeEngine::send`].
pub struct FakeEngine {
    index: u16,
    init: HandshakeInitMessage,
    input: DealerSocket,
    output: PushSocket,
}

impl FakeEngine {
    /// A READY response for attention-DP rank `dp_rank` of `dp_size`, with
    /// a 4096-token window and the other fields at their defaults.
    pub fn ready_response(dp_rank: i64, dp_size: i64) -> EngineCoreReadyResponse {
        EngineCoreReadyResponse {
            max_model_len: 4096,
            data_parallel_size: dp_size,
            data_parallel_rank: dp_rank,
            ..Default::default()
        }
    }

    /// Dial `handshake_address` as engine `index`, complete the handshake
    /// and register with `ready`.
    ///
    /// Must be called within a Tokio runtime; the sockets run on it.
    pub async fn connect(
        handshake_address: &str,
        index: u16,
        ready: EngineCoreReadyResponse,
    ) -> Result<Self> {
        let mut handshake = DealerSocket::with_options(options(index));
        handshake.connect(handshake_address).await?;
        let hello = ReadyMessage {
            status: Some(STATUS_HELLO.to_owned()),
            local: Some(true),
            headless: Some(true),
            parallel_config_hash: None,
        };
        handshake.send(message(encode_frames(&hello))).await?;
        let init = within("INIT from the frontend", handshake.recv())
            .await??
            .into_vec();
        let Some(payload) = init.last() else {
            return Err(Error::Handshake("fake engine: an empty INIT".to_owned()));
        };
        let init: HandshakeInitMessage = decode_frames(std::slice::from_ref(payload))?;
        let ready_status = ReadyMessage {
            status: Some(STATUS_READY.to_owned()),
            ..hello
        };
        handshake
            .send(message(encode_frames(&ready_status)))
            .await?;
        let (Some(input_address), Some(output_address)) = (
            init.addresses.inputs.first(),
            init.addresses.outputs.first(),
        ) else {
            return Err(Error::Handshake(
                "fake engine: INIT carried no input or output address".to_owned(),
            ));
        };
        let mut input = DealerSocket::with_options(options(index));
        input.connect(input_address).await?;
        input.send(message(encode_frames(&ready))).await?;
        let mut output = PushSocket::new();
        output.connect(output_address).await?;
        Ok(Self {
            index,
            init,
            input,
            output,
        })
    }

    /// The engine index this fake registered as.
    pub fn index(&self) -> u16 {
        self.index
    }

    /// The INIT message the frontend sent.
    pub fn init(&self) -> &HandshakeInitMessage {
        &self.init
    }

    /// The next request from the frontend.
    pub async fn recv(&mut self) -> Result<EngineRequest> {
        let frames = within("request from the frontend", self.input.recv())
            .await??
            .into_vec();
        Ok(EngineRequest::from_frames(&frames)?)
    }

    /// Send an output message.
    pub async fn send(&mut self, output: &EngineOutput) -> Result<()> {
        within(
            "send of an output",
            self.output.send(message(output.to_frames())),
        )
        .await??;
        Ok(())
    }

    /// Send one step's outputs.
    pub async fn send_batch(&mut self, batch: BatchTokenIDOutSlim) -> Result<()> {
        self.send(&EngineOutput::Batch(Box::new(batch))).await
    }

    /// Announce shutdown with the sentinel, as a stopping engine does; the
    /// frontend then ends every stream.
    pub async fn shut_down(&mut self) -> Result<()> {
        self.send(&EngineOutput::Dead).await
    }

    /// Serve with a trivial scheduler on the current runtime:
    ///
    /// * a request with `max_new_tokens = n` gets `n` one-token outputs, the
    ///   last finished with `"length"`;
    /// * a request without `max_new_tokens` gets one token and then runs
    ///   until it is aborted, which finishes it with `"abort"`;
    /// * generated token `k` of a request is `1000 * engine_index + k`, so a
    ///   test can tell which engine served it;
    /// * a flush succeeds on engine 0 only, nothing is ever paused, and any
    ///   other control operation ends the serving task.
    pub fn serve(self) -> ServingFake {
        let (received_tx, received) = mpsc::unbounded_channel();
        let (kill, kill_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            if let Err(error) = self.run(received_tx, kill_rx).await {
                tracing::warn!(%error, "the serving fake engine stopped");
            }
        });
        ServingFake {
            received,
            kill,
            task,
        }
    }

    async fn run(
        mut self,
        received: mpsc::UnboundedSender<EngineRequest>,
        mut kill: mpsc::UnboundedReceiver<()>,
    ) -> Result<()> {
        let index = self.index;
        let mut running: HashSet<String> = HashSet::new();
        loop {
            let frames = tokio::select! {
                message = self.input.recv() => message?.into_vec(),
                _ = kill.recv() => return self.shut_down().await,
            };
            let request = EngineRequest::from_frames(&frames)?;
            let _ = received.send(request.clone());
            match request {
                EngineRequest::Add(request) => {
                    let Some(StrOrList::One(rid)) = request.rid else {
                        return Err(Error::InvalidRequest(
                            "fake engine: a request without a single rid".to_owned(),
                        ));
                    };
                    let prompt = request.input_ids.map_or(0, |ids| ids.len() as i64);
                    let steps = request.sampling_params.max_new_tokens;
                    running.insert(rid.clone());
                    for k in 1..=steps.unwrap_or(1) {
                        let finished = steps == Some(k);
                        if finished {
                            running.remove(&rid);
                        }
                        let row = Row {
                            rid: &rid,
                            output_ids: vec![1000 * i32::from(index) + k as i32],
                            finish_reason: finished.then_some("length"),
                            prompt_tokens: prompt,
                            completion_tokens: k,
                        };
                        let batch = batch(index, vec![row], Self::load(&running));
                        self.send_batch(batch).await?;
                    }
                }
                EngineRequest::Abort(rids) => {
                    for rid in rids {
                        if running.remove(&rid) {
                            let row = Row {
                                rid: &rid,
                                output_ids: Vec::new(),
                                finish_reason: Some("abort"),
                                prompt_tokens: 0,
                                completion_tokens: 2,
                            };
                            let batch = batch(index, vec![row], Self::load(&running));
                            self.send_batch(batch).await?;
                        }
                    }
                }
                EngineRequest::Control(request) => {
                    let reply = match *request {
                        ControlRequest::FlushCache(_) => {
                            ControlOutput::FlushCache(FlushCacheReqOutput {
                                rid: None,
                                http_worker_ipc: None,
                                success: index == 0,
                            })
                        }
                        ControlRequest::IsSchedulerPaused(_) => {
                            ControlOutput::IsSchedulerPaused(IsSchedulerPausedReqOutput {
                                rid: None,
                                http_worker_ipc: None,
                                is_paused: false,
                            })
                        }
                        other => {
                            return Err(Error::InvalidRequest(format!(
                                "fake engine: no scripted reply to {}",
                                other.tag()
                            )))
                        }
                    };
                    self.send(&EngineOutput::Control(Box::new(reply))).await?;
                }
            }
        }
    }

    fn load(running: &HashSet<String>) -> EngineLoad {
        EngineLoad {
            num_running: running.len() as i64,
            num_waiting: 0,
            kv_active_pages: 0,
            kv_total_pages: 100,
        }
    }
}

/// Connect and serve in one task, so a test can start the engine before the
/// frontend's handshake completes (see [`FakeEngine::serve`] for the rules).
pub fn spawn_serving(
    handshake_address: &str,
    index: u16,
    ready: EngineCoreReadyResponse,
) -> ServingFake {
    let (received_tx, received) = mpsc::unbounded_channel();
    let (kill, kill_rx) = mpsc::unbounded_channel();
    let handshake_address = handshake_address.to_owned();
    let task = tokio::spawn(async move {
        let served = match FakeEngine::connect(&handshake_address, index, ready).await {
            Ok(engine) => engine.run(received_tx, kill_rx).await,
            Err(error) => Err(error),
        };
        if let Err(error) = served {
            tracing::warn!(%error, "the serving fake engine stopped");
        }
    });
    ServingFake {
        received,
        kill,
        task,
    }
}

/// A fake engine serving by itself (see [`FakeEngine::serve`]).
pub struct ServingFake {
    /// Every request the engine received, in order.
    pub received: mpsc::UnboundedReceiver<EngineRequest>,
    kill: mpsc::UnboundedSender<()>,
    task: JoinHandle<()>,
}

impl ServingFake {
    /// Announce shutdown with the sentinel and stop serving.
    pub fn shut_down(&self) {
        let _ = self.kill.send(());
    }

    /// The next request the engine received.
    pub async fn next_received(&mut self) -> Result<EngineRequest> {
        within("request at the serving fake engine", self.received.recv())
            .await?
            .ok_or_else(|| Error::Handshake("fake engine: the serving task ended".to_owned()))
    }
}

impl Drop for ServingFake {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// An encode node's request intake: the PULL socket the first encode rank
/// binds, receiving what an [`crate::EncodeClient`] submits.
pub struct FakeEncodeIntake {
    endpoint: String,
    socket: PullSocket,
}

impl FakeEncodeIntake {
    /// Bind `address`; `tcp://<host>:0` picks a free port, and
    /// [`FakeEncodeIntake::endpoint`] is the address as bound.
    pub async fn bind(address: &str) -> Result<Self> {
        let mut socket = PullSocket::new();
        let endpoint = socket.bind(address).await?.to_string();
        Ok(Self { endpoint, socket })
    }

    /// The intake's address as bound.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The next encode request pushed to the intake, decoded from its frames
    /// as an encode rank decodes it.
    pub async fn recv(&mut self) -> Result<EncodeRequest> {
        let frames = within("encode request at the intake", self.socket.recv())
            .await??
            .into_vec();
        Ok(decode_frames(&frames)?)
    }
}
