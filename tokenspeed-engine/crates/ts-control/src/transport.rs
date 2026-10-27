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

//! The engine's side of the msgpack ZMQ wire: the frontend binds, the engine
//! connects, completes the handshake (see [`ts_wire::handshake`]) and then
//! receives requests and pushes outputs.
//!
//! Socket I/O runs on its own small Tokio runtime; the loop exchanges
//! messages with it through channels and never blocks on the network.
//!
//! An EPD encode server's first rank dials the same way ([`EncodeIntake`])
//! and takes encode requests only.
//!
//! A control call ([`ControlCall`]) is turned into the engine's own control
//! operation here ([`crate::calls`]) and answered here, once its operation's
//! reply comes back: the loop sees only its own operations. The loop answers
//! each kind of operation in the order it received them, so the transport
//! keeps, per kind, who asked for each one in that order.

use std::{
    collections::{HashMap, VecDeque},
    sync::mpsc,
    time::{Duration, Instant},
};

use bytes::Bytes;
use tokio::runtime::Runtime;
use ts_wire::{
    control::ControlReplySlim,
    decode_frames, encode_frames,
    frames::{REQ_TYPE_CONTROL, REQ_TYPE_ENCODE},
    handshake::{STATUS_HELLO, STATUS_READY},
    BatchTokenIDOutSlim, ControlOutput, ControlRequest, EngineCoreReadyResponse, EngineOutput,
    EngineRequest, HandshakeInitMessage, ReadyMessage,
};
use zeromq::{
    util::PeerIdentity, DealerSocket, PushSocket, Socket, SocketOptions, SocketRecv, SocketSend,
    ZmqMessage,
};

use crate::{
    calls::{self, FLUSH_RETRY_INTERVAL},
    encode::Intake,
    event_loop::{Frontend, Load},
    output::StreamedOutput,
};

/// A transport failure during the handshake.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// A socket operation failed.
    #[error("zmq: {0}")]
    Zmq(#[from] zeromq::ZmqError),
    /// A handshake message did not decode.
    #[error("wire: {0}")]
    Wire(#[from] ts_wire::Error),
    /// The frontend did not answer HELLO in time.
    #[error("the frontend sent no INIT within {0:?}")]
    InitTimeout(Duration),
    /// The frontend's INIT named no input or output address.
    #[error("INIT carried no input or output address")]
    NoAddresses,
    /// The I/O runtime could not start.
    #[error("starting the transport runtime: {0}")]
    Runtime(#[from] std::io::Error),
}

fn options(identity: [u8; 2]) -> SocketOptions {
    let mut options = SocketOptions::default();
    options.peer_identity(PeerIdentity::try_from(identity.to_vec()).expect("two bytes"));
    options
}

fn message(frames: Vec<Bytes>) -> ZmqMessage {
    ZmqMessage::try_from(frames).expect("a message has frames")
}

fn runtime(name: &str) -> Result<Runtime, TransportError> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name(name)
        .enable_all()
        .build()?)
}

/// Complete the handshake as engine `engine_index`, registering with
/// `ready`, and return the connected input and output sockets.
async fn dial(
    handshake_address: &str,
    engine_index: u16,
    ready: &EngineCoreReadyResponse,
    init_timeout: Duration,
) -> Result<(DealerSocket, PushSocket), TransportError> {
    let identity = ts_wire::engine_identity(engine_index);
    let mut handshake = DealerSocket::with_options(options(identity));
    handshake.connect(handshake_address).await?;
    let hello = ReadyMessage {
        status: Some(STATUS_HELLO.to_owned()),
        local: Some(true),
        headless: Some(true),
        parallel_config_hash: None,
        engine_count: Some(ready.data_parallel_size),
    };
    handshake.send(message(encode_frames(&hello))).await?;
    let init = tokio::time::timeout(init_timeout, handshake.recv())
        .await
        .map_err(|_| TransportError::InitTimeout(init_timeout))??
        .into_vec();
    let init: HandshakeInitMessage = decode_frames(&init[init.len() - 1..])?;
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
        return Err(TransportError::NoAddresses);
    };

    let mut input = DealerSocket::with_options(options(identity));
    input.connect(input_address).await?;
    input.send(message(encode_frames(ready))).await?;
    let mut output = PushSocket::new();
    output.connect(output_address).await?;
    tracing::info!(
        engine_index,
        input_address,
        output_address,
        "connected to the frontend"
    );
    Ok((input, output))
}

/// Push every message from the returned channel on `output`.
fn spawn_output(
    runtime: &Runtime,
    mut output: PushSocket,
) -> tokio::sync::mpsc::UnboundedSender<Vec<Bytes>> {
    let (outputs, mut output_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Bytes>>();
    runtime.spawn(async move {
        while let Some(frames) = output_rx.recv().await {
            if let Err(error) = output.send(message(frames)).await {
                tracing::warn!(%error, "output send failed");
                return;
            }
        }
    });
    outputs
}

/// Tell the frontend this engine is gone, so it fails what waits on it, and
/// close the sockets.
fn hang_up(outputs: &tokio::sync::mpsc::UnboundedSender<Vec<Bytes>>, runtime: Option<Runtime>) {
    let _ = outputs.send(EngineOutput::Dead.to_frames());
    if let Some(runtime) = runtime {
        // Give the sentinel a moment to leave before the sockets close.
        runtime.block_on(async { tokio::time::sleep(Duration::from_millis(100)).await });
        runtime.shutdown_timeout(Duration::from_secs(1));
    }
}

/// Who asked for a control operation the loop has: the reply goes back as
/// the operation's own reply, or as a call's.
#[derive(Debug, Clone, PartialEq)]
enum Asker {
    Operation,
    Call {
        call_id: i64,
        /// A flush call's: a refusal is retried until then.
        flush_until: Option<Instant>,
        arrived: Instant,
        /// To ask again with.
        request: Box<ControlRequest>,
    },
}

/// A refused flush call, asked again at `due`.
struct Retry {
    due: Instant,
    asker: Asker,
}

/// A connected engine's request and output channels.
pub struct EngineTransport {
    engine_index: u16,
    requests: mpsc::Receiver<EngineRequest>,
    /// The input socket closed: the frontend is gone, and the socket does
    /// not reconnect.
    closed: bool,
    outputs: tokio::sync::mpsc::UnboundedSender<Vec<Bytes>>,
    /// By reply tag, who asked for each operation the loop has of that
    /// kind, in the order it received them.
    askers: HashMap<&'static str, VecDeque<Asker>>,
    retries: Vec<Retry>,
    runtime: Option<Runtime>,
}

impl EngineTransport {
    /// Connect as engine `engine_index` to the frontend's handshake socket
    /// and register with `ready`.
    pub fn connect(
        handshake_address: &str,
        engine_index: u16,
        ready: EngineCoreReadyResponse,
        init_timeout: Duration,
    ) -> Result<Self, TransportError> {
        let runtime = runtime("ts-transport")?;
        let (mut input, output) =
            runtime.block_on(dial(handshake_address, engine_index, &ready, init_timeout))?;

        let (request_tx, requests) = mpsc::channel();
        runtime.spawn(async move {
            loop {
                let frames = match input.recv().await {
                    Ok(message) => message.into_vec(),
                    Err(error) => {
                        tracing::warn!(%error, "input socket closed");
                        return;
                    }
                };
                match EngineRequest::from_frames(&frames) {
                    Ok(request) => {
                        if request_tx.send(request).is_err() {
                            return;
                        }
                    }
                    // A malformed message (a version-skewed frontend) must not
                    // stop the engine; drop it and keep receiving.
                    Err(error) => tracing::warn!(%error, frames = frames.len(), "dropping a malformed input message"),
                }
            }
        });
        let outputs = spawn_output(&runtime, output);
        Ok(Self {
            engine_index,
            requests,
            closed: false,
            outputs,
            askers: HashMap::new(),
            retries: Vec::new(),
            runtime: Some(runtime),
        })
    }

    /// Hand `request` to the loop on behalf of `asker`.
    fn deliver(&mut self, request: ControlRequest, asker: Asker, to: &mut Vec<EngineRequest>) {
        self.askers
            .entry(request.reply_tag())
            .or_default()
            .push_back(asker);
        to.push(EngineRequest::Control(Box::new(request)));
    }

    /// What the loop gets for `received`: control calls become the engine's
    /// own operations, and a call that names none is answered at once.
    fn translate(&mut self, received: Vec<EngineRequest>) -> Vec<EngineRequest> {
        let now = Instant::now();
        let mut delivered = Vec::with_capacity(received.len());
        let due: Vec<Retry>;
        (due, self.retries) = std::mem::take(&mut self.retries)
            .into_iter()
            .partition(|retry| retry.due <= now);
        for retry in due {
            let Asker::Call { request, .. } = &retry.asker else {
                unreachable!("only calls are retried");
            };
            self.deliver((**request).clone(), retry.asker, &mut delivered);
        }
        for message in received {
            match message {
                EngineRequest::Control(request) => {
                    self.deliver(*request, Asker::Operation, &mut delivered)
                }
                EngineRequest::Call(call) => match calls::translate(&call, now) {
                    Ok(translated) => {
                        let asker = Asker::Call {
                            call_id: call.call_id,
                            flush_until: translated.flush_until,
                            arrived: now,
                            request: Box::new(translated.request.clone()),
                        };
                        self.deliver(translated.request, asker, &mut delivered);
                    }
                    Err(message) => {
                        tracing::warn!(call_id = call.call_id, %message, "refusing a control call");
                        self.send_reply(call.call_id, false, Some(message));
                    }
                },
                EngineRequest::Encode(_) => tracing::warn!(
                    "dropping an encode request: this engine runs a scheduler, not an encoder"
                ),
                other => delivered.push(other),
            }
        }
        delivered
    }

    fn send_reply(&self, call_id: i64, success: bool, message: Option<String>) {
        let reply = ControlReplySlim {
            call_id,
            success,
            message,
            engine_index: i64::from(self.engine_index),
        };
        if self
            .outputs
            .send(EngineOutput::ControlReply(reply).to_frames())
            .is_err()
        {
            tracing::warn!("output channel closed; dropping a control reply");
        }
    }

    /// The input task ended (its socket failed): note it once.
    fn input_closed(&mut self) {
        if !self.closed {
            self.closed = true;
            tracing::error!(
                engine_index = self.engine_index,
                "the frontend's input socket closed; the engine stops"
            );
        }
    }
}

/// One step's outputs as the output wire carries them.
pub fn output_batch(
    outputs: Vec<StreamedOutput>,
    engine_index: u16,
    load: Load,
) -> BatchTokenIDOutSlim {
    let n = outputs.len();
    let mut batch = BatchTokenIDOutSlim {
        rids: Some(Vec::with_capacity(n)),
        output_ids: Vec::with_capacity(n),
        finished_reasons: Vec::with_capacity(n),
        prompt_tokens: Vec::with_capacity(n),
        completion_tokens: Vec::with_capacity(n),
        cached_tokens: Vec::with_capacity(n),
        output_token_logprobs_val: Vec::with_capacity(n),
        output_token_logprobs_idx: Vec::with_capacity(n),
        engine_index: i64::from(engine_index),
        num_running: load.num_running,
        num_waiting: load.num_waiting,
        kv_active_pages: load.kv_active_pages,
        kv_total_pages: load.kv_total_pages,
        finish_reason_details: Vec::with_capacity(n),
        spec_verify_ct: Vec::new(),
        accept_draft_tokens: Vec::new(),
        input_token_logprobs_val: Vec::with_capacity(n),
        input_token_logprobs_idx: Vec::with_capacity(n),
        input_top_logprobs_val: Vec::with_capacity(n),
        input_top_logprobs_idx: Vec::with_capacity(n),
        output_top_logprobs_val: Vec::with_capacity(n),
        output_top_logprobs_idx: Vec::with_capacity(n),
    };
    for output in outputs {
        batch.rids.as_mut().expect("set above").push(output.rid);
        // The logprob ids are the tokens the logprobs belong to.
        let logprob_ids = if output.output_logprobs.is_empty() {
            Vec::new()
        } else {
            output.output_ids.clone()
        };
        batch.output_ids.push(output.output_ids);
        batch
            .finished_reasons
            .push(output.finish_reason.unwrap_or("").to_owned());
        batch.prompt_tokens.push(output.prompt_tokens);
        batch.completion_tokens.push(output.completion_tokens);
        batch.cached_tokens.push(output.cached_tokens);
        batch.output_token_logprobs_val.push(output.output_logprobs);
        batch.output_token_logprobs_idx.push(logprob_ids);
        batch.finish_reason_details.push(output.finish_details);
        if let Some((verify_ct, accepted)) = output.spec {
            batch.spec_verify_ct.push(verify_ct);
            batch.accept_draft_tokens.push(accepted);
        }
        let (input_values, input_ids, input_top) = match output.input_logprobs {
            Some(logprobs) => (logprobs.values, logprobs.ids, logprobs.top),
            None => (Vec::new(), Vec::new(), None),
        };
        batch.input_token_logprobs_val.push(input_values);
        batch.input_token_logprobs_idx.push(input_ids);
        let (top_values, top_ids) = input_top.unwrap_or_default();
        batch.input_top_logprobs_val.push(top_values);
        batch.input_top_logprobs_idx.push(top_ids);
        let (top_values, top_ids) = output.output_top_logprobs.unwrap_or_default();
        batch.output_top_logprobs_val.push(top_values);
        batch.output_top_logprobs_idx.push(top_ids);
    }
    // Each optional column is per request only in a frame where some
    // request has an entry.
    fn frame_column<T>(column: &mut Vec<Vec<T>>) {
        if column.iter().all(Vec::is_empty) {
            column.clear();
        }
    }
    frame_column(&mut batch.input_token_logprobs_val);
    frame_column(&mut batch.input_token_logprobs_idx);
    frame_column(&mut batch.input_top_logprobs_val);
    frame_column(&mut batch.input_top_logprobs_idx);
    frame_column(&mut batch.output_top_logprobs_val);
    frame_column(&mut batch.output_top_logprobs_idx);
    batch
}

impl Frontend for EngineTransport {
    fn receive(&mut self, wait: Option<Duration>) -> Vec<EngineRequest> {
        // A retry falls due within the wait: wait no longer than that.
        let wait = match self.retries.iter().map(|retry| retry.due).min() {
            Some(due) => wait.map(|wait| wait.min(due.saturating_duration_since(Instant::now()))),
            None => wait,
        };
        // What arrived before the input closed is still delivered first.
        let first = match wait {
            Some(timeout) => match self.requests.recv_timeout(timeout) {
                Ok(request) => Some(request),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.input_closed();
                    None
                }
            },
            None => match self.requests.try_recv() {
                Ok(request) => Some(request),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.input_closed();
                    None
                }
            },
        };
        let received = first.into_iter().chain(self.requests.try_iter()).collect();
        self.translate(received)
    }

    fn closed(&self) -> bool {
        self.closed
    }

    fn send(&mut self, outputs: Vec<StreamedOutput>, load: Load) {
        let frames = EngineOutput::Batch(Box::new(output_batch(outputs, self.engine_index, load)))
            .to_frames();
        if self.outputs.send(frames).is_err() {
            tracing::warn!("output channel closed; dropping a batch");
        }
    }

    fn reply(&mut self, output: ControlOutput) {
        let asker = self
            .askers
            .get_mut(output.tag())
            .and_then(VecDeque::pop_front)
            .unwrap_or(Asker::Operation);
        let Asker::Call {
            call_id,
            flush_until,
            arrived,
            ..
        } = &asker
        else {
            let frames = EngineOutput::Control(Box::new(output)).to_frames();
            if self.outputs.send(frames).is_err() {
                tracing::warn!("output channel closed; dropping a control reply");
            }
            return;
        };
        let (success, message) = output.outcome();
        let now = Instant::now();
        match flush_until {
            Some(until) if !success && now < *until => {
                // Not idle yet: ask again shortly, until the call's time is up.
                self.retries.push(Retry {
                    due: now + FLUSH_RETRY_INTERVAL,
                    asker,
                });
            }
            Some(_) if !success => {
                self.send_reply(*call_id, false, Some(calls::flush_refusal(now - *arrived)))
            }
            _ => self.send_reply(*call_id, success, message),
        }
    }
}

impl Drop for EngineTransport {
    fn drop(&mut self) {
        hang_up(&self.outputs, self.runtime.take());
    }
}

/// An EPD encode server's connection to its frontend: the first encode
/// rank's. It takes encode requests, each as its frames without the type
/// frame (`[payload, aux...]`), for the encode loop to decode and relay;
/// a control call is answered as one this server does not take, and any
/// other message is dropped.
pub struct EncodeIntake {
    messages: mpsc::Receiver<Vec<Bytes>>,
    outputs: tokio::sync::mpsc::UnboundedSender<Vec<Bytes>>,
    runtime: Option<Runtime>,
}

impl EncodeIntake {
    /// Connect as engine `engine_index` to the frontend's handshake socket
    /// and register with `ready`.
    pub fn connect(
        handshake_address: &str,
        engine_index: u16,
        ready: EngineCoreReadyResponse,
        init_timeout: Duration,
    ) -> Result<Self, TransportError> {
        let runtime = runtime("ts-encode-intake")?;
        let (mut input, output) =
            runtime.block_on(dial(handshake_address, engine_index, &ready, init_timeout))?;
        let outputs = spawn_output(&runtime, output);
        let (incoming, messages) = mpsc::channel();
        let replies = outputs.clone();
        runtime.spawn(async move {
            loop {
                let mut frames = match input.recv().await {
                    Ok(message) => message.into_vec(),
                    Err(error) => {
                        tracing::warn!(%error, "the encode request socket closed");
                        return;
                    }
                };
                match frames.first().map(|frame| frame.as_ref()) {
                    Some([REQ_TYPE_ENCODE]) => {
                        frames.remove(0);
                        if incoming.send(frames).is_err() {
                            return;
                        }
                    }
                    Some([REQ_TYPE_CONTROL]) => match EngineRequest::from_frames(&frames) {
                        Ok(EngineRequest::Call(call)) => {
                            let reply = ControlReplySlim {
                                call_id: call.call_id,
                                success: false,
                                message: Some(format!(
                                    "{}: an EPD encode server takes no control calls",
                                    call.method
                                )),
                                engine_index: i64::from(engine_index),
                            };
                            let _ = replies.send(EngineOutput::ControlReply(reply).to_frames());
                        }
                        _ => tracing::warn!(
                            "dropping a control message: an encode server takes none"
                        ),
                    },
                    _ => tracing::warn!(
                        frames = frames.len(),
                        "dropping a message: an EPD encode server takes encode requests only"
                    ),
                }
            }
        });
        Ok(Self {
            messages,
            outputs,
            runtime: Some(runtime),
        })
    }
}

impl Intake for EncodeIntake {
    fn receive(&mut self, wait: Duration) -> Vec<Vec<Bytes>> {
        let first = if wait.is_zero() {
            self.messages.try_recv().ok()
        } else {
            self.messages.recv_timeout(wait).ok()
        };
        first.into_iter().chain(self.messages.try_iter()).collect()
    }
}

impl Drop for EncodeIntake {
    fn drop(&mut self) {
        hang_up(&self.outputs, self.runtime.take());
    }
}

#[cfg(test)]
mod tests {
    use ts_wire::{
        control::{ControlCall, FlushCacheReqInput, FlushCacheReqOutput, ProfileReqOutput},
        Value,
    };

    use super::*;

    type Sent = tokio::sync::mpsc::UnboundedReceiver<Vec<Bytes>>;

    fn transport(engine_index: u16) -> (EngineTransport, mpsc::Sender<EngineRequest>, Sent) {
        let (request_tx, requests) = mpsc::channel();
        let (outputs, sent) = tokio::sync::mpsc::unbounded_channel();
        let transport = EngineTransport {
            engine_index,
            requests,
            closed: false,
            outputs,
            askers: HashMap::new(),
            retries: Vec::new(),
            runtime: None,
        };
        (transport, request_tx, sent)
    }

    fn call(call_id: i64, method: &str, args: Vec<Value>) -> EngineRequest {
        EngineRequest::Call(Box::new(ControlCall {
            call_id,
            method: method.to_owned(),
            args,
        }))
    }

    fn flushed(success: bool) -> ControlOutput {
        ControlOutput::FlushCache(FlushCacheReqOutput {
            rid: None,
            http_worker_ipc: None,
            success,
        })
    }

    fn sent(sent: &mut Sent) -> Vec<EngineOutput> {
        let mut outputs = Vec::new();
        while let Ok(frames) = sent.try_recv() {
            outputs.push(EngineOutput::from_frames(&frames).unwrap());
        }
        outputs
    }

    fn reply(call_id: i64, success: bool, message: Option<&str>) -> EngineOutput {
        EngineOutput::ControlReply(ControlReplySlim {
            call_id,
            success,
            message: message.map(str::to_owned),
            engine_index: 2,
        })
    }

    #[test]
    fn a_call_runs_as_the_engines_operation_and_is_answered_under_its_id() {
        let (mut transport, requests, mut outputs) = transport(2);
        requests.send(call(7, "stop_profile", Vec::new())).unwrap();
        let [EngineRequest::Control(request)] =
            <[_; 1]>::try_from(transport.receive(None)).unwrap()
        else {
            panic!("the call reaches the loop as an operation");
        };
        assert!(matches!(*request, ControlRequest::Profile(_)));
        transport.reply(ControlOutput::Profile(ProfileReqOutput {
            rid: None,
            http_worker_ipc: None,
            success: true,
            message: "Stop profiling.".to_owned(),
        }));
        assert_eq!(
            sent(&mut outputs),
            [reply(7, true, Some("Stop profiling."))]
        );
    }

    #[test]
    fn a_call_the_engine_cannot_run_is_answered_at_once() {
        let (mut transport, requests, mut outputs) = transport(2);
        requests
            .send(call(9, "update_weights", Vec::new()))
            .unwrap();
        assert!(transport.receive(None).is_empty());
        let [EngineOutput::ControlReply(refusal)] = <[_; 1]>::try_from(sent(&mut outputs)).unwrap()
        else {
            panic!("one reply");
        };
        assert_eq!((refusal.call_id, refusal.success), (9, false));
        assert!(refusal.message.unwrap().contains("update_weights"));
    }

    #[test]
    fn operations_and_calls_of_one_kind_are_answered_in_order() {
        let (mut transport, requests, mut outputs) = transport(2);
        let operation = || {
            EngineRequest::Control(Box::new(ControlRequest::FlushCache(FlushCacheReqInput {
                rid: None,
                http_worker_ipc: None,
            })))
        };
        requests.send(operation()).unwrap();
        requests.send(call(3, "flush_cache", Vec::new())).unwrap();
        requests.send(operation()).unwrap();
        assert_eq!(transport.receive(None).len(), 3);
        for _ in 0..3 {
            transport.reply(flushed(true));
        }
        assert_eq!(
            sent(&mut outputs),
            [
                EngineOutput::Control(Box::new(flushed(true))),
                reply(3, true, None),
                EngineOutput::Control(Box::new(flushed(true))),
            ]
        );
    }

    #[test]
    fn a_refused_flush_call_is_asked_again_until_its_time_is_up() {
        let (mut transport, requests, mut outputs) = transport(2);
        requests.send(call(4, "flush_cache", Vec::new())).unwrap();
        assert_eq!(transport.receive(None).len(), 1);
        transport.reply(flushed(false));
        assert!(
            sent(&mut outputs).is_empty(),
            "a refusal within the time is retried"
        );
        // The wait ends when the retry falls due, not at the frontend's wait.
        let started = Instant::now();
        let [EngineRequest::Control(retry)] =
            <[_; 1]>::try_from(transport.receive(Some(Duration::from_secs(10)))).unwrap()
        else {
            panic!("the flush is asked again");
        };
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(matches!(*retry, ControlRequest::FlushCache(_)));
        transport.reply(flushed(true));
        assert_eq!(sent(&mut outputs), [reply(4, true, None)]);

        requests.send(call(5, "flush_cache", Vec::new())).unwrap();
        assert_eq!(transport.receive(None).len(), 1);
        let deadline = Instant::now() + Duration::from_secs(5);
        let refusal = loop {
            transport.reply(flushed(false));
            if let [EngineOutput::ControlReply(refusal)] = sent(&mut outputs).as_slice() {
                break refusal.clone();
            }
            assert_eq!(transport.receive(Some(Duration::from_secs(1))).len(), 1);
            assert!(Instant::now() < deadline, "the retries end");
        };
        assert_eq!((refusal.call_id, refusal.success), (5, false));
        assert!(refusal.message.unwrap().contains("not idle"));
    }

    #[test]
    fn a_closed_input_is_noticed_after_its_last_messages() {
        let (request_tx, requests) = mpsc::channel();
        let (outputs, _sent) = tokio::sync::mpsc::unbounded_channel();
        let mut transport = EngineTransport {
            engine_index: 0,
            requests,
            closed: false,
            outputs,
            askers: HashMap::new(),
            retries: Vec::new(),
            runtime: None,
        };
        let abort = EngineRequest::Abort(vec!["r0".to_owned()]);
        request_tx.send(abort.clone()).unwrap();
        // The input task ended.
        drop(request_tx);
        assert_eq!(transport.receive(Some(Duration::from_secs(10))), [abort]);
        assert!(!transport.closed());
        // A wait returns at once: nothing can arrive any more.
        let started = std::time::Instant::now();
        assert!(transport.receive(Some(Duration::from_secs(10))).is_empty());
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(transport.closed());
    }
}
