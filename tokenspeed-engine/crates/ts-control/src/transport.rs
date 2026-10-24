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

use std::{sync::mpsc, time::Duration};

use bytes::Bytes;
use tokio::runtime::Runtime;
use ts_wire::{
    decode_frames, encode_frames,
    handshake::{STATUS_HELLO, STATUS_READY},
    BatchTokenIDOutSlim, ControlOutput, EngineCoreReadyResponse, EngineOutput, EngineRequest,
    HandshakeInitMessage, ReadyMessage,
};
use zeromq::{
    util::PeerIdentity, DealerSocket, PushSocket, Socket, SocketOptions, SocketRecv, SocketSend,
    ZmqMessage,
};

use crate::{
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

/// A connected engine's request and output channels.
pub struct EngineTransport {
    engine_index: u16,
    requests: mpsc::Receiver<EngineRequest>,
    /// The input socket closed: the frontend is gone, and the socket does
    /// not reconnect.
    closed: bool,
    outputs: tokio::sync::mpsc::UnboundedSender<Vec<Bytes>>,
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
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("ts-transport")
            .enable_all()
            .build()?;
        let identity = ts_wire::engine_identity(engine_index);
        let (mut input, output) = runtime.block_on(async {
            let mut handshake = DealerSocket::with_options(options(identity));
            handshake.connect(handshake_address).await?;
            let hello = ReadyMessage {
                status: Some(STATUS_HELLO.to_owned()),
                local: Some(true),
                headless: Some(true),
                parallel_config_hash: None,
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
            input.send(message(encode_frames(&ready))).await?;
            let mut output = PushSocket::new();
            output.connect(output_address).await?;
            tracing::info!(
                engine_index,
                input_address,
                output_address,
                "connected to the frontend"
            );
            Ok::<_, TransportError>((input, output))
        })?;

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
        let (outputs, mut output_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Bytes>>();
        runtime.spawn(async move {
            let mut output = output;
            while let Some(frames) = output_rx.recv().await {
                if let Err(error) = output.send(message(frames)).await {
                    tracing::warn!(%error, "output send failed");
                    return;
                }
            }
        });
        Ok(Self {
            engine_index,
            requests,
            closed: false,
            outputs,
            runtime: Some(runtime),
        })
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
        first.into_iter().chain(self.requests.try_iter()).collect()
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
        let frames = EngineOutput::Control(Box::new(output)).to_frames();
        if self.outputs.send(frames).is_err() {
            tracing::warn!("output channel closed; dropping a control reply");
        }
    }
}

impl Drop for EngineTransport {
    fn drop(&mut self) {
        // Tell the frontend this engine is gone, so it fails the streams
        // instead of waiting on them.
        let _ = self.outputs.send(EngineOutput::Dead.to_frames());
        if let Some(runtime) = self.runtime.take() {
            // Give the sentinel a moment to leave before the sockets close.
            runtime.block_on(async { tokio::time::sleep(Duration::from_millis(100)).await });
            runtime.shutdown_timeout(Duration::from_secs(1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_closed_input_is_noticed_after_its_last_messages() {
        let (request_tx, requests) = mpsc::channel();
        let (outputs, _sent) = tokio::sync::mpsc::unbounded_channel();
        let mut transport = EngineTransport {
            engine_index: 0,
            requests,
            closed: false,
            outputs,
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
