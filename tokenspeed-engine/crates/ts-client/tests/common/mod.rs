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

//! A stand-in engine that speaks the engine side of the wire over real ZMQ
//! sockets, the way `ts-host` does, with a trivial scheduler:
//!
//! * a request with `max_new_tokens = n` gets `n` one-token outputs, the
//!   last finished with `"length"`;
//! * a request without `max_new_tokens` gets one token and then runs until
//!   it is aborted, which finishes it with `"abort"`.
//!
//! Generated token `k` of a request is `1000 * engine_index + k`, so tests
//! can tell which engine served a request.

#![allow(dead_code)]

use std::{
    collections::HashSet,
    sync::atomic::{AtomicU64, Ordering},
};

use bytes::Bytes;
use tokio::{sync::mpsc, task::JoinHandle};
use ts_wire::{
    control::{FlushCacheReqOutput, IsSchedulerPausedReqOutput},
    decode_frames, encode_frames,
    handshake::STATUS_HELLO,
    handshake::STATUS_READY,
    BatchTokenIDOutSlim, ControlOutput, ControlRequest, EngineCoreReadyResponse, EngineOutput,
    EngineRequest, HandshakeInitMessage, ReadyMessage, StrOrList,
};
use zeromq::{
    util::PeerIdentity, DealerSocket, PushSocket, Socket, SocketOptions, SocketRecv, SocketSend,
    ZmqMessage,
};

/// A unique `ipc://` address per call, so tests run in parallel without
/// port races.
pub fn ipc_address(name: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ts-client-test-{}-{name}-{n}.sock",
        std::process::id()
    ));
    format!("ipc://{}", path.display())
}

/// A running fake engine.
pub struct FakeEngine {
    /// Every message the engine received on its input socket, in order.
    pub received: mpsc::UnboundedReceiver<EngineRequest>,
    kill: mpsc::UnboundedSender<()>,
    task: JoinHandle<()>,
}

impl FakeEngine {
    /// Announce shutdown with the sentinel and stop serving.
    pub fn shut_down(&self) {
        let _ = self.kill.send(());
    }

    /// The next message the engine received.
    pub async fn next_received(&mut self) -> EngineRequest {
        tokio::time::timeout(std::time::Duration::from_secs(5), self.received.recv())
            .await
            .expect("engine received a message in time")
            .expect("engine running")
    }
}

impl Drop for FakeEngine {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn options(index: u16) -> SocketOptions {
    let mut options = SocketOptions::default();
    options.peer_identity(PeerIdentity::try_from(index.to_le_bytes().to_vec()).unwrap());
    options
}

fn single(payload: Vec<Bytes>) -> ZmqMessage {
    ZmqMessage::try_from(payload).unwrap()
}

/// Start an engine that dials `handshake_address` as engine `index`,
/// reporting attention-DP rank `dp_rank` of `dp_size`.
pub fn start(handshake_address: &str, index: u16, dp_rank: i64, dp_size: i64) -> FakeEngine {
    let (received_tx, received) = mpsc::unbounded_channel();
    let (kill, kill_rx) = mpsc::unbounded_channel();
    let handshake_address = handshake_address.to_owned();
    let task = tokio::spawn(async move {
        run(
            handshake_address,
            index,
            dp_rank,
            dp_size,
            received_tx,
            kill_rx,
        )
        .await;
    });
    FakeEngine {
        received,
        kill,
        task,
    }
}

async fn run(
    handshake_address: String,
    index: u16,
    dp_rank: i64,
    dp_size: i64,
    received: mpsc::UnboundedSender<EngineRequest>,
    mut kill: mpsc::UnboundedReceiver<()>,
) {
    let mut handshake = DealerSocket::with_options(options(index));
    handshake.connect(&handshake_address).await.unwrap();
    let hello = ReadyMessage {
        status: Some(STATUS_HELLO.to_owned()),
        local: Some(true),
        headless: Some(true),
        parallel_config_hash: None,
    };
    handshake.send(single(encode_frames(&hello))).await.unwrap();
    let init = handshake.recv().await.unwrap().into_vec();
    let init: HandshakeInitMessage = decode_frames(&init[init.len() - 1..]).unwrap();
    let ready = ReadyMessage {
        status: Some(STATUS_READY.to_owned()),
        ..hello
    };
    handshake.send(single(encode_frames(&ready))).await.unwrap();

    let mut input = DealerSocket::with_options(options(index));
    input.connect(&init.addresses.inputs[0]).await.unwrap();
    let registration = EngineCoreReadyResponse {
        max_model_len: 4096,
        data_parallel_size: dp_size,
        data_parallel_rank: dp_rank,
        ..Default::default()
    };
    input
        .send(single(encode_frames(&registration)))
        .await
        .unwrap();
    let mut output = PushSocket::new();
    output.connect(&init.addresses.outputs[0]).await.unwrap();

    let mut running: HashSet<String> = HashSet::new();
    loop {
        let frames = tokio::select! {
            message = input.recv() => message.unwrap().into_vec(),
            _ = kill.recv() => {
                output.send(single(EngineOutput::Dead.to_frames())).await.unwrap();
                return;
            }
        };
        let request = EngineRequest::from_frames(&frames).unwrap();
        let _ = received.send(request.clone());
        match request {
            EngineRequest::Add(request) => {
                let Some(StrOrList::One(rid)) = request.rid else {
                    panic!("request without a rid");
                };
                let prompt = request.input_ids.map_or(0, |ids| ids.len() as i64);
                let steps = request.sampling_params.max_new_tokens;
                running.insert(rid.clone());
                for k in 1..=steps.unwrap_or(1) {
                    let finished = steps == Some(k);
                    if finished {
                        running.remove(&rid);
                    }
                    let batch = step(
                        index,
                        &rid,
                        k,
                        finished.then_some("length"),
                        prompt,
                        &running,
                    );
                    output.send(single(batch)).await.unwrap();
                }
            }
            EngineRequest::Abort(rids) => {
                for rid in rids {
                    if running.remove(&rid) {
                        let batch = step(index, &rid, 2, Some("abort"), 0, &running);
                        output.send(single(batch)).await.unwrap();
                    }
                }
            }
            // A flush succeeds on engine 0 only; nothing else is paused.
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
                    other => panic!("no fake reply to {}", other.tag()),
                };
                output
                    .send(single(EngineOutput::Control(Box::new(reply)).to_frames()))
                    .await
                    .unwrap();
            }
        }
    }
}

fn step(
    index: u16,
    rid: &str,
    k: i64,
    finish: Option<&str>,
    prompt: i64,
    running: &HashSet<String>,
) -> Vec<Bytes> {
    let token = 1000 * i32::from(index) + k as i32;
    EngineOutput::Batch(Box::new(BatchTokenIDOutSlim {
        rids: Some(vec![rid.to_owned()]),
        output_ids: vec![if finish == Some("abort") {
            vec![]
        } else {
            vec![token]
        }],
        finished_reasons: vec![finish.unwrap_or("").to_owned()],
        prompt_tokens: vec![prompt],
        completion_tokens: vec![k],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![vec![]],
        output_token_logprobs_idx: vec![vec![]],
        engine_index: i64::from(index),
        num_running: running.len() as i64,
        num_waiting: 0,
        kv_active_pages: 0,
        kv_total_pages: 100,
        finish_reason_details: vec![None],
        spec_verify_ct: vec![],
        accept_draft_tokens: vec![],
        input_token_logprobs_val: vec![],
        input_token_logprobs_idx: vec![],
        input_top_logprobs_val: vec![],
        input_top_logprobs_idx: vec![],
        output_top_logprobs_val: vec![],
        output_top_logprobs_idx: vec![],
    }))
    .to_frames()
}
