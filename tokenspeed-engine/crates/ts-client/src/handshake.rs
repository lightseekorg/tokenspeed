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

//! The frontend side of the startup handshake (see [`ts_wire::handshake`]).

use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use tokio::time::{timeout_at, Instant};
use tracing::{debug, info};
use ts_wire::{
    decode_bytes, encode_frames,
    handshake::{STATUS_HELLO, STATUS_READY},
    Dict, EngineCoreReadyResponse, HandshakeAddresses, HandshakeInitMessage, ReadyMessage,
};
use zeromq::{PullSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::{
    config::ClientConfig,
    error::{Error, Result},
};

/// An engine that completed the handshake.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// The engine's index (its routing identity).
    pub index: u16,
    /// The configuration it registered with.
    pub ready: EngineCoreReadyResponse,
}

/// The frontend sockets, bound, before any engine has connected.
pub(crate) struct Bound {
    /// The handshake address as bound, with the port the OS chose for `:0`.
    pub handshake_address: String,
    handshake: RouterSocket,
    input: RouterSocket,
    output: PullSocket,
    /// The INIT message sent to every engine.
    init: Vec<Bytes>,
    expected: BTreeSet<u16>,
}

/// Bound sockets with every expected engine registered.
pub(crate) struct Connected {
    pub engines: Vec<EngineInfo>,
    pub input: RouterSocket,
    pub output: PullSocket,
}

/// Bind the frontend sockets.
pub(crate) async fn bind(config: &ClientConfig) -> Result<Bound> {
    let last_index = u32::from(config.first_engine_index) + u32::from(config.num_engines);
    if config.num_engines == 0 || last_index > u32::from(u16::MAX) + 1 {
        return Err(Error::Handshake(format!(
            "engine indices {}..{last_index} are empty or exceed the two-byte identity",
            config.first_engine_index
        )));
    }
    let expected: BTreeSet<u16> = (config.first_engine_index..=(last_index - 1) as u16).collect();

    let mut handshake = RouterSocket::new();
    let handshake_address = handshake.bind(&config.handshake_address).await?.to_string();
    let mut input = RouterSocket::new();
    let input_address = input.bind(&config.input_address).await?.to_string();
    let mut output = PullSocket::new();
    let output_address = output.bind(&config.output_address).await?.to_string();
    info!(
        handshake = %handshake_address,
        input = %input_address,
        output = %output_address,
        engines = ?expected,
        "frontend sockets bound"
    );

    let init = encode_frames(&HandshakeInitMessage {
        addresses: HandshakeAddresses {
            inputs: vec![input_address],
            outputs: vec![output_address],
            coordinator_input: None,
            coordinator_output: None,
            frontend_stats_publish_address: None,
        },
        parallel_config: Dict::default(),
    });
    Ok(Bound {
        handshake_address,
        handshake,
        input,
        output,
        init,
        expected,
    })
}

/// Complete the handshake with every expected engine.
pub(crate) async fn accept(bound: Bound, config: &ClientConfig) -> Result<Connected> {
    let deadline = Instant::now() + config.handshake_timeout;
    let Bound {
        mut handshake,
        mut input,
        output,
        init,
        expected,
        ..
    } = bound;

    // HELLO, then INIT from us, then READY, per engine; engines interleave.
    let mut awaiting_hello = expected.clone();
    let mut awaiting_ready = expected.clone();
    while !awaiting_ready.is_empty() {
        let message =
            recv_before(&mut handshake, deadline, config, "READY", &awaiting_ready).await?;
        let (index, payload) = engine_message(message, &expected)?;
        let status: ReadyMessage = decode_bytes(&payload)?;
        match status.status.as_deref() {
            Some(STATUS_HELLO) => {
                if !awaiting_hello.remove(&index) {
                    return Err(Error::Handshake(format!("engine {index} sent HELLO twice")));
                }
                let mut reply = ZmqMessage::from(Bytes::copy_from_slice(&index.to_le_bytes()));
                for frame in &init {
                    reply.push_back(frame.clone());
                }
                handshake.send(reply).await?;
                debug!(engine = index, "sent INIT");
            }
            Some(STATUS_READY) => {
                if awaiting_hello.contains(&index) || !awaiting_ready.remove(&index) {
                    return Err(Error::Handshake(format!(
                        "engine {index} sent READY out of order"
                    )));
                }
                debug!(engine = index, "engine READY");
            }
            other => {
                return Err(Error::Handshake(format!(
                    "engine {index} sent unknown status {other:?}"
                )))
            }
        }
    }

    // Each engine then registers its configuration on the input socket.
    let mut registered: BTreeMap<u16, EngineCoreReadyResponse> = BTreeMap::new();
    while registered.len() < expected.len() {
        let pending: BTreeSet<u16> = expected
            .iter()
            .filter(|index| !registered.contains_key(index))
            .copied()
            .collect();
        let message = recv_before(&mut input, deadline, config, "registration", &pending).await?;
        let (index, payload) = engine_message(message, &expected)?;
        let ready: EngineCoreReadyResponse = decode_bytes(&payload)?;
        check_rank(config, index, &ready)?;
        if registered.insert(index, ready).is_some() {
            return Err(Error::Handshake(format!("engine {index} registered twice")));
        }
    }
    info!(engines = registered.len(), "all engines registered");

    Ok(Connected {
        engines: registered
            .into_iter()
            .map(|(index, ready)| EngineInfo { index, ready })
            .collect(),
        input,
        output,
    })
}

async fn recv_before(
    socket: &mut RouterSocket,
    deadline: Instant,
    config: &ClientConfig,
    stage: &'static str,
    pending: &BTreeSet<u16>,
) -> Result<ZmqMessage> {
    match timeout_at(deadline, socket.recv()).await {
        Ok(message) => Ok(message?),
        Err(_) => Err(Error::HandshakeTimeout {
            stage,
            pending: pending.iter().copied().collect(),
            timeout: config.handshake_timeout,
        }),
    }
}

/// Split a ROUTER message into the sending engine's index and its payload.
fn engine_message(message: ZmqMessage, expected: &BTreeSet<u16>) -> Result<(u16, Bytes)> {
    let frames = message.into_vec();
    let [identity, payload] = frames.as_slice() else {
        return Err(Error::Handshake(format!(
            "expected an identity and a payload, got {} frames",
            frames.len()
        )));
    };
    let index = match identity.as_ref() {
        [low, high] => u16::from_le_bytes([*low, *high]),
        other => {
            return Err(Error::Handshake(format!(
                "peer identity {other:02x?} is not a two-byte engine index"
            )))
        }
    };
    if !expected.contains(&index) {
        return Err(Error::Handshake(format!(
            "unexpected engine {index}; expecting {expected:?}"
        )));
    }
    Ok((index, payload.clone()))
}

/// Every engine must report the DP layout the client was configured for.
fn check_rank(config: &ClientConfig, index: u16, ready: &EngineCoreReadyResponse) -> Result<()> {
    let rank = i64::from(index - config.first_engine_index);
    if ready.data_parallel_size != i64::from(config.num_engines) || ready.data_parallel_rank != rank
    {
        return Err(Error::Handshake(format!(
            "engine {index} reports DP rank {} of {}, expected rank {rank} of {}",
            ready.data_parallel_rank, ready.data_parallel_size, config.num_engines
        )));
    }
    Ok(())
}
