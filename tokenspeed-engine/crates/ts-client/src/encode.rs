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

//! EPD encode submission: the push link to an encode node's request intake.
//!
//! An encode node's first rank binds a PULL socket (`ts-host` in encode
//! mode, where `epd/encode_host.py` names it `intake`) and relays what
//! arrives to the node's other ranks. A frontend connects a PUSH socket to it
//! and sends [`EncodeRequest`] messages: the request's msgpack payload with
//! its tensors as out-of-band frames, no type byte. The link has no handshake
//! and no reply: the ranks embed the item and ship the embedding to the
//! prefill ranks waiting under the request's `bootstrap_room`; the frontend
//! learns nothing more on this socket, and a submission that the ranks refuse
//! ends on the prefill side.

use std::time::Duration;

use tokio::{sync::Mutex, time::timeout};
use ts_wire::{encode_frames, EncodeRequest};
use zeromq::{PushSocket, Socket, SocketSend, ZmqMessage};

use crate::error::{Error, Result};

/// Where an encode node takes requests. Every field selects behavior, so
/// none has a default.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// The intake's address, `tcp://<host>:<port>`, as the encode node's
    /// first rank bound it.
    pub intake_address: String,
    /// Limit for handing one request to the socket. A stalled intake fails
    /// the submission instead of blocking the caller.
    pub send_timeout: Duration,
}

/// A connection to one encode node's request intake.
///
/// Submissions are fire-and-forget; see the module documentation.
pub struct EncodeClient {
    config: EncodeConfig,
    socket: Mutex<PushSocket>,
}

impl EncodeClient {
    /// Connect to the intake.
    ///
    /// Must be called within a Tokio runtime; the socket runs on it.
    pub async fn connect(config: EncodeConfig) -> Result<Self> {
        let mut socket = PushSocket::new();
        socket.connect(&config.intake_address).await?;
        tracing::info!(intake = %config.intake_address, "connected to the encode intake");
        Ok(Self {
            config,
            socket: Mutex::new(socket),
        })
    }

    /// The intake address this client is connected to.
    pub fn intake_address(&self) -> &str {
        &self.config.intake_address
    }

    /// Submit a request. The encode ranks take exactly one item per request;
    /// a request with another count is refused here, since the ranks answer
    /// nothing on this link.
    pub async fn submit(&self, request: EncodeRequest) -> Result<()> {
        if request.items.len() != 1 {
            return Err(Error::InvalidRequest(format!(
                "an encode request carries one item, got {}",
                request.items.len()
            )));
        }
        let message = ZmqMessage::try_from(encode_frames(&request))
            .expect("an encoded request has at least its payload frame");
        // One bound for the whole submission: the wait for the socket behind
        // other submitters and the send itself.
        let submission = async { self.socket.lock().await.send(message).await };
        match timeout(self.config.send_timeout, submission).await {
            Ok(sent) => Ok(sent?),
            Err(_) => Err(Error::IntakeSendTimeout {
                timeout: self.config.send_timeout,
            }),
        }
    }
}
