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

//! The P/D control sockets: one PULL socket a rank's peers push to, and
//! PUSH sockets to the peers, one per endpoint (`DisaggManagerBase`).
//!
//! Socket I/O runs on a small Tokio runtime; the PD threads exchange
//! messages with it through channels.

use std::{
    collections::{HashMap, HashSet},
    sync::{mpsc, Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use tokio::runtime::Runtime;
use zeromq::{PullSocket, PushSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

/// A socket failure at startup.
#[derive(Debug, thiserror::Error)]
pub enum NetError {
    /// A socket operation failed.
    #[error("zmq: {0}")]
    Zmq(#[from] zeromq::ZmqError),
    /// The runtime did not start.
    #[error("starting the PD socket runtime: {0}")]
    Runtime(#[from] std::io::Error),
    /// The bound endpoint carries no port.
    #[error("the bound endpoint {0} has no port")]
    Port(String),
}

const OUTGOING_CAPACITY: usize = 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(5);

/// A control message that could not be queued. Callers must fail the
/// associated request, or retry an idempotent notification on a later poll.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    #[error("the PD outgoing queue for {0} is full")]
    Full(String),
    #[error("the PD outgoing queue for {0} stopped")]
    Closed(String),
}

struct Endpoint {
    outgoing: tokio::sync::mpsc::Sender<Message>,
    pending: Arc<Mutex<HashSet<Vec<Bytes>>>>,
}

/// A coalesced notification remains pending through both queueing and I/O.
/// Dropping a rejected job or a cancelled task releases that reservation too.
struct Message {
    frames: Vec<Bytes>,
    pending: Option<Arc<Mutex<HashSet<Vec<Bytes>>>>>,
}

impl Drop for Message {
    fn drop(&mut self) {
        if let Some(pending) = &self.pending {
            pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&self.frames);
        }
    }
}

async fn send_messages(endpoint: String, mut queue: tokio::sync::mpsc::Receiver<Message>) {
    let mut socket = None;
    while let Some(message) = queue.recv().await {
        if socket.is_none() {
            let mut push = PushSocket::new();
            match tokio::time::timeout(IO_TIMEOUT, push.connect(&endpoint)).await {
                Ok(Ok(())) => socket = Some(push),
                result => {
                    tracing::warn!(endpoint, ?result, "cannot reach a PD peer");
                    continue;
                }
            }
        }
        let Ok(frames) = ZmqMessage::try_from(message.frames.clone()) else {
            continue;
        };
        let push = socket.as_mut().expect("connected above");
        match tokio::time::timeout(IO_TIMEOUT, push.send(frames)).await {
            Ok(Ok(())) => {}
            result => {
                tracing::warn!(endpoint, ?result, "a PD message did not go out");
                socket = None;
            }
        }
    }
}

/// The control sockets of one rank. Each destination has its own bounded
/// FIFO and I/O task, so a failed peer cannot delay another endpoint.
pub struct Sockets {
    runtime: Runtime,
    outgoing: Mutex<HashMap<String, Endpoint>>,
    port: u16,
}

impl Sockets {
    /// Bind the PULL socket on `host` at a port the OS picks; every message
    /// it receives goes to `incoming`, frames in order.
    pub fn start(host: &str, incoming: mpsc::Sender<Vec<Bytes>>) -> Result<Self, NetError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("ts-pd-net")
            .enable_all()
            .build()?;
        let address = if host.contains(':') {
            format!("tcp://[{host}]:0")
        } else {
            format!("tcp://{host}:0")
        };
        let (pull, port) = runtime.block_on(async {
            let mut pull = PullSocket::new();
            let bound = pull.bind(&address).await?.to_string();
            let port = bound
                .rsplit(':')
                .next()
                .and_then(|port| port.parse().ok())
                .ok_or(NetError::Port(bound.clone()))?;
            Ok::<_, NetError>((pull, port))
        })?;
        runtime.spawn(async move {
            let mut pull = pull;
            loop {
                match pull.recv().await {
                    Ok(message) => {
                        if incoming.send(message.into_vec()).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        tracing::warn!(%error, "PD control socket closed");
                        return;
                    }
                }
            }
        });
        Ok(Self {
            runtime,
            outgoing: Mutex::new(HashMap::new()),
            port,
        })
    }

    /// The PULL socket's port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Queue a message without waiting for network I/O or queue capacity.
    pub fn send(&self, host: &str, port: u16, frames: Vec<Bytes>) -> Result<(), SendError> {
        self.enqueue(host, port, frames, false)
    }

    /// Queue an idempotent notification at most once while an identical
    /// notification is queued or being sent. A later poll can retry it.
    pub fn send_coalesced(
        &self,
        host: &str,
        port: u16,
        frames: Vec<Bytes>,
    ) -> Result<(), SendError> {
        self.enqueue(host, port, frames, true)
    }

    fn enqueue(
        &self,
        host: &str,
        port: u16,
        frames: Vec<Bytes>,
        coalesce: bool,
    ) -> Result<(), SendError> {
        let address = if host.contains(':') && !host.starts_with('[') {
            format!("tcp://[{host}]:{port}")
        } else {
            format!("tcp://{host}:{port}")
        };
        let mut outgoing = self.outgoing.lock().unwrap_or_else(|e| e.into_inner());
        let endpoint = outgoing.entry(address.clone()).or_insert_with(|| {
            let (sender, receiver) = tokio::sync::mpsc::channel(OUTGOING_CAPACITY);
            self.runtime.spawn(send_messages(address.clone(), receiver));
            Endpoint {
                outgoing: sender,
                pending: Arc::new(Mutex::new(HashSet::new())),
            }
        });
        let pending = if coalesce {
            if !endpoint
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(frames.clone())
            {
                return Ok(());
            }
            Some(endpoint.pending.clone())
        } else {
            None
        };
        let message = Message { frames, pending };
        endpoint.outgoing.try_send(message).map_err(|error| {
            let error = match error {
                tokio::sync::mpsc::error::TrySendError::Full(_) => SendError::Full(address),
                tokio::sync::mpsc::error::TrySendError::Closed(_) => SendError::Closed(address),
            };
            tracing::warn!(%error, "could not queue a PD control message");
            error
        })
    }
}

impl Drop for Sockets {
    fn drop(&mut self) {
        // Let queued messages (a failure notice) leave before the sockets
        // close.
        self.runtime
            .block_on(async { tokio::time::sleep(std::time::Duration::from_millis(50)).await });
    }
}

/// An ASCII frame.
pub fn ascii(value: impl ToString) -> Bytes {
    Bytes::from(value.to_string())
}

/// A canonical unsigned ASCII integer frame (`_bounded_ascii_uint`).
pub fn parse_uint(frame: &[u8], maximum: u64) -> Option<u64> {
    let digits = maximum.to_string().len();
    if frame.is_empty()
        || frame.len() > digits
        || !frame.iter().all(u8::is_ascii_digit)
        || (frame.len() > 1 && frame[0] == b'0')
    {
        return None;
    }
    let value: u64 = std::str::from_utf8(frame).ok()?.parse().ok()?;
    (value <= maximum).then_some(value)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_pushed_message_reaches_the_pull_socket() {
        let (incoming, received) = mpsc::channel();
        let receiver = Sockets::start("127.0.0.1", incoming).unwrap();
        let (unused, _never) = mpsc::channel();
        let sender = Sockets::start("127.0.0.1", unused).unwrap();
        sender
            .send(
                "127.0.0.1",
                receiver.port(),
                vec![ascii(7), Bytes::from_static(b"x")],
            )
            .unwrap();
        let frames = received.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(frames, [Bytes::from_static(b"7"), Bytes::from_static(b"x")]);
    }

    fn accept_with_deadline(listener: &std::net::TcpListener) -> std::net::TcpStream {
        listener.set_nonblocking(true).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            match listener.accept() {
                Ok((stream, _)) => return stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "sender never connected"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept: {error}"),
            }
        }
    }

    #[test]
    fn a_blocked_destination_does_not_delay_other_control_sockets() {
        // Accept TCP but never complete ZMQ's handshake.
        let stalled = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = stalled.local_addr().unwrap().port();
        let (incoming, received) = mpsc::channel();
        let sender = Sockets::start("127.0.0.1", incoming).unwrap();
        sender
            .send("127.0.0.1", port, vec![ascii("blocked")])
            .unwrap();
        let _held = accept_with_deadline(&stalled);
        let (incoming, delivered) = mpsc::channel();
        let healthy = Sockets::start("127.0.0.1", incoming).unwrap();
        for label in ["registration", "manifest", "cancel"] {
            sender
                .send("127.0.0.1", healthy.port(), vec![ascii(label)])
                .unwrap();
        }
        healthy
            .send("127.0.0.1", sender.port(), vec![ascii("inbound")])
            .unwrap();
        assert_eq!(
            received.recv_timeout(Duration::from_secs(1)).unwrap(),
            vec![ascii("inbound")]
        );
        for label in ["registration", "manifest", "cancel"] {
            assert_eq!(
                delivered.recv_timeout(Duration::from_secs(1)).unwrap(),
                vec![ascii(label)]
            );
        }
    }

    #[test]
    fn queued_cancellations_are_coalesced_and_endpoint_queues_are_bounded() {
        let stalled = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = stalled.local_addr().unwrap().port();
        let (incoming, _received) = mpsc::channel();
        let sender = Sockets::start("127.0.0.1", incoming).unwrap();
        sender
            .send_coalesced("127.0.0.1", port, vec![ascii("cancel")])
            .unwrap();
        let _held = accept_with_deadline(&stalled);
        for _ in 0..OUTGOING_CAPACITY * 2 {
            sender
                .send_coalesced("127.0.0.1", port, vec![ascii("cancel")])
                .unwrap();
        }
        for sequence in 0..OUTGOING_CAPACITY {
            sender
                .send("127.0.0.1", port, vec![ascii(sequence)])
                .unwrap();
        }
        assert!(matches!(
            sender.send("127.0.0.1", port, vec![ascii("overflow")]),
            Err(SendError::Full(_))
        ));
        let (incoming, delivered) = mpsc::channel();
        let healthy = Sockets::start("127.0.0.1", incoming).unwrap();
        sender
            .send("127.0.0.1", healthy.port(), vec![ascii("healthy")])
            .unwrap();
        assert_eq!(
            delivered.recv_timeout(Duration::from_secs(1)).unwrap(),
            vec![ascii("healthy")]
        );
    }

    #[test]
    fn unsigned_frames_are_canonical() {
        assert_eq!(parse_uint(b"42", 100), Some(42));
        assert_eq!(parse_uint(b"042", 100), None);
        assert_eq!(parse_uint(b"101", 100), None);
        assert_eq!(parse_uint(b"", 100), None);
        assert_eq!(parse_uint(b"4a", 100), None);
    }
}
