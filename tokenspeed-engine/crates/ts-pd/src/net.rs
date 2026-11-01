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

use std::{collections::HashMap, sync::mpsc};

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

/// The control sockets of one rank.
pub struct Sockets {
    runtime: Runtime,
    outgoing: tokio::sync::mpsc::UnboundedSender<(String, Vec<Bytes>)>,
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
        let (outgoing, mut queue) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<Bytes>)>();
        runtime.spawn(async move {
            let mut sockets: HashMap<String, PushSocket> = HashMap::new();
            while let Some((endpoint, frames)) = queue.recv().await {
                if !sockets.contains_key(&endpoint) {
                    let mut push = PushSocket::new();
                    if let Err(error) = push.connect(&endpoint).await {
                        tracing::warn!(endpoint, %error, "cannot reach a PD peer");
                        continue;
                    }
                    sockets.insert(endpoint.clone(), push);
                }
                let Ok(message) = ZmqMessage::try_from(frames) else {
                    continue;
                };
                let push = sockets.get_mut(&endpoint).expect("connected above");
                if let Err(error) = push.send(message).await {
                    tracing::warn!(endpoint, %error, "a PD message did not go out");
                    sockets.remove(&endpoint);
                }
            }
        });
        Ok(Self {
            runtime,
            outgoing,
            port,
        })
    }

    /// The PULL socket's port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Push `frames` to `host:port`; never waits.
    pub fn send(&self, host: &str, port: u16, frames: Vec<Bytes>) {
        let endpoint = if host.contains(':') && !host.starts_with('[') {
            format!("tcp://[{host}]:{port}")
        } else {
            format!("tcp://{host}:{port}")
        };
        if self.outgoing.send((endpoint, frames)).is_err() {
            tracing::warn!("the PD socket runtime stopped");
        }
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
        sender.send(
            "127.0.0.1",
            receiver.port(),
            vec![ascii(7), Bytes::from_static(b"x")],
        );
        let frames = received.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(frames, [Bytes::from_static(b"7"), Bytes::from_static(b"x")]);
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
