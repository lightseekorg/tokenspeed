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
    sync::{mpsc, Arc, Mutex, Weak},
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
const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

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

type Endpoints = Mutex<HashMap<String, Endpoint>>;

/// The sending task owns its registry entry. Identity checking also covers a
/// closed channel being replaced before the old task finishes unwinding.
struct EndpointLease {
    address: String,
    registry: Weak<Endpoints>,
    pending: Arc<Mutex<HashSet<Vec<Bytes>>>>,
}

impl EndpointLease {
    fn remove(&self, endpoints: &mut HashMap<String, Endpoint>) {
        if endpoints
            .get(&self.address)
            .is_some_and(|endpoint| Arc::ptr_eq(&endpoint.pending, &self.pending))
        {
            endpoints.remove(&self.address);
        }
    }

    async fn next(
        &self,
        queue: &mut tokio::sync::mpsc::Receiver<Message>,
        idle: Duration,
    ) -> Option<Message> {
        match tokio::time::timeout(idle, queue.recv()).await {
            Ok(message) => message,
            Err(_) => {
                let registry = self.registry.upgrade()?;
                let mut endpoints = registry.lock().unwrap_or_else(|error| error.into_inner());
                // Enqueue uses this same lock. A message arriving at the idle
                // deadline either keeps this FIFO alive or starts a fresh one.
                match queue.try_recv() {
                    Ok(message) => Some(message),
                    Err(_) => {
                        queue.close();
                        self.remove(&mut endpoints);
                        None
                    }
                }
            }
        }
    }
}

impl Drop for EndpointLease {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            self.remove(&mut registry.lock().unwrap_or_else(|error| error.into_inner()));
        }
    }
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

async fn send_messages(
    lease: EndpointLease,
    mut queue: tokio::sync::mpsc::Receiver<Message>,
    io_timeout: Duration,
    idle_timeout: Duration,
) {
    let mut socket = None;
    while let Some(message) = lease.next(&mut queue, idle_timeout).await {
        if socket.is_none() {
            let mut push = PushSocket::new();
            match tokio::time::timeout(io_timeout, push.connect(&lease.address)).await {
                Ok(Ok(())) => socket = Some(push),
                result => {
                    tracing::warn!(
                        endpoint = lease.address,
                        ?result,
                        discarded = queue.len(),
                        "cannot reach a PD peer; discarding its control backlog"
                    );
                    // Delivery is best effort. Retire this failed connection
                    // as a unit instead of spending one timeout per queued job.
                    // Callers retry cancellation on their next poll.
                    queue.close();
                    return;
                }
            }
        }
        let Ok(frames) = ZmqMessage::try_from(message.frames.clone()) else {
            continue;
        };
        let push = socket.as_mut().expect("connected above");
        match tokio::time::timeout(io_timeout, push.send(frames)).await {
            Ok(Ok(())) => {}
            result => {
                tracing::warn!(
                    endpoint = lease.address,
                    ?result,
                    discarded = queue.len(),
                    "a PD message did not go out; discarding its control backlog"
                );
                queue.close();
                return;
            }
        }
    }
}

/// The control sockets of one rank. Each destination has its own bounded
/// FIFO and I/O task, so a failed peer cannot delay another endpoint.
pub struct Sockets {
    runtime: Runtime,
    outgoing: Arc<Endpoints>,
    io_timeout: Duration,
    idle_timeout: Duration,
    port: u16,
}

impl Sockets {
    /// Bind the PULL socket on `host` at a port the OS picks; every message
    /// it receives goes to `incoming`, frames in order.
    pub fn start(host: &str, incoming: mpsc::Sender<Vec<Bytes>>) -> Result<Self, NetError> {
        Self::with_timeouts(host, incoming, IO_TIMEOUT, IDLE_TIMEOUT)
    }

    fn with_timeouts(
        host: &str,
        incoming: mpsc::Sender<Vec<Bytes>>,
        io_timeout: Duration,
        idle_timeout: Duration,
    ) -> Result<Self, NetError> {
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
            outgoing: Arc::new(Mutex::new(HashMap::new())),
            io_timeout,
            idle_timeout,
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
        if outgoing
            .get(&address)
            .is_some_and(|endpoint| endpoint.outgoing.is_closed())
        {
            outgoing.remove(&address);
        }
        let endpoint = outgoing.entry(address.clone()).or_insert_with(|| {
            let (sender, receiver) = tokio::sync::mpsc::channel(OUTGOING_CAPACITY);
            let pending = Arc::new(Mutex::new(HashSet::new()));
            let lease = EndpointLease {
                address: address.clone(),
                registry: Arc::downgrade(&self.outgoing),
                pending: pending.clone(),
            };
            self.runtime.spawn(send_messages(
                lease,
                receiver,
                self.io_timeout,
                self.idle_timeout,
            ));
            Endpoint {
                outgoing: sender,
                pending,
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

    fn wait_until(mut ready: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !ready() {
            assert!(
                std::time::Instant::now() < deadline,
                "endpoint did not retire"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn idle_endpoints_release_their_tasks_and_reconnect_in_order() {
        let (incoming, _) = mpsc::channel();
        let sender = Sockets::with_timeouts(
            "127.0.0.1",
            incoming,
            IO_TIMEOUT,
            Duration::from_millis(100),
        )
        .unwrap();
        let mut peers = Vec::new();
        let mut retired = Vec::new();
        for _ in 0..4 {
            let (incoming, received) = mpsc::channel();
            let peer = Sockets::start("127.0.0.1", incoming).unwrap();
            sender
                .send("127.0.0.1", peer.port(), vec![ascii("first")])
                .unwrap();
            let address = format!("tcp://127.0.0.1:{}", peer.port());
            retired.push(Arc::downgrade(
                &sender.outgoing.lock().unwrap()[&address].pending,
            ));
            assert_eq!(
                received.recv_timeout(Duration::from_secs(2)).unwrap(),
                vec![ascii("first")]
            );
            peers.push((peer, received));
        }
        wait_until(|| {
            sender.outgoing.lock().unwrap().is_empty()
                && retired.iter().all(|state| state.upgrade().is_none())
        });
        // The same address is recreated on demand, with one FIFO for its new messages.
        let (peer, received) = &peers[0];
        for label in ["registration", "manifest", "cancel"] {
            sender
                .send("127.0.0.1", peer.port(), vec![ascii(label)])
                .unwrap();
        }
        for label in ["registration", "manifest", "cancel"] {
            assert_eq!(
                received.recv_timeout(Duration::from_secs(2)).unwrap(),
                vec![ascii(label)]
            );
        }
    }

    #[test]
    fn a_failed_connection_discards_its_backlog_and_releases_all_reservations() {
        let stalled = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = stalled.local_addr().unwrap().port();
        let address = format!("tcp://127.0.0.1:{port}");
        let (incoming, _) = mpsc::channel();
        let sender =
            Sockets::with_timeouts("127.0.0.1", incoming, Duration::from_secs(1), IDLE_TIMEOUT)
                .unwrap();
        let (incoming, received) = mpsc::channel();
        let healthy = Sockets::start("127.0.0.1", incoming).unwrap();
        // Retry the same dead address after retirement, proving recreation too.
        for attempt in 0..2 {
            sender
                .send_coalesced("127.0.0.1", port, vec![ascii("cancel")])
                .unwrap();
            let _held = accept_with_deadline(&stalled);
            let retired = Arc::downgrade(&sender.outgoing.lock().unwrap()[&address].pending);
            for room in 0..64 {
                sender
                    .send_coalesced("127.0.0.1", port, vec![ascii("cancel"), ascii(room)])
                    .unwrap();
            }
            sender
                .send("127.0.0.1", healthy.port(), vec![ascii(attempt)])
                .unwrap();
            assert_eq!(
                received.recv_timeout(Duration::from_secs(1)).unwrap(),
                vec![ascii(attempt)]
            );
            // One failed I/O retires the whole backlog, not 65 one-second attempts.
            wait_until(|| {
                !sender.outgoing.lock().unwrap().contains_key(&address)
                    && retired.upgrade().is_none()
            });
        }
    }

    #[test]
    fn old_task_cleanup_cannot_remove_a_replacement_endpoint() {
        let stalled = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = stalled.local_addr().unwrap().port();
        let address = format!("tcp://127.0.0.1:{port}");
        let (incoming, _) = mpsc::channel();
        let sender =
            Sockets::with_timeouts("127.0.0.1", incoming, Duration::from_secs(1), IDLE_TIMEOUT)
                .unwrap();
        sender
            .send_coalesced("127.0.0.1", port, vec![ascii("old")])
            .unwrap();
        let _first = accept_with_deadline(&stalled);
        let old = sender.outgoing.lock().unwrap()[&address].pending.clone();
        let retired = Arc::downgrade(&old);
        // Hold notification cleanup after the failed task closes its channel,
        // so enqueue deterministically replaces it before the lease is dropped.
        let hold = old.lock().unwrap();
        wait_until(|| {
            sender.outgoing.lock().unwrap()[&address]
                .outgoing
                .is_closed()
        });
        sender
            .send_coalesced("127.0.0.1", port, vec![ascii("replacement")])
            .unwrap();
        let replacement = Arc::downgrade(&sender.outgoing.lock().unwrap()[&address].pending);
        drop(hold);
        drop(old);
        let _second = accept_with_deadline(&stalled);
        wait_until(|| retired.upgrade().is_none());
        assert!(Arc::ptr_eq(
            &sender.outgoing.lock().unwrap()[&address].pending,
            &replacement.upgrade().unwrap(),
        ));
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
