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

//! The replica's peer endpoint: one PULL socket every other replica pushes
//! requests and replies to, and PUSH sockets to them (`ts_pd::Sockets`). Each
//! message is a kind frame and the encoded request or reply.

use std::sync::mpsc;

use bytes::Bytes;
use ts_pd::{NetError, SendError, Sockets};

use crate::peer::wire::{Malformed, Reply, Request};

const REQUEST: &[u8] = b"q";
const REPLY: &[u8] = b"r";

/// A message off the endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inbound {
    /// Another replica asks this one, as a holder.
    Request(Request),
    /// A holder answers this one, as a reader.
    Reply(Reply),
}

impl Inbound {
    /// A message's frames, decoded.
    pub fn decode(frames: &[Bytes]) -> Result<Self, Malformed> {
        match frames {
            [kind, payload] if kind.as_ref() == REQUEST => {
                Request::decode(payload).map(Inbound::Request)
            }
            [kind, payload] if kind.as_ref() == REPLY => Reply::decode(payload).map(Inbound::Reply),
            _ => Err(Malformed),
        }
    }
}

/// The endpoint.
pub struct PeerNet {
    sockets: Sockets,
}

/// A `host:port` that does not parse.
#[derive(Debug, thiserror::Error)]
#[error("the peer endpoint {0:?} is not host:port")]
pub struct BadEndpoint(pub String);

/// `host:port`, an IPv6 host in brackets, as its parts.
pub fn split_endpoint(endpoint: &str) -> Result<(&str, u16), BadEndpoint> {
    let bad = || BadEndpoint(endpoint.to_owned());
    let (host, port) = endpoint.rsplit_once(':').ok_or_else(bad)?;
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    if host.is_empty() {
        return Err(bad());
    }
    Ok((host, port.parse().map_err(|_| bad())?))
}

/// A message the endpoint could not queue.
#[derive(Debug, thiserror::Error)]
pub enum PeerSendError {
    /// The destination does not parse.
    #[error(transparent)]
    Endpoint(#[from] BadEndpoint),
    /// The outgoing queue refused it.
    #[error(transparent)]
    Queue(#[from] SendError),
}

impl PeerNet {
    /// Bind the endpoint on `host` at `port` (0: the OS picks); every message
    /// it receives goes to `incoming`.
    pub fn start(
        host: &str,
        port: u16,
        incoming: mpsc::Sender<Vec<Bytes>>,
    ) -> Result<Self, NetError> {
        Ok(Self {
            sockets: Sockets::start_at(host, port, incoming)?,
        })
    }

    /// The bound port.
    pub fn port(&self) -> u16 {
        self.sockets.port()
    }

    /// Queue `request` to the holder at `endpoint`.
    pub fn request(&self, endpoint: &str, request: &Request) -> Result<(), PeerSendError> {
        let (host, port) = split_endpoint(endpoint)?;
        Ok(self.sockets.send(
            host,
            port,
            vec![Bytes::from_static(REQUEST), Bytes::from(request.encode())],
        )?)
    }

    /// Queue `reply` to the reader at `endpoint`.
    pub fn reply(&self, endpoint: &str, reply: &Reply) -> Result<(), PeerSendError> {
        let (host, port) = split_endpoint(endpoint)?;
        Ok(self.sockets.send(
            host,
            port,
            vec![Bytes::from_static(REPLY), Bytes::from(reply.encode())],
        )?)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ts_scheduler::PeerLeaseView;

    use super::*;

    #[test]
    fn endpoints_split_with_and_without_brackets() {
        assert_eq!(split_endpoint("10.0.0.2:7100").unwrap(), ("10.0.0.2", 7100));
        assert_eq!(split_endpoint("[::1]:7100").unwrap(), ("::1", 7100));
        assert!(split_endpoint("10.0.0.2").is_err());
        assert!(split_endpoint(":7100").is_err());
        assert!(split_endpoint("host:port").is_err());
    }

    #[test]
    fn a_request_reaches_the_other_endpoint() {
        let (incoming, received) = mpsc::channel();
        let holder = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
        let (unused, _) = mpsc::channel();
        let reader = PeerNet::start("127.0.0.1", 0, unused).unwrap();
        let request = Request::Pin {
            lease: PeerLeaseView {
                reader: format!("127.0.0.1:{}", reader.port()),
                id: 3,
            },
        };
        reader
            .request(&format!("127.0.0.1:{}", holder.port()), &request)
            .unwrap();
        let frames = received.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(Inbound::decode(&frames), Ok(Inbound::Request(request)));
    }
}
