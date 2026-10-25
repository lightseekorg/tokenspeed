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

//! Client errors.

use std::time::Duration;

/// An error from connecting to the engines or submitting a request.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A socket operation failed.
    #[error("zmq: {0}")]
    Zmq(#[from] zeromq::ZmqError),
    /// A message did not match the wire schema.
    #[error("wire: {0}")]
    Wire(#[from] ts_wire::Error),
    /// Not every engine finished a handshake stage in time.
    #[error("handshake timed out after {timeout:?} waiting for {stage} from engines {pending:?}")]
    HandshakeTimeout {
        /// The stage that did not complete.
        stage: &'static str,
        /// Engine indices that had not completed it.
        pending: Vec<u16>,
        /// The configured handshake timeout.
        timeout: Duration,
    },
    /// An engine broke the handshake protocol.
    #[error("handshake: {0}")]
    Handshake(String),
    /// The request cannot be submitted as given.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// A request with this id is already in flight.
    #[error("request id `{0}` is already in flight")]
    DuplicateRequestId(String),
    /// The engine did not accept the message in time.
    #[error("sending to engine {engine_index} timed out after {timeout:?}")]
    SendTimeout {
        /// The engine the message was for.
        engine_index: u16,
        /// The configured send timeout.
        timeout: Duration,
    },
    /// The engines are gone; see [`StreamError`] for why.
    #[error("engines unavailable: {0}")]
    Unavailable(StreamError),
}

/// Why a request's output stream ended without a finished output.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    /// An engine announced its shutdown. On this wire the announcement does
    /// not say which engine, and an engine group cannot serve with a rank
    /// missing, so every stream ends.
    #[error("an engine shut down")]
    EngineDead,
    /// The output socket failed.
    #[error("output socket failed: {0}")]
    Transport(String),
    /// The client was dropped.
    #[error("the client was dropped")]
    ClientClosed,
}

/// Result alias for client operations.
pub type Result<T> = std::result::Result<T, Error>;
