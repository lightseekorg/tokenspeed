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

//! Client configuration. Every field selects behavior, so none has a default.

use std::time::Duration;

/// How requests are assigned to engines (one engine per attention-DP rank).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routing {
    /// Aggregated serving. A request's `data_parallel_rank` pins it to that
    /// rank; other requests are placed by the policy.
    Aggregated(LoadBalance),
    /// Prefill/decode disaggregation: rank `bootstrap_room % num_engines`,
    /// or the rank a request's `data_parallel_rank` pins it to. Every
    /// request must carry a `bootstrap_room`. Pin decode requests only: any
    /// decode rank finds the request's prefill rank from the room, so the
    /// prefill request must stay on the room's rank.
    BootstrapRoom,
}

/// Placement policy for unpinned requests in aggregated serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalance {
    /// Cycle through the engines.
    RoundRobin,
    /// The engine with the fewest of this client's requests in flight.
    LeastInFlight,
    /// The engine whose latest reported load holds the fewest KV pages,
    /// then the fewest of this client's requests in flight.
    MinimumCacheUsage,
}

/// Where the client listens and which engines it expects.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Handshake ROUTER address the engines dial: the engine's
    /// `tcp://{--data-parallel-address}:{--data-parallel-rpc-port}`.
    /// `tcp://<host>:0` binds a free port; see [`crate::ClientListener`].
    pub handshake_address: String,
    /// Input ROUTER address. `tcp://<host>:0` binds a free port; the bound
    /// address is sent to the engines, so `<host>` must be reachable from
    /// them.
    pub input_address: String,
    /// Output PULL address, bound like `input_address`.
    pub output_address: String,
    /// The first engine's index, the engine's `--zmq-engine-index`.
    pub first_engine_index: u16,
    /// Engines to wait for: indices `first_engine_index..+num_engines`, one
    /// per attention-DP rank.
    pub num_engines: u16,
    /// Request placement.
    pub routing: Routing,
    /// Limit for the handshake, from the start of
    /// [`crate::ClientListener::accept`] to the last registration.
    pub handshake_timeout: Duration,
    /// Limit for handing one message to an engine's socket. A stalled
    /// engine fails the submission instead of blocking the caller.
    pub send_timeout: Duration,
}
