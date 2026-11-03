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

//! TokenSpeed's PD transfer peer (RFC 0002): one rank's half of moving a
//! request's KV from the prefill node to the decode node.
//!
//! The peer owns the Mooncake transfer engine, the P/D handshake over ZMQ,
//! the bootstrap-server HTTP calls, the room states, the transfer workers
//! and the per-round agreement on transfer progress. It also plans
//! everything that needs the ranks' cache layouts ([`layout`]): the routes
//! between unequal Prefill and Decode rank sets, the block manifests read
//! off the plans' block tables, and the copies one transfer makes. The data
//! plane only builds the rank's cache transfer contract at startup and
//! tells the forward thread when a plan's remote stream may go out
//! ([`PrefillOutbox`], [`DecodeOutbox`]). The bootstrap server the nodes
//! rendezvous through is here too ([`bootstrap`]). The wire formats are
//! unchanged from the Python peers this crate replaced, so nodes of either
//! version interoperate.

mod batch;
pub mod bootstrap;
mod control;
mod decode;
pub mod epd;
mod http;
pub mod layout;
mod net;
mod prefill;
mod status;
mod wire;

use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};

pub use batch::BatchRows;
pub use bootstrap::{BootstrapServer, Registry};
pub use control::{
    BootstrapInfo, EmbeddingChunk, EmbeddingItem, EmbeddingReceiver, EmbeddingSender, LandedItem,
    LandedRequest, PdError, PdEvent, RemotePrefill, TransferPeer,
};
pub use decode::{DecodeConfig, DecodeOutbox, DecodePeer};
pub use layout::{
    registration::{PrefillLayout, PrefillRoute},
    LayoutError,
};
pub use net::NetError;
pub use prefill::{Layerwise, PrefillConfig, PrefillOutbox, PrefillPeer};
pub use status::TransferPoll;
use ts_mooncake::{Engine, PageGrid};
pub use wire::StatusMessage;

/// What one transfer copies, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Copy {
    /// A pages x fields grid.
    Pages(PageGrid),
    /// Contiguous spans: source, destination, length each.
    Spans {
        /// Local addresses.
        sources: Vec<u64>,
        /// The peer's addresses.
        destinations: Vec<u64>,
        /// Byte counts.
        lengths: Vec<u64>,
    },
}

/// The most descriptors one WRITE batch carries.
const DESCRIPTOR_BATCH: usize = 4096;

/// A peer that failed to start.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    /// The transfer engine failed.
    #[error(transparent)]
    Engine(#[from] ts_mooncake::Error),
    /// The control sockets failed.
    #[error(transparent)]
    Net(#[from] NetError),
    /// A thread did not start.
    #[error("starting a PD thread: {0}")]
    Thread(#[from] std::io::Error),
    /// The rank's cache transfer contract is malformed.
    #[error(transparent)]
    Layout(#[from] LayoutError),
}

/// What both roles start from.
#[derive(Debug, Clone)]
pub struct PeerConfig {
    /// The Mooncake library (`libtokenspeed_transfer_engine.so`).
    pub library: PathBuf,
    /// This rank's address peers reach it at (`get_local_ip_by_remote`).
    pub local_ip: String,
    /// The native transport selected by the data plane at startup.
    pub protocol: String,
    /// The RDMA devices to use, comma-separated; empty for all.
    pub ib_device: String,
    /// The device ranges to register: the cache arena or its regions.
    pub regions: Vec<(usize, usize)>,
    /// This rank's global rank, for messages.
    pub engine_rank: i64,
    /// The transfer metrics, when the engine exports metrics.
    pub metrics: Option<Arc<ts_metrics::KvTransferMetrics>>,
}

/// The engine and sockets both roles share.
struct Transport {
    library: Arc<ts_mooncake::Library>,
    local_ip: String,
    protocol: String,
    ib_device: String,
    /// The device ranges registered with the engine.
    regions: Vec<(usize, usize)>,
    /// Replaced when a sleeping Decode rank wakes under a fresh session.
    engine: RwLock<Arc<Engine>>,
    sockets: net::Sockets,
}

impl Transport {
    fn start(
        config: &PeerConfig,
        incoming: std::sync::mpsc::Sender<Vec<bytes::Bytes>>,
    ) -> Result<Arc<Self>, StartError> {
        let library = ts_mooncake::Library::load(&config.library)?;
        let engine = Engine::start(
            library.clone(),
            &config.local_ip,
            &config.protocol,
            &config.ib_device,
        )?;
        for &(addr, length) in &config.regions {
            engine.register(addr, length)?;
        }
        tracing::info!(
            session = engine.session_id(),
            protocol = config.protocol,
            ib_device = config.ib_device,
            "Mooncake transfer engine ready"
        );
        let sockets = net::Sockets::start(&config.local_ip, incoming)?;
        Ok(Arc::new(Self {
            library,
            local_ip: config.local_ip.clone(),
            protocol: config.protocol.clone(),
            ib_device: config.ib_device.clone(),
            regions: config.regions.clone(),
            engine: RwLock::new(Arc::new(engine)),
            sockets,
        }))
    }

    /// The engine transfers run on.
    fn engine(&self) -> Arc<Engine> {
        self.engine
            .read()
            .expect("the engine lock is poisoned")
            .clone()
    }

    /// The id peers address this rank's memory by.
    fn session_id(&self) -> String {
        self.engine().session_id().to_owned()
    }

    /// Deregister the cache before a sleep unmaps its memory.
    fn release_regions(&self) -> Result<(), String> {
        let engine = self.engine();
        let failed = self
            .regions
            .iter()
            .filter(|&&(addr, _)| engine.unregister(addr).is_err())
            .count();
        if failed > 0 {
            return Err(format!(
                "Mooncake failed to deregister {failed} cache region(s) before the memory release"
            ));
        }
        Ok(())
    }

    /// Register the re-mapped cache after a wake; with `fresh_session`
    /// under a new engine, so no peer keeps writing with the keys of the
    /// pages the sleep freed.
    fn restore_regions(&self, fresh_session: bool) -> Result<(), String> {
        let engine = if fresh_session {
            Arc::new(
                Engine::start(
                    self.library.clone(),
                    &self.local_ip,
                    &self.protocol,
                    &self.ib_device,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            self.engine()
        };
        let failed = self
            .regions
            .iter()
            .filter(|&&(addr, length)| engine.register(addr, length).is_err())
            .count();
        if failed > 0 {
            return Err(format!(
                "Mooncake failed to register {failed} cache region(s) after the memory resume"
            ));
        }
        if fresh_session {
            tracing::info!(session = engine.session_id(), "a fresh transfer session");
            *self.engine.write().expect("the engine lock is poisoned") = engine;
        }
        Ok(())
    }

    /// Run one transfer's copies to `session`.
    fn write(&self, session: &str, copies: &[Copy]) -> Result<(), ts_mooncake::Error> {
        let engine = self.engine();
        for copy in copies {
            match copy {
                Copy::Pages(grid) => engine.write_pages(session, grid, DESCRIPTOR_BATCH)?,
                Copy::Spans {
                    sources,
                    destinations,
                    lengths,
                } => {
                    for start in (0..sources.len()).step_by(DESCRIPTOR_BATCH) {
                        let end = (start + DESCRIPTOR_BATCH).min(sources.len());
                        engine.write(
                            session,
                            &sources[start..end],
                            &destinations[start..end],
                            &lengths[start..end],
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Sleep `duration` unless `stop` fires first; returns whether it fired.
fn sleep_or_stop(stop: &std::sync::mpsc::Receiver<()>, duration: Duration) -> bool {
    !matches!(
        stop.recv_timeout(duration),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    )
}
