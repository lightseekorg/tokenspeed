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
//! rendezvous through is here too ([`bootstrap`]). Cache registrations carry
//! an independent cache identity as well as a transport address; both P/D
//! nodes must use this protocol version.

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

#[cfg(test)]
mod tests_support;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

pub use batch::BatchRows;
pub use bootstrap::{BootstrapServer, Registry};
pub use control::{
    BootstrapInfo, EmbeddingChunk, EmbeddingItem, EmbeddingReceiver, EmbeddingSender, LandedItem,
    LandedRequest, PdError, PdEvent, RemotePrefill, SendRing, SendSlot, TransferPeer,
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

/// An incarnation must not inherit the identity of a reused address.
fn new_instance_id() -> std::io::Result<String> {
    use std::io::Read;
    let mut random = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut random)?;
    Ok(format!("{:032x}", u128::from_be_bytes(random)))
}

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

impl Copy {
    /// Bytes this descriptor moves.
    pub fn bytes(&self) -> u64 {
        match self {
            Copy::Pages(grid) => {
                let per_page: u64 = grid
                    .fields
                    .iter()
                    .map(|field| {
                        debug_assert!(field[4] >= 0, "negative per-page byte length");
                        u64::try_from(field[4]).unwrap_or(0)
                    })
                    .sum();
                grid.src_pages.len() as u64 * per_page
            }
            Copy::Spans { lengths, .. } => lengths.iter().sum(),
        }
    }
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
    /// The cache incarnation could not get an independent identity.
    #[error("creating a cache identity: {0}")]
    CacheIdentity(std::io::Error),
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
    /// Cache writes can replace their engine without changing the address
    /// advertised to EPD senders that still use the public engine.
    cache_sender: Mutex<CacheSender>,
    sockets: net::Sockets,
}

struct CacheSender {
    engine: Arc<Engine>,
    /// The cache incarnation last addressed through each transport endpoint.
    sessions: HashMap<String, String>,
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
        let engine = Arc::new(engine);
        Ok(Arc::new(Self {
            library,
            local_ip: config.local_ip.clone(),
            protocol: config.protocol.clone(),
            ib_device: config.ib_device.clone(),
            regions: config.regions.clone(),
            engine: RwLock::new(engine.clone()),
            cache_sender: Mutex::new(CacheSender {
                engine,
                sessions: HashMap::new(),
            }),
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
        let sender = self
            .cache_sender
            .lock()
            .expect("the cache sender lock is poisoned");
        let mut failed = self
            .regions
            .iter()
            .filter(|&&(addr, _)| engine.unregister(addr).is_err())
            .count();
        if !Arc::ptr_eq(&engine, &sender.engine) {
            failed += self
                .regions
                .iter()
                .filter(|&&(addr, _)| sender.engine.unregister(addr).is_err())
                .count();
        }
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
        let mut sender = self
            .cache_sender
            .lock()
            .expect("the cache sender lock is poisoned");
        let mut failed = self
            .regions
            .iter()
            .filter(|&&(addr, length)| engine.register(addr, length).is_err())
            .count();
        if !fresh_session && !Arc::ptr_eq(&engine, &sender.engine) {
            failed += self
                .regions
                .iter()
                .filter(|&&(addr, length)| sender.engine.register(addr, length).is_err())
                .count();
        }
        if failed > 0 {
            return Err(format!(
                "Mooncake failed to register {failed} cache region(s) after the memory resume"
            ));
        }
        if fresh_session {
            tracing::info!(session = engine.session_id(), "a fresh transfer session");
            *self.engine.write().expect("the engine lock is poisoned") = engine.clone();
            sender.engine = engine;
            sender.sessions.clear();
        }
        Ok(())
    }

    /// Run one transfer's copies to `session`.
    fn write(&self, session: &str, copies: &[Copy]) -> Result<(), ts_mooncake::Error> {
        let engine = self.engine();
        Self::write_on_engine(&engine, session, copies)
    }

    /// Address one cache incarnation without retaining the native keys of
    /// an earlier cache that used the same endpoint. Mooncake's current C
    /// ABI has no segment invalidation call, so replace only the sender.
    fn write_cache(
        &self,
        session: &str,
        cache_id: &str,
        copies: &[Copy],
    ) -> Result<(), ts_mooncake::Error> {
        let engine = {
            let mut sender = self
                .cache_sender
                .lock()
                .expect("the cache sender lock is poisoned");
            if sender
                .sessions
                .get(session)
                .is_some_and(|previous| previous != cache_id)
            {
                let engine = Engine::start(
                    self.library.clone(),
                    &self.local_ip,
                    &self.protocol,
                    &self.ib_device,
                )?;
                for &(addr, length) in &self.regions {
                    engine.register(addr, length)?;
                }
                sender.engine = Arc::new(engine);
                sender.sessions.clear();
            }
            sender
                .sessions
                .insert(session.to_owned(), cache_id.to_owned());
            sender.engine.clone()
        };
        Self::write_on_engine(&engine, session, copies)
    }

    fn write_on_engine(
        engine: &Engine,
        session: &str,
        copies: &[Copy],
    ) -> Result<(), ts_mooncake::Error> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "negative per-page byte length")]
    fn negative_per_page_length_triggers_debug_assertion() {
        Copy::Pages(PageGrid {
            src_pages: vec![1],
            dst_pages: vec![2],
            fields: vec![[0, 0, 0, 0, -1]],
        })
        .bytes();
    }

    #[test]
    fn copy_bytes_counts_pages_and_spans() {
        let grid = Copy::Pages(PageGrid {
            src_pages: vec![1, 2],
            dst_pages: vec![3, 4],
            fields: vec![[0, 0, 0, 0, 128], [0, 0, 0, 0, 64]],
        });
        assert_eq!(grid.bytes(), 2 * (128 + 64));
        let spans = Copy::Spans {
            sources: vec![0, 0],
            destinations: vec![0, 0],
            lengths: vec![10, 20],
        };
        assert_eq!(spans.bytes(), 30);
    }
}

#[cfg(test)]
#[path = "../tests/unit/transport.rs"]
mod transport_tests;
