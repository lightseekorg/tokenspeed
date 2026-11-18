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

//! Peer prefix sharing on the host (`docs/rfcs/0004-peer-prefix-sharing.md`):
//! the endpoint and reads of ts-control's `peer` module, started from the
//! bootstrap's `PeerSharingSettings` (`cache/peer.py`), and the outbox the
//! data plane reports each plan's landing fence through.

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};

use parking_lot::Mutex;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use ts_collective::{Group, Mesh};
use ts_control::peer::{
    geometry::{DeviceField, HostGroup, HostPool, RankGeometry},
    landing::Timeouts,
    read::EngineReader,
    Peer, PeerFence, PeerSettings,
};
use ts_scheduler::Scheduler;

/// Where the data plane tells the host that a plan's landings may be read
/// (`DeviceHandle.execute`).
#[pyclass(frozen, name = "PeerOutbox")]
pub struct Outbox {
    fence: OnceLock<PeerFence>,
}

impl Outbox {
    /// An outbox whose peer starts later.
    pub fn new() -> Self {
        Self {
            fence: OnceLock::new(),
        }
    }
}

#[pymethods]
impl Outbox {
    /// The oldest plan with landings not yet reported ran its device
    /// prerequisites. The forward thread runs this, inside the plan's
    /// execution.
    fn reads_ready(&self) -> PyResult<()> {
        self.fence
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("the host's peer has not started"))?
            .ready();
        Ok(())
    }
}

/// What the bootstrap resolved for this rank.
pub struct Settings {
    peer: PeerSettings,
    port: u16,
    library: PathBuf,
    local_ip: String,
    protocol: String,
    ib_device: String,
    regions: Vec<(usize, usize)>,
    device: Vec<DeviceField>,
    host: HostPool,
}

fn micros(seconds: f64) -> i64 {
    (seconds.max(0.0) * 1e6) as i64
}

impl Settings {
    /// The weight version the cache namespace starts at.
    pub fn weight_version(&self) -> &str {
        &self.peer.weight_version
    }

    /// The settings of `PeerSharingSettings`, or `None` for None.
    pub fn from_python(settings: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if settings.is_none() {
            return Ok(None);
        }
        let geometry = settings.getattr("geometry")?;
        let device = geometry
            .getattr("device")?
            .try_iter()?
            .map(|field| {
                let field = field?;
                Ok(DeviceField {
                    group_id: field.getattr("group")?.extract()?,
                    field_id: field.getattr("field_id")?.extract()?,
                    address: field.getattr("address")?.extract()?,
                    page_stride: field.getattr("page_stride")?.extract()?,
                    bytes: field.getattr("bytes")?.extract()?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let host = geometry.getattr("host")?;
        let host = HostPool {
            address: host.getattr("address")?.extract()?,
            groups: host
                .getattr("groups")?
                .try_iter()?
                .map(|group| {
                    let group = group?;
                    Ok(HostGroup {
                        parent_bytes: group.getattr("parent_bytes")?.extract()?,
                        packing: group.getattr("packing")?.extract()?,
                        block_bytes: group.getattr("block_bytes")?.extract()?,
                        field_offsets: group.getattr("field_offsets")?.extract()?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
        };
        Ok(Some(Self {
            peer: PeerSettings {
                name: settings.getattr("name")?.extract()?,
                bind_host: settings.getattr("bind_host")?.extract()?,
                incarnation: settings.getattr("incarnation")?.extract()?,
                deployment_id: settings.getattr("deployment_id")?.extract()?,
                weight_version: settings.getattr("weight_version")?.extract()?,
                timeouts: Timeouts {
                    answer_us: micros(settings.getattr("answer_timeout_s")?.extract()?),
                    quarantine_us: micros(settings.getattr("quarantine_s")?.extract()?),
                },
                record_lifetime: Duration::from_secs_f64(
                    settings
                        .getattr("record_lifetime_s")?
                        .extract::<f64>()?
                        .max(0.0),
                ),
            },
            port: settings.getattr("port")?.extract()?,
            library: PathBuf::from(settings.getattr("library")?.extract::<String>()?),
            local_ip: settings.getattr("local_ip")?.extract()?,
            protocol: settings.getattr("protocol")?.extract()?,
            ib_device: settings.getattr("ib_device")?.extract()?,
            regions: geometry.getattr("regions")?.extract()?,
            device,
            host,
        }))
    }
}

/// Start this rank's peer sharing over `scheduler`: its transfer engine with
/// the cache registered, the replica's geometries gathered over `mesh` (every
/// member of `replica` calls this), and `outbox` connected to the reads.
pub fn start(
    settings: Settings,
    mesh: &mut Mesh,
    replica: &Group,
    scheduler: Arc<Mutex<Scheduler>>,
    outbox: &Outbox,
) -> Result<Peer, String> {
    let library =
        ts_mooncake::Library::load(&settings.library).map_err(|error| error.to_string())?;
    if !library.reads() {
        return Err(format!(
            "{} has no READ entry points: peer prefix sharing needs a \
             tokenspeed-mooncake build with ts_mooncake_read",
            settings.library.display()
        ));
    }
    let engine = ts_mooncake::Engine::start(
        library,
        &settings.local_ip,
        &settings.protocol,
        &settings.ib_device,
    )
    .map_err(|error| error.to_string())?;
    for &(address, length) in &settings.regions {
        engine
            .register(address, length)
            .map_err(|error| error.to_string())?;
    }
    let local = RankGeometry {
        session: engine.session_id().to_owned(),
        device: settings.device,
        host: Some(settings.host),
    };
    let gathered = mesh
        .all_gather(replica, &local.encode())
        .map_err(|error| error.to_string())?;
    let geometries = gathered
        .iter()
        .map(|bytes| RankGeometry::decode(bytes))
        .collect::<Option<Vec<_>>>()
        .ok_or("a replica member's peer geometry does not decode")?;
    tracing::debug!(
        regions = ?settings
            .regions
            .iter()
            .map(|(address, length)| format!("{address:#x}+{length:#x}"))
            .collect::<Vec<_>>(),
        fields = ?local
            .device
            .iter()
            .map(|field| format!("{}@{:#x}/{:#x}", field.field_id, field.address, field.page_stride))
            .collect::<Vec<_>>(),
        "peer geometry"
    );
    // Replicas read only holders of the same namespace; a mismatch answers
    // as a miss, so the namespace is worth seeing in the log.
    tracing::info!(
        name = settings.peer.name,
        session = engine.session_id(),
        incarnation = settings.peer.incarnation,
        deployment_id = settings.peer.deployment_id,
        weight_version = settings.peer.weight_version,
        "peer prefix sharing up"
    );
    let peer = Peer::start(
        settings.peer,
        replica.index(),
        geometries,
        settings.port,
        scheduler,
        Arc::new(EngineReader(Arc::new(engine))),
    )
    .map_err(|error| error.to_string())?;
    // A second start would be a second peer; there is one per process.
    if outbox.fence.set(peer.fence()).is_err() {
        return Err("the host's peer started twice".to_owned());
    }
    Ok(peer)
}
