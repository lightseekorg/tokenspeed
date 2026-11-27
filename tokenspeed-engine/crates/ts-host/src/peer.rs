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

mod storage;

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};

use parking_lot::Mutex;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use ts_collective::{Group, Mesh};
use ts_control::{
    disk::DiskTier,
    peer::{
        geometry::{DeviceField, HostGroup, HostPool, RankGeometry},
        landing::Timeouts,
        push::{EngineWriter, PushEndpoint, PushSource},
        read::EngineReader,
        registered::{Member, Replica},
        Peer, PeerFence, PeerListener, PeerProgress, PeerSettings, PeerTransfers,
    },
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
    l3: Option<ts_l3::Settings>,
    storage: Option<storage::Settings>,
    peer: PeerSettings,
    node_leaders: Vec<(usize, u32)>,
    bind_host: String,
    port: u16,
    /// The host other replicas reach this rank's push endpoint at.
    push_host: String,
    library: PathBuf,
    local_ip: String,
    protocol: String,
    ib_device: String,
    regions: Vec<(usize, usize)>,
    device: Vec<DeviceField>,
    host: HostPool,
    contract: Vec<u8>,
    host_bounds: Vec<i64>,
    semantic_id: Vec<u8>,
    source_role: String,
}

fn micros(seconds: f64) -> i64 {
    (seconds.max(0.0) * 1e6) as i64
}

impl Settings {
    /// The Store and peer register the same Host allocation. Native transports
    /// may share exact registrations but reject partially overlapping ones.
    pub fn use_storage_registrations(&mut self, l3: &ts_l3::Settings) -> Result<(), String> {
        self.l3 = Some(l3.clone());
        if matches!(l3.store, ts_l3::StoreSettings::Memory) {
            return Ok(());
        }
        if self.host.address != l3.host.base as u64 {
            return Err("L3 and peer must share the same Host allocation".into());
        }
        let regions = l3.host_registrations().map_err(|e| e.to_string())?;
        self.regions = replace_host_regions(&self.regions, l3.host, &regions)?;
        Ok(())
    }

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
            l3: None,
            storage: storage::Settings::from_python(&settings.getattr("storage")?)?,
            peer: PeerSettings {
                name: settings.getattr("name")?.extract()?,
                incarnation: settings.getattr("incarnation")?.extract()?,
                deployment_id: settings.getattr("deployment_id")?.extract()?,
                weight_version: settings.getattr("weight_version")?.extract()?,
                timeouts: Timeouts {
                    answer_us: micros(settings.getattr("answer_timeout_s")?.extract()?),
                    quarantine_us: micros(settings.getattr("quarantine_s")?.extract()?),
                    push_us: micros(settings.getattr("push_timeout_s")?.extract()?),
                },
                record_lifetime: Duration::from_secs_f64(
                    settings
                        .getattr("record_lifetime_s")?
                        .extract::<f64>()?
                        .max(0.0),
                ),
            },
            node_leaders: settings.getattr("node_leaders")?.extract()?,
            bind_host: settings.getattr("bind_host")?.extract()?,
            port: settings.getattr("port")?.extract()?,
            push_host: settings.getattr("push_host")?.extract()?,
            library: PathBuf::from(settings.getattr("library")?.extract::<String>()?),
            local_ip: settings.getattr("local_ip")?.extract()?,
            protocol: settings.getattr("protocol")?.extract()?,
            ib_device: settings.getattr("ib_device")?.extract()?,
            regions: geometry.getattr("regions")?.extract()?,
            contract: geometry.getattr("transfer_contract")?.extract()?,
            host_bounds: geometry.getattr("host_virtual_bounds")?.extract()?,
            semantic_id: settings.getattr("semantic_id")?.extract()?,
            source_role: settings.getattr("source_role")?.extract()?,
            device,
            host,
        }))
    }
}

fn replace_host_regions(
    existing: &[(usize, usize)],
    host: ts_l3::HostBuffer,
    replacements: &[ts_l3::Range],
) -> Result<Vec<(usize, usize)>, String> {
    let end = host
        .base
        .checked_add(host.len)
        .ok_or("Host address overflow")?;
    let mut device = Vec::new();
    let mut old_host = Vec::new();
    for &(address, length) in existing {
        let limit = address.checked_add(length).ok_or("peer address overflow")?;
        if address < end && limit > host.base {
            if address < host.base || limit > end {
                return Err("peer registration partially overlaps the Host pool".into());
            }
            old_host.push((address - host.base, length));
        } else {
            device.push((address, length));
        }
    }
    old_host.sort_unstable();
    let covers_pool = |regions: &[(usize, usize)]| {
        let mut cursor = 0_usize;
        for &(offset, length) in regions {
            if offset != cursor || length == 0 {
                return false;
            }
            let Some(next) = cursor.checked_add(length) else {
                return false;
            };
            cursor = next;
        }
        cursor == host.len
    };
    let replacement_ranges = replacements
        .iter()
        .map(|r| (r.offset, r.size))
        .collect::<Vec<_>>();
    if !covers_pool(&old_host) || !covers_pool(&replacement_ranges) {
        return Err("peer and L3 registrations must cover the same complete Host pool".into());
    }
    device.extend(replacements.iter().map(|&r| (host.address(r), r.size)));
    Ok(device)
}

#[cfg(test)]
mod registration_tests {
    use super::replace_host_regions;
    use ts_l3::{HostBuffer, Range};

    #[test]
    fn store_and_peer_share_exact_regions_without_changing_device_registration() {
        let host = HostBuffer {
            base: 4096,
            len: 512,
        };
        let old = [(16384, 1024), (4096, 256), (4352, 256)];
        let new = [
            Range {
                offset: 0,
                size: 128,
            },
            Range {
                offset: 128,
                size: 384,
            },
        ];
        assert_eq!(
            replace_host_regions(&old, host, &new).unwrap(),
            vec![(16384, 1024), (4096, 128), (4224, 384)]
        );
        assert!(replace_host_regions(&old, host, &new[..1]).is_err());
        assert!(replace_host_regions(&[(4095, 513)], host, &new).is_err());
        assert!(replace_host_regions(&[(4096, 256)], host, &new).is_err());
    }
}

/// Start this rank's peer sharing over `scheduler`: its transfer engine with
/// the cache registered, its push endpoint, serving `disk`'s pages when the
/// rank has a Disk tier, the replica's geometries gathered over `mesh` (every
/// member of `replica` calls this), and `outbox` connected to the reads.
pub fn start(
    settings: Settings,
    mesh: &mut Mesh,
    replica: &Group,
    scheduler: Arc<Mutex<Scheduler>>,
    outbox: &Outbox,
    disk: Option<&DiskTier>,
    l3_executor: Option<&ts_l3::Executor>,
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
    let engine = Arc::new(engine);
    // Every rank answers its own pushes here; one with a Disk tier also
    // pushes its Disk pages from the staging ring, registered for the writes.
    let push = PushEndpoint::bind(
        &settings.bind_host,
        0,
        &settings.push_host,
        settings.peer.incarnation,
    )
    .map_err(|error| error.to_string())?;
    let push_source = disk.and_then(DiskTier::push_source);
    if let Some((_, _, ref ring, _)) = push_source {
        engine
            .register(ring.address(), ring.size())
            .map_err(|error| error.to_string())?;
    }
    let local = RankGeometry {
        session: engine.session_id().to_owned(),
        device: settings.device,
        host: Some(settings.host),
        push: Some(push.name().to_owned()),
    };
    let gathered = mesh
        .all_gather(replica, &local.encode())
        .map_err(|error| error.to_string())?;
    let geometries = gathered
        .iter()
        .map(|bytes| RankGeometry::decode(bytes))
        .collect::<Option<Vec<_>>>()
        .ok_or("a replica member's peer geometry does not decode")?;
    // Agree capability and rank-independent contract before any conditional
    // collective. All ranks participate even when one lacks a pinned identity.
    let mut capability = Vec::new();
    if !settings.contract.is_empty()
        && settings.semantic_id.len() == 32
        && matches!(
            (settings.source_role.as_str(), geometries.len()),
            ("prefill", 8) | ("decode", 1)
        )
    {
        capability.extend_from_slice(&settings.semantic_id);
        capability.extend_from_slice(&settings.contract);
    }
    let capabilities = mesh
        .all_gather(replica, &capability)
        .map_err(|e| e.to_string())?;
    let registered =
        if !capability.is_empty() && capabilities.iter().all(|other| other == &capability) {
            let member = Member {
                geometry: local.clone(),
                bounds: settings.host_bounds,
                regions: settings
                    .regions
                    .iter()
                    .map(|&(a, n)| (a as u64, n as u64))
                    .collect(),
            };
            let members = mesh
                .all_gather(replica, &member.encode(replica.index()))
                .map_err(|e| e.to_string())?;
            let digest = settings
                .semantic_id
                .as_slice()
                .try_into()
                .map_err(|_| "invalid peer semantic digest")?;
            Some(Arc::new(
                Replica::register(digest, &settings.contract, &members)
                    .ok_or("invalid registered peer layout")?,
            ))
        } else {
            None
        };
    if !settings.contract.is_empty()
        && settings.source_role == "prefill"
        && geometries.len() == 8
        && registered.is_none()
    {
        return Err(
            "TP8/KVP8 peer reader requires a pinned weight identity and a common registered layout"
                .into(),
        );
    }
    let catalog = if replica.index() == 0 {
        match (registered.as_ref(), settings.l3.as_ref(), l3_executor) {
            (Some(layout), Some(l3), Some(executor)) => {
                let source = if settings.source_role == "decode" && layout.members.len() == 1 {
                    Some(
                        ts_control::peer::stored_layout::StoredLayout::from_decode(
                            layout,
                            &settings.contract,
                            &l3.namespace,
                            &l3.weight_version,
                        )
                        .map_err(|e| e.to_string())?,
                    )
                } else {
                    None
                };
                if source.is_some() || settings.storage.is_some() {
                    Some(
                        ts_control::peer::catalog::Catalog::start(
                            executor.metadata(),
                            layout.semantic,
                            l3.weight_version.clone(),
                            source,
                        )
                        .map_err(|e| e.to_string())?,
                    )
                } else {
                    None
                }
            }
            _ => None,
        }
    } else {
        None
    };
    if let Some((executor, disk_layout, ring, slot_bytes)) = push_source {
        push.serve(PushSource {
            executor,
            disk_layout,
            ring,
            slot_bytes,
            writer: Arc::new(EngineWriter(engine.clone())),
        })
        .map_err(|error| error.to_string())?;
    }
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
    let progress = PeerProgress::for_replica(geometries.len(), &mut scheduler.lock())
        .map_err(|error| error.to_string())?;
    // Replicas read only holders of the same namespace; a mismatch answers
    // as a miss, so the namespace is worth seeing in the log.
    tracing::info!(
        ?progress,
        members = geometries.len(),
        name = settings.peer.name,
        session = engine.session_id(),
        incarnation = settings.peer.incarnation,
        deployment_id = settings.peer.deployment_id,
        weight_version = settings.peer.weight_version,
        "peer prefix sharing up"
    );
    let mut listener = if replica.index() == 0 {
        Some(
            PeerListener::bind_node(
                mesh,
                super::PEER_NODE_GROUP,
                &settings.node_leaders,
                &settings.bind_host,
                settings.port,
            )
            .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let storage = if replica.index() == 0 {
        settings
            .storage
            .map(|storage| {
                storage.start(
                    &settings.peer,
                    geometries.len(),
                    scheduler.clone(),
                    registered.clone(),
                    catalog.clone(),
                )
            })
            .transpose()?
    } else {
        None
    };
    if let (Some(listener), Some(layout)) = (listener.as_mut(), registered.as_ref()) {
        listener.set_registered_layout(layout.clone());
    }
    let channel = listener.as_ref().map(PeerListener::versioned_channel);
    let name = settings.peer.name.clone();
    let incarnation = settings.peer.incarnation;
    let mut peer = Peer::start(
        settings.peer,
        progress,
        replica.index(),
        geometries,
        listener,
        scheduler,
        PeerTransfers {
            reader: Arc::new(EngineReader(engine)),
            pusher: Arc::new(push),
        },
    )
    .map_err(|error| error.to_string())?;
    if let Some(storage) = storage {
        peer.attach_storage_source(storage);
    }
    if let Some(catalog) = catalog {
        peer.attach_catalog(catalog);
    }
    if let Some(layout) = registered {
        let retirement_bound_ms = match std::env::var("TOKENSPEED_PEER_READ_RETIREMENT_BOUND_MS") {
            Ok(s) => s
                .parse::<std::num::NonZeroU32>()
                .map_err(|_| "TOKENSPEED_PEER_READ_RETIREMENT_BOUND_MS must be a positive u32")?
                .get(),
            Err(std::env::VarError::NotPresent) => 0,
            Err(e) => return Err(e.to_string()),
        };
        if replica.index() == 0 && retirement_bound_ms == 0 {
            tracing::warn!(
                "peer READ retirement bound unset; orphaned grants require explicit recovery"
            );
        }
        peer.enable_versioned(layout, name, incarnation, channel, retirement_bound_ms);
    }
    // A second start would be a second peer; there is one per process.
    if outbox.fence.set(peer.fence()).is_err() {
        return Err("the host's peer started twice".to_owned());
    }
    Ok(peer)
}
