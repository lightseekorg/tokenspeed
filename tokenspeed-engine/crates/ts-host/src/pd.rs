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

//! The PD transfer peer of a disaggregated rank (`ts_pd`), started from the
//! settings the data plane's peer object carries (`pd/host_peer.py`).
//!
//! The forward thread stages each plan's remote stream before it hands the
//! plan to the data plane ([`Streams`]); the data plane hands the stream
//! over through [`Outbox`] once its device prerequisites are met (the
//! drafter's pending round on a Prefill rank, the page zeroing on a Decode
//! rank), and the peer plans and sends it. Nothing in the peer calls into
//! Python.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyBytes};
use ts_control::output::Role;
use ts_pd::{
    epd::receiver::{PoolRegion, ReceiverConfig},
    layout::{
        contract::Contract,
        manifest::{BlockTable, BlockTables},
    },
    BatchRows, DecodeConfig, DecodeOutbox, DecodePeer, EmbeddingReceiver, Layerwise, PeerConfig,
    PrefillConfig, PrefillLayout, PrefillOutbox, PrefillPeer, TransferPeer,
};
use ts_scheduler::{ExecutionPlan, TransferBatchView};

/// The forward thread's face of the peer.
#[derive(Clone)]
pub enum Streams {
    /// A Prefill rank's: remote decodes and layerwise chunks.
    Prefill(PrefillOutbox),
    /// A Decode rank's: remote prefills.
    Decode(DecodeOutbox),
}

/// A plan's batch as the peer reads it.
fn rows(view: TransferBatchView) -> BatchRows {
    let spec_candidate_ids = view
        .spec_candidate_offsets
        .windows(2)
        .map(|bounds| view.spec_candidate_ids[bounds[0]..bounds[1]].to_vec())
        .collect();
    let tables: BlockTables = view
        .block_tables
        .into_iter()
        .map(|table| {
            let blocks = table.blocks.into_iter().map(i64::from).collect();
            (
                table.group_id,
                BlockTable {
                    columns: table.columns,
                    blocks,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    BatchRows {
        request_ids: view.request_ids,
        request_pool_indices: view.request_pool_indices,
        input_lengths: view.input_lengths,
        prefill_lengths: view.prefill_lengths,
        extend_prefix_lens: view.extend_prefix_lens,
        state_prefix_cached: view.state_prefix_cached,
        decode_input_ids: view.decode_input_ids,
        spec_candidate_ids,
        num_extends: view.num_extends,
        tables,
    }
}

impl Streams {
    /// Before the data plane runs `plan`: stage its remote stream and, on a
    /// layerwise Prefill rank whose round launches a forward, arm the
    /// forward's chunks.
    pub fn before_execute(
        &self,
        plan: &ExecutionPlan,
        launches_forward: bool,
    ) -> Result<(), String> {
        match self {
            Streams::Prefill(outbox) => {
                if launches_forward && outbox.streams_layerwise() {
                    if let Some(forward) = plan.forward_transfer() {
                        outbox.arm_layerwise(&rows(forward))?;
                    }
                }
                if let Some(remote_decode) = plan.remote_decode_transfer() {
                    outbox.stage_remote_decode(rows(remote_decode))?;
                }
            }
            Streams::Decode(outbox) => {
                if let Some(remote_prefill) = plan.remote_prefill_transfer() {
                    outbox.stage_remote_prefill(rows(remote_prefill))?;
                }
            }
        }
        Ok(())
    }

    /// After the data plane ran the plan: it handed the remote stream over.
    pub fn after_execute(&self) -> Result<(), String> {
        let pending = match self {
            Streams::Prefill(outbox) => outbox.remote_decode_pending(),
            Streams::Decode(outbox) => outbox.remote_prefill_pending(),
        };
        if pending {
            return Err("the data plane never handed the plan's remote stream over".to_owned());
        }
        Ok(())
    }
}

/// Where the data plane's peer hands the host its work (see
/// `pd/host_peer.py`).
#[pyclass(frozen, name = "PdOutbox")]
pub struct Outbox {
    streams: std::sync::OnceLock<Streams>,
}

impl Outbox {
    /// An outbox whose peer starts later.
    pub fn new() -> Self {
        Self {
            streams: std::sync::OnceLock::new(),
        }
    }

    fn streams(&self) -> PyResult<&Streams> {
        self.streams
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("the host's PD peer has not started"))
    }
}

#[pymethods]
impl Outbox {
    /// The plan's remote stream may go out: its device prerequisites are
    /// met. The forward thread runs this, inside the plan's execution.
    fn remote_stream_ready(&self) -> PyResult<()> {
        match self.streams()? {
            Streams::Prefill(outbox) => outbox.hand_over_remote_decode(),
            Streams::Decode(outbox) => outbox.hand_over_remote_prefill(),
        }
        .map_err(PyRuntimeError::new_err)
    }

    /// A sleep releases the cache: deregister it from the transfer engine
    /// (refused while transfers are in flight). Blocks without the GIL.
    fn release_registration(&self, py: Python<'_>) -> PyResult<()> {
        let streams = self.streams()?;
        py.detach(|| match streams {
            Streams::Prefill(outbox) => outbox.release_registration(),
            Streams::Decode(outbox) => outbox.release_registration(),
        })
        .map_err(PyRuntimeError::new_err)
    }

    /// A wake re-mapped the cache: register it again (a Decode rank under a
    /// fresh session). Blocks without the GIL.
    fn restore_registration(&self, py: Python<'_>) -> PyResult<()> {
        let streams = self.streams()?;
        py.detach(|| match streams {
            Streams::Prefill(outbox) => outbox.restore_registration(),
            Streams::Decode(outbox) => outbox.restore_registration(),
        })
        .map_err(PyRuntimeError::new_err)
    }
}

fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(value.max(0.0))
}

fn contract(settings: &Bound<'_, PyAny>, name: &str) -> PyResult<Contract> {
    let wire = settings.getattr(name)?.cast_into::<PyBytes>()?;
    Contract::from_wire(wire.as_bytes()).map_err(|error| PyRuntimeError::new_err(error.0))
}

/// A rank's disaggregation peers.
pub struct Peers {
    /// The rank's role.
    pub role: Role,
    /// The PD transfer peer's control face, for the scheduling loop.
    pub transfer: Option<Box<dyn TransferPeer + Send>>,
    /// Its forward-thread face.
    pub streams: Option<Streams>,
    /// EPD: the embedding receiver of a multimodal Prefill rank's first
    /// stage.
    pub embeddings: Option<Box<dyn EmbeddingReceiver + Send>>,
}

/// EPD: a receiver's settings (`EpdReceiveSettings`).
fn receiver_config(epd: &Bound<'_, PyAny>) -> PyResult<ReceiverConfig> {
    let slots: usize = epd.getattr("slots")?.extract()?;
    let region = |base: &str, slot_bytes: &str| -> PyResult<Option<PoolRegion>> {
        let slot_bytes: u64 = epd.getattr(slot_bytes)?.extract()?;
        Ok((slots > 0 && slot_bytes > 0).then_some(PoolRegion {
            base: epd.getattr(base)?.extract()?,
            slot_bytes,
            slots,
        }))
    };
    Ok(ReceiverConfig {
        tp_size: epd.getattr("tp_size")?.extract()?,
        tp_rank: epd.getattr("tp_rank")?.extract()?,
        hidden: epd.getattr("hidden")?.extract()?,
        num_deepstack: epd.getattr("num_deepstack")?.extract()?,
        dtype: epd.getattr("dtype")?.extract()?,
        element_size: epd.getattr("element_size")?.extract()?,
        shard: epd.getattr("shard")?.extract()?,
        pool: region("pool_base", "slot_bytes")?,
        deepstack_pool: region("deepstack_base", "deepstack_slot_bytes")?,
        timeout: seconds(epd.getattr("timeout_s")?.extract()?),
    })
}

/// The rank's role and peers, started from the settings of the peer object
/// the bootstrap built (`HostBootstrap.transfer_peer`) and, on an EPD
/// Prefill rank's first stage, of its embedding receive (`epd`); the PD
/// peer's outcomes are counted in `metrics`, and `outbox` reaches it from
/// then on. `Role::Plain` and no peer outside disaggregation.
pub fn start(
    py: Python<'_>,
    peer: Bound<'_, PyAny>,
    epd: Bound<'_, PyAny>,
    outbox: &Outbox,
    metrics: Option<Arc<ts_metrics::KvTransferMetrics>>,
) -> PyResult<Peers> {
    let refuse_epd =
        || PyRuntimeError::new_err("EPD embeddings are received only on a PD Prefill node");
    if peer.is_none() {
        if !epd.is_none() {
            return Err(refuse_epd());
        }
        return Ok(Peers {
            role: Role::Plain,
            transfer: None,
            streams: None,
            embeddings: None,
        });
    }
    let settings = peer.getattr("settings")?;
    let role_settings = peer.getattr("role_settings")?;
    let config = PeerConfig {
        library: PathBuf::from(settings.getattr("library")?.extract::<String>()?),
        local_ip: settings.getattr("local_ip")?.extract()?,
        protocol: settings.getattr("protocol")?.extract()?,
        ib_device: settings.getattr("ib_device")?.extract()?,
        regions: settings.getattr("regions")?.extract()?,
        engine_rank: settings.getattr("engine_rank")?.extract()?,
        metrics,
    };
    let mode: String = peer.getattr("mode")?.extract()?;
    let failed = |error: ts_pd::StartError| PyRuntimeError::new_err(error.to_string());
    match mode.as_str() {
        "prefill" => {
            let payload: String = role_settings.getattr("bootstrap_payload")?.extract()?;
            let step_address: Option<usize> = peer.getattr("layerwise_step_address")?.extract()?;
            let layerwise = match step_address {
                Some(step_address) => Some(Layerwise {
                    step_address,
                    fields_by_step: role_settings.getattr("fields_by_step")?.extract()?,
                    interval: peer.getattr("layerwise_interval")?.extract()?,
                }),
                None => None,
            };
            let config = PrefillConfig {
                peer: config,
                layout: PrefillLayout {
                    local: contract(&role_settings, "layout")?,
                    wire: contract(&role_settings, "wire_layout")?,
                    arena_base: role_settings.getattr("arena_base")?.extract()?,
                    tp_size: role_settings.getattr("tp_size")?.extract()?,
                    tp_rank: role_settings.getattr("tp_rank")?.extract()?,
                    pp_rank: role_settings.getattr("pp_rank")?.extract()?,
                    fields_by_stage: role_settings.getattr("fields_by_stage")?.extract()?,
                },
                status_rank: role_settings.getattr("status_rank")?.extract()?,
                bootstrap_url: role_settings.getattr("bootstrap_url")?.extract()?,
                bootstrap_payload: serde_json::from_str(&payload)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
                bootstrap_timeout: seconds(
                    role_settings.getattr("bootstrap_timeout_s")?.extract()?,
                ),
                failed_session_ttl: seconds(
                    role_settings.getattr("failed_session_ttl_s")?.extract()?,
                ),
                transfer_workers: role_settings.getattr("transfer_workers")?.extract()?,
                layerwise,
            };
            let (peer, prefill_outbox) =
                py.detach(|| PrefillPeer::start(config)).map_err(failed)?;
            tracing::info!(
                port = peer.port(),
                session = peer.session_id(),
                "PD prefill peer up"
            );
            let embeddings = if epd.is_none() {
                None
            } else {
                let config = receiver_config(&epd)?;
                let receiver = py
                    .detach(|| peer.embedding_receiver(config))
                    .map_err(failed)?;
                Some(Box::new(receiver) as Box<dyn EmbeddingReceiver + Send>)
            };
            let streams = Streams::Prefill(prefill_outbox);
            let _ = outbox.streams.set(streams.clone());
            Ok(Peers {
                role: Role::Prefill,
                transfer: Some(Box::new(peer)),
                streams: Some(streams),
                embeddings,
            })
        }
        "decode" => {
            let config = DecodeConfig {
                peer: config,
                arena_base: role_settings.getattr("arena_base")?.extract()?,
                layout: role_settings
                    .getattr("layout")?
                    .cast_into::<PyBytes>()?
                    .as_bytes()
                    .to_vec(),
                tp_size: role_settings.getattr("tp_size")?.extract()?,
                tp_rank: role_settings.getattr("tp_rank")?.extract()?,
                waiting_timeout: seconds(role_settings.getattr("waiting_timeout_s")?.extract()?),
                heartbeat_interval: seconds(
                    role_settings.getattr("heartbeat_interval_s")?.extract()?,
                ),
                heartbeat_max_failures: role_settings
                    .getattr("heartbeat_max_failures")?
                    .extract()?,
            };
            if !epd.is_none() {
                return Err(refuse_epd());
            }
            let (peer, decode_outbox) = py.detach(|| DecodePeer::start(config)).map_err(failed)?;
            tracing::info!(
                port = peer.port(),
                session = peer.session_id(),
                "PD decode peer up"
            );
            let streams = Streams::Decode(decode_outbox);
            let _ = outbox.streams.set(streams.clone());
            Ok(Peers {
                role: Role::Decode,
                transfer: Some(Box::new(peer)),
                streams: Some(streams),
                embeddings: None,
            })
        }
        other => Err(PyRuntimeError::new_err(format!(
            "no PD peer for mode {other}"
        ))),
    }
}
