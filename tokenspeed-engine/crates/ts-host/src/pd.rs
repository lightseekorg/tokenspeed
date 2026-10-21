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

//! The PD transfer peer of a disaggregated rank, over the data plane's
//! planner (`pd/host_peer.py`).
//!
//! The Python peer reaches the Rust one through [`Outbox`], which the
//! bootstrap hands it before the peer exists; the Rust peer's threads call
//! the Python one as its planner, with the GIL. The control thread does
//! neither.

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};

use bytes::Bytes;
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyBytes, PyList},
};
use ts_control::{output::Role, pd::TransferPeer};
use ts_mooncake::PageGrid;
use ts_pd::{
    Copy, DecodeConfig, DecodeOutbox, DecodePeer, DecodePlanner, PeerConfig, PrefillConfig,
    PrefillOutbox, PrefillPeer, PrefillPlanner, PrefillRoute, Registration,
};

enum Inner {
    Prefill(PrefillOutbox),
    Decode(DecodeOutbox),
}

/// Where the Python peer hands the host its work (see `pd/host_peer.py`).
#[pyclass(frozen, name = "PdOutbox")]
pub struct Outbox {
    inner: OnceLock<Inner>,
}

impl Outbox {
    /// An outbox whose peer starts later.
    pub fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    fn inner(&self) -> PyResult<&Inner> {
        self.inner
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("the host's PD peer has not started"))
    }
}

fn wrong_role(method: &str) -> PyErr {
    PyRuntimeError::new_err(format!("PdOutbox.{method} is not this role's"))
}

#[pymethods]
impl Outbox {
    /// Prefill: a request's KV is ready to go to its destinations.
    #[pyo3(signature = (rid, room, bootstrap_token, spec_candidate_ids))]
    fn send(
        &self,
        rid: &str,
        room: u64,
        bootstrap_token: i64,
        spec_candidate_ids: Option<Vec<i32>>,
    ) -> PyResult<()> {
        match self.inner()? {
            Inner::Prefill(outbox) => {
                outbox.send(rid, room, bootstrap_token, spec_candidate_ids);
                Ok(())
            }
            Inner::Decode(_) => Err(wrong_role("send")),
        }
    }

    /// Prefill, layerwise streaming: a prefill chunk is armed; its
    /// intervals' KV is complete at `target_steps`.
    fn send_layerwise(
        &self,
        rid: &str,
        room: u64,
        chunk_id: u64,
        target_steps: Vec<u64>,
        is_last: bool,
    ) -> PyResult<()> {
        match self.inner()? {
            Inner::Prefill(outbox) => {
                outbox.send_layerwise(rid, room, chunk_id, target_steps, is_last);
                Ok(())
            }
            Inner::Decode(_) => Err(wrong_role("send_layerwise")),
        }
    }

    /// Prefill: a request reached its remote decode with no destination on
    /// record.
    fn send_unplanned(&self, rid: &str, room: u64) -> PyResult<()> {
        match self.inner()? {
            Inner::Prefill(outbox) => outbox
                .send_unplanned(rid, room)
                .map_err(PyRuntimeError::new_err),
            Inner::Decode(_) => Err(wrong_role("send_unplanned")),
        }
    }

    /// A sleep releases the cache: deregister it from the transfer engine
    /// (refused while transfers are in flight). Blocks without the GIL.
    fn release_registration(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner()?;
        py.detach(|| match inner {
            Inner::Prefill(outbox) => outbox.release_registration(),
            Inner::Decode(outbox) => outbox.release_registration(),
        })
        .map_err(PyRuntimeError::new_err)
    }

    /// A wake re-mapped the cache: register it again (a Decode rank under a
    /// fresh session). Blocks without the GIL.
    fn restore_registration(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner()?;
        py.detach(|| match inner {
            Inner::Prefill(outbox) => outbox.restore_registration(),
            Inner::Decode(outbox) => outbox.restore_registration(),
        })
        .map_err(PyRuntimeError::new_err)
    }

    /// Decode: a request's pages are admitted; `manifest` goes to its
    /// Prefill ranks.
    fn receive(
        &self,
        rid: &str,
        room: u64,
        request_pool_index: i32,
        manifest: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        match self.inner()? {
            Inner::Decode(outbox) => {
                outbox.receive(rid, room, request_pool_index, manifest.as_bytes());
                Ok(())
            }
            Inner::Prefill(_) => Err(wrong_role("receive")),
        }
    }
}

/// The Python peer as the planner.
struct Planner {
    peer: Py<PyAny>,
}

fn message(error: PyErr) -> String {
    error.to_string()
}

fn frames_list<'py>(py: Python<'py>, frames: &[Bytes]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, frames.iter().map(|frame| PyBytes::new(py, frame)))
}

/// A column of little-endian 8-byte values.
fn column<T>(bytes: &[u8], decode: fn([u8; 8]) -> T) -> Result<Vec<T>, String> {
    if bytes.len() % 8 != 0 {
        return Err("a copy column is not 8-byte values".to_owned());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| decode(chunk.try_into().expect("8 bytes")))
        .collect())
}

impl PrefillPlanner for Planner {
    fn prepare_registration(&self, frames: &[Bytes]) -> Result<Registration, String> {
        Python::attach(|py| {
            type Planned = (String, u16, String, i64, i64, bool, Vec<i64>);
            let planned: Planned = self
                .peer
                .call_method1(py, "prepare_registration", (frames_list(py, frames)?,))?
                .extract(py)?;
            let (endpoint, port, session_id, decode_tp_size, decode_tp_rank, is_dummy, expected) =
                planned;
            Ok(Registration {
                endpoint,
                port,
                session_id,
                decode_tp_size,
                decode_tp_rank,
                is_dummy,
                expected_decode_ranks: expected,
            })
        })
        .map_err(message)
    }

    fn reject_registration(&self, session_id: &str) {
        let rejected = Python::attach(|py| {
            self.peer
                .call_method1(py, "reject_registration", (session_id,))
                .map(drop)
        });
        if let Err(error) = rejected {
            tracing::error!(%error, "the data plane did not forget a rejected session");
        }
    }

    fn accept_preallocation(&self, frames: &[Bytes]) -> Result<bool, String> {
        Python::attach(|py| {
            self.peer
                .call_method1(py, "accept_preallocation", (frames_list(py, frames)?,))?
                .extract(py)
        })
        .map_err(message)
    }

    fn plan_send(&self, room: u64, session_id: &str) -> Result<Vec<Copy>, String> {
        let items = Python::attach(|py| {
            let items = self
                .peer
                .call_method1(py, "plan_send", (room, session_id))
                .map_err(|error| crate::pump::describe(py, &error))?;
            planned_items(items.bind(py))
        })?;
        copies(items)
    }

    fn plan_layerwise(
        &self,
        chunk_id: u64,
        sessions: &[String],
    ) -> Result<Vec<Vec<Vec<Copy>>>, String> {
        let plans: Vec<Vec<Vec<Planned>>> = Python::attach(|py| {
            let plans = self
                .peer
                .call_method1(py, "plan_layerwise", (chunk_id, sessions.to_vec()))
                .map_err(|error| crate::pump::describe(py, &error))?;
            let plans = plans
                .bind(py)
                .cast::<PyList>()
                .map_err(|error| error.to_string())?;
            plans
                .iter()
                .map(|session| {
                    let intervals = session
                        .cast::<PyList>()
                        .map_err(|error| error.to_string())?;
                    intervals
                        .iter()
                        .map(|items| planned_items(&items))
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        plans
            .into_iter()
            .map(|session| session.into_iter().map(copies).collect())
            .collect()
    }

    fn drop_room(&self, room: u64) {
        let dropped =
            Python::attach(|py| self.peer.call_method1(py, "drop_room", (room,)).map(drop));
        if let Err(error) = dropped {
            tracing::error!(%error, room, "the data plane did not drop a room");
        }
    }
}

/// One planned copy: its kind and three byte columns.
type Planned = (String, Vec<u8>, Vec<u8>, Vec<u8>);

/// A Python list of planned copies (`host_peer._copy_items`).
fn planned_items(items: &Bound<'_, PyAny>) -> Result<Vec<Planned>, String> {
    let items = items.cast::<PyList>().map_err(|error| error.to_string())?;
    items
        .iter()
        .map(|item| {
            let (kind, a, b, c): (String, Bound<PyBytes>, Bound<PyBytes>, Bound<PyBytes>) =
                item.extract().map_err(message)?;
            Ok((
                kind,
                a.as_bytes().to_vec(),
                b.as_bytes().to_vec(),
                c.as_bytes().to_vec(),
            ))
        })
        .collect()
}

/// Planned copies as the transfer engine takes them.
fn copies(items: Vec<Planned>) -> Result<Vec<Copy>, String> {
    items
        .into_iter()
        .map(|(kind, a, b, c)| match kind.as_str() {
            "pages" => {
                let fields = column(&c, i64::from_le_bytes)?;
                if fields.len() % 5 != 0 {
                    return Err("a page grid's fields are not five-wide".to_owned());
                }
                Ok(Copy::Pages(PageGrid {
                    src_pages: column(&a, i64::from_le_bytes)?,
                    dst_pages: column(&b, i64::from_le_bytes)?,
                    fields: fields
                        .chunks_exact(5)
                        .map(|field| field.try_into().expect("five values"))
                        .collect(),
                }))
            }
            "spans" => Ok(Copy::Spans {
                sources: column(&a, u64::from_le_bytes)?,
                destinations: column(&b, u64::from_le_bytes)?,
                lengths: column(&c, u64::from_le_bytes)?,
            }),
            other => Err(format!("unknown copy kind {other}")),
        })
        .collect()
}

impl DecodePlanner for Planner {
    fn plan_route(&self, prefill_info: &str) -> Result<PrefillRoute, String> {
        Python::attach(|py| {
            let (target_tp_ranks, dummy_tp_ranks, expected_prefill_ranks, prefill_dp_size): (
                Vec<i64>,
                Vec<i64>,
                Vec<i64>,
                i64,
            ) = self
                .peer
                .call_method1(py, "plan_route", (prefill_info,))?
                .extract(py)?;
            Ok(PrefillRoute {
                target_tp_ranks,
                dummy_tp_ranks,
                expected_prefill_ranks,
                prefill_dp_size,
            })
        })
        .map_err(message)
    }
}

fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(value.max(0.0))
}

/// The rank's role and PD peer, started from the Python peer the bootstrap
/// built (`HostBootstrap.transfer_peer`), with its work routed through
/// `outbox` and its outcomes counted in `metrics`. `Role::Plain` and no peer
/// outside disaggregation.
pub fn start(
    py: Python<'_>,
    peer: Bound<'_, PyAny>,
    outbox: &Outbox,
    metrics: Option<Arc<ts_metrics::KvTransferMetrics>>,
) -> PyResult<(Role, Option<Box<dyn TransferPeer + Send>>)> {
    if peer.is_none() {
        return Ok((Role::Plain, None));
    }
    let settings = peer.getattr("settings")?;
    let role_settings = peer.getattr("role_settings")?;
    let config = PeerConfig {
        library: PathBuf::from(settings.getattr("library")?.extract::<String>()?),
        local_ip: settings.getattr("local_ip")?.extract()?,
        ib_device: settings.getattr("ib_device")?.extract()?,
        regions: settings.getattr("regions")?.extract()?,
        engine_rank: settings.getattr("engine_rank")?.extract()?,
        metrics,
    };
    let mode: String = peer.getattr("mode")?.extract()?;
    let planner = Arc::new(Planner {
        peer: peer.clone().unbind(),
    });
    let failed = |error: ts_pd::StartError| PyRuntimeError::new_err(error.to_string());
    match mode.as_str() {
        "prefill" => {
            let payload: String = role_settings.getattr("bootstrap_payload")?.extract()?;
            let config = PrefillConfig {
                peer: config,
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
                layerwise_step_address: peer.getattr("layerwise_step_address")?.extract()?,
            };
            // The peer's threads take the GIL as soon as a Decode rank
            // registers, so it starts without it.
            let (peer, prefill_outbox) = py
                .detach(|| PrefillPeer::start(config, planner))
                .map_err(failed)?;
            tracing::info!(
                port = peer.port(),
                session = peer.session_id(),
                "PD prefill peer up"
            );
            let _ = outbox.inner.set(Inner::Prefill(prefill_outbox));
            Ok((Role::Prefill, Some(Box::new(peer))))
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
            let (peer, decode_outbox) = py
                .detach(|| DecodePeer::start(config, planner))
                .map_err(failed)?;
            tracing::info!(
                port = peer.port(),
                session = peer.session_id(),
                "PD decode peer up"
            );
            let _ = outbox.inner.set(Inner::Decode(decode_outbox));
            Ok((Role::Decode, Some(Box::new(peer))))
        }
        other => Err(PyRuntimeError::new_err(format!(
            "no PD peer for mode {other}"
        ))),
    }
}
