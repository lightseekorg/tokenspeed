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

//! A Decode rank's messages to a Prefill rank, and the routes they imply:
//! the registration a Prefill rank plans its route to the Decode rank from
//! (`prepare_decode_registration`), the pre-allocation carrying a request's
//! destination manifest, and the route a Decode rank plans from the
//! bootstrap server's description of the Prefill node (`plan_route`).

use std::collections::BTreeSet;

use bytes::Bytes;
use serde_json::Value;

use super::{
    contract::Contract,
    manifest::Manifest,
    route::{pipeline_plan, validate_owner_filters, Fragment, OwnerFilters, Planner, MAX_TP_SIZE},
    LayoutError,
};
use crate::{net::parse_uint, wire};

/// A Decode rank's registration (`KVArgsRegisterInfo`).
#[derive(Debug, Clone)]
pub struct RegistrationInfo {
    /// Where its control socket listens.
    pub endpoint: String,
    /// Its port.
    pub port: u16,
    /// Its transfer session.
    pub session_id: String,
    /// Its arena base.
    pub arena_base: u64,
    /// Its contract, with its landing addresses.
    pub layout: Contract,
    /// Its node's attention TP size.
    pub tp_size: i64,
    /// Its attention TP rank.
    pub tp_rank: i64,
}

impl RegistrationInfo {
    /// Parse a registration's eight frames (`KVArgsRegisterInfo.from_zmq`).
    pub fn from_frames(frames: &[Bytes]) -> Result<Self, LayoutError> {
        let (endpoint, port, session_id) =
            wire::registration_route(frames).map_err(|error| LayoutError::new(error.0))?;
        let pointer: [u8; 8] = frames[4]
            .as_ref()
            .try_into()
            .map_err(|_| LayoutError::new("CachePD slab pointer frame must be exactly 8 bytes"))?;
        let layout = Contract::from_wire(&frames[5])?;
        let tp_size = parse_uint(&frames[6], MAX_TP_SIZE as u64)
            .filter(|&size| size >= 1)
            .ok_or_else(|| LayoutError::new("CachePD decode_tp_size is invalid"))?;
        let tp_rank = parse_uint(&frames[7], tp_size - 1)
            .ok_or_else(|| LayoutError::new("CachePD decode_tp_rank is invalid"))?;
        // struct.pack("Q"): native order.
        let arena_base = u64::from_ne_bytes(pointer);
        if arena_base == 0 {
            return Err(LayoutError::new("CachePD slab pointer must be nonzero"));
        }
        Ok(Self {
            endpoint,
            port,
            session_id,
            arena_base,
            layout,
            tp_size: tp_size as i64,
            tp_rank: tp_rank as i64,
        })
    }
}

/// A Decode rank's pre-allocation of one request (`TransferInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preallocation {
    /// The request's room.
    pub room: u64,
    /// The Decode rank's transfer session.
    pub session_id: String,
    /// Its destination manifest; `None` from a rank that only joins the
    /// status agreement.
    pub manifest: Option<Manifest>,
}

impl Preallocation {
    /// Parse a pre-allocation's three frames (`TransferInfo.from_zmq`).
    pub fn from_frames(frames: &[Bytes]) -> Result<Self, LayoutError> {
        let (room, session_id) =
            wire::preallocation_route(frames).map_err(|error| LayoutError::new(error.0))?;
        let manifest = if frames[2].is_empty() {
            None
        } else {
            Some(Manifest::from_wire(&frames[2])?)
        };
        Ok(Self {
            room,
            session_id,
            manifest,
        })
    }
}

/// What a Prefill rank plans its routes and copies with.
#[derive(Debug, Clone)]
pub struct PrefillLayout {
    /// This rank's contract, with its landing addresses: the stage's
    /// resident fields only under a pipeline.
    pub local: Contract,
    /// The logical contract every rank registers with the bootstrap server:
    /// the complete model's, without addresses.
    pub wire: Contract,
    /// This rank's arena base.
    pub arena_base: u64,
    /// The attention TP size within a stage.
    pub tp_size: i64,
    /// This rank's attention TP rank.
    pub tp_rank: i64,
    /// This rank's pipeline stage.
    pub pp_rank: i64,
    /// Every stage's resident fields.
    pub fields_by_stage: Vec<Vec<String>>,
}

/// The route a registration asks of this Prefill rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRegistration {
    /// The fragments this rank sends; empty copies every field whole.
    pub fragments: Vec<Fragment>,
    /// Its decision for every sharded group.
    pub owner_filters: OwnerFilters,
    /// This rank sends no data and only joins the status agreement.
    pub is_dummy: bool,
    /// The Decode ranks a room's pre-allocations come from.
    pub expected_decode_ranks: BTreeSet<i64>,
}

/// Plan the route a Decode rank's registration asks of this Prefill rank
/// (`prepare_decode_registration`), against the logical layout the Decode
/// rank sees on the wire; the stage's field placement keeps only the
/// fields this rank's arena holds.
pub fn plan_registration(
    layout: &PrefillLayout,
    registration: &RegistrationInfo,
) -> Result<PlannedRegistration, LayoutError> {
    let stage_fields: BTreeSet<String> = usize::try_from(layout.pp_rank)
        .ok()
        .and_then(|stage| layout.fields_by_stage.get(stage))
        .ok_or_else(|| LayoutError::new("this rank's pipeline stage has no field placement"))?
        .iter()
        .cloned()
        .collect();
    let planner = Planner::new(
        layout.tp_size,
        registration.tp_size,
        &layout.wire,
        &registration.layout,
        Some(&stage_fields),
    )?;
    let route = planner.plan_for_decode_rank(registration.tp_rank)?;
    let served = planner
        .decode_ranks_by_prefill_rank()
        .get(&layout.tp_rank)
        .cloned()
        .ok_or_else(|| LayoutError::new("this rank is outside the Prefill TP size"))?;
    let planned = if let Some(fragments) = route.fragments_by_prefill_rank.get(&layout.tp_rank) {
        // A KVP-sharded source is never idle: every rank of the chosen
        // subgroup serves the Decode rank with the blocks it owns.
        PlannedRegistration {
            fragments: fragments.clone(),
            owner_filters: route.owner_filters_by_prefill_rank[&layout.tp_rank].clone(),
            is_dummy: false,
            expected_decode_ranks: served,
        }
    } else if registration.tp_rank == 0 && served.is_empty() {
        // A rendezvous-only rank sends nothing of any group.
        PlannedRegistration {
            fragments: Vec::new(),
            owner_filters: planner
                .sharded_group_ids()
                .into_iter()
                .map(|group| (group, None))
                .collect(),
            is_dummy: true,
            expected_decode_ranks: BTreeSet::from([0]),
        }
    } else {
        return Err(LayoutError::new(
            "CachePD Decode registration targets the wrong Prefill TP rank",
        ));
    };
    // The sender applies the route's owner-filter decisions as given, so
    // they are checked against the local cache groups once, here.
    validate_owner_filters(
        &layout.local.group_specs,
        &planned.fragments,
        &planned.owner_filters,
    )?;
    Ok(planned)
}

/// The Prefill ranks a Decode rank's requests read from, planned from the
/// bootstrap server's description of the Prefill node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillRoute {
    /// The ranks to register with and send each request's manifest to,
    /// sorted.
    pub target_tp_ranks: Vec<i64>,
    /// Those of them that send no data and only join the Prefill node's
    /// status agreement; they get an empty manifest.
    pub dummy_tp_ranks: Vec<i64>,
    /// The ranks whose completion a request waits for, sorted.
    pub expected_prefill_ranks: Vec<i64>,
    /// The Prefill node's attention DP size; a room picks its DP group.
    pub prefill_dp_size: i64,
}

/// A JSON integer, or a string holding one (Python's `int(value)`).
fn integer(value: &Value, name: &str) -> Result<i64, LayoutError> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
        .ok_or_else(|| LayoutError::new(format!("the Prefill description's {name} is invalid")))
}

/// Plan a Decode rank's route from the bootstrap server's Prefill
/// description (its `engine_rank=-1` JSON; `HostDecodePeer.plan_route`).
pub fn plan_route(
    prefill_info: &str,
    decode_layout: &Contract,
    decode_tp_size: i64,
    decode_tp_rank: i64,
) -> Result<PrefillRoute, LayoutError> {
    let payload: Value = serde_json::from_str(prefill_info)
        .map_err(|_| LayoutError::new("the Prefill description is not JSON"))?;
    let field = |key: &str| {
        payload
            .get(key)
            .ok_or_else(|| LayoutError::new(format!("the Prefill description misses {key:?}")))
    };
    let layout_wire = match payload.get("cache_layout") {
        None | Some(Value::Null) => {
            return Err(LayoutError::new(
                "Cache-transfer decode connected to a non-cache-transfer prefill",
            ))
        }
        Some(Value::String(wire)) if wire.is_ascii() => wire,
        Some(_) => {
            return Err(LayoutError::new(
                "the Prefill layout must be an ASCII string",
            ))
        }
    };
    let tp_size = integer(field("prefill_tp_size")?, "prefill_tp_size")?;
    let dp_size = integer(field("prefill_dp_size")?, "prefill_dp_size")?;
    let pp_size = match payload.get("prefill_pp_size") {
        None => 1,
        Some(value) => integer(value, "prefill_pp_size")?,
    };
    let fields_by_stage = field("cache_fields_by_stage")?
        .as_array()
        .ok_or_else(|| LayoutError::new("the Prefill field placement must be a list"))?
        .iter()
        .map(|stage| {
            stage
                .as_array()
                .ok_or_else(|| LayoutError::new("a Prefill stage's fields must be a list"))?
                .iter()
                .map(|field| {
                    field
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| LayoutError::new("a Prefill field id must be a string"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prefill_layout = Contract::from_wire(layout_wire.as_bytes())?;
    if dp_size <= 0 || pp_size <= 0 {
        return Err(LayoutError::new(
            "the Prefill DP and PP sizes must be positive",
        ));
    }
    // The registered world spans the stages; the TP width is per stage.
    let prefill_tp_size = tp_size / (dp_size * pp_size);
    let (plan, dummy) = pipeline_plan(
        prefill_tp_size,
        decode_tp_size,
        decode_tp_rank,
        &prefill_layout,
        decode_layout,
        &fields_by_stage,
    )?;
    let expected = plan.target_prefill_ranks();
    let mut targets: Vec<i64> = expected.iter().chain(&dummy).copied().collect();
    targets.sort_unstable();
    Ok(PrefillRoute {
        target_tp_ranks: targets,
        dummy_tp_ranks: dummy,
        expected_prefill_ranks: expected,
        prefill_dp_size: dp_size,
    })
}
