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

//! Address-free layout metadata for persistent Decode objects and Peer slices.
use super::{
    prepared::StoredFragment,
    registered::Replica,
    storage::{Error, Object, PrefixPlan},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use ts_pd::layout::{contract::Contract, route::Planner};
use ts_scheduler::StoragePrefixPlanView;

/// Packed field offsets and full staged size of one cache-group object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredGroup {
    /// Full object size including padding.
    pub bytes: u64,
    /// Field identifiers and offsets inside the packed object.
    pub fields: Vec<(String, u64)>,
}
/// One immutable physical object layout. It contains no live memory addresses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredLayout {
    /// Persistent descriptor format version.
    pub format: u32,
    /// Layout-independent model and cache compatibility digest.
    pub semantic: [u8; 32],
    /// Address-free source layout contract.
    pub contract: String,
    /// Packed source objects in source group order.
    pub groups: Vec<StoredGroup>,
    /// Original physical namespace used by the writer.
    pub deployment_id: String,
    /// Whether source keys include the draft weight generation.
    pub has_draft: bool,
    /// Immutable source weight generation.
    pub weight_version: String,
}
impl StoredLayout {
    /// V1 Peer's Decode source is TP1/KVP1; other writers retain their usual keys.
    pub fn from_decode(
        layout: &Replica,
        wire: &[u8],
        namespace: &ts_l3::Namespace,
        weight: &str,
    ) -> Result<Self, Error> {
        if layout.members.len() != 1 {
            return Err(Error(
                "persistent Peer source must be the supported TP1 Decode layout".into(),
            ));
        }
        let pool = layout.members[0]
            .geometry
            .host
            .as_ref()
            .ok_or_else(|| Error("missing source Host geometry".into()))?;
        // Capacity and plane addresses are not part of an object's packed format.
        let mut value: serde_json::Value =
            serde_json::from_slice(wire).map_err(|e| Error(e.to_string()))?;
        value["field_addresses"] = serde_json::Value::Null;
        value["plan"]["num_lcm_blocks"] = 1.into();
        if let Some(groups) = value["plan"]["groups"].as_array_mut() {
            for group in groups {
                let packing = group["cache_blocks_per_lcm_block"]
                    .as_i64()
                    .ok_or_else(|| Error("invalid source packing".into()))?;
                group["page_count"] = (1 + packing).into();
            }
        }
        let result = Self {
            format: 1,
            semantic: layout.semantic,
            contract: value.to_string(),
            groups: pool
                .groups
                .iter()
                .map(|g| StoredGroup {
                    bytes: g.block_bytes,
                    fields: g.field_offsets.clone(),
                })
                .collect(),
            deployment_id: namespace.deployment_id.clone(),
            has_draft: namespace.has_draft,
            weight_version: weight.into(),
        };
        result.parsed()?;
        Ok(result)
    }
    /// Validate packed fields and parse the supported source contract.
    pub fn parsed(&self) -> Result<Contract, Error> {
        if self.format != 1 || self.deployment_id.is_empty() || self.weight_version.is_empty() {
            return Err(Error("invalid persistent layout identity".into()));
        }
        let contract =
            Contract::from_wire(self.contract.as_bytes()).map_err(|e| Error(e.to_string()))?;
        if self.groups.len() != contract.group_specs.len()
            || contract.group_specs.iter().any(|g| g.shard_count != 1)
        {
            return Err(Error("invalid Decode object layout".into()));
        }
        for (index, group) in self.groups.iter().enumerate() {
            let mut spans = Vec::new();
            let mut fields = HashSet::new();
            for (name, offset) in &group.fields {
                let f = contract.field(name).map_err(|e| Error(e.to_string()))?;
                let end = offset
                    .checked_add(f.payload_bytes() as u64)
                    .filter(|&v| v <= group.bytes)
                    .ok_or_else(|| Error("stored field exceeds object".into()))?;
                if f.group_id != contract.group_specs[index].group_id || !fields.insert(name) {
                    return Err(Error("invalid stored field".into()));
                }
                spans.push((*offset, end));
            }
            spans.sort_unstable();
            if spans.windows(2).any(|p| p[0].1 > p[1].0)
                || fields.len()
                    != contract
                        .fields_for_group(&contract.group_specs[index].group_id)
                        .len()
            {
                return Err(Error("incomplete stored field layout".into()));
            }
        }
        Ok(contract)
    }
    /// Use the same field-partition planner as live Peer, then address fields
    /// relative to a prepared SSD object instead of a live holder's cache page.
    pub fn plan(
        &self,
        metadata: &StoragePrefixPlanView,
        target: &Replica,
        weight: &str,
        max_keys: usize,
    ) -> Result<Option<PrefixPlan>, Error> {
        if self.semantic != target.semantic
            || self.weight_version != weight
            || metadata.max_tokens <= 0
        {
            return Ok(None);
        }
        let source = self.parsed()?;
        // Retain Peer topology/group validation, not only the general PD planner.
        super::layout::PeerReadPlan::new(&source, &target.contract, 1, target.members.len(), 0)
            .map_err(|e| Error(e.to_string()))?;
        let planner = Planner::new(
            1,
            target.members.len() as i64,
            &source,
            &target.contract,
            None,
        )
        .map_err(|e| Error(e.to_string()))?;
        let prefix = ts_l3::keys::key_prefix(
            &self.deployment_id,
            weight,
            if self.has_draft { weight } else { "" },
        );
        let mut objects = Vec::new();
        let mut keys = HashSet::new();
        for rank in 0..target.members.len() {
            let route = planner
                .plan_for_decode_rank(rank as i64)
                .map_err(|e| Error(e.to_string()))?;
            let fragments = route
                .fragments_by_prefill_rank
                .get(&0)
                .ok_or_else(|| Error("missing Decode source route".into()))?;
            for object in &metadata.objects {
                let group = target
                    .contract
                    .group_specs
                    .get(object.key.group_id as usize)
                    .ok_or_else(|| Error("unknown target group".into()))?;
                let source_index = source
                    .group_specs
                    .iter()
                    .position(|g| g.group_id == group.group_id)
                    .ok_or_else(|| Error("missing source group".into()))?;
                let packed = &self.groups[source_index];
                let offsets: HashMap<_, _> = packed
                    .fields
                    .iter()
                    .map(|(id, offset)| (id.as_str(), *offset))
                    .collect();
                let mut slices = Vec::new();
                for f in fragments.iter().filter(|f| f.group_id == group.group_id) {
                    let offset = offsets
                        .get(f.field_id.as_str())
                        .ok_or_else(|| Error("missing stored field offset".into()))?;
                    let source_offset = offset
                        .checked_add(
                            u64::try_from(f.src_byte_offset).map_err(|e| Error(e.to_string()))?,
                        )
                        .ok_or_else(|| Error("source offset overflow".into()))?;
                    slices.push(StoredFragment {
                        field: f.field_id.clone(),
                        source_offset,
                        target_offset: f.dst_byte_offset as u64,
                        source_stride: f.src_row_stride_bytes as u64,
                        target_stride: f.dst_row_stride_bytes as u64,
                        bytes: f.bytes_per_row as u64,
                        rows: f.rows_per_page as u64,
                    });
                }
                if slices.is_empty() {
                    return Err(Error("empty persistent field route".into()));
                }
                let key = ts_l3::keys::object_key(
                    &prefix,
                    &object.key.content_hash,
                    source_index as u32,
                    object.key.page_offset,
                    0,
                );
                keys.insert(key.clone());
                if keys.len() > max_keys {
                    return Ok(None);
                }
                objects.push(Object {
                    key,
                    rank,
                    group_id: object.key.group_id,
                    slot: object.slot,
                    bytes: packed.bytes,
                    fragments: slices,
                });
            }
        }
        Ok(Some(PrefixPlan {
            num_tokens: metadata.max_tokens,
            members: target.members.len(),
            objects,
        }))
    }
}
