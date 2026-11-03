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

//! One request's transferable blocks (`CachePDBlockManifest`), read off the
//! scheduler's block tables by each group's transfer policy, and the blocks
//! one layerwise prefill chunk makes ready.

use std::collections::{BTreeMap, HashSet};

use serde_json::Value;

use super::{
    contract::{load_wire_json, Contract, GroupSpec},
    LayoutError,
};

/// The largest manifest the wire carries.
pub const MAX_MANIFEST_WIRE_BYTES: usize = 2 << 20;

/// A state group's live block-table slot (`STATE_LIVE_SLOT`): the exact
/// prompt-end state the request lands in.
pub const STATE_LIVE_SLOT: i64 = 1;

/// One group's blocks, in logical slot order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupBlocks {
    /// The group.
    pub group_id: String,
    /// The publishing side's scheduler block ids (virtual ids for a
    /// KVP-sharded group).
    pub block_ids: Vec<i64>,
}

/// One request's transferable blocks per group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The groups, in the contract's group order.
    pub groups: Vec<GroupBlocks>,
    /// Prompt tokens the receiver already holds.
    pub prefix_len: i64,
    /// The prompt's length.
    pub prompt_len: i64,
    /// The receiver found every state group's prompt-end checkpoint in its
    /// own cache: no `latest_snapshot` group transfers.
    pub state_prefix_cached: bool,
}

/// Python's `bool(value)` of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

impl Manifest {
    /// Parse a manifest off the wire (`CachePDBlockManifest.from_wire_bytes`).
    pub fn from_wire(raw: &[u8]) -> Result<Self, LayoutError> {
        let name = "Cache block manifest";
        let payload = load_wire_json(raw, name, MAX_MANIFEST_WIRE_BYTES)?;
        let field = |key: &str| {
            payload
                .get(key)
                .ok_or_else(|| LayoutError::new(format!("{name} misses {key:?}")))
        };
        let int = |value: &Value, what: &str| {
            value
                .as_i64()
                .ok_or_else(|| LayoutError::new(format!("{name} {what} must be an integer")))
        };
        let groups = field("groups")?
            .as_array()
            .ok_or_else(|| LayoutError::new(format!("{name} groups must be a list")))?
            .iter()
            .map(|group| {
                let group_id = group
                    .get("group_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| LayoutError::new(format!("{name} group misses its id")))?;
                let block_ids = group
                    .get("block_ids")
                    .and_then(Value::as_array)
                    .ok_or_else(|| LayoutError::new(format!("{name} group misses its blocks")))?
                    .iter()
                    .map(|block| int(block, "block id"))
                    .collect::<Result<_, _>>()?;
                Ok(GroupBlocks {
                    group_id: group_id.to_owned(),
                    block_ids,
                })
            })
            .collect::<Result<_, LayoutError>>()?;
        Ok(Self {
            groups,
            prefix_len: int(field("prefix_len")?, "prefix_len")?,
            prompt_len: int(field("prompt_len")?, "prompt_len")?,
            state_prefix_cached: truthy(field("state_prefix_cached")?),
        })
    }

    /// The manifest's wire form (`to_wire_bytes`): the Python sender's key
    /// order and compact separators.
    pub fn to_wire(&self) -> Result<Vec<u8>, LayoutError> {
        let mut out = String::from("{\"groups\":[");
        for (index, group) in self.groups.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"group_id\":{},\"block_ids\":{}}}",
                Value::from(group.group_id.as_str()),
                Value::from(group.block_ids.clone())
            ));
        }
        out.push_str(&format!(
            "],\"prefix_len\":{},\"prompt_len\":{},\"state_prefix_cached\":{}}}",
            self.prefix_len, self.prompt_len, self.state_prefix_cached
        ));
        if out.len() > MAX_MANIFEST_WIRE_BYTES {
            return Err(LayoutError::new(format!(
                "Cache block manifest exceeds {MAX_MANIFEST_WIRE_BYTES} wire bytes"
            )));
        }
        Ok(out.into_bytes())
    }
}

/// The block-table slots one group transfers for a prompt (`_logical_slots`):
/// a `full_suffix` group every retained block from the one holding
/// `prefix_len` through the final partial block, a snapshot group its live
/// slot unless the receiver holds the prompt-end checkpoint.
pub fn logical_slots(
    spec: &GroupSpec,
    prefix_len: i64,
    prompt_len: i64,
    state_prefix_cached: bool,
) -> Result<Vec<i64>, LayoutError> {
    if spec.full_suffix() {
        let granularity = spec.block_granularity()?;
        let mut begin = prefix_len.div_euclid(granularity);
        if spec.attn_kind == "sliding_window" {
            // The next decode token attends the preceding window - 1 tokens:
            // every block intersecting that retained tail goes.
            let window = spec.sliding_window_tokens.unwrap_or(0);
            let retained_begin = (prompt_len - window + 1).max(0);
            begin = begin.max(retained_begin.div_euclid(granularity));
        }
        let end = (prompt_len + granularity - 1).div_euclid(granularity);
        return Ok((begin..end).collect());
    }
    Ok(if state_prefix_cached {
        Vec::new()
    } else {
        vec![STATE_LIVE_SLOT]
    })
}

/// Check a manifest against the contract whose blocks it names
/// (`validate_cache_manifest`).
pub fn validate_manifest(
    manifest: &Manifest,
    layout: &Contract,
    peer: &str,
) -> Result<(), LayoutError> {
    if !(0 <= manifest.prefix_len && manifest.prefix_len < manifest.prompt_len) {
        return Err(LayoutError::new(format!(
            "{peer} manifest requires 0 <= prefix_len < prompt_len, got prefix_len={} \
             prompt_len={}",
            manifest.prefix_len, manifest.prompt_len
        )));
    }
    let expected: Vec<&str> = layout
        .group_specs
        .iter()
        .map(|s| s.group_id.as_str())
        .collect();
    let actual: Vec<&str> = manifest
        .groups
        .iter()
        .map(|g| g.group_id.as_str())
        .collect();
    if actual != expected {
        return Err(LayoutError::new(format!(
            "{peer} manifest group order disagrees with layout"
        )));
    }
    for (group, spec) in manifest.groups.iter().zip(&layout.group_specs) {
        let required = logical_slots(
            spec,
            manifest.prefix_len,
            manifest.prompt_len,
            manifest.state_prefix_cached,
        )?;
        if group.block_ids.len() != required.len() {
            return Err(LayoutError::new(format!(
                "{peer} manifest group {:?} block count disagrees with its transfer policy",
                group.group_id
            )));
        }
        let capacity = layout.virtual_block_count(&spec.group_id)?;
        if group
            .block_ids
            .iter()
            .any(|&block| block <= 0 || block >= capacity)
        {
            return Err(LayoutError::new(format!(
                "{peer} manifest group {:?} has an out-of-bounds block",
                group.group_id
            )));
        }
    }
    Ok(())
}

/// One group's block table: a row of scheduler block ids per request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTable {
    /// Block-table slots per row.
    pub columns: usize,
    /// The rows, row-major.
    pub blocks: Vec<i64>,
}

impl BlockTable {
    /// A table from its rows; every row has `columns` entries.
    pub fn from_rows(rows: &[Vec<i64>], columns: usize) -> Self {
        Self {
            columns,
            blocks: rows.iter().flatten().copied().collect(),
        }
    }

    fn rows(&self) -> usize {
        self.blocks.len().checked_div(self.columns).unwrap_or(0)
    }

    fn block(&self, row: usize, slot: i64) -> Result<i64, LayoutError> {
        let slot = usize::try_from(slot)
            .map_err(|_| LayoutError::new(format!("block-table slot {slot} is negative")))?;
        if row >= self.rows() || slot >= self.columns {
            return Err(LayoutError::new(format!(
                "block-table entry ({row}, {slot}) is outside the table"
            )));
        }
        Ok(self.blocks[row * self.columns + slot])
    }
}

/// A forward's block tables by group.
pub type BlockTables = BTreeMap<String, BlockTable>;

/// The tables name exactly the contract's groups.
fn check_tables(tables: &BlockTables, layout: &Contract) -> Result<(), LayoutError> {
    let expected: HashSet<&str> = layout
        .group_specs
        .iter()
        .map(|s| s.group_id.as_str())
        .collect();
    let actual: HashSet<&str> = tables.keys().map(String::as_str).collect();
    if expected != actual {
        return Err(LayoutError::new(
            "scheduler group IDs disagree with the cache-transfer layout",
        ));
    }
    Ok(())
}

/// Read the table entries at `slots`, each a valid block of the group.
fn blocks_at(
    table: &BlockTable,
    layout: &Contract,
    group_id: &str,
    request_row: usize,
    slots: &[i64],
) -> Result<Vec<i64>, LayoutError> {
    if let Some(&last) = slots.last() {
        if last >= table.columns as i64 {
            return Err(LayoutError::new(format!(
                "table {group_id:?} misses logical slot {last}"
            )));
        }
    }
    let capacity = layout.virtual_block_count(group_id)?;
    slots
        .iter()
        .map(|&slot| {
            let block = table.block(request_row, slot)?;
            if block <= 0 || block >= capacity {
                return Err(LayoutError::new(format!(
                    "table {group_id:?} logical slot {slot} has invalid block ID {block}"
                )));
            }
            Ok(block)
        })
        .collect()
}

/// One request's manifest off its block tables
/// (`build_cache_block_manifest`).
pub fn build_manifest(
    tables: &BlockTables,
    layout: &Contract,
    request_row: usize,
    prefix_len: i64,
    prompt_len: i64,
    state_prefix_cached: bool,
) -> Result<Manifest, LayoutError> {
    if prefix_len >= prompt_len {
        return Err(LayoutError::new(
            "Cache-transfer PD requires prefix_len < prompt_len",
        ));
    }
    check_tables(tables, layout)?;
    let groups = layout
        .group_specs
        .iter()
        .map(|spec| {
            let slots = logical_slots(spec, prefix_len, prompt_len, state_prefix_cached)?;
            Ok(GroupBlocks {
                group_id: spec.group_id.clone(),
                block_ids: blocks_at(
                    &tables[&spec.group_id],
                    layout,
                    &spec.group_id,
                    request_row,
                    &slots,
                )?,
            })
        })
        .collect::<Result<_, LayoutError>>()?;
    Ok(Manifest {
        groups,
        prefix_len,
        prompt_len,
        state_prefix_cached,
    })
}

/// One group's blocks a layerwise chunk made ready, and their positions in
/// the receiver's manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupSelection {
    /// This rank's scheduler block ids.
    pub source_block_ids: Vec<i64>,
    /// Their positions in the receiver's group of its manifest.
    pub destination_positions: Vec<usize>,
}

/// The blocks one layerwise prefill chunk made ready, per group
/// (`CachePDLayerwiseBlockSelection`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// The groups, in the contract's order.
    pub groups: Vec<GroupSelection>,
}

/// The blocks that became transferable during one prefill chunk
/// (`build_cache_layerwise_block_selection`): a history group's newly
/// completed retained blocks (and the final partial one), a snapshot group's
/// prompt-end state once the final chunk wrote it.
#[allow(clippy::too_many_arguments)]
pub fn build_selection(
    tables: &BlockTables,
    layout: &Contract,
    request_row: usize,
    prefix_len: i64,
    prompt_len: i64,
    chunk_start: i64,
    chunk_end: i64,
    state_prefix_cached: bool,
) -> Result<Selection, LayoutError> {
    if prefix_len >= prompt_len {
        return Err(LayoutError::new(
            "layerwise selection requires prefix_len < prompt_len",
        ));
    }
    if chunk_start >= chunk_end {
        return Err(LayoutError::new(
            "layerwise selection requires chunk_start < chunk_end",
        ));
    }
    if chunk_end > prompt_len {
        return Err(LayoutError::new(
            "layerwise selection requires chunk_end <= prompt_len",
        ));
    }
    let is_final = chunk_end == prompt_len;
    check_tables(tables, layout)?;
    let mut groups = Vec::with_capacity(layout.group_specs.len());
    for spec in &layout.group_specs {
        let final_slots = logical_slots(spec, prefix_len, prompt_len, state_prefix_cached)?;
        let (positions, slots): (Vec<usize>, Vec<i64>) = match spec.transfer_policy.as_deref() {
            Some("full_suffix") => {
                let granularity = spec.block_granularity()?;
                let count = final_slots.len() as i64;
                let (start, ready) = match final_slots.first() {
                    Some(&first) => {
                        let ready_end = if is_final {
                            (chunk_end + granularity - 1).div_euclid(granularity)
                        } else {
                            chunk_end.div_euclid(granularity)
                        };
                        (
                            (chunk_start.div_euclid(granularity) - first).clamp(0, count),
                            (ready_end - first).clamp(0, count),
                        )
                    }
                    None => (0, 0),
                };
                (start..ready)
                    .map(|position| (position as usize, final_slots[position as usize]))
                    .unzip()
            }
            Some("latest_snapshot") => {
                let positions = if is_final && !final_slots.is_empty() {
                    vec![0]
                } else {
                    Vec::new()
                };
                let slots = if is_final { final_slots } else { Vec::new() };
                (positions, slots)
            }
            _ => {
                return Err(LayoutError::new(format!(
                    "group {:?} has no transfer policy",
                    spec.group_id
                )))
            }
        };
        groups.push(GroupSelection {
            source_block_ids: blocks_at(
                &tables[&spec.group_id],
                layout,
                &spec.group_id,
                request_row,
                &slots,
            )?,
            destination_positions: positions,
        });
    }
    Ok(Selection { groups })
}
