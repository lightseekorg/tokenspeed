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

//! Translate a logical peer landing into bounded per-source READ and Disk push batches.
use super::{
    geometry::{RankGeometry, Spans},
    push::{PushItem, PushSpan},
};
use std::collections::{BTreeMap, BTreeSet};
use ts_pd::layout::{
    contract::Contract,
    manifest::logical_slot_ranges,
    route::{Fragment, Planner},
    LayoutError,
};
use ts_scheduler::{PeerPageView, PeerTier};

/// Verify that the manifest/reservation covers the precise logical token
/// interval, before filtering it by this member's physical ownership.
pub fn covers_interval(layout: &Contract, begin: i32, end: i32, pages: &[PeerPageView]) -> bool {
    if begin < 0 || begin >= end {
        return false;
    }
    let slots: BTreeSet<_> = pages
        .iter()
        .map(|p| (p.group_id as usize, p.slot))
        .collect();
    if slots.len() != pages.len() {
        return false;
    }
    let mut expected = 0usize;
    for (group, spec) in layout.group_specs.iter().enumerate() {
        let Ok(ranges) = logical_slot_ranges(spec, begin.into(), end.into(), false) else {
            return false;
        };
        for mut range in ranges {
            let n = (range.end - range.start).max(0) as usize;
            if n > pages.len() - expected
                || !range.all(|slot| slots.contains(&(group, slot as i32)))
            {
                return false;
            }
            expected += n;
        }
    }
    expected == pages.len()
}

/// Physical layout advertised by one member. Virtual bounds exclude the end.
/// Addresses for Host and Device come from the respective tier's bindings;
/// Disk blocks pack identically to Host blocks and are pushed by the holder.
pub struct ReadEndpoint<'a> {
    /// Per-field Device and packed Host bindings and the source session.
    pub geometry: &'a RankGeometry,
    /// Exclusive block-id limits in the Host allocator, in scheduler order.
    pub host_virtual_bounds: &'a [i64],
    /// Exclusive block-id limits in the Disk allocator, in scheduler order;
    /// empty when this member does not serve Disk pushes.
    pub disk_virtual_bounds: &'a [i64],
    /// Registered base and length pairs on this member.
    pub regions: &'a [(u64, u64)],
}

impl ReadEndpoint<'_> {
    fn local_page(
        &self,
        layout: &Contract,
        page: PeerPageView,
        rank: usize,
    ) -> Result<Option<i64>, LayoutError> {
        let group = page.group_id as usize;
        let spec = layout
            .group_specs
            .get(group)
            .ok_or_else(|| LayoutError::new("unknown peer group"))?;
        let bound = match page.tier {
            PeerTier::Device => layout.virtual_block_count(&spec.group_id)?,
            PeerTier::Host => *self
                .host_virtual_bounds
                .get(group)
                .ok_or_else(|| LayoutError::new("missing Host block bound"))?,
            PeerTier::Disk => *self
                .disk_virtual_bounds
                .get(group)
                .ok_or_else(|| LayoutError::new("missing Disk block bound"))?,
            _ => return Err(LayoutError::new("unsupported peer tier")),
        };
        let id = i64::from(page.block_id);
        if !(1..bound).contains(&id) {
            return Err(LayoutError::new("peer block outside its tier"));
        }
        Ok(
            ((id - 1) % spec.shard_count == rank as i64 % spec.shard_count)
                .then_some(1 + (id - 1) / spec.shard_count),
        )
    }

    /// Validate that `id` fits in a packed Disk block (which shares the Host
    /// block packing) and return `(field_offset_in_block, block_bytes)`.
    fn disk_field_offset(
        &self,
        layout: &Contract,
        page: PeerPageView,
        id: &str,
    ) -> Result<(u64, u64), LayoutError> {
        let binding = self
            .geometry
            .device
            .iter()
            .find(|f| f.group_id == page.group_id && f.field_id == id)
            .ok_or_else(|| LayoutError::new("missing peer field binding"))?;
        let field = layout.field(id)?;
        if binding.bytes != field.payload_bytes() as u64
            || binding.page_stride != field.page_stride_bytes as u64
        {
            return Err(LayoutError::new(
                "peer field binding differs from logical layout",
            ));
        }
        let group = self
            .geometry
            .host
            .as_ref()
            .and_then(|pool| pool.groups.get(page.group_id as usize))
            .ok_or_else(|| LayoutError::new("missing Host peer group for Disk layout"))?;
        let offset = group
            .field_offsets
            .iter()
            .find(|(field_id, _)| field_id == id)
            .map(|(_, offset)| *offset)
            .ok_or_else(|| LayoutError::new("missing Host field offset for Disk layout"))?;
        if group.packing == 0
            || group.block_bytes == 0
            || group
                .packing
                .checked_mul(group.block_bytes)
                .is_none_or(|n| n > group.parent_bytes)
            || offset
                .checked_add(binding.bytes)
                .is_none_or(|n| n > group.block_bytes)
        {
            return Err(LayoutError::new(
                "Disk field exceeds packed block or parent",
            ));
        }
        Ok((offset, group.block_bytes))
    }

    fn base(
        &self,
        layout: &Contract,
        page: PeerPageView,
        local: i64,
        id: &str,
    ) -> Result<u64, LayoutError> {
        let binding = self
            .geometry
            .device
            .iter()
            .find(|f| f.group_id == page.group_id && f.field_id == id)
            .ok_or_else(|| LayoutError::new("missing peer field binding"))?;
        let field = layout.field(id)?;
        if binding.bytes != field.payload_bytes() as u64
            || binding.page_stride != field.page_stride_bytes as u64
        {
            return Err(LayoutError::new(
                "peer field binding differs from logical layout",
            ));
        }
        if page.tier == PeerTier::Host {
            let group = self
                .geometry
                .host
                .as_ref()
                .and_then(|pool| pool.groups.get(page.group_id as usize))
                .ok_or_else(|| LayoutError::new("missing Host peer group"))?;
            let offset = group
                .field_offsets
                .iter()
                .find(|(field_id, _)| field_id == id)
                .map(|(_, offset)| *offset)
                .ok_or_else(|| LayoutError::new("missing Host field offset"))?;
            if group.packing == 0
                || group.block_bytes == 0
                || group
                    .packing
                    .checked_mul(group.block_bytes)
                    .is_none_or(|n| n > group.parent_bytes)
                || offset
                    .checked_add(binding.bytes)
                    .is_none_or(|n| n > group.block_bytes)
            {
                return Err(LayoutError::new(
                    "Host field exceeds packed block or parent",
                ));
            }
        }
        let physical = PeerPageView {
            block_id: i32::try_from(local).map_err(|_| LayoutError::new("peer block overflow"))?,
            ..page
        };
        self.geometry
            .field_address(&physical, binding)
            .map_err(|e| LayoutError::new(e.to_string()))
    }

    fn span(&self, base: u64, offset: i128, length: u64, payload: u64) -> Result<u64, LayoutError> {
        let start = u64::try_from(offset)
            .ok()
            .filter(|n| n.checked_add(length).is_some_and(|end| end <= payload))
            .and_then(|n| base.checked_add(n))
            .ok_or_else(|| LayoutError::new("peer span exceeds its field"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| LayoutError::new("peer span overflow"))?;
        if !self
            .regions
            .iter()
            .any(|&(lo, len)| lo <= start && lo.checked_add(len).is_some_and(|hi| end <= hi))
        {
            return Err(LayoutError::new("peer span outside registered memory"));
        }
        Ok(start)
    }
}

/// One reader member's resolved transfers, split between one-sided READs
/// (Device/Host sources) and holder-pushed Disk reads, each keyed by holder rank.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlannedTransfers {
    /// One-sided READ spans per holder rank.
    pub reads: BTreeMap<usize, Spans>,
    /// Holder-pushed Disk items per holder rank.
    pub pushes: BTreeMap<usize, Vec<PushItem>>,
}

impl PlannedTransfers {
    /// Whether this reader member transfers zero bytes from any tier.
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.pushes.is_empty()
    }
}

/// Cache one reader member's fragment routes for a pair of negotiated layouts.
pub struct PeerReadPlan<'a> {
    source: &'a Contract,
    dest: &'a Contract,
    groups: Vec<Vec<(usize, Fragment)>>,
    reader: usize,
    holders: usize,
}

impl<'a> PeerReadPlan<'a> {
    /// Validate the logical layout pair and build this member's routes.
    pub fn new(
        source: &'a Contract,
        dest: &'a Contract,
        holders: usize,
        readers: usize,
        reader: usize,
    ) -> Result<Self, LayoutError> {
        if readers != 8 || !matches!(holders, 1 | 8) || reader >= readers {
            return Err(LayoutError::new("unsupported peer read topology"));
        }
        for (layout, width) in [(source, holders), (dest, readers)] {
            if !layout.group_specs.iter().any(|g| {
                g.attn_kind == "full_history" && g.full_suffix() && g.shard_count == width as i64
            }) || layout.group_specs.iter().any(|g| {
                g.replayable
                    || match g.attn_kind.as_str() {
                        "state" => {
                            g.shard_count != 1
                                || g.transfer_policy.as_deref() != Some("latest_snapshot")
                        }
                        "sliding_window" => !g.full_suffix() || g.shard_count != 1,
                        "full_history" => !g.full_suffix() || g.shard_count != width as i64,
                        _ => true,
                    }
            }) {
                return Err(LayoutError::new("unsupported peer group topology"));
            }
        }
        let planner = Planner::new(holders as i64, readers as i64, source, dest, None)?;
        let route = planner.plan_for_decode_rank(reader as i64)?;
        let indices: BTreeMap<_, _> = dest
            .group_specs
            .iter()
            .enumerate()
            .map(|(i, g)| (g.group_id.as_str(), i))
            .collect();
        let mut groups = vec![Vec::new(); indices.len()];
        for (rank, fragments) in route.fragments_by_prefill_rank {
            for f in fragments {
                groups[indices[f.group_id.as_str()]].push((rank as usize, f));
            }
        }
        Ok(Self {
            source,
            dest,
            groups,
            reader,
            holders,
        })
    }

    /// Resolve all requested slots into one-sided READs (Device/Host) and
    /// holder-pushed Disk items before posting any DMA. Empty results are
    /// valid when no destination belongs to this member.
    pub fn plan(
        &self,
        sources: &[ReadEndpoint<'_>],
        dest: &ReadEndpoint<'_>,
        pages: &[(PeerPageView, PeerPageView)],
    ) -> Result<PlannedTransfers, LayoutError> {
        if sources.len() != self.holders || pages.is_empty() {
            return Err(LayoutError::new("missing peer members or landing slots"));
        }
        let mut slots = BTreeSet::new();
        let mut blocks = BTreeSet::new();
        let mut destinations = Vec::new();
        let mut batches = BTreeMap::<usize, Spans>::new();
        let mut push_spans = BTreeMap::<(usize, u32, i32), Vec<PushSpan>>::new();
        for &(src, dst) in pages {
            if (src.group_id, src.slot) != (dst.group_id, dst.slot)
                || dst.tier != PeerTier::Device
                || !slots.insert((dst.group_id, dst.slot))
                || !blocks.insert((dst.group_id, dst.block_id))
            {
                return Err(LayoutError::new("invalid or duplicate peer landing slot"));
            }
            let Some(dst_page) = dest.local_page(self.dest, dst, self.reader)? else {
                continue;
            };
            let mut covered = BTreeMap::<&str, Vec<(u64, u64)>>::new();
            for (rank, f) in &self.groups[dst.group_id as usize] {
                let source = &sources[*rank];
                let Some(src_page) = source.local_page(self.source, src, *rank)? else {
                    continue;
                };
                let dst_base = dest.base(self.dest, dst, dst_page, &f.field_id)?;
                let src_bytes = self.source.field(&f.field_id)?.payload_bytes() as u64;
                let dst_bytes = self.dest.field(&f.field_id)?.payload_bytes() as u64;
                let len = u64::try_from(f.bytes_per_row)
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| LayoutError::new("invalid peer row byte length"))?;
                if src.tier == PeerTier::Disk {
                    if source.geometry.push.as_deref().is_none_or(str::is_empty) {
                        return Err(LayoutError::new("Disk source member has no push endpoint"));
                    }
                    let (field_offset, block_bytes) =
                        source.disk_field_offset(self.source, src, &f.field_id)?;
                    let item_spans = push_spans
                        .entry((*rank, src.group_id, src.block_id))
                        .or_default();
                    for row in 0..i128::from(f.rows_per_page) {
                        let rel = u64::try_from(
                            i128::from(f.src_byte_offset)
                                + row * i128::from(f.src_row_stride_bytes),
                        )
                        .ok()
                        .filter(|n| n.checked_add(len).is_some_and(|end| end <= src_bytes))
                        .ok_or_else(|| LayoutError::new("peer Disk span exceeds its field"))?;
                        let block_offset = field_offset
                            .checked_add(rel)
                            .filter(|n| n.checked_add(len).is_some_and(|end| end <= block_bytes))
                            .ok_or_else(|| LayoutError::new("peer Disk span exceeds its block"))?;
                        let to = dest.span(
                            dst_base,
                            i128::from(f.dst_byte_offset)
                                + row * i128::from(f.dst_row_stride_bytes),
                            len,
                            dst_bytes,
                        )?;
                        covered
                            .entry(&f.field_id)
                            .or_default()
                            .push((to - dst_base, len));
                        item_spans.push(PushSpan {
                            offset: block_offset,
                            address: to,
                            length: len,
                        });
                        destinations.push((to, to + len));
                    }
                    continue;
                }
                let src_base = source.base(self.source, src, src_page, &f.field_id)?;
                let spans = batches.entry(*rank).or_default();
                for row in 0..i128::from(f.rows_per_page) {
                    let from = source.span(
                        src_base,
                        i128::from(f.src_byte_offset) + row * i128::from(f.src_row_stride_bytes),
                        len,
                        src_bytes,
                    )?;
                    let to = dest.span(
                        dst_base,
                        i128::from(f.dst_byte_offset) + row * i128::from(f.dst_row_stride_bytes),
                        len,
                        dst_bytes,
                    )?;
                    covered
                        .entry(&f.field_id)
                        .or_default()
                        .push((to - dst_base, len));
                    spans.sources.push(from);
                    spans.destinations.push(to);
                    spans.lengths.push(len);
                    destinations.push((to, to + len));
                    if src.tier == PeerTier::Host {
                        spans.host_bytes = spans
                            .host_bytes
                            .checked_add(len)
                            .ok_or_else(|| LayoutError::new("peer byte count overflow"))?;
                    }
                }
            }
            let group = &self.dest.group_specs[dst.group_id as usize].group_id;
            for field in self.dest.fields_for_group(group) {
                let ranges = covered
                    .get_mut(field.field_id.as_str())
                    .ok_or_else(|| LayoutError::new("missing peer field"))?;
                ranges.sort_unstable();
                let mut end = 0;
                for &(start, len) in ranges.iter() {
                    if start != end {
                        return Err(LayoutError::new("peer field gap or overlap"));
                    }
                    end += len;
                }
                if end != field.payload_bytes() as u64 {
                    return Err(LayoutError::new("incomplete peer field"));
                }
            }
        }
        destinations.sort_unstable();
        if destinations.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(LayoutError::new(
                "peer fields alias the same destination bytes",
            ));
        }
        let mut pushes = BTreeMap::<usize, Vec<PushItem>>::new();
        for ((rank, group, block), spans) in push_spans {
            if !spans.is_empty() {
                pushes.entry(rank).or_default().push(PushItem {
                    group,
                    block,
                    spans,
                });
            }
        }
        Ok(PlannedTransfers {
            reads: batches,
            pushes,
        })
    }
}

#[cfg(test)]
mod disk_tests {
    use super::*;
    use crate::peer::geometry::{DeviceField, HostGroup, HostPool};

    const KV_BYTES: u64 = 64;
    const STATE_HEAD_BYTES: u64 = 16;
    const STATE_GLOBAL_HEADS: i64 = 8;

    fn hybrid_contract(width: i64) -> Contract {
        let state_heads = STATE_GLOBAL_HEADS / width;
        let state_bytes = state_heads * STATE_HEAD_BYTES as i64;
        let lcm = (KV_BYTES as i64).max(state_bytes);
        let payload = serde_json::json!({
            "plan": {
                "prefix_granularity": 4,
                "lcm_block_bytes": lcm,
                "num_lcm_blocks": 16,
                "groups": [
                    { "group_id": "full", "cache_blocks_per_lcm_block": 1, "page_count": 17 },
                    { "group_id": "state", "cache_blocks_per_lcm_block": 1, "page_count": 17 }
                ],
                "planes": [
                    { "plane_id": "kv", "bytes_per_lcm_block": KV_BYTES, "arena_offset_bytes": 0 },
                    { "plane_id": "st", "bytes_per_lcm_block": state_bytes, "arena_offset_bytes": 16 * lcm }
                ],
                "fields": [
                    {
                        "group_id": "full",
                        "field_id": "layer.0.kv",
                        "plane_id": "kv",
                        "shape": [4, 16],
                        "dtype": "uint8",
                        "field_offset_bytes": 0,
                        "page_stride_bytes": KV_BYTES
                    },
                    {
                        "group_id": "state",
                        "field_id": "layer.0.st",
                        "plane_id": "st",
                        "shape": [state_heads, STATE_HEAD_BYTES],
                        "dtype": "uint8",
                        "field_offset_bytes": 0,
                        "page_stride_bytes": state_bytes
                    }
                ]
            },
            "group_specs": [
                {
                    "group_id": "full",
                    "attn_kind": "full_history",
                    "rows_per_page": 4,
                    "entry_stride_tokens": 1,
                    "replayable": false,
                    "transfer_policy": "full_suffix",
                    "shard_count": width
                },
                {
                    "group_id": "state",
                    "attn_kind": "state",
                    "replayable": false,
                    "transfer_policy": "latest_snapshot",
                    "shard_count": 1
                }
            ],
            "transfer_schema": {
                "fields": [
                    {
                        "field_id": "layer.0.st",
                        "partition": {
                            "axis": 0,
                            "global_extent": STATE_GLOBAL_HEADS,
                            "global_parts": []
                        }
                    }
                ]
            }
        });
        Contract::from_wire(&serde_json::to_vec(&payload).unwrap()).unwrap()
    }

    struct RankFixture {
        geometry: RankGeometry,
        host_bounds: Vec<i64>,
        disk_bounds: Vec<i64>,
        regions: Vec<(u64, u64)>,
    }

    impl RankFixture {
        fn new(rank: usize, width: usize, base: u64, with_disk: bool) -> Self {
            let state_bytes = (STATE_GLOBAL_HEADS as u64 / width as u64) * STATE_HEAD_BYTES;
            let kv_dev = base + (rank as u64) * 0x10_000;
            let st_dev = kv_dev + 0x4_000;
            let host_base = kv_dev + 0x8_000;
            let parent_bytes = 256u64;
            let geometry = RankGeometry {
                session: format!("10.0.0.1:{}", 5000 + rank),
                device: vec![
                    DeviceField {
                        group_id: 0,
                        field_id: "layer.0.kv".into(),
                        address: kv_dev,
                        page_stride: KV_BYTES,
                        bytes: KV_BYTES,
                    },
                    DeviceField {
                        group_id: 1,
                        field_id: "layer.0.st".into(),
                        address: st_dev,
                        page_stride: state_bytes,
                        bytes: state_bytes,
                    },
                ],
                host: Some(HostPool {
                    address: host_base,
                    groups: vec![
                        HostGroup {
                            parent_bytes,
                            packing: 1,
                            block_bytes: KV_BYTES + 16,
                            field_offsets: vec![("layer.0.kv".into(), 16)],
                        },
                        HostGroup {
                            parent_bytes,
                            packing: 1,
                            block_bytes: state_bytes + 8,
                            field_offsets: vec![("layer.0.st".into(), 8)],
                        },
                    ],
                }),
                push: with_disk.then(|| format!("10.0.0.1:{}", 7000 + rank)),
            };
            Self {
                geometry,
                host_bounds: vec![1 + 16 * width as i64, 17],
                disk_bounds: if with_disk {
                    vec![1 + 32 * width as i64, 33]
                } else {
                    vec![]
                },
                regions: vec![(kv_dev, 0x10_000)],
            }
        }

        fn endpoint(&self) -> ReadEndpoint<'_> {
            ReadEndpoint {
                geometry: &self.geometry,
                host_virtual_bounds: &self.host_bounds,
                disk_virtual_bounds: &self.disk_bounds,
                regions: &self.regions,
            }
        }
    }

    #[test]
    fn tp8_kvp8_to_tp8_kvp8_disk_remapping_and_mixed_tiers() {
        let src_c = hybrid_contract(8);
        let dst_c = hybrid_contract(8);
        let holders: Vec<_> = (0..8)
            .map(|r| RankFixture::new(r, 8, 0x100_000, true))
            .collect();
        let readers: Vec<_> = (0..8)
            .map(|r| RankFixture::new(r, 8, 0x900_000, false))
            .collect();
        let src_eps: Vec<_> = holders.iter().map(RankFixture::endpoint).collect();

        // Slot 0: KV on Disk owned by holder rank 5 (block_id = 14 -> (14-1)%8 = 5),
        //         landing on reader rank 3 (block_id = 4 -> (4-1)%8 = 3).
        // Slot 1: KV on Host owned by holder rank 1 (block_id = 10 -> (10-1)%8 = 1),
        //         landing on reader rank 3 (block_id = 12 -> (12-1)%8 = 3).
        // State:  on Disk block_id = 7, shard_count = 1 -> holder rank 3 pushes rank 3's shard.
        let pairs = vec![
            (
                PeerPageView {
                    group_id: 0,
                    slot: 0,
                    tier: PeerTier::Disk,
                    block_id: 14,
                },
                PeerPageView {
                    group_id: 0,
                    slot: 0,
                    tier: PeerTier::Device,
                    block_id: 4,
                },
            ),
            (
                PeerPageView {
                    group_id: 0,
                    slot: 1,
                    tier: PeerTier::Host,
                    block_id: 10,
                },
                PeerPageView {
                    group_id: 0,
                    slot: 1,
                    tier: PeerTier::Device,
                    block_id: 12,
                },
            ),
            (
                PeerPageView {
                    group_id: 1,
                    slot: 1,
                    tier: PeerTier::Disk,
                    block_id: 7,
                },
                PeerPageView {
                    group_id: 1,
                    slot: 1,
                    tier: PeerTier::Device,
                    block_id: 2,
                },
            ),
        ];

        let planner = PeerReadPlan::new(&src_c, &dst_c, 8, 8, 3).unwrap();
        let planned = planner
            .plan(&src_eps, &readers[3].endpoint(), &pairs)
            .unwrap();
        // Host KV page -> one-sided READ from holder rank 1.
        assert_eq!(planned.reads.keys().copied().collect::<Vec<_>>(), vec![1]);
        assert_eq!(planned.reads[&1].host_bytes, KV_BYTES);
        // Disk KV page -> push from holder rank 5 (block_id 14, field offset 16);
        // Disk State page -> push from holder rank 3 (block_id 7, field offset 8).
        assert_eq!(
            planned.pushes.keys().copied().collect::<Vec<_>>(),
            vec![3, 5]
        );
        let kv_dst_base = readers[3].geometry.device[0].address;
        let st_dst_base = readers[3].geometry.device[1].address;
        assert_eq!(
            planned.pushes[&5],
            vec![PushItem {
                group: 0,
                block: 14,
                spans: vec![PushSpan {
                    offset: 16,
                    address: kv_dst_base + KV_BYTES, // local_page(4) = 1
                    length: KV_BYTES,
                }],
            }]
        );
        assert_eq!(
            planned.pushes[&3],
            vec![PushItem {
                group: 1,
                block: 7,
                spans: vec![PushSpan {
                    offset: 8,
                    address: st_dst_base + 2 * STATE_HEAD_BYTES,
                    length: STATE_HEAD_BYTES,
                }],
            }]
        );
    }

    #[test]
    fn tp1_kvp1_disk_to_tp8_kvp8_slices_state_and_routes_kvp8() {
        let src_c = hybrid_contract(1);
        let dst_c = hybrid_contract(8);
        let holder = RankFixture::new(0, 1, 0x100_000, true);
        let readers: Vec<_> = (0..8)
            .map(|r| RankFixture::new(r, 8, 0x900_000, false))
            .collect();
        let src_eps = vec![holder.endpoint()];

        // 8 KV pages on Disk (blocks 1..=8 on TP1 -> landing blocks 1..=8 across TP8 ranks 0..7)
        // + 1 State checkpoint on Disk (block 5 on TP1 -> landing block 3 on all 8 TP8 ranks).
        let mut pairs = Vec::new();
        for slot in 0..8 {
            pairs.push((
                PeerPageView {
                    group_id: 0,
                    slot,
                    tier: PeerTier::Disk,
                    block_id: slot + 1,
                },
                PeerPageView {
                    group_id: 0,
                    slot,
                    tier: PeerTier::Device,
                    block_id: slot + 1,
                },
            ));
        }
        pairs.push((
            PeerPageView {
                group_id: 1,
                slot: 1,
                tier: PeerTier::Disk,
                block_id: 5,
            },
            PeerPageView {
                group_id: 1,
                slot: 1,
                tier: PeerTier::Device,
                block_id: 3,
            },
        ));

        for (rank, reader) in readers.iter().enumerate() {
            let planner = PeerReadPlan::new(&src_c, &dst_c, 1, 8, rank).unwrap();
            let planned = planner.plan(&src_eps, &reader.endpoint(), &pairs).unwrap();
            assert!(planned.reads.is_empty());
            assert_eq!(planned.pushes.keys().copied().collect::<Vec<_>>(), vec![0]);
            let items = &planned.pushes[&0];
            assert_eq!(items.len(), 2);
            // Rank `rank` receives KV block `rank + 1` (local dst block 1) and
            // head slice `rank` of state block 5 (offset = field_offset(8) + rank * 16).
            assert_eq!(
                items[0],
                PushItem {
                    group: 0,
                    block: (rank + 1) as i32,
                    spans: vec![PushSpan {
                        offset: 16,
                        address: reader.geometry.device[0].address + KV_BYTES,
                        length: KV_BYTES,
                    }],
                }
            );
            assert_eq!(
                items[1],
                PushItem {
                    group: 1,
                    block: 5,
                    spans: vec![PushSpan {
                        offset: 8 + (rank as u64) * STATE_HEAD_BYTES,
                        address: reader.geometry.device[1].address + 3 * STATE_HEAD_BYTES,
                        length: STATE_HEAD_BYTES,
                    }],
                }
            );
        }

        // Negative checks: Disk source without disk_bounds, out-of-range block, missing push.
        let planner0 = PeerReadPlan::new(&src_c, &dst_c, 1, 8, 0).unwrap();
        let no_disk = RankFixture::new(0, 1, 0x100_000, false);
        assert!(planner0
            .plan(&[no_disk.endpoint()], &readers[0].endpoint(), &pairs)
            .is_err());

        let mut oob = pairs.clone();
        oob[0].0.block_id = holder.disk_bounds[0] as i32;
        assert!(planner0
            .plan(&src_eps, &readers[0].endpoint(), &oob)
            .is_err());

        let mut no_push = RankFixture::new(0, 1, 0x100_000, true);
        no_push.geometry.push = None;
        assert!(planner0
            .plan(&[no_push.endpoint()], &readers[0].endpoint(), &pairs)
            .is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer::geometry::DeviceField;
    use serde_json::json;

    fn contract(width: usize) -> Contract {
        Contract::from_wire(contract_wire(width).to_string().as_bytes()).unwrap()
    }

    fn contract_wire(width: usize) -> serde_json::Value {
        let heads = 8 / width;
        json!({
            "plan": {"prefix_granularity": 4, "num_lcm_blocks": 16, "lcm_block_bytes": 4 + heads,
                "groups": [
                    {"group_id":"kv","cache_blocks_per_lcm_block":1,"page_count":17},
                    {"group_id":"state","cache_blocks_per_lcm_block":1,"page_count":17}],
                "planes": [
                    {"plane_id":"kv","bytes_per_lcm_block":4,"arena_offset_bytes":0},
                    {"plane_id":"state","bytes_per_lcm_block":heads,"arena_offset_bytes":1024}],
                "fields": [
                    {"group_id":"kv","field_id":"layer.0.kv","plane_id":"kv","shape":[4,1],
                     "dtype":"uint8","field_offset_bytes":0,"page_stride_bytes":4},
                    {"group_id":"state","field_id":"layer.1.recurrent_state","plane_id":"state","shape":[heads,1],
                     "dtype":"uint8","field_offset_bytes":0,"page_stride_bytes":heads}],
                "resident_block_bytes":null},
            "group_specs":[
                {"group_id":"kv","attn_kind":"full_history","rows_per_page":4,"entry_stride_tokens":1,
                 "sliding_window_tokens":null,"replayable":false,"transfer_policy":"full_suffix","shard_count":width},
                {"group_id":"state","attn_kind":"state","rows_per_page":null,"entry_stride_tokens":null,
                 "sliding_window_tokens":null,"replayable":false,"transfer_policy":"latest_snapshot","shard_count":1}],
            "transfer_schema":{"fields":[{"field_id":"layer.1.recurrent_state",
                "partition":{"axis":0,"global_extent":8,"global_parts":[]}}]},"field_addresses":null
        })
    }

    fn page(group_id: u32, slot: i32, block_id: i32) -> PeerPageView {
        PeerPageView {
            group_id,
            slot,
            block_id,
            tier: PeerTier::Device,
        }
    }

    fn geometry(width: u64) -> RankGeometry {
        let field = |group_id, id: &str, address, bytes| DeviceField {
            group_id,
            field_id: id.into(),
            address,
            bytes,
            page_stride: bytes,
        };
        RankGeometry {
            session: "test".into(),
            host: None,
            push: None,
            device: vec![
                field(0, "layer.0.kv", 100, 4),
                field(1, "layer.1.recurrent_state", 200, 8 / width),
            ],
        }
    }

    #[test]
    fn interval_requires_each_history_slot_and_one_state_snapshot() {
        let layout = contract(8);
        let pages = [page(0, 1, 1), page(0, 2, 2), page(1, 1, 1)];
        assert!(covers_interval(&layout, 5, 10, &pages));
        assert!(!covers_interval(&layout, 5, 10, &pages[..2]));
        assert!(!covers_interval(&layout, 5, 10, &[pages[0], pages[2]]));
        assert!(!covers_interval(
            &layout,
            5,
            10,
            &[pages[0], pages[0], pages[2]]
        ));
    }

    #[test]
    fn cross_layout_reads_cover_owned_pages_and_the_exact_state_head() {
        let (src, dst) = (contract(1), contract(8));
        let (sg, dg) = (geometry(1), geometry(8));
        let endpoint = |geometry| ReadEndpoint {
            geometry,
            host_virtual_bounds: &[],
            disk_virtual_bounds: &[],
            regions: &[(100, 150)],
        };
        let pages = [
            (page(0, 0, 3), page(0, 0, 6)),
            (page(1, 1, 1), page(1, 1, 1)),
        ];
        for reader in 0..8 {
            let plan = PeerReadPlan::new(&src, &dst, 1, 8, reader).unwrap();
            let planned = plan.plan(&[endpoint(&sg)], &endpoint(&dg), &pages).unwrap();
            assert!(planned.pushes.is_empty());
            let spans = &planned.reads[&0];
            let mut actual: Vec<_> = spans
                .sources
                .iter()
                .copied()
                .zip(spans.destinations.iter().copied())
                .zip(spans.lengths.iter().copied())
                .collect();
            actual.sort_unstable();
            let mut expected = vec![((208 + reader as u64, 201), 1)];
            if reader == 5 {
                expected.insert(0, ((112, 104), 4));
            }
            assert_eq!(actual, expected);
        }
    }

    fn with_draft(width: usize) -> Contract {
        let mut wire = contract_wire(width);
        let heads = 8 / width;
        let page_bytes = 4 * heads;
        wire["plan"]["lcm_block_bytes"] = json!(4 + heads + 2 * page_bytes);
        wire["plan"]["groups"].as_array_mut().unwrap().push(json!({
            "group_id":"draft","cache_blocks_per_lcm_block":1,"page_count":17
        }));
        wire["group_specs"].as_array_mut().unwrap().push(json!({
            "group_id":"draft","attn_kind":"sliding_window","rows_per_page":4,"entry_stride_tokens":1,
            "sliding_window_tokens":5,"sink_tokens":2,"replayable":false,"transfer_policy":"full_suffix","shard_count":1
        }));
        for (index, id) in ["layer.2.k", "layer.2.v"].into_iter().enumerate() {
            wire["plan"]["planes"].as_array_mut().unwrap().push(json!({
                "plane_id":id, "bytes_per_lcm_block":page_bytes,"arena_offset_bytes":2048 + index * 1024
            }));
            wire["plan"]["fields"].as_array_mut().unwrap().push(json!({
                "group_id":"draft","field_id":id,"plane_id":id,"shape":[4,heads,1],
                "dtype":"uint8","field_offset_bytes":0,"page_stride_bytes":page_bytes
            }));
            wire["transfer_schema"]["fields"]
                .as_array_mut()
                .unwrap()
                .push(json!({
                    "field_id":id, "partition":{"axis":1,"global_extent":8,"global_parts":[]}
                }));
        }
        Contract::from_wire(wire.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn sliding_draft_needs_the_sink_and_retained_tail_at_the_offer_end() {
        let layout = with_draft(8);
        let mut pages: Vec<_> = (0..5).map(|slot| page(0, slot, slot + 1)).collect();
        pages.extend([page(1, 1, 1), page(2, 0, 1), page(2, 4, 5)]);
        assert!(covers_interval(&layout, 0, 20, &pages));
        for missing in [6, 7] {
            let mut incomplete = pages.clone();
            incomplete.remove(missing);
            assert!(!covers_interval(&layout, 0, 20, &incomplete));
        }
        let suffix = [page(0, 4, 5), page(1, 1, 1), page(2, 4, 5)];
        assert!(covers_interval(&layout, 16, 20, &suffix));
        assert!(!covers_interval(&layout, 0, i32::MAX, &suffix));
    }

    #[test]
    fn sliding_draft_copies_each_rows_key_and_value_for_the_readers_head() {
        let dst = with_draft(8);
        let geometry = |width: usize| {
            let mut result = geometry(width as u64);
            for (id, base) in [("layer.2.k", 1024), ("layer.2.v", 2048)] {
                let size = 4 * 8 / width as u64;
                result.device.push(DeviceField {
                    group_id: 2,
                    field_id: id.into(),
                    address: base,
                    bytes: size,
                    page_stride: size,
                });
            }
            result
        };
        fn endpoint(geometry: &RankGeometry) -> ReadEndpoint<'_> {
            ReadEndpoint {
                geometry,
                host_virtual_bounds: &[],
                disk_virtual_bounds: &[],
                regions: &[(100, 4096)],
            }
        }
        for holders in [1, 8] {
            let src = with_draft(holders);
            let sg = geometry(holders);
            let dg = geometry(8);
            let sources: Vec<_> = (0..holders).map(|_| endpoint(&sg)).collect();
            for reader in 0..8 {
                let mut plan = PeerReadPlan::new(&src, &dst, holders, 8, reader).unwrap();
                let planned = plan
                    .plan(&sources, &endpoint(&dg), &[(page(2, 4, 3), page(2, 4, 6))])
                    .unwrap();
                assert!(planned.pushes.is_empty());
                assert_eq!(planned.reads.len(), 1);
                let spans = &planned.reads[&(reader % holders)];
                let mut actual = Vec::new();
                for ((src, dst), len) in spans
                    .sources
                    .iter()
                    .zip(&spans.destinations)
                    .zip(&spans.lengths)
                {
                    actual.extend((0..*len).map(|i| (src + i, dst + i)));
                }
                let mut expected = Vec::new();
                let heads = 8 / holders as u64;
                for base in [1024, 2048] {
                    for row in 0..4 {
                        expected.push((
                            base + (3 * 4 + row) * heads + reader as u64 % heads,
                            base + 6 * 4 + row,
                        ));
                    }
                }
                actual.sort_unstable();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn invalid_coverage_registration_and_alias_are_rejected_before_read() {
        let (src, dst) = (contract(1), contract(8));
        let (sg, mut dg) = (geometry(1), geometry(8));
        let source = ReadEndpoint {
            geometry: &sg,
            host_virtual_bounds: &[],
            disk_virtual_bounds: &[],
            regions: &[(100, 150)],
        };
        let pages = [
            (page(0, 0, 1), page(0, 0, 1)),
            (page(1, 1, 1), page(1, 1, 1)),
        ];
        let mut plan = PeerReadPlan::new(&src, &dst, 1, 8, 0).unwrap();
        let end = |g, regions| ReadEndpoint {
            geometry: g,
            host_virtual_bounds: &[],
            disk_virtual_bounds: &[],
            regions,
        };
        // The KV field fits, but the registered state range ends one byte early.
        assert!(plan
            .plan(
                std::slice::from_ref(&source),
                &end(&dg, &[(100, 101)]),
                &pages
            )
            .is_err());
        let short = ReadEndpoint {
            geometry: &sg,
            host_virtual_bounds: &[],
            disk_virtual_bounds: &[],
            regions: &[(100, 108)],
        };
        assert!(plan
            .plan(&[short], &end(&dg, &[(100, 150)]), &pages)
            .is_err());
        dg.device[1].address = 103;
        let alias = ReadEndpoint {
            geometry: &dg,
            host_virtual_bounds: &[],
            disk_virtual_bounds: &[],
            regions: &[(100, 150)],
        };
        assert!(plan
            .plan(std::slice::from_ref(&source), &alias, &pages)
            .unwrap_err()
            .to_string()
            .contains("alias"));
        // Shorten a planned fragment, leaving the last byte of the owned KV page uncovered.
        plan.groups[0][0].1.bytes_per_row = 0;
        assert!(plan.plan(&[source], &alias, &pages[..1]).is_err());
    }
}
