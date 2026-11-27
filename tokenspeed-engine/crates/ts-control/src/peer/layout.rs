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

//! Translate a logical peer landing into bounded per-source READ batches.
use super::geometry::{RankGeometry, Spans};
use std::collections::{BTreeMap, BTreeSet};
use ts_pd::layout::{
    contract::Contract,
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
        let (lo, hi) = if spec.attn_kind == "state" {
            (1, 2)
        } else {
            let Ok(grain) = spec.block_granularity() else {
                return false;
            };
            if grain <= 0 {
                return false;
            }
            (i64::from(begin) / grain, 1 + (i64::from(end) - 1) / grain)
        };
        let n = (hi - lo) as usize;
        if n > pages.len() - expected {
            return false;
        }
        if !(lo..hi).all(|slot| slots.contains(&(group, slot as i32))) {
            return false;
        }
        expected += n;
    }
    expected == pages.len()
}

/// Physical layout advertised by one member. Virtual bounds exclude the end.
/// Addresses for Host and Device come from the respective tier's bindings.
pub struct ReadEndpoint<'a> {
    /// Per-field Device and packed Host bindings and the source session.
    pub geometry: &'a RankGeometry,
    /// Exclusive block-id limits in the Host allocator, in scheduler order.
    pub host_virtual_bounds: &'a [i64],
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
            if !layout
                .group_specs
                .iter()
                .any(|g| g.full_suffix() && g.shard_count == width as i64)
                || layout.group_specs.iter().any(|g| {
                    g.replayable
                        || match g.attn_kind.as_str() {
                            "state" => {
                                g.shard_count != 1
                                    || g.transfer_policy.as_deref() != Some("latest_snapshot")
                            }
                            "full_history" => !g.full_suffix() || g.shard_count != width as i64,
                            _ => true,
                        }
                })
            {
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

    /// Resolve all requested slots before posting any DMA. Empty results are
    /// valid when no destination belongs to this member.
    pub fn reads(
        &self,
        sources: &[ReadEndpoint<'_>],
        dest: &ReadEndpoint<'_>,
        pages: &[(PeerPageView, PeerPageView)],
    ) -> Result<BTreeMap<usize, Spans>, LayoutError> {
        if sources.len() != self.holders || pages.is_empty() {
            return Err(LayoutError::new("missing peer members or landing slots"));
        }
        let mut slots = BTreeSet::new();
        let mut blocks = BTreeSet::new();
        let mut destinations = Vec::new();
        let mut batches = BTreeMap::<usize, Spans>::new();
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
                let src_base = source.base(self.source, src, src_page, &f.field_id)?;
                let dst_base = dest.base(self.dest, dst, dst_page, &f.field_id)?;
                let src_bytes = self.source.field(&f.field_id)?.payload_bytes() as u64;
                let dst_bytes = self.dest.field(&f.field_id)?.payload_bytes() as u64;
                let len = f.bytes_per_row as u64;
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
        Ok(batches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer::geometry::DeviceField;
    use serde_json::json;

    fn contract(width: usize) -> Contract {
        let heads = 8 / width;
        let wire = json!({
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
        });
        Contract::from_wire(wire.to_string().as_bytes()).unwrap()
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
            regions: &[(100, 150)],
        };
        let pages = [
            (page(0, 0, 3), page(0, 0, 6)),
            (page(1, 1, 1), page(1, 1, 1)),
        ];
        for reader in 0..8 {
            let plan = PeerReadPlan::new(&src, &dst, 1, 8, reader).unwrap();
            let spans = &plan
                .reads(&[endpoint(&sg)], &endpoint(&dg), &pages)
                .unwrap()[&0];
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

    #[test]
    fn invalid_coverage_registration_and_alias_are_rejected_before_read() {
        let (src, dst) = (contract(1), contract(8));
        let (sg, mut dg) = (geometry(1), geometry(8));
        let source = ReadEndpoint {
            geometry: &sg,
            host_virtual_bounds: &[],
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
            regions,
        };
        // The KV field fits, but the registered state range ends one byte early.
        assert!(plan
            .reads(
                std::slice::from_ref(&source),
                &end(&dg, &[(100, 101)]),
                &pages
            )
            .is_err());
        let short = ReadEndpoint {
            geometry: &sg,
            host_virtual_bounds: &[],
            regions: &[(100, 108)],
        };
        assert!(plan
            .reads(&[short], &end(&dg, &[(100, 150)]), &pages)
            .is_err());
        dg.device[1].address = 103;
        let alias = ReadEndpoint {
            geometry: &dg,
            host_virtual_bounds: &[],
            regions: &[(100, 150)],
        };
        assert!(plan
            .reads(std::slice::from_ref(&source), &alias, &pages)
            .unwrap_err()
            .to_string()
            .contains("alias"));
        // Shorten a planned fragment, leaving the last byte of the owned KV page uncovered.
        plan.groups[0][0].1.bytes_per_row = 0;
        assert!(plan.reads(&[source], &alias, &pages[..1]).is_err());
    }
}
