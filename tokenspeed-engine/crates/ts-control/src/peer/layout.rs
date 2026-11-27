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

/// A source in a complete joined admission may supply only some dependencies.
pub(super) fn covers_subset(
    layout: &Contract,
    begin: i32,
    end: i32,
    pages: &[PeerPageView],
) -> bool {
    if begin < 0 || begin >= end || pages.is_empty() {
        return false;
    }
    let mut seen = BTreeSet::new();
    pages.iter().all(|page| {
        seen.insert((page.group_id, page.slot))
            && layout
                .group_specs
                .get(page.group_id as usize)
                .is_some_and(|spec| {
                    logical_slot_ranges(spec, begin.into(), end.into(), false).is_ok_and(|ranges| {
                        ranges.iter().any(|r| r.contains(&i64::from(page.slot)))
                    })
                })
    })
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
    pub(super) fn local_page(
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

    fn with_draft(width: usize) -> Contract {
        Contract::from_wire(draft_wire(width).to_string().as_bytes()).unwrap()
    }
    fn draft_wire(width: usize) -> serde_json::Value {
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
        wire
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
                regions: &[(100, 4096)],
            }
        }
        for holders in [1, 8] {
            let src = with_draft(holders);
            let sg = geometry(holders);
            let dg = geometry(8);
            let sources: Vec<_> = (0..holders).map(|_| endpoint(&sg)).collect();
            for reader in 0..8 {
                let plan = PeerReadPlan::new(&src, &dst, holders, 8, reader).unwrap();
                let reads = plan
                    .reads(&sources, &endpoint(&dg), &[(page(2, 4, 3), page(2, 4, 6))])
                    .unwrap();
                assert_eq!(reads.len(), 1);
                let spans = &reads[&(reader % holders)];
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
    #[test]
    fn composed_admission_requires_complete_disjoint_final_checkpoint() {
        use crate::peer::composition::Landings;
        use ts_scheduler::{PeerFetchView, PeerLeaseView};
        let fetch = |id, holder: &str, end, destinations| PeerFetchView {
            request_id: "joined".into(),
            lease: PeerLeaseView {
                reader: "r".into(),
                id,
            },
            holder: holder.into(),
            begin_tokens: 0,
            end_tokens: end,
            destinations,
            checkpoint: vec![],
            prefix_hashes: vec![],
            tokens: vec![],
        };
        let hot = fetch(1, "hot", 4, vec![page(0, 0, 1)]);
        let storage = fetch(2, "l3@reader", 8, vec![page(0, 1, 2), page(1, 1, 1)]);
        assert!(!covers_interval(&contract(8), 0, 4, &hot.destinations));
        assert!(covers_subset(&contract(8), 0, 4, &hot.destinations));
        let partials = Landings::default()
            .register(&[hot.clone(), storage.clone()], &contract(8))
            .unwrap();
        assert_eq!(partials.into_iter().collect::<Vec<_>>(), vec![1]);
        let mut missing = storage.clone();
        missing.destinations.pop();
        assert!(Landings::default()
            .register(&[hot.clone(), missing], &contract(8))
            .is_err());
        let mut overlap = storage.clone();
        overlap.destinations.push(page(0, 0, 1));
        assert!(Landings::default()
            .register(&[hot.clone(), overlap], &contract(8))
            .is_err());
        let mut stale_state = storage.clone();
        stale_state.destinations[1].slot = 0;
        assert!(Landings::default()
            .register(&[hot, stale_state], &contract(8))
            .is_err());
    }

    #[test]
    fn storage_read_maps_owned_history_and_rank_local_state() {
        use crate::peer::{
            geometry::{HostGroup, HostPool},
            prepared::{read_spans_registered, PreparedObject},
            registered::{Member, Replica},
        };
        let members: Vec<_> = (0..8)
            .map(|rank| {
                let mut g = geometry(8);
                g.session = format!("rank{rank}");
                g.host = Some(HostPool {
                    address: 1000,
                    groups: vec![
                        HostGroup {
                            parent_bytes: 32,
                            packing: 8,
                            block_bytes: 4,
                            field_offsets: vec![("layer.0.kv".into(), 0)],
                        },
                        HostGroup {
                            parent_bytes: 1,
                            packing: 1,
                            block_bytes: 1,
                            field_offsets: vec![("layer.1.recurrent_state".into(), 0)],
                        },
                    ],
                });
                Member {
                    geometry: g,
                    bounds: vec![129, 17],
                    regions: vec![(100, 68), (200, 17)],
                }
                .encode(rank)
            })
            .collect();
        let mut replica =
            Replica::register([7; 32], contract_wire(8).to_string().as_bytes(), &members).unwrap();
        let destinations = vec![page(0, 0, 9), page(0, 1, 10), page(1, 1, 2)];
        for rank in 0..8 {
            let objects = vec![
                PreparedObject {
                    group_id: 0,
                    slot: 0,
                    session: "ssd".into(),
                    address: 1000,
                    bytes: 4,
                    fragments: Vec::new(),
                },
                PreparedObject {
                    group_id: 0,
                    slot: 1,
                    session: "ssd".into(),
                    address: 1004,
                    bytes: 4,
                    fragments: Vec::new(),
                },
                PreparedObject {
                    group_id: 1,
                    slot: 1,
                    session: "ssd".into(),
                    address: 2000 + rank as u64,
                    bytes: 1,
                    fragments: Vec::new(),
                },
            ];
            let reads = read_spans_registered(&objects, &replica, rank, &destinations).unwrap();
            let spans = &reads["ssd"];
            if rank < 2 {
                assert_eq!(spans.destinations, vec![108, 202]);
                assert_eq!(
                    spans.sources,
                    vec![1000 + rank as u64 * 4, 2000 + rank as u64]
                );
                assert_eq!(spans.lengths, vec![4, 1]);
            } else {
                assert_eq!(spans.destinations, vec![202]);
                assert_eq!(spans.sources, vec![2000 + rank as u64]);
            }
            // The selected checkpoint must exist even when this rank owns no history page.
            assert!(read_spans_registered(&objects[..2], &replica, rank, &destinations).is_err());
        }
        replica.members[0].regions = vec![(100, 4), (200, 17)];
        let object = PreparedObject {
            group_id: 0,
            slot: 0,
            session: "ssd".into(),
            address: 1000,
            bytes: 4,
            fragments: Vec::new(),
        };
        assert!(read_spans_registered(&[object], &replica, 0, &destinations[..1]).is_err());
    }
    #[test]
    fn persisted_decode_state_restores_into_eight_prefill_ranks_without_the_decode() {
        use crate::peer::{
            geometry::{HostGroup, HostPool},
            prepared::{read_spans_registered, PreparedObject},
            read::{Arena, PageReader},
            registered::{Member, Replica},
            stored_layout::StoredLayout,
            wire::{Reply, Stamp},
        };
        use ts_l3::{HostBuffer, MemoryStore, Namespace, Range, Store};
        use ts_scheduler::{
            PeerLeaseView, StorageKeyView, StoragePrefixObjectView, StoragePrefixPlanView,
        };
        let member = |width: u64, rank: usize| {
            let mut g = geometry(width);
            g.session = format!("rank{rank}");
            g.host = Some(HostPool {
                address: 1000,
                groups: vec![
                    HostGroup {
                        parent_bytes: 4,
                        packing: 1,
                        block_bytes: 4,
                        field_offsets: vec![("layer.0.kv".into(), 0)],
                    },
                    HostGroup {
                        parent_bytes: 8 / width,
                        packing: 1,
                        block_bytes: 8 / width,
                        field_offsets: vec![("layer.1.recurrent_state".into(), 0)],
                    },
                ],
            });
            Member {
                geometry: g,
                bounds: vec![129, 17],
                regions: vec![(100, 150)],
            }
            .encode(rank)
        };
        let decode = Replica::register(
            [7; 32],
            contract_wire(1).to_string().as_bytes(),
            &[member(1, 0)],
        )
        .unwrap();
        let target = Replica::register(
            [7; 32],
            contract_wire(8).to_string().as_bytes(),
            &(0..8).map(|r| member(8, r)).collect::<Vec<_>>(),
        )
        .unwrap();
        let namespace = Namespace {
            deployment_id: "decode-physical-layout".into(),
            has_draft: false,
        };
        let description = StoredLayout::from_decode(
            &decode,
            contract_wire(1).to_string().as_bytes(),
            &namespace,
            "weights1",
        )
        .unwrap();
        // The catalog uses the ordinary Store connection and survives its writer.
        use std::sync::Arc;
        let catalog_store = Arc::new(MemoryStore::new());
        let start = || {
            let layout = ts_l3::layout::HostLayout {
                groups: vec![ts_l3::layout::GroupLayout {
                    packing: 1,
                    block_bytes: 1,
                    shard_count: 1,
                }],
                lcm_block_bytes: 1,
                num_lcm_blocks: 0,
                owner_rank: 0,
                tp_rank: 0,
            };
            ts_l3::Executor::start(
                Box::new(catalog_store.clone()),
                HostBuffer { base: 1, len: 0 },
                layout,
                namespace.clone(),
                "weights1",
                ts_l3::ExecutorConfig {
                    max_operations: 4,
                    max_keys_per_operation: 16,
                    query_workers: 1,
                    read_workers: 1,
                    write_workers: 1,
                    failure_threshold: 1,
                    recovery_interval: std::time::Duration::from_secs(1),
                    clear_cache_drain_timeout: std::time::Duration::from_secs(1),
                },
                ts_l3::monotonic_clock(),
            )
        };
        let catalog_keys = super::super::catalog::keys(&description.semantic, "weights1");
        let mut publisher = start();
        let io = publisher.metadata();
        super::super::catalog::publish(&io, &catalog_keys, &description).unwrap();
        let mut alternative = description.clone();
        alternative.deployment_id = "decode-layout-v2".into();
        // Concurrent advertisements must retain both immutable physical namespaces.
        std::thread::scope(|scope| {
            scope.spawn(|| {
                super::super::catalog::publish(&io, &catalog_keys, &description).unwrap()
            });
            scope.spawn(|| {
                super::super::catalog::publish(&io, &catalog_keys, &alternative).unwrap()
            });
        });
        publisher.close(std::time::Duration::from_secs(1)).unwrap();
        let mut subscriber = start();
        let reader_io = subscriber.metadata();
        let found = super::super::catalog::discover(
            &reader_io,
            &catalog_keys,
            &description.semantic,
            "weights1",
        )
        .unwrap();
        assert_eq!(found.len(), 2);
        assert!(found.iter().any(|v| **v == description));
        assert!(found.iter().any(|v| **v == alternative));
        assert!(super::super::catalog::discover(
            &reader_io,
            &catalog_keys,
            &description.semantic,
            "wrong-weights"
        )
        .unwrap()
        .is_empty());
        reader_io.put(&catalog_keys[2], b"broken record").unwrap();
        assert_eq!(
            super::super::catalog::discover(
                &reader_io,
                &catalog_keys,
                &description.semantic,
                "weights1"
            )
            .unwrap()
            .len(),
            2
        );
        subscriber.close(std::time::Duration::from_secs(1)).unwrap();
        let prefix = ts_l3::keys::key_prefix(&namespace.deployment_id, "weights1", "");
        let keys = (0..2)
            .map(|group| ts_l3::keys::object_key(&prefix, "hash", group, 0, 0))
            .collect::<Vec<_>>();
        let metadata = StoragePrefixPlanView {
            max_tokens: 4,
            objects: (0..2)
                .map(|group| StoragePrefixObjectView {
                    key: StorageKeyView {
                        group_id: group,
                        content_hash: "hash".into(),
                        page_offset: 0,
                    },
                    slot: if group == 0 { 0 } else { 1 },
                })
                .collect(),
        };
        let store = MemoryStore::new();
        let original: Vec<u8> = vec![10, 11, 12, 13, 100, 101, 102, 103, 104, 105, 106, 107];
        let host = HostBuffer {
            base: original.as_ptr() as usize,
            len: original.len(),
        };
        assert!(store
            .batch_put_from(
                &keys,
                host,
                &[Range { offset: 0, size: 4 }, Range { offset: 4, size: 8 }]
            )
            .unwrap()
            .success
            .iter()
            .all(|v| *v));
        let persisted = serde_json::to_vec(&description).unwrap();
        store.metadata_put("layout", &persisted).unwrap();
        drop(decode);
        drop(original);
        drop(description);
        let restored: StoredLayout =
            serde_json::from_slice(&store.metadata_get("layout").unwrap().unwrap()).unwrap();
        let plan = restored
            .plan(&metadata, &target, "weights1", 16)
            .unwrap()
            .unwrap();
        assert_eq!(plan.objects.len(), 16);
        assert!(plan
            .objects
            .iter()
            .all(|o| o.key == keys[o.group_id as usize]));
        let mut staged = vec![0u8; 12];
        assert!(store
            .batch_get_into(
                &keys,
                HostBuffer {
                    base: staged.as_mut_ptr() as usize,
                    len: 12
                },
                &[Range { offset: 0, size: 4 }, Range { offset: 4, size: 8 }]
            )
            .unwrap()
            .success
            .iter()
            .all(|v| *v));
        let arena = Arena::new(2048);
        arena.write(1000, &staged);
        let ranks: Vec<_> = (0..8)
            .map(|rank| {
                plan.objects
                    .iter()
                    .filter(|o| o.rank == rank)
                    .map(|o| PreparedObject {
                        group_id: o.group_id,
                        slot: o.slot,
                        session: "ssd-provider".into(),
                        address: 1000 + if o.group_id == 0 { 0 } else { 4 },
                        bytes: o.bytes,
                        fragments: o.fragments.clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let reply = Reply::Prepared {
            stamp: Stamp {
                name: "l3@reader".into(),
                incarnation: 1,
                identity: "id".into(),
            },
            lease: PeerLeaseView {
                reader: "reader".into(),
                id: 1,
            },
            num_tokens: 4,
            ranks,
        };
        let encoded = reply.encode();
        let decoded = Reply::decode(&encoded).unwrap();
        assert_eq!(decoded, reply);
        let Reply::Prepared { ranks, .. } = decoded else {
            unreachable!()
        };
        for rank in 0..8 {
            // KV block 6 belongs to rank 5; state is one distinct head per rank.
            let reads =
                read_spans_registered(&ranks[rank], &target, rank, &[page(0, 0, 6), page(1, 1, 1)])
                    .unwrap();
            for (session, spans) in reads {
                arena.read(&session, &spans).unwrap();
            }
            assert_eq!(arena.read_bytes(201, 1), vec![100 + rank as u8]);
            if rank == 5 {
                assert_eq!(arena.read_bytes(104, 4), vec![10, 11, 12, 13]);
            }
        }
        assert!(restored
            .plan(&metadata, &target, "weights2", 16)
            .unwrap()
            .is_none());
        let mut other = target.clone();
        other.semantic = [8; 32];
        assert!(restored
            .plan(&metadata, &other, "weights1", 16)
            .unwrap()
            .is_none());
        let mut bad = ranks[0].clone();
        bad[1].fragments[0].source_offset = 8;
        assert!(read_spans_registered(&bad, &target, 0, &[page(1, 1, 1)]).is_err());
        let mut overlap = ranks[0].clone();
        let duplicate = overlap[1].fragments[0].clone();
        overlap[1].fragments.push(duplicate);
        assert!(read_spans_registered(&overlap, &target, 0, &[page(1, 1, 1)]).is_err());
    }

    #[test]
    fn persisted_draft_restores_each_strided_head_from_one_decode_object() {
        use crate::peer::{
            geometry::{HostGroup, HostPool},
            prepared::{read_spans_registered, PreparedObject},
            read::{Arena, PageReader},
            registered::{Member, Replica},
            stored_layout::StoredLayout,
        };
        use ts_scheduler::{StorageKeyView, StoragePrefixObjectView, StoragePrefixPlanView};
        let member = |width: usize, rank: usize| {
            let mut g = geometry(width as u64);
            g.session = format!("rank{rank}");
            for (id, address) in [("layer.2.k", 1024), ("layer.2.v", 2048)] {
                g.device.push(DeviceField {
                    group_id: 2,
                    field_id: id.into(),
                    address,
                    bytes: 32 / width as u64,
                    page_stride: 32 / width as u64,
                });
            }
            g.host = Some(HostPool {
                address: 10000,
                groups: vec![
                    HostGroup {
                        parent_bytes: 4,
                        packing: 1,
                        block_bytes: 4,
                        field_offsets: vec![("layer.0.kv".into(), 0)],
                    },
                    HostGroup {
                        parent_bytes: 8 / width as u64,
                        packing: 1,
                        block_bytes: 8 / width as u64,
                        field_offsets: vec![("layer.1.recurrent_state".into(), 0)],
                    },
                    HostGroup {
                        parent_bytes: 64 / width as u64,
                        packing: 1,
                        block_bytes: 64 / width as u64,
                        field_offsets: vec![
                            ("layer.2.k".into(), 0),
                            ("layer.2.v".into(), 32 / width as u64),
                        ],
                    },
                ],
            });
            Member {
                geometry: g,
                bounds: vec![129, 17, 17],
                regions: vec![(100, 4096)],
            }
            .encode(rank)
        };
        let decode = Replica::register(
            [7; 32],
            draft_wire(1).to_string().as_bytes(),
            &[member(1, 0)],
        )
        .unwrap();
        let target = Replica::register(
            [7; 32],
            draft_wire(8).to_string().as_bytes(),
            &(0..8).map(|r| member(8, r)).collect::<Vec<_>>(),
        )
        .unwrap();
        let source = StoredLayout::from_decode(
            &decode,
            draft_wire(1).to_string().as_bytes(),
            &ts_l3::Namespace {
                deployment_id: "decode-with-draft".into(),
                has_draft: true,
            },
            "v1",
        )
        .unwrap();
        let plan = source
            .plan(
                &StoragePrefixPlanView {
                    max_tokens: 4,
                    objects: vec![StoragePrefixObjectView {
                        key: StorageKeyView {
                            group_id: 2,
                            content_hash: "hash".into(),
                            page_offset: 0,
                        },
                        slot: 0,
                    }],
                },
                &target,
                "v1",
                16,
            )
            .unwrap()
            .unwrap();
        let arena = Arena::new(20000);
        arena.write(10000, &(0..64).collect::<Vec<u8>>());
        for rank in 0..8 {
            let planned = &plan.objects[rank];
            let object = PreparedObject {
                group_id: 2,
                slot: 0,
                session: "ssd".into(),
                address: 10000,
                bytes: 64,
                fragments: planned.fragments.clone(),
            };
            let reads = read_spans_registered(&[object], &target, rank, &[page(2, 0, 1)]).unwrap();
            for (session, spans) in reads {
                arena.read(&session, &spans).unwrap();
            }
            assert_eq!(
                arena.read_bytes(1028, 4),
                (0..4).map(|row| (row * 8 + rank) as u8).collect::<Vec<_>>()
            );
            assert_eq!(
                arena.read_bytes(2052, 4),
                (0..4)
                    .map(|row| (32 + row * 8 + rank) as u8)
                    .collect::<Vec<_>>()
            );
        }
    }
}
