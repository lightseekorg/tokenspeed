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

//! Where a cache block's bytes live on one rank, for a peer read: the Device
//! arena's fields, page-linear, and the Host pool's packed blocks
//! (`docs/rfcs/0004-peer-prefix-sharing.md`).
//!
//! A reader and its holder share the namespace, so their fields agree; only
//! the addresses differ. A read copies, per page, every field of the page's
//! group from the holder's tier to the reader's (its Device field, or its
//! packed Host block).

use std::collections::HashMap;

use ts_scheduler::{PeerPageView, PeerTier};

use crate::peer::push::{PushItem, PushSpan};

/// One Device field of a cache group: page `p` sits at `address + p *
/// page_stride` and holds `bytes` (a replicated group's scheduler block id is
/// its page).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceField {
    /// The cache group.
    pub group_id: u32,
    /// The field (`layer.<id>.<plane>`).
    pub field_id: String,
    /// Page 0's address.
    pub address: u64,
    /// Bytes between consecutive pages.
    pub page_stride: u64,
    /// Bytes of one page's payload.
    pub bytes: u64,
}

/// One cache group's blocks in the Host pool: `packing` packed blocks of
/// `block_bytes` per parent of `parent_bytes`, each laid out field after field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostGroup {
    /// Bytes of one parent.
    pub parent_bytes: u64,
    /// Blocks per parent.
    pub packing: u64,
    /// Bytes of one packed block.
    pub block_bytes: u64,
    /// Each field's offset within the packed block, by field id.
    pub field_offsets: Vec<(String, u64)>,
}

/// The Host pool of one rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPool {
    /// The pool's address.
    pub address: u64,
    /// Each cache group, in the scheduler's order.
    pub groups: Vec<HostGroup>,
    /// Consecutive registered memory region (MR) lengths covering this pool,
    /// starting at `address`. Empty when unsegmented / unknown.
    pub region_lengths: Vec<u64>,
}

/// What a peer needs to read a rank's cache, or to land into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankGeometry {
    /// The transfer engine session the rank's memory is registered with.
    pub session: String,
    /// Every Device field, in the order the rank lists them.
    pub device: Vec<DeviceField>,
    /// Registered Device memory regions `(address, length)` covering `device`
    /// fields; empty when unsegmented / single-region.
    pub device_regions: Vec<(u64, u64)>,
    /// The Host pool; absent without one. A Disk tier packs its blocks the
    /// same way (`docs/rfcs/0006-disk-tier.md`).
    pub host: Option<HostPool>,
    /// The rank's push endpoint (`push::PushEndpoint`): where a reader asks
    /// it to push its Disk pages, and where its own pushes are answered.
    pub push: Option<String>,
}

/// A page or geometry a read cannot place.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GeometryError {
    /// A block id at or below the null block, or one the geometry cannot reach.
    #[error("block {block} of group {group} is outside the cache")]
    Block {
        /// The group.
        group: u32,
        /// The block.
        block: i32,
    },
    /// A Host block on a rank without a Host pool.
    #[error("a Host block on a rank without a Host pool")]
    NoHost,
    /// The two ranks' fields disagree.
    #[error("field {0} is not on both ranks")]
    Field(String),
    /// A page whose tier a read cannot reach: a holder's Disk page (pushed,
    /// never read), or a reader's destination outside its Device and Host.
    #[error("a peer read neither reads a Disk page nor lands outside Device and Host")]
    Tier,
    /// An address computation overflowed.
    #[error("a peer read address overflows")]
    Overflow,
    /// A transfer slice lies outside the tier's registered memory regions.
    #[error("slice {addr:#x}+{len:#x} is outside registered {tier:?} memory regions")]
    Uncovered {
        /// Tier whose MR table rejected the slice.
        tier: PeerTier,
        /// Start address of the slice.
        addr: u64,
        /// Byte length of the slice.
        len: u64,
    },
    /// A transfer slice straddles two registered memory regions.
    #[error("slice {addr:#x}+{len:#x} straddles {tier:?} MR boundary at {region_end:#x}")]
    Straddle {
        /// Tier whose MR boundary was crossed.
        tier: PeerTier,
        /// Start address of the slice.
        addr: u64,
        /// Byte length of the slice.
        len: u64,
        /// Exclusive end address of the MR containing `addr`.
        region_end: u64,
    },
}

/// One read's copies, as the transfer engine takes them: span i copies
/// `lengths[i]` bytes from the holder's `sources[i]` to the reader's
/// `destinations[i]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Spans {
    /// Holder addresses.
    pub sources: Vec<u64>,
    /// Reader addresses.
    pub destinations: Vec<u64>,
    /// Bytes per span.
    pub lengths: Vec<u64>,
    /// Portion of the payload backed by pinned holder Host memory.
    pub host_bytes: u64,
}

impl Spans {
    /// No spans.
    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    /// Bytes the spans copy.
    pub fn total_bytes(&self) -> u64 {
        self.lengths.iter().sum()
    }
}

fn page_address(address: u64, page: i64, stride: u64) -> Result<u64, GeometryError> {
    u64::try_from(page)
        .ok()
        .and_then(|page| page.checked_mul(stride))
        .and_then(|offset| address.checked_add(offset))
        .ok_or(GeometryError::Overflow)
}

impl HostPool {
    /// The address of `group`'s packed `block`.
    fn block_address(&self, group: u32, block: i32) -> Result<u64, GeometryError> {
        let host = self
            .groups
            .get(group as usize)
            .ok_or(GeometryError::Block { group, block })?;
        let index = u64::try_from(i64::from(block) - 1)
            .map_err(|_| GeometryError::Block { group, block })?;
        if host.packing == 0 {
            return Err(GeometryError::Block { group, block });
        }
        // Widened before multiplying: block ids reach the pool's size.
        let offset = (index / host.packing)
            .checked_mul(host.parent_bytes)
            .and_then(|parent| parent.checked_add((index % host.packing) * host.block_bytes))
            .ok_or(GeometryError::Overflow)?;
        self.address
            .checked_add(offset)
            .ok_or(GeometryError::Overflow)
    }

    /// The address of `field` within `group`'s packed `block`.
    fn field_address(&self, group: u32, block: i32, field: &str) -> Result<u64, GeometryError> {
        let offset = self
            .groups
            .get(group as usize)
            .and_then(|host| host.field_offsets.iter().find(|(id, _)| id == field))
            .map(|(_, offset)| *offset)
            .ok_or_else(|| GeometryError::Field(field.to_owned()))?;
        self.block_address(group, block)?
            .checked_add(offset)
            .ok_or(GeometryError::Overflow)
    }
}

impl RankGeometry {
    /// Where `field` of `page` lives on this rank: its Device page, or its
    /// packed Host block.
    fn field_address(
        &self,
        page: &PeerPageView,
        field: &DeviceField,
    ) -> Result<u64, GeometryError> {
        match page.tier {
            PeerTier::Device => {
                let device = self
                    .device
                    .iter()
                    .find(|own| own.group_id == page.group_id && own.field_id == field.field_id)
                    .ok_or_else(|| GeometryError::Field(field.field_id.clone()))?;
                if device.bytes != field.bytes {
                    return Err(GeometryError::Field(field.field_id.clone()));
                }
                page_address(device.address, i64::from(page.block_id), device.page_stride)
            }
            PeerTier::Host => self
                .host
                .as_ref()
                .ok_or(GeometryError::NoHost)?
                .field_address(page.group_id, page.block_id, &field.field_id),
            _ => Err(GeometryError::Tier),
        }
    }
}

/// Tier-scoped MR identifier so Device MRs and Host MRs never alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MrKey {
    Device(u32),
    Host(u32),
}

/// Sorted, non-overlapping registered memory regions `[start, end)` for one tier.
#[derive(Debug, Clone)]
pub struct RegionIndex {
    tier: PeerTier,
    /// `(start, end, original_index)` sorted by `start`, with `start < end`.
    intervals: Vec<(u64, u64, u32)>,
}

impl RegionIndex {
    /// Build a `Host` `RegionIndex` from `pool.region_lengths` (returns `Ok(None)`
    /// when `pool` is `None` or unsegmented).
    pub fn from_host(pool: Option<&HostPool>) -> Result<Option<Self>, GeometryError> {
        let Some(pool) = pool else {
            return Ok(None);
        };
        if pool.region_lengths.is_empty() {
            return Ok(None);
        }
        let mut intervals = Vec::with_capacity(pool.region_lengths.len());
        let mut cursor = pool.address;
        for (idx, &len) in pool.region_lengths.iter().enumerate() {
            if len == 0 {
                return Err(GeometryError::Uncovered {
                    tier: PeerTier::Host,
                    addr: cursor,
                    len: 0,
                });
            }
            let end = cursor.checked_add(len).ok_or(GeometryError::Overflow)?;
            intervals.push((cursor, end, idx as u32));
            cursor = end;
        }
        Ok(Some(Self {
            tier: PeerTier::Host,
            intervals,
        }))
    }

    /// Build a `Device` `RegionIndex` from explicit `(address, length)` MRs
    /// (returns `Ok(None)` when `regions` is empty / unsegmented).
    pub fn from_device(regions: &[(u64, u64)]) -> Result<Option<Self>, GeometryError> {
        if regions.is_empty() {
            return Ok(None);
        }
        let mut intervals = Vec::with_capacity(regions.len());
        for (idx, &(addr, len)) in regions.iter().enumerate() {
            if len == 0 {
                return Err(GeometryError::Uncovered {
                    tier: PeerTier::Device,
                    addr,
                    len: 0,
                });
            }
            let end = addr.checked_add(len).ok_or(GeometryError::Overflow)?;
            intervals.push((addr, end, idx as u32));
        }
        intervals.sort_unstable_by_key(|&(start, _, _)| start);
        for pair in intervals.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(GeometryError::Straddle {
                    tier: PeerTier::Device,
                    addr: pair[1].0,
                    len: pair[1].1 - pair[1].0,
                    region_end: pair[0].1,
                });
            }
        }
        Ok(Some(Self {
            tier: PeerTier::Device,
            intervals,
        }))
    }

    /// Resolve `[addr, addr + len - 1]` to its single enclosing MR index, or
    /// fail closed on uncovered addresses or cross-MR straddles.
    pub fn region_of(&self, addr: u64, len: u64) -> Result<u32, GeometryError> {
        if len == 0 {
            return Err(GeometryError::Uncovered {
                tier: self.tier,
                addr,
                len,
            });
        }
        let last = addr.checked_add(len - 1).ok_or(GeometryError::Overflow)?;
        let pos = self
            .intervals
            .partition_point(|&(start, _, _)| start <= addr);
        if pos == 0 {
            return Err(GeometryError::Uncovered {
                tier: self.tier,
                addr,
                len,
            });
        }
        let (_start, end, region_id) = self.intervals[pos - 1];
        if addr >= end {
            return Err(GeometryError::Uncovered {
                tier: self.tier,
                addr,
                len,
            });
        }
        if last >= end {
            return Err(GeometryError::Straddle {
                tier: self.tier,
                addr,
                len,
                region_end: end,
            });
        }
        Ok(region_id)
    }
}

/// Per-tier (`Device` and `Host`) MR indexes for one rank.
struct TierRegions {
    device: Option<RegionIndex>,
    host: Option<RegionIndex>,
    host_base: Option<u64>,
}

impl TierRegions {
    fn from_geometry(geometry: &RankGeometry) -> Result<Self, GeometryError> {
        Ok(Self {
            device: RegionIndex::from_device(&geometry.device_regions)?,
            host: RegionIndex::from_host(geometry.host.as_ref())?,
            host_base: geometry.host.as_ref().map(|h| h.address),
        })
    }

    fn is_unsegmented(&self) -> bool {
        self.device.is_none() && self.host.is_none()
    }

    fn key_for_tier(&self, tier: PeerTier, addr: u64, len: u64) -> Result<MrKey, GeometryError> {
        if len == 0 {
            return Err(GeometryError::Uncovered { tier, addr, len });
        }
        let _last = addr.checked_add(len - 1).ok_or(GeometryError::Overflow)?;
        match tier {
            PeerTier::Device => match &self.device {
                Some(idx) => idx.region_of(addr, len).map(MrKey::Device),
                None => Ok(MrKey::Device(0)),
            },
            PeerTier::Host => match &self.host {
                Some(idx) => idx.region_of(addr, len).map(MrKey::Host),
                None => {
                    let base = self.host_base.ok_or(GeometryError::NoHost)?;
                    if addr < base {
                        return Err(GeometryError::Uncovered { tier, addr, len });
                    }
                    Ok(MrKey::Host(0))
                }
            },
            _ => Err(GeometryError::Tier),
        }
    }

    /// Infer the tier + MR key of `[addr, addr + len - 1]` when splitting raw
    /// `Spans` without explicit per-span tier tags.
    fn key_for_address(
        &self,
        geometry: &RankGeometry,
        addr: u64,
        len: u64,
    ) -> Result<MrKey, GeometryError> {
        if len == 0 {
            return Err(GeometryError::Uncovered {
                tier: PeerTier::Device,
                addr,
                len,
            });
        }
        let _last = addr.checked_add(len - 1).ok_or(GeometryError::Overflow)?;
        if let Some(host_idx) = &self.host {
            if let Ok(id) = host_idx.region_of(addr, len) {
                return Ok(MrKey::Host(id));
            }
            // If `addr` starts inside the Host pool's registered range, it
            // straddled an MR boundary — fail closed immediately!
            let first_start = host_idx.intervals[0].0;
            let last_end = host_idx.intervals.last().expect("non-empty").1;
            if addr >= first_start && addr < last_end {
                return host_idx.region_of(addr, len).map(MrKey::Host);
            }
        }
        if let Some(dev_idx) = &self.device {
            return dev_idx.region_of(addr, len).map(MrKey::Device);
        }
        // When `device_regions` is unsegmented (`None`), verify `addr` belongs
        // to a Device field if `host` is segmented; otherwise fail closed so an
        // uncovered Host tail (past `last_end`) never silently aliases Device MR 0!
        if self.host.is_some() {
            let matches_device_field = geometry.device.iter().any(|field| {
                addr >= field.address
                    && field.page_stride > 0
                    && (addr - field.address) % field.page_stride == 0
                    && len == field.bytes
            });
            if !matches_device_field {
                return Err(GeometryError::Uncovered {
                    tier: PeerTier::Host,
                    addr,
                    len,
                });
            }
            return Ok(MrKey::Device(0));
        }
        if let Some(host_base) = self.host_base {
            if addr >= host_base {
                return Ok(MrKey::Host(0));
            }
        }
        Ok(MrKey::Device(0))
    }
}

/// Split `spans` strictly along registered `Device` and `Host` MR boundaries
/// so every returned `Spans` batch has all source spans inside ONE MR and all
/// destination spans inside ONE MR. Fails closed (`Err`) if any span is
/// uncovered or straddles an MR boundary.
pub fn split_spans_by_mr(
    spans: &Spans,
    holder: &RankGeometry,
    reader: &RankGeometry,
) -> Result<Vec<Spans>, GeometryError> {
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    let src_regions = TierRegions::from_geometry(holder)?;
    let dst_regions = TierRegions::from_geometry(reader)?;
    if src_regions.is_unsegmented() && dst_regions.is_unsegmented() {
        return Ok(vec![spans.clone()]);
    }

    let mut buckets: Vec<Spans> = Vec::new();
    let mut index_by_pair: HashMap<(MrKey, MrKey), usize> = HashMap::new();
    let all_host = spans.host_bytes == spans.total_bytes();

    for ((&src, &dst), &len) in spans
        .sources
        .iter()
        .zip(&spans.destinations)
        .zip(&spans.lengths)
    {
        let s_key = src_regions.key_for_address(holder, src, len)?;
        let d_key = dst_regions.key_for_address(reader, dst, len)?;
        let bucket_idx = *index_by_pair.entry((s_key, d_key)).or_insert_with(|| {
            let idx = buckets.len();
            buckets.push(Spans::default());
            idx
        });
        let bucket = &mut buckets[bucket_idx];
        bucket.sources.push(src);
        bucket.destinations.push(dst);
        bucket.lengths.push(len);
        if matches!(s_key, MrKey::Host(_)) && (src_regions.host.is_some() || all_host) {
            bucket.host_bytes += len;
        }
    }

    if buckets.len() == 1 {
        buckets[0].host_bytes = spans.host_bytes;
    }
    Ok(buckets)
}

/// Compute `read_spans` and split them into MR-homogeneous `Spans` batches
/// using each page's exact `(source.tier, destination.tier)` `RegionIndex`.
pub fn read_batches(
    holder: &RankGeometry,
    reader: &RankGeometry,
    pages: &[(PeerPageView, PeerPageView)],
) -> Result<(u64, Vec<Spans>), GeometryError> {
    if pages.is_empty() {
        return Ok((0, Vec::new()));
    }
    let src_regions = TierRegions::from_geometry(holder)?;
    let dst_regions = TierRegions::from_geometry(reader)?;
    if src_regions.is_unsegmented() && dst_regions.is_unsegmented() {
        let spans = read_spans(holder, reader, pages)?;
        let total = spans.total_bytes();
        let batches = if spans.is_empty() {
            Vec::new()
        } else {
            vec![spans]
        };
        return Ok((total, batches));
    }

    let mut buckets: Vec<Spans> = Vec::new();
    let mut index_by_pair: HashMap<(MrKey, MrKey), usize> = HashMap::new();
    let mut total_bytes = 0u64;

    for (source, destination) in pages {
        if destination.group_id != source.group_id
            || source.tier == PeerTier::Disk
            || destination.tier == PeerTier::Disk
        {
            return Err(GeometryError::Tier);
        }
        let group = source.group_id;
        if source.block_id <= 0 || destination.block_id <= 0 {
            return Err(GeometryError::Block {
                group,
                block: source.block_id.min(destination.block_id),
            });
        }
        for field in holder.device.iter().filter(|field| field.group_id == group) {
            let src_addr = holder.field_address(source, field)?;
            let dst_addr = reader.field_address(destination, field)?;
            let len = field.bytes;
            let s_key = src_regions.key_for_tier(source.tier, src_addr, len)?;
            let d_key = dst_regions.key_for_tier(destination.tier, dst_addr, len)?;
            let bucket_idx = *index_by_pair.entry((s_key, d_key)).or_insert_with(|| {
                let idx = buckets.len();
                buckets.push(Spans::default());
                idx
            });
            let bucket = &mut buckets[bucket_idx];
            bucket.sources.push(src_addr);
            bucket.destinations.push(dst_addr);
            bucket.lengths.push(len);
            if source.tier == PeerTier::Host {
                bucket.host_bytes += len;
            }
            total_bytes += len;
        }
    }
    Ok((total_bytes, buckets))
}

/// The pushes that land `pages` whose holder pages are on its Disk tier:
/// per page, every field of the page's group, at its offset in the holder's
/// packed block (a Disk block packs like a Host block), to the reader's tier.
///
/// Validates every destination span against the reader's per-tier (`Host` or
/// `Device`) `RegionIndex`, failing closed on uncovered or straddling ranges,
/// and splits spans across destination MRs when needed.
pub fn push_items(
    holder: &RankGeometry,
    reader: &RankGeometry,
    pages: &[(PeerPageView, PeerPageView)],
) -> Result<Vec<PushItem>, GeometryError> {
    let pool = holder.host.as_ref().ok_or(GeometryError::NoHost)?;
    let reader_regions = TierRegions::from_geometry(reader)?;
    let mut items = Vec::with_capacity(pages.len());
    for (source, destination) in pages {
        if destination.group_id != source.group_id
            || source.tier != PeerTier::Disk
            || destination.tier == PeerTier::Disk
        {
            return Err(GeometryError::Tier);
        }
        let group = source.group_id;
        if source.block_id <= 0 || destination.block_id <= 0 {
            return Err(GeometryError::Block {
                group,
                block: source.block_id.min(destination.block_id),
            });
        }
        let host = pool
            .groups
            .get(group as usize)
            .ok_or(GeometryError::Block {
                group,
                block: source.block_id,
            })?;
        let mut grouped: Vec<Vec<PushSpan>> = Vec::new();
        let mut by_mr: HashMap<MrKey, usize> = HashMap::new();
        for field in holder.device.iter().filter(|field| field.group_id == group) {
            let offset = host
                .field_offsets
                .iter()
                .find(|(id, _)| *id == field.field_id)
                .map(|(_, offset)| *offset)
                .ok_or_else(|| GeometryError::Field(field.field_id.clone()))?;
            let address = reader.field_address(destination, field)?;
            let length = field.bytes;
            let key = reader_regions.key_for_tier(destination.tier, address, length)?;
            let idx = *by_mr.entry(key).or_insert_with(|| {
                let i = grouped.len();
                grouped.push(Vec::new());
                i
            });
            grouped[idx].push(PushSpan {
                offset,
                address,
                length,
            });
        }
        if grouped.is_empty() {
            items.push(PushItem {
                group,
                block: source.block_id,
                spans: Vec::new(),
            });
        } else {
            for mr_spans in grouped {
                items.push(PushItem {
                    group,
                    block: source.block_id,
                    spans: mr_spans,
                });
            }
        }
    }
    Ok(items)
}

/// The copies that land `pages` -- each a holder page and the reader's
/// destination for the same group and slot -- every field of the page's
/// group, from the holder's tier to the reader's (its Device or its Host
/// pool). A holder's Disk page is pushed, never read (`push`).
pub fn read_spans(
    holder: &RankGeometry,
    reader: &RankGeometry,
    pages: &[(PeerPageView, PeerPageView)],
) -> Result<Spans, GeometryError> {
    let mut spans = Spans::default();
    for (source, destination) in pages {
        if destination.group_id != source.group_id
            || source.tier == PeerTier::Disk
            || destination.tier == PeerTier::Disk
        {
            return Err(GeometryError::Tier);
        }
        let group = source.group_id;
        if source.block_id <= 0 || destination.block_id <= 0 {
            return Err(GeometryError::Block {
                group,
                block: source.block_id.min(destination.block_id),
            });
        }
        for field in holder.device.iter().filter(|field| field.group_id == group) {
            spans.sources.push(holder.field_address(source, field)?);
            spans
                .destinations
                .push(reader.field_address(destination, field)?);
            if source.tier == PeerTier::Host {
                spans.host_bytes += field.bytes;
            }
            spans.lengths.push(field.bytes);
        }
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE: PeerTier = PeerTier::Device;
    const HOST: PeerTier = PeerTier::Host;

    fn page(group: u32, slot: i32, tier: PeerTier, block: i32) -> PeerPageView {
        PeerPageView {
            group_id: group,
            slot,
            tier,
            block_id: block,
        }
    }

    fn geometry(base: u64, host: Option<HostPool>) -> RankGeometry {
        RankGeometry {
            session: format!("session-{base}"),
            device: vec![
                DeviceField {
                    group_id: 0,
                    field_id: "layer.0.k".into(),
                    address: base,
                    page_stride: 100,
                    bytes: 40,
                },
                DeviceField {
                    group_id: 0,
                    field_id: "layer.0.v".into(),
                    address: base + 10_000,
                    page_stride: 100,
                    bytes: 60,
                },
            ],
            device_regions: Vec::new(),
            host,
            push: None,
        }
    }

    fn host_pool(address: u64) -> HostPool {
        HostPool {
            address,
            groups: vec![HostGroup {
                parent_bytes: 300,
                packing: 3,
                block_bytes: 100,
                field_offsets: vec![("layer.0.k".into(), 0), ("layer.0.v".into(), 40)],
            }],
            region_lengths: Vec::new(),
        }
    }

    #[test]
    fn a_device_page_lands_every_field_page_to_page() {
        let holder = geometry(1_000_000, None);
        let reader = geometry(2_000_000, None);
        let spans = read_spans(
            &holder,
            &reader,
            &[(page(0, 0, DEVICE, 7), page(0, 0, DEVICE, 3))],
        )
        .unwrap();
        assert_eq!(spans.sources, vec![1_000_700, 1_010_700]);
        assert_eq!(spans.destinations, vec![2_000_300, 2_010_300]);
        assert_eq!(spans.lengths, vec![40, 60]);
        assert_eq!(spans.host_bytes, 0);
    }

    #[test]
    fn a_host_page_lands_from_its_packed_block() {
        let holder = geometry(1_000_000, Some(host_pool(5_000_000)));
        let reader = geometry(2_000_000, None);
        // Block 5 is the second block of the second parent.
        let spans = read_spans(
            &holder,
            &reader,
            &[(page(0, 1, HOST, 5), page(0, 1, DEVICE, 4))],
        )
        .unwrap();
        assert_eq!(spans.sources, vec![5_000_400, 5_000_440]);
        assert_eq!(spans.destinations, vec![2_000_400, 2_010_400]);
        assert_eq!(spans.total_bytes(), 100);
        assert_eq!(spans.host_bytes, 100);
        let mixed = read_spans(
            &holder,
            &reader,
            &[
                (page(0, 1, HOST, 5), page(0, 1, DEVICE, 4)),
                (page(0, 0, DEVICE, 7), page(0, 0, DEVICE, 3)),
            ],
        )
        .unwrap();
        assert_eq!((mixed.total_bytes(), mixed.host_bytes), (200, 100));
    }

    #[test]
    fn a_page_lands_into_the_readers_host_block() {
        let holder = geometry(1_000_000, Some(host_pool(5_000_000)));
        let reader = geometry(2_000_000, Some(host_pool(6_000_000)));
        // Reader block 4 is the first block of the second parent.
        let spans = read_spans(
            &holder,
            &reader,
            &[
                (page(0, 0, DEVICE, 7), page(0, 0, HOST, 4)),
                (page(0, 1, HOST, 5), page(0, 1, HOST, 2)),
            ],
        )
        .unwrap();
        assert_eq!(
            spans.sources,
            vec![1_000_700, 1_010_700, 5_000_400, 5_000_440]
        );
        assert_eq!(
            spans.destinations,
            vec![6_000_300, 6_000_340, 6_000_100, 6_000_140]
        );
        assert_eq!(spans.lengths, vec![40, 60, 40, 60]);
        assert_eq!(spans.host_bytes, 100);
    }

    #[test]
    fn what_cannot_be_placed_is_refused() {
        let holder = geometry(1_000_000, None);
        let reader = geometry(2_000_000, None);
        assert_eq!(
            read_spans(
                &holder,
                &reader,
                &[(page(0, 0, HOST, 5), page(0, 0, DEVICE, 4))]
            ),
            Err(GeometryError::NoHost)
        );
        assert_eq!(
            read_spans(
                &holder,
                &reader,
                &[(page(0, 0, DEVICE, 5), page(0, 0, HOST, 4))]
            ),
            Err(GeometryError::NoHost)
        );
        assert_eq!(
            read_spans(
                &holder,
                &reader,
                &[(page(0, 0, PeerTier::Disk, 5), page(0, 0, DEVICE, 4))]
            ),
            Err(GeometryError::Tier),
            "a Disk page is pushed, never read"
        );
        assert!(matches!(
            read_spans(
                &holder,
                &reader,
                &[(page(0, 0, DEVICE, 0), page(0, 0, DEVICE, 4))]
            ),
            Err(GeometryError::Block { .. })
        ));
        let mut narrower = reader.clone();
        narrower.device[1].bytes = 59;
        assert_eq!(
            read_spans(
                &holder,
                &narrower,
                &[(page(0, 0, DEVICE, 5), page(0, 0, DEVICE, 4))]
            ),
            Err(GeometryError::Field("layer.0.v".into()))
        );
        let far = RankGeometry {
            device: vec![DeviceField {
                page_stride: u64::MAX,
                ..holder.device[0].clone()
            }],
            ..holder.clone()
        };
        assert_eq!(
            read_spans(
                &far,
                &reader,
                &[(page(0, 0, DEVICE, 2), page(0, 0, DEVICE, 4))]
            ),
            Err(GeometryError::Overflow)
        );
    }

    #[test]
    fn multi_mr_reads_and_pushes_split_cleanly_by_src_and_dst_mr() {
        let mut hh = host_pool(10_000_000);
        hh.region_lengths = vec![600; 60];
        let mut rh = host_pool(20_000_000);
        rh.region_lengths = vec![1_200; 30];
        let (holder, reader) = (geometry(1_000_000, Some(hh)), geometry(2_000_000, Some(rh)));
        let pages = vec![
            (page(0, 0, HOST, 1), page(0, 0, HOST, 1)),
            (page(0, 1, HOST, 4), page(0, 1, HOST, 4)),
            (page(0, 2, HOST, 7), page(0, 2, HOST, 7)),
            (page(0, 3, HOST, 358), page(0, 3, HOST, 358)),
            (page(0, 4, DEVICE, 7), page(0, 4, HOST, 13)),
            (page(0, 5, HOST, 8), page(0, 5, DEVICE, 3)),
        ];
        let spans = read_spans(&holder, &reader, &pages).unwrap();
        let (total, batches) = read_batches(&holder, &reader, &pages).unwrap();
        assert_eq!((total, batches.len()), (spans.total_bytes(), 5));
        assert_eq!(
            batches.iter().map(|b| b.host_bytes).sum::<u64>(),
            spans.host_bytes
        );
        let (sr, dr) = (
            TierRegions::from_geometry(&holder).unwrap(),
            TierRegions::from_geometry(&reader).unwrap(),
        );
        let mut got = Vec::new();
        for b in &batches {
            let pair = (
                sr.key_for_address(&holder, b.sources[0], b.lengths[0])
                    .unwrap(),
                dr.key_for_address(&reader, b.destinations[0], b.lengths[0])
                    .unwrap(),
            );
            for ((&s, &d), &l) in b.sources.iter().zip(&b.destinations).zip(&b.lengths) {
                assert_eq!(
                    (
                        sr.key_for_address(&holder, s, l).unwrap(),
                        dr.key_for_address(&reader, d, l).unwrap()
                    ),
                    pair
                );
                got.push((s, d, l));
            }
        }
        let mut want: Vec<_> = spans
            .sources
            .iter()
            .copied()
            .zip(spans.destinations.iter().copied())
            .zip(spans.lengths.iter().copied())
            .map(|((s, d), l)| (s, d, l))
            .collect();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, want);

        // Cross-MR host landing splits PushItems by destination MR.
        let mut split_h = host_pool(20_000_000);
        split_h.region_lengths = vec![40, 260];
        let items = push_items(
            &holder,
            &geometry(2_000_000, Some(split_h)),
            &[(page(0, 0, PeerTier::Disk, 1), page(0, 0, HOST, 1))],
        )
        .unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(
            (items[0].spans[0].address, items[1].spans[0].address),
            (20_000_000, 20_000_040)
        );
    }

    #[test]
    fn region_index_rejects_uncovered_and_straddling_host_and_device() {
        let mut hh = host_pool(10_000_000);
        hh.region_lengths = vec![300];
        let holder = geometry(1_000_000, Some(hh));
        assert!(matches!(
            read_batches(
                &holder,
                &geometry(2_000_000, Some(host_pool(20_000_000))),
                &[(page(0, 0, HOST, 4), page(0, 0, DEVICE, 1))]
            ),
            Err(GeometryError::Uncovered {
                tier: PeerTier::Host,
                addr: 10_000_300,
                len: 40
            })
        ));

        let mut straddle_h = host_pool(20_000_000);
        straddle_h.region_lengths = vec![30, 270];
        let straddle_r = geometry(2_000_000, Some(straddle_h));
        assert!(matches!(
            push_items(
                &holder,
                &straddle_r,
                &[(page(0, 0, PeerTier::Disk, 1), page(0, 0, HOST, 1))]
            ),
            Err(GeometryError::Straddle {
                tier: PeerTier::Host,
                addr: 20_000_000,
                len: 40,
                region_end: 20_000_030
            })
        ));

        let mut dev_r = geometry(2_000_000, None);
        dev_r.device_regions = vec![(2_000_000, 1_000), (2_010_000, 1_000)];
        assert_eq!(
            read_batches(
                &geometry(1_000_000, None),
                &dev_r,
                &[(page(0, 0, DEVICE, 2), page(0, 0, DEVICE, 3))]
            )
            .unwrap()
            .1
            .len(),
            2
        );
        assert_eq!(
            push_items(
                &holder,
                &dev_r,
                &[(page(0, 0, PeerTier::Disk, 1), page(0, 0, DEVICE, 3))]
            )
            .unwrap()
            .len(),
            2
        );
        assert!(matches!(
            read_batches(
                &geometry(1_000_000, None),
                &dev_r,
                &[(page(0, 0, DEVICE, 2), page(0, 0, DEVICE, 20))]
            ),
            Err(GeometryError::Uncovered {
                tier: PeerTier::Device,
                ..
            })
        ));
    }
}
