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

//! Map prepared storage objects directly into ordinarily admitted Device pages.
//!
//! Source addresses are per lease, not a fictitious Host pool. The packed
//! object layout comes from the reader's cache namespace. This function only
//! describes reads; the caller owns the source lease and destination blocks.

use std::collections::{BTreeMap, HashMap, HashSet};

use ts_scheduler::PeerPageView;

use crate::peer::geometry::{GeometryError, RankGeometry, Spans};

/// One complete object staged at its storage owner, under a live source lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedObject {
    /// The cache group and absolute logical slot in the request's table.
    pub group_id: u32,
    /// State checkpoints use the ordinary peer State slot.
    pub slot: i32,
    /// Transfer session registering the storage owner's buffer.
    pub session: String,
    /// First byte of the packed object, not a cache-pool base address.
    pub address: u64,
    /// Exact length returned by preparation.
    pub bytes: u64,
}

/// A source description cannot satisfy the admitted destinations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PreparedError {
    /// The same logical dependency appears more than once.
    #[error("duplicate prepared object or destination for group {0}, slot {1}")]
    Duplicate(u32, i32),
    /// A required object is absent.
    #[error("missing prepared object for group {0}, slot {1}")]
    Missing(u32, i32),
    /// The object does not match the namespace's packed layout.
    #[error("invalid prepared object or layout for group {0}")]
    Layout(u32),
    /// Prepared objects may only land in valid Device pages.
    #[error("a prepared storage read requires a valid Device destination")]
    Destination,
    /// A destination or byte address is invalid.
    #[error(transparent)]
    Geometry(#[from] GeometryError),
}

/// Every admitted destination's fields, grouped by actual source session.
///
/// A prepared prefix may include objects the reader already has; only the
/// destinations granted by ordinary admission are read. No L2 block is
/// allocated and no source address is derived from an allocation id.
pub fn read_spans(
    objects: &[PreparedObject],
    reader: &RankGeometry,
    destinations: &[PeerPageView],
) -> Result<BTreeMap<String, Spans>, PreparedError> {
    let mut sources = HashMap::with_capacity(objects.len());
    for object in objects {
        if sources
            .insert((object.group_id, object.slot), object)
            .is_some()
        {
            return Err(PreparedError::Duplicate(object.group_id, object.slot));
        }
    }
    let pool = reader.host.as_ref().ok_or(GeometryError::NoHost)?;
    let mut seen = HashSet::new();
    let mut reads = BTreeMap::<String, Spans>::new();
    for destination in destinations {
        let group_id = destination.group_id;
        if destination.tier != ts_scheduler::PeerTier::Device
            || destination.block_id <= 0
            || destination.slot < 0
        {
            return Err(PreparedError::Destination);
        }
        if !seen.insert((group_id, destination.slot)) {
            return Err(PreparedError::Duplicate(group_id, destination.slot));
        }
        let source = sources
            .get(&(group_id, destination.slot))
            .ok_or(PreparedError::Missing(group_id, destination.slot))?;
        let group = pool
            .groups
            .get(group_id as usize)
            .ok_or(PreparedError::Layout(group_id))?;
        if source.session.is_empty()
            || source.address == 0
            || source.bytes == 0
            || source.bytes != group.block_bytes
        {
            return Err(PreparedError::Layout(group_id));
        }
        source
            .address
            .checked_add(source.bytes)
            .ok_or(GeometryError::Overflow)?;
        let mut offsets = HashMap::new();
        for (field, offset) in &group.field_offsets {
            if offsets.insert(field.as_str(), *offset).is_some() {
                return Err(PreparedError::Layout(group_id));
            }
        }
        let spans = reads.entry(source.session.clone()).or_default();
        let mut fields = HashSet::new();
        for field in reader
            .device
            .iter()
            .filter(|field| field.group_id == group_id)
        {
            if !fields.insert(field.field_id.as_str())
                || field.bytes == 0
                || field.page_stride < field.bytes
            {
                return Err(PreparedError::Layout(group_id));
            }
            let offset = *offsets
                .get(field.field_id.as_str())
                .ok_or(PreparedError::Layout(group_id))?;
            if offset
                .checked_add(field.bytes)
                .is_none_or(|end| end > source.bytes)
            {
                return Err(PreparedError::Layout(group_id));
            }
            let to = (destination.block_id as u64)
                .checked_mul(field.page_stride)
                .and_then(|offset| field.address.checked_add(offset))
                .ok_or(GeometryError::Overflow)?;
            to.checked_add(field.bytes).ok_or(GeometryError::Overflow)?;
            spans.sources.push(
                source
                    .address
                    .checked_add(offset)
                    .ok_or(GeometryError::Overflow)?,
            );
            spans.destinations.push(to);
            spans.lengths.push(field.bytes);
            spans.host_bytes = spans
                .host_bytes
                .checked_add(field.bytes)
                .ok_or(GeometryError::Overflow)?;
        }
        if fields.is_empty() || fields.len() != offsets.len() {
            return Err(PreparedError::Layout(group_id));
        }
    }
    Ok(reads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer::{
        geometry::{DeviceField, HostGroup, HostPool},
        read::{Arena, PageReader},
    };

    fn reader() -> RankGeometry {
        RankGeometry {
            session: "reader".into(),
            push: None,
            device: vec![
                DeviceField {
                    group_id: 0,
                    field_id: "k".into(),
                    address: 4096,
                    page_stride: 64,
                    bytes: 16,
                },
                DeviceField {
                    group_id: 0,
                    field_id: "v".into(),
                    address: 8192,
                    page_stride: 64,
                    bytes: 32,
                },
            ],
            // Only the packed schema is used. Host address and allocation ids
            // play no part in an L3-to-Device read.
            host: Some(HostPool {
                address: 0,
                groups: vec![HostGroup {
                    parent_bytes: 512,
                    packing: 8,
                    block_bytes: 64,
                    field_offsets: vec![("k".into(), 0), ("v".into(), 32)],
                }],
            }),
        }
    }

    fn object(slot: i32) -> PreparedObject {
        PreparedObject {
            group_id: 0,
            slot,
            session: format!("ssd-{slot}"),
            address: 256 + slot as u64 * 128,
            bytes: 64,
        }
    }

    fn page(slot: i32, block_id: i32) -> PeerPageView {
        PeerPageView {
            group_id: 0,
            slot,
            tier: ts_scheduler::PeerTier::Device,
            block_id,
        }
    }

    #[test]
    fn packed_objects_scatter_into_fields_without_touching_padding_or_other_pages() {
        let arena = Arena::new(16384);
        arena.write(0, &vec![173; 16384]);
        let objects = vec![object(0), object(4), object(5)];
        for object in &objects {
            arena.write(object.address, &vec![object.slot as u8; 64]);
        }
        // Slot zero is already local. Window slots keep their absolute indices.
        let reads = read_spans(&objects, &reader(), &[page(4, 2), page(5, 7)]).unwrap();
        assert_eq!(reads.len(), 2);
        for (session, spans) in &reads {
            arena.read(session, spans).unwrap();
        }
        for (slot, block) in [(4, 2), (5, 7)] {
            assert_eq!(arena.read_bytes(4096 + block * 64, 16), vec![slot; 16]);
            assert_eq!(arena.read_bytes(4096 + block * 64 + 16, 48), vec![173; 48]);
            assert_eq!(arena.read_bytes(8192 + block * 64, 32), vec![slot; 32]);
            assert_eq!(arena.read_bytes(8192 + block * 64 + 32, 32), vec![173; 32]);
        }
        assert_eq!(arena.read_bytes(4096 + 64, 64), vec![173; 64]);
        assert_eq!(reads.values().map(Spans::total_bytes).sum::<u64>(), 96);
    }

    #[test]
    fn incomplete_or_ambiguous_dependencies_are_rejected_before_any_read() {
        assert_eq!(
            read_spans(&[object(0)], &reader(), &[page(1, 2)]),
            Err(PreparedError::Missing(0, 1))
        );
        assert_eq!(
            read_spans(&[object(0), object(0)], &reader(), &[page(0, 2)]),
            Err(PreparedError::Duplicate(0, 0))
        );
        assert_eq!(
            read_spans(&[object(0)], &reader(), &[page(0, 2), page(0, 3)]),
            Err(PreparedError::Duplicate(0, 0))
        );
        let mut short = object(0);
        short.bytes -= 1;
        assert_eq!(
            read_spans(&[short], &reader(), &[page(0, 2)]),
            Err(PreparedError::Layout(0))
        );
        let mut missing_field = reader();
        missing_field.device.pop();
        assert_eq!(
            read_spans(&[object(0)], &missing_field, &[page(0, 2)]),
            Err(PreparedError::Layout(0))
        );
        let mut absent_group = reader();
        absent_group.device.clear();
        assert_eq!(
            read_spans(&[object(0)], &absent_group, &[page(0, 2)]),
            Err(PreparedError::Layout(0))
        );
    }

    #[test]
    fn overflow_and_field_extents_are_checked() {
        let mut far = object(0);
        far.address = u64::MAX - 32;
        assert_eq!(
            read_spans(&[far], &reader(), &[page(0, 2)]),
            Err(PreparedError::Geometry(GeometryError::Overflow))
        );
        let mut bad_field = reader();
        bad_field.device[1].bytes = 33;
        assert_eq!(
            read_spans(&[object(0)], &bad_field, &[page(0, 2)]),
            Err(PreparedError::Layout(0))
        );
        let mut far_dest = reader();
        far_dest.device[0].address = u64::MAX - 64;
        assert_eq!(
            read_spans(&[object(0)], &far_dest, &[page(0, 2)]),
            Err(PreparedError::Geometry(GeometryError::Overflow))
        );
        let mut wide = reader();
        wide.device[0].page_stride = 1 << 32;
        let spans = read_spans(&[object(0)], &wide, &[page(0, 2)]).unwrap();
        assert_eq!(spans["ssd-0"].destinations[0], 4096 + (2u64 << 32));
    }
}
