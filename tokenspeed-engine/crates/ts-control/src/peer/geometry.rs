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
}

/// What a peer needs to read a rank's cache, or to land into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankGeometry {
    /// The transfer engine session the rank's memory is registered with.
    pub session: String,
    /// Every Device field, in the order the rank lists them.
    pub device: Vec<DeviceField>,
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
            .and_then(|parent| {
                (index % host.packing)
                    .checked_mul(host.block_bytes)?
                    .checked_add(parent)
            })
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
    pub(super) fn field_address(
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

/// The pushes that land `pages` whose holder pages are on its Disk tier:
/// per page, every field of the page's group, at its offset in the holder's
/// packed block (a Disk block packs like a Host block), to the reader's tier.
pub fn push_items(
    holder: &RankGeometry,
    reader: &RankGeometry,
    pages: &[(PeerPageView, PeerPageView)],
) -> Result<Vec<PushItem>, GeometryError> {
    let pool = holder.host.as_ref().ok_or(GeometryError::NoHost)?;
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
        let mut spans = Vec::new();
        for field in holder.device.iter().filter(|field| field.group_id == group) {
            let offset = host
                .field_offsets
                .iter()
                .find(|(id, _)| *id == field.field_id)
                .map(|(_, offset)| *offset)
                .ok_or_else(|| GeometryError::Field(field.field_id.clone()))?;
            spans.push(PushSpan {
                offset,
                address: reader.field_address(destination, field)?,
                length: field.bytes,
            });
        }
        items.push(PushItem {
            group,
            block: source.block_id,
            spans,
        });
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
}
