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

//! Where a rank's Device cache blocks and a shared L2 pool's pages are, and
//! the copies a load and a write make between them.
//!
//! A pool page holds one scheduler CacheBlock as a Host L2 block does, with
//! the sizes `cache/shared_l2/layout.py` reads off the Host layout: the pool
//! is a row of LCM blocks, a group's pages lie back to back in one, and a
//! page holds the group's fields back to back. Only the LCM blocks are
//! spread over the providers.

use crate::{peer::geometry::Spans, shared_l2::PoolError};

/// One node's share of the pool: `num_lcm_blocks` LCM blocks at `base_address` of its
/// transfer session, the pool's LCM blocks from `first_lcm_block` on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provider {
    /// The provider's transfer session.
    pub session: String,
    /// The address of its first LCM block.
    pub base_address: u64,
    /// How many LCM blocks it holds.
    pub num_lcm_blocks: u64,
    /// The pool's LCM block its first one is.
    pub first_lcm_block: u64,
}

/// One field of a group's Device CacheBlocks, as a pool page holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageField {
    /// The address of the field in Device block zero.
    pub device_address: u64,
    /// Bytes from one Device block to the next.
    pub device_stride: u64,
    /// Bytes of the field's payload.
    pub bytes: u64,
    /// The payload's offset inside a pool page.
    pub offset_in_page: u64,
    /// The layer that reads the field, among the rank's layers in the order
    /// its ready words follow.
    pub layer: usize,
}

/// A cache group's pool pages: `packing` pages of `page_bytes` in an LCM
/// block, each holding `fields` in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageGroup {
    /// Pool pages per LCM block.
    pub packing: u64,
    /// Bytes of one pool page.
    pub page_bytes: u64,
    /// The group's fields, in the order a pool page holds them.
    pub fields: Vec<PageField>,
}

/// A pool page and the Device block copied to it or from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCopy {
    /// The cache group, by index.
    pub group_id: u32,
    /// The page's shard.
    pub bucket: i32,
    /// The pool page, within the group; pages start at 1.
    pub pool_page: i32,
    /// The Device block.
    pub device_block: i32,
}

/// The pool a rank joined and the rank's Device cache. A copy divides by a
/// group's packing and indexes a field's layer without a check: the client
/// checks both once, when it starts from the rank's settings
/// (`client::start`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolGeometry {
    /// Bytes of one pool LCM block.
    pub lcm_block_bytes: u64,
    /// The providers, in the order of their LCM blocks.
    pub providers: Vec<Provider>,
    /// The cache groups, in the scheduler's order.
    pub groups: Vec<PageGroup>,
    /// The rank's layers: every field's `layer` is below it.
    pub num_layers: usize,
}

fn overflow() -> PoolError {
    PoolError("a shared L2 page address overflows".to_owned())
}

impl PoolGeometry {
    fn group(&self, group: u32) -> Result<&PageGroup, PoolError> {
        self.groups
            .get(group as usize)
            .ok_or_else(|| PoolError(format!("the shared L2 pool has no cache group {group}")))
    }

    /// The provider that holds a group's pool page, by index, and the page's
    /// address there. A page lies in its LCM block as
    /// `HostCacheStorage.host_block_offset` places a Host block.
    pub fn page_address(
        &self,
        group: u32,
        bucket: i32,
        page: i32,
    ) -> Result<(usize, u64), PoolError> {
        if bucket != 0 {
            return Err(PoolError("a shared L2 pool holds bucket 0 only".to_owned()));
        }
        let page_group = self.group(group)?;
        let out_of_range = || {
            PoolError(format!(
                "shared L2 page {page} of group {group} is out of range"
            ))
        };
        // Pages start at 1; widened before any product: page numbers reach
        // the pool's size.
        let index = u64::try_from(i64::from(page) - 1).map_err(|_| out_of_range())?;
        let (lcm_block, slot) = (index / page_group.packing, index % page_group.packing);
        // The providers' LCM blocks follow one another: the last provider
        // that starts at or before the block holds it, if any does.
        let starting_at_or_before = self
            .providers
            .partition_point(|provider| provider.first_lcm_block <= lcm_block);
        let (provider_index, provider) = starting_at_or_before
            .checked_sub(1)
            .map(|index| (index, &self.providers[index]))
            .filter(|(_, provider)| lcm_block - provider.first_lcm_block < provider.num_lcm_blocks)
            .ok_or_else(out_of_range)?;
        let page_base = (lcm_block - provider.first_lcm_block)
            .checked_mul(self.lcm_block_bytes)
            .and_then(|offset| offset.checked_add(slot.checked_mul(page_group.page_bytes)?))
            .and_then(|offset| provider.base_address.checked_add(offset))
            .ok_or_else(overflow)?;
        Ok((provider_index, page_base))
    }

    /// The two addresses of one field of a copy: in the pool page at
    /// `page_base` (the page's address at its provider), and in Device
    /// `block`.
    fn field_addresses(
        field: &PageField,
        page_base: u64,
        block: i32,
    ) -> Result<(u64, u64), PoolError> {
        let in_device_block = u64::try_from(block)
            .ok()
            .and_then(|block| block.checked_mul(field.device_stride))
            .and_then(|offset| field.device_address.checked_add(offset))
            .ok_or_else(overflow)?;
        let in_pool_page = page_base
            .checked_add(field.offset_in_page)
            .ok_or_else(overflow)?;
        Ok((in_pool_page, in_device_block))
    }

    /// The reads of a load, per layer and then per provider (by index): each
    /// copies a field of a pool page to its Device block.
    pub fn load_spans(&self, copies: &[PageCopy]) -> Result<Vec<Vec<Spans>>, PoolError> {
        let mut layers = vec![vec![Spans::default(); self.providers.len()]; self.num_layers];
        for copy in copies {
            let (provider_index, page_base) =
                self.page_address(copy.group_id, copy.bucket, copy.pool_page)?;
            for field in &self.group(copy.group_id)?.fields {
                let (in_pool_page, in_device_block) =
                    Self::field_addresses(field, page_base, copy.device_block)?;
                let spans = &mut layers[field.layer][provider_index];
                spans.sources.push(in_pool_page);
                spans.destinations.push(in_device_block);
                spans.lengths.push(field.bytes);
            }
        }
        Ok(layers)
    }

    /// The writes of Device blocks to their pool pages, per provider (by
    /// index).
    pub fn write_spans(&self, copies: &[PageCopy]) -> Result<Vec<Spans>, PoolError> {
        let mut spans_by_provider = vec![Spans::default(); self.providers.len()];
        for copy in copies {
            let (provider_index, page_base) =
                self.page_address(copy.group_id, copy.bucket, copy.pool_page)?;
            for field in &self.group(copy.group_id)?.fields {
                let (in_pool_page, in_device_block) =
                    Self::field_addresses(field, page_base, copy.device_block)?;
                let spans = &mut spans_by_provider[provider_index];
                spans.sources.push(in_device_block);
                spans.destinations.push(in_pool_page);
                spans.lengths.push(field.bytes);
            }
        }
        Ok(spans_by_provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KV: u32 = 0;
    const STATE: u32 = 1;

    /// The pool of the Python layout tests (`test_shared_l2_data_path.py`):
    /// LCM blocks of 28 bytes, 3 on `p0` and 5 on `p1`; a KV page fills an
    /// LCM block with k (12 bytes, layer 0) and v (16 bytes, layer 1); two
    /// State pages of 12 bytes (layer 1) fit in one.
    fn geometry() -> PoolGeometry {
        PoolGeometry {
            lcm_block_bytes: 28,
            providers: vec![
                Provider {
                    session: "p0".to_owned(),
                    base_address: 1_000,
                    num_lcm_blocks: 3,
                    first_lcm_block: 0,
                },
                Provider {
                    session: "p1".to_owned(),
                    base_address: 5_000,
                    num_lcm_blocks: 5,
                    first_lcm_block: 3,
                },
            ],
            groups: vec![
                PageGroup {
                    packing: 1,
                    page_bytes: 28,
                    fields: vec![
                        PageField {
                            device_address: 100,
                            device_stride: 32,
                            bytes: 12,
                            offset_in_page: 0,
                            layer: 0,
                        },
                        PageField {
                            device_address: 116,
                            device_stride: 32,
                            bytes: 16,
                            offset_in_page: 12,
                            layer: 1,
                        },
                    ],
                },
                PageGroup {
                    packing: 2,
                    page_bytes: 12,
                    fields: vec![PageField {
                        device_address: 500,
                        device_stride: 16,
                        bytes: 12,
                        offset_in_page: 0,
                        layer: 1,
                    }],
                },
            ],
            num_layers: 2,
        }
    }

    fn copy(group_id: u32, pool_page: i32, device_block: i32) -> PageCopy {
        PageCopy {
            group_id,
            bucket: 0,
            pool_page,
            device_block,
        }
    }

    #[test]
    fn a_page_lies_where_the_host_tier_puts_the_block_of_that_id() {
        let geometry = geometry();
        // A KV page per LCM block: pages 1 to 3 on p0, 4 to 8 on p1.
        assert_eq!(geometry.page_address(KV, 0, 1).unwrap(), (0, 1_000));
        assert_eq!(geometry.page_address(KV, 0, 3).unwrap(), (0, 1_056));
        assert_eq!(geometry.page_address(KV, 0, 4).unwrap(), (1, 5_000));
        assert_eq!(geometry.page_address(KV, 0, 8).unwrap(), (1, 5_112));
        // Two State pages of an LCM block lie back to back.
        assert_eq!(geometry.page_address(STATE, 0, 6).unwrap(), (0, 1_068));
        assert_eq!(geometry.page_address(STATE, 0, 7).unwrap(), (1, 5_000));
        assert_eq!(geometry.page_address(STATE, 0, 8).unwrap(), (1, 5_012));
        assert_eq!(geometry.page_address(STATE, 0, 16).unwrap(), (1, 5_124));
    }

    #[test]
    fn a_page_the_pool_does_not_have_is_refused() {
        let geometry = geometry();
        for (group, page) in [(STATE, 0), (STATE, 17), (STATE, -3), (KV, 9)] {
            let error = geometry.page_address(group, 0, page).unwrap_err();
            assert!(error.0.contains("out of range"), "{error}");
        }
        let error = geometry.page_address(2, 0, 1).unwrap_err();
        assert!(error.0.contains("no cache group 2"), "{error}");
        let error = geometry.page_address(KV, 1, 1).unwrap_err();
        assert!(error.0.contains("bucket 0 only"), "{error}");
    }

    #[test]
    fn a_load_reads_each_layers_fields_from_the_provider_that_holds_them() {
        let geometry = geometry();
        let layers = geometry
            .load_spans(&[copy(KV, 1, 3), copy(KV, 8, 1), copy(STATE, 4, 8)])
            .unwrap();
        // Layer 0 reads k alone; layer 1 reads v and the state.
        assert_eq!(
            layers[0],
            vec![
                Spans {
                    sources: vec![1_000],
                    destinations: vec![100 + 3 * 32],
                    lengths: vec![12],
                    host_bytes: 0,
                },
                Spans {
                    sources: vec![5_112],
                    destinations: vec![100 + 32],
                    lengths: vec![12],
                    host_bytes: 0,
                },
            ]
        );
        assert_eq!(
            layers[1],
            vec![
                Spans {
                    // KV page 1's v, and State page 4: the second page of
                    // the pool's second LCM block.
                    sources: vec![1_012, 1_000 + 28 + 12],
                    destinations: vec![116 + 3 * 32, 500 + 8 * 16],
                    lengths: vec![16, 12],
                    host_bytes: 0,
                },
                Spans {
                    sources: vec![5_124],
                    destinations: vec![116 + 32],
                    lengths: vec![16],
                    host_bytes: 0,
                },
            ]
        );
    }

    #[test]
    fn a_write_copies_every_field_of_a_block_to_its_page() {
        let geometry = geometry();
        let providers = geometry
            .write_spans(&[copy(KV, 3, 5), copy(STATE, 9, 2)])
            .unwrap();
        assert_eq!(
            providers[0],
            Spans {
                sources: vec![100 + 5 * 32, 116 + 5 * 32],
                destinations: vec![1_056, 1_068],
                lengths: vec![12, 16],
                host_bytes: 0,
            }
        );
        // State page 9 opens the pool's fifth LCM block, p1's second.
        assert_eq!(
            providers[1],
            Spans {
                sources: vec![500 + 2 * 16],
                destinations: vec![5_028],
                lengths: vec![12],
                host_bytes: 0,
            }
        );
    }

    #[test]
    fn addresses_of_a_pool_of_a_hundred_million_pages_do_not_wrap() {
        // 10^8 LCM blocks of 2 MiB on one provider: byte offsets pass 2^47,
        // far past what 32-bit page arithmetic holds.
        let geometry = PoolGeometry {
            lcm_block_bytes: 2 << 20,
            providers: vec![Provider {
                session: "p0".to_owned(),
                base_address: 1 << 40,
                num_lcm_blocks: 100_000_000,
                first_lcm_block: 0,
            }],
            groups: vec![PageGroup {
                packing: 1,
                page_bytes: 2 << 20,
                fields: vec![PageField {
                    device_address: 1 << 46,
                    device_stride: 2 << 20,
                    bytes: 2 << 20,
                    offset_in_page: 0,
                    layer: 0,
                }],
            }],
            num_layers: 1,
        };
        let last = copy(0, 100_000_000, i32::MAX);
        assert_eq!(
            geometry.page_address(0, 0, 100_000_000).unwrap(),
            (0, (1 << 40) + 99_999_999 * (2 << 20))
        );
        let spans = &geometry.load_spans(&[last]).unwrap()[0][0];
        assert_eq!(spans.sources, [(1 << 40) + 99_999_999 * (2 << 20)]);
        assert_eq!(
            spans.destinations,
            [(1 << 46) + u64::from(i32::MAX as u32) * (2 << 20)]
        );
        assert!(geometry.page_address(0, 0, 100_000_001).is_err());

        // An address past 64 bits is an error, not a wrapped address.
        let mut far = geometry;
        far.providers[0].base_address = u64::MAX - 1;
        let error = far.page_address(0, 0, 2).unwrap_err();
        assert!(error.0.contains("overflows"), "{error}");
        far.providers[0].base_address = 0;
        far.groups[0].fields[0].device_address = u64::MAX - 1;
        let error = far.load_spans(&[copy(0, 1, 1)]).unwrap_err();
        assert!(error.0.contains("overflows"), "{error}");
        let error = far.write_spans(&[copy(0, 1, -1)]).unwrap_err();
        assert!(error.0.contains("overflows"), "{error}");
    }
}
