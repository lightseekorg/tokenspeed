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

//! Where a scheduler Host block lives in the shared pinned Host allocation,
//! and whether this rank owns it.
//!
//! The data plane allocates the Host pool (`cache/host_storage.py`) and the
//! L2 tier copies through it; the same bytes are what an L3 GET fills and a
//! PUT reads. One LCM block holds `packing` cache blocks of each group, laid
//! out field after field. A group with `shard_count` D deals its blocks
//! cyclically to D owners: block `v > 0` belongs to rank `(v - 1) % D` and is
//! that rank's local block `(v - 1) / D + 1`
//! (`cache/transfer/ownership.py`).

/// One cache group's blocks in the Host pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupLayout {
    /// Cache blocks per LCM block.
    pub packing: usize,
    /// Bytes of one packed cache block.
    pub block_bytes: usize,
    /// Owners the group's blocks are dealt to; 1 is replicated.
    pub shard_count: usize,
}

/// The Host pool's geometry and this rank's place among the owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostLayout {
    /// Each group, in the scheduler's group order.
    pub groups: Vec<GroupLayout>,
    /// Bytes of one LCM block.
    pub lcm_block_bytes: usize,
    /// LCM blocks in the pool, the null block excluded.
    pub num_lcm_blocks: usize,
    /// This rank among a sharded group's owners (the KV-parallel rank).
    pub owner_rank: usize,
    /// This rank's attention-TP rank.
    pub tp_rank: usize,
}

/// A block or group the layout cannot place.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LayoutError {
    /// No such group.
    #[error("cache transfer names unknown group {0}")]
    Group(u32),
    /// The null block.
    #[error("a cache transfer cannot name the null block 0")]
    Null,
    /// Past the pool.
    #[error("Host block {block} of group {group} is out of range")]
    Block {
        /// The group.
        group: u32,
        /// The block.
        block: i64,
    },
}

/// A byte range of the Host pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    /// From the pool's start.
    pub offset: usize,
    /// Its length.
    pub size: usize,
}

impl HostLayout {
    /// The pool's size.
    pub fn bytes(&self) -> usize {
        self.num_lcm_blocks * self.lcm_block_bytes
    }

    fn group(&self, group: u32) -> Result<&GroupLayout, LayoutError> {
        self.groups
            .get(group as usize)
            .ok_or(LayoutError::Group(group))
    }

    /// The bytes of `group`'s scheduler block `block` on this rank, or None
    /// when another rank owns it.
    pub fn owned_range(&self, group: u32, block: i32) -> Result<Option<Range>, LayoutError> {
        let layout = self.group(group)?;
        if block == 0 {
            return Err(LayoutError::Null);
        }
        let virtual_count = 1 + self.num_lcm_blocks * layout.packing * layout.shard_count;
        let Ok(virtual_block) = usize::try_from(block) else {
            return Err(LayoutError::Block {
                group,
                block: i64::from(block),
            });
        };
        if virtual_block >= virtual_count {
            return Err(LayoutError::Block {
                group,
                block: i64::from(block),
            });
        }
        let dealt = virtual_block - 1;
        if layout.shard_count > 1 && dealt % layout.shard_count != self.owner_rank {
            return Ok(None);
        }
        let local = dealt / layout.shard_count + 1;
        Ok(Some(self.local_range(group, layout, local)?))
    }

    /// The bytes of this rank's local block `local` of `group`
    /// (`HostCacheStorage.host_block_range`).
    fn local_range(
        &self,
        group: u32,
        layout: &GroupLayout,
        local: usize,
    ) -> Result<Range, LayoutError> {
        if local == 0 || local > self.num_lcm_blocks * layout.packing {
            return Err(LayoutError::Block {
                group,
                block: local as i64,
            });
        }
        let zero_based = local - 1;
        let (parent, child) = (zero_based / layout.packing, zero_based % layout.packing);
        Ok(Range {
            offset: parent * self.lcm_block_bytes + child * layout.block_bytes,
            size: layout.block_bytes,
        })
    }

    /// The bytes of one of `group`'s blocks, owned here or not.
    pub fn block_bytes(&self, group: u32) -> Result<usize, LayoutError> {
        Ok(self.group(group)?.block_bytes)
    }

    /// The rank an object key of `group` names: the owners of a sharded
    /// group share one name per shard, so a later allocation may move a
    /// block to another owner of it.
    pub fn key_rank(&self, group: u32) -> Result<i64, LayoutError> {
        Ok((self.tp_rank / self.group(group)?.shard_count) as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shard_count: usize, owner_rank: usize) -> HostLayout {
        HostLayout {
            groups: vec![
                GroupLayout {
                    packing: 2,
                    block_bytes: 10,
                    shard_count: 1,
                },
                GroupLayout {
                    packing: 1,
                    block_bytes: 4,
                    shard_count,
                },
            ],
            lcm_block_bytes: 20,
            num_lcm_blocks: 3,
            owner_rank,
            tp_rank: 5,
        }
    }

    #[test]
    fn packed_blocks_sit_inside_their_lcm_block() {
        let host = layout(1, 0);
        assert_eq!(host.bytes(), 60);
        let range = |block| host.owned_range(0, block).unwrap().unwrap();
        assert_eq!(
            range(1),
            Range {
                offset: 0,
                size: 10
            }
        );
        assert_eq!(
            range(2),
            Range {
                offset: 10,
                size: 10
            }
        );
        assert_eq!(
            range(3),
            Range {
                offset: 20,
                size: 10
            }
        );
        assert_eq!(
            range(6),
            Range {
                offset: 50,
                size: 10
            }
        );
        assert_eq!(
            host.owned_range(0, 7),
            Err(LayoutError::Block { group: 0, block: 7 })
        );
        assert_eq!(host.owned_range(0, 0), Err(LayoutError::Null));
        assert_eq!(host.owned_range(2, 1), Err(LayoutError::Group(2)));
    }

    #[test]
    fn a_sharded_group_deals_its_blocks_to_the_owners() {
        // Blocks 1, 3, 5 belong to rank 0 as its local 1, 2, 3; 2, 4, 6 to
        // rank 1.
        let rank0 = layout(2, 0);
        let rank1 = layout(2, 1);
        assert_eq!(
            rank0.owned_range(1, 3).unwrap(),
            Some(Range {
                offset: 20,
                size: 4
            })
        );
        assert_eq!(rank0.owned_range(1, 4).unwrap(), None);
        assert_eq!(
            rank1.owned_range(1, 4).unwrap(),
            Some(Range {
                offset: 20,
                size: 4
            })
        );
        // The bound counts every owner's blocks.
        assert!(rank1.owned_range(1, 6).unwrap().is_some());
        assert!(rank1.owned_range(1, 7).is_err());
        assert_eq!(rank0.key_rank(1).unwrap(), 2);
        assert_eq!(rank0.key_rank(0).unwrap(), 5);
    }
}
