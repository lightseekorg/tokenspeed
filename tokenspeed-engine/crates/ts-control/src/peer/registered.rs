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

//! The layout advertised after every rank registered its memory.
use super::{geometry::RankGeometry, layout::ReadEndpoint, v1::Digest};
use crate::codec::{Reader, Writer};
use sha2::{Digest as _, Sha256};
use ts_pd::layout::contract::Contract;

/// Registered memory of a single source or destination rank.
#[derive(Debug, Clone)]
pub struct Member {
    /// Session and field bindings from this member.
    pub geometry: RankGeometry,
    /// Exclusive Host virtual block bounds per scheduler group.
    pub bounds: Vec<i64>,
    /// Exclusive Disk virtual block bounds per scheduler group; empty when this
    /// member does not serve Disk pushes.
    pub disk_bounds: Vec<i64>,
    /// Successfully registered (address, length) segments.
    pub regions: Vec<(u64, u64)>,
}

impl Member {
    /// A bounded view for the physical planner.
    pub fn endpoint(&self) -> ReadEndpoint<'_> {
        ReadEndpoint {
            geometry: &self.geometry,
            host_virtual_bounds: &self.bounds,
            disk_virtual_bounds: &self.disk_bounds,
            regions: &self.regions,
        }
    }

    /// Replica bootstrap contribution including the member coordinate.
    pub fn encode(&self, rank: usize) -> Vec<u8> {
        let mut w = Writer::new();
        w.count(rank);
        w.bytes(&self.geometry.encode());
        w.count(self.bounds.len());
        for bound in &self.bounds {
            w.i64(*bound);
        }
        w.count(self.disk_bounds.len());
        for bound in &self.disk_bounds {
            w.i64(*bound);
        }
        w.count(self.regions.len());
        for &(a, n) in &self.regions {
            w.u64(a);
            w.u64(n);
        }
        w.finish()
    }

    fn decode(blob: &[u8], rank: usize, layout: &Contract) -> Option<Self> {
        if blob.len() > 1 << 20 {
            return None;
        }
        let mut r = Reader::new(blob);
        if r.count()? != rank {
            return None;
        }
        let geometry = RankGeometry::decode(r.bytes()?)?;
        if geometry.session.is_empty() || geometry.device.len() != layout.plan.fields.len() {
            return None;
        }
        let n = r.count()?;
        if n != layout.group_specs.len() {
            return None;
        }
        let bounds = (0..n)
            .map(|_| r.i64().filter(|v| *v >= 1))
            .collect::<Option<Vec<_>>>()?;
        let disk_n = r.count()?;
        let disk_bounds = if disk_n == 0 {
            Vec::new()
        } else {
            if disk_n != layout.group_specs.len()
                || geometry.host.is_none()
                || geometry.push.as_deref().is_none_or(str::is_empty)
            {
                return None;
            }
            (0..disk_n)
                .map(|_| r.i64().filter(|v| *v >= 2))
                .collect::<Option<Vec<_>>>()?
        };
        let n = r.count()?;
        if n == 0 || n > 8192 {
            return None;
        }
        let mut regions = Vec::new();
        for _ in 0..n {
            let a = r.u64()?;
            let n = r.u64()?;
            if a == 0 || n == 0 || a.checked_add(n).is_none() {
                return None;
            }
            regions.push((a, n));
        }
        r.finish(Self {
            geometry,
            bounds,
            disk_bounds,
            regions,
        })
    }
}

/// A pinned semantic identity and a single physical layout generation.
#[derive(Debug, Clone)]
pub struct Replica {
    /// Semantic digest from the runtime bootstrap.
    pub semantic: Digest,
    /// SHA-256 of all registered addresses, sessions, ownership and schema.
    pub digest: Digest,
    /// Canonical versioned descriptor, ready to answer discovery.
    pub descriptor: Vec<u8>,
    /// Rank-independent field contract.
    pub contract: Contract,
    /// Members in attention TP order. Width uniquely identifies the v1 role.
    pub members: Vec<Member>,
}

impl Replica {
    /// Whether every member of this replica serves Disk-tier peer pushes.
    pub fn can_push_disk(&self) -> bool {
        self.members
            .first()
            .is_some_and(|m| !m.disk_bounds.is_empty())
    }

    /// Construct a descriptor from a common contract and registered rank contributions.
    pub fn register(semantic: Digest, wire: &[u8], ranks: &[Vec<u8>]) -> Option<Self> {
        if !matches!(ranks.len(), 1 | 8) {
            return None;
        }
        let mut w = Writer::new();
        w.u8(2);
        w.u8(if ranks.len() == 8 { 0 } else { 1 });
        w.bytes(wire);
        w.count(ranks.len());
        for rank in ranks {
            w.bytes(rank);
        }
        Self::decode(semantic, w.finish())
    }
    /// Validate the role, coordinate order, group bounds and logical schema.
    pub fn decode(semantic: Digest, descriptor: Vec<u8>) -> Option<Self> {
        if descriptor.len() > 12 << 20 {
            return None;
        }
        let mut r = Reader::new(&descriptor);
        if r.u8()? != 2 {
            return None;
        }
        let role = r.u8()?;
        let contract = Contract::from_wire(r.bytes()?).ok()?;
        let n = r.count()?;
        if !matches!((role, n), (0, 8) | (1, 1)) {
            return None;
        }
        let members = (0..n)
            .map(|i| Member::decode(r.bytes()?, i, &contract))
            .collect::<Option<Vec<_>>>()?;
        r.finish(())?;
        if members
            .iter()
            .map(|m| &m.geometry.session)
            .collect::<std::collections::HashSet<_>>()
            .len()
            != n
        {
            return None;
        }
        let first_disk = &members[0].disk_bounds;
        if members.iter().any(|m| m.disk_bounds != *first_disk) {
            return None;
        }
        if !first_disk.is_empty()
            && members
                .iter()
                .filter_map(|m| m.geometry.push.as_deref())
                .collect::<std::collections::HashSet<_>>()
                .len()
                != n
        {
            return None;
        }
        let digest = Sha256::digest(&descriptor).into();
        Some(Self {
            semantic,
            digest,
            descriptor,
            contract,
            members,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer::geometry::{DeviceField, HostGroup, HostPool};

    fn test_contract_wire(width: i64) -> Vec<u8> {
        let kv_bytes = 64i64;
        let state_heads = 8 / width;
        let state_bytes = state_heads * 16;
        let lcm = kv_bytes.max(state_bytes);
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
                    { "plane_id": "kv", "bytes_per_lcm_block": kv_bytes, "arena_offset_bytes": 0 },
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
                        "page_stride_bytes": kv_bytes
                    },
                    {
                        "group_id": "state",
                        "field_id": "layer.0.st",
                        "plane_id": "st",
                        "shape": [state_heads, 16],
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
                        "partition": { "axis": 0, "global_extent": 8, "global_parts": [] }
                    }
                ]
            }
        });
        serde_json::to_vec(&payload).unwrap()
    }

    fn member(rank: usize, width: usize, disk_bounds: Vec<i64>, push: Option<String>) -> Member {
        let state_bytes = (8 / width as u64) * 16;
        Member {
            geometry: RankGeometry {
                session: format!("10.0.0.1:{}", 5000 + rank),
                device: vec![
                    DeviceField {
                        group_id: 0,
                        field_id: "layer.0.kv".into(),
                        address: 0x10000 + rank as u64 * 0x10000,
                        page_stride: 64,
                        bytes: 64,
                    },
                    DeviceField {
                        group_id: 1,
                        field_id: "layer.0.st".into(),
                        address: 0x20000 + rank as u64 * 0x10000,
                        page_stride: state_bytes,
                        bytes: state_bytes,
                    },
                ],
                host: Some(HostPool {
                    address: 0x80000,
                    groups: vec![
                        HostGroup {
                            parent_bytes: 128,
                            packing: 1,
                            block_bytes: 64,
                            field_offsets: vec![("layer.0.kv".into(), 0)],
                        },
                        HostGroup {
                            parent_bytes: 128,
                            packing: 1,
                            block_bytes: state_bytes,
                            field_offsets: vec![("layer.0.st".into(), 0)],
                        },
                    ],
                }),
                push,
            },
            bounds: vec![1 + 8 * width as i64, 9],
            disk_bounds,
            regions: vec![(0x10000, 0x100000)],
        }
    }

    #[test]
    fn disk_bounds_roundtrip_and_validation() {
        let sem = [0x42; 32];
        let wire8 = test_contract_wire(8);
        let wire1 = test_contract_wire(1);

        // 8-member TP8/KVP8 with Disk enabled
        let ranks_disk: Vec<Vec<u8>> = (0..8)
            .map(|r| member(r, 8, vec![257, 33], Some(format!("10.0.0.1:{}", 7000 + r))).encode(r))
            .collect();
        let rep8 = Replica::register(sem, &wire8, &ranks_disk).expect("valid TP8 Disk replica");
        assert!(rep8.can_push_disk());
        assert_eq!(rep8.members[0].disk_bounds, vec![257, 33]);
        assert_eq!(rep8.members[7].endpoint().disk_virtual_bounds, &[257, 33]);

        // 1-member TP1/KVP1 with Disk disabled -> can_push_disk() == false
        let ranks_no_disk = vec![member(0, 1, vec![], None).encode(0)];
        let rep1 = Replica::register(sem, &wire1, &ranks_no_disk).expect("valid TP1 replica");
        assert!(!rep1.can_push_disk());

        // Mismatched disk_bounds across members rejected
        let mut mixed = ranks_disk.clone();
        mixed[3] = member(3, 8, vec![], Some("10.0.0.1:7003".into())).encode(3);
        assert!(Replica::register(sem, &wire8, &mixed).is_none());

        // Duplicate push endpoint across members when Disk enabled rejected
        let mut dup_push = ranks_disk.clone();
        dup_push[2] = member(2, 8, vec![257, 33], Some("10.0.0.1:7000".into())).encode(2);
        assert!(Replica::register(sem, &wire8, &dup_push).is_none());

        // Disk bounds without push endpoint or without Host geometry rejected
        let no_push = vec![member(0, 1, vec![33, 33], None).encode(0)];
        assert!(Replica::register(sem, &wire1, &no_push).is_none());

        let mut no_host = member(0, 1, vec![33, 33], Some("10.0.0.1:7000".into()));
        no_host.geometry.host = None;
        assert!(Replica::register(sem, &wire1, &[no_host.encode(0)]).is_none());

        // Disk bound < 2 (cannot hold even block 1) rejected
        let too_small = vec![member(0, 1, vec![1, 33], Some("10.0.0.1:7000".into())).encode(0)];
        assert!(Replica::register(sem, &wire1, &too_small).is_none());

        // Legacy descriptor version byte 1 rejected
        let mut old_ver = rep8.descriptor.clone();
        old_ver[0] = 1;
        assert!(Replica::decode(sem, old_ver).is_none());
    }
}
