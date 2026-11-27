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
    /// Successfully registered (address, length) segments.
    pub regions: Vec<(u64, u64)>,
}

impl Member {
    /// A bounded view for the physical planner.
    pub fn endpoint(&self) -> ReadEndpoint<'_> {
        ReadEndpoint {
            geometry: &self.geometry,
            host_virtual_bounds: &self.bounds,
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
    /// Construct a descriptor from a common contract and registered rank contributions.
    pub fn register(semantic: Digest, wire: &[u8], ranks: &[Vec<u8>]) -> Option<Self> {
        if !matches!(ranks.len(), 1 | 8) {
            return None;
        }
        let mut w = Writer::new();
        w.u8(1);
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
        if r.u8()? != 1 {
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
