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

//! Direct placement of latest-snapshot fields whose device owners differ
//! from the scheduler replica. Every source row splits in group rank order.

use crate::{
    layout::{
        contract::{Contract, FieldLayout},
        manifest::Manifest,
        route::Fragment,
    },
    Copy,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const MAX_GROUP_BYTES: usize = 1 << 20;
const MAX_ROWS: u64 = 65536;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Field {
    base: u64,
    page_stride: u64,
    pages_per_owner: u64,
    rows_per_page: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Member {
    rank: u32,
    pub(crate) session: String,
    pub(crate) cache: String,
    fields: BTreeMap<String, Field>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Group {
    pub(crate) members: Vec<Member>,
    owner: usize,
}

fn payload(field: &FieldLayout) -> Result<u64, String> {
    field
        .shape
        .iter()
        .try_fold(field.element_size() as u64, |n, &d| {
            u64::try_from(d).ok().and_then(|d| n.checked_mul(d))
        })
        .filter(|&size| size > 0)
        .ok_or_else(|| "invalid state payload size".to_owned())
}

impl Field {
    fn range_end(&self, width: u64) -> Result<u64, String> {
        self.page_stride
            .checked_mul(self.pages_per_owner)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| self.base.checked_add(n))
            .ok_or_else(|| "state allocation address overflow".to_owned())
    }
}

impl Group {
    pub(crate) fn parse(
        raw: &[u8],
        cache: &str,
        session: &str,
        layout: &Contract,
    ) -> Result<Self, String> {
        if raw.len() > MAX_GROUP_BYTES {
            return Err("distributed state registration is too large".to_owned());
        }
        let members: Vec<Member> = serde_json::from_slice(raw).map_err(|err| err.to_string())?;
        let width = members.len() as u64;
        if !(2..=256).contains(&width) {
            return Err("invalid distributed state group size".to_owned());
        }
        let first = &members[0].fields;
        if first.is_empty() {
            return Err("distributed state registration has no fields".to_owned());
        }
        let (mut ranks, mut caches, mut sessions) =
            (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
        for member in &members {
            let valid_id = |s: &str| !s.is_empty() && s.len() <= 512 && s.is_ascii();
            if !valid_id(&member.cache)
                || !valid_id(&member.session)
                || !ranks.insert(member.rank)
                || !caches.insert(&member.cache)
                || !sessions.insert(&member.session)
                || member.fields.keys().ne(first.keys())
            {
                return Err("inconsistent distributed state membership or fields".to_owned());
            }
            let mut ranges = Vec::with_capacity(first.len());
            let mut span_count = 0u64;
            for (id, field) in &member.fields {
                let logical = layout.field(id).map_err(|e| e.0)?;
                let group = layout.group_spec(&logical.group_id).map_err(|e| e.0)?;
                let bytes = payload(logical)?;
                let rows = field.rows_per_page;
                span_count = span_count
                    .checked_add(rows)
                    .filter(|&n| n <= MAX_ROWS)
                    .ok_or("too many state rows in one transfer")?;
                if group.attn_kind != "state"
                    || group.shard_count != 1
                    || group.transfer_policy.as_deref() != Some("latest_snapshot")
                    || field.base == 0
                    || rows == 0
                    || rows != first[id].rows_per_page
                    || field.pages_per_owner < 2
                    || bytes % (rows * width) != 0
                    || (bytes / (rows * width)) % logical.element_size() as u64 != 0
                    || field.page_stride < bytes / width
                {
                    return Err(format!("invalid distributed state layout for {id:?}"));
                }
                ranges.push((field.base, field.range_end(width)?));
            }
            ranges.sort_unstable();
            if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
                return Err("overlapping state field allocations".to_owned());
            }
        }
        let owner = members
            .iter()
            .position(|m| m.cache == cache && m.session == session)
            .ok_or("distributed state owner differs from its cache registration")?;
        Ok(Self { members, owner })
    }

    /// Each member checks its own physical addresses against the regions
    /// installed in its native engine before anyone publishes the registration.
    pub(crate) fn check_local(&self, rank: i64, regions: &[(usize, usize)]) -> Result<(), String> {
        let member = &self.members[self.owner];
        if i64::from(member.rank) != rank {
            return Err("distributed cache owner does not match this device rank".to_owned());
        }
        for field in member.fields.values() {
            let end = field.range_end(self.members.len() as u64)?;
            if !regions.iter().any(|&(base, len)| {
                base as u64 <= field.base
                    && base.checked_add(len).is_some_and(|last| end <= last as u64)
            }) {
                return Err("state field outside registered transport memory".to_owned());
            }
        }
        Ok(())
    }

    pub(crate) fn check_source(
        &self,
        source: &Contract,
        destination: &Contract,
    ) -> Result<(), String> {
        for id in self.members[self.owner].fields.keys() {
            let src = source.field(id).map_err(|e| e.0)?;
            let dst = destination.field(id).map_err(|e| e.0)?;
            if src.shape != dst.shape || src.dtype != dst.dtype || src.group_id != dst.group_id {
                return Err(format!("distributed state needs equal full source and logical destination fields for {id:?}"));
            }
        }
        Ok(())
    }

    pub(crate) fn ordinary_fields(&self, layout: &Contract) -> BTreeSet<String> {
        layout
            .plan
            .fields
            .iter()
            .filter(|f| !self.members[self.owner].fields.contains_key(&f.field_id))
            .map(|f| f.field_id.clone())
            .collect()
    }

    pub(crate) fn plan(
        &self,
        contract: &Contract,
        arena: u64,
        source: &Manifest,
        destination: &Manifest,
        fragments: &[Fragment],
    ) -> Result<Vec<Copy>, String> {
        let blocks = |m: &Manifest| -> Result<BTreeMap<String, i64>, String> {
            if m.state_prefix_cached || m.prefix_len != 0 || m.prompt_len <= 0 {
                return Err("distributed state requires a cold full transfer".to_owned());
            }
            Ok(m.groups
                .iter()
                .filter_map(|g| match g.block_ids.as_slice() {
                    &[page] if page > 0 => Some((g.group_id.clone(), page)),
                    _ => None,
                })
                .collect())
        };
        if source.prompt_len != destination.prompt_len {
            return Err("distributed state prompt lengths disagree".to_owned());
        }
        let src = blocks(source)?;
        let dst = blocks(destination)?;
        let width = self.members.len() as u64;
        let mut copies = Vec::with_capacity(self.members.len());
        for (shard, member) in self.members.iter().enumerate() {
            let (mut sources, mut destinations, mut lengths) = (Vec::new(), Vec::new(), Vec::new());
            for (id, field) in &member.fields {
                // The ordinary typed route already selects the source stage and TP rank.
                let mut matches = fragments.iter().filter(|f| f.field_id == *id);
                let fragment = matches.next();
                if !fragments.is_empty() && fragment.is_none() {
                    continue;
                }
                let logical = match contract.field(id) {
                    Ok(field) => field,
                    // With a whole-field route, this stage sends only its resident fields.
                    Err(_) if fragments.is_empty() => continue,
                    Err(error) => return Err(error.0),
                };
                let bytes = payload(logical)?;
                if matches.next().is_some()
                    || fragment.is_some_and(|f| {
                        f.src_byte_offset != 0
                            || f.dst_byte_offset != 0
                            || f.rows_per_page != 1
                            || f.bytes_per_row as u64 != bytes
                    })
                {
                    return Err(format!(
                        "distributed state needs a full source field for {id:?}"
                    ));
                }
                let src_page = *src
                    .get(&logical.group_id)
                    .ok_or("missing source state block")?;
                let dst_page =
                    *dst.get(&logical.group_id)
                        .ok_or("missing destination state block")? as u64;
                if src_page
                    >= contract
                        .group(&logical.group_id)
                        .map_err(|e| e.0)?
                        .page_count
                    || dst_page >= field.pages_per_owner
                {
                    return Err("distributed state block outside owner allocation".to_owned());
                }
                let src_base = contract.field_address(arena, id).map_err(|e| e.0)?
                    + i128::from(src_page) * i128::from(logical.page_stride_bytes);
                let src_base =
                    u64::try_from(src_base).map_err(|_| "source state address overflow")?;
                // The complete allocation and the size of one row slice were validated at registration.
                let dst_base = field.base
                    + (self.owner as u64 * field.pages_per_owner + dst_page) * field.page_stride;
                let slice = bytes / field.rows_per_page / width;
                for row in 0..field.rows_per_page {
                    let offset = (row * width + shard as u64) * slice;
                    let start = src_base
                        .checked_add(offset)
                        .ok_or("source state row overflow")?;
                    start
                        .checked_add(slice)
                        .ok_or("source state row end overflow")?;
                    sources.push(start);
                    destinations.push(dst_base + row * slice);
                    lengths.push(slice);
                }
            }
            copies.push(Copy::Spans {
                sources,
                destinations,
                lengths,
            });
        }
        Ok(copies)
    }
}
