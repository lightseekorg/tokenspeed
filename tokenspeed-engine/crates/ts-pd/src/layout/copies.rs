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

//! The copies one transfer makes (`cache_transfer_blocks` and the host's
//! copy format).
//!
//! Each cache group gives one pages x fields grid ([`Copy::Pages`]) holding
//! every copy with one contiguous span per page -- a whole field on the
//! empty-fragment route, or a fragment whose rows collapsed into one -- so a
//! sharded or pipeline-staged source still goes through the page-gathered
//! WRITE. A fragment with several strided rows per page goes out one span per
//! row, after the grids, adjacent spans merged ([`Copy::Spans`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ts_mooncake::PageGrid;

use super::{
    contract::Contract,
    manifest::{Manifest, Selection},
    route::{local_transfer_pages, Fragment, OwnerFilters},
    LayoutError,
};
use crate::Copy;

/// The source blocks of a transfer.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// A whole request: this rank's manifest, position for position with
    /// the destination's.
    Manifest(&'a Manifest),
    /// One layerwise chunk: its ready blocks and their positions in the
    /// destination's manifest.
    Selection(&'a Selection),
}

/// One transfer to one Decode rank.
#[derive(Debug, Clone, Copy)]
pub struct Transfer<'a> {
    /// This rank's contract, with its landing addresses.
    pub layout: &'a Contract,
    /// This rank's arena base.
    pub arena_base: u64,
    /// The Decode rank's arena base.
    pub destination_base: u64,
    /// Its attention TP rank.
    pub destination_tp_rank: i64,
    /// Its contract.
    pub destination_layout: &'a Contract,
    /// The blocks to send.
    pub source: Source<'a>,
    /// The Decode rank's manifest of the request.
    pub destination: &'a Manifest,
    /// The registration route's fragments; empty copies every field whole.
    pub fragments: &'a [Fragment],
    /// The route's decision for every sharded group.
    pub owner_filters: &'a OwnerFilters,
}

/// One field's copies on a route: the group transfer it rides, its
/// `[source base, source page stride, destination base, destination page
/// stride, length]` row, and the fragment when it has several strided rows
/// per page.
struct FieldRow<'a> {
    group: usize,
    row: [i64; 5],
    strided: Option<&'a Fragment>,
}

/// Field rows by field, in the order the fields are first met.
#[derive(Default)]
struct FieldRows<'a> {
    rows: Vec<(String, Vec<FieldRow<'a>>)>,
    positions: HashMap<String, usize>,
}

impl<'a> FieldRows<'a> {
    fn push(&mut self, field_id: &str, row: FieldRow<'a>) {
        match self.positions.get(field_id) {
            Some(&position) => self.rows[position].1.push(row),
            None => {
                self.positions.insert(field_id.to_owned(), self.rows.len());
                self.rows.push((field_id.to_owned(), vec![row]));
            }
        }
    }
}

/// An address as a copy row's signed 64-bit value.
fn signed(value: i128) -> Result<i64, LayoutError> {
    i64::try_from(value).map_err(|_| LayoutError::new("a cache transfer address overflows int64"))
}

/// A span's unsigned 64-bit value.
fn unsigned(value: i128) -> Result<u64, LayoutError> {
    u64::try_from(value).map_err(|_| LayoutError::new("a cache transfer span overflows uint64"))
}

impl<'a> Transfer<'a> {
    /// The copies for each entry of `step_fields`, the fields that entry
    /// moves (`None`: every field); an empty list when none of them is on
    /// this route. The blocks are translated to pages and every field's
    /// geometry resolved once, for all entries.
    pub fn copies(
        &self,
        step_fields: &[Option<BTreeSet<String>>],
    ) -> Result<Vec<Vec<Copy>>, LayoutError> {
        let groups = self.group_pages()?;
        let rows = self.field_rows(&groups)?;
        step_fields
            .iter()
            .map(|fields| self.step_copies(&groups, &rows, fields.as_ref()))
            .collect()
    }

    /// Each group's source and destination pages on this edge, for the
    /// groups with any.
    #[allow(clippy::type_complexity)]
    fn group_pages(&self) -> Result<Vec<(&str, Vec<i64>, Vec<i64>)>, LayoutError> {
        let source_groups = match self.source {
            Source::Manifest(manifest) => manifest.groups.len(),
            Source::Selection(selection) => selection.groups.len(),
        };
        if source_groups != self.layout.group_specs.len()
            || self.destination.groups.len() != self.layout.group_specs.len()
        {
            return Err(LayoutError::new(
                "a cache transfer's manifests disagree with the layout's groups",
            ));
        }
        let mut groups = Vec::new();
        for (index, spec) in self.layout.group_specs.iter().enumerate() {
            let destination = &self.destination.groups[index].block_ids;
            let (source_blocks, destination_blocks) = match self.source {
                Source::Manifest(manifest) => (
                    manifest.groups[index].block_ids.clone(),
                    destination.clone(),
                ),
                Source::Selection(selection) => {
                    let group = &selection.groups[index];
                    let destination_blocks = group
                        .destination_positions
                        .iter()
                        .map(|&position| {
                            destination.get(position).copied().ok_or_else(|| {
                                LayoutError::new(
                                    "a layerwise selection names a position past the \
                                     destination manifest",
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (group.source_block_ids.clone(), destination_blocks)
                }
            };
            let source_owner_rank = if spec.shard_count > 1 {
                match self.owner_filters.get(&spec.group_id) {
                    None => {
                        return Err(LayoutError::new(format!(
                            "cache group {:?} has no owner-filter decision",
                            spec.group_id
                        )))
                    }
                    Some(None) => continue,
                    Some(Some(filter)) => filter.owner_rank,
                }
            } else {
                0
            };
            // KVP subgroups are aligned runs of consecutive attention-TP
            // ranks, so the destination's owner coordinate is its TP rank
            // modulo its group's shard count (1 for a replicated group).
            let destination_owner_rank =
                self.destination_tp_rank % self.destination_layout.shard_count(&spec.group_id)?;
            let (src_pages, dst_pages) = local_transfer_pages(
                &source_blocks,
                &destination_blocks,
                &spec.group_id,
                self.layout,
                self.destination_layout,
                source_owner_rank,
                destination_owner_rank,
            )?;
            if !src_pages.is_empty() {
                groups.push((spec.group_id.as_str(), src_pages, dst_pages));
            }
        }
        Ok(groups)
    }

    /// Every field's copies on this route, by field, in the order the
    /// fields are first met.
    #[allow(clippy::type_complexity)]
    fn field_rows(
        &self,
        groups: &[(&str, Vec<i64>, Vec<i64>)],
    ) -> Result<Vec<(String, Vec<FieldRow<'a>>)>, LayoutError> {
        let mut rows = FieldRows::default();
        let source_field = |group_id: &str, field_id: &str| {
            self.layout
                .field(field_id)
                .ok()
                .filter(|field| field.group_id == group_id)
                .ok_or_else(|| LayoutError::new(format!("unknown source cache field {field_id:?}")))
        };
        let destination_field = |group_id: &str, field_id: &str| {
            self.destination_layout
                .field(field_id)
                .ok()
                .filter(|field| field.group_id == group_id)
                .ok_or_else(|| {
                    LayoutError::new(format!("unknown destination cache field {field_id:?}"))
                })
        };
        for (index, &(group_id, _, _)) in groups.iter().enumerate() {
            if !self.fragments.is_empty() {
                for fragment in self.fragments.iter().filter(|f| f.group_id == group_id) {
                    let source = source_field(group_id, &fragment.field_id)?;
                    let destination = destination_field(group_id, &fragment.field_id)?;
                    let row = [
                        signed(
                            self.layout
                                .field_address(self.arena_base, &fragment.field_id)?
                                + i128::from(fragment.src_byte_offset),
                        )?,
                        source.page_stride_bytes,
                        signed(
                            self.destination_layout
                                .field_address(self.destination_base, &fragment.field_id)?
                                + i128::from(fragment.dst_byte_offset),
                        )?,
                        destination.page_stride_bytes,
                        fragment.bytes_per_row,
                    ];
                    rows.push(
                        &fragment.field_id,
                        FieldRow {
                            group: index,
                            row,
                            strided: (fragment.rows_per_page > 1).then_some(fragment),
                        },
                    );
                }
            } else {
                for source in self.layout.fields_for_group(group_id) {
                    let destination = destination_field(group_id, &source.field_id)?;
                    let row = [
                        signed(
                            self.layout
                                .field_address(self.arena_base, &source.field_id)?,
                        )?,
                        source.page_stride_bytes,
                        signed(
                            self.destination_layout
                                .field_address(self.destination_base, &destination.field_id)?,
                        )?,
                        destination.page_stride_bytes,
                        source.payload_bytes(),
                    ];
                    rows.push(
                        &source.field_id,
                        FieldRow {
                            group: index,
                            row,
                            strided: None,
                        },
                    );
                }
            }
        }
        Ok(rows.rows)
    }

    /// One entry's copies: a grid per group of its whole-span rows, then the
    /// strided rows one span each, adjacent spans merged.
    fn step_copies(
        &self,
        groups: &[(&str, Vec<i64>, Vec<i64>)],
        rows: &[(String, Vec<FieldRow<'a>>)],
        fields: Option<&BTreeSet<String>>,
    ) -> Result<Vec<Copy>, LayoutError> {
        let selected: Vec<&Vec<FieldRow<'_>>> = match fields {
            None => rows.iter().map(|(_, field_rows)| field_rows).collect(),
            Some(fields) => {
                let mut matching: Vec<&(String, Vec<FieldRow<'_>>)> = rows
                    .iter()
                    .filter(|(field, _)| fields.contains(field))
                    .collect();
                matching.sort_by(|a, b| a.0.cmp(&b.0));
                matching
                    .into_iter()
                    .map(|(_, field_rows)| field_rows)
                    .collect()
            }
        };
        let mut span_rows: BTreeMap<usize, Vec<[i64; 5]>> = BTreeMap::new();
        let mut strided = Vec::new();
        for field_row in selected.into_iter().flatten() {
            match field_row.strided {
                None => span_rows
                    .entry(field_row.group)
                    .or_default()
                    .push(field_row.row),
                Some(fragment) => strided.push((field_row.group, field_row.row, fragment)),
            }
        }
        let mut copies: Vec<Copy> = span_rows
            .into_iter()
            .map(|(group, fields)| {
                Copy::Pages(PageGrid {
                    src_pages: groups[group].1.clone(),
                    dst_pages: groups[group].2.clone(),
                    fields,
                })
            })
            .collect();
        let mut spans = Spans::default();
        for (group, row, fragment) in strided {
            let [src_base, src_page_stride, dst_base, dst_page_stride, _] = row;
            strided_page_spans(
                fragment,
                (src_base, src_page_stride, &groups[group].1),
                (dst_base, dst_page_stride, &groups[group].2),
                &mut spans,
            )?;
        }
        if !spans.sources.is_empty() {
            copies.push(Copy::Spans {
                sources: spans.sources,
                destinations: spans.destinations,
                lengths: spans.lengths,
            });
        }
        Ok(copies)
    }
}

/// Spans being collected, adjacent ones merged (`coalesce_transfer_blocks`).
#[derive(Default)]
struct Spans {
    sources: Vec<u64>,
    destinations: Vec<u64>,
    lengths: Vec<u64>,
}

impl Spans {
    fn push(&mut self, source: i128, destination: i128, length: i128) -> Result<(), LayoutError> {
        let (source, destination, length) =
            (unsigned(source)?, unsigned(destination)?, unsigned(length)?);
        if let (Some(last_source), Some(last_destination), Some(last_length)) = (
            self.sources.last(),
            self.destinations.last(),
            self.lengths.last_mut(),
        ) {
            if last_source.checked_add(*last_length) == Some(source)
                && last_destination.checked_add(*last_length) == Some(destination)
            {
                *last_length = last_length
                    .checked_add(length)
                    .ok_or_else(|| LayoutError::new("a cache transfer span overflows uint64"))?;
                return Ok(());
            }
        }
        self.sources.push(source);
        self.destinations.push(destination);
        self.lengths.push(length);
        Ok(())
    }
}

/// The per-page spans of a fragment with several strided rows per page
/// (`_strided_page_copies`, expanded): rows contiguous on both sides go as
/// one span, else one span per row.
fn strided_page_spans(
    fragment: &Fragment,
    (src_base, src_page_stride, src_pages): (i64, i64, &[i64]),
    (dst_base, dst_page_stride, dst_pages): (i64, i64, &[i64]),
    spans: &mut Spans,
) -> Result<(), LayoutError> {
    let width = i128::from(fragment.bytes_per_row);
    let rows = i128::from(fragment.rows_per_page);
    let src_pitch = i128::from(fragment.src_row_stride_bytes);
    let dst_pitch = i128::from(fragment.dst_row_stride_bytes);
    if dst_pitch == width && src_pitch != width && (width <= 0 || src_pitch < width || rows <= 0) {
        // The packed copy the destination-contiguous rows form is invalid.
        return Err(LayoutError::new(format!(
            "invalid PackedCopy width={width} src_pitch={src_pitch} rows={rows}"
        )));
    }
    for (&src_page, &dst_page) in src_pages.iter().zip(dst_pages) {
        let src_page_addr =
            i128::from(src_base) + i128::from(src_page) * i128::from(src_page_stride);
        let dst_page_addr =
            i128::from(dst_base) + i128::from(dst_page) * i128::from(dst_page_stride);
        if dst_pitch == width && src_pitch == width {
            spans.push(src_page_addr, dst_page_addr, width * rows)?;
            continue;
        }
        for row in 0..rows {
            spans.push(
                src_page_addr + row * src_pitch,
                dst_page_addr + row * dst_pitch,
                width,
            )?;
        }
    }
    Ok(())
}
