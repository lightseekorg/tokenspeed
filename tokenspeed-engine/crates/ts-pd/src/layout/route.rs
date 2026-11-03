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

//! Transfer routes between a Prefill rank set and a Decode rank set
//! (`pd/transfer_plan.py`).
//!
//! Two source geometries compose: head partitions split a page's rows over
//! TP ranks, and KVP page sharding (`GroupSpec::shard_count`) deals whole
//! pages over a consecutive TP subgroup. Source and destination groups shard
//! independently. A route says, per serving Prefill rank, which row
//! fragments of which fields go to one Decode rank, and for every sharded
//! group which of its blocks that rank sends.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{
    contract::{validate_peer, Contract, FieldLayout, Partition},
    LayoutError,
};

/// The widest attention TP a route plans for (`MAX_CACHE_TP_SIZE`).
pub const MAX_TP_SIZE: i64 = 1024;

/// One field-relative row fragment copied for every selected page
/// (`CacheTransferFragment`). Arena bases, page bases and page strides are
/// resolved from the two contracts when the copies are planned, so the
/// fragment never becomes a second, independently trusted cache ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    /// The field's group.
    pub group_id: String,
    /// The field.
    pub field_id: String,
    /// The fragment's offset within a source page of the field.
    pub src_byte_offset: i64,
    /// Its offset within a destination page.
    pub dst_byte_offset: i64,
    /// Bytes between consecutive source rows.
    pub src_row_stride_bytes: i64,
    /// Bytes between consecutive destination rows.
    pub dst_row_stride_bytes: i64,
    /// Bytes copied per row.
    pub bytes_per_row: i64,
    /// Rows per page.
    pub rows_per_page: i64,
}

/// Which of a sharded group's scheduler blocks one source rank holds
/// (`CachePageOwnerFilter`): block `v` when `(v - 1) % owner_count ==
/// owner_rank` (block 0 is the null block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerFilter {
    /// The rank's position in its KVP subgroup.
    pub owner_rank: i64,
    /// The subgroup's size.
    pub owner_count: i64,
}

impl OwnerFilter {
    /// A filter; needs `0 <= owner_rank < owner_count` and `owner_count >= 2`.
    pub fn new(owner_rank: i64, owner_count: i64) -> Result<Self, LayoutError> {
        if owner_count < 2 || !(0..owner_count).contains(&owner_rank) {
            return Err(LayoutError::new(
                "cache page owner filter needs 0 <= rank < count >= 2",
            ));
        }
        Ok(Self {
            owner_rank,
            owner_count,
        })
    }
}

/// One source rank's decision per sharded group: the filter selecting the
/// blocks it sends, or `None` when its route carries no fragment of the
/// group and it sends nothing for it. Replicated groups are never listed.
pub type OwnerFilters = BTreeMap<String, Option<OwnerFilter>>;

/// The route to one Decode rank (`RankTransferPlan`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankPlan {
    /// Each serving Prefill rank's fragments; empty fragments copy every
    /// field whole.
    pub fragments_by_prefill_rank: BTreeMap<i64, Vec<Fragment>>,
    /// Each serving Prefill rank's owner-filter decisions.
    pub owner_filters_by_prefill_rank: BTreeMap<i64, OwnerFilters>,
}

impl RankPlan {
    /// The serving Prefill ranks, ascending.
    pub fn target_prefill_ranks(&self) -> Vec<i64> {
        self.fragments_by_prefill_rank.keys().copied().collect()
    }
}

/// Check one source rank's owner-filter decisions against its cache groups
/// (`validate_rank_owner_filters`): every sharded group has a decision and
/// no replicated one does; a filter matches the group's shard count and
/// comes with fragments of the group, a `None` decision without.
pub fn validate_owner_filters(
    group_specs: &[super::contract::GroupSpec],
    fragments: &[Fragment],
    owner_filters: &OwnerFilters,
) -> Result<(), LayoutError> {
    let shard_counts: Vec<(&str, i64)> = group_specs
        .iter()
        .filter(|spec| spec.shard_count != 1)
        .map(|spec| (spec.group_id.as_str(), spec.shard_count))
        .collect();
    let routed: BTreeSet<&str> = fragments.iter().map(|f| f.group_id.as_str()).collect();
    for group_id in owner_filters.keys() {
        if !shard_counts.iter().any(|(sharded, _)| sharded == group_id) {
            return Err(LayoutError::new(format!(
                "owner filter names cache group {group_id:?}, which is not sharded"
            )));
        }
    }
    for (group_id, shard_count) in shard_counts {
        let Some(filter) = owner_filters.get(group_id) else {
            return Err(LayoutError::new(format!(
                "cache group {group_id:?} is sharded but the registration route carries no \
                 owner-filter decision for it"
            )));
        };
        let Some(filter) = filter else {
            if routed.contains(group_id) {
                return Err(LayoutError::new(format!(
                    "the registration route carries fragments of sharded cache group \
                     {group_id:?} but no owner filter"
                )));
            }
            continue;
        };
        if filter.owner_count != shard_count {
            return Err(LayoutError::new(format!(
                "cache group {group_id:?} owner filter ({} owners) disagrees with its shard \
                 count {shard_count}",
                filter.owner_count
            )));
        }
        if !routed.contains(group_id) {
            return Err(LayoutError::new(format!(
                "cache group {group_id:?} has an owner filter but the registration route \
                 carries no fragment of it"
            )));
        }
    }
    Ok(())
}

/// A half-open range of a partitioned axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    start: i64,
    end: i64,
}

impl Interval {
    fn length(self) -> i64 {
        self.end - self.start
    }

    fn intersect(self, other: Self) -> Option<Self> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        (start < end).then_some(Self { start, end })
    }
}

/// One part of a rank's slice of a partitioned axis, and where it starts in
/// the rank's local extent.
#[derive(Debug, Clone, Copy)]
struct RankPartition {
    interval: Interval,
    local_offset: i64,
}

/// How many distinct shards a field's partitioned axis has: its global
/// extent over its local one.
fn distinct_shards(
    segment: &FieldLayout,
    partition: &Partition,
    tp_size: i64,
) -> Result<i64, LayoutError> {
    let local_extent = usize::try_from(partition.axis)
        .ok()
        .and_then(|axis| segment.shape.get(axis))
        .copied()
        .filter(|&extent| extent > 0)
        .ok_or_else(|| {
            LayoutError::new(format!(
                "cache field {:?} has invalid partition geometry",
                segment.field_id
            ))
        })?;
    let distinct = partition.global_extent / local_extent;
    if distinct <= 0 || distinct > tp_size || tp_size % distinct != 0 {
        return Err(LayoutError::new(format!(
            "Cache field {:?} cannot map global extent {} and local extent {local_extent} to \
             TP={tp_size}",
            segment.field_id, partition.global_extent
        )));
    }
    Ok(distinct)
}

/// The slices of a partitioned field `tp_rank` holds (`_rank_partitions`).
fn rank_partitions(
    segment: &FieldLayout,
    partition: &Partition,
    tp_size: i64,
    tp_rank: i64,
) -> Result<Vec<RankPartition>, LayoutError> {
    let distinct = distinct_shards(segment, partition, tp_size)?;
    let replica_group_size = tp_size / distinct;
    let shard_rank = tp_rank / replica_group_size;
    let parts = if partition.global_parts.is_empty() {
        vec![partition.global_extent]
    } else {
        partition.global_parts.clone()
    };
    let mut partitions = Vec::with_capacity(parts.len());
    let (mut global_offset, mut local_offset) = (0, 0);
    for part in parts {
        let local_part = part / distinct;
        let start = global_offset + shard_rank * local_part;
        partitions.push(RankPartition {
            interval: Interval {
                start,
                end: start + local_part,
            },
            local_offset,
        });
        global_offset += part;
        local_offset += local_part;
    }
    Ok(partitions)
}

/// The rank that sends a partitioned field's slice for its replica group
/// (`_is_representative_rank`).
fn is_representative_rank(
    segment: &FieldLayout,
    partition: &Partition,
    tp_size: i64,
    tp_rank: i64,
) -> Result<bool, LayoutError> {
    let replica_group_size = tp_size / distinct_shards(segment, partition, tp_size)?;
    Ok(tp_rank % replica_group_size == 0)
}

/// The product of `dimensions`, 1 when empty.
fn product(dimensions: &[i64]) -> i64 {
    dimensions.iter().product()
}

/// One fragment of a field pair (`_make_fragment`): a whole-page copy for an
/// unpartitioned field, else the rows of the two slices' intersection, whose
/// contiguous rows collapse into one.
struct FragmentGeometry<'a> {
    group_id: &'a str,
    prefill: &'a FieldLayout,
    decode: &'a FieldLayout,
}

impl FragmentGeometry<'_> {
    fn whole(&self) -> Fragment {
        self.collapse(Fragment {
            group_id: self.group_id.to_owned(),
            field_id: self.prefill.field_id.clone(),
            src_byte_offset: 0,
            dst_byte_offset: 0,
            src_row_stride_bytes: self.prefill.payload_bytes(),
            dst_row_stride_bytes: self.decode.payload_bytes(),
            bytes_per_row: self.prefill.payload_bytes(),
            rows_per_page: 1,
        })
    }

    fn sliced(
        &self,
        partition: &Partition,
        intersection: Interval,
        prefill: RankPartition,
        decode: RankPartition,
    ) -> Fragment {
        let axis = partition.axis as usize;
        let inner_bytes = product(&self.prefill.shape[axis + 1..]) * self.prefill.element_size();
        self.collapse(Fragment {
            group_id: self.group_id.to_owned(),
            field_id: self.prefill.field_id.clone(),
            src_byte_offset: (prefill.local_offset + intersection.start - prefill.interval.start)
                * inner_bytes,
            dst_byte_offset: (decode.local_offset + intersection.start - decode.interval.start)
                * inner_bytes,
            src_row_stride_bytes: self.prefill.shape[axis] * inner_bytes,
            dst_row_stride_bytes: self.decode.shape[axis] * inner_bytes,
            bytes_per_row: intersection.length() * inner_bytes,
            rows_per_page: product(&self.prefill.shape[..axis]),
        })
    }

    /// Rows contiguous on both sides are one row.
    fn collapse(&self, mut fragment: Fragment) -> Fragment {
        if fragment.rows_per_page > 1
            && fragment.src_row_stride_bytes == fragment.bytes_per_row
            && fragment.dst_row_stride_bytes == fragment.bytes_per_row
        {
            fragment.bytes_per_row *= fragment.rows_per_page;
            fragment.src_row_stride_bytes = fragment.bytes_per_row;
            fragment.dst_row_stride_bytes = fragment.bytes_per_row;
            fragment.rows_per_page = 1;
        }
        fragment
    }
}

/// Routes between one Prefill rank set and one Decode rank set
/// (`CacheTransferPlanner`).
#[derive(Debug, Clone)]
pub struct Planner {
    prefill_tp_size: i64,
    decode_tp_size: i64,
    /// The stage's resident fields, `None` for the complete plan.
    field_ids: Option<BTreeSet<String>>,
    has_sharded_cache: bool,
    /// The Prefill side's sharded groups and their shard counts, in group
    /// order.
    shard_counts: Vec<(String, i64)>,
    partitions: HashMap<String, Option<Partition>>,
    /// The transferred fields: group, Prefill field, Decode field.
    segment_pairs: Vec<(String, FieldLayout, FieldLayout)>,
    decode_ranks_by_prefill_rank: BTreeMap<i64, BTreeSet<i64>>,
}

impl Planner {
    /// Plan fragments between the two rank sets; `prefill_field_ids` are the
    /// resident fields to transfer, `None` for the complete plan.
    pub fn new(
        prefill_tp_size: i64,
        decode_tp_size: i64,
        prefill_layout: &Contract,
        decode_layout: &Contract,
        prefill_field_ids: Option<&BTreeSet<String>>,
    ) -> Result<Self, LayoutError> {
        if prefill_tp_size <= 0 || decode_tp_size <= 0 {
            return Err(LayoutError::new("Cache TP sizes must be positive"));
        }
        if prefill_tp_size > MAX_TP_SIZE || decode_tp_size > MAX_TP_SIZE {
            return Err(LayoutError::new(format!(
                "Cache TP sizes cannot exceed {MAX_TP_SIZE}"
            )));
        }
        for (layout, tp_size) in [
            (prefill_layout, prefill_tp_size),
            (decode_layout, decode_tp_size),
        ] {
            for spec in &layout.group_specs {
                if tp_size % spec.shard_count != 0 {
                    return Err(LayoutError::new(
                        "Cache shard count must divide attention TP size",
                    ));
                }
                if spec.shard_count > 1
                    && layout
                        .fields_for_group(&spec.group_id)
                        .iter()
                        .any(|field| layout.partition_for(&field.field_id).is_some())
                {
                    return Err(LayoutError::new(
                        "KVP transfer cannot use head-partitioned fields within page-sharded \
                         cache groups",
                    ));
                }
            }
        }
        let has_sharded_cache = prefill_layout
            .group_specs
            .iter()
            .chain(&decode_layout.group_specs)
            .any(|spec| spec.shard_count > 1);
        let all_fields: BTreeSet<String> = prefill_layout
            .plan
            .fields
            .iter()
            .map(|field| field.field_id.clone())
            .collect();
        if let Some(fields) = prefill_field_ids {
            if !fields.is_subset(&all_fields) {
                return Err(LayoutError::new(
                    "stage placement contains unknown cache fields",
                ));
            }
        }
        let field_ids = prefill_field_ids
            .filter(|&fields| *fields != all_fields)
            .cloned();
        validate_peer(prefill_layout, decode_layout)?;

        let partitions: HashMap<String, Option<Partition>> = prefill_layout
            .plan
            .fields
            .iter()
            .map(|field| {
                (
                    field.field_id.clone(),
                    prefill_layout.partition_for(&field.field_id).cloned(),
                )
            })
            .collect();
        let shard_counts: Vec<(String, i64)> = prefill_layout
            .group_specs
            .iter()
            .filter(|spec| spec.shard_count > 1)
            .map(|spec| (spec.group_id.clone(), spec.shard_count))
            .collect();
        let mut segment_pairs = Vec::new();
        for (prefill_spec, decode_spec) in prefill_layout
            .group_specs
            .iter()
            .zip(&decode_layout.group_specs)
        {
            let prefill_fields = prefill_layout.fields_for_group(&prefill_spec.group_id);
            let decode_fields = decode_layout.fields_for_group(&decode_spec.group_id);
            for (prefill_field, decode_field) in prefill_fields.into_iter().zip(decode_fields) {
                if prefill_field_ids.is_none_or(|fields| fields.contains(&prefill_field.field_id)) {
                    segment_pairs.push((
                        prefill_spec.group_id.clone(),
                        prefill_field.clone(),
                        decode_field.clone(),
                    ));
                }
            }
        }
        let mut planner = Self {
            prefill_tp_size,
            decode_tp_size,
            field_ids,
            has_sharded_cache,
            shard_counts,
            partitions,
            segment_pairs,
            decode_ranks_by_prefill_rank: BTreeMap::new(),
        };
        for (group_id, prefill_field, decode_field) in &planner.segment_pairs {
            planner.validate_tp_mapping(prefill_field, decode_field)?;
            if planner.shard_count(group_id) > 1
                && planner.partitions[&prefill_field.field_id].is_some()
            {
                // Head partitions place a page's rows on distinct TP ranks
                // and page sharding places whole pages on distinct ranks; a
                // field under both would need rows no single rank holds.
                return Err(LayoutError::new(format!(
                    "cache field {:?} is both head-partitioned and page-sharded on Prefill",
                    prefill_field.field_id
                )));
            }
        }
        planner.decode_ranks_by_prefill_rank = planner.source_decode_ranks()?;
        Ok(planner)
    }

    /// The Decode ranks each Prefill rank serves.
    pub fn decode_ranks_by_prefill_rank(&self) -> &BTreeMap<i64, BTreeSet<i64>> {
        &self.decode_ranks_by_prefill_rank
    }

    /// The Prefill side's sharded groups, in group order.
    pub fn sharded_group_ids(&self) -> Vec<String> {
        self.shard_counts
            .iter()
            .map(|(group, _)| group.clone())
            .collect()
    }

    fn shard_count(&self, group_id: &str) -> i64 {
        self.shard_counts
            .iter()
            .find(|(group, _)| group == group_id)
            .map_or(1, |&(_, count)| count)
    }

    /// Equal TP, every field resident, no sharded group: each Decode rank
    /// reads its same-index Prefill rank whole, which empty fragments say.
    fn uses_whole_copy_route(&self) -> bool {
        self.prefill_tp_size == self.decode_tp_size
            && self.field_ids.is_none()
            && !self.has_sharded_cache
    }

    /// The route to `decode_tp_rank` (`plan_for_decode_rank`).
    pub fn plan_for_decode_rank(&self, decode_tp_rank: i64) -> Result<RankPlan, LayoutError> {
        if !(0..self.decode_tp_size).contains(&decode_tp_rank) {
            return Err(LayoutError::new(format!(
                "decode_tp_rank={decode_tp_rank} is out of range"
            )));
        }
        let fragments_by_prefill_rank = if self.uses_whole_copy_route() {
            BTreeMap::from([(decode_tp_rank, Vec::new())])
        } else {
            let fragments = self.fragments_for_decode_rank(decode_tp_rank)?;
            if fragments.is_empty() {
                return Err(LayoutError::new(format!(
                    "Cache-transfer decode TP rank {decode_tp_rank} has no source fragments"
                )));
            }
            fragments
        };
        let owner_filters_by_prefill_rank = fragments_by_prefill_rank
            .iter()
            .map(|(&rank, fragments)| Ok((rank, self.owner_filters_for(rank, fragments)?)))
            .collect::<Result<_, LayoutError>>()?;
        Ok(RankPlan {
            fragments_by_prefill_rank,
            owner_filters_by_prefill_rank,
        })
    }

    /// Every sharded group's decision for one source rank
    /// (`owner_filters_for`): a rank whose fragments name the group owns the
    /// blocks of its KVP subgroup position, one whose fragments do not
    /// sends nothing for it.
    pub fn owner_filters_for(
        &self,
        prefill_rank: i64,
        fragments: &[Fragment],
    ) -> Result<OwnerFilters, LayoutError> {
        let routed: BTreeSet<&str> = fragments.iter().map(|f| f.group_id.as_str()).collect();
        self.shard_counts
            .iter()
            .map(|(group_id, count)| {
                let filter = if routed.contains(group_id.as_str()) {
                    Some(OwnerFilter::new(prefill_rank % count, *count)?)
                } else {
                    None
                };
                Ok((group_id.clone(), filter))
            })
            .collect()
    }

    fn validate_tp_mapping(
        &self,
        prefill: &FieldLayout,
        decode: &FieldLayout,
    ) -> Result<(), LayoutError> {
        if self.prefill_tp_size == self.decode_tp_size
            && (prefill.shape != decode.shape || prefill.payload_bytes() != decode.payload_bytes())
        {
            return Err(LayoutError::new(format!(
                "equal-TP cache field {:?} rank-local geometry differs",
                prefill.field_id
            )));
        }
        if let Some(partition) = &self.partitions[&prefill.field_id] {
            rank_partitions(prefill, partition, self.prefill_tp_size, 0)?;
            rank_partitions(decode, partition, self.decode_tp_size, 0)?;
        }
        Ok(())
    }

    fn fragments_for_decode_rank(
        &self,
        decode_tp_rank: i64,
    ) -> Result<BTreeMap<i64, Vec<Fragment>>, LayoutError> {
        let mut fragments: BTreeMap<i64, Vec<Fragment>> = BTreeMap::new();
        for (group_id, prefill, decode) in &self.segment_pairs {
            let geometry = FragmentGeometry {
                group_id,
                prefill,
                decode,
            };
            let Some(partition) = &self.partitions[&prefill.field_id] else {
                let fragment = geometry.whole();
                // The replica a Decode rank reads whole.
                let replica_rank = decode_tp_rank * self.prefill_tp_size / self.decode_tp_size;
                let shard_count = self.shard_count(group_id);
                if shard_count == 1 {
                    fragments.entry(replica_rank).or_default().push(fragment);
                    continue;
                }
                // That replica is spread over its KVP subgroup (consecutive
                // TP ranks); every member sends the blocks it owns, which
                // `owner_filters_for` records.
                let subgroup_base = replica_rank - replica_rank % shard_count;
                for prefill_rank in subgroup_base..subgroup_base + shard_count {
                    fragments
                        .entry(prefill_rank)
                        .or_default()
                        .push(fragment.clone());
                }
                continue;
            };
            let decode_partitions =
                rank_partitions(decode, partition, self.decode_tp_size, decode_tp_rank)?;
            for prefill_rank in 0..self.prefill_tp_size {
                if !is_representative_rank(prefill, partition, self.prefill_tp_size, prefill_rank)?
                {
                    continue;
                }
                let prefill_partitions =
                    rank_partitions(prefill, partition, self.prefill_tp_size, prefill_rank)?;
                for (source, destination) in prefill_partitions.iter().zip(&decode_partitions) {
                    let Some(intersection) = source.interval.intersect(destination.interval) else {
                        continue;
                    };
                    fragments
                        .entry(prefill_rank)
                        .or_default()
                        .push(geometry.sliced(partition, intersection, *source, *destination));
                }
            }
        }
        Ok(fragments)
    }

    fn source_decode_ranks(&self) -> Result<BTreeMap<i64, BTreeSet<i64>>, LayoutError> {
        if self.uses_whole_copy_route() {
            return Ok((0..self.prefill_tp_size)
                .map(|rank| (rank, BTreeSet::from([rank])))
                .collect());
        }
        let mut decode_ranks: BTreeMap<i64, BTreeSet<i64>> = (0..self.prefill_tp_size)
            .map(|rank| (rank, BTreeSet::new()))
            .collect();
        for decode_tp_rank in 0..self.decode_tp_size {
            for prefill_rank in self.fragments_for_decode_rank(decode_tp_rank)?.into_keys() {
                decode_ranks
                    .entry(prefill_rank)
                    .or_default()
                    .insert(decode_tp_rank);
            }
        }
        Ok(decode_ranks)
    }
}

/// Require a nonempty stage list covering each logical field exactly once
/// (`validate_cache_stage_fields`).
pub fn validate_stage_fields(
    layout: &Contract,
    fields_by_stage: &[Vec<String>],
) -> Result<(), LayoutError> {
    let fields: Vec<&String> = fields_by_stage.iter().flatten().collect();
    let unique: BTreeSet<&String> = fields.iter().copied().collect();
    let planned: BTreeSet<&String> = layout.plan.fields.iter().map(|f| &f.field_id).collect();
    if fields_by_stage.is_empty() || fields.len() != unique.len() || unique != planned {
        return Err(LayoutError::new(
            "cache stage placement must cover every logical field exactly once",
        ));
    }
    Ok(())
}

/// Every Prefill stage's route to one Decode TP rank, stage-major
/// (`build_pipeline_transfer_plan`), and the ranks that join its completion
/// with no data (planned for Decode TP rank 0 only).
pub fn pipeline_plan(
    prefill_tp_size: i64,
    decode_tp_size: i64,
    decode_tp_rank: i64,
    prefill_layout: &Contract,
    decode_layout: &Contract,
    fields_by_stage: &[Vec<String>],
) -> Result<(RankPlan, Vec<i64>), LayoutError> {
    validate_stage_fields(prefill_layout, fields_by_stage)?;
    let mut plan = RankPlan {
        fragments_by_prefill_rank: BTreeMap::new(),
        owner_filters_by_prefill_rank: BTreeMap::new(),
    };
    let mut dummy = Vec::new();
    for (stage, fields) in fields_by_stage.iter().enumerate() {
        let fields: BTreeSet<String> = fields.iter().cloned().collect();
        let planner = Planner::new(
            prefill_tp_size,
            decode_tp_size,
            prefill_layout,
            decode_layout,
            Some(&fields),
        )?;
        let stage_plan = planner.plan_for_decode_rank(decode_tp_rank)?;
        let base = stage as i64 * prefill_tp_size;
        for (rank, fragments) in stage_plan.fragments_by_prefill_rank {
            plan.fragments_by_prefill_rank
                .insert(base + rank, fragments);
        }
        for (rank, filters) in stage_plan.owner_filters_by_prefill_rank {
            plan.owner_filters_by_prefill_rank
                .insert(base + rank, filters);
        }
        if decode_tp_rank == 0 {
            dummy.extend(
                planner
                    .decode_ranks_by_prefill_rank()
                    .iter()
                    .filter(|(_, decode_ranks)| decode_ranks.is_empty())
                    .map(|(&rank, _)| base + rank),
            );
        }
    }
    dummy.sort_unstable();
    Ok((plan, dummy))
}

/// The local pages of the block pairs both ends of a transfer edge own
/// (`local_transfer_pages`): the input pairs logical positions, block ids
/// are each side's own, and the owner ranks are the two ends' positions in
/// their KVP subgroups (0 for a replicated group). Returns equally long
/// source and destination page lists; an edge owning no pair gets empty
/// ones and still joins the acknowledgment.
pub fn local_transfer_pages(
    source_blocks: &[i64],
    destination_blocks: &[i64],
    group_id: &str,
    source_layout: &Contract,
    destination_layout: &Contract,
    source_owner_rank: i64,
    destination_owner_rank: i64,
) -> Result<(Vec<i64>, Vec<i64>), LayoutError> {
    if source_blocks.len() != destination_blocks.len() {
        return Err(LayoutError::new(
            "Cache transfer source and destination pages differ in count",
        ));
    }
    let outside = || LayoutError::new("Cache transfer block is outside the virtual address space");
    if source_blocks
        .iter()
        .chain(destination_blocks)
        .any(|&block| block <= 0)
    {
        return Err(outside());
    }
    let source = Owner::new(source_layout, group_id, source_owner_rank)?;
    let destination = Owner::new(destination_layout, group_id, destination_owner_rank)?;
    for (&src, &dst) in source_blocks.iter().zip(destination_blocks) {
        if src >= source.bound || dst >= destination.bound {
            return Err(outside());
        }
    }
    Ok(source_blocks
        .iter()
        .zip(destination_blocks)
        .filter_map(|(&src, &dst)| Some((source.local(src)?, destination.local(dst)?)))
        .unzip())
}

/// One end's ownership of a group's scheduler blocks (`owned_local_pages`).
struct Owner {
    shard_count: i64,
    rank: i64,
    /// The exclusive bound of the group's scheduler block ids.
    bound: i64,
}

impl Owner {
    fn new(layout: &Contract, group_id: &str, rank: i64) -> Result<Self, LayoutError> {
        let shard_count = layout.shard_count(group_id)?;
        if rank < 0 || (shard_count > 1 && rank >= shard_count) {
            return Err(LayoutError::new("KVP rank is out of range"));
        }
        Ok(Self {
            shard_count,
            rank,
            bound: layout.virtual_block_count(group_id)?,
        })
    }

    /// The local page of an owned block: the null block is never owned, the
    /// rest are dealt cyclically to the group's owners.
    fn local(&self, block: i64) -> Option<i64> {
        let positive = block - 1;
        let owned =
            block > 0 && (self.shard_count == 1 || positive % self.shard_count == self.rank);
        owned.then_some(positive / self.shard_count + 1)
    }
}
