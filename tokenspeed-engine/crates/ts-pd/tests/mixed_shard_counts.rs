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

//! A page-sharded target group beside a replicated head-partitioned one.
//!
//! A block drafter of its own attention family (a DFlash GQA stack) holds K/V
//! pages, which the transfer schema marks head-partitioned; the DSA target
//! holds latent and index-key pages, which it does not partition at all. The
//! planner refuses a head-partitioned field INSIDE a page-sharded group, and
//! both of its refusals are per-group. These tests pin that reading down: a
//! layout whose target group shards while the draft's own group stays
//! replicated plans, and the draft's pages route whole while the target's
//! route from their KVP subgroup.
//!
//! That is the question "does PD support QCP+KVP with a GQA draft" reduces
//! to, and it is answered here rather than on a cluster, because it is a
//! property of the layout contract alone.

use std::collections::BTreeSet;

use serde_json::{json, Value};
use ts_pd::layout::{
    contract::{validate_peer, Contract},
    route::Planner,
};

/// Rows per page of every group here; the planner needs them equal across
/// peers, and one value keeps the fixtures readable.
const ROWS: i64 = 64;
/// DSA latent width in bytes per row (kv_lora_rank 512 + rope 64, bf16).
const LATENT_BYTES: i64 = 1152;
/// Index-key row bytes of the fp8_scaled plane.
const INDEX_BYTES: i64 = 132;

/// A contract with a DSA target group and, optionally, the draft's own K/V
/// group.
///
/// `target_shards` page-shards the target group (KVP); `draft_shards` does
/// the same to the draft's. `kv_heads_per_rank` is the draft's local head
/// count, so a decode side that holds every head passes 8 while a TP8
/// prefill side passes 1 -- the head partition the schema declares.
fn contract(target_shards: i64, draft_shards: Option<i64>, kv_heads_per_rank: i64) -> Contract {
    let mut groups = vec![json!({
        "group_id": "full_attention", "cache_blocks_per_lcm_block": 1, "page_count": 16
    })];
    let mut planes = vec![
        json!({"plane_id": "latent", "bytes_per_lcm_block": ROWS * LATENT_BYTES,
               "arena_offset_bytes": 0}),
        json!({"plane_id": "index_k", "bytes_per_lcm_block": ROWS * INDEX_BYTES,
               "arena_offset_bytes": ROWS * LATENT_BYTES}),
    ];
    // The DSA target's two planes: neither carries a partition, which is why
    // page sharding is open to them.
    let mut fields = vec![
        json!({"group_id": "full_attention", "field_id": "layer.0.latent_kv",
               "plane_id": "latent", "shape": [ROWS, 1, 576], "dtype": "bfloat16",
               "field_offset_bytes": 0, "page_stride_bytes": ROWS * LATENT_BYTES}),
        json!({"group_id": "full_attention", "field_id": "layer.0.index_k",
               "plane_id": "index_k", "shape": [ROWS, INDEX_BYTES], "dtype": "uint8",
               "field_offset_bytes": 0, "page_stride_bytes": ROWS * INDEX_BYTES}),
    ];
    let mut specs = vec![json!({
        "group_id": "full_attention", "attn_kind": "full_history", "rows_per_page": ROWS,
        "entry_stride_tokens": 1, "sliding_window_tokens": null, "replayable": false,
        "transfer_policy": "full_suffix", "shard_count": target_shards
    })];
    let mut schema_fields: Vec<Value> = Vec::new();

    if let Some(draft_shards) = draft_shards {
        let local_bytes = kv_heads_per_rank * 128 * 2;
        let base = ROWS * (LATENT_BYTES + INDEX_BYTES);
        groups.push(json!({
            "group_id": "draft_attention", "cache_blocks_per_lcm_block": 1, "page_count": 16
        }));
        planes.push(json!({
            "plane_id": "draft_kv", "bytes_per_lcm_block": 2 * ROWS * local_bytes,
            "arena_offset_bytes": base
        }));
        for (name, offset) in [("k", 0), ("v", ROWS * local_bytes)] {
            fields.push(json!({
                "group_id": "draft_attention", "field_id": format!("layer.1.{name}"),
                "plane_id": "draft_kv", "shape": [ROWS, kv_heads_per_rank, 128],
                "dtype": "bfloat16", "field_offset_bytes": offset,
                "page_stride_bytes": ROWS * local_bytes
            }));
            // Exactly what _partition_for_field returns for a k/v plane:
            // axis 1 (the head axis), global extent = the model's kv heads.
            schema_fields.push(json!({
                "field_id": format!("layer.1.{name}"),
                "partition": {"axis": 1, "global_extent": 8, "global_parts": []}
            }));
        }
        specs.push(json!({
            "group_id": "draft_attention", "attn_kind": "full_history",
            "rows_per_page": ROWS, "entry_stride_tokens": 1,
            "sliding_window_tokens": null, "replayable": false,
            "transfer_policy": "full_suffix", "shard_count": draft_shards
        }));
    }

    let wire = json!({
        "plan": {
            "prefix_granularity": ROWS,
            "lcm_block_bytes": planes.iter().map(|plane| {
                plane["bytes_per_lcm_block"].as_i64().unwrap()
            }).sum::<i64>(),
            "num_lcm_blocks": 15,
            "groups": groups,
            "planes": planes,
            "fields": fields,
            "resident_block_bytes": null
        },
        "group_specs": specs,
        "transfer_schema": {"fields": schema_fields},
        "field_addresses": null
    });
    Contract::from_wire(wire.to_string().as_bytes()).expect("a well-formed contract")
}

/// Field ids of one decode rank's fragments, by group.
fn routed_groups(planner: &Planner, decode_rank: i64) -> BTreeSet<String> {
    planner
        .plan_for_decode_rank(decode_rank)
        .expect("a planned rank")
        .fragments_by_prefill_rank
        .values()
        .flatten()
        .map(|fragment| fragment.group_id.clone())
        .collect()
}

#[test]
fn a_dsa_target_group_page_shards_on_its_own() {
    // The baseline the draft must not disturb: KVP over a DSA target, whose
    // latent and index-key fields carry no head partition. Shard counts are
    // per side -- a KVP 8 prefill serves a TP 1 decode -- and `validate_peer`
    // compares group semantics, not the counts.
    let prefill = contract(8, None, 0);
    let decode = contract(1, None, 0);
    validate_peer(&prefill, &decode).expect("peers agree");

    let planner = Planner::new(8, 1, &prefill, &decode, None).expect("KVP 8 plans");

    assert_eq!(planner.sharded_group_ids(), vec!["full_attention"]);
}

#[test]
fn a_head_partitioned_field_inside_a_sharded_group_is_refused() {
    // The refusal the DFlash draft hits TODAY, when it shares the target's
    // group: one group id, one shard count, and the draft's k/v inside it.
    let shared = contract(8, Some(8), 1);

    let error = Planner::new(8, 8, &shared, &shared, None)
        .expect_err("a head-partitioned field cannot ride a page-sharded group");

    assert!(
        error.to_string().contains("head-partitioned"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_replicated_draft_group_plans_beside_a_sharded_target_group() {
    // The layout in question: the target's DSA group shards 8 ways on the
    // prefill side while the draft's own group stays replicated. Both
    // planner refusals are per-group, so neither applies -- the
    // head-partitioned fields are in no page-sharded group.
    let prefill = contract(8, Some(1), 1);
    let decode = contract(1, Some(1), 8);
    validate_peer(&prefill, &decode).expect("peers agree");

    let planner = Planner::new(8, 1, &prefill, &decode, None).expect("a mixed layout must plan");

    // Only the target's group shards; the draft's is replicated, so it takes
    // no owner filter.
    assert_eq!(planner.sharded_group_ids(), vec!["full_attention"]);
}

#[test]
fn both_models_pages_reach_the_decode_rank() {
    // "PD walks the target AND the draft": the route must carry both groups,
    // not silently drop the replicated one.
    let prefill = contract(8, Some(1), 1);
    let decode = contract(1, Some(1), 8);
    let planner = Planner::new(8, 1, &prefill, &decode, None).expect("a mixed layout plans");

    let groups = routed_groups(&planner, 0);

    assert!(
        groups.contains("full_attention"),
        "target pages: {groups:?}"
    );
    assert!(
        groups.contains("draft_attention"),
        "draft pages: {groups:?}"
    );
}

#[test]
fn the_draft_pages_are_gathered_from_every_prefill_rank() {
    // The draft's heads are split across the prefill ranks (TP8 -> 1 head
    // each) and the decode rank holds all eight, so every prefill rank must
    // contribute its head slice. This is the head partition doing its job,
    // which page sharding would have made impossible.
    let prefill = contract(8, Some(1), 1);
    let decode = contract(1, Some(1), 8);
    let planner = Planner::new(8, 1, &prefill, &decode, None).expect("a mixed layout plans");

    let plan = planner.plan_for_decode_rank(0).expect("a planned rank");
    let contributors: BTreeSet<i64> = plan
        .fragments_by_prefill_rank
        .iter()
        .filter(|(_, fragments)| {
            fragments
                .iter()
                .any(|fragment| fragment.group_id == "draft_attention")
        })
        .map(|(&rank, _)| rank)
        .collect();

    assert_eq!(contributors, (0..8).collect::<BTreeSet<_>>());
    // Each rank's slice lands at its own head offset, so the eight slices
    // tile the decode page instead of overwriting one another.
    let offsets: BTreeSet<i64> = plan
        .fragments_by_prefill_rank
        .values()
        .flatten()
        .filter(|fragment| fragment.field_id == "layer.1.k")
        .map(|fragment| fragment.dst_byte_offset)
        .collect();
    assert_eq!(
        offsets.len(),
        8,
        "head slices must not collide: {offsets:?}"
    );
}

#[test]
fn the_target_pages_still_come_from_their_kvp_subgroup() {
    // Sharding the target group must keep meaning what it meant without a
    // draft: each scheduler block is sent by the one rank of the KVP
    // subgroup that owns it.
    let prefill = contract(8, Some(1), 1);
    let decode = contract(1, Some(1), 8);
    let planner = Planner::new(8, 1, &prefill, &decode, None).expect("a mixed layout plans");

    let plan = planner.plan_for_decode_rank(0).expect("a planned rank");
    for (&rank, filters) in &plan.owner_filters_by_prefill_rank {
        let target = filters
            .get("full_attention")
            .expect("the sharded target group takes a decision")
            .expect("a rank routing the target group owns a share of it");
        assert_eq!(target.owner_count, 8);
        assert_eq!(target.owner_rank, rank % 8);
        // The replicated draft group is not a sharded group at all, so it
        // carries no decision.
        assert!(filters.get("draft_attention").is_none());
    }
}

#[test]
fn a_sharded_draft_group_is_still_refused_on_its_own() {
    // The mixed layout is allowed because the draft's group is replicated,
    // not because the draft is a draft: sharding ITS group brings the same
    // refusal back, so the shard count is what the rule tracks.
    let sharded_draft = contract(1, Some(8), 1);

    let error = Planner::new(8, 8, &sharded_draft, &sharded_draft, None)
        .expect_err("page-sharding the draft's own group is refused too");

    assert!(
        error.to_string().contains("head-partitioned"),
        "unexpected refusal: {error}"
    );
}
