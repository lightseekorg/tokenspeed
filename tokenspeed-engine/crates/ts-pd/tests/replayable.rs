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

//! Replayable cache windows remain complete across prefix hits and PD transfer planning.

use serde_json::{json, Value};
use ts_pd::layout::{
    contract::{Contract, GroupSpec},
    manifest::{
        build_manifest, build_selection, logical_slots, validate_manifest, BlockTable, BlockTables,
    },
};

fn sliding_group(replayable: bool) -> GroupSpec {
    GroupSpec {
        group_id: "swa".to_owned(),
        attn_kind: "sliding_window".to_owned(),
        rows_per_page: Some(64),
        entry_stride_tokens: Some(1),
        sliding_window_tokens: Some(128),
        replayable,
        transfer_policy: Some("full_suffix".to_owned()),
        shard_count: 1,
        checkpoint_granularity: None,
    }
}

#[test]
fn replayable_windows_transfer_the_whole_retained_tail() {
    for (prefix, prompt, expected) in [
        (0, 320, vec![3, 4]),
        (256, 320, vec![3, 4]),
        (257, 320, vec![3, 4]),
        (257, 258, vec![2, 3, 4]),
        (256, 512, vec![6, 7]),
    ] {
        assert_eq!(
            logical_slots(&sliding_group(true), prefix, prompt, false).unwrap(),
            expected,
            "prefix={prefix}, prompt={prompt}",
        );
    }
}

#[test]
fn cached_windows_still_skip_the_shared_prefix() {
    for prefix in [256, 257] {
        assert_eq!(
            logical_slots(&sliding_group(false), prefix, 320, false).unwrap(),
            vec![4],
        );
    }
}

fn replayable_contract() -> Contract {
    let fixtures: Value =
        serde_json::from_str(include_str!("fixtures/planning/contracts.json")).unwrap();
    let mut wire: Value =
        serde_json::from_str(fixtures["446091a9025dc611"].as_str().unwrap()).unwrap();
    // This captured layout predates replayable windows. Keep its byte geometry
    // and use the current recipe's retention semantics for both private groups.
    for group in wire["group_specs"].as_array_mut().unwrap() {
        if group["attn_kind"] == "sliding_window" {
            group["replayable"] = json!(true);
        }
    }
    Contract::from_wire(&serde_json::to_vec(&wire).unwrap()).unwrap()
}

fn tables(contract: &Contract) -> BlockTables {
    let mut tables: BlockTables = contract
        .group_specs
        .iter()
        .map(|group| {
            let span = group.block_granularity().unwrap();
            let count = ((320 + span - 1) / span) as usize;
            (
                group.group_id.clone(),
                BlockTable::from_rows(&[vec![1; count]], count),
            )
        })
        .collect();
    tables.insert(
        "v41.swa".to_owned(),
        BlockTable::from_rows(&[vec![0, 0, 0, 2, 3]], 5),
    );
    tables
}

#[test]
fn manifests_and_layerwise_selection_include_pages_before_the_hit() {
    let contract = replayable_contract();
    let tables = tables(&contract);
    let swa = contract
        .group_specs
        .iter()
        .position(|group| group.group_id == "v41.swa")
        .unwrap();
    for prefix in [256, 257] {
        let mut manifest = build_manifest(&tables, &contract, 0, prefix, 320, false).unwrap();
        assert_eq!(manifest.groups[swa].block_ids, vec![2, 3]);
        validate_manifest(&manifest, &contract, "destination").unwrap();

        // The final prefill chunk includes its replayed window. Both retained
        // pages must reach the same positions in the destination manifest.
        let selection =
            build_selection(&tables, &contract, 0, prefix, 320, 128, 320, false).unwrap();
        assert_eq!(selection.groups[swa].source_block_ids, vec![2, 3]);
        assert_eq!(selection.groups[swa].destination_positions, vec![0, 1]);

        manifest.groups[swa].block_ids.remove(0);
        assert!(validate_manifest(&manifest, &contract, "destination").is_err());
    }
}

#[test]
fn a_hole_in_the_retained_window_before_the_hit_is_rejected() {
    let contract = replayable_contract();
    let mut tables = tables(&contract);
    tables.insert(
        "v41.swa".to_owned(),
        BlockTable::from_rows(&[vec![0, 0, 0, 0, 3]], 5),
    );
    let error = build_manifest(&tables, &contract, 0, 256, 320, false).unwrap_err();
    assert!(error
        .to_string()
        .contains("logical slot 3 has invalid block ID 0"));
}
