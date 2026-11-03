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

//! The EPLB solver and move planner against the Python ones
//! (`tests/fixtures/eplb.json`, from `compute_placement_maps` and
//! `plan_slot_moves`).

use serde_json::Value;
use ts_control::eplb::{plan_slot_moves, rebalance_experts, Algorithm};

fn rows<T: TryFrom<i64>>(value: &Value) -> Vec<Vec<T>>
where
    T::Error: std::fmt::Debug,
{
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| T::try_from(v.as_i64().unwrap()).unwrap())
                .collect()
        })
        .collect()
}

fn pairs(value: &Value) -> Vec<(usize, usize)> {
    rows::<usize>(value)
        .into_iter()
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

#[test]
fn placements_and_moves_match_the_python_eplb() {
    let path = format!("{}/tests/fixtures/eplb.json", env!("CARGO_MANIFEST_DIR"));
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let load: Vec<Vec<f64>> = rows::<i64>(&case["load"])
            .into_iter()
            .map(|row| row.into_iter().map(|c| c as f64).collect())
            .collect();
        let ep_size = case["ep_size"].as_u64().unwrap() as usize;
        let num_physical = case["num_physical"].as_u64().unwrap() as usize;
        let algorithm = Algorithm::resolve(case["algorithm"].as_str().unwrap(), None, 1).unwrap();
        let new_map = rebalance_experts(
            &load,
            num_physical,
            case["num_groups"].as_u64().map(|g| g as usize),
            case["num_nodes"].as_u64().unwrap() as usize,
            ep_size,
            algorithm,
        )
        .unwrap();
        let expected_map: Vec<Vec<usize>> = rows(&case["new_map"]);
        assert_eq!(new_map, expected_map, "{}", case["algorithm"]);

        let old_map: Vec<Vec<i32>> = rows(&case["old_map"]);
        let nodes: Vec<i64> = case["ep_rank_nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap())
            .collect();
        let new_map: Vec<Vec<i32>> = rows(&case["new_map"]);
        for (rank, layers) in case["moves"].as_array().unwrap().iter().enumerate() {
            for (layer, expected) in layers.as_array().unwrap().iter().enumerate() {
                let moves = plan_slot_moves(
                    &old_map[layer],
                    &new_map[layer],
                    rank,
                    &nodes,
                    num_physical / ep_size,
                )
                .unwrap();
                assert_eq!(
                    moves.recv,
                    pairs(&expected["recv"]),
                    "rank {rank} layer {layer}"
                );
                assert_eq!(
                    moves.send,
                    pairs(&expected["send"]),
                    "rank {rank} layer {layer}"
                );
                assert_eq!(moves.local_copy, pairs(&expected["local_copy"]));
                assert_eq!(moves.free_rider, pairs(&expected["free_rider"]));
            }
        }
    }
}
