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

//! DeepSeek's expert-parallelism load balancer (`moe/eplb_algorithms/
//! deepseek.py`): replicate the loaded experts and pack the replicas onto
//! the GPUs, hierarchically over nodes when the expert groups divide them.
//!
//! The loads are integer route counts held as doubles, so every sum is
//! exact and the result matches the Python implementation's for equal
//! inputs: sorts are stable, ties go to the lowest index.

/// The algorithm variant (`EplbAlgorithm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Replicate and pack over all GPUs as one tier.
    Deepseek,
    /// Pack expert groups onto nodes first, then replicas onto each node's
    /// GPUs.
    DeepseekHierarchical,
}

impl Algorithm {
    /// `compute_algorithm`: `--eplb-algorithm`, with `auto` hierarchical
    /// when the model's expert groups divide over the nodes.
    pub fn resolve(raw: &str, num_groups: Option<i64>, num_nodes: i64) -> Result<Self, String> {
        match raw {
            "deepseek" => Ok(Self::Deepseek),
            "deepseek_hierarchical" => Ok(Self::DeepseekHierarchical),
            "auto" => Ok(match num_groups {
                Some(groups) if groups % num_nodes == 0 => Self::DeepseekHierarchical,
                _ => Self::Deepseek,
            }),
            other => Err(format!("unknown EPLB algorithm {other:?}")),
        }
    }
}

/// Indices of `row` by descending value, ties in index order (a stable
/// descending sort).
fn sorted_descending(row: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..row.len()).collect();
    indices.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
    indices
}

/// Per row, an index for each item.
type Indices = Vec<Vec<usize>>;

/// `balanced_packing`: per row, put `n` weighted items into `num_packs`
/// packs of `n / num_packs` items, heaviest first into the lightest pack
/// with room. Returns each item's pack and its rank in the pack.
pub fn balanced_packing(
    weight: &[Vec<f64>],
    num_packs: usize,
) -> Result<(Indices, Indices), String> {
    let num_groups = weight.first().map_or(0, Vec::len);
    if num_packs == 0 || num_groups % num_packs != 0 {
        return Err(format!(
            "num_groups={num_groups} must be divisible by num_packs={num_packs}."
        ));
    }
    let per_pack = num_groups / num_packs;
    if per_pack == 1 {
        let identity: Vec<usize> = (0..num_groups).collect();
        return Ok((
            vec![identity; weight.len()],
            vec![vec![0; num_groups]; weight.len()],
        ));
    }
    let mut pack_index = vec![vec![0; num_groups]; weight.len()];
    let mut rank_in_pack = vec![vec![0; num_groups]; weight.len()];
    for (i, row) in weight.iter().enumerate() {
        let mut pack_weights = vec![0.0f64; num_packs];
        let mut pack_items = vec![0usize; num_packs];
        for group in sorted_descending(row) {
            // The first of the lightest packs with room.
            let mut best: Option<usize> = None;
            for pack in 0..num_packs {
                if pack_items[pack] < per_pack
                    && best.is_none_or(|b| pack_weights[pack] < pack_weights[b])
                {
                    best = Some(pack);
                }
            }
            let pack = best.ok_or("balanced_packing selected a full pack.")?;
            pack_index[i][group] = pack;
            rank_in_pack[i][group] = pack_items[pack];
            pack_weights[pack] += row[group];
            pack_items[pack] += 1;
        }
    }
    Ok((pack_index, rank_in_pack))
}

/// `replicate_experts`: per row, give the `num_phy - num_log` extra slots one
/// at a time to the expert with the highest load per replica. Returns each
/// physical slot's logical expert, its replica rank, and each expert's
/// replica count.
pub fn replicate_experts(
    weight: &[Vec<f64>],
    num_phy: usize,
) -> Result<(Indices, Indices, Indices), String> {
    let num_log = weight.first().map_or(0, Vec::len);
    if num_phy < num_log {
        return Err(format!(
            "num_phy={num_phy} must be greater than or equal to num_log={num_log}."
        ));
    }
    let mut phy2log = Vec::with_capacity(weight.len());
    let mut rank = Vec::with_capacity(weight.len());
    let mut logcnt = Vec::with_capacity(weight.len());
    for row in weight {
        let mut p2l: Vec<usize> = (0..num_phy).collect();
        let mut r = vec![0usize; num_phy];
        let mut count = vec![1usize; num_log];
        for slot in num_log..num_phy {
            let mut best = 0usize;
            let mut best_load = f64::NEG_INFINITY;
            for (expert, (&load, &replicas)) in row.iter().zip(&count).enumerate() {
                let per_replica = load / replicas as f64;
                if per_replica > best_load {
                    best_load = per_replica;
                    best = expert;
                }
            }
            p2l[slot] = best;
            r[slot] = count[best];
            count[best] += 1;
        }
        phy2log.push(p2l);
        rank.push(r);
        logcnt.push(count);
    }
    Ok((phy2log, rank, logcnt))
}

fn inverse(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (index, &value) in perm.iter().enumerate() {
        inverse[value] = index;
    }
    inverse
}

/// `rebalance_experts_hierarchical`: the physical-to-logical map
/// `[layers, num_physical]` for `weight` `[layers, num_logical]`.
pub fn rebalance_experts_hierarchical(
    weight: &[Vec<f64>],
    num_physical: usize,
    num_groups: usize,
    num_nodes: usize,
    num_gpus: usize,
) -> Result<Vec<Vec<usize>>, String> {
    let num_layers = weight.len();
    let num_logical = weight.first().map_or(0, Vec::len);
    if num_groups == 0 || num_logical % num_groups != 0 {
        return Err(format!(
            "num_logical_experts={num_logical} must be divisible by num_groups={num_groups}."
        ));
    }
    let group_size = num_logical / num_groups;
    if num_nodes == 0 || num_groups % num_nodes != 0 {
        return Err(format!(
            "num_groups={num_groups} must be divisible by num_nodes={num_nodes}."
        ));
    }
    let groups_per_node = num_groups / num_nodes;
    if num_gpus == 0 || num_gpus % num_nodes != 0 {
        return Err(format!(
            "num_gpus={num_gpus} must be divisible by num_nodes={num_nodes}."
        ));
    }
    if num_physical % num_gpus != 0 {
        return Err(format!(
            "num_physical_experts={num_physical} must be divisible by num_gpus={num_gpus}."
        ));
    }
    let phy_per_gpu = num_physical / num_gpus;
    let log_per_node = num_logical / num_nodes;
    let phy_per_node = num_physical / num_nodes;

    // Step 1: expert groups onto nodes.
    let tokens_per_group: Vec<Vec<f64>> = weight
        .iter()
        .map(|row| {
            row.chunks(group_size)
                .map(|chunk| chunk.iter().sum())
                .collect()
        })
        .collect();
    let (group_pack, group_rank) = balanced_packing(&tokens_per_group, num_nodes)?;
    let log2mlog: Vec<Vec<usize>> = (0..num_layers)
        .map(|l| {
            (0..num_logical)
                .map(|e| {
                    let g = e / group_size;
                    (group_pack[l][g] * groups_per_node + group_rank[l][g]) * group_size
                        + e % group_size
                })
                .collect()
        })
        .collect();
    let mlog2log: Vec<Vec<usize>> = log2mlog.iter().map(|row| inverse(row)).collect();

    // Step 2: replicate within each node; rows are (layer, node).
    let tokens_per_mlog: Vec<Vec<f64>> = (0..num_layers)
        .flat_map(|l| {
            let weight = &weight[l];
            let mlog2log = &mlog2log[l];
            (0..num_nodes).map(move |n| {
                (0..log_per_node)
                    .map(|j| weight[mlog2log[n * log_per_node + j]])
                    .collect()
            })
        })
        .collect();
    let (phy2mlog, _, mlogcnt) = replicate_experts(&tokens_per_mlog, phy_per_node)?;

    // Step 3: replicas onto the node's GPUs.
    let tokens_per_phy: Vec<Vec<f64>> = tokens_per_mlog
        .iter()
        .zip(&phy2mlog)
        .zip(&mlogcnt)
        .map(|((tokens, p2m), counts)| p2m.iter().map(|&m| tokens[m] / counts[m] as f64).collect())
        .collect();
    let (pack, rank) = balanced_packing(&tokens_per_phy, num_gpus / num_nodes)?;
    let mut pphy2log = vec![vec![0usize; num_physical]; num_layers];
    for l in 0..num_layers {
        for n in 0..num_nodes {
            let row = l * num_nodes + n;
            let phy2pphy: Vec<usize> = pack[row]
                .iter()
                .zip(&rank[row])
                .map(|(&p, &r)| p * phy_per_gpu + r)
                .collect();
            let pphy2phy = inverse(&phy2pphy);
            for (pphy, &phy) in pphy2phy.iter().enumerate() {
                let mlog = phy2mlog[row][phy] + n * log_per_node;
                pphy2log[l][n * phy_per_node + pphy] = mlog2log[l][mlog];
            }
        }
    }
    Ok(pphy2log)
}

/// `rebalance_experts`: the placement for one load window,
/// `[layers, num_replicas]`.
pub fn rebalance_experts(
    weight: &[Vec<f64>],
    num_replicas: usize,
    num_groups: Option<usize>,
    num_nodes: usize,
    num_gpus: usize,
    algorithm: Algorithm,
) -> Result<Vec<Vec<usize>>, String> {
    match algorithm {
        Algorithm::DeepseekHierarchical => {
            let groups = num_groups.ok_or("the hierarchical EPLB algorithm needs expert groups")?;
            rebalance_experts_hierarchical(weight, num_replicas, groups, num_nodes, num_gpus)
        }
        Algorithm::Deepseek => rebalance_experts_hierarchical(weight, num_replicas, 1, 1, num_gpus),
    }
}
