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

//! Online expert rebalancing (`--enable-eplb`): the controller that decides
//! when a rebalance runs, the placement solver, and the move plan.
//!
//! A rebalance is a sequence of internal operations the loop completes
//! through the same-round gate the weight operations use, so every rank runs
//! each step in the same round:
//!
//! 1. Snapshot: read the route counters since the previous snapshot, reduce
//!    them over the EP group, and start deriving a new placement (EP rank 0,
//!    on a worker thread).
//! 2. Commit, a fixed number of forwards later: rank 0's placement is
//!    broadcast over the EP group and every rank plans the same slot moves.
//! 3. One chunk of layers per round: the weights move and the layers'
//!    routing tables switch.

mod algorithm;
mod moves;

use std::{collections::VecDeque, thread};

pub use algorithm::{rebalance_experts, Algorithm};
pub use moves::{plan_slot_moves, SlotMoves};

/// Forwards between a snapshot and its commit (`COMMIT_DELAY_FORWARDS`).
pub const COMMIT_DELAY_FORWARDS: u64 = 200;

/// The placement geometry and this rank's place in it
/// (`ExpertRebalanceSpecs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EplbSpecs {
    /// MoE layers: rows of the placement.
    pub num_layers: usize,
    /// Routed experts.
    pub num_logical_experts: usize,
    /// Slots per layer.
    pub num_physical_experts: usize,
    /// EP ranks the slots spread over.
    pub ep_size: usize,
    /// This rank's EP rank.
    pub ep_rank: usize,
    /// Every EP rank's node.
    pub ep_rank_nodes: Vec<i64>,
    /// Each rank counted only its own tokens' routes: the group's load is
    /// their sum. Otherwise every rank counted every token and the counters
    /// must agree.
    pub all_to_all_ep: bool,
    /// The model's expert groups, if it has them.
    pub num_groups: Option<usize>,
    /// Nodes the EP ranks span.
    pub num_nodes: usize,
}

impl EplbSpecs {
    /// Slots per rank.
    pub fn num_local(&self) -> usize {
        self.num_physical_experts / self.ep_size
    }
}

/// The controller's settings.
#[derive(Debug, Clone)]
pub struct EplbConfig {
    /// The geometry.
    pub specs: EplbSpecs,
    /// Forwards between snapshots (`--eplb-rebalance-num-iterations`).
    pub rebalance_num_iterations: u64,
    /// Layers switched per round (`--eplb-rebalance-layers-per-chunk`).
    pub layers_per_chunk: usize,
    /// The solver variant.
    pub algorithm: Algorithm,
    /// Forwards between a snapshot and its commit.
    pub commit_delay_forwards: u64,
}

/// One step of a rebalance, completed through the same-round gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InternalOp {
    /// Read and reset the route counters; start the solver.
    Snapshot,
    /// Agree on the new placement and plan the moves.
    Commit,
    /// Move these layers' experts.
    ApplyChunk(Vec<usize>),
}

impl InternalOp {
    /// Its code on the gate.
    pub fn gate_code(&self) -> i64 {
        match self {
            InternalOp::Snapshot => 1,
            InternalOp::Commit => 2,
            InternalOp::ApplyChunk(_) => 3,
        }
    }
}

/// Where a rebalance stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No rebalance in progress.
    Idle,
    /// A snapshot is queued or taken; the placement is being derived.
    Computing,
    /// The commit and chunks are queued.
    Applying,
}

impl Phase {
    /// Its name in messages.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Computing => "computing",
            Phase::Applying => "applying",
        }
    }
}

/// `chunk_layer_ids`: `0..num_layers` in runs of `per_chunk`.
pub fn chunk_layer_ids(num_layers: usize, per_chunk: usize) -> Result<Vec<Vec<usize>>, String> {
    if per_chunk == 0 || per_chunk > num_layers {
        return Err(format!(
            "layers_per_chunk={per_chunk} must be in [1, {num_layers}]"
        ));
    }
    Ok((0..num_layers)
        .step_by(per_chunk)
        .map(|start| (start..(start + per_chunk).min(num_layers)).collect())
        .collect())
}

/// Per-layer mean rank load over the busiest rank's (`load_balancedness`),
/// from per-slot loads `[layers, physical]`; ranks own contiguous slots.
fn balancedness(per_slot: &[Vec<f64>], ep_size: usize) -> Vec<f64> {
    per_slot
        .iter()
        .map(|row| {
            let per_rank: Vec<f64> = row
                .chunks(row.len() / ep_size)
                .map(|chunk| chunk.iter().sum())
                .collect();
            let max = per_rank.iter().copied().fold(0.0f64, f64::max).max(1.0);
            per_rank.iter().sum::<f64>() / per_rank.len() as f64 / max
        })
        .collect()
}

fn summary(values: &[f64]) -> (f64, f64) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    (min, mean)
}

/// The solver's work: the placement for a logical load.
fn solve(
    logical_count: Vec<Vec<i64>>,
    specs: EplbSpecs,
    algorithm: Algorithm,
) -> Result<Vec<Vec<i32>>, String> {
    let weight: Vec<Vec<f64>> = logical_count
        .iter()
        .map(|row| row.iter().map(|&count| count as f64).collect())
        .collect();
    let map = rebalance_experts(
        &weight,
        specs.num_physical_experts,
        specs.num_groups,
        specs.num_nodes,
        specs.ep_size,
        algorithm,
    )?;
    Ok(map
        .into_iter()
        .map(|row| row.into_iter().map(|e| e as i32).collect())
        .collect())
}

/// Decides when a rebalance starts and what each step carries. It never
/// touches the device or the
/// collectives; the loop completes the steps and feeds the results back.
pub struct Controller {
    config: EplbConfig,
    chunks: Vec<Vec<usize>>,
    phase: Phase,
    /// Forwards this rank ran (real or idle), rank-identical.
    forwards: u64,
    next_snapshot_at: u64,
    commit_at: Option<u64>,
    /// EP rank 0's solver.
    solver: Option<thread::JoinHandle<Result<Vec<Vec<i32>>, String>>>,
    old_map: Option<Vec<Vec<i32>>>,
    new_map: Option<Vec<Vec<i32>>>,
    logical_count: Option<Vec<Vec<i64>>>,
    moves_by_layer: Vec<SlotMoves>,
    pending_chunks: VecDeque<Vec<usize>>,
    rebalances_completed: u64,
}

impl Controller {
    /// A controller with `config`.
    pub fn new(config: EplbConfig) -> Result<Self, String> {
        if config.rebalance_num_iterations == 0 {
            return Err("rebalance_num_iterations must be positive".to_owned());
        }
        let chunks = chunk_layer_ids(config.specs.num_layers, config.layers_per_chunk)?;
        Ok(Self {
            next_snapshot_at: config.rebalance_num_iterations,
            config,
            chunks,
            phase: Phase::Idle,
            forwards: 0,
            commit_at: None,
            solver: None,
            old_map: None,
            new_map: None,
            logical_count: None,
            moves_by_layer: Vec::new(),
            pending_chunks: VecDeque::new(),
            rebalances_completed: 0,
        })
    }

    /// The geometry.
    pub fn specs(&self) -> &EplbSpecs {
        &self.config.specs
    }

    /// Where the rebalance stands.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Forwards between a snapshot and its commit.
    pub fn commit_delay_forwards(&self) -> u64 {
        self.config.commit_delay_forwards
    }

    /// Count a round's forward; the steps due now, to queue in order.
    pub fn note_round(&mut self, forwarded: bool) -> Vec<InternalOp> {
        if forwarded {
            self.forwards += 1;
        }
        match self.phase {
            Phase::Idle if self.forwards >= self.next_snapshot_at => vec![self.begin_snapshot()],
            Phase::Computing if self.commit_at.is_some_and(|at| self.forwards >= at) => {
                self.commit_at = None;
                std::iter::once(InternalOp::Commit)
                    .chain(self.chunks.iter().cloned().map(InternalOp::ApplyChunk))
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// A manual trigger: the snapshot step, or `None` while a rebalance is
    /// in progress.
    pub fn request_snapshot(&mut self) -> Option<InternalOp> {
        (self.phase == Phase::Idle).then(|| self.begin_snapshot())
    }

    fn begin_snapshot(&mut self) -> InternalOp {
        self.phase = Phase::Computing;
        self.next_snapshot_at = self.forwards + self.config.rebalance_num_iterations;
        InternalOp::Snapshot
    }

    /// Take the EP group's load `[layers, physical]`, counted under the
    /// placement `old_map`; rank 0 starts the solver; the commit is due
    /// after the delay.
    pub fn on_snapshot(
        &mut self,
        physical_count: Vec<Vec<i64>>,
        old_map: Vec<Vec<i32>>,
    ) -> Result<(), String> {
        if self.phase != Phase::Computing || self.commit_at.is_some() {
            return Err(format!(
                "unexpected expert load snapshot in phase {}",
                self.phase.name()
            ));
        }
        let specs = &self.config.specs;
        let shape_ok = physical_count.len() == specs.num_layers
            && physical_count
                .iter()
                .all(|row| row.len() == specs.num_physical_experts);
        if !shape_ok || old_map.len() != specs.num_layers {
            return Err(format!(
                "expert load has the wrong shape, expected [{}, {}]",
                specs.num_layers, specs.num_physical_experts
            ));
        }
        let mut logical = vec![vec![0i64; specs.num_logical_experts]; specs.num_layers];
        for (layer, (counts, map)) in physical_count.iter().zip(&old_map).enumerate() {
            for (&count, &expert) in counts.iter().zip(map) {
                logical[layer][expert as usize] += count;
            }
        }
        let routes: i64 = physical_count.iter().flatten().sum();
        let loads: Vec<Vec<f64>> = physical_count
            .iter()
            .map(|row| row.iter().map(|&count| count as f64).collect())
            .collect();
        let (min, mean) = summary(&balancedness(&loads, specs.ep_size));
        tracing::info!(
            "Expert rebalance: snapshot of {routes} routes over {} layers; balancedness min \
             {min:.3} mean {mean:.3}; commit in {} forwards",
            specs.num_layers,
            self.config.commit_delay_forwards
        );
        if specs.ep_rank == 0 {
            let (count, specs, algorithm) = (logical.clone(), specs.clone(), self.config.algorithm);
            self.solver = Some(
                thread::Builder::new()
                    .name("ts-eplb-solver".to_owned())
                    .spawn(move || solve(count, specs, algorithm))
                    .map_err(|error| error.to_string())?,
            );
        }
        self.old_map = Some(old_map);
        self.logical_count = Some(logical);
        self.commit_at = Some(self.forwards + self.config.commit_delay_forwards);
        Ok(())
    }

    /// EP rank 0's placement to broadcast, once its solver finished; `None`
    /// on the other ranks, which take the broadcast.
    pub fn placement_for_commit(&mut self) -> Result<Option<Vec<Vec<i32>>>, String> {
        if self.phase != Phase::Computing || self.old_map.is_none() {
            return Err(format!(
                "unexpected expert placement commit in phase {}",
                self.phase.name()
            ));
        }
        match self.solver.take() {
            Some(solver) => solver
                .join()
                .map_err(|_| "the EPLB solver panicked".to_owned())?
                .map(Some),
            None => Ok(None),
        }
    }

    /// Plan every layer's moves from the agreed placement; the chunks follow.
    pub fn on_commit(&mut self, new_map: Vec<Vec<i32>>) -> Result<(), String> {
        let Some(old_map) = self
            .old_map
            .as_ref()
            .filter(|_| self.phase == Phase::Computing)
        else {
            return Err(format!(
                "unexpected expert placement commit in phase {}",
                self.phase.name()
            ));
        };
        let specs = &self.config.specs;
        if new_map.len() != old_map.len()
            || new_map
                .iter()
                .any(|row| row.len() != specs.num_physical_experts)
        {
            return Err("the committed placement has the wrong shape".to_owned());
        }
        self.moves_by_layer = old_map
            .iter()
            .zip(&new_map)
            .map(|(old, new)| {
                plan_slot_moves(
                    old,
                    new,
                    specs.ep_rank,
                    &specs.ep_rank_nodes,
                    specs.num_local(),
                )
            })
            .collect::<Result<_, _>>()?;
        if let Some(logical) = &self.logical_count {
            // Each expert's load split evenly over its replicas.
            let per_slot: Vec<Vec<f64>> = new_map
                .iter()
                .zip(logical)
                .map(|(row, counts)| {
                    let mut replicas = vec![0i64; counts.len()];
                    for &e in row {
                        replicas[e as usize] += 1;
                    }
                    row.iter()
                        .map(|&e| counts[e as usize] as f64 / replicas[e as usize].max(1) as f64)
                        .collect()
                })
                .collect();
            let (min, mean) = summary(&balancedness(&per_slot, specs.ep_size));
            let moved: usize = self
                .moves_by_layer
                .iter()
                .map(SlotMoves::num_changed_slots)
                .sum();
            tracing::info!(
                "Expert rebalance: committed placement moves {moved} local slots; expected \
                 balancedness min {min:.3} mean {mean:.3}"
            );
        }
        self.new_map = Some(new_map);
        self.pending_chunks = self.chunks.iter().cloned().collect();
        self.phase = Phase::Applying;
        Ok(())
    }

    /// The new rows and this rank's moves of the next chunk, which must be
    /// `layer_ids`.
    pub fn chunk_payload(
        &self,
        layer_ids: &[usize],
    ) -> Result<(Vec<Vec<i32>>, Vec<SlotMoves>), String> {
        let Some(new_map) = self
            .new_map
            .as_ref()
            .filter(|_| self.phase == Phase::Applying)
        else {
            return Err(format!(
                "unexpected expert placement chunk in phase {}",
                self.phase.name()
            ));
        };
        if self.pending_chunks.front().map(Vec::as_slice) != Some(layer_ids) {
            return Err(format!(
                "chunk {layer_ids:?} is out of order; expected {:?}",
                self.pending_chunks.front()
            ));
        }
        Ok((
            layer_ids
                .iter()
                .map(|&layer| new_map[layer].clone())
                .collect(),
            layer_ids
                .iter()
                .map(|&layer| self.moves_by_layer[layer].clone())
                .collect(),
        ))
    }

    /// A chunk landed; the last completes the rebalance.
    pub fn on_chunk_applied(&mut self, layer_ids: &[usize]) -> Result<(), String> {
        if self.pending_chunks.front().map(Vec::as_slice) != Some(layer_ids) {
            return Err(format!("chunk {layer_ids:?} was not the pending chunk"));
        }
        self.pending_chunks.pop_front();
        if !self.pending_chunks.is_empty() {
            return Ok(());
        }
        self.phase = Phase::Idle;
        self.old_map = None;
        self.new_map = None;
        self.logical_count = None;
        self.moves_by_layer.clear();
        self.rebalances_completed += 1;
        tracing::info!("Expert rebalance {} applied", self.rebalances_completed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three layers, four experts on six slots over two ranks, snapshots
    /// every 5 forwards, chunks of 2 layers, commits 3 forwards after their
    /// snapshot.
    fn controller(ep_rank: usize) -> Controller {
        Controller::new(EplbConfig {
            specs: EplbSpecs {
                num_layers: 3,
                num_logical_experts: 4,
                num_physical_experts: 6,
                ep_size: 2,
                ep_rank,
                ep_rank_nodes: vec![0, 0],
                all_to_all_ep: false,
                num_groups: None,
                num_nodes: 1,
            },
            rebalance_num_iterations: 5,
            layers_per_chunk: 2,
            algorithm: Algorithm::Deepseek,
            commit_delay_forwards: 3,
        })
        .unwrap()
    }

    fn rounds(controller: &mut Controller, n: usize, forwarded: bool) -> Vec<InternalOp> {
        (0..n)
            .flat_map(|_| controller.note_round(forwarded))
            .collect()
    }

    fn trivial_map() -> Vec<Vec<i32>> {
        vec![vec![0, 1, 2, 3, 0, 1]; 3]
    }

    /// A load under the trivial placement whose expert 3, in one slot, is hot.
    fn snapshot(controller: &mut Controller) -> Result<(), String> {
        let load = vec![vec![1, 1, 1, 40, 1, 1]; 3];
        controller.on_snapshot(load, trivial_map())
    }

    fn apply_all(controller: &mut Controller) {
        for ids in [vec![0, 1], vec![2]] {
            controller.chunk_payload(&ids).unwrap();
            controller.on_chunk_applied(&ids).unwrap();
        }
    }

    #[test]
    fn a_snapshot_is_due_after_n_forwards_and_its_commit_k_later() {
        let mut controller = controller(0);
        assert_eq!(rounds(&mut controller, 4, true), []);
        // Rounds without a forward do not count.
        assert_eq!(rounds(&mut controller, 10, false), []);
        assert_eq!(controller.note_round(true), [InternalOp::Snapshot]);
        assert_eq!(controller.phase(), Phase::Computing);
        // Until the snapshot completes nothing else is due.
        assert_eq!(rounds(&mut controller, 2, true), []);
        snapshot(&mut controller).unwrap();
        assert_eq!(rounds(&mut controller, 2, true), []);
        assert_eq!(
            controller.note_round(true),
            [
                InternalOp::Commit,
                InternalOp::ApplyChunk(vec![0, 1]),
                InternalOp::ApplyChunk(vec![2]),
            ]
        );
        // Computing until the commit lands; nothing is due twice.
        assert_eq!(controller.phase(), Phase::Computing);
        assert_eq!(rounds(&mut controller, 3, true), []);
    }

    #[test]
    fn a_commit_plans_moves_and_chunks_complete_in_order() {
        let mut controller = controller(0);
        rounds(&mut controller, 5, true);
        snapshot(&mut controller).unwrap();
        rounds(&mut controller, 3, true);
        let new_map = controller.placement_for_commit().unwrap().unwrap();
        // Both redundant slots replicate the hot expert.
        for row in &new_map {
            assert_eq!(row.iter().filter(|&&e| e == 3).count(), 3, "{new_map:?}");
        }
        controller.on_commit(new_map.clone()).unwrap();
        assert_eq!(controller.phase(), Phase::Applying);
        let out_of_order = controller.chunk_payload(&[2]).unwrap_err();
        assert!(out_of_order.contains("out of order"), "{out_of_order}");
        let (rows, moves) = controller.chunk_payload(&[0, 1]).unwrap();
        assert_eq!(rows, new_map[..2]);
        assert!(moves.iter().all(|m| m.num_changed_slots() > 0));
        controller.on_chunk_applied(&[0, 1]).unwrap();
        assert_eq!(controller.phase(), Phase::Applying);
        let (rows, _) = controller.chunk_payload(&[2]).unwrap();
        assert_eq!(rows, new_map[2..]);
        controller.on_chunk_applied(&[2]).unwrap();
        assert_eq!(controller.phase(), Phase::Idle);
        // The cadence counts from the snapshot (forward 5): the next one is
        // due at forward 10, two rounds after the 8 forwards so far.
        assert_eq!(controller.note_round(true), []);
        assert_eq!(controller.note_round(true), [InternalOp::Snapshot]);
    }

    #[test]
    fn other_ranks_take_the_broadcast_placement() {
        let mut controller = controller(1);
        rounds(&mut controller, 5, true);
        snapshot(&mut controller).unwrap();
        assert_eq!(controller.placement_for_commit().unwrap(), None);
        let agreed = vec![vec![3, 1, 2, 3, 0, 1]; 3];
        controller.on_commit(agreed.clone()).unwrap();
        let (rows, moves) = controller.chunk_payload(&[0, 1]).unwrap();
        assert_eq!(rows, agreed[..2]);
        // Rank 1 keeps [3, 0, 1]; rank 0's slot 0 turns from 0 into 3, which
        // only rank 1 holds: rank 1 sends its slot 0 to rank 0.
        assert_eq!(moves[0].recv, []);
        assert_eq!(moves[0].send, [(0, 0)]);
    }

    #[test]
    fn a_manual_trigger_runs_only_while_idle_and_steps_are_phase_checked() {
        let mut controller = controller(0);
        assert_eq!(controller.request_snapshot(), Some(InternalOp::Snapshot));
        assert_eq!(controller.request_snapshot(), None);
        assert!(controller.placement_for_commit().is_err());
        assert!(controller.chunk_payload(&[0, 1]).is_err());
        let wrong_shape = controller
            .on_snapshot(vec![vec![1; 6]; 2], trivial_map()[..2].to_vec())
            .unwrap_err();
        assert!(wrong_shape.contains("shape"), "{wrong_shape}");
        snapshot(&mut controller).unwrap();
        assert!(snapshot(&mut controller).is_err());
        // The commit follows the snapshot by 3 forwards, and the manual
        // trigger restarted the cadence from its own round (forward 0).
        assert_eq!(rounds(&mut controller, 2, true), []);
        assert_eq!(controller.note_round(true)[0], InternalOp::Commit);
        let new_map = controller.placement_for_commit().unwrap().unwrap();
        controller.on_commit(new_map).unwrap();
        apply_all(&mut controller);
        assert_eq!(controller.phase(), Phase::Idle);
        assert_eq!(controller.note_round(true), []);
        assert_eq!(controller.note_round(true), [InternalOp::Snapshot]);
    }

    #[test]
    fn layers_split_into_chunks_of_the_configured_size() {
        assert_eq!(
            chunk_layer_ids(5, 2).unwrap(),
            [vec![0, 1], vec![2, 3], vec![4]]
        );
        assert_eq!(chunk_layer_ids(3, 3).unwrap(), [vec![0, 1, 2]]);
        assert!(chunk_layer_ids(3, 4).is_err());
        assert!(chunk_layer_ids(3, 0).is_err());
    }

    #[test]
    fn the_settings_are_validated() {
        let mut config = controller(0).config;
        config.rebalance_num_iterations = 0;
        assert!(Controller::new(config.clone())
            .err()
            .unwrap()
            .contains("positive"));
        config.rebalance_num_iterations = 5;
        config.layers_per_chunk = 4;
        assert!(Controller::new(config)
            .err()
            .unwrap()
            .contains("layers_per_chunk"));
    }
}
