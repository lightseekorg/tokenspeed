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
//! 1. Read the route counters for the sampled forwards, reduce them over the
//!    EP group, and start deriving a new placement on EP rank 0's solver thread.
//! 2. Commit, a fixed number of forwards later: rank 0's placement is
//!    broadcast over the EP group and every rank plans the same slot moves.
//! 3. One chunk of layers per round: the weights move and the layers'
//!    routing tables switch.

mod algorithm;
mod moves;

use std::{collections::VecDeque, ops::Range, sync::Arc, thread};

pub use algorithm::{rebalance_experts, Algorithm};
pub use moves::{plan_slot_moves, SlotMoves};

/// Forwards between a snapshot and its commit (`COMMIT_DELAY_FORWARDS`).
pub const COMMIT_DELAY_FORWARDS: u64 = 200;

/// Physical counts, current slot map and version-correct logical counts per layer.
pub type ExpertLoad = (Vec<Vec<i64>>, Vec<Vec<i32>>, Vec<Vec<i64>>);

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
    /// The target's geometry.
    pub specs: EplbSpecs,
    /// An independently balanced draft in the same EP group.
    pub draft_specs: Option<EplbSpecs>,
    /// Forwards between snapshots (`--eplb-rebalance-num-iterations`).
    pub rebalance_num_iterations: u64,
    /// Number of actual forwards sampled immediately before a periodic snapshot.
    pub window_size: u64,
    /// Layers switched per round (`--eplb-rebalance-layers-per-chunk`).
    pub layers_per_chunk: usize,
    /// The solver variant.
    pub algorithm: Algorithm,
    /// Forwards between a snapshot and its commit.
    pub commit_delay_forwards: u64,
}

impl EplbConfig {
    fn models(&self) -> impl Iterator<Item = (&'static str, &EplbSpecs, Range<usize>)> {
        let target = &self.specs;
        let offset = target.num_layers;
        let draft = self
            .draft_specs
            .iter()
            .map(move |specs| ("draft", specs, offset..offset + specs.num_layers));
        std::iter::once(("target", target, 0..offset)).chain(draft)
    }
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
    logical_count: &[Vec<i64>],
    specs: &EplbSpecs,
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

/// The solver and the commit both read the placement and load at snapshot time.
struct Snapshot {
    placement: Vec<Vec<i32>>,
    logical_count: Vec<Vec<i64>>,
}

/// Decides when a rebalance starts and what each step carries.
/// The loop runs the device operations and collectives and feeds the results back.
pub struct Controller {
    config: EplbConfig,
    chunks: Vec<Vec<usize>>,
    phase: Phase,
    /// Forwards this rank ran (real or idle), rank-identical.
    forwards: u64,
    next_snapshot_at: u64,
    recording: bool,
    commit_at: Option<u64>,
    /// EP rank 0's solver.
    solver: Option<thread::JoinHandle<Result<Vec<Vec<i32>>, String>>>,
    snapshot: Option<Arc<Snapshot>>,
    new_map: Option<Vec<Vec<i32>>>,
    moves_by_layer: Vec<SlotMoves>,
    pending_chunks: VecDeque<Vec<usize>>,
    rebalances_completed: u64,
}

impl Controller {
    /// Balance the target and optional draft independently in the same gated cycle.
    /// Draft layers follow the target layers on the wire; a chunk belongs to one model.
    pub fn new(config: EplbConfig) -> Result<Self, String> {
        if config.rebalance_num_iterations == 0 {
            return Err("rebalance_num_iterations must be positive".to_owned());
        }
        if config.window_size == 0 || config.window_size > config.rebalance_num_iterations {
            return Err("window_size must be in 1..=rebalance_num_iterations".to_owned());
        }
        let mut chunks = chunk_layer_ids(config.specs.num_layers, config.layers_per_chunk)?;
        if let Some(specs) = &config.draft_specs {
            if specs.ep_rank_nodes != config.specs.ep_rank_nodes
                || specs.ep_size != config.specs.ep_size
                || specs.ep_rank != config.specs.ep_rank
                || specs.all_to_all_ep != config.specs.all_to_all_ep
            {
                return Err(
                    "target and draft EPLB must share the EP group and input distribution".into(),
                );
            }
            let mut draft_chunks = chunk_layer_ids(
                specs.num_layers,
                config.layers_per_chunk.min(specs.num_layers),
            )?;
            for layer in draft_chunks.iter_mut().flatten() {
                *layer += config.specs.num_layers;
            }
            chunks.extend(draft_chunks);
        }
        Ok(Self {
            next_snapshot_at: config.rebalance_num_iterations,
            recording: config.window_size == config.rebalance_num_iterations,
            config,
            chunks,
            phase: Phase::Idle,
            forwards: 0,
            commit_at: None,
            solver: None,
            snapshot: None,
            new_map: None,
            moves_by_layer: Vec::new(),
            pending_chunks: VecDeque::new(),
            rebalances_completed: 0,
        })
    }

    /// The geometry.
    pub fn specs(&self) -> &EplbSpecs {
        &self.config.specs
    }

    /// Whether the next forward contributes to the sampling window.
    pub fn recording(&self) -> bool {
        self.recording
    }

    /// A manual request outside a short recording window opens it immediately.
    /// If already sampling, finish the same window instead of resetting it.
    pub fn request_window(&mut self) -> Option<u64> {
        if self.phase != Phase::Idle
            || self.config.window_size == self.config.rebalance_num_iterations
        {
            return None;
        }
        if !self.recording {
            self.recording = true;
            self.next_snapshot_at = self.forwards + self.config.window_size;
        }
        Some(self.next_snapshot_at.saturating_sub(self.forwards))
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
        if self.config.window_size < self.config.rebalance_num_iterations {
            // A window may span publication of the preceding placement. If a
            // transfer actually overruns the next deadline, discard that sample
            // and start a fresh W-forward window once a snapshot can be taken.
            if self.recording && self.forwards >= self.next_snapshot_at && self.phase != Phase::Idle
            {
                self.recording = false;
            }
            let snapshot_consumed = self.phase != Phase::Computing || self.snapshot.is_some();
            if !self.recording
                && snapshot_consumed
                && self.forwards >= self.next_snapshot_at - self.config.window_size
                && (self.forwards < self.next_snapshot_at || self.phase == Phase::Idle)
            {
                self.recording = true;
                self.next_snapshot_at = self.forwards + self.config.window_size;
            }
        }
        match self.phase {
            Phase::Idle if self.recording && self.forwards >= self.next_snapshot_at => {
                vec![self.begin_snapshot()]
            }
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
        tracing::info!(
            forward = self.forwards,
            sampled_forwards = self
                .forwards
                .saturating_sub(self.next_snapshot_at - self.config.window_size),
            "EPLB recording window closed"
        );
        self.recording = self.config.window_size == self.config.rebalance_num_iterations;
        self.next_snapshot_at = self.forwards + self.config.rebalance_num_iterations;
        InternalOp::Snapshot
    }

    /// Consume the physical and version-correct logical load. The current
    /// placement seeds move planning. Rank 0 starts the solver.
    pub fn on_snapshot(
        &mut self,
        physical_count: Vec<Vec<i64>>,
        placement: Vec<Vec<i32>>,
        logical_count: Vec<Vec<i64>>,
    ) -> Result<(), String> {
        if self.phase != Phase::Computing || self.snapshot.is_some() {
            return Err(format!(
                "unexpected expert load snapshot in phase {}",
                self.phase.name()
            ));
        }
        let layers: usize = self
            .config
            .models()
            .map(|(_, specs, _)| specs.num_layers)
            .sum();
        if physical_count.len() != layers
            || placement.len() != layers
            || logical_count.len() != layers
        {
            return Err("expert load has the wrong shape (layer count)".into());
        }
        for (model, specs, range) in self.config.models() {
            for i in range.clone() {
                let (counts, map, credits) = (&physical_count[i], &placement[i], &logical_count[i]);
                if counts.len() != specs.num_physical_experts
                    || map.len() != counts.len()
                    || credits.len() != specs.num_logical_experts
                {
                    return Err(format!("{model} expert load has the wrong shape"));
                }
                if map
                    .iter()
                    .any(|&id| id < 0 || id as usize >= specs.num_logical_experts)
                    || counts.iter().chain(credits).any(|&count| count < 0)
                    || credits.iter().sum::<i64>() != counts.iter().sum::<i64>()
                {
                    return Err(format!("{model} invalid expert load or placement"));
                }
            }
            let routes: i64 = physical_count[range.clone()].iter().flatten().sum();
            let loads: Vec<Vec<f64>> = physical_count[range]
                .iter()
                .map(|row| row.iter().map(|&count| count as f64).collect())
                .collect();
            let (min, mean) = summary(&balancedness(&loads, specs.ep_size));
            tracing::info!("Expert rebalance {model}: snapshot of {routes} routes over {} layers; balancedness min {min:.3} mean {mean:.3}", specs.num_layers);
        }
        let snapshot = Arc::new(Snapshot {
            placement,
            logical_count,
        });
        if self.config.specs.ep_rank == 0 {
            let snapshot = Arc::clone(&snapshot);
            let config = self.config.clone();
            self.solver = Some(
                thread::Builder::new()
                    .name("ts-eplb-solver".to_owned())
                    .spawn(move || {
                        let mut placement = Vec::with_capacity(layers);
                        let counts = &snapshot.logical_count;
                        for (_, specs, range) in config.models() {
                            let mut rows = solve(&counts[range.clone()], specs, config.algorithm)?;
                            // Idle draft stages (e.g. a prefill-only interval) have no
                            // observations from which to improve their live placement.
                            for (row, layer) in rows.iter_mut().zip(range) {
                                if counts[layer].iter().all(|&traffic| traffic == 0) {
                                    row.clone_from(&snapshot.placement[layer]);
                                }
                            }
                            placement.extend(rows);
                        }
                        Ok(placement)
                    })
                    .map_err(|error| error.to_string())?,
            );
        }
        self.snapshot = Some(snapshot);
        self.commit_at = Some(self.forwards + self.config.commit_delay_forwards);
        Ok(())
    }

    /// EP rank 0's placement to broadcast, once its solver finished; `None`
    /// on the other ranks, which take the broadcast.
    pub fn placement_for_commit(&mut self) -> Result<Option<Vec<Vec<i32>>>, String> {
        if self.snapshot.is_none() {
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
        let Some(snapshot) = &self.snapshot else {
            return Err(format!(
                "unexpected expert placement commit in phase {}",
                self.phase.name()
            ));
        };
        let specs = &self.config.specs;
        let old_map = &snapshot.placement;
        if new_map.len() != old_map.len()
            || new_map
                .iter()
                .zip(&snapshot.logical_count)
                .any(|(row, counts)| row.iter().any(|&e| e < 0 || e as usize >= counts.len()))
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
                    old.len() / specs.ep_size,
                )
            })
            .collect::<Result<_, _>>()?;
        for (model, specs, layers) in self.config.models() {
            // Each expert's load split evenly over its replicas.
            let per_slot: Vec<Vec<f64>> = new_map[layers.clone()]
                .iter()
                .zip(&snapshot.logical_count[layers.clone()])
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
            let moved: usize = self.moves_by_layer[layers]
                .iter()
                .map(SlotMoves::num_changed_slots)
                .sum();
            tracing::info!(
                "Expert rebalance {model}: committed placement moves {moved} local slots; expected \
                 balancedness min {min:.3} mean {mean:.3}"
            );
        }
        self.snapshot = None;
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
        self.new_map = None;
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
            draft_specs: None,
            rebalance_num_iterations: 5,
            window_size: 5,
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
        controller.on_snapshot(load, trivial_map(), vec![vec![2, 2, 1, 40]; 3])
    }

    fn apply_all(controller: &mut Controller) {
        for ids in [vec![0, 1], vec![2]] {
            controller.chunk_payload(&ids).unwrap();
            controller.on_chunk_applied(&ids).unwrap();
        }
    }

    #[test]
    fn target_and_draft_balance_their_own_experts_and_apply_separate_chunks() {
        let config = controller(0).config;
        let draft = EplbSpecs {
            num_layers: 2,
            num_logical_experts: 2,
            num_physical_experts: 4,
            ..config.specs.clone()
        };
        let mut controller = Controller::new(EplbConfig {
            draft_specs: Some(draft),
            ..config
        })
        .unwrap();
        let mut old = trivial_map();
        old.extend([vec![0, 0, 1, 1], vec![0, 1, 0, 1]]);
        let mut load = vec![vec![1, 1, 1, 40, 1, 1]; 3];
        load.extend([vec![40, 40, 0, 1], vec![0; 4]]);
        controller.request_snapshot().unwrap();
        let mut logical = vec![vec![2, 40, 1, 2]; 3];
        logical.extend([vec![80, 1], vec![0, 0]]);
        controller.on_snapshot(load, old.clone(), logical).unwrap();
        let ops = rounds(&mut controller, 3, true);
        let new_map = controller.placement_for_commit().unwrap().unwrap();
        assert_eq!(new_map[0].iter().filter(|&&id| id == 1).count(), 3);
        assert_eq!(new_map[3].iter().filter(|&&id| id == 0).count(), 3);
        assert_eq!(new_map[4], old[4]); // Unobserved draft layer keeps its layout.
        controller.on_commit(new_map.clone()).unwrap();
        assert_eq!(ops.last(), Some(&InternalOp::ApplyChunk(vec![3, 4])));
        apply_all(&mut controller); // Target chunks precede the draft.
        assert_eq!(controller.chunk_payload(&[3, 4]).unwrap().0, new_map[3..]);
        controller.on_chunk_applied(&[3, 4]).unwrap();
        assert_eq!(controller.phase(), Phase::Idle);
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
        assert!(snapshot(&mut controller).is_err());
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
        let mut malformed = agreed.clone();
        malformed[0].pop();
        assert!(controller.on_commit(malformed).is_err());
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
            .on_snapshot(vec![vec![1; 6]; 2], trivial_map()[..2].to_vec(), vec![])
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
    fn short_windows_count_forwards_across_an_on_time_or_late_publication() {
        for late in [false, true] {
            let mut cfg = controller(0).config;
            cfg.rebalance_num_iterations = 10;
            cfg.window_size = 7;
            cfg.commit_delay_forwards = 4;
            let mut c = Controller::new(cfg).unwrap();
            rounds(&mut c, 2, true);
            rounds(&mut c, 16, false);
            assert!(!c.recording());
            c.note_round(true);
            assert!(c.recording());
            c.note_round(true);
            assert_eq!(c.request_window(), Some(6)); // Keep the sample already in progress.
            assert_eq!(rounds(&mut c, 6, true), [InternalOp::Snapshot]);
            assert!(!c.recording());
            snapshot(&mut c).unwrap();
            assert!(matches!(rounds(&mut c, 4, true)[0], InternalOp::Commit));
            let new = c.placement_for_commit().unwrap().unwrap();
            c.on_commit(new).unwrap();
            assert!(c.recording()); // The next sample has already opened.
            if late {
                assert!(rounds(&mut c, 6, true).is_empty());
                assert!(!c.recording());
            }
            apply_all(&mut c);
            c.note_round(false);
            assert!(c.recording());
            let remaining = if late { 7 } else { 6 };
            assert!(rounds(&mut c, remaining - 1, true).is_empty());
            assert_eq!(c.note_round(true), [InternalOp::Snapshot]);
        }
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
