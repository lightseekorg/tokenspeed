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

//! How each rank turns its expert slots from one placement into the next
//! (`plan_slot_moves` in `moe/expert_rebalance.py`): a pure function of the
//! two rows and the rank topology, so every rank derives the same plan and
//! the sends mirror the receives without further agreement.

use std::collections::{BTreeMap, HashMap};

/// One rank's part of one layer's weight redistribution (`SlotMoves`).
/// Slots are local indices, ranks EP ranks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SlotMoves {
    /// `(dst_slot, src_rank)` receives, by logical expert.
    pub recv: Vec<(usize, usize)>,
    /// `(src_slot, dst_rank)` sends, by logical expert then destination.
    pub send: Vec<(usize, usize)>,
    /// `(dst_slot, src_slot)` experts moved between two of this rank's slots.
    pub local_copy: Vec<(usize, usize)>,
    /// `(dst_slot, src_slot)` an earlier local slot receiving the same expert
    /// in this update.
    pub free_rider: Vec<(usize, usize)>,
}

impl SlotMoves {
    /// Local slots that receive a new expert.
    pub fn num_changed_slots(&self) -> usize {
        self.recv.len() + self.local_copy.len() + self.free_rider.len()
    }
}

/// `_spread`: assign `elements` to `over` in contiguous, even runs, the
/// remainder to the first targets.
fn spread(elements: &[usize], over: &[usize]) -> HashMap<usize, usize> {
    let mut assignment = HashMap::new();
    if elements.is_empty() || over.is_empty() {
        return assignment;
    }
    let (short, long_runs) = (elements.len() / over.len(), elements.len() % over.len());
    let mut start = 0;
    for (index, &target) in over.iter().enumerate() {
        let end = start + short + usize::from(index < long_runs);
        for &element in &elements[start..end] {
            assignment.insert(element, target);
        }
        start = end;
    }
    assignment
}

/// Plan how `ep_rank` turns its slots of `old` into `new`
/// (`[num_physical]` logical ids each); `ep_rank_nodes` is every EP rank's
/// node.
pub fn plan_slot_moves(
    old: &[i32],
    new: &[i32],
    ep_rank: usize,
    ep_rank_nodes: &[i64],
    num_local: usize,
) -> Result<SlotMoves, String> {
    let ep_size = ep_rank_nodes.len();
    let num_physical = old.len();
    if new.len() != num_physical {
        return Err("old_row and new_row must have the same length".to_owned());
    }
    if num_local == 0 || num_physical != ep_size * num_local {
        return Err(format!(
            "{num_physical} slots are not {ep_size} ranks x {num_local} local slots"
        ));
    }
    if ep_rank >= ep_size {
        return Err(format!(
            "ep_rank={ep_rank} is outside the {ep_size} EP ranks"
        ));
    }
    let first = ep_rank * num_local;

    // Per logical expert: the ranks holding it before (rank order) and the
    // ranks receiving it from elsewhere.
    let mut src_ranks_of: HashMap<i32, Vec<usize>> = HashMap::new();
    for (p, &logical) in old.iter().enumerate() {
        let ranks = src_ranks_of.entry(logical).or_default();
        if ranks.last() != Some(&(p / num_local)) {
            ranks.push(p / num_local);
        }
    }
    let mut dst_ranks_of: HashMap<i32, Vec<usize>> = HashMap::new();
    for (p, &logical) in new.iter().enumerate() {
        let rank = p / num_local;
        let holds = src_ranks_of
            .get(&logical)
            .is_some_and(|ranks| ranks.contains(&rank));
        let ranks = dst_ranks_of.entry(logical).or_default();
        if !holds && ranks.last() != Some(&rank) {
            ranks.push(rank);
        }
    }
    // This rank's slot holding each expert now: the lowest one.
    let mut held_slot_of: HashMap<i32, usize> = HashMap::new();
    for slot in (0..num_local).rev() {
        held_slot_of.insert(old[first + slot], slot);
    }

    let source_for = |logical: i32, dst_rank: usize| -> usize {
        let src_ranks = &src_ranks_of[&logical];
        let dst_ranks = &dst_ranks_of[&logical];
        let dst_node = ep_rank_nodes[dst_rank];
        let same_node_sources: Vec<usize> = src_ranks
            .iter()
            .copied()
            .filter(|&r| ep_rank_nodes[r] == dst_node)
            .collect();
        if !same_node_sources.is_empty() {
            let same_node_dsts: Vec<usize> = dst_ranks
                .iter()
                .copied()
                .filter(|&r| ep_rank_nodes[r] == dst_node)
                .collect();
            return spread(&same_node_dsts, &same_node_sources)[&dst_rank];
        }
        let src_nodes: Vec<i64> = src_ranks.iter().map(|&r| ep_rank_nodes[r]).collect();
        let cross_node_dsts: Vec<usize> = dst_ranks
            .iter()
            .copied()
            .filter(|&r| !src_nodes.contains(&ep_rank_nodes[r]))
            .collect();
        spread(&cross_node_dsts, src_ranks)[&dst_rank]
    };

    let mut recv: Vec<(i32, usize, usize)> = Vec::new();
    let mut moves = SlotMoves::default();
    let mut first_receiving_slot: HashMap<i32, usize> = HashMap::new();
    for dst_slot in 0..num_local {
        let logical = new[first + dst_slot];
        if old[first + dst_slot] == logical {
            continue;
        }
        if let Some(&held) = held_slot_of.get(&logical) {
            moves.local_copy.push((dst_slot, held));
            continue;
        }
        let earlier = *first_receiving_slot.entry(logical).or_insert(dst_slot);
        if earlier != dst_slot {
            moves.free_rider.push((dst_slot, earlier));
            continue;
        }
        recv.push((logical, dst_slot, source_for(logical, ep_rank)));
    }

    let mut send: Vec<(i32, usize, usize)> = Vec::new();
    let by_slot: BTreeMap<usize, i32> = held_slot_of
        .iter()
        .map(|(&logical, &slot)| (slot, logical))
        .collect();
    for (&src_slot, &logical) in &by_slot {
        for &dst_rank in dst_ranks_of.get(&logical).map_or(&[][..], Vec::as_slice) {
            if source_for(logical, dst_rank) == ep_rank {
                send.push((logical, src_slot, dst_rank));
            }
        }
    }
    recv.sort_unstable();
    send.sort_unstable();
    moves.recv = recv
        .into_iter()
        .map(|(_, slot, rank)| (slot, rank))
        .collect();
    moves.send = send
        .into_iter()
        .map(|(_, slot, rank)| (slot, rank))
        .collect();
    Ok(moves)
}
