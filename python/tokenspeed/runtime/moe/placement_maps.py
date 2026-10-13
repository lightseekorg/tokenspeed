# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Placement maps from a load: the CPU-only core of the EPLB placement.

Kept apart from ``expert_location`` on purpose: the online rebalance runs
``compute_placement_maps`` in a spawned CPU worker process, which imports
this module and nothing else of the runtime (torch and the algorithm). The
functions here never touch a device.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.moe import eplb_algorithms

__all__ = ["compute_placement_maps", "pad_replica_table", "replica_table_width"]


def replica_table_width(num_physical_experts: int, num_logical_experts: int) -> int:
    """Columns of the replica table: the most replicas one logical expert can have.

    Every other expert needs at least one slot, so ``P - (E - 1)`` -- ``R + 1``
    with ``R`` redundant slots. Fixed for the server's lifetime so an online
    rebalance rewrites the table in place.
    """
    if num_physical_experts < num_logical_experts:
        raise ValueError(
            f"{num_physical_experts} physical experts cannot hold "
            f"{num_logical_experts} logical experts"
        )
    return num_physical_experts - num_logical_experts + 1


def pad_replica_table(table: torch.Tensor, width: int) -> torch.Tensor:
    """Bring a ``[..., X]`` replica table to ``width`` columns (-1 padded), int32 contiguous."""
    current = table.shape[-1]
    if current > width:
        if bool((table[..., width:] != -1).any()):
            raise ValueError(
                f"replica table holds more than {width} replicas of one expert"
            )
        table = table[..., :width]
    elif current < width:
        pad = torch.full(
            (*table.shape[:-1], width - current),
            -1,
            dtype=table.dtype,
            device=table.device,
        )
        table = torch.cat([table, pad], dim=-1)
    return table.to(torch.int32).contiguous()


def compute_placement_maps(
    logical_count: torch.Tensor,
    *,
    num_physical_experts: int,
    ep_size: int,
    num_groups: int | None,
    num_nodes: int,
    algorithm: eplb_algorithms.EplbAlgorithm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive a placement from a logical load with the EPLB algorithm, on the host.

    Pure CPU work, deterministic for equal inputs (double-precision counts,
    stable sorts), so it may run off the control plane -- in a spawned worker
    process during serving -- and gives the same maps on every rank.

    Args:
        logical_count: ``[layers, logical]`` routes per logical expert, on the
            host.
        num_physical_experts: Slots per layer, ``P = E + R``.
        ep_size: Ranks the slots are spread over.
        num_groups: The model's expert groups, or None.
        num_nodes: Nodes the EP ranks span (the hierarchical algorithm's tier).
        algorithm: The EPLB algorithm variant.

    Returns:
        ``(physical_to_logical_map [layers, P], logical_to_all_physical_map
        [layers, E, X])`` int32 host tensors; the replica table is -1 padded
        to the fixed width ``replica_table_width``.
    """
    if logical_count.device.type != "cpu":
        raise ValueError("compute_placement_maps runs on host tensors only")
    if logical_count.ndim != 2:
        raise ValueError(
            f"logical_count must be [layers, logical], got {tuple(logical_count.shape)}"
        )
    if ep_size <= 0 or num_physical_experts % ep_size:
        raise ValueError(
            f"{num_physical_experts} physical experts do not divide over ep_size={ep_size}"
        )
    num_logical_experts = logical_count.shape[1]
    physical_to_logical_map, logical_to_all_physical_map, _ = (
        eplb_algorithms.rebalance_experts(
            # The algorithm sums recording windows over its leading dim.
            tokens_per_expert=logical_count.unsqueeze(0),
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_physical_experts // ep_size,
            num_groups=num_groups,
            num_nodes=num_nodes,
            algorithm=algorithm,
        )
    )
    return (
        physical_to_logical_map.to(torch.int32).contiguous(),
        pad_replica_table(
            logical_to_all_physical_map,
            replica_table_width(num_physical_experts, num_logical_experts),
        ),
    )
