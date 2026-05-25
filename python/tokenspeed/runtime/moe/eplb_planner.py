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

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from tokenspeed.runtime.moe import eplb_algorithms
from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata


@dataclass
class EplbPlanResult:
    physical_to_logical_map_cpu: torch.Tensor
    changed_layers: list[int]
    balancedness: float


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _group_rank(group=None) -> int:
    if not _dist_ready():
        return 0
    try:
        return dist.get_rank(group=group)
    except TypeError:
        return dist.get_rank()


def _group_src_rank(group=None) -> int:
    if group is None or not hasattr(dist, "get_global_rank"):
        return 0
    try:
        return dist.get_global_rank(group, 0)
    except Exception:
        return 0


def _placement_identical(
    old_metadata: ExpertLocationMetadata,
    new_metadata: ExpertLocationMetadata,
    layer_ids: list[int] | None = None,
) -> bool:
    old_map = old_metadata.physical_to_logical_map_cpu
    new_map = new_metadata.physical_to_logical_map_cpu
    if layer_ids is not None:
        old_map = old_map[layer_ids]
        new_map = new_map[layer_ids]
    return bool(torch.equal(old_map, new_map))


def _diff_layers_from_map(
    old_metadata: ExpertLocationMetadata,
    physical_to_logical_map_cpu: torch.Tensor,
) -> list[int]:
    old_map = old_metadata.physical_to_logical_map_cpu
    new_map = physical_to_logical_map_cpu
    if old_map.shape != new_map.shape:
        return list(range(int(new_map.shape[0])))
    changed = (old_map != new_map).any(dim=1).nonzero(as_tuple=False).flatten()
    return [int(x) for x in changed.tolist()]


def _diff_layers(
    old_metadata: ExpertLocationMetadata, new_metadata: ExpertLocationMetadata
) -> list[int]:
    old_map = old_metadata.physical_to_logical_map_cpu
    new_map = new_metadata.physical_to_logical_map_cpu
    if old_map.shape != new_map.shape:
        return list(range(int(new_map.shape[0])))
    changed = (old_map != new_map).any(dim=1).nonzero(as_tuple=False).flatten()
    return [int(x) for x in changed.tolist()]


def _balancedness(logical_count: torch.Tensor | None) -> float:
    if logical_count is None or logical_count.numel() == 0:
        return math.nan
    counts = logical_count.detach().cpu().float()
    if counts.dim() > 2:
        counts = counts.sum(dim=0)
    max_count = counts.max(dim=-1).values
    mean_count = counts.mean(dim=-1)
    score = (mean_count + 1e-5) / (max_count + 1e-5)
    return float(score.mean().item())


def _plan_physical_to_logical_cpu(
    logical_count: torch.Tensor,
    *,
    server_args,
    model_config,
) -> torch.Tensor:
    if not isinstance(logical_count, torch.Tensor):
        logical_count = torch.tensor(logical_count)
    if len(logical_count.shape) == 2:
        logical_count = logical_count.unsqueeze(0)
    logical_count = logical_count.detach().cpu()

    common = ExpertLocationMetadata._init_common(server_args, model_config)
    model_config_for_expert_location = common["model_config_for_expert_location"]
    physical_to_logical_map, _, _ = eplb_algorithms.rebalance_experts(
        tokens_per_expert=logical_count,
        num_physical_experts=common["num_physical_experts"],
        num_local_physical_experts=common["num_local_physical_experts"],
        num_groups=model_config_for_expert_location.num_groups,
        num_nodes=server_args.mapping.nnodes,
        algorithm=eplb_algorithms.compute_algorithm(
            raw_algorithm=server_args.eplb_algorithm,
            num_groups=model_config_for_expert_location.num_groups,
            num_nodes=server_args.mapping.nnodes,
        ),
    )
    return physical_to_logical_map.detach().cpu()


def run_planner_with_broadcast(
    logical_count: torch.Tensor,
    *,
    rank: int,
    current_metadata: ExpertLocationMetadata,
    server_args,
    model_config,
    eplb_pg=None,
    eplb_control_pg=None,
) -> EplbPlanResult:
    """Run CPU EPLB on group rank 0 and broadcast placement to peers."""

    del rank
    logical_count = logical_count.detach().cpu()
    if not _dist_ready():
        physical_to_logical_map_cpu = _plan_physical_to_logical_cpu(
            logical_count,
            server_args=server_args,
            model_config=model_config,
        )
        return EplbPlanResult(
            physical_to_logical_map_cpu=physical_to_logical_map_cpu,
            changed_layers=_diff_layers_from_map(
                current_metadata, physical_to_logical_map_cpu
            ),
            balancedness=_balancedness(logical_count),
        )

    control_pg = eplb_control_pg if eplb_control_pg is not None else eplb_pg
    group_rank = _group_rank(control_pg)
    payload: list[torch.Tensor | None] = [None]
    if group_rank == 0:
        payload[0] = _plan_physical_to_logical_cpu(
            logical_count,
            server_args=server_args,
            model_config=model_config,
        )

    dist.broadcast_object_list(
        payload, src=_group_src_rank(control_pg), group=control_pg
    )
    physical_to_logical_map = payload[0]
    assert physical_to_logical_map is not None
    physical_to_logical_map = physical_to_logical_map.detach().cpu()
    return EplbPlanResult(
        physical_to_logical_map_cpu=physical_to_logical_map,
        changed_layers=_diff_layers_from_map(current_metadata, physical_to_logical_map),
        balancedness=_balancedness(logical_count),
    )
