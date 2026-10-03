# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the FluentLLM project
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

"""Expert placement: which logical expert every physical expert slot holds.

With ``--ep-num-redundant-experts R`` a MoE layer has ``P = E + R`` physical
slots over ``ep_size`` ranks; a *placement* assigns a logical expert to each
slot, so a hot expert can have several replicas. Routing emits physical ids,
the loader fills every slot from its logical expert's checkpoint tensors, and
the load counters record how many routes each slot received so a better
placement can be derived (``--init-expert-location <load.pt>``).

The placement is process-global for the target model
(``set_global_expert_location_metadata``); drafts route their own experts
trivially. Zero experts (LongCat) never enter these tables: the router keeps
them as ``-1`` and maps only real expert ids.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed
import torch.nn.functional as F

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.model_loader import get_model_architecture
from tokenspeed.runtime.moe import eplb_algorithms
from tokenspeed.runtime.utils.server_args import (
    ServerArgs,
    expert_placement_requested,
)

__all__ = [
    "ExpertLocationMetadata",
    "ModelConfigForExpertLocation",
    "build_expert_placement",
    "compute_initial_expert_location_metadata",
    "compute_logical_to_rank_dispatch_physical_map",
    "expert_load_recording_enabled",
    "expert_placement_requested",
    "get_global_expert_location_metadata",
    "set_global_expert_location_metadata",
]

logger = logging.getLogger(__name__)

STATIC_DISPATCH_ALGORITHMS = frozenset({"static", "static_with_zero_expert"})


@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    physical_to_logical_map_cpu: torch.Tensor
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    # (layers, num_logical_experts): this rank's replica of every logical
    # expert under a static dispatch algorithm, else None.
    logical_to_rank_dispatch_physical_map: torch.Tensor | None
    ep_size: int
    # Routing dispatch tables: the replicas of every logical expert, int32
    # and trimmed to the widest replica count so the router indexes a small
    # [logical, X] table, plus how many of each row are valid.
    dispatch_replicas: torch.Tensor = field(init=False)
    dispatch_num_replicas: torch.Tensor = field(init=False)
    # (layers, num_physical_experts) int32 routes to each physical expert
    # since the last reset; None until load recording is enabled.
    physical_load: torch.Tensor | None = field(init=False, default=None)

    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        return self.num_physical_experts // self.ep_size

    @property
    def num_logical_experts(self) -> int:
        return self.logical_to_all_physical_map.shape[1]

    @property
    def has_redundancy(self) -> bool:
        return self.num_physical_experts != self.num_logical_experts

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        if not num_layers_0 == num_layers_1 == num_layers_2:
            raise ValueError(
                "Expert location maps disagree on layer count: "
                f"{num_layers_0}, {num_layers_1}, {num_layers_2}."
            )
        if num_logical_experts_0 != num_logical_experts_1:
            raise ValueError(
                "Expert location maps disagree on logical expert count: "
                f"{num_logical_experts_0}, {num_logical_experts_1}."
            )
        if num_physical_experts_0 != num_physical_experts_1:
            raise ValueError(
                "Expert location maps disagree on physical expert count: "
                f"{num_physical_experts_0}, {num_physical_experts_1}."
            )
        if self.ep_size <= 0 or num_physical_experts_0 % self.ep_size:
            raise ValueError(
                f"{num_physical_experts_0} physical experts do not divide over "
                f"ep_size={self.ep_size}."
            )
        widest = int(self.logical_to_all_physical_map_num_valid.max().item())
        self.dispatch_replicas = (
            self.logical_to_all_physical_map[..., :widest].to(torch.int32).contiguous()
        )
        self.dispatch_num_replicas = self.logical_to_all_physical_map_num_valid.to(
            torch.int32
        ).contiguous()

    # -------------------------------- placement queries ------------------------------

    def local_physical_slots(
        self, layer_id: int, logical_expert_id: int, ep_rank: int
    ) -> list[int]:
        """Return the rank-local slots (0-based) holding ``logical_expert_id``."""
        local = self.num_local_physical_experts
        return [
            physical - ep_rank * local
            for physical in self.logical_to_all_physical(layer_id, logical_expert_id)
            if physical // local == ep_rank
        ]

    def local_logical_experts(self, layer_id: int, ep_rank: int) -> list[int]:
        """Return the distinct logical experts placed on ``ep_rank``, in slot order."""
        return list(dict.fromkeys(self.local_slot_logical_experts(layer_id, ep_rank)))

    def local_slot_logical_experts(self, layer_id: int, ep_rank: int) -> list[int]:
        """Return the logical expert held by each of ``ep_rank``'s slots, in slot order."""
        local = self.num_local_physical_experts
        return self.physical_to_logical_map_cpu[
            layer_id, ep_rank * local : (ep_rank + 1) * local
        ].tolist()

    # -------------------------------- load recording ---------------------------------

    def enable_load_recording(self) -> None:
        """Allocate the per-physical-expert route counters the router bumps."""
        self.physical_load = torch.zeros(
            (self.num_layers, self.num_physical_experts),
            dtype=torch.int32,
            device=self.physical_to_logical_map.device,
        )

    def reset_load(self) -> None:
        if self.physical_load is None:
            raise RuntimeError("expert load recording is not enabled")
        self.physical_load.zero_()

    def logical_load(self, physical_load: torch.Tensor) -> torch.Tensor:
        """Sum a host ``[layers, physical]`` count over each logical expert's replicas."""
        if physical_load.device.type != "cpu":
            raise ValueError("logical_load reduces host-side counts")
        logical = torch.zeros(
            (self.num_layers, self.num_logical_experts), dtype=torch.int64
        )
        logical.scatter_add_(1, self.physical_to_logical_map_cpu, physical_load.long())
        return logical

    def rank_load(self, physical_load: torch.Tensor) -> torch.Tensor:
        """Sum a ``[layers, physical]`` count over each rank's slots -> ``[layers, ep]``."""
        return physical_load.view(self.num_layers, self.ep_size, -1).sum(-1)

    def load_record(self, physical_load: torch.Tensor) -> dict[str, torch.Tensor]:
        """Package a ``[layers, physical]`` count for ``--init-expert-location``.

        ``logical_count`` is what ``init_by_eplb`` consumes; the rest lets a
        reader judge the placement that produced the counts. ``balancedness``
        is, per layer, the mean rank load over the busiest rank's load.
        """
        physical = physical_load.cpu()
        per_rank = self.rank_load(physical).double()
        balancedness = per_rank.mean(-1) / per_rank.max(-1).values.clamp_min(1)
        return {
            "logical_count": self.logical_load(physical),
            "physical_count": physical,
            "physical_to_logical_map": self.physical_to_logical_map_cpu.clone(),
            "rank_count": per_rank.to(torch.int64),
            "balancedness": balancedness,
        }

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(server_args: ServerArgs, model_config: ModelConfig):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args, model_config)
        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        physical_to_logical_map = (
            torch.arange(0, num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,
    ):
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        if tuple(physical_to_logical_map.shape) != (
            model_config_for_expert_location.num_layers,
            common["num_physical_experts"],
        ):
            raise ValueError(
                f"physical_to_logical_map has shape "
                f"{tuple(physical_to_logical_map.shape)}, expected "
                f"({model_config_for_expert_location.num_layers}, "
                f"{common['num_physical_experts']}) for this model and "
                f"--ep-num-redundant-experts {server_args.ep_num_redundant_experts}."
            )
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.mapping.nnodes
        expected = (
            model_config_for_expert_location.num_layers,
            model_config_for_expert_location.num_logical_experts,
        )
        if tuple(logical_count.shape[-2:]) != expected:
            raise ValueError(
                f"logical_count has shape {tuple(logical_count.shape)}; the model "
                f"has {expected[0]} MoE layers of {expected[1]} routed experts."
            )

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.mapping.moe.ep_size
        if ep_size <= 0 or num_physical_experts % ep_size != 0:
            raise ValueError(
                f"{num_physical_experts} physical experts "
                f"({model_config_for_expert_location.num_logical_experts} routed + "
                f"{server_args.ep_num_redundant_experts} redundant) do not divide "
                f"over ep_size={ep_size}."
            )
        num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
    ):
        return ExpertLocationMetadata.from_maps(
            physical_to_logical_map,
            logical_to_all_physical_map,
            ep_size=ep_size,
            ep_rank=server_args.mapping.moe.ep_rank,
            num_nodes=server_args.mapping.nnodes,
            dispatch_algorithm=server_args.ep_dispatch_algorithm,
        )

    @staticmethod
    def from_physical_to_logical_map(
        physical_to_logical_map: torch.Tensor,
        num_logical_experts: int,
        *,
        ep_size: int,
        ep_rank: int,
        num_nodes: int,
        dispatch_algorithm: str | None,
    ) -> "ExpertLocationMetadata":
        """Build a placement from ``[layers, physical]`` logical ids alone.

        Args:
            physical_to_logical_map: The logical expert held by every slot.
            num_logical_experts: Routed expert count ``E`` of the model.
            ep_size: Ranks the slots are spread over, contiguously.
            ep_rank: This rank, for the static dispatch map.
            num_nodes: Node count, so the static map prefers same-node replicas.
            dispatch_algorithm: ``--ep-dispatch-algorithm``; the static map is
                computed for ``static`` / ``static_with_zero_expert`` only.
        """
        return ExpertLocationMetadata.from_maps(
            physical_to_logical_map,
            _compute_logical_to_all_physical_map(
                physical_to_logical_map, num_logical_experts=num_logical_experts
            ),
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_nodes=num_nodes,
            dispatch_algorithm=dispatch_algorithm,
        )

    @staticmethod
    def from_maps(
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
        *,
        ep_size: int,
        ep_rank: int,
        num_nodes: int,
        dispatch_algorithm: str | None,
    ) -> "ExpertLocationMetadata":
        _, num_physical_experts = physical_to_logical_map.shape

        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    num_gpus=ep_size,
                    num_nodes=num_nodes,
                    num_physical_experts=num_physical_experts,
                    ep_rank=ep_rank,
                )
                if dispatch_algorithm in STATIC_DISPATCH_ALGORITHMS
                else None
            ),
            ep_size=ep_size,
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: "ExpertLocationMetadata",
        update_layer_ids: list[int],
    ):
        """Overwrite ``update_layer_ids`` in place from ``other`` (graph-safe).

        The routing tables are updated in place so views held by MoE layers
        (and captured graphs) see the new placement.
        """
        if self.ep_size != other.ep_size:
            raise ValueError(
                "Cannot update ExpertLocationMetadata with different ep_size."
            )

        for self_field, other_field, name in (
            (
                self.physical_to_logical_map,
                other.physical_to_logical_map,
                "physical_to_logical_map",
            ),
            (
                self.physical_to_logical_map_cpu,
                other.physical_to_logical_map_cpu,
                "physical_to_logical_map_cpu",
            ),
            (
                self.logical_to_all_physical_map,
                other.logical_to_all_physical_map,
                "logical_to_all_physical_map",
            ),
            (
                self.logical_to_all_physical_map_num_valid,
                other.logical_to_all_physical_map_num_valid,
                "logical_to_all_physical_map_num_valid",
            ),
            (
                self.logical_to_rank_dispatch_physical_map,
                other.logical_to_rank_dispatch_physical_map,
                "logical_to_rank_dispatch_physical_map",
            ),
            (self.dispatch_replicas, other.dispatch_replicas, "dispatch_replicas"),
            (
                self.dispatch_num_replicas,
                other.dispatch_num_replicas,
                "dispatch_num_replicas",
            ),
        ):
            if (other_field is not None) != (self_field is not None):
                raise ValueError(
                    f"Cannot update ExpertLocationMetadata with incompatible {name}."
                )
            if self_field is None:
                continue
            if self_field.shape != other_field.shape:
                raise ValueError(
                    f"Cannot update ExpertLocationMetadata: {name} has shape "
                    f"{tuple(other_field.shape)}, expected {tuple(self_field.shape)}."
                )
            mask_update = torch.tensor(
                [i in update_layer_ids for i in range(self.num_layers)]
            )
            mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
            mask_update = mask_update.to(self_field.device, non_blocking=True)
            self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self, layer_id: int, logical_expert_id: int
    ) -> list[int]:
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]


def _compute_logical_to_all_physical_map(
    physical_to_logical_map: torch.Tensor, num_logical_experts: int
):
    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            if not 0 <= logical_expert_id < num_logical_experts:
                raise ValueError(
                    f"physical_to_logical_map[{layer_id}, {physical_expert_id}] = "
                    f"{logical_expert_id} is not a logical expert in "
                    f"[0, {num_logical_experts})."
                )
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    for layer_id, layer_map in enumerate(logical_to_all_physical_map):
        missing = [e for e, slots in enumerate(layer_map) if not slots]
        if missing:
            raise ValueError(
                f"Layer {layer_id}: logical experts {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''} have no physical slot."
            )

    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_nodes: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    """Pick, for every rank, the replica it dispatches each logical expert to.

    Nearest first: a replica on the same GPU, then one on the same node, else a
    seeded fair draw over all replicas so the remote ranks spread evenly.
    Returns ``ep_rank``'s ``[layers, logical]`` slice.
    """
    r = random.Random(seed)

    if num_nodes <= 0 or num_gpus % num_nodes:
        raise ValueError(
            f"ep_size={num_gpus} ranks do not spread evenly over nnodes={num_nodes}"
        )
    num_local_gpu_physical_experts = num_physical_experts // num_gpus
    num_gpus_per_node = num_gpus // num_nodes
    num_local_node_physical_experts = num_local_gpu_physical_experts * num_gpus_per_node
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    logical_to_rank_dispatch_physical_map = torch.full(
        size=(num_gpus, num_layers, num_logical_experts),
        fill_value=-1,
        dtype=dtype,
    )

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )
            output_partial = logical_to_rank_dispatch_physical_map[
                :, layer_id, logical_expert_id
            ]

            for gpu_id in range(num_gpus):
                output_partial[gpu_id] = _find_nearest_expert(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    gpu_id=gpu_id,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

            num_remain = int(torch.sum(output_partial == -1).item())
            if num_remain:
                output_partial[output_partial == -1] = torch.tensor(
                    _fair_choices(candidate_physical_expert_ids, k=num_remain, r=r),
                    dtype=dtype,
                )

    if not torch.all(logical_to_rank_dispatch_physical_map != -1):
        raise RuntimeError(
            "logical_to_rank_dispatch_physical_map contains unassigned entries."
        )

    device = logical_to_all_physical_map.device
    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> list[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_physical_experts: int
) -> int:
    return physical_expert_id // num_local_physical_experts


def _find_nearest_expert(
    candidate_physical_expert_ids: list[int],
    num_local_gpu_physical_experts: int,
    gpu_id: int,
    num_gpus_per_node: int,
    num_local_node_physical_experts: int,
) -> int:
    """Return the same-GPU, else same-node, replica for ``gpu_id``; -1 if neither."""
    if len(candidate_physical_expert_ids) == 1:
        return candidate_physical_expert_ids[0]
    for physical_expert_id in candidate_physical_expert_ids:
        if (
            _compute_gpu_id_of_physical_expert(
                physical_expert_id, num_local_gpu_physical_experts
            )
            == gpu_id
        ):
            return physical_expert_id
    node_id = gpu_id // num_gpus_per_node
    for physical_expert_id in candidate_physical_expert_ids:
        if (
            _compute_gpu_id_of_physical_expert(
                physical_expert_id, num_local_node_physical_experts
            )
            == node_id
        ):
            return physical_expert_id
    return -1


def _fair_choices(arr: list, k: int, r: random.Random) -> list:
    quotient, remainder = divmod(k, len(arr))
    choices = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(choices)
    return choices


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: int | None = None

    @staticmethod
    def init_dummy():
        return ModelConfigForExpertLocation(num_layers=1, num_logical_experts=1)

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return ModelConfigForExpertLocation.init_dummy()


_global_expert_location_metadata: ExpertLocationMetadata | None = None


def set_global_expert_location_metadata(
    metadata: ExpertLocationMetadata | None,
) -> None:
    """Install the target model's placement (None: trivial routing)."""
    global _global_expert_location_metadata
    _global_expert_location_metadata = metadata


def get_global_expert_location_metadata() -> ExpertLocationMetadata | None:
    return _global_expert_location_metadata


def expert_load_recording_enabled() -> bool:
    """Whether the global placement carries the route counters."""
    placement = _global_expert_location_metadata
    return placement is not None and placement.physical_load is not None


def build_expert_placement(
    server_args: ServerArgs, model_config: ModelConfig
) -> ExpertLocationMetadata | None:
    """Build the serving placement, or None when routing stays untouched.

    The placement exists when serving asks for redundant experts, a
    non-trivial initial location or load recording
    (``expert_placement_requested``), and the model declares an
    expert-location geometry. Load recording allocates the counters the
    router bumps.
    """
    if not expert_placement_requested(server_args):
        return None
    geometry = ModelConfigForExpertLocation.from_model_config(model_config)
    if geometry.num_logical_experts <= 1:
        raise ValueError(
            "Expert placement was requested for a model without routed experts."
        )
    placement = compute_initial_expert_location_metadata(server_args, model_config)
    if server_args.expert_distribution_recorder_mode is not None:
        placement.enable_load_recording()
    logger.info(
        f"Expert placement: {placement.num_logical_experts} logical experts on "
        f"{placement.num_physical_experts} physical slots over "
        f"ep_size={placement.ep_size} "
        f"({placement.num_local_physical_experts} per rank), dispatch "
        f"{server_args.ep_dispatch_algorithm}, load recording "
        f"{'on' if placement.physical_load is not None else 'off'}"
    )
    return placement


def compute_initial_expert_location_metadata(
    server_args: ServerArgs, model_config: ModelConfig
) -> ExpertLocationMetadata:
    data = server_args.init_expert_location
    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(server_args, model_config)

    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    # A load record (EXPERT_LOAD profile) carries both its counts and the
    # placement that produced them; the counts win, since the point of the
    # record is to derive a better placement. A bare map pins one exactly.
    if "logical_count" in data_dict:
        logger.info(
            f"init_expert_location: EPLB placement from the logical_count in {data!s}"
        )
        return ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=data_dict["logical_count"]
        )
    elif "physical_to_logical_map" in data_dict:
        logger.info(
            f"init_expert_location: placement pinned by the physical_to_logical_map "
            f"in {data!s}"
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=data_dict["physical_to_logical_map"],
        )
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )
