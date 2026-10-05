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
placement can be derived (``--init-expert-location <records>``).

The placement is process-global for the target model
(``set_global_expert_location_metadata``); drafts route their own experts
trivially. Models opt in explicitly (``BaseCausalLM.supports_expert_placement``)
by building their MoE layers from the placement; the placement is refused for
any other model. Zero experts (LongCat) never enter these tables: the router
keeps them as ``-1`` and maps only real expert ids.

Load records are per rank and never reduced on the serving path (a collective
there would deadlock attention-DP workers that stop a profile independently):
each rank writes its own counters, and ``merge_expert_load_records`` sums the
ranks' files when the record is consumed.

The tables are mutable in place (``update_layers``): the router and the
captured CUDA graphs hold views of them, so an online rebalance rewrites rows
in their existing storage and never reallocates. The replica table therefore
keeps a fixed width, ``R + 1`` (the most replicas one expert can have), rather
than the widest count of the current placement.
"""

from __future__ import annotations

import glob
import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader import get_model_architecture
from tokenspeed.runtime.moe import eplb_algorithms
from tokenspeed.runtime.moe.expert_load_rows import ExpertLoadRowMask
from tokenspeed.runtime.moe.placement_maps import (
    compute_placement_maps,
    pad_replica_table,
    replica_table_width,
)
from tokenspeed.runtime.utils.server_args import (
    ServerArgs,
    expert_placement_requested,
)

__all__ = [
    "EXPERT_LOAD_RECORD_SUFFIX",
    "ExpertLoadSnapshot",
    "ExpertLocationMetadata",
    "InitExpertLocationForm",
    "ModelConfigForExpertLocation",
    "PreparedPlacementRows",
    "build_expert_placement",
    "compute_initial_expert_location_metadata",
    "compute_logical_to_rank_dispatch_physical_map",
    "compute_placement_maps",
    "expert_load_recording_enabled",
    "expert_placement_requested",
    "get_global_expert_location_metadata",
    "init_expert_location_form",
    "load_balancedness",
    "logical_count_of",
    "merge_expert_load_records",
    "pad_replica_table",
    "replica_table_width",
    "set_global_expert_location_metadata",
]

logger = logging.getLogger(__name__)

# File suffix of the per-rank load record the EXPERT_LOAD profile activity
# writes; a directory given to --init-expert-location is scanned for it.
EXPERT_LOAD_RECORD_SUFFIX = ".expert-load.pt"


@dataclass(frozen=True)
class ExpertLoadSnapshot:
    """One rank's route counters since the previous snapshot, on the host.

    Attributes:
        physical_count: ``[layers, physical]`` int64 routes per slot.
        physical_to_logical_map: The placement those routes were counted
            under, so the logical load is derived from the matching rows even
            if the tables move afterwards.
    """

    physical_count: torch.Tensor
    physical_to_logical_map: torch.Tensor

    @property
    def logical_count(self) -> torch.Tensor:
        return logical_count_of(
            self.physical_count,
            self.physical_to_logical_map,
            int(self.physical_to_logical_map.max().item()) + 1,
        )


@dataclass(frozen=True)
class PreparedPlacementRows:
    """Host-side rows of one chunk's table switch (``prepare_layers``).

    Attributes:
        layer_ids: The chunk's layers, in switch order.
        physical_to_logical: ``[n, P]`` int64 logical id per slot.
        replicas: ``[n, E, X]`` int32 replica table rows, -1 padded.
        num_valid: ``[n, E]`` int32 replicas per logical expert.
        rank_dispatch: ``[n, E]`` int32 rows of this rank's static dispatch
            map, or None when the placement never materialized it.
    """

    layer_ids: tuple[int, ...]
    physical_to_logical: torch.Tensor
    replicas: torch.Tensor
    num_valid: torch.Tensor
    rank_dispatch: torch.Tensor | None


@dataclass
class ExpertLocationMetadata:
    """One rank's view of the expert placement.

    The slots are owned contiguously: rank ``r`` holds physical experts
    ``[r * P / ep_size, (r + 1) * P / ep_size)`` of every layer.
    """

    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    physical_to_logical_map_cpu: torch.Tensor
    # (layers, num_logical_experts, X) int32: the replicas of every logical
    # expert, -1 padded to the fixed width X = R + 1 (``replica_table_width``).
    # This is the one routing table; the router indexes a per-layer slice of
    # it, and ``update_layers`` rewrites rows in place.
    logical_to_all_physical_map: torch.Tensor
    # (layers, num_logical_experts) int32 valid entries per row, all >= 1.
    logical_to_all_physical_map_num_valid: torch.Tensor
    ep_size: int
    ep_rank: int
    # The node of every EP rank (len ``ep_size``), so the per-rank static map
    # can prefer a same-node replica. Taken from the mapping's ranks within
    # the MoE group, not from a divisibility assumption.
    ep_rank_nodes: tuple[int, ...]
    # (layers, num_physical_experts) int64 routes to each physical expert
    # since the last reset; None until load recording is enabled.
    physical_load: torch.Tensor | None = field(init=False, default=None)
    # Which rows of the MoE input are real tokens (padded forwards carry
    # filler rows the counters must not see); enabled with the counters.
    load_rows: ExpertLoadRowMask | None = field(init=False, default=None)
    # This rank's static dispatch map (layers, num_logical_experts), computed
    # on first use: only all-to-all EP under a static algorithm needs it.
    _rank_dispatch_map: torch.Tensor | None = field(
        init=False, default=None, repr=False
    )

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

    def __post_init__(self):
        num_layers_0, num_physical_experts = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, _ = self.logical_to_all_physical_map.shape
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
        if self.ep_size <= 0 or num_physical_experts % self.ep_size:
            raise ValueError(
                f"{num_physical_experts} physical experts do not divide over "
                f"ep_size={self.ep_size}."
            )
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"ep_rank={self.ep_rank} is outside ep_size={self.ep_size}"
            )
        if len(self.ep_rank_nodes) != self.ep_size:
            raise ValueError(
                f"ep_rank_nodes names {len(self.ep_rank_nodes)} ranks, "
                f"ep_size={self.ep_size}"
            )
        num_valid = self.logical_to_all_physical_map_num_valid
        if int(num_valid.min().item()) < 1:
            raise ValueError("every logical expert needs at least one physical slot")
        # One routing table at the fixed width R + 1 (never P columns): the
        # router indexes a small [logical, X] slice, and an online rebalance
        # rewrites rows in place, so the width cannot follow the placement.
        self.logical_to_all_physical_map = pad_replica_table(
            self.logical_to_all_physical_map,
            replica_table_width(num_physical_experts, num_logical_experts_0),
        )
        self.logical_to_all_physical_map_num_valid = num_valid.to(
            torch.int32
        ).contiguous()

    # -------------------------------- in-place updates ------------------------------

    def prepare_layers(
        self, layer_ids: Sequence[int], new_rows: torch.Tensor
    ) -> PreparedPlacementRows:
        """Derive, on the host, everything a table switch of ``layer_ids`` writes.

        One call per chunk: the inverse (replica) rows, their valid counts and
        -- when this rank's static dispatch map has been materialized -- its
        rows for exactly these layers. Validation (every logical expert keeps
        a slot) happens here, before any table is touched, so the per-layer
        ``switch_layer`` is a few ``copy_`` calls and the forward thread's
        stall is bounded by the chunk, not by the model.

        Args:
            layer_ids: The layers to switch.
            new_rows: ``[len(layer_ids), num_physical_experts]`` logical ids
                on the host, one row per layer.
        """
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        new_rows = new_rows.to(device="cpu", dtype=torch.int64)
        if tuple(new_rows.shape) != (len(layer_ids), self.num_physical_experts):
            raise ValueError(
                f"new_rows has shape {tuple(new_rows.shape)}, expected "
                f"({len(layer_ids)}, {self.num_physical_experts})"
            )
        for layer_id in layer_ids:
            if not 0 <= layer_id < self.num_layers:
                raise ValueError(
                    f"layer {layer_id} is outside the placement's {self.num_layers}"
                )
        replicas = pad_replica_table(
            _compute_logical_to_all_physical_map(
                new_rows, num_logical_experts=self.num_logical_experts
            ),
            self.logical_to_all_physical_map.shape[-1],
        )
        num_valid = torch.count_nonzero(replicas != -1, dim=-1).to(torch.int32)
        rank_dispatch = None
        if self._rank_dispatch_map is not None:
            # Each (layer, expert) choice depends only on that layer's
            # replicas, so only the chunk's rows are derived; every rank runs
            # the same computation on the same rows.
            rank_dispatch = compute_logical_to_rank_dispatch_physical_map(
                logical_to_all_physical_map=replicas,
                num_physical_experts=self.num_physical_experts,
                ep_rank_nodes=self.ep_rank_nodes,
                ep_rank=self.ep_rank,
            )
        return PreparedPlacementRows(
            layer_ids=layer_ids,
            physical_to_logical=new_rows,
            replicas=replicas,
            num_valid=num_valid,
            rank_dispatch=rank_dispatch,
        )

    def switch_layer(self, prepared: PreparedPlacementRows, index: int) -> None:
        """Rewrite one prepared layer's rows in the existing tables.

        The router and the captured CUDA graphs hold views of these tables, so
        the switch is a ``copy_`` into their storage: the device
        ``physical_to_logical_map`` and replica table rows, the valid counts,
        this rank's static dispatch map when it has been materialized, then
        the host map the loader and the move planner read. Called on the
        forward thread only, after the layer's weights landed, so a forward
        never sees a layer whose table and slots disagree.

        Args:
            prepared: The chunk's rows from ``prepare_layers``.
            index: Position of the layer within ``prepared.layer_ids``.
        """
        layer_id = prepared.layer_ids[index]
        device = self.physical_to_logical_map.device
        self.physical_to_logical_map[layer_id].copy_(
            prepared.physical_to_logical[index].to(self.physical_to_logical_map.dtype)
        )
        self.logical_to_all_physical_map[layer_id].copy_(
            prepared.replicas[index].to(device)
        )
        self.logical_to_all_physical_map_num_valid[layer_id].copy_(
            prepared.num_valid[index].to(device)
        )
        if self._rank_dispatch_map is not None:
            if prepared.rank_dispatch is None:
                raise RuntimeError(
                    "the static dispatch map was materialized after the rows "
                    "were prepared; prepare the chunk again"
                )
            self._rank_dispatch_map[layer_id].copy_(
                prepared.rank_dispatch[index].to(device)
            )
        self.physical_to_logical_map_cpu[layer_id].copy_(
            prepared.physical_to_logical[index].to(
                self.physical_to_logical_map_cpu.dtype
            )
        )

    def update_layers(self, layer_ids: Sequence[int], new_rows: torch.Tensor) -> None:
        """Prepare and switch ``layer_ids`` in one go (``prepare_layers`` + ``switch_layer``)."""
        prepared = self.prepare_layers(layer_ids, new_rows)
        for index in range(len(prepared.layer_ids)):
            self.switch_layer(prepared, index)

    def snapshot_load(self) -> ExpertLoadSnapshot:
        """Read the route counters to the host and start a new window.

        Runs on the forward thread, on the execution stream the routing
        kernels bump the counters on: the read-back (a deliberate host wait)
        lands behind the in-flight forward and the zeroing lands before the
        next one, so the window boundary is exact at forward granularity.
        """
        if self.physical_load is None:
            raise RuntimeError("expert load recording is not enabled")
        physical = self.physical_load.to(device="cpu", dtype=torch.int64, copy=True)
        self.physical_load.zero_()
        return ExpertLoadSnapshot(
            physical_count=physical,
            physical_to_logical_map=self.physical_to_logical_map_cpu.clone(),
        )

    # -------------------------------- placement queries ------------------------------

    def local_slot_logical_experts(self, layer_id: int, ep_rank: int) -> list[int]:
        """Return the logical expert held by each of ``ep_rank``'s slots, in slot order."""
        local = self.num_local_physical_experts
        return self.physical_to_logical_map_cpu[
            layer_id, ep_rank * local : (ep_rank + 1) * local
        ].tolist()

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

    def rank_dispatch_map(self) -> torch.Tensor:
        """This rank's static dispatch map: the replica it sends each logical expert to.

        ``(layers, num_logical_experts)`` int32 on the placement's device,
        computed on first use (nearest replica: same GPU, then same node, else
        a seeded fair draw). Only all-to-all EP under a static dispatch
        algorithm consumes it; replicated-input EP routes through the replica
        table instead and never pays for it.
        """
        if self._rank_dispatch_map is None:
            self._rank_dispatch_map = compute_logical_to_rank_dispatch_physical_map(
                logical_to_all_physical_map=self.logical_to_all_physical_map,
                num_physical_experts=self.num_physical_experts,
                ep_rank_nodes=self.ep_rank_nodes,
                ep_rank=self.ep_rank,
            )
        return self._rank_dispatch_map

    # -------------------------------- load recording ---------------------------------

    def enable_load_recording(self, mapping: Mapping) -> None:
        """Allocate the per-physical-expert route counters the router bumps.

        int64: on replicated-input EP every rank counts every token's routes,
        and a long window on a hot expert overruns int32. The live-row mask
        that keeps padded forwards' filler rows out of the counters comes
        with them; its buffer is reserved before the first forward
        (``reserve_load_rows``).

        Args:
            mapping: The parallel layout, which lays out the ranks' rows in
                the MoE input.
        """
        self.physical_load = torch.zeros(
            (self.num_layers, self.num_physical_experts),
            dtype=torch.int64,
            device=self.physical_to_logical_map.device,
        )
        self.load_rows = ExpertLoadRowMask(mapping)

    def reserve_load_rows(self, max_rows: int) -> None:
        """Reserve the live-row mask for the largest MoE input any forward carries."""
        if self.load_rows is None:
            raise RuntimeError("expert load recording is not enabled")
        self.load_rows.reserve(max_rows, self.physical_to_logical_map.device)

    def reset_load(self) -> None:
        if self.physical_load is None:
            raise RuntimeError("expert load recording is not enabled")
        self.physical_load.zero_()

    def load_record(self, physical_load: torch.Tensor) -> dict[str, torch.Tensor | int]:
        """Package this rank's ``[layers, physical]`` count for ``--init-expert-location``.

        The record is per rank, unreduced: ``physical_count`` is what this
        rank's router counted (its own tokens under all-to-all EP, every
        token under replicated-input EP), ``logical_count`` the same summed
        over each logical expert's replicas, plus the placement that produced
        them and the rank's position in the EP group.
        ``merge_expert_load_records`` sums the ranks' records.
        """
        # A copy even on a CPU device: the record must outlive the next reset.
        physical = physical_load.to(device="cpu", dtype=torch.int64, copy=True)
        return {
            "physical_count": physical,
            "logical_count": logical_count_of(
                physical, self.physical_to_logical_map_cpu, self.num_logical_experts
            ),
            "physical_to_logical_map": self.physical_to_logical_map_cpu.clone(),
            "ep_rank": self.ep_rank,
            "ep_size": self.ep_size,
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
        # The inverse map is built on the host and moved once with the map.
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map.cpu(),
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
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
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        """Startup wrapper of ``compute_placement_maps`` over a recorded load."""
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 3:
            # Several recording windows: the algorithm balances their sum.
            logical_count = logical_count.sum(dim=0)
        logical_count = logical_count.cpu()

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.mapping.nnodes
        expected = (
            model_config_for_expert_location.num_layers,
            model_config_for_expert_location.num_logical_experts,
        )
        if tuple(logical_count.shape) != expected:
            raise ValueError(
                f"logical_count has shape {tuple(logical_count.shape)}; the model "
                f"has {expected[0]} MoE layers of {expected[1]} routed experts."
            )

        physical_to_logical_map, logical_to_all_physical_map = compute_placement_maps(
            logical_count,
            num_physical_experts=num_physical_experts,
            ep_size=common["ep_size"],
            num_groups=num_groups,
            num_nodes=num_nodes,
            algorithm=eplb_algorithms.compute_algorithm(
                raw_algorithm=server_args.eplb_algorithm,
                num_groups=num_groups,
                num_nodes=num_nodes,
            ),
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
        mapping = server_args.mapping
        return ExpertLocationMetadata.from_maps(
            physical_to_logical_map,
            logical_to_all_physical_map,
            ep_size=ep_size,
            ep_rank=mapping.moe.ep_rank,
            # The node of every EP rank, from the global ranks of this rank's
            # EP group: EP ranks are not spread evenly over nodes in general
            # (MoE TP, PP stages, ep_size=1 on a multi-node job).
            ep_rank_nodes=tuple(
                rank // mapping.nprocs_per_node for rank in mapping.moe.ep_group
            ),
        )

    @staticmethod
    def from_physical_to_logical_map(
        physical_to_logical_map: torch.Tensor,
        num_logical_experts: int,
        *,
        ep_size: int,
        ep_rank: int,
        ep_rank_nodes: Sequence[int],
    ) -> ExpertLocationMetadata:
        """Build a placement from ``[layers, physical]`` logical ids alone.

        Args:
            physical_to_logical_map: The logical expert held by every slot.
            num_logical_experts: Routed expert count ``E`` of the model.
            ep_size: Ranks the slots are spread over, contiguously.
            ep_rank: This rank, for the static dispatch map.
            ep_rank_nodes: The node of every EP rank, so the static map
                prefers same-node replicas.
        """
        return ExpertLocationMetadata.from_maps(
            physical_to_logical_map,
            _compute_logical_to_all_physical_map(
                physical_to_logical_map.cpu(), num_logical_experts=num_logical_experts
            ).to(physical_to_logical_map.device),
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_rank_nodes=ep_rank_nodes,
        )

    @staticmethod
    def from_maps(
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
        *,
        ep_size: int,
        ep_rank: int,
        ep_rank_nodes: Sequence[int],
    ) -> ExpertLocationMetadata:
        """Build a placement from its two maps (see ``from_physical_to_logical_map``)."""
        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map,
            logical_to_all_physical_map_num_valid=torch.count_nonzero(
                logical_to_all_physical_map != -1, dim=-1
            ),
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_rank_nodes=tuple(int(node) for node in ep_rank_nodes),
        )


def logical_count_of(
    physical_count: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
) -> torch.Tensor:
    """Sum a host ``[layers, physical]`` int64 count over each logical expert's replicas."""
    num_layers = physical_to_logical_map.shape[0]
    logical = torch.zeros((num_layers, num_logical_experts), dtype=torch.int64)
    logical.scatter_add_(1, physical_to_logical_map.long(), physical_count)
    return logical


def load_balancedness(physical_count: torch.Tensor, ep_size: int) -> torch.Tensor:
    """Per-layer mean rank load over the busiest rank's load, from a ``[layers, physical]`` count.

    1.0 is perfectly balanced; the busiest rank is the layer's critical path.
    Slots are owned contiguously, so the rank loads are the row's chunks.
    """
    num_layers = physical_count.shape[0]
    per_rank = physical_count.view(num_layers, ep_size, -1).sum(-1).double()
    return per_rank.mean(-1) / per_rank.max(-1).values.clamp_min(1)


def _compute_logical_to_all_physical_map(
    physical_to_logical_map: torch.Tensor, num_logical_experts: int
) -> torch.Tensor:
    """Invert a host ``[layers, physical]`` map into ``[layers, logical, X]`` (-1 padded)."""
    if physical_to_logical_map.device.type != "cpu":
        raise ValueError("the inverse map is built from a host tensor")
    num_layers, num_physical_experts = physical_to_logical_map.shape
    rows = physical_to_logical_map.tolist()

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]
    for layer_id, row in enumerate(rows):
        for physical_expert_id, logical_expert_id in enumerate(row):
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

    return torch.tensor(
        _pad_nested_array(logical_to_all_physical_map, pad_value=-1),
        dtype=torch.int32,
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
    num_physical_experts: int,
    ep_rank_nodes: Sequence[int],
    ep_rank: int,
    seed: int = 42,
) -> torch.Tensor:
    """Pick, for every rank, the replica it dispatches each logical expert to.

    Nearest first: a replica on the same GPU, then one on the same node, else a
    seeded fair draw over all replicas so the remote ranks spread evenly. The
    tiers are vectorized over the whole table; only the draws loop, over the
    (layer, expert) pairs that need one, in a fixed order so every rank
    derives the same map.

    Args:
        logical_to_all_physical_map: ``[layers, logical, X]`` replicas, -1 padded.
        num_physical_experts: ``P``, owned contiguously by the EP ranks.
        ep_rank_nodes: The node of every EP rank (its length is the EP size).
        ep_rank: The rank whose ``[layers, logical]`` slice to return.
        seed: Seed of the fair draws.

    Returns:
        ``ep_rank``'s slice, int32 on ``logical_to_all_physical_map``'s device.
    """
    num_gpus = len(ep_rank_nodes)
    if num_gpus <= 0 or num_physical_experts % num_gpus:
        raise ValueError(
            f"{num_physical_experts} physical experts do not divide over "
            f"{num_gpus} EP ranks"
        )
    if not 0 <= ep_rank < num_gpus:
        raise ValueError(f"ep_rank={ep_rank} is outside the {num_gpus} EP ranks")
    num_local = num_physical_experts // num_gpus
    nodes = torch.tensor(list(ep_rank_nodes), dtype=torch.int64)
    num_nodes = int(nodes.max().item()) + 1

    replicas = logical_to_all_physical_map.cpu().to(torch.int64)
    num_layers, num_logical_experts, max_replicas = replicas.shape
    valid = replicas >= 0
    gpu_of = torch.where(valid, replicas // num_local, -1)
    node_of = torch.where(valid, nodes[gpu_of.clamp_min(0)], -1)

    # For every (layer, expert): the column of the FIRST replica on each GPU
    # and on each node, -1 if none. Built by scattering the columns from last
    # to first, so the lowest column wins; max_replicas scatters over
    # [layers, experts, gpus] instead of a loop over the gpus.
    def first_replica_on(owner: torch.Tensor, count: int) -> torch.Tensor:
        first = torch.full(
            (num_layers, num_logical_experts, count), -1, dtype=torch.int64
        )
        for column in reversed(range(max_replicas)):
            index = owner[..., column].clamp_min(0).unsqueeze(-1)
            kept = first.gather(-1, index)
            first.scatter_(
                -1, index, torch.where(valid[..., column : column + 1], column, kept)
            )
        return first

    first_on_gpu = first_replica_on(gpu_of, num_gpus)  # [L, E, G]
    first_on_node = first_replica_on(node_of, num_nodes)[..., nodes]  # [L, E, G]
    column = torch.where(first_on_gpu >= 0, first_on_gpu, first_on_node)
    chosen = replicas.gather(-1, column.clamp_min(0))
    # A single replica serves everyone; otherwise only the nearest tiers.
    single = (valid.sum(-1) == 1).unsqueeze(-1)
    chosen = torch.where(single, replicas[..., :1], chosen)
    output = (
        torch.where((column >= 0) | single, chosen, -1).permute(2, 0, 1).contiguous()
    )

    r = random.Random(seed)
    # Pairs with a rank that has no same-node replica, in row-major order so
    # the draws are identical on every rank.
    for layer_id, logical_expert_id in (output == -1).any(0).nonzero().tolist():
        column = output[:, layer_id, logical_expert_id]
        unassigned = column == -1
        candidates = replicas[layer_id, logical_expert_id][
            valid[layer_id, logical_expert_id]
        ].tolist()
        column[unassigned] = torch.tensor(
            _fair_choices(candidates, k=int(unassigned.sum().item()), r=r),
            dtype=torch.int64,
        )

    return (
        output[ep_rank]
        .to(torch.int32)
        .contiguous()
        .to(logical_to_all_physical_map.device)
    )


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
    (``expert_placement_requested``). The model must opt in
    (``supports_expert_placement``): a placement only a model's MoE layers
    consume is otherwise a silent no-op, so any other model is refused. Load
    recording allocates the counters the router bumps.
    """
    if not expert_placement_requested(server_args):
        return None
    # Deferred: the model base imports the MoE layers, whose router imports
    # this module for its placement view.
    from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM

    model_class, architecture = get_model_architecture(model_config)
    if not (
        issubclass(model_class, BaseCausalLM) and model_class.supports_expert_placement
    ):
        raise ValueError(
            f"{architecture} does not route through an expert placement; "
            "--ep-num-redundant-experts, --init-expert-location, "
            "--ep-dispatch-algorithm and --expert-distribution-recorder-mode "
            "apply only to models that opt in (supports_expert_placement)."
        )
    geometry = ModelConfigForExpertLocation.from_model_config(model_config)
    if geometry.num_logical_experts <= 1:
        raise ValueError(
            "Expert placement was requested for a model without routed experts."
        )
    placement = compute_initial_expert_location_metadata(server_args, model_config)
    if server_args.expert_distribution_recorder_mode is not None:
        placement.enable_load_recording(server_args.mapping)
    logger.info(
        f"Expert placement: {placement.num_logical_experts} logical experts on "
        f"{placement.num_physical_experts} physical slots over "
        f"ep_size={placement.ep_size} "
        f"({placement.num_local_physical_experts} per rank), dispatch "
        f"{server_args.ep_dispatch_algorithm}, load recording "
        f"{'on' if placement.physical_load is not None else 'off'}"
    )
    return placement


def merge_expert_load_records(paths: Sequence[str | Path]) -> dict:
    """Sum the per-rank load records of one profile window.

    Every rank writes its own counters (``ExpertLocationMetadata.load_record``);
    the sum over the EP group is the layer's load. Under all-to-all EP each
    rank counted its own tokens, so all ranks' records are needed; under
    replicated-input EP every rank counted every token, so one record is
    complete and summing more only scales the counts uniformly, which the
    placement algorithm and the balancedness ratio are blind to.

    Returns:
        ``logical_count`` (what ``init_by_eplb`` consumes), the summed
        ``physical_count``, the ``physical_to_logical_map`` and ``ep_size`` the
        records share, the ``ep_ranks`` merged, ``rank_count`` (``[layers,
        ep]`` routes per rank) and the per-layer ``balancedness`` (mean rank
        load over the busiest rank's load).
    """
    if not paths:
        raise ValueError("no expert load records to merge")
    physical_count: torch.Tensor | None = None
    physical_to_logical_map: torch.Tensor | None = None
    ep_size: int | None = None
    ep_ranks: list[int] = []
    for path in paths:
        record = torch.load(path, weights_only=True)
        if physical_count is None:
            physical_count = torch.zeros_like(
                record["physical_count"], dtype=torch.int64
            )
            physical_to_logical_map = record["physical_to_logical_map"]
            ep_size = int(record["ep_size"])
        elif not torch.equal(
            record["physical_to_logical_map"], physical_to_logical_map
        ):
            raise ValueError(
                f"{path} was recorded under a different expert placement than "
                f"{paths[0]}; merge records of one serving run only"
            )
        elif int(record["ep_size"]) != ep_size:
            raise ValueError(
                f"{path} was recorded with ep_size {record['ep_size']}, not {ep_size}"
            )
        physical_count += record["physical_count"].to(torch.int64)
        ep_ranks.append(int(record["ep_rank"]))
    num_layers = physical_to_logical_map.shape[0]
    return {
        # Every logical expert has a slot, so the map's largest id is E - 1.
        "logical_count": logical_count_of(
            physical_count,
            physical_to_logical_map,
            int(physical_to_logical_map.max().item()) + 1,
        ),
        "physical_count": physical_count,
        "physical_to_logical_map": physical_to_logical_map,
        "ep_size": ep_size,
        "ep_ranks": sorted(ep_ranks),
        "rank_count": physical_count.view(num_layers, ep_size, -1).sum(-1),
        "balancedness": load_balancedness(physical_count, ep_size),
    }


InitExpertLocationForm = Literal["trivial", "json", "directory", "file", "glob"]


def init_expert_location_form(data: str) -> InitExpertLocationForm:
    """Decide what ``--init-expert-location`` names, in one fixed order.

    ``trivial`` is the identity placement; a string starting with ``{`` is
    inline JSON (a bare map or a load record); an existing directory holds
    the per-rank ``*.expert-load.pt`` records of one profile; an existing
    file is one ``.pt`` or ``.json`` record or map; anything else is a glob
    over record files. Deciding by form (not by the characters a glob would
    use) keeps inline JSON with ``[`` or ``?`` inside from being read as a
    pattern.
    """
    if data == "trivial":
        return "trivial"
    if data.lstrip().startswith("{"):
        return "json"
    path = Path(data)
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "glob"


def _expert_load_record_paths(data: str, form: InitExpertLocationForm) -> list[str]:
    """The record files a ``directory`` or ``glob`` form names (at least one)."""
    if form == "directory":
        paths = sorted(str(p) for p in Path(data).glob(f"*{EXPERT_LOAD_RECORD_SUFFIX}"))
        if not paths:
            raise ValueError(f"{data} holds no *{EXPERT_LOAD_RECORD_SUFFIX} records")
        return paths
    if form == "glob":
        paths = sorted(glob.glob(data))
        if not paths:
            raise ValueError(
                f"--init-expert-location {data!r} is not 'trivial', inline JSON, "
                "a directory or a file, and matches no expert load record as a glob"
            )
        return paths
    raise ValueError(f"{form} names no record files")


def compute_initial_expert_location_metadata(
    server_args: ServerArgs, model_config: ModelConfig
) -> ExpertLocationMetadata:
    data = server_args.init_expert_location
    form = init_expert_location_form(data)
    if form == "trivial":
        return ExpertLocationMetadata.init_trivial(server_args, model_config)

    if form in ("directory", "glob"):
        record_paths = _expert_load_record_paths(data, form)
        logger.info(
            f"init_expert_location: EPLB placement from {len(record_paths)} merged "
            f"expert load records in {data}"
        )
        return ExpertLocationMetadata.init_by_eplb(
            server_args,
            model_config,
            logical_count=merge_expert_load_records(record_paths)["logical_count"],
        )

    if form == "json":
        data_dict = json.loads(data)
    elif data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        raise ValueError(
            f"--init-expert-location file {data!r} must be a .pt or .json "
            "record or placement map"
        )

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
