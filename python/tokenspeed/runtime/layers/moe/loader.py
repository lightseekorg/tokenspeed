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

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.utils import get_layer_id
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata


@dataclass(frozen=True)
class CheckpointPlanEntry:
    param_name: str
    checkpoint_weight_name: str
    shard_id: str

    def matches(self, checkpoint_name: str) -> bool:
        return self.checkpoint_weight_name in checkpoint_name

    def resolve_param_name(self, checkpoint_name: str) -> str:
        return checkpoint_name.replace(self.checkpoint_weight_name, self.param_name)


@dataclass(frozen=True)
class ExpertWeightPlanEntry(CheckpointPlanEntry):
    local_expert_id: int
    # The decoder layer this entry serves, or None when every layer shares it
    # (contiguous ownership). An expert placement assigns slots per layer.
    # Explicit at every construction: which of the two it is selects how the
    # entry matches checkpoint names.
    layer_id: int | None

    def matches(self, checkpoint_name: str) -> bool:
        return super().matches(checkpoint_name) and (
            self.layer_id is None or get_layer_id(checkpoint_name) == self.layer_id
        )


@dataclass(frozen=True)
class FusedExpertWeightPlanEntry(CheckpointPlanEntry):
    split_dim: int | None = None
    split_chunks: int | None = None
    split_index: int | None = None


class MoECheckpointLoadError(RuntimeError):
    pass


def _ep_partition(num_experts: int, ep_rank: int, ep_size: int) -> int:
    if not (
        num_experts > 0
        and ep_size > 0
        and num_experts % ep_size == 0
        and 0 <= ep_rank < ep_size
    ):
        raise ValueError("experts must divide evenly across valid EP ranks")
    return num_experts // ep_size


def _expert_shards(schema: ExpertCheckpointSchema) -> tuple[tuple[str, str, str], ...]:
    """(param prefix, checkpoint semantic, shard id) of one expert's projections."""
    if schema.gate_proj_name is None:
        w13 = (("experts.w13_", "up_proj", "w13"),)
    else:
        w13 = (
            ("experts.w13_", "gate_proj", "w1"),
            ("experts.w13_", "up_proj", "w3"),
        )
    return (*w13, ("experts.w2_", "down_proj", "w2"))


def _build_default_expert_plan(
    schema: ExpertCheckpointSchema,
    *,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
) -> list[ExpertWeightPlanEntry]:
    # Contiguous ownership: rank r holds logical experts [r*n, (r+1)*n) of
    # every layer. ``_build_placed_expert_plan`` is the placement-aware twin.
    num_local_experts = _ep_partition(num_experts, ep_rank, ep_size)
    start_expert = num_local_experts * ep_rank
    expert_plan: list[ExpertWeightPlanEntry] = []
    for local_expert_id in range(num_local_experts):
        expert_id = start_expert + local_expert_id
        expert_plan.extend(
            ExpertWeightPlanEntry(
                param_name=param_name,
                checkpoint_weight_name=schema.make_expert_weight_name(
                    expert_id, semantic
                ),
                shard_id=shard_id,
                local_expert_id=local_expert_id,
                # Contiguous ownership: the same slot serves every layer.
                layer_id=None,
            )
            for param_name, semantic, shard_id in _expert_shards(schema)
        )
    return expert_plan


def _build_placed_expert_plan(
    schema: ExpertCheckpointSchema,
    *,
    expert_placement: ExpertLocationMetadata,
    ep_rank: int,
) -> list[ExpertWeightPlanEntry]:
    """Plan one entry per (layer, local slot, projection) from the placement.

    Slot ``s`` of this rank in layer ``L`` holds logical expert
    ``physical_to_logical_map_cpu[L, ep_rank * n + s]``; a logical expert
    with several local replicas yields one entry per slot, all matching the
    same checkpoint tensor.
    """
    expert_plan: list[ExpertWeightPlanEntry] = []
    for layer_id in range(expert_placement.num_layers):
        for local_expert_id, expert_id in enumerate(
            expert_placement.local_slot_logical_experts(layer_id, ep_rank)
        ):
            expert_plan.extend(
                ExpertWeightPlanEntry(
                    param_name=param_name,
                    checkpoint_weight_name=schema.make_expert_weight_name(
                        expert_id, semantic
                    ),
                    shard_id=shard_id,
                    local_expert_id=local_expert_id,
                    layer_id=layer_id,
                )
                for param_name, semantic, shard_id in _expert_shards(schema)
            )
    return expert_plan


def _build_global_expert_name_plan(
    schema: ExpertCheckpointSchema,
    *,
    num_experts: int,
) -> list[CheckpointPlanEntry]:
    expert_plan: list[CheckpointPlanEntry] = []
    for expert_id in range(num_experts):
        expert_plan.extend(
            CheckpointPlanEntry(
                param_name=param_name,
                checkpoint_weight_name=schema.make_expert_weight_name(
                    expert_id, semantic
                ),
                shard_id=shard_id,
            )
            for param_name, semantic, shard_id in _expert_shards(schema)
        )
    return expert_plan


def _build_default_fused_plan(
    schema: ExpertCheckpointSchema,
    *,
    fused_gate_up_as_w13: bool = False,
    include_bias: bool = False,
) -> list[FusedExpertWeightPlanEntry]:
    if fused_gate_up_as_w13:
        fused_plan = [
            FusedExpertWeightPlanEntry(
                param_name="experts.w13_weight",
                checkpoint_weight_name=(
                    f"experts.{schema.get_semantic_name('gate_up_fused')}"
                ),
                shard_id="w13",
            ),
            FusedExpertWeightPlanEntry(
                param_name="experts.w2_weight",
                checkpoint_weight_name=f"experts.{schema.get_semantic_name('down_proj')}",
                shard_id="w2",
            ),
        ]
        if include_bias:
            fused_plan.extend(
                (
                    FusedExpertWeightPlanEntry(
                        param_name="experts.w13_weight_bias",
                        checkpoint_weight_name=(
                            f"experts.{schema.get_semantic_name('gate_up_bias')}"
                        ),
                        shard_id="w13",
                    ),
                    FusedExpertWeightPlanEntry(
                        param_name="experts.w2_weight_bias",
                        checkpoint_weight_name=(
                            f"experts.{schema.get_semantic_name('down_bias')}"
                        ),
                        shard_id="w2",
                    ),
                )
            )
        return fused_plan

    fused_plan = [
        FusedExpertWeightPlanEntry(
            param_name="experts.w13_weight",
            checkpoint_weight_name=f"experts.{schema.get_semantic_name('gate_up_fused')}",
            shard_id="w1",
            split_dim=-2,
            split_chunks=2,
            split_index=0,
        ),
        FusedExpertWeightPlanEntry(
            param_name="experts.w13_weight",
            checkpoint_weight_name=f"experts.{schema.get_semantic_name('gate_up_fused')}",
            shard_id="w3",
            split_dim=-2,
            split_chunks=2,
            split_index=1,
        ),
        FusedExpertWeightPlanEntry(
            param_name="experts.w2_weight",
            checkpoint_weight_name=f"experts.{schema.get_semantic_name('down_proj')}",
            shard_id="w2",
        ),
    ]
    if include_bias:
        fused_plan.extend(
            (
                FusedExpertWeightPlanEntry(
                    param_name="experts.w13_weight_bias",
                    checkpoint_weight_name=(
                        f"experts.{schema.get_semantic_name('gate_up_bias')}"
                    ),
                    shard_id="w1",
                    split_dim=-1,
                    split_chunks=2,
                    split_index=0,
                ),
                FusedExpertWeightPlanEntry(
                    param_name="experts.w13_weight_bias",
                    checkpoint_weight_name=(
                        f"experts.{schema.get_semantic_name('gate_up_bias')}"
                    ),
                    shard_id="w3",
                    split_dim=-1,
                    split_chunks=2,
                    split_index=1,
                ),
                FusedExpertWeightPlanEntry(
                    param_name="experts.w2_weight_bias",
                    checkpoint_weight_name=f"experts.{schema.get_semantic_name('down_bias')}",
                    shard_id="w2",
                ),
            )
        )
    return fused_plan


def _local_slot_experts(
    *,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    expert_placement: ExpertLocationMetadata | None,
    layer_id: int | None,
) -> list[int]:
    """The logical expert held by each of this rank's slots, in slot order."""
    if expert_placement is None:
        num_local_experts = _ep_partition(num_experts, ep_rank, ep_size)
        start_expert = num_local_experts * ep_rank
        return list(range(start_expert, start_expert + num_local_experts))
    if layer_id is None:
        raise MoECheckpointLoadError(
            "an expert placement assigns slots per layer, but the checkpoint "
            "tensor name carries no 'layers.<n>.' index"
        )
    return expert_placement.local_slot_logical_experts(layer_id, ep_rank)


def _select_local_experts(
    stacked: torch.Tensor, slot_experts: list[int]
) -> torch.Tensor:
    """Rows of a ``[logical experts, ...]`` tensor for this rank's slots.

    A contiguous ascending range slices (a view, no copy); anything else --
    a placement replicating or reordering experts -- gathers.
    """
    if not slot_experts:
        return stacked[0:0]
    first = slot_experts[0]
    if slot_experts == list(range(first, first + len(slot_experts))):
        return stacked[first : first + len(slot_experts)]
    return stacked[slot_experts]


def _load_fused_expert_tensor(
    param,
    loaded_weight,
    *,
    shard_id: str,
    slot_experts: list[int],
) -> None:
    """Load ``loaded_weight[expert]`` into every local slot, from a fused tensor."""
    weight_loader = param.weight_loader
    for local_expert_id, expert_id in enumerate(slot_experts):
        weight_loader(
            param,
            loaded_weight[expert_id],
            shard_id=shard_id,
            local_expert_id=local_expert_id,
        )


class MoECheckpointLoader:
    def __init__(
        self,
        *,
        params_dict: dict[str, nn.Parameter],
        expert_plan: Sequence[ExpertWeightPlanEntry] = (),
        global_expert_plan: Sequence[CheckpointPlanEntry] = (),
        fused_plan: Sequence[FusedExpertWeightPlanEntry] = (),
        num_experts: int | None = None,
        ep_rank: int = 0,
        ep_size: int = 1,
        fused_load_style: str = "per_expert",
        transpose_local_tensor_non_bias: bool = False,
        expert_placement: ExpertLocationMetadata | None = None,
    ) -> None:
        self._params_dict = params_dict
        self._expert_plan = tuple(expert_plan)
        self._global_expert_plan = tuple(global_expert_plan)
        self._fused_plan = tuple(fused_plan)
        self._num_experts = num_experts
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._fused_load_style = fused_load_style
        self._transpose_local_tensor_non_bias = transpose_local_tensor_non_bias
        # None: contiguous ownership. A placement assigns this rank's slots
        # per layer, and a logical expert may fill several of them.
        self._expert_placement = expert_placement
        # Entries by the layer they serve (None: every layer), so a name is
        # matched against its own layer's slots rather than every layer's.
        by_layer: dict[int | None, list[ExpertWeightPlanEntry]] = defaultdict(list)
        for plan_entry in self._expert_plan:
            by_layer[plan_entry.layer_id].append(plan_entry)
        self._expert_plan_by_layer: dict[
            int | None, tuple[ExpertWeightPlanEntry, ...]
        ] = {layer_id: tuple(entries) for layer_id, entries in by_layer.items()}

        if self._fused_plan and self._num_experts is None:
            raise ValueError("num_experts is required when fused_plan is used")
        if fused_load_style not in {"per_expert", "local_tensor"}:
            raise ValueError(f"Unknown fused_load_style: {fused_load_style}")

    @staticmethod
    def _matches_plan(plan: Sequence[CheckpointPlanEntry], name: str) -> bool:
        return any(plan_entry.matches(name) for plan_entry in plan)

    def _expert_plan_for(self, name: str) -> tuple[ExpertWeightPlanEntry, ...]:
        """The per-expert entries that may serve ``name``: the layer-agnostic
        ones plus those of the layer named in ``name``."""
        shared = self._expert_plan_by_layer.get(None, ())
        if len(self._expert_plan_by_layer) == 1 and shared:
            return shared
        return shared + self._expert_plan_by_layer.get(get_layer_id(name), ())

    def matches(self, name: str) -> bool:
        plan = (
            self._global_expert_plan
            if name.endswith(".input_scale")
            else self._expert_plan_for(name)
        )
        return self._matches_plan(self._fused_plan, name) or self._matches_plan(
            plan, name
        )

    def is_expert_checkpoint_weight(self, name: str) -> bool:
        """Return whether ``name`` belongs to this loader's MoE checkpoint schema.

        Args:
            name: Checkpoint tensor name after any model-specific remapping.

        Returns:
            ``True`` for local, non-local, or fused expert checkpoint tensors
            that this loader is responsible for; ``False`` for unrelated
            checkpoint tensors.
        """
        return self._matches_plan(self._fused_plan, name) or self._matches_plan(
            self._global_expert_plan, name
        )

    def _load_expert(self, name: str, loaded_weight: torch.Tensor) -> str | None:
        input_scale = name.endswith(".input_scale")
        plan = self._global_expert_plan if input_scale else self._expert_plan_for(name)
        mapped_name: str | None = None
        loaded_name: str | None = None
        # Every matching entry loads: under an expert placement one logical
        # expert's tensor fills each local slot that replicates it.
        for plan_entry in plan:
            if not plan_entry.matches(name):
                continue

            mapped_name = plan_entry.resolve_param_name(name)
            param = self._params_dict.get(mapped_name)
            if param is None:
                continue

            param.weight_loader(
                param,
                loaded_weight,
                shard_id=plan_entry.shard_id,
                local_expert_id=None if input_scale else plan_entry.local_expert_id,
            )
            loaded_name = mapped_name
            if input_scale:
                return loaded_name

        if loaded_name is not None:
            return loaded_name
        if mapped_name is not None:
            self._raise_unloaded_match(name, mapped_name)
        return None

    @staticmethod
    def _raise_unloaded_match(name: str, mapped_name: str | None) -> None:
        if mapped_name is None:
            raise MoECheckpointLoadError(
                f"Matched MoE checkpoint mapping for {name!r} but did not load any parameter"
            )
        raise MoECheckpointLoadError(
            f"Matched MoE checkpoint mapping for {name!r} -> {mapped_name!r}, "
            "but the target parameter was not found or no tensor was loaded"
        )

    @staticmethod
    def _raise_unmatched(name: str) -> None:
        raise MoECheckpointLoadError(
            f"{name!r} does not match any MoE checkpoint mapping"
        )

    def _load_fused(self, name: str, loaded_weight: torch.Tensor) -> str | None:
        matched_entries = [
            plan_entry for plan_entry in self._fused_plan if plan_entry.matches(name)
        ]
        if not matched_entries:
            return None

        selected_checkpoint_weight_name = max(
            (plan_entry.checkpoint_weight_name for plan_entry in matched_entries),
            key=len,
        )

        loaded_any = False
        mapped_name: str | None = None

        for plan_entry in matched_entries:
            if plan_entry.checkpoint_weight_name != selected_checkpoint_weight_name:
                continue

            mapped_name = plan_entry.resolve_param_name(name)
            param = self._params_dict.get(mapped_name)
            if param is None:
                continue

            if mapped_name.endswith("_input_scale"):
                param.weight_loader(
                    param,
                    loaded_weight,
                    shard_id=plan_entry.shard_id,
                    local_expert_id=None,
                )
                loaded_any = True
                continue

            tensor_to_load = loaded_weight
            if plan_entry.split_dim is not None:
                tensor_to_load = loaded_weight.chunk(
                    plan_entry.split_chunks, dim=plan_entry.split_dim
                )[plan_entry.split_index]

            slot_experts = _local_slot_experts(
                num_experts=self._num_experts,
                ep_rank=self._ep_rank,
                ep_size=self._ep_size,
                expert_placement=self._expert_placement,
                layer_id=get_layer_id(name),
            )
            if self._fused_load_style == "per_expert":
                _load_fused_expert_tensor(
                    param,
                    tensor_to_load,
                    shard_id=plan_entry.shard_id,
                    slot_experts=slot_experts,
                )
            else:
                if self._transpose_local_tensor_non_bias and "bias" not in mapped_name:
                    tensor_to_load = tensor_to_load.transpose(-2, -1)

                local_num_experts = param.shape[0]
                if local_num_experts != len(slot_experts):
                    raise MoECheckpointLoadError(
                        f"{mapped_name} holds {local_num_experts} local experts, "
                        f"the loader plans {len(slot_experts)} slots"
                    )
                # A fused checkpoint tensor stacks every logical expert. This
                # rank's slots are a view when they form a contiguous range
                # (always without a placement); a placement that repeats or
                # reorders experts needs the gather.
                local_experts = _select_local_experts(tensor_to_load, slot_experts)
                if getattr(param, "block_scale_inv", None) is not None:
                    raise MoECheckpointLoadError(
                        f"{mapped_name} needs online FP8 block quantization, "
                        "which the fused local-tensor loader cannot apply; load a "
                        "pre-quantized checkpoint or drop --quantization fp8."
                    )
                if tensor_to_load.dtype == torch.float8_e5m2:
                    default_weight_loader(param, local_experts.to(torch.bfloat16))
                else:
                    default_weight_loader(param, local_experts)

            loaded_any = True

        if not loaded_any:
            assert mapped_name is not None
            self._raise_unloaded_match(name, mapped_name)
        return mapped_name

    def load(self, name: str, loaded_weight: torch.Tensor) -> str:
        fused_mapped_name = self._load_fused(name, loaded_weight)
        if fused_mapped_name is not None:
            return fused_mapped_name

        expert_mapped_name = self._load_expert(name, loaded_weight)
        if expert_mapped_name is not None:
            return expert_mapped_name

        self._raise_unmatched(name)


def build_moe_checkpoint_loader(
    *,
    params_dict: dict[str, nn.Parameter],
    expert_schema: ExpertCheckpointSchema | None = None,
    fused_schema: ExpertCheckpointSchema | None = None,
    num_experts: int | None = None,
    ep_rank: int = 0,
    ep_size: int = 1,
    fused_gate_up_as_w13: bool = False,
    include_bias: bool = False,
    fused_load_style: str = "per_expert",
    transpose_local_tensor_non_bias: bool = False,
    expert_placement: ExpertLocationMetadata | None = None,
) -> MoECheckpointLoader:
    """Build the loader for a model's routed experts.

    Args:
        expert_placement: The model's expert placement, or None for contiguous
            ownership of ``num_experts`` logical experts. With a placement
            ``num_experts`` is the physical slot count (routed + redundant)
            and each layer's local slots are filled from the logical experts
            the placement assigns them, replicas included.
    """
    if expert_placement is not None:
        if num_experts != expert_placement.num_physical_experts:
            raise ValueError(
                f"num_experts={num_experts} must be the placement's "
                f"{expert_placement.num_physical_experts} physical experts"
            )
        if ep_size != expert_placement.ep_size:
            raise ValueError(
                f"ep_size={ep_size} differs from the placement's {expert_placement.ep_size}"
            )
    expert_plan: Sequence[ExpertWeightPlanEntry] = ()
    global_expert_plan: Sequence[CheckpointPlanEntry] = ()
    if expert_schema is not None:
        if num_experts is None:
            raise ValueError("num_experts is required when expert_schema is used")
        if expert_placement is None:
            expert_plan = _build_default_expert_plan(
                expert_schema,
                num_experts=num_experts,
                ep_rank=ep_rank,
                ep_size=ep_size,
            )
        else:
            expert_plan = _build_placed_expert_plan(
                expert_schema,
                expert_placement=expert_placement,
                ep_rank=ep_rank,
            )
        global_expert_plan = _build_global_expert_name_plan(
            expert_schema,
            num_experts=(
                num_experts
                if expert_placement is None
                else expert_placement.num_logical_experts
            ),
        )

    fused_plan: Sequence[FusedExpertWeightPlanEntry] = ()
    if fused_schema is not None:
        fused_plan = _build_default_fused_plan(
            fused_schema,
            fused_gate_up_as_w13=fused_gate_up_as_w13,
            include_bias=include_bias,
        )

    return MoECheckpointLoader(
        params_dict=params_dict,
        expert_plan=expert_plan,
        global_expert_plan=global_expert_plan,
        fused_plan=fused_plan,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        fused_load_style=fused_load_style,
        transpose_local_tensor_non_bias=transpose_local_tensor_non_bias,
        expert_placement=expert_placement,
    )
