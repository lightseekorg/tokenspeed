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

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader


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


@dataclass(frozen=True)
class FusedExpertWeightPlanEntry(CheckpointPlanEntry):
    split_dim: int | None = None
    split_chunks: int | None = None
    split_index: int | None = None


class MoECheckpointLoadError(RuntimeError):
    pass


def _build_default_expert_plan(
    schema: ExpertCheckpointSchema,
    *,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
) -> list[ExpertWeightPlanEntry]:
    # Expert ownership is assumed to be a contiguous per-rank range here.
    # EPLB-aware remapping would need a different planning step.
    num_local_experts = num_experts // ep_size
    start_expert = num_local_experts * ep_rank
    expert_plan: list[ExpertWeightPlanEntry] = []
    for local_expert_id in range(num_local_experts):
        expert_id = start_expert + local_expert_id
        expert_plan.extend(
            (
                ExpertWeightPlanEntry(
                    param_name="experts.w13_",
                    checkpoint_weight_name=schema.make_expert_weight_name(
                        expert_id, "gate_proj"
                    ),
                    shard_id="w1",
                    local_expert_id=local_expert_id,
                ),
                ExpertWeightPlanEntry(
                    param_name="experts.w13_",
                    checkpoint_weight_name=schema.make_expert_weight_name(
                        expert_id, "up_proj"
                    ),
                    shard_id="w3",
                    local_expert_id=local_expert_id,
                ),
                ExpertWeightPlanEntry(
                    param_name="experts.w2_",
                    checkpoint_weight_name=schema.make_expert_weight_name(
                        expert_id, "down_proj"
                    ),
                    shard_id="w2",
                    local_expert_id=local_expert_id,
                ),
            )
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


def _load_fused_expert_tensor(
    param,
    loaded_weight,
    *,
    shard_id: str,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
) -> None:
    # Expert ownership is assumed to be a contiguous per-rank range here.
    # EPLB-aware remapping would need a different loading step.
    num_local_experts = num_experts // ep_size
    start_expert = num_local_experts * ep_rank
    end_expert = start_expert + num_local_experts
    weight_loader = param.weight_loader
    for expert_id in range(start_expert, end_expert):
        local_expert_id = expert_id - start_expert
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
        fused_plan: Sequence[FusedExpertWeightPlanEntry] = (),
        num_experts: int | None = None,
        ep_rank: int = 0,
        ep_size: int = 1,
        fused_load_style: str = "per_expert",
        transpose_local_tensor_non_bias: bool = False,
    ) -> None:
        self._params_dict = params_dict
        self._expert_plan = tuple(expert_plan)
        self._fused_plan = tuple(fused_plan)
        self._num_experts = num_experts
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._fused_load_style = fused_load_style
        self._transpose_local_tensor_non_bias = transpose_local_tensor_non_bias

        if self._fused_plan and self._num_experts is None:
            raise ValueError("num_experts is required when fused_plan is used")
        if fused_load_style not in {"per_expert", "local_tensor"}:
            raise ValueError(f"Unknown fused_load_style: {fused_load_style}")

    @staticmethod
    def _matches_plan(plan: Sequence[CheckpointPlanEntry], name: str) -> bool:
        return any(plan_entry.matches(name) for plan_entry in plan)

    def matches(self, name: str) -> bool:
        return self._matches_plan(self._fused_plan, name) or self._matches_plan(
            self._expert_plan, name
        )

    def _load_expert(self, name: str, loaded_weight: torch.Tensor) -> str | None:
        mapped_name: str | None = None
        for plan_entry in self._expert_plan:
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
                local_expert_id=plan_entry.local_expert_id,
            )
            return mapped_name

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

            tensor_to_load = loaded_weight
            if plan_entry.split_dim is not None:
                tensor_to_load = loaded_weight.chunk(
                    plan_entry.split_chunks, dim=plan_entry.split_dim
                )[plan_entry.split_index]

            if self._fused_load_style == "per_expert":
                _load_fused_expert_tensor(
                    param,
                    tensor_to_load,
                    shard_id=plan_entry.shard_id,
                    num_experts=self._num_experts,
                    ep_rank=self._ep_rank,
                    ep_size=self._ep_size,
                )
            else:
                if self._transpose_local_tensor_non_bias and "bias" not in mapped_name:
                    tensor_to_load = tensor_to_load.transpose(-2, -1)

                local_num_experts = param.shape[0]
                assert local_num_experts * self._ep_size == tensor_to_load.shape[0]
                local_experts = tensor_to_load[
                    local_num_experts
                    * self._ep_rank : local_num_experts
                    * (self._ep_rank + 1)
                ]
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
) -> MoECheckpointLoader:
    expert_plan: Sequence[ExpertWeightPlanEntry] = ()
    if expert_schema is not None:
        if num_experts is None:
            raise ValueError("num_experts is required when expert_schema is used")
        expert_plan = _build_default_expert_plan(
            expert_schema,
            num_experts=num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
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
        fused_plan=fused_plan,
        num_experts=num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        fused_load_style=fused_load_style,
        transpose_local_tensor_non_bias=transpose_local_tensor_non_bias,
    )
