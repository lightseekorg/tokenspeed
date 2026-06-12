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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from tokenspeed_kernel.thirdparty.cuda import routing_flash as cuda_routing_flash
from tokenspeed_kernel.thirdparty.triton import minimax_biased_grouped_topk

from tokenspeed.runtime.moe.distribution_recorder import (
    get_global_expert_distribution_recorder,
)


class TopKOutputFormat(Enum):
    STANDARD = auto()
    BYPASSED = auto()

    def is_standard(self) -> bool:
        return self == TopKOutputFormat.STANDARD

    def is_bypassed(self) -> bool:
        return self == TopKOutputFormat.BYPASSED


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal[
        "static",
        "dynamic",
        "fake",
        "static_with_zero_expert",
        "dynamic_with_zero_expert",
    ]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: torch.Tensor | None
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int

    @classmethod
    def init_new(
        cls,
        layer_id: int,
        ep_dispatch_algorithm: str | None = None,
        expert_location_metadata: Any | None = None,
    ):
        if ep_dispatch_algorithm is None:
            return None

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor | None,
    info: ExpertLocationDispatchInfo | None,
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits = torch.randn_like(router_logits)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo | None,
    num_experts: int | None = None,
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]
    if info.ep_dispatch_algorithm == "static_with_zero_expert":
        assert num_experts is not None
        return _topk_ids_logical_to_physical_static_with_zero_expert(
            topk_ids, info, num_experts
        )
    if info.ep_dispatch_algorithm == "dynamic_with_zero_expert":
        assert num_experts is not None
        return _topk_ids_logical_to_physical_dynamic_with_zero_expert(
            topk_ids, info, num_experts
        )
    if info.ep_dispatch_algorithm in {"dynamic", "fake"}:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static_with_zero_expert(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
    num_experts: int,
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    topk_ids = topk_ids.flatten()
    mask_less_than_num_experts = topk_ids < num_experts
    converted_part = info.partial_logical_to_rank_dispatch_physical_map[
        topk_ids[mask_less_than_num_experts]
    ]
    topk_ids[mask_less_than_num_experts] = converted_part
    return topk_ids.view(topk_ids_original_shape)


def _topk_ids_logical_to_physical_dynamic_with_zero_expert(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
    num_experts: int,
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    mask_less_than_num_experts = topk_ids < num_experts
    topk_ids_to_convert = topk_ids[mask_less_than_num_experts]

    chosen_dispatch_index = (
        torch.randint(
            0, 65536, topk_ids_to_convert.shape, dtype=torch.int32, device=device
        )
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids_to_convert]
    )
    converted_topk_ids = info.partial_logical_to_all_physical_map[
        topk_ids_to_convert, chosen_dispatch_index
    ]
    topk_ids[mask_less_than_num_experts] = converted_topk_ids
    return topk_ids.view(topk_ids_original_shape)


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    return topk_ids.view(topk_ids_original_shape)


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: torch.Tensor | None = None,
):
    if num_token_non_padded is None:
        return
    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


def torch_native_fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
):
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = gating_output.softmax(dim=-1)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        topk_weights = F.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float | None = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        factor = routed_scaling_factor or 1.0
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if routed_scaling_factor is not None:
            topk_weights *= routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@dataclass
class TopKConfig:
    top_k: int
    use_grouped_topk: bool = False
    topk_group: int | None = None
    num_expert_group: int | None = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Callable | None = None
    correction_bias: torch.Tensor | None = None
    torch_native: bool = False
    routed_scaling_factor: float | None = None
    output_format: TopKOutputFormat | None = None
    zero_expert_num: int | None = 0
    topk_indices_dtype: torch.dtype | None = torch.int32


class StandardTopKOutput(NamedTuple):
    """Standard top-k output format."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


class BypassedTopKOutput(NamedTuple):
    """Bypassed top-k output format."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: TopKConfig
    num_token_non_padded: torch.Tensor | None = None
    expert_location_dispatch_info: ExpertLocationDispatchInfo | None = None

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED


@runtime_checkable
class TopKOutput(Protocol):
    """Protocol for top-k outputs in different formats."""

    @property
    def format(self) -> TopKOutputFormat:
        """The format of the output."""
        ...


class TopK(torch.nn.Module):

    def __init__(
        self,
        top_k: int,
        *,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        renormalize: bool = True,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Callable | None = None,
        correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        output_format: TopKOutputFormat | None = None,
        zero_expert_num: int | None = 0,
        topk_indices_dtype=torch.int32,
    ):
        super().__init__()

        if use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None

        self.topk_config = TopKConfig(
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            output_format=output_format,
            zero_expert_num=zero_expert_num,
            topk_indices_dtype=topk_indices_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: torch.Tensor | None = None,
        expert_location_dispatch_info: ExpertLocationDispatchInfo | None = None,
    ) -> TopKOutput:
        if self.topk_config.output_format is not None:
            output_format = self.topk_config.output_format
        else:
            output_format = TopKOutputFormat.STANDARD

        if output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        else:
            self.topk_config.torch_native = False
            return select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )

    def empty_topk_output(
        self,
        device: torch.device,
        *,
        hidden_states: torch.Tensor | None = None,
        router_logits: torch.Tensor | None = None,
    ) -> TopKOutput:
        output_format = self.topk_config.output_format or TopKOutputFormat.STANDARD
        if output_format.is_bypassed():
            if hidden_states is None:
                hidden_states = torch.empty((0, 0), dtype=torch.float32, device=device)
            if router_logits is None:
                router_logits = torch.empty((0, 0), dtype=torch.float32, device=device)
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
            )

        topk = self.topk_config.top_k - self.topk_config.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_idx = torch.full(
            (0, topk),
            -1,
            dtype=self.topk_config.topk_indices_dtype,
            device=device,
        )
        if router_logits is None:
            router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_idx, router_logits)


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    num_token_non_padded: torch.Tensor | None = None,
    expert_location_dispatch_info: ExpertLocationDispatchInfo | None = None,
) -> StandardTopKOutput:

    top_k = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    correction_bias = topk_config.correction_bias
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor

    router_logits, correction_bias = transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

    # DeepSeek V2/V3/R1 series models use grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk_gpu(
                hidden_states,
                router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            mapped_in_kernel = False
            logical_to_physical_map = None
            if (
                expert_location_dispatch_info is not None
                and expert_location_dispatch_info.ep_dispatch_algorithm == "static"
            ):
                logical_to_physical_map = (
                    expert_location_dispatch_info.partial_logical_to_rank_dispatch_physical_map
                )
                mapped_in_kernel = True
            topk_weights, topk_ids = minimax_biased_grouped_topk(
                hidden_states,
                router_logits,
                correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                logical_to_physical_map=logical_to_physical_map,
            )
            if mapped_in_kernel:
                expert_location_dispatch_info = None

        topk_ids = topk_ids_logical_to_physical(
            topk_ids,
            expert_location_dispatch_info,
            num_experts=router_logits.shape[1],
        )
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        topk_weights, topk_ids = torch_native_fused_topk(
            hidden_states,
            router_logits,
            topk=top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
        )
        if routed_scaling_factor is not None:
            topk_weights *= routed_scaling_factor
    elif correction_bias is not None:
        # Bias-corrected top-k uses the CUDA fused_topk_bias kernel.
        num_tokens = router_logits.shape[0]
        topk_ids = torch.empty(
            num_tokens,
            top_k,
            device=router_logits.device,
            dtype=topk_config.topk_indices_dtype,
        )
        topk_weights = torch.empty(
            num_tokens, top_k, device=router_logits.device, dtype=torch.float32
        )
        num_real_experts = router_logits.shape[1] - topk_config.zero_expert_num
        cuda_routing_flash(
            router_logits,
            correction_bias,
            topk_ids,
            topk_weights,
            num_real_experts,
            routed_scaling_factor,
            True,
        )
    elif custom_routing_function is None:
        topk_weights, topk_ids = torch_native_fused_topk(
            hidden_states,
            router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
        if routed_scaling_factor is not None:
            topk_weights *= routed_scaling_factor
        topk_ids = topk_ids_logical_to_physical(
            topk_ids,
            expert_location_dispatch_info,
            num_experts=router_logits.shape[1],
        )
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)

    else:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
        if routed_scaling_factor is not None:
            topk_weights *= routed_scaling_factor

    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
