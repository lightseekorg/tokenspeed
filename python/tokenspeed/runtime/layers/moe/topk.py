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
from typing import NamedTuple, Protocol, runtime_checkable

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from tokenspeed_kernel.ops.moe.reference import _mask_topk_ids_padded_region

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
    apply_routed_scaling_factor_on_output: bool = False
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
        apply_routed_scaling_factor_on_output: bool | None = False,
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
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
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
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )

    from tokenspeed_kernel.ops.moe import transform_select_experts_inputs

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
            topk_weights, topk_ids = tokenspeed_kernel.moe_route(
                hidden_states,
                router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
                dtype=router_logits.dtype,
                traits={
                    "output_type": "topk",
                    "biased": False,
                    "grouped": True,
                    "ep": True,
                },
                expected_kernel_name="torch_compile_grouped_topk",
            )
        else:
            route_traits = {
                "output_type": "topk",
                "biased": True,
                "grouped": True,
                "ep": True,
                "num_expert_group": num_expert_group,
                "topk_group": topk_group,
                "topk": top_k,
                "num_fused_shared_experts": num_fused_shared_experts,
            }
            topk_weights, topk_ids = tokenspeed_kernel.moe_route(
                hidden_states,
                router_logits,
                correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
                dtype=router_logits.dtype,
                traits=route_traits,
            )
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        topk_weights, topk_ids = tokenspeed_kernel.moe_route(
            hidden_states,
            router_logits,
            topk=top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            dtype=router_logits.dtype,
            traits={
                "output_type": "topk",
                "biased": False,
                "grouped": False,
                "ep": False,
            },
            expected_kernel_name="torch_native_fused_topk",
        )
        if apply_routed_scaling_factor_on_output and routed_scaling_factor is not None:
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
        tokenspeed_kernel.moe_route(
            router_logits,
            correction_bias,
            topk_ids,
            topk_weights,
            num_real_experts,
            routed_scaling_factor,
            False,
            dtype=router_logits.dtype,
            traits={
                "output_type": "topk",
                "biased": True,
                "grouped": False,
                "ep": False,
            },
            expected_kernel_name="cuda_routing_flash",
        )
    elif custom_routing_function is None:
        topk_weights, topk_ids = tokenspeed_kernel.moe_route(
            hidden_states,
            router_logits,
            topk=top_k,
            renormalize=renormalize,
            dtype=router_logits.dtype,
            traits={
                "output_type": "topk",
                "biased": False,
                "grouped": False,
                "ep": False,
            },
            expected_kernel_name="torch_native_fused_topk",
        )
        if apply_routed_scaling_factor_on_output and routed_scaling_factor is not None:
            topk_weights *= routed_scaling_factor
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
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
        if apply_routed_scaling_factor_on_output and routed_scaling_factor is not None:
            topk_weights *= routed_scaling_factor

    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
