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
from tokenspeed_kernel.ops.moe import (
    ExpertDispatch,
    dispatch_topk_ids,
    moe_topk,
)
from tokenspeed_kernel.ops.moe.sigmoid_topk import minimax_biased_grouped_topk
from tokenspeed_kernel.ops.moe.triton.inkling_topk import inkling_topk
from tokenspeed_kernel.thirdparty.cuda import routing_flash as cuda_routing_flash

from tokenspeed.runtime.moe.dispatch_algorithm import STATIC_EP_DISPATCH_ALGORITHMS
from tokenspeed.runtime.moe.expert_load_rows import LayerExpertLoad
from tokenspeed.runtime.utils.env import global_server_args_dict


class TopKOutputFormat(Enum):
    STANDARD = auto()
    BYPASSED = auto()

    def is_standard(self) -> bool:
        return self == TopKOutputFormat.STANDARD

    def is_bypassed(self) -> bool:
        return self == TopKOutputFormat.BYPASSED


@dataclass
class ExpertLocationDispatchInfo:
    """One MoE layer's view of the expert placement, for routing.

    Views of the global ``ExpertLocationMetadata`` tables, so an in-place
    placement update reaches captured graphs. Two static flavours exist:
    all-to-all EP routes each rank's own tokens, so every rank dispatches to
    its nearest replica (``partial_logical_to_rank_dispatch_physical_map``);
    replicated-input EP routes every token on every rank and exactly one rank
    must compute each route, so the replica is a pure function of the token
    (``replica_dispatch``, see ``ExpertDispatch``).
    """

    layer_id: int
    ep_dispatch_algorithm: Literal[
        "static",
        "dynamic",
        "fake",
        "static_with_zero_expert",
        "dynamic_with_zero_expert",
    ]
    # (num_logical_experts,) this rank's static map; None unless all-to-all
    # EP under a static algorithm, the only consumer.
    partial_logical_to_rank_dispatch_physical_map: torch.Tensor | None
    # (num_logical_experts, X) replicas of every logical expert, -1 padded.
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int
    # Rank-agnostic replica tables for replicated-input EP; None under
    # all-to-all EP, where the per-rank static map applies.
    replica_dispatch: ExpertDispatch | None
    # This layer's route counters with the model-wide live-row mask, or None
    # when load recording is off.
    load: LayerExpertLoad | None

    @classmethod
    def init_new(
        cls,
        layer_id: int,
        ep_dispatch_algorithm: str,
        expert_location_metadata: Any,
        *,
        all_to_all_ep: bool,
    ):
        """Slice ``layer_id``'s routing tables out of the placement.

        Args:
            layer_id: The MoE layer (row of the placement tables).
            ep_dispatch_algorithm: ``--ep-dispatch-algorithm``.
            expert_location_metadata: The ``ExpertLocationMetadata`` placement.
            all_to_all_ep: Whether the layer's MoE kernel owns all-to-all
                dispatch (DeepEP), so each rank routes only its own tokens.
                Otherwise every rank routes every token and a static
                algorithm must pick the same replica on every rank.
        """
        static = ep_dispatch_algorithm in STATIC_EP_DISPATCH_ALGORITHMS
        if not all_to_all_ep and not static:
            raise ValueError(
                f"--ep-dispatch-algorithm {ep_dispatch_algorithm} draws replicas "
                "at random per rank, but without all-to-all EP every rank routes "
                "every token and must agree on one replica per route; use "
                "static or static_with_zero_expert."
            )
        replicas = expert_location_metadata.logical_to_all_physical_map[layer_id]
        num_replicas = expert_location_metadata.logical_to_all_physical_map_num_valid[
            layer_id
        ]
        return cls(
            layer_id=layer_id,
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            # The per-rank static map is computed on first request, so a
            # replicated-input EP server never builds it.
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.rank_dispatch_map()[layer_id]
                if static and all_to_all_ep
                else None
            ),
            partial_logical_to_all_physical_map=replicas,
            partial_logical_to_all_physical_map_num_valid=num_replicas,
            num_physical_experts=expert_location_metadata.num_physical_experts,
            replica_dispatch=(
                ExpertDispatch(replicas, num_replicas)
                if static and not all_to_all_ep
                else None
            ),
            load=(
                LayerExpertLoad(
                    expert_location_metadata.physical_load[layer_id],
                    expert_location_metadata.load_rows,
                )
                if expert_location_metadata.physical_load is not None
                else None
            ),
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
    """Map logical expert ids onto the physical replicas routing dispatches to.

    Args:
        topk_ids: ``[tokens, top_k]`` logical ids; every entry must be a real
            expert for the plain algorithms. The ``*_with_zero_expert``
            algorithms leave ids outside ``[0, num_experts)`` (zero experts,
            ``-1`` masked slots) untouched.
        info: The layer's placement view, or None to keep logical ids.
        num_experts: Routed expert count, required by the zero-expert
            algorithms.

    Returns:
        Physical ids in ``topk_ids``' dtype.
    """
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm == "static_with_zero_expert":
        assert num_experts is not None
        return _map_real_expert_ids(
            topk_ids,
            num_experts,
            lambda ids: _topk_ids_logical_to_physical_static(ids, info),
        )
    if info.ep_dispatch_algorithm == "dynamic_with_zero_expert":
        assert num_experts is not None
        return _map_real_expert_ids(
            topk_ids,
            num_experts,
            lambda ids: _topk_ids_logical_to_physical_dynamic(ids, info),
        )
    if info.ep_dispatch_algorithm in {"dynamic", "fake"}:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info).to(topk_ids.dtype)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
) -> torch.Tensor:
    if info.replica_dispatch is not None:
        return dispatch_topk_ids(topk_ids, info.replica_dispatch)
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids].to(
        topk_ids.dtype
    )


def _map_real_expert_ids(
    topk_ids: torch.Tensor,
    num_experts: int,
    convert: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply ``convert`` to the ids in ``[0, num_experts)`` and keep the rest.

    Zero experts (ids at or beyond ``num_experts``, or ``-1``) have no
    physical slot. Written without boolean indexing so it captures into
    CUDA graphs.
    """
    real = (topk_ids >= 0) & (topk_ids < num_experts)
    converted = convert(topk_ids.masked_fill(~real, 0))
    return torch.where(real, converted.to(topk_ids.dtype), topk_ids)


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


def record_expert_load(load: LayerExpertLoad | None, topk_ids: torch.Tensor) -> None:
    """Count the real routes of ``topk_ids`` into the layer's counters (None: off).

    Routes of filler rows (a padded replay) and ``-1`` entries (zero experts,
    masked slots) are not traffic; see ``LayerExpertLoad.record``.
    """
    if load is None:
        return
    load.record(topk_ids)


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


def torch_router_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    num_real_experts: int,
    routed_scaling_factor: float,
    indices_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correction-bias routing in the trainer's torch order (``--router-topk torch``).

    fp32 ``torch.softmax`` keeps every probability bit; ``torch.topk`` on the
    biased probabilities with ``sorted=True`` gives PyTorch's tie order; the
    weights are the UNBIASED probabilities times ``routed_scaling_factor``.
    Zero experts (ids past the real experts) become ``-1`` and keep their
    weight so the model can apply its identity residual.

    Args:
        router_logits: ``[tokens, num_candidates]`` router logits (any float
            dtype); candidates are the real experts followed by zero experts.
        correction_bias: ``[num_candidates]`` fp32 selection bias.
        top_k: Experts selected per token.
        num_real_experts: Candidates below this id are real experts.
        routed_scaling_factor: Multiplier on the selected probabilities.
        indices_dtype: dtype of the returned ids.

    Returns:
        ``(weights, ids)`` with shapes ``[tokens, top_k]``; weights fp32, ids
        in ``indices_dtype`` with ``-1`` for zero experts.
    """
    probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    _, ids = torch.topk(probs + correction_bias, k=top_k, dim=-1, sorted=True)
    weights = probs.gather(1, ids) * routed_scaling_factor
    ids = ids.masked_fill(ids >= num_real_experts, -1).to(indices_dtype)
    return weights, ids


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
    # --router-topk: how the correction-bias route selects experts ("fused" or
    # "torch"). Behaviour-selecting, so it has no default.
    router_topk: str
    # The MoE layer this router serves; an expert placement's dispatch info
    # must come from the same layer.
    layer_id: int | None = None
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
    topk_indices_dtype: torch.dtype = torch.int32
    # Weights dtype for the biased-grouped path; bf16 lets consumers skip a cast.
    topk_weights_dtype: torch.dtype = torch.float32
    # Shared-expert sink (Inkling)
    num_sink_experts: int = 0
    sink_global_scale: torch.Tensor | None = None


class StandardTopKOutput(NamedTuple):
    """Precomputed routing; logits may be omitted once IDs and weights suffice."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor | None
    output_scale: float | torch.Tensor = 1.0

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
    output_scale: float | torch.Tensor = 1.0

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED


@runtime_checkable
class TopKOutput(Protocol):
    """Protocol for top-k outputs in different formats."""

    @property
    def output_scale(self) -> float | torch.Tensor:
        """Post-kernel scale to apply to the routed output."""
        ...

    @property
    def format(self) -> TopKOutputFormat:
        """The format of the output."""
        ...


class TopK(torch.nn.Module):

    def __init__(
        self,
        top_k: int,
        *,
        layer_id: int | None = None,
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
        topk_indices_dtype: torch.dtype = torch.int32,
        topk_weights_dtype: torch.dtype = torch.float32,
        num_sink_experts: int = 0,
        sink_global_scale: torch.Tensor | None = None,
    ):
        super().__init__()

        if use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        if num_sink_experts > 0:
            assert correction_bias is not None
            assert sink_global_scale is not None
            assert routed_scaling_factor is not None
        router_topk = global_server_args_dict["router_topk"]
        # The correction-bias route (select_experts) is the one --router-topk
        # selects; the trainer's torch order has no renormalization step, so a
        # model asking for one is refused here, at construction.
        if (
            router_topk == "torch"
            and renormalize
            and correction_bias is not None
            and not use_grouped_topk
            and num_sink_experts == 0
        ):
            raise ValueError(
                "--router-topk torch routes unnormalized probabilities, as the "
                "trainer does; this model asks to renormalize them"
            )

        self.topk_config = TopKConfig(
            top_k=top_k,
            router_topk=router_topk,
            layer_id=layer_id,
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
            topk_weights_dtype=topk_weights_dtype,
            num_sink_experts=num_sink_experts,
            sink_global_scale=sink_global_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        output_format: TopKOutputFormat | None = None,
        num_token_non_padded: torch.Tensor | None = None,
        expert_location_dispatch_info: ExpertLocationDispatchInfo | None = None,
    ) -> TopKOutput:
        output_format = (
            output_format or self.topk_config.output_format or TopKOutputFormat.STANDARD
        )

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


def map_zero_expert_routes(
    topk_ids: torch.Tensor,
    info: ExpertLocationDispatchInfo,
    num_real_experts: int,
) -> torch.Tensor:
    """Map a zero-expert router's ids (``-1`` for zero experts) onto replicas.

    Zero-expert slots have no physical expert: they are masked out of the
    mapping and restored to ``-1`` afterwards, so a consumer that tests
    ``ids < 0`` still finds them while ids in ``[num_real_experts, P)`` are
    real replicas.
    """
    zero_mask = topk_ids < 0
    mapped = topk_ids_logical_to_physical(
        topk_ids.masked_fill(zero_mask, 0), info, num_experts=num_real_experts
    )
    return mapped.to(topk_ids.dtype).masked_fill(zero_mask, -1)


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

    if (
        expert_location_dispatch_info is not None
        and topk_config.layer_id is not None
        and expert_location_dispatch_info.layer_id != topk_config.layer_id
    ):
        raise ValueError(
            f"expert placement of layer {expert_location_dispatch_info.layer_id} "
            f"handed to the router of layer {topk_config.layer_id}"
        )
    # Taken before the branches: the grouped path drops the dispatch info once
    # its kernel has mapped the ids, and the counters still apply to those.
    expert_load = (
        expert_location_dispatch_info.load
        if expert_location_dispatch_info is not None
        else None
    )

    router_logits, correction_bias = transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

    # Shared-expert-sink routing (Inkling)
    if topk_config.num_sink_experts > 0:
        assert num_token_non_padded is None
        assert expert_location_dispatch_info is None
        topk_weights, topk_ids = inkling_topk(
            router_logits,
            correction_bias,
            topk_config.sink_global_scale,
            top_k=top_k,
            n_routed=router_logits.shape[1] - topk_config.num_sink_experts,
            route_scale=routed_scaling_factor,
        )
    # DeepSeek V2/V3/R1 series models use grouped_top_k
    elif use_grouped_topk:
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
            # The kernels take the per-rank static map; the token-pure replica
            # choice of replicated-input EP is applied after routing instead.
            if (
                expert_location_dispatch_info is not None
                and expert_location_dispatch_info.ep_dispatch_algorithm == "static"
                and expert_location_dispatch_info.replica_dispatch is None
            ):
                logical_to_physical_map = (
                    expert_location_dispatch_info.partial_logical_to_rank_dispatch_physical_map
                )
                mapped_in_kernel = True
            num_experts = router_logits.shape[1]
            use_sigmoid_bias_topk = (
                0 < top_k <= num_experts
                and num_expert_group == 1
                and topk_group == 1
                and num_fused_shared_experts == 0
                and routed_scaling_factor is not None
                and num_token_non_padded is None
                and (
                    expert_location_dispatch_info is None
                    or logical_to_physical_map is not None
                )
            )
            if use_sigmoid_bias_topk:
                topk_weights, topk_ids = moe_topk(
                    router_logits,
                    top_k,
                    score_function="sigmoid",
                    selection_method="topk",
                    renormalize=renormalize,
                    routed_scaling_factor=float(routed_scaling_factor),
                    correction_bias=correction_bias,
                    logical_to_physical_map=logical_to_physical_map,
                    topk_weights_dtype=topk_config.topk_weights_dtype,
                )
            else:
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
                    weights_dtype=topk_config.topk_weights_dtype,
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
        num_real_experts = router_logits.shape[1] - topk_config.zero_expert_num
        if topk_config.router_topk == "torch":
            # The trainer's order; TopK.__init__ refused renormalize with it.
            topk_weights, topk_ids = torch_router_topk(
                router_logits,
                correction_bias,
                top_k,
                num_real_experts,
                1.0 if routed_scaling_factor is None else float(routed_scaling_factor),
                topk_config.topk_indices_dtype,
            )
        else:
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
            cuda_routing_flash(
                router_logits,
                correction_bias,
                topk_ids,
                topk_weights,
                num_real_experts,
                routed_scaling_factor,
                renormalize,
            )
        # Either router marks zero experts -1; the placement maps the rest.
        if expert_location_dispatch_info is not None:
            topk_ids = map_zero_expert_routes(
                topk_ids, expert_location_dispatch_info, num_real_experts
            )
    elif custom_routing_function is None:
        assert (
            hidden_states.shape[0] == router_logits.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {router_logits.shape=}"
        topk_weights, topk_ids = moe_topk(
            router_logits,
            top_k,
            score_function="softmax",
            selection_method="topk",
            renormalize=renormalize,
            routed_scaling_factor=(
                1.0 if routed_scaling_factor is None else routed_scaling_factor
            ),
            topk_indices_dtype=topk_config.topk_indices_dtype,
        )
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

    record_expert_load(expert_load, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
