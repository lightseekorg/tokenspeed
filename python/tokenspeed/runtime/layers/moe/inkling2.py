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

"""Original-precision Inkling2 expert parameters and common kernel dispatch."""

from functools import partial

import torch
from tokenspeed_kernel.ops.communication.mailbox import MailboxTarget
from tokenspeed_kernel.ops.moe.block_scaled import (
    block_scaled_moe_routes,
    block_scaled_moe_routes_plan,
    block_scaled_moe_routes_process_tensor,
    block_scaled_router_mm,
    block_scaled_topk,
)
from torch import nn

from tokenspeed.runtime.configs.inkling2 import Inkling2ModelConfig
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.moe.topk import (
    ExpertLocationDispatchInfo,
    record_expert_load,
)
from tokenspeed.runtime.layers.quantization.inkling2 import (
    Inkling2QuantConfig,
    load_inkling2_expert_tensor,
)
from tokenspeed.runtime.moe.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from tokenspeed.runtime.utils import set_weight_attrs
from tokenspeed.runtime.utils.env import global_server_args_dict


class _Gate(nn.Module):
    def __init__(self, hidden: int, experts: int, use_bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(experts, hidden, dtype=torch.bfloat16), requires_grad=False
        )
        if use_bias:
            self.bias = nn.Parameter(
                torch.empty(experts, dtype=torch.float32), requires_grad=False
            )
        else:
            self.register_buffer(
                "bias", torch.zeros(experts, dtype=torch.float32), persistent=False
            )


class _Experts(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        mapping: Mapping,
        quant_config: Inkling2QuantConfig,
        prefix: str,
        local: int,
        slot_experts: tuple[int, ...] | None,
    ) -> None:
        super().__init__()
        loader = partial(
            load_inkling2_expert_tensor,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            expert_axis=True,
            slot_experts=slot_experts,
        )
        replicated_loader = partial(
            load_inkling2_expert_tensor,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            expert_axis=False,
            slot_experts=None,
        )
        self.formats = tuple(
            quant_config.precision_for_tensor(f"{prefix}.{leaf}_weight").compute_dtype
            for leaf in ("w13", "w2")
        )
        for leaf, rows, width, fmt in (
            ("w13", 2 * config.intermediate_size, config.hidden_size, self.formats[0]),
            ("w2", config.hidden_size, config.intermediate_size, self.formats[1]),
        ):
            if fmt not in {"nvfp4", "mxfp8"}:
                raise ValueError(
                    f"Inkling2 routed experts require NVFP4 or MXFP8, got {fmt}"
                )
            nvfp4 = fmt == "nvfp4"
            group = 16 if nvfp4 else 32
            if width % group:
                raise ValueError(f"Expert width must be divisible by {group}")
            weight = nn.Parameter(
                torch.empty(
                    local,
                    rows,
                    width // 2 if nvfp4 else width,
                    dtype=torch.uint8 if nvfp4 else torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            scale = nn.Parameter(
                torch.empty(
                    local,
                    rows,
                    width // group,
                    dtype=torch.float8_e4m3fn if nvfp4 else torch.uint8,
                ),
                requires_grad=False,
            )
            self.register_parameter(leaf + "_weight", weight)
            self.register_parameter(leaf + "_scale", scale)
            for kind, param in (("weight", weight), ("scale", scale)):
                load = partial(self._load_planned, loader, role=f"{leaf}_{kind}")
                set_weight_attrs(param, {"weight_loader": load})
            amax = (
                nn.Parameter(
                    torch.empty(local, dtype=torch.float32), requires_grad=False
                )
                if nvfp4
                else None
            )
            denominator = (
                nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
                if nvfp4
                else None
            )
            self.register_parameter(leaf + "_amax", amax)
            self.register_parameter(leaf + "_outer_scale_denominator", denominator)
            if nvfp4:
                set_weight_attrs(amax, {"weight_loader": loader})
                set_weight_attrs(denominator, {"weight_loader": replicated_loader})
        # The route kernel for the device the experts are built on; every load
        # of a weight or scale lands in the layout it reads (_load_planned).
        self.plan = block_scaled_moe_routes_plan(
            formats=self.formats,
            hidden=config.hidden_size,
            intermediate=config.intermediate_size,
            device=self.w13_weight.device,
            solution=None,
        )

    def _load_planned(
        self, loader, param: torch.Tensor, loaded: torch.Tensor, *, role: str
    ) -> None:
        """Copy the rank's shard, then rewrite it into the planned layout."""
        loader(param, loaded)
        block_scaled_moe_routes_process_tensor(self.plan, param, role=role)


class Inkling2MoE(nn.Module):
    """Route and execute original mixed-precision experts with explicit EP order.

    Args:
        config: Checkpoint decoder configuration.
        mapping: Shared runtime parallel mapping; complete experts use EP.
        layer_id: Decoder layer index for diagnostics.
        quant_config: Original tensor precision manifest.
        prefix: Exact checkpoint module prefix, such as
            ``model.llm.layers.0.mlp``.
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: Inkling2QuantConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        if mapping.moe.tp_size != 1 or mapping.moe.dp_size != 1:
            raise ValueError(
                "Inkling2 original-precision experts require TP1 and no MoE DP"
            )
        if config.n_shared_experts:
            raise ValueError(
                "The current Inkling2 expert contract requires no shared experts"
            )
        self.layer_id = layer_id
        self.mapping = mapping
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        # None without a placement; replicas keep one owner per route row, so outputs match.
        placement = get_global_expert_location_metadata()
        if placement is not None:
            _check_placement(placement, config, layer_id, mapping)
        self.num_experts = (
            config.n_routed_experts
            if placement is None
            else placement.num_physical_experts
        )
        if self.num_experts % mapping.moe.ep_size:
            raise ValueError("Expert count must be divisible by expert parallelism")
        self.local_experts = self.num_experts // mapping.moe.ep_size
        self.expert_start = mapping.moe.ep_rank * self.local_experts
        # Every rank routes every token, so replica choice must not depend on rank.
        self.dispatch_info = (
            None
            if placement is None
            else ExpertLocationDispatchInfo.init_new(
                layer_id=layer_id,
                ep_dispatch_algorithm=global_server_args_dict["ep_dispatch_algorithm"],
                expert_location_metadata=placement,
                all_to_all_ep=False,
            )
        )
        self.gate = _Gate(
            config.hidden_size, config.n_routed_experts, config.use_gate_bias
        )
        self.experts = _Experts(
            config,
            mapping,
            quant_config,
            prefix + ".experts",
            local=self.local_experts,
            slot_experts=(
                None
                if placement is None
                else tuple(placement.physical_to_logical_map_cpu[layer_id].tolist())
            ),
        )

        def load_gate(
            destination: torch.Tensor, loaded: torch.Tensor, checkpoint_name: str
        ) -> None:
            quant_config.validate_tensor(checkpoint_name, loaded)
            if loaded.shape != destination.shape or loaded.dtype != destination.dtype:
                raise ValueError(
                    f"Router checkpoint tensor mismatch: {checkpoint_name}"
                )
            with torch.no_grad():
                destination.copy_(loaded)

        for name, param in self.gate.named_parameters():
            set_weight_attrs(
                param,
                {
                    "weight_loader": partial(
                        load_gate, checkpoint_name=f"{prefix}.gate.{name}"
                    )
                },
            )

    def all_reduces(self, tokens: int) -> bool:
        """Whether ``tokens`` rows reach the EP group as one all-reduce of
        the ranks' route sums, which ``forward(reduce=False)`` leaves to the
        caller."""
        return self.mapping.moe.has_ep and tokens > 0

    def forward(
        self,
        x: torch.Tensor,
        ctx: ForwardContext,
        *,
        reduce: bool,
        fold_into: MailboxTarget | torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return BF16 ``[T, 1, H]`` rows after the EP reduction.

        Each rank sums its local routes in FP32 slot order and the EP
        all-reduce adds the ranks' BF16 sums. Without ``reduce`` (only where
        ``all_reduces``) the rows are this rank's sums, which the caller
        all-reduces over the EP group; a mailbox ``fold_into`` publishes them
        into that parity for its consumer to sum (returns None), a tensor
        ``fold_into`` (BF16 ``[T, H]``) receives them.
        """
        if not reduce and not self.all_reduces(x.shape[0]):
            raise ValueError("only an all-reduced batch can leave its reduction")
        if fold_into is not None and reduce:
            raise ValueError("a published fold is reduced by its consumer")
        logits = block_scaled_router_mm(x, self.gate.weight, solution=None)
        info = self.dispatch_info
        weights, ids = block_scaled_topk(
            logits,
            self.gate.bias,
            top_k=self.top_k,
            route_scale=self.route_scale,
            dispatch=None if info is None else info.replica_dispatch,
            solution=None,
        )
        record_expert_load(None if info is None else info.load, ids)
        experts = self.experts
        rows = block_scaled_moe_routes(
            experts.plan,
            x,
            experts.w13_weight,
            experts.w13_scale,
            experts.w13_amax,
            experts.w13_outer_scale_denominator,
            experts.w2_weight,
            experts.w2_scale,
            experts.w2_amax,
            experts.w2_outer_scale_denominator,
            weights,
            ids,
            experts=self.num_experts,
            expert_start=self.expert_start,
            gate_cap=self.gate_cap,
            up_cap=self.up_cap,
            zero_remote_routes=False,
            fold_slots=True,
            fold_into=fold_into,
        )
        if self.mapping.moe.has_ep and x.shape[0] and reduce:
            rows = all_reduce(
                rows,
                group=self.mapping.moe.ep_group,
                backend=None,
                op=torch.distributed.ReduceOp.SUM,
            )
        return rows


def _check_placement(
    placement: ExpertLocationMetadata,
    config: Inkling2ModelConfig,
    layer_id: int,
    mapping: Mapping,
) -> None:
    """Refuse a placement whose geometry or dispatch is not this model's."""
    if placement.num_logical_experts != config.n_routed_experts:
        raise ValueError(
            f"expert placement has {placement.num_logical_experts} logical experts, "
            f"Inkling2 routes {config.n_routed_experts}"
        )
    if not 0 <= layer_id < placement.num_layers:
        raise ValueError(
            f"Inkling2 MoE layer {layer_id} is outside the placement's "
            f"{placement.num_layers} layers"
        )
    if placement.ep_size != mapping.moe.ep_size:
        raise ValueError(
            f"expert placement spans ep_size={placement.ep_size}, the MoE mapping "
            f"has ep_size={mapping.moe.ep_size}"
        )
    algorithm = global_server_args_dict["ep_dispatch_algorithm"]
    if algorithm != "static":
        raise ValueError(
            "Inkling2 routes every token on every EP rank and has no zero "
            f"experts; use --ep-dispatch-algorithm static, not {algorithm}"
        )
