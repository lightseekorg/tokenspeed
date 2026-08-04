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

"""Runtime orchestration for Kimi-style latent mixture-of-experts layers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.communication import (
    allreduce_lane_latent_norm_supported,
)
from tokenspeed_kernel.ops.moe import native_latent_moe_available
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import (
    COMM_ONESHOT_MAX_BYTES,
    all_reduce,
    all_reduce_latent_norm,
    all_reduce_two,
    prepare_all_reduce_fusion,
    prepare_all_reduce_lane,
)
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_cuda_graph_phase
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.utils.cuda_stream import StreamFork

TensorReducer = Callable[[torch.Tensor], torch.Tensor]
TensorPairReducer = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]
# Projects hidden states to router logits, routed latent, and the unreduced
# shared-expert partial in one pass, or returns None to use the modules.
InputProjector = Callable[
    [torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]
_SUPPORTED_EP_SIZES = {1, 2, 4, 8}


def kimi3_join_reduce_moe(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    *,
    lane: torch.Tensor | None,
    routed_hidden: int,
    routed_norm: nn.Module | None,
    group: tuple[int, ...],
    enable_lane_norm: bool,
    max_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join the routed/shared partials and reduce them, owning the strategy.

    Three regimes, all element-wise identical:

    * Lane hit (decode batch=1): the partials were produced straight into the
      persistent fused lane, one one-shot reduce with an eligible norm
      epilogue and zero copies.
    * Small partials: cat into one contiguous operand and take a single
      one-shot reduce; the copy is a couple of microseconds there.
    * Partials past the one-shot window (prefill-sized chunks): the cat would
      copy a few hundred MB per layer just to feed one NCCL call, while a
      grouped NCCL launch reduces both tensors in place with the same
      single-launch latency -- so skip the join entirely.
    """

    if lane is not None and routed_partial.data_ptr() == lane.data_ptr():
        fused = lane
    elif (
        routed_partial.numel() * routed_partial.element_size() > COMM_ONESHOT_MAX_BYTES
    ):
        routed_out, shared_out = all_reduce_two(
            routed_partial,
            shared_partial,
            group=group,
        )
        if routed_norm is not None:
            routed_out = routed_norm(routed_out)
        return routed_out, shared_out
    else:
        fused = torch.cat((routed_partial, shared_partial), dim=-1)

    lane_norm_applied = routed_norm is not None and (
        allreduce_lane_latent_norm_supported(
            fused,
            enabled=enable_lane_norm,
        )
    )
    if lane_norm_applied:
        fused = all_reduce_latent_norm(
            fused,
            routed_norm.weight,
            routed_hidden,
            group,
            eps=routed_norm.variance_epsilon,
            max_token_num=max_token_num,
        )
    else:
        fused = all_reduce(fused, group)
    routed_out = fused[:, :routed_hidden]
    shared_out = fused[:, routed_hidden:]
    if routed_norm is not None and not lane_norm_applied:
        routed_out = routed_norm(routed_out)
    return routed_out, shared_out


@dataclass(frozen=True)
class Kimi3MoEExecutionPlan:
    """Construction-time orchestration selected for Kimi-K3 latent MoE."""

    use_native: bool
    use_trtllm: bool
    overlap_shared_experts: bool
    joint_moe_reduce: bool
    fused_moe_ar: bool = False
    lane_latent_norm_ar: bool = False
    comm_fusion_max_num_tokens: int = 0

    @property
    def use_precomputed_topk(self) -> bool:
        return self.use_native or self.use_trtllm

    @classmethod
    def build(
        cls,
        mapping,
        moe_backend,
        alt_stream: torch.cuda.Stream | None,
        *,
        enforce_eager: bool,
    ) -> "Kimi3MoEExecutionPlan":
        """Select orchestration without exposing platform policy to the model."""

        use_native = native_latent_moe_available()
        use_trtllm = not use_native and (
            moe_backend.is_auto() or moe_backend.is_flashinfer_trtllm()
        )
        return cls(
            use_native=use_native,
            use_trtllm=use_trtllm,
            overlap_shared_experts=(
                use_native
                and enforce_eager
                and alt_stream is not None
                and mapping.moe.tp_ep_size == 1
            ),
            joint_moe_reduce=(
                use_native
                and mapping.moe.tp_size == 1
                and mapping.moe.ep_size > 1
                and mapping.moe.ep_group == mapping.moe.tp_ep_group
            ),
        )

    def prepare_latent_fusion(
        self,
        mapping,
        *,
        lane_width: int,
        has_latent_norm: bool,
        max_token_num: int,
    ) -> "Kimi3MoEExecutionPlan":
        """Prepare optional communication fusions before graph capture."""

        fused_moe_ar = (
            self.use_trtllm
            and mapping.moe.has_tp_ep
            and prepare_all_reduce_lane(mapping.moe.tp_ep_group, lane_width)
        )
        lane_latent_norm_ar = (
            fused_moe_ar
            and has_latent_norm
            and prepare_all_reduce_fusion(
                mapping.moe.tp_ep_group,
                lane_width,
                max_token_num,
            )
        )
        return replace(
            self,
            fused_moe_ar=fused_moe_ar,
            lane_latent_norm_ar=lane_latent_norm_ar,
            comm_fusion_max_num_tokens=max_token_num,
        )


class Kimi3LatentProjection(ReplicatedLinear):
    """Replicated latent projection with kernel-owned specialization.

    Tuned shapes use registered accelerator kernels. Other shapes retain the
    ordinary dense projection without requiring model-side shape selection.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
        solution: str = "auto",
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.solution = solution

    def forward(self, hidden_states: torch.Tensor):
        output = tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            self.weight,
            solution=self.solution,
        )
        return output, None

    def forward_add3(
        self,
        hidden_states: torch.Tensor,
        addend_a: torch.Tensor,
        addend_c: torch.Tensor,
    ) -> torch.Tensor:
        """Project routed latents and accumulate two full-width addends."""
        return tokenspeed_kernel.kimi3_latent_projection_add3(
            hidden_states,
            self.weight,
            addend_a,
            addend_c,
        )


def _module_tensor_output(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the tensor output of Torch or TokenSpeed linear-like modules."""

    output = module(x)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"{type(module).__name__} must return a tensor or (tensor, bias)"
        )
    return output


def _check_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    name: str,
) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must preserve shape {expected}, got {tuple(tensor.shape)}"
        )


class LatentMoELayer(nn.Module):
    """Route at H width, execute/reduce at latent width, then project to H."""

    def __init__(
        self,
        *,
        router: nn.Module,
        topk: nn.Module,
        routed_down_proj: nn.Module,
        experts: nn.Module,
        routed_up_proj: nn.Module,
        routed_norm: nn.Module | None = None,
        shared_experts: nn.Module | None = None,
        latent_reduce: TensorReducer | None = None,
        shared_reduce: TensorReducer | None = None,
        joint_reduce: TensorPairReducer | None = None,
        shared_expert_stream: torch.cuda.Stream | None = None,
        expert_parallel_group: tuple[int, ...] | None = None,
        return_separate_outputs: bool = False,
        input_projections: InputProjector | None = None,
    ) -> None:
        super().__init__()
        if input_projections is not None and shared_experts is None:
            raise ValueError("input_projections requires shared_experts")
        if shared_reduce is not None and shared_experts is None:
            raise ValueError("shared_reduce requires shared_experts")
        if joint_reduce is not None and shared_experts is None:
            raise ValueError("joint_reduce requires shared_experts")
        if joint_reduce is not None and (
            latent_reduce is not None or shared_reduce is not None
        ):
            raise ValueError(
                "joint_reduce cannot be combined with latent_reduce or shared_reduce"
            )
        expert_parallel_size = int(getattr(experts, "ep_size", 1))
        num_experts = int(getattr(experts, "num_experts", 1))
        if (
            expert_parallel_size not in _SUPPORTED_EP_SIZES
            or num_experts % expert_parallel_size
        ):
            raise ValueError(
                "Kimi 3 requires ep_size in {1, 2, 4, 8} dividing num_experts"
            )
        if expert_parallel_group is None:
            expert_parallel_group = getattr(experts, "ep_group", None)
        if expert_parallel_group is not None:
            expert_parallel_group = tuple(expert_parallel_group)
            if len(expert_parallel_group) != expert_parallel_size:
                raise ValueError(
                    "expert_parallel_group size must match experts.ep_size: "
                    f"{len(expert_parallel_group)} != {expert_parallel_size}"
                )
        if expert_parallel_size > 1 and latent_reduce is None and joint_reduce is None:
            if expert_parallel_group is None:
                raise ValueError(
                    "Kimi 3 EP requires expert_parallel_group or an explicit "
                    "latent_reduce callback"
                )
            latent_reduce = partial(all_reduce, group=expert_parallel_group)
        self.router = router
        self.topk = topk
        self.routed_down_proj = routed_down_proj
        self.experts = experts
        self.routed_norm = routed_norm
        self.routed_up_proj = routed_up_proj
        self.shared_experts = shared_experts
        self.latent_reduce = latent_reduce
        self.shared_reduce = shared_reduce
        self.joint_reduce = joint_reduce
        self.stream_fork = StreamFork(shared_expert_stream)
        self.return_separate_outputs = return_separate_outputs
        self.input_projections = input_projections

    def finalize_output(
        self,
        routed_latent: torch.Tensor,
        prefix_sum: torch.Tensor,
        shared_output: torch.Tensor,
    ) -> torch.Tensor:
        """Project the routed latent and add both full-width residuals."""

        output_shape = tuple(shared_output.shape)
        _check_shape(prefix_sum, output_shape, "prefix_sum")
        output = self.routed_up_proj.forward_add3(
            routed_latent,
            prefix_sum,
            shared_output,
        )
        _check_shape(output, output_shape, "routed_up_proj")
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        prefix_sum: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"latent MoE expects hidden states [T, H], got {tuple(hidden_states.shape)}"
            )
        if prefix_sum is not None and self.shared_experts is None:
            raise ValueError("prefix_sum requires shared_experts")
        num_tokens, hidden_size = hidden_states.shape
        num_global_tokens = (
            num_tokens if num_global_tokens is None else num_global_tokens
        )
        max_num_tokens_per_gpu = (
            num_tokens if max_num_tokens_per_gpu is None else max_num_tokens_per_gpu
        )

        output_shape = (num_tokens, hidden_size)
        shared_output = None
        overlap_shared = (
            self.shared_experts is not None
            and self.stream_fork.aux_stream is not None
            and num_tokens > 0
            # HIP graph warmup can deadlock when an auxiliary GEMM competes
            # with Iris's spin-wait all-reduce kernels. Keep the captured path
            # serial; eager serving can safely use the compute-stream overlap.
            and not get_is_cuda_graph_phase()
        )
        shared_reduction_applied = False

        # Projecting from the packed weight in one GEMM serializes the
        # shared branch against the routed one by construction, so it is only
        # worth taking when the branches were not going to overlap anyway.
        packed = None
        if self.input_projections is not None and not overlap_shared and num_tokens > 0:
            packed = self.input_projections(hidden_states)
        packed_router, packed_routed, packed_shared = (
            (None, None, None) if packed is None else packed
        )

        def run_shared_branch() -> None:
            nonlocal shared_output, shared_reduction_applied
            if self.shared_experts is None:
                return
            # This helper is entered through ``fork.branch()`` below.  With
            # overlap enabled, the full-width H->FFN->H shared-expert MLP runs
            # on the auxiliary stream while the primary stream executes the
            # routed H->L->MoE path.  Otherwise ``fork.branch()`` is a no-op and
            # both paths run serially on the primary stream.  The fork joins
            # before collectives, and this H-width result is added to the
            # routed result at the end of the layer.
            shared_output = (
                packed_shared
                if packed_shared is not None
                else _module_tensor_output(self.shared_experts, hidden_states)
            )
            _check_shape(shared_output, output_shape, "shared_experts")
            # In graph mode the branch is serial. Reduce here to retain the
            # established shared-before-routed Iris collective order. Eager
            # overlap defers this reduction until after the fork joins,
            # keeping collectives off the auxiliary stream.
            if (
                self.shared_reduce is not None
                and self.stream_fork.aux_stream is not None
                and not overlap_shared
            ):
                shared_output = self.shared_reduce(shared_output)
                _check_shape(shared_output, output_shape, "shared_reduce")
                shared_reduction_applied = True

        # When overlap is enabled, StreamFork runs the shared-expert MLP on
        # its auxiliary stream while routed-expert work remains on the primary
        # stream. Leaving the scope joins both streams before Iris collectives.
        with self.stream_fork.scope(enable=overlap_shared) as fork:
            with fork.branch():
                run_shared_branch()

            router_logits = (
                packed_router
                if packed_router is not None
                else _module_tensor_output(self.router, hidden_states)
            )
            if router_logits.ndim != 2 or router_logits.shape[0] != num_tokens:
                raise ValueError("router must return logits shaped [T, E]")
            if num_tokens > 0:
                topk_output = self.topk(hidden_states, router_logits)
            else:
                topk_output = self.topk.empty_topk_output(
                    hidden_states.device,
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

            routed_input = (
                packed_routed
                if packed_routed is not None
                else _module_tensor_output(self.routed_down_proj, hidden_states)
            )
            if routed_input.ndim != 2 or routed_input.shape[0] != num_tokens:
                raise ValueError("routed_down_proj must return [T, L]")
            latent_shape = tuple(routed_input.shape)
            routed_latent = self.experts(
                hidden_states=routed_input,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            )
            _check_shape(routed_latent, latent_shape, "routed experts")

        # Spin-wait collectives cannot safely overlap an all-device GEMM.
        # Join both compute branches first. Individual reducers retain
        # the established shared-before-routed order; a joint reducer handles
        # both partials after the routed experts finish.
        if overlap_shared and self.shared_reduce is not None:
            shared_output = self.shared_reduce(shared_output)
            _check_shape(shared_output, output_shape, "shared_reduce")
            shared_reduction_applied = True

        if self.joint_reduce is not None:
            shared_output, routed_latent = self.joint_reduce(
                shared_output,
                routed_latent,
            )
            _check_shape(shared_output, output_shape, "joint_reduce shared output")
            _check_shape(routed_latent, latent_shape, "joint_reduce routed latent")
            shared_reduction_applied = True
        elif self.latent_reduce is not None:
            routed_latent = self.latent_reduce(routed_latent)
            _check_shape(routed_latent, latent_shape, "latent_reduce")
        if self.routed_norm is not None:
            routed_latent = _module_tensor_output(self.routed_norm, routed_latent)
            _check_shape(routed_latent, latent_shape, "routed_norm")

        if shared_output is None:
            routed_output = _module_tensor_output(self.routed_up_proj, routed_latent)
            _check_shape(routed_output, output_shape, "routed_up_proj")
            return routed_output
        if prefix_sum is None:
            routed_output = _module_tensor_output(self.routed_up_proj, routed_latent)
            _check_shape(routed_output, output_shape, "routed_up_proj")
        if self.shared_reduce is not None and not shared_reduction_applied:
            shared_output = self.shared_reduce(shared_output)
            _check_shape(shared_output, output_shape, "shared_reduce")
        if prefix_sum is not None:
            return self.finalize_output(routed_latent, prefix_sum, shared_output)
        if self.return_separate_outputs:
            return routed_output, shared_output
        return routed_output + shared_output


__all__ = [
    "Kimi3LatentProjection",
    "Kimi3MoEExecutionPlan",
    "LatentMoELayer",
    "kimi3_reduce_fused_moe",
]
