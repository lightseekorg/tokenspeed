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
from tokenspeed_kernel.ops.moe import (
    latent_moe_expert_shared,
    native_latent_moe_available,
)
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed.comm_backend import Group
from tokenspeed.runtime.distributed.comm_ops import (
    COMM_ONESHOT_MAX_BYTES,
    acquire_all_reduce_outputs,
    all_gather_into_tensor,
    all_reduce,
    all_reduce_latent_norm,
    prepare_all_reduce_fusion,
    prepare_all_reduce_lane,
)
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_cuda_graph_phase
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.utils.cuda_stream import StreamFork

TensorReducer = Callable[[torch.Tensor], torch.Tensor]
# Projects hidden states to router logits, routed latent, and the unreduced
# shared-expert partial in one pass, or returns None to use the modules.
InputProjector = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]
# Joint decode uses the same packed projection but returns the activated shared
# input so the selected expert kernel can own the shared down projection.
JointInputProjector = InputProjector
_SUPPORTED_EP_SIZES = {1, 2, 4, 8}


def _marlin_moe_available() -> bool:
    """Whether the Marlin W4A16 MXFP4 MoE path can run here (NVIDIA SM90+)."""
    from tokenspeed_kernel.platform import ArchVersion, current_platform
    from tokenspeed_kernel.thirdparty.cuda.marlin_moe import is_marlin_moe_available

    platform = current_platform()
    return (
        platform.is_nvidia
        and platform.arch_version >= ArchVersion(9, 0)
        and is_marlin_moe_available()
    )


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
        routed_out, shared_out = all_reduce(
            (routed_partial, shared_partial),
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


def latent_moe_expert_shared_all_reduce(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_weight: torch.Tensor,
    *,
    activation_clamp: float,
    linear_clamp: float | None,
    expert_start: int,
    w13_interleaved: bool,
    group: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce and reduce the routed latent and shared-expert output."""
    outputs = acquire_all_reduce_outputs(
        (
            (hidden_states.shape[0], shared_weight.shape[0]),
            tuple(hidden_states.shape),
        ),
        hidden_states,
        group,
    )
    shared_out, routed_out = outputs
    latent_moe_expert_shared(
        hidden_states,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        topk_weights,
        topk_ids,
        shared_input,
        shared_weight,
        activation_clamp=activation_clamp,
        linear_clamp=linear_clamp,
        expert_start=expert_start,
        w13_interleaved=w13_interleaved,
        routed_out=routed_out,
        shared_out=shared_out,
    )
    shared_out, routed_out = all_reduce(outputs, group)
    return routed_out, shared_out


@dataclass(frozen=True)
class Kimi3MoEExecutionPlan:
    """Construction-time orchestration selected for Kimi-K3 latent MoE."""

    use_native: bool
    use_trtllm: bool
    overlap_shared_experts: bool
    joint_moe_reduce: bool
    use_marlin: bool = False
    fused_moe_ar: bool = False
    lane_latent_norm_ar: bool = False
    comm_fusion_max_num_tokens: int = 0

    @property
    def use_precomputed_topk(self) -> bool:
        return self.use_native or self.use_trtllm or self.use_marlin

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
        # Hopper (SM90) has no native FP4 tensor cores and no flashinfer SiTU
        # cubin, so K3's MXFP4 SiTU MoE runs weight-only through the Marlin
        # W4A16 GEMM with a fused Triton SiTU epilogue. AUTO picks it whenever
        # neither the AMD-native nor the (Blackwell) TRT-LLM path is available;
        # it can also be forced with ``--moe-backend marlin``.
        use_marlin = not use_native and (
            moe_backend.is_marlin()
            or (moe_backend.is_auto() and _marlin_moe_available())
        )
        use_trtllm = (
            not use_native
            and not use_marlin
            and (moe_backend.is_auto() or moe_backend.is_flashinfer_trtllm())
        )
        return cls(
            use_native=use_native,
            use_trtllm=use_trtllm,
            use_marlin=use_marlin,
            overlap_shared_experts=(
                use_native
                and enforce_eager
                and alt_stream is not None
                and mapping.moe.tp_ep_size == 1
            ),
            joint_moe_reduce=use_native and mapping.moe.tp_ep_size > 1,
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
    """Latent projection with kernel-owned specialization.

    Tuned shapes use registered accelerator kernels. Other shapes retain the
    ordinary dense projection without requiring model-side shape selection.

    With ``shard_group`` the output dimension is partitioned across the group
    (column parallel): each rank stores ``output_size / tp`` rows of the weight
    and the ordinary projection ends with one all-gather. K3's tail paths avoid
    that gather: the fused multicast tail consumes the shard directly, while
    the other tiers inject each rank's column block into a reduction they
    already run. Thus every tail retains ``(tp-1)/tp`` of the weight's memory
    savings without adding collective wire bytes.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
        solution: str = "auto",
        shard_group: Group | None = None,
        shard_rank: int = 0,
        shard_size: int = 1,
    ) -> None:
        if shard_group is not None and output_size % shard_size:
            raise ValueError(
                f"output_size {output_size} is not divisible by the shard size "
                f"{shard_size}"
            )
        self.shard_group = shard_group
        self.shard_rank = shard_rank
        self.shard_size = shard_size if shard_group is not None else 1
        self.output_size_full = output_size
        super().__init__(
            input_size=input_size,
            output_size=output_size // self.shard_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.solution = solution

    def weight_loader(self, param, loaded_weight, shard_id=None, begin_size=None):
        """Take this rank's contiguous row block of the full weight."""
        if self.shard_group is not None and (
            shard_id is not None or begin_size is not None
        ):
            # Chunked loads use full-weight row offsets, which sharded params cannot honor.
            raise ValueError(
                "column-parallel Kimi3LatentProjection only supports whole-"
                "tensor loads"
            )
        if self.shard_group is not None:
            rows = self.output_size_full // self.shard_size
            loaded_weight = loaded_weight.narrow(0, self.shard_rank * rows, rows)
        return super().weight_loader(
            param, loaded_weight, shard_id=shard_id, begin_size=begin_size
        )

    def _gather_shards(self, local: torch.Tensor) -> torch.Tensor:
        """Concatenate every rank's column block into the full width."""
        if self.shard_group is None:
            return local
        num_tokens, shard = local.shape
        stacked = torch.empty(
            (self.shard_size, num_tokens, shard),
            dtype=local.dtype,
            device=local.device,
        )
        all_gather_into_tensor(stacked, local.contiguous(), self.shard_group)
        return stacked.permute(1, 0, 2).reshape(num_tokens, self.output_size_full)

    def project_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``output_size/tp`` columns, ungathered.

        For callers that fold the gather into a collective they already run:
        the column blocks are disjoint, so placing each rank's block into a
        buffer that is about to be summed makes the sum concatenate them.
        """
        if self.shard_group is None:
            raise ValueError("project_shard requires a column-parallel projection")
        return tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            self.weight,
            solution=self.solution,
        )

    @property
    def shard_slice(self) -> tuple[int, int]:
        """``(start, width)`` of this rank's column block in the full width."""
        width = self.output_size_full // self.shard_size
        return self.shard_rank * width, width

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        output = tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            self.weight,
            solution=self.solution,
        )
        return self._gather_shards(output), None

    def project_shard_add_(
        self,
        hidden_states: torch.Tensor,
        shared_output: torch.Tensor,
        *,
        norm_weight: torch.Tensor | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        """Project this rank's column shard into its shared-output slice."""

        if self.shard_group is None:
            raise ValueError("project_shard_add_ requires a sharded projection")
        start, width = self.shard_slice
        output = shared_output.narrow(-1, start, width)
        return tokenspeed_kernel.kimi3_latent_projection_add(
            hidden_states,
            self.weight,
            output,
            norm_weight=norm_weight,
            eps=eps,
            out=output,
        )

    def forward_add3(
        self,
        hidden_states: torch.Tensor,
        addend_a: torch.Tensor,
        addend_c: torch.Tensor,
        *,
        norm_weight: torch.Tensor | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        """Project routed latents and accumulate two full-width addends.

        ``result = addend_a + hidden_states @ self.weight.T + addend_c``

        Args:
            hidden_states: Routed latent ``[m, input_size]``; with
                ``norm_weight`` the projection fuses its RMSNorm.
            addend_a: Full-width addend ``[m, output_size]`` (the residual
                prefix); column-parallel instances consume only this rank's
                column block.
            addend_c: Second full-width addend (the reduced shared output),
                consumed the same way.
            norm_weight: Optional fused-RMSNorm weight over ``input_size``.
            eps: Epsilon for the fused norm; required with ``norm_weight``.

        Returns:
            The accumulated projection at full width; column-parallel
            instances gather their blocks before returning.
        """
        if self.shard_group is None:
            return tokenspeed_kernel.kimi3_latent_projection_add3(
                hidden_states,
                self.weight,
                addend_a,
                addend_c,
                norm_weight=norm_weight,
                eps=eps,
            )
        rows = self.output_size_full // self.shard_size
        start = self.shard_rank * rows
        local = tokenspeed_kernel.kimi3_latent_projection_add3(
            hidden_states,
            self.weight,
            addend_a.narrow(-1, start, rows),
            addend_c.narrow(-1, start, rows),
            norm_weight=norm_weight,
            eps=eps,
        )
        return self._gather_shards(local)


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
        joint_reduce: bool = False,
        shared_expert_stream: torch.cuda.Stream | None = None,
        expert_parallel_group: tuple[int, ...] | None = None,
        joint_reduce_group: tuple[int, ...] | None = None,
        return_separate_outputs: bool = False,
        input_projections: InputProjector | None = None,
        joint_input_projections: JointInputProjector | None = None,
        joint_shared_weight: torch.Tensor | None = None,
        joint_expert_shared_max_tokens: int = 0,
        sharded_output_min_tokens: int = 0,
    ) -> None:
        super().__init__()
        if input_projections is not None and shared_experts is None:
            raise ValueError("input_projections requires shared_experts")
        if (joint_input_projections is None) != (joint_shared_weight is None):
            raise ValueError(
                "joint input projections and shared weight must be provided together"
            )
        if joint_input_projections is not None and not joint_reduce:
            raise ValueError("joint expert/shared projection requires joint_reduce")
        if joint_input_projections is not None and joint_expert_shared_max_tokens <= 0:
            raise ValueError("joint expert/shared projection requires a token limit")
        if joint_input_projections is None and joint_expert_shared_max_tokens:
            raise ValueError("joint expert/shared token limit requires projections")
        if sharded_output_min_tokens < 0:
            raise ValueError("sharded output token limit must be nonnegative")
        if sharded_output_min_tokens and not joint_reduce:
            raise ValueError("sharded output projection requires joint_reduce")
        if shared_reduce is not None and shared_experts is None:
            raise ValueError("shared_reduce requires shared_experts")
        if joint_reduce and shared_experts is None:
            raise ValueError("joint_reduce requires shared_experts")
        if joint_reduce and (latent_reduce is not None or shared_reduce is not None):
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
        if joint_reduce_group is not None:
            joint_reduce_group = tuple(joint_reduce_group)
        if joint_reduce and joint_reduce_group is None:
            joint_reduce_group = expert_parallel_group
        if joint_reduce and joint_reduce_group is None:
            raise ValueError("joint_reduce requires a reduction group")
        if expert_parallel_size > 1 and latent_reduce is None and not joint_reduce:
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
        self.expert_parallel_group = expert_parallel_group
        self.joint_reduce_group = joint_reduce_group
        self.stream_fork = StreamFork(shared_expert_stream)
        self.return_separate_outputs = return_separate_outputs
        self.input_projections = input_projections
        self.joint_input_projections = joint_input_projections
        self.joint_shared_weight = joint_shared_weight
        self.joint_expert_shared_max_tokens = joint_expert_shared_max_tokens
        self.sharded_output_min_tokens = sharded_output_min_tokens

    def finalize_output(
        self,
        routed_latent: torch.Tensor,
        prefix_sum: torch.Tensor,
        shared_output: torch.Tensor,
    ) -> torch.Tensor:
        """Finish routed normalization/projection and add both residuals."""

        output_shape = tuple(shared_output.shape)
        _check_shape(prefix_sum, output_shape, "prefix_sum")
        if self.routed_norm is None:
            output = self.routed_up_proj.forward_add3(
                routed_latent,
                prefix_sum,
                shared_output,
            )
        elif routed_latent.shape[0] == 1 and current_platform().is_cdna4:
            output = self.routed_up_proj.forward_add3(
                routed_latent,
                prefix_sum,
                shared_output,
                norm_weight=self.routed_norm.weight,
                eps=self.routed_norm.variance_epsilon,
            )
        else:
            routed_latent = _module_tensor_output(self.routed_norm, routed_latent)
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
        use_sharded_output = (
            self.sharded_output_min_tokens > 0
            and num_tokens >= self.sharded_output_min_tokens
            and prefix_sum is not None
            and getattr(self.routed_up_proj, "shard_group", None) is not None
        )
        reduction_outputs = (
            acquire_all_reduce_outputs(
                (output_shape, (num_tokens, int(self.experts.hidden_size))),
                hidden_states,
                self.joint_reduce_group,
            )
            if self.joint_reduce and not use_sharded_output and num_tokens > 0
            else None
        )
        shared_target, routed_target = (
            reduction_outputs if reduction_outputs is not None else (None, None)
        )
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
        fuse_expert_shared = (
            self.joint_input_projections is not None
            and not overlap_shared
            and 0 < num_tokens <= self.joint_expert_shared_max_tokens
        )
        if fuse_expert_shared:
            packed = self.joint_input_projections(hidden_states, shared_target)
            fuse_expert_shared = packed is not None
        if (
            packed is None
            and self.input_projections is not None
            and not overlap_shared
            and num_tokens > 0
        ):
            packed = self.input_projections(hidden_states, shared_target)
        packed_router, packed_routed, packed_shared = (
            (None, None, None) if packed is None else packed
        )

        def run_shared_branch() -> None:
            nonlocal shared_output, shared_reduction_applied
            if self.shared_experts is None:
                return
            if fuse_expert_shared:
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
                else (
                    self.shared_experts(hidden_states, down_out=shared_target)
                    if shared_target is not None
                    else _module_tensor_output(self.shared_experts, hidden_states)
                )
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
            previous_output = getattr(self.experts, "_situ_output_buffer", None)
            if routed_target is not None:
                self.experts._situ_output_buffer = routed_target
            try:
                expert_kwargs = (
                    {
                        "shared_input": packed_shared,
                        "shared_weight": self.joint_shared_weight,
                        "shared_out": shared_target,
                    }
                    if fuse_expert_shared
                    else {}
                )
                expert_result = self.experts(
                    hidden_states=routed_input,
                    topk_output=topk_output,
                    num_global_tokens=num_global_tokens,
                    max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                    **expert_kwargs,
                )
            finally:
                if routed_target is not None:
                    self.experts._situ_output_buffer = previous_output
            if fuse_expert_shared:
                if not isinstance(expert_result, tuple):
                    raise RuntimeError(
                        "joint expert/shared kernel did not return both outputs"
                    )
                routed_latent, shared_output = expert_result
            else:
                routed_latent = expert_result
            _check_shape(routed_latent, latent_shape, "routed experts")

        # Spin-wait collectives cannot safely overlap an all-device GEMM.
        # Join both compute branches first. Individual reducers retain
        # the established shared-before-routed order; a joint reducer handles
        # both partials after the routed experts finish.
        if overlap_shared and self.shared_reduce is not None:
            shared_output = self.shared_reduce(shared_output)
            _check_shape(shared_output, output_shape, "shared_reduce")
            shared_reduction_applied = True

        if reduction_outputs is not None:
            shared_output, routed_latent = all_reduce(
                reduction_outputs,
                self.joint_reduce_group,
            )
            _check_shape(shared_output, output_shape, "joint shared output")
            _check_shape(routed_latent, latent_shape, "joint routed latent")
            shared_reduction_applied = True
        elif use_sharded_output:
            routed_latent = all_reduce(routed_latent, self.joint_reduce_group)
            _check_shape(routed_latent, latent_shape, "latent_reduce")
        elif self.latent_reduce is not None:
            routed_latent = self.latent_reduce(routed_latent)
            _check_shape(routed_latent, latent_shape, "latent_reduce")
        if shared_output is None:
            if self.routed_norm is not None:
                routed_latent = _module_tensor_output(self.routed_norm, routed_latent)
                _check_shape(routed_latent, latent_shape, "routed_norm")
            routed_output = _module_tensor_output(self.routed_up_proj, routed_latent)
            _check_shape(routed_output, output_shape, "routed_up_proj")
            return routed_output
        if use_sharded_output:
            norm_weight = None if self.routed_norm is None else self.routed_norm.weight
            eps = (
                None if self.routed_norm is None else self.routed_norm.variance_epsilon
            )
            self.routed_up_proj.project_shard_add_(
                routed_latent,
                shared_output,
                norm_weight=norm_weight,
                eps=eps,
            )
            shared_output = all_reduce(shared_output, self.joint_reduce_group)
            _check_shape(shared_output, output_shape, "sharded output reduce")
            return prefix_sum + shared_output
        if prefix_sum is None:
            if self.routed_norm is not None:
                routed_latent = _module_tensor_output(self.routed_norm, routed_latent)
                _check_shape(routed_latent, latent_shape, "routed_norm")
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
    "kimi3_join_reduce_moe",
    "latent_moe_expert_shared_all_reduce",
]
