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
from tokenspeed_kernel.ops.moe.latent_down import KimiK3LatentDownOp
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed.comm_backend import Group
from tokenspeed.runtime.distributed.comm_ops import (
    COMM_ONESHOT_MAX_BYTES,
    acquire_all_reduce_outputs,
    all_gather,
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
_SUPPORTED_EP_SIZES = {1, 2, 4, 8}
# Widest batch the down mailbox claims; above it the column gather takes over.
DOWN_MAILBOX_MAX_TOKENS = 1280


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
    # Whether the routed and shared partials can be reduced together at all,
    # independent of whether a backend-owned lane is available to avoid the
    # concatenation. ``fused_moe_ar`` implies a lane and is TRT-LLM only;
    # ``join_moe_reduce`` only needs a grouped or concatenated all-reduce,
    # which every backend provides, so the join is available on AMD too.
    join_moe_reduce: bool = False
    lane_latent_norm_ar: bool = False
    comm_fusion_max_num_tokens: int = 0

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
        shard_up_projection: bool = False,
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
        # The join itself needs no backend-owned lane: kimi3_join_reduce_moe
        # falls back to a concatenated one-shot, or a grouped all-reduce when
        # the payload exceeds COMM_ONESHOT_MAX_BYTES. Both are portable, so the
        # tail can issue one collective per MoE layer instead of two wherever a
        # TP x EP group exists -- not only where TRT-LLM can arm a lane.
        #
        # Excluded when the up projection is sharded: that tail folds the
        # projection between two sequential all-reduces
        # (_tail_fused_lane_ar_sharded) rather than calling the join, so the
        # collective count is unchanged, while leaving SEPARATE_REDUCE would
        # also give up the routed_in_fork overlap with the shared branch.
        join_moe_reduce = mapping.moe.has_tp_ep and not shard_up_projection
        return replace(
            self,
            fused_moe_ar=fused_moe_ar,
            join_moe_reduce=join_moe_reduce,
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

    With ``multicast_down`` the projection instead takes a column shard that
    publishes each rank's block into every peer's mailbox, which keeps each
    output element computed on one rank and so leaves the numerics untouched.
    It covers the widths the op claims -- the decode graph's captures.

    With ``column_group`` the projection splits the same output columns over
    the group and one all-gather concatenates the blocks. It is taken only
    where a mailbox exists, which is to say only where the fabric can map
    symmetric memory, and it narrows storage the same way ``shard_group``
    does. That coupling is deliberate, and measured rather than inferred: the
    replica wins at every width where the fabric cannot carry the gather. The
    numbers are on ``_gather_shards``, which is where the two routes are
    described together; they are not repeated here because an earlier eager
    set that was repriced away lived in this paragraph, two decimal places
    from a value in the other arm.

    Where the fabric is there, the split pays from the mailbox ceiling
    upward: measured on GB300 TP8 under graph replay with cold weights, the
    shard wins from 1280 by about 5 percent out to 36 at 8192, and the
    fused NVLink gather writes the final layout for 135us where the buffer
    path needs 239us. It wins eager as well -- 64.3/53.8 at 2048, 101.8/78.7
    at 4096, 208.9/138.6 at 8192 -- so the width does not depend on whether
    the chunk was captured, and a deployment whose chunks exceed the capture
    ceiling needs nothing done here.

    ``get_is_cuda_graph_phase`` cannot decide those two regimes apart, which
    is worth knowing because it reads as though it could. Its only setters
    are the decode wrapper's capture and its comm prewarm, and the executor
    runs prefill capture as a later statement, after that capture has
    already restored the flag -- so it is False throughout prefill capture.
    Nothing here keys on it, and nothing should.
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
        column_group: Group | None = None,
        shard_rank: int = 0,
        shard_size: int = 1,
        multicast_down: KimiK3LatentDownOp | None = None,
    ) -> None:
        if shard_group is not None and column_group is not None:
            raise ValueError("a column shard already gathers over its own group")
        group = shard_group if shard_group is not None else column_group
        if group is not None and output_size % shard_size:
            raise ValueError(
                f"output_size {output_size} is not divisible by the shard size "
                f"{shard_size}"
            )
        if group is not None and len(group) != shard_size:
            raise ValueError(
                f"group of {len(group)} ranks does not match the shard "
                f"size {shard_size}"
            )
        if multicast_down is not None:
            if shard_group is not None:
                raise ValueError("a column shard cannot also multicast")
            if multicast_down.rank != shard_rank:
                raise ValueError(
                    f"multicast rank {multicast_down.rank} does not match the "
                    f"projection's {shard_rank}"
                )
            if multicast_down.shard_dim * shard_size != output_size:
                raise ValueError(
                    f"multicast block {multicast_down.shard_dim} x {shard_size} "
                    f"does not cover the {output_size} outputs"
                )
        if not 0 <= shard_rank < shard_size:
            raise ValueError(
                f"shard_rank {shard_rank} is outside the shard size {shard_size}"
            )
        self.multicast_down = multicast_down
        self.shard_group = shard_group
        self.column_group = column_group
        self.shard_rank = shard_rank
        self.shard_size = shard_size if group is not None else 1
        self.output_size_full = output_size
        # Narrow only where the replica is unreachable: where a mailbox exists.
        self.narrowed = shard_group is not None or (
            column_group is not None and multicast_down is not None
        )
        super().__init__(
            input_size=input_size,
            output_size=(output_size // shard_size if self.narrowed else output_size),
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.solution = solution

    def weight_loader(self, param, loaded_weight, shard_id=None, begin_size=None):
        """Take this rank's column block; replicated instances load full width."""
        if self.narrowed and (shard_id is not None or begin_size is not None):
            # Chunked loads use full-weight row offsets, which sharded params cannot honor.
            raise ValueError(
                "column-parallel Kimi3LatentProjection only supports whole-tensor loads"
            )
        if self.narrowed:
            rows = self.output_size_full // self.shard_size
            loaded_weight = loaded_weight.narrow(0, self.shard_rank * rows, rows)
        return super().weight_loader(
            param, loaded_weight, shard_id=shard_id, begin_size=begin_size
        )

    def _gather_shards(self, local: torch.Tensor) -> torch.Tensor:
        """Concatenate every rank's column block into the full output width.

        Two routes, chosen by which group this instance was given; they are
        mutually exclusive at every construction site and a caller never picks.

        ``shard_group`` gathers into a ``[tp, tokens, shard]`` buffer and
        permutes it by hand, deliberately not asking the backend for the
        last-dim layout: that would move this projection onto a symmetric
        memory path it has never been measured on, and would rendezvous a
        second workspace sized to the full hidden width.

        ``column_group`` asks the backend for that layout instead, and only
        where storage narrowed: a full-width projection already holds every
        column and has nothing to concatenate. That request reaches
        ``all_gather_inner`` -- the fused NVLink gather writes the final layout
        in one pass, 135us at 8152 tokens where a buffer-and-permute needs 239.

        It reaches that kernel only on NVIDIA, with a 2-D bf16 tensor gathered
        on the last dim; anything else, and any group the fabric cannot map,
        falls back to the NCCL backend's allocate-gather-transpose, and that
        fallback loses to the replica at every width. Measured under graph
        replay on 8 ranks over two hosts, with the route witnessed rather than
        assumed -- nccl 33 calls and rsag 0 with the probe declined, the
        reverse with it live:

            m       no fabric (nccl)      with fabric (rsag)
            1280    37.2 rep / 76.8 col   37.7 rep / 35.6 col
            4096   111.4     / 121.5     111.5     /  72.6
            8192   226.0     / 254.0     224.5     / 142.8

        Which is why a projection that cannot narrow does not take this route
        at all. An earlier eager measurement here put the no-fabric gap at
        4.5x rather than 2.1x; it timed host submission inside the interval,
        and the two arms submit different amounts of Python.

        Returns:
            The full-width projection. **It may alias the backend's workspace**,
            which the next gather reuses -- measured same-address across
            consecutive calls. Read it on the issuing stream before gathering
            again. The contract is stated for both routes even though only the
            column one can alias, because a caller that had to know which route
            it got would be the branch this method exists to remove.
        """
        if self.shard_group is not None:
            num_tokens, shard = local.shape
            stacked = torch.empty(
                (self.shard_size, num_tokens, shard),
                dtype=local.dtype,
                device=local.device,
            )
            all_gather_into_tensor(stacked, local.contiguous(), self.shard_group)
            return stacked.permute(1, 0, 2).reshape(num_tokens, self.output_size_full)
        if not self.narrowed or self.column_group is None:
            return local
        return all_gather(local.contiguous(), self.column_group, dim=-1)

    @property
    def weight_block(self) -> torch.Tensor:
        """This rank's weight rows, which narrowed storage already holds.

        Both sharding routes narrow, and a projection that did not narrow never
        reaches here: without a mailbox the column path is not taken at all, so
        there is no full-width tensor left to slice.
        """
        return self.weight

    def project_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``output_size/tp`` columns, ungathered.

        For callers that fold the gather into a collective they already run:
        the column blocks are disjoint, so placing each rank's block into a
        buffer that is about to be summed makes the sum concatenate them.
        """
        if self.shard_group is None and self.column_group is None:
            raise ValueError("project_shard requires a column-parallel projection")
        return tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            self.weight_block,
            solution=self.solution,
        )

    @property
    def shard_slice(self) -> tuple[int, int]:
        """``(start, width)`` of this rank's column block in the full width."""
        if self.shard_group is None and self.column_group is None:
            raise ValueError("shard_slice requires a column-parallel projection")
        width = self.output_size_full // self.shard_size
        return self.shard_rank * width, width

    def _project_replicated(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project through whatever this rank stores, gathering nothing."""
        return tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            self.weight,
            solution=self.solution,
        )

    def _multicast_block(self) -> torch.Tensor:
        """This rank's rows for the mailbox, whatever the storage layout."""
        if self.narrowed:
            return self.weight
        rows = self.multicast_down.shard_dim
        return self.weight[self.shard_rank * rows : (self.shard_rank + 1) * rows]

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        num_tokens = hidden_states.shape[0]
        if self.multicast_down is not None and self.multicast_down.handles(num_tokens):
            return self.multicast_down(hidden_states, self._multicast_block()), None
        if self.narrowed and self.column_group is not None:
            return self._gather_shards(self.project_shard(hidden_states)), None
        return self._gather_shards(self._project_replicated(hidden_states)), None

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
                prefix); ``shard_group`` instances consume only this rank's
                column block.
            addend_c: Second full-width addend (the reduced shared output),
                consumed the same way.
            norm_weight: Optional fused-RMSNorm weight over ``input_size``.
            eps: Epsilon for the fused norm; required with ``norm_weight``.

        Returns:
            The accumulated projection at full width; instances that narrowed
            their storage project their block and gather it.

        The branch keys on ``narrowed`` rather than on ``shard_group``, which
        used to be the same question. It stopped being so when the column path
        gained its own way to narrow: the proxy quietly lost its meaning, and
        the full-width branch would have projected a block while calling it the
        whole width. Anyone adding a third narrowing path should re-run that
        enumeration over every ``shard_group is None`` left in this class.
        """
        if not self.narrowed:
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
        return_separate_outputs: bool = False,
        input_projections: InputProjector | None = None,
    ) -> None:
        super().__init__()
        if input_projections is not None and shared_experts is None:
            raise ValueError("input_projections requires shared_experts")
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
        if joint_reduce and expert_parallel_group is None:
            raise ValueError("joint_reduce requires expert_parallel_group")
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
        self.stream_fork = StreamFork(shared_expert_stream)
        self.return_separate_outputs = return_separate_outputs
        self.input_projections = input_projections

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
        reduction_outputs = (
            acquire_all_reduce_outputs(
                (output_shape, (num_tokens, int(self.experts.hidden_size))),
                hidden_states,
                self.expert_parallel_group,
            )
            if self.joint_reduce and num_tokens > 0
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
        if self.input_projections is not None and not overlap_shared and num_tokens > 0:
            packed = self.input_projections(hidden_states, shared_target)
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
                routed_latent = self.experts(
                    hidden_states=routed_input,
                    topk_output=topk_output,
                    num_global_tokens=num_global_tokens,
                    max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                )
            finally:
                if routed_target is not None:
                    self.experts._situ_output_buffer = previous_output
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
                self.expert_parallel_group,
            )
            _check_shape(shared_output, output_shape, "joint shared output")
            _check_shape(routed_latent, latent_shape, "joint routed latent")
            shared_reduction_applied = True
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
