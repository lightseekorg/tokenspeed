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

"""Auto backend: per-call strategy selection.

Wraps NCCL and optional low-latency GPU backends. CUDA IPC and symmetric-memory
backends are only selected for node-local groups; groups spanning nodes fall
back to NCCL. ``AutoBackend.route`` is the one place that decides which
implementation serves a collective; every public method asks it first.
"""

import enum
import logging

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import (
    CommBackend,
    CommWork,
    Group,
)
from tokenspeed.runtime.distributed.comm_backend.iris import (
    IrisBackend,
)
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_backend.projection import (
    ProjectionBackend,
    ProjectionWorkspace,
)
from tokenspeed.runtime.distributed.comm_backend.triton_rsag import TritonRSAGBackend
from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
    MAX_ONESHOT_BYTES,
    TrtllmAllReduceBackend,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)


def ordered_fold_sum(parts: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Fold ``[world, ...]`` partial sums left to right in fp32 into ``out``.

    The association order depends only on the rank axis, never on the payload
    shape, so a row's folded value is bitwise identical at any batch size.
    GPU partials whose last axis is contiguous fold in one kernel with the
    same order and rounding.
    """
    if parts.is_cuda and parts.stride(-1) == 1 and out.stride(-1) == 1:
        from tokenspeed_kernel.ops.communication.triton import rank_ordered_fold

        width = out.shape[-1] if out.dim() else 1
        rank_ordered_fold(parts.reshape(parts.shape[0], -1, width), out.view(-1, width))
        return out
    accumulator = parts[0].to(torch.float32)
    for rank in range(1, parts.shape[0]):
        accumulator = accumulator + parts[rank].to(torch.float32)
    # A -0.0 sum leaves as +0.0, as the kernel stores it.
    out.copy_((accumulator + 0.0).to(out.dtype))
    return out


class Collective(enum.Enum):
    """The collectives ``AutoBackend.route`` decides for."""

    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    TOKEN_REDUCE_SCATTER = "token_reduce_scatter"
    ALL_GATHER = "all_gather"
    TOKEN_ALL_GATHER = "token_all_gather"

    @property
    def is_reduction(self) -> bool:
        return self in (
            Collective.ALL_REDUCE,
            Collective.REDUCE_SCATTER,
            Collective.TOKEN_REDUCE_SCATTER,
        )


class Route(enum.Enum):
    """Which implementation serves a collective."""

    # The vendor collective (NCCL / RCCL).
    NCCL = "nccl"
    # NCCL data movement plus the rank-ordered fp32 fold: batch-invariant,
    # at world_size x the traffic for an all-reduce.
    ORDERED_FOLD = "ordered_fold"
    # Symmetric-memory multicast: multicast stores for the gathers, the
    # in-switch reduction for the token reduce-scatter (performance only).
    MULTIMEM = "multimem"
    # The all-reduce's performance tiers (trtllm / Iris / NCCL), outside
    # any numerics contract.
    TIERED = "tiered"


class AutoBackend(CommBackend):
    """Composite backend that selects the best strategy per call."""

    def __init__(self):
        self._nccl = NcclBackend()
        self._trtllm_ar = TrtllmAllReduceBackend(fallback=self._nccl)
        self._iris = IrisBackend(fallback=self._nccl)
        self._rsag = TritonRSAGBackend(fallback=self._nccl)
        # Projection fallbacks retain Auto's ordinary routing and numerics.
        self._projection = ProjectionBackend(fallback=self)

    @property
    def nccl(self) -> NcclBackend:
        return self._nccl

    @property
    def iris(self) -> IrisBackend:
        return self._iris

    @property
    def trtllm_ar(self) -> TrtllmAllReduceBackend:
        return self._trtllm_ar

    def configure(self, use_pynccl: bool = False) -> None:
        self._nccl.configure(use_pynccl=use_pynccl)

    def for_named_groups(self, name: str) -> CommBackend:
        # NCCL alone: the low-latency backends hold per-group state prepared
        # for the shared groups.
        return self._nccl.for_named_groups(name)

    @staticmethod
    def _force_deterministic_rsag() -> bool:
        return bool(global_server_args_dict.get("force_deterministic_rsag", False))

    @staticmethod
    def _batch_invariant_collectives() -> bool:
        return bool(global_server_args_dict.get("batch_invariant_collectives", False))

    def _ordered_fold_all_reduce(
        self, tensor: torch.Tensor, group: Group, op
    ) -> torch.Tensor:
        """All-gather the partials and fold them in fixed rank order.

        Every element folds rank 0..n-1 left to right in fp32 with one final
        rounding, independent of the tensor's shape -- unlike a ring
        all-reduce, whose per-element association order follows the
        size-dependent chunking. An armed trtllm workspace that holds the
        rows folds the same way in one kernel and returns a new tensor; the
        gather path works in place like the NCCL all-reduce.
        """
        if op is not None and op != torch.distributed.ReduceOp.SUM:
            raise ValueError("batch-invariant collectives fold SUM reductions only")
        world_size = len(group)
        if world_size == 1:
            return tensor
        if not self._force_deterministic_rsag() and self._trtllm_ar.has_trtllm_ar(
            group
        ):
            folded = self._trtllm_ar.ordered_all_reduce(tensor, group)
            if folded is not None:
                return folded
        rows = tensor.contiguous().view(-1, tensor.shape[-1] if tensor.dim() else 1)
        # Last-dim gather on the multicast kernels: [rows, world * width], rank-major.
        gathered = self.all_gather(rows, group, dim=-1)
        parts = gathered.view(rows.shape[0], world_size, rows.shape[1]).transpose(0, 1)
        ordered_fold_sum(parts, rows)
        if rows.data_ptr() != tensor.data_ptr():
            tensor.copy_(rows.view(tensor.shape))
        return tensor

    def _ordered_fold_reduce_scatter(
        self, tensor: torch.Tensor, group: Group
    ) -> torch.Tensor:
        """Exchange each rank's chunks, then fold them in fixed rank order.

        ``tensor`` holds ``len(group)`` equal chunks along dim 0, chunk ``i``
        destined for the group's ``i``-th rank. The all-to-all moves the same
        bytes a reduce-scatter does; each rank then folds the chunks it
        received from ranks 0..n-1 in fp32, so its output does not depend on
        the payload size the way a ring reduce-scatter's association does.
        """
        world_size = len(group)
        if world_size == 1:
            return tensor
        if tensor.shape[0] % world_size:
            raise ValueError(
                f"reduce-scatter input rows ({tensor.shape[0]}) must split evenly "
                f"across {world_size} ranks"
            )
        received = torch.empty_like(tensor)
        self._nccl.all_to_all_single(received, tensor.contiguous(), group)
        parts = received.view(
            (world_size, tensor.shape[0] // world_size, *tensor.shape[1:])
        )
        return ordered_fold_sum(parts, torch.empty_like(parts[0]))

    def _ordered_fold_token_reduce_scatter(
        self, tensor: torch.Tensor, group: Group, scattered_num_tokens: list[int]
    ) -> torch.Tensor:
        """Token reduce-scatter through the ordered fold (uneven token split).

        Pads every rank's token slice to the largest one, as the NCCL path
        does, so the exchange splits evenly; the padding rows are dropped.
        """
        max_tokens = max(scattered_num_tokens)
        padded = tensor.new_zeros(
            (len(scattered_num_tokens) * max_tokens, tensor.shape[-1])
        )
        offset = 0
        for rank_index, count in enumerate(scattered_num_tokens):
            padded[rank_index * max_tokens : rank_index * max_tokens + count].copy_(
                tensor[offset : offset + count]
            )
            offset += count
        folded = self._ordered_fold_reduce_scatter(padded, group)
        rank_index = group.index(torch.distributed.get_rank())
        return folded[: scattered_num_tokens[rank_index]].contiguous()

    @staticmethod
    def _group_spans_nodes(group: Group) -> bool:
        mapping = global_server_args_dict.get("mapping")
        if mapping is None or not mapping.nprocs_per_node:
            return False
        nprocs_per_node = mapping.nprocs_per_node
        return len({rank // nprocs_per_node for rank in group}) > 1

    @staticmethod
    def _multicast_reachable(group: Group) -> bool:
        """Whether symmetric-memory multicast can map across ``group``.

        The rsag paths rendezvous a symmetric buffer and store through its
        multicast pointer, so a group the fabric cannot map hangs inside the
        rendezvous instead of falling back. Topology alone was too strict --
        a rack's NVLink domain can span hosts, and vetoing on spread gives
        those groups NCCL forever -- so it now only admits, and anything
        crossing a host is probed rather than refused.

        The rank count is not a substitute for the topology test. ``Mapping``
        builds strided groups: an attention DP group is ``(0, 8)`` at
        ``attn_tp_size=8``, which is smaller than one host's device count while
        living on two hosts, so counting would admit it with no probe at all.

        The world fabric map is gathered during distributed initialization, so
        the group verdict is a local lookup with no dispatch-time collective.
        """
        from tokenspeed_kernel.ops.communication.fabric import group_has_fabric

        if not AutoBackend._group_spans_nodes(group):
            return True
        return group_has_fabric(group)

    def route(
        self, collective: Collective, tensor: torch.Tensor, group: Group
    ) -> Route:
        """The one routing decision for every collective this backend serves.

        In precedence order:

        1. ``--force-deterministic-rsag``: no symmetric-memory path. A
           reduction takes the ordered fold under
           ``--batch-invariant-collectives`` and NCCL otherwise; a gather
           takes NCCL.
        2. ``--batch-invariant-collectives``: one association order per
           reduction, independent of the batch. Every reduction takes the
           ordered fold. The NVLS in-switch reduction does not qualify even
           with a fixed issuer: its association order moves with a row's
           offset in the buffer, so a row reduced at another position in
           another batch can round differently (docs/design/numerics.md,
           "Why not the in-switch reduction").
        3. Otherwise the performance defaults: gathers and the token
           reduce-scatter take the multicast kernels where multicast reaches
           (the RSAG backend itself falls back to NCCL past its capacity),
           the plain reduce-scatter NCCL, the all-reduce its tiered dispatch.

        Gathers are pure data movement, so the envelope leaves them to the
        performance default.
        """
        if collective.is_reduction and self._batch_invariant_collectives():
            return Route.ORDERED_FOLD
        if self._force_deterministic_rsag():
            return Route.NCCL
        if collective is Collective.ALL_REDUCE:
            return Route.TIERED
        if collective is Collective.REDUCE_SCATTER:
            return Route.NCCL
        return Route.MULTIMEM if self._multicast_reachable(group) else Route.NCCL

    # ---- Token-aware ops ----

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        if self.route(Collective.TOKEN_ALL_GATHER, tensor, group) is Route.MULTIMEM:
            return self._rsag.token_all_gather(tensor, group, scattered_num_tokens)
        return self._nccl.token_all_gather(tensor, group, scattered_num_tokens)

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        route = self.route(Collective.TOKEN_REDUCE_SCATTER, tensor, group)
        if route is Route.ORDERED_FOLD:
            return self._ordered_fold_token_reduce_scatter(
                tensor, group, scattered_num_tokens
            )
        if route is Route.MULTIMEM:
            return self._rsag.token_reduce_scatter(tensor, group, scattered_num_tokens)
        return self._nccl.token_reduce_scatter(tensor, group, scattered_num_tokens)

    # ---- Public CommBackend interface ----

    def get_prepared_all_gather(self, group: Group, dtype: torch.dtype):
        if (
            not current_platform().is_cdna4
            or len(group) not in (4, 8)
            or self._force_deterministic_rsag()
            or self._group_spans_nodes(group)
            or self._trtllm_ar.has_trtllm_ar(group)
        ):
            return None
        return self._iris.get_prepared_all_gather(group, dtype)

    def broadcast_from_first_rank(self, tensor: torch.Tensor, group: Group) -> bool:
        if (
            not current_platform().is_cdna4
            or len(group) not in (4, 8)
            or self._group_spans_nodes(group)
        ):
            return False
        return self._iris.broadcast_from_first_rank(tensor, group)

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self._batch_invariant_collectives():
            # Each tensor folds on its own; grouping is a NCCL optimization
            # the contract has no use for.
            if isinstance(tensor, torch.Tensor):
                return self._all_reduce_one(tensor, group, op)
            return tuple(self._all_reduce_one(value, group, op) for value in tensor)
        if not isinstance(tensor, torch.Tensor):
            tensors = tensor
            if len(tensors) == 0:
                raise ValueError("all-reduce requires at least one tensor")
            use_nccl = self._force_deterministic_rsag() or self._group_spans_nodes(
                group
            )
            if (
                not use_nccl
                and current_platform().is_amd
                and self._iris.can_reduce_outputs(tensors, group, op=op)
            ):
                return self._iris.all_reduce(tensors, group, op=op)
            # Collections past the one-shot window are headed for NCCL;
            # grouping avoids the copy required to concatenate them first.
            use_nccl = use_nccl or all(
                value.numel() * value.element_size() > MAX_ONESHOT_BYTES
                for value in tensors
            )
            use_nccl = use_nccl or (
                current_platform().is_amd
                and sum(value.numel() * value.element_size() for value in tensors)
                > self._iris.producer_direct_max_bytes
            )
            if use_nccl and len(tensors) == 2:
                return self._nccl.all_reduce_two(*tensors, group, op=op)
            return super().all_reduce(tensors, group, op=op)
        return self._all_reduce_one(tensor, group, op)

    def _all_reduce_one(self, tensor: torch.Tensor, group: Group, op) -> torch.Tensor:
        """All-reduce a single tensor along the route ``route`` picked for it."""
        route = self.route(Collective.ALL_REDUCE, tensor, group)
        if route is Route.ORDERED_FOLD and op in (
            torch.distributed.ReduceOp.MAX,
            torch.distributed.ReduceOp.MIN,
        ):
            # Max and min are exact in any reduction order, so NCCL already
            # gives the batch-invariant result the fold exists for.
            return self._nccl.all_reduce(tensor, group, op=op)
        if route is Route.ORDERED_FOLD:
            return self._ordered_fold_all_reduce(tensor, group, op)
        if route is Route.NCCL:
            return self._nccl.all_reduce(tensor, group, op=op)
        # Tiered dispatch -- first match wins. This is Tier 1 (which backend);
        # the trtllm backend then runs Tier 2 (mnnvl vs IPC, by pattern, payload
        # bytes and the IPC kernel's strategy) inside _ar_fusion_workspace.
        #   1. trtllm_ar armed for this group ...... trtllm_ar   (mnnvl / IPC fusion)
        #   2. group spans nodes ................... NCCL
        #   3. producer-owned memory or Iris can run Iris
        #   4. otherwise ........................... NCCL
        spans_nodes = self._group_spans_nodes(group)
        # trtllm_ar carries an mnnvl workspace that spans nodes; it is only
        # armed for a group when that succeeded, so has_trtllm_ar() is itself
        # the "usable here" test. Checking it before the cross-node NCCL
        # fallback is what lets a cross-node group use mnnvl at all -- otherwise
        # the workspace is armed and never called.
        if self._trtllm_ar.has_trtllm_ar(group):
            return self._trtllm_ar.all_reduce(tensor, group, op=op)
        if spans_nodes:
            return self._nccl.all_reduce(tensor, group, op=op)
        if self._iris.can_reduce_outputs((tensor,), group, op=op):
            # The producer wrote into acquired Iris symmetric memory, so the
            # reduction skips the staging copy.
            return self._iris.all_reduce((tensor,), group, op=op)[0]
        if self._iris.can_run(tensor, group, op=op):
            return self._iris.all_reduce(tensor, group, op=op)
        return self._nccl.all_reduce(tensor, group, op=op)

    def supports_all_reduce_mhc_norm(
        self, x: torch.Tensor, norm_weight: torch.Tensor, group: Group
    ) -> bool:
        return (
            not self._batch_invariant_collectives()
            and not self._force_deterministic_rsag()
            and self._trtllm_ar.supports_all_reduce_mhc_norm(x, norm_weight, group)
        )

    def all_reduce_mhc_norm(
        self, x, residual, post, comb, pre, weight, eps, group: Group
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._trtllm_ar.all_reduce_mhc_norm(
            x, residual, post, comb, pre, weight, eps, group
        )

    def all_reduce_mhc_post(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        group: Group,
    ) -> torch.Tensor | None:
        """Delegate fused all-reduce and mHC post to Iris when routing permits.

        Return None unless AMD uses a node-local tiered route without trtllm.
        """
        # Same eligibility as the producer-direct all-reduce in _all_reduce_one.
        if (
            not current_platform().is_amd
            or self.route(Collective.ALL_REDUCE, tensor, group) is not Route.TIERED
            or self._trtllm_ar.has_trtllm_ar(group)
            or self._group_spans_nodes(group)
        ):
            return None
        return self._iris.all_reduce_mhc_post(tensor, residual, post, comb, group)

    def prepare_all_reduce_lane(self, group: Group, hidden_dim: int) -> bool:
        return self._trtllm_ar.ensure_group_lane(group, hidden_dim)

    def can_prepare_all_reduce_buffers(self, group: Group, dtype: torch.dtype) -> bool:
        """Whether this group's performance route permits Iris preparation."""
        return not (
            not current_platform().is_amd
            or len(group) <= 1
            or dtype != torch.bfloat16
            or self._force_deterministic_rsag()
            or self._batch_invariant_collectives()
            or self._group_spans_nodes(group)
            or self._trtllm_ar.has_trtllm_ar(group)
        )

    def prepare_all_reduce_buffers(
        self,
        group: Group,
        *,
        staged_max_numel: int,
        producer_direct_max_numel: int,
        dtype: torch.dtype,
    ) -> bool:
        if not self.can_prepare_all_reduce_buffers(group, dtype):
            return False
        return self._iris.prepare_all_reduce_buffers(
            group,
            staged_max_numel=staged_max_numel,
            producer_direct_max_numel=producer_direct_max_numel,
            dtype=dtype,
        )

    def can_acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Whether ``acquire_all_reduce_outputs`` returns producer-direct memory.

        Mirrors the routing in ``acquire_all_reduce_outputs`` below: the cases
        that fall through to ``super()`` there get ordinary allocations, so they
        answer False here.
        """
        if (
            self._force_deterministic_rsag()
            or self._batch_invariant_collectives()
            or self._group_spans_nodes(group)
            or self._trtllm_ar.has_trtllm_ar(group)
        ):
            return False
        if current_platform().is_amd:
            return self._iris.can_acquire_outputs(shapes, like, group, op=op)
        return self._iris.can_acquire_all_reduce_outputs(shapes, like, group, op=op)

    def acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, ...]:
        """Acquire ordinary or producer-direct all-reduce outputs."""
        # The fold and NCCL read ordinary tensors; the
        # producer-direct staging below only serves the Iris all-reduce.
        if (
            self._force_deterministic_rsag()
            or self._batch_invariant_collectives()
            or self._group_spans_nodes(group)
            or self._trtllm_ar.has_trtllm_ar(group)
        ):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)
        if current_platform().is_amd and not self._iris.can_acquire_outputs(
            shapes,
            like,
            group,
            op=op,
        ):
            return super().acquire_all_reduce_outputs(shapes, like, group, op=op)
        return self._iris.acquire_all_reduce_outputs(
            shapes,
            like,
            group,
            op=op,
        )

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        if (
            self.route(Collective.ALL_GATHER, tensor, group) is Route.MULTIMEM
            and tensor.dim() == 2
            and dim in (-1, tensor.dim() - 1)
        ):
            if (
                current_platform().is_cdna4
                and not self._group_spans_nodes(group)
                and not self._trtllm_ar.has_trtllm_ar(group)
            ):
                return self._iris.all_gather(tensor, group, dim)
            # Small rows ride the all-reduce workspace's barrier-free exchange.
            gathered = self._trtllm_ar.lamport_all_gather(tensor, group)
            if gathered is not None:
                return gathered
            return self._rsag.all_gather(tensor, group, dim)
        return self._nccl.all_gather(tensor, group, dim)

    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        return self._nccl.all_gather_single(output, input, group)

    def all_gather_single_async(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> CommWork:
        return self._nccl.all_gather_single_async(output, input, group)

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        if self.route(Collective.REDUCE_SCATTER, tensor, group) is Route.ORDERED_FOLD:
            return self._ordered_fold_reduce_scatter(tensor, group)
        return self._nccl.reduce_scatter(tensor, group)

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: Group,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
    ) -> None:
        # Pure data movement: no reduction, so nothing to fold for the
        # batch-invariant envelope and no symmetric-memory path to veto.
        return self._nccl.all_to_all_single(
            output,
            input,
            group,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

    def prepare_projection(self, spec, scratch_pool=None):
        """Prepare optional Lamport paths without changing ordinary routing.

        Projection collectives have an explicit pre-capture lifetime. Their
        large-message and unsupported-topology paths return to this backend;
        batch-invariant reductions retain the rank-ordered fold at every size.
        """
        use_lamport = (
            current_platform().is_nvidia
            and spec.device.type == "cuda"
            and spec.dtype == torch.bfloat16
            and not self._force_deterministic_rsag()
        )
        batch_invariant = self._batch_invariant_collectives()
        return self._projection.prepare(
            spec,
            use_lamport,
            use_lamport and not batch_invariant,
            batch_invariant,
            scratch_pool,
        )

    def try_fused_dp_column_linear(
        self,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor | None,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
        workspace: ProjectionWorkspace,
    ) -> bool:
        return self._projection.try_fused_dp_column_linear(
            tensor, weight, weight_scales, owner_rows, out, workspace
        )

    def try_fused_dp_row_linear(
        self,
        tensor: torch.Tensor,
        input_scales: torch.Tensor | None,
        weight: torch.Tensor,
        weight_scales: torch.Tensor | None,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
        workspace: ProjectionWorkspace,
    ) -> bool:
        return self._projection.try_fused_dp_row_linear(
            tensor, input_scales, weight, weight_scales, owner_rows, out, workspace
        )

    def projection_all_gather(
        self,
        tensor: torch.Tensor,
        rows: int,
        quantize: bool,
        workspace: ProjectionWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._projection.all_gather(tensor, rows, quantize, workspace)

    def projection_all_to_all(
        self,
        tensor: torch.Tensor,
        rows: int,
        inverse: bool,
        quantize: bool,
        out: torch.Tensor | None,
        workspace: ProjectionWorkspace,
        *,
        input_scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._projection.all_to_all(
            tensor, rows, inverse, quantize, out, workspace, input_scales=input_scales
        )

    def acquire_projection_output(
        self, rows: int, workspace: ProjectionWorkspace
    ) -> torch.Tensor:
        return self._projection.acquire_output(rows, workspace)

    def projection_reduce_scatter(
        self, tensor: torch.Tensor, rows: int, workspace: ProjectionWorkspace
    ) -> torch.Tensor:
        return self._projection.reduce_scatter(tensor, rows, workspace)

    def send(self, tensor: torch.Tensor, dst: int, group: Group) -> None:
        return self._nccl.send(tensor, dst, group)

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        group: Group,
    ) -> torch.Tensor:
        return self._nccl.recv(size, dtype, device, src, group)
