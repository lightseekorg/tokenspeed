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
#
# Orchestrates the CuTe-DSL kernels vendored under
# thirdparty/cute_dsl/latent_moe_tail/ (from the vLLM project, Apache-2.0).

"""Multicast latent-MoE tail for Kimi-K3 decode.

Replaces the ``all-reduce(latent+shared lanes) -> replicated up-projection``
tail with three fused stages: one kernel doing AR(latent)+RMSNorm+RS(shared),
a *sharded* up-projection (each rank computes ``hidden/tp`` rows — 1/tp of the
weight traffic, and 1/tp of the weight storage when the caller passes a
column-parallel weight) whose epilogue multicast-stores the shard into every
rank's mailbox (NVLS), and a barrier-free Lamport gather. Symmetric buffers
come from stock ``torch.distributed._symmetric_memory``. A depth-d rotation
pool (d=2 by default) reduces their footprint from about 0.95 GB per rank for
roughly 92 MoE layers to d x about 6 MB per rank. The small per-call staging
can come from the runtime's shared workspace pool via ``scratch_allocator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

if TYPE_CHECKING:
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import (
        AdaptiveUpProjectionKernel,
        CollectiveKernel,
    )

_ScratchAllocator = Callable[..., list[torch.Tensor]]

logger = logging.getLogger(__name__)

_MAX_NUM_TOKENS = 64
_SKINNY_MAX_NUM_TOKENS = 5
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 8
_LAMPORT_COPY_CTAS = 32
_LAMPORT_COPY_THREADS = 224
_SUPPORTED_TP_SIZES = (8, 16)
# The gather releases its PDL dependents before re-arming this slot's sentinels.
# Rotating by decoder-layer index keeps adjacent layers off the same mailbox and,
# in the supported sequential decoder, leaves the intervening layer's stream work
# between same-slot writes. This is a scheduling assumption, not synchronization:
# different modules scheduled with repeated/non-sequential indices need separate
# model scopes (and concurrently executing models must not share a scope).
_TAIL_POOL_DEPTH = 2


def _tail_pool_slot(layer_index: int, depth: int = _TAIL_POOL_DEPTH) -> int:
    """Map a decoder layer index to its rank-identical pool slot."""
    if depth <= 0:
        raise ValueError("tail pool depth must be greater than zero")
    return layer_index % depth


def _allocator_identity(allocator: _ScratchAllocator | None) -> object:
    """Return the stable owner identity of a possibly bound allocator."""
    return getattr(allocator, "__self__", allocator)


def multicast_reachable(group: dist.ProcessGroup | None = None) -> bool:
    """Whether NVLS multicast can actually map across ``group``'s ranks.

    ``symm_mem`` importing is not enough: a cross-host group without fabric or
    IMEX still reports multicast support locally and then hangs inside the
    rendezvous instead of letting the caller fall back. The host-span test is
    at group granularity: a node-local subgroup of a multi-host job never
    needs fabric.
    """
    import torch.distributed as dist
    from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported

    if not dist.is_initialized():
        return False
    if dist.get_world_size(group) <= torch.cuda.device_count():
        return True
    return fabric_allocation_supported(torch.cuda.current_device())


def multicast_backend_unavailable_reason(
    group: dist.ProcessGroup | None = None,
) -> str | None:
    """Which term of the backend's eligibility fails here, or None.

    Callers that only branch want ``multicast_backend_available``. Callers that
    have to tell a machine which cannot host this from one which should have
    and did not need the term: capability and the optional imports say the
    former, an unreachable fabric says the latter, and only that last one is a
    fault rather than a configuration.
    """
    if not torch.cuda.is_available():
        return "no CUDA device"
    platform = current_platform()
    if not platform.is_nvidia:
        return f"{platform.vendor} does not carry the NVLS multicast path"
    # Only sm100 is measured; sm90 lacks the fabric handles this mailbox needs.
    if platform.arch_version.major < 10:
        return f"compute capability {platform.arch_version.major}, below 10"
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from torch.distributed import _symmetric_memory  # noqa: F401
    except ImportError:
        return "cutlass or symmetric memory is not importable"
    if not multicast_reachable(group):
        return "fabric unreachable"
    return None


def multicast_backend_available(group: dist.ProcessGroup | None = None) -> bool:
    """Whether the CuteDSL multicast backend can run here at all.

    Separate from any one op's shapes: capability, the optional imports, and
    fabric reachability. All three must hold before a collective rendezvous.
    """
    return multicast_backend_unavailable_reason(group) is None


def latent_tail_supported(
    *,
    tp_size: int,
    hidden_size: int,
    latent_size: int,
    dtype: torch.dtype,
    group: dist.ProcessGroup | None = None,
) -> bool:
    """Cheap, non-collective eligibility probe (no rendezvous).

    ``group`` scopes the multicast reachability test; None means the default
    (world) group.
    """
    if tp_size not in _SUPPORTED_TP_SIZES:
        return False
    if (hidden_size, latent_size) != (7168, 3584) or dtype != torch.bfloat16:
        return False
    return multicast_backend_available(group)


@dataclass
class _Contract:
    """Dimensions and identities shared by op construction and pooled slots."""

    # The registered group name, not id(group): stable for the process
    # lifetime and the same identity symmetric-memory rendezvous keys on.
    group_name: str
    tp_size: int
    device: torch.device
    hidden_size: int
    latent_size: int
    rms_eps: float
    # Non-None arms the deferred-finalize input mode (the collective kernel
    # inlines the MoE finalize); part of the pool identity because it selects
    # which kernel variants the pooled slot compiles.
    finalize_top_k: int | None = None
    split_collective: bool = False


@dataclass
class _SymmetricPoolSlot:
    """One indivisible collective-workspace and multicast-mailbox bundle."""

    collective: CollectiveKernel
    up_projection: AdaptiveUpProjectionKernel
    split_shared_output: torch.Tensor | None
    scratch_allocator_key: object


class KimiK3LatentTailOp:
    """Multicast tail for one module, statically bound to a rotation-pool slot.

    Construction performs a collective symmetric-memory rendezvous — every
    rank in ``group`` must construct with identical arguments in lockstep
    (model-layer initialization satisfies this).
    """

    _symmetric_pools: dict[
        tuple[str, int, torch.device, int, int, float, int | None, bool, str, int],
        _SymmetricPoolSlot,
    ] = {}

    @classmethod
    def initialize(
        cls,
        *,
        group: dist.ProcessGroup,
        hidden_size: int,
        latent_size: int,
        rms_eps: float,
        device: torch.device,
        layer_index: int,
        model_scope: str,
        scratch_allocator: _ScratchAllocator | None = None,
        finalize_top_k: int | None = None,
        split_collective: bool = False,
    ) -> "KimiK3LatentTailOp":
        """Construct this caller's tail op with a statically bound pool slot.

        Args:
            group: Process group every rank constructs with, in lockstep.
            hidden_size: Model hidden width.
            latent_size: Routed-expert latent width.
            rms_eps: Epsilon of the routed-expert RMS norm.
            device: CUDA device owning the mailbox.
            layer_index: Decoder layer index used to select the rotation slot.
            model_scope: Rank-identical model scope separating pool bundles,
                including base and draft models.
            scratch_allocator: Optional ``(*specs) -> list[Tensor]`` carving
                per-call staging views from a shared block (the runtime's
                workspace pool). The views are re-fetched on every collective
                call per that pool's contract; None keeps private buffers.
            finalize_top_k: When set (rank-uniform), additionally compiles the
                deferred-finalize input mode so :meth:`call_deferred` can
                consume the MoE kernel's ``(gemm2 permuted rows, expert
                weights, expanded->permuted index)`` triple directly, skipping
                the standalone finalize kernel and its ``[M, latent]``
                intermediate. The standard mode stays available.
            split_collective: Precompile independent shared-ReduceScatter and
                routed-AllReduce roles for multistream execution.
        """
        contract = _Contract(
            group_name=group.group_name,
            tp_size=dist.get_world_size(group),
            device=device,
            hidden_size=hidden_size,
            latent_size=latent_size,
            rms_eps=float(rms_eps),
            finalize_top_k=finalize_top_k,
            split_collective=split_collective,
        )
        return cls(
            contract,
            group,
            layer_index=layer_index,
            model_scope=model_scope,
            scratch_allocator=scratch_allocator,
        )

    def __init__(
        self,
        contract: _Contract,
        group: dist.ProcessGroup,
        layer_index: int,
        model_scope: str,
        scratch_allocator: _ScratchAllocator | None = None,
    ) -> None:
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import (
            AdaptiveUpProjectionKernel,
            CollectiveKernel,
            LamportCopyKernel,
        )

        self.contract = contract
        self.rank = dist.get_rank(group)
        slot_index = _tail_pool_slot(layer_index)
        pool_key = (
            contract.group_name,
            contract.tp_size,
            contract.device,
            contract.hidden_size,
            contract.latent_size,
            contract.rms_eps,
            contract.finalize_top_k,
            contract.split_collective,
            model_scope,
            slot_index,
        )

        # Split output lives across two streams, so it cannot borrow storage
        # from the moving workspace pool.
        effective_scratch_allocator = (
            None if contract.split_collective else scratch_allocator
        )
        allocator_key = _allocator_identity(effective_scratch_allocator)
        with torch.accelerator.device_index(contract.device.index):
            slot = type(self)._symmetric_pools.get(pool_key)
            if slot is None:
                # Allocate the rendezvous bundle atomically for this static slot.
                slot = _SymmetricPoolSlot(
                    collective=CollectiveKernel(
                        group=group,
                        rank=self.rank,
                        tp_size=contract.tp_size,
                        latent_dim=contract.latent_size,
                        hidden_dim=contract.hidden_size,
                        max_m=_MAX_NUM_TOKENS,
                        max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
                        rms_eps=contract.rms_eps,
                        fp32_internal=True,
                        scratch_allocator=effective_scratch_allocator,
                        finalize_top_k=contract.finalize_top_k,
                        precompile_split=contract.split_collective,
                    ),
                    up_projection=AdaptiveUpProjectionKernel(
                        group=group,
                        rank=self.rank,
                        tp_size=contract.tp_size,
                        latent_dim=contract.latent_size,
                        hidden_dim=contract.hidden_size,
                        max_m=_MAX_NUM_TOKENS,
                        skinny_max_m=_SKINNY_MAX_NUM_TOKENS,
                        mma_tiler_mn=_MMA_TILER_MN,
                        cluster_shape_mn=_GEMM_CLUSTER_MN,
                        b_prime_stages=_B_PRIME_STAGES,
                    ),
                    split_shared_output=(
                        torch.empty(
                            (_MAX_NUM_TOKENS, contract.hidden_size),
                            dtype=torch.bfloat16,
                            device=contract.device,
                        )
                        if contract.split_collective
                        else None
                    ),
                    scratch_allocator_key=allocator_key,
                )
                type(self)._symmetric_pools[pool_key] = slot
            elif slot.scratch_allocator_key is not allocator_key:
                raise RuntimeError(
                    "KimiK3 latent-tail pool slot already exists with a "
                    "different scratch_allocator"
                )
            self._collective = slot.collective
            self._up_projection = slot.up_projection
            self._split_shared_output = slot.split_shared_output
            self._lamport_copy = LamportCopyKernel(
                hidden_dim=contract.hidden_size,
                max_m=_MAX_NUM_TOKENS,
                ctas=_LAMPORT_COPY_CTAS,
                threads=_LAMPORT_COPY_THREADS,
            )

    @property
    def max_num_tokens(self) -> int:
        return _MAX_NUM_TOKENS

    @property
    def supports_deferred_finalize(self) -> bool:
        """Whether :meth:`call_deferred` may consume the deferred triple."""
        return self.contract.finalize_top_k is not None

    @property
    def supports_split_collective(self) -> bool:
        return self.contract.split_collective

    @property
    def split_collective_min_tokens(self) -> int:
        """First M whose token work spans more than one collective CTA wave."""
        return _COLLECTIVE_TOKEN_CTAS + 1

    def __call__(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        prefix: torch.Tensor | None = None,
        prepared_shared_shard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused tail for one decode step.

        Args:
            routed_partial: This rank's routed-expert partial ``[M, 3584]``
                (contiguous BF16, pre-all-reduce).
            shared_partial: This rank's shared-expert partial ``[M, 7168]``.
            rms_weight: Latent RMSNorm weight ``[3584]``.
            up_weight: Up-projection weight. Either replicated
                ``[7168, 3584]``, of which this rank's ``hidden/tp`` row shard
                is consumed, or that shard already stored on its own
                (``[7168/tp, 3584]``) when the caller keeps the weight column
                parallel.
            prefix: Optional residual stream ``[M, 7168]``; when given the
                lamport gather fuses ``+ prefix`` (same rounding as an eager
                add) and the caller's accumulate disappears.
            prepared_shared_shard: Shared ReduceScatter output produced by
                :meth:`reduce_scatter_shared` on an auxiliary stream.

        Returns:
            ``[M, 7168]`` post-communication hidden (up-projection + shared,
            plus ``prefix`` when provided).
        """
        # Same-slot calls are stream-ordered, preserving Lamport buffer rotation.
        m = routed_partial.shape[0]
        self._assert_capture_compiled(m)
        if prepared_shared_shard is None:
            latent, shared_shard = self._collective(
                routed_partial,
                shared_partial,
                rms_weight,
            )
        else:
            self._validate_prepared_shared_shard(prepared_shared_shard)
            latent, _ = self._collective(
                routed_partial,
                self._collective.shared_output[:m],
                rms_weight,
                include_reduce_scatter=False,
                include_routed=True,
            )
            shared_shard = prepared_shared_shard
        return self._project_and_gather(latent, shared_shard, m, up_weight, prefix)

    def call_deferred(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        *,
        num_tokens: int,
        prefix: torch.Tensor | None = None,
        prepared_shared_shard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused tail consuming the MoE kernel's deferred-finalize triple.

        The collective kernel inlines the finalize (FP32 accumulate over the
        fixed top-k order, semantically equivalent to the standalone finalize
        kernel; bitwise identity is not guaranteed across compilers) into its
        publish phase, so no ``[M, latent]`` routed partial is materialized.

        Args:
            gemm2_output: Permuted expert rows ``[total_padded_rows, >=3584]``
                BF16 from the MoE kernel's ``do_finalize=False`` path; extra
                padded width is sliced off.
            expert_weights: Per-slot scales, ``[M, top_k]`` or flat
                ``[M * top_k]`` (coerced to contiguous BF16).
            expanded_idx_to_permuted_idx: Expanded->permuted row map,
                ``[M, top_k]`` or flat ``[M * top_k]`` (coerced to contiguous
                int32); ``-1`` drops the slot (EP non-local expert / padding).
            shared_partial: This rank's shared-expert partial ``[M, 7168]``.
            rms_weight: Latent RMSNorm weight ``[3584]``.
            up_weight: Up-projection weight, as in :meth:`__call__`.
            num_tokens: Token count M for this step.
            prefix: Optional residual stream ``[M, 7168]``, as in
                :meth:`__call__`.
            prepared_shared_shard: Shared ReduceScatter output produced by
                :meth:`reduce_scatter_shared` on an auxiliary stream.

        Returns:
            ``[M, 7168]`` post-communication hidden, as in :meth:`__call__`.
        """
        if not self.supports_deferred_finalize:
            raise RuntimeError(
                "KimiK3LatentTailOp was initialized without finalize_top_k; "
                "the deferred-finalize mode is unavailable"
            )
        m = int(num_tokens)
        top_k = self.contract.finalize_top_k
        self._assert_capture_compiled(m)
        # The deferred trtllm MoE returns scale/index tensors as either
        # [m, top_k] or flat [m * top_k]; both are memory-identical when
        # contiguous, and the fused kernel indexes the flat form. Reshape is
        # a view and contiguous() a no-op on the K3 SiTU path.
        if expert_weights.dtype != torch.bfloat16:
            # No silent cast: the collective kernel scalar-loads raw BF16
            # bits, so a .to(bfloat16) here would quietly halve the scale
            # precision of a producer emitting fp32 weights (e.g. a routing
            # path with _routing_logits_dtype=float32). Refuse instead.
            raise ValueError(
                "deferred-finalize expert_weights must be BF16 (the trtllm "
                "SiTU path echoes the caller's bf16 topk weights); got "
                f"{expert_weights.dtype}. An fp32-scale producer needs an "
                "fp32 weight-load variant of the latent-tail kernel first."
            )
        if expert_weights.shape != (m * top_k,) or not expert_weights.is_contiguous():
            expert_weights = expert_weights.reshape(m * top_k).contiguous()
        if (
            expanded_idx_to_permuted_idx.dtype != torch.int32
            or expanded_idx_to_permuted_idx.shape != (m * top_k,)
            or not expanded_idx_to_permuted_idx.is_contiguous()
        ):
            expanded_idx_to_permuted_idx = (
                expanded_idx_to_permuted_idx.reshape(m * top_k)
                .to(torch.int32)
                .contiguous()
            )
        if gemm2_output.shape[-1] != self.contract.latent_size:
            gemm2_output = gemm2_output[:, : self.contract.latent_size].contiguous()
        if gemm2_output.shape[0] == 0:
            # EP corner: every slot routed away from this rank (idx all -1).
            # This rank must still publish zeros and join the collective, so
            # substitute a one-row zero placeholder rather than early-return
            # (which would strand peers in the AR poll). The kernel only
            # dereferences rows named by non-negative indices, and a zero row
            # contributes nothing even if a contract-violating index reads it.
            gemm2_output = torch.zeros(
                (1, self.contract.latent_size),
                dtype=torch.bfloat16,
                device=gemm2_output.device,
            )
        split_shared = prepared_shared_shard is not None
        if split_shared:
            self._validate_prepared_shared_shard(prepared_shared_shard)
        latent, shared_shard = self._collective.call_deferred(
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            (self._collective.shared_output[:m] if split_shared else shared_partial),
            rms_weight,
            num_tokens=m,
            include_reduce_scatter=not split_shared,
        )
        if split_shared:
            shared_shard = prepared_shared_shard
        return self._project_and_gather(latent, shared_shard, m, up_weight, prefix)

    def reduce_scatter_shared(
        self,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run the shared ReduceScatter into stable split-mode staging.

        Args:
            shared_partial: This rank's contiguous CUDA BF16 shared-expert
                partial ``[M, hidden_size]``, where
                ``1 <= M <= max_num_tokens``.
            rms_weight: Contiguous CUDA BF16 latent RMSNorm weight
                ``[latent_size]`` required by the collective signature.

        Returns:
            The rank-local BF16 shard view
            ``[max_num_tokens, hidden_size / tp_size]`` with stride
            ``(hidden_size, 1)``. Only the first ``M`` rows contain this
            launch's output. The view aliases the pooled split staging and is
            valid until the next shared ReduceScatter using the same pool
            slot. A consumer on another stream must wait for this launch's
            stream before reading it.
        """
        if not self.supports_split_collective:
            raise RuntimeError("shared-only ReduceScatter is not initialized")
        m = shared_partial.shape[0]
        _, shared_shard = self._collective(
            self._collective.latent_output[:m],
            shared_partial,
            rms_weight,
            include_reduce_scatter=True,
            include_routed=False,
            shared_output_override=self._split_shared_output,
        )
        return shared_shard

    def _validate_prepared_shared_shard(self, shared_shard: torch.Tensor) -> None:
        expected = (
            _MAX_NUM_TOKENS,
            self.contract.hidden_size // self.contract.tp_size,
        )
        if (
            not self.supports_split_collective
            or shared_shard.shape != expected
            or shared_shard.dtype != torch.bfloat16
            or shared_shard.device != self.contract.device
            or shared_shard.stride() != (self.contract.hidden_size, 1)
        ):
            raise ValueError(
                "prepared shared shard must be the split collective's "
                f"BF16 {list(expected)} output view"
            )

    def _assert_capture_compiled(self, m: int) -> None:
        # JIT compilation launches kernels; under capture they would be
        # recorded into the graph. Warmup must have compiled this m already.
        assert not torch.cuda.is_current_stream_capturing() or (
            self._up_projection.is_compiled(m)
        ), f"latent-tail up-projection for M={m} must compile in warmup, not capture"
        self._up_projection.ensure_compiled(m)

    def _project_and_gather(
        self,
        latent: torch.Tensor,
        shared_shard: torch.Tensor,
        m: int,
        up_weight: torch.Tensor,
        prefix: torch.Tensor | None,
    ) -> torch.Tensor:
        local_hidden = self.contract.hidden_size // self.contract.tp_size
        if up_weight.shape[0] == local_hidden:
            local_up_weight = up_weight
        elif up_weight.shape[0] == self.contract.hidden_size:
            local_up_weight = up_weight.narrow(
                0, self.rank * local_hidden, local_hidden
            )
        else:
            raise ValueError(
                f"up_weight must have {self.contract.hidden_size} rows "
                f"(replicated) or {local_hidden} (this rank's shard), got "
                f"{up_weight.shape[0]}"
            )
        mailbox = self._up_projection(latent, local_up_weight, shared_shard)
        return self._lamport_copy(mailbox, m=m, residual=prefix).squeeze(0)


__all__ = ["KimiK3LatentTailOp", "latent_tail_supported"]
