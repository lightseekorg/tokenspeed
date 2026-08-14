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
come from stock ``torch.distributed._symmetric_memory``; the small per-call
staging can come from the runtime's shared workspace pool via
``scratch_allocator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MAX_NUM_TOKENS = 16
_SKINNY_MAX_NUM_TOKENS = 5
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 8
_LAMPORT_COPY_CTAS = 32
_LAMPORT_COPY_THREADS = 224
_SUPPORTED_TP_SIZES = (8, 16)


def _multicast_reachable() -> bool:
    """Whether NVLS multicast can actually map across this job's ranks.

    ``symm_mem`` importing is not enough: a cross-host group without fabric or
    IMEX still reports multicast support locally and then hangs inside the
    rendezvous instead of letting the caller fall back.
    """
    import torch.distributed as dist
    from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported

    if not dist.is_initialized():
        return False
    if dist.get_world_size() <= torch.cuda.device_count():
        return True
    return fabric_allocation_supported(torch.cuda.current_device())


def latent_tail_supported(
    *,
    tp_size: int,
    hidden_size: int,
    latent_size: int,
    dtype: torch.dtype,
) -> bool:
    """Cheap, non-collective eligibility probe (no rendezvous)."""
    if tp_size not in _SUPPORTED_TP_SIZES:
        return False
    if (hidden_size, latent_size) != (7168, 3584) or dtype != torch.bfloat16:
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from torch.distributed import _symmetric_memory  # noqa: F401
    except ImportError:
        return False
    return _multicast_reachable()


@dataclass(frozen=True)
class _Contract:
    group_id: int
    tp_size: int
    device: torch.device
    hidden_size: int
    latent_size: int
    rms_eps: float
    # Mailboxes are owner-private because cleanup may overlap same-stream successors.
    # Use module prefixes: draft models can repeat base-model layer indices.
    owner: str


class KimiK3LatentTailOp:
    """Multicast tail for one calling module; one instance (and mailbox) each.

    Construction performs a collective symmetric-memory rendezvous — every
    rank in ``group`` must construct with identical arguments in lockstep
    (model-layer initialization satisfies this).
    """

    _instances: dict[_Contract, "KimiK3LatentTailOp"] = {}

    @classmethod
    def initialize(
        cls,
        *,
        group: dist.ProcessGroup,
        hidden_size: int,
        latent_size: int,
        rms_eps: float,
        device: torch.device,
        owner: str,
        scratch_allocator=None,
    ) -> "KimiK3LatentTailOp":
        """Return this caller's tail op, constructing it on first use.

        Args:
            group: Process group every rank constructs with, in lockstep.
            hidden_size: Model hidden width.
            latent_size: Routed-expert latent width.
            rms_eps: Epsilon of the routed-expert RMS norm.
            device: CUDA device owning the mailbox.
            owner: Globally unique name for the calling module (its weight
                prefix); part of the key so each caller gets its own mailbox.
                Must be rank-identical -- it takes part in the rendezvous.
            scratch_allocator: Optional ``(*specs) -> list[Tensor]`` carving
                per-call staging views from a shared block (the runtime's
                workspace pool). The views are re-fetched on every collective
                call per that pool's contract; None keeps private buffers.
        """
        contract = _Contract(
            group_id=id(group),
            tp_size=dist.get_world_size(group),
            device=device,
            hidden_size=hidden_size,
            latent_size=latent_size,
            rms_eps=float(rms_eps),
            owner=owner,
        )

        def _alloc_identity(alloc):
            # Bound method identity is unstable; compare the owning pool object.
            return getattr(alloc, "__self__", alloc)

        op = cls._instances.get(contract)
        if op is None:
            op = cls(contract, group, scratch_allocator=scratch_allocator)
            op._scratch_allocator_key = _alloc_identity(scratch_allocator)
            cls._instances[contract] = op
        elif getattr(op, "_scratch_allocator_key", None) is not _alloc_identity(
            scratch_allocator
        ):
            # Cached ops must not retain an allocator from a replaced workspace pool.
            raise RuntimeError(
                "KimiK3LatentTailOp for this contract already exists with a "
                "different scratch_allocator; reset the instance cache when "
                "rebuilding engines in one process"
            )
        return op

    def __init__(
        self,
        contract: _Contract,
        group: dist.ProcessGroup,
        scratch_allocator=None,
    ) -> None:
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import (
            AdaptiveUpProjectionKernel,
            CollectiveKernel,
            LamportCopyKernel,
        )

        self.contract = contract
        self.rank = dist.get_rank(group)
        with torch.accelerator.device_index(contract.device.index):
            self._collective = CollectiveKernel(
                group=group,
                rank=self.rank,
                tp_size=contract.tp_size,
                latent_dim=contract.latent_size,
                hidden_dim=contract.hidden_size,
                max_m=_MAX_NUM_TOKENS,
                max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
                rms_eps=contract.rms_eps,
                fp32_internal=True,
                scratch_allocator=scratch_allocator,
            )
            self._up_projection = AdaptiveUpProjectionKernel(
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
            )
            self._lamport_copy = LamportCopyKernel(
                hidden_dim=contract.hidden_size,
                max_m=_MAX_NUM_TOKENS,
                ctas=_LAMPORT_COPY_CTAS,
                threads=_LAMPORT_COPY_THREADS,
            )

    @property
    def max_num_tokens(self) -> int:
        return _MAX_NUM_TOKENS

    def __call__(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        prefix: torch.Tensor | None = None,
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

        Returns:
            ``[M, 7168]`` post-communication hidden (up-projection + shared,
            plus ``prefix`` when provided).
        """
        # Owner-private mailboxes are reused only after every layer's peer-blocking gather.
        m = routed_partial.shape[0]
        self._up_projection.ensure_compiled(m)
        latent, shared_shard = self._collective(
            routed_partial,
            shared_partial,
            rms_weight,
        )
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
