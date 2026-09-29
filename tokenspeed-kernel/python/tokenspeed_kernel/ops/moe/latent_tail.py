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

"""K3 stage-2 workspace ownership and the attention-reduction adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

if TYPE_CHECKING:
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import CollectiveKernel

_MULTICAST_MIN_ARCH = 10
K3_SHARED_RS_MAX_TOKENS = 32
_MAX_NUM_TOKENS = 64
_SKINNY_MAX_NUM_TOKENS = 5
_COLLECTIVE_TOKEN_CTAS = 8


def multicast_reachable(group: dist.ProcessGroup) -> bool:
    """Whether NVLS multicast can actually map across ``group``'s ranks.

    ``symm_mem`` importing is not enough: a cross-host group without fabric or
    IMEX still reports multicast support locally and then hangs inside the
    rendezvous instead of letting the caller fall back. The host-span test is
    at group granularity: a node-local subgroup of a multi-host job never
    needs fabric, and probing one would decline a group that works over plain
    NVLink on a machine with no fabric at all.

    Size alone does not establish node-locality, and neither does alignment to
    the group's own width: at eight devices a host, ``[6, 7, 8]`` is contiguous
    and starts on a multiple of three while still living on two hosts. What
    decides it is which host each rank sits on, which the world map records
    beside the fabric verdict. Nothing is divided out of the visible device
    count here: a job running fewer workers than a host has GPUs puts two hosts
    inside one such window, and the group would skip the fabric test entirely.

    Both terms come from the map gathered at distributed initialization, so
    every rank makes the same local decision, and a map never gathered declines
    rather than guessing at placement.
    """
    import torch.distributed as dist
    from tokenspeed_kernel.ops.communication.fabric import (
        group_has_fabric,
        group_host_span,
    )

    if not dist.is_initialized():
        return False
    ranks = dist.get_process_group_ranks(group)
    span = group_host_span(ranks)
    if span is None:
        return False
    if span <= 1:
        return True
    return group_has_fabric(ranks)


def multicast_backend_unavailable_reason(
    group: dist.ProcessGroup,
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
    # Whether to attempt the path. Deliberately not the same number as
    # latent_down's _MULTICAST_VALIDATED_ARCH, which decides whether a failure
    # to rendezvous is a broken machine: a later architecture should get to try.
    if platform.arch_version.major < _MULTICAST_MIN_ARCH:
        return (
            f"compute capability {platform.arch_version.major}, "
            f"below {_MULTICAST_MIN_ARCH}"
        )
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from torch.distributed import _symmetric_memory  # noqa: F401
    except ImportError:
        return "cutlass or symmetric memory is not importable"
    if not multicast_reachable(group):
        return "fabric unreachable"
    return None


def multicast_backend_available(group: dist.ProcessGroup) -> bool:
    """Whether the CuteDSL multicast backend can run here at all.

    Separate from any one op's shapes: capability, the optional imports, and
    fabric reachability. All three must hold before a collective rendezvous.
    """
    return multicast_backend_unavailable_reason(group) is None


def attn_reduce_shape_supported(*, tp_size: int, hidden_size: int) -> bool:
    """Whether the collective's geometry admits an attention reduce this wide.

    Args:
        tp_size: Attention tensor-parallel width.
        hidden_size: Model hidden width. The attention reduce carries no
            latent projection, so this is both the latent and hidden dim.

    Returns:
        ``True`` when :func:`build_attn_reduce_collective` can be built for
        this pair; ``False`` when the cluster geometry rules it out or the
        platform cannot import the collective at all. The constructor raises
        rather than declining, so a caller wanting a capability answer asks
        here first.
    """
    try:
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.allreduce_rmsnorm_reduce_scatter_early_exit import (  # noqa: E501
            validate_shape,
        )

        validate_shape(tp_size=tp_size, latent_dim=hidden_size, hidden_dim=hidden_size)
    except (ImportError, ValueError):
        # The collective needs cuda bindings; a platform without them declines.
        return False
    return True


def build_attn_reduce_collective(
    *,
    group: dist.ProcessGroup,
    rank: int,
    tp_size: int,
    hidden_size: int,
    max_tokens: int,
) -> "CollectiveKernel":
    """Build the collective that serves Kimi-K3's attention reduce.

    The epilogue emits ``all_reduce(partial) + residual`` instead of a
    RMSNorm. The norm weight goes unread in this mode, but ``__call__`` still
    shape-checks it, so callers must keep passing one.

    Args:
        group: Attention tensor-parallel process group. Every rank in it must
            call this, in lockstep: the constructor rendezvouses.
        rank: This rank's index within ``group``.
        tp_size: Size of ``group``.
        hidden_size: Model hidden width; see
            :func:`attn_reduce_shape_supported`, which must accept the pair
            before this is called.
        max_tokens: Widest reduce this instance will serve. The result comes
            back as a view of the collective's own buffer, valid until the
            next call.

    Returns:
        A ``CollectiveKernel`` to be called with
        ``include_reduce_scatter=False, include_routed=True``. Its first
        return is a view of the instance's own latent buffer and stays valid
        only until the next call on this instance -- which, since the runtime
        holds one per process, means any caller's next call.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import CollectiveKernel

    return CollectiveKernel(
        group=group,
        rank=rank,
        tp_size=tp_size,
        latent_dim=hidden_size,
        hidden_dim=hidden_size,
        max_m=max_tokens,
        max_token_ctas=max_tokens,
        rms_eps=1.0,
        fp32_internal=True,
        scratch_allocator=None,
        finalize_top_k=None,
        precompile_split=True,
        residual_from_shared=True,
    )


class KimiK3LatentTailOp:
    """Stage-2 shared RS, sharded up-projection, and multicast gather.

    One instance owns the collective buffers and projection mailbox for
    sequential MoE layers. Layer weights are supplied at execution time.
    """

    def __init__(
        self, *, group: dist.ProcessGroup, hidden_size: int, latent_size: int
    ) -> None:
        """Allocate and prepare the existing kernels before graph capture.

        Args:
            group: TP4, TP8, or TP16 multicast group, constructed collectively.
            hidden_size: Full output width, currently 7168.
            latent_size: Replicated routed width, currently 3584.
        """
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail import (
            AdaptiveUpProjectionKernel,
            CollectiveKernel,
            LamportCopyKernel,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
            NEG_ZERO_F32_BITS,
        )

        tp_size = dist.get_world_size(group)
        if tp_size not in (4, 8, 16) or (hidden_size, latent_size) != (7168, 3584):
            raise ValueError(
                "K3 stage-2 multicast requires TP4/TP8/TP16, H7168, and L3584"
            )
        rank = dist.get_rank(group)
        device = torch.device("cuda", torch.cuda.current_device())
        self._collective = CollectiveKernel(
            group=group,
            rank=rank,
            tp_size=tp_size,
            latent_dim=latent_size,
            hidden_dim=hidden_size,
            max_m=_MAX_NUM_TOKENS,
            max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
            rms_eps=1.0,
            fp32_internal=True,
            residual_from_shared=False,
            scratch_allocator=None,
            finalize_top_k=None,
            precompile_split=True,
        )
        # The RS-only specialization keeps an unused gamma in its launch signature.
        self._unused_gamma = torch.empty(
            latent_size, dtype=torch.bfloat16, device=device
        )
        self._up_projection = AdaptiveUpProjectionKernel(
            group=group,
            rank=rank,
            tp_size=tp_size,
            latent_dim=latent_size,
            hidden_dim=hidden_size,
            max_m=_MAX_NUM_TOKENS,
            skinny_max_m=_SKINNY_MAX_NUM_TOKENS,
            mma_tiler_mn=(64, 32),
            cluster_shape_mn=(1, 8),
            b_prime_stages=2,
        )
        for m in range(1, _SKINNY_MAX_NUM_TOKENS + 1):
            self._up_projection.compile_skinny(m)
        self._up_projection.compile_dynamic()
        self._lamport_copy = LamportCopyKernel(
            hidden_dim=hidden_size,
            max_m=_MAX_NUM_TOKENS,
            ctas=32,
            threads=224,
            sentinel=NEG_ZERO_F32_BITS,
        )
        # Preserve the completion dependency across separate captured graphs.
        self._gather_complete = torch.cuda.Event(external=True)
        self._gather_complete.record()

    def reduce_scatter_shared(self, shared_partial: torch.Tensor) -> torch.Tensor:
        """Run only shared RS on the stream selected for the shared experts.

        Args:
            shared_partial: Rank-local BF16 [M,7168], with M in [1,32].

        Returns:
            Padded BF16 [64,7168/TP] shard with row stride 7168. Only the first M
            rows are live; consume them before the next shared RS call.
        """
        m = shared_partial.shape[0]
        if not 1 <= m <= K3_SHARED_RS_MAX_TOKENS:
            raise ValueError("K3 shared RS requires M in [1,32]")
        # Wait for the gather's sentinel cleanup before reusing the one mailbox.
        self._gather_complete.wait()
        _, shared_shard = self._collective(
            self._collective.latent_output[:m],
            shared_partial,
            self._unused_gamma,
            include_reduce_scatter=True,
            include_routed=False,
        )
        return shared_shard

    def project_and_gather(
        self,
        latent: torch.Tensor,
        weight: torch.Tensor,
        shared_shard: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Project the local columns and gather after joining shared RS.

        Args:
            latent: Replicated normalized BF16 [M,3584], with M in [1,32].
            weight: This layer's BF16 [7168/TP,3584] up-projection weight shard.
            shared_shard: Padded rank-local output from reduce_scatter_shared.
            residual: Replicated BF16 [M,7168] attention residual.

        Returns:
            Fresh BF16 [M,7168] output on every rank, including the residual.
        """
        m = latent.shape[0]
        if not 1 <= m <= K3_SHARED_RS_MAX_TOKENS:
            raise ValueError("K3 multicast up-projection requires M in [1,32]")
        mailbox = self._up_projection(latent, weight, shared_shard)
        output = self._lamport_copy(mailbox, m=m, residual=residual).squeeze(0)
        self._gather_complete.record()
        return output


__all__ = [
    "K3_SHARED_RS_MAX_TOKENS",
    "KimiK3LatentTailOp",
    "multicast_reachable",
    "multicast_backend_unavailable_reason",
    "multicast_backend_available",
    "attn_reduce_shape_supported",
    "build_attn_reduce_collective",
]
