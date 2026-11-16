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

"""All-reduce with RMSNorm and optional expert finalization, or plain.

The plain all-reduce runs the same FlashInfer protocols without the norm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.allreduce_fusion import (
    MNNVLAllReduceFusionBackend,
    allreduce_fusion_support_error,
)

if TYPE_CHECKING:
    from tokenspeed_kernel.thirdparty.flashinfer.plain_all_reduce import (
        MNNVLPlainAllReduceBackend,
    )


class AllReduceFusionPattern(Enum):
    """Whether the input is a local contribution or deferred expert rows."""

    ALLREDUCE_RMSNORM = "allreduce_rmsnorm"
    MOE_FINALIZE_ALLREDUCE_RMSNORM = "moe_finalize_allreduce_rmsnorm"


_SIGNATURE = format_signature(
    input=dense_tensor_format(torch.bfloat16),
    gamma=dense_tensor_format(torch.bfloat16),
)
_PLAIN_SIGNATURE = format_signature(input=dense_tensor_format(torch.bfloat16))


@dataclass
class AllReduceFusionWorkspace:
    """Persistent output and protocol state for sequential collective calls.

    The output remains valid until the next call using this workspace. All
    ranks must use identical token counts and patterns; concurrent streams or
    model instances require independent workspaces.
    """

    backend: MNNVLAllReduceFusionBackend
    kernel: SelectedKernel
    hidden_size: int
    top_k: int
    max_num_tokens: int

    def supports_num_tokens(self, num_tokens: int) -> bool:
        """Return whether the preallocated workspace covers this positive M."""
        return type(num_tokens) is int and 1 <= num_tokens <= self.max_num_tokens


def allreduce_fusion_supported(
    *,
    group: dist.ProcessGroup,
    hidden_size: int,
    top_k: int,
    max_num_tokens: int,
    dtype: torch.dtype,
) -> bool:
    """Probe K3 TP4/TP8/TP16 BF16 support without allocation or collectives.

    Args:
        group: Initialized process group shared by the operation's ranks.
        hidden_size: Width of each rank's routed contribution, currently 3584.
        top_k: Number of routed experts, currently 16.
        max_num_tokens: Required positive capacity from the serving token limit.
        dtype: Input and output dtype; the current implementation supports BF16.

    Returns:
        Whether this rank has the required hardware and dependency interfaces.
    """
    return (
        allreduce_fusion_support_error(group, hidden_size, top_k, max_num_tokens, dtype)
        is None
    )


def create_allreduce_fusion_workspace(
    *,
    group: dist.ProcessGroup,
    hidden_size: int,
    top_k: int,
    max_num_tokens: int,
    rms_eps: float,
) -> AllReduceFusionWorkspace:
    """Collectively prepare FlashInfer LL/BT/HT before graph capture.

    Args:
        group: TP4, TP8, or TP16 group; all ranks must supply identical configuration
            and enter in the same order.
        hidden_size: Routed latent width, currently 3584.
        top_k: Number of expert contributions per token, currently 16.
        max_num_tokens: Positive serving capacity, independent of graph buckets.
        rms_eps: Finite nonnegative RMSNorm epsilon baked into the kernels.

    Returns:
        One workspace supporting finalized and deferred inputs in eager and graphs.
    """
    error = allreduce_fusion_support_error(
        group, hidden_size, top_k, max_num_tokens, torch.bfloat16
    )
    if not math.isfinite(rms_eps) or rms_eps < 0:
        error = "RMSNorm epsilon must be finite and nonnegative"
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("allreduce fusion must be initialized before graph capture")
    if error is not None:
        raise RuntimeError(error)
    kernel = select_kernel(
        "communication",
        "allreduce_fusion",
        _SIGNATURE,
        traits={"hidden_size": hidden_size, "top_k": top_k, "tp_size": group.size()},
        solution="flashinfer_cutedsl",
        override=None,
    )
    backend = MNNVLAllReduceFusionBackend(
        group, hidden_size, top_k, max_num_tokens, rms_eps
    )
    return AllReduceFusionWorkspace(backend, kernel, hidden_size, top_k, max_num_tokens)


def allreduce_fusion(
    input: torch.Tensor,
    workspace: AllReduceFusionWorkspace,
    *,
    pattern: AllReduceFusionPattern,
    rms_gamma: torch.Tensor,
    num_tokens: int,
    expert_weights: torch.Tensor | None,
    expanded_idx_to_permuted_idx: torch.Tensor | None,
) -> torch.Tensor:
    """Produce the replicated normalized routed latent through one selected backend.

    Args:
        input: Contiguous CUDA BF16 [M,H] local contribution, or [rows,H]
            permuted expert output when finalization is requested.
        workspace: Preallocated AllReduceFusionWorkspace, used sequentially.
        pattern: Explicit AllReduceFusionPattern selecting the input contract.
        rms_gamma: Contiguous CUDA BF16 [H] normalization weight.
        num_tokens: Rank-uniform positive logical/padded token count M.
        expert_weights: BF16 [M,K] or flat [M*K] for deferred inputs; None otherwise.
        expanded_idx_to_permuted_idx: Int32 [M,K] or flat [M*K] row map;
            -1 omits a contribution. None for finalized inputs.

    Returns:
        Persistent BF16 [M,H] view, overwritten by the next workspace call.
    """
    if not isinstance(pattern, AllReduceFusionPattern):
        raise ValueError("an explicit AllReduceFusionPattern is required")
    if not workspace.supports_num_tokens(num_tokens):
        raise ValueError(
            "token count is outside the prepared allreduce fusion capacity"
        )
    h, k = workspace.hidden_size, workspace.top_k
    if input.ndim != 2:
        raise ValueError("allreduce fusion input must have rank two")
    for tensor, shape, dtype in (
        (input, (input.shape[0], h), torch.bfloat16),
        (rms_gamma, (h,), torch.bfloat16),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != workspace.backend.device
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
        ):
            raise ValueError(
                "allreduce fusion requires aligned contiguous BF16 inputs on its workspace device"
            )
    finalize = pattern is AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
    if finalize:
        for tensor, dtype in (
            (expert_weights, torch.bfloat16),
            (expanded_idx_to_permuted_idx, torch.int32),
        ):
            if (
                tensor is None
                or tuple(tensor.shape) not in ((num_tokens, k), (num_tokens * k,))
                or tensor.dtype != dtype
                or tensor.device != input.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "deferred inputs require contiguous BF16 weights and int32 indices for every token and expert"
                )
        expert_weights = expert_weights.view(num_tokens, k)
        expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.view(num_tokens, k)
    elif (
        expert_weights is not None
        or expanded_idx_to_permuted_idx is not None
        or input.shape != (num_tokens, h)
    ):
        raise ValueError(
            "finalized inputs must be [M,H] with no expert weights or indices"
        )
    return workspace.kernel.impl(
        input,
        workspace,
        pattern,
        rms_gamma,
        num_tokens,
        expert_weights,
        expanded_idx_to_permuted_idx,
    )


@register_kernel(
    "communication",
    "allreduce_fusion",
    name="flashinfer_allreduce_fusion",
    solution="flashinfer_cutedsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=frozenset({_SIGNATURE}),
    traits={
        "hidden_size": frozenset({3584}),
        "top_k": frozenset({16}),
        "tp_size": frozenset({4, 8, 16}),
    },
    priority=Priority.SPECIALIZED,
)
def flashinfer_allreduce_fusion(
    input, workspace, pattern, rms_gamma, num_tokens, expert_weights, expanded_idx
):
    """Dispatch a validated first-stage call to FlashInfer LL/BT/HT."""
    return workspace.backend.run(
        input,
        rms_gamma,
        num_tokens,
        pattern is AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM,
        expert_weights,
        expanded_idx,
    )


@dataclass
class AllReduceWorkspace:
    """Persistent protocol state of the plain all-reduce of one group.

    Calls are sequential on one stream; every rank issues the same calls in
    the same order.
    """

    backend: "MNNVLPlainAllReduceBackend"
    kernel: SelectedKernel

    def serves(self, num_tokens: int, hidden_size: int, dtype: torch.dtype) -> bool:
        """Whether a ``[num_tokens, hidden_size]`` reduction takes this workspace.

        Below a payload threshold the caller's Lamport kernels are faster.
        """
        return self.backend.serves(num_tokens, hidden_size, dtype)


def all_reduce_supported(
    *,
    group: dist.ProcessGroup,
    hidden_size: int,
    max_num_tokens: int,
    dtype: torch.dtype,
) -> bool:
    """Probe this rank for the plain all-reduce without allocation or collectives.

    Args:
        group: Initialized process group of the reduction.
        hidden_size: Row width, currently 7168.
        max_num_tokens: Positive row capacity.
        dtype: Payload dtype; BF16 is supported.

    Returns:
        Whether this rank has the hardware, group size and FlashInfer presets.
        Callers vote on it across the group before building the workspace.
    """
    # Imported here: a forward reload of the plain module re-executes nothing else.
    from tokenspeed_kernel.thirdparty.flashinfer.plain_all_reduce import (
        plain_all_reduce_support_error,
    )

    return (
        plain_all_reduce_support_error(group, hidden_size, max_num_tokens, dtype)
        is None
    )


def create_all_reduce_workspace(
    *,
    group: dist.ProcessGroup,
    hidden_size: int,
    max_num_tokens: int,
) -> AllReduceWorkspace:
    """Collectively build the plain all-reduce before graph capture.

    Args:
        group: TP8 group; every rank calls with identical arguments, in order.
        hidden_size: Row width, currently 7168.
        max_num_tokens: Positive row capacity.

    Returns:
        A workspace serving eager calls and graph replays.
    """
    from tokenspeed_kernel.thirdparty.flashinfer.plain_all_reduce import (
        MNNVLPlainAllReduceBackend,
        plain_all_reduce_support_error,
    )

    error = plain_all_reduce_support_error(
        group, hidden_size, max_num_tokens, torch.bfloat16
    )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("the plain all-reduce must be built before graph capture")
    if error is not None:
        raise RuntimeError(error)
    kernel = select_kernel(
        "communication",
        "all_reduce",
        _PLAIN_SIGNATURE,
        traits={"hidden_size": hidden_size, "tp_size": group.size()},
        solution="flashinfer_cutedsl",
        override=None,
    )
    backend = MNNVLPlainAllReduceBackend(group, hidden_size, max_num_tokens)
    return AllReduceWorkspace(backend, kernel)


def all_reduce(
    input: torch.Tensor, workspace: AllReduceWorkspace, out: torch.Tensor
) -> torch.Tensor:
    """Sum ``input`` over the workspace's group into ``out``.

    Args:
        input: Contiguous 16-byte aligned CUDA BF16 [M, H] with
            ``workspace.serves(M, H, torch.bfloat16)``.
        workspace: From ``create_all_reduce_workspace``.
        out: Contiguous 16-byte aligned CUDA BF16 [M, H] destination.

    Returns:
        ``out``: on every rank the ranks' rows summed in rank order in fp32
        and rounded once to BF16.
    """
    if (
        input.ndim != 2
        or not workspace.serves(input.shape[0], input.shape[1], input.dtype)
        or out.shape != input.shape
        or out.dtype != input.dtype
        or not input.is_contiguous()
        or not out.is_contiguous()
        or input.data_ptr() % 16
        or out.data_ptr() % 16
    ):
        raise ValueError("the plain all-reduce does not serve these tensors")
    return workspace.kernel.impl(input, workspace, out)


@register_kernel(
    "communication",
    "all_reduce",
    name="flashinfer_all_reduce",
    solution="flashinfer_cutedsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
    ),
    signatures=frozenset({_PLAIN_SIGNATURE}),
    traits={"hidden_size": frozenset({7168}), "tp_size": frozenset({8})},
    priority=Priority.SPECIALIZED,
)
def flashinfer_all_reduce(input, workspace, out):
    """Dispatch a validated plain all-reduce to FlashInfer BT."""
    workspace.backend.run(input, out)
    return out
