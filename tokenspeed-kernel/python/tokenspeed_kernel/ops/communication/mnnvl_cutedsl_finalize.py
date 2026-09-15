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

"""Kimi K3 deferred MoE finalize plus MNNVL all-reduce and RMSNorm.

This module defines TokenSpeed-kernel's stable public contract and registered
operator for the balanced-tree (BT) and native-H3584 hierarchical-tree (HT)
implementations.  FlashInfer imports, protocol construction, and protocol
execution remain behind the third-party adapter boundary.

The wrapper deliberately exposes a caller-selected candidate token range and
tuning routes.  It raises outside that range, allowing runtime dispatch to keep
the established materialize/finalize/all-reduce path as a safe fallback until
each range has passed numerical and performance qualification.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.mnnvl_cutedsl_finalize import (
    MNNVLCuteDSLDeferredFinalizeBackend,
    bt_support_error,
    build_bt_backend,
    build_ht_backend,
    ht_support_error,
)

_K3_LATENT_SIZE = 3584
_K3_TOP_K = 16
_K3_TP_SIZE = 8
_BF16_VECTOR_SIZE = 8
_WARP_SIZE = 32
_DEFERRED_FINALIZE_SIGNATURE = format_signature(
    gemm2_output=dense_tensor_format(torch.bfloat16),
    expert_weights=dense_tensor_format(torch.bfloat16),
    expanded_idx=dense_tensor_format(torch.int32),
    gamma=dense_tensor_format(torch.bfloat16),
)


@dataclass(frozen=True, slots=True)
class MNNVLCuteDSLBTFinalizeTuning:
    """One explicitly bounded BT finalize tuning.

    Args:
        max_tokens: Inclusive upper M bound for this tuning.  Bounds must be
            strictly increasing and the last one must equal the workspace's
            ``candidate_max_tokens``.
        elements_per_thread: Number of adjacent BF16 values handled by one
            finalize thread; FlashInfer supports 1, 2, 4, and 8.
        threads: Threads per finalize CTA.
        prefetch_group: Number of top-k rows prefetched together.
        reduction_threads: Threads per BT owner-reduce CTA.
        rms_threads: Threads per BT RMSNorm CTA.
        enable_pdl: Whether all three stages use programmatic dependent launch.
    """

    max_tokens: int
    elements_per_thread: int
    threads: int
    prefetch_group: int
    reduction_threads: int
    rms_threads: int
    enable_pdl: bool

    def as_signature(self) -> tuple[int, int, int, int, int, int, bool]:
        """Return a process-independent signature for rank agreement."""

        return (
            self.max_tokens,
            self.elements_per_thread,
            self.threads,
            self.prefetch_group,
            self.reduction_threads,
            self.rms_threads,
            self.enable_pdl,
        )


@dataclass(frozen=True, slots=True)
class MNNVLCuteDSLHTFinalizeTuning:
    """One explicitly bounded native-H3584 HT finalize tuning.

    HT keeps a persistent set of CTAs resident while it finalizes routed rows,
    reduces the rank-local BF16 contributions through MNNVL, and materializes
    RMSNorm.  The TokenSpeed specialization tail-predicates the final TP shard,
    which is required because K3's 3584-wide latent is 56 BF16x8 packs per
    rank rather than a multiple of a warp.

    Args:
        max_tokens: Inclusive upper M bound for this tuning route.
        persistent_ctas: Resident CTA count, or ``None`` to use all legal SMs.
        consumer_threads: Threads in each persistent consumer CTA.
        vectors_per_thread: BF16x8 vectors consumed per thread iteration.
        stages: Producer/consumer shared-memory pipeline stages.
        reduction_warps: Warps in each MNNVL reduction shard.
        reduction_cta_groups: Reduction CTA groups, or ``None`` for the
            protocol-derived default.
        rms_token_groups: Tokens normalized concurrently in one CTA.
        rms_pipeline_stages: RMSNorm shared-memory pipeline stages.
        rms_shard_major: Whether RMSNorm assigns work shard-major.
        enable_pdl: Whether to use programmatic dependent launch.
    """

    max_tokens: int
    persistent_ctas: int | None
    consumer_threads: int
    vectors_per_thread: int
    stages: int
    reduction_warps: int
    reduction_cta_groups: int | None
    rms_token_groups: int
    rms_pipeline_stages: int
    rms_shard_major: bool
    enable_pdl: bool

    def as_signature(self) -> tuple[Any, ...]:
        """Return a process-independent signature for rank agreement."""

        return (
            self.max_tokens,
            self.persistent_ctas,
            self.consumer_threads,
            self.vectors_per_thread,
            self.stages,
            self.reduction_warps,
            self.reduction_cta_groups,
            self.rms_token_groups,
            self.rms_pipeline_stages,
            self.rms_shard_major,
            self.enable_pdl,
        )


def _support_error(
    *,
    group: dist.ProcessGroup,
    tp_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    candidate_min_tokens: int,
    candidate_max_tokens: int,
) -> str | None:
    return bt_support_error(
        group=group,
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype=dtype,
        candidate_min_tokens=candidate_min_tokens,
        candidate_max_tokens=candidate_max_tokens,
    )


def mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported(
    *,
    group: dist.ProcessGroup,
    tp_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    candidate_min_tokens: int,
    candidate_max_tokens: int,
) -> bool:
    """Return local support for K3's FlashInfer BT deferred-finalize path.

    This function does not allocate or enter a collective.  Its result must be
    combined across ``group`` before construction; :meth:`initialize` performs
    that vote.  Any import, version, capability, or API-inspection failure
    returns ``False`` rather than partially enabling the path.  The version and
    signature checks protect normal wheel installs; a deployment that overlays
    FlashInfer source on an installed wheel must additionally pin and record
    that source revision because package metadata cannot identify an overlay.

    Args:
        group: Intended TP process group.
        tp_size: Intended TP degree; K3 currently requires eight.
        hidden_size: Routed latent width; K3 requires 3584.
        top_k: Routes per token; K3 requires sixteen.
        dtype: Deferred GEMM2/output dtype; only BF16 is supported.
        candidate_min_tokens: First qualified M for candidate dispatch.
        candidate_max_tokens: Last qualified M and BT workspace capacity.

    Returns:
        Whether this rank has the exact validated FlashInfer API and hardware.
    """

    return (
        _support_error(
            group=group,
            tp_size=tp_size,
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=dtype,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
        )
        is None
    )


def _ht_support_error(
    *,
    group: dist.ProcessGroup,
    tp_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    candidate_min_tokens: int,
    candidate_max_tokens: int,
) -> str | None:
    return ht_support_error(
        group=group,
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype=dtype,
        candidate_min_tokens=candidate_min_tokens,
        candidate_max_tokens=candidate_max_tokens,
    )


def mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported(
    *,
    group: dist.ProcessGroup,
    tp_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    candidate_min_tokens: int,
    candidate_max_tokens: int,
) -> bool:
    """Return local support for K3's native-H3584 HT deferred path."""

    return (
        _ht_support_error(
            group=group,
            tp_size=tp_size,
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=dtype,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
        )
        is None
    )


def _validate_tuning_routes(
    *,
    routes: tuple[MNNVLCuteDSLBTFinalizeTuning, ...],
    candidate_min_tokens: int,
    candidate_max_tokens: int,
    hidden_size: int,
    top_k: int,
) -> None:
    if not routes:
        raise ValueError("at least one BT finalize tuning route is required")
    previous = candidate_min_tokens - 1
    for route in routes:
        if route.max_tokens <= previous or route.max_tokens > candidate_max_tokens:
            raise ValueError(
                "BT tuning max_tokens must be strictly increasing within the "
                "candidate range"
            )
        if route.elements_per_thread not in (1, 2, 4, 8):
            raise ValueError("elements_per_thread must be 1, 2, 4, or 8")
        if hidden_size % route.elements_per_thread:
            raise ValueError("elements_per_thread must divide hidden_size")
        for name, threads in (
            ("threads", route.threads),
            ("reduction_threads", route.reduction_threads),
            ("rms_threads", route.rms_threads),
        ):
            if not 32 <= threads <= 1024 or threads % 32:
                raise ValueError(f"{name} must be a multiple of 32 in [32, 1024]")
        if not 1 <= route.prefetch_group <= top_k:
            raise ValueError("prefetch_group must be in [1, top_k]")
        if not isinstance(route.enable_pdl, bool):
            raise ValueError("enable_pdl must be bool")
        previous = route.max_tokens
    if routes[-1].max_tokens != candidate_max_tokens:
        raise ValueError("the final BT tuning route must end at candidate_max_tokens")


def _configuration_signature(
    *,
    hidden_size: int,
    top_k: int,
    rms_eps: float,
    candidate_min_tokens: int,
    candidate_max_tokens: int,
    tuning_routes: tuple[Any, ...],
) -> tuple[Any, ...]:
    return (
        hidden_size,
        top_k,
        float(rms_eps).hex(),
        candidate_min_tokens,
        candidate_max_tokens,
        tuple(route.as_signature() for route in tuning_routes),
    )


def _require_collective_agreement(
    group: dist.ProcessGroup,
    signature: tuple[Any, ...],
) -> None:
    signatures: list[tuple[Any, ...] | None] = [
        None for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather_object(signatures, signature, group=group)
    if any(peer != signature for peer in signatures):
        raise RuntimeError(
            "MNNVL CuTe DSL deferred-finalize configuration differs across ranks"
        )


@register_kernel(
    "communication",
    "deferred_finalize_allreduce_rmsnorm",
    name="flashinfer_cutedsl_mnnvl_deferred_finalize_allreduce_rmsnorm",
    features={"mnnvl", "deferred_finalize", "rmsnorm"},
    solution="flashinfer_cutedsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=frozenset({_DEFERRED_FINALIZE_SIGNATURE}),
    traits={
        "tp_size": frozenset({_K3_TP_SIZE}),
        "hidden_size": frozenset({_K3_LATENT_SIZE}),
        "top_k": frozenset({_K3_TOP_K}),
        "protocol": frozenset({"bt", "ht"}),
        "capturing": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
    tags={"blackwell", "cuda_graph", "determinism", "throughput"},
)
def mnnvl_cutedsl_deferred_finalize_allreduce_rmsnorm(
    backend: MNNVLCuteDSLDeferredFinalizeBackend,
    route_index: int,
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx: torch.Tensor,
    gamma: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Run a prebuilt MNNVL deferred-finalize backend.

    Args:
        backend: Pointer-stable third-party protocol adapter built collectively.
        route_index: Index of the tuning selected for ``num_tokens``.
        gemm2_output: Local BF16 routed rows shaped ``[R, 3584]``.
        expert_weights: BF16 route weights shaped ``[M, 16]`` or flat.
        expanded_idx: Int32 routed-row map shaped ``[M, 16]`` or flat.
        gamma: BF16 RMSNorm weight shaped ``[3584]``.
        num_tokens: Logical M for this invocation.

    Returns:
        BF16 normalized output shaped ``[M, 3584]``, aliasing persistent
        backend storage until the next invocation.
    """

    return backend.launch(
        route_index,
        gemm2_output,
        expert_weights,
        expanded_idx,
        gamma,
        num_tokens,
    )


def _select_deferred_finalize_kernel(protocol: str) -> SelectedKernel:
    return select_kernel(
        "communication",
        "deferred_finalize_allreduce_rmsnorm",
        _DEFERRED_FINALIZE_SIGNATURE,
        features=frozenset({"mnnvl", "deferred_finalize", "rmsnorm"}),
        traits={
            "tp_size": _K3_TP_SIZE,
            "hidden_size": _K3_LATENT_SIZE,
            "top_k": _K3_TOP_K,
            "protocol": protocol,
            "capturing": True,
        },
        solution="flashinfer_cutedsl",
    )


def _select_deferred_finalize_kernel_collectively(
    group: dist.ProcessGroup,
    protocol: str,
) -> SelectedKernel:
    kernel: SelectedKernel | None = None
    local_error: str | None = None
    try:
        kernel = _select_deferred_finalize_kernel(protocol)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    local_result = (kernel.name if kernel is not None else None, local_error)
    results: list[tuple[str | None, str | None] | None] = [
        None for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather_object(results, local_result, group=group)
    errors = [
        f"rank {rank}: {result[1]}"
        for rank, result in enumerate(results)
        if result is not None and result[1] is not None
    ]
    names = {result[0] for result in results if result is not None}
    if errors or len(names) != 1 or None in names or kernel is None:
        details = ", ".join(errors) if errors else f"rank selections: {results}"
        raise RuntimeError(
            "MNNVL deferred-finalize kernel selection vote failed: " + details
        )
    return kernel


class MNNVLCuteDSLFinalizeAllReduceRMSNorm:
    """Pointer-stable K3 deferred finalize, TP8 BT all-reduce, and RMSNorm.

    Construction collectively compiles every caller-supplied tuning and owns
    one symmetric BT mailbox bundle.  The instance also owns persistent output
    and empty-GEMM2 placeholder tensors; no tensor allocation or dtype/shape
    conversion occurs in :meth:`__call__`, including during CUDA Graph capture.

    Calls on one instance must not overlap.  The returned tensor aliases the
    instance's output buffer and remains valid until the next call.  A runtime
    should therefore share an instance only when each consumer is ordered
    before the next invocation on the same stream.  Concurrent streams or
    overlapping graph replays require distinct instances.
    """

    def __init__(
        self,
        *,
        backend: MNNVLCuteDSLDeferredFinalizeBackend,
        kernel: SelectedKernel,
        route_upper_bounds: tuple[int, ...],
        candidate_min_tokens: int,
        candidate_max_tokens: int,
    ) -> None:
        self._backend = backend
        self._kernel = kernel
        self._route_upper_bounds = route_upper_bounds
        self._device = backend.device
        self.candidate_min_tokens = candidate_min_tokens
        self.candidate_max_tokens = candidate_max_tokens

    @classmethod
    def initialize(
        cls,
        *,
        group: dist.ProcessGroup,
        hidden_size: int,
        top_k: int,
        rms_eps: float,
        candidate_min_tokens: int,
        candidate_max_tokens: int,
        tuning_routes: tuple[MNNVLCuteDSLBTFinalizeTuning, ...],
    ) -> "MNNVLCuteDSLFinalizeAllReduceRMSNorm":
        """Collectively allocate and compile a K3 BT deferred workspace.

        Args:
            group: TP8 process group.  Every rank must call in lockstep.
            hidden_size: K3 routed latent width, exactly 3584.
            top_k: K3 routes per token, exactly sixteen.
            rms_eps: Epsilon compiled into BT's RMSNorm materialization stage.
            candidate_min_tokens: First M qualified for candidate dispatch.
            candidate_max_tokens: Last qualified M and workspace capacity.
            tuning_routes: Increasing inclusive M bounds and complete BT
                tuning choices.  The final bound must equal
                ``candidate_max_tokens``.

        Returns:
            A non-overlapping, CUDA-graph pointer-stable BT workspace.
        """

        signature = _configuration_signature(
            hidden_size=hidden_size,
            top_k=top_k,
            rms_eps=rms_eps,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
            tuning_routes=tuning_routes,
        )
        _require_collective_agreement(group, signature)
        if hidden_size != _K3_LATENT_SIZE or top_k != _K3_TOP_K:
            raise ValueError("this workspace only supports K3 H3584/top-k 16")
        if not math.isfinite(rms_eps) or rms_eps < 0:
            raise ValueError("rms_eps must be finite and non-negative")
        if not 1 <= candidate_min_tokens <= candidate_max_tokens:
            raise ValueError("candidate token range must be positive and ordered")
        _validate_tuning_routes(
            routes=tuning_routes,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
        )

        tp_size = dist.get_world_size(group)
        local_error = _support_error(
            group=group,
            tp_size=tp_size,
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=torch.bfloat16,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
        )
        support_errors: list[str | None] = [None for _ in range(tp_size)]
        dist.all_gather_object(support_errors, local_error, group=group)
        if any(error is not None for error in support_errors):
            details = ", ".join(
                f"rank {rank}: {error}"
                for rank, error in enumerate(support_errors)
                if error is not None
            )
            raise RuntimeError(
                "MNNVL CuTe DSL deferred-finalize support vote failed: " + details
            )

        kernel = _select_deferred_finalize_kernel_collectively(group, "bt")
        backend = build_bt_backend(
            group=group,
            hidden_size=hidden_size,
            top_k=top_k,
            rms_eps=rms_eps,
            candidate_max_tokens=candidate_max_tokens,
            tuning_routes=tuning_routes,
        )
        return cls(
            backend=backend,
            kernel=kernel,
            route_upper_bounds=tuple(route.max_tokens for route in tuning_routes),
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
        )

    def supports_num_tokens(self, num_tokens: int) -> bool:
        """Return whether M is inside the explicitly qualified range."""

        return self.candidate_min_tokens <= num_tokens <= self.candidate_max_tokens

    def _token_count(self, tensor: torch.Tensor, name: str) -> int:
        if tensor.ndim == 2:
            if tensor.shape[1] != _K3_TOP_K:
                raise ValueError(f"{name} must have shape [M, {_K3_TOP_K}] or flat")
            return tensor.shape[0]
        if tensor.ndim == 1 and tensor.numel() % _K3_TOP_K == 0:
            return tensor.numel() // _K3_TOP_K
        raise ValueError(f"{name} must have shape [M, {_K3_TOP_K}] or flat")

    def _validate_common_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        dtype: torch.dtype,
        alignment: int,
    ) -> None:
        if tensor.device != self._device:
            raise ValueError(f"{name} must be on {self._device}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % alignment:
            raise ValueError(f"{name} must be {alignment}-byte aligned")

    def __call__(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Finalize routed rows, TP8-reduce them, and apply latent RMSNorm.

        Args:
            gemm2_output: Contiguous CUDA BF16 permuted expert rows
                ``[R, 3584]``.  ``R`` may be zero only under the producer's
                contract that every corresponding ``expanded_idx`` is ``-1``;
                a persistent zero row supplies the otherwise-empty pointer.
            expert_weights: Contiguous CUDA BF16 per-slot weights as
                ``[M, 16]`` or flat ``[M * 16]``.  These weights must already
                include K3's routed scaling factor.
            expanded_idx: Contiguous CUDA int32 expanded-to-permuted map as
                ``[M, 16]`` or flat ``[M * 16]``.  A value of ``-1`` drops a
                non-local or padded slot; every non-negative value must name a
                row in ``gemm2_output``.  Bounds remain a producer contract:
                inspecting device values here would synchronize eager serving
                and cannot be made part of the CUDA Graph host path.
            gamma: Contiguous CUDA BF16 latent RMSNorm weight ``[3584]``.

        Returns:
            BF16 ``[M, 3584]`` normalized routed output.  The result aliases a
            persistent workspace buffer and is overwritten by the next call.
        """

        self._validate_common_tensor(gemm2_output, "gemm2_output", torch.bfloat16, 16)
        if gemm2_output.ndim != 2 or gemm2_output.shape[1] != _K3_LATENT_SIZE:
            raise ValueError("gemm2_output must have shape [R, 3584]")
        self._validate_common_tensor(
            expert_weights, "expert_weights", torch.bfloat16, 2
        )
        self._validate_common_tensor(expanded_idx, "expanded_idx", torch.int32, 4)
        m = self._token_count(expert_weights, "expert_weights")
        if self._token_count(expanded_idx, "expanded_idx") != m:
            raise ValueError("expert_weights and expanded_idx disagree on M")
        if not self.supports_num_tokens(m):
            raise ValueError(
                f"M={m} is outside candidate range "
                f"[{self.candidate_min_tokens}, {self.candidate_max_tokens}]"
            )
        self._validate_common_tensor(gamma, "gamma", torch.bfloat16, 16)
        if gamma.shape != (_K3_LATENT_SIZE,):
            raise ValueError("gamma must have shape [3584]")

        route_index = bisect_left(self._route_upper_bounds, m)
        if route_index == self._backend.num_routes:
            raise RuntimeError("no compiled MNNVL tuning covers this token count")
        return self._kernel(
            self._backend,
            route_index,
            gemm2_output,
            expert_weights,
            expanded_idx,
            gamma,
            m,
        )


def _validate_ht_tuning_routes(
    *,
    routes: tuple[MNNVLCuteDSLHTFinalizeTuning, ...],
    candidate_min_tokens: int,
    candidate_max_tokens: int,
) -> None:
    if not routes:
        raise ValueError("at least one HT finalize tuning route is required")
    previous = candidate_min_tokens - 1
    for route in routes:
        if route.max_tokens <= previous or route.max_tokens > candidate_max_tokens:
            raise ValueError(
                "HT tuning max_tokens must be strictly increasing within the "
                "candidate range"
            )
        if route.persistent_ctas is not None and (
            route.persistent_ctas <= 0 or route.persistent_ctas % _K3_TP_SIZE
        ):
            raise ValueError("persistent_ctas must be positive and divisible by TP8")
        if not 32 <= route.consumer_threads <= 1024 or route.consumer_threads % 32:
            raise ValueError("consumer_threads must be a multiple of 32 in [32, 1024]")
        if route.vectors_per_thread not in (1, 2, 4, 8):
            raise ValueError("vectors_per_thread must be 1, 2, 4, or 8")
        if route.stages < 2:
            raise ValueError("HT stages must be at least 2")
        consumer_warps = route.consumer_threads // _WARP_SIZE
        if route.reduction_warps not in (1, 2, 4, 8):
            raise ValueError("reduction_warps must be 1, 2, 4, or 8")
        if route.reduction_warps > consumer_warps:
            raise ValueError("reduction_warps must fit in the consumer CTA")
        if route.reduction_cta_groups is not None and route.reduction_cta_groups <= 0:
            raise ValueError("reduction_cta_groups must be positive when specified")
        if route.rms_token_groups not in (1, 2, 4):
            raise ValueError("rms_token_groups must be 1, 2, or 4")
        if (
            route.consumer_threads % route.rms_token_groups
            or (route.consumer_threads // route.rms_token_groups) % _WARP_SIZE
        ):
            raise ValueError(
                "rms_token_groups must split consumer_threads into whole warps"
            )
        if route.rms_pipeline_stages not in (1, 2, 3):
            raise ValueError("rms_pipeline_stages must be 1, 2, or 3")
        block_threads = (
            route.consumer_threads + (2 + route.reduction_warps) * _WARP_SIZE
        )
        if block_threads > 1024:
            raise ValueError("HT warp roles exceed the CUDA block limit")
        shard_elements = (
            route.consumer_threads * _BF16_VECTOR_SIZE * route.vectors_per_thread
        )
        if _K3_LATENT_SIZE % shard_elements:
            raise ValueError(
                "K3 latent width must divide evenly across HT consumer shards"
            )
        packs_per_token = _K3_LATENT_SIZE // _BF16_VECTOR_SIZE
        if packs_per_token % route.consumer_threads:
            raise ValueError("K3 token vectors must divide evenly across HT consumers")
        rms_threads_per_token = route.consumer_threads // route.rms_token_groups
        if packs_per_token % rms_threads_per_token:
            raise ValueError(
                "K3 token vectors must divide evenly across RMSNorm threads"
            )
        if (
            route.rms_pipeline_stages > 1
            and route.rms_token_groups * route.rms_pipeline_stages * _K3_LATENT_SIZE
            > shard_elements * route.stages
        ):
            raise ValueError("HT finalize storage cannot hold the RMSNorm pipeline")
        if route.rms_shard_major:
            rms_warps_per_token = rms_threads_per_token // _WARP_SIZE
            if _K3_TP_SIZE < rms_warps_per_token or _K3_TP_SIZE % rms_warps_per_token:
                raise ValueError(
                    "shard-major RMSNorm requires an integer number of "
                    "reduction shards per RMSNorm warp"
                )
            reduction_shards_per_rms_warp = _K3_TP_SIZE // rms_warps_per_token
            rms_vectors_per_thread = packs_per_token // rms_threads_per_token
            packs_per_reduction_shard = packs_per_token // _K3_TP_SIZE
            if (
                rms_vectors_per_thread * _WARP_SIZE
                != packs_per_reduction_shard * reduction_shards_per_rms_warp
            ):
                raise ValueError(
                    "shard-major RMSNorm coverage must match reduction shards"
                )
        if not isinstance(route.rms_shard_major, bool) or not isinstance(
            route.enable_pdl, bool
        ):
            raise ValueError("rms_shard_major and enable_pdl must be bool")
        previous = route.max_tokens
    if routes[-1].max_tokens != candidate_max_tokens:
        raise ValueError("the final HT tuning route must end at candidate_max_tokens")


class MNNVLCuteDSLHTFinalizeAllReduceRMSNorm(MNNVLCuteDSLFinalizeAllReduceRMSNorm):
    """Native-H3584 HT variant of deferred finalize + TP8 AR + RMSNorm."""

    @classmethod
    def initialize(
        cls,
        *,
        group: dist.ProcessGroup,
        hidden_size: int,
        top_k: int,
        rms_eps: float,
        candidate_min_tokens: int,
        candidate_max_tokens: int,
        tuning_routes: tuple[MNNVLCuteDSLHTFinalizeTuning, ...],
    ) -> "MNNVLCuteDSLHTFinalizeAllReduceRMSNorm":
        """Collectively allocate and compile native K3 HT tuning routes."""

        signature = (
            "native-h3584-ht",
            *_configuration_signature(
                hidden_size=hidden_size,
                top_k=top_k,
                rms_eps=rms_eps,
                candidate_min_tokens=candidate_min_tokens,
                candidate_max_tokens=candidate_max_tokens,
                tuning_routes=tuning_routes,
            ),
        )
        _require_collective_agreement(group, signature)
        if hidden_size != _K3_LATENT_SIZE or top_k != _K3_TOP_K:
            raise ValueError("this workspace only supports K3 H3584/top-k 16")
        if not math.isfinite(rms_eps) or rms_eps < 0:
            raise ValueError("rms_eps must be finite and non-negative")
        if not 1 <= candidate_min_tokens <= candidate_max_tokens:
            raise ValueError("candidate token range must be positive and ordered")
        _validate_ht_tuning_routes(
            routes=tuning_routes,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
        )

        tp_size = dist.get_world_size(group)
        local_error = _ht_support_error(
            group=group,
            tp_size=tp_size,
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=torch.bfloat16,
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
        )
        support_errors: list[str | None] = [None for _ in range(tp_size)]
        dist.all_gather_object(support_errors, local_error, group=group)
        if any(error is not None for error in support_errors):
            details = ", ".join(
                f"rank {rank}: {error}"
                for rank, error in enumerate(support_errors)
                if error is not None
            )
            raise RuntimeError(
                "native-H3584 HT deferred-finalize support vote failed: " + details
            )

        kernel = _select_deferred_finalize_kernel_collectively(group, "ht")
        backend = build_ht_backend(
            group=group,
            hidden_size=hidden_size,
            top_k=top_k,
            rms_eps=rms_eps,
            candidate_max_tokens=candidate_max_tokens,
            tuning_routes=tuning_routes,
        )
        return cls(
            backend=backend,
            kernel=kernel,
            route_upper_bounds=tuple(route.max_tokens for route in tuning_routes),
            candidate_min_tokens=candidate_min_tokens,
            candidate_max_tokens=candidate_max_tokens,
        )


__all__ = [
    "MNNVLCuteDSLBTFinalizeTuning",
    "MNNVLCuteDSLFinalizeAllReduceRMSNorm",
    "MNNVLCuteDSLHTFinalizeAllReduceRMSNorm",
    "MNNVLCuteDSLHTFinalizeTuning",
    "mnnvl_cutedsl_deferred_finalize_allreduce_rmsnorm",
    "mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported",
    "mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported",
]
