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

"""FlashInfer MNNVL CuTe DSL adapter for K3 deferred MoE finalize.

This is the only module in the integration that knows FlashInfer's protocol,
tuning, and launch APIs.  Imports stay lazy so importing TokenSpeed-kernel on a
machine without FlashInfer or CUDA remains safe.  The public operator wrapper
under :mod:`tokenspeed_kernel.ops.communication` owns the stable API and
validation contract; this adapter owns third-party protocol construction and
execution state.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.distributed as dist

_FLASHINFER_VERSION = "0.6.18"
_K3_LATENT_SIZE = 3584
_K3_TOP_K = 16
_K3_TP_SIZE = 8


def _is_supported_blackwell_capability(capability: tuple[int, int]) -> bool:
    """Return whether ``capability`` is in the validated SM100--SM103 family."""

    return (10, 0) <= capability <= (10, 3)


class MNNVLCuteDSLDeferredFinalizeBackend:
    """Pointer-stable storage and launch adapter for one MNNVL protocol."""

    def __init__(
        self,
        *,
        protocol: Any,
        kernels: tuple[Any, ...],
        output: torch.Tensor,
        empty_routed_row: torch.Tensor,
    ) -> None:
        self._protocol = protocol
        self._kernels = kernels
        self._output = output
        self._empty_routed_row = empty_routed_row

    @property
    def device(self) -> torch.device:
        """Return the device holding this backend's persistent buffers."""

        return self._output.device

    @property
    def num_routes(self) -> int:
        """Return the number of compiled token-range routes."""

        return len(self._kernels)

    def launch(
        self,
        route_index: int,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx: torch.Tensor,
        gamma: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Launch one precompiled protocol route into persistent output.

        Args:
            route_index: Index of the precompiled tuning route.
            gemm2_output: Local routed rows shaped ``[R, 3584]``.
            expert_weights: Per-route BF16 weights.
            expanded_idx: Per-route int32 row map; ``-1`` means no local row.
            gamma: BF16 RMSNorm weight shaped ``[3584]``.
            num_tokens: Logical token count for this launch.

        Returns:
            A view of the persistent BF16 norm-output buffer.
        """

        if not 0 <= route_index < len(self._kernels):
            raise RuntimeError("no compiled MNNVL tuning covers this token count")
        routed = self._empty_routed_row if gemm2_output.shape[0] == 0 else gemm2_output
        output = self._output[:num_tokens]
        norm, _ = self._kernels[route_index](
            routed,
            expert_weights,
            expanded_idx,
            None,
            None,
            gamma,
            num_tokens,
            state=self._protocol.state,
            norm_output=output,
            residual_output=None,
        )
        return norm


def bt_support_error(
    *,
    group: dist.ProcessGroup,
    tp_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    candidate_min_tokens: int,
    candidate_max_tokens: int,
) -> str | None:
    """Return why the FlashInfer BT backend is unavailable, or ``None``."""

    try:
        if not dist.is_initialized():
            return "torch.distributed is not initialized"
        if tp_size != _K3_TP_SIZE or dist.get_world_size(group) != tp_size:
            return "the deferred BT path requires a TP8 process group"
        if not 0 <= dist.get_rank(group) < tp_size:
            return "the process-group rank is invalid"
        if hidden_size != _K3_LATENT_SIZE or top_k != _K3_TOP_K:
            return "the deferred BT path only supports K3 H3584/top-k 16"
        if dtype != torch.bfloat16:
            return "the deferred BT path only supports BF16"
        if not 1 <= candidate_min_tokens <= candidate_max_tokens:
            return "the candidate token range is invalid"
        if not torch.cuda.is_available():
            return "CUDA is unavailable"

        device = torch.device("cuda", torch.cuda.current_device())
        capability = torch.cuda.get_device_capability(device)
        if not _is_supported_blackwell_capability(capability):
            return "the deferred BT path requires data-center Blackwell"

        import flashinfer
        import torch.distributed._symmetric_memory as symm_mem
        from flashinfer.comm.mnnvl import is_multicast_supported
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
            BTCollectiveTuning,
            BTFinalizeTuning,
        )
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt.protocol import BTProtocol

        version = str(flashinfer.__version__).split("+", 1)[0]
        if version != _FLASHINFER_VERSION:
            return (
                f"FlashInfer {_FLASHINFER_VERSION} is required; found "
                f"{flashinfer.__version__}"
            )
        finalize_parameters = inspect.signature(BTFinalizeTuning).parameters
        if not {
            "elements_per_thread",
            "threads",
            "prefetch_group",
            "load_shared_expert_before_pdl",
            "collective",
        }.issubset(finalize_parameters):
            return "FlashInfer's BT finalize tuning API is incompatible"
        collective_parameters = inspect.signature(BTCollectiveTuning).parameters
        if not {"reduction_threads", "rms_threads", "enable_pdl"}.issubset(
            collective_parameters
        ):
            return "FlashInfer's BT collective tuning API is incompatible"
        protocol_parameters = inspect.signature(BTProtocol).parameters
        if not {
            "hidden_size",
            "top_k",
            "tp_size",
            "rank",
            "capacity_m",
            "rms_epsilon",
            "finalize_tunings",
            "all_reduce_tunings",
            "group",
        }.issubset(protocol_parameters):
            return "FlashInfer's BT protocol API is incompatible"
        if symm_mem.get_backend(device) is None:
            return "PyTorch symmetric memory is unavailable"
        if not is_multicast_supported(device.index):
            return "NVLink multicast is unavailable"
    except Exception as error:  # The public probe must fail closed.
        return f"support probe raised {type(error).__name__}"
    return None


def ht_support_error(
    *,
    group: dist.ProcessGroup,
    tp_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    candidate_min_tokens: int,
    candidate_max_tokens: int,
) -> str | None:
    """Return why the native-H3584 HT backend is unavailable, or ``None``."""

    error = bt_support_error(
        group=group,
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype=dtype,
        candidate_min_tokens=candidate_min_tokens,
        candidate_max_tokens=candidate_max_tokens,
    )
    if error is not None:
        return error
    try:
        from flashinfer.comm.mnnvl_cutedsl.kernel_ht.protocol import HTFinalizeTuning
        from tokenspeed_kernel.thirdparty.cute_dsl.mnnvl_k3_ht import (
            K3H3584HTProtocol,
        )

        tuning_parameters = inspect.signature(HTFinalizeTuning).parameters
        if not {
            "persistent_ctas",
            "consumer_threads",
            "vectors_per_thread",
            "stages",
            "reduction_warps",
            "reduction_cta_groups",
            "rms_token_groups",
            "rms_pipeline_stages",
            "rms_shard_major",
            "enable_pdl",
        }.issubset(tuning_parameters):
            return "FlashInfer's HT finalize tuning API is incompatible"
        protocol_parameters = inspect.signature(K3H3584HTProtocol).parameters
        if not {
            "hidden_size",
            "top_k",
            "tp_size",
            "rank",
            "capacity_m",
            "rms_epsilon",
            "finalize_tunings",
            "all_reduce_tunings",
            "group",
        }.issubset(protocol_parameters):
            return "TokenSpeed's native-H3584 HT protocol API is incompatible"
    except Exception as error:  # The public probe must fail closed.
        return f"native-H3584 HT support probe raised {type(error).__name__}"
    return None


def _allocate_backend(
    *,
    protocol: Any,
    kernels: tuple[Any, ...],
    candidate_max_tokens: int,
    hidden_size: int,
    group: dist.ProcessGroup,
) -> MNNVLCuteDSLDeferredFinalizeBackend:
    device = torch.device("cuda", torch.cuda.current_device())
    output = torch.empty(
        (candidate_max_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    # A rank can own no GEMM2 rows while global M remains positive.  The
    # protocol still needs a valid base pointer; -1 map entries make this
    # persistent zero row semantically inert.
    empty_routed_row = torch.zeros(
        (1, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    return MNNVLCuteDSLDeferredFinalizeBackend(
        protocol=protocol,
        kernels=kernels,
        output=output,
        empty_routed_row=empty_routed_row,
    )


def build_bt_backend(
    *,
    group: dist.ProcessGroup,
    hidden_size: int,
    top_k: int,
    rms_eps: float,
    candidate_max_tokens: int,
    tuning_routes: tuple[Any, ...],
) -> MNNVLCuteDSLDeferredFinalizeBackend:
    """Construct FlashInfer's BT protocol and persistent launch adapter."""

    from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
        BTCollectiveTuning,
        BTFinalizeTuning,
    )
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt.protocol import BTProtocol

    compiled_tunings = []
    for route in tuning_routes:
        collective = BTCollectiveTuning(
            reduction_threads=route.reduction_threads,
            rms_threads=route.rms_threads,
            enable_pdl=route.enable_pdl,
        )
        compiled_tunings.append(
            BTFinalizeTuning(
                elements_per_thread=route.elements_per_thread,
                threads=route.threads,
                prefetch_group=route.prefetch_group,
                load_shared_expert_before_pdl=False,
                collective=collective,
            )
        )
    tunings = tuple(compiled_tunings)
    tp_size = dist.get_world_size(group)
    protocol = BTProtocol(
        hidden_size=hidden_size,
        top_k=top_k,
        tp_size=tp_size,
        rank=dist.get_rank(group),
        capacity_m=candidate_max_tokens,
        rms_epsilon=float(rms_eps),
        routed_scaling_factor=1.0,
        weight_bias=0.0,
        include_shared_expert=False,
        add_residual=False,
        write_residual_output=False,
        finalize_tunings=tunings,
        all_reduce_tunings=(),
        group=group,
    )
    kernels = tuple(protocol.finalize_kernels[tuning] for tuning in tunings)
    return _allocate_backend(
        protocol=protocol,
        kernels=kernels,
        candidate_max_tokens=candidate_max_tokens,
        hidden_size=hidden_size,
        group=group,
    )


def build_ht_backend(
    *,
    group: dist.ProcessGroup,
    hidden_size: int,
    top_k: int,
    rms_eps: float,
    candidate_max_tokens: int,
    tuning_routes: tuple[Any, ...],
) -> MNNVLCuteDSLDeferredFinalizeBackend:
    """Construct the native-H3584 HT protocol and persistent adapter."""

    from flashinfer.comm.mnnvl_cutedsl.kernel_ht.protocol import HTFinalizeTuning
    from tokenspeed_kernel.thirdparty.cute_dsl.mnnvl_k3_ht import K3H3584HTProtocol

    compiled_tunings = tuple(
        HTFinalizeTuning(
            persistent_ctas=route.persistent_ctas,
            consumer_threads=route.consumer_threads,
            vectors_per_thread=route.vectors_per_thread,
            stages=route.stages,
            reduction_warps=route.reduction_warps,
            reduction_cta_groups=route.reduction_cta_groups,
            rms_token_groups=route.rms_token_groups,
            rms_pipeline_stages=route.rms_pipeline_stages,
            rms_shard_major=route.rms_shard_major,
            enable_pdl=route.enable_pdl,
        )
        for route in tuning_routes
    )
    tp_size = dist.get_world_size(group)
    protocol = K3H3584HTProtocol(
        hidden_size=hidden_size,
        top_k=top_k,
        tp_size=tp_size,
        rank=dist.get_rank(group),
        capacity_m=candidate_max_tokens,
        rms_epsilon=float(rms_eps),
        routed_scaling_factor=1.0,
        weight_bias=0.0,
        include_shared_expert=False,
        add_residual=False,
        write_residual_output=False,
        finalize_tunings=compiled_tunings,
        all_reduce_tunings=(),
        group=group,
    )
    kernels = tuple(protocol.finalize_kernels[tuning] for tuning in compiled_tunings)
    return _allocate_backend(
        protocol=protocol,
        kernels=kernels,
        candidate_max_tokens=candidate_max_tokens,
        hidden_size=hidden_size,
        group=group,
    )


__all__ = [
    "MNNVLCuteDSLDeferredFinalizeBackend",
    "bt_support_error",
    "build_bt_backend",
    "build_ht_backend",
    "ht_support_error",
]
