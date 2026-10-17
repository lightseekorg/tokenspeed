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

"""CuTe DSL cluster kernel for routed normalization and reduction."""

from __future__ import annotations

import logging
import threading
from typing import Callable

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

logger = logging.getLogger(__name__)

# Past these batch sizes the per-row cluster overhead loses to the Triton kernels.
CLUSTER_ROUTED_SUM_MAX_TOKENS = 64
_PTR_ALIGN = 16

# Clusters need sm_90+; the dispatcher's cluster_supported trait gates the rest.
_CAPABILITY = CapabilityRequirement(
    vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
)
_CLUSTER_TRAITS = {"cluster_supported": frozenset({True})}

_compile_cache: dict[tuple, object] = {}
# cute.compile is expensive and not thread-safe.
_compile_lock = threading.Lock()


def _cluster_arch() -> bool:
    try:
        p = current_platform()
    except Exception:
        return False
    if not p.is_nvidia:
        return False
    # Clusters need sm_90+; the 16-CTA size is the non-portable maximum there.
    return p.arch_version.major >= 9


_CUTE_AVAILABLE = False
if _cluster_arch():
    try:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_tensor
        from tokenspeed_kernel.thirdparty.cute_dsl.cluster_routed_sum import (
            ROUTED_CTA_THREADS,
            ClusterRoutedSumKernel,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
            to_cute,
        )

        _CUTE_AVAILABLE = True
    except Exception as error:  # noqa: BLE001 - a DSL skew must not take Triton down
        logger.warning(
            "Routed normalization cluster kernel unavailable; using Triton: %s",
            error,
        )


def is_available() -> bool:
    """Whether the cluster kernels can run on this platform.

    Returns:
        True on NVIDIA sm_90+ with the CuTe DSL importable.
    """
    return _CUTE_AVAILABLE


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _compile_once(
    key: tuple, device: torch.device, build: Callable[[], Callable[..., None]]
) -> Callable[..., None]:
    compiled = _compile_cache.get(key)
    if compiled is not None:
        return compiled
    # cute.compile targets the current device, not the operands'.
    with _compile_lock, torch.cuda.device(device):
        compiled = _compile_cache.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the cluster kernel was first seen inside a CUDA graph capture; "
                    "run one eager decode step before capturing"
                )
            compiled = build()
            _compile_cache[key] = compiled
    return compiled


def _fake(dtype: type[cutlass.Numeric], shape: tuple[object, ...]) -> cute.Tensor:
    """A row-major fake tensor for compilation."""
    strides = []
    for i in range(len(shape)):
        stride = 1
        for extent in shape[i + 1 :]:
            stride = stride * extent
        strides.append(stride)
    return make_fake_tensor(dtype, shape, tuple(strides), assumed_align=_PTR_ALIGN)


def _stream(device: torch.device) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def cluster_routed_sum_supported(
    gemm2_out: torch.Tensor,
    permute_map: torch.Tensor,
    num_tokens: int,
    topk: int,
    route_weights: torch.Tensor,
    norm_weight: torch.Tensor,
) -> bool:
    """Whether the cluster routed sum accepts these operands' shapes, dtypes and alignment.

    Arguments follow :func:`tokenspeed_kernel.ops.moe.routed_input_scale_rmsnorm_sum_gathered`.
    """
    if not _CUTE_AVAILABLE:
        return False
    if not 1 <= num_tokens <= CLUSTER_ROUTED_SUM_MAX_TOKENS:
        return False
    if norm_weight.ndim != 1 or not norm_weight.is_contiguous():
        return False
    width = norm_weight.shape[0]
    if not 1 <= topk <= 16 or width % (topk * ROUTED_CTA_THREADS):
        return False
    if not _power_of_two(width):
        return False
    if gemm2_out.ndim != 2 or gemm2_out.stride(1) != 1 or gemm2_out.shape[1] < width:
        return False
    # Every row start must keep the 16-byte alignment the loads assume.
    if gemm2_out.stride(0) % (_PTR_ALIGN // gemm2_out.element_size()):
        return False
    if gemm2_out.dtype != torch.bfloat16 or norm_weight.dtype != torch.bfloat16:
        return False
    if permute_map.dtype != torch.int32 or not permute_map.is_contiguous():
        return False
    if permute_map.numel() != num_tokens * topk:
        return False
    if (
        route_weights.shape != (num_tokens, topk)
        or route_weights.dtype != torch.float32
    ):
        return False
    if not route_weights.is_contiguous():
        return False
    tensors = (gemm2_out, permute_map, route_weights, norm_weight)
    if not gemm2_out.is_cuda or any(t.device != gemm2_out.device for t in tensors):
        return False
    return all(t.data_ptr() % _PTR_ALIGN == 0 for t in tensors)


def _compiled_routed_sum(
    width: int,
    topk: int,
    device: torch.device,
    eps: float,
    weight_offset: float,
    enable_pdl: bool,
) -> Callable[..., None]:
    key = ("routed", device.index, width, topk, eps, weight_offset, enable_pdl)

    def build() -> Callable[..., None]:
        kernel = ClusterRoutedSumKernel(width, topk, eps, weight_offset, enable_pdl)
        # Separate symbols: only the padded width is promised divisible by eight.
        padded = cute.sym_int()
        routes = make_fake_tensor(
            cutlass.BFloat16,
            (cute.sym_int(), padded),
            (padded, 1),
            assumed_align=_PTR_ALIGN,
        )
        permute_map = _fake(cutlass.Int32, (cute.sym_int(),))
        route_weights = _fake(cutlass.Float32, (cute.sym_int(),))
        norm_weight = _fake(cutlass.BFloat16, (width,))
        output = _fake(cutlass.BFloat16, (cute.sym_int(), width))
        return cute.compile(
            kernel,
            routes,
            permute_map,
            route_weights,
            norm_weight,
            output,
            1,
            1,
            _stream(device),
        )

    return _compile_once(key, device, build)


@register_kernel(
    "moe",
    "routed_input_scale_rmsnorm_sum_gathered",
    name="cute_dsl_routed_input_scale_rmsnorm_sum_gathered",
    solution="cute_dsl",
    capability=_CAPABILITY,
    signatures=format_signatures("gemm2_out", "dense", {torch.bfloat16}),
    traits=_CLUSTER_TRAITS,
    priority=Priority.SPECIALIZED,
)
def cute_dsl_routed_input_scale_rmsnorm_sum_gathered(
    gemm2_out: torch.Tensor,
    permute_map: torch.Tensor,
    num_tokens: int,
    topk: int,
    route_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    enable_pdl: bool,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """The routed input-scale RMSNorm and top-k sum on one cluster per token.

    Same contract as :func:`tokenspeed_kernel.ops.moe.routed_input_scale_rmsnorm_sum_gathered`;
    each route's variance is summed in a fixed warp, CTA order.
    """
    if not cluster_routed_sum_supported(
        gemm2_out, permute_map, num_tokens, topk, route_weights, norm_weight
    ):
        raise ValueError("the cluster routed sum does not support these operands")
    if output is not None:
        raise ValueError("the cluster routed sum does not support caller-owned output")
    width = norm_weight.shape[0]
    output = torch.empty(
        (num_tokens, width), dtype=torch.bfloat16, device=gemm2_out.device
    )
    compiled = _compiled_routed_sum(
        width,
        topk,
        gemm2_out.device,
        eps,
        weight_offset,
        enable_pdl,
    )
    with torch.cuda.device(gemm2_out.device):
        compiled(
            to_cute(gemm2_out, _PTR_ALIGN),
            to_cute(permute_map, _PTR_ALIGN),
            to_cute(route_weights.view(-1), _PTR_ALIGN),
            to_cute(norm_weight, _PTR_ALIGN),
            to_cute(output, _PTR_ALIGN),
            num_tokens,
            gemm2_out.stride(0),
            _stream(gemm2_out.device),
        )
    return output


__all__ = [
    "CLUSTER_ROUTED_SUM_MAX_TOKENS",
    "cluster_routed_sum_supported",
    "is_available",
    "cute_dsl_routed_input_scale_rmsnorm_sum_gathered",
]
