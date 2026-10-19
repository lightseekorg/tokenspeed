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

"""CuTe DSL cluster kernel for a norm-preserving residual boundary."""

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
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
)

logger = logging.getLogger(__name__)

# Past these batch sizes the per-row cluster overhead loses to the Triton kernels.
CLUSTER_BOUNDARY_MAX_ROWS = 64
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
        from tokenspeed_kernel.thirdparty.cute_dsl.cluster_boundary import (
            CLUSTER_CTAS,
            CTA_THREADS,
            ClusterBoundaryKernel,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
            to_cute,
        )

        _CUTE_AVAILABLE = True
    except Exception as error:  # noqa: BLE001 - a DSL skew must not take Triton down
        logger.warning(
            "Gated residual cluster kernel unavailable; using Triton: %s",
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


def mailbox_boundary_supported(rows: int, width: int, ranks: int) -> bool:
    """Whether :func:`tokenspeed_kernel.ops.residual.post_norm_residual_pre_norm` can gather a mailbox branch.

    Args:
        rows: Rows of the batch, the mailbox's first rows.
        width: Row width.
        ranks: Partials per mailbox row the boundary would reduce; 1 to gather.

    Returns:
        True when the cluster kernel is available here and serves this row
        count with whole 32-bit words per thread.
    """
    if not _CUTE_AVAILABLE or not 1 <= rows <= CLUSTER_BOUNDARY_MAX_ROWS:
        return False
    if ranks < 1:
        return False
    return _power_of_two(width) and width % (CLUSTER_CTAS * CTA_THREADS * 2) == 0


def cluster_boundary_supported(
    stream: torch.Tensor,
    branch: torch.Tensor,
    post_weight: torch.Tensor | None,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_weight: torch.Tensor | None,
    branch_sentinel: int | None,
    branch_ranks: int,
) -> bool:
    """Whether the cluster boundary accepts these operands' shapes, dtypes and alignment.

    Arguments follow :func:`tokenspeed_kernel.ops.residual.post_norm_residual_pre_norm`.
    """
    if not _CUTE_AVAILABLE:
        return False
    if stream.ndim != 2 or branch.ndim != 2:
        return False
    rows, width = stream.shape
    if branch_sentinel is None:
        if branch.shape != stream.shape or branch_ranks != 1:
            return False
    elif branch_ranks < 1 or branch.shape[1] != width * branch_ranks:
        return False
    elif branch.shape[0] < rows:
        return False
    # A mailbox needs whole 32-bit words per thread.
    quantum = CLUSTER_CTAS * CTA_THREADS * (1 if branch_sentinel is None else 2)
    if not 1 <= rows <= CLUSTER_BOUNDARY_MAX_ROWS:
        return False
    if width % quantum or not _power_of_two(width):
        return False
    if stream.dtype != torch.float32 or branch.dtype != torch.bfloat16:
        return False
    vectors = [alpha, beta] + [w for w in (post_weight, pre_weight) if w is not None]
    for vector in vectors:
        if vector.shape != (width,) or not vector.is_contiguous():
            return False
        if vector.device != stream.device or vector.data_ptr() % _PTR_ALIGN:
            return False
    for weight in (post_weight, pre_weight):
        if weight is not None and weight.dtype != torch.bfloat16:
            return False
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        return False
    if not stream.is_cuda or branch.device != stream.device:
        return False
    if not stream.is_contiguous() or not branch.is_contiguous():
        return False
    return stream.data_ptr() % _PTR_ALIGN == 0 and branch.data_ptr() % _PTR_ALIGN == 0


def _compiled(
    width: int,
    device: torch.device,
    post_has_weight: bool,
    has_pre: bool,
    post_eps: float,
    post_weight_offset: float,
    pre_eps: float,
    pre_weight_offset: float,
    enable_pdl: bool,
    mailbox_sentinel: int | None,
    mailbox_ranks: int,
) -> Callable[..., None]:
    key = (
        device.index,
        width,
        post_has_weight,
        has_pre,
        post_eps,
        post_weight_offset,
        pre_eps,
        pre_weight_offset,
        enable_pdl,
        mailbox_sentinel,
        mailbox_ranks,
    )

    def build() -> Callable[..., None]:
        kernel = ClusterBoundaryKernel(
            width,
            post_has_weight,
            has_pre,
            post_eps,
            post_weight_offset,
            pre_eps,
            pre_weight_offset,
            enable_pdl,
            mailbox_sentinel,
            mailbox_ranks,
        )
        rows = cute.sym_int()
        f32_rows = _fake(cutlass.Float32, (rows, width))
        bf16_rows = _fake(cutlass.BFloat16, (rows, width))
        # A mailbox has its own capacity, unrelated to the batch, and one
        # partial per rank in each row.
        branch_rows = (
            bf16_rows
            if mailbox_sentinel is None
            else _fake(cutlass.BFloat16, (cute.sym_int(), width * mailbox_ranks))
        )
        f32_vec = _fake(cutlass.Float32, (width,))
        bf16_vec = _fake(cutlass.BFloat16, (width,))
        return cute.compile(
            kernel,
            f32_rows,
            branch_rows,
            bf16_vec,
            f32_vec,
            f32_vec,
            bf16_vec,
            f32_rows,
            f32_rows,
            bf16_rows,
            1,
            _stream(device),
        )

    return _compile_once(key, device, build)


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


@register_kernel(
    "residual",
    "post_norm_residual_pre_norm",
    name="cute_dsl_post_norm_residual_pre_norm",
    solution="cute_dsl",
    capability=_CAPABILITY,
    signatures=frozenset(
        {
            format_signature(
                stream=dense_tensor_format(torch.float32),
                branch=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits=_CLUSTER_TRAITS,
    priority=Priority.SPECIALIZED,
)
def cute_dsl_post_norm_residual_pre_norm(
    stream: torch.Tensor,
    branch: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float,
    post_weight_offset: float,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_weight: torch.Tensor | None,
    pre_eps: float,
    pre_weight_offset: float,
    enable_pdl: bool,
    branch_sentinel: int | None,
    branch_ranks: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """The fused layer boundary on one thread-block cluster per row.

    Same contract as :func:`tokenspeed_kernel.ops.residual.post_norm_residual_pre_norm`; the row
    variances are summed in a fixed warp, CTA, cluster order.
    """
    if not cluster_boundary_supported(
        stream,
        branch,
        post_weight,
        alpha,
        beta,
        pre_weight,
        branch_sentinel,
        branch_ranks,
    ):
        raise ValueError("the cluster boundary does not support these operands")
    rows, width = stream.shape
    has_pre = pre_weight is not None
    out_stream = torch.empty_like(stream)
    pre_fp32 = torch.empty_like(stream) if has_pre else None
    pre_bf16 = (
        torch.empty((rows, width), dtype=torch.bfloat16, device=stream.device)
        if has_pre
        else None
    )
    # An absent weight is never read, so an existing BF16 row stands in, as in Triton.
    post_has_weight = post_weight is not None
    if post_weight is None:
        post_weight = branch[0, :width]
    if pre_weight is None:
        pre_weight = branch[0, :width]
    compiled = _compiled(
        width,
        stream.device,
        post_has_weight,
        has_pre,
        post_eps,
        post_weight_offset,
        pre_eps,
        pre_weight_offset,
        enable_pdl,
        branch_sentinel,
        branch_ranks,
    )
    # The launch goes to the current context, so bind the operands' device.
    with torch.cuda.device(stream.device):
        compiled(
            to_cute(stream, _PTR_ALIGN),
            to_cute(branch, _PTR_ALIGN),
            to_cute(post_weight, _PTR_ALIGN),
            to_cute(alpha, _PTR_ALIGN),
            to_cute(beta, _PTR_ALIGN),
            to_cute(pre_weight, _PTR_ALIGN),
            to_cute(out_stream, _PTR_ALIGN),
            to_cute(pre_fp32 if has_pre else out_stream, _PTR_ALIGN),
            to_cute(pre_bf16 if has_pre else branch[:, :width], _PTR_ALIGN),
            rows,
            _stream(stream.device),
        )
    return out_stream, pre_fp32, pre_bf16


__all__ = [
    "CLUSTER_BOUNDARY_MAX_ROWS",
    "cluster_boundary_supported",
    "cute_dsl_post_norm_residual_pre_norm",
    "is_available",
    "mailbox_boundary_supported",
]
