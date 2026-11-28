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

"""CuTe DSL 16-CTA DSMEM cluster shifted mHC boundary backend."""

from __future__ import annotations

import torch
from tokenspeed_kernel._optional import has_optional_package
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    max_arch_version=ArchVersion(10, 9),
    vendors=frozenset({"nvidia"}),
)


def _blackwell_cute_dsl_available() -> bool:
    platform = current_platform()
    return (
        platform.is_nvidia
        and ArchVersion(10, 0) <= platform.arch_version <= ArchVersion(10, 9)
        and has_optional_package("cutlass")
        and has_optional_package("cuda.bindings")
    )


def mhc_boundary_max_decode_tokens_cute_dsl(device: torch.device | str | int) -> int:
    """Return max tokens fitting one 16-CTA cluster wave on ``device`` (or 0)."""
    if not _blackwell_cute_dsl_available():
        return 0
    from tokenspeed_kernel.thirdparty.cute_dsl.mhc_boundary import (
        max_decode_wave_tokens,
    )

    return max_decode_wave_tokens(device)


def prepare_boundary_weight_cute_dsl(weight: torch.Tensor) -> torch.Tensor | None:
    """Pack ``[24, 4 * hidden]`` FP32 weights for Blackwell ``mhc_boundary``, or None."""
    if not _blackwell_cute_dsl_available() or not weight.is_cuda:
        return None
    from tokenspeed_kernel.thirdparty.cute_dsl.mhc_boundary import (
        MIXES,
        MULT,
        TILE,
        MhcBoundaryKernel,
        prepare_boundary_weight,
    )

    if (
        weight.ndim != 2
        or weight.shape[0] != MIXES
        or weight.dtype != torch.float32
        or weight.shape[1] % MULT != 0
        or not MhcBoundaryKernel.supports(weight.shape[1] // MULT, MULT, TILE)
    ):
        return None
    return prepare_boundary_weight(weight)


def warmup_boundary_cute_dsl(
    hidden: int,
    sinkhorn_iters: int,
    *,
    device: torch.device | str = "cuda",
) -> None:
    """Compile all bucketed ``MhcBoundaryKernel`` tiles before CUDA graph capture."""
    if not _blackwell_cute_dsl_available():
        return
    from tokenspeed_kernel.thirdparty.cute_dsl.mhc_boundary import warmup_mhc_boundary

    warmup_mhc_boundary(
        hidden,
        sinkhorn_iters,
        use_pdl=pdl_enabled(),
        device=device,
    )


def mhc_boundary_eligible(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    boundary_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor,
) -> bool:
    """Check whether inputs meet the Blackwell DSMEM ``mhc_boundary`` contract."""
    if not _blackwell_cute_dsl_available():
        return False
    from tokenspeed_kernel.thirdparty.cute_dsl.mhc_boundary import (
        boundary_inputs_eligible,
    )

    return boundary_inputs_eligible(
        x,
        residual,
        pre,
        post,
        comb,
        boundary_weight,
        scale,
        base,
        sinkhorn_iters,
        norm_weight,
    )


@register_kernel(
    "residual",
    "mhc_shifted_post_pre_boundary",
    name="cute_dsl_mhc_shifted_post_pre_boundary",
    solution="cute_dsl",
    capability=_CAPABILITY,
    signatures=frozenset(
        {format_signature(residual=dense_tensor_format(torch.bfloat16))}
    ),
    priority=Priority.PERFORMANT,
)
def cute_dsl_mhc_shifted_post_pre_boundary(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    boundary_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
):
    from tokenspeed_kernel.thirdparty.cute_dsl.mhc_boundary import (
        mhc_shifted_post_pre_boundary,
    )

    return mhc_shifted_post_pre_boundary(
        x,
        residual,
        pre,
        post,
        comb,
        boundary_weight,
        scale,
        base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        norm_weight,
        norm_eps,
        use_pdl=pdl_enabled(),
    )
