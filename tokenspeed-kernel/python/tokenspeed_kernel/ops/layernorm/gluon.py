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

"""Narrow gfx950 producer fusion for Beam's existing SSMax norm operation."""

import torch
from tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax import (
    QK_RMSNORM_SSMAX_SIGNATURES,
    qk_rmsnorm_ssmax_traits,
    validate_qk_norm_arguments,
)
from tokenspeed_kernel.ops.layernorm.triton import UnscaledFP8QKVGate, rmsnorm
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel

NAME = "gluon_qk_rmsnorm_ssmax_gfx950"
TRAITS = {
    "tokens_min": frozenset({1}),
    "head_dim": frozenset({128}),
    "eps_ok": frozenset({True}),
    "heads": frozenset({(6, 1)}),
    "is_cuda": frozenset({True}),
    "ndim": frozenset({2}),
    "positions_contiguous": frozenset({True}),
    "rows_disjoint": frozenset({True}),
    "unit_stride": frozenset({True}),
    "weights_contiguous": frozenset({True}),
}


@register_kernel(
    "layernorm",
    "qk_rmsnorm_ssmax",
    name=NAME,
    solution="gluon",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=QK_RMSNORM_SSMAX_SIGNATURES,
    traits=TRAITS,
    priority=Priority.SPECIALIZED,
)
def gluon_qk_rmsnorm_ssmax_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    *,
    dequant: UnscaledFP8QKVGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse only Q normalization and SSMax; keep the portable K norm unchanged.

    Accept BF16 [M,768]/[M,128] Q/K, dense BF16 [128] weights and dense
    int32/int64 [M] positions on gfx950. Disjoint, unit-column-stride views
    are supported. Optional unscaled-FP8 dequantization is not admitted and
    ranked selection retains the portable operation for that feature.
    """
    validate_qk_norm_arguments(q, k, q_weight, k_weight, positions)
    if dequant is not None:
        raise ValueError(f"{NAME} does not support unscaled FP8 dequantization")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
    ):
        raise ValueError(f"{NAME} requires BF16 queries, keys and weights")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{NAME} requires int32/int64 positions")
    traits = qk_rmsnorm_ssmax_traits(q, k, q_weight, k_weight, eps, positions)
    if traits["tokens"] < 1 or any(
        traits[name] not in values
        for name, values in TRAITS.items()
        if name != "tokens_min"
    ):
        raise ValueError(
            f"{NAME} requires positive rows, Q6/KV1 D128 and supported contiguous-column metadata"
        )
    platform = current_platform()
    if not platform.is_amd or platform.arch_version != ArchVersion(9, 5):
        raise ValueError(f"{NAME} requires AMD gfx950")
    from tokenspeed_kernel_amd.ops.gfx950.layernorm.qk_rmsnorm_ssmax import (
        launch_gluon_qk_rmsnorm_ssmax_gfx950,
    )

    q_out = launch_gluon_qk_rmsnorm_ssmax_gfx950(q, q_weight, positions, eps)
    k_out = rmsnorm(
        k.reshape(-1, 1, 128), k_weight, eps, residual=None, out=None, enable_pdl=None
    )
    return q_out, k_out.flatten(1)
