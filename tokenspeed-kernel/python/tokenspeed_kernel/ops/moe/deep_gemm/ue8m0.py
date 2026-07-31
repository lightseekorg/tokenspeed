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

"""UE8M0 (power-of-two) block scales for DeepGEMM's FP8 path.

DeepGEMM ships two FP8 block-scale kernels on Hopper -- ``1d2d``, which consumes
arbitrary FP32 scales (the recipe most FP8 checkpoints are quantized with), and
``1d1d``. On sm100+ only ``1d1d`` exists, and it reads a scale's exponent bits
only: anything that is not an exact power of two is silently misread, producing
NaN or wrong results rather than an error.

So on sm100+ every FP8 operand handed to DeepGEMM must carry power-of-two
scales. Weights are converted once at load by dequantizing with the checkpoint
scale and re-quantizing against a power-of-two one (:func:`requantize_to_ue8m0_`)
-- rounding the scale alone would keep the old FP8 values and lose up to a bit
of their mantissa. Activations are quantized directly against a power-of-two
scale (:func:`per_token_group_quant_fp8_ue8m0`).

The scales stay in an FP32 tensor holding power-of-two values, which DeepGEMM
accepts on every architecture; the packed-int32 UE8M0 layout is a bandwidth
optimization, not a correctness requirement.
"""

from __future__ import annotations

import logging
import os

import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform

__all__ = [
    "deep_gemm_requires_ue8m0",
    "is_ue8m0",
    "per_token_group_quant_fp8_ue8m0",
    "requantize_to_ue8m0_",
]

logger = logging.getLogger(__name__)

# First architecture whose only DeepGEMM FP8 block-scale kernel is ``1d1d``.
_UE8M0_MIN_ARCH = ArchVersion(10, 0)
_FP8_MAX = 448.0
_DISABLE_ENV = "TOKENSPEED_DISABLE_DEEP_GEMM_UE8M0"


def deep_gemm_requires_ue8m0() -> bool:
    """Whether DeepGEMM on this device reads block scales as UE8M0.

    Returns:
        True on sm100+ NVIDIA devices, where DeepGEMM only offers the ``1d1d``
        FP8 kernel. Set ``TOKENSPEED_DISABLE_DEEP_GEMM_UE8M0=1`` to opt out (for
        example to compare against the unconverted checkpoint).
    """
    if os.environ.get(_DISABLE_ENV) == "1":
        return False
    platform = current_platform()
    return platform.is_nvidia and platform.arch_version >= _UE8M0_MIN_ARCH


def is_ue8m0(scale: torch.Tensor) -> bool:
    """Whether every non-zero entry of ``scale`` is an exact power of two."""
    values = scale.detach().float().flatten()
    values = values[values > 0]
    if values.numel() == 0:
        return True
    exponents = torch.log2(values)
    return bool(torch.equal(exponents, exponents.round()))


def _ceil_to_pow2(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(x)))


def per_token_group_quant_fp8_ue8m0(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to FP8 against power-of-two per-group scales.

    The repo's ``per_token_group_quant_fp8`` can emit UE8M0 scales only in the
    packed-int32 TMA layout, which cannot be sent over DeepEP's normal dispatch
    (it requires ``[tokens, groups]`` FP32). This returns that layout instead,
    with power-of-two values.

    Args:
        x: ``[tokens, hidden]`` activations; ``hidden`` must be a multiple of
            ``group_size``.
        group_size: Elements sharing one scale.

    Returns:
        ``(quantized, scales)`` with ``quantized`` ``[tokens, hidden]``
        float8_e4m3fn and ``scales`` ``[tokens, hidden / group_size]`` float32.
    """
    tokens, hidden = x.shape
    if hidden % group_size:
        raise ValueError(f"hidden {hidden} is not a multiple of {group_size}")

    grouped = x.float().view(tokens, hidden // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=1e-10)
    scales = _ceil_to_pow2(amax / _FP8_MAX)
    quantized = (grouped / scales.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).view(tokens, hidden), scales


def requantize_to_ue8m0_(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_shape: tuple[int, int] = (128, 128),
) -> None:
    """Re-quantize block-scaled FP8 weights onto power-of-two scales, in place.

    Dequantizes with the checkpoint scale and re-quantizes against a
    power-of-two one, so both the FP8 values and the scales are rewritten. Only
    rounding the scale up would leave the old values scaled by up to 2x, wasting
    a bit of their mantissa; recomputing them costs one extra FP8 rounding.

    Args:
        weight: ``[experts, n, k]`` or ``[n, k]`` float8_e4m3fn weights,
            mutated in place.
        scale: Matching block scales, ``[..., ceil(n / block_n),
            ceil(k / block_k)]`` float32, mutated in place. Trailing blocks that
            cover padding beyond ``n`` / ``k`` are left untouched.
        block_shape: ``(block_n, block_k)`` quantization granularity.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"expected float8_e4m3fn weights, got {weight.dtype}")
    if weight.numel() == 0:
        return

    block_n, block_k = block_shape
    weights = weight if weight.dim() == 3 else weight.unsqueeze(0)
    scales = scale if scale.dim() == 3 else scale.unsqueeze(0)
    _, n, k = weights.shape
    if n % block_n or k % block_k:
        raise ValueError(
            f"weight [{n}, {k}] is not tiled by block shape {block_shape}; "
            "requantization would straddle a block boundary"
        )
    blocks_n, blocks_k = n // block_n, k // block_k

    # One expert at a time: the FP32 dequantized copy is 4x the FP8 weight, and
    # this runs while the rest of the checkpoint is still being loaded.
    for expert in range(weights.shape[0]):
        w = weights[expert]
        s = scales[expert, :blocks_n, :blocks_k].float()
        expanded = s.repeat_interleave(block_n, dim=0).repeat_interleave(
            block_k, dim=1
        )
        dequantized = w.float() * expanded

        tiled = dequantized.view(blocks_n, block_n, blocks_k, block_k)
        amax = tiled.abs().amax(dim=(1, 3)).clamp(min=1e-10)
        new_scale = _ceil_to_pow2(amax / _FP8_MAX)
        requantized = tiled / new_scale[:, None, :, None]
        w.copy_(
            requantized.clamp(-_FP8_MAX, _FP8_MAX)
            .view(n, k)
            .to(torch.float8_e4m3fn)
        )
        scales[expert, :blocks_n, :blocks_k] = new_scale
