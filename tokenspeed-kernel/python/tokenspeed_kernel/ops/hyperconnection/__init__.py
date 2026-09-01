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

"""Qwen gated-residual hyperconnection kernel entry points."""

from __future__ import annotations

import math

import torch as _torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "gated_residual_combine",
    "gated_residual_mix",
    "refresh_gated_residual_weight_cache",
]


def _flatten_rows(value: _torch.Tensor, width: int, name: str) -> _torch.Tensor:
    if value.ndim < 1 or value.shape[-1] != width:
        raise ValueError(
            f"{name} must have last dimension {width}, got {tuple(value.shape)}"
        )
    return value.reshape(-1, width)


def _same_tensor_contract(
    reference: _torch.Tensor, value: _torch.Tensor, name: str
) -> None:
    if value.dtype != reference.dtype:
        raise ValueError(
            f"{name} dtype must match the input ({reference.dtype}), got {value.dtype}"
        )
    if value.device != reference.device:
        raise ValueError(
            f"{name} device must match the input ({reference.device}), got {value.device}"
        )


def refresh_gated_residual_weight_cache(up_weight: _torch.Tensor, lowrank: int) -> bool:
    """Refresh cached derived mix-up weights after an in-place weight load.

    CUDA graphs retain the address of backend-specific derived weights. This
    function updates any existing derived allocation without changing that
    address. It is a no-op when no backend has cached the weight.

    Args:
        up_weight: Source mix-up weight shaped ``[wide, lowrank]``.
        lowrank: Rank of the mix gate bottleneck.

    Returns:
        Whether a cached derived allocation existed and was refreshed.
    """
    if lowrank <= 0:
        raise ValueError("lowrank must be positive")
    if up_weight.ndim != 2 or int(up_weight.shape[1]) != lowrank:
        raise ValueError(
            f"up_weight must have shape [wide, {lowrank}], got "
            f"{tuple(up_weight.shape)}"
        )
    from tokenspeed_kernel.ops.hyperconnection.cute_dsl import (
        _refresh_padded_up_weight,
    )

    return _refresh_padded_up_weight(up_weight, lowrank)


def gated_residual_mix(
    normalized: _torch.Tensor,
    projection_weight: _torch.Tensor,
    up_weight: _torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    *,
    projection_scale: float = 1.0,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[_torch.Tensor, _torch.Tensor | None]:
    """Mix normalized hyperconnection branches and optionally form inject logits.

    The first projection is stored as one matrix. Its leading ``lowrank`` rows
    are the mix-down weight and, when present, its final ``hc_count`` rows are
    the block-injection weight. This preserves a single read of the wide input.

    Args:
        normalized: Normalized GPU residual streams shaped
            ``[..., hc_count * hidden_size]``.
        projection_weight: Fused down/inject weight shaped either
            ``[lowrank, hc_count * hidden_size]`` or
            ``[lowrank + hc_count, hc_count * hidden_size]``.
        up_weight: Mix-up weight shaped
            ``[hc_count * hidden_size, lowrank]``.
        hc_count: Number of residual branches.
        hidden_size: Width of one branch.
        lowrank: Rank of the mix gate bottleneck.
        projection_scale: Scale applied to down and inject projection results.
            It is ``1`` when an exact power-of-two scale was folded into the
            checkpoint weight and ``1 / hc_count`` otherwise.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        A pair containing the mixed tensor shaped ``[..., hidden_size]`` and
        optional inject logits shaped ``[..., hc_count]``.
    """
    if hc_count <= 1 or hidden_size <= 0 or lowrank <= 0:
        raise ValueError("hc_count must exceed one and all dimensions must be positive")
    if not math.isfinite(projection_scale) or projection_scale <= 0:
        raise ValueError("projection_scale must be finite and positive")

    wide = hc_count * hidden_size
    flat = _flatten_rows(normalized, wide, "normalized")
    if projection_weight.ndim != 2 or projection_weight.shape[1] != wide:
        raise ValueError(
            "projection_weight must have shape "
            f"[{lowrank} or {lowrank + hc_count}, {wide}], got "
            f"{tuple(projection_weight.shape)}"
        )
    projection_rows = int(projection_weight.shape[0])
    if projection_rows not in (lowrank, lowrank + hc_count):
        raise ValueError(
            f"projection_weight has {projection_rows} rows; expected {lowrank} "
            f"or {lowrank + hc_count}"
        )
    if up_weight.shape != (wide, lowrank):
        raise ValueError(
            f"up_weight must have shape {(wide, lowrank)}, got {tuple(up_weight.shape)}"
        )
    _same_tensor_contract(flat, projection_weight, "projection_weight")
    _same_tensor_contract(flat, up_weight, "up_weight")
    if not flat.is_cuda:
        raise ValueError("gated_residual_mix requires GPU tensors")

    rows = int(flat.shape[0])
    has_inject = projection_rows != lowrank
    leading_shape = normalized.shape[:-1]
    if rows == 0:
        mixed = normalized.new_empty((*leading_shape, hidden_size))
        inject = (
            normalized.new_empty((*leading_shape, hc_count)) if has_inject else None
        )
        return mixed, inject

    traits = {
        "num_tokens": rows,
        "hc_count": hc_count,
        "hidden_size": hidden_size,
        "lowrank": lowrank,
        "has_inject": has_inject,
        "contiguous": bool(
            flat.is_contiguous()
            and projection_weight.is_contiguous()
            and up_weight.is_contiguous()
        ),
        "folded_scale": projection_scale == 1.0,
        "deterministic": _torch.are_deterministic_algorithms_enabled(),
        "capturing": bool(flat.is_cuda and _torch.cuda.is_current_stream_capturing()),
    }
    signature = format_signature(
        normalized=dense_tensor_format(flat.dtype),
        projection_weight=dense_tensor_format(projection_weight.dtype),
        up_weight=dense_tensor_format(up_weight.dtype),
    )
    kernel = select_kernel(
        "hyperconnection",
        "mix",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record("hyperconnection", "mix", kernel.name, flat.dtype, traits)
    with kernel_scope(
        "hyperconnection",
        "mix",
        flat.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        mixed, inject = kernel(
            flat,
            projection_weight,
            up_weight,
            hc_count,
            hidden_size,
            lowrank,
            projection_scale,
        )
    mixed = mixed.reshape(*leading_shape, hidden_size)
    if inject is not None:
        inject = inject.reshape(*leading_shape, hc_count)
    return mixed, inject


def gated_residual_combine(
    block_output: _torch.Tensor,
    residual: _torch.Tensor,
    inject_logits: _torch.Tensor,
    hc_count: int,
    hidden_size: int,
    *,
    override: str | None = None,
    solution: str | None = None,
) -> _torch.Tensor:
    """Gate one sublayer output and inject it into every residual branch.

    Args:
        block_output: GPU sublayer output shaped ``[..., hidden_size]``.
        residual: Hyperconnection stream shaped
            ``[..., hc_count * hidden_size]``.
        inject_logits: Per-branch logits shaped ``[..., hc_count]``.
        hc_count: Number of residual branches.
        hidden_size: Width of one branch.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Updated residual stream with the same shape as ``residual``.
    """
    wide = hc_count * hidden_size
    block_flat = _flatten_rows(block_output, hidden_size, "block_output")
    residual_flat = _flatten_rows(residual, wide, "residual")
    inject_flat = _flatten_rows(inject_logits, hc_count, "inject_logits")
    rows = int(block_flat.shape[0])
    if residual_flat.shape[0] != rows or inject_flat.shape[0] != rows:
        raise ValueError("block_output, residual, and inject_logits must share rows")
    _same_tensor_contract(block_flat, residual_flat, "residual")
    _same_tensor_contract(block_flat, inject_flat, "inject_logits")
    if not block_flat.is_cuda:
        raise ValueError("gated_residual_combine requires GPU tensors")
    if rows == 0:
        return residual.to(block_output.dtype)

    traits = {
        "num_tokens": rows,
        "hc_count": hc_count,
        "hidden_size": hidden_size,
    }
    signature = format_signature(
        block_output=dense_tensor_format(block_flat.dtype),
        residual=dense_tensor_format(residual_flat.dtype),
        inject_logits=dense_tensor_format(inject_flat.dtype),
    )
    kernel = select_kernel(
        "hyperconnection",
        "combine",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record(
        "hyperconnection", "combine", kernel.name, block_flat.dtype, traits
    )
    with kernel_scope(
        "hyperconnection",
        "combine",
        block_flat.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        result = kernel(
            block_flat,
            residual_flat,
            inject_flat,
            hc_count,
            hidden_size,
        )
    return result.reshape(residual.shape)


import tokenspeed_kernel.ops.hyperconnection.cute_dsl  # noqa: E402,F401

# Registration side effects must run after the public API is defined.
import tokenspeed_kernel.ops.hyperconnection.triton  # noqa: E402,F401
