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


"""Model-load helpers for the gfx950 A4W4 MoE package.

The package-prefill kernels consume the same gdot128-preshuffled weight storage
that the decode/prefill stages use, but with the ``(E, N, K/2)`` /
``(E, N, K/32)`` logical shapes. This module exposes ``attach_prefill_aliases``,
which the top-level weight preprocessor calls once per layer to attach
metadata-only views (``torch.as_strided``) of that storage onto the runtime
weight tensors. The views share storage with the preshuffled weights -- no
second model-sized copy is retained -- so attaching them is effectively free.
"""

from __future__ import annotations

import torch


def _is_gdot128_weight(weight: torch.Tensor) -> bool:
    """True if ``weight`` is a gdot128-preshuffled 3-D expert weight tensor."""
    return weight.ndim == 3 and bool(
        getattr(weight, "is_shuffled_for_gluon_dot", False)
    )


def _is_cdna4_scale(scale: torch.Tensor) -> bool:
    """True if ``scale`` is a CDNA4-swizzled 3-D scale tensor."""
    return scale.ndim == 3 and scale.stride(-2) == 1 and int(scale.shape[1]) % 32 == 0


def _make_gdot128_weight_alias(weight: torch.Tensor) -> torch.Tensor:
    """Expose gdot128 weight storage with the ``(E, N, K/2)`` logical shape.

    This is a metadata-only alias. The package kernels use the
    ``is_gdot128_shuffled`` marker to apply gdot128 byte addressing instead of
    interpreting the storage as a 16x16 shuffle.
    """
    if weight.ndim != 3 or not bool(
        getattr(weight, "is_shuffled_for_gluon_dot", False)
    ):
        raise ValueError("package prefill requires gdot128-preshuffled 3-D weights")
    experts, k_packed, n_cols = map(int, weight.shape)
    alias = torch.as_strided(
        weight,
        (experts, n_cols, k_packed),
        (n_cols * k_packed, k_packed, 1),
    )
    if hasattr(torch, "float4_e2m1fn_x2"):
        alias = alias.view(torch.float4_e2m1fn_x2)
    alias.is_shuffled = True
    alias.is_gdot128_shuffled = True
    return alias


def _make_gdot128_scale_alias(scale: torch.Tensor) -> torch.Tensor:
    """Expose CDNA4-swizzled scales with the ``(E, N, K/32)`` logical shape."""
    if scale.ndim != 3 or scale.stride(-2) != 1:
        raise ValueError("package prefill requires CDNA4-swizzled 3-D scales")
    experts, scale_linear, n_blocks = map(int, scale.shape)
    if scale_linear % 32 != 0:
        raise ValueError(f"invalid CDNA4 scale shape: {tuple(scale.shape)}")
    n_cols = n_blocks * 32
    k_scale = scale_linear // 32
    return torch.as_strided(
        scale,
        (experts, n_cols, k_scale),
        (n_cols * k_scale, k_scale, 1),
    )


def attach_prefill_aliases(
    w: torch.nn.Module,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> None:
    """Attach zero-copy package-prefill weight/scale aliases onto ``w``.

    Call once, after preshuffle has produced the gdot128 layout on
    ``w.w13_weight_triton_tensor`` / ``w.w2_weight_triton_tensor``. The aliases
    are metadata-only views over that existing storage, so this is free and
    lets the flag toggle at runtime without a restart. The dynamic apply entry
    reads these attributes off the runtime W tensors, so they are carried there.

    Best-effort: the aliases are only attached when the weights are
    gdot128-preshuffled 3-D tensors and the scales are CDNA4-swizzled. Models
    whose weights are not in that layout are left untouched and fall back to the
    reference path (the dynamic entry returns None without these aliases).

    Args:
        w: the MoE layer whose ``w13_weight_triton_tensor`` /
            ``w2_weight_triton_tensor`` were gdot128-preshuffled.
        w13_scale: swizzled CDNA4 gate/up scale tensor.
        w2_scale: swizzled CDNA4 down-projection scale tensor.
    """
    w13 = w.w13_weight_triton_tensor
    w2 = w.w2_weight_triton_tensor
    if not (
        _is_gdot128_weight(w13)
        and _is_gdot128_weight(w2)
        and _is_cdna4_scale(w13_scale)
        and _is_cdna4_scale(w2_scale)
    ):
        return
    w13.gluon_package_prefill_weight = _make_gdot128_weight_alias(w13)
    w13.gluon_package_prefill_scale = _make_gdot128_scale_alias(w13_scale)
    w2.gluon_package_prefill_weight = _make_gdot128_weight_alias(w2)
    w2.gluon_package_prefill_scale = _make_gdot128_scale_alias(w2_scale)
