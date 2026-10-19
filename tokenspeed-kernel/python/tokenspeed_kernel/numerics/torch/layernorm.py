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

"""PyTorch references for ``tokenspeed_kernel.ops.layernorm``.

Each reference takes the arguments its operator's dispatcher passes to the
registered kernels, does the math in FP32, and rounds once on output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.embedding import rotate
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FLOAT_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 RMSNorm over the last dimension, without a weight."""
    values = x.float()
    return values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)


def _centered_weight(weight: torch.Tensor, offset: float) -> torch.Tensor:
    """``weight + offset`` rounded through BF16, as the kernels load it."""
    return (weight.float() + offset).to(torch.bfloat16).float()


@register_kernel(
    "layernorm",
    "rmsnorm",
    name="torch_rmsnorm",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm of the last dimension scaled by ``weight``, in ``x``'s dtype.

    With ``residual``, normalizes the FP32 sum ``x + residual`` and also
    returns that sum rounded to ``x``'s dtype.
    """
    if residual is None:
        return (_rms_normalize(x, eps) * weight.float()).to(x.dtype)
    total = x.float() + residual.float()
    return (_rms_normalize(total, eps) * weight.float()).to(x.dtype), total.to(x.dtype)


@register_kernel(
    "layernorm",
    "qk_rmsnorm",
    name="torch_qk_rmsnorm",
    solution="torch",
    signatures=format_signatures(("q", "k"), "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head Q/K RMSNorm scaled by ``weight_offset + weight`` formed in FP32.

    The head width is the weights' length; ``q``/``k`` are
    ``[tokens, heads * head_dim]`` rows of any row stride.
    """

    def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        heads = x.reshape(-1, weight.shape[0])
        out = _rms_normalize(heads, eps) * (weight.float() + weight_offset)
        return out.to(x.dtype).view(x.shape)

    return norm(q, q_weight), norm(k, k_weight)


@register_kernel(
    "layernorm",
    "weightless_rmsnorm",
    name="torch_weightless_rmsnorm",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_weightless_rmsnorm(
    x: torch.Tensor, eps: float, out_dtype: torch.dtype
) -> torch.Tensor:
    """RMSNorm of the last dimension without a weight, in ``out_dtype``."""
    return _rms_normalize(x, eps).to(out_dtype)


@register_kernel(
    "layernorm",
    "affine_rmsnorm",
    name="torch_affine_rmsnorm",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_affine_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """RMSNorm scaled by the centered weight ``weight + weight_offset``."""
    return (_rms_normalize(x, eps) * _centered_weight(weight, weight_offset)).to(
        out_dtype
    )


@register_kernel(
    "layernorm",
    "weightless_qk_rmsnorm",
    name="torch_weightless_qk_rmsnorm",
    solution="torch",
    signatures=format_signatures(("q", "k"), "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_weightless_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    eps: float,
    q_scale: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight-free per-head Q/K RMSNorm; normalized queries scale by ``q_scale``.

    ``enable_pdl`` is a launch hint and has no effect here.
    """
    del enable_pdl
    q_out = _rms_normalize(q.reshape(-1, head_dim), eps) * q_scale
    k_out = _rms_normalize(k.reshape(-1, head_dim), eps)
    return (
        q_out.to(q.dtype).view(q.shape[0], -1),
        k_out.to(k.dtype).view(k.shape[0], -1),
    )


@register_kernel(
    "layernorm",
    "weightless_rmsnorm_gated_sigmoid",
    name="torch_weightless_rmsnorm_gated_sigmoid",
    solution="torch",
    signatures=format_signatures(("x", "gate"), "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_weightless_rmsnorm_gated_sigmoid(
    x: torch.Tensor,
    gate: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
    enable_pdl: bool,
) -> torch.Tensor:
    """Weight-free per-head RMSNorm times ``sigmoid(gate)``.

    ``enable_pdl`` is a launch hint and has no effect here.
    """
    del enable_pdl
    tokens = x.shape[0]
    normalized = _rms_normalize(x.reshape(tokens, num_heads, head_dim), eps)
    gated = normalized * gate.float().reshape(tokens, num_heads, head_dim).sigmoid()
    return gated.to(x.dtype).view(x.shape)


@register_kernel(
    "layernorm",
    "gated_rmsnorm",
    name="torch_gated_rmsnorm",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8_scale: torch.Tensor | None,
    weights_independent: bool,
) -> torch.Tensor:
    """Gated RMSNorm over groups of ``group_size`` features.

    ``norm(x) * gate(z)`` when ``norm_before_gate``, else ``norm(x * gate(z))``,
    with ``gate`` sigmoid or SiLU. An ``fp8_scale`` quantizes the
    ``x``-dtype result to ``float8_e4m3fn``. ``weights_independent`` is a
    launch hint and has no effect here.
    """
    del weights_independent
    values = x.float()
    gate = None
    if z is not None:
        gate = torch.sigmoid(z.float()) if sigmoid_gate else F.silu(z.float())
        if not norm_before_gate:
            values = values * gate
    groups = values.view(*values.shape[:-1], -1, group_size)
    weight = weight.float()
    if weight.shape[0] == group_size:
        normalized = _rms_normalize(groups, eps) * weight
    else:
        normalized = _rms_normalize(groups, eps).view_as(values) * weight
    out = normalized.view_as(values)
    if gate is not None and norm_before_gate:
        out = out * gate
    out = out.to(x.dtype)
    if fp8_scale is None:
        return out
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return (
        (out.float() / fp8_scale.float())
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )


@register_kernel(
    "layernorm",
    "fused_qk_rmsnorm_rope",
    name="torch_fused_qk_rmsnorm_rope",
    solution="torch",
    signatures=format_signatures(("q", "k"), "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_fused_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm rounded to the input dtype, then NeoX RoPE in FP32.

    ``cos_sin_cache`` packs ``[cos | sin]`` per position; outputs are
    contiguous ``[tokens, heads * head_dim]`` in the input dtype.
    """
    table = cos_sin_cache[positions.long()][:, None, :].float()
    cos, sin = table[..., : head_dim // 2], table[..., head_dim // 2 :]

    def norm_rope(x: torch.Tensor, weight: torch.Tensor, heads: int) -> torch.Tensor:
        values = x.reshape(x.shape[0], heads, head_dim)
        normed = (_rms_normalize(values, eps) * weight.float()).to(x.dtype).float()
        return rotate(normed, cos, sin, is_neox=True).to(x.dtype).flatten(1)

    return norm_rope(q, q_weight, num_q_heads), norm_rope(k, k_weight, num_kv_heads)
