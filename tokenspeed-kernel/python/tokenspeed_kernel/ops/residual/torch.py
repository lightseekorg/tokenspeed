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
#
# Portable torch fallback for the Attention-Residual mix. Also selected when the
# shape falls outside the Blackwell kernel's supported range.
from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


@register_kernel(
    "residual",
    "attn_res_fwd",
    name="torch_attn_res_fwd",
    solution="torch",
    signatures=format_signatures(
        ("layer_residual", "block_residual"), "dense", {torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
)
def torch_attn_res_fwd(
    *,
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    eps,
    out_norm_weight=None,
    out_norm_eps=None,
    delta=None,
    num_valid_blocks=None,
    block_write_idx=-1,
) -> torch.Tensor:
    valid_blocks = (
        block_residual.shape[0] if num_valid_blocks is None else num_valid_blocks
    )
    if delta is not None:
        layer_residual.add_(delta)
    if block_write_idx >= 0:
        block_residual[block_write_idx].copy_(layer_residual)

    # Candidates [N, T, H] = blocks then layer; RMSNorm + softmax score + mix in
    # fp32 (this sits on the global residual backbone), bf16 out.
    values = torch.cat(
        (block_residual[:valid_blocks], layer_residual.unsqueeze(0)), dim=0
    ).float()
    rs = (values.square().mean(-1, keepdim=True) + eps).rsqrt()
    score_weight = rms_weight.float() * res_weight.float()  # [H]
    logits = (values * rs * score_weight).sum(-1)  # [N, T]
    probs = logits.softmax(0)  # over candidates
    out = (probs.unsqueeze(-1) * values).sum(0)  # [T, H]
    out = out.to(layer_residual.dtype)
    if out_norm_weight is not None:
        # Fused following RMSNorm: stats over the bf16-rounded mix (matches the
        # separate rmsnorm-kernel path).
        of = out.float()
        output_eps = eps if out_norm_eps is None else out_norm_eps
        rs = (of.square().mean(-1, keepdim=True) + output_eps).rsqrt()
        out = (of * rs * out_norm_weight.float()).to(layer_residual.dtype)
    return out


# ===-----------------------------------------------------------------------===#
# HY V4 independent Hyper-Connections (iHC)
# ===-----------------------------------------------------------------------#
#
# iHC replaces the single residual stream with ``hc_mult`` parallel channels.
# Unlike DeepSeek V4's mHC there is no combination matrix: the pre boundary
# emits one pre gate and one post gate per channel, the post boundary scatters
# the sub-block output across every channel, and the head merges the channels.
# Every projection weight is FP32 and every gate is evaluated in FP32. The
# channels may be BF16 or FP32; sub-block inputs and the head output are BF16.
_IHC_RESIDUAL_DTYPES = (torch.bfloat16, torch.float32)


def _ihc_check_common(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    gate_count: int,
) -> tuple[int, int, int]:
    """Validate the shared iHC operand contract and return (tokens, hc, hidden)."""
    if residual.ndim < 2:
        raise ValueError(f"iHC residual must be at least 2D, got {residual.ndim}D")
    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    if hc_mult <= 1:
        raise ValueError(f"iHC requires more than one channel, got {hc_mult}")
    if residual.dtype not in _IHC_RESIDUAL_DTYPES:
        raise ValueError(
            f"iHC residual must be bfloat16 or float32, got {residual.dtype}"
        )
    if fn.dtype != torch.float32 or hc_scale.dtype != torch.float32:
        raise ValueError("iHC projection and scale parameters must be float32")
    if hc_base.dtype != torch.float32:
        raise ValueError("iHC biases must be float32")
    if fn.shape != (gate_count, hc_mult * hidden_size):
        raise ValueError(
            f"iHC projection must have shape {(gate_count, hc_mult * hidden_size)}, "
            f"got {tuple(fn.shape)}"
        )
    if hc_scale.shape != (gate_count // hc_mult,):
        raise ValueError(
            f"iHC scale must have shape {(gate_count // hc_mult,)}, "
            f"got {tuple(hc_scale.shape)}"
        )
    if hc_base.shape != (gate_count,):
        raise ValueError(
            f"iHC bias must have shape {(gate_count,)}, got {tuple(hc_base.shape)}"
        )
    if residual.device != fn.device or residual.device != hc_base.device:
        raise ValueError("iHC operands must share one device")
    tokens = int(residual.numel() // (hc_mult * hidden_size))
    return tokens, hc_mult, hidden_size


def _ihc_gates(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    *,
    gate_count: int,
    magnitude: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Project the flattened channels and return gate logits turned into gates.

    The reduced row is RMS-normalized with ``rms_eps`` exactly like the
    reference iHC boundary: the projection reads the raw row and the RMS factor
    is applied to the projection output, not to the row.
    """
    tokens, hc_mult, hidden_size = _ihc_check_common(
        residual, fn, hc_scale, hc_base, gate_count
    )
    flat = residual.reshape(tokens, hc_mult * hidden_size).float()
    squared_mean = flat.square().mean(-1, keepdim=True) + rms_eps
    rsqrt = squared_mean.rsqrt()
    mixes = torch.nn.functional.linear(flat, fn) * rsqrt

    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_eps
    if magnitude is None:
        return pre, None
    post = (
        magnitude
        * torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        + hc_eps
    )
    return pre, post


@register_kernel(
    "residual",
    "ihc_pre",
    name="torch_ihc_pre",
    solution="torch",
    signatures=frozenset(
        format_signature(
            residual=dense_tensor_format(residual_dtype),
            fn=dense_tensor_format(torch.float32),
            hc_scale=dense_tensor_format(torch.float32),
            hc_base=dense_tensor_format(torch.float32),
        )
        for residual_dtype in _IHC_RESIDUAL_DTYPES
    ),
    priority=Priority.PORTABLE,
)
def torch_ihc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    rms_eps: float,
    hc_eps: float,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the portable iHC pre boundary: reduce channels, emit post gates."""
    if (norm_weight is None) != (norm_eps is None):
        raise ValueError("norm_weight and norm_eps must be provided together")
    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    outer_shape = residual.shape[:-2]
    pre, post = _ihc_gates(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        gate_count=2 * hc_mult,
        magnitude=magnitude,
    )
    assert post is not None
    reduced = torch.sum(
        pre.unsqueeze(-1) * residual.reshape(-1, hc_mult, hidden_size).float(), dim=1
    )
    # The norm reads the reduction rounded to the residual dtype and runs in
    # FP32, so the output is rounded to BF16 once.
    layer_input = reduced.to(residual.dtype).float()
    if norm_weight is not None:
        layer_input = torch.nn.functional.rms_norm(
            layer_input, (hidden_size,), norm_weight.float(), norm_eps
        )
    return (
        layer_input.to(torch.bfloat16).reshape(*outer_shape, hidden_size),
        post.reshape(*outer_shape, hc_mult),
    )


@register_kernel(
    "residual",
    "ihc_post",
    name="torch_ihc_post",
    solution="torch",
    signatures=frozenset(
        format_signature(
            block_output=dense_tensor_format(torch.bfloat16),
            residual=dense_tensor_format(residual_dtype),
            gates=dense_tensor_format(torch.float32),
        )
        for residual_dtype in _IHC_RESIDUAL_DTYPES
    ),
    priority=Priority.PORTABLE,
)
def torch_ihc_post(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Run the portable iHC post boundary: gate the output onto every channel."""
    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    if block_output.shape[-1] != hidden_size:
        raise ValueError(
            f"iHC block output must have last dimension {hidden_size}, "
            f"got {tuple(block_output.shape)}"
        )
    if gates.shape[-1] != hc_mult:
        raise ValueError(
            f"iHC post gates must have last dimension {hc_mult}, "
            f"got {tuple(gates.shape)}"
        )
    scattered = gates.float().unsqueeze(-1) * block_output.float().unsqueeze(-2)
    return (scattered + residual.float()).to(residual.dtype)


@register_kernel(
    "residual",
    "ihc_head",
    name="torch_ihc_head",
    solution="torch",
    signatures=frozenset(
        format_signature(
            residual=dense_tensor_format(residual_dtype),
            fn=dense_tensor_format(torch.float32),
            hc_scale=dense_tensor_format(torch.float32),
            hc_base=dense_tensor_format(torch.float32),
        )
        for residual_dtype in _IHC_RESIDUAL_DTYPES
    ),
    priority=Priority.PORTABLE,
)
def torch_ihc_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Run the portable iHC head: merge the channels into one hidden state."""
    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    gates, _ = _ihc_gates(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        gate_count=hc_mult,
        magnitude=None,
    )
    merged = torch.sum(
        gates.unsqueeze(-1) * residual.reshape(-1, hc_mult, hidden_size).float(), dim=1
    )
    return merged.to(torch.bfloat16).reshape(*residual.shape[:-2], hidden_size)
