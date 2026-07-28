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
# Attention-Residual mixing op: RMSNorm + per-candidate softmax score + weighted
# sum over a set of residual-stream snapshots (the Kimi-K3 AttnRes block-mix).
# Dispatches to the Blackwell TMA kernel, falling back to a portable torch path
# on other hardware or for shapes outside the kernel's supported range.
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["attn_res_fwd", "attn_res_fwd_v2", "attn_res_fwd_v2_cuda_available"]

# Blackwell kernel coverage. This must stay a subset of the authoritative
# TVM_FFI_ICHECK bounds in csrc/attn_res_binding.cu; this gate only decides when
# to fall back to torch.
_SUPPORTED_H = frozenset({4096, 5120, 6144, 7168, 8192})
_MAX_T = 16384
_MAX_N = 12

# v2 (online-softmax, PDL) kernel coverage; subset of the TVM_FFI_ICHECK bounds
# in csrc/attn_res_v2_binding.cu.
_V2_H = 7168
_V2_MAX_KB = 8
_V2_MAX_T = 16384


def attn_res_fwd(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    eps=1e-6,
    out_norm_weight=None,
):
    """Fused Attention-Residual forward.

    Candidates are ``block_residual[0..K-1]`` followed by ``layer_residual``
    (N = K + 1). Computes ``softmax_n(<RMSNorm(v_n), rms_weight * res_weight>)``
    over candidates, then the weighted sum of the raw candidates.

    Args:
        layer_residual: bf16 ``[T, H]`` current residual stream.
        block_residual: bf16 ``[K, T, H]`` K periodic snapshots.
        res_weight: bf16 ``[H]`` scorer projection weight.
        rms_weight: bf16 ``[H]`` RMSNorm weight.
        eps: RMSNorm epsilon.
        out_norm_weight: optional bf16 ``[H]``; when given, the following
            RMSNorm (same eps) is fused into the epilogue and the return value
            is the normed mix.

    Returns:
        bf16 ``[T, H]`` mixed residual (normed when ``out_norm_weight`` given).
    """
    T, H = layer_residual.shape
    N = block_residual.shape[0] + 1
    eligible = H in _SUPPORTED_H and 1 <= T <= _MAX_T and 1 <= N <= _MAX_N
    signature = format_signature(
        layer_residual=dense_tensor_format(layer_residual.dtype),
        block_residual=dense_tensor_format(block_residual.dtype),
    )
    kernel = select_kernel(
        "attn_res",
        "fwd",
        signature,
        traits={
            "fused_output_norm": out_norm_weight is not None,
            "large_prefill": T > 32,
        },
        solution=None if eligible else "torch",
    )
    return kernel(
        layer_residual=layer_residual,
        block_residual=block_residual,
        res_weight=res_weight,
        rms_weight=rms_weight,
        eps=eps,
        out_norm_weight=out_norm_weight,
    )


def attn_res_fwd_v2_cuda_available(hidden_size: int, max_num_blocks: int) -> bool:
    """Capability probe: can ``attn_res_fwd_v2`` serve every mix of this model
    on the CUDA solution (no silent torch fallback in the hot path)?

    Args:
        hidden_size: model hidden size (only 7168 is instantiated).
        max_num_blocks: largest candidate-block count any mix will pass.

    Returns:
        True when the SM100 v2 kernel is loadable and covers the shapes.
    """
    from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL_V2

    return (
        _HAS_CUDA_KERNEL_V2
        and hidden_size == _V2_H
        and 1 <= max_num_blocks <= _V2_MAX_KB
    )


def attn_res_fwd_v2(
    prefix,
    delta,
    block_residual,
    res_weight,
    rms_weight,
    out_norm_weight,
    eps,
    out_norm_eps,
    enable_pdl=False,
    out=None,
):
    """Single-kernel AttnRes forward: optional residual accumulate + mix + norm.

    Candidates are ``block_residual[0..KB-1]`` followed by ``prefix``
    (N = KB + 1). Computes ``softmax_n(<RMSNorm(v_n), rms_weight *
    res_weight>)`` over candidates, the weighted sum of the raw candidates,
    then the following RMSNorm with ``out_norm_weight``.

    Args:
        prefix: bf16 ``[T, H]`` contiguous residual stream. When ``delta`` is
            given, updated IN PLACE with ``prefix += delta`` (bf16) before the
            mix, so the caller's residual accumulate is fused away.
        delta: optional bf16 ``[T, H]`` contiguous residual increment (e.g. the
            all-reduced attention output).
        block_residual: bf16 ``[KB, T, H]`` snapshots; leading dims may be
            strided, rows dense.
        res_weight: bf16 ``[H]`` scorer projection weight.
        rms_weight: bf16 ``[H]`` candidate RMSNorm weight.
        out_norm_weight: bf16 ``[H]`` fused following-RMSNorm weight.
        eps: candidate RMSNorm epsilon.
        out_norm_eps: following RMSNorm epsilon.
        enable_pdl: launch with programmatic stream serialization when the CUDA
            solution runs (ignored by the torch fallback).
        out: optional preallocated bf16 ``[T, H]`` destination.

    Returns:
        bf16 ``[T, H]`` normed mix.
    """
    T, H = prefix.shape
    KB = block_residual.shape[0]
    eligible = H == _V2_H and 1 <= T <= _V2_MAX_T and 1 <= KB <= _V2_MAX_KB
    signature = format_signature(
        prefix=dense_tensor_format(prefix.dtype),
        block_residual=dense_tensor_format(block_residual.dtype),
    )
    kernel = select_kernel(
        "attn_res",
        "fwd_v2",
        signature,
        traits={
            "has_delta": delta is not None,
            "large_prefill": T > 32,
        },
        solution=None if eligible else "torch",
    )
    return kernel(
        prefix=prefix,
        delta=delta,
        block_residual=block_residual,
        res_weight=res_weight,
        rms_weight=rms_weight,
        out_norm_weight=out_norm_weight,
        eps=eps,
        out_norm_eps=out_norm_eps,
        enable_pdl=enable_pdl,
        out=out,
    )


import tokenspeed_kernel.ops.attn_res.cuda  # noqa: E402,F401

# Registration side effects (must run so select_kernel can find the backends).
import tokenspeed_kernel.ops.attn_res.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attn_res.torch  # noqa: E402,F401
