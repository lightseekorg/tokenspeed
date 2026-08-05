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
# Dispatches to a specialized kernel when supported, otherwise to portable torch.
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["attn_res_fwd"]

# Common specialized-kernel coverage. The larger gfx950 bound is restricted to
# the Kimi K3 shape and its fused output RMSNorm below.
_SUPPORTED_H = frozenset({4096, 5120, 6144, 7168, 8192})
_MAX_T = 16384
_MAX_GFX950_T = 65536
_MAX_N = 12


def _specialized_shape_eligible(
    tokens: int,
    hidden: int,
    candidates: int,
    *,
    fused_output_norm: bool,
) -> bool:
    if hidden not in _SUPPORTED_H or not 1 <= candidates <= _MAX_N:
        return False
    if 1 <= tokens <= _MAX_T:
        return True
    return (
        Platform.get().is_cdna4
        and hidden == 7168
        and fused_output_norm
        and _MAX_T < tokens <= _MAX_GFX950_T
    )


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
    eligible = _specialized_shape_eligible(
        T,
        H,
        N,
        fused_output_norm=out_norm_weight is not None,
    )
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


import tokenspeed_kernel.ops.attn_res.cuda  # noqa: E402,F401

# Registration side effects (must run so select_kernel can find the backends).
import tokenspeed_kernel.ops.attn_res.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attn_res.torch  # noqa: E402,F401
