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
# Loader for the packed Blackwell Attention-Residual forward kernel. The
# compiled ``.so`` exports ``attn_res_fwd`` via tvm_ffi.
import functools
from pathlib import Path

import torch


@functools.cache
def _load_attn_res_module():
    import tvm_ffi

    so_path = Path(__file__).resolve().parent / "objs" / "attn_res" / "attn_res.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel attn_res library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def has_attn_res_fwd() -> bool:
    """True when the Blackwell attn_res kernel is built and loadable."""
    try:
        module = _load_attn_res_module()
    except Exception:
        return False
    return hasattr(module, "attn_res_fwd")


def attn_res_fwd_packed(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float = 1e-6,
    out_norm_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused Attention-Residual forward (RMSNorm + per-candidate softmax + mix).

    Candidates are ``block_residual[0..K-1]`` then ``layer_residual`` (N=K+1).
    All inputs must be contiguous bf16 CUDA tensors; supports B=1, N in [1,12],
    T in [1,16384], H a multiple of 1024 in [4096,8192].

    Args:
        layer_residual: bf16 ``[T, B, H]`` current residual stream.
        block_residual: bf16 ``[K, T, B, H]`` periodic snapshots.
        res_weight: bf16 ``[H]`` scorer projection weight.
        rms_weight: bf16 ``[H]`` RMSNorm weight.
        rms_eps: RMSNorm epsilon.
        out_norm_weight: optional bf16 ``[H]``; when given the following RMSNorm
            (same eps) is fused into the epilogue: ``rmsnorm(mix) * weight``.

    Returns:
        bf16 ``[T, B, H]`` mixed residual.
    """
    T, B, H = layer_residual.shape
    N = block_residual.shape[0] + 1
    output = torch.empty_like(layer_residual)
    # rsigma/probs/logits: mandatory caller-allocated aux buffers the kernel always
    # writes (per-candidate rsigma / softmax probs / logits); discarded here.
    rsigma = torch.empty((N, T, B), device=layer_residual.device, dtype=torch.float32)
    probs = torch.empty_like(rsigma)
    logits = torch.empty_like(rsigma)
    module = _load_attn_res_module()
    if out_norm_weight is None:
        module.attn_res_fwd(
            layer_residual,
            block_residual,
            res_weight,
            rms_weight,
            output,
            rsigma,
            probs,
            logits,
            float(rms_eps),
        )
    else:
        module.attn_res_fwd_out_norm(
            layer_residual,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            output,
            rsigma,
            probs,
            logits,
            float(rms_eps),
        )
    return output
