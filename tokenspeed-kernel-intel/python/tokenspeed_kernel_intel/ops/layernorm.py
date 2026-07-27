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

"""Intel XPU RMSNorm (vllm-xpu-kernels backend).

Direct-call functions (not registry-dispatched) mirroring the signature of
``tokenspeed_kernel.ops.layernorm.triton.rmsnorm`` so the TokenSpeed runtime
layer can swap the import in place when running on Intel XPU.

vllm-xpu-kernels Torch ops (registered via ``vllm_xpu_kernels._C``):
* ``torch.ops._C.rms_norm(out, input, weight, epsilon)``            -> writes out
* ``torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)`` -> in place
"""

from __future__ import annotations

import torch

_C = getattr(torch.ops, "_C", None)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm on Intel XPU. Contract matches the Triton ``rmsnorm``.

    TODO(intel): verify exact vllm op names/arg order vs v0.1.7 and validate
    against the numerics reference (dtype upcast, epsilon placement).
    """
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual

    if residual is not None:
        # Fused add + RMSNorm, in place on x and residual.
        _C.fused_add_rms_norm(x, residual, weight, eps)
        return x, residual

    out = torch.empty_like(x) if out is None else out
    _C.rms_norm(out, x, weight, eps)
    return out
