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

"""Intel XPU SiLU-and-mul (vllm-xpu-kernels backend).

Direct-call function mirroring ``tokenspeed_kernel.ops.activation.triton.
silu_and_mul`` so the TokenSpeed runtime layer can swap the import in place on
Intel XPU.

vllm-xpu-kernels Torch op: ``torch.ops._C.silu_and_mul(out, input)`` -> writes
``out`` (shape ``[..., input.shape[-1] // 2]``).
"""

from __future__ import annotations

import torch

_C = getattr(torch.ops, "_C", None)


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """SiLU(x[..., :d]) * x[..., d:] on Intel XPU, written into ``out``.

    TODO(intel): verify exact vllm op name/arg order vs v0.1.7. ``enable_pdl``
    is accepted for signature parity with the Triton kernel and ignored.
    """
    _C.silu_and_mul(out, x)
    return out
