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

"""Fused long-context log-scaling tau from per-token positions."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["log_scaling_tau"]


@triton.jit
def _log_scaling_tau_kernel(
    positions_ptr,
    out_ptr,
    n,
    n_floor,
    alpha,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    pos = tl.load(positions_ptr + offs, mask=mask, other=0)
    effective_n = (pos + 1).to(tl.float32) / n_floor
    tau = 1.0 + alpha * tl.log(tl.maximum(effective_n, 1.0))
    tl.store(out_ptr + offs, tau, mask=mask)


def log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    """Per-token attention-logit scale ``1 + alpha * log(max((pos+1)/n_floor, 1))``.

    One fused launch replacing the eager elementwise chain; valid for any
    position value (no precomputed-table bound).

    Args:
        positions: ``[T]`` integer absolute positions (CUDA).
        n_floor: Context length below which tau is exactly 1.
        alpha: Log-scaling slope from the model config.

    Returns:
        ``[T]`` fp32 tau values.
    """
    n = positions.shape[0]
    out = torch.empty(n, dtype=torch.float32, device=positions.device)
    if n == 0:
        return out
    BLOCK = 1024
    _log_scaling_tau_kernel[(triton.cdiv(n, BLOCK),)](
        positions,
        out,
        n,
        float(n_floor),
        float(alpha),
        BLOCK=BLOCK,
    )
    return out
