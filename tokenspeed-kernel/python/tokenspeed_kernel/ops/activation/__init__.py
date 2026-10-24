# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Activation kernel entry points."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.activation.triton import (
    add3,
    relu2,
)
from tokenspeed_kernel.ops.activation.triton import silu_and_mul as triton_silu_and_mul
from tokenspeed_kernel.ops.activation.triton import situ_and_mul as triton_situ_and_mul
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.registry import error_fn


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Apply SwiGLU through the platform implementation.

    Positive ``limit`` values use the portable Triton implementation because
    the CUDA implementation does not expose the checkpoint's clamp semantics.

    """
    if current_platform().is_npu and limit is None:
        from tokenspeed_kernel.ops.activation.ascend import silu_and_mul as npu_swiglu

        return npu_swiglu(x, out)
    if (
        x.device.type == "npu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] <= 65535
        and x.shape[-1] % 64 == 0
        and limit is not None
        and limit > 0
        and (
            out is None
            or (
                out.shape == (x.shape[0], x.shape[1] // 2)
                and out.dtype == x.dtype
                and out.device == x.device
                and out.is_contiguous()
            )
        )
    ):
        from tokenspeed_kernel_npu.ops.dsv41_swiglu import swiglu, swiglu_batched

        if x.shape[0] >= 16 and x.shape[1] <= 2048:
            # Shared-expert shards are narrow; four rows amortize vector
            # scheduling without changing the scalar activation arithmetic.
            return swiglu_batched(
                x, limit, block_size=1024, rows_per_program=4, out=out
            )
        return swiglu(x, limit, block_size=512, out=out)
    if (
        limit is not None
        or current_platform().is_amd
        or flashinfer_silu_and_mul is error_fn
    ):
        return triton_silu_and_mul(x, out, limit=limit)
    return flashinfer_silu_and_mul(x, out, enable_pdl=pdl_enabled())


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """Apply SiTU through the portable Triton implementation."""

    return triton_situ_and_mul(
        x,
        out,
        beta=beta,
        linear_beta=linear_beta,
    )


__all__ = [
    "add3",
    "relu2",
    "silu_and_mul",
    "situ_and_mul",
]
