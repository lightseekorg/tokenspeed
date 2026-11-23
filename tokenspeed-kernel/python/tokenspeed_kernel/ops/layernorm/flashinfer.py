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

"""FlashInfer layernorm kernels."""

import torch
from tokenspeed_kernel.platform import (
    CapabilityRequirement,
    current_platform,
    with_pdl_default,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

fused_add_rmsnorm = error_fn
gemma_fused_add_rmsnorm = error_fn
gemma_rmsnorm = error_fn
layernorm = error_fn
rmsnorm = error_fn


if current_platform().is_nvidia:
    try:
        from flashinfer import fused_add_rmsnorm as _fused_add_rmsnorm
        from flashinfer import gemma_fused_add_rmsnorm as _gemma_fused_add_rmsnorm
        from flashinfer import gemma_rmsnorm as _gemma_rmsnorm
        from flashinfer import (
            layernorm,
        )
        from flashinfer import rmsnorm as _rmsnorm

        fused_add_rmsnorm = with_pdl_default(_fused_add_rmsnorm)
        gemma_fused_add_rmsnorm = with_pdl_default(_gemma_fused_add_rmsnorm)
        gemma_rmsnorm = with_pdl_default(_gemma_rmsnorm)
        rmsnorm = with_pdl_default(_rmsnorm)

        @register_kernel(
            "layernorm",
            "gemma_rmsnorm",
            name="flashinfer_gemma_rmsnorm",
            solution="flashinfer",
            capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
            signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
            priority=Priority.PERFORMANT,
        )
        def flashinfer_gemma_rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            residual: torch.Tensor | None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            """FlashInfer kernel of :func:`tokenspeed_kernel.ops.layernorm.gemma_rmsnorm`."""
            if residual is None:
                return gemma_rmsnorm(x, weight, eps)
            gemma_fused_add_rmsnorm(x, residual, weight, eps)
            return x, residual

    except ImportError:
        pass

__all__ = [
    "fused_add_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "layernorm",
    "rmsnorm",
]
