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

from functools import wraps
from inspect import signature

import torch
from tokenspeed_kernel.platform import (
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import (
    Priority,
    WarmupBehavior,
    error_fn,
    register_kernel,
)
from tokenspeed_kernel.signature import format_signatures

fused_add_rmsnorm = error_fn
gemma_fused_add_rmsnorm = error_fn
gemma_rmsnorm = error_fn
layernorm = error_fn
rmsnorm = error_fn


def _resolve_enable_pdl(enable_pdl: bool | None) -> bool:
    return pdl_enabled() if enable_pdl is None else enable_pdl


def _with_pdl_default(function):
    enable_pdl_index = tuple(signature(function).parameters).index("enable_pdl")

    @wraps(function)
    def wrapper(*args, **kwargs):
        if len(args) > enable_pdl_index:
            args = (
                *args[:enable_pdl_index],
                _resolve_enable_pdl(args[enable_pdl_index]),
                *args[enable_pdl_index + 1 :],
            )
        else:
            kwargs["enable_pdl"] = _resolve_enable_pdl(kwargs.get("enable_pdl"))
        return function(*args, **kwargs)

    return wrapper


if current_platform().is_nvidia:
    try:
        from flashinfer import fused_add_rmsnorm as _fused_add_rmsnorm
        from flashinfer import gemma_fused_add_rmsnorm as _gemma_fused_add_rmsnorm
        from flashinfer import gemma_rmsnorm as _gemma_rmsnorm
        from flashinfer import (
            layernorm,
        )
        from flashinfer import rmsnorm as _rmsnorm

        _fused_add_rmsnorm = _with_pdl_default(_fused_add_rmsnorm)
        gemma_fused_add_rmsnorm = _with_pdl_default(_gemma_fused_add_rmsnorm)
        gemma_rmsnorm = _with_pdl_default(_gemma_rmsnorm)
        _rmsnorm = _with_pdl_default(_rmsnorm)

        @register_kernel(
            "layernorm",
            "rmsnorm",
            name="flashinfer_rmsnorm",
            solution="flashinfer",
            capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
            signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
            traits={
                "has_residual": frozenset({False}),
                "has_out": frozenset({False, True}),
            },
            priority=Priority.PERFORMANT + 1,
            warmup_behavior=WarmupBehavior.JIT_COMPILE,
        )
        def rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            residual: torch.Tensor | None,
            out: torch.Tensor | None,
            enable_pdl: bool | None,
        ) -> torch.Tensor:
            if residual is not None:
                raise ValueError("FlashInfer rmsnorm does not accept a residual")
            return _rmsnorm(x, weight, eps, out=out, enable_pdl=enable_pdl)

        @register_kernel(
            "layernorm",
            "rmsnorm",
            name="flashinfer_fused_add_rmsnorm",
            solution="flashinfer",
            capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
            signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
            traits={
                "has_residual": frozenset({True}),
                "has_out": frozenset({False}),
            },
            priority=Priority.PERFORMANT + 1,
            warmup_behavior=WarmupBehavior.JIT_COMPILE,
        )
        def fused_add_rmsnorm(
            x: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
            residual: torch.Tensor | None,
            out: torch.Tensor | None,
            enable_pdl: bool | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if residual is None:
                raise ValueError("FlashInfer fused_add_rmsnorm requires a residual")
            if out is not None:
                raise ValueError("FlashInfer fused_add_rmsnorm does not accept out")
            _fused_add_rmsnorm(x, residual, weight, eps, enable_pdl=enable_pdl)
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
