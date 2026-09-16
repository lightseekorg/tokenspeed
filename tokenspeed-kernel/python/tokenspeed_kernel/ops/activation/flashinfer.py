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

"""FlashInfer activation kernels."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import (
    Priority,
    WarmupBehavior,
    error_fn,
    register_kernel,
)
from tokenspeed_kernel.signature import format_signatures

silu_and_mul = error_fn

if current_platform().is_nvidia:
    try:
        from flashinfer import silu_and_mul as _silu_and_mul

        @register_kernel(
            "activation",
            "silu_and_mul",
            name="flashinfer_silu_and_mul",
            solution="flashinfer",
            capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
            signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
            traits={"has_limit": frozenset({False})},
            priority=Priority.PERFORMANT + 1,
            warmup_behavior=WarmupBehavior.JIT_COMPILE,
        )
        def silu_and_mul(
            x: torch.Tensor,
            out: torch.Tensor | None,
            enable_pdl: bool,
            limit: float | None,
        ) -> torch.Tensor:
            if limit is not None:
                raise ValueError("FlashInfer silu_and_mul does not support a limit")
            return _silu_and_mul(x, out, enable_pdl=enable_pdl)

    except ImportError:
        pass

__all__ = ["silu_and_mul"]
