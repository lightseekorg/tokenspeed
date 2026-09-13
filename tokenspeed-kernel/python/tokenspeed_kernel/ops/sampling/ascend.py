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

"""Registration shim for Ascend (NPU) sampling kernels.

``argmax`` is served by ``torch.argmax`` on Ascend NPU: the CuTe DSL and
Gluon solutions are NVIDIA/AMD-only, and the plain torch reduction is the
portable equivalent (same NaN-handling contract as the torch fallback).
"""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:

    @register_kernel(
        "sampling",
        "argmax",
        name="npu_argmax",
        solution="torch_npu",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            "logits",
            "dense",
            {torch.float16, torch.bfloat16, torch.float32},
        ),
        priority=Priority.PERFORMANT,
        tags={"portability"},
    )
    def npu_argmax(
        logits: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Row-wise argmax via ``torch.argmax`` on Ascend NPU."""
        result = torch.argmax(logits, dim=-1)
        if out is not None:
            out.copy_(result)
            return out
        return result


__all__ = ["npu_argmax"]