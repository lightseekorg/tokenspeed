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

"""Registration shims for AMD Gluon transform kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    try:
        from tokenspeed_kernel_amd.ops.gfx1250.transform import (
            launch_gluon_hadamard_transform_128_gfx1250 as _hadamard_128_gfx1250,
        )
    except ImportError:
        _hadamard_128_gfx1250 = None
else:
    _hadamard_128_gfx1250 = None


if _hadamard_128_gfx1250 is not None:

    @register_kernel(
        "transform",
        "hadamard_transform",
        name="gluon_hadamard_transform_128_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.bfloat16, torch.float16, torch.float32},
        ),
        traits={
            "last_dim": frozenset({128}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_hadamard_transform_128_gfx1250(
        x: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply a length-128 Sylvester Hadamard transform along the last dim."""
        return _hadamard_128_gfx1250(x, scale=scale)

    __all__ = ["gluon_hadamard_transform_128_gfx1250"]
else:
    __all__ = []
