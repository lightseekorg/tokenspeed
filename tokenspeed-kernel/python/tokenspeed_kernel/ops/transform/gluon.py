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

"""Registration shim for AMD Gluon transforms."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.transform.triton import triton_hadamard_transform_128
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# The portable one-row kernel is faster than the gfx1250 WMMA tiles below this.
_HADAMARD_GFX1250_MIN_ROWS = 256

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.transform.hadamard import (
        launch_gluon_hadamard_transform_128_gfx950 as _hadamard_gfx950_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.transform import (
        launch_gluon_hadamard_transform_128_gfx1250 as _hadamard_gfx1250_impl,
    )

    @register_kernel(
        "transform",
        "hadamard_transform",
        name="gluon_hadamard_transform_128_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "last_dim": frozenset({128}),
            "contiguous": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_hadamard_transform_128_gfx950(
        x: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Apply a GFX950 BF16 length-128 butterfly transform."""
        return _hadamard_gfx950_impl(x, scale=scale)

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
        scale: float,
    ) -> torch.Tensor:
        """Apply a GFX1250 length-128 Sylvester Hadamard transform."""
        if x.numel() < _HADAMARD_GFX1250_MIN_ROWS * 128:
            return triton_hadamard_transform_128(x, scale=scale)
        return _hadamard_gfx1250_impl(x, scale=scale)


__all__ = [
    "gluon_hadamard_transform_128_gfx950",
    "gluon_hadamard_transform_128_gfx1250",
]
