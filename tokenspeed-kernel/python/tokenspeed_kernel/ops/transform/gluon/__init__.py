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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.transform import (
        launch_gluon_hadamard_transform_128_gfx950 as _hadamard_gfx950_impl,
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


__all__ = ["gluon_hadamard_transform_128_gfx950"]
