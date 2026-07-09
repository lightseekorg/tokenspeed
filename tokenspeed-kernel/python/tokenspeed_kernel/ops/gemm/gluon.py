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

"""Registration shim for AMD Gluon GEMM kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_dense16_impl = None

if current_platform().is_cdna4:
    try:
        from tokenspeed_kernel_amd.ops.gemm.mm_a16w16_gfx950 import (
            gluon_mm_a16w16_gfx950 as _dense16_impl,
        )
    except ImportError:
        _dense16_impl = None

if _dense16_impl is not None:

    @register_kernel(
        "gemm",
        "mm",
        name="gluon_mm_a16w16_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
            required_features=frozenset({"tensor_core:f16"}),
        ),
        signatures=format_signatures(
            ("a", "b"), "dense", {torch.float16, torch.bfloat16}
        ),
        traits={
            "n_align_128": frozenset({True}),
            "k_align_64": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_mm_a16w16_gfx950(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
    ):
        if A_scales is not None:
            raise ValueError("A_scales are not supported for dense16 Gluon GEMM")
        if B_scales is not None:
            raise ValueError("B_scales are not supported for dense16 Gluon GEMM")
        if block_size is not None:
            raise ValueError("block_size is not supported for dense16 Gluon GEMM")

        output = _dense16_impl(A, B, out_dtype, alpha=alpha)
        if output is not None:
            return output

        # TODO: Optimize M >= 256 and M <= 1024 dense16 cases in Gluon.
        output = F.linear(A, B)
        if alpha is not None:
            output = output * alpha.to(dtype=output.dtype)
        if output.dtype != out_dtype:
            output = output.to(out_dtype)
        return output
