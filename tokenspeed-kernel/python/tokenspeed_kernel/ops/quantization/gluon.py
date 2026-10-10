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

"""Registration shims for AMD Gluon quantization kernels."""

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
    from tokenspeed_kernel_amd.ops.gfx950.quantization import (
        launch_gluon_quantize_fp8_group32_ue8m0_gfx950,
    )

    # Bit-exact with ``triton_quantize_fp8_group32_ue8m0`` (the V4.1 reference
    # rounding, which differs from generic MXFP8 quantizers).
    @register_kernel(
        "quantization",
        "fp8_with_scale",
        name="gluon_quantize_fp8_group32_ue8m0_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "granularity": frozenset({"token_group_32"}),
            "scale_encoding": frozenset({"ue8m0"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_quantize_fp8_group32_ue8m0_gfx950(
        x: torch.Tensor,
        granularity: str,
        group_size: int,
        scale_encoding: str,
        enable_pdl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize group-32 E4M3 data with UE8M0 scales, matching V4.1 act_quant.

        Args:
            x: [M,K] floating input with contiguous K divisible by 32.
            granularity: Must be ``token_group``.
            group_size: Must be 32.
            scale_encoding: Must be ``ue8m0``.
            enable_pdl: Must be False; this kernel has no PDL.

        Returns:
            FP8 [M,K] data and uint8 UE8M0 [M,K/32] scales.
        """
        if (
            granularity != "token_group"
            or group_size != 32
            or scale_encoding != "ue8m0"
        ):
            raise ValueError(
                "gfx950 FP8 quantization requires token_group, group_size=32, ue8m0"
            )
        if enable_pdl:
            raise ValueError("gfx950 FP8 group-32 quantization does not support PDL")
        return launch_gluon_quantize_fp8_group32_ue8m0_gfx950(x)
