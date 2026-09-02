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

"""Registration shim for the GFX950 mHC specialization."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.mhc.triton import (
    _mhc_pre_impl,
    _mhc_prenorm_gemm_triton,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.mhc import (
        gluon_mhc_pre_reduce_apply_gfx950 as _mhc_pre_reduce_apply_impl,
    )

    @register_kernel(
        "mhc",
        "pre",
        name="gluon_mhc_pre_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    residual=dense_tensor_format(torch.bfloat16),
                    fn=dense_tensor_format(torch.float32),
                    hc_scale=dense_tensor_format(torch.float32),
                    hc_base=dense_tensor_format(torch.float32),
                )
            }
        ),
        traits={
            "num_tokens": frozenset(range(1, 65)),
            "hc_mult": frozenset({4}),
            "hidden_size": frozenset({4096, 7168}),
            "sinkhorn_iters": frozenset({20}),
        },
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "latency"},
    )
    def gluon_mhc_pre_gfx950(
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run split-K prenorm with a GFX950 Gluon reduction/Sinkhorn stage."""
        return _mhc_pre_impl(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            _mhc_prenorm_gemm_triton,
            pre_reduce_apply_impl=_mhc_pre_reduce_apply_impl,
        )
