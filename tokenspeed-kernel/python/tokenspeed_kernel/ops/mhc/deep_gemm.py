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

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.mhc.triton import _mhc_pre_impl
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

try:
    from tokenspeed_kernel.thirdparty.deep_gemm import (
        get_pdl,
        set_pdl,
        tf32_hc_prenorm_gemm,
    )
except Exception:
    tf32_hc_prenorm_gemm = None  # type: ignore[assignment]

try:
    from tokenspeed_kernel.thirdparty.cuda.mhc import mhc_big_fuse
except Exception:
    mhc_big_fuse = None  # type: ignore[assignment]


if tf32_hc_prenorm_gemm is not None:

    @register_kernel(
        "mhc",
        "pre",
        name="deep_gemm_mhc_pre",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
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
        priority=Priority.PERFORMANT,
        tags={"throughput"},
    )
    def deep_gemm_mhc_pre(
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run mHC pre-mapping with DeepGEMM prenorm and Triton mixing."""
        if get_pdl() != pdl_enabled():
            set_pdl(pdl_enabled())
        return _mhc_pre_impl(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            tf32_hc_prenorm_gemm,
            pre_reduce_apply_impl=mhc_big_fuse,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
