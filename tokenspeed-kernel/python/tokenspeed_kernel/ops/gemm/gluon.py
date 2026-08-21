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
from tokenspeed_kernel.ops.gemm.kimi3 import KIMI3_HIDDEN_SIZE, KIMI3_QKVFAB_SIZE
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        gluon_bmm_a16w16_gfx950 as _bmm_a16w16_impl,
    )

    try:
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.linear_attnres_partials_gfx950 import (
            gluon_linear_attnres_partials_gfx950 as _linear_attnres_partials_impl,
        )
    except ImportError as exc:
        _IMPORT_ERROR = exc
        _linear_attnres_partials_impl = None
    else:
        _IMPORT_ERROR = None

    @register_kernel(
        "gemm",
        "bmm",
        name="gluon_bmm_a16w16_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    a=dense_tensor_format(torch.bfloat16),
                    b=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch": frozenset({12, 16}),
            "m": frozenset({1}),
            "n": frozenset({512}),
            "k": frozenset({128}),
            "a_inner_stride_one": frozenset({True}),
            "b_n_stride_one": frozenset({True}),
            "out_inner_stride_one": frozenset({True}),
            "out_dtype": frozenset({torch.bfloat16}),
        },
    )
    def gluon_bmm_a16w16_gfx950(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if A_scales is not None or B_scales is not None:
            raise ValueError("dense16 Gluon BMM does not accept quantization scales")
        if block_size is not None:
            raise ValueError("dense16 Gluon BMM does not accept block_size")

        output = _bmm_a16w16_impl(A, B, out_dtype, alpha=alpha, out=out)
        if output is not None:
            return output

        weight = B.transpose(1, 2)
        if out is not None and out_dtype == A.dtype:
            output = torch.bmm(A, weight, out=out)
        else:
            output = torch.bmm(A, weight)
            if output.dtype != out_dtype:
                output = output.to(out_dtype)
            if out is not None:
                out.copy_(output)
                output = out
        if alpha is not None:
            output.mul_(alpha.to(device=output.device, dtype=output.dtype))
        return output

    if _linear_attnres_partials_impl is not None:

        @register_kernel(
            "gemm",
            "linear_attnres_partials",
            name="gluon_linear_attnres_partials_gfx950",
            solution="gluon",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(9, 5),
                max_arch_version=ArchVersion(9, 5),
                vendors=frozenset({"amd"}),
            ),
            signatures=frozenset(
                {
                    format_signature(
                        hidden_states=dense_tensor_format(torch.bfloat16),
                        weight=dense_tensor_format(torch.bfloat16),
                        blocks=dense_tensor_format(torch.bfloat16),
                        score_weight_a=dense_tensor_format(torch.bfloat16),
                        score_weight_b=dense_tensor_format(torch.bfloat16),
                        out=dense_tensor_format(torch.bfloat16),
                    )
                }
            ),
            priority=Priority.SPECIALIZED,
            traits={
                "tokens": frozenset({1, 2, 4}),
                "input_size": frozenset({KIMI3_HIDDEN_SIZE}),
                "output_size": frozenset({3648, KIMI3_QKVFAB_SIZE}),
                "num_blocks": frozenset(range(1, 12)),
                "inputs_contiguous": frozenset({True}),
            },
        )
        def gluon_linear_attnres_partials_gfx950(**kwargs):
            return _linear_attnres_partials_impl(**kwargs)

    else:

        def gluon_linear_attnres_partials_gfx950(**kwargs):
            raise ImportError(
                "gluon_linear_attnres_partials_gfx950 requires tokenspeed-kernel-amd"
            ) from _IMPORT_ERROR

else:

    def gluon_linear_attnres_partials_gfx950(**kwargs):
        raise ImportError(
            "gluon_linear_attnres_partials_gfx950 requires tokenspeed-kernel-amd"
        )


__all__ = ["gluon_linear_attnres_partials_gfx950"]
