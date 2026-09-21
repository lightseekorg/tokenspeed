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
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

_FP8_DTYPE = torch.float8_e4m3fn
_MXFP8_UE8M0_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(1, 32),
)
_FP8_BLOCK_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        gluon_bmm_a16w16_gfx950 as _bmm_a16w16_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
        gluon_mm_mxfp8_gfx950 as _mm_mxfp8_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
        supports_mxfp8_gemm_shape as _supports_mxfp8_gemm_shape,
    )

    try:
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.linear_attnres_partials_gfx950 import (
            gluon_linear_attnres_partials_gfx950 as _linear_attnres_partials_impl,
        )
    except ImportError as exc:
        # Keep the message only: an exception object carries its traceback,
        # which pins every frame that was importing at the time.
        _IMPORT_ERROR_MESSAGE = str(exc)
        _linear_attnres_partials_impl = None
    else:
        _IMPORT_ERROR_MESSAGE = None

    _MXFP8_SIGNATURES = frozenset(
        {
            format_signature(
                a=tensor_format("mxfp8", torch.float8_e4m3fn, scale=_MXFP8_UE8M0_SCALE),
                b=tensor_format("mxfp8", torch.float8_e4m3fn, scale=_MXFP8_UE8M0_SCALE),
            )
        }
    )

    def _is_mxfp8_prefill_problem(m: int, n: int, k: int) -> bool:
        return (
            _supports_mxfp8_gemm_shape(m, n, k)
            and m >= 1024
            and n >= 1536
            and k >= 1024
        )

    @register_kernel(
        "gemm",
        "mm",
        name="gluon_mm_mxfp8_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=_MXFP8_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "mnk_problem_filter": frozenset({_is_mxfp8_prefill_problem}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "block_scale_layout": frozenset({"canonical"}),
            # GEMM format signatures currently describe input roles only.
            "out_dtype": frozenset({torch.bfloat16, torch.float16}),
        },
    )
    def gluon_mm_mxfp8_gfx950(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None,
        block_size: list[int] | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch the canonical true-MXFP8 layout to the gfx950 kernel."""
        if A_scales is None or B_scales is None:
            raise ValueError("gfx950 MXFP8 GEMM requires both scale tensors")
        if block_size is None:
            raise ValueError("gfx950 MXFP8 GEMM requires block_size")
        return _mm_mxfp8_impl(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            out=out,
        )

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
            "out_dtype": frozenset({torch.bfloat16}),
            "out_inner_stride_one": frozenset({True}),
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

    if current_platform().is_cdna5:
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.decode_mm import (
            gluon_mm_fp8_blockscale_gfx1250 as _mm_fp8_blockscale_gfx1250_impl,
        )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.decode_mm import (
            gluon_mm_mxfp8_ue8m0_gfx1250 as _mm_mxfp8_ue8m0_gfx1250_impl,
        )

        _GFX1250_MXFP8_COMMON_TRAITS = {
            "a_inner_stride_one": frozenset({True}),
            "a_scales_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "b_scales_inner_stride_one": frozenset({True}),
            "block_scale_layout": frozenset({"canonical"}),
            "out_dtype": frozenset({torch.bfloat16}),
            "out_inner_stride_one": frozenset({True}),
        }

        @register_kernel(
            "gemm",
            "mm",
            name="gluon_mm_mxfp8_ue8m0_gfx1250",
            solution="gluon",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(12, 5),
                max_arch_version=ArchVersion(12, 5),
                vendors=frozenset({"amd"}),
            ),
            signatures=frozenset(
                {
                    format_signature(
                        a=tensor_format(
                            "mxfp8",
                            _FP8_DTYPE,
                            scale=_MXFP8_UE8M0_SCALE,
                        ),
                        b=tensor_format(
                            "mxfp8",
                            _FP8_DTYPE,
                            scale=_MXFP8_UE8M0_SCALE,
                        ),
                    )
                }
            ),
            priority=Priority.SPECIALIZED,
            traits={
                "m": frozenset(range(1, 17)),
                "n_align": frozenset({16}),
                "k_align": frozenset({32}),
                "k_min": frozenset({256}),
                **_GFX1250_MXFP8_COMMON_TRAITS,
            },
        )
        def gluon_mm_mxfp8_ue8m0_gfx1250(
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
            return _mm_mxfp8_ue8m0_gfx1250_impl(
                A,
                B,
                A_scales,
                B_scales,
                out_dtype,
                alpha=alpha,
                block_size=block_size,
                out=out,
            )

        @register_kernel(
            "gemm",
            "mm",
            name="gluon_mm_fp8_blockscale_gfx1250",
            solution="gluon",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(12, 5),
                max_arch_version=ArchVersion(12, 5),
                vendors=frozenset({"amd"}),
            ),
            signatures=frozenset(
                {
                    format_signature(
                        a=tensor_format(
                            "mxfp8",
                            _FP8_DTYPE,
                            scale=_FP8_BLOCK_SCALE,
                        ),
                        b=tensor_format(
                            "mxfp8",
                            _FP8_DTYPE,
                            scale=_FP8_BLOCK_SCALE,
                        ),
                    )
                }
            ),
            priority=Priority.SPECIALIZED,
            traits={
                "m": frozenset(range(1, 17)),
                "n_align": frozenset({128}),
                "k_align": frozenset({128}),
                "k_min": frozenset({128}),
                **_GFX1250_MXFP8_COMMON_TRAITS,
            },
        )
        def gluon_mm_fp8_blockscale_gfx1250(
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
            return _mm_fp8_blockscale_gfx1250_impl(
                A,
                B,
                A_scales,
                B_scales,
                out_dtype,
                alpha=alpha,
                block_size=block_size,
                out=out,
            )

        try:
            from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.linear_attnres_partials_gfx1250 import (
                gluon_linear_attnres_partials_gfx1250 as _linear_attnres_partials_gfx1250_impl,
            )
        except ImportError:
            _linear_attnres_partials_gfx1250_impl = None
        if _linear_attnres_partials_gfx1250_impl is None:

            def gluon_linear_attnres_partials_gfx1250(**kwargs):
                raise RuntimeError(
                    "gluon_linear_attnres_partials_gfx1250 requires "
                    "tokenspeed-kernel-amd with the gfx1250 AttnRes kernel"
                )

        else:

            @register_kernel(
                "gemm",
                "linear_attnres_partials",
                name="gluon_linear_attnres_partials_gfx1250",
                solution="gluon",
                capability=CapabilityRequirement(
                    min_arch_version=ArchVersion(12, 5),
                    max_arch_version=ArchVersion(12, 5),
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
                    "tokens": frozenset({1}),
                    "input_size": frozenset({KIMI3_HIDDEN_SIZE}),
                    "output_size": frozenset({3648, KIMI3_QKVFAB_SIZE}),
                    "num_blocks": frozenset(range(1, 12)),
                    "inputs_contiguous": frozenset({True}),
                    "gfx1250_linear_attnres_enabled": frozenset({True}),
                },
            )
            def gluon_linear_attnres_partials_gfx1250(**kwargs):
                return _linear_attnres_partials_gfx1250_impl(**kwargs)

    else:

        def gluon_mm_mxfp8_ue8m0_gfx1250(*args, **kwargs):
            raise RuntimeError("gluon_mm_mxfp8_ue8m0_gfx1250 requires CDNA5")

        def gluon_mm_fp8_blockscale_gfx1250(*args, **kwargs):
            raise RuntimeError("gluon_mm_fp8_blockscale_gfx1250 requires CDNA5")

        def gluon_linear_attnres_partials_gfx1250(**kwargs):
            raise RuntimeError("gluon_linear_attnres_partials_gfx1250 requires CDNA5")

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
                "gluon_linear_attnres_partials_gfx950 requires "
                f"tokenspeed-kernel-amd: {_IMPORT_ERROR_MESSAGE}"
            )

else:

    def gluon_mm_mxfp8_gfx950(**kwargs):
        raise ImportError("gluon_mm_mxfp8_gfx950 requires tokenspeed-kernel-amd")

    def gluon_mm_mxfp8_ue8m0_gfx1250(*args, **kwargs):
        raise ImportError("gluon_mm_mxfp8_ue8m0_gfx1250 requires AMD CDNA5")

    def gluon_mm_fp8_blockscale_gfx1250(*args, **kwargs):
        raise ImportError("gluon_mm_fp8_blockscale_gfx1250 requires AMD CDNA5")

    def gluon_linear_attnres_partials_gfx950(**kwargs):
        raise ImportError(
            "gluon_linear_attnres_partials_gfx950 requires tokenspeed-kernel-amd"
        )

    def gluon_linear_attnres_partials_gfx1250(**kwargs):
        raise ImportError(
            "gluon_linear_attnres_partials_gfx1250 requires tokenspeed-kernel-amd"
        )


__all__ = [
    "gluon_mm_mxfp8_gfx950",
    "gluon_mm_fp8_blockscale_gfx1250",
    "gluon_mm_mxfp8_ue8m0_gfx1250",
    "gluon_linear_attnres_partials_gfx950",
    "gluon_linear_attnres_partials_gfx1250",
]
