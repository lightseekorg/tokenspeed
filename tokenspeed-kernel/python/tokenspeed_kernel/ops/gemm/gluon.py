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

"""Registration shim for Gluon GEMM kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm import (
    KIMI3_HIDDEN_SIZE,
    KIMI3_QKVFAB_SIZE,
    grouped_bf16_projection,
)
from tokenspeed_kernel.ops.gemm._gluon.gemv import gluon_simt_gemm_fp32
from tokenspeed_kernel.ops.gemm.triton import MXFP4_LINEAR_SIGNATURES
from tokenspeed_kernel.ops.quantization import quantize_fp8
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
from tokenspeed_kernel.workspace import zeroed_state

platform = current_platform()
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
_GFX950_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(9, 5),
    max_arch_version=ArchVersion(9, 5),
    vendors=frozenset({"amd"}),
)


# ===-----------------------------------------------------------------------===#
# BF16: Gluon (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        gluon_rowcta_bf16_gemv_gfx950 as _rowcta_bf16_gemv_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        launch_gluon_gated_rowcta_bf16_gemv_gfx950 as _gated_rowcta_bf16_gemv_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        supports_gluon_gated_rowcta_bf16_gemv_gfx950 as _supports_gated_rowcta,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        supports_gluon_rowcta_bf16_gemv_gfx950 as _supports_rowcta_bf16_gemv,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv import (
        launch_gluon_gemv_a16w16_gfx950 as _gemv_a16w16_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv import (
        prefers_gluon_gemv_a16w16_gfx950 as _prefers_gemv_a16w16,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv import (
        supports_gluon_gemv_a16w16_gfx950 as _supports_gemv_a16w16,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        launch_gluon_mm_a16w16_prefill_gfx950 as _mm_a16w16_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        supports_gluon_mm_a16w16_prefill_gfx950 as _supports_mm_a16w16_prefill,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        gluon_grouped_mm_a16w16_decode_num_counters_gfx950 as _grouped_decode_counters,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        gluon_mm_a16w16_decode_num_counters_gfx950 as _mm_a16w16_decode_counters,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        launch_gluon_bmm_a16w16_gfx950 as _bmm_a16w16_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        launch_gluon_grouped_mm_a16w16_decode_gfx950 as _grouped_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        launch_gluon_mm_a16w16_decode_gfx950 as _mm_a16w16_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        supports_gluon_grouped_mm_a16w16_decode_gfx950 as _supports_grouped_decode,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        supports_gluon_mm_a16w16_decode_gfx950 as _supports_mm_a16w16_decode,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp32.router_gemv import (
        launch_gluon_fp32_router_gemv_gfx950 as _fp32_router_gemv_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp32.router_gemv import (
        transpose_fp32_router_weight_gfx950 as _transpose_fp32_router_weight,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (
        launch_gluon_mm_mxfp8_decode_gfx950 as _mm_mxfp8_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
        launch_gluon_mm_mxfp8_gfx950 as _mm_mxfp8_impl,
    )

    @register_kernel(
        "gemm",
        "mv",
        name="gluon_rowcta_bf16_gemv_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "n_min": frozenset({128}),
            "k_min": frozenset({128}),
            "mnk_problem_filter": frozenset({_supports_rowcta_bf16_gemv}),
        },
    )
    def gluon_rowcta_bf16_gemv_gfx950(
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch single rows, and measured few-row shapes, to the gfx950 GEMV."""
        if out is None:
            out = torch.empty(
                (x.shape[0], weight.shape[0]),
                dtype=x.dtype,
                device=x.device,
            )
        return _rowcta_bf16_gemv_impl(x, weight, out)

    def _gated_rowcta_bf16_gemv_filter(m: int, n: int, k: int) -> bool:
        # Heads are 64 wide, so K fixes the head count.
        return _supports_gated_rowcta(n, k, k // 64, 64)

    @register_kernel(
        "gemm",
        "mv_weightless_rmsnorm_gated_sigmoid",
        name="gluon_gated_rowcta_bf16_gemv_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    gate=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "m": frozenset({1}),
            "mnk_problem_filter": frozenset({_gated_rowcta_bf16_gemv_filter}),
            "head_dim": frozenset({64}),
        },
    )
    def gluon_gated_rowcta_bf16_gemv_gfx950(
        x: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        num_heads: int,
        eps: float,
    ) -> torch.Tensor:
        """Dispatch one gated row of the measured shapes to the gfx950 GEMV."""
        return _gated_rowcta_bf16_gemv_impl(x, gate, weight, out, num_heads, eps)

    @register_kernel(
        "gemm",
        "prepared_fp32_router_gemv",
        name="gluon_fp32_router_gemv_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.float32),
                    weight=dense_tensor_format(torch.float32),
                )
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        weight_preprocessor=_transpose_fp32_router_weight,
    )
    def gluon_fp32_router_gemv_gfx950(
        x: torch.Tensor,
        weight: torch.Tensor,
        multiplier: float,
        counters: torch.Tensor,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """Dispatch decode-sized FP32 router GEMV to the gfx950 kernel."""
        # Split-K counters belong to the portable kernel; this one keeps no state.
        del counters
        return _fp32_router_gemv_impl(x, weight, multiplier, enable_pdl)

    try:
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.linear_attnres_partials_gfx950 import (
            launch_gluon_linear_attnres_partials_gfx950 as _linear_attnres_partials_impl,
        )
    except ImportError as exc:
        # Keep the message only: an exception object carries its traceback,
        # which pins every frame that was importing at the time.
        _IMPORT_ERROR_MESSAGE = str(exc)
        _linear_attnres_partials_impl = None
    else:
        _IMPORT_ERROR_MESSAGE = None

    _GFX950_CAPABILITY = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _DENSE16_SIGNATURES = frozenset(
        {
            format_signature(
                a=dense_tensor_format(torch.bfloat16),
                b=dense_tensor_format(torch.bfloat16),
            )
        }
    )

    def _validate_dense16_mm_arguments(
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        block_size: list[int] | None,
    ) -> None:
        if A_scales is not None or B_scales is not None:
            raise ValueError("dense16 Gluon MM does not accept quantization scales")
        if block_size is not None:
            raise ValueError("dense16 Gluon MM does not accept block_size")

    @register_kernel(
        "gemm",
        "mm",
        name="gluon_mm_a16w16_prefill_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=_DENSE16_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "mnk_problem_filter": frozenset({_supports_mm_a16w16_prefill}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "out_dtype": frozenset({torch.bfloat16}),
        },
    )
    def gluon_mm_a16w16_prefill_gfx950(
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
        """Dispatch measured Kimi K3 projection prefills to the large-M kernel."""
        _validate_dense16_mm_arguments(A_scales, B_scales, block_size)
        output = _mm_a16w16_prefill_impl(A, B, out_dtype, alpha=alpha, out=out)
        if output is None:
            raise RuntimeError("registered gfx950 prefill shape was rejected")
        return output

    @register_kernel(
        "gemm",
        "mv",
        name="gluon_mm_a16w16_decode_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={"mnk_problem_filter": frozenset({_supports_mm_a16w16_decode})},
    )
    def gluon_mm_a16w16_decode_gfx950(
        x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x @ weight.T`` for the measured K3 decode shapes (M >= 2).

        Args:
            x: ``[M, K]`` contiguous bf16 activation.
            weight: ``[N, K]`` contiguous bf16 weight.
            out: optional ``[M, N]`` destination.

        Returns:
            ``[M, N]`` output in ``x``'s dtype.
        """
        counters = zeroed_state(
            _mm_a16w16_decode_counters(x.shape[0], weight.shape[0], x.shape[1]),
            dtype=torch.int32,
            device=x.device,
        )
        return _mm_a16w16_decode_impl(x, weight, x.dtype, counters=counters, out=out)

    @register_kernel(
        "gemm",
        "dsv4_linear_fp32",
        name="gluon_dsv4_linear_fp32_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            # Shapes with measured decode GEMM configs.
            "n": frozenset({384}),
            "k": frozenset({5120}),
            "has_tokens": frozenset({True}),
            "hidden_inner_stride_one": frozenset({True}),
            "hidden_rank": frozenset({2}),
            "weight_contiguous": frozenset({True}),
        },
    )
    def gluon_dsv4_linear_fp32_gfx950(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """FP32 hidden_states @ weight.T for the measured decode row buckets.

        Fall back to hipBLASLt for other row counts.
        """
        if enable_pdl:
            raise ValueError("gfx950 DSV4 FP32 linear does not support PDL")
        m, n, k = hidden_states.shape[0], weight.shape[0], weight.shape[1]
        if not hidden_states.is_contiguous() or not _supports_mm_a16w16_decode(m, n, k):
            return torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
        counters = zeroed_state(
            _mm_a16w16_decode_counters(m, n, k),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        return _mm_a16w16_decode_impl(
            hidden_states, weight, torch.float32, counters=counters
        )

    @register_kernel(
        "gemm",
        "grouped_bf16_projection",
        name="gluon_grouped_bf16_projection_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            # Shapes with measured grouped decode GEMM configs (V4.1 wo_a TP4).
            "batch": frozenset({2}),
            "n": frozenset({1024}),
            "k": frozenset({4096}),
            "a_inner_stride_one": frozenset({True}),
            "b_contiguous": frozenset({True}),
            "is_cuda": frozenset({True}),
        },
    )
    def gluon_grouped_bf16_projection_gfx950(
        x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
    ) -> torch.Tensor:
        """One-launch grouped decode GEMM writing [tokens, groups, rows] directly.

        Fall back to torch.einsum for row counts without a measured config.
        """
        m, g, k = x.shape
        n = weight.shape[1]
        if not _supports_grouped_decode(m, g, n, k):
            result = torch.einsum("tgd,grd->tgr", x, weight)
            return result if out is None else out.copy_(result)
        counters = zeroed_state(
            _grouped_decode_counters(m, g, n, k), dtype=torch.int32, device=x.device
        )
        return _grouped_decode_impl(
            x, weight, counters=counters, quantize=False, out=out
        )

    @register_kernel(
        "gemm",
        "grouped_bf16_projection_fp8",
        name="gluon_grouped_bf16_projection_fp8_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch": frozenset({2}),
            "n": frozenset({1024}),
            "k": frozenset({4096}),
            "a_inner_stride_one": frozenset({True}),
            "b_contiguous": frozenset({True}),
            "is_cuda": frozenset({True}),
        },
    )
    def gluon_grouped_bf16_projection_fp8_gfx950(
        x: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grouped decode GEMM with a group-32 FP8 quantizing epilogue."""
        m, g, k = x.shape
        n = weight.shape[1]
        if not _supports_grouped_decode(m, g, n, k):
            projected = torch.einsum("tgd,grd->tgr", x, weight).flatten(1)
            return quantize_fp8(
                projected,
                granularity="token_group",
                group_size=32,
                scale_encoding="ue8m0",
                enable_pdl=False,
                override=None,
                solution=None,
            )
        counters = zeroed_state(
            _grouped_decode_counters(m, g, n, k), dtype=torch.int32, device=x.device
        )
        return _grouped_decode_impl(x, weight, counters=counters, quantize=True)

    def _gemv_a16w16_filter(m: int, n: int, k: int) -> bool:
        return _supports_gemv_a16w16(m, n, k) and _prefers_gemv_a16w16(m, n, k)

    # Ranked above the decode GEMM: the GEMV's measured buckets are the
    # shapes where it beat the default selection.
    @register_kernel(
        "gemm",
        "mv",
        name="gluon_gemv_a16w16_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        # Above gluon_rowcta_bf16_gemv_gfx950, which takes any single row:
        # this kernel claims only the shapes where it measured faster.
        priority=Priority.SPECIALIZED + 2,
        traits={"mnk_problem_filter": frozenset({_gemv_a16w16_filter})},
    )
    def gluon_gemv_a16w16_gfx950(
        x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x @ weight.T`` for decode batches (M <= 16), one program per CU.

        Selection takes it only for the shapes and row buckets where it
        measured faster than the default; a direct call serves any supported
        shape.

        Args:
            x: ``[M, K]`` fp16/bf16 activation with unit inner stride.
            weight: Contiguous ``[N, K]`` weight in ``x``'s dtype.
            out: Optional ``[M, N]`` row-major destination.

        Returns:
            ``[M, N]`` output in ``x``'s dtype.
        """
        if out is None:
            out = x.new_empty((x.shape[0], weight.shape[0]))
        return _gemv_a16w16_impl(x, weight, out)

    @register_kernel(
        "gemm",
        "bmm",
        name="gluon_bmm_a16w16_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=_DENSE16_SIGNATURES,
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

else:

    def gluon_mm_a16w16_prefill_gfx950(**kwargs):
        raise ImportError(
            "gluon_mm_a16w16_prefill_gfx950 requires tokenspeed-kernel-amd"
        )


# ===-----------------------------------------------------------------------===#
# MXFP8: Gluon (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (
        gluon_mm_mxfp8_decode_num_counters_gfx950 as _mm_mxfp8_decode_counters,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (
        launch_gluon_mm_mxfp8_decode_gfx950 as _mm_mxfp8_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import (
        supports_gluon_mm_mxfp8_decode_gfx950 as _supports_mxfp8_decode_problem,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
        is_fp32_blockscale_decode_problem as _is_fp32_blockscale_decode_problem,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
        launch_gluon_mm_fp8_blockscale_gfx950 as _mm_fp8_blockscale_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
        launch_gluon_mm_mxfp8_gfx950 as _mm_mxfp8_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm import (
        supports_mxfp8_gemm_shape as _supports_mxfp8_gemm_shape,
    )

    _MXFP8_SIGNATURES = frozenset(
        {
            format_signature(
                a=tensor_format("mxfp8", torch.float8_e4m3fn, scale=_MXFP8_UE8M0_SCALE),
                b=tensor_format("mxfp8", torch.float8_e4m3fn, scale=_MXFP8_UE8M0_SCALE),
            )
        }
    )
    _FP32_BLOCK_SIGNATURES = frozenset(
        {
            format_signature(
                a=tensor_format("mxfp8", _FP8_DTYPE, scale=_FP8_BLOCK_SCALE),
                b=tensor_format("mxfp8", _FP8_DTYPE, scale=_FP8_BLOCK_SCALE),
            )
        }
    )

    def _prefers_mxfp8_large_tiles(m: int, n: int, k: int) -> bool:
        # Large weights favor the matrix path from 64 rows onward.
        return m > 256 or (m >= 64 and n * k >= 1 << 26)

    def _is_mxfp8_prefill_problem(m: int, n: int, k: int) -> bool:
        return _supports_mxfp8_gemm_shape(m, n, k) and _prefers_mxfp8_large_tiles(
            m, n, k
        )

    def _is_mxfp8_decode_problem(m: int, n: int, k: int) -> bool:
        return _supports_mxfp8_decode_problem(
            m, n, k
        ) and not _prefers_mxfp8_large_tiles(m, n, k)

    def _is_fp32_blockscale_problem(m: int, n: int, k: int) -> bool:
        return _supports_mxfp8_gemm_shape(
            m, n, k
        ) or _is_fp32_blockscale_decode_problem(m, n, k)

    @register_kernel(
        "gemm",
        "mm",
        name="gluon_mm_mxfp8_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
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
        "mm",
        name="gluon_mm_fp8_blockscale_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=_FP32_BLOCK_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "mnk_problem_filter": frozenset({_is_fp32_blockscale_problem}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "block_scale_layout": frozenset({"canonical"}),
            "out_dtype": frozenset({torch.bfloat16}),
            "out_inner_stride_one": frozenset({True}),
        },
    )
    def gluon_mm_fp8_blockscale_gfx950(
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
        """Multiply canonical E4M3 matrices with exact FP32 K128 block scales."""
        return _mm_fp8_blockscale_impl(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            out=out,
            allocate_zeroed=zeroed_state,
        )

    @register_kernel(
        "gemm",
        "mm",
        name="gluon_mm_mxfp8_decode_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=_MXFP8_SIGNATURES,
        priority=Priority.SPECIALIZED + 1,
        traits={
            "mnk_problem_filter": frozenset({_is_mxfp8_decode_problem}),
            "a_inner_stride_one": frozenset({True}),
            "a_scales_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "b_scales_inner_stride_one": frozenset({True}),
            "block_scale_layout": frozenset({"canonical"}),
            "out_dtype": frozenset({torch.bfloat16, torch.float16}),
            "out_inner_stride_one": frozenset({True}),
        },
    )
    def gluon_mm_mxfp8_decode_gfx950(
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
        """Dispatch decode-sized true-MXFP8 GEMMs to the gfx950 kernel."""
        if A_scales is None or B_scales is None:
            raise ValueError("gfx950 MXFP8 decode GEMM requires both scale tensors")
        if block_size is None:
            raise ValueError("gfx950 MXFP8 decode GEMM requires block_size")
        counters = zeroed_state(
            _mm_mxfp8_decode_counters(A.shape[0], B.shape[0], A.shape[1]),
            dtype=torch.int32,
            device=A.device,
        )
        return _mm_mxfp8_decode_impl(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            counters=counters,
            out=out,
        )

else:

    def gluon_mm_mxfp8_gfx950(**kwargs):
        raise ImportError("gluon_mm_mxfp8_gfx950 requires tokenspeed-kernel-amd")

    def gluon_mm_fp8_blockscale_gfx950(**kwargs):
        raise ImportError(
            "gluon_mm_fp8_blockscale_gfx950 requires tokenspeed-kernel-amd"
        )

    def gluon_mm_mxfp8_decode_gfx950(**kwargs):
        raise ImportError("gluon_mm_mxfp8_decode_gfx950 requires tokenspeed-kernel-amd")


# ===-----------------------------------------------------------------------===#
# MXFP4 linear with dynamic MXFP4 activations: Gluon (AMD gfx950)
# ===-----------------------------------------------------------------------===#


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp4 import (
        gluon_mxfp4_linear_num_counters_gfx950 as _mxfp4_linear_counters,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp4 import (
        launch_gluon_mxfp4_linear_gfx950 as _mxfp4_linear_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp4 import (
        launch_gluon_mxfp4_swiglu_linear_gfx950 as _mxfp4_swiglu_linear_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp4 import (
        preshuffle_mxfp4_weight as _preshuffle_mxfp4_weight,
    )

    # Packed E2M1 weight in 16x16 MFMA-tile order (preshuffle_mxfp4_weight).
    MXFP4_LINEAR_MFMA_LAYOUT = "mfma_16x16"
    # The kernels address the weight with 32-bit buffer offsets.
    _MXFP4_LINEAR_MAX_WEIGHT_ELEMENTS = 2**31 - 1

    def prepare_mxfp4_linear_mfma_weight(weight: torch.Tensor) -> torch.Tensor:
        """Reorder a checkpoint ``[N, K/2]`` MXFP4 weight for the gfx950 kernels."""
        return _preshuffle_mxfp4_weight(weight)

    @register_kernel(
        "gemm",
        "mxfp4_linear",
        name="gluon_mxfp4_linear_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=MXFP4_LINEAR_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "n_align": frozenset({16}),
            "k_align": frozenset({64}),
            "weight_elements_max": frozenset({_MXFP4_LINEAR_MAX_WEIGHT_ELEMENTS}),
            "weight_layout": frozenset({MXFP4_LINEAR_MFMA_LAYOUT}),
        },
    )
    def gluon_mxfp4_linear_gfx950(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """``mxfp4(x) @ W.T`` with in-kernel activation quantization."""
        counters = zeroed_state(
            _mxfp4_linear_counters(x.shape[0], weight.shape[0], x.shape[1], False),
            dtype=torch.int32,
            device=x.device,
        )
        return _mxfp4_linear_impl(x, weight, weight_scale, counters=counters, out=out)

    @register_kernel(
        "gemm",
        "mxfp4_swiglu_linear",
        name="gluon_mxfp4_swiglu_linear_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
        signatures=MXFP4_LINEAR_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "n_align": frozenset({32}),
            "k_align": frozenset({64}),
            "weight_elements_max": frozenset({_MXFP4_LINEAR_MAX_WEIGHT_ELEMENTS}),
            "weight_layout": frozenset({MXFP4_LINEAR_MFMA_LAYOUT}),
        },
    )
    def gluon_mxfp4_swiglu_linear_gfx950(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """``silu(gate) * up`` of ``mxfp4(x) @ W.T`` fused into one GEMM."""
        counters = zeroed_state(
            _mxfp4_linear_counters(x.shape[0], weight.shape[0], x.shape[1], True),
            dtype=torch.int32,
            device=x.device,
        )
        return _mxfp4_swiglu_linear_impl(
            x, weight, weight_scale, counters=counters, out=out
        )

else:
    MXFP4_LINEAR_MFMA_LAYOUT = None

    def prepare_mxfp4_linear_mfma_weight(weight):
        raise ImportError(
            "prepare_mxfp4_linear_mfma_weight requires tokenspeed-kernel-amd"
        )

    def gluon_mxfp4_linear_gfx950(*args, **kwargs):
        raise ImportError("gluon_mxfp4_linear_gfx950 requires tokenspeed-kernel-amd")

    def gluon_mxfp4_swiglu_linear_gfx950(*args, **kwargs):
        raise ImportError(
            "gluon_mxfp4_swiglu_linear_gfx950 requires tokenspeed-kernel-amd"
        )


# ===-----------------------------------------------------------------------===#
# MXFP8 and FP8 block-scale: Gluon (AMD gfx1250)
# ===-----------------------------------------------------------------------===#


if platform.is_cdna5:
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.decode_mm import (
        launch_gluon_mm_fp8_blockscale_gfx1250 as _mm_fp8_blockscale_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.decode_mm import (
        launch_gluon_mm_mxfp8_ue8m0_gfx1250 as _mm_mxfp8_ue8m0_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.mm import (
        launch_gluon_mm_fp8_blockscale_largem_gfx1250 as _mm_fp8_blockscale_largem_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8.mm import (
        launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250 as _mm_mxfp8_ue8m0_largem_gfx1250_impl,
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
    _GFX1250_MXFP8_UE8M0_SIGNATURES = frozenset(
        {
            format_signature(
                a=tensor_format("mxfp8", _FP8_DTYPE, scale=_MXFP8_UE8M0_SCALE),
                b=tensor_format("mxfp8", _FP8_DTYPE, scale=_MXFP8_UE8M0_SCALE),
            )
        }
    )

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
        signatures=_GFX1250_MXFP8_UE8M0_SIGNATURES,
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
        name="gluon_mm_mxfp8_ue8m0_largem_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=_GFX1250_MXFP8_UE8M0_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "m_min": frozenset({17}),
            "n_align": frozenset({16}),
            "k_align": frozenset({32}),
            **_GFX1250_MXFP8_COMMON_TRAITS,
        },
    )
    def gluon_mm_mxfp8_ue8m0_largem_gfx1250(
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
        return _mm_mxfp8_ue8m0_largem_gfx1250_impl(
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
            "m": frozenset(range(1, 65)),
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

    @register_kernel(
        "gemm",
        "mm",
        name="gluon_mm_fp8_blockscale_largem_gfx1250",
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
            "m_min": frozenset({65}),
            "n_align": frozenset({128}),
            "k_align": frozenset({512}),
            **_GFX1250_MXFP8_COMMON_TRAITS,
        },
    )
    def gluon_mm_fp8_blockscale_largem_gfx1250(
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
        return _mm_fp8_blockscale_largem_gfx1250_impl(
            A,
            B,
            A_scales,
            B_scales,
            out_dtype,
            alpha=alpha,
            block_size=block_size,
            out=out,
        )

else:

    def gluon_mm_mxfp8_ue8m0_gfx1250(*args, **kwargs):
        raise RuntimeError("gluon_mm_mxfp8_ue8m0_gfx1250 requires CDNA5")

    def gluon_mm_mxfp8_ue8m0_largem_gfx1250(*args, **kwargs):
        raise RuntimeError("gluon_mm_mxfp8_ue8m0_largem_gfx1250 requires CDNA5")

    def gluon_mm_fp8_blockscale_gfx1250(*args, **kwargs):
        raise RuntimeError("gluon_mm_fp8_blockscale_gfx1250 requires CDNA5")

    def gluon_mm_fp8_blockscale_largem_gfx1250(*args, **kwargs):
        raise RuntimeError("gluon_mm_fp8_blockscale_largem_gfx1250 requires CDNA5")


# ===-----------------------------------------------------------------------===#
# BF16 dense projections: Gluon WMMA (AMD gfx1250)
# ===-----------------------------------------------------------------------===#


if platform.is_cdna5:
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        gluon_grouped_mm_a16w16_gfx1250 as _grouped_mm_a16w16_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        gluon_mm_a16w16_gfx1250 as _mm_a16w16_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        supports_gluon_grouped_mm_a16w16_fp8_gfx1250 as _supports_grouped_mm_a16w16_fp8_gfx1250,
    )

    @register_kernel(
        "gemm",
        "dsv4_linear_fp32",
        name="gluon_dsv4_linear_fp32_gfx1250",
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
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "n_align": frozenset({16}),
            "k_align": frozenset({128}),
            "has_tokens": frozenset({True}),
            "hidden_inner_stride_one": frozenset({True}),
            "hidden_rank": frozenset({2}),
            "weight_contiguous": frozenset({True}),
        },
    )
    def gluon_dsv4_linear_fp32_gfx1250(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """Project ``[M, K]`` BF16 hidden states to FP32 with CDNA5 WMMA."""
        if enable_pdl:
            raise ValueError("gfx1250 DSV4 FP32 linear does not support PDL")
        return _mm_a16w16_gfx1250_impl(hidden_states, weight, torch.float32)

    @register_kernel(
        "gemm",
        "grouped_bf16_projection",
        name="gluon_grouped_bf16_projection_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            # DeepSeek V4.1 wo_a: one group per rank at TP8, two at TP4.
            "batch": frozenset({1, 2}),
            "m_min": frozenset({1}),
            "m_max": frozenset({2048}),
            "n_align": frozenset({16}),
            "k_align": frozenset({128}),
            "a_inner_stride_one": frozenset({True}),
            "b_contiguous": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "is_cuda": frozenset({True}),
        },
    )
    def gluon_grouped_bf16_projection_gfx1250(
        x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
    ) -> torch.Tensor:
        """Project every group with CDNA5 WMMA; the API validates layouts."""
        return _grouped_mm_a16w16_gfx1250_impl(x, weight, out=out, quantize=False)

    @register_kernel(
        "gemm",
        "grouped_bf16_projection_fp8",
        name="gluon_grouped_bf16_projection_fp8_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "batch": frozenset({1, 2}),
            "n_align": frozenset({32}),
            "k_align": frozenset({128}),
            "a_inner_stride_one": frozenset({True}),
            "b_contiguous": frozenset({True}),
            "is_cuda": frozenset({True}),
        },
    )
    def gluon_grouped_bf16_projection_fp8_gfx1250(
        x: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grouped projection whose split-K reduction quantizes to group-32 FP8.

        Row counts whose tiles do not split K run the selected projection,
        then quantize.
        """
        m, g, k = x.shape
        n = weight.shape[1]
        if not _supports_grouped_mm_a16w16_fp8_gfx1250(m, g, n, k):
            projected = grouped_bf16_projection(x, weight, None, None)
            return quantize_fp8(
                projected.flatten(1),
                granularity="token_group",
                group_size=32,
                scale_encoding="ue8m0",
                enable_pdl=False,
                override=None,
                solution=None,
            )
        return _grouped_mm_a16w16_gfx1250_impl(x, weight, out=None, quantize=True)

else:

    def gluon_dsv4_linear_fp32_gfx1250(*args, **kwargs):
        raise RuntimeError("gluon_dsv4_linear_fp32_gfx1250 requires CDNA5")

    def gluon_grouped_bf16_projection_gfx1250(*args, **kwargs):
        raise RuntimeError("gluon_grouped_bf16_projection_gfx1250 requires CDNA5")

    def gluon_grouped_bf16_projection_fp8_gfx1250(*args, **kwargs):
        raise RuntimeError("gluon_grouped_bf16_projection_fp8_gfx1250 requires CDNA5")


# ===-----------------------------------------------------------------------===#
# Linear AttnRes partials: Gluon (AMD gfx950, gfx1250)
# ===-----------------------------------------------------------------------===#


if platform.is_cdna5:
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.linear_attnres_partials_gfx1250 import (
        launch_gluon_linear_attnres_partials_gfx1250 as _linear_attnres_partials_gfx1250_impl,
    )

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

    def gluon_linear_attnres_partials_gfx1250(**kwargs):
        raise RuntimeError("gluon_linear_attnres_partials_gfx1250 requires CDNA5")


if platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.linear_attnres_partials_gfx950 import (
        launch_gluon_linear_attnres_partials_gfx950 as _linear_attnres_partials_impl,
    )

    @register_kernel(
        "gemm",
        "linear_attnres_partials",
        name="gluon_linear_attnres_partials_gfx950",
        solution="gluon",
        capability=_GFX950_CAPABILITY,
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
        )


# ===-----------------------------------------------------------------------===#
# BF16 decode GEMV: Gluon WMMA (AMD gfx1250)
# ===-----------------------------------------------------------------------===#


def _gfx1250_wmma_dense_problem(m: int, n: int, k: int) -> bool:
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        use_gluon_wmma_dense_gfx1250,
    )

    return use_gluon_wmma_dense_gfx1250(m, k, n)


@register_kernel(
    "gemm",
    "mv",
    name="gluon_wmma_dense_gemv_gfx1250",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "m": frozenset(range(2, 65)),
        "mnk_problem_filter": frozenset({_gfx1250_wmma_dense_problem}),
    },
    priority=Priority.SPECIALIZED,
)
def gluon_wmma_dense_gemv_gfx1250(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for small-M decode activations on CDNA5.

    Args:
        x: ``[M, K]`` contiguous bf16 activation, K a multiple of 128.
        weight: ``[N, K]`` contiguous bf16 weight, N a multiple of 16.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        gluon_wmma_tdm_dense_gfx1250,
    )

    return gluon_wmma_tdm_dense_gfx1250(x, weight, out=out, split_k=None)


# ===-----------------------------------------------------------------------===#
# FP32 decode GEMM: Gluon SIMT (NVIDIA)
# ===-----------------------------------------------------------------------===#


# Registered where it beat Torch's FP32 path at every point of a GB200 module
# bench (CUDA graphs of 49 calls over cold weights, 291 points): BF16
# activations, 132 points with N 4 to 256 and K 512 to 8192 from 17 to 96
# rows, at 1.23x to 5.17x (median 2.72x). FP32 activations, and N 512 or 1024,
# were slower than Torch at some points in those rows (28 of 105 and 3 of 36)
# and are left to Torch. Against the BF16x3 split kernel on 256-row weights (K
# 3072 to 7168, separate runs), this kernel was faster at every K up to 48
# rows, while at 64 and 96 rows the split kernel was faster at some K from
# 4096, by up to 1.29x; the band is not cut there, since without a split this
# kernel is the only fast path. One step above the band start, so a broader
# FP32 kernel registered at the band start for the same rows does not take
# them.
register_kernel(
    "gemm",
    "mv",
    name="gluon_simt_gemm_fp32",
    solution="gluon",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.float32),
            )
        }
    ),
    traits={
        "m": frozenset(range(17, 97)),
        "n_align": frozenset({4}),
        "n_max": frozenset({256}),
        "k_align": frozenset({512}),
        "k_min": frozenset({512}),
        "k_max": frozenset({8192}),
    },
    priority=Priority.SPECIALIZED + 1,
)(gluon_simt_gemm_fp32)


__all__ = [
    "MXFP4_LINEAR_MFMA_LAYOUT",
    "gluon_mxfp4_linear_gfx950",
    "gluon_mxfp4_swiglu_linear_gfx950",
    "prepare_mxfp4_linear_mfma_weight",
    "gluon_mm_a16w16_prefill_gfx950",
    "gluon_mm_mxfp8_gfx950",
    "gluon_mm_fp8_blockscale_gfx950",
    "gluon_mm_mxfp8_decode_gfx950",
    "gluon_mm_fp8_blockscale_gfx1250",
    "gluon_mm_fp8_blockscale_largem_gfx1250",
    "gluon_mm_mxfp8_ue8m0_gfx1250",
    "gluon_mm_mxfp8_ue8m0_largem_gfx1250",
    "gluon_dsv4_linear_fp32_gfx1250",
    "gluon_grouped_bf16_projection_gfx1250",
    "gluon_grouped_bf16_projection_fp8_gfx1250",
    "gluon_linear_attnres_partials_gfx950",
    "gluon_linear_attnres_partials_gfx1250",
]
