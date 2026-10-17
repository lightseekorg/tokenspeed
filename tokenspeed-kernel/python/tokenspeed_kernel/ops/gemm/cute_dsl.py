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
from tokenspeed_kernel.compile_monitor import is_serving
from tokenspeed_kernel.ops.gemm.flashinfer import cublas_fp8_gemm
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
)
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import MAX_M, ll_bf16_router
from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# ===-----------------------------------------------------------------------===#
# Per-tensor FP8 GEMV (M == 1): CuTe DSL skinny GEMM (NVIDIA SM100+)
# ===-----------------------------------------------------------------------===#


# Captures of an unwarmed shape fall back to the cuBLASLt FP8 GEMM.
if cublas_fp8_gemm is not error_fn:
    from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_tensor_scaled

    # 128 threads each load 16 FP8 weights (16 bytes) per K tile.
    _FP8_SKINNY_BLOCK, _FP8_SKINNY_VECTOR = 128, 16
    _fp8_skinny_warmed: set[tuple[int, int, int]] = set()

    @register_kernel(
        "gemm",
        "mm",
        name="cute_dsl_mm_fp8_tensor_scaled_m1",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("a", "b"),
            "scaled-fp8",
            {torch.float8_e4m3fn},
            scale=ScaleFormat(storage_dtype=torch.float32, granularity="tensor"),
        ),
        traits={
            "m": frozenset({1}),
            "n_align": frozenset({4}),
            "k_align": frozenset({_FP8_SKINNY_BLOCK * _FP8_SKINNY_VECTOR}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({False}),
            "out_dtype": frozenset({torch.bfloat16}),
        },
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_mm_fp8_tensor_scaled_m1(
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
        """Per-tensor scaled FP8 GEMV for one decode row on CUDA cores.

        At M == 1 the weight read is the whole cost, and this streams it at a
        higher fraction of HBM bandwidth than cuBLASLt's FP8 tiles.

        Args:
            A: ``[1, K]`` row-major FP8 activations.
            B: ``[K, N]`` column-major FP8 weights (a transposed ``[N, K]``).
            A_scales: One-element FP32 activation dequant scale.
            B_scales: One-element FP32 weight dequant scale.
            out_dtype: BF16.
            alpha: Must be None; the per-tensor scales carry the dequant.
            block_size: Must be None; the scales are per tensor.
            out: Optional ``[1, N]`` output buffer.

        Returns:
            ``[1, N]`` BF16 output, ``out`` when given.
        """
        if alpha is not None or block_size is not None:
            raise ValueError("per-tensor FP8 GEMM takes no alpha or block_size")
        n, k = B.shape[1], A.shape[1]
        key = (A.device.index or 0, n, k)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in _fp8_skinny_warmed:
            # Compiling inside a capture is unsafe; eager warmup compiles first.
            return flashinfer_mm_fp8_tensor_scaled(
                A, B, A_scales, B_scales, out_dtype, out=out
            )
        config = SkinnyGemmConfig(
            1, _FP8_SKINNY_BLOCK, 2 if k >= 8192 else 4, 1, _FP8_SKINNY_VECTOR
        )
        direct = out is not None and out.is_contiguous()
        result = shape_dynamic_skinny_gemm(
            A.contiguous(),
            # A padded leading dimension needs the dense copy cuBLASLt also makes.
            B.t().contiguous(),
            config,
            out=out if direct else None,
            scales=(A_scales.reshape(1), B_scales.reshape(1)),
        )
        if not capturing:
            _fp8_skinny_warmed.add(key)
        if out is None or direct:
            return result
        return out.copy_(result)


# ===-----------------------------------------------------------------------===#
# BF16 router GEMM: CuTe DSL low-latency (NVIDIA SM90+)
# ===-----------------------------------------------------------------------===#


# Registry entry for the vendored CuTe low-latency BF16 router GEMM.

_BF16_IN_FP32_OUT = {
    format_signature(
        hidden_states=dense_tensor_format(torch.bfloat16),
        weight=dense_tensor_format(torch.bfloat16),
    )
}


@register_kernel(
    "gemm",
    "router_projection",
    name="cute_dsl_ll_bf16_router",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        # Split-K reduces through DSMEM inside a thread block cluster.
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=_BF16_IN_FP32_OUT,
    priority=Priority.SPECIALIZED,
    traits={"out_dtype": frozenset({torch.float32})},
)
def cute_dsl_ll_bf16_router(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Router logits in FP32 from BF16 operands, for decode-sized M.

    Args:
        hidden_states: ``[M, K]`` contiguous BF16 activation.
        weight: ``[N, K]`` contiguous BF16 router weight.
        out: Optional ``[M, N]`` FP32 destination; allocated when omitted.

    Returns:
        ``[M, N]`` FP32 router logits, ``out`` when it was given.
    """
    return ll_bf16_router(hidden_states, weight, out, enable_pdl=pdl_enabled())


def ll_bf16_router_supported(
    hidden_states: torch.Tensor, weight: torch.Tensor, m: int
) -> bool:
    """Whether the vendored driver can serve this call.

    Once serving, it declines the calls the dot-product kernel would take: that
    kernel compiles once per exact M, so the caller's other GEMM runs them.
    This guard also applies to DSV4 projection selection; startup CUDA graphs
    retain their captured kernels when serving begins.

    Args:
        hidden_states: ``[M, K]`` activation; weight: ``[N, K]`` weight.
        m: Token count, which selects the dot-product or split-K backend.

    Returns:
        True when a vendored kernel is compilable and applicable here.
    """
    return ll_bf16_router.supports(hidden_states, weight, m) and not (
        is_serving() and ll_bf16_router.uses_dotprod(m, hidden_states.shape[1])
    )


@register_kernel(
    "gemm",
    "dsv4_linear_fp32",
    name="cute_dsl_dsv4_linear_fp32",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=_BF16_IN_FP32_OUT,
    priority=Priority.SPECIALIZED + 1,
    traits={
        "ll_bf16_supported": frozenset({True}),
    },
)
def cute_dsl_dsv4_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Project DSV4 inputs with the shared low-latency BF16 router driver.

    Args:
        hidden_states: Contiguous BF16 ``[M, K]`` activations, up to ``MAX_M``
            rows.
        weight: Contiguous BF16 ``[N, K]`` projection weights.
        enable_pdl: PDL policy used for compilation and launch.

    Returns:
        FP32 router logits, accumulated and reduced in FP32.
    """
    return ll_bf16_router(hidden_states, weight, None, enable_pdl=enable_pdl)


__all__ = [
    "MAX_M",
    "cute_dsl_dsv4_linear_fp32",
    "cute_dsl_ll_bf16_router",
    "ll_bf16_router_supported",
]
