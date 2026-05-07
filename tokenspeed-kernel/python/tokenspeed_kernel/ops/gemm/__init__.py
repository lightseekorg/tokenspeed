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

import logging

# Backend registration (side-effect imports)
import tokenspeed_kernel.numerics.reference.gemm  # noqa: F401
import tokenspeed_kernel.ops.gemm.deep_gemm  # noqa: F401
import tokenspeed_kernel.ops.gemm.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.gemm.triton  # noqa: F401
import tokenspeed_kernel.ops.gemm.trtllm  # noqa: F401
import torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import (
    SelectedKernel,
    SelectionObjective,
    select_kernel,
)

logger = logging.getLogger(__name__)

__all__ = ["mm", "select_mm_kernel"]

_platform = Platform.get()
_fp8_dtype = _platform.fp8e4m3fn.dtype

# Kernels that natively fuse a bias-vector add inside their GEMM kernel.
# For any kernel not listed here, ``mm`` applies the bias with a post-GEMM
# add instead of passing it to the kernel.
_KERNELS_WITH_FUSED_BIAS: frozenset[str] = frozenset(
    {
        "torch_mm",
        "triton_mm_fp8_scaled",
    }
)

# Kernels that accept an ``enable_pdl`` kwarg for Programmatic Dependent Launch.
_KERNELS_WITH_PDL: frozenset[str] = frozenset(
    {
        "flashinfer_mm_nvfp4",
    }
)

# Kernels that accept BF16 activations for MXFP8 GEMM and perform activation
# quantization internally, so ``mm`` must not pre-quantize A or pass A_scales.
#
# This is intentionally a narrow allowlist while SelectedKernel only exposes a
# kernel name.  If more kernels need this capability, promote it to a registry
# trait instead of growing backend-specific branching in ``mm``.
_KERNELS_WITH_INTERNAL_MXFP8_ACTIVATION_QUANT: frozenset[str] = frozenset(
    {
        "flashinfer_sm90_mm_fp8_blockscale",
    }
)


def _infer_scale_type(
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
) -> str | None:
    """For fp8, distinguish per-tensor from per-channel scaling."""
    if A_scales is None or B_scales is None:
        return None
    if A_scales.numel() == 1 and B_scales.numel() == 1:
        return "per_tensor"
    return "per_channel"


def _online_quantize_mxfp8(
    A: torch.Tensor,
    block_size: list[int],
    kernel_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform online activation quantization for mxfp8 block-scaled GEMM.

    The quantization approach is chosen based on the selected kernel's
    name because different backends require different scale layouts.
    """
    block_k = block_size[1]

    def ensure_row_major_scales(
        qA: torch.Tensor,
        A_scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # On NVIDIA, the TRT-LLM helper used by per_token_group_quant_fp8
        # returns [num_groups, num_tokens] scales. FlashInfer and Triton GEMMs
        # consume [num_tokens, num_groups].
        expected_groups = (qA.shape[-1] + block_k - 1) // block_k
        if (
            A_scales.shape[-1] != expected_groups
            and A_scales.shape[0] == expected_groups
        ):
            A_scales = A_scales.transpose(0, 1).contiguous()
        return qA, A_scales

    if kernel_name == "deep_gemm_mm_fp8_blockscale":
        from tokenspeed_kernel.ops.gemm.fp8_utils import (
            per_token_group_quant_fp8,
        )

        return per_token_group_quant_fp8(
            A,
            block_k,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
    elif kernel_name == "flashinfer_mm_fp8_blockscale":
        from tokenspeed_kernel.ops.gemm.fp8_utils import (
            per_token_group_quant_fp8,
        )

        return ensure_row_major_scales(
            *per_token_group_quant_fp8(
                A,
                block_k,
                column_major_scales=False,
            )
        )
    elif kernel_name == "triton_mm_fp8_blockscale":
        from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8

        return ensure_row_major_scales(
            *per_token_group_quant_fp8(A, block_k, column_major_scales=False)
        )
    else:
        raise ValueError(f"No online quantization defined for kernel {kernel_name!r}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _infer_select_dtype(A_dtype: torch.dtype, quant: str | None) -> torch.dtype:
    if quant in ("mxfp8", "fp8"):
        return _fp8_dtype
    return A_dtype


def _build_mm_traits(
    *,
    K: int,
    N: int,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    quant: str | None = None,
) -> dict[str, object]:
    traits: dict[str, object] = {
        "n_align_16": N % 16 == 0,
        "k_align_16": K % 16 == 0,
        "n_align_64": N % 64 == 0,
        "n_align_128": N % 128 == 0,
        "k_align_128": K % 128 == 0,
    }

    if quant is not None:
        traits["quant"] = quant

    if quant == "fp8":
        scale_type = _infer_scale_type(A_scales, B_scales)
        if scale_type is not None:
            traits["scale_type"] = scale_type

    return traits


def select_mm_kernel(
    *,
    A_dtype: torch.dtype,
    K: int,
    N: int,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    quant: str | None = None,
    objective: SelectionObjective = SelectionObjective.DEFAULT,
    override: str | None = None,
    expected_kernel_name: str | None = None,
) -> SelectedKernel:
    """Select the GEMM kernel for a stable Linear shape.

    ``mm`` still records per-call shapes and handles online activation
    quantization. This helper only moves the selection lookup out of hot
    forward paths whose dtype and K/N dimensions are fixed after weight load.
    """
    select_dtype = _infer_select_dtype(A_dtype, quant)
    traits = _build_mm_traits(
        K=K,
        N=N,
        A_scales=A_scales,
        B_scales=B_scales,
        quant=quant,
    )
    return select_kernel(
        "gemm",
        "mm",
        select_dtype,
        traits=traits,
        objective=objective,
        override=override,
        expected_kernel_name=expected_kernel_name,
    )


def mm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    quant: str | None = None,
    enable_pdl: bool = False,
    override: str | None = None,
    expected_kernel_name: str | None = None,
    selected_kernel: SelectedKernel | None = None,
) -> torch.Tensor:
    """Dense matrix multiply with automatic kernel selection.

    Quantization type is inferred from input dtype and the presence of
    scales, or can be set explicitly via ``quant``.  When ``A_scales``
    is ``None`` for a quantized mode (e.g. ``quant="mxfp8"``), online
    activation quantization is performed here before calling the kernel.

    Args:
        A: Activation matrix ``[M, K]``.
        B: Weight matrix.
        A_scales: Activation scales.
        B_scales: Weight scales (layout depends on quant type).
        bias: Optional bias vector of shape ``[N]`` added to the
            output.  When the selected kernel supports a fused bias
            epilogue (see ``_KERNELS_WITH_FUSED_BIAS``) it is passed
            into the kernel; otherwise it is added after the GEMM.
        out_dtype: Output dtype (defaults to ``A.dtype``).
        alpha: Global scaling factor (nvfp4 only).
        block_size: Block size for block-wise quantization, e.g.
            ``[128, 128]``
        quant: Explicit quant type override.  One of ``"mxfp8"``,
            ``"fp8"``, ``"nvfp4"``, ``"none"``.
            If ``None``, inferred from input dtypes and scales.
        override: Force selection of a specific kernel by name (e.g.
            ``"cublaslt_mm_nvfp4"``). Bypasses heuristic scoring.
        expected_kernel_name: Debug hint for expected kernel selection.
    """
    out_dtype = out_dtype or A.dtype

    K = A.shape[-1]
    M = A.shape[0]
    N = B.shape[-1] if B.shape[0] == K else B.shape[0]

    if selected_kernel is not None:
        kernel = selected_kernel
        select_dtype = _infer_select_dtype(A.dtype, quant)
    else:
        kernel = select_mm_kernel(
            A_dtype=A.dtype,
            K=K,
            N=N,
            A_scales=A_scales,
            B_scales=B_scales,
            quant=quant,
            override=override,
            expected_kernel_name=expected_kernel_name,
        )
        select_dtype = _infer_select_dtype(A.dtype, quant)

    # Online activation quantization
    if (
        quant == "mxfp8"
        and A_scales is None
        and kernel.name not in _KERNELS_WITH_INTERNAL_MXFP8_ACTIVATION_QUANT
    ):
        assert (
            block_size is not None
        ), "block_size is required for online activation quantization"
        A, A_scales = _online_quantize_mxfp8(A, block_size, kernel.name)

    kernel_args = (A, B, A_scales, B_scales, out_dtype)
    kernel_kwargs: dict[str, object] = {"alpha": alpha, "block_size": block_size}

    fused_bias = bias is not None and kernel.name in _KERNELS_WITH_FUSED_BIAS
    if fused_bias:
        kernel_kwargs["bias"] = bias

    if kernel.name in _KERNELS_WITH_PDL:
        kernel_kwargs["enable_pdl"] = enable_pdl

    shape_params = {"M": M, "N": N, "K": K}
    ShapeCapture.get().record(
        "gemm",
        "mm",
        kernel.name,
        select_dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "mm",
        select_dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        output = kernel(*kernel_args, **kernel_kwargs)

    if bias is not None and not fused_bias:
        output = output + bias.to(dtype=output.dtype)
    return output
