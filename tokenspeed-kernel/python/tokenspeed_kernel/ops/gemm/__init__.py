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
import tokenspeed_kernel.ops.gemm.gluon  # noqa: F401
import tokenspeed_kernel.ops.gemm.triton  # noqa: F401
import tokenspeed_kernel.ops.gemm.trtllm  # noqa: F401
import torch
from tokenspeed_kernel.platform import ArchVersion, Platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

logger = logging.getLogger(__name__)

__all__ = ["bmm", "mm"]

_platform = Platform.get()
_fp8_dtype = _platform.fp8e4m3fn.dtype

# Kernels that accept and own bias application inside their GEMM wrapper.
# For any kernel not listed here, dispatch applies the bias with a post-GEMM
# add instead of passing it to the kernel.
_KERNELS_WITH_FUSED_BIAS: frozenset[str] = frozenset(
    {
        "torch_bmm",
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


def _infer_scale_type(
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
) -> str | None:
    """For fp8, distinguish tensor/channel/scalar scaling."""
    if A_scales is None or B_scales is None:
        return None
    if A_scales.numel() == 1 and B_scales.numel() == 1:
        return "tensor"
    return "channel"


def _scale_storage_dtype(*scales: torch.Tensor | None) -> torch.dtype:
    for scale in scales:
        if scale is not None:
            return scale.dtype
    return torch.float32


def _gemm_format_signature(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    quant: str | None,
    block_size: list[int] | None,
):
    _ = out_dtype
    if quant == "mxfp8":
        if block_size is None:
            raise ValueError("mxfp8 format selection requires block_size")
        if B_scales is None:
            raise ValueError("mxfp8 format selection requires B_scales")
        a_scale = ScaleFormat(
            storage_dtype=(A_scales.dtype if A_scales is not None else torch.float32),
            granularity="block",
            block_shape=tuple(block_size),
        )
        b_scale = ScaleFormat(
            storage_dtype=B_scales.dtype,
            granularity="block",
            block_shape=tuple(block_size),
        )
        a_storage_dtype = _fp8_dtype if A_scales is None else A.dtype
        return format_signature(
            a=tensor_format("mxfp8", a_storage_dtype, scale=a_scale),
            b=tensor_format("mxfp8", B.dtype, scale=b_scale),
        )
    if quant == "fp8":
        scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales, B_scales),
            granularity=_infer_scale_type(A_scales, B_scales) or "unknown",
        )
        return format_signature(
            a=tensor_format("scaled-fp8", A.dtype, scale=scale),
            b=tensor_format("scaled-fp8", B.dtype, scale=scale),
        )
    if quant == "nvfp4":
        a_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales),
            granularity="block",
            block_shape=(16,),
        )
        b_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(B_scales),
            granularity="block",
            block_shape=(16,),
        )
        return format_signature(
            a=tensor_format("nvfp4", A.dtype, scale=a_scale),
            b=tensor_format("nvfp4", B.dtype, scale=b_scale),
        )
    if quant == "mxfp4":
        a_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales),
            granularity="block",
            block_shape=(32,),
        )
        b_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(B_scales),
            granularity="block",
            block_shape=(32,),
        )
        return format_signature(
            a=tensor_format("mxfp4", A.dtype, scale=a_scale),
            b=tensor_format("mxfp4", B.dtype, scale=b_scale),
        )
    return format_signature(
        a=dense_tensor_format(A.dtype), b=dense_tensor_format(B.dtype)
    )


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

    if kernel_name == "triton_mm_fp8_blockscale" and block_k == 32:
        from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

        return quantize_fp8_with_scale(
            A,
            granularity="token_group",
            group_size=block_k,
            scale_encoding="float32",
            solution="triton",
        )

    if (
        kernel_name in {"flashinfer_mm_fp8_blockscale", "triton_mm_fp8_blockscale"}
        and _platform.is_nvidia
        and _platform.arch_version == ArchVersion(12, 0)
    ):
        from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

        return quantize_fp8_with_scale(
            A,
            granularity="token_group",
            group_size=block_k,
            scale_encoding="float32",
            solution="triton",
        )

    def ensure_row_major_scales(
        qA: torch.Tensor,
        A_scales: torch.Tensor,
        *,
        group_major_scales: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # On NVIDIA, the TRT-LLM helper used by per_token_group_quant_fp8
        # returns [num_groups, num_tokens] scales. FlashInfer and Triton GEMMs
        # consume [num_tokens, num_groups].
        expected_groups = (qA.shape[-1] + block_k - 1) // block_k
        if group_major_scales:
            if A_scales.dim() != 2 or A_scales.shape[0] != expected_groups:
                raise ValueError(
                    "TRTLLM per-token-group quantization returned unexpected "
                    f"scale shape {tuple(A_scales.shape)} for "
                    f"tokens={qA.shape[0]}, groups={expected_groups}."
                )
            A_scales = A_scales.transpose(0, 1).contiguous()
            return qA, A_scales
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
            scale_ue8m0=_platform.is_blackwell_plus,
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
            ),
            group_major_scales=_platform.is_nvidia,
        )
    elif kernel_name == "triton_mm_fp8_blockscale":
        from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8

        return ensure_row_major_scales(
            *per_token_group_quant_fp8(A, block_k, column_major_scales=False),
            group_major_scales=_platform.is_nvidia,
        )
    else:
        raise ValueError(f"No online quantization defined for kernel {kernel_name!r}")


def _kernel_handles_online_mxfp8(kernel_name: str) -> bool:
    spec = KernelRegistry.get().get_by_name(kernel_name)
    return spec is not None and spec.solution == "reference"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _validate_gemm_out(
    out: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    op: str,
) -> None:
    if tuple(out.shape) != shape:
        raise ValueError(f"{op} out expects shape {shape}, got {tuple(out.shape)}")
    if out.dtype != dtype:
        raise ValueError(f"{op} out expects dtype {dtype}, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"{op} out expects device {device}, got {out.device}")
    if out.stride(-1) != 1:
        raise ValueError(f"{op} out must have stride(-1) == 1")


def mm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    quant: str | None = None,
    enable_pdl: bool = False,
    override: str | None = None,
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
        out: Optional output buffer. The output may be a strided view
            but must have contiguous rows (``stride(-1) == 1``).
        out_dtype: Output dtype (defaults to ``A.dtype``).
        alpha: Global scaling factor (nvfp4 only).
        block_size: Block size for block-wise quantization, e.g.
            ``[128, 128]``
        quant: Explicit quant type override.  One of ``"mxfp8"``,
            ``"fp8"``, ``"nvfp4"``, ``"mxfp4"``, ``"none"``.
            If ``None``, inferred from input dtypes and scales.
        enable_pdl: Whether to request Programmatic Dependent Launch support
            from kernels that accept it.
        override: Force selection of a specific kernel by name (e.g.
            ``"cublaslt_mm_nvfp4"``). Bypasses heuristic scoring.
    """
    out_dtype = out_dtype or (out.dtype if out is not None else A.dtype)

    M = A.shape[0]
    if quant == "mxfp4":
        K = A.shape[-1] * 2
        N = B.shape[0]
    else:
        K = A.shape[-1]
        N = B.shape[-1] if B.shape[0] == K else B.shape[0]

    if out is not None:
        _validate_gemm_out(
            out,
            shape=(M, N),
            dtype=out_dtype,
            device=A.device,
            op="mm",
        )

    traits: dict[str, object] = {
        "n_align_16": N % 16 == 0,
        "k_align_16": K % 16 == 0,
        "n_align_64": N % 64 == 0,
        "n_align_128": N % 128 == 0,
        "k_align_64": K % 64 == 0,
        "k_align_128": K % 128 == 0,
    }

    signature = _gemm_format_signature(
        A, B, A_scales, B_scales, out_dtype, quant, block_size
    )
    select_dtype = signature.storage_dtype_for("a") or A.dtype

    kernel = select_kernel(
        "gemm",
        "mm",
        signature,
        traits=traits,
        override=override,
    )

    # Online activation quantization
    if (
        quant == "mxfp8"
        and A_scales is None
        and not _kernel_handles_online_mxfp8(kernel.name)
    ):
        assert (
            block_size is not None
        ), "block_size is required for online activation quantization"
        A, A_scales = _online_quantize_mxfp8(A, block_size, kernel.name)

    kernel_args = (A, B, A_scales, B_scales, out_dtype)
    kernel_kwargs: dict[str, object] = {
        "alpha": alpha,
        "block_size": block_size,
    }
    if out is not None:
        kernel_kwargs["out"] = out

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
        has_out=out is not None,
    ):
        output = kernel(*kernel_args, **kernel_kwargs)

    if bias is not None and not fused_bias:
        if out is not None:
            output.add_(bias.to(dtype=output.dtype))
        else:
            output = output + bias.to(dtype=output.dtype)
    return output


def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    quant: str | None = None,
    enable_pdl: bool = False,
    override: str | None = None,
) -> torch.Tensor:
    """Batched matrix multiply with automatic kernel selection.

    Mirrors :func:`mm` for batched inputs with an outer batch dimension.
    ``A`` must use ``[B, M, K]`` layout and ``B`` must use ``[B, N, K]``
    layout. The result shape is ``[B, M, N]``.
    If ``out`` is provided, kernels that support direct output write into that
    buffer. The output may be a strided view but must have contiguous rows
    (``stride(-1) == 1``).

    Args:
        A: Activation batch ``[batch, M, K]``.
        B: Weight batch ``[batch, N, K]``.
        A_scales: Activation scales.
        B_scales: Weight scales (layout depends on quant type).
        bias: Optional bias vector of shape ``[N]`` or per-batch bias matrix
            of shape ``[batch, N]`` added to the output. When the selected
            kernel supports a fused bias epilogue (see
            ``_KERNELS_WITH_FUSED_BIAS``) it is passed into the kernel;
            otherwise it is added after the BMM.
        out: Optional output buffer. The output may be a strided view
            but must have contiguous rows (``stride(-1) == 1``).
        out_dtype: Output dtype (defaults to ``A.dtype``).
        alpha: Global scaling factor (nvfp4 only).
        block_size: Block size for block-wise quantization, e.g.
            ``[128, 128]``.
        quant: Explicit quant type override. One of ``"mxfp8"``,
            ``"fp8"``, ``"nvfp4"``, ``"mxfp4"``, ``"none"``.
            If ``None``, inferred from input dtypes and scales.
        enable_pdl: Whether to request Programmatic Dependent Launch support
            from kernels that accept it.
        override: Force selection of a specific kernel by name. Bypasses
            heuristic scoring.
    """
    out_dtype = out_dtype or (out.dtype if out is not None else A.dtype)
    if A.ndim != 3:
        raise ValueError(f"bmm expects A with shape [B, M, K], got {tuple(A.shape)}")
    if B.ndim != 3:
        raise ValueError(f"bmm expects B with shape [B, N, K], got {tuple(B.shape)}")

    batch, M, A_storage_K = A.shape
    B_batch, N, B_storage_K = B.shape
    if B_batch != batch:
        raise ValueError(f"bmm batch mismatch: A batch={batch}, B batch={B_batch}")
    if B_storage_K != A_storage_K:
        raise ValueError(f"bmm K mismatch: A K={A_storage_K}, B K={B_storage_K}")
    K = A_storage_K * 2 if quant == "mxfp4" else A_storage_K

    if out is not None:
        _validate_gemm_out(
            out,
            shape=(batch, M, N),
            dtype=out_dtype,
            device=A.device,
            op="bmm",
        )

    traits: dict[str, object] = {
        "n_align_16": N % 16 == 0,
        "k_align_16": K % 16 == 0,
        "n_align_64": N % 64 == 0,
        "n_align_128": N % 128 == 0,
        "k_align_64": K % 64 == 0,
        "k_align_128": K % 128 == 0,
    }

    signature = _gemm_format_signature(
        A, B, A_scales, B_scales, out_dtype, quant, block_size
    )
    select_dtype = signature.storage_dtype_for("a") or A.dtype

    kernel = select_kernel(
        "gemm",
        "bmm",
        signature,
        traits=traits,
        override=override,
    )

    if (
        quant == "mxfp8"
        and A_scales is None
        and not _kernel_handles_online_mxfp8(kernel.name)
    ):
        assert (
            block_size is not None
        ), "block_size is required for online activation quantization"
        A, A_scales = _online_quantize_mxfp8(A, block_size, kernel.name)

    kernel_args = (A, B, A_scales, B_scales, out_dtype)
    kernel_kwargs: dict[str, object] = {
        "alpha": alpha,
        "block_size": block_size,
    }
    if out is not None:
        kernel_kwargs["out"] = out

    fused_bias = bias is not None and kernel.name in _KERNELS_WITH_FUSED_BIAS
    if fused_bias:
        kernel_kwargs["bias"] = bias

    if kernel.name in _KERNELS_WITH_PDL:
        kernel_kwargs["enable_pdl"] = enable_pdl

    shape_params = {"B": batch, "M": M, "N": N, "K": K}
    ShapeCapture.get().record(
        "gemm",
        "bmm",
        kernel.name,
        select_dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "bmm",
        select_dtype,
        kernel_name=kernel.name,
        **shape_params,
        has_out=out is not None,
    ):
        output = kernel(*kernel_args, **kernel_kwargs)

    if bias is not None and not fused_bias:
        bias = bias.to(dtype=output.dtype)
        if bias.ndim == 1:
            bias_view = bias.view(1, 1, -1)
        elif bias.ndim == 2:
            bias_view = bias.view(bias.shape[0], 1, bias.shape[1])
        else:
            raise ValueError(f"bmm bias expects shape [N] or [B, N], got {bias.shape}")
        if out is not None:
            output.add_(bias_view)
        else:
            output = output + bias_view
    return output
