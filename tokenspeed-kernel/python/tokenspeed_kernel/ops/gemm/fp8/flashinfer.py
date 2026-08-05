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
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    format_signature,
    format_signatures,
    tensor_format,
)

platform = current_platform()
_fp8_dtype = torch.float8_e4m3fn

_trtllm_per_token_group_quant_fp8 = error_fn
if platform.is_nvidia:
    from tokenspeed_kernel.ops.other.native.trtllm import (
        per_token_group_quant_8bit as _trtllm_per_token_group_quant_fp8,
    )

_MXFP8_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_SCALE
)

# ---- FlashInfer block-scaled FP8 ----------------------------------------

gemm_fp8_nt_groupwise = error_fn

if platform.is_hopper_plus:
    from flashinfer.gemm import gemm_fp8_nt_groupwise


def has_flashinfer_fp8_blockscale() -> bool:
    """Return whether the native FlashInfer FP8 block-scale GEMM is usable."""
    return (
        gemm_fp8_nt_groupwise is not error_fn
        and platform.arch_version == ArchVersion(10, 0)
    )


def prepare_flashinfer_fp8_blockscale_weight_scales(
    scales: torch.Tensor,
) -> torch.Tensor:
    """Pack canonical weight scales into FlashInfer's MN-major layout.

    Args:
        scales: Contiguous canonical scales shaped ``[N / 128, K / 128]``.

    Returns:
        A contiguous tensor shaped ``[K / 128, N / 128]``. This conversion is
        intended to run once after weight loading rather than in every GEMM.
    """
    if scales.ndim != 2:
        raise ValueError(f"weight scales must be 2-D, got shape {tuple(scales.shape)}")
    if scales.dtype != torch.float32:
        raise ValueError(
            "FlashInfer FP8 block-scale weight scales must use float32, "
            f"got {scales.dtype}"
        )
    return scales.transpose(0, 1).contiguous()


def swizzle_mxfp8_scale(scales: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Pack row-major MXFP8 scales into FlashInfer's F8_128x4 layout."""
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + 127) // 128
    scale_columns = K // 32
    padded = torch.zeros(
        (num_m_tiles * 128, num_k_tiles * 4),
        dtype=scales.dtype,
        device=scales.device,
    )
    padded[:M, :scale_columns] = scales
    return (
        padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
        .transpose(1, 3)
        .contiguous()
        .view(-1)
    )


@triton.jit
def _flashinfer_fp8_blockscale_quantize_prepacked_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    group_size,
    columns,
    valid_rows,
    padded_rows,
    bit8_min,
    bit8_max,
    BLOCK: tl.constexpr,
):
    group_id = tl.program_id(0)
    groups_per_row = columns // group_size
    row = group_id // groups_per_row
    scale_column = group_id % groups_per_row
    cols = tl.arange(0, BLOCK)
    col_mask = cols < group_size
    row_offset = row.to(tl.int64) * columns
    group_offset = scale_column.to(tl.int64) * group_size
    offsets = row_offset + group_offset + cols

    x = tl.load(x_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    scale_inv = tl.where(amax == 0.0, 1.0, bit8_max / amax)
    scale = 1.0 / scale_inv
    q = tl.clamp(x * scale_inv, bit8_min, bit8_max).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + offsets, q, mask=col_mask)
    tl.store(scale_ptr + scale_column.to(tl.int64) * padded_rows + row, scale)

    for pad_offset in tl.static_range(0, 3):
        pad_row = valid_rows + pad_offset
        pad_mask = (row == 0) & (pad_row < padded_rows)
        pad_offsets = pad_row.to(tl.int64) * columns + group_offset + cols
        tl.store(out_ptr + pad_offsets, 0.0, mask=pad_mask & col_mask)
        tl.store(
            scale_ptr + scale_column.to(tl.int64) * padded_rows + pad_row,
            1.0,
            mask=pad_mask,
        )


def flashinfer_fp8_blockscale_quantize_prepacked(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations into FlashInfer's native MN-major scale layout."""
    if not platform.is_nvidia:
        raise RuntimeError("FlashInfer FP8 block-scale quantization requires NVIDIA")
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.shape[0] <= 0:
        raise ValueError(f"x must contain at least one row, got {x.shape[0]}")
    if group_size != 128:
        raise ValueError(
            "FlashInfer FP8 block-scale prepacking requires group_size=128, "
            f"got {group_size}"
        )
    if x.shape[1] % group_size:
        raise ValueError(
            f"x.shape[1] must be divisible by {group_size}, got {x.shape[1]}"
        )

    valid_rows, columns = x.shape
    padded_rows = (valid_rows + 3) // 4 * 4
    if (
        padded_rows == valid_rows
        and x.dtype == torch.bfloat16
        and _trtllm_per_token_group_quant_fp8 is not error_fn
    ):
        q, scales = _trtllm_per_token_group_quant_fp8(x, group_size, False)
        expected_shape = (columns // group_size, valid_rows)
        if tuple(scales.shape) != expected_shape or not scales.is_contiguous():
            raise RuntimeError(
                "TRT-LLM FP8 quantizer returned unexpected prepared scales: "
                f"shape={tuple(scales.shape)}, stride={tuple(scales.stride())}, "
                f"expected contiguous {expected_shape}"
            )
        return q, scales

    q = torch.empty((padded_rows, columns), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (columns // group_size, padded_rows),
        device=x.device,
        dtype=torch.float32,
    )
    groups = valid_rows * (columns // group_size)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    _flashinfer_fp8_blockscale_quantize_prepacked_kernel[(groups,)](
        x,
        q,
        scales,
        group_size,
        columns,
        valid_rows,
        padded_rows,
        bit8_min=fp8_info.min,
        bit8_max=fp8_info.max,
        BLOCK=group_size,
        num_warps=1,
        num_stages=1,
    )
    return q, scales


def _prepare_flashinfer_fp8_blockscale_inputs(
    A: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    original_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert canonical values and scales to FlashInfer's native layout."""
    scale_m = A_scales.shape[0]
    if original_m % 4 != 0 or scale_m != original_m:
        padded_m = max(((original_m + 3) // 4) * 4, scale_m)
        padded_A = A.new_zeros((padded_m, A.shape[1]))
        padded_A[:original_m] = A
        A = padded_A

        if scale_m != padded_m:
            padded_A_scales = A_scales.new_ones((padded_m, A_scales.shape[1]))
            padded_A_scales[:scale_m] = A_scales
            A_scales = padded_A_scales

    return (
        A,
        A_scales.transpose(0, 1).contiguous(),
        B_scales.transpose(0, 1).contiguous(),
    )


def _validate_flashinfer_fp8_blockscale_prepacked(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    original_m: int,
    block_size: list[int] | None,
) -> None:
    """Validate the prepared-layout contract without modifying its inputs."""
    if block_size is not None and tuple(block_size) != (128, 128):
        raise ValueError(
            "prepacked FlashInfer scales require block_size=[128, 128], "
            f"got {block_size}"
        )
    if not 0 < original_m <= A.shape[0]:
        raise ValueError(f"original_m must be in [1, {A.shape[0]}], got {original_m}")
    if A.shape[0] % 4:
        raise ValueError(
            "prepacked FlashInfer activations must have an M dimension "
            f"divisible by four, got {A.shape[0]}"
        )

    expected_a_scales = (A.shape[1] // 128, A.shape[0])
    expected_b_scales = (B.shape[1] // 128, B.shape[0] // 128)
    if tuple(A_scales.shape) != expected_a_scales or not A_scales.is_contiguous():
        raise ValueError(
            "prepacked activation scales must be contiguous with shape "
            f"{expected_a_scales}, got shape={tuple(A_scales.shape)} "
            f"stride={tuple(A_scales.stride())}"
        )
    if tuple(B_scales.shape) != expected_b_scales or not B_scales.is_contiguous():
        raise ValueError(
            "prepacked weight scales must be contiguous with shape "
            f"{expected_b_scales}, got shape={tuple(B_scales.shape)} "
            f"stride={tuple(B_scales.stride())}"
        )


if gemm_fp8_nt_groupwise is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_fp8_blockscale",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_FORMAT_SIGNATURES,
        traits={
            "n_align_128": frozenset({True}),
            "k_align_128": frozenset({True}),
            "block_scale_layout": frozenset(
                {"canonical", "canonical_blackwell", "flashinfer_mn"}
            ),
        },
        priority=Priority.SPECIALIZED + 3,
        tags={"throughput"},
    )
    def flashinfer_mm_fp8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
        prepacked_scales: bool = False,
        original_m: int | None = None,
    ) -> torch.Tensor:
        """Run FlashInfer FP8 GEMM with canonical or prepared scales.

        Set ``prepacked_scales`` only when ``A_scales`` and ``B_scales`` already
        use FlashInfer's contiguous MN-major layout. The default canonical path
        performs the required padding and transposes for compatibility.
        """
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        assert B_scales is not None, "B_scales is required for FP8 blockscale GEMM"
        orig_m = A.shape[0] if original_m is None else int(original_m)
        if prepacked_scales:
            _validate_flashinfer_fp8_blockscale_prepacked(
                A,
                B,
                A_scales,
                B_scales,
                orig_m,
                block_size,
            )
            gemm_a_scales = A_scales
            gemm_b_scales = B_scales
        else:
            A, gemm_a_scales, gemm_b_scales = _prepare_flashinfer_fp8_blockscale_inputs(
                A, A_scales, B_scales, orig_m
            )

        output = gemm_fp8_nt_groupwise(
            A,
            B,
            gemm_a_scales,
            gemm_b_scales,
            scale_major_mode="MN",
            out_dtype=out_dtype,
        )
        output = output[:orig_m] if output.shape[0] != orig_m else output
        if out is not None:
            out.copy_(output)
            return out
        return output


# ---- FlashInfer MXFP8 (1,32) ue8m0, cute-dsl backend ---------------------

mm_mxfp8 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    from flashinfer.gemm import mm_mxfp8

_MXFP8_UE8M0_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_FLOAT_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_1X32_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=tensor_format("mxfp8", _fp8_dtype, scale=a_scale),
        b=tensor_format("mxfp8", _fp8_dtype, scale=_MXFP8_UE8M0_1X32_SCALE),
    )
    for a_scale in (_MXFP8_FLOAT_1X32_SCALE, _MXFP8_UE8M0_1X32_SCALE)
)


def has_flashinfer_mxfp8() -> bool:
    """Whether the flashinfer cute-dsl MXFP8 (1,32) GEMM is usable here.

    Returns:
        True when running on an NVIDIA Blackwell (SM10x) GPU with a
        flashinfer build that provides ``mm_mxfp8``.
    """
    return mm_mxfp8 is not error_fn


if mm_mxfp8 is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_mxfp8",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_1X32_FORMAT_SIGNATURES,
        traits={
            "k_align_32": frozenset({True}),
            "n_min_128": frozenset({True}),
            "k_min_128": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_mm_mxfp8(
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
        """MXFP8 (1,32)-block ue8m0 GEMM via flashinfer's cute-dsl backend.

        Args:
            A: ``[M, K]`` float8_e4m3fn activations.
            B: ``[N, K]`` (or ``[K, N]`` column-major) float8_e4m3fn weight.
            A_scales: uint8 e8m0 activation scales, either 1D in the
                F8_128x4 swizzled layout or ``[M, K // 32]`` row-major
                (re-swizzled per call; prefer pre-swizzled).
            B_scales: uint8 e8m0 weight scales, same layout options with
                ``[N, K // 32]`` row-major.
            out_dtype: Output dtype (bf16/fp16).
            alpha: Unused.
            block_size: Must be ``[1, 32]``.
            out: Optional output buffer.

        Returns:
            ``[M, N]`` tensor of ``out_dtype``.
        """
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        assert B_scales is not None, "B_scales is required for MXFP8 GEMM"
        assert block_size == [1, 32], f"expected block_size [1, 32], got {block_size}"
        k = A.shape[1]
        # B follows the dispatch convention of a [N, K] weight (row-major,
        # like the Triton kernel assumes); mm_mxfp8 wants the [K, N]
        # column-major view. Shape alone cannot disambiguate square weights,
        # so decide by memory layout.
        if B.shape[0] == k and B.stride(0) == 1:
            b = B
        else:
            b = B.t()
        n = b.shape[1]
        if k < 128 or k % 32 != 0 or n < 128:
            raise ValueError(
                f"flashinfer_mm_mxfp8 requires K >= 128, K % 32 == 0 and "
                f"N >= 128, got K={k}, N={n}"
            )
        if A_scales.dtype != torch.uint8 or B_scales.dtype != torch.uint8:
            raise ValueError(
                "flashinfer_mm_mxfp8 requires uint8 e8m0 scales, got "
                f"A_scales={A_scales.dtype}, B_scales={B_scales.dtype}"
            )
        if A_scales.dim() != 1:
            A_scales = swizzle_mxfp8_scale(A_scales.contiguous(), A.shape[0], k)
        if B_scales.dim() != 1:
            B_scales = swizzle_mxfp8_scale(B_scales.contiguous(), n, k)
        output = mm_mxfp8(
            A,
            b,
            A_scales,
            B_scales,
            out_dtype=out_dtype,
            backend="cute-dsl",
        )
        if out is not None:
            out.copy_(output)
            return out
        return output
