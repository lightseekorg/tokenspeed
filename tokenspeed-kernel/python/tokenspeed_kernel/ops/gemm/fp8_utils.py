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

from typing import Tuple

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import error_fn

_is_amd = Platform.get().is_amd
_is_nvidia = Platform.get().is_nvidia
platform = Platform.get()
fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = torch.finfo(fp8_dtype).min

_trtllm_per_token_group_quant_fp8 = error_fn

if _is_nvidia:
    from tokenspeed_kernel.ops.quantization.flashinfer import (
        fp8_blockscale_quantize_runner_sm90 as _flashinfer_fp8_blockscale_quantize_runner_sm90,
    )
    from tokenspeed_kernel.thirdparty.trtllm import (
        per_token_group_quant_8bit as _trtllm_per_token_group_quant_fp8,
    )
    from tokenspeed_kernel.thirdparty.trtllm import (
        per_token_quant_fp8 as _trtllm_per_token_quant_fp8,
    )


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Re-layout row-major MXFP8 (1,32) block scales into the F8_128x4
    swizzled layout consumed by flashinfer's block-scaled GEMMs.

    Args:
        sf: ``[M, K // 32]`` uint8 e8m0 scales, row-major.
        M: Number of rows of the scaled tensor.
        K: Number of columns of the scaled tensor (multiple of 32).

    Returns:
        1D uint8 tensor of ``round_up(M, 128) * round_up(K // 32, 4)``
        elements in the 128x4 tile layout (rows padded with zeros).
    """
    num_m_tiles = ceil_div(M, 128)
    num_k_tiles = ceil_div(K, 128)

    scale_cols = K // 32
    sf_padded = torch.zeros(
        (num_m_tiles * 128, num_k_tiles * 4), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_tiled = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    return sf_tiled.transpose(1, 3).contiguous().view(-1)


@triton.jit
def _per_token_group_quant_8bit(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_8bit_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id.to(tl.int64) * group_size
    y_q_ptr += g_id.to(tl.int64) * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    if SCALE_UE8M0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_8bit_padded_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    valid_rows,
    padded_rows,
    bit8_min,
    bit8_max,
    BLOCK: tl.constexpr,
):
    """Quantize valid rows and initialize the GEMM's M-padding in one launch."""
    group_id = tl.program_id(0)
    groups_per_row = y_num_columns // group_size
    row = group_id // groups_per_row
    scale_col = group_id % groups_per_row

    cols = tl.arange(0, BLOCK)
    col_mask = cols < group_size
    row_offset = row.to(tl.int64) * y_num_columns
    group_offset = scale_col.to(tl.int64) * group_size
    offsets = row_offset + group_offset + cols

    y = tl.load(y_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(y))
    # Match TRT-LLM's scale_1x128_kernel: an all-zero group uses a neutral
    # scale of one, while every other group uses amax / FP8_MAX.
    y_s_inv = tl.where(amax == 0.0, 1.0, bit8_max / amax)
    y_s = 1.0 / y_s_inv
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + offsets, y_q, mask=col_mask)
    tl.store(
        y_s_ptr + scale_col.to(tl.int64) * padded_rows + row,
        y_s,
    )

    # Only the first valid row's programs initialize the at-most-three tail
    # rows. Each program owns one 128-column group, so these stores do not race.
    for pad_offset in tl.static_range(0, 3):
        pad_row = valid_rows + pad_offset
        pad_mask = (row == 0) & (pad_row < padded_rows)
        pad_offsets = pad_row.to(tl.int64) * y_num_columns + group_offset + cols
        tl.store(y_q_ptr + pad_offsets, 0.0, mask=pad_mask & col_mask)
        tl.store(
            y_s_ptr + scale_col.to(tl.int64) * padded_rows + pad_row,
            1.0,
            mask=pad_mask,
        )


@triton.jit
def _per_token_group_quant_8bit_packed_ue8m0(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Num columns of y
    y_num_columns,
    # Stride from one packed scale column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    GROUP_SIZE: tl.constexpr,
    PACK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Quantize per token group and pack UE8M0 scales for DeepGEMM."""

    # One CTA owns all four exponent bytes of a packed int32. Besides avoiding
    # atomic RMW contention, this lets the caller allocate the scale buffer
    # without a preceding zero-fill kernel.
    pid = tl.program_id(0)
    groups_per_row = y_num_columns // GROUP_SIZE
    packs_per_row = (groups_per_row + PACK - 1) // PACK
    row = pid // packs_per_row
    pack_col = pid % packs_per_row

    BLOCK: tl.constexpr = GROUP_SIZE * PACK
    col0 = pack_col * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    col_mask = cols < y_num_columns

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row_offset = row.to(tl.int64) * y_num_columns
    y = tl.load(y_ptr + row_offset + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = tl.reshape(y, (PACK, GROUP_SIZE))
    _absmax = tl.max(tl.abs(y), axis=1)
    scale_raw = tl.maximum(_absmax / bit8_max, eps)
    exponent = tl.ceil(tl.log2(scale_raw))
    y_s = tl.exp2(exponent)
    y_q = tl.clamp(y / y_s[:, None], bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(
        y_q_ptr + row_offset + cols,
        tl.reshape(y_q, (BLOCK,)),
        mask=col_mask,
    )

    group_ids = pack_col * PACK + tl.arange(0, PACK)
    group_mask = group_ids < groups_per_row
    exponent_biased = tl.where(
        group_mask, tl.clamp(exponent + 127.0, 0.0, 255.0), 0.0
    ).to(tl.uint32)
    packed_scale = tl.sum(exponent_biased << (tl.arange(0, PACK) * 8))
    scale_offset = pack_col.to(tl.int64) * y_s_col_stride + row.to(tl.int64)
    tl.store(y_s_ptr + scale_offset, packed_scale)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _create_packed_ue8m0_scale(
    x_shape,
    device,
    group_size: int,
    *,
    zero: bool,
):
    """Allocate the packed column-major UE8M0 scale view."""
    assert len(x_shape) == 2, "UE8M0 packed scales currently require 2D input"
    assert group_size == 128, "UE8M0 packed scales currently require group_size=128"
    *x_batch, x_q_mn, x_q_k = x_shape
    x_s_mn, x_s_k = x_q_mn, x_q_k // group_size
    aligned_mn = align(x_s_mn, 4)
    packed_k = ceil_div(x_s_k, 4)
    scale_base = torch.empty(
        (*x_batch, packed_k, aligned_mn),
        device=device,
        dtype=torch.int,
    )
    if zero:
        scale_base.zero_()
    return scale_base.transpose(-1, -2)[..., :x_s_mn, :]


def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        return _create_packed_ue8m0_scale(
            x_shape,
            device,
            group_size,
            zero=True,
        )
    elif column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = align(x_shape[-2], 4)
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )


def _per_token_group_quant_8bit_raw(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor.
        column_major_scales: Store scale groups as columns.
        scale_tma_aligned: Pad the scale storage for TMA alignment.
        scale_ue8m0: Encode four power-of-two scale exponents per int32.
        enable_pdl: Join a Programmatic Dependent Launch chain for the packed
            UE8M0 kernel.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if _is_amd:
        if dtype == torch.int8:
            bit8_max = 127.0
            bit8_min = -128.0
        else:
            bit8_max = fp8_max
            bit8_min = -bit8_max
    else:
        if dtype == torch.int8:
            info = torch.iinfo(dtype)
        else:
            info = torch.finfo(dtype)
        bit8_max = info.max
        bit8_min = info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    if scale_ue8m0:
        # The packed kernel writes every byte of every valid scale word, so no
        # initialization kernel is needed in front of the PDL chain.
        x_s = _create_packed_ue8m0_scale(
            x.shape,
            x.device,
            group_size,
            zero=False,
        )
    else:
        x_s = create_per_token_group_quant_fp8_output_scale(
            x_shape=x.shape,
            device=x.device,
            group_size=group_size,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=False,
        )

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        assert group_size == 128
        pack = 4
        groups_per_row = x.shape[1] // group_size
        packs_per_row = ceil_div(groups_per_row, pack)
        _per_token_group_quant_8bit_packed_ue8m0[(x.shape[0] * packs_per_row,)](
            x,
            x_q,
            x_s,
            x.shape[1],
            x_s.stride(-1),
            eps,
            bit8_min=bit8_min,
            bit8_max=bit8_max,
            GROUP_SIZE=group_size,
            PACK=pack,
            ENABLE_PDL=enable_pdl,
            num_warps=2,
            num_stages=num_stages,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    elif column_major_scales:
        _per_token_group_quant_8bit_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x_s.stride(1),
            eps,
            bit8_min=bit8_min,
            bit8_max=bit8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
            SCALE_UE8M0=scale_ue8m0,
        )
    else:
        assert not scale_ue8m0
        _per_token_group_quant_8bit[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            bit8_min=bit8_min,
            bit8_max=bit8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


def _flashinfer_sm90_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if not (
        _is_nvidia
        and platform.is_hopper
        and group_size == 128
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and column_major_scales
        and scale_tma_aligned
        and not scale_ue8m0
    ):
        return None

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=x.shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=False,
    )
    if _flashinfer_fp8_blockscale_quantize_runner_sm90 is error_fn:
        return None
    try:
        runner = _flashinfer_fp8_blockscale_quantize_runner_sm90()
        runner.fp8_quantize_1x128(x, x_q, x_s, False)
    except RuntimeError:
        return None
    return x_q, x_s


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    enable_pdl: bool = False,
):
    """Quantize each contiguous token group to FP8 and return its scale.

    Args:
        x: Contiguous input tensor.
        group_size: Number of values represented by one scale.
        column_major_scales: Return group-major/TMA-friendly scale storage.
        scale_tma_aligned: Pad scale storage to its backend alignment.
        scale_ue8m0: Pack four UE8M0 exponent scales in each int32.
        enable_pdl: Join a Programmatic Dependent Launch chain when using the
            packed UE8M0 Triton kernel.

    Returns:
        The quantized FP8 tensor and its block scales.
    """
    flashinfer_quantized = _flashinfer_sm90_per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )
    if flashinfer_quantized is not None:
        return flashinfer_quantized

    if (
        _is_nvidia
        and group_size == 128
        and not column_major_scales
        and not scale_tma_aligned
        and not scale_ue8m0
    ):
        return _trtllm_per_token_group_quant_fp8(x, group_size)

    return _per_token_group_quant_8bit_raw(
        x,
        group_size,
        dtype=fp8_dtype,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        enable_pdl=enable_pdl,
    )


def flashinfer_fp8_blockscale_quantize_prepacked(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations into FlashInfer's native MN-major scale layout.

    For row counts already divisible by four, this exposes TRT-LLM's native
    ``[K / 128, M]`` scale output directly. Otherwise a fused Triton kernel
    writes the valid quantized rows plus zero/one padding directly into
    ``[round_up(M, 4), K]`` values and ``[K / 128, round_up(M, 4)]`` scales.

    Args:
        x: Contiguous BF16/FP16 activation matrix shaped ``[M, K]``.
        group_size: Number of K elements represented by one scale. FlashInfer's
            FP8 block-scale GEMM currently requires 128.

    Returns:
        A tuple containing padded FP8 values and contiguous MN-major FP32
        scales. The returned row count equals ``round_up(M, 4)``.
    """
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
    padded_rows = align(valid_rows, 4)
    if (
        padded_rows == valid_rows
        and x.dtype == torch.bfloat16
        and _trtllm_per_token_group_quant_fp8 is not error_fn
    ):
        x_q, x_s = _trtllm_per_token_group_quant_fp8(x, group_size, False)
        expected_shape = (columns // group_size, valid_rows)
        if tuple(x_s.shape) != expected_shape or not x_s.is_contiguous():
            raise RuntimeError(
                "TRT-LLM FP8 quantizer returned unexpected prepared scales: "
                f"shape={tuple(x_s.shape)}, stride={tuple(x_s.stride())}, "
                f"expected contiguous {expected_shape}"
            )
        return x_q, x_s

    x_q = torch.empty(
        (padded_rows, columns),
        device=x.device,
        dtype=fp8_dtype,
    )
    x_s = torch.empty(
        (columns // group_size, padded_rows),
        device=x.device,
        dtype=torch.float32,
    )
    groups = valid_rows * (columns // group_size)
    _per_token_group_quant_8bit_padded_colmajor[(groups,)](
        x,
        x_q,
        x_s,
        group_size,
        columns,
        valid_rows,
        padded_rows,
        bit8_min=fp8_min,
        bit8_max=fp8_max,
        BLOCK=group_size,
        num_warps=1,
        num_stages=1,
    )
    return x_q, x_s


@triton.jit
def _per_block_quant_fp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    N,
    K,
    x_stride_n,
    s_stride_n,
    block_n,
    block_k,
    eps,
    bit8_min,
    bit8_max,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Quantize one ``[block_n, block_k]`` tile with a single shared scale."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    offs_n = pid_n * block_n + rows
    offs_k = pid_k * block_k + cols
    mask = (rows[:, None] < block_n) & (cols[None, :] < block_k)
    mask &= (offs_n[:, None] < N) & (offs_k[None, :] < K)
    offsets = offs_n[:, None].to(tl.int64) * x_stride_n + offs_k[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / bit8_max
    q = tl.clamp(x / scale, bit8_min, bit8_max).to(q_ptr.dtype.element_ty)

    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(s_ptr + pid_n * s_stride_n + pid_k, scale)


def per_block_quant_fp8(
    x: torch.Tensor,
    block_size: Tuple[int, int] = (128, 128),
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 with one scale per 2D block.

    Args:
        x: Weight matrix ``[N, K]`` in a floating-point dtype, row-major.
        block_size: ``(block_n, block_k)`` tile shape of one scale.
        eps: Lower bound on a tile's absmax, to avoid dividing by zero.

    Returns:
        Tuple of the FP8 tensor shaped like ``x`` and its ``float32`` scales
        shaped ``[ceil(N / block_n), ceil(K / block_k)]``. The scales are
        dequantization multipliers: ``x ~= q * scale``.
    """
    if x.dim() != 2:
        raise ValueError(f"per_block_quant_fp8 expects a 2D tensor, got {x.dim()}D.")
    block_n, block_k = int(block_size[0]), int(block_size[1])

    x = x.contiguous()
    N, K = x.shape
    q = torch.empty_like(x, dtype=fp8_dtype)
    scales = torch.empty(
        (ceil_div(N, block_n), ceil_div(K, block_k)),
        dtype=torch.float32,
        device=x.device,
    )

    _per_block_quant_fp8_kernel[(scales.shape[0], scales.shape[1])](
        x,
        q,
        scales,
        N,
        K,
        x.stride(0),
        scales.stride(0),
        block_n,
        block_k,
        eps,
        fp8_min,
        fp8_max,
        BLOCK_N=triton.next_power_of_2(block_n),
        BLOCK_K=triton.next_power_of_2(block_k),
        num_warps=8,
    )
    return q, scales


def per_token_quant_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = fp8_dtype,
):
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[0],
        1,
        device=x.device,
        dtype=torch.float32,
    )

    _trtllm_per_token_quant_fp8(x, x_q, x_s)

    return x_q, x_s


@triton.jit
def _static_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_s_repeat_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    REPEAT_SCALE: tl.constexpr,
):
    """A Triton-accelerated function to perform quantization using the given scale on a
    tensor

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    if REPEAT_SCALE:
        y_s_repeat_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y_s = tl.load(y_s_ptr).to(tl.float32)
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    if REPEAT_SCALE:
        tl.store(y_s_repeat_ptr, y_s)


def static_quant_fp8(
    x: torch.Tensor,
    x_s: torch.Tensor,
    repeat_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform static quantization using the given scale on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        x_s: The quantization scale.
        repeat_scale: Whether to broadcast per-tensor scale to per-channel scale.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert x.is_contiguous(), "`x` is not contiguous"
    assert x_s.numel() == 1, "only supports per-tensor scale"

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    if repeat_scale:
        x_s_repeat = torch.empty(
            (M, 1),
            device=x.device,
            dtype=torch.float32,
        )
    else:
        x_s_repeat = None

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _static_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        x_s_repeat,
        N,
        N,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        REPEAT_SCALE=repeat_scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    x_s = x_s_repeat if repeat_scale else x_s
    return x_q, x_s
