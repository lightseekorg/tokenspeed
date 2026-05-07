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

from typing import Optional, Tuple

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import error_fn

_is_amd = Platform.get().is_amd
_is_nvidia = Platform.get().is_nvidia
platform = Platform.get()
fp8_dtype = platform.fp8e4m3fn.dtype
fp8_max = platform.fp8e4m3fn.max
fp8_min = platform.fp8e4m3fn.min

if _is_nvidia:
    from tokenspeed_kernel.ops.quantization.flashinfer import (
        fp8_blockscale_quantize_runner_sm90 as _flashinfer_fp8_blockscale_quantize_runner_sm90,
    )
    from tokenspeed_kernel.thirdparty.trtllm import (
        per_tensor_quant_fp8 as _trtllm_per_tensor_quant_fp8,
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
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = align(x_s_mn, 4)
        aligned_k = align(x_s_k, 4)
        # The CUDA kernel expects the aligned scale tensor layout.
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
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
    dtype: torch.dtype = platform.fp8e4m3fn.dtype,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor.

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
            bit8_max = 224.0
            bit8_min = -bit8_max
    else:
        if dtype == torch.int8:
            info = torch.iinfo(dtype)
        else:
            info = torch.finfo(dtype)
        bit8_max = info.max
        bit8_min = info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
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
    if column_major_scales:
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

    if scale_ue8m0:
        from tokenspeed_kernel.thirdparty.deep_gemm import (
            transform_sf_into_required_layout,
        )

        assert group_size == 128
        x_s = transform_sf_into_required_layout(
            x_s,
            num_groups=None,
            mn=x_q.shape[0],
            k=x_q.shape[1],
            recipe=(1, group_size, group_size),
            is_sfa=True,
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
):
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
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and not column_major_scales
        and not scale_tma_aligned
        and not scale_ue8m0
    ):
        assert (
            x.shape[-1] % group_size == 0
        ), "the last dimension of `x` cannot be divisible by `group_size`"
        M = x.shape[0]
        x_q, x_s = _trtllm_per_token_group_quant_fp8(x, group_size)
        # TRT-LLM returns scales grouped by K-block, with the M dimension
        # padded to a multiple of 4:
        #   [group0_m0..group0_m_aligned, group1_m0..]
        # Convert that flat group-major layout back to the row-major
        # (M, K/group_size) layout expected by downstream kernels.
        num_groups = x.shape[-1] // group_size
        aligned_m = align(M, 4)
        x_s = (
            x_s[: num_groups * aligned_m]
            .reshape(num_groups, aligned_m)
            .transpose(0, 1)[:M]
            .contiguous()
        )
        return x_q, x_s

    return _per_token_group_quant_8bit_raw(
        x,
        group_size,
        dtype=fp8_dtype,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )


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


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    if _is_amd:
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        shape = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                    output, input.contiguous(), scale, None
                )
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
        else:
            # Static scaling
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"
            torch.ops._C.static_scaled_fp8_quant(output, input, scale)

        return output, scale
    else:
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        shape = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                _trtllm_per_token_quant_fp8(input, output, scale)
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                _trtllm_per_tensor_quant_fp8(input, output, scale)
        else:
            # Static scaling
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"
            _trtllm_per_tensor_quant_fp8(input, output, scale)

        return output, scale
