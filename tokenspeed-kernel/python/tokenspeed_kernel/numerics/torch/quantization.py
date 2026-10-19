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

"""PyTorch references for ``tokenspeed_kernel.ops.quantization``.

The ``quantize`` family references serve the numerics CLI: each returns
``qweight.float()`` — the fp8 quantized values cast back to fp32 for
comparison. The scale tensor's layout differs across producers (SM90 vs SM100,
row-major vs column-major) so we don't compare it directly; the qweight values
tell us whether the per-group statistics + rounding agree.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_FINFO = torch.finfo(_FP8_DTYPE)
_FP8_MAX = _FP8_FINFO.max  # 448 for e4m3fn


def ue8m0_to_float(scales: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 exponent bytes (``uint8``) to FP32 powers of two."""
    return torch.exp2(scales.float() - 127.0)


def quantize_fp8_groups(
    x: torch.Tensor,
    group_size: int,
    *,
    power_of_two_scale: bool,
    min_scale: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each ``group_size`` run of the last dimension to FP8 E4M3.

    The group's scale is ``amax / 448`` clamped below by ``min_scale``, rounded
    up to a power of two when ``power_of_two_scale`` (UE8M0 scales).

    Returns:
        ``(codes, scales)``: FP8 codes shaped like ``x`` and FP32 scales
        ``[..., last_dim / group_size]``.
    """
    groups = x.float().unflatten(-1, (-1, group_size))
    scale = (groups.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp_min(min_scale)
    if power_of_two_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    codes = (groups / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return codes.flatten(-2), scale.squeeze(-1)


def dequantize_fp8_groups(
    codes: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """FP32 values of FP8 ``codes`` with one scale per ``group_size`` columns.

    ``scales`` are FP32, or UE8M0 exponent bytes when ``uint8``.
    """
    if scales.dtype == torch.uint8:
        scales = ue8m0_to_float(scales)
    width = codes.shape[-1]
    return (
        codes.float()
        * scales.float().repeat_interleave(group_size, dim=-1)[..., :width]
    )


# OCP E2M1 values by 4-bit code: sign bit 3, magnitude bits 0-2.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_VALUES = _E2M1_VALUES + tuple(-v for v in _E2M1_VALUES)


def dequantize_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """FP32 values of packed E2M1 codes, low nibble first, doubling the last dim."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    return table[codes.long()]


def quantize_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Pack values already in E2M1 range to the nearest codes, low nibble first."""
    magnitudes = torch.tensor(
        _E2M1_VALUES[:8], dtype=torch.float32, device=values.device
    )
    codes = (values.abs().unsqueeze(-1) - magnitudes).abs().argmin(dim=-1)
    codes = codes.to(torch.uint8) | ((values < 0).to(torch.uint8) << 3)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def dequantize_mxfp4(
    packed: torch.Tensor, scales: torch.Tensor, *, group_size: int = 32
) -> torch.Tensor:
    """Decode packed OCP E2M1 values with one UE8M0 scale per group."""
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 payload and scales must use uint8 storage")
    values = dequantize_e2m1(packed)
    expanded_scales = ue8m0_to_float(scales).repeat_interleave(group_size, dim=-1)
    if values.shape != expanded_scales.shape:
        raise ValueError("packed values and scale groups describe different shapes")
    return values * expanded_scales


def quantize_nvfp4_blocks(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to packed E2M1 with one FP8 E4M3 scale per 16 values.

    The block scale is ``amax / 6`` (floor ``2**-9``) rounded to FP8; there is
    no global scale.

    Returns:
        ``(packed, scales)``: ``uint8`` codes ``[..., K / 2]`` and linear
        (unswizzled) FP8 scales ``[..., K / 16]``.
    """
    assert x.shape[-1] % 16 == 0
    blocks = x.float().unflatten(-1, (-1, 16))
    scales = (blocks.abs().amax(dim=-1) / 6.0).clamp_min(2**-9).to(_FP8_DTYPE)
    normalized = (blocks / scales.float().unsqueeze(-1)).flatten(-2)
    return quantize_e2m1(normalized).contiguous(), scales


def dequantize_nvfp4_blocks(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 values of packed E2M1 codes with linear FP8 scales per 16 values."""
    return dequantize_e2m1(packed) * scales.float().repeat_interleave(16, dim=-1)


def fake_quantize_mxfp4_even(values: torch.Tensor) -> torch.Tensor:
    """MXFP4 round trip with Quark's "even" scale and saturating E2M1 RNE.

    Each 32-value group's ``amax = m * 2**e`` takes scale exponent ``e - 2``
    when ``m < 1.75``, otherwise ``e - 1``; scaled values above 6 saturate,
    and a tie rounds to the neighbour with the even E2M1 code.
    """
    groups = values.float().reshape(*values.shape[:-1], -1, 32)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    exponent = ((amax.view(torch.int32) + 0x200000) >> 23) - 2
    scale = torch.exp2(exponent.clamp(0, 254).float() - 127)
    scaled = groups.abs() / scale
    grid = torch.tensor(_E2M1_VALUES[:8], device=values.device)
    midpoints = (grid[1:] + grid[:-1]) / 2
    index = (scaled[..., None] > midpoints).sum(-1) + (
        (scaled[..., None] == midpoints)
        & (torch.arange(7, device=values.device) % 2 == 1)
    ).sum(-1)
    return (grid[index] * groups.sign() * scale).reshape(values.shape)


def round_to_bf16_exact(x: torch.Tensor) -> torch.Tensor:
    """FP64 values rounded once to the nearest BF16 (RNE), kept in FP64.

    ``x.to(torch.bfloat16)`` of FP64 rounds through FP32 first, which can
    round twice.
    """
    mantissa, exponent = torch.frexp(x)
    return torch.ldexp(torch.round(mantissa * 256) / 256, exponent)


def quantize_fp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP8 E4M3 with correctly rounded quotients.

    The FP32 scale is ``amax / 448`` (floor ``2**-126``); codes are
    ``RNE(x / scale)`` from FP64 quotients, which round to the FP32 ones
    (Torch would divide through a reciprocal).

    Returns:
        ``(codes, scales)``: FP8 codes like ``x`` and FP32 ``[rows]`` scales.
    """
    values = x.double()
    scale = (values.abs().amax(dim=1) / 448.0).float().clamp(min=2.0**-126)
    codes = (values / scale.double()[:, None]).float()
    return codes.clamp(-448.0, 448.0).to(_FP8_DTYPE), scale


def quantize_fp8_rows_trtllm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TRT-LLM's ``perTokenQuantization`` with ``fp8RowWise`` on BF16 rows.

    ``amax`` is at least BF16(1e-6), the scale ``max(amax / 448, 1 / (448 *
    512))`` and the codes ``RNE_sat(E4M3(BF16(x * min(448 / amax, 448 *
    512))))``: the FP32 product rounds to BF16 first, as ``cuda_cast`` to
    ``__nv_fp8x2_e4m3`` does. The codes use the reciprocal, the
    dequantization the scale.

    Returns:
        ``(codes, scales)``: FP8 codes like ``x`` and FP32 ``[rows]`` scales.
    """
    rows = x.float()
    amax = rows.abs().amax(dim=1).clamp(min=float(torch.tensor(1e-6).bfloat16()))
    scale = torch.maximum(amax / 448.0, torch.tensor(1.0 / (448.0 * 512.0)))
    multiplier = torch.minimum(448.0 / amax, torch.tensor(448.0 * 512.0))
    product = (rows * multiplier[:, None]).bfloat16().float()
    return product.clamp(-448.0, 448.0).to(_FP8_DTYPE), scale


def fake_quantize_nvfp4(x: torch.Tensor, global_scale: float) -> torch.Tensor:
    """NVFP4 round trip with a global scale: per 16 values an FP8 E4M3 block
    scale ``amax / (6 * global_scale)``, then E2M1 RNE (ties to even codes)."""
    magnitudes = torch.tensor(_E2M1_VALUES[:8], device=x.device)
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    sf = (blocks.abs().amax(-1) / (6 * global_scale)).to(_FP8_DTYPE).float()
    normalized = blocks / (sf.unsqueeze(-1) * global_scale).clamp_min(1e-30)
    distances = (normalized.abs().unsqueeze(-1) - magnitudes).abs()
    nearest = distances.argmin(-1)
    tie = (nearest % 2 == 1) & (nearest < 7)
    next_distance = distances.gather(
        -1, (nearest + 1).clamp_max(7).unsqueeze(-1)
    ).squeeze(-1)
    nearest = torch.where(
        tie & (next_distance == distances.amin(-1)), nearest + 1, nearest
    )
    values = magnitudes[nearest] * normalized.sign() * sf.unsqueeze(-1) * global_scale
    return values.reshape_as(x)


@register_kernel(
    "quantization",
    "fp8_with_scale",
    name="torch_quantize_fp8_with_scale",
    solution="torch",
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    traits={"scale_encoding": frozenset({"float32", "ue8m0"})},
    priority=Priority.REFERENCE,
)
def torch_quantize_fp8_with_scale(
    x: torch.Tensor,
    *,
    granularity: str,
    group_size: int | None,
    scale_encoding: str,
    enable_pdl: bool,
    block_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic FP8 quantization with ``amax / 448`` scales (floor ``1e-10``).

    ``granularity`` is ``"token"`` (scales ``[M, 1]``), ``"token_group"``
    (``[M, K / group_size]``) or ``"block"`` (``[ceil(M / bm), ceil(K / bk)]``
    for the 2-D ``block_size``, given only with block granularity).
    ``scale_encoding="ue8m0"`` rounds scales up to powers of two and returns
    their exponent bytes. ``enable_pdl`` is a launch hint with no effect here.
    """
    del enable_pdl
    ue8m0 = scale_encoding == "ue8m0"
    if granularity == "block":
        assert block_size is not None and x.ndim == 2
        rows, cols = x.shape
        bm, bk = block_size
        padded = x.float().new_zeros(-(-rows // bm) * bm, -(-cols // bk) * bk)
        padded[:rows, :cols] = x.float()
        tiles = padded.unflatten(0, (-1, bm)).unflatten(-1, (-1, bk))
        scales = (tiles.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-10)
        if ue8m0:
            scales = torch.exp2(torch.ceil(torch.log2(scales)))
        expanded = scales.repeat_interleave(bm, 0).repeat_interleave(bk, 1)
        codes = (x.float() / expanded[:rows, :cols]).clamp(-_FP8_MAX, _FP8_MAX)
        codes = codes.to(_FP8_DTYPE)
    else:
        width = x.shape[-1] if granularity == "token" else group_size
        codes, scales = quantize_fp8_groups(x, width, power_of_two_scale=ue8m0)
    if ue8m0:
        scales = (torch.log2(scales) + 127).to(torch.uint8)
    return codes, scales


@register_kernel(
    "quantize",
    "fp8_token_group_128",
    name="torch_fp8_token_group_128",
    solution="torch",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=Priority.REFERENCE,
)
def torch_fp8_token_group_128(x: torch.Tensor) -> torch.Tensor:
    """Per-token grouped fp8 quantization with group size 128."""
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    assert x.shape[1] % 128 == 0, f"K={x.shape[1]} must be divisible by 128"
    return quantize_fp8_groups(x, 128, power_of_two_scale=False)[0].float()


@register_kernel(
    "quantize",
    "fp8_token",
    name="torch_fp8_token",
    solution="torch",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=Priority.REFERENCE,
)
def torch_fp8_token(x: torch.Tensor) -> torch.Tensor:
    """Per-token fp8 quantization (one scale per row)."""
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    return quantize_fp8_groups(x, x.shape[1], power_of_two_scale=False)[0].float()


@register_kernel(
    "quantize",
    "fp8_tensor",
    name="torch_fp8_tensor",
    solution="torch",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=Priority.REFERENCE,
)
def torch_fp8_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor fp8 quantization (one scalar scale for the whole tensor)."""
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    codes, _ = quantize_fp8_groups(x.flatten(), x.numel(), power_of_two_scale=False)
    return codes.float().view_as(x)
