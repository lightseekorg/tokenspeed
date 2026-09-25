from __future__ import annotations

import enum

import torch


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


class MoeKernelLayout(enum.Enum):
    native_mxfp4 = "native_mxfp4"
    petit_mxfp4 = "petit_mxfp4"


_PETIT_SIGN_OFFSETS = (7, 15, 23, 31, 24, 16, 8, 0)
_PETIT_VALUE_OFFSETS = (1, 9, 17, 25, 28, 20, 12, 4)


def _bitreverse3(v: torch.Tensor) -> torch.Tensor:
    return ((v & 0x1) << 2) | (v & 0x2) | ((v & 0x4) >> 2)


def _petit_format_words(words: torch.Tensor) -> torch.Tensor:
    _require(words.dtype == torch.int32, "words must be int32")

    out = torch.zeros_like(words)
    for i in range(8):
        sgn_off = _PETIT_SIGN_OFFSETS[i]
        val_off = _PETIT_VALUE_OFFSETS[i]

        u = (words >> (i * 4)) & 0xF
        val = u & 0x7
        sgn = (u >> 3) * (val != 0).to(torch.int32)
        if i >= 4:
            # Petit reshuffles each 8-element group by bit-reversing the
            # 3-bit magnitude in the upper 4 lanes.
            val = _bitreverse3(val)
        out |= sgn << sgn_off
        out |= val << val_off
    return out


def _pack_petit_mxfp4_weight_kernel_layout(
    qweight_u8: torch.Tensor,
    *,
    petit_format: bool = True,
) -> torch.Tensor:
    _require(qweight_u8.dtype == torch.uint8, "qweight_u8 must be uint8")
    _require(qweight_u8.ndim == 2, "qweight_u8 must be rank-2")
    _require(qweight_u8.is_contiguous(), "qweight_u8 must be contiguous")

    size_n = qweight_u8.size(0)
    size_k = qweight_u8.size(1) * 2
    _require(size_n % 256 == 0, "size_n must be divisible by 256")
    _require(size_k % 128 == 0, "size_k must be divisible by 128")

    words = qweight_u8.view(torch.int32).reshape(
        size_n // 256,
        4,
        4,
        16,
        size_k // 128,
        4,
        4,
    )
    if petit_format:
        words = _petit_format_words(words)
    words = words.permute(0, 1, 2, 4, 6, 3, 5)
    return words.contiguous().view(torch.uint8).reshape(size_n, size_k // 2)


def _pack_petit_mxfp4_scale_kernel_layout(
    scales_e8m0: torch.Tensor,
) -> torch.Tensor:
    _require(scales_e8m0.dtype == torch.uint8, "scales_e8m0 must be uint8")
    _require(scales_e8m0.ndim == 2, "scales_e8m0 must be rank-2")
    _require(scales_e8m0.is_contiguous(), "scales_e8m0 must be contiguous")
    size_n = scales_e8m0.size(0)
    scale_cols = scales_e8m0.size(1)
    _require(size_n % 32 == 0, "size_n must be divisible by 32")
    _require(scale_cols % 8 == 0, "scale_cols must be divisible by 8")

    scales = scales_e8m0.reshape(
        size_n // 32,
        8,
        4,
        scale_cols // 8,
        8,
    )
    return scales.permute(0, 3, 1, 4, 2).contiguous().reshape_as(scales_e8m0)


def _pack_native_mxfp4_weight_kernel_layout(
    qweight_u8: torch.Tensor,
) -> torch.Tensor:
    _require(qweight_u8.dtype == torch.uint8, "qweight_u8 must be uint8")
    _require(qweight_u8.ndim == 2, "qweight_u8 must be rank-2")
    _require(qweight_u8.is_contiguous(), "qweight_u8 must be contiguous")

    size_n = qweight_u8.size(0)
    size_k = qweight_u8.size(1) * 2
    _require(size_n % 256 == 0, "size_n must be divisible by 256")
    _require(size_k % 128 == 0, "size_k must be divisible by 128")

    words = qweight_u8.view(torch.int32).reshape(
        size_n // 256,
        4,
        4,
        16,
        size_k // 128,
        4,
        4,
    )
    words = words.permute(0, 1, 2, 4, 5, 3, 6)
    return words.contiguous().view(torch.uint8).reshape(size_n, size_k // 2)


def _pack_native_mxfp4_scale_kernel_layout(
    scales_e8m0: torch.Tensor,
) -> torch.Tensor:
    _require(scales_e8m0.dtype == torch.uint8, "scales_e8m0 must be uint8")
    _require(scales_e8m0.ndim == 2, "scales_e8m0 must be rank-2")
    _require(scales_e8m0.is_contiguous(), "scales_e8m0 must be contiguous")
    size_n = scales_e8m0.size(0)
    scale_cols = scales_e8m0.size(1)
    _require(size_n % 32 == 0, "size_n must be divisible by 32")
    _require(scale_cols % 8 == 0, "scale_cols must be divisible by 8")

    scales = scales_e8m0.reshape(
        size_n // 32,
        2,
        16,
        scale_cols // 8,
        2,
        4,
    )
    return scales.permute(0, 3, 5, 2, 4, 1).contiguous().reshape_as(scales_e8m0)


def _repack_native_mxfp4(
    qweight_u8: torch.Tensor,
    scales_e8m0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if qweight_u8.ndim == 2:
        _require(scales_e8m0.ndim == 2, "scales_e8m0 must be rank-2")
        _require(
            qweight_u8.size(0) == scales_e8m0.size(0),
            "row dimension mismatch",
        )
        _require(
            qweight_u8.size(1) * 2 == scales_e8m0.size(1) * 32,
            "shape mismatch",
        )
        return (
            _pack_native_mxfp4_weight_kernel_layout(qweight_u8),
            _pack_native_mxfp4_scale_kernel_layout(scales_e8m0),
        )

    _require(qweight_u8.ndim == 3, "qweight_u8 must be rank-2 or rank-3")
    _require(scales_e8m0.ndim == 3, "scales_e8m0 must be rank-2 or rank-3")
    experts = qweight_u8.size(0)
    qw = _pack_native_mxfp4_weight_kernel_layout(
        qweight_u8.view(-1, qweight_u8.size(2)),
    )
    so = _pack_native_mxfp4_scale_kernel_layout(
        scales_e8m0.view(-1, scales_e8m0.size(2)),
    )
    return (
        qw.view(experts, qweight_u8.size(1), -1),
        so.view(experts, scales_e8m0.size(1), -1),
    )


def _repack_petit_mxfp4(
    qweight_u8: torch.Tensor,
    scales_e8m0: torch.Tensor,
    *,
    petit_format: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if qweight_u8.ndim == 2:
        _require(scales_e8m0.ndim == 2, "scales_e8m0 must be rank-2")
        _require(qweight_u8.size(0) == scales_e8m0.size(0), "row dimension mismatch")
        _require(
            qweight_u8.size(1) * 2 == scales_e8m0.size(1) * 32,
            "shape mismatch",
        )
        return (
            _pack_petit_mxfp4_weight_kernel_layout(
                qweight_u8, petit_format=petit_format
            ),
            _pack_petit_mxfp4_scale_kernel_layout(scales_e8m0),
        )

    _require(qweight_u8.ndim == 3, "qweight_u8 must be rank-2 or rank-3")
    _require(scales_e8m0.ndim == 3, "scales_e8m0 must be rank-2 or rank-3")
    experts = qweight_u8.size(0)
    qw = _pack_petit_mxfp4_weight_kernel_layout(
        qweight_u8.view(-1, qweight_u8.size(2)), petit_format=petit_format
    )
    so = _pack_petit_mxfp4_scale_kernel_layout(
        scales_e8m0.view(-1, scales_e8m0.size(2))
    )
    return (
        qw.view(experts, qweight_u8.size(1), -1),
        so.view(experts, scales_e8m0.size(1), -1),
    )


def _repack_mxfp4_bias(bias: torch.Tensor) -> torch.Tensor:
    _require(bias.dtype == torch.bfloat16, "bias must be bfloat16")
    _require(bias.ndim in (2, 3), "bias must be rank-2 or rank-3")
    _require(bias.is_contiguous(), "bias must be contiguous")

    prefix = tuple(bias.shape[:-1])
    rows = bias.numel() // bias.size(-1)
    cols = bias.size(-1)
    flat_bias = bias.reshape(rows, cols)
    padded_cols = ((cols + 255) // 256) * 256
    if padded_cols != cols:
        padded = torch.zeros((rows, padded_cols), dtype=bias.dtype, device=bias.device)
        padded[:, :cols] = flat_bias
        flat_bias = padded

    tiles = padded_cols // 256
    blocked = flat_bias.reshape(rows, tiles, 4, 4, 4, 4)
    packed = blocked.permute(0, 1, 2, 4, 3, 5).contiguous()
    return packed.reshape(*prefix, padded_cols)


def repack_moe_kernel_layout(
    data: torch.Tensor,
    scales_e8m0: torch.Tensor | None = None,
    *,
    layout: MoeKernelLayout,
    petit_format: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    _require(
        layout in (MoeKernelLayout.native_mxfp4, MoeKernelLayout.petit_mxfp4),
        f"unsupported MoE kernel layout: {layout}",
    )
    if scales_e8m0 is None:
        # All W4 schedules use weight as MFMA operand A and therefore share
        # the same adjacent-N accumulator ownership and bias permutation.
        return _repack_mxfp4_bias(data)
    if layout == MoeKernelLayout.native_mxfp4:
        return _repack_native_mxfp4(data, scales_e8m0)
    return _repack_petit_mxfp4(data, scales_e8m0, petit_format=petit_format)
