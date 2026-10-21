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

"""Host-side weight layouts that the megakernel GEMV (``mx.py``) reads."""

from __future__ import annotations

import torch

MXFP4_BLOCK = 32
# E2M1 code -> magnitude. Bit 3 is the sign.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4_tiles(packed: torch.Tensor, k: int) -> torch.Tensor:
    """N16 tile order of a packed ``[N, K0/2]`` E2M1 weight, K padded to ``k``.

    Per 16-row block, each 64-byte K chunk becomes a 1 KiB block in MFMA lane
    order (lane ``g * 16 + n`` owns row ``n``, bytes ``16 g .. 16 g + 15``).
    This is the byte order of ``preshuffle_mxfp4_weight``. Padding codes are
    0 (+0.0).
    """
    n, kp = packed.shape
    if packed.dtype != torch.uint8 or n % 16 or k % 128 or 2 * kp > k:
        raise ValueError(
            f"MXFP4 tiles need uint8 [N, K/2] with N % 16 == 0 and K <= {k}, "
            f"got {packed.dtype} {tuple(packed.shape)}"
        )
    padded = torch.nn.functional.pad(packed, (0, k // 2 - kp))
    return (
        padded.view(n // 16, 16, k // 128, 4, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(n, k // 2)
    )


def mxfp4_scales(scales: torch.Tensor, k: int) -> torch.Tensor:
    """``[N/16][ceil(KI/4)][64]`` int32 scale dwords of ``[N, K0/32]`` UE8M0 scales.

    Lane ``g * 16 + n``'s dword ``j`` holds the scales of row ``n``, K group
    ``g`` for k-iterations ``4 j .. 4 j + 3`` (byte ``i`` = iteration
    ``4 j + i``). Padding scales are 127 (1.0).
    """
    n, groups = scales.shape
    ki = k // 128
    kj = -(-ki // 4)
    if scales.dtype != torch.uint8 or n % 16 or groups * MXFP4_BLOCK > k:
        raise ValueError(f"bad MXFP4 scales {scales.dtype} {tuple(scales.shape)}")
    padded = torch.nn.functional.pad(scales, (0, kj * 16 - groups), value=127)
    # [N/16, 16 rows, kj, 4 iterations, 4 groups]
    #   -> [N/16, kj, 4 groups, 16 rows, 4 iterations]
    cells = padded.view(n // 16, 16, kj, 4, 4).permute(0, 2, 4, 1, 3).contiguous()
    return cells.view(torch.int32).view(n // 16, kj, 64)


def quantize_mxfp4_exact(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a weight whose values already lie on an MXFP4 grid, losslessly.

    Some checkpoints store weights (such as the hyper-connection mix
    projections) in BF16 after rounding them to MXFP4 values. Each 32-value
    block takes the smallest scale ``2**e`` with ``amax / 2**e <= 6``.

    Returns:
        ``(packed uint8 [N, K/2], scales uint8 [N, K/32])``.

    Raises:
        ValueError: A block does not encode exactly (so the weight was not
            on the grid and encoding would lose precision).
    """
    n, k = weight.shape
    if k % MXFP4_BLOCK:
        raise ValueError(f"K={k} is not a multiple of {MXFP4_BLOCK}")
    blocks = weight.float().view(n, k // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = blocks.abs().amax(-1)
    exponent = torch.ceil(torch.log2(torch.clamp(amax / 6.0, min=2.0**-126)))
    exponent = torch.clamp(exponent, -126, 127)
    scaled = blocks / torch.exp2(exponent)[..., None]
    values = torch.tensor(E2M1_VALUES, device=weight.device)
    codes = (scaled.abs()[..., None] - values).abs().argmin(-1)
    decoded = values[codes] * torch.sign(scaled)
    if not torch.equal(decoded * torch.exp2(exponent)[..., None], blocks):
        raise ValueError("weight is not on an MXFP4 grid")
    codes = codes | ((scaled < 0) & (codes > 0)).to(codes.dtype) << 3
    codes = codes.to(torch.uint8).view(n, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), (exponent + 127).to(torch.uint8)


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 values of a packed ``[N, K/2]`` E2M1 weight and its UE8M0 scales."""
    values = torch.tensor(E2M1_VALUES, device=packed.device)
    codes = torch.stack([packed & 15, packed >> 4], -1).view(packed.shape[0], -1).long()
    out = values[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return out * torch.exp2(scales.float() - 127).repeat_interleave(MXFP4_BLOCK, 1)


def n32_scales(scales: torch.Tensor, k: int) -> torch.Tensor:
    """UE8M0 scales ``[N, K0/32]`` in TokenSpeed's N16 MoE bank cells.

    Byte ``(n // 32) * cells * 256 + (g // 8) * 256 + (g % 4) * 64 + (n % 16) *
    4 + ((g % 8) // 4) * 2 + (n % 32) // 16`` holds row ``n``, group ``g``
    (``n16_weights._n32_scale_offset``). ``cells = ceil(k / 256)``. Padding
    groups are 127 (1.0). This layout lets a dense projection share the MoE
    banks' scale addressing.
    """
    n, groups = scales.shape
    cells = -(-k // 256)
    if scales.dtype != torch.uint8 or n % 32 or groups * MXFP4_BLOCK > k:
        raise ValueError(f"bad MXFP4 scales {scales.dtype} {tuple(scales.shape)}")
    padded = torch.nn.functional.pad(scales, (0, cells * 8 - groups), value=127)
    # [n/32, pair, 16 rows, cells, half, 4 groups] -> [n/32, cells, 4, 16, half, pair]
    cells_view = padded.view(n // 32, 2, 16, cells, 2, 4).permute(0, 3, 5, 2, 4, 1)
    return cells_view.contiguous().view(-1)


__all__ = [
    "dequantize_mxfp4",
    "n32_scales",
    "mxfp4_scales",
    "mxfp4_tiles",
    "quantize_mxfp4_exact",
]
