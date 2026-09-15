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

"""Explicit, non-owning prequantized MoE activation contract."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel.signature import ScaleFormat, TensorFormat

NVFP4_ACTIVATION_FORMAT = TensorFormat(
    storage_dtype=torch.uint8,
    format="nvfp4",
    scale=ScaleFormat(torch.float8_e4m3fn, "block", block_shape=(1, 16)),
)


@dataclass(frozen=True)
class Nvfp4Activation:
    """NVFP4 activation with linear scales and its original BF16 shape.

    Args:
        data: Contiguous packed E2M1 ``uint8[M, H/2]`` (low nibble first).
        scales: Contiguous linear E4M3 ``[M, H/16]`` block scales.
        logical_shape: Original ``(M, H)``, not the packed byte dimensions.
        input_scale_quant: Scalar FP32 encoding multiplier, borrowed from the
            receiving MoE's processed weights. It must remain pointer-stable.

    Tensor storage is borrowed. Its owner must keep it alive and refrain from
    reuse until the consuming MoE has completed on the issuing stream. Metadata
    validation does not read device values or synchronize the host.
    """

    data: torch.Tensor
    scales: torch.Tensor
    logical_shape: tuple[int, int]
    input_scale_quant: torch.Tensor

    def __post_init__(self) -> None:
        m, h = self.logical_shape
        if m < 0 or h <= 0 or h % 16:
            raise ValueError("NVFP4 requires M >= 0 and H a positive multiple of 16")
        if self.data.dtype != torch.uint8 or self.data.shape != (m, h // 2):
            raise ValueError("NVFP4 data must be uint8 [M,H/2]")
        if self.scales.dtype != torch.float8_e4m3fn or self.scales.shape != (
            m,
            h // 16,
        ):
            raise ValueError("NVFP4 scales must be linear E4M3 [M,H/16]")
        if not self.data.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError("NVFP4 payload and scales must be contiguous")
        if (
            self.input_scale_quant.dtype != torch.float32
            or self.input_scale_quant.numel() != 1
        ):
            raise ValueError("NVFP4 input_scale_quant must be scalar FP32")
        if (
            self.scales.device != self.data.device
            or self.input_scale_quant.device != self.data.device
        ):
            raise ValueError("NVFP4 payload, scales and multiplier must share a device")

    @property
    def shape(self) -> tuple[int, int]:
        """Logical BF16 shape used by MoE output allocation."""
        return self.logical_shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    def validate_receiver(self, input_scale_quant: torch.Tensor) -> None:
        """Reject a multiplier from a different MoE without a device read."""
        if self.input_scale_quant.data_ptr() != input_scale_quant.data_ptr():
            raise ValueError("NVFP4 activation was encoded for a different MoE scale")


__all__ = ["NVFP4_ACTIVATION_FORMAT", "Nvfp4Activation"]
