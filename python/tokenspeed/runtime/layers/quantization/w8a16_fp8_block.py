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

"""Serialized E4M3 block weights with BF16 activations and computation."""

import torch

from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config


class W8A16Fp8BlockConfig(Fp8Config):
    """Preserve checkpoint codes/scales; decode weights only for each operation.

    Storage uses the same 128x128 FP32 block scales as an FP8 checkpoint.
    ``activation_scheme`` describes its checkpoint metadata only. This method
    never quantizes activations or regenerates checkpoint weight scales.
    """

    supports_weight_discard = True

    def __init__(self, *, ignored_layers):
        super().__init__(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            scale_fmt="float32",
            ignored_layers=ignored_layers,
        )

    @classmethod
    def get_name(cls):
        return "w8a16_fp8_block"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    def moe_weight_dtype(self, prefix=""):
        return self.get_name()

    @classmethod
    def from_config(cls, config):
        if (
            config.get("quant_method") not in ("fp8", cls.get_name())
            or config.get("weight_block_size") != [128, 128]
            or config.get("fmt", "e4m3") != "e4m3"
            or config.get("scale_fmt") not in (None, "float32")
        ):
            raise ValueError(
                "W8A16 requires serialized E4M3 weights with 128x128 FP32 scales"
            )
        return cls(
            ignored_layers=config.get(
                "ignored_layers", config.get("modules_to_not_convert", [])
            )
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        # Selecting weight-only computation is explicit; ordinary FP8 keeps W8A8.
        if user_quant == cls.get_name() and hf_quant_cfg.get("quant_method") == "fp8":
            return cls.get_name()
        return None

    def get_quant_method(self, layer, prefix):
        from tokenspeed.runtime.layers.dense.w8a16_fp8_block import (
            W8A16Fp8BlockLinearMethod,
        )

        return W8A16Fp8BlockLinearMethod(self)


def validate_w8a16_weights(codes, scales):
    """Reject malformed serialized storage without preparing a weight copy."""
    if codes.ndim not in (2, 3):
        raise ValueError("W8A16 weights must be matrices or batches of matrices")
    expected = (
        *codes.shape[:-2],
        (codes.shape[-2] + 127) // 128,
        (codes.shape[-1] + 127) // 128,
    )
    if (
        codes.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float32
        or codes.device != scales.device
        or tuple(scales.shape) != expected
        or not codes.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("W8A16 requires contiguous E4M3 codes and 128x128 FP32 scales")
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("W8A16 block scales must be finite and positive")
