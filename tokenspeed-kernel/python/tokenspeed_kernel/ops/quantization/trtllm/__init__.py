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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

trtllm_fp8_token_group_128 = error_fn
trtllm_fp8_token = error_fn
trtllm_fp8_tensor = error_fn

if platform.is_nvidia:
    from tokenspeed_kernel.ops.other.native.trtllm import (
        per_tensor_quant_fp8 as _trtllm_per_tensor_quant_fp8,
    )
    from tokenspeed_kernel.ops.other.native.trtllm import (
        per_token_group_quant_8bit as _trtllm_per_token_group_quant_8bit,
    )
    from tokenspeed_kernel.ops.other.native.trtllm import (
        per_token_quant_fp8 as _trtllm_per_token_quant_fp8,
    )

    _FP8_DTYPE = torch.float8_e4m3fn

    def trtllm_fp8_token_group_128(x: torch.Tensor) -> torch.Tensor:
        qweight, _scale = _trtllm_per_token_group_quant_8bit(x, group_size=128)
        return qweight.float()

    def trtllm_fp8_token(x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x, dtype=_FP8_DTYPE)
        scale = torch.empty(x.size(0), dtype=torch.float32, device=x.device)
        _trtllm_per_token_quant_fp8(x, output, scale)
        return output.float()

    def trtllm_fp8_tensor(x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x, dtype=_FP8_DTYPE)
        scale = torch.zeros(1, dtype=torch.float32, device=x.device)
        _trtllm_per_tensor_quant_fp8(x, output, scale)
        return output.float()

    @register_kernel(
        "quantization",
        "fp8_with_scale",
        name="trtllm_quantize_fp8_tensor_or_token",
        solution="trtllm",
        capability=CapabilityRequirement(
            max_arch_version=ArchVersion(10, 9),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "granularity": frozenset({"tensor", "token"}),
            "scale_encoding": frozenset({"float32"}),
            "scale_layout": frozenset({"row_major"}),
        },
        priority=Priority.PERFORMANT,
    )
    def trtllm_quantize_fp8_tensor_or_token(
        x: torch.Tensor,
        granularity: str = "tensor",
        group_size: int | None = None,
        scale_encoding: str = "float32",
        scale_layout: str = "row_major",
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if granularity not in {"tensor", "token"}:
            raise ValueError(f"unsupported TRT-LLM FP8 granularity: {granularity!r}")
        if scale_encoding != "float32" or scale_layout != "row_major":
            raise ValueError(
                f"TRT-LLM {granularity} FP8 requires row-major float32 scales"
            )

        if granularity == "tensor":
            x = x.contiguous()
            q = torch.empty_like(x, dtype=_FP8_DTYPE)
            scale = torch.empty(1, dtype=torch.float32, device=x.device)
            _trtllm_per_tensor_quant_fp8(x, q, scale)
        else:
            x_2d = x.contiguous().view(-1, x.shape[-1])
            q_2d = torch.empty_like(x_2d, dtype=_FP8_DTYPE)
            scale = torch.empty(x_2d.shape[0], dtype=torch.float32, device=x.device)
            _trtllm_per_token_quant_fp8(x_2d, q_2d, scale)
            q = q_2d.view(x.shape)
            scale = scale.unsqueeze(-1)
        return q, scale

    @register_kernel(
        "quantization",
        "fp8_with_scale",
        name="trtllm_quantize_fp8_token_group",
        solution="trtllm",
        capability=CapabilityRequirement(
            max_arch_version=ArchVersion(10, 9),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "granularity": frozenset({"token_group_128"}),
            "scale_encoding": frozenset({"float32", "ue8m0"}),
            "scale_layout": frozenset({"row_major", "column_major"}),
        },
        priority=Priority.PERFORMANT,
    )
    def trtllm_quantize_fp8_token_group(
        x: torch.Tensor,
        granularity: str = "token_group",
        group_size: int | None = None,
        scale_encoding: str = "float32",
        scale_layout: str = "row_major",
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if granularity != "token_group" or group_size != 128:
            raise ValueError("TRT-LLM token-group FP8 requires group_size=128")
        if scale_encoding not in {"float32", "ue8m0"}:
            raise ValueError(
                f"unsupported TRT-LLM FP8 scale encoding: {scale_encoding}"
            )
        x_2d = x.contiguous().view(-1, x.shape[-1])
        q, scales = _trtllm_per_token_group_quant_8bit(
            x_2d,
            group_size=group_size,
            use_ue8m0=scale_encoding == "ue8m0",
        )
        if scale_layout == "row_major":
            scales = scales.transpose(0, 1).contiguous()
        elif scale_layout != "column_major":
            raise ValueError(
                "TRT-LLM token-group FP8 supports row_major or column_major scales"
            )
        return q.view(x.shape), scales


__all__ = [
    "trtllm_fp8_token_group_128",
    "trtllm_fp8_token",
    "trtllm_fp8_tensor",
]
