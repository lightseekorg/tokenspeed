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

_trtllm_fp8_quantize_1x128_packed_ue8m0 = error_fn
_supports_trtllm_fp8_quant_packed = None
trtllm_fp8_token_group_128 = error_fn
trtllm_fp8_packed_ue8m0 = error_fn
trtllm_fp8_token = error_fn
trtllm_fp8_tensor = error_fn

if platform.is_nvidia:
    from tokenspeed_kernel.thirdparty.cuda.trtllm_fp8_quant import (
        has_trtllm_fp8_quant_packed,
    )
    from tokenspeed_kernel.thirdparty.cuda.trtllm_fp8_quant import (
        supports_trtllm_fp8_quant_packed as _supports_trtllm_fp8_quant_packed,
    )
    from tokenspeed_kernel.thirdparty.cuda.trtllm_fp8_quant import (
        trtllm_fp8_quantize_1x128_packed_ue8m0,
    )
    from tokenspeed_kernel.thirdparty.trtllm import (
        per_tensor_quant_fp8 as _trtllm_per_tensor_quant_fp8,
    )
    from tokenspeed_kernel.thirdparty.trtllm import (
        per_token_group_quant_8bit as _trtllm_per_token_group_quant_8bit,
    )
    from tokenspeed_kernel.thirdparty.trtllm import (
        per_token_quant_fp8 as _trtllm_per_token_quant_fp8,
    )

    _FP8_DTYPE = platform.fp8e4m3fn.dtype
    if has_trtllm_fp8_quant_packed():
        _trtllm_fp8_quantize_1x128_packed_ue8m0 = trtllm_fp8_quantize_1x128_packed_ue8m0

    def trtllm_fp8_token_group_128(x: torch.Tensor) -> torch.Tensor:
        qweight, _scale = _trtllm_per_token_group_quant_8bit(x, group_size=128)
        return qweight.float()

    if _trtllm_fp8_quantize_1x128_packed_ue8m0 is not error_fn:

        @register_kernel(
            "quantization",
            "fp8_with_scale",
            name="trtllm_quantize_fp8_packed_ue8m0",
            solution="trtllm",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(10, 0),
                max_arch_version=ArchVersion(10, 0),
                vendors=frozenset({"nvidia"}),
            ),
            signatures=format_signatures("x", "dense", {torch.bfloat16}),
            traits={
                "granularity": frozenset({"token_group_128"}),
                "scale_encoding": frozenset({"packed_ue8m0"}),
            },
            priority=Priority.SPECIALIZED,
        )
        def trtllm_fp8_packed_ue8m0(
            x: torch.Tensor,
            granularity: str = "token_group",
            group_size: int | None = 128,
            scale_encoding: str = "packed_ue8m0",
            enable_pdl: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Quantize a BF16 matrix with 1x128 packed UE8M0 scales.

            Args:
                x: Contiguous, 32-byte-aligned CUDA BF16 matrix. Its column
                    count must be divisible by 128.
                granularity: Must be ``"token_group"``.
                group_size: Must be 128.
                scale_encoding: Must be ``"packed_ue8m0"``.
                enable_pdl: Whether to enable Programmatic Dependent Launch.

            Returns:
                Quantized E4M3 values and INT32 packed UE8M0 scales. The scale
                view has shape ``[M, ceil(K / 512)]`` and stride
                ``(1, align(M, 4))``.
            """
            if granularity != "token_group" or group_size != 128:
                raise ValueError(
                    "TRT-LLM packed UE8M0 requires token_group granularity "
                    "with group_size=128"
                )
            if scale_encoding != "packed_ue8m0":
                raise ValueError(
                    "TRT-LLM packed UE8M0 requires scale_encoding='packed_ue8m0'"
                )
            return _trtllm_fp8_quantize_1x128_packed_ue8m0(
                x,
                enable_pdl=enable_pdl,
            )

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
        name="trtllm_quantize_fp8_with_scale",
        solution="trtllm",
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "granularity": frozenset({"tensor", "token", "token_group_128"}),
            "scale_encoding": frozenset({"float32", "ue8m0"}),
        },
        priority=Priority.PERFORMANT,
    )
    def trtllm_quantize_fp8_with_scale(
        x: torch.Tensor,
        granularity: str = "tensor",
        group_size: int | None = None,
        scale_encoding: str = "float32",
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if granularity in {"tensor", "token"}:
            if scale_encoding != "float32":
                raise ValueError(f"TRT-LLM {granularity} FP8 requires float32 scales")

            q = torch.empty_like(x, dtype=_FP8_DTYPE)
            if granularity == "tensor":
                scale = torch.empty(1, dtype=torch.float32, device=x.device)
                _trtllm_per_tensor_quant_fp8(x, q, scale)
            else:
                scale = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
                _trtllm_per_token_quant_fp8(x, q, scale)
                scale = scale.unsqueeze(-1)
            return q, scale

        if granularity == "token_group":
            return _trtllm_per_token_group_quant_8bit(
                x,
                group_size=group_size,
                use_ue8m0=scale_encoding == "ue8m0",
            )

        raise ValueError(f"unsupported TRT-LLM FP8 granularity: {granularity!r}")


def has_trtllm_fp8_packed_ue8m0() -> bool:
    """Return whether the optional TRT-LLM packed quantizer is importable.

    Returns:
        ``True`` when the vendored shared library and packed op are available.
    """
    return _trtllm_fp8_quantize_1x128_packed_ue8m0 is not error_fn


def supports_trtllm_fp8_packed_ue8m0(x: torch.Tensor) -> bool:
    """Check whether an input satisfies the SM100 packed-quantizer contract.

    Args:
        x: Candidate activation tensor.

    Returns:
        ``True`` when the op is present and ``x`` has the required device,
        dtype, shape, contiguity, and alignment.
    """
    return (
        _supports_trtllm_fp8_quant_packed is not None
        and _supports_trtllm_fp8_quant_packed(x)
    )


__all__ = [
    "has_trtllm_fp8_packed_ue8m0",
    "supports_trtllm_fp8_packed_ue8m0",
    "trtllm_fp8_packed_ue8m0",
    "trtllm_fp8_token_group_128",
    "trtllm_fp8_token",
    "trtllm_fp8_tensor",
]
