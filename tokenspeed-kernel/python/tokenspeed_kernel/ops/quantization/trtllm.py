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

import importlib.util

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()
_HAS_TRTLLM_KERNEL = (
    platform.is_nvidia
    and importlib.util.find_spec("trtllm_kernel") is not None
    and importlib.util.find_spec("deep_ep") is not None
)
__all__ = []


def _per_token_group_quant_8bit(
    x: torch.Tensor, group_size: int, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRTLLM_KERNEL:
        raise RuntimeError(
            "TRT-LLM FP8 quantization requires a compatible trtllm_kernel package"
        )
    if group_size != 128:
        raise ValueError(
            "trtllm fp8_quantize_1x128 only supports group_size=128, "
            f"got {group_size}"
        )
    q, scale = torch.ops.trtllm.fp8_quantize_1x128(x, use_ue8m0)
    if scale.ndim == 1:
        # The flat-buffer ABI uses 4-row alignment and trailing padding.
        rows, columns = x.shape
        padded_rows = (rows + 3) // 4 * 4
        groups = columns // group_size
        scale = scale[: groups * padded_rows].view(groups, padded_rows)[:, :rows]
    return q, scale


if _HAS_TRTLLM_KERNEL:
    # DeepEP must initialize before TRT-LLM's static CUDA runtime.
    import deep_ep  # noqa: F401
    import trtllm_kernel  # noqa: F401

    _FP8_DTYPE = torch.float8_e4m3fn

    def _per_token_quant_fp8(
        input: torch.Tensor, output: torch.Tensor, scale: torch.Tensor
    ) -> None:
        q, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(input)
        output.copy_(q)
        scale.copy_(s.float().squeeze(-1))

    def trtllm_fp8_token_group_128(x: torch.Tensor) -> torch.Tensor:
        qweight, _scale = _per_token_group_quant_8bit(
            x, group_size=128, use_ue8m0=False
        )
        return qweight.float()

    def trtllm_fp8_token(x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x, dtype=_FP8_DTYPE)
        scale = torch.empty(x.size(0), dtype=torch.float32, device=x.device)
        _per_token_quant_fp8(x, output, scale)
        return output.float()

    @register_kernel(
        "quantization",
        "fp8_with_scale",
        name="trtllm_quantize_fp8_with_scale",
        features={"token_fp8_min_scaling"},
        solution="trtllm",
        capability=CapabilityRequirement(
            max_arch_version=ArchVersion(10, 9),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "granularity": frozenset({"token", "token_group_128"}),
            "scale_encoding": frozenset({"float32", "ue8m0"}),
        },
        priority=Priority.PERFORMANT,
    )
    def trtllm_quantize_fp8_with_scale(
        x: torch.Tensor,
        granularity: str = "token",
        group_size: int | None = None,
        scale_encoding: str = "float32",
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if granularity == "token":
            if scale_encoding != "float32":
                raise ValueError("TRT-LLM token FP8 requires float32 scales")

            # The op's own codes and its scales as FP32 [..., 1]: one launch,
            # no copies into new buffers (the scales are reshaped, or
            # converted only if the op wrote another dtype).
            q, scale = torch.ops.tensorrt_llm.quantize_e4m3_activation(x)
            if q.dtype != _FP8_DTYPE:
                raise ValueError(f"TRT-LLM token FP8 wrote {q.dtype}, not E4M3")
            return q.reshape(x.shape), scale.float().reshape(*x.shape[:-1], 1)

        if granularity == "token_group":
            q, scale = _per_token_group_quant_8bit(
                x,
                group_size=group_size,
                use_ue8m0=scale_encoding == "ue8m0",
            )
            return q, scale.t().contiguous()

        raise ValueError(f"unsupported TRT-LLM FP8 granularity: {granularity!r}")

    @register_kernel(
        "quantization",
        "fp8_with_scale",
        name="trtllm_quantize_fp8_mn_major",
        solution="trtllm",
        capability=CapabilityRequirement(
            max_arch_version=ArchVersion(10, 9),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "granularity": frozenset({"token_group_128"}),
            "scale_encoding": frozenset({"float32"}),
            "scale_layout": frozenset({"mn_major"}),
        },
        features=frozenset({"mn_major_scales"}),
        # A solution override (override="trtllm") skips trait matching and
        # takes the solution's highest priority; the lowest band keeps it on
        # trtllm_quantize_fp8_with_scale's contiguous scales.
        priority=Priority.REFERENCE,
    )
    def trtllm_quantize_fp8_mn_major(
        x: torch.Tensor,
        *,
        granularity: str,
        group_size: int,
        scale_encoding: str,
        enable_pdl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP8 values and a logical [M,K/128] view of native scales."""
        if granularity != "token_group" or scale_encoding != "float32":
            raise ValueError(
                "MN-major FP8 quantization requires FP32 token-group scales"
            )
        if enable_pdl:
            raise ValueError("TRT-LLM FP8 quantization does not expose PDL control")
        q, scale = _per_token_group_quant_8bit(x, group_size, use_ue8m0=False)
        # Preserve the native M stride; padding depends on the TRT-LLM ABI.
        # Consumers may still need to align or repack these scales.
        return q, scale.t()

    __all__ = [
        "trtllm_fp8_token_group_128",
        "trtllm_fp8_token",
    ]
else:
    # Keep FlashInfer's existing Triton fallback available.
    _per_token_group_quant_8bit = error_fn
