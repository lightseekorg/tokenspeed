# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

fp4_quantize = error_fn
flashinfer_quantize_mxfp8 = error_fn
flashinfer_quantize_nvfp4 = error_fn
nvfp4_quantize_per_token_activation = error_fn
mxfp8_quantize = error_fn
nvfp4_block_scale_interleave = error_fn
prepare_nvfp4_weight_for_fp4_gemm = error_fn
fp8_blockscale_quantize_runner_sm90 = error_fn

if platform.is_nvidia:
    from flashinfer import mxfp8_quantize

    if platform.is_hopper:
        from flashinfer.gemm.gemm_base import (
            get_fp8_blockscale_gemm_runner_sm90 as fp8_blockscale_quantize_runner_sm90,
        )

    @register_kernel(
        "quantization",
        "mxfp8",
        name="flashinfer_quantize_mxfp8",
        solution="flashinfer",
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={},
        priority=Priority.PERFORMANT,
    )
    def flashinfer_quantize_mxfp8(
        x: torch.Tensor,
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mxfp8_quantize(x, False)


if platform.is_nvidia and platform.is_blackwell:
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize
    from flashinfer import (
        nvfp4_block_scale_interleave,
    )

    try:
        from flashinfer import SfLayout as _FlashInferSfLayout
        from flashinfer import nvfp4_quantize as _flashinfer_nvfp4_quantize
    except ImportError:
        _FlashInferSfLayout = None
        _flashinfer_nvfp4_quantize = None

    _FLASHINFER_FP4_QUANTIZE_BACKEND = (
        "cute-dsl" if platform.arch_version.major == 10 else "cuda"
    )

    def fp4_quantize(
        input: torch.Tensor,
        global_scale: torch.Tensor | float | None = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: bool | None = None,
        backend: str = _FLASHINFER_FP4_QUANTIZE_BACKEND,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _flashinfer_fp4_quantize(
            input=input,
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
            backend=backend,
        )

    @register_kernel(
        "quantization",
        "nvfp4",
        name="flashinfer_quantize_nvfp4",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        traits={
            "has_scale": frozenset({True}),
        },
        priority=Priority.PERFORMANT,
    )
    def flashinfer_quantize_nvfp4(
        x: torch.Tensor,
        scale: float | torch.Tensor | None = None,
        scale_layout: str = "swizzled",
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The public quantization API uses the actual scale; FlashInfer's FP4
        # helper expects the inverse scale used before packing.
        scale_inv = 1.0 / scale
        return fp4_quantize(
            x,
            global_scale=scale_inv,
            sf_vec_size=16,
            is_sf_swizzled_layout=scale_layout == "swizzled",
            enable_pdl=enable_pdl,
            backend=_FLASHINFER_FP4_QUANTIZE_BACKEND,
        )

    if _flashinfer_nvfp4_quantize is not None:

        def nvfp4_quantize_per_token_activation(
            input: torch.Tensor,
            global_scale: float = 1.0 / (448.0 * 6.0),
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _flashinfer_nvfp4_quantize(
                input,
                global_scale,
                sfLayout=_FlashInferSfLayout.layout_linear,
                per_token_activation=True,
            )

    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def prepare_nvfp4_weight_for_fp4_gemm(
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        n_alignment: int = 32,
        k_alignment: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Pad NVFP4 checkpoint weights for FlashInfer ``mm_fp4``.

        Returns ``(weight, scale, output_size, weight_k_padding_cols)`` where
        ``output_size`` is the original unpadded N dimension.
        """
        output_size = weight.shape[0]
        weight_col_elements = weight.shape[1] * 2

        padded_n = _round_up_to_multiple(output_size, n_alignment)
        padded_k = _round_up_to_multiple(weight_col_elements, k_alignment)
        weights_padding_cols = (padded_k - weight_col_elements) // 2
        if padded_n != output_size:
            pad_n = padded_n - output_size
        else:
            pad_n = 0

        if pad_n or weights_padding_cols:
            weight = torch.nn.functional.pad(
                weight, (0, weights_padding_cols, 0, pad_n)
            ).contiguous()

        scale_ndim = scale.ndim
        if scale_ndim == 2:
            scale = scale.unsqueeze(0)
        assert scale.ndim == 3
        batch, rows, cols = scale.shape
        padded_rows = _round_up_to_multiple(rows, 128)
        padded_cols = _round_up_to_multiple(cols, 4)
        padded_scale = torch.zeros(
            (batch, padded_rows, padded_cols),
            dtype=scale.dtype,
            device=scale.device,
        )
        padded_scale[:, :rows, :cols] = scale
        padded_scale = padded_scale.reshape(
            batch,
            padded_rows // 128,
            4,
            32,
            padded_cols // 4,
            4,
        )
        padded_scale = padded_scale.permute(0, 1, 4, 3, 2, 5).contiguous()
        scale = (
            padded_scale.reshape(padded_rows, padded_cols)
            if scale_ndim == 2
            else padded_scale.reshape(batch, padded_rows, padded_cols)
        )
        return weight, scale, output_size, weights_padding_cols


__all__ = [
    "fp4_quantize",
    "mxfp8_quantize",
    "nvfp4_quantize_per_token_activation",
    "nvfp4_block_scale_interleave",
    "prepare_nvfp4_weight_for_fp4_gemm",
    "fp8_blockscale_quantize_runner_sm90",
]
