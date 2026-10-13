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

"""ModelOpt single-algorithm MXFP8 checkpoints (``quant_algo: MXFP8``).

Every quantized linear stores an E4M3 ``weight`` with one E8M0
``weight_scale`` byte per 32 input channels. Routed experts stay MXFP8 and run
on a weight-only MXFP8 MoE kernel (Marlin on Hopper, which has no MXFP8 tensor
cores). Dense projections are dequantized to bf16 while loading: an E4M3 value
times a power of two is exact in bf16, so a bf16 GEMM over the dequantized
weight computes the same products as a weight-only MXFP8 GEMM and differs only
in accumulation order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config

MXFP8_BLOCK_SIZE = 32
_E8M0_NAN = 0xFF


def _quantization_section(config: dict[str, Any]) -> dict[str, Any]:
    # hf_quant_config.json nests under "quantization"; config.json's
    # quantization_config may be flat.
    section = config.get("quantization", config)
    return section if isinstance(section, dict) else config


class ModelOptMxfp8Config(QuantizationConfig):
    """Config for ModelOpt checkpoints quantized with ``quant_algo: MXFP8``.

    Args:
        exclude_modules: Checkpoint module globs left unquantized.
        kv_cache_quant_algo: Must be None; the format carries no KV-cache
            scales.
    """

    def __init__(
        self, exclude_modules: list[str], kv_cache_quant_algo: str | None
    ) -> None:
        super().__init__(exclude_modules=exclude_modules)
        if kv_cache_quant_algo is not None:
            raise ValueError(
                "ModelOpt MXFP8 checkpoints with KV-cache quantization "
                f"({kv_cache_quant_algo!r}) are not supported"
            )
        self.moe_config = Mxfp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, MXFP8_BLOCK_SIZE],
            scale_fmt="ue8m0",
        )

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelOptMxfp8Config":
        section = _quantization_section(config)
        quant_algo = section.get("quant_algo")
        if quant_algo != "MXFP8":
            raise ValueError(
                f"ModelOptMxfp8Config only supports MXFP8, got {quant_algo!r}"
            )
        return cls(
            exclude_modules=list(
                section.get("exclude_modules") or section.get("ignore") or []
            ),
            kv_cache_quant_algo=section.get("kv_cache_quant_algo"),
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if not isinstance(hf_quant_cfg, dict):
            return None
        if hf_quant_cfg.get("quant_method", "modelopt") != "modelopt":
            return None
        if _quantization_section(hf_quant_cfg).get("quant_algo") == "MXFP8":
            return cls.get_name()
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase:
        from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod

        # Dense weights arrive in bf16 from preprocess_modelopt_mxfp8_weights.
        return UnquantizedLinearMethod()

    def get_moe_quant_config(self, prefix: str) -> QuantizationConfig:
        return self.moe_config

    def moe_weight_dtype(self, prefix: str = "") -> str:
        return self.moe_config.moe_weight_dtype(prefix)

    def get_scaled_act_names(self) -> list[str]:
        return []


def dequantize_mxfp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a 2-D MXFP8 weight ``[n, k]`` with E8M0 scales ``[n, k/32]``
    to bf16. Runs on the current CUDA device."""
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected a float8_e4m3fn MXFP8 weight, got {weight.dtype}")
    if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError(f"expected E8M0 (uint8) MXFP8 scales, got {scale.dtype}")
    if weight.dim() != 2 or weight.shape[1] % MXFP8_BLOCK_SIZE:
        raise ValueError(
            f"MXFP8 weight must be 2-D with K a multiple of {MXFP8_BLOCK_SIZE}, "
            f"got {tuple(weight.shape)}"
        )
    n, k = weight.shape
    if tuple(scale.shape) != (n, k // MXFP8_BLOCK_SIZE):
        raise ValueError(
            f"MXFP8 scale shape {tuple(scale.shape)} does not match weight "
            f"{tuple(weight.shape)} with 1x{MXFP8_BLOCK_SIZE} blocks"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    weight = weight.to(device)
    scale = scale.to(device).view(torch.uint8)
    if bool((scale == _E8M0_NAN).any()):
        raise ValueError("MXFP8 scales contain the E8M0 NaN encoding (0xFF)")
    factor = torch.exp2(scale.to(torch.float32) - 127.0).to(torch.bfloat16)
    blocks = weight.to(torch.bfloat16).view(n, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (blocks * factor.unsqueeze(-1)).view(n, k)


def preprocess_modelopt_mxfp8_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: QuantizationConfig | None,
    *,
    keeps_mxfp8: Callable[[str], bool],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dequantize dense MXFP8 checkpoint tensors to bf16 while streaming.

    Every FP8 ``<module>.weight`` is paired with its ``<module>.weight_scale``
    and yielded once as a bf16 ``<module>.weight``. Tensors for which
    ``keeps_mxfp8(name)`` is true (the routed experts) pass through untouched
    for the MXFP8 MoE loader, as does every 16/32-bit tensor (unquantized
    modules and bf16 refit streams). Only modules waiting for their partner
    are buffered.
    """
    if not isinstance(quant_config, ModelOptMxfp8Config):
        yield from weights
        return

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        is_scale = name.endswith(".weight_scale")
        is_fp8_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
        if not (is_scale or is_fp8_weight) or keeps_mxfp8(name):
            yield name, tensor
            continue
        module = name.rsplit(".", 1)[0]
        entry = pending.setdefault(module, {})
        entry["scale" if is_scale else "weight"] = tensor
        if len(entry) == 2:
            del pending[module]
            yield (
                module + ".weight",
                dequantize_mxfp8_weight(entry["weight"], entry["scale"]),
            )
    if pending:
        raise RuntimeError(
            "MXFP8 modules missing their weight/weight_scale pair at the end "
            f"of the checkpoint stream: {sorted(pending)}"
        )
