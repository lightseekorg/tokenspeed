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

"""MXFP8 projections with NVFP4 routed experts.

Such checkpoints declare ``quant_method: mxfp8`` plus
``moe_quant_algo: NVFP4``. Every projection loads as MXFP8; the routed experts
are NVFP4 in the compressed-tensors ``nvfp4-pack-quantized`` layout, which
:func:`preprocess_mxfp8_nvfp4_weights` renames to the runtime NVFP4 tensors.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config

# compressed-tensors stores the NVFP4 global scales as quantization
# multipliers; the runtime loads their reciprocals under ModelOpt's names.
_EXPERT_GLOBAL_SCALE_NAMES = {
    "weight_global_scale": "weight_scale_2",
    "input_global_scale": "input_scale",
}


class Mxfp8Nvfp4Config(Mxfp8Config):
    """MXFP8 config whose routed experts resolve to NVFP4.

    Args:
        activation_scheme: MXFP8 activation quantization scheme.
        ignored_layers: Checkpoint module names left unquantized.
        weight_block_size: MXFP8 weight scale block.
        scale_fmt: MXFP8 scale encoding.
        group_size: NVFP4 expert weight group size.
    """

    def __init__(
        self,
        activation_scheme: str,
        ignored_layers: list[str] | None,
        weight_block_size: list[int],
        scale_fmt: str,
        group_size: int,
    ) -> None:
        super().__init__(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            scale_fmt=scale_fmt,
        )
        self.nvfp4_config = Nvfp4Config(group_size=group_size)

    @classmethod
    def get_name(cls) -> str:
        return "mxfp8_nvfp4"

    @classmethod
    def get_min_capability(cls) -> int:
        return 100  # NVFP4 experts require Blackwell

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Mxfp8Nvfp4Config:
        moe_quant_format = cls.get_from_keys(config, ["moe_quant_format"])
        if moe_quant_format != "nvfp4-pack-quantized":
            raise ValueError(
                f"Unsupported NVFP4 expert format {moe_quant_format!r}; "
                "expected 'nvfp4-pack-quantized'."
            )
        return cls(
            activation_scheme=cls.get_from_keys(config, ["activation_scheme"]),
            ignored_layers=cls.get_from_keys_or(
                config, ["ignored_layers", "modules_to_not_convert"], None
            ),
            weight_block_size=cls.get_from_keys(config, ["weight_block_size"]),
            # MXFP8 scales are UE8M0 when the checkpoint does not say.
            scale_fmt=config.get("scale_fmt") or "ue8m0",
            group_size=cls.get_from_keys(config, ["group_size"]),
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") == "mxfp8"
            and hf_quant_cfg.get("moe_quant_algo") == "NVFP4"
            and user_quant in (None, "mxfp8", cls.get_name())
        ):
            return cls.get_name()
        return None

    def get_moe_quant_config(self, prefix: str) -> QuantizationConfig:
        return self.nvfp4_config

    def moe_weight_dtype(self, prefix: str = "") -> str:
        return self.nvfp4_config.moe_weight_dtype(prefix)


def preprocess_mxfp8_nvfp4_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: QuantizationConfig | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Rename compressed-tensors NVFP4 expert tensors to the runtime names.

    ``weight_packed`` becomes ``weight`` and each global scale becomes its
    reciprocal under the ModelOpt name. Block scales and non-expert tensors
    pass through unchanged, as does every tensor of other quant configs.
    """
    if not isinstance(quant_config, Mxfp8Nvfp4Config):
        yield from weights
        return

    for name, weight in weights:
        if ".experts." in name:
            module, _, leaf = name.rpartition(".")
            if leaf == "weight_packed":
                name = f"{module}.weight"
            elif leaf in _EXPERT_GLOBAL_SCALE_NAMES:
                name = f"{module}.{_EXPERT_GLOBAL_SCALE_NAMES[leaf]}"
                weight = weight.to(torch.float32).reciprocal()
        yield name, weight
