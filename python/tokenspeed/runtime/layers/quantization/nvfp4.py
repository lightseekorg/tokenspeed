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

"""NVFP4 quantization config for tokenspeed runtime (ModelOpt-produced checkpoints)."""

import logging
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)


class Nvfp4Config(QuantizationConfig):
    """Config class for NVFP4 quantization (ModelOpt-produced checkpoints)."""

    def __init__(
        self,
        kv_cache_quant_algo: str | None = None,
        group_size: int = 16,
        exclude_modules: list[str] | None = None,
    ) -> None:
        super().__init__(exclude_modules=exclude_modules)
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.group_size = group_size
        self.weight_block_size = None  # FP4 uses group_size, not weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100  # Blackwell required

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Nvfp4Config":
        kv_cache_quant_algo = None
        group_size = 16
        exclude_modules = []

        # Try flat format first (config.json quantization_config)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            kv_cache_quant_algo = config.get("kv_cache_quant_algo", "auto")
            group_size = config.get("group_size", 16)
            exclude_modules = config.get("ignore", [])
        else:
            # Fall back to nested format (hf_quant_config.json)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                kv_cache_quant_algo = quant_config.get("kv_cache_quant_algo", "auto")
                group_size = quant_config.get("group_size", 16)
                exclude_modules = quant_config.get("exclude_modules", [])
            except (ValueError, KeyError):
                raise ValueError(
                    "Cannot find quant_algo in the model quantization config."
                )

        if quant_method != "NVFP4":
            raise ValueError(f"Nvfp4Config only supports NVFP4, got {quant_method}")

        return cls(
            kv_cache_quant_algo=kv_cache_quant_algo,
            group_size=group_size,
            exclude_modules=exclude_modules,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        """Detect NVFP4 from hf_quant_config and override."""
        quant_algo = ""
        if isinstance(hf_quant_cfg, dict):
            quant_algo = hf_quant_cfg.get("quant_algo", "")
            if not quant_algo:
                q = hf_quant_cfg.get("quantization", {})
                if isinstance(q, dict):
                    quant_algo = q.get("quant_algo", "")
        if "NVFP4" in quant_algo.upper() or "FP4" in quant_algo.upper():
            return "nvfp4"
        # Fallback: user requested nvfp4 and the checkpoint was produced by ModelOpt.
        if user_quant == "nvfp4" and hf_quant_cfg.get("quant_method") == "modelopt":
            return "nvfp4"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []
