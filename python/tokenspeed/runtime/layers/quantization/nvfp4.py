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
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

logger = logging.getLogger(__name__)


class Nvfp4Config(QuantizationConfig):
    """Config class for NVFP4 quantization (ModelOpt-produced checkpoints)."""

    def __init__(
        self,
        kv_cache_quant_algo: str | None = None,
        group_size: int = 16,
        exclude_modules: list[str] | None = None,
        fp4_modules: list[str] | None = None,
        fp8_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.group_size = group_size
        self.exclude_modules = exclude_modules or []
        self.fp4_modules = self._normalize_targets(fp4_modules or [])
        self.fp8_modules = self._normalize_targets(fp8_modules or [])
        self.is_mixed_precision = bool(self.fp4_modules or self.fp8_modules)
        self.fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
        )
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
        fp4_modules = []
        fp8_modules = []
        quant_source = config

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
                quant_source = quant_config
            except (ValueError, KeyError):
                raise ValueError(
                    "Cannot find quant_algo in the model quantization config."
                )

        quant_method = quant_method.upper()
        if quant_method == "MIXED_PRECISION":
            fp4_group = cls._find_fp4_config_group(quant_source)
            if fp4_group is not None:
                group_size = fp4_group.get("weights", {}).get(
                    "group_size",
                    fp4_group.get("input_activations", {}).get(
                        "group_size", group_size
                    ),
                )
                fp4_modules = fp4_group.get("targets", [])
                fp8_modules = cls._get_config_group_targets(quant_source, num_bits=8)
            else:
                fp4_modules = cls._get_quantized_layer_targets(
                    quant_source, quant_algo="NVFP4"
                )
                fp8_modules = cls._get_quantized_layer_targets(
                    quant_source, quant_algo="FP8"
                )
                group_size = cls._get_first_quantized_layer_group_size(
                    quant_source, quant_algo="NVFP4", default=group_size
                )
            if not fp4_modules:
                raise ValueError(
                    "Nvfp4Config only supports mixed precision configs with an FP4 "
                    f"group, got {quant_method}"
                )
        elif quant_method != "NVFP4":
            raise ValueError(f"Nvfp4Config only supports NVFP4, got {quant_method}")

        return cls(
            kv_cache_quant_algo=kv_cache_quant_algo,
            group_size=group_size,
            exclude_modules=exclude_modules,
            fp4_modules=fp4_modules,
            fp8_modules=fp8_modules,
        )

    @staticmethod
    def _normalize_target(target: str) -> str:
        if target.startswith("backbone."):
            target = "model." + target[len("backbone.") :]
        return target

    @classmethod
    def _normalize_targets(cls, targets: list[str]) -> list[str]:
        return [cls._normalize_target(target) for target in targets]

    @staticmethod
    def _config_group_has_bits(group: dict[str, Any], num_bits: int) -> bool:
        weights = group.get("weights", {})
        activations = group.get("input_activations", {})
        return (
            weights.get("type") == "float"
            and weights.get("num_bits") == num_bits
            and activations.get("type") == "float"
            and activations.get("num_bits") == num_bits
        )

    @classmethod
    def _get_config_group_targets(
        cls, config: dict[str, Any], *, num_bits: int
    ) -> list[str]:
        config_groups = config.get("config_groups", {})
        if not isinstance(config_groups, dict):
            return []

        targets = []
        for group in config_groups.values():
            if isinstance(group, dict) and cls._config_group_has_bits(group, num_bits):
                targets.extend(group.get("targets", []))
        return targets

    @staticmethod
    def _get_quantized_layer_targets(
        config: dict[str, Any], *, quant_algo: str
    ) -> list[str]:
        quantized_layers = config.get("quantized_layers", {})
        if not isinstance(quantized_layers, dict):
            return []

        return [
            name
            for name, layer_config in quantized_layers.items()
            if isinstance(layer_config, dict)
            and layer_config.get("quant_algo", "").upper() == quant_algo
        ]

    @staticmethod
    def _get_first_quantized_layer_group_size(
        config: dict[str, Any], *, quant_algo: str, default: int
    ) -> int:
        quantized_layers = config.get("quantized_layers", {})
        if not isinstance(quantized_layers, dict):
            return default

        for layer_config in quantized_layers.values():
            if (
                isinstance(layer_config, dict)
                and layer_config.get("quant_algo", "").upper() == quant_algo
            ):
                return layer_config.get("group_size", default)
        return default

    @classmethod
    def _find_fp4_config_group(cls, config: dict[str, Any]) -> dict[str, Any] | None:
        config_groups = config.get("config_groups", {})
        if not isinstance(config_groups, dict):
            return None

        for group in config_groups.values():
            if not isinstance(group, dict):
                continue
            if cls._config_group_has_bits(group, 4):
                return group
        return None

    @staticmethod
    def _prefix_matches(prefix: str, targets: list[str]) -> bool:
        return any(
            prefix == target or prefix.startswith(f"{target}.") for target in targets
        )

    def is_layer_fp8(self, prefix: str) -> bool:
        return self._prefix_matches(prefix, self.fp8_modules)

    def is_layer_nvfp4(self, prefix: str) -> bool:
        if not self.is_mixed_precision:
            return True
        return self._prefix_matches(prefix, self.fp4_modules)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        """Detect NVFP4 from hf_quant_config and override."""
        quant_algo = ""
        if isinstance(hf_quant_cfg, dict):
            quant_method = hf_quant_cfg.get("quant_method", "").lower()
            if quant_method in {"modelopt_mixed", "modelopt_fp4"}:
                return quant_method
            quant_source = hf_quant_cfg.get("quantization", hf_quant_cfg)
            quant_algo = quant_source.get("quant_algo", "")
            if not quant_algo:
                q = hf_quant_cfg.get("quantization", {})
                if isinstance(q, dict):
                    quant_algo = q.get("quant_algo", "")
            if quant_algo.upper() == "MIXED_PRECISION":
                has_fp4_group = cls._find_fp4_config_group(quant_source) is not None
                has_fp4_layers = bool(
                    cls._get_quantized_layer_targets(quant_source, quant_algo="NVFP4")
                )
                if has_fp4_group or has_fp4_layers:
                    return "modelopt_mixed"
        if "NVFP4" in quant_algo.upper() or "FP4" in quant_algo.upper():
            if user_quant in {"modelopt", "modelopt_fp4"}:
                return "modelopt_fp4"
            return "nvfp4"
        # Fallback: user requested nvfp4 and the checkpoint was produced by ModelOpt.
        if (
            user_quant in {"nvfp4", "modelopt", "modelopt_fp4"}
            and hf_quant_cfg.get("quant_method") == "modelopt"
        ):
            return "nvfp4"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []

    def is_layer_excluded(self, prefix: str) -> bool:
        """Check if a layer should be excluded from FP4 quantization."""
        import re

        for pattern in self.exclude_modules:
            regex_str = pattern.replace(".", r"\.").replace("*", ".*")
            if re.fullmatch(regex_str, prefix):
                return True
        return False
