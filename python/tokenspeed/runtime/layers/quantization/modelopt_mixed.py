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

"""Per-layer mixed-precision quantization for ModelOpt MIXED_PRECISION exports.

Such checkpoints (e.g. nvidia/MiniMax-M3-NVFP4) declare a per-module
``quant_algo`` in the ``quantized_layers`` dict of their quantization config,
plus ``exclude_modules`` for unquantized modules. This config resolves each
runtime layer to the matching algorithm and delegates weight handling to the
corresponding single-algorithm config (MXFP8, NVFP4).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.layers.quantization.utils import should_exclude_quant_module

logger = logging.getLogger(__name__)

_MOE_WEIGHT_DTYPES = {
    "NVFP4": "nvfp4",
    "MXFP8": "fp8",
}

_FUSED_PROJECTION_SHARDS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


class ModelOptMixedConfig(QuantizationConfig):
    """Config for ModelOpt MIXED_PRECISION checkpoints.

    Args:
        quantized_layers: Checkpoint module name -> upper-cased quant_algo
            (``"NVFP4"`` or ``"MXFP8"``). Keys use checkpoint naming until
            :meth:`apply_checkpoint_name_replacements` rewrites them to
            runtime naming.
        exclude_modules: Checkpoint module names left unquantized.
        kv_cache_quant_algo: KV-cache quantization algorithm, if any.
        group_size: NVFP4 weight group size.
    """

    def __init__(
        self,
        quantized_layers: dict[str, str],
        exclude_modules: list[str] | None = None,
        kv_cache_quant_algo: str | None = None,
        group_size: int = 16,
    ) -> None:
        super().__init__(exclude_modules=exclude_modules)
        self.quantized_layers = quantized_layers
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.group_size = group_size
        self.mxfp8_config = Mxfp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            scale_fmt="ue8m0",
        )
        self.nvfp4_config = Nvfp4Config(
            kv_cache_quant_algo=kv_cache_quant_algo,
            group_size=group_size,
        )
        # MoE layers read this when their experts resolve to an fp8 algo.
        self.weight_block_size = self.mxfp8_config.weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_mixed"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100  # NVFP4 members require Blackwell

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @staticmethod
    def _quantization_section(config: dict[str, Any]) -> dict[str, Any]:
        # hf_quant_config.json nests under "quantization"; config.json's
        # quantization_config is flat.
        section = config.get("quantization", config)
        return section if isinstance(section, dict) else config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelOptMixedConfig":
        section = cls._quantization_section(config)
        quant_algo = section.get("quant_algo", "")
        if quant_algo != "MIXED_PRECISION":
            raise ValueError(
                f"ModelOptMixedConfig only supports MIXED_PRECISION, got {quant_algo!r}"
            )
        raw_layers = section.get("quantized_layers", {})
        if not raw_layers:
            raise ValueError(
                "MIXED_PRECISION requires a non-empty 'quantized_layers' "
                "mapping in the quantization config."
            )

        quantized_layers: dict[str, str] = {}
        group_size: int | None = None
        unknown: set[str] = set()
        for name, info in raw_layers.items():
            algo = str(info.get("quant_algo", "")).upper()
            if algo not in _MOE_WEIGHT_DTYPES:
                unknown.add(algo)
                continue
            quantized_layers[name] = algo
            if algo == "NVFP4" and group_size is None:
                group_size = int(info.get("group_size", 16))
        if unknown:
            raise ValueError(
                f"Unsupported quant_algo values in quantized_layers: {sorted(unknown)}. "
                f"Supported: {sorted(_MOE_WEIGHT_DTYPES)}."
            )

        return cls(
            quantized_layers=quantized_layers,
            exclude_modules=section.get("exclude_modules", []),
            kv_cache_quant_algo=section.get("kv_cache_quant_algo"),
            group_size=group_size if group_size is not None else 16,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if not isinstance(hf_quant_cfg, dict):
            return None
        section = cls._quantization_section(hf_quant_cfg)
        if section.get("quant_algo") == "MIXED_PRECISION":
            return "modelopt_mixed"
        return None

    def apply_checkpoint_name_replacements(
        self, replacements: tuple[tuple[str, str], ...]
    ) -> None:
        """Rewrite checkpoint module names to runtime module prefixes.

        ``replacements`` is the model's ordered (old, new) substring table
        (``quant_module_name_replacements``). After this, layer lookups are
        direct string matches against construction-time prefixes.
        """

        def rename(name: str) -> str:
            for old, new in replacements:
                name = name.replace(old, new)
            return name

        self.quantized_layers = {
            rename(name): algo for name, algo in self.quantized_layers.items()
        }
        self.exclude_modules = [rename(name) for name in self.exclude_modules]

    def _resolve_quant_algo(self, prefix: str) -> str | None:
        """Resolve the quant_algo for a runtime module prefix.

        Lookup order: direct hit; fused-projection unfuse (members must
        agree); child-prefix scan (a parent module such as fused experts
        matches its children's entries).
        """
        if prefix in self.quantized_layers:
            return self.quantized_layers[prefix]

        leaf = prefix.rsplit(".", 1)[-1]
        shards = _FUSED_PROJECTION_SHARDS.get(leaf)
        if shards is not None:
            base = prefix.rsplit(".", 1)[0]
            algos = {
                self.quantized_layers[f"{base}.{shard}"]
                for shard in shards
                if f"{base}.{shard}" in self.quantized_layers
            }
            if len(algos) > 1:
                raise ValueError(
                    f"Mixed quant_algo within fused layer {prefix}: {sorted(algos)}. "
                    "All members must use the same quantization."
                )
            if algos:
                return algos.pop()

        child_prefix = prefix + "."
        child_algos = {
            algo
            for name, algo in self.quantized_layers.items()
            if name.startswith(child_prefix)
        }
        if len(child_algos) > 1:
            raise ValueError(
                f"Module {prefix} has children with mixed quant_algo "
                f"{sorted(child_algos)}; resolve a more specific prefix."
            )
        if child_algos:
            return child_algos.pop()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase:
        from tokenspeed.runtime.layers.dense import (
            Fp8LinearMethod,
            Nvfp4LinearMethod,
            UnquantizedLinearMethod,
        )

        if should_exclude_quant_module(prefix, self.exclude_modules):
            return UnquantizedLinearMethod()
        algo = self._resolve_quant_algo(prefix)
        if algo is None:
            return UnquantizedLinearMethod()
        if algo == "MXFP8":
            return Fp8LinearMethod(self.mxfp8_config)
        if algo == "NVFP4":
            return Nvfp4LinearMethod(self.nvfp4_config)
        raise ValueError(f"Unsupported quant_algo {algo!r} for layer {prefix!r}")

    def moe_weight_dtype(self, prefix: str = "") -> str:
        # Prefer the experts subtree: a MoE block prefix (e.g. "...mlp") can
        # also contain differently-quantized shared experts.
        candidates = (
            (prefix,) if prefix.endswith(".experts") else (f"{prefix}.experts", prefix)
        )
        for candidate in candidates:
            algo = self._resolve_quant_algo(candidate)
            if algo is not None:
                return _MOE_WEIGHT_DTYPES[algo]
        raise ValueError(
            f"No quantized_layers entry resolves the MoE prefix {prefix!r}; "
            "cannot infer the experts' weight dtype."
        )

    def get_scaled_act_names(self) -> list[str]:
        return []
