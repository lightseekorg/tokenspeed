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

import re
from collections.abc import Mapping
from typing import Any

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig


def _is_fp4_e8m0_per_group(stage: object, *, is_dynamic: bool | None = None) -> bool:
    if not isinstance(stage, Mapping):
        return False
    if is_dynamic is not None and stage.get("is_dynamic") is not is_dynamic:
        return False
    return (
        str(stage.get("dtype", "")).lower() in {"fp4", "mxfp4"}
        and str(stage.get("qscheme", "")).lower() == "per_group"
        and stage.get("group_size") in {32, "32"}
        and str(stage.get("scale_format", "")).lower() == "e8m0"
    )


def _is_amd_quark_w_mxfp4_a_fp8(config: Mapping[str, Any]) -> bool:
    if not isinstance(config, Mapping):
        return False
    if not current_platform().is_amd:
        return False
    if str(config.get("quant_method", "")).lower() != "quark":
        return False
    global_quant_config = config.get("global_quant_config") or {}
    export = config.get("export") or {}
    if not isinstance(global_quant_config, Mapping) or not isinstance(export, Mapping):
        return False
    input_tensors = global_quant_config.get("input_tensors") or {}
    weight = global_quant_config.get("weight") or {}
    return (
        isinstance(input_tensors, Mapping)
        and "fp8" in str(input_tensors.get("dtype", "")).lower()
        and _is_fp4_e8m0_per_group(weight, is_dynamic=False)
        and str(export.get("pack_method", "")).lower() == "reorder"
        and str(export.get("weight_format", "")).lower() == "real_quantized"
    )


def _is_amd_quark_dynamic_mxfp4(config: Mapping[str, Any]) -> bool:
    if not isinstance(config, Mapping):
        return False
    if not current_platform().is_amd:
        return False
    if str(config.get("quant_method", "")).lower() != "quark":
        return False
    global_quant_config = config.get("global_quant_config") or {}
    export = config.get("export") or {}
    if not isinstance(global_quant_config, Mapping) or not isinstance(export, Mapping):
        return False
    input_tensors = global_quant_config.get("input_tensors") or {}
    weight = global_quant_config.get("weight") or {}
    return (
        _is_fp4_e8m0_per_group(input_tensors, is_dynamic=True)
        and _is_fp4_e8m0_per_group(weight, is_dynamic=False)
        and str(export.get("pack_method", "")).lower() == "reorder"
        and str(export.get("weight_format", "")).lower() == "real_quantized"
    )


def _is_amd_quark_mxfp4_checkpoint(config: dict) -> bool:
    if not isinstance(config, Mapping):
        return False
    return _is_amd_quark_w_mxfp4_a_fp8(config) or _is_amd_quark_dynamic_mxfp4(config)


def _iter_ignored_layer_pattern_aliases(raw: str):
    yield raw
    if raw.startswith("language_model."):
        yield raw.removeprefix("language_model.")
        return

    if "model.language_model." in raw:
        yield raw.replace("model.language_model.", "model.")
        return

    if raw.startswith("re:"):
        regex = raw[3:]
        for prefix in ("language_model.", re.escape("language_model.")):
            if regex.startswith(prefix):
                yield f"re:{regex.removeprefix(prefix)}"
                return


def _to_ignore_pattern(raw: str) -> str:
    if raw.startswith("re:") or "*" not in raw:
        return raw
    regex = re.escape(raw).replace(r"\*", ".*")
    return f"re:{regex}"


def _normalize_ignored_layer_patterns(patterns: list[str] | None) -> list[str]:
    """Normalize ignored-layer patterns into the form understood by
    ``should_ignore_quant_layer``.

    Some exporters (notably AMD-Quark) accept shell-style globs such as
    ``"*lm_head"`` or ``"*self_attn*"``. ``should_ignore_quant_layer``
    expects either an exact name or a regex prefixed with ``re:``. Convert
    glob-like entries to regex while passing through plain literals.
    """
    if not patterns:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in patterns:
        if not isinstance(raw, str) or not raw:
            continue
        for alias in _iter_ignored_layer_pattern_aliases(raw):
            pattern = _to_ignore_pattern(alias)
            if pattern in seen:
                continue
            seen.add(pattern)
            normalized.append(pattern)
    return normalized


class Mxfp4Config(QuantizationConfig):

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        is_checkpoint_mxfp4_serialized: bool = False,
        is_w4a8_fp8: bool = False,
        use_dynamic_mxfp4_activations: bool = False,
    ):
        super().__init__(ignored_layers=ignored_layers)
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.is_w4a8_fp8 = is_w4a8_fp8
        self.use_dynamic_mxfp4_activations = use_dynamic_mxfp4_activations
        self.group_size = 32

    @classmethod
    def from_config(cls, config):
        quant_method = str(config.get("quant_method", "")).lower()
        is_w4a8_fp8 = _is_amd_quark_w_mxfp4_a_fp8(config)
        use_dynamic_mxfp4_activations = _is_amd_quark_dynamic_mxfp4(config)
        is_checkpoint_mxfp4_serialized = (
            "mxfp4" in quant_method or is_w4a8_fp8 or use_dynamic_mxfp4_activations
        )

        raw_ignored = cls.get_from_keys_or(config, ["ignored_layers", "exclude"], None)
        ignored_layers = _normalize_ignored_layer_patterns(raw_ignored)

        return cls(
            ignored_layers=ignored_layers,
            is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized,
            is_w4a8_fp8=is_w4a8_fp8,
            use_dynamic_mxfp4_activations=use_dynamic_mxfp4_activations,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        """Promote AMD Quark MXFP4 checkpoint metadata to mxfp4."""
        if user_quant in {"mxfp4", None} and _is_amd_quark_mxfp4_checkpoint(
            hf_quant_cfg
        ):
            return "mxfp4"
        return None

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def is_static_cfg(self):
        return self.is_checkpoint_mxfp4_serialized

    def get_scaled_act_names(self) -> list[str]:
        return []
