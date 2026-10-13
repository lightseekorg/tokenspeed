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
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer
from tokenspeed.runtime.layers.quantization.w8a8_fp8 import W8A8Fp8Config


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


# Raw-consumed attention weights: model code reads these as BF16 tensors
# (Kimi-K3's KDA backend GEMV reads f_b_proj; MLA absorbs kv_b_proj into
# w_kc/w_vc after load), so FP8 checkpoints dequantize them during load.
_FP8_OVERRIDE_DEQUANT_LEAVES = frozenset({"f_b_proj", "kv_b_proj"})


def _is_fp8_per_channel_w8a8_spec(spec: object) -> bool:
    """Whether a per-layer override is FP8 W8A8 with per-channel weight scales.

    Weights are static FP8 E4M3 with one scale per output channel; activations
    are quantized to FP8 per token at runtime, which the config labels
    ``per_channel`` on the token axis.
    """
    if not isinstance(spec, Mapping):
        return False
    weight = spec.get("weight") or {}
    inputs = spec.get("input_tensors") or {}
    if not isinstance(weight, Mapping) or not isinstance(inputs, Mapping):
        return False
    return (
        str(weight.get("dtype", "")).lower() == "fp8_e4m3"
        and weight.get("is_dynamic") is False
        and str(weight.get("qscheme", "")).lower() == "per_channel"
        and weight.get("ch_axis") in {0, "0"}
        and str(inputs.get("dtype", "")).lower() == "fp8_e4m3"
        and inputs.get("is_dynamic") is True
        and str(inputs.get("qscheme", "")).lower() == "per_channel"
        and spec.get("output_tensors") is None
    )


def _fp8_override_layer_patterns(config: Mapping[str, Any]) -> list[str]:
    """Collect the module globs of the checkpoint's per-layer FP8 overrides.

    ``layer_quant_config`` maps module globs (e.g. ``*self_attn*``) to a scheme
    that replaces the global MXFP4 one for matching modules. Only FP8 W8A8 with
    per-channel weight scales is supported; any other scheme would otherwise
    silently load as MXFP4, so it is rejected. The globs are normalized like
    ignored-layer patterns.
    """
    layer_quant_config = config.get("layer_quant_config") or {}
    if not isinstance(layer_quant_config, Mapping):
        raise ValueError("layer_quant_config must map module globs to schemes")
    patterns: list[str] = []
    for glob, spec in layer_quant_config.items():
        if not _is_fp8_per_channel_w8a8_spec(spec):
            raise ValueError(
                f"Unsupported per-layer quantization scheme for {glob!r}; only "
                "static per-channel FP8 E4M3 weights with dynamic per-token FP8 "
                "activations are supported"
            )
        patterns.append(glob)
    return _normalize_ignored_layer_patterns(patterns)


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

    Some checkpoint exporters write shell-style globs such as
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
        quant_method: str | None = None,
        fp8_layer_patterns: list[str] | None = None,
    ):
        super().__init__(ignored_layers=ignored_layers)
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.is_w4a8_fp8 = is_w4a8_fp8
        self.use_dynamic_mxfp4_activations = use_dynamic_mxfp4_activations
        self.quant_method = quant_method
        self.group_size = 32
        # Per-layer FP8 overrides from the checkpoint (e.g. ``*self_attn*``),
        # matched like ignored-layer patterns against runtime and checkpoint
        # module names.
        self.fp8_layer_patterns = fp8_layer_patterns or []
        self.fp8_config: W8A8Fp8Config | None = (
            W8A8Fp8Config(is_checkpoint_fp8_serialized=True)
            if self.fp8_layer_patterns
            else None
        )

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
        fp8_layer_patterns = (
            _fp8_override_layer_patterns(config)
            if _is_amd_quark_mxfp4_checkpoint(config)
            else []
        )

        return cls(
            ignored_layers=ignored_layers,
            is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized,
            is_w4a8_fp8=is_w4a8_fp8,
            use_dynamic_mxfp4_activations=use_dynamic_mxfp4_activations,
            quant_method=quant_method,
            fp8_layer_patterns=fp8_layer_patterns,
        )

    def fp8_override_route(self, module_name: str) -> str | None:
        """Return how a module under a per-layer FP8 override is loaded and run.

        Returns ``"w8a8"`` for FP8-resident modules, ``"dequant"`` for weights
        model code consumes raw (dequantized to BF16 at load), or ``None``
        when no FP8 override applies (the module keeps the global scheme).
        """
        if not self.fp8_layer_patterns or should_ignore_quant_layer(
            prefix=module_name, ignored_layers=self.ignored_layers
        ):
            return None
        if not should_ignore_quant_layer(
            prefix=module_name, ignored_layers=self.fp8_layer_patterns
        ):
            return None
        leaf = module_name.rsplit(".", 1)[-1]
        return "dequant" if leaf in _FP8_OVERRIDE_DEQUANT_LEAVES else "w8a8"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        """Select mxfp4 for AMD checkpoints whose global scheme has MXFP4 weights."""
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


_FP8_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
# E2M1 code -> value; the sign is the code's high bit.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_mxfp4_to_bf16(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize a serialized MXFP4 weight to BF16.

    Args:
        packed: ``[N, K // 2]`` uint8, two E2M1 codes per byte, the even
            element in the low nibble (OCP MX packing).
        scales: ``[N, K // 32]`` uint8 E8M0 exponents, one per 32 elements.

    Returns:
        ``[N, K]`` BF16 weight.
    """
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("MXFP4 dequant expects uint8 codes and E8M0 scales")
    rows, half_k = packed.shape
    if scales.shape != (rows, half_k * 2 // 32):
        raise ValueError(
            f"MXFP4 scale shape {tuple(scales.shape)} does not match codes "
            f"{tuple(packed.shape)}"
        )
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    lut = torch.cat((lut, -lut))
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(rows, -1)
    values = lut[codes.long()].view(rows, -1, 32)
    scale = torch.exp2(scales.float() - 127.0).unsqueeze(-1)
    return (values * scale).reshape(rows, -1).to(torch.bfloat16)


def preprocess_mxfp4_checkpoint_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: QuantizationConfig | None,
    dequantize_mxfp4_module: Callable[[str], bool],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Adapt an MXFP4 checkpoint stream with per-layer FP8 overrides to runtime.

    * FP8 ``"w8a8"`` modules pass their codes through; the per-channel
      ``weight_scale`` ``[N]`` is reshaped to the ``[N, 1]`` parameter layout.
    * FP8 ``"dequant"`` modules (raw-consumed weights) and the MXFP4 modules
      selected by ``dequantize_mxfp4_module`` pair each weight with its
      ``weight_scale`` and are yielded as one BF16 ``.weight``.

    Everything else passes through. At most one unpaired tensor is buffered
    per dequantized module.
    """
    if not isinstance(quant_config, Mxfp4Config):
        yield from weights
        return

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, weight in weights:
        is_scale = name.endswith(".weight_scale")
        if not (is_scale or name.endswith(".weight")):
            yield name, weight
            continue
        module = name.rsplit(".", 1)[0]
        route = quant_config.fp8_override_route(module)
        if route == "w8a8":
            yield name, weight.reshape(-1, 1) if is_scale else weight
            continue
        if route is None and (
            not dequantize_mxfp4_module(module)
            or (not is_scale and weight.dtype != torch.uint8)
        ):
            # Not selected, or a module the checkpoint keeps in BF16.
            yield name, weight
            continue
        entry = pending.setdefault(module, {})
        entry["scale" if is_scale else "weight"] = weight
        if "weight" not in entry or "scale" not in entry:
            continue
        del pending[module]
        codes, scale = entry["weight"], entry["scale"]
        if codes.dtype in _FP8_WEIGHT_DTYPES:
            dequantized = (codes.float() * scale.float().reshape(-1, 1)).to(
                torch.bfloat16
            )
        else:
            dequantized = dequantize_mxfp4_to_bf16(codes, scale)
        yield module + ".weight", dequantized
    if pending:
        raise RuntimeError(
            "Dequantized modules missing their weight/weight_scale pair "
            f"at end of checkpoint stream: {sorted(pending)}"
        )
