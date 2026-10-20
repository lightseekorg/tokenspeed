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

"""Quantized Beam checkpoints: admitted formats and their loading.

- compressed-tensors FP8 dynamic W8A8 (llm-compressor's FP8_DYNAMIC: FP8 E4M3
  weights with a static scale per output channel, activations quantized per
  token at run time). The FP8 weights and their scales load unchanged: the
  attention and dense MLP linears run the compressed-tensors FP8 W8A8 scheme
  (FP8 GEMMs with per-token and per-channel scales) and the experts
  FlashInfer's TRT-LLM FP8 per-channel MoE (``fp8ch_vendor_fp32_route``). The
  router and the LM head stay unquantized.
- ModelOpt NVFP4 (group 16) of the experts only. The expert tensors load
  unchanged into FlashInfer's TRT-LLM NVFP4 MoE; attention, the dense MLP
  layers, the router and the LM head must be stored unquantized.
- Quark MXFP4 (group 32) of the experts only, with dynamic MXFP4 activations.
  The portable Triton MoE retains linear packed E2M1 values and E8M0 scales.
  Attention, dense MLPs, routers and the output head remain unquantized.

Any other quantization is refused with the reason, as are a declared KV-cache
quantization (the KV cache stays BF16) and a router or correction bias that is
not unquantized FP32.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.layers.quantization.utils import (
    check_equal_or_regex_match,
    should_exclude_quant_module,
    should_ignore_quant_layer,
)

ADMITTED = (
    "admitted: compressed-tensors FP8 dynamic W8A8 with channel weight scales, "
    "ModelOpt NVFP4 experts with group size 16, and dynamic Quark MXFP4 experts "
    "with group size 32"
)
_CT_FORMATS = ("float-quantized", "naive-quantized")
# The weight scale dtypes a compressed-tensors tool writes (the model's dtype or
# FP32); each converts exactly to FP64.
_CT_SCALE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SCALES = frozenset(
    {"weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale"}
)


def _ct_args_refusal(role: str, args: Any, **expected: Any) -> str | None:
    """Why compressed-tensors ``args`` of ``role`` differ from ``expected``
    (field: value, or a tuple of admitted values); None when they match."""
    if args is None:
        return f"without {role}"
    for name, admitted in expected.items():
        values = admitted if isinstance(admitted, tuple) else (admitted,)
        if getattr(args, name) not in values:
            return f"{role} {name} {getattr(args, name)!r} (admitted: {admitted!r})"
    return None


def _ct_fp8_dynamic_refusal(config: CompressedTensorsConfig) -> str | None:
    """Why a compressed-tensors checkpoint is not FP8 dynamic W8A8 with static
    symmetric channel weight scales and FP8 activations quantized per token at
    run time; None when it is. compressed-tensors itself refuses a channel or
    token strategy with a group size, block structure, activation order or
    the other dynamic setting."""
    raw = config.config or {}
    if config.quant_format not in _CT_FORMATS:
        return f"format {config.quant_format!r} (admitted: {_CT_FORMATS})"
    if raw.get("quantization_status") not in (None, "compressed"):
        return f"quantization_status {raw['quantization_status']!r}"
    for section in ("sparsity_config", "transform_config"):
        if raw.get(section):
            return f"with a {section}"
    if raw.get("kv_cache_scheme") is not None:
        return "with a kv_cache_scheme (Beam keeps a BF16 KV cache)"
    groups = raw.get("config_groups") or {}
    if not groups or not config.target_scheme_map:
        return "without config_groups"
    for name, group in groups.items():
        if group.get("format") not in (None, *_CT_FORMATS):
            return f"config group {name} format {group.get('format')!r}"
        if group.get("output_activations") is not None:
            return f"config group {name} with output_activations"
    for target, scheme in config.target_scheme_map.items():
        refusal = _ct_args_refusal(
            f"weights of {target}",
            scheme.get("weights"),
            num_bits=8,
            type=QuantizationType.FLOAT.value,
            symmetric=True,
            strategy=QuantizationStrategy.CHANNEL.value,
            scale_dtype=(None, *_CT_SCALE_DTYPES),
        ) or _ct_args_refusal(
            f"input_activations of {target}",
            scheme.get("input_activations"),
            num_bits=8,
            type=QuantizationType.FLOAT.value,
            symmetric=True,
            strategy=QuantizationStrategy.TOKEN.value,
        )
        if refusal is not None:
            return refusal
    return None


def beam_checkpoint_format(quant_config: Any) -> str:
    """Return ``fp8_channel``, ``nvfp4`` or ``mxfp4`` for an admitted export.

    Raise ValueError with the unsupported format or quantization setting.
    """
    if type(quant_config) is CompressedTensorsConfig:
        refusal = _ct_fp8_dynamic_refusal(quant_config)
        if refusal is not None:
            raise ValueError(f"Beam refuses compressed-tensors {refusal}; {ADMITTED}")
        return "fp8_channel"
    if type(quant_config) is Mxfp4Config:
        if (
            quant_config.is_checkpoint_mxfp4_serialized
            and quant_config.use_dynamic_mxfp4_activations
            and not quant_config.is_w4a8_fp8
            and not quant_config.fp8_layer_patterns
            and quant_config.group_size == 32
        ):
            return "mxfp4"
        raise ValueError(f"Beam requires serialized dynamic MXFP4 experts; {ADMITTED}")
    if type(quant_config) is Nvfp4Config:
        kv = quant_config.kv_cache_quant_algo
        if quant_config.group_size != 16:
            reason = f"group_size {quant_config.group_size}"
        elif isinstance(kv, str) and kv.lower() not in ("", "auto", "none"):
            reason = f"kv_cache_quant_algo {kv} (Beam keeps a BF16 KV cache)"
        else:
            return "nvfp4"
        raise ValueError(f"Beam refuses ModelOpt NVFP4 with {reason}; {ADMITTED}")
    raise ValueError(
        f"Beam refuses quantization {quant_config.get_name()} "
        f"({type(quant_config).__name__}); {ADMITTED}"
    )


def _round_to_bf16(exact: torch.Tensor) -> torch.Tensor:
    # FP8 codes times a BF16, FP16 or FP32 scale are exact in FP64 (at most 28
    # significant bits). Round once to BF16, to nearest even: a float64 cast
    # rounds twice (through FP32).
    mantissa, exponent = torch.frexp(exact)
    return torch.ldexp(torch.round(mantissa * 256) / 256, exponent).to(torch.bfloat16)


def dequantize_fp8_channel(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """BF16 of FP8 E4M3 weights ``[N, K]`` times the scale of their output
    channel (``[N, 1]``), on the GPU when there is one: the BF16 reference of
    an FP8 weight (Beam's loading before it ran FP8 natively)."""
    if torch.cuda.is_available():
        weight, scale = weight.cuda(), scale.cuda()
    return _round_to_bf16(weight.double() * scale.double().reshape(-1, 1))


def _check_fp8_channel(module: str, weight: torch.Tensor, scale: torch.Tensor):
    rows = weight.shape[0]
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise TypeError(
            f"{module}.weight: expected an FP8 E4M3 matrix, got {weight.dtype} "
            f"with shape {tuple(weight.shape)}"
        )
    if scale.dtype not in _CT_SCALE_DTYPES or tuple(scale.shape) != (rows, 1):
        raise TypeError(
            f"{module}.weight_scale: expected BF16, FP16 or FP32 [{rows}, 1], got "
            f"{scale.dtype} with shape {tuple(scale.shape)}"
        )
    values = scale.float()
    if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError(f"{module}.weight_scale: scales must be finite and positive")


def _check_nvfp4_scalar(name: str, tensor: torch.Tensor) -> float:
    """An NVFP4 weight_scale_2 or input_scale: one FP32 value, finite and
    positive, or 0 for an input scale. The MoE divides by a layer's largest
    input scale; ModelOpt writes 0 for an expert no calibration token reached."""
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise TypeError(
            f"{name}: expected one FP32 value, got {tensor.dtype} with shape "
            f"{tuple(tensor.shape)}"
        )
    value = float(tensor)
    zero_ok = name.endswith(".input_scale")
    if not 0.0 <= value < float("inf") or value == 0.0 and not zero_ok:
        raise ValueError(
            f"{name}: an NVFP4 scale must be finite and "
            f"{'non-negative' if zero_ok else 'positive'}, got {value}"
        )
    return value


def _module_role(module: str) -> str:
    if ".mlp.experts." in module:
        return "experts"
    if module.endswith(".mlp.gate"):
        return "router"
    if ".self_attn." in module:
        return "attention"
    if ".mlp." in module:
        return "dense_mlp"
    return module.rpartition(".")[2]


def _check_mxfp4(module: str, weight: torch.Tensor, scale: torch.Tensor) -> None:
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise TypeError(f"{module}.weight: expected a packed uint8 MXFP4 matrix")
    rows, packed_width = weight.shape
    if (
        packed_width % 16
        or scale.dtype != torch.uint8
        or scale.shape != (rows, packed_width // 16)
    ):
        raise TypeError(f"{module}.weight_scale: expected uint8 group-32 E8M0 scales")
    if bool((scale == 255).any()):
        raise ValueError(f"{module}.weight_scale: E8M0 scales must be finite")


@dataclass
class BeamQuantPolicy:
    """How Beam loads one admitted quantized checkpoint (``fmt``: its
    :func:`beam_checkpoint_format`)."""

    quant_config: Any
    fmt: str
    counts: Counter = field(default_factory=Counter)
    _input_scales: dict = field(default_factory=dict)

    def _quantizes(self, module: str) -> bool:
        config = self.quant_config
        if self.fmt == "nvfp4":
            return not should_exclude_quant_module(module, config.exclude_modules)
        if self.fmt == "mxfp4":
            return not should_ignore_quant_layer(module, config.ignored_layers)
        if should_ignore_quant_layer(module, config.ignored_layers):
            return False
        return any(
            target == "Linear" or check_equal_or_regex_match(module, [target])
            for target in config.target_scheme_map
        )

    def load_witness(self) -> str:
        counts = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items()))
        if self.fmt == "mxfp4":
            return f"Beam Quark MXFP4 checkpoint (experts only) loaded as packed MXFP4: {counts}"
        if self.fmt == "fp8_channel":
            return (
                "Beam FP8 checkpoint (compressed-tensors FP8 dynamic W8A8) "
                f"loaded as FP8, FP8 GEMMs and FP8 experts run it: {counts}"
            )
        return f"Beam NVFP4 checkpoint (ModelOpt NVFP4, experts only) loaded: {counts}"

    def adapt(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """The checkpoint stream as Beam's parameters take it: FP8 weights
        and their channel scales passed on unchanged once both are checked,
        NVFP4 expert tensors passed on unchanged. Refuses a quantized or
        non-FP32 router, a quantized tensor of a module the config leaves
        unquantized or Beam keeps BF16, a scale without its weight or of a BF16
        weight, and NVFP4 scales off :func:`_check_nvfp4_scalar` or whose input
        scales are 0 in every expert of a layer's w13 or w2."""
        # Each load stands alone: a weight update loads every tensor again.
        self.counts.clear()
        self._input_scales.clear()
        pending: dict[str, dict[str, torch.Tensor]] = {}
        plain: set[str] = set()
        done: set[str] = set()
        for name, tensor in weights:
            module, _, leaf = name.rpartition(".")
            role = _module_role(module)
            quantized = leaf in _SCALES or (
                leaf == "weight" and tensor.dtype in (torch.float8_e4m3fn, torch.uint8)
            )
            if role == "router":
                if quantized or tensor.dtype != torch.float32:
                    raise TypeError(
                        "Beam's router and e_score_correction_bias must be "
                        f"unquantized FP32; {name} is "
                        f"{'quantized' if quantized else tensor.dtype}"
                    )
            elif not quantized:
                if leaf == "weight":
                    if (
                        self.fmt == "mxfp4"
                        and role == "experts"
                        and self._quantizes(module)
                    ):
                        raise TypeError(
                            f"{name}: MXFP4 expert weights must be packed uint8"
                        )
                    if module in pending:
                        raise ValueError(f"{name}: an unquantized weight with scales")
                    plain.add(module)
            elif module in plain:
                raise ValueError(f"{name}: a scale of the unquantized {module}")
            elif not self._quantizes(module):
                raise ValueError(
                    f"{name} is quantized, but the checkpoint's quantization "
                    f"config does not quantize {module}"
                )
            elif self.fmt == "nvfp4":
                if role != "experts":
                    raise ValueError(
                        f"{name}: Beam runs ModelOpt NVFP4 on its experts only; "
                        f"its {role} must be stored unquantized"
                    )
                if leaf in ("weight_scale_2", "input_scale"):
                    value = _check_nvfp4_scalar(name, tensor)
                    if leaf == "input_scale":
                        self._note_input_scale(module, value)
                self.counts["expert tensors kept NVFP4"] += 1
            else:
                if self.fmt == "mxfp4" and role != "experts":
                    raise ValueError(
                        f"{name}: Beam quantizes only routed experts to MXFP4"
                    )
                if leaf not in ("weight", "weight_scale"):
                    raise ValueError(
                        f"{name}: {self.fmt} has no "
                        f"{leaf} (activations are quantized per token at run time)"
                    )
                entry = pending.setdefault(module, {})
                if leaf in entry or module in done:
                    raise ValueError(f"Duplicate quantized tensor: {name}")
                entry[leaf] = tensor
                if len(entry) < 2:
                    continue
                del pending[module]
                done.add(module)
                if self.fmt == "mxfp4":
                    _check_mxfp4(module, entry["weight"], entry["weight_scale"])
                    self.counts["experts kept MXFP4"] += 1
                else:
                    _check_fp8_channel(module, entry["weight"], entry["weight_scale"])
                    self.counts[f"{role} kept FP8"] += 1
                yield f"{module}.weight", entry["weight"]
                yield f"{module}.weight_scale", entry["weight_scale"]
                continue
            yield name, tensor
        if pending:
            raise ValueError(
                f"Quantized tensors without their weight or scale: {sorted(pending)}"
            )
        zero = sorted(group for group, top in self._input_scales.items() if top == 0.0)
        if zero:
            raise ValueError(
                f"NVFP4 input scales that are 0 in every expert: {zero}; the MoE "
                "divides by their maximum"
            )

    def _note_input_scale(self, module: str, value: float) -> None:
        # Every rank, also under --ep-size, keeps the largest input scale of a
        # layer's gate and up projections (w13) and of its down ones (w2).
        experts, _, projection = module.rpartition(".")
        kind = "down_proj" if projection == "down_proj" else "{gate,up}_proj"
        group = f"{experts.rpartition('.')[0]}.*.{kind}.input_scale"
        self._input_scales[group] = max(self._input_scales.get(group, 0.0), value)


def beam_quant_policy(quant_config: Any) -> BeamQuantPolicy:
    """The policy of a quantized checkpoint Beam admits; ValueError says why a
    checkpoint is refused."""
    return BeamQuantPolicy(quant_config, beam_checkpoint_format(quant_config))


__all__ = [
    "BeamQuantPolicy",
    "beam_checkpoint_format",
    "beam_quant_policy",
    "dequantize_fp8_channel",
]
