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

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.moe.core.registry import get_backend_cls
from tokenspeed.runtime.layers.moe.core.types import BackendKey, MoELayerSpec
from tokenspeed.runtime.layers.moe.utils import get_moe_backend
from tokenspeed.runtime.layers.quantization import (
    CompressedTensorsConfig,
    Fp8Config,
    Mxfp4Config,
    Nvfp4Config,
    W8A8Fp8Config,
)
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer

_AUTO_IMPL_PREFERENCE = {
    "unquantized": (
        "flashinfer_trtllm",
        "flashinfer_cutlass",
        "triton",
    ),
    "nvfp4": (
        "flashinfer_trtllm",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
    ),
    "mxfp4": (
        "flashinfer_mxfp4",
        "triton_kernel",
    ),
    "fp8": (
        "flashinfer_cutlass",
        "triton",
    ),
    "w8a8_fp8": ("triton",),
    "wna16": ("marlin",),
}


def _normalize_quant_kind(quant_config: object, prefix: str = "") -> str:
    # Handle ignored layers or no quantization
    if quant_config is None or should_ignore_quant_layer(
        prefix=prefix,
        ignored_layers=getattr(quant_config, "ignored_layers", []),
    ):
        return "unquantized"

    # ModelOpt FP4 quantization
    if isinstance(quant_config, Nvfp4Config):
        return "nvfp4"
    # MXFP4 quantization
    if isinstance(quant_config, Mxfp4Config):
        return "mxfp4"
    if (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size is not None
    ):
        return "fp8"
    # W8A8 quantization configs
    if isinstance(quant_config, W8A8Fp8Config):
        return "w8a8_fp8"
    if isinstance(quant_config, CompressedTensorsConfig):
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            return "wna16"

    raise RuntimeError(f"Unsupported MoE quant_config: {quant_config}")


def _detect_arch() -> str:
    platform = current_platform()

    major, minor = platform.arch_version.major, platform.arch_version.minor
    if major >= 9:
        return f"sm{major}0"
    return f"sm{major}{minor}"


def _resolve_impl_candidates(quant_kind: str) -> tuple[str, ...]:
    backend = get_moe_backend()
    auto_candidates = _AUTO_IMPL_PREFERENCE.get(quant_kind, ())

    if not backend.is_auto():
        # If a specific MoE backend is forced globally, only honor it
        # when it is actually registered for this quant_kind. Otherwise
        # fallback to the auto preference (e.g. draft model is unquantized
        # but target model is configured with flashinfer_cutedsl).
        if backend.value in auto_candidates:
            return (backend.value,)

    if auto_candidates:
        return auto_candidates

    raise RuntimeError(f"Unsupported MoE quant kind: {quant_kind}")


def select_backend(
    spec: MoELayerSpec,
    quant_config: object,
    routing_config: dict | None = None,
):
    from tokenspeed.runtime.layers.moe.backends import ensure_backend_family_registered

    quant_kind = _normalize_quant_kind(quant_config, prefix=spec.prefix)
    if quant_kind == "unquantized":
        quant_config = None

    arch = _detect_arch()
    tried = []

    for impl in _resolve_impl_candidates(quant_kind):
        key = BackendKey(arch=arch, quant=quant_kind, impl=impl)
        try:
            ensure_backend_family_registered(key.quant, key.impl)
            backend_cls = get_backend_cls(key)
        except KeyError:
            tried.append(f"{impl}:not-registered")
            continue

        if not backend_cls.supports(spec, quant_config):
            tried.append(f"{impl}:unsupported")
            continue

        return backend_cls(
            key=key,
            spec=spec,
            quant_config=quant_config,
            routing_config=routing_config,
        )

    tried_str = ", ".join(tried) if tried else "<none>"
    raise RuntimeError(
        f"No MoE backend available for {arch}/{quant_kind}. Tried: {tried_str}"
    )
