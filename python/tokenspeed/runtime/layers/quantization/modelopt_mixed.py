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

Such checkpoints (e.g. nvidia/MiniMax-M3-NVFP4, nvidia/Kimi-K3-NVFP4) declare
a per-module ``quant_algo`` in the ``quantized_layers`` dict of their
quantization config, plus ``exclude_modules`` for unquantized modules. This
config resolves each runtime layer to the matching algorithm and delegates
weight handling to the corresponding single-algorithm config (MXFP8, NVFP4,
FP8_PB_WO).

``FP8_PB_WO`` is ModelOpt's per-block FP8 format (128x128 blocks, float32
dequant scales; ``weight_scale`` is numerically identical to DeepSeek's
``weight_scale_inv`` — both are ``amax / 448``). Following TRT-LLM, which
aliases FP8_PB_WO to FP8_BLOCK_SCALES, nearly every layer stays FP8-resident
on the DeepSeek-style w8a8 blockwise path (:class:`Fp8LinearMethod` or the
model's own w8a8 fallbacks: dynamic per-token-128-group activation
quantization + block-scale GEMM); fused consumers reassemble the FP8
segments verbatim (Kimi-K3: the merged KDA buffer concatenates scale grids,
the MLA fused_qkv_a reorders segments onto its private 128-aligned layout).
Only the few weights consumed raw by bf16-only kernels with no fallback
(Kimi-K3: ``f_b_proj`` plus two checkpoint-alias names) are block-dequantized
to bf16 at load time (see :data:`_FP8_PB_WO_DEQUANT_LEAVES` and
:func:`preprocess_fp8_pb_wo_weights`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, Mxfp8Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.layers.quantization.utils import (
    modelopt_block_scale_to_2d,
    should_exclude_quant_module,
)

logger = logging.getLogger(__name__)

_MOE_WEIGHT_DTYPES = {
    "NVFP4": "nvfp4",
    "MXFP8": "fp8",
}

_FP8_PB_WO_BLOCK_SIZE = [128, 128]

_SUPPORTED_QUANT_ALGOS = frozenset({"NVFP4", "MXFP8", "FP8_PB_WO"})

# FP8_PB_WO runtime routing. A module keeps FP8 weights (w8a8 blockwise,
# TRT-LLM's FP8_BLOCK_SCALES alias) if its weight either flows through
# ``quant_method.apply()`` as a plain LinearBase call or has an fp8-aware
# w8a8 fallback in the model. Kimi-K3 coverage:
#   q/k/v/g/f_a/b_proj   -> KDA merged qkvgb projection: FP8-resident buffer
#                           (rows padded to the 128 grid, per-segment scale
#                           grids concatenated) consumed by the w8a8
#                           blockscale GEMM branch of kimi3_qkvfab_projection
#   q_a/kv_a/g (MLA) /
#   fused_qkv_a aliases  -> reassembled VERBATIM onto the fused_qkv_a
#                           layer's private 128-aligned row order at load,
#                           quantized module call
#   q_b_proj             -> gated MLA falls back to fused norms + the
#                           quantized q_b GEMM for FP8 weights
#   o_proj / kv_b_proj   -> plain LinearBase w8a8
# Modules whose raw ``.weight`` is consumed by bf16-only kernels with no
# fallback are block-dequantized to bf16 at load and dispatched
# UnquantizedLinearMethod instead:
#   f_b_proj             -> raw ``f_b_weight`` GEMV inside the KDA attention
#                           backend (fused decode scan; kept bf16 by
#                           decision -- 6 MB/rank is not worth touching the
#                           KDA-NaN-sensitive megafuse kernels)
#   fused_qkvg_proj /
#   in_proj_qkvgfab      -> checkpoint aliases of the KDA merged buffer
#                           (no runtime module consumes these prefixes)
_FP8_PB_WO_DEQUANT_LEAVES = frozenset(
    {
        "f_b_proj",
        "fused_qkvg_proj",
        "in_proj_qkvgfab",
    }
)

_FUSED_PROJECTION_SHARDS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


class ModelOptMixedConfig(QuantizationConfig):
    """Config for ModelOpt MIXED_PRECISION checkpoints.

    Args:
        quantized_layers: Checkpoint module name -> upper-cased quant_algo
            (``"NVFP4"``, ``"MXFP8"`` or ``"FP8_PB_WO"``). Keys use checkpoint
            naming until :meth:`apply_checkpoint_name_replacements` rewrites
            them to runtime naming.
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
        # FP8_PB_WO aliases to FP8_BLOCK_SCALES (TRT-LLM precedent): the
        # DeepSeek-style w8a8 blockwise path with float32 128x128 scales.
        # scale_fmt stays None — the checkpoint scales are arbitrary amax/448
        # floats, NOT powers of two, so the ue8m0 deep_gemm transform (which
        # rounds scales up) must not engage.
        self.fp8_block_scales_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=list(_FP8_PB_WO_BLOCK_SIZE),
            scale_fmt=None,
        )
        self.has_fp8_pb_wo = "FP8_PB_WO" in set(quantized_layers.values())
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
            if algo not in _SUPPORTED_QUANT_ALGOS:
                unknown.add(algo)
                continue
            quantized_layers[name] = algo
            if algo == "NVFP4" and group_size is None:
                group_size = int(info.get("group_size", 16))
        if unknown:
            raise ValueError(
                f"Unsupported quant_algo values in quantized_layers: {sorted(unknown)}. "
                f"Supported: {sorted(_SUPPORTED_QUANT_ALGOS)}."
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

    def _fp8_pb_wo_leaf_route(self, module_name: str) -> str:
        """Route an FP8_PB_WO module: ``"dequant"`` (bf16 at load) or ``"w8a8"``."""
        leaf = module_name.rsplit(".", 1)[-1]
        # g_proj needs no per-sublayer distinction anymore: the KDA merged
        # buffer and the MLA fused projection both keep it FP8 (w8a8); the
        # model loader tells them apart by which runtime module exists.
        return "dequant" if leaf in _FP8_PB_WO_DEQUANT_LEAVES else "w8a8"

    def fp8_pb_wo_route(self, module_name: str) -> str | None:
        """``"dequant"`` / ``"w8a8"`` for FP8_PB_WO checkpoint modules, else None."""
        if self.quantized_layers.get(module_name) != "FP8_PB_WO":
            return None
        return self._fp8_pb_wo_leaf_route(module_name)

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
        if algo == "FP8_PB_WO":
            if self._fp8_pb_wo_leaf_route(prefix) == "dequant":
                # Raw-consumed weight: block-dequantized to bf16 by
                # preprocess_fp8_pb_wo_weights during load.
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self.fp8_block_scales_config)
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
                if algo not in _MOE_WEIGHT_DTYPES:
                    raise ValueError(
                        f"MoE experts at {prefix!r} resolve to {algo!r}, which "
                        "has no MoE kernel path; supported expert algos: "
                        f"{sorted(_MOE_WEIGHT_DTYPES)}."
                    )
                return _MOE_WEIGHT_DTYPES[algo]
        raise ValueError(
            f"No quantized_layers entry resolves the MoE prefix {prefix!r}; "
            "cannot infer the experts' weight dtype."
        )

    def get_scaled_act_names(self) -> list[str]:
        return []


_FP8_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _assert_routed_or_known(
    quant_config: "ModelOptMixedConfig",
    module: str,
    tensor_name: str,
    weight: torch.Tensor,
    is_weight: bool,
) -> None:
    """Fail loudly on quantized-looking tensors with no ownership.

    A raw FP8 weight (or a ``weight_scale`` sidecar) whose module resolves to
    no quant_algo and is not excluded means checkpoint/runtime module-name
    drift: without a route the FP8 bytes would be silently raw-copied into a
    bf16 parameter (wrong values, no error).
    """
    if is_weight and weight.dtype not in _FP8_WEIGHT_DTYPES:
        return  # plain 16/32-bit weights are legitimately unquantized
    if quant_config._resolve_quant_algo(module) is not None:
        return  # another algo's loader owns it (MXFP8, direct entries, ...)
    # quantized_layers entries may name an ANCESTOR module covering a whole
    # subtree — ModelOpt MIXED_PRECISION exports declare fused experts as
    # e.g. "model.layers.49.block_sparse_moe.experts", which owns every
    # "...experts.<i>.w{1,2,3}" tensor (the same parent-first semantics
    # moe_weight_dtype uses). _resolve_quant_algo only scans downward
    # (direct hit / fused shards / children), so walk the ancestors here.
    # Only non-FP8_PB_WO subtrees are legitimate owners (NVFP4/MXFP8 fused
    # experts): FP8_PB_WO entries name leaf projections, so an FP8_PB_WO
    # ancestor over an unrouted child is itself name drift and must raise.
    parts = module.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        algo = quant_config.quantized_layers.get(".".join(parts[:depth]))
        if algo is not None and algo != "FP8_PB_WO":
            return  # a subtree entry (e.g. NVFP4 fused experts) owns it
    if should_exclude_quant_module(module, quant_config.exclude_modules):
        return
    raise RuntimeError(
        f"Checkpoint tensor {tensor_name!r} looks quantized "
        f"({'FP8 weight' if is_weight else 'weight_scale sidecar'}) but module "
        f"{module!r} resolves to no quant_algo in quantized_layers and is not "
        "excluded. This is likely checkpoint/runtime module-name drift; "
        "loading would raw-copy FP8 bytes into a bf16 parameter. Fix the "
        "name mapping (quant_module_name_replacements / "
        "_FP8_PB_WO_DEQUANT_LEAVES) or add the module to exclude_modules."
    )


def _expand_block_scale(
    scale: torch.Tensor, n: int, k: int, block_n: int, block_k: int
) -> torch.Tensor:
    """Expand a 2-D block-scale grid to per-element [n, k] (ragged-clipped)."""
    return (
        scale.to(torch.float32)
        .repeat_interleave(block_n, dim=0)[:n]
        .repeat_interleave(block_k, dim=1)[:, :k]
    )


def _block_dequant_fp8_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor, block_n: int, block_k: int
) -> torch.Tensor:
    """Vectorized per-block dequant ``(fp8 * f32 scale) -> bf16``.

    Ragged trailing blocks are handled by clipping the expanded scale grid.
    Runs on CUDA when available (load-time throughput; the quantization utils
    ``block_dequant`` loops per tile — tolerable for today's few small
    dequant-routed tensors such as Kimi-K3's per-layer f_b_proj, but this
    vectorized form keeps the path shape-agnostic).
    """
    if weight.dtype not in _FP8_WEIGHT_DTYPES:
        raise TypeError(f"expected an FP8 weight to dequantize, got {weight.dtype}")
    if weight.device.type != "cuda" and torch.cuda.is_available():
        weight = weight.cuda()
        scale = scale.cuda()
    n, k = weight.shape
    s_full = _expand_block_scale(scale, n, k, block_n, block_k)
    return (weight.to(torch.float32) * s_full).to(torch.bfloat16)


def preprocess_fp8_pb_wo_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: QuantizationConfig | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Adapt a checkpoint weight stream for FP8_PB_WO modules.

    * ``"dequant"`` modules (raw-consumed weights, see
      :data:`_FP8_PB_WO_DEQUANT_LEAVES`): the FP8 weight is paired with its
      ``weight_scale``, block-dequantized to bf16, and yielded under the
      original ``.weight`` name; the scale tensor is consumed and never
      surfaces (the runtime parameter is plain bf16, e.g. Kimi-K3's
      f_b_proj, whose raw weight feeds the KDA attention backend's GEMV).
    * ``"w8a8"`` modules: the FP8 weight passes through unchanged;
      ``weight_scale`` is renamed to ``weight_scale_inv`` (the parameter
      :class:`Fp8LinearMethod` registers — ModelOpt's ``weight_scale`` is the
      same ``amax/448`` dequant multiplier as DeepSeek's ``weight_scale_inv``)
      and ModelOpt's 4-D scale layout is squeezed to the 2-D block grid.

    Everything else passes through untouched (NVFP4 experts keep their
    ``weight_scale`` names for the MoE loader). Streaming: at most one
    unpaired tensor is buffered per dequant module.
    """
    if (
        not isinstance(quant_config, ModelOptMixedConfig)
        or not quant_config.has_fp8_pb_wo
    ):
        yield from weights
        return

    block_n, block_k = _FP8_PB_WO_BLOCK_SIZE
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, weight in weights:
        is_scale = name.endswith(".weight_scale")
        is_weight = name.endswith(".weight")
        if not (is_scale or is_weight):
            yield name, weight
            continue
        module = name.rsplit(".", 1)[0]
        route = quant_config.fp8_pb_wo_route(module)
        if route is None:
            _assert_routed_or_known(quant_config, module, name, weight, is_weight)
            yield name, weight
        elif route == "w8a8":
            if is_scale:
                yield (
                    module + ".weight_scale_inv",
                    modelopt_block_scale_to_2d(weight),
                )
            elif weight.dtype not in _FP8_WEIGHT_DTYPES:
                # A 16-bit weight for an FP8-resident module can only be a
                # runtime refit stream; copying it into the FP8 parameter (or
                # the fused buffer at checkpoint-canonical offsets) would
                # silently corrupt the weights.
                raise TypeError(
                    f"FP8-resident module {module!r} cannot load a "
                    f"{weight.dtype} weight (bf16 refit of FP8_PB_WO w8a8 "
                    "layers is unsupported)."
                )
            else:
                yield name, weight
        elif is_weight and weight.dtype not in _FP8_WEIGHT_DTYPES:
            # Runtime bf16 refit streams (update_weights_from_distributed)
            # re-send already-dequantized weights with no scale sidecar; pass
            # them straight to the bf16 parameter instead of pairing.
            yield name, weight
        else:  # dequant: pair the FP8 weight with its scale sidecar
            entry = pending.setdefault(module, {})
            entry["scale" if is_scale else "weight"] = weight
            if "weight" in entry and "scale" in entry:
                del pending[module]
                yield (
                    module + ".weight",
                    _block_dequant_fp8_to_bf16(
                        entry["weight"],
                        modelopt_block_scale_to_2d(entry["scale"]),
                        block_n,
                        block_k,
                    ),
                )
    if pending:
        raise RuntimeError(
            "FP8_PB_WO dequant modules missing their weight/weight_scale "
            f"pair at end of checkpoint stream: {sorted(pending)}"
        )
