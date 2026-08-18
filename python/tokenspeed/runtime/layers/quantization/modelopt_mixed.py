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
aliases FP8_PB_WO to FP8_BLOCK_SCALES, layers whose weight naturally flows
through ``quant_method.apply`` run the existing DeepSeek-style w8a8 blockwise
path (:class:`Fp8LinearMethod`: dynamic per-token-128-group activation
quantization + block-scale GEMM). Layers whose raw weight is consumed by
fused/custom bf16 kernels are instead block-dequantized to bf16 at load time
(see :data:`_FP8_PB_WO_DEQUANT_LEAVES` and
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
#   fused_qkv_a aliases  -> spliced onto the fused_qkv_a layer's clean
#                           128-block grid at load, quantized module call
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
    parts = module.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        if ".".join(parts[:depth]) in quant_config.quantized_layers:
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
    ``block_dequant`` loops per tile and is too slow for ~600 large tensors).
    """
    if weight.dtype not in _FP8_WEIGHT_DTYPES:
        raise TypeError(f"expected an FP8 weight to dequantize, got {weight.dtype}")
    if weight.device.type != "cuda" and torch.cuda.is_available():
        weight = weight.cuda()
        scale = scale.cuda()
    n, k = weight.shape
    s_full = _expand_block_scale(scale, n, k, block_n, block_k)
    return (weight.to(torch.float32) * s_full).to(torch.bfloat16)


def splice_requant_fp8_block_rows(
    segments: list[tuple[torch.Tensor, torch.Tensor]],
    block_n: int = 128,
    block_k: int = 128,
    pad_rows_to_multiple: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-concatenate per-block-quantized FP8 segments onto a clean grid.

    Fused projections can concatenate segments at row offsets that are not
    multiples of ``block_n`` (Kimi-K3 ``fused_qkv_a``: ``[q_a 1536 | kv_a 576
    | gate]`` puts the gate at row 2112), so the per-segment scale grids
    cannot be stitched together directly. Each segment is exactly
    dequantized to fp32 with its own grid, the concatenation is requantized
    on a fresh grid anchored at row 0, and the result stays FP8-resident.

    Numerics: a fused block whose span coincides with exactly one source
    scale cell reproduces that cell's scale (the cell's amax element
    requantizes to the same ±448 code), so its codes are bit-identical and
    its scale matches within 1 float32 ulp. Blocks that straddle several
    source cells (segment boundaries, or segments whose offset is misaligned
    with the fused grid) take the merged amax scale and re-round values
    within the FP8 quantization band.

    Args:
        segments: ``(weight_fp8 [n_i, K], scale_f32 2-D block grid)`` per
            segment, all with the same ``K``.
        block_n: Fused-grid block rows.
        block_k: Fused-grid block columns.
        pad_rows_to_multiple: When set, append zero rows after the last
            segment until the total row count is a multiple of this value
            (e.g. 128 keeps a w8a8 blockscale GEMM on kernel paths requiring
            ``N % 128 == 0``). Zero rows cannot change any block's amax, so
            every scale — and therefore every real row's quantized code — is
            bit-identical to the unpadded result, and the pad rows themselves
            quantize to exact zeros. Segment offsets are unaffected (padding
            is appended at the tail only).

    Returns:
        ``(weight_fp8 [n_padded, K], scale_f32 [ceil(n_padded/block_n),
        ceil(K/block_k)])`` where ``n_padded`` is ``sum(n_i)`` rounded up to
        ``pad_rows_to_multiple`` (or exactly ``sum(n_i)`` when unset).
    """
    if not segments:
        raise ValueError("splice_requant_fp8_block_rows needs at least one segment")
    for weight, _ in segments:
        if weight.dtype not in _FP8_WEIGHT_DTYPES:
            raise TypeError(
                f"expected FP8 segment weights, got {weight.dtype}; bf16 "
                "segments belong on the plain fused loading path"
            )
    n_real = sum(weight.shape[0] for weight, _ in segments)
    k = segments[0][0].shape[1]
    n = n_real
    if pad_rows_to_multiple is not None and n % pad_rows_to_multiple:
        n += pad_rows_to_multiple - n % pad_rows_to_multiple
    nb = (n + block_n - 1) // block_n
    kb = (k + block_k - 1) // block_k
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    device = "cuda" if torch.cuda.is_available() else segments[0][0].device
    # Dequantize every segment straight into one padded fp32 workspace: the
    # earlier per-segment tensors + torch.cat doubled the transient footprint
    # and left mixed-size blocks in the caching allocator (which the KV-pool
    # sizing then can't use).
    padded = torch.zeros(nb * block_n, kb * block_k, dtype=torch.float32, device=device)
    row = 0
    for weight, scale in segments:
        if weight.device != padded.device:
            weight = weight.to(padded.device)
            scale = scale.to(padded.device)
        ni, ki = weight.shape
        if ki != k:
            raise ValueError(f"segment K mismatch: {ki} != {k}")
        padded[row : row + ni, :k] = weight.to(torch.float32) * _expand_block_scale(
            scale, ni, ki, block_n, block_k
        )
        row += ni
    blocks = padded.view(nb, block_n, kb, block_k)
    amax = blocks.abs().amax(dim=(1, 3))
    # All-zero blocks (zeroed weight regions, or full pad blocks) quantize to
    # exact zeros under any scale; pick 1.0 so downstream kernels never see a
    # subnormal scale or a zero divisor in the grid.
    scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    quant = (blocks / scale[:, None, :, None]).clamp(-fp8_max, fp8_max)
    quant = (
        quant.view(nb * block_n, kb * block_k)[:n, :k]
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return quant, scale.contiguous()


def preprocess_fp8_pb_wo_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: QuantizationConfig | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Adapt a checkpoint weight stream for FP8_PB_WO modules.

    * ``"dequant"`` modules (raw-consumed weights, see
      :data:`_FP8_PB_WO_DEQUANT_LEAVES`): the FP8 weight is paired with its
      ``weight_scale``, block-dequantized to bf16, and yielded under the
      original ``.weight`` name; the scale tensor is consumed and never
      surfaces (the runtime parameter is plain bf16). Fusing dequantized
      segments downstream (KDA merged / fused_qkv_a) is therefore exact and
      free of block-alignment constraints.
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
