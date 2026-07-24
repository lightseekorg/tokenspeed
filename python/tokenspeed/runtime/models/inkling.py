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

"""Inkling model (text decoder + lightweight audio/vision embedding towers).

Naming here follows the checkpoint-native ``config.json``
(``model_type: inkling_mm_model``, ``InklingForConditionalGeneration``), so
the raw snapshots serve unmodified.

Decoder-only MoE transformer with four non-standard pieces:

* **Relative attention** — no RoPE; a learned, query-conditioned per-head
  bias over causal distance is added to pre-softmax logits via the
  ``rel_mha`` attention operator family (fused sheared-bias FA4 path with a
  ``score_mod`` gather fallback). Softmax scale is ``1/head_dim`` (not
  ``1/sqrt``).
* **Sconv** — residual per-channel causal FIR (window ``sconv_kernel_size``)
  at four sites per block (K/V streams pre-KV-cache, attention output, MLP
  output), with per-request rolling state in ``InklingConvStatePool``.
* **Shared-expert-sink MoE** — sigmoid+bias top-k selection over routed
  experts only; weights are logsigmoid-normalized over selected-routed plus
  shared logits, scaled by ``route_scale``.
* **muP logits** — hidden states divided by ``logits_mup_width_multiplier``
  before the LM head.

Layer layout: ``local_layer_ids`` use SWA (``sliding_window_size``) with
``rel_extent == sliding_window_size``; the rest are full attention with the
config ``rel_extent``. KV heads default to the hetero layout (#647
byte-uniform slots): full-attention layers keep their checkpoint-native
head count (unconditional; the INKLING_HETERO_KV gate is retired) — uniform serving with
full-layer KV replicated at load (see the config docstring).

Multimodal towers (gated on ``*_config.decoder_dmodel`` being set; text-only
checkpoints leave both off):

* **Vision** — :class:`InklingHMLPPatchEncoder`: folds each patch's time/space
  interior to channel depth, then a Linear(+RMSNorm+GELU) stack maps it to one
  ``decoder_dmodel`` token per patch. All media preprocessing (pixels ->
  patches) happens in the SMG gateway; the engine only embeds the shipped
  patch tensors.
* **Audio** — dMel: per-(mel-bin, quantized-value) embedding rows summed over
  the mel bins of each audio frame, optional final RMSNorm.

Both towers splice their outputs into the LM input embeddings through the
shared :class:`~tokenspeed.runtime.multimodal.embedder.VisionEmbedder`
(offsets come pre-expanded from the gateway). ``embed_norm`` is folded into
the text-token embedding for that path so tower outputs are never re-normed
(reference parity).

Prefix caching is supported under the paged-conv defaults; only the rolling
conv-state fallback requires ``--no-enable-prefix-caching`` (asserted at
init). Weight loading supports dummy, real BF16, Quark MXFP4, and ModelOpt
NVFP4 checkpoints (routed experts quantized, quant-exclusion lists translated
to this module tree). MTP speculative decoding is served by the NextN draft
model (see inkling_nextn.py).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.conv import (
    sconv_decode,
    sconv_decode_paged,
    sconv_prefill,
    sconv_prefill_paged,
)
from tokenspeed_kernel.ops.gemm.cuda import dsv3_router_gemm
from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.inkling_config import (
    InklingAudioConfig,
    InklingConvStream,
    InklingMMConfig,
    InklingModelConfig,
    InklingVisionConfig,
    inkling_conv_stream_layout,
    inkling_kv_heads_for_layer,
)
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput, TopK
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import MoEGate as _DSV3MoEGate
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix, make_layers
from tokenspeed.runtime.utils.common import set_weight_attrs
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.pdl import pdl_enabled

logger = logging.getLogger(__name__)

_is_hopper_plus = current_platform().is_hopper_plus

# Escape hatch: disable the rel-logits aux-stream fork (serial pre-attention).
_ATTN_RELFORK = os.environ.get("INKLING_ATTN_RELFORK", "1") == "1"
# Paged conv decode: signal PDL dependents after y, overlapping the pool-persist tail.
_SCONV_EARLY_RELEASE = os.environ.get("INKLING_SCONV_EARLY_RELEASE", "0") == "1"


# Hetero KV (#647 byte-uniform slots): full layers keep their native half KV
# head count. Unconditional since 2026-07-15 (INKLING_HETERO_KV gate retired).
_HETERO_KV = True


# Checkpoint attention projections -> qkvr fused shard ids.
_QKVR_CHECKPOINT_SHARDS = (("wq_du", 0), ("wk_dv", 1), ("wv_dv", 2), ("wr_du", 3))

# Tensors with KV-head leading dim: replicated at load to the served uniform head count.
_KV_REPLICATED_SUFFIXES = (
    ".attn.wk_dv.weight",
    ".attn.wv_dv.weight",
    ".attn.k_sconv.weight",
    ".attn.v_sconv.weight",
)


# ModelOpt NVFP4: input_scale = amax/(448*6) (E4M3 max * E2M1 max); ckpt stores raw amax.
_NVFP4_AMAX_TO_SCALE = 448.0 * 6.0


def _translate_inkling_quant_pattern(pattern: str) -> str:
    is_regex = pattern.startswith("re:")
    body = pattern[3:] if is_regex else pattern
    replacements = (
        ("model\\.llm\\.embed_norm", "model\\.embed_norm"),
        ("model\\.llm\\.embed", "model\\.embed_tokens"),
        ("model\\.llm\\.unembed", "lm_head"),
        ("model\\.llm\\.", "model\\."),
        ("model\\.audio", "audio"),
        ("model\\.visual\\.layers", "visual\\.vision_encoder\\.layers"),
        ("model\\.visual", "visual\\.vision_encoder"),
        ("\\.attn\\.wq_du", "\\.attn\\.qkvr"),
        ("\\.attn\\.wk_dv", "\\.attn\\.qkvr"),
        ("\\.attn\\.wv_dv", "\\.attn\\.qkvr"),
        ("\\.attn\\.wr_du", "\\.attn\\.qkvr"),
        ("\\.mlp\\.w13_dn", "\\.mlp\\.gate_up_proj"),
        ("\\.mlp\\.w2_md", "\\.mlp\\.down_proj"),
    )
    if not is_regex:
        replacements = tuple(
            (old.replace("\\", ""), new.replace("\\", "")) for old, new in replacements
        )
    for old, new in replacements:
        body = body.replace(old, new)
    return f"re:{body}" if is_regex else body


def _translate_quant_exclusions(quant_config: QuantizationConfig) -> None:
    if quant_config is None or getattr(quant_config, "_inkling_excl_translated", False):
        return

    def translate(patterns: list[str], *, child_glob: bool = False) -> list[str]:
        out: list[str] = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern:
                continue
            translated = _translate_inkling_quant_pattern(pattern)
            out.append(translated)
            if child_glob and not translated.startswith("re:"):
                out.append(translated + ".*")
            if ".mlp.experts." in translated:
                out.append(translated.split(".mlp.experts.", 1)[0] + ".mlp")
        return list(dict.fromkeys(out))

    if getattr(quant_config, "exclude_modules", None):
        quant_config.exclude_modules = translate(
            quant_config.exclude_modules, child_glob=True
        )
    if getattr(quant_config, "ignored_layers", None):
        quant_config.ignored_layers = translate(quant_config.ignored_layers)
    quant_config._inkling_excl_translated = True


def _deinterleave_w13(weight: torch.Tensor) -> torch.Tensor:
    """Rows ``[g0, u0, g1, u1, ...]`` -> block ``[gate | up]`` on dim -2.

    The checkpoint stores SwiGLU w13 (dense, routed and shared experts — and
    their NVFP4 block scales) gate/up-interleaved; every kernel in this
    runtime consumes the concatenated layout.
    """
    rows = weight.shape[-2]
    assert rows % 2 == 0, f"w13 rows must be even, got {weight.shape}"
    return (
        weight.reshape(*weight.shape[:-2], rows // 2, 2, weight.shape[-1])
        .transpose(-3, -2)
        .reshape(weight.shape)
        .contiguous()
    )


def _make_param_loader(
    params_dict: dict[str, nn.Parameter],
    loaded: set[str],
    dropped: list[str],
) -> Callable[..., None]:
    """``load_param(name, w, shard_id=None)`` shared by base + NextN loaders."""

    def load_param(name: str, w: torch.Tensor, shard_id: int | None = None) -> None:
        if name not in params_dict:
            dropped.append(name)
            return
        param = params_dict[name]
        loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is not None:
            loader(param, w, shard_id)
        elif loader is default_weight_loader or param.data.shape == w.shape:
            default_weight_loader(param, w)
        else:
            loader(param, w)
        loaded.add(name)

    return load_param


def _load_block_param(
    load_param: Callable[..., None],
    name: str,
    w: torch.Tensor,
    interleaved: bool,
) -> bool:
    """Transformer-block checkpoint translation shared by base + NextN loaders:
    qkvr shard fusion, w13 gate/up split (+ de-interleave), w2 rename.
    Returns True if the tensor was consumed."""
    for weight_name, shard_id in _QKVR_CHECKPOINT_SHARDS:
        token = f".attn.{weight_name}."
        if token in name:
            load_param(name.replace(token, ".attn.qkvr."), w, shard_id)
            return True
    if ".mlp.w13_dn." in name:
        if interleaved:
            w = _deinterleave_w13(w)
        target = name.replace(".w13_dn.", ".gate_up_proj.")
        half = w.shape[0] // 2
        load_param(target, w.narrow(0, 0, half), 0)
        load_param(target, w.narrow(0, half, half), 1)
        return True
    if ".mlp.w2_md." in name:
        load_param(name.replace(".w2_md.", ".down_proj."), w)
        return True
    return False


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    """Long-context query scaling factor (config-gated; see InklingAttention)."""
    effective_n = (positions + 1).to(torch.float32)
    return 1.0 + alpha * torch.log(torch.clamp(effective_n / float(n_floor), min=1.0))


def _apply_log_scaling_tau(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return x * tau.to(dtype=x.dtype)


class InklingShortConvolution(nn.Module):
    """Residual per-channel causal FIR (sconv) over one stream.

    Reads/updates its rolling ``[W-1]`` state in the engine-side
    ``InklingConvStatePool`` (channel slice at ``channel_offset``), addressed by
    the conv metadata the ``InklingAttnBackend`` wrapper derives per forward.
    Uses the ``tokenspeed_kernel.ops.conv`` sconv kernels: fused
    conv+cache-shift on decode; prefill conv + final-state writeback on
    extend (residual and PAD_SLOT_ID handling are in-kernel).
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        stream: InklingConvStream,
        layer_id: int,
        channel_offset: int,
        tp_rank: int = 0,
        on_load: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stream = stream
        self.layer_id = layer_id
        self.channel_offset = channel_offset
        self.tp_rank = tp_rank
        self._on_load = on_load
        # Checkpoint Conv1d depthwise shape [dim, 1, W]; the last tap multiplies the current token.
        self.weight = nn.Parameter(
            torch.empty(dim, 1, kernel_size), requires_grad=False
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Narrow TP-sharded K/V sconv weights to this rank's shard."""
        if loaded_weight.shape[0] != param.shape[0]:
            shard = param.shape[0]
            loaded_weight = loaded_weight.narrow(0, self.tp_rank * shard, shard)
        assert (
            param.shape == loaded_weight.shape
        ), f"sconv weight shape mismatch: {param.shape} vs {loaded_weight.shape}"
        param.data.copy_(loaded_weight)
        if self._on_load is not None:
            self._on_load()

    def forward(self, x: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        weight = self.weight.squeeze(1)  # [dim, W]
        # K/V stream instances never run forward (InklingAttention fuses them
        # into one _sconv_apply call); a KeyError here means that broke.
        conv_group = {
            InklingConvStream.ATTN: "attnconv",
            InklingConvStream.MLP: "mlpconv",
        }[self.stream]
        return _sconv_apply(
            x,
            weight,
            ctx,
            self.layer_id,
            self.channel_offset,
            self.dim,
            conv_group=conv_group,
        )


def _sconv_apply(
    x: torch.Tensor,
    weight: torch.Tensor,
    ctx: ForwardContext,
    layer_id: int,
    channel_offset: int,
    dim: int,
    conv_group: str | None = None,
) -> torch.Tensor:
    """Run one sconv stream (or several adjacent ones fused) over ``x``.

    ``weight`` is ``[dim, W]``; ``channel_offset``/``dim`` select the state
    channels in the layer's conv pool buffer. Fusing adjacent streams (K+V)
    is just a wider ``dim`` with concatenated weights.
    """
    backend = ctx.attn_backend
    md = backend.conv_metadata

    geo = backend.conv_columns if md.col_page_table is not None else None
    if geo is not None and conv_group in ("attnconv", "mlpconv"):
        if geo.get("hidden_group_of_layer") is None:
            geo = None  # hidden sites stay rolling without their groups
        else:
            # Hidden-conv-as-swa: ATTN site's columns ride the layer's K slot, MLP site's its V slot.
            table = md.col_page_table[geo["hidden_group_of_layer"][layer_id]]
            bt = geo["hidden_block_tokens"]
            cols = ctx.token_to_kv_pool.conv_slot_view(
                layer_id, "k" if conv_group == "attnconv" else "v", bt, dim
            )
            if md.is_decode:
                return sconv_decode_paged(
                    x,
                    weight,
                    cols,
                    table,
                    md.col_seq_lens,
                    block_tokens=bt,
                    col_offset=0,
                    activation=None,
                    use_residual=True,
                    enable_pdl=pdl_enabled(),
                    early_release=_SCONV_EARLY_RELEASE,
                )
            return sconv_prefill_paged(
                x,
                weight,
                cols,
                table,
                md.seq_idx,
                md.query_start_loc,
                md.col_prefix_lens,
                block_tokens=bt,
                col_offset=0,
                lcm_align=geo["lcm_align"],
                activation=None,
                use_residual=True,
            )

    if geo is not None and conv_group == "kvconv":
        # kvconv-as-swa: columns stay 3D views — a 2D reshape would silently COPY, breaking persistence.
        table = md.col_page_table[geo["conv_group_of_layer"][layer_id]]
        k_cols, v_cols = ctx.token_to_kv_pool.kvconv_slot_views_for_layer(
            layer_id, geo["block_tokens"]
        )
        half = dim // 2
        if md.is_decode:
            return sconv_decode_paged(
                x,
                weight,
                k_cols,
                table,
                md.col_seq_lens,
                block_tokens=geo["block_tokens"],
                col_offset=0,
                col_pool2=v_cols,
                half_d=half,
                activation=None,
                use_residual=True,
                enable_pdl=pdl_enabled(),
                early_release=_SCONV_EARLY_RELEASE,
            )
        return sconv_prefill_paged(
            x,
            weight,
            k_cols,
            table,
            md.seq_idx,
            md.query_start_loc,
            md.col_prefix_lens,
            block_tokens=geo["block_tokens"],
            col_offset=0,
            col_pool2=v_cols,
            half_d=half,
            lcm_align=geo["lcm_align"],
            activation=None,
            use_residual=True,
        )

    # Channel slice of the layer's [slots, W-1, conv_dim] state buffer.
    # Draft lookback window passes re-run the last D committed rows, so
    # their conv init state is the LAGGED window (D rows behind); the
    # backend's valid_len update then advances both windows off it.
    pool_state = (
        backend.draft_lag_conv_state_wd(layer_id)
        if md.lookback > 0
        else backend.conv_pool.layer_state_wd(layer_id)
    )
    state = pool_state[:, :, channel_offset : channel_offset + dim]

    if md.is_decode:
        # Fused conv + residual + in-place cache shift.
        return sconv_decode(
            x,
            weight,
            state,
            md.cache_indices,
            activation=None,
            use_residual=True,
            enable_pdl=pdl_enabled(),
        )
    y = sconv_prefill(
        x,
        weight,
        state,
        md.query_start_loc,
        md.seq_idx,
        md.cache_indices,
        md.has_initial_state,
        activation=None,
        use_residual=True,
    )
    # Backend owns the window write: mode-dependent under spec decoding (verify stash / catch-up).
    backend.apply_conv_state_update(
        x,
        state,
        md,
        layer_id,
        channel_offset,
        dim,
        accept_lengths=getattr(ctx, "accept_lengths", None),
    )
    return y


class InklingRelLogitsProj(nn.Module):
    """Project per-token R features to relative-distance bias logits."""

    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)

    def forward(self, r_out: torch.Tensor) -> torch.Tensor:
        # r_out: [T, num_heads, d_rel] -> rel_logits [T, num_heads, rel_extent]
        return torch.einsum("thd,de->the", r_out, self.proj)


class InklingAttention(nn.Module):
    """Relative attention with QKVR fused projection and K/V sconv streams."""

    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        layer_id: int,
        is_local: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.stream_fork = StreamFork(alt_stream)
        attn_tp_rank = mapping.attn.tp_rank
        attn_tp_size = mapping.attn.tp_size
        attn_tp_group = mapping.attn.tp_group

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.d_rel = config.d_rel
        self.is_local = is_local
        self.layer_id = layer_id
        # Inkling trains with 1/d attention scale (muP + QK-norm), not 1/sqrt(d).
        self.scaling = 1.0 / self.head_dim

        total_heads = config.num_attention_heads
        total_kv_heads = inkling_kv_heads_for_layer(config, layer_id, _HETERO_KV)
        assert total_heads % attn_tp_size == 0
        self.num_tp_heads = total_heads // attn_tp_size
        if total_kv_heads >= attn_tp_size:
            assert total_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % total_kv_heads == 0
        self.num_tp_kv_heads = max(1, total_kv_heads // attn_tp_size)
        kv_total_for_sizing = max(total_kv_heads, attn_tp_size)

        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [
                self.head_dim * total_heads,
                self.head_dim * kv_total_for_sizing,
                self.head_dim * kv_total_for_sizing,
                self.d_rel * total_heads,
            ],
            bias=config.q_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("qkvr", prefix),
        )
        self.wo_ud = RowParallelLinear(
            self.head_dim * total_heads,
            config.hidden_size,
            bias=config.o_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            tp_group=attn_tp_group,
            reduce_results=True,
            prefix=add_prefix("wo_ud", prefix),
        )

        # Local layers: bias table exactly covers the window.
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.rel_logits_proj = InklingRelLogitsProj(self.d_rel, self.rel_extent)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        kv_dim = self.head_dim * self.num_tp_kv_heads
        layout = inkling_conv_stream_layout(config, attn_tp_size)
        self.use_sconv = config.use_sconv
        if self.use_sconv:
            self.k_sconv = InklingShortConvolution(
                kv_dim,
                config.sconv_kernel_size,
                InklingConvStream.K,
                layer_id,
                channel_offset=layout[InklingConvStream.K][0],
                tp_rank=attn_tp_rank,
                on_load=self._invalidate_kv_sconv_cache,
            )
            self.v_sconv = InklingShortConvolution(
                kv_dim,
                config.sconv_kernel_size,
                InklingConvStream.V,
                layer_id,
                channel_offset=layout[InklingConvStream.V][0],
                tp_rank=attn_tp_rank,
                on_load=self._invalidate_kv_sconv_cache,
            )
        else:
            self.k_sconv = None
            self.v_sconv = None
        # Fused K+V sconv: adjacent pool regions + lazily cached concat of the frozen tap weights.
        self._kv_channel_offset = layout[InklingConvStream.K][0]
        self._kv_sconv_w_cache: torch.Tensor | None = None

        self.attn = PagedAttention(
            self.num_tp_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_tp_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(config.sliding_window_size - 1) if is_local else -1,
            # Group ids == config.paged_cache_layer_types labels (sliding sub-groups included).
            group_id=config.paged_cache_layer_types[layer_id],
        )

        self.q_size = self.head_dim * self.num_tp_heads
        self.kv_size = kv_dim
        self.r_size = self.d_rel * self.num_tp_heads

    def _invalidate_kv_sconv_cache(self) -> None:
        self._kv_sconv_w_cache = None

    def _merged_kv_sconv_weight(self) -> torch.Tensor:
        """Cached ``[2*kv_dim, W]`` concat of the frozen K/V sconv taps.

        Built lazily on first forward; the sconv weight loaders invalidate the
        cache on every (re-)load via ``_invalidate_kv_sconv_cache``.
        """
        w = self._kv_sconv_w_cache
        if w is None:
            w = (
                torch.cat([self.k_sconv.weight, self.v_sconv.weight], dim=0)
                .squeeze(1)
                .contiguous()
            )
            self._kv_sconv_w_cache = w
        return w

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        log_scaling_tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkvr, _ = self.qkvr(hidden_states)
        q, kv, r = qkvr.split([self.q_size, 2 * self.kv_size, self.r_size], dim=-1)

        # Fork the R-slice-only rel projection onto the aux stream (capture
        # only; joins before attention). INKLING_ATTN_RELFORK=0 = serial.
        with self.stream_fork.scope(
            enable=_ATTN_RELFORK and get_is_capture_mode()
        ) as fork:
            with fork.branch():
                rel_logits = self.rel_logits_proj(
                    r.view(num_tokens, self.num_tp_heads, self.d_rel)
                )
                if log_scaling_tau is not None and not self.is_local:
                    rel_logits = _apply_log_scaling_tau(
                        rel_logits, log_scaling_tau.view(-1, 1, 1)
                    )

            # K/V are adjacent in the qkvr output AND the conv pool, so both sconv streams fuse into one call.
            if self.use_sconv:
                kv = _sconv_apply(
                    kv.contiguous(),
                    self._merged_kv_sconv_weight(),
                    ctx,
                    self.layer_id,
                    self._kv_channel_offset,
                    2 * self.kv_size,
                    conv_group="kvconv",
                )
            k = kv[:, : self.kv_size]
            v = kv[:, self.kv_size :]

            # Fused q/k RMSNorm returns contiguous q/k; v stays a strided view (KV scatter/FA4 allow it).
            q, k = qk_rmsnorm(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                enable_pdl=pdl_enabled(),
            )

            if log_scaling_tau is not None and not self.is_local:
                q = _apply_log_scaling_tau(q, log_scaling_tau.view(-1, 1))

        attn_output = self.attn(
            q,
            k,
            v,
            ctx,
            out_cache_loc,
            rel_logits=rel_logits.contiguous(),
        )
        output, _ = self.wo_ud(attn_output)
        return output


class InklingDenseMLP(nn.Module):
    """Dense SwiGLU MLP for the first ``dense_mlp_idx`` layers."""

    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.dense_intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.dense_intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            reduce_results=True,
            prefix=add_prefix("down_proj", prefix),
        )
        self.global_scale = (
            nn.Parameter(torch.empty(1), requires_grad=False)
            if config.use_global_scale
            else None
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        if self.global_scale is not None:
            x = x * self.global_scale
        return x


class InklingGate(nn.Module):
    """Sigmoid router with gate-bias-in-selection and logsigmoid weighting.

    Selection: ``sigmoid(routed logits) + bias`` -> top-k over routed only.
    Weighting: raw logits of the selected routed experts concatenated with
    the (last ``n_shared``) shared logits, logsigmoid-normalized to sum 1,
    scaled by ``route_scale`` (and optional ``global_scale``). Shared experts
    thus compete for probability mass without entering top-k (a "sink").

    Selection and weighting run in one fused Triton kernel
    (``tokenspeed_kernel.ops.moe.triton.inkling_topk``, reached through
    ``TopK`` / ``select_experts``), which implements the reference's
    deterministic lowest-index tie-breaking and computes the sink
    normalization in linear space (identical to the reference's
    logsigmoid/logsumexp form; see the kernel docstring).
    """

    def __init__(self, config: InklingModelConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        assert config.gate_activation == "sigmoid", "Inkling v1 supports sigmoid gate"
        assert config.norm_after_topk and config.shared_expert_sink
        self.weight = nn.Parameter(
            torch.empty(self.n_routed + self.n_shared, config.hidden_size),
            requires_grad=False,
        )
        assert config.use_gate_bias
        self.bias = nn.Parameter(
            torch.empty(self.n_routed, dtype=torch.float32), requires_grad=False
        )
        assert config.use_global_scale
        self.global_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        # Router-gemm eligibility as deepseek_v3.MoEGate; n_routed <= 256 asserted below.
        self.use_dsv3_router_gemm = (
            _is_hopper_plus
            and self.weight.dtype in (torch.bfloat16, torch.float32)
            and config.hidden_size in _DSV3MoEGate._DSV3_ROUTER_GEMM_HIDDEN
        )
        assert 1 <= self.top_k <= 8
        assert self.n_routed <= 256
        self.topk = TopK(
            self.top_k,
            renormalize=False,
            correction_bias=self.bias,
            routed_scaling_factor=self.route_scale,
            num_sink_experts=self.n_shared,
            sink_global_scale=self.global_scale,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (weights [T,k+S] f32, topk_ids [T,k] i32,
        raw logits [T,R+S]).

        ``weights`` holds the selected-routed weights (first ``k`` columns)
        and the shared sink gammas (last ``S`` columns) from one joint
        normalization; callers slice as needed."""
        if self.use_dsv3_router_gemm and x.size(0) > 0:
            logits = dsv3_router_gemm(
                x, self.weight, out_dtype=torch.float32, enable_pdl=pdl_enabled()
            )
        else:
            logits = F.linear(x, self.weight, None)

        # Full logits (routed + shared sink) pass as router_logits; the routing fn consumes the tail.
        topk_output = self.topk(x, logits)
        return topk_output.topk_weights, topk_output.topk_ids, logits


class InklingSharedExperts(nn.Module):
    """Batched dense shared experts weighted by per-token sink gammas.

    Weights are stored in the non-interleaved ``[gate | up]`` convention;
    interleaved checkpoints are de-interleaved at load time.
    """

    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = mapping.attn.tp_size
        assert config.intermediate_size % tp_size == 0
        self.tp_rank = mapping.attn.tp_rank
        self.n_shared = config.n_shared_experts
        self.intermediate_per_rank = config.intermediate_size // tp_size
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.n_shared, 2 * self.intermediate_per_rank, config.hidden_size
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(self.n_shared, config.hidden_size, self.intermediate_per_rank),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, {"weight_loader": self._load_w13})
        set_weight_attrs(self.w2_weight, {"weight_loader": self._load_w2})
        self.act_fn = SiluAndMul()

    def _load_w13(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Narrow a full ``[S, 2I, H]`` (gate|up block layout) tensor to this
        rank's per-half slice. De-interleaving happens in ``load_weights``."""
        per_rank = param.shape[1] // 2
        full = loaded_weight.shape[1] // 2
        if per_rank != full:
            gate = loaded_weight.narrow(1, self.tp_rank * per_rank, per_rank)
            up = loaded_weight.narrow(1, full + self.tp_rank * per_rank, per_rank)
            loaded_weight = torch.cat([gate, up], dim=1)
        param.data.copy_(loaded_weight)

    def _load_w2(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        per_rank = param.shape[2]
        if loaded_weight.shape[2] != per_rank:
            loaded_weight = loaded_weight.narrow(2, self.tp_rank * per_rank, per_rank)
        param.data.copy_(loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
        gammas: torch.Tensor | None = None,
        do_finalize: bool = True,
    ) -> torch.Tensor:
        # Batched GEMM over S shared experts, batch stride 0 on x: [S,T,H] @ [S,H,2I] -> [S,T,2I].
        h = torch.bmm(
            x.unsqueeze(0).expand(self.n_shared, -1, -1),
            self.w13_weight.transpose(1, 2),
        )
        h = self.act_fn(h)
        if not do_finalize:
            # Un-weighted [S, T, H] per-expert outputs; caller applies sink gammas in the fused finalize.
            return torch.bmm(h, self.w2_weight.transpose(1, 2))
        assert gammas is not None
        # Fold gammas before down-proj (linear, so equivalent); scaling [S,T,I] beats weighting [S,T,H].
        h = h * gammas.to(h.dtype).t().unsqueeze(-1)
        return torch.bmm(h, self.w2_weight.transpose(1, 2)).sum(dim=0)


class InklingTorchMoEExperts(nn.Module):
    """Torch-native routed experts (correctness fallback, ``INKLING_TORCH_MOE=1``).

    Loops over activated experts; only suitable for tiny test configs.
    """

    def __init__(self, config: InklingModelConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_experts, 2 * config.intermediate_size, config.hidden_size
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, config.intermediate_size),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros_like(x)
        for expert_id in topk_ids.unique():
            token_idx, k_idx = (topk_ids == expert_id).nonzero(as_tuple=True)
            xe = x[token_idx]
            h = xe @ self.w13_weight[expert_id].t().to(x.dtype)
            gate, up = h.chunk(2, dim=-1)
            ye = (F.silu(gate) * up) @ self.w2_weight[expert_id].t().to(x.dtype)
            out.index_add_(
                0, token_idx, ye * topk_weights[token_idx, k_idx, None].to(x.dtype)
            )
        return out


class InklingSparseMoeBlock(nn.Module):
    """Shared-expert-sink MoE: InklingGate + routed experts + weighted shared."""

    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.mapping = mapping
        self.stream_fork = StreamFork(alt_stream)
        self.gate = InklingGate(config)
        self.shared_experts = InklingSharedExperts(
            config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
        )
        self.use_torch_moe = os.environ.get("INKLING_TORCH_MOE", "0") == "1"
        if self.use_torch_moe:
            self.experts: nn.Module = InklingTorchMoEExperts(config)
        else:
            self.experts = MoELayer(
                top_k=config.num_experts_per_tok,
                num_experts=config.n_routed_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                layer_index=layer_id,
                # Deepseek-convention MLP prefix: MoELayer probes quant exclusion via "<prefix>.experts".
                prefix=prefix,
                tp_rank=mapping.moe.tp_rank,
                tp_size=mapping.moe.tp_size,
                ep_rank=mapping.moe.ep_rank,
                ep_size=mapping.moe.ep_size,
                # Inkling routing is computed in InklingGate; the MoE kernel consumes the precomputed top-k plan.
                routing_mode="precomputed_topk",
            )
            assert not self.experts.support_routing
        self.comm_manager = CommManager(
            mapping=mapping, layer_id=layer_id, is_moe=True, prev_is_moe=True
        )
        # sconv shifts along the token dim, so this block must return full token rows (no reduce-scatter).
        assert self.comm_manager.use_all_reduce(is_moe=True), (
            "Inkling requires attn_tp_size == moe tp_size * ep_size "
            "(all-reduce MoE communication pattern)"
        )

    def forward(self, x: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        # weights: [T, k + S] — selected-routed weights then shared sink gammas (joint normalization).
        top_k = self.gate.top_k
        deferred = not self.use_torch_moe and self.experts.supports_deferred_finalize
        if deferred:
            num_global_tokens, max_tokens_per_gpu = self.comm_manager.get_num_tokens(
                ctx
            )
            # Shared experts depend only on x: fork before the gate (capture-only, allocator stream safety).
            with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
                with fork.branch():
                    shared_out = self.shared_experts(x, do_finalize=False)
                weights, topk_ids, router_logits = self.gate(x)
                topk_output = StandardTopKOutput(
                    # Non-contiguous view is fine: the routed kernel repacks the weights.
                    topk_weights=weights[:, :top_k],
                    topk_ids=topk_ids,
                    router_logits=router_logits,
                )
                gemm2_out, _, expanded_idx = self.experts(
                    hidden_states=x,
                    topk_output=topk_output,
                    num_global_tokens=num_global_tokens,
                    max_num_tokens_per_gpu=max_tokens_per_gpu,
                    do_finalize=False,
                )
            # One fused epilogue applies routed weights + shared sink gammas (full [T, k+S]) and sums.
            out = moe_finalize_fuse_shared(
                gemm2_out,
                expanded_idx,
                weights,
                shared_out,
                top_k=top_k,
                enable_pdl=pdl_enabled(),
            )
        else:
            weights, topk_ids, router_logits = self.gate(x)
            routed_weights = weights[:, :top_k].contiguous()
            shared_gammas = weights[:, top_k:].contiguous()
            if self.use_torch_moe:
                routed_out = self.experts(x, routed_weights, topk_ids)
                shared_out = self.shared_experts(x, shared_gammas)
            else:
                num_global_tokens, max_tokens_per_gpu = (
                    self.comm_manager.get_num_tokens(ctx)
                )
                topk_output = StandardTopKOutput(
                    topk_weights=routed_weights,
                    topk_ids=topk_ids,
                    router_logits=router_logits,
                )
                with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
                    routed_out = self.experts(
                        hidden_states=x,
                        topk_output=topk_output,
                        num_global_tokens=num_global_tokens,
                        max_num_tokens_per_gpu=max_tokens_per_gpu,
                    )
                    with fork.branch():
                        shared_out = self.shared_experts(x, shared_gammas)
            out = routed_out + shared_out
        # Rank-local partial sum (TP/EP sharded); combine across the MoE group in one collective.
        out, _ = self.comm_manager.post_moe_comm(out, None, ctx)
        return out


class InklingDecoderLayer(nn.Module):

    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        is_local = layer_id in config.local_layer_ids
        self.attn = InklingAttention(
            config,
            mapping,
            layer_id=layer_id,
            is_local=is_local,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            alt_stream=alt_stream,
        )
        if layer_id < config.dense_mlp_idx:
            self.mlp: nn.Module = InklingDenseMLP(
                config, mapping, quant_config, prefix=add_prefix("mlp", prefix)
            )
            self.is_moe = False
        else:
            self.mlp = InklingSparseMoeBlock(
                config,
                mapping,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
            self.is_moe = True

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        layout = inkling_conv_stream_layout(config, mapping.attn.tp_size)
        if config.use_sconv:
            self.attn_sconv = InklingShortConvolution(
                config.hidden_size,
                config.sconv_kernel_size,
                InklingConvStream.ATTN,
                layer_id,
                channel_offset=layout[InklingConvStream.ATTN][0],
            )
            self.mlp_sconv = InklingShortConvolution(
                config.hidden_size,
                config.sconv_kernel_size,
                InklingConvStream.MLP,
                layer_id,
                channel_offset=layout[InklingConvStream.MLP][0],
            )
        else:
            self.attn_sconv = None
            self.mlp_sconv = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        log_scaling_tau: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ctx.forward_mode.is_idle():
            return hidden_states, residual

        if residual is None:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        else:
            hidden_states, residual = self.attn_norm(hidden_states, residual)

        attn_out = self.attn(
            hidden_states,
            ctx,
            out_cache_loc,
            log_scaling_tau=log_scaling_tau,
        )
        if self.attn_sconv is not None:
            attn_out = self.attn_sconv(attn_out, ctx)
        mlp_input, residual = self.mlp_norm(attn_out, residual)

        if self.is_moe:
            mlp_out = self.mlp(mlp_input, ctx)
        else:
            mlp_out = self.mlp(mlp_input)
        if self.mlp_sconv is not None:
            mlp_out = self.mlp_sconv(mlp_out, ctx)
        return mlp_out, residual


def _prime_factors(n: int) -> list[int]:
    """Prime factors of ``n`` in ascending order (n >= 1)."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    factors: list[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def _min_cost_increasing_assignment(cost: list[list[float]]) -> list[int]:
    """Min-cost injective row->column assignment, columns strictly increasing.

    Scipy-free replacement for ``scipy.optimize.linear_sum_assignment`` in
    :func:`inkling_plan_out_scales`: both the ideal log-scales (rows) and the
    realizable log-scales (columns) are sorted ascending, so the
    absolute-difference cost matrix is a Monge matrix and an optimal
    assignment with strictly increasing column indices exists. A simple DP
    over increasing column choices therefore attains the Hungarian optimum.
    At exact cost ties scipy may return a different (even non-monotone)
    optimal assignment; ties require the ideal log-scale to fall exactly
    midway between two realizable scales, which real configs (verified for
    the shipped checkpoint) do not hit.

    Args:
        cost: Dense cost matrix as nested lists, ``len(cost) <= len(cost[0])``.

    Returns:
        One column index per row, strictly increasing, minimizing total cost.
    """
    n_rows, n_cols = len(cost), len(cost[0])
    assert n_rows <= n_cols
    inf = float("inf")
    # dp[j]: best total cost with the current row assigned to column j.
    dp = [cost[0][j] for j in range(n_cols)]
    parents: list[list[int]] = []
    for i in range(1, n_rows):
        new_dp = [inf] * n_cols
        parent = [-1] * n_cols
        best, best_j = inf, -1
        for j in range(i, n_cols):
            if dp[j - 1] < best:
                best, best_j = dp[j - 1], j - 1
            new_dp[j] = best + cost[i][j]
            parent[j] = best_j
        dp = new_dp
        parents.append(parent)
    j = min(range(n_cols), key=lambda k: dp[k])
    out = [j]
    for parent in reversed(parents):
        j = parent[j]
        out.append(j)
    return out[::-1]


def inkling_plan_out_scales(
    temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int = 3
) -> list[tuple[int, int, int, int]]:
    """Plan the per-layer (t, h, w, c) folding schedule of the hMLP encoder.

    Port of the reference ``plan_out_scales``: enumerate the realizable fold
    scales (spatial prime factors first, then temporal), round channel counts
    up to multiples of 64, and pick ``n_layers + 1`` of them whose log-sizes
    best match a geometric progression from 1 to the full patch volume. The
    first and last entries are always the identity and the full-patch scale.

    Args:
        temporal_patch_size: Temporal extent of one input patch.
        patch_size: Spatial (height == width) extent of one input patch.
        n_layers: Number of Linear layers in the encoder.
        n_channels: Input channels (3 for RGB).

    Returns:
        ``n_layers + 1`` tuples ``(t, h, w, c)``: the fold state entering each
        layer, plus the final state.
    """
    if patch_size <= 1:
        raise ValueError("patch_size must be greater than 1")

    def _round_up(x: int) -> int:
        return math.ceil(x / 64) * 64

    last_h_scale = 1
    scales: list[tuple[int, int, int, int]] = [(1, 1, 1, n_channels)]
    for pscale in _prime_factors(patch_size)[::-1]:
        last_h_scale *= pscale
        scales.append(
            (1, last_h_scale, last_h_scale, _round_up((last_h_scale**2) * n_channels))
        )
    last_t_scale = 1
    for tscale in _prime_factors(temporal_patch_size)[::-1]:
        last_t_scale *= tscale
        scales.append(
            (
                last_t_scale,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels * last_t_scale),
            )
        )

    log_size_reduction = [math.log(t * h * w) for t, h, w, _ in scales]
    span = math.log(patch_size * patch_size * temporal_patch_size * n_channels)
    # Match np.linspace bit-exactly so argmin tie-breaks agree with the reference.
    step = span / n_layers
    log_ideal_scales = [i * step for i in range(n_layers + 1)]
    log_ideal_scales[-1] = span
    cost = [
        [abs(ideal - real) for real in log_size_reduction] for ideal in log_ideal_scales
    ]

    if n_layers >= len(scales):
        idxs = [min(range(len(scales)), key=lambda j: row[j]) for row in cost]
    else:
        idxs = _min_cost_increasing_assignment(cost)

    assert len(idxs) >= 2
    idxs[0] = 0
    idxs[-1] = len(scales) - 1
    return [scales[i] for i in idxs]


def _fold_timespace_to_depth(
    vision_patches_bthwc: torch.Tensor, t_fold: int, hw_fold: int
) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, T/t, H/hw, W/hw, C * t * hw**2)."""
    B, T, H, W, C = vision_patches_bthwc.shape
    assert T % t_fold == 0, f"Temporal dim {T} not divisible by {t_fold}"
    assert H % hw_fold == 0, f"Height dim {H} not divisible by {hw_fold}"
    assert W % hw_fold == 0, f"Width dim {W} not divisible by {hw_fold}"
    x = vision_patches_bthwc.reshape(
        B, T // t_fold, t_fold, H // hw_fold, hw_fold, W // hw_fold, hw_fold, C
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(
        B, T // t_fold, H // hw_fold, W // hw_fold, t_fold * hw_fold * hw_fold * C
    )


class InklingHMLPPatchEncoder(nn.Module):
    """Hierarchical-MLP patch encoder (torch-native port of the reference).

    Consumes pre-extracted patches ``(num_patches, T, H, W, C)`` (the SMG
    gateway does the pixel->patch preprocessing) and emits one
    ``decoder_dmodel`` embedding per patch: each layer folds part of the
    patch's time/space interior into the channel axis, applies a bias-free
    Linear, and (except on the last layer) RMSNorm + GELU.
    """

    def __init__(self, config: InklingVisionConfig):
        super().__init__()
        self.decoder_dmodel = config.decoder_dmodel
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.n_channels = config.n_channels
        self.n_layers = config.n_layers
        self.use_vision_norm = config.use_vision_norm

        self.scales = inkling_plan_out_scales(
            self.temporal_patch_size, self.patch_size, self.n_layers, self.n_channels
        )
        self.layers = nn.ModuleDict()
        for i, (start_scale, end_scale) in enumerate(
            zip(self.scales[:-1], self.scales[1:])
        ):
            shuffle_mult = (
                (end_scale[0] // start_scale[0])
                * (end_scale[1] // start_scale[1])
                * (end_scale[2] // start_scale[2])
            )
            out_features = (
                self.decoder_dmodel if i == self.n_layers - 1 else end_scale[3]
            )
            self.layers[f"linear_{i}"] = nn.Linear(
                start_scale[3] * shuffle_mult, out_features, bias=False
            )
            if i < self.n_layers - 1:
                self.layers[f"norm_{i}"] = RMSNorm(end_scale[3])

        self.final_norm: RMSNorm | None = None
        if self.use_vision_norm:
            self.final_norm = RMSNorm(self.decoder_dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patches ``(num_patches, T, H, W, C)`` to
        ``(num_patches, decoder_dmodel)``."""
        num_patches, T, H, W, _ = x.shape
        for i, (start_scale, end_scale) in enumerate(
            zip(self.scales[:-1], self.scales[1:])
        ):
            t_fold = end_scale[0] // start_scale[0]
            hw_fold = end_scale[1] // start_scale[1]
            if hw_fold > 1 or t_fold > 1:
                x = _fold_timespace_to_depth(x, t_fold, hw_fold)
            assert x.shape[1:-1] == (
                T // end_scale[0],
                H // end_scale[1],
                W // end_scale[2],
            )
            x = self.layers[f"linear_{i}"](x)
            if i < self.n_layers - 1:
                norm = self.layers[f"norm_{i}"]
                # The RMSNorm kernel expects 2-D input; fold the spatial axes and restore after.
                x = norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
                x = F.gelu(x)
        if self.final_norm is not None:
            x = self.final_norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return x.reshape(num_patches, -1)


class InklingVisionTower(nn.Module):
    """Vision tower: wraps the hMLP patch encoder (checkpoint key ``visual``)."""

    def __init__(self, config: InklingVisionConfig, prefix: str = ""):
        del prefix
        super().__init__()
        assert config.vision_encoder_type == "hmlp", config.vision_encoder_type
        self.vision_encoder = InklingHMLPPatchEncoder(config)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(vision_features)


class InklingAudioTower(nn.Module):
    """dMel audio tower (checkpoint key ``audio``).

    Each audio frame arrives as ``n_mel_bins`` integer dMel values in
    ``[0, mel_vocab_size)``. Every (bin, value) pair owns an embedding row at
    index ``bin * mel_vocab_size + value``; the frame embedding is the sum of
    its per-bin rows, optionally RMSNorm-ed (``use_audio_norm``).
    """

    def __init__(self, config: InklingAudioConfig, prefix: str = ""):
        del prefix
        super().__init__()
        assert config.audio_mode == "dmel", config.audio_mode
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.encoder = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size, config.decoder_dmodel
        )
        self.final_norm: RMSNorm | None = None
        if config.use_audio_norm:
            self.final_norm = RMSNorm(config.decoder_dmodel, eps=1e-6)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Embed dMel frames ``(num_tokens, n_mel_bins)`` (integer values) to
        ``(num_tokens, decoder_dmodel)``."""
        assert audio_features.shape[1] == self.n_mel_bins, audio_features.shape
        device = self.encoder.weight.device
        dmel = audio_features.to(device=device, dtype=torch.long)
        bin_offsets = (
            torch.arange(self.n_mel_bins, device=device) * self.mel_vocab_size
        ).unsqueeze(0)
        # Fused lookup+sum: the unfused form materializes an intermediate of
        # ``(num_tokens * n_mel_bins, decoder_dmodel)``, i.e. ``n_mel_bins``
        # (80) times the output, which dominates peak memory on long clips --
        # ~0.95 MB per audio token at decoder_dmodel=6144.
        hidden_states = F.embedding_bag(
            bin_offsets + dmel, self.encoder.weight, mode="sum"
        )
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        return hidden_states


class InklingTextModel(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.padded_vocab_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )

        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def get_layer(idx: int, prefix: str) -> InklingDecoderLayer:
            return InklingDecoderLayer(
                config,
                mapping,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=add_prefix("layers", prefix)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def mm_text_embedding(self):
        """Text-token embedding for the MM embed path.

        Folds ``embed_norm`` into the lookup so the ``VisionEmbedder`` norms
        text tokens while the tower features it splices in (which carry their
        own final norm) stay un-renormed — matching the reference's
        ``general_mm_embed_routine`` semantics. The ``num_embeddings``
        attribute is the clamp bound the embedder uses for the
        hash-derived multimodal pad ids in ``input_ids``.
        """
        embed_tokens, embed_norm = self.embed_tokens, self.embed_norm

        def embed(input_ids: torch.Tensor) -> torch.Tensor:
            embeds = embed_tokens(input_ids)
            return embed_norm(embeds) if embed_norm is not None else embeds

        embed.num_embeddings = embed_tokens.num_embeddings
        embed.embedding_dim = embed_tokens.embedding_dim
        embed.weight = embed_tokens.weight
        return embed

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            if self.embed_norm is not None:
                hidden_states = self.embed_norm(hidden_states)
        else:
            # embed_norm is already folded into mm_text_embedding; tower features carry their own norm.
            hidden_states = input_embeds

        log_scaling_tau = None
        if self.config.log_scaling_n_floor is not None:
            log_scaling_tau = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                ctx,
                out_cache_loc,
                log_scaling_tau=log_scaling_tau,
            )
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, None


class InklingForConditionalGeneration(nn.Module):
    """Inkling top-level model (text decoder + optional audio/vision towers)."""

    def __init__(
        self,
        config: InklingMMConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ):
        # Towers are attention-free; the knob is just a VLM-wrapper interface requirement.
        del mm_attention_backend
        super().__init__()

        self.config = config
        self.mapping = mapping
        text_config = config.get_text_config()
        self.text_config = text_config
        _translate_quant_exclusions(quant_config)
        if (
            quant_config is not None
            and quant_config.get_name() == "mxfp4"
            and getattr(quant_config, "is_checkpoint_mxfp4_serialized", False)
            and getattr(quant_config, "quant_method", None) != "quark"
        ):
            raise ValueError("Inkling MXFP4 checkpoints must be quantized by Quark")
        self.quant_config = quant_config

        self.model = InklingTextModel(
            text_config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # Tower presence is a checkpoint fact; --language-model-only turns them off at runtime.
        self.is_multimodal_active = is_multimodal_active
        audio_enabled = (
            is_multimodal_active and config.audio_config.decoder_dmodel is not None
        )
        vision_enabled = (
            is_multimodal_active and config.vision_config.decoder_dmodel is not None
        )
        if audio_enabled:
            assert (
                config.audio_config.decoder_dmodel == text_config.hidden_size
            ), "audio decoder_dmodel must match the text hidden size"
        if vision_enabled:
            assert (
                config.vision_config.decoder_dmodel == text_config.hidden_size
            ), "vision decoder_dmodel must match the text hidden size"
        self.audio = (
            InklingAudioTower(config.audio_config, prefix=add_prefix("audio", prefix))
            if audio_enabled
            else None
        )
        self.visual = (
            InklingVisionTower(
                config.vision_config, prefix=add_prefix("visual", prefix)
            )
            if vision_enabled
            else None
        )
        self.vision_embedder = (
            VisionEmbedder(encoder_mapping=mapping.vision)
            if (self.audio is not None or self.visual is not None)
            else None
        )

        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size,
            text_config.hidden_size,
            org_num_embeddings=text_config.padded_vocab_size,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("lm_head", prefix),
        )
        # All-gather vocab-sharded logits before sampling; do_argmax would bypass the pad masking below.
        self.logits_processor = LogitsProcessor(
            text_config,
            skip_all_gather=mapping.attn.has_dp,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Weights the MTP draft model shares (see inkling_nextn.py)."""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        """Rewrite each placeholder run to the item's content-derived pad id."""
        return pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Encode image patch features to LM embeddings.

        Args:
            items: Items whose ``feature`` is a patch tensor
                ``(num_patches, T, H, W, C)`` as shipped by the gateway.

        Returns:
            ``(total_patches, hidden_size)`` embeddings, one row per patch,
            concatenated in item order.
        """
        patches = torch.cat([item.feature for item in items], dim=0)
        param = next(self.visual.parameters())
        return self.visual(patches.to(device=param.device, dtype=param.dtype))

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Encode dMel audio features to LM embeddings.

        Args:
            items: Items whose ``feature`` is an integer dMel tensor
                ``(num_tokens, n_mel_bins)`` with values in
                ``[0, mel_vocab_size)``.

        Returns:
            ``(total_tokens, hidden_size)`` embeddings concatenated in item
            order.
        """
        dmel = torch.cat([item.feature for item in items], dim=0)
        return self.audio(dmel)

    def _embed_multimodal(self, ctx, input_ids, multimodal_context):
        """Run the shared MM embed pipeline; ``None`` -> text-only path."""
        if (
            multimodal_context is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        if self.vision_embedder is None:
            # Out-of-vocab MM pad ids would otherwise die as a cryptic CUDA indexing assert below.
            raise RuntimeError(
                "Multimodal inputs were provided but this Inkling checkpoint has "
                "no audio/vision towers (audio_config/vision_config "
                "decoder_dmodel unset) or runs with --language-model-only."
            )
        encoders = {}
        if self.visual is not None:
            encoders[Modality.IMAGE] = EncoderSpec(self.get_image_feature)
        if self.audio is not None:
            encoders[Modality.AUDIO] = EncoderSpec(self.get_audio_feature)
        input_embeds, _ = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.model.mm_text_embedding(),
            ctx=multimodal_context,
            encoders=encoders,
            multimodal_model=self,
        )
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ):
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self._embed_multimodal(ctx, input_ids, multimodal_context)
        hidden_states, aux_hidden_states = self.model(
            input_ids, positions, ctx, out_cache_loc, input_embeds=input_embeds
        )
        return self._compute_logits(
            input_ids, hidden_states, ctx, aux_hidden_states=aux_hidden_states
        )

    def _compute_logits(
        self, input_ids, hidden_states, ctx: ForwardContext, aux_hidden_states=None
    ):
        """muP logits epilogue (shared with the NextN drafter).

        Divides hidden by ``logits_mup_width_multiplier`` before the shared
        lm_head, then masks the padded-vocab tail so the sampler can never
        emit an undecodable id.
        """
        mup = self.text_config.logits_mup_width_multiplier
        if mup:
            hidden_states = hidden_states / mup
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        output = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            logits_metadata,
            aux_hidden_states=aux_hidden_states,
        )
        unpadded = self.text_config.unpadded_vocab_size
        if (
            output.next_token_logits is not None
            and output.next_token_logits.shape[-1] > unpadded
        ):
            output.next_token_logits[..., unpadded:] = float("-inf")
        return output

    def _replicate_kv_heads(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        """Replicate a checkpoint KV tensor's head dim to the served count.

        The engine serves a uniform KV pool of ``num_key_value_heads``
        (= max over layer kinds) heads on every layer, sized up to
        ``attn_tp_size`` when TP exceeds the head count. ``repeat_interleave``
        keeps replication block-contiguous, so the downstream per-rank shard
        narrowing (qkvr / sconv weight loaders) picks the KV head
        ``(rank * ckpt_heads) // tp_size`` — the reference's mapping.
        """
        cfg = self.text_config
        layer_id = int(name.split(".")[2])
        is_local = layer_id in cfg.local_layer_ids
        ckpt_heads = (
            cfg.swa_num_key_value_heads if is_local else cfg.ckpt_num_key_value_heads
        )
        served = inkling_kv_heads_for_layer(cfg, layer_id, _HETERO_KV)
        target_heads = max(served, self.mapping.attn.tp_size)
        if target_heads == ckpt_heads:
            return loaded_weight
        assert (
            target_heads % ckpt_heads == 0
        ), f"cannot replicate {ckpt_heads} KV heads to {target_heads} ({name})"
        head_dim = cfg.head_dim
        assert loaded_weight.shape[0] == ckpt_heads * head_dim, (
            f"{name}: expected leading dim {ckpt_heads * head_dim}, "
            f"got {tuple(loaded_weight.shape)}"
        )
        w = loaded_weight.reshape(ckpt_heads, head_dim, *loaded_weight.shape[1:])
        w = torch.repeat_interleave(w, target_heads // ckpt_heads, dim=0)
        return w.reshape(target_heads * head_dim, *loaded_weight.shape[1:])

    def _load_expert_param(
        self,
        params_dict: dict[str, nn.Parameter],
        modules_dict: dict[str, nn.Module],
        checkpoint_name: str,
        param_name: str,
        loaded_weight: torch.Tensor,
        loaded: set[str],
        dropped: list[str],
        interleaved: bool,
        weight_name: str,
        kind: str = "weight",
    ) -> bool:
        """Load one routed-expert param from a stacked ``[E, ...]`` tensor."""
        is_w13 = weight_name == "w13_weight"
        if param_name not in params_dict:
            dropped.append(checkpoint_name)
            return True

        param = params_dict[param_name]
        module_name = param_name.rsplit(".", 1)[0]
        experts_mod = modules_dict.get(module_name)
        num_experts = self.text_config.n_routed_experts
        num_local = getattr(experts_mod, "num_local_experts", num_experts)
        ep_rank = getattr(experts_mod, "ep_rank", 0)
        loader = getattr(param, "weight_loader", None)
        shard_ids = ("w1", "w3") if is_w13 else ("w2",)
        if kind == "mxfp4_scale" and loaded_weight.dim() == 2:
            rows = (
                2 * self.text_config.intermediate_size
                if is_w13
                else self.text_config.hidden_size
            )
            loaded_weight = loaded_weight.reshape(
                num_experts, rows, loaded_weight.shape[1]
            )

        for expert_id in range(num_experts):
            local_id = expert_id - ep_rank * num_local
            if not 0 <= local_id < num_local:
                continue
            if kind == "nvfp4_input_amax":
                # One scalar per layer/tensor; convert amax to the input_scale the NVFP4 kernels consume.
                scale = (loaded_weight.detach().float() / _NVFP4_AMAX_TO_SCALE).reshape(
                    ()
                )
                for sid in shard_ids:
                    loader(param, scale, shard_id=sid, local_expert_id=local_id)
                continue
            expert_weight = loaded_weight[expert_id]
            if kind == "nvfp4_scale2":
                for sid in shard_ids:
                    loader(param, expert_weight, shard_id=sid, local_expert_id=local_id)
                continue
            # Block scales share the w13 row layout (per-16-group scales follow weight rows 1:1).
            if is_w13 and interleaved:
                expert_weight = _deinterleave_w13(expert_weight)
            if loader is None:  # torch-native fallback experts: plain params
                param.data[local_id].copy_(expert_weight)
                continue
            if is_w13:
                half = expert_weight.shape[0] // 2
                loader(
                    param,
                    expert_weight.narrow(0, 0, half),
                    shard_id="w1",
                    local_expert_id=local_id,
                )
                loader(
                    param,
                    expert_weight.narrow(0, half, half),
                    shard_id="w3",
                    local_expert_id=local_id,
                )
            else:
                loader(param, expert_weight, shard_id="w2", local_expert_id=local_id)
        loaded.add(param_name)
        return True

    def _is_quark_mxfp4(self) -> bool:
        quant = self.quant_config
        return (
            quant is not None
            and quant.get_name() == "mxfp4"
            and getattr(quant, "is_checkpoint_mxfp4_serialized", False)
        )

    def _load_dense_stacked_experts(
        self,
        params_dict: dict[str, nn.Parameter],
        modules_dict: dict[str, nn.Module],
        name: str,
        loaded_weight: torch.Tensor,
        loaded: set[str],
        dropped: list[str],
        interleaved: bool,
    ) -> bool:
        module_name, leaf = name.split(".experts.", 1)
        module_name = module_name + ".experts"
        weight_name, _, aux = leaf.partition(".")
        if weight_name not in ("w13_weight", "w2_weight"):
            return False
        if aux == "original_shape":
            return True
        if aux:
            dropped.append(name)
            return True
        return self._load_expert_param(
            params_dict,
            modules_dict,
            name,
            f"{module_name}.{weight_name}",
            loaded_weight,
            loaded,
            dropped,
            interleaved,
            weight_name,
        )

    def _load_mxfp4_stacked_experts(
        self,
        params_dict: dict[str, nn.Parameter],
        modules_dict: dict[str, nn.Module],
        name: str,
        loaded_weight: torch.Tensor,
        loaded: set[str],
        dropped: list[str],
        interleaved: bool,
    ) -> bool:
        module_name, weight_name = name.split(".experts.", 1)
        module_name = module_name + ".experts"
        if "." in weight_name:
            dropped.append(name)
            return True
        kind = "weight"
        param_name = f"{module_name}.{weight_name}"
        if weight_name.endswith("_scale"):
            weight_name = weight_name.removesuffix("_scale")
            param_name = f"{module_name}.{weight_name}_scale"
            kind = "mxfp4_scale"
        if weight_name not in ("w13_weight", "w2_weight"):
            return False
        return self._load_expert_param(
            params_dict,
            modules_dict,
            name,
            param_name,
            loaded_weight,
            loaded,
            dropped,
            interleaved,
            weight_name,
            kind,
        )

    def _load_nvfp4_stacked_experts(
        self,
        params_dict: dict[str, nn.Parameter],
        modules_dict: dict[str, nn.Module],
        name: str,
        loaded_weight: torch.Tensor,
        loaded: set[str],
        dropped: list[str],
        interleaved: bool,
    ) -> bool:
        module_name, leaf = name.split(".experts.", 1)
        module_name = module_name + ".experts"
        weight_name, _, aux = leaf.partition(".")
        if weight_name not in ("w13_weight", "w2_weight"):
            return False
        if aux == "original_shape":
            return True
        if aux:
            aux_map = {
                "scale": (f"{module_name}.{weight_name}_scale", "weight"),
                "scale2": (f"{module_name}.{weight_name}_scale_2", "nvfp4_scale2"),
                "input_amax": (
                    f"{module_name}.{'w13' if weight_name == 'w13_weight' else 'w2'}_input_scale",
                    "nvfp4_input_amax",
                ),
            }
            if aux not in aux_map:
                dropped.append(name)
                return True
            param_name, kind = aux_map[aux]
        else:
            param_name = f"{module_name}.{weight_name}"
            kind = "weight"
        return self._load_expert_param(
            params_dict,
            modules_dict,
            name,
            param_name,
            loaded_weight,
            loaded,
            dropped,
            interleaved,
            weight_name,
            kind,
        )

    def _load_stacked_experts(
        self,
        params_dict: dict[str, nn.Parameter],
        modules_dict: dict[str, nn.Module],
        name: str,
        loaded_weight: torch.Tensor,
        loaded: set[str],
        dropped: list[str],
        interleaved: bool,
    ) -> bool:
        if ".mlp.experts." not in name:
            return False
        if self._is_quark_mxfp4():
            return self._load_mxfp4_stacked_experts(
                params_dict,
                modules_dict,
                name,
                loaded_weight,
                loaded,
                dropped,
                interleaved,
            )
        if self.quant_config is not None and self.quant_config.get_name() == "nvfp4":
            return self._load_nvfp4_stacked_experts(
                params_dict,
                modules_dict,
                name,
                loaded_weight,
                loaded,
                dropped,
                interleaved,
            )
        return self._load_dense_stacked_experts(
            params_dict,
            modules_dict,
            name,
            loaded_weight,
            loaded,
            dropped,
            interleaved,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load real Inkling checkpoint weights (dummy loading bypasses this).

        Checkpoint namespace (``model.llm/audio/visual/mtp``) maps onto this
        module tree (``model``/``audio``/``visual``/``lm_head``); the
        non-trivial moves are the qkvr fusion, full-layer KV replication to
        the served head count (identity under the hetero-KV default), gate/up
        de-interleaving of every w13 tensor, per-expert fan-out of the stacked
        expert tensors, and the NVFP4 aux-tensor renames. The MTP chain
        (``model.mtp.*``) belongs to the NextN draft worker's loader (see
        inkling_nextn.py) and is skipped here.
        """
        cfg = self.text_config
        interleaved = cfg.inference_moe_w13_interleaved
        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        loaded: set[str] = set()
        dropped: list[str] = []
        embed_tokens_weight: torch.Tensor | None = None
        load_param = _make_param_loader(params_dict, loaded, dropped)

        for name, w in weights:
            name = name.removeprefix("model.")
            if name.startswith("mtp."):
                continue  # MTP chain lands with speculative decoding
            if name.startswith("audio."):
                if self.audio is not None:
                    load_param(name, w)
                continue
            if name.startswith("visual."):
                if self.visual is not None:
                    load_param(name.replace("visual.", "visual.vision_encoder.", 1), w)
                continue
            if not name.startswith("llm."):
                dropped.append(name)
                continue
            name = "model." + name.removeprefix("llm.")

            if name == "model.embed.weight":
                embed_tokens_weight = w
                load_param("model.embed_tokens.weight", w)
                continue
            if name == "model.unembed.weight":
                load_param("lm_head.weight", w)
                continue

            if name.endswith(_KV_REPLICATED_SUFFIXES):
                w = self._replicate_kv_heads(name, w)

            if _load_block_param(load_param, name, w, interleaved):
                continue

            if ".mlp.shared_experts.shared_w13_weight" in name:
                if name.endswith("shared_w13_weight") and interleaved:
                    w = _deinterleave_w13(w)
                load_param(name.replace("shared_w13_weight", "w13_weight"), w)
                continue
            if ".mlp.shared_experts.shared_w2_weight" in name:
                load_param(name.replace("shared_w2_weight", "w2_weight"), w)
                continue

            if ".mlp.experts." in name and self._load_stacked_experts(
                params_dict,
                modules_dict,
                name,
                w,
                loaded,
                dropped,
                interleaved,
            ):
                continue

            load_param(name, w)

        if (
            "lm_head.weight" not in loaded
            and "lm_head.weight" in params_dict
            and embed_tokens_weight is not None
        ):
            load_param("lm_head.weight", embed_tokens_weight)

        if dropped:
            logger.warning(
                "Inkling load_weights dropped %d checkpoint tensors (first: %s)",
                len(dropped),
                dropped[:8],
            )
        if not loaded:
            raise RuntimeError("Inkling load_weights consumed no checkpoint tensors")
        return loaded


EntryClass = [InklingForConditionalGeneration]
