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

"""Inkling model configuration definitions.

Adapted from the Inkling reference implementation. The runtime-facing surface is
intentionally *scheduler-blind*: the C++ scheduler must see Inkling as a plain
dense GQA model, so this config MUST NOT expose a non-empty
``linear_layer_ids`` (the probe that registers a linear-attention component
and with it mamba scheduling). The sconv rolling state is managed entirely
engine-side, keyed on request pool indices
(see ``tokenspeed.runtime.layers.attention.backends.inkling``).

KV-head note: the checkpoint uses 8 KV heads on full-attention layers and 16
on sliding-window layers. The KV pools are uniform-head, so the runtime
serves all layers with the maximum count and replicates full-layer K/V (and
their sconv channels) at load time; ``ckpt_num_key_value_heads`` preserves
the raw checkpoint value for the weight loader.
"""

from __future__ import annotations

import copy
import enum
from typing import Any, Literal

from tokenspeed.runtime.configs.base_config import BaseConfig, TextConfigBase

# TMLv0 marker/target IDs retained for compatibility.
INKLING_CONTENT_IMAGE_TOKEN_ID = 200005
INKLING_MODEL_END_SAMPLING_TOKEN_ID = 200006
INKLING_AUDIO_TOKEN_ID = 200023

# Checkpoint-provided soft-placeholder IDs. The target path overwrites these
# positions with encoder embeddings; the MTP draft path restores the same
# in-vocab IDs before its embedding lookup.
INKLING_IMAGE_PLACEHOLDER_TOKEN_ID = 200054
INKLING_AUDIO_PLACEHOLDER_TOKEN_ID = 200053


class InklingConvStream(enum.IntEnum):
    """The four sconv sites per decoder block, in cache channel order."""

    K = 0
    V = 1
    ATTN = 2
    MLP = 3


def inkling_conv_stream_layout(
    config: "InklingModelConfig", attn_tp_size: int
) -> dict[InklingConvStream, tuple[int, int]]:
    """Channel layout of the per-layer sconv state cache.

    The engine-side conv pool stores all four sconv streams of a decoder
    block in one channel-concatenated buffer of shape
    ``[num_slots, sconv_kernel_size - 1, total_dim]``. This helper returns,
    for each stream, its ``(channel_offset, channel_dim)`` within that
    buffer for the given attention TP degree.

    K/V regions are sized by the uniform (post-replication) KV head count so
    the layout is identical for SWA and full-attention layers.

    Args:
        config: The Inkling text config.
        attn_tp_size: Attention tensor-parallel degree.

    Returns:
        Mapping from stream to ``(offset, dim)``; offsets are contiguous and
        ordered K, V, ATTN, MLP.
    """
    kv_dim = max(1, config.num_key_value_heads // attn_tp_size) * config.head_dim
    hidden = config.hidden_size
    dims = {
        InklingConvStream.K: kv_dim,
        InklingConvStream.V: kv_dim,
        InklingConvStream.ATTN: hidden,
        InklingConvStream.MLP: hidden,
    }
    layout: dict[InklingConvStream, tuple[int, int]] = {}
    offset = 0
    for stream in InklingConvStream:
        layout[stream] = (offset, dims[stream])
        offset += dims[stream]
    return layout


def inkling_mtp_text_config(
    config: "InklingModelConfig", num_steps: int | None = None
) -> "InklingModelConfig":
    """Text config specialized for the Inkling MTP draft worker's depth blocks.

    The base decoder and the MTP head have independent local/full attention
    patterns: the checkpoint records the head's depth-local ids in top-level
    ``mtp_config.local_layer_ids`` (copied here as ``mtp_local_layer_ids``).
    The returned config drives draft layer construction, attention metadata,
    and cache-group layout, so its ``local_layer_ids`` are the DEPTH-local
    ids and its ``num_hidden_layers`` is the depth count.

    With ``num_steps`` set, depths beyond it are pruned: an MTP chain only
    ever runs depths ``0..steps-1`` (``spec_step_idx`` indexes the chain
    front), so trailing depths would occupy KV/conv pool and weight memory
    without ever executing.

    Idempotent: a config that already went through this transform is
    returned unchanged (the draft model applies it defensively on top of
    ModelConfig's swap).
    """
    if getattr(config, "is_mtp_text_config", False):
        return config
    cfg = copy.deepcopy(config)
    num_depths = config.num_nextn_predict_layers
    if num_steps is not None and 0 < num_steps < num_depths:
        num_depths = num_steps
    cfg.num_hidden_layers = num_depths
    cfg.num_nextn_predict_layers = num_depths
    cfg.local_layer_ids = [
        i for i in getattr(config, "mtp_local_layer_ids", []) if i < num_depths
    ]
    cfg.mtp_local_layer_ids = list(cfg.local_layer_ids)
    cfg.dense_mlp_idx = num_depths  # dense MLP on every depth
    cfg.is_mtp_text_config = True
    return cfg


def inkling_conv_total_dim(config: "InklingModelConfig", attn_tp_size: int) -> int:
    """Total channel width of the per-layer sconv state buffer at this TP."""
    layout = inkling_conv_stream_layout(config, attn_tp_size)
    last_offset, last_dim = layout[InklingConvStream.MLP]
    return last_offset + last_dim


class InklingModelConfig(TextConfigBase):
    """Text/decoder configuration (checkpoint ``model_type: inkling_model``)."""

    model_type = "inkling_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    model_max_length: int = 1048576
    vocab_size: int = 201024
    hidden_size: int = 1536
    intermediate_size: int = 768
    dense_intermediate_size: int | None = None
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int | None = None
    v_head_dim: int | None = None
    d_rel: int = 16
    rel_extent: int = 1024
    local_layer_ids: list[int] | None = None
    sliding_window_size: int = 512
    swa_num_attention_heads: int | None = None
    swa_num_key_value_heads: int | None = None
    swa_head_dim: int | None = None
    swa_v_head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    q_bias: bool = False
    o_bias: bool = False
    use_embed_norm: bool = False
    use_sconv: bool = False
    sconv_kernel_size: int = 4
    dense_mlp_idx: int = 0
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    num_experts_per_tok: int = 1
    route_scale: float = 1.0
    use_gate_bias: bool = False
    use_global_scale: bool = False
    norm_after_topk: bool = True
    gate_activation: Literal["sigmoid", "softmax"] = "sigmoid"
    shared_expert_sink: bool = False
    shared_experts_size: int = 1
    inference_moe_w13_interleaved: bool = True
    log_scaling_n_floor: int | None = None
    log_scaling_alpha: float = 0.1
    unpadded_vocab_size: int | None = None
    padded_vocab_size: int | None = None
    logits_mup_width_multiplier: float | None = None
    final_logit_softcapping: float | None = None
    num_nextn_predict_layers: int = 0
    chain_hidden_post_norm: bool = True
    mtp_local_layer_ids: list[int] | None = None
    tie_word_embeddings: bool = False
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim
        if self.swa_num_attention_heads is None:
            self.swa_num_attention_heads = self.num_attention_heads
        if self.swa_num_key_value_heads is None:
            self.swa_num_key_value_heads = self.num_key_value_heads
        if self.swa_head_dim is None:
            self.swa_head_dim = self.head_dim
        if self.swa_v_head_dim is None:
            self.swa_v_head_dim = self.swa_head_dim
        if self.dense_intermediate_size is None:
            self.dense_intermediate_size = self.intermediate_size
        if self.local_layer_ids is None:
            self.local_layer_ids = []
        if self.mtp_local_layer_ids is None:
            self.mtp_local_layer_ids = []

        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
            if (
                self.unpadded_vocab_size is not None
                and self.unpadded_vocab_size < self.padded_vocab_size
            ):
                self.vocab_size = self.unpadded_vocab_size

        # TMLv0's fixed stop token is part of the production vocabulary. Tiny
        # synthetic fixtures intentionally use a much smaller vocabulary, so
        # leave their EOS unset instead of installing an out-of-range id.
        if (
            self.eos_token_id is None
            and self.vocab_size > INKLING_MODEL_END_SAMPLING_TOKEN_ID
        ):
            self.eos_token_id = INKLING_MODEL_END_SAMPLING_TOKEN_ID

        # Uniform KV pool: serve all layers at max KV-head count; KV weights
        # replicated at load. BaseConfig stores the resolved count, while
        # the raw checkpoint count is kept separately for the weight loader.
        raw_num_key_value_heads = self.num_key_value_heads
        self.num_key_value_heads = max(
            self.num_key_value_heads, self.swa_num_key_value_heads
        )

        super().__post_init__(**kwargs)

        self.ckpt_num_key_value_heads = raw_num_key_value_heads
        self.num_experts = self.n_routed_experts
        self.num_shared_experts = self.n_shared_experts
        self.unpadded_vocab_size = self.vocab_size

        if self.num_attention_heads != self.swa_num_attention_heads:
            raise ValueError(
                "Inkling requires matching Q-head counts on full and SWA layers, "
                f"got {self.num_attention_heads} vs {self.swa_num_attention_heads}."
            )
        if self.head_dim != self.swa_head_dim:
            raise ValueError(
                "Inkling requires matching head_dim on full and SWA layers for the "
                f"uniform KV pool, got {self.head_dim} vs {self.swa_head_dim}."
            )
        if not set(self.local_layer_ids) <= set(range(self.num_hidden_layers)):
            raise ValueError("local_layer_ids contains out-of-range layer ids.")
        if len(set(self.local_layer_ids)) != len(self.local_layer_ids):
            raise ValueError("local_layer_ids contains duplicates.")
        if self.num_nextn_predict_layers < 0:
            raise ValueError("num_nextn_predict_layers must be >= 0.")
        if not set(self.mtp_local_layer_ids) <= set(
            range(self.num_nextn_predict_layers)
        ):
            raise ValueError("mtp_local_layer_ids contains out-of-range layer ids.")
        if len(set(self.mtp_local_layer_ids)) != len(self.mtp_local_layer_ids):
            raise ValueError("mtp_local_layer_ids contains duplicates.")

    @property
    def swa_attention_layer_ids(self) -> list[int]:
        """Layer ids using sliding-window attention (unit-test convenience; no runtime consumer)."""
        return list(self.local_layer_ids)

    @property
    def global_attention_layer_ids(self) -> list[int]:
        """Layer ids using full (non-windowed) attention (unit-test convenience; no runtime consumer).

        Named to avoid ``full_attention_layer_ids``, which hybrid-GDN engine
        paths probe on text configs.
        """
        local = set(self.local_layer_ids)
        return [i for i in range(self.num_hidden_layers) if i not in local]

    @property
    def cache_layer_types(self) -> list[str]:
        """Per-layer cache-group labels derived from ``local_layer_ids``.

        Deliberately NOT named ``layer_types`` because the sub-group labels
        below are runtime KV-cache groups rather than model architecture
        layer types; ``MHAConfig.generate`` prefers this attribute.

        Sliding layers are split round-robin (by rank among sliding layers)
        into equal-count sub-groups ``sliding_attention_<k>``, sized so no
        sub-group exceeds the full group's layer count: under the hybrid
        slab layout every slab is then bound by one layer of every group,
        so an owned block has no dead slab rows regardless of the owning
        group (Inkling: 55 sliding + 11 full -> 5 sub-groups of 11 -> 11
        six-way-bound slabs). All sub-groups share the one window, so
        eviction semantics per layer are unchanged vs a single sliding
        group. Consumed by the cache-group path (cache-group
        publication and the hybrid slab layout); inert on a single-table-built
        scheduler ext, so the scheduler-blind contract above still holds
        there.
        """
        local = set(self.local_layer_ids)
        full_count = self.num_hidden_layers - len(local)
        if not local:
            return ["full_attention"] * self.num_hidden_layers
        if not full_count:
            return ["sliding_attention"] * self.num_hidden_layers
        subgroups = -(-len(local) // full_count)  # ceil div
        labels: list[str] = []
        rank = 0
        for i in range(self.num_hidden_layers):
            if i in local:
                labels.append(f"sliding_attention_{rank % subgroups}")
                rank += 1
            else:
                labels.append("full_attention")
        return labels

    @property
    def sliding_window(self) -> int:
        """HF-conventional alias of ``sliding_window_size`` (the window in
        tokens including the current one), read alongside ``layer_types``."""
        return self.sliding_window_size


class InklingAudioConfig(BaseConfig):
    """Audio tower configuration (parsed but unused in text-only serving)."""

    model_type = "inkling_audio_model"

    decoder_dmodel: int | None = None
    n_mel_bins: int = 80
    mel_vocab_size: int = 16
    dmel_min_value: float = -7.0
    dmel_max_value: float = 2.0
    audio_rms_norm_floor: float = 0.01
    use_audio_norm: bool = False
    audio_mode: Literal["dmel", "flow"] = "dmel"

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)


class InklingVisionConfig(BaseConfig):
    """Vision tower configuration (parsed but unused in text-only serving)."""

    model_type = "inkling_vision_model"

    vision_encoder_type: Literal["linear", "hmlp"] = "hmlp"
    decoder_dmodel: int | None = None
    patch_size: int = 16
    temporal_patch_size: int = 1
    n_channels: int = 3
    n_layers: int = 1
    use_vision_norm: bool = False

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)


class InklingMMConfig(BaseConfig):
    """Container configuration (checkpoint ``model_type: inkling_mm_model``).

    Wraps the text/audio/vision sub-configs so the shipped checkpoint
    ``config.json`` loads unmodified. Text-model attributes are forwarded so
    generic engine code (``get_hf_text_config`` and friends) works.
    """

    model_type = "inkling_mm_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "text_config": InklingModelConfig,
        "audio_config": InklingAudioConfig,
        "vision_config": InklingVisionConfig,
    }

    text_config: dict[str, Any] | InklingModelConfig | None = None
    audio_config: dict[str, Any] | InklingAudioConfig | None = None
    vision_config: dict[str, Any] | InklingVisionConfig | None = None
    mtp_config: dict[str, Any] | None = None
    image_placeholder_token_id: int | None = INKLING_IMAGE_PLACEHOLDER_TOKEN_ID
    audio_placeholder_token_id: int | None = INKLING_AUDIO_PLACEHOLDER_TOKEN_ID
    tie_word_embeddings: bool = False
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        # Gateway-expanded per media item; transport input_ids are unsigned.
        if (
            self.image_placeholder_token_id is not None
            and self.image_placeholder_token_id < 0
        ):
            raise ValueError("image_placeholder_token_id must be a non-negative id")
        if (
            self.audio_placeholder_token_id is not None
            and self.audio_placeholder_token_id < 0
        ):
            raise ValueError("audio_placeholder_token_id must be a non-negative id")

        # Coerce dict/None sub-configs to their classes per the declared mapping.
        _given = {
            "text_config": self.text_config,
            "audio_config": self.audio_config,
            "vision_config": self.vision_config,
        }
        for name, cls in self.sub_configs.items():
            cfg = _given[name]
            setattr(self, name, cfg if isinstance(cfg, cls) else cls(**(cfg or {})))
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id
        self.text_config.eos_token_id = self.eos_token_id
        # ``mtp_config`` is a sibling of ``text_config`` in the checkpoint
        # config.json; copy its fields onto the text config, which is what
        # draft-worker sizing (``ModelConfig``) and the NextN model read.
        if self.mtp_config:
            num_nextn_predict_layers = int(
                self.mtp_config.get("num_nextn_predict_layers", 0)
            )
            mtp_local_layer_ids = list(self.mtp_config.get("local_layer_ids", []))
            if not set(mtp_local_layer_ids) <= set(range(num_nextn_predict_layers)):
                raise ValueError(
                    "mtp_config.local_layer_ids contains out-of-range layer ids."
                )
            if len(set(mtp_local_layer_ids)) != len(mtp_local_layer_ids):
                raise ValueError("mtp_config.local_layer_ids contains duplicates.")
            self.text_config.num_nextn_predict_layers = num_nextn_predict_layers
            self.text_config.chain_hidden_post_norm = bool(
                self.mtp_config.get("chain_hidden_post_norm", True)
            )
            self.text_config.mtp_local_layer_ids = mtp_local_layer_ids

        super().__post_init__(**kwargs)

    def get_text_config(self, *args: Any, **kwargs: Any) -> InklingModelConfig:
        return self.text_config

    @property
    def cache_layer_types(self) -> list[str]:
        return self.text_config.cache_layer_types

    @property
    def sliding_window(self) -> int:
        return self.text_config.sliding_window

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim
