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

"""Kimi-K3 model configuration.

Kimi-K3 is a multimodal model that pairs a MoonViT3d vision tower (reused from
Kimi-K2.5) with the ``KimiLinear`` text backbone: a hybrid decoder that
interleaves KDA (Kimi Delta Attention, a per-channel gated delta-rule linear
attention) layers with NoPE-MLA full-attention layers, on top of a DeepSeek-V3
style sigmoid / noaux_tc MoE (with a Kimi-specific "latent" MoE and ``situ``
activation) and block-level attention residuals (``attn_res_block_size``).

All default values below are taken verbatim from the reference checkpoint
``config.json`` (``moonshotai/kimi-k3``, snapshot
``a527b42cb673d79569f3ffe0b3a0a655df98a739``).
"""

from transformers.configuration_utils import PretrainedConfig

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

# "linear_attention" comes from cache_spec (the KV-cache label vocabulary
# the hybrid allocator keys on). "attention" is the value of
# ``HybridLayerType.full_attention`` (configs/qwen3_5_text_base_config.py);
# kept as a literal to avoid a cross-config import in the model-loading path.
_ATTENTION_LAYER = "attention"
_LINEAR_ATTENTION_LAYER = LINEAR_ATTENTION


class KimiK3VisionConfig(PretrainedConfig):
    """Vision-tower + mm-projector configuration for Kimi-K3.

    This config keeps the checkpoint's native ``vt_*`` field names. The shared
    MoonViT implementation normalizes them without adding K2.5 compatibility
    aliases. Defaults mirror the checkpoint's ``vision_config``.
    """

    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        pos_emb_interpolation_mode: str = "bilinear",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        qkv_hidden_size: int = 1536,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        text_hidden_size: int = 7168,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.qkv_hidden_size = qkv_hidden_size
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        )
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.text_hidden_size = text_hidden_size


class KimiLinearConfig(PretrainedConfig):
    """Text-backbone configuration for Kimi-K3 (``model_type = "kimi_linear"``).

    Carries three groups of fields:

    * **MLA** (``q_lora_rank`` .. ``mla_use_output_gate``) consumed by the full
      attention layers and by ``configure_mla_attention``.
    * **MoE** (``num_experts`` .. ``topk_method``) consumed by the MoE block.
    * **KDA** (``linear_attn_config``) consumed by the linear-attention layers
      and, via the mixed-layer protocol properties below, by the hybrid
      KV-cache allocator.
    * **DSA** (``dsa_config``, Kimi-K3.1 only): the lightning indexer on the
      MLA layers, exposed as ``index_topk`` / ``index_head_dim`` /
      ``index_n_heads`` for ``configure_dsa_attention``.

    The ``layers_block_type`` / ``layer_types`` / ``linear_layer_ids`` /
    ``full_attention_layer_ids``
    properties are the **interface consumed by the KV-cache / hybrid-attention
    layer** (owned by the KV-cache team). Their shapes are derived from the KDA
    config here; the final state layout/dtype is validated on the cache side.
    """

    model_type = "kimi_linear"

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 33792,
        num_hidden_layers: int = 93,
        num_attention_heads: int = 96,
        num_key_value_heads: int | None = 96,
        hidden_act: str = "situ",
        activation_situ_beta: float | None = 4.0,
        activation_situ_linear_beta: float | None = 25.0,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 1048576,
        tie_word_embeddings: bool = False,
        # MLA
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int | None = 512,
        qk_nope_head_dim: int | None = 128,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        mla_use_nope: bool = True,
        mla_use_output_gate: bool = True,
        # MoE
        num_experts: int | None = 896,
        num_experts_per_token: int | None = 16,
        num_shared_experts: int = 2,
        moe_intermediate_size: int | None = 3072,
        routed_expert_hidden_size: int | None = 3584,
        latent_moe_use_norm: bool = True,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        # Kimi extras
        attn_res_block_size: int | None = 12,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        dsa_config: dict | None = None,
        pad_token_id: int = 163839,
        bos_token_id: int = 163584,
        eos_token_id: int = 163586,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.hidden_act = hidden_act
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings

        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate

        # MoE
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.moe_renormalize = moe_renormalize
        self.moe_router_activation_func = moe_router_activation_func
        if self.moe_router_activation_func not in ("softmax", "sigmoid"):
            raise ValueError(
                "moe_router_activation_func must be 'softmax' or 'sigmoid', got "
                f"{self.moe_router_activation_func!r}"
            )
        self.topk_method = topk_method
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq

        # Kimi extras
        self.attn_res_block_size = attn_res_block_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.linear_attn_config = (
            linear_attn_config
            if linear_attn_config is not None
            else _default_linear_attn_config()
        )
        # full-attention layers are derived as the complement of kda_layers
        # (see full_attention_layer_ids), so only kda_layers is required here.
        if self.linear_attn_config.get("kda_layers") is None:
            raise ValueError("linear_attn_config must provide 'kda_layers'")

        # Absent (Kimi-K3), the DSA fields stay None and the MLA layers dense.
        self.dsa_config = dsa_config
        self.index_topk: int | None = None
        self.index_head_dim: int | None = None
        self.index_n_heads: int | None = None
        self.dsa_layer_ids: list[int] = []
        if dsa_config is not None:
            (
                self.index_topk,
                self.index_head_dim,
                self.index_n_heads,
                self.dsa_layer_ids,
            ) = _parse_dsa_config(dsa_config, self.full_attention_layer_ids)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ----- helpers -----

    def is_dsa_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` (0-based) is an MLA layer with a DSA indexer."""
        return layer_idx in self.dsa_layer_ids

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` (0-based) is a KDA linear-attention layer.

        ``linear_attn_config["kda_layers"]`` lists 1-based layer numbers, so a
        0-based index maps via ``layer_idx + 1``.
        """
        return (layer_idx + 1) in self.linear_attn_config["kda_layers"]

    # ----- mixed-layer protocol (consumed by the KV-cache / hybrid layer) -----

    @property
    def layers_block_type(self) -> list[str]:
        """Per-layer type: ``"linear_attention"`` (KDA) or ``"attention"`` (MLA)."""
        return [
            (_LINEAR_ATTENTION_LAYER if self.is_kda_layer(i) else _ATTENTION_LAYER)
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layer_types(self) -> list[str]:
        """``layers_block_type`` translated to cache-group labels."""
        return [
            FULL_ATTENTION if layer_type == _ATTENTION_LAYER else layer_type
            for layer_type in self.layers_block_type
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i
            for i, t in enumerate(self.layers_block_type)
            if t == _LINEAR_ATTENTION_LAYER
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i for i, t in enumerate(self.layers_block_type) if t == _ATTENTION_LAYER
        ]


# Unknown dsa_config keys fail the parse: they may change the indexer's semantics.
_DSA_CONFIG_KEYS = frozenset(
    {
        "dsa_layers",
        "index_topk",
        "index_head_dim",
        "index_num_heads",
        "index_rope_dim",
        "index_k_norm_type",
        "index_topk_freq",
        "index_skip_topk_offset",
    }
)
# Absent from the released checkpoint, whose indexer weights are BF16.
_OPTIONAL_DSA_CONFIG_KEYS = frozenset({"indexer_use_fp8"})


def _positive_int(dsa_config: dict, key: str) -> int:
    value = dsa_config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"dsa_config.{key} must be a positive integer, got {value!r}")
    return value


def _parse_dsa_config(
    dsa_config: dict, full_attention_layer_ids: list[int]
) -> tuple[int, int, int, list[int]]:
    """Validate Kimi-K3.1's ``dsa_config`` against the implemented indexer.

    The implemented indexer has no RoPE part, a LayerNorm key norm, BF16
    weights, and its own top-k on every DSA layer. A config asking for
    anything else fails here instead of serving different numerics.

    Args:
        dsa_config: The text config's ``dsa_config`` mapping.
        full_attention_layer_ids: 0-based MLA layer ids from
            ``linear_attn_config``.

    Returns:
        ``(index_topk, index_head_dim, index_n_heads, dsa_layer_ids)`` with
        0-based layer ids.
    """
    if not isinstance(dsa_config, dict):
        raise TypeError(f"dsa_config must be a mapping, got {type(dsa_config)}")
    keys = set(dsa_config)
    missing = _DSA_CONFIG_KEYS - keys
    unknown = keys - _DSA_CONFIG_KEYS - _OPTIONAL_DSA_CONFIG_KEYS
    if missing or unknown:
        raise ValueError(
            "dsa_config keys differ from the implemented indexer: missing "
            f"{sorted(missing)}, unknown {sorted(unknown)}"
        )
    if dsa_config["index_rope_dim"] != 0:
        raise NotImplementedError(
            "Kimi DSA indexer implements index_rope_dim=0 only, got "
            f"{dsa_config['index_rope_dim']!r}"
        )
    if dsa_config["index_k_norm_type"] != "layernorm":
        raise NotImplementedError(
            "Kimi DSA indexer implements index_k_norm_type='layernorm' only, got "
            f"{dsa_config['index_k_norm_type']!r}"
        )
    if dsa_config.get("indexer_use_fp8", False) is not False:
        raise NotImplementedError(
            "Kimi DSA indexer implements BF16 indexer weights "
            f"(indexer_use_fp8=False) only, got {dsa_config['indexer_use_fp8']!r}"
        )
    if dsa_config["index_topk_freq"] != 1:
        raise NotImplementedError(
            "Kimi DSA indexer computes its own top-k on every DSA layer "
            f"(index_topk_freq=1) only, got {dsa_config['index_topk_freq']!r}"
        )
    # The skip offset only places shared layers when index_topk_freq > 1.
    _positive_int(dsa_config, "index_skip_topk_offset")
    layers = dsa_config["dsa_layers"]
    if not isinstance(layers, (list, tuple)) or any(
        isinstance(layer, bool) or not isinstance(layer, int) for layer in layers
    ):
        raise ValueError("dsa_config.dsa_layers must list 1-based layer numbers")
    dsa_layer_ids = sorted(layer - 1 for layer in layers)
    if len(set(dsa_layer_ids)) != len(dsa_layer_ids) or dsa_layer_ids != sorted(
        full_attention_layer_ids
    ):
        raise ValueError(
            "dsa_config.dsa_layers must name every full-attention layer exactly "
            f"once; got {sorted(layers)}, full-attention layers (1-based) are "
            f"{[layer + 1 for layer in full_attention_layer_ids]}"
        )
    return (
        _positive_int(dsa_config, "index_topk"),
        _positive_int(dsa_config, "index_head_dim"),
        _positive_int(dsa_config, "index_num_heads"),
        dsa_layer_ids,
    )


def _default_linear_attn_config() -> dict:
    """KDA config matching the reference checkpoint (69 KDA / 24 MLA layers).

    ``kda_layers`` / ``full_attn_layers`` are 1-based and partition ``1..93``;
    the full-attention layers are every 4th layer plus the last layer.
    """
    full_attn_layers = list(range(4, 93, 4)) + [93]
    kda_layers = [i for i in range(1, 94) if i not in set(full_attn_layers)]
    return {
        "kda_layers": kda_layers,
        "full_attn_layers": full_attn_layers,
        "num_heads": 96,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


class KimiK3Config(PretrainedConfig):
    """Top-level Kimi-K3 config: ``KimiLinear`` text + MoonViT3d vision."""

    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | KimiLinearConfig | None = None,
        vision_config: dict | KimiK3VisionConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        image_placeholder: str = "<|kimi_image_placeholder|>",
        pad_token_id: int = 163839,
        **kwargs,
    ):
        if text_config is None:
            text_config = KimiLinearConfig()
        elif isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)
        self.text_config = text_config

        if vision_config is None:
            vision_config = KimiK3VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = KimiK3VisionConfig(**vision_config)
        self.vision_config = vision_config

        # The vision projector must emit the text model's hidden size.
        self.vision_config.text_hidden_size = self.text_config.hidden_size

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.image_placeholder = image_placeholder

        # Propagate quantization config from the text model.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size


class KimiK31Config(KimiK3Config):
    """Kimi-K3.1: the Kimi-K3 config whose text config carries ``dsa_config``."""

    model_type = "kimi_k31"

    # Explicit, so transformers keeps KimiK3Config's sub-config construction.
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
