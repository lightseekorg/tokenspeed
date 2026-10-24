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

"""LongCat model configuration definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs


def _normalize_longcat_rope(
    rope: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if rope is None:
        return None
    normalized = dict(rope)
    rope_type = normalized.get("rope_type", normalized.get("type"))
    if rope_type == "yarn" and (
        "mscale" in normalized or "mscale_all_dim" in normalized
    ):
        # The PTQ preparation code rewrites ``deepseek_yarn`` to ``yarn`` so
        # Transformers can instantiate its calibration model. Quantization
        # does not change RoPE semantics; restore the runtime's exact variant.
        normalized["rope_type"] = "deepseek_yarn"
        if "type" in normalized:
            normalized["type"] = "deepseek_yarn"
    return normalized


class LongcatConfig(PretrainedConfig):
    """Configuration for LongCat-2.0 checkpoints.

    The official LongCat-2.0 ``config.json`` does not declare ``model_type``
    and contains LongCat-specific MLA, LSA, MoE, MTP, and over-embedding
    fields. Keeping a local, permissive config makes those fields available to
    the runtime without executing checkpoint-provided Python code.

    Args:
        vocab_size: Number of vocabulary entries.
        hidden_size: Transformer hidden dimension.
        num_layers: Number of logical LongCat layers.
        num_attention_heads: Number of query attention heads.
        ffn_hidden_size: Dense feed-forward intermediate dimension.
        expert_ffn_hidden_size: Per-expert feed-forward intermediate dimension.
        n_routed_experts: Number of routed MoE experts.
        moe_topk: Number of selected experts per token.
        kwargs: Additional public checkpoint metadata preserved verbatim.
    """

    model_type = "longcat"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 8192,
        num_layers: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        ffn_hidden_size: int = 12288,
        expert_ffn_hidden_size: int = 2048,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 1_000_000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        # MLA
        attention_method: str = "MLA",
        use_mla: int | bool = 1,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_scale_q_lora: bool = True,
        mla_scale_kv_lora: bool = True,
        # LSA
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_init_tokens: int = 16,
        index_local_tokens: int = 1024,
        index_k_norm_type: str = "rms",
        cli_factor: int = 2,
        dsa_mtp_cli: bool = True,
        # MoE
        n_routed_experts: int = 768,
        moe_topk: int = 12,
        routed_scaling_factor: float = 9.0,
        norm_topk_prob: bool = False,
        zero_expert_num: int = 128,
        zero_expert_type: str = "identity",
        moe_impl: str = "mix",
        moe_switch_token_num: int = 1024,
        # Over embedding
        oe_neighbor_num: int = 5,
        oe_split_num: int = 4,
        oe_vocab_size_ratio: float = 100.567,
        # MTP
        mtp_num_layers: int = 3,
        mtp_disable_over_tokenizer: bool = True,
        mtp_replicate_modules: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        logical_layers = num_layers
        if logical_layers is None:
            logical_layers = num_hidden_layers
        if logical_layers is None:
            logical_layers = 38

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = logical_layers
        self.num_hidden_layers = logical_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.ffn_hidden_size = ffn_hidden_size
        self.intermediate_size = ffn_hidden_size
        self.expert_ffn_hidden_size = expert_ffn_hidden_size
        self.moe_intermediate_size = expert_ffn_hidden_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        normalized_rope = _normalize_longcat_rope(
            rope_scaling if rope_scaling is not None else rope_parameters
        )
        self.rope_scaling = normalized_rope
        self.rope_parameters = _normalize_longcat_rope(
            rope_parameters if rope_parameters is not None else normalized_rope
        )

        self.attention_method = attention_method
        self.use_mla = use_mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.v_head_dim = v_head_dim
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_init_tokens = index_init_tokens
        self.index_local_tokens = index_local_tokens
        self.index_k_norm_type = index_k_norm_type
        self.cli_factor = cli_factor
        self.dsa_mtp_cli = dsa_mtp_cli

        self.n_routed_experts = n_routed_experts
        self.num_local_experts = n_routed_experts
        self.moe_topk = moe_topk
        self.num_experts_per_tok = moe_topk
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.moe_impl = moe_impl
        self.moe_switch_token_num = moe_switch_token_num

        self.oe_neighbor_num = oe_neighbor_num
        self.oe_split_num = oe_split_num
        self.oe_vocab_size_ratio = oe_vocab_size_ratio

        self.mtp_num_layers = mtp_num_layers
        self.mtp_disable_over_tokenizer = mtp_disable_over_tokenizer
        self.mtp_replicate_modules = mtp_replicate_modules

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


_LONGCAT_ARCHITECTURES = frozenset({"LongcatCausalLM", "LongcatFlashForCausalLM"})
_LONGCAT_LSA_EXPECTED_GEOMETRY = {
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_init_tokens": 16,
    "index_local_tokens": 1024,
    "cli_factor": 2,
    "index_k_norm_type": "rms",
}


def is_longcat_lsa_config(
    hf_config: PretrainedConfig,
    hf_text_config: PretrainedConfig,
) -> bool:
    """Return whether a LongCat checkpoint declares the complete LSA geometry."""

    architectures = (getattr(hf_config, "architectures", None) or []) + (
        getattr(hf_text_config, "architectures", None) or []
    )
    if not any(arch in _LONGCAT_ARCHITECTURES for arch in architectures):
        return False
    attention_config = (
        hf_text_config if hasattr(hf_text_config, "kv_lora_rank") else hf_config
    )
    return all(
        hasattr(attention_config, field) for field in _LONGCAT_LSA_EXPECTED_GEOMETRY
    )


def configure_longcat_lsa_attention(model_config, server_args: ServerArgs) -> None:
    """Configure the supported LongCat-2.0 LSA/SI/CLI attention geometry."""

    attention_config = (
        model_config.hf_text_config
        if hasattr(model_config.hf_text_config, "kv_lora_rank")
        else model_config.hf_config
    )
    for field, expected in _LONGCAT_LSA_EXPECTED_GEOMETRY.items():
        actual = getattr(attention_config, field)
        if actual != expected:
            raise ValueError(
                f"Unsupported LongCat LSA {field}: expected {expected}, got {actual}"
            )

    from tokenspeed.runtime.configs.model_config import configure_dsa_attention

    configure_dsa_attention(model_config, server_args)

    # The projection weight format does not select the index-cache format.
    model_config.index_k_format = "fp8_scaled"


__all__ = [
    "LongcatConfig",
    "configure_longcat_lsa_attention",
    "is_longcat_lsa_config",
    "get_longcat_model_profile",
]


def get_longcat_model_profile(hf_config):
    """Return Pro/OE runtime facts; ordinary LongCat keeps its existing path."""
    lsa = is_longcat_lsa_config(hf_config, hf_config)
    oe = bool(
        getattr(hf_config, "use_over_embedding", False)
        or getattr(hf_config, "ngram_vocab_size_ratio", None) is not None
        or getattr(hf_config, "oe_vocab_size_ratio", None) is not None
    )
    if not lsa and not oe:
        return None
    from tokenspeed.runtime.configs.model_config import configure_mla_attention
    from tokenspeed.runtime.configs.model_profile import ModelProfile

    return ModelProfile(
        configure_attention=(
            configure_longcat_lsa_attention if lsa else configure_mla_attention
        ),
        cache_family="dsa" if lsa else "mla",
        linear_attention=None,
        default_attention_backend="dsa" if lsa else None,
        default_prefix_granularity=None,
        request_token_history=oe,
        is_multimodal=False,
        tokenizer_kwargs={},
        attention_instances_per_layer=2,
        numerics_envelopes=frozenset({"auto"}),
    )


def register() -> None:
    """Register LongCat's config and model classes through the plugin API."""
    from tokenspeed.runtime.models.longcat_flash import EntryClass
    from tokenspeed.runtime.plugins import registry
    from tokenspeed.runtime.utils import hf_transformers_utils as hf

    if hf._resolve_registered_config({"model_type": LongcatConfig.model_type}) is None:
        registry.register_config(LongcatConfig, model_type=LongcatConfig.model_type)
    alias = {"model_type": None, "architectures": ["LongcatCausalLM"]}
    if hf._resolve_registered_config(alias) is None:
        registry.register_config(
            LongcatConfig, model_type=None, architectures=("LongcatCausalLM",)
        )
    for cls in EntryClass:
        if registry.registered_model(cls.__name__) is None:
            registry.register_model(cls, override=True)
