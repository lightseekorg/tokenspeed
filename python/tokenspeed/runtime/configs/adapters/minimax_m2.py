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

"""MiniMax-M2: HF config → EngineModelSpec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.configs.engine_spec import (
    ENGINE_SPEC_SCHEMA_VERSION,
    EngineModelSpec,
    GQAAttentionSpec,
    MinimaxM2ModelSpec,
    MoEMLPSpec,
    RMSNormSpec,
    RopePositionSpec,
)
from tokenspeed.runtime.utils.hf_transformers_utils import get_context_length


@dataclass
class MiniMaxM2RuntimeView:
    """Temporary bridge for ``models/minimax_m2.py`` until it reads ``body``."""

    model_type: str
    architectures: list[str]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    rotary_dim: int
    attention_bias: bool
    num_local_experts: int
    num_experts_per_tok: int
    scoring_func: str
    use_routing_bias: bool
    norm_topk_prob: bool
    use_qk_norm: bool
    qk_norm_type: str | None
    num_mtp_modules: int
    mtp_transformer_layers: int

    @classmethod
    def from_engine_spec(cls, spec: EngineModelSpec) -> MiniMaxM2RuntimeView:
        body = spec.body
        if body.type != "minimax_m2":
            raise ValueError(f"expected minimax_m2 body, got {body.type!r}")
        attn = body.attention
        mlp = body.mlp
        return cls(
            model_type=spec.model_type,
            architectures=list(spec.architectures),
            vocab_size=body.vocab_size,
            hidden_size=body.hidden_size,
            intermediate_size=mlp.intermediate_size,
            num_hidden_layers=body.num_layers,
            num_attention_heads=attn.num_query_heads,
            num_key_value_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            hidden_act=body.hidden_act,
            max_position_embeddings=body.position.max_position_embeddings,
            rms_norm_eps=body.norm.eps,
            tie_word_embeddings=body.tie_word_embeddings,
            rope_theta=body.position.rope_theta,
            rope_scaling=body.position.scaling,
            rotary_dim=attn.rotary_dim,
            attention_bias=attn.attention_bias,
            num_local_experts=mlp.num_experts,
            num_experts_per_tok=mlp.top_k,
            scoring_func=mlp.scoring_func,
            use_routing_bias=mlp.use_routing_bias,
            norm_topk_prob=mlp.norm_topk_prob,
            use_qk_norm=attn.use_qk_norm,
            qk_norm_type=attn.qk_norm_type,
            num_mtp_modules=body.num_mtp_modules,
            mtp_transformer_layers=body.mtp_transformer_layers,
        )


def from_hugging_face(hf_config: Any) -> EngineModelSpec:
    if getattr(hf_config, "model_type", None) != "minimax_m2":
        raise ValueError(
            f"expected model_type='minimax_m2', got {getattr(hf_config, 'model_type', None)!r}"
        )

    architectures = [
        str(name)
        for name in (getattr(hf_config, "architectures", None) or [])
        if isinstance(name, str) and name
    ]
    if not architectures:
        raise ValueError("minimax_m2 config must declare architectures")

    num_query_heads = int(hf_config.num_attention_heads)
    num_kv_heads = int(getattr(hf_config, "num_key_value_heads", num_query_heads))
    head_dim = int(hf_config.head_dim)
    rotary_dim = int(getattr(hf_config, "rotary_dim", head_dim))
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        rope_scaling = None

    dtype = getattr(hf_config, "dtype", None)
    if dtype is None:
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        dtype = str(torch_dtype).removeprefix("torch.") if torch_dtype else None
    else:
        dtype = str(dtype)

    quant_cfg = getattr(hf_config, "quantization_config", None)
    if quant_cfg is None:
        quant_cfg = getattr(hf_config, "compression_config", None)

    body = MinimaxM2ModelSpec(
        type="minimax_m2",
        architecture_kind="decoder_only_lm",
        num_layers=int(hf_config.num_hidden_layers),
        hidden_size=int(hf_config.hidden_size),
        vocab_size=int(hf_config.vocab_size),
        hidden_act=str(getattr(hf_config, "hidden_act", "silu")),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        attention=GQAAttentionSpec(
            type="gqa",
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            attention_bias=bool(getattr(hf_config, "attention_bias", False)),
            use_qk_norm=bool(getattr(hf_config, "use_qk_norm", False)),
            qk_norm_type=getattr(hf_config, "qk_norm_type", None),
        ),
        mlp=MoEMLPSpec(
            type="moe",
            num_experts=int(hf_config.num_local_experts),
            top_k=int(hf_config.num_experts_per_tok),
            intermediate_size=int(hf_config.intermediate_size),
            scoring_func=str(getattr(hf_config, "scoring_func", "sigmoid")),
            use_routing_bias=bool(getattr(hf_config, "use_routing_bias", False)),
            norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", False)),
        ),
        norm=RMSNormSpec(
            type="rmsnorm",
            eps=float(getattr(hf_config, "rms_norm_eps", 1e-6)),
        ),
        position=RopePositionSpec(
            type="rope",
            rope_theta=float(hf_config.rope_theta),
            scaling=rope_scaling,
            max_position_embeddings=int(hf_config.max_position_embeddings),
            context_len=int(get_context_length(hf_config)),
        ),
        num_mtp_modules=int(getattr(hf_config, "num_mtp_modules", 0) or 0),
        mtp_transformer_layers=int(
            getattr(hf_config, "mtp_transformer_layers", 1) or 1
        ),
    )

    return EngineModelSpec(
        schema_version=ENGINE_SPEC_SCHEMA_VERSION,
        model_type="minimax_m2",
        architecture=architectures[0],
        architectures=tuple(architectures),
        dtype=dtype,
        quantization=quant_cfg,
        body=body,
    )


__all__ = ["MiniMaxM2RuntimeView", "from_hugging_face"]
