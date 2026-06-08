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

"""Engine model IR: stable shell + typed architecture/component variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ENGINE_SPEC_SCHEMA_VERSION = 1

ArchitectureKind = Literal["decoder_only_lm"]


@dataclass(frozen=True)
class GQAAttentionSpec:
    type: Literal["gqa"]
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    attention_bias: bool
    use_qk_norm: bool
    qk_norm_type: str | None


@dataclass(frozen=True)
class MoEMLPSpec:
    type: Literal["moe"]
    num_experts: int
    top_k: int
    intermediate_size: int
    scoring_func: str
    use_routing_bias: bool
    norm_topk_prob: bool


@dataclass(frozen=True)
class RopePositionSpec:
    type: Literal["rope"]
    rope_theta: float
    scaling: dict[str, Any] | None
    max_position_embeddings: int
    context_len: int


@dataclass(frozen=True)
class RMSNormSpec:
    type: Literal["rmsnorm"]
    eps: float


@dataclass(frozen=True)
class MinimaxM2ModelSpec:
    """First architecture variant; add more typed bodies as adapters land."""

    type: Literal["minimax_m2"]
    architecture_kind: ArchitectureKind
    num_layers: int
    hidden_size: int
    vocab_size: int
    hidden_act: str
    tie_word_embeddings: bool
    attention: GQAAttentionSpec
    mlp: MoEMLPSpec
    norm: RMSNormSpec
    position: RopePositionSpec
    num_mtp_modules: int = 0
    mtp_transformer_layers: int = 1


ModelBody = MinimaxM2ModelSpec


@dataclass(frozen=True)
class EngineModelSpec:
    """Stable entry protocol; execution fields live in ``body``."""

    schema_version: int
    model_type: str
    architecture: str
    architectures: tuple[str, ...]
    dtype: str | None
    quantization: dict[str, Any] | None
    body: ModelBody


def build_engine_spec(hf_config: Any) -> EngineModelSpec | None:
    model_type = getattr(hf_config, "model_type", None)
    if model_type == "minimax_m2":
        from tokenspeed.runtime.configs.adapters.minimax_m2 import from_hugging_face

        return from_hugging_face(hf_config)
    return None


__all__ = [
    "ENGINE_SPEC_SCHEMA_VERSION",
    "EngineModelSpec",
    "GQAAttentionSpec",
    "MinimaxM2ModelSpec",
    "MoEMLPSpec",
    "ModelBody",
    "RMSNormSpec",
    "RopePositionSpec",
    "build_engine_spec",
]
