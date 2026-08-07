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

"""GLM-5 (GLM-MoE-DSA) model configuration.

GLM-5 checkpoints ship a custom ``model_type`` of ``glm_moe_dsa`` (architecture
``GlmMoeDsaForCausalLM``). Structurally the model is DeepSeek-V3's MLA/MoE
backbone plus a DSA sparse-attention indexer, so this extends
:class:`DeepseekV3Config` with the indexer geometry and per-layer dispatch
fields the DSA attention reads.
"""

from typing import Any

from tokenspeed.runtime.configs.deepseek_v3_config import DeepseekV3Config

_INDEXER_TYPE_ALIASES = {
    "F": "full",
    "S": "shared",
    "full": "full",
    "shared": "shared",
}


def _normalize_indexer_types(indexer_types: list[str]) -> list[str]:
    normalized = []
    for indexer_type in indexer_types:
        try:
            normalized.append(_INDEXER_TYPE_ALIASES[indexer_type])
        except (KeyError, TypeError) as error:
            raise ValueError(
                "`indexer_types` entries must be one of "
                f"{tuple(_INDEXER_TYPE_ALIASES)}, got {indexer_type!r}"
            ) from error
    return normalized


class GlmMoeDsaConfig(DeepseekV3Config):
    model_type = "glm_moe_dsa"

    # --- GLM-5 checkpoint dimensions (overriding DeepSeek-V3 defaults) ---
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 2048
    v_head_dim: int = 256
    qk_nope_head_dim: int = 192
    n_group: int = 1
    topk_group: int = 1
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5

    # --- DSA sparse-attention indexer geometry ---
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    mlp_bias: bool = False

    # --- per-layer dispatch (derived in ``__post_init__`` when absent) ---
    indexer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        # Some GLM-5 checkpoints ship ``num_experts`` for the routed expert
        # count instead of ``num_local_experts`` / ``n_routed_experts``.
        if (num_experts := kwargs.pop("num_experts", None)) is not None:
            self.n_routed_experts = num_experts

        # Per-layer indexer mode: an explicit ``indexer_types`` wins; otherwise
        # a pattern (e.g. ``"FSSF..."``) overrides the freq/offset schedule.
        index_topk_pattern = kwargs.pop("index_topk_pattern", None)
        index_topk_freq = kwargs.pop("index_topk_freq", None)
        index_skip_topk_offset = kwargs.pop("index_skip_topk_offset", None)

        indexer_types = self.indexer_types
        if indexer_types is None:
            if index_topk_pattern is not None:
                indexer_types = list(index_topk_pattern)
            else:
                freq = 1 if index_topk_freq is None else int(index_topk_freq)
                if freq <= 0:
                    raise ValueError(f"`index_topk_freq` must be positive, got {freq}")
                offset = (
                    int(index_skip_topk_offset)
                    if index_skip_topk_offset is not None
                    else 2
                )
                if freq > 1 and offset <= 0:
                    raise ValueError(
                        "`index_skip_topk_offset` must be positive when "
                        f"`index_topk_freq` is greater than 1, got {offset}"
                    )
                indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]

        self.indexer_types = _normalize_indexer_types(list(indexer_types))
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must equal the "
                f"number of `indexer_types` entries ({len(self.indexer_types)})"
            )
        if self.indexer_types and self.indexer_types[0] == "shared":
            raise ValueError(
                "The first `indexer_types` entry must be `full`; a shared "
                "indexer has no prior layer's top-k to reuse"
            )

        # Per-layer MLP dispatch: the leading dense layers precede the MoE.
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace or 0, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (
                self.num_hidden_layers - n_dense
            )

        # Every layer is DSA -- drives cache-class dispatch.
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers

        super().__post_init__(**kwargs)


__all__ = ["GlmMoeDsaConfig"]
