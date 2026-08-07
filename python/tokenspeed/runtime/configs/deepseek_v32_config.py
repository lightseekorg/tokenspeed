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

"""DeepSeek-V3.2 (``DeepseekV32ForCausalLM``) model configuration.

DeepSeek-V3.2 checkpoints ship ``model_type`` ``deepseek_v32`` (architecture
``DeepseekV32ForCausalLM``). Structurally the model is DeepSeek-V3's MLA/MoE
backbone plus a DSA sparse-attention indexer -- the same shape as GLM-DSA -- so
this extends :class:`DeepseekV3Config` with the indexer geometry and per-layer
dispatch fields the DSA attention reads.
"""

from typing import Any

from tokenspeed.runtime.configs.deepseek_v3_config import DeepseekV3Config


class DeepseekV32Config(DeepseekV3Config):
    model_type = "deepseek_v32"

    # DeepSeek-V3.2 trains at a longer context than the V3 base default.
    max_position_embeddings: int = 163840

    # --- DSA sparse-attention indexer geometry ---
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
    mlp_bias: bool = False

    # --- per-layer dispatch (derived in ``__post_init__`` when absent) ---
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        # Some V3.2 checkpoints ship ``num_experts`` for the routed expert
        # count instead of ``num_local_experts`` / ``n_routed_experts``.
        if (num_experts := kwargs.pop("num_experts", None)) is not None:
            self.n_routed_experts = num_experts

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


__all__ = ["DeepseekV32Config"]
