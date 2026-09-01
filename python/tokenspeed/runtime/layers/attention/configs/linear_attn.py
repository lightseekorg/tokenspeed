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

"""Linear-attention (KDA / GDN) boot-time configuration.

``LinearAttnConfig`` is the linear-attention component: registered per
architecture (``registry._LINEAR_ATTN_CLS``), built through the same
``generate()`` protocol as the softmax families, and carried in
``AttnConfig.components``. Models without linear-attention layers simply
have no such component.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import AttnComponentSpec
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(kw_only=True)
class LinearAttnConfig(AttnComponentSpec):
    """Boot-constant facts about a model's linear-attention layers.

    Field names are unified across the two in-tree families:

    * Kimi-K3 KDA — ``linear_attn_config`` dict: symmetric heads/dims
      (``num_heads``/``head_dim`` for both k and v).
    * Qwen3.5 GDN — flat ``linear_*`` fields: asymmetric key/value heads.

    ``tp_size`` is the linear-attention TP width distilled from
    ``mapping.linear_attn`` (the same copy-a-scalar convention as
    ``SoftmaxAttnConfig.attn_tp_size``).
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    # 0-based global indices of the linear-attention layers. Non-empty by
    # construction: a model without linear layers gets no LinearAttnConfig.
    layer_ids: tuple[int, ...]
    tp_size: int
    # Whether verify replays the SSM state instead of checkpoint-restoring.
    # Resolved by the GDN cache recipe after checking the engine option,
    # verify width, device, and registered kernel support.
    replay_ssm: bool = False

    def __post_init__(self):
        if not self.layer_ids:
            raise ValueError("layer_ids must be non-empty")
        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")
        if self.num_k_heads % self.tp_size or self.num_v_heads % self.tp_size:
            raise ValueError(
                f"linear-attention heads (k={self.num_k_heads}, "
                f"v={self.num_v_heads}) must be divisible by "
                f"tp_size={self.tp_size}"
            )

    @property
    def conv_dim(self) -> int:
        """Width of the short causal conv over q/k/v (q and k share key geometry)."""
        return (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )

    @property
    def conv_state_shape(self) -> tuple[int, int]:
        """Per-rank rolling conv state: (conv_dim / tp, kernel - 1)."""
        return (self.conv_dim // self.tp_size, self.conv_kernel_size - 1)

    @property
    def temporal_state_shape(self) -> tuple[int, int, int]:
        """Per-rank recurrent (delta-rule/SSM) state, K-last: (Hv / tp, V, K)."""
        return (
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
            self.head_k_dim,
        )

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> LinearAttnConfig | None:
        """Build the linear-attention config, or None for checkpoints without one.

        Which architectures may carry this component is declared in
        ``registry._LINEAR_ATTN_CLS``; here the checkpoint decides presence
        via a non-empty ``linear_layer_ids`` (not field presence — base
        configs expose the geometry fields with defaults even for
        full-attention-only variants, and NextN drafts may have no linear
        layers). ``is_draft`` is accepted for construction-protocol parity;
        the linear half has no draft-specific facts yet.
        """
        del is_draft
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        linear_layer_ids = getattr(text_config, "linear_layer_ids", None)
        if not linear_layer_ids:
            return None

        tp_size = server_args.mapping.linear_attn.tp_size
        kda = getattr(text_config, "linear_attn_config", None)
        if isinstance(kda, dict):
            # Kimi-K3 KDA: symmetric k/v geometry in one dict.
            num_heads = int(kda["num_heads"])
            head_dim = int(kda["head_dim"])
            return cls(
                num_k_heads=num_heads,
                num_v_heads=num_heads,
                head_k_dim=head_dim,
                head_v_dim=head_dim,
                conv_kernel_size=int(kda["short_conv_kernel_size"]),
                layer_ids=tuple(linear_layer_ids),
                tp_size=tp_size,
            )
        # Qwen3.5 GDN: flat fields, asymmetric k/v heads.
        return cls(
            num_k_heads=int(text_config.linear_num_key_heads),
            num_v_heads=int(text_config.linear_num_value_heads),
            head_k_dim=int(text_config.linear_key_head_dim),
            head_v_dim=int(text_config.linear_value_head_dim),
            conv_kernel_size=int(text_config.linear_conv_kernel_dim),
            layer_ids=tuple(linear_layer_ids),
            tp_size=tp_size,
        )
