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

"""Gated DeltaNet layers (the Qwen3.5 family) on the shared recurrent-state backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill_capturable

from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
    CapacityPrefillBackend,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )


class GdnAttnBackend(CapacityPrefillBackend):
    """GDN layers, captured inline by the prefill graph where the scan allows it.

    The shared base already runs GDN; this class only admits capacity-shaped
    prefills, which needs a chunk-prefill kernel that reads its sequence bounds
    on device (see ``gdn_chunk_prefill_capturable``).
    """

    # The scan reads its bounds on device, so packing would only add work to eager forwards.
    _capacity_layout_when_uncaptured = False

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        linear_attn = config.component(LinearAttnConfig)
        self._prefill_capturable = gdn_chunk_prefill_capturable(
            config.dtype,
            head_dim=linear_attn.head_k_dim,
            value_head_dim=linear_attn.head_v_dim,
            num_q_heads=linear_attn.num_k_heads // linear_attn.tp_size,
            num_v_heads=linear_attn.num_v_heads // linear_attn.tp_size,
            qk_l2norm=True,
        )

    def _admits_capacity_prefill(self) -> bool:
        return self._prefill_capturable
