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

"""Nemotron-H configuration with the cache-layer view the runtime plans from.

A Nemotron-H block is one of Mamba2, attention, MoE or MLP. Only Mamba2 and
attention blocks own cache state, so they get dense cache-layer ids in block
order; MoE and MLP blocks have none. Every cache-facing property below is in
cache-layer id space, which is also the ``layer_id`` each mixer passes to the
attention backend.
"""

from __future__ import annotations

from transformers.models.nemotron_h.configuration_nemotron_h import (
    NemotronHConfig as HFNemotronHConfig,
)

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

_CACHE_LABELS = {"mamba": LINEAR_ATTENTION, "attention": FULL_ATTENTION}


class NemotronHConfig(HFNemotronHConfig):
    """transformers' ``NemotronHConfig`` plus dense cache-layer numbering."""

    @property
    def cache_layer_ids(self) -> list[int | None]:
        """Per block, its cache-layer id, or None for a block without cache state."""
        ids: list[int | None] = []
        next_id = 0
        for block_type in self.layers_block_type:
            if block_type in _CACHE_LABELS:
                ids.append(next_id)
                next_id += 1
            else:
                ids.append(None)
        return ids

    @property
    def cache_layer_types(self) -> list[str]:
        """Cache-group label of each cache layer, in cache-layer id order."""
        return [
            _CACHE_LABELS[block_type]
            for block_type in self.layers_block_type
            if block_type in _CACHE_LABELS
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        """Cache-layer ids of the Mamba2 blocks."""
        return [
            i
            for i, label in enumerate(self.cache_layer_types)
            if label == LINEAR_ATTENTION
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        """Cache-layer ids of the attention blocks."""
        return [
            i
            for i, label in enumerate(self.cache_layer_types)
            if label == FULL_ATTENTION
        ]
