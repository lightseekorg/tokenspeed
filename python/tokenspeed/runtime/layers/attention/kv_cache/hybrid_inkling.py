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

"""Inkling views over a hybrid MHA cache buffer."""

from __future__ import annotations

import re
from functools import cached_property

import torch

from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
    HybridMHATokenToKVPoolMXFP8,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import (
    conv_checkpoint_field_prefix,
)

_SHARED_CONV_FIELD = re.compile(r"layer\.(\d+)\.conv(\d+)\.kvconv_k")


class HybridInklingTokenToKVPool(HybridMHATokenToKVPool):
    """Hybrid MHA pool with Inkling ShortConv checkpoint views.

    The checkpoint lookups take a convolution layer id, which is the cache
    layer's own unless the recipe placed the layer's checkpoints on another
    cache layer's pages (a later pass under a shared loop cache).
    """

    @cached_property
    def _shared_conv_cache_layers(self) -> dict[int, int]:
        """View-local convolution layer -> the view-local cache layer whose
        pages carry its checkpoints, for those the plan names on another
        cache layer (field ids number both from the merged plan's start)."""
        owned = self.field_layer_range
        shared = {}
        for field in self.arena.plan.fields:
            match = _SHARED_CONV_FIELD.fullmatch(field.field_id)
            if match and int(match[1]) in owned:
                shared[int(match[2]) - owned.start] = int(match[1]) - owned.start
        return shared

    def _conv_checkpoint(self, layer_id: int) -> tuple[int, str]:
        """``(view-local cache layer, field-id prefix)`` of one convolution
        layer's checkpoints."""
        cache_layer = self._shared_conv_cache_layers.get(layer_id, layer_id)
        field_layer = self._field_layer_id(cache_layer)
        return cache_layer, conv_checkpoint_field_prefix(
            field_layer, field_layer + layer_id - cache_layer
        )

    def kvconv_checkpoint_buffers(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_layer, prefix = self._conv_checkpoint(layer_id)
        # The K/V short conv reads these before the layer's attention waits for
        # its Host load, and a hit loads the input checkpoint.
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(cache_layer)
        return (
            self.arena.field(f"{prefix}.kvconv_k"),
            self.arena.field(f"{prefix}.kvconv_v"),
        )

    def hiddenconv_checkpoint(
        self, layer_id: int, component: str
    ) -> tuple[str, torch.Tensor]:
        """The hiddenconv group holding this stream's checkpoint, and its view."""
        if component not in ("attnconv", "mlpconv"):
            raise ValueError(f"unknown Inkling hidden-conv component {component!r}")
        field_id = f"{self._conv_checkpoint(layer_id)[1]}.{component}"
        return self.arena.plan.field(field_id).group_id, self.arena.field(field_id)


class HybridInklingTokenToKVPoolMXFP8(
    HybridInklingTokenToKVPool,
    HybridMHATokenToKVPoolMXFP8,
):
    pass
