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

import torch

from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
    HybridMHATokenToKVPoolMXFP8,
)


class HybridInklingTokenToKVPool(HybridMHATokenToKVPool):
    """Hybrid MHA pool with Inkling ShortConv checkpoint views."""

    def kvconv_checkpoint_buffers(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        field_layer = self._field_layer_id(layer_id)
        return (
            self.arena.field(f"layer.{field_layer}.kvconv_k"),
            self.arena.field(f"layer.{field_layer}.kvconv_v"),
        )

    def hiddenconv_checkpoint_buffer(
        self, layer_id: int, component: str
    ) -> torch.Tensor:
        if component not in ("attnconv", "mlpconv"):
            raise ValueError(f"unknown Inkling hidden-conv component {component!r}")
        return self.arena.field(f"layer.{self._field_layer_id(layer_id)}.{component}")


class HybridInklingTokenToKVPoolMXFP8(
    HybridInklingTokenToKVPool,
    HybridMHATokenToKVPoolMXFP8,
):
    pass
