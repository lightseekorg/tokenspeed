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

"""``LayerMappedKVPool.set_mla_kv_buffer`` layer remap + write pass-through.

Context (the Kimi-K3 CUDA-graph "!!!" bug): under the prefill breakable graph,
the dummy-batch capture (out_cache_loc == the reserved ``dummy_kv_slot``)
writes NaN K/V. The paged MLA decode kernel then reads that shared dummy slot
through the zero-padded block-table entries and computes ``q·k`` *before* the
causal mask, so ``NaN + -inf = NaN`` survives the mask and poisons a live row's
softmax -> all-NaN logits -> ``argmax`` picks token 0 ("!"). Eager prefill leaves
the dummy slot finite (``q·0`` masks cleanly), so the bug only appears with the
prefill graph on. The wrapper requests sanitization from the cache writer so
the cache never stores NaN without allocating temporary tensors.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    LayerMappedKVPool,
)


class _RecordingInnerPool:
    """Minimal stand-in that records exactly what the wrapper forwards."""

    page_size = 64

    def __init__(self):
        self.received = None

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope, sanitize=False):
        self.received = (
            layer.layer_id,
            cache_k_nope,
            cache_k_rope,
            sanitize,
        )


class _Layer:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id


def test_set_mla_kv_buffer_remaps_layer_and_forwards_untouched():
    inner = _RecordingInnerPool()
    # Hybrid model: only layers 3/7/11 are full-attention -> pool slots 0/1/2.
    pool = LayerMappedKVPool(inner, [3, 7, 11])
    layer = _Layer(7)

    # NaN/Inf pass through by design: the write kernel squashes them in-kernel.
    k_nope = torch.tensor([[1.0, float("nan")], [float("inf"), 2.0]])
    k_rope = torch.tensor([[float("nan"), 3.0]])
    loc = torch.zeros(2, dtype=torch.int64)

    pool.set_mla_kv_buffer(layer, loc, k_nope, k_rope)

    got_lid, got_nope, got_rope, sanitize = inner.received
    # Global layer id is remapped to its pool slot for the inner write (7 -> 1)
    # and restored on the layer object afterwards.
    assert got_lid == 1
    assert layer.layer_id == 7
    # Sanitization is owned by the cache writer so the wrapper forwards the
    # original views without allocating two temporary tensors.
    assert sanitize is True
    assert got_nope is k_nope
    assert got_rope is k_rope


def test_set_mla_kv_buffer_noop_for_finite_input():
    inner = _RecordingInnerPool()
    pool = LayerMappedKVPool(inner, [3, 7, 11])
    k_nope = torch.randn(4, 8)
    k_rope = torch.randn(4, 2)
    loc = torch.arange(4, dtype=torch.int64)

    pool.set_mla_kv_buffer(_Layer(3), loc, k_nope, k_rope)

    _, got_nope, got_rope, sanitize = inner.received
    assert sanitize is True
    assert got_nope is k_nope
    assert got_rope is k_rope
