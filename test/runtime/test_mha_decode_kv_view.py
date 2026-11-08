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

"""The MHA backend's decode KV view: what a fused kernel may read instead of
the backend's decode, and the layers and rounds it declines. CPU only: no
kernel runs."""

import os
import sys

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeVerifyInputs,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.paged_attention import PagedAttention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# This suite is CPU-only, but the runtime suite is consumed by AMD CI too.
register_cuda_ci(est_time=5, suite="runtime-1gpu")

BF16, FP8 = torch.bfloat16, torch.float8_e4m3fn
PAGE, KV_HEADS, HEAD_DIM = 16, 2, 64


class _Pool:
    """The two per-layer buffers the backend views (the real pool needs a
    device allocator)."""

    def __init__(self, dtype):
        self.k = torch.zeros(8 * PAGE, KV_HEADS, HEAD_DIM, dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v


def _backend(kv_cache_dtype, *, draft_block_decode=False):
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=8,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=BF16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_method="none",
        context_len=256,
        max_bs=4,
        speculative_num_draft_tokens=1,
        is_draft=False,
        draft_block_decode=draft_block_decode,
        components=(spec,),
    )
    backend = MHAAttnBackend(config, spec, kernel_page_size=PAGE)
    backend.init_cuda_graph_state(4)
    return backend


def _layer(window_left):
    return PagedAttention(
        8,
        HEAD_DIM,
        0.125,
        KV_HEADS,
        0,
        sliding_window_size=window_left,
        rotary_emb=None,
        qk_norm=None,
    )


def _publish(backend, bs=2):
    backend.seq_lens_buf[:bs] = torch.tensor([40, 100])[:bs]
    backend.publish_decode_metadata(bs, backend.seq_lens_buf[:bs], num_extends=0)


def test_sliding_fp8_layer_views_the_decode_cache():
    backend = _backend(FP8)
    _publish(backend)
    pool = _Pool(FP8)
    view = backend.decode_kv_view(_layer(2047), pool)
    assert view is not None
    assert view.k_cache.shape == (8, PAGE, KV_HEADS, HEAD_DIM)
    assert view.k_cache.data_ptr() == pool.k.data_ptr()
    assert view.page_table is backend.forward_decode_metadata.page_table
    assert view.seq_lens is backend.forward_decode_metadata.seq_lens
    assert view.max_seqlen_k == backend.max_context_len


@pytest.mark.parametrize(
    "case",
    ["no decode metadata", "bf16 cache", "full attention", "scaled cache"],
)
def test_view_declines_what_it_cannot_carry(case):
    backend = _backend(BF16 if case == "bf16 cache" else FP8)
    if case != "no decode metadata":
        _publish(backend)
    layer = _layer(-1 if case == "full attention" else 2047)
    if case == "scaled cache":
        layer.load_kv_cache_scale("self_attn.kv_scale", torch.tensor(0.5))
    assert backend.decode_kv_view(layer, _Pool(FP8)) is None


def test_view_declines_block_draft_decode():
    backend = _backend(FP8, draft_block_decode=True)
    _publish(backend)
    assert backend.decode_kv_view(_layer(2047), _Pool(FP8)) is None


def test_view_declines_while_a_draft_tree_is_bound():
    """Verify with a draft tree attends through the tree mask, which the view
    does not carry."""
    backend = _backend(FP8)
    _publish(backend)
    nodes = 4
    backend.bind_tree_verify(
        TreeVerifyInputs(
            mask=torch.zeros(4 * nodes, dtype=torch.int64),
            num_nodes=nodes,
            parent=torch.full((4, nodes), -1, dtype=torch.int32),
        )
    )
    assert backend.decode_kv_view(_layer(2047), _Pool(FP8)) is None
