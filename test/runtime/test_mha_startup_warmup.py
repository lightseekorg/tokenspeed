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

"""Startup kernel warmup reaches the MHA backend's cached-prefix extend.

CPU only: the attention kernels, which need a GPU, are replaced by stand-ins
with their public signatures.
"""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.ops.tuning import kernel_warmup

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged import mha
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.paged_attention import PagedAttention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# This suite is CPU-only, but the runtime suite is consumed by AMD CI too.
register_cuda_ci(est_time=5, suite="runtime-1gpu")

HEADS, KV_HEADS, DIM, PAGE = 8, 2, 64, 16


class _Cache:
    """Tiny cache replacing GPU-backed pool storage, not runtime dispatch."""

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def get_key_buffer(self, layer_id):
        return self.key

    def get_value_buffer(self, layer_id):
        return self.value


def _prefill_cpu(*, q, **kwargs):
    # Any deterministic function of the rows: the test compares the prefill
    # result with and without warmup, not attention numerics.
    return q * 2


@pytest.mark.parametrize("window", [-1, 7])
def test_fresh_prefill_warmup_also_runs_the_cached_extend(monkeypatch, window):
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=DIM,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=256,
        max_bs=4,
        speculative_num_draft_tokens=1,
        is_draft=False,
        draft_block_decode=False,
        components=(spec,),
    )
    backend = mha.MHAAttnBackend(config, spec, kernel_page_size=PAGE)
    # Solutions that attend fresh rows directly report "postwrite".
    monkeypatch.setattr(backend, "plan", lambda **kwargs: {"extend_mode": "postwrite"})
    extends = []
    monkeypatch.setattr(mha, "mha_prefill", _prefill_cpu)
    monkeypatch.setattr(
        mha,
        "mha_extend_with_kvcache",
        lambda **kwargs: extends.append(kwargs) or torch.zeros_like(kwargs["q"]),
    )

    queries = [5, 3]
    tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    lengths = torch.tensor(queries, dtype=torch.int32)
    backend.init_forward_metadata(
        len(queries),
        len(queries),
        lengths,
        tables,
        ForwardMode.EXTEND,
        extend_seq_lens=lengths,
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=torch.zeros_like(lengths),
        extend_prefix_lens_cpu=torch.zeros_like(lengths),
        extend_with_prefix=False,
        query_shard=None,
        page_table_cpu=None,
    )
    layer = PagedAttention(
        HEADS,
        DIM,
        DIM**-0.5,
        KV_HEADS,
        0,
        sliding_window_size=window,
        rotary_emb=None,
        qk_norm=None,
    )
    generator = torch.Generator().manual_seed(7)
    cache_shape = (5, PAGE, KV_HEADS, DIM)
    cache = _Cache(
        torch.randn(cache_shape, generator=generator).bfloat16(),
        torch.randn(cache_shape, generator=generator).bfloat16(),
    )
    rows = torch.randn((sum(queries), HEADS, DIM), generator=generator).bfloat16()

    def forward():
        return backend.forward_extend(
            rows,
            rows[:, :KV_HEADS],
            rows[:, :KV_HEADS],
            layer,
            torch.empty(0, dtype=torch.int32),
            cache,
            len(queries),
        )

    served = forward()
    assert extends == []
    with kernel_warmup(enabled=True):
        warmed = forward()
    # The warmup extend runs once with the layer's window, and its output is
    # dropped.
    assert torch.equal(warmed, served)
    assert len(extends) == 1
    assert extends[0]["window_left"] == window
