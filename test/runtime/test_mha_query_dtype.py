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

"""The MHA backend names the query dtype its attention reads, which the GQA
prologue returns the query in. CPU only: no kernel runs."""

import os
import sys

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.paged_attention import PagedAttention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# This suite is CPU-only, but the runtime suite is consumed by AMD CI too.
register_cuda_ci(est_time=5, suite="runtime-1gpu")

BF16, FP8 = torch.bfloat16, torch.float8_e4m3fn


def _backend(kv_cache_dtype: torch.dtype) -> MHAAttnBackend:
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=64,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=BF16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_method="none",
        prefix_granularity=16,
        context_len=256,
        max_bs=4,
        speculative_num_draft_tokens=1,
        is_draft=False,
        draft_block_decode=False,
        components=(spec,),
    )
    return MHAAttnBackend(config, spec, kernel_page_size=16)


@pytest.mark.parametrize(
    "kv_cache_dtype,mode,expected",
    [
        # Decode reads the FP8 cache with FP8 queries.
        (FP8, ForwardMode.DECODE, FP8),
        # A fresh prefill attends the activation-dtype K/V rows.
        (FP8, ForwardMode.EXTEND, BF16),
        (FP8, ForwardMode.MIXED, BF16),
        (BF16, ForwardMode.DECODE, BF16),
    ],
)
def test_decode_over_an_fp8_cache_reads_fp8_queries(kv_cache_dtype, mode, expected):
    layer = PagedAttention(8, 64, 0.125, 2, 0, rotary_emb=None, qk_norm=None)
    assert _backend(kv_cache_dtype).query_dtype(layer, mode, BF16) is expected
