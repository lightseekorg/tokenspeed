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
prologue returns the query in: FP8 where a decode kernel reads FP8 queries
over the FP8 cache, else the activation dtype (the kernel widens K/V). No
kernel runs."""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.paged_attention import PagedAttention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# No kernel runs, but the answer follows the platform's registered kernels;
# the runtime suite is consumed by AMD CI too.
register_cuda_ci(est_time=5, suite="runtime-1gpu")

BF16, FP8 = torch.bfloat16, torch.float8_e4m3fn


def _backend(
    kv_cache_dtype: torch.dtype, backend_name: str, head_dim: int, page_size: int
) -> MHAAttnBackend:
    spec = MHAConfig(
        backend_name=backend_name,
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
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
        draft_block_decode=False,
        components=(spec,),
    )
    return MHAAttnBackend(config, spec, kernel_page_size=page_size)


def _layer(head_dim: int) -> PagedAttention:
    return PagedAttention(
        8, head_dim, head_dim**-0.5, 2, 0, rotary_emb=None, qk_norm=None
    )


@pytest.mark.parametrize(
    "kv_cache_dtype,mode",
    [
        # A fresh prefill attends the activation-dtype K/V rows.
        (FP8, ForwardMode.EXTEND),
        (FP8, ForwardMode.MIXED),
        (BF16, ForwardMode.DECODE),
    ],
)
def test_activation_dtype_queries_outside_fp8_decode(kv_cache_dtype, mode):
    backend = _backend(kv_cache_dtype, "mha", head_dim=64, page_size=16)
    assert backend.query_dtype(_layer(64), mode, BF16) is BF16


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_triton_decode_widens_an_fp8_cache_under_bf16_queries(head_dim):
    backend = _backend(FP8, "triton", head_dim=head_dim, page_size=16)
    assert backend.query_dtype(_layer(head_dim), ForwardMode.DECODE, BF16) is BF16


@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fp8_decode_query_dtype_names_a_kernel(head_dim, page_size):
    """Whatever the platform's best kernel takes, the chosen query dtype must
    select a decode kernel over the FP8 cache for the layer's traits."""
    backend = _backend(FP8, "mha", head_dim=head_dim, page_size=page_size)
    q_dtype = backend.query_dtype(_layer(head_dim), ForwardMode.DECODE, BF16)
    assert q_dtype in (FP8, BF16)
    select_kernel(
        "attention",
        "mha_decode_with_kvcache",
        format_signature(
            q=dense_tensor_format(q_dtype),
            k_cache=dense_tensor_format(FP8),
            v_cache=dense_tensor_format(FP8),
        ),
        traits={
            "head_dim": head_dim,
            "page_size": page_size,
            "logit_cap": False,
            "return_lse": False,
            "sliding_window": False,
        },
    )


@pytest.mark.skipif(not current_platform().is_cdna4, reason="gfx950 kernels")
@pytest.mark.parametrize(
    "head_dim,page_size,expected",
    [
        # The Gluon decode reads FP8 queries at hd128, page 64.
        (128, 64, FP8),
        # hd256 has no FP8-query kernel; Triton widens the cache.
        (256, 64, BF16),
        (128, 16, BF16),
    ],
)
def test_gfx950_fp8_decode_query_dtype(head_dim, page_size, expected):
    backend = _backend(FP8, "mha", head_dim=head_dim, page_size=page_size)
    assert backend.query_dtype(_layer(head_dim), ForwardMode.DECODE, BF16) is expected
