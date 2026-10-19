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

"""Numerics and graph replay for split-K sliding MHA decode on gfx950."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
    get_config,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (  # noqa: E402
    launch_gluon_mha_decode_gfx950 as gluon_mha_decode_gfx950,
)

HEAD_DIM = 64
Q_HEADS = 32
KV_HEADS = 4
PAGE_SIZE = 64
SEQUENCE_LENGTH = 512
WINDOW_LEFT = SEQUENCE_LENGTH - 1


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    pages_per_request = SEQUENCE_LENGTH // PAGE_SIZE
    outputs = []
    for batch_index in range(q.shape[0]):
        pages = slice(
            batch_index * pages_per_request, (batch_index + 1) * pages_per_request
        )
        out, _ = torch_dense_attention(
            q[batch_index, None],
            k_cache[pages].flatten(0, 1),
            v_cache[pages].flatten(0, 1),
            scale=HEAD_DIM**-0.5,
            causal_offset=None,
            window_left=None,
            sinks=sinks,
        )
        outputs.append(out[0])
    return torch.stack(outputs).to(torch.bfloat16)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_split_sliding_decode_matches_reference_and_replays(batch: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260915 + batch)
    pages_per_request = SEQUENCE_LENGTH // PAGE_SIZE
    q = torch.randn(
        batch,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_cache = torch.randn(
        batch * pages_per_request,
        PAGE_SIZE,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v_cache = torch.randn(
        k_cache.shape,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    page_table = torch.arange(
        batch * pages_per_request,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch, pages_per_request)
    cache_seqlens = torch.full(
        (batch,),
        SEQUENCE_LENGTH,
        dtype=torch.int32,
        device="cuda",
    )
    sinks = torch.randn(
        Q_HEADS, dtype=torch.float32, device="cuda", generator=generator
    )
    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=SEQUENCE_LENGTH,
        window_left=WINDOW_LEFT,
        softmax_scale=None,
    )
    expected_splits = 8 if batch <= 4 else 4
    assert config.num_kv_splits == expected_splits

    def invoke() -> torch.Tensor:
        return gluon_mha_decode_gfx950(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=SEQUENCE_LENGTH,
            max_seqlen_q=1,
            window_left=WINDOW_LEFT,
            logit_cap=0.0,
            sinks=sinks,
            return_lse=False,
            softmax_scale=None,
            q_scale=None,
            k_scale=None,
            v_scale=None,
        )

    expected = _reference(q, k_cache, v_cache, sinks)
    actual = invoke()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = invoke()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, actual, atol=0, rtol=0)
