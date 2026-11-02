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

"""FP8 QSA graph refresh across target and draft row geometries on Rubin."""

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform().arch_version != ArchVersion(10, 7),
    reason="Rubin FP8 QSA head-tile regression",
)


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    ("rows", "query_width", "large_offsets", "kv_heads"),
    [
        (96, 1, False, 2),
        (105, 1, False, 2),
        (106, 1, False, 2),
        (128, 4, False, 2),
        (384, 4, False, 2),
        (128, 4, True, 2),
        (384, 4, False, 1),
    ],
)
def test_fp8_target_and_draft_graphs_refresh_without_compilation(
    rows, query_width, large_offsets, kv_heads, enable_pdl
):
    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse
    from tokenspeed_kernel.workspace import capture_session
    from utils import move_pages_past_int32

    torch.manual_seed(443 + rows)
    query = torch.randn((rows, kv_heads * 12, 256), device="cuda", dtype=torch.bfloat16)
    keys = (torch.randn((4096, kv_heads, 256), device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    values = (torch.randn((4096, kv_heads, 256), device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    keys[0].fill_(float("nan"))
    values[0].fill_(float("nan"))
    selected = torch.randint(1, 4096, (rows, 2051), device="cuda", dtype=torch.int32)
    selected[:, 9::17] = 0
    selected[::4, -3:] = -1
    selected[-query_width:] = -1
    if large_offsets:
        (keys, values), far = move_pages_past_int32((keys, values), selected)
        selected = torch.where(selected > 0, far, selected)
        assert int(selected[selected > 0].min()) * keys[0].numel() > 2**31
        keys[1].zero_()
        values[1].zero_()
    qsa_sparse.warmup(query, keys, values, max_rows=384, enable_pdl=enable_pdl)
    compiled = dict(qsa_sparse._COMPILED_KERNELS)
    previous = pdl_enabled()
    pdl_enabled(enable_pdl)
    try:

        def run():
            return qsa_sparse_attention(
                query,
                keys,
                values,
                selected,
                scale=1 / 16,
                max_seqlen_q=query_width,
                metadata_capacity_rows=None,
                k_scale=0.5,
                v_scale=0.25,
                override=None,
                solution="cute_dsl",
                sparse_metadata=None,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with capture_session(graph), torch.cuda.graph(graph):
            output = run()
        for factor in (0.75, -0.5):
            query.mul_(factor)
            selected[:, 1000:2048] = -1
            values[-4095:].copy_(
                (torch.randn_like(values[-4095:], dtype=torch.float32) * 0.25).to(
                    values.dtype
                )
            )
            graph.replay()
            expected = torch_qsa_selected_attention(
                query, keys, values, selected, 1 / 16, 0.5, 0.25
            )
            torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)
            torch.testing.assert_close(run(), output, atol=0, rtol=0)
            assert torch.count_nonzero(output[-query_width:]) == 0
        assert qsa_sparse._COMPILED_KERNELS == compiled
    finally:
        pdl_enabled(previous)
