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

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_blackwell
    or current_platform().arch_version.minor > 3,
    reason="requires SM100-SM103 FlashMLA sparse prefill",
)


def test_prefill_one_call_output_beyond_int32_matches_query_tiles():
    # One call over a whole prefill batch: the output [rows, 64, 512] holds
    # more than 2**31 elements, so its last rows start beyond int32 offsets.
    rows, heads, kv_rows, width = 67_584, 64, 640, 640
    assert (rows - 1) * heads * 512 >= 2**31
    torch.manual_seed(0)
    q = torch.randn((rows, heads, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((kv_rows, 1, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(
        -1, kv_rows, (rows, width), device="cuda", dtype=torch.int32
    )
    # Rows near both ends select nothing and must produce zeros.
    empty = torch.tensor([0, 1, rows - 2, rows - 1], device="cuda")
    indices[empty] = -1
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)

    def run(query_chunk_size):
        return dsv41.selected_attention(
            q,
            None,
            None,
            None,
            None,
            None,
            None,
            sink,
            512**-0.5,
            None,
            query_chunk_size,
            None,
            kv,
            indices,
            lse=None,
        )

    whole = run(rows)
    assert whole.shape == q.shape and whole.is_contiguous()
    tiled = run(2048)
    torch.testing.assert_close(whole, tiled, rtol=0, atol=0)
    assert not whole[empty].any()
