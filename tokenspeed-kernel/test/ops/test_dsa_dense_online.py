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
from tokenspeed_kernel.ops.attention.dsa._triton.bf16_tiled import run_bf16_tiled
from tokenspeed_kernel.ops.attention.dsa.triton import _dsa_dense_kv_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("implementation", ["scalar", "tiled"])
@pytest.mark.parametrize("rank,rope", [(128, 0), (128, 64), (512, 0), (512, 64)])
@pytest.mark.parametrize("heads", [1, 17])
@pytest.mark.parametrize("width", [37, 2051])
def test_dense_online_softmax(rank, rope, width, implementation, heads, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    if implementation == "tiled" and torch.version.hip:
        pytest.skip("Tiled path has only been validated on NVIDIA")
    torch.manual_seed(37)
    n, page = 5, 64
    q = torch.randn(n, heads, rank + rope, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(64, page, 2 * (rank + rope), device="cuda", dtype=q.dtype)[
        ..., ::2
    ]
    slots = torch.randint(0, 4096, (n, width), device="cuda", dtype=torch.int32)
    slots[1].fill_(-1)
    slots[2, :32] = -1
    lengths = torch.tensor(
        [0, width, width, 17, width], device="cuda", dtype=torch.int32
    )
    out = torch.empty(n, heads, rank, device="cuda", dtype=q.dtype)
    lse = torch.empty(n, heads, device="cuda", dtype=torch.float32)

    def launch():
        if implementation == "tiled":
            _, result_lse = run_bf16_tiled(
                q,
                kv,
                slots,
                lengths,
                kv_lora_rank=rank,
                qk_rope_head_dim=rope,
                scale=0.07,
                return_lse=True,
                out=out,
            )
            lse.copy_(result_lse)
            return
        _dsa_dense_kv_kernel[(n, heads, rank // 64)](
            q,
            kv,
            slots,
            lengths,
            out,
            lse,
            True,
            heads,
            rank + rope,
            rank,
            rope,
            *kv.stride(),
            page,
            width,
            0.07,
            BLOCK_TOPK=32,
            BLOCK_K=64,
            BLOCK_V=64,
            num_warps=4,
            num_stages=1,
        )

    def check():
        flat = kv.reshape(-1, rank + rope).float()
        for row in range(n):
            mask = (torch.arange(width, device="cuda") < lengths[row]) & (
                slots[row] >= 0
            )
            selected = flat[slots[row, mask].long()]
            if selected.shape[0]:
                scores = q[row].float() @ selected.T * 0.07
                ref = scores.softmax(-1) @ selected[:, :rank]
                torch.testing.assert_close(
                    out[row].float(), ref, atol=0.004, rtol=0.008
                )
                torch.testing.assert_close(
                    lse[row], scores.logsumexp(-1), atol=1e-5, rtol=1e-5
                )
            else:
                assert torch.count_nonzero(out[row]) == 0
                assert torch.isneginf(lse[row]).all()

    launch()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    lengths.fill_(width)
    slots[1].fill_(0)
    q.mul_(0.5)
    graph.replay()
    check()
