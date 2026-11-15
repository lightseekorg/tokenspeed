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

"""RoPE row selection preserves native eager and dynamic graph results."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel_npu.ops.rotary_embedding import apply_rope


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "is_neox,key_only", [(True, False), (False, False), (True, True)]
)
def test_selected_rope_cache(dtype, is_neox, key_only):
    torch.npu.set_device(0)
    torch.manual_seed(37)
    cache = torch.randn(4096, 128, device="npu", dtype=torch.float32)
    positions = torch.tensor(
        [4095, 0, 37, 37, 128, 4094, 1, 2], device="npu", dtype=torch.int64
    )
    q = torch.randn(8, 0 if key_only else 512, device="npu", dtype=dtype)
    k = torch.randn(8, 128, device="npu", dtype=dtype)
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)

    def run():
        apply_rope(
            positions=positions,
            q=q,
            k=k,
            head_size=128,
            cos_sin_cache=cache,
            is_neox=is_neox,
            q_rope_out=qo,
            k_rope_out=ko,
        )

    def check():
        native_q = torch.zeros_like(k) if key_only else q
        eq, ek = torch_npu.npu_mrope(
            positions,
            native_q,
            k,
            cache.to(dtype),
            128,
            mrope_section=[0, 0, 0],
            rotary_mode="half" if is_neox else "interleave",
        )
        if not key_only:
            torch.testing.assert_close(qo, eq, atol=0, rtol=0)
        torch.testing.assert_close(ko, ek, atol=0, rtol=0)

    import torch_npu

    run()
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for offset in [1, 137, 4095]:
        positions.add_(offset).remainder_(4096)
        q.add_(0.01)
        k.sub_(0.02)
        graph.replay()
        check()
