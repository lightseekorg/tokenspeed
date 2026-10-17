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

torch_npu = pytest.importorskip("torch_npu")


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("tokens", (8, 512))
@pytest.mark.parametrize("masked", (False, True))
@torch.inference_mode()
def test_mla_cache_mask_graph(tokens: int, masked: bool):
    from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton

    torch.npu.set_device(0)
    cache = torch.full((tokens + 1, 576), 7, dtype=torch.bfloat16, device="npu")
    latent = torch.randn(tokens, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(tokens, 64, dtype=torch.bfloat16, device="npu")
    slots = torch.arange(1, tokens + 1, dtype=torch.int64, device="npu")
    owned = torch.arange(tokens, device="npu") % 2 == 0
    mask = owned if masked else None
    if masked:
        slots[~owned] = 0

    def write():
        set_mla_kv_buffer_triton(
            cache, slots, latent, rope, enable_pdl=False, write_mask=mask
        )

    def check():
        expected = torch.full_like(cache, 7).cpu()
        values = torch.cat((latent, rope), dim=-1).cpu()
        select = owned.cpu() if masked else torch.ones(tokens, dtype=torch.bool)
        expected[slots.cpu()[select]] = values[select]
        torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)

    write()
    check()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        write()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        write()
    for iteration in range(3):
        latent.add_(1)
        rope.mul_(-1)
        if masked and iteration == 2:
            owned.zero_()
        cache.fill_(7)
        graph.replay()
        torch.npu.synchronize()
        check()
    graph.reset()
