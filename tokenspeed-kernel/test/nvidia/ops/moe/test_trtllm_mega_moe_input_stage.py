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

"""Check the fused stager against FlashInfer quantization and captured reuse."""

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize  # noqa: E402
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.input_stage import (  # noqa: E402
    Nvfp4InputStage,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden", [128, 3584, 8192])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("routes_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("scores_dtype", [torch.float32, torch.bfloat16])
def test_fused_nvfp4_input_stage(hidden, ids_dtype, routes_dtype, scores_dtype):
    torch.manual_seed(710)
    capacity, top_k = 16, 10
    stager = Nvfp4InputStage(hidden, top_k)
    scale = torch.tensor([1 / 0.75], device="cuda")
    q = torch.empty(capacity, hidden // 2, device="cuda", dtype=torch.uint8)
    sf = torch.empty(capacity, hidden // 16, device="cuda", dtype=torch.float8_e4m3fn)
    routes = torch.empty(capacity, top_k, device="cuda", dtype=routes_dtype)
    weights = torch.empty(capacity, top_k, device="cuda", dtype=torch.float32)

    def inputs(n):
        x = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
        x[:, :16] = 0
        ids = torch.randint(0, 512, (n, top_k), device="cuda", dtype=ids_dtype)
        scores = torch.rand(n, top_k, device="cuda").to(scores_dtype)
        return x, ids, scores

    def run(x, ids, scores):
        stager.run(x, scale, ids, scores, q, sf, routes, weights)

    def check(x, ids, scores):
        n = x.shape[0]
        if n:
            expected_q, expected_sf = fp4_quantize(
                x, scale, is_sf_swizzled_layout=False, enable_pdl=False
            )
            torch.testing.assert_close(q[:n], expected_q, atol=0, rtol=0)
            torch.testing.assert_close(
                sf[:n].view(torch.uint8), expected_sf.view(torch.uint8), atol=0, rtol=0
            )
        torch.testing.assert_close(routes[:n], ids.to(routes_dtype), atol=0, rtol=0)
        torch.testing.assert_close(weights[:n], scores.float(), atol=0, rtol=0)
        assert (routes[n:] == -1).all()
        assert (weights[n:] == 0).all()

    # The same compiled stager must handle changing live counts, zero owners,
    # and a shrinking batch without leaving previously live routes behind.
    for n in (3, 13, 1, 0, 7):
        x, ids, scores = inputs(n)
        run(x, ids, scores)
        check(x, ids, scores)

    captured = []
    for n in (3, 13, 0):
        x, ids, scores = inputs(n)
        run(x, ids, scores)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(x, ids, scores)
        captured.append((graph, x, ids, scores))

    # Update both activations and the layer scale after capture. Replays must
    # read the current values, including when a smaller graph follows a larger one.
    scale.fill_(0.5)
    for graph, x, ids, scores in captured:
        new_x, new_ids, new_scores = inputs(len(x))
        x.copy_(new_x)
        ids.copy_(new_ids)
        scores.copy_(new_scores)
    for saved in (captured[1], captured[0], captured[2], captured[1]):
        graph, x, ids, scores = saved
        graph.replay()
        check(x, ids, scores)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_input_stage_reuses_compiled_routes_by_output_dtype():
    stager = Nvfp4InputStage(128, 2)
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    ids = torch.tensor([[1, 5], [7, 2], [3, 0]], device="cuda")
    scores = torch.rand(3, 2, device="cuda")
    q = torch.empty(16, 64, device="cuda", dtype=torch.uint8)
    sf = torch.empty(16, 8, device="cuda", dtype=torch.float8_e4m3fn)
    weights = torch.empty(16, 2, device="cuda")
    for dtype in (torch.int32, torch.int64, torch.int32):
        routes = torch.empty(16, 2, device="cuda", dtype=dtype)
        stager.run(x, scale, ids, scores, q, sf, routes, weights)
        torch.testing.assert_close(routes[:3], ids.to(dtype), atol=0, rtol=0)
        assert (routes[3:] == -1).all()
