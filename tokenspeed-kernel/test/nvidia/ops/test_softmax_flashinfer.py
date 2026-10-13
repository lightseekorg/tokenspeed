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

"""PDL coverage for the vendored CUDA softmax's per-row temperature input."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.flashinfer_softmax import softmax

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper_plus, reason="PDL requires NVIDIA SM90+"
)


@triton.jit
def _publish_after_release(source, target, rows, BLOCK: tl.constexpr):
    # Release the softmax first, then write the temperatures after a delay.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
    )
    now = start
    while now - start < 3000000:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
        )
    offsets = tl.arange(0, BLOCK)
    mask = offsets < rows
    tl.store(target + offsets, tl.load(source + offsets, mask=mask), mask=mask)


# Map-reduce needs rows <= 128 and vocab >= 24576. Otherwise the fused kernel
# caches the row in shared memory when it fits.
@pytest.mark.parametrize(
    "rows,vocab",
    [
        pytest.param(4, 4096, id="fused-cached"),
        pytest.param(129, 65536, id="fused-uncached"),
        pytest.param(4, 32768, id="map-reduce"),
    ],
)
@pytest.mark.parametrize("graph_replay", [False, True])
def test_softmax_waits_for_temperature(
    rows: int, vocab: int, graph_replay: bool, device: str
) -> None:
    logits = torch.randn(rows, vocab, device=device)
    published = torch.linspace(0.5, 1.5, rows, device=device)
    expected = torch.softmax(logits / published[:, None], dim=-1)
    temperature = torch.empty_like(published)

    def run() -> torch.Tensor:
        temperature.fill_(float("nan"))  # Visible only to a read before the wait.
        _publish_after_release[(1,)](
            published,
            temperature,
            rows,
            BLOCK=triton.next_power_of_2(rows),
            launch_pdl=True,
        )
        return softmax(logits, temperature=temperature, enable_pdl=True)

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        run()  # Compile and load both kernels before the checked run.
        if graph_replay:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = run()
            graph.replay()
        else:
            actual = run()
    stream.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
