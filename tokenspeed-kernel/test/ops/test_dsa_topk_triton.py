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

"""Portable DSA top-k: stable indices, runtime-width reuse, and graph replay."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa._triton import topk as dsa_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA or ROCm"
)
requires_pdl = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 9
    ),
    reason="PDL requires NVIDIA SM90+",
)


def _logits(cols: int, topk: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randint(-16, 17, (4, cols), generator=generator).float() / 8
    logits[1, torch.rand(cols, generator=generator) < 0.4] = -torch.inf
    logits[2].fill_(-torch.inf)
    # Fewer than k valid entries, all at the end of the last input tile.
    logits[3, : max(0, cols - topk // 2)] = -torch.inf
    if cols:
        # Equal maxima in distant tiles must keep the lower index first.
        logits[0, 0] = logits[0, -1] = 4
    return logits


def _reference(logits: torch.Tensor, topk: int) -> torch.Tensor:
    indices = logits.argsort(dim=1, descending=True, stable=True)[:, :topk]
    indices = indices.masked_fill(logits.gather(1, indices) == -torch.inf, -1)
    out = torch.full((logits.shape[0], topk), -1, dtype=torch.int32)
    out[:, : indices.shape[1]] = indices.to(torch.int32)
    return out


@pytest.mark.parametrize(
    "cols,topk",
    [
        (0, 128),
        (0, 2048),
        (17, 128),
        (127, 2048),
        (8192, 128),
        (8192, 2048),
        (16384, 2048),
        (32768, 2048),
        (49152, 128),
        (49152, 2048),
        (65535, 2048),
        (65536, 128),
        (65536, 2048),
        (81920, 2048),
    ],
)
def test_dsa_topk_stable_indices(cols: int, topk: int) -> None:
    logits = _logits(cols, topk, seed=0)
    actual = dsa_topk.triton_topk_from_logits(logits.cuda(), topk, enable_pdl=False)
    torch.testing.assert_close(actual.cpu(), _reference(logits, topk), rtol=0, atol=0)


@pytest.mark.parametrize(
    "cols,enable_pdl",
    [
        (49152, False),
        (65536, False),
        pytest.param(49152, True, marks=requires_pdl),
    ],
)
def test_dsa_topk_graph_replay(cols: int, enable_pdl: bool) -> None:
    topk = 2048
    initial = _logits(cols, topk, seed=1)
    changed = _logits(cols, topk, seed=2)
    expected = [_reference(host, topk) for host in (initial, changed)]
    assert not torch.equal(expected[0], expected[1])
    logits = initial.cuda()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup = dsa_topk.triton_topk_from_logits(logits, topk, enable_pdl=enable_pdl)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(warmup.cpu(), expected[0], rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = dsa_topk.triton_topk_from_logits(logits, topk, enable_pdl=enable_pdl)

    for host, reference in zip((initial, changed), expected):
        logits.copy_(host)
        actual.fill_(-99)
        graph.replay()
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


def test_dsa_topk_reuses_compiled_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aligned widths change loop bounds, not the compiled specialization."""
    kernel = dsa_topk._dsa_logits_topk_kernel
    original_run = kernel.run
    compiled = []

    def record_run(*args, **kwargs):
        assert kwargs["BLOCK_N"] == kwargs["topk"] == 2048
        result = original_run(*args, **kwargs)
        compiled.append(result)
        return result

    monkeypatch.setattr(kernel, "run", record_run)
    # Same alignment class, but four distinct padded widths below the radix path.
    widths = (8192, 16384, 32768, 49152)
    for cols in widths:
        logits = _logits(cols, 2048, seed=3)
        actual = dsa_topk.triton_topk_from_logits(logits.cuda(), 2048, enable_pdl=False)
        torch.testing.assert_close(
            actual.cpu(), _reference(logits, 2048), rtol=0, atol=0
        )

    assert len(compiled) == len(widths)
    assert compiled[0] is not None
    assert all(
        result is compiled[0] for result in compiled
    ), "Aligned widths with fixed BLOCK_N/topk must reuse one compiled kernel"
