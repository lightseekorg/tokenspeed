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

"""Exercise underfilled init windows with a compact single-CTA geometry."""

import importlib.util
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 9,
    reason="requires an NVIDIA SM90+ GPU",
)

deep_select = pytest.importorskip(
    "tokenspeed_kernel.ops.attention.dsa._cute_dsl.deep_select"
)


def _check_row(scores, end, topk, indices, values):
    if end <= topk:
        assert indices.tolist() == list(range(end)) + [-1] * (topk - end)
        torch.testing.assert_close(values[:end], scores[:end], atol=0, rtol=0)
        assert (values[end:] == -torch.inf).all()
        return
    assert indices.min() >= 0 and indices.max() < end
    assert indices.unique().numel() == topk
    torch.testing.assert_close(values, scores[indices.long()], atol=0, rtol=0)
    reference = torch.topk(scores[:end], topk).values
    torch.testing.assert_close(
        values.sort(descending=True).values, reference, atol=0, rtol=0
    )


@pytest.fixture(scope="module")
def compact_selector():
    # Isolate globals and compiled selectors from the production geometry.
    # A round consisting entirely of a padded tail can start with fewer than k
    # real entries while leaving earlier segments to be scanned.
    name = f"{deep_select.__name__}_compact_regression"
    spec = importlib.util.spec_from_file_location(name, deep_select.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.NUM_THREADS = 128
        module.NUM_WARPS = 4
        module.ROUND_ELEMS = 2048
        module.SEGS_PER_ROUND = 4
        module.TAIL_ELEMS = 2048
        module.TAIL_SEGS = 4
        module.RECONSTRUCT_THRESHOLD = 1024
        module.INCOMING_SLOTS = 3072
        module.NUM_STAGES = 2
        yield module
    finally:
        del sys.modules[name]


@pytest.mark.parametrize("end", [2049, 3069, 6138, 8191])
@pytest.mark.parametrize("distribution", ["ascending", "high_tail"])
def test_underfilled_init_keeps_lower_scoring_candidates(
    compact_selector, end, distribution
):
    scores = torch.full((1, 8192), 1e6, device="cuda")
    if distribution == "ascending":
        scores[0, :end] = torch.arange(end, device="cuda").float()
    else:
        scores[0, :end] = 1
        scores[0, end - end % 2048 : end] = 2
    ends = torch.tensor([end], dtype=torch.int32, device="cuda")
    indices, values = compact_selector.deepselect_topk(
        scores, ends, 2048, capacity=2048, cluster_size=1
    )
    torch.cuda.synchronize()
    _check_row(scores[0], end, 2048, indices[0], values[0])


def test_underfilled_init_replays_with_new_ends(compact_selector):
    scores = torch.arange(8192, device="cuda").float().repeat(4, 1)
    ends = torch.tensor([0, 2048, 3069, 8191], dtype=torch.int32, device="cuda")
    indices = torch.empty((4, 2048), dtype=torch.int32, device="cuda")
    values = torch.empty((4, 2048), device="cuda")

    def select():
        return compact_selector.deepselect_topk(
            scores,
            ends,
            2048,
            capacity=2048,
            cluster_size=1,
            indices=indices,
            values=values,
        )

    select()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    for lengths in ([0, 2048, 3069, 8191], [2049, 4096, 6138, 8192]):
        ends.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        for row, end in enumerate(lengths):
            _check_row(scores[row], end, 2048, indices[row], values[row])
