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

"""The CuTe DSL DeepSelect-style selector against torch.topk on Hopper."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or not current_platform().is_hopper,
    reason="requires an NVIDIA Hopper GPU",
)

deep_select = pytest.importorskip(
    "tokenspeed_kernel.ops.attention.dsa._cute_dsl.deep_select"
)


def _scores(rows, width, distribution, generator):
    device = torch.device("cuda")
    if distribution == "randn":
        values = torch.randn((rows, width), device=device, generator=generator)
    elif distribution == "ties":
        values = torch.randint(0, 8, (rows, width), device=device, generator=generator)
        values = values.float()
    elif distribution == "ascending":
        values = torch.arange(width, device=device).float().repeat(rows, 1)
    elif distribution == "descending":
        values = -torch.arange(width, device=device).float().repeat(rows, 1)
    elif distribution == "masked":
        values = torch.randn((rows, width), device=device, generator=generator)
        values[:, ::3] = -torch.inf
    else:
        raise ValueError(distribution)
    # Rows are padded to the load granularity with NaN, which must never leak.
    padded = -(-width // deep_select.SCORE_ALIGNMENT) * deep_select.SCORE_ALIGNMENT
    storage = torch.full((rows, padded), torch.nan, device=device)
    storage[:, :width] = values
    return storage[:, :width]


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
    chosen = values.sort(descending=True).values
    torch.testing.assert_close(chosen, reference, atol=0, rtol=0, equal_nan=True)


def _select(scores, ends, topk, cluster_size, **kwargs):
    return deep_select.deepselect_topk(
        scores,
        ends,
        topk,
        capacity=deep_select.capacity_for(topk),
        cluster_size=cluster_size,
        **kwargs,
    )


@pytest.mark.parametrize("cluster_size", [1, 8])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize(
    "width", [1000, 4096, 8192, 8193, 12000, 40000, 65536, 131072 + 777]
)
@pytest.mark.parametrize(
    "distribution", ["randn", "ties", "ascending", "descending", "masked"]
)
def test_selects_the_top_values(width, topk, cluster_size, distribution):
    if cluster_size not in deep_select.cluster_sizes_for(
        deep_select.capacity_for(topk)
    ):
        pytest.skip("cluster does not fit the candidate buffer for this topk")
    generator = torch.Generator(device="cuda").manual_seed(width * 7 + topk)
    rows = 3
    scores = _scores(rows, width, distribution, generator)
    ends = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices, values = _select(scores, ends, topk, cluster_size)
    torch.cuda.synchronize()
    for row in range(rows):
        _check_row(scores[row], width, topk, indices[row], values[row])


@pytest.mark.parametrize("cluster_size", [1, 2, 4, 8])
def test_ragged_ends_and_shortcut_rows(cluster_size):
    generator = torch.Generator(device="cuda").manual_seed(11)
    width = 131072
    scores = _scores(6, width, "randn", generator)
    # Below topk, exactly topk, one above, one init window, tail-only splits,
    # and the full row.
    ends = torch.tensor(
        [100, 512, 513, 8192, 8193, width], dtype=torch.int32, device="cuda"
    )
    indices, values = _select(scores, ends, 512, cluster_size)
    torch.cuda.synchronize()
    for row in range(6):
        _check_row(scores[row], int(ends[row]), 512, indices[row], values[row])


@pytest.mark.parametrize("cluster_size", [1, 8])
def test_nan_entries_are_never_selected(cluster_size):
    width = 65536
    scores = _scores(2, width, "randn", torch.Generator(device="cuda").manual_seed(3))
    # Row 0 spreads NaN over the scanned segments (5, 15000) and the init
    # window's tail, where a positive quiet NaN would sort above +inf if its
    # key were not sanitized.
    nan_positions = (5, 15000, width - 1, width - 4096)
    for position in nan_positions:
        scores[0, position] = torch.nan
    # Row 1 holds nothing but NaN, so every slot must be a placeholder.
    scores[1] = torch.nan
    ends = torch.full((2,), width, dtype=torch.int32, device="cuda")
    indices, values = _select(scores, ends, 512, cluster_size)
    torch.cuda.synchronize()
    picked = indices[0]
    assert not any(position in picked.tolist() for position in nan_positions)
    valid = picked[picked >= 0]
    assert not torch.isnan(scores[0, valid.long()]).any()
    assert not torch.isnan(values[0]).any()
    assert (values[0][picked < 0] == -torch.inf).all()
    assert (indices[1] == -1).all()
    assert (values[1] == -torch.inf).all()


def test_shortcut_rows_replace_nan_with_placeholders():
    scores = _scores(1, 4096, "randn", torch.Generator(device="cuda").manual_seed(7))
    scores[0, 7] = torch.nan
    ends = torch.tensor([100], dtype=torch.int32, device="cuda")
    indices, values = _select(scores, ends, 512, 1)
    torch.cuda.synchronize()
    expected = list(range(100))
    expected[7] = -1
    assert indices[0].tolist() == expected + [-1] * 412
    assert values[0, 7] == -torch.inf
    assert (values[0, 100:] == -torch.inf).all()
    kept = [i for i in range(100) if i != 7]
    torch.testing.assert_close(values[0, kept], scores[0, kept], atol=0, rtol=0)


def test_warmup_serves_an_index_less_device():
    deep_select.warmup((512,), torch.device("cuda"))
    misses_after_warmup = deep_select.compiled_selector.cache_info().misses
    scores = _scores(1, 8192, "randn", torch.Generator(device="cuda").manual_seed(9))
    ends = torch.full((1,), 8192, dtype=torch.int32, device="cuda")
    _select(scores, ends, 512, 1)
    assert deep_select.compiled_selector.cache_info().misses == misses_after_warmup


def test_replays_inside_a_cuda_graph():
    generator = torch.Generator(device="cuda").manual_seed(5)
    scores = _scores(4, 65536, "randn", generator)
    ends = torch.tensor([65536, 30000, 700, 65536], dtype=torch.int32, device="cuda")
    out = torch.empty((4, 512), dtype=torch.int32, device="cuda")
    values = torch.empty((4, 512), dtype=torch.float32, device="cuda")
    deep_select.warmup((512,), scores.device)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _select(scores, ends, 512, 4, indices=out, values=values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _select(scores, ends, 512, 4, indices=out, values=values)
    for _ in range(2):
        out.fill_(-7)
        scores.copy_(_scores(4, 65536, "randn", generator))
        graph.replay()
        torch.cuda.synchronize()
        for row in range(4):
            _check_row(scores[row], int(ends[row]), 512, out[row], values[row])


def test_cluster_choice_keeps_every_cluster_resident():
    choose = deep_select.choose_cluster_size
    assert choose(1, 8192, 512, 78) == 1
    assert choose(1, 131072, 512, 78) == 8
    assert choose(6, 131072, 512, 78) == 8
    assert choose(7, 131072, 512, 78) == 4
    assert choose(12, 131072, 512, 78) == 4
    assert choose(13, 131072, 512, 78) == 2
    assert choose(32, 131072, 512, 78) == 2
    assert choose(33, 131072, 512, 78) == 1
    # 2048 survivors per CTA only leave room to gather three peers.
    assert choose(1, 131072, 2048, 78) == 4


def test_rejects_unaligned_rows():
    scores = torch.randn((2, 4098), device="cuda")[:, :4097]
    ends = torch.full((2,), 4097, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="stride"):
        _select(scores, ends, 512, 1)
    aligned = torch.randn((2, 4096), device="cuda")
    full = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="cluster_size"):
        _select(aligned, full, 2048, 8)
    with pytest.raises(ValueError, match="capacity"):
        deep_select.deepselect_topk(aligned, full, 600, capacity=512, cluster_size=1)
