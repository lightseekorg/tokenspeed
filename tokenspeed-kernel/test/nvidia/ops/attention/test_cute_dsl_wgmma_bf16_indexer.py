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

"""Direct Hopper MQA scoring against an independent FP32 reference."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper, reason="requires SM90 WGMMA"
)


@pytest.fixture(autouse=True)
def fp32_reference_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


def _metadata(table, ends, query_group, device):
    """Build test inputs on the CPU; the production kernel only scores them."""
    pages, positions, lengths, causal = [], [], [], []
    for request, end in enumerate(ends):
        for offset in range(query_group):
            visible = max(0, end - query_group + 1 + offset)
            owned = [
                (page, column)
                for column, page in enumerate(table[request])
                if page > 0 and column * 64 < visible
            ]
            padding = len(table[request]) - len(owned)
            pages.append([page for page, _ in owned] + [0] * padding)
            positions.append([column for _, column in owned] + [-1] * padding)
            lengths.append(sum(min(64, visible - column * 64) for _, column in owned))
            causal.append(visible)
    return tuple(
        torch.tensor(value, dtype=torch.int32, device=device)
        for value in (pages, positions, lengths, causal)
    )


def _reference(
    q, weights, keys, pages, positions, lengths, causal, width, initial, recent
):
    output = torch.full((q.shape[0], width), -float("inf"), device=q.device)
    for row in range(q.shape[0]):
        count = min(int(lengths[row]), width)
        if not count:
            continue
        col = torch.arange(count, device=q.device)
        physical = pages[row, col // 64].long()
        logical = positions[row, col // 64].long() * 64 + col % 64
        source = physical * 64 + col % 64
        values = (q[row].float() @ keys[source].float().T).relu()
        values = (values * weights[row].float()[:, None]).sum(0) * 0.125
        forced = (logical < initial) | (logical >= causal[row] - recent)
        values = torch.where(forced, float("inf"), values)
        output[row, :count] = values
    return output


def _inputs(batch, query_group, stride, weight_dtype, *, heads, head_dim):
    torch.manual_seed(53)
    rows = batch * query_group
    # Offset Q and pad the other rows to exercise the launcher's normalization.
    storage = torch.randn(
        rows * heads * head_dim + 1, device="cuda", dtype=torch.bfloat16
    )
    storage.mul_(0.1)
    q = storage[1:].view(rows, heads, head_dim)
    weights = torch.randn(rows, heads + 1, device="cuda", dtype=weight_dtype)[:, :heads]
    keys = torch.randn((batch * 4 + 1) * 64, stride, device="cuda").bfloat16()[
        :, :head_dim
    ]
    keys.mul_(0.1)
    keys[:64].fill_(float("nan"))
    table = [[4 * b + 3, -1, 4 * b + 1, 4 * b + 4, 4 * b + 2] for b in range(batch)]
    return q, weights, keys, table


def _score(
    q,
    weights,
    keys,
    metadata,
    width,
    query_group,
    initial,
    recent,
    clear_padding,
    *,
    strategy
):
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer import (
        index_scores,
    )

    pages, positions, lengths, causal = metadata
    return index_scores(
        q,
        weights,
        keys,
        pages,
        positions,
        lengths,
        causal,
        causal,
        causal,
        scale=0.125,
        initial=initial,
        recent=recent,
        paged=True,
        page_size=64,
        width=width,
        clear_padding=clear_padding,
        query_group=query_group,
        strategy=strategy,
    )


@pytest.mark.parametrize(
    "batch,query_group,width", [(2, 1, 1025), (4, 8, 16384), (8, 8, 65536)]
)
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads,head_dim", [(8, 64), (16, 128), (32, 128), (64, 256)])
@pytest.mark.parametrize("strategy", ["normal", "swap_ab", "auto"])
def test_scores_and_graph_refresh(
    batch, query_group, width, weight_dtype, heads, head_dim, strategy
):
    q, weights, keys, table = _inputs(
        batch, query_group, head_dim + 16, weight_dtype, heads=heads, head_dim=head_dim
    )
    ends = [258 if b % 2 == 0 else 68 for b in range(batch)]
    metadata = _metadata(table, ends, query_group, "cuda")

    def run():
        return _score(
            q,
            weights,
            keys,
            metadata,
            width,
            query_group,
            2,
            3,
            True,
            strategy=strategy,
        )

    actual = run()
    expected = _reference(q, weights, keys, *metadata, width, 2, 3)
    torch.testing.assert_close(actual[:, :width], expected, atol=2e-6, rtol=2e-4)
    for _ in range(2):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for refresh in (
        [0] * batch,
        [2 if b % 2 else 130 for b in range(batch)],
        [258] * batch,
    ):
        table = [[row[2], row[1], row[0], *row[3:]] for row in table]
        updated = _metadata(table, refresh, query_group, "cuda")
        for target, value in zip(metadata, updated):
            target.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(q, weights, keys, *metadata, width, 2, 3)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual[:, :width], expected, atol=2e-6, rtol=2e-4)
        assert torch.isneginf(actual[:, width:]).all()


def test_uninitialized_padding_and_integer_metadata():
    q, weights, keys, table = _inputs(1, 8, 128, torch.float32, heads=16, head_dim=128)
    metadata = tuple(t.long() for t in _metadata(table, [258], 8, "cuda"))
    actual = _score(q, weights, keys, metadata, 193, 8, 0, 0, False, strategy="auto")
    expected = _reference(q, weights, keys, *metadata, 193, 0, 0)
    for row, count in enumerate(metadata[2].cpu().tolist()):
        end = min(count, 193)
        torch.testing.assert_close(
            actual[row, :end], expected[row, :end], atol=2e-6, rtol=2e-4
        )


@pytest.mark.parametrize("strategy", ["normal", "swap_ab"])
def test_runtime_geometry_reuses_binary(strategy):
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer_normal import (
        compile_normal_score,
    )
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer_swapAB import (
        compile_swap_ab_score,
    )

    factory = compile_normal_score if strategy == "normal" else compile_swap_ab_score

    def run(batch, width, stride):
        q, weights, keys, table = _inputs(
            batch, 8, stride, torch.float32, heads=32, head_dim=128
        )
        table = [row + [0] * (batch % 3) for row in table]
        metadata = _metadata(table, [258] * batch, 8, "cuda")
        return _score(
            q, weights, keys, metadata, width, 8, 0, 0, True, strategy=strategy
        )

    run(4, 16384, 128)
    misses = factory.cache_info().misses
    run(5, 16388, 144)
    torch.cuda.synchronize()
    assert factory.cache_info().misses == misses


@pytest.mark.parametrize("rows,key_rows,width", [(0, 128, 64), (1, 0, 67), (1, 128, 0)])
def test_empty_inputs(rows, key_rows, width):
    q = torch.empty(rows, 16, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(rows, 16, device="cuda")
    keys = torch.empty(key_rows, 128, device="cuda", dtype=torch.bfloat16)
    pages = torch.zeros(rows, 1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    result = _score(
        q,
        weights,
        keys,
        (pages, pages, lengths, lengths),
        width,
        1,
        0,
        0,
        True,
        strategy="auto",
    )
    assert result.shape == (rows, (width + 3) // 4 * 4)
    assert result.dtype == torch.float32
    assert torch.isneginf(result).all()


@pytest.mark.parametrize("query_group", [0, -1])
def test_unsupported_query_group(query_group):
    q, weights, keys, table = _inputs(1, 1, 128, torch.float32, heads=16, head_dim=128)
    metadata = _metadata(table, [68], 1, "cuda")
    with pytest.raises(ValueError, match="request width"):
        _score(
            q, weights, keys, metadata, 128, query_group, 0, 0, True, strategy="auto"
        )


@pytest.mark.parametrize("heads,head_dim", [(8, 192), (24, 128), (128, 64), (256, 256)])
def test_general_head_geometry(heads, head_dim):
    q, weights, keys, table = _inputs(
        1, 8, head_dim + 16, torch.float32, heads=heads, head_dim=head_dim
    )
    metadata = _metadata(table, [258], 8, "cuda")
    actual = _score(q, weights, keys, metadata, 65536, 8, 0, 0, True, strategy="auto")
    expected = _reference(q, weights, keys, *metadata, 65536, 0, 0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("request_width", [2, 3, 4, 6, 7, 9, 32])
def test_request_query_width(request_width):
    q, weights, keys, table = _inputs(
        2, request_width, 128, torch.float32, heads=32, head_dim=128
    )
    metadata = _metadata(table, [258, 68], request_width, "cuda")
    actual = _score(
        q, weights, keys, metadata, 65536, request_width, 0, 0, True, strategy="auto"
    )
    expected = _reference(q, weights, keys, *metadata, 65536, 0, 0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("heads,head_dim", [(4, 128), (264, 128), (16, 96), (16, 320)])
def test_unsupported_head_geometry(heads, head_dim):
    q, weights, keys, table = _inputs(
        1, 1, head_dim, torch.float32, heads=heads, head_dim=head_dim
    )
    metadata = _metadata(table, [68], 1, "cuda")
    with pytest.raises(ValueError, match="multiple of"):
        _score(q, weights, keys, metadata, 128, 1, 0, 0, True, strategy="auto")


@pytest.mark.parametrize("operand", ["q", "keys"])
def test_reject_non_bf16_operands(operand):
    q, weights, keys, table = _inputs(1, 1, 128, torch.float32, heads=32, head_dim=128)
    metadata = _metadata(table, [68], 1, "cuda")
    if operand == "q":
        q = q.float()
    else:
        keys = keys.float()
    with pytest.raises(ValueError, match="dtype bfloat16"):
        _score(q, weights, keys, metadata, 128, 1, 0, 0, True, strategy="auto")


@pytest.mark.parametrize("strategy", ["normal", "swap_ab", "auto"])
@pytest.mark.parametrize("queries", [1, 8, 17, 64])
def test_prefill_workspace(strategy, queries):
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer import index_scores

    torch.manual_seed(53)
    rows = 2 * queries
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(rows, 32, device="cuda")
    keys = torch.randn(768, 144, device="cuda").bfloat16()[:, :128] * 0.1
    starts = torch.tensor(
        [17, 400], device="cuda", dtype=torch.int64
    ).repeat_interleave(queries)
    lengths = torch.arange(
        258 - queries + 1, 259, device="cuda", dtype=torch.int32
    ).repeat(2)
    valid = torch.ones(768, device="cuda", dtype=torch.bool)
    valid[48] = False
    valid[144] = False
    valid[500] = False
    pages = torch.zeros(rows, 1, device="cuda", dtype=torch.int32)
    out = index_scores(
        q,
        weights,
        keys,
        pages,
        pages,
        lengths,
        lengths,
        starts,
        valid,
        scale=0.125,
        initial=2,
        recent=3,
        paged=False,
        page_size=64,
        width=259,
        clear_padding=True,
        query_group=queries,
        strategy=strategy,
    )
    expected = torch.full_like(out, -float("inf"))
    for row in range(rows):
        count = int(lengths[row])
        source = starts[row] + torch.arange(count, device="cuda")
        values = (q[row].float() @ keys[source].float().T).relu()
        values = (values * weights[row, :, None]).sum(0) * 0.125
        forced = (torch.arange(count, device="cuda") < 2) | (
            torch.arange(count, device="cuda") >= count - 3
        )
        values = torch.where(forced, float("inf"), values)
        expected[row, :count] = torch.where(valid[source], values, -float("inf"))
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize(
    "rows,queries,heads,paged,expected",
    [
        (8, 1, 32, True, "swap_ab"),
        (64, 8, 32, True, "normal"),
        (1024, 1024, 32, False, "normal"),
        (8, 8, 8, True, "normal"),
        (9, 8, 32, True, "swap_ab"),
    ],
)
def test_static_orientation_selection(rows, queries, heads, paged, expected):
    from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer import (
        select_score_config,
    )

    orientation, group, split = select_score_config(
        rows, 32768, queries, heads, paged=paged, strategy="auto"
    )
    assert orientation == expected
    assert rows % group == 0 and queries % group == 0
    assert 1 <= split <= 512
    for forced in ("normal", "swap_ab"):
        selected, _, _ = select_score_config(
            rows, 32768, queries, heads, paged=paged, strategy=forced
        )
        assert selected == forced
