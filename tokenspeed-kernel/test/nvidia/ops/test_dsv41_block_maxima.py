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
from tokenspeed_kernel.numerics.torch.attention.dsv41 import (
    torch_dsv41_block_maxima,
)
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as impl
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _block_maxima_kernel,
    block_maxima,
    clean_logits,
    clean_logits_for_selection,
    prepare_native_block_scores,
    prepare_scores,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)


# The downstream selector is registered and imported only on Hopper.
requires_hopper_selector = pytest.mark.skipif(
    not impl.platform.is_hopper,
    reason="requires the Hopper DeepSelect path",
)


@pytest.mark.parametrize("width", [1, 7, 8, 9, 1023, 1024, 1025, 16384, 65537])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_block_maxima(width, strided, dtype):
    torch.manual_seed(37)
    x = torch.randn((9, width * (2 if strided else 1)), device="cuda", dtype=dtype)
    if strided:
        x = x[:, ::2]
    visible = torch.tensor(
        [0, 1, 7, 8, 9, width, width + 8, -1, 0], device="cuda", dtype=torch.int32
    )
    x[0].fill_(-float("inf"))
    x[1, 0] = float("nan")
    x[2, -1] = float("inf")
    x[3, -1] = float("nan")
    x[4].fill_(-float("inf"))
    torch.testing.assert_close(
        block_maxima(x, visible).float(),
        torch_dsv41_block_maxima(x.float(), visible).float(),
        atol=0,
        rtol=0,
        equal_nan=True,
    )


def test_graph_refresh():
    x = torch.randn((8, 1025), device="cuda")
    visible = torch.full((8,), 1025, device="cuda", dtype=torch.int32)
    block_maxima(x, visible)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = block_maxima(x, visible)
    for length in [0, 1, 8, 1025]:
        x.normal_()
        visible.fill_(length)
        graph.replay()
        torch.testing.assert_close(
            output, torch_dsv41_block_maxima(x, visible), atol=0, rtol=0
        )


def test_empty_rows():
    x = torch.empty((0, 17), device="cuda")
    visible = torch.empty((0,), device="cuda", dtype=torch.int32)
    assert block_maxima(x, visible).shape == (0, 3)


def test_capacity_does_not_recompile():
    def run(rows, width):
        x = torch.randn((rows, width), device="cuda")
        visible = torch.full((rows,), width, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(
            block_maxima(x, visible),
            torch_dsv41_block_maxima(x, visible),
            atol=0,
            rtol=0,
        )

    for width in [1024, 1025, 1032]:
        run(8, width)
    with assert_no_triton_compile(_block_maxima_kernel):
        for rows, width in [(1, 2048), (17, 2049), (32, 2056), (8, 16384), (3, 16385)]:
            run(rows, width)


def select(logits, visible, bounded):
    maxima, block_ends = impl.block_maxima_with_lengths(logits, visible)
    k = min(2048, maxima.shape[1])
    ids = torch.empty((logits.shape[0], k), device="cuda", dtype=torch.int32)
    lengths = torch.empty((logits.shape[0],), device="cuda", dtype=torch.int32)
    ends = block_ends if bounded else None
    impl._select(maxima, k, 2048, ids, lengths, ends=ends)
    return ids, lengths


def canonical(ids):
    return ids.sort(dim=1).values


@pytest.mark.parametrize(
    "visible_values",
    [[0, 1, 1001, 1330], [7, 8, 9, 16384], [16385, 20000, 65535, 65536]],
)
@requires_hopper_selector
def test_candidate_sets_and_downstream(visible_values):
    torch.manual_seed(37)
    width = 1048640
    visible = torch.tensor(visible_values, device="cuda", dtype=torch.int32)
    logits = torch.randn((4, width), device="cuda")
    logits.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= visible[:, None], -float("inf")
    )
    # Invalid pages can leave holes inside the visible prefix.
    logits[:, 64:128] = -float("inf")
    old, old_lengths = select(logits, visible, False)
    new, new_lengths = select(logits, visible, True)
    torch.testing.assert_close(old_lengths, new_lengths, atol=0, rtol=0)
    torch.testing.assert_close(canonical(old), canonical(new), atol=0, rtol=0)
    results = []
    # Feed both candidate orders through the actual candidate gather and TopK.
    for candidates in (old, new):
        scores = impl.candidate_scores(logits, candidates)
        ids = torch.empty((4, 512), device="cuda", dtype=torch.int32)
        lengths = torch.empty((4,), device="cuda", dtype=torch.int32)
        impl._select(scores, 512, 512, ids, lengths, candidates)
        results.append((canonical(ids), lengths))
    for x, y in zip(*results):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


@requires_hopper_selector
def test_bounded_selection_graph_refresh():
    width = 1048640
    torch.manual_seed(73)
    original = torch.randn((2, width), device="cuda")
    logits = original.clone()
    visible = torch.tensor([0, 1001], device="cuda", dtype=torch.int32)
    columns = torch.arange(width, device="cuda")[None, :]
    logits.masked_fill_(columns >= visible[:, None], -float("inf"))
    select(logits, visible, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = select(logits, visible, True)
    for values in [[1330, 8], [0, 0], [20000, 16385]]:
        visible.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        logits.copy_(original).masked_fill_(columns >= visible[:, None], -float("inf"))
        graph.replay()
        expected = select(logits, visible, False)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        torch.testing.assert_close(
            canonical(actual[0]), canonical(expected[0]), atol=0, rtol=0
        )


def assert_cleaned_prefixes(scores, visible, table, num_pages, actual):
    expected = impl.clean_logits(scores, visible, table, num_pages, scores.shape[1], 64)
    maxima, ends = impl.block_maxima_with_lengths(expected.float(), visible)
    assert actual[0].dtype == scores.dtype
    assert actual[1].dtype == torch.float32
    torch.testing.assert_close(actual[2], ends, atol=0, rtol=0)
    for row, length in enumerate(visible.cpu().tolist()):
        length = max(0, min(length, scores.shape[1]))
        for old, new, count in (
            (expected, actual[0], length),
            (maxima, actual[1], (length + 7) // 8),
        ):
            torch.testing.assert_close(
                old[row, :count].float(),
                new[row, :count].float(),
                atol=0,
                rtol=0,
                equal_nan=True,
            )
    return expected


@pytest.mark.parametrize("table_layout", ["contiguous", "strided", "broadcast"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clean_block_prefixes(table_layout, dtype):
    width = 4103
    torch.manual_seed(37)
    scores = torch.randn((8, width + 13), device="cuda", dtype=dtype)[:, :width]
    visible = torch.tensor(
        [-1, 0, 1, 7, 8, 9, 1330, width + 64], device="cuda", dtype=torch.int32
    ).repeat_interleave(2)[::2]
    pages = (width + 63) // 64
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(8, 1)
    table[:, 2] = -1
    table[:, 4] = pages
    if table_layout == "strided":
        table = table.repeat_interleave(2, dim=1)[:, ::2]
    elif table_layout == "broadcast":
        table = table[:1].expand(8, -1)
    scores[3, 0] = float("nan")
    scores[7, -1] = float("nan")
    scores[6, 8] = float("inf")
    actual = impl.clean_block_maxima(scores, visible, table, pages)
    assert_cleaned_prefixes(scores, visible, table, pages, actual)


@pytest.mark.parametrize("cleaner", [clean_logits, clean_logits_for_selection])
def test_clean_logits_bf16_prefixes(cleaner):
    width = 2053
    torch.manual_seed(37)
    values = torch.randn((4, width), device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([0, 1, 1330, width], device="cuda", dtype=torch.int32)
    pages = (width + 63) // 64
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(4, 1)
    table[:, 2] = -1
    table[:, 4] = pages

    alignment = 16 if cleaner is clean_logits_for_selection else 1
    if cleaner is clean_logits_for_selection:
        actual = cleaner(values, lengths, table, pages, width, 64, alignment)
    else:
        actual = cleaner(values, lengths, table, pages, width, 64)
    assert actual.dtype == values.dtype
    expected_width = (
        (width + alignment - 1) // alignment * alignment
        if cleaner is clean_logits_for_selection
        else width
    )
    assert actual.shape == (4, expected_width)

    expected = values.float().clone()
    for row, length in enumerate(lengths.cpu().tolist()):
        length = max(0, min(length, width))
        expected[row, length:] = -float("inf")
        for page in range((length + 63) // 64):
            page_id = table[row, page].item()
            if page_id < 0 or page_id >= pages:
                start = page * 64
                end = min(start + 64, length)
                expected[row, start:end] = -float("inf")
        initialized = width
        if cleaner is clean_logits_for_selection:
            initialized = min(width, ((length + 1023) // 1024) * 1024)
        torch.testing.assert_close(
            actual[row, :length].float(),
            expected[row, :length],
            atol=0,
            rtol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            actual[row, length:initialized].float(),
            expected[row, length:initialized],
            atol=0,
            rtol=0,
            equal_nan=True,
        )
    if cleaner is clean_logits_for_selection:
        torch.testing.assert_close(
            actual[3, width:].float(),
            torch.full_like(actual[3, width:], -float("inf"), dtype=torch.float32),
            atol=0,
            rtol=0,
        )


def test_clean_block_bf16_tied_maxima():
    # These distinct FP32 values round to the same BF16 score and tie across blocks.
    scores = torch.full((1, 24), -float("inf"), device="cuda", dtype=torch.bfloat16)
    scores[0, 0] = 1.0
    scores[0, 8] = 1.003
    visible = torch.tensor([17], device="cuda", dtype=torch.int32)
    table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    cleaned, maxima, ends = impl.clean_block_maxima(scores, visible, table, 1)

    assert cleaned.dtype == torch.bfloat16
    assert maxima.dtype == torch.float32
    torch.testing.assert_close(
        ends, torch.tensor([3], device="cuda", dtype=torch.int32)
    )
    torch.testing.assert_close(
        maxima[0],
        torch.tensor([1.0, 1.0, -float("inf")], device="cuda", dtype=torch.float32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(cleaned[0, :17].float(), scores[0, :17].float())


def native_selection(scores, visible, block_stats):
    rows = scores.shape[0]
    out = (
        torch.empty((rows, 512), device="cuda", dtype=torch.int32),
        torch.empty((rows,), device="cuda", dtype=torch.int32),
        torch.empty((rows, 2048), device="cuda", dtype=torch.int32),
        torch.empty((rows,), device="cuda", dtype=torch.int32),
    )
    impl._native_select(scores, visible, None, 512, 2048, out, None, block_stats)
    return out


@requires_hopper_selector
def test_clean_block_selection_graph_refresh():
    width = 65536
    torch.manual_seed(73)
    scores = torch.randn((4, width), device="cuda")
    visible = torch.tensor([0, 1, 9, 1330], device="cuda", dtype=torch.int32)
    pages = (width + 63) // 64
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(4, 1)
    table[:, 2] = -1

    def run():
        cleaned, maxima, ends = impl.clean_block_maxima(scores, visible, table, pages)
        return cleaned, maxima, ends, native_selection(cleaned, visible, (maxima, ends))

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for lengths in ([width] * 4, [0] * 4, [1, 1330, 9, 32768]):
        visible.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        scores.normal_()
        table[:, 3] = -1 if lengths[0] == 0 else 3
        # Poison unwritten tails to prove that selection obeys device lengths.
        actual[0].fill_(float("inf"))
        actual[1].fill_(float("inf"))
        graph.replay()
        expected = assert_cleaned_prefixes(scores, visible, table, pages, actual[:3])
        selected = native_selection(expected, visible, None)
        for i, (old, new) in enumerate(zip(selected, actual[3])):
            if i % 2 == 0:
                old, new = canonical(old), canonical(new)
            torch.testing.assert_close(old, new, atol=0, rtol=0)


def test_clean_block_empty_rows():
    actual = impl.clean_block_maxima(
        torch.empty((0, 17), device="cuda"),
        torch.empty((0,), device="cuda", dtype=torch.int32),
        torch.empty((0, 1), device="cuda", dtype=torch.int32),
        1,
    )
    assert [x.shape for x in actual] == [(0, 17), (0, 3), (0,)]


def test_clean_block_capacity_does_not_recompile():
    from tokenspeed_kernel.ops.attention.dsv41.triton import _clean_block_maxima_kernel

    def run(rows, width):
        pages = (width + 63) // 64
        scores = torch.randn((rows, width), device="cuda")
        visible = torch.full((rows,), width, device="cuda", dtype=torch.int32)
        table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
        actual = impl.clean_block_maxima(scores, visible, table, pages)
        assert_cleaned_prefixes(scores, visible, table, pages, actual)

    for width in (1024, 1025, 1032):
        run(8, width)
    with assert_no_triton_compile(_clean_block_maxima_kernel):
        for rows, width in ((1, 2048), (17, 2049), (3, 16385), (6, 1048640)):
            run(rows, width)


@pytest.mark.parametrize("width", [35, 32769])
def test_native_block_scores_are_aligned_and_pin_newest(width):
    block_size, alignment = 8, 16
    logits = torch.randn((4, width), device="cuda")
    lengths = torch.tensor([0, 9, 17, width], dtype=torch.int32, device="cuda")
    logits[1, :8] = float("nan")
    logits[2, 0] = float("nan")
    logits[1, 8] = 0
    logits[3, width - 1] = float("inf")
    logits[3, 24:32] = -1000

    actual, ends = prepare_native_block_scores(logits, lengths, block_size, alignment)
    columns = torch.arange(width, device="cuda")[None, :]
    reference = torch_dsv41_block_maxima(
        logits.masked_fill(columns >= lengths[:, None], -torch.inf), lengths
    )
    expected = torch.full_like(actual, -torch.inf)
    expected[:, : reference.shape[1]] = reference
    expected_ends = (lengths.clamp(0, width) + block_size - 1) // block_size
    assert actual.shape == (4, ((width + 7) // 8 + 15) // 16 * 16)
    torch.testing.assert_close(ends, expected_ends, rtol=0, atol=0)
    for row, end in enumerate(ends.tolist()):
        initialized = min(
            actual.shape[1], (end + alignment - 1) // alignment * alignment
        )
        torch.testing.assert_close(
            actual[row, :initialized],
            expected[row, :initialized],
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def test_native_block_scores_graph_replay_refreshes_partial_rows():
    block_size, alignment = 8, 16
    logits = torch.randn((3, 35), device="cuda")
    lengths = torch.tensor([0, 9, 17], dtype=torch.int32, device="cuda")
    columns = torch.arange(logits.shape[1], device="cuda")[None, :]
    logits.masked_fill_(columns >= lengths[:, None], float("nan"))
    prepare_native_block_scores(logits, lengths, block_size, alignment)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = prepare_native_block_scores(logits, lengths, block_size, alignment)

    for visible in ([35, 0, 1], [8, 17, 33], [0, 9, 35]):
        lengths.copy_(torch.tensor(visible, dtype=torch.int32, device="cuda"))
        logits.normal_()
        logits.masked_fill_(columns >= lengths[:, None], float("nan"))
        for output in actual:
            output.fill_(123456)
        graph.replay()

        expected, expected_ends = prepare_scores(
            logits, lengths, block_size, alignment, True
        )
        torch.testing.assert_close(actual[1], expected_ends, rtol=0, atol=0)
        for row, end in enumerate(actual[1].tolist()):
            initialized = min(
                actual[0].shape[1],
                (end + alignment - 1) // alignment * alignment,
            )
            torch.testing.assert_close(
                actual[0][row, :initialized],
                expected[row, :initialized],
                rtol=0,
                atol=0,
                equal_nan=True,
            )


def test_native_block_scores_width_changes_do_not_recompile():
    from tokenspeed_kernel.ops.attention.dsv41.triton import _reduce_blocks

    def run(width):
        values = torch.randn((3, width), dtype=torch.float32, device="cuda")
        ends = torch.tensor([0, 9, width], dtype=torch.int32, device="cuda")
        actual, counts = prepare_native_block_scores(values, ends, 8, 16)
        expected, expected_counts = prepare_scores(values, ends, 8, 16, True)
        torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
        for row, count in enumerate(counts.tolist()):
            initialized = (count + 15) // 16 * 16
            torch.testing.assert_close(
                actual[row, :initialized], expected[row, :initialized], rtol=0, atol=0
            )

    run(35)
    with assert_no_triton_compile(_reduce_blocks):
        for width in (1025, 32769, 65537):
            run(width)


def test_native_block_scores_reach_rows_past_int32_offsets():
    # The last of 2049 BF16 rows starts past 2**31 elements. Only that row is
    # live, so the reduction reads and writes nothing but the far end.
    rows, width, block_size, alignment = 2049, (1 << 20) + 512, 8, 256
    logits = torch.empty((rows, width), device="cuda", dtype=torch.bfloat16)
    assert (rows - 1) * logits.stride(0) >= 2**31
    logits[-1].normal_()
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    lengths[-1] = (1 << 20) + 37

    actual, ends = prepare_native_block_scores(logits, lengths, block_size, alignment)

    columns = torch.arange(width, device="cuda")
    live = logits[-1].float().masked_fill(columns >= lengths[-1], -torch.inf)
    expected = torch_dsv41_block_maxima(live[None, :], lengths[-1:])[0]
    count = ((1 << 20) + 37 + block_size - 1) // block_size
    initialized = (count + alignment - 1) // alignment * alignment
    expected_ends = torch.zeros_like(ends)
    expected_ends[-1] = count
    torch.testing.assert_close(ends, expected_ends, rtol=0, atol=0)
    torch.testing.assert_close(actual[-1, :count], expected[:count], rtol=0, atol=0)
    padding = actual[-1, count:initialized]
    torch.testing.assert_close(
        padding, torch.full_like(padding, -torch.inf), rtol=0, atol=0
    )


def test_selection_cleaner_reaches_rows_past_int32_offsets():
    # The last of 2049 rows starts past 2**31 elements in both the native
    # logits and the aligned output. Only that row is live.
    rows, capacity, page_size, alignment = 2049, (1 << 20) + 64, 64, 512
    pages = capacity // page_size
    values = torch.empty((rows, capacity), device="cuda", dtype=torch.bfloat16)
    values[-1].normal_()
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    lengths[-1] = capacity
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
    table[-1, 1::3] = -1
    table[-1, 2::5] = pages + 3

    actual = clean_logits_for_selection(
        values, lengths, table, pages, capacity, page_size, alignment
    )

    assert (rows - 1) * actual.stride(0) >= 2**31
    page = table[-1, torch.arange(capacity, device="cuda") // page_size]
    expected = values[-1].masked_fill((page < 0) | (page >= pages), -torch.inf)
    torch.testing.assert_close(actual[-1, :capacity], expected, rtol=0, atol=0)
    tail = actual[-1, capacity:]
    assert tail.numel()
    torch.testing.assert_close(tail, torch.full_like(tail, -torch.inf), rtol=0, atol=0)


def test_selection_cleaner_capacity_does_not_recompile():
    from tokenspeed_kernel.ops.attention.dsv41.triton import _clean_logits

    torch.manual_seed(0)

    def run(rows, capacity):
        pages = (capacity + 63) // 64
        values = torch.randn((rows, capacity), device="cuda", dtype=torch.bfloat16)
        lengths = torch.randint(
            0, capacity + 1, (rows,), device="cuda", dtype=torch.int32
        )
        table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
        actual = clean_logits_for_selection(
            values, lengths, table, pages, capacity, 64, 512
        )
        expected = clean_logits(values, lengths, table, pages, capacity, 64)
        columns = torch.arange(capacity, device="cuda")[None, :]
        live = columns < (lengths[:, None] + 1023) // 1024 * 1024
        torch.testing.assert_close(
            actual[:, :capacity][live], expected[live], rtol=0, atol=0
        )

    run(8, 1024)
    with assert_no_triton_compile(_clean_logits):
        for rows, capacity in ((1, 2048), (17, 2112), (3, 16448), (6, 1048640)):
            run(rows, capacity)
