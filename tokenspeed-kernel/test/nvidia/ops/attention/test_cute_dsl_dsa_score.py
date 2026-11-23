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

"""Numerical and CUDA Graph checks for the Hopper paged DSA scorer."""

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia, move_pages_past_int32

if not is_nvidia() or torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("requires NVIDIA SM90", allow_module_level=True)
pytest.importorskip("cutlass")

from tokenspeed_kernel.ops.attention.dsa import dsa_paged_score  # noqa: E402
from tokenspeed_kernel.selection import NoKernelFoundError  # noqa: E402


def _reference(q, keys, weights, table, lengths, rows, width, page_size):
    tokens = torch.arange(width, device=q.device)
    valid_rows = (rows >= 0) & (rows < table.shape[0])
    in_table = tokens // page_size < table.shape[1]
    pages = table[
        rows.clamp(0, table.shape[0] - 1).long()[:, None],
        (tokens // page_size).clamp_max(table.shape[1] - 1),
    ].long()
    slots = pages * page_size + tokens % page_size
    valid = (
        valid_rows[:, None]
        & in_table[None, :]
        & (tokens < lengths[:, None])
        & (pages >= 0)
        & (slots < keys.shape[0])
    )
    gathered = keys[slots.clamp(0, keys.shape[0] - 1)].float()
    scores = (
        torch.einsum("rhd,rtd->rht", q.float(), gathered).relu() * weights[:, :, None]
    ).sum(1)
    return scores.masked_fill(~valid, -float("inf"))


@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("page_size", [64, 128])
def test_paged_bf16_scores(heads, page_size):
    rng = torch.Generator(device="cuda").manual_seed(312)
    q = (
        torch.randn((6, heads, 128), device="cuda", dtype=torch.bfloat16, generator=rng)
        * 0.2
    )
    k = (
        torch.randn(
            (8 * page_size, 128), device="cuda", dtype=torch.bfloat16, generator=rng
        )
        * 0.2
    )
    weights = torch.randn((6, heads * 2), device="cuda", generator=rng)[:, :heads]
    # Page 0 is live, and rows may have a padded outer stride.
    table = torch.tensor(
        [[5, 0, -1, 2, 0], [7, 100, 3, 0, 0]], device="cuda", dtype=torch.int32
    )[:, :4]
    lengths = torch.tensor(
        [0, 1, 65, page_size + 3, 4 * page_size, 4 * page_size - 1],
        device="cuda",
        dtype=torch.int32,
    )
    rows = torch.tensor([1, 0, 1, 0, 0, 1], device="cuda", dtype=torch.int32)
    width = 4 * page_size + 3
    expected = _reference(q, k, weights, table, lengths, rows, width, page_size)
    actual = dsa_paged_score(
        q,
        k,
        weights,
        table,
        lengths,
        max_seq_len=width,
        page_size=page_size,
        q_len_per_req=None,
        request_ids=rows,
    )
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    rows[2] = -1
    rows[5] = table.shape[0]
    actual = dsa_paged_score(
        q,
        k,
        weights,
        table,
        lengths,
        max_seq_len=width,
        page_size=page_size,
        q_len_per_req=None,
        request_ids=rows,
    )
    assert torch.isneginf(actual[[2, 5]]).all()


def test_paged_bf16_graph_replay_and_dynamic_shapes():
    rng = torch.Generator(device="cuda").manual_seed(406)
    page_size, q_len, heads, width = 64, 8, 16, 8193
    rows = 64
    q = (
        torch.randn(
            (rows, heads, 128), device="cuda", dtype=torch.bfloat16, generator=rng
        )
        * 0.2
    )
    k = (
        torch.randn(
            (width + 63, 128), device="cuda", dtype=torch.bfloat16, generator=rng
        )
        * 0.2
    )
    w = torch.randn((rows, heads), device="cuda", generator=rng)
    table = torch.randperm(
        k.shape[0] // 64, device="cuda", dtype=torch.int32, generator=rng
    )
    table = table.repeat(rows // q_len, 1)
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    output = torch.empty((rows, width), device="cuda")
    kwargs = dict(max_seq_len=width, page_size=64, q_len_per_req=q_len, out=output)
    dsa_paged_score(q, k, w, table, lengths, **kwargs)
    with assert_no_cute_compile():
        dsa_paged_score(
            q[:2],
            k,
            w[:2],
            table[:, :2],
            lengths[:2],
            max_seq_len=65,
            page_size=64,
            q_len_per_req=1,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = dsa_paged_score(q, k, w, table, lengths, **kwargs)
        table[0, 1] = -1
        table[1, 0] = 0
        lengths[:8] = torch.tensor([0, 1, 63, 64, 65, 66, 8192, 8193], device="cuda")
        k.mul_(0.75)
        w.mul_(-0.25)
        output.fill_(float("nan"))
        graph.replay()
    row_ids = torch.arange(rows, device="cuda") // q_len
    for start in range(0, rows, 8):
        ix = slice(start, start + 8)
        expected = _reference(
            q[ix], k, w[ix], table, lengths[ix], row_ids[ix], width, page_size
        )
        torch.testing.assert_close(actual[ix], expected, atol=3e-5, rtol=3e-4)
    assert actual.data_ptr() == output.data_ptr()


def test_paged_bf16_offsets_past_int32():
    page_size = 64
    rng = torch.Generator(device="cuda").manual_seed(71)
    q = (
        torch.randn((1, 16, 128), device="cuda", dtype=torch.bfloat16, generator=rng)
        * 0.2
    )
    keys = (
        torch.randn(
            (2, page_size, 128), device="cuda", dtype=torch.bfloat16, generator=rng
        )
        * 0.2
    )
    weight = torch.ones((1, 16), device="cuda")
    table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([65], device="cuda", dtype=torch.int32)
    expected = _reference(
        q,
        keys.flatten(0, 1),
        weight,
        table,
        lengths,
        torch.zeros_like(lengths),
        128,
        page_size,
    )
    (far_keys,), far_table = move_pages_past_int32((keys,), table)
    actual = dsa_paged_score(
        q,
        far_keys.flatten(0, 1),
        weight,
        far_table,
        lengths,
        max_seq_len=128,
        page_size=page_size,
        q_len_per_req=1,
    )
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


def test_paged_bf16_empty_and_unsupported():
    device = "cuda"
    q = torch.zeros((2, 16, 128), device=device, dtype=torch.bfloat16)
    keys = q.new_empty((0, 128))
    w = torch.ones((2, 16), device=device)
    table = torch.zeros((2, 1), device=device, dtype=torch.int32)
    lengths = torch.zeros((2,), device=device, dtype=torch.int32)
    kwargs = dict(max_seq_len=64, page_size=64, q_len_per_req=1)
    assert torch.isneginf(dsa_paged_score(q, keys, w, table, lengths, **kwargs)).all()
    assert dsa_paged_score(q[:0], keys, w[:0], table, lengths[:0], **kwargs).shape == (
        0,
        64,
    )
    with pytest.raises(NoKernelFoundError):
        dsa_paged_score(q.float(), keys, w, table, lengths, **kwargs)
