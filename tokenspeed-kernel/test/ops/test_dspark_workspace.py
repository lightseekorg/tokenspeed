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

"""DSpark history and encoded current rows land directly in their final layout."""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41.triton import _dspark_workspace_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _inputs(batch, block, history, strided):
    device = "cuda"
    window = torch.randn(
        11, 97 if strided else 64, 512, device=device, dtype=torch.bfloat16
    )[:, :64]
    slots = (
        torch.arange(history, device=device)[None, :]
        + torch.arange(batch, device=device)[:, None] * 37
    ) % 512 + 64
    if batch:
        slots[0] = -1
    if batch > 1:
        slots[1, : history // 2] = -1
        slots[-1] = -1
    starts = torch.arange(batch, device=device, dtype=torch.int64) * 5 + 40
    _, _, _, _, page, row, indices = dsv41.dspark_block(
        torch.zeros_like(starts),
        starts,
        slots,
        0,
        64,
        1,
        block,
    )
    if strided:

        def sliced(x):
            storage = torch.empty(batch, history * 2, device=device, dtype=x.dtype)
            out = storage[:, ::2]
            out.copy_(x)
            return out

        page, row = sliced(page), sliced(row)
    values = torch.randn(
        batch * block, 1024 if strided else 512, device=device, dtype=torch.bfloat16
    )[:, :512]
    positions = torch.arange(batch * block * 2, device=device, dtype=torch.int64)[::2]
    if positions.numel():
        positions[-1] = -1
    angles = (
        torch.arange(4096, device=device)[:, None]
        * torch.exp(-torch.arange(32, device=device).float() / 8)[None, :]
    )
    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return values, positions, table, window, page, row, indices


def _workspace(args, block, with_rope):
    values, positions, table, window, page, row, _ = args
    return dsv41.dspark_workspace(
        values,
        None,
        None,
        positions if with_rope else None,
        table if with_rope else None,
        window,
        page,
        row,
        block,
    )


def _reference(args, block, with_rope):
    values, positions, table, window, page, row, _ = args
    current = values.clone()
    if current.shape[0]:
        if with_rope:
            current = dsv41.rope_inplace(current, positions, table, None)
        current = dsv41.cache_unpack(
            dsv41.cache_pack(current, "swa", None), "swa", None
        )
    return torch.cat(
        (window[page, row], current.reshape(page.shape[0], block, 512)), dim=1
    )


@pytest.mark.parametrize("batch", [0, 1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("block", [1, 6])
@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_workspace_matches_separate_producers(batch, block, with_rope, strided):
    args = _inputs(batch, block, 128, strided)
    values, _, _, window, _, _, indices = args
    before = [x.clone() for x in (values, window, indices)]
    expected = _reference(args, block, with_rope)
    actual = _workspace(args, block, with_rope)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    for source, original in zip((values, window, indices), before, strict=True):
        torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_workspace_runtime_geometry_does_not_compile():
    _workspace(_inputs(1, 1, 64, False), 1, True)
    with assert_no_triton_compile(_dspark_workspace_kernel):
        for batch, block, history in [(2, 6, 128), (4, 1, 96), (32, 6, 64)]:
            args = _inputs(batch, block, history, True)
            torch.testing.assert_close(
                _workspace(args, block, True),
                _reference(args, block, True),
                rtol=0,
                atol=0,
            )


def test_workspace_eager_capture_replay_eager_own_their_outputs():
    args = _inputs(4, 6, 128, True)
    values, positions, _, window, page, row, _ = args
    live = page != 0
    eager_before = _workspace(args, 6, True)
    expected_before = eager_before.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _workspace(args, 6, True)
    for shift in [1, 7, 23]:
        values.normal_()
        window.normal_()
        positions.add_(1)
        # Change physical history addressing and payload, retaining dummy
        # addresses and the original selection mask for invalid history.
        page.copy_(torch.where(live, (page + shift) % (window.shape[0] - 1) + 1, page))
        row.copy_(torch.where(live, (row + shift) % window.shape[1], row))
        graph.replay()
        expected = _reference(args, 6, True)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        eager_after = _workspace(args, 6, True)
        after_copy = eager_after.clone()
        values.add_(0.125)
        window.add_(0.125)
        graph.replay()
        torch.testing.assert_close(eager_after, after_copy, rtol=0, atol=0)
        torch.testing.assert_close(eager_before, expected_before, rtol=0, atol=0)


@pytest.mark.parametrize("with_rope", [False, True])
def test_workspace_preserves_deferred_norm(with_rope):
    values, positions, table, window, page, row, _ = _inputs(4, 6, 128, True)
    weight = torch.rand(512, device="cuda", dtype=torch.bfloat16) + 0.5
    pos = positions if with_rope else None
    rope = table if with_rope else None
    current = dsv41.dspark_rows(
        values, weight, 1e-6, pos, rope, None, None, torch.empty_like(values)
    )
    expected = torch.cat((window[page, row], current.reshape(4, 6, 512)), dim=1)
    actual = dsv41.dspark_workspace(
        values, weight, 1e-6, pos, rope, window, page, row, 6
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
