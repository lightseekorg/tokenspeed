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

"""Exact query-prefix reuse must remain valid as captured inputs change."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform


def _require_blackwell() -> None:
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("grouped QSA requires NVIDIA SM100 or SM103")


@pytest.mark.parametrize(
    ("heads", "kv_heads", "rows", "tokens", "expected"),
    [
        (12, 2, 80, 8, 2),
        (12, 2, 192, 6, 2),
        (12, 2, 512, 8, 8),
        (12, 2, 256, 4, 4),
        (12, 2, 576, 6, 6),
        (12, 2, 1024, 8, 8),
        (12, 1, 160, 8, 2),
        (12, 1, 512, 4, 4),
        (6, 1, 256, 8, 2),
        (6, 1, 192, 6, 3),
        (6, 1, 512, 4, 4),
        (12, 2, 1024, 1, 1),
        (12, 2, 1024, 3, 1),
        (3, 2, 512, 4, 1),
    ],
)
def test_qsa_query_groups_stay_inside_requests(
    heads: int, kv_heads: int, rows: int, tokens: int, expected: int
) -> None:
    _require_blackwell()
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import _query_group_size

    actual = _query_group_size(rows, kv_heads, heads, tokens, True, 152)
    assert actual == expected
    assert tokens % actual == 0
    assert _query_group_size(rows, kv_heads, heads, tokens, False, 152) == 1


def _reference_rows(q, k, v, selected, row_ids):
    heads_per_kv = q.shape[1] // k.shape[1]
    expected = torch.zeros(
        (len(row_ids), q.shape[1], q.shape[2]), dtype=torch.float64, device=q.device
    )
    for index, row in enumerate(row_ids):
        slots = selected[row]
        slots = slots[slots > 0].long()
        if slots.numel() == 0:
            continue
        query = q[row].double().view(k.shape[1], heads_per_kv, q.shape[2])
        keys = k[slots].double()
        values = v[slots].double()
        scores = torch.einsum("hrd,shd->hrs", query, keys) / 16
        result = torch.einsum("hrs,shd->hrd", scores.softmax(-1), values)
        expected[index] = result.reshape(q.shape[1], q.shape[2])
    return expected


@pytest.mark.parametrize(
    "selected_strides", [(2051, 1), (2056, 1)], ids=["compact", "row_padded"]
)
@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "batch", "tokens"),
    [
        (24, 2, 10, 8),
        (24, 2, 32, 6),
        (24, 2, 64, 8),
        (24, 2, 64, 4),
        (24, 2, 96, 6),
        (24, 2, 128, 8),
        (12, 1, 20, 8),
        (12, 1, 128, 4),
        (6, 1, 32, 8),
        (6, 1, 32, 6),
        (6, 1, 128, 4),
    ],
)
def test_qsa_query_groups_refresh_prefix_and_private_tail(
    device: str,
    q_heads: int,
    kv_heads: int,
    batch: int,
    tokens: int,
    selected_strides: tuple[int, int],
) -> None:
    _require_blackwell()
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import kernel

    torch.manual_seed(107 + q_heads + batch + tokens)
    rows, cache_slots = batch * tokens, 8192
    q = torch.randn(rows, q_heads, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(cache_slots, kv_heads, 256, device=device, dtype=q.dtype) * 0.25
    v = torch.randn_like(k) * 0.25
    blocks = (torch.randperm(1023, device=device)[:512] + 1) * 4
    prefix = (blocks[:, None] + torch.arange(4, device=device)).reshape(2048)
    selected = torch.empty_strided(
        (rows, 2051), selected_strides, device=device, dtype=torch.int32
    )
    selected[:, :2048] = prefix
    selected[:, 2048:] = 4096 + torch.arange(rows * 3, device=device).reshape(rows, 3)
    original_slots = selected.clone()

    def forward():
        return kernel(
            q,
            k,
            v,
            selected,
            scale=1 / 16,
            max_seqlen_q=tokens,
            k_scale=None,
            v_scale=None,
            enable_pdl=True,
        )

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    row_ids = sorted({0, 1, tokens - 1, tokens, rows // 2, rows - 1})
    for phase in ("shared", "late_mismatch", "masked", "distinct", "shared_again"):
        selected.copy_(original_slots)
        if phase == "late_mismatch":
            # Equality of the first tile must never establish full-prefix reuse.
            selected[1::2, 2047] = cache_slots - 1
            q[..., 0] = 16
            k[-1, :, 0] = 64
        elif phase == "masked":
            selected.fill_(-1)
        elif phase == "distinct":
            for row in range(rows):
                selected[row, :2048] = prefix.roll((row % tokens) * 4)
            selected[::3, 2048:] = -1
        elif phase == "shared_again":
            v.mul_(-0.5)
            selected[::2, 2048] = -1
        eager = forward()
        for _ in range(3):
            graph.replay()
        expected = _reference_rows(q, k, v, selected, row_ids)
        torch.testing.assert_close(
            output[row_ids].double(), expected, atol=2e-3, rtol=2e-2
        )
        torch.testing.assert_close(output, eager, atol=0, rtol=0)
        if phase == "masked":
            assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize(("kv_heads", "query_group"), [(1, 4), (2, 4), (2, 6), (2, 8)])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_qsa_prefix_flags_refresh_across_slot_layouts(
    device: str, kv_heads: int, query_group: int, enable_pdl: bool
) -> None:
    _require_blackwell()
    from tokenspeed_kernel.thirdparty.cute_dsl._qsa_sparse.prefix import (
        _COMPILED_PREPARE,
        _prepare_query_groups,
    )

    _COMPILED_PREPARE.clear()
    prefix = torch.arange(1, 2049, device=device, dtype=torch.int32)
    for groups in (2, 3):
        rows = groups * query_group
        # Compile with padded rows first, then reuse it for each different layout.
        for strides in ((2056, 1), (2051, 1), (4102, 2), (1, rows)):
            selected = torch.empty_strided(
                (rows, 2051), strides, device=device, dtype=torch.int32
            )
            selected[:, :2048] = prefix
            selected[:, 2048:] = torch.arange(rows * 3, device=device).reshape(rows, 3)
            shared = torch.empty(groups, device=device, dtype=torch.int32)
            expected = torch.ones_like(shared)

            def forward():
                _prepare_query_groups(
                    selected, shared, kv_heads, query_group, enable_pdl
                )

            forward()
            torch.testing.assert_close(shared, expected, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()

            selected[query_group + 1, 2047] = 2049
            expected[1] = 0
            graph.replay()
            torch.testing.assert_close(shared, expected, atol=0, rtol=0)

            selected[1, 0] = 2049
            expected[0] = 0
            graph.replay()
            torch.testing.assert_close(shared, expected, atol=0, rtol=0)

            selected[:, :2048] = prefix
            selected[:, 2048:] = -1
            expected.fill_(1)
            graph.replay()
            torch.testing.assert_close(shared, expected, atol=0, rtol=0)
