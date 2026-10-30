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

from contextlib import nullcontext

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform


def _require_blackwell() -> None:
    if current_platform().arch_version not in (
        ArchVersion(10, 0),
        ArchVersion(10, 3),
        ArchVersion(10, 7),
    ):
        pytest.skip("grouped QSA requires NVIDIA SM100, SM103, or SM107")


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
    ("q_heads", "kv_heads", "tokens", "row_counts", "cache_dtype"),
    [
        (24, 2, 1, (33, 35, 37), torch.bfloat16),
        (24, 2, 2, (80, 82, 86), torch.bfloat16),
        (24, 2, 4, (256, 260, 268), torch.bfloat16),
        (24, 2, 6, (576, 582, 594), torch.bfloat16),
        (24, 2, 8, (512, 520, 528), torch.bfloat16),
        (6, 1, 6, (192, 198, 204), torch.bfloat16),
        (24, 2, 1, (17, 19, 23), torch.float8_e4m3fn),
    ],
)
def test_qsa_dynamic_rows_reuse_compilation(
    device, monkeypatch, q_heads, kv_heads, tokens, row_counts, cache_dtype
):
    _require_blackwell()
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse

    monkeypatch.setattr(qsa_sparse, "_COMPILED_KERNELS", {})
    torch.manual_seed(109)
    keys = (torch.randn(256, kv_heads, 256, device=device) * 0.25).to(cache_dtype)
    values = (torch.randn_like(keys, dtype=torch.float32) * 0.25).to(cache_dtype)
    queries = [
        torch.randn(rows, q_heads, 256, device=device, dtype=torch.bfloat16)
        for rows in row_counts
    ]
    selections = [
        torch.empty_strided(
            (rows, 2051), strides, device=device, dtype=torch.int32
        ).fill_(-1)
        for rows, strides in zip(row_counts, ((2051, 1), (2056, 1), (4102, 2)))
    ]
    for selected in selections:
        selected[:, :8] = torch.arange(1, 9, device=device)
        # Exercise both exact prefix reuse and independent-query fallback.
        selected[tokens :: 2 * tokens, 7] = 9
        selected[:, -3:] = torch.arange(10, 13, device=device)

    def forward(index):
        return qsa_sparse.kernel(
            queries[index],
            keys,
            values,
            selections[index],
            scale=1 / 16,
            max_seqlen_q=tokens,
            k_scale=None,
            v_scale=None,
            enable_pdl=True,
        )

    forward(0)
    compiled = dict(qsa_sparse._COMPILED_KERNELS)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [forward(index) for index in (1, 2)]
    for factor in (0.75, -0.5):
        for query in queries:
            query.mul_(factor)
        graph.replay()
        for index, output in zip((1, 2), captured):
            row_ids = sorted({0, 1, tokens, tokens + 1, row_counts[index] - 1})
            expected = _reference_rows(
                queries[index], keys, values, selections[index], row_ids
            )
            torch.testing.assert_close(
                output[row_ids].double(), expected, rtol=0.01, atol=0.003
            )
            torch.testing.assert_close(forward(index), output, rtol=0, atol=0)
    # Compiling during capture or a later eager forward is serving behavior.
    assert qsa_sparse._COMPILED_KERNELS == compiled


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "cache_dtype"),
    [
        (24, 2, torch.bfloat16),
        (6, 1, torch.bfloat16),
        (24, 2, torch.float8_e4m3fn),
    ],
)
def test_qsa_startup_warms_launch_configurations(
    device, monkeypatch, q_heads, kv_heads, cache_dtype
):
    _require_blackwell()
    from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention
    from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens, kernel_warmup
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse

    monkeypatch.setattr(qsa_sparse, "_COMPILED_KERNELS", {})
    monkeypatch.setattr(qsa_sparse, "_WARMED", set())
    torch.manual_seed(113)
    keys = (torch.randn(256, kv_heads, 256, device=device) * 0.25).to(cache_dtype)
    values = (torch.randn_like(keys, dtype=torch.float32) * 0.25).to(cache_dtype)

    def inputs(rows):
        query = torch.randn(rows, q_heads, 256, device=device, dtype=torch.bfloat16)
        selected = torch.full((rows, 2051), -1, device=device, dtype=torch.int32)
        selected[:, :8] = torch.arange(1, 9, device=device)
        return query, selected

    def forward(query, selected, width):
        return qsa_sparse_attention(
            query,
            keys,
            values,
            selected,
            scale=1 / 16,
            max_seqlen_q=width,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution="cute_dsl",
            sparse_metadata=None,
        )

    first = inputs(1)
    with kernel_warmup(enabled=True):
        forward(*first, 1)
    compiled = dict(qsa_sparse._COMPILED_KERNELS)
    assert compiled
    # Scan every legal request width, independently of warmup's representatives.
    # A new grouping rule must remain covered by the startup traversal.
    configurations = {}
    max_rows = get_autotune_max_num_tokens()
    for width in range(1, max_rows + 1):
        for rows in range(width, max_rows + 1, width):
            config = qsa_sparse._configuration(
                rows,
                q_heads,
                kv_heads,
                width,
                cache_dtype is torch.bfloat16,
                qsa_sparse._sm_count(keys.device.index),
                keys.device.index,
            )
            configurations.setdefault(config, (rows, width))
    for rows, width in configurations.values():
        query, selected = inputs(rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward(query, selected, width)
        graph.replay()
        row_ids = [0, rows - 1]
        expected = _reference_rows(query, keys, values, selected, row_ids)
        torch.testing.assert_close(
            output[row_ids].double(), expected, rtol=0.01, atol=0.003
        )
    assert qsa_sparse._COMPILED_KERNELS == compiled


@pytest.mark.parametrize(
    "unsupported",
    [
        "query_rank",
        "selected_rank",
        "key_head_dim",
        "value_shape",
        "value_dtype",
        "selected_dtype",
        "selected_rows",
        "query_device",
        "key_device",
        "value_device",
        "selected_device",
    ],
)
def test_qsa_warmup_skips_unsupported_tensors(device, monkeypatch, unsupported):
    _require_blackwell()
    from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
        _warmup_qsa_sparse_attention,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse

    monkeypatch.setattr(qsa_sparse, "_COMPILED_KERNELS", {})
    monkeypatch.setattr(qsa_sparse, "_WARMED", set())
    query = torch.empty(1, 24, 256, device=device, dtype=torch.bfloat16)
    keys = torch.empty(16, 2, 256, device=device, dtype=torch.bfloat16)
    values = torch.empty_like(keys)
    selected = torch.empty(1, 2051, device=device, dtype=torch.int32)
    match unsupported:
        case "query_rank":
            query = query[0]
        case "selected_rank":
            selected = selected[0]
        case "key_head_dim":
            keys, values = keys[:, :, :128], values[:, :, :128]
        case "value_shape":
            values = values[:, :, :128]
        case "value_dtype":
            values = values.float()
        case "selected_dtype":
            selected = selected.long()
        case "selected_rows":
            selected = selected[:0]
        case "query_device":
            query = query.cpu()
        case "key_device":
            keys = keys.cpu()
        case "value_device":
            values = values.cpu()
        case "selected_device":
            selected = selected.cpu()
    _warmup_qsa_sparse_attention(query, keys, values, selected, max_rows=1)
    assert not qsa_sparse._COMPILED_KERNELS
    assert not qsa_sparse._WARMED


@pytest.mark.parametrize("override_source", ["context", "environment"])
def test_qsa_startup_respects_kernel_override(device, monkeypatch, override_source):
    _require_blackwell()
    from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import kernel_override
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse

    monkeypatch.setattr(qsa_sparse, "_COMPILED_KERNELS", {})
    monkeypatch.setattr(qsa_sparse, "_WARMED", set())
    torch.manual_seed(179)
    query = torch.randn(2, 24, 256, device=device, dtype=torch.bfloat16)
    keys = torch.randn(16, 2, 256, device=device, dtype=torch.bfloat16) * 0.25
    values = torch.randn_like(keys) * 0.25
    selected = torch.full((2, 2051), -1, device=device, dtype=torch.int32)
    selected[:, :8] = torch.arange(1, 9, device=device)
    if override_source == "environment":
        monkeypatch.setenv(
            "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_SPARSE_ATTENTION", "triton"
        )
        context = nullcontext()
    else:
        context = kernel_override("attention", "qsa_sparse_attention", "triton")
    with context, kernel_warmup(enabled=True):
        output = qsa_sparse_attention(
            query,
            keys,
            values,
            selected,
            scale=1 / 16,
            max_seqlen_q=1,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override="cute_dsl",
            solution=None,
            sparse_metadata=None,
        )
    expected = _reference_rows(query, keys, values, selected, (0, 1))
    torch.testing.assert_close(output.double(), expected, rtol=0.01, atol=0.003)
    assert not qsa_sparse._COMPILED_KERNELS
    assert not qsa_sparse._WARMED


@pytest.mark.parametrize("override_source", ["context", "environment", "callsite"])
def test_qsa_startup_warms_override_with_conflicting_solution(
    device, monkeypatch, override_source
):
    _require_blackwell()
    from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel.selection import kernel_override
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse

    monkeypatch.setattr(qsa_sparse, "_COMPILED_KERNELS", {})
    monkeypatch.setattr(qsa_sparse, "_WARMED", set())
    env_key = "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_SPARSE_ATTENTION"
    monkeypatch.delenv(env_key, raising=False)
    forced = "cute_dsl_blackwell_qsa_sparse_attention"
    override = forced if override_source == "callsite" else "triton"
    if override_source == "environment":
        monkeypatch.setenv(env_key, forced)
        context = nullcontext()
    elif override_source == "context":
        context = kernel_override("attention", "qsa_sparse_attention", forced)
    else:
        context = nullcontext()

    torch.manual_seed(193)
    keys = torch.randn(16, 2, 256, device=device, dtype=torch.bfloat16) * 0.25
    values = torch.randn_like(keys) * 0.25

    def inputs(rows):
        query = torch.randn(rows, 24, 256, device=device, dtype=torch.bfloat16)
        selected = torch.full((rows, 2051), -1, device=device, dtype=torch.int32)
        selected[:, :8] = torch.arange(1, 9, device=device)
        return query, selected

    def forward(query, selected, width):
        return qsa_sparse_attention(
            query,
            keys,
            values,
            selected,
            scale=1 / 16,
            max_seqlen_q=width,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=override,
            solution="triton",
            sparse_metadata=None,
        )

    with context:
        with kernel_warmup(enabled=True):
            forward(*inputs(1), 1)
        compiled = dict(qsa_sparse._COMPILED_KERNELS)
        assert compiled
        for rows, width in ((80, 2), (256, 4), (520, 8), (594, 6)):
            query, selected = inputs(rows)
            eager = forward(query, selected, width)
            assert qsa_sparse._COMPILED_KERNELS == compiled
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward(query, selected, width)
            graph.replay()
            torch.testing.assert_close(eager, output, rtol=0, atol=0)
            query.mul_(0.75)
            graph.replay()
            row_ids = [0, rows - 1]
            expected = _reference_rows(query, keys, values, selected, row_ids)
            torch.testing.assert_close(
                output[row_ids].double(), expected, rtol=0.01, atol=0.003
            )
            assert qsa_sparse._COMPILED_KERNELS == compiled


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
