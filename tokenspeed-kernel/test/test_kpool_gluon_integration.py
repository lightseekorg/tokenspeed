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

from __future__ import annotations

import pytest
import tokenspeed_kernel.ops.attention as attention
import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 (GFX950) is required for Gluon KPool integration tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.attention.gluon import kpool_select  # isort: skip


class _FakeSortKernel:
    def __getitem__(self, grid):
        del grid

        def launch(logits, candidate_lens, values, indices, pool_offset, **kwargs):
            del candidate_lens, pool_offset, kwargs
            values.copy_(logits)
            row = torch.arange(logits.shape[1], dtype=torch.int32, device=logits.device)
            indices.copy_(row.expand_as(indices))

        return launch


def test_short_plan_scoring_keeps_ordered_fold_and_honors_workspace_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool, torch.Tensor, torch.Tensor]] = []

    def fake_plan_scorer(
        q,
        pooled_k_cache,
        weights,
        pool_workspace_slots,
        row_starts,
        row_ends,
        **kwargs,
    ):
        del pooled_k_cache, weights, pool_workspace_slots
        calls.append(
            (
                q.shape[0],
                kwargs["ordered_head_fold"],
                row_starts.clone(),
                row_ends.clone(),
            )
        )
        logits = kwargs["out"]
        logits.copy_(
            torch.arange(logits.shape[1], dtype=torch.float32).expand_as(logits)
        )
        kwargs["row_ends_out"].fill_(513)
        return logits, kwargs["row_ends_out"]

    monkeypatch.setattr(
        kpool_select,
        "gluon_dsa_kpool_prefill_plan_logits_gfx950",
        fake_plan_scorer,
    )
    monkeypatch.setattr(kpool_select, "_kpool_sort_topk_kernel", _FakeSortKernel())

    tokens = 3
    q = torch.empty((tokens, 32, 128), dtype=torch.bfloat16)
    weights = torch.empty((tokens, 32), dtype=torch.float32)
    cache = torch.empty((1, 16, 132), dtype=torch.uint8)
    pool_workspace_slots = torch.arange(1539, dtype=torch.int64)
    row_starts = torch.tensor((0, 513, 1026), dtype=torch.int32)
    row_ends = row_starts + 513
    causal_lens = torch.full((tokens,), 513 * 4, dtype=torch.int32)
    req_ids = torch.arange(tokens, dtype=torch.int32)
    index_block_table = torch.zeros((tokens, 1), dtype=torch.int32)
    window_width = 1024
    one_row_cap = (3 * window_width) * torch.float32.itemsize + 4

    result = kpool_select._select_pools_chunked_gluon(
        q,
        cache,
        weights,
        causal_lens,
        req_ids,
        index_block_table,
        pool_workspace_slots=pool_workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        pool_size=4,
        page_size=16,
        topk_pools=512,
        softmax_scale=128**-0.5,
        max_num_pools=513,
        chunk_pools=8192,
        max_logits_bytes=one_row_cap,
    )

    assert result.shape == (tokens, 512)
    assert [call[0] for call in calls] == [1, 1, 1]
    assert all(call[1] for call in calls)
    assert torch.equal(calls[0][2], row_starts[:1])
    assert torch.equal(calls[-1][3], row_ends[-1:])


def test_no_plan_scoring_uses_table_addressing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_table_scorer(
        q,
        pooled_k_cache,
        weights,
        causal_lens,
        req_ids,
        index_block_table,
        **kwargs,
    ):
        del pooled_k_cache, weights
        captured.update(
            q=q,
            causal_lens=causal_lens,
            req_ids=req_ids,
            index_block_table=index_block_table,
            ordered_head_fold=kwargs["ordered_head_fold"],
        )
        logits = kwargs["out"]
        logits.copy_(
            torch.arange(logits.shape[1], dtype=torch.float32).expand_as(logits)
        )
        kwargs["row_ends_out"].fill_(513)
        return logits, kwargs["row_ends_out"]

    def unexpected_plan_scorer(*args, **kwargs):
        raise AssertionError("no-plan selection used the physical-slot scorer")

    monkeypatch.setattr(
        kpool_select,
        "gluon_dsa_kpool_prefill_logits_gfx950",
        fake_table_scorer,
    )
    monkeypatch.setattr(
        kpool_select,
        "gluon_dsa_kpool_prefill_plan_logits_gfx950",
        unexpected_plan_scorer,
    )
    monkeypatch.setattr(kpool_select, "_kpool_sort_topk_kernel", _FakeSortKernel())

    q = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    cache = torch.empty((1, 16, 132), dtype=torch.uint8)
    weights = torch.empty((1, 32), dtype=torch.float32)
    causal_lens = torch.tensor((513 * 4,), dtype=torch.int32)
    req_ids = torch.zeros(1, dtype=torch.int32)
    index_block_table = torch.zeros((1, 33), dtype=torch.int32)
    result = kpool_select._select_pools_chunked_gluon(
        q,
        cache,
        weights,
        causal_lens,
        req_ids,
        index_block_table,
        pool_size=4,
        page_size=16,
        topk_pools=512,
        softmax_scale=128**-0.5,
        max_num_pools=513,
        chunk_pools=8192,
        max_logits_bytes=None,
    )

    assert result.shape == (1, 512)
    assert captured["q"].data_ptr() == q.data_ptr()
    assert torch.equal(captured["causal_lens"], causal_lens)
    assert torch.equal(captured["req_ids"], req_ids)
    assert captured["index_block_table"] is index_block_table
    assert captured["ordered_head_fold"] is True


@pytest.mark.parametrize(
    ("max_num_pools", "chunk_pools", "expected"),
    ((2048, 8192, True), (2049, 8192, False), (2048, 1024, False)),
)
def test_ordered_fold_is_limited_to_short_single_windows(
    max_num_pools: int,
    chunk_pools: int,
    expected: bool,
) -> None:
    width = kpool_select._normalized_window_width(
        max_num_pools,
        chunk_pools,
        512,
    )

    assert (
        kpool_select._uses_deterministic_single_window_sort(
            max_num_pools=max_num_pools,
            window_width=width,
        )
        is expected
    )


def test_public_kpool_prefill_dispatch_forwards_complete_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    expected = (object(), object())

    class FakeKernel:
        name = "fake_kpool_prefill"

        def __call__(self, **kwargs):
            captured["kwargs"] = kwargs
            return expected

    def fake_select_kernel(*args, **kwargs):
        captured["selection"] = (args, kwargs)
        return FakeKernel()

    monkeypatch.setattr(attention, "select_kernel", fake_select_kernel)
    q = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    weights = torch.empty((1, 32), dtype=torch.float32)
    cache = torch.empty((1, 16, 132), dtype=torch.uint8)
    positions = torch.zeros(1, dtype=torch.int32)
    query_start_loc = torch.tensor((0, 1), dtype=torch.int32)
    table = torch.zeros((1, 1), dtype=torch.int32)
    req_ids = torch.zeros(1, dtype=torch.int32)
    causal_lens = torch.ones(1, dtype=torch.int32)
    pool_workspace_slots = torch.zeros(1, dtype=torch.int64)
    row_starts = torch.zeros(1, dtype=torch.int32)
    row_ends = torch.ones(1, dtype=torch.int32)

    actual = attention.kpool_prefill_topk(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        table,
        table,
        pool_size=4,
        page_size=16,
        kv_page_size=64,
        topk_pools=512,
        softmax_scale=128**-0.5,
        req_ids=req_ids,
        causal_lens=causal_lens,
        pool_workspace_slots=pool_workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        max_num_pools=1,
        max_logits_bytes=4096,
    )

    assert actual is expected
    traits = captured["selection"][1]["traits"]
    assert traits["index_heads"] == 32
    assert traits["topk_pools"] == 512
    assert traits["prefill_plan"] is True
    forwarded = captured["kwargs"]
    assert forwarded["pool_workspace_slots"] is pool_workspace_slots
    assert forwarded["row_starts"] is row_starts
    assert forwarded["row_ends"] is row_ends
    assert forwarded["max_logits_bytes"] == 4096


def test_gluon_registration_covers_both_addressing_modes() -> None:
    from tokenspeed_kernel.registry import KernelRegistry

    spec = KernelRegistry.get().get_by_name("gluon_kpool_prefill_topk_fp8_gfx950")
    if spec is None:
        pytest.skip("gfx950 Gluon registrations are unavailable")

    assert spec.traits["prefill_plan"] == frozenset({False, True})
    assert spec.traits["index_heads"] == frozenset({32})
    assert spec.traits["topk_pools"] == frozenset({512})


def test_registry_supports_both_modes_and_geometry_fallback() -> None:
    from tokenspeed_kernel.registry import KernelRegistry

    spec = KernelRegistry.get().get_by_name("gluon_kpool_prefill_topk_fp8_gfx950")
    if spec is None:
        pytest.skip("gfx950 Gluon registrations are unavailable")

    traits = {
        "index_heads": 32,
        "head_dim": 128,
        "pool_size": 4,
        "page_size": 16,
        "topk_pools": 512,
        "index_k_format": "fp8_scaled",
        "score_activation": "relu",
        "topk_layout": "global_slots",
        "prefill_plan": True,
    }
    signature = format_signature(q=dense_tensor_format(torch.bfloat16))

    selected = select_kernel(
        "attention",
        "kpool_prefill_topk",
        signature,
        traits=traits,
    )
    assert selected.name == "gluon_kpool_prefill_topk_fp8_gfx950"

    traits["prefill_plan"] = False
    selected_without_plan = select_kernel(
        "attention",
        "kpool_prefill_topk",
        signature,
        traits=traits,
    )
    assert selected_without_plan.name == "gluon_kpool_prefill_topk_fp8_gfx950"

    traits["topk_pools"] = 64
    fallback = select_kernel(
        "attention",
        "kpool_prefill_topk",
        signature,
        traits=traits,
    )
    assert fallback.name == "triton_kpool_prefill_topk"
