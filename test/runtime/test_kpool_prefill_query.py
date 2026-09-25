"""KPool prefill query preparation resolves with, and feeds, the planned top-k."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.attention import kpool as kpool_runtime
from tokenspeed.runtime.layers.attention.kpool import KPoolRuntime

_HEAD_DIM = 128


def _fake_forward(
    starts: list[int], lengths: list[int], token_capacity: int
) -> tuple[KPoolRuntime, SimpleNamespace, SimpleNamespace]:
    requests = len(starts)
    index_cache = torch.zeros((8, 16, _HEAD_DIM + 4), dtype=torch.uint8)
    tail_k = torch.zeros((requests + 2, 4, _HEAD_DIM), dtype=torch.bfloat16)
    pool = SimpleNamespace(
        get_kpool_buffers=lambda _layer_id: (index_cache, tail_k, tail_k.clone()),
        arena=SimpleNamespace(kv_page_size=64),
    )
    ctx = SimpleNamespace(
        num_extends=requests,
        token_to_kv_pool=pool,
        attn_backend=SimpleNamespace(max_context_len=4096),
    )
    lengths_cpu = torch.tensor(lengths, dtype=torch.int64)
    starts_cpu = torch.tensor(starts, dtype=torch.int64)
    backend = SimpleNamespace(
        chunked_prefill_metadata=SimpleNamespace(
            extend_prefix_lens_cpu=starts_cpu,
            extend_seq_lens_cpu=lengths_cpu,
            extend_prefix_lens=starts_cpu,
            extend_seq_lens=lengths_cpu,
        ),
        kpool_prefill_page_table=lambda n: torch.arange(
            1, 4 * n + 1, dtype=torch.int32
        ).view(n, 4),
    )
    runtime = KPoolRuntime(pool_size=4, index_topk=2048)
    runtime.reset_forward(torch.arange(1, requests + 1, dtype=torch.int64))
    runtime.ensure_prefill_plan(ctx, backend, 3, token_capacity=token_capacity)
    return runtime, ctx, backend


def test_prepare_prefill_query_selects_with_planned_topk_traits(monkeypatch) -> None:
    captured = {}

    def fake_prepare(q, cache, weights, **kwargs):
        captured["q"] = q
        captured["cache"] = cache
        captured["weights"] = weights
        captured["kwargs"] = kwargs
        return "prepared"

    monkeypatch.setattr(kpool_runtime, "kpool_prefill_prepare_query", fake_prepare)
    runtime, ctx, _ = _fake_forward([70], [70], 70)
    query = torch.zeros((70, 2, _HEAD_DIM), dtype=torch.bfloat16)
    weights = torch.zeros((70, 2), dtype=torch.bfloat16)

    prepared = runtime.prepare_prefill_query(
        query=query, weights=weights, softmax_scale=0.25, ctx=ctx, layer_id=3
    )

    assert prepared == "prepared"
    assert captured["q"] is query and captured["weights"] is weights
    assert captured["cache"] is ctx.token_to_kv_pool.get_kpool_buffers(3)[0]
    assert captured["kwargs"] == {
        "pool_size": 4,
        "page_size": 16,
        "topk_pools": 512,
        "softmax_scale": 0.25,
        "apply_relu": True,
    }


def test_select_prefill_forwards_prepared_query(monkeypatch) -> None:
    captured = {}

    def fake_topk(*args, **kwargs):
        captured["kwargs"] = kwargs
        tokens = args[0].shape[0]
        return (
            torch.full((tokens, 3), -1, dtype=torch.int32),
            torch.zeros(tokens, dtype=torch.int32),
        )

    monkeypatch.setattr(kpool_runtime, "kpool_prefill_topk", fake_topk)
    monkeypatch.setitem(
        kpool_runtime.global_server_args_dict,
        "deepseek_v4_indexer_prefill_max_logits_mb",
        1,
    )
    runtime, ctx, backend = _fake_forward([70], [70], 70)
    prepared = object()

    selected = runtime.select_prefill(
        query=torch.zeros((70, 2, _HEAD_DIM), dtype=torch.bfloat16),
        weights=torch.zeros((70, 2), dtype=torch.bfloat16),
        softmax_scale=0.25,
        prepared_query=prepared,
        ctx=ctx,
        backend=backend,
        layer_id=3,
        num_prefill_tokens=70,
    )

    assert captured["kwargs"]["prepared_query"] is prepared
    assert captured["kwargs"]["max_num_pools"] == 35
    assert selected is not None
    assert selected.kv_workspace_slots.tolist() == [-1] * (70 * 3)
