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

"""End-to-end GFX950 coverage for plan-aware Gluon KPool selection."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 (GFX950) is required for Gluon KPool plan tests",
        allow_module_level=True,
    )

from tokenspeed_kernel import kpool_prefill_topk  # isort: skip
from tokenspeed_kernel.ops.attention.gluon import (  # isort: skip
    kpool_select as gluon_kpool_select,
)
from tokenspeed_kernel.ops.attention.gluon.kpool_select import (  # isort: skip
    gluon_kpool_prefill_topk_fp8_gfx950,
)
from tokenspeed_kernel.ops.attention.triton.kpool_select import (  # isort: skip
    triton_kpool_prefill_topk,
)
from tokenspeed_kernel.selection import select_kernel  # isort: skip
from tokenspeed_kernel.signature import (  # isort: skip
    dense_tensor_format,
    format_signature,
)

_DEVICE = "cuda"
_FP8_MAX = 448.0
_POOL = 4
_PAGE = 16
_KV_PAGE = 64
_HEADS = 32
_DIM = 128
_ROW_BYTES = _DIM + 4
_TOPK = 512
_SCALE = _DIM**-0.5


def _build_cache(
    num_pages: int,
    *,
    seed: int,
    page_padding_bytes: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    rows = num_pages * _PAGE
    values = (
        torch.randn(
            (rows, _DIM),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        )
        * 0.2
    )
    scales = values.abs().amax(-1, keepdim=True).clamp_min(1.0e-4) / _FP8_MAX
    fp8 = (values / scales).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    page_bytes = _PAGE * _ROW_BYTES
    backing = torch.zeros(
        (num_pages, page_bytes + page_padding_bytes),
        dtype=torch.uint8,
        device=_DEVICE,
    )
    cache = backing[:, :page_bytes]
    cache[:, : _PAGE * _DIM].copy_(fp8.view(torch.uint8).view(num_pages, -1))
    cache[:, _PAGE * _DIM :].copy_(
        scales.contiguous().view(torch.uint8).view(num_pages, -1)
    )
    return cache, fp8.float() * scales


def _setup_rows(
    causal_lens: tuple[int, ...],
    *,
    seed: int,
    page_padding_bytes: int = 0,
    permute_index_pages: bool = False,
) -> tuple[torch.Tensor, ...]:
    requests = len(causal_lens)
    max_num_pools = max(causal_lens) // _POOL
    index_pages_per_req = (max_num_pools + _PAGE - 1) // _PAGE + 1
    kv_pages_per_req = (max(causal_lens) + _KV_PAGE - 1) // _KV_PAGE + 1
    cache, dequantized = _build_cache(
        requests * index_pages_per_req,
        seed=seed + 1,
        page_padding_bytes=page_padding_bytes,
    )
    index_table = torch.arange(
        requests * index_pages_per_req,
        dtype=torch.int32,
        device=_DEVICE,
    ).view(requests, index_pages_per_req)
    if permute_index_pages:
        index_table = index_table.flip(1).contiguous()
    kv_table = torch.arange(
        requests * kv_pages_per_req,
        dtype=torch.int32,
        device=_DEVICE,
    ).view(requests, kv_pages_per_req)
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    q = (
        torch.randn(
            (requests, _HEADS, _DIM),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        )
        * 0.3
    ).to(torch.bfloat16)
    weights = torch.randn(
        (requests, _HEADS),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )
    causal = torch.tensor(causal_lens, dtype=torch.int32, device=_DEVICE)
    req_ids = torch.arange(requests, dtype=torch.int32, device=_DEVICE)
    positions = causal - 1
    query_start_loc = torch.arange(requests + 1, dtype=torch.int32, device=_DEVICE)
    return (
        q,
        cache,
        dequantized,
        weights,
        positions,
        query_start_loc,
        req_ids,
        causal,
        index_table,
        kv_table,
    )


def _build_physical_plan(
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    request_counts: list[int] = []
    for req in range(index_table.shape[0]):
        request_rows = causal_lens[req_ids == req]
        request_counts.append(
            0 if request_rows.numel() == 0 else int(request_rows.max().item()) // _POOL
        )

    workspace_parts = []
    request_starts = []
    workspace_start = 0
    for req, count in enumerate(request_counts):
        request_starts.append(workspace_start)
        if count:
            pools = torch.arange(count, dtype=torch.int64, device=_DEVICE)
            pages = index_table[req, pools // _PAGE].to(torch.int64)
            workspace_parts.append(pages * _PAGE + pools % _PAGE)
        workspace_start += count
    if workspace_parts:
        workspace_slots = torch.cat(workspace_parts).contiguous()
    else:
        workspace_slots = torch.empty(0, dtype=torch.int64, device=_DEVICE)
    row_starts = torch.tensor(
        [request_starts[int(req)] for req in req_ids.tolist()],
        dtype=torch.int32,
        device=_DEVICE,
    )
    row_ends = row_starts + causal_lens // _POOL
    return workspace_slots, row_starts, row_ends, max(request_counts, default=0)


def _reference(
    q: torch.Tensor,
    pooled_k: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_table: torch.Tensor,
    kv_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = _TOPK * _POOL + _POOL - 1
    slots = torch.full((q.shape[0], width), -1, dtype=torch.int32, device=_DEVICE)
    lens = torch.empty(q.shape[0], dtype=torch.int32, device=_DEVICE)
    pooled_k = pooled_k.view(-1, _PAGE, _DIM)
    for token in range(q.shape[0]):
        req = int(req_ids[token])
        causal = max(int(causal_lens[token]), 0)
        num_pools = causal // _POOL
        if num_pools <= _TOPK:
            selected = torch.arange(num_pools, device=_DEVICE)
        else:
            pools = torch.arange(num_pools, device=_DEVICE)
            pages = index_table[req, pools // _PAGE].long()
            keys = pooled_k[pages, pools % _PAGE]
            head_scores = torch.einsum("hd,pd->hp", q[token].float(), keys)
            scores = (head_scores.mul(_SCALE).relu() * weights[token, :, None]).sum(0)
            selected = torch.topk(scores, k=_TOPK, sorted=False).indices
        raw_slots = (
            selected[:, None] * _POOL + torch.arange(_POOL, device=_DEVICE)
        ).flatten()
        raw_slots = torch.cat(
            (
                raw_slots,
                torch.arange(num_pools * _POOL, causal, device=_DEVICE),
            )
        ).long()
        pages = kv_table[req, raw_slots // _KV_PAGE].long()
        global_slots = pages * _KV_PAGE + raw_slots % _KV_PAGE
        slots[token, : raw_slots.numel()] = global_slots.to(torch.int32)
        lens[token] = raw_slots.numel()
    return slots, lens


def _row_sets(
    result: tuple[torch.Tensor, torch.Tensor],
) -> list[list[int]]:
    slots, lens = result
    return [sorted(slots[row, : int(lens[row])].tolist()) for row in range(len(lens))]


def _plan_kwargs(
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_table: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    workspace_slots, row_starts, row_ends, max_num_pools = _build_physical_plan(
        causal_lens, req_ids, index_table
    )
    return {
        "req_ids": req_ids,
        "causal_lens": causal_lens,
        "pool_workspace_slots": workspace_slots,
        "row_starts": row_starts,
        "row_ends": row_ends,
        "max_num_pools": max_num_pools,
    }


def _common_kwargs() -> dict[str, int | float]:
    return {
        "pool_size": _POOL,
        "page_size": _PAGE,
        "kv_page_size": _KV_PAGE,
        "topk_pools": _TOPK,
        "softmax_scale": _SCALE,
    }


def test_plan_matches_table_and_reference_for_ragged_padded_late_windows() -> None:
    pool_counts = (511, 512, 513, 1024, 2048, 8193)
    tails = (0, 1, 2, 3, 0, 3)
    causal_lens = tuple(
        pools * _POOL + tail for pools, tail in zip(pool_counts, tails, strict=True)
    )
    (
        q,
        cache,
        pooled_k,
        weights,
        positions,
        query_start_loc,
        req_ids,
        causal,
        index_table,
        kv_table,
    ) = _setup_rows(
        causal_lens,
        seed=67,
        page_padding_bytes=64,
        permute_index_pages=True,
    )
    assert cache.stride(0) > cache.shape[1]

    long_row = len(pool_counts) - 1
    late_pool = 8192
    late_page = int(index_table[long_row, late_pool // _PAGE])
    late_row = late_pool % _PAGE
    q[long_row].fill_(1)
    weights[long_row].fill_(1)
    cache[late_page, late_row * _DIM : (late_row + 1) * _DIM].copy_(
        torch.full((_DIM,), _FP8_MAX, dtype=torch.float32, device=_DEVICE)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scale_offset = _PAGE * _DIM + late_row * torch.float32.itemsize
    cache[late_page, scale_offset : scale_offset + torch.float32.itemsize].copy_(
        torch.ones(1, dtype=torch.float32, device=_DEVICE).view(torch.uint8)
    )
    pooled_k[late_page * _PAGE + late_row].fill_(_FP8_MAX)

    expected = _reference(q, pooled_k, weights, causal, req_ids, index_table, kv_table)
    common = _common_kwargs() | {"chunk_pools": 8192}
    out = torch.empty(
        (q.shape[0], _TOPK * _POOL + _POOL - 1),
        dtype=torch.int32,
        device=_DEVICE,
    )
    lens_out = torch.empty(q.shape[0], dtype=torch.int32, device=_DEVICE)
    planned = gluon_kpool_prefill_topk_fp8_gfx950(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **common,
        **_plan_kwargs(causal, req_ids, index_table),
        out=out,
        lens_out=lens_out,
    )
    table_addressed = gluon_kpool_prefill_topk_fp8_gfx950(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **common,
    )
    portable = triton_kpool_prefill_topk(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **common,
    )

    assert planned[0] is out
    assert planned[1] is lens_out
    assert torch.equal(planned[1], expected[1])
    assert _row_sets(planned) == _row_sets(expected)
    assert _row_sets(planned) == _row_sets(table_addressed)
    assert _row_sets(planned) == _row_sets(portable)

    long_slots = planned[0][long_row, : _TOPK * _POOL]
    long_pages = torch.div(long_slots, _KV_PAGE, rounding_mode="floor")
    long_raw_slots = (
        long_pages - kv_table[long_row, 0]
    ) * _KV_PAGE + long_slots % _KV_PAGE
    assert (long_raw_slots >= late_pool * _POOL).any()


def test_plan_masks_fixed_bucket_padding_for_gluon_and_portable() -> None:
    (
        live_q,
        cache,
        pooled_k,
        live_weights,
        live_positions,
        query_start_loc,
        live_req_ids,
        live_causal,
        index_table,
        kv_table,
    ) = _setup_rows((2051, 2052), seed=68)
    q = torch.cat(
        (
            live_q,
            torch.full(
                (2, _HEADS, _DIM),
                float("nan"),
                dtype=torch.bfloat16,
                device=_DEVICE,
            ),
        )
    )
    weights = torch.cat(
        (
            live_weights,
            torch.full(
                (2, _HEADS),
                float("nan"),
                dtype=live_weights.dtype,
                device=_DEVICE,
            ),
        )
    )
    positions = torch.cat(
        (live_positions, torch.zeros(2, dtype=torch.int32, device=_DEVICE))
    )
    req_ids = torch.cat(
        (live_req_ids, torch.zeros(2, dtype=torch.int32, device=_DEVICE))
    )
    causal = torch.cat((live_causal, torch.zeros(2, dtype=torch.int32, device=_DEVICE)))
    expected = _reference(q, pooled_k, weights, causal, req_ids, index_table, kv_table)
    kwargs = _common_kwargs() | _plan_kwargs(causal, req_ids, index_table)

    gluon_result = gluon_kpool_prefill_topk_fp8_gfx950(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **kwargs,
    )
    portable_result = triton_kpool_prefill_topk(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **kwargs,
    )

    assert _row_sets(gluon_result) == _row_sets(expected)
    assert _row_sets(portable_result) == _row_sets(expected)
    assert gluon_result[1].tolist()[-2:] == [0, 0]
    assert portable_result[1].tolist()[-2:] == [0, 0]
    assert gluon_result[0][-2:].eq(-1).all()
    assert portable_result[0][-2:].eq(-1).all()


def test_short_plan_has_stable_logical_order_and_flatkv_boundaries() -> None:
    causal_len = 1024 * _POOL
    (
        q,
        cache,
        _,
        weights,
        positions,
        query_start_loc,
        req_ids,
        causal,
        index_table,
        kv_table,
    ) = _setup_rows(
        (causal_len,),
        seed=69,
        page_padding_bytes=32,
        permute_index_pages=True,
    )
    q.zero_()
    weights = weights.fill_(1).to(torch.float16)
    common = _common_kwargs()
    plan = _plan_kwargs(causal, req_ids, index_table)
    expected_slots = torch.arange(_TOPK * _POOL, dtype=torch.int32, device=_DEVICE)

    first = None
    for _ in range(3):
        actual = gluon_kpool_prefill_topk_fp8_gfx950(
            q,
            cache,
            weights,
            positions,
            query_start_loc,
            index_table,
            kv_table,
            **common,
            **plan,
        )
        assert actual[1].tolist() == [_TOPK * _POOL]
        assert torch.equal(actual[0][0, : _TOPK * _POOL], expected_slots)
        assert (actual[0][0, _TOPK * _POOL :] == -1).all()
        if first is None:
            first = actual[0].clone()
        else:
            assert torch.equal(actual[0], first)


def test_plan_workspace_cap_tiles_rows_while_running_real_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = 3
    causal_len = 513 * _POOL
    (
        q,
        cache,
        _,
        weights,
        _,
        _,
        _,
        _,
        index_table,
        _,
    ) = _setup_rows((causal_len,), seed=75)
    q = q.expand(tokens, -1, -1).contiguous()
    weights = weights.expand(tokens, -1).contiguous()
    req_ids = torch.zeros(tokens, dtype=torch.int32, device=_DEVICE)
    causal = torch.full((tokens,), causal_len, dtype=torch.int32, device=_DEVICE)
    workspace_slots, row_starts, row_ends, max_num_pools = _build_physical_plan(
        causal, req_ids, index_table
    )
    calls = []
    plan_scorer = gluon_kpool_select.gluon_dsa_kpool_prefill_plan_logits_gfx950

    def tracked_plan_scorer(q_tile, *args, **kwargs):
        calls.append(int(q_tile.shape[0]))
        return plan_scorer(q_tile, *args, **kwargs)

    monkeypatch.setattr(
        gluon_kpool_select,
        "gluon_dsa_kpool_prefill_plan_logits_gfx950",
        tracked_plan_scorer,
    )
    window_width = 1024
    one_row_bytes = 3 * window_width * torch.float32.itemsize + 4
    selected = gluon_kpool_select._select_pools_chunked_gluon(
        q,
        cache,
        weights,
        causal,
        req_ids,
        index_table,
        pool_workspace_slots=workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        pool_size=_POOL,
        page_size=_PAGE,
        topk_pools=_TOPK,
        softmax_scale=_SCALE,
        max_num_pools=max_num_pools,
        chunk_pools=8192,
        max_logits_bytes=one_row_bytes,
    )

    assert calls == [1, 1, 1]
    assert selected.shape == (tokens, _TOPK)


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_plan_filters_nonfinite_scores(nonfinite: float) -> None:
    causal_len = 513 * _POOL
    (
        q,
        cache,
        _,
        weights,
        positions,
        query_start_loc,
        req_ids,
        causal,
        index_table,
        kv_table,
    ) = _setup_rows((causal_len,), seed=73, page_padding_bytes=16)
    q.fill_(1)
    weights.fill_(1)
    cache[:, : _PAGE * _DIM].copy_(
        torch.ones(
            (cache.shape[0], _PAGE * _DIM),
            dtype=torch.float32,
            device=_DEVICE,
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    cache[:, _PAGE * _DIM :].copy_(
        torch.full(
            (cache.shape[0], _PAGE),
            nonfinite,
            dtype=torch.float32,
            device=_DEVICE,
        )
        .view(torch.uint8)
        .view(cache.shape[0], -1)
    )

    actual = gluon_kpool_prefill_topk_fp8_gfx950(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **_common_kwargs(),
        **_plan_kwargs(causal, req_ids, index_table),
    )

    assert int(actual[1].item()) == 0
    assert (actual[0] == -1).all()


def test_plan_short_sort_supports_cuda_graph_replay() -> None:
    causal_len = 1024 * _POOL
    (
        q,
        cache,
        _,
        weights,
        positions,
        query_start_loc,
        req_ids,
        causal,
        index_table,
        kv_table,
    ) = _setup_rows((causal_len,), seed=70, page_padding_bytes=32)
    q.zero_()
    weights.fill_(1)
    out = torch.empty((1, _TOPK * _POOL + _POOL - 1), dtype=torch.int32, device=_DEVICE)
    lens_out = torch.empty(1, dtype=torch.int32, device=_DEVICE)
    kwargs = _common_kwargs() | _plan_kwargs(causal, req_ids, index_table)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return gluon_kpool_prefill_topk_fp8_gfx950(
            q,
            cache,
            weights,
            positions,
            query_start_loc,
            index_table,
            kv_table,
            **kwargs,
            out=out,
            lens_out=lens_out,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    expected = torch.arange(_TOPK * _POOL, dtype=torch.int32, device=_DEVICE)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert captured[0] is out
        assert captured[1] is lens_out
        assert int(lens_out.item()) == _TOPK * _POOL
        assert torch.equal(out[0, : _TOPK * _POOL], expected)


def test_public_dispatch_supports_both_addressing_modes_and_geometry_fallbacks() -> (
    None
):
    signature = format_signature(q=dense_tensor_format(torch.bfloat16))
    traits = {
        "index_heads": _HEADS,
        "head_dim": _DIM,
        "pool_size": _POOL,
        "page_size": _PAGE,
        "topk_pools": _TOPK,
        "index_k_format": "fp8_scaled",
        "score_activation": "relu",
        "topk_layout": "global_slots",
        "prefill_plan": True,
    }

    planned = select_kernel("attention", "kpool_prefill_topk", signature, traits=traits)
    traits["prefill_plan"] = False
    table_addressed = select_kernel(
        "attention", "kpool_prefill_topk", signature, traits=traits
    )

    assert planned.name == "gluon_kpool_prefill_topk_fp8_gfx950"
    assert table_addressed.name == "gluon_kpool_prefill_topk_fp8_gfx950"
    for trait, unsupported in (
        ("index_heads", 64),
        ("page_size", 64),
        ("topk_pools", 64),
        ("score_activation", "none"),
    ):
        fallback_traits = traits | {trait: unsupported}
        fallback = select_kernel(
            "attention",
            "kpool_prefill_topk",
            signature,
            traits=fallback_traits,
        )
        assert fallback.name == "triton_kpool_prefill_topk"


def test_public_no_plan_entry_uses_optimized_table_addressing() -> None:
    causal_len = 513 * _POOL + 3
    (
        q,
        cache,
        _,
        weights,
        positions,
        query_start_loc,
        _,
        _,
        index_table,
        kv_table,
    ) = _setup_rows((causal_len,), seed=81, page_padding_bytes=48)
    kwargs = _common_kwargs()

    direct = gluon_kpool_prefill_topk_fp8_gfx950(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **kwargs,
    )
    public = kpool_prefill_topk(
        q,
        cache,
        weights,
        positions,
        query_start_loc,
        index_table,
        kv_table,
        **kwargs,
    )

    assert torch.equal(public[0], direct[0])
    assert torch.equal(public[1], direct[1])
