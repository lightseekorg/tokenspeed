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

"""GFX950 Gluon scoring coverage for GLM pooled index keys."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 (GFX950) is required for pooled Gluon DSA scorer tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (  # isort: skip
    gluon_dsa_kpool_prefill_logits_gfx950,
    gluon_dsa_kpool_prefill_plan_logits_gfx950,
)

_DEVICE = "cuda"
_HEADS = 32
_HEAD_DIM = 128
_PAGE_SIZE = 16
_POOL_SIZE = 4
_ROW_BYTES = _HEAD_DIM + 4
_SOFTMAX_SCALE = _HEAD_DIM**-0.5
_FP8_MAX = 448.0


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device=_DEVICE)
    generator.manual_seed(seed)
    return generator


def _packed_cache(
    num_pages: int,
    *,
    padding_bytes: int,
    as_rows: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = (
        torch.randn(
            (num_pages, _PAGE_SIZE, _HEAD_DIM),
            device=_DEVICE,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.15
    )
    scales = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / _FP8_MAX
    fp8 = (values / scales).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    page_bytes = _PAGE_SIZE * _ROW_BYTES
    backing = torch.zeros(
        (num_pages, page_bytes + padding_bytes),
        device=_DEVICE,
        dtype=torch.uint8,
    )
    cache = backing[:, :page_bytes]
    cache[:, : _PAGE_SIZE * _HEAD_DIM].copy_(fp8.view(torch.uint8).view(num_pages, -1))
    cache[:, _PAGE_SIZE * _HEAD_DIM :].copy_(
        scales.contiguous().view(torch.uint8).view(num_pages, -1)
    )
    if as_rows:
        cache = cache.view(num_pages, _PAGE_SIZE, _ROW_BYTES)
    return cache, fp8.float() * scales


def _pack_explicit_cache(
    fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    padding_bytes: int = 0,
) -> torch.Tensor:
    assert fp8.dtype == torch.float8_e4m3fn
    assert fp8.shape[1:] == (_PAGE_SIZE, _HEAD_DIM)
    assert scales.dtype == torch.float32
    assert scales.shape == fp8.shape[:2]
    page_bytes = _PAGE_SIZE * _ROW_BYTES
    backing = torch.zeros(
        (fp8.shape[0], page_bytes + padding_bytes),
        device=_DEVICE,
        dtype=torch.uint8,
    )
    cache = backing[:, :page_bytes]
    cache[:, : _PAGE_SIZE * _HEAD_DIM].copy_(
        fp8.contiguous().view(torch.uint8).view(fp8.shape[0], -1)
    )
    cache[:, _PAGE_SIZE * _HEAD_DIM :].copy_(
        scales.contiguous().view(torch.uint8).view(fp8.shape[0], -1)
    )
    return cache


def _strided_inputs(
    tokens: int,
    weight_dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_values = torch.randn(
        (tokens, _HEADS, _HEAD_DIM + 8),
        device=_DEVICE,
        dtype=torch.float32,
        generator=generator,
    )
    q_backing = torch.empty_like(q_values, dtype=torch.bfloat16)
    q = q_backing[..., :_HEAD_DIM]
    q.copy_((q_values[..., :_HEAD_DIM] * 0.15).to(torch.bfloat16))
    weight_values = torch.randn(
        (tokens, _HEADS + 5),
        device=_DEVICE,
        dtype=torch.float32,
        generator=generator,
    )
    weight_backing = torch.empty_like(weight_values, dtype=weight_dtype)
    weights = weight_backing[:, :_HEADS]
    weight_values[:, 0].abs_().add_(0.1)
    weight_values[:, 1].abs_().add_(0.1).neg_()
    weights.copy_(weight_values[:, :_HEADS].to(weight_dtype))
    assert q.stride(-1) == weights.stride(-1) == 1
    assert not q.is_contiguous() and not weights.is_contiguous()
    return q, weights


def _reference(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    *,
    pool_offset: int,
    window_cols: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    capacity = block_table.shape[1] * _PAGE_SIZE
    row_ends = (
        causal_lens.clamp_min(0)
        .div(_POOL_SIZE, rounding_mode="floor")
        .clamp_max(capacity)
        .sub(pool_offset)
        .clamp(min=0, max=window_cols)
        .to(torch.int32)
    )
    expected = []
    for row, end_tensor in enumerate(row_ends):
        end = int(end_tensor.item())
        pools = pool_offset + torch.arange(end, device=_DEVICE)
        request = int(req_ids[row].item())
        pages = block_table[request, pools // _PAGE_SIZE].long()
        selected_keys = keys[pages, pools.remainder(_PAGE_SIZE)]
        head_scores = torch.einsum("hd,pd->hp", q[row].float(), selected_keys.float())
        expected.append(
            (head_scores.relu() * weights[row, :, None]).sum(dim=0) * _SOFTMAX_SCALE
        )
    return expected, row_ends


@pytest.mark.parametrize(("padding_bytes", "as_rows"), ((0, False), (192, True)))
@pytest.mark.parametrize(("pool_offset", "window_cols"), ((13, 173), (240, 64)))
@pytest.mark.parametrize("weight_dtype", (torch.bfloat16, torch.float32))
def test_kpool_prefill_logits_match_weighted_relu_and_local_window(
    padding_bytes: int,
    as_rows: bool,
    pool_offset: int,
    window_cols: int,
    weight_dtype: torch.dtype,
) -> None:
    generator = _generator(701 + padding_bytes)
    requests = 2
    table_cols = 16
    num_pages = requests * table_cols
    cache, keys = _packed_cache(
        num_pages,
        padding_bytes=padding_bytes,
        as_rows=as_rows,
        generator=generator,
    )
    block_table = torch.stack(
        (
            torch.arange(table_cols - 1, -1, -1, device=_DEVICE),
            torch.arange(table_cols, num_pages, device=_DEVICE),
        )
    ).to(torch.int32)
    pool_counts = torch.tensor((12, 13, 14, 190, 300), device=_DEVICE)
    causal_lens = (pool_counts * _POOL_SIZE + torch.arange(5, device=_DEVICE) % 4).to(
        torch.int32
    )
    req_ids = torch.tensor((1, 0, 1, 0, 1), device=_DEVICE, dtype=torch.int32)
    q, weights = _strided_inputs(causal_lens.numel(), weight_dtype, generator)
    out = torch.full(
        (q.shape[0], window_cols), float("nan"), device=_DEVICE, dtype=torch.float32
    )
    row_ends = torch.full((q.shape[0],), -1, device=_DEVICE, dtype=torch.int32)

    actual, actual_ends = gluon_dsa_kpool_prefill_logits_gfx950(
        q,
        cache,
        weights,
        causal_lens,
        req_ids,
        block_table,
        pool_size=_POOL_SIZE,
        page_size=_PAGE_SIZE,
        pool_offset=pool_offset,
        window_cols=window_cols,
        softmax_scale=_SOFTMAX_SCALE,
        out=out,
        row_ends_out=row_ends,
    )
    expected, expected_ends = _reference(
        q,
        keys,
        weights,
        causal_lens,
        req_ids,
        block_table,
        pool_offset=pool_offset,
        window_cols=window_cols,
    )

    assert actual is out
    assert actual_ends is row_ends
    torch.testing.assert_close(actual_ends.cpu(), expected_ends.cpu())
    for row, end_tensor in enumerate(expected_ends):
        end = int(end_tensor.item())
        torch.testing.assert_close(
            actual[row, :end], expected[row], rtol=2.0e-2, atol=2.0e-2
        )
        assert actual[row, end:].isnan().all()


@pytest.mark.parametrize("weight_dtype", (torch.bfloat16, torch.float32))
def test_kpool_plan_logits_match_ragged_physical_slots_and_padded_stride(
    weight_dtype: torch.dtype,
) -> None:
    generator = _generator(911)
    requests = 2
    table_cols = 24
    num_pages = requests * table_cols
    cache, keys = _packed_cache(
        num_pages,
        padding_bytes=192,
        as_rows=False,
        generator=generator,
    )
    block_table = torch.stack(
        (
            torch.arange(table_cols - 1, -1, -1, device=_DEVICE),
            torch.arange(num_pages - 1, table_cols - 1, -1, device=_DEVICE),
        )
    ).to(torch.int32)
    req_ids = torch.tensor((1, 0, 1, 0, 1), device=_DEVICE, dtype=torch.int32)
    pool_counts = torch.tensor((13, 14, 190, 220, 300), device=_DEVICE)
    causal_lens = (pool_counts * _POOL_SIZE + torch.arange(5, device=_DEVICE) % 4).to(
        torch.int32
    )
    q, weights = _strided_inputs(causal_lens.numel(), weight_dtype, generator)

    request_counts = (220, 300)
    workspace_parts = []
    request_starts = []
    workspace_start = 0
    for req, count in enumerate(request_counts):
        request_starts.append(workspace_start)
        pools = torch.arange(count, device=_DEVICE, dtype=torch.int64)
        pages = block_table[req, pools // _PAGE_SIZE].to(torch.int64)
        workspace_parts.append(pages * _PAGE_SIZE + pools % _PAGE_SIZE)
        workspace_start += count
    workspace_slots = torch.cat(workspace_parts).contiguous()
    row_starts = torch.tensor(
        [request_starts[req] for req in req_ids.tolist()],
        device=_DEVICE,
        dtype=torch.int32,
    )
    row_ends = row_starts + causal_lens // _POOL_SIZE
    pool_offset = 13
    window_cols = 173
    out = torch.full(
        (q.shape[0], window_cols), float("nan"), device=_DEVICE, dtype=torch.float32
    )
    local_ends = torch.full((q.shape[0],), -1, device=_DEVICE, dtype=torch.int32)

    actual, actual_ends = gluon_dsa_kpool_prefill_plan_logits_gfx950(
        q,
        cache,
        weights,
        workspace_slots,
        row_starts,
        row_ends,
        pool_size=_POOL_SIZE,
        page_size=_PAGE_SIZE,
        pool_offset=pool_offset,
        window_cols=window_cols,
        softmax_scale=_SOFTMAX_SCALE,
        out=out,
        row_ends_out=local_ends,
    )
    expected, expected_ends = _reference(
        q,
        keys,
        weights,
        causal_lens,
        req_ids,
        block_table,
        pool_offset=pool_offset,
        window_cols=window_cols,
    )

    assert actual is out
    assert actual_ends is local_ends
    torch.testing.assert_close(actual_ends.cpu(), expected_ends.cpu())
    for row, end_tensor in enumerate(expected_ends):
        end = int(end_tensor.item())
        torch.testing.assert_close(
            actual[row, :end], expected[row], rtol=2.0e-2, atol=2.0e-2
        )
        assert actual[row, end:].isnan().all()


def test_kpool_ordered_head_fold_matches_sequential_signed_reference() -> None:
    q = torch.zeros((1, _HEADS, _HEAD_DIM), device=_DEVICE, dtype=torch.bfloat16)
    q[:, :, 0] = 1.0
    weights = torch.zeros((1, _HEADS), device=_DEVICE, dtype=torch.float32)
    weights[0, :4] = torch.tensor(
        (1.0e8, 1.0, -1.0e8, 1.0),
        device=_DEVICE,
        dtype=torch.float32,
    )
    fp8 = torch.zeros(
        (1, _PAGE_SIZE, _HEAD_DIM),
        device=_DEVICE,
        dtype=torch.float8_e4m3fn,
    )
    fp8[:, :, 0] = 1.0
    scales = torch.ones((1, _PAGE_SIZE), device=_DEVICE, dtype=torch.float32)
    cache = _pack_explicit_cache(fp8, scales)
    causal_lens = torch.full(
        (1,), _PAGE_SIZE * _POOL_SIZE, device=_DEVICE, dtype=torch.int32
    )
    req_ids = torch.zeros((1,), device=_DEVICE, dtype=torch.int32)
    block_table = torch.zeros((1, 1), device=_DEVICE, dtype=torch.int32)

    actual, row_ends = gluon_dsa_kpool_prefill_logits_gfx950(
        q,
        cache,
        weights,
        causal_lens,
        req_ids,
        block_table,
        pool_size=_POOL_SIZE,
        page_size=_PAGE_SIZE,
        pool_offset=0,
        window_cols=_PAGE_SIZE,
        softmax_scale=1.0,
        ordered_head_fold=True,
    )
    expected = torch.zeros((_PAGE_SIZE,), device=_DEVICE, dtype=torch.float32)
    for head in range(_HEADS):
        expected += weights[0, head]

    assert int(row_ends.item()) == _PAGE_SIZE
    assert torch.equal(actual[0], expected)
    assert torch.equal(expected, torch.ones_like(expected))


def test_kpool_prefill_logits_empty_page_table_returns_zero_bounds() -> None:
    q = torch.empty((3, _HEADS, _HEAD_DIM), device=_DEVICE, dtype=torch.bfloat16)
    weights = torch.empty((3, _HEADS), device=_DEVICE, dtype=torch.float32)
    cache = torch.empty((0, _PAGE_SIZE * _ROW_BYTES), device=_DEVICE, dtype=torch.uint8)
    causal_lens = torch.full((3,), 4096, device=_DEVICE, dtype=torch.int32)
    req_ids = torch.zeros((3,), device=_DEVICE, dtype=torch.int32)
    block_table = torch.empty((1, 0), device=_DEVICE, dtype=torch.int32)
    out = torch.full((3, 512), float("nan"), device=_DEVICE, dtype=torch.float32)

    actual, row_ends = gluon_dsa_kpool_prefill_logits_gfx950(
        q,
        cache,
        weights,
        causal_lens,
        req_ids,
        block_table,
        pool_size=_POOL_SIZE,
        page_size=_PAGE_SIZE,
        pool_offset=0,
        window_cols=512,
        softmax_scale=_SOFTMAX_SCALE,
        out=out,
    )

    assert actual is out
    assert actual.isnan().all()
    assert (row_ends == 0).all()


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_kpool_prefill_logits_preserve_nonfinite_visible_scores(
    nonfinite: float,
) -> None:
    q = torch.ones((1, _HEADS, _HEAD_DIM), device=_DEVICE, dtype=torch.bfloat16)
    weights = torch.ones((1, _HEADS), device=_DEVICE, dtype=torch.float32)
    page_bytes = _PAGE_SIZE * _ROW_BYTES
    backing = torch.zeros((1, page_bytes + 192), device=_DEVICE, dtype=torch.uint8)
    cache = backing[:, :page_bytes]
    cache[:, : _PAGE_SIZE * _HEAD_DIM].copy_(
        torch.ones((1, _PAGE_SIZE * _HEAD_DIM), device=_DEVICE, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    cache[:, _PAGE_SIZE * _HEAD_DIM :].copy_(
        torch.full(
            (1, _PAGE_SIZE), nonfinite, device=_DEVICE, dtype=torch.float32
        ).view(torch.uint8)
    )
    metadata = torch.tensor((_PAGE_SIZE * _POOL_SIZE,), device=_DEVICE).to(torch.int32)
    req_ids = torch.zeros((1,), device=_DEVICE, dtype=torch.int32)
    block_table = torch.zeros((1, 1), device=_DEVICE, dtype=torch.int32)

    actual, row_ends = gluon_dsa_kpool_prefill_logits_gfx950(
        q,
        cache,
        weights,
        metadata,
        req_ids,
        block_table,
        pool_size=_POOL_SIZE,
        page_size=_PAGE_SIZE,
        pool_offset=0,
        window_cols=_PAGE_SIZE,
        softmax_scale=_SOFTMAX_SCALE,
    )

    assert int(row_ends.item()) == _PAGE_SIZE
    if torch.isnan(torch.tensor(nonfinite)):
        assert actual.isnan().all()
    else:
        assert torch.isposinf(actual).all()


def test_kpool_ordered_head_fold_rejects_non_bool_flag() -> None:
    with pytest.raises(TypeError, match="ordered_head_fold must be bool"):
        gluon_dsa_kpool_prefill_logits_gfx950(
            None,
            None,
            None,
            None,
            None,
            None,
            pool_size=_POOL_SIZE,
            page_size=_PAGE_SIZE,
            pool_offset=0,
            window_cols=512,
            softmax_scale=_SOFTMAX_SCALE,
            ordered_head_fold=1,
        )


@pytest.mark.parametrize(
    ("pool_size", "page_size", "match"),
    ((2, _PAGE_SIZE, "pool_size=4"), (_POOL_SIZE, 64, "page_size=16")),
)
def test_kpool_prefill_logits_reject_unsupported_geometry(
    pool_size: int,
    page_size: int,
    match: str,
) -> None:
    q = torch.empty((1, _HEADS, _HEAD_DIM), device=_DEVICE, dtype=torch.bfloat16)
    weights = torch.empty((1, _HEADS), device=_DEVICE, dtype=torch.float32)
    cache = torch.empty((1, _PAGE_SIZE * _ROW_BYTES), device=_DEVICE, dtype=torch.uint8)
    metadata = torch.zeros((1,), device=_DEVICE, dtype=torch.int32)
    block_table = torch.zeros((1, 1), device=_DEVICE, dtype=torch.int32)

    with pytest.raises(ValueError, match=match):
        gluon_dsa_kpool_prefill_logits_gfx950(
            q,
            cache,
            weights,
            metadata,
            metadata,
            block_table,
            pool_size=pool_size,
            page_size=page_size,
            pool_offset=0,
            window_cols=512,
            softmax_scale=_SOFTMAX_SCALE,
        )
