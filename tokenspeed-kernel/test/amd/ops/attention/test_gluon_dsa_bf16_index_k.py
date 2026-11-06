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

"""GFX950 DSA top-k over a bf16 index-K plane (``[slots, head_dim]``, no scales).

The selection runs through the ``dsa_decode_topk`` / ``dsa_prefill_topk``
facades, so these tests also cover that a bf16 plane selects the bf16 scorers.
The reference scores the same bf16 keys and queries in FP32; only the FP32
summation order differs, so the selected sets match up to near ties at the
k-th score.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsa import (
    torch_dsa_paged_index_topk,
    torch_dsa_workspace_index_topk,
)
from tokenspeed_kernel.ops.attention import dsa as dsa_ops
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD GFX950 is required for the bf16 index-K Gluon scorers",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (  # isort: skip
    gluon_dsa_decode_topk_bf16_gfx950,
    gluon_dsa_prefill_topk_bf16_gfx950,
)

_DEVICE = "cuda"
_PAGE_SIZE = 64
_HEAD_DIM = 128
_TOPK = 2048
_SOFTMAX_SCALE = _HEAD_DIM**-0.5
_MIN_RECALL = 0.999


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device=_DEVICE)
    generator.manual_seed(seed)
    return generator


def _bf16(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        shape, device=_DEVICE, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)


def _weights(
    tokens: int, heads: int, dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    """Head weights of both signs in a row-padded view, like the fused projection."""
    backing = torch.empty((tokens, heads + 16), device=_DEVICE, dtype=dtype)
    weights = backing[:, 16:]
    weights.copy_(
        torch.randn(
            (tokens, heads), device=_DEVICE, dtype=torch.float32, generator=generator
        ).to(dtype)
    )
    return weights


def _paged_requests(
    seq_lens: list[int], num_pages: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffled page tables for ragged requests, padded to the longest one."""
    max_pages = max(-(-length // _PAGE_SIZE) for length in seq_lens)
    pages = torch.randperm(num_pages, device=_DEVICE, generator=generator)
    assert pages.numel() >= len(seq_lens) * max_pages
    block_table = pages[: len(seq_lens) * max_pages].reshape(len(seq_lens), max_pages)
    return (
        torch.tensor(seq_lens, device=_DEVICE, dtype=torch.int32),
        block_table.to(torch.int32).contiguous(),
    )


def _assert_selection(
    actual: torch.Tensor,
    actual_lens: torch.Tensor,
    expected: torch.Tensor,
    expected_lens: torch.Tensor,
    candidates: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Same count, and the same set up to near ties at the k-th score.

    Every selected id must be a candidate scoring at least the reference's
    k-th score less a small FP32-ordering margin, and at least ``_MIN_RECALL``
    of the reference set must be selected.
    """
    torch.testing.assert_close(actual_lens.cpu(), expected_lens.cpu())
    for row, (ids, scores) in enumerate(candidates):
        count = int(expected_lens[row])
        assert (actual[row, count:] == -1).all()
        if count == 0:
            continue
        selected = actual[row, :count].long()
        assert selected.unique().numel() == count
        reference = expected[row, :count].long()
        recall = torch.isin(selected, reference).float().mean().item()
        assert recall >= _MIN_RECALL, f"row {row}: recall {recall:.4f}"
        if count == ids.numel():
            continue
        score_of = torch.full((int(ids.max()) + 1,), float("-inf"), device=ids.device)
        score_of[ids] = scores
        kth = scores.topk(count).values[-1]
        margin = 1.0e-5 * scores.abs().max()
        assert (score_of[selected] >= kth - margin).all(), f"row {row}"


def _decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    index_k: torch.Tensor,
    q_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dsa_ops.dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        batch_invariant=False,
        q_len_per_req=q_len,
        index_k_cache=index_k,
        slot_order="selection",
    )


@pytest.mark.parametrize("q_len", (1, 4))
@pytest.mark.parametrize(
    ("heads", "weight_dtype"),
    ((32, torch.float32), (32, torch.bfloat16), (64, torch.bfloat16)),
)
def test_bf16_plane_decode_matches_fp32_reference(
    q_len: int, heads: int, weight_dtype: torch.dtype
) -> None:
    generator = _generator(10 * q_len + heads)
    # Ragged: shorter than top-k, around one page, and well past top-k.
    seq_lens, block_table = _paged_requests(
        [1500, 65, 2048 + q_len, 7000], num_pages=512, generator=generator
    )
    index_k = _bf16((512 * _PAGE_SIZE, _HEAD_DIM), generator)
    tokens = seq_lens.numel() * q_len
    q = _bf16((tokens, heads, _HEAD_DIM), generator)
    weights = _weights(tokens, heads, weight_dtype, generator)

    actual, actual_lens = _decode(q, weights, seq_lens, block_table, index_k, q_len)
    expected, expected_lens, candidates = torch_dsa_paged_index_topk(
        q.float(),
        weights.float(),
        index_k.float(),
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        q_len_per_req=q_len,
    )

    _assert_selection(actual, actual_lens, expected, expected_lens, candidates)


def _prefill_workspace(
    prefix_lens: list[int], extend_lens: list[int], num_pages: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Workspace rows the way runtime prefill builds them.

    Each request contributes its ``prefix + extend`` positions as consecutive
    workspace rows (slots through a shuffled page table); its query token at
    extend offset ``j`` sees its first ``prefix + j + 1`` rows.
    """
    generator = _generator(seed)
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens, strict=True)]
    _, block_table = _paged_requests(seq_lens, num_pages, generator)
    slots, starts, ends = [], [], []
    base = 0
    for request, (prefix, extend) in enumerate(
        zip(prefix_lens, extend_lens, strict=True)
    ):
        positions = torch.arange(prefix + extend, device=_DEVICE)
        pages = block_table[request].long()[positions // _PAGE_SIZE]
        slots.append(pages * _PAGE_SIZE + positions % _PAGE_SIZE)
        offsets = torch.arange(extend, device=_DEVICE)
        starts.append(torch.full((extend,), base, device=_DEVICE))
        ends.append(base + prefix + offsets + 1)
        base += prefix + extend
    return (
        torch.cat(slots).to(torch.int64).contiguous(),
        torch.cat(starts).to(torch.int32).contiguous(),
        torch.cat(ends).to(torch.int32).contiguous(),
    )


@pytest.mark.parametrize(
    ("heads", "weight_dtype"), ((32, torch.float32), (64, torch.bfloat16))
)
def test_bf16_plane_prefill_matches_fp32_reference(
    heads: int, weight_dtype: torch.dtype
) -> None:
    # Ragged extends: fresh short prompt, chunk crossing top-k, long cached prefix.
    prefix_lens, extend_lens = [0, 1990, 6000], [300, 120, 40]
    workspace_slots, row_starts, row_ends = _prefill_workspace(
        prefix_lens, extend_lens, num_pages=512, seed=heads
    )
    generator = _generator(100 + heads)
    index_k = _bf16((512 * _PAGE_SIZE, _HEAD_DIM), generator)
    tokens = row_starts.numel()
    q = _bf16((tokens, heads, _HEAD_DIM), generator)
    weights = _weights(tokens, heads, weight_dtype, generator)

    actual, actual_lens = dsa_ops.dsa_prefill_topk(
        q,
        weights,
        workspace_slots,
        row_starts,
        row_ends,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        batch_invariant=False,
        index_k_cache=index_k,
        page_size=_PAGE_SIZE,
        # Forces several query-row chunks of logits.
        max_logits_bytes=64 * workspace_slots.numel() * 4,
        slot_order="selection",
    )
    expected, expected_lens, candidates = torch_dsa_workspace_index_topk(
        q.float(),
        weights.float(),
        index_k.float(),
        workspace_slots,
        row_starts,
        row_ends,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
    )

    _assert_selection(actual, actual_lens, expected, expected_lens, candidates)


def _sorted_selection(ids: torch.Tensor) -> torch.Tensor:
    return torch.sort(ids.long(), dim=1).values


@pytest.mark.parametrize("heads", (32, 64))
def test_bf16_plane_prefill_selection_is_independent_of_row_blocking(
    heads: int,
) -> None:
    """Rows scored together select exactly what each row selects alone.

    A scorer program shares key tiles across a block of query rows and a
    chunk of workspace rows. Here blocks straddle request boundaries, spans
    start mid-tile and cross chunk boundaries, and the row count is not a
    multiple of the block, so every row's own masking is exercised.
    """
    prefix_lens, extend_lens = [0, 4000, 9000, 37], [13, 21, 11, 5]
    workspace_slots, row_starts, row_ends = _prefill_workspace(
        prefix_lens, extend_lens, num_pages=1024, seed=7 + heads
    )
    generator = _generator(200 + heads)
    index_k = _bf16((1024 * _PAGE_SIZE, _HEAD_DIM), generator)
    tokens = row_starts.numel()
    q = _bf16((tokens, heads, _HEAD_DIM), generator)
    weights = _weights(tokens, heads, torch.float32, generator)

    def select(max_logits_bytes: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        return dsa_ops.dsa_prefill_topk(
            q,
            weights,
            workspace_slots,
            row_starts,
            row_ends,
            topk=_TOPK,
            softmax_scale=_SOFTMAX_SCALE,
            batch_invariant=False,
            index_k_cache=index_k,
            page_size=_PAGE_SIZE,
            max_logits_bytes=max_logits_bytes,
            slot_order="selection",
        )

    together, together_lens = select(None)
    # A one-row logits budget scores every query row in its own launch.
    alone, alone_lens = select(workspace_slots.numel() * 4)
    torch.testing.assert_close(together_lens, alone_lens)
    assert torch.equal(_sorted_selection(together), _sorted_selection(alone))

    expected, expected_lens, candidates = torch_dsa_workspace_index_topk(
        q.float(),
        weights.float(),
        index_k.float(),
        workspace_slots,
        row_starts,
        row_ends,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
    )
    _assert_selection(together, together_lens, expected, expected_lens, candidates)


@pytest.mark.parametrize("q_len", (1, 2, 3, 5, 6))
def test_bf16_plane_decode_selection_is_independent_of_batching(q_len: int) -> None:
    """Every q_len_per_req selects per request what the batch selects.

    A decode program scores all rows of one request against a chunk of
    positions; lengths here are ragged and end mid-chunk.
    """
    generator = _generator(300 + q_len)
    lengths = [2048 + q_len, 9000, 70, 4097, 2300]
    seq_lens, block_table = _paged_requests(
        lengths, num_pages=1024, generator=generator
    )
    index_k = _bf16((1024 * _PAGE_SIZE, _HEAD_DIM), generator)
    tokens = seq_lens.numel() * q_len
    q = _bf16((tokens, 32, _HEAD_DIM), generator)
    weights = _weights(tokens, 32, torch.bfloat16, generator)

    batched, batched_lens = _decode(q, weights, seq_lens, block_table, index_k, q_len)
    for request in range(seq_lens.numel()):
        rows = slice(request * q_len, (request + 1) * q_len)
        alone, alone_lens = _decode(
            q[rows],
            weights[rows],
            seq_lens[request : request + 1],
            block_table[request : request + 1],
            index_k,
            q_len,
        )
        torch.testing.assert_close(batched_lens[rows], alone_lens)
        assert torch.equal(
            _sorted_selection(batched[rows]), _sorted_selection(alone)
        ), f"request {request}"

    expected, expected_lens, candidates = torch_dsa_paged_index_topk(
        q.float(),
        weights.float(),
        index_k.float(),
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        q_len_per_req=q_len,
    )
    _assert_selection(batched, batched_lens, expected, expected_lens, candidates)


def test_bf16_plane_decode_cuda_graph_replays_changed_inputs() -> None:
    q_len, heads = 4, 32
    generator = _generator(77)
    seq_lens, block_table = _paged_requests(
        [3000, 2500], num_pages=128, generator=generator
    )
    index_k = _bf16((128 * _PAGE_SIZE, _HEAD_DIM), generator)
    q = _bf16((2 * q_len, heads, _HEAD_DIM), generator)
    weights = _weights(2 * q_len, heads, torch.float32, generator)
    out = torch.empty((2 * q_len, _TOPK), device=_DEVICE, dtype=torch.int32)
    lens_out = torch.empty((2 * q_len,), device=_DEVICE, dtype=torch.int32)

    def invoke() -> None:
        dsa_ops.dsa_decode_topk(
            q,
            weights,
            seq_lens,
            block_table,
            page_size=_PAGE_SIZE,
            topk=_TOPK,
            softmax_scale=_SOFTMAX_SCALE,
            batch_invariant=False,
            q_len_per_req=q_len,
            index_k_cache=index_k,
            out=out,
            lens_out=lens_out,
            slot_order="selection",
        )

    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    q.copy_(_bf16(tuple(q.shape), generator))
    weights.mul_(-0.5)
    index_k.copy_(_bf16(tuple(index_k.shape), generator))
    seq_lens.copy_(torch.tensor((1200, 2900), device=_DEVICE, dtype=torch.int32))
    block_table.copy_(torch.roll(block_table, shifts=5, dims=1))
    graph.replay()
    torch.cuda.synchronize()

    expected, expected_lens, candidates = torch_dsa_paged_index_topk(
        q.float(),
        weights.float(),
        index_k.float(),
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        q_len_per_req=q_len,
    )
    _assert_selection(out, lens_out, expected, expected_lens, candidates)


def test_bf16_plane_scorers_do_not_recompile_across_batches() -> None:
    """Batch size, history length, table width, workspace size, and
    q_len_per_req within its power-of-two row bucket are runtime arguments of
    the bf16 scorers, not compile-time ones."""
    heads = 32
    generator = _generator(5)
    index_k = _bf16((512 * _PAGE_SIZE, _HEAD_DIM), generator)
    q = _bf16((64, heads, _HEAD_DIM), generator)
    weights = _weights(64, heads, torch.float32, generator)

    def decode(seq_lens: list[int], q_len: int) -> None:
        lens, table = _paged_requests(seq_lens, 512, generator)
        rows = len(seq_lens) * q_len
        _decode(q[:rows], weights[:rows], lens, table, index_k, q_len)

    def prefill(prefix_lens: list[int], extend_lens: list[int]) -> None:
        slots, starts, ends = _prefill_workspace(
            prefix_lens, extend_lens, num_pages=512, seed=sum(prefix_lens)
        )
        rows = starts.numel()
        dsa_ops.dsa_prefill_topk(
            q[:rows],
            weights[:rows],
            slots,
            starts,
            ends,
            topk=_TOPK,
            softmax_scale=_SOFTMAX_SCALE,
            batch_invariant=False,
            index_k_cache=index_k,
            page_size=_PAGE_SIZE,
            slot_order="selection",
        )

    # Warm every int specialization: q_len 1 or not, widths divisible by 16 or not.
    decode([64 * 16], 1)
    decode([64 * 16 + 1], 4)
    decode([64 * 17], 1)
    decode([64 * 17], 4)
    prefill([0], [16])
    prefill([1], [16])
    with assert_no_triton_compile(
        gluon_dsa_decode_topk_bf16_gfx950, gluon_dsa_prefill_topk_bf16_gfx950
    ):
        for seq_lens, q_len in (
            ([3000, 70, 9000], 1),
            ([5, 2049], 4),
            ([12000] * 2, 1),
            ([800] * 9, 4),
            # 3 rows per request share the 4-row program of q_len 4.
            ([3000, 4100], 3),
        ):
            decode(seq_lens, q_len)
        for prefix_lens, extend_lens in (
            ([0, 300], [20, 7]),
            ([4000], [33]),
            ([10, 2000, 5], [3, 1, 60]),
        ):
            prefill(prefix_lens, extend_lens)
    torch.cuda.synchronize()
