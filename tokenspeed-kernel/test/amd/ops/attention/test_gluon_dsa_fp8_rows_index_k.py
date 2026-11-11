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

"""GFX950 DSA top-k of E4M3 queries over fp8_scaled index-K pages.

Each page holds 64 rows of 128 E4M3 codes, then their 64 FP32 scales; each
query row and head has one FP32 scale. The selection runs through the
``dsa_decode_topk`` / ``dsa_prefill_topk`` facades, so these tests also cover
that E4M3 queries over fp8_scaled pages reach the multi-row scorers. The
reference scores the dequantized queries and keys in FP32; E4M3 products are
exact in FP32, so only the summation order differs and the selected sets
match up to near ties at the k-th score.
"""

from __future__ import annotations

from unittest.mock import patch

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
        "AMD GFX950 is required for the fp8_scaled multi-row Gluon scorers",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (  # isort: skip
    gluon_dsa_decode_topk_fp8_rows_gfx950,
    gluon_dsa_prefill_topk_fp8_rows_gfx950,
)

_DEVICE = "cuda"
_PAGE_SIZE = 64
_HEAD_DIM = 128
_PAGE_BYTES = _PAGE_SIZE * (_HEAD_DIM + 4)
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


def _fp8_query(
    shape: tuple[int, ...], generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """E4M3 ``[tokens, heads, dim]`` queries, their ``[tokens, heads]`` scales,
    and the dequantized FP32 queries."""
    values = _bf16(shape, generator).float()
    scales = values.abs().amax(dim=-1).clamp_min(1.0e-10) / 448.0
    codes = (values / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return codes, scales.contiguous(), codes.float() * scales[..., None]


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


def _fp8_pages(
    num_pages: int, generator: torch.Generator, page_pad: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp8_scaled plane and its dequantized ``[slots, head_dim]`` keys.

    ``page_pad == 0`` gives the packed ``[slots, 132]`` slot matrix; otherwise a
    ``[pages, page_bytes]`` page view whose pages sit ``page_pad`` bytes apart.
    Row scales vary over two decades so the scale multiply matters.
    """
    slots = num_pages * _PAGE_SIZE
    keys = torch.randn(
        (slots, _HEAD_DIM), device=_DEVICE, dtype=torch.float32, generator=generator
    )
    keys *= torch.exp(2.3 * torch.rand((slots, 1), device=_DEVICE, generator=generator))
    scales = keys.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10) / 448.0
    codes = (keys / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    backing = torch.zeros(
        (num_pages, _PAGE_BYTES + page_pad), device=_DEVICE, dtype=torch.uint8
    )
    pages = backing[:, :_PAGE_BYTES]
    pages[:, : _PAGE_SIZE * _HEAD_DIM] = codes.view(torch.uint8).reshape(num_pages, -1)
    pages[:, _PAGE_SIZE * _HEAD_DIM :] = (
        scales.reshape(num_pages, _PAGE_SIZE).contiguous().view(torch.uint8)
    )
    plane = pages.reshape(slots, -1) if page_pad == 0 else pages
    return plane, codes.float() * scales


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
    """Same count, and the same set up to near ties at the k-th score."""
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
    q_scales: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    plane: torch.Tensor,
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
        index_k_cache=plane,
        q_scales=q_scales,
        slot_order="selection",
    )


def _prefill(
    q: torch.Tensor,
    q_scales: torch.Tensor,
    weights: torch.Tensor,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    plane: torch.Tensor,
    max_logits_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    slots, starts, ends = workspace
    return dsa_ops.dsa_prefill_topk(
        q,
        weights,
        slots,
        starts,
        ends,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        batch_invariant=False,
        index_k_cache=plane,
        page_size=_PAGE_SIZE,
        max_logits_bytes=max_logits_bytes,
        q_scales=q_scales,
        slot_order="selection",
    )


def _expected_decode(q, weights, keys, seq_lens, block_table, q_len):
    return torch_dsa_paged_index_topk(
        q,
        weights.float(),
        keys,
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
        q_len_per_req=q_len,
    )


def _expected_prefill(q, weights, keys, workspace):
    slots, starts, ends = workspace
    return torch_dsa_workspace_index_topk(
        q,
        weights.float(),
        keys,
        slots,
        starts,
        ends,
        topk=_TOPK,
        softmax_scale=_SOFTMAX_SCALE,
    )


@pytest.mark.parametrize("q_len", (1, 4))
@pytest.mark.parametrize(
    ("heads", "weight_dtype", "page_pad"),
    (
        (32, torch.float32, 0),
        (32, torch.bfloat16, 512),
        (64, torch.bfloat16, 0),
    ),
)
def test_fp8_pages_decode_matches_fp32_reference(
    q_len: int, heads: int, weight_dtype: torch.dtype, page_pad: int
) -> None:
    generator = _generator(10 * q_len + heads + page_pad)
    # Ragged: shorter than top-k, around one page, and well past top-k.
    seq_lens, block_table = _paged_requests(
        [1500, 65, 2048 + q_len, 7000], num_pages=512, generator=generator
    )
    plane, keys = _fp8_pages(512, generator, page_pad)
    tokens = seq_lens.numel() * q_len
    q, q_scales, q_ref = _fp8_query((tokens, heads, _HEAD_DIM), generator)
    weights = _weights(tokens, heads, weight_dtype, generator)

    actual, actual_lens = _decode(
        q, q_scales, weights, seq_lens, block_table, plane, q_len
    )

    _assert_selection(
        actual,
        actual_lens,
        *_expected_decode(q_ref, weights, keys, seq_lens, block_table, q_len),
    )


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
    ("heads", "weight_dtype", "page_pad"),
    ((32, torch.float32, 0), (32, torch.bfloat16, 512), (64, torch.bfloat16, 0)),
)
def test_fp8_pages_prefill_matches_fp32_reference(
    heads: int, weight_dtype: torch.dtype, page_pad: int
) -> None:
    # Ragged extends: fresh short prompt, chunk crossing top-k, long cached prefix.
    workspace = _prefill_workspace(
        [0, 1990, 6000], [300, 120, 40], num_pages=512, seed=heads + page_pad
    )
    generator = _generator(100 + heads + page_pad)
    plane, keys = _fp8_pages(512, generator, page_pad)
    tokens = workspace[1].numel()
    q, q_scales, q_ref = _fp8_query((tokens, heads, _HEAD_DIM), generator)
    weights = _weights(tokens, heads, weight_dtype, generator)

    # A small logits budget forces several query-row chunks.
    actual, actual_lens = _prefill(
        q,
        q_scales,
        weights,
        workspace,
        plane,
        max_logits_bytes=64 * workspace[0].numel() * 4,
    )

    _assert_selection(
        actual, actual_lens, *_expected_prefill(q_ref, weights, keys, workspace)
    )


def _sorted_selection(ids: torch.Tensor) -> torch.Tensor:
    return torch.sort(ids.long(), dim=1).values


@pytest.mark.parametrize("heads", (32, 64))
def test_fp8_pages_prefill_selection_is_independent_of_row_blocking(
    heads: int,
) -> None:
    """Rows scored together select exactly what each row selects alone.

    Blocks of query rows straddle request boundaries, spans start mid-tile
    and cross chunk boundaries, and the row count is not a multiple of the
    block, so every row's own masking and scale lookup is exercised.
    """
    workspace = _prefill_workspace(
        [0, 4000, 9000, 37], [13, 21, 11, 5], num_pages=1024, seed=7 + heads
    )
    generator = _generator(200 + heads)
    plane, keys = _fp8_pages(1024, generator)
    tokens = workspace[1].numel()
    q, q_scales, q_ref = _fp8_query((tokens, heads, _HEAD_DIM), generator)
    weights = _weights(tokens, heads, torch.float32, generator)

    together, together_lens = _prefill(q, q_scales, weights, workspace, plane)
    # A one-row logits budget scores every query row in its own launch.
    alone, alone_lens = _prefill(
        q,
        q_scales,
        weights,
        workspace,
        plane,
        max_logits_bytes=workspace[0].numel() * 4,
    )

    torch.testing.assert_close(together_lens, alone_lens)
    assert torch.equal(_sorted_selection(together), _sorted_selection(alone))
    _assert_selection(
        together, together_lens, *_expected_prefill(q_ref, weights, keys, workspace)
    )


@pytest.mark.parametrize("q_len", (1, 2, 3, 5, 6))
def test_fp8_pages_decode_selection_is_independent_of_batching(q_len: int) -> None:
    """Every q_len_per_req selects per request what the batch selects."""
    generator = _generator(300 + q_len)
    seq_lens, block_table = _paged_requests(
        [2048 + q_len, 9000, 70, 4097, 2300], num_pages=1024, generator=generator
    )
    plane, keys = _fp8_pages(1024, generator)
    tokens = seq_lens.numel() * q_len
    q, q_scales, q_ref = _fp8_query((tokens, 32, _HEAD_DIM), generator)
    weights = _weights(tokens, 32, torch.bfloat16, generator)

    batched, batched_lens = _decode(
        q, q_scales, weights, seq_lens, block_table, plane, q_len
    )
    for request in range(seq_lens.numel()):
        rows = slice(request * q_len, (request + 1) * q_len)
        alone, alone_lens = _decode(
            q[rows],
            q_scales[rows],
            weights[rows],
            seq_lens[request : request + 1],
            block_table[request : request + 1],
            plane,
            q_len,
        )
        torch.testing.assert_close(batched_lens[rows], alone_lens)
        assert torch.equal(
            _sorted_selection(batched[rows]), _sorted_selection(alone)
        ), f"request {request}"

    _assert_selection(
        batched,
        batched_lens,
        *_expected_decode(q_ref, weights, keys, seq_lens, block_table, q_len),
    )


@pytest.mark.parametrize("splits", (1, 3, 64))
def test_fp8_pages_decode_selection_is_independent_of_programs_per_request(
    monkeypatch, splits: int
) -> None:
    """However many programs share a request's chunks, decode selects what
    one program per chunk of a table as wide as max-model-len selects."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import sparse_mla

    q_len = 4
    generator = _generator(500 + splits)
    seq_lens, table = _paged_requests(
        [9000, 70, 4097, 2300], num_pages=1024, generator=generator
    )
    block_table = torch.zeros((table.shape[0], 1024), device=_DEVICE, dtype=torch.int32)
    block_table[:, : table.shape[1]] = table
    plane, keys = _fp8_pages(1024, generator)
    tokens = seq_lens.numel() * q_len
    q, q_scales, q_ref = _fp8_query((tokens, 32, _HEAD_DIM), generator)
    weights = _weights(tokens, 32, torch.bfloat16, generator)

    def decode(programs) -> tuple[torch.Tensor, torch.Tensor]:
        monkeypatch.setattr(sparse_mla, "_fp8_rows_decode_splits", programs)
        return _decode(q, q_scales, weights, seq_lens, block_table, plane, q_len)

    per_chunk, per_chunk_lens = decode(lambda requests, chunks, *_: chunks)
    shared, shared_lens = decode(lambda *_: splits)

    torch.testing.assert_close(shared_lens, per_chunk_lens)
    assert torch.equal(_sorted_selection(shared), _sorted_selection(per_chunk))
    _assert_selection(
        shared,
        shared_lens,
        *_expected_decode(q_ref, weights, keys, seq_lens, block_table, q_len),
    )


def test_fp8_pages_decode_programs_follow_the_gpu_not_the_table_width() -> None:
    """One program per chunk up to 16 a compute unit; past that a request
    gets at most twice its share of one residency wave."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import sparse_mla

    device = torch.device(_DEVICE)
    units = sparse_mla._device_compute_units(torch.cuda.current_device())
    for requests in (1, 4, 16, 32):
        wave = -(-4 * units // requests)
        for chunks in (8, 256, 1024, 4096, 16384):
            splits = sparse_mla._fp8_rows_decode_splits(requests, chunks, 4, device)
            if requests * chunks <= 16 * units:
                assert splits == chunks
            else:
                assert splits < chunks
                assert splits <= 2 * wave


def test_fp8_pages_launch_tuning_keeps_the_selection_bitwise(monkeypatch) -> None:
    """The FP8-query launch shapes (key tile, chunk, stages, waves) only
    reschedule the scores: prefill spans crossing 2048- and 4096-slot chunks
    and every decode row select exactly what the untuned shapes select."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import sparse_mla

    heads = 32
    workspace = _prefill_workspace(
        [0, 9000, 4500], [40, 24, 17], num_pages=512, seed=11
    )
    generator = _generator(411)
    plane, _ = _fp8_pages(512, generator)
    tokens = workspace[1].numel()
    q, q_scales, _ = _fp8_query((tokens, heads, _HEAD_DIM), generator)
    weights = _weights(tokens, heads, torch.bfloat16, generator)
    seq_lens, block_table = _paged_requests(
        [9000, 2052, 6100], num_pages=512, generator=generator
    )

    def run() -> tuple[list[torch.Tensor], tuple[int, ...]]:
        with (
            patch.object(
                gluon_dsa_prefill_topk_fp8_rows_gfx950,
                "run",
                wraps=gluon_dsa_prefill_topk_fp8_rows_gfx950.run,
            ) as prefill_launch,
            patch.object(
                gluon_dsa_decode_topk_fp8_rows_gfx950,
                "run",
                wraps=gluon_dsa_decode_topk_fp8_rows_gfx950.run,
            ) as decode_launch,
        ):
            prefill_ids, prefill_lens = _prefill(q, q_scales, weights, workspace, plane)
            decode_ids, decode_lens = _decode(
                q[:12], q_scales[:12], weights[:12], seq_lens, block_table, plane, 4
            )
        prefill_kwargs = prefill_launch.call_args.kwargs
        decode_kwargs = decode_launch.call_args.kwargs
        shapes = (
            prefill_kwargs["BLOCK_N"],
            prefill_kwargs["CHUNK_N"],
            decode_kwargs["NUM_STAGES"],
            decode_kwargs["waves_per_eu"],
        )
        return [prefill_ids, prefill_lens, decode_ids, decode_lens], shapes

    tuned, tuned_shapes = run()
    monkeypatch.setattr(sparse_mla, "_FP8_ROWS_PREFILL_TILES", {})
    monkeypatch.setattr(
        sparse_mla, "_FP8_ROWS_DECODE_NUM_STAGES", sparse_mla._BF16_DECODE_NUM_STAGES
    )
    monkeypatch.setattr(sparse_mla, "_FP8_ROWS_DECODE_WAVES_PER_EU", 2)
    untuned, untuned_shapes = run()

    assert tuned_shapes == (128, 2048, 2, 4)
    assert untuned_shapes == (64, 4096, 3, 2)
    for actual, expected in zip(tuned, untuned, strict=True):
        if actual.dim() == 2:
            actual, expected = _sorted_selection(actual), _sorted_selection(expected)
        assert torch.equal(actual, expected)


def test_fp8_pages_decode_cuda_graph_replays_changed_inputs() -> None:
    q_len, heads = 4, 32
    generator = _generator(77)
    seq_lens, block_table = _paged_requests(
        [3000, 2500], num_pages=128, generator=generator
    )
    plane, _ = _fp8_pages(128, generator)
    q, q_scales, _ = _fp8_query((2 * q_len, heads, _HEAD_DIM), generator)
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
            index_k_cache=plane,
            q_scales=q_scales,
            out=out,
            lens_out=lens_out,
            slot_order="selection",
        )

    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    new_q, new_scales, q_ref = _fp8_query(tuple(q.shape), generator)
    q.copy_(new_q)
    q_scales.copy_(new_scales)
    weights.mul_(-0.5)
    new_plane, keys = _fp8_pages(128, generator)
    plane.copy_(new_plane)
    seq_lens.copy_(torch.tensor((1200, 2900), device=_DEVICE, dtype=torch.int32))
    block_table.copy_(torch.roll(block_table, shifts=5, dims=1))
    graph.replay()
    torch.cuda.synchronize()

    _assert_selection(
        out,
        lens_out,
        *_expected_decode(q_ref, weights, keys, seq_lens, block_table, q_len),
    )


def test_fp8_pages_scorers_do_not_recompile_across_batches() -> None:
    """Batch size, history length, table width, workspace size, and
    q_len_per_req within its power-of-two row bucket are runtime arguments."""
    heads = 32
    generator = _generator(5)
    plane, _ = _fp8_pages(512, generator)
    q, q_scales, _ = _fp8_query((64, heads, _HEAD_DIM), generator)
    weights = _weights(64, heads, torch.float32, generator)

    def decode(seq_lens: list[int], q_len: int) -> None:
        lens, table = _paged_requests(seq_lens, 512, generator)
        rows = len(seq_lens) * q_len
        _decode(q[:rows], q_scales[:rows], weights[:rows], lens, table, plane, q_len)

    def prefill(prefix_lens: list[int], extend_lens: list[int]) -> None:
        workspace = _prefill_workspace(
            prefix_lens, extend_lens, num_pages=512, seed=sum(prefix_lens)
        )
        rows = workspace[1].numel()
        _prefill(q[:rows], q_scales[:rows], weights[:rows], workspace, plane)

    # Warm every int specialization: q_len 1 or not, widths divisible by 16 or not.
    decode([64 * 16], 1)
    decode([64 * 16 + 1], 4)
    decode([64 * 17], 1)
    decode([64 * 17], 4)
    prefill([0], [16])
    prefill([1], [16])
    with assert_no_triton_compile(
        gluon_dsa_decode_topk_fp8_rows_gfx950, gluon_dsa_prefill_topk_fp8_rows_gfx950
    ):
        for seq_lens, q_len in (
            ([3000, 70, 9000], 1),
            ([5, 2049], 4),
            ([12000] * 2, 1),
            ([800] * 9, 4),
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
