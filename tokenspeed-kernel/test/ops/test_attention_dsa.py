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

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel import (
    dsa_decode,
    dsa_decode_topk,
    dsa_plan,
    dsa_prefill,
    dsa_prefill_topk,
)
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    workspace_topk_to_global_slots as dsa_workspace_topk_to_global_slots,
)

torch.manual_seed(42)


@dataclass(frozen=True)
class _TopKDecodeCase:
    name: str
    seq_lens: tuple[int, ...]
    index_heads: int
    topk: int
    q_len_per_req: int
    seed: int


@dataclass(frozen=True)
class _TopKPrefillCase:
    name: str
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]
    index_heads: int
    topk: int
    seed: int


@dataclass(frozen=True)
class _DSACase:
    name: str
    mode: str
    kv_layout: str
    topk: int
    seed: int
    num_heads: int = 8
    qk_nope_head_dim: int = 192
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    q_len_per_req: int = 1
    visible_lens: tuple[int, ...] | None = None
    topk_lens: tuple[int, ...] | None = None
    prefix_lens: tuple[int, ...] | None = None
    extend_lens: tuple[int, ...] | None = None


_GLM52_TOPK_DECODE_CASES = (
    _TopKDecodeCase(
        "decode_batch_mixed_512",
        seq_lens=(128, 257, 511, 1024),
        index_heads=2,
        topk=512,
        q_len_per_req=1,
        seed=101,
    ),
    _TopKDecodeCase(
        "decode_q3_boundary_512",
        seq_lens=(510, 511, 512, 1022, 1023, 1024),
        index_heads=2,
        topk=512,
        q_len_per_req=3,
        seed=102,
    ),
    _TopKDecodeCase(
        "decode_long_1024",
        seq_lens=(2048, 3072, 4096),
        index_heads=4,
        topk=1024,
        q_len_per_req=1,
        seed=103,
    ),
    _TopKDecodeCase(
        "decode_long_2048",
        seq_lens=(1536, 4096),
        index_heads=2,
        topk=2048,
        q_len_per_req=1,
        seed=104,
    ),
)


_GLM52_TOPK_PREFILL_CASES = (
    _TopKPrefillCase(
        "prefill_short_512",
        prefix_lens=(64, 128),
        extend_lens=(16, 32),
        index_heads=2,
        topk=512,
        seed=201,
    ),
    _TopKPrefillCase(
        "prefill_chunk_512",
        prefix_lens=(512, 1024),
        extend_lens=(32, 32),
        index_heads=2,
        topk=512,
        seed=202,
    ),
    _TopKPrefillCase(
        "prefill_mixed_1024",
        prefix_lens=(256, 1024, 1536),
        extend_lens=(16, 24, 16),
        index_heads=4,
        topk=1024,
        seed=203,
    ),
    _TopKPrefillCase(
        "prefill_long_2048",
        prefix_lens=(1536, 2048),
        extend_lens=(16, 16),
        index_heads=2,
        topk=2048,
        seed=204,
    ),
)


_GLM52_DSA_CASES = (
    _DSACase(
        "decode_sparse_mixed_512",
        mode="decode",
        kv_layout="sparse",
        topk=512,
        visible_lens=(128, 257, 512, 1024),
        topk_lens=(64, 257, 512, 384),
        seed=301,
    ),
    _DSACase(
        "decode_dense_q3_512",
        mode="decode",
        kv_layout="dense",
        topk=512,
        q_len_per_req=3,
        visible_lens=(512, 513, 514, 1024, 1025, 1026),
        topk_lens=(128, 256, 512, 300, 511, 64),
        seed=302,
    ),
    _DSACase(
        "decode_sparse_long_1024",
        mode="decode",
        kv_layout="sparse",
        topk=1024,
        visible_lens=(2048, 3072, 4096),
        topk_lens=(640, 1024, 777),
        seed=303,
    ),
    _DSACase(
        "decode_dense_long_2048",
        mode="decode",
        kv_layout="dense",
        topk=2048,
        visible_lens=(2048, 4096),
        topk_lens=(1536, 2048),
        seed=304,
    ),
    _DSACase(
        "prefill_sparse_short_512",
        mode="prefill",
        kv_layout="sparse",
        topk=512,
        prefix_lens=(64, 128),
        extend_lens=(8, 8),
        topk_lens=(
            32,
            64,
            96,
            128,
            48,
            80,
            112,
            136,
            33,
            65,
            97,
            129,
            49,
            81,
            113,
            136,
        ),
        seed=305,
    ),
    _DSACase(
        "prefill_dense_chunk_512",
        mode="prefill",
        kv_layout="dense",
        topk=512,
        prefix_lens=(512, 1024),
        extend_lens=(8, 8),
        topk_lens=(
            128,
            192,
            256,
            320,
            384,
            448,
            512,
            256,
            96,
            160,
            224,
            288,
            352,
            416,
            480,
            512,
        ),
        seed=306,
    ),
    _DSACase(
        "prefill_sparse_mixed_1024",
        mode="prefill",
        kv_layout="sparse",
        topk=1024,
        prefix_lens=(256, 1024, 1536),
        extend_lens=(4, 6, 4),
        topk_lens=(
            128,
            256,
            384,
            260,
            512,
            640,
            768,
            896,
            1024,
            768,
            512,
            1024,
            768,
            1024,
        ),
        seed=307,
    ),
    _DSACase(
        "prefill_dense_long_2048",
        mode="prefill",
        kv_layout="dense",
        topk=2048,
        prefix_lens=(1536, 2048),
        extend_lens=(4, 4),
        topk_lens=(512, 1024, 1536, 1539, 1024, 1536, 2048, 2048),
        seed=308,
    ),
)


def _pack_index_k_cache(
    index_k: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = index_k.shape[1]
    num_groups = head_dim // 128
    row_bytes = head_dim + num_groups * 4
    num_slots = index_k.shape[0]
    num_pages = num_slots // page_size
    packed = torch.empty(
        (num_slots, row_bytes),
        device=index_k.device,
        dtype=torch.uint8,
    )
    x = index_k.float().reshape(num_slots, num_groups, 128)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    x_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    flat = packed.reshape(-1)
    page_bytes = page_size * row_bytes
    fp8_view = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, page_size, head_dim),
        (page_bytes, head_dim, 1),
    )
    scale_view = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, page_size, num_groups),
        (page_bytes // 4, num_groups, 1),
        (page_size * head_dim) // 4,
    )
    fp8_view.copy_(x_fp8.reshape(num_pages, page_size, head_dim))
    scale_view.copy_(scale.reshape(num_pages, page_size, num_groups))
    return packed, (x_fp8.float() * scale).reshape_as(index_k)


def _generator(device: str, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _randn_bf16(
    shape: Sequence[int],
    *,
    device: str,
    generator: torch.Generator,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        * scale
    ).to(torch.bfloat16)


def _normal_weights(
    shape: Sequence[int],
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    logits = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    return torch.softmax(logits, dim=-1).contiguous()


def _round_up_to_page(slots: int, page_size: int) -> int:
    return int(math.ceil(slots / page_size) * page_size)


def _make_decode_block_table(
    seq_lens: Sequence[int],
    page_size: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    max_pages = max(math.ceil(seq_len / page_size) for seq_len in seq_lens)
    pages = torch.arange(
        len(seq_lens) * max_pages, device=device, dtype=torch.int32
    ).reshape(len(seq_lens), max_pages)
    return pages, int(len(seq_lens) * max_pages * page_size)


def _make_prefill_workspace(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[range]]:
    kv_workspace_slots: list[int] = []
    row_starts: list[int] = []
    row_ends: list[int] = []
    visible_ranges: list[range] = []
    cursor = 0
    for prefix_len, extend_len in zip(prefix_lens, extend_lens, strict=True):
        req_start = cursor
        seq_len = int(prefix_len) + int(extend_len)
        kv_workspace_slots.extend(range(req_start, req_start + seq_len))
        for query_offset in range(int(extend_len)):
            visible_end = req_start + int(prefix_len) + query_offset + 1
            row_starts.append(req_start)
            row_ends.append(visible_end)
            visible_ranges.append(range(req_start, visible_end))
        cursor += seq_len

    return (
        torch.tensor(kv_workspace_slots, device=device, dtype=torch.int64),
        torch.tensor(row_starts, device=device, dtype=torch.int32),
        torch.tensor(row_ends, device=device, dtype=torch.int32),
        visible_ranges,
    )


def _index_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    per_head = index_k.float() @ q.float().transpose(0, 1)
    return (per_head * weights.float()).sum(dim=1) * softmax_scale


def _reference_decode_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    lens = torch.minimum(seq_lens.to(torch.int32), torch.full_like(seq_lens, int(topk)))
    for token in range(q.shape[0]):
        seq_len = int(seq_lens[token].item())
        count = int(lens[token].item())
        if count == 0:
            continue
        offsets = torch.arange(seq_len, device=q.device, dtype=torch.long)
        pages = block_table[token].long().index_select(0, offsets // page_size)
        slots = pages * int(page_size) + offsets.remainder(page_size)
        scores = _index_scores(
            q[token],
            weights[token],
            index_k.index_select(0, slots),
            softmax_scale,
        )
        selected = torch.topk(scores, count).indices
        out[token, :count] = slots.index_select(0, selected).to(torch.int32)
    return out, lens


def _reference_prefill_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    candidate_lens = (row_ends - row_starts).clamp_min(0)
    lens = torch.minimum(candidate_lens, torch.full_like(candidate_lens, int(topk)))
    for token in range(q.shape[0]):
        count = int(lens[token].item())
        if count == 0:
            continue
        rows = torch.arange(
            int(row_starts[token].item()),
            int(row_ends[token].item()),
            device=q.device,
            dtype=torch.long,
        )
        slots = kv_workspace_slots.index_select(0, rows).long()
        scores = _index_scores(
            q[token],
            weights[token],
            index_k.index_select(0, slots),
            softmax_scale,
        )
        selected = torch.topk(scores, count).indices
        out[token, :count] = rows.index_select(0, selected).to(torch.int32)
    return out, lens


def _assert_topk_matches(
    actual: torch.Tensor,
    actual_lens: torch.Tensor,
    expected: torch.Tensor,
    expected_lens: torch.Tensor,
) -> None:
    torch.testing.assert_close(actual_lens.cpu(), expected_lens.cpu())
    for token in range(actual.shape[0]):
        count = int(expected_lens[token].item())
        actual_selected = torch.sort(actual[token, :count].cpu()).values
        expected_selected = torch.sort(expected[token, :count].cpu()).values
        torch.testing.assert_close(
            actual_selected,
            expected_selected,
        )
        assert (actual[token, count:] == -1).all()


def test_dsa_decode_topk_fp8(device: str, require) -> None:
    require("attention", "dsa_decode_topk", "triton", torch.bfloat16, "q")

    page_size = 64
    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    packed_index_k, index_k = _pack_index_k_cache(
        torch.randn((4 * page_size, 128), device=device, dtype=torch.bfloat16),
        page_size,
    )
    seq_lens = torch.tensor([20, 65, 3], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [[1, 3], [0, 2], [2, 1]], device=device, dtype=torch.int32
    )

    topk_slots, topk_lens = dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_cache=packed_index_k,
        solution="triton",
    )

    expected = torch.full_like(topk_slots, -1)
    expected_lens = torch.minimum(seq_lens, torch.full_like(seq_lens, topk))
    for token in range(q.shape[0]):
        scores = []
        slots = []
        for offset in range(int(seq_lens[token].item())):
            page = int(block_table[token, offset // page_size].item())
            slot = page * page_size + offset % page_size
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            slots.append(slot)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [slots[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(topk_slots[:, :65].cpu(), expected[:, :65].cpu())
    assert (topk_slots[0, int(expected_lens[0].item()) :] == -1).all()


@pytest.mark.parametrize("q_len_per_req", [2, 4])
def test_dsa_decode_topk_fp8_mtp(device: str, q_len_per_req: int, require) -> None:
    """Per-request (MTP) decode: seq_lens/block_table are per-request and the
    kernel derives each token's causal bound seq_lens[req] - (q-1) + j."""
    require("attention", "dsa_decode_topk", "triton", torch.bfloat16, "q")

    page_size = 64
    topk = 512
    num_reqs = 2
    pages = 8  # capacity 8*64=512 >= topk
    tokens = num_reqs * q_len_per_req
    q = torch.randn((tokens, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((tokens, 2), device=device, dtype=torch.float32)
    packed_index_k, index_k = _pack_index_k_cache(
        torch.randn((pages * page_size, 128), device=device, dtype=torch.bfloat16),
        page_size,
    )
    seq_lens = torch.tensor([200, 130], device=device, dtype=torch.int32)  # per-req
    block_table = (
        torch.arange(pages, device=device, dtype=torch.int32)
        .view(1, -1)
        .repeat(num_reqs, 1)
    )

    topk_slots, topk_lens = dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=128**-0.5,
        q_len_per_req=q_len_per_req,
        index_k_cache=packed_index_k,
        solution="triton",
    )

    for r in range(num_reqs):
        for jj in range(q_len_per_req):
            token = r * q_len_per_req + jj
            causal_len = int(seq_lens[r].item()) - (q_len_per_req - 1) + jj
            scores = []
            slots = []
            for off in range(causal_len):
                page = int(block_table[r, off // page_size].item())
                slot = page * page_size + off % page_size
                per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
                scores.append((per_head * weights[token]).sum() * (128**-0.5))
                slots.append(slot)
            k = min(causal_len, topk)
            local = torch.topk(torch.stack(scores), k).indices
            ref = {slots[int(i)] for i in local.tolist()}
            got = {int(x) for x in topk_slots[token, :k].tolist() if x >= 0}
            assert int(topk_lens[token].item()) == k, (token, topk_lens[token], k)
            assert ref == got, f"token {token}: {len(ref ^ got)} slots differ"


def test_dsa_prefill_topk_fp8(device: str, require) -> None:
    require("attention", "dsa_prefill_topk", "triton", torch.bfloat16, "q")

    page_size = 64
    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    packed_index_k, index_k = _pack_index_k_cache(
        torch.randn((4 * page_size, 128), device=device, dtype=torch.bfloat16),
        page_size,
    )
    kv_workspace_slots = torch.arange(85, device=device, dtype=torch.int64) + 17
    row_starts = torch.tensor([0, 10, 70], device=device, dtype=torch.int32)
    row_ends = torch.tensor([20, 75, 85], device=device, dtype=torch.int32)

    workspace_indices, topk_lens = dsa_prefill_topk(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_cache=packed_index_k,
        page_size=page_size,
        solution="triton",
    )

    expected = torch.full_like(workspace_indices, -1)
    expected_lens = torch.minimum(
        row_ends - row_starts, torch.full_like(row_ends, topk)
    )
    for token in range(q.shape[0]):
        scores = []
        rows = []
        for row in range(int(row_starts[token].item()), int(row_ends[token].item())):
            slot = int(kv_workspace_slots[row].item())
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            rows.append(row)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [rows[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(workspace_indices[:, :65].cpu(), expected[:, :65].cpu())
    assert (workspace_indices[0, int(expected_lens[0].item()) :] == -1).all()


@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize(
    "case",
    _GLM52_TOPK_DECODE_CASES,
    ids=lambda case: case.name,
)
def test_dsa_decode_topk_fp8_glm52_cases(
    device: str,
    require,
    solution: str,
    case: _TopKDecodeCase,
) -> None:
    require("attention", "dsa_decode_topk", solution, torch.bfloat16, "q")

    page_size = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5
    gen = _generator(device, case.seed)
    block_table, num_slots = _make_decode_block_table(case.seq_lens, page_size, device)
    q = _randn_bf16(
        (len(case.seq_lens), case.index_heads, head_dim),
        device=device,
        generator=gen,
    )
    weights = _normal_weights(
        (len(case.seq_lens), case.index_heads), device=device, generator=gen
    )
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )
    seq_lens = torch.tensor(case.seq_lens, device=device, dtype=torch.int32)

    topk_slots, topk_lens = dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=case.topk,
        softmax_scale=softmax_scale,
        q_len_per_req=case.q_len_per_req,
        index_k_cache=packed_index_k,
        solution=solution,
    )
    expected_slots, expected_lens = _reference_decode_topk(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=case.topk,
        softmax_scale=softmax_scale,
    )

    _assert_topk_matches(topk_slots, topk_lens, expected_slots, expected_lens)


@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize(
    "case",
    _GLM52_TOPK_PREFILL_CASES,
    ids=lambda case: case.name,
)
def test_dsa_prefill_topk_fp8_glm52_cases(
    device: str,
    require,
    solution: str,
    case: _TopKPrefillCase,
) -> None:
    require("attention", "dsa_prefill_topk", solution, torch.bfloat16, "q")

    page_size = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5
    gen = _generator(device, case.seed)
    kv_workspace_slots, row_starts, row_ends, _ = _make_prefill_workspace(
        case.prefix_lens, case.extend_lens, device=device
    )
    num_tokens = int(sum(case.extend_lens))
    num_slots = _round_up_to_page(int(kv_workspace_slots.numel()), page_size)
    q = _randn_bf16(
        (num_tokens, case.index_heads, head_dim),
        device=device,
        generator=gen,
    )
    weights = _normal_weights(
        (num_tokens, case.index_heads), device=device, generator=gen
    )
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )

    workspace_indices, topk_lens = dsa_prefill_topk(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=case.topk,
        softmax_scale=softmax_scale,
        index_k_cache=packed_index_k,
        page_size=page_size,
        solution=solution,
    )
    expected_indices, expected_lens = _reference_prefill_topk(
        q,
        weights,
        index_k,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=case.topk,
        softmax_scale=softmax_scale,
    )

    _assert_topk_matches(workspace_indices, topk_lens, expected_indices, expected_lens)


def test_dsa_decode_topk_gluon_long_row_uses_radix_path(
    device: str,
    require,
) -> None:
    require("attention", "dsa_decode_topk", "gluon", torch.bfloat16, "q")

    page_size = 64
    seq_len = 65536
    topk = 2048
    head_dim = 128
    q = torch.ones((1, 1, head_dim), device=device, dtype=torch.bfloat16)
    weights = torch.ones((1, 1), device=device, dtype=torch.float32)
    index_k = torch.zeros((seq_len, head_dim), device=device, dtype=torch.bfloat16)
    index_k[:topk].fill_(1.0)
    packed_index_k, _ = _pack_index_k_cache(index_k, page_size)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    block_table = torch.arange(
        seq_len // page_size, device=device, dtype=torch.int32
    ).reshape(1, -1)

    topk_slots, topk_lens = dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=head_dim**-0.5,
        index_k_cache=packed_index_k,
        solution="gluon",
    )

    expected = torch.arange(topk, device=device, dtype=torch.int32)
    torch.testing.assert_close(topk_lens.cpu(), torch.tensor([topk], dtype=torch.int32))
    torch.testing.assert_close(torch.sort(topk_slots[0]).values.cpu(), expected.cpu())


def test_dsa_prefill_topk_gluon_long_row_uses_radix_path(
    device: str,
    require,
) -> None:
    require("attention", "dsa_prefill_topk", "gluon", torch.bfloat16, "q")

    page_size = 64
    seq_len = 65536
    topk = 2048
    head_dim = 128
    q = torch.ones((1, 1, head_dim), device=device, dtype=torch.bfloat16)
    weights = torch.ones((1, 1), device=device, dtype=torch.float32)
    index_k = torch.zeros((seq_len, head_dim), device=device, dtype=torch.bfloat16)
    index_k[:topk].fill_(1.0)
    packed_index_k, _ = _pack_index_k_cache(index_k, page_size)
    kv_workspace_slots = torch.arange(seq_len, device=device, dtype=torch.int64)
    row_starts = torch.tensor([0], device=device, dtype=torch.int32)
    row_ends = torch.tensor([seq_len], device=device, dtype=torch.int32)

    workspace_indices, topk_lens = dsa_prefill_topk(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=head_dim**-0.5,
        index_k_cache=packed_index_k,
        page_size=page_size,
        solution="gluon",
    )

    expected = torch.arange(topk, device=device, dtype=torch.int32)
    torch.testing.assert_close(topk_lens.cpu(), torch.tensor([topk], dtype=torch.int32))
    torch.testing.assert_close(
        torch.sort(workspace_indices[0]).values.cpu(),
        expected.cpu(),
    )


def test_dsa_plan_triton(device: str) -> None:
    # The triton decode kernel derives its own causal bounds and ignores the
    # plan, so triton_dsa_plan is a no-op returning an opaque, non-None
    # placeholder; passing out= returns that same placeholder.
    seq_lens_2d = torch.tensor([[20], [65], [3]], device=device, dtype=torch.int32)
    plan = dsa_plan(seq_lens_2d=seq_lens_2d, page_size=64, solution="triton")
    if plan is None:
        pytest.skip("triton dsa_plan is not registered on this platform")

    refreshed = dsa_plan(
        seq_lens_2d=seq_lens_2d, page_size=64, out=plan, solution="triton"
    )
    assert refreshed is plan


def test_dsa_plan_returns_none_without_kernel(device: str) -> None:
    seq_lens_2d = torch.tensor([[1]], device=device, dtype=torch.int32)

    assert dsa_plan(seq_lens_2d=seq_lens_2d, page_size=64, solution="missing") is None


def test_dsa_workspace_topk_to_global_slots(device: str) -> None:
    workspace_indices = torch.tensor(
        [[2, -1, 0], [1, 3, -1]],
        device=device,
        dtype=torch.int32,
    )
    kv_workspace_slots = torch.tensor(
        [10, 20, 30, 40],
        device=device,
        dtype=torch.int64,
    )
    out = torch.empty_like(workspace_indices)

    slots = dsa_workspace_topk_to_global_slots(
        workspace_indices=workspace_indices,
        kv_workspace_slots=kv_workspace_slots,
        out=out,
    )

    expected = torch.tensor(
        [[30, -1, 10], [20, 40, -1]],
        device=device,
        dtype=torch.int32,
    )
    assert slots.data_ptr() == out.data_ptr()
    torch.testing.assert_close(slots.cpu(), expected.cpu())


def _pack_sparse_kv(
    latent: torch.Tensor,
    rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_lora_rank = latent.shape[1]
    qk_rope_head_dim = rope.shape[1]
    scale = latent.float().abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6) / 448.0
    latent_fp8 = (latent.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    row_bytes = kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2
    sparse = torch.empty(
        (latent.shape[0], row_bytes),
        dtype=torch.uint8,
        device=latent.device,
    )
    sparse[:, :kv_lora_rank].copy_(latent_fp8.view(torch.uint8))
    scale_start = kv_lora_rank
    scale_end = scale_start + kv_lora_rank // 128 * 4
    sparse[:, scale_start:scale_end].view(torch.float32).copy_(scale)
    sparse[:, scale_end:].view(torch.bfloat16).copy_(rope)
    return sparse, latent_fp8.float() * scale


def _dsa_reference(
    q: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    refs = []
    kv_lora_rank = latent.shape[1]
    for token in range(q.shape[0]):
        valid_slots = topk_slots[token, : int(topk_lens[token].item())].long()
        valid_slots = valid_slots[valid_slots >= 0]
        q_nope = q[token, :, :kv_lora_rank].float()
        q_rope = q[token, :, kv_lora_rank:].float()
        if valid_slots.numel() == 0:
            refs.append(torch.zeros_like(q_nope))
            continue
        k_nope = latent.index_select(0, valid_slots).float()
        k_rope = rope.index_select(0, valid_slots).float()
        scores = torch.einsum("hd,kd->hk", q_nope, k_nope)
        scores += torch.einsum("hd,kd->hk", q_rope, k_rope)
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        refs.append(torch.matmul(probs, k_nope))
    return torch.stack(refs, dim=0).to(torch.bfloat16)


def _dsa_visible_ranges(case: _DSACase, device: str) -> tuple[list[range], int]:
    if case.mode == "decode":
        assert case.visible_lens is not None
        ranges = [range(0, int(visible_len)) for visible_len in case.visible_lens]
        return ranges, _round_up_to_page(max(case.visible_lens), 64)

    assert case.prefix_lens is not None
    assert case.extend_lens is not None
    kv_workspace_slots, _, _, ranges = _make_prefill_workspace(
        case.prefix_lens,
        case.extend_lens,
        device=device,
    )
    return ranges, _round_up_to_page(int(kv_workspace_slots.numel()), 64)


def _make_selected_topk_slots(
    case: _DSACase,
    visible_ranges: Sequence[range],
    *,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert case.topk_lens is not None
    assert len(case.topk_lens) == len(visible_ranges)
    topk_slots = torch.full(
        (len(visible_ranges), case.topk), -1, device=device, dtype=torch.int32
    )
    lens: list[int] = []
    for token, visible_range in enumerate(visible_ranges):
        visible_count = len(visible_range)
        count = min(int(case.topk_lens[token]), visible_count, int(case.topk))
        lens.append(count)
        if count == 0:
            continue
        candidates = torch.arange(
            visible_range.start,
            visible_range.stop,
            device=device,
            dtype=torch.int32,
        )
        perm = torch.randperm(visible_count, device=device, generator=generator)[:count]
        topk_slots[token, :count] = candidates.index_select(0, perm)
    return topk_slots, torch.tensor(lens, device=device, dtype=torch.int32)


def _assert_slots_visible(
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    visible_ranges: Sequence[range],
) -> None:
    for token, visible_range in enumerate(visible_ranges):
        count = int(topk_lens[token].item())
        valid = topk_slots[token, :count]
        if count:
            assert (valid >= visible_range.start).all()
            assert (valid < visible_range.stop).all()
        assert (topk_slots[token, count:] == -1).all()


@pytest.mark.parametrize(
    "mode,api_name",
    [
        pytest.param("decode", "dsa_decode", id="decode"),
        pytest.param("prefill", "dsa_prefill", id="prefill"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_fp8"),
    ],
)
def test_dsa_with_kvcache(
    device: str,
    mode: str,
    api_name: str,
    solution: str,
    q_dtype: torch.dtype,
    require,
) -> None:
    require("attention", api_name, solution, q_dtype, "q")

    tokens = 3
    num_heads = 2
    num_slots = 16
    topk = 512
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q_bf16 = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_bf16.to(q_dtype)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    sparse_kv, dequant_latent = _pack_sparse_kv(latent, rope)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([5, 7, 4], device=device, dtype=torch.int32)
    for token in range(tokens):
        count = int(topk_lens[token].item())
        topk_slots[token, :count] = torch.randperm(num_slots, device=device)[:count]

    api = dsa_decode if mode == "decode" else dsa_prefill
    out = api(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
        solution=solution,
    )

    ref = _dsa_reference(
        q,
        dequant_latent,
        rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_fp8"),
    ],
)
def test_dsa_decode_dense_kvcache(device: str, q_dtype: torch.dtype, require) -> None:
    require("attention", "dsa_decode", "triton", q_dtype, "q")

    tokens = 3
    num_heads = 2
    num_slots = 16
    topk = 512
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q_bf16 = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_bf16.to(q_dtype)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    kv_cache = torch.cat([latent, rope], dim=-1).to(q_dtype)
    dequant_latent = kv_cache[:, :kv_lora_rank].float().to(torch.bfloat16)
    dequant_rope = kv_cache[:, kv_lora_rank:].float().to(torch.bfloat16)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([5, 7, 4], device=device, dtype=torch.int32)
    for token in range(tokens):
        count = int(topk_lens[token].item())
        topk_slots[token, :count] = torch.randperm(num_slots, device=device)[:count]

    out = dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
        solution="triton",
    )

    ref = _dsa_reference(
        q,
        dequant_latent,
        dequant_rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "case",
    _GLM52_DSA_CASES,
    ids=lambda case: case.name,
)
@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_dsa_glm52_selected_attention_cases(
    device: str,
    require,
    solution: str,
    case: _DSACase,
) -> None:
    api_name = "dsa_decode" if case.mode == "decode" else "dsa_prefill"
    require("attention", api_name, solution, torch.bfloat16, "q")

    page_size = 64
    gen = _generator(device, case.seed)
    visible_ranges, num_slots = _dsa_visible_ranges(case, device)
    tokens = len(visible_ranges)
    q = _randn_bf16(
        (tokens, case.num_heads, case.kv_lora_rank + case.qk_rope_head_dim),
        device=device,
        generator=gen,
    )
    latent = _randn_bf16((num_slots, case.kv_lora_rank), device=device, generator=gen)
    rope = _randn_bf16((num_slots, case.qk_rope_head_dim), device=device, generator=gen)
    topk_slots, topk_lens = _make_selected_topk_slots(
        case, visible_ranges, device=device, generator=gen
    )
    _assert_slots_visible(topk_slots, topk_lens, visible_ranges)

    kv_cache = None
    sparse_kv_cache = None
    if case.kv_layout == "dense":
        kv_cache = torch.cat([latent, rope], dim=-1).contiguous()
        reference_latent = kv_cache[:, : case.kv_lora_rank]
        reference_rope = kv_cache[:, case.kv_lora_rank :]
    elif case.kv_layout == "sparse":
        sparse_kv_cache, reference_latent = _pack_sparse_kv(latent, rope)
        reference_rope = rope
    else:
        raise AssertionError(f"unknown DSA KV layout {case.kv_layout!r}")

    softmax_scale = 1.0 / math.sqrt(case.qk_nope_head_dim + case.qk_rope_head_dim)
    common_kwargs = {
        "q": q,
        "kv_cache": kv_cache,
        "sparse_kv_cache": sparse_kv_cache,
        "topk_slots": topk_slots,
        "topk_lens": topk_lens,
        "max_seqlen_k": max(len(visible_range) for visible_range in visible_ranges),
        "qk_nope_head_dim": case.qk_nope_head_dim,
        "kv_lora_rank": case.kv_lora_rank,
        "qk_rope_head_dim": case.qk_rope_head_dim,
        "softmax_scale": softmax_scale,
        "page_size": page_size,
        "solution": solution,
    }
    if case.mode == "decode":
        out = dsa_decode(q_len_per_req=case.q_len_per_req, **common_kwargs)
    else:
        out = dsa_prefill(**common_kwargs)

    ref = _dsa_reference(
        q,
        reference_latent,
        reference_rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, case.num_heads, case.kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)
