# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch
from utils import is_cdna4, is_cdna5

if not (is_cdna4() or is_cdna5()):
    pytest.skip(
        "AMD CDNA4 or CDNA5 is required for Gluon DSA tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import (  # noqa: E402
    attention as gfx950_dsa_backend,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (  # noqa: E402
    _trim_topk_slots_for_context,
)

if is_cdna4():
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import (  # noqa: E402
        sparse_mla as dsa_topk_backend,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
        gluon_dsa_decode_gfx950 as gluon_dsa_decode,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
        gluon_dsa_prefill_gfx950 as gluon_dsa_prefill,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (  # noqa: E402
        gluon_dsa_decode_topk_fp8_gfx950 as gluon_dsa_decode_topk_fp8,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_logical_topk_gfx950 as gluon_logical_topk_indices,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_fp8_gfx950 as gluon_dsa_prefill_topk_fp8,
    )
else:
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa import (  # noqa: E402
        sparse_mla as dsa_topk_backend,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (  # noqa: E402
        gluon_dsa_decode_gfx1250 as gluon_dsa_decode,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (
        gluon_dsa_prefill_gfx1250 as gluon_dsa_prefill,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (  # noqa: E402
        gluon_dsa_decode_topk_fp8_gfx1250 as gluon_dsa_decode_topk_fp8,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_fp8_gfx1250 as gluon_dsa_prefill_topk_fp8,
    )

torch.manual_seed(42)


@pytest.mark.parametrize(
    ("max_seqlen_k", "expected_topk"),
    ((25, 512), (608, 1024), (1537, 2048)),
)
@pytest.mark.skipif(
    not is_cdna4(),
    reason="registered-width trimming is specific to the gfx950 implementation",
)
def test_dsa_attention_trims_topk_to_registered_context_width(
    max_seqlen_k: int,
    expected_topk: int,
) -> None:
    topk_slots = torch.arange(2 * 2048, device="cuda", dtype=torch.int32).reshape(
        2, 2048
    )

    trimmed = _trim_topk_slots_for_context(topk_slots, max_seqlen_k)

    assert trimmed.shape == (2, expected_topk)
    torch.testing.assert_close(trimmed, topk_slots[:, :expected_topk])


@dataclass(frozen=True)
class _TopKDecodeCase:
    name: str
    seq_lens: tuple[int, ...]
    index_heads: int
    topk: int
    q_len_per_req: int
    seed: int
    reverse_pages: bool = False


@dataclass(frozen=True)
class _TopKPrefillCase:
    name: str
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]
    index_heads: int
    topk: int
    seed: int
    reverse_slots: bool = False


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
    _TopKDecodeCase(
        "decode_trivial_q2_2048",
        seq_lens=(511,),
        index_heads=2,
        topk=2048,
        q_len_per_req=2,
        seed=105,
        reverse_pages=True,
    ),
    _TopKDecodeCase(
        "decode_persistent_q2_2048",
        seq_lens=(131071,),
        index_heads=2,
        topk=2048,
        q_len_per_req=2,
        seed=106,
        reverse_pages=True,
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
    _TopKPrefillCase(
        "prefill_persistent_2048",
        prefix_lens=(65535, 65533),
        extend_lens=(1, 2),
        index_heads=2,
        topk=2048,
        seed=205,
        reverse_slots=True,
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
    _DSACase(
        "prefill_sparse_first_prompt_2048",
        mode="prefill",
        kv_layout="sparse",
        topk=2048,
        num_heads=16,
        prefix_lens=(0,),
        extend_lens=(25,),
        topk_lens=tuple(range(1, 26)),
        seed=309,
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


def _split_bf16_weights(
    tokens: int,
    heads: int,
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    backing = _randn_bf16(
        (tokens, 128 + heads),
        device=device,
        generator=generator,
    )
    weights = backing[:, 128:]
    assert not weights.is_contiguous()
    return weights


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
    return (per_head.relu() * weights.float()).sum(dim=1) * softmax_scale


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
    q_len_per_req: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    lens = torch.empty((q.shape[0],), device=q.device, dtype=torch.int32)
    candidate_ids_and_scores: list[tuple[torch.Tensor, torch.Tensor]] = []
    for token in range(q.shape[0]):
        req = token // int(q_len_per_req)
        q_offset = token - req * int(q_len_per_req)
        seq_len = int(seq_lens[req].item())
        if q_len_per_req != 1:
            seq_len = seq_len - (int(q_len_per_req) - 1) + q_offset
        count = min(seq_len, int(topk))
        lens[token] = count
        if count == 0:
            candidate_ids_and_scores.append(
                (
                    torch.empty((0,), device=q.device, dtype=torch.long),
                    torch.empty((0,), device=q.device, dtype=torch.float32),
                )
            )
            continue
        offsets = torch.arange(seq_len, device=q.device, dtype=torch.long)
        pages = block_table[req].long().index_select(0, offsets // page_size)
        slots = pages * int(page_size) + offsets.remainder(page_size)
        scores = _index_scores(
            q[token],
            weights[token],
            index_k.index_select(0, slots),
            softmax_scale,
        )
        selected = torch.topk(scores, count).indices
        out[token, :count] = slots.index_select(0, selected).to(torch.int32)
        candidate_ids_and_scores.append((slots, scores))
    return out, lens, candidate_ids_and_scores


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
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    candidate_lens = (row_ends - row_starts).clamp_min(0)
    lens = torch.minimum(candidate_lens, torch.full_like(candidate_lens, int(topk)))
    candidate_ids_and_scores: list[tuple[torch.Tensor, torch.Tensor]] = []
    for token in range(q.shape[0]):
        count = int(lens[token].item())
        if count == 0:
            candidate_ids_and_scores.append(
                (
                    torch.empty((0,), device=q.device, dtype=torch.long),
                    torch.empty((0,), device=q.device, dtype=torch.float32),
                )
            )
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
        candidate_ids_and_scores.append((rows, scores))
    return out, lens, candidate_ids_and_scores


def _assert_topk_matches(
    actual: torch.Tensor,
    actual_lens: torch.Tensor,
    expected: torch.Tensor,
    expected_lens: torch.Tensor,
    candidate_ids_and_scores: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    torch.testing.assert_close(actual_lens.cpu(), expected_lens.cpu())
    for token in range(actual.shape[0]):
        count = int(expected_lens[token].item())
        candidate_ids, candidate_scores = candidate_ids_and_scores[token]
        actual_ids = actual[token, :count].long()
        assert torch.unique(actual_ids).numel() == count
        sorted_ids, order = torch.sort(candidate_ids)
        locations = torch.searchsorted(sorted_ids, actual_ids)
        assert (locations < sorted_ids.numel()).all()
        torch.testing.assert_close(sorted_ids[locations], actual_ids)
        actual_selected = torch.sort(actual[token, :count].cpu()).values
        expected_selected = torch.sort(expected[token, :count].cpu()).values
        if not torch.equal(actual_selected, expected_selected):
            # Per-head ReLU can create exact zero-score ties at the cutoff.
            actual_only = actual_ids[~torch.isin(actual_ids, expected[token, :count])]
            expected_only = expected[token, :count].long()[
                ~torch.isin(expected[token, :count], actual_ids)
            ]
            assert actual_only.numel() == expected_only.numel()
            cutoff = torch.topk(candidate_scores, count).values[-1]
            for tied_ids in (actual_only, expected_only):
                tied_locations = torch.searchsorted(sorted_ids, tied_ids)
                tied_scores = candidate_scores[order[tied_locations]]
                torch.testing.assert_close(
                    tied_scores,
                    torch.full_like(tied_scores, cutoff),
                    rtol=0.0,
                    atol=0.0,
                )
        assert (actual[token, count:] == -1).all()


@pytest.mark.parametrize(
    "case",
    _GLM52_TOPK_DECODE_CASES,
    ids=lambda case: case.name,
)
def test_dsa_decode_topk_fp8_glm52_cases(case: _TopKDecodeCase) -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5
    gen = _generator(device, case.seed)
    block_table, num_slots = _make_decode_block_table(case.seq_lens, page_size, device)
    if case.reverse_pages:
        block_table = block_table.flip(1).contiguous()
    tokens = len(case.seq_lens) * case.q_len_per_req
    q = _randn_bf16(
        (tokens, case.index_heads, head_dim),
        device=device,
        generator=gen,
    )
    weights = _normal_weights((tokens, case.index_heads), device=device, generator=gen)
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )
    seq_lens = torch.tensor(case.seq_lens, device=device, dtype=torch.int32)

    topk_slots, topk_lens = gluon_dsa_decode_topk_fp8(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=case.topk,
        softmax_scale=softmax_scale,
        seq_lens_2d=seq_lens.unsqueeze(1).expand(-1, case.q_len_per_req),
        q_len_per_req=case.q_len_per_req,
        index_k_cache=packed_index_k,
    )
    expected_slots, expected_lens, candidate_ids_and_scores = _reference_decode_topk(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=case.topk,
        softmax_scale=softmax_scale,
        q_len_per_req=case.q_len_per_req,
    )

    _assert_topk_matches(
        topk_slots,
        topk_lens,
        expected_slots,
        expected_lens,
        candidate_ids_and_scores,
    )


@pytest.mark.parametrize(
    "case",
    _GLM52_TOPK_PREFILL_CASES,
    ids=lambda case: case.name,
)
def test_dsa_prefill_topk_fp8_glm52_cases(case: _TopKPrefillCase) -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5
    gen = _generator(device, case.seed)
    kv_workspace_slots, row_starts, row_ends, _ = _make_prefill_workspace(
        case.prefix_lens, case.extend_lens, device=device
    )
    if case.reverse_slots:
        kv_workspace_slots = kv_workspace_slots.flip(0).contiguous()
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

    workspace_indices, topk_lens = gluon_dsa_prefill_topk_fp8(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=case.topk,
        softmax_scale=softmax_scale,
        index_k_cache=packed_index_k,
        page_size=page_size,
    )
    expected_indices, expected_lens, candidate_ids_and_scores = _reference_prefill_topk(
        q,
        weights,
        index_k,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=case.topk,
        softmax_scale=softmax_scale,
    )

    _assert_topk_matches(
        workspace_indices,
        topk_lens,
        expected_indices,
        expected_lens,
        candidate_ids_and_scores,
    )


if is_cdna4():
    _BF16_SPLIT_WEIGHT_DECODE_CASES = (
        (8192, False),
        (8192, True),
        (16384, False),
        (16384, True),
    )
else:
    _BF16_SPLIT_WEIGHT_DECODE_CASES = (
        (49152, False),
        (49152, True),
        (49153, False),
        (49153, True),
    )


@pytest.mark.parametrize(
    ("seq_len", "capture_graph"),
    _BF16_SPLIT_WEIGHT_DECODE_CASES,
    ids=("short-eager", "short-graph", "long-eager", "long-graph"),
)
def test_dsa_decode_topk_fp8_accepts_glm52_bf16_split_weights(
    seq_len: int,
    capture_graph: bool,
) -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    index_heads = 32
    topk = 2048
    q_len_per_req = 2
    gen = _generator(device, 107)
    block_table, num_slots = _make_decode_block_table((seq_len,), page_size, device)
    q = _randn_bf16(
        (q_len_per_req, index_heads, head_dim),
        device=device,
        generator=gen,
    )
    weights = _split_bf16_weights(
        q_len_per_req,
        index_heads,
        device=device,
        generator=gen,
    )
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )
    seq_lens = torch.tensor((seq_len,), device=device, dtype=torch.int32)

    topk_slots = torch.empty((q_len_per_req, topk), device=device, dtype=torch.int32)
    topk_lens = torch.empty((q_len_per_req,), device=device, dtype=torch.int32)

    def invoke() -> None:
        gluon_dsa_decode_topk_fp8(
            q,
            weights,
            seq_lens,
            block_table,
            page_size=page_size,
            topk=topk,
            softmax_scale=head_dim**-0.5 * index_heads**-0.5,
            q_len_per_req=q_len_per_req,
            index_k_cache=packed_index_k,
            out=topk_slots,
            lens_out=topk_lens,
        )

    invoke()
    torch.cuda.synchronize()
    if capture_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        graph.replay()
        torch.cuda.synchronize()
    expected_slots, expected_lens, candidate_ids_and_scores = _reference_decode_topk(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=head_dim**-0.5 * index_heads**-0.5,
        q_len_per_req=q_len_per_req,
    )

    _assert_topk_matches(
        topk_slots,
        topk_lens,
        expected_slots,
        expected_lens,
        candidate_ids_and_scores,
    )


def test_dsa_prefill_topk_fp8_accepts_glm52_bf16_split_weights() -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    index_heads = 32
    topk = 2048
    gen = _generator(device, 206)
    workspace, row_starts, row_ends, _ = _make_prefill_workspace(
        (4094,), (2,), device=device
    )
    num_tokens = 2
    num_slots = _round_up_to_page(int(workspace.numel()), page_size)
    q = _randn_bf16(
        (num_tokens, index_heads, head_dim),
        device=device,
        generator=gen,
    )
    weights = _split_bf16_weights(
        num_tokens,
        index_heads,
        device=device,
        generator=gen,
    )
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )

    workspace_indices, topk_lens = gluon_dsa_prefill_topk_fp8(
        q,
        weights,
        workspace,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=head_dim**-0.5 * index_heads**-0.5,
        index_k_cache=packed_index_k,
        page_size=page_size,
    )
    expected_indices, expected_lens, candidate_ids_and_scores = _reference_prefill_topk(
        q,
        weights,
        index_k,
        workspace,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=head_dim**-0.5 * index_heads**-0.5,
    )

    _assert_topk_matches(
        workspace_indices,
        topk_lens,
        expected_indices,
        expected_lens,
        candidate_ids_and_scores,
    )


def test_dsa_prefill_topk_keeps_late_values_above_threshold() -> None:
    cols = 16384
    topk = 2048
    logits = torch.full((1, cols), -10.0, device="cuda", dtype=torch.float32)
    equal_indices = torch.arange(0, 32, device="cuda", dtype=torch.int32)
    greater_indices = torch.cat(
        (
            torch.arange(
                4096,
                4096 + topk - 2,
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor([cols - 3], device="cuda", dtype=torch.int32),
        )
    )
    logits[0, equal_indices.long()] = 1.0
    logits[0, greater_indices.long()] = 2.0
    row_starts = torch.zeros((1,), device="cuda", dtype=torch.int32)
    row_ends = torch.full((1,), cols, device="cuda", dtype=torch.int32)
    out = torch.empty((1, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)

    dsa_topk_backend._dsa_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    selected_set = set(out[0].cpu().tolist())
    assert selected_set.issuperset(set(greater_indices.cpu().tolist()))
    assert len(selected_set.intersection(set(equal_indices.cpu().tolist()))) == 1
    torch.testing.assert_close(lens_out.cpu(), torch.tensor([topk], dtype=torch.int32))


def test_dsa_decode_topk_keeps_late_values_above_threshold() -> None:
    page_size = 64
    cols = 16384
    topk = 2048
    logits = torch.full((1, cols), -10.0, device="cuda", dtype=torch.float32)
    equal_indices = torch.arange(0, 32, device="cuda", dtype=torch.int32)
    greater_indices = torch.cat(
        (
            torch.arange(
                4096,
                4096 + topk - 2,
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor([cols - 3], device="cuda", dtype=torch.int32),
        )
    )
    logits[0, equal_indices.long()] = 1.0
    logits[0, greater_indices.long()] = 2.0
    seq_lens = torch.full((1,), cols, device="cuda", dtype=torch.int32)
    block_table = torch.arange(
        cols // page_size,
        device="cuda",
        dtype=torch.int32,
    ).reshape(1, -1)
    out = torch.empty((1, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)

    dsa_topk_backend._dsa_topk_indices(
        logits,
        seq_lens,
        seq_lens,
        block_table=block_table,
        page_size=page_size,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    selected_set = set(out[0].cpu().tolist())
    assert selected_set.issuperset(set(greater_indices.cpu().tolist()))
    assert len(selected_set.intersection(set(equal_indices.cpu().tolist()))) == 1
    torch.testing.assert_close(lens_out.cpu(), torch.tensor([topk], dtype=torch.int32))


@pytest.mark.skipif(not is_cdna4(), reason="requires the gfx950 logical top-k")
def test_gluon_logical_topk_indices_respects_ragged_ranges_and_outputs() -> None:
    cols = 1024
    topk = 512
    ascending = torch.arange(cols, device="cuda", dtype=torch.float32)
    logits = torch.stack((ascending, -ascending, ascending)).contiguous()
    row_starts = torch.tensor([0, 123, 400], device="cuda", dtype=torch.int32)
    row_ends = torch.tensor([cols, 900, 410], device="cuda", dtype=torch.int32)
    out = torch.empty((3, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((3,), device="cuda", dtype=torch.int32)

    actual_out, actual_lens = gluon_logical_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    assert actual_out is out
    assert actual_lens is lens_out
    expected = torch.full_like(out, -1)
    expected[0] = torch.arange(512, 1024, device="cuda", dtype=torch.int32)
    expected[1] = torch.arange(123, 635, device="cuda", dtype=torch.int32)
    expected[2, :10] = torch.arange(400, 410, device="cuda", dtype=torch.int32)
    expected_lens = torch.tensor([512, 512, 10], device="cuda", dtype=torch.int32)
    candidates = [
        (
            torch.arange(int(start), int(end), device="cuda", dtype=torch.long),
            logits[row, int(start) : int(end)],
        )
        for row, (start, end) in enumerate(zip(row_starts, row_ends, strict=True))
    ]
    _assert_topk_matches(
        actual_out,
        actual_lens,
        expected,
        expected_lens,
        candidates,
    )


@pytest.mark.skipif(not is_cdna4(), reason="requires the gfx950 logical top-k")
def test_gluon_logical_topk_indices_validates_tensor_contract() -> None:
    logits = torch.empty((2, 1024), device="cuda", dtype=torch.float32)
    row_starts = torch.zeros((2,), device="cuda", dtype=torch.int32)
    row_ends = torch.full((2,), 1024, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="2-D"):
        gluon_logical_topk_indices(logits[0], row_starts, row_ends, topk=512)
    with pytest.raises(TypeError, match="float32"):
        gluon_logical_topk_indices(
            logits.to(torch.bfloat16), row_starts, row_ends, topk=512
        )
    with pytest.raises(ValueError, match="logits must be contiguous"):
        gluon_logical_topk_indices(
            torch.empty((1024, 2), device="cuda", dtype=torch.float32).T,
            row_starts,
            row_ends,
            topk=512,
        )
    with pytest.raises(ValueError, match="row_starts must have shape"):
        gluon_logical_topk_indices(logits, row_starts[:1], row_ends, topk=512)
    with pytest.raises(TypeError, match="row_ends must be int32"):
        gluon_logical_topk_indices(
            logits, row_starts, row_ends.to(torch.int64), topk=512
        )
    with pytest.raises(ValueError, match="out must have shape"):
        gluon_logical_topk_indices(
            logits,
            row_starts,
            row_ends,
            topk=512,
            out=torch.empty((2, 511), device="cuda", dtype=torch.int32),
        )
    with pytest.raises(TypeError, match="lens_out must be int32"):
        gluon_logical_topk_indices(
            logits,
            row_starts,
            row_ends,
            topk=512,
            lens_out=torch.empty((2,), device="cuda", dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="supports topk"):
        gluon_logical_topk_indices(logits, row_starts, row_ends, topk=64)


def test_dsa_decode_topk_gluon_long_row_uses_radix_path() -> None:
    device = "cuda"
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

    topk_slots, topk_lens = gluon_dsa_decode_topk_fp8(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=head_dim**-0.5,
        index_k_cache=packed_index_k,
    )

    expected = torch.arange(topk, device=device, dtype=torch.int32)
    torch.testing.assert_close(topk_lens.cpu(), torch.tensor([topk], dtype=torch.int32))
    torch.testing.assert_close(torch.sort(topk_slots[0]).values.cpu(), expected.cpu())


def test_dsa_prefill_topk_gluon_long_row_uses_radix_path() -> None:
    device = "cuda"
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

    workspace_indices, topk_lens = gluon_dsa_prefill_topk_fp8(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=head_dim**-0.5,
        index_k_cache=packed_index_k,
        page_size=page_size,
    )

    expected = torch.arange(topk, device=device, dtype=torch.int32)
    torch.testing.assert_close(topk_lens.cpu(), torch.tensor([topk], dtype=torch.int32))
    torch.testing.assert_close(
        torch.sort(workspace_indices[0]).values.cpu(),
        expected.cpu(),
    )


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
    "mode,api",
    [
        pytest.param("decode", gluon_dsa_decode, id="decode"),
        pytest.param("prefill", gluon_dsa_prefill, id="prefill"),
    ],
)
@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_e4m3"),
        pytest.param(torch.float8_e5m2, id="q_e5m2"),
    ],
)
def test_dsa_with_sparse_kvcache(mode: str, api, q_dtype: torch.dtype) -> None:
    device = "cuda"
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
    "mode,api",
    [
        pytest.param("decode", gluon_dsa_decode, id="decode"),
        pytest.param("prefill", gluon_dsa_prefill, id="prefill"),
    ],
)
@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_e4m3"),
        pytest.param(torch.float8_e5m2, id="q_e5m2"),
    ],
)
@pytest.mark.parametrize(
    "kv_lora_rank",
    [
        pytest.param(128, id="rank_128"),
        pytest.param(512, id="rank_512"),
    ],
)
def test_dsa_dense_kvcache(
    mode: str,
    api,
    q_dtype: torch.dtype,
    kv_lora_rank: int,
) -> None:
    device = "cuda"
    tokens = 3
    num_heads = 2
    num_slots = 16
    topk = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 192 if kv_lora_rank == 512 else 128
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

    out = api(
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
    "api",
    [
        pytest.param(gluon_dsa_decode, id="decode"),
        pytest.param(gluon_dsa_prefill, id="prefill"),
    ],
)
@pytest.mark.parametrize(
    ("valid_lengths", "max_seqlen_k"),
    [
        pytest.param(
            (0, 1, 31, 32, 33, 63, 64, 65, 2048),
            2112,
            id="static-full-width",
        ),
        pytest.param(
            (0, 1, 31, 32, 33, 63, 64, 65),
            65,
            id="dynamic-short-list",
        ),
        pytest.param(
            (2048,),
            2112,
            id="single-row-full-width",
        ),
        pytest.param(
            (2048, 2048),
            2112,
            id="multi-row-full-width",
        ),
        pytest.param(
            (1024,),
            1024,
            id="registered-width-1024",
        ),
        pytest.param(
            (0, 1, 31, 32, 33, 65, 1025),
            1025,
            id="dynamic-split-rows",
        ),
    ],
)
@pytest.mark.parametrize(
    "fp8_dtype",
    [
        pytest.param(torch.float8_e4m3fn, id="e4m3"),
        pytest.param(torch.float8_e5m2, id="e5m2"),
    ],
)
def test_dsa_dense_fp8_glm52_production_shape(
    api,
    valid_lengths: tuple[int, ...],
    max_seqlen_k: int,
    fp8_dtype: torch.dtype,
) -> None:
    device = "cuda"
    tokens = len(valid_lengths)
    num_heads = 16
    num_slots = 2112
    topk = 2048
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 192
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    gen = _generator(device, 401)

    q = _randn_bf16(
        (tokens, num_heads, kv_lora_rank + qk_rope_head_dim),
        device=device,
        generator=gen,
    ).to(fp8_dtype)
    kv_cache = _randn_bf16(
        (num_slots, kv_lora_rank + qk_rope_head_dim),
        device=device,
        generator=gen,
    ).to(fp8_dtype)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor(valid_lengths, device=device, dtype=torch.int32)
    for token, count in enumerate(topk_lens.tolist()):
        topk_slots[token, :count] = torch.randperm(
            max_seqlen_k, device=device, generator=gen, dtype=torch.int32
        )[:count]

    out = api(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
        q_len_per_req=4 if api is gluon_dsa_decode else 1,
    )

    ref = _dsa_reference(
        q,
        kv_cache[:, :kv_lora_rank],
        kv_cache[:, kv_lora_rank:],
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    empty_rows = torch.tensor(
        [row for row, valid_len in enumerate(valid_lengths) if valid_len == 0],
        device=device,
        dtype=torch.int64,
    )
    empty_out = out.index_select(0, empty_rows)
    torch.testing.assert_close(
        empty_out, torch.zeros_like(empty_out), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


def test_dsa_decode_sparse_kvcache_trims_large_topk_for_tiny_lens() -> None:
    device = "cuda"
    tokens = 2
    num_heads = 2
    num_slots = 8
    topk = 2048
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    sparse_kv, dequant_latent = _pack_sparse_kv(latent, rope)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([1, 3], device=device, dtype=torch.int32)
    topk_slots[0, 0] = 2
    topk_slots[1, :3] = torch.tensor([0, 5, 7], device=device, dtype=torch.int32)

    out = gluon_dsa_decode(
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


@pytest.mark.skipif(
    not is_cdna4(),
    reason="GLM-5.3-Flash native FP8 MFMA decode is specific to gfx950",
)
def test_dsa_decode_glm53_flash_native_fp8_mfma() -> None:
    device = "cuda"
    num_heads = 16
    num_slots = 2051
    topk = 2051
    valid_topk = 2048
    kv_lora_rank = 512
    qk_nope_head_dim = 256
    generator = _generator(device, 1201)

    q = _randn_bf16(
        (1, num_heads, kv_lora_rank), device=device, generator=generator
    ).to(torch.float8_e4m3fn)
    kv_cache = _randn_bf16(
        (num_slots, kv_lora_rank), device=device, generator=generator
    ).to(torch.float8_e4m3fn)
    topk_slots = torch.full((1, topk), -1, device=device, dtype=torch.int32)
    topk_slots[0, :valid_topk] = torch.randperm(
        num_slots, device=device, generator=generator
    )[:valid_topk]
    topk_lens = torch.tensor([valid_topk], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim)

    out = gluon_dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=0,
        softmax_scale=softmax_scale,
        page_size=64,
    )
    selected = kv_cache[topk_slots[0, :valid_topk].long()].float()
    probs = torch.softmax(q[0].float() @ selected.T * softmax_scale, dim=-1)
    ref = probs @ selected

    assert out.shape == (1, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out[0].float(), ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    ("tokens", "topk", "expected_splits"),
    ((1, 1024, 4), (16, 1024, 4), (32, 2048, 8), (64, 2048, 4)),
)
def test_dsa_native_fp8_split_schedule(
    tokens: int, topk: int, expected_splits: int
) -> None:
    assert (
        gfx950_dsa_backend._select_num_kv_splits(
            num_tokens=tokens,
            num_heads=16,
            topk_width=topk,
            max_seqlen_k=topk,
            is_fp8=True,
            native_fp8=True,
        )
        == expected_splits
    )


@pytest.mark.parametrize(
    (
        "tokens",
        "topk_width",
        "max_seqlen_k",
        "qk_rope_head_dim",
        "expected_splits",
    ),
    (
        (1, 2048, 2047, 0, 1),
        (16, 2048, 2047, 0, 16),
        (32, 2048, 2047, 0, 1),
        (64, 2048, 2047, 0, 4),
        (16, 2051, 131072, 0, 16),
        (64, 2049, 131072, 0, 4),
        (64, 2051, 131072, 0, 4),
        (64, 2051, 1961, 0, 1),
        (64, 2051, 131072, 64, 1),
        (64, 2052, 131072, 0, 1),
    ),
)
def test_dsa_glm53_bf16_split_schedule(
    tokens: int,
    topk_width: int,
    max_seqlen_k: int,
    qk_rope_head_dim: int,
    expected_splits: int,
) -> None:
    assert (
        gfx950_dsa_backend._select_num_kv_splits(
            num_tokens=tokens,
            num_heads=16,
            topk_width=topk_width,
            max_seqlen_k=max_seqlen_k,
            is_fp8=False,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        == expected_splits
    )


@pytest.mark.skipif(
    not is_cdna4(),
    reason="GLM-5.3-Flash BF16 decode is specific to gfx950",
)
@pytest.mark.parametrize("tokens", (16, 64))
def test_dsa_decode_glm53_flash_bf16_split_mfma(tokens: int) -> None:
    device = "cuda"
    num_heads = 16
    num_slots = 256
    topk = 2051
    kv_lora_rank = 512
    qk_nope_head_dim = 256
    generator = _generator(device, 1204 + tokens)

    q = _randn_bf16(
        (tokens, num_heads, kv_lora_rank), device=device, generator=generator
    )
    kv_cache = _randn_bf16(
        (num_slots, kv_lora_rank), device=device, generator=generator
    )
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor(
        [(row * 17) % (num_slots + 1) for row in range(tokens)],
        device=device,
        dtype=torch.int32,
    )
    for row, length in enumerate(topk_lens.tolist()):
        topk_slots[row, :length] = torch.randperm(
            num_slots, device=device, generator=generator
        )[:length]
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim)

    out = gluon_dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=131072,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=0,
        softmax_scale=softmax_scale,
        page_size=64,
        q_len_per_req=4,
    )
    ref = _dsa_reference(
        q,
        kv_cache,
        torch.empty((num_slots, 0), dtype=q.dtype, device=device),
        topk_slots,
        topk_lens,
        softmax_scale,
    )

    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


@pytest.mark.skipif(
    not is_cdna4(),
    reason="the tested DSA kernel currently only supports gfx950",
)
def test_dsa_decode_glm53_flash_bf16_split_graph_tracks_live_lengths() -> None:
    device = "cuda"
    tokens = 16
    num_heads = 16
    num_slots = 512
    topk = 2051
    kv_lora_rank = 512
    qk_nope_head_dim = 256
    generator = _generator(device, 1220)
    q = _randn_bf16(
        (tokens, num_heads, kv_lora_rank), device=device, generator=generator
    )
    kv_cache = _randn_bf16(
        (num_slots, kv_lora_rank), device=device, generator=generator
    )
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.empty(tokens, device=device, dtype=torch.int32)
    out = torch.empty(
        (tokens, num_heads, kv_lora_rank), device=device, dtype=torch.bfloat16
    )
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim)

    def invoke() -> None:
        gluon_dsa_decode(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=131072,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=0,
            softmax_scale=softmax_scale,
            page_size=64,
            q_len_per_req=4,
            out=out,
        )

    topk_lens.fill_(13)
    topk_slots[:, :13] = torch.arange(13, device=device, dtype=torch.int32)
    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()

    for visible in (13, 449):
        topk_slots.fill_(-1)
        topk_lens.fill_(visible)
        selected = torch.arange(visible, device=device, dtype=torch.int32)
        for row in range(tokens):
            topk_slots[row, :visible] = torch.roll(selected, row)
        graph.replay()
        torch.cuda.synchronize()
        ref = _dsa_reference(
            q,
            kv_cache,
            torch.empty((num_slots, 0), dtype=q.dtype, device=device),
            topk_slots,
            topk_lens,
            softmax_scale,
        )
        torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


@pytest.mark.skipif(
    not is_cdna4(),
    reason="GLM-5.3-Flash BF16 prefill is specific to gfx950",
)
def test_dsa_prefill_glm53_flash_bf16_dense_mfma(monkeypatch) -> None:
    device = "cuda"
    num_heads = 16
    num_slots = 2051
    topk = 2051
    valid_topk = 2048
    kv_lora_rank = 512
    qk_nope_head_dim = 256
    generator = _generator(device, 1202)

    q = _randn_bf16((1, num_heads, kv_lora_rank), device=device, generator=generator)
    kv_cache = _randn_bf16(
        (num_slots, kv_lora_rank), device=device, generator=generator
    )
    topk_slots = torch.full((1, topk), -1, device=device, dtype=torch.int32)
    topk_slots[0, :valid_topk] = torch.randperm(
        num_slots, device=device, generator=generator
    )[:valid_topk]
    topk_lens = torch.tensor([valid_topk], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim)

    mfma_calls = 0
    mfma_kernel = gfx950_dsa_backend._dsa_dense_mfma_kv_kernel

    class _TrackedMfmaKernel:
        def __getitem__(self, grid):
            launch = mfma_kernel[grid]

            def track_launch(*args, **kwargs):
                nonlocal mfma_calls
                mfma_calls += 1
                return launch(*args, **kwargs)

            return track_launch

    monkeypatch.setattr(
        gfx950_dsa_backend, "_dsa_dense_mfma_kv_kernel", _TrackedMfmaKernel()
    )
    out = gluon_dsa_prefill(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=0,
        softmax_scale=softmax_scale,
        page_size=64,
    )
    selected = kv_cache[topk_slots[0, :valid_topk].long()].float()
    probs = torch.softmax(q[0].float() @ selected.T * softmax_scale, dim=-1)
    ref = probs @ selected

    assert mfma_calls == 1
    assert out.shape == (1, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out[0].float(), ref, rtol=8e-2, atol=8e-2)


@pytest.mark.skipif(
    not is_cdna4(),
    reason="GLM-5.3-Flash padded prefill is specific to gfx950",
)
def test_dsa_prefill_glm53_flash_zero_length_rows_are_zero() -> None:
    device = "cuda"
    tokens = 4
    num_heads = 16
    num_slots = 64
    topk = 2051
    kv_lora_rank = 512
    generator = _generator(device, 1203)
    q = _randn_bf16(
        (tokens, num_heads, kv_lora_rank),
        device=device,
        generator=generator,
    )
    q[2:].fill_(float("nan"))
    kv_cache = _randn_bf16(
        (num_slots, kv_lora_rank),
        device=device,
        generator=generator,
    )
    kv_cache[0].fill_(float("nan"))
    topk_slots = torch.full(
        (tokens, topk),
        -1,
        device=device,
        dtype=torch.int32,
    )
    topk_lens = torch.tensor([16, 12, 0, 0], device=device, dtype=torch.int32)
    for row, length in enumerate(topk_lens[:2].tolist()):
        topk_slots[row, :length] = (
            torch.randperm(
                num_slots - 1,
                generator=generator,
                device=device,
            )[:length]
            + 1
        )
    softmax_scale = kv_lora_rank**-0.5

    out = gluon_dsa_prefill(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=256,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=0,
        softmax_scale=softmax_scale,
        page_size=64,
    )

    for row, length in enumerate(topk_lens[:2].tolist()):
        selected = kv_cache[topk_slots[row, :length].long()].float()
        probs = torch.softmax(q[row].float() @ selected.T * softmax_scale, dim=-1)
        torch.testing.assert_close(
            out[row].float(),
            probs @ selected,
            rtol=8e-2,
            atol=8e-2,
        )
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[2:]).item() == 0


@pytest.mark.parametrize(
    "case",
    _GLM52_DSA_CASES,
    ids=lambda case: case.name,
)
def test_dsa_glm52_selected_attention_cases(case: _DSACase) -> None:
    device = "cuda"
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
    }
    if case.mode == "decode":
        out = gluon_dsa_decode(q_len_per_req=case.q_len_per_req, **common_kwargs)
    else:
        out = gluon_dsa_prefill(**common_kwargs)

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
