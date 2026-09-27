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

"""DeepSeek V4 decode context parallelism kernels.

Virtual block translation, the LSE merge of per-shard attention partials, the
sink applied once after merging, and the owner-only compressed cache stores.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv4 import (
    dsv4_decode,
    dsv4_decode_supports_partials,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
    dcp_apply_sink,
    dcp_weight_for_reduce_scatter,
    normalize_dcp_partials,
)
from tokenspeed_kernel.ops.attention.dsv4.triton import (
    dsv4_fused_sparse_compress_cache_insert,
)
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_slots_to_local
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

HEAD_DIM = 512
PAGE_ROWS = 64
SWA_TOKEN_STRIDE = 576  # 448 FP8 nope + 64 BF16 rope per row (fp8_swa_page_planar).
SWA_ROW_BYTES = SWA_TOKEN_STRIDE + 8  # plus one FP8 scale per 64-wide nope block.
# The arena aligns every page's stride to the token stride; SM100 FlashMLA
# requires that alignment for its TMA loads.
SWA_PAGE_BYTES = -(-PAGE_ROWS * SWA_ROW_BYTES // SWA_TOKEN_STRIDE) * SWA_TOKEN_STRIDE


def _reference_translation(slots, rows, virtual_count, degree, rank):
    """Owner and local slot of every virtual slot, straight from the contract."""
    block = slots.clamp_min(0) // rows
    owned = (
        (slots >= rows)
        & (slots < virtual_count * rows)
        & ((block - 1) % degree == rank)
    )
    local = ((block - 1) // degree + 1) * rows + slots.clamp_min(0) % rows
    return torch.where(owned, local, torch.zeros_like(local)), owned


# ---------------------------------------------------------------------------
# Virtual block translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 64, 256])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("parents,packing", [(3, 1), (2, 4), (5, 2)])
def test_owned_pages_partition_the_virtual_space(rows, degree, parents, packing):
    virtual_count = 1 + parents * degree * packing
    local_pages = 1 + parents * packing
    slots = torch.arange(-3, virtual_count * rows + 5, dtype=torch.int64)
    covered = torch.zeros(virtual_count, dtype=torch.int64)
    for rank in range(degree):
        local, owned = virtual_slots_to_local(
            slots,
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        expected_local, expected_owned = _reference_translation(
            slots, rows, virtual_count, degree, rank
        )
        assert torch.equal(local, expected_local)
        assert torch.equal(owned, expected_owned)
        # Unowned slots park on slot 0; nothing negative, null or out of range is owned.
        assert not owned[slots < rows].any()
        assert not owned[slots >= virtual_count * rows].any()
        assert (local[~owned] == 0).all()
        pages = local[owned] // rows
        assert pages.min() >= 1 and pages.max() == local_pages - 1
        # Each owned local page receives exactly ``rows`` slots.
        assert torch.equal(
            torch.bincount(pages, minlength=local_pages)[1:],
            torch.full((local_pages - 1,), rows),
        )
        covered += torch.bincount(slots[owned] // rows, minlength=virtual_count)
    # Every non-null virtual block has exactly one owner across the ranks.
    assert covered[0] == 0
    assert (covered[1:] == rows).all()


@requires_cuda
@pytest.mark.parametrize("rows", [1, 64])
@pytest.mark.parametrize("degree", [1, 4])
def test_virtual_slot_translation_matches_between_cpu_and_cuda(rows, degree):
    virtual_count = 1 + 7 * degree * 4
    slots = torch.randint(-5, virtual_count * rows + 9, (1531,), dtype=torch.int64)
    for rank in range(degree):
        cpu_local, cpu_owned = virtual_slots_to_local(
            slots,
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        cuda_local, cuda_owned = virtual_slots_to_local(
            slots.cuda(),
            rows_per_page=rows,
            virtual_block_count=virtual_count,
            degree=degree,
            rank=rank,
        )
        assert torch.equal(cuda_local.cpu(), cpu_local)
        assert torch.equal(cuda_owned.cpu(), cpu_owned)


@requires_cuda
def test_virtual_slot_translation_reuses_caller_outputs():
    slots = torch.arange(0, 400, dtype=torch.int32, device="cuda").view(20, 20)
    out = torch.empty_like(slots)
    mask = torch.empty(slots.shape, dtype=torch.bool, device="cuda")
    local, owned = virtual_slots_to_local(
        slots,
        rows_per_page=1,
        virtual_block_count=401,
        degree=2,
        rank=1,
        out=out,
        owner_mask=mask,
    )
    assert local.data_ptr() == out.data_ptr()
    assert owned.data_ptr() == mask.data_ptr()
    expected_local, expected_owned = _reference_translation(
        slots.long().cpu(), 1, 401, 2, 1
    )
    assert torch.equal(local.cpu().long(), expected_local)
    assert torch.equal(owned.cpu(), expected_owned)


def test_virtual_slot_translation_rejects_invalid_geometry():
    slots = torch.arange(8, dtype=torch.int64)
    with pytest.raises(ValueError, match="owner rank"):
        virtual_slots_to_local(
            slots, rows_per_page=1, virtual_block_count=9, degree=2, rank=2
        )
    with pytest.raises(ValueError, match="usable pages"):
        virtual_slots_to_local(
            slots, rows_per_page=1, virtual_block_count=1, degree=1, rank=0
        )
    with pytest.raises(TypeError):
        virtual_slots_to_local(
            slots.float(), rows_per_page=1, virtual_block_count=9, degree=1, rank=0
        )


# ---------------------------------------------------------------------------
# Partial normalization and the LSE merge
# ---------------------------------------------------------------------------


def _merge_reference(partials, lses, sink):
    """Combine per-shard no-sink partials by their LSE, then apply the sink once."""
    stacked_lse = torch.stack(lses, dim=0)  # [shards, tokens, heads]
    finite = torch.where(
        torch.isfinite(stacked_lse),
        stacked_lse,
        torch.full_like(stacked_lse, -torch.inf),
    )
    total = torch.logsumexp(finite, dim=0)
    weights = torch.exp(finite - total)
    weights = torch.nan_to_num(weights, nan=0.0)
    merged = sum(
        w[..., None] * p.float() for w, p in zip(weights, partials, strict=True)
    )
    factor = torch.where(
        torch.isinf(total), torch.zeros_like(total), torch.sigmoid(total - sink)
    )
    return merged * factor[..., None], total


@requires_cuda
def test_normalize_marks_empty_and_non_finite_rows_as_empty():
    tokens, heads, dim = 5, 4, 8
    output = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    original = output.clone()
    lse = torch.randn(tokens, heads, device="cuda")
    lse[3, 1] = torch.inf  # FlashMLA's value for a row that attended to nothing.
    lse[4, 2] = torch.nan
    swa_lens = torch.tensor([3, 0, 0, 2, 5], device="cuda", dtype=torch.int32)
    extra_lens = torch.tensor([0, 0, 4, 1, 1], device="cuda", dtype=torch.int32)

    normalized_output, normalized_lse = normalize_dcp_partials(
        output, lse, swa_lens, extra_lens
    )

    assert normalized_output.data_ptr() == output.data_ptr()
    empty_rows = torch.tensor([False, True, False, False, False], device="cuda")
    assert torch.equal(
        normalized_output[empty_rows], torch.zeros_like(original[empty_rows])
    )
    assert torch.equal(
        normalized_output[~empty_rows][:, [0, 3]], original[~empty_rows][:, [0, 3]]
    )
    assert torch.isneginf(normalized_lse[1]).all()
    assert torch.isneginf(normalized_lse[3, 1]) and torch.isneginf(normalized_lse[4, 2])
    assert torch.equal(
        normalized_output[3, 1], torch.zeros(dim, device="cuda", dtype=torch.bfloat16)
    )
    kept = torch.isfinite(lse) & ~empty_rows[:, None]
    assert torch.equal(normalized_lse[kept], lse[kept])


@requires_cuda
@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_weight_and_sink_match_the_reference_merge_with_empty_shards(degree, dtype):
    torch.manual_seed(degree)
    tokens, local_heads, dim = 6, 4, 16
    heads = local_heads * degree
    partials = [
        torch.randn(tokens, heads, dim, device="cuda", dtype=dtype)
        for _ in range(degree)
    ]
    lses = [torch.randn(tokens, heads, device="cuda") * 3 for _ in range(degree)]
    # Token 0 is empty on shard 1, token 1 is empty everywhere, token 2 leaks
    # FlashMLA's +inf from shard 0 without a matching zero partial.
    lses[1][0] = -torch.inf
    partials[1][0] = 0
    for rank in range(degree):
        lses[rank][1] = -torch.inf
        partials[rank][1] = 0
    lses[0][2] = torch.inf
    sink = torch.randn(heads, device="cuda")

    expected, expected_lse = _merge_reference(partials, lses, sink)
    all_lse = torch.stack(lses, dim=0)
    summed = None
    global_lses = []
    for rank in range(degree):
        weighted, global_lse = dcp_weight_for_reduce_scatter(
            partials[rank], all_lse, rank
        )
        assert (
            weighted.shape == (heads, tokens, dim) and weighted.dtype == torch.float32
        )
        assert global_lse.shape == (tokens, local_heads)
        summed = weighted if summed is None else summed + weighted
        global_lses.append(global_lse)
    actual = torch.cat(
        [
            dcp_apply_sink(
                summed[rank * local_heads : (rank + 1) * local_heads].movedim(0, 1),
                global_lses[rank],
                sink[rank * local_heads : (rank + 1) * local_heads].contiguous(),
                dtype=dtype,
            )
            for rank in range(degree)
        ],
        dim=1,
    )
    global_lse = torch.cat(global_lses, dim=1)

    assert not actual.isnan().any() and not global_lse.isnan().any()
    assert torch.equal(torch.isneginf(global_lse), torch.isneginf(expected_lse))
    finite = torch.isfinite(expected_lse)
    torch.testing.assert_close(
        global_lse[finite], expected_lse[finite], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))


@requires_cuda
def test_weight_propagates_an_unexpected_nan_lse():
    partials = torch.ones(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    all_lse = torch.zeros(2, 2, 4, device="cuda")
    all_lse[1, 0, 1] = torch.nan  # shard 1, token 0, head 1 (rank 0's head slice)
    weighted, global_lse = dcp_weight_for_reduce_scatter(partials, all_lse, 0)
    assert weighted[1, 0].isnan().all()
    assert not weighted[0].isnan().any() and not weighted[2:].isnan().any()
    assert not weighted[1, 1].isnan().any()
    assert global_lse[0, 1].isnan()
    assert not global_lse[0, 0].isnan() and not global_lse[1].isnan().any()


@requires_cuda
def test_weight_and_sink_reject_inconsistent_shapes():
    partials = torch.ones(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="topology"):
        dcp_weight_for_reduce_scatter(partials, torch.zeros(3, 2, 4, device="cuda"), 0)
    with pytest.raises(ValueError, match="shapes disagree"):
        dcp_weight_for_reduce_scatter(partials, torch.zeros(2, 2, 3, device="cuda"), 0)
    with pytest.raises(ValueError, match="shapes disagree"):
        dcp_apply_sink(
            partials.float(),
            torch.zeros(2, 4, device="cuda"),
            torch.zeros(2, device="cuda"),
            dtype=torch.bfloat16,
        )


# ---------------------------------------------------------------------------
# Kernel capability and the no-sink LSE contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "traits,expected",
    [
        pytest.param({}, False, id="undeclared"),
        pytest.param({"sinks": frozenset({True})}, False, id="sink-only"),
        pytest.param({"sinks": frozenset({False})}, False, id="missing-lse"),
        pytest.param({"return_lse": frozenset({True})}, False, id="missing-sinks"),
        pytest.param(
            {"return_lse": frozenset({True}), "sinks": frozenset({True})},
            False,
            id="lse-includes-sink",
        ),
        pytest.param(
            {"return_lse": frozenset({False}), "sinks": frozenset({False})},
            False,
            id="no-lse",
        ),
        pytest.param(
            {"return_lse": frozenset({True}), "sinks": frozenset({False})},
            True,
            id="no-sink-lse",
        ),
        pytest.param(
            {"return_lse": frozenset({False, True}), "sinks": frozenset({False, True})},
            True,
            id="optional-sink-and-lse",
        ),
    ],
)
def test_decode_partials_require_explicit_no_sink_lse_support(
    fresh_registry, h100_platform, traits, expected
):
    KernelRegistry.get().register(
        KernelSpec(
            name="test_dsv4_decode",
            family="attention",
            mode="dsv4_decode",
            traits=traits,
        ),
        lambda: None,
    )

    assert dsv4_decode_supports_partials(h100_platform) is expected


def test_decode_partials_need_a_kernel_registered_for_the_platform(
    a100_platform, mi350_platform, h100_platform
):
    # No portable kernel returns a no-sink LSE yet; only FlashMLA does, and
    # it is registered for NVIDIA SM90+ hosts only.
    assert not dsv4_decode_supports_partials(a100_platform)
    assert not dsv4_decode_supports_partials(mi350_platform)
    platform = current_platform()
    assert dsv4_decode_supports_partials(h100_platform) == (
        platform.is_nvidia and platform.is_hopper_plus
    )


def test_decode_refuses_a_sink_together_with_lse():
    q = torch.empty(1, 64, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="no-sink LSE"):
        dsv4_decode(
            q=q,
            swa_kv_cache=torch.empty(1, SWA_PAGE_BYTES, dtype=torch.uint8),
            swa_slots=torch.zeros(1, 8, dtype=torch.int32),
            swa_lens=torch.zeros(1, dtype=torch.int32),
            swa_page_size=PAGE_ROWS,
            attn_sink=torch.zeros(64),
            softmax_scale=HEAD_DIM**-0.5,
            return_lse=True,
        )


def _page_planar_cache(pages, page_size):
    """A random FP8 SWA-layout cache plus its BF16 row reference [pages*page_size, 512]."""
    assert page_size == PAGE_ROWS
    nope = torch.randn(pages, page_size, 448, device="cuda").to(torch.float8_e4m3fn)
    rope = torch.randn(pages, page_size, 64, device="cuda", dtype=torch.bfloat16)
    # 127 is the UE8M0 scale for 1.0, so every quant block dequantizes as-is.
    cache = torch.full((pages, SWA_PAGE_BYTES), 127, device="cuda", dtype=torch.uint8)
    payload = torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
    cache[:, : page_size * SWA_TOKEN_STRIDE] = payload.reshape(pages, -1)
    reference = torch.cat((nope.to(torch.bfloat16), rope), dim=-1).reshape(-1, HEAD_DIM)
    return cache, reference


def _selected_attention_reference(q, kv, slots, lens, sink):
    """Softmax over each token's selected rows with the sink, plus the no-sink LSE."""
    result = torch.zeros_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], device=q.device)
    for token in range(q.shape[0]):
        selected = slots[token, : int(lens[token])].long()
        selected = selected[(selected >= 0) & (selected < kv.shape[0])]
        keys = kv[selected].float()
        logits = q[token].float() @ keys.T * HEAD_DIM**-0.5
        lse[token] = torch.logsumexp(logits, dim=1) if keys.shape[0] else -torch.inf
        probabilities = torch.cat((logits, sink[:, None]), dim=1).softmax(dim=1)[:, :-1]
        result[token] = (probabilities @ keys).to(q.dtype)
    return result, lse


@requires_cuda
@pytest.mark.parametrize("heads,degree", [(64, 2), (64, 4), (128, 2), (128, 8)])
def test_flashmla_partials_merge_to_the_full_softmax_with_empty_shards(
    require, heads, degree
):
    require("attention", "dsv4_decode", "flashmla", torch.bfloat16, "q")
    from tokenspeed_kernel.ops.attention.dsv4.cuda import reset_dsv4_tile_metadata

    torch.manual_seed(74)
    tokens = 5
    swa_cache, swa_kv = _page_planar_cache(3, PAGE_ROWS)
    extra_cache, extra_kv = _page_planar_cache(4, PAGE_ROWS)
    q = torch.randn(tokens, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    swa_slots = torch.arange(128, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    swa_slots[0, 10] = -1
    swa_lens = torch.tensor([91, 1, 40, 0, 0], device="cuda", dtype=torch.int32)
    extra_width = 128
    extra_slots = torch.arange(extra_width, device="cuda", dtype=torch.int32).repeat(
        tokens, 1
    )
    extra_slots[1, 5] = -1
    # Token 1 selects three compressed rows, so some shards own none of them;
    # token 4 attends to nothing at all.
    extra_lens = torch.tensor([120, 3, 0, 64, 0], device="cuda", dtype=torch.int32)

    swa_selected = swa_slots.masked_fill(
        torch.arange(128, device="cuda")[None] >= swa_lens[:, None], -1
    )
    extra_selected = (extra_slots + swa_kv.shape[0]).masked_fill(
        torch.arange(extra_width, device="cuda")[None] >= extra_lens[:, None], -1
    )
    selected = torch.cat((swa_selected, extra_selected), dim=1)
    expected, expected_lse = _selected_attention_reference(
        q,
        torch.cat((swa_kv, extra_kv), dim=0),
        selected,
        torch.full((tokens,), selected.shape[1], device="cuda", dtype=torch.int32),
        sink,
    )

    # Shard compressed pages cyclically and keep the scan length, as the
    # runtime does; only rank 0 counts the replicated SWA cache.
    partials, lses = [], []
    for rank in range(degree):
        owned = ((extra_slots // PAGE_ROWS) % degree == rank) & (extra_slots >= 0)
        shard_slots = torch.where(owned, extra_slots, torch.full_like(extra_slots, -1))
        in_prefix = torch.arange(extra_width, device="cuda")[None] < extra_lens[:, None]
        valid_lens = (owned & in_prefix).sum(dim=1, dtype=torch.int32)
        rank_swa_lens = swa_lens if rank == 0 else torch.zeros_like(swa_lens)
        reset_dsv4_tile_metadata()
        out, lse = dsv4_decode(
            q=q,
            swa_kv_cache=swa_cache,
            swa_slots=swa_slots,
            swa_lens=rank_swa_lens,
            swa_page_size=PAGE_ROWS,
            attn_sink=None,
            softmax_scale=HEAD_DIM**-0.5,
            extra_kv_cache=extra_cache,
            extra_slots=shard_slots,
            extra_lens=extra_lens,
            extra_page_size=PAGE_ROWS,
            solution="flashmla",
            return_lse=True,
        )
        out, lse = normalize_dcp_partials(
            out, lse.squeeze(-1), rank_swa_lens, valid_lens
        )
        partials.append(out)
        lses.append(lse)

    all_lse = torch.stack(lses, dim=0)
    local_heads = heads // degree
    summed = None
    global_lses = []
    for rank in range(degree):
        weighted, global_lse = dcp_weight_for_reduce_scatter(
            partials[rank], all_lse, rank
        )
        summed = weighted if summed is None else summed + weighted
        global_lses.append(global_lse)
    actual = torch.cat(
        [
            dcp_apply_sink(
                summed[rank * local_heads : (rank + 1) * local_heads].movedim(0, 1),
                global_lses[rank],
                sink[rank * local_heads : (rank + 1) * local_heads].contiguous(),
                dtype=q.dtype,
            )
            for rank in range(degree)
        ],
        dim=1,
    )
    global_lse = torch.cat(global_lses, dim=1)

    assert not actual.isnan().any() and not global_lse.isnan().any()
    assert torch.equal(torch.isinf(global_lse), torch.isinf(expected_lse))
    finite = torch.isfinite(expected_lse)
    torch.testing.assert_close(
        global_lse[finite], expected_lse[finite], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)


# ---------------------------------------------------------------------------
# Owner-only compressed cache stores
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_masked_compress_stores_leave_every_other_byte_untouched(compress_ratio):
    torch.manual_seed(compress_ratio)
    tokens = 8
    positions = torch.arange(
        tokens, device="cuda", dtype=torch.int32
    ) * compress_ratio + (compress_ratio - 1)
    token_to_req = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    state_rows = compress_ratio
    state_cache = torch.randn(
        tokens * 2, state_rows, HEAD_DIM * 2, device="cuda", dtype=torch.float32
    )
    kv_cache = torch.randint(
        0, 255, (tokens + 2, SWA_PAGE_BYTES), device="cuda", dtype=torch.uint8
    )
    before = kv_cache.clone()
    kv_slots = (torch.arange(tokens, device="cuda", dtype=torch.int64) + 1) * PAGE_ROWS
    mask = torch.tensor([True, False] * (tokens // 2), device="cuda")
    dsv4_fused_sparse_compress_cache_insert(
        state_cache=state_cache,
        token_to_req_indices=token_to_req,
        positions=positions,
        compressor_slot_mapping=torch.arange(tokens, device="cuda", dtype=torch.int32),
        block_table=torch.arange(tokens, device="cuda", dtype=torch.int32).view(
            1, tokens
        ),
        compressor_block_size=1,
        rms_norm_weight=torch.ones(HEAD_DIM, device="cuda", dtype=torch.float32),
        rms_norm_eps=1e-6,
        cos_sin_cache=torch.randn(
            positions.max().item() + 1, 64, device="cuda", dtype=torch.float32
        ),
        kv_cache_2d=kv_cache,
        kv_slot_mapping=kv_slots,
        kv_cache_block_size=PAGE_ROWS,
        compress_ratio=compress_ratio,
        overlap=compress_ratio == 4,
        block_table_base_offsets=None,
        kv_write_mask=mask,
    )
    torch.cuda.synchronize()
    changed_pages = (kv_cache != before).any(dim=1)
    written_pages = (kv_slots // PAGE_ROWS)[mask]
    unwritten_pages = (kv_slots // PAGE_ROWS)[~mask]
    assert changed_pages[written_pages].all()
    assert not changed_pages[unwritten_pages].any()
    # Masked tokens park on slot 0; the null page must stay untouched too.
    assert not changed_pages[0]
    assert not changed_pages[tokens + 1]


@requires_cuda
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_mxfp4_sharded_index_candidates(degree):
    from tokenspeed_kernel.ops.attention.dsv4.triton import triton_dsv4_index_candidates

    torch.manual_seed(943)
    device = "cuda"
    pages, page_size, heads, topk = 10, 64, 32, 512
    table = torch.tensor(
        [[5, 2, 8, 1, 6, 3, 4, 7, 9]], device=device, dtype=torch.int32
    )
    q = torch.randint(0, 256, (3, heads, 64), device=device, dtype=torch.uint8)
    scales = torch.full((3, heads), 0x7F7F7F7F, device=device, dtype=torch.int32)
    cache = torch.randint(
        0, 256, (pages, page_size * 68), device=device, dtype=torch.uint8
    )
    cache[:, page_size * 64 :] = 127
    weights = torch.randn(3, heads, device=device)
    lengths = torch.tensor([0, 37, 573], device=device, dtype=torch.int32)
    requests = torch.zeros(3, device=device, dtype=torch.int32)

    def unpack(x):
        code = torch.stack((x & 15, x >> 4), -1).flatten(-2).long()
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
        return lut[code & 7] * torch.where(code < 8, 1.0, -1.0)

    query = unpack(q)
    keys = unpack(cache[:, : page_size * 64].reshape(pages, page_size, 64))[
        table[0].long()
    ].reshape(-1, 128)
    reference = (
        torch.einsum("thd,sd->ths", query, keys).relu() * weights.unsqueeze(-1)
    ).sum(1)
    reference.masked_fill_(
        torch.arange(keys.shape[0], device=device) >= lengths[:, None], -float("inf")
    )
    candidates, values = [], []
    for rank in range(degree):
        local = cache[[0] + list(range(rank + 1, pages, degree))].contiguous()
        local_table = torch.where(
            (table - 1) % degree == rank, (table - 1) // degree + 1, -1
        )
        indices, scores = triton_dsv4_index_candidates(
            (q, scales),
            weights,
            local,
            local_table,
            requests,
            lengths,
            page_size=page_size,
            topk=topk,
        )
        valid = indices >= 0
        torch.testing.assert_close(
            scores[valid],
            reference.gather(1, indices.clamp_min(0).long())[valid],
            rtol=2e-5,
            atol=2e-3,
        )
        assert (scores[~valid] == -float("inf")).all()
        candidates.append(indices)
        values.append(scores)
        # Refresh lengths in place: captured kernels must not retain old validity.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_indices, captured_scores = triton_dsv4_index_candidates(
                (q, scales),
                weights,
                local,
                local_table,
                requests,
                lengths,
                page_size=page_size,
                topk=topk,
            )
        graph.replay()
        torch.testing.assert_close(captured_indices, indices)
        torch.testing.assert_close(captured_scores, scores)
        lengths[0] = 19
        graph.replay()
        updated_indices, updated_scores = triton_dsv4_index_candidates(
            (q, scales),
            weights,
            local,
            local_table,
            requests,
            lengths,
            page_size=page_size,
            topk=topk,
        )
        torch.testing.assert_close(captured_indices, updated_indices)
        torch.testing.assert_close(captured_scores, updated_scores)
        lengths[0] = 0
    indices, scores = torch.cat(candidates, 1), torch.cat(values, 1)
    selected = indices.gather(
        1, torch.argsort(scores, descending=True, stable=True)[:, :topk]
    )
    for row in (1, 2):
        count = min(topk, int(lengths[row]))
        assert set(selected[row, :count].tolist()) == set(
            torch.argsort(reference[row], descending=True, stable=True)[:count].tolist()
        )
