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

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import (
    deep_gemm,
    dsa_decode,
    dsa_index_candidates,
    dsa_prefill,
)
from tokenspeed_kernel.ops.attention.dsa._triton import index_candidates
from tokenspeed_kernel.ops.attention.dsa.cuda import has_ragged_decode_topk
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
    combine_dcp_attention_partials,
    dispatch_dcp_attention_partials,
)
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    compact_owned_slots,
    virtual_slots_to_local,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()


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


@pytest.mark.skipif(
    not (platform.is_nvidia and platform.is_hopper_plus),
    reason="The flashmla DSA solution registers on Hopper+ NVIDIA GPUs only",
)
@pytest.mark.parametrize("phase,q_len", [("prefill", 1), ("decode", 1), ("decode", 3)])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_flashmla_dsa_dcp_partials(phase, q_len, degree):
    packed = phase == "decode"
    attention = dsa_decode if packed else dsa_prefill
    torch.manual_seed(17)
    latent = torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
    rope = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    query = torch.randn((3, max(4, degree), 576), device="cuda", dtype=torch.bfloat16)
    dense = torch.cat((latent, rope), dim=-1)
    sparse, reference_latent = (
        _pack_sparse_kv(latent, rope) if packed else (None, latent)
    )
    slots = torch.full((3, 512), -1, device="cuda", dtype=torch.int32)
    slots[0, :4] = torch.tensor([64, 65, 128, 193], device="cuda")
    slots[1, :1] = 70
    slots[1, 1] = 130  # Excluded by topk_lens, even when the slot is valid.
    # Third row and some ranks have no candidates.
    kwargs = dict(
        q=query,
        kv_cache=None if packed else dense,
        sparse_kv_cache=sparse,
        topk_lens=torch.tensor([4, 1, 0], device="cuda", dtype=torch.int32),
        max_seqlen_k=256,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=576**-0.5,
        page_size=64,
        return_lse=True,
        solution="flashmla",
        slot_order="selection",
    )
    if packed:
        kwargs["q_len_per_req"] = q_len
    reference, ref_lse = attention(topk_slots=slots, **kwargs)
    slots[1, 1] = -1
    reference_kv = torch.cat((reference_latent.float(), rope.float()), dim=-1)
    scores = (
        torch.einsum(
            "thd,tkd->thk", query.float(), reference_kv[slots.clamp_min(0).long()]
        )
        * 576**-0.5
    )
    expected_lse = torch.logsumexp(
        scores.masked_fill((slots < 0)[:, None, :], -float("inf")), dim=-1
    )
    torch.testing.assert_close(ref_lse, expected_lse, atol=2e-3, rtol=2e-3)
    probs = torch.softmax(
        scores.masked_fill((slots < 0)[:, None, :], -float("inf")), dim=-1
    ).nan_to_num()
    expected = torch.einsum(
        "thk,tkd->thd", probs, reference_latent.float()[slots.clamp_min(0).long()]
    )
    torch.testing.assert_close(reference.float(), expected, atol=0.015, rtol=0.015)
    tensor_kwargs = dict(kwargs, return_lse=False)
    plain = attention(topk_slots=slots, **tensor_kwargs)
    torch.testing.assert_close(plain, reference)
    supplied_out = torch.empty_like(reference)
    returned_out, _ = attention(topk_slots=slots, out=supplied_out, **kwargs)
    assert returned_out is supplied_out
    torch.testing.assert_close(returned_out, reference)

    outputs, lses = [], []
    for rank in range(degree):
        owned = (slots >= 64) & ((slots // 64 - 1) % degree == rank)
        out, lse = attention(topk_slots=torch.where(owned, slots, -1), **kwargs)
        outputs.append(out.float())
        lses.append(lse)
    lses = torch.stack(lses)
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.where(torch.isfinite(lses), (lses - merged_lse).exp(), 0.0)
    merged = (torch.stack(outputs) * weights[..., None]).sum(0)
    sent = [
        dispatch_dcp_attention_partials(output.to(query.dtype), lse, degree=degree)
        for output, lse in zip(outputs, lses)
    ]
    local_heads = query.shape[1] // degree
    for owner in range(degree):
        received = torch.stack([payload[owner] for payload in sent])
        actual = combine_dcp_attention_partials(
            received,
            head_dim=512,
            partial_dtype=query.dtype,
            output_dtype=torch.float32,
            sink=None,
        )
        torch.testing.assert_close(
            actual,
            merged[:, owner * local_heads : (owner + 1) * local_heads],
            rtol=3e-6,
            atol=2e-6,
        )
    torch.testing.assert_close(merged, reference.float(), atol=0.015, rtol=0.015)
    torch.testing.assert_close(merged_lse, ref_lse, atol=2e-3, rtol=2e-3)
    assert not reference[2].any()
    assert torch.isneginf(ref_lse[2]).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = attention(topk_slots=slots, **kwargs)
    slots.fill_(-1)
    graph.replay()
    assert not graph_out.any()
    assert torch.isneginf(graph_lse).all()


# Kimi-K3.1 DCP geometry (96 gathered query heads); verify windows are 8 rows.
K31_BLOCK = 128
K31_DEGREE = 8
K31_VIRTUAL_COUNT = 1 + 12 * K31_DEGREE
K31_HEADS = 96
K31_TOPK = 2048


def _k31_selection(tokens, seed):
    # Distinct selected virtual slots with stale entries past each length.
    generator = torch.Generator().manual_seed(seed)
    capacity = (K31_VIRTUAL_COUNT - 1) * K31_BLOCK
    slots = torch.stack(
        [
            torch.randperm(capacity, generator=generator)[:K31_TOPK] + K31_BLOCK
            for _ in range(tokens)
        ]
    )
    lengths = torch.randint(0, K31_TOPK + 1, (tokens,), generator=generator)
    lengths[:3] = torch.tensor([K31_TOPK, 0, 1])
    # One row selects only rank 0's blocks, so the other ranks are empty there.
    rank0 = torch.arange(K31_BLOCK, K31_VIRTUAL_COUNT * K31_BLOCK)
    rank0 = rank0[(rank0 // K31_BLOCK - 1) % K31_DEGREE == 0]
    order = torch.randperm(rank0.numel(), generator=generator)
    slots[3, : rank0.numel()] = rank0[order]
    lengths[3] = rank0.numel()
    # -1 holes inside a live prefix.
    slots[4, ::7] = -1
    return (
        slots.to(device="cuda", dtype=torch.int32),
        lengths.to(device="cuda", dtype=torch.int32),
    )


def _k31_local_caches(kv):
    # Every rank's cache holds its owned virtual blocks at their local slots.
    slots = torch.arange(kv.shape[0], device="cuda")
    local_rows = ((K31_VIRTUAL_COUNT - 2) // K31_DEGREE + 2) * K31_BLOCK
    caches = []
    for rank in range(K31_DEGREE):
        local, owned = virtual_slots_to_local(
            slots,
            rows_per_page=K31_BLOCK,
            virtual_block_count=K31_VIRTUAL_COUNT,
            degree=K31_DEGREE,
            rank=rank,
        )
        cache = torch.zeros((local_rows, kv.shape[1]), device="cuda", dtype=kv.dtype)
        cache[local[owned]] = kv[owned]
        caches.append(cache)
    return caches


def _merge_lse_partials(partials):
    lses = torch.stack([lse for _, lse in partials])
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.where(torch.isfinite(lses), (lses - merged_lse).exp(), 0.0)
    outputs = torch.stack([out.float() for out, _ in partials])
    return (outputs * weights[..., None]).sum(0), merged_lse


@pytest.mark.parametrize("solution", ["triton", "flashmla"])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_dsa_dcp_compacted_partials_match_masked(
    phase, kv_dtype, solution, monkeypatch
):
    topk_lengths = None
    if solution == "flashmla":
        # FlashMLA sparse prefill is the BF16-KV core of P under KVP.
        if phase != "prefill" or kv_dtype != torch.bfloat16:
            pytest.skip("flashmla_dsa_prefill serves BF16-KV prefill")
        if not (platform.is_nvidia and platform.is_hopper_plus):
            pytest.skip("FlashMLA sparse prefill requires an NVIDIA Hopper+ GPU")
        from tokenspeed_kernel.ops.attention.dsa import cuda as dsa_cuda

        sparse_fwd = dsa_cuda.flash_mla_sparse_fwd
        topk_lengths = []

        def recorded(*args, **kwargs):
            topk_lengths.append(kwargs["topk_length"])
            return sparse_fwd(*args, **kwargs)

        monkeypatch.setattr(dsa_cuda, "flash_mla_sparse_fwd", recorded)
    # Two requests of eight verify rows, or sixteen prefill rows.
    tokens, q_len = 16, 8 if phase == "decode" else 1
    torch.manual_seed(31)
    # FP8 caches hold exactly the BF16 values they are built from.
    values = torch.randn(
        (K31_VIRTUAL_COUNT * K31_BLOCK, 576), device="cuda", dtype=torch.bfloat16
    )
    values = values.to(kv_dtype).to(torch.bfloat16)
    full_kv = values.to(kv_dtype)
    caches = [cache.to(kv_dtype) for cache in _k31_local_caches(values)]
    # Backends hand FP8-KV cores an FP8 query.
    q = torch.randn((tokens, K31_HEADS, 576), device="cuda").to(kv_dtype)
    slots, lengths = _k31_selection(tokens, seed=0)
    geometry = dict(
        rows_per_page=K31_BLOCK,
        virtual_block_count=K31_VIRTUAL_COUNT,
        degree=K31_DEGREE,
    )

    def attend(kv_cache, topk_slots, topk_lens):
        kwargs = dict(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=K31_VIRTUAL_COUNT * K31_BLOCK,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=576**-0.5,
            page_size=64,
            return_lse=True,
            solution=solution,
            slot_order="selection",
        )
        if phase == "decode":
            return dsa_decode(q_len_per_req=q_len, **kwargs)
        return dsa_prefill(**kwargs)

    def sharded():
        partials = []
        for rank in range(K31_DEGREE):
            local, counts = compact_owned_slots(slots, lengths, rank=rank, **geometry)
            partials.append(attend(caches[rank], local, counts))
        return partials

    partials = sharded()
    if topk_lengths is not None:
        # FlashMLA stops each compacted row after the rank's owned count.
        assert len(topk_lengths) == K31_DEGREE
        for rank, topk_length in enumerate(topk_lengths):
            _, counts = compact_owned_slots(slots, lengths, rank=rank, **geometry)
            assert topk_length.dtype == torch.int32 and topk_length.is_contiguous()
            assert torch.equal(topk_length, counts)
        topk_lengths.clear()
    owned_total = torch.zeros_like(lengths)
    for rank, (out, lse) in enumerate(partials):
        # The previous layout: non-owned entries masked in place, global lengths.
        local, owned = virtual_slots_to_local(slots, rank=rank, **geometry)
        masked_out, masked_lse = attend(
            caches[rank], torch.where(owned, local, -1), lengths
        )
        # Moved softmax tiles round differently (max 3.9e-3); a dropped slot errs 0.1-1.
        torch.testing.assert_close(
            out.float(), masked_out.float(), atol=0.015, rtol=0.015
        )
        torch.testing.assert_close(lse, masked_lse, atol=1e-3, rtol=1e-3)
        _, counts = compact_owned_slots(slots, lengths, rank=rank, **geometry)
        owned_total += counts
    live = torch.arange(K31_TOPK, device="cuda") < lengths[:, None]
    assert torch.equal(owned_total, (live & (slots >= 0)).sum(1).to(torch.int32))
    # Row 3 lives on rank 0 only: every other rank's partial is empty.
    for out, lse in partials[1:]:
        assert not out[3].any() and torch.isneginf(lse[3]).all()

    expected_out, expected_lse = attend(full_kv, slots, lengths)
    merged_out, merged_lse = _merge_lse_partials(partials)
    # Merged BF16 partials against full attention: the file's merge bound (above).
    torch.testing.assert_close(merged_out, expected_out.float(), atol=0.015, rtol=0.015)
    torch.testing.assert_close(merged_lse, expected_lse, atol=2e-3, rtol=2e-3)
    assert torch.isneginf(merged_lse[1]).all() and not merged_out[1].any()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = _merge_lse_partials(sharded())
    for seed in (1, 2):
        new_slots, new_lengths = _k31_selection(tokens, seed=seed)
        slots.copy_(new_slots)
        lengths.copy_(new_lengths)
        graph.replay()
        expected_out, expected_lse = attend(full_kv, slots, lengths)
        torch.testing.assert_close(
            graph_out, expected_out.float(), atol=0.015, rtol=0.015
        )
        torch.testing.assert_close(graph_lse, expected_lse, atol=2e-3, rtol=2e-3)


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or not platform.is_hopper,
    reason="requires an NVIDIA Hopper GPU",
)

requires_ragged_topk = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or not platform.is_hopper_plus
    or not has_ragged_decode_topk(),
    reason="requires the CUDA persistent_topk ragged selector",
)

PAGE_SIZE = 64
INITIAL_TOKENS = 4
LOCAL_TOKENS = 8


@pytest.fixture
def deep_select():
    # Optional selector dependencies must not skip the FlashMLA tests.
    return pytest.importorskip(
        "tokenspeed_kernel.ops.attention.dsa._cute_dsl.deep_select"
    )


def _scores(rows, width, *, ties):
    # Keep a padded, non-contiguous row view, as DeepGEMM can return one.
    stride = (width + 3) // 4 * 4 + 4
    storage = torch.full((rows, stride), torch.nan, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(421)
    values = torch.randperm(width, generator=generator, device="cuda").float()
    if ties:
        values = values.remainder(7)
    scores = storage[:, :width]
    scores.copy_(values.expand(rows, -1))
    scores[:, 7] = torch.nan
    scores[:, 8] = torch.inf
    scores[:, 9] = -torch.inf
    return scores


def _metadata(width, lengths):
    rows = len(lengths)
    # Missing foreign pages leave gaps in global logical positions.
    positions = (
        torch.arange((width + PAGE_SIZE - 1) // PAGE_SIZE, device="cuda") * 2
    ).to(torch.int32)
    positions = positions.repeat(rows, 1)
    causal = []
    for length in lengths:
        last = max(0, length - 1)
        causal.append(
            (last // PAGE_SIZE * 2) * PAGE_SIZE + last % PAGE_SIZE + 1 if length else 0
        )
    return (
        positions,
        torch.tensor(lengths, dtype=torch.int32, device="cuda").reshape(-1, 1),
        torch.tensor(causal, dtype=torch.int32, device="cuda"),
    )


def _select_candidates(scores, positions, lengths, causal, topk):
    return deep_gemm._select_dcp_index_candidates(
        scores,
        positions,
        lengths,
        causal,
        page_size=PAGE_SIZE,
        topk=topk,
        initial_tokens=INITIAL_TOKENS,
        local_tokens=LOCAL_TOKENS,
    )


def _check_candidates(raw, positions, lengths, causal, topk, logical, values):
    # Build the oracle from the unmasked input and global positions, rather
    # than reusing either the production mask or its selected offsets.
    raw, positions = raw.cpu(), positions.cpu()
    logical, values = logical.cpu(), values.cpu()
    for row, length in enumerate(lengths.flatten().cpu().tolist()):
        columns = torch.arange(length)
        global_positions = (
            positions[row, columns // PAGE_SIZE] * PAGE_SIZE + columns % PAGE_SIZE
        )
        expected = raw[row, :length].clone()
        expected[~torch.isfinite(expected)] = -torch.inf
        forced = (global_positions < INITIAL_TOKENS) | (
            global_positions >= int(causal[row]) - LOCAL_TOKENS
        )
        expected[forced] = torch.inf
        expected_count = min(topk, int((expected > -torch.inf).sum()))
        valid = logical[row] >= 0
        assert int(valid.sum()) == expected_count
        assert (logical[row, ~valid] == -1).all()
        assert torch.isneginf(values[row, ~valid]).all()
        chosen_positions = logical[row, valid].tolist()
        assert len(set(chosen_positions)) == expected_count
        expected_by_position = dict(zip(global_positions.tolist(), expected.tolist()))
        assert set(chosen_positions).issubset(expected_by_position)
        assert set(global_positions[forced].tolist()).issubset(chosen_positions)
        chosen_values = values[row, valid]
        torch.testing.assert_close(
            chosen_values,
            torch.tensor([expected_by_position[p] for p in chosen_positions]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            chosen_values.sort(descending=True).values,
            torch.topk(expected, expected_count).values,
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "width,topk",
    [(128, 512), (512, 512), (8193, 512), (8193, 1024), (65536, 2048)],
)
@pytest.mark.parametrize("ties", [False, True])
@requires_hopper
def test_dcp_mask_select_gather(width, topk, ties, deep_select):
    lengths = [min(width, n) for n in (0, 1, topk - 1, topk, topk + 1, width)]
    scores = _scores(len(lengths), width, ties=ties)
    raw = scores.clone()
    positions, lengths, causal = _metadata(width, lengths)
    logical, values = _select_candidates(scores, positions, lengths, causal, topk)
    _check_candidates(raw, positions, lengths, causal, topk, logical, values)


@requires_hopper
def test_dcp_selector_graph_refresh_and_precompiled_clusters(deep_select):
    # An eager short-row call must also prepare the larger cluster variants
    # that a subsequent batch can choose, without specializing on row geometry.
    small = torch.ones((1, 128), device="cuda")
    small_positions, small_lengths, small_causal = _metadata(128, [128])
    _select_candidates(small, small_positions, small_lengths, small_causal, 512)
    compiled = deep_select.compiled_selector.cache_info().misses

    width, topk = 65536, 512
    raw = _scores(3, width, ties=False)
    scores = torch.empty_strided(raw.shape, raw.stride(), device="cuda")
    positions, lengths, causal = _metadata(width, [width, topk + 1, 0])

    def run():
        scores.copy_(raw)
        return _select_candidates(scores, positions, lengths, causal, topk)

    run()
    assert deep_select.compiled_selector.cache_info().misses == compiled
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logical, values = run()
    for new_lengths in ([width, topk + 1, 0], [0, 0, 0], [1, 87, topk], [width, 0, 1]):
        _, updated_lengths, updated_causal = _metadata(width, new_lengths)
        lengths.copy_(updated_lengths)
        causal.copy_(updated_causal)
        graph.replay()
        _check_candidates(raw, positions, lengths, causal, topk, logical, values)
    assert deep_select.compiled_selector.cache_info().misses == compiled


@pytest.mark.parametrize(
    "unsupported",
    ["topk", "row_stride", "base_alignment", "column_stride", "storage", "dependency"],
)
@requires_hopper
def test_dcp_selector_retains_torch_fallback(unsupported, monkeypatch, deep_select):
    topk = 512
    if unsupported == "topk":
        scores = torch.randn((2, 128), device="cuda")
        topk = 4096
    elif unsupported == "row_stride":
        scores = torch.randn((2, 514), device="cuda")[:, :513]
    elif unsupported == "base_alignment":
        scores = torch.randn((2, 520), device="cuda")[:, 1:514]
    elif unsupported == "column_stride":
        scores = torch.randn((2, 1024), device="cuda")[:, ::2]
    elif unsupported == "storage":
        # The logical row fits, but the final aligned vector load would not.
        scores = torch.randn(517, device="cuda").as_strided((1, 513), (516, 1), 4)
    else:
        scores = torch.randn((2, 1024), device="cuda")
        monkeypatch.setattr(deep_gemm, "_dcp_deep_select", None)
        # Without DeepSelect, the ragged CUDA selector is next in line.
        monkeypatch.setattr(deep_gemm, "has_ragged_decode_topk", lambda: False)
    lengths = torch.full(
        (scores.shape[0], 1), scores.shape[1], dtype=torch.int32, device="cuda"
    )

    def unexpected_warmup(*args):
        pytest.fail("unsupported inputs must not compile or launch DeepSelect")

    monkeypatch.setattr(deep_gemm, "_warmup_dcp_topk", unexpected_warmup)
    assert not deep_gemm._dcp_topk_supports_lengths(scores, topk)
    raw = scores.clone()
    positions, lengths, causal = _metadata(scores.shape[1], lengths.flatten().tolist())
    logical, values = _select_candidates(scores, positions, lengths, causal, topk)
    _check_candidates(raw, positions, lengths, causal, topk, logical, values)


@requires_hopper
def test_deep_gemm_facade_selects_with_original_empty_lengths(monkeypatch, deep_select):
    table = torch.tensor(
        [[3, -1, 1, 2], [-1, -1, -1, -1]], dtype=torch.int32, device="cuda"
    )
    requests = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda")
    causal = torch.tensor([193, 80, 0], dtype=torch.int32, device="cuda")
    q = torch.ones((3, 16, 128), dtype=torch.bfloat16, device="cuda")
    weights = torch.ones((3, 16), device="cuda")
    cache = torch.zeros((256, 132), dtype=torch.uint8, device="cuda")
    select = deep_select.deepselect_topk
    calls = []

    def checked_select(scores, ends, *args, **kwargs):
        assert ends.tolist() == [129, 0, 0]
        calls.append(True)
        return select(scores, ends, *args, **kwargs)

    monkeypatch.setattr(deep_select, "deepselect_topk", checked_select)
    logical, values = dsa_index_candidates(
        q,
        weights,
        cache,
        table,
        requests,
        causal,
        page_size=PAGE_SIZE,
        topk=512,
        softmax_scale=0.1,
        initial_tokens=INITIAL_TOKENS,
        local_tokens=LOCAL_TOKENS,
        solution="deep_gemm",
    )
    assert calls == [True]
    assert set(logical[0, logical[0] >= 0].tolist()) == set(range(64)) | set(
        range(128, 193)
    )
    assert (logical[1:] == -1).all()
    assert torch.isneginf(values[1:]).all()


def _record_ragged_calls(monkeypatch):
    # Disable DeepSelect so Hopper also exercises the ragged selector.
    monkeypatch.setattr(deep_gemm, "_dcp_deep_select", None)
    select = deep_gemm.ragged_decode_topk
    calls = []

    def recorded(logits, *args, **kwargs):
        calls.append(logits)
        return select(logits, *args, **kwargs)

    monkeypatch.setattr(deep_gemm, "ragged_decode_topk", recorded)
    return calls


def _padded_scores(rows, width):
    # DeepGEMM pads logits rows to a multiple of 256 and returns the slice.
    stride = (width + 255) // 256 * 256
    return torch.full((rows, stride), torch.nan, device="cuda")[:, :width]


@pytest.mark.parametrize("width", [2048, 8193, 65536, 1048704])
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("rows", [6, 40])
@requires_ragged_topk
def test_dcp_ragged_select_gather(width, topk, ties, rows, monkeypatch):
    calls = _record_ragged_calls(monkeypatch)
    # Long rows use cluster launches up to 32 rows; 40 rows reach the per-row kernel.
    pattern = [min(width, n) for n in (0, 1, topk - 1, topk, topk + 1, width)]
    lengths = [pattern[row % len(pattern)] for row in range(rows)]
    scores = _scores(len(lengths), width, ties=ties)
    raw = scores.clone()
    positions, lengths, causal = _metadata(width, lengths)
    logical, values = _select_candidates(scores, positions, lengths, causal, topk)
    # The selector reads the padded rows in place instead of copying them.
    assert len(calls) == 1
    assert calls[0].is_contiguous()
    assert calls[0].data_ptr() == scores.data_ptr()
    assert calls[0].stride(0) == scores.stride(0)
    _check_candidates(raw, positions, lengths, causal, topk, logical, values)


@pytest.mark.parametrize("rows", [6, 1024])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_dcp_length_aware_mask_leaves_tail_untouched(rows):
    # Many rows narrow the grid, so each program strides over several blocks of its row.
    width = 20000
    generator = torch.Generator(device="cuda").manual_seed(5)
    row_lengths = torch.randint(
        0, width + 1, (rows,), generator=generator, device="cuda"
    ).tolist()
    row_lengths[:6] = [0, 1, 255, 256, 257, width]
    positions, lengths, causal = _metadata(width, row_lengths)
    scores = _scores(rows, width, ties=False)
    prefix = torch.arange(width, device="cuda") < lengths
    scores.masked_fill_(~prefix, torch.nan)
    expected = scores.clone()
    args = (positions, lengths, causal, PAGE_SIZE, INITIAL_TOKENS, LOCAL_TOKENS)
    index_candidates.mask_index_scores(expected, *args, mask_tail=True)
    index_candidates.mask_index_scores(scores, *args, mask_tail=False)
    torch.testing.assert_close(scores[prefix], expected[prefix], atol=0, rtol=0)
    assert torch.isnan(scores[~prefix]).all()
    # Allocation padding past the logical width is never written either.
    padded = scores.as_strided((rows, scores.stride(0)), scores.stride())
    assert torch.isnan(padded[:, width:]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_dcp_mask_and_gather_rows_past_int32_offsets(monkeypatch):
    # Strides fit int32 but the last row starts past 2**31, as in a 1M-wide table.
    rows, width, topk = 3, 4096, 512
    stride = (1 << 30) + 256
    elements = (rows - 1) * stride + width
    if torch.cuda.mem_get_info()[0] < elements * 4 + (1 << 30):
        pytest.skip("needs 9 GiB of free device memory")
    raw = _scores(rows, width, ties=False).contiguous()
    positions, lengths, causal = _metadata(width, [width, 1000, width - 7])
    args = (positions, lengths, causal, PAGE_SIZE, INITIAL_TOKENS, LOCAL_TOKENS)
    scores = torch.empty(elements, device="cuda").as_strided((rows, width), (stride, 1))
    for mask_tail in (False, True):
        scores.copy_(raw)
        expected = raw.clone()
        index_candidates.mask_index_scores(scores, *args, mask_tail=mask_tail)
        index_candidates.mask_index_scores(expected, *args, mask_tail=mask_tail)
        torch.testing.assert_close(scores, expected, atol=0, rtol=0, equal_nan=True)
    # This layout falls back to torch.topk, whose gather also reads past 2**31.
    monkeypatch.setattr(deep_gemm, "_dcp_deep_select", None)
    assert deep_gemm._dcp_ragged_rows(scores, topk) is None
    scores.copy_(raw)
    logical, values = _select_candidates(scores, positions, lengths, causal, topk)
    _check_candidates(raw, positions, lengths, causal, topk, logical, values)


@pytest.mark.parametrize("width,topk", [(65536, 512), (1048704, 2048)])
@requires_ragged_topk
def test_dcp_ragged_selector_graph_replay(width, topk, monkeypatch):
    calls = _record_ragged_calls(monkeypatch)
    raw = _scores(3, width, ties=False)
    scores = _padded_scores(3, width)
    positions, lengths, causal = _metadata(width, [width, topk + 1, 0])

    def run():
        scores.copy_(raw)
        return _select_candidates(scores, positions, lengths, causal, topk)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logical, values = run()
    # Eager warmup and capture each launch once; replays reuse the capture.
    assert len(calls) == 2
    for new_lengths in ([width, topk + 1, 0], [0, 0, 0], [1, 87, topk], [width, 0, 1]):
        _, updated_lengths, updated_causal = _metadata(width, new_lengths)
        lengths.copy_(updated_lengths)
        causal.copy_(updated_causal)
        graph.replay()
        _check_candidates(raw, positions, lengths, causal, topk, logical, values)


@pytest.mark.parametrize(
    "unsupported", ["topk", "narrow", "column_stride", "base_alignment"]
)
@requires_ragged_topk
def test_dcp_ragged_selector_fallbacks(unsupported, monkeypatch):
    monkeypatch.setattr(deep_gemm, "_dcp_deep_select", None)
    topk = 512
    if unsupported == "topk":
        scores = _padded_scores(2, 8192).normal_()
        topk = 4096
    elif unsupported == "narrow":
        scores = _padded_scores(2, topk - 64).normal_()
    elif unsupported == "column_stride":
        scores = torch.randn((2, 2048), device="cuda")[:, ::2]
    else:
        scores = torch.randn(2 * 1024 + 1, device="cuda")[1:].view(2, 1024)

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported inputs must not launch the ragged top-k")

    monkeypatch.setattr(deep_gemm, "ragged_decode_topk", unexpected)
    assert deep_gemm._dcp_ragged_rows(scores, topk) is None
    raw = scores.clone()
    positions, lengths, causal = _metadata(
        scores.shape[1], [scores.shape[1]] * scores.shape[0]
    )
    logical, values = _select_candidates(scores, positions, lengths, causal, topk)
    _check_candidates(raw, positions, lengths, causal, topk, logical, values)


@pytest.mark.skipif(
    not (platform.is_nvidia and platform.is_hopper_plus),
    reason="FlashMLA query padding requires Hopper+ NVIDIA GPUs",
)
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_flashmla_query_padding_lifetime(phase):
    from tokenspeed_kernel.ops.attention.dsa.cuda import (
        _pad_decode_query,
        _pad_prefill_query,
    )

    def pad(q):
        return (
            _pad_prefill_query(q)[0]
            if phase == "prefill"
            else _pad_decode_query(q, 1)[0].flatten(0, 1)
        )

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for length in range(31, 63):
        q = torch.full((length, 4, 576), 1.25, device="cuda", dtype=torch.bfloat16)
        padded = pad(q)
        torch.testing.assert_close(padded[:, :4], q, atol=0, rtol=0)
        assert not padded[:, 4:].any()
        del q, padded
    torch.cuda.synchronize()
    # Check live allocations, not reusable allocator reservations.
    assert torch.cuda.memory_allocated() <= baseline + 1024**2

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    def enqueue(index):
        with torch.cuda.stream(streams[index]):
            q = torch.full((7, 4, 576), index + 1, device="cuda", dtype=torch.bfloat16)
            return pad(q)

    # Hold both results alive: calls with identical shapes must not alias.
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(enqueue, range(2)))
    for stream in streams:
        stream.synchronize()
    for index, output in enumerate(outputs):
        assert (output[:, :4] == index + 1).all()
        assert not output[:, 4:].any()


@pytest.mark.skipif(
    not (platform.is_nvidia and platform.is_hopper_plus),
    reason="FlashMLA query padding requires Hopper+ NVIDIA GPUs",
)
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_flashmla_query_padding_independent_graphs(phase):
    from tokenspeed_kernel.ops.attention.dsa.cuda import (
        _pad_decode_query,
        _pad_prefill_query,
    )

    def pad(q):
        return (
            _pad_prefill_query(q)[0]
            if phase == "prefill"
            else _pad_decode_query(q, 3)[0].flatten(0, 1)
        )

    inputs = [
        torch.full((6, 4, 576), i + 1, device="cuda", dtype=torch.bfloat16)
        for i in range(2)
    ]
    # Warmup must not cause captures to share eager storage.
    for q in inputs:
        pad(q)
    torch.cuda.synchronize()
    graphs, outputs = [], []
    for q in inputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = pad(q)
        graphs.append(graph)
        outputs.append(output)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for iteration in range(3):
        for i, q in enumerate(inputs):
            q.fill_(10 * iteration + i + 1)
        torch.cuda.synchronize()
        for graph, stream in zip(graphs, streams, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
        # Exercise the same eager shape while both graphs remain live.
        eager = pad(torch.full_like(inputs[0], -1))
        torch.cuda.synchronize()
        for q, output in zip(inputs, outputs, strict=True):
            torch.testing.assert_close(output[:, :4], q, atol=0, rtol=0)
            assert not output[:, 4:].any()
        assert (eager[:, :4] == -1).all()


# --- owned-width index pages: bounded compaction -----------------------------


def _compact_inputs(cols, *, rows, seed):
    """A local index table of three requests (foreign pages -1, null pages 0,
    stale pages past every length) and query rows over it: a row of no
    request, one past the table's requests, empty and one-token rows, a
    length past the table, and last the longest row of request 0."""
    generator = torch.Generator().manual_seed(seed)
    table = torch.randint(1, 1 << 20, (3, cols), generator=generator)
    absent = torch.rand((3, cols), generator=generator)
    table[absent < 0.5] = -1
    table[(absent >= 0.5) & (absent < 0.55)] = 0
    # The longest row (last) always has a page to keep.
    table[0, 0] = 5
    width = cols * PAGE_SIZE
    requests = torch.randint(0, 3, (rows,), generator=generator)
    causal = torch.randint(0, width + 1, (rows,), generator=generator)
    special = [(-1, width), (3, width), (0, 0), (1, 1), (2, PAGE_SIZE + 1)]
    special.append((1, width + 1000))
    for row, (request, length) in enumerate(special[: rows - 1]):
        requests[row], causal[row] = request, length
    requests[-1], causal[-1] = 0, width
    return (
        table.to(device="cuda", dtype=torch.int32),
        requests.to(device="cuda", dtype=torch.int32),
        causal.to(device="cuda", dtype=torch.int32),
    )


def _bounded_reference(full, causal, columns):
    """What compacting to ``columns`` must give, from the full-width
    compaction: every row's first ``columns`` pages and positions, the tokens
    of only those pages, and the count of pages left out."""
    pages, positions, _ = (tensor.cpu() for tensor in full)
    causal = causal.cpu()
    rows = pages.shape[0]
    found = (pages > 0).sum(1)
    kept = found.clamp(max=columns)
    expected_pages = torch.zeros((rows, columns), dtype=torch.int32)
    expected_positions = torch.zeros_like(expected_pages)
    expected_lengths = torch.zeros((rows, 1), dtype=torch.int32)
    for row in range(rows):
        count = int(kept[row])
        expected_pages[row, :count] = pages[row, :count]
        expected_positions[row, :count] = positions[row, :count]
        expected_lengths[row] = sum(
            min(PAGE_SIZE, int(causal[row]) - col * PAGE_SIZE)
            for col in positions[row, :count].tolist()
        )
    return (
        expected_pages,
        expected_positions,
        expected_lengths,
        (found - kept).to(torch.int32),
    )


def _check_bounded_compact(table, requests, causal, columns):
    full = index_candidates.compact_index_pages(table, requests, causal, PAGE_SIZE)
    overflow = torch.full((requests.numel(),), -7, dtype=torch.int32, device="cuda")
    bounded = index_candidates.compact_index_pages(
        table, requests, causal, PAGE_SIZE, columns=columns, overflow=overflow
    )
    expected = _bounded_reference(full, causal, columns)
    for actual, want in zip((*bounded, overflow), expected, strict=True):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
    return full, expected


@pytest.mark.parametrize("cols", [1, 7, 16, 1023, 1024, 1025, 3381, 16384])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_bounded_compact_matches_full_width_compact(cols):
    """Compacting to ``columns`` keeps the full-width compaction's first
    ``columns`` pages of every row, counts only their tokens, and reports
    the rest as overflow; an output at least the widest row's is the
    full-width compaction, zero-padded."""
    table, requests, causal = _compact_inputs(cols, rows=12, seed=cols)
    full = index_candidates.compact_index_pages(table, requests, causal, PAGE_SIZE)
    widest = int((full[0] > 0).sum(1).max())
    last = int((full[0][-1] > 0).sum())
    assert last > 0
    for columns in sorted({1, 16, max(1, widest - 1), max(1, widest), widest + 5}):
        _, (_, _, lengths, overflow) = _check_bounded_compact(
            table, requests, causal, columns
        )
        if columns >= widest:
            assert not overflow.any()
            torch.testing.assert_close(lengths, full[2].cpu(), rtol=0, atol=0)

    # The longest row overflows a narrower output; stores must stop before the guard.
    columns = max(1, last - 1)
    rows = requests.numel()
    guard = 4096
    storage = torch.full(
        (2, rows * columns + guard), -7, dtype=torch.int32, device="cuda"
    )
    storage[:, : rows * columns] = 0
    pages = storage[0, : rows * columns].view(rows, columns)
    positions = storage[1, : rows * columns].view(rows, columns)
    lengths = torch.empty((rows, 1), dtype=torch.int32, device="cuda")
    index_candidates._launch_compact_bounded(
        table, requests, causal, PAGE_SIZE, pages, positions, lengths, None
    )
    assert (storage[:, rows * columns :] == -7).all()
    # No overflow output on this launch, so compare the first three planes only.
    expected = _bounded_reference(full, causal, columns)[:3]
    for actual, want in zip((pages, positions, lengths), expected, strict=True):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_bounded_compact_compiles_once_across_widths():
    """Table width, output width, row and request counts are runtime and
    unspecialized: one launch warms every batch shape."""
    from utils import assert_no_triton_compile

    def run(cols, columns, rows):
        table, requests, causal = _compact_inputs(cols, rows=rows, seed=cols + rows)
        _check_bounded_compact(table, requests, causal, columns)

    run(37, 21, 5)
    with assert_no_triton_compile(index_candidates._compact_bounded):
        for cols, columns, rows in (
            (1, 1, 1),
            (16, 16, 16),
            (1024, 32, 3),
            (1025, 1024, 17),
            (3381, 432, 64),
            (16384, 2368, 2),
        ):
            run(cols, columns, rows)


def _owned_width_index_plane(pages, generator):
    """FP8 index keys for every local page, in the block-split plane."""
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    slots = pages * PAGE_SIZE
    keys = torch.randn((slots, 128), generator=generator).to(
        device="cuda", dtype=torch.bfloat16
    )
    values, scales = quantize_fp8(
        keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    plane = torch.zeros((slots, 132), dtype=torch.uint8, device="cuda")
    index_k_block_split_scatter(
        plane,
        values,
        scales,
        torch.arange(slots, device="cuda"),
        page_size=PAGE_SIZE,
        head_dim=128,
        group_size=128,
        write_mask=None,
    )
    return plane


def _assert_same_candidates(actual, expected, topk):
    """Equal selections up to ties at the smallest selected score: the same
    count, the same score bits as a multiset, the same positions above that
    score (all of them when a row selects fewer than ``topk``), and the same
    score bits for a position either selects."""
    for row in range(actual[0].shape[0]):
        picks = []
        for logical, values in (actual, expected):
            live = logical[row] >= 0
            positions = logical[row, live].tolist()
            bits = values[row, live].view(torch.int32).tolist()
            scores = values[row, live].tolist()
            picks.append((dict(zip(positions, bits)), dict(zip(positions, scores))))
        (got_bits, got), (want_bits, want) = picks
        assert len(got) == len(want)
        assert sorted(got_bits.values()) == sorted(want_bits.values())
        if len(got) < topk:
            assert got.keys() == want.keys()
        elif got:
            cut = min(want.values())
            assert {p for p, s in got.items() if s > cut} == {
                p for p, s in want.items() if s > cut
            }
        for position in got.keys() & want.keys():
            assert got_bits[position] == want_bits[position]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and platform.is_nvidia and platform.is_hopper_plus),
    reason="the DeepGEMM index-candidate scorer requires NVIDIA Hopper or newer",
)
def test_owned_width_candidates_match_full_width():
    """Pages planned at the widest row's owned count select what full-width
    pages select, tile by tile; a width below it stays inside each row's
    stored pages and reports the rest as overflow."""
    from tokenspeed_kernel.ops.attention.dsa import plan_dsa_index_candidates

    heads, topk, cols = 32, 2048, 160
    generator = torch.Generator().manual_seed(53)
    # Skewed ownership: request 0 owns its first half and more, request 1 about a third.
    owned = torch.rand((2, cols), generator=generator) < 0.35
    owned[0, : cols // 2] = True
    local = torch.randperm(2 * cols, generator=generator).reshape(2, cols) + 1
    table = torch.where(owned, local, -1).to(device="cuda", dtype=torch.int32)
    plane = _owned_width_index_plane(2 * cols + 1, generator)
    width = cols * PAGE_SIZE
    lengths = [1, 63, 64, 65, 2047, 2048, 2049, 5000, width // 2, width - 1, width]
    requests = torch.tensor(
        [0] * len(lengths) + [1] * len(lengths), dtype=torch.int32, device="cuda"
    )
    causal = torch.tensor(lengths * 2, dtype=torch.int32, device="cuda")
    rows = requests.numel()
    q = torch.randn((rows, heads, 128), generator=generator).to(
        device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn((rows, heads), generator=generator).to(device="cuda")

    def plan(columns=None):
        return plan_dsa_index_candidates(
            q,
            weights,
            plane,
            table,
            requests,
            causal,
            page_size=PAGE_SIZE,
            solution="deep_gemm",
            columns=columns,
        )

    def candidates(start, end, pages):
        return dsa_index_candidates(
            q[start:end],
            weights[start:end],
            plane,
            table,
            requests[start:end],
            causal[start:end],
            page_size=PAGE_SIZE,
            topk=topk,
            softmax_scale=0.1,
            initial_tokens=INITIAL_TOKENS,
            local_tokens=LOCAL_TOKENS,
            solution="deep_gemm",
            pages=pages.rows(start, end),
        )

    full = plan()
    assert full.overflow is None
    widest = int((full.pages > 0).sum(1).max())
    # The runtime's width: widest row, at least topk tokens, rounded up to 16 pages.
    columns = -(-max(widest, topk // PAGE_SIZE) // 16) * 16
    assert widest <= columns < cols
    planned = plan(columns)
    assert planned.pages.shape == planned.positions.shape == (rows, columns)
    assert not planned.overflow.any()
    for actual, want in (
        (planned.pages, full.pages[:, :columns]),
        (planned.positions, full.positions[:, :columns]),
        (planned.lengths, full.lengths),
    ):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    for start, end in ((0, rows), (0, 7), (7, rows)):
        _assert_same_candidates(
            candidates(start, end, planned), candidates(start, end, full), topk
        )

    # Too narrow: long rows overflow and select only among the pages they stored.
    narrow = plan(16)
    assert narrow.overflow.any()
    logical, _ = candidates(0, rows, narrow)
    stored_lengths = narrow.lengths.flatten().tolist()
    for row in range(rows):
        kept = int((narrow.pages[row] > 0).sum())
        stored = set(narrow.positions[row, :kept].tolist())
        chosen = logical[row][logical[row] >= 0].tolist()
        assert len(chosen) == min(topk, stored_lengths[row])
        assert {position // PAGE_SIZE for position in chosen} <= stored
