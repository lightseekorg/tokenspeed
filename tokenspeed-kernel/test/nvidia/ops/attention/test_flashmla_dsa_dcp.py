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
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
    combine_dcp_attention_partials,
    dispatch_dcp_attention_partials,
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


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or not platform.is_hopper,
    reason="requires an NVIDIA Hopper GPU",
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
