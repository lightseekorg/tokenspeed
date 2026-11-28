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

"""MegaMoE input staging: UE8M0 FP8 hidden states and widened router ids."""

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (  # noqa: E402
    _dispatch_and_record_topk_ids_kernel,
    _stage_mxfp4_mega_moe_inputs_kernel,
    stage_mxfp4_mega_moe_inputs,
    triton_dispatch_and_record_topk_ids,
)
from utils import assert_no_triton_compile  # noqa: E402


@pytest.mark.parametrize("width", [1, 13])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_captured_token_replica_dispatch_tracks_placement_and_live_rows(width, dtype):
    rows = 97
    ids = torch.tensor([0, 0, 1, 0, 2, 0], device="cuda", dtype=dtype).repeat(rows, 1)[
        :, ::2
    ]
    replicas = torch.full((3, width), -1, device="cuda", dtype=torch.int32)
    replicas[0] = torch.arange(width, device="cuda")
    replicas[1, 0], replicas[2, 0] = width, width + 1
    counts = torch.tensor([width, 1, 1], device="cuda", dtype=torch.int32)
    row = torch.arange(rows, device="cuda")
    live = row % 3 != 0
    initial = 2**33
    load = torch.full((width + 2,), initial, device="cuda", dtype=torch.int64)

    enabled = torch.tensor(False, device="cuda")
    triton_dispatch_and_record_topk_ids(ids, replicas, counts, live, load, enabled)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mapped = triton_dispatch_and_record_topk_ids(
            ids, replicas, counts, live, load, enabled
        )

    for hot_expert in (0, 2):
        if hot_expert == 2:
            replicas.copy_(replicas.flip([0]))
            counts.copy_(counts.flip([0]))
            live.logical_not_()
        load.fill_(initial)
        enabled.fill_(False)
        graph.replay()
        assert (load == initial).all()
        disabled_ids = mapped.clone()
        enabled.fill_(True)
        graph.replay()
        assert torch.equal(mapped, disabled_ids)
        hashed = (row[:, None] * 2654435769) & 0xFFFFFFFF
        logical = ids.long()
        expected = replicas[logical, hashed % counts[logical]]
        assert mapped.dtype == ids.dtype
        assert torch.equal(mapped, expected.to(dtype))
        assert mapped[:, hot_expert].unique().numel() == width
        assert torch.equal(
            load - initial,
            torch.bincount(expected[live].long().flatten(), minlength=load.numel()),
        )


def test_token_replica_dispatch_varies_row_count_without_recompile():
    ids = torch.zeros((128, 6), device="cuda", dtype=torch.int32)
    replicas = torch.tensor([[1], [0]], device="cuda", dtype=torch.int32)
    counts = torch.ones(2, device="cuda", dtype=torch.int32)
    live = torch.ones(128, device="cuda", dtype=torch.bool)
    load = torch.zeros(2, device="cuda", dtype=torch.int64)
    for rows in (1, 16, 33):
        triton_dispatch_and_record_topk_ids(
            ids[:rows], replicas, counts, live[:rows], load
        )
    load.zero_()
    with assert_no_triton_compile(_dispatch_and_record_topk_ids_kernel):
        for rows in (0, 2, 17, 32, 65):
            mapped = triton_dispatch_and_record_topk_ids(
                ids[:rows], replicas, counts, live[:rows], load
            )
            assert mapped.shape == (rows, 6) and (mapped == 1).all()
    assert load.tolist() == [0, (2 + 17 + 32 + 65) * 6]


def test_token_replica_dispatch_reads_strided_ids_past_int32_offsets():
    stride = 2**31
    storage = torch.empty(stride + 1, device="cuda", dtype=torch.int32)
    ids = storage.as_strided((2, 1), (stride, 1))
    ids[:, 0].copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    replicas = torch.tensor([[1], [0]], device="cuda", dtype=torch.int32)
    counts = torch.ones(2, device="cuda", dtype=torch.int32)
    live = torch.tensor([False, True], device="cuda")
    load = torch.zeros(2, device="cuda", dtype=torch.int64)
    mapped = triton_dispatch_and_record_topk_ids(ids, replicas, counts, live, load)
    assert mapped.tolist() == [[1], [0]]
    assert load.tolist() == [1, 0]


def _stage(hidden, weights, ids):
    tokens, size = hidden.shape
    out = (
        torch.empty(tokens, size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(tokens, size // 128, device="cuda", dtype=torch.int32),
        torch.full((tokens, ids.shape[1]), -1, device="cuda", dtype=torch.int64),
        torch.empty(tokens, ids.shape[1], device="cuda", dtype=torch.float32),
    )
    stage_mxfp4_mega_moe_inputs(hidden, weights, ids, *out, None, None, None)
    return out


@pytest.mark.parametrize("tokens", [1, 37])
def test_int32_ids_stage_like_int64_ids(tokens):
    torch.manual_seed(tokens)
    hidden = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)

    narrow = _stage(hidden, weights, ids)
    wide = _stage(hidden, weights, ids.long())
    assert torch.equal(narrow[2], ids.long())
    for got, want in zip(narrow, wide):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def _reference(hidden, weights, ids):
    """Group-32 UE8M0 FP8 codes and per-128-column packed exponents."""
    tokens, size = hidden.shape
    groups = hidden.float().view(tokens, size // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(1.0e-4)
    bits = (amax / 448.0).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    exponent = exponent.clamp(1, 254)
    scale = (exponent << 23).view(torch.float32)
    codes = (groups * (1.0 / scale)[..., None]).view(tokens, size)
    packed = (
        exponent.view(tokens, size // 128, 4).long()
        << torch.arange(0, 32, 8, device=hidden.device)
    ).sum(-1)
    # Four bytes reinterpreted as one int32 word.
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return (
        codes.to(torch.float8_e4m3fn),
        packed.to(torch.int32),
        ids.long(),
        weights,
    )


@pytest.mark.parametrize("tokens", [1, 37, 63, 64, 65, 255, 1023, 1024, 1500])
@pytest.mark.parametrize("size", [512, 5120])
def test_stage_matches_reference(tokens, size):
    torch.manual_seed(tokens * size)
    hidden = torch.randn(tokens, size, device="cuda", dtype=torch.bfloat16)
    hidden[0, :32] = 0.0
    hidden[-1, 32:64] = 1.0e-7
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)

    for got, want in zip(
        _stage(hidden, weights, ids), _reference(hidden, weights, ids)
    ):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_stage_token_count_does_not_recompile():
    hidden = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(2048, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (2048, 6), device="cuda", dtype=torch.int32)
    for tokens in (1, 64, 1024):
        _stage(hidden[:tokens], weights[:tokens], ids[:tokens])
    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for tokens in (2, 17, 63, 65, 200, 1023, 1025, 2048):
            _stage(hidden[:tokens], weights[:tokens], ids[:tokens])


def test_stage_reads_rows_past_int32_offsets():
    # Two rows 2**31 elements apart: the second row's offset needs int64.
    stride = 2**31
    base = torch.zeros(stride + 128, device="cuda", dtype=torch.bfloat16)
    hidden = base.as_strided((2, 128), (stride, 1))
    hidden[1].copy_(torch.randn(128, device="cuda", dtype=torch.bfloat16))
    weights = torch.rand(2, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (2, 6), device="cuda", dtype=torch.int32)

    got = _stage(hidden, weights, ids)
    want = _reference(hidden.contiguous(), weights, ids)
    for a, b in zip(got, want):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _shared_rows(tokens, block_m):
    """Rows of the shared expert's MN-major scales: grouped by BLOCK_M, UTCCP within 128."""
    rows = []
    aligned = (block_m + 127) // 128 * 128
    for token in range(tokens):
        m = token % block_m
        rows.append(
            token // block_m * aligned + m // 128 * 128 + m % 32 * 4 + m % 128 // 32
        )
    return torch.tensor(rows, device="cuda")


def _shared_scales(size):
    """A shared-expert scale buffer, MN-major like DeepGEMM's, filled with a sentinel.

    Its row count is fixed, as the symmetric buffer's is, so its stride is too.
    """
    return torch.full((size // 128, 1024), -1, device="cuda", dtype=torch.int32).t()


@pytest.mark.parametrize("tokens,block_m", [(1, 16), (37, 16), (200, 64), (300, 192)])
def test_shared_scales_land_in_block_m_rows(tokens, block_m):
    torch.manual_seed(tokens + block_m)
    hidden = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)
    plain = _stage(hidden, weights, ids)
    shared = _shared_scales(512)
    out = (
        torch.empty_like(plain[0]),
        torch.empty_like(plain[1]),
        torch.empty_like(plain[2]),
        torch.empty_like(plain[3]),
    )
    stage_mxfp4_mega_moe_inputs(hidden, weights, ids, *out, shared, block_m, None)

    for got, want in zip(out, plain):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    rows = _shared_rows(tokens, block_m)
    assert torch.equal(shared[rows], plain[1])
    untouched = torch.ones(shared.shape[0], dtype=torch.bool, device="cuda")
    untouched[rows] = False
    assert (shared[untouched] == -1).all()


def test_shared_block_m_does_not_recompile():
    hidden = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(512, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (512, 6), device="cuda", dtype=torch.int32)

    def stage(tokens, block_m):
        out = (
            torch.empty(tokens, 512, device="cuda", dtype=torch.float8_e4m3fn),
            torch.empty(tokens, 4, device="cuda", dtype=torch.int32),
            torch.empty(tokens, 6, device="cuda", dtype=torch.int64),
            torch.empty(tokens, 6, device="cuda", dtype=torch.float32),
        )
        stage_mxfp4_mega_moe_inputs(
            hidden[:tokens],
            weights[:tokens],
            ids[:tokens],
            *out,
            _shared_scales(512),
            block_m,
            None,
        )

    for tokens in (1, 64, 300):
        stage(tokens, 16)
    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for tokens, block_m in ((2, 32), (65, 48), (300, 192), (512, 128)):
            stage(tokens, block_m)


@pytest.mark.parametrize(
    "tokens,live,padded",
    [
        (48, 5, 8),
        (48, 0, 8),
        (96, 16, 16),
        (1500, 7, 250),
        # tokens * live exceeds int32.
        (1500, 2**21, 2**22),
    ],
)
def test_filler_rows_route_no_experts(tokens, live, padded):
    torch.manual_seed(tokens + live)
    hidden = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)
    plain = _stage(hidden, weights, ids)
    out = tuple(torch.empty_like(t) for t in plain)
    counts = torch.tensor([live, padded], device="cuda", dtype=torch.int32)
    stage_mxfp4_mega_moe_inputs(hidden, weights, ids, *out, None, None, counts)

    rows = tokens * live // padded
    # Codes and scales are staged for every row; routing only for live ones.
    for got, want in zip(out[:2], plain[:2]):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    assert torch.equal(out[2][:rows], plain[2][:rows])
    assert torch.equal(out[3][:rows], plain[3][:rows])
    assert (out[2][rows:] == -1).all()
    assert (out[3][rows:] == 0).all()


def test_live_counts_do_not_recompile():
    hidden = torch.randn(96, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(96, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (96, 6), device="cuda", dtype=torch.int32)
    counts = torch.ones(2, device="cuda", dtype=torch.int32)
    out = _stage(hidden, weights, ids)
    stage_mxfp4_mega_moe_inputs(hidden, weights, ids, *out, None, None, counts)
    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for live, padded in ((3, 16), (0, 16), (16, 16)):
            counts.copy_(torch.tensor([live, padded]))
            stage_mxfp4_mega_moe_inputs(hidden, weights, ids, *out, None, None, counts)
