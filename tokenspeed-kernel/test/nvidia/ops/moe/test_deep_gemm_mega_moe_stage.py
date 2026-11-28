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

from tokenspeed_kernel.ops.moe import EplbDispatch  # noqa: E402
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (  # noqa: E402
    _stage_mxfp4_mega_moe_inputs_kernel,
    stage_mxfp4_mega_moe_inputs,
)
from utils import assert_no_triton_compile  # noqa: E402


def test_staging_maps_strided_ids_past_int32_offsets():
    stride = 2**30
    storage = torch.empty(4 * stride + 1, device="cuda", dtype=torch.int32)
    # If 2 * stride wraps to a negative int32, the load reads this guard.
    storage[0] = 0
    ids = storage.as_strided((3, 1), (stride, 1), storage_offset=2 * stride)
    ids[:, 0].copy_(torch.arange(3, device="cuda", dtype=torch.int32))
    replicas = torch.tensor([[2], [0], [1]], device="cuda", dtype=torch.int32)
    counts = torch.ones(3, device="cuda", dtype=torch.int32)
    load = torch.zeros(3, device="cuda", dtype=torch.int64)
    dispatch = EplbDispatch(replicas, counts, load, None)
    for logical_ids, expected in ((ids, [[2], [0], [1]]), (ids.T, [[2, 0, 1]])):
        load.zero_()
        rows = logical_ids.shape[0]
        live = torch.ones(rows, dtype=torch.bool, device="cuda")
        hidden = torch.zeros(rows, 128, dtype=torch.bfloat16, device="cuda")
        weights = torch.ones(logical_ids.shape, device="cuda")
        staged = _stage(hidden, weights, logical_ids, routing_prepared=(dispatch, live))
        assert staged[2].tolist() == expected
        assert load.tolist() == [1, 1, 1]


def _stage(hidden, weights, ids, *, live_requests=None, routing_prepared=None):
    tokens, size = hidden.shape
    out = (
        torch.empty(tokens, size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(tokens, size // 128, device="cuda", dtype=torch.int32),
        torch.full((tokens, ids.shape[1]), -1, device="cuda", dtype=torch.int64),
        torch.empty(tokens, ids.shape[1], device="cuda", dtype=torch.float32),
    )
    stage_mxfp4_mega_moe_inputs(
        hidden, weights, ids, *out, None, None, live_requests, routing_prepared
    )
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


@pytest.mark.parametrize("windowed", [False, True])
def test_staging_replays_with_current_replicas_and_recording_mask(windowed):
    tokens = 72
    hidden = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, 3, device="cuda")
    rows = torch.arange(tokens, device="cuda")
    ids = ((torch.arange(3, device="cuda")[:, None] + rows) % 3 - 1).T
    replicas = torch.tensor(
        [[0, 3, 4], [1, 5, -1], [2, -1, -1]], dtype=torch.int32, device="cuda"
    )
    counts = torch.tensor([3, 2, 1], dtype=torch.int32, device="cuda")
    load = torch.zeros(6, dtype=torch.int64, device="cuda")
    enabled = torch.tensor(False, device="cuda") if windowed else None
    record_rows = rows % 3 != 0
    requests = torch.tensor([9, 9], dtype=torch.int32, device="cuda")
    routing = EplbDispatch(replicas, counts, load, enabled), record_rows
    _stage(hidden, weights, ids, live_requests=requests, routing_prepared=routing)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        staged = _stage(
            hidden, weights, ids, live_requests=requests, routing_prepared=routing
        )

    for live, record in ((9, False), (5, True), (0, True)):
        requests[0].fill_(live)
        if enabled is not None:
            enabled.fill_(record)
        replicas.copy_(replicas.flip((0,)))
        counts.copy_(counts.flip((0,)))
        cutoff = tokens * live // 9
        record_rows.copy_((rows < cutoff) & (rows % 3 != 0))
        load.fill_(2**33)
        graph.replay()

        hashed = (rows[:, None] * 2654435769) & 0xFFFFFFFF
        physical = replicas[ids, hashed % counts[ids]].long()
        filler = rows[:, None] >= cutoff
        expected = _reference(
            hidden, weights.masked_fill(filler, 0), physical.masked_fill(filler, -1)
        )
        for got, want in zip(staged, expected, strict=True):
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        histogram = torch.bincount(
            physical[record_rows].flatten(), minlength=load.numel()
        )
        assert torch.equal(load - 2**33, histogram * (not windowed or record))


@pytest.mark.parametrize("eplb", [False, True])
def test_stage_token_count_does_not_recompile(eplb):
    hidden = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(2048, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (2048, 6), device="cuda", dtype=torch.int32)
    routing = None
    if eplb:
        replicas = torch.arange(384, dtype=torch.int32, device="cuda")[:, None]
        dispatch = EplbDispatch(
            replicas,
            torch.ones_like(replicas[:, 0]),
            torch.zeros(384, dtype=torch.int64, device="cuda"),
            None,
        )
        routing = dispatch, torch.ones(2048, dtype=torch.bool, device="cuda")

    def stage(tokens):
        # A transposed router output has a column stride that follows the batch size.
        logical_ids = ids[:tokens].reshape(6, tokens).T
        return _stage(
            hidden[:tokens], weights[:tokens], logical_ids, routing_prepared=routing
        )

    for tokens in (1, 64, 1024):
        stage(tokens)
    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for tokens in (2, 17, 63, 65, 200, 1023, 1025, 2048):
            stage(tokens)


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
