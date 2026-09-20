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

"""FlashInfer KDA preparation, indexed state and dependent-launch contracts."""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid
from tokenspeed_kernel.ops.attention import kda as facade
from tokenspeed_kernel.ops.attention.kda import (
    try_kda_fused_paged_decode,
    try_kda_fused_paged_verify,
)
from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
    fused_kda_verify_conv_update,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.thirdparty.flashinfer.kda import _epilogue as epilogue_api
from tokenspeed_kernel.thirdparty.flashinfer.kda import _prepare as prepare_api
from tokenspeed_kernel.thirdparty.flashinfer.kda import (
    flashinfer_kda_recurrent_available,
)
from tokenspeed_kernel.thirdparty.flashinfer.kda._epilogue import _gated_rmsnorm_bf16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_blackwell,
    reason="requires Blackwell KDA kernels",
)
requires_flashinfer = pytest.mark.skipif(
    not flashinfer_kda_recurrent_available(),
    reason="requires FlashInfer recurrent/frozen KDA",
)
H, D = 12, 128
P = H * D


def _make_inputs(batch, tokens, opaque, state_aligned):
    torch.manual_seed(619 + batch * tokens)
    rows, pages = batch * tokens, 2 * batch + 5

    def rnd(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1

    packed = rnd(rows, 6400)
    conv = torch.empty_strided(
        (pages, 4608, 3), (14080, 3, 1), device="cuda", dtype=torch.bfloat16
    )
    conv.copy_(rnd(*conv.shape))
    storage = torch.empty(pages * 196864 + 1, device="cuda", dtype=torch.bfloat16)
    state = storage.as_strided(
        (pages, 12, 128, 128),
        (196864, 16384, 128, 1),
        0 if state_aligned else 1,
    )
    state.copy_(rnd(*state.shape))
    if opaque:
        # Include NaN encodings: staging must preserve words, not values.
        for tensor in (packed, state):
            bits = torch.arange(tensor.numel(), device="cuda", dtype=torch.int32)
            tensor.view(torch.int16).copy_(bits.to(torch.int16).reshape(tensor.shape))
    conv[0].zero_()
    state[0].zero_()
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    writes = reads + batch
    if batch > 1:
        reads[1] = reads[0]
        reads[-1] = -1
    if batch > 3:
        writes[-2] = -1
    payload = tuple(
        torch.full((rows + 3, width + 8), 271, device="cuda", dtype=torch.int16).view(
            torch.bfloat16
        )
        for width in (4608, 128, 12)
    )
    scratch = torch.empty(batch + 2, 12, 128, 128, device="cuda", dtype=torch.bfloat16)
    scratch.view(torch.int16).fill_(311)
    return SimpleNamespace(
        packed=packed,
        payload=payload,
        scratch=scratch,
        args=(
            packed[:, :4608],
            packed[:, 6144:6272],
            rnd(1536, 128),
            packed[:, 6272:6284],
            rnd(4608, 4),
            conv,
            state,
            reads,
            writes,
        ),
    )


def rnd(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.2


def check(actual, ref, label):
    difference = actual.float() - ref.float()
    error = (
        (difference.square().mean() / ref.float().square().mean().clamp_min(1e-12))
        .sqrt()
        .item()
    )
    assert torch.isfinite(actual).all(), label
    assert error < 0.015, (label, error, difference.abs().max().item())
    return error


@requires_flashinfer
@pytest.mark.parametrize(
    "batch,tokens", [(1, 1), (5, 1), (129, 1), (5, 3), (16, 4), (16, 8)]
)
def test_registered_paths_and_refreshed_graph_indices(batch, tokens, monkeypatch):
    selected = []
    real_select = facade.select_kernel

    def select(*args, **kwargs):
        kernel = real_select(*args, **kwargs)
        selected.append(kernel.name)
        return kernel

    monkeypatch.setattr(facade, "select_kernel", select)
    torch.manual_seed(4709 + 17 * batch + tokens)
    rows, pages = batch * tokens, batch * 2 + 8
    data = rnd(rows, 4 * P + 256)
    raw, out_gate, fa, beta = (
        data[:, : 3 * P],
        data[:, 3 * P : 4 * P],
        data[:, 4 * P : 4 * P + D],
        data[:, 4 * P + D : 4 * P + D + H],
    )
    fb, weights = rnd(P, D), rnd(3 * P, 4)
    norm = torch.ones(D, device="cuda", dtype=torch.bfloat16)
    a = torch.nn.Parameter(torch.zeros(H, device="cuda"))
    bias = torch.nn.Parameter(torch.zeros(P, device="cuda"))
    conv = torch.empty_strided(
        (pages, 3 * P, 3), (9 * P + 256, 3, 1), device="cuda", dtype=torch.bfloat16
    )
    state = torch.empty_strided(
        (pages, H, D, D),
        (H * D * D + 256, D * D, D, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    conv.copy_(rnd(*conv.shape))
    state.copy_(rnd(*state.shape))
    conv[0].zero_()
    state[0].zero_()
    c0, s0 = conv.clone(), state.clone()
    cref, sref = c0.clone(), s0.clone()
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    writes = reads + batch
    if batch >= 4:
        reads[1] = reads[0]
        writes[2] = reads[2]
        reads[-1] = -1
        writes[-1] = -1
    if batch >= 5:
        reads[3] = -1
        writes[3] = batch + 4
    boundaries = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    scratch_indices = torch.zeros(batch, tokens, device="cuda", dtype=torch.int32)
    state_scratch = torch.empty(batch, H, D, D, device="cuda", dtype=torch.bfloat16)

    payload = tuple(torch.empty_like(value) for value in (raw, fa, beta))

    def run(c, s, solution):
        if tokens == 1:
            result = try_kda_fused_paged_decode(
                raw,
                weights,
                c,
                fa,
                fb,
                beta,
                a,
                bias,
                state_pool=s,
                read_indices=reads,
                write_indices=writes,
                num_heads=H,
                head_dim=D,
                cu_seqlens=boundaries,
                lower_bound=-5.0,
                output_gate=out_gate,
                norm_weight=norm,
                norm_eps=1e-5,
                recurrent_layout="v_major",
                solution=solution,
            )
            assert result is not None and result.output_norm_applied
            return result.out
        extra = {}
        if solution == "triton":
            extra = dict(
                g_raw=torch.nn.functional.linear(fa, fb),
                conv_qkv=fused_kda_verify_conv_update(
                    raw,
                    weights,
                    c,
                    reads,
                    num_heads=H,
                    head_dim=D,
                    draft_token_num=tokens,
                ),
            )
        return try_kda_fused_paged_verify(
            raw,
            weights,
            c,
            c,
            fa,
            fb,
            beta,
            a,
            bias,
            replay_payload=payload if solution is None else None,
            state_pool=s,
            state_scratch=state_scratch,
            read_indices=reads,
            write_indices=scratch_indices,
            num_heads=H,
            head_dim=D,
            draft_token_num=tokens,
            lower_bound=-5.0,
            store_states=False,
            recurrent_layout="v_major",
            solution=solution,
            **extra,
        )

    live = (
        writes >= 0
        if tokens == 1
        else torch.ones(rows, device="cuda", dtype=torch.bool)
    )
    actual, expected = run(conv, state, None), run(cref, sref, "triton")
    assert selected[0].startswith("flashinfer_kda_recurrent_producer_")
    if tokens > 1:
        for src, dst in zip((raw, fa, beta), payload):
            assert torch.equal(src.view(torch.int16), dst.view(torch.int16))
    check(
        actual.view(rows, H, D)[live],
        expected.view(rows, H, D)[live],
        "registered output",
    )
    if tokens == 1:
        torch.testing.assert_close(conv, cref, atol=0, rtol=0)
        check(state, sref, "registered state")
    else:
        torch.testing.assert_close(conv, c0, atol=0, rtol=0)
        torch.testing.assert_close(state, s0, atol=0, rtol=0)
    # Capture once, change the indices, and replay against a fresh reference.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(conv, state, None)
    conv.copy_(c0)
    state.copy_(s0)
    cref.copy_(c0)
    sref.copy_(s0)
    reads.fill_(1)
    writes.copy_(
        torch.arange(batch + 1, 2 * batch + 1, device="cuda", dtype=torch.int32)
    )
    graph.replay()
    expected = run(cref, sref, "triton")
    check(captured, expected, "registered graph output")
    if tokens == 1:
        torch.testing.assert_close(conv, cref, atol=0, rtol=0)
        check(state, sref, "registered graph state")
    else:
        torch.testing.assert_close(conv, c0, atol=0, rtol=0)
        torch.testing.assert_close(state, s0, atol=0, rtol=0)

    if tokens > 1:
        for src, dst in zip((raw, fa, beta), payload):
            assert torch.equal(src.view(torch.int16), dst.view(torch.int16))


def _run_generic(raw, fa, fb, beta, cw, conv, state, reads, writes):
    batch = raw.shape[0]
    # Fixed reference layout, independent of production dispatch tuning.
    bt, bk, warps = 1, 16, 2
    qkv = torch.empty((3, batch, 12, 128), device=raw.device, dtype=raw.dtype)
    gate = torch.empty((batch, 12, 128), device=raw.device, dtype=raw.dtype)
    logits = torch.empty((batch, 12), device=raw.device, dtype=raw.dtype)
    prepare_api._kda_recurrent_producer[(12, (batch + bt - 1) // bt, 128 // bk)](
        raw,
        fa,
        fb,
        beta,
        cw,
        conv,
        state,
        reads,
        writes,
        qkv,
        gate,
        logits,
        state,
        None,
        None,
        None,
        None,
        REPLAY_RAW_STRIDE=0,
        REPLAY_FA_STRIDE=0,
        REPLAY_BETA_STRIDE=0,
        CAPTURE_REPLAY=False,
        ROWS=batch,
        H=12,
        K=128,
        DFA=128,
        TOKENS=1,
        RAW_STRIDE=raw.stride(0),
        FA_STRIDE=fa.stride(0),
        BETA_STRIDE=beta.stride(0),
        CONV_STRIDE=conv.stride(0),
        STATE_STRIDE=state.stride(0),
        STATE_OUT_STRIDE=state.stride(0),
        BT=bt,
        BK=bk,
        STORE_CONV=True,
        GATHER_STATE=False,
        DOT=False,
        PDL=False,
        num_warps=warps,
    )
    return qkv, gate, logits


def _prepare(case, tokens, payload):
    return prepare_api.prepare_kda_recurrent_inputs(
        *case.args,
        tokens=tokens,
        state_scratch=None if tokens == 1 else case.scratch,
        replay_payload=payload,
    )


@pytest.mark.parametrize(
    "batch,pattern,state_aligned",
    [
        (1, "random", True),
        (5, "cancellation", False),
        (16, "scaled", True),
        (65, "zero", True),
        (129, "random", False),
    ],
)
def test_compact_matches_generic(batch, pattern, state_aligned, monkeypatch):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("compact producer requires SM100 or SM103")
    case = _make_inputs(batch, 1, False, state_aligned)
    raw, fa, fb, beta, cw, conv, state, reads, writes = case.args
    if pattern == "cancellation":
        fb[:, 1::2] = -fb[:, ::2]
        fa[:, 1::2] = fa[:, ::2]
    elif pattern == "scaled":
        fb *= 128
        fa /= 128
    elif pattern == "zero":
        raw.zero_()
        fa.zero_()
    before, state_before = conv.clone(), state.view(torch.int16).clone()
    reference_conv = torch.empty_strided(
        conv.shape, conv.stride(), device="cuda", dtype=conv.dtype
    )
    reference_conv.copy_(conv)
    monkeypatch.setattr(prepare_api, "pdl_enabled", lambda: False)

    def compare(actual):
        expected = _run_generic(
            raw, fa, fb, beta, cw, reference_conv, state, reads, writes
        )
        for value, ref in zip(actual[:3], expected):
            assert torch.equal(value.view(torch.int16), ref.view(torch.int16))
        assert torch.equal(conv.view(torch.int16), reference_conv.view(torch.int16))
        assert torch.equal(state.view(torch.int16), state_before)
        protected = torch.ones(conv.shape[0], device="cuda", dtype=torch.bool)
        protected[writes[writes >= 0].long()] = False
        assert torch.equal(conv[protected], before[protected])

    compare(_prepare(case, 1, None))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _prepare(case, 1, None)
    reads.fill_(1)
    writes.copy_(
        torch.arange(batch + 1, 2 * batch + 1, device="cuda", dtype=torch.int32)
    )
    case.packed.normal_(0, 0.1)
    conv.copy_(before)
    reference_conv.copy_(before)
    graph.replay()
    compare(captured)


@pytest.mark.parametrize(
    "batch,tokens,opaque",
    [
        (1, 2, False),
        (5, 3, True),
        (5, 4, False),
        (17, 3, True),
        (17, 7, False),
        (17, 8, True),
        (17, 16, False),
    ],
)
def test_verify_staging_and_payload_are_bitwise(batch, tokens, opaque):
    case = _make_inputs(batch, tokens, opaque, True)
    raw, fa, _, beta, conv_w, conv, state, reads, _ = case.args
    before_c, before_s = conv.view(torch.int16).clone(), state.view(torch.int16).clone()
    rows = batch * tokens

    def check(actual):
        if not opaque:
            # Independent conv reference: each sample comes from history or
            # the current window, including fresh/shared source requests.
            history = conv.index_select(0, reads.clamp_min(0).long()).float()
            window = torch.cat(
                (history, raw.view(batch, tokens, -1).transpose(1, 2).float()),
                dim=2,
            )
            expected_conv = torch.zeros_like(raw, dtype=torch.float32)
            for tap in range(4):
                sample = window[:, :, tap : tap + tokens].transpose(1, 2)
                expected_conv += sample.reshape(rows, -1) * conv_w[:, tap].float()
            expected_conv = torch.nn.functional.silu(expected_conv).to(raw.dtype)
            actual_conv = torch.cat([v.reshape(rows, -1) for v in actual[0]], dim=1)
            torch.testing.assert_close(actual_conv, expected_conv, atol=1e-5, rtol=0.02)
        for source, dest in zip((raw, fa, beta), case.payload):
            width = source.shape[1]
            assert torch.equal(
                source.view(torch.int16), dest[:rows, :width].view(torch.int16)
            )
            assert torch.all(dest[rows:].view(torch.int16) == 271)
            assert torch.all(dest[:rows, width:].view(torch.int16) == 271)
        expected = state.index_select(0, reads.clamp_min(0).long())
        assert torch.equal(actual[3].view(torch.int16), expected.view(torch.int16))
        assert torch.equal(
            actual[4], torch.arange(batch, device="cuda", dtype=torch.int32)
        )
        assert torch.all(case.scratch[batch:].view(torch.int16) == 311)
        assert torch.equal(conv.view(torch.int16), before_c)
        assert torch.equal(state.view(torch.int16), before_s)

    actual = _prepare(case, tokens, case.payload)
    check(actual)
    expected = _prepare(case, tokens, None)
    for a, b in zip(actual[:3], expected[:3]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _prepare(case, tokens, case.payload)
    case.packed.normal_(0, 0.1)
    reads.fill_(1)
    for dest in case.payload:
        dest.view(torch.int16).fill_(271)
    graph.replay()
    check(captured)


def test_invalid_replay_capture_is_rejected():
    case = _make_inputs(2, 1, False, True)
    with pytest.raises(ValueError, match="frozen verification"):
        _prepare(case, 1, case.payload)
    case = _make_inputs(2, 3, False, True)
    with pytest.raises(ValueError, match="three destination"):
        _prepare(case, 3, case.payload[:2])
    with pytest.raises(ValueError, match="Invalid replay"):
        _prepare(case, 3, (case.payload[0][:1], *case.payload[1:]))


@triton.jit
def _publish_projected_inputs(
    source,
    output,
    source_reads,
    source_writes,
    reads,
    writes,
    N: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The consumer may prefetch immutable weights, but must wait before
    # consuming either the projected activation or the dynamic indices.
    tl.extra.cuda.gdc_launch_dependents()
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + idx, mask=idx < N, other=0)
    tl.store(output + idx, value, mask=idx < N)
    r = tl.load(source_reads + idx, mask=idx < B, other=-1)
    w = tl.load(source_writes + idx, mask=idx < B, other=-1)
    tl.store(reads + idx, r, mask=idx < B)
    tl.store(writes + idx, w, mask=idx < B)


def test_decode_pdl_waits_for_projected_inputs_and_indices(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires programmatic dependent launch")
    batch = 5
    case = _make_inputs(batch, 1, False, True)
    wide = case.packed
    _, _, fb, _, cw, conv, state, reads, writes = case.args
    staging = torch.empty_like(wide)
    staged_reads, staged_writes = torch.empty_like(reads), torch.empty_like(writes)
    reference_conv, original_conv = conv.clone(), conv.clone()
    candidate_inputs = SimpleNamespace(
        args=(
            staging[:, :4608],
            staging[:, 6144:6272],
            fb,
            staging[:, 6272:6284],
            cw,
            conv,
            state,
            staged_reads,
            staged_writes,
        )
    )
    reference_inputs = SimpleNamespace(
        args=(
            case.args[0],
            case.args[1],
            fb,
            case.args[3],
            cw,
            reference_conv,
            state,
            reads,
            writes,
        )
    )
    # Warm both flag values before capture; no state contents enter a cache key.
    monkeypatch.setattr(prepare_api, "pdl_enabled", lambda: False)
    _prepare(reference_inputs, 1, None)
    staging.copy_(wide)
    staged_reads.copy_(reads)
    staged_writes.copy_(writes)
    monkeypatch.setattr(prepare_api, "pdl_enabled", lambda: True)
    _prepare(candidate_inputs, 1, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _publish_projected_inputs[(triton.cdiv(wide.numel(), 256),)](
            wide,
            staging,
            reads,
            writes,
            staged_reads,
            staged_writes,
            wide.numel(),
            batch,
            256,
            num_warps=4,
            launch_pdl=True,
        )
        captured = _prepare(candidate_inputs, 1, None)
    for step in range(4):
        wide.copy_(torch.randn_like(wide) * 0.1)
        reads.fill_(1 if step % 2 else -1)
        conv.copy_(original_conv)
        reference_conv.copy_(original_conv)
        graph.replay()
        monkeypatch.setattr(prepare_api, "pdl_enabled", lambda: False)
        expected = _prepare(reference_inputs, 1, None)
        for index in (0, 1, 2):
            assert torch.equal(
                captured[index].view(torch.int16), expected[index].view(torch.int16)
            )
        assert torch.equal(conv.view(torch.int16), reference_conv.view(torch.int16))


@pytest.mark.parametrize("batch,zero", [(1, False), (65, True)])
def test_bf16_decode_norm_matches_same_input(batch, zero):
    torch.manual_seed(4709 + batch)
    heads, dim = 12, 128
    x = torch.randn(batch, heads * dim, device="cuda", dtype=torch.bfloat16)
    if zero:
        x.zero_()
    packed_gate = torch.randn(batch, 2 * heads * dim, device="cuda", dtype=x.dtype)
    gate = packed_gate[:, heads * dim :]
    weight = torch.randn(dim, device="cuda", dtype=x.dtype)
    reference = rmsnorm_gated_sigmoid(
        x, gate, weight, 1e-5, heads, dim, enable_pdl=True
    )
    actual = _gated_rmsnorm_bf16(x, gate, weight, 1e-5, heads, dim, enable_pdl=True)
    # Different warp layouts can straddle a BF16 rounding midpoint. Check
    # both implementations against the mathematical result, and bound their
    # difference to one BF16 encoding step instead of requiring identical bits.
    values = x.double().view(batch, heads, dim)
    exact = (
        values
        * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-5)
        * weight.double()
        * torch.sigmoid(gate.double().view(batch, heads, dim))
    ).reshape_as(x)
    rounding_rtol = (
        torch.finfo(torch.bfloat16).eps / 2 + 8 * torch.finfo(torch.float32).eps
    )
    for output in (actual, reference):
        torch.testing.assert_close(output.double(), exact, rtol=rounding_rtol, atol=0)
    ulps = (actual.view(torch.int16).int() - reference.view(torch.int16).int()).abs()
    assert ulps.max().item() <= 1


@triton.jit
def _late_publish(source, target, N: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    begin = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
    )
    now = begin
    while now - begin < 4096:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %clock64;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
        )
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + idx, mask=idx < N, other=0)
    tl.store(target + idx, value, mask=idx < N)


@triton.jit
def _dependent_copy(source, target, N: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + idx, mask=idx < N, other=0)
    tl.store(target + idx, value, mask=idx < N)


@pytest.mark.parametrize("batch,fp8", [(5, False), (65, True)])
def test_epilogue_early_signal_waits_for_source_and_publishes_result(batch, fp8):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("early epilogue specialization requires SM100 or SM103")
    torch.manual_seed(9210 + batch)
    source = torch.randn(batch, 1536, device="cuda", dtype=torch.bfloat16)
    staging = torch.empty_like(source)
    gate = torch.randn_like(source)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 + 1

    def norm(value, use_pdl):
        if fp8:
            return epilogue_api.gated_rmsnorm_fp8_prepacked(
                value,
                gate,
                weight,
                eps=1e-5,
                num_heads=12,
                head_dim=128,
                enable_pdl=use_pdl,
            )
        return (
            epilogue_api._gated_rmsnorm_bf16(
                value, gate, weight, 1e-5, 12, 128, enable_pdl=use_pdl
            ),
        )

    staging.copy_(source)
    warm = norm(staging, True)
    norm(source, False)
    copies = tuple(torch.empty_like(value) for value in warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _late_publish[(triton.cdiv(source.numel(), 256),)](
            source, staging, source.numel(), 256, num_warps=4, launch_pdl=True
        )
        produced = norm(staging, True)
        for value, copied in zip(produced, copies):
            # Copy opaque bytes: a masked FP8 scalar fill itself is not a
            # supported Triton integer-to-FP8 cast and is irrelevant here.
            source_bytes, target_bytes = value.view(torch.uint8), copied.view(
                torch.uint8
            )
            _dependent_copy[(triton.cdiv(source_bytes.numel(), 256),)](
                source_bytes,
                target_bytes,
                source_bytes.numel(),
                256,
                num_warps=4,
                launch_pdl=True,
            )
    for _ in range(4):
        source.copy_(torch.randn_like(source))
        graph.replay()
        expected = norm(source, False)
        for actual, reference in zip(copies, expected):
            assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
