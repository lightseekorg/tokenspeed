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

"""CuTe DSL RecoverSSM verify against the Triton verify and an FP64 recurrence.

Every case fills outputs and records with sentinels, derives the expected
write set from the request lengths and record metadata, and checks: outputs
within one BF16 step of the FP64 recurrence (or, near zero, within FP32
cancellation noise), corrections against it, key/gate records and stamps
exactly equal to the Triton verify's, nothing written outside the write set,
and the checkpoint unchanged.
"""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
pytest.importorskip("quack")
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    vllm_triton_kda_recoverssm_verify,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)
import tokenspeed_kernel.ops.attention.kda.cute_dsl as cute_dsl_module  # noqa: E402
from tokenspeed_kernel.ops.attention.kda._cute_dsl.recoverssm_verify import (  # noqa: E402
    recoverssm_verify_supported,
)
from tokenspeed_kernel.ops.attention.kda.cute_dsl import (  # noqa: E402
    cute_dsl_kda_recoverssm_verify,
)

HEADS = 12
# Sentinels that no kernel writes for these inputs.
OUT_FILL, CORRECTION_FILL, KEY_GATE_FILL, STAMP_FILL = 7.0, 13.0, 5.0, -7
# Rows after the packed tokens; the widest window could store this far past them.
OUT_GUARD_ROWS = 16
# Elements after each serving payload row.
CORRECTION_ROW_PAD, KEY_GATE_ROW_PAD = 16, 32
# Near-zero outputs carry FP32 cancellation noise of this size relative to the largest.
OUTPUT_NOISE = 4e-7
# Corrections must match FP64 to this fraction of the largest correction.
CORRECTION_TOLERANCE = 1e-6


def _inputs(
    *,
    batch,
    width,
    lower_bound,
    seed,
    key_dim=128,
    value_dim=128,
    heads=HEADS,
    start=0,
    serving_records=False,
    index_dtype=torch.int32,
    beta_dtype=torch.bfloat16,
    state_scale=0.02,
    value_scale=0.2,
    first_gate=None,
    second_gate=None,
    zero_token=None,
    separate_rows=False,
    empty=False,
):
    """Serving-shaped verify inputs: conv-chunk q/k/v, packed beta, scattered pages.

    Request 0 fills its window, request 1 has padding page 0, request 3 has
    padding page -1 and a full window, and with more than two requests the last
    has one token, which leaves the most room for stores past the packed tokens.
    """
    torch.manual_seed(seed)
    device = "cuda"
    lens = torch.randint(0, width + 1, (batch,))
    if empty:
        lens.zero_()
    else:
        lens[0] = width
        if batch > 3:
            lens[3] = width
        if batch > 2:
            lens[-1] = 1
    query_start_loc = torch.zeros(batch + 1, dtype=torch.int64)
    query_start_loc[1:] = torch.cumsum(lens, 0)
    tokens = int(query_start_loc[-1])
    widths = [heads * key_dim, heads * key_dim, heads * value_dim]
    conv = torch.randn(tokens, sum(widths), device=device).bfloat16()
    dims = (key_dim, key_dim, value_dim)
    q, k, v = (
        part.view(1, tokens, heads, dim)
        for part, dim in zip(conv.split(widths, dim=-1), dims)
    )
    if separate_rows:
        # Distinct aligned row pitches catch a token stride used for the wrong tensor.
        q, k, v = (_repitch(x, extra) for x, extra in ((q, 8), (k, 24), (v, 40)))
    v.mul_(value_scale)
    raw_g = torch.randn(tokens, heads * key_dim, device=device).bfloat16()
    for offset, gate in ((0, first_gate), (1, second_gate)):
        if gate is not None:
            rows = (query_start_loc[:-1] + offset).clamp(max=tokens - 1)
            raw_g[rows.to(device)] = gate
    if zero_token is not None:
        rows = (query_start_loc[:-1] + zero_token).clamp(max=tokens - 1).to(device)
        q[0, rows] = 0
        k[0, rows] = 0
    packed_beta = torch.randn(tokens, heads + 40, device=device).bfloat16()
    pages = batch + 5
    state_indices = torch.randperm(pages - 1)[:batch] + 1
    if batch > 1:
        state_indices[1] = 0
    if batch > 3:
        state_indices[3] = -1
    capacity = width if serving_records else 2 * width
    slots = batch if serving_records else batch + 3
    record_slots = (
        torch.arange(batch) if serving_records else torch.randperm(slots)[:batch]
    )
    metadata = torch.stack(
        [
            record_slots,
            torch.randint(0, 9, (batch,)),
            torch.randint(0, 1000, (batch,)) + start,
        ],
        1,
    )
    state = torch.randn(pages, heads, value_dim, key_dim, device=device)
    return dict(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g.view(1, tokens, heads, key_dim),
        raw_beta=packed_beta[:, 8 : 8 + heads].to(beta_dtype).view(1, tokens, heads),
        A_log=torch.randn(heads, device=device).mul_(0.1),
        dt_bias=torch.randn(heads, key_dim, device=device).mul_(0.1),
        checkpoint_state=state.mul_(state_scale),
        query_start_loc=query_start_loc.to(device, index_dtype),
        state_indices=state_indices.to(device, index_dtype),
        spec_query_len=width,
        lower_bound=lower_bound,
        record_metadata=metadata.to(device).contiguous(),
        capacity=capacity,
        slots=slots,
        serving_records=serving_records,
    )


def _repitch(x, extra):
    """Copy of x whose token rows sit ``extra`` elements apart in a fresh buffer."""
    tokens, row = x.shape[1], x.shape[2] * x.shape[3]
    rows = torch.zeros(tokens, row + extra, device=x.device).bfloat16()[:, :row]
    rows.copy_(x.reshape(tokens, row))
    return rows.view(x.shape)


def _destinations(inputs):
    """Sentinel-filled outputs and records; serving records view padded payload rows."""
    _, tokens, heads, key_dim = inputs["q"].shape
    value_dim = inputs["v"].shape[-1]
    cap, slots = inputs["capacity"], inputs["slots"]
    device = inputs["q"].device
    out_rows = torch.full(
        (1, tokens + OUT_GUARD_ROWS, heads, value_dim),
        OUT_FILL,
        device=device,
        dtype=torch.bfloat16,
    )
    result = dict(out_rows=out_rows, out=out_rows[:, :tokens])
    if inputs["serving_records"]:
        width = inputs["spec_query_len"]
        correction_rows = torch.full(
            (slots * width, heads * value_dim + CORRECTION_ROW_PAD),
            CORRECTION_FILL,
            device=device,
        )
        key_gate_rows = torch.full(
            (slots * width, 2 * heads * key_dim + KEY_GATE_ROW_PAD),
            KEY_GATE_FILL,
            device=device,
            dtype=torch.bfloat16,
        )
        # As in serving: [slot, head, position, channel] views of payload rows.
        result["correction_rows"] = correction_rows
        result["key_gate_rows"] = key_gate_rows
        result["correction_cache"] = (
            correction_rows[:, : heads * value_dim]
            .view(slots, width, heads, value_dim)
            .transpose(1, 2)
        )
        result["kg_cache"] = (
            key_gate_rows[:, : 2 * heads * key_dim]
            .view(slots, width, heads, 2 * key_dim)
            .transpose(1, 2)
        )
        stamp_pool = torch.full(
            (2, slots + 2, cap, 2), STAMP_FILL, device=device, dtype=torch.int64
        )
        result["stamp_pool"] = stamp_pool
        result["record_stamps"] = stamp_pool[1, :slots]
    else:
        result["correction_cache"] = torch.full(
            (slots, heads, cap, value_dim), CORRECTION_FILL, device=device
        )
        result["kg_cache"] = torch.full(
            (slots, heads, cap, 2 * key_dim),
            KEY_GATE_FILL,
            device=device,
            dtype=torch.bfloat16,
        )
        result["record_stamps"] = torch.full(
            (slots, cap, 2), STAMP_FILL, device=device, dtype=torch.int64
        )
    return result


def _run(verify, inputs, with_stamps=True, dest=None, **overrides):
    """Run ``verify`` into ``dest`` (fresh sentinel destinations when None)."""
    result = _destinations(inputs) if dest is None else dest
    args = {**inputs, **overrides}
    verify(
        args["q"],
        args["k"],
        args["v"],
        args["raw_g"],
        args["raw_beta"],
        args["A_log"],
        args["dt_bias"],
        checkpoint_state=args["checkpoint_state"],
        correction_cache=result["correction_cache"],
        kg_cache=result["kg_cache"],
        query_start_loc=args["query_start_loc"],
        state_indices=args["state_indices"],
        spec_query_len=args["spec_query_len"],
        lower_bound=args["lower_bound"],
        out=result["out"],
        record_metadata=args["record_metadata"] if with_stamps else None,
        record_stamps=result["record_stamps"] if with_stamps else None,
    )
    return result


def _supported(inputs, result, **overrides):
    args = {**inputs, **overrides}
    return recoverssm_verify_supported(
        args["q"],
        args["k"],
        args["v"],
        args["raw_g"],
        args["raw_beta"],
        args["A_log"],
        args["dt_bias"],
        args["checkpoint_state"],
        result["correction_cache"],
        result["kg_cache"],
        args["query_start_loc"],
        args["state_indices"],
        args["spec_query_len"],
        args["lower_bound"],
        result["out"],
    )


def _reference(inputs):
    """FP64 recurrence over each request's valid window."""
    starts = inputs["query_start_loc"].tolist()
    key_dim = inputs["q"].shape[-1]
    out = torch.zeros(inputs["v"].shape, dtype=torch.float64, device="cuda")
    corrections = {}
    A = inputs["A_log"].double().exp()
    for b in range(len(starts) - 1):
        page = int(inputs["state_indices"][b])
        if page <= 0:
            continue
        S = inputs["checkpoint_state"][page].double().clone()
        for t in range(starts[b + 1] - starts[b]):
            tok = starts[b] + t
            q = inputs["q"][0, tok].double()
            k = inputs["k"][0, tok].double()
            x = inputs["raw_g"][0, tok].double() + inputs["dt_bias"].double()
            if inputs["lower_bound"] is not None:
                gate = inputs["lower_bound"] * torch.sigmoid(A[:, None] * x)
            else:
                gate = -A[:, None] * torch.nn.functional.softplus(x)
            beta = torch.sigmoid(inputs["raw_beta"][0, tok].double())
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * key_dim**-0.5
            S = S * gate.exp()[:, None, :]
            value = inputs["v"][0, tok].double()
            c = (value - (S * k[:, None, :]).sum(-1)) * beta[:, None]
            S = S + c[:, :, None] * k[:, None, :]
            out[0, tok] = (S * q[:, None, :]).sum(-1)
            corrections[(b, t)] = c
    return out, corrections


def _assert_outputs_match(actual, expected):
    """Each output is within max(one BF16 step, OUTPUT_NOISE * max|expected|).

    The BF16 step is the spacing next to the rounded FP64 value in the direction
    of the error; the second term is the FP32 cancellation noise near zero.
    """
    rounded = expected.to(torch.bfloat16).double()
    mantissa, exponent = torch.frexp(rounded.abs())
    away = torch.where(
        rounded != 0, torch.ldexp(torch.ones_like(rounded), exponent - 8), 0.0
    )
    # Below a power of two the BF16 spacing halves.
    toward = torch.where(mantissa == 0.5, away / 2, away)
    diff = actual.double() - rounded
    step = torch.where(diff * rounded > 0, away, toward)
    allowed = torch.clamp(step, min=OUTPUT_NOISE * float(expected.abs().max()))
    excess = diff.abs() - allowed
    worst = int(excess.argmax())
    where = torch.unravel_index(torch.tensor(worst), excess.shape)
    assert float(excess.max()) <= 0, (
        f"output [token, head, channel] {[int(i) for i in where[1:]]} exceeds "
        f"its allowance by {float(excess.max())}"
    )


def _check(inputs, cute, triton, snapshot, with_stamps=True):
    torch.cuda.synchronize()
    assert torch.equal(inputs["checkpoint_state"], snapshot)
    ref_out, ref_corr = _reference(inputs)
    _check_outputs(inputs, cute, ref_out)
    _check_records(inputs, cute, triton, ref_corr, with_stamps)


def _check_outputs(inputs, cute, ref_out):
    tokens = inputs["q"].shape[1]
    assert bool((cute["out_rows"][:, tokens:] == OUT_FILL).all())
    _assert_outputs_match(cute["out"], ref_out)
    starts = inputs["query_start_loc"].tolist()
    for b in range(len(starts) - 1):
        if int(inputs["state_indices"][b]) <= 0:
            assert bool((cute["out"][0, starts[b] : starts[b + 1]] == 0).all()), b


def _check_records(inputs, cute, triton, ref_corr, with_stamps):
    assert torch.equal(cute["kg_cache"], triton["kg_cache"]), _first_difference(
        cute["kg_cache"], triton["kg_cache"]
    )
    assert torch.equal(
        cute["record_stamps"], triton["record_stamps"]
    ), _first_difference(cute["record_stamps"], triton["record_stamps"])
    key_dim = inputs["q"].shape[-1]
    starts = inputs["query_start_loc"].tolist()
    meta = inputs["record_metadata"].tolist()
    cap = inputs["capacity"]
    written = torch.zeros(cute["correction_cache"].shape[0], cap, dtype=torch.bool)
    scale = max([float(c.abs().max()) for c in ref_corr.values()] + [0.0])
    for (b, t), c in ref_corr.items():
        slot, generation, start = meta[b] if with_stamps else (b, 0, 0)
        pos = (start + t) % cap
        written[slot, pos] = True
        got = cute["correction_cache"][slot, :, pos].double()
        error = float((got - c).abs().max())
        assert error <= CORRECTION_TOLERANCE * scale + 1e-9, (b, t, error)
        kg = cute["kg_cache"][slot, :, pos]
        assert torch.equal(kg[:, :key_dim], inputs["k"][0, starts[b] + t]), (b, t)
        assert torch.equal(kg[:, key_dim:], inputs["raw_g"][0, starts[b] + t]), (b, t)
        if with_stamps:
            assert cute["record_stamps"][slot, pos].tolist() == [start + t, generation]
    _check_untouched(inputs, cute, written, with_stamps)


def _check_untouched(inputs, cute, written, with_stamps):
    """Record slots and positions outside the write set keep their sentinels."""
    untouched = ~written.to(cute["correction_cache"].device)
    for name, records, fill in (
        ("correction", cute["correction_cache"].transpose(1, 2), CORRECTION_FILL),
        ("key/gate", cute["kg_cache"].transpose(1, 2), KEY_GATE_FILL),
    ) + ((("stamp", cute["record_stamps"], STAMP_FILL),) if with_stamps else ()):
        changed = (records != fill).flatten(2).any(-1) & untouched
        assert not bool(
            changed.any()
        ), f"{name} written at [slot, position] {changed.nonzero()[0].tolist()}"
    if inputs["serving_records"]:
        pool = cute["stamp_pool"].clone()
        pool[1, : inputs["slots"]] = STAMP_FILL
        assert bool((pool == STAMP_FILL).all())
        for name, fill, pad in (
            ("correction_rows", CORRECTION_FILL, CORRECTION_ROW_PAD),
            ("key_gate_rows", KEY_GATE_FILL, KEY_GATE_ROW_PAD),
        ):
            assert bool((cute[name][:, -pad:] == fill).all()), name


def _first_difference(actual, expected):
    index = tuple((actual != expected).nonzero()[0].tolist())
    return f"first difference at {list(index)}: {actual[index]} vs {expected[index]}"


def _no_fallback(*args, **kwargs):
    raise AssertionError("supported inputs reached the Triton fallback")


def _compare(inputs, with_stamps=True):
    """Triton, then the CuTe DSL entry with its Triton fallback forbidden; check both."""
    snapshot = inputs["checkpoint_state"].clone()
    triton = _run(vllm_triton_kda_recoverssm_verify, inputs, with_stamps)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cute_dsl_module, "vllm_triton_kda_recoverssm_verify", _no_fallback
        )
        cute = _run(cute_dsl_kda_recoverssm_verify, inputs, with_stamps)
    assert _supported(inputs, cute)
    _check(inputs, cute, triton, snapshot, with_stamps)


def _case(case_id, **case):
    return pytest.param(case, id=case_id)


_GATES = (("bounded", -5.0), ("softplus", None))
# Two, sixteen and thirty-two requests select 32-, 64- and 128-row tiles for W <= 8.
SHAPE_CASES = [
    _case(f"w{width}_b{batch}_{gate}", batch=batch, width=width, lower_bound=bound)
    for width in (1, 2, 3, 4, 8, 16)
    for batch in (2, 16, 32)
    for gate, bound in _GATES
]
DIM_CASES = [
    _case(
        f"k{key_dim}_v{value_dim}_w{width}_{gate}",
        batch=16,
        width=width,
        key_dim=key_dim,
        value_dim=value_dim,
        lower_bound=bound,
    )
    for key_dim, value_dim in ((64, 64), (96, 96), (64, 128), (128, 64), (32, 32))
    for width in (4, 8)
    for gate, bound in _GATES
]
# Windows above eight use 32-row tiles; W * |lower_bound| <= 48 keeps them factored.
FACTORED_BOUNDARY_CASES = [
    _case("w16_bound3_factored", batch=16, width=16, lower_bound=-3.0),
    _case("w12_bound4_factored", batch=16, width=12, lower_bound=-4.0),
    _case("w12_softplus", batch=16, width=12, lower_bound=None),
    _case("w5_bound5", batch=16, width=5, lower_bound=-5.0),
]
LAYOUT_CASES = [
    _case(
        f"serving_records_b{batch}_start{start}",
        batch=batch,
        width=8,
        lower_bound=-5.0,
        serving_records=True,
        start=start,
    )
    for batch in (2, 16, 32)
    for start in (0, 2**32 - 2)
] + [
    # Capacities that do not divide 2^32 pin the 64-bit ring position.
    _case(
        "capacity12_large_start", batch=16, width=6, lower_bound=-5.0, start=2**32 - 2
    ),
    _case(
        "serving_capacity6_large_start",
        batch=16,
        width=6,
        lower_bound=-5.0,
        serving_records=True,
        start=2**32 - 2,
    ),
    _case(
        "int64_indices", batch=16, width=8, lower_bound=-5.0, index_dtype=torch.int64
    ),
    _case("fp32_beta", batch=16, width=8, lower_bound=-5.0, beta_dtype=torch.float32),
    _case("one_head", batch=16, width=8, lower_bound=-5.0, heads=1),
    _case("separate_rows", batch=16, width=8, lower_bound=-5.0, separate_rows=True),
]
VALUE_CASES = [
    _case(
        f"large_state_{gate}",
        batch=16,
        width=8,
        lower_bound=bound,
        state_scale=0.5,
        value_scale=1.0,
    )
    for gate, bound in _GATES
] + [
    _case("zero_key_query_rows", batch=16, width=8, lower_bound=-5.0, zero_token=1),
    # A huge early gate must not swallow the later gates of the window.
    _case(
        "huge_first_gate_w8", batch=16, width=8, lower_bound=None, first_gate=2.0**26
    ),
    _case(
        "huge_first_gate_w2", batch=16, width=2, lower_bound=None, first_gate=2.0**26
    ),
    _case(
        "two_huge_gates_w3",
        batch=16,
        width=3,
        lower_bound=None,
        first_gate=2.0**60,
        second_gate=2.0**26,
    ),
    _case(
        "two_huge_gates_w8",
        batch=16,
        width=8,
        lower_bound=None,
        first_gate=2.0**60,
        second_gate=2.0**26,
    ),
    # Finite gates whose sum overflows FP32.
    _case(
        "overflowing_gates_w2",
        batch=16,
        width=2,
        lower_bound=None,
        first_gate=2.0**127,
        second_gate=2.0**127,
    ),
    _case(
        "overflowing_gates_w8",
        batch=16,
        width=8,
        lower_bound=None,
        first_gate=2.0**127,
        second_gate=2.0**127,
    ),
]


@pytest.mark.parametrize(
    "case",
    SHAPE_CASES + DIM_CASES + FACTORED_BOUNDARY_CASES + LAYOUT_CASES + VALUE_CASES,
)
def test_cute_dsl_recoverssm_verify_matches_reference(case: dict) -> None:
    _compare(_inputs(seed=case["batch"] * 7 + case["width"], **case))


def test_cute_dsl_recoverssm_verify_does_not_recompile_across_batches() -> None:
    # At twelve heads, batches 4 to 16 share the 64-row tile: warm once, then vary
    # the batch, contents, boundaries, pages and record starts without a trace.
    _compare(_inputs(batch=8, width=8, lower_bound=-5.0, seed=31))
    with assert_no_cute_compile():
        for batch, seed in ((4, 32), (13, 33), (16, 34)):
            _compare(
                _inputs(
                    batch=batch, width=8, lower_bound=-5.0, seed=seed, start=seed * 997
                )
            )


def test_all_empty_windows_write_nothing() -> None:
    inputs = _inputs(batch=4, width=8, lower_bound=-5.0, seed=21, empty=True)
    for verify in (vllm_triton_kda_recoverssm_verify, cute_dsl_kda_recoverssm_verify):
        result = _run(verify, inputs)
        torch.cuda.synchronize()
        assert bool((result["out_rows"] == OUT_FILL).all())
        assert bool((result["correction_cache"] == CORRECTION_FILL).all())
        assert bool((result["kg_cache"] == KEY_GATE_FILL).all())
        assert bool((result["record_stamps"] == STAMP_FILL).all())


def test_cute_dsl_recoverssm_verify_without_metadata_and_stamps() -> None:
    inputs = _inputs(batch=6, width=4, lower_bound=-5.0, seed=11)
    inputs["slots"] = 6
    _compare(inputs, with_stamps=False)


def test_cute_dsl_recoverssm_verify_replays_in_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cute_dsl_module, "vllm_triton_kda_recoverssm_verify", _no_fallback
    )
    inputs = _inputs(batch=16, width=8, lower_bound=-5.0, seed=3)
    snapshot = inputs["checkpoint_state"].clone()
    triton = _run(vllm_triton_kda_recoverssm_verify, inputs)
    # Compile eagerly, then capture into fresh sentinel destinations.
    _run(cute_dsl_kda_recoverssm_verify, inputs)
    cute = _destinations(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(cute_dsl_kda_recoverssm_verify, inputs, dest=cute)
    graph.replay()
    _check(inputs, cute, triton, snapshot)
    # Replay again on new contents, boundaries, pages and record starts.
    inputs["v"].mul_(-1.5)
    inputs["raw_g"].add_(0.25)
    lens = torch.roll(inputs["query_start_loc"].diff(), 1)
    inputs["query_start_loc"][1:] = torch.cumsum(lens, 0)
    inputs["state_indices"].copy_(torch.roll(inputs["state_indices"], 1))
    inputs["record_metadata"][:, 2] += 3
    triton = _run(vllm_triton_kda_recoverssm_verify, inputs)
    for name, fill in (
        ("out", OUT_FILL),
        ("correction_cache", CORRECTION_FILL),
        ("kg_cache", KEY_GATE_FILL),
        ("record_stamps", STAMP_FILL),
    ):
        cute[name].fill_(fill)
    graph.replay()
    _check(inputs, cute, triton, snapshot)


def _misaligned_rows(inputs):
    """q, k and v copied into one buffer one element off 16-byte alignment."""
    q = inputs["q"]
    tokens, row = q.shape[1], q.shape[2] * q.shape[3]
    wide = torch.zeros(tokens, 3 * row + 1, device="cuda").bfloat16()
    views = {}
    for i, name in enumerate(("q", "k", "v")):
        part = wide[:, 1 + i * row : 1 + (i + 1) * row]
        part.copy_(inputs[name].reshape(tokens, row))
        views[name] = part.view(q.shape)
    return views


def _shifted(x):
    """Copy of x one element past a 16-byte boundary."""
    flat = torch.zeros(x.numel() + 8, device="cuda", dtype=x.dtype)
    moved = flat[1 : 1 + x.numel()].view(x.shape)
    moved.copy_(x)
    return moved


def _strided(x, length):
    """``x`` as every other element of a buffer of ``length``."""
    backing = torch.zeros(length, device="cuda", dtype=x.dtype)
    backing[::2] = x
    return backing[::2]


def _sparse_state(inputs):
    """The checkpoint as the first half of each 256-wide key row."""
    state = inputs["checkpoint_state"]
    wide = torch.zeros(*state.shape[:3], 256, device="cuda")
    wide[..., :128] = state
    return wide[..., :128]


UNADDRESSABLE_INPUTS = {
    "misaligned_rows": _misaligned_rows,
    "misaligned_gate": lambda a: dict(raw_g=_shifted(a["raw_g"])),
    "misaligned_state": lambda a: dict(
        checkpoint_state=_shifted(a["checkpoint_state"])
    ),
    "sparse_state": lambda a: dict(checkpoint_state=_sparse_state(a)),
    "strided_indices": lambda a: dict(state_indices=_strided(a["state_indices"], 32)),
    "strided_starts": lambda a: dict(
        query_start_loc=_strided(a["query_start_loc"], 34)
    ),
}
UNADDRESSABLE_RECORDS = {
    "strided_correction_channels": "correction_cache",
    "strided_key_gate_channels": "kg_cache",
}


def _strided_record_channels(inputs, name):
    """Destinations whose ``name`` record has a channel stride of two."""
    dest = _destinations(inputs)
    record = dest[name]
    fill = CORRECTION_FILL if name == "correction_cache" else KEY_GATE_FILL
    wide = torch.full(
        (*record.shape[:3], 2 * record.shape[3]),
        fill,
        device="cuda",
        dtype=record.dtype,
    )
    dest[name] = wide[..., ::2]
    return dest


def _assert_same_results(cute, triton):
    for name in ("out", "correction_cache", "kg_cache", "record_stamps"):
        assert torch.equal(cute[name], triton[name]), name


@pytest.mark.parametrize("layout", [*UNADDRESSABLE_INPUTS, *UNADDRESSABLE_RECORDS])
def test_unaddressable_layouts_run_the_triton_verify(layout: str) -> None:
    inputs = _inputs(batch=16, width=8, lower_bound=-5.0, seed=5)
    overrides = (
        UNADDRESSABLE_INPUTS[layout](inputs) if layout in UNADDRESSABLE_INPUTS else {}
    )
    record = UNADDRESSABLE_RECORDS.get(layout)
    destinations = [
        _strided_record_channels(inputs, record) if record else None for _ in range(2)
    ]
    triton = _run(
        vllm_triton_kda_recoverssm_verify, inputs, dest=destinations[0], **overrides
    )
    cute = _run(
        cute_dsl_kda_recoverssm_verify, inputs, dest=destinations[1], **overrides
    )
    assert not _supported(inputs, cute, **overrides)
    _assert_same_results(cute, triton)


def test_singleton_a_log_with_a_stride_runs_the_triton_verify() -> None:
    inputs = _inputs(batch=16, width=8, lower_bound=-5.0, seed=13, heads=1)
    overrides = dict(A_log=_strided(inputs["A_log"], 2))
    triton = _run(vllm_triton_kda_recoverssm_verify, inputs, **overrides)
    cute = _run(cute_dsl_kda_recoverssm_verify, inputs, **overrides)
    assert not _supported(inputs, cute, **overrides)
    _assert_same_results(cute, triton)


def test_positive_lower_bound_runs_the_triton_verify() -> None:
    # A positive bound grows the state; a zero checkpoint keeps Triton's result finite.
    inputs = _inputs(batch=16, width=2, lower_bound=60.0, seed=17, state_scale=0.0)
    triton = _run(vllm_triton_kda_recoverssm_verify, inputs)
    cute = _run(cute_dsl_kda_recoverssm_verify, inputs)
    assert not _supported(inputs, cute)
    _assert_same_results(cute, triton)


def test_fp16_gate_parameters() -> None:
    inputs = _inputs(batch=16, width=8, lower_bound=-5.0, seed=9)
    inputs["A_log"] = inputs["A_log"].half()
    inputs["dt_bias"] = inputs["dt_bias"].half()
    _compare(inputs)


@pytest.mark.parametrize(
    "dtype,head_dim,width,state_dtype,expected",
    [
        (torch.bfloat16, 128, 8, torch.float32, "cute_dsl_kda_recoverssm_verify"),
        (torch.bfloat16, 64, 16, torch.float32, "cute_dsl_kda_recoverssm_verify"),
        (torch.bfloat16, 128, 17, torch.float32, "vllm_triton_kda_recoverssm_verify"),
        (torch.bfloat16, 80, 8, torch.float32, "vllm_triton_kda_recoverssm_verify"),
        (torch.bfloat16, 160, 8, torch.float32, "vllm_triton_kda_recoverssm_verify"),
        (torch.float16, 128, 8, torch.float32, "vllm_triton_kda_recoverssm_verify"),
        (torch.bfloat16, 128, 8, torch.bfloat16, "vllm_triton_kda_recoverssm_verify"),
    ],
)
def test_recoverssm_verify_selection(
    dtype: torch.dtype,
    head_dim: int,
    width: int,
    state_dtype: torch.dtype,
    expected: str,
) -> None:
    fmt = dense_tensor_format(dtype)
    kernel = select_kernel(
        "attention",
        "kda_recoverssm_verify",
        format_signature(q=fmt, k=fmt, v=fmt),
        traits={
            "head_dim": head_dim,
            "spec_query_len": width,
            "paged_state": True,
            "recurrent_layout": "v_major",
            "state_dtype": state_dtype,
        },
    )
    assert kernel.name == expected
