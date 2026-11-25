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

"""CuTe DSL multi-token verify with fused producers against the split producers.

The reference projects the gate with ``F.linear``, convolves with the Triton
verify producer and runs the registered record verify, as serving's split
producers do. Inputs are serving-shaped: q|k|v,
f_a and beta are column views of one packed projection row, records view padded
payload rows, and pages include padding (0 and -1). Destinations start as
sentinels, so equality on whole tensors also checks that nothing outside the
write set changes. Key records, stamps and retained rows must match exactly;
the gate projection sums in a different order, so a raw gate may differ by one
BF16 step and outputs and corrections by what that step propagates. The fused
output norm is checked against the norm kernel on the plain output.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
pytest.importorskip("quack")
from tokenspeed_kernel.platform import ArchVersion, current_platform

if not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)
from kda_retained_reference import prepare_history  # noqa: E402
from tokenspeed_kernel.ops.activation.triton import (  # noqa: E402
    rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel.ops.attention.kda import (  # noqa: E402
    KdaOutputNorm,
    KdaRetainedHistory,
    KdaRetainedRecords,
    kda_fused_paged_verify,
    kda_fused_paged_verify_fuses_producers,
)
from tokenspeed_kernel.ops.attention.kda._cute_dsl import (  # noqa: E402
    fused_verify_window as window_kernel,
)
from tokenspeed_kernel.ops.attention.kda._cute_dsl import (  # noqa: E402
    recoverssm_verify as recoverssm_kernel,
)
from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (  # noqa: E402
    fused_kda_verify_conv_update,
)

HEADS, DIM, RANK = 12, 128, 128
WIDTH = HEADS * DIM
# q|k|v|g, f_a, beta and padding, as in the Kimi-K3 TP8 projection row.
PROJECTION = 4 * WIDTH + RANK + HEADS + 4
CORRECTION_FILL, KEY_GATE_FILL, STAMP_FILL, ORIGINAL_FILL = 13.0, 5.0, -7, 3.0
CORRECTION_ROW_PAD, KEY_GATE_ROW_PAD = 16, 32


def _inputs(batch, window, *, seed, lower_bound=-5.0, metadata=True, f_a_pitch=0):
    """Request 1 has padding page 0 and request 2 page -1 (zero history)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    rows = batch * window
    projection = randn(rows, PROJECTION + f_a_pitch).bfloat16()
    pages = batch + 6
    state_indices = torch.randperm(pages - 1, device="cuda")[:batch].to(torch.int32) + 1
    if batch > 2:
        state_indices[1] = 0
        state_indices[2] = -1
    slots = batch + 2
    record_metadata = None
    if metadata:
        record_metadata = torch.stack(
            [
                torch.randperm(slots, device="cuda")[:batch],
                torch.randint(1, 99, (batch,), device="cuda"),
                torch.randint(0, 1000, (batch,), device="cuda"),
            ],
            1,
        ).to(torch.int64)
    return dict(
        mixed_qkv=projection[:, : 3 * WIDTH],
        out_gate=projection[:, 3 * WIDTH : 4 * WIDTH],
        f_a_out=projection[:, 4 * WIDTH + f_a_pitch : 4 * WIDTH + f_a_pitch + RANK],
        beta_logits=projection[:, 4 * WIDTH + f_a_pitch + RANK :][:, :HEADS],
        conv_weights=randn(3 * WIDTH, 4, scale=0.5).bfloat16(),
        # Pages carry more channels than q|k|v, as serving pools may.
        conv_states=randn(pages, 3 * WIDTH + 64, 3).bfloat16(),
        f_b_weight=randn(WIDTH, RANK, scale=0.1).bfloat16(),
        A_log=randn(HEADS, scale=0.5),
        dt_bias=randn(HEADS * DIM, scale=0.5),
        state_pool=randn(pages, HEADS, DIM, DIM, scale=0.02),
        state_indices=state_indices,
        query_start_loc=torch.arange(
            0, rows + 1, window, device="cuda", dtype=torch.int32
        ),
        record_metadata=record_metadata,
        slots=slots,
        window=window,
        lower_bound=lower_bound,
    )


def _records(x, *, original, capacity, correction_stride=1, history_length=0):
    """Sentinel-filled request records, with padding around each slot.

    With ``history_length``, every slot first holds that many accepted records.
    """
    slots, rows = x["slots"], x["mixed_qkv"].shape[0]
    correction_rows = torch.full(
        (slots, capacity * WIDTH * correction_stride + CORRECTION_ROW_PAD),
        CORRECTION_FILL,
        device="cuda",
    )
    key_gate_rows = torch.full(
        (slots, capacity * 2 * WIDTH + KEY_GATE_ROW_PAD), KEY_GATE_FILL, device="cuda"
    )
    packed = (
        key_gate_rows[:, : capacity * 2 * WIDTH]
        .view(slots, capacity, HEADS, 2 * DIM)
        .transpose(1, 2)
    )
    history = KdaRetainedHistory(
        correction_rows[:, : capacity * WIDTH * correction_stride]
        .view(slots, capacity, HEADS, DIM * correction_stride)
        .transpose(1, 2)[..., ::correction_stride],
        packed[..., :DIM],
        packed[..., DIM:],
        torch.zeros(slots, 1, dtype=torch.int32, device="cuda"),
        x["state_indices"],
        torch.arange(x["state_indices"].numel(), device="cuda", dtype=torch.int32),
    )
    if history_length:
        gen = torch.Generator(device="cuda").manual_seed(history_length)
        shape = (slots, HEADS, history_length, DIM)
        accepted = slice(0, history_length)
        history.correction[:, :, accepted] = 0.02 * torch.randn(
            shape, device="cuda", generator=gen
        )
        history.key[:, :, accepted] = F.normalize(
            torch.randn(shape, device="cuda", generator=gen), dim=-1
        )
        history.decay[:, :, accepted] = 0.7 + 0.25 * torch.rand(
            shape, device="cuda", generator=gen
        )
        history.length.fill_(history_length)
    stamps = torch.full(
        (x["slots"], capacity, 2), STAMP_FILL, device="cuda", dtype=torch.int64
    )
    retained = torch.full(
        (rows, 3 * WIDTH + 8), ORIGINAL_FILL, device="cuda"
    ).bfloat16()
    records = KdaRetainedRecords(
        original_qkv=retained[:, : 3 * WIDTH] if original else None,
        history=history,
        metadata=x["record_metadata"],
        stamps=stamps if x["record_metadata"] is not None else None,
        query_start_loc=x["query_start_loc"],
    )
    return records, (correction_rows, key_gate_rows, stamps, retained)


def _verify(x, records, output_norm, *, override=None, **producers):
    # Serving readies the history before every verify round.
    prepare_history(
        x["state_pool"],
        records.history,
        x["state_indices"],
        x["query_start_loc"],
        draft_token_num=x["window"],
    )
    return kda_fused_paged_verify(
        x["mixed_qkv"],
        x["conv_weights"],
        x["conv_states"],
        x["f_a_out"],
        x["f_b_weight"],
        x["beta_logits"],
        x["A_log"],
        x["dt_bias"],
        state_pool=x["state_pool"],
        read_indices=x["state_indices"],
        num_heads=HEADS,
        head_dim=DIM,
        draft_token_num=x["window"],
        lower_bound=x["lower_bound"],
        recurrent_layout="v_major",
        retained_records=records,
        output_norm=output_norm,
        override=override,
        solution=None,
        **producers,
    )


def _split(x, capacity, *, correction_stride=1, history_length=0):
    """The serving split producers, then the registered record verify."""
    records, buffers = _records(
        x,
        original=False,
        capacity=capacity,
        correction_stride=correction_stride,
        history_length=history_length,
    )
    conv_qkv = fused_kda_verify_conv_update(
        x["mixed_qkv"],
        x["conv_weights"],
        x["conv_states"],
        x["state_indices"],
        num_heads=HEADS,
        head_dim=DIM,
        draft_token_num=x["window"],
        original_qkv=buffers[3][:, : 3 * WIDTH],
        query_start_loc=x["query_start_loc"],
    )
    g_raw = F.linear(x["f_a_out"], x["f_b_weight"])
    return _verify(x, records, None, g_raw=g_raw, conv_qkv=conv_qkv), buffers


def _fused(x, capacity, output_norm=None, *, history_length=0):
    records, buffers = _records(
        x, original=True, capacity=capacity, history_length=history_length
    )
    return _verify(x, records, output_norm), buffers


def _gate_input_step(x, decays_ref):
    """The decay change one BF16 step of each gate projection can make.

    The split producers round the gate projection to BF16 in a GEMM whose
    accumulation order varies across GPUs, where the fused kernel keeps it in
    FP32. The input z = projection + dt_bias of each written decay follows
    from the reference decay, and the decay moves by about decay times the
    log-decay slope times one BF16 step of the projection.
    """
    column = torch.arange(decays_ref.shape[1], device="cuda")
    head = column // DIM % HEADS
    a = x["A_log"].float().exp()[head]
    bias = x["dt_bias"].float()[head * DIM + column % DIM]
    decay = decays_ref.clamp(min=2.0**-126, max=1.0)
    if x["lower_bound"] is None:
        # log decay = -a softplus(z); softplus(z) is its negation over a.
        softplus = -torch.log(decay) / a
        z = torch.log(torch.expm1(softplus))
        slope = a * -torch.expm1(-softplus)
    else:
        # log decay = lower_bound sigmoid(a z).
        gate = (torch.log(decay) / x["lower_bound"]).clamp(2.0**-24, 1 - 2.0**-24)
        z = torch.logit(gate) / a
        slope = -x["lower_bound"] * a * gate * (1 - gate)
    step = decay * slope * (z - bias).abs() * 2.0**-7
    written = decays_ref != KEY_GATE_FILL
    return torch.where(written, torch.nan_to_num(step, nan=0.0, posinf=0.0), 0.0)


def _assert_matches(x, expected, actual, capacity):
    out_ref, (corr_ref, kg_ref, stamps_ref, kept_ref) = expected
    out, (corr, kg, stamps, kept) = actual
    assert torch.equal(kept, kept_ref)
    assert torch.equal(stamps, stamps_ref)
    key_cols = torch.cat(
        [
            torch.arange(base, base + DIM)
            for base in range(0, capacity * 2 * WIDTH, 2 * DIM)
        ]
    ).cuda()
    torch.testing.assert_close(
        kg[:, key_cols], kg_ref[:, key_cols], atol=1e-7, rtol=2e-6
    )
    tail = capacity * 2 * WIDTH
    assert torch.equal(kg[:, tail:], kg_ref[:, tail:])
    gate_cols = key_cols + DIM
    gates, gates_ref = kg[:, gate_cols].float(), kg_ref[:, gate_cols].float()
    # One BF16 step of the larger gate, a floor for sums that cancel to near
    # zero, and what one BF16 step of the gate projection moves it by.
    step = torch.maximum(gates.abs(), gates_ref.abs()) * 2.0**-7
    step = step + gates_ref.abs().max() * 2.0**-16 + _gate_input_step(x, gates_ref)
    assert ((gates - gates_ref).abs() <= step).all()
    tail = capacity * WIDTH
    assert torch.equal(corr[:, tail:], corr_ref[:, tail:])
    written = corr_ref != CORRECTION_FILL
    assert torch.equal(written, corr != CORRECTION_FILL)
    scale = corr_ref[written].abs().max() if written.any() else 1.0
    assert ((corr - corr_ref).abs() <= 1e-4 * scale).all()
    out_ref, out = out_ref.float(), out.float()
    scale = out_ref.abs().max()
    assert ((out - out_ref).abs() <= 1e-3 * scale + out_ref.abs() * 2.0**-7).all()


@pytest.mark.parametrize("batch", [1, 2, 3, 5, 9, 16])
@pytest.mark.parametrize("window", [2, 3, 8, 16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_fused_window_matches_split_producers(batch, window, lower_bound):
    x = _inputs(batch, window, seed=batch * 31 + window, lower_bound=lower_bound)
    state, history = x["state_pool"].clone(), x["conv_states"].clone()
    _assert_matches(x, _split(x, window), _fused(x, window), window)
    assert torch.equal(x["state_pool"], state)
    assert torch.equal(x["conv_states"], history)


@pytest.mark.parametrize("capacity", [9, 13])
def test_record_positions_wrap_the_ring(capacity):
    x = _inputs(4, 8, seed=capacity)
    _assert_matches(x, _split(x, capacity), _fused(x, capacity), capacity)


def test_far_pages_of_a_pool_past_int32_verify_like_near_ones():
    """Page offsets past 2**31 elements verify exactly like near pages.

    The last pages of the large pool hold the same contents as the first pages
    of a small pool, so outputs and records must be equal.
    """
    x = _inputs(2, 8, seed=70)
    near = _fused(x, 8)
    pages = 2**31 // (HEADS * DIM * DIM) + 2
    assert (pages - 1) * HEADS * DIM * DIM > 2**31
    far_indices = torch.arange(pages - 2, pages, device="cuda", dtype=torch.int32)
    rows = x["state_indices"].long()
    state_pool = torch.empty(pages, HEADS, DIM, DIM, device="cuda")
    state_pool[far_indices.long()] = x["state_pool"][rows]
    conv_states = torch.empty(
        pages, *x["conv_states"].shape[1:], device="cuda", dtype=torch.bfloat16
    )
    conv_states[far_indices.long()] = x["conv_states"][rows]
    far = _fused(
        dict(
            x,
            state_pool=state_pool,
            conv_states=conv_states,
            state_indices=far_indices,
        ),
        8,
    )
    assert torch.equal(far[0], near[0])
    for got, want in zip(far[1], near[1]):
        assert torch.equal(got, want)


def test_fused_window_without_metadata_records_by_request():
    x = _inputs(4, 8, seed=40, metadata=False)
    _assert_matches(x, _split(x, 8), _fused(x, 8), 8)


@pytest.mark.parametrize("layout", ["f_a rows", "conv weights", "correction channels"])
def test_strided_or_unaligned_inputs_match_split_producers(layout):
    x = _inputs(3, 8, seed=41, f_a_pitch=int(layout == "f_a rows"))
    if layout == "conv weights":
        weights = x["conv_weights"]
        shifted = torch.empty(weights.numel() + 1, device="cuda", dtype=weights.dtype)
        x["conv_weights"] = shifted[1:].view_as(weights).copy_(weights)
    correction_stride = 2 if layout == "correction channels" else 1
    expected = _split(x, 8, correction_stride=correction_stride)
    records, buffers = _records(
        x, original=True, capacity=8, correction_stride=correction_stride
    )
    actual = _verify(x, records, None), buffers
    # The fallback runs the reference producers, so every result is exact.
    assert torch.equal(expected[0], actual[0])
    for ref, got in zip(expected[1], actual[1]):
        assert torch.equal(ref, got)


@pytest.mark.parametrize(
    ("window", "batch", "fused"),
    [(2, 1, True), (8, 16, True), (16, 4, True), (17, 1, False), (8, 17, False)],
)
def test_windows_up_to_sixteen_tokens_and_requests_fuse_their_producers(
    window, batch, fused
):
    assert (
        kda_fused_paged_verify_fuses_producers(
            torch.bfloat16,
            recurrent_layout="v_major",
            num_heads=HEADS,
            head_dim=DIM,
            draft_token_num=window,
            batch_size=batch,
        )
        is fused
    )


def _compile_buckets(window, *, packed):
    """Batches of 1 to 16 requests by the kernel they compile on the test device.

    A compile is keyed by the value tile rows, whether the grid holds one to
    two CTAs per SM (the shared-memory floor) and whether it copies records,
    which only packed records on 64-row grids of at most one CTA per SM do.
    With ``packed`` set, it assumes room for records that fit a CTA, as the
    packed geometry used here to predict the copy bucket has; unpacked
    buckets never copy, whatever their capacity.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    buckets = {}
    for batch in range(1, 17):
        rows = window_kernel._value_tile_rows(batch, HEADS, window, DIM)
        ctas = batch * HEADS * (DIM // rows)
        copies = packed and rows == 64 and ctas <= sms
        buckets.setdefault((rows, sms < ctas <= 2 * sms, copies), []).append(batch)
    return buckets


def test_serving_batches_reuse_one_compiled_kernel(monkeypatch):
    # One compile per value tile and spread on the test device's SMs: the first
    # batch of each bucket warms it and every other batch reuses it. The
    # compile cache starts empty, so no earlier test's kernel can stand in for
    # a batch's own, and every launch must get its bucket's kernel.
    window_kernel._compile.cache_clear()
    launches = _spy_compiles(monkeypatch)
    buckets = _compile_buckets(8, packed=False)
    warm = {}
    for bucket, batches in buckets.items():
        _fused(_inputs(batches[0], 8, seed=50 + batches[0]), 8)
        assert len(launches) == len(warm) + 1
        warm[bucket] = launches[-1]
    with assert_no_cute_compile():
        for bucket, batches in buckets.items():
            for batch in batches[1:]:
                launched = len(launches)
                _fused(_inputs(batch, 8, seed=60 + batch), 8)
                assert len(launches) == launched + 1
                arguments, kernel = launches[-1]
                assert arguments == warm[bucket][0] and kernel is warm[bucket][1]
    torch.cuda.synchronize()
    # Each bucket is its own compile: its tile and spread, never copies.
    for (rows, one_to_two, _), (arguments, _) in warm.items():
        assert arguments["value_tile_rows"] == rows
        assert (arguments["smem_floor"] > 0) == one_to_two
        assert arguments["copy_records"] is False
    assert len({id(kernel) for _, kernel in warm.values()}) == len(warm)


def test_grids_of_one_to_two_ctas_per_sm_spread_over_all_sms():
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    capacity = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
    floor = window_kernel._smem_floor(2 * sms, torch.device("cuda", 0))
    # A CTA claiming the floor plus the 1 KiB per-CTA reserve fits twice, not three times.
    assert 2 * (floor + 1024) <= capacity < 3 * (floor + 1024)
    for ctas in (sms, 2 * sms + 1):
        assert window_kernel._smem_floor(ctas, torch.device("cuda", 0)) == 0


def _output_norm(x, weight_dtype):
    gen = torch.Generator(device="cuda").manual_seed(91)
    weight = (1 + 0.1 * torch.randn(DIM, device="cuda", generator=gen)).to(weight_dtype)
    return KdaOutputNorm(x["out_gate"], weight, 1e-5)


@pytest.mark.parametrize("batch", [1, 3, 8, 16])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_fused_output_norm_matches_the_norm_kernel(batch, weight_dtype):
    """The epilogue stores the norm kernel's result on the BF16 output.

    Only the order of the sum of squares differs, so an element may move by one
    BF16 step; the records do not change at all.
    """
    x = _inputs(batch, 8, seed=80 + batch)
    norm = _output_norm(x, weight_dtype)
    plain, plain_records = _fused(x, 8)
    normed, normed_records = _fused(x, 8, norm)
    for got, want in zip(normed_records, plain_records):
        assert torch.equal(got, want)
    rows = x["mixed_qkv"].shape[0]
    expected = rmsnorm_gated_sigmoid(
        plain.view(rows, WIDTH),
        norm.gate,
        norm.weight,
        norm.eps,
        HEADS,
        DIM,
        enable_pdl=False,
    ).float()
    normed = normed.float()
    assert normed.shape == (rows, WIDTH)
    step = expected.abs() * 2.0**-7
    assert ((normed - expected).abs() <= step).all()
    assert (normed != expected).float().mean() < 0.01
    if batch > 2:
        assert not normed[8:24].any()


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("window", [3, 8])
def test_every_value_tile_gives_the_same_bits(batch, window, monkeypatch):
    """The value tile only decides which CTAs share the producers and norm sums.

    Every tile, down to one unclustered CTA per head, must store the same
    outputs and records, padding requests included.
    """
    x = _inputs(batch, window, seed=70 + batch + window)
    norm = _output_norm(x, torch.bfloat16)
    results = []
    for rows in (32, 64, 128):
        monkeypatch.setattr(window_kernel, "_value_tile_rows", lambda *_, n=rows: n)
        results.append(_fused(x, window, norm))
    out, records = results[0]
    for other_out, other_records in results[1:]:
        assert torch.equal(other_out, out)
        for got, want in zip(other_records, records):
            assert torch.equal(got, want)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("history_length", [5, 12])
def test_accepted_history_gives_the_same_bits_in_every_value_tile(
    batch, history_length, monkeypatch
):
    """Accepted records reach the outputs alike however the values split.

    The fused window and the split verify share the history projection. Each
    must store the same outputs and records for every value tile, up to one
    unclustered CTA per head.
    """
    x = _inputs(batch, 4, seed=90 + batch + history_length)
    fused, split = [], []
    for rows in (32, 64, 128):
        monkeypatch.setattr(window_kernel, "_value_tile_rows", lambda *_, n=rows: n)
        monkeypatch.setattr(recoverssm_kernel, "_value_tile_rows", lambda *_, n=rows: n)
        fused.append(_fused(x, 16, history_length=history_length))
        split.append(_split(x, 16, history_length=history_length))
    for results in (fused, split):
        out, records = results[0]
        for other_out, other_records in results[1:]:
            assert torch.equal(other_out, out)
            for got, want in zip(other_records, records):
                assert torch.equal(got, want)
    _assert_matches(x, split[0], fused[0], 16)


def test_unaligned_inputs_apply_the_output_norm_after_the_split_producers():
    x = _inputs(3, 8, seed=43, f_a_pitch=1)
    norm = _output_norm(x, torch.bfloat16)
    plain = _split(x, 8)[0]
    normed = _fused(x, 8, norm)[0]
    expected = rmsnorm_gated_sigmoid(
        plain.view(24, WIDTH),
        norm.gate,
        norm.weight,
        norm.eps,
        HEADS,
        DIM,
        enable_pdl=False,
    )
    assert torch.equal(normed, expected)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("raw_gate_only", [False, True])
def test_retained_prefix_append_preserves_base_and_publishes_on_commit(
    state_dtype, raw_gate_only
):
    from kda_retained_reference import check_fused_retained_append

    x = _inputs(2, 3, seed=143)
    x["state_pool"] = x["state_pool"].to(state_dtype)
    records = _records(x, original=True, capacity=8)[0]
    expected = _records(x, original=True, capacity=8)[0]
    check_fused_retained_append(
        x, records, expected, _verify, width=3, raw_gate_only=raw_gate_only
    )


def test_a_page_past_the_pool_verifies_as_padding():
    """A page past the checkpoint pool is padding, exactly like page -1: its
    checkpoint and convolution history are not read, it records nothing, and
    every result matches the page -1 run bit for bit."""
    runs = []
    for past_pool in (False, True):
        # _inputs draws its pages from the global generator.
        torch.manual_seed(223)
        x = _inputs(5, 4, seed=223)
        if past_pool:
            x["state_indices"][2] = x["state_pool"].shape[0]
        state, history = x["state_pool"].clone(), x["conv_states"].clone()
        runs.append(_fused(x, 4))
        assert torch.equal(x["state_pool"], state)
        assert torch.equal(x["conv_states"], history)
    (out, buffers), (past_out, past_buffers) = runs
    assert torch.equal(past_out, out)
    for past, padded in zip(past_buffers, buffers, strict=True):
        assert torch.equal(past, padded)


def _packed_records(x, capacity, history_length, first_slot=0):
    """Records laid out as serving allocates them: dense [slot, head, position, channel].

    Unlike ``_records``' interleaved views, these satisfy
    ``packed_record_width``, so 64-row tiles may bulk-copy them. The requests'
    record slots start at ``first_slot``; the buffers returned are the
    storage and lengths of the ``x["slots"]`` slots from there on, which hold
    the same contents whatever ``first_slot`` is. Only those slots are
    initialized: no request addresses the slots before them.
    """
    slots, rows = x["slots"], x["mixed_qkv"].shape[0]
    storage = torch.empty((first_slot + slots, 3, HEADS, capacity, DIM), device="cuda")
    lengths = torch.empty((first_slot + slots, 1), dtype=torch.int32, device="cuda")
    own = slice(first_slot, None)
    storage[own] = KEY_GATE_FILL
    storage[own, 0] = CORRECTION_FILL
    lengths[own] = history_length
    requests = x["state_indices"].numel()
    history = KdaRetainedHistory(
        *storage.unbind(1),
        lengths,
        x["state_indices"],
        torch.arange(
            first_slot, first_slot + requests, device="cuda", dtype=torch.int32
        ),
    )
    assert history.packed_record_width == DIM
    if history_length:
        gen = torch.Generator(device="cuda").manual_seed(history_length)
        shape = (slots, HEADS, history_length, DIM)
        accepted = slice(0, history_length)
        history.correction[own, :, accepted] = 0.02 * torch.randn(
            shape, device="cuda", generator=gen
        )
        history.key[own, :, accepted] = F.normalize(
            torch.randn(shape, device="cuda", generator=gen), dim=-1
        )
        history.decay[own, :, accepted] = 0.7 + 0.25 * torch.rand(
            shape, device="cuda", generator=gen
        )
    stamps = torch.full(
        (slots, capacity, 2), STAMP_FILL, device="cuda", dtype=torch.int64
    )
    retained = torch.full(
        (rows, 3 * WIDTH + 8), ORIGINAL_FILL, device="cuda"
    ).bfloat16()
    records = KdaRetainedRecords(
        original_qkv=retained[:, : 3 * WIDTH],
        history=history,
        metadata=x["record_metadata"],
        stamps=stamps,
        query_start_loc=x["query_start_loc"],
    )
    return records, (storage[own], history.length[own], stamps, retained)


def _fused_packed(x, capacity, history_length, output_norm=None, first_slot=0):
    records, buffers = _packed_records(x, capacity, history_length, first_slot)
    return _verify(x, records, output_norm), buffers


def _assert_same_bytes(got, want):
    """Bitwise equality: unlike ``torch.equal``, -0.0 differs from 0.0."""
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(
        got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)
    )


def _copy_runs(x, runs):
    """Each run's output and buffers; the checkpoint pool must stay as it was.

    ``runs`` yields a function per run; the history the prepare readies fits
    beside the window, so neither it nor the verify writes the pool.
    """
    pool = x["state_pool"].clone()
    results = []
    for run in runs:
        results.append(run())
        _assert_same_bytes(x["state_pool"], pool)
    return results


def _spy_compiles(monkeypatch):
    """The window kernel's compile lookups, one per launch, in launch order.

    Each entry holds the compile's arguments by name and the kernel the real,
    cached ``_compile`` returned, so a test sees which specialization a launch
    ran and whether launches share one.
    """
    compile_kernel = window_kernel._compile
    signature = inspect.signature(compile_kernel)
    launches = []

    def compile_and_record(*args, **kwargs):
        kernel = compile_kernel(*args, **kwargs)
        launches.append((signature.bind(*args, **kwargs).arguments, kernel))
        return kernel

    monkeypatch.setattr(window_kernel, "_compile", compile_and_record)
    return launches


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("window", "capacity", "history_length"),
    [
        *[
            (window, window + 12, length)
            for window in (2, 4, 8)
            for length in (0, 6, 12)
        ],
        # Records 32 on: lane 0's second copy, alone in the last group.
        (4, 37, 33),
    ],
)
def test_record_copies_give_the_same_bits_as_register_loads(
    state_dtype, window, capacity, history_length, monkeypatch
):
    """A 64-row tile that bulk-copies its records stores what one loading them does.

    Batch 4 (12 heads) runs 64-row tiles, padding requests included; the
    prepare leaves at most ``capacity - window`` records, which the longest
    histories fill.
    """
    assert history_length <= capacity - window
    x = _inputs(4, window, seed=300 + window + history_length)
    x["state_pool"] = x["state_pool"].to(state_dtype)
    assert window_kernel._value_tile_rows(4, HEADS, window, DIM) == 64
    norm = _output_norm(x, torch.bfloat16)
    taken = []
    launches = _spy_compiles(monkeypatch)

    def run(copies):
        def verify():
            monkeypatch.setattr(
                window_kernel,
                "_copies_records",
                lambda *_: taken.append(copies) or copies,
            )
            return _fused_packed(x, capacity, history_length, norm)

        return verify

    (out, records), (copied_out, copied_records) = _copy_runs(
        x, [run(False), run(True)]
    )
    # Both runs reached the window kernel, the second with copies.
    assert taken == [False, True]
    # Each launch compiled its choice: alike but for the copies, and not shared.
    (load_args, load_kernel), (copy_args, copy_kernel) = launches
    assert load_args["copy_records"] is False and copy_args["copy_records"] is True
    assert load_args == {**copy_args, "copy_records": False}
    assert copy_kernel is not load_kernel
    _assert_same_bytes(copied_out, out)
    for got, want in zip(copied_records, records, strict=True):
        _assert_same_bytes(got, want)
    # The prepare kept the history for the verify instead of folding it.
    assert (records[1] == history_length).all()


def test_far_record_slots_past_int32_copy_like_near_ones(monkeypatch):
    """Record slots whose offsets pass 2**31 elements copy like near ones.

    The slot stride fits 32 bits; the far slots' offsets do not. Both runs
    bulk-copy the same records on 64-row tiles, so every result must have
    the same bits.
    """
    window, capacity, history_length = 4, 16, 12
    stride = 3 * HEADS * capacity * DIM
    far = 2**31 // stride + 1
    assert stride < 2**31 < far * stride
    x = _inputs(4, window, seed=320)
    assert window_kernel._value_tile_rows(4, HEADS, window, DIM) == 64
    taken = []
    monkeypatch.setattr(
        window_kernel, "_copies_records", lambda *_: taken.append(True) or True
    )
    launches = _spy_compiles(monkeypatch)
    (near_out, near), (far_out, far_buffers) = _copy_runs(
        x,
        [
            lambda first_slot=first_slot: _fused_packed(
                x, capacity, history_length, first_slot=first_slot
            )
            for first_slot in (0, far)
        ],
    )
    assert taken == [True, True]
    # Both launches ran the one record-copy kernel.
    (near_args, near_kernel), (far_args, far_kernel) = launches
    assert near_args["copy_records"] is True and far_args == near_args
    assert far_kernel is near_kernel
    _assert_same_bytes(far_out, near_out)
    for got, want in zip(far_buffers, near, strict=True):
        _assert_same_bytes(got, want)
    assert (near[1] == history_length).all()


def test_only_grids_of_one_cta_per_sm_copy_records_on_64_row_tiles():
    """The copy region can cost a 64-row tile a CTA per SM, so only grids that
    hold one CTA per SM take it, and only for dense records with room beside
    the window that fit a CTA. Real launches check each condition in
    ``test_interleaved_records_run_the_register_load_kernel`` and
    ``test_each_copy_condition_decides_the_compiled_kernel``."""
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    def copies(rows=64, ctas=sms, packed=True, window=4, capacity=16):
        return window_kernel._copies_records(
            rows, ctas, packed, HEADS, window, capacity, True, True, True, True, device
        )

    assert copies()
    assert copies(window=2, capacity=14) and copies(window=8, capacity=20)
    assert not copies(ctas=sms + 1)
    assert not copies(rows=32) and not copies(rows=128)
    assert not copies(packed=False)
    # No records fit beside the window.
    assert not copies(capacity=4)
    # The region of 1020 records exceeds what a CTA may claim.
    assert not copies(capacity=1024)


def test_batches_of_one_bucket_share_one_compiled_kernel(monkeypatch):
    """Compiles happen once per (value tile, grid class) bucket, the record-copy
    bucket included: every other batch of a bucket reuses its kernel.

    The buckets follow from the SM count of the device that runs the test. The
    compile cache starts empty, so no earlier test's kernel can stand in for a
    batch's own. The compile lookup of every launch is observed: 64-row tiles
    of at most one CTA per SM must run the record-copy kernel, which differs
    from the kernel the same launch compiles with its copies turned off; no
    other bucket copies.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    window, capacity = 4, 16
    buckets = _compile_buckets(window, packed=True)
    window_kernel._compile.cache_clear()
    launches = _spy_compiles(monkeypatch)
    compiled = {}

    def run(batch):
        x = _inputs(batch, window, seed=400 + batch)
        assert x["state_indices"].device == device
        launched = len(launches)
        _fused_packed(x, capacity, 4)
        # Each run is one launch of the window kernel.
        assert len(launches) == launched + 1
        return launches[-1]

    for batches in buckets.values():
        compiled[batches[0]] = run(batches[0])
    with assert_no_cute_compile():
        for batches in buckets.values():
            for batch in batches[1:]:
                compiled[batch] = run(batch)
    torch.cuda.synchronize()
    assert sorted(compiled) == list(range(1, 17))
    for (rows, one_to_two, copies), batches in buckets.items():
        arguments, kernel = compiled[batches[0]]
        assert arguments["value_tile_rows"] == rows
        assert (arguments["smem_floor"] > 0) == one_to_two
        assert arguments["copy_records"] is copies
        for batch in batches[1:]:
            assert compiled[batch][0] == arguments
            assert compiled[batch][1] is kernel
    # The same launches with their copies turned off compile another kernel.
    monkeypatch.setattr(window_kernel, "_copies_records", lambda *_: False)
    for (_, _, copies), batches in buckets.items():
        for batch in batches if copies else ():
            arguments, kernel = compiled[batch]
            loads, load_kernel = run(batch)
            assert loads == {**arguments, "copy_records": False}
            assert load_kernel is not kernel
    torch.cuda.synchronize()


def _copy_bucket(window):
    """The batches of 1 to 16 requests whose packed records copy on the test device.

    Skips where no 64-row grid of them holds at most one CTA per SM.
    """
    for (_, _, copies), batches in _compile_buckets(window, packed=True).items():
        if copies:
            return batches
    pytest.skip("no 64-row grid of 1 to 16 requests fits one CTA per SM here")


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_interleaved_records_run_the_register_load_kernel(state_dtype, monkeypatch):
    """Only packed records take the copies.

    Every batch of the device's record-copy bucket verifies the same requests
    and accepted history twice: from packed records, then from ``_records``'
    interleaved views. The two launches compile alike but for the copies,
    only the packed one copies, and both store the same bits.
    """
    window, capacity, history_length = 4, 16, 6
    batches = _copy_bucket(window)
    launches = _spy_compiles(monkeypatch)
    for batch in batches:
        x = _inputs(batch, window, seed=500 + batch)
        x["state_pool"] = x["state_pool"].to(state_dtype)
        packed = _packed_records(x, capacity, history_length)[0]
        interleaved = _records(
            x, original=True, capacity=capacity, history_length=history_length
        )[0]
        assert packed.history.packed_record_width == DIM
        assert interleaved.history.packed_record_width is None
        launched = len(launches)
        packed_out, interleaved_out = _copy_runs(
            x,
            [
                lambda records=records: _verify(x, records, None)
                for records in (packed, interleaved)
            ],
        )
        (copy_args, copy_kernel), (load_args, load_kernel) = launches[launched:]
        assert copy_args["copy_records"] is True
        assert load_args == {**copy_args, "copy_records": False}
        assert load_kernel is not copy_kernel
        _assert_same_bytes(interleaved_out, packed_out)
        for field in ("correction", "key", "decay", "length"):
            _assert_same_bytes(
                getattr(interleaved.history, field), getattr(packed.history, field)
            )
        _assert_same_bytes(interleaved.stamps, packed.stamps)
        _assert_same_bytes(interleaved.original_qkv, packed.original_qkv)
        # The prepare kept the accepted history for the verify.
        assert (packed.history.length == history_length).all()


@pytest.mark.parametrize(
    "condition", ["64-row tile", "room for records", "one CTA per SM", "fits a CTA"]
)
def test_each_copy_condition_decides_the_compiled_kernel(condition, monkeypatch):
    """Each other condition ``_copies_records`` checks, through real launches.

    A launch of packed records that copies is paired with launches that break
    only that condition. Their compile lookups differ from its lookup in the
    condition's own arguments and the copies alone, and only it gets the
    record-copy kernel:

    - 64-row tile: the same requests on forced 64-, 32- and 128-row tiles, the
      most requests up to 3 whose 32-row grid holds one CTA per SM; every
      tile stores the same bits;
    - room for records: a capacity of the window plus one against the window;
    - one CTA per SM: the record-copy bucket's last batch against the next;
    - fits a CTA: the largest capacity whose copying CTA fits the device's
      opt-in shared memory against the next.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    window, capacity, history_length = 4, 16, 4
    launches = _spy_compiles(monkeypatch)

    def launch(x, capacity, history_length, rows=None):
        if rows is not None:
            monkeypatch.setattr(window_kernel, "_value_tile_rows", lambda *_: rows)
        launched = len(launches)
        result = _copy_runs(x, [lambda: _fused_packed(x, capacity, history_length)])[0]
        assert len(launches) == launched + 1
        return launches[-1], result

    if condition == "64-row tile":
        batch = min(3, props.multi_processor_count // (HEADS * DIM // 32))
        if batch == 0:
            pytest.skip("no 32-row grid fits one CTA per SM here")
        x = _inputs(batch, window, seed=600)
        copied = launch(x, capacity, history_length, rows=64)
        others = [launch(x, capacity, history_length, rows=rows) for rows in (32, 128)]
        changed = {"value_tile_rows", "copy_records"}
        out, buffers = copied[1]
        for _, (other_out, other_buffers) in others:
            _assert_same_bytes(other_out, out)
            for got, want in zip(other_buffers, buffers, strict=True):
                _assert_same_bytes(got, want)
    elif condition == "room for records":
        x = _inputs(_copy_bucket(window)[0], window, seed=610)
        copied = launch(x, window + 1, 1)
        others = [launch(x, window, 0)]
        changed = {"capacity", "copy_records"}
    elif condition == "one CTA per SM":
        buckets = _compile_buckets(window, packed=True)
        last = _copy_bucket(window)[-1]
        if last == 16:
            pytest.skip("every 64-row grid of up to 16 requests fits one CTA per SM")
        (rows, one_to_two, copies), _ = next(
            (key, batches) for key, batches in buckets.items() if last + 1 in batches
        )
        assert rows == 64 and not copies
        copied = launch(_inputs(last, window, seed=620), capacity, history_length)
        others = [launch(_inputs(last + 1, window, seed=621), capacity, history_length)]
        changed = {"copy_records", "smem_floor"} if one_to_two else {"copy_records"}
    else:
        x = _inputs(_copy_bucket(window)[0], window, seed=630)
        factored = window * abs(x["lower_bound"]) <= window_kernel.FACTORED_DECAY

        def fits(capacity):
            return (
                window_kernel.FusedVerifyWindow(
                    HEADS, window, 64, capacity, True, factored, True, False, 0, True
                ).smem_bytes
                <= props.shared_memory_per_block_optin
            )

        largest = capacity
        assert fits(largest)
        while fits(largest + 1):
            largest += 1
            if largest == 1024:
                pytest.skip("a CTA here fits the region of 1020 records")
        copied = launch(x, largest, history_length)
        others = [launch(x, largest + 1, history_length)]
        changed = {"capacity", "copy_records"}
        # The footprint above is that of the launched configuration.
        arguments = copied[0][0]
        assert arguments["bounded_gate"] and arguments["factored"] == factored
        assert arguments["has_metadata"] and arguments["norm_dtype"] is None
    torch.cuda.synchronize()
    (copy_args, copy_kernel), _ = copied
    assert copy_args["copy_records"] is True
    for (arguments, kernel), _ in others:
        assert arguments["copy_records"] is False
        assert {name for name in arguments if arguments[name] != copy_args[name]} == (
            changed
        )
        assert kernel is not copy_kernel
