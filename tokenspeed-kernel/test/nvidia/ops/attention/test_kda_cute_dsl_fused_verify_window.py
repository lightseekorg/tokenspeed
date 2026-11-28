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
    two CTAs per SM (the shared-memory floor) and whether it copies records:
    packed records with room beside the window copy on 64-row grids of at
    most one CTA per SM; ``_fused``'s interleaved records never copy.
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


@pytest.mark.parametrize("packed", [False, True])
def test_serving_batches_reuse_one_compiled_kernel(packed, monkeypatch):
    # One compile per value tile, spread on the test device's SMs and record
    # copy: the first batch of each bucket warms it and every other batch
    # reuses it. The compile cache starts empty, so no earlier test's kernel
    # can stand in for a batch's own; every launch asks for its bucket's copy.
    compile_kernel = window_kernel._compile
    compile_kernel.cache_clear()
    copy_records = []
    monkeypatch.setattr(
        window_kernel,
        "_compile",
        lambda *args: copy_records.append(args[-1]) or compile_kernel(*args),
    )
    buckets = _compile_buckets(8, packed=packed)
    assert any(copies for *_, copies in buckets) == packed

    def fused(batch, seed, copies):
        x = _inputs(batch, 8, seed=seed)
        if packed:  # room for 12 records beside the window
            _fused_packed(x, 20, 0)
        else:
            _fused(x, 8)
        assert copy_records.pop() is copies

    for (*_, copies), batches in buckets.items():
        fused(batches[0], 50 + batches[0], copies)
    assert compile_kernel.cache_info().currsize == len(buckets)
    with assert_no_cute_compile():
        for (*_, copies), batches in buckets.items():
            for batch in batches[1:]:
                fused(batch, 60 + batch, copies)
    torch.cuda.synchronize()


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
    verify applies at most ``capacity - window`` records, which the longest
    histories fill.
    """
    assert history_length <= capacity - window
    x = _inputs(4, window, seed=300 + window + history_length)
    x["state_pool"] = x["state_pool"].to(state_dtype)
    assert window_kernel._value_tile_rows(4, HEADS, window, DIM) == 64
    norm = _output_norm(x, torch.bfloat16)
    taken = []

    def verify(copies):
        monkeypatch.setattr(
            window_kernel, "_copies_records", lambda *_: taken.append(copies) or copies
        )
        return _fused_packed(x, capacity, history_length, norm)

    (out, records), (copied_out, copied_records) = _copy_runs(
        x, [lambda copies=copies: verify(copies) for copies in (False, True)]
    )
    # Both runs reached the window kernel, the second with copies.
    assert taken == [False, True]
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
    _assert_same_bytes(far_out, near_out)
    for got, want in zip(far_buffers, near, strict=True):
        _assert_same_bytes(got, want)
    assert (near[1] == history_length).all()


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("window", [2, 4, 8])
@pytest.mark.parametrize("full", [False, True])
def test_record_copies_read_none_of_an_overflowing_history(
    state_dtype, window, full, monkeypatch
):
    """A count past ``capacity - window`` folds into the page and copies nothing.

    The prepare applies every accepted record in the live page and leaves the
    count as it was, so the verify applies none: copying the records, loading
    them, or starting from that page with no records stores the same bits.
    """
    capacity = window + 12
    # One record past what fits beside the window, or every row.
    history_length = capacity if full else capacity - window + 1
    x = _inputs(4, window, seed=340 + window + history_length)
    x["state_pool"] = x["state_pool"].to(state_dtype)
    assert window_kernel._value_tile_rows(4, HEADS, window, DIM) == 64
    norm = _output_norm(x, torch.bfloat16)
    pool = x["state_pool"].clone()
    folded = pool.clone()
    records, _ = _packed_records(dict(x, state_pool=folded), capacity, history_length)
    prepare_history(
        folded,
        records.history,
        x["state_indices"],
        x["query_start_loc"],
        draft_token_num=window,
    )
    assert not torch.equal(folded, pool)
    taken = []

    def verify(copies, start, length):
        monkeypatch.setattr(
            window_kernel, "_copies_records", lambda *_: taken.append(copies) or copies
        )
        run = dict(x, state_pool=start.clone())
        out, buffers = _fused_packed(run, capacity, length, norm)
        return out, buffers, run["state_pool"]

    loaded = verify(False, pool, history_length)
    copied = verify(True, pool, history_length)
    empty = verify(True, folded, 0)
    assert taken == [False, True, True]
    for got, want in zip(copied, loaded, strict=True):
        if isinstance(got, tuple):
            for field, peer in zip(got, want, strict=True):
                _assert_same_bytes(field, peer)
        else:
            _assert_same_bytes(got, want)
    _assert_same_bytes(copied[2], folded)
    _assert_same_bytes(copied[0], empty[0])
    _assert_same_bytes(copied[2], empty[2])
    # Live requests record the window from position 0, as without history;
    # request r owns slot r, and padding requests record nothing.
    live = (x["state_indices"] > 0).nonzero().flatten()
    assert live.numel() == 2
    window_rows = (live, slice(None), slice(None), slice(0, window))
    _assert_same_bytes(copied[1][0][window_rows], empty[1][0][window_rows])
    _assert_same_bytes(copied[1][2], empty[1][2])
    # The prepare only read the count.
    assert (copied[1][1] == history_length).all()


def test_only_grids_of_one_cta_per_sm_copy_records_on_64_row_tiles():
    """The copy region can cost a 64-row tile a CTA per SM, so only grids that
    hold one CTA per SM take it, and only for packed records with room beside
    the window whose region fits the device's opt-in shared memory."""
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    def copies(rows=64, ctas=sms, packed=True, window=4, capacity=16):
        return window_kernel._copies_records(
            rows, ctas, packed, HEADS, window, capacity, True, True, True, True, device
        )

    assert copies()
    assert copies(window=2, capacity=14) and copies(window=8, capacity=20)
    assert not copies(rows=32) and not copies(rows=128)  # 64-row tiles only
    # No records fit beside the window.
    assert not copies(capacity=4)
    assert not copies(packed=False)  # interleaved records keep register loads
    assert not copies(ctas=sms + 1)  # more than one CTA per SM
    # The region of 1020 records exceeds what a CTA may claim.
    assert not copies(capacity=1024)
