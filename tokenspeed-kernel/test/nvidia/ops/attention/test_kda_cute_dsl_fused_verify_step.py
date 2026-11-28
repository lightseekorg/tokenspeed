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

"""CuTe DSL one-token verify with fused producers against the split producers.

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

import dataclasses

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
from kda_retained_reference import (  # noqa: E402
    prepare_history,
    replay_length,
    snapshot,
)
from tokenspeed_kernel.ops.activation.triton import (  # noqa: E402
    rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel.ops.attention.kda import (  # noqa: E402
    KdaOutputNorm,
    KdaPreconvRing,
    KdaRetainedHistory,
    KdaRetainedRecords,
    kda_fused_paged_verify,
    kda_fused_paged_verify_fuses_producers,
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


def _inputs(batch, *, seed, lower_bound=-5.0, metadata=True, f_a_pitch=0):
    """Request 1 has padding page 0 and request 2 page -1 (zero history)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    projection = randn(batch, PROJECTION + f_a_pitch).bfloat16()
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
        query_start_loc=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        record_metadata=record_metadata,
        slots=slots,
        lower_bound=lower_bound,
    )


def _records(x, *, capacity=1, correction_stride=1):
    """Sentinel-filled request records, with padding around each slot."""
    slots = x["slots"]
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
        replay_length(slots),
        x["state_indices"],
        torch.arange(x["state_indices"].numel(), device="cuda", dtype=torch.int32),
    )
    stamps = torch.full(
        (x["slots"], capacity, 2), STAMP_FILL, device="cuda", dtype=torch.int64
    )
    # Each slot's pre-convolution ring, padded; the prepare seeds rows 0..2.
    retained = torch.full(
        (slots, 4, 3 * WIDTH + 8), ORIGINAL_FILL, device="cuda"
    ).bfloat16()
    records = KdaRetainedRecords(
        preconv=KdaPreconvRing(
            retained[..., : 3 * WIDTH],
            snapshot(history, x["state_indices"].numel())[:, 1],
        ),
        history=history,
        metadata=x["record_metadata"],
        stamps=stamps if x["record_metadata"] is not None else None,
        query_start_loc=x["query_start_loc"],
    )
    return records, (correction_rows, key_gate_rows, stamps, retained)


def _prepare(x, records):
    # Serving readies the history and the ring before every verify round.
    prepare_history(
        x["state_pool"],
        records.history,
        x["state_indices"],
        x["query_start_loc"],
        draft_token_num=1,
        conv=x["conv_states"],
        preconv=records.preconv,
    )


def _verify(x, records, output_norm, *, override=None, prepare=True, **producers):
    if prepare:
        _prepare(x, records)
    return kda_fused_paged_verify(
        x["mixed_qkv"],
        x["conv_weights"],
        x["f_a_out"],
        x["f_b_weight"],
        x["beta_logits"],
        x["A_log"],
        x["dt_bias"],
        state_pool=x["state_pool"],
        read_indices=x["state_indices"],
        num_heads=HEADS,
        head_dim=DIM,
        draft_token_num=1,
        lower_bound=x["lower_bound"],
        recurrent_layout="v_major",
        retained_records=records,
        output_norm=output_norm,
        override=override,
        solution=None,
        **producers,
    )


def _split(x, prepared=None):
    """The serving split producers, then the registered record verify."""
    records, buffers = _records(x) if prepared is None else prepared
    _prepare(x, records)
    conv_qkv = fused_kda_verify_conv_update(
        x["mixed_qkv"],
        x["conv_weights"],
        x["state_indices"],
        live_indices=records.history.live_state_indices,
        preconv=records.preconv,
        record_slots=records.history.record_slots,
        page_count=x["state_pool"].shape[0],
        num_heads=HEADS,
        head_dim=DIM,
        draft_token_num=1,
        query_start_loc=x["query_start_loc"],
    )
    g_raw = F.linear(x["f_a_out"], x["f_b_weight"])
    records = dataclasses.replace(records, preconv=None)
    out = _verify(x, records, None, prepare=False, g_raw=g_raw, conv_qkv=conv_qkv)
    return out, buffers


def _fused(x, output_norm=None):
    records, buffers = _records(x)
    return _verify(x, records, output_norm), buffers


def _assert_matches(expected, actual):
    out_ref, (corr_ref, kg_ref, stamps_ref, kept_ref) = expected
    out, (corr, kg, stamps, kept) = actual
    assert torch.equal(kept, kept_ref)
    assert torch.equal(stamps, stamps_ref)
    key_cols = torch.cat(
        [torch.arange(h * 2 * DIM, h * 2 * DIM + DIM) for h in range(HEADS)]
    ).cuda()
    torch.testing.assert_close(
        kg[:, key_cols], kg_ref[:, key_cols], atol=1e-7, rtol=2e-6
    )
    assert torch.equal(kg[:, 2 * WIDTH :], kg_ref[:, 2 * WIDTH :])
    gate_cols = key_cols + DIM
    gates, gates_ref = kg[:, gate_cols].float(), kg_ref[:, gate_cols].float()
    # One BF16 step of the larger gate, plus a floor for sums that cancel to near zero.
    step = torch.maximum(gates.abs(), gates_ref.abs()) * 2.0**-7
    step = step + gates_ref.abs().max() * 2.0**-16
    assert ((gates - gates_ref).abs() <= step).all()
    assert torch.equal(corr[:, WIDTH:], corr_ref[:, WIDTH:])
    written = corr_ref != CORRECTION_FILL
    assert torch.equal(written, corr != CORRECTION_FILL)
    scale = corr_ref[written].abs().max() if written.any() else 1.0
    assert ((corr - corr_ref).abs() <= 1e-4 * scale).all()
    out_ref, out = out_ref.float(), out.float()
    scale = out_ref.abs().max()
    assert ((out - out_ref).abs() <= 1e-3 * scale + out_ref.abs() * 2.0**-7).all()


@pytest.mark.parametrize("batch", [1, 2, 3, 5, 8, 16, 33, 64])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_fused_step_matches_split_producers(batch, lower_bound):
    x = _inputs(batch, seed=batch, lower_bound=lower_bound)
    state, history = x["state_pool"].clone(), x["conv_states"].clone()
    _assert_matches(_split(x), _fused(x))
    assert torch.equal(x["state_pool"], state)
    assert torch.equal(x["conv_states"], history)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("raw_gate_only", [False, True])
def test_retained_prefix_append_preserves_base_and_publishes_on_commit(
    state_dtype, raw_gate_only
):
    from kda_retained_reference import check_fused_retained_append

    x = _inputs(2, seed=140)
    x["state_pool"] = x["state_pool"].to(state_dtype)
    records = _records(x, capacity=8)[0]
    expected = _records(x, capacity=8)[0]
    check_fused_retained_append(
        x, records, expected, _verify, width=1, raw_gate_only=raw_gate_only
    )


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("record_layout", ["interleaved", "packed", "unaligned"])
def test_fused_step_matches_split_with_mixed_retained_lengths(
    state_dtype, record_layout
):
    x = _inputs(13, seed=147)
    state = x["state_pool"].to(state_dtype)
    if state_dtype == torch.bfloat16:
        # Four extra BF16 values alternate aligned and 8-byte-offset head tiles.
        # Verify both copy paths against the same split reference.
        padded = state.new_empty((*state.shape[:2], DIM * DIM + 4))
        state = padded[..., : DIM * DIM].view_as(state).copy_(state)
    x["state_pool"] = state
    x["state_indices"].copy_(
        torch.tensor([1, 0, -1, *range(2, 12)], device="cuda", dtype=torch.int32)
    )
    records, _ = _records(x, capacity=32)
    reference, buffers = _records(x, capacity=32)
    live = x["state_indices"].clone()
    live[-1] = 12
    records = dataclasses.replace(
        records, history=dataclasses.replace(records.history, live_state_indices=live)
    )
    reference = dataclasses.replace(
        reference,
        history=dataclasses.replace(reference.history, live_state_indices=live),
    )
    if record_layout != "interleaved":
        history = records.history
        key = history.key.contiguous()
        if record_layout == "unaligned":
            # A contiguous K view can still start four bytes into its storage.
            key = key.new_empty(key.numel() + 1)[1:].view_as(key)
        records = dataclasses.replace(
            records,
            history=dataclasses.replace(
                history,
                correction=history.correction.contiguous(),
                key=key,
                decay=history.decay.contiguous(),
            ),
        )
    history = records.history
    # Row r owns slot r. Padding rows 1 and 2 keep their stale counts; row 11
    # overflows the capacity and row 12 copies its source to another page.
    history.length[:13, 0] = torch.tensor(
        [0, 5, 7, 1, 2, 3, 8, 9, 15, 16, 17, 32, 8], device="cuda"
    )
    gen = torch.Generator(device="cuda").manual_seed(149)
    for field, peer, scale, bias in (
        (history.correction, reference.history.correction, 0.02, 0.0),
        (history.key, reference.history.key, 0.08, 0.0),
        (history.decay, reference.history.decay, 0.2, 0.7),
    ):
        field.copy_(
            torch.rand(field.shape, device="cuda", generator=gen) * scale + bias
        )
        peer.copy_(field)
    reference.history.length.copy_(history.length)
    ref_x = dict(x, state_pool=x["state_pool"].clone())
    expected, _ = _split(ref_x, (reference, buffers))
    norm = None
    if record_layout == "packed":
        # Exercise the shared reductions used by retained replay and gated RMSNorm
        # together, including padded requests and both sides of the length cutoff.
        norm = _output_norm(x, torch.float32)
        expected = rmsnorm_gated_sigmoid(
            expected.view(-1, WIDTH),
            norm.gate,
            norm.weight,
            norm.eps,
            HEADS,
            DIM,
            enable_pdl=False,
        )
    actual = _verify(x, records, norm, override="cute_dsl_kda_fused_paged_verify_step")
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-4, rtol=1e-2)
    for field, peer in (
        (history.correction, reference.history.correction),
        (history.key, reference.history.key),
        (history.decay, reference.history.decay),
    ):
        torch.testing.assert_close(field, peer, atol=5e-5, rtol=2e-3)
    assert torch.equal(records.stamps, reference.stamps)
    assert torch.equal(records.preconv.rows, buffers[3][..., : 3 * WIDTH])
    # The prepare only reads the counts: row 11's overflowing count folded
    # into its page and the verify read none of it; row 12's copy keeps its 8.
    counts = [0, 5, 7, 1, 2, 3, 8, 9, 15, 16, 17, 32, 8]
    assert history.length[:13, 0].tolist() == counts
    assert torch.equal(history.length, reference.history.length)
    torch.testing.assert_close(
        x["state_pool"].float(), ref_x["state_pool"].float(), atol=2e-6, rtol=2e-5
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 << 30,
    reason="ring slots past 2**31 elements need 12 GiB free",
)
def test_far_ring_slots_past_int32_verify_like_near_ones():
    """Ring slot offsets past 2**31 elements verify exactly like near slots.

    Rows 0 and 1 own slots 3 and 2, whose rows in the far ring start past
    2**31 elements; both rings start empty and the prepare seeds them.
    """
    x = _inputs(2, seed=71)
    ring_rows, slots = 4, x["slots"]

    def run(ring):
        records, buffers = _records(x)
        history = dataclasses.replace(
            records.history,
            record_slots=torch.tensor([3, 2], device="cuda", dtype=torch.int32),
        )
        records = dataclasses.replace(
            records,
            history=history,
            preconv=KdaPreconvRing(ring, records.preconv.heads),
        )
        return _verify(x, records, None), buffers

    near_ring = torch.zeros(slots, ring_rows, 3 * WIDTH, device="cuda").bfloat16()
    stride = (1 << 30) + ring_rows * 3 * WIDTH
    storage = torch.zeros(
        (slots - 1) * stride + ring_rows * 3 * WIDTH,
        device="cuda",
        dtype=torch.bfloat16,
    )
    far_ring = storage.as_strided((slots, ring_rows, 3 * WIDTH), (stride, 3 * WIDTH, 1))
    assert 2 * stride > 2**31
    near, far = run(near_ring), run(far_ring)
    assert torch.equal(far[0], near[0])
    for got, want in zip(far[1][:3], near[1][:3]):
        assert torch.equal(got, want)
    assert torch.equal(far_ring[2:], near_ring[2:])
    # Each row's ring holds its source taps and its candidate at row 3.
    for row, slot in enumerate((3, 2)):
        page = int(x["state_indices"][row])
        assert torch.equal(far_ring[slot, :3].T, x["conv_states"][page, : 3 * WIDTH])
        assert torch.equal(far_ring[slot, 3], x["mixed_qkv"][row])


def test_far_pages_of_a_pool_past_int32_verify_like_near_ones():
    """Page offsets past 2**31 elements verify exactly like near pages.

    The last pages of the large pool hold the same contents as the first pages
    of a small pool, so outputs and records must be equal.
    """
    x = _inputs(2, seed=70)
    near = _fused(x)
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
        )
    )
    assert torch.equal(far[0], near[0])
    for got, want in zip(far[1], near[1]):
        assert torch.equal(got, want)


def test_fused_step_without_metadata_records_by_request():
    x = _inputs(4, seed=40, metadata=False)
    _assert_matches(_split(x), _fused(x))


@pytest.mark.parametrize("strided", ["metadata", "stamps"])
def test_strided_replay_records_are_rejected(strided):
    """The kernel reads the record metadata and stamps as dense INT64 rows."""
    x = _inputs(3, seed=42)
    records = _records(x)[0]
    dense = records.metadata if strided == "metadata" else records.stamps
    wide = dense.new_zeros(*dense.shape[:-1], 2 * dense.shape[-1])
    view = wide[..., ::2]
    view.copy_(dense)
    assert not view.is_contiguous() and torch.equal(view, dense)
    records = dataclasses.replace(records, **{strided: view})
    with pytest.raises(ValueError, match="contiguous INT64"):
        _verify(x, records, None)


@pytest.mark.parametrize(
    "layout", ["f_a rows", "conv weights", "request indices", "correction channels"]
)
def test_strided_or_unaligned_inputs_match_split_producers(layout):
    x = _inputs(3, seed=41, f_a_pitch=int(layout == "f_a rows"))
    if layout == "conv weights":
        weights = x["conv_weights"]
        shifted = torch.empty(weights.numel() + 1, device="cuda", dtype=weights.dtype)
        x["conv_weights"] = shifted[1:].view_as(weights).copy_(weights)
    if layout == "request indices":
        for name in ("state_indices", "query_start_loc"):
            values = x[name]
            interleaved = values.new_zeros(2 * values.numel())
            x[name] = interleaved[::2].copy_(values)
    correction_stride = 2 if layout == "correction channels" else 1
    expected = _split(x, _records(x, correction_stride=correction_stride))
    records, buffers = _records(x, correction_stride=correction_stride)
    actual = _verify(x, records, None), buffers
    # The fallback runs the reference producers, so every result is exact.
    assert torch.equal(expected[0], actual[0])
    for ref, got in zip(expected[1], actual[1]):
        assert torch.equal(ref, got)


def test_one_token_windows_fuse_their_producers():
    assert kda_fused_paged_verify_fuses_producers(
        torch.bfloat16,
        recurrent_layout="v_major",
        num_heads=HEADS,
        head_dim=DIM,
        draft_token_num=1,
        batch_size=64,
    )


def test_serving_batches_reuse_one_compiled_kernel():
    for batch in (1, 9):
        _fused(_inputs(batch, seed=50 + batch))
    with assert_no_cute_compile():
        for batch in (2, 7, 64):
            _fused(_inputs(batch, seed=60 + batch))
    torch.cuda.synchronize()


def _output_norm(x, weight_dtype):
    gen = torch.Generator(device="cuda").manual_seed(91)
    weight = (1 + 0.1 * torch.randn(DIM, device="cuda", generator=gen)).to(weight_dtype)
    return KdaOutputNorm(x["out_gate"], weight, 1e-5)


@pytest.mark.parametrize("batch", [1, 3, 8, 64])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_fused_output_norm_matches_the_norm_kernel(batch, weight_dtype):
    """The epilogue stores the norm kernel's result on the BF16 output.

    Only the order of the sum of squares differs, so an element may move by one
    BF16 step; the records do not change at all.
    """
    x = _inputs(batch, seed=80 + batch)
    norm = _output_norm(x, weight_dtype)
    plain, plain_records = _fused(x)
    normed, normed_records = _fused(x, norm)
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
        assert not normed[1:3].any()


def test_unaligned_inputs_apply_the_output_norm_after_the_split_producers():
    x = _inputs(3, seed=43, f_a_pitch=1)
    norm = _output_norm(x, torch.bfloat16)
    plain = _split(x)[0]
    normed = _fused(x, norm)[0]
    expected = rmsnorm_gated_sigmoid(
        plain.view(3, WIDTH),
        norm.gate,
        norm.weight,
        norm.eps,
        HEADS,
        DIM,
        enable_pdl=False,
    )
    assert torch.equal(normed, expected)


def test_misshaped_output_norm_is_rejected():
    x = _inputs(3, seed=44)
    norm = _output_norm(x, torch.bfloat16)
    with pytest.raises(ValueError, match="output norm"):
        _fused(x, KdaOutputNorm(norm.gate, norm.weight[:64], norm.eps))


def test_a_page_past_the_pool_verifies_as_padding():
    """A page past the checkpoint pool is padding, exactly like page -1: its
    checkpoint and convolution history are not read, it records nothing, and
    every result matches the page -1 run bit for bit."""
    runs = []
    for past_pool in (False, True):
        # _inputs draws its pages from the global generator.
        torch.manual_seed(211)
        x = _inputs(5, seed=211)
        if past_pool:
            x["state_indices"][2] = x["state_pool"].shape[0]
        state, history = x["state_pool"].clone(), x["conv_states"].clone()
        runs.append(_fused(x))
        assert torch.equal(x["state_pool"], state)
        assert torch.equal(x["conv_states"], history)
    (out, buffers), (past_out, past_buffers) = runs
    assert torch.equal(past_out, out)
    for past, padded in zip(past_buffers, buffers, strict=True):
        assert torch.equal(past, padded)


def _share_live_slot_zero(records):
    """Rows 1 to 3 share live row 0's record slot 0, with ring head 0.

    Rows 1 and 2 have pages 0 and -1; row 3 has a source page but a masked
    live page. Head 0 points their taps at slot 0's rows 1..3, which row 0
    reads and writes (head 3). Row 4 keeps its own slot.
    """
    live = records.history.live_state_indices.clone()
    live[3] = 0
    slots = records.history.record_slots.clone()
    slots[:4] = 0
    records = dataclasses.replace(
        records,
        history=dataclasses.replace(
            records.history, live_state_indices=live, record_slots=slots
        ),
    )
    return records


def test_padding_rows_beside_live_slot_zero_touch_no_ring_row():
    """Padding rows sharing live slot 0 read a zero history and write no row.

    The fused kernel and the split producers give the same bits; slot 0 holds
    exactly row 0's taps and candidate, slots of no taking-part row keep their
    fill, and the padding rows' outputs are zero.
    """
    x = _inputs(5, seed=233)
    runs = []
    for fused in (False, True):
        records, buffers = _records(x)
        records = _share_live_slot_zero(records)
        _prepare(x, records)
        records.preconv.heads[1:4] = 0
        if fused:
            out = _verify(
                x,
                records,
                None,
                override="cute_dsl_kda_fused_paged_verify_step",
                prepare=False,
            )
        else:
            conv_qkv = fused_kda_verify_conv_update(
                x["mixed_qkv"],
                x["conv_weights"],
                x["state_indices"],
                live_indices=records.history.live_state_indices,
                preconv=records.preconv,
                record_slots=records.history.record_slots,
                page_count=x["state_pool"].shape[0],
                num_heads=HEADS,
                head_dim=DIM,
                draft_token_num=1,
                query_start_loc=x["query_start_loc"],
            )
            out = _verify(
                x,
                dataclasses.replace(records, preconv=None),
                None,
                prepare=False,
                g_raw=F.linear(x["f_a_out"], x["f_b_weight"]),
                conv_qkv=conv_qkv,
            )
        runs.append((out, buffers))
    _assert_matches(*runs)
    out, (_, _, _, ring) = runs[1]
    page = int(x["state_indices"][0])
    assert torch.equal(ring[0, :3, : 3 * WIDTH].T, x["conv_states"][page, : 3 * WIDTH])
    assert torch.equal(ring[0, 3, : 3 * WIDTH], x["mixed_qkv"][0])
    assert (ring[1:4] == ORIGINAL_FILL).all()
    assert torch.equal(ring[4, 3, : 3 * WIDTH], x["mixed_qkv"][4])
    assert (out[0, 1:4] == 0).all()


def test_an_empty_ring_verifies_padding_without_reading_it():
    """With no record slot every row is padding: zero outputs, no ring access."""
    x = dict(_inputs(2, seed=239, metadata=False), slots=0)
    records, _ = _records(x)
    assert records.preconv.rows.numel() == 0
    out = _verify(
        x, records, None, override="cute_dsl_kda_fused_paged_verify_step", prepare=False
    )
    torch.cuda.synchronize()
    assert (out == 0).all()


def test_a_ring_sharing_the_projection_storage_is_rejected():
    """The fused entry rejects a ring inside the projection's storage before launch."""
    x = _inputs(3, seed=241)
    records, _ = _records(x)
    rows, ring = x["mixed_qkv"].shape[0], records.preconv.rows
    arena = torch.zeros(
        rows * 3 * WIDTH + ring.numel(), device="cuda", dtype=torch.bfloat16
    )
    mixed = arena[: rows * 3 * WIDTH].view(rows, 3 * WIDTH)
    mixed.copy_(x["mixed_qkv"])
    shared = arena[rows * 3 * WIDTH :].view(ring.shape)
    records = dataclasses.replace(
        records, preconv=KdaPreconvRing(shared, records.preconv.heads)
    )
    before = arena.clone()
    with pytest.raises(ValueError, match="storage separate"):
        _verify(
            dict(x, mixed_qkv=mixed),
            records,
            None,
            override="cute_dsl_kda_fused_paged_verify_step",
            prepare=False,
        )
    torch.cuda.synchronize()
    assert torch.equal(arena, before)


def _checkpoint_ring(x, records):
    """Records whose ring is a view, in the ring's shape, of a live checkpoint page."""
    page = int(x["state_indices"][0])
    rows = records.preconv.rows
    inside = x["state_pool"][page].flatten()[: rows.numel()].view(rows.shape)
    return dataclasses.replace(
        records, preconv=KdaPreconvRing(inside, records.preconv.heads)
    )


@pytest.mark.parametrize("route", ["fused", "split fallback", "triton split"])
def test_a_ring_viewing_the_checkpoint_is_rejected_before_any_launch(
    route, monkeypatch
):
    """Every verify route rejects a ring inside a live checkpoint page.

    The ring is a differently shaped view of a BF16 page the verify reads,
    so its candidate stores would overwrite that checkpoint. The fused route
    must reject it itself rather than fall back; with a raw gate the CuTe
    entry runs the split producers, and the Triton split verify runs its own
    producer. Each raises before any launch, and the pool keeps its bits.
    """
    from tokenspeed_kernel.ops.attention.kda import cute_dsl
    from tokenspeed_kernel.ops.attention.kda.triton import (
        triton_nvidia_kda_fused_paged_verify_split,
    )

    x = _inputs(3, seed=251)
    x["state_pool"] = x["state_pool"].bfloat16()
    records, _ = _records(x)
    records = _checkpoint_ring(x, records)
    before = x["state_pool"].clone()
    producers = {}
    if route == "fused":

        def fallback(*args, **kwargs):
            raise AssertionError("the fused route fell back to the split producers")

        monkeypatch.setattr(cute_dsl, "_split_producer_verify", fallback)
    else:
        producers["g_raw"] = F.linear(x["f_a_out"], x["f_b_weight"])
    with pytest.raises(ValueError, match="storage separate"):
        if route == "triton split":
            triton_nvidia_kda_fused_paged_verify_split(
                x["mixed_qkv"],
                x["conv_weights"],
                x["f_a_out"],
                x["f_b_weight"],
                x["beta_logits"],
                x["A_log"],
                x["dt_bias"],
                state_pool=x["state_pool"],
                read_indices=x["state_indices"],
                num_heads=HEADS,
                head_dim=DIM,
                draft_token_num=1,
                lower_bound=x["lower_bound"],
                retained_records=records,
                output_norm=None,
                **producers,
            )
        else:
            _verify(
                x,
                records,
                None,
                override="cute_dsl_kda_fused_paged_verify_step",
                prepare=False,
                **producers,
            )
    torch.cuda.synchronize()
    assert torch.equal(x["state_pool"], before)


@pytest.mark.parametrize("backing", ["ring", "own"])
@pytest.mark.parametrize(
    "case",
    ["raw gate, split fallback", "raw gate, triton split", "gate input", "conv bias"],
)
def test_an_empty_required_operand_is_rejected_before_any_producer(case, backing):
    """An active verify rejects an empty gate, gate input or bias before any launch.

    Request 0 reads a live page, so its convolution would store its
    candidate row in the ring. A zero-element raw gate (to the CuTe entry,
    which runs the split producers, or to the Triton split verify), gate
    projection input (which the fused kernel cannot address, so the CuTe
    entry falls back to the split producers) or convolution bias, viewing
    the ring's allocation or its own, raises ValueError before the
    convolution producer starts. The whole ring allocation, the checkpoint
    and the records keep their bits.
    """
    from tokenspeed_kernel.ops.attention.kda.triton import (
        triton_nvidia_kda_fused_paged_verify_split,
    )

    x = _inputs(3, seed=263)
    records, buffers = _records(x)
    _prepare(x, records)
    ring = buffers[3]
    if backing == "ring":
        empty = ring.flatten()[:0]
    else:
        empty = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    watched = (ring, x["state_pool"], *buffers[:3])
    torch.cuda.synchronize()
    before = [value.clone() for value in watched]
    with pytest.raises(ValueError):
        if case == "conv bias":
            fused_kda_verify_conv_update(
                x["mixed_qkv"],
                x["conv_weights"],
                x["state_indices"],
                live_indices=records.history.live_state_indices,
                preconv=records.preconv,
                record_slots=records.history.record_slots,
                page_count=x["state_pool"].shape[0],
                num_heads=HEADS,
                head_dim=DIM,
                draft_token_num=1,
                query_start_loc=x["query_start_loc"],
                bias=empty,
            )
        elif case == "gate input":
            _verify(
                dict(x, f_a_out=empty.view(0, RANK)),
                records,
                None,
                override="cute_dsl_kda_fused_paged_verify_step",
                prepare=False,
            )
        elif case == "raw gate, split fallback":
            _verify(
                x,
                records,
                None,
                override="cute_dsl_kda_fused_paged_verify_step",
                prepare=False,
                g_raw=empty.view(0, WIDTH),
            )
        else:
            triton_nvidia_kda_fused_paged_verify_split(
                x["mixed_qkv"],
                x["conv_weights"],
                x["f_a_out"],
                x["f_b_weight"],
                x["beta_logits"],
                x["A_log"],
                x["dt_bias"],
                state_pool=x["state_pool"],
                read_indices=x["state_indices"],
                num_heads=HEADS,
                head_dim=DIM,
                draft_token_num=1,
                lower_bound=x["lower_bound"],
                retained_records=records,
                output_norm=None,
                g_raw=empty.view(0, WIDTH),
            )
    torch.cuda.synchronize()
    for value, original in zip(watched, before):
        assert torch.equal(value, original)


def test_the_fused_step_cannot_address_an_empty_gate_input():
    """The support predicate needs a gate input and beta row for every token."""
    from tokenspeed_kernel.ops.attention.kda._cute_dsl.fused_verify_step import (
        fused_verify_step_supported,
    )

    x = _inputs(3, seed=269)
    records, _ = _records(x)

    def supported(**changes):
        y = dict(x, **changes)
        return fused_verify_step_supported(
            y["mixed_qkv"],
            y["conv_weights"],
            records.preconv,
            y["f_a_out"],
            y["f_b_weight"],
            y["beta_logits"],
            y["A_log"],
            y["dt_bias"],
            y["state_pool"],
            records.history,
            y["query_start_loc"],
            y["state_indices"],
            HEADS,
            DIM,
            y["lower_bound"],
        )

    assert supported()
    assert not supported(f_a_out=x["f_a_out"][:0])
    assert not supported(f_a_out=x["f_a_out"][:2])
    assert not supported(beta_logits=x["beta_logits"][:0])


def _count_convolutions(monkeypatch):
    """Calls reaching the convolution producer, which every split route runs."""
    from tokenspeed_kernel.ops.attention.kda._triton import recurrent

    calls = []
    producer = recurrent.fused_kda_verify_conv_update

    def counted(*args, **kwargs):
        calls.append(kwargs.get("record_slots"))
        return producer(*args, **kwargs)

    monkeypatch.setattr(recurrent, "fused_kda_verify_conv_update", counted)
    return calls


def _malformed(x, operand):
    """A separately allocated beta, A_log or dt_bias the split verify cannot read."""
    if operand == "beta":
        # One logit too many per row: the consumer's [1, -1, heads] view fails.
        rows = x["beta_logits"].shape[0]
        return {"beta_logits": torch.zeros(rows, HEADS + 1, device="cuda").bfloat16()}
    return {operand: torch.empty(0, device="cuda")}


@pytest.mark.parametrize("route", ["split fallback", "triton split"])
@pytest.mark.parametrize("operand", ["beta", "A_log", "dt_bias"])
def test_a_malformed_beta_or_gate_parameter_is_rejected_before_the_convolution(
    operand, route, monkeypatch
):
    """The split verify checks beta and the gate parameters before its producer.

    Request 0 reads a live page, so the convolution would store its
    candidate row in the ring before the recurrent verify read beta,
    A_log and dt_bias. A beta whose rows do not view as [1, tokens, heads],
    or an empty A_log or dt_bias, each in its own allocation, raises
    ValueError on the CuTe entry's split fallback (with a raw gate) and on
    the Triton split verify (projecting its gate) without a convolution
    call. The whole ring allocation, the checkpoint, the records and the
    stamps keep their bits.
    """
    from tokenspeed_kernel.ops.attention.kda.triton import (
        triton_nvidia_kda_fused_paged_verify_split,
    )

    x = _inputs(3, seed=271)
    records, buffers = _records(x)
    _prepare(x, records)
    y = dict(x, **_malformed(x, operand))
    calls = _count_convolutions(monkeypatch)
    watched = (buffers[3], x["state_pool"], *buffers[:3])
    torch.cuda.synchronize()
    before = [value.clone() for value in watched]
    with pytest.raises(ValueError, match="beta|gate parameters"):
        if route == "split fallback":
            _verify(
                y,
                records,
                None,
                override="cute_dsl_kda_fused_paged_verify_step",
                prepare=False,
                g_raw=F.linear(x["f_a_out"], x["f_b_weight"]),
            )
        else:
            triton_nvidia_kda_fused_paged_verify_split(
                y["mixed_qkv"],
                y["conv_weights"],
                y["f_a_out"],
                y["f_b_weight"],
                y["beta_logits"],
                y["A_log"],
                y["dt_bias"],
                state_pool=y["state_pool"],
                read_indices=y["state_indices"],
                num_heads=HEADS,
                head_dim=DIM,
                draft_token_num=1,
                lower_bound=y["lower_bound"],
                retained_records=records,
                output_norm=None,
            )
    torch.cuda.synchronize()
    assert not calls
    for value, original in zip(watched, before):
        assert torch.equal(value, original)


def test_a_padded_beta_buffer_verifies_its_leading_rows():
    """A [rows, heads] beta padded past the tokens gives its first rows to the verify.

    The Triton split verify with the same tokens and a beta buffer of twice
    the rows (the extra rows hold garbage) matches the exact-size beta bit
    for bit: outputs, records and stamps.
    """
    from tokenspeed_kernel.ops.attention.kda.triton import (
        triton_nvidia_kda_fused_paged_verify_split,
    )

    x = _inputs(3, seed=277)
    beta = x["beta_logits"]
    padded = torch.full(
        (2 * beta.shape[0], HEADS), 1e4, device="cuda", dtype=beta.dtype
    )
    padded[: beta.shape[0]] = beta
    results = []
    for logits in (beta, padded):
        # Each run prepares its own copy of the pages.
        y = dict(x, state_pool=x["state_pool"].clone())
        records, buffers = _records(y)
        _prepare(y, records)
        out = triton_nvidia_kda_fused_paged_verify_split(
            y["mixed_qkv"],
            y["conv_weights"],
            y["f_a_out"],
            y["f_b_weight"],
            logits,
            y["A_log"],
            y["dt_bias"],
            state_pool=y["state_pool"],
            read_indices=y["state_indices"],
            num_heads=HEADS,
            head_dim=DIM,
            draft_token_num=1,
            lower_bound=y["lower_bound"],
            retained_records=records,
            output_norm=None,
        )
        results.append((out, buffers))
    (out, buffers), (padded_out, padded_buffers) = results
    assert torch.equal(out, padded_out)
    for value, padded_value in zip(buffers, padded_buffers):
        assert torch.equal(value, padded_value)


def _direct_step(x, records, **changes):
    """Call the fused step directly with ``changes`` to its operands."""
    from tokenspeed_kernel.ops.attention.kda._cute_dsl.fused_verify_step import (
        fused_verify_step,
    )

    y = dict(x, record_metadata=records.metadata, record_stamps=records.stamps)
    y.update(changes)
    tokens = y["mixed_qkv"].shape[0]
    out = y.get("out")
    if out is None:
        out = torch.empty(tokens, WIDTH, device="cuda", dtype=torch.bfloat16)
    result = fused_verify_step(
        y["mixed_qkv"],
        y["conv_weights"],
        records.preconv,
        y["f_a_out"],
        y["f_b_weight"],
        y["beta_logits"],
        y["A_log"],
        y["dt_bias"],
        y["lower_bound"],
        y["state_pool"][:, :HEADS],
        records.history,
        y["query_start_loc"],
        y["state_indices"],
        HEADS,
        DIM,
        out,
        y["record_metadata"],
        y["record_stamps"],
        None,
    )
    return result, out


def _forbid_compile(monkeypatch, module):
    def compile_(*args, **kwargs):
        raise AssertionError("the fused verify compiled for inputs it rejects")

    monkeypatch.setattr(module, "_compile", compile_)


@pytest.mark.parametrize(
    "operand", ["A_log", "dt_bias", "query_start_loc", "record_stamps"]
)
def test_an_empty_gate_parameter_boundary_or_stamp_table_is_rejected(
    operand, monkeypatch
):
    """The fused step rejects each empty operand it would address before compiling.

    Request 0 is live with metadata enabled. Its gate loads A_log[head] and
    dt_bias[head * 128 + k], it reads boundaries 0 and 1, and it stores a
    stamp pair in the slot its metadata names. An empty FP32 A_log or
    dt_bias, an empty INT32 query_start_loc or an empty [0, capacity, 2]
    stamp table, each in its own allocation, makes the support predicate
    (where it takes the operand) refuse and the direct entry raise
    ValueError without compiling; the ring, checkpoint, records and stamps
    keep their bits.
    """
    from tokenspeed_kernel.ops.attention.kda._cute_dsl import (
        fused_verify_step as step_kernel,
    )

    x = _inputs(3, seed=281)
    records, buffers = _records(x)
    _prepare(x, records)
    capacity = records.stamps.shape[1]
    empty = {
        "A_log": torch.empty(0, device="cuda"),
        "dt_bias": torch.empty(0, device="cuda"),
        "query_start_loc": torch.empty(0, device="cuda", dtype=torch.int32),
        "record_stamps": torch.empty(0, capacity, 2, device="cuda", dtype=torch.int64),
    }[operand]
    if operand != "record_stamps":
        y = dict(x, **{operand: empty})
        assert step_kernel.fused_verify_step_supported(
            x["mixed_qkv"],
            x["conv_weights"],
            records.preconv,
            x["f_a_out"],
            x["f_b_weight"],
            x["beta_logits"],
            x["A_log"],
            x["dt_bias"],
            x["state_pool"][:, :HEADS],
            records.history,
            x["query_start_loc"],
            x["state_indices"],
            HEADS,
            DIM,
            x["lower_bound"],
        )
        assert not step_kernel.fused_verify_step_supported(
            y["mixed_qkv"],
            y["conv_weights"],
            records.preconv,
            y["f_a_out"],
            y["f_b_weight"],
            y["beta_logits"],
            y["A_log"],
            y["dt_bias"],
            y["state_pool"][:, :HEADS],
            records.history,
            y["query_start_loc"],
            y["state_indices"],
            HEADS,
            DIM,
            y["lower_bound"],
        )
    _forbid_compile(monkeypatch, step_kernel)
    watched = (buffers[3], x["state_pool"], *buffers[:3])
    torch.cuda.synchronize()
    before = [value.clone() for value in watched]
    with pytest.raises(ValueError):
        _direct_step(x, records, **{operand: empty})
    torch.cuda.synchronize()
    for value, original in zip(watched, before):
        assert torch.equal(value, original)


def test_a_zero_token_step_returns_its_output_without_compiling(monkeypatch):
    """With no tokens the fused step returns its output before any check or launch."""
    from tokenspeed_kernel.ops.attention.kda._cute_dsl import (
        fused_verify_step as step_kernel,
    )

    x = _inputs(3, seed=283)
    records, _ = _records(x)
    _forbid_compile(monkeypatch, step_kernel)
    empty = dict(
        mixed_qkv=x["mixed_qkv"][:0],
        f_a_out=x["f_a_out"][:0],
        beta_logits=x["beta_logits"][:0],
    )
    result, out = _direct_step(x, records, **empty)
    assert result is out and out.shape == (0, WIDTH)


def _count_projections(monkeypatch):
    """Calls reaching the gate projection, which the split producers run first."""
    calls = []
    linear = F.linear

    def counted(*args, **kwargs):
        calls.append(args)
        return linear(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "linear", counted)
    return calls


@pytest.mark.parametrize("route", ["cute step", "cute split", "triton split"])
@pytest.mark.parametrize("operand", ["conv weights", "ring"])
def test_a_producer_operand_on_another_device_is_rejected_before_either_producer(
    route, operand, monkeypatch
):
    """Every split route checks the convolution operands' device before a producer.

    The convolution launch takes the weights and the ring as bare pointers.
    A host copy of the dense BF16 [3 * width, 4] weights, or a ring whose
    rows and heads are host copies, would be addressed as device memory.
    The CuTe step entry (which cannot fuse such operands and runs the split
    producers), the CuTe split producers called directly and the Triton
    split verify each raise ValueError before the gate projection and
    before the convolution producer. The device ring, checkpoint and
    records keep their bits.
    """
    from tokenspeed_kernel.ops.attention.kda import cute_dsl
    from tokenspeed_kernel.ops.attention.kda.triton import (
        triton_nvidia_kda_fused_paged_verify_split,
    )

    x = _inputs(3, seed=307)
    records, buffers = _records(x)
    _prepare(x, records)
    y = dict(x)
    if operand == "conv weights":
        y["conv_weights"] = x["conv_weights"].cpu()
    else:
        records = dataclasses.replace(
            records,
            preconv=KdaPreconvRing(
                records.preconv.rows.cpu(), records.preconv.heads.cpu()
            ),
        )
    watched = (buffers[3], x["state_pool"], *buffers[:3])
    torch.cuda.synchronize()
    before = [value.clone() for value in watched]
    convolutions = _count_convolutions(monkeypatch)
    projections = _count_projections(monkeypatch)
    with pytest.raises(ValueError, match="share one device"):
        if route == "cute step":
            _verify(
                y,
                records,
                None,
                override="cute_dsl_kda_fused_paged_verify_step",
                prepare=False,
            )
        else:
            verify = (
                cute_dsl._split_producer_verify
                if route == "cute split"
                else triton_nvidia_kda_fused_paged_verify_split
            )
            verify(
                y["mixed_qkv"],
                y["conv_weights"],
                y["f_a_out"],
                y["f_b_weight"],
                y["beta_logits"],
                y["A_log"],
                y["dt_bias"],
                g_raw=None,
                state_pool=y["state_pool"],
                read_indices=y["state_indices"],
                num_heads=HEADS,
                head_dim=DIM,
                draft_token_num=1,
                lower_bound=y["lower_bound"],
                retained_records=records,
                output_norm=None,
            )
    torch.cuda.synchronize()
    assert not projections
    assert not convolutions
    for value, original in zip(watched, before):
        assert torch.equal(value, original)


@pytest.mark.parametrize(
    "stamp_slots", [(0, 1), (-1, 0)], ids=["short table", "padding slot"]
)
def test_a_stamp_slot_outside_the_table_stores_no_stamp(stamp_slots):
    """The fused step stamps only slot rows its stamp table holds.

    Two live requests name stamp slots 0 and 1, or -1 (padding) and 0, of
    a one-row [1, 1, 2] stamp table cut from the middle of a three-row
    sentinel allocation. The verify completes without a fault, the request
    naming slot 0 stamps its start and generation in the table's row, and
    the sentinel rows on either side keep their bits.
    """
    x = _inputs(2, seed=311)
    x["record_metadata"] = torch.tensor(
        [[stamp_slots[0], 5, 10], [stamp_slots[1], 6, 11]],
        device="cuda",
        dtype=torch.int64,
    )
    records, _ = _records(x)
    _prepare(x, records)
    backing = torch.full((3, 1, 2), STAMP_FILL, device="cuda", dtype=torch.int64)
    _direct_step(x, records, record_stamps=backing[1:2])
    torch.cuda.synchronize()
    expected = torch.full_like(backing, STAMP_FILL)
    for slot, generation, start in x["record_metadata"].tolist():
        if slot == 0:
            expected[1, 0] = torch.tensor([start, generation])
    assert torch.equal(backing, expected)


@pytest.mark.parametrize(
    "layout", ["expanded", "overlapping", "strided channels", "fp32"]
)
def test_an_output_whose_token_rows_share_elements_is_rejected(layout, monkeypatch):
    """The fused step rejects an output it cannot store every token's row to.

    Three tokens store a row of 12 * 128 BF16 channels each at ``out +
    token * stride(0) + channel``. An expanded output (row pitch 0), rows
    overlapping at half a row's pitch, a channel stride of 2 or FP32
    elements raise ValueError without compiling, and the ring, checkpoint,
    records and stamps keep their bits.
    """
    from tokenspeed_kernel.ops.attention.kda._cute_dsl import (
        fused_verify_step as step_kernel,
    )

    x = _inputs(3, seed=313)
    records, buffers = _records(x)
    _prepare(x, records)
    tokens = x["mixed_qkv"].shape[0]
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    if layout == "expanded":
        out = torch.empty(1, WIDTH, **bf16).expand(tokens, WIDTH)
    elif layout == "overlapping":
        out = torch.empty(tokens * WIDTH, **bf16).as_strided(
            (tokens, WIDTH), (WIDTH // 2, 1)
        )
    elif layout == "strided channels":
        out = torch.empty(tokens, 2 * WIDTH, **bf16)[:, ::2]
    else:
        out = torch.empty(tokens, WIDTH, device="cuda")
    _forbid_compile(monkeypatch, step_kernel)
    watched = (buffers[3], x["state_pool"], *buffers[:3])
    torch.cuda.synchronize()
    before = [value.clone() for value in watched]
    with pytest.raises(ValueError, match="no two tokens sharing"):
        _direct_step(x, records, out=out)
    torch.cuda.synchronize()
    for value, original in zip(watched, before):
        assert torch.equal(value, original)


def test_a_pitched_output_gives_the_same_bits_as_a_dense_one():
    """Output rows with a pitch past their width hold the dense output's bits.

    The same three tokens verify into a dense [3, 12 * 128] output and into
    the leading columns of a [3, 12 * 128 + 64] sentinel buffer. The rows
    match bit for bit, the pitch padding keeps its sentinel, and the ring,
    records and stamps of the two runs are equal.
    """
    x = _inputs(3, seed=317)
    tokens = x["mixed_qkv"].shape[0]
    runs = []
    for pad in (0, 64):
        y = dict(x, state_pool=x["state_pool"].clone())
        records, buffers = _records(y)
        _prepare(y, records)
        backing = torch.full(
            (tokens, WIDTH + pad), 7.0, device="cuda", dtype=torch.bfloat16
        )
        out = backing[:, :WIDTH]
        assert out.stride() == (WIDTH + pad, 1)
        result, _ = _direct_step(y, records, out=out)
        assert result is out
        runs.append((backing, buffers))
    torch.cuda.synchronize()
    (dense, dense_buffers), (pitched, pitched_buffers) = runs
    assert torch.equal(pitched[:, :WIDTH], dense)
    assert bool((pitched[:, WIDTH:] == 7.0).all())
    for value, pitched_value in zip(dense_buffers, pitched_buffers):
        assert torch.equal(value, pitched_value)
