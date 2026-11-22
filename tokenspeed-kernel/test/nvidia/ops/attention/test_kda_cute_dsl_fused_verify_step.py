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


def _records(x, *, original, capacity=1, correction_stride=1):
    """Sentinel-filled request records, with padding around each slot."""
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
        x["state_pool"], records.history, x["state_indices"], x["query_start_loc"]
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
    records, buffers = _records(x, original=False) if prepared is None else prepared
    conv_qkv = fused_kda_verify_conv_update(
        x["mixed_qkv"],
        x["conv_weights"],
        x["conv_states"],
        x["state_indices"],
        num_heads=HEADS,
        head_dim=DIM,
        draft_token_num=1,
        original_qkv=buffers[3][:, : 3 * WIDTH],
        query_start_loc=x["query_start_loc"],
    )
    g_raw = F.linear(x["f_a_out"], x["f_b_weight"])
    return _verify(x, records, None, g_raw=g_raw, conv_qkv=conv_qkv), buffers


def _fused(x, output_norm=None):
    records, buffers = _records(x, original=True)
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
    records = _records(x, original=True, capacity=8)[0]
    expected = _records(x, original=True, capacity=8)[0]
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
    records, _ = _records(x, original=True, capacity=32)
    reference, buffers = _records(x, original=False, capacity=32)
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
    assert torch.equal(records.original_qkv, buffers[3][:, : 3 * WIDTH])
    assert history.length[:13, 0].tolist() == [0, 5, 7, 1, 2, 3, 8, 9, 15, 16, 17, 0, 0]
    assert torch.equal(history.length, reference.history.length)
    torch.testing.assert_close(
        x["state_pool"].float(), ref_x["state_pool"].float(), atol=2e-6, rtol=2e-5
    )


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
    records = _records(x, original=True)[0]
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
    expected = _split(
        x, _records(x, original=False, correction_stride=correction_stride)
    )
    records, buffers = _records(x, original=True, correction_stride=correction_stride)
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
