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

"""Retained serving contract: batched prepare, public verify, gathered endpoints,
batched commit."""

import pytest
import torch
from kda_retained_reference import prepare_history
from tokenspeed_kernel.ops.attention import kda
from tokenspeed_kernel.ops.metadata import REQUEST_ENDED, REQUEST_ENDED_INCOMPLETE
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _history(heads=2, capacity=8, dim=128, values=128):
    # Deliberately interleaved views: no slot/record axis is contiguous.
    shape = (3, heads, capacity, dim * 2)
    correction_shape = (3, heads, capacity, values * 2)
    correction = (
        torch.randn(correction_shape, device="cuda", dtype=torch.float32) * 0.03
    )[..., ::2]
    key = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.03)[..., ::2]
    decay = (torch.rand(shape, device="cuda", dtype=torch.float32) * 0.2 + 0.7)[
        ..., ::2
    ]
    # One strided accepted count per request slot.
    length = torch.zeros(3, 7, device="cuda", dtype=torch.int32)[:, 2:3]
    live = torch.tensor([3, 4, 5, 0], device="cuda", dtype=torch.int32)
    slots = torch.tensor([2, -1, 0, -1, 1, -1, -1, -1], device="cuda")[::2]
    return kda.KdaRetainedHistory(correction, key, decay, length, live, slots)


def _endpoint(base, history, slot, count):
    state = base.clone()
    for token in range(count):
        state = (
            state * history.decay[slot, :, token, None, :]
            + history.correction[slot, :, token, :, None]
            * history.key[slot, :, token, None, :]
        )
    return state


def _reference_tokens(running, taps, raw, weights, gate, beta, begin, end):
    """Verify tokens [begin, end) of one request from ``running`` and its taps.

    Two heads of dim 128, gate bound -5. Returns the per-token outputs,
    corrections, keys and decays as [tokens, heads, dim] and the final state.
    """
    outputs, corrections, keys, decays = [], [], [], []
    for token in range(begin, end):
        window_taps = torch.cat([taps, raw[token, :, None].float()], dim=-1)
        convolved = (
            torch.nn.functional.silu((window_taps * weights.float()).sum(-1))
            .to(raw.dtype)
            .float()
        )
        taps = window_taps[:, 1:]
        q, key, value = (x.reshape(2, 128) for x in convolved.chunk(3))
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / (128**0.5)
        key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        decay = (-5.0 * torch.sigmoid(gate[token].float().view(2, 128))).exp()
        running = running * decay[:, None, :]
        correction = (value - (running * key[:, None, :]).sum(-1)) * torch.sigmoid(
            beta[token].float()
        )[:, None]
        running = running + correction[:, :, None] * key[:, None, :]
        outputs.append((running * q[:, None, :]).sum(-1))
        corrections.append(correction)
        keys.append(key)
        decays.append(decay)
    stacked = [torch.stack(x) for x in (outputs, corrections, keys, decays)]
    return (*stacked, running)


@pytest.mark.parametrize("layout", ["v_major", "k_major"])
def test_materialize_shared_source_and_reset_private_lengths(layout):
    history = _history()
    # The shared checkpoint is dense. Only the third request (slot 1) has
    # pending records.
    history.length[1] = 1
    reads = torch.tensor([1, 1, 2, 0], device="cuda", dtype=torch.int32)
    base = torch.randn(8, 2, 128, 128, device="cuda")
    gathered = base[reads.long()].clone()
    gathered[-1].zero_()
    expected = gathered.clone()
    expected[2] = _endpoint(gathered[2], history, 1, 1)
    if layout == "k_major":
        gathered = gathered.transpose(-1, -2).contiguous()
        expected = expected.transpose(-1, -2)
    before = [
        x.clone()
        for x in (base, history.correction, history.key, history.decay, history.length)
    ]
    kda.kda_materialize_retained_state(
        gathered, history=history, source_indices=reads, recurrent_layout=layout
    )
    torch.testing.assert_close(gathered, expected, atol=2e-6, rtol=2e-6)
    for value, original in zip(
        (base, history.correction, history.key, history.decay, history.length), before
    ):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    history.length[:, 0] = torch.tensor([5, 6, 7], device="cuda")
    # Rows own slots 2, 0, 1 and -1. A nonpositive page keeps slot 0's count,
    # and a page past the pool keeps slot 1's, as materialization skips both
    # rows; the invalid slot is ignored.
    pages = torch.tensor([3, 0, 5, 4], device="cuda", dtype=torch.int32)
    kda.kda_reset_retained_length(
        history.length,
        write_indices=pages,
        record_slots=history.record_slots,
        page_count=5,
    )
    assert history.length[:, 0].tolist() == [5, 6, 0]
    kda.kda_reset_retained_length(
        history.length,
        write_indices=pages,
        record_slots=history.record_slots,
        page_count=6,
    )
    assert history.length[:, 0].tolist() == [5, 0, 0]


@pytest.mark.parametrize("layout", ["v_major", "k_major"])
def test_materialize_private_pages_uses_request_slots_and_clears_lengths(layout):
    history = _history(dim=19, values=37)
    pages = torch.tensor([3, 4, 5, 6], device="cuda", dtype=torch.int32)
    # Rows own slots 2, 0, 1 and -1; the invalid slot's row is skipped.
    history.length[:, 0] = torch.tensor([1, 0, 3], device="cuda")
    state = torch.randn(8, 2, 37, 19, device="cuda")
    expected = state.clone()
    expected[3] = _endpoint(state[3], history, 2, 3)
    expected[4] = _endpoint(state[4], history, 0, 1)
    if layout == "k_major":
        state = state.transpose(-1, -2).contiguous()
        expected = expected.transpose(-1, -2)
    kda.kda_materialize_retained_pages(
        state, history=history, source_indices=pages, recurrent_layout=layout
    )
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-6)
    assert history.length[:, 0].tolist() == [0, 0, 0]


def _fold(base, records, slot, layer, count):
    """Apply a slot's first ``count`` records of one layer to a dense state."""
    correction, key, decay = (field[slot, layer] for field in records)
    state = base.float().clone()
    for token in range(count):
        state = (
            state * decay[:, token, None, :]
            + correction[:, token, :, None] * key[:, token, None, :]
        )
    return state


def _commit_case(window, *, stale=False, stopped_slots=(), state_dtype=torch.float32):
    heads, dim, channels, layers, capacity = 2, 128, 768, 3, 8
    reads = torch.tensor(
        [[1, 1, 2, 0], [5, 1, 2, -1]], device="cuda", dtype=torch.int32
    )
    writes = torch.tensor(
        [[3, 4, 2, 0], [3, 4, 2, 0]], device="cuda", dtype=torch.int32
    )
    groups = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    starts = torch.tensor(
        [
            0,
            window,
            window + max(1, window - 1),
            2 * window + max(1, window - 1),
            2 * window + max(1, window - 1),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    accepted = torch.tensor(
        [0, max(1, window - 1), window, 0], device="cuda", dtype=torch.int32
    )
    metadata = torch.tensor(
        [[2, 9, 5], [0, 8, 11], [1, 7, 2], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    # Accepted counts live in request slots, independent of the stamp slots.
    record_slots = torch.tensor([1, 3, 0, -1], device="cuda", dtype=torch.int64)
    stamps = torch.full((layers, 3, 7, 2), -1, device="cuda", dtype=torch.int64)
    for row, (slot, generation, start) in enumerate(metadata[:3].tolist()):
        for token in range(window):
            stamps[:, slot, (start + token) % 7] = torch.tensor(
                [start + token, generation], device="cuda"
            )
    if stale:
        stamps[1, 0, 11 % 7, 1] = -1
    # FP32 records per request slot and layer; no axis is contiguous.
    shape = (4, layers + 1, heads, capacity, dim * 2)
    records = tuple(
        field[:, :layers, ..., ::2]
        for field in (
            torch.randn(shape, device="cuda") * 0.03,
            torch.randn(shape, device="cuda") * 0.03,
            torch.rand(shape, device="cuda") * 0.2 + 0.7,
        )
    )
    # One more flag than request slots; slot 2 ends without a row here.
    request_stopped = torch.zeros(5, device="cuda", dtype=torch.bool)
    request_stopped[list(stopped_slots)] = True
    # Each row's report as the stop conditions leave it.
    request_ends = torch.tensor(
        [REQUEST_ENDED if slot in stopped_slots else 0 for slot in (1, 3, 0, -1)],
        device="cuda",
        dtype=torch.int32,
    )
    (
        originals,
        convs,
        lengths,
        states,
        descriptors,
        expected_convs,
        expected_lengths,
        expected_states,
    ) = ([], [], [], [], [], [], [], [])
    for layer in range(layers):
        pages = 7 + layer
        qkv = torch.randn(
            int(starts[-1]), channels + layer * 4, device="cuda", dtype=torch.bfloat16
        )[:, :channels]
        conv = torch.randn(
            pages, channels + layer * 3, 8, device="cuda", dtype=torch.float32
        )[:, :channels, 1:7:2]
        length = torch.full((4, 5 + layer), 2, device="cuda", dtype=torch.int32)[:, 1:2]
        state = (torch.randn(pages, heads + 1, dim, dim, device="cuda") * 0.05).to(
            state_dtype
        )[:, :heads]
        expected_conv, expected_length = conv.clone(), length.clone()
        expected_state = state.to(torch.float32, copy=True)
        group = int(groups[layer])
        for row in range(3):
            if stale and layer == 1 and row == 1:
                continue
            source, target = int(reads[group, row]), int(writes[group, row])
            take, start = int(accepted[row]), int(starts[row])
            expected_conv[target] = torch.cat(
                (conv[source], qkv[start : start + take].float().T), dim=-1
            )[:, -3:]
            slot = int(record_slots[row])
            if slot in stopped_slots:
                expected_state[target] = _fold(
                    state[target], records, slot, layer, 2 + take
                )
                expected_length[slot] = 0
            else:
                expected_length[slot] += take
        descriptors.append(
            [
                qkv.data_ptr(),
                conv.data_ptr(),
                length.data_ptr(),
                pages,
                qkv.stride(0),
                conv.stride(0),
                length.stride(0),
                conv.stride(1),
                conv.stride(2),
                state.data_ptr(),
                *state.stride(),
            ]
        )
        originals.append(qkv)
        convs.append(conv)
        lengths.append(length)
        states.append(state)
        expected_convs.append(expected_conv)
        expected_lengths.append(expected_length)
        expected_states.append(expected_state.to(state_dtype))
    kwargs = dict(
        descriptors=torch.tensor(descriptors, device="cuda", dtype=torch.uint64),
        group_indices=groups,
        read_indices=reads,
        write_indices=writes,
        accepted_length=accepted,
        query_start_loc=starts,
        record_slots=record_slots,
        record_slot_count=4,
        record_metadata=metadata,
        record_stamps=stamps,
        retained_correction=records[0],
        retained_key=records[1],
        retained_decay=records[2],
        request_stopped=request_stopped,
        request_ends=request_ends,
        draft_token_num=window,
        history_capacity=capacity,
        num_heads=heads,
        head_dim=dim,
        conv_width=4,
        activation_dtype=torch.bfloat16,
        conv_state_dtype=torch.float32,
        recurrent_state_dtype=state_dtype,
    )
    expected = (expected_convs, expected_lengths, expected_states)
    return kwargs, originals, (convs, lengths, states), expected


@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("stale", [False, True])
def test_batched_retained_commit_packed_groups_and_stamps(window, stale):
    args, keepalive, actual, expected = _commit_case(window, stale=stale)
    states_before = [state.clone() for state in actual[2]]
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert commit is not None
    commit(**args)
    # No request ended, so a stale layer reports nothing.
    assert args["request_ends"].tolist() == [0, 0, 0, 0]
    for values, expected_values in zip(actual[:2], expected[:2]):
        for value, expected_value in zip(values, expected_values):
            torch.testing.assert_close(value, expected_value, atol=0, rtol=0)
    # Without an ended request the commit writes no dense state.
    for state, before in zip(actual[2], states_before):
        torch.testing.assert_close(state, before, atol=0, rtol=0)


@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_batched_retained_commit_folds_ended_requests(window, stale, state_dtype):
    """A request that ended in this forward leaves a dense live page.

    Slots 3 and 0 (rows 1 and 2) ended, so their records through this
    round's acceptance fold into their live pages and their counts clear.
    Slot 2 ended earlier and has no row. A stale layer folds nothing.
    """
    args, keepalive, actual, expected = _commit_case(
        window, stale=stale, stopped_slots=(3, 0, 2), state_dtype=state_dtype
    )
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    commit(**args)
    # Row 1's request ended, but its stale layer kept its records outside the
    # live page: its report says the state is incomplete.
    assert args["request_ends"].tolist() == [
        0,
        REQUEST_ENDED_INCOMPLETE if stale else REQUEST_ENDED,
        REQUEST_ENDED,
        0,
    ]
    convs, lengths, states = actual
    expected_convs, expected_lengths, expected_states = expected
    for conv, expected_conv in zip(convs, expected_convs):
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    for length, expected_length in zip(lengths, expected_lengths):
        torch.testing.assert_close(length, expected_length, atol=0, rtol=0)
    tolerance = (
        dict(atol=2e-6, rtol=2e-6)
        if state_dtype == torch.float32
        else dict(atol=1e-5, rtol=8e-3)
    )
    for state, expected_state in zip(states, expected_states):
        torch.testing.assert_close(state, expected_state, **tolerance)


@pytest.mark.parametrize(
    "window,far_slot", [(1, False), (4, False), pytest.param(4, True, id="far_slot")]
)
def test_public_retained_verify_keeps_shared_source_and_materializes_endpoint(
    window, far_slot
):
    torch.manual_seed(42)
    history = _history()
    if far_slot:
        # Two request strides fit individually in int32. The last slot's
        # element offset exceeds 2**31. Only the three short rows are touched.
        sources = (history.correction, history.key, history.decay)
        slots, heads, capacity, dim = history.correction.shape
        row_size = heads * capacity * dim
        stride = (1 << 30) + 3 * row_size
        if torch.cuda.mem_get_info()[0] < 10 << 30:
            pytest.skip("large request-slot offset needs 10 GiB free")
        storage = torch.empty(2 * stride + 3 * row_size, device="cuda")
        payload = [
            storage.as_strided(
                (slots, heads, capacity, dim),
                (stride, capacity * dim, dim, 1),
                i * row_size,
            ).copy_(source)
            for i, source in enumerate(sources)
        ]
        history = kda.KdaRetainedHistory(
            *payload, history.length, history.live_state_indices, history.record_slots
        )
        history.record_slots.copy_(torch.tensor([1, 0, 2, -1], device="cuda"))
    # Rows 0 and 1 resume from the shared dense page 1. Row 2 continues its
    # own page 2 with two accepted records in its slot.
    history.length[int(history.record_slots[2])] = 2
    reads = torch.tensor([1, 1, 2, 0], device="cuda", dtype=torch.int32)
    total = 2 * window + max(1, window - 1)
    starts = torch.tensor(
        [0, window, window + max(1, window - 1), total, total],
        device="cuda",
        dtype=torch.int32,
    )
    metadata = torch.tensor(
        [[0, 3, 5], [1, 4, 7], [2, 5, 9], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((3, 7, 2), -1, device="cuda", dtype=torch.int64)
    raw = torch.randn(total, 768, device="cuda", dtype=torch.bfloat16) * 0.2
    original = torch.empty_like(raw)
    conv = torch.randn(8, 768, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(768, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(total, 256, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(total, 2, device="cuda", dtype=torch.bfloat16)
    a = torch.zeros(2, device="cuda")
    bias = torch.zeros(256, device="cuda")
    state = torch.randn(8, 2, 128, 128, device="cuda") * 0.1
    saved = state[1].clone()
    records = kda.KdaRetainedRecords(
        history=history,
        original_qkv=original,
        metadata=metadata,
        stamps=stamps,
        query_start_loc=starts,
    )
    producer = {}
    # The split producer consumes the same packed boundaries as verification.
    if current_platform().is_nvidia:
        producer["conv_qkv"] = kda.kda_verify_conv_update(
            raw,
            weights,
            conv,
            reads,
            num_heads=2,
            head_dim=128,
            query_start_loc=starts,
            draft_token_num=window,
            original_qkv=None,
            bias=None,
            recurrent_layout="v_major",
        )
    expected_out = torch.empty(total, 2, 128, device="cuda", dtype=torch.float32)
    expected_final = []
    expected_u, expected_key, expected_decay = (
        torch.empty_like(expected_out) for _ in range(3)
    )
    for row, (begin, end) in enumerate(zip(starts[:3].tolist(), starts[1:4].tolist())):
        page = int(reads[row])
        slot = int(history.record_slots[row])
        running = _endpoint(state[page], history, slot, int(history.length[slot]))
        (
            expected_out[begin:end],
            expected_u[begin:end],
            expected_key[begin:end],
            expected_decay[begin:end],
            running,
        ) = _reference_tokens(
            running, conv[page].float(), raw, weights, gate, beta, begin, end
        )
        expected_final.append(running)
    # Every row moves to a private live page, so the prepare copies them all.
    prepare_history(state, history, reads, starts, draft_token_num=window)
    out = kda.kda_fused_paged_verify(
        raw,
        weights,
        conv,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        beta,
        a,
        bias,
        state_pool=state,
        read_indices=reads,
        num_heads=2,
        head_dim=128,
        draft_token_num=window,
        lower_bound=-5.0,
        recurrent_layout="v_major",
        retained_records=records,
        output_norm=None,
        g_raw=gate,
        **producer,
        override=None,
        solution=None,
    )
    assert out is not None
    torch.testing.assert_close(original, raw, atol=0, rtol=0)
    torch.testing.assert_close(state[1], saved, atol=0, rtol=0)
    torch.testing.assert_close(state[3], saved, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(state[4], saved, atol=2e-6, rtol=2e-6)
    # Each copy to a private live page starts its slot's history over.
    assert history.length[:, 0].tolist() == [0, 0, 0]
    torch.testing.assert_close(
        out.reshape(total, 2, 128).float(), expected_out, atol=2e-4, rtol=1e-2
    )
    for row in range(3):
        slot = int(history.record_slots[row])
        begin, end = int(starts[row]), int(starts[row + 1])
        torch.testing.assert_close(
            history.correction[slot, :, : end - begin],
            expected_u[begin:end].transpose(0, 1),
            atol=2e-4,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            history.key[slot, :, : end - begin],
            expected_key[begin:end].transpose(0, 1),
            atol=1e-3,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            history.decay[slot, :, : end - begin],
            expected_decay[begin:end].transpose(0, 1),
            atol=1e-6,
            rtol=1e-6,
        )
        history.length[slot] = end - begin
    endpoint = state[history.live_state_indices[:3].long()].clone()
    kda.kda_materialize_retained_state(
        endpoint,
        history=history,
        source_indices=history.live_state_indices[:3],
        recurrent_layout="v_major",
    )
    torch.testing.assert_close(
        endpoint, torch.stack(expected_final), atol=2e-4, rtol=1e-2
    )


@pytest.mark.parametrize(
    "window,capacity,count", [(8, 24, 20), (16, 16, 5)], ids=["rows16", "rows0"]
)
def test_a_short_window_folds_the_records_verify_cannot_read(window, capacity, count):
    """A request shorter than the verify's window folds as a full one would.

    The window kernels read at most capacity - window records (16 when the
    capacity is 24 and the window 8; none when both are 16). A 2-token request
    holding more fits its own window, yet the prepare must fold it, or verify
    would leave records out of its outputs.
    """
    torch.manual_seed(11)
    history = _history(capacity=capacity)
    # Row 0 continues its own page 3 with two tokens; rows 1-3 are empty.
    slot = int(history.record_slots[0])
    history.length[slot] = count
    reads = torch.tensor([3, 0, 0, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 2, 2, 2, 2], device="cuda", dtype=torch.int32)
    metadata = torch.tensor(
        [[0, 3, 5], [1, 4, 7], [2, 5, 9], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((3, window, 2), -1, device="cuda", dtype=torch.int64)
    raw = torch.randn(2, 768, device="cuda", dtype=torch.bfloat16) * 0.2
    original = torch.empty_like(raw)
    conv = torch.randn(8, 768, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(768, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(2, 2, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(8, 2, 128, 128, device="cuda") * 0.1
    endpoint = _endpoint(state[3], history, slot, count)
    expected_out, *_ = _reference_tokens(
        endpoint, conv[3].float(), raw, weights, gate, beta, 0, 2
    )

    prepare_history(state, history, reads, starts, draft_token_num=window)
    assert int(history.length[slot]) == 0
    torch.testing.assert_close(state[3], endpoint, atol=2e-6, rtol=2e-5)

    producer = {}
    if current_platform().is_nvidia:
        producer["conv_qkv"] = kda.kda_verify_conv_update(
            raw,
            weights,
            conv,
            reads,
            num_heads=2,
            head_dim=128,
            query_start_loc=starts,
            draft_token_num=window,
            original_qkv=None,
            bias=None,
            recurrent_layout="v_major",
        )
    out = kda.kda_fused_paged_verify(
        raw,
        weights,
        conv,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        beta,
        torch.zeros(2, device="cuda"),
        torch.zeros(256, device="cuda"),
        state_pool=state,
        read_indices=reads,
        num_heads=2,
        head_dim=128,
        draft_token_num=window,
        lower_bound=-5.0,
        recurrent_layout="v_major",
        retained_records=kda.KdaRetainedRecords(
            history=history,
            original_qkv=original,
            metadata=metadata,
            stamps=stamps,
            query_start_loc=starts,
        ),
        output_norm=None,
        g_raw=gate,
        **producer,
        override=None,
        solution=None,
    )
    assert out is not None
    torch.testing.assert_close(
        out.reshape(2, 2, 128).float(), expected_out, atol=2e-4, rtol=1e-2
    )


def test_packed_verify_convolution_uses_request_boundaries():
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA split producer; AMD verifies packed convolution in fusion")
    torch.manual_seed(7)
    raw = torch.randn(7, 768, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(768, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    conv = torch.randn(6, 768, 3, device="cuda", dtype=torch.bfloat16)
    reads = torch.tensor([1, 2, 3, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 4, 5, 7, 7], device="cuda", dtype=torch.int32)
    expected = torch.empty_like(raw)
    for row, (begin, end) in enumerate(zip(starts[:-1].tolist(), starts[1:].tolist())):
        history = conv[int(reads[row])].float()
        for token in range(begin, end):
            taps = torch.cat([history, raw[token, :, None].float()], dim=-1)
            expected[token] = torch.nn.functional.silu((taps * weights.float()).sum(-1))
            history = taps[:, 1:]
    output = kda.kda_verify_conv_update(
        raw,
        weights,
        conv,
        reads,
        num_heads=2,
        head_dim=128,
        draft_token_num=4,
        query_start_loc=starts,
        recurrent_layout="v_major",
        bias=None,
        original_qkv=None,
    )
    torch.testing.assert_close(output, expected, atol=0.003, rtol=0.005)


def test_retained_commit_graph_refresh_and_layer_subset():
    args, keepalive, (convs, lengths, _), _ = _commit_case(4)
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert commit is not None
    # Noncontiguous stamp and record layer views must retain their parent
    # strides; record layers follow the descriptor rows.
    subset = dict(
        args,
        descriptors=args["descriptors"][::2].contiguous(),
        group_indices=args["group_indices"][::2].contiguous(),
        record_stamps=args["record_stamps"][::2],
        retained_correction=args["retained_correction"][:, ::2],
        retained_key=args["retained_key"][:, ::2],
        retained_decay=args["retained_decay"][:, ::2],
    )
    before_conv = [x.clone() for x in convs]
    before_length = [x.clone() for x in lengths]
    commit(**subset)
    for actual, initial in zip(convs, before_conv):
        actual.copy_(initial)
    for actual, initial in zip(lengths, before_length):
        actual.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit(**subset)
    for actual, initial in zip(convs, before_conv):
        actual.copy_(initial)
    for actual, initial in zip(lengths, before_length):
        actual.copy_(initial)
    subset["accepted_length"].zero_()
    graph.replay()
    for layer in (0, 2):
        for row in range(3):
            group = int(subset["group_indices"][layer // 2])
            source = int(args["read_indices"][group, row])
            target = int(args["write_indices"][group, row])
            torch.testing.assert_close(
                convs[layer][target], before_conv[layer][source], atol=0, rtol=0
            )
        torch.testing.assert_close(lengths[layer], before_length[layer], atol=0, rtol=0)
    torch.testing.assert_close(convs[1], before_conv[1], atol=0, rtol=0)
    torch.testing.assert_close(lengths[1], before_length[1], atol=0, rtol=0)


def test_retained_commit_does_not_specialize_dynamic_metadata():
    from tokenspeed_kernel.ops.attention.kda._retained import _kernels

    args, keepalive, _, _ = _commit_case(4)
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert commit is not None
    args["accepted_length"].zero_()
    commit(**args)
    kernels, _ = _kernels()
    shorter = dict(
        args,
        read_indices=args["read_indices"][:, :2],
        write_indices=args["write_indices"][:, :2],
        accepted_length=args["accepted_length"][:2],
        query_start_loc=args["query_start_loc"][:3],
        record_slots=args["record_slots"][:2],
        record_metadata=args["record_metadata"][:2],
        record_stamps=torch.full((3, 5, 11, 2), -1, device="cuda", dtype=torch.int64),
        request_ends=args["request_ends"][:2],
    )
    with assert_no_triton_compile(
        kernels.retained_batched_conv_kernel, kernels.retained_batched_publish_kernel
    ):
        commit(**shorter)
        # Serving packs each [groups, batch] page table contiguously at the
        # start of its buffer, so its row stride is the batch size: 1,
        # multiples of 16 and the rest.
        for batch in (1, 2, 16, 17, 33):
            reads, writes = (
                torch.full((2, batch), -1, device="cuda", dtype=torch.int32)
                for _ in range(2)
            )
            commit(
                **dict(
                    args,
                    read_indices=reads,
                    write_indices=writes,
                    accepted_length=torch.zeros(
                        batch, device="cuda", dtype=torch.int32
                    ),
                    query_start_loc=torch.arange(
                        batch + 1, device="cuda", dtype=torch.int32
                    ),
                    record_slots=torch.zeros(batch, device="cuda", dtype=torch.int64),
                    record_metadata=torch.full(
                        (batch, 3), -1, device="cuda", dtype=torch.int64
                    ),
                    request_ends=torch.zeros(batch, device="cuda", dtype=torch.int32),
                )
            )


def _prepare_case(window, *, state_dtype=torch.float32):
    """Three layers in two State groups, each layer with its own arena strides.

    Row 0 moves from a shared source page to its private live page. Row 1
    continues its own page and overflows the capacity by one record. Row 2
    continues its own page and fits exactly, except in layer 1, whose count
    for its slot overflows. Rows 3-6 are padding -- an empty window, a source
    one past the pool, a slot past the record slots, the null page -- and
    keep their pages and counts. Row 7 continues its own page with fewer
    tokens than the window: its own tokens fit, but it folds whenever a full
    window would overflow.
    """
    heads, dim, layers, capacity, slots, pages, rows = 2, 128, 3, 8, 7, 10, 8
    groups = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    # [groups, rows] views of 16-row tables, like serving's padded per-batch
    # tables: every row starts on a 16-byte boundary.
    tables = torch.full((2, 2, 16), -1, device="cuda", dtype=torch.int32)
    sources, lives = tables[0, :, :rows], tables[1, :, :rows]
    # Each group names its own pages; live pages are private within a group.
    sources.copy_(
        torch.tensor([[1, 2, 3, 4, pages, 1, 0, 4], [2, 6, 7, 4, pages, 2, 0, 4]])
    )
    lives.copy_(torch.tensor([[5, 2, 3, 6, 7, 8, 9, 4], [9, 6, 7, 5, 8, 1, 3, 4]]))
    sizes = [window, window, window, 0, window, window, window, max(1, window - 3)]
    starts = torch.tensor(
        [0] + torch.tensor(sizes).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    record_slots = torch.tensor([2, 4, 0, 1, 3, slots, 5, 6], device="cuda")
    counts = {
        0: capacity - window,
        1: 5,
        2: 3,
        3: 6,
        4: capacity - window + 1,
        5: 7,
        6: capacity - window + 1,
    }
    shape = (slots, layers + 1, heads, capacity, dim * 2)
    records = tuple(
        field[:, :layers, ..., ::2]
        for field in (
            torch.randn(shape, device="cuda") * 0.03,
            torch.randn(shape, device="cuda") * 0.03,
            torch.rand(shape, device="cuda") * 0.2 + 0.7,
        )
    )
    states, lengths, descriptors = [], [], []
    expected_states, expected_lengths = [], []
    for layer in range(layers):
        state = (
            torch.randn(pages, heads + 1 + layer, dim, dim, device="cuda") * 0.05
        ).to(state_dtype)[:, :heads]
        length = torch.zeros(slots, 5 + layer, device="cuda", dtype=torch.int32)[:, 1:2]
        for slot, count in counts.items():
            length[slot] = count + (layer == 1 and slot == 0)
        expected_state = state.clone()
        expected_length = length.clone()
        group = int(groups[layer])
        for row in range(rows):
            source, live = int(sources[group, row]), int(lives[group, row])
            slot, tokens = int(record_slots[row]), sizes[row]
            if not (
                0 < source < pages and 0 < live < pages and tokens and slot < slots
            ):
                continue
            count = int(length[slot])
            if source != live or count + max(tokens, window) > capacity:
                expected_state[live] = _fold(
                    state[source], records, slot, layer, count
                ).to(state_dtype)
                expected_length[slot] = 0
        row = [0] * kda.KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS
        row[2], row[3], row[6] = length.data_ptr(), pages, length.stride(0)
        row[9:14] = state.data_ptr(), *state.stride()
        descriptors.append(row)
        states.append(state)
        lengths.append(length)
        expected_states.append(expected_state)
        expected_lengths.append(expected_length)
    args = dict(
        descriptors=torch.tensor(descriptors, device="cuda", dtype=torch.uint64),
        group_indices=groups,
        source_indices=sources,
        live_indices=lives,
        query_start_loc=starts,
        record_slots=record_slots,
        record_slot_count=slots,
        retained_correction=records[0],
        retained_key=records[1],
        retained_decay=records[2],
        history_capacity=capacity,
        draft_token_num=window,
        num_heads=heads,
        head_dim=dim,
        recurrent_state_dtype=state_dtype,
    )
    return args, (states, lengths), (expected_states, expected_lengths)


def _resolve_prepare():
    prepare = kda.resolve_kda_batched_retained_prepare(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert prepare is not None
    return prepare


@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("captured", [False, True])
def test_batched_retained_prepare_folds_private_and_overflowing_rows(
    window, state_dtype, captured
):
    args, (states, lengths), (expected_states, expected_lengths) = _prepare_case(
        window, state_dtype=state_dtype
    )
    fields = ("retained_correction", "retained_key", "retained_decay")
    records = [args[name].clone() for name in fields]
    initial = [x.clone() for x in (*states, *lengths)]
    prepare = _resolve_prepare()
    prepare(**args)
    if captured:
        for value, before in zip((*states, *lengths), initial):
            value.copy_(before)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare(**args)
        for value, before in zip((*states, *lengths), initial):
            value.copy_(before)
        graph.replay()
    tolerance = (
        dict(atol=2e-6, rtol=2e-5)
        if state_dtype == torch.float32
        else dict(atol=2e-3, rtol=4e-3)
    )
    for state, expected, before in zip(states, expected_states, initial):
        changed = (expected != before).flatten(1).any(1)
        torch.testing.assert_close(state[changed], expected[changed], **tolerance)
        torch.testing.assert_close(state[~changed], before[~changed], atol=0, rtol=0)
    for length, expected in zip(lengths, expected_lengths):
        torch.testing.assert_close(length, expected, atol=0, rtol=0)
    # Layer 1 alone overflows row 2's slot: each layer decides on its own count.
    assert [int(length[0]) for length in lengths] == [8 - window, 0, 8 - window]
    for name, before in zip(fields, records):
        torch.testing.assert_close(args[name], before, atol=0, rtol=0)


def test_batched_retained_prepare_does_not_specialize_dynamic_metadata():
    from tokenspeed_kernel.ops.attention.kda._triton.retained import (
        retained_batched_consolidate_kernel,
        retained_batched_reset_kernel,
    )

    args, _, _ = _prepare_case(4)
    prepare = _resolve_prepare()
    prepare(**args)
    with assert_no_triton_compile(
        retained_batched_consolidate_kernel, retained_batched_reset_kernel
    ):
        for batch, padded in ((1, 1), (2, 5), (5, 16), (17, 17), (33, 48)):
            # Padding rows only: the launches still run with these strides. As
            # in serving, each row of the padded tables starts on 128 bytes.
            pitch = -(-padded // 32) * 32
            tables = torch.full((2, 2, pitch), -1, device="cuda", dtype=torch.int32)
            prepare(
                **dict(
                    args,
                    source_indices=tables[0, :, :batch],
                    live_indices=tables[1, :, :batch],
                    query_start_loc=torch.arange(
                        batch + 1, device="cuda", dtype=torch.int32
                    ),
                    record_slots=torch.zeros(batch, device="cuda", dtype=torch.int64),
                )
            )


def test_far_pages_of_a_pool_past_int32_prepare_like_near_ones():
    """A page whose element offset passes 2**31 folds like a near one."""
    heads, dim, capacity = 2, 128, 8
    page_stride = (1 << 30) + heads * dim * dim
    if torch.cuda.mem_get_info()[0] < 6 << 30:
        pytest.skip("a pool past 2**31 elements needs 6 GiB free")
    storage = torch.empty(
        2 * page_stride + heads * dim * dim, device="cuda", dtype=torch.bfloat16
    )
    state = storage.as_strided((3, heads, dim, dim), (page_stride, dim * dim, dim, 1))
    state[1:].normal_(std=0.05)
    history = _history(heads=heads, capacity=capacity)
    history.length.fill_(3)
    reads = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.int32)
    history.live_state_indices.copy_(
        torch.tensor([2, 0, 0, 0], device="cuda", dtype=torch.int32)
    )
    starts = torch.tensor([0, 4, 4, 4, 4], device="cuda", dtype=torch.int32)
    slot = int(history.record_slots[0])
    expected = _endpoint(state[1].float(), history, slot, 3).to(torch.bfloat16)
    source = state[1].clone()
    prepare_history(state, history, reads, starts, draft_token_num=4)
    torch.testing.assert_close(state[2], expected, atol=2e-3, rtol=4e-3)
    torch.testing.assert_close(state[1], source, atol=0, rtol=0)
    assert int(history.length[slot]) == 0


def test_retained_capability_requires_every_explicit_contract():
    assert kda.kda_retained_supported(
        torch.bfloat16,
        recurrent_layout="v_major",
        num_heads=2,
        head_dim=128,
        conv_kernel_size=4,
    )
    for resolve in (
        kda.resolve_kda_batched_retained_prepare,
        kda.resolve_kda_batched_retained_commit,
    ):
        assert resolve(
            torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
        )
        assert not resolve(
            torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="k_major"
        )
    for dtype, layout, heads, dim, width in [
        (torch.float32, "v_major", 2, 128, 4),
        (torch.bfloat16, "k_major", 2, 128, 4),
        (torch.bfloat16, "v_major", 2, 128, 3),
        (torch.bfloat16, "v_major", 0, 128, 4),
    ]:
        assert not kda.kda_retained_supported(
            dtype,
            recurrent_layout=layout,
            num_heads=heads,
            head_dim=dim,
            conv_kernel_size=width,
        )
    assert (
        kda.kda_fused_paged_verify_uses_split_producers(
            torch.bfloat16,
            recurrent_layout="v_major",
            num_heads=2,
            head_dim=128,
        )
        == current_platform().is_nvidia
    )


def test_invalid_checkpoint_verify_raises_without_mutation():
    """A conv/state page-count mismatch fails before touching any page or record."""
    history = _history(heads=1, dim=128)
    reads = torch.tensor([1, 1, 2, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 1, 2, 3, 3], device="cuda", dtype=torch.int32)
    raw = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    conv = torch.randn(8, 384, 3, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(7, 1, 128, 128, device="cuda")
    original = torch.full_like(raw, 7)
    metadata = torch.tensor(
        [[0, 1, 0], [1, 1, 0], [2, 1, 0], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((3, 8, 2), -1, device="cuda", dtype=torch.int64)
    records = kda.KdaRetainedRecords(original, history, metadata, stamps, starts)
    mutable = (
        raw,
        conv,
        state,
        original,
        history.correction,
        history.key,
        history.decay,
        history.length,
        stamps,
    )
    before = [value.clone() for value in mutable]
    with pytest.raises(ValueError):
        kda.kda_fused_paged_verify(
            raw,
            torch.ones(384, 4, device="cuda", dtype=torch.bfloat16),
            conv,
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
            torch.zeros(3, 1, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, device="cuda"),
            torch.zeros(128, device="cuda"),
            state_pool=state,
            read_indices=reads,
            num_heads=1,
            head_dim=128,
            draft_token_num=1,
            recurrent_layout="v_major",
            retained_records=records,
            output_norm=None,
            g_raw=torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16),
            conv_qkv=raw if current_platform().is_nvidia else None,
            lower_bound=-5.0,
            override=None,
            solution=None,
        )
    for actual, expected in zip(mutable, before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
