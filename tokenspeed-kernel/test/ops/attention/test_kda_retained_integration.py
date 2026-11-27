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

from dataclasses import replace

import pytest
import torch
from kda_retained_reference import (
    preconv_ring,
    prepare_history,
    replay_state,
    snapshot,
)
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
    # One strided accepted count per request slot, beside its ring state.
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


def _ring(slots, window, channels, *, dtype=torch.bfloat16):
    """A padded [slots, window + 3, channels] ring of distinct values and its taps.

    ``taps(slot, state)`` is the [channels, 3] window a seeded ring state
    (the row of the slot's next input, plus one) leaves before its head.
    """
    rows = window + 3
    ring = torch.randn(slots + 1, rows, channels + 5, device="cuda").to(dtype)[
        1:, :, :channels
    ]

    def taps(slot, state):
        head = state - 1
        return ring[slot, [(head - 3 + i) % rows for i in range(3)]].T

    return ring, taps


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
    # Rows own slots 2, 0, 1 and -1 and live pages 3, 4, 5 and 0. Slot 2's
    # ring wraps (its next input goes to row 1), slot 0's is not seeded and
    # slot 1's holds the window before row 5: the gather copied each source
    # page's taps into its live page, and a seeded ring's window replaces
    # them; the unseeded slot's and the padding row's pages keep theirs.
    ring, taps = _ring(3, 4, 768)
    conv = torch.randn(8, 768, 3, device="cuda").to(torch.bfloat16)
    expected_conv = conv.clone()
    expected_conv[3] = taps(2, 2)
    expected_conv[5] = taps(1, 6)
    replay_state(history)[:, 1] = torch.tensor([0, 6, 2], device="cuda")
    before = [
        x.clone()
        for x in (
            base,
            history.correction,
            history.key,
            history.decay,
            replay_state(history),
            ring,
        )
    ]
    kda.kda_materialize_retained_state(
        gathered,
        history=history,
        source_indices=reads,
        recurrent_layout=layout,
        conv_pool=conv,
        preconv=ring,
        ring_state=replay_state(history)[:, 1:],
    )
    torch.testing.assert_close(gathered, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    for value, original in zip(
        (
            base,
            history.correction,
            history.key,
            history.decay,
            replay_state(history),
            ring,
        ),
        before,
    ):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    state = replay_state(history)
    state.copy_(torch.tensor([[5, 2], [6, 3], [7, 4]], device="cuda"))
    # Rows own slots 2, 0, 1 and -1. A nonpositive page keeps slot 0's count
    # and ring state, and a page past the pool keeps slot 1's, as
    # materialization skips both rows; the invalid slot is ignored.
    pages = torch.tensor([3, 0, 5, 4], device="cuda", dtype=torch.int32)
    kda.kda_reset_retained_length(
        state,
        write_indices=pages,
        record_slots=history.record_slots,
        page_count=5,
    )
    assert state.tolist() == [[5, 2], [6, 3], [0, 0]]
    kda.kda_reset_retained_length(
        state,
        write_indices=pages,
        record_slots=history.record_slots,
        page_count=6,
    )
    assert state.tolist() == [[5, 2], [0, 0], [0, 0]]


@pytest.mark.parametrize("layout", ["v_major", "k_major"])
def test_materialize_private_pages_uses_request_slots_and_keeps_counts(layout):
    history = _history(dim=19, values=37)
    pages = torch.tensor([3, 4, 5, 6], device="cuda", dtype=torch.int32)
    history = kda.KdaRetainedHistory(
        history.correction,
        history.key,
        history.decay,
        history.length,
        pages,
        history.record_slots,
    )
    # Rows own slots 2, 0, 1 and -1; the invalid slot's row is skipped.
    history.length[:, 0] = torch.tensor([1, 0, 3], device="cuda")
    state = torch.randn(8, 2, 37, 19, device="cuda")
    expected = state.clone()
    expected[3] = _endpoint(state[3], history, 2, 3)
    expected[4] = _endpoint(state[4], history, 0, 1)
    if layout == "k_major":
        state = state.transpose(-1, -2).contiguous()
        expected = expected.transpose(-1, -2)
    # Slot 1 (page 5) holds no record but a seeded ring: its window still
    # goes into its page. Slot 0's ring is not seeded, so page 4 keeps its
    # taps while its record folds; slot 2's head is row 3. The 600 channels
    # end inside a 256-channel block.
    ring, taps = _ring(3, 1, 600, dtype=torch.float16)
    conv = torch.randn(8, 600, 3, device="cuda").to(torch.float16)
    expected_conv = conv.clone()
    expected_conv[3] = taps(2, 4)
    expected_conv[5] = taps(1, 2)
    replay_state(history)[:, 1] = torch.tensor([0, 2, 4], device="cuda")
    kda.kda_materialize_retained_pages(
        state,
        history=history,
        source_indices=pages,
        recurrent_layout=layout,
        conv_pool=conv,
        preconv=ring,
        ring_state=replay_state(history)[:, 1:],
    )
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    # Every layer folds from the counts and ring states its request shares,
    # so a fold keeps them; one reset after the last layer clears both.
    assert replay_state(history).tolist() == [[1, 0], [0, 2], [3, 4]]
    kda.kda_reset_retained_length(
        replay_state(history),
        write_indices=pages,
        record_slots=history.record_slots,
        page_count=state.shape[0],
    )
    assert replay_state(history).tolist() == [[0, 0], [0, 0], [0, 0]]


@pytest.mark.parametrize("layout", ["v_major", "k_major"])
def test_a_paged_fold_writes_the_listed_page_not_the_historys_live_page(layout):
    """A retraction lists the page it folds into; history.live_state_indices
    names other pages. Each listed page gets the slot's records and, for a
    seeded ring, its window; the history's live pages stay as they were."""
    history = _history(dim=19, values=37)
    # Rows own slots 2, 0, 1 and -1. The history names live pages 3, 4, 1
    # and 0; the fold lists pages 5, 6, 7 and 0.
    history = kda.KdaRetainedHistory(
        history.correction,
        history.key,
        history.decay,
        history.length,
        torch.tensor([3, 4, 1, 0], device="cuda", dtype=torch.int32),
        history.record_slots,
    )
    pages = torch.tensor([5, 6, 7, 0], device="cuda", dtype=torch.int32)
    # Slot 2 has records and a seeded ring, slot 0 records only and slot 1 a
    # seeded ring only.
    history.length[:, 0] = torch.tensor([1, 0, 3], device="cuda")
    replay_state(history)[:, 1] = torch.tensor([0, 2, 4], device="cuda")
    state = torch.randn(8, 2, 37, 19, device="cuda")
    expected = state.clone()
    expected[5] = _endpoint(state[5], history, 2, 3)
    expected[6] = _endpoint(state[6], history, 0, 1)
    if layout == "k_major":
        state = state.transpose(-1, -2).contiguous()
        expected = expected.transpose(-1, -2)
    ring, taps = _ring(3, 1, 600)
    conv = torch.randn(8, 600, 3, device="cuda").to(torch.bfloat16)
    expected_conv = conv.clone()
    expected_conv[5] = taps(2, 4)
    expected_conv[7] = taps(1, 2)
    kda.kda_materialize_retained_pages(
        state,
        history=history,
        source_indices=pages,
        recurrent_layout=layout,
        conv_pool=conv,
        preconv=ring,
        ring_state=replay_state(history)[:, 1:],
    )
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    # The history's live pages 3, 4 and 1 keep their recurrent and
    # convolution states bit for bit.
    for page in (1, 3, 4):
        assert torch.equal(conv[page], expected_conv[page])
        assert torch.equal(state[page], expected[page])
    assert replay_state(history).tolist() == [[1, 0], [0, 2], [3, 4]]


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
    """Three layers in two State groups commit four rows; row 3 is padding.

    Every layer of a request shares its count and ring state: the rows' slots
    hold 2, the records their verify read, and slot 2, which has no row,
    holds 7. The prepare's snapshot gives each row that count and the ring
    head its verify started the window at, where its candidates' raw rows
    wait in each layer's ring (wrapping past the last ring row) after the
    three taps it read, its source page's. The commit writes each row's
    count and ring state once, from the snapshot and its acceptance, and
    writes no page of a request that keeps decoding. A request that ended
    (``stopped_slots``) gets, per layer, its records folded and the ring's
    window after its acceptance written into its live page. With
    ``stale``, layer 1 rejects row 1's commit and leaves its page alone.
    """
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
    # Strided counts and ring states the layers share; slot 2 has no row.
    replay = torch.full((4, 5), 2, device="cuda", dtype=torch.int32)[:, 1:3]
    replay[:, 1] = 5
    replay[2] = torch.tensor([7, 6])
    ring_rows = window + 3
    # Each row's verify started its window at this ring row; row 3 is padding.
    ring_heads = [(window + 3 * row) % ring_rows for row in range(4)]
    snapshot_rows = torch.tensor(
        [[2, head] for head in ring_heads], device="cuda", dtype=torch.int32
    )
    expected_replay = replay.clone()
    for row in range(3):
        slot, take = int(record_slots[row]), int(accepted[row])
        expected_replay[slot] = (
            torch.tensor([0, 0])
            if slot in stopped_slots
            else torch.tensor([2 + take, (ring_heads[row] + take) % ring_rows + 1])
        )
    originals, convs, states, descriptors = [], [], [], []
    expected_convs, expected_states = [], []
    for layer in range(layers):
        pages = 7 + layer
        qkv = torch.randn(
            int(starts[-1]), channels + layer * 4, device="cuda", dtype=torch.bfloat16
        )[:, :channels]
        conv = torch.randn(
            pages, channels + layer * 3, 8, device="cuda", dtype=torch.float32
        )[:, :channels, 1:7:2]
        # The verify read each row's taps from its slot's ring rows
        # (head - 3 + i) mod (window + 3) and wrote its raw candidates to
        # rows (head + t) mod (window + 3), in the convolution state's dtype;
        # the ring rows and slots are padded.
        ring = torch.full(
            (5, ring_rows + 1, channels + 2 + layer),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )[1:, :ring_rows, :channels]
        group = int(groups[layer])
        for row in range(3):
            slot, start = int(record_slots[row]), int(starts[row])
            source = int(reads[group, row])
            for tap in range(3):
                ring[slot, (ring_heads[row] - 3 + tap) % ring_rows] = conv[
                    source, :, tap
                ]
            for token in range(int(starts[row + 1]) - start):
                ring[slot, (ring_heads[row] + token) % ring_rows] = qkv[start + token]
        state = (torch.randn(pages, heads + 1, dim, dim, device="cuda") * 0.05).to(
            state_dtype
        )[:, :heads]
        expected_conv = conv.clone()
        expected_state = state.to(torch.float32, copy=True)
        for row in range(3):
            slot = int(record_slots[row])
            if (stale and layer == 1 and row == 1) or slot not in stopped_slots:
                continue
            source, target = int(reads[group, row]), int(writes[group, row])
            take, start = int(accepted[row]), int(starts[row])
            expected_conv[target] = torch.cat(
                (conv[source], qkv[start : start + take].float().T), dim=-1
            )[:, -3:]
            expected_state[target] = _fold(
                state[target], records, slot, layer, 2 + take
            )
        descriptors.append(
            [
                ring.data_ptr(),
                conv.data_ptr(),
                pages,
                ring.stride(1),
                conv.stride(0),
                conv.stride(1),
                conv.stride(2),
                state.data_ptr(),
                *state.stride(),
                ring.stride(0),
            ]
        )
        originals.append(ring)
        convs.append(conv)
        states.append(state)
        expected_convs.append(expected_conv)
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
        replay_state=replay,
        replay_snapshot=snapshot_rows,
        request_stopped=request_stopped,
        request_ends=request_ends,
        draft_token_num=window,
        history_capacity=capacity,
        num_heads=heads,
        head_dim=dim,
        conv_width=4,
        conv_state_dtype=torch.float32,
        recurrent_state_dtype=state_dtype,
    )
    expected = (expected_convs, expected_replay, expected_states)
    return kwargs, originals, (convs, replay, states), expected


def _resolve_commit():
    commit = kda.resolve_kda_batched_retained_commit(
        torch.bfloat16, num_heads=2, head_dim=128, recurrent_layout="v_major"
    )
    assert commit is not None
    return commit


@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("stale", [False, True])
def test_batched_retained_commit_packed_groups_and_stamps(window, stale):
    args, keepalive, actual, expected = _commit_case(window, stale=stale)
    states_before = [state.clone() for state in actual[2]]
    convs_before = [conv.clone() for conv in actual[0]]
    _resolve_commit()(**args)
    # No request ended, so a stale layer reports nothing, and the rings keep
    # every request's window: no layer writes a page.
    assert args["request_ends"].tolist() == [0, 0, 0, 0]
    for value, expected_value, before in zip(actual[0], expected[0], convs_before):
        torch.testing.assert_close(value, expected_value, atol=0, rtol=0)
        torch.testing.assert_close(value, before, atol=0, rtol=0)
    # One count per request: the records its verify read plus its acceptance,
    # and the ring state past its accepted candidates. Slot 2 has no row and
    # keeps both.
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    assert actual[1][:, 0].tolist() == [2 + window, 2, 7, 2 + max(1, window - 1)]
    # Without an ended request the commit writes no dense state.
    for state, before in zip(actual[2], states_before):
        torch.testing.assert_close(state, before, atol=0, rtol=0)


@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_batched_retained_commit_folds_ended_requests(window, stale, state_dtype):
    """A request that ended in this forward leaves a dense live page.

    Slots 3 and 0 (rows 1 and 2) ended, so their records through this
    round's acceptance fold into their live pages and their count clears.
    Slot 2 ended earlier and has no row. A stale layer folds nothing.
    """
    args, keepalive, actual, expected = _commit_case(
        window, stale=stale, stopped_slots=(3, 0, 2), state_dtype=state_dtype
    )
    _resolve_commit()(**args)
    # Row 1's request ended, but its stale layer kept its records outside the
    # live page: its report says the state is incomplete.
    assert args["request_ends"].tolist() == [
        0,
        REQUEST_ENDED_INCOMPLETE if stale else REQUEST_ENDED,
        REQUEST_ENDED,
        0,
    ]
    convs, length, states = actual
    expected_convs, expected_length, expected_states = expected
    for conv, expected_conv in zip(convs, expected_convs):
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(length, expected_length, atol=0, rtol=0)
    tolerance = (
        dict(atol=2e-6, rtol=2e-6)
        if state_dtype == torch.float32
        else dict(atol=1e-5, rtol=8e-3)
    )
    for state, expected_state in zip(states, expected_states):
        torch.testing.assert_close(state, expected_state, **tolerance)


@pytest.mark.parametrize("window", [1, 4])
def test_a_commit_reads_stop_flags_through_their_stride(window):
    """The stop flags may be a strided view. The flags between its elements
    contradict slots 1 and 3, so a commit that read the view as contiguous
    would end row 0's request and keep row 1's decoding. Row 1's request
    ended with a stale stamp in layer 1, which reports it incomplete."""
    args, keepalive, actual, expected = _commit_case(
        window, stale=True, stopped_slots=(3, 0, 2)
    )
    stopped = args["request_stopped"]
    flags = torch.zeros(2 * stopped.numel(), device="cuda", dtype=torch.bool)
    flags[::2] = stopped
    flags[1::2] = torch.tensor([True, False, True, False, True], device="cuda")
    assert bool(flags[1]) != bool(stopped[1]) and bool(flags[3]) != bool(stopped[3])
    args["request_stopped"] = flags[::2]
    assert args["request_stopped"].stride(0) == 2
    _resolve_commit()(**args)
    assert args["request_ends"].tolist() == [
        0,
        REQUEST_ENDED_INCOMPLETE,
        REQUEST_ENDED,
        0,
    ]
    convs, replay, states = actual
    expected_convs, expected_replay, expected_states = expected
    # Ended slots 3 and 0 clear; slot 1 advances and slot 2 has no row.
    torch.testing.assert_close(replay, expected_replay, atol=0, rtol=0)
    assert replay[3].tolist() == [0, 0] and replay[0].tolist() == [0, 0]
    # Layers 0 and 2 materialize rows 1 and 2; layer 1 leaves row 1's page.
    for conv, expected_conv in zip(convs, expected_convs):
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    for state, expected_state in zip(states, expected_states):
        torch.testing.assert_close(state, expected_state, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("count", [-1, 8 - 4 + 1, 8, 9])
def test_a_commit_applies_no_record_of_an_overflowing_count(count):
    """A count outside [0, capacity - window] overflowed: the round's prepare
    folded its records into the live page, and its verify read none. The
    commit starts that request from zero records, so an ended request folds
    only this round's, and a decoding one counts only its acceptance."""
    for stopped in ((), (3,)):
        args, keepalive, actual, expected = _commit_case(4, stopped_slots=stopped)
        convs, replay, states = actual
        replay[3, 0] = count
        # The prepare's snapshot applies none of an overflowing count.
        args["replay_snapshot"][1, 0] = 0
        before = [state.clone() for state in states]
        _resolve_commit()(**args)
        take = 3
        assert int(replay[3, 0]) == (0 if stopped else take)
        # The ring's window after the acceptance becomes row 1's live page's
        # convolution state only when its request ended.
        for conv, expected_conv in zip(convs, expected[0]):
            torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
        if stopped:
            for layer, state in enumerate(states):
                target = int(
                    args["write_indices"][int(args["group_indices"][layer]), 1]
                )
                expected_state = _fold(
                    before[layer][target],
                    (
                        args["retained_correction"],
                        args["retained_key"],
                        args["retained_decay"],
                    ),
                    3,
                    layer,
                    take,
                )
                torch.testing.assert_close(
                    state[target], expected_state, atol=2e-6, rtol=2e-6
                )


@pytest.mark.parametrize("stopped", [True, False], ids=["masked_stopped", "padding"])
def test_rows_no_layer_carries_keep_their_count(stopped):
    """Row 1 (slot 3) has invalid pages in every group: a masked overlap row
    whose request ended earlier, or graph padding. No layer carries it, so
    it reports nothing new and its slot keeps its count."""
    args, keepalive, actual, _ = _commit_case(4, stopped_slots=(3,) if stopped else ())
    replay = actual[1]
    args["read_indices"][:, 1] = -1
    args["write_indices"][:, 1] = -1
    _resolve_commit()(**args)
    assert args["request_ends"].tolist()[1] == (REQUEST_ENDED if stopped else 0)
    assert replay[3].tolist() == [2, 5]


@pytest.mark.parametrize(
    "window,far_slot", [(1, False), (4, False), pytest.param(4, True, id="far_slot")]
)
def test_public_retained_verify_keeps_shared_source_and_materializes_endpoint(
    window, far_slot
):
    torch.manual_seed(42)
    history = _history()
    preconv = preconv_ring(history, 4, 768, width=window)
    if far_slot:
        # Two request strides fit individually in int32. The last slot's
        # element offset exceeds 2**31, in the records and in the ring. Only
        # the three short rows are touched.
        sources = (history.correction, history.key, history.decay)
        slots, heads, capacity, dim = history.correction.shape
        row_size = heads * capacity * dim
        stride = (1 << 30) + 3 * row_size
        ring_size = (window + 3) * 768
        ring_stride = (1 << 30) + 3 * ring_size
        if torch.cuda.mem_get_info()[0] < 16 << 30:
            pytest.skip("large request-slot offsets need 16 GiB free")
        ring = torch.zeros(
            2 * ring_stride + ring_size, device="cuda", dtype=torch.bfloat16
        )
        preconv = kda.KdaPreconvRing(
            ring.as_strided((slots, window + 3, 768), (ring_stride, 768, 1)),
            preconv.heads,
        )
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
        preconv=preconv,
        metadata=metadata,
        stamps=stamps,
        query_start_loc=starts,
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
    # Every row moves to a private live page, so the prepare copies them all;
    # row 2's two records stay valid on top of its copied base. No ring holds
    # rows yet, so the prepare seeds each from its source page's taps.
    counts = history.length[:, 0].tolist()
    source_2 = state[2].clone()
    prepare_history(
        state,
        history,
        reads,
        starts,
        draft_token_num=window,
        conv=conv,
        preconv=preconv,
    )
    assert snapshot(history, 4)[:3].tolist() == [[0, 3], [0, 3], [2, 3]]
    producer = {}
    # The split producer consumes the same packed boundaries as verification.
    if current_platform().is_nvidia:
        producer["conv_qkv"] = kda.kda_verify_conv_update(
            raw,
            weights,
            reads,
            live_indices=history.live_state_indices,
            preconv=preconv,
            record_slots=history.record_slots,
            page_count=state.shape[0],
            num_heads=2,
            head_dim=128,
            query_start_loc=starts,
            draft_token_num=window,
            bias=None,
            recurrent_layout="v_major",
        )
        records = replace(records, preconv=None)
    out = kda.kda_fused_paged_verify(
        raw,
        weights,
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
    # Each request's ring holds its source taps before row 3 and its window's
    # raw rows from row 3; the padding row's slot keeps its ring.
    for row in range(3):
        slot = int(history.record_slots[row])
        begin, end = int(starts[row]), int(starts[row + 1])
        torch.testing.assert_close(
            preconv.rows[slot, :3].T, conv[int(reads[row])], atol=0, rtol=0
        )
        torch.testing.assert_close(
            preconv.rows[slot, 3 : 3 + end - begin], raw[begin:end], atol=0, rtol=0
        )
    assert replay_state(history)[:, 1].tolist() == [0, 0, 0]
    torch.testing.assert_close(state[1], saved, atol=0, rtol=0)
    torch.testing.assert_close(state[3], saved, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(state[4], saved, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(state[5], source_2, atol=0, rtol=0)
    assert history.length[:, 0].tolist() == counts
    torch.testing.assert_close(
        out.reshape(total, 2, 128).float(), expected_out, atol=2e-4, rtol=1e-2
    )
    for row in range(3):
        slot = int(history.record_slots[row])
        begin, end = int(starts[row]), int(starts[row + 1])
        # Candidates follow the records the slot already holds.
        first = counts[slot]
        last = first + end - begin
        torch.testing.assert_close(
            history.correction[slot, :, first:last],
            expected_u[begin:end].transpose(0, 1),
            atol=2e-4,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            history.key[slot, :, first:last],
            expected_key[begin:end].transpose(0, 1),
            atol=1e-3,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            history.decay[slot, :, first:last],
            expected_decay[begin:end].transpose(0, 1),
            atol=1e-6,
            rtol=1e-6,
        )
        history.length[slot] = last
    endpoint = state[history.live_state_indices[:3].long()].clone()
    kda.kda_materialize_retained_state(
        endpoint,
        history=history,
        source_indices=history.live_state_indices[:3],
        recurrent_layout="v_major",
        conv_pool=conv,
        preconv=preconv.rows,
        ring_state=replay_state(history)[:, 1:],
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
    preconv = preconv_ring(history, 4, 768, width=window)
    conv = torch.randn(8, 768, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(768, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(2, 2, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(8, 2, 128, 128, device="cuda") * 0.1
    endpoint = _endpoint(state[3], history, slot, count)
    expected_out, *_ = _reference_tokens(
        endpoint, conv[3].float(), raw, weights, gate, beta, 0, 2
    )

    prepare_history(
        state,
        history,
        reads,
        starts,
        draft_token_num=window,
        conv=conv,
        preconv=preconv,
    )
    # The count stays; past capacity - window, the verify applies none of it,
    # as the snapshot says.
    assert int(history.length[slot]) == count
    assert snapshot(history, 4)[0].tolist() == [
        count if count <= capacity - window else 0,
        3,
    ]
    torch.testing.assert_close(state[3], endpoint, atol=2e-6, rtol=2e-5)

    producer = {}
    records = kda.KdaRetainedRecords(
        history=history,
        preconv=preconv,
        metadata=metadata,
        stamps=stamps,
        query_start_loc=starts,
    )
    if current_platform().is_nvidia:
        producer["conv_qkv"] = kda.kda_verify_conv_update(
            raw,
            weights,
            reads,
            live_indices=history.live_state_indices,
            preconv=preconv,
            record_slots=history.record_slots,
            page_count=state.shape[0],
            num_heads=2,
            head_dim=128,
            query_start_loc=starts,
            draft_token_num=window,
            bias=None,
            recurrent_layout="v_major",
        )
        records = replace(records, preconv=None)
    out = kda.kda_fused_paged_verify(
        raw,
        weights,
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
        retained_records=records,
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
    """The split producer convolves each request from its ring, wrapping past its end.

    Row r's slot is 3 - r, so slots and rows differ; each ring head puts the
    three past inputs and the window on different sides of the ring's last
    row. Row 3 is padding: no page and no window.
    """
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA split producer; AMD verifies packed convolution in fusion")
    torch.manual_seed(7)
    window, rows = 4, 7
    raw = torch.randn(7, 768, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(768, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    taps = torch.randn(4, 768, 3, device="cuda", dtype=torch.bfloat16)
    reads = torch.tensor([1, 2, 3, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 4, 5, 7, 7], device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 2, 1, 0], device="cuda", dtype=torch.int64)
    heads = torch.tensor([5, 3, 1, 6], device="cuda", dtype=torch.int32)
    ring = torch.full((4, rows, 768), 9.0, device="cuda", dtype=torch.bfloat16)
    for row in range(3):
        for i in range(3):
            ring[int(slots[row]), (int(heads[row]) - 3 + i) % rows] = taps[row, :, i]
    before = ring.clone()
    expected = torch.empty_like(raw)
    for row, (begin, end) in enumerate(zip(starts[:-1].tolist(), starts[1:].tolist())):
        history = taps[row].float()
        for token in range(begin, end):
            joined = torch.cat([history, raw[token, :, None].float()], dim=-1)
            expected[token] = torch.nn.functional.silu(
                (joined * weights.float()).sum(-1)
            )
            history = joined[:, 1:]
    output = kda.kda_verify_conv_update(
        raw,
        weights,
        reads,
        live_indices=reads,
        preconv=kda.KdaPreconvRing(ring, heads),
        record_slots=slots,
        page_count=6,
        num_heads=2,
        head_dim=128,
        draft_token_num=window,
        query_start_loc=starts,
        recurrent_layout="v_major",
        bias=None,
    )
    torch.testing.assert_close(output, expected, atol=0.003, rtol=0.005)
    # Each window's raw rows follow the head, wrapping; nothing else changes.
    for row, (begin, end) in enumerate(zip(starts[:-1].tolist(), starts[1:].tolist())):
        for token in range(begin, end):
            before[int(slots[row]), (int(heads[row]) + token - begin) % rows] = raw[
                token
            ]
    torch.testing.assert_close(ring, before, atol=0, rtol=0)


def _zero_history_convolution(raw, weights, history):
    """FP32 convolution and SiLU of each token after ``history`` [channels, 3]."""
    expected = torch.empty_like(raw)
    history = history.float()
    for token in range(raw.shape[0]):
        joined = torch.cat([history, raw[token, :, None].float()], dim=-1)
        expected[token] = torch.nn.functional.silu((joined * weights.float()).sum(-1))
        history = joined[:, 1:]
    return expected


def test_split_producer_takes_part_only_with_a_live_page():
    """A row with a source page but a masked live page is padding, as in the prepare.

    Row 0 takes part from slot 1 at head 3. Row 1 has source page 3 but live
    page 0 and shares slot 1 with its ring head at 0, so a ring write would
    overwrite the taps row 0 reads. Row 2 has source page 4, live page 0 and
    its own resident slot 2. Both masked rows convolve their inputs over a zero
    history and leave every ring row as it was.
    """
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA split producer; AMD verifies packed convolution in fusion")
    torch.manual_seed(13)
    window, rows, channels = 4, 7, 768
    raw = torch.randn(10, channels, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    reads = torch.tensor([1, 3, 4], device="cuda", dtype=torch.int32)
    lives = torch.tensor([2, 0, 0], device="cuda", dtype=torch.int64)
    starts = torch.tensor([0, 4, 8, 10], device="cuda", dtype=torch.int32)
    slots = torch.tensor([1, 1, 2], device="cuda", dtype=torch.int32)
    heads = torch.tensor([3, 0, 3], device="cuda", dtype=torch.int32)
    ring = torch.randn(3, rows, channels, device="cuda").to(torch.bfloat16)
    before = ring.clone()
    output = kda.kda_verify_conv_update(
        raw,
        weights,
        reads,
        live_indices=lives,
        preconv=kda.KdaPreconvRing(ring, heads),
        record_slots=slots,
        page_count=6,
        num_heads=2,
        head_dim=128,
        draft_token_num=window,
        query_start_loc=starts,
        recurrent_layout="v_major",
        bias=None,
    )
    zero = torch.zeros(channels, 3, device="cuda")
    expected = torch.cat(
        [
            _zero_history_convolution(raw[0:4], weights, before[1, 0:3].T),
            _zero_history_convolution(raw[4:8], weights, zero),
            _zero_history_convolution(raw[8:10], weights, zero),
        ]
    )
    torch.testing.assert_close(output, expected, atol=0.003, rtol=0.005)
    # Only row 0 wrote its window, to rows 3..6 of slot 1.
    before[1, 3:7] = raw[0:4]
    torch.testing.assert_close(ring, before, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 10 << 30,
    reason="ring rows past 2**31 elements need 10 GiB free",
)
def test_split_producer_addresses_ring_rows_past_int32():
    """Ring rows whose offset passes 2**31 elements convolve like near rows.

    The row pitch is 2**29 elements, so rows 4..6 of a four-token window's
    ring start at or past 2**31 elements. Head 5 reads rows 2..4 and writes
    rows 5, 6, 0 and 1; only those rows are backed by initialized values.
    """
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA split producer; AMD verifies packed convolution in fusion")
    torch.manual_seed(17)
    window, rows, channels, pitch = 4, 7, 768, 1 << 29
    assert 4 * pitch >= 2**31
    storage = torch.empty(
        (rows - 1) * pitch + channels, device="cuda", dtype=torch.bfloat16
    )
    far = storage.as_strided((1, rows, channels), (storage.numel(), pitch, 1))
    near = torch.randn(1, rows, channels, device="cuda").to(torch.bfloat16)
    for row in range(rows):
        far[0, row] = near[0, row]
    raw = torch.randn(window, channels, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    pages = torch.ones(1, device="cuda", dtype=torch.int32)
    outputs = []
    for ring in (near, far):
        outputs.append(
            kda.kda_verify_conv_update(
                raw,
                weights,
                pages,
                live_indices=pages,
                preconv=kda.KdaPreconvRing(
                    ring, torch.full((1,), 5, device="cuda", dtype=torch.int32)
                ),
                record_slots=torch.zeros(1, device="cuda", dtype=torch.int32),
                page_count=2,
                num_heads=2,
                head_dim=128,
                draft_token_num=window,
                query_start_loc=torch.tensor(
                    [0, window], device="cuda", dtype=torch.int32
                ),
                recurrent_layout="v_major",
                bias=None,
            )
        )
    assert torch.equal(outputs[1], outputs[0])
    for row in range(rows):
        assert torch.equal(far[0, row], near[0, row])
    assert torch.equal(near[0, [5, 6, 0, 1]], raw)


def test_preconv_ring_rows_and_slots_need_their_own_memory():
    """Expanded or overlapping ring layouts fail at construction, aligned or not.

    Every layout keeps the alignment the fused kernels ask of a ring (row and
    slot strides of eight elements, a 16-byte base), so only the overlap
    rejects it. One layer of a [slots, layers, rows, channels] ring and a
    slot-inner layout keep each element distinct and stay valid.
    """
    slots, rows, channels = 4, 7, 768
    storage = torch.zeros(slots * 3 * rows * channels, dtype=torch.bfloat16)
    heads = torch.full((slots,), 3, dtype=torch.int32)
    overlapping = [
        # Rows eight elements apart share 760 channels with their neighbours.
        storage.as_strided((slots, rows, channels), (rows * channels, 8, 1)),
        # Slots one row apart share all but one of their rows.
        storage.as_strided((slots, rows, channels), (channels, channels, 1)),
        # Expanded rows and expanded slots share every element.
        storage[:channels].view(1, 1, channels).expand(slots, rows, channels),
        storage[: rows * channels]
        .view(1, rows, channels)
        .expand(slots, rows, channels),
    ]
    for ring in overlapping:
        with pytest.raises(ValueError, match="must not share memory"):
            kda.KdaPreconvRing(ring, heads)
    layers = storage.view(slots, 3, rows, channels)
    kda.KdaPreconvRing(layers[:, 1], heads)
    kda.KdaPreconvRing(
        storage[: slots * rows * channels].view(rows, slots, channels).transpose(0, 1),
        heads,
    )


def test_retained_commit_graph_refresh_and_layer_subset():
    """A captured commit over a layer subset materializes the ended row it carries.

    Row 1 (slot 3) ended; the replay accepts nothing, so each launched
    layer writes the three taps its verify read, its source page's, into
    its live page. Rows that keep decoding leave their pages.
    """
    args, keepalive, (convs, length, states), _ = _commit_case(4, stopped_slots=(3,))
    commit = _resolve_commit()
    # Noncontiguous stamp and record layer views must retain their parent
    # strides; record layers follow the descriptor rows. The layers a commit
    # lists are the request's: their counts advance together.
    subset = dict(
        args,
        descriptors=args["descriptors"][::2].contiguous(),
        group_indices=args["group_indices"][::2].contiguous(),
        record_stamps=args["record_stamps"][::2],
        retained_correction=args["retained_correction"][:, ::2],
        retained_key=args["retained_key"][:, ::2],
        retained_decay=args["retained_decay"][:, ::2],
    )
    before = [[x.clone() for x in values] for values in (convs, states)]
    before_length = length.clone()

    def restore():
        for values, saved in zip((convs, states), before):
            for actual, initial in zip(values, saved):
                actual.copy_(initial)
        length.copy_(before_length)

    commit(**subset)
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit(**subset)
    restore()
    subset["accepted_length"].zero_()
    graph.replay()
    before_conv = before[0]
    for layer in (0, 2):
        group = int(subset["group_indices"][layer // 2])
        for row in range(3):
            source = int(args["read_indices"][group, row])
            target = int(args["write_indices"][group, row])
            torch.testing.assert_close(
                convs[layer][target],
                before_conv[layer][source if row == 1 else target],
                atol=0,
                rtol=0,
            )
    torch.testing.assert_close(convs[1], before_conv[1], atol=0, rtol=0)
    torch.testing.assert_close(states[1], before[1][1], atol=0, rtol=0)
    assert args["request_ends"].tolist() == [0, REQUEST_ENDED, 0, 0]
    # The replay accepted nothing, so each decoding row's count stays the
    # records its verify read; the ended row's slot clears and slot 2 has no
    # row and keeps its count.
    assert length[:, 0].tolist() == [2, 2, 7, 0]

    # Rows paged only in group 0, which no launched layer lists, are carried
    # by no layer: their counts stay, whether the request continues (row 1,
    # slot 3) or ended in this forward (row 0, slot 1). Row 2 (slot 0) is
    # paged in group 1 and advances.
    args, keepalive, (convs, length, _), _ = _commit_case(4, stopped_slots=(1,))
    args["read_indices"][1, 0] = 0
    args["write_indices"][1, 1] = 0
    subset = dict(
        args,
        descriptors=args["descriptors"][::2].contiguous(),
        group_indices=args["group_indices"][::2].contiguous(),
        record_stamps=args["record_stamps"][::2],
        retained_correction=args["retained_correction"][:, ::2],
        retained_key=args["retained_key"][:, ::2],
        retained_decay=args["retained_decay"][:, ::2],
    )
    assert subset["group_indices"].tolist() == [1, 1]
    before_conv = [x.clone() for x in convs]
    before_ends = args["request_ends"].clone()
    commit(**subset)
    assert length[:, 0].tolist() == [2 + int(args["accepted_length"][2]), 2, 7, 2]
    for layer in (0, 2):
        for row in (0, 1):
            target = int(args["write_indices"][1, row])
            torch.testing.assert_close(
                convs[layer][target], before_conv[layer][target], atol=0, rtol=0
            )
    torch.testing.assert_close(convs[1], before_conv[1], atol=0, rtol=0)
    torch.testing.assert_close(args["request_ends"], before_ends, atol=0, rtol=0)


def test_retained_commit_does_not_specialize_dynamic_metadata():
    from tokenspeed_kernel.ops.attention.kda._retained import _kernels

    args, keepalive, _, _ = _commit_case(4)
    commit = _resolve_commit()
    args["accepted_length"].zero_()
    assert args["request_ends"].data_ptr() % 16 == 0
    commit(**args)
    kernels, _ = _kernels()
    # A mixed forward's decode rows follow its extend rows in the stop
    # report: their contiguous view starts one INT32 past a 16-byte boundary.
    ends = torch.zeros(5, device="cuda", dtype=torch.int32)[1:]
    assert ends.is_contiguous() and ends.data_ptr() % 16 != 0
    offset = dict(args, request_ends=ends)
    shorter = dict(
        args,
        read_indices=args["read_indices"][:, :2],
        write_indices=args["write_indices"][:, :2],
        accepted_length=args["accepted_length"][:2],
        query_start_loc=args["query_start_loc"][:3],
        record_slots=args["record_slots"][:2],
        record_metadata=args["record_metadata"][:2],
        record_stamps=torch.full((3, 5, 11, 2), -1, device="cuda", dtype=torch.int64),
        replay_snapshot=args["replay_snapshot"][:2],
        request_ends=args["request_ends"][:2],
    )
    with assert_no_triton_compile(kernels.retained_batched_commit_kernel):
        commit(**offset)
        commit(**shorter)
        # The stop flags' element stride is a runtime scalar too.
        for step in (2, 3):
            flags = torch.zeros(5 * step, device="cuda", dtype=torch.bool)[::step]
            commit(**dict(args, request_stopped=flags))
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
                    replay_snapshot=torch.zeros(
                        batch, 2, device="cuda", dtype=torch.int32
                    ),
                    request_ends=torch.zeros(batch, device="cuda", dtype=torch.int32),
                )
            )


def _single_layer_commit(batch, ended_rows, *, window=4, ring_pitch=None):
    """One layer commits ``batch`` rows; the requests of ``ended_rows`` ended.

    Row r reads page r + 1, writes page batch + r + 1 and owns slot
    batch - 1 - r, whose ring holds the taps its verify read before its head
    and its candidates from the head, wrapping; it accepts 1 + r % window
    candidates after r % 3 records. ``ring_pitch`` spaces the slots' rings
    that many elements apart. Returns the commit's arguments, the pages and
    ring, and each row's expected convolution window.
    """
    heads, dim, channels, capacity = 2, 128, 768, 8
    rows = window + 3
    pages = 2 * batch + 2
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    writes = reads + batch
    slots = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int64)
    starts = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * window
    accepted = 1 + torch.arange(batch, device="cuda", dtype=torch.int32) % window
    heads_at = [(5 * row) % rows for row in range(batch)]
    old = [row % 3 for row in range(batch)]
    snapshot_rows = torch.tensor(
        [[o, h] for o, h in zip(old, heads_at)], device="cuda", dtype=torch.int32
    )
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 3
    metadata[:, 2] = 2**33 + 1
    stamps = torch.full((1, batch, window, 2), -1, device="cuda", dtype=torch.int64)
    for token in range(window):
        position = 2**33 + 1 + token
        stamps[0, :, position % window] = torch.tensor([position, 3], device="cuda")
    conv = torch.randn(pages, channels, 3, device="cuda").to(torch.bfloat16)
    state = torch.randn(pages, heads, dim, dim, device="cuda") * 0.05
    if ring_pitch is None:
        ring = torch.randn(batch, rows, channels, device="cuda").to(torch.bfloat16)
    else:
        storage = torch.empty(
            (batch - 1) * ring_pitch + rows * channels,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ring = storage.as_strided((batch, rows, channels), (ring_pitch, channels, 1))
        for slot in range(batch):
            ring[slot] = torch.randn(rows, channels, device="cuda").to(torch.bfloat16)
    windows = []
    for row in range(batch):
        slot, head = batch - 1 - row, heads_at[row]
        last = head + int(accepted[row])
        windows.append(ring[slot, [(last - 3 + i) % rows for i in range(3)]].T.clone())
    shape = (batch, 1, heads, capacity, dim)
    records = (
        torch.randn(shape, device="cuda") * 0.03,
        torch.randn(shape, device="cuda") * 0.03,
        torch.rand(shape, device="cuda") * 0.2 + 0.7,
    )
    request_stopped = torch.zeros(batch, device="cuda", dtype=torch.bool)
    request_stopped[[batch - 1 - row for row in ended_rows]] = True
    args = dict(
        descriptors=torch.tensor(
            [
                [
                    ring.data_ptr(),
                    conv.data_ptr(),
                    pages,
                    ring.stride(1),
                    conv.stride(0),
                    conv.stride(1),
                    conv.stride(2),
                    state.data_ptr(),
                    *state.stride(),
                    ring.stride(0),
                ]
            ],
            device="cuda",
            dtype=torch.uint64,
        ),
        group_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        read_indices=reads[None],
        write_indices=writes[None],
        accepted_length=accepted,
        query_start_loc=starts,
        record_slots=slots,
        record_slot_count=batch,
        record_metadata=metadata,
        record_stamps=stamps,
        retained_correction=records[0],
        retained_key=records[1],
        retained_decay=records[2],
        replay_state=torch.zeros(batch, 2, device="cuda", dtype=torch.int32),
        replay_snapshot=snapshot_rows,
        request_stopped=request_stopped,
        request_ends=torch.where(
            torch.tensor([row in ended_rows for row in range(batch)], device="cuda"),
            REQUEST_ENDED,
            0,
        ).to(torch.int32),
        draft_token_num=window,
        history_capacity=capacity,
        num_heads=heads,
        head_dim=dim,
        conv_width=4,
        conv_state_dtype=torch.bfloat16,
        recurrent_state_dtype=torch.float32,
    )
    return args, (conv, state, ring), windows


def _check_single_layer_commit(args, pages, windows, ended_rows, before):
    conv, state, _ = pages
    conv_before, state_before = before
    batch = args["accepted_length"].numel()
    window = args["draft_token_num"]
    records = (
        args["retained_correction"],
        args["retained_key"],
        args["retained_decay"],
    )
    expected_conv, expected_state = conv_before.clone(), state_before.clone()
    expected_replay = torch.zeros(batch, 2, dtype=torch.int32)
    for row in range(batch):
        slot, target = batch - 1 - row, int(args["write_indices"][0, row])
        old, head = args["replay_snapshot"][row].tolist()
        take = int(args["accepted_length"][row])
        if row in ended_rows:
            expected_conv[target] = windows[row]
            expected_state[target] = _fold(
                state_before[target], records, slot, 0, old + take
            )
        else:
            expected_replay[slot] = torch.tensor(
                [old + take, (head + take) % (window + 3) + 1]
            )
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        args["replay_state"].cpu(), expected_replay, atol=0, rtol=0
    )
    assert args["request_ends"].tolist() == [
        REQUEST_ENDED if row in ended_rows else 0 for row in range(batch)
    ]


@pytest.mark.parametrize(
    "batch,ended_rows",
    [(1, ()), (1, (0,)), (5, (0, 3, 4)), (260, (1, 255, 256, 259))],
    ids=["one_decoding", "one_ended", "five", "two_row_blocks"],
)
def test_one_commit_launch_publishes_and_materializes_only_ended_rows(
    batch, ended_rows
):
    """The commit is a single launch; only ended requests' pages change.

    Every request gets its count and ring state from the snapshot and its
    acceptance, or both cleared when it ended. An ended request's live page
    gets its records folded and the ring's window after its acceptance,
    which wraps past the ring's last row for some rows, as its taps; every
    other page keeps its state. With 260 rows the end programs scan two
    blocks of stop flags, ended rows on both sides of the boundary.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    args, pages, windows = _single_layer_commit(batch, ended_rows)
    before = (pages[0].clone(), pages[1].clone())
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        _resolve_commit()(**args)
        torch.cuda.synchronize()
    launched = [
        event.name for event in trace.events() if event.device_type == DeviceType.CUDA
    ]
    assert len(launched) == 1 and "retained_batched_commit" in launched[0]
    _check_single_layer_commit(args, pages, windows, ended_rows, before)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 10 << 30,
    reason="ring slots past 2**31 elements need 10 GiB free",
)
def test_an_ended_request_on_a_far_ring_slot_materializes_like_a_near_one():
    """Ring slots 2**31 elements apart: the far slot's window is written whole.

    Row 0 owns slot 1, whose ring starts at element 2**31; it ends. Row 1 owns
    slot 0 and keeps decoding.
    """
    args, pages, windows = _single_layer_commit(2, (0,), ring_pitch=1 << 31)
    assert pages[2].stride(0) >= 2**31
    before = (pages[0].clone(), pages[1].clone())
    _resolve_commit()(**args)
    _check_single_layer_commit(args, pages, windows, (0,), before)


def _prepare_case(window, *, state_dtype=torch.float32):
    """Three layers in two State groups, each layer with its own arena strides.

    Every layer of a request reads the count its slot keeps, and a count
    past capacity - window overflows. Row 0 moves from a shared source page
    to its private live page in both groups and fits: its live pages get the
    source state, with no record applied. Row 1 continues its own pages and
    overflows the capacity by one record: it folds in place. Row 2 continues
    its own pages and fits exactly. Rows 3-6 are padding -- an empty window,
    a source one past the pool, a slot past the record slots, the null page
    -- and keep their pages. Row 7 continues its own pages with fewer tokens
    than the window: its own tokens fit, but it folds whenever a full window
    would overflow. Row 8 moves to a private page in group 0 only and fits:
    the group-0 layer copies its source, and the group-1 layers leave it.

    Even slots hold no pre-convolution rows yet (ring state 0): each layer
    that carries their row seeds ring rows 0..2 from its source page's taps.
    Odd slots' rings are seeded, and stay as they are. Each row's snapshot
    gets the count its verify applies and its ring head.
    """
    heads, dim, layers, capacity, slots, pages, rows = 2, 128, 3, 8, 8, 10, 9
    channels, ring_rows = 3 * heads * dim, window + 3
    groups = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    # [groups, rows] views of 16-row tables, like serving's padded per-batch
    # tables: every row starts on a 16-byte boundary.
    tables = torch.full((2, 2, 16), -1, device="cuda", dtype=torch.int32)
    sources, lives = tables[0, :, :rows], tables[1, :, :rows]
    # Each group names its own pages; live pages are private within a group.
    sources.copy_(
        torch.tensor([[1, 2, 3, 4, pages, 1, 0, 4, 7], [2, 6, 7, 4, pages, 2, 0, 4, 3]])
    )
    lives.copy_(
        torch.tensor([[5, 2, 3, 6, 7, 8, 9, 4, 8], [9, 6, 7, 5, 8, 1, 3, 4, 3]])
    )
    sizes = [window, window, window, 0, window, window, window, max(1, window - 3)]
    sizes.append(window)
    starts = torch.tensor(
        [0] + torch.tensor(sizes).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    record_slots = torch.tensor([2, 4, 0, 1, 3, slots, 5, 6, 7], device="cuda")
    counts = {
        0: capacity - window,
        1: 5,
        2: 3,
        3: 6,
        4: capacity - window + 1,
        5: 7,
        6: capacity - window + 1,
        7: 2,
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
    # Strided counts and ring states the layers share.
    replay = torch.zeros(slots, 5, device="cuda", dtype=torch.int32)[:, 1:3]
    for slot, count in counts.items():
        replay[slot] = torch.tensor(
            [count, 0 if slot % 2 == 0 else slot % ring_rows + 1]
        )
    expected_snapshot = torch.zeros(rows, 2, device="cuda", dtype=torch.int32)
    for row in range(rows):
        slot = int(record_slots[row])
        count, ring_state = replay[slot].tolist() if slot < slots else (0, 0)
        expected_snapshot[row, 0] = count if 0 <= count <= capacity - window else 0
        expected_snapshot[row, 1] = ring_state - 1 if ring_state else 3
    states, descriptors, expected_states = [], [], []
    convs, rings, expected_rings = [], [], []
    for layer in range(layers):
        state = (
            torch.randn(pages, heads + 1 + layer, dim, dim, device="cuda") * 0.05
        ).to(state_dtype)[:, :heads]
        expected_state = state.clone()
        conv = torch.randn(
            pages, channels + 2 * layer, 3, device="cuda", dtype=torch.bfloat16
        )[:, :channels]
        ring = torch.full(
            (slots, ring_rows + 1, channels + 3), -3.0, device="cuda"
        ).bfloat16()[:, 1:, :channels]
        expected_ring = ring.clone()
        group = int(groups[layer])
        for row in range(rows):
            source, live = int(sources[group, row]), int(lives[group, row])
            slot, tokens = int(record_slots[row]), sizes[row]
            if not (
                0 < source < pages and 0 < live < pages and tokens and slot < slots
            ):
                continue
            if slot % 2 == 0:
                expected_ring[slot, :3] = conv[source].T
            if counts[slot] > capacity - window:
                expected_state[live] = _fold(
                    state[source], records, slot, layer, counts[slot]
                ).to(state_dtype)
            elif source != live:
                expected_state[live] = state[source]
        descriptors.append(
            [
                ring.data_ptr(),
                conv.data_ptr(),
                pages,
                ring.stride(1),
                conv.stride(0),
                conv.stride(1),
                conv.stride(2),
                state.data_ptr(),
                *state.stride(),
                ring.stride(0),
            ]
        )
        states.append(state)
        convs.append(conv)
        expected_states.append(expected_state)
        rings.append(ring)
        expected_rings.append(expected_ring)
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
        replay_state=replay,
        replay_snapshot=torch.full((rows, 2), -9, device="cuda", dtype=torch.int32),
        history_capacity=capacity,
        draft_token_num=window,
        num_heads=heads,
        head_dim=dim,
        conv_state_dtype=torch.bfloat16,
        recurrent_state_dtype=state_dtype,
    )
    # The descriptors address the convolution pools and rings: keep them alive.
    seeding = (rings, expected_rings, expected_snapshot, convs)
    return args, states, expected_states, seeding


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
    args, states, expected_states, seeding = _prepare_case(
        window, state_dtype=state_dtype
    )
    rings, expected_rings, expected_snapshot, _ = seeding
    fields = ("retained_correction", "retained_key", "retained_decay")
    records = [args[name].clone() for name in fields]
    replay = args["replay_state"].clone()
    initial = [x.clone() for x in states]
    prepare = _resolve_prepare()
    prepare(**args)
    if captured:
        for value, before in zip(states, initial):
            value.copy_(before)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare(**args)
        for value, before in zip(states, initial):
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
    # A move that fits copies the dense base and applies no record: row 0's
    # live pages and row 8's group-0 live page hold their sources exactly.
    torch.testing.assert_close(states[1][5], initial[1][1], atol=0, rtol=0)
    torch.testing.assert_close(states[1][8], initial[1][7], atol=0, rtol=0)
    for layer in (0, 2):
        torch.testing.assert_close(states[layer][9], initial[layer][2], atol=0, rtol=0)
        # Row 8 did not move in group 1, so its layers leave its page.
        torch.testing.assert_close(states[layer][3], initial[layer][3], atol=0, rtol=0)
    # The commit writes the counts and ring states; a prepare only reads them.
    torch.testing.assert_close(args["replay_state"], replay, atol=0, rtol=0)
    for name, before in zip(fields, records):
        torch.testing.assert_close(args[name], before, atol=0, rtol=0)
    # Unseeded rings of carried rows hold their source taps; others are kept.
    for ring, expected_ring in zip(rings, expected_rings):
        torch.testing.assert_close(ring, expected_ring, atol=0, rtol=0)
    torch.testing.assert_close(
        args["replay_snapshot"], expected_snapshot, atol=0, rtol=0
    )


def test_batched_retained_prepare_does_not_specialize_dynamic_metadata():
    from tokenspeed_kernel.ops.attention.kda._triton.retained import (
        retained_batched_prepare_kernel,
    )

    args, _, _, keepalive = _prepare_case(4)
    prepare = _resolve_prepare()
    prepare(**args)
    with assert_no_triton_compile(retained_batched_prepare_kernel):
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
                    replay_snapshot=torch.zeros(
                        batch, 2, device="cuda", dtype=torch.int32
                    ),
                )
            )


@pytest.mark.parametrize("count", [-1, 9])
def test_a_count_outside_the_history_applies_no_record(count):
    """A count below 0 or past the capacity overflows, so the verify applies
    no record, but the prepare has no records to fold: a row that keeps its
    page leaves it as it is, and a row that moves gets its source copied."""
    args, states, _, keepalive = _prepare_case(4)
    # Row 2 (slot 0) keeps its pages; row 0 (slot 2) moves in both groups.
    args["replay_state"][[0, 2], 0] = count
    before = [state.clone() for state in states]
    _resolve_prepare()(**args)
    assert args["replay_state"][[0, 2], 0].tolist() == [count, count]
    # Their verifies apply none of the count.
    assert args["replay_snapshot"][[0, 2], 0].tolist() == [0, 0]
    for layer, state in enumerate(states):
        kept = 3 if layer == 1 else 7
        torch.testing.assert_close(state[kept], before[layer][kept], atol=0, rtol=0)
        source, live = (1, 5) if layer == 1 else (2, 9)
        torch.testing.assert_close(state[live], before[layer][source], atol=0, rtol=0)


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
    # Past capacity - window, so the prepare folds the records.
    history.length.fill_(5)
    reads = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.int32)
    history.live_state_indices.copy_(
        torch.tensor([2, 0, 0, 0], device="cuda", dtype=torch.int32)
    )
    starts = torch.tensor([0, 4, 4, 4, 4], device="cuda", dtype=torch.int32)
    slot = int(history.record_slots[0])
    expected = _endpoint(state[1].float(), history, slot, 5).to(torch.bfloat16)
    source = state[1].clone()
    prepare_history(state, history, reads, starts, draft_token_num=4)
    torch.testing.assert_close(state[2], expected, atol=2e-3, rtol=4e-3)
    torch.testing.assert_close(state[1], source, atol=0, rtol=0)
    assert int(history.length[slot]) == 5


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


def test_invalid_ring_verify_raises_without_mutation():
    """A ring that does not keep the window's rows fails before touching any
    page, ring or record."""
    history = _history(heads=1, dim=128)
    reads = torch.tensor([1, 1, 2, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 2, 4, 6, 6], device="cuda", dtype=torch.int32)
    raw = torch.randn(6, 384, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(7, 1, 128, 128, device="cuda")
    # Four rows a slot: a one-token window's ring, not a two-token one's.
    ring = torch.full((3, 4, 384), 7.0, device="cuda", dtype=torch.bfloat16)
    metadata = torch.tensor(
        [[0, 1, 0], [1, 1, 0], [2, 1, 0], [-1, -1, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.full((3, 8, 2), -1, device="cuda", dtype=torch.int64)
    records = kda.KdaRetainedRecords(
        kda.KdaPreconvRing(ring, torch.full((4,), 3, device="cuda", dtype=torch.int32)),
        history,
        metadata,
        stamps,
        starts,
    )
    mutable = (
        raw,
        state,
        ring,
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
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
            torch.zeros(6, 1, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, device="cuda"),
            torch.zeros(128, device="cuda"),
            state_pool=state,
            read_indices=reads,
            num_heads=1,
            head_dim=128,
            draft_token_num=2,
            recurrent_layout="v_major",
            retained_records=records,
            output_norm=None,
            g_raw=torch.zeros(6, 128, device="cuda", dtype=torch.bfloat16),
            lower_bound=-5.0,
            override=None,
            solution=None,
        )
    for actual, expected in zip(mutable, before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
