"""KDA pre-convolution ring lifecycle through the real backend, pool and kernels.

Each decoding request slot keeps a ring of window + 3 raw pre-convolution QKV
rows per layer. Between materializations the ring, not the request's page,
holds its convolution window; the page is written at a request's end, a
retraction fold and a prefill continuation. These tests drive whole rounds
(prepare, verify, commit) through ``KdaAttnBackend`` and compare them with the
torch reference and, where it applies, with the ring disabled by
construction: after every round the ring's window is written into the live
page and the ring is unseeded, so the next prepare reseeds it from the page.
Both must give the same bits.
"""

import gc
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# CI registration and absolute test imports use the same paths as the suite.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, suite="runtime-1gpu")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import block_tables_for as _tables_for
from test.runtime.conftest import kimi_recipe as _kimi_recipe
from test.runtime.test_kimi_k3_kda_eager_commit import (
    DEV,
    _Harness,
    _logical_conv,
    _logical_state,
    _memory_saver_kimi_pool,
    _reference_round,
)

from tokenspeed_kernel.ops.attention.kda import (
    KdaPreconvRing,
    KdaRetainedHistory,
    kda_materialize_retained_pages,
    kda_verify_conv_update,
)
from tokenspeed_kernel.ops.metadata import REQUEST_ENDED, REQUEST_ENDED_INCOMPLETE

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
    kda_request_slot_bytes,
)

D = 128
# Kimi-K3 at TP8: 69 KDA layers of 12 heads per rank, 3 * 12 * 128 channels.
K3_LAYERS, K3_TP8_CHANNELS = 69, 3 * 12 * D


def _bits(tensor):
    """The raw bits of ``tensor``, so a comparison is exact (signed zeros too)."""
    integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return tensor.contiguous().view(integer[tensor.element_size()])


def _assert_same_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(_bits(actual), _bits(expected))


def _assert_same_outputs(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        _assert_same_bits(got, want)


def _pages(rpis, first=2):
    """Distinct pages per State group: the groups share page memory."""
    return {
        group: [first + index * len(rpis) + row for row in range(len(rpis))]
        for index, group in enumerate(_STATE_GROUPS)
    }


def _page_values(harness, pages, seed):
    """Random convolution taps and recurrent states for ``pages`` in every layer."""
    generator = torch.Generator().manual_seed(seed)
    values = {}
    for layer in harness.backend._state_layer_ids():
        ids = pages[harness.pool.state_group_by_layer[layer]]
        conv = harness.pool.get_component(layer, "conv_state")
        recurrent = harness.pool.get_component(layer, "recurrent_state")
        values[layer] = (
            torch.randn((len(ids), *conv.shape[1:]), generator=generator).to(
                DEV, conv.dtype
            ),
            (
                torch.randn((len(ids), *recurrent.shape[1:]), generator=generator)
                * 0.05
            ).to(DEV, recurrent.dtype),
        )
    return values


def _write_pages(harness, pages, values):
    """Write ``values`` into ``pages``, as a transfer or a restore lands them."""
    for layer, (conv, recurrent) in values.items():
        ids = pages[harness.pool.state_group_by_layer[layer]]
        harness.pool.get_component(layer, "conv_state")[ids] = conv
        harness.pool.get_component(layer, "recurrent_state")[ids] = recurrent


def _copy_pages(harness, sources, destinations):
    """Copy every layer's state from ``sources`` to ``destinations``, as an image does."""
    for layer in harness.backend._state_layer_ids():
        group = harness.pool.state_group_by_layer[layer]
        for name in ("conv_state", "recurrent_state"):
            component = harness.pool.get_component(layer, name)
            component[destinations[group]] = component[sources[group]]


def _decode_round(harness, rpis, pages, seq_lens, inputs, accepted, *, sources=None):
    """Refresh (prepare), verify every layer and commit one decode round.

    ``sources`` names each row's input page per group when it differs from
    its live page (a page move); otherwise the table's input column is 0.
    """
    tables = {
        group: np.asarray(
            [
                [0 if sources is None else sources[group][row], page]
                for row, page in enumerate(ids)
            ],
            dtype=np.int32,
        )
        for group, ids in pages.items()
    }
    batch = len(rpis)
    harness.backend.refresh_decode_metadata(
        batch,
        batch,
        torch.tensor(rpis, dtype=torch.int32, device=DEV),
        torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.DECODE,
        block_tables=_tables_for(harness.contract, tables, DEV),
    )
    outputs = harness.forward(inputs, batch)
    harness.backend.commit_verified_state(
        torch.tensor(accepted, dtype=torch.int32, device=DEV), accepted_path=None
    )
    return outputs


def _disable_ring(harness, rpis, pages):
    """Write each slot's ring window into its live page and unseed its ring.

    The materialization kernel writes the window (no record is folded: the
    history it gets counts none), and the ring state goes to 0, so the next
    round's prepare reseeds the ring from the page: the page holds every
    round's convolution state, as when each round committed it.
    """
    backend = harness.backend
    slots = torch.tensor(rpis, dtype=torch.int64, device=DEV)
    no_records = torch.zeros_like(backend._replay_length)
    for layer in backend._state_layer_ids():
        row = backend._descriptor_row_by_layer[layer]
        listed = torch.tensor(
            pages[harness.pool.state_group_by_layer[layer]],
            dtype=torch.int32,
            device=DEV,
        )
        correction, key, decay = backend._retained_records[:, row].unbind(1)
        conv, recurrent = backend._state_components(layer)
        kda_materialize_retained_pages(
            recurrent,
            history=KdaRetainedHistory(
                correction, key, decay, no_records, listed, slots
            ),
            source_indices=listed,
            recurrent_layout=backend.kda_recurrent_layout,
            conv_pool=conv,
            preconv=backend._preconv_ring[:, row],
            ring_state=backend._replay_state[:, 1:],
        )
    backend._replay_state[rpis, 1] = 0


def _conv_pages(harness, pages):
    return {
        layer: harness.pool.get_component(layer, "conv_state")[
            pages[harness.pool.state_group_by_layer[layer]]
        ].clone()
        for layer in harness.backend._state_layer_ids()
    }


def _assert_same_requests(actual, expected, actual_pages, expected_pages, slots):
    """Two harnesses keep the same logical requests: windows, states, counts."""
    actual_slots, expected_slots = slots
    assert (
        actual.backend._replay_length[actual_slots].tolist()
        == expected.backend._replay_length[expected_slots].tolist()
    )
    # Records past a slot's count are rejected candidates or another
    # owner's: only the counted ones are the request's.
    counts = actual.backend._replay_length[actual_slots].flatten().tolist()
    for actual_slot, expected_slot, count in zip(
        actual_slots, expected_slots, counts, strict=True
    ):
        _assert_same_bits(
            actual.backend._retained_records[actual_slot, ..., :count, :],
            expected.backend._retained_records[expected_slot, ..., :count, :],
        )
    for layer in actual.backend._state_layer_ids():
        group = actual.pool.state_group_by_layer[layer]
        _assert_same_bits(
            _logical_conv(actual, layer, actual_pages[group], actual_slots),
            _logical_conv(expected, layer, expected_pages[group], expected_slots),
        )
        _assert_same_bits(
            actual.pool.get_component(layer, "recurrent_state")[actual_pages[group]],
            expected.pool.get_component(layer, "recurrent_state")[
                expected_pages[group]
            ],
        )


def _check_reference(harness, layer, pages, rpis, expected, *, raw_gate):
    """The round's committed window (exact) and state follow the torch reference."""
    group = harness.pool.state_group_by_layer[layer]
    state, conv = expected[layer]
    _assert_same_bits(_logical_conv(harness, layer, pages[group], rpis), conv)
    if raw_gate:
        torch.testing.assert_close(
            _logical_state(harness, layer, pages[group], rpis),
            state,
            atol=2e-6,
            rtol=2e-5,
        )


def _tracked_round(harness, tracked, inputs, accepted):
    """Advance an independent torch reference of every tracked layer one round.

    ``tracked`` maps each layer to its requests' convolution windows and FP32
    recurrent states. A round starts from the previous round's reference,
    never from the device, so an accepted update a kernel drops stays
    missing from the device and is caught. Returns each layer's outputs and
    the advanced ``tracked``.
    """
    from test.runtime.kda_torch_reference import (
        checked_convolution,
        kda_torch_reference,
    )

    batch = len(accepted)
    raw = inputs["mixed_qkv"].view(batch, -1, harness.conv_dim)
    rows = torch.arange(batch, device=DEV)
    counts = torch.tensor(accepted, device=DEV).clamp(min=1) - 1
    outputs, advanced = {}, {}
    for layer, (conv, state) in tracked.items():
        params = harness.params[layer]
        gate = inputs.get("g_raw")
        if gate is None:
            gate = torch.nn.functional.linear(inputs["f_a_out"], params["f_b_weight"])
        weights = dict(conv_weights=params["conv_weights"], bias=params.get("bias"))
        output, tape, windows = kda_torch_reference(
            raw,
            conv,
            state,
            inputs["beta_raw"],
            gate=gate,
            A_log=params["A_log"],
            dt_bias=params["dt_bias"],
            lower_bound=params["lower_bound"],
            convolved=checked_convolution(raw, conv, prefill=False, **weights)[0],
            **weights,
        )
        outputs[layer] = output.reshape(-1, harness.H, D)
        advanced[layer] = (windows[rows, counts], tape[rows, counts])
    return outputs, advanced


def _check_tracked(harness, pages, slots, tracked, outputs, expected_outputs):
    """Every layer's window (exact), recurrent state and outputs follow the
    tracked reference."""
    for layer, output in zip(harness.layer_ids, outputs, strict=True):
        conv, state = tracked[layer]
        ids = pages[harness.pool.state_group_by_layer[layer]]
        _assert_same_bits(_logical_conv(harness, layer, ids, slots), conv)
        torch.testing.assert_close(
            _logical_state(harness, layer, ids, slots), state, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            output.reshape_as(expected_outputs[layer]),
            expected_outputs[layer],
            atol=2e-3,
            rtol=2e-3,
        )


def _assert_dense_pages_follow(harness, pages, tracked):
    """``pages`` themselves hold the tracked windows (exact) and recurrent
    states, as a materialized request's pages must: nothing outside them
    (records, rings) is read."""
    for layer, (conv, state) in tracked.items():
        ids = pages[harness.pool.state_group_by_layer[layer]]
        _assert_same_bits(harness.pool.get_component(layer, "conv_state")[ids], conv)
        torch.testing.assert_close(
            harness.pool.get_component(layer, "recurrent_state")[ids].float(),
            state,
            atol=2e-6,
            rtol=2e-5,
        )


def _inputs(harness, batch, seed, *, raw_gate):
    inputs = harness.inputs(batch, seed)
    if raw_gate:
        # Keep the gate projection's rounding outside the state comparison;
        # a raw gate also routes the verify through the split producers.
        inputs["g_raw"] = inputs["mixed_qkv"][:, : harness.key_dim]
    return inputs


@pytest.mark.parametrize("raw_gate", [False, True], ids=["fused", "split"])
@pytest.mark.parametrize("draft_tokens", [1, 2, 4])
def test_acceptance_patterns_wrap_the_ring_like_a_reset_every_round(
    draft_tokens, raw_gate
):
    """Full, minimal (0, clamped to one) and partial acceptance over more than
    two ring turns (R = T + 3 rows): every round's outputs, records, counts,
    dense states and windows match a run whose ring is reset every round bit
    for bit, the windows match the torch reference exactly, and the pages'
    convolution state is never written while the requests decode."""
    steady = _Harness(draft_tokens=draft_tokens, seed=3101)
    reset = _Harness(draft_tokens=draft_tokens, seed=3101)
    reset.params = steady.params
    width = steady.width
    rows = width + 3
    rpis = [3, 1, 4]
    pages = _pages(rpis)
    values = _page_values(steady, pages, 3102)
    for harness in (steady, reset):
        _write_pages(harness, pages, values)
    untouched = _conv_pages(steady, pages)
    seq_lens = [8 + width] * len(rpis)
    heads = [3] * len(rpis)
    turns = [0] * len(rpis)
    first = steady.layer_ids[0]
    for round_index in range(2 * rows + 2):
        accepted = [width, 0, 1 + round_index % width]
        inputs = _inputs(steady, len(rpis), 3110 + round_index, raw_gate=raw_gate)
        expected_outputs, expected = _reference_round(
            steady, pages, inputs, accepted, slots=rpis, prefill=False, layers=[first]
        )
        outputs = _decode_round(steady, rpis, pages, seq_lens, inputs, accepted)
        _assert_same_outputs(
            outputs, _decode_round(reset, rpis, pages, seq_lens, inputs, accepted)
        )
        _disable_ring(reset, rpis, pages)
        takes = [max(1, take) for take in accepted]
        turns = [
            turn + (head + take) // rows
            for turn, head, take in zip(turns, heads, takes)
        ]
        heads = [(head + take) % rows for head, take in zip(heads, takes)]
        assert steady.backend._replay_state[rpis, 1].tolist() == [h + 1 for h in heads]
        _assert_same_requests(steady, reset, pages, pages, (rpis, rpis))
        _check_reference(steady, first, pages, rpis, expected, raw_gate=raw_gate)
        if raw_gate:
            torch.testing.assert_close(
                outputs[0].reshape_as(expected_outputs[0]),
                expected_outputs[0],
                atol=2e-3,
                rtol=2e-3,
            )
        seq_lens = [length + take for length, take in zip(seq_lens, takes)]
    # Every row's head went round the ring at least twice.
    assert min(turns) >= 2
    for layer, conv in untouched.items():
        _assert_same_bits(
            steady.pool.get_component(layer, "conv_state")[
                pages[steady.pool.state_group_by_layer[layer]]
            ],
            conv,
        )


@pytest.mark.parametrize("history_capacity", [4, 8], ids=["one_window", "two_windows"])
def test_records_overflow_folds_while_the_ring_keeps_its_window(history_capacity):
    """A count past capacity - window folds the records in the prepare; the
    fold writes the recurrent state only, so the ring head keeps advancing,
    the pages' convolution state stays as it was, and every round matches
    the ring-reset run bit for bit and the torch reference."""
    width = 4
    steady = _Harness(draft_tokens=width, seed=3201, history_capacity=history_capacity)
    reset = _Harness(draft_tokens=width, seed=3201, history_capacity=history_capacity)
    reset.params = steady.params
    assert steady.backend._retained_capacity == history_capacity
    rows = width + 3
    rpis = [2, 5]
    pages = _pages(rpis)
    values = _page_values(steady, pages, 3202)
    for harness in (steady, reset):
        _write_pages(harness, pages, values)
    untouched = _conv_pages(steady, pages)
    seq_lens = [16] * len(rpis)
    counts, heads, folds = [0, 0], [3, 3], 0
    first = steady.layer_ids[0]
    for round_index in range(10):
        accepted = [width, 1 + (round_index * 3) % width]
        inputs = _inputs(steady, len(rpis), 3210 + round_index, raw_gate=True)
        _, expected = _reference_round(
            steady, pages, inputs, accepted, slots=rpis, prefill=False, layers=[first]
        )
        _assert_same_outputs(
            _decode_round(steady, rpis, pages, seq_lens, inputs, accepted),
            _decode_round(reset, rpis, pages, seq_lens, inputs, accepted),
        )
        _disable_ring(reset, rpis, pages)
        overflow = [count + width > history_capacity for count in counts]
        folds += sum(overflow)
        counts = [
            (0 if over else count) + take
            for count, over, take in zip(counts, overflow, accepted)
        ]
        heads = [(head + take) % rows for head, take in zip(heads, accepted)]
        assert steady.backend._replay_state[rpis].tolist() == [
            [count, head + 1] for count, head in zip(counts, heads)
        ]
        _assert_same_requests(steady, reset, pages, pages, (rpis, rpis))
        _check_reference(steady, first, pages, rpis, expected, raw_gate=True)
        for layer, conv in untouched.items():
            _assert_same_bits(
                steady.pool.get_component(layer, "conv_state")[
                    pages[steady.pool.state_group_by_layer[layer]]
                ],
                conv,
            )
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    assert folds >= 4


@pytest.mark.parametrize("moves", ["every_round", "once"])
def test_page_moves_leave_the_ring_and_the_pages_convolution_alone(moves):
    """A request whose input page differs from its live page gets its dense
    base copied there and keeps its ring: the round matches a run whose
    pages never move bit for bit, though the new live page's convolution
    state holds garbage the verify must not read. The page is written only
    when the request ends, and then holds the window the static run's does."""
    width = 4
    moved = _Harness(draft_tokens=width, seed=3301)
    static = _Harness(draft_tokens=width, seed=3301)
    static.params = moved.params
    rpis = [1, 3]
    homes = _pages(rpis)
    # Two page sets per row; a move swaps the input and live set.
    sets = (homes, _pages(rpis, first=2 + 2 * len(_STATE_GROUPS) * len(rpis)))
    values = _page_values(moved, homes, 3302)
    _write_pages(moved, homes, values)
    _write_pages(static, homes, values)
    seq_lens = [12] * len(rpis)
    live = 0
    for round_index in range(9):
        accepted = [1 + round_index % width, width]
        if round_index == 8:
            for harness in (moved, static):
                harness.stopped[rpis[0]] = True
                harness.request_ends[0] = REQUEST_ENDED
        move = round_index > 0 and (moves == "every_round" or round_index == 4)
        sources = None
        if move:
            sources, live = sets[live], 1 - live
            garbage = {
                layer: (torch.full_like(conv, 7.0), recurrent)
                for layer, (conv, recurrent) in _page_values(
                    moved, sets[live], 3303 + round_index
                ).items()
            }
            # The new live page's taps are garbage, its recurrent state
            # another request's: the prepare copies the dense base over the
            # latter and nothing reads the former.
            _write_pages(moved, sets[live], garbage)
        inputs = _inputs(moved, len(rpis), 3310 + round_index, raw_gate=False)
        _assert_same_outputs(
            _decode_round(
                moved, rpis, sets[live], seq_lens, inputs, accepted, sources=sources
            ),
            _decode_round(static, rpis, homes, seq_lens, inputs, accepted),
        )
        _assert_same_requests(moved, static, sets[live], homes, (rpis, rpis))
        assert moved.backend._replay_state[rpis[1:], 1].tolist() == (
            static.backend._replay_state[rpis[1:], 1].tolist()
        )
        if move and round_index < 8:
            for layer in moved.backend._state_layer_ids():
                group = moved.pool.state_group_by_layer[layer]
                conv = moved.pool.get_component(layer, "conv_state")
                assert torch.all(conv[sets[live][group]] == 7.0)
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    # The ended request's live page holds its window, as the static run's.
    assert moved.request_ends[0].item() == REQUEST_ENDED
    for layer in moved.backend._state_layer_ids():
        group = moved.pool.state_group_by_layer[layer]
        _assert_same_bits(
            moved.pool.get_component(layer, "conv_state")[sets[live][group][0]],
            static.pool.get_component(layer, "conv_state")[homes[group][0]],
        )


@pytest.mark.parametrize("draft_tokens", [1, 4])
def test_an_ended_request_publishes_its_ring_window_in_its_live_page(draft_tokens):
    """The commit of the round that ends a request (``REQUEST_ENDED``) folds
    its records and writes the ring's last three accepted inputs into its
    live page, in the same launch that clears its count and ring state; a
    request that keeps decoding keeps its ring and its stale page. The ended
    page matches the ring-reset run's bit for bit and the torch reference."""
    steady = _Harness(draft_tokens=draft_tokens, seed=3401)
    reset = _Harness(draft_tokens=draft_tokens, seed=3401)
    reset.params = steady.params
    width = steady.width
    rpis = [4, 2]
    pages = _pages(rpis)
    values = _page_values(steady, pages, 3402)
    for harness in (steady, reset):
        _write_pages(harness, pages, values)
    untouched = _conv_pages(steady, pages)
    seq_lens = [10] * len(rpis)
    for round_index in range(6):
        ending = round_index == 5
        accepted = [max(1, width - 1), 1 + round_index % width]
        inputs = _inputs(steady, len(rpis), 3410 + round_index, raw_gate=True)
        if ending:
            _, expected = _reference_round(
                steady, pages, inputs, accepted, slots=rpis, prefill=False
            )
            for harness in (steady, reset):
                harness.stopped[rpis[0]] = True
                harness.request_ends[0] = REQUEST_ENDED
        _assert_same_outputs(
            _decode_round(steady, rpis, pages, seq_lens, inputs, accepted),
            _decode_round(reset, rpis, pages, seq_lens, inputs, accepted),
        )
        _disable_ring(reset, rpis, pages)
        _assert_same_requests(steady, reset, pages, pages, (rpis, rpis))
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    assert steady.request_ends[0].item() == REQUEST_ENDED
    assert steady.backend._replay_state[rpis[0]].tolist() == [0, 0]
    assert steady.backend._replay_state[rpis[1], 1].item() > 0
    for layer, (state, conv) in expected.items():
        group = steady.pool.state_group_by_layer[layer]
        ended = pages[group][0]
        page_conv = steady.pool.get_component(layer, "conv_state")
        _assert_same_bits(page_conv[ended], conv[0])
        _assert_same_bits(
            page_conv[ended],
            reset.pool.get_component(layer, "conv_state")[ended],
        )
        # The decoding request's page keeps the taps it had before its ring.
        _assert_same_bits(page_conv[pages[group][1]], untouched[layer][1])
        torch.testing.assert_close(
            steady.pool.get_component(layer, "recurrent_state")[ended],
            state[0],
            atol=2e-6,
            rtol=2e-5,
        )


def test_an_incomplete_end_writes_no_page_a_rejecting_layer_carries():
    """Only the first layer verifies the round that ends a request; the other
    layers' stamps are a round old, so they reject its commit and its report
    is ``REQUEST_ENDED_INCOMPLETE``. The accepting layer's live page holds
    the ring's window; a rejecting layer's page keeps both its states. The
    count and ring state clear: the request ended."""
    width = 4
    harness = _Harness(draft_tokens=width, seed=3451)
    rpis = [1]
    pages = _pages(rpis)
    _write_pages(harness, pages, _page_values(harness, pages, 3452))
    harness.round(rpis, pages, [10], 3453, [width])
    harness.round(rpis, pages, [10 + width], 3454, [2])
    first = harness.layer_ids[0]
    before = {
        (layer, name): harness.pool.get_component(layer, name)[
            pages[harness.pool.state_group_by_layer[layer]]
        ].clone()
        for layer in harness.layer_ids
        for name in ("conv_state", "recurrent_state")
    }
    inputs = _inputs(harness, 1, 3455, raw_gate=True)
    _, expected = _reference_round(
        harness, pages, inputs, [3], slots=rpis, prefill=False, layers=[first]
    )
    harness.layer_ids = [first]
    harness.stopped[1] = True
    harness.request_ends[0] = REQUEST_ENDED
    _decode_round(harness, rpis, pages, [12 + width], inputs, [3])
    assert harness.request_ends[0].item() == REQUEST_ENDED_INCOMPLETE
    assert harness.backend._replay_state[1].tolist() == [0, 0]
    group = harness.pool.state_group_by_layer[first]
    _assert_same_bits(
        harness.pool.get_component(first, "conv_state")[pages[group][0]],
        expected[first][1][0],
    )
    for (layer, name), snapshot in before.items():
        if layer == first:
            continue
        _assert_same_bits(
            harness.pool.get_component(layer, name)[
                pages[harness.pool.state_group_by_layer[layer]]
            ],
            snapshot,
        )


def test_a_retracted_request_restored_on_another_slot_reseeds_from_its_image():
    """A retraction folds the victim's records and ring window into its live
    page and clears its count and ring state. Restored on another slot, which
    a previous owner left seeded, onto another page holding the image, its
    first verify reseeds the ring from that page: it decodes bit for bit like
    the same request folded in place. One torch reference follows the request
    from the pages it started on through every round, the fold and the
    restore, and is never reset from the device: the folded, imaged and
    restored pages hold its window bit for bit and its recurrent state, so a
    fold that dropped accepted records fails though both runs agree."""
    width = 4
    restored = _Harness(draft_tokens=width, seed=3601)
    stayed = _Harness(draft_tokens=width, seed=3601)
    stayed.params = restored.params
    victim, other = 3, 6
    pages = _pages([victim])
    image = _pages([victim], first=12)
    previous = _pages([other], first=20)
    values = _page_values(restored, pages, 3602)
    previous_values = _page_values(restored, previous, 3603)
    for harness in (restored, stayed):
        _write_pages(harness, pages, values)
        _write_pages(harness, previous, previous_values)
        # A previous owner of the other slot decodes and leaves it seeded.
        for round_index in range(2):
            harness.round([other], previous, [9 + round_index], 3604 + round_index, [2])
    assert restored.backend._replay_state[other].tolist()[1] > 0
    tracked = {
        layer: (conv.clone(), recurrent.float().clone())
        for layer, (conv, recurrent) in values.items()
    }
    seq_lens = [11]
    for round_index in range(3):
        accepted = [1 + round_index]
        inputs = _inputs(restored, 1, 3610 + round_index, raw_gate=True)
        expected_outputs, tracked = _tracked_round(restored, tracked, inputs, accepted)
        outputs = _decode_round(restored, [victim], pages, seq_lens, inputs, accepted)
        _assert_same_outputs(
            outputs,
            _decode_round(stayed, [victim], pages, seq_lens, inputs, accepted),
        )
        _check_tracked(restored, pages, [victim], tracked, outputs, expected_outputs)
        seq_lens = [seq_lens[0] + accepted[0]]
    # Six accepted updates are still records the fold must apply, and the
    # ring head (seeded at row 3) wrapped.
    assert restored.backend._replay_state[victim].tolist() == [6, 9 % (width + 3) + 1]
    windows = {
        layer: _logical_conv(
            restored, layer, pages[restored.pool.state_group_by_layer[layer]], [victim]
        )
        for layer in restored.backend._state_layer_ids()
    }
    folds = {
        group: torch.tensor([[victim, ids[0]]], dtype=torch.int32)
        for group, ids in pages.items()
    }
    for harness in (restored, stayed):
        harness.backend.fold_deferred_state(folds, stream=torch.cuda.current_stream())
        assert harness.backend._replay_state[victim].tolist() == [0, 0]
        _assert_dense_pages_follow(harness, pages, tracked)
    for layer, window in windows.items():
        group = restored.pool.state_group_by_layer[layer]
        _assert_same_bits(
            restored.pool.get_component(layer, "conv_state")[pages[group]], window
        )
    # Restore: the image lands on other pages, the request on the other slot.
    _copy_pages(restored, pages, image)
    restored.backend.import_slot_state(
        other,
        torch.empty(restored.backend.slot_state_bytes(), dtype=torch.uint8),
        torch.cuda.current_stream(),
        admission_id=1,
    )
    assert restored.backend._replay_state[other].tolist() == [0, 0]
    _assert_dense_pages_follow(restored, image, tracked)
    for round_index in range(4):
        accepted = [width - round_index % 2]
        inputs = _inputs(restored, 1, 3620 + round_index, raw_gate=True)
        expected_outputs, tracked = _tracked_round(restored, tracked, inputs, accepted)
        outputs = _decode_round(restored, [other], image, seq_lens, inputs, accepted)
        _assert_same_outputs(
            outputs,
            _decode_round(stayed, [victim], pages, seq_lens, inputs, accepted),
        )
        _assert_same_requests(restored, stayed, image, pages, ([other], [victim]))
        _check_tracked(restored, image, [other], tracked, outputs, expected_outputs)
        seq_lens = [seq_lens[0] + accepted[0]]


@pytest.mark.parametrize("draft_tokens", [1, 4])
def test_a_prefill_continuation_materializes_a_decoding_slots_ring(draft_tokens):
    """A mixed forward continues the prefill of a request whose slot has been
    decoding (its ring seeded and wrapped) beside a decoding request. The
    gather writes the ring's window over the live page's taps before the
    prefill convolution, the continued slot's count and ring state clear,
    and its next decode reseeds the ring from the page the prefill wrote.
    Every forward matches the ring-reset run bit for bit and the torch
    reference."""
    steady = _Harness(draft_tokens=draft_tokens, seed=3701)
    reset = _Harness(draft_tokens=draft_tokens, seed=3701)
    reset.params = steady.params
    width, rows = steady.width, steady.width + 3
    rpis = [1, 2]
    pages = _pages(rpis)
    values = _page_values(steady, pages, 3702)
    for harness in (steady, reset):
        _write_pages(harness, pages, values)
    seq_lens = [9, 9]
    for round_index in range(5):
        accepted = [width, 1 + round_index % width]
        inputs = _inputs(steady, 2, 3710 + round_index, raw_gate=True)
        _assert_same_outputs(
            _decode_round(steady, rpis, pages, seq_lens, inputs, accepted),
            _decode_round(reset, rpis, pages, seq_lens, inputs, accepted),
        )
        _disable_ring(reset, rpis, pages)
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    # Five full windows wrapped slot 1's ring head.
    assert steady.backend._replay_state[1, 1].item() == (3 + 5 * width) % rows + 1

    prompt, count = 5, max(1, width - 1)
    inputs = {
        name: value[: prompt + width]
        for name, value in steady.inputs(prompt + width, 3720).items()
    }
    inputs["g_raw"] = inputs["mixed_qkv"][:, : steady.key_dim]
    prompt_outputs, prompt_states = _reference_round(
        steady,
        {group: ids[:1] for group, ids in pages.items()},
        {name: value[:prompt] for name, value in inputs.items()},
        [prompt],
        slots=[1],
        prefill=True,
    )
    decode_outputs, decode_states = _reference_round(
        steady,
        {group: ids[1:] for group, ids in pages.items()},
        {name: value[prompt:] for name, value in inputs.items()},
        [count],
        slots=[2],
        prefill=False,
    )

    def mixed_forward(harness):
        new = torch.tensor([prompt], dtype=torch.int32)
        prefix = torch.tensor([seq_lens[0]], dtype=torch.int32)
        tables = {
            group: np.asarray([[ids[0], ids[0]], [0, ids[1]]], dtype=np.int32)
            for group, ids in pages.items()
        }
        harness.backend.init_forward_metadata(
            2,
            1,
            torch.tensor(rpis, device=DEV, dtype=torch.int32),
            torch.tensor([seq_lens[0] + prompt, seq_lens[1]], device=DEV).int(),
            ForwardMode.MIXED,
            block_tables=_tables_for(harness.contract, tables, DEV),
            extend_seq_lens=new.to(DEV),
            extend_seq_lens_cpu=new,
            extend_prefix_lens=prefix.to(DEV),
            extend_prefix_lens_cpu=prefix,
            extend_replay_lens_cpu=torch.zeros_like(prefix),
            extend_prompt_lens_cpu=prefix + new,
            extend_with_prefix=True,
            extend_state_checkpoint_lens_cpu=torch.zeros_like(prefix),
            query_shard=None,
        )
        outputs = []
        for layer in harness.layer_ids:
            params = harness.params[layer]
            outputs.append(
                harness.backend.forward_extend(
                    None,
                    None,
                    None,
                    layer=None,
                    token_to_kv_pool=harness.pool,
                    bs=2,
                    forward_mode=ForwardMode.MIXED,
                    mixed_qkv=inputs["mixed_qkv"].clone(),
                    f_a_out=inputs["f_a_out"],
                    g_raw=inputs["g_raw"],
                    beta_raw=inputs["beta_raw"],
                    conv_weights=params["conv_weights"],
                    bias=None,
                    activation="silu",
                    key_dim=harness.key_dim,
                    value_dim=harness.key_dim,
                    attention_tp_size=1,
                    head_k_dim=D,
                    head_v_dim=D,
                    A_log=params["A_log"],
                    dt_bias=params["dt_bias"],
                    f_b_weight=params["f_b_weight"],
                    lower_bound=params["lower_bound"],
                    layer_id=layer,
                    seq_len=inputs["mixed_qkv"].shape[0],
                ).reshape(-1, harness.H, D)
            )
        harness.backend.commit_state_after_acceptance(
            torch.tensor([1, count], device=DEV, dtype=torch.int32),
            num_extends=1,
            accepted_path=None,
        )
        return outputs

    head = steady.backend._replay_state[2, 1].item() - 1
    outputs = mixed_forward(steady)
    _assert_same_outputs(outputs, mixed_forward(reset))
    _disable_ring(reset, rpis[1:], {group: ids[1:] for group, ids in pages.items()})
    assert steady.backend._replay_state[1].tolist() == [0, 0]
    assert steady.backend._replay_state[2, 1].item() == (head + count) % rows + 1
    _assert_same_requests(steady, reset, pages, pages, (rpis, rpis))
    for layer, output, prefill, decode in zip(
        steady.layer_ids, outputs, prompt_outputs, decode_outputs, strict=True
    ):
        torch.testing.assert_close(
            output, torch.cat((prefill, decode)), atol=2e-3, rtol=2e-3
        )
        group = steady.pool.state_group_by_layer[layer]
        _assert_same_bits(
            _logical_conv(steady, layer, pages[group], rpis),
            torch.cat((prompt_states[layer][1], decode_states[layer][1])),
        )
        # The continued request's page holds the prefill's window.
        _assert_same_bits(
            steady.pool.get_component(layer, "conv_state")[pages[group][0]],
            prompt_states[layer][1][0],
        )
        torch.testing.assert_close(
            _logical_state(steady, layer, pages[group], rpis),
            torch.cat((prompt_states[layer][0], decode_states[layer][0])),
            atol=2e-6,
            rtol=2e-5,
        )
    seq_lens = [seq_lens[0] + prompt, seq_lens[1] + count]
    accepted = [width, 1]
    inputs = _inputs(steady, 2, 3730, raw_gate=True)
    first = steady.layer_ids[0]
    _, expected = _reference_round(
        steady, pages, inputs, accepted, slots=rpis, prefill=False, layers=[first]
    )
    _assert_same_outputs(
        _decode_round(steady, rpis, pages, seq_lens, inputs, accepted),
        _decode_round(reset, rpis, pages, seq_lens, inputs, accepted),
    )
    # The continued slot's ring was reseeded from its page: its head is
    # three rows past the seed plus the round's acceptance.
    assert steady.backend._replay_state[1, 1].item() == (3 + width) % rows + 1
    _check_reference(steady, first, pages, rpis, expected, raw_gate=True)


def test_a_landed_request_reads_its_transferred_page_on_its_first_decode():
    """A PD landing puts a request whose prompt ran on another node on a
    slot a previous owner left seeded. The landing's reset unseeds the ring
    (a retraction fold before the first forward then changes no page), so
    the first verify reads the transferred page's convolution state: the
    request decodes bit for bit like one on a never-used slot and follows
    the torch reference from the transferred taps on."""
    width, slot = 4, 5
    landed = _Harness(draft_tokens=width, seed=3801)
    fresh = _Harness(draft_tokens=width, seed=3801)
    fresh.params = landed.params
    previous = _pages([slot])
    transferred = _pages([slot], first=10)
    _write_pages(landed, previous, _page_values(landed, previous, 3802))
    for round_index in range(3):
        landed.round([slot], previous, [9 + width * round_index], 3803, [width])
    assert landed.backend._replay_state[slot].tolist() == [
        3 * width,
        (3 + 3 * width) % (width + 3) + 1,
    ]
    values = _page_values(landed, transferred, 3810)
    for harness in (landed, fresh):
        _write_pages(harness, transferred, values)
    stream = torch.cuda.current_stream()
    landed.backend.reset_request_workspace(
        torch.tensor([slot], device=DEV), stream=stream
    )
    assert landed.backend._replay_state[slot].tolist() == [0, 0]
    landed.backend.fold_deferred_state(
        {
            group: torch.tensor([[slot, ids[0]]], dtype=torch.int32)
            for group, ids in transferred.items()
        },
        stream=stream,
    )
    for layer, (conv, recurrent) in values.items():
        ids = transferred[landed.pool.state_group_by_layer[layer]]
        _assert_same_bits(landed.pool.get_component(layer, "conv_state")[ids], conv)
        _assert_same_bits(
            landed.pool.get_component(layer, "recurrent_state")[ids], recurrent
        )
    first = landed.layer_ids[0]
    seq_lens = [33]
    for round_index in range(4):
        accepted = [1 + round_index % width]
        inputs = _inputs(landed, 1, 3820 + round_index, raw_gate=True)
        expected_outputs, expected = _reference_round(
            landed,
            transferred,
            inputs,
            accepted,
            slots=[slot],
            prefill=False,
            layers=[first],
        )
        outputs = _decode_round(landed, [slot], transferred, seq_lens, inputs, accepted)
        _assert_same_outputs(
            outputs,
            _decode_round(fresh, [slot], transferred, seq_lens, inputs, accepted),
        )
        _assert_same_requests(landed, fresh, transferred, transferred, ([slot], [slot]))
        _check_reference(landed, first, transferred, [slot], expected, raw_gate=True)
        torch.testing.assert_close(
            outputs[0].reshape_as(expected_outputs[0]),
            expected_outputs[0],
            atol=2e-3,
            rtol=2e-3,
        )
        seq_lens = [seq_lens[0] + accepted[0]]


def test_a_new_owner_of_a_slot_whose_end_the_device_missed_reseeds_its_ring():
    """An owner aborted or finished on the host leaves its slot's ring seeded
    and its records counted. The new owner's reset unseeds the ring, so it
    decodes bit for bit like a request on a never-used slot; without the
    reset its first verify would read the old owner's window and records."""
    width, slot = 4, 2
    reused = _Harness(draft_tokens=width, seed=3851)
    fresh = _Harness(draft_tokens=width, seed=3851)
    stale = _Harness(draft_tokens=width, seed=3851)
    fresh.params = stale.params = reused.params
    old = _pages([slot])
    new = _pages([slot], first=10)
    old_values = _page_values(reused, old, 3852)
    for harness in (reused, stale):
        _write_pages(harness, old, old_values)
        for round_index in range(3):
            harness.round([slot], old, [9 + 3 * round_index], 3853, [3])
    values = _page_values(reused, new, 3860)
    for harness in (reused, fresh, stale):
        _write_pages(harness, new, values)
    reused.backend.reset_request_workspace(
        torch.tensor([slot], device=DEV), stream=torch.cuda.current_stream()
    )
    seq_lens = [21]
    for round_index in range(3):
        accepted = [width - round_index]
        inputs = _inputs(reused, 1, 3870 + round_index, raw_gate=False)
        outputs = _decode_round(reused, [slot], new, seq_lens, inputs, accepted)
        _assert_same_outputs(
            outputs, _decode_round(fresh, [slot], new, seq_lens, inputs, accepted)
        )
        _assert_same_requests(reused, fresh, new, new, ([slot], [slot]))
        if round_index == 0:
            stale_outputs = _decode_round(
                stale, [slot], new, seq_lens, inputs, accepted
            )
            assert not torch.equal(stale_outputs[0], outputs[0])
        seq_lens = [seq_lens[0] + accepted[0]]


def test_graph_replays_match_eager_rounds_bit_for_bit_across_ring_wraps():
    """Replays of one captured verify graph, each after an eager prepare and
    followed by the eager commit, match eager rounds bit for bit over
    several ring turns: outputs, ring rows, counts and ring states, records,
    windows, and the page a request's last round publishes."""
    width, bs = 4, 2
    captured = _Harness(draft_tokens=width, seed=3901)
    eager = _Harness(draft_tokens=width, seed=3901)
    eager.params = captured.params
    rpis = [2, 1]
    pages = _pages(rpis)
    values = _page_values(captured, pages, 3902)
    for harness in (captured, eager):
        _write_pages(harness, pages, values)
    seq_lens = [12] * bs

    def tables():
        return _tables_for(
            captured.contract,
            {
                group: np.asarray([[0, page] for page in ids], dtype=np.int32)
                for group, ids in pages.items()
            },
            DEV,
        )

    captured.prepare_metadata(rpis, pages, seq_lens)
    captured.forward(captured.inputs(bs, 3903), bs)
    torch.cuda.synchronize()
    req_pool_indices = torch.tensor(rpis, dtype=torch.int32, device=DEV)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    captured.backend.init_forward_metadata_capture_cuda_graph(
        bs,
        req_pool_indices,
        seq_lens_tensor,
        ForwardMode.DECODE,
        block_tables={
            group: torch.full((bs, 2), -1, dtype=torch.int32, device=DEV)
            for group in _STATE_GROUPS
        },
    )
    stable_inputs = captured.inputs(bs, 3904)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        captured.forward(stable_inputs, bs)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = captured.forward(stable_inputs, bs)

    rows = width + 3
    heads = [3] * bs
    for round_index in range(3 * rows):
        ending = round_index == 3 * rows - 1
        accepted = [1 + round_index % width, width - round_index % 2]
        if ending:
            for harness in (captured, eager):
                harness.stopped[rpis[0]] = True
                harness.request_ends[0] = REQUEST_ENDED
        replay_inputs = captured.inputs(bs, 3910 + round_index)
        seq_lens_tensor.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=DEV))
        captured.backend.refresh_decode_metadata(
            bs,
            bs,
            req_pool_indices,
            seq_lens_tensor,
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables=tables(),
        )
        for name, value in replay_inputs.items():
            stable_inputs[name].copy_(value)
        graph.replay()
        captured.backend.commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV), accepted_path=None
        )
        expected = _decode_round(eager, rpis, pages, seq_lens, replay_inputs, accepted)
        torch.cuda.synchronize()
        _assert_same_outputs(graph_outputs, expected)
        assert (
            captured.backend._replay_state[rpis].tolist()
            == eager.backend._replay_state[rpis].tolist()
        )
        _assert_same_bits(
            captured.backend._preconv_ring[rpis], eager.backend._preconv_ring[rpis]
        )
        _assert_same_requests(captured, eager, pages, pages, (rpis, rpis))
        heads = [(head + take) % rows for head, take in zip(heads, accepted)]
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    assert captured.backend._replay_state[rpis].tolist()[0] == [0, 0]
    assert captured.backend._replay_state[rpis[1], 1].item() == heads[1] + 1
    for layer in captured.backend._state_layer_ids():
        page = pages[captured.pool.state_group_by_layer[layer]][0]
        _assert_same_bits(
            captured.pool.get_component(layer, "conv_state")[page],
            eager.pool.get_component(layer, "conv_state")[page],
        )


def _check_ring_lifecycle_survives_release_on_cuda():
    """Decode with the backend's state in the memory saver's regions, end one
    request on the device, reset a slot for a new owner whose old one the
    device never saw end, and snapshot everything the next round reads. Run
    that round, put the snapshot back, release and resume twice, and run it
    again: no ordinary CUDA allocator segment survives a release, every
    persistent tensor keeps its address and contents, the ended and reset
    slots stay unseeded, and the round gives the same bits."""
    pool = _memory_saver_kimi_pool(usable_pages=24)
    saver = pool.arena.memory_saver_adapter
    width = 4
    harness = _Harness(draft_tokens=width, seed=3951, pool=pool)
    backend = harness.backend
    rpis = [3, 1, 5]
    pages = _pages(rpis)
    with saver.named_region("attention_state"):
        backend._verify_scratch_grid(len(rpis), width)

    def round_(pages, seq_lens, seed, accepted):
        with saver.named_region("attention_state"):
            tables = _tables_for(
                harness.contract,
                {
                    group: np.asarray([[0, page] for page in ids], dtype=np.int32)
                    for group, ids in pages.items()
                },
                DEV,
            )
        with saver.named_region("runtime_states"):
            indices = torch.tensor(rpis, dtype=torch.int32, device=DEV)
            lengths = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
            inputs = harness.inputs(len(rpis), seed)
            taken = torch.tensor(accepted, dtype=torch.int32, device=DEV)
        backend.refresh_decode_metadata(
            len(rpis),
            len(rpis),
            indices,
            lengths,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
        )
        outputs = [output.cpu() for output in harness.forward(inputs, len(rpis))]
        backend.commit_verified_state(taken, accepted_path=None)
        return outputs

    _write_pages(harness, pages, _page_values(harness, pages, 3952))
    seq_lens = [10] * len(rpis)
    for round_index in range(3):
        accepted = [width, 1 + round_index, 2]
        # Slot 1's request ends in the last round; the device sees it.
        harness.stopped[1] = round_index == 2
        round_(pages, seq_lens, 3960 + round_index, accepted)
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    harness.stopped[1] = False
    # Slot 5's request is aborted on the host: its new owner's reset.
    with saver.named_region("runtime_states"):
        reused = torch.tensor([5], dtype=torch.int64, device=DEV)
    backend.reset_request_workspace(reused, stream=torch.cuda.current_stream())
    assert backend._replay_state[[1, 5]].tolist() == [[0, 0], [0, 0]]
    assert backend._replay_state[3, 1].item() > 0
    # New owners of slots 1 and 5 decode on their own pages.
    successors = _pages(rpis, first=11)
    following = {
        group: [ids[0], *successors[group][1:]] for group, ids in pages.items()
    }
    _write_pages(harness, successors, _page_values(harness, successors, 3970))
    seq_lens = [seq_lens[0], 30, 30]
    persistent = {
        "ring": backend._preconv_ring,
        "replay_state": backend._replay_state,
        "records": backend._retained_records,
        "snapshot": backend._replay_snapshot,
        "stamps": backend._replay_record_stamps,
        "descriptors": backend._replay_descriptors.view(torch.int64),
    }
    pointers = {name: tensor.data_ptr() for name, tensor in persistent.items()}

    def host_pages():
        return {
            (layer, name): pool.get_component(layer, name)[
                following[pool.state_group_by_layer[layer]]
            ].cpu()
            for layer in backend._state_layer_ids()
            for name in ("conv_state", "recurrent_state")
        }

    snapshot = {name: tensor.cpu() for name, tensor in persistent.items()}
    page_snapshot = host_pages()
    expected = round_(following, seq_lens, 3980, [2, width, 1])
    after = (backend._preconv_ring.cpu(), backend._replay_state.cpu())
    # Put back everything the round changed, so it can run again.
    for name, tensor in persistent.items():
        tensor.copy_(snapshot[name])
    for (layer, name), host in page_snapshot.items():
        pool.get_component(layer, name)[following[pool.state_group_by_layer[layer]]] = (
            host.to(DEV)
        )
    gc.collect()
    for _ in range(2):
        torch.cuda.synchronize()
        saver.pause(tag="weights")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ordinary_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == torch.cuda.current_device()
            and tuple(segment["segment_pool_id"]) == (0, 0)
        )
        assert ordinary_bytes == 0, ordinary_bytes
        saver.resume(tag="weights")
        for name, tensor in persistent.items():
            assert tensor.data_ptr() == pointers[name]
            assert torch.equal(_bits(tensor.cpu()), _bits(snapshot[name]))
        for key, host in host_pages().items():
            assert torch.equal(_bits(host), _bits(page_snapshot[key]))
        assert backend._replay_state[[1, 5]].tolist() == [[0, 0], [0, 0]]
    for got, want in zip(
        round_(following, seq_lens, 3980, [2, width, 1]), expected, strict=True
    ):
        assert torch.equal(_bits(got), _bits(want))
    assert torch.equal(_bits(backend._preconv_ring.cpu()), _bits(after[0]))
    assert torch.equal(backend._replay_state.cpu(), after[1])


@pytest.mark.skipif(
    torch.version.hip is not None, reason="requires native CUDA VMM memory-saver hooks"
)
def test_release_and_resume_leave_no_stale_ring_state():
    memory_saver = pytest.importorskip("torch_memory_saver")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_kda_preconv_ring import "
                "_check_ring_lifecycle_survives_release_on_cuda; "
                "_check_ring_lifecycle_survives_release_on_cuda()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=900,
        )


def test_far_record_slots_address_rings_and_records_past_int32():
    """Requests on the last of 1000 resident slots, whose ring and record
    offsets exceed 2^31 elements, decode, end and fold bit for bit like the
    same requests on slots 1 and 2, through both the fused and the split
    verify producers."""
    width = 4
    far = _Harness(draft_tokens=width, seed=4001, max_resident_requests=1000)
    near = _Harness(draft_tokens=width, seed=4001)
    near.params = far.params
    far_slots, near_slots = [1000, 999], [1, 2]
    for tensor in (far.backend._preconv_ring, far.backend._retained_records):
        assert min(far_slots) * tensor.stride(0) > 2**31
    pages = _pages(far_slots)
    values = _page_values(far, pages, 4002)
    for harness in (far, near):
        _write_pages(harness, pages, values)
    seq_lens = [10, 10]
    for round_index in range(8):
        accepted = [1 + round_index % width, width]
        inputs = _inputs(far, 2, 4010 + round_index, raw_gate=bool(round_index % 2))
        if round_index == 7:
            far.stopped[far_slots[0]] = True
            near.stopped[near_slots[0]] = True
        _assert_same_outputs(
            _decode_round(far, far_slots, pages, seq_lens, inputs, accepted),
            _decode_round(near, near_slots, pages, seq_lens, inputs, accepted),
        )
        assert (
            far.backend._replay_state[far_slots].tolist()
            == near.backend._replay_state[near_slots].tolist()
        )
        _assert_same_bits(
            far.backend._preconv_ring[far_slots], near.backend._preconv_ring[near_slots]
        )
        _assert_same_requests(far, near, pages, pages, (far_slots, near_slots))
        seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
    for harness, slots in ((far, far_slots), (near, near_slots)):
        harness.backend.fold_deferred_state(
            {
                group: torch.tensor([[slots[1], ids[1]]], dtype=torch.int32)
                for group, ids in pages.items()
            },
            stream=torch.cuda.current_stream(),
        )
    assert far.backend._replay_state[far_slots].tolist() == [[0, 0], [0, 0]]
    for layer in far.backend._state_layer_ids():
        ids = pages[far.pool.state_group_by_layer[layer]]
        for name in ("conv_state", "recurrent_state"):
            _assert_same_bits(
                far.pool.get_component(layer, name)[ids],
                near.pool.get_component(layer, name)[ids],
            )


def _kernel_test_utils():
    kernel_test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tokenspeed-kernel",
        "test",
    )
    spec = importlib.util.spec_from_file_location(
        "kda_preconv_ring_kernel_test_utils", os.path.join(kernel_test_dir, "utils.py")
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


def _verify_launch_class(batch, window):
    """The batch-size class the CuTe DSL verify kernels compile once for.

    They key their compile caches on the value-tile rows the batch's
    request-head pairs choose and, for the fused window, on whether its grid
    fits in one CTA per SM (record copies) or in two (the shared-memory
    floor). The Triton ring and page kernels take every batch-dependent
    scalar unspecialized, so they add no class.
    """
    from tokenspeed_kernel.ops.attention.kda._cute_dsl.recoverssm_verify import (
        _value_tile_rows,
    )

    rows = _value_tile_rows(batch, _Harness.H, window, D)
    ctas = batch * _Harness.H * (D // rows)
    sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    return rows, ctas <= sms, ctas <= 2 * sms


@pytest.mark.skipif(
    torch.version.hip is not None,
    reason="watches the NVIDIA Triton and CuTe DSL kernels",
)
@pytest.mark.parametrize("draft_tokens", [1, 4])
def test_ring_kernels_compile_nothing_new_across_batch_sizes_and_slot_counts(
    draft_tokens,
):
    """From empty JIT caches, one backend serves one batch size of each
    launch class (``_verify_launch_class``) through both verify producers,
    ends a request and folds the rest. A second backend with another
    resident slot count and other slots then serves every other batch size
    up to its maximum, each unseen in its class, without compiling any
    Triton or CuTe DSL kernel: a kernel that specialized on the batch size
    or the slot count would compile again."""
    from tokenspeed_kernel.ops.attention.kda._cute_dsl import (
        fused_verify_step,
        fused_verify_window,
        recoverssm_verify,
    )
    from tokenspeed_kernel.ops.attention.kda._triton import (
        recurrent,
        retained,
        state_pages,
    )

    utils = _kernel_test_utils()
    # Every KDA Triton kernel of the decode path: the ring and record
    # kernels, the verify convolution producer and the state page planners.
    kernels = tuple(
        {
            id(value): value
            for module in (retained, recurrent, state_pages)
            for value in vars(module).values()
            # Triton JIT functions, whichever Triton build defines them.
            if hasattr(value, "_do_compile") and hasattr(value, "device_caches")
        }.values()
    )
    assert retained.retained_batched_commit_kernel in kernels
    assert recurrent.fused_kda_verify_conv_update_kernel in kernels
    max_bs = 16
    classes = {}
    for batch in range(1, max_bs + 1):
        classes.setdefault(_verify_launch_class(batch, draft_tokens), []).append(batch)
    warmed = [sizes[0] for sizes in classes.values()]
    unseen = [batch for sizes in classes.values() for batch in sizes[1:]]
    # Most batch sizes are unseen, odd and even ones, in several classes.
    assert len(unseen) >= max_bs // 2 and {batch % 2 for batch in unseen} == {0, 1}
    assert sum(len(sizes) > 1 for sizes in classes.values()) >= 2

    def serve(harness, slots, batches):
        width = harness.width
        for batch in batches:
            rpis = slots[:batch]
            pages = _pages(rpis)
            seq_lens = [9] * batch
            for round_index, raw_gate in enumerate((False, True)):
                # The second round ends the last request.
                harness.stopped[rpis[-1]] = round_index == 1
                accepted = [1 + (row + round_index) % width for row in range(batch)]
                inputs = _inputs(harness, batch, 4100 + batch, raw_gate=raw_gate)
                _decode_round(harness, rpis, pages, seq_lens, inputs, accepted)
                seq_lens = [length + take for length, take in zip(seq_lens, accepted)]
            harness.stopped[rpis[-1]] = False
            if batch > 1:
                harness.backend.fold_deferred_state(
                    {
                        group: torch.tensor(
                            list(zip(rpis[:-1], ids[:-1], strict=True)),
                            dtype=torch.int32,
                        )
                        for group, ids in pages.items()
                    },
                    stream=torch.cuda.current_stream(),
                )

    # Binaries an earlier test compiled must not stand in for the warmup's.
    for kernel in kernels:
        kernel.device_caches.clear()
    for module in (fused_verify_step, fused_verify_window, recoverssm_verify):
        module._compile.cache_clear()
    warm = _Harness(
        draft_tokens=draft_tokens,
        max_bs=max_bs,
        max_resident_requests=max_bs,
        usable_pages=96,
    )
    serve(warm, [3, 1, 8, 2, 16, 7, 4, 11, 6, 5, 14, 9, 10, 12, 13, 15], warmed)
    torch.cuda.synchronize()
    with utils.assert_no_triton_compile(*kernels), utils.assert_no_cute_compile():
        other = _Harness(
            draft_tokens=draft_tokens,
            max_bs=max_bs,
            max_resident_requests=23,
            usable_pages=96,
        )
        serve(
            other,
            [23, 9, 1, 17, 5, 21, 12, 2, 13, 19, 6, 22, 11, 18, 3, 20],
            unseen,
        )
        torch.cuda.synchronize()


@pytest.mark.parametrize("draft_tokens", [1, 2, 4, 8])
def test_the_recipe_prices_the_ring_rows_the_backend_allocates(draft_tokens):
    """Every resident slot (and the null and padding slots) keeps T + 3 BF16
    pre-convolution rows per KDA layer; the recipe's slot and workspace
    bytes equal the backend's allocation."""
    harness = _Harness(draft_tokens=draft_tokens, max_bs=4, max_resident_requests=6)
    recipe = _kimi_recipe(
        max_bs=4,
        max_resident_requests=6,
        speculative_algorithm=None if draft_tokens == 1 else "EAGLE3",
        speculative_num_draft_tokens=draft_tokens,
    )
    assert (
        harness.backend.preallocate_verify_workspace(4, draft_tokens)
        == recipe.workspace_bytes()
    )
    ring = harness.backend._preconv_ring
    assert ring.shape == (8, K3_LAYERS, draft_tokens + 3, K3_TP8_CHANNELS)
    assert ring.dtype == torch.bfloat16
    assert ring[0].nbytes == (draft_tokens + 3) * K3_LAYERS * K3_TP8_CHANNELS * 2
    assert kda_request_slot_bytes(
        recipe.attn_config, recipe.groups(), draft_token_num=draft_tokens
    ) == (
        harness.backend._retained_records[0].nbytes
        + ring[0].nbytes
        + harness.backend._replay_state[0].nbytes
    )


def test_kimi_k3_tp8_keeps_seven_ring_rows_per_slot_at_four_draft_tokens():
    """The figures the design docs quote: at T = 4 a Kimi-K3 TP8 slot keeps
    T + 3 = 7 pre-convolution rows of 635,904 bytes (69 layers of 4,608
    BF16 channels), 4,451,328 bytes (4.245 MiB), beside 20,348,928 bytes of
    FP32 records (the default capacity of 16) and its two INT32 counters."""
    row = K3_LAYERS * K3_TP8_CHANNELS * torch.bfloat16.itemsize
    assert row == 635_904
    ring = 7 * row
    assert ring == 4_451_328 and f"{ring / 2**20:.3f}" == "4.245"
    # A ring of capacity + 3 rows (19 at the default capacity 16) would
    # keep 11.52 MiB a slot.
    assert f"{19 * row / 2**20:.2f}" == "11.52"
    records = K3_LAYERS * 12 * 16 * (D + 2 * D) * torch.float32.itemsize
    assert records == 20_348_928
    recipe = _kimi_recipe(
        tp_size=8,
        max_bs=8,
        max_resident_requests=13,
        speculative_algorithm="EAGLE3",
        speculative_num_draft_tokens=4,
    )
    assert (
        kda_request_slot_bytes(recipe.attn_config, recipe.groups(), draft_token_num=4)
        == records + ring + 8
        == 24_800_264
    )
    harness = _Harness(draft_tokens=4, max_bs=8, max_resident_requests=13)
    assert harness.backend._preconv_ring.shape == (15, 69, 7, 4608)
    assert harness.backend._preconv_ring[0].nbytes == ring
    assert (
        harness.backend.preallocate_verify_workspace(8, 4) == recipe.workspace_bytes()
    )


def test_a_large_window_convolution_update_matches_the_reference_bit_for_bit():
    """A 2049-token window spans 129 token tiles of the split convolution
    producer. With integer inputs and power-of-two tap weights every product
    and sum is exact in FP32 and SiLU saturates (each pre-activation is at
    least 24), so the reference is exact: the outputs and the ring (each
    live request's history read from rows (head - 3 + i) mod R and its
    inputs written to rows (head + t) mod R, wrapping; a padding request
    reads a zero history and writes nothing) match bit for bit. With signed
    random inputs, where SiLU does not saturate, the ring matches bit for bit,
    the outputs follow the torch reference within the verify convolution's
    tolerance and equal the same kernel's one-token windows."""
    width, heads = 2049, 12
    channels, rows = 3 * heads * D, width + 3
    generator = torch.Generator().manual_seed(4201)
    batch = 3
    read = torch.tensor([1, 2, 3], dtype=torch.int32, device=DEV)
    live = torch.tensor([1, 2, 0], dtype=torch.int32, device=DEV)
    record_slots = torch.tensor([2, 0, 3], dtype=torch.int64, device=DEV)
    ring_heads = torch.tensor([1, rows - 2, 7], dtype=torch.int32, device=DEV)
    bounds = torch.arange(batch + 1, dtype=torch.int32, device=DEV) * width

    def update(raw, weights, ring):
        return kda_verify_conv_update(
            raw,
            weights,
            read,
            live_indices=live,
            preconv=KdaPreconvRing(ring, ring_heads),
            record_slots=record_slots,
            page_count=4,
            num_heads=heads,
            head_dim=D,
            draft_token_num=width,
            recurrent_layout="v_major",
            bias=None,
            query_start_loc=bounds,
        )

    def history(ring, row):
        if live[row].item() <= 0:
            return torch.zeros(3, channels, device=DEV, dtype=ring.dtype)
        slot, head = record_slots[row].item(), ring_heads[row].item()
        return ring[slot, [(head - 3 + i) % rows for i in range(3)]]

    raw = torch.randint(24, 88, (batch * width, channels), generator=generator)
    raw = raw.to(DEV, torch.bfloat16)
    scale = 1.0 + 0.5 * torch.randint(0, 2, (channels, 1), generator=generator)
    weights = scale * torch.tensor([0.125, 0.25, 0.5, 1.0])
    weights = weights.to(DEV, torch.bfloat16).contiguous()
    ring = torch.randint(24, 88, (4, rows, channels), generator=generator)
    ring = ring.to(DEV, torch.bfloat16)
    expected_ring, expected = ring.clone(), torch.empty_like(raw)
    for row in range(batch):
        tokens = raw[row * width : (row + 1) * width]
        window = torch.cat((history(ring, row), tokens)).float()
        pre = torch.zeros(width, channels, device=DEV)
        for tap in range(4):
            pre = pre + window[tap : tap + width] * weights[:, tap].float()
        assert pre.min().item() >= 24
        expected[row * width : (row + 1) * width] = (pre * torch.sigmoid(pre)).to(
            torch.bfloat16
        )
        if live[row].item() > 0:
            slot, head = record_slots[row].item(), ring_heads[row].item()
            expected_ring[slot, [(head + t) % rows for t in range(width)]] = tokens
    _assert_same_bits(update(raw, weights, ring), expected)
    _assert_same_bits(ring, expected_ring)

    signed = torch.randn((batch * width, channels), generator=generator)
    signed = signed.to(DEV, torch.bfloat16)
    signed_weights = (torch.randn((channels, 4), generator=generator) * 0.5).to(
        DEV, torch.bfloat16
    )
    signed_ring = torch.randn((4, rows, channels), generator=generator)
    signed_ring = signed_ring.to(DEV, torch.bfloat16)
    histories = [history(signed_ring, row) for row in range(batch)]
    expected_signed_ring = signed_ring.clone()
    for row in range(batch):
        if live[row].item() > 0:
            slot, head = record_slots[row].item(), ring_heads[row].item()
            expected_signed_ring[slot, [(head + t) % rows for t in range(width)]] = (
                signed[row * width : (row + 1) * width]
            )
    actual = update(signed, signed_weights, signed_ring)
    _assert_same_bits(signed_ring, expected_signed_ring)
    # The torch reference with no saturation: FP32 taps, then SiLU, rounded
    # to BF16 within the verify convolution's tolerance.
    from test.runtime.kda_torch_reference import kda_convolution_reference

    reference, _ = kda_convolution_reference(
        signed.view(batch, width, channels),
        torch.stack(histories).transpose(1, 2),
        conv_weights=signed_weights,
        bias=None,
        prefill=False,
    )
    torch.testing.assert_close(
        actual.float(), reference.reshape_as(actual).float(), atol=1e-4, rtol=8e-3
    )
    ones = torch.ones(width, dtype=torch.int32, device=DEV)
    for row in range(batch):
        if live[row].item() <= 0:
            continue
        tokens = signed[row * width : (row + 1) * width]
        window = torch.cat((histories[row], tokens))
        # Token t alone: its ring rows 0..2 hold the three inputs before it.
        singles = window.unfold(0, 4, 1).permute(0, 2, 1).contiguous()
        _assert_same_bits(
            actual[row * width : (row + 1) * width],
            kda_verify_conv_update(
                tokens,
                signed_weights,
                ones,
                live_indices=ones,
                preconv=KdaPreconvRing(
                    singles, torch.full_like(ones, 3, dtype=torch.int32)
                ),
                record_slots=torch.arange(width, device=DEV),
                page_count=2,
                num_heads=heads,
                head_dim=D,
                draft_token_num=1,
                recurrent_layout="v_major",
                bias=None,
                query_start_loc=torch.arange(width + 1, dtype=torch.int32, device=DEV),
            ),
        )
