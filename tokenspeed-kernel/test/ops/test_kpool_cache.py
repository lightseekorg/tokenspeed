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

"""KPool completed-pool and speculative-tail cache tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel import kpool_decode_append, kpool_prefill_write
from tokenspeed_kernel.ops.attention import kpool_prefill_tail_write

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_POOL = 4
_DIM = 128
_ROWS_PER_PAGE = 64 // _POOL
_FP8_MAX = 448.0
_FP8_E4M3FN_MAX_FINITE_MAGNITUDE_CODE = 0x7E
_MAX_FP8_QUANTUM_DISTANCE = 2


def _sylvester_h(n: int) -> torch.Tensor:
    idx = torch.arange(n, device="cuda")
    bits = idx[:, None] & idx[None, :]
    parity = torch.zeros_like(bits)
    for shift in range(7):
        parity ^= (bits >> shift) & 1
    return torch.where(parity == 0, 1.0, -1.0).float()


def _reference_compress(
    keys: torch.Tensor, gates: torch.Tensor, ape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gates.float() + ape.unsqueeze(0)
    pooled = (torch.softmax(logits, dim=1) * keys.float()).sum(dim=1)
    pooled = pooled.to(torch.bfloat16).float()
    pooled = ((pooled @ _sylvester_h(_DIM)) / _DIM**0.5).to(torch.bfloat16).float()
    scale = pooled.abs().amax(dim=-1).clamp(min=1e-4) / _FP8_MAX
    values = (pooled / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX)
    return values.to(torch.float8_e4m3fn), scale


def _random(
    shape: tuple[int, ...], generator: torch.Generator, scale: float
) -> torch.Tensor:
    return (torch.randn(shape, device="cuda", generator=generator) * scale).to(
        torch.bfloat16
    )


def _fp8_quantum_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Return the number of E4M3FN representable steps separating two values."""
    assert actual.dtype == expected.dtype == torch.float8_e4m3fn

    def ordered_rank(values: torch.Tensor) -> torch.Tensor:
        bits = values.contiguous().view(torch.uint8)
        magnitude = (bits & 0x7F).to(torch.int16)
        assert not bool(
            (magnitude == 0x7F).any()
        ), "FP8 quantum distance is undefined for NaN encodings"
        negative = (bits & 0x80) != 0
        return torch.where(
            negative,
            _FP8_E4M3FN_MAX_FINITE_MAGNITUDE_CODE - magnitude,
            _FP8_E4M3FN_MAX_FINITE_MAGNITUDE_CODE + magnitude,
        )

    return (ordered_rank(actual) - ordered_rank(expected)).abs()


def _assert_encoded(
    actual_values: torch.Tensor,
    actual_scales: torch.Tensor,
    expected_values: torch.Tensor,
    expected_scales: torch.Tensor,
) -> None:
    actual_scales = actual_scales.reshape(-1)
    expected_scales = expected_scales.reshape(-1)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-5, atol=1e-7)
    active = expected_scales >= 0
    actual_values = actual_values.reshape(-1, _DIM)[active]
    expected_values = expected_values.reshape(-1, _DIM)[active]
    quantum_distance = _fp8_quantum_distance(actual_values, expected_values)
    max_distance = int(quantum_distance.max().item())
    assert max_distance <= _MAX_FP8_QUANTUM_DISTANCE, (
        f"FP8 values differ by up to {max_distance} representable steps; "
        f"expected at most {_MAX_FP8_QUANTUM_DISTANCE}"
    )


def test_fp8_quantum_distance_counts_representable_steps() -> None:
    actual_bits = torch.tensor(
        [0x66, 0x67, 0xEA, 0xEB, 0x80, 0x01, 0xFE],
        dtype=torch.uint8,
        device="cuda",
    )
    expected_bits = torch.tensor(
        [0x65, 0x65, 0xE9, 0xE9, 0x00, 0x81, 0x7E],
        dtype=torch.uint8,
        device="cuda",
    )

    distance = _fp8_quantum_distance(
        actual_bits.view(torch.float8_e4m3fn),
        expected_bits.view(torch.float8_e4m3fn),
    )

    assert torch.equal(
        distance,
        torch.tensor([1, 2, 1, 2, 0, 2, 252], device="cuda", dtype=torch.int16),
    )


def test_fp8_quantum_distance_rejects_nan() -> None:
    nan = torch.tensor([0x7F], dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    zero = torch.tensor([0x00], dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )

    with pytest.raises(AssertionError, match="undefined for NaN"):
        _fp8_quantum_distance(nan, zero)


def test_assert_encoded_limits_fp8_quantum_distance() -> None:
    expected_bits = torch.full((1, _DIM), 0x65, dtype=torch.uint8, device="cuda")
    actual_bits = expected_bits.clone()
    scales = torch.ones(1, dtype=torch.float32, device="cuda")

    actual_bits[0, 0] = 0x67
    _assert_encoded(
        actual_bits.view(torch.float8_e4m3fn),
        scales,
        expected_bits.view(torch.float8_e4m3fn),
        scales,
    )

    actual_bits[0, 0] = 0x68
    with pytest.raises(AssertionError, match="up to 3 representable steps"):
        _assert_encoded(
            actual_bits.view(torch.float8_e4m3fn),
            scales,
            expected_bits.view(torch.float8_e4m3fn),
            scales,
        )


@dataclass
class _CacheState:
    generator: torch.Generator
    request_slots: torch.Tensor
    block_table: torch.Tensor
    tail_k: torch.Tensor
    tail_gate: torch.Tensor
    index_values: torch.Tensor
    index_scales: torch.Tensor
    ape: torch.Tensor


def _state(
    requests: int, *, tail_size: int = 7, table_cols: int = 2, seed: int = 0
) -> _CacheState:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    block_table = (
        torch.arange(requests * table_cols, dtype=torch.int32, device="cuda").view(
            requests, table_cols
        )
        + 1
    )
    num_pages = requests * table_cols + 1
    tail_shape = (requests + 2, tail_size, _DIM)
    return _CacheState(
        generator=generator,
        request_slots=torch.arange(1, requests + 1, dtype=torch.int32, device="cuda"),
        block_table=block_table,
        tail_k=_random(tail_shape, generator, 1.0),
        tail_gate=_random(tail_shape, generator, 1.0),
        index_values=torch.zeros(
            num_pages,
            _ROWS_PER_PAGE,
            _DIM,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        index_scales=torch.full(
            (num_pages, _ROWS_PER_PAGE, 1),
            -1.0,
            dtype=torch.float32,
            device="cuda",
        ),
        ape=torch.randn(
            _POOL, _DIM, dtype=torch.float32, device="cuda", generator=generator
        ),
    )


def _reference_append(
    state: _CacheState,
    keys: torch.Tensor,
    gates: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    tail_k = state.tail_k.clone()
    tail_gate = state.tail_gate.clone()
    values = state.index_values.clone()
    scales = state.index_scales.clone()
    steps = keys.shape[1]

    for req in range(keys.shape[0]):
        request_slot = int(state.request_slots[req])
        first_position = int(seq_lens[req]) - steps
        for step in range(steps):
            position = first_position + step
            tail_slot = position % tail_k.shape[1]
            tail_k[request_slot, tail_slot] = keys[req, step]
            tail_gate[request_slot, tail_slot] = gates[req, step]
            if position % _POOL != _POOL - 1:
                continue

            pool_start = position - (_POOL - 1)
            tail_slots = torch.arange(pool_start, position + 1, device="cuda")
            tail_slots.remainder_(tail_k.shape[1])
            pooled, pooled_scale = _reference_compress(
                tail_k[request_slot, tail_slots].unsqueeze(0).contiguous(),
                tail_gate[request_slot, tail_slots].unsqueeze(0).contiguous(),
                state.ape,
            )
            pool_id = position // _POOL
            table_col = pool_id // _ROWS_PER_PAGE
            page = int(state.block_table[req, table_col])
            row = pool_id % _ROWS_PER_PAGE
            values[page, row] = pooled[0]
            scales[page, row, 0] = pooled_scale[0]
    return tail_k, tail_gate, values, scales


def test_prefill_write_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7)
    keys = _random((32, _POOL, _DIM), generator, 0.3)
    gates = _random((32, _POOL, _DIM), generator, 1.5)
    ape = torch.randn(
        _POOL, _DIM, dtype=torch.float32, device="cuda", generator=generator
    )
    write_slots = torch.arange(32, dtype=torch.int64, device="cuda") + _ROWS_PER_PAGE
    values = torch.zeros(
        4, _ROWS_PER_PAGE, _DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    scales = torch.full(
        (4, _ROWS_PER_PAGE, 1), -1.0, dtype=torch.float32, device="cuda"
    )

    kpool_prefill_write(keys, gates, write_slots, values, scales, ape)

    expected_values, expected_scales = _reference_compress(keys, gates, ape)
    pages = write_slots // _ROWS_PER_PAGE
    rows = write_slots % _ROWS_PER_PAGE
    _assert_encoded(
        values[pages, rows], scales[pages, rows, 0], expected_values, expected_scales
    )


def _reference_tail_write(
    keys: torch.Tensor,
    gates: torch.Tensor,
    tail_k: torch.Tensor,
    tail_gate: torch.Tensor,
    source_starts: torch.Tensor,
    destination_slots: torch.Tensor,
    destination_positions: torch.Tensor,
    valid_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_k = tail_k.clone()
    expected_gate = tail_gate.clone()
    metadata = zip(
        source_starts.tolist(),
        destination_slots.tolist(),
        destination_positions.tolist(),
        valid_counts.tolist(),
    )
    for source, destination_slot, destination_position, count in metadata:
        if not (0 <= destination_slot < tail_k.shape[0]) or destination_position < 0:
            continue
        for offset in range(min(count, _POOL)):
            source_row = source + offset
            if not 0 <= source_row < keys.shape[0]:
                continue
            destination_row = (destination_position + offset) % tail_k.shape[1]
            expected_k[destination_slot, destination_row] = keys[source_row]
            expected_gate[destination_slot, destination_row] = gates[source_row]
    return expected_k, expected_gate


def test_prefill_tail_write_masks_padded_and_invalid_rows() -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    keys = _random((12, _DIM), generator, 0.3)
    gates = _random((12, _DIM), generator, 1.5)
    tail_k = _random((5, 7, _DIM), generator, 1.0)
    tail_gate = _random((5, 7, _DIM), generator, 1.0)
    initial_k = tail_k.clone()
    initial_gate = tail_gate.clone()
    source_starts = torch.tensor([1, 8, 4, 99, -3], device="cuda")
    destination_slots = torch.tensor([1, 2, 0, 3, 4], device="cuda")
    destination_positions = torch.tensor([6, 13, 4, 1, -1], device="cuda")
    valid_counts = torch.tensor([3, 2, 0, 0, 2], dtype=torch.int32, device="cuda")
    expected = _reference_tail_write(
        keys,
        gates,
        initial_k,
        initial_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
    )

    kpool_prefill_tail_write(
        keys,
        gates,
        tail_k,
        tail_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
        pool_size=_POOL,
    )

    assert torch.equal(tail_k, expected[0])
    assert torch.equal(tail_gate, expected[1])
    assert torch.equal(tail_k[0], initial_k[0])
    assert torch.equal(tail_gate[0], initial_gate[0])


@pytest.mark.parametrize("num_tokens", [1, 7, 179, 206])
def test_prefill_tail_write_tracks_runtime_source_bound(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(32 + num_tokens)
    keys = _random((num_tokens, _DIM), generator, 0.3)
    gates = _random((num_tokens, _DIM), generator, 1.5)
    tail_k = _random((5, 7, _DIM), generator, 1.0)
    tail_gate = _random((5, 7, _DIM), generator, 1.0)
    initial_k = tail_k.clone()
    initial_gate = tail_gate.clone()
    source_starts = torch.tensor(
        [0, num_tokens - 1, num_tokens, max(num_tokens - 2, 0), -1],
        device="cuda",
    )
    destination_slots = torch.arange(5, device="cuda")
    destination_positions = torch.tensor([0, 6, 3, 4, 2], device="cuda")
    valid_counts = torch.tensor([4, 4, 4, 3, 2], device="cuda")
    expected = _reference_tail_write(
        keys,
        gates,
        initial_k,
        initial_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
    )

    kpool_prefill_tail_write(
        keys,
        gates,
        tail_k,
        tail_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
        pool_size=_POOL,
    )

    assert torch.equal(tail_k, expected[0])
    assert torch.equal(tail_gate, expected[1])


def test_prefill_tail_write_empty_metadata_is_noop() -> None:
    generator = torch.Generator(device="cuda").manual_seed(33)
    keys = _random((8, _DIM), generator, 0.3)
    gates = _random((8, _DIM), generator, 1.5)
    tail_k = _random((2, 7, _DIM), generator, 1.0)
    tail_gate = _random((2, 7, _DIM), generator, 1.0)
    initial_k = tail_k.clone()
    initial_gate = tail_gate.clone()
    metadata = [torch.empty(0, dtype=torch.int32, device="cuda") for _ in range(4)]

    kpool_prefill_tail_write(
        keys,
        gates,
        tail_k,
        tail_gate,
        *metadata,
        pool_size=_POOL,
    )

    assert torch.equal(tail_k, initial_k)
    assert torch.equal(tail_gate, initial_gate)


def test_prefill_tail_write_accepts_distinct_row_strides() -> None:
    generator = torch.Generator(device="cuda").manual_seed(30)
    keys = _random((12, _DIM + 8), generator, 0.3)[:, :_DIM]
    gates = _random((12, _DIM), generator, 1.5)
    tail_k = _random((5, 7, _DIM), generator, 1.0)
    tail_gate = _random((5, 7, _DIM + 5), generator, 1.0)[:, :, :_DIM]
    assert keys.stride() != gates.stride()
    assert tail_k.stride() != tail_gate.stride()

    initial_k = tail_k.clone()
    initial_gate = tail_gate.clone()
    source_starts = torch.tensor([0, 5, 8], device="cuda")
    destination_slots = torch.tensor([1, 2, 4], device="cuda")
    destination_positions = torch.tensor([6, 2, 5], device="cuda")
    valid_counts = torch.tensor([4, 2, 3], dtype=torch.int32, device="cuda")
    expected = _reference_tail_write(
        keys,
        gates,
        initial_k,
        initial_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
    )

    kpool_prefill_tail_write(
        keys,
        gates,
        tail_k,
        tail_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
        pool_size=_POOL,
    )

    assert torch.equal(tail_k, expected[0])
    assert torch.equal(tail_gate, expected[1])


def test_prefill_tail_write_graph_replay_tracks_live_counts() -> None:
    generator = torch.Generator(device="cuda").manual_seed(31)
    keys = _random((16, _DIM), generator, 0.3)
    gates = _random((16, _DIM), generator, 1.5)
    tail_k = _random((5, 7, _DIM), generator, 1.0)
    tail_gate = _random((5, 7, _DIM), generator, 1.0)
    initial_k = tail_k.clone()
    initial_gate = tail_gate.clone()
    source_starts = torch.tensor([0, 4, 8, 12], device="cuda")
    destination_slots = torch.tensor([1, 2, 3, 0], device="cuda")
    destination_positions = torch.tensor([5, 9, 13, 0], device="cuda")
    valid_counts = torch.tensor([1, 0, 2, 0], dtype=torch.int32, device="cuda")

    def run() -> None:
        kpool_prefill_tail_write(
            keys,
            gates,
            tail_k,
            tail_gate,
            source_starts,
            destination_slots,
            destination_positions,
            valid_counts,
            pool_size=_POOL,
        )

    run()
    torch.cuda.synchronize()
    tail_k.copy_(initial_k)
    tail_gate.copy_(initial_gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    tail_k.copy_(initial_k)
    tail_gate.copy_(initial_gate)
    source_starts.copy_(torch.tensor([2, 6, 10, 0], device="cuda"))
    destination_slots.copy_(torch.tensor([4, 3, 2, 0], device="cuda"))
    destination_positions.copy_(torch.tensor([4, 11, 6, 0], device="cuda"))
    valid_counts.copy_(torch.tensor([4, 1, 0, 3], dtype=torch.int32, device="cuda"))
    expected = _reference_tail_write(
        keys,
        gates,
        initial_k,
        initial_gate,
        source_starts,
        destination_slots,
        destination_positions,
        valid_counts,
    )

    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(tail_k, expected[0])
    assert torch.equal(tail_gate, expected[1])
    assert not torch.equal(tail_k[0], initial_k[0])
    assert not torch.equal(tail_gate[0], initial_gate[0])


def test_decode_append_matches_reference() -> None:
    state = _state(4, seed=11)
    keys = _random((4, 3, _DIM), state.generator, 0.3)
    gates = _random((4, 3, _DIM), state.generator, 1.5)
    seq_lens = torch.tensor([63, 64, 65, 68], dtype=torch.int32, device="cuda")
    expected = _reference_append(state, keys, gates, seq_lens)

    kpool_decode_append(
        keys,
        gates,
        state.tail_k,
        state.tail_gate,
        seq_lens,
        state.request_slots,
        state.block_table,
        state.index_values,
        state.index_scales,
        state.ape,
    )

    assert torch.equal(state.tail_k, expected[0])
    assert torch.equal(state.tail_gate, expected[1])
    _assert_encoded(state.index_values, state.index_scales, expected[2], expected[3])


def test_decode_append_defers_unmaterialized_pool() -> None:
    state = _state(1, seed=17)
    committed_slot = 64 % state.tail_k.shape[1]
    committed_k = state.tail_k[1, committed_slot].clone()
    committed_gate = state.tail_gate[1, committed_slot].clone()
    draft_k = _random((1, 3, _DIM), state.generator, 0.3)
    draft_gate = _random((1, 3, _DIM), state.generator, 1.5)
    unavailable_table = state.block_table[:, :1]

    for step in range(3):
        kpool_decode_append(
            draft_k[:, step : step + 1].contiguous(),
            draft_gate[:, step : step + 1].contiguous(),
            state.tail_k,
            state.tail_gate,
            torch.tensor([66 + step], dtype=torch.int32, device="cuda"),
            state.request_slots,
            unavailable_table,
            state.index_values,
            state.index_scales,
            state.ape,
        )

    draft_slots = torch.arange(65, 68, device="cuda").remainder(state.tail_k.shape[1])
    assert torch.equal(state.tail_k[1, draft_slots], draft_k[0])
    assert torch.equal(state.tail_gate[1, draft_slots], draft_gate[0])
    assert torch.equal(state.tail_k[1, committed_slot], committed_k)
    assert (state.index_scales == -1).all()

    accepted_k = _random((1, 3, _DIM), state.generator, 0.3)
    accepted_gate = _random((1, 3, _DIM), state.generator, 1.5)
    kpool_decode_append(
        accepted_k,
        accepted_gate,
        state.tail_k,
        state.tail_gate,
        torch.tensor([68], dtype=torch.int32, device="cuda"),
        state.request_slots,
        state.block_table,
        state.index_values,
        state.index_scales,
        state.ape,
    )

    expected_values, expected_scales = _reference_compress(
        torch.cat((committed_k.unsqueeze(0), accepted_k[0])).unsqueeze(0),
        torch.cat((committed_gate.unsqueeze(0), accepted_gate[0])).unsqueeze(0),
        state.ape,
    )
    assert torch.equal(state.tail_k[1, draft_slots], accepted_k[0])
    _assert_encoded(
        state.index_values[2, 0].unsqueeze(0),
        state.index_scales[2, 0, 0].unsqueeze(0),
        expected_values,
        expected_scales,
    )


def test_decode_append_skips_graph_padding_slot() -> None:
    state = _state(2, seed=23)
    state.request_slots[1] = 0
    state.block_table[1].zero_()
    sentinel = (
        state.tail_k[0].clone(),
        state.tail_gate[0].clone(),
        state.index_values[0].clone(),
        state.index_scales[0].clone(),
    )
    keys = _random((2, 3, _DIM), state.generator, 0.3)
    gates = _random((2, 3, _DIM), state.generator, 1.5)

    kpool_decode_append(
        keys,
        gates,
        state.tail_k,
        state.tail_gate,
        torch.tensor([67, 67], dtype=torch.int32, device="cuda"),
        state.request_slots,
        state.block_table,
        state.index_values,
        state.index_scales,
        state.ape,
    )

    actual = (
        state.tail_k[0],
        state.tail_gate[0],
        state.index_values[0],
        state.index_scales[0],
    )
    assert all(torch.equal(a, b) for a, b in zip(actual, sentinel, strict=True))
