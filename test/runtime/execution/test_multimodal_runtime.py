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

"""Tests for MultimodalRuntime (mrope overrides factored out of ModelExecutor)."""

from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime


class _FakeInputBuffers:
    def __init__(self, capacity: int = 64):
        self.max_num_tokens = capacity
        self.positions_buf = torch.arange(capacity, dtype=torch.int64)
        self.mrope_positions_buf = torch.zeros(3, capacity, dtype=torch.int64)


class _FakeForwardOp:
    def __init__(self, input_lengths, num_extends=0, extend_prefix_lens=()):
        self.input_lengths = list(input_lengths)
        self.extend_prefix_lens = list(extend_prefix_lens)
        self._num_extends = num_extends

    def num_extends(self):
        return self._num_extends


class _FakeMmContext:
    def __init__(self, mm_inputs):
        self.mm_inputs = mm_inputs

    def has_inputs(self):
        return any(item is not None for item in self.mm_inputs)


def _mm_input(
    delta_scalar=None,
    delta_tensor=None,
    positions=None,
):
    return SimpleNamespace(
        mrope_position_delta_scalar=delta_scalar,
        mrope_position_delta=delta_tensor,
        mrope_positions=positions,
    )


def _runtime(mrope=True, capacity=64):
    ib = _FakeInputBuffers(capacity)
    return MultimodalRuntime(model_is_mrope=mrope, input_buffers=ib, device="cpu"), ib


def test_non_mrope_model_returns_none():
    rt, _ = _runtime(mrope=False)
    op = _FakeForwardOp([1, 1])
    assert rt.build_positions_override(op, None, 2) is None


def test_zero_tokens_returns_none():
    rt, _ = _runtime()
    op = _FakeForwardOp([])
    assert rt.build_positions_override(op, None, 0) is None


def test_decode_text_only_short_circuits_to_base_positions():
    rt, ib = _runtime()
    op = _FakeForwardOp([1, 1])
    out = rt.build_positions_override(op, None, 2)
    assert out.shape == (3, 2)
    assert torch.equal(out, ib.positions_buf[:2].unsqueeze(0).expand(3, -1))


def test_decode_applies_scalar_delta_per_request():
    rt, _ = _runtime()
    op = _FakeForwardOp([1, 1])
    ctx = _FakeMmContext([None, _mm_input(delta_scalar=5)])
    out = rt.build_positions_override(op, ctx, 2)
    # req 0: text -> base 0; req 1: base 1 + delta 5
    assert out[:, 0].tolist() == [0, 0, 0]
    assert out[:, 1].tolist() == [6, 6, 6]


def test_decode_reads_tensor_delta_without_writing_back():
    rt, _ = _runtime()
    op = _FakeForwardOp([1])
    mm = _mm_input(delta_tensor=torch.tensor([[3]], dtype=torch.int64))
    out = rt.build_positions_override(op, _FakeMmContext([mm]), 1)
    # base 0 + delta 3.
    assert out[:, 0].tolist() == [3, 3, 3]
    # The forward runs on the data plane and must not edit the struct it was
    # handed; the control plane resolves the scalar at gather time
    # (multimodal_context_for_forward) instead of memoizing it from here.
    assert mm.mrope_position_delta_scalar is None


def test_decode_staging_ping_pongs_between_two_buffers():
    rt, _ = _runtime()
    op = _FakeForwardOp([1])
    ctx = _FakeMmContext([_mm_input(delta_scalar=1)])
    idx0 = rt._mrope_decode_deltas_cpu_idx
    rt.build_positions_override(op, ctx, 1)
    idx1 = rt._mrope_decode_deltas_cpu_idx
    rt.build_positions_override(op, ctx, 1)
    idx2 = rt._mrope_decode_deltas_cpu_idx
    assert idx0 != idx1 and idx1 != idx2 and idx0 == idx2


def test_prefill_slices_precomputed_positions_by_prefix():
    rt, _ = _runtime()
    # One extend request: 4 new tokens after a prefix of 2.
    positions = torch.arange(30, dtype=torch.int64).reshape(1, -1).repeat(3, 1)
    mm = _mm_input(positions=positions)
    op = _FakeForwardOp([4], num_extends=1, extend_prefix_lens=[2])
    out = rt.build_positions_override(op, _FakeMmContext([mm]), 4)
    assert out.shape == (3, 4)
    assert out[0].tolist() == [2, 3, 4, 5]


def test_prefill_text_request_falls_back_to_linear():
    rt, ib = _runtime()
    op = _FakeForwardOp([4], num_extends=1, extend_prefix_lens=[0])
    out = rt.build_positions_override(op, _FakeMmContext([None]), 4)
    assert torch.equal(out, ib.positions_buf[:4].unsqueeze(0).expand(3, -1))


def test_timing_counts():
    ctx = _FakeMmContext(
        [
            None,
            _mm_input(delta_tensor=torch.tensor([1])),
            _mm_input(),
        ]
    )
    has_mm, mm_count, mm_delta_count = MultimodalRuntime.timing_counts(ctx)
    assert has_mm and mm_count == 2 and mm_delta_count == 1
    has_mm, mm_count, mm_delta_count = MultimodalRuntime.timing_counts(None)
    assert not has_mm and mm_count == 0 and mm_delta_count == 0


def test_wire_drafter_sets_pad_ids_only_when_supported():
    class _Drafter:
        def __init__(self):
            self.ids = None

        def set_mm_pad_substitute_ids(self, ids):
            self.ids = ids

    # A config with no mm pad ids resolves to falsy -> drafter untouched.
    # Text-only, so the multimodal branch that would raise is not taken.
    drafter = _Drafter()
    MultimodalRuntime.wire_drafter(
        drafter,
        SimpleNamespace(
            hf_config=SimpleNamespace(),
            vocab_size=32000,
            is_multimodal_active=False,
        ),
    )
    assert drafter.ids is None
