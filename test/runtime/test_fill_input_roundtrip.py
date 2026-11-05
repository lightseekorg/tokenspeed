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

"""fill_input_buffers lands the extend rows' tokens through one cast per field.

The token path converts the plan's Python lists via numpy (a C-level int32
store instead of boxing every token) and sizes the pinned staging from the
converted tensors; these tests pin the round trip: input ids and shifted ids
land intact, the host-side extend facts keep their values, and the padding
tail is still scrubbed — a staging sized wrong would leave stale ids past the
active prefix, which is exactly what the scrub exists to prevent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution import input_buffer
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates

VOCAB_SIZE = 128
MAX_BS, MAX_TOKENS, PAD_POOL = 8, 64, 5


@pytest.fixture(params=["cpu", "cuda"])
def buffers(request, monkeypatch):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "cpu":
        # The kernels the fill path launches are GPU-only; the CPU runs here
        # check the list/tensor plumbing around them, so they stand in with
        # the same tensor contract.
        def bulk(self, *specs):
            return [torch.empty(n, dtype=dtype) for n, dtype in specs]

        def positions(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum, out):
            values = [
                torch.arange(int(p), int(p + n))
                for p, n in zip(extend_prefix_lens, extend_seq_lens)
            ]
            out.copy_(torch.cat(values) if values else torch.empty(0))
            return out, None

        def decode_positions(
            positions_ptr,
            seq_lens_out_ptr,
            req_pool_indices,
            valid_cache_lengths,
            uniform_input_length,
        ):
            starts = valid_cache_lengths[req_pool_indices]
            positions_ptr.copy_(
                (starts[:, None] + torch.arange(uniform_input_length)).flatten()
            )
            seq_lens_out_ptr.copy_(starts + uniform_input_length)

        monkeypatch.setattr(InputBuffers, "_bulk_pinned", bulk)
        monkeypatch.setattr(input_buffer, "compute_position_triton", positions)
        monkeypatch.setattr(input_buffer, "fused_decode_input_prep", decode_positions)
    ib = InputBuffers(
        max_bs=MAX_BS,
        max_num_tokens=MAX_TOKENS,
        state_write_padding_pool_index=PAD_POOL,
        device=device,
    )
    runtime = RuntimeStates(
        req_pool_size=8, vocab_size=VOCAB_SIZE, output_length=1, device=device
    )
    return ib, runtime


def _op(rids, slots, lengths, prefixes, prompt_lens, overrides=None):
    """The fill reads only the plan's attribute surface; the values keep the
    scheduler's invariants (len(input_ids) == sum(input_lengths[:num_extends]),
    prefix + chunk <= prompt)."""
    ids = []
    for length, prefix, prompt in zip(lengths, prefixes, prompt_lens):
        assert prefix + length <= prompt
        ids.extend(range(prompt)[prefix : prefix + length])
    return SimpleNamespace(
        request_ids=list(rids),
        request_pool_indices=list(slots),
        input_lengths=list(lengths),
        prefill_lengths=list(prompt_lens),
        input_ids=ids,
        shifted_input_ids=ids[1:] + [-1] if ids else [],
        extend_prefix_lens=list(prefixes),
        extend_replay_lens=[0] * len(rids),
        decode_input_ids=overrides,
        num_extends=lambda: len(prefixes),
    )


def _fill(ib, runtime, op):
    n = op.num_extends()
    runtime.reset_states(
        torch.tensor(op.request_pool_indices[:n], dtype=torch.int64, device=ib.device),
        torch.tensor(op.extend_prefix_lens, dtype=torch.int32, device=ib.device),
    )
    total = sum(op.input_lengths)
    ib.fill_input_buffers(op, runtime, total)
    return total


def test_mid_chunk_predicate_needs_every_row_non_final(buffers):
    ib, runtime = buffers
    # Two extends, both mid-prompt chunks (prefix + chunk < prompt).
    op = _op(["a", "b"], [1, 2], [4, 4], [0, 4], [16, 16])
    _fill(ib, runtime, op)
    assert ib.all_extends_mid_chunk is True

    # The second row's chunk completes its prompt: not all mid-chunk.
    op = _op(["a", "b"], [1, 2], [4, 4], [0, 12], [16, 16])
    _fill(ib, runtime, op)
    assert ib.all_extends_mid_chunk is False

    # A decode row in the batch (num_extends < batch_size).
    op = _op(["a", "c"], [1, 3], [4, 1], [0], [16, 5])
    _fill(ib, runtime, op)
    assert ib.all_extends_mid_chunk is False

    # Decode-only rounds never claim mid-chunk.
    op = _op(["a", "b"], [1, 2], [1, 1], [], [16, 16])
    _fill(ib, runtime, op)
    assert ib.all_extends_mid_chunk is False


def test_tokens_and_extend_facts_land_intact(buffers):
    ib, runtime = buffers
    rids, slots = ["a", "b"], [1, 2]
    lengths, prefixes, prompt_lens = [4, 4], [0, 4], [16, 16]
    op = _op(rids, slots, lengths, prefixes, prompt_lens)
    total = _fill(ib, runtime, op)

    assert ib.input_ids_buf[:total].tolist() == op.input_ids
    assert ib.shifted_prefill_ids_buf[:total].tolist() == op.shifted_input_ids
    assert ib.input_lengths_buf[: len(slots)].tolist() == lengths
    assert ib.req_pool_indices_buf[: len(slots)].tolist() == slots
    n = op.num_extends()
    assert ib.extend_prefix_lens_cpu[:n].tolist() == prefixes
    assert ib.extend_seq_lens_cpu[:n].tolist() == lengths[:n]
    assert ib.extend_prompt_lens_cpu[:n].tolist() == prompt_lens[:n]

    # The padding tail keeps its safe defaults: a mis-sized staging copy
    # would leave stale ids past the active prefix for padded rows to read.
    assert ib.input_ids_buf[total:].tolist() == [1] * (ib.max_num_tokens - total)
