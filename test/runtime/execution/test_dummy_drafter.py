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

"""Synthetic proposals preserve the selected token and ignore execution metadata.

These tensor-level tests cover proposal determinism, not target acceptance or
equivalence to a non-speculative greedy trajectory.
"""

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.dummy import DummyDrafter
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.runtime_states import RuntimeStates


def _drafter(*, verify_width=4, device="cpu") -> DummyDrafter:
    states = RuntimeStates(3, 128, verify_width, device=device)
    states.valid_cache_lengths.copy_(
        torch.tensor([9, 10, 11, 12], dtype=torch.int32, device=device)
    )
    buffers = InputBuffers(
        max_bs=3,
        max_num_tokens=3 * verify_width,
        state_write_padding_pool_index=-1,
        device=device,
    )
    buffers.req_pool_indices_buf.copy_(
        torch.tensor([2, 0, 3], dtype=torch.int64, device=device)
    )
    return DummyDrafter(
        spec_num_tokens=verify_width,
        spec_num_steps=verify_width - 1,
        draft_model_runner=None,
        runtime_states=states,
        input_buffers=buffers,
        attn_backend=None,
        token_to_kv_pool=None,
        vocab_size=128,
    )


def _context(*, bs, num_extends, verify_width):
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=num_extends,
        input_num_tokens=num_extends + (bs - num_extends) * verify_width,
        forward_mode=ForwardMode.MIXED if num_extends else ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(
            num_extends, num_extends, bs - num_extends, verify_width
        ),
    )


def _propose(drafter, *, selected, lengths, slots):
    device = drafter.runtime_states.valid_cache_lengths.device
    indices = torch.tensor(slots, dtype=torch.int64, device=device)
    drafter.runtime_states.valid_cache_lengths.index_copy_(
        0, indices, torch.tensor(lengths, dtype=torch.int32, device=device)
    )
    drafter.input_buffers.req_pool_indices_buf[: len(slots)].copy_(indices)
    output = torch.tensor(selected, dtype=torch.int32, device=device).repeat_interleave(
        drafter.spec_num_tokens
    )
    return drafter.run(
        _context(bs=len(slots), num_extends=0, verify_width=drafter.spec_num_tokens),
        None,
        output,
        torch.ones(len(slots), dtype=torch.int32, device=device),
    )


def test_dummy_drafter_mixed_window_and_determinism() -> None:
    drafter = _drafter()
    ctx = _context(bs=3, num_extends=1, verify_width=4)
    output = torch.tensor([7, 20, 21, 22, 23, 30, 31, 32, 33], dtype=torch.int32)
    accepts = torch.tensor([1, 3, 1], dtype=torch.int32)
    result = drafter.run(ctx, None, output, accepts)
    assert result.shape == (3, 4)
    assert result[:, 0].tolist() == [7, 22, 30]
    torch.testing.assert_close(result, drafter.run(ctx, None, output, accepts))
    assert result[:, 1:].min() >= 0
    assert result[:, 1:].max() < 128

    drafter.runtime_states.valid_cache_lengths[2] += 1
    changed = drafter.run(ctx, None, output, accepts)
    torch.testing.assert_close(changed, result, rtol=0, atol=0)
    assert (
        drafter.idle_forward_global_num_tokens([9], [3], global_num_decodes=[3]) == []
    )


def test_dummy_is_registered_without_a_draft_model() -> None:
    assert get_drafter_impl("DUMMY", None) is DummyDrafter
    drafter = _drafter()
    with pytest.raises(ValueError, match="must not load a draft model"):
        DummyDrafter(
            spec_num_tokens=4,
            draft_model_runner=object(),
            runtime_states=drafter.runtime_states,
            input_buffers=drafter.input_buffers,
            vocab_size=128,
        )


def test_dummy_candidates_do_not_require_runtime_metadata() -> None:
    drafter = DummyDrafter(spec_num_tokens=4, vocab_size=128)
    ctx = _context(bs=1, num_extends=1, verify_width=4)
    selected = torch.tensor([17], dtype=torch.int32)
    accepts = torch.ones(1, dtype=torch.int32)
    result = drafter.run(ctx, None, selected, accepts)
    expected = _drafter().run(ctx, None, selected, accepts)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("verify_width", [4, 8])
def test_same_prefix_relocated_to_another_pool_slot_keeps_its_window(verify_width):
    drafter = _drafter(verify_width=verify_width)
    first = _propose(drafter, selected=[17], lengths=[127], slots=[1]).clone()
    relocated = _propose(drafter, selected=[17], lengths=[127], slots=[3])
    torch.testing.assert_close(relocated, first, rtol=0, atol=0)


@pytest.mark.parametrize("verify_width", [4, 8])
def test_solo_batch_and_reordered_relocated_requests_propose_identical_windows(
    verify_width,
):
    selected, lengths = [7, 22, 30], [33, 65, 129]
    drafter = _drafter(verify_width=verify_width)
    batch = _propose(
        drafter, selected=selected, lengths=lengths, slots=[0, 1, 2]
    ).clone()
    for row in range(3):
        solo = _propose(
            _drafter(verify_width=verify_width),
            selected=selected[row : row + 1],
            lengths=lengths[row : row + 1],
            slots=[3],
        )
        torch.testing.assert_close(solo[0], batch[row], rtol=0, atol=0)
    order = [2, 0, 1]
    reordered = _propose(
        _drafter(verify_width=verify_width),
        selected=[selected[row] for row in order],
        lengths=[lengths[row] for row in order],
        slots=[1, 3, 0],
    )
    torch.testing.assert_close(reordered, batch[order], rtol=0, atol=0)


def test_proposals_read_changed_selected_tokens_and_ignore_cache_frontiers():
    drafter = _drafter(verify_width=8)
    first = _propose(drafter, selected=[7, 22], lengths=[33, 65], slots=[1, 2]).clone()
    changed = _propose(drafter, selected=[8, 22], lengths=[33, 66], slots=[1, 2])
    assert changed[:, 0].tolist() == [8, 22]
    assert not torch.equal(changed[0, 1:], first[0, 1:])
    torch.testing.assert_close(changed[1], first[1], rtol=0, atol=0)


@pytest.mark.parametrize("verify_width", [4, 8])
def test_prefill_drafting_before_frontier_commit_ignores_cached_prefix_length(
    verify_width,
):
    drafter = _drafter(verify_width=verify_width)
    ctx = _context(bs=3, num_extends=3, verify_width=verify_width)
    selected = torch.tensor([17, 17, 17], dtype=torch.int32)
    accepts = torch.ones(3, dtype=torch.int32)

    # The forward prologue installs each cached prefix. Dummy runs before the
    # executor advances these frontiers by the new prefill input lengths.
    slots = torch.tensor([2, 0, 3], dtype=torch.int64)
    prefixes = torch.tensor([0, 22, 5], dtype=torch.int32)
    drafter.input_buffers.req_pool_indices_buf.copy_(slots)
    drafter.input_buffers.input_lengths_buf[:3].copy_(128 - prefixes)
    drafter.runtime_states.reset_states(slots, prefixes)
    first = drafter.run(ctx, None, selected, accepts).clone()
    torch.testing.assert_close(first, first[:1].expand_as(first), rtol=0, atol=0)
    torch.testing.assert_close(
        drafter.runtime_states.valid_cache_lengths.index_select(0, slots),
        prefixes,
        rtol=0,
        atol=0,
    )

    # The same selected roots after another prefix-cache hit and slot order
    # retain their proposals even though the uncommitted frontiers differ.
    relocated_slots = torch.tensor([1, 3, 0], dtype=torch.int64)
    relocated_prefixes = torch.tensor([22, 0, 9], dtype=torch.int32)
    drafter.input_buffers.req_pool_indices_buf.copy_(relocated_slots)
    drafter.input_buffers.input_lengths_buf[:3].copy_(128 - relocated_prefixes)
    drafter.runtime_states.reset_states(relocated_slots, relocated_prefixes)
    relocated = drafter.run(ctx, None, selected, accepts)
    torch.testing.assert_close(relocated, first, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graphs")
def test_captured_proposals_read_live_tokens_and_ignore_frontiers_and_slots():
    drafter = _drafter(verify_width=8, device="cuda")
    ctx = _context(bs=3, num_extends=0, verify_width=8)
    output = torch.tensor(
        [7, 22, 30], dtype=torch.int32, device="cuda"
    ).repeat_interleave(8)
    accepts = torch.ones(3, dtype=torch.int32, device="cuda")
    first = drafter.run(ctx, None, output, accepts).clone()
    for _ in range(3):
        drafter.run(ctx, None, output, accepts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = drafter.run(ctx, None, output, accepts)
    # Relocate rows and change the uncommitted frontiers, retaining every pointer.
    drafter.input_buffers.req_pool_indices_buf.copy_(
        torch.tensor([1, 3, 0], dtype=torch.int64, device="cuda")
    )
    drafter.runtime_states.valid_cache_lengths.copy_(
        torch.tensor([91, 3, 100, 27], dtype=torch.int32, device="cuda")
    )
    graph.replay()
    torch.testing.assert_close(captured, first, rtol=0, atol=0)
    output.copy_(
        torch.tensor([13, 77, 28], dtype=torch.int32, device="cuda").repeat_interleave(
            8
        )
    )
    drafter.runtime_states.valid_cache_lengths.add_(1)
    expected = drafter.run(ctx, None, output, accepts).clone()
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert not torch.equal(captured, first)
