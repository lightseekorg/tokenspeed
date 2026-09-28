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

"""PD bootstrap inputs must preserve remote candidates without a CUDA host wait."""

from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates

register_cuda_ci(est_time=10, suite="runtime-1gpu")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("mixed", [False, True])
def test_pd_bootstrap_input_prep_does_not_synchronize(width, mixed):
    buffers = InputBuffers(
        max_bs=5, max_num_tokens=32, state_write_padding_pool_index=6, device="cuda"
    )
    states = RuntimeStates(6, 128, width, device="cuda")
    states.valid_cache_lengths.fill_(64)
    before = torch.arange(7 * width, dtype=torch.int32).reshape(7, width) + 50
    flags = torch.tensor([True, False, True, True, True, True, True])
    slots = ([2] if mixed else []) + [1, 3, 4]
    lengths = ([2] if mixed else []) + [width] * 3
    op = SimpleNamespace(
        request_ids=[f"r{slot}" for slot in slots],
        request_pool_indices=slots,
        input_lengths=lengths,
        prefill_lengths=[9] + [64] * 3 if mixed else [64] * 3,
        input_ids=[17, 18] if mixed else [],
        shifted_input_ids=[18, -1] if mixed else [],
        extend_prefix_lens=[7] if mixed else [],
        extend_replay_lens=[0] if mixed else [],
        decode_input_ids=[31, 41, -1],
        num_extends=lambda: int(mixed),
    )
    # Compile metadata kernels before checking the forward-thread hot path.
    states.future_input_map.copy_(before)
    buffers.fill_input_buffers(op, states, sum(lengths), ngram_inputs=None)
    states.future_input_map.copy_(before)
    states.remote_spec_candidate_ready.copy_(flags)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        buffers.fill_input_buffers(op, states, sum(lengths), ngram_inputs=None)
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    expected = before.clone()
    expected[1].fill_(31)  # Bootstrap with no candidates uses a safe dummy tail.
    expected[3, 0] = 41  # Remote candidates survive an explicit first-token override.
    torch.testing.assert_close(states.future_input_map.cpu(), expected, atol=0, rtol=0)
    assert states.remote_spec_candidate_ready.cpu().tolist() == [
        True,
        False,
        True,
        False,
        False,
        True,
        True,
    ]
    expected_ids = ([17, 18] if mixed else []) + expected[[1, 3, 4]].flatten().tolist()
    assert buffers.input_ids_buf[: sum(lengths)].cpu().tolist() == expected_ids
    assert buffers.force_single_token_verify_buf.cpu().tolist() == (
        [False, True, False, False, False]
        if mixed
        else [True, False, False, False, False]
    )
