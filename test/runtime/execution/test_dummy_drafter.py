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

"""Model-free draft windows preserve the target's accepted token."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.dummy import DummyDrafter


def _drafter() -> DummyDrafter:
    return DummyDrafter(
        spec_num_tokens=4,
        spec_num_steps=3,
        draft_model_runner=None,
        runtime_states=SimpleNamespace(
            valid_cache_lengths=torch.tensor([9, 10, 11, 12], dtype=torch.int32)
        ),
        input_buffers=SimpleNamespace(
            req_pool_indices_buf=torch.tensor([2, 0, 3], dtype=torch.int32)
        ),
        attn_backend=None,
        token_to_kv_pool=None,
        vocab_size=128,
    )


def test_dummy_drafter_mixed_window_and_determinism() -> None:
    drafter = _drafter()
    ctx = SimpleNamespace(bs=3, num_extends=1)
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
    assert not torch.equal(result[0, 1:], changed[0, 1:])
    assert drafter.idle_forward_global_num_tokens([9], [3]) == []


def test_dummy_is_registered_without_a_draft_model() -> None:
    assert get_drafter_impl("DUMMY", None) is DummyDrafter
    with pytest.raises(ValueError, match="must not load a draft model"):
        DummyDrafter(
            spec_num_tokens=4,
            draft_model_runner=object(),
            runtime_states=SimpleNamespace(),
            input_buffers=SimpleNamespace(),
            vocab_size=128,
        )
