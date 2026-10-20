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

"""Parity tests: the fused frontier advance must equal the eager op chain."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.metadata import (
    torch_advance_accepted_frontier,
)
from tokenspeed_kernel.ops.metadata import advance_accepted_frontier

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _batch(batch, num_extends, pool, padding_rows, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    slots = torch.randperm(pool - 1, device=dev)[:batch].to(torch.int64)
    slots[batch - padding_rows :] = pool - 1
    input_lengths = torch.randint(1, 6, (batch,), device=dev, dtype=torch.int32)
    input_lengths[num_extends:] = 6
    accept_lengths = torch.randint(1, 7, (batch,), device=dev, dtype=torch.int32)
    accept_lengths[:num_extends] = 0
    valid = torch.randint(0, 500, (pool,), device=dev, dtype=torch.int32)
    return slots, input_lengths, accept_lengths, valid


@pytest.mark.parametrize(
    ("batch", "num_extends", "padding_rows"),
    (
        (7, 3, 0),
        (16, 0, 5),
        (4, 4, 0),
        (33, 1, 2),
        (1, 0, 0),
        # Several programs.
        (5000, 1700, 9),
    ),
)
def test_fused_advance_matches_eager_chain(batch, num_extends, padding_rows):
    pool = max(64, batch + 1)
    slots, inputs, accepts, valid = _batch(
        batch, num_extends, pool, padding_rows, seed=batch
    )
    expected = valid.clone()
    torch_advance_accepted_frontier(
        slots, inputs, accepts, expected, num_extends, pool - 1
    )

    advance_accepted_frontier(slots, inputs, accepts, valid, num_extends, pool - 1)
    torch.cuda.synchronize()

    assert torch.equal(valid, expected)


def test_fused_advance_rejects_mistyped_cache_lengths():
    pool = 16
    slots, inputs, accepts, valid = _batch(5, 2, pool, 1, seed=11)
    with pytest.raises(ValueError, match="int32"):
        advance_accepted_frontier(slots, inputs, accepts, valid.long(), 2, pool - 1)


def test_fused_advance_compiles_once_across_batch_shapes():
    """Batch and extend counts change every round; none may key a new binary."""
    from tokenspeed_kernel.ops.metadata.accepted_frontier import (
        _advance_accepted_frontier_kernel,
    )
    from utils import assert_no_triton_compile

    # The pool and its padding slot are fixed for a server's lifetime.
    pool = 512

    def run(batch, num_extends):
        slots, inputs, accepts, valid = _batch(batch, num_extends, pool, 0, seed=batch)
        expected = valid.clone()
        torch_advance_accepted_frontier(
            slots, inputs, accepts, expected, num_extends, pool - 1
        )
        advance_accepted_frontier(slots, inputs, accepts, valid, num_extends, pool - 1)
        assert torch.equal(valid, expected)

    run(7, 3)
    with assert_no_triton_compile(_advance_accepted_frontier_kernel):
        for batch, num_extends in (
            (1, 0),
            (1, 1),
            (16, 16),
            (17, 1),
            (64, 5),
            (300, 33),
        ):
            run(batch, num_extends)


def test_advance_runs_on_cpu_tensors_as_tensor_ops():
    """Devices without the kernel (NPU, CPU tests) take the tensor path."""
    pool = 16
    slots, inputs, accepts, valid = _batch(6, 2, pool, 1, seed=5)
    cpu = [t.cpu() for t in (slots, inputs, accepts, valid)]
    advance_accepted_frontier(slots, inputs, accepts, valid, 2, pool - 1)
    advance_accepted_frontier(*cpu, 2, pool - 1)
    torch.cuda.synchronize()
    assert torch.equal(valid.cpu(), cpu[3])
    with pytest.raises(ValueError, match="colocated"):
        advance_accepted_frontier(cpu[0], inputs, accepts, valid, 2, pool - 1)
