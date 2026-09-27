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
from tokenspeed_kernel.ops.metadata import advance_accepted_frontier

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _eager_reference(
    req_pool_indices,
    input_lengths,
    accept_lengths,
    valid_cache_lengths,
    num_extends,
    padding_index,
    tail,
    previous_tokens,
    token_mask,
    input_ids,
):
    deltas = torch.cat([input_lengths[:num_extends], accept_lengths[num_extends:]])
    deltas = torch.where(req_pool_indices != padding_index, deltas, 0).to(torch.int32)
    if tail is not None:
        last = (input_lengths.cumsum(0) - input_lengths + deltas - 1).clamp(
            0, input_ids.shape[0] - 1
        )
        current = torch.where(token_mask[last], input_ids[last], -1)
        history = torch.cat([current[:, None], previous_tokens[last, :-1]], dim=1)
        tail[req_pool_indices] = torch.where(
            (deltas > 0)[:, None], history, tail[req_pool_indices]
        )
    valid_cache_lengths.index_add_(0, req_pool_indices, deltas)


def _batch(batch, num_extends, pool, tokens, context, padding_rows, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    slots = torch.randperm(pool - 1, device=dev)[:batch].to(torch.int64)
    slots[batch - padding_rows :] = pool - 1
    input_lengths = torch.randint(1, 6, (batch,), device=dev, dtype=torch.int32)
    input_lengths[num_extends:] = 6
    accept_lengths = torch.randint(1, 7, (batch,), device=dev, dtype=torch.int32)
    accept_lengths[:num_extends] = 0
    valid = torch.randint(0, 500, (pool,), device=dev, dtype=torch.int32)
    tail = torch.randint(-1, 100, (pool, context), device=dev, dtype=torch.int64)
    previous = torch.randint(-1, 100, (tokens, context), device=dev, dtype=torch.int64)
    mask = torch.rand(tokens, device=dev) > 0.3
    ids = torch.randint(0, 100, (tokens,), device=dev, dtype=torch.int32)
    return slots, input_lengths, accept_lengths, valid, tail, previous, mask, ids


@pytest.mark.parametrize(
    ("batch", "num_extends", "padding_rows", "context"),
    (
        (7, 3, 0, 3),
        (16, 0, 5, 3),
        (4, 4, 0, 1),
        (33, 1, 2, 4),
        (1, 0, 0, 3),
        # Several programs: the row starts must carry across blocks.
        (5000, 1700, 9, 3),
    ),
)
def test_fused_advance_matches_eager_chain(batch, num_extends, padding_rows, context):
    pool, tokens = max(64, batch + 1), max(256, batch * 6)
    slots, inputs, accepts, valid, tail, previous, mask, ids = _batch(
        batch, num_extends, pool, tokens, context, padding_rows, seed=batch
    )
    expected_valid, expected_tail = valid.clone(), tail.clone()
    _eager_reference(
        slots,
        inputs,
        accepts,
        expected_valid,
        num_extends,
        pool - 1,
        expected_tail,
        previous,
        mask,
        ids,
    )

    advance_accepted_frontier(
        slots,
        inputs,
        accepts,
        valid,
        num_extends,
        pool - 1,
        ngram_tail=tail,
        ngram_previous_tokens=previous,
        ngram_token_mask=mask,
        input_ids=ids,
    )
    torch.cuda.synchronize()

    assert torch.equal(valid, expected_valid)
    assert torch.equal(tail, expected_tail)


def test_fused_advance_without_ngram_history_only_moves_cache_lengths():
    pool = 16
    slots, inputs, accepts, valid, tail, previous, mask, ids = _batch(
        5, 2, pool, 64, 3, 1, seed=11
    )
    expected_valid = valid.clone()
    _eager_reference(
        slots, inputs, accepts, expected_valid, 2, pool - 1, None, None, None, None
    )

    advance_accepted_frontier(
        slots,
        inputs,
        accepts,
        valid,
        2,
        pool - 1,
        ngram_tail=None,
        ngram_previous_tokens=None,
        ngram_token_mask=None,
        input_ids=None,
    )
    torch.cuda.synchronize()

    assert torch.equal(valid, expected_valid)
    with pytest.raises(ValueError, match="all be given"):
        advance_accepted_frontier(
            slots,
            inputs,
            accepts,
            valid,
            2,
            pool - 1,
            ngram_tail=tail,
            ngram_previous_tokens=None,
            ngram_token_mask=None,
            input_ids=None,
        )
    with pytest.raises(ValueError, match="int32"):
        advance_accepted_frontier(
            slots,
            inputs,
            accepts,
            valid.long(),
            2,
            pool - 1,
            ngram_tail=None,
            ngram_previous_tokens=None,
            ngram_token_mask=None,
            input_ids=None,
        )


def test_advance_runs_on_cpu_tensors_as_tensor_ops():
    """Devices without the kernel (NPU, CPU tests) take the tensor path."""
    pool = 16
    args = _batch(6, 2, pool, 64, 3, 1, seed=5)
    cpu = [t.cpu() for t in args]
    slots, inputs, accepts, valid, tail, previous, mask, ids = args
    advance_accepted_frontier(
        slots,
        inputs,
        accepts,
        valid,
        2,
        pool - 1,
        ngram_tail=tail,
        ngram_previous_tokens=previous,
        ngram_token_mask=mask,
        input_ids=ids,
    )
    advance_accepted_frontier(
        cpu[0],
        cpu[1],
        cpu[2],
        cpu[3],
        2,
        pool - 1,
        ngram_tail=cpu[4],
        ngram_previous_tokens=cpu[5],
        ngram_token_mask=cpu[6],
        input_ids=cpu[7],
    )
    torch.cuda.synchronize()
    assert torch.equal(valid.cpu(), cpu[3])
    assert torch.equal(tail.cpu(), cpu[4])
    with pytest.raises(ValueError, match="colocated"):
        advance_accepted_frontier(
            cpu[0],
            inputs,
            accepts,
            valid,
            2,
            pool - 1,
            ngram_tail=None,
            ngram_previous_tokens=None,
            ngram_token_mask=None,
            input_ids=None,
        )
