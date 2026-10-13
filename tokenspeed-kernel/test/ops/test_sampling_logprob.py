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

"""``sampling.vocab_parallel_logprobs`` and its portable block-sumexp leaf.

The tree is what matters: one row's result must depend only on the vocab
size and that row's own values, never on the batch it shares. Equality with
``torch.log_softmax`` holds up to the two trees' rounding and is documented
here with a tolerance, not asserted bitwise.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling import vocab_parallel_logprobs
from tokenspeed_kernel.ops.sampling.torch import torch_block_sumexp
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

BLOCK = 32768
VOCAB = 4 * BLOCK


def _logits(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, VOCAB, generator=gen) * 4


def test_leaf_is_registered_as_the_batch_invariant_torch_solution():
    spec = KernelRegistry.get().get_by_name("torch_block_sumexp")
    assert spec is not None
    assert spec.family == "sampling" and spec.mode == "block_sumexp"
    assert spec.solution == "torch"
    assert "batch_invariant" in spec.features
    kernel = select_kernel(
        "sampling",
        "block_sumexp",
        format_signature(shifted=dense_tensor_format(torch.float32)),
        features=frozenset({"batch_invariant"}),
    )
    assert kernel.name == "torch_block_sumexp"


def test_block_sumexp_is_a_fixed_pairwise_tree():
    shifted = _logits(3, 1) - 20.0
    sums = torch_block_sumexp(shifted, block_size=BLOCK)
    assert sums.shape == (3, VOCAB // BLOCK)
    # Hand-rolled halving tree over one block of one row.
    values = torch.exp(shifted[1, BLOCK : 2 * BLOCK])
    while values.numel() > 1:
        half = values.numel() // 2
        values = values[:half] + values[half:]
    assert torch.equal(sums[1, 1], values[0])
    # And it is a sum, up to fp32 reassociation.
    torch.testing.assert_close(
        sums, torch.exp(shifted).view(3, -1, BLOCK).sum(-1), rtol=1e-5, atol=0
    )
    # A non-contiguous input (a column slice of a wider matrix) folds to the
    # same bits, and the input itself is never written.
    wide = torch.cat((shifted, torch.full((3, 7), 5.0)), dim=1)
    strided = wide[:, :VOCAB]
    assert not strided.is_contiguous()
    before = strided.clone()
    assert torch.equal(torch_block_sumexp(strided, block_size=BLOCK), sums)
    assert torch.equal(strided, before)
    with pytest.raises(ValueError, match="power of two"):
        torch_block_sumexp(shifted, block_size=3000)
    with pytest.raises(ValueError, match="multiple"):
        torch_block_sumexp(shifted[:, : VOCAB - 1], block_size=BLOCK)


def test_logprobs_do_not_depend_on_the_batch():
    logits = _logits(6, 2)
    targets = torch.randint(0, VOCAB, (6,), generator=torch.Generator().manual_seed(3))
    full = vocab_parallel_logprobs(logits, targets, vocab_block=BLOCK)
    solo = vocab_parallel_logprobs(logits[2:3], targets[2:3], vocab_block=BLOCK)
    assert torch.equal(full[2], solo[0])
    permutation = torch.tensor([5, 0, 3, 1, 4, 2])
    shuffled = vocab_parallel_logprobs(
        logits[permutation], targets[permutation], vocab_block=BLOCK
    )
    assert torch.equal(shuffled, full[permutation])
    # Run invariance, and the dtype contract.
    assert torch.equal(
        full, vocab_parallel_logprobs(logits, targets, vocab_block=BLOCK)
    )
    assert full.dtype == torch.float32
    assert full.shape == (6,)


def test_logprobs_match_log_softmax_up_to_tree_rounding():
    logits = _logits(4, 5)
    targets = torch.randint(0, VOCAB, (4,), generator=torch.Generator().manual_seed(6))
    got = vocab_parallel_logprobs(logits, targets, vocab_block=BLOCK)
    ref = torch.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    # Different association orders in the denominator: equal to ~1e-6
    # relative, not bitwise — the leaf's tree is the contract, not torch's.
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    # bf16 logits are promoted exactly as torch would promote them.
    bf16 = vocab_parallel_logprobs(
        logits.to(torch.bfloat16), targets, vocab_block=BLOCK
    )
    torch.testing.assert_close(
        bf16,
        torch.log_softmax(logits.to(torch.bfloat16).float(), dim=-1)
        .gather(1, targets.unsqueeze(1))
        .squeeze(1),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.equal(
        vocab_parallel_logprobs(logits, targets.to(torch.int32), vocab_block=BLOCK), got
    )


def test_logprobs_refuse_misshaped_inputs():
    logits = _logits(2, 7)
    targets = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="multiple of vocab_block"):
        vocab_parallel_logprobs(logits[:, :-1], targets, vocab_block=BLOCK)
    with pytest.raises(ValueError, match="target_ids must have shape"):
        vocab_parallel_logprobs(logits, targets[:1], vocab_block=BLOCK)
    with pytest.raises(ValueError, match="int32 or int64"):
        vocab_parallel_logprobs(logits, targets.float(), vocab_block=BLOCK)
    with pytest.raises(ValueError, match=r"logits must be \[rows, vocab\]"):
        vocab_parallel_logprobs(logits[0], targets[:1], vocab_block=BLOCK)
