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

"""CPU regressions for empty-prefix partial attention state semantics."""

import pytest
import torch
from tokenspeed_kernel_npu.ops._mla_varlen import set_empty_attention_rows


def endpoints(lengths):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )


@pytest.mark.parametrize(
    "queries,keys",
    [
        ([128, 384], [1024, 0]),
        ([3, 2, 4], [0, 7, 0]),
        ([0, 3, 0, 2], [0, 0, 7, 4]),
        ([2, 1], [0, 0]),
        ([2, 3], [4, 7]),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_only_empty_kv_requests_become_zero_mass_states(queries, keys, return_lse):
    rows = sum(queries)
    output = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 2, 3)
    lse = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    original, original_lse = output.clone(), lse.clone()
    offset = 0
    empty = torch.zeros(rows, dtype=torch.bool)
    for qlen, klen in zip(queries, keys, strict=True):
        if klen == 0:
            empty[offset : offset + qlen] = True
        offset += qlen
    output[empty] = torch.nan
    lse[empty] = torch.inf
    set_empty_attention_rows(
        output, lse if return_lse else None, endpoints(queries), endpoints(keys)
    )
    torch.testing.assert_close(output[~empty], original[~empty], rtol=0, atol=0)
    assert torch.equal(output[empty], torch.zeros_like(output[empty]))
    if return_lse:
        torch.testing.assert_close(lse[~empty], original_lse[~empty], rtol=0, atol=0)
        assert torch.isneginf(lse[empty]).all()
    else:
        assert torch.isposinf(lse[empty]).all()


def merge_fp64(a, la, b, lb):
    mass = torch.logaddexp(la.double(), lb.double())
    return (
        a.double() * torch.exp(la.double() - mass)[..., None]
        + b.double() * torch.exp(lb.double() - mass)[..., None]
    ), mass


def test_repeated_empty_prefix_merges_preserve_current_attention():
    qends, kends = endpoints([128, 384]), endpoints([1024, 0])
    generator = torch.Generator().manual_seed(7341)
    current = torch.randn(512, 4, 8, generator=generator).bfloat16()
    current_lse = torch.randn(512, 4, generator=generator)
    expected = current[128:].double().clone()
    actual, actual_lse = current.clone(), current_lse.clone()
    for _ in range(3):
        prefix = torch.randn(512, 4, 8, generator=generator).bfloat16()
        prefix[128:] = 0
        prefix_lse = torch.randn(512, 4, generator=generator)
        prefix_lse[128:] = torch.inf
        set_empty_attention_rows(prefix, prefix_lse, qends, kends)
        actual, actual_lse = merge_fp64(actual, actual_lse, prefix, prefix_lse)
    torch.testing.assert_close(actual[128:], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_lse[128:], current_lse[128:].double(), rtol=0, atol=0
    )
    assert torch.isfinite(actual).all()


def test_nonempty_nonfinite_values_are_not_hidden():
    output = torch.tensor([[[float("nan")]], [[float("nan")]]])
    lse = torch.tensor([[float("inf")], [float("inf")]])
    set_empty_attention_rows(output, lse, endpoints([1, 1]), endpoints([1, 0]))
    assert torch.isnan(output[0]).all() and torch.isposinf(lse[0]).all()
    assert output[1].eq(0).all() and torch.isneginf(lse[1]).all()
