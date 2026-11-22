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

"""Bounded byte diagnostics use real CPU tensors and preserve strict equality."""

import pytest
import torch
from attention_prologue_helpers import assert_bytes_equal


class FailureOnlyName(str):
    def __format__(self, format_spec):
        raise AssertionError("A passing comparison must not format diagnostics")


def test_equal_strided_tensors_do_not_construct_failure_diagnostics():
    a = torch.arange(24, dtype=torch.bfloat16, device="cpu").reshape(4, 6)[:, ::2]
    assert_bytes_equal(a, a.clone(), name=FailureOnlyName("Q"))


def test_byte_equality_is_not_redefined_by_metadata_checks():
    # The existing predicate compares uint8 views, not the original dtype/shape.
    assert_bytes_equal(
        torch.zeros(2, dtype=torch.int16, device="cpu"),
        torch.zeros(4, dtype=torch.uint8, device="cpu"),
        name=FailureOnlyName("returnedV"),
    )


def test_byte_differences_report_logical_offsets_and_limit_the_values():
    a = torch.zeros((2, 20), dtype=torch.uint8, device="cpu")[:, ::2]
    b = torch.arange(1, 21, dtype=torch.uint8, device="cpu").reshape(2, 10)
    with pytest.raises(AssertionError) as error:
        assert_bytes_equal(a, b, name="K-cache")
    assert str(error.value) == (
        "K-cache: a(dtype=torch.uint8, shape=(2, 10)), "
        "b(dtype=torch.uint8, shape=(2, 10)); mismatch_byte_count=20; "
        f"byte_offsets={list(range(16))}; "
        f"a_uint8={[0] * 16}; b_uint8={list(range(1, 17))}"
    )


def test_signed_zero_difference_is_still_a_byte_failure():
    a = torch.zeros(2, dtype=torch.bfloat16, device="cpu")
    b = a.clone()
    b.view(torch.uint8)[1] = 128
    assert torch.equal(a, b)
    with pytest.raises(AssertionError) as error:
        assert_bytes_equal(a, b, name="returnedK")
    assert str(error.value) == (
        "returnedK: a(dtype=torch.bfloat16, shape=(2,)), "
        "b(dtype=torch.bfloat16, shape=(2,)); mismatch_byte_count=1; "
        "byte_offsets=[1]; a_uint8=[0]; b_uint8=[128]"
    )


@pytest.mark.parametrize(
    "shape,dtype,metadata",
    [
        ((3, 2), torch.int16, "torch.int16, shape=(3, 2)"),
        ((2, 3), torch.uint8, "torch.uint8, shape=(2, 3)"),
    ],
)
def test_incompatible_tensors_report_both_metadata_without_broadcasting(
    shape, dtype, metadata
):
    b = torch.zeros(shape, dtype=dtype, device="cpu")
    with pytest.raises(AssertionError) as error:
        assert_bytes_equal(
            torch.zeros((2, 3), dtype=torch.int16, device="cpu"), b, name="V-cache"
        )
    assert str(error.value) == (
        "V-cache: a(dtype=torch.int16, shape=(2, 3)), "
        f"b(dtype={metadata}); shape/dtype mismatch; mismatch_byte_count=not comparable"
    )
