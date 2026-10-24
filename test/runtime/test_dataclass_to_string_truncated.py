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

"""Request-log formatting by ``dataclass_to_string_truncated``."""

import dataclasses
import os
import sys

import numpy as np
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.utils.common import dataclass_to_string_truncated  # noqa: E402


@dataclasses.dataclass
class _Request:
    rid: str
    features: list


def test_arrays_in_a_list_are_summarized_by_shape():
    request = _Request(
        rid="r0", features=[np.zeros((2, 3), dtype=np.float32), torch.zeros(4)]
    )
    assert dataclass_to_string_truncated(request) == (
        "_Request(rid='r0', features=[ndarray(shape=(2, 3), dtype=float32), "
        "Tensor(shape=(4,), dtype=torch.float32)])"
    )


def test_long_list_is_elided_in_the_middle():
    assert dataclass_to_string_truncated(list(range(10)), max_length=4) == (
        "[0, 1, ..., 8, 9]"
    )


def test_long_list_of_arrays_is_elided_in_the_middle():
    arrays = [np.zeros(i + 1) for i in range(5)]
    assert dataclass_to_string_truncated(arrays, max_length=2) == (
        "[ndarray(shape=(1,), dtype=float64), ..., "
        "ndarray(shape=(5,), dtype=float64)]"
    )


def test_ellipsis_string_element_is_shown():
    assert dataclass_to_string_truncated(["a", "..."]) == "['a', '...']"
