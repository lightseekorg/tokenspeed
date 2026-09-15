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


"""Numerical helpers for fused shared-RS/up-projection/AG kernel tests."""

import math

import torch


def equal_outputs(actual, expected):
    return all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(actual, expected)
    )


def reference_errors(actual, reference):
    difference = actual.float() - reference.float()
    absolute = difference.abs().max().item()
    relative_l2 = (difference.norm() / reference.float().norm().clamp_min(1e-12)).item()
    maximum = reference.abs().max().item()
    # Preserve the original BF16 reference bounds, without benchmark machinery.
    passed = (
        all(math.isfinite(x) for x in (absolute, relative_l2, maximum))
        and absolute >= 0
        and relative_l2 >= 0
        and maximum >= 0
        and absolute <= 0.046875 + 0.01 * maximum
        and relative_l2 <= 0.006
    )
    return {
        "max_absolute": absolute,
        "relative_l2": relative_l2,
        "reference_max": maximum,
        "passed": passed,
    }
