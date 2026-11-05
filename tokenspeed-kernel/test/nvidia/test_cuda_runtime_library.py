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

"""CUDA runtime discovery excludes compiler stub libraries."""

import io
from unittest.mock import patch

import pytest
from tokenspeed_kernel.thirdparty.cuda.cuda_ipc import find_loaded_library


@pytest.mark.parametrize(
    "paths, expected",
    [
        (
            ["/opt/compiler/libcudart_stub.so", "/opt/cuda/libcudart.so.12"],
            "/opt/cuda/libcudart.so.12",
        ),
        (
            ["/opt/cuda/stubs/libcudart.so", "/opt/cuda/libcudart.so"],
            "/opt/cuda/libcudart.so",
        ),
        (["/opt/compiler/libcudart_stub.so"], None),
        (["/opt/cuda/libcudart.so.12.8"], "/opt/cuda/libcudart.so.12.8"),
    ],
)
def test_runtime_soname(paths, expected):
    maps = "".join(f"1000-2000 r-xp 00000000 00:00 1 {path}\n" for path in paths)
    # /proc belongs to the OS; the real discovery function parses its contents.
    with patch("builtins.open", return_value=io.StringIO(maps)):
        assert find_loaded_library("libcudart") == expected
