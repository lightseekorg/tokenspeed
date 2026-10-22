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

"""Required Ascend C libraries for the V4.1 serving profiles."""

import ctypes
from functools import cache
from pathlib import Path


@cache
def load_library(name):
    path = Path(__file__).resolve().parents[1] / "lib" / f"lib{name}.so"
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path.name}; run bash tokenspeed-kernel-npu/csrc/{name}/build.sh"
        )
    return ctypes.CDLL(str(path))


@cache
def require_v41_libraries(*, w4a4):
    """Check the profile's native dependencies before weight preparation."""
    for name in (
        "dsv41_cache_gather",
        "dsv41_candidate_gather",
        "dsv41_index_metadata",
        "dsv41_index_scan",
        "dsv41_paged_metadata",
        "dsv41_route_weights",
        "dsv41_window_metadata",
        "dsv41_ngram",
        "assert_indices",
    ):
        load_library(name)
    if w4a4:
        load_library("dsv41_w4a4")
