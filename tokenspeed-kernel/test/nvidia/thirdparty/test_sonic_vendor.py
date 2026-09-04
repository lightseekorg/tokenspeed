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

"""The vendored sonic-sampler kernels: compiled by tokenspeed-triton, no upstream package or msgspec host layer, and the config table that drives them."""

from __future__ import annotations

import pathlib
import sys

import pytest
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.sampling.sonic import (
    TABLE,
    configs,
    pick,
    scratchpad_columns,
)
from tokenspeed_kernel.thirdparty.sonic.ops import multistep, prologue, singular, verify


def test_kernels_are_tokenspeed_triton_jit_functions():
    kernels = (
        prologue.bitpacked_reduction_kernel,
        singular.cumulative_selection_kernel,
        multistep.cumulative_unpack_kernel,
        verify.chain_speculative_verification_kernel,
    )
    for kernel in kernels:
        assert isinstance(kernel, triton.JITFunction)
    assert not any(name.split(".")[0] == "sonic_sampler" for name in sys.modules)


def test_the_vendored_kernels_need_no_msgspec():
    root = pathlib.Path(prologue.__file__).parents[1]
    assert not any("msgspec" in path.read_text() for path in root.rglob("*.py"))


@pytest.mark.parametrize(
    "vocab_size", [512, 1000, 1024, 3000, 32000, 65536, 128256, 151936, 201088, 262144]
)
def test_every_pick_tiles_the_vocabulary_in_two_blocks_or_more(vocab_size):
    for rows in [*range(1, 70), 127, 128, 129, 512, 4096]:
        config = pick(vocab_size, rows)
        assert 2 * config.block_n <= vocab_size
        assert config in configs(vocab_size)


def test_the_table_closes_every_class_and_row_range():
    assert TABLE[-1][0] is None
    sizes = [size for size, _ in TABLE[:-1]]
    assert sizes == sorted(sizes)
    for _, rows_table in TABLE:
        assert rows_table[-1][0] is None
        bounds = [bound for bound, _ in rows_table[:-1]]
        assert bounds == sorted(bounds)


def test_the_scratchpad_holds_the_narrowest_picked_tiles_blocks():
    narrowest = min(c.block_n for c in configs(151936))
    assert scratchpad_columns(151936) == -(-151936 // narrowest) * 128


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
