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

import inspect

import pytest

cute = pytest.importorskip("cutlass.cute")
gdn_prefill = pytest.importorskip("flashinfer.gdn_prefill")

from tokenspeed_kernel.thirdparty.msa.cute.src.common import utils as msa_utils


def test_cutlass_4_7_dependency_contract() -> None:
    assert "use_cp" in inspect.signature(
        gdn_prefill.chunk_gated_delta_rule
    ).parameters
    assert callable(cute.compile)
    assert not hasattr(cute.compile, "__getitem__")
    assert hasattr(cute.arch, "sub_packed_f32x2")
    assert callable(msa_utils.ex2_emulation_2)
