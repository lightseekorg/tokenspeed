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

"""Other operators remain importable without the optional TRT-LLM extension."""

import importlib.util

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import deep_gemm  # noqa: F401
from tokenspeed_kernel.ops.gemm.trtllm import dsv3_fused_a_gemm
from tokenspeed_kernel.ops.quantization.trtllm import _per_token_group_quant_8bit

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("trtllm_kernel") is not None,
    reason="requires an environment without the optional TRT-LLM GPU extension",
)


def test_fused_a_projection_reports_missing_optional_extension():
    with pytest.raises(RuntimeError, match="compatible trtllm_kernel package"):
        dsv3_fused_a_gemm(torch.ones(1, 128), torch.ones(128, 128))


def test_trtllm_quantization_reports_missing_optional_extension():
    with pytest.raises(RuntimeError, match="compatible trtllm_kernel package"):
        _per_token_group_quant_8bit(torch.ones(1, 128), 128, False)
