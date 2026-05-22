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

"""TRT-LLM fp8 quantize helpers.

Each helper returns ``qweight.float()`` so callers can compare quantized values
without depending on TRT-LLM's scale tensor layout.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.thirdparty.trtllm import (
    per_tensor_quant_fp8 as _trtllm_per_tensor_quant_fp8,
)
from tokenspeed_kernel.thirdparty.trtllm import (
    per_token_group_quant_8bit as _trtllm_per_token_group_quant_8bit,
)
from tokenspeed_kernel.thirdparty.trtllm import (
    per_token_quant_fp8 as _trtllm_per_token_quant_fp8,
)

_FP8_DTYPE = Platform.get().fp8e4m3fn.dtype


def trtllm_fp8_token_group_128(x: torch.Tensor) -> torch.Tensor:
    qweight, _scale = _trtllm_per_token_group_quant_8bit(x, group_size=128)
    return qweight.float()


def trtllm_fp8_token(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x, dtype=_FP8_DTYPE)
    scale = torch.empty(x.size(0), dtype=torch.float32, device=x.device)
    _trtllm_per_token_quant_fp8(x, output, scale)
    return output.float()


def trtllm_fp8_tensor(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x, dtype=_FP8_DTYPE)
    scale = torch.zeros(1, dtype=torch.float32, device=x.device)
    _trtllm_per_tensor_quant_fp8(x, output, scale)
    return output.float()
