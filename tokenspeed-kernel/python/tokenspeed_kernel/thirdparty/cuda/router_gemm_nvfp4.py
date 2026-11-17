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

"""CUDA router GEMM and linear-layout NVFP4 input producer."""

from functools import cache
from pathlib import Path

import torch


@cache
def _module(hidden_size: int, num_experts: int):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("router GEMM must warm up before CUDA graph capture")
    from flashinfer.jit.core import gen_jit_spec

    source = Path(__file__).parent / "csrc" / "router_gemm_nvfp4.cu"
    return gen_jit_spec(
        f"tokenspeed_router_nvfp4_{hidden_size}_{num_experts}",
        [source],
        extra_cuda_cflags=[
            "-gencode=arch=compute_107a,code=sm_107a",
            f"-DROUTER_HIDDEN={hidden_size}",
            f"-DROUTER_EXPERTS={num_experts}",
        ],
        extra_include_paths=[source.parent],
        use_fast_math=False,
    ).build_and_load()


def router_gemm_nvfp4(x, weight, scale):
    if not x.is_cuda or torch.cuda.get_device_capability(x.device) != (10, 7):
        raise ValueError("router GEMM NVFP4 requires SM107")
    rows, hidden = x.shape
    experts = weight.shape[0]
    if not (1 <= rows <= 8 and hidden % 64 == 0 and 256 <= hidden <= 4096):
        raise ValueError(
            "router GEMM requires 1–8 rows and hidden size 256–4096 aligned to 64"
        )
    if experts < 64 or experts % 8:
        raise ValueError("router GEMM requires at least 64 experts, aligned to 8")
    logits = torch.empty((rows, experts), device=x.device, dtype=x.dtype)
    packed = torch.empty((rows, hidden // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty(
        (rows, hidden // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    _module(hidden, experts).run(
        x, weight, scale, logits, packed, scales.view(torch.uint8)
    )
    return logits, (packed, scales)
