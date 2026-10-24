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

"""Reuse the verified V4.1 group32 decoder for Engram embedding rows."""

import torch
import torch.nn.functional as F
from tokenspeed_kernel_npu.ops.dsv41_pointwise import decode_rows


def fp8_e8m0_rows(codes, scales):
    if codes.ndim < 1 or codes.device.type != "npu" or scales.device != codes.device:
        raise ValueError("FP8/E8M0 rows require matching NPU tensors")
    if codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("FP8/E8M0 row decoding requires byte storage")
    if codes.shape[-1] % 32 or scales.shape != (
        *codes.shape[:-1],
        codes.shape[-1] // 32,
    ):
        raise ValueError("One E8M0 scale is required per 32 FP8 values")
    flat = codes.contiguous().reshape(-1)
    scale = scales.contiguous().reshape(-1)
    pad = (-flat.numel()) % 512
    if pad:
        flat = F.pad(flat, (0, pad))
        scale = F.pad(scale, (0, pad // 32), value=127)
    out = torch.empty(flat.numel(), dtype=torch.bfloat16, device=codes.device)
    decode_rows(flat.reshape(-1, 512), scale.reshape(-1, 16), None, out, fp4=False)
    return out[: codes.numel()].reshape(codes.shape)
