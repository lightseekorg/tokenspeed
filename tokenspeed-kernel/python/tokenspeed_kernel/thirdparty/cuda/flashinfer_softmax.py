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

"""Vendored flashinfer softmax with bf16/fp16/fp32 input, fp32 output."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional, Union

import torch

_WORKSPACE_BYTES = 1 * 1024 * 1024


@functools.cache
def _load_module():
    import tvm_ffi

    so_path = (
        Path(__file__).parent / "objs" / "flashinfer_softmax" / "flashinfer_softmax.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel flashinfer_softmax library not found at {so_path}. "
            "Run: pip install -e tokenspeed-kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


@functools.cache
def _get_workspace(device: torch.device) -> torch.Tensor:
    return torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)


def softmax(
    logits: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """softmax(logits / temperature). Returns fp32 probs."""
    assert logits.is_contiguous(), "softmax expects contiguous logits"
    assert logits.is_cuda, "softmax requires CUDA tensors"
    assert logits.dim() == 2
    assert logits.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"softmax: unsupported logits dtype {logits.dtype}"

    if isinstance(temperature, torch.Tensor):
        assert temperature.is_contiguous() and temperature.dtype == torch.float32
        temp_arr: Optional[torch.Tensor] = temperature.view(-1)
        temp_val = 0.0
    else:
        temp_arr = None
        temp_val = 1.0 if temperature is None else float(temperature)

    output = torch.empty_like(logits, dtype=torch.float32)
    workspace = _get_workspace(logits.device)
    _load_module().softmax(
        workspace,
        logits,
        output,
        temp_arr,
        float(temp_val),
        bool(enable_pdl),
    )
    return output
