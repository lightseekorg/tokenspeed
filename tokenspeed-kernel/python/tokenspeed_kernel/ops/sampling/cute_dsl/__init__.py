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

from __future__ import annotations

import torch
from tokenspeed_kernel.registrations._vendor import export_vendor_symbols, false_fn

__all__ = ["argmax", "argmax_pair", "is_available"]

_SUPPORTED_OUT_DTYPES = (torch.int32, torch.int64)

# TODO: Replace these torch fallbacks with a default Triton argmax implementation,
# then delete the temporary torch fallback code.


def _validate_argmax_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _validate_argmax_pair_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    shape = (logits.shape[0], 2)
    if out.shape != shape:
        raise ValueError(f"out must have shape (M, 2)={shape}, got {tuple(out.shape)}")
    if out.dtype != torch.float32 or out.device != logits.device:
        raise ValueError("out must be float32 on the same device as logits")


def _argmax_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _argmax_pair_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")
    if out is None:
        out = torch.empty(
            (logits.shape[0], 2), dtype=torch.float32, device=logits.device
        )
    else:
        _validate_argmax_pair_out(logits, out)
    max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
    out[:, 0:1].copy_(max_vals.to(torch.float32))
    out[:, 1:2].copy_(max_indices.to(torch.float32))
    return out


globals().update(
    export_vendor_symbols(
        "nvidia",
        "tokenspeed_kernel_nvidia.sampling.cute_dsl",
        __all__,
        fallback_by_name={
            "argmax": _argmax_torch_fallback,
            "argmax_pair": _argmax_pair_torch_fallback,
            "is_available": false_fn,
        },
    )
)
