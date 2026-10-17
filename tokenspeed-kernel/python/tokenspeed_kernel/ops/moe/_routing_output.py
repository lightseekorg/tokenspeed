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

import torch


def routing_output_buffers(
    logits: torch.Tensor,
    topk: int,
    weights_dtype: torch.dtype,
    out: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate routing outputs or validate supplied weights and expert IDs.

    FP32 storage may hold weights rounded to BF16 by the producer kernel.
    The integer output may use INT32 or INT64. Both outputs are contiguous
    [tokens, topk] tensors on the logits device.
    """
    shape = (logits.shape[0], topk)
    if out is None:
        return (
            torch.empty(shape, dtype=weights_dtype, device=logits.device),
            torch.empty(shape, dtype=torch.int32, device=logits.device),
        )
    weights, ids = out
    for tensor in out:
        if (
            tensor.shape != shape
            or tensor.device != logits.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Routing output must be contiguous {shape} on {logits.device}"
            )
    if weights.dtype != weights_dtype and not (
        weights_dtype == torch.bfloat16 and weights.dtype == torch.float32
    ):
        raise ValueError(
            "Routing weights must use the requested dtype or BF16 values in FP32"
        )
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Routing expert IDs must use INT32 or INT64")
    return out
