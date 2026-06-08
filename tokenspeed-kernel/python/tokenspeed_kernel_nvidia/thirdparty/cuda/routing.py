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

"""Routing ops: routing_flash (softmax + topk with correction bias)."""

import functools
from pathlib import Path

import torch


@functools.cache
def _load_routing_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "routing"
    so_path = objs_dir / "routing.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel routing library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def routing_flash(
    input: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts_real: int,
    scaling_factor: float,
    renorm: bool = False,
) -> None:
    _load_routing_module().softmax_topk_flash(
        input,
        correction_bias,
        topk_indices,
        topk_weights,
        int(num_experts_real),
        float(scaling_factor),
        bool(renorm),
    )


def softplus_sqrt_topk_flash(
    input: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    scaling_factor: float,
    renorm: bool = False,
) -> None:
    _load_routing_module().softplus_sqrt_topk_flash(
        input,
        correction_bias,
        topk_indices,
        topk_weights,
        bool(renorm),
        float(scaling_factor),
    )


def hash_softplus_sqrt_topk_flash(
    input: torch.Tensor,
    input_ids: torch.Tensor,
    hash_indices_table: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    scaling_factor: float,
    renorm: bool = False,
) -> None:
    _load_routing_module().hash_softplus_sqrt_topk_flash(
        input,
        input_ids,
        hash_indices_table,
        topk_indices,
        topk_weights,
        bool(renorm),
        float(scaling_factor),
    )
