# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

"""Fused operators for activation layers."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from tokenspeed_kernel import prepare_fp8_linear_activation, silu_and_mul
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.utils import (
    get_colorful_logger,
)

_is_amd = current_platform().is_amd

logger = get_colorful_logger(__name__)


class SiluAndMul(torch.nn.Module):
    def __init__(self, swiglu_limit: float | None = None) -> None:
        super().__init__()
        self.swiglu_limit = (
            float(swiglu_limit)
            if swiglu_limit is not None and swiglu_limit > 0
            else None
        )

    def forward(self, x: torch.Tensor, fp8_out: bool = False) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"SwiGLU expects an even [gate, up] width, got {x.shape[-1]}"
            )
        if not x.is_cuda:
            if fp8_out:
                raise NotImplementedError("CPU fp8_out silu_and_mul is not implemented")
            return self.forward_native(x)

        if not _is_amd:

            def get_tma_aligned_scale(x):
                aligned_size = (x.shape[-2] + 3) // 4 * 4
                x_s = torch.empty(
                    x.shape[:-2] + (x.shape[-1] // 128, aligned_size),
                    device=x.device,
                    dtype=torch.float32,
                ).permute(-1, -2)[: x.shape[-2], :]
                return x_s

            d = x.shape[-1] // 2
            output_shape = x.shape[:-1] + (d,)
            if fp8_out:
                if self.swiglu_limit is not None:
                    raise NotImplementedError(
                        "clamped fp8_out silu_and_mul is not implemented"
                    )
                out = torch.empty(
                    output_shape, dtype=torch.float8_e4m3fn, device=x.device
                )
                scale = get_tma_aligned_scale(out)
                from tokenspeed_kernel.ops.activation.cuda import (
                    silu_and_mul_fuse_block_quant,
                )

                out, scale = silu_and_mul_fuse_block_quant(x, scale, out)
                return out, scale
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            return silu_and_mul(
                x,
                out,
                limit=self.swiglu_limit,
            )

        if fp8_out:
            raise NotImplementedError("AMD fp8_out silu_and_mul is not implemented")
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        return silu_and_mul(
            x,
            out,
            limit=self.swiglu_limit,
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        if self.swiglu_limit is not None:
            gate = gate.clamp_max(self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        return (torch.nn.functional.silu(gate) * up).to(x.dtype)

    def prepare_for_fp8_linear(
        self, x: torch.Tensor, plan: object
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fuse SwiGLU and quantization when the prepared linear supports it."""
        return prepare_fp8_linear_activation(
            plan,
            x,
            activation="swiglu",
            limit=self.swiglu_limit,
        )


class SituAndMul(torch.nn.Module):
    """SiTU / SituGLU gated activation used by Kimi models (e.g. Kimi-K3).

    Over the fused ``[..., gate | up]`` input, computes::

        gate = beta * tanh(gate / beta) * sigmoid(gate)
        up   = linear_beta * tanh(up / linear_beta)   # only if linear_beta set
        out  = gate * up

    Args:
        beta: Softplus-like temperature applied to the gate branch.
        linear_beta: If set, softly clips the up branch; disabled when ``None``.

    CUDA inputs run the fused Triton kernel (one launch instead of the ~10
    sliced elementwise kernels of the native path, ULP-identical fp32 math);
    other devices keep the native reference implementation.
    """

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"SiTU beta must be positive, got {beta}")
        if linear_beta is not None and linear_beta <= 0.0:
            raise ValueError(f"SiTU linear_beta must be positive, got {linear_beta}")
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"SiTU expects an even [gate, up] width, got {x.shape[-1]}"
            )
        if x.is_cuda:
            from tokenspeed_kernel import situ_and_mul

            return situ_and_mul(
                x,
                beta=self.beta,
                linear_beta=self.linear_beta,
            )
        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(x.dtype)


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def compute_swiglu(gelu, linear, scale, alpha, limit):
    gelu = gelu.to(tl.float32) * scale
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32) * scale
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)

    s = gelu / (1 + tl.exp(-alpha * gelu))

    return tl.fma(s, linear, s)  # (s * (linear + 1))


@triton.jit(repr=lambda _: "_swiglu")
def swiglu_fn(input, alpha, limit, exclusive_sum, local_num_experts):
    begin = exclusive_sum[0]
    end = exclusive_sum[local_num_experts]
    input = input[begin:end]

    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return compute_swiglu(gelu, linear, 1.0, alpha, limit)


@dataclass
class SwigluArg:
    alpha: float
    limit: float
