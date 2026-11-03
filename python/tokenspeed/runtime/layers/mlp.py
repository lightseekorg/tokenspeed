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

"""Gated MLP shared by dense layers and shared experts."""

from __future__ import annotations

import enum
from collections.abc import Sequence

import torch
from tokenspeed_kernel.ops.gemm._cute_dsl.nvfp4_gemm_swiglu import (
    nvfp4_gemm_swiglu_nvfp4_quant,
)
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed.mapping import Group
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.dense.mxfp4 import Mxfp4LinearMethod
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod
from tokenspeed.runtime.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import envs

_is_blackwell = current_platform().is_blackwell


class GatedMLPFusion(enum.Enum):
    """A kernel that applies SwiGLU inside the gate/up GEMM, or ``NONE``."""

    NONE = "none"
    # NVFP4 GEMM whose epilogue also quantizes for an NVFP4 down_proj.
    NVFP4_GEMM_SWIGLU_QUANT = "nvfp4_gemm_swiglu_quant"
    MXFP4_GEMM_SWIGLU = "mxfp4_gemm_swiglu"

    @property
    def is_nvfp4_gemm_swiglu_quant(self) -> bool:
        return self is GatedMLPFusion.NVFP4_GEMM_SWIGLU_QUANT

    @property
    def is_mxfp4_gemm_swiglu(self) -> bool:
        return self is GatedMLPFusion.MXFP4_GEMM_SWIGLU


def _supports(
    fusion: GatedMLPFusion, gate_up_proj: LinearBase, down_proj: LinearBase
) -> bool:
    """Whether both projections' quantization can run ``fusion``."""
    gate_up_method = gate_up_proj.quant_method
    down_method = down_proj.quant_method
    if fusion.is_nvfp4_gemm_swiglu_quant:
        return (
            envs.TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT.get()
            and _is_blackwell
            and isinstance(gate_up_method, Nvfp4LinearMethod)
            and isinstance(down_method, Nvfp4LinearMethod)
        )
    if fusion.is_mxfp4_gemm_swiglu:
        return isinstance(gate_up_method, Mxfp4LinearMethod) and isinstance(
            down_method, Mxfp4LinearMethod
        )
    raise ValueError(f"Not a fused gated MLP kernel: {fusion}")


def gated_activation(hidden_act: str, swiglu_limit: float | None) -> nn.Module:
    """The gated activation for a checkpoint's ``hidden_act`` and SwiGLU clamp."""
    if hidden_act != "silu":
        raise ValueError(f"Unsupported gated MLP activation: {hidden_act}")
    return SiluAndMul(swiglu_limit)


class GatedMLP(nn.Module):
    """``down_proj(activation(gate_up_proj(x)))`` with gate and up stacked.

    The caller builds the projections, which fixes their parallel layout and
    quantization; ``tensor_parallel`` builds the usual pair. Construction
    picks the first of ``fusions`` that both projections support, else runs
    the gate/up GEMM and then ``down_proj.forward_with_activation``, whose
    quantization method may fuse the activation into its input quantization.
    ``fusions`` lists the fused kernels validated for the model, since each
    rounds differently from the separate kernels.
    """

    def __init__(
        self,
        gate_up_proj: LinearBase,
        down_proj: LinearBase,
        activation: nn.Module,
        fusions: Sequence[GatedMLPFusion],
    ) -> None:
        super().__init__()
        if fusions and not (
            isinstance(activation, SiluAndMul) and activation.swiglu_limit is None
        ):
            raise ValueError(
                f"Fused gated MLP kernels compute SwiGLU, not {activation}"
            )
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.act_fn = activation
        self.fusion = next(
            (f for f in fusions if _supports(f, gate_up_proj, down_proj)),
            GatedMLPFusion.NONE,
        )
        # Both fused GEMMs need their weight layout chosen before loading.
        gate_up_proj.interleave_linear_and_gate = self.fusion.is_nvfp4_gemm_swiglu_quant
        if self.fusion.is_mxfp4_gemm_swiglu:
            gate_up_proj.quant_method.enable_swiglu(gate_up_proj)

    @classmethod
    def tensor_parallel(
        cls,
        hidden_size: int,
        intermediate_size: int,
        activation: nn.Module,
        tp_rank: int,
        tp_size: int,
        tp_group: Group | None,
        reduce_results: bool,
        fusions: Sequence[GatedMLPFusion],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> GatedMLP:
        """Column-parallel gate/up and row-parallel down over one TP group."""
        gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )
        return cls(gate_up_proj, down_proj, activation, fusions)

    def nvfp4_global_scale(self) -> torch.Tensor | None:
        """The global scale gate_up_proj quantizes its input to NVFP4 with, if it does."""
        return self.gate_up_proj.quant_method.nvfp4_global_scale(self.gate_up_proj)

    def forward(
        self,
        x: torch.Tensor,
        x_nvfp4: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the MLP; ``x_nvfp4`` is gate_up_proj's input when its producer quantized ``x``."""
        if x.shape[0] == 0:
            return x
        if self.fusion.is_nvfp4_gemm_swiglu_quant:
            if x_nvfp4 is None:
                x_nvfp4 = fp4_quantize(x, self.gate_up_proj.input_scale_inv)
            activated = nvfp4_gemm_swiglu_nvfp4_quant(
                *x_nvfp4,
                self.gate_up_proj.weight_swiglu_interleaved,
                self.gate_up_proj.weight_scale_swiglu_interleaved,
                self.gate_up_proj.alpha,
                self.down_proj.input_scale_inv,
            )
            output, _ = self.down_proj(activated)
        elif self.fusion.is_mxfp4_gemm_swiglu:
            activated = self.gate_up_proj.quant_method.apply_swiglu(
                self.gate_up_proj, x
            )
            output, _ = self.down_proj(activated)
        else:
            gate_up, _ = self.gate_up_proj(x if x_nvfp4 is None else x_nvfp4)
            output, _ = self.down_proj.forward_with_activation(gate_up, self.act_fn)
        return output
