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

"""Ascend grouped MoE building blocks with explicit BF16 and INT4 paths.

Inputs are already expanded and sorted by expert. Dispatch, combine, HCCL and
LCM ownership remain runtime responsibilities. W4A4 consumes converted weights
and explicit FlatQuant parameters; it never invents calibration parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
import torch_npu

if TYPE_CHECKING:
    from tokenspeed_kernel.selection import KernelPlan


@dataclass(frozen=True)
class FlatQuantParameters:
    """Runtime-oriented matrices (left already transposed), with explicit clipping."""

    left: torch.Tensor
    right: torch.Tensor
    clip_ratio: float

    def __post_init__(self):
        if self.left.ndim != 2 or self.left.shape[0] != self.left.shape[1]:
            raise ValueError("FlatQuant left matrix must be square")
        if self.right.ndim != 2 or self.right.shape[0] != self.right.shape[1]:
            raise ValueError("FlatQuant right matrix must be square")
        if not 0 < self.clip_ratio <= 1:
            raise ValueError("FlatQuant clip_ratio must be in (0, 1]")


@dataclass(frozen=True)
class AscendGroupedLinear:
    """Prepared BF16 or packed INT4 GMM; mode and transform are always explicit.

    BF16 weight is [experts, K, N]. INT4 retains FluentLLM's NZ-packed [E,N,K/8]
    storage. The transpose to [E,K/8,N] stays in forward so TorchAir records the
    GMM transpose instead of treating an NZ view as a standalone constant.
    Its scale is uint64 [E,1,N], with float32 scale bits in each word's lower half.
    """

    mode: Literal["bf16", "w4a4"]
    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    transform: FlatQuantParameters | None
    input_size: int
    output_size: int
    w4a4_plan: KernelPlan | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.mode not in ("bf16", "w4a4"):
            raise ValueError(f"Unsupported mode {self.mode}")
        if self.mode == "w4a4":
            if (
                self.weight.dtype != torch.int32
                or self.weight_scale is None
                or self.transform is None
            ):
                raise ValueError(
                    "W4A4 requires packed INT4 weights, scales and FlatQuant parameters"
                )
            if (
                self.transform.left.shape[0] * self.transform.right.shape[0]
                != self.input_size
            ):
                raise ValueError("FlatQuant matrix dimensions do not match input size")
            from tokenspeed_kernel.ops.gemm.ascend_w4a4 import plan_grouped_w4a4

            object.__setattr__(
                self,
                "w4a4_plan",
                plan_grouped_w4a4(self.input_size, self.output_size, solution=None),
            )
        elif self.weight_scale is not None or self.transform is not None:
            raise ValueError("BF16 cannot silently discard quantization parameters")

    def __call__(
        self, sorted_x: torch.Tensor, expert_counts: torch.Tensor
    ) -> torch.Tensor:
        """Run native GMM for count-format expert groups, including empty groups."""
        if sorted_x.ndim != 2 or sorted_x.shape[1] != self.input_size:
            raise ValueError("Incorrect grouped input shape")
        if sorted_x.shape[0] == 0:
            return sorted_x.new_empty((0, self.output_size))
        if self.mode == "bf16":
            return torch_npu.npu_grouped_matmul(
                [sorted_x],
                [self.weight],
                group_list=expert_counts,
                split_item=3,
                group_type=0,
                group_list_type=1,
                output_dtype=torch.bfloat16,
            )[0]
        # Same FlatQuant and GMM conventions as FluentLLM's
        # NpuCompressedTensorsW4A4Int4MoEMethod.quantize_moe_input/apply.
        import flash_ops  # noqa: F401 -- registers custom.flat_quant

        quantized, token_scale = torch.ops.custom.flat_quant(
            sorted_x.reshape(
                sorted_x.shape[0],
                self.transform.left.shape[0],
                self.transform.right.shape[0],
            ),
            self.transform.left,
            self.transform.right,
            group_list=expert_counts,
            clip_ratio=self.transform.clip_ratio,
            group_list_type=1,
        )
        return self.w4a4_plan.kernel(
            quantized.flatten(1),
            self.weight,
            self.weight_scale,
            token_scale,
            expert_counts,
        )


@dataclass(frozen=True)
class AscendV41GroupedMoE:
    """Local routed experts with V4.1 clamp and pre-down-projection route weights.

    This returns expanded expert rows. The caller combines rows and adds the
    shared expert. Reusing LongCat's unmodified SwiGLU/combine would change V4.1
    model semantics, so only the quantization and GEMM path is shared.
    """

    gate_up: AscendGroupedLinear
    down: AscendGroupedLinear
    swiglu_limit: float

    def __post_init__(self):
        if self.gate_up.mode != self.down.mode:
            raise ValueError(
                "Both projections must use the explicitly selected precision"
            )
        if self.gate_up.output_size != 2 * self.down.input_size:
            raise ValueError("Gate/up output must match the down-projection input")

    def __call__(
        self,
        sorted_x: torch.Tensor,
        expert_counts: torch.Tensor,
        sorted_route_weights: torch.Tensor,
    ) -> torch.Tensor:
        if sorted_route_weights.shape != (sorted_x.shape[0],):
            raise ValueError("Each expanded row requires exactly one route weight")
        from tokenspeed_kernel_npu.ops.dsv41_weighted_swiglu import (
            weighted_swiglu_active,
        )

        gate_up = self.gate_up(sorted_x, expert_counts)
        hidden = weighted_swiglu_active(
            gate_up, sorted_route_weights, expert_counts, self.swiglu_limit
        )
        return self.down(hidden, expert_counts)


def routed_moe_native(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    moe: AscendV41GroupedMoE,
    *,
    expert_start: int,
    num_experts: int,
) -> torch.Tensor:
    """Route with device counts and fixed capacity; preserve V4.1 pre-down weights.

    The native routing map is source-slot to expanded-row, with -1 for remote
    experts. Remote weights scatter to a discarded sink row. Expert counts
    delimit the valid prefix throughout both GMMs; finalize never reads the
    uninitialized padding. No device count is read back to Python.
    """
    if x.shape[0] == 0:
        return torch.empty_like(x)
    ids = topk_ids.to(torch.int32)
    capacity = ids.numel()
    expanded_x, source_to_dest, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        expert_idx=ids,
        active_num=capacity,
        expert_num=num_experts,
        quant_mode=-1,
        active_expert_range=[expert_start, expert_start + moe.gate_up.weight.shape[0]],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )
    from tokenspeed_kernel_npu.ops.dsv41_route_weights import (
        route_weights as prepare_route_weights,
    )

    route_weights = prepare_route_weights(source_to_dest, topk_weights)
    expanded = moe(expanded_x, counts, route_weights)
    # Route scaling already happened before the down projection. Gather only
    # valid BF16 rows and accumulate in FP32 without casting reserved capacity.
    from tokenspeed_kernel_npu.ops.dsv41_moe_combine import combine_routed_rows

    return combine_routed_rows(
        expanded, source_to_dest, rows=x.shape[0], topk=topk_ids.shape[1]
    )
