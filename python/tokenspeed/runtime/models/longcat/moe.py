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

"""Expert weights and quantization scales for the LongCat layer program."""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.dense.nvfp4 import interleave_linear_and_gate

_HIDDEN = 8192
_INTERMEDIATE = 256


def _swizzle_blockscale(scales: torch.Tensor) -> torch.Tensor:
    """Batched ``swizzle_blockscale_2d`` for ``[E, M, K]`` with M % 128 == 0, K % 4 == 0."""
    e, m, k = scales.shape
    if m % 128 or k % 4:
        raise ValueError(
            f"block scales {tuple(scales.shape)} need M % 128 == 0, K % 4 == 0"
        )
    return (
        scales.reshape(e, m // 128, 4, 32, k // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(e, m, k)
    )


class LongcatExpertWeights:
    """Freeze expert operands before the ordinary MoE backend repacks the checkpoint."""

    def __init__(self, moe, mapping) -> None:
        experts = moe.experts
        self.tp_ep_size = int(mapping.moe.tp_ep_size)
        if moe.zero_expert_num <= 0 or moe.zero_expert_type not in ("identity", "copy"):
            raise ValueError("MoE engine expects identity zero experts")
        if int(mapping.moe.ep_size) != 1:
            raise ValueError("MoE engine expects TP-sharded experts (EP size 1)")
        for name, shape, dtype in (
            ("w13_weight", (None, 2 * _INTERMEDIATE, _HIDDEN // 2), torch.uint8),
            (
                "w13_weight_scale",
                (None, 2 * _INTERMEDIATE, _HIDDEN // 16),
                torch.float8_e4m3fn,
            ),
            ("w2_weight", (None, _HIDDEN, _INTERMEDIATE // 2), torch.uint8),
            (
                "w2_weight_scale",
                (None, _HIDDEN, _INTERMEDIATE // 16),
                torch.float8_e4m3fn,
            ),
        ):
            t = getattr(experts, name, None)
            if t is None:
                raise ValueError(
                    f"MoE engine needs the raw {name}; build it before the MoE "
                    "backend processes its weights"
                )
            if tuple(t.shape[1:]) != shape[1:] or t.dtype != dtype:
                raise ValueError(
                    f"MoE engine expects {name} [E, {shape[1]}, {shape[2]}] {dtype}, "
                    f"got {tuple(t.shape)} {t.dtype}"
                )
        self.gate_clusters = 16
        self.down_clusters = 8
        self.down_group_splits = 2
        self.device = experts.w13_weight.device
        self.weights = None
        self.w13_input_scale_quant = self.output_global_scale = None
        self.gate_alpha = self.up_alpha = self.down_alpha = None
        self._build(experts)

    @torch.no_grad()
    def _build(self, experts) -> None:
        """Kernel layout (as the dense NVFP4 MLP's ``*_swiglu_interleaved``):
        gate/up rows re-ordered ``[up, gate]`` then 64-row interleaved, block
        scales swizzled; alphas from the max input scale like the flashinfer
        backend."""
        gate, up = experts.w13_weight.data.chunk(2, dim=1)
        gate_weight = interleave_linear_and_gate(
            torch.cat((up, gate), dim=1), 64, dim=1
        )
        gate_s, up_s = experts.w13_weight_scale.data.chunk(2, dim=1)
        gate_scale = _swizzle_blockscale(
            interleave_linear_and_gate(torch.cat((up_s, gate_s), dim=1), 64, dim=1)
        )
        # The ordinary MoE backend may mutate/repack its checkpoint tensors.
        down_weight = experts.w2_weight.data.clone(
            memory_format=torch.contiguous_format
        )
        down_scale = _swizzle_blockscale(experts.w2_weight_scale.data)
        self.weights = {
            "gate_weight": gate_weight,
            "gate_scale": gate_scale,
            "down_weight": down_weight,
            "down_scale": down_scale,
        }

        ws2 = experts.w13_weight_scale_2.data.float()
        gate_ws2, up_ws2 = (ws2[:, 0], ws2[:, 1]) if ws2.dim() == 2 else (ws2, ws2)
        w13_input_scale = experts.w13_input_scale.data.max().float()
        w2_input_scale = experts.w2_input_scale.data.max().float()
        self.w13_input_scale_quant = (1.0 / w13_input_scale).reshape(1).contiguous()
        self.gate_alpha = (w13_input_scale * gate_ws2).contiguous()
        self.up_alpha = (w13_input_scale * up_ws2).contiguous()
        self.output_global_scale = (1.0 / w2_input_scale).reshape(1).contiguous()
        self.down_alpha = (
            w2_input_scale * experts.w2_weight_scale_2.data.float().reshape(-1)
        ).contiguous()

    @property
    def input_scale_inv(self) -> torch.Tensor:
        """Reciprocal input scale of the experts; attention 0's P7 packs its
        second NVFP4 output with it."""
        return self.w13_input_scale_quant
