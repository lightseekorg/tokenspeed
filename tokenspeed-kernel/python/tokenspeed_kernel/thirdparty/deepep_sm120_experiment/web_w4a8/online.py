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

"""Prepared weight views and shared decode scratch for SM120 W4A8."""

import torch
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 import (
    epi_tactics,
    out_sf_shape,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_gemm import (
    _Mxfp8Mxfp4Runner,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_gemm import (
    epi_tactics as gemm_epi_tactics,
)
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.web_w4a8 import (
    pack_weight_scales,
)
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.web_w4a8._compact import (
    branch_offsets,
    prepare_wire,
)
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.web_w4a8.flashinfer import (
    AdaptiveFlashInferW4A8MoE,
)
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.web_w4a8.layout import (
    unpack_recv_rows,
)


class SharedWorkspace:
    """One stream-exclusive scratch allocation shared across model layers."""

    def __init__(
        self, *, experts, capacity, compact_capacity, hidden, intermediate, device
    ):
        self.device = device
        self.experts = experts
        self.capacity = capacity
        self.compact_capacity = compact_capacity
        self.hidden = hidden
        self.intermediate = intermediate
        self.offsets = torch.empty(experts + 1, device=device, dtype=torch.int32)
        self.low_offsets = torch.empty_like(self.offsets)
        self.high_offsets = torch.empty_like(self.offsets)
        self.a = torch.empty(
            (compact_capacity, hidden), device=device, dtype=torch.float8_e4m3fn
        )
        self.sa = torch.zeros(
            out_sf_shape(compact_capacity, hidden, experts),
            device=device,
            dtype=torch.int32,
        )
        self.down = torch.empty(
            (compact_capacity, intermediate), device=device, dtype=torch.float8_e4m3fn
        )
        self.sd = torch.zeros(
            out_sf_shape(compact_capacity, intermediate, experts),
            device=device,
            dtype=torch.int32,
        )
        self.compact_out = torch.empty(
            (compact_capacity, hidden), device=device, dtype=torch.bfloat16
        )
        self.output = torch.empty(
            (experts, capacity, hidden), device=device, dtype=torch.bfloat16
        )
        self.empty = torch.empty(0, device=device, dtype=torch.float32)


class PreparedAdaptiveExperts(AdaptiveFlashInferW4A8MoE):
    """Reuse prepared weights and shared scratch; accept all supported wires."""

    def __init__(self, w13, sf13, w2, sf2, workspace, *, limit, threshold):
        self.device = workspace.device
        self.experts = workspace.experts
        self.capacity = workspace.capacity
        self.compact_capacity = workspace.compact_capacity
        self.hidden = workspace.hidden
        self.intermediate = workspace.intermediate
        self.limit = limit
        self.threshold = threshold
        self.tile_fc1 = (8, 128, 128)
        self.tile_fc2 = (8, 128, 128)
        self.epi_fc1 = epi_tactics(self.tile_fc1)[0]
        self.epi_fc2 = gemm_epi_tactics(self.tile_fc2)[0]
        self.high_tile = (32, 128, 128)
        self.high_epi_fc1 = epi_tactics(self.high_tile)[0]
        self.high_epi_fc2 = gemm_epi_tactics(self.high_tile)[0]
        self.row_splits = min(16, self.capacity)
        self.w13, self.sf13, self.w2, self.sf2 = w13, sf13, w2, sf2
        self.offsets = workspace.offsets
        self.low_offsets = workspace.low_offsets
        self.high_offsets = workspace.high_offsets
        self.a, self.sa = workspace.a, workspace.sa
        self.down, self.sd = workspace.down, workspace.sd
        self.compact_out, self.output = workspace.compact_out, workspace.output
        self.empty = workspace.empty
        self.fc2 = _Mxfp8Mxfp4Runner(torch.bfloat16, enable_pdl=False)

    def forward_wire(self, recv, scales, global_scales, counts, *, wire_dtype, out):
        width = self.hidden // 2 if wire_dtype == "nvfp4" else self.hidden
        dtype = {
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
            "nvfp4": torch.uint8,
        }.get(wire_dtype)
        if (
            dtype is None
            or recv.shape != (self.experts, self.capacity, width)
            or recv.dtype != dtype
            or recv.stride(-1) != 1
            or recv.device != self.device
        ):
            raise ValueError("Receive data does not match the configured DeepEP wire")
        if (
            counts.shape != (self.experts,)
            or counts.dtype != torch.int32
            or counts.device != self.device
        ):
            raise ValueError("Require CUDA int32 expert counts")
        if (
            out.shape != self.output.shape
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError("Require contiguous BF16 combine staging output")
        if wire_dtype != "bf16":
            group = 16 if wire_dtype == "nvfp4" else 128
            sdtype = torch.float8_e4m3fn if wire_dtype == "nvfp4" else torch.float32
            if (
                scales is None
                or scales.shape != (self.experts, self.capacity, self.hidden // group)
                or scales.dtype != sdtype
                or scales.device != self.device
            ):
                raise ValueError("Receive scales do not match wire granularity/dtype")
        if wire_dtype == "nvfp4" and (
            global_scales is None
            or global_scales.shape != (self.experts, self.capacity)
            or global_scales.dtype != torch.float32
            or global_scales.device != self.device
        ):
            raise ValueError("NVFP4 requires FP32 per-token reconstruction scales")
        prepare_wire(
            recv,
            self.empty if scales is None else scales,
            self.empty if global_scales is None else global_scales,
            counts,
            self.offsets,
            self.a,
            self.sa,
            row_splits=self.row_splits,
            wire_dtype=wire_dtype,
            capacity=self.capacity,
        )
        branch_offsets(
            self.offsets, self.low_offsets, self.high_offsets, threshold=self.threshold
        )
        self.compute(
            self.low_offsets, self.tile_fc1, self.epi_fc1, self.tile_fc2, self.epi_fc2
        )
        self.compute(
            self.high_offsets,
            self.high_tile,
            self.high_epi_fc1,
            self.high_tile,
            self.high_epi_fc2,
        )
        return unpack_recv_rows(self.compact_out, counts, self.offsets, out, 16)


def prepare_weight_views(w13, sf13, w2, sf2):
    """Keep FP4 bytes; reorder W13 to [up; gate] and prepare scale ABI."""
    half = w13.shape[1] // 2
    reordered = torch.cat((w13[:, half:], w13[:, :half]), 1).contiguous()
    scales = torch.cat((sf13[:, half:], sf13[:, :half]), 1).contiguous()
    return (
        reordered,
        pack_weight_scales(scales).transpose(1, 2).contiguous().view(torch.uint8),
        w2,
        pack_weight_scales(sf2).transpose(1, 2).contiguous().view(torch.uint8),
    )
