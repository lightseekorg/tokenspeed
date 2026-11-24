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

"""Post-receive FC1/activation/Q1 fusion followed by unweighted grouped FC2."""

import torch
from flashinfer import ActivationType
from flashinfer.fused_moe import cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4
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
    prepare_compact,
)
from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.web_w4a8.layout import (
    unpack_recv_rows,
)


class FlashInferW4A8MaskedMoE:
    """Prepare GPU-counted receive rows and return unweighted expert output.

    Static configuration includes receive/compact capacities, two tile tuples,
    row splits and SwiGLU limit. Weight W13 is loader-order [gate; up]. FP4
    bytes are retained; only row order and UE8M0 scale layout change at startup.
    This isolated adapter uses FlashInfer 0.7.2's lower-level FC2 launcher to
    provide persistent output storage without token finalize or a zero-fill.
    Calls share scratch and must run on one stream; warm before graph capture.
    """

    def __init__(
        self,
        w13,
        sf13,
        w2,
        sf2,
        *,
        capacity,
        compact_capacity,
        swiglu_limit,
        tile_fc1,
        tile_fc2,
        row_splits
    ):
        if not w13.is_cuda or torch.cuda.get_device_capability(w13.device) != (12, 0):
            raise RuntimeError("This validated adapter requires SM120")
        if any(
            t.dtype != torch.uint8
            or t.ndim != 3
            or not t.is_contiguous()
            or t.device != w13.device
            for t in (w13, sf13, w2, sf2)
        ):
            raise ValueError("Weights/scales require contiguous CUDA uint8 tensors")
        e, two_i, half_h = w13.shape
        h, i = half_h * 2, two_i // 2
        if two_i % 2 or h % 128 or i % 128 or w2.shape != (e, h, i // 2):
            raise ValueError("Incompatible W13/W2 or H/I alignment")
        if sf13.shape != (e, two_i, h // 32) or sf2.shape != (e, h, i // 32):
            raise ValueError("Require group32 UE8M0 weight scales")
        if (
            capacity <= 0
            or compact_capacity <= 0
            or compact_capacity > e * capacity
            or row_splits <= 0
        ):
            raise ValueError("Invalid receive/compact capacity or row splits")
        self.device = w13.device
        self.experts = e
        self.capacity = capacity
        self.compact_capacity = compact_capacity
        self.hidden = h
        self.intermediate = i
        self.limit = swiglu_limit
        self.tile_fc1 = tuple(tile_fc1)
        self.epi_fc1 = epi_tactics(self.tile_fc1)[0]
        self.tile_fc2 = tuple(tile_fc2)
        self.epi_fc2 = gemm_epi_tactics(self.tile_fc2)[0]
        self.row_splits = min(row_splits, capacity)
        self.w13 = torch.cat((w13[:, i:], w13[:, :i]), 1).contiguous()
        reordered_sf = torch.cat((sf13[:, i:], sf13[:, :i]), 1).contiguous()
        self.sf13 = (
            pack_weight_scales(reordered_sf)
            .transpose(1, 2)
            .contiguous()
            .view(torch.uint8)
        )
        self.w2 = w2
        self.sf2 = (
            pack_weight_scales(sf2).transpose(1, 2).contiguous().view(torch.uint8)
        )
        self.offsets = torch.empty(e + 1, device=self.device, dtype=torch.int32)
        self.a = torch.empty(
            (compact_capacity, h), device=self.device, dtype=torch.float8_e4m3fn
        )
        self.sa = torch.zeros(
            out_sf_shape(compact_capacity, h, e), device=self.device, dtype=torch.int32
        )
        self.down = torch.empty(
            (compact_capacity, i), device=self.device, dtype=torch.float8_e4m3fn
        )
        self.sd = torch.zeros(
            out_sf_shape(compact_capacity, i, e), device=self.device, dtype=torch.int32
        )
        self.compact_out = torch.empty(
            (compact_capacity, h), device=self.device, dtype=torch.bfloat16
        )
        self.output = torch.empty(
            (e, capacity, h), device=self.device, dtype=torch.bfloat16
        )
        self.fc2 = _Mxfp8Mxfp4Runner(torch.bfloat16, enable_pdl=False)

    def prepare(self, recv_bf16, counts, *, out):
        out = self.output if out is None else out
        if (
            recv_bf16.shape != (self.experts, self.capacity, self.hidden)
            or recv_bf16.dtype != torch.bfloat16
            or recv_bf16.device != self.device
            or not recv_bf16.is_contiguous()
        ):
            raise ValueError("Require contiguous BF16 receive [E,capacity,H]")
        if (
            counts.shape != (self.experts,)
            or counts.dtype != torch.int32
            or counts.device != self.device
            or not counts.is_contiguous()
        ):
            raise ValueError("Require contiguous CUDA int32 expert counts")
        if (
            out.shape != self.output.shape
            or out.dtype != torch.bfloat16
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError("Require contiguous BF16 output [E,capacity,H]")
        # The prefix kernel checks receive and compact bounds before packing.
        # Fuse this check instead of launching a series of tiny CUDA kernels.
        prepare_compact(
            recv_bf16, counts, self.offsets, self.a, self.sa, row_splits=self.row_splits
        )
        return out

    def compute(self, offsets, tile_fc1, epi_fc1, tile_fc2, epi_fc2):
        cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
            self.a,
            self.sa,
            self.w13,
            self.sf13,
            offsets,
            tile=tile_fc1,
            epi=epi_fc1,
            tune=False,
            enable_pdl=False,
            activation=ActivationType.Swiglu,
            swiglu_limit=self.limit,
            out_q=self.down,
            out_sf=self.sd,
        )
        self.fc2.launch(
            (self.down, self.sd, self.w2, self.sf2, offsets),
            self.compact_out,
            (tile_fc2[0], tile_fc2[1], epi_fc2),
        )

    def forward(self, recv_bf16, counts, *, out=None):
        """Write valid rows to contiguous BF16 output; combine applies weights."""
        out = self.prepare(recv_bf16, counts, out=out)
        self.compute(
            self.offsets, self.tile_fc1, self.epi_fc1, self.tile_fc2, self.epi_fc2
        )
        return unpack_recv_rows(self.compact_out, counts, self.offsets, out, 16)


class AdaptiveFlashInferW4A8MoE(FlashInferW4A8MaskedMoE):
    """GPU-select M8/M32 from route count, sharing quantization and weights.

    Both configurations are prewarmed and launched. Zeroed prefix metadata
    makes the inactive compute kernels exit without touching outputs. This
    keeps CUDA Graph replay and never downloads counts or recompiles on them.
    The extra empty launches are included in all performance measurements.
    """

    def __init__(
        self,
        w13,
        sf13,
        w2,
        sf2,
        *,
        capacity,
        compact_capacity,
        swiglu_limit,
        row_splits,
        threshold
    ):
        super().__init__(
            w13,
            sf13,
            w2,
            sf2,
            capacity=capacity,
            compact_capacity=compact_capacity,
            swiglu_limit=swiglu_limit,
            tile_fc1=(8, 128, 128),
            tile_fc2=(8, 128, 128),
            row_splits=row_splits,
        )
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self.low_offsets = torch.empty_like(self.offsets)
        self.high_offsets = torch.empty_like(self.offsets)
        self.high_tile = (32, 128, 128)
        self.high_epi_fc1 = epi_tactics(self.high_tile)[0]
        self.high_epi_fc2 = gemm_epi_tactics(self.high_tile)[0]

    def forward(self, recv_bf16, counts, *, out=None):
        out = self.prepare(recv_bf16, counts, out=out)
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
