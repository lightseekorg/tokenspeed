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

"""Linear method of :class:`Fp8WeightOnlyConfig`."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm import mv_fp8_weight
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.utils import set_weight_attrs

# Largest finite E4M3 (float8_e4m3fn) magnitude.
_E4M3_MAX = 448.0
_QUANTIZE_CHUNK_ROWS = 4096


def quantize_fp8_weight_rows(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` weight rows to E4M3 with one FP32 scale each.

    Args:
        weight: 16-bit ``[N, K]`` weight.

    Returns:
        ``(q, scale)``: the contiguous ``float8_e4m3fn`` weight and the FP32
        ``[N]`` scales, ``weight ~= q * scale[:, None]``. An all-zero row
        keeps scale 1.
    """
    q = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    # Row chunks bound the FP32 temporaries (an LM-head shard is 25k rows).
    for start in range(0, weight.shape[0], _QUANTIZE_CHUNK_ROWS):
        rows = weight[start : start + _QUANTIZE_CHUNK_ROWS].float()
        amax = rows.abs().amax(dim=1)
        part = torch.where(amax > 0, amax / _E4M3_MAX, torch.ones_like(amax))
        q[start : start + rows.shape[0]] = (rows / part[:, None]).to(q.dtype)
        scale[start : start + rows.shape[0]] = part
    return q, scale


class Fp8WeightOnlyLinearMethod(LinearMethodBase):
    """Loads a 16-bit weight, then keeps it as E4M3 rows with FP32 scales."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # The checkpoint's 16-bit weight; process_weights_after_loading
        # replaces it once every shard (gate and up) is loaded.
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        q, scale = quantize_fp8_weight_rows(layer.weight.data)
        layer.weight = Parameter(q, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = mv_fp8_weight(
            x.reshape(-1, x.shape[-1]), layer.weight, layer.weight_scale
        ).view(*x.shape[:-1], layer.weight.shape[0])
        if bias is not None:
            out = out + bias
        return out
