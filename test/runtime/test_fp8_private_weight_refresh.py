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

"""Keep private block-FP8 weights current after model-loading writes."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from tokenspeed_kernel.ops.gemm import _PreparedFp8Linear
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8 import (
    GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    pack_gluon_fp8_blockscale_weight,
)

from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.quantization.base_config import (
    finalize_quantized_weights_after_loading,
    invalidate_quantized_weights_before_loading,
)


def test_runtime_refresh_keeps_one_packed_buffer() -> None:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((128, 256), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.quant_method = Fp8LinearMethod(SimpleNamespace(weight_block_size=(128, 128)))
    packed = pack_gluon_fp8_blockscale_weight(layer.weight)
    plan = _PreparedFp8Linear(
        override="gluon_mm_fp8_blockscale_largem_gfx950",
        block_size=(128, 128),
        prepared_weight=packed,
        prepared_weight_source=layer.weight,
        prepared_weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
        eligible_rows=frozenset({8144, 8192}),
    )
    layer._prepared_fp8_linear = plan
    packed_pointer = packed.data_ptr()

    invalidate_quantized_weights_before_loading(layer)
    assert not plan.prepared_weight_is_current(layer.weight)
    with torch.no_grad():
        layer.weight.copy_(torch.ones_like(layer.weight))
    finalize_quantized_weights_after_loading(layer)

    assert plan.prepared_weight_is_current(layer.weight)
    assert plan.prepared_weight.data_ptr() == packed_pointer
    assert plan.state_dict() == {}
    assert torch.count_nonzero(plan.prepared_weight.float()) == layer.weight.numel()
