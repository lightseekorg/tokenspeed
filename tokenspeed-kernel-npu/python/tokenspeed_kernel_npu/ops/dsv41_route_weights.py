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

"""Small native routing-weight maps with contiguous Ascend output writes.

The decode path uses a bounded quadratic comparison (at most 512 routes) to
avoid random vector stores. Larger prefill shapes retain the native scatter.
Each valid destination is unique; repeated expert IDs still have separate
expanded rows. The only reduction is over integer bit patterns with at most
one nonzero term per destination. Floating-point accumulation is unchanged.
"""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _route_weights(Map, Weights, Out, N, MS, WS):
    out = tl.program_id(0) * 16 + tl.arange(0, 16)
    result = tl.full((16,), 0, tl.int32)
    for start in range(0, N, 64):
        src = start + tl.arange(0, 64)
        dst = tl.load(Map + src * MS, src < N, other=-1).to(tl.int32)
        value = tl.load(Weights + src * WS, src < N, other=0).to(tl.float32)

        # Preserve zero-initialized CANN scatter_add's +0 and canonical NaN.
        value = tl.where(value == 0, 0.0, value)
        canonical_nan = tl.full((), 0x7FFFFFFF, tl.int32).to(tl.float32, bitcast=True)
        value = tl.where(value != value, canonical_nan, value)
        bits = value.to(tl.int32, bitcast=True)

        # npu_moe_init_routing_v2 gives each valid source slot a unique
        # destination, including repeated expert IDs. Therefore at most one
        # nonzero integer bit pattern contributes across all source chunks.
        matches = (dst[None, :] == out[:, None]) & (src[None, :] < N)
        result += tl.sum(tl.where(matches, bits[None, :], 0), 1)

    # Each program owns one aligned 64-byte block; avoid random vector stores
    # and writes from different programs sharing an Ascend sub-block.
    tl.store(Out + out, result.to(tl.float32, bitcast=True), out < N)


def route_weights(source_to_dest, weights):
    """Prepare native routing weights; this is not a general scatter primitive.

    The caller must supply npu_moe_init_routing_v2's source-slot-to-destination
    map: valid destinations are unique and remote expert slots are negative.
    This preserves the existing routed_moe_native runtime boundary.
    """
    weights = weights.flatten()
    n = source_to_dest.numel()
    if source_to_dest.ndim != 1 or weights.numel() != n:
        raise ValueError("Routing map and flattened weights must have equal lengths")

    from tokenspeed_kernel_npu.ops.dsv41_route_weights_native import (
        route_weights_native,
    )

    native = route_weights_native(source_to_dest, weights)
    if native is not None:
        return native

    # Static decode tuning boundary; retain the old implementation for larger
    # prefill shapes instead of increasing the quadratic comparison workload.
    if n > 512 or source_to_dest.device.type != "npu":
        destinations = torch.where(source_to_dest >= 0, source_to_dest, n).long()
        output = torch.zeros(n + 1, device=weights.device, dtype=torch.float32)
        output.scatter_add_(0, destinations, weights.float())
        return output[:n]

    output = torch.empty(n, device=weights.device, dtype=torch.float32)
    if n:
        _route_weights[(triton.cdiv(n, 16),)](
            source_to_dest,
            weights,
            output,
            n,
            source_to_dest.stride(0),
            weights.stride(0),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return output
