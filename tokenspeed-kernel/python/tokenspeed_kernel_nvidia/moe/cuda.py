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

"""Registration of CUDA-based MoE kernels.

The actual implementations live in tokenspeed_kernel_nvidia.thirdparty.cuda.
This module imports and registers them so they are discoverable via
select_kernel("moe", ...).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_nvidia.registration import Priority, error_fn, register_kernel

try:
    from tokenspeed_kernel_nvidia.thirdparty.cuda.activation import (
        silu_and_mul_fuse_block_quant,
    )
except ImportError:
    silu_and_mul_fuse_block_quant = error_fn

try:
    from tokenspeed_kernel_nvidia.thirdparty.cuda.moe import moe_finalize_fuse_shared
except ImportError:
    moe_finalize_fuse_shared = error_fn

# --- CUDA routing_flash (fused softmax + bias + top-k) ---
try:
    from tokenspeed_kernel_nvidia.routing.cuda import routing_flash

    if routing_flash is not error_fn:
        routing_flash = register_kernel(
            "moe",
            "route",
            name="cuda_routing_flash",
            solution="cuda",
            signatures=format_signatures(
                "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
            ),
            traits={
                "output_type": frozenset({"topk"}),
                "biased": frozenset({True}),
                "grouped": frozenset({False}),
                "ep": frozenset({False}),
            },
            priority=Priority.SPECIALIZED + 3,
            tags={"latency"},
        )(routing_flash)
except ImportError:
    pass
