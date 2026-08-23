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

"""Registration shim for GFX950 DeepSeek V4 routing."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe import (
        gluon_dsv4_select_experts_gfx950 as _select_experts_impl,
    )

    @register_kernel(
        "moe",
        "dsv4_select_experts",
        name="gluon_dsv4_select_experts_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            format_signature(router_logits=dense_tensor_format(dtype))
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ),
        traits={
            "tokens": frozenset({1, 2, 3, 4, 5, 6}),
            "experts": frozenset({256, 384}),
            "top_k": frozenset({6}),
            "renormalize": frozenset({False, True}),
            "routing_kind": frozenset({"plain", "bias", "hash"}),
        },
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "routing", "latency"},
    )
    def gluon_dsv4_select_experts_gfx950(*args, **kwargs):
        return _select_experts_impl(*args, **kwargs)
